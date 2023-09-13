import torch
import torchaudio
from ..base import FXBase
from ..chain import FXChain
from .signal import fft_filt, frequencysampling_filt, biquad_filt, biquad_filt_2

filter_function = fft_filt

class LowShelf(FXBase):
    def __init__(
        self,
        samplerate:float
    ):
        super(FXBase, self).__init__()
        self.samplerate = samplerate
        self.controls_names = ['f0', 'gain_dB', 'Q']
        self.controls_ranges = torch.Tensor([
            [20, samplerate/2], [-12, 12], [0.1, 3]
        ])
        self.num_controls = 3

    
    def set_controls_to_range(self, q):
        ranges = self.controls_ranges.clone().to(q.device)
        ranges[0] = torch.log(ranges[0])
        m = ranges[:,0].reshape(1, -1)
        M = ranges[:,1].reshape(1, -1)
        p = q*(M-m) + m
        p[:,0] = torch.exp(p[:,0])
        return p
    
    def process(self, x, p):
        batch_size, num_controls = p.size()

        f0 = p[:,0]
        gain_dB = p[:,1]
        Q = p[:,2]

        A = torch.pow(10, gain_dB/40)
        w0 = 2*torch.pi*f0/self.samplerate

        alpha = torch.sin(w0)/(2*Q)

        a = torch.zeros((batch_size, 3), device = x.device)
        b = torch.zeros((batch_size, 3), device = x.device)

        b[:,0] =    A*( (A+1) - (A-1)*torch.cos(w0) + 2*torch.sqrt(A)*alpha )
        b[:,1] =  2*A*( (A-1) - (A+1)*torch.cos(w0)                   )
        b[:,2] =    A*( (A+1) - (A-1)*torch.cos(w0) - 2*torch.sqrt(A)*alpha )
        a[:,0] =        (A+1) + (A-1)*torch.cos(w0) + 2*torch.sqrt(A)*alpha
        a[:,1] =   -2*( (A-1) + (A+1)*torch.cos(w0)                   )
        a[:,2] =        (A+1) + (A-1)*torch.cos(w0) - 2*torch.sqrt(A)*alpha

        out = filter_function(
            x=x,
            a_coeffs=a,
            b_coeffs=b
        )
        return out

class HighShelf(FXBase):
    def __init__(
        self,
        samplerate:float
    ):
        super(FXBase, self).__init__()
        self.samplerate = samplerate
        self.controls_names = ['f0', 'gain_dB', 'Q']
        self.controls_ranges = torch.Tensor([
            [20, samplerate/2], [-12, 12], [0.1, 3]
        ])
        self.num_controls = 3
    
    def set_controls_to_range(self, q):
        ranges = self.controls_ranges.clone().to(q.device)
        ranges[0] = torch.log(ranges[0])

        m = ranges[:,0].reshape(1, -1)
        M = ranges[:,1].reshape(1, -1)
        p = q*(M-m) + m
        p[:,0] = torch.exp(p[:,0])
        return p
    
    def process(self, x, p):
        batch_size, num_controls = p.size()

        f0 = p[:,0]
        gain_dB = p[:,1]
        Q = p[:,2]

        A = torch.pow(10, gain_dB/40)
        w0 = 2*torch.pi*f0/self.samplerate

        alpha = torch.sin(w0)/(2*Q)

        a = torch.zeros((batch_size, 3), device = x.device)
        b = torch.zeros((batch_size, 3), device = x.device)

        b[:,0] =    A*( (A+1) + (A-1)*torch.cos(w0) + 2*torch.sqrt(A)*alpha )
        b[:,1] = -2*A*( (A-1) + (A+1)*torch.cos(w0)                   )
        b[:,2] =    A*( (A+1) + (A-1)*torch.cos(w0) - 2*torch.sqrt(A)*alpha )
        a[:,0] =        (A+1) - (A-1)*torch.cos(w0) + 2*torch.sqrt(A)*alpha
        a[:,1] =    2*( (A-1) - (A+1)*torch.cos(w0)                   )
        a[:,2] =        (A+1) - (A-1)*torch.cos(w0) - 2*torch.sqrt(A)*alpha

        out = filter_function(
            x=x,
            a_coeffs=a,
            b_coeffs=b
        )
        return out

class Peak(FXBase):
    def __init__(
        self,
        samplerate:float
    ):
        super(FXBase, self).__init__()
        self.samplerate = samplerate
        self.controls_names = ['f0', 'gain_dB', 'Q']
        self.controls_ranges = torch.Tensor([
            [20, samplerate/2], [-12, 12], [0.1, 3]
        ])
        self.num_controls = 3
    
    def set_controls_to_range(self, q):
        ranges = self.controls_ranges.clone().to(q.device)
        ranges[0] = torch.log(ranges[0])

        m = ranges[:,0].reshape(1, -1)
        M = ranges[:,1].reshape(1, -1)
        p = q*(M-m) + m
        p[:,0] = torch.exp(p[:,0])
        return p
    
    def process(self, x, p):
        batch_size, num_controls = p.size()

        f0 = p[:,0]
        gain_dB = p[:,1]
        Q = p[:,2]

        A = torch.pow(10, gain_dB/40)
        w0 = 2*torch.pi*f0/self.samplerate

        alpha = torch.sin(w0)/(2*Q)

        a = torch.zeros((batch_size, 3), device = x.device)
        b = torch.zeros((batch_size, 3), device = x.device)

        b[:,0] = 1+alpha*A
        b[:,1] = -2*torch.cos(w0)
        b[:,2] = 1-alpha*A
        a[:,0] = 1+alpha/A
        a[:,1] = -2*torch.cos(w0)
        a[:,2] = 1 - alpha/A

        out = filter_function(
            x=x,
            a_coeffs=a,
            b_coeffs=b
        )
        return out


class FixedPeak(FXBase):
    def __init__(
        self,
        f0:float,
        Q:float = 0.667,
        samplerate:float = 44100
    ):
        super(FXBase, self).__init__()
        self.f0 = f0
        self.Q = Q
        self.samplerate = samplerate
        self.controls_names = ['gain_dB']
        self.controls_ranges = torch.Tensor([[-15, 15]])
        self.num_controls = 1
    
    def process(self, x, p):
        batch_size, num_controls = p.size()
        
        f0 = torch.ones((batch_size), device = p.device)*self.f0
        Q = torch.ones((batch_size), device = p.device)*self.Q

        gain_dB = p[:,0]

        A = torch.pow(10, gain_dB/40)
        w0 = 2*torch.pi*f0/self.samplerate

        alpha = torch.sin(w0)/(2*Q)

        a = torch.zeros((batch_size, 3), device = x.device)
        b = torch.zeros((batch_size, 3), device = x.device)

        b[:,0] = 1+alpha*A
        b[:,1] = -2*torch.cos(w0)
        b[:,2] = 1-alpha*A
        a[:,0] = 1+alpha/A
        a[:,1] = -2*torch.cos(w0)
        a[:,2] = 1 - alpha/A

        out = filter_function(
            x=x,
            a_coeffs=a,
            b_coeffs=b
        )
        return out


class FixedLowShelf(FXBase):
    def __init__(
        self,
        f0:float,
        Q:float = 0.667,
        samplerate:float = 44100
    ):
        super(FXBase, self).__init__()
        self.f0 = f0
        self.Q = Q
        self.samplerate = samplerate
        self.controls_names = ['gain_dB']
        self.controls_ranges = torch.Tensor([[-15, 15]])
        self.num_controls = 1
    
    def process(self, x, p):
        batch_size, num_controls = p.size()
        
        f0 = torch.ones((batch_size), device = p.device)*self.f0
        Q = torch.ones((batch_size), device = p.device)*self.Q

        gain_dB = p[:,0]

        A = torch.pow(10, gain_dB/40)
        w0 = 2*torch.pi*f0/self.samplerate

        alpha = torch.sin(w0)/(2*Q)

        a = torch.zeros((batch_size, 3), device = x.device)
        b = torch.zeros((batch_size, 3), device = x.device)

        b[:,0] =    A*( (A+1) - (A-1)*torch.cos(w0) + 2*torch.sqrt(A)*alpha )
        b[:,1] =  2*A*( (A-1) - (A+1)*torch.cos(w0)                   )
        b[:,2] =    A*( (A+1) - (A-1)*torch.cos(w0) - 2*torch.sqrt(A)*alpha )
        a[:,0] =        (A+1) + (A-1)*torch.cos(w0) + 2*torch.sqrt(A)*alpha
        a[:,1] =   -2*( (A-1) + (A+1)*torch.cos(w0)                   )
        a[:,2] =        (A+1) + (A-1)*torch.cos(w0) - 2*torch.sqrt(A)*alpha

        out = filter_function(
            x=x,
            a_coeffs=a,
            b_coeffs=b
        )
        return out


class FixedHighShelf(FXBase):
    def __init__(
        self,
        f0:float,
        Q:float = 0.667,
        samplerate:float = 44100
    ):
        super(FXBase, self).__init__()
        self.f0 = f0
        self.Q = Q
        self.samplerate = samplerate
        self.controls_names = ['gain_dB']
        self.controls_ranges = torch.Tensor([[-15, 15]])
        self.num_controls = 1
    
    def process(self, x, p):
        batch_size, num_controls = p.size()
        
        f0 = torch.ones((batch_size), device = p.device)*self.f0
        Q = torch.ones((batch_size), device = p.device)*self.Q

        gain_dB = p[:,0]

        A = torch.pow(10, gain_dB/40)
        w0 = 2*torch.pi*f0/self.samplerate

        alpha = torch.sin(w0)/(2*Q)

        a = torch.zeros((batch_size, 3), device = x.device)
        b = torch.zeros((batch_size, 3), device = x.device)

        b[:,0] =    A*( (A+1) + (A-1)*torch.cos(w0) + 2*torch.sqrt(A)*alpha )
        b[:,1] = -2*A*( (A-1) + (A+1)*torch.cos(w0)                   )
        b[:,2] =    A*( (A+1) + (A-1)*torch.cos(w0) - 2*torch.sqrt(A)*alpha )
        a[:,0] =        (A+1) - (A-1)*torch.cos(w0) + 2*torch.sqrt(A)*alpha
        a[:,1] =    2*( (A-1) - (A+1)*torch.cos(w0)                   )
        a[:,2] =        (A+1) - (A-1)*torch.cos(w0) - 2*torch.sqrt(A)*alpha
        
        out = filter_function(
            x=x,
            a_coeffs=a,
            b_coeffs=b
        )
        return out