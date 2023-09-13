import numpy as np
from numba import jit
import scipy.signal as sig
from .dsp import NPFXBase
import torch

class Peak(NPFXBase):
    def __init__(
        self,
        samplerate:float
    ):
        super(NPFXBase, self).__init__()

        self.samplerate = samplerate
        self.controls_names = ['f0', 'gain_dB', 'Q']
        self.controls_ranges = torch.Tensor([
            [20, samplerate/2], [-12, 12], [0.1, 3]
        ])
        self.num_controls = 3
    
    def set_controls_to_range(self, p):
        ranges = self.controls_ranges.clone()
        ranges[0] = torch.log(ranges[0])

        m = ranges[:,0].reshape(1, -1)
        M = ranges[:,1].reshape(1, -1)
        p = p*(M-m) + m
        p[:,0] = torch.exp(p[:,0])
        return p
    
    def effect(
        self,
        x,
        f0:float,
        Q:float = .71,
        gain_dB:float = 0,
        samplerate = 44100
    ):
        A = np.power(10,gain_dB/40)
        w0 = 2*np.pi*f0/samplerate
        alpha = np.sin(w0)/(2*Q)
        b = np.array([
            1+alpha*A,
            -2*np.cos(w0),
            1-alpha*A
        ])
        a = np.array([
            1+alpha/A,
            -2*np.cos(w0),
            1-alpha/A
        ])
        out = sig.lfilter(b, a, x)
        return out

class LowShelf(NPFXBase):
    def __init__(
        self,
        samplerate:float
    ):
        super(NPFXBase, self).__init__()

        self.samplerate = samplerate
        self.controls_names = ['f0', 'gain_dB', 'Q']
        self.controls_ranges = torch.Tensor([
            [20, samplerate/2], [-12, 12], [0.1, 3]
        ])
        self.num_controls = 3
    
    def set_controls_to_range(self, p):
        ranges = self.controls_ranges.clone()
        ranges[0] = torch.log(ranges[0])

        m = ranges[:,0].reshape(1, -1)
        M = ranges[:,1].reshape(1, -1)
        p = p*(M-m) + m
        p[:,0] = torch.exp(p[:,0])
        return p
    
    def effect(
        self,
        x,
        f0:float,
        Q:float,
        gain_dB:float,
        samplerate = 44100
    ):
        A = np.power(10,gain_dB/40)
        w0 = 2*np.pi*f0/samplerate
        alpha = np.sin(w0)/(2*Q)
        b = np.array([
            A*((A+1) - (A-1)*np.cos(w0) + 2*np.sqrt(A)*alpha),
            2*A*((A-1) - (A+1)*np.cos(w0)),
            A*((A+1) - (A-1)*np.cos(w0) - 2*np.sqrt(A)*alpha)
        ])
        a = np.array([
            (A+1) + (A-1)*np.cos(w0) + 2*np.sqrt(A)*alpha,
            -2*( (A-1) + (A+1)*np.cos(w0)),
            (A+1) + (A-1)*np.cos(w0) - 2*np.sqrt(A)*alpha
        ])
        out = sig.lfilter(b, a, x)
        return out

class HighShelf(NPFXBase):
    def __init__(
        self,
        samplerate:float
    ):
        super(NPFXBase, self).__init__()
        
        self.samplerate = samplerate
        self.controls_names = ['f0', 'gain_dB', 'Q']
        self.controls_ranges = torch.Tensor([
            [20, samplerate/2], [-12, 12], [0.1, 3]
        ])
        self.num_controls = 3
    
    def set_controls_to_range(self, p):
        ranges = self.controls_ranges.clone()
        ranges[0] = torch.log(ranges[0])

        m = ranges[:,0].reshape(1, -1)
        M = ranges[:,1].reshape(1, -1)
        p = p*(M-m) + m
        p[:,0] = torch.exp(p[:,0])
        return p
    
    def effect(
        self,
        x,
        f0:float,
        Q:float = .71,
        gain_dB:float = 0,
        samplerate = 44100
    ):
        A = np.power(10,gain_dB/40)
        w0 = 2*np.pi*f0/samplerate
        alpha = np.sin(w0)/(2*Q)
        b = np.array([
            A*((A+1) + (A-1)*np.cos(w0) + 2*np.sqrt(A)*alpha),
            -2*A*((A-1) + (A+1)*np.cos(w0)),
            A*((A+1) + (A-1)*np.cos(w0) - 2*np.sqrt(A)*alpha)
        ])
        a = np.array([
            (A+1) - (A-1)*np.cos(w0) + 2*np.sqrt(A)*alpha,
            2*( (A-1) - (A+1)*np.cos(w0)),
            (A+1) - (A-1)*np.cos(w0) - 2*np.sqrt(A)*alpha
        ])
        out = sig.lfilter(b, a, x)
        return out