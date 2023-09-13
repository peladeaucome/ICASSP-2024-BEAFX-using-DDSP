import torch
from ..base import FXBase

class TanhClipper(FXBase):
    def __init__(
            self,
            samplerate = 44100
    ):
        super(FXBase, self).__init__()
        self.controls_ranges = torch.Tensor([[-20, 20], [0,1]])
        self.controls_names = ['drive_dB', 'offset']
        self.num_controls = 2
    
    def process(self, x, p):
        batch_size = p.size(0)
        drive_dB = p[:,0].reshape(batch_size, 1, 1)
        offset = p[:,1].reshape(batch_size, 1, 1)
        drive = torch.pow(10, drive_dB/20)

        out = (torch.tanh(x*drive+offset)-torch.tanh(offset))/drive
        return out


def hc(x):
    return (torch.abs(x+1)-torch.abs(x-1))/2

def cc(x):
    out = x-4/27*torch.pow(x, 3)
    out = torch.where(
        condition = torch.abs(x)<1.5,
        input = out,
        other = torch.sign(x))
    return out

class HardnessDist(FXBase):
    def __init__(
            self,
            samplerate = 44100
    ):
        super(FXBase, self).__init__()
        self.controls_ranges = torch.Tensor([[-20, 20], [0,1], [0, 2]])
        self.controls_names = ['drive_dB', 'offset', 'dist_choice']
        self.num_controls = 3
    
    def tanh_clipper(self, x, drive, offset):
        return (torch.tanh(x*drive+offset)-torch.tanh(offset))/drive
    
    def cubic_clipper(self, x, drive, offset):
        return (cc(x*drive+offset)-cc(offset))/drive
    
    def hard_clipper(self, x, drive, offset):
        return (hc(x*drive+offset)-hc(offset))/drive

    def process(self, x, p):
        batch_size = p.size(0)
        drive_dB = p[:,0].reshape(batch_size, 1, 1)
        offset = p[:,1].reshape(batch_size, 1, 1)
        dist_choice = p[:,2].reshape(batch_size, 1, 1)
        drive = torch.pow(10, drive_dB/20)

        out = torch.zeros_like(x)

        out0 = self.tanh_clipper(x, drive, offset)
        out1 = self.cubic_clipper(x, drive, offset)
        out2 = self.hard_clipper(x, drive, offset)

        out = torch.where(
            condition=dist_choice<1,
            input = (1-dist_choice)*out0 + dist_choice*out1,
            other = (2-dist_choice)*out1 + (dist_choice-1)*out2
            )

        return out


class TaylorHarmonics(FXBase):
    def __init__(
            self,
            num_harmonics:int = 10,
            samplerate:float = 44100
    ):
        super(FXBase, self).__init__()

        self.num_harmonics = num_harmonics #Number of harmonics
        self.num_controls = num_harmonics

        self.controls_ranges = [[-1, 1] for i in range(self.num_controls)]
        self.controls_ranges = torch.Tensor(self.controls_ranges)
        self.controls_names = [f'harmonic {i+1} level' for i in range(self.num_controls)]
        self.effect_name = 'harmonics_generator'
        self.norm_fun = no_norm
    

    def process(self, x:torch.Tensor, p:torch.Tensor):
        # x is of shape (batch_size, num_channel, num_samples)
        # p is of shape (batch_size, num_controls)

        x = remove_DC(x)
        batch_size, num_channels, num_samples = x.size()

        x = x.reshape(batch_size, num_channels, num_samples, 1)
        p = p.reshape(batch_size, 1, 1,self.num_controls)
        h = torch.arange(start = 0, end = self.num_controls, device = x.device).reshape(1, 1, 1, self.num_controls)
        y = torch.pow(x, h+1)*p

        #one = torch.ones((1, 1, 1, self.num_controls), device = x.device)
        #
        #y=y*torch.pow(-one, (h+1)//2)
        y = y.sum(3)

        return y


class ChebyshevHarmonics(FXBase):
    def __init__(
            self,
            num_harmonics:int = 10,
            samplerate:float = 44100
    ):
        super(FXBase, self).__init__()

        self.num_harmonics = num_harmonics #Number of harmonics
        self.num_controls = num_harmonics

        self.controls_ranges = [[-1, 1] for i in range(self.num_controls)]
        self.controls_ranges = torch.Tensor(self.controls_ranges)
        self.controls_names = [f'harmonic {i+1} level' for i in range(self.num_controls)]
        self.effect_name = 'harmonics_generator'
        self.norm_fun = no_norm
    

    def process(self, x:torch.Tensor, p:torch.Tensor):
        # x is of shape (batch_size, num_channel, num_samples)
        # p is of shape (batch_size, num_controls)
        
        x = remove_DC(x)
        batch_size, num_channels, num_samples = x.size()
        x_=x.clone()
        y = torch.zeros((batch_size, 1, num_samples, self.num_controls), device= x.device)
        y[:,:,:,0] = torch.ones_like(x)
        y[:,:,:,1] = x_
        for harmo_idx in range(2, self.num_harmonics):
            y[:,:,:,harmo_idx] = 2*x_*y[:,:,:,harmo_idx-1]-y[:,:,:,harmo_idx-2]

        y = y*p.reshape(batch_size, 1, 1, self.num_harmonics)
        #one = torch.ones((1, 1, 1, self.num_controls), device = x.device)
        #h = torch.arange(start = 0, end = self.num_controls, device = x.device).reshape(1, 1, 1, self.num_controls)
        #y=y*torch.pow(-one, (h+1)//2)

        y = y.sum(3)
        return y


def max_norm(x):
        """Normalizes by the maximum"""
        xmax, _ = torch.max(torch.abs(x), dim=2)
        x=x/(xmax.reshape(-1, 1, 1))
        return x

def rms_norm(x):
    """Normalizes by the standard deviation"""
    x = x/(torch.std(x, dim=2).reshape(-1, 1, 1))
    return x

def no_norm(x):
      """Identity Function"""
      return x

def remove_DC(x):
    batch_size, num_channels, num_samples = x.size() 
    DC = x.mean(-1).reshape(batch_size, num_channels, 1)
    x=x-DC
    return x


