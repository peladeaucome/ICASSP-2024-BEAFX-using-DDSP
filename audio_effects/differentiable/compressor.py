import torch
import torchaudio
from .signal import fft_filt

from ..base import FXBase

filter_function = fft_filt

def dB20(x):
    x_dB = torch.abs(x)
    x_dB = torch.where(
        x_dB > 0.000001,
        input = 20*torch.log10(torch.abs(x_dB)),
        other = torch.ones_like(x_dB)*(-120))
    return x_dB

def idB20(x):
    return torch.pow(10, x*0.05)

class SimpleCompressor(FXBase):
    def __init__(
        self,
        samplerate:int = 44100
        ):
        super(FXBase, self).__init__()

        self.samplerate = samplerate

        self.controls_names = [
            "threshold_dB",
            "time_constant",
            "ratio",
            "knee_dB",
            "makeup_gain_dB"]

        self.controls_ranges = torch.Tensor([
            [-30, 0],
            [0.1, 100],
            [1, 10],
            [0, 12],
            [0, 20]
        ])
        self.num_controls = 5
        self.num_channels = 1
    
    def set_controls_to_range(self, q):
        ranges = self.controls_ranges.clone().to(q.device)
        ranges[1] = torch.log(ranges[1])

        m = ranges[:,0].reshape(1, -1)
        M = ranges[:,1].reshape(1, -1)
        p = q*(M-m) + m
        p[:,1] = torch.exp(p[:,0])
        return p

    def process(
        self,
        x,
        p):

        threshold_dB = p[:,0].reshape(-1, 1, 1)
        time_constant = p[:,1].reshape(-1, 1, 1)
        ratio = p[:,2].reshape(-1, 1, 1)
        knee_dB = p[:,3].reshape(-1, 1, 1)
        makeup_gain_dB = p[:,4].reshape(-1, 1, 1)

        x_G = dB20(x)
        batch_size, num_channels, num_samples = x.size()

        if num_channels != self.num_channels:
            raise ValueError(f"Wrong number of channels, should be {self.num_channels}")

        alpha = torch.exp(-1/(time_constant*0.001*self.samplerate))
        b = torch.zeros((batch_size, 2), device = alpha.device)
        b[:,0] = (1-alpha)[:,0,0]
        a = torch.ones((batch_size, 2), device = alpha.device)
        a[:,1] = -alpha[:,0,0]
        #y_G = torch.zeros_like(x_G)

        y_G = torch.where(
            2*(x_G-threshold_dB) < -knee_dB*torch.ones_like(x),
            input = x_G,
            other = torch.where(
                torch.abs(2*(x_G - threshold_dB)) <= knee_dB*torch.ones_like(x),
                input = x_G + (1/ratio-1)*torch.square(x_G - threshold_dB+knee_dB/2)/(2*knee_dB),
                other = threshold_dB+(x_G-threshold_dB)/ratio
            )
        )

        x_L = x_G - y_G

        #x_L_pad = torch.zeros(batch_size, num_channels, num_samples+4096-1)
        #x_L_pad[:,:,:num_samples] = x_L


        y_L = filter_function(
            x = x_L,
            a_coeffs = a,
            b_coeffs = b
        )
        
        c = torch.pow(10, (makeup_gain_dB - y_L)/20)

        return x*c



