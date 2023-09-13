import torch
import torch.nn.functional as F
import numpy as np

class GaborFrontEnd(torch.nn.Module):
    def __init__(
        self,
        out_channels:int = 16,
        stride:int = 16,
        kernel_size:int = 33
    ):
        super(GaborFrontEnd, self).__init__()

        self.freqs = torch.nn.Parameter((torch.arange(out_channels, dtype=torch.float32)-out_channels/2)/out_channels*4)
        self.BWs = torch.nn.Parameter(torch.zeros(out_channels))

        self.kernel_size = kernel_size
        self.stride = stride
        self.out_channels = out_channels
        self.pi = np.pi
        self.sqrt2pi = np.sqrt(2*self.pi)
        self.sqrt2log2 = np.sqrt(2*np.log(2))

    def forward(self, x):
        freqs = F.sigmoid(self.freqs.reshape(self.out_channels, 1, 1))*.5

        BWs = F.sigmoid(self.BWs.reshape(self.out_channels, 1, 1))
        BWs = (BWs*(2*self.kernel_size-4) + 4)*self.sqrt2log2

        device = freqs.device
        pi = self.pi

        t = torch.arange(self.kernel_size, device = device).reshape(1, 1, self.kernel_size)-self.kernel_size//2


        weights_real = torch.cos(2*pi*freqs*t)*1/(self.sqrt2pi*BWs)*torch.exp(-torch.square(t/BWs)/2)
        weights_imag = torch.sin(2*pi*freqs*t)*1/(self.sqrt2pi*BWs)*torch.exp(-torch.square(t/BWs)/2)

        out_real = F.conv1d(
            input = x,
            weight = weights_real,
            stride=self.stride
        )

        out_imag = F.conv1d(
            input = x,
            weight = weights_imag,
            stride=self.stride
        )

        return torch.square(out_real)+torch.square(out_imag)

    
    #def _apply(self, fn):
    #    self.freqs = fn(self.freqs)
    #    self.BWs = fn(self.BWs)
    #    return self



class Conv1dFrontEnd(torch.nn.Module):
    def __init__(
        self,
        out_channels:int = 16,
        stride:int = 16,
        kernel_size:int = 33
    ):
        super(Conv1dFrontEnd, self).__init__()

        self.conv1d_real = torch.nn.Conv1d(in_channels=1, out_channels=out_channels, stride=stride, kernel_size = kernel_size, bias = False)
        self.conv1d_imag = torch.nn.Conv1d(in_channels=1, out_channels=out_channels, stride=stride, kernel_size = kernel_size, bias = False) 

    def forward(self, x):
        out = torch.square(self.conv1d_real(x))+torch.square(self.conv1d_imag(x))
        return out