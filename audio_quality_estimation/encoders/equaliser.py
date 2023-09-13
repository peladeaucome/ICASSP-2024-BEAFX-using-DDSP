import torch
import lightning.pytorch as pl
from ..conv_layers import *
import torchaudio
import numpy as np
import nnAudio
from ..normalize import max_norm, rms_norm, no_norm



class EQ_Encoder(torch.nn.Module):
    def __init__(
        self,
        kernels:list=[25, 5],
        strides:list=[1,1],
        channels:list=[64, 128],
        dilations:list = [1, 1],
        samplerate:int=44100,
        n_bins:int = 113,
        pad = True,
        conv_class=torch.nn.Conv2d,
        NL_class = torch.nn.PReLU,
        device=None,
        normalization_function = no_norm,
        eps = 1e-4
    ):
        super(EQ_Encoder, self).__init__()

        channels = [1]+channels

        self.kernels = kernels
        self.channels = channels
        self.pad=pad
        self.out_dim = channels[-1]
        self.normalization_function = normalization_function
        self.n_bins = n_bins
        self.eps = eps

        self.name="ENC_equaliser"

        self.stft = nnAudio.features.CQT1992v2(
            n_bins = self.n_bins,
            sr=samplerate
        )

        self.num_convLayers = len(channels)
        
        self.conv1 = torch.nn.Sequential(
            Padded_Conv2d(
                    in_channels = channels[0],
                    out_channels = channels[1],
                    kernel_size = kernels,
                    stride = strides[0],
                    bias = True,
                    device=device,
                    pad_value=eps
            ),
            torch.nn.BatchNorm2d(num_features = channels[1], device=device),
            NL_class()
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv1d(
                    in_channels = channels[1],
                    out_channels = channels[2],
                    kernel_size = n_bins,
                    stride = 1,
                    padding = 0,
                    bias = True,
                    device=device
            ),
            torch.nn.BatchNorm1d(num_features = channels[2], device=device),
            NL_class()
        )
        
    
    def forward(self, x):
        batch_size, num_channels, num_samples = x.size()
        
        x=self.normalization_function(x)
        x=self.stft(x).reshape(batch_size, 1, self.n_bins, -1)
        x = torch.log(x+self.eps)
        x = self.conv1(x)+x
        x = x.mean(3) # Mean across time frames
        x = self.conv2(x)
        x = x.reshape(batch_size, self.channels[2]) # Mean across frequency bins
        return x



class EQNoPad_Encoder(torch.nn.Module):
    def __init__(
        self,
        kernels:list=[25, 5],
        strides:list=[1,1],
        channels:list=[64, 128],
        dilations:list = [1, 1],
        samplerate:int=44100,
        n_bins:int = 113,
        pad = True,
        conv_class=torch.nn.Conv2d,
        NL_class = torch.nn.PReLU,
        device=None,
        normalization_function = no_norm,
        eps = 1e-4
    ):
        super(EQNoPad_Encoder, self).__init__()

        channels = [1]+channels

        self.kernels = kernels
        self.channels = channels
        self.pad=pad
        self.out_dim = channels[-1]
        self.normalization_function = normalization_function
        self.n_bins = n_bins
        self.eps = eps

        self.name="ENC_equaliser_nopad"

        self.stft = nnAudio.features.CQT1992v2(
            n_bins = self.n_bins,
            sr=samplerate
        )

        self.num_convLayers = len(channels)
        
        self.conv1 = torch.nn.Sequential(
            torch.nn.ConstantPad2d(padding = (kernels[1]//2, kernels[1]//2, 0, 0), value = eps),
            torch.nn.Conv2d(
                    in_channels = channels[0],
                    out_channels = channels[1],
                    kernel_size = kernels,
                    stride = strides[0],
                    bias = True,
                    device=device,
            ),
            torch.nn.BatchNorm2d(num_features = channels[1], device=device),
            NL_class()
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv1d(
                    in_channels = channels[1],
                    out_channels = channels[2],
                    kernel_size = n_bins-kernels[0]+1,
                    stride = 1,
                    padding = 0,
                    bias = True,
                    device=device
            ),
            torch.nn.BatchNorm1d(num_features = channels[2], device=device),
            NL_class()
        )
        
    
    def forward(self, x):
        batch_size, num_channels, num_samples = x.size()
        
        x=self.normalization_function(x)
        x=self.stft(x).reshape(batch_size, 1, self.n_bins, -1)
        x = torch.log(x+self.eps)
        x = self.conv1(x)
        x = x.mean(3) # Mean across time frames
        x = self.conv2(x)
        x = x.reshape(batch_size, self.channels[2]) # Mean across frequency bins
        return x


