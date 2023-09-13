import torch
import lightning.pytorch as pl
from ..conv_layers import DWPW_Conv2d, Rank1_Conv2d, DWPW_Conv1d
import torchaudio
import numpy as np
from .frontends import *

class Wavenet_Encoder(torch.nn.Module):
    def __init__(
            self,
            kernels_list = [9, 9, 9, 9, 9],
            channels_list = [4, 8, 16, 32, 64],
            stride_list = [1, 1, 1, 1, 1],
            dilation_growth = 9,
            conv_class = torch.nn.Conv1d,
            NL_class=torch.nn.PReLU,
            normalization_function = lambda x: x
    ):
        super(Wavenet_Encoder, self).__init__()

        self.name = "ENC_wavenet"
        self.out_dim = channels_list[-1]

        channels_list = [1]+channels_list

        self.kernels_list = kernels_list
        dilation_list = [dilation_growth**i for i in range(len(kernels_list))]
        self.dilation_list = dilation_list

        self.normalization_function=normalization_function

        self.conv_layers = torch.nn.Sequential()

        for i in range(len(kernels_list)):
            self.conv_layers.append(
                conv_class(
                    in_channels = channels_list[i],
                    out_channels = channels_list[i+1],
                    kernel_size = kernels_list[i],
                    stride = stride_list[i],
                    dilation = dilation_list[i],
                    padding="valid"
                )
            )
            self.conv_layers.append(
                torch.nn.BatchNorm1d(num_features = channels_list[i+1])
            )
            self.conv_layers.append(
                NL_class()
            )
            
        self.pooling = torch.nn.AdaptiveAvgPool1d(1)
    
    def forward(self, x):
        x=self.normalization_function(x)
        x = self.conv_layers(x)
        x = self.pooling(x).squeeze(-1)
        return x


class Compression_Encoder(torch.nn.Module):
    def __init__(
            self,
            eps = 1e-6,
            channels = [16, 32, 64],
            frontend='conv1d',
            normalization_function = torch.nn.BatchNorm1d,
            NL_class = torch.nn.PReLU):
        super(Compression_Encoder, self).__init__()
        self.eps = eps
        self.channels = channels

        self.name = "ENC_compressor"
        self.out_dim = channels[-1]

        if frontend=='gabor':
            self.FrontEnd = GaborFrontEnd(
                out_channels = channels[0],
                kernel_size = channels[0]*2+1,
                stride = channels[0]
            )
        if frontend == 'conv1d':
            self.FrontEnd= Conv1dFrontEnd(
                out_channels = channels[0],
                kernel_size = channels[0]*2+1,
                stride = channels[0]
            )

        self.conv1d = torch.nn.Sequential(
            torch.nn.Conv1d(
                in_channels = channels[0],
                out_channels = channels[0]**2,
                kernel_size = 1
            ),
            torch.nn.BatchNorm1d(num_features = channels[0]**2),
            NL_class(),
            torch.nn.Conv1d(
                in_channels = channels[0]**2,
                out_channels = channels[1],
                kernel_size = 1
            ),
            torch.nn.BatchNorm1d(num_features = channels[1]),
            NL_class(),
            torch.nn.Conv1d(
                in_channels = channels[1],
                out_channels = channels[2],
                kernel_size = 64
            ),
            torch.nn.BatchNorm1d(num_features = channels[2]),
            NL_class()
        )

    
    def forward(self, x):
        x = self.FrontEnd(x)
        x = torch.log(x+self.eps)

        x = self.conv1d(x)

        x = x.mean(2)

        return x
