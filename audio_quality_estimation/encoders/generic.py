import torch
import audio_quality_estimation.conv_layers as conv_layers
from ..normalize import max_norm, rms_norm, no_norm


def center_crop(x, num_samples):
    return x[...,num_samples//2:num_samples//2]


class ResEncoder_1d(torch.nn.Module):
    def __init__(
            self,
            kernels:list = [25, 25, 15, 15, 10, 10, 10, 10, 5, 5, 5, 5],
            channels:list  = [16, 32, 64, 128, 256, 256, 512, 512, 1024, 1024, 2048, 2048],
            strides:list = [4, 4, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1],
            dilation_growth:int = 1,
            conv_class= torch.nn.Conv1d,
            NL_class = torch.nn.PReLU,
            device=None,
            normalization_function=no_norm
        ):
        super(ResEncoder_1d, self).__init__()

        self.normalization_function=normalization_function

        self.out_dim = channels[-1]

        self.channels = [1]
        for c in channels:
            self.channels.append(c)
        
        self.convs = torch.nn.Sequential()

        self.name="ENC_MEE"

        for i in range(len(kernels)):
            self.convs.append(
                conv_layers.Res_Conv1d(
                    in_channels = self.channels[i],
                    out_channels=  self.channels[i+1],
                    kernel_size = kernels[i],
                    stride = strides[i],
                    dilation = dilation_growth**i,
                    conv_class=conv_class,
                    device=device
                )
            )
            #self.convs.append(
            #    torch.nn.BatchNorm1d(num_features =self.channels[i+1])
            #)
            #self.convs.append(
            #    NL_class()
            #)
        
        self.pooling = torch.nn.AdaptiveAvgPool1d(1)
        
    def forward(self, x):
        x = self.normalization_function(x)
        x = self.convs(x)
        x = self.pooling(x).squeeze(-1)
        return x

class ResEncoder_1d_Hilbert(torch.nn.Module):
    def __init__(
            self,
            kernels:list = [25, 25, 15, 15, 10, 10, 10, 10, 5, 5, 5, 5],
            #channels:list  = [16, 32, 64, 128, 256, 256, 512, 512, 1024, 1024, 2048, 2048],
            channels:list  = [4, 8, 16, 32, 64, 64, 128, 128, 256, 256, 512, 512],
            strides:list = [4, 4, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1],
            dilation_growth:int = 1,
            conv_class= torch.nn.Conv1d,
            NL_class = torch.nn.PReLU,
            device=None,
            normalization_function=no_norm
        ):
        super(ResEncoder_1d_Hilbert, self).__init__()

        self.normalization_function=normalization_function

        self.out_dim = channels[-1]

        self.channels = [1]
        for c in channels:
            self.channels.append(c)
        
        self.convs = torch.nn.Sequential()

        self.name="ENC_MEE"

        for i in range(len(kernels)):
            self.convs.append(
                conv_layers.Res_Conv1d(
                    in_channels = self.channels[i],
                    out_channels=  self.channels[i+1],
                    kernel_size = kernels[i],
                    stride = strides[i],
                    dilation = dilation_growth**i,
                    conv_class=conv_layers.Hilbert_Conv1d,
                    device=device
                )
            )
            #self.convs.append(
            #    torch.nn.BatchNorm1d(num_features =self.channels[i+1])
            #)
            #self.convs.append(
            #    NL_class()
            #)
        
        self.pooling = torch.nn.AdaptiveAvgPool1d(1)
        
    def forward(self, x):
        x = self.normalization_function(x)
        x = self.convs(x)
        x = self.pooling(x).squeeze(-1)
        return x


class ResEncoderMax_1d(torch.nn.Module):
    def __init__(
            self,
            kernels:list = [25, 25, 15, 15, 10, 10, 10, 10, 5, 5, 5, 5],
            channels:list  = [16, 32, 64, 128, 256, 256, 512, 512, 1024, 1024, 2048, 2048],
            strides:list = [4, 4, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1],
            #strides:list = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            dilation_growth:int = 1,
            conv_class= torch.nn.Conv1d,
            NL_class = torch.nn.PReLU,
            device=None,
            normalization_function=no_norm
        ):
        super(ResEncoderMax_1d, self).__init__()

        self.normalization_function=normalization_function

        self.out_dim = channels[-1]

        self.channels = [1]
        for c in channels:
            self.channels.append(c)
        
        self.convs = torch.nn.Sequential()

        self.name="ENC_MEE_max"

        for i in range(len(kernels)):
            self.convs.append(
                conv_layers.Res_Conv1d(
                    in_channels = self.channels[i],
                    out_channels=  self.channels[i+1],
                    kernel_size = kernels[i],
                    stride = strides[i],
                    dilation = dilation_growth**i,
                    conv_class=conv_class,
                    device=device
                )
            )
            self.convs.append(
                torch.nn.BatchNorm1d(num_features =self.channels[i+1])
            )
            self.convs.append(
                NL_class()
            )
        
        self.pooling = torch.nn.AdaptiveMaxPool1d(1)
        
    def forward(self, x):
        x = self.normalization_function(x)
        x = self.convs(x)
        x = self.pooling(x).squeeze(-1)
        return x

