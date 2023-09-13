import torch
import torchaudio
from ..conv_layers import DWPW_Conv2d, Padded_Conv2d, harmonic_CQT, ParallelTimbre_Conv2d
import numpy as np
import nnAudio.features
from ..normalize import rms_norm, max_norm, no_norm

class MultiTimbreCQT_Encoder(torch.nn.Module):
    def __init__(
        self,
        samplerate:int=44100,
        n_bins:int = 113,
        conv_class=torch.nn.Conv2d,
        NL_class = torch.nn.PReLU,
        device=None,
        normalization_function = no_norm,
        eps = 1e-6,
        compute_representation=True
    ):
        super(MultiTimbreCQT_Encoder, self).__init__()
        self.n_bins= n_bins

        self.eps = eps

        channels_list = [10, 6, 3, 3,
                         10, 6, 3, 3,
                         10, 6, 3, 3]
        kernels_list = [
            [85, 1], [85, 3], [85, 5], [85, 7],
            [51, 1], [51, 3], [51, 5], [51, 7],
            [25, 1], [25, 3], [25, 5], [25, 7],
        ]


        self.normalization_function = normalization_function
        self.out_dim = sum(channels_list)
        self.name="ENC_MultiTimbreCQT"
        self.compute_representation=compute_representation

        if compute_representation:
            self.cqt = nnAudio.features.CQT1992v2(
                n_bins = n_bins,
                sr=samplerate
            )
        
        self.conv1 = ParallelTimbre_Conv2d(
            in_channels = 1,
            channels_list = channels_list,
            kernels_list=kernels_list,
            pad_value = self.eps,
            n_bins=n_bins
        )
        self.norm = torch.nn.BatchNorm1d(num_features=self.out_dim)
        self.activation = NL_class()
        
    
    def forward(self, x):
        batch_size, num_channels, num_samples = x.size()
        x=self.normalization_function(x)

        x=self.cqt(x).reshape(batch_size, 1, self.n_bins, -1)
        x = torch.log(x+self.eps)

        x = self.conv1(x)
        x = x.mean(3) # Mean across time frames
        x = x.reshape(batch_size, self.out_dim) # Mean across frequency bins
        x = self.norm(x)
        x = self.activation(x)
        return x


class MultiTimbreHCQT_Encoder(torch.nn.Module):
    def __init__(
        self,
        samplerate:int=44100,
        n_bins:int = 101,
        conv_class=torch.nn.Conv2d,
        NL_class = torch.nn.PReLU,
        device=None,
        normalization_function = no_norm,
        eps = 1e-6,
        n_harmonics=6
    ):
        super(MultiTimbreHCQT_Encoder, self).__init__()
        self.n_bins= n_bins
        self.n_harmonics = n_harmonics
        self.eps = eps

        channels_list = [6, 3, 1, 1,
                         6, 3, 1, 1,
                         6, 3, 1, 1]
        kernels_list = [
            [85, 1], [85, 3], [85, 5], [85, 7],
            [51, 1], [51, 3], [51, 5], [51, 7],
            [25, 1], [25, 3], [25, 5], [25, 7],
        ]


        self.normalization_function = normalization_function
        self.out_dim = sum(channels_list)
        self.name="ENC_MultiTimbreHCQT"

        self.cqt = harmonic_CQT(
            sr = 44100,
            n_bins=n_bins,
            n_harmonics=n_harmonics
        )

        
        self.conv1 = ParallelTimbre_Conv2d(
            in_channels = n_harmonics,
            channels_list = channels_list,
            kernels_list=kernels_list,
            pad_value = self.eps,
            n_bins=n_bins
        )
        self.norm = torch.nn.BatchNorm1d(num_features=self.out_dim)
        self.activation = NL_class()
        
    
    def forward(self, x):
        batch_size, num_channels, num_samples = x.size()
        x=self.normalization_function(x)

        x=self.cqt(x).reshape(batch_size, self.n_harmonics, self.n_bins, -1)
        x = torch.log(x+self.eps)

        x = self.conv1(x)
        x = x.mean(3) # Mean across time frames
        x = x.reshape(batch_size, self.out_dim) # Mean across frequency bins
        x = self.norm(x)
        x = self.activation(x)
        return x


class TimeCQT_Encoder(torch.nn.Module):
    def __init__(
        self,
        samplerate:int=44100,
        n_bins:int = 113,
        conv_class=torch.nn.Conv2d,
        NL_class = torch.nn.PReLU,
        out_channels = 64,
        kernel_size = [1, 128],
        device=None,
        normalization_function = no_norm,
        eps = 1e-6,
        compute_representation = True
    ):
        super(TimeCQT_Encoder, self).__init__()
        self.n_bins= n_bins

        self.eps = eps

        self.normalization_function = normalization_function
        self.out_dim = out_channels
        self.compute_representation = compute_representation
        self.name="ENC_TimeCQT"
        if compute_representation:
            self.cqt = nnAudio.features.CQT1992v2(
                n_bins = n_bins,
                sr=samplerate
            )
            print(f"CQT kernel width : {self.cqt.kernel_width}")
        
        self.conv1 = torch.nn.Sequential(
            conv_class(
                in_channels = 1,
                out_channels=out_channels,
                kernel_size=kernel_size,
            ),
            torch.nn.BatchNorm2d(num_features = out_channels),
            NL_class()
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv1d(
                in_channels = out_channels,
                out_channels = out_channels,
                kernel_size = 31
            ),
            torch.nn.BatchNorm1d(num_features = out_channels),
            NL_class()
        )
        
    
    def forward(self, x):
        batch_size = x.size(0)
        x=self.normalization_function(x)

        if self.compute_representation:
            x=self.cqt(x).reshape(batch_size, 1, self.n_bins, -1)
            x = torch.log(x+self.eps)

        x = self.conv1(x)
        x, _ = torch.max(x, dim=2)
        #x = torch.mean(x, dim=2)
        x = self.conv2(x)
        x = x.mean(2) # Mean across time frames
        x = x.reshape(batch_size, self.out_dim) # Mean across frequency bins
        return x

class FrequencyCQT_Encoder(torch.nn.Module):
    def __init__(
        self,
        samplerate:int=44100,
        n_bins:int = 113,
        conv_class=torch.nn.Conv2d,
        NL_class = torch.nn.PReLU,
        out_channels = 64,
        kernel_size = [37, 1],
        device=None,
        normalization_function = no_norm,
        eps = 1e-6,
        compute_representation=True
    ):
        super(FrequencyCQT_Encoder, self).__init__()
        self.n_bins= n_bins

        self.eps = eps

        self.normalization_function = normalization_function
        self.out_dim = out_channels
        self.compute_representation = compute_representation
        self.name="ENC_FrequencyCQT"

        if compute_representation:
            self.cqt = nnAudio.features.CQT1992v2(
                n_bins = n_bins,
                sr=samplerate
            )
            print(f"CQT kernel width : {self.cqt.kernel_width}")
        

        self.conv1 = torch.nn.Sequential(
            conv_class(
                in_channels = 1,
                out_channels=out_channels,
                kernel_size=kernel_size,
            ),
            torch.nn.BatchNorm2d(num_features = out_channels),
            NL_class()
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv1d(
                in_channels = out_channels,
                out_channels = out_channels,
                kernel_size = n_bins-kernel_size[0]+1
            ),
            torch.nn.BatchNorm1d(num_features = out_channels),
            NL_class()
        )
        
    
    def forward(self, x):
        batch_size = x.size(0)
        x=self.normalization_function(x)

        if self.compute_representation:
            x=self.cqt(x).reshape(batch_size, 1, self.n_bins, -1)
            x = torch.log(x+self.eps)

        x = self.conv1(x)
        x, _ = torch.max(x, dim=3)
        #x = torch.mean(x, dim=3)
        #x = x.mean(2) # Mean across time frames
        x = self.conv2(x)
        x = x.reshape(batch_size, self.out_dim) # Mean across frequency bins
        return x


class TimeFrequencyCQT_Encoder(torch.nn.Module):
    def __init__(
        self,
        samplerate:int=44100,
        n_bins:int = 113,
        conv_class=torch.nn.Conv2d,
        NL_class = torch.nn.PReLU,
        device=None,
        normalization_function = no_norm,
        eps = 1e-6,
        compute_representation=True
    ):
        super(TimeFrequencyCQT_Encoder, self).__init__()
        self.n_bins= n_bins

        self.eps = eps

        self.normalization_function = normalization_function
        self.compute_representation = compute_representation
        self.name="ENC_TimeFrequencyCQT"

        if self.compute_representation:
            self.cqt = nnAudio.features.CQT1992v2(
                n_bins = n_bins,
                sr=samplerate
            )
            print(f"CQT kernel width : {self.cqt.kernel_width}")
        
        self.time = TimeCQT_Encoder(NL_class=NL_class, compute_representation=False)
        self.frequency = FrequencyCQT_Encoder(NL_class=NL_class, compute_representation=False)
        self.out_dim = self.time.out_dim+self.frequency.out_dim

    
    def forward(self, x):
        batch_size = x.size(0)
        x=self.normalization_function(x)

        if self.compute_representation:
            x=self.cqt(x).reshape(batch_size, 1, self.n_bins, -1)
            x = torch.log(x+self.eps)
        
        out = torch.zeros((batch_size, self.out_dim), device= x.device)

        out[:, :self.time.out_dim] = self.time(x)
        out[:, self.time.out_dim:] = self.frequency(x)
        return out