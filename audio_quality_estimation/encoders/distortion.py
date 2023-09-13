import torch
import torchaudio
from ..conv_layers import DWPW_Conv2d, Padded_Conv2d, harmonic_CQT, ParallelTimbre_Conv2d
import numpy as np
import nnAudio.features
from ..normalize import rms_norm, max_norm, no_norm

class TimbreHCQT_Encoder(torch.nn.Module):
    def __init__(
        self,
        kernel_size:list=[25, 5],
        stride:list=1,
        channels:list=32,
        dilation=[1, 1],
        pad=True,
        conv_class=torch.nn.Conv2d,
        NL_class = torch.nn.PReLU,
        device=None,
        normalization_function = no_norm,
        eps=1e-6,
        n_bins = 101,
        n_harmonics = 6
    ):
        super(TimbreHCQT_Encoder, self).__init__()

        self.out_dim = channels
        self.pad= pad
        self.kernel_size = kernel_size
        self.normalization_function = normalization_function
        self.eps = eps

        self.n_harmonics = n_harmonics
        self.n_bins = n_bins

        self.name="ENC_TimbreCQT"
        
        #self.stft = torch.nn.Sequential(
        #    torchaudio.transforms.Spectrogram(
        #        n_fft = n_fft,
        #        win_length = n_fft,
        #        hop_length = n_fft//2,
        #        return_complex = None,
        #        power=2
        #    ),
        #    torchaudio.transforms.MelScale(
        #        n_mels = 129,
        #        sample_rate = 44100,
        #        n_stft = n_fft//2+1,
        #        f_min = 20,
        #        f_max = 20000
        #    )
        #)
        self.stft = harmonic_CQT(sr = 44100,n_bins=n_bins, n_harmonics=n_harmonics)
        
        self.conv_layers = torch.nn.Sequential(
            Padded_Conv2d(
                    in_channels = n_harmonics,
                    out_channels = channels,
                    kernel_size = kernel_size,
                    stride = stride,
                    bias = True,
                    device=device,
                    pad_value=eps
            ),
            torch.nn.BatchNorm2d(num_features = channels, device= device),
            NL_class()
        )

        
    
    def forward(self, x):
        batch_size, num_channels, num_samples = x.size()
        #x = x/(torch.std(x, axis = 2).reshape(x.size(0), 1, 1)*10)
        x = self.normalization_function(x)
        x = self.stft(x).reshape(batch_size, self.n_harmonics, self.n_bins, -1)
        #print(x)
        x = torch.log(x+self.eps)
        
        #x = torch.log(1+self.stft(x))
        x = self.conv_layers(x)
        x = x.mean(3) # Mean across time frames
        x = x.mean(2) # Mean across frequency bins
        return x
    


