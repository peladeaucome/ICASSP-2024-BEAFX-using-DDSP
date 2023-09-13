import torch
import nnAudio


class harmonic_CQT(torch.nn.Module):
    def __init__(
            self,
            n_harmonics=6,
            sr=22050,
            hop_length=512,
            fmin=32.70,
            fmax=None,
            n_bins=101,
            bins_per_octave=12,
            filter_scale=1,
            norm=1,
            window="hann",
            center=True,
            pad_mode="reflect",
            trainable=False,
            output_format="Magnitude",
            verbose=True,
    ):
        super(harmonic_CQT, self).__init__()
        self.CQTs=torch.nn.ModuleList()
        self.n_harmonics=n_harmonics

        fmin = fmin*2
        for h in range(n_harmonics):
            self.CQTs.append(
                    nnAudio.features.CQT1992v2(
                        sr=sr,
                        hop_length=hop_length,
                        fmin = fmin/(h+1),
                        fmax=fmax,
                        n_bins=n_bins,
                        bins_per_octave=bins_per_octave,
                        filter_scale=filter_scale,
                        norm=1,
                        window=window,
                        center=center,
                        pad_mode=pad_mode,
                        trainable=trainable,
                        output_format=output_format,
                        verbose=verbose
                ))
        
    
    def forward(self,x):
        batch_size, _, num_samples = x.size()

        x_cqt1 = self.CQTs[0](x)

        batch_size, n_bins, n_time = x_cqt1.size()

        h_CQT = torch.zeros((batch_size, self.n_harmonics, n_bins, n_time), device = x.device)
        h_CQT[:,0,:,:] = x_cqt1

        for h in range(1,self.n_harmonics):
            h_CQT[:,h,:,:] = self.CQTs[h](x)

        return h_CQT