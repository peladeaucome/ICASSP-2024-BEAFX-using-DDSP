import torch
import torchaudio
import lightning.pytorch as pl
import numpy as np
from .conv_layers import Rank1_Conv2d, DWPW_Conv2d
import auraloss

def dB20(x):
    out = torch.zeros_like(x)
    out = torch.where(
        condition = x>1e-6,
        input = 20*torch.log10(torch.abs(x)),
        other = -120.*torch.ones_like(x)
    )
    return out


class Controller_Network(pl.LightningModule):
    def __init__(
            self,
            audio_fx,
            encoder,
            mlp_layers = [2048, 1024, 512],
            learning_rate:float=1e-3,
            NL_class= torch.nn.PReLU,
            loss_type:str='mel',
            fft_sizes = [256, 1024, 4096],
            hop_sizes = [64, 256, 1024],
            win_lengths = [256, 1024, 4096],
            device='cpu'
    ):
        super(Controller_Network, self).__init__()

        self.num_controls = audio_fx.num_controls

        self.encoder = encoder
        encoded_dim = encoder.out_dim

        mlp_layers = [encoded_dim]+mlp_layers
        
        
        self.num_FCLayers = len(mlp_layers)
        self.learning_rate = learning_rate

        self.MLP = torch.nn.Sequential()

        for i in range(self.num_FCLayers-1):
            self.MLP.append(
                    torch.nn.Linear(
                    in_features  = mlp_layers[i],
                    out_features = mlp_layers[i+1]
                )
            )
            self.MLP.append(torch.nn.BatchNorm1d(num_features = mlp_layers[i+1]))
            self.MLP.append(NL_class())

        self.MLP.append(
                torch.nn.Linear(
                in_features = mlp_layers[-1],
                out_features = self.num_controls
            )
        )
        self.MLP.append(
            torch.nn.Sigmoid()
        )


    def forward(self, x):

        x = self.encoder(x)
        q = self.MLP(x)

        return q
    
    def training_step(self, batch, batch_idx):
        input, target = batch
        params = self(target)

        estimate = self.FX_chain(input, params)

        loss = self.loss_fn(target, estimate)

        return loss

    def validation_step(self, batch, batch_idx):
        input, target = batch
        params = self(target)

        estimate = self.FX_chain(input, params)

        loss = self.loss_fn(target, estimate)
        
        pl.log("val_loss", loss)
        return loss
    
    def test_step(self, batch, batch_idx):
        input, target = batch
        params = self(target)

        estimate = self.FX_chain(input, params)

        loss = self.loss_fn(target, estimate)

        pl.log("test_loss", loss)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr = self.learning_rate)
