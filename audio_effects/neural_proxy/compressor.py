import os
import torch
import torchaudio
#import torchsummary
import lightning.pytorch as pl
from ..base import FXBase


def causal_crop(x, num_spl):
    x = x[...,num_spl:]
    return x

def center_crop(x, num_spl):
    len_x = x.size(-1)
    x = x[..., num_spl//2: len_x-num_spl//2]
    return x

class DWPW_Conv1d(pl.LightningModule):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=0,
            dilation=1,
            bias=True,
            decoupling_rank=1,
            padding_mode='zeros',
            device=None,
            dtype=None):
        super(DWPW_Conv1d, self).__init__()
        self.decoupling_rank = decoupling_rank
        self.in_channels=in_channels
        self.out_channels=out_channels
        self.kernel_size=kernel_size

        self.convs = torch.nn.ModuleList()

        for i in range(self.decoupling_rank):
            self.convs.append(
                torch.nn.Sequential(
                    torch.nn.Conv1d(
                        in_channels=in_channels,
                        out_channels=in_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding,
                        dilation=dilation,
                        bias=bias,
                        padding_mode=padding_mode,
                        device=device,
                        dtype=dtype,
                        groups=in_channels
                    ),
                    torch.nn.Conv1d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                        dilation=1,
                        bias=bias,
                        padding_mode=padding_mode,
                        device=device,
                        dtype=dtype,
                        groups=1
            )))
    
    def forward(self, x):
        out = self.convs[0](x)
        for i in range(self.decoupling_rank-1):
            out = out+self.convs[i+1](x)
        return out

class FiLM(torch.nn.Module):
    def __init__(self, 
                 num_features, 
                 cond_dim):
        super(FiLM, self).__init__()
        self.num_features = num_features
        self.bn = torch.nn.BatchNorm1d(num_features, affine=False)
        self.adaptor = torch.nn.Linear(cond_dim, num_features * 2)

    def forward(self, x, cond):

        cond = self.adaptor(cond)
        g, b = torch.chunk(cond, 2, dim=-1)
        g = g.permute(0,2,1)
        b = b.permute(0,2,1)

        #x = self.bn(x)      # apply BatchNorm without affine
        x = (x * g) + b     # then apply conditional affine

        return x

class TCNBlock(torch.nn.Module):
    def __init__(self, 
                in_ch, 
                out_ch, 
                kernel_size=3, 
                padding="same", 
                dilation=1, 
                grouped=False, 
                causal=False,
                conditional=False, 
                **kwargs):
        super(TCNBlock, self).__init__()

        self.in_ch = in_ch
        self.out_ch = out_ch
        self.kernel_size = kernel_size
        self.padding = padding
        self.dilation = dilation
        self.grouped = grouped
        self.causal = causal
        self.conditional = conditional

        conv_class = torch.nn.Conv1d
        #conv_class = DWPW_Conv1d

        groups = out_ch if grouped and (in_ch % out_ch == 0) else 1

        #if padding == "same":
        #    pad_value = (kernel_size - 1) + ((kernel_size - 1) * (dilation-1))
        #elif padding in ["none", "valid"]:
        #    pad_value = 0
        
        if self.causal:
            padding = ((kernel_size-1)*dilation, 0)
        else:
            padding = (dilation*(kernel_size//2),dilation*(kernel_size//2))
        
        self.pad = torch.nn.ConstantPad1d(padding, value = 0)


        
        self.conv1 = conv_class(
                in_ch, 
                out_ch, 
                kernel_size=kernel_size, 
                padding="valid", # testing a change in padding was pad_value//2
                dilation=dilation,
                bias=False)

        if grouped:
            self.conv1b = torch.nn.Conv1d(out_ch, out_ch, kernel_size=1)

        if conditional:
            self.film = FiLM(out_ch, 32)
        else:
            self.bn = torch.nn.BatchNorm1d(out_ch)

        self.relu = torch.nn.PReLU(out_ch)
        self.res = torch.nn.Conv1d(in_ch, 
                                   out_ch, 
                                   kernel_size=1,
                                   groups=in_ch,
                                   bias=False)

    def forward(self, x, p):
        x_in = x
        
        #x = self.pad(x)
        x = self.conv1(x)
        #if self.grouped: # apply pointwise conv
        #    x = self.conv1b(x)
        #if p is not None:   
        x = self.film(x, p) # apply FiLM conditioning
        #else:
        #    x = self.bn(x)
        x = self.relu(x)

        if self.causal:
            x = x + causal_crop(self.res(x_in), (self.kernel_size-1)*self.dilation)
        else:
            x = x + center_crop(self.res(x_in), (self.kernel_size-1)*self.dilation)

        return x

class Compressor(FXBase):
    """ Temporal convolutional network with conditioning module.

        Args:
            `nparams` : int
                 Number of conditioning parameters.
            `ninputs` : int
                Number of input channels (mono = 1, stereo 2). Default: 1
            `noutputs` : int
                Number of output channels (mono = 1, stereo 2). Default: 1
            `nblocks` : int
                Number of total TCN blocks. Default: 10
            `kernel_size` : int
                Width of the convolutional kernels. Default: 3
            `dialation_growth` : int
                Compute the dilation factor at each block as dilation_growth ** (n % stack_size). Default: 1
            `channel_growth` : int
                Compute the output channels at each black as in_ch * channel_growth. Default: 2
            `channel_width` : int
                When channel_growth = 1 all blocks use convolutions with this many channels. Default: 64
            `stack_size` : int
                Number of blocks that constitute a single stack of blocks. Default: 10
            `grouped` : bool
                Use grouped convolutions to reduce the total number of parameters. Default: False
            `causal` : bool
                Causal TCN configuration does not consider future input values. Default: False
            `skip_connections` : bool
                Skip connections from each block to the output. Default: False
            `num_examples` : int
                Number of evaluation audio examples to log after each epochs. Default: 4
        """
    def __init__(self,
                 ninputs:int=1,
                 noutputs:int=1,
                 model_type:str= 'standard',
                 nblocks:int=10, 
                 kernel_size:int=3, 
                 dilation_growth:int=1, 
                 channel_growth:int=1, 
                 channel_width:int=32, 
                 stack_size:int=10,
                 grouped:bool=False,
                 causal:bool=True,
                 skip_connections:bool=False,
                 loss_fn = torch.nn.L1Loss(reduction = "mean"),
                 explicit_makeup:bool = True,
                 explicit_thresh:bool = True):
        super(FXBase, self).__init__()
        self.save_hyperparameters(ignore = ["loss_fn"])
        self.loss_fn = loss_fn

        self.controls_names = ["threshold_dB", "attack_time", "release_time", "ratio", 'knee_dB', "makeup_gain_dB"]
        self.controls_ranges = torch.Tensor([[-40, 0], [1, 60], [1, 500], [2, 10], [0, 12], [0, 20]])
        self.num_controls = 6
        self.learnable_params = self.num_controls-1*explicit_makeup-1*explicit_thresh
        
        self.gen = torch.nn.Sequential(
            torch.nn.Linear(self.learnable_params, 16),
            torch.nn.BatchNorm1d(16),
            torch.nn.PReLU(),
            torch.nn.Linear(16, 32),
            torch.nn.BatchNorm1d(32),
            torch.nn.PReLU(),
            torch.nn.Linear(32, 32),
            torch.nn.BatchNorm1d(32),
            torch.nn.PReLU()
        )

        self.blocks = torch.nn.ModuleList()
        for n in range(nblocks):
            in_ch = out_ch if n > 0 else ninputs
            
            if self.hparams.channel_growth > 1:
                out_ch = in_ch * self.hparams.channel_growth 
            else:
                out_ch = self.hparams.channel_width

            dilation = self.hparams.dilation_growth ** (n % self.hparams.stack_size)
            self.blocks.append(TCNBlock(in_ch, 
                                        out_ch, 
                                        kernel_size=self.hparams.kernel_size, 
                                        dilation=dilation,
                                        padding="same" if self.hparams.causal else "valid",
                                        causal=self.hparams.causal,
                                        grouped=self.hparams.grouped,
                                        conditional=True if self.num_controls > 0 else False))

        self.output = torch.nn.Conv1d(out_ch, noutputs, kernel_size=1)

        self.receptive_field = self.compute_receptive_field()
        if self.hparams.causal:
            self.pad = torch.nn.ConstantPad1d(
                padding = (self.receptive_field-1, 0), 
                value = 0
            )
        else:
            self.pad = torch.nn.ConstantPad1d(
                padding = (self.receptive_field//2, self.receptive_field//2), 
                value = 0
            )
        
        if self.hparams.model_type=='compressor':
            self.out = lambda x, x_in, skips: torch.nn.functional.sigmoid(self.output(x + skips))*x_in
        elif self.hparams.model_type=='standard':
            self.out = lambda x, x_in, skips: torch.nn.functional.tanh(self.output(x + skips))
        
        if explicit_thresh and explicit_makeup:
            self.in_gains = self.in_gains_explicit
            self.out_gains = self.out_gains_explicit
        else:
            self.in_gains = lambda x, p, ranges: x
            self.out_gains = lambda x, p, ranges: x*(10**(ranges[5][1]/20))
        
    def set_controls_to_range(self, q):
        ranges = self.controls_ranges.clone().to(q.device)
        ranges[1:4] = torch.log(ranges[1:4])

        m = ranges[:,0].reshape(1, -1)
        M = ranges[:,1].reshape(1, -1)
        p = q*(M-m) + m
        p[:,1:4] = torch.exp(p[:,1:4])
        return p

    def in_gains_explicit(self, x, p, ranges):
        batch_size = x.size(0)

        threshold_dB = p[:, 0].clone().reshape(batch_size, 1, 1)
        in_gain_threshold = torch.pow(
            10.,
            (threshold_dB/20)
        )
        x = x/in_gain_threshold
        return x
    
    def out_gains_explicit(self, x, p, ranges):
        batch_size = x.size(0)
        threshold_dB = p[:, 0].clone().reshape(batch_size, 1, 1)
        makeup_gain_dB = p[:, 5].clone().reshape(batch_size, 1, 1)
        in_gain_threshold = torch.pow(
            10.,
            (threshold_dB/20)
        )
        makeup_gain = torch.pow(10., makeup_gain_dB/20)
        out = x*makeup_gain
        out = out*in_gain_threshold
        return out
    
    def forward(self, x, q):

        batch_size, num_channels, num_samples = x.size()


        # Setting Controls

        p = self.set_controls_to_range(q)

        cond = self.gen(q[:, 1*self.hparams.explicit_thresh:6-1*self.hparams.explicit_makeup])
        cond = cond.reshape(batch_size, 1, 32)

        x = self.in_gains(x, p, self.controls_ranges)       
        x_in = x.clone()
 #       
        x = self.pad(x)

        # iterate over blocks passing conditioning
        for idx, block in enumerate(self.blocks):
            x = block(x, cond)
            if self.hparams.skip_connections:
                if idx == 0:
                    skips = x
                else:
                    skips = skips + x
            else:
                skips = 0

        out = self.out(x, x_in, skips)
        
        out =  self.out_gains(out, p, self.controls_ranges)

        return out
    
    def get_gains(self, x, q):

        batch_size, num_channels, num_samples = x.size()


        # Setting Controls

        p = self.set_controls_to_range(q)

        cond = self.gen(q[:, 1*self.hparams.explicit_thresh:6-1*self.hparams.explicit_makeup])
        cond = cond.reshape(batch_size, 1, 32)

        x = self.in_gains(x, p, self.controls_ranges)       
        x_in = x.clone()
 #       
        x = self.pad(x)

        # iterate over blocks passing conditioning
        for idx, block in enumerate(self.blocks):
            x = block(x, cond)
            if self.hparams.skip_connections:
                if idx == 0:
                    skips = x
                else:
                    skips = skips + x
            else:
                skips = 0

        out = self.out(x, x_in, skips)
        
        out =  self.out_gains(out, p, self.controls_ranges)

        return out

    
    def training_step(self, batch, batch_idx):
        input, target, params = batch
        pred = self(input, params)  
        loss = self.loss_fn(pred, target)

        self.log('train_loss', 
                 loss, 
                 on_step=True, 
                 on_epoch=True, 
                 prog_bar=True, 
                 logger=True)

        return loss

    def compute_receptive_field(self):
        """ Compute the receptive field in samples."""
        rf = self.hparams.kernel_size
        for n in range(1,self.hparams.nblocks):
            dilation = self.hparams.dilation_growth ** (n % self.hparams.stack_size)
            rf = rf + ((self.hparams.kernel_size-1) * dilation)
        return rf