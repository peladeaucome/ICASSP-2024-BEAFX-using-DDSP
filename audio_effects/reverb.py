import lightning.pytorch as pl
import torch

def reverb(
        x:torch.Tensor,
        decay_time:torch.Tensor,
        dry_wet:torch.Tensor = .5,
        IR_length:int=None,
        samplerate:int = 44100
):
    device = x.device
    if IR_length==None:
        IR_length=3*decay_time*samplerate
    
    noise = torch.randn((2, IR_length), device)
    time = torch.arange(IR_length, device)/samplerate

    IR = torch.zeros((x.size(0), 2, IR_length), device)
    IR = noise*torch.pow(10, 3*time.view(1,1,-1)/decay_time)

    out = torch.functionnal.conv1d(x, IR)
    return out



class SimpleReverb(pl.LightningModule):
    def __init__(
            self,
            noise_length:int = 88200,
            samplerate:int = 44100,
            IR_length = 88200
    ):
        super().__init__()
        self.save_hyperparameters()
        self.num_controls = 2

        if IR_length==None:
            self.IR_length = 2*samplerate
        
        self.noise = torch.rand((2,2*self.hparams.samplerate))
        self.time = torch.arange(IR_length)/self.hparams.samplerate
        
    def forward(
            self,
            x,
            decay_time,
            dry_wet
    ):
        ir = self.noise*torch.power(10, -decay_time/self.hparams.samplerate*1000)
        
    def _apply(self, fn):
        self.noise = fn(self.noise)
        self.time = fn(self.time)