import torch
import lightning.pytorch as pl

class FXBase(pl.LightningModule):
    """Base class for all FX in this project"""
    def __init__(self):
        super(FXBase).__init__()
        self.num_controls = 0
        self.controls_names = []
        self.controls_ranges = torch.Tensor([[]])
        self.effect_name = ''
    
    def apply(self, fn):
        for module in self.children():
            module.apply(fn)
        #self.controls_ranges=fn(self.controls_ranges)
        return self

    def set_controls_to_range(self, p:torch.Tensor):
        """Denormalizes parameters from [0;1] to [p_min; p_max]"""
        self.controls_ranges = self.controls_ranges.to(p.device)
        m = self.controls_ranges[:,0] # Min
        M = self.controls_ranges[:,1] # Max
        p = p*(M-m)+m
        return p
    
    def set_ranges_from_dict(self, ranges_dict):
        """Utility. Allows to set the effects' parameters ranges from a dictionnary"""
        for i, control in enumerate(self.controls_names):
            self.controls_ranges[i] = torch.Tensor(ranges_dict[control])
    
    def get_controls_dict(self, p):
        out = {}
        for i, name in enumerate(self.controls_names):
            out[name] = p[:,i]
    
    def forward(self, x, q):
        """Applies the effects with normalized parameters `q` to the audio `x`"""
        if self.num_controls != q.size(1):
            raise ValueError(f'Wrong number of controls. Was given {q.size(1)} controls, expected {self.num_controls}')
        p = self.set_controls_to_range(q)
        return self.process(x, p)