import lightning.pytorch as pl
import torch
import audio_effects.base



class FXChain(audio_effects.base.FXBase):
    """Class used to group multiple FX into one chain."""
    def __init__(
        self,
        norm = 'max'):
        """
        `norm` : str
            Normalisation between eache effect. Can be None, max or rms. Default : None
        """
        super(audio_effects.base.FXBase, self).__init__()
        self.norm = norm
        self.num_controls=0
        self.controls_names = []
        self.controls_ranges = torch.Tensor([[]])
        self.module_list = []

        if norm in [None, 'max', 'rms']:
            self.norm = norm
            if self.norm =='max':
                self.norm_func = self.max_norm
            elif self.norm == 'rms':
                self.norm_func = self.rms_norm
            else:
                self.norm_func = lambda x: x
        else:
            raise ValueError('Invalid value pro "norm". It should be None, "max" or "rms"')
    
    def set_controls_to_range(self, q):
        return q
    
    def append_FX(self, FX_module):
        """Adds the effect `FX_module` to the chain."""
        self.module_list.append(FX_module)
        
        old_num = self.num_controls
        old_ranges = self.controls_ranges.clone()
        self.num_controls+=FX_module.num_controls
        
        if old_num!=0:
            self.controls_ranges = torch.zeros((self.num_controls,2))
            self.controls_ranges[0:old_num, :] = old_ranges
            self.controls_ranges[old_num:, :] = FX_module.controls_ranges

        else:
            self.controls_ranges=FX_module.controls_ranges.clone()
        
        for name in FX_module.controls_names:
            self.controls_names.append(name)

    def process(self, x, q):
        """Processes `x` through the chain of effects with normalized parameters `q`."""
        control_index = 0
        for i, module in enumerate(self.module_list):
            x = x - torch.mean(x, 2).reshape(-1, 1, 1)
            x = self.norm_func(x)
            x = module(x, q[:,control_index: control_index+module.num_controls])
            control_index+=module.num_controls
        return x

    def max_norm(self, x):
        """Normalizes the audio `x` to 0dBFS."""
        xmax, _ = torch.max(torch.abs(x), dim=2)
        x=x/(xmax.reshape(-1, 1, 1))
        return x

    def rms_norm(self, x):
        """Normalizes the audio `x` to 0dB RMS."""
        x = x/(torch.std(x, dim=2).reshape(-1, 1, 1))
        return x

    def _apply(self, fn):
        for module in self.module_list:
            module = fn(module)
        
        return self
    
    def eval(self):
        for module in self.module_list:
            module.eval()
    
    def train(self):
        for module in self.module_list:
            module.train()