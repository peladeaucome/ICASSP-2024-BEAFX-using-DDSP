import numpy as np
from numba import jit
import torch
from ..base import FXBase

class NPFXBase:
    def __init__(self):
        self.controls_names = []
        self.controls_ranges = [[]]
        self.num_controls=0
    
    def set_ranges_from_dict(self, ranges_dict):
        for i, control in enumerate(self.controls_names):
            self.controls_ranges[i] = torch.Tensor(ranges_dict[control])
    
    def __call__(self, x, q):
        
        if self.num_controls != q.size(1):
            raise ValueError(f'Wrong number of controls. Was given {p.size(1)} controls, expected {self.num_controls}')
        p = self.set_controls_to_range(q)
        return self.process(x, p)

    def get_controls_dict(self, p):
        out={}
        for i,name in enumerate(self.controls_names):
            out[name]=float(p[i])
        return out
        
    
    def process(
        self,
        x,
        q):

        batch_size, num_channels, num_samples = x.size()
        x_np = x.clone().cpu().numpy()
        out = np.zeros(x_np.shape)
        for batch_idx in range(batch_size):
            p_dict = self.get_controls_dict(q[batch_idx])
            out[batch_idx, 0, :] = self.effect(
                x=x_np[batch_idx, 0, :],
                **p_dict
            )

        out = torch.Tensor(out)
        out = out.to(x.device)
        return out
