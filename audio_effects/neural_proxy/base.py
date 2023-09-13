import torch
import numpy as np
from ..base import FXBase

class Half_Proxy(FXBase):
    def __init__(
        self,
        proxy_model,
        dsp_model
    ):
        super(FXBase, self).__init__()

        
        self.proxy_model = proxy_model
        self.proxy_model.eval()
        self.dsp_model = dsp_model
        #if self.proxy_model.controls_ranges == self.dsp_model.controls_ranges:
        self.controls_ranges = proxy_model.controls_ranges.clone()
        
        if self.proxy_model.num_controls == self.dsp_model.num_controls:
            self.num_controls = proxy_model.num_controls
        else:
            ValueError('wrong number of controls')

        if self.proxy_model.controls_names == self.dsp_model.controls_names:
            self.controls_names = proxy_model.controls_names
        else:
            ValueError('wrong ranges')
        

    def apply(self, fn):
        for module in self.children():
            module.apply(fn)
        self.proxy_model.eval()
        #self.controls_ranges=fn(self.controls_ranges)
        return self
    
    def eval(self):
        self.process = lambda x, q: self.dsp_model(x, q)
    
    def train(self):
        self.proxy_model.eval()
        self.process = lambda x, q: self.proxy_model(x, q)
        
    
    def forward(self, x, q):
        out = self.process(x, q)
        return out