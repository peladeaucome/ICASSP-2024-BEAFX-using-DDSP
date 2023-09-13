import torch
from .equaliser import EQNoPad_Encoder
from .generic import ResEncoder_1d


class Parallel_encoders(torch.nn.Module):
    def __init__(self, *modules):
        super(Parallel_encoders, self).__init__()
        self.encoders = torch.nn.ModuleList()

        self.out_dim = 0
        for module in modules:
            self.encoders.append(module)
            self.out_dim += module.out_dim
    
    def forward(self, x):
        batch_size, num_channels, num_samples = x.size()
        out = torch.zeros((batch_size, self.out_dim), device = x.device)

        curr_dim = 0
        for module in self.encoders:
            out[:,curr_dim:curr_dim+module.out_dim] = module(x)
            curr_dim+=module.out_dim
        
        return out


class MEE_Equaliser_Encoder(torch.nn.Module):
    def __init__(
            self,
            NL_class,
            normalization_function
    ):
        super(MEE_Equaliser_Encoder, self).__init__()
        self.name='ENC_MEE_EQ'
        self.encoders = torch.nn.ModuleList()
        self.encoders.append(EQNoPad_Encoder(
            NL_class=NL_class,
            normalization_function=normalization_function
        ))
        self.encoders.append(
        ResEncoder_1d(
            NL_class=NL_class,
            normalization_function=normalization_function
        ))

        self.out_dim=0
        for e in self.encoders:
            self.out_dim+= e.out_dim
        
    def forward(self, x):
        batch_size, _, _ = x.size()
        out = torch.zeros((batch_size, self.out_dim), device = x.device)

        curr_dim = 0
        for module in self.encoders:
            out[:,curr_dim:curr_dim+module.out_dim] = module(x)
            curr_dim+=module.out_dim
        
        return out