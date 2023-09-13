import torch
import lightning.pytorch as pl
import numpy as np

class DWPW_Conv1d(torch.nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=0,
            dilation=1,
            bias=True,
            decoupling_rank:int=1,
            padding_mode='zeros',
            device=None,
            dtype=None,
            groups:int = 1):
        super(DWPW_Conv1d, self).__init__()
        self.decoupling_rank = decoupling_rank
        self.in_channels=in_channels
        self.out_channels=out_channels
        self.kernel_size=kernel_size

        self.convs = torch.nn.Sequential(
            torch.nn.Conv1d(
                in_channels=in_channels,
                out_channels=in_channels*decoupling_rank,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                bias=False,
                padding_mode=padding_mode,
                device=device,
                dtype=dtype,
                groups=in_channels
            ),
            torch.nn.Conv1d(
                in_channels=in_channels*decoupling_rank,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                dilation=1,
                bias=bias,
                padding_mode=padding_mode,
                device=device,
                dtype=dtype,
                groups=groups
        ))
    
    def forward(self, x):
        out = self.convs(x)
        return out



class Res_Conv1d(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        dilation=1,
        groups=1,
        bias=True,
        device = None,
        dtype= None,
        conv_class = torch.nn.Conv1d,
        NL_class = torch.nn.PReLU,
    ):
        super(Res_Conv1d, self).__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.stride = stride
        
        
        self.crop_samples= (self.kernel_size-1)*self.dilation

        self.conv1 = torch.nn.Sequential(
                conv_class(
                    in_channels = in_channels,
                    out_channels = in_channels,
                    kernel_size = kernel_size,
                    stride = 1,
                    dilation = dilation,
                    groups = 1,
                    bias = bias,
                    #device = device,
                    #dtype = dtype,
                    padding='same'
                ),
                torch.nn.BatchNorm1d(num_features=in_channels),
                NL_class()
        )
        
        self.conv2=torch.nn.Sequential(
            torch.nn.Conv1d(
                in_channels = in_channels,
                out_channels = out_channels,
                kernel_size = kernel_size,
                stride = stride,
                dilation = dilation,
                groups = groups,
                bias = True,
                #device = device,
                #dtype = dtype,
                padding='valid'
            ),
            torch.nn.BatchNorm1d(num_features=out_channels),
            NL_class()
        )
        
    def crop(self, x, num_samples):
        len_x = x.size(-1)
        spls = [num_samples//2, num_samples-num_samples//2]
        out = x[:,:,spls[0]:len_x-spls[1]]
        return out
    
    def forward(self, x):
        out = self.conv1(x) + x
        return self.conv2(out)


class Hilbert_Conv1d(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        dilation=1,
        groups=1,
        padding=0,
        padding_mode='zeros',
        bias=True,
    ):
        super(Hilbert_Conv1d, self).__init__()
        self.kernel_size = kernel_size
        self.out_channels = out_channels
        self.dilation = dilation
        self.stride = stride
        self.groups=groups,
        self.padding=padding
        self.padding_mode=padding_mode

        if in_channels%groups!=0:
            raise ValueError('please')
        
        sqrtk=np.sqrt(groups/(in_channels*kernel_size))
        self.weight_r = torch.nn.Parameter(((torch.rand(out_channels, in_channels//groups, kernel_size)-.5)*2*sqrtk))

        if bias:
            self.bias = torch.nn.Parameter(((torch.rand(out_channels)-.5)*2*sqrtk))
        else:
            self.bias=0
    
    def get_weight_i(self):
        w_fft = torch.fft.rfft(self.weight_r, dim=-1)
        w_fft = w_fft*np.exp(1j*np.pi*.5)
        weight_i=torch.real(torch.fft.irfft(w_fft, dim=-1, n=self.kernel_size))
        return weight_i

    def forward(self, x):

        self.weight_i = self.get_weight_i()

        out_r = torch.nn.functional.conv1d(
            input=x,
            weight=self.weight_r,
            stride=self.stride,
            padding=self.padding
        )

        out_i = torch.nn.functional.conv1d(
            input=x,
            weight=self.weight_i,
            stride=self.stride,
            padding=self.padding
        )

        out = torch.sqrt(torch.square(out_r)+torch.square(out_i))+self.bias.reshape(1, self.out_channels,1)
        return out