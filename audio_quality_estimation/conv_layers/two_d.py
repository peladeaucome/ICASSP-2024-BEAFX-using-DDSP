import torch

class DWPW_Conv2d(torch.nn.Module):
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
            dtype=None,
            groups:int = 1):
        super(DWPW_Conv2d, self).__init__()
        self.decoupling_rank = decoupling_rank
        self.in_channels=in_channels
        self.out_channels=out_channels
        self.kernel_size=kernel_size

        self.convs = torch.nn.Sequential(
            torch.nn.Conv2d(
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
            torch.nn.Conv2d(
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

class PWDW_Conv2d(torch.nn.Module):
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
            dtype=None,
            groups:int = 1):
        super(PWDW_Conv2d, self).__init__()
        self.decoupling_rank = decoupling_rank
        self.in_channels=in_channels
        self.out_channels=out_channels
        self.kernel_size=kernel_size

        self.convs = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=in_channels,
                out_channels=in_channels*decoupling_rank,
                kernel_size=1,
                stride=1,
                padding=0,
                dilation=1,
                bias=bias,
                padding_mode=padding_mode,
                device=device,
                dtype=dtype,
                groups=groups
            ),
            torch.nn.Conv2d(
                in_channels=in_channels*decoupling_rank,
                out_channels=out_channels,
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
        )
    
    def forward(self, x):
        out = self.convs(x)
        return out


class Rank1_Conv2d(torch.nn.Module):
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
        super(Rank1_Conv2d, self).__init__()
        self.decoupling_rank = decoupling_rank
        self.in_channels=in_channels
        self.out_channels=out_channels
        self.kernel_size=kernel_size

        self.convs = torch.nn.ModuleList()

        for i in range(self.decoupling_rank):
            self.convs.append(
                torch.nn.Sequential(
                    torch.nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=(kernel_size[0], 1),
                        stride= (stride[0],1),
                        padding= (0, padding[1]),
                        dilation=dilation,
                        bias=bias,
                        padding_mode=padding_mode,
                        device=device,
                        dtype=dtype,
                        groups=1
                    ),
                    torch.nn.Conv2d(
                        in_channels=out_channels,
                        out_channels=out_channels,
                        kernel_size=(1, kernel_size[1]),
                        stride= (1, stride[1]),
                        padding= (0, padding[1]),
                        dilation=(1,1),
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


class Res_Conv2d(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=[1,1],
        dilation=[1,1],
        groups=1,
        bias=True,
        device = None,
        dtype= None,
        conv_class = torch.nn.Conv1d
    ):
        super(Res_Conv2d, self).__init__()

        stride=self.fromIntToList(stride)
        dilation = self.fromIntToList(dilation)
        kernel_size = self.fromIntToList(kernel_size)

        self.kernel_size = kernel_size
        self.dilation = dilation
        self.stride = stride
        
        
        
        self.crop_samples = [
            (self.kernel_size[0]-1)*self.dilation[0],
            (self.kernel_size[1]-1)*self.dilation[1]
        ]


        self.conv1 = conv_class(
            in_channels = in_channels,
            out_channels = in_channels,
            kernel_size = kernel_size,
            stride = stride,
            dilation = dilation,
            groups = groups,
            bias = bias,
            device = device,
            dtype = dtype,
            padding='valid'
        )
        
        if in_channels==out_channels:
            self.conv2 = lambda x: x
        else:
            self.conv2=torch.nn.Conv1d(
                in_channels = in_channels,
                out_channels = out_channels,
                kernel_size = 1,
                stride = 1,
                dilation = 1,
                groups = 1,
                bias = True,
                device = device,
                dtype = dtype,
                padding='valid'
            )
    
    def fromIntToList(self, a):
        if type(a)==int:
            return [a,a]
        else:
            return a
    
    def crop(self, x, num_samples):
        _, _, height, width = x.size()
        spls = [
            num_samples[0]//2, num_samples[0]-num_samples[0]//2,
            num_samples[1]//2, num_samples[1]-num_samples[1]//2
        ]
        out = x[
            :,
            :,
            spls[0][0]:height-spls[0][1],
            spls[1][0]:width-spls[1][1]
        ]
        return out
    
    def forward(self, x):
        out = self.conv1(x) + self.crop(x, self.crop_samples)[:,:,::self.stride[0],::self.stride[1]]
        return self.conv2(out)


class Padded_Conv2d(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=[1,1],
        dilation=[1,1],
        groups=1,
        bias=True,
        device = None,
        dtype= None,
        pad_value:float = 0,
        conv_class = torch.nn.Conv2d
    ):
        super(Padded_Conv2d, self).__init__()
        kernel_size = self.fromIntToList(kernel_size)
        dilation = self.fromIntToList(dilation)
        
        self.pad = torch.nn.ConstantPad2d(
            padding = (
                (kernel_size[1]//2)*dilation[1],
                (kernel_size[1]-kernel_size[1]//2-1)*dilation[1],
                (kernel_size[0]-1)*dilation[0],
                0),
            value = pad_value)
        self.conv = conv_class(
            in_channels = in_channels,
            out_channels = out_channels,
            kernel_size = kernel_size,
            stride=stride,
            dilation = dilation,
            groups = groups,
            bias = bias,
            device = device,
            dtype = dtype
        )
    
    def fromIntToList(self, p):
        if type(p)==int:
            p=[p,p]
        return p
    
    def forward(self, x):
        x = self.pad(x)
        x = self.conv(x)
        return x
    
