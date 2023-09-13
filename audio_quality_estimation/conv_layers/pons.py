import torch


class MaxPooled_Conv2d(torch.nn.Module):
    def __init__(
        self,
        n_bins,
        in_channels,
        out_channels,
        kernel_size,
        stride=[1,1],
        dilation=[1,1],
        pool_size = 4,
        groups=1,
        bias=True,
        device = None,
        dtype= None,
        pad_value:float = 0,
        conv_class = torch.nn.Conv2d
    ):
        super(MaxPooled_Conv2d, self).__init__()
        kernel_size = self.fromIntToList(kernel_size)
        dilation = self.fromIntToList(dilation)
        
        self.pad = torch.nn.ConstantPad2d(
            padding = (
                (kernel_size[1]//2)*dilation[1],
                (kernel_size[1]-kernel_size[1]//2-1)*dilation[1],
                0,
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
        self.pool = torch.nn.MaxPool2d(
            kernel_size = [n_bins-kernel_size[0]+1, pool_size]
        )
    
    def fromIntToList(self, p):
        if type(p)==int:
            p=[p,p]
        return p
    
    def forward(self, x):
        x = self.pad(x)
        x = self.conv(x)
        x = self.pool(x)
        return x


class MeanPooled_Conv2d(torch.nn.Module):
    def __init__(
        self,
        n_bins,
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
        super(MeanPooled_Conv2d, self).__init__()
        kernel_size = self.fromIntToList(kernel_size)
        dilation = self.fromIntToList(dilation)
        
        self.pad = torch.nn.ConstantPad2d(
            padding = (
                (kernel_size[1]//2)*dilation[1],
                (kernel_size[1]-kernel_size[1]//2-1)*dilation[1],
                0,
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
        x = x.mean(2)
        return x
    
class ParallelTimbre_Conv2d(torch.nn.Module):
    def __init__(
            self,
            kernels_list:list,
            in_channels:int,
            channels_list:list,
            n_bins:int,
            conv_class = torch.nn.Conv2d,
            device=None,
            eps:float=0,
            pool_size = 4,
            pad_value=0
    ):
        super(ParallelTimbre_Conv2d, self).__init__()

        self.eps=eps

        self.convs = torch.nn.ModuleList()

        self.channels_list = channels_list
        self.out_dim=sum(channels_list)
        self.pad_value = pad_value
        self.pool_size = pool_size

        for i in range(len(channels_list)):
            self.convs.append(MaxPooled_Conv2d(
                n_bins=n_bins,
                in_channels=in_channels,
                out_channels=channels_list[i],
                kernel_size=kernels_list[i],
                conv_class=conv_class,
                pad_value=pad_value,
                pool_size=pool_size
            ))
        
    
    def forward(self, x):
        batch_size, in_channels, n_bins, n_t = x.size()
        y = torch.zeros((batch_size, self.out_dim, 1, int(n_t//self.pool_size)), device= x.device)

        channel_idx = 0
        for i, conv in enumerate(self.convs):
            y[:, channel_idx:channel_idx+self.channels_list[i],:,:] = conv(x)
        
        return y


class ParallelTimbre_Mean_Conv2d(torch.nn.Module):
    def __init__(
            self,
            kernels_list:list,
            in_channels:int,
            channels_list:list,
            n_bins:int,
            conv_class = torch.nn.Conv2d,
            device=None,
            eps:float=0,
            pad_value=0
    ):
        super(ParallelTimbre_Conv2d, self).__init__()

        self.eps=eps

        self.convs = torch.nn.ModuleList()

        self.channels_list = channels_list
        self.out_dim=sum(channels_list)

        for i in range(len(channels_list)):
            self.convs.append(MeanPooled_Conv2d(
                n_bins=n_bins,
                in_channels=in_channels,
                out_channels=channels_list[i],
                kernel_size=kernels_list[i],
                conv_class=conv_class,
                pad_value=pad_value
            ))
        
    
    def forward(self, x):
        batch_size, in_channels, n_bins, n_t = x.size()
        y = torch.zeros((batch_size, self.out_dim, 1,n_t), device= x.device)

        channel_idx = 0
        for i, conv in enumerate(self.convs):
            y[:, channel_idx:channel_idx+self.channels_list[i],:,:] = conv(x)
        
        return y