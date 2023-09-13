import torch

def init_model(m):
    if isinstance(m, torch.nn.Conv1d) or isinstance(m, torch.nn.Conv2d):
        torch.nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
        torch.nn.init.constant_(m.bias, 0)
    
def initialize_conv(conv_layer):
    """
    Convolutional Layer initialization taken from [1].

    [1] K. He, X. Zhang, S. Ren, and J. Sun, “Delving Deep into Rectifiers:
    Surpassing Human-Level Performance on ImageNet Classification,”
    in 2015 IEEE International Conference on Computer Vision (ICCV),
    Dec. 2015, pp. 1026–1034. doi: 10.1109/ICCV.2015.123.
    """
    n = conv_layer.in_channels*(conv_layer.kernel_size**2)
    n = conv_layer.in_channels

    if type(conv_layer.kernel_size) == int:
        k=conv_layer.kernel_size
    else:
        k=1
        for i in range(len(conv_layer.kernel_size)):
            k=k*conv_layer.kernel_size[i]
    
    n = conv_layer.in_channels*k

    conv_layer.weight = torch.randn_like(conv_layer.weight)*torch.sqrt(2/n)
    conv_layer.bias= torch.nn.zeros_like(conv_layer.bias)
    return conv_layer