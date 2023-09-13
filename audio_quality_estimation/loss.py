import torch

def l2norm(input, target=0):
    return torch.sum(torch.square(input-target), dim=-1)

def SI_SDR(input, target):
    batch_size, num_channels, num_samples = input.size()

    coeff = torch.sum(input*target, dim=-1)/l2norm(target)
    coeff = coeff.reshape(batch_size, num_channels, 1)
    out = l2norm(coeff*target)/l2norm(coeff*target,input)
    out=10*torch.log10(out)
    return -out.mean()