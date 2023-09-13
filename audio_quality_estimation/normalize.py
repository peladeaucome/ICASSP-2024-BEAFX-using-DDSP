import torch

def max_norm(x):
        """Normalizes by the maximum"""
        xmax, _ = torch.max(torch.abs(x), dim=2)
        x=x/(xmax.reshape(-1, 1, 1))
        return x

def rms_norm(x):
    """Normalizes by the standard deviation"""
    x = x/(torch.std(x, dim=2).reshape(-1, 1, 1))
    return x

def no_norm(x):
      """Identity Function"""
      return x