import torch
import torchaudio
import numpy as np

def biquad_filt(
        x,
        a_coeffs,
        b_coeffs,
        win_length = 4461
):
    if a_coeffs.size()[1]!=3 or b_coeffs.size()[1]!=3:
        raise ValueError('wrong filter length, please use a biquad filter')

    batch_size, num_channels, num_samples = x.size()
    if num_channels !=1:
        raise ValueError('channel_width sould be 1')

    n = torch.arange(win_length, device = x.device)

    h_a = torch.zeros((batch_size, win_length), device=  x.device)

    Delta = a_coeffs[:,1]**2 - 4*a_coeffs[:,0]*a_coeffs[:,2]

    for batch_idx in range(batch_size):
        if Delta[batch_idx] >0:
            r1 = (-a_coeffs[batch_idx,1] - torch.sqrt(Delta[batch_idx]))/(2*a_coeffs[batch_idx,0])
            r2 = (-a_coeffs[batch_idx,1] + torch.sqrt(Delta[batch_idx]))/(2*a_coeffs[batch_idx,0])

            alpha_1 = r1/(a_coeffs[batch_idx,0]*(r1-r2))
            alpha_2 = r2/(a_coeffs[batch_idx,0]*(r2-r1))

            h_a[batch_idx,:] = alpha_1 * torch.pow(r1, n) + alpha_2*torch.pow(r2, n)
        elif Delta[batch_idx] <0:
            r1 = (-a_coeffs[batch_idx, 1] - 1j*torch.sqrt(-Delta[batch_idx]))/(2*a_coeffs[batch_idx, 0])
            r2 = (-a_coeffs[batch_idx, 1] + 1j*torch.sqrt(-Delta[batch_idx]))/(2*a_coeffs[batch_idx, 0])

            alpha_1 = r1/(a_coeffs[batch_idx, 0]*(r1-r2))
            alpha_2 = r2/(a_coeffs[batch_idx, 0]*(r2-r1))
            h_a[batch_idx,:] = torch.real(alpha_1 * torch.pow(r1, n) + alpha_2*torch.pow(r2, n))
        elif Delta[batch_idx] ==0:
            r0 = -a_coeffs[batch_idx, 1]/(2*a_coeffs[batch_idx, 0])
            alpha = 1/a_coeffs[batch_idx, 0]
            h_a[batch_idx,:] = (alpha + alpha*n) * torch.pow(r0, n)
        
    h = b_coeffs[:, 0].reshape(batch_size, 1)*h_a
    h[:, 1:] = h[:, 1:] + b_coeffs[:, 1].reshape(batch_size, 1)*h_a[:, :-1]
    h[:, 2:] = h[:, 2:] + b_coeffs[:, 2].reshape(batch_size, 1)*h_a[:, :-2]
    
    h = h.reshape(batch_size, 1, win_length)

    x=x.reshape(1,num_channels*batch_size,num_samples)
    x = torch.nn.functional.pad(
        input = x,
        pad = (0,win_length-1)
    )

    y = torch.nn.functional.conv1d(
        input = x,
        weight=h,
        bias = None,
        groups = num_channels*batch_size
    ).reshape(batch_size, num_channels ,num_samples)

    return y

def biquad_filt_2(
        x,
        a_coeffs,
        b_coeffs,
        win_length = 512
):
    if a_coeffs.size()[1]!=3 or b_coeffs.size()[1]!=3:
        raise ValueError('wrong filter length, please use a biquad filter')

    batch_size, num_channels, num_samples = x.size()
    if num_channels !=1:
        raise ValueError('channel_width sould be 1')

    n = torch.arange(win_length, device = x.device).reshape(1, win_length)*torch.ones((batch_size, win_length), device= x.device)

    h_a = torch.zeros((batch_size, win_length), device=  x.device)

    Delta = a_coeffs[:,1]**2 - 4*a_coeffs[:,0]*a_coeffs[:,2]

    one_or_j = torch.where(
        condition = Delta>=0,
        input=torch.ones_like(Delta),
        other=1j*torch.ones_like(Delta)
    )
    
    r1 = (-a_coeffs[:,1] - one_or_j * torch.sqrt(torch.abs(Delta)))/(2*a_coeffs[:,0])
    r2 = (-a_coeffs[:,1] + one_or_j * torch.sqrt(torch.abs(Delta)))/(2*a_coeffs[:,0])

    alpha_1 = r1/(a_coeffs[:,0]*(r1-r2))
    alpha_2 = r2/(a_coeffs[:,0]*(r2-r1))

    r1 = r1.reshape(batch_size, 1)
    r2 = r2.reshape(batch_size, 1)
    alpha_1 = alpha_1.reshape(batch_size, 1)
    alpha_2 = alpha_2.reshape(batch_size, 1)

    h_a = torch.real(alpha_1 * torch.pow(r1, n) + alpha_2*torch.pow(r2, n))
        
    h = b_coeffs[:, 0].reshape(batch_size, 1)*h_a
    h[:, 1:] = h[:, 1:] + b_coeffs[:, 1].reshape(batch_size, 1)*h_a[:, :-1]
    h[:, 2:] = h[:, 2:] + b_coeffs[:, 2].reshape(batch_size, 1)*h_a[:, :-2]
    
    h = h.reshape(batch_size, 1, win_length)

    x=x.reshape(1,num_channels*batch_size,num_samples)
    x = torch.nn.functional.pad(
        input = x,
        pad = (0,win_length-1)
    )

    y = torch.nn.functional.conv1d(
        input = x,
        weight=h,
        bias = None,
        groups = num_channels*batch_size
    ).reshape(batch_size, num_channels ,num_samples)

    return y


def frequencysampling_filt(
        x,
        a_coeffs,
        b_coeffs,
        clamp:bool=False,
        win_length:int = 512
):
    batch_size, num_channels, num_samples = x.size()

    B = torch.fft.rfft(b_coeffs, n=win_length, dim=1)
    A = torch.fft.rfft(a_coeffs, n=win_length, dim=1)

    H = B/A

    h = torch.fft.irfft(H, dim=1)
    h = h.reshape(batch_size*num_channels, 1, win_length)

    x=x.reshape(1,num_channels*batch_size,num_samples)
    x = torch.nn.functional.pad(
        input = x,
        pad = (0,win_length-1)
    )

    y = torch.nn.functional.conv1d(
        input = x,
        weight=h,
        bias = None,
        groups = num_channels*batch_size
    ).reshape(batch_size, num_channels ,num_samples)

    return y


def fft_filt(
        x,
        a_coeffs,
        b_coeffs,
        quality_length=3019,
):
    batch_size, num_channels, N = x.size()
    n_fft = 2**(np.ceil(np.log2(N)))
    if n_fft > N+quality_length:
        n_fft = n_fft*2
    n_fft = int(n_fft)

    A = torch.fft.rfft(a_coeffs, n_fft, dim=1)
    B = torch.fft.rfft(b_coeffs, n_fft, dim=1)

    H = (B/A).reshape(batch_size, num_channels, n_fft//2+1) # Filter FR

    X = torch.fft.rfft(x, n_fft, dim=2)

    Y = X*H
    y = torch.fft.irfft(Y, dim=2)
    y = y[:,:,:N]
    return y