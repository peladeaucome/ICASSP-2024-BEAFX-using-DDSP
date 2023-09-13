import torch


def fun_delay(
        dry_signal,
        delay_time_sec,
        feedback,
        dry_wet,
        sample_rate
):
    num_samples = dry_signal.size(-1)
    delay_time_sp = int(delay_time_sec*sample_rate)

    wet_signal = torch.zeros_like(dry_signal)

    num_iter = num_samples//int(delay_time_sp)+1
    if num_iter>1 and feedback>0:
        for iter_idx in range(num_iter):
            start = iter_idx*delay_time_sp
            add = (feedback**iter_idx)*dry_signal[:num_samples-start]
            wet_signal[start:] = wet_signal[start:] + add
    
    return dry_wet*wet_signal + (1-dry_wet)*dry_signal

class fun_Delay_diff(torch.autograd.Function):
    @staticmethod
    def forward(
        dry_signal:torch.Tensor,
        delay_time_sec:float=1.,
        feedback:float=0,
        dry_wet:float=.5,
        sample_rate:int=44100):

        wet_signal = torch.zeros(
            2*wet_signal.size(0),
            wet_signal.size(1),
            wet_signal.size(2)
        )

        return wet_signal
    
    @staticmethod
    def setup_context(
            ctx
    ):
        return ctx
    
    @staticmethod
    def backward(ctx, grad_out):
        g_dry_signal = None
        g_sample_rate = None

        g_delay_time = grad_out
        g_feedback = grad_out
        g_dry_wet = grad_out
        
        return g_dry_signal, g_delay_time, g_feedback, g_dry_wet, g_sample_rate