import numpy as np
import numpy.typing as npt
from numba import jit
from .dsp import NPFXBase
import torch



class Compressor(NPFXBase):
    def __init__(
        self,
        samplerate:int = 44100
        ):
        super(NPFXBase, self).__init__()

        self.samplerate = samplerate

        self.controls_names = [
            "threshold_dB",
            "attack_time",
            "release_time",
            "ratio",
            "knee_dB",
            "makeup_gain_dB"]

        self.controls_ranges = torch.Tensor([[-40, 0], [0.1, 60], [10, 500], [2, 10], [0, 12], [0, 20]])
        self.num_controls = 6
        self.num_channels = 1
    
    def set_controls_to_range(self, q):
        ranges = self.controls_ranges.clone().to(q.device)
        ranges[1:4] = torch.log(ranges[1:4])

        m = ranges[:,0].reshape(1, -1)
        M = ranges[:,1].reshape(1, -1)
        p = q*(M-m) + m
        p[:,1:4] = torch.exp(p[:,1:4])
        return p
    
    def effect(
            self,
            x:np,
            attack_time:float,
            release_time:float,
            ratio:float,
            threshold_dB:float,
            knee_dB:float=0,
            makeup_gain_dB:float = 0.,
    ):
        out = self.compress(
            x=x,
            attack_time=attack_time,
            release_time=release_time,
            ratio = ratio,
            threshold_dB=threshold_dB,
            knee_dB=knee_dB,
            makeup_gain_dB=makeup_gain_dB,
            samplerate=self.samplerate)
        return out

    @staticmethod
    @jit(nopython=True)
    def compress(
            x:npt.ArrayLike,
            attack_time:float,
            release_time:float,
            ratio:float,
            threshold_dB:float,
            knee_dB:float=0,
            makeup_gain_dB:float = 0.,
            samplerate:int = 44100,
    ):

        x_G = x

        idx = np.where(np.abs(x_G)<0.0001)
        x_G[idx] = .0001

        x_G = 20*np.log10(np.abs(x_G))



        num_samples = len(x)

        alpha_attack = np.exp(-1/(attack_time*.001*samplerate))
        alpha_release = np.exp(-1/(release_time*.001*samplerate))

        y_G = np.zeros(np.shape(x_G))

        where_result = np.where(2*(x_G-threshold_dB) < -knee_dB)
        y_G[where_result] = x_G[where_result]

        #y_G = x_G #Skipping the first_condition to gain time

        if knee_dB>0:
            where_result = np.where(2*np.abs(x_G - threshold_dB)<=knee_dB)
            y_G[where_result] = x_G[where_result] + (1/ratio-1)*np.square(x_G[where_result] - threshold_dB + knee_dB/2)/(2*knee_dB) #Middle condition

        where_result = np.where(2*(x_G-threshold_dB) > knee_dB)
        y_G[where_result] = threshold_dB + (x_G[where_result]-threshold_dB)/ratio

        x_L = x_G - y_G

        y_L = np.zeros(num_samples)
        
        y_L[0] = 0

        for n in range(1,num_samples):
            if x_L[n]> y_L[n-1]:
                y_L[n] = alpha_attack*y_L[n-1] + (1-alpha_attack)*x_L[n]
            else:
                y_L[n] = alpha_release*y_L[n-1] + (1-alpha_release)*x_L[n]
        
        c = np.power(10, (makeup_gain_dB - y_L)*0.05)
        return x*c