import numpy as np
import numpy.typing as npt
import scipy
from numba import jit

@jit(nopython=True)
def feedback_delay(
        x:npt.ArrayLike,
        time_sp:float,
        feedback:float) -> npt.ArrayLike:
    N = len(x)
    buffer = np.zeros(time_sp)
    y = np.zeros(N)
    for i in range(len(x)):
        y[i] += buffer[i%time_sp]
        buffer[i%time_sp] = buffer[i%time_sp]*feedback + x[i]
    return y

@jit(nopython=True)
def colorlessReverb(
        x:npt.ArrayLike,
        reverb_time:float,
        drywet:float=1,
        samplerate:float = 44100) -> npt.ArrayLike:

    gamma_list = [-3,-9,-15,-21,-27]
    
    x = np.asarray(x)

    y = x
    for gamma in gamma_list:
        tau = -gamma*reverb_time/60
        time_sp = int(tau*samplerate)
        g = np.power(10, gamma/20)
        y = g*y -  (1-g**2)*feedback_delay(y, time_sp=time_sp, feedback = g)

    return y*drywet+x*(1-drywet)