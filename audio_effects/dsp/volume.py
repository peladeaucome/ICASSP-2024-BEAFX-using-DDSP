import numpy as np

def volume(x, g_dB, samplerate = 44100):
    g = np.power(10, g_dB/20)
    x = x*g
    return x