from ..chain import FXChain
from .equaliser import FixedPeak, FixedLowShelf, FixedHighShelf, Peak, HighShelf, LowShelf
import numpy as np

def get_GraphicEQ(samplerate=44100, B=2):
    GraphicEQ=FXChain()
    f0=31.25
    om0 = f0*2*np.pi/samplerate
    Q = 1/(2*np.sinh(np.log(2)*B*om0/(2*np.sin(om0))))
    GraphicEQ.append_FX(FixedLowShelf(f0=f0, Q=Q, samplerate=samplerate))

    EQ_freq = [62.5, 125, 250, 500, 1000, 2000, 4000, 8000]
    for i,f0 in enumerate(EQ_freq):
        om0 = f0*2*np.pi/samplerate
        Q = 1/(2*np.sinh(np.log(2)*B*om0/(2*np.sin(om0))))
        band = FixedPeak(f0 = f0, Q=Q, samplerate = samplerate)
        GraphicEQ.append_FX(band)

    f0=16000
    om0 = f0*2*np.pi/samplerate
    Q = 1/(2*np.sinh(np.log(2)*B*om0/(2*np.sin(om0))))
    GraphicEQ.append_FX(FixedHighShelf(f0=f0, Q=Q, samplerate=samplerate))
    return GraphicEQ



#GraphicEQ_Third=FXChain()
#
#GraphicEQ_Third.append_FX(FixedLowShelf(f0=25, Q=2.145,samplerate=44100))
#
#EQ_freq = [
#    31.5, 40,
#    50, 63, 80,
#    100, 125, 160,
#    200, 250, 315,
#    400, 500, 630,
#    800, 1000, 1250,
#    1600, 2000, 2500,
#    3150, 4000, 5000,
#    6300, 8000, 10000,
#    12500, 16000]
#
#for i,f0 in enumerate(EQ_freq):
#    band = FixedPeak(f0 = f0, Q=2.145, samplerate = 44100)
#    GraphicEQ_Third.append_FX(band)
#
#GraphicEQ_Third.append_FX(FixedHighShelf(f0=20000, Q=2.145, samplerate=44100))