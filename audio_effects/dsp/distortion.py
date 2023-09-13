import numpy as np
from numba import jit
import numpy
from.dsp import NPFXBase
import torch

class TanhClipper(NPFXBase):
    def __init__(
        self,
        samplerate:float
    ):
        super(NPFXBase, self).__init__()
        self.samplerate = samplerate
        self.controls_names = ['drive_dB', 'offset']
        self.controls_ranges = torch.Tensor([
            [-20, 20], [0, 1]
        ])
        self.num_controls = 2
    
    def effect(
        self,
        x,
        drive_dB,
        offset,
        samplerate=44100
    ):
        drive = np.power(10,drive_dB/20)
        return (np.tanh(x*drive+offset)-np.tanh(offset))/drive

def tanh_distortion(
        x,
        drive_dB,
        offset,
        samplerate=44100
):
    drive = np.power(10,drive_dB/20)
    return (np.tanh(x*drive+offset)-np.tanh(offset))/drive



def hard_clipper(
    x,
    drive_dB,
    offset,
    samplerate=44100
):
    def hc(x):
        return ((np.abs(x+1)-np.abs(x-1))/2)
    drive = np.power(10,drive_dB/20)
    return (hc(x*drive+offset)-hc(offset))/drive

def sin_clipper(
        x,
        drive_dB,
        offset,
        samplerate=44100
):
    def sc(x_):
        halfpi = np.pi*0.5
        sup = np.where(x_>halfpi)
        inf = np.where(x_<-halfpi)
        y = np.sin(x_)
        y[sup]=1
        y[inf]=-1
        return y
    drive = np.power(10,drive_dB/20)
    return (sc(x*drive+offset)-sc(np.array([offset]))[0])/drive

def sin_wavefolder(
        x,
        drive_dB,
        offset,
        samplerate=44100
):
    drive = np.power(10,drive_dB/20)
    return (np.sin(x*drive+offset)-np.sin(offset))/drive

def cubic_clipper(
        x,
        drive_dB,
        offset,
        samplerate=44100
):
    def cc(x_):
        #x_=x_/1.5
        sup = np.where(x_>1.5)
        inf = np.where(x_<-1.5)
        y = x_-np.power(x_, 3)*4/27
        y[sup]=1
        y[inf]=-1
        return y
    drive = np.power(10, drive_dB/20)
    return  (cc(x*drive+offset)-cc(np.array([offset]))[0])/drive

def random_dist(
        x,
        drive_dB,
        offset,
        dist_choice,
        samplerate=44100,
):
    if dist_choice<1:
        out = (1-dist_choice)*tanh_distortion(x, drive_dB, offset, samplerate)+ dist_choice*cubic_clipper(x, drive_dB, offset, samplerate)
    else:
        out = (2-dist_choice)*cubic_clipper(x, drive_dB, offset, samplerate) + (dist_choice-1)*hard_clipper(x, drive_dB, offset, samplerate)
    return out
