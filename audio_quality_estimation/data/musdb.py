from torch.utils.data import Dataset
import torch
import torchaudio
import os
import pickle
import numpy as np
import random
import yaml
import scipy.io
from scipy.stats import qmc
import musdb
import librosa


def RMS(x):
    return torch.sqrt(torch.mean(torch.square(x)))

def get_effect_controls(controls_ranges):
    keys = list(controls_ranges)
    out = {}
    for key in keys:
        out[key] = controls_ranges[key][0] + np.random.rand() * (controls_ranges[key][1] -controls_ranges[key][0])
    return out

def get_chain_controls(chain_c_ranges):
    effects = list(chain_c_ranges)
    out = {}
    for effect in effects:
        out[effect] = get_effect_controls(chain_c_ranges[effect])
    return out

class MUSDB18_Dataset(Dataset):
    def __init__(
            self,
            FX_Chain,
            root_dir:str,
            is_wav=False,
            random_polarity:bool = True,
            subsets = 'train',
            samplerate=44100,
            audio_length_s = 10,
            return_effects_params=False
    ):
        super(MUSDB18_Dataset, self).__init__()
        self.root_dir = root_dir
        self.num_controls=FX_Chain.num_controls
        self.FX_Chain=FX_Chain
        self.random_polarity=random_polarity
        self.samplerate=samplerate
        self.audio_length_s = audio_length_s
        self.subsets = subsets
        if subsets == 'train':
            self.mus = musdb.DB(root = self.root_dir, subsets='train', split='train', is_wav = is_wav)
        if subsets == 'valid':
            self.mus = musdb.DB(root = self.root_dir, subsets='train', split='valid', is_wav = is_wav)
        if subsets == 'test':
            self.mus = musdb.DB(root = self.root_dir, subsets='test', is_wav = is_wav)
        
        self.all_controls = []


    def __len__(self):
        return len(self.mus)
    
    def __getitem__(self, idx:int):
        track = self.mus[idx]

        max_val = 0
        while max_val < 1e-5:
            track.chunk_duration = self.audio_length_s
            track.chunk_start = random.uniform(0, track.duration - track.chunk_duration)
            dry_waveform = np.mean(track.audio, axis=1)
            max_val = np.amax(np.abs(dry_waveform))

        if np.amax(np.abs(dry_waveform)) > 1e-5:
            dry_waveform = dry_waveform/np.amax(np.abs(dry_waveform))
        else:
            dry_waveform = torch.Tensor(dry_waveform).reshape(1, -1)
            print('audio not loud enough')
            return dry_waveform, dry_waveform

        if self.random_polarity:
            polarity = np.sign(np.random.randn(1)[0])
            dry_waveform = dry_waveform*polarity
        
        
        with torch.no_grad():
            dry_waveform = torch.Tensor(dry_waveform).reshape(1, 1, -1)
            q = torch.rand(1, self.num_controls)
            wet_waveform = self.FX_Chain(dry_waveform.clone(), q)
        
        dry_waveform = dry_waveform.reshape(1, -1)
        wet_waveform = wet_waveform.reshape(1, -1)
        q=q.reshape(self.num_controls)
        return dry_waveform, wet_waveform, q

