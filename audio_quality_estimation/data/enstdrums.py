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

class ENSTDrums_Extracts(Dataset):
    def __init__(
            self,
            effects_list:list,
            controls_ranges_dicts_list:list,
            effects_aug:list=[],
            controls_ranges_aug:list=[],
            root_dir:str = '/tsi/data_doctorants/peladeau/data/ENST-drums_extracts',
            controls_ranges_path:str='configs/synthesis_controls_ranges.yaml',
            random_polarity:bool = True,
            songs_list_path:str = 'none',
            training = True,
            len_dataset:int = None,
    ):
        super(ENSTDrums_Extracts, self).__init__()
        self.root_dir = root_dir
        self.effects_list = effects_list
        self.controls_ranges_dicts_list = controls_ranges_dicts_list
        self.effects_aug = effects_aug
        self.controls_ranges_aug = controls_ranges_aug
        self.random_polarity=random_polarity
        self.training=training

        self.drummers_names = [f'drummer_{i}' for i in range(1, 4)]
        self.microphones_names = [
            'hi-hat',
            'kick',
            'overhead_L',
            'overhead_R',
            'snare',
            'tom_1',
            'tom_2',
        ]

        self.audio_files_list = os.listdir(root_dir)

        self.all_controls = []
        if training == False:
            for effect_idx in range(len(effects_list)):
                controls_ranges_dict = controls_ranges_dicts_list[effect_idx]
                num_controls= len(controls_ranges_dict)
                sampler = qmc.LatinHypercube(d=num_controls)
                latin_params = sampler.random(len(self.audio_files_list))

                controls_names = list(controls_ranges_dicts_list[effect_idx].keys())

                

                effect_controls=[]
                for song_idx in range(len(self.audio_files_list)):
                    curr_dict = {}
                    for control_idx, control_name in enumerate(controls_names):
                        p=latin_params[song_idx][control_idx]
                        curr_dict[control_name] = controls_ranges_dict[control_name][0] + p * (controls_ranges_dict[control_name][1] -controls_ranges_dict[control_name][0])
                    effect_controls.append(curr_dict)
                self.all_controls.append(effect_controls)
        
        if len_dataset==None:
            self.len_dataset = len(self.audio_files_list)
        else:
            self.len_dataset = len_dataset


    def __len__(self):
        return self.len_dataset
    
    def __getitem__(self, idx:int):
        audio_path = os.path.join(self.root_dir, self.audio_files_list[idx])
        dry_waveform, samplerate = torchaudio.load(audio_path)
        dry_waveform = dry_waveform.numpy()
        dry_waveform = dry_waveform/np.amax(np.abs(dry_waveform))

        if self.random_polarity:
            polarity = np.sign(np.random.randn(1)[0])
            dry_waveform = dry_waveform*polarity
        
        aug_waveform = np.zeros(dry_waveform.shape[1])
        aug_waveform[:] = dry_waveform[0,:]

        for effect_idx, effect in enumerate(self.effects_aug):
            controls = get_effect_controls(self.controls_ranges_aug[effect_idx])
            aug_waveform = effect(x=aug_waveform, **controls, samplerate = samplerate)
            aug_waveform = aug_waveform-np.mean(aug_waveform)
            aug_waveform = aug_waveform/np.amax(np.abs(aug_waveform))

        wet_waveform = np.zeros(dry_waveform.shape[1])
        wet_waveform[:] = aug_waveform[:]

        for effect_idx, effect in enumerate(self.effects_list):
            if self.training == False:
                controls  = self.all_controls[effect_idx][idx]
            else:
                controls = get_effect_controls(self.controls_ranges_dicts_list[effect_idx])
            
            wet_waveform = effect(x=wet_waveform, **controls, samplerate = samplerate)
            wet_waveform = wet_waveform-np.mean(wet_waveform)
            wet_waveform = wet_waveform/np.amax(np.abs(wet_waveform))
        

        aug_waveform = torch.Tensor(aug_waveform).reshape(1, -1)
        wet_waveform = torch.Tensor(wet_waveform).reshape(1, -1)

        return aug_waveform, wet_waveform
