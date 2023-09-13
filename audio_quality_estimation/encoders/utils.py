import torch
from .compression import *
from .distortion import *
from .equaliser import *
from .generic import *
from .pons import *
from .base import *

def get_encoder(
    encoder_type:str,
    encoder_NL = torch.nn.PReLU,
    encoder_normalization=None):
    encoder_type = encoder_type.lower()
    if encoder_type.lower() == 'equaliser':
        encoder = EQNoPad_Encoder(
            NL_class= encoder_NL,
            normalization_function=encoder_normalization)
    elif encoder_type.lower() == 'multitimbre_cqt':
        encoder = MultiTimbreCQT_Encoder(
            NL_class= encoder_NL,
            normalization_function=encoder_normalization)
    elif encoder_type.lower() == 'multitimbre_hcqt':
        encoder = MultiTimbreHCQT_Encoder(
            NL_class= encoder_NL,
            normalization_function=encoder_normalization)
    elif encoder_type.lower() == 'time':
        encoder = TimeCQT_Encoder(
            NL_class= encoder_NL,
            normalization_function=encoder_normalization)
    elif encoder_type.lower() == 'frequency':
        encoder = FrequencyCQT_Encoder(
            NL_class= encoder_NL,
            normalization_function=encoder_normalization)
    elif encoder_type.lower() == 'time_frequency':
        encoder = TimeFrequencyCQT_Encoder(
            NL_class= encoder_NL,
            normalization_function=encoder_normalization)
    elif encoder_type.lower() == 'mee':
        encoder = ResEncoder_1d(
            NL_class= encoder_NL,
            normalization_function=encoder_normalization)
    elif encoder_type.lower() == 'mee_test':
        encoder = ResEncoder_1d_Hilbert(
            NL_class= encoder_NL,
            normalization_function=encoder_normalization)
    elif encoder_type.lower() == 'wavenet':
        encoder=Wavenet_Encoder(
            NL_class=  encoder_NL,
            normalization_function = encoder_normalization
        )
    elif encoder_type.lower() == 'mee_equaliser':
        encoder = MEE_Equaliser_Encoder(
            NL_class=  encoder_NL,
            normalization_function = encoder_normalization
        )
    elif encoder_type.lower() == 'compressor_conv1d':
        encoder = Compression_Encoder(
            NL_class=  encoder_NL,
            normalization_function = encoder_normalization,
            frontend='conv1d'
        )
    elif encoder_type.lower() == 'compressor_gabor':
        encoder = Compression_Encoder(
            NL_class=  encoder_NL,
            normalization_function = encoder_normalization,
            frontend='gabor'
        )
    elif encoder_type.lower() == 'mee+time_frequency':
        enc_1 = TimeFrequencyCQT_Encoder(
            NL_class= encoder_NL,
            normalization_function=encoder_normalization)
        enc_2 = ResEncoder_1d(
            NL_class= encoder_NL,
            normalization_function=encoder_normalization)
        encoder = Parallel_encoders(enc_1, enc_2)
    else:
        raise ValueError(f"{encoder_type} is no a valid encoder type")
    
    return encoder