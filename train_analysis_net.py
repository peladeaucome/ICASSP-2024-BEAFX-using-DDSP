#### IMPORTS

#import numpy as np
import torch, torchaudio

import audio_effects
import audio_quality_estimation
from audio_quality_estimation.conv_layers.base import init_model

import yaml

import lightning.pytorch as pl

from lightning.pytorch import seed_everything
seed_everything(0, workers = True)
#
#
import auraloss
#
#import proxy_models
#
import os
#
import time

from torchinfo import summary

from torch.utils.tensorboard import SummaryWriter

from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--config_path', type = str, default = 'configs/compressor/compressor_Rf3000.yaml')
parser.add_argument('--encoder', type = str, default = 'equaliser')
parser.add_argument('--prefix', type = str, default = '')
parser.add_argument('--suffix', type = str, default = '')
parser.add_argument('--sfx', type = str, default = 'peq')
parser.add_argument('--afx', type = str, default = '')
parser.add_argument('--loss', type = str, default = 'mrstft')

parser.add_argument('--lr', type=float, default=1e-4)
args = parser.parse_args()
config_path = args.config_path
encoder_type=args.encoder.lower()
prefix=args.prefix
suffix = args.suffix
lr = args.lr
sfx = args.sfx.lower()
afx = args.afx.lower()
loss_type = args.loss.lower()
#### PARAMS
encoder_NL = torch.nn.PReLU
encoder_normalization = audio_quality_estimation.normalize.max_norm


encoder = audio_quality_estimation.encoders.get_encoder(
    encoder_type = encoder_type,
    encoder_NL = encoder_NL,
    encoder_normalization = encoder_normalization
)

encoder.apply(init_model)

experience_name = f'quality/{prefix}_SFX_{sfx}_AFX_{afx}_ENC_{encoder_type}_LOSS_{loss_type}_{suffix}'
writer = SummaryWriter(os.path.join('tensorboard',experience_name))
print(experience_name)


root_dir = 'root/dir'

num_epochs = 400
train_audio_length_s = 10
test_audio_length_s = 10
train_audio_length_sp = 44100*train_audio_length_s


train_batch_size = 16
valid_batch_size = train_batch_size
test_batch_size = train_batch_size
num_threads = 5

num_loops_train = 5
num_loops_valid = 5
num_loops_test = 10

plateau_patience=30
early_stopping_patience=150

print(f"Training Loops : {num_loops_train}")
print(f"Validation Loops : {num_loops_valid}")
print(f"Test Loops : {num_loops_test}")

#Loss 

fft_sizes = [256, 1024, 4096]
hop_sizes = [64, 256, 1024]
win_lengths = [256, 1024, 4096]


print(f"Learning rate : {lr}")
print(f"Batch Size : {train_batch_size}")


device  = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

if device !=torch.device("cuda:0"):
    raise ValueError("Please run script on gpu")



with open('configs/analysis_controls_ranges.yaml', 'rb') as f:
    controls_config = yaml.load(f, Loader=yaml.FullLoader)


### Models
AFX_Chain = audio_effects.FXChain(norm = 'max')

if 'geq' in afx:
    GEQ = audio_effects.differentiable.presets.get_GraphicEQ(B=2)
    AFX_Chain.append_FX(GEQ)

if 'peq' in afx:
    band = audio_effects.differentiable.LowShelf(samplerate = 44100)
    band.set_ranges_from_dict(controls_config['lowShelf_controls'])
    AFX_Chain.append_FX(band)

    band = audio_effects.differentiable.Peak(samplerate = 44100)
    band.set_ranges_from_dict(controls_config['peak1_controls'])
    AFX_Chain.append_FX(band)

    band = audio_effects.differentiable.Peak(samplerate = 44100)
    band.set_ranges_from_dict(controls_config['peak2_controls'])
    AFX_Chain.append_FX(band)

    band = audio_effects.differentiable.Peak(samplerate = 44100)
    band.set_ranges_from_dict(controls_config['peak3_controls'])
    AFX_Chain.append_FX(band)

    band = audio_effects.differentiable.HighShelf(samplerate = 44100)
    band.set_ranges_from_dict(controls_config['highShelf_controls'])
    AFX_Chain.append_FX(band)

if 'compfull' in afx:
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    model_config = config['model_config']
    compproxy = audio_effects.neural_proxy.Compressor(**model_config)
    compproxy.load_state_dict(
                torch.load(os.path.join(root_dir, 'trained_models', 'compressor/compressor_Rf3000_C_fullLoss_8_best.pt'))
    )
    compproxy.set_ranges_from_dict(controls_config['compressor_controls'])
    for p in compproxy.parameters():
        p.requires_grad = False
    compproxy = compproxy.to(device)
    compproxy.eval()
    
    full_proxy = audio_effects.neural_proxy.Half_Proxy(
        proxy_model = compproxy,
        dsp_model = compproxy
    )
    AFX_Chain.append_FX(full_proxy)

if 'comphalf' in afx:
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    model_config = config['model_config']
    compproxy = audio_effects.neural_proxy.Compressor(**model_config)
    compproxy.load_state_dict(
                torch.load(os.path.join(root_dir, 'trained_models', 'compressor/compressor_Rf3000_C_fullLoss_8_best.pt'))
    )
    compproxy.set_ranges_from_dict(controls_config['compressor_controls'])
    for p in compproxy.parameters():
        p.requires_grad = False
    compproxy = compproxy.to(device)
    compproxy.eval()

    DSP = audio_effects.dsp.Compressor()
    DSP.set_ranges_from_dict(controls_config['compressor_controls'])

    half_proxy = audio_effects.neural_proxy.Half_Proxy(
        proxy_model = compproxy,
        dsp_model = DSP
    )
    AFX_Chain.append_FX(half_proxy)

if 'compsimple' in afx:
    compsimple = audio_effects.differentiable.SimpleCompressor()
    compsimple.set_ranges_from_dict(controls_config['compressor_simple_controls'])
    AFX_Chain.append_FX(compsimple)


if 'taylor' in afx:
    #harmo = audio_effects.differentiable.HarmonicsGenerator(num_harmonics = 10, range_dB=20, samplerate = 44100)
    taylor = audio_effects.differentiable.TaylorHarmonics(num_harmonics = 24, samplerate = 44100)
    AFX_Chain.append_FX(taylor)

if 'chebyshev' in afx:
    cheby = audio_effects.differentiable.ChebyshevHarmonics(num_harmonics=24, samplerate = 44100)
    AFX_Chain.append_FX(cheby)

if 'dist' in afx:
    dist = audio_effects.differentiable.HardnessDist()
    dist.set_ranges_from_dict(controls_config['distortion_controls'])
    AFX_Chain.append_FX(dist)



print(AFX_Chain.num_controls)
print(AFX_Chain.controls_names)
print(AFX_Chain.controls_ranges)

controller = audio_quality_estimation.Controller_Network(
    audio_fx = AFX_Chain,
    encoder=encoder,
    NL_class= encoder_NL
)

controller = controller.to(device)
summary(controller, input_size = (train_batch_size, 1, train_audio_length_sp), device = device)




#augmentation_fx_list.append(audio_effects.dsp.volume)
#augmentation_ranges_list.append(synthesis_ranges_dict['volume_controls'])
def get_fx_list(fx_list_str, synthesis_ranges_dict_path):
    with open(synthesis_ranges_dict_path, 'rb') as f:
        synthesis_ranges_dict = yaml.load(f, Loader=yaml.FullLoader)
    
    SFX_Chain_ = audio_effects.FXChain(norm = 'max')

    if 'peq' in fx_list_str:
        band = audio_effects.dsp.LowShelf(samplerate = 44100)
        band.set_ranges_from_dict(synthesis_ranges_dict['lowShelf_controls'])
        SFX_Chain_.append_FX(band)

        band = audio_effects.dsp.Peak(samplerate = 44100)
        band.set_ranges_from_dict(synthesis_ranges_dict['peak1_controls'])
        SFX_Chain_.append_FX(band)

        band = audio_effects.dsp.Peak(samplerate = 44100)
        band.set_ranges_from_dict(synthesis_ranges_dict['peak2_controls'])
        SFX_Chain_.append_FX(band)

        band = audio_effects.dsp.Peak(samplerate = 44100)
        band.set_ranges_from_dict(synthesis_ranges_dict['peak3_controls'])
        SFX_Chain_.append_FX(band)

        band = audio_effects.dsp.HighShelf(samplerate = 44100)
        band.set_ranges_from_dict(synthesis_ranges_dict['highShelf_controls'])
        SFX_Chain_.append_FX(band)

    if ('compfull' in fx_list_str) or ('comphalf' in fx_list_str):
        comp = audio_effects.dsp.Compressor()
        comp.set_ranges_from_dict(controls_config['compressor_controls'])
        SFX_Chain_.append_FX(comp)


    if 'dist' in fx_list_str:
        dist = audio_effects.differentiable.HardnessDist()
        dist.set_ranges_from_dict(controls_config['distortion_controls'])
        SFX_Chain_.append_FX(dist)
    
    return SFX_Chain_


synthesis_ranges_dict_path='configs/synthesis_controls_ranges.yaml'
SFX_Chain=get_fx_list(sfx, synthesis_ranges_dict_path)

train_dataset = audio_quality_estimation.data.MUSDB18_Dataset(
    FX_Chain=SFX_Chain,
    subsets = 'train',
    audio_length_s = train_audio_length_s
)
valid_dataset = audio_quality_estimation.data.MUSDB18_Dataset(
    FX_Chain=SFX_Chain,
    subsets = 'valid',
    audio_length_s = train_audio_length_s
)

synthesis_ranges_dict_path='configs/synthesis_controls_ranges.yaml'
SFX_Chain=get_fx_list(sfx, synthesis_ranges_dict_path)

test_dataset = audio_quality_estimation.data.MUSDB18_Dataset(
    FX_Chain=SFX_Chain,
    subsets = 'test',
    audio_length_s = test_audio_length_s
)
#train_dataset.training = True

train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    batch_size=train_batch_size,
    num_workers=num_threads,
    shuffle=True
)
valid_loader = torch.utils.data.DataLoader(
    dataset=valid_dataset,
    batch_size=valid_batch_size,
    num_workers=num_threads,
    shuffle=True
)
test_loader = torch.utils.data.DataLoader(
    dataset=test_dataset,
    batch_size=test_batch_size,
    num_workers=num_threads,
    shuffle=False
)

print(f'Training dataset length : {len(train_dataset)}')
print(f'Validation dataset length : {len(valid_dataset)}')
print(f'Test dataset length : {len(test_dataset)}')


optimizer = torch.optim.Adam(controller.parameters(), lr = lr)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer=optimizer,
    mode='min',
    factor=0.1,
    patience=plateau_patience
)


MRSTFT_loss = auraloss.freq.MultiResolutionSTFTLoss(
    fft_sizes = fft_sizes,
    hop_sizes = hop_sizes,
    win_lengths = win_lengths,
    w_sc = 1,
    device=device
).to(device)
MSE_loss = torch.nn.MSELoss()
Mel_loss = auraloss.freq.MelSTFTLoss(
    sample_rate=44100,
    fft_size = 4096,
    hop_size = 1024,
    win_length = 4096,
    n_mels=128,
    device = device,
    w_sc=0
    ).to(device)


if loss_type=='params':
    def compute_loss(q, q_hat, x, y, AFX):
        y_hat = AFX(x, q_hat)
        return 10*MSE_loss(q_hat, q), y_hat
if loss_type=='mel':
    def compute_loss(q, q_hat, x, y, AFX):
        y_hat = AFX(x, q_hat)

        y=y-torch.mean(y, dim=2).reshape(-1, 1, 1)
        y = audio_quality_estimation.normalize.rms_norm(y)

        y_hat=y_hat-torch.mean(y_hat, dim=2).reshape(-1, 1, 1)
        y_hat = audio_quality_estimation.normalize.rms_norm(y_hat)
        
        return Mel_loss(y_hat, y), y_hat
if loss_type=='mrstft':
    def compute_loss(q, q_hat, x, y, AFX):
        y_hat = AFX(x, q_hat)

        y=y-torch.mean(y, dim=2).reshape(-1, 1, 1)
        y = audio_quality_estimation.normalize.rms_norm(y)

        y_hat=y_hat-torch.mean(y_hat, dim=2).reshape(-1, 1, 1)
        y_hat = audio_quality_estimation.normalize.rms_norm(y_hat)

        return MRSTFT_loss(y_hat, y), y_hat
if loss_type=='mrstft+mel':
    def compute_loss(q, q_hat, x, y, AFX):
        y_hat = AFX(x, q_hat)

        y=y-torch.mean(y, dim=2).reshape(-1, 1, 1)
        y = audio_quality_estimation.normalize.rms_norm(y)

        y_hat=y_hat-torch.mean(y_hat, dim=2).reshape(-1, 1, 1)
        y_hat = audio_quality_estimation.normalize.rms_norm(y_hat)

        return (MRSTFT_loss(y_hat, y) + Mel_loss(y_hat, y))/2, y_hat

#if loss_type=='mrstft':
#    training_loss_fn = MRSTFT_loss
#if loss_type =='mel':
#    training_loss_fn = Mel_loss
#if loss_type == 'mse':
#    training_loss_fn = MSE_loss
#if loss_type == 'mrstft+mel':
#    def training_loss_fn(input, target):
#        return (MRSTFT_loss(input, target) + Mel_loss(input, target))/2, 




def eval_loss_fn(input, target):
    out = torch.zeros(3, device = input.device)
    out[0] = MRSTFT_loss(input, target)
    out[1] = MSE_loss(input, target)
    out[2] = Mel_loss(input, target)
    return out



torch.cuda.synchronize()
tic = time.time()
controller.eval()
AFX_Chain.eval()

with torch.no_grad():
    num_examples = 0
    test_loss = torch.zeros(3, device = device)
    num_examples = 0
    for i in range(num_loops_test):
        for batch_idx, (input, target, q) in enumerate(iter(test_loader)):
            num_examples+=input.size(0)

            batch_size= input.size(0)
            input = input.to(device)
            target = target.to(device)
            
            q_hat = torch.rand((batch_size, AFX_Chain.num_controls), device = device)
            estimate = AFX_Chain(input, q_hat)
            
            #print(torch.min(torch.std(input, dim=2).reshape(-1, 1, 1)))
            #print(torch.min(torch.std(estimate, dim=2).reshape(-1, 1, 1)))
            #print(torch.mean(torch.max(input, dim=2)[0])/torch.mean(torch.std(input, dim=2)))
            estimate = estimate-torch.mean(estimate, dim=2).reshape(-1, 1, 1)
            estimate = audio_quality_estimation.normalize.rms_norm(estimate)

            target = target-torch.mean(target, dim=2).reshape(-1, 1, 1)
            target = audio_quality_estimation.normalize.rms_norm(target)

            loss = eval_loss_fn(estimate, target)
            test_loss+=loss*input.size(0)
    test_loss = test_loss/num_examples
    print(f'Test Loss (Random FX): {test_loss}')
    test_loss = torch.zeros(3, device = device)
    num_examples = 0
    for i in range(num_loops_test):
        for batch_idx, (input, target, q) in enumerate(iter(test_loader)):
            num_examples+=input.size(0)

            batch_size= input.size(0)
            input = input.to(device)
            target = target.to(device)
            estimate = input
            
            #print(torch.min(torch.std(input, dim=2).reshape(-1, 1, 1)))
            #print(torch.min(torch.std(estimate, dim=2).reshape(-1, 1, 1)))
            #print(torch.mean(torch.max(input, dim=2)[0])/torch.mean(torch.std(input, dim=2)))
            estimate = estimate-torch.mean(estimate, dim=2).reshape(-1, 1, 1)
            estimate = audio_quality_estimation.normalize.rms_norm(estimate)

            target = target-torch.mean(target, dim=2).reshape(-1, 1, 1)
            target = audio_quality_estimation.normalize.rms_norm(target)

            loss = eval_loss_fn(estimate, target)
            test_loss+=loss*input.size(0)
    test_loss = test_loss/num_examples
    print(f'Test Loss (Without FX): {test_loss}')

torch.cuda.synchronize()
toc= time.time()
print(f'Starting eval time : {toc-tic}')

print('')

## Train Loop
continue_training=True

def max_norm(x):
    xmax,_ = torch.max(torch.abs(x), dim= 2)
    xmax = xmax.reshape(-1, 1, 1)
    return x/xmax

epochs_since_best=0
training_time = time.time()

for epoch in range(num_epochs):
    epoch_tic = time.time()
    epochs_since_best+=1
    train_loss = torch.zeros(1, device = device)
    num_examples = 0

    controller.train()
    AFX_Chain.train()
    for i in range(num_loops_train):
        for batch_idx, (input, target, q) in enumerate(iter(train_loader)):
            num_examples+=input.size(0)

            input = input.to(device)
            target = target.to(device).requires_grad_(requires_grad=True)
            q=q.to(device)
            
            q_hat = controller(target)
            loss, estimate = compute_loss(
                q=q,
                q_hat=q_hat,
                x=input,
                y=target,
                AFX=AFX_Chain
            )
            train_loss+=loss*input.size(0)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    #torch.cuda.synchronize(device=device)
    #print(f'Training loop timing : {round(toc-tic, 2)}s')

    train_loss = train_loss/num_examples
    #print(f'Training Loss : {train_loss}')
    

    writer.add_scalar('Training Loss', train_loss, global_step = epoch)

    with torch.no_grad():
        valid_loss = torch.zeros(1, device = device)
        num_examples = 0

        controller.eval()
        AFX_Chain.eval()

        for i in range(num_loops_valid):
            for batch_idx, (input, target, q) in enumerate(iter(valid_loader)):
                num_examples+=input.size(0)

                input = input.to(device)
                target = target.to(device)
                q=q.to(device)

                q_hat = controller(target)

                loss, estimate = compute_loss(
                    q=q,
                    q_hat=q_hat,
                    x=input,
                    y=target,
                    AFX=AFX_Chain
                )

                valid_loss+=loss*input.size(0)
        valid_loss = valid_loss/num_examples
        scheduler.step(valid_loss)
        
        #print(f'Validation Loss : {valid_loss}')

        valid_loss = valid_loss[0]

        writer.add_scalar('Validation Loss', valid_loss, global_step = epoch)
        
        target =     max_norm(target)
        estimate = max_norm(estimate)
        input =       max_norm(input)

        for i in range(1):
            writer.add_audio(f"{i} - Input",       input[i], sample_rate = 44100, global_step = epoch)
            writer.add_audio(f"{i} - Target",     target[i], sample_rate = 44100, global_step = epoch)
            writer.add_audio(f"{i} - Estimate", estimate[i], sample_rate = 44100, global_step = epoch)
    
    
    if epoch==0:
        epochs_since_best=0
        best_valid_loss=valid_loss
        torch.save(
            controller.state_dict(),
            os.path.join(root_dir, 'trained_models', f'{experience_name}.pt'))
        print(f'Epoch : {epoch}')
        print(f'Best validation : {valid_loss}')
        print(f'Training Loss : {train_loss[0]}')
    elif valid_loss < best_valid_loss:
        torch.save(
            controller.state_dict(),
            os.path.join(root_dir, 'trained_models', f'{experience_name}.pt'))
        best_valid_loss=valid_loss
        epochs_since_best=0
        print(f'Epoch : {epoch}')
        print(f'Best validation : {valid_loss}')
        print(f'Training Loss : {train_loss[0]}')
    elif (epochs_since_best+1)%plateau_patience==0:
        #controller.load_state_dict(
        #    torch.load(os.path.join(root_dir, 'trained_models', f'{experience_name}.pt'))
        #)
        print(f'Epoch : {epoch}')
        print(f'Decreased learning rate.')
    if (epochs_since_best+1)%early_stopping_patience==0:
        continue_training = False

    
    
    #torch.cuda.synchronize(device= device)
    #epoch_toc = time.time()
    #print(f"Epoch time : {round(epoch_toc - epoch_tic, 2)}s")
    if continue_training==False:
        print(f'Epoch : {epoch}')
        print('Stopped Training')
        break

        
torch.cuda.synchronize(device= device)
training_time = time.time()-training_time
print(f'Training Time : {round(training_time/3600, 2)} hours')
print(f'Mean Time per Epoch : {round(training_time/epoch, 2)}s')


controller.load_state_dict(
    torch.load(os.path.join(root_dir, 'trained_models', f'{experience_name}.pt'))
)
controller.eval()
AFX_Chain.eval()



with torch.no_grad():
    test_loss = torch.zeros(3, device = device)
    params_loss=torch.zeros(1, device=device)
    num_examples = 0

    controller.eval()
    AFX_Chain.eval()

    for i in range(num_loops_test):
        for batch_idx, (input, target, q) in enumerate(iter(test_loader)):
            num_examples+=input.size(0)

            input = input.to(device)
            target = target.to(device)
            q=q.to(device)
            
            q_hat = controller(target)
            if afx==sfx:
                params_loss+=MSE_loss(q_hat, q)*10*input.size(0)
            estimate = AFX_Chain(input, q_hat)

            
            estimate = estimate-torch.mean(estimate, dim=2).reshape(-1, 1, 1)
            estimate = audio_quality_estimation.normalize.rms_norm(estimate)

            target = target-torch.mean(target, dim=2).reshape(-1, 1, 1)
            target = audio_quality_estimation.normalize.rms_norm(target)

            loss = eval_loss_fn(estimate, target)
            test_loss+=loss*input.size(0)
    test_loss = test_loss/num_examples
    params_loss = params_loss/num_examples
    print(f'Test Losses : {test_loss}')
    print(f'MRSTFT : {test_loss[0]}')
    print(f'MSE    : {test_loss[1]}')
    print(f'Mel    : {test_loss[2]}')
    if afx==sfx:
        print(f'\nParameter Loss : {params_loss[0]}')

print('End')