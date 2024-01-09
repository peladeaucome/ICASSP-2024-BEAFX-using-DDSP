#!/bin/bash

#SBATCH --time=21:00:00
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --partition=P100

#SBATCH -o FX_peq_ENC_time_frequency_LOSS_mel.out

set -x

loss=mel
encoder=time_frequency
afx=peq
sfx=peq

python train_analysis_net.py --encoder=$encoder --afx=$afx --sfx=$sfx --loss=$loss --prefix=$prefix --suffix=$suffix