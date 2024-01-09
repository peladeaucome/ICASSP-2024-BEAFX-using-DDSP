#!/bin/bash

#SBATCH --time=01:30:00
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --mem=16G

#SBATCH --output=./results/compressor/standard_Rf300_channels8_causal_nonSkip_fullLoss_expMU.out
#SBATCH --error=./results/compressor/standard_Rf300_channels8_causal_nonSkip_fullLoss_expMU.err

set -x

experience_name=compressor/standard_Rf300_channels8_causal_nonSkip_fullLoss_expMU
config_path=configs/compressor/explicit_params/standard_Rf300_channels8_causal_nonSkip_fullLoss_expMU.yaml

python proxy_compressor_train.py --config_path=$config_path --experience_name=$experience_name 
python proxy_compressor_eval.py --config_path=$config_path --experience_name=$experience_name 

