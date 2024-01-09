#!/bin/bash

#SBATCH --time=20:00:00
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --mem=16G

#SBATCH --output=./results/compressor/standard_params_fullLoss.out
#SBATCH --error=./results/compressor/standard_params_fullLoss.err

set -x

config_path=configs/compressor/compressor_Rf100.yaml

experience_name=compressor/standard_params_Rf100_C_fullLoss

python proxy_compressor_train.py --config_path=$config_path --experience_name=$experience_name --model_type=standard 
python proxy_compressor_eval.py --config_path=$config_path --experience_name=$experience_name --model_type=standard 

experience_name=compressor/standard_params_Rf100_N_fullLoss

python proxy_compressor_train.py --config_path=$config_path --experience_name=$experience_name --model_type=standard --causal=False
python proxy_compressor_eval.py --config_path=$config_path --experience_name=$experience_name --model_type=standard --causal=False


config_path=configs/compressor/compressor_Rf300.yaml

experience_name=compressor/standard_params_Rf300_C_fullLoss

python proxy_compressor_train.py --config_path=$config_path --experience_name=$experience_name --model_type=standard  
python proxy_compressor_eval.py --config_path=$config_path --experience_name=$experience_name --model_type=standard 

experience_name=compressor/standard_params_Rf300_N_fullLoss

python proxy_compressor_train.py --config_path=$config_path --experience_name=$experience_name --model_type=standard --causal=False
python proxy_compressor_eval.py --config_path=$config_path --experience_name=$experience_name --model_type=standard --causal=False


config_path=configs/compressor/compressor_Rf1000.yaml

experience_name=compressor/standard_params_Rf1000_C_fullLoss

python proxy_compressor_train.py --config_path=$config_path --experience_name=$experience_name --model_type=standard  
python proxy_compressor_eval.py --config_path=$config_path --experience_name=$experience_name --model_type=standard 

experience_name=compressor/standard_params_Rf1000_N_fullLoss

python proxy_compressor_train.py --config_path=$config_path --experience_name=$experience_name --model_type=standard --causal=False
python proxy_compressor_eval.py --config_path=$config_path --experience_name=$experience_name --model_type=standard --causal=False


config_path=configs/compressor/compressor_Rf3000.yaml

experience_name=compressor/standard_params_Rf3000_C_fullLoss

python proxy_compressor_train.py --config_path=$config_path --experience_name=$experience_name --model_type=standard  
python proxy_compressor_eval.py --config_path=$config_path --experience_name=$experience_name --model_type=standard 

experience_name=compressor/standard_params_Rf3000_N_fullLoss

python proxy_compressor_train.py --config_path=$config_path --experience_name=$experience_name --model_type=standard --causal=False
python proxy_compressor_eval.py --config_path=$config_path --experience_name=$experience_name --model_type=standard --causal=False