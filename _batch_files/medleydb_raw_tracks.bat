#!/bin/bash

#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --gpus=0
#SBATCH --mem=16G

set -x

python misc/construct_list_raw_tracks_MedleyDB.py