#!/bin/bash

#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=P100

#SBATCH -e ./results/AE_AFX/Constant.out
#SBATCH -o ./results/AE_AFX/Constant.out

set -x

python single_song_AFX.py --no-verbose