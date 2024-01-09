#!/bin/bash

#SBATCH --time=16:00:00
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=V100

#SBATCH -e ./results/AE_AFX/AE_base_PEQ7_SC_PEQ5_DRC.out
#SBATCH -o ./results/AE_AFX/AE_base_PEQ7_SC_PEQ5_DRC.out

set -x

python AE_AFX.py --no-progress_bar --model=AE --fx=PEQ7_SC_PEQ5_DRC