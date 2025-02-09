#!/bin/bash
#SBATCH -p gpu_short
#SBATCH -t 3:00:00
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:titanx:1
#SBATCH --output=log.txt

cd /home/is/shuyi-yu/SafeDecoding/exp
source /home/is/shuyi-yu/miniconda3/etc/profile.d/conda.sh #initiate cuda by hand

conda activate SafeDecoding
ToxicTokenMask.py
