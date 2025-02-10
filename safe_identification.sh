#!/bin/bash
#SBATCH -p gpu_long
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:a100:1
#SBATCH --output=log.txt

cd /cl/work10/shuyi-yu/LogitsEditingBasedDefense
source /home/is/shuyi-yu/miniconda3/etc/profile.d/conda.sh #initiate cuda by hand

conda activate logitlens
python safe_layer_identification.py
