#!/bin/bash
#SBATCH --job-name train_unet
#SBATCH --nodes 1
#SBATCH --tasks-per-node 1
#SBATCH --cpus-per-task 2
#SBATCH --gpus-per-node a100:1
#SBATCH --mem 32gb
#SBATCH --time 6:00:00
#SBATCH --constraint interconnect_hdr
source activate universal
cd /home/aniemcz/CAVSSegMini
python train.py
