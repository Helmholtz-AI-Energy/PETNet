#!/bin/bash
#
#SBATCH --job-name=SNN_MSET
#
#SBATCH --time=8:00:00
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:2g.10gb:1
#SBATCH --output=SNN/Outfiles/pet_snn_train_mse_count_timing_loss.out

python3 -u train_snn.py --epoch 30 --datapath /hkfs/work/workspace_haic/scratch/kj3268-PetNet/data/petsim --modelname petsim_mse_count_timing_loss.pth.tar --loss mse_count_timing_loss
