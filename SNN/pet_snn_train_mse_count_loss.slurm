#!/bin/bash
#
#SBATCH --job-name=SNN_MSE
#
#SBATCH --time=8:00:00
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:2g.10gb:1
#SBATCH --output=Outfiles/pet_snn_train_mse_count_loss.out

python3 -u ../train_snn.py --epoch 30 --datapath /hkfs/work/workspace_haic/scratch/kj3268-PetNet/data/petsim --modelname models/petsim_mse_count_loss.pth.tar --loss mse_count_loss
