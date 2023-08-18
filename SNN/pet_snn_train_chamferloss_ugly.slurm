#!/bin/bash
#
#SBATCH --job-name=SNN_CHUG
#
#SBATCH --time=8:00:00
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:2g.10gb:1
#SBATCH --output=SNN/Outfiles/pet_snn_train_chamferloss_ugly.out

python3 -u train_snn.py --epoch 30 --datapath /hkfs/work/workspace_haic/scratch/kj3268-PetNet/data/petsim --modelname petsim_chamferloss_ugly.pth.tar --loss mse_count_chamfer_loss_ugly
