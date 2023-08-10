#!/bin/bash
#
#SBATCH --job-name=SNN_CHPR
#
#SBATCH --time=8:00:00
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:2g.10gb:1
#SBATCH --output=Outfiles/pet_snn_train_chamferloss_pretty.out

python3 -u ../train_snn.py --epoch 30 --datapath /hkfs/work/workspace_haic/scratch/kj3268-PetNet/data/petsim --modelname models/petsim_chamferloss_pretty.pth.tar --loss mse_count_chamfer_loss_pretty
