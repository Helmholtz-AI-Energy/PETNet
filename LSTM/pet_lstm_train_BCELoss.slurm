#!/bin/bash
#
#SBATCH --job-name=LSTM_BCE
#
#SBATCH --time=8:00:00
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:4g.20gb:1
#SBATCH --output=Outfiles/pet_lstm_train_BCELoss.out
#SBATCH --mem=64000

python3 -u ../train_lstm.py --epoch 30 --datapath /hkfs/work/workspace_haic/scratch/kj3268-PetNet/data/petsim --modelname models/petsim_lstm_BCELoss.pth.tar --batch 64 --loss BCELoss
