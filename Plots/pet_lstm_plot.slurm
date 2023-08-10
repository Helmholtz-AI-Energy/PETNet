#!/bin/bash
#
#SBATCH --job-name=PLOT_LSTM
#
#SBATCH --time=10:00
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:1g.5gb:1
#SBATCH --output=Outfiles/pet_lstm_plot.out

python3 -u ../plot_lstm.py --datapath /hkfs/work/workspace_haic/scratch/kj3268-PetNet/data/petsim --modelpath ../LSTM/models/checkpoint_petsim_lstm_BCELoss.pth.tar --savedir checkpoint_petsim_lstm_BCELoss
