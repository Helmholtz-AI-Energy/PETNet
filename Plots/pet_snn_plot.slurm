#!/bin/bash
#
#SBATCH --job-name=PLOT_SNN
#
#SBATCH --time=10:00
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:1g.5gb:1
#SBATCH --output=Outfiles/pet_snn_plot.out

python3 -u ../plot_snn.py --datapath /hkfs/work/workspace_haic/scratch/kj3268-PetNet/data/petsim --modelpath ../SNN/models/checkpoint_petsim_deeper.pth.tar --savedir petsim_snn_deeper
