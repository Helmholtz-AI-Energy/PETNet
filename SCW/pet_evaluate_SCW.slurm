#!/bin/bash
#
#SBATCH --job-name=SCW_EVAL
#
#SBATCH --time=8:00:00
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:2g.10gb:1
#SBATCH --output=Outfiles/pet_evaluate_SCW.out

python3 -u ../evaluate_SCW.py --datapath /hkfs/work/workspace_haic/scratch/kj3268-PetNet/data/petsim
