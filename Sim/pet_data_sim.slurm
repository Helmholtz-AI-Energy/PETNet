#!/bin/bash
#
#SBATCH --job-name=PET_SIM
#
#SBATCH --time=8:00:00
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:2g.10gb:1
#SBATCH --output=Outfiles/pet_data_sim.out

python3 -u ../simulatePET.py --savedir /hkfs/work/workspace_haic/scratch/kj3268-PetNet/data --measurementTime 1000
