#!/bin/bash

#SBATCH --job-name=PETNet
#SBATCH --nodes=2
#SBATCH --gres=gpu:4 # number of requested GPUs (GPU nodes shared btwn multiple jobs)
#SBATCH --partition=accelerated
#SBATCH --exclusive
#SBATCH --account=hk-project-test-petnetcp
#SBATCH --ntasks-per-gpu=1
#SBATCH --time=12:00:00 # wall-clock time limit
#SBATCH --mem=128000
#SBATCH --output=PETNet_Train.out

module restore PETNet

source ../venv/petnet-ddp/bin/activate      # Activate your virtual environment.

unset SLURM_NTASKS_PER_TRES
LOSSFUNC="mse_count_timing_loss"
DATA="data/TimeSteps1000/petsim"

srun python -u train_ddp.py --epoch 30 --batch 128 --lr 5e-4 --earlystopping 2 --datapath $DATA  --modeltype PETNet --modelname PETNet.pth.tar --loss $LOSS
