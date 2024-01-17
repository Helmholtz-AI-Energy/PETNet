#!/bin/bash

#SBATCH --job-name=PETNet
#SBATCH --nodes=2
#SBATCH --gres=gpu:4 # number of requested GPUs (GPU nodes shared btwn multiple jobs)
#SBATCH --ntasks-per-gpu=1
#SBATCH --time=12:00:00 # wall-clock time limit
#SBATCH --mem=128000
#SBATCH --output=PETNet_Train.out

module purge                                        # Unload currently loaded modules.
#module load mpi/openmpi/4.1
#module load module load devel/cuda/11.8

#source ../venv/petnet/bin/activate      # Activate your virtual environment.

unset SLURM_NTASKS_PER_TRES
LOSSFUNC="mse_count_timing_loss"
DATA="data/TimeSteps2000/Standard/petsim"

srun python -u train_ddp.py --epoch 30 --batch 128 --lr 5e-4 --earlystopping 2 --datapath $DATA  --modeltype PETNet --modelname PETNet.pth.tar --loss $LOSS