#!/bin/bash

#SBATCH --job-name=snn_ddp
#SBATCH --exclusive
#SBATCH --nodes=1
#SBATCH --gres=gpu:1 # number of requested GPUs (GPU nodes shared btwn multiple jobs)
#SBATCH --ntasks-per-gpu=1
#SBATCH --time=24:00:00 # wall-clock time limit
#SBATCH --mem=128000
#SBATCH --output=PETNet_evaluate.out

#module load mpi/openmpi/4.1
#module load module load devel/cuda/11.8

#source ../venv/petnet/bin/activate      # Activate your virtual environment.

unset SLURM_NTASKS_PER_TRES

# Define a string variable with a value
MODELS=$(find SNN/models -name "model_best*")

# Iterate the string variable using for loop
for MODEL in $MODELS; do
    if [[ "$MODEL" == *"SAFIR"* ]]; then
        if [[ "$MODEL" == *"geo"* ]]; then
          srun python -u eval_model.py --model $MODEL  --datapath /home/hk-project-scc/ih5525/PETNet/Clone/PETNet/data/NewSim/SAFIR/petsim_plusgeometry --innodes 5760 --hidden 4416 --outnodes 2880 --timesteps 2000 --samplestart 50000 --nsamples 10000 --batch=8
        else
          srun python -u eval_model.py --model $MODEL  --datapath /home/hk-project-scc/ih5525/PETNet/Clone/PETNet/data/NewSim/SAFIR/petsim --innodes 2880 --hidden 4416 --outnodes 2880 --timesteps 2000 --samplestart 50000 --nsamples 10000 --batch=8
        fi
    else
        if [[ "$MODEL" == *"geo"* ]]; then
          srun python -u eval_model.py --model $MODEL  --datapath /home/hk-project-scc/ih5525/PETNet/Clone/PETNet/data/NewSim/Standard/petsim_plusgeometry --innodes 480 --hidden 368 --outnodes 240 --timesteps 2000 --samplestart 50000 --nsamples 10000 --batch=64
        else
          srun python -u eval_model.py --model $MODEL  --datapath /home/hk-project-scc/ih5525/PETNet/Clone/PETNet/data/NewSim/Standard/petsim --innodes 240 --hidden 368 --outnodes 240 --timesteps 2000 --samplestart 50000 --nsamples 10000 --batch=64
        fi
        
    fi
    
done