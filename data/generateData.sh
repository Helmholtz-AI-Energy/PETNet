#!/bin/bash
#
#SBATCH --job-name=PET_SIM
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=76
#SBATCH --mem=128000
#SBATCH --output=pet_data_simulation.out

#module load mpi/openmpi/4.1
#module load module load devel/cuda/11.8
#source /home/hk-project-scc/ih5525/PETNet/venv/petnet-ddp/bin/activate
python -u simulatePET.py --savedir data/NewSim/ --parameters parameters_SAFIR --measurementTime 2000
