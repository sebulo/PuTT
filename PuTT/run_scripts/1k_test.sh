#!/bin/bash

#SBATCH --job-name=simple-gpu    # Job name
#SBATCH --output=output_files/job.%j.out      # Name of output file (%j expands to jobId)
#SBATCH --cpus-per-task=4        # Schedule 8 cores (includes hyperthreading)
#SBATCH --gres=gpu               # Schedule a GPU, or more with gpu:2 etc
#SBATCH --time=00:10:00          # Run time (hh:mm:ss) - run for one hour max
#SBATCH --partition=red,brown    # Run on either the Red or Brown queue

echo "Running on $(hostname):"
nvidia-smi

eval "$(conda shell.bash hook)"
conda activate quimb


python3 train.py --config configs/girl1k_QTT.yaml 
