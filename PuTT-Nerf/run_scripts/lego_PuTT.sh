#!/bin/bash
#SBATCH --job-name=simple-gpu    # Job name
#SBATCH --output=output_files/job.%j.out      # Name of output file (%j expands to jobId)
#SBATCH --cpus-per-task=8        # Schedule 8 cores (includes hyperthreading)
#SBATCH --gres=gpu:1              # Schedule a GPU, or more with gpu:2 etc
#SBATCH --time=4:00:00          # Run time (hh:mm:ss) - run for one hour max
#SBATCH --partition=red,brown    # Run on either the Red or Brown queue


#your script, in this case: write the hostname and the ids of the chosen gpus.
hostname
echo $CUDA_VISIBLE_DEVICES

nvidia-smi

echo "Running on $(hostname):"
eval "$(conda shell.bash hook)"
conda activate quimb


python3 -m train --config configs/lego_PuTT.txt --seed 42
