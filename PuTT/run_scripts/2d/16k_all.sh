#!/bin/bash

#SBATCH --job-name=simple-gpu    # Job name
#SBATCH --output=output_files/job.%j.out      # Name of output file (%j expands to jobId)
#SBATCH --cpus-per-task=8        # Schedule 8 cores (includes hyperthreading)
#SBATCH --gres=gpu:v100:1               # Schedule a GPU, or more with gpu:2 etc
#SBATCH --time=10:00:00          # Run time (hh:mm:ss) - run for one hour max
#SBATCH --partition=red,brown    # Run on either the Red or Brown queue

echo "Running on $(hostname):"
nvidia-smi

eval "$(conda shell.bash hook)"
conda activate quimb


python3 large_exps/main_script.py --config configs/2d/tokyo/CP/tokyo16k.yaml
python3 large_exps/main_script.py --config configs/2d/tokyo/TT/tokyo16k.yaml
python3 large_exps/main_script.py --config configs/2d/tokyo/Tucker/tokyo16k.yaml
python3 large_exps/main_script.py --config configs/2d/girl/TT/girl16k.yaml
python3 large_exps/main_script.py --config configs/2d/girl/Tucker/girl16k.yaml
python3 large_exps/main_script.py --config configs/2d/girl/CP/girl16k.yaml
