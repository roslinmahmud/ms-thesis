#!/bin/bash
#SBATCH --job-name=test_preprocessing
#SBATCH --account=project_2019043       
#SBATCH --partition=small               # Standard CPU partition
#SBATCH --time=04:00:00                 # 4 hour is plenty for this test
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4               # 4 cores to handle file I/O smoothly
#SBATCH --mem=16G                       # 16 GB of RAM for holding sequences

# 1. Navigate to the folder where your script and lo2-data folder live
cd /scratch/project_2019043/ms-thesis

echo "Loading environment..."
module load pytorch

echo "Starting preprocessing test..."

# 2. Run your Python script
python preprocessing.py

echo "Preprocessing test finished!"
