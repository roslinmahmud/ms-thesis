#!/bin/bash
#SBATCH --job-name=dataset_stats
#SBATCH --account=project_2019043
#SBATCH --partition=small
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G  

# 1. Navigate to project folder
cd /scratch/project_2019043/ms-thesis

echo "Loading environment..."

# Load the correct PyTorch module
module load pytorch

echo "Starting dataset description test..."

python dataset_description.py

echo "Dataset description test finished!"
