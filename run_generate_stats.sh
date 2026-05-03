#!/bin/bash
# run_generate_stats.sh — CPU job for generating dataset statistics
#SBATCH --job-name=generate_stats
#SBATCH --account=project_2019043
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=slurm-%j-stats.out

cd /scratch/project_2019043/ms-thesis

module load pytorch

echo "Job started: $(date)"

python generate_stats.py

echo "Job finished: $(date)"
