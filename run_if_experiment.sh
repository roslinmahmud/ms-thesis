#!/bin/bash
# run_if_experiment.sh — CPU job for Isolation Forest baseline
#SBATCH --job-name=if_baseline
#SBATCH --account=project_2019043
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=16:00:00
#SBATCH --output=slurm-%j-if_baseline.out

cd /scratch/project_2019043/ms-thesis

module load pytorch

echo "Job started: $(date)"

python experiment_if.py

echo "Job finished: $(date)"
