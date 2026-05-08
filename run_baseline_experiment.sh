#!/bin/bash
# run_baseline_experiment.sh — GPU job for Isolation Forest + LogBERT baselines
#SBATCH --job-name=baselines
#SBATCH --account=project_2019043
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:v100:1
#SBATCH --time=36:00:00
#SBATCH --output=slurm-%j-baselines.out
 
cd /scratch/project_2019043/ms-thesis
 
module load pytorch
 
 
echo "Job started: $(date)"
 
python experiment_baselines.py
 
echo "Job finished: $(date)"