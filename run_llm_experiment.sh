#!/bin/bash
# submit_llm.sh — GPU job for LLM + LoRA experiment
#SBATCH --job-name=llm_lora
#SBATCH --account=project_2019043
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --gres=gpu:v100:1
#SBATCH --time=36:00:00
#SBATCH --output=slurm-%j-llm.out
 
cd /scratch/project_2019043/ms-thesis
 
module load pytorch
 
 
echo "Job started: $(date)"
 
python experiment_llm.py
 
echo "Job finished: $(date)"