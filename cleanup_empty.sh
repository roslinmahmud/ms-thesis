#!/bin/bash
#SBATCH --job-name=clean_empty_files
#SBATCH --account=project_2019043       
#SBATCH --partition=small               
#SBATCH --time=02:00:00                 # 2 hours is plenty for a file search
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2               
#SBATCH --mem=4G                        

# Navigate to your main thesis directory
cd /scratch/project_2019043/ms-thesis

echo "Starting recursive search for empty files..."

# 1. Count how many empty files exist before we delete them (optional but helpful!)
empty_count=$(find . -type f -empty | wc -l)
echo "Found $empty_count empty files taking up quota space."

# 2. Find all files (-type f) that are completely empty (-empty) and delete them (-delete)
find . -type f -empty -delete

echo "Cleanup complete! Freed up $empty_count slots in your file quota."
