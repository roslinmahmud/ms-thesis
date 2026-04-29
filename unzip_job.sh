#!/bin/bash
#SBATCH --job-name=unzip_lo2
#SBATCH --account=project_2019043       # Your specific project ID
#SBATCH --partition=small               # Standard CPU partition
#SBATCH --time=04:00:00                 # 4 hours to handle 600GB
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2               
#SBATCH --mem=8G                        

# 1. Navigate directly to your thesis workspace
cd /scratch/project_2019043/ms-thesis

# 2. Unzip the file exactly as it was packaged
echo "Starting extraction..."
unzip lo2-data.zip -d lo2-data
echo "Extraction complete!"
