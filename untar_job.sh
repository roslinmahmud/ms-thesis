#!/bin/bash
#SBATCH --job-name=untar_and_clean
#SBATCH --account=project_2019043       
#SBATCH --partition=small               
#SBATCH --time=04:00:00                 
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2               
#SBATCH --mem=8G                        

cd /scratch/project_2019043/ms-thesis/lo2-data

echo "Starting bulletproof extraction and cleanup..."

for archive in *.tar; do
    echo "Extracting $archive..."
    
    # 1. Run the extraction (ignoring soft warnings)
    tar -xf "$archive" --exclude="*/metrics/*"
    
    # 2. Calculate what the extracted folder name should be (removes '.tar')
    folder_name="${archive%.tar}"
    
    # 3. Explicitly check if the folder exists now
    if [ -d "$folder_name" ]; then
        rm "$archive"
        echo "Successfully verified and deleted $archive"
    else
        echo "WARNING: $folder_name not found. Skipping deletion to be safe."
    fi
done

echo "Process complete!"
