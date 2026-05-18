#!/bin/bash
#SBATCH --job-name=entry_descriptors
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=8G
#SBATCH --output=phase6_%j.log

cd ~/antibiotic_project
python phase_6_entry_descriptors.py \
    --input smiles_labels_valid.csv \
    --output entry_descriptors.csv \
    --n-conformers 10 \
    --n-jobs 16
