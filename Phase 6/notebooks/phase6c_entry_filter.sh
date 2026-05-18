#!/bin/bash
#SBATCH --job-name=phase6c
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=phase6c_%j.out
#SBATCH --error=phase6c_%j.err

echo "Job $SLURM_JOB_ID on $SLURMD_NODENAME — $(date)"
cd $SLURM_SUBMIT_DIR

python -u phase_6c_entry_filter.py \
    --smiles    smiles_labels_valid.csv \
    --labels    results_phase5/data/y.npy \
    --split     results_phase5/data/scaffold_split.npz \
    --features  results_phase5/data/X_features.npy \
    --entry-csv entry_descriptors.csv \
    --top-n     50 \
    --outdir    results_phase6

echo "Finished — $(date)"