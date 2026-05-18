#!/bin/bash
#SBATCH --job-name=phase6_model
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=phase6_model_%j.out
#SBATCH --error=phase6_model_%j.err

echo "Job $SLURM_JOB_ID on $SLURMD_NODENAME — $(date)"
cd $SLURM_SUBMIT_DIR

python -u phase_6_model.py \
    --features  results_phase5/data/X_features.npy \
    --labels    results_phase5/data/y.npy \
    --split     results_phase5/data/scaffold_split.npz \
    --entry-csv entry_descriptors.csv \
    --outdir    results_phase6

echo "Finished — $(date)"