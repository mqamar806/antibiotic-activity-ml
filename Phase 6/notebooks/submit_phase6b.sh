#!/bin/bash
#SBATCH --job-name=phase6b
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=phase6b_%j.out
#SBATCH --error=phase6b_%j.err

echo "Job $SLURM_JOB_ID on $SLURMD_NODENAME — $(date)"
cd $SLURM_SUBMIT_DIR

python -u phase_6b_entry_only.py \
    --labels    results_phase5/data/y.npy \
    --split     results_phase5/data/scaffold_split.npz \
    --entry-csv entry_descriptors.csv \
    --outdir    results_phase6

echo "Finished — $(date)"