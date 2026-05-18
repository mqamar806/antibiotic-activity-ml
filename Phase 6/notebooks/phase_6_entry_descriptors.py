# -*- coding: utf-8 -*-
"""
Phase 6 — eNTRy Rule + Physicochemical Descriptor Calculation
==============================================================

Goal
----
For every molecule in the cleaned Stokes E. coli dataset, compute the
descriptors needed to apply the eNTRy rules (Richter et al., Nature 2017)
plus standard physicochemical descriptors that govern Gram-negative
permeation (TPSA, LogP, etc.).

These features will then be combined with the existing ECFP-2048 fingerprints
in Phase 6 to train an extended XGBoost model.

What gets computed for every SMILES
-----------------------------------
2D descriptors (cheap, deterministic):
    - molecular weight
    - cLogP (Crippen)
    - TPSA
    - H-bond donors, H-bond acceptors
    - rotatable bonds  (RDKit strict definition: excludes amide N-C, terminal, ring)
    - fraction sp3 carbons
    - number of rings, aromatic rings
    - has_primary_amine   (Hergenrother SMARTS: NH2/NH3+ on sp3 C)

3D descriptors (require conformer generation, averaged over N conformers):
    - glob_hergenrother  (smallest/largest eigenvalue of atom-coord cov matrix)
    - spherocity         (RDKit Descriptors3D, analogous shape measure)
    - asphericity, NPR1, NPR2, radius_of_gyration  (extra shape descriptors)
    - pbf                (Plane of Best Fit: avg atom distance to best-fit plane)

Derived flag:
    - passes_eNTRy  = has_primary_amine AND rot_bonds <= 5 AND glob_hergenrother <= 0.25

Two definitions of globularity are computed and saved so we can compare
Hergenrother's PCA-based glob to RDKit's SpherocityIndex on this dataset.

Usage
-----
On Colab:
    !python phase_6_entry_descriptors.py \\
        --input smiles_labels_valid.csv \\
        --output descriptors.csv \\
        --n-conformers 5 \\
        --n-jobs 2

On HPC (with e.g. 16 cores):
    python phase_6_entry_descriptors.py \\
        --input smiles_labels_valid.csv \\
        --output descriptors.csv \\
        --n-conformers 10 \\
        --n-jobs 16

Dependencies
------------
    pip install rdkit pandas numpy joblib tqdm
"""

import argparse
import logging
import os
import sys
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

from rdkit import Chem, RDLogger
from rdkit.Chem import (
    AllChem,
    Crippen,
    Descriptors,
    Descriptors3D,
    Lipinski,
    rdMolDescriptors,
)

# Silence RDKit warnings about valence / parsing — we handle invalid mols ourselves.
RDLogger.DisableLog("rdApp.*")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("phase5")


# --------------------------------------------------------------------------- #
# SMARTS for the Hergenrother primary-amine definition.
# Matches NH2 (X3 neutral) OR NH3+ (X4 cationic) attached to an sp3 carbon.
# This excludes anilines, amides, secondary/tertiary amines — exactly the
# "non-sterically-encumbered ionizable nitrogen" referenced in the eNTRy rules.
# Source: Hergenrother entry-cli/calc_props.py.
# --------------------------------------------------------------------------- #
PRIMARY_AMINE_SMARTS = "[$([N;H2;X3][CX4]),$([N;H3;X4+][CX4])]"
PRIMARY_AMINE_PATTERN = Chem.MolFromSmarts(PRIMARY_AMINE_SMARTS)


# --------------------------------------------------------------------------- #
# Globularity (Hergenrother definition)
# --------------------------------------------------------------------------- #
def calc_glob_hergenrother(mol, conf_id):
    """
    Hergenrother globularity: ratio of the smallest to largest eigenvalue of
    the covariance matrix of atomic coordinates.

    Geometric interpretation: PCA on the atom cloud.
      - Flat molecule (e.g. benzene) -> smallest eigenvalue ~ 0 -> glob ~ 0
      - Spherical molecule (e.g. adamantane) -> all eigenvalues similar -> glob ~ 1

    Returns -1 if the largest eigenvalue is zero (degenerate / single atom).
    """
    conf = mol.GetConformer(conf_id)
    pts = conf.GetPositions()         # (N_atoms, 3)
    pts = pts.T                       # (3, N_atoms)
    cov = np.cov(pts)
    vals = np.linalg.eigvalsh(cov)    # ascending order, real symmetric
    if vals[-1] > 0:
        return float(vals[0] / vals[-1])
    return -1.0


# --------------------------------------------------------------------------- #
# Plane of Best Fit (Hergenrother / Firth et al.)
# Average distance of each atom from the best-fit plane through the molecule.
# --------------------------------------------------------------------------- #
def calc_pbf(mol, conf_id):
    """Average orthogonal distance of atoms to the best-fit plane (Angstroms)."""
    conf = mol.GetConformer(conf_id)
    pts = conf.GetPositions()                    # (N, 3)
    centroid = pts.mean(axis=0)
    centered = pts - centroid
    # SVD: last row of V is the normal to the best-fit plane (smallest singular value)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    normal = vh[-1]
    distances = np.abs(centered @ normal)
    return float(distances.mean())


# --------------------------------------------------------------------------- #
# Per-molecule featurization
# --------------------------------------------------------------------------- #
NAN_3D_KEYS = (
    "glob_hergenrother",
    "spherocity",
    "asphericity",
    "npr1",
    "npr2",
    "radius_of_gyration",
    "pbf",
)


def _nan_3d():
    return {k: np.nan for k in NAN_3D_KEYS}


def featurize(smiles, n_conformers=5, seed=42):
    """
    Compute all descriptors for one SMILES string.

    Returns a dict of descriptors, or None if the SMILES itself is invalid.
    If 3D embedding fails, 2D descriptors are still returned and the 3D
    fields are filled with NaN (so we can audit how many molecules failed).
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # ----- 2D descriptors -----------------------------------------------------
    desc = {
        "smiles": smiles,
        "mw": float(Descriptors.MolWt(mol)),
        "logp": float(Crippen.MolLogP(mol)),
        "tpsa": float(rdMolDescriptors.CalcTPSA(mol)),
        "hbd": int(Lipinski.NumHDonors(mol)),
        "hba": int(Lipinski.NumHAcceptors(mol)),
        "rotatable_bonds": int(Lipinski.NumRotatableBonds(mol)),
        "fraction_sp3": float(rdMolDescriptors.CalcFractionCSP3(mol)),
        "num_rings": int(rdMolDescriptors.CalcNumRings(mol)),
        "num_aromatic_rings": int(rdMolDescriptors.CalcNumAromaticRings(mol)),
        "heavy_atoms": int(mol.GetNumHeavyAtoms()),
        "has_primary_amine": int(mol.HasSubstructMatch(PRIMARY_AMINE_PATTERN)),
    }

    # ----- 3D descriptors -----------------------------------------------------
    # Add hydrogens before embedding (required for sensible 3D geometry).
    mol_h = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    params.useRandomCoords = False

    try:
        cids = AllChem.EmbedMultipleConfs(mol_h, numConfs=n_conformers, params=params)
        if len(cids) == 0:
            # Fallback: try with random coords + relaxed parameters
            params.useRandomCoords = True
            cids = AllChem.EmbedMultipleConfs(
                mol_h, numConfs=n_conformers, params=params
            )
        if len(cids) == 0:
            desc.update(_nan_3d())
            desc["n_confs_used"] = 0
            desc["passes_eNTRy"] = 0  # cannot evaluate without 3D
            return desc

        # MMFF-optimize each conformer (matches Hergenrother's approach).
        for cid in cids:
            try:
                AllChem.MMFFOptimizeMolecule(mol_h, confId=cid)
            except Exception:
                pass  # if MMFF fails on a conformer, keep the embedded coords

        # Compute 3D descriptors per conformer, then average.
        globs, spheros, aspheros, npr1s, npr2s, rgs, pbfs = ([] for _ in range(7))
        for cid in cids:
            globs.append(calc_glob_hergenrother(mol_h, cid))
            pbfs.append(calc_pbf(mol_h, cid))
            spheros.append(Descriptors3D.SpherocityIndex(mol_h, confId=cid))
            aspheros.append(Descriptors3D.Asphericity(mol_h, confId=cid))
            npr1s.append(Descriptors3D.NPR1(mol_h, confId=cid))
            npr2s.append(Descriptors3D.NPR2(mol_h, confId=cid))
            rgs.append(Descriptors3D.RadiusOfGyration(mol_h, confId=cid))

        desc["glob_hergenrother"] = float(np.mean(globs))
        desc["spherocity"] = float(np.mean(spheros))
        desc["asphericity"] = float(np.mean(aspheros))
        desc["npr1"] = float(np.mean(npr1s))
        desc["npr2"] = float(np.mean(npr2s))
        desc["radius_of_gyration"] = float(np.mean(rgs))
        desc["pbf"] = float(np.mean(pbfs))
        desc["n_confs_used"] = len(cids)

    except Exception as e:
        log.warning("3D failure for %s: %s", smiles, repr(e))
        desc.update(_nan_3d())
        desc["n_confs_used"] = 0

    # ----- Derived eNTRy pass flag -------------------------------------------
    glob = desc["glob_hergenrother"]
    desc["passes_eNTRy"] = int(
        bool(desc["has_primary_amine"])
        and desc["rotatable_bonds"] <= 5
        and (not np.isnan(glob))
        and (glob <= 0.25)
    )
    return desc


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def _featurize_row(args):
    """Worker function suitable for joblib. Returns (idx, desc_or_None, label, smi)."""
    idx, smi, label, n_conformers, seed = args
    try:
        d = featurize(smi, n_conformers=n_conformers, seed=seed)
    except Exception as e:
        log.warning("Hard failure on idx=%d smi=%s : %s", idx, smi, repr(e))
        d = None
    return idx, d, label, smi


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--input",
        required=True,
        help="CSV with at least columns: SMILES, label (e.g. smiles_labels_valid.csv from Phase 2)",
    )
    p.add_argument(
        "--output",
        required=True,
        help="Output CSV path (will contain SMILES, label, and all descriptors)",
    )
    p.add_argument(
        "--n-conformers",
        type=int,
        default=5,
        help="Number of conformers per molecule (default: 5). More = slower, more stable 3D descriptors.",
    )
    p.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Parallel workers (default: 1). Set to number of cores on HPC.",
    )
    p.add_argument(
        "--seed", type=int, default=42, help="Random seed for ETKDG embedding"
    )
    p.add_argument(
        "--limit", type=int, default=None, help="Optional: only process first N rows (debugging)"
    )
    args = p.parse_args()

    log.info("Reading input: %s", args.input)
    df = pd.read_csv(args.input)
    if "SMILES" not in df.columns or "label" not in df.columns:
        log.error("Input must have columns 'SMILES' and 'label'. Found: %s",
                  list(df.columns))
        sys.exit(1)

    if args.limit is not None:
        df = df.head(args.limit).copy()
        log.info("Limiting to first %d rows for debug", args.limit)

    log.info("Will process %d molecules with %d conformers each on %d worker(s)",
             len(df), args.n_conformers, args.n_jobs)

    job_args = [
        (i, row["SMILES"], int(row["label"]), args.n_conformers, args.seed)
        for i, row in df.iterrows()
    ]

    t0 = time.time()
    if args.n_jobs == 1:
        results = []
        for ja in tqdm(job_args, desc="Featurizing"):
            results.append(_featurize_row(ja))
    else:
        # joblib handles the progress bar via tqdm wrapping
        results = Parallel(n_jobs=args.n_jobs, backend="loky", verbose=0)(
            delayed(_featurize_row)(ja)
            for ja in tqdm(job_args, desc="Featurizing")
        )
    elapsed = time.time() - t0
    log.info("Featurization done in %.1f s (%.2f s/mol)", elapsed, elapsed / len(df))

    # ----- Assemble output ----------------------------------------------------
    rows = []
    n_bad_smiles = 0
    n_3d_failed = 0
    for idx, d, label, smi in results:
        if d is None:
            n_bad_smiles += 1
            continue
        d["label"] = label
        # Keep SMILES first / label second for sanity
        rows.append(d)
        if np.isnan(d.get("glob_hergenrother", np.nan)):
            n_3d_failed += 1

    out_df = pd.DataFrame(rows)

    # Reorder columns: identifiers first, then 2D, then 3D, then derived
    col_order = [
        "smiles", "label",
        # 2D
        "mw", "logp", "tpsa", "hbd", "hba", "rotatable_bonds",
        "fraction_sp3", "num_rings", "num_aromatic_rings", "heavy_atoms",
        "has_primary_amine",
        # 3D
        "glob_hergenrother", "spherocity", "asphericity",
        "npr1", "npr2", "radius_of_gyration", "pbf",
        "n_confs_used",
        # derived
        "passes_eNTRy",
    ]
    out_df = out_df[[c for c in col_order if c in out_df.columns]]
    out_df.to_csv(args.output, index=False)

    # ----- Summary ------------------------------------------------------------
    log.info("=" * 60)
    log.info("Wrote %s", args.output)
    log.info("Rows: %d (input had %d)", len(out_df), len(df))
    log.info("Invalid SMILES skipped: %d", n_bad_smiles)
    log.info("3D embedding failed (glob NaN): %d", n_3d_failed)
    n_pass = int(out_df["passes_eNTRy"].sum())
    n_total = len(out_df)
    log.info("Passing eNTRy rules: %d / %d  (%.2f%%)",
             n_pass, n_total, 100.0 * n_pass / max(n_total, 1))

    # Breakdown by class label
    if "label" in out_df.columns:
        for lbl in sorted(out_df["label"].unique()):
            sub = out_df[out_df["label"] == lbl]
            passed = int(sub["passes_eNTRy"].sum())
            log.info("  label=%s: %d compounds, %d pass eNTRy (%.2f%%)",
                     lbl, len(sub), passed,
                     100.0 * passed / max(len(sub), 1))

    log.info("=" * 60)
    log.info("Next step (Phase 6): combine these descriptors with X_ecfp2048.npy")
    log.info("and retrain XGBoost. Use the same scaffold split as Phase 4.")


if __name__ == "__main__":
    main()
