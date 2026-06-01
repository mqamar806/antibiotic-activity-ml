# -*- coding: utf-8 -*-
"""
Phase 6D — Structural Novelty Analysis (Tanimoto Similarity)
=============================================================

Checks whether the top predicted active compounds are structurally
novel or are rediscoveries of known antibiotics.

For each compound in the top-N predictions:
  1. Compute Tanimoto similarity against a reference panel of 20 known
     antibiotics covering all major classes (beta-lactams, fluoroquinolones,
     tetracyclines, macrolides, aminoglycosides, glycopeptides, etc.)
  2. Flag compounds with max similarity > 0.4 as "similar to known antibiotic"
     (0.4 is a commonly used scaffold-level similarity threshold in drug discovery)
  3. Report which known antibiotic each prediction most resembles
  4. Highlight structurally novel candidates (max Tanimoto < 0.4)

Thresholds (standard drug discovery practice):
  > 0.70 — highly similar, likely same scaffold (rediscovery)
  0.40–0.70 — related scaffold, possibly same class
  < 0.40 — structurally distinct, candidate for novelty

Usage
-----
python phase_6d_novelty.py \\
    --smiles    smiles_labels_valid.csv \\
    --labels    results_phase5/data/y.npy \\
    --split     results_phase5/data/scaffold_split.npz \\
    --features  results_phase5/data/X_features.npy \\
    --top-n     50 \\
    --outdir    results_phase6
"""

import argparse
import json
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, DataStructs, rdMolDescriptors
from rdkit.Chem.rdMolDescriptors import CalcTPSA
from rdkit.Chem.Crippen import MolLogP
from xgboost import XGBClassifier

RDLogger.DisableLog("rdApp.*")
RNG = 42

XGB_PARAMS = dict(
    max_depth=7, learning_rate=0.036, n_estimators=1300,
    subsample=0.74, colsample_bytree=0.44,
    random_state=RNG, tree_method="hist", eval_metric="aucpr",
)

# ── Reference antibiotic panel ────────────────────────────────────────────────
# 20 antibiotics covering all major classes active against Gram-negative bacteria.
# SMILES sourced from PubChem / DrugBank.
KNOWN_ANTIBIOTICS = {
    # Beta-lactams
    "Ampicillin":       "O=C(O)[C@@H]2N3C(=O)[C@@H](NC(=O)[C@@H](c1ccccc1)N)[C@H]3SC2(C)C",
    "Amoxicillin":      "O=C(O)[C@@H]2N3C(=O)[C@@H](NC(=O)[C@@H](c1ccc(O)cc1)N)[C@H]3SC2(C)C",
    "Piperacillin":     "O=C(O)[C@@H]3N4C(=O)[C@@H](NC(=O)c1ccccc1NC(=O)N2CCNCC2)[C@H]4SC3(C)C",
    "Meropenem":        "O=C(O)[C@@H]3C[C@@H]2CC(=C(C(=O)N[C@H](C)c1ccccc1)[C@H]2[C@H]3SC)C(=O)O",
    # Fluoroquinolones
    "Ciprofloxacin":    "O=C(O)c1cn(C2CC2)c2cc(N3CCNCC3)c(F)cc2c1=O",
    "Levofloxacin":     "O=C(O)c1cn2c(=O)c(C(=O)O)cn2c2cc(N3CCNCC3)c(F)cc12",
    "Norfloxacin":      "O=C(O)c1cn(CC)c2cc(N3CCNCC3)c(F)cc2c1=O",
    # Tetracyclines
    "Tetracycline":     "OC1=C(C(=O)[C@H]2C[C@@H]3CC4C(O)(C(=O)C(N(C)C)=C4O)C(=O)[C@@H]3[C@@H]2[C@@H]1O)c1ccccc1=O",
    "Doxycycline":      "OC1=C(C(=O)[C@H]2C[C@@H]3C[C@H]4C(O)(C(=O)C(N(C)C)=C4O)C(=O)[C@@H]3[C@@H]2[C@@H]1O)c1ccccc1=O",
    # Macrolides
    "Erythromycin":     "CCC1OC(=O)[C@H](C)[C@@H](OC2C[C@@](C)(OC)C[C@H](C)[C@@H]2O)[C@H](C)[C@@H](O)[C@](C)(O)C[C@@H](C)C(=O)[C@H](C)[C@@H]1OC1C[C@@H](N(C)C)[C@H](O)[C@@H](C)O1",
    "Azithromycin":     "CCC1OC(=O)[C@H](C)[C@@H](OC2C[C@@](C)(OC)C[C@H](C)[C@@H]2O)[C@H](C)[C@@H](O)[C@@](C)(O)C[C@H](CN(C)C)[C@@H](C)C(=O)[C@H](C)[C@@H]1OC1C[C@@H](N(C)C)[C@H](O)[C@@H](C)O1",
    # Aminoglycosides
    "Gentamicin":       "O([C@@H]1[C@H](O)[C@@H](N)[C@H](O[C@@H]2[C@H](O)[C@@H](NC)[C@@H](O[C@@H]3[C@@H](N)C[C@@H](N)[C@H](O)[C@@H]3O)[C@H]2O)[C@@H](C)O1)[H]",
    "Streptomycin":     "O=C/N=C(\\N)NCC1OC(O[C@H]2[C@H](O)[C@@H](O)[C@H](NC(=N)N)[C@@H](O)[C@H]2O[C@@H]2O[C@H](CO)[C@@H](O)[C@H](O)[C@H]2NC(=N)N)[C@@H](O)[C@H](O)[C@H]1O",
    # Glycopeptides
    "Vancomycin":       "C[C@H]1[C@H]([C@@](C[C@@H](O1)O[C@@H]2[C@H]([C@@H]([C@H](O[C@H]2Oc3c4cc5cc3Oc6ccc(cc6Cl)[C@H]([C@H](C(=O)N[C@H](C(=O)N[C@H]5C(=O)N[C@@H]7c8ccc(c(c8)-c9c(cc(cc9O)O)[C@H](NC(=O)[C@H]([C@@H](c1ccc(c(c1)Cl)O4)O)NC7=O)C(=O)O)O)CC(=O)N)NC(=O)[C@@H](CC(C)C)NC)O)CO)O)O)(C)N)O",
    # Sulfonamides / folate inhibitors
    "Trimethoprim":     "COc1cc(Cc2cnc(N)nc2N)cc(OC)c1OC",
    "Sulfamethoxazole": "Cc1cc(NS(=O)(=O)c2ccc(N)cc2)no1",
    # Polymyxins / membrane disruptors
    "Colistin":         "CCCCCCCCCC(=O)N[C@@H](CCN)C(=O)N[C@H]1CCNC(=O)[C@@H](NC(=O)[C@H](CCN)NC(=O)[C@@H](NC(=O)[C@H](CCN)NC(=O)[C@H](Cc2ccc(O)cc2)NC(=O)[C@@H](NC1=O)CCN)CCN)CCN",
    # Others
    "Chloramphenicol":  "O=C(NC(c1ccc([N+](=O)[O-])cc1)[C@@H](O)CCl)CCl",
    "Rifampicin":       "CO[C@H]1/C=C/O[C@@]2(C)Oc3c(C)c(O)c4c(=O)/C(=C\\N/N=C/c5c(O)c(OC)c(O)cc5C)C(=O)c4c3[C@@H]2CC[C@H]1OC(=O)/C=C/C",
    "Linezolid":        "O=C(N[C@@H](Cc1ccc(N2CC(=O)OC2=O)cc1F)CO)c1cncc(F)c1",
    "Halicin":          "c1cc2c(cc1[N+](=O)[O-])SC(=O)N2",  # from Stokes et al. 2020
}

SIMILARITY_THRESHOLDS = {
    "high":   0.70,   # likely rediscovery
    "medium": 0.40,   # related scaffold
}


def smiles_to_fp(smiles, radius=2, nbits=2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=nbits)


def tanimoto(fp1, fp2):
    if fp1 is None or fp2 is None:
        return 0.0
    return DataStructs.TanimotoSimilarity(fp1, fp2)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--smiles",   required=True)
    p.add_argument("--labels",   required=True)
    p.add_argument("--split",    required=True)
    p.add_argument("--features", required=True)
    p.add_argument("--top-n",    type=int, default=50)
    p.add_argument("--outdir",   default="results_phase6")
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    os.makedirs(f"{args.outdir}/figures", exist_ok=True)

    # ── 1. Load and predict ──────────────────────────────────────────────────
    print("=" * 70)
    print("STEP 1: Loading data and getting predictions")
    print("=" * 70)

    smiles_df = pd.read_csv(args.smiles)
    y         = np.load(args.labels)
    split     = np.load(args.split)
    X         = np.load(args.features)

    train_idx = split["train_idx"]
    test_idx  = split["test_idx"]
    y_train = y[train_idx]; y_test = y[test_idx]
    X_train = X[train_idx]; X_test = X[test_idx]
    test_smiles = smiles_df.iloc[test_idx]["SMILES"].values

    spw = float((y_train == 0).sum()) / float((y_train == 1).sum())
    model = XGBClassifier(**XGB_PARAMS, scale_pos_weight=spw)
    model.fit(X_train, y_train)
    y_probs = model.predict_proba(X_test)[:, 1]

    # Sort by probability, take top-N
    sorted_idx = np.argsort(y_probs)[::-1]
    top_idx    = sorted_idx[:args.top_n]

    print(f"Test set: {len(y_test)} compounds, {int(y_test.sum())} true actives")
    print(f"Evaluating top {args.top_n} predictions")

    # ── 2. Pre-compute reference fingerprints ────────────────────────────────
    print("\n" + "=" * 70)
    print("STEP 2: Computing reference antibiotic fingerprints")
    print("=" * 70)

    ref_fps = {}
    for name, smi in KNOWN_ANTIBIOTICS.items():
        fp = smiles_to_fp(smi)
        if fp is not None:
            ref_fps[name] = fp
            print(f"  OK: {name}")
        else:
            print(f"  WARN: could not parse {name}")

    print(f"\n  Reference panel: {len(ref_fps)} antibiotics")

    # ── 3. Similarity analysis for top-N ────────────────────────────────────
    print("\n" + "=" * 70)
    print("STEP 3: Tanimoto similarity analysis")
    print("=" * 70)

    records = []
    for rank, idx in enumerate(top_idx, 1):
        smi   = test_smiles[idx]
        label = int(y_test[idx])
        prob  = float(y_probs[idx])

        query_fp = smiles_to_fp(smi)

        # Compute similarity to every reference antibiotic
        sims = {name: tanimoto(query_fp, fp) for name, fp in ref_fps.items()}
        max_sim    = max(sims.values())
        most_sim   = max(sims, key=sims.get)

        # Classify novelty
        if max_sim >= SIMILARITY_THRESHOLDS["high"]:
            novelty = "Likely rediscovery"
        elif max_sim >= SIMILARITY_THRESHOLDS["medium"]:
            novelty = "Related scaffold"
        else:
            novelty = "Structurally novel"

        records.append({
            "rank":         rank,
            "smiles":       smi,
            "true_label":   label,
            "pred_prob":    round(prob, 4),
            "max_tanimoto": round(max_sim, 4),
            "most_similar": most_sim,
            "sim_to_most":  round(sims[most_sim], 4),
            "novelty":      novelty,
            **{f"sim_{n}": round(v, 3) for n, v in sims.items()},
        })

    df = pd.DataFrame(records)

    # ── 4. Print summary ─────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"STEP 4: Novelty summary — top {args.top_n} predictions")
    print("=" * 70)

    counts = df["novelty"].value_counts()
    for cat in ["Structurally novel", "Related scaffold", "Likely rediscovery"]:
        n = counts.get(cat, 0)
        print(f"  {cat:<25}: {n:>3} / {args.top_n}")

    print(f"\n  {'Rank':<6} {'Prob':>6} {'True':>5} {'MaxSim':>8} "
          f"{'Most Similar To':<22} {'Novelty'}")
    print("  " + "-" * 80)
    for _, row in df.iterrows():
        label_str = "ACTIVE" if row["true_label"] == 1 else "      "
        novel_str = row["novelty"]
        print(f"  {int(row['rank']):<6} {row['pred_prob']:>6.3f} "
              f"{label_str:>5} {row['max_tanimoto']:>8.3f} "
              f"{row['most_similar']:<22} {novel_str}")

    # Highlight novel true actives
    novel_actives = df[(df["novelty"] == "Structurally novel") & (df["true_label"] == 1)]
    print(f"\n  >>> Structurally novel TRUE ACTIVES: {len(novel_actives)}")
    if len(novel_actives) > 0:
        for _, row in novel_actives.iterrows():
            print(f"      Rank {int(row['rank'])}: prob={row['pred_prob']:.3f}, "
                  f"max_sim={row['max_tanimoto']:.3f} (vs {row['most_similar']})")
            smi_short = row["smiles"][:70] + "..." if len(row["smiles"]) > 70 else row["smiles"]
            print(f"      SMILES: {smi_short}")

    # ── 5. Figures ───────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("STEP 5: Generating figures")
    print("=" * 70)

    color_map = {
        "Structurally novel": "#2ECC71",
        "Related scaffold":   "#F39C12",
        "Likely rediscovery": "#E74C3C",
    }

    # Figure 1: Max Tanimoto distribution for top-N
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: histogram of max Tanimoto, coloured by novelty category
    for cat, color in color_map.items():
        subset = df[df["novelty"] == cat]["max_tanimoto"]
        if len(subset):
            axes[0].hist(subset, bins=15, color=color, alpha=0.75, label=cat)
    axes[0].axvline(SIMILARITY_THRESHOLDS["medium"], color="grey",
                    linestyle="--", linewidth=1.5, label="Novel threshold (0.40)")
    axes[0].axvline(SIMILARITY_THRESHOLDS["high"],   color="black",
                    linestyle="--", linewidth=1.5, label="Rediscovery threshold (0.70)")
    axes[0].set_xlabel("Max Tanimoto Similarity to Any Known Antibiotic")
    axes[0].set_ylabel("Count")
    axes[0].set_title(f"Structural Novelty Distribution\n(Top {args.top_n} Predictions)")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    # Right: scatter rank vs max Tanimoto, shaped by true label
    for _, row in df.iterrows():
        color  = color_map[row["novelty"]]
        marker = "*" if row["true_label"] == 1 else "o"
        size   = 150 if row["true_label"] == 1 else 60
        axes[1].scatter(row["rank"], row["max_tanimoto"],
                        c=color, marker=marker, s=size,
                        alpha=0.8, linewidths=0.5, edgecolors="white")

    axes[1].axhline(SIMILARITY_THRESHOLDS["medium"], color="grey",
                    linestyle="--", linewidth=1.5, label="Novel < 0.40")
    axes[1].axhline(SIMILARITY_THRESHOLDS["high"],   color="black",
                    linestyle="--", linewidth=1.5, label="Rediscovery > 0.70")
    axes[1].set_xlabel("Rank (by predicted probability)")
    axes[1].set_ylabel("Max Tanimoto Similarity")
    axes[1].set_title(f"Max Tanimoto vs Rank\n(\u2605 = true active)")

    legend_handles = [
        mpatches.Patch(color=color_map["Structurally novel"], label="Structurally novel"),
        mpatches.Patch(color=color_map["Related scaffold"],   label="Related scaffold"),
        mpatches.Patch(color=color_map["Likely rediscovery"], label="Likely rediscovery"),
    ]
    axes[1].legend(handles=legend_handles, fontsize=8)
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    fig1_path = f"{args.outdir}/figures/tanimoto_novelty_top{args.top_n}.png"
    fig.savefig(fig1_path, dpi=150)
    plt.close()
    print(f"  Saved: {fig1_path}")

    # Figure 2: Heatmap — top 20 predictions vs all reference antibiotics
    top20 = df.head(20)
    sim_cols = [c for c in df.columns if c.startswith("sim_")]
    ref_names = [c.replace("sim_", "") for c in sim_cols]
    heatmap_data = top20[sim_cols].values

    fig2, ax = plt.subplots(figsize=(14, 7))
    im = ax.imshow(heatmap_data, aspect="auto", cmap="YlOrRd",
                   vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, label="Tanimoto Similarity")
    ax.set_xticks(range(len(ref_names)))
    ax.set_xticklabels(ref_names, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(top20)))
    y_labels = [
        f"Rank {int(r['rank'])} {'★ ACTIVE' if r['true_label']==1 else ''} (p={r['pred_prob']:.2f})"
        for _, r in top20.iterrows()
    ]
    ax.set_yticklabels(y_labels, fontsize=8)
    ax.set_title("Tanimoto Similarity Heatmap — Top 20 Predictions vs Known Antibiotics",
                 fontsize=11, pad=12)

    # Add novelty colour strip on left
    for i, (_, row) in enumerate(top20.iterrows()):
        color = color_map[row["novelty"]]
        ax.add_patch(mpatches.Rectangle((-0.5 - len(ref_names)*0.08, i-0.5),
                                         0.5, 1, color=color, clip_on=False))

    plt.tight_layout()
    fig2_path = f"{args.outdir}/figures/tanimoto_heatmap_top20.png"
    fig2.savefig(fig2_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fig2_path}")

    # ── 6. Save outputs ──────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("STEP 6: Saving outputs")
    print("=" * 70)

    # Full table
    out_cols = ["rank", "smiles", "true_label", "pred_prob",
                "max_tanimoto", "most_similar", "sim_to_most", "novelty"]
    table_path = f"{args.outdir}/novelty_analysis_top{args.top_n}.csv"
    df[out_cols].to_csv(table_path, index=False)
    print(f"  Saved: {table_path}")

    # Novel candidates (max_tanimoto < 0.40)
    novel_df = df[df["novelty"] == "Structurally novel"][out_cols].copy()
    novel_path = f"{args.outdir}/novel_candidates.csv"
    novel_df.to_csv(novel_path, index=False)
    print(f"  Saved: {novel_path}  ({len(novel_df)} novel compounds)")

    # JSON summary
    summary = {
        "top_n": args.top_n,
        "reference_antibiotics": list(ref_fps.keys()),
        "thresholds": SIMILARITY_THRESHOLDS,
        "novelty_counts": {
            "structurally_novel":  int(counts.get("Structurally novel", 0)),
            "related_scaffold":    int(counts.get("Related scaffold", 0)),
            "likely_rediscovery":  int(counts.get("Likely rediscovery", 0)),
        },
        "novel_true_actives": len(novel_actives),
        "true_actives_in_top_n": int(df["true_label"].sum()),
    }
    json_path = f"{args.outdir}/results_phase6d.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved: {json_path}")

    print("\n" + "=" * 70)
    print("Phase 6D complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
