# -*- coding: utf-8 -*-
"""
Phase 6C — eNTRy Filter on Top Model Predictions
=================================================

This is what the supervisor asked for:
  1. Take the top N compounds predicted as active by the best Phase 5 model
  2. Apply eNTRy rules + TPSA/logP to check if they can even enter
     a Gram-negative bacterial cell
  3. Flag compounds that are BOTH predicted active AND likely to accumulate
     — these are the highest-priority candidates for wet-lab testing

This answers the question: "Of the compounds our model says are active,
which ones can actually get inside E. coli to reach their target?"

Usage
-----
python phase_6c_entry_filter.py \
    --smiles     smiles_labels_valid.csv \
    --labels     results_phase5/data/y.npy \
    --split      results_phase5/data/scaffold_split.npz \
    --features   results_phase5/data/X_features.npy \
    --entry-csv  entry_descriptors.csv \
    --model      results_phase5/models/best_XGBoost.json \
    --top-n      50 \
    --outdir     results_phase6
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

from xgboost import XGBClassifier
from sklearn.metrics import average_precision_score

RNG = 42

# Phase 5 best XGBoost hyperparameters
XGB_PARAMS = dict(
    max_depth=7, learning_rate=0.036, n_estimators=1300,
    subsample=0.74, colsample_bytree=0.44,
    random_state=RNG, tree_method="hist", eval_metric="aucpr",
)

# eNTRy thresholds (Richter et al., Nature 2017)
GLOB_THRESHOLD = 0.25
ROTBOND_THRESHOLD = 5

# Lipinski/permeability reference thresholds for context
TPSA_THRESHOLD  = 140   # above this = poor membrane permeability
LOGP_MIN        = -2
LOGP_MAX        = 6     # outside this range = permeability issues


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--smiles",    required=True, help="smiles_labels_valid.csv")
    p.add_argument("--labels",    required=True, help="results_phase5/data/y.npy")
    p.add_argument("--split",     required=True, help="results_phase5/data/scaffold_split.npz")
    p.add_argument("--features",  required=True, help="results_phase5/data/X_features.npy")
    p.add_argument("--entry-csv", required=True, help="entry_descriptors.csv")
    p.add_argument("--model",     default=None,
                   help="Optional: saved XGBoost model .json. If not provided, retrains from features.")
    p.add_argument("--top-n",     type=int, default=50,
                   help="Number of top predicted actives to evaluate (default: 50)")
    p.add_argument("--outdir",    default="results_phase6")
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    os.makedirs(f"{args.outdir}/figures", exist_ok=True)

    # ----------------------------------------------------------------------- #
    # 1. Load everything
    # ----------------------------------------------------------------------- #
    print("=" * 70)
    print("STEP 1: Loading data")
    print("=" * 70)

    smiles_df = pd.read_csv(args.smiles)
    y         = np.load(args.labels)
    split     = np.load(args.split)
    X         = np.load(args.features)
    entry_df  = pd.read_csv(args.entry_csv)

    train_idx = split["train_idx"]
    test_idx  = split["test_idx"]

    y_train = y[train_idx]
    y_test  = y[test_idx]
    X_train = X[train_idx]
    X_test  = X[test_idx]

    # Align SMILES and entry descriptors with test set
    test_smiles = smiles_df.iloc[test_idx]["SMILES"].values
    test_entry  = entry_df.iloc[test_idx].reset_index(drop=True)

    print(f"Test set: {len(y_test)} compounds, {int(y_test.sum())} true actives")

    # ----------------------------------------------------------------------- #
    # 2. Get model predictions
    # ----------------------------------------------------------------------- #
    print("\n" + "=" * 70)
    print("STEP 2: Getting model predictions")
    print("=" * 70)

    spw = float((y_train == 0).sum()) / float((y_train == 1).sum())

    if args.model and os.path.exists(args.model):
        print(f"  Loading saved model: {args.model}")
        model = XGBClassifier(**XGB_PARAMS, scale_pos_weight=spw)
        model.load_model(args.model)
    else:
        print("  Retraining XGBoost with Phase 5 best params...")
        model = XGBClassifier(**XGB_PARAMS, scale_pos_weight=spw)
        model.fit(X_train, y_train)

    y_probs = model.predict_proba(X_test)[:, 1]
    auprc = average_precision_score(y_test, y_probs)
    print(f"  AUPRC on test set: {auprc:.4f}")

    # ----------------------------------------------------------------------- #
    # 3. Build the ranked prediction table
    # ----------------------------------------------------------------------- #
    print("\n" + "=" * 70)
    print("STEP 3: Building ranked prediction table")
    print("=" * 70)

    results = pd.DataFrame({
        "smiles":            test_smiles,
        "true_label":        y_test.astype(int),
        "predicted_prob":    y_probs,
    })

    # Attach eNTRy descriptors
    entry_cols = ["mw", "logp", "tpsa", "hbd", "hba",
                  "rotatable_bonds", "has_primary_amine",
                  "glob_hergenrother", "spherocity", "pbf",
                  "passes_eNTRy"]
    for col in entry_cols:
        if col in test_entry.columns:
            results[col] = test_entry[col].values

    # Sort by predicted probability (highest first)
    results = results.sort_values("predicted_prob", ascending=False).reset_index(drop=True)
    results["rank"] = results.index + 1

    # Additional permeability flags
    if "tpsa" in results.columns:
        results["tpsa_ok"]  = (results["tpsa"] <= TPSA_THRESHOLD).astype(int)
    if "logp" in results.columns:
        results["logp_ok"]  = (
            (results["logp"] >= LOGP_MIN) & (results["logp"] <= LOGP_MAX)
        ).astype(int)

    # Combined priority flag: high predicted probability + passes eNTRy
    # Using top-N threshold for "high probability"
    top_n = args.top_n
    results["in_top_n"] = (results["rank"] <= top_n).astype(int)
    if "passes_eNTRy" in results.columns:
        results["priority_candidate"] = (
            (results["in_top_n"] == 1) & (results["passes_eNTRy"] == 1)
        ).astype(int)

    # ----------------------------------------------------------------------- #
    # 4. Print analysis
    # ----------------------------------------------------------------------- #
    print("\n" + "=" * 70)
    print(f"STEP 4: eNTRy filter analysis on top {top_n} predictions")
    print("=" * 70)

    top = results[results["rank"] <= top_n].copy()
    n_true_active_in_top = int(top["true_label"].sum())
    n_pass_entry = int(top["passes_eNTRy"].sum()) if "passes_eNTRy" in top.columns else "N/A"
    n_priority   = int(top["priority_candidate"].sum()) if "priority_candidate" in top.columns else "N/A"

    print(f"\n  Top {top_n} predicted actives:")
    print(f"    True actives in top {top_n}          : {n_true_active_in_top} / {top_n}  "
          f"(Precision@{top_n} = {n_true_active_in_top/top_n:.2f})")
    print(f"    Passing eNTRy rules                 : {n_pass_entry} / {top_n}")
    print(f"    Priority candidates (active + eNTRy): {n_priority} / {top_n}")

    if "tpsa" in top.columns:
        print(f"    TPSA ≤ {TPSA_THRESHOLD} (membrane-permeable)    : {int(top['tpsa_ok'].sum())} / {top_n}")
    if "logp" in top.columns:
        print(f"    logP in [{LOGP_MIN}, {LOGP_MAX}]                    : {int(top['logp_ok'].sum())} / {top_n}")

    # Breakdown of priority candidates
    if "priority_candidate" in top.columns and n_priority > 0:
        print(f"\n  Priority candidates (predicted active + pass eNTRy):")
        print(f"  {'Rank':<6} {'Prob':>6} {'True':>6} {'logP':>7} {'TPSA':>7} "
              f"{'RotB':>5} {'Glob':>7} {'SMILES'}")
        print("  " + "-" * 100)
        priority = top[top["priority_candidate"] == 1].copy()
        for _, row in priority.iterrows():
            smi_short = row["smiles"][:50] + "..." if len(row["smiles"]) > 50 else row["smiles"]
            logp_val = f"{row['logp']:.2f}" if "logp" in row else "N/A"
            tpsa_val = f"{row['tpsa']:.1f}" if "tpsa" in row else "N/A"
            rotb_val = f"{int(row['rotatable_bonds'])}" if "rotatable_bonds" in row else "N/A"
            glob_val = f"{row['glob_hergenrother']:.3f}" if "glob_hergenrother" in row else "N/A"
            print(f"  {int(row['rank']):<6} {row['predicted_prob']:>6.3f} "
                  f"{int(row['true_label']):>6} {logp_val:>7} {tpsa_val:>7} "
                  f"{rotb_val:>5} {glob_val:>7}  {smi_short}")

    # Compare eNTRy pass rate: top-N vs rest of test set
    rest = results[results["rank"] > top_n]
    if "passes_eNTRy" in results.columns:
        top_pass_rate  = top["passes_eNTRy"].mean() * 100
        rest_pass_rate = rest["passes_eNTRy"].mean() * 100
        print(f"\n  eNTRy pass rate — top {top_n}: {top_pass_rate:.1f}%  |  "
              f"rest of test set: {rest_pass_rate:.1f}%")

    # ----------------------------------------------------------------------- #
    # 5. Visualisations
    # ----------------------------------------------------------------------- #
    print("\n" + "=" * 70)
    print("STEP 5: Generating figures")
    print("=" * 70)

    # --- Figure 1: Score distribution with eNTRy overlay ---
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: predicted probability distribution, coloured by eNTRy pass
    if "passes_eNTRy" in results.columns:
        pass_mask = results["passes_eNTRy"] == 1
        axes[0].hist(results.loc[~pass_mask, "predicted_prob"], bins=30,
                     alpha=0.6, color="#4C72B0", label="Fails eNTRy")
        axes[0].hist(results.loc[pass_mask,  "predicted_prob"], bins=30,
                     alpha=0.8, color="#DD8452", label="Passes eNTRy")
        axes[0].axvline(results.iloc[top_n-1]["predicted_prob"], color="red",
                        linestyle="--", linewidth=1.5,
                        label=f"Top-{top_n} threshold")
        axes[0].set_xlabel("Predicted Probability (Active)")
        axes[0].set_ylabel("Count")
        axes[0].set_title("Score Distribution by eNTRy Status")
        axes[0].legend(fontsize=9)
        axes[0].grid(alpha=0.3)

    # Right: logP vs TPSA for top-N, coloured by eNTRy + true label
    if "logp" in top.columns and "tpsa" in top.columns:
        for _, row in top.iterrows():
            color  = "#E74C3C" if row.get("passes_eNTRy", 0) == 1 else "#4C72B0"
            marker = "★" if row["true_label"] == 1 else "o"
            size   = 120 if row["true_label"] == 1 else 60
            axes[1].scatter(row["logp"], row["tpsa"],
                            c=color, marker="*" if row["true_label"] == 1 else "o",
                            s=size, alpha=0.75, linewidths=0.5, edgecolors="white")

        # Reference lines
        axes[1].axhline(TPSA_THRESHOLD, color="grey", linestyle="--",
                        linewidth=1, label=f"TPSA = {TPSA_THRESHOLD} Å²")
        axes[1].axvline(LOGP_MIN, color="grey", linestyle=":", linewidth=1)
        axes[1].axvline(LOGP_MAX, color="grey", linestyle=":",
                        linewidth=1, label=f"logP = [{LOGP_MIN}, {LOGP_MAX}]")

        # Legend proxies
        from matplotlib.lines import Line2D
        from matplotlib.patches import Patch
        handles = [
            Patch(facecolor="#E74C3C", label="Passes eNTRy"),
            Patch(facecolor="#4C72B0", label="Fails eNTRy"),
            Line2D([0],[0], marker="*", color="w", markerfacecolor="k",
                   markersize=10, label="True active"),
            Line2D([0],[0], marker="o", color="w", markerfacecolor="k",
                   markersize=8,  label="True inactive"),
        ]
        axes[1].legend(handles=handles, fontsize=8, loc="upper right")
        axes[1].set_xlabel("logP")
        axes[1].set_ylabel("TPSA (Å²)")
        axes[1].set_title(f"logP vs TPSA — Top {top_n} Predictions\n"
                          "(★ = true active, red = passes eNTRy)")
        axes[1].grid(alpha=0.3)

    plt.tight_layout()
    fig1_path = f"{args.outdir}/figures/entry_filter_top{top_n}.png"
    fig.savefig(fig1_path, dpi=150)
    plt.close()
    print(f"  Saved: {fig1_path}")

    # --- Figure 2: Cumulative precision vs eNTRy pass rate across ranks ---
    ranks = np.arange(1, len(results) + 1)
    cum_precision = np.cumsum(results["true_label"].values) / ranks
    if "passes_eNTRy" in results.columns:
        cum_entry_rate = np.cumsum(results["passes_eNTRy"].values) / ranks

        fig2, ax = plt.subplots(figsize=(8, 5))
        ax.plot(ranks, cum_precision,   color="#E74C3C", linewidth=2,
                label="Cumulative Precision (true actives)")
        ax.plot(ranks, cum_entry_rate,  color="#4C72B0", linewidth=2,
                linestyle="--", label="Cumulative eNTRy pass rate")
        ax.axhline(y_test.mean(), color="grey", linestyle=":",
                   linewidth=1, label=f"Dataset positive rate ({y_test.mean():.3f})")
        ax.axvline(top_n, color="black", linestyle="--", linewidth=1,
                   alpha=0.5, label=f"Top-{top_n} cutoff")
        ax.set_xlabel("Rank (by predicted probability, highest first)")
        ax.set_ylabel("Rate")
        ax.set_title("Cumulative Precision and eNTRy Pass Rate vs Rank")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        ax.set_xlim(0, len(results))

        plt.tight_layout()
        fig2_path = f"{args.outdir}/figures/cumulative_precision_entry.png"
        fig2.savefig(fig2_path, dpi=150)
        plt.close()
        print(f"  Saved: {fig2_path}")

    # ----------------------------------------------------------------------- #
    # 6. Save outputs
    # ----------------------------------------------------------------------- #
    print("\n" + "=" * 70)
    print("STEP 6: Saving outputs")
    print("=" * 70)

    # Full ranked table
    ranked_path = f"{args.outdir}/ranked_predictions.csv"
    results.to_csv(ranked_path, index=False)
    print(f"  Saved: {ranked_path}  ({len(results)} compounds)")

    # Priority candidates only
    if "priority_candidate" in results.columns:
        priority_path = f"{args.outdir}/priority_candidates.csv"
        priority_df = results[results["priority_candidate"] == 1].copy()
        priority_df.to_csv(priority_path, index=False)
        print(f"  Saved: {priority_path}  ({len(priority_df)} compounds)")

    # Summary JSON
    summary = {
        "top_n": top_n,
        "test_set_size": len(y_test),
        "test_set_true_actives": int(y_test.sum()),
        "model_auprc": round(float(auprc), 4),
        "top_n_analysis": {
            "true_actives_in_top_n": n_true_active_in_top,
            "precision_at_n": round(n_true_active_in_top / top_n, 4),
            "passing_entry": int(n_pass_entry) if isinstance(n_pass_entry, (int, np.integer)) else None,
            "priority_candidates": int(n_priority) if isinstance(n_priority, (int, np.integer)) else None,
        }
    }
    with open(f"{args.outdir}/results_phase6c.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved: {args.outdir}/results_phase6c.json")

    print("\n" + "=" * 70)
    print("Phase 6C complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
