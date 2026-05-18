# -*- coding: utf-8 -*-
"""
Phase 6B — eNTRy Descriptors Only (Ablation)
=============================================

Trains XGBoost and Logistic Regression using ONLY the 10 eNTRy/3D
descriptors as features. This isolates exactly how much predictive
signal the eNTRy rules carry on their own, independent of fingerprints.

Uses the exact same scaffold split as Phase 5/6 for direct comparison.

Usage
-----
python phase_6b_entry_only.py \
    --labels   results_phase5/data/y.npy \
    --split    results_phase5/data/scaffold_split.npz \
    --entry-csv entry_descriptors.csv \
    --outdir   results_phase6
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

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, roc_auc_score,
    precision_recall_curve, brier_score_loss,
)
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from xgboost import XGBClassifier

RNG = 42
np.random.seed(RNG)

ENTRY_COLS = [
    "has_primary_amine",
    "rotatable_bonds",
    "glob_hergenrother",
    "spherocity",
    "asphericity",
    "npr1",
    "npr2",
    "radius_of_gyration",
    "pbf",
    "passes_eNTRy",
]

XGB_PARAMS = dict(
    max_depth=7, learning_rate=0.036, n_estimators=1300,
    subsample=0.74, colsample_bytree=0.44,
    random_state=RNG, tree_method="hist", eval_metric="aucpr",
)

RF_PARAMS = dict(
    n_estimators=800, max_depth=23, min_samples_split=4,
    max_features="sqrt", random_state=RNG, n_jobs=-1,
)


def bootstrap_auprc(y_true, y_score, n=1000, seed=42):
    rng = np.random.RandomState(seed)
    scores = []
    for _ in range(n):
        idx = rng.choice(len(y_true), len(y_true), replace=True)
        if y_true[idx].sum() == 0:
            continue
        scores.append(average_precision_score(y_true[idx], y_score[idx]))
    return np.percentile(scores, [2.5, 97.5])


def precision_at_k(y_true, y_score, k):
    top_k = np.argsort(y_score)[::-1][:k]
    return y_true[top_k].sum() / k


def evaluate(y_true, y_score, label=""):
    auprc  = average_precision_score(y_true, y_score)
    rocauc = roc_auc_score(y_true, y_score)
    brier  = brier_score_loss(y_true, y_score)
    ci_lo, ci_hi = bootstrap_auprc(y_true, y_score)
    p10 = precision_at_k(y_true, y_score, 10)
    p20 = precision_at_k(y_true, y_score, 20)
    print(f"  {label:<30} AUPRC={auprc:.4f} [{ci_lo:.2f},{ci_hi:.2f}]  "
          f"ROC-AUC={rocauc:.4f}  Brier={brier:.3f}  "
          f"P@10={p10:.2f}  P@20={p20:.2f}")
    return dict(auprc=auprc, ci_lo=ci_lo, ci_hi=ci_hi,
                rocauc=rocauc, brier=brier, p10=p10, p20=p20)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--labels",    required=True)
    p.add_argument("--split",     required=True)
    p.add_argument("--entry-csv", required=True)
    p.add_argument("--outdir",    default="results_phase6")
    args = p.parse_args()

    os.makedirs(f"{args.outdir}/figures", exist_ok=True)

    # ----------------------------------------------------------------------- #
    # Load data
    # ----------------------------------------------------------------------- #
    print("=" * 70)
    print("Loading data")
    print("=" * 70)

    y     = np.load(args.labels)
    split = np.load(args.split)
    train_idx = split["train_idx"]
    test_idx  = split["test_idx"]

    entry_df = pd.read_csv(args.entry_csv)
    cols = [c for c in ENTRY_COLS if c in entry_df.columns]
    X_entry = entry_df[cols].fillna(entry_df[cols].median()).values.astype(np.float32)

    y_train = y[train_idx]
    y_test  = y[test_idx]

    # Scale (fit on train only)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_entry[train_idx])
    X_test  = scaler.transform(X_entry[test_idx])

    spw = float((y_train == 0).sum()) / float((y_train == 1).sum())

    print(f"Feature matrix : {X_entry.shape}  (eNTRy only, {len(cols)} cols)")
    print(f"Train          : {len(y_train)}  positives={y_train.sum()}")
    print(f"Test           : {len(y_test)}   positives={y_test.sum()}")
    print(f"Columns used   : {cols}")

    # ----------------------------------------------------------------------- #
    # Train and evaluate
    # ----------------------------------------------------------------------- #
    print("\n" + "=" * 70)
    print("Model evaluation — eNTRy descriptors ONLY")
    print("=" * 70)

    results = {}
    pr_curves = {}

    # XGBoost
    xgb = XGBClassifier(**XGB_PARAMS, scale_pos_weight=spw)
    xgb.fit(X_train, y_train)
    prob_xgb = xgb.predict_proba(X_test)[:, 1]
    results["XGBoost"] = evaluate(y_test, prob_xgb, "XGBoost")
    prec, rec, _ = precision_recall_curve(y_test, prob_xgb)
    pr_curves["XGBoost"] = (rec, prec, results["XGBoost"]["auprc"])

    # Random Forest
    rf = RandomForestClassifier(**RF_PARAMS, class_weight={0:1, 1:int(spw)})
    rf.fit(X_train, y_train)
    prob_rf = rf.predict_proba(X_test)[:, 1]
    results["RandomForest"] = evaluate(y_test, prob_rf, "RandomForest")
    prec, rec, _ = precision_recall_curve(y_test, prob_rf)
    pr_curves["RandomForest"] = (rec, prec, results["RandomForest"]["auprc"])

    # Logistic Regression
    lr = LogisticRegression(C=0.077, penalty="l2", solver="liblinear",
                            class_weight="balanced", random_state=RNG,
                            max_iter=1000)
    lr.fit(X_train, y_train)
    prob_lr = lr.predict_proba(X_test)[:, 1]
    results["LogisticRegression"] = evaluate(y_test, prob_lr, "LogisticRegression")
    prec, rec, _ = precision_recall_curve(y_test, prob_lr)
    pr_curves["LogisticRegression"] = (rec, prec, results["LogisticRegression"]["auprc"])

    # ----------------------------------------------------------------------- #
    # Full comparison table (reference values from Phase 5 and 6)
    # ----------------------------------------------------------------------- #
    print("\n" + "=" * 70)
    print("Full ablation summary")
    print("=" * 70)

    # Reference numbers from Phase 5 and Phase 6 runs
    reference = {
        "Phase 4 baseline (ECFP only)":       {"auprc": 0.6125},
        "Phase 5 best (4480-dim, tuned)":      {"auprc": 0.6460},
        "Phase 6 extended (4490-dim)":         {"auprc": 0.6085},  # ensemble
    }

    print(f"\n  {'Experiment':<45} {'AUPRC':>8}")
    print("  " + "-" * 55)
    for name, d in reference.items():
        print(f"  {name:<45} {d['auprc']:>8.4f}")
    print("  " + "-" * 55)
    for model, d in results.items():
        label = f"eNTRy only — {model}"
        print(f"  {label:<45} {d['auprc']:>8.4f}")
    baseline_rate = y_test.mean()
    print(f"  {'Random baseline':<45} {baseline_rate:>8.4f}")

    # ----------------------------------------------------------------------- #
    # PR curve plot
    # ----------------------------------------------------------------------- #
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = {"XGBoost": "#E07B39", "RandomForest": "#56A15E",
               "LogisticRegression": "#9B59B6"}
    for model, (rec, prec, auprc) in pr_curves.items():
        ax.plot(rec, prec, label=f"{model} ({auprc:.3f})",
                color=colors[model], linewidth=2)
    ax.axhline(baseline_rate, color="black", linestyle="--",
               linewidth=1, label=f"Random ({baseline_rate:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("PR Curves — eNTRy Descriptors Only (10 features)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(f"{args.outdir}/figures/pr_curves_entry_only.png", dpi=150)
    plt.close()
    print(f"\n  Saved: {args.outdir}/figures/pr_curves_entry_only.png")

    # ----------------------------------------------------------------------- #
    # Save results
    # ----------------------------------------------------------------------- #
    out = {
        "experiment": "eNTRy descriptors only (10 features)",
        "features_used": cols,
        "n_features": len(cols),
        "test_set": {"n_total": int(len(y_test)), "n_positive": int(y_test.sum())},
        "results": {k: {m: round(v, 4) for m, v in d.items()}
                    for k, d in results.items()},
    }
    out_path = f"{args.outdir}/results_phase6b.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  Saved: {out_path}")

    print("\n" + "=" * 70)
    print("Phase 6B complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
