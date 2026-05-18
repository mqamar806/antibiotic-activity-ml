# -*- coding: utf-8 -*-
"""
Phase 6 — eNTRy Descriptor Integration & Model Evaluation
==========================================================

What this script does
---------------------
1. Loads the Phase 5 feature matrix (X_features.npy, 4480-dim) and the
   exact same scaffold split indices used in Phase 5 (scaffold_split.npz).
2. Loads the new eNTRy/3D descriptors from entry_descriptors.csv and
   appends the 9 genuinely new columns to the feature matrix → 4489-dim.
3. Retrains the best models from Phase 5 (tuned XGBoost, LightGBM,
   Random Forest, Stacking Ensemble) using the best hyperparameters found
   in Phase 5 — no re-tuning needed, same params, just new features.
4. Evaluates on the identical held-out test set and compares AUPRC.
5. Runs SHAP on the tuned XGBoost to see where the new features rank.
6. Runs two biological validation analyses:
   - eNTRy pass rate enrichment in actives vs inactives
   - Correlation between glob_hergenrother and spherocity

Usage (HPC)
-----------
python phase_6_model.py \
    --features   results_phase5/data/X_features.npy \
    --labels     results_phase5/data/y.npy \
    --split      results_phase5/data/scaffold_split.npz \
    --entry-csv  entry_descriptors.csv \
    --outdir     results_phase6/

Dependencies
------------
    pip install xgboost lightgbm scikit-learn shap matplotlib pandas numpy joblib
"""

import argparse
import json
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for HPC
import matplotlib.pyplot as plt
import shap

from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_recall_curve,
    brier_score_loss,
)
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample

from xgboost import XGBClassifier
import lightgbm as lgb

RNG = 42
np.random.seed(RNG)

# --------------------------------------------------------------------------- #
# Best hyperparameters from Phase 5 Optuna search — copied directly
# --------------------------------------------------------------------------- #
XGB_PARAMS = dict(
    max_depth=7,
    learning_rate=0.036,
    n_estimators=1300,
    subsample=0.74,
    colsample_bytree=0.44,
    random_state=RNG,
    tree_method="hist",
    eval_metric="aucpr",
)

LGB_PARAMS = dict(
    num_leaves=25,
    max_depth=7,
    learning_rate=0.0077,
    n_estimators=1000,
    min_child_samples=26,
    random_state=RNG,
    verbose=-1,
)

RF_PARAMS = dict(
    n_estimators=800,
    max_depth=23,
    min_samples_split=4,
    max_features="sqrt",
    random_state=RNG,
    n_jobs=-1,
)

LR_PARAMS = dict(
    C=0.077,
    penalty="l2",
    solver="liblinear",
    random_state=RNG,
    max_iter=1000,
)

# The 9 new columns from entry_descriptors.csv that don't exist in X_features
ENTRY_COLS = [
    "has_primary_amine",
    "rotatable_bonds",      # Hergenrother definition (excludes amides/rings)
    "glob_hergenrother",
    "spherocity",
    "asphericity",
    "npr1",
    "npr2",
    "radius_of_gyration",
    "pbf",
    "passes_eNTRy",         # binary derived flag
]


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def bootstrap_auprc(y_true, y_score, n=1000, seed=42):
    """95% CI on AUPRC via bootstrap resampling."""
    rng = np.random.RandomState(seed)
    scores = []
    for _ in range(n):
        idx = rng.choice(len(y_true), len(y_true), replace=True)
        if y_true[idx].sum() == 0:
            continue
        scores.append(average_precision_score(y_true[idx], y_score[idx]))
    lo, hi = np.percentile(scores, [2.5, 97.5])
    return lo, hi


def precision_at_k(y_true, y_score, k):
    top_k = np.argsort(y_score)[::-1][:k]
    return y_true[top_k].sum() / k


def evaluate(y_true, y_score, label=""):
    auprc = average_precision_score(y_true, y_score)
    rocauc = roc_auc_score(y_true, y_score)
    brier = brier_score_loss(y_true, y_score)
    ci_lo, ci_hi = bootstrap_auprc(y_true, y_score)
    p10 = precision_at_k(y_true, y_score, 10)
    p20 = precision_at_k(y_true, y_score, 20)
    print(f"  {label:<30} AUPRC={auprc:.4f} [{ci_lo:.2f},{ci_hi:.2f}]  "
          f"ROC-AUC={rocauc:.4f}  Brier={brier:.3f}  "
          f"P@10={p10:.2f}  P@20={p20:.2f}")
    return dict(auprc=auprc, ci_lo=ci_lo, ci_hi=ci_hi,
                rocauc=rocauc, brier=brier, p10=p10, p20=p20)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--features",  required=True,
                   help="Phase 5 feature matrix: results_phase5/data/X_features.npy")
    p.add_argument("--labels",    required=True,
                   help="Label vector: results_phase5/data/y.npy")
    p.add_argument("--split",     required=True,
                   help="Scaffold split indices: results_phase5/data/scaffold_split.npz")
    p.add_argument("--entry-csv", required=True,
                   help="Descriptor output from Phase 6 Step 1: entry_descriptors.csv")
    p.add_argument("--outdir",    default="results_phase6",
                   help="Output directory (default: results_phase6/)")
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    os.makedirs(f"{args.outdir}/figures", exist_ok=True)
    os.makedirs(f"{args.outdir}/models",  exist_ok=True)

    # ----------------------------------------------------------------------- #
    # 1. Load Phase 5 data
    # ----------------------------------------------------------------------- #
    print("=" * 70)
    print("STEP 1: Loading Phase 5 features and split")
    print("=" * 70)

    X_phase5 = np.load(args.features)
    y        = np.load(args.labels)
    split    = np.load(args.split)
    train_idx = split["train_idx"]
    test_idx  = split["test_idx"]

    print(f"Phase 5 feature matrix : {X_phase5.shape}")
    print(f"Labels                 : {y.shape}  positives={y.sum()}")
    print(f"Train indices          : {len(train_idx)}  positives={y[train_idx].sum()}")
    print(f"Test  indices          : {len(test_idx)}   positives={y[test_idx].sum()}")

    # ----------------------------------------------------------------------- #
    # 2. Load eNTRy descriptors and build extended feature matrix
    # ----------------------------------------------------------------------- #
    print("\n" + "=" * 70)
    print("STEP 2: Appending eNTRy descriptors")
    print("=" * 70)

    entry_df = pd.read_csv(args.entry_csv)
    print(f"entry_descriptors.csv  : {entry_df.shape}")

    # Verify row alignment with the feature matrix
    assert len(entry_df) == X_phase5.shape[0], (
        f"Row mismatch: entry_descriptors has {len(entry_df)} rows "
        f"but X_features has {X_phase5.shape[0]}"
    )

    # Use only columns that exist in the file
    cols_to_add = [c for c in ENTRY_COLS if c in entry_df.columns]
    missing = [c for c in ENTRY_COLS if c not in entry_df.columns]
    if missing:
        print(f"  Warning: columns not found in entry CSV, skipping: {missing}")

    # Fill NaN from the 1 failed 3D embedding with column medians
    entry_block = entry_df[cols_to_add].copy()
    n_nan = entry_block.isna().sum().sum()
    if n_nan > 0:
        print(f"  Filling {n_nan} NaN values with column medians")
        entry_block = entry_block.fillna(entry_block.median())
    entry_arr = entry_block.values.astype(np.float32)

    # Scale the new features (fit on train only, apply to both)
    scaler_entry = StandardScaler()
    entry_arr[train_idx] = scaler_entry.fit_transform(entry_arr[train_idx])
    entry_arr[test_idx]  = scaler_entry.transform(entry_arr[test_idx])

    X_extended = np.hstack([X_phase5, entry_arr])
    print(f"Extended feature matrix: {X_extended.shape}  "
          f"(+{len(cols_to_add)} eNTRy columns)")
    print(f"Columns added: {cols_to_add}")

    X_train_p5 = X_phase5[train_idx];  X_test_p5 = X_phase5[test_idx]
    X_train_ex = X_extended[train_idx]; X_test_ex = X_extended[test_idx]
    y_train = y[train_idx];             y_test    = y[test_idx]

    # ----------------------------------------------------------------------- #
    # 3. Biological validation — eNTRy enrichment analysis
    # ----------------------------------------------------------------------- #
    print("\n" + "=" * 70)
    print("STEP 3: eNTRy biological validation")
    print("=" * 70)

    # Use raw (unscaled) values for the biological analysis
    raw_entry = entry_df[cols_to_add].fillna(entry_df[cols_to_add].median())
    raw_entry["label"] = y

    if "passes_eNTRy" in raw_entry.columns:
        n_actives   = (y == 1).sum()
        n_inactives = (y == 0).sum()
        pass_active   = ((raw_entry["passes_eNTRy"] == 1) & (y == 1)).sum()
        pass_inactive = ((raw_entry["passes_eNTRy"] == 1) & (y == 0)).sum()
        print(f"  Actives   passing eNTRy: {pass_active}/{n_actives} "
              f"({100*pass_active/n_actives:.1f}%)")
        print(f"  Inactives passing eNTRy: {pass_inactive}/{n_inactives} "
              f"({100*pass_inactive/n_inactives:.1f}%)")

    # Glob vs spherocity scatter — correlation between the two definitions
    if "glob_hergenrother" in raw_entry.columns and "spherocity" in raw_entry.columns:
        valid = raw_entry[["glob_hergenrother", "spherocity", "label"]].dropna()
        corr = valid["glob_hergenrother"].corr(valid["spherocity"])
        print(f"  Pearson correlation (glob_hergenrother vs spherocity): {corr:.4f}")

        fig, ax = plt.subplots(figsize=(6, 5))
        colors = valid["label"].map({0: "#4C72B0", 1: "#DD8452"})
        ax.scatter(valid["glob_hergenrother"], valid["spherocity"],
                   c=colors, alpha=0.35, s=12, linewidths=0)
        # Legend proxies
        from matplotlib.lines import Line2D
        handles = [
            Line2D([0],[0], marker='o', color='w', markerfacecolor='#4C72B0',
                   markersize=8, label='Inactive'),
            Line2D([0],[0], marker='o', color='w', markerfacecolor='#DD8452',
                   markersize=8, label='Active'),
        ]
        ax.legend(handles=handles)
        ax.set_xlabel("Globularity (Hergenrother PCA)")
        ax.set_ylabel("Spherocity Index (RDKit)")
        ax.set_title(f"Globularity vs Spherocity  (r = {corr:.3f})")
        ax.axvline(0.25, color="red", linestyle="--", linewidth=1,
                   label="eNTRy threshold (glob ≤ 0.25)")
        ax.legend(handles=handles + [
            Line2D([0],[0], color='red', linestyle='--', label='eNTRy glob threshold')
        ])
        plt.tight_layout()
        plt.savefig(f"{args.outdir}/figures/glob_vs_spherocity.png", dpi=150)
        plt.close()
        print(f"  Saved: {args.outdir}/figures/glob_vs_spherocity.png")

    # ----------------------------------------------------------------------- #
    # 4. Train and evaluate all models — Phase 5 features vs Extended features
    # ----------------------------------------------------------------------- #
    print("\n" + "=" * 70)
    print("STEP 4: Model training and evaluation")
    print("=" * 70)

    spw = float((y_train == 0).sum()) / float((y_train == 1).sum())
    print(f"  scale_pos_weight = {spw:.2f}")

    all_results = {}
    pr_data = {}   # store curves for combined plot

    def run_experiment(label, X_tr, X_te):
        """Train all 4 models + ensemble, return results dict."""
        print(f"\n  --- {label} ---")
        results = {}
        preds   = {}

        # XGBoost
        xgb = XGBClassifier(**XGB_PARAMS, scale_pos_weight=spw)
        xgb.fit(X_tr, y_train)
        preds["xgb"] = xgb.predict_proba(X_te)[:, 1]
        results["XGBoost"] = evaluate(y_test, preds["xgb"], "XGBoost")

        # LightGBM
        lgbm = lgb.LGBMClassifier(**LGB_PARAMS,
                                   class_weight={0: 1, 1: int(spw)})
        lgbm.fit(X_tr, y_train)
        preds["lgb"] = lgbm.predict_proba(X_te)[:, 1]
        results["LightGBM"] = evaluate(y_test, preds["lgb"], "LightGBM")

        # Random Forest
        rf = RandomForestClassifier(**RF_PARAMS,
                                    class_weight={0: 1, 1: int(spw)})
        rf.fit(X_tr, y_train)
        preds["rf"] = rf.predict_proba(X_te)[:, 1]
        results["RandomForest"] = evaluate(y_test, preds["rf"], "RandomForest")

        # Logistic Regression (needs scaling)
        sc = StandardScaler()
        X_tr_sc = sc.fit_transform(X_tr.astype(np.float32))
        X_te_sc = sc.transform(X_te.astype(np.float32))
        lr = LogisticRegression(**LR_PARAMS, class_weight="balanced")
        lr.fit(X_tr_sc, y_train)
        preds["lr"] = lr.predict_proba(X_te_sc)[:, 1]
        results["LogisticRegression"] = evaluate(y_test, preds["lr"],
                                                  "LogisticRegression")

        # Stacking ensemble (simple average — keeps it comparable to Phase 5)
        stack_score = np.mean([preds["xgb"], preds["lgb"],
                               preds["rf"],  preds["lr"]], axis=0)
        results["Ensemble"] = evaluate(y_test, stack_score, "Ensemble (avg)")
        preds["ensemble"]   = stack_score

        # PR curves for plotting
        for name, prob in preds.items():
            prec, rec, _ = precision_recall_curve(y_test, prob)
            pr_data[f"{label}_{name}"] = (rec, prec,
                                           results.get(name.upper(),
                                           results.get("Ensemble", {}))
                                           .get("auprc", 0))
        return results, xgb  # return XGBoost for SHAP

    res_p5, xgb_p5 = run_experiment("Phase5_features",   X_train_p5, X_test_p5)
    res_ex, xgb_ex = run_experiment("Extended_features", X_train_ex, X_test_ex)

    # ----------------------------------------------------------------------- #
    # 5. Comparison table
    # ----------------------------------------------------------------------- #
    print("\n" + "=" * 70)
    print("STEP 5: Comparison — Phase 5 vs Extended (eNTRy added)")
    print("=" * 70)
    print(f"  {'Model':<22} {'Phase5 AUPRC':>14} {'Extended AUPRC':>15} {'Delta':>8}")
    print("  " + "-" * 62)
    for model in ["XGBoost", "LightGBM", "RandomForest",
                  "LogisticRegression", "Ensemble"]:
        a5 = res_p5.get(model, {}).get("auprc", float("nan"))
        ax = res_ex.get(model, {}).get("auprc", float("nan"))
        delta = ax - a5
        sign = "+" if delta >= 0 else ""
        print(f"  {model:<22} {a5:>14.4f} {ax:>15.4f} {sign}{delta:>7.4f}")

    # ----------------------------------------------------------------------- #
    # 6. PR curve comparison plot
    # ----------------------------------------------------------------------- #
    print("\n" + "=" * 70)
    print("STEP 6: Plotting PR curves")
    print("=" * 70)

    # Plot: Phase 5 ensemble vs Extended ensemble
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: all models for extended features
    colors_map = {
        "xgb": "#E07B39", "lgb": "#4C9BE8", "rf": "#56A15E",
        "lr": "#9B59B6", "ensemble": "#E74C3C"
    }
    labels_map = {
        "xgb": "XGBoost", "lgb": "LightGBM", "rf": "Random Forest",
        "lr": "Logistic Regression", "ensemble": "Ensemble"
    }
    for key, (rec, prec, auprc) in pr_data.items():
        if not key.startswith("Extended"):
            continue
        model_key = key.replace("Extended_features_", "")
        lw = 2.5 if model_key == "ensemble" else 1.5
        axes[0].plot(rec, prec,
                     label=f"{labels_map.get(model_key, model_key)} ({auprc:.3f})",
                     color=colors_map.get(model_key, "gray"),
                     linewidth=lw)
    baseline = y_test.mean()
    axes[0].axhline(baseline, color="black", linestyle="--", linewidth=1,
                    label=f"Random ({baseline:.3f})")
    axes[0].set_xlabel("Recall"); axes[0].set_ylabel("Precision")
    axes[0].set_title("Phase 6 — Extended Features (all models)")
    axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)

    # Right: ensemble head-to-head Phase 5 vs Extended
    for tag, color, lw in [("Phase5_features", "#4C72B0", 2),
                             ("Extended_features", "#DD8452", 2.5)]:
        key = f"{tag}_ensemble"
        if key in pr_data:
            rec, prec, auprc = pr_data[key]
            lbl = ("Phase 5 (4480-dim)" if tag == "Phase5_features"
                   else "Phase 6 Extended (4489-dim)")
            axes[1].plot(rec, prec, label=f"{lbl}  AUPRC={auprc:.4f}",
                         color=color, linewidth=lw)
    axes[1].axhline(baseline, color="black", linestyle="--", linewidth=1,
                    label=f"Random ({baseline:.3f})")
    axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision")
    axes[1].set_title("Ensemble: Phase 5 vs Phase 6 (eNTRy added)")
    axes[1].legend(fontsize=9); axes[1].grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(f"{args.outdir}/figures/pr_curves_phase6.png", dpi=150)
    plt.close()
    print(f"  Saved: {args.outdir}/figures/pr_curves_phase6.png")

    # ----------------------------------------------------------------------- #
    # 7. SHAP — where do the eNTRy features rank?
    # ----------------------------------------------------------------------- #
    print("\n" + "=" * 70)
    print("STEP 7: SHAP feature importance (Extended XGBoost)")
    print("=" * 70)

    # Build feature names: Phase 5 uses index numbers, new ones are named
    n_p5_feats = X_phase5.shape[1]
    feature_names = [f"feat_{i}" for i in range(n_p5_feats)] + cols_to_add

    explainer = shap.TreeExplainer(xgb_ex)
    # Use a sample for speed — 500 test + 500 train is sufficient
    sample_idx = np.random.choice(len(X_train_ex),
                                  min(500, len(X_train_ex)), replace=False)
    X_sample = np.vstack([X_train_ex[sample_idx], X_test_ex])
    shap_values = explainer.shap_values(X_sample)

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    sorted_idx = np.argsort(mean_abs_shap)[::-1]

    print("\n  Top 30 features by mean |SHAP|:")
    print(f"  {'Rank':<6} {'Feature':<35} {'Mean |SHAP|':>12}")
    print("  " + "-" * 55)
    entry_col_set = set(cols_to_add)
    for rank, fidx in enumerate(sorted_idx[:30], 1):
        fname = feature_names[fidx]
        is_new = "  ← eNTRy" if fname in entry_col_set else ""
        print(f"  {rank:<6} {fname:<35} {mean_abs_shap[fidx]:>12.5f}{is_new}")

    # SHAP bar plot — top 30
    top30_idx   = sorted_idx[:30]
    top30_names = [feature_names[i] for i in top30_idx]
    top30_shap  = mean_abs_shap[top30_idx]
    colors_bar  = ["#DD8452" if n in entry_col_set else "#4C72B0"
                   for n in top30_names]

    fig, ax = plt.subplots(figsize=(8, 9))
    bars = ax.barh(range(len(top30_names)), top30_shap[::-1],
                   color=colors_bar[::-1])
    ax.set_yticks(range(len(top30_names)))
    ax.set_yticklabels(top30_names[::-1], fontsize=8)
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title("Top 30 Features by SHAP Importance\n"
                 "(orange = new eNTRy/3D descriptor)")
    # Add legend
    from matplotlib.patches import Patch
    legend_elems = [Patch(facecolor="#4C72B0", label="Phase 5 features"),
                    Patch(facecolor="#DD8452", label="eNTRy/3D (Phase 6 new)")]
    ax.legend(handles=legend_elems, loc="lower right", fontsize=8)
    plt.tight_layout()
    fig.savefig(f"{args.outdir}/figures/shap_phase6.png", dpi=150)
    plt.close()
    print(f"\n  Saved: {args.outdir}/figures/shap_phase6.png")

    # Ranks of eNTRy features specifically
    print("\n  eNTRy/3D feature ranks in top features:")
    rank_lookup = {feature_names[fidx]: rank
                   for rank, fidx in enumerate(sorted_idx, 1)}
    for col in cols_to_add:
        if col in rank_lookup:
            print(f"    {col:<35} rank {rank_lookup[col]} / {len(feature_names)}")

    # ----------------------------------------------------------------------- #
    # 8. Save results JSON
    # ----------------------------------------------------------------------- #
    print("\n" + "=" * 70)
    print("STEP 8: Saving results")
    print("=" * 70)

    output = {
        "phase5_features_dim": int(X_phase5.shape[1]),
        "extended_features_dim": int(X_extended.shape[1]),
        "new_entry_columns": cols_to_add,
        "test_set": {
            "n_total": int(len(y_test)),
            "n_positive": int(y_test.sum()),
        },
        "results_phase5_features": {k: {m: round(v, 4) for m, v in d.items()}
                                     for k, d in res_p5.items()},
        "results_extended_features": {k: {m: round(v, 4) for m, v in d.items()}
                                       for k, d in res_ex.items()},
        "entry_shap_ranks": {col: rank_lookup.get(col, -1) for col in cols_to_add},
    }

    with open(f"{args.outdir}/results_phase6.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"  Saved: {args.outdir}/results_phase6.json")

    # Save extended feature matrix for any future phases
    np.save(f"{args.outdir}/X_extended.npy", X_extended)
    print(f"  Saved: {args.outdir}/X_extended.npy")

    print("\n" + "=" * 70)
    print("Phase 6 complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
