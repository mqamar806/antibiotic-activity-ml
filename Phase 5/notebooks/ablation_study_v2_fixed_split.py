# -*- coding: utf-8 -*-
"""
================================================================================
Ablation Study — Isolating the Source of Improvement (Phase 4 → Phase 5)
Predicting Antibiotic Activity from Chemical Structures
================================================================================

MOTIVATION
----------
Phase 4 achieved AUPRC = 0.36.
Phase 5 achieved AUPRC = 0.64.
Three things changed at once: features, scaffold split strategy, and model.
This script runs four controlled experiments to isolate the contribution
of each change individually.

EXPERIMENTS
-----------
  All four experiments use the SAME Phase 5 stratified scaffold split
  (468 test molecules, 20 positives). This is the only way to make a
  fair comparison — if the test set differs between experiments, any
  AUPRC difference could be due to which molecules were tested, not
  the model or features.

  Exp A — Phase 4 model on the corrected split (true baseline)
           Features : ECFP4 binary (2048)        <- Phase 4 style
           Split    : Phase 5 stratified split   <- same for ALL experiments
           Model    : Default XGBoost (no tuning)
           Purpose  : What does the Phase 4 model score on a valid,
                      comparable test set?

  Exp B — same as A (sanity / reproducibility check)
           Features : ECFP4 binary (2048)
           Split    : Phase 5 stratified split
           Model    : Default XGBoost (no tuning)

  Exp C — features change only
           Features : Combined 4480              <- changed
           Split    : Phase 5 stratified split
           Model    : Default XGBoost (no tuning)
           Purpose  : How much do the extra features contribute?

  Exp D — full Phase 5
           Features : Combined 4480
           Split    : Phase 5 stratified split
           Model    : Tuned XGBoost (Optuna best params)
           Purpose  : How much does tuning add on top of features?

HOW TO INTERPRET
----------------
  A == B  -> confirmed reproducible (identical inputs, sanity check)
  A -> C  -> improvement purely from combined features
  C -> D  -> additional improvement from hyperparameter tuning
  A -> D  -> total real improvement of Phase 5 over Phase 4 model

Input  (working dir):
    smiles_labels_valid.csv        (from Phase 2)
    results_phase5/studies/study_xgb.pkl  (Phase 5 best params, optional)

Outputs (./results_ablation/):
    ablation_results.json
    ablation_pr_curves.png
    ablation_summary.txt
================================================================================
"""

import json, time, joblib, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import MACCSkeys, rdFingerprintGenerator, Descriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.ML.Descriptors.MoleculeDescriptors import MolecularDescriptorCalculator

from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_curve
import xgboost as xgb

warnings.filterwarnings("ignore")
RDLogger.DisableLog("rdApp.*")

# --- Config ------------------------------------------------------------------
RNG     = 42
USE_GPU = True
OUT     = Path("results_ablation_v2")
OUT.mkdir(exist_ok=True)
np.random.seed(RNG)

print("=" * 72)
print("Ablation Study: Phase 4 → Phase 5 Improvement")
print("=" * 72)


# =============================================================================
# 1. Build BOTH feature sets once
#    Set A: ECFP4 binary only  (2048 dims)  — Phase 4 style
#    Set B: Combined 4480      (all types)  — Phase 5 style
# =============================================================================
print("\n>>> Building feature sets ...")

df = pd.read_csv("smiles_labels_valid.csv")
print(f"    Loaded {len(df)} SMILES")

morgan_bin = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
morgan_cnt = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
DESC_NAMES = [name for name, _ in Descriptors._descList]
desc_calc  = MolecularDescriptorCalculator(DESC_NAMES)


def ecfp4_only(smiles):
    """Phase 4 feature: ECFP4 binary, 2048 bits."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    arr = np.zeros(2048, dtype=np.float32)
    DataStructs.ConvertToNumpyArray(morgan_bin.GetFingerprint(mol), arr)
    return arr


def combined_features(smiles):
    """Phase 5 feature: ECFP4 binary + ECFP4 count + MACCS + RDKit2D."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    ecfp_b = np.zeros(2048, dtype=np.float32)
    DataStructs.ConvertToNumpyArray(morgan_bin.GetFingerprint(mol), ecfp_b)
    ecfp_c = np.zeros(2048, dtype=np.float32)
    for idx, cnt in morgan_cnt.GetCountFingerprint(mol).GetNonzeroElements().items():
        ecfp_c[idx] = cnt
    maccs = np.zeros(167, dtype=np.float32)
    DataStructs.ConvertToNumpyArray(MACCSkeys.GenMACCSKeys(mol), maccs)
    desc = np.array(desc_calc.CalcDescriptors(mol), dtype=np.float32)
    return np.concatenate([ecfp_b, ecfp_c, maccs, desc])


# Build both matrices in one pass
X_ecfp_rows, X_comb_rows, y_rows, smi_rows = [], [], [], []
for smi, lbl in zip(df["SMILES"], df["label"]):
    e = ecfp4_only(smi)
    c = combined_features(smi)
    if e is None or c is None:
        continue
    if not np.all(np.isfinite(c)):
        c[~np.isfinite(c)] = 0.0
    X_ecfp_rows.append(e)
    X_comb_rows.append(c)
    y_rows.append(lbl)
    smi_rows.append(smi)

X_ecfp = np.vstack(X_ecfp_rows).astype(np.float32)   # (n, 2048)
X_comb = np.vstack(X_comb_rows).astype(np.float32)   # (n, 4480)
y      = np.asarray(y_rows, dtype=np.int8)
df     = pd.DataFrame({"SMILES": smi_rows, "label": y})

print(f"    ECFP4-only matrix:  {X_ecfp.shape}")
print(f"    Combined matrix:    {X_comb.shape}")
print(f"    Positives: {int(y.sum())} / {len(y)} ({y.mean()*100:.2f}%)")


# =============================================================================
# 2. Build BOTH splits
#    Split A: Phase 4 style (shuffle all scaffolds, fill test bucket first)
#    Split B: Phase 5 stratified (20% of positive-containing scaffolds → test)
# =============================================================================
print("\n>>> Building split ...")

n_total     = len(df)
test_target = int(n_total * 0.20)
def murcko(smi):
    m = Chem.MolFromSmiles(smi)
    return MurckoScaffold.MurckoScaffoldSmiles(mol=m) if m else ""

df["scaffold"] = df["SMILES"].apply(murcko)

# --- Phase 5 stratified scaffold split --------------------------------------
# Used by ALL experiments so every comparison is on the same 20 test positives.
scaffold_groups_p5 = {}
for i, s in enumerate(df["scaffold"]):
    scaffold_groups_p5.setdefault(s, []).append(i)

pos_scaffolds, neg_scaffolds = [], []
for scaf, idxs in scaffold_groups_p5.items():
    if any(y[i] == 1 for i in idxs):
        pos_scaffolds.append(idxs)
    else:
        neg_scaffolds.append(idxs)

rng_p5 = np.random.default_rng(RNG)
rng_p5.shuffle(pos_scaffolds)
rng_p5.shuffle(neg_scaffolds)

pos_test_tgt = int(len(pos_scaffolds) * 0.20)
train_p5, test_p5 = [], []
for grp in pos_scaffolds[:pos_test_tgt]:
    test_p5.extend(grp)
for grp in pos_scaffolds[pos_test_tgt:]:
    train_p5.extend(grp)
for grp in neg_scaffolds:
    if len(test_p5) < test_target:
        test_p5.extend(grp)
    else:
        train_p5.extend(grp)

train_p5 = np.array(sorted(train_p5))
test_p5  = np.array(sorted(test_p5))

# Print split summaries
print(f"    All experiments: train={len(train_p5)} pos={int(y[train_p5].sum())}  "
      f"test={len(test_p5)} pos={int(y[test_p5].sum())}")


# =============================================================================
# 3. Load Phase 5 best XGBoost params (for Exp D)
#    Falls back to sensible defaults if study file not found.
# =============================================================================
study_path = Path("results_phase5/studies/study_xgb.pkl")
if study_path.exists():
    study = joblib.load(study_path)
    best_p5_params = dict(study.best_params)
    print(f"\n>>> Loaded Phase 5 XGBoost params (CV AUPRC = {study.best_value:.4f})")
else:
    # Sensible defaults if Phase 5 results aren't present
    best_p5_params = {
        "max_depth": 7, "learning_rate": 0.036, "n_estimators": 1300,
        "min_child_weight": 5, "subsample": 0.74, "colsample_bytree": 0.44,
        "reg_lambda": 0.015, "reg_alpha": 0.11, "gamma": 0.20,
    }
    print("\n>>> Phase 5 study not found — using hardcoded best params as fallback")

best_p5_params["tree_method"] = "hist"
if USE_GPU:
    best_p5_params["device"] = "cuda"


# =============================================================================
# 4. Run all four experiments
# =============================================================================
print("\n" + "=" * 72)
print("RUNNING EXPERIMENTS")
print("=" * 72)

def run_experiment(name, description, X_tr, X_te, y_tr, y_te, xgb_params):
    """Train one default or tuned XGBoost, return metrics dict."""
    t0 = time.time()
    spw = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)

    model = xgb.XGBClassifier(
        scale_pos_weight=spw,
        random_state=RNG,
        eval_metric="aucpr",
        **xgb_params,
    )
    model.fit(X_tr, y_tr, verbose=False)
    probs = model.predict_proba(X_te)[:, 1]

    auprc  = average_precision_score(y_te, probs)
    rocauc = roc_auc_score(y_te, probs)
    elapsed = time.time() - t0

    print(f"\n  [{name}] {description}")
    print(f"    Train: {len(y_tr)} samples  positives: {int(y_tr.sum())}")
    print(f"    Test : {len(y_te)} samples  positives: {int(y_te.sum())}")
    print(f"    AUPRC = {auprc:.4f}   ROC-AUC = {rocauc:.4f}   (t={elapsed:.0f}s)")

    pr, rc, _ = precision_recall_curve(y_te, probs)
    return {
        "name":        name,
        "description": description,
        "AUPRC":       round(auprc, 4),
        "ROC_AUC":     round(rocauc, 4),
        "n_train":     int(len(y_tr)),
        "n_test":      int(len(y_te)),
        "test_pos":    int(y_te.sum()),
        "pr_curve":    {"precision": pr.tolist(), "recall": rc.tolist()},
        "probs":       probs,
    }

# Default XGBoost params (Phase 4 style — no tuning)
default_params = {
    "tree_method": "hist",
}
if USE_GPU:
    default_params["device"] = "cuda"

experiments = []

# Exp A — Phase 4 model (ECFP4 + default XGBoost) on the Phase 5 split
# This is the TRUE comparable baseline: same 20 test positives as C and D.
experiments.append(run_experiment(
    name        = "A: Phase 4 model",
    description = "ECFP4 only | Phase 5 split | Default XGBoost",
    X_tr        = X_ecfp[train_p5],
    X_te        = X_ecfp[test_p5],
    y_tr        = y[train_p5],
    y_te        = y[test_p5],
    xgb_params  = default_params,
))

# Exp B — identical to A (sanity check: same inputs must give same output)
experiments.append(run_experiment(
    name        = "B: Sanity check (=A)",
    description = "ECFP4 only | Phase 5 split | Default XGBoost",
    X_tr        = X_ecfp[train_p5],
    X_te        = X_ecfp[test_p5],
    y_tr        = y[train_p5],
    y_te        = y[test_p5],
    xgb_params  = default_params,
))

# Exp C — features change only (on top of fixed split)
experiments.append(run_experiment(
    name        = "C: Features change only",
    description = "Combined 4480 | Phase 5 split | Default XGBoost",
    X_tr        = X_comb[train_p5],
    X_te        = X_comb[test_p5],
    y_tr        = y[train_p5],
    y_te        = y[test_p5],
    xgb_params  = default_params,
))

# Exp D — full Phase 5
experiments.append(run_experiment(
    name        = "D: Full Phase 5",
    description = "Combined 4480 | Phase 5 split | Tuned XGBoost",
    X_tr        = X_comb[train_p5],
    X_te        = X_comb[test_p5],
    y_tr        = y[train_p5],
    y_te        = y[test_p5],
    xgb_params  = best_p5_params,
))


# =============================================================================
# 5. Summary table
# =============================================================================
print("\n" + "=" * 72)
print("ABLATION SUMMARY")
print("=" * 72)

header = f"{'Experiment':<28} {'AUPRC':>8} {'vs prev':>9} {'ROC-AUC':>9} {'Test pos':>10}"
print(header)
print("-" * 72)

prev_auprc = None
summary_lines = [header, "-" * 72]
for exp in experiments:
    delta = f"{exp['AUPRC'] - prev_auprc:+.4f}" if prev_auprc is not None else "baseline"
    line = (f"{exp['name']:<28} {exp['AUPRC']:>8.4f} {delta:>9} "
            f"{exp['ROC_AUC']:>9.4f} {exp['test_pos']:>10}")
    print(line)
    summary_lines.append(line)
    prev_auprc = exp["AUPRC"]

print("-" * 72)
total_gain = experiments[-1]["AUPRC"] - experiments[0]["AUPRC"]
footer = f"Total improvement A → D: {total_gain:+.4f}"
print(footer)
summary_lines.extend(["-" * 72, footer])


# =============================================================================
# 6. PR curve comparison plot
# =============================================================================
colors = ["#e41a1c", "#ff7f00", "#4daf4a", "#984ea3"]
labels_short = ["A: Phase 4 replication", "B: Split only",
                "C: Features only", "D: Full Phase 5"]

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Left: all 4 PR curves together
ax = axes[0]
for exp, col in zip(experiments, colors):
    pr = exp["pr_curve"]["precision"]
    rc = exp["pr_curve"]["recall"]
    ax.plot(rc, pr, color=col,
            label=f"{exp['name']}  AUPRC={exp['AUPRC']:.3f}")
ax.axhline(y[test_p5].mean(), ls="--", c="grey",
           label=f"random ({y[test_p5].mean():.3f})")
ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
ax.set_title("PR Curves — All Four Experiments")
ax.legend(loc="upper right", fontsize=8)
ax.grid(alpha=0.3)

# Right: bar chart of AUPRC per experiment — cleaner visual for a report
ax2 = axes[1]
names  = [e["name"].split(":")[0] for e in experiments]
auprc  = [e["AUPRC"] for e in experiments]
bar_colors = colors
bars = ax2.bar(names, auprc, color=bar_colors, width=0.5, edgecolor="white")
ax2.axhline(0.45, ls="--", c="black", linewidth=1.2, label="Literature target (0.45)")
ax2.axhline(experiments[0]["AUPRC"], ls=":", c="grey", linewidth=1,
            label=f"Phase 4 baseline ({experiments[0]['AUPRC']:.3f})")
for bar, val in zip(bars, auprc):
    ax2.text(bar.get_x() + bar.get_width()/2, val + 0.01,
             f"{val:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
ax2.set_ylim(0, 1.0)
ax2.set_ylabel("AUPRC")
ax2.set_title("AUPRC by Experiment")
ax2.legend(fontsize=8)
ax2.grid(axis="y", alpha=0.3)

plt.suptitle("Ablation Study: Isolating Sources of Improvement",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(OUT / "ablation_pr_curves.png", dpi=150)
plt.close()
print(f"\n>>> Saved figure -> {OUT/'ablation_pr_curves.png'}")


# =============================================================================
# 7. Save results
# =============================================================================
results_out = []
for exp in experiments:
    d = {k: v for k, v in exp.items() if k not in ("pr_curve", "probs")}
    results_out.append(d)

with open(OUT / "ablation_results.json", "w") as f:
    json.dump(results_out, f, indent=2)

with open(OUT / "ablation_summary.txt", "w") as f:
    f.write("\n".join(summary_lines))
    f.write(f"\n\nInterpretation guide:\n")
    f.write("  All experiments use the same 20 test positives (Phase 5 stratified split).\n")
    f.write("  A == B  -> sanity check confirmed: reproducible results\n")
    f.write("  A -> C  -> improvement purely from combined features (4480 vs 2048)\n")
    f.write("  C -> D  -> additional improvement from hyperparameter tuning\n")
    f.write("  A -> D  -> total real Phase 5 improvement over Phase 4 model\n")

print(f">>> Saved results -> {OUT/'ablation_results.json'}")
print(f">>> Saved summary -> {OUT/'ablation_summary.txt'}")
print("\n" + "=" * 72)
print("Done.")
print("=" * 72)
