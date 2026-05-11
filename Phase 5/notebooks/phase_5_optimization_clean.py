# -*- coding: utf-8 -*-
"""
================================================================================
Phase 5 - Model Optimization
Predicting Antibiotic Activity from Chemical Structures
================================================================================

This single script performs the full Phase 5 optimization pipeline:

    1. Featurization        : ECFP4 binary + ECFP4 count + MACCS + RDKit 2D
                              descriptors (4480 features per molecule)
    2. Scaffold split       : Stratified Bemis-Murcko scaffold split that
                              guarantees positive samples in both train and test
    3. Hyperparameter search: Optuna TPE search per base learner, scored on
                              AUPRC (the metric that matters for imbalanced data)
    4. Out-of-fold preds    : 5-fold OOF predictions for each base learner,
                              used to train a stacking meta-learner
    5. Stacking ensemble    : Logistic regression meta-learner over base models
    6. Evaluation           : AUPRC + ROC-AUC + Brier + bootstrap 95% CIs +
                              top-K precision + calibration on the held-out
                              scaffold test set
    7. Interpretability     : SHAP feature importances on the tuned XGBoost

Phase 4 baseline:    AUPRC = 0.3632, ROC-AUC = 0.7925
Literature target:   AUPRC ~ 0.45  (Lin et al. 2025, Sci. Rep.)

Input  (working dir):
    smiles_labels_valid.csv       (from Phase 2: SMILES + binary labels)

Outputs (./results_phase5/):
    data/X_features.npy           (n_molecules x 4480 feature matrix)
    data/y.npy                    (binary labels)
    data/scaffold_split.npz       (train/test indices)
    data/feature_index.json       (which columns belong to which fp block)
    data/descriptor_scaler.joblib (StandardScaler for RDKit descriptors)
    data/oof_predictions.npz      (out-of-fold and test predictions)

    studies/study_<model>.pkl     (full Optuna search history)

    models/best_<model>.{json,joblib}
    models/stacking_model.joblib  (meta-learner)

    figures/pr_curves.png         (Precision-Recall on test set)
    figures/calibration.png       (probability calibration plot)
    figures/shap_top_features.png (top 30 features by mean |SHAP|)

    results.json                  (all metrics and best hyperparameters)

Author: HZI antibiotic discovery project
================================================================================
"""

# =============================================================================
# 0. Imports and configuration
# =============================================================================
import os, json, time, joblib, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import MACCSkeys, rdFingerprintGenerator, Descriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.ML.Descriptors.MoleculeDescriptors import MolecularDescriptorCalculator

from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score, roc_auc_score, precision_recall_curve,
    brier_score_loss,
)

import xgboost as xgb
import lightgbm as lgb
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

warnings.filterwarnings("ignore")
RDLogger.DisableLog("rdApp.*")
optuna.logging.set_verbosity(optuna.logging.WARNING)

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

# ---- Configuration ----------------------------------------------------------
RNG       = 42                  # global random seed for reproducibility
N_TRIALS  = 200                 # Optuna trials per heavy base learner
CV_FOLDS  = 5                   # cross-validation folds
N_BOOT    = 1000                # bootstrap iterations for AUPRC CIs
USE_GPU   = True                # GPU-accelerated XGBoost / LightGBM
N_JOBS    = -1                  # use all CPU cores
TOPK_LIST = [10, 20, 50]        # top-K precision values to report

np.random.seed(RNG)

# ---- Output directory layout -----------------------------------------------
OUT      = Path("results_phase5")
DATA     = OUT / "data"
STUDIES  = OUT / "studies"
MODELS   = OUT / "models"
FIGURES  = OUT / "figures"
for d in (OUT, DATA, STUDIES, MODELS, FIGURES):
    d.mkdir(parents=True, exist_ok=True)


# =============================================================================
# STEP 1: Featurization
#         Convert each SMILES into a numerical feature vector by concatenating
#         four complementary molecular representations:
#           - ECFP4 binary (radius=2, 2048 bits) : presence of substructures
#           - ECFP4 count  (radius=2, 2048 bits) : frequency of substructures
#           - MACCS keys   (167 bits)            : curated structural alerts
#           - RDKit 2D     (~210 descriptors)    : global physicochemistry
# =============================================================================
print("=" * 72)
print("STEP 1: Featurization")
print("=" * 72)

df = pd.read_csv("smiles_labels_valid.csv")
print(f"Loaded {len(df)} valid SMILES from Phase 2")

morgan_bin = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
morgan_cnt = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

DESC_NAMES = [name for name, _ in Descriptors._descList]
desc_calc  = MolecularDescriptorCalculator(DESC_NAMES)


def featurize(smiles: str):
    """Convert one SMILES string to its concatenated feature vector."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # ECFP4 binary
    ecfp_b = np.zeros(2048, dtype=np.float32)
    DataStructs.ConvertToNumpyArray(morgan_bin.GetFingerprint(mol), ecfp_b)

    # ECFP4 count (folded into the same 2048 bins)
    ecfp_c = np.zeros(2048, dtype=np.float32)
    for idx, cnt in morgan_cnt.GetCountFingerprint(mol).GetNonzeroElements().items():
        ecfp_c[idx] = cnt

    # MACCS keys (167 structural fingerprints)
    maccs = np.zeros(167, dtype=np.float32)
    DataStructs.ConvertToNumpyArray(MACCSkeys.GenMACCSKeys(mol), maccs)

    # RDKit 2D descriptors (~210 physicochemical properties)
    desc_vals = np.array(desc_calc.CalcDescriptors(mol), dtype=np.float32)

    return np.concatenate([ecfp_b, ecfp_c, maccs, desc_vals])


# Build feature matrix; drop molecules whose descriptors are not finite
X_rows, y_rows, smi_rows = [], [], []
for smi, lbl in zip(df["SMILES"], df["label"]):
    v = featurize(smi)
    if v is None:
        continue
    if not np.all(np.isfinite(v)):
        v[~np.isfinite(v)] = 0.0   # rare descriptor blow-ups -> zero
    X_rows.append(v)
    y_rows.append(lbl)
    smi_rows.append(smi)

X = np.vstack(X_rows).astype(np.float32)
y = np.asarray(y_rows, dtype=np.int8)
df = pd.DataFrame({"SMILES": smi_rows, "label": y})

# Track which columns belong to which fingerprint block (used by SHAP later)
n_eb, n_ec, n_m, n_d = 2048, 2048, 167, len(DESC_NAMES)
feat_blocks = {
    "ECFP4_bin":   [0,                 n_eb],
    "ECFP4_count": [n_eb,              n_eb + n_ec],
    "MACCS":       [n_eb + n_ec,       n_eb + n_ec + n_m],
    "RDKit2D":     [n_eb + n_ec + n_m, n_eb + n_ec + n_m + n_d],
}

print(f"X shape: {X.shape}  (ECFP_bin {n_eb} + ECFP_count {n_ec} "
      f"+ MACCS {n_m} + RDKit2D {n_d})")
print(f"y shape: {y.shape}  positives: {int(y.sum())} ({y.mean()*100:.2f}%)")

np.save(DATA / "X_features.npy", X)
np.save(DATA / "y.npy", y)
with open(DATA / "feature_index.json", "w") as f:
    json.dump(feat_blocks, f, indent=2)


# =============================================================================
# STEP 2: Stratified scaffold split
#         A simple scaffold split can leave the test set with zero positives
#         when the dataset is highly imbalanced. We avoid that by:
#           1. Sorting scaffolds into "contains positives" vs "all negatives"
#           2. Sending ~20% of positive-containing scaffolds to test first
#           3. Filling test up to 20% of total with negative-only scaffolds
#         Scaffold separation between train and test is fully preserved.
# =============================================================================
print("\n" + "=" * 72)
print("STEP 2: Stratified scaffold split")
print("=" * 72)

def murcko(smi: str) -> str:
    m = Chem.MolFromSmiles(smi)
    return MurckoScaffold.MurckoScaffoldSmiles(mol=m) if m else ""

df["scaffold"] = df["SMILES"].apply(murcko)

# Group molecule indices by scaffold
scaffold_groups = {}
for i, s in enumerate(df["scaffold"]):
    scaffold_groups.setdefault(s, []).append(i)

# Separate into positive-containing and negative-only scaffolds
pos_scaffolds, neg_scaffolds = [], []
for scaf, idxs in scaffold_groups.items():
    if any(y[i] == 1 for i in idxs):
        pos_scaffolds.append(idxs)
    else:
        neg_scaffolds.append(idxs)

rng = np.random.default_rng(RNG)
rng.shuffle(pos_scaffolds)
rng.shuffle(neg_scaffolds)

n_total      = len(df)
test_target  = int(n_total * 0.20)
pos_test_tgt = int(len(pos_scaffolds) * 0.20)

train_idx, test_idx = [], []
for grp in pos_scaffolds[:pos_test_tgt]:
    test_idx.extend(grp)
for grp in pos_scaffolds[pos_test_tgt:]:
    train_idx.extend(grp)
for grp in neg_scaffolds:
    if len(test_idx) < test_target:
        test_idx.extend(grp)
    else:
        train_idx.extend(grp)

train_idx = np.array(sorted(train_idx))
test_idx  = np.array(sorted(test_idx))

X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

# Standardize the descriptor block (LR benefits; tree models indifferent)
scaler   = StandardScaler()
desc_cols = slice(*feat_blocks["RDKit2D"])
X_train_s = X_train.copy()
X_test_s  = X_test.copy()
X_train_s[:, desc_cols] = scaler.fit_transform(X_train[:, desc_cols])
X_test_s[:,  desc_cols] = scaler.transform(X_test[:, desc_cols])

# Sanity: zero scaffold leakage between train and test
overlap = set(df.iloc[train_idx]["scaffold"]) & set(df.iloc[test_idx]["scaffold"])
overlap.discard("")  # empty scaffolds (acyclic molecules) are allowed everywhere
assert not overlap, f"Scaffold leakage detected: {overlap}"

print(f"Train: {len(train_idx)}  positives: {int(y_train.sum())} "
      f"({y_train.mean()*100:.2f}%)")
print(f"Test : {len(test_idx)}  positives: {int(y_test.sum())} "
      f"({y_test.mean()*100:.2f}%)")
print(f"Scaffold leakage: 0  ✓")

np.savez(DATA / "scaffold_split.npz", train_idx=train_idx, test_idx=test_idx)
joblib.dump(scaler, DATA / "descriptor_scaler.joblib")

SPW = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
print(f"scale_pos_weight = {SPW:.2f}")


# =============================================================================
# STEP 3: Hyperparameter search per base learner (Optuna TPE)
#         Each model is searched independently with the same scoring metric
#         (average_precision = AUPRC, the metric we ultimately care about).
#         We use 5-fold CV on the training set only - the test set is
#         strictly held out.
# =============================================================================
print("\n" + "=" * 72)
print(f"STEP 3: Optuna search ({N_TRIALS} trials/heavy-model, {CV_FOLDS}-fold CV)")
print("=" * 72)

cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RNG)

def cv_auprc(model_factory, X_, y_):
    """Generic 5-fold CV with AUPRC. model_factory() returns a fresh estimator."""
    scores = []
    for i_tr, i_va in cv.split(X_, y_):
        m = model_factory()
        m.fit(X_[i_tr], y_[i_tr])
        p = m.predict_proba(X_[i_va])[:, 1]
        scores.append(average_precision_score(y_[i_va], p))
    return float(np.mean(scores))


# ---- 3a. XGBoost ------------------------------------------------------------
def xgb_objective(trial):
    p = {
        "max_depth":        trial.suggest_int("max_depth", 3, 12),
        "learning_rate":    trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
        "n_estimators":     trial.suggest_int("n_estimators", 200, 2500, step=100),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "reg_lambda":       trial.suggest_float("reg_lambda", 1e-2, 100.0, log=True),
        "reg_alpha":        trial.suggest_float("reg_alpha", 1e-3, 10.0,  log=True),
        "gamma":            trial.suggest_float("gamma",      1e-3, 5.0,  log=True),
        "tree_method":      "hist",
    }
    if USE_GPU:
        p["device"] = "cuda"

    fold_scores = []
    for fold, (i_tr, i_va) in enumerate(cv.split(X_train, y_train)):
        m = xgb.XGBClassifier(scale_pos_weight=SPW, random_state=RNG,
                               eval_metric="aucpr", **p)
        m.fit(X_train[i_tr], y_train[i_tr],
              eval_set=[(X_train[i_va], y_train[i_va])], verbose=False)
        prob = m.predict_proba(X_train[i_va])[:, 1]
        fold_scores.append(average_precision_score(y_train[i_va], prob))
        trial.report(np.mean(fold_scores), fold)
        if trial.should_prune():
            raise optuna.TrialPruned()
    return float(np.mean(fold_scores))


print("\n[XGBoost] starting search ...")
t0 = time.time()
study_xgb = optuna.create_study(
    direction="maximize", sampler=TPESampler(seed=RNG),
    pruner=MedianPruner(n_startup_trials=20, n_warmup_steps=2),
    study_name="xgb",
)
study_xgb.optimize(xgb_objective, n_trials=N_TRIALS, show_progress_bar=False)
print(f"[XGBoost] best CV AUPRC = {study_xgb.best_value:.4f}  "
      f"(t = {time.time()-t0:.0f}s)")
joblib.dump(study_xgb, STUDIES / "study_xgb.pkl")

best_xgb_params = dict(study_xgb.best_params)
best_xgb_params["tree_method"] = "hist"
if USE_GPU:
    best_xgb_params["device"] = "cuda"


# ---- 3b. LightGBM -----------------------------------------------------------
def lgb_objective(trial):
    p = {
        "num_leaves":        trial.suggest_int("num_leaves", 15, 255),
        "max_depth":         trial.suggest_int("max_depth", -1, 12),
        "learning_rate":     trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
        "n_estimators":      trial.suggest_int("n_estimators", 200, 2500, step=100),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
        "subsample":         trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "reg_lambda":        trial.suggest_float("reg_lambda", 1e-3, 100.0, log=True),
        "reg_alpha":         trial.suggest_float("reg_alpha",  1e-3, 10.0,  log=True),
    }
    if USE_GPU:
        p["device"] = "gpu"

    fold_scores = []
    for fold, (i_tr, i_va) in enumerate(cv.split(X_train, y_train)):
        m = lgb.LGBMClassifier(class_weight="balanced", random_state=RNG,
                                objective="binary", verbose=-1, **p)
        m.fit(X_train[i_tr], y_train[i_tr])
        prob = m.predict_proba(X_train[i_va])[:, 1]
        fold_scores.append(average_precision_score(y_train[i_va], prob))
        trial.report(np.mean(fold_scores), fold)
        if trial.should_prune():
            raise optuna.TrialPruned()
    return float(np.mean(fold_scores))


print("\n[LightGBM] starting search ...")
t0 = time.time()
study_lgb = optuna.create_study(
    direction="maximize", sampler=TPESampler(seed=RNG),
    pruner=MedianPruner(n_startup_trials=20, n_warmup_steps=2),
    study_name="lgb",
)
study_lgb.optimize(lgb_objective, n_trials=N_TRIALS, show_progress_bar=False)
print(f"[LightGBM] best CV AUPRC = {study_lgb.best_value:.4f}  "
      f"(t = {time.time()-t0:.0f}s)")
joblib.dump(study_lgb, STUDIES / "study_lgb.pkl")

best_lgb_params = dict(study_lgb.best_params)
if USE_GPU:
    best_lgb_params["device"] = "gpu"


# ---- 3c. Random Forest ------------------------------------------------------
def rf_objective(trial):
    p = {
        "n_estimators":      trial.suggest_int("n_estimators", 200, 1500, step=100),
        "max_depth":         trial.suggest_int("max_depth", 5, 40),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf":  trial.suggest_int("min_samples_leaf", 1, 10),
        "max_features":      trial.suggest_categorical("max_features", ["sqrt", "log2", 0.3, 0.5]),
    }
    return cv_auprc(
        lambda: RandomForestClassifier(**p, class_weight="balanced",
                                        n_jobs=N_JOBS, random_state=RNG),
        X_train, y_train,
    )

print("\n[RandomForest] starting search ...")
t0 = time.time()
study_rf = optuna.create_study(direction="maximize", sampler=TPESampler(seed=RNG),
                                study_name="rf")
study_rf.optimize(rf_objective, n_trials=80, show_progress_bar=False)
print(f"[RandomForest] best CV AUPRC = {study_rf.best_value:.4f}  "
      f"(t = {time.time()-t0:.0f}s)")
joblib.dump(study_rf, STUDIES / "study_rf.pkl")
best_rf_params = dict(study_rf.best_params)


# ---- 3d. Logistic Regression ------------------------------------------------
# Note: liblinear + L2 is the only safe combination for wide (~4500) feature
# matrices. The saga + L1 combination hangs indefinitely on this dataset.
def lr_objective(trial):
    C = trial.suggest_float("C", 1e-3, 1e2, log=True)
    return cv_auprc(
        lambda: LogisticRegression(
            C=C, penalty="l2", solver="liblinear",
            class_weight="balanced", max_iter=1000, tol=1e-3,
            random_state=RNG,
        ),
        X_train_s, y_train,                     # standardised features for LR
    )

print("\n[LogisticRegression] starting search ...")
t0 = time.time()
study_lr = optuna.create_study(direction="maximize", sampler=TPESampler(seed=RNG),
                                study_name="lr")
study_lr.optimize(lr_objective, n_trials=30, timeout=1800, show_progress_bar=False)
print(f"[LogisticRegression] best CV AUPRC = {study_lr.best_value:.4f}  "
      f"(t = {time.time()-t0:.0f}s)")
joblib.dump(study_lr, STUDIES / "study_lr.pkl")
best_lr_params = dict(study_lr.best_params)


# =============================================================================
# STEP 4: Out-of-fold predictions for stacking
#         For each base model:
#           - Generate honest OOF probabilities on the training set
#             (each fold's predictions come from a model that did NOT see it)
#           - Refit the model on the FULL training set and predict the test set
# =============================================================================
print("\n" + "=" * 72)
print("STEP 4: Out-of-fold predictions")
print("=" * 72)

def make_xgb():
    return xgb.XGBClassifier(scale_pos_weight=SPW, random_state=RNG,
                              eval_metric="aucpr", **best_xgb_params)
def make_lgb():
    return lgb.LGBMClassifier(class_weight="balanced", random_state=RNG,
                               objective="binary", verbose=-1, **best_lgb_params)
def make_rf():
    return RandomForestClassifier(**best_rf_params, class_weight="balanced",
                                   n_jobs=N_JOBS, random_state=RNG)
def make_lr():
    return LogisticRegression(**best_lr_params, penalty="l2",
                               solver="liblinear", class_weight="balanced",
                               max_iter=1000, tol=1e-3, random_state=RNG)

# (name, factory, train features to use, test features to use)
base_specs = [
    ("xgb", make_xgb, X_train,   X_test),
    ("lgb", make_lgb, X_train,   X_test),
    ("rf",  make_rf,  X_train,   X_test),
    ("lr",  make_lr,  X_train_s, X_test_s),
]

oof_preds, test_preds = {}, {}
for name, factory, Xtr, Xte in base_specs:
    print(f"  fitting {name} ...")
    oof = np.zeros(len(y_train), dtype=np.float32)
    for i_tr, i_va in cv.split(Xtr, y_train):
        m = factory()
        m.fit(Xtr[i_tr], y_train[i_tr])
        oof[i_va] = m.predict_proba(Xtr[i_va])[:, 1]

    # Refit on full training set for the final test prediction
    m_full = factory()
    m_full.fit(Xtr, y_train)
    test_p = m_full.predict_proba(Xte)[:, 1]

    oof_preds[name]  = oof
    test_preds[name] = test_p

    auprc_oof = average_precision_score(y_train, oof)
    auprc_te  = average_precision_score(y_test,  test_p)
    print(f"    {name}  OOF AUPRC = {auprc_oof:.4f}   TEST AUPRC = {auprc_te:.4f}")

    # Persist refitted model
    if name == "xgb":
        m_full.save_model(MODELS / "best_xgb.json")
    else:
        joblib.dump(m_full, MODELS / f"best_{name}.joblib")

np.savez(DATA / "oof_predictions.npz",
         **{f"oof_{k}":  v for k, v in oof_preds.items()},
         **{f"test_{k}": v for k, v in test_preds.items()},
         y_train=y_train, y_test=y_test)


# =============================================================================
# STEP 5: Stacking meta-learner
#         A simple logistic regression takes the four OOF probability columns
#         as input and learns to combine them. Its coefficients reveal which
#         base model was most useful.
# =============================================================================
print("\n" + "=" * 72)
print("STEP 5: Stacking meta-learner")
print("=" * 72)

base_names = [n for n, _, _, _ in base_specs]
Z_train = np.column_stack([oof_preds[n]  for n in base_names])
Z_test  = np.column_stack([test_preds[n] for n in base_names])

meta = LogisticRegression(C=1.0, class_weight="balanced", max_iter=5000,
                           random_state=RNG)
meta.fit(Z_train, y_train)
p_stack = meta.predict_proba(Z_test)[:, 1]

print("Meta-learner coefficients:")
for n, c in zip(base_names, meta.coef_.ravel()):
    print(f"  {n:>4s}: {c:+.3f}")

joblib.dump(meta, MODELS / "stacking_model.joblib")


# =============================================================================
# STEP 6: Test-set evaluation
#         Reports for every model:
#           - AUPRC, ROC-AUC, Brier (probability calibration)
#           - Bootstrap 95% CI on AUPRC (1000 resamples)
#           - Top-K precision (= fraction of true actives in the top K
#             predictions). Most relevant for real-world prioritisation.
# =============================================================================
print("\n" + "=" * 72)
print("STEP 6: Test-set evaluation")
print("=" * 72)

def bootstrap_auprc(y_true, y_prob, n_boot=N_BOOT, seed=RNG):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    scores = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if y_true[idx].sum() == 0:    # need at least one positive in resample
            continue
        scores.append(average_precision_score(y_true[idx], y_prob[idx]))
    lo, hi = np.percentile(scores, [2.5, 97.5])
    return float(np.mean(scores)), float(lo), float(hi)

def topk_precision(y_true, y_prob, k):
    order = np.argsort(-y_prob)[:k]
    return float(y_true[order].mean())

all_probs = {**test_preds, "stack": p_stack}
results = {}
for name, prob in all_probs.items():
    auprc       = average_precision_score(y_test, prob)
    rocauc      = roc_auc_score(y_test, prob)
    brier       = brier_score_loss(y_test, prob)
    mean, lo, hi = bootstrap_auprc(y_test, prob)
    topk = {f"P@{k}": topk_precision(y_test, prob, k) for k in TOPK_LIST}
    results[name] = {
        "AUPRC": auprc, "ROC_AUC": rocauc, "Brier": brier,
        "AUPRC_boot_mean": mean, "AUPRC_95CI": [lo, hi],
        **topk,
    }

print(f"{'Model':<8}{'AUPRC':>8}{'95% CI':>20}{'ROC-AUC':>10}{'Brier':>9}"
      f"{'P@10':>8}{'P@20':>8}{'P@50':>8}")
for n, r in results.items():
    print(f"{n:<8}{r['AUPRC']:>8.4f}  [{r['AUPRC_95CI'][0]:.3f},{r['AUPRC_95CI'][1]:.3f}]"
          f"{r['ROC_AUC']:>10.4f}{r['Brier']:>9.4f}"
          f"{r['P@10']:>8.2f}{r['P@20']:>8.2f}{r['P@50']:>8.2f}")

# ---- Precision-Recall curves -----------------------------------------------
plt.figure(figsize=(8, 6))
for n, prob in all_probs.items():
    pr, rc, _ = precision_recall_curve(y_test, prob)
    plt.plot(rc, pr, label=f"{n}  AUPRC={results[n]['AUPRC']:.3f}")
plt.axhline(y_test.mean(), ls="--", c="grey",
            label=f"random ({y_test.mean():.3f})")
plt.xlabel("Recall"); plt.ylabel("Precision")
plt.title("Phase 5 - Precision-Recall on scaffold-split test set")
plt.legend(loc="upper right"); plt.grid(alpha=0.3); plt.tight_layout()
plt.savefig(FIGURES / "pr_curves.png", dpi=150); plt.close()

# ---- Calibration plot -------------------------------------------------------
plt.figure(figsize=(7, 6))
for n, prob in all_probs.items():
    frac_pos, mean_pred = calibration_curve(y_test, prob, n_bins=10, strategy="quantile")
    plt.plot(mean_pred, frac_pos, marker="o", label=n)
plt.plot([0, 1], [0, 1], "--", c="grey", label="perfect")
plt.xlabel("Mean predicted probability"); plt.ylabel("Fraction of positives")
plt.title("Calibration (quantile bins)"); plt.legend(); plt.grid(alpha=0.3)
plt.tight_layout(); plt.savefig(FIGURES / "calibration.png", dpi=150); plt.close()


# =============================================================================
# STEP 7: SHAP feature importances on tuned XGBoost
#         Identifies the substructures and physicochemical descriptors that
#         drive the model's activity predictions. Useful for chemists.
# =============================================================================
if HAS_SHAP:
    print("\n" + "=" * 72)
    print("STEP 7: SHAP feature importances (XGBoost)")
    print("=" * 72)
    try:
        m_full = xgb.XGBClassifier()
        m_full.load_model(MODELS / "best_xgb.json")
        explainer   = shap.TreeExplainer(m_full)
        shap_values = explainer.shap_values(X_test)
        mean_abs    = np.abs(shap_values).mean(axis=0)
        top = np.argsort(mean_abs)[-30:][::-1]

        def block_of(idx):
            """Map a feature index back to which fingerprint block it came from."""
            for blk, (a, b) in feat_blocks.items():
                if a <= idx < b:
                    if blk == "RDKit2D":
                        return f"{blk}:{DESC_NAMES[idx - a]}"
                    return f"{blk}[{idx - a}]"
            return str(idx)

        plt.figure(figsize=(8, 9))
        plt.barh(range(len(top))[::-1], mean_abs[top])
        plt.yticks(range(len(top))[::-1], [block_of(i) for i in top])
        plt.xlabel("Mean |SHAP value|")
        plt.title("Top 30 features driving activity prediction (XGBoost)")
        plt.tight_layout()
        plt.savefig(FIGURES / "shap_top_features.png", dpi=150); plt.close()
        print(f"  saved -> {FIGURES/'shap_top_features.png'}")
    except Exception as e:
        print(f"  SHAP step failed: {e}")
else:
    print("\n[SHAP] not installed - skipping importance plot.")


# =============================================================================
# STEP 8: Persist consolidated results
# =============================================================================
summary = {
    "phase4_baseline":  {"AUPRC": 0.3632, "ROC_AUC": 0.7925},
    "literature_target":{"AUPRC": 0.45,   "source": "Lin et al. 2025"},
    "feature_dim":      int(X.shape[1]),
    "n_train":          int(len(y_train)),
    "n_test":           int(len(y_test)),
    "scale_pos_weight": float(SPW),
    "best_params": {
        "xgb": study_xgb.best_params,
        "lgb": study_lgb.best_params,
        "rf":  study_rf.best_params,
        "lr":  study_lr.best_params,
    },
    "results": results,
}
with open(OUT / "results.json", "w") as f:
    json.dump(summary, f, indent=2)

print("\n" + "=" * 72)
print(f"Done. All outputs in {OUT}/")
print("=" * 72)
