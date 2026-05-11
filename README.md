# Predicting Antibiotic Activity Against *E. coli* Using Machine Learning

A computational drug-discovery pipeline developed at the **Helmholtz Centre for Infection Research (HZI)** that predicts the antibacterial activity of small molecules against *Escherichia coli* from molecular structure alone.

---

## Overview

Antibiotic resistance is one of the most pressing global health threats. This project builds a machine-learning classifier that takes a compound's SMILES string as input and outputs a binary activity prediction (active / inactive against *E. coli*), enabling rapid *in silico* screening of large chemical libraries.

The pipeline is structured in five phases:

| Phase | Description |
|-------|-------------|
| 1 | Data acquisition & preprocessing |
| 2 | Molecular featurisation (ECFP fingerprints) |
| 3 | Scaffold-based dataset splitting |
| 4 | Model training & evaluation |
| 5 | Model optimisation — feature expansion, hyperparameter tuning, stacking ensemble |

---

## Dataset

- **Source:** Supplementary data (Sheet S1B) from a published antibacterial activity study (`mmc1.xlsx`).
- **Size:** 2,334 compounds after quality filtering.
- **Labels:** Binary — `1` = Active, `0` = Inactive against *E. coli*.
- **Class imbalance:** ~5.1% positive (active) compounds.

---

## Pipeline

### Phase 1 — Data Preprocessing (`Phase 1/`)
- Reads the raw Excel file and extracts SMILES + activity labels.
- Performs quality checks: missing values, duplicate SMILES, class distribution.
- Outputs `ecoli_dataset_cleaned.csv`.

### Phase 2 — Molecular Representation (`Phase 2/`)
- Validates all SMILES strings with RDKit.
- Converts valid SMILES to **ECFP4 (Morgan) fingerprints** — 2,048-bit binary vectors (radius = 2).
- Outputs `X_ecfp2048.npy`, `y.npy`, and `smiles_labels_valid.csv`.

### Phase 3 — Scaffold-Based Splitting (`Phase 3-4/`)
- Generates **Murcko scaffolds** for all compounds.
- Assigns entire scaffold families to either train or test (80 / 20 split) to prevent data leakage from structurally similar molecules.
- Train: 1,868 compounds (100 actives) | Test: 466 compounds (20 actives).

### Phase 4 — Modelling & Evaluation (`Phase 3-4/`)
Two strategies for handling class imbalance are compared:

| Strategy | Model | AUPRC | ROC-AUC |
|----------|-------|-------|---------|
| Baseline | XGBoost + `scale_pos_weight` | 0.363 | 0.793 |
| SMOTE | XGBoost on oversampled train set | — | — |

Evaluation metric of primary interest: **AUPRC** (Area Under the Precision-Recall Curve), which is more informative than ROC-AUC on highly imbalanced datasets.

Precision-recall curves are saved as `AUPRC_Baseline.png` and `AUPRC_SMOTE.png`.

### Phase 5 — Model Optimisation (`Phase 5/`)
Targets the gap between the Phase 4 baseline (AUPRC = 0.363) and the literature benchmark of ~0.45 (Lin et al. 2025, *Sci. Rep.*). Three changes are stacked and their individual contributions isolated via a controlled ablation study:

1. **Expanded features** — ECFP4 binary + ECFP4 count + MACCS keys + RDKit 2D descriptors → **4,480 features** (vs. 2,048 in Phase 4).
2. **Stratified scaffold split** — guarantees active compounds appear in both train and test sets; Phase 4 was at risk of all actives landing in one partition.
3. **Bayesian hyperparameter tuning** — Optuna TPE search optimising AUPRC for XGBoost, LightGBM, Random Forest, and Logistic Regression independently.
4. **Stacking ensemble** — 5-fold out-of-fold predictions from all base learners fed to a Logistic Regression meta-learner.

**Results on the held-out scaffold test set (n = 468, 20 positives):**

| Model | AUPRC | ROC-AUC | Brier | P@10 |
|-------|-------|---------|-------|------|
| Phase 4 baseline | 0.363 | 0.793 | — | — |
| XGBoost (tuned) | 0.629 | 0.911 | 0.029 | 0.80 |
| LightGBM (tuned) | 0.624 | 0.926 | 0.028 | 0.80 |
| Random Forest (tuned) | 0.643 | 0.922 | 0.027 | 0.90 |
| **Stacking ensemble** | **0.646** | **0.919** | 0.104 | **0.90** |

**Ablation (all experiments on the same test split):**

| Experiment | AUPRC | vs. baseline |
|------------|-------|-------------|
| A — Phase 4 model, corrected split | 0.613 | — |
| C — expanded features only | 0.624 | +0.012 |
| D — expanded features + tuning | 0.629 | +0.016 |

Outputs include PR curves, a calibration plot, SHAP top-30 feature importances, and a full Optuna study per model.
SHAP analysis confirms that ECFP4 count bits and specific RDKit descriptors drive most predictive signal.

---

## Repository Structure

```
.
├── Phase 1/
│   ├── phase_1_data_preprocessing.py        # Data loading, labelling, QC
│   ├── ecoli_dataset_cleaned.csv             # Cleaned SMILES + labels
│   └── Phase_1_Dataset_Acquisition_Report.docx
├── Phase 2/
│   ├── molecular_representation.py          # SMILES → ECFP fingerprints
│   ├── Molecular_Representation.ipynb
│   ├── smiles_labels_valid.csv               # Filtered SMILES + labels
│   └── Phase_2_Molecular_Representation_Report.docx
├── Phase 3-4/
│   ├── Data_Preparation.ipynb               # Splitting, modelling, evaluation
│   ├── Input/                               # smiles_labels_valid.csv (npy files excluded)
│   ├── AUPRC_Baseline.png
│   ├── AUPRC_SMOTE.png
│   └── Phase_3&4_Report.docx
├── Phase 5/
│   ├── notebooks/
│   │   ├── phase_5_optimization_clean.py    # Full optimisation pipeline
│   │   ├── ablation_study_v2_fixed_split.py # Controlled ablation experiments
│   │   ├── submit_phase5.sh                 # HPC job submission script
│   │   └── submit_ablation_v2.sh
│   ├── results_phase5/
│   │   ├── figures/
│   │   │   ├── pr_curves.png                # PR curves for all models
│   │   │   ├── calibration.png              # Probability calibration plot
│   │   │   └── shap_top_features.png        # Top-30 SHAP feature importances
│   │   ├── results_ablation_v2/
│   │   │   ├── ablation_pr_curves.png
│   │   │   └── ablation_summary.txt         # Ablation AUPRC table
│   │   └── results.json                     # All metrics and best hyperparameters
│   └── Phase_5_Report_v2.docx
└── README.md
```

> **Note:** `.npy` feature matrices and `.xlsx` source files are excluded from version control via `.gitignore`. Store them externally (e.g., Google Drive, Zenodo) or use Git LFS.

---

## Requirements

```
rdkit
numpy
pandas
scikit-learn
xgboost
lightgbm
imbalanced-learn
optuna
shap
matplotlib
joblib
```

Install with:

```bash
pip install rdkit numpy pandas scikit-learn xgboost lightgbm imbalanced-learn optuna shap matplotlib joblib
```

The notebooks were originally developed in **Google Colab**; remove the `google.colab` file-upload cells when running locally.

---

## Key Design Decisions

- **Scaffold split over random split** — prevents over-optimistic evaluation caused by structurally similar train/test compounds; better reflects real-world prospective screening performance.
- **AUPRC as primary metric** — preferred over accuracy and ROC-AUC for imbalanced binary classification in drug discovery.
- **ECFP4 fingerprints** — fast, interpretable, and well-validated for QSAR modelling; a natural baseline before graph neural networks.
- **Combined feature set (Phase 5)** — stacking ECFP4 binary, ECFP4 count, MACCS, and RDKit 2D descriptors provides complementary signal; count bits capture frequency information lost in binary hashing.
- **Optuna TPE for tuning** — tree-structured Parzen estimator efficiently explores high-dimensional hyperparameter spaces; AUPRC (not accuracy) is used as the tuning objective.
- **Stacking ensemble** — out-of-fold predictions prevent target leakage into the meta-learner without needing a separate validation fold.

---

## Affiliation

Developed as part of a computational biology project at the  
**Helmholtz Centre for Infection Research (HZI)**, Braunschweig, Germany.

---

## License

This project is for research and educational purposes. See individual data sources for their respective usage terms.
