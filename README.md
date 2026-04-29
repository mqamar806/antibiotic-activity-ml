# Predicting Antibiotic Activity Against *E. coli* Using Machine Learning

A computational drug-discovery pipeline developed at the **Helmholtz Centre for Infection Research (HZI)** that predicts the antibacterial activity of small molecules against *Escherichia coli* from molecular structure alone.

---

## Overview

Antibiotic resistance is one of the most pressing global health threats. This project builds a machine-learning classifier that takes a compound's SMILES string as input and outputs a binary activity prediction (active / inactive against *E. coli*), enabling rapid *in silico* screening of large chemical libraries.

The pipeline is structured in four phases:

| Phase | Description |
|-------|-------------|
| 1 | Data acquisition & preprocessing |
| 2 | Molecular featurisation (ECFP fingerprints) |
| 3 | Scaffold-based dataset splitting |
| 4 | Model training & evaluation |

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
| Baseline | XGBoost + `scale_pos_weight` | 0.363 | — |
| SMOTE | XGBoost on oversampled train set | — | — |

Evaluation metric of primary interest: **AUPRC** (Area Under the Precision-Recall Curve), which is more informative than ROC-AUC on highly imbalanced datasets.

Precision-recall curves are saved as `AUPRC_Baseline.png` and `AUPRC_SMOTE.png`.

---

## Repository Structure

```
.
├── Phase 1/
│   ├── phase_1_data_preprocessing.py   # Data loading, labelling, QC
│   ├── ecoli_dataset_cleaned.csv        # Cleaned SMILES + labels
│   └── Phase_1_Dataset_Acquisition_Report.docx
├── Phase 2/
│   ├── molecular_representation.py     # SMILES → ECFP fingerprints
│   ├── Molecular_Representation.ipynb
│   ├── smiles_labels_valid.csv          # Filtered SMILES + labels
│   └── Phase_2_Molecular_Representation_Report.docx
├── Phase 3-4/
│   ├── Data_Preparation.ipynb          # Splitting, modelling, evaluation
│   ├── Input/                          # Copies of X, y, and SMILES files
│   ├── AUPRC_Baseline.png
│   ├── AUPRC_SMOTE.png
│   └── Phase_3&4_Report.docx
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
imbalanced-learn
matplotlib
```

Install with:

```bash
pip install rdkit numpy pandas scikit-learn xgboost imbalanced-learn matplotlib
```

The notebooks were originally developed in **Google Colab**; remove the `google.colab` file-upload cells when running locally.

---

## Key Design Decisions

- **Scaffold split over random split** — prevents over-optimistic evaluation caused by structurally similar train/test compounds; better reflects real-world prospective screening performance.
- **AUPRC as primary metric** — preferred over accuracy and ROC-AUC for imbalanced binary classification in drug discovery.
- **ECFP4 fingerprints** — fast, interpretable, and well-validated for QSAR modelling; a natural baseline before graph neural networks.

---

## Affiliation

Developed as part of a computational biology project at the  
**Helmholtz Centre for Infection Research (HZI)**, Braunschweig, Germany.

---

## License

This project is for research and educational purposes. See individual data sources for their respective usage terms.
