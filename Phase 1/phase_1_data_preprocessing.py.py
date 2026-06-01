# -*- coding: utf-8 -*-
"""
Phase 1 — Data Acquisition & Preprocessing
===========================================

Reads the HZI antibacterial activity dataset (mmc1.xlsx, Sheet S1B),
extracts SMILES strings and binary activity labels (1 = Active, 0 = Inactive),
performs quality checks (missing values, duplicates, class distribution),
and writes the cleaned dataset to ecoli_dataset_cleaned.csv.
"""

from google.colab import files
uploaded = files.upload()

import pandas as pd

df = pd.read_excel("mmc1.xlsx", sheet_name="S1B", skiprows=1)

#binary labelling
df['label'] = (df['Activity'] == 'Active').astype(int)

#keeping only SMILES and label
df = df[['SMILES', 'label']]
df.to_csv("ecoli_dataset_cleaned.csv", index=False)

print(df.head())
print(df.columns)
print(df.shape)

#lass Imablance
print(df['label'].value_counts())
print(df['label'].value_counts(normalize=True)*100)

#Data Quality

#missing values
print(df.isnull().sum())

#duplicate molecules
print(df['SMILES'].duplicated().sum())

#unique compunds
print (df['SMILES'].nunique())