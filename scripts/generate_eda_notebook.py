"""Generate reproducible EDA notebook for UNSW-NB15."""

import json
from pathlib import Path

def create_notebook():
    nb = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# UNSW-NB15: Exploratory Data Analysis & Leakage Investigation\n",
                    "**Adaptive Threat Intelligence** — DataNova Global Data-thon 2026\n",
                    "\n",
                    "This notebook provides an empirical, reproducible analysis of the canonical **UNSW-NB15** network intrusion detection benchmark dataset. It evaluates class imbalance, distribution shifts, rare attack categories, duplicate contamination, and potential target leakage risks."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import sys\n",
                    "from pathlib import Path\n",
                    "import numpy as np\n",
                    "import pandas as pd\n",
                    "import matplotlib.pyplot as plt\n",
                    "import seaborn as sns\n",
                    "\n",
                    "# Set plotting defaults\n",
                    "plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')\n",
                    "plt.rcParams['figure.dpi'] = 150\n",
                    "\n",
                    "data_dir = Path('../../data/raw/unsw_nb15')\n",
                    "train_path = data_dir / 'UNSW_NB15_training-set.csv'\n",
                    "test_path = data_dir / 'UNSW_NB15_testing-set.csv'\n",
                    "\n",
                    "df_train = pd.read_csv(train_path)\n",
                    "df_test = pd.read_csv(test_path)\n",
                    "print(f'Loaded Train: {len(df_train):,} rows x {len(df_train.columns)} cols')\n",
                    "print(f'Loaded Test:  {len(df_test):,} rows x {len(df_test.columns)} cols')"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 1. Dataset Dimensions & Schema Validation"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "assert list(df_train.columns) == list(df_test.columns), 'Column mismatch between splits!'\n",
                    "print('Schema alignment confirmed across all 45 columns.')\n",
                    "df_train.head(3)"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 2. Missing Values, Duplicates & Contamination Analysis\n",
                    "Investigating whether exact duplicate records span across the training and testing partitions."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "print(f'Missing cells in Train: {df_train.isnull().sum().sum()}')\n",
                    "print(f'Missing cells in Test:  {df_test.isnull().sum().sum()}')\n",
                    "\n",
                    "# Duplicates without ID column\n",
                    "tr_dupes = df_train.drop(columns=['id']).duplicated().sum()\n",
                    "te_dupes = df_test.drop(columns=['id']).duplicated().sum()\n",
                    "print(f'Train internal duplicates (no id): {tr_dupes:,} ({tr_dupes/len(df_train)*100:.2f}%)')\n",
                    "print(f'Test internal duplicates (no id):  {te_dupes:,} ({te_dupes/len(df_test)*100:.2f}%)')\n",
                    "\n",
                    "# Exact overlap between Train and Test\n",
                    "train_no_id = df_train.drop(columns=['id'])\n",
                    "test_no_id = df_test.drop(columns=['id'])\n",
                    "overlap = pd.merge(train_no_id, test_no_id, how='inner')\n",
                    "print(f'Exact cross-partition contamination: {len(overlap):,} rows ({len(overlap)/len(df_train)*100:.2f}% of train)')"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 3. Target Distribution & Imbalance Analysis\n",
                    "Investigating binary `label` and multiclass `attack_cat` across both partitions."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))\n",
                    "\n",
                    "# Binary distribution\n",
                    "bin_df = pd.DataFrame({\n",
                    "    'Train': df_train['label'].value_counts(normalize=True) * 100,\n",
                    "    'Test': df_test['label'].value_counts(normalize=True) * 100\n",
                    "}, index=[0, 1])\n",
                    "bin_df.index = ['Normal (0)', 'Attack (1)']\n",
                    "bin_df.plot(kind='bar', ax=ax1, color=['#2b5c8f', '#d9534f'], rot=0)\n",
                    "ax1.set_ylabel('Percentage (%)')\n",
                    "ax1.set_title('Binary Label Distribution (Train vs. Test)')\n",
                    "\n",
                    "# Multiclass distribution\n",
                    "cat_df = pd.DataFrame({\n",
                    "    'Train': df_train['attack_cat'].value_counts(normalize=True) * 100,\n",
                    "    'Test': df_test['attack_cat'].value_counts(normalize=True) * 100\n",
                    "}).sort_values(by='Train', ascending=True)\n",
                    "cat_df.plot(kind='barh', ax=ax2, color=['#337ab7', '#e67e22'])\n",
                    "ax2.set_xlabel('Percentage (%)')\n",
                    "ax2.set_title('Attack Category Distribution (Train vs. Test)')\n",
                    "plt.tight_layout()\n",
                    "plt.show()"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 4. Extreme Minority Classes"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "minority = ['Analysis', 'Backdoor', 'Shellcode', 'Worms']\n",
                    "min_table = pd.DataFrame({\n",
                    "    'Train Count': df_train['attack_cat'].value_counts().reindex(minority),\n",
                    "    'Train Pct (%)': (df_train['attack_cat'].value_counts(normalize=True).reindex(minority) * 100).round(3),\n",
                    "    'Test Count': df_test['attack_cat'].value_counts().reindex(minority),\n",
                    "    'Test Pct (%)': (df_test['attack_cat'].value_counts(normalize=True).reindex(minority) * 100).round(3),\n",
                    "})\n",
                    "min_table"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 5. Potential Target Leakage & Synthetic Artifacts\n",
                    "1. **Sequential ID Leakage**: The index `id` has high correlation with `label`.\n",
                    "2. **Source TTL (`sttl`)**: Distinct synthetic OS generation default."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "print(f\"Correlation between id and label (Train): {df_train['id'].corr(df_train['label']):.4f}\")\n",
                    "print(f\"Correlation between id and label (Test):  {df_test['id'].corr(df_test['label']):.4f}\")\n",
                    "\n",
                    "fig, ax = plt.subplots(figsize=(10, 4))\n",
                    "df_test['label'].rolling(1000).mean().plot(ax=ax, color='#d9534f', label='Test (rolling mean 1000)')\n",
                    "df_train['label'].rolling(1000).mean().plot(ax=ax, color='#2b5c8f', label='Train (rolling mean 1000)')\n",
                    "ax.set_title(\"Demonstrating Sequential Grouping in Dataset: Rolling Attack Rate\")\n",
                    "ax.set_ylabel(\"Attack Proportion\")\n",
                    "ax.legend()\n",
                    "plt.show()"
                ]
            }
        ],
        "metadata": {
            "language_info": {
                "name": "python",
                "version": "3.13.14"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }

    out_path = Path("research/notebooks/01_unsw_nb15_eda.ipynb")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)
    print("Saved notebook to:", out_path)

if __name__ == "__main__":
    create_notebook()
