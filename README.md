# A Systematic Validation & Enhancement of ML-iAMR with Interpretability Analysis

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

> **Reproducibility and Enhancement of Machine Learning-Based Antimicrobial Resistance Prediction: A Systematic Validation of ML-iAMR with Interpretability Analysis**

This repository contains the complete code, data, and results for our independent reproduction and enhancement of the ML-iAMR framework ([Ren et al., 2022](https://academic.oup.com/bioinformatics/article/38/2/325/6375039)).

**The Things I tried:**
- First independent reproduction of ML-iAMR baseline models
- Application of GradCAM interpretability to genomic AMR prediction
- Statistical validation with DeLong tests and bootstrap CIs
- SHAP feature importance analysis identifying resistance-associated SNPs
- CNN models achieving AUROC=0.953 for ciprofloxacin (exceeding original paper)

---

## Honest Limitations Section

"We acknowledge several constraints: 
(1) Our dataset comprised 809 samples versus the paper's 987 (82% coverage), 
(2) We utilized label encoding exclusively due to unavailability of positional SNP metadata required for one-hot encoding, 
(3) Public dataset validation was infeasible due to format incompatibility (gene presence/absence vs. SNP matrix), 
(4) CNN comparison was limited as FCGR encoding requires raw sequence data.

---

## Overview

Antimicrobial resistance (AMR) is a global health crisis requiring rapid, accurate diagnostic methods. This repo:

1. **Reproduces** the ML-iAMR baseline using Random Forest, Logistic Regression, SVM, and CNN models
2. **Tries to Enhance** performance through hyperparameter optimization and ensemble learning
3. **Interprets** predictions using GradCAM and SHAP to identify resistance-associated SNPs
4. **Validates** results with rigorous statistical testing

### Original Paper
Yunxiao Ren, Trinad Chakraborty, Swapnil Doijad, Linda Falgenhauer, Jane Falgenhauer, Alexander Goesmann, Anne-Christin Hauschild, Oliver Schwengers, Dominik Heider, Prediction of antimicrobial resistance based on whole-genome sequencing and machine learning, Bioinformatics, Volume 38, Issue 2, January 2022, Pages 325–334, https://doi.org/10.1093/bioinformatics/btab681

---

## Results Summary

### Baseline Reproduction

| Antibiotic | Our RF | Paper RF | Status | CNN (Ours) | CNN (Paper) |
|------------|--------|----------|--------|------------|-------------|
| **Ciprofloxacin (CIP)** | 0.951 | 0.96 | Reproduced | **0.953** | 0.90 |
| **Cefotaxime (CTX)** | 0.840 | 0.81 | Reproduced | **0.862** | 0.77 |
| **Ceftazidime (CTZ)** | 0.845 | 0.93 | Gap | 0.798 | 0.88 |
| **Gentamicin (GEN)** | 0.766 | 0.95 | Gap | 0.648 | 0.85 |

**Key Findings:**
- **Successfully reproduced** CIP and CTX within 5% margin (4/12 experiments)
- **CNN outperformed** original paper for balanced classes (CIP, CTX)
- **Severe class imbalance** (GEN: 23% resistant) poses unresolved challenges
- **Sample size reduction** (809 vs 987, 82% coverage) explains performance gaps

### Enhancement Results

| Experiment | Antibiotic | Baseline | Enhanced | Improvement | Significance |
|------------|------------|----------|----------|-------------|--------------|
| Hyperparameter Tuning | GEN | 0.766 | 0.784 | +1.8% | p=0.028* |
| Hyperparameter Tuning | CTZ | 0.845 | 0.843 | -0.2% | n.s. |
| Ensemble Learning | CIP | 0.951 | 0.945 | -0.6% | n.s. |

*Statistically significant (permutation test)

### Interpretability Insights

**SHAP Top 3 SNPs per Antibiotic:**

| Antibiotic | Position | SHAP | Gene | Biological Relevance |
|------------|----------|------|------|----------------------|
| **CIP** | 4441487 | 0.0042 | *ytfL* upstream | DNA topology regulation |
| **CIP** | 4428463 | 0.0037 | intergenic | Novel locus |
| **CTX** | 4466572 | 0.0018 | *treR* | Metabolic regulator |
| **CTZ** | 4466572 | 0.0022 | *treR* | β-lactamase expression |
| **GEN** | 4114164 | 0.0020 | *yiiS* | Unknown function |

**GradCAM Activation:** Localized hotspots at genomic positions ~200K, ~323K, ~448K for CIP resistant samples.

---

## Installation

### Prerequisites
- Python 3.10 or higher
- 16GB RAM recommended
- GPU optional (CNN training: 4GB VRAM)
- Google Colab compatible

### Option 1: Conda Environment (Recommended)

```bash
# Clone repository
git clone https://github.com/Muhammad-Lukman/Validation_of_ML-iAMR.git
cd Validation_of_ML-iAMR

```

### Option 2: Google Colab

```python
# Run this in a Colab notebook cell
!git clone https://github.com/Muhammad-Lukman/Validation_of_ML-iAMR.git
%cd Validation_of_ML-iAMR
!pip install -r requirements.txt
```

### Dependencies

**Core ML/DL:**
- scikit-learn==1.3.0
- tensorflow==2.15.0
- keras==3.0.0
- xgboost==2.0.0

**Optimization & Interpretability:**
- optuna==3.4.0
- shap==0.43.0
- imbalanced-learn==0.11.0

**Data & Visualization:**
- pandas==2.1.0
- numpy==1.24.0
- matplotlib==3.8.0
- seaborn==0.13.0

**Full list:** See `requirements.txt`

---

## Data

### Giessen Dataset

**Source:** [ML-iAMR GitHub Repository](https://github.com/YunxiaoRen/ML-iAMR)

**Download Instructions:**

```bash
# Download from original repository
wget https://github.com/YunxiaoRen/ML-iAMR/raw/main/Giessen_dataset.zip

# Or use our preprocessed version
cd data/raw/
# - cip_ctx_ctz_gen_multi_data.csv
# - cip_ctx_ctz_gen_pheno.csv
```

**Dataset Details:**
- **Samples:** 809 *E. coli* isolates (note: paper used 987, public data has 809)
- **Features:** 60,936 SNP positions (label encoded: A=1, T=2, G=3, C=4, N=0)
- **Labels:** Binary resistance (0=Susceptible, 1=Resistant) for 4 antibiotics
- **Class Distribution:**
  - CIP: 366R / 443S (45.2%)
  - CTX: 358R / 451S (44.3%)
  - CTZ: 276R / 533S (34.1%)
  - GEN: 188R / 621S (23.2%) ← Severe imbalance

**File Structure:**
```
data/
├── raw/giessen/
│   ├── cip_ctx_ctz_gen_multi_data.csv    # SNP matrix (809 x 60937)
│   └── cip_ctx_ctz_gen_pheno.csv         # Phenotypes (809 x 5)
├── processed/
│   └── normalized/                        # CNN-ready normalized data
└── metadata/
    └── class_distributions.csv                   # Class distributions
```

### Public Dataset (Optional)

The original paper also used a public validation dataset from Moradigaravand et al. (2018). Due to format incompatibility (gene presence/absence vs. SNP matrix), we focused on Giessen data only.

---

## Usage

### Quick Start: Reproduce Baseline Results

```bash
# Run complete baseline reproduction (all models, all antibiotics)

# Expected output:
# CIP RF: AUROC = 0.951 (Paper: 0.96)
# CTX RF: AUROC = 0.840 (Paper: 0.81)
# CTZ RF: AUROC = 0.845 (Paper: 0.93)
# GEN RF: AUROC = 0.766 (Paper: 0.95)

# Results saved to: .../ML-iAMR_Recreation_archive/05_evaluation/results/baseline_results.csv
```

### Step-by-Step Workflow

#### 1. Baseline Models
```bash
# Train all baseline models (RF, LR, SVM)
notebooks/03_train_RF_baseline_models_all_4_antibiotics.ipynb --antibiotics CIP CTX CTZ GEN
notebooks/04_train_All_baseline_models.ipynb --antibiotics CIP CTX CTZ GEN (All Models)

# Train single antibiotic
notebooks/02_train_RF_baseline_model_CIP.ipynb --antibiotics CIP --models RF
```

#### 2. Class Imbalance Testing
```bash
notebooks/05_class_imbalance_handling_for_CTZ&GEN.ipynb --antibiotics CTZ GEN

# Tests: baseline, class_weight, SMOTE, SMOTE+balanced
```

#### 4. Hyperparameter Tuning
```bash
# Bayesian optimization (50 trials, ~3 hours)
notebooks/06_Hyperparameter_tuning_for_CTZ&GEN(Optuna).ipynb --antibiotics CTZ GEN --trials 50

# Quick test (10 trials, ~30 min)
notebooks/06_Hyperparameter_tuning_for_CTZ&GEN(Optuna).ipynb --trials 10 --quick
```

#### 5. CNN Training & GradCAM
```bash
# Train CNN with GradCAM analysis
notebook/10_CNN_GradCAM.ipynb --antibiotics CIP CTX CTZ GEN --epochs 20

# Outputs:
# - models/CNN_CIP.keras
# - results/gradcam_visualizations.png
# - results/cnn_top_snps.csv
```

#### 6. SHAP Analysis
```bash
# Generate SHAP feature importance
notebook/07_SHAP_Interpretability.ipynb --antibiotics CIP CTX CTZ GEN --samples 100

# Outputs:
# - results/shap_top20_snps.csv
# - results/shap_plots/
```

#### 7. Statistical Validation
```bash
# Run all statistical tests
notebook/09_Statistical_Validation.ipynb

# Tests: DeLong, Bootstrap CI, McNemar, Permutation, Cohen's d
```

---

## Project Structure

```
Validation_of_ML-iAMR/
│
├── README.md                          # This file
├── LICENSE                            # MIT License
├── environment.yml                    # Conda environment
├── requirements.txt                   # pip dependencies
├── .gitignore
│
├── 01_data/
│   ├── raw/                           # Original Giessen data
|   |    ├──giessen/
|   |    ├──public/
│   ├── processed/                     # Processed/normalized data
│   └── metadata/                      # Data summaries
│
├── notebooks/                         # Jupyter notebooks
│   ├── 01_data_preparation.ipynb
│   ├── 02_train_RF_baseline_model_CIP.ipynb
│   ├── 03_train_RF_baseline_models_all_4_antibiotics.ipynb
│   ├── 04_train_All_baseline_models.ipynb
│   ├── 05_class_imbalance_handling_for_CTZ&GEN.ipynb
│   ├── 06_Hyperparameter_tuning_for_CTZ&GEN(Optuna).ipynb
│   ├── 07_SHAP_Interpretability.ipynb
│   └── 08_Ensemble_Learning.ipynb
│   ├── 09_Statistical_Validation.ipynb
│   ├── 10_CNN_GradCAM.ipynb
│
├── 05_evaluation/results/                           # All experiment outputs
│   ├── baseline_results.csv
│   ├── cnn_results.csv
│   ├── statistical_tests.csv
│   ├── figures/                       # figures
│
├── models/                            # Trained models
│   ├── CNN_CIP.keras
│   ├── CNN_CTX.keras
│   ├── CNN_CTZ.keras
│   ├── CNN_GEN.keras
│   └── optimized_rf_params.json

```

---

## Reproducibility

### Random Seeds

All experiments use **fixed random seeds** for reproducibility:

```python
RANDOM_SEED = 42

# Python
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# TensorFlow
tf.random.set_seed(RANDOM_SEED)

# Scikit-learn
random_state=RANDOM_SEED
```

### Expected Variations

Due to non-deterministic GPU operations, expect minor variations (±0.01 AUROC) in CNN results. All other results should be exactly reproducible.

### Computing Environment

**Tested on:**
- Google Colab (Tesla T4 GPU, 12GB RAM)
- Local machine (Ubuntu 22.04, 16GB RAM)

**Approximate Runtimes:**

| Task | Google Colab (GPU) | Local (GPU) | Local (CPU) |
|------|-------------------|-------------|-------------|
| Baseline RF/LR/SVM | 45 min | 30 min | 60 min |
| CNN Training (all) | 60 min | 40 min | 4 hours |
| Hyperparameter Tuning | 180 min | 120 min | 8 hours |
| SHAP Analysis | 80 min | 60 min | 180 min |
| **Total Pipeline** | ~6 hours | ~4 hours | ~14 hours |

---

## Results & Figures

### Key Outputs

All results are saved in `05_evaluation/results/` directory:

**CSV Files:**
- `baseline_results.csv` - All baseline model performance
- `cnn_results.csv` - CNN training metrics
- `hyperparameter_tuning.csv` - Optimization results
- `statistical_tests.csv` - All p-values and effect sizes
- `shap_top20_snps.csv` - Feature importance rankings

**Figures:**
- `roc_curves_all.png` - ROC curves for all models
- `gradcam_heatmaps.png` - GradCAM visualizations
- `shap_importance.png` - SHAP feature importance
- `performance_comparison.png` - Baseline vs enhanced

**Models:**
- `models/*.keras` - Trained CNN models
- `models/optimized_rf_params.json` - Best hyperparameters

---

## Citation

If you use this code or findings in your research, please cite:

```bibtex
@github{muhammadlukman2025mliamr,
  title={Reproducibility and Enhancement of Machine Learning-Based Antimicrobial Resistance Prediction: A Systematic Validation of ML-iAMR with Interpretability Analysis},
  author={Muhammad Lukman},
  year={2025},
  note={GitHub repository},
  url={https://github.com/Muhammad-Lukman/Validation_of_ML-iAMR.git}
}
```

**Also cite the original ML-iAMR paper:**

```bibtex
Yunxiao Ren, Trinad Chakraborty, Swapnil Doijad, Linda Falgenhauer, Jane Falgenhauer, Alexander Goesmann, Anne-Christin Hauschild, Oliver Schwengers, Dominik Heider, Prediction of antimicrobial resistance based on whole-genome sequencing and machine learning, Bioinformatics, Volume 38, Issue 2, January 2022, Pages 325–334, https://doi.org/10.1093/bioinformatics/btab681
```

---

## License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

**Summary:**
- Commercial use allowed
- Modification allowed
- Distribution allowed
- Private use allowed
- Liability and warranty disclaimed

---

## Acknowledgments

- **Ren et al. (2022)** for developing ML-iAMR and making the dataset publicly available
- **Google Colab** for providing free GPU resources
- **Giessen University Hospital** for collecting and sharing the bacterial isolates
- **scikit-learn, TensorFlow, SHAP, and Optuna** development teams

### Related Projects

- [Original ML-iAMR Repository](https://github.com/YunxiaoRen/ML-iAMR)
- [CARD Database](https://card.mcmaster.ca/)
- [ResFinder](https://cge.food.dtu.dk/services/ResFinder/)
- [WHO GLASS](https://www.who.int/initiatives/glass)

---

## Contact

**Maintainer:** Muhammad Lukman  
**Email:** dr.mlukmanuaf@gmail.com  

---

**Last Updated:** January 2026  
**Version:** 1.0.0  
**Status:** Closed

---


**Built with ❤️ for reproducible science in antimicrobial resistance research**
