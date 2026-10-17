# Pumpkin Leaf Disease Localization, Segmentation, and Severity Estimation

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A multi-task deep learning framework for focal leaf isolation, pathological lesion segmentation, and continuous foliar disease severity estimation in open pumpkin fields.

---

## Table of Contents
* [Overview & Research Motivation](#overview--research-motivation)
* [Methodological Workflow](#methodological-workflow)
* [System Architecture](#system-architecture)
* [Key Empirical Findings](#key-empirical-findings)
  * [1. In-Field Clutter Audit (N = 200)](#1-in-field-clutter-audit-n--200)
  * [2. Published Classification Re-implementation](#2-published-classification-re-implementation)
  * [3. 5-Fold Cross-Validation Segmentation Benchmark](#3-5-fold-cross-validation-segmentation-benchmark)
* [Repository Structure](#repository-structure)
* [Getting Started](#getting-started)
* [Citation & References](#citation--references)

---

## Overview & Research Motivation

Existing literature on pumpkin foliar disease is dominated by image-level classification (86%–96% accuracy). Under uncontrolled field conditions, this paradigm faces three critical challenges:

1. **Semantic Granularity Gap:** A single categorical label cannot distinguish a 2% trace infection from a 60% destructive canopy epidemic.
2. **Canopy Clutter:** In multi-leaf vine canopies, whole-image models cannot indicate which specific leaf is diseased. Our completed 200-image clutter audit establishes an **84.0% multi-leaf presence rate** (168/200), mandating target-leaf isolation.
3. **Absence of Quantitative Masks:** Pixel-level leaf lamina and lesion annotations have been absent from published pumpkin datasets.

This repository provides the experimental code, published model re-implementations, stratified 5-fold cross-validation segmentation benchmark, and continuous severity estimation pipeline for the B.Sc. thesis proposal.

---

## Methodological Workflow

The research methodology is structured across three systematic stages (Steps A–C), decoupling core methodological design from timeline execution:

![Methodological Workflow Steps A to C](results/figures/methodological_workflow_steps.png)

* **Step A: Ground-Truth Protocol & Empirical Audit**
  * Completed 200-image field clutter audit across all 5 classes (`dataset_audit_sheet_200_completed.xlsx`, Cohen's $\kappa = 0.814$).
  * Stratified 80-image pilot benchmark dataset (4 classes: DM, HL, MD, PM; Bacterial Leaf Spot withheld pending expert review).
  * Semi-automated HSV+Otsu initialization (Dice = 0.542) followed by human expert Labelme polygon refinement.
* **Step B: Multi-Task Architecture & 5-Fold Benchmark**
  * Standardized replication of 6 published classification architectures on Tesla T4 GPUs.
  * 5-fold cross-validation segmentation benchmark comparing DualTaskUNet against Two-Stage U-Net, LinkNet, and DeepLabV3+.
* **Step C: Target-Leaf Isolation & Model Optimization (Phase 2 & 3)**
  * Benchmarking standalone target-leaf isolation (RT-DETR) vs. unified DualTaskUNet to resolve the 84.0% clutter mandate.
  * Calibration of upstream confidence gating ($\tau = 0.85$) across remaining classes to suppress healthy false alarms.
  * Full-dataset scaling (N = 640 images) and edge mobile export.

---

## System Architecture

Our primary architecture, **DualTaskUNet**, utilizes a shared multi-scale feature encoder with dual decoders to simultaneously predict focal leaf boundaries ($M_{\text{leaf}}$) and pathological lesion masks ($M_{\text{lesion}}$) in a single forward pass:

![DualTaskUNet Shared-Encoder Dual-Head Architecture](results/figures/dualtask_unet_architecture.png)

### Joint Multi-Task Loss Formulation
To handle severe class imbalance between small lesion spots and large leaf surfaces, we optimize a weighted composite loss:

$$\mathcal{L}_{\text{total}} = 1.0 \cdot \mathcal{L}_{\text{DiceFocal}}(M_{\text{leaf}}, Y_{\text{leaf}}) + 2.5 \cdot \mathcal{L}_{\text{DiceFocal}}(M_{\text{lesion}}, Y_{\text{lesion}})$$

Focal loss parameters ($\alpha = 0.8$, $\gamma = 2.0$) counter lesion pixel sparsity without eroding lamina boundaries.

### Continuous Severity Engine
Continuous disease severity is computed strictly within the segmented focal leaf boundary:

$$\text{Severity \%} = \frac{|M_{\text{lesion}} \cap M_{\text{leaf}}|}{|M_{\text{leaf}}|} \times 100\%$$

---

## Key Empirical Findings

### 1. In-Field Clutter Audit (N = 200)

A systematic manual audit of 200 randomly sampled field images (40 per class, seed 42) was conducted using our standardized rubric (`data/dataset_audit_sheet_200_completed.xlsx`):

![Field Clutter Audit Distribution](results/figures/figure_e_audit_distribution.png)

* **Multi-Leaf Presence:** **84.0%** (168/200 images contain $>1$ leaf). Because this significantly exceeds our pre-registered 25% threshold, target-leaf isolation is formally mandated.
* **Focal Leaf Identifiability:** 99.0% (198/200) contain an unambiguous focal leaf.
* **Lighting Quality:** 99.5% (199/200) shot under natural daylight conditions.
* **Foliar Defects:** 56.5% diffuse chlorosis, 46.5% lamina necrosis, 27.0% background weeds/tearing.

---

### 2. Published Classification Re-implementation

Published pumpkin disease classifiers were re-implemented under a common standardized protocol on Tesla T4 GPUs:

| Model Architecture | Source Study | Reported Acc (%) | Replicated Acc (%) | Split Protocol | Discrepancy (pp) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **ResNet-50** | Khandaker et al. (2024) [2] | 90.50% | **88.00%** | 80/10/10 | -2.50 |
| **DenseNet-121** | Khaldi & Kalmoun (2024) [3] | 86.00% | **83.50%** | 80/10/10 | -2.50 |
| **DenseNet-121** | Gakhar et al. (2025) [4] | 94.00% | **80.50%** | 70/20/10 | -13.50 |
| **DualFusion-CBAM** | Bhuria et al. (2026) [5] | 96.00% | **85.00%** | 80/10/10 | -11.00 |
| **MobileNetV3-Large** | Brar et al. (2025) [6] | 88.00% | **84.50%** | 80/10/10 | -3.50 |
| **ResNet-50** | Kumar et al. (2026) [7] | 91.00% | **85.50%** | 80/10/10 | -5.50 |

*Discrepancy Analysis:* Binomial sampling error on a 200-image test set ($SE \approx \pm 2.5$ pp) accounts for smaller gaps (ResNet-50 and DenseNet-121 at -2.5 pp fall within 1 SE; MobileNetV3 at -3.5 pp falls within 2 SE). Larger discrepancies (Gakhar -13.5 pp, Bhuria DualFusion -11.0 pp) reflect unstated hyperparameters, optimizer schedules, or domain augmentations in original publications.

---

### 3. 5-Fold Cross-Validation Segmentation Benchmark

Evaluated across 80 pilot field images (20 DM, 20 HL, 20 MD, 20 PM; 16 test images per fold; Bacterial Leaf Spot withheld pending expert review):

| Model Architecture | Params (M) | Leaf Dice (Lamina) | Lesion Dice (Diseased) | Severity MAE (Diseased) | Severity MAE (Overall) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **DualTaskUNet (Ours)** | 31.04M | **0.9584 ± 0.0100** | **0.8456 ± 0.0153** | 4.06% ± 2.64% | 7.19% ± 3.00% |
| **Two-Stage Separate U-Net** | 49.54M | 0.9540 ± 0.0028 | 0.8019 ± 0.0079 | **3.21% ± 0.91%** | **3.72% ± 1.77%** |
| **LinkNet-ResNet18** | 11.54M | 0.9434 ± 0.0121 | 0.7621 ± 0.0216 | 4.17% ± 0.79% | 4.71% ± 1.24% |
| **DeepLabV3+ (MobileNetV2)**| 3.21M | 0.9342 ± 0.0103 | 0.7246 ± 0.0353 | 4.15% ± 1.70% | 4.25% ± 0.86% |
| **Classical Baseline (HSV+Otsu)**| 0.00M | 0.9376 ± 0.0339 | 0.0696 ± 0.0566 | 18.99% ± 14.65% | 22.67% ± 14.60% |

![Benchmark Accuracy and Trade-Off](results/figures/figure_f_benchmark_tradeoff.png)

#### Architectural Insights & Trade-Offs
* **Multi-Task Feature Sharing:** DualTaskUNet achieved higher lesion Dice across all 5 folds (+4.37 pp over Two-Stage; +8.35 pp over LinkNet), supporting the hypothesis that shared encoder representations guide subtle lesion boundary delineation.
* **Healthy False Alarms:** Unconstrained decoders incur false-alarm errors on healthy leaves (fold-level mean range 8.7% to 25.1%; overall healthy MAE 16.59%), elevating overall MAE. Two-Stage U-Net limits healthy error to 5.25% by cropping the leaf first.
* **Mitigation:** Upstream Classification Confidence Gating (initial $\tau = 0.85$, to be calibrated across remaining classes) is introduced to suppress false-positive lesion noise on healthy foliage.

---

## Repository Structure

```
├── README.md                               <- Master project overview & empirical benchmark
├── requirements.txt                        <- Python environment dependencies
├── generate_benchmark_charts.py            <- Matplotlib chart generation script
├── data/
│   ├── README.md                           <- Dataset download and annotation guidelines
│   ├── dataset_audit_sheet_200_completed.xlsx <- Completed 200-image field clutter audit (manual review)
│   ├── dataset_audit_sheet_200.xlsx        <- Pre-audit scoring protocol template
│   └── splits/
│       ├── cv5_segmentation_splits.json    <- Stratified 5-fold CV splits (80 pilot images across 4 classes)
│       ├── pilot_100_manifest.csv          <- 100-image pilot cohort manifest (20 per class, 5 classes)
│       └── pilot_100_severity_annotations.csv <- Verified pilot continuous severities
├── notebooks/
│   ├── 01_classification_resnet50.ipynb
│   ├── 02_classification_densenet121.ipynb
│   ├── 03_classification_densenet121_gakhar.ipynb
│   ├── 04_classification_dualfusion_cbam.ipynb
│   ├── 05_classification_mobilenetv3_large.ipynb
│   ├── 06_classification_resnet50_kumar.ipynb
│   ├── 07_segmentation_dualtask_unet_train.ipynb
│   ├── 08_segmentation_dualtask_unet_cv5.ipynb
│   ├── 09_segmentation_twostage_unet_cv5.ipynb
│   ├── 10_segmentation_5fold_benchmark.ipynb
│   ├── 11_segmentation_linknet_resnet18_cv5.ipynb
│   ├── 12_segmentation_deeplabv3plus_cv5.ipynb
│   └── 13_severity_estimation_pipeline.ipynb
└── results/
    ├── run_log.csv                         <- Full training run records
    ├── segmentation_cv5_benchmark.csv      <- Per-fold segmentation evaluation metrics
    ├── classification_comparison.csv       <- Replication comparison table
    └── figures/
        ├── dualtask_unet_architecture.png  <- Proposed DualTaskUNet architectural schematic
        ├── methodological_workflow_steps.png <- Methodological execution workflow (Steps A-C)
        ├── figure_e_audit_distribution.png <- 200-image canopy clutter audit chart
        ├── figure_f_benchmark_tradeoff.png <- 5-fold benchmark & healthy leaf trade-off
        ├── dataset_samples_grid.png        <- Representative class samples
        └── figure1_field_powdery_mildew.png<- Authentic in-field leaf sample
```

---

## Getting Started

### Installation
```bash
git clone https://github.com/mhjayeed715/pumpkin-leaf-disease-localization-severity.git
cd pumpkin-leaf-disease-localization-severity
pip install -r requirements.txt
```

### Reproducing Benchmark Notebooks
To inspect the 5-fold cross-validation segmentation benchmark:
```bash
jupyter notebook notebooks/10_segmentation_5fold_benchmark.ipynb
```

---

## Citation & References
1. Tahmid Mir. *Pumpkin Leaf Diseases Dataset (from Bangladesh)*. Kaggle, 2023.
2. M. A. A. Khandaker et al. *Explainable AI-enhanced deep learning for pumpkin leaf disease detection*. In Proc. 27th ICCIT, 2024.
3. A. Khaldi and E. M. Kalmoun. *Automated disease diagnosis in pumpkin plants using advanced CNN models*. arXiv:2410.00062, 2024.
4. D. Gakhar et al. *Pumpkin leaf disease detection using DenseNet-121*. In Proc. DASA, 2025.
5. R. Bhuria et al. *AI-powered detection of pumpkin leaf diseases using DualFusion-CBAM-stochastic*. Front. Plant Sci., vol. 16, 2026.
6. R. Brar et al. *Smart detection of pumpkin leaf diseases using MobileNetV3*. In Proc. ICICIS, 2025.
7. D. L. Kumar et al. *ResNet50 pipeline for multi-class pumpkin leaf disease diagnosis in field conditions*. In Proc. MPCON, 2026.
