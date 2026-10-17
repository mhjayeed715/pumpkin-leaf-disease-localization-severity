# Dataset Documentation & Access Guide

## Primary Field Dataset
* **Name:** Pumpkin Leaf Diseases Dataset (from Bangladesh)
* **Publisher / Author:** Tahmid Mir (2023)
* **Platform:** Kaggle
* **URL:** [https://www.kaggle.com/datasets/tahmidmir/pumpkin-leaf-diseases-dataset-from-bangladesh](https://www.kaggle.com/datasets/tahmidmir/pumpkin-leaf-diseases-dataset-from-bangladesh)
* **Format:** 2,000 in-field RGB images ($800 \times 600$ pixels) across 5 balanced classes (400 images per class):
  1. `Bacterial Leaf Spot` (BLS) — *Xanthomonas cucurbitae*
  2. `Downy Mildew` (DM) — *Pseudoperonospora cubensis*
  3. `Healthy Leaf` (HL) — Control
  4. `Mosaic Disease` (MD) — Viral complex (PRSV / WMV / ZYMV)
  5. `Powdery Mildew` (PM) — *Podosphaera xanthii*

### Download Instructions via Kaggle API
```bash
# Ensure kaggle.json API key is in ~/.kaggle/
kaggle datasets download -d tahmidmir/pumpkin-leaf-diseases-dataset-from-bangladesh
unzip -q pumpkin-leaf-diseases-dataset-from-bangladesh.zip -d data/raw/
```

---

## Benchmark Ground Truth & Subsets

### 1. 80-Image 4-Class Pilot Segmentation Benchmark
* **Composition:** 80 field images (20 Downy Mildew, 20 Healthy Leaf, 20 Mosaic Disease, 20 Powdery Mildew).
* **Ground Truth Annotations:**
  - `M_leaf`: Binary polygon mask isolating the primary focal leaf lamina.
  - `M_lesion`: Pathological lesion mask within the focal leaf.
* **Bacterial Leaf Spot (BLS) Status:** Withheld from the pilot benchmark pending expert plant pathology review due to miniature punctate morphology (1–3 mm).
* **Severity Calculation:**
  $$\text{Severity (\%)} = \frac{|M_{\text{lesion}} \cap M_{\text{leaf}}|}{|M_{\text{leaf}}|} \times 100\%$$

### 2. Standardized 200-Image Field Clutter Audit Protocol
* Located in `data/dataset_audit_sheet_200.xlsx`.
* Pre-declared protocol scoring 10 structured criteria across 40 stratified images per class (seed 42).
* Evaluates multi-leaf presence, background weed clutter, lighting glare, and lesion boundary sharpness with a pre-declared 25% decision rule for upstream detection necessity.
