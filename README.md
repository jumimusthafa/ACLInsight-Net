# ACLInsight-Net 🦴

> **ACL Tear Detection from Knee MRI using EfficientNet-B0 + Grad-CAM Explainability**  
> Targeting IEEE publication | Dataset: MRNet-v1 (Stanford)

---

## Overview

ACLInsight-Net is a deep learning framework for **automated ACL (Anterior Cruciate Ligament) tear detection** from knee MRI scans. It combines a lightweight EfficientNet-B0 backbone with exam-level aggregation, weighted ensemble TTA, and Grad-CAM explainability — designed to be both clinically interpretable and computationally efficient.

---

## Results

### Classification Metrics

| Metric | Score |
|---|---|
| **AUC-ROC** | **0.9650** |
| Accuracy | 0.9083 |
| F1-Score | 0.9009 |
| Precision | 0.8772 |
| Recall / Sensitivity | 0.9259 |
| Specificity | 0.8939 |
| Balanced Accuracy | 0.9099 |
| MCC | 0.8176 |
| Log Loss | 0.2207 |
| Brier Score | 0.0684 |
| FPR | 0.1061 |
| FNR | 0.0741 |
| NPV | 0.9365 |

### Computational Performance

| Metric | Value |
|---|---|
| Total Parameters | 4,008,829 (~4M) |
| Avg Inference Time | 1.22 ms/slice |
| Throughput | 822 slices/sec |

### Ablation Study

| Variant | Accuracy | AUC-ROC | F1-Score | Recall | Specificity |
|---|---|---|---|---|---|
| Stage-1 Only (Frozen backbone) | 0.650 | 0.642 | 0.585 | 0.889 | 0.515 |
| Stage-2 Fine-Tune (Last 2 blocks) | 0.483 | 0.601 | 0.563 | 0.963 | 0.197 |
| **Full ACLInsight-Net (Ensemble + TTA)** | **0.908** | **0.965** | **0.901** | **0.926** | **0.894** |

> The full model with weighted ensemble TTA is necessary for achieving optimal performance. Stage-2 fine-tuning alone improves recall but severely hurts specificity — the complete pipeline balances both.

---

## Key Features

- **EfficientNet-B0** backbone with ImageNet pretraining, adapted for MRI domain
- **Two-stage fine-tuning** — frozen backbone (Stage 1) → last 2 blocks unfrozen (Stage 2)
- **Exam-level aggregation** — slice predictions pooled via mean to case-level diagnosis
- **Weighted Ensemble + TTA** — final prediction via test-time augmentation ensemble
- **Grad-CAM XAI** — spatial heatmaps showing which knee regions drive predictions
- **Comprehensive evaluation** — 13 metrics covering all IEEE/SCI medical imaging requirements
- **5-fold cross-validation** with mean ± std reporting
- All outputs auto-saved to Google Drive

---

## Metrics Covered

| Category | Metrics |
|---|---|
| Classification | Accuracy, Precision, Recall, Specificity, F1-Score |
| ROC-Based | AUC-ROC, ROC Curve |
| Probabilistic | Log Loss, Brier Score |
| Class Imbalance | Balanced Accuracy, MCC |
| Clinical | FPR, FNR, NPV |
| XAI | Deletion Score, Insertion Score, CAM Localization |
| Robustness | 5-Fold CV mean ± std |
| Computational | Inference Time, Parameters, FLOPs |
| Ablation | Stage-1 vs Stage-2 vs Full Model |

---

## Model Architecture

```
Input MRI Slice (224×224×3)
        ↓
EfficientNet-B0 Backbone (pretrained ImageNet)
  Stage 1: Backbone frozen
  Stage 2: Last 2 MBConv blocks unfrozen
        ↓
Global Average Pooling → 1280-d embedding
        ↓
Dropout(0.4) → Linear(1280, 1)
        ↓
BCEWithLogitsLoss (pos_weight=1.8)
        ↓
Sigmoid → Exam-level mean aggregation
        ↓
Weighted Ensemble + TTA
        ↓
Binary Prediction (ACL Intact / ACL Torn)
```

---

## Setup & Usage

### 1. Open in Google Colab
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)

Upload `acl_detection.ipynb` to Colab. Use a **T4 or A100 GPU** runtime.

### 2. Install Dependencies
```bash
pip install timm albumentations kaggle seaborn grad-cam thop
```

### 3. Dataset

The notebook uses **MRNet-v1** from Stanford ML Group, hosted on Kaggle.

1. Get your `kaggle.json` API key from [kaggle.com/settings](https://www.kaggle.com/settings)
2. Upload it when prompted
3. The notebook auto-downloads and extracts the dataset

```
MRNet-v1.0/
├── train/
│   ├── axial/        # .npy volumes (n_slices, 256, 256) uint8
│   ├── coronal/
│   └── sagittal/
├── valid/
│   ├── axial/
│   ├── coronal/
│   └── sagittal/
├── train-acl.csv
└── valid-acl.csv
```

📎 [Kaggle: cjinny/mrnet-v1](https://www.kaggle.com/datasets/cjinny/mrnet-v1)

### 4. Run All Cells

Run top to bottom. Expected runtime on T4: ~25–35 min for full training.

---

## Hyperparameters

| Parameter | Value |
|---|---|
| Image Size | 224×224 |
| Batch Size | 32 |
| Stage 1 Epochs | 5 |
| Stage 2 Epochs | 10 |
| Stage 1 LR | 3e-4 |
| Stage 2 LR | 5e-5 |
| Scheduler | CosineAnnealingLR |
| Dropout | 0.4 |
| Pos Weight | 1.8 |
| Patience | 4 |

---

## Requirements

```
Python          >= 3.9
PyTorch         >= 2.0
timm            >= 0.9
albumentations
grad-cam
scikit-learn
thop
opencv-python
scipy
seaborn
pandas
numpy
tqdm
```

---

## Dataset

**MRNet-v1** — Stanford ML Group  
~1,370 knee MRI exams | 3 planes: axial, coronal, sagittal | Binary ACL labels

📄 [Original MRNet Paper — Bien et al., 2018](https://journals.plos.org/plosmedicine/article?id=10.1371/journal.pmed.1002686)

---

## Citation

```bibtex
@article{aclinsightnet2026,
  title   = {ACLInsight-Net: Explainable Deep Learning for ACL Tear Detection from Knee MRI},
  journal = {IEEE},
  year    = {2026}
}
```

---

## Acknowledgements

- [MRNet Dataset](https://stanfordmlgroup.github.io/competitions/mrnet/) — Stanford ML Group
- [timm](https://github.com/huggingface/pytorch-image-models) — EfficientNet-B0 backbone
- [pytorch-grad-cam](https://github.com/jacobgil/pytorch-grad-cam) — Grad-CAM implementation

---

## License

MIT License — see [LICENSE](LICENSE) for details.
