# -*- coding: utf-8 -*-
"""
acl_config.py — ACLInsight-Net
Global configuration: installs, imports, seeds, device, hyperparameters, paths.
Run this first (or import it) before any other module.
"""

# ── 1. Install Dependencies ───────────────────────────────────────────────
# Uncomment when running in Colab / Kaggle
# import subprocess
# subprocess.run(["pip", "install", "-q", "timm", "albumentations", "kaggle", "seaborn", "grad-cam", "thop"])

# ── 2. Dataset Download (Colab only) ─────────────────────────────────────
# from google.colab import drive, files
# drive.mount('/content/drive')
# files.upload()   # upload kaggle.json
# import os, subprocess
# os.makedirs(os.path.expanduser("~/.kaggle"), exist_ok=True)
# subprocess.run(["cp", "kaggle.json", os.path.expanduser("~/.kaggle/")])
# subprocess.run(["chmod", "600", os.path.expanduser("~/.kaggle/kaggle.json")])
# subprocess.run(["kaggle", "datasets", "download", "-d", "cjinny/mrnet-v1", "-p", "/content/mrnet", "--unzip"])

# ── 3. Imports ────────────────────────────────────────────────────────────
import os
import random
import time

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import cv2

import timm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, balanced_accuracy_score,
    confusion_matrix, matthews_corrcoef, log_loss, brier_score_loss
)
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold

try:
    from thop import profile as thop_profile
    THOP_AVAILABLE = True
except ImportError:
    THOP_AVAILABLE = False
    print("thop not installed — run: pip install thop")

print("All libraries imported successfully!")

# ── 4. Seed & Device ──────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ── 5. Hyperparameters ────────────────────────────────────────────────────
IMAGE_SIZE  = 224
BATCH_SIZE  = 32        # Safe on T4 with AMP
EPOCHS_S1   = 10        # Stage 1: head-only
EPOCHS_S2   = 30        # Stage 2: backbone fine-tune
LR_S1       = 3e-4
LR_S2       = 5e-5
PATIENCE    = 8
NUM_WORKERS = 4
POS_WEIGHT  = 2.5

PLANES = ['axial', 'coronal', 'sagittal']

# ── 6. Paths ──────────────────────────────────────────────────────────────
DRIVE_DIR       = '/content/drive/MyDrive/ACLInsight-Net'
CKPT_DIR        = f'{DRIVE_DIR}/checkpoints'
FIG_DIR         = f'{DRIVE_DIR}/figures'
RESULTS_DIR     = f'{DRIVE_DIR}/results'
BEST_MODEL_PATH = f'{CKPT_DIR}/best_acl_model.pth'

DATA_ROOT     = '/content/mrnet/MRNet-v1.0'
TRAIN_DIR     = f'{DATA_ROOT}/train'
VALID_DIR     = f'{DATA_ROOT}/valid'
TRAIN_ACL_CSV = f'{DATA_ROOT}/train-acl.csv'
VALID_ACL_CSV = f'{DATA_ROOT}/valid-acl.csv'

for d in [CKPT_DIR, FIG_DIR, RESULTS_DIR]:
    os.makedirs(d, exist_ok=True)

print("Hyperparameters & paths set.")
print(f"Drive output dir : {DRIVE_DIR}")
print(f"Dataset root     : {DATA_ROOT}")
