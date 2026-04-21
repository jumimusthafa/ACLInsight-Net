# -*- coding: utf-8 -*-
"""
acl_train.py — ACLInsight-Net
Loss, optimizer, training / validation functions, and two-stage training loop.
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from tqdm import tqdm

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, balanced_accuracy_score,
    confusion_matrix, matthews_corrcoef, log_loss, brier_score_loss
)

from acl_config import (
    device, PLANES, CKPT_DIR,
    EPOCHS_S1, EPOCHS_S2, LR_S1, LR_S2, PATIENCE,
    POS_WEIGHT
)
from acl_model import ACLModel, freeze_backbone, unfreeze_last_n_blocks


# ── Loss & Scaler ─────────────────────────────────────────────────────────
criterion = torch.nn.BCEWithLogitsLoss(
    pos_weight=torch.tensor([POS_WEIGHT]).to(device)
)
# torch.amp.GradScaler replaces deprecated torch.cuda.amp.GradScaler
scaler = torch.amp.GradScaler('cuda')

print("Loss and scaler initialized.")


# ── Training Step ─────────────────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer):
    """One full pass over the training DataLoader."""
    model.train()
    running_loss = 0.0

    for images, labels, _ in tqdm(loader, desc="Train", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.unsqueeze(1).to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast('cuda'):
            loss = criterion(model(images), labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running_loss += loss.item()

    return running_loss / len(loader)


# ── Validation (standard) ─────────────────────────────────────────────────
@torch.no_grad()
def validate(model, loader):
    """Returns (y_true, y_prob, case_ids) numpy arrays — slice level."""
    model.eval()
    y_true, y_prob, case_ids = [], [], []

    for images, labels, cases in tqdm(loader, desc="Val  ", leave=False):
        images = images.to(device, non_blocking=True)
        probs  = torch.sigmoid(model(images)).cpu().numpy().flatten()
        y_prob.extend(probs)
        y_true.extend(labels.numpy())
        case_ids.extend(cases.numpy())

    return np.array(y_true), np.array(y_prob), np.array(case_ids)


# ── Validation with TTA ───────────────────────────────────────────────────
@torch.no_grad()
def validate_tta(model, loader):
    """Validate with Test-Time Augmentation (original + horizontal flip)."""
    model.eval()
    y_true, y_prob, case_ids = [], [], []

    for images, labels, cases in tqdm(loader, desc="Val TTA", leave=False):
        images = images.to(device, non_blocking=True)
        p1 = torch.sigmoid(model(images)).cpu().numpy().flatten()
        p2 = torch.sigmoid(model(torch.flip(images, [-1]))).cpu().numpy().flatten()
        avg = (p1 + p2) / 2.0
        y_prob.extend(avg)
        y_true.extend(labels.numpy())
        case_ids.extend(cases.numpy())

    return np.array(y_true), np.array(y_prob), np.array(case_ids)


# ── Exam-Level Metrics (max pooling) ─────────────────────────────────────
def exam_level_metrics(y_true, y_prob, case_ids, threshold=0.5):
    """
    Aggregate slice-level probabilities to exam level via max pooling,
    then compute a full set of classification and probabilistic metrics.
    Returns (metrics_dict, aggregated_dataframe).
    """
    df = pd.DataFrame({'case': case_ids, 'prob': y_prob, 'true': y_true})
    # max pooling: any torn slice → torn exam
    df = df.groupby('case').agg({'prob': 'max', 'true': 'first'}).reset_index()

    y_true_exam = df['true']
    y_prob_exam = df['prob']
    y_pred_exam = (y_prob_exam > threshold).astype(int)

    acc  = accuracy_score(y_true_exam, y_pred_exam)
    prec = precision_score(y_true_exam, y_pred_exam, zero_division=0)
    rec  = recall_score(y_true_exam, y_pred_exam, zero_division=0)
    f1   = f1_score(y_true_exam, y_pred_exam, zero_division=0)

    tn, fp, fn, tp = confusion_matrix(y_true_exam, y_pred_exam).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    fpr         = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr         = fn / (fn + tp) if (fn + tp) > 0 else 0
    npv         = tn / (tn + fn) if (tn + fn) > 0 else 0

    auc_roc = roc_auc_score(y_true_exam, y_prob_exam)
    bal_acc = balanced_accuracy_score(y_true_exam, y_pred_exam)
    mcc     = matthews_corrcoef(y_true_exam, y_pred_exam)
    logloss = log_loss(y_true_exam, y_prob_exam)
    brier   = brier_score_loss(y_true_exam, y_prob_exam)

    return {
        'Accuracy': acc, 'Precision': prec, 'Recall': rec,
        'Specificity': specificity, 'F1-Score': f1,
        'AUC-ROC': auc_roc, 'Balanced Accuracy': bal_acc,
        'MCC': mcc, 'Log Loss': logloss, 'Brier Score': brier,
        'FPR': fpr, 'FNR': fnr, 'NPV': npv
    }, df


# ── Single Stage Training ─────────────────────────────────────────────────
def run_training_stage(model, train_loader, val_loader, epochs, lr,
                       stage_name, plane=''):
    """
    Trains for up to `epochs` epochs with early stopping on AUC-ROC.
    Saves best checkpoint to CKPT_DIR/best_acl_{plane}.pth.
    Returns (best_auc, history_dataframe).
    """
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=1e-4
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=lr / 10
    )

    best_auc        = 0.0
    patience_counter = 0
    history         = []
    model_save_path = os.path.join(CKPT_DIR, f'best_acl_{plane}.pth')
    os.makedirs(CKPT_DIR, exist_ok=True)

    print(f"\n{'='*55}")
    print(f" {stage_name}")
    print(f"{'-'*55}")

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer)
        y_true, y_prob, case_ids = validate(model, val_loader)
        metrics, _ = exam_level_metrics(y_true, y_prob, case_ids)
        scheduler.step()

        history.append({'epoch': epoch, 'loss': train_loss, **metrics})
        print(
            f"  Ep {epoch:02d}/{epochs} | Loss {train_loss:.4f} | "
            f"AUC {metrics['AUC-ROC']:.4f} | F1 {metrics['F1-Score']:.4f} | "
            f"Recall {metrics['Recall']:.4f}"
        )

        if metrics['AUC-ROC'] > best_auc:
            best_auc = metrics['AUC-ROC']
            patience_counter = 0
            torch.save({
                'state_dict': model.state_dict(),
                'epoch':      epoch,
                'best_auc':   best_auc,
                'stage':      stage_name
            }, model_save_path)
            print(f"    ✓ Saved (AUC={best_auc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print("  Early stopping triggered.")
                break

    return best_auc, pd.DataFrame(history)


# ── Two-Stage Training Loop ───────────────────────────────────────────────
def train_all_planes(train_loaders, val_loaders, train=True):
    """
    Iterates over all three planes.
    Stage 1 — head-only (backbone frozen).
    Stage 2 — last 2 backbone blocks + head unfrozen, lower LR.
    Returns dict of {plane: (hist_s1, hist_s2)}.
    """
    if not train:
        print("Training skipped. Load existing checkpoints with load_plane_models().")
        return {}

    histories = {}
    for plane in PLANES:
        print(f"\n{'='*55}\n TRAINING PLANE: {plane.upper()}\n{'='*55}")
        model = ACLModel().to(device)

        # ── Stage 1: head only ───────────────────────────────────────
        freeze_backbone(model, freeze=True)
        best_s1, hist_s1 = run_training_stage(
            model, train_loaders[plane], val_loaders[plane],
            epochs=EPOCHS_S1, lr=LR_S1,
            stage_name="Stage 1 – Head Only",
            plane=plane
        )

        # ── Stage 2: last 2 blocks + head ───────────────────────────
        unfreeze_last_n_blocks(model, n=2)
        best_s2, hist_s2 = run_training_stage(
            model, train_loaders[plane], val_loaders[plane],
            epochs=EPOCHS_S2, lr=LR_S2,
            stage_name="Stage 2 – Last 2 Blocks + Head",
            plane=plane
        )

        print(f"\nBest AUC ({plane}) → S1: {best_s1:.4f}  |  S2: {best_s2:.4f}")
        histories[plane] = (hist_s1, hist_s2)

    return histories


# ── Load Saved Models ─────────────────────────────────────────────────────
def load_plane_models():
    """Load best checkpoint for each plane. Returns {plane: model}."""
    models = {}
    for plane in PLANES:
        ckpt_path = f'{CKPT_DIR}/best_acl_{plane}.pth'
        if not os.path.exists(ckpt_path):
            print(f"[WARN] Checkpoint not found: {ckpt_path}")
            continue
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        m = ACLModel().to(device)
        m.load_state_dict(ckpt['state_dict'])
        m.eval()
        models[plane] = m
        print(f"Loaded {plane} model (epoch {ckpt['epoch']}, AUC {ckpt['best_auc']:.4f})")
    return models


if __name__ == '__main__':
    print("Import this module — do not run it standalone.")
