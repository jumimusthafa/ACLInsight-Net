# -*- coding: utf-8 -*-
"""
acl_evaluate.py — ACLInsight-Net
Final exam-level evaluation: weighted ensemble, optimal threshold, training
curves, ROC curve, confusion matrix, and computational performance metrics.
"""

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch

from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, balanced_accuracy_score,
    matthews_corrcoef, confusion_matrix, roc_curve, auc,
    log_loss, brier_score_loss
)

from acl_config import (
    device, PLANES, CKPT_DIR, FIG_DIR, RESULTS_DIR,
    THOP_AVAILABLE, IMAGE_SIZE
)
from acl_train import validate_tta, exam_level_metrics, load_plane_models

try:
    from thop import profile as thop_profile
except ImportError:
    pass


# ── Training Curves ───────────────────────────────────────────────────────
def plot_training_curves(histories: dict):
    """
    Plot combined loss + metric curves for all planes.
    histories: {plane: (hist_s1_df, hist_s2_df)}
    """
    for plane, (hist_s1, hist_s2) in histories.items():
        hist = pd.concat([hist_s1, hist_s2], ignore_index=True)
        hist['global_epoch'] = range(1, len(hist) + 1)

        fig, axes = plt.subplots(1, 2, figsize=(13, 4))
        axes[0].plot(hist['global_epoch'], hist['loss'], marker='o', label='Train Loss')
        axes[0].set(title=f'Training Loss — {plane}', xlabel='Epoch', ylabel='BCE Loss')
        axes[0].legend()

        for metric in ['AUC-ROC', 'F1-Score', 'Recall']:
            axes[1].plot(hist['global_epoch'], hist[metric], marker='o', label=metric)
        axes[1].set(title=f'Validation Metrics — {plane}', xlabel='Epoch', ylim=(0, 1))
        axes[1].legend()

        s1_len = len(hist_s1)
        for ax in axes:
            ax.axvline(s1_len + 0.5, color='gray', linestyle='--',
                       alpha=0.6, label='Stage 2 start')

        plt.tight_layout()
        path = f'{FIG_DIR}/training_curves_{plane}.png'
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f'Saved → {path}')
        plt.show()


# ── Weighted Ensemble ─────────────────────────────────────────────────────
def build_ensemble(val_loaders, models=None):
    """
    Load (or accept) per-plane models, run TTA validation, build a
    sagittal-weighted ensemble (axial 25 %, coronal 25 %, sagittal 50 %).
    Returns (ens_dataframe, best_threshold, metrics_dict).
    """
    if models is None:
        models = load_plane_models()

    weights   = {'axial': 0.25, 'coronal': 0.25, 'sagittal': 0.50}
    plane_agg = {}

    for plane in PLANES:
        if plane not in models:
            continue
        y_true, y_prob, case_ids = validate_tta(models[plane], val_loaders[plane])
        _, agg_df = exam_level_metrics(y_true, y_prob, case_ids)
        plane_agg[plane] = agg_df.set_index('case')[['prob', 'true']]

    ens = pd.DataFrame({p: plane_agg[p]['prob'] for p in plane_agg})
    ens['true'] = plane_agg['axial']['true']
    ens['prob'] = sum(ens[p] * weights[p] for p in plane_agg if p in weights)

    # Find best threshold on validation set
    best_acc, best_thresh = 0, 0.5
    for t in np.arange(0.05, 0.95, 0.01):
        acc = accuracy_score(ens['true'], (ens['prob'] > t).astype(int))
        if acc > best_acc:
            best_acc, best_thresh = acc, t

    y_true_exam = ens['true'].values
    y_prob_exam = ens['prob'].values
    y_pred_exam = (y_prob_exam > best_thresh).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true_exam, y_pred_exam).ravel()
    metrics = {
        'Accuracy':          accuracy_score(y_true_exam, y_pred_exam),
        'Precision':         precision_score(y_true_exam, y_pred_exam, zero_division=0),
        'Recall':            recall_score(y_true_exam, y_pred_exam, zero_division=0),
        'Specificity':       tn / (tn + fp) if (tn + fp) > 0 else 0,
        'F1-Score':          f1_score(y_true_exam, y_pred_exam, zero_division=0),
        'AUC-ROC':           roc_auc_score(y_true_exam, y_prob_exam),
        'Balanced Accuracy': balanced_accuracy_score(y_true_exam, y_pred_exam),
        'MCC':               matthews_corrcoef(y_true_exam, y_pred_exam),
        'Log Loss':          log_loss(y_true_exam, y_prob_exam),
        'Brier Score':       brier_score_loss(y_true_exam, y_prob_exam),
        'FPR':               fp / (fp + tn) if (fp + tn) > 0 else 0,
        'FNR':               fn / (fn + tp) if (fn + tp) > 0 else 0,
        'NPV':               tn / (tn + fn) if (tn + fn) > 0 else 0
    }

    print(f"\n── Final Exam-Level Metrics (Weighted Ensemble, Threshold={best_thresh:.2f}) ──")
    for k, v in metrics.items():
        print(f"  {k:<20}: {v:.4f}")

    return ens, best_thresh, metrics, y_true_exam, y_prob_exam, y_pred_exam


# ── ROC Curve ─────────────────────────────────────────────────────────────
def plot_roc_curve(y_true, y_prob):
    fpr_c, tpr_c, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr_c, tpr_c)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr_c, tpr_c, color='darkorange', lw=2,
             label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR) / Sensitivity')
    plt.title('ROC Curve — Final Ensemble')
    plt.legend(loc='lower right'); plt.grid(True)
    plt.savefig(f'{FIG_DIR}/roc_curve.png', dpi=150, bbox_inches='tight')
    print(f'Saved → {FIG_DIR}/roc_curve.png')
    plt.show()


# ── Confusion Matrix ──────────────────────────────────────────────────────
def plot_confusion_matrix(y_true, y_pred, threshold):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['ACL Intact', 'ACL Torn'],
                yticklabels=['ACL Intact', 'ACL Torn'], ax=ax)
    ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
    ax.set_title(f'Confusion Matrix — Ensemble (Thresh={threshold:.2f})')
    plt.tight_layout()
    plt.savefig(f'{FIG_DIR}/confusion_matrix.png', dpi=150, bbox_inches='tight')
    print(f'Saved → {FIG_DIR}/confusion_matrix.png')
    plt.show()


# ── Accuracy vs. Threshold ────────────────────────────────────────────────
def plot_threshold_sweep(ens_df):
    thresholds     = np.arange(0.05, 0.96, 0.01)
    accuracy_vals  = []
    best_acc, best_t = 0.0, 0.5

    for th in thresholds:
        acc = accuracy_score(ens_df['true'], (ens_df['prob'] > th).astype(int))
        accuracy_vals.append(acc)
        if acc > best_acc:
            best_acc, best_t = acc, th

    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, accuracy_vals, marker='o', linestyle='-', color='blue')
    plt.title('Ensemble Accuracy vs. Classification Threshold')
    plt.xlabel('Threshold'); plt.ylabel('Accuracy'); plt.grid(True)
    plt.axvline(x=best_t, color='red', linestyle='--',
                label=f'Best Threshold: {best_t:.2f} (Acc={best_acc:.4f})')
    plt.legend(); plt.tight_layout()
    plt.savefig(f'{FIG_DIR}/ensemble_accuracy_vs_threshold.png', dpi=150, bbox_inches='tight')
    print(f'Saved → {FIG_DIR}/ensemble_accuracy_vs_threshold.png')
    plt.show()


# ── Computational Performance ─────────────────────────────────────────────
def compute_performance_metrics(model, val_loader):
    """Measure inference speed, FLOPs, and parameter counts."""
    model.eval()
    total_time, num_slices = 0.0, 0

    with torch.no_grad():
        for images, _, _ in val_loader:
            images = images.to(device)
            t0 = time.time()
            _ = model(images)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            total_time  += time.time() - t0
            num_slices  += images.shape[0]

    avg_ms         = (total_time / num_slices) * 1000
    n_params       = sum(p.numel() for p in model.parameters())
    trainable      = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("\n── Computational Performance ──────────────────────────────")
    print(f"  Total Parameters  : {n_params:,}")
    print(f"  Trainable Params  : {trainable:,}")
    print(f"  Avg Inference Time: {avg_ms:.2f} ms/slice")
    print(f"  Throughput        : {1000/avg_ms:.1f} slices/sec")

    flops = None
    if THOP_AVAILABLE:
        dummy = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE).to(device)
        flops, _ = thop_profile(model, inputs=(dummy,), verbose=False)
        print(f"  FLOPs             : {flops/1e9:.3f} GFLOPs")
    else:
        print("  FLOPs             : install thop → pip install thop")

    comp_data = {
        'total_params':               n_params,
        'trainable_params':           trainable,
        'avg_inference_ms':           avg_ms,
        'throughput_slices_per_sec':  1000 / avg_ms,
        'flops_gflops':               (flops / 1e9) if flops else None
    }
    pd.DataFrame([comp_data]).to_csv(
        f'{RESULTS_DIR}/computational_metrics.csv', index=False
    )
    print(f'Saved → {RESULTS_DIR}/computational_metrics.csv')
    return comp_data


# ── Save Final Metrics ────────────────────────────────────────────────────
def save_final_metrics(metrics: dict):
    path = f'{RESULTS_DIR}/final_eval_metrics.csv'
    pd.DataFrame([metrics]).to_csv(path, index=False)
    print(f'Saved → {path}')


# ── Run Everything ────────────────────────────────────────────────────────
def run_full_evaluation(val_loaders, histories=None, models=None):
    """
    Convenience wrapper: runs training curves (if histories given),
    ensemble evaluation, all plots, and computational metrics.
    """
    if histories:
        plot_training_curves(histories)

    ens, best_thresh, metrics, y_true, y_prob, y_pred = build_ensemble(
        val_loaders, models
    )
    plot_roc_curve(y_true, y_prob)
    plot_confusion_matrix(y_true, y_pred, best_thresh)
    plot_threshold_sweep(ens)
    save_final_metrics(metrics)

    # Compute performance on axial loader with loaded sagittal model as proxy
    if models and 'sagittal' in models:
        compute_performance_metrics(models['sagittal'], val_loaders['axial'])

    return ens, metrics


if __name__ == '__main__':
    print("Import this module or call run_full_evaluation(val_loaders).")
