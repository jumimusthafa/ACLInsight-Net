# -*- coding: utf-8 -*-
"""
acl_crossval_ablation.py — ACLInsight-Net
5-Fold stratified cross-validation + ablation study.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold

from acl_config import (
    device, SEED, PLANES, FIG_DIR, RESULTS_DIR,
    BATCH_SIZE, NUM_WORKERS, LR_S1, TRAIN_ACL_CSV, TRAIN_DIR
)
from acl_dataset import (
    MRNetSliceDataset, train_transform, val_transform
)
from acl_model import ACLModel, freeze_backbone
from acl_train import train_one_epoch, validate, exam_level_metrics


# ── 5-Fold Stratified Cross-Validation ───────────────────────────────────
K_FOLDS      = 5
KFOLD_EPOCHS = 3      # Quick run — increase to EPOCHS_S1 for the paper


def run_kfold_cv(k_folds=K_FOLDS, kfold_epochs=KFOLD_EPOCHS):
    """
    Runs stratified k-fold CV on the training split (head-only for speed).
    Returns a DataFrame with per-fold metrics.
    """
    all_train = pd.read_csv(
        TRAIN_ACL_CSV, header=None, names=['case', 'label']
    )
    cases_arr  = all_train['case'].values
    labels_arr = all_train['label'].values

    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=SEED)
    fold_metrics_list = []

    print(f"Running {k_folds}-fold CV | {kfold_epochs} epochs/fold | head-only...")

    for fold, (tr_idx, va_idx) in enumerate(
        skf.split(cases_arr, labels_arr), 1
    ):
        tr_cases = set(cases_arr[tr_idx].tolist())
        va_cases = set(cases_arr[va_idx].tolist())

        # Build full dataset, then filter by case IDs
        tr_ds = MRNetSliceDataset(
            root_dir=TRAIN_DIR, csv_path=TRAIN_ACL_CSV,
            transform=train_transform
        )
        va_ds = MRNetSliceDataset(
            root_dir=TRAIN_DIR, csv_path=TRAIN_ACL_CSV,
            transform=val_transform
        )
        tr_ds.samples = [
            (p, s, l, c) for p, s, l, c in tr_ds.samples if c in tr_cases
        ]
        va_ds.samples = [
            (p, s, l, c) for p, s, l, c in va_ds.samples if c in va_cases
        ]

        tr_ldr = DataLoader(
            tr_ds, batch_size=BATCH_SIZE, shuffle=True,
            num_workers=NUM_WORKERS, pin_memory=True
        )
        va_ldr = DataLoader(
            va_ds, batch_size=BATCH_SIZE, shuffle=False,
            num_workers=NUM_WORKERS, pin_memory=True
        )

        fold_model = ACLModel().to(device)
        freeze_backbone(fold_model, freeze=True)
        opt = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, fold_model.parameters()),
            lr=LR_S1, weight_decay=1e-4
        )
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=kfold_epochs, eta_min=LR_S1 / 10
        )

        for ep in range(kfold_epochs):
            train_one_epoch(fold_model, tr_ldr, opt)
            sch.step()

        y_t, y_p, c_ids = validate(fold_model, va_ldr)
        m, _             = exam_level_metrics(y_t, y_p, c_ids)
        fold_metrics_list.append(m)
        print(
            f"  Fold {fold}/{k_folds} — "
            f"AUC: {m['AUC-ROC']:.4f} | F1: {m['F1-Score']:.4f} | "
            f"Acc: {m['Accuracy']:.4f} | MCC: {m['MCC']:.4f}"
        )

    return pd.DataFrame(fold_metrics_list)


def print_and_save_kfold(kf_df: pd.DataFrame):
    report_cols = [
        'AUC-ROC', 'Accuracy', 'F1-Score', 'Recall',
        'Specificity', 'Precision', 'Balanced Accuracy', 'MCC',
        'Log Loss', 'Brier Score'
    ]

    print(f"\n── {K_FOLDS}-Fold CV Summary ──────────────────────────────────────")
    print(f"  {'Metric':<22} {'Mean':>7}  {'Std':>7}")
    print("  " + "-" * 38)
    for col in report_cols:
        if col in kf_df.columns:
            print(
                f"  {col:<22} {kf_df[col].mean():.4f}  ±{kf_df[col].std():.4f}"
            )

    # Bar chart
    plot_cols = [
        'AUC-ROC', 'Accuracy', 'F1-Score', 'Recall',
        'Specificity', 'Balanced Accuracy', 'MCC'
    ]
    means  = [kf_df[c].mean() for c in plot_cols if c in kf_df]
    stds   = [kf_df[c].std()  for c in plot_cols if c in kf_df]
    labels = [c for c in plot_cols if c in kf_df]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(labels, means, yerr=stds, capsize=5, color='steelblue', alpha=0.8)
    ax.set_ylim(0, 1.1); ax.set_ylabel('Score')
    ax.set_title(f'{K_FOLDS}-Fold CV Results — ACLInsight-Net (mean ± std)')
    ax.grid(axis='y', alpha=0.3)
    plt.xticks(rotation=20, ha='right'); plt.tight_layout()
    plt.savefig(f'{FIG_DIR}/kfold_cv_results.png', dpi=150, bbox_inches='tight')
    print(f'Saved → {FIG_DIR}/kfold_cv_results.png')
    plt.show()

    # Save CSVs
    kf_df.to_csv(f'{RESULTS_DIR}/kfold_per_fold.csv', index_label='fold')
    summary = kf_df.agg(['mean', 'std']).T
    summary.columns = ['mean', 'std']
    summary.to_csv(f'{RESULTS_DIR}/kfold_summary.csv')
    print(f'Saved → {RESULTS_DIR}/kfold_per_fold.csv')
    print(f'Saved → {RESULTS_DIR}/kfold_summary.csv')


# ── Ablation Study ────────────────────────────────────────────────────────
def run_ablation_study(metrics: dict, hist_s1=None, hist_s2=None):
    """
    Compares three model variants:
      A) Stage-1 only (frozen backbone)
      B) Stage-2 fine-tune (last 2 blocks)
      C) Full ACLInsight-Net (weighted ensemble + TTA)
    metrics: final exam-level metrics dict from acl_evaluate.py.
    hist_s1/hist_s2: optional DataFrames from run_training_stage().
    """
    ablation_results = []

    if hist_s1 is not None:
        s1_best = hist_s1.loc[hist_s1['AUC-ROC'].idxmax()]
        ablation_results.append({
            'Variant':   'Stage-1 Only\n(Frozen backbone)',
            'Accuracy':  s1_best.get('Accuracy',    float('nan')),
            'AUC-ROC':   s1_best.get('AUC-ROC',     float('nan')),
            'F1-Score':  s1_best.get('F1-Score',     float('nan')),
            'Recall':    s1_best.get('Recall',       float('nan')),
            'Specificity': s1_best.get('Specificity', float('nan')),
        })

    if hist_s2 is not None:
        s2_best = hist_s2.loc[hist_s2['AUC-ROC'].idxmax()]
        ablation_results.append({
            'Variant':   'Stage-2 Fine-Tune\n(Last 2 blocks)',
            'Accuracy':  s2_best.get('Accuracy',    float('nan')),
            'AUC-ROC':   s2_best.get('AUC-ROC',     float('nan')),
            'F1-Score':  s2_best.get('F1-Score',     float('nan')),
            'Recall':    s2_best.get('Recall',       float('nan')),
            'Specificity': s2_best.get('Specificity', float('nan')),
        })

    ablation_results.append({
        'Variant':     'Full ACLInsight-Net\n(Weighted Ensemble TTA)',
        'Accuracy':    metrics['Accuracy'],
        'AUC-ROC':     metrics['AUC-ROC'],
        'F1-Score':    metrics['F1-Score'],
        'Recall':      metrics['Recall'],
        'Specificity': metrics['Specificity'],
    })

    abl_df = pd.DataFrame(ablation_results).set_index('Variant')
    cols   = ['AUC-ROC', 'Accuracy', 'F1-Score', 'Recall', 'Specificity']
    print("\n── Ablation Study ────────────────────────────────────────────────")
    print(abl_df[cols].to_string(float_format='{:.4f}'.format))

    # Bar chart
    x      = np.arange(len(cols))
    width  = 0.25
    colors = ['#4C72B0', '#DD8452', '#55A868']
    short  = [r.split('\n')[0] for r in abl_df.index]

    fig, ax = plt.subplots(figsize=(11, 5))
    for i, (row, color, lbl) in enumerate(
        zip(ablation_results, colors, short)
    ):
        vals = [row[m] for m in cols]
        ax.bar(x + i * width, vals, width, label=lbl, color=color)

    ax.set_xticks(x + width); ax.set_xticklabels(cols)
    ax.set_ylim(0.5, 1.05); ax.set_ylabel('Score')
    ax.set_title('Ablation Study — ACLInsight-Net')
    ax.legend(); ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{FIG_DIR}/ablation_study.png', dpi=150, bbox_inches='tight')
    print(f'Saved → {FIG_DIR}/ablation_study.png')
    plt.show()

    # Save CSV
    abl_df.reset_index().to_csv(
        f'{RESULTS_DIR}/ablation_study.csv', index=False
    )
    print(f'Saved → {RESULTS_DIR}/ablation_study.csv')

    return abl_df


if __name__ == '__main__':
    # Quick standalone run (needs dataset paths set in acl_config.py)
    kf_df = run_kfold_cv()
    print_and_save_kfold(kf_df)
