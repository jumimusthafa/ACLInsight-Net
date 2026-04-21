# -*- coding: utf-8 -*-
"""
acl_xai.py — ACLInsight-Net
Explainability: Grad-CAM visualization, qualitative panel figures,
and quantitative faithfulness metrics (deletion / insertion scores).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

from acl_config import (
    device, DATA_ROOT, VALID_ACL_CSV, FIG_DIR, RESULTS_DIR
)
from acl_dataset import val_transform, _MEAN, _STD


# ── Helper: load & preprocess a single MRI slice ──────────────────────────
def load_mid_slice(case_id: int, split='valid', plane='sagittal'):
    """
    Loads the middle slice of a .npy volume.
    Returns (raw_norm, display_img_float32, input_tensor_4d).
      raw_norm     : H×W float32 [0, 1]
      display_img  : H×W×3 float32 [0, 1]  (denormalized, for imshow)
      input_tensor : 1×3×H×W CUDA tensor
    """
    npy_path = f'{DATA_ROOT}/{split}/{plane}/{case_id:04d}.npy'
    vol      = np.load(npy_path)                     # (S, H, W) uint8
    mid      = vol.shape[0] // 2
    raw_norm = vol[mid].astype(np.float32) / 255.0   # [0, 1]

    rgb_float    = np.stack([raw_norm] * 3, axis=-1)
    img_tensor   = val_transform(image=rgb_float)['image']   # 3×H×W
    input_tensor = img_tensor.unsqueeze(0).to(device)

    # Denormalize for matplotlib
    mean_np    = np.array(_MEAN)
    std_np     = np.array(_STD)
    display    = img_tensor.permute(1, 2, 0).numpy() * std_np + mean_np
    display    = np.clip(display, 0, 1).astype(np.float32)

    return raw_norm, display, input_tensor, img_tensor


# ── Single Grad-CAM plot ──────────────────────────────────────────────────
def plot_gradcam(model, case_id: int, split='valid', plane='sagittal'):
    """
    Three-panel Grad-CAM figure:
    (a) original MRI  (b) Grad-CAM overlay  (c) heatmap only
    """
    model.eval()
    raw_norm, display_img, input_tensor, _ = load_mid_slice(
        case_id, split, plane
    )

    target_layer = [model.backbone.blocks[-1]]
    cam_engine   = GradCAM(model=model, target_layers=target_layer)

    grayscale_cam = cam_engine(
        input_tensor, targets=[ClassifierOutputTarget(0)]
    )[0]
    grayscale_cam = gaussian_filter(grayscale_cam, sigma=3)
    grayscale_cam = (
        (grayscale_cam - grayscale_cam.min())
        / (grayscale_cam.max() - grayscale_cam.min() + 1e-8)
    )

    cam_overlay = show_cam_on_image(
        display_img, grayscale_cam, use_rgb=True, image_weight=0.45
    )

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].imshow(display_img, cmap='gray')
    axes[0].set_title(f'(a) Original {plane.capitalize()} — Case {case_id:04d} (ACL+)', fontsize=11)
    axes[1].imshow(cam_overlay)
    axes[1].set_title('(b) Grad-CAM Activation', fontsize=11)
    im = axes[2].imshow(grayscale_cam, cmap='jet', vmin=0, vmax=1)
    axes[2].set_title('(c) CAM Heatmap Only', fontsize=11)
    plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
    for ax in axes:
        ax.axis('off')

    plt.suptitle('ACLInsight-Net — Grad-CAM Explainability', fontsize=13)
    plt.tight_layout()
    path = f'{FIG_DIR}/gradcam_{plane}.png'
    plt.savefig(path, dpi=200, bbox_inches='tight', facecolor='white')
    print(f'Saved → {path}')
    plt.show()


# ── Qualitative 4-panel helper ────────────────────────────────────────────
def make_qual_panel(case_id: int, model, cam_engine,
                    split='valid', plane='sagittal'):
    """
    Returns (raw_norm, cam_overlay, pred_mask, overlay_rgb, conf).
    """
    model.eval()
    raw_norm, _, input_tensor, img_tensor = load_mid_slice(case_id, split, plane)

    with torch.no_grad():
        conf = torch.sigmoid(model(input_tensor)).item()

    cam_raw    = cam_engine(input_tensor, targets=[ClassifierOutputTarget(0)])[0]
    cam_smooth = gaussian_filter(cam_raw, sigma=3)
    cam_norm   = (
        (cam_smooth - cam_smooth.min())
        / (cam_smooth.max() - cam_smooth.min() + 1e-8)
    )

    H, W        = raw_norm.shape
    cam_resized = cv2.resize(cam_norm, (W, H), interpolation=cv2.INTER_LINEAR)

    thr       = np.percentile(cam_resized, 85)   # top 15 % activation
    pred_mask = (cam_resized >= thr).astype(np.uint8) * 255
    kernel    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    pred_mask = cv2.morphologyEx(pred_mask, cv2.MORPH_OPEN,  kernel)
    pred_mask = cv2.morphologyEx(pred_mask, cv2.MORPH_CLOSE, kernel)

    raw_uint8   = (raw_norm * 255).astype(np.uint8)
    mri_bgr     = cv2.cvtColor(raw_uint8, cv2.COLOR_GRAY2BGR)
    red_layer   = np.zeros_like(mri_bgr)
    red_layer[pred_mask > 0] = (0, 0, 200)
    overlay_bgr = cv2.addWeighted(mri_bgr, 0.55, red_layer, 0.45, 0)
    overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)

    display_rgb = np.stack([raw_norm] * 3, axis=-1).astype(np.float32)
    cam_overlay = show_cam_on_image(
        display_rgb, cam_resized, use_rgb=True, image_weight=0.45
    )

    return raw_norm, cam_overlay, pred_mask, overlay_rgb, conf


# ── Qualitative 4-panel figure (IEEE style) ───────────────────────────────
def plot_qualitative_results(model, top_cases, n_cases=3, plane='sagittal'):
    """
    N-row × 4-column IEEE-style figure:
    (a) Input MRI  (b) Grad-CAM  (c) Activation region  (d) Overlay
    """
    target_layers = [model.backbone.blocks[-1]]
    cam_viz       = GradCAM(model=model, target_layers=target_layers)

    col_titles = [
        '(a) Input MRI\n(Sagittal)',
        '(b) Grad-CAM\nHeatmap',
        '(c) Activation\nRegion (top 15%)',
        '(d) Overlay on\nMRI'
    ]

    fig, axes = plt.subplots(
        n_cases, 4,
        figsize=(13, 4.5 * n_cases),
        gridspec_kw={'wspace': 0.05, 'hspace': 0.12}
    )

    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=11, fontweight='bold', pad=10)

    for row, case_id in enumerate(top_cases[:n_cases]):
        display, cam_norm, pred_mask, overlay, conf = make_qual_panel(
            case_id, model, cam_viz, plane=plane
        )
        imgs  = [display, cam_norm, pred_mask, overlay]
        cmaps = ['gray',   None,     'gray',    None]
        vmins = [0, 0, 0, None]
        vmaxs = [1, 1, 255, None]

        for col, (img, cmap, vmin, vmax) in enumerate(
            zip(imgs, cmaps, vmins, vmaxs)
        ):
            ax = axes[row, col]
            ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)

        axes[row, 0].set_ylabel(
            f'Case {case_id:04d}\n(Conf: {conf:.2f})',
            fontsize=9, rotation=90, labelpad=6
        )

    plt.suptitle(
        'ACLInsight-Net — Qualitative Explainability Results\n'
        'Grad-CAM localization on ACL-positive sagittal MRI slices',
        fontsize=12, y=1.01
    )
    path = f'{FIG_DIR}/qualitative_results.png'
    plt.savefig(path, dpi=200, bbox_inches='tight', facecolor='white')
    print(f'Saved → {path}')
    plt.show()


# ── Faithfulness: Deletion Score ──────────────────────────────────────────
def deletion_score(model, img_tensor, cam_map, steps=10):
    """
    Progressively mask top-k % most salient pixels.
    Lower AUC → explanation focuses on truly important pixels.
    """
    model.eval()
    flat_cam   = cam_map.flatten()
    sorted_idx = np.argsort(flat_cam)[::-1]
    h, w       = cam_map.shape
    total_px   = h * w
    confs      = []
    orig       = img_tensor.clone().to(device).unsqueeze(0)

    for step in range(steps + 1):
        n_masked = int((step / steps) * total_px)
        masked   = orig.clone()
        if n_masked > 0:
            ri = sorted_idx[:n_masked] // w
            ci = sorted_idx[:n_masked]  % w
            masked[0, :, ri, ci] = 0.0
        with torch.no_grad():
            confs.append(torch.sigmoid(model(masked)).item())

    x = np.linspace(0, 1, steps + 1)
    return float(np.trapz(confs, x)), confs


# ── Faithfulness: Insertion Score ─────────────────────────────────────────
def insertion_score(model, img_tensor, cam_map, steps=10):
    """
    Progressively reveal pixels on a blurred baseline.
    Higher AUC → model relies on the salient region.
    """
    model.eval()
    flat_cam   = cam_map.flatten()
    sorted_idx = np.argsort(flat_cam)[::-1]
    h, w       = cam_map.shape
    total_px   = h * w
    confs      = []
    orig       = img_tensor.clone().to(device).unsqueeze(0)
    baseline   = F.avg_pool2d(orig, kernel_size=15, stride=1, padding=7)

    for step in range(steps + 1):
        n_reveal = int((step / steps) * total_px)
        revealed = baseline.clone()
        if n_reveal > 0:
            ri = sorted_idx[:n_reveal] // w
            ci = sorted_idx[:n_reveal]  % w
            revealed[0, :, ri, ci] = orig[0, :, ri, ci]
        with torch.no_grad():
            confs.append(torch.sigmoid(model(revealed)).item())

    x = np.linspace(0, 1, steps + 1)
    return float(np.trapz(confs, x)), confs


# ── Proxy Localization Stats ───────────────────────────────────────────────
def cam_localization_stats(cam_map, threshold=0.5):
    high     = cam_map >= threshold
    focus    = float(high.mean() * 100)
    centroid = (
        np.argwhere(high).mean(axis=0) if high.any()
        else np.array([np.nan, np.nan])
    )
    return {
        'max_activation': float(cam_map.max()),
        'focus_area_pct': focus,
        'centroid_row':   centroid[0],
        'centroid_col':   centroid[1]
    }


# ── Run XAI Evaluation on N positive val samples ──────────────────────────
def run_xai_evaluation(model, val_dataset, n_xai=5):
    """
    Computes deletion/insertion scores + localization stats on the
    first n_xai ACL-positive validation samples.
    Returns (del_scores, ins_scores, loc_stats) lists.
    """
    target_layers  = [model.backbone.blocks[-1]]
    cam_engine_xai = GradCAM(model=model, target_layers=target_layers)

    del_scores, ins_scores, loc_stats = [], [], []
    last_del_curve = last_ins_curve = None
    count = 0

    for img_t, lbl, _ in val_dataset:
        if int(lbl) != 1:
            continue
        inp     = img_t.unsqueeze(0).to(device)
        cam_map = cam_engine_xai(inp, targets=[ClassifierOutputTarget(0)])[0]

        d_auc, d_curve = deletion_score(model, img_t, cam_map)
        i_auc, i_curve = insertion_score(model, img_t, cam_map)
        loc            = cam_localization_stats(cam_map)

        del_scores.append(d_auc)
        ins_scores.append(i_auc)
        loc_stats.append(loc)
        last_del_curve, last_ins_curve = d_curve, i_curve

        count += 1
        if count >= n_xai:
            break

    print(f"\n── XAI Quantitative Metrics (N={n_xai} ACL+ val samples) ──")
    print(f"  Deletion  AUC : {np.mean(del_scores):.4f} ± {np.std(del_scores):.4f}  ↓ lower = better")
    print(f"  Insertion AUC : {np.mean(ins_scores):.4f} ± {np.std(ins_scores):.4f}  ↑ higher = better")
    print(f"  Avg Focus Area: {np.mean([l['focus_area_pct'] for l in loc_stats]):.1f} %")
    print(f"  Avg Max Act.  : {np.mean([l['max_activation']  for l in loc_stats]):.4f}")

    # Plot faithfulness curves
    steps  = len(last_del_curve) - 1
    x_pct  = np.linspace(0, 100, steps + 1)
    plt.figure(figsize=(8, 4))
    plt.plot(x_pct, last_del_curve, 'r-o', markersize=4,
             label=f'Deletion  AUC={np.mean(del_scores):.3f}')
    plt.plot(x_pct, last_ins_curve, 'g-o', markersize=4,
             label=f'Insertion AUC={np.mean(ins_scores):.3f}')
    plt.xlabel('% Pixels Modified'); plt.ylabel('Model Confidence (Sigmoid)')
    plt.title('Faithfulness: Deletion & Insertion Curves')
    plt.legend(); plt.grid(True, alpha=0.4); plt.tight_layout()
    plt.savefig(f'{FIG_DIR}/faithfulness_curves.png', dpi=150, bbox_inches='tight')
    print(f'Saved → {FIG_DIR}/faithfulness_curves.png')
    plt.show()

    # Save XAI metrics CSV
    xai_data = {
        'deletion_score_mean':  float(np.mean(del_scores)),
        'deletion_score_std':   float(np.std(del_scores)),
        'insertion_score_mean': float(np.mean(ins_scores)),
        'insertion_score_std':  float(np.std(ins_scores)),
        'avg_focus_area_pct':   float(np.mean([l['focus_area_pct'] for l in loc_stats])),
        'avg_max_activation':   float(np.mean([l['max_activation']  for l in loc_stats])),
        'n_samples':            len(del_scores)
    }
    pd.DataFrame([xai_data]).to_csv(f'{RESULTS_DIR}/xai_metrics.csv', index=False)
    print(f'Saved → {RESULTS_DIR}/xai_metrics.csv')

    return del_scores, ins_scores, loc_stats


# ── Select high-confidence positive cases ────────────────────────────────
def find_top_cases(model, val_csv_path, data_root, split='valid',
                   plane='sagittal', min_conf=0.70, top_n=3):
    """Returns list of case_ids sorted by descending model confidence."""
    val_csv   = pd.read_csv(val_csv_path, header=None, names=['case', 'label'])
    pos_cases = val_csv[val_csv['label'] == 1]['case'].tolist()
    confident = []

    for cid in pos_cases:
        try:
            npy      = np.load(f'{data_root}/{split}/{plane}/{cid:04d}.npy')
            mid      = npy.shape[0] // 2
            raw_norm = npy[mid].astype(np.float32) / 255.0
            rgb      = np.stack([raw_norm] * 3, axis=-1)
            t        = val_transform(image=rgb)['image'].unsqueeze(0).to(device)
            with torch.no_grad():
                c = torch.sigmoid(model(t)).item()
            if c >= min_conf:
                confident.append((cid, c))
        except Exception:
            continue

    confident.sort(key=lambda x: -x[1])
    print(f"Found {len(confident)} high-confidence positives (≥{min_conf:.0%})")
    for cid, c in confident[:5]:
        print(f"  Case {cid:04d} — conf {c:.3f}")

    return [cid for cid, _ in confident[:top_n]]


if __name__ == '__main__':
    print("Import this module — do not run it standalone.")
