# -*- coding: utf-8 -*-
"""
acl_main.py — ACLInsight-Net
Master entry point: runs the full pipeline end-to-end.

Module layout (mirrors the BrainTumor repo structure):
  acl_config.py             → seeds, hyperparameters, paths
  acl_dataset.py            → augmentations, MRNetSliceDataset, DataLoaders
  acl_model.py              → ACLModel, freeze/unfreeze helpers
  acl_train.py              → loss, train loop, TTA, exam-level metrics
  acl_evaluate.py           → ensemble, ROC, CM, threshold sweep, perf metrics
  acl_xai.py                → Grad-CAM, qualitative panels, deletion/insertion
  acl_crossval_ablation.py  → 5-fold CV + ablation study
"""

# ── Imports ───────────────────────────────────────────────────────────────
from acl_config   import device, VALID_ACL_CSV, DATA_ROOT
from acl_dataset  import build_loaders
from acl_train    import train_all_planes, load_plane_models
from acl_evaluate import run_full_evaluation
from acl_xai      import (
    find_top_cases, plot_gradcam,
    plot_qualitative_results, run_xai_evaluation
)
from acl_crossval_ablation import run_kfold_cv, print_and_save_kfold, run_ablation_study


def main():
    # ── 1. Build DataLoaders ──────────────────────────────────────────────
    train_datasets, val_datasets, train_loaders, val_loaders = build_loaders()

    # ── 2. Train (set TRAIN_MODEL=False to skip and load existing ckpts) ──
    TRAIN_MODEL = True
    histories   = train_all_planes(train_loaders, val_loaders, train=TRAIN_MODEL)
    models      = load_plane_models()

    # ── 3. Final Evaluation ───────────────────────────────────────────────
    # Pass the per-plane histories (if available) so training curves are plotted
    hist_args = None
    if histories:
        # Use sagittal histories as representative curves
        hist_args = {
            plane: (h1, h2) for plane, (h1, h2) in histories.items()
        }

    ens, metrics = run_full_evaluation(
        val_loaders, histories=hist_args, models=models
    )

    # ── 4. XAI (Grad-CAM) ────────────────────────────────────────────────
    if 'sagittal' in models:
        model_sag  = models['sagittal']
        top_cases  = find_top_cases(
            model_sag, VALID_ACL_CSV, DATA_ROOT,
            split='valid', plane='sagittal', min_conf=0.70, top_n=3
        )
        plot_gradcam(model_sag, top_cases[0])
        plot_qualitative_results(model_sag, top_cases, n_cases=3)

        val_ds_sag = val_datasets['sagittal']
        run_xai_evaluation(model_sag, val_ds_sag, n_xai=5)

    # ── 5. Cross-Validation ───────────────────────────────────────────────
    kf_df = run_kfold_cv()
    print_and_save_kfold(kf_df)

    # ── 6. Ablation Study ─────────────────────────────────────────────────
    # Optionally pass Stage-1/Stage-2 history for a single representative plane
    if histories and 'sagittal' in histories:
        hist_s1, hist_s2 = histories['sagittal']
    else:
        hist_s1 = hist_s2 = None

    run_ablation_study(metrics, hist_s1=hist_s1, hist_s2=hist_s2)

    print("\n✅ ACLInsight-Net pipeline complete.")


if __name__ == '__main__':
    main()
