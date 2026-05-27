#!/usr/bin/env python3
"""
03_mvpa.py — Whole-brain SVM decoding from trial beta weights (batch/SLURM version)

Usage:
    python 03_mvpa.py --subject sub-1
    python 03_mvpa.py --subject sub-1 --analysis-name faces_only --conditions face scrambledpix

Mirrors mvpa_analysis.ipynb but runs non-interactively.
"""
import argparse, json, os, sys, time, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import (LeaveOneGroupOut, StratifiedKFold,
                                     cross_val_predict)
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             confusion_matrix, classification_report,
                             roc_auc_score, ConfusionMatrixDisplay)
from nilearn import image, plotting
from nilearn.maskers import NiftiMasker
try:
    import ants
    ANTS_AVAILABLE = True
except ImportError:
    ANTS_AVAILABLE = False

# ── CLI ────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Whole-brain MVPA (SVM decoding)")
    p.add_argument("--subject",       required=True)
    p.add_argument("--analysis-name", default="all_conditions_lss")
    p.add_argument("--beta-method",   default="lss", choices=["lsa","lss"])
    p.add_argument("--conditions",    nargs='*', default=None,
                   help="Subset of conditions (default: all)")
    p.add_argument("--feature-sel",   default="anova",
                   choices=["all","anova"])
    p.add_argument("--anova-pct",     type=float, default=10.0,
                   help="Top-N percentile of voxels to keep (ANOVA)")
    p.add_argument("--svm-c",         type=float, default=1.0)
    p.add_argument("--svm-kernel",    default="linear")
    p.add_argument("--leave-one-run-out", action="store_true", default=True)
    p.add_argument("--n-splits",         type=int, default=5)
    p.add_argument("--n-permutations",   type=int, default=1000,
                   help="Label-shuffle permutations for significance test "
                        "(0 = skip permutation test)")
    p.add_argument("--perm-seed",        type=int, default=0,
                   help="RNG seed for permutation sampling (default 0)")
    return p.parse_args()

# ── Paths ──────────────────────────────────────────────────────────────────────
FMRIPREP_ROOT = "/pl/active/courses/2026_summer/neuroclass2026/ds000105/derivatives/fmriprep"
DERIV_ROOT    = "/pl/active/courses/2026_summer/neuroclass2026/ds000105/derivatives"
_LPS_TO_RAS   = np.diag([-1., -1., 1., 1.])

def load_beta_matrix(manifest_csv, mask_img, conditions=None):
    """Load beta images, stack into (n_trials × n_voxels) matrix."""
    manifest = pd.read_csv(manifest_csv)
    if conditions:
        manifest = manifest[manifest['condition'].isin(conditions)].copy()
    masker = NiftiMasker(mask_img=mask_img, standardize=False, detrend=False)
    masker.fit()
    X_rows, labels, runs = [], [], []
    for _, row in manifest.iterrows():
        img   = nib.load(row['beta_file'])
        x     = masker.transform(img).ravel()
        X_rows.append(x)
        labels.append(row['condition'])
        runs.append(row['run'])
    X      = np.vstack(X_rows)
    y_raw  = np.array(labels)
    runs   = np.array(runs, dtype=int)
    # Drop all-zero voxel columns (outside mask edge effects)
    active = (X != 0).any(axis=0)
    n_mask = X.shape[1]
    X      = X[:, active]
    return X, y_raw, runs, masker, active, n_mask

def warp_to_mni(func_nifti, func_ref_path, mni_to_t1w, t1w_to_func, out_path):
    """Warp a func-space NIfTI to MNI using ANTs (func → MNI pullback)."""
    mni_ref_path = ("/usr/share/fsl/data/standard/"
                    "MNI152_T1_2mm_brain.nii.gz")  # adjust if needed
    if not os.path.exists(mni_ref_path):
        return None
    imp_ants = ants.image_read(func_nifti)
    mni_ref  = ants.image_read(mni_ref_path)
    warped   = ants.apply_transforms(
        fixed         = mni_ref,
        moving        = imp_ants,
        transformlist = [t1w_to_func, mni_to_t1w],
        whichtoinvert = [False, False],
        interpolator  = "linear",
    )
    spacing   = np.array(warped.spacing)
    direction = np.array(warped.direction).reshape(3, 3)
    origin    = np.array(warped.origin)
    aff_lps         = np.eye(4)
    aff_lps[:3, :3] = direction * spacing
    aff_lps[:3,  3] = origin
    aff_ras   = _LPS_TO_RAS @ aff_lps
    nib_warped = nib.Nifti1Image(warped.numpy().astype(np.float32), aff_ras)
    nib_warped.to_filename(out_path)
    return out_path

# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    args          = parse_args()
    SUBJECT       = args.subject
    ANALYSIS_NAME = args.analysis_name
    t_start       = time.perf_counter()

    output_dir = os.path.join(DERIV_ROOT, "mvpa", ANALYSIS_NAME, SUBJECT)
    os.makedirs(output_dir, exist_ok=True)

    print(f"{'='*60}")
    print(f"  MVPA: {SUBJECT}  analysis={ANALYSIS_NAME}")
    print(f"{'='*60}")

    # ── Load data ────────────────────────────────────────────────────────────
    manifest_csv = os.path.join(DERIV_ROOT, "beta_weights", SUBJECT,
                                f"beta_manifest_{args.beta_method}.csv")
    if not os.path.exists(manifest_csv):
        print(f"ERROR: manifest not found: {manifest_csv}"); sys.exit(1)

    mask_file = os.path.join(DERIV_ROOT, "beta_weights", SUBJECT, "warped_masks",
                             f"{SUBJECT}_task-objectviewing_run-01"
                             "_desc-warpedGroupMask_funcspace.nii.gz")
    if not os.path.exists(mask_file):
        print("Warped mask not found — deriving from first beta image...")
        first_beta = pd.read_csv(manifest_csv).iloc[0]['beta_file']
        mask_file  = None  # NiftiMasker will compute from data

    mask_img = image.load_img(mask_file) if mask_file else None
    X, y_raw, runs_arr, masker, active_voxels, n_mask_voxels = \
        load_beta_matrix(manifest_csv, mask_img, args.conditions)

    le = LabelEncoder(); y = le.fit_transform(y_raw)
    conditions = list(le.classes_)
    n_active   = int(active_voxels.sum())

    print(f"\n  Conditions  : {conditions}  ({len(conditions)})")
    print(f"  Trials      : {len(y)}")
    print(f"  Mask voxels : {n_mask_voxels}")
    print(f"  Active vox  : {n_active}")

    # ── Feature selection ────────────────────────────────────────────────────
    fs_mode = args.feature_sel
    if fs_mode == "anova":
        selector   = SelectPercentile(f_classif, percentile=args.anova_pct)
        selector.fit(X, y)
        selected   = selector.get_support()
        n_selected = int(selected.sum())
        print(f"  ANOVA selection: {n_selected} / {n_active} voxels (top {args.anova_pct}%)")
        # Save ANOVA F-score map
        F_scores   = selector.scores_
        full_f     = np.zeros(n_mask_voxels)
        full_f[active_voxels] = F_scores
        F_img = masker.inverse_transform(full_f)
        F_img.to_filename(os.path.join(output_dir, f"{SUBJECT}_anova_f_scores.nii.gz"))
        sel_full = np.zeros(n_mask_voxels)
        sel_full[active_voxels] = selected.astype(float)
        sel_img = masker.inverse_transform(sel_full)
        sel_img.to_filename(os.path.join(output_dir, f"{SUBJECT}_anova_selected_mask.nii.gz"))
    else:
        selector   = None
        n_selected = n_active

    # ── Cross-validation ─────────────────────────────────────────────────────
    if args.leave_one_run_out:
        cv      = LeaveOneGroupOut()
        groups  = runs_arr
        cv_name = "Leave-One-Run-Out"
    else:
        cv      = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=42)
        groups  = None
        cv_name = f"Stratified {args.n_splits}-Fold"

    svm  = SVC(C=args.svm_c, kernel=args.svm_kernel,
               class_weight='balanced', random_state=42,
               probability=True)
    pipe = Pipeline([
        ("fs",  selector if selector else "passthrough"),
        ("svm", svm),
    ])

    print(f"\n  Running {cv_name} CV ...")
    t0     = time.perf_counter()
    y_pred = cross_val_predict(pipe, X, y, cv=cv, groups=groups, n_jobs=-1)
    y_proba = cross_val_predict(pipe, X, y, cv=cv, groups=groups,
                                method='predict_proba', n_jobs=-1)
    elapsed_cv = time.perf_counter() - t0

    # Per-fold accuracies
    fold_accs = []
    splits = list((cv.split(X, y, groups) if groups is not None
                   else cv.split(X, y)))
    for train, test in splits:
        fold_accs.append(accuracy_score(y[test], y_pred[test]))

    mean_acc = np.mean(fold_accs)
    sem_acc  = np.std(fold_accs) / np.sqrt(len(fold_accs))
    bal_acc  = balanced_accuracy_score(y, y_pred)

    print(f"  CV done in {elapsed_cv:.1f}s")
    print(f"  Mean accuracy   : {mean_acc*100:.2f}% ± {sem_acc*100:.2f}% SEM")
    print(f"  Balanced acc    : {bal_acc*100:.2f}%")
    print(f"  Chance          : {100/len(conditions):.1f}%")
    print(classification_report(y, y_pred, target_names=le.classes_))

    # ── Permutation test ─────────────────────────────────────────────────────
    p_value   = None
    null_accs = np.array([])
    if args.n_permutations > 0:
        print(f"\n  Running permutation test ({args.n_permutations} permutations) ...")
        rng      = np.random.default_rng(seed=args.perm_seed)
        null_accs = []
        t_perm   = time.perf_counter()

        for perm_i in range(args.n_permutations):
            y_shuffled = rng.permutation(y)
            y_null     = cross_val_predict(
                pipe, X, y_shuffled, cv=cv,
                groups=groups, n_jobs=-1
            )
            null_accs.append(accuracy_score(y_shuffled, y_null))
            if (perm_i + 1) % 100 == 0:
                elapsed_perm = time.perf_counter() - t_perm
                rate         = (perm_i + 1) / elapsed_perm
                eta          = (args.n_permutations - perm_i - 1) / rate
                print(f"    {perm_i+1}/{args.n_permutations}  "
                      f"null mean={np.mean(null_accs)*100:.2f}%  "
                      f"ETA {eta/60:.1f} min")

        null_accs = np.array(null_accs)
        p_value   = float((null_accs >= mean_acc).mean())
        elapsed_perm = time.perf_counter() - t_perm

        print(f"  Permutation test done in {elapsed_perm/60:.1f} min")
        print(f"  Observed accuracy : {mean_acc*100:.2f}%")
        print(f"  Null mean ± SD    : {null_accs.mean()*100:.2f}% ± {null_accs.std()*100:.2f}%")
        print(f"  p-value           : {p_value:.4f}  "
              f"({'*significant*' if p_value < 0.05 else 'not significant'} at α=0.05)")

        # Save null distribution as CSV
        pd.DataFrame({'null_accuracy': null_accs}).to_csv(
            os.path.join(output_dir, 'permutation_null_distribution.csv'), index=False)

        # Plot null distribution
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(null_accs * 100, bins=40, color='steelblue',
                edgecolor='k', alpha=0.8, label='Null distribution')
        ax.axvline(mean_acc * 100, color='crimson', lw=2, ls='--',
                   label=f'Observed: {mean_acc*100:.2f}%')
        ax.axvline(100 / len(conditions), color='gray', lw=1.2, ls=':',
                   label=f'Chance: {100/len(conditions):.1f}%')
        ax.set_xlabel('Accuracy (%)')
        ax.set_ylabel('Count')
        ax.set_title(
            f'Permutation test  (n={args.n_permutations},  p={p_value:.4f})\n'
            f'{SUBJECT} — {ANALYSIS_NAME}'
        )
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'permutation_test.png'), dpi=120)
        plt.close()
        print(f"  Saved: permutation_test.png")
    else:
        print("\n  Permutation test skipped (--n-permutations 0)")

    # ── Confusion matrix plot ────────────────────────────────────────────────
    cm     = confusion_matrix(y, y_pred, normalize='true')
    cm_raw = confusion_matrix(y, y_pred)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ConfusionMatrixDisplay(cm_raw, display_labels=le.classes_).plot(
        ax=axes[0], colorbar=False, cmap='Blues')
    axes[0].set_title('Confusion matrix (raw counts)')
    ConfusionMatrixDisplay(cm, display_labels=le.classes_).plot(
        ax=axes[1], colorbar=True, cmap='Blues', values_format='.2f')
    axes[1].set_title('Confusion matrix (row-normalised)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=120)
    plt.close()

    # ── Per-condition accuracy ────────────────────────────────────────────────
    per_cond = {c: cm[i, i] for i, c in enumerate(le.classes_)}
    fig, ax  = plt.subplots(figsize=(max(5, len(conditions)*1.2), 4))
    bars = ax.bar(per_cond.keys(), [v*100 for v in per_cond.values()],
                  color='steelblue', edgecolor='k', alpha=0.85)
    ax.axhline(100/len(conditions), color='gray', lw=1, ls='--', label='Chance')
    ax.bar_label(bars, fmt='%.1f%%', padding=3, fontsize=9)
    ax.set_ylim(0, 110); ax.set_ylabel('Accuracy (%)');
    ax.set_title(f'Per-condition decoding — mean: {mean_acc*100:.2f}%'); ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'per_condition_accuracy.png'), dpi=120)
    plt.close()

    # ── Importance map ───────────────────────────────────────────────────────
    print("\n  Computing importance maps...")
    pipe_full = Pipeline([
        ("fs",  selector if selector else "passthrough"),
        ("svm", SVC(C=args.svm_c, kernel=args.svm_kernel,
                    class_weight='balanced', random_state=42)),
    ])
    pipe_full.fit(X, y)
    svm_full = pipe_full.named_steps['svm']
    coef = svm_full.coef_
    importance_vec_selected = np.mean(np.abs(coef), axis=0)

    fs_step = pipe_full.named_steps['fs']
    if fs_step != "passthrough" and hasattr(fs_step, 'get_support'):
        support = fs_step.get_support()
        imp_active = np.zeros(X.shape[1])
        imp_active[support] = importance_vec_selected
    else:
        imp_active = importance_vec_selected

    imp_full = np.zeros(n_mask_voxels)
    imp_full[active_voxels] = imp_active
    imp_img  = masker.inverse_transform(imp_full)
    imp_path = os.path.join(output_dir, f"{SUBJECT}_importance_map_funcspace.nii.gz")
    imp_img.to_filename(imp_path)
    print(f"  Saved: {imp_path}")

    # Plot importance map
    fig, ax = plt.subplots(figsize=(14, 4))
    plotting.plot_stat_map(
        imp_img, display_mode='z', cut_coords=7, colorbar=True,
        title='SVM importance map (func space)', cmap='hot',
        threshold=np.percentile(imp_full[imp_full > 0], 50) if (imp_full > 0).any() else None,
        axes=ax)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'importance_map.png'), dpi=120)
    plt.close()

    # ── CSV outputs ──────────────────────────────────────────────────────────
    sub = SUBJECT; ana = ANALYSIS_NAME; out = output_dir
    # Confusion matrix normalised
    cm_norm_df = pd.DataFrame(cm, index=pd.Index(list(le.classes_), name='true_condition'),
                               columns=list(le.classes_))
    cm_norm_df.insert(0, 'subject', sub); cm_norm_df.insert(1, 'analysis_name', ana)
    cm_norm_df.to_csv(os.path.join(out, 'confusion_matrix_normalised.csv'))
    # Confusion matrix raw
    cm_raw_df = pd.DataFrame(cm_raw, index=pd.Index(list(le.classes_), name='true_condition'),
                              columns=list(le.classes_))
    cm_raw_df.insert(0, 'subject', sub); cm_raw_df.insert(1, 'analysis_name', ana)
    cm_raw_df.to_csv(os.path.join(out, 'confusion_matrix_raw.csv'))
    # Accuracy metrics
    acc_row = {'subject': sub, 'analysis_name': ana,
               'mean_accuracy': round(float(mean_acc), 4),
               'sem_accuracy':  round(float(sem_acc),  4),
               'balanced_accuracy': round(float(bal_acc), 4),
               'chance_level': round(1/len(conditions), 4),
               'n_trials': int(len(y)), 'n_conditions': int(len(conditions)),
               'cv_strategy': cv_name, 'feature_selection': fs_mode,
               'n_voxels_selected': int(n_selected),
               'permutation_p': round(float(p_value), 4) if p_value is not None else None,
               'n_permutations': args.n_permutations,
               'null_mean': round(float(null_accs.mean()), 4) if len(null_accs) else None,
               'null_sd':   round(float(null_accs.std()),  4) if len(null_accs) else None}
    for i, acc in enumerate(fold_accs):
        acc_row[f'acc_run_{np.unique(runs_arr)[i]:02d}'] = round(float(acc), 4)
    for i, cond in enumerate(le.classes_):
        acc_row[f'acc_{cond}'] = round(float(cm[i, i]), 4)
    pd.DataFrame([acc_row]).to_csv(os.path.join(out, 'accuracy_metrics.csv'), index=False)
    # AUC
    try:
        auc_vals = roc_auc_score(y, y_proba, multi_class='ovr', average=None)
        auc_df   = pd.DataFrame({'subject': sub, 'analysis_name': ana,
                                 'condition': list(le.classes_),
                                 'auc_ovr': [round(float(v), 4) for v in auc_vals]})
        auc_df.to_csv(os.path.join(out, 'auc_per_condition.csv'), index=False)
    except Exception as e:
        print(f"  AUC skipped: {e}")

    print(f"\n  CSVs saved to: {out}/")

    # ── Report ───────────────────────────────────────────────────────────────
    elapsed = time.perf_counter() - t_start
    report  = dict(subject=SUBJECT, analysis_name=ANALYSIS_NAME,
                   conditions=list(le.classes_), n_trials=int(len(y)),
                   mean_accuracy=round(float(mean_acc), 4),
                   balanced_accuracy=round(float(bal_acc), 4),
                   chance_level=round(1/len(conditions), 4),
                   cv_strategy=cv_name, feature_selection=fs_mode,
                   n_voxels_selected=int(n_selected),
                   fold_accuracies=[round(float(a), 4) for a in fold_accs],
                   permutation_p=round(float(p_value), 4) if p_value is not None else None,
                   n_permutations=args.n_permutations,
                   null_mean=round(float(null_accs.mean()), 4) if len(null_accs) else None,
                   null_sd=round(float(null_accs.std()),   4) if len(null_accs) else None,
                   elapsed_s=round(elapsed, 1))
    with open(os.path.join(output_dir, 'mvpa_report.json'), 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\n  Done in {elapsed/60:.1f} min")
    print(f"  Outputs: {output_dir}/")

if __name__ == "__main__":
    main()
