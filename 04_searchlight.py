#!/usr/bin/env python3
"""
04_searchlight.py — Voxel-wise searchlight decoding (batch/SLURM version)

Usage:
    python 04_searchlight.py --subject sub-1
    python 04_searchlight.py --subject sub-1 --radius 8 --n-jobs 8

Mirrors searchlight_analysis.ipynb but runs non-interactively.
"""
import argparse, json, os, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold
from sklearn.metrics import roc_auc_score
from nilearn import image, plotting
from nilearn.maskers import NiftiMasker
from nilearn.decoding import SearchLight
try:
    import ants
    ANTS_AVAILABLE = True
except ImportError:
    ANTS_AVAILABLE = False

# ── CLI ────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Searchlight decoding")
    p.add_argument("--subject",       required=True)
    p.add_argument("--analysis-name", default="searchlight_r6_lss")
    p.add_argument("--beta-method",   default="lss", choices=["lsa","lss"])
    p.add_argument("--conditions",    nargs='*', default=None)
    p.add_argument("--radius",        type=float, default=6.0,
                   help="Searchlight radius in mm")
    p.add_argument("--svm-c",         type=float, default=1.0)
    p.add_argument("--n-jobs",        type=int,   default=-1)
    p.add_argument("--leave-one-run-out", action="store_true", default=True)
    p.add_argument("--n-splits",      type=int, default=5)
    p.add_argument("--top-n-peaks",      type=int,   default=20)
    p.add_argument("--acc-threshold",    type=float, default=None,
                   help="Accuracy threshold for reporting; default = chance + 0.1")
    p.add_argument("--n-permutations",   type=int,   default=100,
                   help="Label-shuffle permutations for significance test. "
                        "Each permutation re-runs the full searchlight, so "
                        "keep this low (50–200) relative to MVPA. 0 = skip.")
    p.add_argument("--perm-seed",        type=int,   default=0,
                   help="RNG seed for permutation sampling (default 0)")
    return p.parse_args()

# ── Paths ──────────────────────────────────────────────────────────────────────
FMRIPREP_ROOT = "/pl/active/courses/2026_summer/neuroclass2026/ds000105/derivatives/fmriprep"
DERIV_ROOT    = "/pl/active/courses/2026_summer/neuroclass2026/ds000105/derivatives"
_LPS_TO_RAS   = np.diag([-1., -1., 1., 1.])

# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    args          = parse_args()
    SUBJECT       = args.subject
    ANALYSIS_NAME = args.analysis_name
    t_start       = time.perf_counter()

    output_dir = os.path.join(DERIV_ROOT, "mvpa", ANALYSIS_NAME, SUBJECT)
    os.makedirs(output_dir, exist_ok=True)

    print(f"{'='*60}")
    print(f"  SEARCHLIGHT: {SUBJECT}  r={args.radius}mm  analysis={ANALYSIS_NAME}")
    print(f"{'='*60}")

    # ── Load manifest and labels ──────────────────────────────────────────────
    manifest_csv = os.path.join(DERIV_ROOT, "beta_weights", SUBJECT,
                                f"beta_manifest_{args.beta_method}.csv")
    if not os.path.exists(manifest_csv):
        print(f"ERROR: manifest not found: {manifest_csv}"); sys.exit(1)

    manifest = pd.read_csv(manifest_csv)
    if args.conditions:
        manifest = manifest[manifest['condition'].isin(args.conditions)].copy()

    le       = LabelEncoder()
    y        = le.fit_transform(manifest['condition'].values)
    runs_arr = manifest['run'].values.astype(int)
    conditions = list(le.classes_)
    chance   = 1.0 / len(conditions)

    print(f"\n  Conditions : {conditions}  ({len(conditions)})")
    print(f"  Trials     : {len(y)}")
    print(f"  Chance     : {chance*100:.1f}%")

    # ── Load mask ─────────────────────────────────────────────────────────────
    mask_file = os.path.join(DERIV_ROOT, "beta_weights", SUBJECT, "warped_masks",
                             f"{SUBJECT}_task-objectviewing_run-01"
                             "_desc-warpedGroupMask_funcspace.nii.gz")
    if not os.path.exists(mask_file):
        print("Warped mask not found — using first beta image as reference.")
        mask_file = manifest.iloc[0]['beta_file']
    mask_img = image.load_img(mask_file)
    process_mask_img = mask_img  # full brain search

    # ── Build 4D NIfTI of beta images ─────────────────────────────────────────
    print("\n  Stacking beta images into 4D NIfTI...")
    beta_imgs = [nib.load(p) for p in manifest['beta_file']]
    imgs_4d   = image.concat_imgs(beta_imgs)
    print(f"  4D shape: {imgs_4d.shape}")

    # ── Cross-validation ──────────────────────────────────────────────────────
    if args.leave_one_run_out:
        cv      = LeaveOneGroupOut()
        groups  = runs_arr
        cv_name = "Leave-One-Run-Out"
    else:
        cv      = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=42)
        groups  = None
        cv_name = f"Stratified {args.n_splits}-Fold"

    estimator = SVC(C=args.svm_c, kernel='linear',
                    class_weight='balanced', random_state=42)

    # ── SearchLight ───────────────────────────────────────────────────────────
    sl = SearchLight(
        mask_img         = mask_img,
        process_mask_img = process_mask_img,
        radius           = args.radius,
        estimator        = estimator,
        cv               = cv,
        scoring          = 'accuracy',
        n_jobs           = args.n_jobs,
        verbose          = 1,
    )

    n_centres = int((process_mask_img.get_fdata() > 0).sum())
    print(f"\n  Searchlight: {n_centres} centres  radius={args.radius}mm  CV={cv_name}")
    print(f"  Estimated runtime: several minutes for a whole-brain mask.")

    t0 = time.perf_counter()
    if groups is not None:
        sl.fit(imgs_4d, y, groups=groups)
    else:
        sl.fit(imgs_4d, y)
    elapsed_sl = time.perf_counter() - t0
    print(f"  Searchlight complete in {elapsed_sl/60:.1f} min")

    # ── Pack into NIfTI ────────────────────────────────────────────────────────
    scores_arr = sl.scores_
    scores_img = image.new_img_like(mask_img, scores_arr, copy_header=True)

    raw_path = os.path.join(output_dir,
                            f"{SUBJECT}_searchlight_accuracy_funcspace.nii.gz")
    scores_img.to_filename(raw_path)
    print(f"  Saved: {raw_path}")

    # ── Summary stats ─────────────────────────────────────────────────────────
    mask_scores  = scores_arr[process_mask_img.get_fdata() > 0]
    above_scores = mask_scores[mask_scores > chance]
    print(f"\n  Mean accuracy  (all)      : {mask_scores.mean()*100:.2f}%")
    print(f"  Peak accuracy             : {mask_scores.max()*100:.2f}%")
    print(f"  Voxels above chance       : {len(above_scores)} / {n_centres}"
          f"  ({100*len(above_scores)/max(1,n_centres):.1f}%)")

    # ── Permutation test (mean-accuracy statistic) ────────────────────────────
    # Voxel-wise FWE permutation (shuffle → refit full searchlight → take max)
    # is the gold-standard but costs n_perms × sl_runtime.  We record the MEAN
    # accuracy across all centre voxels as the test statistic instead — much
    # faster while still providing a valid brain-level p-value.
    #
    # Observed statistic: mean accuracy across all centre voxels.
    # Null distribution : same statistic computed after shuffling the trial labels.
    obs_mean_acc = float(mask_scores.mean())
    p_value_perm  = None
    null_mean_accs = np.array([])

    if args.n_permutations > 0:
        print(f"\n  Permutation test ({args.n_permutations} × full searchlight) ...")
        print(f"  Observed mean accuracy: {obs_mean_acc*100:.3f}%")
        print(f"  ⚠  Each permutation re-runs SearchLight — budget time accordingly.")

        rng           = np.random.default_rng(seed=args.perm_seed)
        null_mean_accs = []
        t_perm        = time.perf_counter()

        for perm_i in range(args.n_permutations):
            y_shuffled = rng.permutation(y)

            sl_perm = SearchLight(
                mask_img         = mask_img,
                process_mask_img = process_mask_img,
                radius           = args.radius,
                estimator        = SVC(C=args.svm_c, kernel='linear',
                                       class_weight='balanced', random_state=42),
                cv               = cv,
                scoring          = 'accuracy',
                n_jobs           = args.n_jobs,
                verbose          = 0,
            )
            if groups is not None:
                sl_perm.fit(imgs_4d, y_shuffled, groups=groups)
            else:
                sl_perm.fit(imgs_4d, y_shuffled)

            null_scores  = sl_perm.scores_[process_mask_img.get_fdata() > 0]
            null_mean_accs.append(float(null_scores.mean()))

            elapsed_perm = time.perf_counter() - t_perm
            rate         = (perm_i + 1) / elapsed_perm
            eta          = (args.n_permutations - perm_i - 1) / rate
            print(f"    perm {perm_i+1:>3}/{args.n_permutations}  "
                  f"null_mean={null_mean_accs[-1]*100:.3f}%  "
                  f"ETA {eta/60:.1f} min")

        null_mean_accs = np.array(null_mean_accs)
        p_value_perm   = float((null_mean_accs >= obs_mean_acc).mean())
        elapsed_perm   = time.perf_counter() - t_perm

        print(f"  Permutation test done in {elapsed_perm/60:.1f} min")
        print(f"  Null mean ± SD    : {null_mean_accs.mean()*100:.3f}% ± {null_mean_accs.std()*100:.3f}%")
        print(f"  p-value (mean acc): {p_value_perm:.4f}  "
              f"({'*significant*' if p_value_perm < 0.05 else 'not significant'} at α=0.05)")

        # Save null distribution
        pd.DataFrame({'null_mean_accuracy': null_mean_accs}).to_csv(
            os.path.join(output_dir, 'permutation_null_distribution.csv'), index=False)

        # Plot null distribution
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(null_mean_accs * 100, bins=max(10, args.n_permutations // 10),
                color='steelblue', edgecolor='k', alpha=0.8, label='Null distribution')
        ax.axvline(obs_mean_acc * 100, color='crimson', lw=2, ls='--',
                   label=f'Observed mean: {obs_mean_acc*100:.3f}%')
        ax.axvline(chance * 100, color='gray', lw=1.2, ls=':',
                   label=f'Chance: {chance*100:.1f}%')
        ax.set_xlabel('Mean searchlight accuracy (%)')
        ax.set_ylabel('Count')
        ax.set_title(
            f'Searchlight permutation test  (n={args.n_permutations},  p={p_value_perm:.4f})\n'
            f'{SUBJECT} — {ANALYSIS_NAME}'
        )
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'permutation_test.png'), dpi=120)
        plt.close()
        print(f"  Saved: permutation_test.png")
    else:
        print("\n  Permutation test skipped (--n-permutations 0)")

    # ── Plots ──────────────────────────────────────────────────────────────────
    func_ref_path = os.path.join(FMRIPREP_ROOT, SUBJECT, "func",
                                  f"{SUBJECT}_task-objectviewing_run-1_boldref.nii.gz")
    func_bg = nib.load(func_ref_path) if os.path.exists(func_ref_path) else None

    fig, ax = plt.subplots(figsize=(14, 4))
    plotting.plot_stat_map(
        scores_img, bg_img=func_bg, display_mode='z', cut_coords=7,
        colorbar=True, title=f'Searchlight accuracy — r={args.radius}mm (chance={chance*100:.1f}%)',
        cmap='hot', threshold=chance, vmin=chance, axes=ax)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'searchlight_accuracy_map.png'), dpi=120)
    plt.close()

    # Above-chance map
    above_data = scores_arr.copy(); above_data[above_data <= chance] = 0.0
    above_img  = image.new_img_like(scores_img, above_data)
    fig, ax    = plt.subplots(figsize=(14, 4))
    plotting.plot_stat_map(
        above_img, bg_img=func_bg, display_mode='z', cut_coords=7,
        colorbar=True, title=f'Above-chance searchlight (>{chance*100:.1f}%)',
        cmap='hot', threshold=chance+0.001, vmin=chance, axes=ax)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'searchlight_above_chance.png'), dpi=120)
    plt.close()

    # ── Peak voxels table ──────────────────────────────────────────────────────
    aff = scores_img.affine
    vox_ijk  = np.column_stack(np.where(process_mask_img.get_fdata() > 0))
    vox_accs = scores_arr[vox_ijk[:, 0], vox_ijk[:, 1], vox_ijk[:, 2]]
    top_n    = min(args.top_n_peaks, len(vox_ijk))
    top_idx  = np.argsort(vox_accs)[::-1][:top_n]
    peaks_rows = []
    for idx in top_idx:
        i, j, k = vox_ijk[idx]
        mni_xyz = nib.affines.apply_affine(aff, [i, j, k])
        peaks_rows.append({'subject': SUBJECT, 'analysis_name': ANALYSIS_NAME,
                           'vox_i': i, 'vox_j': j, 'vox_k': k,
                           'mni_x': round(float(mni_xyz[0]), 1),
                           'mni_y': round(float(mni_xyz[1]), 1),
                           'mni_z': round(float(mni_xyz[2]), 1),
                           'accuracy': round(float(vox_accs[idx]), 4)})
    peaks_df = pd.DataFrame(peaks_rows)
    peaks_df.to_csv(os.path.join(output_dir, 'searchlight_top_voxels.csv'), index=False)

    # All voxel accuracy distribution
    all_rows = []
    for idx in range(len(vox_ijk)):
        i, j, k = vox_ijk[idx]
        mni_xyz = nib.affines.apply_affine(aff, [i, j, k])
        all_rows.append({'subject': SUBJECT, 'analysis_name': ANALYSIS_NAME,
                         'mni_x': round(float(mni_xyz[0]), 1),
                         'mni_y': round(float(mni_xyz[1]), 1),
                         'mni_z': round(float(mni_xyz[2]), 1),
                         'accuracy': round(float(vox_accs[idx]), 4)})
    pd.DataFrame(all_rows).to_csv(
        os.path.join(output_dir, 'searchlight_accuracy_distribution.csv'), index=False)

    # Summary row
    ACC_THRESHOLD = args.acc_threshold if args.acc_threshold else chance + 0.1
    n_above = int((mask_scores > ACC_THRESHOLD).sum())
    summary = dict(subject=SUBJECT, analysis_name=ANALYSIS_NAME,
                   searchlight_radius_mm=args.radius, cv_strategy=cv_name,
                   n_conditions=len(conditions),
                   conditions="|".join(str(c) for c in conditions),
                   n_trials=int(len(y)), chance_level=round(float(chance), 4),
                   n_centre_voxels=n_centres,
                   n_above_chance=int((mask_scores > chance).sum()),
                   n_above_threshold=n_above,
                   mean_acc_all=round(float(mask_scores.mean()), 4),
                   mean_acc_above_chance=(round(float(above_scores.mean()), 4)
                                          if len(above_scores) else None),
                   max_acc=round(float(mask_scores.max()), 4),
                   permutation_p=round(float(p_value_perm), 4) if p_value_perm is not None else None,
                   n_permutations=args.n_permutations,
                   null_mean=round(float(null_mean_accs.mean()), 4) if len(null_mean_accs) else None,
                   null_sd=round(float(null_mean_accs.std()),   4) if len(null_mean_accs) else None,
                   elapsed_s=round(time.perf_counter()-t_start, 1))
    pd.DataFrame([summary]).to_csv(
        os.path.join(output_dir, 'searchlight_summary.csv'), index=False)

    # ── Report ────────────────────────────────────────────────────────────────
    with open(os.path.join(output_dir, 'searchlight_report.json'), 'w') as f:
        json.dump({k: (v if not isinstance(v, np.integer) else int(v))
                   for k, v in summary.items()}, f, indent=2)

    elapsed = time.perf_counter() - t_start
    print(f"\n  Done in {elapsed/60:.1f} min")
    print(f"  Outputs: {output_dir}/")

if __name__ == "__main__":
    main()
