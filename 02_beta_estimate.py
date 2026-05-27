#!/usr/bin/env python3
"""
02_beta_estimate.py — Trial beta estimation: LSA and/or LSS (batch/SLURM version)

Usage:
    python 02_beta_estimate.py --subject sub-1
    python 02_beta_estimate.py --subject sub-1 --method lss --n-jobs 4

Mirrors beta_estimation.ipynb but runs non-interactively.
"""
import argparse, json, os, sys, time, warnings
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from nilearn import image
from nilearn.glm.first_level import (FirstLevelModel, make_first_level_design_matrix)
from nilearn.maskers import NiftiMasker
try:
    import ants
    ANTS_AVAILABLE = True
except ImportError:
    ANTS_AVAILABLE = False

# ── CLI ────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Trial beta estimation (LSA/LSS)")
    p.add_argument("--subject",  required=True)
    p.add_argument("--method",   default="both", choices=["lsa","lss","both"])
    p.add_argument("--strategy", default="hmp24",
                   help="Preprocessing strategy used (for bold filename desc-<strategy>)")
    p.add_argument("--n-jobs",   type=int, default=-1, help="Parallel jobs for LSS")
    p.add_argument("--n-runs",   type=int, default=12)
    p.add_argument("--tr",       type=float, default=2.5)
    p.add_argument("--hrf",      default="spm",
                   choices=["spm","glover","spm + derivative"])
    p.add_argument("--high-pass",type=float, default=None)
    return p.parse_args()

# ── Paths ──────────────────────────────────────────────────────────────────────
FMRIPREP_ROOT = "/pl/active/courses/2026_summer/neuroclass2026/ds000105/derivatives/fmriprep"
BIDS_ROOT     = "/pl/active/courses/2026_summer/neuroclass2026/ds000105"
DERIV_ROOT    = "/pl/active/courses/2026_summer/neuroclass2026/ds000105/derivatives"
_LPS_TO_RAS   = np.diag([-1., -1., 1., 1.])

# ── Helpers ────────────────────────────────────────────────────────────────────
def make_lss_events(events_df, target_idx):
    ev = events_df.copy().reset_index(drop=True)
    target_cond = ev.loc[target_idx, 'trial_type']
    rows = []
    for i, row in ev.iterrows():
        if i == target_idx:
            rows.append({'onset': row['onset'], 'duration': row['duration'],
                         'trial_type': f'target__{target_idx:04d}'})
        elif row['trial_type'] == target_cond:
            rows.append({'onset': row['onset'], 'duration': row['duration'],
                         'trial_type': f'{target_cond}__others'})
        else:
            rows.append({'onset': row['onset'], 'duration': row['duration'],
                         'trial_type': row['trial_type']})
    return pd.DataFrame(rows)

def warp_mask_to_func(group_mask_path, mni_to_t1w_xfm, t1w_to_func_xfm,
                      func_ref_path, subject_id, run_id, out_dir):
    """Warp a group MNI-space mask to functional space using ANTs."""
    if not ANTS_AVAILABLE:
        print('skipping ANTS')
        return image.load_img(func_ref_path)  # fallback: return func ref
    func_ref_ants = ants.image_read(func_ref_path)
    mask_ants     = ants.image_read(group_mask_path)
    warped = ants.apply_transforms(
        fixed         = func_ref_ants,
        moving        = mask_ants,
        transformlist = [t1w_to_func_xfm, mni_to_t1w_xfm],
        whichtoinvert = [False, False],
        interpolator  = "nearestNeighbor",
    )
    spacing   = np.array(warped.spacing)
    direction = np.array(warped.direction).reshape(3, 3)
    origin    = np.array(warped.origin)
    aff_lps         = np.eye(4)
    aff_lps[:3, :3] = direction * spacing
    aff_lps[:3,  3] = origin
    aff_ras = _LPS_TO_RAS @ aff_lps
    nib_mask = nib.Nifti1Image((warped.numpy() > 0.5).astype(np.int16), aff_ras)
    out_path = os.path.join(out_dir, "warped_masks",
                            f"{subject_id}_task-objectviewing_run-{run_id:02d}"
                            "_desc-warpedGroupMask_funcspace.nii.gz")
    print(_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    nib_mask.to_filename(out_path)
    return nib_mask, out_path

def zero_fill_outside_mask(beta_img, bold_ref, mask_img):
    """Set beta voxels outside the brain mask to exactly 0."""
    mask_rs  = image.resample_to_img(mask_img, bold_ref, interpolation='nearest')
    mask_bin = mask_rs.get_fdata() > 0
    data     = beta_img.get_fdata().copy()
    data[~mask_bin] = 0.0
    return nib.Nifti1Image(data, beta_img.affine, beta_img.header)

def run_lsa_run(bold_img, events_df, mask_img, tr, hrf_model, high_pass, run_id):
    """Fit LSA GLM for one run, return dict (trial_idx -> beta img)."""
    n_vols      = bold_img.shape[-1]
    frame_times = np.arange(n_vols) * tr
    ev_lsa      = events_df.copy().reset_index(drop=True)
    ev_lsa['trial_type'] = [f"{r['trial_type']}__trial_{i:04d}"
                            for i, r in ev_lsa.iterrows()]
    dm = make_first_level_design_matrix(
        frame_times=frame_times, events=ev_lsa,
        hrf_model=hrf_model,
        drift_model='polynomial' if high_pass is None else 'cosine',
        high_pass=high_pass)
    glm = FirstLevelModel(t_r=tr, mask_img=mask_img,
                          standardize=False, noise_model='ar1',
                          minimize_memory=False)
    glm.fit(bold_img, design_matrices=dm)
    bold_ref = image.index_img(bold_img, 0)
    out = {}
    for i, row in events_df.reset_index(drop=True).iterrows():
        contrast_id = f"{row['trial_type']}__trial_{i:04d}"
        if contrast_id not in dm.columns:
            continue
        vec = np.zeros(dm.shape[1])
        vec[dm.columns.get_loc(contrast_id)] = 1.0
        img = glm.compute_contrast(vec, output_type='effect_size')
        img = zero_fill_outside_mask(img, bold_ref, mask_img)
        out[i] = {'img': img, 'condition': row['trial_type'],
                  'onset': row['onset'], 'run': run_id}
    return out

def _lss_one_trial(trial_idx, bold_img, events_df, mask_img, tr, hrf_model, high_pass):
    ev = make_lss_events(events_df, trial_idx)
    n_vols      = bold_img.shape[-1]
    frame_times = np.arange(n_vols) * tr
    dm = make_first_level_design_matrix(
        frame_times=frame_times, events=ev,
        hrf_model=hrf_model,
        drift_model='polynomial' if high_pass is None else 'cosine',
        high_pass=high_pass)
    glm = FirstLevelModel(t_r=tr, mask_img=mask_img,
                          standardize=False, noise_model='ar1',
                          minimize_memory=False)
    glm.fit(bold_img, design_matrices=dm)
    target_col = f'target__{trial_idx:04d}'
    if target_col not in dm.columns:
        return None
    vec = np.zeros(dm.shape[1])
    vec[dm.columns.get_loc(target_col)] = 1.0
    img = glm.compute_contrast(vec, output_type='effect_size')
    bold_ref = image.index_img(bold_img, 0)
    img = zero_fill_outside_mask(img, bold_ref, mask_img)
    return img

def run_lss_run(bold_img, events_df, mask_img, tr, hrf_model, high_pass, run_id, n_jobs):
    """Fit one LSS GLM per trial in parallel, return dict."""
    n_trials = len(events_df)
    results  = Parallel(n_jobs=n_jobs, prefer='threads')(
        delayed(_lss_one_trial)(i, bold_img, events_df, mask_img, tr, hrf_model, high_pass)
        for i in range(n_trials)
    )
    out = {}
    for i, (res, row) in enumerate(zip(results, events_df.reset_index(drop=True).itertuples())):
        if res is not None:
            out[i] = {'img': res, 'condition': row.trial_type,
                      'onset': row.onset, 'run': run_id}
    return out

def save_betas(betas_dict, method_name, output_dir, subject_id):
    """Save 3D beta NIfTIs and return manifest DataFrame."""
    beta_dir = Path(output_dir) / method_name
    beta_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for (run_id, trial_idx), info in sorted(betas_dict.items()):
        fname = (f"{subject_id}_run-{run_id:02d}_trial-{trial_idx:04d}"
                 f"_condition-{info['condition']}_{method_name}_beta.nii.gz")
        fpath = str(beta_dir / fname)
        info['img'].to_filename(fpath)
        rows.append(dict(subject=subject_id, run=run_id,
                         trial_idx=trial_idx, condition=info['condition'],
                         onset_s=round(info['onset'], 3),
                         beta_file=fpath, method=method_name))
    df = pd.DataFrame(rows)
    csv_path = str(Path(output_dir) / f'beta_manifest_{method_name}.csv')
    df.to_csv(csv_path, index=False)
    print(f"  [{method_name.upper()}] {len(df)} betas → {csv_path}")
    return df

# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    args    = parse_args()
    SUBJECT = args.subject
    TR      = args.tr
    t_start = time.perf_counter()

    output_dir = os.path.join(DERIV_ROOT, "beta_weights", SUBJECT)
    os.makedirs(output_dir, exist_ok=True)

    print(f"{'='*60}")
    print(f"  BETA ESTIMATION: {SUBJECT}  method={args.method}")
    print(f"{'='*60}")

    # ── Load per-run data ────────────────────────────────────────────────────
    bold_imgs, mask_imgs, events_list, run_ids = [], [], [], []
    for run in range(1, args.n_runs + 1):
        bold_path = (f"{DERIV_ROOT}/preprocessed/{SUBJECT}/"
                     f"{SUBJECT}_task-objectviewing_run-{run}_desc-{args.strategy}_bold.nii.gz")
        if not os.path.exists(bold_path):
            # Fallback to fmriprep output
            bold_path = (f"{FMRIPREP_ROOT}/{SUBJECT}/func/"
                         f"{SUBJECT}_task-objectviewing_run-{run}_desc-preproc_bold.nii.gz")
        if not os.path.exists(bold_path):
            print(f"  Run {run:02d}: BOLD not found — skipping"); continue

        mask_path = (f"{FMRIPREP_ROOT}/{SUBJECT}/func/"
                     f"{SUBJECT}_task-objectviewing_run-{run}_desc-brain_mask.nii.gz")
        ev_path = (f"{BIDS_ROOT}/{SUBJECT}/func/"
                   f"{SUBJECT}_task-objectviewing_run-{run:02d}_events.tsv")
        if not os.path.exists(ev_path):
            ev_path = ev_path.replace(f"run-{run:02d}", f"run-{run}")
        if not os.path.exists(ev_path):
            print(f"  Run {run:02d}: events not found — skipping"); continue

        bold_imgs.append(image.load_img(bold_path))
        mask_imgs.append(image.load_img(mask_path))
        events_list.append(pd.read_csv(ev_path, sep='\t'))
        run_ids.append(run)

    if not bold_imgs:
        print("ERROR: no valid runs found."); sys.exit(1)

    n_runs     = len(bold_imgs)
    conditions = sorted(set(t for ev in events_list for t in ev['trial_type']))
    total_t    = sum(len(ev) for ev in events_list)
    print(f"\n  Runs loaded : {run_ids}")
    print(f"  Conditions  : {conditions}")
    print(f"  Total trials: {total_t}")

    # ── Warp group mask to func space ────────────────────────────────────────
    group_mask_path = (f"{FMRIPREP_ROOT}/group_mask/"
                       "tpl-MNI152NLin2009cAsym_res-02_desc-brain_mask.nii.gz")
    warped_mask_imgs = {}
    for ri, run in enumerate(run_ids):
        func_ref = (f"{FMRIPREP_ROOT}/{SUBJECT}/func/"
                    f"{SUBJECT}_task-objectviewing_run-{run}_boldref.nii.gz")
        if ANTS_AVAILABLE and os.path.exists(group_mask_path) and os.path.exists(func_ref):
            mni_to_t1w = (f"{FMRIPREP_ROOT}/{SUBJECT}/anat/"
                          f"{SUBJECT}_from-MNI152NLin2009cAsym_to-T1w_mode-image_xfm.h5")
            t1w_to_func = (f"{FMRIPREP_ROOT}/{SUBJECT}/func/"
                           f"{SUBJECT}_task-objectviewing_run-{run}"
                           "_from-T1w_to-scanner_mode-image_xfm.txt")
            if os.path.exists(mni_to_t1w) and os.path.exists(t1w_to_func):
                result = warp_mask_to_func(group_mask_path, mni_to_t1w, t1w_to_func,
                                           func_ref, SUBJECT, run, output_dir)
                warped_mask_imgs[run] = result[0] if isinstance(result, tuple) else result
                continue
        # Fallback: use per-run brain mask
        warped_mask_imgs[run] = mask_imgs[ri]

    # ── LSA ──────────────────────────────────────────────────────────────────
    lsa_betas = {}
    if args.method in ("lsa", "both"):
        print("\n  Running LSA...")
        for ri, run in enumerate(run_ids):
            print(f"    Run {run:02d}...", end=' ', flush=True)
            t0 = time.perf_counter()
            run_result = run_lsa_run(
                bold_imgs[ri], events_list[ri], warped_mask_imgs[run],
                TR, args.hrf, args.high_pass, run)
            for k, v in run_result.items():
                lsa_betas[(run, k)] = v
            print(f"{len(run_result)} betas  {time.perf_counter()-t0:.0f}s")

    # ── LSS ──────────────────────────────────────────────────────────────────
    lss_betas = {}
    if args.method in ("lss", "both"):
        print("\n  Running LSS...")
        for ri, run in enumerate(run_ids):
            print(f"    Run {run:02d}  ({len(events_list[ri])} trials)...", end=' ', flush=True)
            t0 = time.perf_counter()
            run_result = run_lss_run(
                bold_imgs[ri], events_list[ri], warped_mask_imgs[run],
                TR, args.hrf, args.high_pass, run, args.n_jobs)
            for k, v in run_result.items():
                lss_betas[(run, k)] = v
            print(f"{len(run_result)} betas  {time.perf_counter()-t0:.1f}s")

    # ── Save ─────────────────────────────────────────────────────────────────
    print("\n  Saving beta images...")
    manifests = {}
    if lsa_betas:
        manifests["lsa"] = save_betas(lsa_betas, "lsa", output_dir, SUBJECT)
    if lss_betas:
        manifests["lss"] = save_betas(lss_betas, "lss", output_dir, SUBJECT)

    # ── Report ───────────────────────────────────────────────────────────────
    elapsed = time.perf_counter() - t_start
    report  = dict(subject=SUBJECT, method=args.method, conditions=conditions,
                   n_runs=n_runs, total_trials_lsa=len(lsa_betas),
                   total_trials_lss=len(lss_betas), tr=TR,
                   hrf_model=args.hrf, elapsed_s=round(elapsed, 1),
                   manifests={k: str(Path(output_dir)/f'beta_manifest_{k}.csv')
                              for k in manifests})
    with open(os.path.join(output_dir, 'beta_estimation_report.json'), 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\n  Done in {elapsed/60:.1f} min")
    print(f"  Outputs: {output_dir}/")

if __name__ == "__main__":
    main()
