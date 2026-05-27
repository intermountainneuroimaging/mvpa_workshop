#!/usr/bin/env python3
"""
01_preprocess.py — fMRI nuisance regression pipeline (batch/SLURM version)

Usage:
    python 01_preprocess.py --subject sub-1
    python 01_preprocess.py --subject sub-1 --strategy hmp24 --poly-order 2

Mirrors fmri_preprocessing.ipynb but runs non-interactively.
All plots are saved to <output_dir>/qc/.
"""
import argparse, json, os, time, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from nilearn import image, signal
from nilearn.maskers import NiftiMasker

# ── CLI ────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="fMRI preprocessing — nuisance regression")
    p.add_argument("--subject",    required=True,            help="e.g. sub-1")
    p.add_argument("--strategy",   default="hmp24",
                   choices=["hmp6","hmp24","compcor5","aroma","custom"],
                   help="Confound strategy (default: hmp24)")
    p.add_argument("--poly-order", type=int, default=2,      help="Detrending polynomial order")
    p.add_argument("--fd-thresh",  type=float, default=0.5,  help="FD threshold for scrubbing (mm)")
    p.add_argument("--no-scrub",   action="store_true",      help="Skip volume scrubbing")
    p.add_argument("--n-runs",     type=int, default=12,     help="Number of runs")
    p.add_argument("--tr",         type=float, default=2.5,  help="TR in seconds")
    p.add_argument("--dummy-scans",type=int, default=0,      help="Dummy scans to drop")
    return p.parse_args()

# ── Paths ──────────────────────────────────────────────────────────────────────
FMRIPREP_ROOT = "/pl/active/courses/2026_summer/neuroclass2026/ds000105/derivatives/fmriprep"
BIDS_ROOT     = "/pl/active/courses/2026_summer/neuroclass2026/ds000105"
DERIV_ROOT    = "/pl/active/courses/2026_summer/neuroclass2026/ds000105/derivatives"

# ── Confound column sets ───────────────────────────────────────────────────────
HMP6      = ['trans_x','trans_y','trans_z','rot_x','rot_y','rot_z']
HMP_DERIV = [f'{p}_derivative1'       for p in HMP6]
HMP_SQ    = [f'{p}_power2'            for p in HMP6]
HMP_DSQ   = [f'{p}_derivative1_power2' for p in HMP6]
HMP24     = HMP6 + HMP_DERIV + HMP_SQ + HMP_DSQ

STRATEGY_COLS = {
    "hmp6":    HMP6,
    "hmp24":   HMP24,
    "compcor5": HMP6 + [f'a_comp_cor_0{i}' for i in range(5)],
    "aroma":   [],   # ICA-AROMA handled via non-aggressive denoising
}

def load_confounds(tsv_path, strategy, fd_threshold=0.5):
    df = pd.read_csv(tsv_path, sep='\t')
    cols = [c for c in STRATEGY_COLS.get(strategy, HMP24) if c in df.columns]
    # Spike regressors for high-motion volumes
    fd = df.get('framewise_displacement', pd.Series(dtype=float)).fillna(0).values
    spikes = np.where(fd > fd_threshold)[0]
    if len(spikes):
        spike_dict = {}
        for t in spikes:
            sc = np.zeros(len(df)); sc[t] = 1.0
            spike_dict[f'spike_{t:04d}'] = sc
            cols.append(f'spike_{t:04d}')
        df = pd.concat([df, pd.DataFrame(spike_dict, index=df.index)], axis=1)
    conf_arr = df[cols].fillna(0).values
    return conf_arr, cols, fd

def compute_fd(hmp_array):
    R_MM = 50.0
    delta = np.diff(hmp_array, axis=0)
    delta[:, 3:] *= R_MM
    fd = np.abs(delta).sum(axis=1)
    return np.concatenate([[0.0], fd])

def clean_bold(bold_img, mask_img, confounds, tr, poly_order=2):
    masker = NiftiMasker(mask_img=mask_img, standardize=None,
                         detrend=False, t_r=tr)
    bold_2d = masker.fit_transform(bold_img)
    # Compute mean TSNR before cleaning
    tsnr_before = (bold_2d.mean(0) / (bold_2d.std(0) + 1e-8)).mean()
    # Clean: polynomial detrend + confound regression in one OLS step
    poly = np.vstack([np.arange(bold_2d.shape[0])**i
                      for i in range(poly_order + 1)]).T
    design = np.hstack([confounds, poly])
    design = (design - design.mean(0)) / (design.std(0) + 1e-10)
    clean_2d = signal.clean(bold_2d, detrend=False, standardize=None,
                            confounds=design, t_r=tr)
    tsnr_after = (clean_2d.mean(0) / (clean_2d.std(0) + 1e-8)).mean()
    return clean_2d, masker, tsnr_before, tsnr_after

def save_qc_plots(subject, n_runs, fd_per_run, tsnr_before, tsnr_after, qc_dir):
    os.makedirs(qc_dir, exist_ok=True)
    # FD plot
    fig, axes = plt.subplots(n_runs, 1, figsize=(12, 2*n_runs), sharex=False)
    for ri in range(n_runs):
        axes[ri].plot(fd_per_run[ri], lw=0.8, color='steelblue')
        axes[ri].axhline(0.5, color='red', lw=0.8, ls='--')
        axes[ri].set_ylabel(f'Run {ri+1}', fontsize=7)
    axes[-1].set_xlabel('Volume'); fig.suptitle(f'{subject} — Framewise Displacement', fontsize=10)
    plt.tight_layout(); fig.savefig(os.path.join(qc_dir, 'motion_fd.png'), dpi=100); plt.close()
    # TSNR bar chart
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(n_runs); w = 0.35
    ax.bar(x - w/2, tsnr_before, w, label='Before', color='steelblue', alpha=0.8)
    ax.bar(x + w/2, tsnr_after,  w, label='After',  color='seagreen',  alpha=0.8)
    ax.set_xticks(x); ax.set_xticklabels([f'R{i+1}' for i in range(n_runs)], fontsize=8)
    ax.set_xlabel('Run'); ax.set_ylabel('Mean TSNR'); ax.legend()
    ax.set_title(f'{subject} — TSNR before/after nuisance regression')
    plt.tight_layout(); fig.savefig(os.path.join(qc_dir, 'tsnr_comparison.png'), dpi=100); plt.close()

# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    SUBJECT  = args.subject
    STRATEGY = args.strategy
    TR       = args.tr
    N_RUNS   = args.n_runs
    t_start  = time.perf_counter()

    output_dir = os.path.join(DERIV_ROOT, "preprocessed", SUBJECT)
    qc_dir     = os.path.join(output_dir, "qc")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(qc_dir, exist_ok=True)

    print(f"{'='*60}")
    print(f"  PREPROCESSING: {SUBJECT}  strategy={STRATEGY}")
    print(f"{'='*60}")

    saved_files  = {}
    fd_per_run   = []
    tsnr_before  = []
    tsnr_after   = []

    for run in range(1, N_RUNS + 1):
        bold_path  = (f"{FMRIPREP_ROOT}/{SUBJECT}/func/"
                      f"{SUBJECT}_task-objectviewing_run-{run}_desc-preproc_bold.nii.gz")
        mask_path  = (f"{FMRIPREP_ROOT}/{SUBJECT}/func/"
                      f"{SUBJECT}_task-objectviewing_run-{run}_desc-brain_mask.nii.gz")
        conf_path  = (f"{FMRIPREP_ROOT}/{SUBJECT}/func/"
                      f"{SUBJECT}_task-objectviewing_run-{run}_desc-confounds_timeseries.tsv")

        if not os.path.exists(bold_path):
            print(f"  Run {run:02d}: BOLD not found — skipping"); continue

        print(f"  Run {run:02d}: loading ...", end=' ', flush=True)
        bold_img = image.load_img(bold_path)
        mask_img = image.load_img(mask_path)

        if args.dummy_scans:
            bold_img = image.index_img(bold_img, slice(args.dummy_scans, None))

        confounds, conf_cols, fd = load_confounds(conf_path, STRATEGY, args.fd_thresh)
        fd_per_run.append(fd)

        clean_2d, masker, tsb, tsa = clean_bold(
            bold_img, mask_img, confounds, TR, poly_order=args.poly_order)
        tsnr_before.append(tsb); tsnr_after.append(tsa)

        # Reconstruct 4D NIfTI
        clean_img = masker.inverse_transform(clean_2d)

        # Output path — replace desc-preproc with desc-<strategy>
        out_name = (f"{SUBJECT}_task-objectviewing_run-{run}"
                    f"_desc-{STRATEGY}_bold.nii.gz")
        out_path = os.path.join(output_dir, out_name)
        clean_img.to_filename(out_path)
        saved_files[run] = out_path
        print(f"TSNR {tsb:.1f} → {tsa:.1f}  saved.")

    # QC plots
    if fd_per_run:
        save_qc_plots(SUBJECT, len(fd_per_run), fd_per_run,
                      tsnr_before, tsnr_after, qc_dir)
        print(f"\n  QC plots → {qc_dir}/")

    # Report JSON
    report = dict(subject=SUBJECT, strategy=STRATEGY, poly_order=args.poly_order,
                  tr=TR, n_runs_processed=len(saved_files),
                  mean_tsnr_before=round(float(np.mean(tsnr_before)), 2),
                  mean_tsnr_after=round(float(np.mean(tsnr_after)), 2),
                  output_files=list(saved_files.values()),
                  elapsed_s=round(time.perf_counter()-t_start, 1))
    with open(os.path.join(output_dir, 'preprocessing_report.json'), 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\n  Done in {(time.perf_counter()-t_start)/60:.1f} min")
    print(f"  Outputs: {output_dir}/")

if __name__ == "__main__":
    main()
