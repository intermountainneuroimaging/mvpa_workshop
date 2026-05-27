"""
fmri_helpers.py
===============
Shared helper functions for the neuroclass2026 fMRI MVPA tutorial series.

Functions are grouped into four modules matching the tutorial notebooks:
  Module 1 — Preprocessing
  Module 2 — Beta estimation (GLM, LSA, LSS)
  Module 3 — MVPA decoding (SVM, cross-validation, feature selection)
  Module 4 — Searchlight analysis

All functions are documented with NumPy-style docstrings and example usage.

Usage:
  from fmri_helpers import (load_bold_and_mask, load_confounds,
                             clean_bold_run, compute_fd, ...)
"""

from __future__ import annotations
import os, warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import time

import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
from nilearn import image, signal
from nilearn.maskers import NiftiMasker

# ══════════════════════════════════════════════════════════════════════════════
# MODULE 1 — PREPROCESSING
# ══════════════════════════════════════════════════════════════════════════════

# ── Standard confound column sets (fMRIPrep >= 21) ───────────────────────────
HMP6 = ['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z']
HMP_DERIV    = [f'{p}_derivative1'        for p in HMP6]
HMP_SQ       = [f'{p}_power2'             for p in HMP6]
HMP_DERIV_SQ = [f'{p}_derivative1_power2' for p in HMP6]
HMP24        = HMP6 + HMP_DERIV + HMP_SQ + HMP_DERIV_SQ

COMPCOR5     = HMP6 + [f'a_comp_cor_0{i}' for i in range(5)]

CONFOUND_STRATEGIES = {
    'hmp6':    HMP6,
    'hmp24':   HMP24,
    'compcor5': COMPCOR5,
}


def load_bold_and_mask(bold_path: str,
                       mask_path: str,
                       n_dummy_scans: int = 0
                       ) -> Tuple[nib.Nifti1Image, nib.Nifti1Image]:
    """Load a 4-D BOLD NIfTI and its brain mask, optionally dropping dummy scans.

    Parameters
    ----------
    bold_path : str
        Path to the 4-D BOLD NIfTI (.nii or .nii.gz).
    mask_path : str
        Path to the binary brain mask NIfTI.
    n_dummy_scans : int, optional
        Number of initial volumes to discard (default 0).

    Returns
    -------
    bold_img : nib.Nifti1Image
        4-D BOLD image (shape: x, y, z, T).
    mask_img : nib.Nifti1Image
        3-D binary brain mask.

    Example
    -------
    >>> bold, mask = load_bold_and_mask("sub-1_bold.nii.gz", "sub-1_mask.nii.gz")
    >>> print(bold.shape)   # (64, 64, 35, 184)
    """
    bold_img = image.load_img(bold_path)
    mask_img = image.load_img(mask_path)
    if n_dummy_scans > 0:
        bold_img = image.index_img(bold_img, slice(n_dummy_scans, None))
        print(f"  Dropped {n_dummy_scans} dummy scans → {bold_img.shape[-1]} volumes remain.")
    return bold_img, mask_img


def load_confounds(confound_tsv: str,
                   strategy: str = 'hmp24',
                   fd_threshold: float = 0.5,
                   add_spike_regressors: bool = True
                   ) -> Tuple[np.ndarray, List[str], np.ndarray]:
    """Load confound regressors from an fMRIPrep confounds TSV file.

    Parameters
    ----------
    confound_tsv : str
        Path to the *_desc-confounds_timeseries.tsv file.
    strategy : str, optional
        Column set to select. One of 'hmp6', 'hmp24', 'compcor5' (default 'hmp24').
    fd_threshold : float, optional
        FD (mm) above which volumes receive spike regressors (default 0.5).
    add_spike_regressors : bool, optional
        Whether to add binary spike regressors for flagged volumes (default True).

    Returns
    -------
    conf_array : np.ndarray
        Confound matrix, shape (n_volumes, n_regressors). NaNs replaced with 0.
    conf_cols : list of str
        Column names used.
    fd : np.ndarray
        Framewise displacement trace, shape (n_volumes,).  First value is 0.

    Example
    -------
    >>> conf, cols, fd = load_confounds("sub-1_run-1_confounds.tsv", strategy='hmp24')
    >>> print(f"{len(cols)} confound columns loaded")
    """
    df   = pd.read_csv(confound_tsv, sep='\t')
    cols = [c for c in CONFOUND_STRATEGIES.get(strategy, HMP24) if c in df.columns]
    missing = set(CONFOUND_STRATEGIES.get(strategy, HMP24)) - set(cols)
    if missing:
        warnings.warn(f"Confound columns not found in TSV: {missing}")

    fd_col = df.get('framewise_displacement', pd.Series(dtype=float)).fillna(0).values
    fd     = np.concatenate([[0.0], np.abs(np.diff(fd_col))]) if 'framewise_displacement' not in df else fd_col.copy()
    if 'framewise_displacement' in df:
        fd = df['framewise_displacement'].fillna(0).values

    if add_spike_regressors:
        spike_idx = np.where(fd > fd_threshold)[0]
        for t in spike_idx:
            sc = np.zeros(len(df)); sc[t] = 1.0
            df[f'spike_{t:04d}'] = sc
            cols.append(f'spike_{t:04d}')

    conf_array = df[cols].fillna(0).values
    return conf_array, cols, fd


def compute_fd(hmp_array: np.ndarray, brain_radius_mm: float = 50.0) -> np.ndarray:
    """Compute framewise displacement from head motion parameters.

    FD_t = |Δtx| + |Δty| + |Δtz| + r·(|Δrx| + |Δry| + |Δrz|)
    where r = brain_radius_mm, and rotation columns are assumed to be in radians.

    Parameters
    ----------
    hmp_array : np.ndarray
        Shape (n_volumes, 6). Columns: tx, ty, tz, rx, ry, rz.
    brain_radius_mm : float
        Assumed brain radius for rotation→mm conversion (default 50).

    Returns
    -------
    fd : np.ndarray
        Shape (n_volumes,). First value is always 0.

    Example
    -------
    >>> fd = compute_fd(conf_array[:, :6])
    >>> print(f"Mean FD: {fd.mean():.3f} mm,  Max FD: {fd.max():.3f} mm")
    """
    delta = np.diff(hmp_array, axis=0)
    delta[:, 3:] *= brain_radius_mm          # radians → mm
    fd = np.abs(delta).sum(axis=1)
    return np.concatenate([[0.0], fd])


def clean_bold_run(bold_img: nib.Nifti1Image,
                   mask_img: nib.Nifti1Image,
                   confound_array: np.ndarray,
                   tr: float,
                   poly_order: int = 2,
                   standardize=None
                   ) -> Tuple[np.ndarray, NiftiMasker, float, float]:
    """Apply polynomial detrending + confound regression to one BOLD run.

    Uses nilearn.signal.clean() to perform detrending and confound regression
    in a single OLS step, avoiding double-dipping artefacts.

    Parameters
    ----------
    bold_img : nib.Nifti1Image
        4-D BOLD image.
    mask_img : nib.Nifti1Image
        Brain mask.
    confound_array : np.ndarray
        Confound matrix, shape (n_volumes, n_regressors).
    tr : float
        Repetition time in seconds.
    poly_order : int, optional
        Polynomial order for detrending trend model (default 2).
    standardize : optional
        Passed to nilearn.signal.clean. Use None (default, no standardisation),
        'zscore_sample', or 'psc'. Avoid True/False (deprecated in nilearn 0.15).

    Returns
    -------
    clean_2d : np.ndarray
        Cleaned data, shape (n_volumes, n_voxels).
    masker : NiftiMasker
        Fitted masker (use masker.inverse_transform to rebuild NIfTI).
    tsnr_before : float
        Mean TSNR across voxels before cleaning.
    tsnr_after : float
        Mean TSNR across voxels after cleaning.

    Example
    -------
    >>> clean_2d, masker, tsb, tsa = clean_bold_run(bold, mask, confounds, tr=2.5)
    >>> print(f"TSNR improved from {tsb:.1f} to {tsa:.1f}")
    """
    masker  = NiftiMasker(mask_img=mask_img, standardize=None, detrend=False, t_r=tr)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*mask was given at masker creation.*")
        bold_2d = masker.fit_transform(bold_img)

    tsnr_before = _mean_tsnr(bold_2d)

    # Build polynomial drift regressors
    n_vols = bold_2d.shape[0]
    t      = np.arange(n_vols, dtype=float) / n_vols   # normalised to [0,1]
    poly   = np.vstack([t ** i for i in range(poly_order + 1)]).T  # (n_vols, order+1)

    # Concatenate confounds + polynomial drift
    design = np.hstack([confound_array, poly])
    # Standardise each column to avoid numerical issues
    col_std = design.std(axis=0)
    col_std[col_std == 0] = 1.0
    design  = (design - design.mean(axis=0)) / col_std

    clean_2d = signal.clean(bold_2d, detrend=False, standardize=standardize,
                             confounds=design, t_r=tr)
    tsnr_after = _mean_tsnr(clean_2d)
    return clean_2d, masker, tsnr_before, tsnr_after


def _mean_tsnr(data_2d: np.ndarray) -> float:
    """Compute mean TSNR (mean/std) across voxels."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float((data_2d.mean(0) / (data_2d.std(0) + 1e-8)).mean())


def plot_motion_qc(fd_per_run: List[np.ndarray],
                   tsnr_before: List[float],
                   tsnr_after: List[float],
                   subject_id: str = "",
                   fd_threshold: float = 0.5) -> plt.Figure:
    """Plot framewise displacement and TSNR quality-control summary.

    Parameters
    ----------
    fd_per_run : list of np.ndarray
        One FD trace per run.
    tsnr_before, tsnr_after : list of float
        Mean TSNR before/after cleaning per run.
    subject_id : str, optional
        Title prefix.
    fd_threshold : float
        Threshold line drawn on FD plots.

    Returns
    -------
    fig : plt.Figure

    Example
    -------
    >>> fig = plot_motion_qc(fd_per_run, tsnr_before, tsnr_after, subject_id='sub-1')
    >>> fig.savefig("motion_qc.png", dpi=150)
    """
    n_runs = len(fd_per_run)
    fig    = plt.figure(figsize=(16, 4 + 2*n_runs))
    gs     = fig.add_gridspec(n_runs + 1, 2, hspace=0.4, wspace=0.35)

    for ri in range(n_runs):
        ax = fig.add_subplot(gs[ri, 0])
        ax.plot(fd_per_run[ri], lw=0.9, color='steelblue')
        ax.axhline(fd_threshold, color='red', lw=0.8, ls='--',
                   label=f'FD={fd_threshold}' if ri == 0 else None)
        ax.set_ylabel(f'Run {ri+1}', fontsize=8)
        ax.set_ylim(bottom=0)
        if ri == 0:
            ax.set_title('Framewise displacement (mm)')
            ax.legend(fontsize=7)
    fig.axes[-1].set_xlabel('Volume')

    ax_tsnr = fig.add_subplot(gs[:, 1])
    x = np.arange(n_runs); w = 0.35
    ax_tsnr.bar(x - w/2, tsnr_before, w, label='Before', color='steelblue', alpha=0.8)
    ax_tsnr.bar(x + w/2, tsnr_after,  w, label='After',  color='seagreen',  alpha=0.8)
    ax_tsnr.set_xticks(x)
    ax_tsnr.set_xticklabels([f'R{i+1}' for i in range(n_runs)], fontsize=8)
    ax_tsnr.set_xlabel('Run'); ax_tsnr.set_ylabel('Mean TSNR')
    ax_tsnr.set_title('TSNR before / after cleaning')
    ax_tsnr.legend()

    fig.suptitle(f'{subject_id} — Motion QC', fontsize=11, y=1.01)
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 2 — BETA ESTIMATION
# ══════════════════════════════════════════════════════════════════════════════

from nilearn.glm.first_level import FirstLevelModel, make_first_level_design_matrix


def make_lsa_events(events_df: pd.DataFrame) -> pd.DataFrame:
    """Relabel events for Least Squares All (LSA): one unique regressor per trial.

    Each trial gets a unique trial_type label of the form
    ``<condition>__trial_<index>``.  When used in FirstLevelModel, each trial
    has its own HRF-convolved regressor.

    Parameters
    ----------
    events_df : pd.DataFrame
        Must have columns: onset, duration, trial_type.

    Returns
    -------
    ev_lsa : pd.DataFrame
        Same rows as events_df with relabelled trial_type column.

    Example
    -------
    >>> ev_lsa = make_lsa_events(events_df)
    >>> dm_lsa = make_first_level_design_matrix(frame_times, events=ev_lsa, ...)
    """
    ev = events_df.copy().reset_index(drop=True)
    ev['trial_type'] = [f"{row['trial_type']}__trial_{i:04d}"
                        for i, row in ev.iterrows()]
    return ev


def make_lss_events(events_df: pd.DataFrame, target_idx: int) -> pd.DataFrame:
    """Build the LSS design events for a single target trial (Mumford 2012).

    Three regressor types are created:
    - ``target__<idx>``          : the trial being estimated (its own regressor)
    - ``<condition>__others``    : all other trials of the same condition merged
    - ``<other_condition>``      : one regressor per other condition

    This structure greatly reduces collinearity compared to LSA when ITIs are short.

    Parameters
    ----------
    events_df : pd.DataFrame
        Must have columns: onset, duration, trial_type.
    target_idx : int
        Row index (0-based) of the trial to estimate.

    Returns
    -------
    ev_lss : pd.DataFrame
        Relabelled events for the LSS GLM of this trial.

    Example
    -------
    >>> ev_lss = make_lss_events(events_df, target_idx=5)
    >>> dm_lss = make_first_level_design_matrix(frame_times, events=ev_lss, ...)
    """
    ev          = events_df.copy().reset_index(drop=True)
    target_cond = ev.loc[target_idx, 'trial_type']
    rows = []
    for i, row in ev.iterrows():
        if i == target_idx:
            label = f'target__{target_idx:04d}'
        elif row['trial_type'] == target_cond:
            label = f'{target_cond}__others'
        else:
            label = row['trial_type']
        rows.append({'onset': row['onset'], 'duration': row['duration'],
                     'trial_type': label})
    return pd.DataFrame(rows)


def fit_lsa_run(bold_img: nib.Nifti1Image,
                events_df: pd.DataFrame,
                mask_img: nib.Nifti1Image,
                tr: float,
                hrf_model: str = 'spm',
                high_pass: Optional[float] = None,
                run_id: int = 1
                ) -> Dict[int, dict]:
    """Fit an LSA GLM to one BOLD run and return trial beta images.

    Parameters
    ----------
    bold_img : nib.Nifti1Image
        4-D BOLD run.
    events_df : pd.DataFrame
        Trial onset/duration/trial_type for this run.
    mask_img : nib.Nifti1Image
        Brain mask applied during GLM.
    tr : float
        Repetition time (s).
    hrf_model : str
        HRF used for convolution (default 'spm').
    high_pass : float or None
        High-pass cutoff (Hz). None → polynomial drift (default).
    run_id : int
        Run identifier stored in returned dict.

    Returns
    -------
    betas : dict
        Keys: trial_idx (int).
        Values: dict with keys 'img' (Nifti1Image), 'condition' (str),
                'onset' (float), 'run' (int).

    Example
    -------
    >>> betas = fit_lsa_run(bold, events, mask, tr=2.5)
    >>> img = betas[0]['img']   # NIfTI of the first trial's beta map
    """
    ev_lsa      = make_lsa_events(events_df)
    n_vols      = bold_img.shape[-1]
    frame_times = np.arange(n_vols) * tr
    drift       = 'polynomial' if high_pass is None else 'cosine'

    dm = make_first_level_design_matrix(
        frame_times=frame_times, events=ev_lsa,
        hrf_model=hrf_model, drift_model=drift, high_pass=high_pass)

    glm = FirstLevelModel(t_r=tr, mask_img=mask_img, standardize=False,
                          noise_model='ar1', minimize_memory=False)
    glm.fit(bold_img, design_matrices=dm)

    bold_ref = image.index_img(bold_img, 0)
    betas    = {}
    for i, row in events_df.reset_index(drop=True).iterrows():
        col = f"{row['trial_type']}__trial_{i:04d}"
        if col not in dm.columns:
            continue
        vec = np.zeros(dm.shape[1])
        vec[dm.columns.get_loc(col)] = 1.0
        img = glm.compute_contrast(vec, output_type='effect_size')
        img = _zero_fill_outside_mask(img, bold_ref, mask_img)
        betas[i] = {'img': img, 'condition': row['trial_type'],
                    'onset': row['onset'], 'run': run_id}
    return betas


def fit_lss_run(bold_img: nib.Nifti1Image,
                events_df: pd.DataFrame,
                mask_img: nib.Nifti1Image,
                tr: float,
                hrf_model: str = 'spm',
                high_pass: Optional[float] = None,
                run_id: int = 1,
                n_jobs: int = -1
                ) -> Dict[int, dict]:
    """Fit one LSS GLM per trial (Mumford 2012) in parallel; return beta images.

    Parameters
    ----------
    (same as fit_lsa_run, plus)
    n_jobs : int
        Parallel workers for joblib. -1 = all CPUs (default).

    Returns
    -------
    betas : dict
        Same structure as fit_lsa_run output.

    Example
    -------
    >>> betas = fit_lss_run(bold, events, mask, tr=2.5, n_jobs=4)
    """
    from joblib import Parallel, delayed

    def _one_trial(ti):
        ev  = make_lss_events(events_df, ti)
        n   = bold_img.shape[-1]
        ft  = np.arange(n) * tr
        dr  = 'polynomial' if high_pass is None else 'cosine'
        dm  = make_first_level_design_matrix(
            frame_times=ft, events=ev,
            hrf_model=hrf_model, drift_model=dr, high_pass=high_pass)
        glm = FirstLevelModel(t_r=tr, mask_img=mask_img, standardize=False,
                              noise_model='ar1', minimize_memory=False)
        glm.fit(bold_img, design_matrices=dm)
        col = f'target__{ti:04d}'
        if col not in dm.columns:
            return None
        vec = np.zeros(dm.shape[1]); vec[dm.columns.get_loc(col)] = 1.0
        img = glm.compute_contrast(vec, output_type='effect_size')
        bold_ref = image.index_img(bold_img, 0)
        return _zero_fill_outside_mask(img, bold_ref, mask_img)

    n_trials = len(events_df)
    results  = Parallel(n_jobs=n_jobs, prefer='threads')(
        delayed(_one_trial)(i) for i in range(n_trials))

    betas = {}
    for i, (res, (_, row)) in enumerate(zip(results, events_df.reset_index(drop=True).iterrows())):
        if res is not None:
            betas[i] = {'img': res, 'condition': row['trial_type'],
                        'onset': row['onset'], 'run': run_id}
    return betas


def _zero_fill_outside_mask(beta_img: nib.Nifti1Image,
                             bold_ref: nib.Nifti1Image,
                             mask_img: nib.Nifti1Image) -> nib.Nifti1Image:
    """Set voxels outside the brain mask to exactly 0.0 in a beta image.

    Resamples the mask to the beta image's voxel grid using nearest-neighbour
    interpolation before applying the fill, ensuring no border artefacts.
    """
    mask_rs  = image.resample_to_img(mask_img, bold_ref, interpolation='nearest')
    mask_bin = mask_rs.get_fdata() > 0
    data     = beta_img.get_fdata().copy()
    data[~mask_bin] = 0.0
    return nib.Nifti1Image(data, beta_img.affine, beta_img.header)


def save_beta_manifest(betas_dict: Dict[Tuple[int, int], dict],
                       method_name: str,
                       output_dir: str,
                       subject_id: str,
                       save_4d: bool = False
                       ) -> pd.DataFrame:
    """Save 3-D beta NIfTIs to disk and return a manifest DataFrame.

    Parameters
    ----------
    betas_dict : dict
        Keys: (run_id, trial_idx).
        Values: dict with 'img', 'condition', 'onset', 'run'.
    method_name : str
        'lsa' or 'lss' — used as subdirectory name and CSV suffix.
    output_dir : str
        Root output directory.
    subject_id : str
    save_4d : bool
        If True, also save 4-D NIfTIs stacked by condition × run.

    Returns
    -------
    manifest_df : pd.DataFrame
        One row per trial. Columns: subject, run, trial_idx, condition,
        onset_s, beta_file, method.

    Example
    -------
    >>> manifest = save_beta_manifest(lss_betas, "lss", output_dir, "sub-1")
    >>> manifest.to_csv("beta_manifest_lss.csv", index=False)
    """
    beta_dir = Path(output_dir) / method_name
    beta_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for (run_id, trial_idx), info in sorted(betas_dict.items()):
        fname = (f"{subject_id}_run-{run_id:02d}_trial-{trial_idx:04d}"
                 f"_condition-{info['condition']}_{method_name}_beta.nii.gz")
        fpath = str(beta_dir / fname)
        info['img'].to_filename(fpath)
        rows.append(dict(subject=subject_id, run=run_id, trial_idx=trial_idx,
                         condition=info['condition'], onset_s=round(info['onset'], 3),
                         beta_file=fpath, method=method_name))
    df       = pd.DataFrame(rows)
    csv_path = str(Path(output_dir) / f'beta_manifest_{method_name}.csv')
    df.to_csv(csv_path, index=False)
    print(f"  [{method_name.upper()}] {len(df)} betas → {csv_path}")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 3 — MVPA DECODING
# ══════════════════════════════════════════════════════════════════════════════

from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import (LeaveOneGroupOut, StratifiedKFold,
                                     cross_val_predict)
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                              confusion_matrix, roc_auc_score,
                              ConfusionMatrixDisplay, classification_report)


def load_beta_matrix(manifest_csv: str,
                     mask_img: Optional[nib.Nifti1Image] = None,
                     conditions: Optional[List[str]] = None
                     ) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
                                NiftiMasker, np.ndarray, int]:
    """Load beta images from a manifest CSV and stack into a feature matrix.

    Parameters
    ----------
    manifest_csv : str
        Path to the CSV produced by save_beta_manifest().
    mask_img : nib.Nifti1Image or None
        Brain mask. If None, derived from the NiftiMasker default.
    conditions : list of str or None
        Subset of conditions to keep. None → all conditions.

    Returns
    -------
    X : np.ndarray
        Feature matrix, shape (n_trials, n_active_voxels).
        Active voxels = voxels with at least one non-zero trial.
    y_raw : np.ndarray
        String condition labels, shape (n_trials,).
    runs_arr : np.ndarray
        Run index per trial, shape (n_trials,).
    masker : NiftiMasker
        Fitted masker for inverse_transform later.
    active_voxels : np.ndarray
        Boolean mask, shape (n_mask_voxels,). True = kept in X.
    n_mask_voxels : int
        Total voxels in the mask (before dropping zeros).

    Example
    -------
    >>> X, y_raw, runs, masker, active, n_mask = load_beta_matrix(
    ...     "beta_manifest_lss.csv", mask_img=mask)
    >>> print(f"X shape: {X.shape}")  # (n_trials, n_active_voxels)
    """
    manifest = pd.read_csv(manifest_csv)
    if conditions:
        manifest = manifest[manifest['condition'].isin(conditions)].copy()

    masker = NiftiMasker(mask_img=mask_img, standardize=False, detrend=False)
    masker.fit(nib.load(manifest.iloc[0]['beta_file']))

    rows, labels, runs = [], [], []
    for _, row in manifest.iterrows():
        img = nib.load(row['beta_file'])
        rows.append(masker.transform(img).ravel())
        labels.append(row['condition'])
        runs.append(int(row['run']))

    X         = np.vstack(rows)
    n_mask    = X.shape[1]
    active    = (X != 0).any(axis=0)
    X         = X[:, active]
    print(f"  Beta matrix: {X.shape[0]} trials × {X.shape[1]} active voxels"
          f"  ({100*active.sum()/n_mask:.1f}% of {n_mask} mask voxels)")
    return X, np.array(labels), np.array(runs, dtype=int), masker, active, n_mask


def run_svm_decoding(X: np.ndarray,
                     y_raw: np.ndarray,
                     runs_arr: np.ndarray,
                     feature_selection: str = 'anova',
                     anova_percentile: float = 10.0,
                     svm_C: float = 1.0,
                     svm_kernel: str = 'linear',
                     leave_one_run_out: bool = True,
                     n_splits: int = 5
                     ) -> dict:
    """Run SVM classification with cross-validation on a beta-weight matrix.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix (n_trials, n_voxels).
    y_raw : np.ndarray
        String condition labels.
    runs_arr : np.ndarray
        Run index per trial (used for leave-one-run-out CV).
    feature_selection : str
        'anova' or 'all'. ANOVA keeps the top percentile of voxels.
    anova_percentile : float
        Percentile kept by ANOVA feature selection (default 10).
    svm_C : float
        SVM regularisation parameter (default 1.0).
    svm_kernel : str
        SVM kernel (default 'linear').
    leave_one_run_out : bool
        Use leave-one-run-out CV if True, else StratifiedKFold (default True).
    n_splits : int
        Folds for StratifiedKFold when leave_one_run_out=False (default 5).

    Returns
    -------
    results : dict with keys
        'le'         : LabelEncoder
        'y'          : integer labels
        'y_pred'     : cross-validated predictions
        'y_proba'    : cross-validated class probabilities
        'fold_accs'  : list of per-fold accuracy
        'mean_acc'   : float
        'sem_acc'    : float
        'cm'         : normalised confusion matrix
        'cm_raw'     : raw confusion matrix
        'cv_name'    : str
        'pipe'       : fitted Pipeline (trained on all data)
        'selector'   : feature selector (or None)
        'n_selected' : int

    Example
    -------
    >>> res = run_svm_decoding(X, y_raw, runs_arr)
    >>> print(f"Mean accuracy: {res['mean_acc']*100:.2f}%")
    """
    le = LabelEncoder(); y = le.fit_transform(y_raw)

    if feature_selection == 'anova':
        selector = SelectPercentile(f_classif, percentile=anova_percentile)
    else:
        selector = None

    svm_clf = SVC(C=svm_C, kernel=svm_kernel, class_weight='balanced',
                  random_state=42, probability=True)
    pipe = Pipeline([
        ('fs',  selector if selector is not None else 'passthrough'),
        ('svm', svm_clf),
    ])

    if leave_one_run_out:
        cv = LeaveOneGroupOut(); groups = runs_arr; cv_name = "Leave-One-Run-Out"
    else:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        groups  = None; cv_name = f"Stratified {n_splits}-Fold"

    t0      = time.perf_counter()
    y_pred  = cross_val_predict(pipe, X, y, cv=cv, groups=groups, n_jobs=-1)
    y_proba = cross_val_predict(pipe, X, y, cv=cv, groups=groups,
                                method='predict_proba', n_jobs=-1)
    print(f"  CV done in {time.perf_counter()-t0:.1f}s")

    splits    = list(cv.split(X, y, groups) if groups is not None else cv.split(X, y))
    fold_accs = [accuracy_score(y[test], y_pred[test]) for _, test in splits]
    mean_acc  = float(np.mean(fold_accs))
    sem_acc   = float(np.std(fold_accs) / np.sqrt(len(fold_accs)))

    cm     = confusion_matrix(y, y_pred, normalize='true')
    cm_raw = confusion_matrix(y, y_pred)

    # Fit final model on all data
    pipe.fit(X, y)
    n_sel = int(pipe.named_steps['fs'].get_support().sum()) \
            if selector is not None else X.shape[1]

    return dict(le=le, y=y, y_pred=y_pred, y_proba=y_proba,
                fold_accs=fold_accs, mean_acc=mean_acc, sem_acc=sem_acc,
                cm=cm, cm_raw=cm_raw, cv_name=cv_name,
                pipe=pipe, selector=selector, n_selected=n_sel)


def compute_importance_map(pipe: Pipeline,
                           X: np.ndarray,
                           masker: NiftiMasker,
                           active_voxels: np.ndarray,
                           n_mask_voxels: int
                           ) -> nib.Nifti1Image:
    """Extract mean |SVM weight| importance map and convert to NIfTI.

    Handles the two-layer index expansion needed when ANOVA feature selection
    reduces the feature space before classification:
      n_selected → n_active_voxels → n_mask_voxels

    Parameters
    ----------
    pipe : Pipeline
        Fitted pipeline with steps 'fs' and 'svm'.
    X : np.ndarray
        Feature matrix used for fitting (n_trials, n_active_voxels).
    masker : NiftiMasker
        Fitted masker (to call inverse_transform).
    active_voxels : np.ndarray
        Boolean array, shape (n_mask_voxels,). True where X columns come from.
    n_mask_voxels : int
        Total voxels in the mask.

    Returns
    -------
    importance_img : nib.Nifti1Image
        3-D NIfTI of mean absolute SVM weights, in mask voxel space.

    Example
    -------
    >>> imp_img = compute_importance_map(pipe, X, masker, active_voxels, n_mask)
    >>> imp_img.to_filename("importance_map.nii.gz")
    """
    svm_step = pipe.named_steps['svm']
    fs_step  = pipe.named_steps['fs']
    coef     = svm_step.coef_                         # (n_hyperplanes, n_selected)
    imp_sel  = np.mean(np.abs(coef), axis=0)          # (n_selected,)

    # Layer 1: n_selected → n_active_voxels
    imp_active = np.zeros(X.shape[1])
    if fs_step != 'passthrough' and hasattr(fs_step, 'get_support'):
        imp_active[fs_step.get_support()] = imp_sel
    else:
        imp_active = imp_sel

    # Layer 2: n_active_voxels → n_mask_voxels
    imp_full = np.zeros(n_mask_voxels)
    imp_full[active_voxels] = imp_active

    return masker.inverse_transform(imp_full)


def plot_confusion_matrix(cm: np.ndarray, cm_raw: np.ndarray,
                          class_names: List[str],
                          title_prefix: str = "") -> plt.Figure:
    """Plot raw and row-normalised confusion matrices side by side.

    Parameters
    ----------
    cm : np.ndarray
        Row-normalised confusion matrix (values 0-1).
    cm_raw : np.ndarray
        Raw count confusion matrix.
    class_names : list of str
    title_prefix : str

    Returns
    -------
    fig : plt.Figure

    Example
    -------
    >>> fig = plot_confusion_matrix(cm, cm_raw, le.classes_, title_prefix="sub-1")
    >>> plt.show()
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ConfusionMatrixDisplay(cm_raw, display_labels=class_names).plot(
        ax=axes[0], colorbar=False, cmap='Blues')
    axes[0].set_title(f'{title_prefix} Confusion matrix (counts)')
    ConfusionMatrixDisplay(cm, display_labels=class_names).plot(
        ax=axes[1], colorbar=True, cmap='Blues', values_format='.2f')
    axes[1].set_title(f'{title_prefix} Confusion matrix (row-normalised)')
    plt.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 4 — SEARCHLIGHT
# ══════════════════════════════════════════════════════════════════════════════

from nilearn.decoding import SearchLight


def run_searchlight(manifest_csv: str,
                    mask_img: nib.Nifti1Image,
                    conditions: Optional[List[str]] = None,
                    radius_mm: float = 6.0,
                    svm_C: float = 1.0,
                    leave_one_run_out: bool = True,
                    n_splits: int = 5,
                    n_jobs: int = -1,
                    process_mask_img: Optional[nib.Nifti1Image] = None
                    ) -> Tuple[nib.Nifti1Image, np.ndarray, LabelEncoder]:
    """Run a whole-brain searchlight decoding analysis.

    Loads beta images from a manifest CSV, stacks them into a 4D NIfTI, and
    fits a SearchLight estimator at every centre voxel in process_mask_img.

    Parameters
    ----------
    manifest_csv : str
        Path to beta_manifest_*.csv from save_beta_manifest().
    mask_img : nib.Nifti1Image
        Brain mask: voxels eligible to be included in any searchlight sphere.
    conditions : list of str or None
        Subset of conditions. None → all conditions.
    radius_mm : float
        Searchlight sphere radius (default 6 mm).
    svm_C : float
        SVM regularisation.
    leave_one_run_out : bool
        Use leave-one-run-out CV (default True).
    n_splits : int
        Folds for StratifiedKFold when leave_one_run_out=False.
    n_jobs : int
        Parallel workers (-1 = all CPUs).
    process_mask_img : nib.Nifti1Image or None
        Voxels to use as searchlight centres. None → same as mask_img.

    Returns
    -------
    scores_img : nib.Nifti1Image
        3-D NIfTI of per-voxel decoding accuracy.
    y : np.ndarray
        Integer labels (for reference).
    le : LabelEncoder

    Example
    -------
    >>> scores_img, y, le = run_searchlight("beta_manifest_lss.csv", mask_img)
    >>> scores_img.to_filename("searchlight_accuracy.nii.gz")
    """
    manifest = pd.read_csv(manifest_csv)
    if conditions:
        manifest = manifest[manifest['condition'].isin(conditions)].copy()

    le       = LabelEncoder()
    y        = le.fit_transform(manifest['condition'].values)
    runs_arr = manifest['run'].values.astype(int)

    # Build 4D NIfTI (trials × voxels stored as 4th dim)
    beta_imgs = [nib.load(p) for p in manifest['beta_file']]
    imgs_4d   = image.concat_imgs(beta_imgs)
    print(f"  4D image: {imgs_4d.shape}  ({len(beta_imgs)} trials)")

    if process_mask_img is None:
        process_mask_img = mask_img

    if leave_one_run_out:
        cv = LeaveOneGroupOut(); groups = runs_arr
    else:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        groups = None

    estimator = SVC(C=svm_C, kernel='linear', class_weight='balanced', random_state=42)
    sl = SearchLight(
        mask_img=mask_img, process_mask_img=process_mask_img,
        radius=radius_mm, estimator=estimator,
        cv=cv, scoring='accuracy', n_jobs=n_jobs, verbose=1)

    t0 = time.perf_counter()
    if groups is not None:
        sl.fit(imgs_4d, y, groups=groups)
    else:
        sl.fit(imgs_4d, y)
    print(f"  Searchlight complete in {(time.perf_counter()-t0)/60:.1f} min")

    scores_img = image.new_img_like(mask_img, sl.scores_, copy_header=True)
    return scores_img, y, le


def summarise_searchlight(scores_img: nib.Nifti1Image,
                           process_mask_img: nib.Nifti1Image,
                           chance: float,
                           top_n: int = 20
                           ) -> Tuple[dict, pd.DataFrame]:
    """Compute summary statistics and peak voxel table for a searchlight map.

    Parameters
    ----------
    scores_img : nib.Nifti1Image
        Per-voxel accuracy NIfTI from run_searchlight().
    process_mask_img : nib.Nifti1Image
        The process mask used during search (defines which voxels to summarise).
    chance : float
        Chance level accuracy (e.g. 1/8 for 8-class decoding).
    top_n : int
        Number of peak voxels to report (default 20).

    Returns
    -------
    summary : dict
        Scalar summary metrics.
    peaks_df : pd.DataFrame
        Top-N voxels sorted by accuracy with MNI coordinates.

    Example
    -------
    >>> summary, peaks = summarise_searchlight(scores_img, mask, chance=0.125)
    >>> print(f"Peak accuracy: {summary['max_acc']*100:.2f}%")
    """
    scores_arr  = scores_img.get_fdata()
    aff         = scores_img.affine
    pmask       = process_mask_img.get_fdata() > 0

    mask_scores = scores_arr[pmask]
    above       = mask_scores[mask_scores > chance]

    summary = dict(
        chance_level=round(float(chance), 4),
        mean_acc_all=round(float(mask_scores.mean()), 4),
        max_acc=round(float(mask_scores.max()), 4),
        n_centre_voxels=int(pmask.sum()),
        n_above_chance=int((mask_scores > chance).sum()),
        pct_above_chance=round(100*float((mask_scores > chance).mean()), 2),
        mean_acc_above_chance=(round(float(above.mean()), 4) if len(above) else None),
    )

    # Peak voxels
    vox_ijk = np.column_stack(np.where(pmask))
    vox_acc = scores_arr[vox_ijk[:, 0], vox_ijk[:, 1], vox_ijk[:, 2]]
    top_idx = np.argsort(vox_acc)[::-1][:top_n]
    rows = []
    for idx in top_idx:
        i, j, k = vox_ijk[idx]
        xyz = nib.affines.apply_affine(aff, [i, j, k])
        rows.append({'vox_i': int(i), 'vox_j': int(j), 'vox_k': int(k),
                     'mni_x': round(float(xyz[0]), 1),
                     'mni_y': round(float(xyz[1]), 1),
                     'mni_z': round(float(xyz[2]), 1),
                     'accuracy': round(float(vox_acc[idx]), 4)})
    return summary, pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# UTILITY
# ══════════════════════════════════════════════════════════════════════════════

@contextmanager
def track_runtime(label: str = "block"):
    """Context manager that prints elapsed time after a code block."""
    t0 = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - t0
        print(f"  [{label}] {elapsed:.1f}s")


def get_bold_paths(fmriprep_root: str, subject: str,
                   task: str = "objectviewing", n_runs: int = 12,
                   desc: str = "preproc") -> List[str]:
    """Return a list of fMRIPrep BOLD file paths for all runs of a subject.

    Parameters
    ----------
    fmriprep_root : str
    subject : str   e.g. 'sub-1'
    task : str
    n_runs : int
    desc : str      fMRIPrep desc entity (default 'preproc')

    Returns
    -------
    paths : list of str  (only existing files)
    """
    paths = []
    for run in range(1, n_runs + 1):
        p = (f"{fmriprep_root}/{subject}/func/"
             f"{subject}_task-{task}_run-{run}_desc-{desc}_bold.nii.gz")
        if os.path.exists(p):
            paths.append(p)
    return paths


def print_section(title: str, width: int = 55) -> None:
    """Print a formatted section header."""
    print()
    print("=" * width)
    print(f"  {title}")
    print("=" * width)
