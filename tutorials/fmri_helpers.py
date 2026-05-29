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


# ── LPS→RAS affine correction constant (ANTsPy uses LPS; nibabel uses RAS) ───
_LPS_TO_RAS = np.diag([-1., -1., 1., 1.])


def ants_to_nibabel(ants_img, dtype=np.float32) -> nib.Nifti1Image:
    """Convert an ANTsPy image to a nibabel NIfTI, correcting LPS→RAS coordinates.

    ANTsPy stores images in **LPS** (Left-Posterior-Superior) convention
    internally, whereas nibabel assumes **RAS** (Right-Anterior-Superior).
    This function corrects the affine matrix by flipping the x and y axes
    with the diagonal sign matrix ``diag([-1, -1, 1, 1])``.

    Parameters
    ----------
    ants_img : ants.ANTsImage
        ANTsPy image returned by ``ants.apply_transforms()`` or similar.
    dtype : numpy dtype, optional
        Data type for the output array (default float32).

    Returns
    -------
    nib_img : nib.Nifti1Image
        NIfTI image in RAS coordinates, compatible with nibabel / nilearn.

    Example
    -------
    >>> warped_ants = ants.apply_transforms(fixed=ref, moving=mni_mask,
    ...                                     transformlist=[xfm1, xfm2],
    ...                                     interpolator='genericLabel')
    >>> warped_nib = ants_to_nibabel(warped_ants, dtype=np.uint8)
    >>> print(warped_nib.shape)  # same spatial shape as ref
    """
    data = ants_img.numpy().astype(dtype)
    # Build LPS affine from ANTsPy metadata
    direction = np.array(ants_img.direction).reshape(3, 3)
    spacing   = np.array(ants_img.spacing)
    origin    = np.array(ants_img.origin)

    affine_lps          = np.eye(4)
    affine_lps[:3, :3]  = direction * spacing[np.newaxis, :]
    affine_lps[:3, 3]   = origin

    affine_ras = _LPS_TO_RAS @ affine_lps
    return nib.Nifti1Image(data, affine_ras)


def warp_group_mask_to_func(
    group_mask_mni,
    func_ref_img,
    mni_to_t1w_xfm: str,
    t1w_to_func_xfm: str,
    ants_available: bool = True,
) -> nib.Nifti1Image:
    """Warp a group-level MNI mask to functional (scanner) space.

    Applies the two-step fMRIPrep transform chain in right-to-left order:

    ``MNI mask → [MNI→T1w .h5] → [T1w→func .txt] → functional space``

    Uses ANTsPy with the ``genericLabel`` interpolator when available, or
    falls back to ``nilearn.image.resample_to_img`` (nearest-neighbour) if
    ANTsPy is not installed.

    Parameters
    ----------
    group_mask_mni : str or nib.Nifti1Image
        Binary mask in MNI152NLin6Asym space (any resolution).
    func_ref_img : str or nib.Nifti1Image
        Functional reference image that defines the target voxel grid
        (e.g., fMRIPrep ``_boldref.nii.gz``).
    mni_to_t1w_xfm : str
        Path to the composite MNI→T1w warp file (.h5) from fMRIPrep.
    t1w_to_func_xfm : str
        Path to the T1w→functional affine file (.txt, BBR) from fMRIPrep.
    ants_available : bool, optional
        If True (default), use ANTsPy. If False, fall back to nilearn.

    Returns
    -------
    warped_mask : nib.Nifti1Image
        Binary mask in functional space, resampled to the grid of
        ``func_ref_img``.

    Notes
    -----
    The transform list passed to ANTsPy is ``[t1w_to_func_xfm, mni_to_t1w_xfm]``
    (innermost first, applied right-to-left). This matches fMRIPrep's convention.

    Example
    -------
    >>> warped = warp_group_mask_to_func(
    ...     group_mask_mni  = "/path/to/group_mask_MNI.nii.gz",
    ...     func_ref_img    = nib.load("sub-1_run-01_boldref.nii.gz"),
    ...     mni_to_t1w_xfm  = "sub-1_from-MNI152_to-T1w_mode-image_xfm.h5",
    ...     t1w_to_func_xfm = "sub-1_run-01_from-T1w_to-scanner_mode-image_xfm.txt",
    ... )
    >>> print(warped.get_fdata().sum())  # number of mask voxels in func space
    """
    # Ensure nibabel images are on disk for ANTsPy (it needs paths)
    if not isinstance(group_mask_mni, str):
        import tempfile, os as _os
        _tmp = tempfile.NamedTemporaryFile(suffix='.nii.gz', delete=False)
        group_mask_mni.to_filename(_tmp.name)
        group_mask_mni = _tmp.name

    func_ref_img = image.load_img(func_ref_img)

    if ants_available:
        try:
            import ants  # type: ignore
            ants_ref    = ants.image_read(func_ref_img.get_filename()
                                          if hasattr(func_ref_img, 'get_filename')
                                          else _nibabel_to_ants_tmpfile(func_ref_img))
            ants_moving = ants.image_read(group_mask_mni)

            warped_ants = ants.apply_transforms(
                fixed        = ants_ref,
                moving       = ants_moving,
                transformlist= [t1w_to_func_xfm, mni_to_t1w_xfm],
                interpolator = 'genericLabel',
            )
            warped_nib = ants_to_nibabel(warped_ants, dtype=np.uint8)
            # Binarise (genericLabel can return small fractions near borders)
            warped_data = (warped_nib.get_fdata() > 0.5).astype(np.uint8)
            return nib.Nifti1Image(warped_data, warped_nib.affine, warped_nib.header)
        except Exception as e:
            warnings.warn(f"ANTsPy warp failed ({e}); falling back to nilearn resampling.")

    # ── nilearn fallback ─────────────────────────────────────────────────────
    mask_img = image.load_img(group_mask_mni)
    resampled = image.resample_to_img(mask_img, func_ref_img,
                                      interpolation='nearest')
    data = (resampled.get_fdata() > 0.5).astype(np.uint8)
    return nib.Nifti1Image(data, resampled.affine, resampled.header)


def _nibabel_to_ants_tmpfile(nib_img: nib.Nifti1Image) -> str:
    """Save a nibabel image to a temporary file and return the path."""
    import tempfile
    tmp = tempfile.NamedTemporaryFile(suffix='.nii.gz', delete=False)
    nib_img.to_filename(tmp.name)
    return tmp.name


def filter_mask_by_variance(
    warped_mask_img: nib.Nifti1Image,
    bold_img: nib.Nifti1Image,
    var_threshold: float = 1e-6,
) -> Tuple[nib.Nifti1Image, int]:
    """Remove voxels with near-zero temporal variance from a functional mask.

    Voxels that never change over time (e.g., outside the brain, corrupted
    by signal dropout) carry no information and can cause numerical instability
    in sklearn estimators. This function removes any mask voxel whose temporal
    variance in the BOLD signal falls below ``var_threshold``.

    Parameters
    ----------
    warped_mask_img : nib.Nifti1Image
        Binary mask in functional space (output of
        :func:`warp_group_mask_to_func`).
    bold_img : nib.Nifti1Image
        4-D BOLD image in the same space.  Must share the same voxel grid
        as ``warped_mask_img`` (resample first if needed).
    var_threshold : float, optional
        Variance threshold below which a voxel is excluded (default 1e-6).

    Returns
    -------
    filtered_mask : nib.Nifti1Image
        Binary mask with zero-variance voxels removed.
    n_removed : int
        Number of voxels removed from the mask.

    Example
    -------
    >>> filtered, n = filter_mask_by_variance(warped_mask, bold_img)
    >>> print(f"Removed {n} zero-variance voxels from mask")
    """
    bold_img_res = image.resample_to_img(bold_img, warped_mask_img,
                                          interpolation='continuous')
    mask_data = warped_mask_img.get_fdata().astype(bool)
    bold_data = bold_img_res.get_fdata()

    # Extract time series for masked voxels (shape: n_voxels × n_timepoints)
    masked_ts = bold_data[mask_data]             # (n_vox, T)
    voxel_var = masked_ts.var(axis=-1)           # (n_vox,)

    keep    = voxel_var >= var_threshold
    n_removed = int((~keep).sum())

    # Rebuild 3-D mask keeping only high-variance voxels
    new_mask_data    = np.zeros_like(mask_data, dtype=np.uint8)
    mask_indices     = np.where(mask_data)
    kept_coords      = tuple(ax[keep] for ax in mask_indices)
    new_mask_data[kept_coords] = 1

    filtered_mask = nib.Nifti1Image(new_mask_data,
                                    warped_mask_img.affine,
                                    warped_mask_img.header)
    return filtered_mask, n_removed


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
    # Detect SVM step: first step that has coef_ after fitting
    svm_step = next(
        (s for s in pipe.named_steps.values() if hasattr(s, 'coef_')),
        None,
    )
    if svm_step is None:
        raise ValueError(
            "compute_importance_map: no fitted step with coef_ found in pipeline. "
            f"Steps: {list(pipe.named_steps.keys())}"
        )
 
    # Detect feature-selection step: first step that has get_support()
    fs_step = next(
        (s for s in pipe.named_steps.values()
         if s != 'passthrough' and hasattr(s, 'get_support')),
        None,
    )
 
    coef    = svm_step.coef_                          # (n_hyperplanes, n_selected)
    imp_sel = np.mean(np.abs(coef), axis=0)           # (n_selected,)
 
    # Layer 1: n_selected → n_active_voxels
    imp_active = np.zeros(X.shape[1])
    if fs_step is not None:
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


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 5 — TUTORIAL HELPERS  (beta → MVPA combined pipeline)
# ══════════════════════════════════════════════════════════════════════════════

def stack_betas_from_dict(betas_dict: dict,
                          mask_img: nib.Nifti1Image,
                          conditions: Optional[List[str]] = None
                          ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, NiftiMasker]:
    """Stack a betas dict (from fit_lsa_run / fit_lss_run) into a feature matrix.

    Converts the per-trial beta dict produced by ``fit_lsa_run`` or
    ``fit_lss_run`` into the (n_trials × n_voxels) matrix required by
    the MVPA classifier, along with condition labels and run indices.

    Parameters
    ----------
    betas_dict : dict
        Keys: (run_id, trial_idx).
        Values: dict with keys 'img' (Nifti1Image), 'condition' (str), 'run' (int).
        Produced by ``fit_lsa_run`` / ``fit_lss_run``.
    mask_img : nib.Nifti1Image
        Brain mask. Used to fit a NiftiMasker that applies to every beta image.
    conditions : list of str or None
        Subset of conditions to keep.  None → all conditions.

    Returns
    -------
    X : np.ndarray
        Feature matrix, shape (n_trials, n_voxels), dtype float32.
    y_raw : np.ndarray
        String condition labels, shape (n_trials,).
    runs_arr : np.ndarray
        Run index per trial, shape (n_trials,), dtype int.
    masker : NiftiMasker
        Fitted masker — call masker.inverse_transform(vec) to rebuild NIfTIs.

    Example
    -------
    >>> X, y_raw, runs_arr, masker = stack_betas_from_dict(all_betas, mask_img)
    >>> print(f"Feature matrix: {X.shape}")   # (n_trials, n_voxels)
    """
    # Sort keys for reproducible ordering
    sorted_keys = sorted(betas_dict.keys())

    # Optional condition filter
    if conditions is not None:
        sorted_keys = [k for k in sorted_keys
                       if betas_dict[k]['condition'] in conditions]

    if not sorted_keys:
        raise ValueError("No trials remain after condition filtering.")

    # Fit masker on first image to set up resampler
    first_img = betas_dict[sorted_keys[0]]['img']
    masker = NiftiMasker(mask_img=mask_img, standardize=False, detrend=False)
    masker.fit(first_img)

    rows, labels, runs = [], [], []
    for key in sorted_keys:
        info = betas_dict[key]
        vec = masker.transform(info['img']).ravel()
        rows.append(vec)
        labels.append(info['condition'])
        runs.append(int(info['run']))

    X = np.vstack(rows).astype(np.float32)
    print(f"  Feature matrix: {X.shape[0]} trials × {X.shape[1]} voxels")
    return X, np.array(labels), np.array(runs, dtype=int), masker


def run_permutation_test(X: np.ndarray,
                         y: np.ndarray,
                         runs_arr: np.ndarray,
                         pipe,
                         observed_acc: float,
                         n_permutations: int = 1000,
                         method: str = 'label_shuffle',
                         seed: int = 42
                         ) -> Tuple[np.ndarray, float]:
    """Non-parametric permutation test for MVPA classification accuracy.

    Builds a null distribution by either shuffling class labels or applying
    sign flips (for zero-centred weight maps), then computes a one-sided
    p-value: ``p = #{null_acc >= observed_acc} / n_permutations``.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix (n_trials, n_voxels).
    y : np.ndarray
        Integer class labels (n_trials,).
    runs_arr : np.ndarray
        Run index per trial — used as ``groups`` for LeaveOneGroupOut.
    pipe : sklearn Pipeline
        Fitted (or unfitted) pipeline with steps 'feature_selection' and 'svm'.
        Re-used structure only; a fresh clone is fitted per permutation.
    observed_acc : float
        The true mean CV accuracy to compare against.
    n_permutations : int
        Number of shuffles (default 1000; use ≥ 5000 for publication).
    method : str
        ``'label_shuffle'`` (default): randomly permute y across all trials.
        ``'sign_flip'``    : multiply each trial's row in X by ±1 (for mean-centred
                             importance maps; not suitable for raw BOLD).
    seed : int
        Random seed (default 42).

    Returns
    -------
    null_accs : np.ndarray
        Shape (n_permutations,). Accuracy under each permutation.
    p_value : float
        One-sided p-value.  Floored at ``1 / n_permutations``.

    Example
    -------
    >>> null_accs, p_val = run_permutation_test(
    ...     X, y, runs_arr, pipe, observed_acc=mean_acc, n_permutations=1000)
    >>> print(f"p = {p_val:.4f}")
    """
    from sklearn.base import clone
    from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict
    from sklearn.metrics import accuracy_score

    rng  = np.random.default_rng(seed)
    cv   = LeaveOneGroupOut()
    null_accs = np.empty(n_permutations)

    for i in range(n_permutations):
        if method == 'sign_flip':
            signs  = rng.choice([-1.0, 1.0], size=(X.shape[0], 1))
            X_perm = X * signs
            y_perm = y
        else:  # label_shuffle
            X_perm = X
            y_perm = rng.permutation(y)

        y_null = cross_val_predict(
            clone(pipe), X_perm, y_perm,
            cv=cv, groups=runs_arr, n_jobs=-1
        )
        null_accs[i] = accuracy_score(y_perm, y_null)

        if (i + 1) % 200 == 0:
            print(f"  Permutation {i+1}/{n_permutations} — "
                  f"null mean so far: {null_accs[:i+1].mean()*100:.2f}%")

    p_value = float(np.maximum((null_accs >= observed_acc).mean(),
                               1.0 / n_permutations))
    print(f"\nPermutation test complete.")
    print(f"  Observed accuracy : {observed_acc*100:.2f}%")
    print(f"  Null distribution : {null_accs.mean()*100:.2f}% ± "
          f"{null_accs.std()*100:.2f}% SD")
    print(f"  p-value           : {p_value:.4f}  "
          f"({'significant ✓' if p_value < 0.05 else 'not significant ✗'} at α = 0.05)")
    return null_accs, p_value


def make_block_events(events_df: pd.DataFrame,
                      block_gap_s: float = 2.0
                      ) -> pd.DataFrame:
    """Merge consecutive same-condition trials into block-level events.

    Trials are grouped into blocks when consecutive trials of the same
    ``trial_type`` are separated by less than ``block_gap_s`` seconds
    (approximated as onset_next − onset_current − duration_current < block_gap_s).
    The resulting DataFrame has one row per block, with onset = first trial onset,
    duration = last trial offset − first trial onset, and the shared trial_type.

    This is useful for visualising block structure or fitting a block-level
    (rather than trial-level) GLM as a sanity check.

    Parameters
    ----------
    events_df : pd.DataFrame
        Must have columns: onset, duration, trial_type.  Should be sorted by onset.
    block_gap_s : float
        Maximum gap (s) between consecutive trials of the same condition
        for them to be merged into one block (default 2.0 s).

    Returns
    -------
    blocks_df : pd.DataFrame
        Columns: onset, duration, trial_type, n_trials.

    Example
    -------
    >>> blocks = make_block_events(events_df, block_gap_s=2.0)
    >>> print(blocks[['onset', 'duration', 'trial_type', 'n_trials']])
    """
    df = events_df.sort_values('onset').reset_index(drop=True)
    blocks, current = [], None

    for _, row in df.iterrows():
        if current is None:
            current = {'onset':      row['onset'],
                       'end':        row['onset'] + row['duration'],
                       'trial_type': row['trial_type'],
                       'n_trials':   1}
        else:
            gap = row['onset'] - current['end']
            if row['trial_type'] == current['trial_type'] and gap <= block_gap_s:
                current['end']      = row['onset'] + row['duration']
                current['n_trials'] += 1
            else:
                blocks.append(current)
                current = {'onset':      row['onset'],
                           'end':        row['onset'] + row['duration'],
                           'trial_type': row['trial_type'],
                           'n_trials':   1}

    if current is not None:
        blocks.append(current)

    out = pd.DataFrame(blocks)
    out['duration'] = out['end'] - out['onset']
    return out[['onset', 'duration', 'trial_type', 'n_trials']].reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════════════
# HELP / DISCOVERY
# ══════════════════════════════════════════════════════════════════════════════

def list_helpers(verbose: bool = False) -> None:
    """Print a formatted catalogue of all public helper functions in this module.

    For each function, shows its one-line summary (first line of docstring),
    call signature, and — when ``verbose=True`` — the full parameter list.

    Parameters
    ----------
    verbose : bool
        If True, print the full docstring for every function (default False).

    Example
    -------
    >>> list_helpers()            # compact view
    >>> list_helpers(verbose=True)  # full docstrings
    """
    import inspect, textwrap

    # Registry: module → list of (function_name, function_object, one-liner)
    _REGISTRY = [
        # Module 1 — Preprocessing
        ("Module 1 — Preprocessing", [
            load_bold_and_mask,
            load_confounds,
            compute_fd,
            clean_bold_run,
            plot_motion_qc,
            ants_to_nibabel,
            warp_group_mask_to_func,
            filter_mask_by_variance,
        ]),
        # Module 2 — Beta Estimation
        ("Module 2 — Beta Estimation", [
            make_lsa_events,
            make_lss_events,
            fit_lsa_run,
            fit_lss_run,
            save_beta_manifest,
        ]),
        # Module 3 — MVPA Decoding
        ("Module 3 — MVPA Decoding", [
            load_beta_matrix,
            run_svm_decoding,
            compute_importance_map,
            plot_confusion_matrix,
        ]),
        # Module 4 — Searchlight
        ("Module 4 — Searchlight", [
            run_searchlight,
            summarise_searchlight,
        ]),
        # Module 5 — Tutorial helpers
        ("Module 5 — Tutorial (beta → MVPA combined)", [
            stack_betas_from_dict,
            run_permutation_test,
            make_block_events,
        ]),
        # Utilities
        ("Utility", [
            track_runtime,
            get_bold_paths,
            print_section,
            list_helpers,
        ]),
    ]

    W = 72
    print()
    print("█" * W)
    print(f"{'fmri_helpers — Available Functions':^{W}}")
    print("█" * W)

    for module_name, funcs in _REGISTRY:
        print()
        print(f"  ┌─ {module_name} {'─'*(W - len(module_name) - 6)}")
        for fn in funcs:
            try:
                sig     = str(inspect.signature(fn))
                # Trim long signatures
                if len(sig) > 55:
                    sig = sig[:52] + "..."
                doc     = (fn.__doc__ or "").strip()
                one_line = doc.split("\n")[0] if doc else "(no docstring)"
                print(f"  │  {fn.__name__}{sig}")
                print(f"  │      → {one_line}")
                if verbose and doc:
                    wrapped = textwrap.indent(
                        textwrap.fill(doc, width=60), prefix="  │      ")
                    print(wrapped)
                print("  │")
            except (TypeError, ValueError):
                print(f"  │  {fn.__name__}  (signature unavailable)")
                print("  │")

    print("  └" + "─" * (W - 2))
    print()
    print("  Usage examples:")
    print("    list_helpers()              # this compact view")
    print("    list_helpers(verbose=True)  # full docstrings")
    print("    help(fit_lsa_run)           # single function deep-dive")
    print()

    
    

import pandas as pd
import numpy as np
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# Core grouping function
# ─────────────────────────────────────────────────────────────────────────────

def events_to_blocks(
    events: pd.DataFrame,
    gap_threshold: float = 2.0,
    condition_col: str = "trial_type",
    onset_col: str = "onset",
    duration_col: str = "duration",
    keep_cols: Optional[list] = None,
    strategy: str = "gap",
) -> pd.DataFrame:
    """
    Merge individual BIDS event trials into block-level regressors.

    Parameters
    ----------
    events : pd.DataFrame
        BIDS events table with at least `onset`, `duration`, and
        `trial_type` columns.
    gap_threshold : float
        Maximum inter-trial gap (seconds) allowed within a block.
        Trials of the same type whose ITI <= this value are merged.
        Ignored when strategy="adjacent".
    condition_col : str
        Column name that identifies the condition / trial type.
    onset_col : str
        Column name for trial onset (seconds).
    duration_col : str
        Column name for trial duration (seconds).
    keep_cols : list of str, optional
        Additional columns to carry forward.  When trials within a
        block have identical values the value is preserved; if they
        differ, the block cell contains a pipe-separated list of
        unique values.
    strategy : {"gap", "adjacent"}
        Grouping strategy (see module docstring).

    Returns
    -------
    pd.DataFrame
        BIDS-compatible blocks table: onset, duration, trial_type
        (+ any keep_cols).  Sorted by onset.
    """
    if events.empty:
        return events.copy()

    required = {onset_col, duration_col, condition_col}
    missing  = required - set(events.columns)
    if missing:
        raise ValueError(f"events DataFrame is missing columns: {missing}")

    if strategy not in ("gap", "adjacent"):
        raise ValueError(f"strategy must be 'gap' or 'adjacent', got {strategy!r}")

    df = (
        events
        .copy()
        .sort_values(onset_col)
        .reset_index(drop=True)
    )

    extra_cols = keep_cols or []

    blocks = []

    # ── Group trials into blocks ──────────────────────────────────────────────
    # Walk through rows; start a new block whenever the condition changes
    # OR the gap to the previous trial exceeds gap_threshold.

    block_start_idx = 0

    for i in range(1, len(df) + 1):
        # Sentinel: treat end of file as always starting a new block
        new_block = (i == len(df))

        if not new_block:
            same_cond = (
                df.loc[i, condition_col] == df.loc[i - 1, condition_col]
            )
            prev_end  = (
                df.loc[i - 1, onset_col] + df.loc[i - 1, duration_col]
            )
            gap       = df.loc[i, onset_col] - prev_end

            if strategy == "gap":
                new_block = not same_cond or gap > gap_threshold
            else:  # adjacent
                new_block = not same_cond or gap > 0

        if new_block:
            chunk = df.iloc[block_start_idx:i]
            block = _summarise_block(
                chunk, onset_col, duration_col, condition_col, extra_cols
            )
            blocks.append(block)
            block_start_idx = i

    result = pd.DataFrame(blocks)

    # Enforce BIDS column order; n_trials always appended after extra_cols
    base_cols = [onset_col, duration_col, condition_col]
    col_order = base_cols + [c for c in extra_cols if c in result.columns] + ["n_trials"]
    result = result[col_order].sort_values(onset_col).reset_index(drop=True)

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Helper — summarise one block
# ─────────────────────────────────────────────────────────────────────────────

def _summarise_block(
    chunk: pd.DataFrame,
    onset_col: str,
    duration_col: str,
    condition_col: str,
    extra_cols: list,
) -> dict:
    """Collapse a group of same-condition trials into a single block row."""
    first = chunk.iloc[0]
    last  = chunk.iloc[-1]

    onset    = first[onset_col]
    end_time = last[onset_col] + last[duration_col]
    duration = end_time - onset

    row = {
        onset_col:     onset,
        duration_col:  round(duration, 4),
        condition_col: first[condition_col],
        "n_trials":    len(chunk),
    }

    for col in extra_cols:
        if col not in chunk.columns:
            continue
        unique_vals = chunk[col].dropna().unique()
        row[col] = unique_vals[0] if len(unique_vals) == 1 else "|".join(
            str(v) for v in unique_vals
        )

    return row


# ─────────────────────────────────────────────────────────────────────────────
# Convenience: load TSV → blocks in one call
# ─────────────────────────────────────────────────────────────────────────────

def tsv_to_blocks(
    tsv_path: str,
    gap_threshold: float = 2.0,
    **kwargs,
) -> pd.DataFrame:
    """
    Load a BIDS events TSV and return block-level regressors.

    Parameters
    ----------
    tsv_path : str
        Path to the *_events.tsv file.
    gap_threshold : float
        Maximum within-block gap (seconds).
    **kwargs
        Passed to events_to_blocks().

    Returns
    -------
    pd.DataFrame
        Block-level events table.
    """
    events = pd.read_csv(tsv_path, sep="\t")
    return events_to_blocks(events, gap_threshold=gap_threshold, **kwargs)


# ─────────────────────────────────────────────────────────────────────────────
# Convenience: build a nilearn-compatible design matrix from blocks
# ─────────────────────────────────────────────────────────────────────────────

def blocks_to_design_matrix(
    blocks: pd.DataFrame,
    frame_times: np.ndarray,
    hrf_model: str = "spm",
    drift_model: str = "cosine",
    high_pass: float = 1 / 128,
    condition_col: str = "trial_type",
    onset_col: str = "onset",
    duration_col: str = "duration",
):
    """
    Build a first-level design matrix from block-level regressors.

    Requires nilearn >= 0.10.

    Parameters
    ----------
    blocks : pd.DataFrame
        Output of events_to_blocks().
    frame_times : np.ndarray
        TR onset times in seconds: np.arange(n_scans) * TR
    hrf_model : str
        HRF model passed to nilearn (e.g. "spm", "glover").
    drift_model : str
        Drift model ("cosine", "polynomial", None).
    high_pass : float
        High-pass cutoff in Hz (default 1/128 s = 0.0078 Hz).
    condition_col, onset_col, duration_col : str
        Column names.

    Returns
    -------
    pd.DataFrame
        Design matrix (time × regressors).
    """
    from nilearn.glm.first_level import make_first_level_design_matrix

    # nilearn expects columns: onset, duration, trial_type
    events_nl = blocks.rename(columns={
        onset_col:     "onset",
        duration_col:  "duration",
        condition_col: "trial_type",
    })[["onset", "duration", "trial_type"]]

    dm = make_first_level_design_matrix(
        frame_times=frame_times,
        events=events_nl,
        hrf_model=hrf_model,
        drift_model=drift_model,
        high_pass=high_pass,
    )
    return dm
