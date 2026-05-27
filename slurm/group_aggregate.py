#!/usr/bin/env python3
"""
group_aggregate.py — Aggregate per-subject CSV outputs into group-level tables.

Usage:
    python group_aggregate.py               # aggregate all analyses found
    python group_aggregate.py --analysis all_conditions_lss

Reads from:
  derivatives/mvpa/<analysis_name>/<subject>/accuracy_metrics.csv
  derivatives/mvpa/<analysis_name>/<subject>/confusion_matrix_normalised.csv
  derivatives/mvpa/<analysis_name>/<subject>/auc_per_condition.csv
  derivatives/mvpa/<analysis_name>/<subject>/searchlight_summary.csv  (if present)

Writes to:
  derivatives/mvpa/group/<analysis_name>_accuracy_metrics.csv
  derivatives/mvpa/group/<analysis_name>_confusion_matrix.csv
  derivatives/mvpa/group/<analysis_name>_auc.csv
  derivatives/mvpa/group/<analysis_name>_searchlight_summary.csv  (if present)
"""
import argparse, glob, os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DERIV_ROOT = "/pl/active/courses/2026_summer/neuroclass2026/ds000105/derivatives"
MVPA_ROOT  = os.path.join(DERIV_ROOT, "mvpa")

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--analysis", default=None,
                   help="Analysis name (default: discover all found directories)")
    p.add_argument("--mvpa-root", default=MVPA_ROOT)
    return p.parse_args()

def discover_analyses(mvpa_root):
    """Find all analysis_name directories that contain per-subject subfolders."""
    analyses = []
    for d in sorted(os.listdir(mvpa_root)):
        full = os.path.join(mvpa_root, d)
        if os.path.isdir(full) and d != "group":
            sub_dirs = [s for s in os.listdir(full)
                        if s.startswith("sub-") and os.path.isdir(os.path.join(full, s))]
            if sub_dirs:
                analyses.append(d)
    return analyses

def aggregate_analysis(analysis_name, mvpa_root, out_dir):
    analysis_dir = os.path.join(mvpa_root, analysis_name)
    sub_dirs = sorted([
        d for d in os.listdir(analysis_dir)
        if d.startswith("sub-") and os.path.isdir(os.path.join(analysis_dir, d))
    ])
    if not sub_dirs:
        print(f"  {analysis_name}: no subject directories found — skipping"); return

    print(f"\n  Analysis: {analysis_name}  ({len(sub_dirs)} subjects)")

    # ── Accuracy metrics ──────────────────────────────────────────────────────
    acc_dfs = []
    for sub in sub_dirs:
        csv = os.path.join(analysis_dir, sub, 'accuracy_metrics.csv')
        if os.path.exists(csv):
            acc_dfs.append(pd.read_csv(csv))
    if acc_dfs:
        acc_group = pd.concat(acc_dfs, ignore_index=True)
        out_path  = os.path.join(out_dir, f"{analysis_name}_accuracy_metrics.csv")
        acc_group.to_csv(out_path, index=False)
        print(f"    accuracy_metrics   : {len(acc_group)} rows → {out_path}")
        # Quick group summary
        summary = acc_group[['mean_accuracy','balanced_accuracy']].agg(['mean','std'])
        print(f"    Group mean accuracy: {summary.loc['mean','mean_accuracy']*100:.2f}%"
              f" ± {summary.loc['std','mean_accuracy']*100:.2f}% SD")

    # ── Confusion matrix (average across subjects) ─────────────────────────────
    cm_dfs = []
    for sub in sub_dirs:
        csv = os.path.join(analysis_dir, sub, 'confusion_matrix_normalised.csv')
        if os.path.exists(csv):
            df = pd.read_csv(csv, index_col='true_condition')
            cond_cols = [c for c in df.columns if c not in ('subject','analysis_name')]
            cm_dfs.append(df[cond_cols].values.astype(float))
    if cm_dfs:
        cm_avg  = np.mean(cm_dfs, axis=0)
        cond_cols_final = [c for c in pd.read_csv(
            os.path.join(analysis_dir, sub_dirs[0], 'confusion_matrix_normalised.csv')
        ).columns if c not in ('subject','analysis_name','true_condition')]
        cm_avg_df = pd.DataFrame(cm_avg, index=cond_cols_final, columns=cond_cols_final)
        out_path  = os.path.join(out_dir, f"{analysis_name}_confusion_matrix_avg.csv")
        cm_avg_df.to_csv(out_path)
        print(f"    confusion matrix   : avg over {len(cm_dfs)} subjects → {out_path}")
        # Plot
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(cm_avg, cmap='Blues', vmin=0, vmax=1)
        ax.set_xticks(range(len(cond_cols_final))); ax.set_xticklabels(cond_cols_final, rotation=45, ha='right')
        ax.set_yticks(range(len(cond_cols_final))); ax.set_yticklabels(cond_cols_final)
        for i in range(len(cond_cols_final)):
            for j in range(len(cond_cols_final)):
                ax.text(j, i, f'{cm_avg[i,j]:.2f}', ha='center', va='center', fontsize=8,
                        color='white' if cm_avg[i,j] > 0.5 else 'black')
        plt.colorbar(im, ax=ax)
        ax.set_xlabel('Predicted'); ax.set_ylabel('True')
        ax.set_title(f'{analysis_name} — Group average confusion matrix (n={len(cm_dfs)})')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{analysis_name}_confusion_matrix_avg.png"), dpi=120)
        plt.close()

    # ── AUC ───────────────────────────────────────────────────────────────────
    auc_dfs = []
    for sub in sub_dirs:
        csv = os.path.join(analysis_dir, sub, 'auc_per_condition.csv')
        if os.path.exists(csv):
            auc_dfs.append(pd.read_csv(csv))
    if auc_dfs:
        auc_group = pd.concat(auc_dfs, ignore_index=True)
        out_path  = os.path.join(out_dir, f"{analysis_name}_auc.csv")
        auc_group.to_csv(out_path, index=False)
        print(f"    auc_per_condition  : {len(auc_group)} rows → {out_path}")

    # ── Searchlight summary ────────────────────────────────────────────────────
    sl_dfs = []
    for sub in sub_dirs:
        csv = os.path.join(analysis_dir, sub, 'searchlight_summary.csv')
        if os.path.exists(csv):
            sl_dfs.append(pd.read_csv(csv))
    if sl_dfs:
        sl_group = pd.concat(sl_dfs, ignore_index=True)
        out_path = os.path.join(out_dir, f"{analysis_name}_searchlight_summary.csv")
        sl_group.to_csv(out_path, index=False)
        print(f"    searchlight summary: {len(sl_group)} rows → {out_path}")

def main():
    args = parse_args()
    out_dir = os.path.join(args.mvpa_root, "group")
    os.makedirs(out_dir, exist_ok=True)
    if args.analysis:
        analyses = [args.analysis]
    else:
        analyses = discover_analyses(args.mvpa_root)
        print(f"Found {len(analyses)} analysis/analyses: {analyses}")
    for a in analyses:
        aggregate_analysis(a, args.mvpa_root, out_dir)
    print(f"\nGroup outputs → {out_dir}/")

if __name__ == "__main__":
    main()
