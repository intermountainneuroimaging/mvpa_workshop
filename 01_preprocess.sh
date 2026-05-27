#!/bin/bash
#SBATCH --job-name=fmri_preproc
#SBATCH --array=1-6                  # one task per subject; adjust to match subjects.txt
#SBATCH --time=02:00:00              # 2 h per subject is typically sufficient
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --qos=normal
#SBATCH --partition=amilan
#SBATCH --account=ucb-general
#SBATCH --output=logs/%x_%A_%a.out      # lands in submit dir; body moves later logs to logs/
#SBATCH --error=logs/%x_%A_%a.err

# ── Load environment ───────────────────────────────────────────────────────────
module purge
module load anaconda                 # or: source /path/to/venv/bin/activate
conda activate incenv        # change to your conda environment name

# ── Resolve subject from array task ID ────────────────────────────────────────
SUBJECT=$(sed -n "${SLURM_ARRAY_TASK_ID}p" subjects.txt)
echo "=============================================="
echo "  SLURM array task : ${SLURM_ARRAY_TASK_ID}"
echo "  Subject          : ${SUBJECT}"
echo "  Node             : $(hostname)"
echo "  Start            : $(date)"
echo "=============================================="

# ── Change to script directory ────────────────────────────────────────────────
# SLURM_SUBMIT_DIR is set by sbatch to the directory you ran sbatch from.
# BASH_SOURCE[0] resolves to the spool copy — do NOT use it.
SCRIPT_DIR="${SLURM_SUBMIT_DIR}"
cd "${SCRIPT_DIR}" || exit 1
mkdir -p logs

# ── Run pipeline ──────────────────────────────────────────────────────────────
python 01_preprocess.py \
    --subject   "${SUBJECT}" \
    --strategy  hmp24 \
    --poly-order 2 \
    --fd-thresh 0.5 \
    --n-runs    12 \
    --tr        2.5

EXIT_CODE=$?
echo "Finished: ${SUBJECT}  exit_code=${EXIT_CODE}  $(date)"
exit ${EXIT_CODE}
