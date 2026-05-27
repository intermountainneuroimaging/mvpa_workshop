#!/bin/bash
#SBATCH --job-name=mvpa
#SBATCH --array=1-6
#SBATCH --time=04:00:00              # ~1h CV + ~2h for 1000 permutations
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8            # permutation loop uses n_jobs=-1
#SBATCH --mem=16G
#SBATCH --qos=normal
#SBATCH --partition=amilan
#SBATCH --account=ucb-general
#SBATCH --output=logs/mvpa_%A_%a.out
#SBATCH --error=logs/mvpa_%A_%a.err

module purge
module load anaconda
conda activate incenv2

SUBJECT=$(sed -n "${SLURM_ARRAY_TASK_ID}p" subjects.txt)
ANALYSIS_NAME="all_conditions_lss"   # change to run a different analysis

echo "=============================================="
echo "  SLURM array task : ${SLURM_ARRAY_TASK_ID}"
echo "  Subject          : ${SUBJECT}"
echo "  Analysis         : ${ANALYSIS_NAME}"
echo "  Node             : $(hostname)"
echo "  Start            : $(date)"
echo "=============================================="

SCRIPT_DIR="${SLURM_SUBMIT_DIR}"
cd "${SCRIPT_DIR}" || exit 1
mkdir -p logs

N_PERMS=1000   # set to 0 to skip permutation test

python 03_mvpa.py \
    --subject          "${SUBJECT}" \
    --analysis-name    "${ANALYSIS_NAME}" \
    --beta-method      lss \
    --feature-sel      anova \
    --anova-pct        10 \
    --svm-c            1.0 \
    --svm-kernel       linear \
    --leave-one-run-out \
    --n-permutations   "${N_PERMS}" \
    --perm-seed        42

EXIT_CODE=$?
echo "Finished: ${SUBJECT}  exit_code=${EXIT_CODE}  $(date)"
exit ${EXIT_CODE}
