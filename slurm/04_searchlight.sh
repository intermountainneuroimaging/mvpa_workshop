#!/bin/bash
#SBATCH --job-name=searchlight
#SBATCH --array=1-6
#SBATCH --time=24:00:00              # ~8h searchlight + ~15h for 100 permutations
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16           # n_jobs=-1 will use all allocated CPUs
#SBATCH --mem=32G
#SBATCH --qos=normal
#SBATCH --partition=amilan
#SBATCH --account=ucb-general
#SBATCH --output=logs/searchlight_%A_%a.out
#SBATCH --error=logs/searchlight_%A_%a.err

module purge
module load anaconda
conda activate incenv2

SUBJECT=$(sed -n "${SLURM_ARRAY_TASK_ID}p" subjects.txt)
ANALYSIS_NAME="searchlight_r6_lss"

echo "=============================================="
echo "  SLURM array task : ${SLURM_ARRAY_TASK_ID}"
echo "  Subject          : ${SUBJECT}"
echo "  Analysis         : ${ANALYSIS_NAME}"
echo "  CPUs allocated   : ${SLURM_CPUS_PER_TASK}"
echo "  Node             : $(hostname)"
echo "  Start            : $(date)"
echo "=============================================="

SCRIPT_DIR="${SLURM_SUBMIT_DIR}"
cd "${SCRIPT_DIR}" || exit 1
mkdir -p logs

N_PERMS=100   # set to 0 to skip; each permutation re-runs the full searchlight

python 04_searchlight.py \
    --subject          "${SUBJECT}" \
    --analysis-name    "${ANALYSIS_NAME}" \
    --beta-method      lss \
    --radius           6.0 \
    --svm-c            1.0 \
    --n-jobs           ${SLURM_CPUS_PER_TASK} \
    --leave-one-run-out \
    --top-n-peaks      20 \
    --n-permutations   "${N_PERMS}" \
    --perm-seed        42

EXIT_CODE=$?
echo "Finished: ${SUBJECT}  exit_code=${EXIT_CODE}  $(date)"
exit ${EXIT_CODE}
