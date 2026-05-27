#!/bin/bash
#SBATCH --job-name=beta_est
#SBATCH --array=1-6
#SBATCH --time=06:00:00              # LSS on 12 runs × ~100 trials/condition is slow
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8            # LSS uses parallel jobs
#SBATCH --mem=32G                    # beta images accumulate in memory
#SBATCH --qos=normal
#SBATCH --partition=amilan
#SBATCH --account=ucb-general
#SBATCH --output=logs/beta_%A_%a.out
#SBATCH --error=logs/beta_%A_%a.err

module purge
module load anaconda
conda activate incenv2

SUBJECT=$(sed -n "${SLURM_ARRAY_TASK_ID}p" subjects.txt)
echo "=============================================="
echo "  SLURM array task : ${SLURM_ARRAY_TASK_ID}"
echo "  Subject          : ${SUBJECT}"
echo "  Node             : $(hostname)"
echo "  Start            : $(date)"
echo "=============================================="

SCRIPT_DIR="${SLURM_SUBMIT_DIR}"
cd "${SCRIPT_DIR}" || exit 1
mkdir -p logs

python 02_beta_estimate.py \
    --subject  "${SUBJECT}" \
    --method   both \
    --strategy hmp24 \
    --n-jobs   8 \
    --n-runs   12 \
    --tr       2.5 \
    --hrf      spm

EXIT_CODE=$?
echo "Finished: ${SUBJECT}  exit_code=${EXIT_CODE}  $(date)"
exit ${EXIT_CODE}
