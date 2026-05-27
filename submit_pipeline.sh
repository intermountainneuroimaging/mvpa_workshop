#!/bin/bash
# submit_pipeline.sh — Submit all four pipeline steps as dependent SLURM job arrays.
#
# Usage:
#   bash submit_pipeline.sh              # submit all four steps
#   bash submit_pipeline.sh --dry-run    # print sbatch commands without submitting
#
# Each step waits for the previous step to finish successfully for every array task
# before proceeding (afterok dependency). This guarantees correct ordering.
#
# Steps:
#   Step 1: Preprocessing       (~2 h/subject)
#   Step 2: Beta estimation     (~6 h/subject, LSS)
#   Step 3: MVPA decoding       (~1 h/subject)
#   Step 4: Searchlight         (~8 h/subject)

set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "${SCRIPT_DIR}"

DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=true
    echo "=== DRY RUN — no jobs will be submitted ==="
fi

# ── Count subjects ─────────────────────────────────────────────────────────────
N_SUBJECTS=$(wc -l < subjects.txt | tr -d ' ')
echo "Subjects file : subjects.txt  (${N_SUBJECTS} subjects)"

mkdir -p logs

submit() {
    local script="$1"
    local dep_opt="${2:-}"
    local array_range="1-${N_SUBJECTS}"

    if $DRY_RUN; then
        echo "  [DRY RUN] sbatch --array=${array_range} ${dep_opt} ${script}"
        echo "0"   # fake job ID
    else
        local jid
        # shellcheck disable=SC2086
        jid=$(sbatch --array="${array_range}" ${dep_opt} "${script}" | awk '{print $NF}')
        echo "${jid}"
    fi
}

echo ""
echo "Submitting Step 1: Preprocessing..."
JID1=$(submit 01_preprocess.sh)
echo "  Job ID: ${JID1}"

echo "Submitting Step 2: Beta estimation  (depends on Step 1)..."
JID2=$(submit 02_beta_estimate.sh "--dependency=afterok:${JID1}")
echo "  Job ID: ${JID2}"

echo "Submitting Step 3: MVPA decoding    (depends on Step 2)..."
JID3=$(submit 03_mvpa.sh "--dependency=afterok:${JID2}")
echo "  Job ID: ${JID3}"

echo "Submitting Step 4: Searchlight      (depends on Step 2)..."
JID4=$(submit 04_searchlight.sh "--dependency=afterok:${JID2}")
echo "  Job ID: ${JID4}"

echo ""
echo "=============================================="
echo "  Pipeline submitted successfully!"
echo "  Step 1 (preproc)     job array: ${JID1}"
echo "  Step 2 (beta est)    job array: ${JID2}"
echo "  Step 3 (MVPA)        job array: ${JID3}"
echo "  Step 4 (searchlight) job array: ${JID4}"
echo ""
echo "  Monitor with:  squeue --me"
echo "  Cancel all:    scancel ${JID1} ${JID2} ${JID3} ${JID4}"
echo "  Log files in:  ${SCRIPT_DIR}/logs/"
echo "=============================================="
