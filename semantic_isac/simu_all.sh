#!/usr/bin/env bash
# run_batch_screen.sh -- launch several simulation scripts as separate detached `screen` sessions,
# auto-dividing this machine's CPU/GPU budget fairly across them.

set -euo pipefail

VENV_ACTIVATE="$HOME/venv/MAC-for-ISAC/bin/activate"
LOG_DIR="logs"
mkdir -p "$LOG_DIR"

# Take common arguments
args=()
for arg in "$@"; do
    args+=("--$arg")
done

# One entry per job: "session_name|command to run". Add/remove lines for
# whatever combination of scripts/configs you want running concurrently.
JOBS=(
    "MAF_vs_ue|python simulate_vs_ue.py --ra_method MAF"
    "SRI_vs_ue|python simulate_vs_ue.py --ra_method SRI"
    "SRL_vs_ue|python simulate_vs_ue.py --ra_method SRL"
    "PBI_vs_ue|python simulate_vs_ue.py --ra_method PBI"
    "QBC_vs_ue|python simulate_vs_ue.py --ra_method QBC"
)

NUM_SCRIPTS="${#JOBS[@]}"

# --- CPU budget: split available cores across the scripts being launched ---
detect_cpu_threads() {
    local total_cpus
    total_cpus=$(nproc 2>/dev/null || getconf _NPROCESSORS_ONLN 2>/dev/null || echo 1)
    local per_script=$(( total_cpus / NUM_SCRIPTS ))
    echo $(( per_script > 0 ? per_script : 1 ))
}
CPU_SLOTS=$(detect_cpu_threads)
echo "Detected $(nproc 2>/dev/null || echo '?') CPUs, ${NUM_SCRIPTS} scripts -> ${CPU_SLOTS} threads/script"

# --- GPU budget: how many ~2GB-sized jobs the card(s) can hold, split across scripts ---
detect_gpu_slots() {
    if ! command -v nvidia-smi &>/dev/null; then
        echo 0   # no GPU visible -- scripts should fall back to CPU-only
        return
    fi
    local total_vram_mib
    total_vram_mib=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | awk '{s+=$1} END {print s}')
    local total_slots=$(( total_vram_mib ))
    local per_script=$(( total_slots / NUM_SCRIPTS ))
    echo $(( per_script > 0 ? per_script : 1 ))
}
TOTAL_GPU_VRAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | awk '{s+=$1} END {print s}')
GPU_SLOTS=$(detect_gpu_slots)
if [ "$GPU_SLOTS" -eq 0 ]; then
    echo "No GPU detected -- scripts will run CPU-only."
else
    echo "GPU budget ${TOTAL_GPU_VRAM} -> ${GPU_SLOTS} max VRAM jobs/script"
fi

# Cycle through jobs
for job in "${JOBS[@]}"; do
    IFS='|' read -r name cmd <<< "$job"
    logfile="${LOG_DIR}/${name}.log"

    if screen -list | grep -q "\.${name}\b"; then
        echo "Session '${name}' already running -- skipping."
        continue
    fi

    full_cmd="${cmd} --concurrent ${NUM_SCRIPTS} ${args[*]}"
    echo "Launching '${name}': ${full_cmd}  [OMP/MKL/OPENBLAS/NUMEXPR=1]"

    screen -dmS "$name" bash -c "
        source '${VENV_ACTIVATE}'
        export OMP_NUM_THREADS=1
        export MKL_NUM_THREADS=1
        export OPENBLAS_NUM_THREADS=1
        export NUMEXPR_NUM_THREADS=1
        ${full_cmd} 2>&1 | tee -a '${logfile}'
    "
done

echo
echo "Launched. 'screen -ls' to list, 'screen -r <name>' to attach, Ctrl-A D to detach again."