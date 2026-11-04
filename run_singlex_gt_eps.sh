# Re-run of the GT single-example bars with eps_low=0.2, eps_high=0.28. The originals from
# run_singlex_rewards.sh all used eps_low=1, eps_high=0.1. MBPP, GSM8K and Countdown only, 9 runs
# in series, everything else as the originals: seed 0, 600 steps, eval checkpoint-600 at n=32.
# Runs get new names, so the originals stay in outputs/ and results/ untouched:
#
#   outputs/<ds>_<model>_gt_singlex_low0.2_high0.28, result label <that name>-600
#
# No "=" in the names: eval.py splits --models entries on the first "=".
#
# GPUs: waits for GPU 0 or 2. A GPU counts as free once it has stayed under FREE_MIB for CONFIRM
# seconds (a job stopping can just be a gap between stages). Before each train and eval it checks
# the GPU it holds is still free, and if someone else took it, goes back to watching both.
#
# Launch:  screen -dmS singlex_gt_eps bash -c 'source venv/bin/activate; bash run_singlex_gt_eps.sh; exec bash'
# Progress: cat logs/singlex_gt_eps/STATUS ; per-run logs in logs/singlex_gt_eps/<run>.{train,eval}.log
# No `set -e`: a failed run is logged and skipped. Relaunching redoes only what is missing, since
# finished checkpoints are skipped and eval.py reuses stored results.

# env (HF/W&B tokens, caches) from the original sweep script
source <(grep '^export ' /local/scratch/ae581/archived_scripts/run_singlex.sh)

set -uo pipefail
cd /local/scratch/ae581/zzsp

GPUS=(0 2)
FREE_MIB=2000
POLL=60
CONFIRM=600
SETTLE=30   # lets our own finished process release its memory before checking the GPU
STEPS=600
N=32

declare -A MODELS=(
    [qwen]=Qwen/Qwen2.5-1.5B-Instruct
    [llama]=meta-llama/Llama-3.2-1B-Instruct
    [olmo]=allenai/OLMo-2-0425-1B-Instruct
)

LOGS=logs/singlex_gt_eps
mkdir -p $LOGS
STATUS=$LOGS/STATUS
GPU=""
log() { echo "$(date '+%a %H:%M') $*${GPU:+ [GPU $GPU]}" | tee -a $STATUS; }

gpu_free() {
    local used
    used=$(nvidia-smi -i $1 --query-gpu=memory.used --format=csv,noheader,nounits) && [ "$used" -le $FREE_MIB ]
}

# sets GPU to the first of GPUS to stay free for CONFIRM seconds
acquire_gpu() {
    local g now
    local -A since=()
    log "watching GPUs ${GPUS[*]}"
    while true; do
        now=$(date +%s)
        for g in "${GPUS[@]}"; do
            if ! gpu_free $g; then
                if [ -n "${since[$g]:-}" ]; then
                    log "GPU $g busy again, not available"
                    unset "since[$g]"
                fi
            elif [ -z "${since[$g]:-}" ]; then
                log "GPU $g looks free, confirming over ${CONFIRM}s"
                since[$g]=$now
            elif [ $((now - since[$g])) -ge $CONFIRM ]; then
                GPU=$g
                log "GPU $g confirmed free"
                return
            fi
        done
        sleep $POLL
    done
}

# call before anything that needs the GPU
ensure_gpu() {
    if [ -n "$GPU" ]; then
        sleep $SETTLE
        gpu_free $GPU && return
        log "GPU in use by another job, going back to watching"
        GPU=""
    fi
    acquire_gpu
}

# train_eval <dataset> <model key>
train_eval() {
    local ds=$1 m=$2
    local run=${ds}_${m}_gt_singlex_low0.2_high0.28
    local out=outputs/$run

    if [ -d $out/checkpoint-$STEPS ]; then
        log "SKIP train $run (checkpoint-$STEPS exists)"
    else
        ensure_gpu
        log "START train $run"
        if ! CUDA_VISIBLE_DEVICES=$GPU PYTORCH_ALLOC_CONF=expandable_segments:True python trainer.py \
                --max_steps $STEPS --seed 0 --reward $ds --dataset $ds --model ${MODELS[$m]} \
                --eps_low 0.2 --eps_high 0.28 --single_example --run_name $run --save_step $STEPS \
                > $LOGS/$run.train.log 2>&1; then
            log "FAIL train $run (see $LOGS/$run.train.log)"
            return
        fi
        log "DONE train $run"
    fi

    ensure_gpu
    log "START eval $run"
    if CUDA_VISIBLE_DEVICES=$GPU python eval.py --n $N --dataset $ds \
            --models $run-$STEPS=$out/checkpoint-$STEPS \
            > $LOGS/$run.eval.log 2>&1; then
        log "DONE eval $run: $(grep -E "^\[$run-$STEPS\] accuracy" $LOGS/$run.eval.log | tr '\n' ' ')"
    else
        log "FAIL eval $run (see $LOGS/$run.eval.log)"
    fi
}

log "===== GT eps 0.2/0.28 sweep start ====="
for ds in mbpp gsm8k countdown; do
    for m in qwen llama olmo; do train_eval $ds $m; done
done
log "===== GT eps 0.2/0.28 sweep end ====="
