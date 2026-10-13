#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Kimi K3 GSM8K over HTTP against an AGGREGATED trtllm-serve deployment.
#
# WHY THIS EXISTS. Every other K3 accuracy job runs `trtllm-eval` in-process
# against the LLM API. The DEP8 disaggregated runs could not: the disagg
# harness drives the proxy with `lm_eval --model local-completions` over HTTP.
# Those two paths are not interchangeable. This script serves the SAME
# config file the in-process run used, so graphs / overlap / chunked prefill /
# batch size / KV fraction are identical, then evaluates it with the SAME
# lm_eval command line the disagg harness generates. The remaining difference
# is the evaluation path itself.
#
# This is a control, not a benchmark. Do not read throughput off it.
#
# Submit from the repository root of a built TensorRT-LLM checkout. MegaMoE
# uses MNNVL symmetric memory, so ALL nodes must sit in one NVL72 domain --
# submit via k3sub.sh (which pins a single rack) rather than bare sbatch:
#
#   PART=batch NODES=2 TIME=3:00:00 JOB=k3-gsm8k-served \
#   EVAL_CONFIG=$REPO/examples/kimi_k3/eval_extra_llm_options_nvfp4_dep8_megamoe.yaml \
#   TP_SIZE=8 \
#   ./k3sub.sh --export=ALL examples/kimi_k3/run_gsm8k_served_kimi_k3.sbatch \
#       --model /path/to/kimi-k3-checkpoint --image /path/to/container.sqsh
#
# HF_HOME_EVAL must contain a Hugging Face cache with the gsm8k dataset
# already downloaded (hub/ + datasets/): compute nodes here have no egress,
# and lm_eval resolves the task offline. Defaults to $REPO/hf_home_eval,
# populated with:
#     cp -r ~/.cache/huggingface/{hub,datasets,modules} $REPO/hf_home_eval/
#
#SBATCH --job-name=kimi-k3-gsm8k-served
#SBATCH --partition=batch
#SBATCH --account=${account}
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --time=03:00:00
#SBATCH --output=kimi-k3-gsm8k-served-%j.log
#SBATCH --comment='{"OccupiedIdleGPUsJobReaper":{"exemptIdleTimeMins":"60","reason":"benchmarking","description":"Kimi K3 DEP8 server initialization and weight loading takes 20-30 minutes before evaluation traffic starts"}}'

set -euo pipefail

usage() {
    echo "Usage: sbatch $0 --model PATH --image PATH"
}

MODEL=""
CONTAINER_IMAGE=""
PORT=${PORT:-8000}
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model|--image)
            [[ $# -ge 2 ]] || { echo "error: $1 requires a value" >&2; usage >&2; exit 2; }
            VALUE=$2; OPT=$1; shift 2 ;;
        --model=*|--image=*)
            VALUE=${1#*=}; OPT=${1%%=*}; shift ;;
        -h|--help)
            usage; exit 0 ;;
        *)
            echo "error: unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
    case "$OPT" in
        --model) MODEL=$VALUE ;;
        --image) CONTAINER_IMAGE=$VALUE ;;
    esac
done

[[ -n "$MODEL" ]] || { echo "error: --model is required" >&2; usage >&2; exit 2; }
[[ -n "$CONTAINER_IMAGE" ]] || { echo "error: --image is required" >&2; usage >&2; exit 2; }
[[ -e "$MODEL" ]] || { echo "error: model path does not exist: $MODEL" >&2; exit 2; }
[[ -e "$CONTAINER_IMAGE" ]] || { echo "error: image path does not exist: $CONTAINER_IMAGE" >&2; exit 2; }

REPO=${REPO:-$SLURM_SUBMIT_DIR}
WORK_DIR=$SLURM_SUBMIT_DIR/kimi-k3-gsm8k-served-$SLURM_JOB_ID
mkdir -p "$WORK_DIR"

TP_SIZE=${TP_SIZE:-8}
MAX_BATCH_SIZE=${MAX_BATCH_SIZE:-8}
MAX_NUM_TOKENS=${MAX_NUM_TOKENS:-8192}
MAX_SEQ_LEN=${MAX_SEQ_LEN:-8192}

# The serving config is NOT written inline here on purpose. The whole point of
# this job is that the server is configured identically to the in-process
# trtllm-eval run it is being compared against, so it takes the same file
# rather than a copy that can drift.
EVAL_CONFIG=${EVAL_CONFIG:-$REPO/examples/kimi_k3/eval_extra_llm_options_nvfp4_dep8_megamoe.yaml}
[[ -e "$EVAL_CONFIG" ]] || { echo "error: EVAL_CONFIG does not exist: $EVAL_CONFIG" >&2; exit 2; }
cp "$EVAL_CONFIG" "$WORK_DIR/serve_config.used.yaml"

# 🔴 KV_FRAC is passed on the command line and OVERRIDES the yaml, exactly as
# run_gsm8k_kimi_k3.sbatch does. Sharing the yaml is not enough to make this
# job comparable to the in-process one, because that wrapper injects
# `--kv_cache_free_gpu_memory_fraction 0.35` while every DEP8 yaml still says
# 0.25. So the yaml is NOT the effective configuration -- reading it alone
# tells you the wrong KV fraction. 0.35 is the value that runs GSM8K to
# completion on this configuration.
KV_FRAC=${KV_FRAC:-0.35}

# Client-side in-flight request cap. 64 is the default because it is what
# submit.py generates for the disagg harness, so the default keeps this job a
# faithful copy of it.
#
# It is a knob because in-process and HTTP evaluation of the same served
# model are not interchangeable. Tokenization, stop-string handling,
# truncation, and sampling strategy were each checked. What remains is
# request scheduling. The server admits at most
# max_batch_size x dp_size (8 x 8 = 64) sequences, so raising this above 64
# cannot enlarge a batch -- but at exactly 64 the client must receive a
# response before issuing the next request, and that HTTP round-trip leaves
# the decode batch partially filled. trtllm-eval submits all 1319 requests up
# front and never has that bubble. Set 512 to keep the queue saturated and
# test whether a remaining gap is scheduling-induced.
NUM_CONCURRENT=${NUM_CONCURRENT:-64}

# 🔴 Few-shot seed alignment -- the reason this job's number is comparable to a
# trtllm-eval number at all. lm-eval's --seed is "python,numpy,torch,fewshot"
# and defaults to 0,1234,1234,1234; trtllm-eval calls
# set_fewshot_seed(seed=0). Different seeds mean different 5-shot
# demonstrations for the same question, and nothing in either output warns
# that the prompts differed. Only the 4th field is changed.
LM_EVAL_SEED=${LM_EVAL_SEED:-0,1234,1234,0}

# Offline HF cache for lm_eval's gsm8k. A missing dataset must fail loudly
# rather than silently attempt a hub fetch that cannot succeed here.
HF_HOME_EVAL=${HF_HOME_EVAL:-$REPO/hf_home_eval}
if [[ ! -d "$HF_HOME_EVAL/datasets" && ! -d "$HF_HOME_EVAL/hub" ]]; then
    echo "error: HF_HOME_EVAL=$HF_HOME_EVAL has no hub/ or datasets/ cache." >&2
    echo "       cp -r ~/.cache/huggingface/{hub,datasets,modules} $HF_HOME_EVAL/" >&2
    exit 2
fi

CACHE_DIR=${CACHE_DIR:-$SLURM_SUBMIT_DIR/.kimi_k3_cache}
mkdir -p "$CACHE_DIR"

# ⚠️ /lustre and /scratch are mounted wholesale IN ADDITION to $REPO, and this
# is not belt-and-braces. The venv console scripts (trtllm-serve, ...) bake
# their install-time PHYSICAL path into the shebang. On this cluster $REPO is
# usually given as the /lustre/... symlinked path while the venv was built at
# the /scratch/... physical path, so mounting only $REPO leaves the shebang
# interpreter missing and trtllm-serve dies with **exit 127**, ~90 s in, long
# before any weight is read. Job 493856 died exactly this way.
MOUNTS="$REPO:$REPO:rw,$MODEL:$MODEL:ro,$WORK_DIR:$WORK_DIR:rw,$CACHE_DIR:$CACHE_DIR:rw,/lustre:/lustre,/scratch:/scratch"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
HEAD_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | awk 'NR == 1')

# Forward the K3 online FP8 MLP-cast switch only when the submitter set it.
# Exporting an unset switch as "" would override its default-on behavior.
K3_FP8_SWITCHES="
KIMI_K3_FP8_WEIGHT_READ_MOE_MLP
"
K3_SUBSWITCH_EXPORTS=""
for _n in $K3_FP8_SWITCHES; do
    if [ -n "${!_n+set}" ]; then
        K3_SUBSWITCH_EXPORTS+="export $_n='${!_n}'; "
    fi
done
K3_FP8_SWITCH_NAMES=$(echo $K3_FP8_SWITCHES)

SERVER_PID=""
cleanup() { [[ -n "$SERVER_PID" ]] && kill "$SERVER_PID" 2>/dev/null || true; }
trap cleanup EXIT

srun --mpi=pmix --kill-on-bad-exit=1 \
    --container-image="$CONTAINER_IMAGE" \
    --container-mount-home \
    --container-mounts="$MOUNTS" \
    --output="$WORK_DIR/serve.log" \
    bash -c "
        set -x
        ulimit -n 65536

        export TRITON_CACHE_DIR=/tmp/triton-cache-rank\${SLURM_PROCID:-0}
        mkdir -p \"\$TRITON_CACHE_DIR\"
        export FLASHINFER_WORKSPACE_BASE=$CACHE_DIR/fi_workspace
        export FLASHINFER_CUBIN_DIR=$CACHE_DIR/fi_cubins
        export HF_HOME=\${HF_HOME:-$CACHE_DIR/hf_home}
        mkdir -p \"\$FLASHINFER_WORKSPACE_BASE\" \"\$FLASHINFER_CUBIN_DIR\" \"\$HF_HOME\"
        export HF_MODULES_CACHE=/tmp/hf-modules-rank\${SLURM_PROCID:-0}

        unset TRTLLM_FORCE_COMM_METHOD

        export PATH=\"$REPO/.venv-3.12/bin:\$PATH\"
        export PYTHONPATH=\"$REPO\${PYTHONPATH:+:\$PYTHONPATH}\"
        export TLLM_LOG_LEVEL=\${TLLM_LOG_LEVEL:-info}

        $K3_SUBSWITCH_EXPORTS
        for n in $K3_FP8_SWITCH_NAMES; do
            echo \"[k3] \$n=\${!n:-(default)}\"
        done

        exec '$REPO/tensorrt_llm/llmapi/trtllm-llmapi-launch' trtllm-serve \
            \"$MODEL\" \
            --backend pytorch \
            --host 0.0.0.0 \
            --port $PORT \
            --tp_size $TP_SIZE \
            --max_batch_size $MAX_BATCH_SIZE \
            --max_num_tokens $MAX_NUM_TOKENS \
            --max_seq_len $MAX_SEQ_LEN \
            --kv_cache_free_gpu_memory_fraction $KV_FRAC \
            --trust_remote_code \
            --config '$EVAL_CONFIG'
    " &
SERVER_PID=$!

# 50 min, not the 25 the serving benchmark uses, so weight load has room to
# finish.
deadline=$((SECONDS + 3000))
until curl -sf -o /dev/null "http://127.0.0.1:$PORT/health"; do
    # Report WHICH condition tripped. Collapsing "the server process died" into
    # "not healthy within 50 min" reads as a hang and sends the next reader
    # hunting a slow load that never happened.
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        wait "$SERVER_PID"; rc=$?
        echo "error: server process exited (rc=$rc) after ${SECONDS}s, before reporting healthy." >&2
        echo "       rc=127 almost always means the venv console-script shebang's" >&2
        echo "       interpreter is not mounted -- check MOUNTS covers the PHYSICAL" >&2
        echo "       path \$REPO resolves to. See $WORK_DIR/serve.log." >&2
        scancel "$SLURM_JOB_ID"; sleep 120; exit 1
    fi
    if (( SECONDS >= deadline )); then
        echo "error: server still not healthy after ${SECONDS}s (deadline 50 min); process is alive." >&2
        echo "       See $WORK_DIR/serve.log for load progress; cancelling job." >&2
        scancel "$SLURM_JOB_ID"; sleep 120; exit 1
    fi
    sleep 15
done
echo "Server is healthy; starting lm_eval (see $WORK_DIR/lm_eval.log)."

# The lm_eval invocation is a deliberate byte-for-byte match of what
# examples/disaggregated/slurm/benchmark/submit.py generates for a disagg
# config with an `accuracy:` (lm_eval GSM8K) block -- same model wrapper, same
# model_args, same flags. If these drift apart the comparison this job exists for is void.
srun --overlap --kill-on-bad-exit=1 --nodes=1 --ntasks=1 -w "$HEAD_NODE" \
    --container-image="$CONTAINER_IMAGE" \
    --no-container-mount-home \
    --container-mounts="$MOUNTS" \
    --output="$WORK_DIR/lm_eval.log" \
    bash -c "
        set -x
        export HF_HOME='$HF_HOME_EVAL'
        export HF_HUB_OFFLINE=1
        export HF_DATASETS_OFFLINE=1

        lm_eval --model local-completions --tasks gsm8k \
            --model_args model=$MODEL,base_url=http://$HEAD_NODE:$PORT/v1/completions,num_concurrent=$NUM_CONCURRENT,max_retries=3,tokenized_requests=false,timeout=1200,max_gen_toks=256,max_length=4096 \
            --seed $LM_EVAL_SEED \
            --log_samples --output_path '$WORK_DIR/accuracy_eval_gsm8k' \
            --trust_remote_code
    "

echo "GSM8K over HTTP complete; results in $WORK_DIR"
kill "$SERVER_PID" 2>/dev/null || true
wait "$SERVER_PID" 2>/dev/null || true
