#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
cd "${REPO_DIR}"

OUT_DIR="${OUT_DIR:-${REPO_DIR}/tmp/pearl_ibverbs_one_case}"
mkdir -p "${OUT_DIR}"

TARGET_MODEL="${TARGET_MODEL:-/scratch.trt_llm_data/llm-models/llama-3.1-model/Meta-Llama-3.1-70B-Instruct}"
DRAFT_MODEL="${DRAFT_MODEL:-/scratch.trt_llm_data/llm-models/llama-3.1-model/Llama-3.1-8B-Instruct}"
TARGET_VISIBLE_GPUS="${TARGET_VISIBLE_GPUS:-6}"
DRAFT_HOST="${DRAFT_HOST:-127.0.0.1}"
DRAFT_CONTROL_PORT="${DRAFT_CONTROL_PORT:-47331}"
DRAFT_PORT="${DRAFT_PORT:-0}"
TRANSPORT="${TRANSPORT:-ibverbs}"
NIC="${NIC:-mlx5_0}"
SHM_NAME="${SHM_NAME:-pearl_shm_default}"
TP_SIZE="${TP_SIZE:-1}"
MAX_DRAFT_LEN="${MAX_DRAFT_LEN:-5}"
MAX_TOKENS="${MAX_TOKENS:-1024}"
WARMUP_TOKENS="${WARMUP_TOKENS:-${MAX_TOKENS}}"
KV_CACHE_FREE_FRACTION="${KV_CACHE_FREE_FRACTION:-0.25}"
RESPONSE_TIMEOUT_S="${RESPONSE_TIMEOUT_S:-30}"
PROMPT="${PROMPT:-请讲一个约5000字的中文故事，要求情节完整、人物清楚、语言自然。}"
CUDA_GRAPH="${CUDA_GRAPH:-0}"
TRACE="${TRACE:-1}"
NSYS="${NSYS:-0}"
NSYS_OUT="${NSYS_OUT:-${OUT_DIR}/pearl_phase2}"

TARGET_TRACE_LOG="${TARGET_TRACE_LOG:-${OUT_DIR}/target_trace.jsonl}"
DRAFT_TRACE_LOG="${DRAFT_TRACE_LOG:-${OUT_DIR}/draft_trace.jsonl}"
TARGET_LOG="${TARGET_LOG:-${OUT_DIR}/target.log}"

export PYTHONPATH="${PYTHONPATH:-${REPO_DIR}}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${TARGET_VISIBLE_GPUS}}"
export GPU_ID="${GPU_ID:-0}"
export TLLM_RDMA_DRAFT_OFFLOAD_RESPONSE_TIMEOUT_S="${TLLM_RDMA_DRAFT_OFFLOAD_RESPONSE_TIMEOUT_S:-${RESPONSE_TIMEOUT_S}}"

# Keep the target trace/log scoped to this run. The draft server may be a
# long-lived process, so its trace is filtered by target run_start/run_finish
# in the summary below instead of being truncated here.
if [[ "${TRACE}" == "1" || "${TRACE}" == "true" ]]; then
  : > "${TARGET_TRACE_LOG}"
fi
: > "${TARGET_LOG}"

echo "===== PEARL target one case ====="
echo "repo: ${REPO_DIR}"
echo "out_dir: ${OUT_DIR}"
echo "target_model: ${TARGET_MODEL}"
echo "draft_model: ${DRAFT_MODEL}"
echo "gpu: ${CUDA_VISIBLE_DEVICES}"
echo "transport: ${TRANSPORT}"
echo "nic: ${NIC}"
echo "tp_size: ${TP_SIZE}"
echo "max_draft_len: ${MAX_DRAFT_LEN}"
echo "max_tokens: ${MAX_TOKENS}"
echo "warmup_tokens: ${WARMUP_TOKENS}"
echo "response_timeout_s: ${TLLM_RDMA_DRAFT_OFFLOAD_RESPONSE_TIMEOUT_S}"
echo "cuda_graph: ${CUDA_GRAPH}"
echo "trace: ${TRACE}"
echo "nsys: ${NSYS}"

EXTRA_ARGS=()
if [[ "${CUDA_GRAPH}" == "1" || "${CUDA_GRAPH}" == "true" ]]; then
  EXTRA_ARGS+=(--cuda-graph)
fi
if [[ "${TRACE}" == "1" || "${TRACE}" == "true" ]]; then
  EXTRA_ARGS+=(--trace-log "${TARGET_TRACE_LOG}")
fi
if [[ -n "${ATTN_BACKEND:-}" ]]; then
  EXTRA_ARGS+=(--attn-backend "${ATTN_BACKEND}")
fi
if [[ "${TRANSPORT}" == "shm" ]]; then
  EXTRA_ARGS+=(--shm-name "${SHM_NAME}")
fi

LAUNCHER=()
if [[ "${NSYS}" == "1" || "${NSYS}" == "true" ]]; then
  # cudaProfilerStart in the launcher process does NOT cover MPI worker
  # processes (workers spawn from a different proc tree and never see the
  # API call). Use a time-window capture instead: skip the long setup
  # (~5 min for 70B model load + CUDA graph capture + app warmup) and
  # record ~25s spanning the measured run.
  LAUNCHER=(nsys profile
    -t cuda,nvtx,osrt
    -o "${NSYS_OUT}"
    --delay=${NSYS_DELAY:-280}
    --duration=${NSYS_DURATION:-30}
    --capture-range=none
    --kill=none
    --force-overwrite=true)
  echo "nsys_out: ${NSYS_OUT}.nsys-rep (delay=${NSYS_DELAY:-280}s duration=${NSYS_DURATION:-30}s)"
fi

"${LAUNCHER[@]}" python3 examples/llm-api/rdma/spec_dec_pearl_target_main.py \
  --target-model "${TARGET_MODEL}" \
  --draft-model "${DRAFT_MODEL}" \
  --draft-host "${DRAFT_HOST}" \
  --draft-control-port "${DRAFT_CONTROL_PORT}" \
  --draft-port "${DRAFT_PORT}" \
  --transport "${TRANSPORT}" \
  --nic "${NIC}" \
  --tp-size "${TP_SIZE}" \
  --max-draft-len "${MAX_DRAFT_LEN}" \
  --no-adaptive-gamma \
  --max-tokens "${MAX_TOKENS}" \
  --warmup-tokens "${WARMUP_TOKENS}" \
  --kv-cache-free-fraction "${KV_CACHE_FREE_FRACTION}" \
  --prompt "${PROMPT}" \
  "${EXTRA_ARGS[@]}" \
  2>&1 | tee -a "${TARGET_LOG}"

OUT_DIR="${OUT_DIR}" \
MAX_DRAFT_LEN="${MAX_DRAFT_LEN}" \
TARGET_TRACE_LOG="${TARGET_TRACE_LOG}" \
DRAFT_TRACE_LOG="${DRAFT_TRACE_LOG}" \
TARGET_LOG="${TARGET_LOG}" \
python3 - <<'PY'
import json
import os
import re
from pathlib import Path

out_dir = Path(os.environ["OUT_DIR"])
max_draft_len = int(os.environ.get("MAX_DRAFT_LEN", "0") or 0)
target_trace_path = (
    Path(os.environ["TARGET_TRACE_LOG"])
    if os.environ.get("TARGET_TRACE_LOG")
    else out_dir / "target_trace.jsonl"
)
draft_trace_path = (
    Path(os.environ["DRAFT_TRACE_LOG"])
    if os.environ.get("DRAFT_TRACE_LOG")
    else out_dir / "draft_trace.jsonl"
)
target_log_path = (
    Path(os.environ["TARGET_LOG"]) if os.environ.get("TARGET_LOG") else out_dir / "target.log"
)


def read_jsonl(path):
    records = []
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def token_count(value):
    if value is None:
        return 0
    try:
        return len(list(value))
    except TypeError:
        return 0


target_records = read_jsonl(target_trace_path)
draft_records_all = read_jsonl(draft_trace_path)

run_starts = [r for r in target_records if r.get("event") == "run_start"]
run_finishes = [r for r in target_records if r.get("event") == "run_finish"]
run_start_ns = int(run_starts[-1].get("time_ns", 0)) if run_starts else 0
run_finish_ns = int(run_finishes[-1].get("time_ns", 0)) if run_finishes else 0

if run_start_ns and run_finish_ns:
    draft_records = [
        r
        for r in draft_records_all
        if run_start_ns <= int(r.get("time_ns", 0)) <= run_finish_ns
    ]
else:
    draft_records = draft_records_all

normal_generated = 0
prefetch_generated = 0
normal_sent = 0
prefetch_sent = 0
send_events = 0
prefetch_send_events = 0
pearl_hit_events = 0

target_accepted_draft_tokens = sum(
    max(0, int(record.get("accepted_token_count") or 0))
    for record in target_records
    if record.get("event") == "send_target_to_draft"
    and run_start_ns <= int(record.get("time_ns", 0)) <= run_finish_ns
)

for record in draft_records:
    event = record.get("event")
    request_id = int(record.get("request_id", -1))

    if event == "prompt_session_init":
        normal_generated += 1 if record.get("preverify_token") is not None else 0

    elif event == "recv_target_to_draft":
        pass

    elif event == "backend_step":
        generated = token_count(record.get("generated_draft_tokens"))
        runner = str(record.get("runner", ""))
        if "prefetch" in runner:
            prefetch_generated += generated
        else:
            normal_generated += generated

    elif event == "prefetch_computed":
        # backend_step from the branch path already accounts for these tokens.
        pass

    elif event == "prefetch_cache_hit":
        pearl_hit_events += 1

    elif event == "send_draft_to_target":
        sent = token_count(record.get("draft_tokens"))
        send_events += 1
        if record.get("compute_source") == "prefetch_cache":
            prefetch_sent += sent
            prefetch_send_events += 1
        else:
            normal_sent += sent

generated_total = normal_generated + prefetch_generated
sent_total = normal_sent + prefetch_sent
generated_not_sent = max(0, generated_total - sent_total)

target_tokens_per_sec = None
if run_finishes:
    try:
        target_tokens_per_sec = float(run_finishes[-1].get("tokens_per_sec"))
    except (TypeError, ValueError):
        target_tokens_per_sec = None
if target_tokens_per_sec is None and target_log_path.exists():
    text = target_log_path.read_text(encoding="utf-8", errors="replace")
    matches = re.findall(r"tokens_per_sec:\s*([0-9]+(?:\.[0-9]+)?)", text)
    if matches:
        target_tokens_per_sec = float(matches[-1])


def rate(num, den):
    return (float(num) / float(den)) if den else 0.0


print("=== PEARL draft summary ===")
print(f"max_draft_len: {max_draft_len}")
print(f"draft_generated_tokens_total: {generated_total}")
print(f"  normal_generated_tokens: {normal_generated}")
print(f"  prefetch_generated_tokens: {prefetch_generated}")
print(f"draft_sent_tokens_to_target: {sent_total}")
print(f"  normal_sent_tokens: {normal_sent}")
print(f"  prefetch_sent_tokens: {prefetch_sent}")
print(f"draft_generated_not_sent_tokens: {generated_not_sent}")
print(f"target_accepted_draft_tokens: {target_accepted_draft_tokens}")
print(
    "total_accept_rate: "
    f"{rate(target_accepted_draft_tokens, generated_total):.4f} "
    f"({target_accepted_draft_tokens}/{generated_total})"
)
print(
    "target_measured_accept_rate: "
    f"{rate(target_accepted_draft_tokens, sent_total):.4f} "
    f"({target_accepted_draft_tokens}/{sent_total})"
)
print(f"draft_send_rate: {rate(sent_total, generated_total):.4f} ({sent_total}/{generated_total})")
print(
    "pearl_token_reuse_rate: "
    f"{rate(prefetch_sent, generated_total):.4f} "
    f"({prefetch_sent}/{generated_total})"
)
print(
    "pearl_event_hit_rate: "
    f"{rate(prefetch_send_events, send_events):.4f} "
    f"({prefetch_send_events}/{send_events})"
)
if target_tokens_per_sec is not None:
    print(f"target_tokens_per_sec: {target_tokens_per_sec:.2f}")
else:
    print("target_tokens_per_sec: <not found>")
PY
