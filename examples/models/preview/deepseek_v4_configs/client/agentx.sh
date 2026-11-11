#!/bin/bash
# AgentX client of one point, as the measured runs ran it: aiperf from
# SemiAnalysisAI/agentx-harness@754356e9 (installed by client/install.sh into node-local /tmp unless
# CLIENT_VENV names a ready venv), scenario inferencex-agentx-mvp, public dataset
# semianalysis_cc_traces_weka_062126 (whole corpus).
# slurm/launch.sbatch runs it inside the TensorRT-LLM container. Its last line is
# "SUMMARY PASS|FAIL ...", and it exits non-zero on FAIL.
#   usage: agentx.sh <orchestrator host> <port> <model path> <concurrency> <output dir> <worker GPUs>
#   env:   CLIENT_VENV CLIENT_HF_HOME CLIENT_OFFLINE SMOKE UCX_WARMUP_REQUESTS SUBAGENT_SESSION_HEADER
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -euo pipefail
host=$1 port=$2 model=$3 conc=$4 out=$5 gpus=$6

ulimit -n "$(ulimit -Hn)" 2> /dev/null || true
if [ ! -x "${CLIENT_VENV:-}/bin/aiperf" ]; then
    CLIENT_VENV=/tmp/agentx-client-${SLURM_JOB_ID:-$$}/venv
    echo "installing the AgentX client into $CLIENT_VENV"
    bash "$(dirname "$0")/install.sh" "$CLIENT_VENV"
fi
mkdir -p "$CLIENT_HF_HOME"
export HF_HOME=$CLIENT_HF_HOME AIPERF_DATASET_MMAP_CACHE_DIR=$CLIENT_HF_HOME/aiperf_mmap_cache
export AIPERF_HTTP_TCP_USER_TIMEOUT=300000 AIPERF_HTTP_X_SESSION_ID_FROM_CORRELATION_ID=true
# Building the corpus's backing store (about 6 GB, cached in the mmap dir afterwards) outlasts
# aiperf's default 300 s; the measured runs' harness set both timeouts to 1800 s.
export AIPERF_DATASET_CONFIGURATION_TIMEOUT=1800 AIPERF_SERVICE_PROFILE_CONFIGURE_TIMEOUT=1800
if [ "$SUBAGENT_SESSION_HEADER" = 1 ]; then
    # Subagent requests carry X-Dynamo-Parent-Session-ID; the orchestrator routes them to the
    # parent conversation's context server.
    export AIPERF_HTTP_X_DYNAMO_SESSION_ID_FROM_CORRELATION_ID=true
fi
if [ "$CLIENT_OFFLINE" = 1 ]; then export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1; fi

# Prime every context-to-generation KV path with short random requests (TensorRT-LLM's client).
python3 -m tensorrt_llm.serve.scripts.benchmark_serving --model "$model" \
    --dataset-name random --random-ids --random-input-len 100 --random-output-len 10 \
    --num-prompts "$UCX_WARMUP_REQUESTS" --host "$host" --port "$port" \
    --ignore-eos --trust-remote-code --non-streaming || true

args=(--scenario inferencex-agentx-mvp -m "$model" --tokenizer-trust-remote-code
    --url "http://$host:$port" --endpoint /v1/chat/completions --endpoint-type chat --streaming
    --public-dataset semianalysis_cc_traces_weka_062126 --max-context-length 996579
    --trajectory-start-min-ratio 0.25 --trajectory-start-max-ratio 0.75
    --trace-idle-gap-cap-seconds 300 --failed-request-threshold 0.1 --use-server-token-count
    --no-gpu-telemetry --random-seed 42 --concurrency "$conc" --artifact-dir "$out" --ui simple)
if [ "$SMOKE" = 1 ]; then
    # Functional check: a 5-minute window (the scenario's minimum is 15 minutes unless overridden)
    # and only the mandatory primer requests as warmup.
    args+=(--benchmark-duration 300 --unsafe-override)
else
    args+=(--benchmark-duration 3600 --warmup-requests-per-lane 3)
fi

mkdir -p "$out"
rc=0
"$CLIENT_VENV/bin/aiperf" profile "${args[@]}" || rc=$?

python3 - "$out/profile_export_aiperf.json" "$gpus" "$rc" << 'EOF'
import json, sys
path, gpus, rc = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
try:
    data = json.load(open(path))
except OSError:
    print(f"SUMMARY FAIL aiperf exit code {rc}, no {path}")
    sys.exit(1)
def get(tag, stat="avg"):
    return (data.get(tag) or {}).get(stat) or 0.0
ok, err = get("request_count"), get("error_request_count")
rate = err / (ok + err) if ok + err else 1.0
itl = get("inter_token_latency", "p90")
ttft50, ttft90 = get("time_to_first_token", "p50") / 1000, get("time_to_first_token", "p90") / 1000
verdict = "PASS" if rc == 0 and rate <= 0.1 else "FAIL"
print(f"SUMMARY {verdict} {1000 / itl if itl else 0:.2f} tok/s/user (1000 / P90 ITL), "
      f"TTFT P50/P90 {ttft50:.2f}/{ttft90:.2f} s, {get('total_token_throughput') / gpus:,.0f} tok/s/GPU, "
      f"{get('output_token_throughput') / gpus:,.0f} output tok/s/GPU, "
      f"errors {err:.0f}/{ok + err:.0f} ({rate:.1%}), aiperf exit code {rc}")
sys.exit(verdict != "PASS")
EOF
