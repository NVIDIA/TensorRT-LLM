#!/bin/bash
# Runs one AgentX point in its Slurm allocation; ./run.sh submits it. Starts the generation and
# context servers, waits for /health, starts the orchestrator and runs the AgentX client, all with
# srun + pyxis in one named container per node. Ends with one line: RESULT PASS|FAIL <point>: ...
# Everything of the run goes to $RUN_DIR: logs/, results/, the rendered configs and the scripts.
#
# Env from run.sh: PKG_DIR CLUSTER_ENV POINT_DIR RUN_DIR SMOKE CLIENT_CONCURRENCY SOURCE_COMMIT RUNTIME_PARENT
#                  (dry run: DRY_RUN=1 DRY_NODES="<host> ..." SLURM_JOB_ID)
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
source "$CLUSTER_ENV"
source "$POINT_DIR/point.env"
DRY_RUN=${DRY_RUN:-0}
point=$(basename "$POINT_DIR")
gpus_per_node=4
health_timeout=${HEALTH_TIMEOUT:-7200}   # seconds per server; loading 527 GB from NFS plus cold JIT and autotuning took over 1 h
orch_port=8333

# ---- Nodes, in the measured order: generation servers, context servers, then the client's own node.
if ((DRY_RUN)); then
    read -ra nodes <<< "$DRY_NODES"
else
    mapfile -t nodes < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
fi
ctx_per_node=${#CTX_GPU_ORDER[@]}
n_gen=$((GEN_SERVERS * GEN_NODES_PER_SERVER))
n_ctx=$(((CTX_SERVERS + ctx_per_node - 1) / ctx_per_node))
gen_nodes=("${nodes[@]:0:n_gen}")
ctx_nodes=("${nodes[@]:n_gen:n_ctx}")
case $ORCH_NODE in
    last_gen) orch=${gen_nodes[n_gen - 1]} ;;
    first_gen) orch=${gen_nodes[0]} ;;
esac
case $CLIENT_NODE in
    dedicated) client=${nodes[n_gen + n_ctx]} ;;
    ctx) client=${ctx_nodes[n_ctx - 1]} ;;
esac
IFS=, read -ra ctx_ranks <<< "${CTX_GPU_ORDER[0]}"
worker_gpus=$((n_gen * gpus_per_node + CTX_SERVERS * ${#ctx_ranks[@]}))
gen_port() { echo $((8677 + $1)); }
ctx_port() { echo $((8677 + GEN_SERVERS + $1)); }

# ---- Run directory: configs with this run's hosts, auth key and prompt-token cache dir.
umask 077   # the rendered configs hold the auth key
mkdir -p "$RUN_DIR/logs" "$RUN_DIR/results" "$RUN_DIR/cache"
repo=$(git -C "$PKG_DIR" rev-parse --show-toplevel)
cp "$repo/docker/common/validate_wheel_metadata.py" "$RUN_DIR/validate_wheel_metadata.py"
cp "$CLUSTER_ENV" "$RUN_DIR/cluster.env"
cp "$POINT_DIR/point.env" "$RUN_DIR/point.env"
cp "$PKG_DIR/slurm/worker.sh" "$PKG_DIR/slurm/stage.sh" "$PKG_DIR/client/agentx.sh" "$PKG_DIR/client/install.sh" "$RUN_DIR/"
# The servers load the checkpoint from node-local storage when STAGE_DIR is set (slurm/stage.sh);
# the client reads only the tokenizer, from MODEL_PATH.
server_model=$MODEL_PATH
if [ -n "${STAGE_DIR:-}" ]; then server_model=$STAGE_DIR/$(basename "$MODEL_PATH"); fi
auth_key=$(head -c 32 /dev/urandom | od -An -tx1 | tr -d ' \n')
ptc_dir=/dev/shm/dsv41-ptc-$SLURM_JOB_ID
gen_urls=() ctx_urls=()
for ((g = 0; g < GEN_SERVERS; g++)); do
    gen_urls+=("${gen_nodes[g * GEN_NODES_PER_SERVER]}:$(gen_port $g)")
done
for ((i = 0; i < CTX_SERVERS; i++)); do
    ctx_urls+=("${ctx_nodes[i / ctx_per_node]}:$(ctx_port $i)")
done
for f in ctx gen server; do
    sed -e "s|<MODEL_PATH>|$server_model|g; s|<AUTH_KEY>|$auth_key|g; s|<PTC_DIR>|$ptc_dir|g" \
        -e "s|<CTX_URLS>|[$(IFS=,; echo "${ctx_urls[*]}")]|; s|<GEN_URLS>|[$(IFS=,; echo "${gen_urls[*]}")]|" \
        "$POINT_DIR/$f.yaml" > "$RUN_DIR/$f.yaml"
done
mode=full
if [ "$SMOKE" = 1 ]; then mode=smoke; fi
cat > "$RUN_DIR/manifest.txt" << EOF
point:        $point ($mode, concurrency $CLIENT_CONCURRENCY, forced accepted tokens ${FORCE_ACCEPTED_TOKENS:-off})
job:          $SLURM_JOB_ID, started $(date -u +%FT%TZ)
image:        $IMAGE
source:       $SOURCE_COMMIT
compatible parent build: $RUNTIME_PARENT
model:        $MODEL_PATH${STAGE_DIR:+ (servers load the copy in $server_model)}
gen nodes:    ${gen_nodes[*]}
ctx nodes:    ${ctx_nodes[*]} (GPU order ${CTX_GPU_ORDER[*]})
orchestrator: $orch:$orch_port${ORCH_NUMA:+ (NUMA $ORCH_NUMA)}
client:       $client${CLIENT_NUMA:+ (NUMA $CLIENT_NUMA)}
package:      $PKG_DIR
EOF
cat "$RUN_DIR/manifest.txt"

# ---- Steps. Each runs in the background with its output in logs/<name>.log; a dry run prints them.
ctr=(--container-image="$IMAGE" --container-name=dsv41 --container-mounts="$CONTAINER_MOUNTS${STAGE_DIR:+,$STAGE_DIR:$STAGE_DIR}"
    --no-container-mount-home)
step=(srun -l --overlap --mpi=pmix "${ctr[@]}")
declare -A pid
start() {
    local name=$1
    shift
    if ((DRY_RUN)); then echo "[$name] $*"; return; fi
    "$@" &> "$RUN_DIR/logs/$name.log" &
    pid[$name]=$!
}
finish() {   # PASS|FAIL <detail>: print the result line, stop every step, exit
    echo "RESULT $1 $point ($mode, concurrency $CLIENT_CONCURRENCY): $2" | tee "$RUN_DIR/result.txt"
    if ((!DRY_RUN)); then
        kill "${pid[@]}" 2> /dev/null || true
        for _ in $(seq 60); do [ -z "$(jobs -rp)" ] && break; sleep 1; done
        srun --overlap "${ctr[@]}" --nodelist="$(IFS=,; echo "${ctx_nodes[*]}")" -N "$n_ctx" \
            --ntasks-per-node=1 rm -rf "$ptc_dir" &> /dev/null || true
    fi
    if [ "$1" = PASS ]; then exit 0; fi
    exit 1
}
wait_health() {   # <step name> <host:port>; fails as soon as any started step exits
    ((DRY_RUN)) && return 0
    local t=0 name
    until curl -sf -o /dev/null --max-time 5 "http://$2/health"; do
        for name in "${!pid[@]}"; do
            kill -0 "${pid[$name]}" 2> /dev/null || finish FAIL "$name exited; see logs/$name.log"
        done
        ((t < health_timeout)) || finish FAIL "$1 not healthy after ${health_timeout}s; see logs/$1.log"
        sleep 10
        t=$((t + 10))
    done
    echo "$1 healthy at $2 after ${t}s"
}

# Prepare the node-local bind mount before creating containers.
if [ -n "${STAGE_DIR:-}" ]; then
    if ((DRY_RUN)); then
        echo "[stage-dir] srun -N ${#nodes[@]} --ntasks-per-node=1 mkdir -p $STAGE_DIR"
    else
        srun --overlap -N "${#nodes[@]}" --ntasks-per-node=1 mkdir -p "$STAGE_DIR" ||
            finish FAIL "cannot create STAGE_DIR $STAGE_DIR"
    fi
fi
# Create the container on every node once; later steps reuse it by name.
if ((DRY_RUN)); then
    echo "[container] srun -N ${#nodes[@]} --ntasks-per-node=1 ${ctr[*]} true"
else
    srun -N "${#nodes[@]}" --ntasks-per-node=1 "${ctr[@]}" true &> "$RUN_DIR/logs/container.log" ||
        finish FAIL "container start failed; see logs/container.log"
fi

# Check the installed wheel, native imports and shared mounts on every node before loading weights.
if ((DRY_RUN)); then
    echo "[runtime-check] source commit $SOURCE_COMMIT; native import and shared mounts on every node"
else
    srun -N "${#nodes[@]}" --ntasks-per-node=1 "${ctr[@]}" bash -c '
        set -euo pipefail
        python3 "$1/validate_wheel_metadata.py" --expected-commit "$2" ||
            python3 "$1/validate_wheel_metadata.py" --expected-commit "$3"
        python3 -c "import tensorrt_llm; print(tensorrt_llm.__file__)"
        command -v trtllm-serve trtllm-llmapi-launch
        test -r "$4/config.json"
        test -w "$1"
    ' runtime-check "$RUN_DIR" "$SOURCE_COMMIT" "$RUNTIME_PARENT" "$MODEL_PATH" \
        &> "$RUN_DIR/logs/runtime-check.log" ||
        finish FAIL "runtime or mounts do not match this checkout; see logs/runtime-check.log"
fi

# Stage the checkpoint onto every server node (outside the container); the client node gets the
# empty mount point only.
if [ -n "${STAGE_DIR:-}" ]; then
    server_nodes=$(IFS=,; echo "${gen_nodes[*]},${ctx_nodes[*]}")
    if ((DRY_RUN)); then
        echo "[stage] srun -N $((n_gen + n_ctx)) --nodelist=$server_nodes bash stage.sh $MODEL_PATH $server_model"
    else
        srun --overlap -N $((n_gen + n_ctx)) --nodelist="$server_nodes" --ntasks-per-node=1 \
            bash "$RUN_DIR/stage.sh" "$MODEL_PATH" "$server_model" &> "$RUN_DIR/logs/stage.log" ||
            finish FAIL "staging the checkpoint failed; see logs/stage.log"
        cat "$RUN_DIR/logs/stage.log"
    fi
fi

# Generation servers: GEN_NODES_PER_SERVER x 4 ranks; the hostfile puts rank 0 on its first node.
for ((g = 0; g < GEN_SERVERS; g++)); do
    hosts=("${gen_nodes[@]:g * GEN_NODES_PER_SERVER:GEN_NODES_PER_SERVER}")
    hostfile=$RUN_DIR/hostfile_gen$g
    for h in "${hosts[@]}"; do for ((k = 0; k < gpus_per_node; k++)); do echo "$h"; done; done > "$hostfile"
    start gen$g env SLURM_HOSTFILE="$hostfile" "${step[@]}" --kill-on-bad-exit=1 \
        --ntasks=$((GEN_NODES_PER_SERVER * gpus_per_node)) --distribution=arbitrary \
        bash "$RUN_DIR/worker.sh" "$RUN_DIR" gen 0,1,2,3 "$(gen_port $g)" "${hosts[0]}"
done
# Context servers: a node's servers take the GPU lists of CTX_GPU_ORDER in turn.
for ((i = 0; i < CTX_SERVERS; i++)); do
    node=${ctx_nodes[i / ctx_per_node]}
    start ctx$i "${step[@]}" --kill-on-bad-exit=1 --nodelist="$node" -N 1 --ntasks=${#ctx_ranks[@]} \
        bash "$RUN_DIR/worker.sh" "$RUN_DIR" ctx "${CTX_GPU_ORDER[i % ctx_per_node]}" "$(ctx_port $i)" "$node"
done
for ((g = 0; g < GEN_SERVERS; g++)); do wait_health gen$g "${gen_urls[g]}"; done
for ((i = 0; i < CTX_SERVERS; i++)); do wait_health ctx$i "${ctx_urls[i]}"; done

# Orchestrator.
orch_cmd=(trtllm-serve disaggregated -c "$RUN_DIR/server.yaml" -t 7200 -r 7200)
if [ -n "$ORCH_NUMA" ]; then
    orch_cmd=(numactl --cpunodebind="$ORCH_NUMA" --preferred="$ORCH_NUMA" "${orch_cmd[@]}")
fi
start server "${step[@]}" --nodelist="$orch" -N 1 --ntasks=1 env TRTLLM_SERVER_DISABLE_GC=1 \
    bash -c 'ulimit -n "$(ulimit -Hn)"; exec "$@"' orchestrator "${orch_cmd[@]}"
wait_health server "$orch:$orch_port"

# AgentX client, in the foreground.
client_cmd=(env SMOKE="$SMOKE" UCX_WARMUP_REQUESTS="$UCX_WARMUP_REQUESTS"
    SUBAGENT_SESSION_HEADER="$SUBAGENT_SESSION_HEADER" CLIENT_VENV="$CLIENT_VENV"
    CLIENT_HF_HOME="$CLIENT_HF_HOME" CLIENT_OFFLINE="$CLIENT_OFFLINE"
    bash "$RUN_DIR/agentx.sh" "$orch" "$orch_port" "$MODEL_PATH" "$CLIENT_CONCURRENCY"
    "$RUN_DIR/results" "$worker_gpus")
if [ -n "$CLIENT_NUMA" ]; then
    client_cmd=(numactl --cpunodebind="$CLIENT_NUMA" --preferred="$CLIENT_NUMA" "${client_cmd[@]}")
fi
if ((DRY_RUN)); then
    echo "[client] ${step[*]} --nodelist=$client -N 1 --ntasks=1 ${client_cmd[*]}"
    echo "dry run: nothing submitted; rendered configs in $RUN_DIR"
    exit 0
fi
client_rc=0
"${step[@]}" --nodelist="$client" -N 1 --ntasks=1 "${client_cmd[@]}" &> "$RUN_DIR/logs/client.log" ||
    client_rc=$?
summary=$(sed -n 's/^.*SUMMARY \(PASS\|FAIL\) //p' "$RUN_DIR/logs/client.log" | tail -1)
if [ "$client_rc" = 0 ] && [ -f "$RUN_DIR/results/profile_export_aiperf.json" ]; then
    finish PASS "$summary"
fi
finish FAIL "client exit code $client_rc${summary:+; $summary}; see logs/client.log"
