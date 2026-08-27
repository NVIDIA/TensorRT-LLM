#!/bin/bash
set -euo pipefail

# Parse named arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        # Benchmark Configuration
        --benchmark-mode) benchmark_mode="$2"; shift 2 ;;

        # Environment and paths
        --trtllm-repo) trtllm_repo="$2"; shift 2 ;;
        --work-dir) work_dir="$2"; shift 2 ;;
        --full-logdir) full_logdir="$2"; shift 2 ;;
        --container-name) container_name="$2"; shift 2 ;;
        --container-mount) container_mount="$2"; shift 2 ;;
        --container-image) container_image="$2"; shift 2 ;;
        --build-wheel) build_wheel="$2"; shift 2 ;;
        --cuda-architectures) cuda_architectures="$2"; shift 2 ;;
        --trtllm-wheel-path) trtllm_wheel_path="$2"; shift 2 ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

# Print all parsed arguments
echo "Parsed arguments:"
echo
echo "Benchmark Configuration:"
echo "  benchmark_mode: ${benchmark_mode}"
echo
echo "Environment Configuration:"
echo "  trtllm_repo: ${trtllm_repo}"
echo "  work_dir: ${work_dir}"
echo "  full_logdir: ${full_logdir}"
echo "  container_mount: ${container_mount}"
echo "  container_image: ${container_image}"
echo "  build_wheel: ${build_wheel}"
echo "  cuda_architectures: ${cuda_architectures}"
echo "  trtllm_wheel_path: ${trtllm_wheel_path}"

# Set TRTLLM_DISAGG_BENCHMARK_GEN_ONLY=1 for gen_only_no_context mode
if [ "${benchmark_mode}" = "gen_only_no_context" ]; then
    export TRTLLM_DISAGG_BENCHMARK_GEN_ONLY=1
    echo "Setting TRTLLM_DISAGG_BENCHMARK_GEN_ONLY=1 for gen_only_no_context mode"
fi

# Function to cleanup on failure
cleanup_on_failure() {
    echo "Error: $1"
    scancel ${SLURM_JOB_ID}
    exit 1
}

replace_placeholder() {
    file_path="$1"
    all_nodes_str="$2"
    new_file_path="$3"
    cp "$file_path" "$new_file_path"
    IFS=',' read -r -a node_array <<< "$all_nodes_str"
    for i in "${!node_array[@]}"; do
        current_val="${node_array[$i]}"
        placeholder="<node${i}_placeholder>"

        # Use sed to replace the placeholder with the value in-place
        sed -i "s|$placeholder|$current_val|g" "${new_file_path}"
        echo "Replaced $placeholder with $current_val in ${new_file_path}"
    done
}

env > ${full_logdir}/environment.txt

# Start container
echo "Starting container..."
if ! srun -l --container-image=${container_image} \
        --container-name=${container_name} \
        --container-mounts=${container_mount} \
        --mpi=pmix \
        echo "Container up." &> ${full_logdir}/1_container_launch.log; then
    cleanup_on_failure "Failed to start container. Check ${full_logdir}/1_container_launch.log"
fi

# Install TensorRT-LLM
if [ -n "${trtllm_wheel_path}" ]; then
    # Install from pre-built wheel if path is provided
    echo "Installing TensorRT-LLM from wheel: ${trtllm_wheel_path}..."
    if ! srun --container-name=${container_name} \
        --container-mounts=${container_mount} --no-container-mount-home \
        --mpi=pmix --overlap -N $SLURM_NNODES --ntasks-per-node=1 \
        bash -c "pip install ${trtllm_wheel_path}[devel]" \
        &> ${full_logdir}/2_install.log; then
        cleanup_on_failure "TensorRT-LLM wheel installation failed. Check ${full_logdir}/2_install.log for details"
    fi
    echo "TensorRT-LLM wheel installation completed successfully"
elif [ -d "${trtllm_repo}" ]; then
    # Build and install from repository if no wheel path provided
    echo "Installing TensorRT-LLM from ${trtllm_repo}..."
    TRT_LLM_GIT_COMMIT=$(git -C ${trtllm_repo} rev-parse --short HEAD 2>/dev/null || echo "unknown")
    echo "TRT_LLM_GIT_COMMIT: ${TRT_LLM_GIT_COMMIT}"

    if [ "${build_wheel}" = "true" ]; then
        echo "Building TensorRT-LLM wheel on one node..."
        build_command="python3 ./scripts/build_wheel.py --use_ccache"
        if [ -n "${cuda_architectures:-}" ]; then
            build_command="${build_command} --cuda_architectures \"${cuda_architectures}\""
        fi
        if ! srun --container-name=${container_name} \
            --container-mounts=${container_mount} \
            --mpi=pmix --overlap -N 1 --ntasks-per-node=1 \
            bash -c "cd ${trtllm_repo} && ${build_command}" \
            &> ${full_logdir}/2_build.log; then
            cleanup_on_failure "TensorRT-LLM build failed. Check ${full_logdir}/2_build.log for details"
        fi
        echo "TensorRT-LLM build completed successfully"
    fi

    echo "Installing TensorRT-LLM..."
    if ! srun --container-name=${container_name} \
        --container-mounts=${container_mount} --no-container-mount-home \
        --mpi=pmix --overlap -N $SLURM_NNODES --ntasks-per-node=1 \
        bash -c "cd ${trtllm_repo} && pip install -e .[devel]" \
        &> ${full_logdir}/2_install.log; then
        cleanup_on_failure "TensorRT-LLM installation failed. Check ${full_logdir}/2_install.log for details"
    fi
    echo "TensorRT-LLM installation completed successfully"
else
    echo "trtllm_wheel_path and trtllm_repo are not provided, will use the installed TensorRT-LLM from the container"
    # get_env file is in the same directory as this script
    get_env_file=${work_dir}/get_env.py
    if ! srun --container-name=${container_name} \
        --container-mounts=${container_mount} --no-container-mount-home \
        --mpi=pmix --overlap -N 1 --ntasks-per-node=1 \
        bash -c "python ${get_env_file} -e ${full_logdir}/env_vars.json" \
        &> ${full_logdir}/2_get_env.log; then
        cleanup_on_failure "Failed to get TensorRT-LLM environment variables. Check ${full_logdir}/2_get_env.log for details"
    fi
    echo "TensorRT-LLM environment variables saved to ${full_logdir}/env_vars.json"
fi

# Install the Mooncake Python store bindings, but only when a worker config asks
# for the mooncake-store KV connector. The bindings shipped in the container
# images are unusable (see mooncake_disagg/README.md section 2), and the fix has
# to be reapplied per job: --container-name gives each node a container that
# lives for the job, so anything installed here survives to the worker sruns but
# not into the next job.
mooncake_enabled=false
if grep -qs "mooncake-store" "${full_logdir}/ctx_config.yaml" "${full_logdir}/gen_config.yaml"; then
    mooncake_enabled=true
    mooncake_install_script=""
    mooncake_donor_script=""
    if [ -n "${trtllm_repo:-}" ]; then
        mooncake_install_script="${trtllm_repo}/mooncake_disagg/install_mooncake_runtime.sh"
        mooncake_donor_script="${trtllm_repo}/mooncake_disagg/mooncake_segment_donor.py"
    fi
    if [ -z "${mooncake_install_script}" ] || [ ! -f "${mooncake_install_script}" ]; then
        cleanup_on_failure "A worker config requests the mooncake-store connector, but mooncake_disagg/install_mooncake_runtime.sh was not found under trtllm_repo='${trtllm_repo:-}'. Set environment.trtllm_repo to a checkout that contains it, or bake the bindings into the container image."
    fi
    if [ "${MOONCAKE_DONOR_SEGMENT_SIZE:-32GiB}" != "0" ] && [ ! -f "${mooncake_donor_script}" ]; then
        cleanup_on_failure "mooncake_disagg/mooncake_segment_donor.py was not found under trtllm_repo='${trtllm_repo:-}'. It contributes the generation nodes' host memory to the pool; without it the pool is prefill-node memory only. Set MOONCAKE_DONOR_SEGMENT_SIZE=0 to accept that and skip the donors."
    fi
    echo "Installing Mooncake store bindings on all nodes..."
    if ! srun --container-name=${container_name} \
        --container-mounts=${container_mount} --no-container-mount-home \
        --mpi=pmix --overlap -N $SLURM_NNODES --ntasks-per-node=1 \
        bash -c "MOONCAKE_WHEEL='${MOONCAKE_WHEEL:-}' bash ${mooncake_install_script}" \
        &> ${full_logdir}/2_install_mooncake.log; then
        cleanup_on_failure "Mooncake store bindings installation failed. Check ${full_logdir}/2_install_mooncake.log for details"
    fi
    echo "Mooncake store bindings installation completed successfully"
fi

# Get node lists and replace the placeholder with the actual node names
echo "SLURM_NODELIST: ${SLURM_NODELIST}"
all_nodes=($(scontrol show hostname $SLURM_NODELIST | sort))
all_nodes_str=$(IFS=','; echo "${all_nodes[*]}")
echo "all_nodes_str: ${all_nodes_str}"

start_server_cmds_base_file=${full_logdir}/start_server_cmds_base.sh
start_server_cmds_file=${full_logdir}/start_server_cmds.sh
replace_placeholder "${start_server_cmds_base_file}" "${all_nodes_str}" "${start_server_cmds_file}"
server_config_base_file=${full_logdir}/server_config_base.yaml
server_config_file=${full_logdir}/server_config.yaml
replace_placeholder "${server_config_base_file}" "${all_nodes_str}" "${server_config_file}"
client_cmds_base_file=${full_logdir}/client_cmds_base.sh
client_cmds_file=${full_logdir}/client_cmds.sh
replace_placeholder "${client_cmds_base_file}" "${all_nodes_str}" "${client_cmds_file}"

# Bring up the Mooncake store's pool: a master process plus the client config
# that names it. Both are per job and land in the log directory, so nothing has
# to be prepared by hand and concurrent jobs do not share state.
#
# The master lives inside this job, which means the pool dies with it. That is
# the right default for the single-job experiments, but it makes the
# survival-across-restarts case impossible to test, so set
# MOONCAKE_MASTER_ADDRESS in the submitting environment to reuse a master
# running as its own longer-lived job (mooncake_disagg/README.md section 4);
# this block then only writes the client config.
if [ "${mooncake_enabled}" = "true" ]; then
    mooncake_master_port=50051
    mooncake_master_metrics_port=9004
    mooncake_master_addr="${MOONCAKE_MASTER_ADDRESS:-}"

    if [ -z "${mooncake_master_addr}" ]; then
        mooncake_master_node="${all_nodes[0]}"
        mooncake_addr_file="${full_logdir}/mooncake_master.addr"
        rm -f "${mooncake_addr_file}"
        echo "Starting mooncake_master on ${mooncake_master_node}..."
        # --overlap because every GPU on this node is already claimed by a
        # worker; the master is a CPU-only process sharing the node with them.
        #
        # mooncake_master logs through glog, which writes to files under /tmp
        # inside the container unless told otherwise -- so without
        # GLOG_logtostderr the log below would be empty. GLOG_v=1 adds the
        # per-RPC lines that show workers registering segments and putting and
        # getting keys, which is the only view of the pool's side of the
        # conversation short of scraping the metrics port.
        srun -l --container-name=${container_name} \
            --container-mounts=${container_mount} --no-container-mount-home \
            --mpi=pmix --overlap --nodelist=${mooncake_master_node} -N 1 -n 1 \
            bash -c "export GLOG_logtostderr=1 GLOG_v=\"${MOONCAKE_MASTER_GLOG_V:-1}\"; \
                addr=\$(hostname -I 2>/dev/null | awk '{print \$1}'); \
                [ -n \"\${addr}\" ] || addr=\$(hostname -f); \
                echo \"\${addr}\" > ${mooncake_addr_file}; \
                echo \"mooncake_master starting on \$(hostname) at \${addr}:${mooncake_master_port} (metrics ${mooncake_master_metrics_port})\"; \
                exec mooncake_master --rpc_port=${mooncake_master_port} --metrics_port=${mooncake_master_metrics_port} --eviction_ratio=0.05" \
            &> ${full_logdir}/2_mooncake_master.log &

        for _ in $(seq 1 60); do
            if [ -s "${mooncake_addr_file}" ]; then
                break
            fi
            sleep 1
        done
        if [ ! -s "${mooncake_addr_file}" ]; then
            cleanup_on_failure "mooncake_master did not report its address within 60s. Check ${full_logdir}/2_mooncake_master.log for details"
        fi
        mooncake_master_addr="$(tr -d '[:space:]' < ${mooncake_addr_file}):${mooncake_master_port}"

        # A worker that opens its store handle before the master accepts
        # connections fails at startup, so wait for the port rather than the
        # process. /dev/tcp keeps this dependency-free.
        mooncake_master_ready=false
        mooncake_wait_start=${SECONDS}
        for _ in $(seq 1 60); do
            if (exec 3<>/dev/tcp/${mooncake_master_addr%:*}/${mooncake_master_port}) 2>/dev/null; then
                mooncake_master_ready=true
                break
            fi
            sleep 1
        done
        if [ "${mooncake_master_ready}" != "true" ]; then
            cleanup_on_failure "mooncake_master at ${mooncake_master_addr} is not accepting connections. Check ${full_logdir}/2_mooncake_master.log for details"
        fi
        echo "mooncake_master ready at ${mooncake_master_addr} on node ${mooncake_master_node} after $((SECONDS - mooncake_wait_start))s of waiting"
        echo "mooncake_master metrics: http://${mooncake_master_addr%:*}:${mooncake_master_metrics_port}"
        echo "mooncake_master log: ${full_logdir}/2_mooncake_master.log"
    else
        echo "Using externally managed mooncake_master at ${mooncake_master_addr}"
    fi

    # The schema is vLLM's, so one pool can serve both engines. Defaults are
    # the first-bring-up ones: TCP removes RDMA from the variable list, at the
    # cost of any performance conclusion. Set MOONCAKE_PROTOCOL=rdma with a
    # MOONCAKE_DEVICE_NAME from ibv_devinfo for a run worth quoting.
    # global_segment_size is contributed per worker process, so the pool is
    # this value times (ctx instances x world size).
    cat > "${full_logdir}/mooncake.json" <<EOF
{
  "metadata_server": "P2PHANDSHAKE",
  "master_server_address": "${mooncake_master_addr}",
  "protocol": "${MOONCAKE_PROTOCOL:-tcp}",
  "device_name": "${MOONCAKE_DEVICE_NAME:-}",
  "global_segment_size": "${MOONCAKE_GLOBAL_SEGMENT_SIZE:-16GiB}",
  "local_buffer_size": "${MOONCAKE_LOCAL_BUFFER_SIZE:-1GiB}",
  "role": "both",
  "cache_prefix": "trtllm",
  "transfer_batch_size": 64
}
EOF
    echo "Wrote Mooncake client config to ${full_logdir}/mooncake.json:"
    cat "${full_logdir}/mooncake.json"

    # Give the pool some decode-side capacity. Only processes that open a store
    # handle contribute memory to it, and only the context workers configure the
    # connector -- so by default every byte of the pool is prefill-node DRAM and
    # the store is just another way to do what TensorRT-LLM's native host offload
    # already does. One donor process per generation node puts decode-side memory
    # into the same pool, so blocks written by prefill can live there, while the
    # generation engine stays connector-free: it never reads or writes the store
    # and keeps its cache transceiver for the prefill-to-decode KV handoff.
    #
    # A donor is a separate process rather than a connector role because the
    # roles describe traffic, not capacity: producer/consumer/both all read or
    # write, and none of them means "contribute memory only".
    #
    # The donated memory is charged to the donor process and competes with the
    # generation worker's own kv_cache_config.host_cache_size on that node, so
    # size the two together. Set MOONCAKE_DONOR_SEGMENT_SIZE=0 to skip donation
    # and keep the pool prefill-only.
    mooncake_donor_size="${MOONCAKE_DONOR_SEGMENT_SIZE:-32GiB}"
    if [ "${mooncake_donor_size}" = "0" ]; then
        echo "MOONCAKE_DONOR_SEGMENT_SIZE=0: pool is prefill-node memory only"
    else
        # Read the generation nodes off the already-generated worker commands
        # rather than recomputing submit.py's node assignment here.
        if [ -n "${MOONCAKE_DONOR_NODES:-}" ]; then
            IFS=',' read -r -a mooncake_donor_nodes <<< "${MOONCAKE_DONOR_NODES}"
        else
            mooncake_donor_nodes=($(awk '/start_worker.sh GEN/ {
                    for (i = 1; i <= NF; i++) {
                        if ($i == "--nodelist") { print $(i + 1) }
                        else if ($i ~ /^--nodelist=/) { sub(/^--nodelist=/, "", $i); print $i }
                    }
                }' "${start_server_cmds_file}" | tr ',' '\n' | sort -u))
        fi

        if [ ${#mooncake_donor_nodes[@]} -eq 0 ]; then
            cleanup_on_failure "Could not determine the generation nodes to donate memory from. Set MOONCAKE_DONOR_NODES to a comma-separated list, or MOONCAKE_DONOR_SEGMENT_SIZE=0 to keep the pool prefill-only"
        fi

        mooncake_donor_ready_files=()
        for donor_node in "${mooncake_donor_nodes[@]}"; do
            donor_ready_file="${full_logdir}/mooncake_donor_${donor_node}.ready"
            rm -f "${donor_ready_file}"
            mooncake_donor_ready_files+=("${donor_ready_file}")
            echo "Donating ${mooncake_donor_size} of host memory on ${donor_node} to the pool..."
            srun -l --container-name=${container_name} \
                --container-mounts=${container_mount} --no-container-mount-home \
                --mpi=pmix --overlap --nodelist=${donor_node} -N 1 -n 1 \
                python3 ${mooncake_donor_script} \
                    --config ${full_logdir}/mooncake.json \
                    --segment-size ${mooncake_donor_size} \
                    --ready-file ${donor_ready_file} \
                &> ${full_logdir}/2_mooncake_donor_${donor_node}.log &
        done

        # Wait for the segments before any worker starts, so that the first
        # blocks prefill writes can already be placed on a decode node. A donor
        # that never mounts is a hard failure: the run would otherwise quietly
        # fall back to the prefill-only pool this is meant to replace.
        for donor_ready_file in "${mooncake_donor_ready_files[@]}"; do
            for _ in $(seq 1 120); do
                if [ -s "${donor_ready_file}" ]; then
                    break
                fi
                sleep 1
            done
            if [ ! -s "${donor_ready_file}" ]; then
                donor_node="$(basename "${donor_ready_file}" .ready)"
                donor_node="${donor_node#mooncake_donor_}"
                cleanup_on_failure "The memory donor on ${donor_node} did not mount its segment within 120s. Check ${full_logdir}/2_mooncake_donor_${donor_node}.log; if the node is short on memory, lower MOONCAKE_DONOR_SEGMENT_SIZE or the generation worker's kv_cache_config.host_cache_size"
            fi
        done
        echo "Pool capacity: ${MOONCAKE_GLOBAL_SEGMENT_SIZE:-16GiB} per context worker process" \
             "+ ${mooncake_donor_size} per generation node (${#mooncake_donor_nodes[@]} donor(s):" \
             "${mooncake_donor_nodes[*]})"
    fi
fi

# Per-worker hostfile / gpu_map files for srun --distribution=arbitrary.
# submit.py emits *_base.txt with <nodeN_placeholder>; rewrite them here.
for base_file in "${full_logdir}"/hostfile_*_base.txt "${full_logdir}"/gpu_map_*_base.txt; do
    [ -f "${base_file}" ] || continue
    runtime_file="${base_file%_base.txt}.txt"
    replace_placeholder "${base_file}" "${all_nodes_str}" "${runtime_file}"
done

# start the servers (skip ctx workers if TRTLLM_DISAGG_BENCHMARK_GEN_ONLY is set).
echo "Starting worker commands from ${start_server_cmds_file}..."
cat ${start_server_cmds_file} | while read cmd; do
    # Skip ctx worker commands if in gen-only mode
    # CTX appears as argument to start_worker.sh and in log filename
    if [ "${TRTLLM_DISAGG_BENCHMARK_GEN_ONLY:-0}" = "1" ] && [[ "$cmd" == *"start_worker.sh CTX"* ]]; then
        echo "Skipping ctx worker command (TRTLLM_DISAGG_BENCHMARK_GEN_ONLY is set): ${cmd}"
        continue
    fi
    echo "Executing command: ${cmd}"
    eval "${cmd}"
done
echo "Server is ready!"

# A connector that failed to open its store handle does not stop the worker from
# serving, it just silently never hits, so surface the startup lines here rather
# than leaving them to be discovered after the benchmark. The registration line
# carries the bytes/page figure the pool sizing depends on.
if [ "${mooncake_enabled}" = "true" ]; then
    echo "Mooncake store startup lines from the context workers:"
    if ! grep -h "mooncake-store" "${full_logdir}"/3_output_CTX_*.log 2>/dev/null; then
        echo "  WARNING: no mooncake-store lines found. The connector may not have" \
             "loaded; check ${full_logdir}/3_output_CTX_*.log and" \
             "${full_logdir}/2_mooncake_master.log. The benchmark will still run," \
             "but without the store."
    fi
fi

# Start client commands
echo "Starting client commands from ${client_cmds_file}..."
while read -r cmd <&3; do
    echo "Starting client command: ${cmd}"
    eval "${cmd}"
    if [ $? -ne 0 ]; then
        cleanup_on_failure "Command failed: ${cmd}."
    fi
done 3< "${client_cmds_file}"

# Collect the store's traffic into one file. The per-event lines live at DEBUG in
# the worker logs (module _torch), so this is only populated when a config asks
# for that verbosity; the counts are what distinguish "the store ran" from "the
# store ran and did something".
if [ "${mooncake_enabled}" = "true" ]; then
    mooncake_summary="${full_logdir}/9_mooncake_summary.log"
    {
        echo "master: ${mooncake_master_addr}"
        echo
        echo "== startup =="
        grep -h "mooncake-store.*\(ready\|registered layout\)" "${full_logdir}"/3_output_CTX_*.log 2>/dev/null || echo "(none)"
        echo
        echo "== event counts =="
        for pattern in "matched" "loaded" "failed to load" "failed to save" \
                       "lookup failed" "could not reserve connector prefix"; do
            count=$(grep -h "mooncake-store.*${pattern}" "${full_logdir}"/3_output_CTX_*.log 2>/dev/null | wc -l || true)
            echo "${pattern}: ${count}"
        done
        echo
        # Where the blocks physically went. The master names the segment for
        # every allocation, and a segment is one client process's donated
        # memory, so grouping by segment host answers the question the donors
        # exist for: how much of the pool's contents lives on a decode node
        # rather than on the prefill node that computed it. Without a donor this
        # section shows a single host, which is the prefill node.
        echo "== block placement by segment host =="
        echo "(donor hosts: $(cat "${full_logdir}"/mooncake_donor_*.ready 2>/dev/null | awk '{print $1}' | paste -sd, - || echo none))"
        grep -o "allocation_succeeded size=[0-9]* segment=[0-9.]*:[0-9]*" \
            "${full_logdir}/2_mooncake_master.log" 2>/dev/null \
            | awk '{
                    sub(/size=/, "", $2); sub(/segment=/, "", $3);
                    split($3, parts, ":"); host = parts[1]; port = parts[2];
                    allocs[host]++; bytes[host] += $2;
                    if (!((host, port) in seen)) {
                        seen[host, port] = 1;
                        segs[host] = segs[host] " " port;
                    }
                    total_allocs++; total_bytes += $2;
                }
                END {
                    if (total_allocs == 0) { print "(no allocations)"; exit }
                    for (h in allocs) {
                        printf "%-16s pages=%-7d %8.2f GiB  %5.1f%% of pool contents  segments:%s\n",
                            h, allocs[h], bytes[h] / 1073741824,
                            100 * bytes[h] / total_bytes, segs[h];
                    }
                    printf "%-16s pages=%-7d %8.2f GiB\n", "TOTAL", total_allocs, total_bytes / 1073741824;
                }' || echo "(could not parse master log)"
        echo
        echo "== donors =="
        for donor_log in "${full_logdir}"/2_mooncake_donor_*.log; do
            [ -f "${donor_log}" ] || continue
            echo "--- $(basename "${donor_log}") ---"
            grep -h "donating\|mounted\|failed\|exited" "${donor_log}" 2>/dev/null || tail -n 5 "${donor_log}"
        done
        echo
        echo "== master =="
        tail -n 50 "${full_logdir}/2_mooncake_master.log" 2>/dev/null || echo "(no master log)"
    } > "${mooncake_summary}" 2>&1
    echo "Mooncake store summary written to ${mooncake_summary}"
    cat "${mooncake_summary}"
fi

echo "Job completed successfully, total runtime: $SECONDS seconds"

# try to kill the server and workers
scancel ${SLURM_JOB_ID}
