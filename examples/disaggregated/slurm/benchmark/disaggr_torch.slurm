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
        build_command="python3 ./scripts/build_wheel.py --trt_root /usr/local/tensorrt --use_ccache --clean"
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

# The Mooncake store bindings, when a worker config asks for the connector.
# Images built from this repo bake them in (docker/common/install_mooncake.sh),
# so the install below is only a fallback for images that predate it. Either
# way it is per job, since --container-name gives each node a container that
# lives for the job: anything installed here survives to the worker sruns but
# not into the next job.
mooncake_enabled=false
if grep -qs "mooncake-store" "${full_logdir}/ctx_config.yaml" "${full_logdir}/gen_config.yaml"; then
    mooncake_enabled=true
    # Both halves of the wheel are checked because they fail at different
    # times: the connector needs mooncake.store in every context rank, and
    # 'trtllm-serve mooncake_master' needs the binary on PATH.
    if srun --container-name=${container_name} \
        --container-mounts=${container_mount} --no-container-mount-home \
        --mpi=pmix --overlap -N 1 -n 1 \
        bash -c 'python3 -c "import mooncake.store" && command -v mooncake_master' \
        &> ${full_logdir}/2_install_mooncake.log; then
        echo "Mooncake store bindings already in the image; nothing to install"
    else
        mooncake_install_script=""
        if [ -n "${trtllm_repo:-}" ]; then
            mooncake_install_script="${trtllm_repo}/mooncake_disagg/install_mooncake_runtime.sh"
        fi
        if [ -z "${mooncake_install_script}" ] || [ ! -f "${mooncake_install_script}" ]; then
            cleanup_on_failure "A worker config requests the mooncake-store connector, this image does not have the bindings, and mooncake_disagg/install_mooncake_runtime.sh was not found under trtllm_repo='${trtllm_repo:-}'. Use an image built from this repo, or set environment.trtllm_repo to a checkout that contains the script."
        fi
        echo "Installing Mooncake store bindings on all nodes..."
        if ! srun --container-name=${container_name} \
            --container-mounts=${container_mount} --no-container-mount-home \
            --mpi=pmix --overlap -N $SLURM_NNODES --ntasks-per-node=1 \
            bash -c "MOONCAKE_WHEEL='${MOONCAKE_WHEEL:-}' bash ${mooncake_install_script}" \
            &>> ${full_logdir}/2_install_mooncake.log; then
            cleanup_on_failure "Mooncake store bindings installation failed. Check ${full_logdir}/2_install_mooncake.log for details"
        fi
        echo "Mooncake store bindings installation completed successfully"
    fi
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

# The pool is described in the worker configs and provisioned by trtllm-serve
# during its own bringup: the context server starts the master, renders the
# client config and publishes the master's address, and the generation servers
# read that address to lend the pool their memory. Nothing here starts, waits
# for or configures any of it.
#
# The one value a config written before submission cannot know is this job's log
# directory, which the master's address is published into, so a __LOG_DIR__
# placeholder in either worker config is filled in here.
if [ "${mooncake_enabled}" = "true" ]; then
    sed -i "s|__LOG_DIR__|${full_logdir}|g" \
        "${full_logdir}/ctx_config.yaml" "${full_logdir}/gen_config.yaml"
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
# serving, it just never hits, so surface the startup lines here rather than
# leaving them to be discovered after the benchmark. The registration line
# carries the bytes/page figure the pool sizing depends on.
if [ "${mooncake_enabled}" = "true" ]; then
    echo "Mooncake store startup lines from the context workers:"
    if ! grep -h "mooncake-store" "${full_logdir}"/3_output_CTX_*.log 2>/dev/null; then
        echo "  WARNING: no mooncake-store lines found. The connector may not have" \
             "loaded; check ${full_logdir}/3_output_CTX_*.log and" \
             "${full_logdir}/mooncake_master.log. The benchmark will still run," \
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

# Collect the store's traffic into one file. The per-event lines live at DEBUG
# in the worker logs (module _torch), so this is only populated when a config
# asks for that verbosity. The counts are what show the store did work rather
# than merely started.
if [ "${mooncake_enabled}" = "true" ]; then
    mooncake_summary="${full_logdir}/9_mooncake_summary.log"
    {
        # The address file is retracted when the master stops, so fall back to
        # the context server's log, which still names the master the run used.
        echo "master: $(tr -d '[:space:]' < "${full_logdir}/master.addr" 2>/dev/null \
            || grep -hoE "master at [0-9.]+:[0-9]+" "${full_logdir}"/3_output_CTX_*.log 2>/dev/null \
               | head -n 1 | awk '{print $3}' || echo unknown)"
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
        # Where the blocks physically went. The master names a segment for
        # every allocation and a segment is one client process's donated
        # memory, so grouping by segment host shows how much of the pool lives
        # on a decode node rather than on the prefill node that computed it.
        # Without a donor this section shows only the prefill node.
        echo "== block placement by segment host =="
        echo "(lending hosts: $(grep -hoE "GiB of [0-9.]+ is now part of the pool" \
            "${full_logdir}"/3_output_GEN_*.log 2>/dev/null \
            | awk '{print $3}' | sort -u | paste -sd, - || echo none))"
        grep -o "allocation_succeeded size=[0-9]* segment=[0-9.]*:[0-9]*" \
            "${full_logdir}/mooncake_master.log" 2>/dev/null \
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
        # Lending memory happens inside the generation servers, so their own
        # logs are where a segment that never mounted shows up.
        echo "== lent segments =="
        grep -h "mooncake-store: .*\(lending memory\|part of the pool\|withdrew\)" \
            "${full_logdir}"/3_output_GEN_*.log 2>/dev/null || echo "(none)"
        echo
        echo "== pool bringup (context server) =="
        grep -h "mooncake-store:" "${full_logdir}"/3_output_CTX_*.log 2>/dev/null \
            | head -n 30 || echo "(none)"
        echo
        # The master's own glog output, not anything TensorRT-LLM writes.
        echo "== master =="
        tail -n 50 "${full_logdir}/mooncake_master.log" 2>/dev/null || echo "(no master log)"
    } > "${mooncake_summary}" 2>&1
    echo "Mooncake store summary written to ${mooncake_summary}"
    cat "${mooncake_summary}"
fi

echo "Job completed successfully, total runtime: $SECONDS seconds"

# try to kill the server and workers
scancel ${SLURM_JOB_ID}
