#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Set up error handling
set -xEeuo pipefail
trap 'rc=$?; echo "Error in file ${BASH_SOURCE[0]} on line $LINENO: $BASH_COMMAND (exit $rc)"; exit $rc' ERR

cd $resourcePathNode
llmSrcNode=$resourcePathNode/TensorRT-LLM/src

set_value_in_command() {
    # Parameters
    local key="$1"
    local value="$2"
    local command="$3"

    # Transform the key
    local placeholder="__PLACEHOLDER_${key}__"

    # Check if placeholder exists
    if [[ "$command" != *"$placeholder"* ]]; then
        echo "Error: placeholder '$placeholder' not found in the command" >&2
        return 1
    fi

    # Replace all occurrences
    local result="${command//${placeholder}/${value}}"

    # Return the result
    echo "$result"
}

# Record the host/container /dev/shm relationship before installation can
# start helper processes or remove the host-created canary.
if [[ "$pytestCommand" == *"l0_b200_nvbug_6336747.txt"* ]]; then
    export TLLM_SHM_TRACE_DIR="$jobWorkspace/shared_tensor_shm_trace"
    mkdir -p "$TLLM_SHM_TRACE_DIR"
    innerShmStat=$(stat -Lc '%d:%i' /dev/shm)
    innerCanaryStat="missing"
    sharedHostShm="not-configured"
    if [[ -n "${TLLM_HOST_SHM_CANARY_PATH:-}" ]]; then
        sharedHostShm="no"
        if [[ -e "$TLLM_HOST_SHM_CANARY_PATH" ]]; then
            innerCanaryStat=$(stat -Lc '%d:%i' "$TLLM_HOST_SHM_CANARY_PATH")
            if [[ "$innerCanaryStat" == "${TLLM_HOST_SHM_CANARY_STAT:-}" ]]; then
                sharedHostShm="yes"
            else
                sharedHostShm="mismatch"
            fi
        fi
    fi
    {
        echo "time=$(date +%s.%N)"
        echo "host_mntns=${TLLM_HOST_MNTNS:-?}"
        echo "inner_mntns=$(readlink /proc/self/ns/mnt)"
        echo "inner_ipcns=$(readlink /proc/self/ns/ipc)"
        echo "host_shm_stat=${TLLM_HOST_SHM_STAT:-?}"
        echo "inner_shm_stat=$innerShmStat"
        echo "host_canary_path=${TLLM_HOST_SHM_CANARY_PATH:-?}"
        echo "host_canary_stat=${TLLM_HOST_SHM_CANARY_STAT:-?}"
        echo "inner_canary_stat=$innerCanaryStat"
        echo "shared_host_dev_shm=$sharedHostShm"
        awk '$5 == "/dev/shm" {print "inner_mountinfo=" $0}' /proc/self/mountinfo
    } > "$TLLM_SHM_TRACE_DIR/shm_mount_topology.${SLURM_PROCID}.log"
fi

# Only the first process will set the git config
if [ $SLURM_PROCID -eq 0 ]; then
    # Update HOME/.gitconfig
    if ! git config --global --get-all safe.directory | grep -Fxq "*"; then
        git config --global --add safe.directory "*"
    fi
fi

# Aggregated mode will run install together with pytest in slurm_run.sh
# Disaggregated mode will run install separately in slurm_install.sh
if [[ "$stageName" != *Disagg* ]]; then
    installScriptPath="$(dirname "${BASH_SOURCE[0]}")/$(basename "${BASH_SOURCE[0]}" | sed 's/slurm_run\.sh/slurm_install.sh/')"
    source "$installScriptPath"
    slurm_install_setup
fi

# NVBug 6336747 runs through this native-sbatch path, after the source archive is
# unpacked. Build and validate the preload tracer here so pytest, MPI children,
# and torch_shm_manager all inherit it.
if [[ "$pytestCommand" == *"l0_b200_nvbug_6336747.txt"* ]]; then
    traceLibrary="/tmp/libnvbug6336747_shm_trace_${SLURM_JOB_ID}_${SLURM_PROCID}.so"
    gcc -shared -fPIC -Wall -Wextra -Werror \
        -o "$traceLibrary" \
        "$llmSrcNode/tests/integration/defs/shared_tensor_shm_trace.c" -ldl
    export LD_PRELOAD="${traceLibrary}${LD_PRELOAD:+:${LD_PRELOAD}}"

    traceProbe="torch_nvbug6336747_probe_${SLURM_JOB_ID}_${SLURM_PROCID}"
    TRACE_PROBE="$traceProbe" python3 -c \
        'import os; from multiprocessing import shared_memory; shm = shared_memory.SharedMemory(name=os.environ["TRACE_PROBE"], create=True, size=4096); shm.unlink(); shm.close()'
    if ! grep -R -Fq -- "$traceProbe" "$TLLM_SHM_TRACE_DIR"; then
        echo "NVBug 6336747 tracer self-test failed: no record for $traceProbe" >&2
        exit 1
    fi

    # The preload shim cannot see raw syscalls or libc-private calls. Build a
    # ptrace/seccomp wrapper that follows pytest descendants and records the
    # result of every unlink/unlinkat syscall without tracing unrelated calls.
    syscallTraceBinary="/tmp/nvbug6336747_unlink_trace_${SLURM_JOB_ID}_${SLURM_PROCID}"
    gcc -std=c11 -Wall -Wextra -Werror \
        -o "$syscallTraceBinary" \
        "$llmSrcNode/tests/integration/defs/shared_tensor_unlink_syscall_trace.c"
    syscallTraceProbe="/tmp/torch_nvbug6336747_syscall_probe_${SLURM_JOB_ID}_${SLURM_PROCID}"
    syscallTraceProbeLog="$TLLM_SHM_TRACE_DIR/unlink_syscalls_probe.${SLURM_PROCID}.log"
    SYSCALL_TRACE_PROBE="$syscallTraceProbe" "$syscallTraceBinary" "$syscallTraceProbeLog" -- python3 -c \
        'import os; path = os.environ["SYSCALL_TRACE_PROBE"]; open(path, "wb").close(); os.unlink(path)'
    if ! grep -Fq -- "result=0 errno=0 path=$syscallTraceProbe" "$syscallTraceProbeLog"; then
        echo "NVBug 6336747 syscall tracer self-test failed: no unlink record for $syscallTraceProbe" >&2
        exit 1
    fi
    export TLLM_UNLINK_SYSCALL_TRACER="$syscallTraceBinary"
    export TLLM_UNLINK_SYSCALL_TRACE_FILE="$TLLM_SHM_TRACE_DIR/unlink_syscalls.${SLURM_PROCID}.log"

    # inotify observes namespace mutations from every process that can modify
    # this /dev/shm directory, including processes outside the pytest tree. It
    # cannot identify the actor, but distinguishes unlink, rename, and unmount.
    shmEventTraceBinary="/tmp/nvbug6336747_shm_event_trace_${SLURM_JOB_ID}_${SLURM_PROCID}"
    gcc -std=c11 -Wall -Wextra -Werror \
        -o "$shmEventTraceBinary" \
        "$llmSrcNode/tests/integration/defs/shared_tensor_shm_event_trace.c"
    shmEventTraceProbeLog="$TLLM_SHM_TRACE_DIR/shm_namespace_events_probe.${SLURM_PROCID}.log"
    "$shmEventTraceBinary" "$shmEventTraceProbeLog" /dev/shm &
    shmEventTraceProbePid=$!
    shmEventTraceProbeReady=0
    for _ in {1..50}; do
        if grep -Fq -- "event=watch_start" "$shmEventTraceProbeLog"; then
            shmEventTraceProbeReady=1
            break
        fi
        if ! kill -0 "$shmEventTraceProbePid" 2>/dev/null; then
            break
        fi
        sleep 0.1
    done
    if [[ $shmEventTraceProbeReady -ne 1 ]]; then
        kill -TERM "$shmEventTraceProbePid" 2>/dev/null || true
        wait "$shmEventTraceProbePid" || true
        echo "NVBug 6336747 shm namespace tracer failed to start" >&2
        exit 1
    fi
    shmEventTraceProbe="/dev/shm/torch_nvbug6336747_event_probe_${SLURM_JOB_ID}_${SLURM_PROCID}"
    : > "$shmEventTraceProbe"
    rm -f "$shmEventTraceProbe"
    kill -TERM "$shmEventTraceProbePid"
    wait "$shmEventTraceProbePid"
    shmEventTraceProbeName=$(basename "$shmEventTraceProbe")
    if ! awk -v name="$shmEventTraceProbeName" \
        'index($0, "create=1") && index($0, "name=" name) {found=1} END {exit !found}' \
        "$shmEventTraceProbeLog"; then
        echo "NVBug 6336747 shm namespace tracer self-test failed: no create event" >&2
        exit 1
    fi
    if ! awk -v name="$shmEventTraceProbeName" \
        'index($0, "delete=1") && index($0, "name=" name) {found=1} END {exit !found}' \
        "$shmEventTraceProbeLog"; then
        echo "NVBug 6336747 shm namespace tracer self-test failed: no delete event" >&2
        exit 1
    fi
    export TLLM_SHM_EVENT_TRACER="$shmEventTraceBinary"
    export TLLM_SHM_EVENT_TRACE_FILE="$TLLM_SHM_TRACE_DIR/shm_namespace_events.${SLURM_PROCID}.log"

    # inotify names the removed file but not the remover. fanotify directory-entry
    # events are delivered with the actor's pid, so this tracer records who removes
    # each /dev/shm/torch_* entry (pid/ppid/uid/comm/exe + mnt/ipc namespace),
    # including processes outside the pytest tree. Best-effort: it needs
    # CAP_SYS_ADMIN and tmpfs file-handle support, so degrade quietly (the required
    # tracers above still run) if it cannot build or start.
    actorTraceBinary="/tmp/nvbug6336747_shm_actor_trace_${SLURM_JOB_ID}_${SLURM_PROCID}"
    actorTraceReady=0
    actorTraceBuildLog="$TLLM_SHM_TRACE_DIR/shm_actor_build.${SLURM_PROCID}.log"
    if gcc -std=c11 -Wall -Wextra -Werror -o "$actorTraceBinary" \
            "$llmSrcNode/tests/integration/defs/shared_tensor_shm_actor_trace.c" \
            2>"$actorTraceBuildLog"; then
        actorTraceProbeLog="$TLLM_SHM_TRACE_DIR/shm_delete_actors_probe.${SLURM_PROCID}.log"
        "$actorTraceBinary" "$actorTraceProbeLog" /dev/shm &
        actorTraceProbePid=$!
        actorTraceProbeStarted=0
        for _ in {1..50}; do
            if grep -Fq -- "event=watch_start" "$actorTraceProbeLog" 2>/dev/null; then
                actorTraceProbeStarted=1
                break
            fi
            if grep -Fq -- "event=watch_error" "$actorTraceProbeLog" 2>/dev/null; then
                break
            fi
            if ! kill -0 "$actorTraceProbePid" 2>/dev/null; then
                break
            fi
            sleep 0.1
        done
        if [[ $actorTraceProbeStarted -eq 1 ]]; then
            actorTraceProbe="/dev/shm/torch_nvbug6336747_actor_probe_${SLURM_JOB_ID}_${SLURM_PROCID}"
            : > "$actorTraceProbe" || true
            rm -f "$actorTraceProbe" || true
            sleep 0.2
            kill -TERM "$actorTraceProbePid" 2>/dev/null || true
            wait "$actorTraceProbePid" 2>/dev/null || true
            actorTraceProbeName=$(basename "$actorTraceProbe")
            if awk -v name="$actorTraceProbeName" \
                'index($0, "event=removal op=delete name=" name " ") && \
                 match($0, /actor_pid=[1-9][0-9]*/) {found=1} END {exit !found}' \
                "$actorTraceProbeLog"; then
                actorTraceReady=1
            else
                echo "NVBug 6336747 actor tracer self-test saw no delete record with a usable actor PID; " \
                    "disabling (non-fatal)" >&2
            fi
        else
            kill -TERM "$actorTraceProbePid" 2>/dev/null || true
            wait "$actorTraceProbePid" 2>/dev/null || true
            echo "NVBug 6336747 actor tracer could not start (needs CAP_SYS_ADMIN + tmpfs FID support); disabling (non-fatal)" >&2
        fi
    else
        echo "NVBug 6336747 actor tracer failed to build; disabling (non-fatal). See $(basename "$actorTraceBuildLog")" >&2
    fi
    if [[ $actorTraceReady -eq 1 ]]; then
        export TLLM_SHM_ACTOR_TRACER="$actorTraceBinary"
        export TLLM_SHM_ACTOR_TRACE_FILE="$TLLM_SHM_TRACE_DIR/shm_delete_actors.${SLURM_PROCID}.log"
    fi

    export TLLM_SHUTDOWN_TRACE_DIR="$TLLM_SHM_TRACE_DIR"
    export TLLM_SHUTDOWN_TRACE_TIMEOUT_SEC=60
    export TLLM_SHUTDOWN_FORCE_EXIT=1
    export TLLM_SHM_TRACE_REQUIRED=1
fi

if [[ "$stageName" == *GB200* ]]; then
    echo "Checking Coherent GPU mapping (for GB200)..."
    grep Coherent /proc/driver/nvidia/params || echo "Unable to grep Coherent from /proc/driver/nvidia/params"
fi

llmapiLaunchScript="$llmSrcNode/tensorrt_llm/llmapi/trtllm-llmapi-launch"
chmod +x $llmapiLaunchScript
cd $llmSrcNode/tests/integration/defs

# get trtllm wheel path and add to pytest command
trtllmWhlPath=$(pip3 show tensorrt_llm | grep Location | cut -d ' ' -f 2)
trtllmWhlPath=$(echo "$trtllmWhlPath" | sed 's/[[:space:]]+/_/g')
echo "TRTLLM WHEEL PATH: $trtllmWhlPath"
# In disaggregated mode, we only set coverage config file in benchmark pytest.
if [[ -z "${DISAGG_SERVING_TYPE:-}" || "${DISAGG_SERVING_TYPE}" == "BENCHMARK" ]]; then
    pytestCommand=$(set_value_in_command "TRTLLM_WHL_PATH" "$trtllmWhlPath" "$pytestCommand")
fi

# Only the first process will save the coverage config file
if [ $SLURM_PROCID -eq 0 ]; then
    sed -i "s|---wheel_path---|$trtllmWhlPath|g" "$coverageConfigFile"
else
    # Sleep 30 seconds to wait for the coverage config file to be saved
    sleep 30
fi

containerPipLLMLibPath=$(pip3 show tensorrt_llm | grep "Location" | awk -F ":" '{ gsub(/ /, "", $2); print $2"/tensorrt_llm/libs"}')
containerPipLLMLibPath=$(echo "$containerPipLLMLibPath" | sed 's/[[:space:]]+/_/g')
containerLDLibPath=$LD_LIBRARY_PATH
containerLDLibPath=$(echo "$containerLDLibPath" | sed 's/[[:space:]]+/_/g')
if [[ "$containerLDLibPath" != *"$containerPipLLMLibPath"* ]]; then
  containerLDLibPath="$containerPipLLMLibPath:$containerLDLibPath"
  containerLDLibPath="${containerLDLibPath%:}"
fi
export LD_LIBRARY_PATH=$containerLDLibPath

# Slurm ENROOT/pyxis may inject UCX_TLS=tcp from the host MPI stack (intended for
# host-only MPI jobs). That disables CUDA transports and breaks NIXL GPU memory
# registration. Unset it so UCX can auto-select.
if [ "${UCX_TLS:-}" = "tcp" ]; then
    unset UCX_TLS
    echo "Unset UCX_TLS (cluster injected UCX_TLS=tcp)"
fi

# Force PMIx to use the in-memory hash GDS instead of ds12/ds21 shared-memory.
# Under `srun --mpi=pmix` with the DLFW 26.04 OpenMPI build, the shared-memory
# GDS modes can fail to publish UCX worker addresses across nodes, producing:
#   pml_ucx.c:178  Error: Failed to receive UCX worker address: Not found (-13)
#   pml_ucx.c:482  Error: Failed to resolve UCX endpoint for rank N
# See https://github.com/open-mpi/ompi/issues/6981. Setting this is a no-op
# when PMIx isn't used.
export PMIX_MCA_gds=hash
echo "Library Path:"
echo "$LD_LIBRARY_PATH"
env | sort

echo "Full Command: $pytestCommand"

# For single-node test runs or disaggregated benchmark/server runs, clear all
# environment variables related to Slurm and MPI. This prevents test processes
# (e.g., pytest) from incorrectly initializing MPI when running under a
# single-node srun environment.
# TODO: check if we can take advantage of --export=None arg when execute srun instead
# of unset them in the script
if [ "${SLURM_JOB_NUM_NODES:-1}" -eq 1 ] || \
   [ "${DISAGG_SERVING_TYPE:-}" == "BENCHMARK" ] || \
   [ "${DISAGG_SERVING_TYPE:-}" == "DISAGG_SERVER" ]; then
    for v in ${!PMI@} ${!PMIX@} ${!MPI@} ${!OMPI@} ${!SLURM@}; do
        if [ "$v" != "SLURM_PROCID" ]; then
            unset "$v"
        fi
    done
fi

# Start the directory watcher outside the ptrace wrapper so it continues to
# observe the namespace if pytest or the wrapper aborts.
shmEventTracePid=""
if [[ -n "${TLLM_SHM_EVENT_TRACE_FILE:-}" ]]; then
    "$TLLM_SHM_EVENT_TRACER" "$TLLM_SHM_EVENT_TRACE_FILE" /dev/shm &
    shmEventTracePid=$!
    shmEventTraceReady=0
    for _ in {1..50}; do
        if grep -Fq -- "event=watch_start" "$TLLM_SHM_EVENT_TRACE_FILE"; then
            shmEventTraceReady=1
            break
        fi
        if ! kill -0 "$shmEventTracePid" 2>/dev/null; then
            break
        fi
        sleep 0.1
    done
    if [[ $shmEventTraceReady -ne 1 ]]; then
        kill -TERM "$shmEventTracePid" 2>/dev/null || true
        wait "$shmEventTracePid" || true
        echo "NVBug 6336747 shm namespace tracer failed to start for pytest" >&2
        exit 1
    fi
fi

# Turn off "exit on error" so the following lines always run
set +e

pytest_exit_code=0
perf_check_exit_code=0
perf_report_exit_code=0

# Best-effort node-level removers (NVBug 6336747), started outside the ptrace
# wrapper so they keep observing if pytest or the wrapper aborts. Neither is
# allowed to fail the run.
shmActorTracePid=""
if [[ -n "${TLLM_SHM_ACTOR_TRACE_FILE:-}" ]]; then
    "$TLLM_SHM_ACTOR_TRACER" "$TLLM_SHM_ACTOR_TRACE_FILE" /dev/shm &
    shmActorTracePid=$!
    for _ in {1..50}; do
        if grep -Fq -- "event=watch_start" "$TLLM_SHM_ACTOR_TRACE_FILE" 2>/dev/null; then
            break
        fi
        if ! kill -0 "$shmActorTracePid" 2>/dev/null; then
            shmActorTracePid=""
            break
        fi
        sleep 0.1
    done
fi

# Optional bpftrace call-path tracer: only where bpftrace + BTF + CAP_BPF exist.
vfsUnlinkTracePid=""
if [[ -n "${TLLM_SHM_TRACE_DIR:-}" ]] && command -v bpftrace >/dev/null 2>&1; then
    vfsUnlinkTraceLog="$TLLM_SHM_TRACE_DIR/vfs_unlink.${SLURM_PROCID}.log"
    bpftrace "$llmSrcNode/tests/integration/defs/shared_tensor_vfs_unlink.bt" \
        >"$vfsUnlinkTraceLog" 2>&1 &
    vfsUnlinkTraceCandidatePid=$!
    vfsUnlinkTraceReady=0
    for _ in {1..50}; do
        if grep -Fq -- "vfs_unlink actor/call-path tracer started" "$vfsUnlinkTraceLog" 2>/dev/null; then
            if kill -0 "$vfsUnlinkTraceCandidatePid" 2>/dev/null; then
                vfsUnlinkTraceReady=1
            fi
            break
        fi
        if ! kill -0 "$vfsUnlinkTraceCandidatePid" 2>/dev/null; then
            break
        fi
        sleep 0.1
    done
    if [[ $vfsUnlinkTraceReady -eq 1 ]]; then
        vfsUnlinkTracePid=$vfsUnlinkTraceCandidatePid
    else
        kill -INT "$vfsUnlinkTraceCandidatePid" 2>/dev/null || true
        wait "$vfsUnlinkTraceCandidatePid" 2>/dev/null || true
        echo "NVBug 6336747 vfs_unlink bpftrace failed to become ready; skipping stack tracer (non-fatal). " \
            "See $(basename "$vfsUnlinkTraceLog")" >&2
    fi
fi

if [[ -n "${TLLM_UNLINK_SYSCALL_TRACE_FILE:-}" ]]; then
    "$TLLM_UNLINK_SYSCALL_TRACER" "$TLLM_UNLINK_SYSCALL_TRACE_FILE" -- bash -c "$pytestCommand"
else
    eval "$pytestCommand"
fi
pytest_exit_code=$?
echo "Rank${SLURM_PROCID} Pytest finished execution with exit code $pytest_exit_code"

if [[ -n "$shmEventTracePid" ]]; then
    kill -TERM "$shmEventTracePid" 2>/dev/null
    shm_event_trace_kill_exit_code=$?
    wait "$shmEventTracePid"
    shm_event_trace_exit_code=$?
    if [[ $shm_event_trace_kill_exit_code -ne 0 || $shm_event_trace_exit_code -ne 0 ]]; then
        echo "NVBug 6336747 shm namespace tracer exited unexpectedly: " \
            "kill=$shm_event_trace_kill_exit_code, wait=$shm_event_trace_exit_code" >&2
        if [[ $pytest_exit_code -eq 0 ]]; then
            pytest_exit_code=1
        fi
    fi
fi

# Stop the best-effort node-level removers. Failures here never affect the run.
if [[ -n "$shmActorTracePid" ]]; then
    kill -TERM "$shmActorTracePid" 2>/dev/null || true
    wait "$shmActorTracePid" 2>/dev/null || true
fi
if [[ -n "$vfsUnlinkTracePid" ]]; then
    # SIGINT so bpftrace runs its END probes and flushes the aggregation maps.
    kill -INT "$vfsUnlinkTracePid" 2>/dev/null || true
    wait "$vfsUnlinkTracePid" 2>/dev/null || true
fi

# DEBUG: Diagnose intermittent "unrecognized arguments" failure (Exit Code 4)
# Remove this after the issue is resolved
if [ $pytest_exit_code -eq 4 ]; then
    echo "DEBUG: Pytest failed with usage error (exit code 4)"
    echo "DEBUG: Directory state at $(pwd):"
    ls -l
    echo "DEBUG: Directory state at $llmSrcNode/tests/integration/defs:"
    ls -l $llmSrcNode/tests/integration/defs

    echo "DEBUG: conftest.py content:"
    md5sum $llmSrcNode/tests/integration/defs/conftest.py

    echo "DEBUG: pytest.ini content:"
    md5sum $llmSrcNode/tests/integration/defs/pytest.ini

    echo "DEBUG: Check importability of conftest.py"
    python3 -c "import sys; sys.path.insert(0, '.'); import conftest; print('DEBUG: conftest imported successfully')"
fi

if [ $SLURM_PROCID -eq 0 ] && [ "$perfMode" = "true" ]; then
    # Only PyTorch perf stages remain; the TensorRT perf baseline was removed.
    basePerfFilename="base_perf_pytorch.csv"
    basePerfPath="$llmSrcNode/tests/integration/defs/perf/$basePerfFilename"
    echo "Check Perf Result"
    python3 $llmSrcNode/tests/integration/defs/perf/sanity_perf_check.py \
        $stageName/perf_script_test_results.csv \
        $basePerfPath
    perf_check_exit_code=$?

    echo "Create Perf Report"
    python3 $llmSrcNode/tests/integration/defs/perf/create_perf_comparison_report.py \
        --output_path $stageName/report.pdf \
        --files $stageName/perf_script_test_results.csv \
        $basePerfPath
    perf_report_exit_code=$?
    echo "Rank${SLURM_PROCID} Perf report finished execution with exit code $perf_report_exit_code"

    if [ "$perf_check_exit_code" -eq 0 ] && [ "$perf_report_exit_code" -ne 0 ]; then
        perf_check_exit_code=$perf_report_exit_code
    fi
    echo "Rank${SLURM_PROCID} Perf check finished execution with exit code $perf_check_exit_code"
fi

if [ "$pytest_exit_code" -ne 0 ]; then
    final_exit_code=$pytest_exit_code
elif [ "$perf_check_exit_code" -ne 0 ]; then
    final_exit_code=$perf_check_exit_code
else
    final_exit_code=0
fi
echo "Rank${SLURM_PROCID} Final Slurm run finished execution with exit code $final_exit_code"
exit $final_exit_code
