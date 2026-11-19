#!/bin/bash
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

# Redis and ModelExpress server for the ModelExpress donor/receiver tests of a single-node sbatch
# job. Source this file from the sbatch launch script, after the job environment is exported and
# before the test step, then call `mx_services_start`.
#
# The two services run as `srun --overlap` steps beside the test step, from the same images as
# the Kubernetes sidecars, and listen on loopback only. A pyxis/enroot step shares the host
# network, so the ports are chosen per job from a range and checked to be free instead of using
# the fixed sidecar ports. Jobs that share a host pick their ports and start their services in
# turn, under a host-wide lock, and a service counts as ready only while its own step runs and,
# where the host can tell, a process of this job listens on its port.
#
# Inputs (environment):
#   jobWorkspace            job workspace on the shared filesystem; the logs go under it
#   MX_REDIS_IMAGE          Redis image; `host#path:tag` form when MX_SERVICES_ENROOT=1
#   MX_SERVER_IMAGE         ModelExpress server image, in the same form
#   MX_SERVICES_ENROOT      1 to import both images to squashfs files first, 0 to let pyxis pull
#   MX_CONTAINER_DIR        directory of the cached squashfs files (MX_SERVICES_ENROOT=1)
#   MX_ENROOT_CONFIG_PATH   enroot config directory holding the registry credentials; removed
#                           once the images are imported (MX_SERVICES_ENROOT=1)
#
# Outputs (exported): MODEL_EXPRESS_URL, TRTLLM_MX_E2E_REQUIRED, MX_METADATA_PORT.
# Logs: $jobWorkspace/model_express_services/{redis,server}.log and ports.env.

mxRedisPid=""
mxServerPid=""
mxPortLockFile=/tmp/trtllm-model-express-ports.lock

# A connection to the port succeeds only while something listens on it.
mx_port_is_free() {
    ! (echo > "/dev/tcp/127.0.0.1/$1") > /dev/null 2>&1
}

# Takes the host-wide lock that a job holds from picking its ports until its services listen on
# them, so that two jobs on one host cannot both find a range free before either binds it. Best
# effort: without the lock, the ports are still checked to be free.
mx_port_lock() {
    (umask 0 && : >> "$mxPortLockFile") 2> /dev/null || true
    if exec 9>> "$mxPortLockFile"; then
        flock -w 900 9 || echo "WARNING: could not take $mxPortLockFile; picking ports without it"
    else
        echo "WARNING: could not open $mxPortLockFile; picking ports without it"
    fi
}

mx_port_unlock() {
    flock -u 9 2> /dev/null || true
    exec 9>&- || true
}

# Succeeds when a process of this Slurm job listens on the port, judged from the per-job cgroups
# of the listening processes that `ss` names. Where the host cannot tell (no per-job cgroups, or
# `ss` names no listener), it succeeds too, and mx_wait_for_port relies on the step check alone.
mx_port_owned_by_job() {
    local port=$1
    local job="job_${SLURM_JOB_ID:-}/"
    local pids pid
    if [ -z "${SLURM_JOB_ID:-}" ] || ! grep -q "$job" /proc/self/cgroup 2> /dev/null; then
        return 0
    fi
    pids=$(ss -Hltnp "sport = :$port" 2> /dev/null | grep -o 'pid=[0-9]*' | cut -d= -f2) || true
    if [ -z "$pids" ]; then
        return 0
    fi
    for pid in $pids; do
        if grep -q "$job" "/proc/$pid/cgroup" 2> /dev/null; then
            return 0
        fi
    done
    return 1
}

# Reserves ten consecutive ports starting at a job-derived base: the Redis port, the server
# port, and eight ports for the NIXL metadata bases (donor `base`, receiver `base + tp`, each
# binding one port per rank). The range stays below the Linux ephemeral port range. Call it
# under mx_port_lock: a range whose Redis and server ports another job holds is skipped.
mx_pick_ports() {
    local attempt offset candidate ok
    for attempt in $(seq 0 49); do
        candidate=$((10000 + ((${SLURM_JOB_ID:-0} + attempt) % 2000) * 10))
        ok=1
        for offset in $(seq 0 9); do
            if ! mx_port_is_free $((candidate + offset)); then
                ok=0
                break
            fi
        done
        if [ "$ok" -eq 1 ]; then
            mxRedisPort=$candidate
            mxServerPort=$((candidate + 1))
            mxNixlPort=$((candidate + 2))
            return 0
        fi
    done
    echo "ERROR: no free ModelExpress port range found"
    return 1
}

# Imports a docker image to a squashfs file; a file that is already there is reused.
mx_import_image() {
    local uri=$1
    local output=$2
    local maxAttempts=3
    local attempt=1
    local tmpFile="${output}.${SLURM_JOB_ID:-0}.tmp"
    local tmpDir

    mkdir -p "$(dirname "$output")"
    # Best-effort lock so racing jobs do not all import the same image.
    exec 8> "${output}.lock" || true
    flock 8 || true
    if [ -f "$output" ]; then
        echo "Reusing cached container image: $output"
        touch "$output" || true
        flock -u 8 || true
        return 0
    fi

    # A job-private unpack directory keeps the import away from directories that the prolog of
    # a new job can chown to another user.
    tmpDir=$(mktemp -d "/tmp/.enroot-import-mx.XXXXXX")
    rm -f "$tmpFile"
    until ENROOT_CONFIG_PATH="$MX_ENROOT_CONFIG_PATH" ENROOT_TEMP_PATH="$tmpDir" \
        enroot import -o "$tmpFile" -- "docker://$uri"; do
        if [ "$attempt" -ge "$maxAttempts" ]; then
            echo "enroot import of $uri failed after $maxAttempts attempts"
            rm -rf "$tmpFile" "$tmpDir"
            flock -u 8 || true
            return 1
        fi
        echo "enroot import of $uri failed (attempt $attempt of $maxAttempts). Retrying in 60s..."
        rm -f "$tmpFile"
        sleep 60
        attempt=$((attempt + 1))
    done
    rm -rf "$tmpDir"
    mv -f "$tmpFile" "$output"
    flock -u 8 || true
}

# Prints the value for `--container-image`: the squashfs file of the imported image, or the
# image reference itself when pyxis pulls it.
mx_resolve_image() {
    local uri=$1
    local sqsh
    if [ "${MX_SERVICES_ENROOT:-0}" != "1" ]; then
        echo "$uri"
        return 0
    fi
    sqsh="$MX_CONTAINER_DIR/container-$(printf '%s' "$uri" | sha256sum | cut -d' ' -f1).sqsh"
    mx_import_image "$uri" "$sqsh" >&2
    echo "$sqsh"
}

# Waits until the service of the given step listens on the port. Fails when the step exits
# first, when the timeout passes, or when the step exits right after the port starts answering:
# a service that cannot bind its port exits at startup, so the listener then belongs to another
# process.
mx_wait_for_port() {
    local name=$1
    local port=$2
    local pid=$3
    local logFile=$4
    local deadline=$((SECONDS + ${5:-300}))
    while mx_port_is_free "$port" || ! mx_port_owned_by_job "$port"; do
        if ! kill -0 "$pid" 2> /dev/null; then
            echo "ERROR: the $name step exited before listening on port $port"
            tail -n 50 "$logFile" || true
            return 1
        fi
        if [ "$SECONDS" -ge "$deadline" ]; then
            echo "ERROR: the $name step did not listen on port $port in time"
            tail -n 50 "$logFile" || true
            return 1
        fi
        sleep 2
    done
    sleep 5
    if ! kill -0 "$pid" 2> /dev/null; then
        echo "ERROR: the $name step exited although port $port answers; another process holds it"
        tail -n 50 "$logFile" || true
        return 1
    fi
    echo "$name is listening on port $port"
}

# Stops both services. It is the job's EXIT trap once the services are started, so it runs on
# every exit path and leaves the exit status alone.
mx_services_stop() {
    local pid
    for pid in "$mxServerPid" "$mxRedisPid"; do
        if [ -n "$pid" ]; then
            kill "$pid" 2> /dev/null || true
        fi
    done
    wait 2> /dev/null || true
}

mx_services_start() {
    local logDir="$jobWorkspace/model_express_services"
    local redisImage serverImage
    mkdir -p "$logDir"

    redisImage=$(mx_resolve_image "$MX_REDIS_IMAGE")
    serverImage=$(mx_resolve_image "$MX_SERVER_IMAGE")
    if [ "${MX_SERVICES_ENROOT:-0}" = "1" ]; then
        rm -rf "$MX_ENROOT_CONFIG_PATH"
    fi

    trap mx_services_stop EXIT

    # The service steps close the lock's descriptor, so the lock ends with mx_port_unlock.
    mx_port_lock
    mx_pick_ports || { mx_port_unlock; return 1; }

    echo "Starting Redis on 127.0.0.1:$mxRedisPort"
    env -u BASH_ENV srun --overlap --nodes=1 --ntasks=1 \
        --container-image="$redisImage" --no-container-mount-home \
        redis-server --bind 127.0.0.1 --port "$mxRedisPort" --save "" --appendonly no \
        > "$logDir/redis.log" 2>&1 9>&- &
    mxRedisPid=$!
    mx_wait_for_port redis "$mxRedisPort" "$mxRedisPid" "$logDir/redis.log" \
        || { mx_port_unlock; return 1; }

    echo "Starting the ModelExpress server on 127.0.0.1:$mxServerPort"
    env -u BASH_ENV srun --overlap --nodes=1 --ntasks=1 \
        --container-image="$serverImage" --container-workdir=/app --no-container-mount-home \
        /bin/bash -c 'MX_METADATA_BACKEND=redis REDIS_URL="redis://127.0.0.1:$1" exec /app/modelexpress-server --host 127.0.0.1 --port "$2"' \
        _ "$mxRedisPort" "$mxServerPort" \
        > "$logDir/server.log" 2>&1 9>&- &
    mxServerPid=$!
    mx_wait_for_port model-express-server "$mxServerPort" "$mxServerPid" "$logDir/server.log" \
        || { mx_port_unlock; return 1; }
    mx_port_unlock

    export MODEL_EXPRESS_URL="http://127.0.0.1:$mxServerPort"
    export TRTLLM_MX_E2E_REQUIRED=1
    export MX_METADATA_PORT="$mxNixlPort"
    {
        echo "MODEL_EXPRESS_URL=$MODEL_EXPRESS_URL"
        echo "MX_METADATA_PORT=$MX_METADATA_PORT"
        echo "REDIS_PORT=$mxRedisPort"
    } > "$logDir/ports.env"
}
