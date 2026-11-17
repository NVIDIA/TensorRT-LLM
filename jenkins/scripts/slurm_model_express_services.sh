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
# the Kubernetes sidecars. A pyxis/enroot step shares the host network, so the ports are chosen
# per job from a range and checked to be free instead of using the fixed sidecar ports.
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

# A connection to the port succeeds only while something listens on it.
mx_port_is_free() {
    ! (echo > "/dev/tcp/127.0.0.1/$1") > /dev/null 2>&1
}

# Reserves ten consecutive ports starting at a job-derived base: the Redis port, the server
# port, and eight ports for the NIXL metadata bases (donor `base`, receiver `base + tp`, each
# binding one port per rank). The range stays below the Linux ephemeral port range.
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

# Waits until something listens on the port; fails when the step that should listen exits first
# or the timeout passes.
mx_wait_for_port() {
    local name=$1
    local port=$2
    local pid=$3
    local logFile=$4
    local deadline=$((SECONDS + ${5:-300}))
    while mx_port_is_free "$port"; do
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

    mx_pick_ports
    redisImage=$(mx_resolve_image "$MX_REDIS_IMAGE")
    serverImage=$(mx_resolve_image "$MX_SERVER_IMAGE")
    if [ "${MX_SERVICES_ENROOT:-0}" = "1" ]; then
        rm -rf "$MX_ENROOT_CONFIG_PATH"
    fi

    trap mx_services_stop EXIT

    echo "Starting Redis on port $mxRedisPort"
    env -u BASH_ENV srun --overlap --nodes=1 --ntasks=1 \
        --container-image="$redisImage" --no-container-mount-home \
        redis-server --bind 127.0.0.1 --port "$mxRedisPort" --save "" --appendonly no \
        > "$logDir/redis.log" 2>&1 &
    mxRedisPid=$!
    mx_wait_for_port redis "$mxRedisPort" "$mxRedisPid" "$logDir/redis.log"

    echo "Starting the ModelExpress server on port $mxServerPort"
    env -u BASH_ENV srun --overlap --nodes=1 --ntasks=1 \
        --container-image="$serverImage" --container-workdir=/app --no-container-mount-home \
        /bin/bash -c 'MX_METADATA_BACKEND=redis REDIS_URL="redis://127.0.0.1:$1" exec /app/modelexpress-server --port "$2"' \
        _ "$mxRedisPort" "$mxServerPort" \
        > "$logDir/server.log" 2>&1 &
    mxServerPid=$!
    mx_wait_for_port model-express-server "$mxServerPort" "$mxServerPid" "$logDir/server.log"

    export MODEL_EXPRESS_URL="http://127.0.0.1:$mxServerPort"
    export TRTLLM_MX_E2E_REQUIRED=1
    export MX_METADATA_PORT="$mxNixlPort"
    {
        echo "MODEL_EXPRESS_URL=$MODEL_EXPRESS_URL"
        echo "MX_METADATA_PORT=$MX_METADATA_PORT"
        echo "REDIS_PORT=$mxRedisPort"
    } > "$logDir/ports.env"
}
