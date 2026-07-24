#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Submits a CPU fat-sqsh builder sbatch job (if no fat sqsh is cached yet) and
# polls until it completes. Called from a thin generated wrapper that sets the
# following required environment variables:
#
#   FAT_SQSH_DIR            - scratch path where fat .sqsh files are stored
#   FAT_LLM_TARFILE         - HTTPS URL of the TRT-LLM wheel tarball (for cache-key hashing)
#   FAT_LLM_DOCKER_IMAGE    - full docker image tag (for cache-key hashing)
#   FAT_BUILD_SBATCH_PATH   - path to the fat_build_sbatch.sh wrapper on the login node
#   FAT_BUILD_LOG_TEMPLATE  - SLURM log path template (may contain %j for job ID)

set -euxo pipefail
echo "=== [Prepare Container] STAGE START: $(date '+%Y-%m-%d %H:%M:%S') on $(hostname) ==="

if ! mkdir -p "${FAT_SQSH_DIR}" 2>&1; then
    echo "Warning: mkdir -p ${FAT_SQSH_DIR} failed; fat sqsh will not be built on this run."
    echo "=== [Prepare Container] STAGE END (mkdir failed): $(date '+%Y-%m-%d %H:%M:%S') ==="
    exit 0
fi

fatHash=$(printf '%s' "${FAT_LLM_TARFILE}|${FAT_LLM_DOCKER_IMAGE}" | sha256sum | cut -d' ' -f1 | head -c 16)
fatSqshPath="${FAT_SQSH_DIR}/fat-${fatHash}.sqsh"

if [ -f "$fatSqshPath" ]; then
    echo "Fat sqsh already cached: $fatSqshPath"
    echo "=== [Prepare Container] STAGE END (cache hit): $(date '+%Y-%m-%d %H:%M:%S') fat_sqsh=$fatSqshPath ==="
    exit 0
fi

BUILDER_ID=""
STATUS=""
actualFatBuildLogPath=""
FAT_BUILD_JOB_NAME="fat_build_${fatHash}"
EXISTING=$(squeue -h -n "$FAT_BUILD_JOB_NAME" -o "%i" 2>/dev/null | head -1 || true)
if [ -n "$EXISTING" ]; then
    echo "Fat sqsh builder already running as job $EXISTING, will wait for it"
    BUILDER_ID="$EXISTING"
    # Locate the log file of the existing job (submitted by a different Jenkins run with a different jobUID)
    actualFatBuildLogPath=$(scontrol show job "$BUILDER_ID" 2>/dev/null | grep -oP '(?<=StdOut=)\S+' || true)
else
    # Submit the CPU builder. Best-effort: any submission failure is non-fatal;
    # the GPU job will fall back to base sqsh + full install.
    set +e
    BUILDER_OUT=$(sbatch --parsable --job-name="$FAT_BUILD_JOB_NAME" --dependency=singleton "${FAT_BUILD_SBATCH_PATH}" 2>&1)
    BUILDER_RC=$?
    set -e
    # sbatch may print unrelated WARNINGs to stderr (captured via 2>&1); --parsable's real
    # output is a line "<jobid>" or "<jobid>;<cluster>". Extract the numeric job id only.
    BUILDER_ID=$(printf '%s\n' "$BUILDER_OUT" | grep -oE '^[0-9]+' | tail -1)
    if [ "$BUILDER_RC" -eq 0 ] && [ -n "$BUILDER_ID" ]; then
        echo "Submitted fat sqsh builder as cpu_builder_job_id=$BUILDER_ID at $(date '+%Y-%m-%d %H:%M:%S')"
        _fatLogTemplate="${FAT_BUILD_LOG_TEMPLATE}"
        actualFatBuildLogPath="${_fatLogTemplate//%j/$BUILDER_ID}"
    else
        echo "Fat sqsh builder submission failed (rc=$BUILDER_RC): $BUILDER_OUT"
        echo "GPU job will fall back to base sqsh + full install"
        echo "=== [Prepare Container] STAGE END (builder submission failed): $(date '+%Y-%m-%d %H:%M:%S') ==="
        exit 0
    fi
fi

# Poll until the builder job finishes (success or failure); tail the builder log live.
echo "Waiting for fat sqsh builder job $BUILDER_ID to complete..."
FAT_TAIL_PID=""
if [ -n "$actualFatBuildLogPath" ]; then
    touch "$actualFatBuildLogPath" 2>/dev/null || true
    tail -F -n +1 "$actualFatBuildLogPath" &
    FAT_TAIL_PID=$!
else
    echo "Warning: could not determine fat build log path; builder output will not be shown here"
fi
while true; do
    if ! STATUS=$(sacct -j "$BUILDER_ID" --format=State -Pn --allocations 2>&1); then
        echo "Warning: sacct failed, retrying in 60s..."
        sleep 60
        continue
    fi
    case "${STATUS:-}" in
        COMPLETED|FAILED|CANCELLED|TIMEOUT|NODE_FAIL|OUT_OF_MEMORY)
            echo "Fat sqsh builder job $BUILDER_ID finished: $STATUS"
            break
            ;;
        *)
            echo "Fat sqsh builder job $BUILDER_ID state: ${STATUS:-UNKNOWN}, waiting..."
            sleep 60
            ;;
    esac
done
[ -n "$FAT_TAIL_PID" ] && kill $FAT_TAIL_PID 2>/dev/null || true

if [ -f "$fatSqshPath" ]; then
    echo "Fat sqsh ready: $fatSqshPath"
else
    echo "Fat sqsh not found after builder completed (status=$STATUS); GPU job will fall back to base sqsh + full install"
fi
echo "=== [Prepare Container] STAGE END: $(date '+%Y-%m-%d %H:%M:%S') cpu_builder_job_id=$BUILDER_ID final_status=$STATUS ==="
