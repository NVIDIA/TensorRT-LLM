# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Body of the fat-sqsh SLURM sbatch job. Called from a thin generated wrapper
# that sets the following required environment variables:
#
#   FAT_CONTAINER_DIR       - scratch path for enroot base .sqsh files
#   FAT_CONTAINER           - docker URI (urm.nvidia.com#...) for the base image
#   FAT_SQSH_DIR            - scratch path where fat .sqsh files are stored
#   FAT_LLM_TARFILE         - HTTPS URL of the TRT-LLM wheel tarball
#   FAT_LLM_DOCKER_IMAGE    - full docker image tag (used for cache-key hashing)
#   FAT_BUILD_SCRIPT_PATH   - path to fat_build_inline.sh on the SLURM node
#   FAT_TAR_NAME            - tarball filename (e.g. TensorRT-LLM.tar.gz)

set -euo pipefail
trap 'rc=$?; echo "Error on line $LINENO: exit $rc"; exit $rc' ERR
export ENROOT_CACHE_PATH='/home/svc_tensorrt/.cache/enroot'

# Resolve base sqsh using image digest (shared cache, same logic as srunPrologue).
mkdir -p "${FAT_CONTAINER_DIR}"
if printf '%s' "${FAT_CONTAINER}" | grep -q '@sha256:'; then
    imageDigest=$(printf '%s' "${FAT_CONTAINER}" | grep -oP '(?<=@sha256:)[a-f0-9]+')
else
    imageDigest=$(printf '%s' "${FAT_CONTAINER}" | sha256sum | cut -d' ' -f1)
fi
baseSqshPath="${FAT_CONTAINER_DIR}/container-${imageDigest}.sqsh"

# Import base sqsh if not already cached (flock so concurrent builds share one import).
importContainerWithRetries() {
    local docker_uri=$1
    local output_path=$2
    local max_attempts=${3:-3}
    local delay=${4:-60}
    local attempt=1
    local tmp_path
    exec 9>"${output_path}.lock" || true
    flock 9 || true
    if [ -f "$output_path" ]; then
        echo "[fat_build_sbatch] Reusing cached base sqsh: $output_path"
        touch "$output_path" || true
        flock -u 9 || true
        return 0
    fi
    tmp_path="${output_path}.${SLURM_JOB_ID:-$$}.tmp"
    rm -f "$tmp_path"
    until enroot import -o "$tmp_path" -- "docker://$docker_uri"; do
        if (( attempt >= max_attempts )); then
            echo "[fat_build_sbatch] enroot import failed after $max_attempts attempts"
            rm -f "$tmp_path"
            flock -u 9 || true
            return 1
        fi
        echo "[fat_build_sbatch] enroot import failed (attempt $attempt of $max_attempts). Retrying in ${delay}s..."
        rm -f "$tmp_path"
        sleep $delay
        attempt=$((attempt + 1))
    done
    mv -f "$tmp_path" "$output_path"
    flock -u 9 || true
}
importContainerWithRetries "${FAT_CONTAINER}" "$baseSqshPath"

# Build fat sqsh from cached base sqsh.
mkdir -p "${FAT_SQSH_DIR}"
fatHash=$(printf '%s' "${FAT_LLM_TARFILE}|${FAT_LLM_DOCKER_IMAGE}" | sha256sum | cut -d' ' -f1 | head -c 16)
fatSqshPath="${FAT_SQSH_DIR}/fat-${fatHash}.sqsh"
if [ -f "$fatSqshPath" ]; then
    echo "[fat_build_sbatch] Fat sqsh already exists: $fatSqshPath"
    exit 0
fi
bash "${FAT_BUILD_SCRIPT_PATH}" "$fatSqshPath" "$baseSqshPath" "${FAT_LLM_TARFILE}" "${FAT_TAR_NAME}" || echo "[fat_build_sbatch] Build failed (non-fatal); GPU job will fall back to base sqsh + full install"
