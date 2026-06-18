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

# Build a "fat" enroot sqsh (base image + trtllm pre-installed).
# Called by the CPU fat-builder sbatch job after the base sqsh has already been
# imported/cached. The GPU test job depends on this builder job completing
# (--dependency=afterok) and then uses the fat sqsh directly (SKIP_INSTALL=1).
#
# Usage:
#   fat_build_inline.sh <fat_sqsh_path> <base_sqsh_path> <llm_tarfile_url> <tar_name>
#
# Exits 0 on success (fat sqsh written atomically to <fat_sqsh_path>).
# The caller uses || true so failures are non-fatal; the GPU test job will then
# fall back to the base sqsh + full slurm_install.sh.

set -euo pipefail

FAT_SQSH_PATH="$1"
BASE_SQSH_PATH="$2"
LLM_TARFILE_URL="$3"
TAR_NAME="$4"

FAT_HASH=$(printf '%s' "$LLM_TARFILE_URL" | sha256sum | cut -d' ' -f1 | head -c 16)
FAT_CONTAINER="fat_build_${FAT_HASH}_${SLURM_JOB_ID:-$$}"
WORK_DIR=$(mktemp -d /tmp/fat_build_XXXXXX)
FAT_TMP="${FAT_SQSH_PATH}.${SLURM_JOB_ID:-$$}.tmp"

cleanup() {
    enroot remove -f "$FAT_CONTAINER" 2>/dev/null || true
    rm -rf "$WORK_DIR" "$FAT_TMP" 2>/dev/null || true
}
trap cleanup EXIT

echo "[fat_build] Starting: $FAT_SQSH_PATH"
echo "[fat_build] Base sqsh: $BASE_SQSH_PATH"
echo "[fat_build] LLM tarfile: ${LLM_TARFILE_URL%%\?*}"

# Write the install script into WORK_DIR (mounted as /work inside the container).
# Variable expansion happens here in the outer shell; the container sees literal values.
cat > "$WORK_DIR/install.sh" << INSTALL_EOF
#!/bin/bash
set -euo pipefail
cd /work
echo "[fat_build] Downloading $TAR_NAME..."
wget -nv "$LLM_TARFILE_URL"
tar -zxf "$TAR_NAME"
echo "[fat_build] Installing requirements-dev.txt..."
pip3 install --retries 10 -r TensorRT-LLM/src/requirements-dev.txt
echo "[fat_build] Installing trtllm wheel..."
pip3 install --retries 10 --force-reinstall --no-deps TensorRT-LLM/tensorrt_llm-*.whl
echo "$LLM_TARFILE_URL" > /opt/trtllm_installed.txt
echo "[fat_build] Install complete."
INSTALL_EOF

echo "[fat_build] Creating writable container from base sqsh..."
enroot create --name "$FAT_CONTAINER" "$BASE_SQSH_PATH"

echo "[fat_build] Running install inside container..."
enroot start \
    --rw \
    --mount "$WORK_DIR:/work" \
    "$FAT_CONTAINER" -- bash /work/install.sh

echo "[fat_build] Exporting fat sqsh..."
enroot export --output "$FAT_TMP" "$FAT_CONTAINER"
mv -f "$FAT_TMP" "$FAT_SQSH_PATH"

echo "[fat_build] Done: $(ls -lh "$FAT_SQSH_PATH" | awk '{print $5, $9}')"
