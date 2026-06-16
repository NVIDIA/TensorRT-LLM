#!/bin/bash
# Build a "fat" enroot sqsh (base image + trtllm pre-installed) on the compute node.
# Called from srunPrologue when the fat sqsh for this commit is not yet cached.
#
# Usage:
#   fat_build_inline.sh <fat_sqsh_path> <base_sqsh_path> <llm_tarfile_url> <tar_name>
#
# Exits 0 on success (fat sqsh written to <fat_sqsh_path>), non-zero on any
# failure. The caller wraps this with || true so failures are non-fatal and
# the job falls back to the base sqsh + full slurm_install.sh.

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
echo "[fat_build] LLM tarfile: $LLM_TARFILE_URL"

# Write the install script to WORK_DIR (mounted into container as /work).
# Variable expansion happens here in the outer shell; the container sees
# literal values already substituted. Install steps mirror slurm_install.sh
# exactly so the fat sqsh matches what a regular CI job would produce.
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
