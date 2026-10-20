#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Slurm + pyxis/enroot path (no Docker daemon needed): apply a make_payload.sh payload to the base image's squashfs
# and save the result as a new squashfs, then check the saved image in a second step.
#   docker/qwen35_pareto_repro/save_sqsh.sh <base .sqsh> <payload dir> <out .sqsh>
# Env: SBATCH_ACCOUNT / SLURM_ACCOUNT (or ACCOUNT), PARTITION (default cpu; any aarch64 partition with pyxis works,
#      no GPU needed), TIME (default 01:00:00).
set -euo pipefail
BASE=${1:?usage: save_sqsh.sh <base .sqsh> <payload dir> <out .sqsh>}
P=$(readlink -f "${2:?payload dir}")
OUT=${3:?out .sqsh}
[ -e "$OUT" ] && { echo "$OUT exists; never overwrite an image that jobs may be using" >&2; exit 1; }
ACCOUNT=${ACCOUNT:-${SBATCH_ACCOUNT:-${SLURM_ACCOUNT:?set ACCOUNT}}}
PART=${PARTITION:-cpu}
TIME=${TIME:-01:00:00}
srun -A "$ACCOUNT" -p "$PART" -N 1 -t "$TIME" --job-name=q35_repro_image \
  --container-image="$BASE" --container-remap-root --container-mounts="$P:/q35_payload:ro" \
  --container-save="$OUT" bash /q35_payload/apply_payload.sh /q35_payload
# Check: the saved image carries the payload (sha256 of every file); on a node with a CUDA driver also that the
# package imports (tensorrt_llm.bindings needs libcuda.so.1, which CPU nodes do not have).
srun -A "$ACCOUNT" -p "$PART" -N 1 -t 00:30:00 --job-name=q35_repro_image_check \
  --container-image="$OUT" --container-mounts="$P:/q35_payload:ro" \
  bash -c 'cd /usr/local/lib/python3.12/dist-packages && sha256sum --quiet -c /q35_payload/MANIFEST.sha256 &&
           echo "image files match MANIFEST.sha256" && cat /opt/qwen35_pareto_repro/IMAGE_INFO.txt &&
           if ldconfig -p | grep -q "libcuda.so.1"; then
             python3 -c "import tensorrt_llm.bindings as b; print(\"bindings\", b.__file__); print(\"int32 ingest bridge\", hasattr(b.executor.Request, \"input_token_ids_i32\"))"
           else echo "no CUDA driver on this node: import check skipped (run it on a GPU node)"; fi'
ls -la "$OUT"
