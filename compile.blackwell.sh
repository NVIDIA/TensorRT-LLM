#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -uex

PARTITION=${1:-batch}
CLEAN=${2:---clean}

ACCOUNT=${ACCOUNT:-coreai_comparch_inferencex}
REPO_DIR=$(readlink -f "$PWD")

# Mount layout is cluster-specific. hecate keeps everything under /lustre and
# the build has historically mounted the whole filesystem in. oci-jhb has no
# /lustre/share at all -- there /lustre is a symlink to /scratch and the real
# lustre mount is /scratch/fsw -- so "/lustre:/lustre" resolves to a target the
# base image does not have. enroot then auto-creates the missing target as a
# directory the submitting user cannot traverse, and that single directory makes
# --container-save abort with "find: Permission denied / pyxis: failed to export
# container", leaving a successful build with no image (measured on hecate
# 2026-09-18). /workspace exists in the base image, and the build itself needs
# nothing but the repo: every path in the container command below is relative to
# WORKDIR.
#
# The presence of /lustre/share tells the two clusters apart.
if [[ -d /lustre/share ]]; then
    MOUNT="/lustre:/lustre,${REPO_DIR}:/code/tensorrt_llm"
    WORKDIR="/code/tensorrt_llm"
else
    MOUNT="${REPO_DIR}:/workspace"
    WORKDIR="/workspace"
fi
MOUNT=${TRTLLM_BUILD_MOUNT:-${MOUNT}}
WORKDIR=${TRTLLM_BUILD_WORKDIR:-${WORKDIR}}

source "${REPO_DIR}/jenkins/current_image_tags.properties"
IMAGE=${TRTLLM_TEST_IMAGE:-${LLM_SBSA_DOCKER_IMAGE}}

HASH=$(git -C "${REPO_DIR}" rev-parse --short HEAD)
BUILD_DIR="${REPO_DIR}/build_images/${HASH}-blackwell"
JOB_NAME="${ACCOUNT}-blackwell-build.${HASH}"

if [[ -d ${BUILD_DIR} ]]; then
    rm -r "${BUILD_DIR}"
fi
mkdir -p "${BUILD_DIR}"

rm "${REPO_DIR}"/build/tensorrt_llm-*.whl || true

srun -A "${ACCOUNT}" -p "${PARTITION}" --job-name="${JOB_NAME}" --container-image="${IMAGE}" -G 4 \
    --container-mounts="${MOUNT}" \
    --container-workdir="${WORKDIR}" \
    --container-save="${BUILD_DIR}/trtllm.sqsh" \
    -N 1 --ntasks=1 --ntasks-per-node=1 -t 2:00:00 \
    bash -c "python3 -m pip install --upgrade -r requirements-dev.txt && python3 -m pip install --no-deps --force-reinstall 'flashinfer-python==0.6.16' 'nvidia-cutlass-dsl[cu13]==4.7.1' && python scripts/build_wheel.py -G Ninja -a '103-real' ${CLEAN} && pip install build/tensorrt_llm-*.whl"

cp "${REPO_DIR}"/build/tensorrt_llm-*.whl "${BUILD_DIR}/"
