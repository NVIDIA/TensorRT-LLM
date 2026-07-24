#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# cuda_env.sh - SOURCE this to put the CONTAINER's CUDA libraries first on the
# loader path, so the BOLT wheel's dynamically-loaded libnvrtc/libnvJitLink
# resolve to the container's matched (e.g. 13.2) stack rather than any older
# pip cu13 wheels that requirements.txt may have pulled into site-packages.
#
#   source scripts/bolt/cuda_env.sh
#
# Use together with setup_env.sh (which removes the shadowing pip CUDA libs).
# Symptom this prevents: "Jitify fatal error: libnvJitLink.so.13 is too old".

_arch="$(uname -m)"
case "$_arch" in
    aarch64) _triple="sbsa-linux" ;;
    x86_64)  _triple="x86_64-linux" ;;
    *)       _triple="${_arch}-linux" ;;
esac

for _d in "/usr/local/cuda/targets/${_triple}/lib" "/usr/local/cuda/lib64"; do
    if [[ -d "$_d" ]]; then
        export LD_LIBRARY_PATH="${_d}:${LD_LIBRARY_PATH:-}"
    fi
done
unset _arch _triple _d

echo "[INFO] cuda_env: container CUDA libs prepended to LD_LIBRARY_PATH" >&2
