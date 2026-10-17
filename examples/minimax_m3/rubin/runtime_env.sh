#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Source this file in every server and worker process environment to run
# MiniMax-M3 sparse prefill on the native MiniMax MSA wrappers.
#
# Requirements:
# - MSA_ROOT: the checkout prepared by setup_msa.sh.
# - CUDA_HOME: a CUDA Toolkit 13.5 or newer; the patched NVFP4 wrapper
#   compiles its SM107 kernel with it on first use.
# - On SM107, nvidia-cutlass-dsl[cu13] 4.8; the 4.5.2 cu13 backend has no
#   sm_107a target for the wrappers' shared CuTe DSL stages.
# - On SM107, a TensorRT-LLM build that includes SM107 kernels (for example
#   --cuda_architectures "107-real"); the dense 128-token-page attention
#   kernels are generated at runtime with NVRTC. The tested build also used
#   --nvrtc_dynamic_linking.

export MSA_ROOT="${MSA_ROOT:?Set MSA_ROOT to the checkout prepared by setup_msa.sh}"
export CUDA_HOME="${CUDA_HOME:?Set CUDA_HOME to a CUDA Toolkit 13.5 or newer}"
export PYTHONPATH="${MSA_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
# Compile MSA kernels on first use; no ahead-of-time kernel cache is shipped.
export FMHA_SM100_ALLOW_JIT=1
export TRTLLM_MSA_NVDEV_PREFILL=1
