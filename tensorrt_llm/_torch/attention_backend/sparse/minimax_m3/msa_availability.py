# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Availability checks for the MiniMax-M3 MSA sparse attention kernels.

The MSA path runs prefill on the fmha_sm100 kernels bundled with TensorRT-LLM
and decode on the CuTe DSL indexer scorer plus the Triton and trtllm-gen decode
kernels. Both halves are required: there is no fallback from one to the other,
so a missing package makes the path unavailable rather than slower. These
helpers gate backend selection so that fails early with a clear message.
"""

from __future__ import annotations

from tensorrt_llm._utils import get_sm_version, is_sm_100f

from ..minimax_m3_kernels.msa_utils import msa_package_available
from ..minimax_m3_kernels.trtllm_gen_dense_decode import flashinfer_available

# fmha_sm100 runs on the SM100 architecture family (SM100 and SM103). Other
# architectures, including SM120, are not supported.
MSA_PACKAGE = "fmha_sm100"


def ensure_msa_available() -> None:
    """Raise RuntimeError if the MSA sparse attention path cannot run here."""
    # Function-local: msa_indexer reaches the trtllm attention classes through
    # this package's init, which a module-scope import here would cycle with.
    from .msa_indexer import cutedsl_score_runner

    if not msa_package_available():
        raise RuntimeError(
            f"MiniMax-M3 MSA sparse attention requires the {MSA_PACKAGE} kernels "
            "packaged with TensorRT-LLM. Reinstall TensorRT-LLM from a complete build."
        )
    if not is_sm_100f():
        sm_version = get_sm_version()
        raise RuntimeError(
            "MiniMax-M3 MSA sparse attention requires an SM100 or SM103 device, "
            f"but the current device reports SM version {sm_version}."
        )
    if cutedsl_score_runner() is None:
        raise RuntimeError(
            "MiniMax-M3 MSA sparse attention scores decode steps on the CuTe DSL "
            "indexer kernel, which requires the nvidia-cutlass-dsl package. "
            "Install it or select the 'triton' implementation."
        )
    if not flashinfer_available():
        raise RuntimeError(
            "MiniMax-M3 MSA sparse attention runs its dense layers through the "
            "trtllm-gen decode kernel, which requires flashinfer. Install it or "
            "select the 'triton' implementation."
        )


__all__ = [
    "MSA_PACKAGE",
    "ensure_msa_available",
]
