# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The tp4 target's T2SC index scorer runner: the family ``kernels/m3_index_score_t2sc`` with this target's
T2SC_QOWN switch (opt-in, ``TRTLLM_M31_ENABLE_T2SC_QOWN=1``; read at import), which compiles the early schedule of
FP8 index keys as the ``qown`` kernel."""

from ...kernels import m3_index_score_t2sc as _family
from ...kernels.m3_index_score_t2sc import IndexDecodeScoreKernel, IndexDecodeScoreKernelQown
from ..m31_switches import m31_enabled

__all__ = ["IndexDecodeScoreKernel", "IndexDecodeScoreKernelQown", "IndexScoreRunner", "T2SC_QOWN"]

# Opt-in schedule of the early-issue kernel (read at import; it enters the CuTe AOT key through the environment).
T2SC_QOWN = m31_enabled("T2SC_QOWN")


class IndexScoreRunner(_family.IndexScoreRunner):
    """The family runner with this target's ``QOWN`` and its own compiled kernels."""

    kernel_cache: dict = {}
    QOWN = T2SC_QOWN
