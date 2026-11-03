# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The routed experts' output before the top-k weighted sum (no CuTe DSL dependency)."""

from __future__ import annotations

from typing import NamedTuple

import torch


class UnfinalizedRoutedOutput(NamedTuple):
    """Routed-expert output before the top-k weighted sum.

    ``gemm2_permuted`` is ``[permuted rows, latent]`` BF16, ``expert_weights``
    ``[M, top_k]`` BF16 and ``expanded_idx_to_permuted_idx`` ``[M, top_k]``
    int32 (negative for routes this rank does not own).
    """

    gemm2_permuted: torch.Tensor
    expert_weights: torch.Tensor
    expanded_idx_to_permuted_idx: torch.Tensor
