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
"""Pinned GEMM kernels for the DSpark MLA drafter's BF16 projections.

The drafter's projections run at 7 to 32 token rows. The generic low-M BF16
dispatcher picks a kernel per shape from an isolated timing race, which does
not see what the decode graph sees: cold weights (the drafter streams ~80 MB
per layer) and a PDL head start from the predecessor. Under those conditions
(measured on GB300 as predecessor RMSNorm + GEMM in a CUDA graph with PDL,
weights rotated beyond L2) the low-M split-K tensor-core kernel at the tactics
below beats the dispatcher's in-graph choices -- the SIMT direct kernel for
q_b runs 6.8 us in the decode graph against 2.2 us here, and the fused
[q_a | kv_a] projection, created after the dispatcher labelled the modules,
falls back to cuBLAS (8-10 us against 6.6 us).

``TLLM_DSPARK_DRAFT_GEMM=1`` selects them. They change the drafter's GEMM
accumulation order (split-K partial sums), i.e. draft-token rounding only; the
target's verification keeps the output distribution unchanged.
"""

import functools
from typing import Optional

import torch
import torch.nn.functional as F

from ..features import active as _features

# role -> ((max rows, kernel, split-K tactic (mma_m, mma_n, split_k, ab_stages)), ...)
_PINS = {
    # [q_a | kv_a]: 2112 x 7168
    "q_kv_a": (
        (8, "splitk", (64, 16, 4, 8)),
        (16, "splitk", (64, 16, 4, 9)),
        (32, "splitk", (64, 16, 2, 11)),
    ),
    # q_b: 768 x 1536 per rank
    "q_b": ((16, "splitk", (64, 8, 4, 10)), (32, "splitk", (64, 16, 3, 10))),
    # gate_up: 1792 x 7168 per rank
    "gate_up": ((16, "splitk", (64, 16, 4, 8)), (32, "splitk", (64, 32, 4, 8))),
    # down: 7168 x 896 per rank
    "down": ((32, "cublas", None),),
    # context latents of all layers: 2880 x 7168
    "ctx_latent": (
        (8, "splitk", (128, 8, 4, 6)),
        (16, "splitk", (64, 16, 2, 10)),
        (32, "splitk", (64, 32, 2, 9)),
    ),
    # the draft block's LM head shard: 10240 x 7168
    "lm_head": ((16, "splitk", (128, 16, 1, 6)), (32, "splitk", (128, 32, 1, 5))),
}


# Rubin (SM107), the helix16 drafter's per-rank shapes, keyed by (N, K); same entry format. Takes
# precedence over ``_PINS`` for these shapes on SM107. Measured as CUDA graphs of 12 back-to-back
# GEMMs with PDL over > L2 of rotating weights, us per call, cuBLAS -> pinned split-K at 7 / 14 /
# 28 rows.
_PINS_SM107 = {
    # [q_a | kv_a] (replicated): 6.46 / 6.48 / 7.17 -> 3.95 / 4.07 / 4.95.
    (2112, 7168): (
        (8, "splitk", (64, 8, 4, 12)),
        (16, "splitk", (64, 16, 4, 10)),
        (32, "splitk", (64, 32, 4, 8)),
    ),
    # context latents of all layers (replicated): 6.79 / 7.45 / 7.15 -> 4.61 / 4.71 / 5.38.
    (2880, 7168): (
        (8, "splitk", (64, 8, 2, 12)),
        (16, "splitk", (64, 16, 4, 10)),
        (32, "splitk", (64, 32, 4, 8)),
    ),
    # The MLP and LM head shards of the TP16 group (HBM locked): gate_up 6.02 / 6.11 / 5.72 -> 3.66 /
    # 3.86 / 4.54; down 4.22 / 3.31 / 3.37 -> 2.50 / 2.48 / 2.60; LM head 15.65 / 14.56 / 16.09 ->
    # 14.12 / 14.14 / 14.45.
    # gate_up: 2 x 896
    (1792, 7168): (
        (8, "splitk", (64, 8, 4, 12)),
        (16, "splitk", (64, 16, 4, 10)),
        (32, "splitk", (64, 32, 4, 8)),
    ),
    # down
    (7168, 896): (
        (8, "splitk", (64, 8, 1, 12)),
        (16, "splitk", (64, 16, 1, 11)),
        (32, "splitk", (64, 32, 1, 9)),
    ),
    # the draft block's LM head shard
    (10240, 7168): (
        (8, "splitk", (128, 8, 2, 6)),
        (16, "splitk", (64, 16, 1, 11)),
        (32, "splitk", (64, 32, 1, 9)),
    ),
    # q_b of a helix drafter (latent pages sharded over the group, every head on every rank):
    # 64 heads x 192. cuBLAS -> pinned at 8 / 16 / 32 rows: 4.75 / 4.79 / 6.65 -> 4.26 / 4.34 /
    # 4.49 us (one rank, HBM locked).
    (12288, 1536): (
        (8, "splitk", (64, 8, 1, 12)),
        (16, "splitk", (64, 16, 1, 11)),
        (32, "splitk", (64, 32, 1, 9)),
    ),
}


def _pins(role: str, n: int, k: int, device: torch.device):
    if (n, k) in _PINS_SM107 and _sm107(device.index):
        return _PINS_SM107[(n, k)]
    return _PINS.get(role)


@functools.cache
def _sm107(device_index: int) -> bool:
    return torch.cuda.get_device_capability(device_index) == (10, 7)


def draft_gemm_enabled() -> bool:
    """Whether ``TLLM_DSPARK_DRAFT_GEMM=1`` pins the drafter's GEMM kernels."""
    return _features().dspark_draft_gemm


def draft_gemm(
    x: torch.Tensor, weight: torch.Tensor, role: str, strided: bool = False
) -> Optional[torch.Tensor]:
    """``x @ weight.T`` with the kernel pinned for ``role`` at this row count.

    Returns None (the caller keeps its module path) when the switch is off, the
    row count is above the pinned range, or the operands do not fit the kernel.
    ``strided``: a 2-D ``x`` may be a row-major view with a 16-byte-aligned row
    stride (e.g. the q_a columns of the [q_a | kv_a] rows), read in place.
    """
    if not draft_gemm_enabled() or weight.dim() != 2 or not x.is_cuda:
        return None
    k = x.shape[-1]
    m = x.numel() // k if k else 0
    pins = _pins(role, weight.shape[0], k, x.device)
    pin = next((p for p in pins or () if 0 < m <= p[0]), None)
    if (
        pin is None
        or x.dtype != torch.bfloat16
        or weight.dtype != torch.bfloat16
        or weight.dim() != 2
        or weight.shape[1] != k
        or not weight.is_contiguous()
        or not x.is_cuda
    ):
        return None
    _, kind, tactic = pin
    if kind == "cublas":
        return F.linear(x, weight)
    from tensorrt_llm._torch.flashinfer_utils import get_env_enable_pdl

    from ..kernels.dspark_draft_gemm_push import ensure_cute_runtime
    from ..kernels.low_m_bf16_splitk import SplitKTactic, run_splitk_dense

    # detach(): the kernel's DLPack export refuses tensors that require grad,
    # which module parameters do even under inference mode.
    strided = strided and x.dim() == 2 and x.stride(1) == 1 and x.stride(0) % 8 == 0
    x2d = x.detach() if strided else x.detach().reshape(m, k)
    weight = weight.detach()
    if (
        not (strided or x2d.is_contiguous())
        or k % 128
        or x2d.data_ptr() % 32
        or weight.data_ptr() % 32
    ):
        return None
    out = torch.empty((m, weight.shape[0]), dtype=x.dtype, device=x.device)
    ensure_cute_runtime()
    run_splitk_dense(
        x2d,
        weight.t(),
        None,
        out,
        get_env_enable_pdl(),
        SplitKTactic(*tactic),
        strided_a=strided and not x2d.is_contiguous(),
    )
    return out.view(*x.shape[:-1], weight.shape[0])
