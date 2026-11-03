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

``TLLM_DSPARK_L2_PREFETCH=1`` (:class:`DraftL2Prefetch`) prefetches the
drafter layers' weights into L2 from a side stream; it changes no value.
"""

import functools
import os
from typing import Iterable, Optional

import torch
import torch.nn.functional as F

DRAFT_GEMM_ENV_VAR = "TLLM_DSPARK_DRAFT_GEMM"
L2_PREFETCH_ENV_VAR = "TLLM_DSPARK_L2_PREFETCH"

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


# GB300 (SM103), the TP32 drafter's per-rank shapes, keyed by (N, K); same entry format. Takes precedence over
# ``_PINS`` for these shapes on SM103 (the TP16 shapes keep the role pins). Only tile / stages differ from the role
# pin (same split_k), which keeps every output element's bits (test_dspark_draft_gemm_pins.py). CUDA graph of
# back-to-back GEMMs with PDL over > 256 MB of rotating weights, us per call at 8 / 16 / 32 rows, role pin -> this
# pin:
_PINS_SM103 = {
    # the draft block's LM head shard of the TP32 group: 15.96 / 16.35 / 18.33 -> 13.54 / 13.89 / 14.41.
    (5120, 7168): (
        (8, "splitk", (64, 8, 1, 11)),
        (16, "splitk", (64, 16, 1, 11)),
        (32, "splitk", (64, 32, 1, 9)),
    ),
}


def _pins(role: str, n: int, k: int, device: torch.device):
    if (n, k) in _PINS_SM103 and _sm103(device.index):
        return _PINS_SM103[(n, k)]
    return _PINS.get(role)


@functools.cache
def _sm103(device_index: int) -> bool:
    return torch.cuda.get_device_capability(device_index) == (10, 3)


def draft_gemm_enabled() -> bool:
    """Whether ``TLLM_DSPARK_DRAFT_GEMM=1`` pins the drafter's GEMM kernels."""
    return os.environ.get(DRAFT_GEMM_ENV_VAR, "0") == "1"


def draft_gemm(x: torch.Tensor, weight: torch.Tensor, role: str) -> Optional[torch.Tensor]:
    """``x @ weight.T`` with the kernel pinned for ``role`` at this row count.

    Returns None (the caller keeps its module path) when the switch is off, the
    row count is above the pinned range, or the operands do not fit the kernel.
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
    try:
        from flashinfer.gemm.kernels.dense_bf16_gemm_sm100_splitk import (
            SplitKTactic,
            run_splitk_dense,
        )

        from ..kernels.blackwell.dspark_draft_gemm_push import ensure_cute_runtime
    except ImportError:
        return None

    from ..flashinfer_utils import get_env_enable_pdl

    # detach(): the kernel's DLPack export refuses tensors that require grad,
    # which module parameters do even under inference mode.
    x2d = x.detach().reshape(m, k)
    weight = weight.detach()
    if not x2d.is_contiguous() or k % 128 or x2d.data_ptr() % 32 or weight.data_ptr() % 32:
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
    )
    return out.view(*x.shape[:-1], weight.shape[0])


def l2_prefetch_enabled() -> bool:
    """Whether ``TLLM_DSPARK_L2_PREFETCH=1`` prefetches the drafter's weights into L2."""
    return os.environ.get(L2_PREFETCH_ENV_VAR, "0") == "1"


class DraftL2Prefetch:
    """Pulls upcoming drafter weights into L2 from a side stream.

    A drafter layer streams ~80 MB of projection weights at 7-32 rows, while
    its attention and all-reduces leave HBM idle for ~35 us. After a layer's
    [q_a | kv_a] GEMM, :meth:`prefetch` forks a side stream that issues L2
    prefetches (``k3_l2_prefetch``) for that layer's o_proj, gate_up and down
    weights and the next layer's [q_a | kv_a], so those GEMMs read L2 (GB300,
    14 rows, emulated layer: 23.3 -> 19.0 us of GEMM per layer). Prefetches are
    hints: no value changes. The side-stream kernel depends fully on the GEMM
    it follows, so it does not compete with that GEMM for HBM.

    In the TP16 b2 decode graph (gen-only, GB300) it does not pay off: o_proj,
    gate_up, down and the next [q_a | kv_a] get 3.5-5 us faster per layer, but
    the prefetch traffic delays the kernels issued right after the fork (q_b
    3.4 -> ~7.9 us, the block RoPE and the split-KV reduction by 0.3-1 us), so
    the five layers take ~14 us longer per step. Keep it off unless the fork
    point or the prefetch rate is tuned again.

    :meth:`join` must run on the main stream before the step (or graph capture)
    ends; the kernel after the join loses its programmatic early launch.
    """

    def __init__(self):
        self._stream = None
        self._pending = False

    def prefetch(self, tensors: Iterable[Optional[torch.Tensor]]) -> None:
        from ..kernels.blackwell.k3_l2_prefetch import MAX_RANGES, run_k3_l2_prefetch, tensor_range

        ranges = [tensor_range(t.detach()) for t in tensors if t is not None and t.is_cuda]
        ranges = [r for r in ranges if r[1] > 0][:MAX_RANGES]
        if not ranges:
            return
        if self._stream is None:
            if torch.cuda.is_current_stream_capturing():
                return  # first use inside a capture: skip (hints only)
            self._stream = torch.cuda.Stream()
        main = torch.cuda.current_stream()
        fork = torch.cuda.Event()
        fork.record(main)
        self._stream.wait_event(fork)
        with torch.cuda.stream(self._stream):
            run_k3_l2_prefetch(ranges)
        self._pending = True

    def join(self) -> None:
        if self._pending:
            torch.cuda.current_stream().wait_stream(self._stream)
            self._pending = False
