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
"""``trtllm::k3_moe_front``: the Kimi K3 MoE front at decode size in one kernel (``k3_moe_front.py``).

Replaces, per MoE layer: the sharded head GEMV, ``trtllm::k3_route_quant_ag`` (head all-gather, top-16 routing, MXFP8
latent) and, on the shared-expert stream, the shared gate_up GEMV and SiTU-and-mul. The head all-gather uses
``op.head_workspace``'s buffers and protocol, so the front and ``k3_route_quant_ag`` must not both serve one layer.
The kernel compiles on the first call for each configuration, which must happen outside CUDA-graph capture.
"""

from __future__ import annotations

import functools
import os
import threading
from typing import Dict, Optional, Tuple

import torch

TOP_K = 16
HIDDEN_SIZE = 3584
NUM_EXPERTS = 896
SF_VEC = 32
MAX_TOKENS = 8
DEFAULT_RING = 4

_lock = threading.Lock()
_compiled: Dict[tuple, object] = {}


def _kernel():
    from . import k3_moe_front as kernel

    return kernel


def front_weight(
    head_weight: torch.Tensor, gate_up_weight: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """The front's one weight: ``head_weight`` (``[3584/W + 896/W, K]``: this rank's latent-down rows, then its router
    rows) zero-padded to whole 128-row tiles, then ``gate_up_weight`` (``[2 I, K]``: gate rows, then up rows) with
    every 32 rows holding 16 gate rows and the 16 up rows of the same columns. Without ``gate_up_weight`` the front
    computes the head alone (``shared_cols`` 0)."""
    rows, k_in = head_weight.shape
    if gate_up_weight is None:
        gate_up_weight = head_weight[:0]
    inter = gate_up_weight.shape[0] // 2
    if (
        gate_up_weight.shape[1] != k_in
        or inter % 64 != 0
        or head_weight.dtype != gate_up_weight.dtype
    ):
        raise ValueError(
            f"k3_moe_front weight: head {tuple(head_weight.shape)}, gate_up {tuple(gate_up_weight.shape)} "
            f"(needs equal K and dtype and a multiple of 64 shared columns)"
        )
    padded = -(-rows // 128) * 128
    head = torch.zeros(padded, k_in, dtype=head_weight.dtype, device=head_weight.device)
    head[:rows].copy_(head_weight)
    gate = gate_up_weight[:inter].reshape(inter // 16, 16, k_in)
    up = gate_up_weight[inter:].reshape(inter // 16, 16, k_in)
    shared = torch.stack([gate, up], dim=1).reshape(2 * inter, k_in)
    return torch.cat([head, shared]).contiguous()


@functools.lru_cache(maxsize=None)
def _max_clusters_of(device_index: int) -> int:
    from cutlass.utils.hardware_info import HardwareInfo

    return HardwareInfo(device_index).get_max_active_clusters(_kernel().SPLIT)


def max_clusters(device: torch.device) -> int:
    """Clusters of the kernel's SPLIT CTAs, one CTA per SM, that ``device`` holds at once."""
    index = device.index if device.index is not None else torch.cuda.current_device()
    return _max_clusters_of(index)


def weight_supported(
    world: int, shared_cols: int, k_in: int, device: torch.device, ring: int = DEFAULT_RING
) -> bool:
    """Whether the front runs this TP world, shared activation width and hidden size (any M <= 8)."""
    return _kernel().supports(world, shared_cols, max_clusters(device), k_in, ring)


def supports(
    x: torch.Tensor, w_front: torch.Tensor, world: int, shared_cols: int, ring: int = DEFAULT_RING
) -> bool:
    kernel = _kernel()
    return (
        x.dim() == 2
        and 0 < x.shape[0] <= MAX_TOKENS
        and x.dtype == torch.bfloat16
        and w_front.dtype == torch.bfloat16
        and x.shape[1] == w_front.shape[1]
        and w_front.shape[0] == (kernel.head_tiles(world) * 128 + 2 * shared_cols)
        and kernel.supports(world, shared_cols, max_clusters(x.device), x.shape[1], ring)
    )


@torch.library.custom_op("trtllm::k3_moe_front", mutates_args=())
def k3_moe_front(
    x: torch.Tensor,
    w_front: torch.Tensor,
    bias: torch.Tensor,
    routed_scaling_factor: float,
    shared_cols: int,
    gate_cap: float,
    linear_cap: float,
    ag_uc: torch.Tensor,
    ag_mc: torch.Tensor,
    ag_flags: torch.Tensor,
    ag_rank: int,
    ag_world: int,
    ring: int = DEFAULT_RING,
    ag_ready: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """``x`` bf16 ``[M <= 8, 7168]`` (the MoE input, the same on every rank), ``w_front`` from ``front_weight``,
    ``bias`` the routing bias fp32 ``[896]``. Returns ``(topk_ids, topk_weights, quantized, scales, shared)``: what
    ``trtllm::k3_route_quant_ag`` returns for the gathered head, and the shared experts' activation bf16
    ``[M, shared_cols]``. With ``ag_ready`` it also releases the per-token ready words as route_quant_ag does."""
    import cuda.bindings.driver as cuda_driver
    import cutlass.cute as cute
    from cutlass.cute.runtime import from_dlpack

    kernel = _kernel()
    if not supports(x, w_front, ag_world, shared_cols, ring):
        raise ValueError(
            f"k3_moe_front: unsupported call x {tuple(x.shape)} {x.dtype}, w_front {tuple(w_front.shape)} "
            f"{w_front.dtype}, world {ag_world}, shared_cols {shared_cols}, ring {ring}"
        )
    from . import k3_route_quant_ag as layout

    ag_words = layout.workspace_words(ag_world)
    if ag_uc.numel() < ag_words or ag_mc.numel() < ag_words:
        raise ValueError(
            f"k3_moe_front: the head workspace holds {ag_uc.numel()} words, the front needs {ag_words} "
            "(op.head_workspace: the all-gather's buffers, then the router partials)"
        )
    num_tokens, k_in = x.shape
    device = x.device
    topk_ids = torch.empty(num_tokens, TOP_K, dtype=torch.int32, device=device)
    topk_weights = torch.empty(num_tokens, TOP_K, dtype=torch.bfloat16, device=device)
    quantized = torch.empty(num_tokens, HIDDEN_SIZE, dtype=torch.float8_e4m3fn, device=device)
    scales = torch.empty(num_tokens, HIDDEN_SIZE // SF_VEC, dtype=torch.uint8, device=device)
    shared = torch.empty(num_tokens, shared_cols, dtype=torch.bfloat16, device=device)
    # Head-only fronts (shared_cols 0) still hand the kernel an aligned shared-output argument it never writes.
    shared_arg = shared if shared_cols else ag_flags.view(torch.int16).view(-1)

    def arg2(t):
        return from_dlpack(t.detach(), assumed_align=16).mark_layout_dynamic(leading_dim=1)

    def arg(t):
        return from_dlpack(t.detach(), assumed_align=16).mark_layout_dynamic(leading_dim=0)

    args = (
        arg2(w_front),
        arg2(x.contiguous()),
        arg(bias.contiguous().view(-1)),
        arg(ag_uc.view(-1)),
        arg(ag_mc.view(-1)),
        arg(ag_flags.view(-1)),
        arg(ag_ready.view(-1) if ag_ready is not None else ag_flags.view(-1)),
        arg(topk_ids.view(-1)),
        arg(topk_weights.view(-1).view(torch.int16)),
        arg(quantized.view(-1).view(torch.int32)),
        arg(scales.view(-1)),
        arg(shared_arg.view(-1).view(torch.int16)),
    )
    capacity = max_clusters(device)
    # One round with the head in 64-row half-tiles when it fits (TP16): every head k-tile on chip before the wait.
    half = kernel.half_geometry(ag_world, shared_cols, capacity, k_in, ring)
    head_half = half is not None
    ht, st, clusters = half if head_half else kernel.geometry(ag_world, shared_cols, capacity)
    publish = ag_ready is not None
    use_pdl = os.environ.get("TRTLLM_ENABLE_PDL", "1") == "1"
    stream = cuda_driver.CUstream(torch.cuda.current_stream(device).cuda_stream)
    key = (
        ag_world,
        shared_cols,
        ht + st,
        clusters,
        k_in,
        ring,
        float(gate_cap),
        float(linear_cap),
        publish,
        use_pdl,
        head_half,
    )
    fn = _compiled.get(key)
    if fn is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "trtllm::k3_moe_front must run once per configuration outside CUDA-graph capture first"
            )
        with _lock:
            fn = _compiled.get(key)
            if fn is None:
                fn = _compiled[key] = cute.compile(
                    kernel.k3_moe_front, *args, num_tokens, ag_rank, float(routed_scaling_factor), ag_world,
                    shared_cols, ht, ht + st, clusters, k_in, ring, float(gate_cap), float(linear_cap), publish,
                    use_pdl, head_half, stream,
                )  # fmt: skip
    fn(*args, num_tokens, ag_rank, float(routed_scaling_factor), stream)
    return topk_ids, topk_weights, quantized, scales, shared


@k3_moe_front.register_fake
def _(x, w_front, bias, routed_scaling_factor, shared_cols, gate_cap, linear_cap, ag_uc, ag_mc, ag_flags, ag_rank,
      ag_world, ring=DEFAULT_RING, ag_ready=None):  # fmt: skip
    num_tokens = x.shape[0]
    return (
        x.new_empty((num_tokens, TOP_K), dtype=torch.int32),
        x.new_empty((num_tokens, TOP_K), dtype=torch.bfloat16),
        x.new_empty((num_tokens, HIDDEN_SIZE), dtype=torch.float8_e4m3fn),
        x.new_empty((num_tokens, HIDDEN_SIZE // SF_VEC), dtype=torch.uint8),
        x.new_empty((num_tokens, shared_cols), dtype=torch.bfloat16),
    )
