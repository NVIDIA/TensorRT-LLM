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
"""``trtllm::k3_head_gemv``: ``x @ weight^T`` for a large weight (an lm_head vocab shard) and M <= 8 rows.

One CTA per SM, and under stream-K at most one per (tile, k-tile), streams the weight, EVICT_FIRST except for the
first ``keep_tiles`` 128-row tiles, and adds each tile's k-split partials in a fixed order before one bf16 rounding:
the result is bit-identical from run to run (not bit-identical to cuBLAS or pdl_gemv, whose accumulation orders
differ). Two schedules: ``streamk`` (every CTA an equal share of the flat (tile, k-tile) space; the default) and
``dynamic`` ((tile, k-chunk) units claimed from a counter, ``chunk_tiles`` k-tiles each).

The op keeps one workspace per device and shape (the partials, a per-tile count and the unit counter, the counters
zero between launches), so calls of one shape must be ordered on one stream. Each kernel compiles on the first
call for its shape, which must happen outside CUDA-graph capture.
"""

from __future__ import annotations

import os
import threading
from typing import Dict

import torch

MAX_TOKENS = 8
UNITS_PER_CTA = 4  # the chunk choice keeps at least this many units per CTA for the dynamic balance

_lock = threading.Lock()
_compiled: Dict[tuple, object] = {}
_workspaces: Dict[tuple, tuple] = {}


def _arg(t: torch.Tensor):
    from cutlass.cute.runtime import from_dlpack

    # detach(): DLPack refuses tensors that require grad (weights are parameters).
    return from_dlpack(t.detach(), assumed_align=16).mark_layout_dynamic(leading_dim=t.dim() - 1)


def _use_pdl() -> bool:
    return os.environ.get("TRTLLM_ENABLE_PDL", "1") == "1"


def _stream(t: torch.Tensor):
    import cuda.bindings.driver as cuda_driver

    return cuda_driver.CUstream(torch.cuda.current_stream(t.device).cuda_stream)


def _grid(device) -> int:
    return torch.cuda.get_device_properties(device).multi_processor_count


def pick_chunk(n_out: int, k_in: int, grid: int) -> int:
    """k-tiles per unit: the largest divisor of the k-tiles that leaves at least UNITS_PER_CTA units per CTA, or the
    smallest divisor above 1 when none does (every divisor if the k-tiles are prime)."""
    from . import k3_head_gemv_kernel as kernel

    k_tiles = kernel.num_k_tiles(k_in)
    tiles = n_out // kernel.CTA_M
    divisors = [d for d in range(1, k_tiles + 1) if k_tiles % d == 0]
    fitting = [d for d in divisors if tiles * (k_tiles // d) >= UNITS_PER_CTA * grid]
    if fitting:
        return max(fitting)
    return min([d for d in divisors if d > 1] or divisors)


def supports(
    x: torch.Tensor,
    weight: torch.Tensor,
    chunk_tiles: int = 0,
    ring: int = 6,
    schedule: str = "streamk",
) -> bool:
    """Whether ``k3_head_gemv`` runs ``x @ weight^T`` (``chunk_tiles`` 0: the op's choice)."""
    from . import k3_head_gemv_kernel as kernel

    if not (
        x.is_cuda
        and x.dtype == torch.bfloat16
        and weight.dtype == torch.bfloat16
        and x.dim() == 2
        and weight.dim() == 2
        and 0 < x.shape[0] <= MAX_TOKENS
        and x.shape[1] == weight.shape[1]
        and x.is_contiguous()
        and weight.is_contiguous()
        and weight.shape[0] % kernel.CTA_M == 0
        and weight.shape[1] % kernel.CTA_K == 0
    ):
        return False
    n_out, k_in = weight.shape
    if schedule == "streamk":
        return kernel.streamk_supports(n_out, k_in, ring)
    if schedule != "dynamic":
        return False
    chunk = chunk_tiles or pick_chunk(n_out, k_in, _grid(x.device))
    return kernel.supports(n_out, k_in, chunk, ring)


def _workspace(device, units: int, tiles: int):
    """(partials fp32 [units * 128 * 8], ``tiles`` count / flag words, claim counter), the counters zero."""
    key = (device.index, units, tiles)
    ws = _workspaces.get(key)
    if ws is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "k3_head_gemv: the workspace must be allocated outside CUDA-graph capture"
            )
        ws = (
            torch.empty(max(units, 1) * 128 * 8, dtype=torch.float32, device=device),
            torch.zeros(tiles, dtype=torch.int32, device=device),
            torch.zeros(1, dtype=torch.int32, device=device),
        )
        _workspaces[key] = ws
    return ws


def _compiled_fn(key, entry, *args):
    fn = _compiled.get(key)
    if fn is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "k3_head_gemv: run once per shape outside CUDA-graph capture first (it compiles)."
            )
        import cutlass.cute as cute

        with _lock:
            fn = _compiled.get(key)
            if fn is None:
                fn = _compiled[key] = cute.compile(entry, *args)
    return fn


@torch.library.custom_op("trtllm::k3_head_gemv", mutates_args=())
def k3_head_gemv(
    x: torch.Tensor,
    weight: torch.Tensor,
    keep_tiles: int = 0,
    chunk_tiles: int = 0,
    ring: int = 6,
    schedule: str = "streamk",
    prefetch: int = 16,
) -> torch.Tensor:
    """``x @ weight.T`` for bf16 ``x`` [M <= 8, K] and ``weight`` [N, K] (N and K multiples of 128); returns bf16
    [M, N]. Weight tiles (128 rows) below ``keep_tiles`` are read at normal L2 priority, the rest EVICT_FIRST.
    ``schedule``: ``streamk`` or ``dynamic``; ``chunk_tiles`` (dynamic; 0: the op's choice) sets the k-tiles per
    work unit; ``prefetch`` (stream-K) the k-tiles after the ring that each CTA prefetches into L2 before the grid
    dependency."""
    if not supports(x, weight, chunk_tiles, ring, schedule):
        raise ValueError(
            f"k3_head_gemv: unsupported call x {tuple(x.shape)} {x.dtype}, weight {tuple(weight.shape)} "
            f"{weight.dtype}, chunk_tiles {chunk_tiles}, ring {ring}, schedule {schedule}"
        )
    from . import k3_head_gemv_kernel as kernel

    num_tokens, k_in = x.shape
    n_out = weight.shape[0]
    tiles = n_out // kernel.CTA_M
    grid = _grid(x.device)
    # All 8 token rows are written (the rows past M from TMA's zero-filled x rows); the result is the first M.
    y = torch.empty(kernel.MMA_N, n_out, dtype=torch.bfloat16, device=x.device)
    stream = _stream(x)
    use_pdl = _use_pdl()
    if schedule == "streamk":
        # At most one CTA per (tile, k-tile): every CTA's share is then non-empty, so each CTA between a split tile's
        # first and last piece holds part of that tile and raises the flag its finalizer waits for.
        grid = min(grid, tiles * kernel.num_k_tiles(k_in))
        pieces = kernel.streamk_max_pieces(n_out, k_in, grid)
        ws, flags, _ = _workspace(x.device, tiles * pieces, tiles * pieces)
        args = (_arg(weight), _arg(x), _arg(y.view(-1)), _arg(ws), _arg(flags))
        fn = _compiled_fn(("k3_head_gemv_sk", n_out, k_in, ring, grid, pieces, prefetch, use_pdl),
                          kernel.k3_head_gemv_sk, *args, num_tokens, keep_tiles, n_out, k_in, ring, grid, pieces,
                          prefetch, use_pdl, stream)  # fmt: skip
    else:
        chunk = chunk_tiles or pick_chunk(n_out, k_in, grid)
        ws, cnt, claim = _workspace(x.device, kernel.num_units(n_out, k_in, chunk), tiles)
        args = (_arg(weight), _arg(x), _arg(y.view(-1)), _arg(ws), _arg(cnt), _arg(claim))
        fn = _compiled_fn(("k3_head_gemv", n_out, k_in, chunk, ring, grid, use_pdl), kernel.k3_head_gemv, *args,
                          num_tokens, keep_tiles, n_out, k_in, chunk, ring, grid, use_pdl, stream)  # fmt: skip
    fn(*args, num_tokens, keep_tiles, stream)
    return y[:num_tokens]


@k3_head_gemv.register_fake
def _(x, weight, keep_tiles=0, chunk_tiles=0, ring=6, schedule="streamk", prefetch=16):
    return x.new_empty((x.shape[0], weight.shape[0]), dtype=torch.bfloat16)
