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
"""MARKOV_GRID: this target's own launch of the catalog's batch-1 Markov chain (S4B8, and its MARKOV_LL exchange) on
a wider grid than the catalog picks.

The catalog op (``ops/dspark/dspark_markov_chain``) caps a call at ``MAX_GRID`` = 152 CTAs and prefers 128,
so the M3.1 TP4 shard (50,016 rows padded to 50,688) runs on 132 CTAs x 384 markov_w2 rows (6 eight-row blocks per
warp). This module launches the same kernel modules (``dspark_markov_chain_kernel_b1`` / ``_b1_ll``, unchanged: the
grid is a compile-time parameter of theirs) for batch-1 calls on the widest grid this GPU keeps co-resident, up to
``MAX_GRID_B1``: on SM107 (212 SMs) 198 CTAs x 256 rows (4 blocks per warp). Every CTA still computes its rows with
the same FMA order and roundings and every rank folds the same set of (value, index) entries under the same total
order, so the tokens are bitwise the catalog call's; only the split of the rows over CTAs and the exchange's entry
count (slots x grid) change. The exchange goes through the catalog's workspace (``workspace_args``: a batch-1 call at
198 CTAs uses 11,088 of its 155,648-word Lamport buffers and 22,176 words of an LL region), with the catalog's flags,
so batch-1 calls here and the catalog's batch 2..8 calls share one rotation exactly as the catalog's own calls of
different shapes do.

Every rank must launch the same grid for a call (the exchange layout follows it): the grid is a function of the
padded rows, the block, the switches and the SM count only, and the worker checks that every rank agrees at prebuild.
Compiled on the first call per shape, outside CUDA-graph capture (``precompile``).
"""

from __future__ import annotations

import threading
from typing import Dict, Optional

import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.ops.dspark import (
    dspark_markov_chain as chain,
)

MAX_GRID_B1 = (
    256  # capacity: fixed: this launcher's CTA bound (all co-resident: one CTA per SM at 256 rows)
)
ROWS_PER_CTA_MAX = (
    256  # the widest grid's rows per CTA: 4 eight-row blocks per warp (the catalog's TP4 call: 6)
)

_lock = threading.Lock()
_compiled: Dict[tuple, object] = {}


def _sm_count(device) -> int:
    return torch.cuda.get_device_properties(device).multi_processor_count


def _b1_module(ll: bool):
    return chain._kernel_module(batch1=True, ll=ll)


def grid_ok(rows: int, grid: int, block: int, ll: bool) -> bool:
    """Whether the batch-1 kernel (``ll``: its LL variant) takes ``rows`` on ``grid`` CTAs with every markov_w2 row in
    shared memory and at most ``ROWS_PER_CTA_MAX`` rows per CTA."""
    if grid <= 1 or rows % grid or rows // grid > ROWS_PER_CTA_MAX:
        return False
    kern = _b1_module(ll)
    return kern.supports(rows, grid, block, 1) and rows // grid <= kern.SMEM_ROW_BUDGET


def pick_grid(rows: int, block: int, sm_count: int, ll: bool) -> int:
    """The batch-1 grid for a padded shard of ``rows``: the most CTAs, up to ``min(MAX_GRID_B1, sm_count)``, that
    ``grid_ok`` takes; 0 when none (the catalog op runs)."""
    for grid in range(min(MAX_GRID_B1, int(sm_count)), 1, -1):
        if grid_ok(rows, grid, block, ll):
            return grid
    return 0


def catalog_grid(rows: int, block: int, device) -> int:
    """The grid the catalog op picks for this batch-1 call."""
    return chain.pick_grid(rows, block, 1, device)


def plan(rows: int, block: int, device, sm_count: Optional[int] = None) -> int:
    """This launcher's grid for batch-1 calls of (``rows``, ``block``) under the catalog's current switches, or 0: no
    batch-1 variant, no wider grid than the catalog's own.

    The catalog's switches are process-wide and off by default: a caller outside the worker (a micro bench) turns on
    what the S4 worker's prebuild turns on before calling this -- ``chain.enable_batch1_variant(True)`` (else 0, and
    ``markov_chain_b1`` refuses grid 0), ``chain.enable_ll_exchange(True)`` for the deployed LL exchange (the plan and
    the compiled kernel follow it), ``chain.enable_fp32_sum(True)`` and ``chain.enable_cached_inputs(True)`` -- and
    compares against the catalog's ``dspark_markov_chain`` under the same switches."""
    if not chain._batch1_variant[0] or not 0 < block <= chain.MAX_BLOCK:
        return 0
    sms = _sm_count(device) if sm_count is None else int(sm_count)
    ll = bool(chain._ll_exchange[0])
    grid = pick_grid(rows, block, sms, ll)
    if grid == 0 or grid <= catalog_grid(rows, block, device):
        return 0
    return grid


def fits_workspace(block: int, grid: int, ws: dict) -> bool:
    """Whether a batch-1 call of ``block`` positions on ``grid`` CTAs fits the catalog workspace ``ws``."""
    kern = chain._kernel_module()
    slots, buf_words = int(ws["slots"]), int(ws["buf_words"])
    if kern.buffer_words(block, 1, slots, grid) > buf_words:
        return False
    if chain._ll_exchange[0]:
        ll = _b1_module(True)
        return (
            ll.ll_region_words(block, slots, grid) <= buf_words
            and ws["ws_uc"].numel() >= ll.ll_workspace_words(buf_words)
            and ws["ws_flags"].numel() >= ll.LL_FLAG_WORDS
        )
    return True


def supports(base: torch.Tensor, w1: torch.Tensor, w2_shard: torch.Tensor, grid: int) -> bool:
    """Whether this launcher takes the call on ``grid``: the catalog op's operand rules, batch 1, the batch-1 variant
    on, and the grid's split of the padded rows."""
    if grid <= 0 or not chain.supports(base, w1, w2_shard) or base.shape[0] != 1:
        return False
    if not chain._batch1_variant[0]:
        return False
    return grid_ok(base.shape[2], grid, base.shape[1], bool(chain._ll_exchange[0]))


def markov_chain_b1(
    base: torch.Tensor,
    first_prev: torch.Tensor,
    w1: torch.Tensor,
    w2_shard: torch.Tensor,
    shard_offset: int,
    grid: int,
    *,
    ws_uc: torch.Tensor,
    ws_mc: torch.Tensor,
    ws_flags: torch.Tensor,
    rank: int,
    slots: int,
    buf_words: int,
    kv_lens: Optional[torch.Tensor] = None,
    rewind: Optional[torch.Tensor] = None,
    rewind_first: int = 0,
) -> torch.Tensor:
    """The catalog op's batch-1 call (``dspark_markov_chain`` / ``dspark_markov_chain_rewind``: the same arguments,
    tokens and KV-length rewind) on ``grid`` CTAs. Check ``supports`` first; a collective as the catalog call."""
    import cuda.bindings.driver as cuda_driver
    import cutlass.cute as cute

    from tensorrt_llm._torch.flashinfer_utils import get_env_enable_pdl

    if not supports(base, w1, w2_shard, grid):
        raise ValueError(
            f"markov_chain_b1: unsupported operands base {tuple(base.shape)} {base.dtype} on {grid} CTAs"
        )
    if first_prev.dtype != torch.int64 or first_prev.numel() != 1:
        raise ValueError("markov_chain_b1: first_prev must be int64 [1]")
    ws = dict(
        ws_uc=ws_uc, ws_mc=ws_mc, ws_flags=ws_flags, rank=rank, slots=slots, buf_words=buf_words
    )
    _, block, rows = base.shape
    if not fits_workspace(block, grid, ws):
        raise ValueError("markov_chain_b1: the exchange does not fit the workspace")
    ll = bool(chain._ll_exchange[0])
    device = base.device
    tokens = torch.empty(1, block, dtype=torch.int32, device=device)
    next_new = torch.empty(1, block + 1, dtype=torch.int32, device=device)
    dummy = chain._dummy.get(device)
    if dummy is None:
        dummy = chain._dummy[device] = torch.zeros(4, dtype=torch.int32, device=device)
    rewind_count = 0 if rewind is None else int(rewind.numel())
    if rewind_count:
        if not chain.supports_rewind(kv_lens, rewind, rewind_first):
            raise ValueError(
                "markov_chain_b1: unsupported KV-length rewind (see supports_rewind())"
            )
    else:
        kv_lens, rewind = dummy, dummy
    nacc, rows_idx, acc = chain._accepted_inputs(1, block, device)
    args = (
        chain._arg(base.contiguous().view(-1).view(torch.int16)),
        chain._arg(first_prev.contiguous().view(-1), align=8),
        chain._arg(w1.contiguous().view(-1).view(torch.int32)),
        chain._arg(w2_shard.contiguous().view(-1).view(torch.int32)),
        chain._arg(dummy.view(torch.float32)),
        chain._arg(tokens.view(-1)),
        chain._arg(next_new.view(-1)),
        chain._arg(acc.view(-1), align=4),
        chain._arg(nacc, align=4),
        chain._arg(rows_idx, align=4),
        chain._arg(kv_lens, align=4),
        chain._arg(rewind, align=4),
        chain._arg(ws_uc.view(-1)),
        chain._arg(ws_mc.view(-1)),
        chain._arg(ws_flags.view(-1)),
    )
    scalars = (
        int(w1.shape[0]),
        int(shard_offset),
        int(rank),
        int(block + 1),
        int(rewind_first),
        rewind_count,
    )
    use_pdl = get_env_enable_pdl()
    stream = cuda_driver.CUstream(torch.cuda.current_stream(device).cuda_stream)
    consts = (
        grid,
        rows // grid,
        rows,
        block,
        1,
        int(slots),
        1,
        int(buf_words),
        True,
        chain._fp32_sum[0],
    )
    key = consts + (use_pdl, ll)
    fn = _compiled.get(key)
    if fn is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "markov_chain_b1 must run once per shape outside CUDA-graph capture first"
            )
        with _lock:
            fn = _compiled.get(key)
            if fn is None:
                fn = _compiled[key] = cute.compile(
                    _b1_module(ll).k3_markov, *args, *scalars, *consts, use_pdl, stream
                )
    fn(*args, *scalars, stream)
    return tokens


def is_compiled(rows: int, block: int, grid: int, slots: int, buf_words: int) -> bool:
    """Whether the batch-1 call of this shape on ``grid`` is compiled under the current switches."""
    from tensorrt_llm._torch.flashinfer_utils import get_env_enable_pdl

    consts = (
        grid,
        rows // grid,
        rows,
        block,
        1,
        int(slots),
        1,
        int(buf_words),
        True,
        chain._fp32_sum[0],
    )
    return consts + (get_env_enable_pdl(), bool(chain._ll_exchange[0])) in _compiled


def precompile(
    rows: int,
    w1: torch.Tensor,
    w2_shard: torch.Tensor,
    shard_offset: int,
    block: int,
    grid: int,
    ws: dict,
) -> None:
    """Compile (and run once) the batch-1 call of ``block`` positions on ``grid`` outside capture. A collective: every
    rank calls it at the same point, after the catalog op's own precompile."""
    base = chain.pad_logits_buffer(1, block, rows, w2_shard.device)
    prev = torch.zeros(1, dtype=torch.long, device=w2_shard.device)
    markov_chain_b1(base, prev, w1, w2_shard, shard_offset, grid, **ws)
    torch.cuda.synchronize()
