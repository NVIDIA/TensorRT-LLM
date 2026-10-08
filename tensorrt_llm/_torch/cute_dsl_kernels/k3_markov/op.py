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
"""``trtllm::k3_markov``: the DSpark vanilla-Markov draft chain over a vocab-sharded draft head in one CTM kernel.

For the block logits ``base`` [B, K, S] of this rank's vocab shard it returns the Markov-corrected logits (fp32, as
``dspark_markov_chain`` with fp32 logits), every position's global greedy token (first maximum, lowest vocabulary
index, across the tensor-parallel ranks) and the next step's input tokens
``[accepted[row_b, num_accepted[b] - 1], tokens[b, :]]``: what ``dspark_markov_chain`` over the gathered logits
followed by the TP-gathered greedy sampler computes.

The ranks exchange one (value, index) entry per CTA and position through this module's own MNNVL multicast buffers
(``markov_workspace``), allocated collectively on the first call of a TP group, which must happen outside CUDA-graph
capture (the kernel also compiles there). Every CTA spins on the other ranks' entries, so all of the grid's CTAs (at
most the SM count) must be resident at once: no concurrent kernel may hold SMs while it waits on this grid, and no SM
cap (MPS or green contexts) may sit below the grid.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import threading
from typing import Dict, Optional, Tuple

import torch

_lock = threading.Lock()
_compiled: Dict[tuple, object] = {}
_workspaces: Dict[object, dict] = {}
_modules: Dict[str, object] = {}

WORKSPACE_MAX_BLOCK = 8
WORKSPACE_MAX_GRID = 152


def _kernel_module():
    """k3_markov_kernel.py next to this file (also when op.py is loaded outside the package)."""
    mod = _modules.get("kernel")
    if mod is None:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "k3_markov_kernel.py")
        spec = importlib.util.spec_from_file_location(f"{__name__}_k3_markov_kernel", path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
        _modules["kernel"] = mod
    return mod


def _sm_count(device) -> int:
    return torch.cuda.get_device_properties(device).multi_processor_count


def pick_grid(shard: int, block: int, batch: int, device=None) -> int:
    """CTAs for a shard: 128 when the kernel supports that split (see ``k3_markov_kernel.supports``), else the
    largest supported count up to the SM count; 0 if none."""
    kern = _kernel_module()
    limit = min(
        WORKSPACE_MAX_GRID, _sm_count(device if device is not None else torch.cuda.current_device())
    )
    if 128 <= limit and kern.supports(shard, 128, block, batch):
        return 128
    for grid in range(limit, 0, -1):
        if kern.supports(shard, grid, block, batch):
            return grid
    return 0


def supports(base: torch.Tensor, markov_w1: torch.Tensor, markov_w2_shard: torch.Tensor) -> bool:
    """Whether ``k3_markov`` runs these operands (the TP group also needs MNNVL, see ``markov_workspace``)."""
    kern = _kernel_module()
    if base.dim() != 3 or base.dtype not in (torch.float32, torch.bfloat16) or not base.is_cuda:
        return False
    batch, block, shard = base.shape
    return (
        markov_w1.dtype == torch.bfloat16
        and markov_w2_shard.dtype == torch.bfloat16
        and markov_w1.dim() == 2
        and markov_w1.shape[1] == kern.MARKOV_RANK
        and tuple(markov_w2_shard.shape) == (shard, kern.MARKOV_RANK)
        and block <= WORKSPACE_MAX_BLOCK
        and pick_grid(shard, block, batch, base.device) > 0
    )


def markov_workspace(mapping, push_copies: int = 1) -> dict:
    """This TP group's Lamport buffers (3 rotating buffers, every word 0x80000000 when armed) behind one multicast
    mapping, and its flag words. Collective on first use: every rank of the group must make the first call at the
    same point, outside CUDA-graph capture. ``push_copies`` > 1 (tests only) gives every rank that many slots, to
    emulate a larger group's exchange volume."""
    key = (mapping, push_copies)
    ws = _workspaces.get(key)
    if ws is not None:
        return ws
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("k3_markov: the workspace must be allocated outside CUDA-graph capture")
    from tensorrt_llm._torch.distributed.ops import (
        _get_mnnvl_workspace_comm,
        _make_mnnvl_mcast_buffer,
        _mnnvl_workspace_all_succeeded,
    )

    kern = _kernel_module()
    world = mapping.tp_size
    slots = world * push_copies
    buf_words = kern.buffer_words(WORKSPACE_MAX_BLOCK, kern.MAX_BATCH, slots, WORKSPACE_MAX_GRID)
    comm = _get_mnnvl_workspace_comm(mapping)
    use_fabric_handle = (
        os.environ.get("TRTLLM_FORCE_MNNVL_AR", "0") == "1" or mapping.is_multi_node()
    )
    error: Optional[Exception] = None
    try:
        words = kern.BUFFERS * buf_words
        handle = _make_mnnvl_mcast_buffer(comm, words * 4, mapping, use_fabric_handle)
        uc = handle.get_uc_buffer(mapping.tp_rank, (words,), torch.int32, 0)
        mc = handle.get_mc_buffer((words,), torch.int32, 0)
        with torch.inference_mode():
            uc.fill_(kern.EMPTY_WORD)
            flags = torch.zeros(kern.FLAG_WORDS, dtype=torch.int32, device=uc.device)
        torch.cuda.synchronize()
        ws = dict(handle=handle, comm=comm, uc=uc, mc=mc, flags=flags, rank=mapping.tp_rank, world=world,
                  slots=slots, push_copies=push_copies, buf_words=buf_words)  # fmt: skip
    except Exception as exc:  # noqa: BLE001 -- reported to every rank below, then re-raised
        error = exc
    # Also the barrier that keeps any rank from pushing into a peer's buffer before the peer has armed it.
    if not _mnnvl_workspace_all_succeeded(comm, error is None):
        raise RuntimeError("k3_markov Lamport buffers failed on at least one rank") from error
    _workspaces[key] = ws
    return ws


def _arg(t: torch.Tensor, align: int = 16):
    from cutlass.cute.runtime import from_dlpack

    return from_dlpack(t.detach(), assumed_align=align).mark_layout_dynamic(leading_dim=0)


@torch.library.custom_op("trtllm::k3_markov", mutates_args=("kv_lens",))
def k3_markov(
    base: torch.Tensor,
    first_prev: torch.Tensor,
    markov_w1: torch.Tensor,
    markov_w2_shard: torch.Tensor,
    shard_offset: int,
    ws_uc: torch.Tensor,
    ws_mc: torch.Tensor,
    ws_flags: torch.Tensor,
    rank: int,
    slots: int,
    push_copies: int,
    buf_words: int,
    accepted: torch.Tensor,
    num_accepted: torch.Tensor,
    accepted_rows: torch.Tensor,
    kv_lens: torch.Tensor,
    rewind: torch.Tensor,
    rewind_first: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(corrected [B, K, S] fp32, tokens [B, K] int32, next_new [B, K + 1] int32)`` for the block logits ``base``
    [B, K, S] (fp32, or bf16 converted exactly) of this rank's shard, which starts at vocabulary index
    ``shard_offset``. ``ws_*``, ``slots``, ``push_copies``, ``buf_words``: ``markov_workspace``'s. ``accepted``
    [rows, K + 1] int32, ``num_accepted`` [B] int32 and ``accepted_rows`` [B] int32 give next_new's first column
    ``accepted[accepted_rows[b], num_accepted[b] - 1]``. ``kv_lens`` (int32) is rewound in place:
    ``kv_lens[rewind_first + b] = max(kv_lens[rewind_first + b] - rewind[b], 0)`` for b < ``rewind.numel()`` (<= B;
    empty: no rewind)."""
    import cuda.bindings.driver as cuda_driver
    import cutlass.cute as cute

    kern = _kernel_module()
    if not supports(base, markov_w1, markov_w2_shard):
        raise ValueError(
            f"k3_markov: unsupported operands: base {tuple(base.shape)} {base.dtype} (fp32/bf16 [B <= "
            f"{kern.MAX_BATCH}, K <= {WORKSPACE_MAX_BLOCK}, S]), markov_w1 {tuple(markov_w1.shape)} "
            f"{markov_w1.dtype} (bf16 [V, {kern.MARKOV_RANK}]), markov_w2_shard {tuple(markov_w2_shard.shape)} "
            f"{markov_w2_shard.dtype} (bf16 [S, {kern.MARKOV_RANK}]); S must split over an even number of CTAs "
            f"(at most the SM count) into a multiple of 64 rows each, at most {kern.SMEM_ROW_BUDGET}"
        )
    batch, block, shard = base.shape
    grid = pick_grid(shard, block, batch, base.device)
    rows = kern.rows_per_cta(shard, grid)
    if kern.buffer_words(block, batch, slots, grid) > buf_words:
        raise ValueError("k3_markov: the exchange does not fit the workspace buffers")
    if grid > _sm_count(base.device):
        raise ValueError(f"k3_markov: {grid} CTAs must all be resident; the GPU has fewer SMs")
    if first_prev.dtype != torch.int64 or first_prev.numel() != batch:
        raise ValueError("k3_markov: first_prev must be int64 [B]")
    if (
        accepted.dtype != torch.int32
        or accepted.dim() != 2
        or num_accepted.dtype != torch.int32
        or accepted_rows.dtype != torch.int32
        or num_accepted.numel() < batch
        or accepted_rows.numel() < batch
    ):
        raise ValueError(
            "k3_markov: accepted must be int32 [rows, K + 1], num_accepted / accepted_rows int32 [>= B]"
        )
    rewind_count = rewind.numel()
    if (
        kv_lens.dtype != torch.int32
        or rewind.dtype != torch.int32
        or rewind_count > batch
        or (
            rewind_count > 0 and (rewind_first < 0 or rewind_first + rewind_count > kv_lens.numel())
        )
    ):
        raise ValueError(
            "k3_markov: kv_lens / rewind must be int32, the rewind at most B rows inside kv_lens"
        )
    device = base.device
    corrected = torch.empty(batch, block, shard, dtype=torch.float32, device=device)
    tokens = torch.empty(batch, block, dtype=torch.int32, device=device)
    next_new = torch.empty(batch, block + 1, dtype=torch.int32, device=device)
    base_bf16 = base.dtype == torch.bfloat16
    base_view = base.contiguous().view(-1)
    if base_bf16:
        base_view = base_view.view(torch.int16)
    # The kernel reads first_prev, accepted, num_accepted, accepted_rows, kv_lens and rewind one element at a time:
    # declared at their element alignment, they may be views at any offset (e.g. the generation requests' rows after
    # the context requests').
    args = (
        _arg(base_view),
        _arg(first_prev.contiguous().view(-1), align=8),
        _arg(markov_w1.contiguous().view(-1).view(torch.int32)),
        _arg(markov_w2_shard.contiguous().view(-1).view(torch.int32)),
        _arg(corrected.view(-1)),
        _arg(tokens.view(-1)),
        _arg(next_new.view(-1)),
        _arg(accepted.contiguous().view(-1), align=4),
        _arg(num_accepted.contiguous().view(-1), align=4),
        _arg(accepted_rows.contiguous().view(-1), align=4),
        _arg(kv_lens.view(-1), align=4),
        _arg(rewind.contiguous().view(-1) if rewind_count > 0 else kv_lens.view(-1), align=4),
        _arg(ws_uc.view(-1)),
        _arg(ws_mc.view(-1)),
        _arg(ws_flags.view(-1)),
    )
    scalars = (int(markov_w1.shape[0]), int(shard_offset), int(rank), int(accepted.shape[1]), int(rewind_first),
               int(rewind_count))  # fmt: skip
    use_pdl = os.environ.get("TRTLLM_ENABLE_PDL", "1") == "1"
    stream = cuda_driver.CUstream(torch.cuda.current_stream(device).cuda_stream)
    consts = (grid, rows, shard, block, batch, slots, push_copies, buf_words, base_bf16)
    key = consts + (use_pdl,)
    fn = _compiled.get(key)
    if fn is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "trtllm::k3_markov must run once per shape outside CUDA-graph capture first"
            )
        with _lock:
            fn = _compiled.get(key)
            if fn is None:
                fn = _compiled[key] = cute.compile(
                    kern.k3_markov, *args, *scalars, *consts, use_pdl, stream
                )
    fn(*args, *scalars, stream)
    return corrected, tokens, next_new


@k3_markov.register_fake
def _(base, first_prev, markov_w1, markov_w2_shard, shard_offset, ws_uc, ws_mc, ws_flags, rank, slots, push_copies,
      buf_words, accepted, num_accepted, accepted_rows, kv_lens, rewind, rewind_first):  # fmt: skip
    batch, block, shard = base.shape
    return (
        base.new_empty((batch, block, shard), dtype=torch.float32),
        base.new_empty((batch, block), dtype=torch.int32),
        base.new_empty((batch, block + 1), dtype=torch.int32),
    )


def markov_chain(
    mapping,
    base: torch.Tensor,
    first_prev: torch.Tensor,
    markov_w1: torch.Tensor,
    markov_w2_shard: torch.Tensor,
    shard_offset: int,
    accepted: torch.Tensor,
    num_accepted: torch.Tensor,
    accepted_rows: torch.Tensor,
    push_copies: int = 1,
    kv_lens: Optional[torch.Tensor] = None,
    rewind: Optional[torch.Tensor] = None,
    rewind_first: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``trtllm::k3_markov`` on ``mapping``'s TP group workspace; with ``kv_lens`` and ``rewind`` it also applies the
    KV-length rewind (see ``k3_markov``)."""
    ws = markov_workspace(mapping, push_copies)
    if kv_lens is None or rewind is None:
        # No rewind: an empty rewind leaves the kernel's kv_lens argument (the flags words) untouched.
        kv_lens, rewind = ws["flags"], ws["flags"][:0]
    return torch.ops.trtllm.k3_markov(
        base, first_prev, markov_w1, markov_w2_shard, int(shard_offset), ws["uc"], ws["mc"], ws["flags"],
        ws["rank"], ws["slots"], ws["push_copies"], ws["buf_words"], accepted, num_accepted, accepted_rows, kv_lens,
        rewind, int(rewind_first),
    )  # fmt: skip
