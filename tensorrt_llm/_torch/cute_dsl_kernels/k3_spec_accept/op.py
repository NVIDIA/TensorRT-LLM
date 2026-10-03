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
"""``trtllm::k3_spec_accept``: one decode step's speculative acceptance and the block drafter's inputs in one CTM
kernel (greedy strict acceptance of B generation requests, K drafts each; see k3_spec_accept_kernel.py).

Bit-identical to the Python path it replaces: ``_refresh_ctx_block_tables``, ``_sample_and_accept_draft_tokens_base``
(with the forced-acceptance override), the KDA replay record of ``update_mamba_states``, the kv_lens update of
``_prepare_kv_for_draft_forward`` and ``prepare_1st_drafter_inputs`` up to the fc (bonus, positions, noise embedding).
With a :func:`workspace`, the target logits may stay vocabulary-sharded (this rank's bf16 columns of a TP
column-parallel head): the ranks exchange their row maxima over a multicast Lamport buffer instead of all-gathering
the logits, with the same argmax. Compiled on the first call per shape, which must happen outside CUDA-graph capture.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import threading
from typing import Dict, List, Optional

import torch

_lock = threading.Lock()
_compiled: Dict[tuple, object] = {}
_scratch: Dict[tuple, tuple] = {}
_modules: Dict[str, object] = {}
_workspaces: Dict[object, dict] = {}


def _kernel_module():
    mod = _modules.get("kernel")
    if mod is None:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "k3_spec_accept_kernel.py")
        spec = importlib.util.spec_from_file_location(f"{__name__}_k3_spec_accept_kernel", path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
        _modules["kernel"] = mod
    return mod


def supports(vocab: int, batch: int, block: int, drafts: int, hidden: int, slots: int = 1) -> bool:
    """``vocab``: the columns of the logits this rank holds (its shard when ``slots`` > 1)."""
    return _kernel_module().supports(vocab, batch, block, drafts, hidden, slots)


def supports_columns(vocab: int, slots: int = 1) -> bool:
    """Whether the kernel splits ``vocab`` columns (a rank's shard of a ``slots``-rank group when ``slots`` > 1)."""
    return _kernel_module().supports_columns(vocab, slots)


def existing_workspace(mapping, push_copies: int = 1) -> Optional[dict]:
    """The group's :func:`workspace` if it has been allocated (inside CUDA-graph capture it cannot be)."""
    return _workspaces.get((mapping, push_copies))


def workspace(mapping, push_copies: int = 1) -> dict:
    """This TP group's Lamport buffers for the sharded argmax exchange (3 rotating buffers, every word 0x80000000
    when armed) behind one multicast mapping, and the flag words. Collective on first use: every rank of the group
    must make the first call at the same point, outside CUDA-graph capture. ``push_copies`` > 1 (tests only) gives
    every rank that many slots, to emulate a larger group."""
    key = (mapping, push_copies)
    ws = _workspaces.get(key)
    if ws is not None:
        return ws
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            "k3_spec_accept: the workspace must be allocated outside CUDA-graph capture"
        )
    from tensorrt_llm._torch.distributed.ops import (
        _get_mnnvl_workspace_comm,
        _make_mnnvl_mcast_buffer,
        _mnnvl_workspace_all_succeeded,
    )

    kern = _kernel_module()
    slots = mapping.tp_size * push_copies
    if not (slots % 2 == 0 and slots <= kern.MAX_SLOTS):
        raise ValueError(
            f"k3_spec_accept: {slots} exchange slots (an even count up to {kern.MAX_SLOTS})"
        )
    comm = _get_mnnvl_workspace_comm(mapping)
    use_fabric_handle = (
        os.environ.get("TRTLLM_FORCE_MNNVL_AR", "0") == "1" or mapping.is_multi_node()
    )
    error: Optional[Exception] = None
    try:
        words = kern.BUFFERS * kern.BUF_WORDS
        handle = _make_mnnvl_mcast_buffer(comm, words * 4, mapping, use_fabric_handle)
        uc = handle.get_uc_buffer(mapping.tp_rank, (words,), torch.int32, 0)
        mc = handle.get_mc_buffer((words,), torch.int32, 0)
        with torch.inference_mode():
            uc.fill_(kern.EMPTY_WORD)
            flags = torch.zeros(kern.FLAG_WORDS, dtype=torch.int32, device=uc.device)
        torch.cuda.synchronize()
        ws = dict(handle=handle, comm=comm, uc=uc, mc=mc, flags=flags, rank=mapping.tp_rank, slots=slots,
                  push_copies=push_copies)  # fmt: skip
    except Exception as exc:  # noqa: BLE001 -- reported to every rank below, then re-raised
        error = exc
    # Also the barrier that keeps any rank from pushing into a peer's buffer before the peer has armed it.
    if not _mnnvl_workspace_all_succeeded(comm, error is None):
        raise RuntimeError("k3_spec_accept Lamport buffers failed on at least one rank") from error
    _workspaces[key] = ws
    return ws


def force_mode(force: float, drafts: int):
    """(mode, base total, fraction) of ``_apply_force_accepted_tokens`` for the forced value ``force`` (0: off)."""
    kern = _kernel_module()
    if force == 0.0:
        return kern.FORCE_OFF, 0, 0.0
    int_part = int(force)
    frac = force - int_part
    total = min(int_part + 1, drafts + 1)
    if frac > 0.0 and total < drafts + 1:
        return kern.FORCE_FRAC, total, frac
    return kern.FORCE_INT, total, 0.0


def _arg(t: torch.Tensor):
    from cutlass.cute.runtime import from_dlpack

    return from_dlpack(t.detach(), assumed_align=16).mark_layout_dynamic(leading_dim=0)


@torch.library.custom_op(
    "trtllm::k3_spec_accept",
    mutates_args=("block_counts", "block_tables", "prev_acc", "kv_lens", "rng_counter"),
)
def k3_spec_accept(
    logits: torch.Tensor,
    draft: torch.Tensor,
    block_off: torch.Tensor,
    pool_idx: int,
    divisor: int,
    block_counts: torch.Tensor,
    block_tables: torch.Tensor,
    prev_acc: torch.Tensor,
    state_idx: torch.Tensor,
    dummy: torch.Tensor,
    kv_lens: torch.Tensor,
    batch_to_slot: torch.Tensor,
    ctx_len: torch.Tensor,
    max_ctx: int,
    embed: torch.Tensor,
    mask_row: torch.Tensor,
    rng_pool: torch.Tensor,
    rng_counter: torch.Tensor,
    force: float,
    block: int,
    ws_uc: Optional[torch.Tensor] = None,
    ws_mc: Optional[torch.Tensor] = None,
    ws_flags: Optional[torch.Tensor] = None,
    rank: int = 0,
    slots: int = 1,
    push_copies: int = 1,
    shard_offset: int = 0,
) -> List[torch.Tensor]:
    """Returns ``[accepted [B, K + 1] int32, num_accepted [B] int32, rewind [B] int32, bonus [B] int64,
    query_positions [B, block] int64, ctx_positions [B, K + 1] int64, noise [B, block, H] bf16]`` and updates
    ``block_counts[:B]`` / ``block_tables[:B]`` (the draft pool's block table decoded from ``block_off[pool_idx, :B,
    0]`` / ``divisor``), ``prev_acc`` (the KDA replay record), ``kv_lens[:B]`` (+ 1) and ``rng_counter`` (forced
    fractional acceptance). ``logits``: fp32 [B (K + 1), V]; or, with ``slots`` > 1 and a :func:`workspace`
    (``ws_*``, ``rank`` and ``push_copies`` from it), this rank's bf16 columns [B (K + 1), V / TP] starting at
    ``shard_offset``. ``draft``: int32 [B, K]."""
    import cuda.bindings.driver as cuda_driver
    import cutlass.cute as cute

    kern = _kernel_module()
    batch, drafts = draft.shape
    rows, vocab = logits.shape
    hidden = embed.shape[1]
    sharded = slots > 1
    logits_dtype = torch.bfloat16 if sharded else torch.float32
    if (
        logits.dtype != logits_dtype
        or rows != batch * (drafts + 1)
        or draft.dtype != torch.int32
        or not kern.supports(vocab, batch, block, drafts, hidden, slots)
        or (sharded and (ws_uc is None or ws_mc is None or ws_flags is None))
    ):
        raise ValueError(
            f"k3_spec_accept: unsupported call: logits {tuple(logits.shape)} {logits.dtype}, draft "
            f"{tuple(draft.shape)} {draft.dtype}, block {block}, hidden {hidden}, slots {slots} (fp32 [B (K + 1), V], "
            f"or bf16 shards with a workspace; V % {kern.COLS_PER_CTA} == 0, int32 [B <= {kern.MAX_BATCH}, K], "
            f"block <= 16)"
        )  # fmt: skip
    if (
        block_off.dtype != torch.int32
        or block_off.dim() != 4
        or not block_off.is_contiguous()
        or block_tables.dtype != torch.int32
        or block_tables.shape[1] != block_off.shape[3]
        or block_counts.dtype != torch.int64
        or prev_acc.dtype != torch.int32
        or state_idx.dtype != torch.int32
        or kv_lens.dtype != torch.int32
        or batch_to_slot.dtype != torch.int64
        or ctx_len.dtype != torch.int64
        or embed.dtype != torch.bfloat16
        or mask_row.dtype != torch.bfloat16
        or mask_row.numel() != hidden
    ):
        raise ValueError("k3_spec_accept: unexpected state tensor dtypes or layouts")  # fmt: skip
    device = logits.device
    grid = vocab // kern.COLS_PER_CTA
    scratch = _scratch.get((device, grid))
    if scratch is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "trtllm::k3_spec_accept must run once outside CUDA-graph capture first"
            )
        scratch = _scratch[(device, grid)] = (
            torch.empty(kern.MAX_ROWS * grid * 2, dtype=torch.int32, device=device),
            torch.zeros(1, dtype=torch.int32, device=device),
        )
    mode, total, frac = force_mode(force, drafts)
    accepted = torch.empty(batch, drafts + 1, dtype=torch.int32, device=device)
    num_acc = torch.empty(batch, dtype=torch.int32, device=device)
    rewind = torch.empty(batch, dtype=torch.int32, device=device)
    bonus = torch.empty(batch, dtype=torch.int64, device=device)
    qpos = torch.empty(batch, block, dtype=torch.int64, device=device)
    cpos = torch.empty(batch, drafts + 1, dtype=torch.int64, device=device)
    noise = torch.empty(batch, block, hidden, dtype=torch.bfloat16, device=device)
    logits_arg = logits.contiguous().view(-1)
    if sharded:
        logits_arg = logits_arg.view(torch.int32)
    else:
        ws_uc = ws_mc = ws_flags = scratch[1]
    args = (
        _arg(logits_arg),
        _arg(draft.contiguous().view(-1)),
        _arg(block_off.view(-1)),
        _arg(block_counts.view(-1)),
        _arg(block_tables.view(-1)),
        _arg(prev_acc.view(-1)),
        _arg(state_idx.view(-1)),
        _arg(dummy.view(-1).view(torch.uint8)),
        _arg(kv_lens.view(-1)),
        _arg(batch_to_slot.view(-1)),
        _arg(ctx_len.view(-1)),
        _arg(embed.view(-1).view(torch.int32)),
        _arg(mask_row.contiguous().view(-1).view(torch.int32)),
        _arg(rng_pool.view(-1)),
        _arg(rng_counter.view(-1)),
        _arg(accepted.view(-1)),
        _arg(num_acc.view(-1)),
        _arg(rewind.view(-1)),
        _arg(bonus.view(-1)),
        _arg(qpos.view(-1)),
        _arg(cpos.view(-1)),
        _arg(noise.view(-1).view(torch.int32)),
        _arg(scratch[0]),
        _arg(scratch[1]),
        _arg(ws_uc),
        _arg(ws_mc),
        _arg(ws_flags),
    )
    n_seq, max_blocks = block_off.shape[1], block_off.shape[3]
    scalars = (int(pool_idx) * n_seq * 2 * max_blocks, 2 * max_blocks, int(divisor), int(max_ctx), int(total),
               float(frac), int(mode), int(rank), int(shard_offset))  # fmt: skip
    use_pdl = os.environ.get("TRTLLM_ENABLE_PDL", "1") == "1"
    consts = (grid, vocab, batch, drafts, block, hidden, max_blocks, int(slots), int(push_copies))
    key = consts + (use_pdl,)
    fn = _compiled.get(key)
    if fn is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "trtllm::k3_spec_accept must run once per shape outside CUDA-graph capture first"
            )
        with _lock:
            fn = _compiled.get(key)
            if fn is None:
                fn = _compiled[key] = cute.compile(
                    kern.k3_spec_accept, *args, *scalars, *consts, use_pdl,
                    cuda_driver.CUstream(torch.cuda.current_stream(device).cuda_stream),
                )  # fmt: skip
    fn(*args, *scalars, cuda_driver.CUstream(torch.cuda.current_stream(device).cuda_stream))
    return [accepted, num_acc, rewind, bonus, qpos, cpos, noise]


@k3_spec_accept.register_fake
def _(logits, draft, block_off, pool_idx, divisor, block_counts, block_tables, prev_acc, state_idx, dummy, kv_lens,
      batch_to_slot, ctx_len, max_ctx, embed, mask_row, rng_pool, rng_counter, force, block, ws_uc=None, ws_mc=None,
      ws_flags=None, rank=0, slots=1, push_copies=1, shard_offset=0):  # fmt: skip
    batch, drafts = draft.shape
    hidden = embed.shape[1]
    return [
        draft.new_empty((batch, drafts + 1)),
        draft.new_empty((batch,)),
        draft.new_empty((batch,)),
        draft.new_empty((batch,), dtype=torch.int64),
        draft.new_empty((batch, block), dtype=torch.int64),
        draft.new_empty((batch, drafts + 1), dtype=torch.int64),
        embed.new_empty((batch, block, hidden)),
    ]
