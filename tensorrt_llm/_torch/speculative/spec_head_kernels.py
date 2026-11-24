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
"""Fused kernels for the head of a speculative-decoding decode step.

The head is the device work a captured decode step runs before its first
layer: the overlap-scheduler correction of position ids and KV lengths, the
Helix verify-group buffers derived from the corrected positions, and the
token-embedding gather. The engine's eager head issues the position and KV
length additions, the Helix position correction and the Helix buffer
derivation as separate launches, and the embedding is a scalar-load
index-select kernel; here it is one launch for the bookkeeping and one for the
gather. Every value written is the one the eager code writes (integer
arithmetic with PyTorch's floor-division and remainder semantics; the gather
is a copy).

``TLLM_SPEC_HEAD_FUSION=1`` or the model's decode-step shell (``StepShell.spec_head_fusion``)
selects them. Launch shapes depend only on host
batch sizes, so both are CUDA-graph capturable.
"""

import os

import torch
import triton
import triton.language as tl

SPEC_HEAD_FUSION_ENV_VAR = "TLLM_SPEC_HEAD_FUSION"


def spec_head_fusion_enabled() -> bool:
    """Whether the fused decode-step head runs: ``TLLM_SPEC_HEAD_FUSION=1``, or the model's
    decode-step shell asks for it (``StepShell.spec_head_fusion``)."""
    if os.environ.get(SPEC_HEAD_FUSION_ENV_VAR, "0") == "1":
        return True
    from tensorrt_llm._torch.pyexecutor.ull_shell import spec_head_fusion_requested

    return spec_head_fusion_requested()


@triton.jit
def _floor_div(a, b):
    # torch.div(a, b, rounding_mode="floor") for integers (b > 0 here, but the
    # sign correction keeps it exact for any non-zero b).
    q = a // b
    fix = ((a % b) != 0) & ((a < 0) != (b < 0))
    return q - fix.to(q.dtype)


@triton.jit
def _helix_local_len(g, phys, ledger, rank_base):
    # KVCacheManagerV2._helix_local_len: tokens among the first g whose ledger
    # page (page b -> rank b % cp_size) lives on this rank.
    full = _floor_div(g, ledger)
    rem = g - full * ledger
    return full * phys + tl.minimum(tl.maximum(rem - rank_base, 0), phys)


@triton.jit
def _spec_head_prep_kernel(
    pos_ids_ptr,
    pos_offset_ptr,
    num_pos,
    kv_lens_ptr,
    kv_offset_ptr,
    kv_start,
    num_kv,
    helix_pos_ptr,
    helix_slots_ptr,
    helix_bounds_ptr,
    num_helix,
    phys,
    cp_size,
    cp_rank,
    kv_bound_start,
    tokens_per_seq,
    HAS_OVERLAP: tl.constexpr,
    HAS_HELIX: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    t = tl.arange(0, BLOCK_T)
    s = tl.arange(0, BLOCK_S)
    if HAS_OVERLAP:
        # position_ids[0, num_ctx_tokens:] += previous_pos_id_offsets
        pmask = t < num_pos
        pos = tl.load(pos_ids_ptr + t, mask=pmask)
        off = tl.load(pos_offset_ptr + t, mask=pmask)
        tl.store(pos_ids_ptr + t, pos + off, mask=pmask)
        # kv_lens_cuda[first:first + n] += previous_kv_lens_offsets
        kmask = s < num_kv
        kv = tl.load(kv_lens_ptr + kv_start + s, mask=kmask)
        koff = tl.load(kv_offset_ptr + s, mask=kmask)
        tl.store(kv_lens_ptr + kv_start + s, kv + koff, mask=kmask)
    if HAS_HELIX:
        hmask = t < num_helix
        hpos = tl.load(helix_pos_ptr + t, mask=hmask, other=0)
        if HAS_OVERLAP:
            hpos = hpos + tl.load(pos_offset_ptr + t, mask=hmask, other=0)
            tl.store(helix_pos_ptr + t, hpos, mask=hmask)
        owner = _floor_div(hpos, phys)
        owner = owner - _floor_div(owner, cp_size) * cp_size
        ledger = phys * cp_size
        rank_base = cp_rank * phys
        local_before = _helix_local_len(hpos, phys, ledger, rank_base)
        slots = tl.where(owner == cp_rank, local_before, -1)
        tl.store(helix_slots_ptr + t, slots, mask=hmask)
        bounds = _helix_local_len(hpos + 1, phys, ledger, rank_base)
        tl.store(helix_bounds_ptr + t, bounds, mask=hmask)
        # kv_lens_cuda[num_contexts + i] = bound of sequence i's last token.
        # The bounds this program just stored are read back after a barrier.
        tl.debug_barrier()
        num_seqs = num_helix // tokens_per_seq
        smask = s < num_seqs
        last = tl.load(helix_bounds_ptr + s * tokens_per_seq + tokens_per_seq - 1, mask=smask)
        tl.store(kv_lens_ptr + kv_bound_start + s, last, mask=smask)


def fused_spec_head_prep(
    position_ids: torch.Tensor,
    pos_offsets: torch.Tensor,
    num_pos: int,
    kv_lens: torch.Tensor,
    kv_offsets: torch.Tensor,
    kv_start: int,
    num_kv: int,
    helix_positions: torch.Tensor = None,
    helix_slots: torch.Tensor = None,
    helix_bounds: torch.Tensor = None,
    num_helix: int = 0,
    tokens_per_block: int = 1,
    cp_size: int = 1,
    cp_rank: int = 0,
    kv_bound_start: int = 0,
    tokens_per_seq: int = 1,
    has_overlap: bool = True,
) -> None:
    """The decode-step head bookkeeping in one launch.

    With ``has_overlap`` (the overlap scheduler's device correction):
    ``position_ids[:num_pos] += pos_offsets[:num_pos]`` and
    ``kv_lens[kv_start:kv_start + num_kv] += kv_offsets[:num_kv]``. With Helix
    buffers (``num_helix > 0``), what ``TrtllmAttentionMetadata.
    recompute_helix_spec_buffers(num_helix, tokens_per_seq)`` computes after
    ``helix_positions[:num_helix] += pos_offsets[:num_helix]`` (the addition only
    under ``has_overlap``): per-token rank-local write slots (-1 when another
    rank owns the position), per-token rank-local KV bounds, and
    ``kv_lens[kv_bound_start + i]`` = the bound of sequence ``i``'s last token.

    All tensors int32 with unit stride (``position_ids`` is the flat slice
    of the generation tokens' positions).
    """
    for t in (position_ids, pos_offsets, kv_lens, kv_offsets):
        assert t.dtype == torch.int32 and t.stride(-1) == 1
    has_helix = num_helix > 0
    if has_helix:
        for t in (helix_positions, helix_slots, helix_bounds):
            assert t.dtype == torch.int32 and t.is_contiguous()
        assert num_helix % tokens_per_seq == 0
    else:
        helix_positions = helix_slots = helix_bounds = kv_lens
    block_t = triton.next_power_of_2(max(num_pos if has_overlap else 0, num_helix, 1))
    block_s = triton.next_power_of_2(
        max(num_kv if has_overlap else 0, num_helix // max(tokens_per_seq, 1), 1)
    )
    _spec_head_prep_kernel[(1,)](
        position_ids,
        pos_offsets,
        num_pos,
        kv_lens,
        kv_offsets,
        kv_start,
        num_kv,
        helix_positions,
        helix_slots,
        helix_bounds,
        num_helix,
        tokens_per_block,
        cp_size,
        cp_rank,
        kv_bound_start,
        tokens_per_seq,
        HAS_OVERLAP=has_overlap,
        HAS_HELIX=has_helix,
        BLOCK_T=block_t,
        BLOCK_S=block_s,
        num_warps=4,
    )


@triton.jit
def _embedding_gather_kernel(
    ids_ptr, weight_ptr, out_ptr, vocab, hidden, BLOCK_H: tl.constexpr, VEC: tl.constexpr
):
    row = tl.program_id(0)
    offs = tl.program_id(1) * BLOCK_H + tl.arange(0, BLOCK_H)
    offs = tl.max_contiguous(tl.multiple_of(offs, VEC), VEC)
    mask = offs < hidden
    token = tl.load(ids_ptr + row).to(tl.int64)
    valid = (token >= 0) & (token < vocab)
    src = weight_ptr + tl.where(valid, token, 0) * hidden
    vals = tl.load(src + offs, mask=mask & valid, other=0.0)
    tl.store(out_ptr + row.to(tl.int64) * hidden + offs, vals, mask=mask)


def embedding_gather(ids: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """``F.embedding(ids, weight)`` for 1-D ids and a contiguous table, as a
    vectorized row copy (ids outside the table give a zero row)."""
    assert ids.dim() == 1 and weight.dim() == 2 and weight.is_contiguous()
    vocab, hidden = weight.shape
    out = torch.empty((ids.shape[0], hidden), dtype=weight.dtype, device=weight.device)
    if ids.shape[0] == 0:
        return out
    vec = 16 // weight.element_size()
    assert hidden % vec == 0
    block_h = 1024
    _embedding_gather_kernel[(ids.shape[0], triton.cdiv(hidden, block_h))](
        ids, weight, out, vocab, hidden, BLOCK_H=block_h, VEC=vec, num_warps=4
    )
    return out
