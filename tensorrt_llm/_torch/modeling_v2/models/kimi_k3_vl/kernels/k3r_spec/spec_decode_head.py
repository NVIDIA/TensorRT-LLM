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
"""The head of a speculative decode step in one launch.

The head is the device work a captured decode step runs before its first layer:
the overlap-scheduler correction of position ids and KV lengths and the Helix
verify-group buffers derived from the corrected positions
(``TrtllmAttentionMetadata.recompute_helix_spec_buffers``). ``SpecDecodeHead``
is a ``RuntimeHooks.decode_prologue`` that writes the same values with one
kernel (``ops/spec/k3_spec_decode_head``) instead of one launch per buffer.

It reads main's ``DecoderInputPreparer`` and ``TrtllmAttentionMetadata`` directly
(``_config.disable_overlap_scheduler``, ``_buffers.previous_*_offsets_cuda``,
``_helix_spec_tokens_valid``); ``test_k3_engine_probes`` pins those names.
"""

from typing import TYPE_CHECKING, Any, Optional

import torch
import triton
import triton.language as tl

from tensorrt_llm._torch.attention.backends.trtllm import TrtllmAttentionMetadata

# The module, not the instance: the entry imports this module for its launcher.
from ...ops.spec import k3_spec_decode_head as _entry

if TYPE_CHECKING:
    from tensorrt_llm._torch.pyexecutor.engine.runners.decoder.inputs import DecoderInputPreparer


@triton.jit
def _helix_local_len(g, tokens_per_block, ledger, rank_base):
    # Rank-local prefix length of the first g global tokens (populate_helix_spec_buffers).
    full = g // ledger
    return full * tokens_per_block + tl.minimum(
        tl.maximum(g - full * ledger - rank_base, 0), tokens_per_block
    )


_SCALARS = [
    "num_pos",
    "kv_start",
    "num_kv",
    "num_helix",
    "tokens_per_block",
    "cp_size",
    "cp_rank",
    "kv_bound_start",
    "tokens_per_seq",
]


@triton.jit(do_not_specialize=_SCALARS, do_not_specialize_on_alignment=_SCALARS)
def _spec_head_kernel(
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
    helix_zero_ptr,
    num_helix,
    tokens_per_block,
    cp_size,
    cp_rank,
    kv_bound_start,
    tokens_per_seq,
    HAS_OVERLAP: tl.constexpr,
    HAS_HELIX: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # One program: the per-sequence KV lengths read back the bounds it stores.
    lanes = tl.arange(0, BLOCK)
    if HAS_OVERLAP:
        for start in range(0, num_pos, BLOCK):
            t = start + lanes
            mask = t < num_pos
            pos = tl.load(pos_ids_ptr + t, mask=mask)
            tl.store(pos_ids_ptr + t, pos + tl.load(pos_offset_ptr + t, mask=mask), mask=mask)
        for start in range(0, num_kv, BLOCK):
            s = start + lanes
            mask = s < num_kv
            kv = tl.load(kv_lens_ptr + kv_start + s, mask=mask)
            tl.store(
                kv_lens_ptr + kv_start + s, kv + tl.load(kv_offset_ptr + s, mask=mask), mask=mask
            )
    if HAS_HELIX:
        ledger = tokens_per_block * cp_size
        rank_base = cp_rank * tokens_per_block
        for start in range(0, num_helix, BLOCK):
            t = start + lanes
            mask = t < num_helix
            position = tl.load(helix_pos_ptr + t, mask=mask, other=0)
            if HAS_OVERLAP:
                position += tl.load(pos_offset_ptr + t, mask=mask, other=0)
                tl.store(helix_pos_ptr + t, position, mask=mask)
            local_before = _helix_local_len(position, tokens_per_block, ledger, rank_base)
            bound = _helix_local_len(position + 1, tokens_per_block, ledger, rank_base)
            owned = (position // tokens_per_block) % cp_size == cp_rank
            tl.store(helix_slots_ptr + t, tl.where(owned, local_before, -1), mask=mask)
            tl.store(helix_bounds_ptr + t, bound, mask=mask)
            tl.store(helix_zero_ptr + t, bound == 0, mask=mask)
        # A sequence's KV length is the bound of its last token. The lane that
        # stores it is the one that stored its overlap-corrected value above.
        tl.debug_barrier()
        num_seqs = num_helix // tokens_per_seq
        for start in range(0, num_seqs, BLOCK):
            s = start + lanes
            mask = s < num_seqs
            last = tl.load(helix_bounds_ptr + s * tokens_per_seq + tokens_per_seq - 1, mask=mask)
            tl.store(kv_lens_ptr + kv_bound_start + s, last, mask=mask)


def fused_spec_head_prep(
    position_ids: torch.Tensor,
    pos_offsets: torch.Tensor,
    num_pos: int,
    kv_lens: torch.Tensor,
    kv_offsets: torch.Tensor,
    kv_start: int,
    num_kv: int,
    *,
    has_overlap: bool,
    helix_positions: Optional[torch.Tensor] = None,
    helix_slots: Optional[torch.Tensor] = None,
    helix_bounds: Optional[torch.Tensor] = None,
    helix_zero_kv: Optional[torch.Tensor] = None,
    num_helix: int = 0,
    tokens_per_block: int = 1,
    cp_size: int = 1,
    cp_rank: int = 0,
    kv_bound_start: int = 0,
    tokens_per_seq: int = 1,
) -> None:
    """The decode-step head bookkeeping in one launch.

    With ``has_overlap``: ``position_ids[:num_pos] += pos_offsets[:num_pos]`` and
    ``kv_lens[kv_start:kv_start + num_kv] += kv_offsets[:num_kv]``. With Helix
    buffers (``num_helix > 0``), after ``helix_positions[:num_helix] +=
    pos_offsets[:num_helix]`` (only with ``has_overlap``), what
    ``TrtllmAttentionMetadata.recompute_helix_spec_buffers(num_helix, tokens_per_seq)``
    writes: rank-local write slots, rank-local KV bounds, the zero-bound mask and
    ``kv_lens[kv_bound_start + i]`` = the bound of sequence ``i``'s last token.

    All integer tensors are int32 with unit stride; ``helix_zero_kv`` is bool.
    """
    for tensor in (position_ids, pos_offsets, kv_lens, kv_offsets):
        assert tensor.dtype == torch.int32 and tensor.stride(-1) == 1
    has_helix = num_helix > 0
    if has_helix:
        for tensor in (helix_positions, helix_slots, helix_bounds):
            assert tensor.dtype == torch.int32 and tensor.is_contiguous()
        assert helix_zero_kv.dtype == torch.bool and helix_zero_kv.is_contiguous()
        assert num_helix % tokens_per_seq == 0
    else:
        helix_positions = helix_slots = helix_bounds = helix_zero_kv = kv_lens
    _spec_head_kernel[(1,)](
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
        helix_zero_kv,
        num_helix,
        tokens_per_block,
        cp_size,
        cp_rank,
        kv_bound_start,
        tokens_per_seq,
        HAS_OVERLAP=has_overlap,
        HAS_HELIX=has_helix,
        BLOCK=512,
        num_warps=4,
    )


def spec_decode_kv_range(attn_metadata: TrtllmAttentionMetadata) -> tuple[int, int]:
    """``(first sequence, count)`` whose KV lengths a speculative step corrects: the generation
    requests, or the context requests they run as when they extend a context
    (``num_chunked_ctx_requests``); the two branches of ``DecoderInputPreparer.preprocess_inputs``."""
    num_ctx = attn_metadata.num_contexts
    num_chunked = attn_metadata.num_chunked_ctx_requests
    if num_ctx >= num_chunked and num_chunked > 0:
        return num_ctx - num_chunked, num_chunked
    return num_ctx, attn_metadata.num_generations


class SpecDecodeHead:
    """``RuntimeHooks.decode_prologue`` running a speculative decode step's head in one kernel.

    Covers int32 two-dimensional position ids and int32 KV lengths of
    TRT-LLM attention metadata; any other step keeps the native prologue.
    The kernel also writes the per-token zero-KV flags (``helix_kv_bounds == 0``);
    main's metadata keeps no buffer for them, so they go to a scratch buffer here.
    """

    def __init__(self) -> None:
        self._zero_kv: Optional[torch.Tensor] = None

    def _zero_kv_scratch(self, like: torch.Tensor) -> torch.Tensor:
        scratch = self._zero_kv
        if scratch is None or scratch.device != like.device or scratch.numel() < like.numel():
            scratch = torch.empty(like.numel(), dtype=torch.bool, device=like.device)
            self._zero_kv = scratch
        return scratch

    def __call__(
        self,
        preparer: "DecoderInputPreparer",
        inputs: dict[str, Any],
        *,
        enable_spec_decode: bool,
        runtime_draft_len: int,
    ) -> bool:
        md = inputs.get("attn_metadata")
        if (
            not enable_spec_decode
            or not isinstance(md, TrtllmAttentionMetadata)
            or md.kv_cache_manager is None
        ):
            return False
        kv_lens = md.kv_lens_cuda
        overlap = not preparer._config.disable_overlap_scheduler
        helix = preparer.mapping.has_cp_helix() and md._helix_spec_tokens_valid
        position_ids = inputs["position_ids"]
        if (
            not (overlap or helix)
            or kv_lens.dtype != torch.int32
            or position_ids.ndim != 2
            or position_ids.dtype != torch.int32
        ):
            return False
        num_ctx_tokens = md.num_ctx_tokens
        gen_tokens = inputs["input_ids"].shape[0] - num_ctx_tokens
        num_ctx_requests = md.num_contexts
        kv_start, num_kv = spec_decode_kv_range(md)
        helix_kwargs = {}
        if helix:
            tokens_per_seq = preparer.get_runtime_tokens_per_gen_step(runtime_draft_len)
            assert gen_tokens % tokens_per_seq == 0 and (
                gen_tokens // tokens_per_seq == md.num_generations
            ), (
                f"helix spec expects uniform verify groups: {gen_tokens} gen tokens over "
                f"{md.num_generations} generation rows of width {tokens_per_seq}"
            )
            helix_kwargs = dict(
                helix_positions=md.helix_position_offsets,
                helix_slots=md.helix_local_slots,
                helix_bounds=md.helix_kv_bounds,
                helix_zero_kv=self._zero_kv_scratch(md.helix_kv_bounds),
                num_helix=gen_tokens,
                tokens_per_block=md.kv_cache_manager.tokens_per_block,
                cp_size=md.mapping.cp_size,
                cp_rank=md.mapping.cp_rank,
                kv_bound_start=num_ctx_requests,
                tokens_per_seq=tokens_per_seq,
            )
        buffers = preparer._buffers
        _entry.k3_spec_decode_head(
            position_ids[0, num_ctx_tokens:],
            buffers.previous_pos_id_offsets_cuda,
            gen_tokens,
            kv_lens,
            buffers.previous_kv_lens_offsets_cuda,
            kv_start,
            num_kv,
            has_overlap=overlap,
            **helix_kwargs,
        )
        md.on_update_kv_lens()
        return True


spec_decode_head = SpecDecodeHead()
