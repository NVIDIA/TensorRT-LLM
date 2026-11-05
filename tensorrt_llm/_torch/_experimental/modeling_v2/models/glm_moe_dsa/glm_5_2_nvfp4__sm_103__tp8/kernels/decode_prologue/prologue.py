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
"""Single-launch target decode-metadata prologue and target embedding (decode_prologue).

A decode step's captured forward begins with ``DecoderInputPreparer.preprocess_inputs``:
``on_update_kv_lens()``, the overlap scheduler's accepted-count correction of
the position ids and the KV lengths, and ``on_update_kv_lens()`` again. On the
fused DSA decode-metadata path each call writes the 2-D KV-length view, three
DeepGEMM paged-MQA-logits schedules (each computed into a fresh tensor and
copied into its persistent buffer), the expanded lengths of the DSL atom split,
the fused decode metadata (request index, indexer K-cache slot mappings,
generation indptrs) and, when the indexer is skipped, the dense top-k rows.
The target model then embeds the input ids with the replicated ``Embedding``
(a range mask, ``F.embedding`` and ``masked_fill_``). About thirty serial
launches precede the first layer.

``target_prologue`` produces the identical device state with one launch:
program 0 applies the correction and derives every metadata buffer from the
corrected lengths held in registers (the state the second native call leaves;
the first call's writes are all overwritten by the second); programs 1.. copy
the embedding rows (``weight[id]``, or +0.0 for an id outside the table). The
host replays the host-side statements of both native calls. The schedule is
draft_loop's reproduction of DeepGEMM's SM100 partition, confirmed against the native
op by draft_loop's ``schedule_self_check``. Every decision is made on the host before
any device work; outside the supported domain the native code runs unchanged.

The embedding rows are handed to the model forward through
``take_embedding``: the launch records ``(input_ids, weight, output)`` on the
step's attention metadata, and the model forward consumes it once for exactly
that ``input_ids`` tensor and embedding weight (otherwise it embeds natively).
"""

from typing import Optional

import torch
import triton
import triton.language as tl

from ..draft_loop import draft_step as _draft_step
from ..draft_loop import embedding as _draft_embedding
from ..draft_loop.draft_step import _write_schedule

_MAX_BATCH = 64
_NUM_WARPS = 4
_BLOCK_H = 2048
# Elements of one dense top-k tile (BLOCK_N rows x BLOCK_K columns).
_DENSE_TILE = 4096
_PACKET_ATTR = "_decode_prologue_embedding"


@triton.jit
def _target_prologue_kernel(
    kv_lens_ptr,  # int32 [>= B] (in/out)
    kv_off_ptr,  # int32 [>= B] previous_kv_lens_offsets (CORRECT)
    seq_lens_ptr,  # int32 [>= B]
    pos_ptr,  # int32 [>= N_POS] position ids of the step (in/out, CORRECT)
    pos_off_ptr,  # int32 [>= N_POS] previous_pos_id_offsets (CORRECT)
    kv2d_ptr,  # int32 [max_seqs, cap] (out)
    kv2d_stride,
    sched_ptr,  # int32 [(NUM_SMS + 1) * 2] (out)
    sched_full_ptr,  # int32 [(NUM_SMS + 1) * 2] (out)
    kv_exp_ptr,  # int32 [>= B * EXPAND] (out, EXPAND > 1)
    sched_exp_ptr,  # int32 [(NUM_SMS + 1) * 2] (out, EXPAND > 1)
    block_offsets_ptr,  # int32 [max_seqs, max_blocks]
    block_offsets_stride_0,
    req_idx_ptr,  # int32 [>= num_tokens] (out)
    slot_fp8_ptr,  # int64 [>= num_tokens] (out)
    slot_scale_ptr,  # int64 [>= num_tokens] (out)
    gen_kv_indptr_ptr,  # int64 [>= B + 1] (out)
    gen_cached_indptr_ptr,  # int64 [>= B + 1] (out)
    topk_ptr,  # int32 [>= num_tokens, topk_stride] (out, DENSE)
    topk_stride,
    ids_ptr,  # int32/int64 [N_EMB] (EMBED)
    weight_ptr,  # [vocab_size, HIDDEN] (EMBED)
    emb_ptr,  # [N_EMB, HIDDEN] (out, EMBED)
    vocab_size,
    weight_row_stride,
    emb_row_stride,
    num_tokens,
    N_POS: tl.constexpr,
    B: tl.constexpr,
    CORRECT: tl.constexpr,
    WIDTH: tl.constexpr,
    EXPAND: tl.constexpr,
    NUM_SMS: tl.constexpr,
    SPLIT_KV: tl.constexpr,
    max_blocks: tl.constexpr,
    block_offsets_stride_1: tl.constexpr,
    tokens_per_block: tl.constexpr,
    data_bytes_per_token: tl.constexpr,
    scale_size: tl.constexpr,
    block_stride: tl.constexpr,
    scale_base_offset: tl.constexpr,
    DENSE: tl.constexpr,
    TOPK: tl.constexpr,
    EMBED: tl.constexpr,
    HIDDEN: tl.constexpr,
    NCB: tl.constexpr,
    BLOCK_B: tl.constexpr,
    BLOCK_W: tl.constexpr,
    BLOCK_E: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_SM: tl.constexpr,
    BLOCK_P: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid == 0:
        offs_b = tl.arange(0, BLOCK_B)
        mask_b = offs_b < B

        # ----- accepted-count correction (in place) and the step's lengths ---
        kv = tl.load(kv_lens_ptr + offs_b, mask=mask_b, other=0)
        if CORRECT:
            kv = kv + tl.load(kv_off_ptr + offs_b, mask=mask_b, other=0)
            tl.store(kv_lens_ptr + offs_b, kv, mask=mask_b)
            offs_p = tl.arange(0, BLOCK_P)
            mask_p = offs_p < N_POS
            pos = tl.load(pos_ptr + offs_p, mask=mask_p, other=0)
            pos = pos + tl.load(pos_off_ptr + offs_p, mask=mask_p, other=0)
            tl.store(pos_ptr + offs_p, pos, mask=mask_p)
        kv = tl.where(mask_b, kv, 0)
        seq = tl.load(seq_lens_ptr + offs_b, mask=mask_b, other=0)
        seq = tl.where(mask_b, seq, 0)

        # ----- 2-D KV lengths and the paged-MQA-logits schedules -------------
        offs_w = tl.arange(0, BLOCK_W)
        kv2d_mask = mask_b[:, None] & (offs_w[None, :] < WIDTH)
        tl.store(
            kv2d_ptr + offs_b[:, None] * kv2d_stride + offs_w[None, :],
            kv[:, None] + tl.zeros((BLOCK_B, BLOCK_W), tl.int32),
            mask=kv2d_mask,
        )
        _write_schedule(kv, mask_b, B, sched_ptr, 1, NUM_SMS, SPLIT_KV, BLOCK_SM)
        _write_schedule(kv, mask_b, B, sched_full_ptr, WIDTH, NUM_SMS, SPLIT_KV, BLOCK_SM)
        if EXPAND > 1:
            offs_e = tl.arange(0, BLOCK_E)
            mask_e = offs_e < B * EXPAND
            owner = (offs_e[:, None] // EXPAND) == offs_b[None, :]
            kv_e = tl.sum(tl.where(owner, kv[None, :], 0), 1)
            tl.store(kv_exp_ptr + offs_e, kv_e, mask=mask_e)
            _write_schedule(kv_e, mask_e, B * EXPAND, sched_exp_ptr, 1, NUM_SMS, SPLIT_KV, BLOCK_SM)

        # ----- fused DSA decode metadata (the native kernel's arithmetic) ----
        kv64 = kv.to(tl.int64)
        seq64 = seq.to(tl.int64)
        tl.store(gen_kv_indptr_ptr, tl.full((), 0, tl.int64))
        tl.store(gen_cached_indptr_ptr, tl.full((), 0, tl.int64))
        tl.store(gen_kv_indptr_ptr + 1 + offs_b, tl.cumsum(kv64, 0), mask=mask_b)
        tl.store(gen_cached_indptr_ptr + 1 + offs_b, tl.cumsum(kv64 - seq64, 0), mask=mask_b)

        flat_start = tl.cumsum(seq64, 0) - seq64
        start_pos = kv64 - seq64
        offs_t = tl.arange(0, BLOCK_T).to(tl.int64)
        g = flat_start[:, None] + offs_t[None, :]
        mask_t = mask_b[:, None] & (offs_t[None, :] < seq64[:, None]) & (g < num_tokens)
        tl.store(
            req_idx_ptr + g, offs_b[:, None] + tl.zeros((BLOCK_B, BLOCK_T), tl.int32), mask=mask_t
        )
        gpos = start_pos[:, None] + offs_t[None, :]
        blk = gpos // tokens_per_block
        pos_in_blk = gpos % tokens_per_block
        pos_in_blk = tl.where(pos_in_blk < 0, pos_in_blk + tokens_per_block, pos_in_blk)
        blk = tl.minimum(tl.maximum(blk, 0), max_blocks - 1)
        bid = tl.load(
            block_offsets_ptr
            + offs_b[:, None].to(tl.int64) * block_offsets_stride_0
            + blk * block_offsets_stride_1,
            mask=mask_t,
            other=0,
        ).to(tl.int64)
        tl.store(
            slot_fp8_ptr + g, bid * block_stride + pos_in_blk * data_bytes_per_token, mask=mask_t
        )
        tl.store(
            slot_scale_ptr + g,
            bid * block_stride + scale_base_offset + pos_in_blk * scale_size,
            mask=mask_t,
        )

        # ----- dense top-k rows of a skipped indexer ------------------------
        if DENSE:
            # position(g) = (kv - seq)[b] + (g - flat_start[b]) for the request b
            # whose token range holds g (``_get_dense_topk_indices``).
            offs_n = tl.arange(0, BLOCK_N).to(tl.int64)
            seq_end = tl.cumsum(seq64, 0)
            owned = (seq_end[None, :] <= offs_n[:, None]) & mask_b[None, :]
            req = tl.sum(owned.to(tl.int32), 1)
            onehot = req[:, None] == offs_b[None, :]
            shift = tl.sum(tl.where(onehot, (start_pos - flat_start)[None, :], 0), 1)
            tok_pos = offs_n + shift
            mask_n = offs_n < num_tokens
            for k0 in tl.static_range(0, TOPK, BLOCK_K):
                offs_k = k0 + tl.arange(0, BLOCK_K)
                rows = tl.where(
                    offs_k[None, :].to(tl.int64) <= tok_pos[:, None], offs_k[None, :], -1
                )
                tl.store(
                    topk_ptr + offs_n[:, None] * topk_stride + offs_k[None, :],
                    rows.to(tl.int32),
                    mask=mask_n[:, None] & (offs_k[None, :] < TOPK),
                )
    else:
        if EMBED:
            e = pid - 1
            row = e // NCB
            cb = e % NCB
            token = tl.load(ids_ptr + row).to(tl.int64)
            valid = (token >= 0) & (token < vocab_size)
            safe_token = tl.where(valid, token, 0)
            cols = cb * BLOCK_H + tl.arange(0, BLOCK_H)
            col_mask = cols < HIDDEN
            values = tl.load(
                weight_ptr + safe_token * weight_row_stride + cols,
                mask=col_mask & valid,
                other=0.0,
            )
            tl.store(emb_ptr + row.to(tl.int64) * emb_row_stride + cols, values, mask=col_mask)


def _metadata_ineligibility(md, B: int) -> Optional[str]:
    """Why the fused-metadata part cannot replace the native calls, or None."""
    from tensorrt_llm._torch.attention.backends.sparse.dsa import is_dsa_cache_manager
    from tensorrt_llm._torch.attention.backends.sparse.dsa.metadata import (
        DSAtrtllmAttentionMetadata,
        _fused_dsa_meta_enabled,
    )

    if not isinstance(md, DSAtrtllmAttentionMetadata):
        return "metadata is not DSAtrtllmAttentionMetadata"
    cls = type(md)
    for name in ("on_update_kv_lens", "_compute_kv_lens_row_reorder", "get_indexer_kv_lens"):
        if getattr(cls, name) is not getattr(DSAtrtllmAttentionMetadata, name):
            return f"metadata overrides {name}"
    if md.kv_cache_manager is None or not is_dsa_cache_manager(md.kv_cache_manager):
        return "KV cache manager is not a DSA cache manager"
    if not _fused_dsa_meta_enabled():
        return "native fused DSA metadata disabled"
    if md.num_contexts != 0 or md.num_generations <= 0 or md.num_tokens <= 0:
        return "not a generation-only step"
    if md.use_fp8_ds_mla or getattr(md, "fp4_mla_state", None) is not None:
        return "FP8-DS/FP4 MLA metadata state"
    if getattr(md, "enable_helix", False):
        return "helix metadata"
    if md.enable_ragged_verification or md.is_ragged_verify:
        return "ragged verification"
    if md._indexer_compress_ratio > 1:
        return "compressed indexer lengths"
    if md.use_expanded_buffers_for_mtp:
        return "expanded MTP buffers"
    if md.max_draft_tokens <= 0:
        return "no draft tokens"
    if B <= 0 or B > _MAX_BATCH or B != md.num_seqs or B != md.num_generations:
        return f"batch size {B} outside the fused domain"
    if md.needs_gvr_prior and B * md.gen_token_stride >= 2 * md.num_sms:
        return "GVR row reorder active"
    host_seq = md.seq_lens[:B]
    if int(host_seq.sum()) != md.num_tokens or int(host_seq.max()) > 1 + md.max_draft_tokens:
        return "query lengths outside the decode layout"
    int32_attrs = (
        "kv_lens_cuda",
        "seq_lens_cuda",
        "kv_lens_cuda_2d",
        "indexer_k_cache_block_offsets",
        "scheduler_metadata_buffer",
        "scheduler_metadata_buffer_full_next_n",
        "req_idx_per_token",
    )
    for name in int32_attrs:
        if getattr(md, name).dtype != torch.int32:
            return f"unexpected dtype of {name}"
    for name in (
        "slot_mapping_fp8",
        "slot_mapping_scale",
        "gen_kv_indptr",
        "gen_cached_token_indptr",
    ):
        if getattr(md, name).dtype != torch.int64:
            return f"unexpected dtype of {name}"
    if md.expand_for_dsl and md.dsl_expand_factor > 1:
        if md.kv_lens_expanded_cuda.dtype != torch.int32:
            return "unexpected dtype of kv_lens_expanded_cuda"
        if B * md.dsl_expand_factor > md.kv_lens_expanded_cuda.shape[0]:
            return "expanded lengths exceed their buffer"
    if md.skip_indexer_for_gen_reqs:
        topk = md.topk_indices_buffer
        if topk.dtype != torch.int32 or topk.stride(1) != 1 or topk.shape[1] != md.num_sparse_topk:
            return "unexpected dense top-k buffer layout"
    if _draft_step.schedule_self_check(md.num_sms, md.kv_lens_cuda.device) is None:
        return "DeepGEMM schedule not reproduced"
    return None


def _correction_ineligibility(preparer, inputs, md, B: int) -> Optional[str]:
    """Why the accepted-count correction cannot be fused, or None."""
    pos = inputs.get("position_ids")
    ids = inputs.get("input_ids")
    if pos is None or ids is None:
        return "no position ids or input ids"
    if pos.dim() != 2 or pos.shape[0] != 1 or pos.dtype != torch.int32 or pos.stride(1) != 1:
        return "position ids are not a [1, N] int32 row"
    num_pos = ids.shape[0] - md.num_ctx_tokens
    if md.num_ctx_tokens != 0 or pos.shape[1] != num_pos:
        return "position ids do not match the input ids"
    # Program 0 rewrites the positions while the embedding programs read the ids.
    if pos.untyped_storage().data_ptr() == ids.untyped_storage().data_ptr():
        return "position ids share storage with the input ids"
    buffers = getattr(preparer, "_buffers", None)
    pos_off = getattr(buffers, "previous_pos_id_offsets_cuda", None)
    kv_off = getattr(buffers, "previous_kv_lens_offsets_cuda", None)
    if pos_off is None or kv_off is None:
        return "no overlap-scheduler offsets"
    if pos_off.dtype != torch.int32 or kv_off.dtype != torch.int32:
        return "unexpected offset dtype"
    if not pos_off.is_contiguous() or not kv_off.is_contiguous():
        return "offsets are not contiguous"
    if pos_off.shape[0] < num_pos or kv_off.shape[0] < B:
        return "offsets shorter than the step"
    if getattr(md, "num_chunked_ctx_requests", 0) != 0:
        return "chunked context rows"
    return None


def _embedding_target(preparer, inputs):
    """The target ``Embedding`` decode_prologue may run, or (None, reason)."""
    from tensorrt_llm._torch.models.modeling_deepseekv3 import DeepseekV3Model

    if inputs.get("inputs_embeds") is not None:
        return None, "inputs_embeds given"
    target = getattr(getattr(preparer, "model", None), "model", None)
    if not isinstance(target, DeepseekV3Model):
        return None, "target model is not DeepseekV3Model"
    embed = getattr(target, "embed_tokens", None)
    reason = _draft_embedding.module_ineligibility(embed)
    if reason is not None:
        return None, reason
    ids = inputs.get("input_ids")
    reason = _draft_embedding.ids_ineligibility(ids, embed.weight)
    if reason is not None:
        return None, reason
    return embed, None


def _replay_host_on_update_kv_lens(md) -> None:
    """Host statements of one fused-eligible ``on_update_kv_lens()`` (no device work)."""
    from tensorrt_llm._torch.attention.backends.sparse.dsa.metadata import (
        DSAtrtllmAttentionMetadata,
    )
    from tensorrt_llm.logger import logger

    super(DSAtrtllmAttentionMetadata, md).on_update_kv_lens()
    md._invalidate_pool_view_cache()
    md.gen_indexer_kv_lens_cuda_runtime = md.get_indexer_kv_lens(
        md.kv_lens_cuda[md.num_contexts : md.num_seqs]
    )
    if not getattr(md, "_fused_dsa_meta_armed", False):
        logger.info(
            "Fused DSA decode-metadata "
            f"kernel armed (num_seqs={md.num_seqs}, "
            f"num_tokens={md.num_tokens})."
        )
        md._fused_dsa_meta_armed = True
    # In the eligible domain the row reorder is off, so this only resets the
    # host attribute; the dense top-k rows are written by the launch.
    md._compute_kv_lens_row_reorder()


def prologue_ineligibility(preparer, inputs, enable_spec_decode: bool) -> Optional[str]:
    """Return why ``target_prologue`` must leave the native code in place, or None."""
    md = inputs.get("attn_metadata")
    if md is None:
        return "no attention metadata"
    mapping = getattr(preparer, "mapping", None)
    if mapping is not None and mapping.has_cp_helix():
        return "helix context parallelism"
    B = md.num_seqs
    reason = _metadata_ineligibility(md, B)
    if reason is not None:
        return reason
    config = getattr(preparer, "_config", None)
    if config is None:
        return "no preparer config"
    if enable_spec_decode and not config.disable_overlap_scheduler:
        return _correction_ineligibility(preparer, inputs, md, B)
    return None


def target_prologue(preparer, inputs, *, enable_spec_decode: bool, runtime_draft_len: int) -> bool:
    """Run the decode-metadata prologue (and the target embedding) in one launch.

    Replaces the device work and reproduces the host statements of
    ``DecoderInputPreparer.preprocess_inputs`` up to (not including) the guided
    decoder event, which the caller keeps.

    Args:
        preparer: the ``DecoderInputPreparer`` (reads ``_config``, ``mapping``,
            ``model`` and the overlap-scheduler offset buffers in ``_buffers``).
        inputs: the step's input dict (``attn_metadata``, ``input_ids``,
            ``position_ids``).
        enable_spec_decode: as passed to ``preprocess_inputs``.
        runtime_draft_len: as passed to ``preprocess_inputs`` (unused by the
            fused path; the DSA branch of the native code does not read it).

    Returns:
        True when the launch replaced the native prologue; False when the
        caller must run the native code (nothing has been modified then).
    """
    from ... import dispatch

    del runtime_draft_len
    reason = prologue_ineligibility(preparer, inputs, enable_spec_decode)
    # One log line per distinct outcome: context and warmup steps decline
    # before the first decode step engages.
    key = "target_prologue" if reason is None else f"target_prologue[{reason}]"
    dispatch.log_capability("decode_prologue", key, reason is None, reason or "")
    if reason is not None:
        return False

    md = inputs["attn_metadata"]
    B = md.num_seqs
    correct = enable_spec_decode and not preparer._config.disable_overlap_scheduler
    embed, embed_reason = _embedding_target(preparer, inputs)
    key = "target_embedding" if embed is not None else f"target_embedding[{embed_reason}]"
    dispatch.log_capability("decode_prologue", key, embed is not None, embed_reason or "")

    kv_mgr = md.kv_cache_manager
    index_head_dim = kv_mgr.index_head_dim
    use_fp4 = getattr(kv_mgr, "use_fp4", False)
    data_bytes_per_token = index_head_dim // 2 if use_fp4 else index_head_dim
    scale_size = index_head_dim // kv_mgr.quant_block_size * 4
    tokens_per_block = md._tokens_per_block
    block_stride = tokens_per_block * (data_bytes_per_token + scale_size)
    width = min(md.gen_token_stride, md.kv_lens_cuda_2d.shape[1])
    expand = md.dsl_expand_factor if md.expand_for_dsl else 1
    num_sms = md.num_sms
    block_offsets = md.indexer_k_cache_block_offsets
    dense = bool(md.skip_indexer_for_gen_reqs)
    num_tokens = md.num_tokens
    dummy = md.kv_lens_cuda

    if correct:
        pos = inputs["position_ids"]
        num_pos = inputs["input_ids"].shape[0] - md.num_ctx_tokens
        pos_off = preparer._buffers.previous_pos_id_offsets_cuda
        kv_off = preparer._buffers.previous_kv_lens_offsets_cuda
    else:
        pos = pos_off = kv_off = dummy
        num_pos = 1

    if embed is not None:
        ids = inputs["input_ids"]
        weight = embed.weight
        num_rows = ids.shape[0]
        hidden = weight.shape[1]
        emb_out = torch.empty((num_rows, hidden), dtype=weight.dtype, device=weight.device)
        ncb = triton.cdiv(hidden, _BLOCK_H)
        grid = (1 + num_rows * ncb,)
    else:
        ids = weight = emb_out = dummy
        num_rows = 0
        hidden = 1
        ncb = 1
        grid = (1,)

    topk = md.topk_indices_buffer if dense else dummy
    block_n = max(2, triton.next_power_of_2(num_tokens))
    _target_prologue_kernel[grid](
        md.kv_lens_cuda,
        kv_off,
        md.seq_lens_cuda,
        pos,
        pos_off,
        md.kv_lens_cuda_2d,
        md.kv_lens_cuda_2d.stride(0),
        md.scheduler_metadata_buffer,
        md.scheduler_metadata_buffer_full_next_n,
        md.kv_lens_expanded_cuda if expand > 1 else dummy,
        md.scheduler_metadata_buffer_expanded if expand > 1 else dummy,
        block_offsets,
        block_offsets.stride(0),
        md.req_idx_per_token,
        md.slot_mapping_fp8,
        md.slot_mapping_scale,
        md.gen_kv_indptr,
        md.gen_cached_token_indptr,
        topk,
        topk.stride(0) if dense else 0,
        ids,
        weight,
        emb_out,
        weight.shape[0] if embed is not None else 0,
        weight.stride(0) if embed is not None else 0,
        emb_out.stride(0) if embed is not None else 0,
        num_tokens,
        N_POS=num_pos,
        B=B,
        CORRECT=correct,
        WIDTH=width,
        EXPAND=expand,
        NUM_SMS=num_sms,
        SPLIT_KV=_draft_step._SPLIT_KV_BY_SMS[num_sms],
        max_blocks=block_offsets.shape[1],
        block_offsets_stride_1=block_offsets.stride(1),
        tokens_per_block=tokens_per_block,
        data_bytes_per_token=data_bytes_per_token,
        scale_size=scale_size,
        block_stride=block_stride,
        scale_base_offset=tokens_per_block * data_bytes_per_token,
        DENSE=dense,
        TOPK=md.num_sparse_topk if dense else 1,
        EMBED=embed is not None,
        HIDDEN=hidden,
        NCB=ncb,
        BLOCK_B=max(2, triton.next_power_of_2(B)),
        BLOCK_W=max(2, triton.next_power_of_2(width)),
        BLOCK_E=max(2, triton.next_power_of_2(B * expand)),
        BLOCK_T=max(2, triton.next_power_of_2(1 + md.max_draft_tokens)),
        BLOCK_SM=triton.next_power_of_2(num_sms + 1),
        BLOCK_P=max(2, triton.next_power_of_2(num_pos)),
        BLOCK_N=block_n,
        BLOCK_K=max(16, _DENSE_TILE // block_n),
        BLOCK_H=_BLOCK_H,
        num_warps=_NUM_WARPS,
    )

    # Host statements of the native sequence: on_update_kv_lens() (call A),
    # the correction (device only), on_update_kv_lens() (call B).
    _replay_host_on_update_kv_lens(md)
    if correct:
        _replay_host_on_update_kv_lens(md)

    if embed is not None:
        setattr(md, _PACKET_ATTR, (inputs["input_ids"], weight, emb_out))
    return True


def take_embedding(attn_metadata, input_ids: torch.Tensor, embed_tokens) -> Optional[torch.Tensor]:
    """Return the target embedding rows the prologue launch produced, or None.

    Consumes the step's packet. The rows are returned only for exactly the
    ``input_ids`` tensor and the embedding weight the launch used; otherwise
    the caller embeds natively (the launch's rows are then unused).

    Args:
        attn_metadata: the step's attention metadata.
        input_ids: ``[N]`` input ids the model forward embeds.
        embed_tokens: the model's ``Embedding`` module.

    Returns:
        ``[N, hidden]`` embedding rows, or None.
    """
    packet = vars(attn_metadata).pop(_PACKET_ATTR, None)
    if packet is None:
        return None
    ids, weight, rows = packet
    if ids is not input_ids or getattr(embed_tokens, "weight", None) is not weight:
        from ... import dispatch

        dispatch.log_capability("decode_prologue", "take_embedding", False, "packet does not match")
        return None
    return rows


class DecodePrologueHook:
    """The decode-metadata prologue and the target embedding in one launch (``DecodePrologue``).

    The engine calls it first in the step's input preprocessing, inside CUDA graph capture
    as well (``target_prologue``); the decode target then takes the rows that launch
    embedded (``take_embedding``).
    """

    def __call__(
        self, preparer, inputs, *, enable_spec_decode: bool, runtime_draft_len: int
    ) -> bool:
        return target_prologue(
            preparer,
            inputs,
            enable_spec_decode=enable_spec_decode,
            runtime_draft_len=runtime_draft_len,
        )

    def take_embedding(
        self, attn_metadata, input_ids: torch.Tensor, embed_tokens
    ) -> Optional[torch.Tensor]:
        return take_embedding(attn_metadata, input_ids, embed_tokens)
