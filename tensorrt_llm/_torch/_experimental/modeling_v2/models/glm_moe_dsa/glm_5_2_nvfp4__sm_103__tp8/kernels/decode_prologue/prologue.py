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
"""Single-launch target decode prologue and target embedding (decode_prologue).

A decode step's captured forward begins with ``DecoderInputPreparer.preprocess_inputs``,
which applies the overlap scheduler's accepted-count correction to the position ids; the
KV-length correction and the DSA decode metadata derived from the corrected lengths are
already in place, from the metadata's ``prepare(BatchInputs(kv_lens_offsets=...))``.
The target model then embeds the input ids with the replicated ``Embedding`` (a range
mask, ``F.embedding`` and ``masked_fill_``).

``target_prologue`` produces the identical device state with one launch: program 0
applies the position correction; programs 1.. copy the embedding rows (``weight[id]``, or
+0.0 for an id outside the table). Every decision is made on the host before any device work;
outside the supported domain the native code runs unchanged.

The embedding rows are handed to the model forward through ``take_embedding``: the
launch records ``(attention metadata, input_ids, weight, output)`` on the model's
``DecodePrologueHook``, and the model forward consumes it once for exactly that
metadata, ``input_ids`` tensor and embedding weight (otherwise it embeds natively).
"""

from typing import Optional

import torch
import triton
import triton.language as tl

from ..draft_loop import embedding as _draft_embedding

_MAX_BATCH = 64
_NUM_WARPS = 4
_BLOCK_H = 2048


@triton.jit
def _target_prologue_kernel(
    pos_ptr,  # int32 [>= N_POS] position ids of the step (in/out, CORRECT)
    pos_off_ptr,  # int32 [>= N_POS] previous_pos_id_offsets (CORRECT)
    ids_ptr,  # int32/int64 [N_EMB] (EMBED)
    weight_ptr,  # [vocab_size, HIDDEN] (EMBED)
    emb_ptr,  # [N_EMB, HIDDEN] (out, EMBED)
    vocab_size,
    weight_row_stride,
    emb_row_stride,
    N_POS: tl.constexpr,
    CORRECT: tl.constexpr,
    EMBED: tl.constexpr,
    HIDDEN: tl.constexpr,
    NCB: tl.constexpr,
    BLOCK_P: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid == 0:
        # ----- accepted-count correction of the position ids (in place) -----
        if CORRECT:
            offs_p = tl.arange(0, BLOCK_P)
            mask_p = offs_p < N_POS
            pos = tl.load(pos_ptr + offs_p, mask=mask_p, other=0)
            pos = pos + tl.load(pos_off_ptr + offs_p, mask=mask_p, other=0)
            tl.store(pos_ptr + offs_p, pos, mask=mask_p)
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
    if pos_off is None:
        return "no overlap-scheduler offsets"
    if pos_off.dtype != torch.int32:
        return "unexpected offset dtype"
    if not pos_off.is_contiguous():
        return "offsets are not contiguous"
    if pos_off.shape[0] < num_pos:
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


def prologue_ineligibility(preparer, inputs, enable_spec_decode: bool) -> Optional[str]:
    """Return why ``target_prologue`` must leave the native code in place, or None."""
    md = inputs.get("attn_metadata")
    if md is None:
        return "no attention metadata"
    mapping = getattr(preparer, "mapping", None)
    if mapping is not None and mapping.has_cp_helix():
        return "helix context parallelism"
    B = md.num_seqs
    if md.num_contexts != 0 or md.num_generations <= 0 or md.num_tokens <= 0:
        return "not a generation-only step"
    if B <= 0 or B > _MAX_BATCH:
        return f"batch size {B} outside the fused domain"
    config = getattr(preparer, "_config", None)
    if config is None:
        return "no preparer config"
    if enable_spec_decode and not config.disable_overlap_scheduler:
        return _correction_ineligibility(preparer, inputs, md, B)
    return None


def target_prologue(
    preparer,
    inputs,
    *,
    enable_spec_decode: bool,
    runtime_draft_len: int,
    hook: Optional["DecodePrologueHook"] = None,
) -> bool:
    """Run the decode-metadata prologue (and the target embedding) in one launch.

    Replaces the device work of ``DecoderInputPreparer.preprocess_inputs`` up to
    (not including) the guided decoder event, which the caller keeps.

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
    correct = enable_spec_decode and not preparer._config.disable_overlap_scheduler
    embed, embed_reason = _embedding_target(preparer, inputs)
    key = "target_embedding" if embed is not None else f"target_embedding[{embed_reason}]"
    dispatch.log_capability("decode_prologue", key, embed is not None, embed_reason or "")

    if not correct and embed is None:
        # Nothing of the native prologue to replace on this step.
        return False
    dummy = md.seq_lens_cuda

    if correct:
        pos = inputs["position_ids"]
        num_pos = inputs["input_ids"].shape[0] - md.num_ctx_tokens
        pos_off = preparer._buffers.previous_pos_id_offsets_cuda
    else:
        pos = pos_off = dummy
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

    _target_prologue_kernel[grid](
        pos,
        pos_off,
        ids,
        weight,
        emb_out,
        weight.shape[0] if embed is not None else 0,
        weight.stride(0) if embed is not None else 0,
        emb_out.stride(0) if embed is not None else 0,
        N_POS=num_pos,
        CORRECT=correct,
        EMBED=embed is not None,
        HIDDEN=hidden,
        NCB=ncb,
        BLOCK_P=max(2, triton.next_power_of_2(num_pos)),
        BLOCK_H=_BLOCK_H,
        num_warps=_NUM_WARPS,
    )

    if embed is not None and hook is not None:
        hook.embedding_packet = (md, inputs["input_ids"], weight, emb_out)
    return True


def take_embedding(
    attn_metadata,
    input_ids: torch.Tensor,
    embed_tokens,
    hook: Optional["DecodePrologueHook"] = None,
) -> Optional[torch.Tensor]:
    """Return the target embedding rows the prologue launch produced, or None.

    Consumes the packet ``hook`` holds. The rows are returned only for exactly
    the attention metadata, ``input_ids`` tensor and embedding weight the
    launch used; otherwise the caller embeds natively (the launch's rows are
    then unused).

    Args:
        attn_metadata: the step's attention metadata.
        input_ids: ``[N]`` input ids the model forward embeds.
        embed_tokens: the model's ``Embedding`` module.
        hook: the model's ``DecodePrologueHook`` the launch recorded on.

    Returns:
        ``[N, hidden]`` embedding rows, or None.
    """
    packet = None if hook is None else hook.embedding_packet
    if packet is None:
        return None
    hook.embedding_packet = None
    md, ids, weight, rows = packet
    if (
        md is not attn_metadata
        or ids is not input_ids
        or getattr(embed_tokens, "weight", None) is not weight
    ):
        from ... import dispatch

        dispatch.log_capability("decode_prologue", "take_embedding", False, "packet does not match")
        return None
    return rows


class DecodePrologueHook:
    """The decode-metadata prologue and the target embedding in one launch (``DecodePrologue``).

    The engine calls it first in the step's input preprocessing, inside CUDA graph capture
    as well (``target_prologue``); the decode target then takes the rows that launch
    embedded (``take_embedding``). The rows of the last launch wait here, with the
    attention metadata, ids and weight they were embedded for, until the forward
    takes them.
    """

    def __init__(self) -> None:
        self.embedding_packet: Optional[tuple] = None

    def __call__(
        self, preparer, inputs, *, enable_spec_decode: bool, runtime_draft_len: int
    ) -> bool:
        return target_prologue(
            preparer,
            inputs,
            enable_spec_decode=enable_spec_decode,
            runtime_draft_len=runtime_draft_len,
            hook=self,
        )

    def take_embedding(
        self, attn_metadata, input_ids: torch.Tensor, embed_tokens
    ) -> Optional[torch.Tensor]:
        return take_embedding(attn_metadata, input_ids, embed_tokens, hook=self)
