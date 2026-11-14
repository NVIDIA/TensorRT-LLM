# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A captured verify step's input prologue in one launch: the overlap scheduler's accepted-count
correction of the mRoPE position ids and the KV lengths, and the QSA row mapping over the
corrected lengths.

It replaces what `DecoderInputPreparer.preprocess_inputs` does on a generation-only verify step
(the position and KV-length corrections and `on_update_kv_lens()`'s row mapping);
`SpecStepMetadataPrologue` is the engine hook that calls it.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

#: Largest batch the single program covers (requests, rows).
MAX_SEQS = 64
MAX_ROWS = 512
#: Requests a step of the row-mapping loop covers once the `[rows, requests]` comparison matrix
#: would pass CHUNK_MIN_ELEMS elements (a single larger int64 matrix spills). Same output bits.
SEQ_CHUNK = 16
CHUNK_MIN_ELEMS = 8192


@triton.jit
def _spec_step_metadata_kernel(
    positions,
    stride_coord,
    pos_offsets,
    num_tokens,
    kv_lens,
    kv_offsets,
    seq_lens,
    request_indices,
    logical_positions,
    sequence_lengths,
    visible_blocks,
    num_seqs,
    pos_out,
    COMPRESS_RATIO: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_SEQS: tl.constexpr,
    HAS_POS_OUT: tl.constexpr,
    SEQ_CHUNK_: tl.constexpr,
):
    rows = tl.arange(0, BLOCK_ROWS)
    valid = rows < num_tokens
    seqs = tl.arange(0, BLOCK_SEQS)
    sv = seqs < num_seqs
    # Every load before any store (the stores overwrite positions and lengths it reads).
    off = tl.load(pos_offsets + rows, mask=valid, other=0)
    kv_new = tl.load(kv_lens + seqs, mask=sv, other=0) + tl.load(
        kv_offsets + seqs, mask=sv, other=0
    )
    # A row's request is the number of requests ending at or before it, its start the sum of
    # their lengths (`_qsa_generation_token_mapping_kernel`'s arithmetic).
    request = tl.zeros([BLOCK_ROWS], tl.int32)
    request_start = tl.zeros([BLOCK_ROWS], tl.int64)
    carry = tl.zeros([1], tl.int64)
    rows64 = rows.to(tl.int64)
    for s0 in tl.static_range(0, BLOCK_SEQS, SEQ_CHUNK_):
        cseq = s0 + tl.arange(0, SEQ_CHUNK_)
        csv = cseq < num_seqs
        lengths = tl.load(seq_lens + cseq, mask=csv, other=0).to(tl.int64)
        ends = tl.cumsum(lengths, axis=0) + carry
        ended = (ends[None, :] <= rows64[:, None]) & csv[None, :]
        request += tl.sum(ended.to(tl.int32), axis=1)
        request_start += tl.sum(tl.where(ended, lengths[None, :], 0), axis=1)
        carry += tl.sum(lengths, axis=0)
    kv_len = tl.load(kv_lens + request, mask=valid, other=0) + tl.load(
        kv_offsets + request, mask=valid, other=0
    )
    seq_len = tl.load(seq_lens + request, mask=valid, other=0)
    p0 = tl.load(positions + rows, mask=valid, other=0)
    p1 = tl.load(positions + stride_coord + rows, mask=valid, other=0)
    p2 = tl.load(positions + 2 * stride_coord + rows, mask=valid, other=0)
    tl.debug_barrier()
    tl.store(positions + rows, p0 + off, mask=valid)
    tl.store(positions + stride_coord + rows, p1 + off, mask=valid)
    tl.store(positions + 2 * stride_coord + rows, p2 + off, mask=valid)
    if HAS_POS_OUT:
        tl.store(pos_out + rows, p0 + off, mask=valid)
        tl.store(pos_out + num_tokens + rows, p1 + off, mask=valid)
        tl.store(pos_out + 2 * num_tokens + rows, p2 + off, mask=valid)
    tl.store(kv_lens + seqs, kv_new, mask=sv)
    logical = kv_len.to(tl.int64) - seq_len.to(tl.int64) + rows.to(tl.int64) - request_start
    visible_num = logical + 1
    visible = visible_num // COMPRESS_RATIO
    visible = tl.where(
        (visible_num % COMPRESS_RATIO != 0) & (visible_num < 0), visible - 1, visible
    )
    tl.store(request_indices + rows, request, mask=valid)
    tl.store(logical_positions + rows, logical, mask=valid)
    tl.store(sequence_lengths + rows, kv_len, mask=valid)
    tl.store(visible_blocks + rows, visible.to(tl.int32), mask=valid)


def _cells():
    return tuple(
        Cell(why=f"{n} request(s) x {p} verify rows", spec=dict(n=n, p=p, capture=capture))
        for n, p, capture in (
            (1, 3, True),
            (8, 4, False),
            (40, 3, True),
            (56, 3, True),
            (64, 3, False),
        )
    )


class QsaSpecStepMetadata(OpWrapper):
    """`qsa_spec_step_metadata(positions, pos_offsets, kv_lens, kv_offsets, seq_lens, mapping,
    compress_ratio)`, in place:

    - `positions` `[C, 1, T]` int32 (the step's mRoPE positions, unit token stride, any
      coordinate stride): `positions[:, :, :T] += pos_offsets[:T]`;
    - `kv_lens` `[N]` int32: `kv_lens += kv_offsets[:N]`;
    - `mapping = (request_indices, logical_positions, sequence_lengths, visible_blocks)`
      (int32, int64, int32, int32, at least `T` long): every row's request, logical position,
      KV length and visible-block count over the corrected lengths, with each request's
      `seq_lens[i]` rows consecutive -- bitwise the composed corrections followed by
      `triton_qsa_generation_token_mapping`.
    """

    ARCHS = frozenset({Arch.SM_103})
    CELLS: tuple[Cell, ...] = _cells()

    note = """
    Source: one Triton program (`_spec_step_metadata_kernel`), up to 64 requests and 512 rows;
    integer arithmetic only.
    """

    def __call__(
        self,
        positions,
        pos_offsets,
        kv_lens,
        kv_offsets,
        seq_lens,
        mapping,
        compress_ratio: int,
        pos_out=None,
    ):
        num_tokens = positions.shape[-1]
        num_seqs = kv_lens.shape[0]
        block_rows = max(16, triton.next_power_of_2(num_tokens))
        block_seqs = max(16, triton.next_power_of_2(num_seqs))
        chunked = block_rows * block_seqs > CHUNK_MIN_ELEMS
        _spec_step_metadata_kernel[(1,)](
            positions,
            positions.stride(0),
            pos_offsets,
            num_tokens,
            kv_lens,
            kv_offsets,
            seq_lens,
            *mapping,
            num_seqs,
            pos_out if pos_out is not None else positions,
            COMPRESS_RATIO=compress_ratio,
            BLOCK_ROWS=block_rows,
            BLOCK_SEQS=block_seqs,
            HAS_POS_OUT=pos_out is not None,
            SEQ_CHUNK_=SEQ_CHUNK if chunked else block_seqs,
            num_warps=8 if chunked else 4,
        )

    def reference(
        self,
        positions,
        pos_offsets,
        kv_lens,
        kv_offsets,
        seq_lens,
        mapping,
        compress_ratio: int,
        pos_out=None,
    ):
        """The composition, in place: the two corrections, then the row mapping in torch."""
        num_tokens = positions.shape[-1]
        num_seqs = kv_lens.shape[0]
        positions += pos_offsets[:num_tokens]
        if pos_out is not None:
            pos_out.copy_(positions.reshape(positions.shape[0], num_tokens))
        kv_lens += kv_offsets[:num_seqs]
        request_indices, logical_positions, sequence_lengths, visible_blocks = mapping
        lens = seq_lens[:num_seqs].long()
        req = torch.repeat_interleave(torch.arange(num_seqs, device=kv_lens.device), lens)[
            :num_tokens
        ]
        starts = torch.cumsum(lens, 0) - lens
        rows = torch.arange(num_tokens, device=kv_lens.device)
        kv = kv_lens[req].long()
        logical = kv - lens[req] + rows - starts[req]
        request_indices[:num_tokens] = req.to(request_indices.dtype)
        logical_positions[:num_tokens] = logical.to(logical_positions.dtype)
        sequence_lengths[:num_tokens] = kv.to(sequence_lengths.dtype)
        visible_blocks[:num_tokens] = torch.div(
            logical + 1, compress_ratio, rounding_mode="floor"
        ).to(visible_blocks.dtype)

    def is_valid(
        self,
        positions,
        pos_offsets,
        kv_lens,
        kv_offsets,
        seq_lens,
        mapping,
        compress_ratio: int,
        pos_out=None,
    ) -> None:
        num_tokens = positions.shape[-1]
        num_seqs = kv_lens.shape[0]
        assert (
            positions.dim() == 3 and positions.shape[:2] == (3, 1) and positions.stride(2) == 1
        ), (
            tuple(positions.shape),
            positions.stride(),
        )
        assert positions.dtype == pos_offsets.dtype == torch.int32
        assert kv_lens.dtype == kv_offsets.dtype == seq_lens.dtype == torch.int32
        assert kv_lens.dim() == 1 and kv_lens.is_contiguous()
        assert 0 < num_seqs <= MAX_SEQS and 0 < num_tokens <= MAX_ROWS, (num_seqs, num_tokens)
        assert pos_offsets.numel() >= num_tokens and kv_offsets.numel() >= num_seqs
        assert seq_lens.numel() >= num_seqs
        dtypes = (torch.int32, torch.int64, torch.int32, torch.int32)
        for t, dt in zip(mapping, dtypes):
            assert t.dtype == dt and t.numel() >= num_tokens and t.is_contiguous(), (
                t.dtype,
                t.numel(),
            )


qsa_spec_step_metadata = QsaSpecStepMetadata()


class SpecStepMetadataPrologue:
    """The engine's decode-prologue hook (`RuntimeHooks.decode_prologue`): on a generation-only
    verify step with the overlap scheduler, runs `qsa_spec_step_metadata` plus
    `on_update_kv_lens()`'s host statements and returns True; anything else returns False and
    keeps the native code."""

    def __call__(
        self, preparer, inputs, *, enable_spec_decode: bool, runtime_draft_len: int
    ) -> bool:
        from tensorrt_llm._torch.attention.backends.sparse.qsa.metadata import QSAAttentionMetadata

        md = inputs.get("attn_metadata")
        if md is not None:
            # A packet of an earlier step never reaches this step's forward.
            md.spec_step_positions = None
        if not enable_spec_decode or preparer._config.disable_overlap_scheduler:
            return False
        if not isinstance(md, QSAAttentionMetadata) or md.kv_cache_manager is None:
            return False
        if not md.qsa_has_local_layers or preparer.mapping.has_cp_helix():
            return False
        num_seqs, num_tokens = md.num_seqs, md.num_tokens
        if md.num_contexts != 0 or md.num_chunked_ctx_requests != 0 or md.num_ctx_tokens != 0:
            return False
        # the native mapping's generation path (not its one-row-a-request decode path)
        if num_tokens == num_seqs or not (0 < num_seqs <= MAX_SEQS) or num_tokens > MAX_ROWS:
            return False
        pos = inputs["position_ids"]
        if (
            pos.ndim != 3
            or tuple(pos.shape[:2]) != (3, 1)
            or pos.stride(2) != 1
            or pos.dtype != torch.int32
        ):
            return False
        if inputs["input_ids"].shape[0] != num_tokens or pos.shape[2] != num_tokens:
            return False
        buffers = preparer._buffers
        pos_off, kv_off = (
            buffers.previous_pos_id_offsets_cuda,
            buffers.previous_kv_lens_offsets_cuda,
        )
        kv_lens = md.kv_lens_cuda[:num_seqs]
        # the mapping reads the runtime view: it must be these lengths
        runtime = getattr(md, "kv_lens_cuda_runtime", None)
        if runtime is None or runtime.data_ptr() != kv_lens.data_ptr():
            return False
        if (
            pos_off.dtype != torch.int32
            or kv_off.dtype != torch.int32
            or kv_lens.dtype != torch.int32
        ):
            return False
        seq_lens = md.seq_lens_cuda[:num_seqs]
        if seq_lens.dtype != torch.int32:
            return False
        mapping = (
            md.qsa_req_idx_per_token,
            md.qsa_logical_positions,
            md.qsa_sequence_lengths,
            md.qsa_visible_blocks,
        )
        if (mapping[0].dtype, mapping[1].dtype, mapping[2].dtype, mapping[3].dtype) != (
            torch.int32,
            torch.int64,
            torch.int32,
            torch.int32,
        ):
            return False
        # The corrected positions also as the contiguous `[C, T]` the decode forward reads
        # (`take_step_positions`).
        pos3 = torch.empty((pos.shape[0], num_tokens), dtype=torch.int32, device=pos.device)
        qsa_spec_step_metadata(
            pos, pos_off, kv_lens, kv_off, seq_lens, mapping, md.sparse_metadata_params.compress_ratio,
            pos_out=pos3,
        )  # fmt: skip
        md.spec_step_positions = (pos, pos3)
        # `on_update_kv_lens()`'s host statements (the QSA override adds only the mapping).
        super(QSAAttentionMetadata, md).on_update_kv_lens()
        return True


def take_step_positions(md, position_ids):
    """The contiguous int32 `[C, T]` positions `SpecStepMetadataPrologue` wrote for exactly this
    `position_ids` tensor in this step (consumed once), or None."""
    packet = getattr(md, "spec_step_positions", None)
    if packet is None:
        return None
    md.spec_step_positions = None
    return packet[1] if packet[0] is position_ids else None
