# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""QSA MTP IndexShare: draft-decode selections rebuilt from the row captured at draft-extend."""

from __future__ import annotations

import torch

# Tier-2 sourcing: a vendored Triton callable reached as ordinary Python. QSA's
# kernel plane has no torch.ops surface.
from tensorrt_llm._torch.attention.backends.sparse.qsa import kernels as _qsa_kernels

from .._op import Arch, Cell, OpWrapper

#: This checkpoint's selection width: `indexer_budget` 2048 tokens over
#: `indexer_compress_ratio` 4.
_BLOCK_TOPK = 512
#: `(slots, this layer's slot)` of the shared buffers: with the MTP layer in the
#: target's cache manager, 12 trunk full-attention layers plus the MTP layer.
#: This is dep4's layout (attention DP keeps the draft layer in the trunk's manager).
_COMBINED_SLOTS = (13, 12)
#: tp4 / tep4's layout: attention TP puts the draft layer in a separate draft KV
#: cache, so the metadata sizes the buffers by the larger manager (the trunk's 12
#: sparse layers) and the draft loop reads the draft manager's only slot, 0.
_SEPARATE_DRAFT_SLOTS = (12, 0)

#: The engines' `max_batch_size` at every member and variant, and so the requests
#: one draft step carries (one row each) and the shared buffers' height
#: (`max_num_sequences` = `max_batch_size * max_beam_width`, beam width 1).
_MAX_BATCH = 128
#: The longest sequence the unified target serves: the tune's perf configs run
#: `max_seq_len` 262144 (the MTP gates 16384).
_ENVELOPE_SEQ_LEN = 262144


def _spec(contexts, **overrides) -> dict:
    """A cell spec at this checkpoint's geometry and the engine's metadata dtypes.

    `contexts` says what each request's captured row and current row are:
      ("mixed", B)        B requests; the first 16 at fixed boundary positions
                          (empty, short, 511, 512 and full rows), plus one at the
                          last position `max_seq_len` allows, the rest seeded
      ("full", B)         captured positions >= 2047: all 512 slots valid
      ("explicit", P)     one request per captured position in P
    A request captured at position `p` (its last accepted row) holds
    `(p + 1) // 4` complete groups; its draft row `d` positions later sees
    `(p + d + 1) // 4`. `drafted` cycles over the requests. Positions stay below
    `max_seq_len - d`.
    """
    base = dict(
        contexts=contexts,
        drafted=(1, 2, 3),
        rows_per_request=1,
        shared_rows=64,
        slots=_COMBINED_SLOTS,
        block_topk=_BLOCK_TOPK,
        metadata_dtype=torch.int32,
        graph_batches=(),
        max_seq_len=8192,
    )
    return {**base, **overrides}


def _unified(drafted, shared_rows) -> dict:
    """The unified target's draft step: 128 requests, tp4 / tep4's separate-draft
    slots, positions over the 262144 envelope, captured at 1 and 128 requests."""
    return _spec(
        ("mixed", _MAX_BATCH),
        drafted=drafted,
        shared_rows=shared_rows,
        slots=_SEPARATE_DRAFT_SLOTS,
        graph_batches=(1, _MAX_BATCH),
        max_seq_len=_ENVELOPE_SEQ_LEN,
    )


class QsaMtpReuseTopk(OpWrapper):
    """Rebuild each query row's group selection from its request's captured row,
    with the groups that completed since the capture written into it.

    Wraps `tensorrt_llm._torch.attention.backends.sparse.qsa.kernels.
    triton_qsa_mtp_reuse_topk`: one Triton launch, one program per output row.
    For row `r`, with `q = request_indices[r]` and `K = output.shape[1]`
    (`block_topk`):

      1. start from the captured row `shared_indices[q]`: valid group ids as a
         prefix, then `-1`;
      2. `captured = shared_visible_blocks[q]`, `current = visible_blocks[r]`,
         `fresh = clamp(current - captured, 0, K)`: the groups
         `captured .. captured + fresh - 1` completed after the capture;
      3. `filled` = the count of non-negative ids in the captured row; the
         fresh ids go, ascending, into slots `start .. start + fresh - 1` with
         `start = min(filled, K - fresh)` -- right after the prefix when there
         is room, otherwise after the captured row's newest `K - fresh` ids:
         the oldest `filled + fresh - K` ids are displaced and the survivors
         shift to the front;
      4. every other slot is the captured row's survivors in their order, `-1`
         padding included.

    Every slot of `output[:rows]` is written and nothing else; returns `output`.
    The inputs are read-only. Nothing is sorted, compacted or de-duplicated and
    the ranking is not recomputed: the captured row is taken as ascending (the
    target makes it canonical), so the displaced ids are the oldest groups, not the
    lowest-scored ones, and the result is ascending again.

    Fusion boundary: the copy-and-append only. Capturing each request's row
    into the shared buffers, the per-row request map and visible counts, and
    the expansion into token indices plus the incomplete causal tail
    (`qsa_expand_block_indices`) are the caller's.

    The unified target's variant of the qwen4_exp tp1 target's entry of the same name.
    """

    ARCHS = frozenset({Arch.SM_103})

    CELLS: tuple[Cell, ...] = (
        # ---- decode batch: one row per request; the first 16 requests sit at the
        # boundaries (empty, short, 511-id, 512-id and full rows), fresh 0 and 1.
        Cell(
            why="decode MTP: 32 requests (envelope), 256-row shared buffer",
            spec=_spec(("mixed", 32), shared_rows=256),
        ),
        # ---- the riskiest edge: two fresh groups displace a full row's two oldest
        Cell(
            why="8 requests at fresh 2: full, 511-id and short rows gain two groups",
            spec=_spec(
                ("explicit", (2050, 402, 2046, 8186, 6, 2047, 1022, 2049)),
                drafted=(5, 6, 5, 6, 6, 4, 5, 6),
                shared_rows=8,
            ),
        ),
        # ---- CUDA graph -----------------------------------------------------
        Cell(
            why="CUDA graph, drafted 1..6: captured at 1, 8 and 32 requests, new-data replay",
            spec=_spec(("mixed", 32), drafted=(1, 2, 3, 4, 5, 6), graph_batches=(1, 8, 32)),
        ),
        # ---- the unified target (tp4 / tep4 / dep4 alike: one selection row per
        # request, no head or rank axis): its 128-request draft steps.
        Cell(
            why="128 requests 3 positions past the capture, 512-row shared buffer (tep4 slots)",
            spec=_unified(drafted=(3,), shared_rows=512),
        ),
        Cell(
            why="128 requests 1 position past the capture, 256-row shared buffer (tep4 slots)",
            spec=_unified(drafted=(1,), shared_rows=256),
        ),
        Cell(
            why="128 requests 7 positions past the capture, 1024-row shared buffer (tep4 slots)",
            spec=_unified(drafted=(7,), shared_rows=1024),
        ),
        Cell(
            why="served mtp7 step: 128 requests filling the engine's 128-row buffer, drafted 1..6",
            spec=_unified(drafted=(1, 2, 3, 4, 5, 6), shared_rows=_MAX_BATCH),
        ),
    )

    note = """
    Metadata consumed: none; every input is an argument, all int32 at the call
    site:
      shared_indices         `qsa_shared_topk_indices[slot]`, [max_num_sequences, block_topk]
      shared_visible_blocks  `qsa_shared_topk_visible_blocks[slot]`, [max_num_sequences]
      request_indices        `qsa_req_idx_per_token[:rows]`
      visible_blocks         `qsa_visible_blocks[:rows]`
      output                 `qsa_topk_indices[:rows]`, the [max_num_tokens, block_topk] buffer
    `slot` is the layer's row in the cache manager's `qsa_shared_topk_slots`.

    It runs in the MTP draft loop's decode steps after the first: no context
    rows, one row per request, so rows = requests, and the request map is the
    row index. Draft step `i` sits `i` positions past the captured row, so
    `max_draft_len` 1 never reaches this op, 3 reaches 1 and 2 positions, and 7
    (the target's ceiling) reaches 1..6. With compress_ratio 4, `fresh =
    ((p + 1) % 4 + i) // 4` for a capture at position `p`: 0 or 1 up to i = 3
    and 0, 1 or 2 from i = 4. In a full row
    the fresh ids displace the row's oldest ids (one at fresh 1, two at fresh
    2), and a 511-id row at fresh 2 loses its oldest id.

    Preconditions that are values, so not `is_valid` guards:
      - each captured row is prefix-packed (every valid id before every -1), as
        `qsa_topk_select` emits it. The append slot comes from the COUNT of
        non-negative ids, so a -1 hole puts a fresh id over a valid one and
        loses a group silently;
      - captured ids are below that row's `shared_visible_blocks`, so a fresh id
        never duplicates a captured one;
      - every `request_indices[r]` is a row of both shared tensors;
      - `shared_visible_blocks[q]` is the visible count of the very row that was
        captured, and `visible_blocks[r]` is `(position + 1) // cr` of row `r`.

    Loud rejections (not pinned by a test): `ValueError` when the shared width differs from the output
    width, when `request_indices` or `visible_blocks` does not hold one element
    per output row, or on CPU tensors; a 1-D `output` or `shared_indices`; a
    width that is not a power of two (Triton compilation).

    Accepted and silently wrong, hence `is_valid`: any non-contiguous operand
    (the kernel addresses raw pointers densely and ignores strides); a 3-D
    `shared_indices` whose middle extent equals `block_topk` (read as slot 0);
    a 2-D `shared_visible_blocks` (the whole `[slots, seqs]` buffer is read as
    slot 0's counts). The test pins the last.

    Source: tier 2, an in-tree Triton kernel in the QSA attention backend,
    JIT-compiled on first call for the device it runs on; the cells were driven
    through that JIT path on sm_103 (GB300). One program per output row, no
    autotune, no PDL.

    The unified target (tp4 / tep4 / dep4) calls it with the same geometry on every
    rank: one selection row per request, no head or rank axis (the indexer is
    replicated). Under attention TP (tp4, tep4: the MTP unit's member) every rank
    runs the whole batch and the draft layer sits in a separate draft KV cache, so
    the buffers have the trunk's 12 slots and the draft loop reads slot 0
    (`_SEPARATE_DRAFT_SLOTS`); under attention DP the draft layer is a trunk layer,
    13 slots, slot 12 (`_COMBINED_SLOTS`). The buffers are `max_num_sequences` rows
    tall, `max_batch_size * max_beam_width` (`build_attention_metadata`): 128 at
    every member. The 256-, 512- and 1024-row cells are taller than any engine
    allocates and are certified as stated. A draft step carries at most 128
    requests, one row each.

    The `drafted` axis is the offset past the capture (above), not
    `max_draft_len`: mtp7's offsets 1..6 are the "served mtp7 step" cell, and the
    offset-7 cell is beyond any served draft step. Every cell holds a full row
    gaining the widest fresh count its offsets allow.

    Hard limits (read from the kernel, not probed): both row offsets are int64
    (`request.to(tl.int64) * BLOCK_TOPK`, `row.to(tl.int64) * BLOCK_TOPK`), so no
    buffer size the envelope reaches wraps them. Ids and visible counts are int32;
    positions below `max_seq_len` 262144 give at most 65536 groups. `block_topk`
    must be a power of two (`tl.arange`).

    Certified: int32 ids and metadata, `block_topk` 512, the identity request map,
    up to 128 requests in 8- to 1024-row shared buffers (128 requests filling a
    128-row buffer included), slots (13, 12) and (12, 0), captured positions up to
    the 262144 envelope, CUDA graph capture at 1, 8, 32 and 128 requests with a
    new-data replay.
    Not certified: a permuted request map, more than one row per request, int64
    metadata, 128 requests at dep4's (13, 12) slots (MTP is served at tep4).
    """

    def __call__(
        self,
        *,
        shared_indices: torch.Tensor,
        shared_visible_blocks: torch.Tensor,
        request_indices: torch.Tensor,
        visible_blocks: torch.Tensor,
        output: torch.Tensor,
    ) -> torch.Tensor:
        return _qsa_kernels.triton_qsa_mtp_reuse_topk(
            shared_indices=shared_indices,
            shared_visible_blocks=shared_visible_blocks,
            request_indices=request_indices,
            visible_blocks=visible_blocks,
            output=output,
            displace_oldest=True,
        )

    def reference(
        self,
        *,
        shared_indices: torch.Tensor,
        shared_visible_blocks: torch.Tensor,
        request_indices: torch.Tensor,
        visible_blocks: torch.Tensor,
        output: torch.Tensor,
    ) -> torch.Tensor:
        """The rebuilt rows as the contract states them, in int64 torch.

        `output` supplies only the shape and dtype; its contents are ignored.
        """
        rows, block_topk = output.shape
        device = output.device
        requests = request_indices.long().reshape(-1)
        captured = shared_visible_blocks.long()[requests]
        current = visible_blocks.long().reshape(-1)
        rebuilt = shared_indices.long()[requests].clone()

        filled = (rebuilt >= 0).sum(dim=1)
        fresh = (current - captured).clamp(min=0, max=block_topk)
        start = torch.minimum(filled, block_topk - fresh)
        slot = torch.arange(block_topk, device=device)[None, :]
        # The oldest `filled - start` ids go; the survivors move to the front.
        source = slot + (filled - start)[:, None]
        rebuilt = torch.where(
            source < block_topk,
            rebuilt.gather(1, source.clamp(max=block_topk - 1)),
            torch.full_like(rebuilt, -1),
        )
        rank = slot - start[:, None]
        appended = (rank >= 0) & (rank < fresh[:, None])
        rebuilt[appended] = (captured[:, None] + rank)[appended]
        return rebuilt.to(output.dtype)

    def is_valid(
        self,
        *,
        shared_indices: torch.Tensor,
        shared_visible_blocks: torch.Tensor,
        request_indices: torch.Tensor,
        visible_blocks: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        # The kernel takes raw pointers and addresses them densely; each case
        # below returns a plausible wrong answer rather than an error.
        if shared_indices.dim() != 2 or not shared_indices.is_contiguous():
            raise ValueError(
                "shared_indices must be one slot's contiguous [num_seqs, block_topk] "
                f"rows; got shape {tuple(shared_indices.shape)}, "
                f"stride {shared_indices.stride()}"
            )
        if shared_visible_blocks.dim() != 1 or not shared_visible_blocks.is_contiguous():
            raise ValueError(
                "shared_visible_blocks must be one slot's contiguous [num_seqs] counts; "
                f"got shape {tuple(shared_visible_blocks.shape)}, "
                f"stride {shared_visible_blocks.stride()}"
            )
        for name, tensor in (
            ("request_indices", request_indices),
            ("visible_blocks", visible_blocks),
            ("output", output),
        ):
            if not tensor.is_contiguous():
                raise ValueError(
                    f"{name} must be contiguous; got shape {tuple(tensor.shape)}, "
                    f"stride {tensor.stride()}"
                )


qsa_mtp_reuse_topk = QsaMtpReuseTopk()
