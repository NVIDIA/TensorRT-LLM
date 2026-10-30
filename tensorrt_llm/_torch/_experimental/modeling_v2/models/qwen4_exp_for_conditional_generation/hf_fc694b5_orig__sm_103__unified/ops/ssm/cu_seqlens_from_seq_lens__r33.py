# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepend-zero cumulative sum of a per-request query-row count: the
``cu_seqlens`` segmentation vector every varlen recurrent op takes.

A macro entry composed from native torch (``torch.zeros`` + ``torch.cumsum``).
No registered op or ``modules/`` callable produces this vector in one launch:
the recurrent metadata's ``query_start_loc*`` is built inline by
``Mamba2Metadata.prepare`` and segments the generation tail one row per
sequence, which is short under speculative decoding; the tree's other builders
of the same vector (``speculative/ragged_helpers.build_qo_indptr``, and
FlashInfer's private ``utils.get_indptr``) are this same composition.
"""

from __future__ import annotations

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

#: Integer dtypes accepted on either side. ``torch.cumsum`` silently accepts
#: bool and floating dtypes and answers wrongly with them.
_INTEGER_DTYPES = (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)

#: Batch shapes this checkpoint's serving path produces where it calls this entry.
#: ``p`` is the trunk's query rows per generation request, ``1 + runtime_draft_len``
#: under MTP (2..8: `max_draft_len` 1..7); at ``p = 1`` the tp1 target reuses the
#: recurrent metadata's vector instead, and ``decode_p1_128`` certifies the call
#: there all the same.
#: The ``_128`` batches are the serving envelope's widest step at every member of
#: the unified target (``max_batch_size`` 128 requests per rank, ``max_num_tokens``
#: 8192 rows): ``mixed_p8_128`` is one context chunk beside 127 MTP-7 verifies,
#: 7176 + 127 * 8 = 8192 rows.
BATCHES = {
    "mixed_p2": (1024, 37, 2, 2, 2, 2),
    "padded_decode_p2": (2, 2, 2, 0, 0, 0),
    "decode_p1_128": (1,) * 128,
    "verify_p8_128": (8,) * 128,
    "chunk_8192": (8192,),
    "mixed_p8_128": (7176,) + (8,) * 127,
}


class CuSeqlensFromSeqLens(OpWrapper):
    """`out[0] = 0`, `out[s] = seq_lens[0] + ... + seq_lens[s - 1]`: a fresh
    contiguous `[num_seqs + 1]` tensor of `dtype` on `seq_lens.device`.

    The scan accumulates in `dtype`, not in `seq_lens`'s dtype. Fusion boundary:
    the leading zero, the scan and the dtype. Choosing which length vector to
    scan, and keeping the packed activation exactly `out[-1]` rows long, are the
    caller's. `seq_lens` is read-only.
    """

    ARCHS = frozenset({Arch.SM_103})

    # spec: lengths, in_dtype, out_dtype; `layout` "contiguous" (default) or
    # "prefix" (the first len(lengths) of a longer buffer -- the call site's
    # `seq_lens_cuda[:batch]`).
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="call site: prefix of seq_lens_cuda, int32->int64",
            spec=dict(
                lengths=BATCHES["mixed_p2"],
                in_dtype=torch.int32,
                out_dtype=torch.int64,
                layout="prefix",
            ),
        ),
        Cell(
            why="MTP verify p=8, 32 requests: prefix int32->int64",
            spec=dict(
                lengths=(8,) * 32, in_dtype=torch.int32, out_dtype=torch.int64, layout="prefix"
            ),
        ),
        Cell(
            why="padded_decode_p2: zero-length padded rows, int32->int64",
            spec=dict(
                lengths=BATCHES["padded_decode_p2"], in_dtype=torch.int32, out_dtype=torch.int64
            ),
        ),
        Cell(
            why="decode at con 128: 128 one-row requests, prefix int32->int64",
            spec=dict(
                lengths=BATCHES["decode_p1_128"],
                in_dtype=torch.int32,
                out_dtype=torch.int64,
                layout="prefix",
            ),
        ),
        Cell(
            why="MTP-7 verify at con 128: 128 x 8 rows, prefix int32->int64",
            spec=dict(
                lengths=BATCHES["verify_p8_128"],
                in_dtype=torch.int32,
                out_dtype=torch.int64,
                layout="prefix",
            ),
        ),
        Cell(
            why="one 8192-row prefill chunk (max_num_tokens), prefix int32->int64",
            spec=dict(
                lengths=BATCHES["chunk_8192"],
                in_dtype=torch.int32,
                out_dtype=torch.int64,
                layout="prefix",
            ),
        ),
        Cell(
            why="widest step: 7176-row chunk + 127 MTP-7 verifies = 8192 rows, 128 requests",
            spec=dict(
                lengths=BATCHES["mixed_p8_128"],
                in_dtype=torch.int32,
                out_dtype=torch.int64,
                layout="prefix",
            ),
        ),
    )

    note = """
    Metadata consumed: none. The call site reads `attn_metadata.seq_lens_cuda`
    prefix, which still holds the true per-request query-row count during the
    trunk's forward (the spec worker rewrites it only afterwards). Not the
    recurrent metadata's `query_start_loc_long`: that one has one row per
    generation sequence and is short under MTP; the two agree exactly when
    every generation request contributes one row.

    Preconditions, none checked (data, not metadata -- a check would sync):
    entries non-negative (a negative length was silently wrong in three
    consumer probes and a device-side assert in the fourth); a padded row is
    length 0, never -1; `out[-1]` equals the packed activation's row count.

    `dtype` has no default: `causal_conv1d_fwd` requires int32 and
    `chunk_gated_delta_rule` keys its autotuner on `cu_seqlens`' dtype.

    Certified bitwise at the call site's prefix view of `seq_lens_cuda` (a
    mixed step at p 2, and the p 8 verify of 32 requests) and at a decode
    batch padded with zero-length rows, int32 in and int64 out. Also at the
    serving envelope of every unified member (a rank's batch is at most
    `max_batch_size` 128 requests and `max_num_tokens` 8192 rows at tp4, tep4
    and dep4 alike; the vector does not depend on the parallel layout): 128
    one-row requests, 128 MTP-7 verifies of 8 rows, one 8192-row chunk, and a
    7176-row chunk beside 127 verifies (128 requests, 8192 rows). CUDA-graph
    capture is certified at 32 requests (int32 out) and 128 requests (int64
    out): only `seq_lens.shape[0]` reaches Python, and a captured graph follows
    lengths rewritten in place, up to a mixed step of 8192 rows.

    Composition, measured on GB300 at the 128-request and 8192-row cells
    (int32 in, int64 out): four kernels -- a fill, the int32 -> int64 cast copy,
    and cub's scan-init and scan, all compiled into torch (no JIT path). Eager
    35-41 us a call (host launch bound); 8-10 us a CUDA-graph replay.
    Searched for one kernel that builds this vector: `torch.ops.trtllm` (no
    scan / cu_seqlens op), the catalog and the in-tree targets' ops, the
    engine's modules (`speculative/ragged_helpers.build_qo_indptr` is this same
    composition at int32) and FlashInfer (`utils.get_indptr`, private, the same
    composition). None exists.

    Not certified: a `seq_lens` dtype other than int32 (the guard admits any
    integer dtype), a non-contiguous `seq_lens` view, more than 128 requests,
    and a step over 8192 rows (a larger `max_num_tokens`, as the tune may set;
    an int64 `out` holds any reachable total).

    Hard limits
      out[-1]  accumulated in `dtype`  boundary 2^31 - 1 for int32
        probed at [2e9, 2e9]: int64 gives [0, 2e9, 4e9]; int32 wraps silently
        to [0, 2000000000, -294967296]. Not guarded (data).
    """

    def __call__(self, seq_lens: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        out = torch.zeros(seq_lens.shape[0] + 1, dtype=dtype, device=seq_lens.device)
        torch.cumsum(seq_lens, dim=0, dtype=dtype, out=out[1:])
        return out

    def reference(self, seq_lens: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        """A scalar Python prefix sum; no torch op computes it."""
        total, out = 0, [0]
        for length in seq_lens.tolist():
            total += int(length)
            out.append(total)
        return torch.tensor(out, dtype=dtype, device=seq_lens.device)

    def is_valid(self, seq_lens: torch.Tensor, dtype: torch.dtype) -> None:
        # Measured: a [1, n] row vector returned a length-2 [0, n_0]; an [n, 2]
        # input a length-(n + 1) vector of nonsense (only a resize warning); a
        # float32 [7.9, 1.2, 13.6] was truncated to [0, 7, 8, 21]; a bool input
        # was counted as nonzero rows; a float `dtype` returned a float vector.
        assert seq_lens.dim() == 1, (
            f"seq_lens must be a 1-D per-request row count, got shape {tuple(seq_lens.shape)}"
        )
        assert seq_lens.dtype in _INTEGER_DTYPES, (
            f"seq_lens must have an integer dtype, got {seq_lens.dtype}"
        )
        assert dtype in _INTEGER_DTYPES, f"dtype must be an integer dtype, got {dtype}"


cu_seqlens_from_seq_lens = CuSeqlensFromSeqLens()
