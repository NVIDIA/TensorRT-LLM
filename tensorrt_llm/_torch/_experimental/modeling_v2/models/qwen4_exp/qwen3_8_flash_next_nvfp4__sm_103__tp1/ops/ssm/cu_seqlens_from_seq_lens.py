# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepend-zero cumulative sum of a per-request query-row count: the
``cu_seqlens`` segmentation vector every varlen recurrent op takes.

Tier-3 composed from native torch (``torch.zeros`` + ``torch.cumsum``, three
launches). No registered op or ``modules/`` callable produces this vector: the
recurrent metadata's ``query_start_loc*`` is built inline by
``Mamba2Metadata.prepare`` and segments the generation tail one row per
sequence, which is short under speculative decoding.
"""

from __future__ import annotations

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

#: Integer dtypes accepted on either side. ``torch.cumsum`` silently accepts
#: bool and floating dtypes and answers wrongly with them.
_INTEGER_DTYPES = (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)

#: Batch shapes this checkpoint's serving path produces where it calls this entry.
#: ``p`` is the trunk's query rows per generation request, ``1 + runtime_draft_len``
#: under MTP (2..8: `max_draft_len` 1..7); at ``p = 1`` the target reuses the
#: recurrent metadata's vector instead.
BATCHES = {
    "mixed_p2": (1024, 37, 2, 2, 2, 2),
    "padded_decode_p2": (2, 2, 2, 0, 0, 0),
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
    batch padded with zero-length rows, int32 in and int64 out. CUDA-graph
    capture is certified (only `seq_lens.shape[0]` reaches Python; a captured
    graph follows lengths rewritten in place). Three launches, about 0.03 ms
    at batch 8..256 -- launch bound.

    Not certified: a `seq_lens` dtype other than int32 (the guard admits any
    integer dtype), a non-contiguous `seq_lens` view.

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
