# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Inputs of the Kimi K3.1 DSA top-k and metadata entries' tests.

Top-k logits are a random permutation of distinct, exactly representable fp32 values per row, so
the top-k of a row is one set with no tie at the k-th value; the columns a kernel must not read
hold +inf, which any read would select.
"""

from __future__ import annotations

import torch

__extra_import_path__ = ["."]
# Shared with the indexer entries' tests.
from _k31_dsa_inputs import (  # noqa: E402,F401
    atom_ctx_offsets,
    chunk_windows,
    num_sms,
    on_sm103,
    seed_of,
)


def distinct_logits(rows: int, width: int, gen: torch.Generator, row_pad: int = 0) -> torch.Tensor:
    """fp32 [rows, width] CUDA view (row stride ``width + row_pad``): each row a permutation of
    ``(0 .. width - 1 - width / 2) / 4``."""
    keys = torch.rand(rows, width + row_pad, generator=gen).cuda()
    perm = keys[:, :width].argsort(dim=1).float()
    base = torch.empty(rows, width + row_pad, dtype=torch.float32, device="cuda")
    base[:, width:] = float("nan")
    base[:, :width] = (perm - width // 2) * 0.25
    return base[:, :width]


def poison_outside(logits: torch.Tensor, starts: torch.Tensor, ends: torch.Tensor) -> None:
    """+inf in every column outside ``[starts[r], ends[r])``, in place."""
    cols = torch.arange(logits.shape[1], device=logits.device).unsqueeze(0)
    outside = (cols < starts.long().unsqueeze(1)) | (cols >= ends.long().unsqueeze(1))
    logits.masked_fill_(outside, float("inf"))


def assert_same_sets(out: torch.Tensor, ref: torch.Tensor) -> None:
    """Each row of ``out`` holds the same multiset of indices as ``ref`` (order free)."""
    assert out.shape == ref.shape, (out.shape, ref.shape)
    a = out.sort(dim=1).values
    b = ref.sort(dim=1).values
    bad = (a != b).any(dim=1).nonzero().flatten()
    assert bad.numel() == 0, (
        f"{bad.numel()} of {out.shape[0]} rows select a different set; first row {int(bad[0])}: "
        f"{int((a[bad[0]] != b[bad[0]]).sum())} slots differ"
    )
