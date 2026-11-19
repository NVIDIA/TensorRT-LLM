# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DSA indexer prefill Top-K: the self-sampling GVR engine (CuTe DSL), deterministic order.

The K3.1 indexer's prefill Top-K on Rubin wherever a chunk's logits are wider than the Top-K: the
engine ``TopK`` module's self-sampling prefill engine with ``gvr_deterministic_topk``. Chunks no
wider than the Top-K take the radix ``indexer_topk_prefill`` (the engine module's own choice).
"""

from __future__ import annotations

import torch

from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper


def _spec(windows, width, **overrides) -> dict:
    """``windows``: (start, end) column window of each row, ``width`` the logits' columns."""
    spec = dict(windows=tuple(windows), width=width, top_k=2048)
    spec.update(overrides)
    return spec


class RubinGvrTopkPrefill(OpWrapper):
    """``indices[r]`` = the ``k`` (``indices.shape[1]``) largest of ``logits[r, ks:ke]`` as local
    columns (column - ``ks``), ``-1`` padded; a window no wider than ``k`` gets ``0 .. ke - ks - 1``
    then ``-1``. Ties go to the smaller column.

    ``logits`` FP32 ``[M, N]`` row-major (row stride a multiple of 4, 16-byte aligned base: the
    DeepGEMM prefill logits); ``row_starts`` / ``row_ends`` int32 ``[M]`` contiguous with
    ``0 <= ks <= ke <= N``; ``indices`` int32 ``[M, k]``, written in place and returned.
    """

    ARCHS = frozenset({Arch.SM_107})
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="one 4k prompt: causal windows, the first 2048 rows on the short path",
            spec=_spec([(0, e) for e in range(1, 4097)], 4096),
        ),
        Cell(
            why="a chunk after an 8k cached prefix: every window wider than the Top-K",
            spec=_spec([(0, 8192 + e) for e in range(1, 1025)], 9216),
        ),
        Cell(
            why="two packed requests: windows start at each request's first key",
            spec=_spec(
                [(0, e) for e in range(1, 3001)] + [(3000, 3000 + e) for e in range(1, 2501)],
                5500,
            ),
        ),
        Cell(
            why="ties: integer logits, so the smaller-index rule decides the set",
            spec=_spec([(0, 6000 + e) for e in range(1, 257)], 6256, data="ties"),
        ),
    )
    note = """
    The engine's ``selfsampling_topk_run_prefill(..., deterministic=True)``: no host read, it
    never compiles under capture (the targets' prefill is eager). The order of a row's selected
    indices is the kernel's, so the test compares sets.
    """

    def __call__(
        self,
        logits: torch.Tensor,
        row_starts: torch.Tensor,
        row_ends: torch.Tensor,
        indices: torch.Tensor,
    ) -> torch.Tensor:
        from tensorrt_llm._torch.kernels.blackwell.top_k import selfsampling_topk_run_prefill

        selfsampling_topk_run_prefill(
            logits, row_starts, row_ends, indices, deterministic=True, sorting=None
        )
        return indices

    def reference(
        self,
        logits: torch.Tensor,
        row_starts: torch.Tensor,
        row_ends: torch.Tensor,
        indices: torch.Tensor,
    ) -> torch.Tensor:
        """A new ``[M, k]`` tensor: each row's selected local columns in ascending order, then -1.
        ``indices`` is left untouched."""
        rows, k = indices.shape
        out = torch.full((rows, k), -1, dtype=torch.int32, device=logits.device)
        for r in range(rows):
            ks, ke = int(row_starts[r]), int(row_ends[r])
            n = ke - ks
            if n <= 0:
                continue
            if n <= k:
                out[r, :n] = torch.arange(n, dtype=torch.int32, device=logits.device)
                continue
            order = torch.sort(-logits[r, ks:ke].double(), stable=True).indices[:k]
            out[r] = torch.sort(order).values.to(torch.int32)
        return out

    def is_valid(
        self,
        logits: torch.Tensor,
        row_starts: torch.Tensor,
        row_ends: torch.Tensor,
        indices: torch.Tensor,
    ) -> None:
        # The engine trusts the windows: one past the logits' width reads the next row.
        assert int(row_starts.min()) >= 0 and int(row_ends.max()) <= logits.shape[1], (
            "every window must lie inside the logits row"
        )
        assert bool((row_starts <= row_ends).all()), "a window ends before it starts"
        assert logits.stride(1) == 1 and logits.stride(0) % 4 == 0 and logits.data_ptr() % 16 == 0


rubin_gvr_topk_prefill = RubinGvrTopkPrefill()
