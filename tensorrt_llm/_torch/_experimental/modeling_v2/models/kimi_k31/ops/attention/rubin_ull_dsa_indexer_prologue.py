# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The K3.1 decode indexer prologue in one CuTe DSL launch (``kernels/attention/rubin_ull_dsa_index_prologue.py``
``IndexerPrologue``): FP8 queries, scaled head weights and the keys into the indexer K cache."""

from __future__ import annotations

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from ._rubin_ull_indexer_math import DIM, HEADS, key_layer_norm, quantize_queries


def _cell(why: str, rows: int, n_write: int | None = None) -> Cell:
    return Cell(why=why, spec=dict(rows=rows, n_write=rows if n_write is None else n_write))


class DsaIndexerPrologue(OpWrapper):
    """For each row ``t`` of BF16 queries ``q`` [M, 32, 128], head weights ``w`` [M, 32] and keys
    ``k`` [M, 128]: ``q_out[t] , w_out[t] = quantize_queries(q[t], w[t], weight_scale)`` (per head
    a UE8M0 scale of ``amax / 448``, E4M3 codes, ``(w * scale) * weight_scale`` in FP32), and for
    ``t < n_write`` with both slots non-negative the key ``LayerNorm(k[t]; ln_w, ln_b, eps)``
    rounded to BF16 and quantized the same way, its codes at byte ``slot_fp8[t]`` and its FP32
    scale at byte ``slot_scale[t]`` of ``cache``. Writes ``q_out`` and ``w_out``; returns None.

    ``runner``: ``kernels.attention.rubin_ull_dsa_index.indexer_prologue(device)`` (the compiled kernel of the device).
    """

    ARCHS = frozenset({Arch.SM_103})

    CELLS: tuple[Cell, ...] = (
        _cell("served b4: 4 requests x 8 rows (32 rows; rows 17-32 run the two launches)", 32),
        _cell("served b3: 24 rows", 24),
        _cell("17 rows: the first count past the fused front", 17),
        _cell("padded graph rows: 32 rows, the first 20 keys written", 32, n_write=20),
        _cell("no cache write", 8, n_write=0),
    )

    note = """
    The K3.1 ULL generation target's indexer prologue where ``dsa_index_front`` does not serve the
    rows (17-32: the query rows come from cuBLAS on the indexer stream first). One CTA of 8 warps per
    token; the key's LayerNorm is FP32 with ``rsqrt.approx`` and a BF16 rounding before the FP8
    one. Waits for the earlier grids before its first load, then triggers its dependents.
    """

    def __call__(
        self,
        runner,
        k: torch.Tensor,
        q: torch.Tensor,
        w: torch.Tensor,
        ln_w: torch.Tensor,
        ln_b: torch.Tensor,
        eps: float,
        weight_scale: float,
        q_out: torch.Tensor,
        w_out: torch.Tensor,
        cache: torch.Tensor | None,
        slot_fp8: torch.Tensor | None,
        slot_scale: torch.Tensor | None,
        n_write: int,
    ) -> None:
        runner(k, q, w, ln_w, ln_b, eps, weight_scale, q_out, w_out, cache, slot_fp8, slot_scale,
               n_write)  # fmt: skip

    def reference(
        self,
        runner,
        k: torch.Tensor,
        q: torch.Tensor,
        w: torch.Tensor,
        ln_w: torch.Tensor,
        ln_b: torch.Tensor,
        eps: float,
        weight_scale: float,
        q_out: torch.Tensor,
        w_out: torch.Tensor,
        cache: torch.Tensor | None,
        slot_fp8: torch.Tensor | None,
        slot_scale: torch.Tensor | None,
        n_write: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """``(q_fp8, weights, keys)``: the exact query quantization and the FP64 LayerNorm of the
        first ``n_write`` keys. The outputs, the cache and its slots are not read."""
        q_fp8, weights = quantize_queries(q.reshape(-1, HEADS * DIM), w, weight_scale)
        return q_fp8, weights, key_layer_norm(k[:n_write], ln_w, ln_b, eps)

    def is_valid(
        self,
        runner,
        k: torch.Tensor,
        q: torch.Tensor,
        w: torch.Tensor,
        ln_w: torch.Tensor,
        ln_b: torch.Tensor,
        eps: float,
        weight_scale: float,
        q_out: torch.Tensor,
        w_out: torch.Tensor,
        cache: torch.Tensor | None,
        slot_fp8: torch.Tensor | None,
        slot_scale: torch.Tensor | None,
        n_write: int,
    ) -> None:
        """The runner asserts dtypes and layouts; it does not check the row counts of the outputs,
        the number of slots or their range."""
        rows = k.shape[0]
        if q.shape[0] != rows or w.shape[0] != rows:
            raise ValueError("dsa_indexer_prologue: k, q and w must have the same rows")
        if q_out.shape != (rows, HEADS, DIM) or w_out.shape != (rows, HEADS):
            raise ValueError("dsa_indexer_prologue: q_out / w_out must hold every row")
        if cache is not None and n_write > 0:
            if n_write > rows:
                raise ValueError("dsa_indexer_prologue: n_write past the step's rows")
            bytes_ = cache.numel() * cache.element_size()
            for slots, width in ((slot_fp8, DIM), (slot_scale, 4)):
                if slots.shape[0] < n_write or int(slots[:n_write].max()) + width > bytes_:
                    raise ValueError("dsa_indexer_prologue: a slot past the cache")


dsa_indexer_prologue = DsaIndexerPrologue()
