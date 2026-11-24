# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The DSA indexer front in one launch (``IndexFrontGemm``, the ULL target's ``modeling.py``): q_a
RMSNorm + the indexer query GEMM + the indexer prologue (FP8 queries, scaled head weights, the key
into the cache)."""

from __future__ import annotations

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import (
    Arch,
    Cell,
    OpWrapper,
    true_fp32_matmul,
)

from ._rubin_ull_indexer_math import DIM, HEADS, key_layer_norm, quantize_queries


def _cell(why: str, rows: int, n_write: int | None = None) -> Cell:
    return Cell(why=why, spec=dict(rows=rows, n_write=rows if n_write is None else n_write))


class DsaIndexFront(OpWrapper):
    """For each of the ``M`` rows of q_a ``qa`` [M, 1536] BF16:

    - ``q_rows[t] = rmsnorm(qa[t]; front.gamma, front.eps) @ front.weight^T`` (BF16 [4096], FP32
      accumulation), written into ``q_rows`` (allocated when None);
    - per head ``h``: ``scale = ue8m0(max |q_rows[t, h]| / 448)``, ``q_fp8[t, h] = e4m3(q / scale)``
      and ``index_w[t, h] = (iw[t, h] * scale) * weight_scale`` (FP32);
    - for ``t < n_write`` with both slots non-negative: the key ``LayerNorm(ik[t]; front.index_norm,
      front.index_bias, front.index_eps)`` rounded to BF16, quantized the same way, its 128 codes at
      byte ``slot_fp8[t]`` and its FP32 scale at byte ``slot_scale[t]`` of ``cache``.

    ``front``: the layer's ``IndexFront`` (the ULL target's ``modeling.py``; the compiled kernel of
    ``M`` rows, the weight's TMA view and map), prepared outside graph capture. ``ik`` [M, 128] and
    ``iw`` [M, 32] are BF16 column views of the DSA hidden projection rows. Returns
    ``(q_fp8 [M, 32, 128], index_w [M, 32])``, freshly allocated.
    """

    ARCHS = frozenset({Arch.SM_103})

    CELLS: tuple[Cell, ...] = (
        _cell("served b1: 1 request x next_n 8 rows, every key written", 8),
        _cell("served b2: 16 rows (the 16-token tile)", 16),
        _cell("q1 decode row", 1),
        _cell("padded graph rows: 8 rows, the first 3 keys written", 8, n_write=3),
        _cell("no cache write (n_write 0): queries and weights only", 4, n_write=0),
        _cell("9 rows: the 16-token tile, partly filled", 9),
        _cell("served b3 (Rubin): 24 rows over two 16-token tiles", 24),
        _cell("served b4 (Rubin): 32 rows, two full 16-token tiles", 32),
        _cell("25 rows: second tile partly filled, first 20 keys written", 25, n_write=20),
    )

    note = """
    The K3.1 ULL generation target's DSA indexer front at rows 1-16 (``dsa_front.TACTICS``) and, on
    Rubin, rows 1-32 (``RUBIN_TACTICS_FRONT``: rows 17-32 run the 16-row tactic over two 16-token
    tiles, grid y = ceil(rows / 16) x split; every token's work is guarded by ``token < rows``).
    Wider steps, and rows 17-32 elsewhere, run the query projection and ``dsa_indexer_prologue`` as
    two launches. The query GEMM is
    tcgen05 BF16 with the K split over a cluster; the epilogue quantizes the BF16 rows it stored, so
    the FP8 queries and head weights are a pure function of ``q_rows`` (``quantize_queries``). The
    key warp waits (``griddepcontrol.wait``) before its first read; the kernel triggers its
    dependents before its own wait (``splitk.py``, ``trigger_after_wait=False``). Launched with PDL.
    """

    def __call__(
        self,
        front,
        qa: torch.Tensor,
        ik: torch.Tensor,
        iw: torch.Tensor,
        weight_scale: float,
        cache: torch.Tensor | None = None,
        slot_fp8: torch.Tensor | None = None,
        slot_scale: torch.Tensor | None = None,
        n_write: int = 0,
        q_rows: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return front.run(qa, ik, iw, weight_scale, cache, slot_fp8, slot_scale, n_write, q_rows)

    def reference(
        self,
        front,
        qa: torch.Tensor,
        ik: torch.Tensor,
        iw: torch.Tensor,
        weight_scale: float,
        cache: torch.Tensor | None = None,
        slot_fp8: torch.Tensor | None = None,
        slot_scale: torch.Tensor | None = None,
        n_write: int = 0,
        q_rows: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """``(q_fp8, index_w, q_rows, keys)``: the query rows from the FP32 RMSNorm rounded to BF16
        and an FP32 GEMM rounded once, their exact quantization, and the FP64 LayerNorm of the
        first ``n_write`` keys (before rounding). ``cache``, the slots and ``q_rows`` are not read."""
        x = qa.float()
        normed = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + front.eps)
        normed = (normed * front.gamma.float()).bfloat16()
        with true_fp32_matmul():
            rows = (normed.float() @ front.weight.float().t()).bfloat16()
        q_fp8, index_w = quantize_queries(rows, iw, weight_scale)
        keys = key_layer_norm(ik[:n_write], front.index_norm, front.index_bias, front.index_eps)
        return q_fp8, index_w, rows, keys

    def is_valid(
        self,
        front,
        qa: torch.Tensor,
        ik: torch.Tensor,
        iw: torch.Tensor,
        weight_scale: float,
        cache: torch.Tensor | None = None,
        slot_fp8: torch.Tensor | None = None,
        slot_scale: torch.Tensor | None = None,
        n_write: int = 0,
        q_rows: torch.Tensor | None = None,
    ) -> None:
        """``front.run`` refuses rows it has no kernel for and a non-page cache view; it does not
        check the slot count, the slot range or the ``q_rows`` buffer."""
        if not front.supports(qa, ik, iw):
            raise ValueError("dsa_index_front: rows or operand layout the front was not built for")
        rows = qa.shape[0]
        if q_rows is not None and (
            q_rows.dtype != torch.bfloat16
            or q_rows.shape != (rows, HEADS * DIM)
            or q_rows.stride(1) != 1
        ):
            raise ValueError("dsa_index_front: q_rows must be BF16 [M, 4096] rows")
        if cache is not None and n_write > 0:
            if not 0 < n_write <= rows:
                raise ValueError("dsa_index_front: n_write past the step's rows")
            bytes_ = cache.numel() * cache.element_size()
            for slots, width in ((slot_fp8, DIM), (slot_scale, 4)):
                if slots.shape[0] < n_write:
                    raise ValueError("dsa_index_front: fewer slots than keys to write")
                if int(slots[:n_write].max()) + width > bytes_:
                    raise ValueError("dsa_index_front: a slot past the cache")


dsa_index_front = DsaIndexFront()
