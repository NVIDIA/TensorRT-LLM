# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DSA indexer decode Top-K: the self-sampling GVR engine (CuTe DSL), deterministic order.

The K3.1 indexer's decode Top-K on Rubin: the engine ``TopK`` module with the self-sampling GVR
implementation and ``gvr_deterministic_topk`` (ties go to the smaller index, a fixed output order
per route). The engine kernel is called directly; no module, no torch op.
"""

from __future__ import annotations

import torch

from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper


def _spec(batch, next_n, kv_lens, max_seq_len, **overrides) -> dict:
    """``kv_lens``: (low, high) of each request's total KV length, drawn per cell."""
    spec = dict(
        batch=batch, next_n=next_n, kv_lens=tuple(kv_lens), max_seq_len=max_seq_len, top_k=2048
    )
    spec.update(overrides)
    return spec


class RubinGvrTopkDecode(OpWrapper):
    """``indices[r]`` = the ``k`` (``indices.shape[1]``) largest of ``logits[r, :n_r]`` as column
    indices, ``-1`` padded, where ``n_r = kv_lens[r // next_n] - next_n + r % next_n + 1``; a row
    with ``n_r <= k`` gets ``0 .. n_r - 1`` then ``-1``. Ties go to the smaller column.

    ``logits`` FP32 ``[B * next_n, >= n_r]`` row-major with a row stride that is a multiple of 4
    and a 16-byte aligned base (the DSL paged logits arena); ``kv_lens`` int32 ``[B]`` (total KV
    length per request, in tokens); ``indices`` int32 ``[B * next_n, k]``, written in place and
    returned. ``max_seq_len`` is the engine's static logits width (the kernel's tuning envelope:
    no host read, CUDA-graph safe).
    """

    ARCHS = frozenset({Arch.SM_107})
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="TEP8 GEN verify b1 k7: 8 rows over one long context",
            spec=_spec(1, 8, (200000, 260000), 262144),
        ),
        Cell(
            why="DEP16 GEN verify b12 k7: 96 rows",
            spec=_spec(12, 8, (24000, 64000), 65536),
        ),
        Cell(
            why="rows at the top-k edge and below it: the short path's identity and -1 tail",
            spec=_spec(4, 8, (2040, 2060), 4096),
        ),
        Cell(
            why="plain decode, several requests: next_n 1",
            spec=_spec(16, 1, (3000, 9000), 16384),
        ),
        Cell(
            why="ties: integer logits, so the smaller-index rule decides the set",
            spec=_spec(4, 8, (5000, 9000), 16384, data="ties"),
        ),
    )
    note = """
    The deterministic kernels are the engine's ``selfsampling_topk_run_varlen(..., deterministic=
    True)``; the order of a row's selected indices is the kernel's (fixed per route, not sorted),
    so the test compares sets. The format gate (FP32, unit column stride, row stride % 4, 16-byte
    base, and a single row's width % 4) is the caller's: outside it the engine module took the
    radix ``indexer_topk_decode``, which the targets bind where the gate can fail.
    """

    def __call__(
        self,
        logits: torch.Tensor,
        kv_lens: torch.Tensor,
        indices: torch.Tensor,
        next_n: int,
        max_seq_len: int,
    ) -> torch.Tensor:
        from tensorrt_llm._torch.kernels.blackwell.top_k import selfsampling_topk_run_varlen

        selfsampling_topk_run_varlen(
            logits,
            kv_lens,
            indices,
            next_n=next_n,
            compress_ratio=1,
            max_seq_len=max_seq_len,
            deterministic=True,
            sorting=None,
        )
        return indices

    def reference(
        self,
        logits: torch.Tensor,
        kv_lens: torch.Tensor,
        indices: torch.Tensor,
        next_n: int,
        max_seq_len: int,
    ) -> torch.Tensor:
        """A new ``[rows, k]`` tensor: each row's selected columns in ascending order, then -1.
        Ties by the smaller column (a stable sort on ``-logit``). ``max_seq_len`` does not change
        the set; ``indices`` is left untouched."""
        rows, k = indices.shape
        out = torch.full((rows, k), -1, dtype=torch.int32, device=logits.device)
        for r in range(rows):
            n = int(kv_lens[r // next_n]) - next_n + r % next_n + 1
            if n <= 0:
                continue
            if n <= k:
                out[r, :n] = torch.arange(n, dtype=torch.int32, device=logits.device)
                continue
            order = torch.sort(-logits[r, :n].double(), stable=True).indices[:k]
            out[r] = torch.sort(order).values.to(torch.int32)
        return out

    def is_valid(
        self,
        logits: torch.Tensor,
        kv_lens: torch.Tensor,
        indices: torch.Tensor,
        next_n: int,
        max_seq_len: int,
    ) -> None:
        # The kernel reads rows through the row stride as float4 vectors: another layout reads
        # misaligned columns.
        assert logits.dtype == torch.float32 and logits.stride(1) == 1, "FP32, unit column stride"
        assert logits.stride(0) % 4 == 0 and logits.data_ptr() % 16 == 0, (
            "a row stride of whole float4 vectors and a 16-byte aligned base"
        )
        assert logits.shape[0] > 1 or logits.shape[1] % 4 == 0, (
            "a single row derives its window from the width, which must be whole float4s"
        )
        # The tuning envelope comes from max_seq_len: a row longer than it is cut short.
        assert int(kv_lens.max()) <= max_seq_len, "a KV length past max_seq_len"
        assert indices.dtype == torch.int32 and indices.shape[0] == logits.shape[0]


rubin_gvr_topk_decode = RubinGvrTopkDecode()


def rubin_gvr_topk_warmup(
    top_k: int,
    max_seq_len: int,
    compress_ratio: int,
    next_n: int,
    num_rows: tuple[int, ...],
    row_stride: int | None,
) -> None:
    """Compile the self-sampling GVR Top-K the targets run, before serving (a warmup step, not an
    entry): the prefill engine (``rubin_gvr_topk_prefill``) and, unless ``row_stride`` is None,
    the decode engine (``rubin_gvr_topk_decode``) for each row count of ``num_rows`` at that
    logits row stride. Out of memory only skips a part; a missing engine module raises."""
    from tensorrt_llm._torch.kernels.blackwell.top_k import (
        gvr_topk_decode_self_sampling_host as ss_host,
    )
    from tensorrt_llm.logger import logger

    try:
        ss_host.warmup_prefill(top_k, max(max_seq_len, 32768), deterministic=True)
    except torch.cuda.OutOfMemoryError:
        logger.warning("self-sampling GVR prefill warmup ran out of memory")
    if row_stride is None:
        return
    try:
        ss_host.warmup_varlen(
            top_k,
            max_seq_len * compress_ratio,
            compress_ratio=compress_ratio,
            next_n=next_n,
            num_rows_list=num_rows,
            row_stride=row_stride,
            deterministic=True,
        )
    except torch.cuda.OutOfMemoryError:
        logger.warning("self-sampling GVR warmup ran out of memory")
