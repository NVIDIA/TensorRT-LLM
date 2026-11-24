# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The key-shard DSA decode indexer's FP32 logits: ``kernels/attention/rubin_ull_dsa_index_score.py``'s key-balanced
persistent tcgen05 scorer (after vLLM e3ddbdbd60's ``IndexerScore``)."""

from __future__ import annotations

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from ...kernels.attention import rubin_ull_dsa_index_score as _score

PAGE = _score.PAGE
DIM = _score.DIM
HEADS = _score.HEADS


def _cell(why: str, lens: list[list[int]], tops_from: int = 0) -> Cell:
    """``lens[b]``: request ``b``'s local key count per row (``next_n`` rows a request)."""
    return Cell(why=why, spec=dict(lens=tuple(map(tuple, lens)), tops_from=tops_from))


def _helix_rows(local: int, next_n: int = 8) -> list[int]:
    """The rows of one request on a Helix rank: row ``t`` sees ``local - (next_n - 1) + t`` keys."""
    return [local - (next_n - 1) + t for t in range(next_n)]


class DsaIndexerScore(OpWrapper):
    """``out[t, k] = scale_k * sum_h relu(q[t, h] . key_k) * weights[t, h]`` for every key ``k`` below
    ``row_lens[t]``, from one launch whose 128-key tiles are spread evenly over the SMs.

    ``q`` [B, next_n, 32, 128] FP8 (row ``t = b * next_n + i``), ``kv_cache`` the indexer K pool
    [blocks, 64, 1, 132] uint8, page-major (64 x 128 FP8 codes, then 64 FP32 scale words),
    ``weights`` [B * next_n, 32] FP32 (already scaled), ``row_lens`` int32 (row ``t``'s local key
    count; request ``b`` reads ``block_table[b, :ceil(max_i row_lens / 64)]``), ``out`` FP32
    [>= rows, pitch] with unit column stride. With ``tops`` (uint32 [>= rows, pitch / 8]) also each
    8 keys' largest logit as its order-preserving image, on the rows of at least ``tops_from`` keys.
    Writes ``out`` (and ``tops``) in place; returns None.
    """

    ARCHS = frozenset({Arch.SM_103})

    CELLS: tuple[Cell, ...] = (
        _cell(
            "served b1 decode: 1 request x 8 rows, ~225k context over 16 ranks",
            [_helix_rows(14080)],
        ),
        _cell(
            "512k context over 16 ranks: more tiles than one CTA's scale stages",
            [_helix_rows(32768)],
        ),
        _cell(
            "b4: 4 requests x 8 rows (32 rows, the two-warpgroup layout), unequal contexts",
            [_helix_rows(n) for n in (14080, 3000, 129, 9000)],
        ),
        _cell("q1 decode row (next_n 1, two accumulators)", [[1000]]),
        _cell("tile edges: 1, 64, 128 and 129 keys", [[1], [64], [128], [129]]),
        _cell(
            "the LL exchange's vector maxima, every row (tops_from 1)",
            [_helix_rows(14080)],
            tops_from=1,
        ),
        _cell(
            "vector maxima only on rows past one register chunk (tops_from 4097)",
            [_helix_rows(4100)],
            tops_from=4097,
        ),
    )

    note = """
    The decode indexer of the K3.1 ULL generation target (Helix CP16 key shards): rows of up to 8
    tokens per request (DFlash2 verify), up to 512 requests. FP32 accumulator and epilogue in
    FP8MQALogitsKernel's order, so the logits are the tree kernel's bits for every key a row sees;
    columns at or past a row's length are not written. Callers check ``unsupported`` first (the
    kernel does not); ``is_valid`` raises where it would refuse. The first call per shape compiles,
    so it must run outside CUDA-graph capture. Launched with PDL.
    """

    def __call__(
        self,
        q: torch.Tensor,
        kv_cache: torch.Tensor,
        weights: torch.Tensor,
        row_lens: torch.Tensor,
        block_table: torch.Tensor,
        out: torch.Tensor,
        tops: torch.Tensor | None = None,
        tops_from: int = 1,
    ) -> None:
        _score.score_logits_pdl(
            q, kv_cache, weights, row_lens, block_table, out, tops=tops, tops_from=tops_from
        )

    def reference(
        self,
        q: torch.Tensor,
        kv_cache: torch.Tensor,
        weights: torch.Tensor,
        row_lens: torch.Tensor,
        block_table: torch.Tensor,
        out: torch.Tensor,
        tops: torch.Tensor | None = None,
        tops_from: int = 1,
    ) -> list[torch.Tensor]:
        """Each row's FP64 logits over its keys (a list, row ``t``: ``row_lens[t]`` values). ``out``,
        ``tops`` and ``tops_from`` are not read: the vector maxima are a function of the logits."""
        requests, next_n = q.shape[0], q.shape[1]
        flat = kv_cache.reshape(kv_cache.shape[0], PAGE * (DIM + 4))
        codes = flat[:, : PAGE * DIM].contiguous().view(torch.float8_e4m3fn).view(-1, PAGE, DIM)
        scales = flat[:, PAGE * DIM :].contiguous().view(torch.float32).view(-1, PAGE)
        lens = row_lens[: requests * next_n].tolist()
        rows = []
        for t, n in enumerate(lens):
            b = t // next_n
            pages = block_table[b, : -(-n // PAGE)].long()
            keys = codes[pages].reshape(-1, DIM)[:n].float().double()
            scale = scales[pages].reshape(-1)[:n].double()
            x = q[b, t % next_n].float().double() @ keys.T  # [heads, n]
            w = weights.view(-1, HEADS)[t].double()
            rows.append((x.clamp_min(0) * w[:, None]).sum(0) * scale)
        return rows

    def is_valid(
        self,
        q: torch.Tensor,
        kv_cache: torch.Tensor,
        weights: torch.Tensor,
        row_lens: torch.Tensor,
        block_table: torch.Tensor,
        out: torch.Tensor,
        tops: torch.Tensor | None = None,
        tops_from: int = 1,
    ) -> None:
        """The kernel does not check its arguments: raise where ``score.unsupported`` refuses them,
        where a row's keys run past ``out``'s pitch, or where the maxima are laid out otherwise."""
        reason = _score.unsupported(q, kv_cache, weights, row_lens, block_table, out)
        if reason is not None:
            raise ValueError(f"dsa_indexer_score: unsupported {reason}")
        rows = q.shape[0] * q.shape[1]
        if int(row_lens[:rows].max()) > out.shape[1]:
            raise ValueError("dsa_indexer_score: a row's keys run past the logits pitch")
        if tops is not None and (
            tops.dtype not in (torch.int32, torch.uint32)
            or tops.stride(1) != 1
            or tops.stride(0) * 8 != out.stride(0)
            or tops.shape[0] < rows
        ):
            raise ValueError("dsa_indexer_score: tops must be 32-bit [rows, pitch / 8] beside out")


dsa_indexer_score = DsaIndexerScore()
