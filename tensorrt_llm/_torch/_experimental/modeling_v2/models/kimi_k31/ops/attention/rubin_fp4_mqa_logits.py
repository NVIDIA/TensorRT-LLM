# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DSA indexer prefill logits: MXFP4 queries against a dense MXFP4 key matrix (DeepGEMM).

The FP4 counterpart of ``fp8_mqa_logits``: the same windows, weights and schedule, with queries and
keys in ``rubin_idx_fp4_quantize``'s format, whose block scales DeepGEMM's FP4 kernel applies in
the MMA. DeepGEMM is looked up on the first call, so the entry imports on a host without the built
extension.
"""

from __future__ import annotations

import functools

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from .rubin_idx_fp4_quantize import mxfp4_values

#: Bytes of one 128-element query head or key: packed E2M1 pairs.
DATA_BYTES = 64


@functools.cache
def _deep_gemm():
    """``tensorrt_llm.deep_gemm``, on the first call."""
    from tensorrt_llm import deep_gemm

    return deep_gemm


def _spec(requests, **overrides) -> dict:
    """``requests``: (cached keys, new query rows) per request of one prefill chunk."""
    spec = dict(
        requests=tuple(requests),
        q_rows=None,
        num_heads=32,
        head_dim=128,
        clean_logits=False,
        data="int",
        scheduled=False,
    )
    spec.update(overrides)
    return spec


class RubinFp4MqaLogits(OpWrapper):
    """``logits[i, j] = sum_h weights[i, h] * relu(q[i, h] . k[j])`` for
    ``cu_seqlen_ks[i] <= j < cu_seqlen_ke[i]``.

    ``q[i, h]`` and ``k[j]`` are MXFP4 rows (``rubin_idx_fp4_quantize``): ``q`` ``[M, H, 64]`` and
    ``k`` ``[N, 64]`` packed E2M1 pairs (any 1-byte dtype), ``sf_q`` int32 ``[M, H]`` and ``sf_k``
    int32 ``[N]`` or ``[N, 1]`` their four UE8M0 block exponents. ``weights`` fp32 ``[M, H]`` (the
    indexer folds the softmax scale and ``H**-0.5`` into them, not the query scales, which the MMA
    applies). ``cu_seqlen_ks`` / ``cu_seqlen_ke`` int32 ``[M]`` are the key window of each query
    row. Returns a new fp32 ``[M, N]`` tensor. With ``clean_logits=False`` (what the indexer
    passes) columns outside each row's window hold unspecified values; with True they are -inf.

    ``schedule_meta``, when given, is DeepGEMM's schedule of this call: ``get_mqa_logits_metadata``
    over the same windows, key count and head count. It only places the work; the logits are
    bitwise those of the call without it.
    """

    ARCHS = frozenset({Arch.SM_107})
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="one-token prefill: the smallest chunk, one key",
            spec=_spec([(0, 1)]),
        ),
        Cell(
            why="a short prompt below the top-k: 257 rows over 257 causal keys",
            spec=_spec([(0, 257)]),
        ),
        Cell(
            why="a prompt at the top-k edge: 2049 rows, the last one sees 2049 keys",
            spec=_spec([(0, 2049)]),
        ),
        Cell(
            why="three packed requests: per-request key offsets in one chunk",
            spec=_spec([(0, 100), (0, 2049), (0, 700)]),
        ),
        Cell(
            why="chunked prefill continuation: 1024 rows over an 8192-key cached prefix",
            spec=_spec([(8192, 1024)]),
        ),
        Cell(
            why="a query tile: rows 1000..2999 of a 4096-row request against all its keys",
            spec=_spec([(0, 4096)], q_rows=(1000, 3000)),
        ),
        Cell(
            why="real-valued MXFP4 queries, keys and weights: accumulation order visible",
            spec=_spec([(512, 1536)], data="randn"),
        ),
        Cell(
            why="clean_logits=True: every out-of-window column is -inf",
            spec=_spec([(0, 300), (64, 200)], clean_logits=True),
        ),
        Cell(
            why="three packed requests with their schedule: the keys split over the SMs",
            spec=_spec([(0, 100), (0, 2049), (0, 700)], scheduled=True),
        ),
        Cell(
            why="tep8 context: one rank's 2048 rows of a 16k chunk behind a 64k prefix, scheduled",
            spec=_spec([(65536, 16384)], q_rows=(6144, 8192), scheduled=True),
        ),
        Cell(
            why="dep16 context: a 1795-row tile of a 16k chunk behind a 283k prefix, scheduled",
            spec=_spec([(282624, 16384)], q_rows=(0, 1795), scheduled=True),
        ),
    )
    note = """
    The Rubin default target calls this from the indexer's prefill when its index-key cache is
    FP4, for each query tile of each prefill chunk: q / sf_q the indexer's FP4 queries (32 heads,
    not split over attention TP), k / sf_k the chunk's FP4 index keys gathered from the cache
    (``k31_gather_k_cache_paged`` with the 64-byte payload; its key bytes and exponent words are
    reinterpreted, not converted), the weights, the row windows and the tile's schedule from the
    chunk metadata. It passes clean_logits=False: the prefill top-k reads only each row's window.
    The output is a fresh allocation per call; the indexer bounds its size with a logits element
    budget by tiling the query rows. Deterministic for equal inputs. The schedule is DeepGEMM's own
    for its FP8 / FP4 MQA logits kernels; a scheduled call passes fp32 logits explicitly.
    """

    def __call__(
        self,
        q: torch.Tensor,
        sf_q: torch.Tensor,
        k: torch.Tensor,
        sf_k: torch.Tensor,
        weights: torch.Tensor,
        cu_seqlen_ks: torch.Tensor,
        cu_seqlen_ke: torch.Tensor,
        clean_logits: bool,
        *,
        schedule_meta: torch.Tensor | None = None,
    ) -> torch.Tensor:
        queries = (q.view(torch.int8), sf_q)
        keys = (k.view(torch.int8), sf_k.reshape(-1))
        if schedule_meta is None:
            return _deep_gemm().fp8_fp4_mqa_logits(
                queries, keys, weights, cu_seqlen_ks, cu_seqlen_ke, clean_logits=clean_logits
            )
        return _deep_gemm().fp8_fp4_mqa_logits(
            queries,
            keys,
            weights,
            cu_seqlen_ks,
            cu_seqlen_ke,
            clean_logits=clean_logits,
            logits_dtype=torch.float32,
            schedule_meta=schedule_meta,
        )

    def reference(
        self,
        q: torch.Tensor,
        sf_q: torch.Tensor,
        k: torch.Tensor,
        sf_k: torch.Tensor,
        weights: torch.Tensor,
        cu_seqlen_ks: torch.Tensor,
        cu_seqlen_ke: torch.Tensor,
        clean_logits: bool,
        *,
        schedule_meta: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """fp64 evaluation; -inf outside each row's window (``clean_logits`` is ignored: the
        caller compares only the window unless the op cleaned). ``schedule_meta`` is ignored: it
        places the work, not the result."""
        queries = mxfp4_values(q, sf_q).double()
        keys = mxfp4_values(k, sf_k).double()
        logits = torch.zeros(q.shape[0], k.shape[0], dtype=torch.float64, device=q.device)
        for h in range(q.shape[1]):
            logits += (queries[:, h] @ keys.t()).relu() * weights[:, h : h + 1].double()
        cols = torch.arange(k.shape[0], device=q.device)[None, :]
        window = (cols >= cu_seqlen_ks[:, None]) & (cols < cu_seqlen_ke[:, None])
        return logits.float().masked_fill(~window, float("-inf"))

    def is_valid(
        self,
        q: torch.Tensor,
        sf_q: torch.Tensor,
        k: torch.Tensor,
        sf_k: torch.Tensor,
        weights: torch.Tensor,
        cu_seqlen_ks: torch.Tensor,
        cu_seqlen_ke: torch.Tensor,
        clean_logits: bool,
        *,
        schedule_meta: torch.Tensor | None = None,
    ) -> None:
        # The bytes are read as E2M1 pairs, 64 per 128-element row: 128-byte rows are FP8 data.
        assert q.element_size() == 1 and q.dim() == 3 and q.shape[2] == DATA_BYTES, (
            f"q must be packed E2M1 pairs [M, H, 64], got {q.dtype} {tuple(q.shape)}"
        )
        assert k.element_size() == 1 and k.dim() == 2 and k.shape[1] == DATA_BYTES, (
            f"k must be packed E2M1 pairs [N, 64], got {k.dtype} {tuple(k.shape)}"
        )
        # An fp32 scale is an FP8 row's: its bits would be read as four block exponents.
        assert sf_q.dtype == torch.int32 and sf_q.shape == q.shape[:2], (
            f"sf_q must be int32 [M, H], got {sf_q.dtype} {tuple(sf_q.shape)}"
        )
        assert sf_k.dtype == torch.int32 and sf_k.numel() == k.shape[0], (
            f"sf_k must be one int32 per key, got {sf_k.dtype} {tuple(sf_k.shape)}"
        )
        # The op checks weights.stride(1) but not stride(0), then reads a packed [M, H] block from
        # the base pointer: a row-padded view scores each query with another row's head weights.
        assert weights.stride(0) == weights.shape[1], "weights rows must be packed"
        # fp16 weights select a separate kernel with a half epilogue, bf16 another logits dtype.
        assert weights.dtype == torch.float32, "weights must be fp32"
        # A window past the key matrix reads another allocation's memory.
        assert int(cu_seqlen_ke.max()) <= k.shape[0], "a row window ends past the key matrix"
        assert int(cu_seqlen_ks.min()) >= 0, "a row window starts before the key matrix"
        if schedule_meta is not None:
            # DeepGEMM takes the schedule as given and does not check it against the call: one
            # built for other windows, another key count or another head count does not describe
            # this call's work.
            built = _deep_gemm().get_mqa_logits_metadata(
                cu_seqlen_ks, cu_seqlen_ke, k.shape[0], q.shape[1]
            )
            assert torch.equal(schedule_meta, built), "the schedule was built for another call"


rubin_fp4_mqa_logits = RubinFp4MqaLogits()
