# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DSA indexer prefill logits: FP8 queries against a dense FP8 key matrix (DeepGEMM)."""

from __future__ import annotations

import torch

from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper
from tensorrt_llm.deep_gemm import fp8_fp4_mqa_logits as _deep_gemm_fp8_fp4_mqa_logits
from tensorrt_llm.deep_gemm import fp8_mqa_logits as _deep_gemm_fp8_mqa_logits
from tensorrt_llm.deep_gemm import get_mqa_logits_metadata as _deep_gemm_mqa_logits_metadata


def _spec(requests, **overrides) -> dict:
    """``requests``: (cached keys, new query rows) per request of one prefill chunk."""
    spec = dict(
        requests=tuple(requests),
        q_rows=None,
        num_heads=32,
        head_dim=128,
        data="int",
        scheduled=False,
    )
    spec.update(overrides)
    return spec


class Fp8MqaLogits(OpWrapper):
    """``logits[i, j] = k_scale[j] * sum_h weights[i, h] * relu(q[i, h] . k[j])`` for
    ``cu_seqlen_ks[i] <= j < cu_seqlen_ke[i]``.

    ``q`` is E4M3 [M, H, 128], ``k`` E4M3 [N, 128] with one fp32 scale per key (``k_scale`` [N]),
    ``weights`` fp32 [M, H] (the indexer folds the softmax scale, ``H**-0.5`` and the per-head Q
    scale into them), ``cu_seqlen_ks`` / ``cu_seqlen_ke`` int32 [M] the key window of each query
    row. Returns a new fp32 [M, N] tensor; columns outside each row's window hold unspecified
    values (DeepGEMM's clean-logits pass, the -inf fill outside the windows, does not run).

    ``schedule_meta``, when given, is DeepGEMM's schedule of this call: ``get_mqa_logits_metadata``
    over the same windows, key count and head count. It only places the work (the kernel splits
    the keys over the SMs instead of running its query blocks grid-stride); the logits are
    bitwise those of the call without it.
    """

    ARCHS = frozenset({Arch.SM_103, Arch.SM_107})
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
            why="real-valued UE8M0-scaled keys and weights: accumulation order visible",
            spec=_spec([(512, 1536)], data="randn"),
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
    ARCHS lists what was driven: sm_103 (GB300) and sm_107 (VR200).

    The K3.1 general target calls this from the DSA indexer's prefill for each query tile of each
    prefill chunk: q is the indexer's FP8 Q (32 heads x 128, not split over attention TP), k / k_scale
    the chunk's gathered FP8 index keys and their per-key fp32 scales (k_scale reshaped to 1-D), the
    weights already scaled, and the row windows from the chunk metadata. The prefill top-k reads
    only each row's window, so the out-of-window columns are never read (and the CUDA radix
    prefill top-k reads only the window). The output is
    a fresh allocation per call; the indexer bounds its size
    with a logits element budget by tiling the query rows. Deterministic for equal inputs.
    The general target passes no schedule. The Rubin targets' context steps pass the schedule
    their DSA metadata builds for each query tile once a step; every DSA layer's call of that
    tile reads it. DeepGEMM's fp8_mqa_logits is its fp8_fp4_mqa_logits with no query scale and
    fp32 logits, and only the latter takes a schedule, so a scheduled call goes there.
    No workspace or caller-visible state; the schedule is only read.
    """

    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        k_scale: torch.Tensor,
        weights: torch.Tensor,
        cu_seqlen_ks: torch.Tensor,
        cu_seqlen_ke: torch.Tensor,
        *,
        schedule_meta: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if schedule_meta is None:
            return _deep_gemm_fp8_mqa_logits(
                q, (k, k_scale), weights, cu_seqlen_ks, cu_seqlen_ke, clean_logits=False
            )
        return _deep_gemm_fp8_fp4_mqa_logits(
            (q, None),
            (k, k_scale),
            weights,
            cu_seqlen_ks,
            cu_seqlen_ke,
            clean_logits=False,
            logits_dtype=torch.float32,
            schedule_meta=schedule_meta,
        )

    def reference(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        k_scale: torch.Tensor,
        weights: torch.Tensor,
        cu_seqlen_ks: torch.Tensor,
        cu_seqlen_ke: torch.Tensor,
        *,
        schedule_meta: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """fp64 evaluation; -inf outside each row's window (the caller compares only the window).
        ``schedule_meta`` is ignored: it places the work, not the result."""
        num_heads = q.shape[1]
        kf = k.double() * k_scale.double()[:, None]
        logits = torch.zeros(q.shape[0], k.shape[0], dtype=torch.float64, device=q.device)
        for h in range(num_heads):
            logits += (q[:, h].double() @ kf.t()).relu() * weights[:, h : h + 1].double()
        cols = torch.arange(k.shape[0], device=q.device)[None, :]
        window = (cols >= cu_seqlen_ks[:, None]) & (cols < cu_seqlen_ke[:, None])
        return logits.float().masked_fill(~window, float("-inf"))

    def is_valid(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        k_scale: torch.Tensor,
        weights: torch.Tensor,
        cu_seqlen_ks: torch.Tensor,
        cu_seqlen_ke: torch.Tensor,
        *,
        schedule_meta: torch.Tensor | None = None,
    ) -> None:
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
            built = _deep_gemm_mqa_logits_metadata(
                cu_seqlen_ks, cu_seqlen_ke, k.shape[0], q.shape[1]
            )
            assert torch.equal(schedule_meta, built), "the schedule was built for another call"


fp8_mqa_logits = Fp8MqaLogits()
