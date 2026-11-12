# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Ragged FP8 MQA logits: the DSA indexer's prefill scores, in one DeepGEMM call.

Wraps the pybind `tensorrt_llm.deep_gemm.fp8_mqa_logits`, not a
`torch.ops.trtllm.*` op: the ragged prefill scorer has no registered-op form,
and the DSA indexer calls the pybind directly.
"""

from __future__ import annotations

from typing import Tuple

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper
from tensorrt_llm.deep_gemm import fp8_mqa_logits as _deep_gemm_fp8_mqa_logits

#: GLM-5.2's indexer: 32 heads of 128, fp8 keys with one fp32 scale per key.
_GLM = dict(heads=32, head_dim=128)


class Fp8MqaLogits(OpWrapper):
    """Score every query row against every key of one ragged prefill chunk.

    Returns `[T, N]` fp32 logits (a row-padded view, not contiguous) with

        logits[i, j] = k_scale[j] * sum_h weights[i, h] * relu(q[i, h, :] . k[j, :])

    for `cu_seq_len_k_start[i] <= j < cu_seq_len_k_end[i]`, and -inf
    elsewhere when `clean_logits`. `kv` is the pair `(k [N, D] fp8, k_scale [N]
    fp32)`. The fp8 codes of `q` enter the dot product raw: there is no q-side
    scale, so the caller folds it, with the indexer's `D**-0.5 * H**-0.5`, into
    `weights`.
    """

    ARCHS = frozenset({Arch.SM_103})

    # GLM's prefill tiles each chunk's queries so one call's [T, N] buffer
    # stays bounded. `seqs` lists each request in the chunk as (cached keys,
    # new tokens); token `t` of a request sees its own keys up to `cached + t`.
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="one request, no cache: causal windows over the chunk's own keys",
            spec=dict(**_GLM, seqs=[(0, 2048)]),
        ),
        Cell(
            why="a cached prefix: every window starts at 0 and runs past the chunk",
            spec=dict(**_GLM, seqs=[(4096, 1024)]),
        ),
        Cell(
            why="two requests in one call: the second's windows start mid-row",
            spec=dict(**_GLM, seqs=[(0, 300), (1700, 500)]),
        ),
        Cell(
            why="a query count that is not a multiple of the 4-row query block",
            spec=dict(**_GLM, seqs=[(100, 37)]),
        ),
    )

    note = """
    Only `clean_logits=True` with `max_seqlen_k=0` is certified, which is
    what GLM passes. `clean_logits=False` leaves out-of-window entries
    unspecified, and `max_seqlen_k > 0` switches to a compressed layout where
    column `j` of row `i` is key `cu_seq_len_k_start[i] + j`; `reference`
    implements both, no cell drives them.

    fp32 `weights` only. fp16 weights select a separate kernel with a half
    epilogue, and bf16 weights require bf16 logits; GLM uses neither.
    """

    def __call__(
        self,
        q: torch.Tensor,
        kv: Tuple[torch.Tensor, torch.Tensor],
        weights: torch.Tensor,
        cu_seq_len_k_start: torch.Tensor,
        cu_seq_len_k_end: torch.Tensor,
        clean_logits: bool = True,
        max_seqlen_k: int = 0,
    ) -> torch.Tensor:
        return _deep_gemm_fp8_mqa_logits(
            q,
            kv,
            weights,
            cu_seq_len_k_start,
            cu_seq_len_k_end,
            clean_logits,
            max_seqlen_k,
        )

    def reference(
        self,
        q: torch.Tensor,
        kv: Tuple[torch.Tensor, torch.Tensor],
        weights: torch.Tensor,
        cu_seq_len_k_start: torch.Tensor,
        cu_seq_len_k_end: torch.Tensor,
        clean_logits: bool = True,
        max_seqlen_k: int = 0,
    ) -> torch.Tensor:
        """The scores in fp64, rounded once to fp32.

        Outside each row's window the entry is -inf when `clean_logits`, else
        NaN, which the test's gate reads as "unspecified".
        """
        k, k_scale = kv
        qd, kd, wd = q.float().double(), k.float().double(), weights.double()
        n = k.shape[0]
        # One head at a time keeps the transient at [T, N] rather than [T, H, N].
        full = torch.zeros(q.shape[0], n, dtype=torch.float64, device=q.device)
        for h in range(q.shape[1]):
            full += (qd[:, h] @ kd.t()).relu() * wd[:, h : h + 1]
        full *= k_scale.double().unsqueeze(0)
        ks = cu_seq_len_k_start.long().unsqueeze(1)
        ke = cu_seq_len_k_end.long().unsqueeze(1)
        fill = float("-inf") if clean_logits else float("nan")
        if max_seqlen_k == 0:
            cols = torch.arange(n, device=q.device).unsqueeze(0)
            out = full.masked_fill((cols < ks) | (cols >= ke), fill)
        else:
            cols = torch.arange(max_seqlen_k, device=q.device).unsqueeze(0)
            src = (ks + cols).clamp(max=n - 1)
            out = torch.gather(full, 1, src).masked_fill(cols >= ke - ks, fill)
        return out.float()

    def is_valid(
        self,
        q: torch.Tensor,
        kv: Tuple[torch.Tensor, torch.Tensor],
        weights: torch.Tensor,
        cu_seq_len_k_start: torch.Tensor,
        cu_seq_len_k_end: torch.Tensor,
        clean_logits: bool = True,
        max_seqlen_k: int = 0,
    ) -> None:
        # The op asserts `weights.stride(1) == 1` but not stride(0), then reads
        # a packed [T, H] block from the base pointer. A row-padded view scores
        # each query with another row's head weights.
        assert weights.dim() != 2 or weights.stride(0) == weights.size(1), (
            f"weights rows must be packed (stride(0) == num_heads); got stride "
            f"{weights.stride()} for shape {tuple(weights.shape)}"
        )
        assert weights.dtype == torch.float32, (
            f"weights must be fp32; got {weights.dtype}, which selects an uncertified kernel"
        )
        # A window past the last key reads beyond `k`.
        if cu_seq_len_k_end.numel():
            assert int(cu_seq_len_k_start.min()) >= 0, "a window starts before key 0"
            assert int(cu_seq_len_k_end.max()) <= kv[0].shape[0], (
                f"a window ends past the {kv[0].shape[0]} keys passed; the kernel reads beyond k"
            )


fp8_mqa_logits = Fp8MqaLogits()
