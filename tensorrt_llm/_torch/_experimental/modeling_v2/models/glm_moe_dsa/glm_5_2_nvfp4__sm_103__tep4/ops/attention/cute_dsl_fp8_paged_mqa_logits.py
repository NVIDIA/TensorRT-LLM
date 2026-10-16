# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Paged FP8 MQA logits: the DSA indexer's decode scores, in one CuTe DSL call."""

from __future__ import annotations

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

#: GLM-5.2's indexer: 32 heads of 128; the K cache holds 32 tokens per page,
#: each as 128 fp8 bytes plus one fp32 scale (132 bytes).
_GLM = dict(heads=32, head_dim=128, tokens_per_block=32)


class CuteDslFp8PagedMqaLogits(OpWrapper):
    """Score each decode query token against its own sequence's paged keys.

    `q` is `[B, next_n, H, D]` fp8; `kv_fused` is the indexer K cache
    `[num_pages, page, 1, D + 4]` uint8, read through `block_table[b]`. Returns
    `[B * next_n, max_context_len]` fp32 with

        logits[b * next_n + t, j] = k_scale[j] * sum_h weights[b * next_n + t, h]
                                    * relu(q[b, t, h, :] . k[j, :])

    where `k` / `k_scale` are sequence `b`'s keys. Only the causal part of
    each row, `j < context_lens[b] - next_n + t + 1`, is defined; that is all
    the decode top-k reads. As with `fp8_mqa_logits`, the q-side scale must be
    folded into `weights`.

    The result is a view of a shared arena buffer, not a fresh tensor: the
    next call returns the same storage and overwrites it. Consume the logits
    before scoring again.
    """

    ARCHS = frozenset({Arch.SM_103})

    # GLM scores every generation token of a decode step in one call: up to 32
    # sequences x MTP 2. `max_context_len` is the engine's max sequence length,
    # so most of each row is past the sequence's end. Context lengths are the
    # sequences' KV lengths after this step.
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="one sequence, one token",
            spec=dict(**_GLM, next_n=1, context_lens=[3000], max_context_len=4096),
        ),
        Cell(
            why="MTP 2: two query tokens per sequence, each with its own causal end",
            spec=dict(**_GLM, next_n=2, context_lens=[2, 33, 1000], max_context_len=2048),
        ),
        Cell(
            why="the full decode batch: 32 sequences x MTP 2",
            spec=dict(
                **_GLM,
                next_n=2,
                context_lens=[100 + 251 * i for i in range(32)],
                max_context_len=8192,
            ),
        ),
        Cell(
            why="long contexts against a wide row, with a ragged page count",
            spec=dict(**_GLM, next_n=1, context_lens=[65536, 12345], max_context_len=65536),
        ),
    )

    note = """
    The schedule comes from `tensorrt_llm.deep_gemm.get_paged_mqa_logits_metadata`
    over `context_lens.view(-1, 1)`, a 64-token schedule block, and the
    device's SM count -- which is how the DSA indexer builds it.

    The op is SM100-family only and raises elsewhere. It does not validate its
    inputs: the shapes are the caller's contract, and `is_valid` states the
    one a wrong value of which still returns plausible logits.

    `next_n` above 2, `num_epi_subtiles` other than 1, and non-fp32
    `epi_dtype` / `acc_dtype` / `output_dtype` are not certified.
    """

    def __call__(
        self,
        q: torch.Tensor,
        kv_fused: torch.Tensor,
        weights: torch.Tensor,
        context_lens: torch.Tensor,
        block_table: torch.Tensor,
        schedule_meta: torch.Tensor,
        max_context_len: int,
        num_epi_subtiles: int = 1,
        epi_dtype: torch.dtype = torch.float32,
        acc_dtype: torch.dtype = torch.float32,
        output_dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        return torch.ops.trtllm.cute_dsl_fp8_paged_mqa_logits(
            q,
            kv_fused,
            weights,
            context_lens,
            block_table,
            schedule_meta,
            max_context_len,
            num_epi_subtiles,
            epi_dtype,
            acc_dtype,
            output_dtype,
        )

    @staticmethod
    def unpack_cache(kv_fused: torch.Tensor, head_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
        """`(k [num_pages, page, D] uint8 codes, k_scale [num_pages, page] fp32)`.

        A page stores its tokens' data first, then their scales. The codes stay
        uint8 so they can be gathered by page; view them as e4m3 afterwards.
        """
        pages, page = kv_fused.shape[0], kv_fused.shape[1]
        flat = kv_fused.view(torch.uint8).reshape(pages, -1)
        k = flat[:, : page * head_dim].reshape(pages, page, head_dim)
        k_scale = flat[:, page * head_dim : page * (head_dim + 4)].contiguous()
        return k, k_scale.view(torch.float32).reshape(pages, page)

    def reference(
        self,
        q: torch.Tensor,
        kv_fused: torch.Tensor,
        weights: torch.Tensor,
        context_lens: torch.Tensor,
        block_table: torch.Tensor,
        schedule_meta: torch.Tensor,
        max_context_len: int,
        num_epi_subtiles: int = 1,
        epi_dtype: torch.dtype = torch.float32,
        acc_dtype: torch.dtype = torch.float32,
        output_dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """The causal scores in fp64, rounded once; NaN where unspecified.

        `schedule_meta` and the three dtype/tiling knobs do not change the
        value and are not read.
        """
        batch, next_n, heads, head_dim = q.shape
        page = kv_fused.shape[1]
        k_pages, s_pages = self.unpack_cache(kv_fused, head_dim)
        out = torch.full(
            (batch * next_n, max_context_len), float("nan"), dtype=torch.float64, device=q.device
        )
        lens = context_lens.tolist()
        for b in range(batch):
            n = lens[b]
            if n <= 0:
                continue
            pages = block_table[b, : (n + page - 1) // page].long()
            codes = k_pages[pages].reshape(-1, head_dim)[:n].contiguous()
            k = codes.view(torch.float8_e4m3fn).float().double()
            s = s_pages[pages].reshape(-1)[:n].double()
            for t in range(next_n):
                row = b * next_n + t
                end = n - next_n + t + 1
                if end <= 0:
                    continue
                qd = q[b, t].float().double()
                dots = (qd @ k[:end].t()).relu()
                w = weights[row].double().unsqueeze(1)
                out[row, :end] = (dots * w).sum(0) * s[:end]
        return out.float()

    def is_valid(
        self,
        q: torch.Tensor,
        kv_fused: torch.Tensor,
        weights: torch.Tensor,
        context_lens: torch.Tensor,
        block_table: torch.Tensor,
        schedule_meta: torch.Tensor,
        max_context_len: int,
        num_epi_subtiles: int = 1,
        epi_dtype: torch.dtype = torch.float32,
        acc_dtype: torch.dtype = torch.float32,
        output_dtype: torch.dtype = torch.float32,
    ) -> None:
        # One schedule row per CTA plus one, and the op launches one CTA per
        # SM. The schedule's producer takes its own SM count, so the two can
        # disagree: a schedule built for more SMs returns wrong logits without
        # raising, one built for fewer faults the context.
        expected = torch.cuda.get_device_properties(q.device).multi_processor_count + 1
        assert tuple(schedule_meta.shape) == (expected, 2), (
            f"schedule_meta must be [{expected}, 2] (SM count + 1 rows); got "
            f"{tuple(schedule_meta.shape)}"
        )
        # The output has `max_context_len` columns; a longer context has no
        # column to land in and is not checked.
        if context_lens.numel():
            assert int(context_lens.max()) <= max_context_len, (
                f"a context of {int(context_lens.max())} keys exceeds max_context_len "
                f"{max_context_len}"
            )


cute_dsl_fp8_paged_mqa_logits = CuteDslFp8PagedMqaLogits()
