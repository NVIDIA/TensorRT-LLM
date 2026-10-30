# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MiniMax-M3 MSA indexer proxy: segmented, KV-split fmha_sm100 plans."""

import pytest
import torch

from tensorrt_llm._torch.attention.backends.sparse.minimax_m3 import msa_indexer
from tensorrt_llm._torch.attention.backends.sparse.minimax_m3.kernels.msa_utils import (
    msa_package_available,
    per_token_valid_blocks,
    require_msa_module,
)

PAGE_SIZE = HEAD_DIM = 128


def _proxy_inputs(rows, num_index_heads, indexer_dtype):
    """Index Q and K of these (query tokens, cached prefix) rows, and the unsplit proxy's scores.

    The pages are scattered over a larger pool and Q is a strided view, as in production.
    """
    qo_lens = torch.tensor([qo_len for qo_len, _ in rows], dtype=torch.int32)
    qo_offsets = torch.tensor([prefix for _, prefix in rows], dtype=torch.int32)
    kv_lens = qo_lens + qo_offsets
    num_pages = int(((kv_lens + PAGE_SIZE - 1) // PAGE_SIZE).sum())
    num_tokens = int(qo_lens.sum())
    generator = torch.Generator(device="cuda").manual_seed(num_tokens + num_index_heads)
    index_k = torch.randn(
        num_pages + 64, 1, PAGE_SIZE, HEAD_DIM, generator=generator, device="cuda"
    ).to(indexer_dtype)
    kv_indices = torch.randperm(num_pages + 64, generator=generator, device="cuda")[:num_pages]
    kv_indices = kv_indices.to(torch.int32)
    q_width = num_index_heads * HEAD_DIM
    index_q = (
        torch.randn(num_tokens, q_width + 64, generator=generator, device="cuda")
        .to(indexer_dtype)[:, :q_width]
        .view(num_tokens, num_index_heads, HEAD_DIM)
    )
    reference = msa_indexer._proxy_max_score(
        index_q,
        index_k,
        qo_lens_cpu=qo_lens,
        kv_lens_cpu=kv_lens,
        qo_offset_cpu=qo_offsets,
        kv_indices=kv_indices,
        sm_scale=HEAD_DIM**-0.5,
        causal=True,
    )
    return index_q, index_k, kv_indices, qo_lens, kv_lens, reference


def _skip_without_msa():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("SM100 (Blackwell) required")
    if not msa_package_available():
        pytest.skip("fmha_sm100 (MSA) not importable")


@pytest.mark.parametrize(
    "rows",
    [((1025, 131072),), ((16288, 3000),), ((1500, 60000), (50, 20000), (300, 0))],
    ids=["1k_over_131k", "16k_over_unaligned_3k", "multi_row"],
)
@pytest.mark.parametrize("num_index_heads", [1, 2])
@pytest.mark.parametrize(
    "indexer_dtype", [torch.bfloat16, torch.float8_e4m3fn], ids=["bf16", "fp8"]
)
def test_segmented_proxy_scores_match_unsplit_bitwise(
    indexer_dtype: torch.dtype, num_index_heads: int, rows: tuple[tuple[int, int], ...]
) -> None:
    """Top-k ranks these scores, so the whole buffer, -inf entries included, must match."""
    _skip_without_msa()
    fmha_sm100 = require_msa_module()
    index_q, index_k, kv_indices, qo_lens, kv_lens, reference = _proxy_inputs(
        rows, num_index_heads, indexer_dtype
    )
    for num_kv_splits in (1, 3, 16):
        plan, plan_kv_indices = msa_indexer._segmented_proxy_plan(
            fmha_sm100,
            qo_lens.tolist(),
            (kv_lens - qo_lens).tolist(),
            num_index_heads=num_index_heads,
            page_size=PAGE_SIZE,
            num_kv_splits=num_kv_splits,
            kv_indices=kv_indices,
        )
        _, scores = fmha_sm100.fmha_sm100(
            index_q,
            index_k,
            index_k,
            plan,
            kv_indices=plan_kv_indices,
            output_o=False,
            output_maxscore=True,
            sm_scale=HEAD_DIM**-0.5,
        )
        assert torch.equal(scores.view(torch.int32), reference.view(torch.int32))


@pytest.mark.parametrize(
    "rows",
    [((1025, 131072),), ((16288, 3000),), ((1500, 60000), (50, 20000), (300, 0))],
    ids=["1k_over_131k", "16k_over_unaligned_3k", "multi_row"],
)
@pytest.mark.parametrize("num_index_heads", [1, 2, 4])
def test_fp8_context_scores_match_proxy_bitwise(
    num_index_heads: int, rows: tuple[tuple[int, int], ...]
) -> None:
    """The CuTe DSL context scorer gives the proxy's FP8 scores bit for bit on every visible block.

    fmha_sm100 takes the max score off the accumulator before its softmax scale, so the raw Q.K
    compare directly; blocks past a token's valid count are undefined on both sides.
    """
    _skip_without_msa()
    index_q, index_k, kv_indices, qo_lens, kv_lens, reference = _proxy_inputs(
        rows, num_index_heads, torch.float8_e4m3fn
    )
    scores = msa_indexer._context_scores(index_q, index_k, qo_lens, kv_lens, kv_indices)
    n_valid = per_token_valid_blocks(
        qo_lens, kv_lens, kv_lens - qo_lens, causal=True, block_size=PAGE_SIZE
    ).cuda()
    valid = torch.arange(scores.shape[1], device="cuda")[:, None] < n_valid
    assert torch.equal(
        scores[:, valid].view(torch.int32),
        reference[:, : scores.shape[1]][:, valid].view(torch.int32),
    )
