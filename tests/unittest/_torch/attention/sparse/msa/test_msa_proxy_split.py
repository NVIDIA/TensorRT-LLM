# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MiniMax-M3 MSA indexer proxy: segmented, KV-split fmha_sm100 plans."""

from types import ModuleType, SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.attention.backends.sparse.minimax_m3 import msa_indexer
from tensorrt_llm._torch.attention.backends.sparse.minimax_m3.kernels.msa_utils import (
    msa_package_available,
    require_msa_module,
)


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    ("qo_lens", "qo_offsets", "num_index_heads", "expected"),
    [
        ([1025], [131072], 1, 16),
        ([2080], [131072], 1, 8),
        ([4128], [131072], 1, 4),
        ([8224], [131072], 1, 2),
        ([16288], [131072], 1, 1),
        ([4128], [131072], 2, 2),
        ([1500, 50, 300], [60000, 20000, 0], 1, 9),
        # Unsplit: a wave of segments already, a short prefix, short rows.
        ([16288], [131072], 2, 0),
        ([1025], [4095], 1, 0),
        ([64], [131072], 1, 0),
    ],
)
def test_proxy_kv_splits_fill_one_wave(
    monkeypatch: pytest.MonkeyPatch,
    qo_lens: list[int],
    qo_offsets: list[int],
    num_index_heads: int,
    expected: int,
) -> None:
    monkeypatch.setattr(msa_indexer, "_num_sms", lambda: 148)
    assert msa_indexer._proxy_kv_splits(qo_lens, qo_offsets, num_index_heads) == expected


@pytest.mark.cpu_only
def test_segmented_proxy_plan_keeps_causal_windows_and_row_pages() -> None:
    planned = {}

    def fmha_sm100_plan(qo_lens, kv_lens, **kwargs):
        planned.update(qo_lens=qo_lens.tolist(), kv_lens=kv_lens.tolist(), **kwargs)
        return "plan"

    plan, kv_indices = msa_indexer._segmented_proxy_plan(
        SimpleNamespace(fmha_sm100_plan=fmha_sm100_plan),
        [300, 5],
        [1000, 130],
        num_index_heads=2,
        page_size=128,
        num_kv_splits=4,
        kv_indices=torch.arange(100, 113, dtype=torch.int32),
    )

    assert plan == "plan"
    assert planned["qo_lens"] == [128, 128, 44, 5]
    assert planned["kv_lens"] == [1128, 1256, 1300, 135]
    assert planned["qo_offset"].tolist() == [1000, 1128, 1256, 130]
    assert planned["num_kv_splits"] == 4
    # Row 0 owns pages 100-110 and row 1 pages 111-112.
    assert kv_indices.tolist() == [*range(100, 109), *range(100, 110), *range(100, 111), 111, 112]


def _fmha_sm100_or_skip() -> ModuleType:
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    if torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("SM100 (Blackwell) required")
    if not msa_package_available():
        pytest.skip("fmha_sm100 (MSA) not importable")
    return require_msa_module()


# (query tokens, cached prefix) per context row.
_SEGMENTED_PROXY_ROWS = {
    "1k_over_131k": ((1025, 131072),),
    "65_over_70k": ((65, 70000),),
    "4k_over_8k": ((4128, 8192),),
    "16k_over_unaligned_3k": ((16288, 3000),),
    # 33 KV iterations in one segment leave 8+ splits an empty last piece.
    "empty_split_piece": ((1025, 7900),),
    "multi_row": ((1500, 60000), (50, 20000), (300, 0)),
    # The order the unsplit plan cuts into two sub-plans.
    "short_row_first": ((50, 20000), (1500, 60000)),
}


@pytest.mark.parametrize(
    "rows", list(_SEGMENTED_PROXY_ROWS.values()), ids=list(_SEGMENTED_PROXY_ROWS)
)
@pytest.mark.parametrize("num_index_heads", [1, 2, 4])
@pytest.mark.parametrize(
    "indexer_dtype", [torch.bfloat16, torch.float8_e4m3fn], ids=["bf16", "fp8_e4m3fn"]
)
def test_segmented_proxy_scores_match_unsplit_bitwise(
    indexer_dtype: torch.dtype, num_index_heads: int, rows: tuple[tuple[int, int], ...]
) -> None:
    """Top-k ranks these scores, so the whole buffer, -inf entries included, must match."""
    fmha_sm100 = _fmha_sm100_or_skip()
    page_size = head_dim = 128
    sm_scale = head_dim**-0.5
    qo_lens = torch.tensor([qo_len for qo_len, _ in rows], dtype=torch.int32)
    qo_offsets = torch.tensor([prefix for _, prefix in rows], dtype=torch.int32)
    kv_lens = qo_lens + qo_offsets
    num_pages = int(((kv_lens + page_size - 1) // page_size).sum())
    num_tokens = int(qo_lens.sum())
    generator = torch.Generator(device="cuda").manual_seed(num_tokens + num_index_heads)
    # Scattered pages of a larger pool and a strided Q view, as in production.
    index_k = torch.randn(
        num_pages + 64, 1, page_size, head_dim, generator=generator, device="cuda"
    ).to(indexer_dtype)
    kv_indices = torch.randperm(num_pages + 64, generator=generator, device="cuda")[:num_pages]
    kv_indices = kv_indices.to(torch.int32)
    q_width = num_index_heads * head_dim
    index_q = (
        torch.randn(num_tokens, q_width + 64, generator=generator, device="cuda")
        .to(indexer_dtype)[:, :q_width]
        .view(num_tokens, num_index_heads, head_dim)
    )
    reference = msa_indexer._proxy_max_score(
        index_q,
        index_k,
        qo_lens_cpu=qo_lens,
        kv_lens_cpu=kv_lens,
        qo_offset_cpu=qo_offsets,
        kv_indices=kv_indices,
        sm_scale=sm_scale,
        causal=True,
    )
    plans = [
        msa_indexer.plan_proxy(
            fmha_sm100,
            qo_lens,
            kv_lens,
            qo_offsets,
            num_index_heads=num_index_heads,
            page_size=page_size,
            kv_indices=kv_indices,
        )
    ]
    for num_kv_splits in (1, 2, 4, 8, 16):
        plans.append(
            msa_indexer._segmented_proxy_plan(
                fmha_sm100,
                qo_lens.tolist(),
                qo_offsets.tolist(),
                num_index_heads=num_index_heads,
                page_size=page_size,
                num_kv_splits=num_kv_splits,
                kv_indices=kv_indices,
            )
        )
    for plan, plan_kv_indices in plans:
        _, scores = fmha_sm100.fmha_sm100(
            index_q,
            index_k,
            index_k,
            plan,
            kv_indices=plan_kv_indices,
            output_o=False,
            output_maxscore=True,
            sm_scale=sm_scale,
        )
        assert torch.equal(scores.view(torch.int32), reference.view(torch.int32))


def test_prewarm_loads_every_segmented_proxy_variant(monkeypatch: pytest.MonkeyPatch) -> None:
    fmha_sm100 = _fmha_sm100_or_skip()
    from fmha_sm100 import api as fmha_api

    variants = []
    load_variant = fmha_api.get_fmha_variant

    def recording_load(*key):
        variants.append(key)
        return load_variant(*key)

    monkeypatch.setattr(fmha_api, "get_fmha_variant", recording_load)
    page_size = head_dim = 128
    prefix = 4096
    for dtype in (torch.bfloat16, torch.float8_e4m3fn):
        for num_index_heads in (1, 2, 4):
            variants.clear()
            msa_indexer.prewarm_split_proxy_variants.__wrapped__(dtype, num_index_heads, page_size)
            prewarmed = set(variants)
            variants.clear()
            for qo_len in (65, 128, 129, 1025, 4096):
                num_pages = (prefix + qo_len + page_size - 1) // page_size
                kv_indices = torch.arange(num_pages, dtype=torch.int32, device="cuda")
                qo_lens = torch.tensor([qo_len], dtype=torch.int32)
                qo_offsets = torch.tensor([prefix], dtype=torch.int32)
                plan, plan_kv_indices = msa_indexer.plan_proxy(
                    fmha_sm100,
                    qo_lens,
                    qo_lens + qo_offsets,
                    qo_offsets,
                    num_index_heads=num_index_heads,
                    page_size=page_size,
                    kv_indices=kv_indices,
                )
                assert plan_kv_indices is not kv_indices, "expected a segmented plan"
                index_q = torch.zeros(qo_len, num_index_heads, head_dim, dtype=dtype, device="cuda")
                index_k = torch.zeros(num_pages, 1, page_size, head_dim, dtype=dtype, device="cuda")
                fmha_sm100.fmha_sm100(
                    index_q,
                    index_k,
                    index_k,
                    plan,
                    kv_indices=plan_kv_indices,
                    output_o=False,
                    output_maxscore=True,
                )
            assert variants and set(variants) <= prewarmed
