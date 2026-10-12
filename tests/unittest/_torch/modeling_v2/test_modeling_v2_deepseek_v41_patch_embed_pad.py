# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The vision patch embed never hands cublas_mm a row count its is_valid refuses.

``cublas_mm`` refuses bf16 ``[M, 588] x [588, 1024]`` at M 2305..2432, where
cuBLASLt picks a split-K kernel with bf16 partial sums. ``is_valid`` does not
run on the hot path, so the target pads the patch rows past that band itself.
These checks hold that padding to the entry's own statement, for every patch
count the tower can see. They run on the CPU: ``is_valid`` reads only shapes,
dtypes and strides, so meta tensors are enough, and the GEMM in the
``_patch_embed`` check is a stand-in.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog.gemm import cublas_mm
from tensorrt_llm._torch._experimental.modeling_v2.models.deepseek_v41.deepseek_v4_1_flash_nvfp4__sm_103__unified import (  # noqa: E501
    modeling,
)

_K, _N = modeling._VIT_PATCH_IN, modeling._VIT_DIM


def _is_valid(rows: int) -> None:
    """The entry's guard on the patch embed's call, as the target binds it."""
    meta = dict(dtype=torch.bfloat16, device="meta")
    cublas_mm.CublasMm().is_valid(
        mat_a=torch.empty(rows, _K, **meta),
        mat_b=torch.empty(_N, _K, **meta).t(),
        bias=torch.empty(_N, **meta),
    )


def test_the_band_is_the_one_the_entry_refuses():
    lo, hi = modeling._VIT_SPLITK_ROWS
    assert range(lo, hi + 1) == cublas_mm._PATCH_EMBED_SPLIT_K_ROWS
    assert modeling._VIT_SPLITK_PAD == hi + 1
    for rows in (lo, hi):
        with pytest.raises(AssertionError, match="pad M"):
            _is_valid(rows)
    for rows in (lo - 1, hi + 1):
        _is_valid(rows)


def test_every_patch_count_reaches_cublas_mm_with_rows_it_accepts():
    """Each count from one patch up to the tower's per-call envelope."""
    for patches in range(1, modeling._VIT_MAX_PATCHES + 1):
        rows = modeling._patch_embed_rows(patches)
        assert rows >= patches
        assert rows == patches or patches in cublas_mm._PATCH_EMBED_SPLIT_K_ROWS
        assert rows not in cublas_mm._PATCH_EMBED_SPLIT_K_ROWS
        _is_valid(rows)


def test_no_graph_bucket_needs_the_pad():
    for bucket in modeling._VIT_GRAPH_BUCKETS:
        assert modeling._patch_embed_rows(bucket) == bucket
        _is_valid(bucket)


@pytest.mark.parametrize("patches", [1, 2304, 2305, 2368, 2432, 2433, 9216])
def test_patch_embed_pads_with_zero_rows_and_keeps_the_first_ones(patches):
    calls = []

    def gemm(x):
        calls.append(tuple(x.shape))
        assert not torch.any(x[patches:]), "padded rows must be zero"
        return x[:, :1].expand(-1, _N) + 0  # output row r carries input row r

    px = torch.arange(1, patches + 1, dtype=torch.float32)[:, None].expand(-1, _K).contiguous()
    out = modeling.VisionTower._patch_embed(SimpleNamespace(_patch=gemm), px)

    assert calls == [(modeling._patch_embed_rows(patches), _K)]
    assert out.shape == (patches, _N)
    assert torch.equal(out[:, 0], px[:, 0])
