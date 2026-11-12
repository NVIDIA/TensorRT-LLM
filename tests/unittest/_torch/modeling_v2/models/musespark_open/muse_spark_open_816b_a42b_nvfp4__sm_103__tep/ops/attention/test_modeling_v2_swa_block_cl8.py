# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the one-cluster sliding-window block of ``swa_block`` (``MUSESPARK_OPEN_SWA_CL8=all``).

A TEP4 decode step of one request (4 KV heads, 32 query heads, 32-key FP8 pages, window 2048, fused
prep, norm-gate, FP8 Q) is run twice on identical inputs: through the CL8 cluster kernel and through
the split + combine pair it replaces. The CL8 kernel is not bit-identical to the pair (another softmax
partition, P in f16), so the gated output is compared within a BF16 tolerance; the step's K/V append
into the FP8 cache is the stock one and must match byte for byte.
"""

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.ops.attention import (  # noqa: E501
    paged_decode_attention as paged_mod,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.ops.attention import (  # noqa: E501
    swa_block as swa_mod,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 3),
    reason="the CL8 kernel is built for sm_103a",
)

HQ, HKV, D, PAGE, WINDOW = 32, 4, 64, 32, 2048
MAX_POSITIONS = 4096


def _step(rows, length, pos_dtype, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)

    def randn(*shape):
        return torch.randn(*shape, generator=g, device="cuda", dtype=torch.float32)

    npages = (length + PAGE - 1) // PAGE
    pool = randn(2 * npages, HKV, PAGE, D).to(torch.float8_e4m3fn)
    # K page ids then V page ids, a shuffled (non-contiguous) mapping into the pool
    perm = torch.randperm(2 * npages, generator=g, device="cuda").to(torch.int32)
    block_tables = perm.view(2, npages).contiguous()
    # raw projection [q | gate] per query head, then K, V of the KV heads
    qkv = randn(rows, 2 * (HQ + HKV) * D).to(torch.bfloat16)
    angle = randn(MAX_POSITIONS, D // 2)
    cos_sin = torch.stack([angle.cos(), angle.sin()], dim=1).contiguous()
    positions = torch.arange(length - rows, length, device="cuda", dtype=pos_dtype)
    seq_lens = torch.tensor([length], dtype=torch.int32, device="cuda")
    scale = torch.ones(1, dtype=torch.float32, device="cuda")
    return pool, block_tables, qkv, cos_sin, positions, seq_lens, scale


def _run(use_cl8, monkeypatch, inputs, rows):
    pool, block_tables, qkv, cos_sin, positions, seq_lens, scale = inputs
    pool, qkv = pool.clone(), qkv.clone()
    monkeypatch.setattr(swa_mod, "_SWA_CL8", "all" if use_cl8 else "0")
    launches = []
    real_launch = swa_mod._cl8_launch
    monkeypatch.setattr(
        swa_mod, "_cl8_launch", lambda *a, **k: launches.append(1) or real_launch(*a, **k)
    )
    gate = qkv[:, : 2 * HQ * D].view(rows, HQ, 2 * D)[..., D:]  # the gate stays in the projection
    out = swa_mod.swa_block_attention(
        qkv,
        pool,
        block_tables,
        seq_lens,
        scale,
        HQ,
        HKV,
        WINDOW,
        paged_mod.window_splits(WINDOW),
        q_fp8=True,
        fused_prep=True,
        cos_sin=cos_sin,
        position_ids=positions,
        prep_eps=1e-6,
        q_scale=1.0,
        gate=gate,
        gate_eps=1e-6,
    )
    torch.cuda.synchronize()
    assert len(launches) == (1 if use_cl8 else 0), "wrong path ran (CL8 fell back or ran when off)"
    return out, pool


@pytest.mark.parametrize(
    "rows,length,pos_dtype",
    [
        (1, 3000, torch.int64),  # window slides: 2048 of 3000 keys
        (1, 3000, torch.int32),  # the int32-position build
        (4, 3000, torch.int64),  # a multi-row (speculative) step
        (1, 1000, torch.int64),  # context shorter than the window
    ],
)
def test_swa_cl8_matches_split_combine_pair(rows, length, pos_dtype, monkeypatch):
    torch.manual_seed(0)
    inputs = _step(rows, length, pos_dtype, seed=length + rows)
    out_pair, pool_pair = _run(False, monkeypatch, inputs, rows)
    out_cl8, pool_cl8 = _run(True, monkeypatch, inputs, rows)

    # The step's K/V append is the stock write: the same FP8 bytes, and it did change the cache.
    assert torch.equal(pool_cl8.view(torch.uint8), pool_pair.view(torch.uint8))
    assert not torch.equal(pool_cl8.view(torch.uint8), inputs[0].view(torch.uint8))

    assert out_cl8.shape == (rows, HQ * D) and out_cl8.dtype == torch.bfloat16
    assert torch.isfinite(out_cl8.float()).all()
    # Outputs are RMS-normalized (O(1)) times a sigmoid; one BF16 ulp is 2^-8 relative, and the pair
    # keeps P in BF16 against CL8's f16, so a few ulps of per-element slack, with a tight mean bound.
    # A wrong window, page or partial merge moves elements by O(0.1..1).
    diff = (out_cl8.float() - out_pair.float()).abs()
    torch.testing.assert_close(out_cl8, out_pair, atol=5e-2, rtol=5e-2)
    assert diff.mean().item() < 1e-2, f"mean abs diff {diff.mean().item()}"
