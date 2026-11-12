# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for ``swa_block_multi_attention`` (``MUSESPARK_OPEN_SWA_BLOCK_MULTI=1``).

A TEP4 decode step of 2..4 requests of 1, 4 or 8 rows each (4 KV heads, 32 query heads, 32-key FP8
pages, window 2048, fused prep, norm-gate, FP8 Q) runs once through the multi-request call and once
per request through ``swa_block_attention``. Each request's output rows and every cache byte the
step writes must be identical: each multi-request program is the one-request program on that
request's rows, length, page ids and positions. Under ``MUSESPARK_OPEN_SWA_CL8=all`` 2..3 requests
take one CL8 launch (one cluster per request and KV head); 4 requests take the split + combine pair,
as the one-request calls do with CL8 off. Context lengths sit on the window's edges (2048 + 8, + 9,
+ 15, + 16) and far past it (90,001); both raw projection layouts and two KV scales are covered.
"""

import pytest
import torch

pytest.importorskip(
    "cuda.bindings"
)  # the CL8 kernel is NVRTC-compiled and launched via the driver API

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
LENGTHS = (2056, 2057, 2063, 2064, 90_001)


def _step(reqs, rows, lengths, permuted, kv_scale, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)

    def randn(*shape):
        return torch.randn(*shape, generator=g, device="cuda", dtype=torch.float32)

    npages = (max(lengths) + PAGE - 1) // PAGE
    pool = randn(reqs * 2 * npages, HKV, PAGE, D).to(torch.float8_e4m3fn)
    # each request's K then V page ids: disjoint, shuffled (non-contiguous) slices of one pool
    perm = torch.randperm(reqs * 2 * npages, generator=g, device="cuda").to(torch.int32)
    block_tables = perm.view(reqs, 2, npages).contiguous()
    # raw projection, request-major rows: [q_0 | gate_0 | ... | k | v] or [q | k | v | gate]
    qkv = randn(reqs * rows, 2 * (HQ + HKV) * D).to(torch.bfloat16)
    if permuted:
        gate = qkv[:, (HQ + 2 * HKV) * D :]
    else:
        gate = qkv[:, : 2 * HQ * D].view(reqs * rows, HQ, 2 * D)[..., D:]
    angle = randn(max(lengths), D // 2)
    cos_sin = torch.stack([angle.cos(), angle.sin()], dim=1).contiguous()
    positions = torch.cat(
        [torch.arange(n - rows, n, device="cuda", dtype=torch.int64) for n in lengths]
    )
    seq_lens = torch.tensor(lengths, dtype=torch.int32, device="cuda")
    scale = torch.full((1,), kv_scale, dtype=torch.float32, device="cuda")
    return pool, block_tables, qkv, gate, cos_sin, positions, seq_lens, scale


def _call(op, qkv, pool, block_tables, seq_lens, scale, cos_sin, positions, gate, permuted, rows):
    return op(
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
        qkv_permuted=permuted,
        gate=gate,
        gate_eps=1e-6,
    )


@pytest.mark.parametrize("kv_scale", [0.5, 2.0])
@pytest.mark.parametrize("permuted", [False, True], ids=["checkpoint", "permuted"])
@pytest.mark.parametrize("cl8", ["all", "0"])
@pytest.mark.parametrize("rows", [8, 4, 1])
@pytest.mark.parametrize("reqs", [2, 3, 4])
def test_swa_block_multi_matches_one_request_calls(
    reqs, rows, cl8, permuted, kv_scale, monkeypatch
):
    # every edge length appears across the request counts; 90,001 in every step of 3..4 requests
    lengths = [LENGTHS[(reqs + i) % len(LENGTHS)] for i in range(reqs)]
    pool, block_tables, qkv, gate, cos_sin, positions, seq_lens, scale = _step(
        reqs, rows, lengths, permuted, kv_scale, seed=100 * reqs + rows
    )
    launches = []
    real_launch = swa_mod._cl8_launch
    monkeypatch.setattr(
        swa_mod, "_cl8_launch", lambda *a, **k: launches.append(1) or real_launch(*a, **k)
    )
    multi_cl8 = cl8 == "all" and reqs <= swa_mod.CL8_MAX_REQUESTS

    monkeypatch.setattr(swa_mod, "_SWA_CL8", cl8)
    pool_multi = pool.clone()
    out_multi = _call(
        swa_mod.swa_block_multi_attention,
        qkv,
        pool_multi,
        block_tables,
        seq_lens,
        scale,
        cos_sin,
        positions,
        gate,
        permuted,
        rows,
    )
    torch.cuda.synchronize()
    assert len(launches) == (1 if multi_cl8 else 0), "wrong multi path (CL8 fell back or ran)"

    # the one-request reference runs the kernel the multi call ran: CL8 at 2..3, the pair at 4
    monkeypatch.setattr(swa_mod, "_SWA_CL8", "all" if multi_cl8 else "0")
    launches.clear()
    pool_one = pool.clone()
    outs = []
    for b in range(reqs):
        sl = slice(b * rows, (b + 1) * rows)
        outs.append(
            _call(
                swa_mod.swa_block_attention,
                qkv[sl],
                pool_one,
                block_tables[b],
                seq_lens[b : b + 1],
                scale,
                cos_sin,
                positions[sl],
                gate[sl],
                permuted,
                rows,
            )
        )
    torch.cuda.synchronize()
    assert len(launches) == (reqs if multi_cl8 else 0), "wrong one-request path"
    out_one = torch.cat(outs)

    orig, b_multi, b_one = (p.view(torch.uint8) for p in (pool, pool_multi, pool_one))
    assert (b_one != orig).any(), "the K/V append did not change the cache"
    assert torch.equal(b_multi, b_one), "the K/V cache bytes differ from the one-request calls"

    assert out_multi.shape == (reqs * rows, HQ * D) and out_multi.dtype == torch.bfloat16
    assert torch.isfinite(out_multi.float()).all()
    assert torch.equal(out_multi, out_one), (
        f"output differs from the one-request calls: max abs diff "
        f"{(out_multi.float() - out_one.float()).abs().max().item()}"
    )
