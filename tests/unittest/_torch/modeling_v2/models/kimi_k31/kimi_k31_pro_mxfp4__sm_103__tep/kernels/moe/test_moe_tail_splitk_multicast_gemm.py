# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the fused latent-MoE tail's split-K up projection (``SplitKMulticastUpProjection``).

``out[t, c] = bf16(bf16(sum_k latent[t, k] * w[c, k]) + shared[t, c])`` (``-0`` stored as ``+0``) for this
rank's ``shard`` output columns, written at the rank's column offset of a ``[max_m, hidden]`` mailbox
(a plain device buffer here instead of the multicast mapping). With values whose products and partial
sums are exact in FP32 the result is checked bitwise against a torch reference, which pins the
cluster's split-K reduction and the epilogue; with random values every token tile of the same split-K
map must give the same bits, since the token tile changes only which CTA computes a column.
"""

import pytest
import torch

from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE

if not IS_CUTLASS_DSL_AVAILABLE:
    pytest.skip("the up projection is a CuTe DSL kernel", allow_module_level=True)
if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
    pytest.skip("the up projection runs on SM100 / SM103", allow_module_level=True)

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kimi_k31_pro_mxfp4__sm_103__tep.kernels.moe import (  # noqa: E402
    moe_tail_splitk_multicast_gemm as smg,
)

LATENT, HIDDEN, MAX_M = 3584, 7168, 128
BUCKETS = (8, 16, 32, 64, 128)
SENTINEL = -0.0


def _bucket(m: int) -> int:
    return next(b for b in BUCKETS if b >= m)


def _run(up, compiled, w, latent, shared, rank, shard):
    mailbox = torch.full((MAX_M, HIDDEN), SENTINEL, dtype=torch.bfloat16, device="cuda")
    up(
        compiled,
        w,
        latent,
        shared[:, rank * shard : (rank + 1) * shard],
        mailbox.data_ptr() + rank * shard * 2,
    )
    torch.cuda.synchronize()
    return mailbox


def _reference(w, latent, shared, rank, shard):
    m = latent.shape[0]
    up = (latent.double() @ w.double().t()).float().to(torch.bfloat16)
    out = (up.float() + shared[:m, rank * shard : (rank + 1) * shard].float()).to(torch.bfloat16)
    out = torch.where(out == 0, torch.zeros_like(out), out)
    mailbox = torch.full((MAX_M, HIDDEN), SENTINEL, dtype=torch.bfloat16, device="cuda")
    mailbox[:m, rank * shard : (rank + 1) * shard] = out
    return mailbox


@pytest.mark.parametrize("shard", [896, 448, 224])
@pytest.mark.parametrize("rows", [2, 8, 9, 16, 20, 32, 33, 48, 64, 65, 128])
def test_exact_values_bitwise(shard, rows):
    """Small integers times multiples of 1/16: every product and partial sum is exact in FP32, so any
    split-K order gives the reference's bits; a misplaced partial or row shows up as a wrong value."""
    if shard == 224 and rows > 32:
        pytest.skip("TP32 serves up to 32 rows")
    g = torch.Generator(device="cuda").manual_seed(rows * 1000 + shard)
    w = (torch.randint(-16, 17, (shard, LATENT), generator=g, device="cuda") / 16.0).to(
        torch.bfloat16
    )
    latent = torch.randint(-4, 5, (rows, LATENT), generator=g, device="cuda").to(torch.bfloat16)
    shared = (torch.randint(-64, 65, (MAX_M, HIDDEN), generator=g, device="cuda") / 4.0).to(
        torch.bfloat16
    )
    shared[0, :] = 0.0
    latent[0, :] = 0.0  # row 0: the GEMM is +0, the stored value must be +0, never the -0 marker
    up = smg.SplitKMulticastUpProjection(
        latent_dim=LATENT, hidden_dim=HIDDEN, shard_dim=shard, max_m=MAX_M
    )
    compiled = up.compile_for(_bucket(rows), torch.device("cuda", torch.cuda.current_device()))
    rank = 1
    got = _run(up, compiled, w, latent, shared, rank, shard)
    ref = _reference(w, latent, shared, rank, shard)
    assert torch.equal(got.view(torch.int16), ref.view(torch.int16))


def _alternative_tiles(tactic):
    """Tactics with the same split-K map and stages but another token tile."""
    alts = []
    for n in (8, 16, 32):
        if n == tactic.mma_n:
            continue
        alt = smg.SplitKTactic(tactic.mma_m, n, tactic.split_k, tactic.ab_stages)
        try:
            smg.validate_tactic(alt, 1, 1, LATENT)
        except ValueError:
            continue
        alts.append(alt)
    return alts


@pytest.mark.parametrize("rows", [9, 16, 17, 20, 30, 32, 33, 40, 48, 64])
def test_token_tile_does_not_change_bits(rows):
    """TP8 shard: the chosen tactic and every other token tile of its split-K map, random values,
    two input sets each, identical bits."""
    shard = 896
    dev = torch.device("cuda", torch.cuda.current_device())
    up = smg.SplitKMulticastUpProjection(
        latent_dim=LATENT, hidden_dim=HIDDEN, shard_dim=shard, max_m=MAX_M
    )
    chosen = smg.tactic_for_m(_bucket(rows), shard)
    for seed in range(2):
        g = torch.Generator(device="cuda").manual_seed(rows * 7 + seed)
        w = (torch.randn(shard, LATENT, generator=g, device="cuda") * LATENT**-0.5).to(
            torch.bfloat16
        )
        latent = torch.randn(rows, LATENT, generator=g, device="cuda").to(torch.bfloat16)
        shared = (torch.randn(MAX_M, HIDDEN, generator=g, device="cuda") * 0.25).to(torch.bfloat16)
        outs = []
        for tactic in [chosen] + _alternative_tiles(chosen):
            kernel = smg.SplitKDenseGemmKernel(
                tactic=tactic, use_pdl=True, has_bias=True, hidden_dim=HIDDEN
            )
            compiled = smg.cute_ext.compile(
                smg._launch,
                kernel,
                smg.from_dlpack(w, assumed_align=32),
                smg.from_dlpack(latent, assumed_align=32).mark_compact_shape_dynamic(
                    mode=0, stride_order=(0, 1)
                ),
                smg.from_dlpack(shared[:, :shard], assumed_align=32),
                smg.Int32(MAX_M),
                smg.cutlass.Int64(0),
                HIDDEN,
                shard,
                MAX_M,
                smg._cuda.CUstream(torch.cuda.current_stream(dev).cuda_stream),
            )
            outs.append(_run(up, compiled, w, latent, shared, 3, shard))
        for other in outs[1:]:
            assert torch.equal(outs[0].view(torch.int16), other.view(torch.int16))
