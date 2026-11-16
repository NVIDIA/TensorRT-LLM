# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test of the one K3KdaBuffers set per device that this target passes to both KDA ops it launches on it.

The target's plain decode runs the family's ``ssm/k3_kda_decode_attn`` and its speculative verify runs its own
``ssm/k3_kda_attn``, every KDA layer on one caller-owned set. Plain-decode launches on the pools of a real
``KimiK3HybridCacheManager`` (``_kda_cells.build_manager``) and verify launches on the pools of one built with
MTP-style speculation of 7 drafts, interleaved in stream order on one set, against the same launches on sets of
their own.

Every launch here follows a plain copy kernel (its input), as each KDA launch in the model follows other layers'
kernels: the head CTAs read the slots' pools before their grid-dependency wait, so a launch must not directly follow
another launch on the same pools.
"""

import importlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.ops.ssm.k3_kda_buffers import (
    K3KdaBuffers,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.ops.ssm.k3_kda_decode_attn import (
    k3_kda_decode_attn,
)

__extra_import_path__ = ["../ops/ssm"]
import _kda_cells as kc

k3_kda_attn = importlib.import_module(
    "tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl."
    "kimi_k3_mxfp4__sm_100__tp16_moetp4ep4.ops.ssm.k3_kda_attn"
).k3_kda_attn

pytestmark = pytest.mark.skipif(not kc.sm100(), reason="needs SM100 (tcgen05, TMA, clusters)")

LAYERS = 3


@pytest.fixture(scope="module")
def mgr():
    kc.load_ops()
    m = kc.build_manager(LAYERS)
    try:
        yield m
    finally:
        m.shutdown()


@pytest.fixture(scope="module")
def slots(mgr):
    return kc.request_slots(mgr, 8, first_id=200)


@pytest.fixture(scope="module")
def spec_mgr():
    kc.load_ops()
    m = kc.build_manager(1, num_spec=kc.NUM_SPEC)
    try:
        yield m
    finally:
        m.shutdown()


def _launch(wt, pools, x_src, slots, bufs) -> torch.Tensor:
    """One plain-decode launch, after a plain copy kernel (the input)."""
    x = x_src.clone()
    return k3_kda_decode_attn(
        x, wt["w"], wt["w_fb"], wt["w_q"], wt["w_k"], wt["w_v"], wt["a_log"], wt["dt_bias"], wt["onorm_w"],
        pools["conv"], pools["ssm"], slots, bufs, kc.LOWER_BOUND, kc.SCALE, kc.EPS,
    )  # fmt: skip


def _x(num: int, gen: torch.Generator) -> torch.Tensor:
    return torch.randn(num, kc.K_IN, generator=gen, device="cuda").bfloat16()


def test_shared_with_k3_kda_attn(mgr, slots, spec_mgr) -> None:
    """Plain-decode launches and ``ssm/k3_kda_attn`` verify launches interleaved in stream order on one set (the
    model's single set per device) give the bits of the same launches on sets of their own, one at a time. Both run
    the same stream role and move every CTA's index once per launch, so either may follow the other."""
    vslot = kc.request_slots(spec_mgr, 1, first_id=300)
    dpools = kc.layer_pools(mgr, 0)
    vpools = kc.layer_pools(spec_mgr, 0)
    with torch.inference_mode():
        kc.fill_pools(dpools, 900)
        kc.fill_pools(vpools, 901)
        own_d, own_v = kc.clone_pools(dpools), kc.clone_pools(vpools)
        wt = kc.make_weights(130)
        gen = torch.Generator(device="cuda").manual_seed(43)
        calls = []
        for num in (1, 4, 8, 3):
            calls.append(("decode", _x(num, gen), slots[:num]))
            calls.append(("verify", _x(kc.NT, gen), vslot))

        def verify(pools, x, slot, bufs):
            return k3_kda_attn(
                x.clone(), wt["w"], wt["w_fb"], wt["w_q"], wt["w_k"], wt["w_v"], wt["a_log"], wt["dt_bias"],
                wt["onorm_w"], pools["cs_q"], pools["cs_k"], pools["cs_v"], pools["ssm"], pools["state_tok"], slot,
                pools["pending"], bufs, kc.NUM_SPEC, kc.LOWER_BOUND, kc.SCALE, kc.EPS,
            )  # fmt: skip

        own_bufs = {"decode": K3KdaBuffers.create("cuda"), "verify": K3KdaBuffers.create("cuda")}
        want = []
        for kind, x, s in calls:
            if kind == "decode":
                want.append(_launch(wt, own_d, x, s, own_bufs[kind]).clone())
            else:
                want.append(verify(own_v, x, s, own_bufs[kind]).clone())
            torch.cuda.synchronize()
        shared = K3KdaBuffers.create("cuda")
        got = [
            (
                _launch(wt, dpools, x, s, shared)
                if kind == "decode"
                else verify(vpools, x, s, shared)
            ).clone()
            for kind, x, s in calls
        ]
        torch.cuda.synchronize()
    assert all(torch.equal(a, b) for a, b in zip(got, want))
    assert all(torch.equal(dpools[n], own_d[n]) for n in ("conv", "ssm"))
    assert all(
        torch.equal(vpools[n], own_v[n]) for n in ("cs_q", "cs_k", "cs_v", "ssm", "state_tok")
    )
    assert bool((shared.epoch == len(calls) % 3).all()), shared.epoch.unique().tolist()
