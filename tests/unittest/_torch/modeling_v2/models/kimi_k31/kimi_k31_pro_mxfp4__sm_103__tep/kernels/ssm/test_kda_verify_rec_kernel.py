# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The family's ``kda_verify_rec`` (FP32 CUDA-core KDA verify) against an FP64 replay-semantics reference and the
3xTF32 tensor-core verify the general target runs today (``kda_mtp_decode_impl``), over chained rounds with mixed
acceptance, in the runtime layouts."""

import pytest
import torch

from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE

if not IS_CUTLASS_DSL_AVAILABLE:
    pytest.skip("kda_verify_rec is a CuTe DSL kernel", allow_module_level=True)
if not torch.cuda.is_available() or torch.cuda.get_device_capability() not in (
    (10, 0),
    (10, 3),
    (10, 7),
):
    pytest.skip("kda_verify_rec runs on SM100 / SM103 / SM107", allow_module_level=True)

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kimi_k31_pro_mxfp4__sm_103__tep.kernels.ssm import (  # noqa: E402
    kda_verify_rec as rec,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kimi_k31_pro_mxfp4__sm_103__tep.ops.ssm.kda_verify_rec import (  # noqa: E402, E501
    k31_kda_verify_rec as op,
)
from tensorrt_llm._torch.custom_ops.cute_dsl_kimi_k3_kda_mtp_ops import (  # noqa: E402
    kda_mtp_decode_impl,
)

__extra_import_path__ = ["../../../kernels/ssm"]
from _kda_rec_layer import K, RecLayer, V, W, check_against_reference, metadata  # noqa: E402


def _kernel_args(layer, slot_idx, num_acc):
    args = layer.args(slot_idx, num_acc)
    args.pop("out")
    return args


def _run_rec(layer, slot_idx, num_acc, **kw):
    return rec.kda_verify_rec(**_kernel_args(layer, slot_idx, num_acc), **kw)


def _run_tc(layer, slot_idx, num_acc):
    """The verify kernel the general target runs today (3xTF32 tensor cores, packed rows, fused norm)."""
    return kda_mtp_decode_impl(
        **_kernel_args(layer, slot_idx, num_acc),
        cu_seqlens=None,
        packed_token_layout=True,
        fuse_output_norm=True,
    )


@torch.no_grad()
@pytest.mark.parametrize(
    "H,num_spec,B,cluster,use_pdl",
    [
        (12, 7, 1, None, False),
        (12, 7, 4, None, True),
        (12, 7, 16, None, False),
        (12, 6, 4, 2, False),
        (12, 6, 4, 4, False),
        (24, 7, 8, None, False),
        (6, 5, 8, None, True),
        (6, 4, 64, None, False),
        (12, 2, 5, None, False),
        (8, 6, 2, 8, False),
    ],
)
def test_matches_reference_over_rounds(H, num_spec, B, cluster, use_pdl):
    """Three chained rounds (nothing pending, then random accepted counts) against the FP64 reference; slots
    outside the batch stay untouched."""
    slots = 2 * B + 3
    layer = RecLayer(H, B, num_spec, slots, seed=11 * B + H)
    for rnd in range(3):
        slot_idx, num_acc = metadata(B, slots, num_spec, rnd, seed=B)
        layer.new_tokens(seed=100 * rnd + B)
        before = layer.snapshot()
        ref = op.reference(**layer.args(slot_idx, num_acc))
        _run_rec(layer, slot_idx, num_acc, out=layer.out, cluster=cluster, use_pdl=use_pdl)
        torch.cuda.synchronize()
        check_against_reference(layer, slot_idx, ref)
        others = torch.ones(slots, dtype=torch.bool, device="cuda")
        others[slot_idx.long()] = False
        for t, b in zip(layer.pools(), before):
            assert torch.equal(t[others], b[others]), "a slot outside the batch changed"


@torch.no_grad()
@pytest.mark.parametrize("H,num_spec,B", [(12, 7, 4), (12, 7, 16), (6, 7, 24), (24, 4, 8)])
def test_against_the_current_verify_kernel(H, num_spec, B):
    """The conv windows / tails and the gate / value / beta caches are bitwise those of the 3xTF32 verify kernel,
    the k cache within one BF16 step, outputs to BF16 rounding and the committed state to FP32 rounding. The
    measured differences are printed (run with -s)."""
    slots = B + 5
    la = RecLayer(H, B, num_spec, slots, seed=5)
    lb = RecLayer(H, B, num_spec, slots, seed=5)
    worst = dict(out_max=0.0, out_rms=0.0, state_max=0.0, state_rel=0.0, out_ne=0.0)
    for rnd in range(3):
        slot_idx, num_acc = metadata(B, slots, num_spec, rnd, seed=3)
        la.new_tokens(seed=rnd)
        lb.new_tokens(seed=rnd)
        oa = _run_rec(la, slot_idx, num_acc)
        ob = _run_tc(lb, slot_idx, num_acc)
        torch.cuda.synchronize()
        s = slot_idx.long()
        for name, a, b in zip(
            ("g_cache", "v_cache", "beta", "cs_q", "cs_k", "cs_v"),
            [la.g_cache, la.v_cache, la.beta_buf, *la.cs],
            [lb.g_cache, lb.v_cache, lb.beta_buf, *lb.cs],
        ):
            assert torch.equal(a[s], b[s]), name
        torch.testing.assert_close(
            la.k_cache[s].float(), lb.k_cache[s].float(), atol=1e-5, rtol=8e-3
        )
        torch.testing.assert_close(oa.float(), ob.float(), atol=3e-2, rtol=2e-2)
        torch.testing.assert_close(la.state[s], lb.state[s], atol=1e-4, rtol=1e-4)
        d_out = (oa.float() - ob.float()).abs()
        d_st = (la.state[s] - lb.state[s]).abs()
        worst["out_max"] = max(worst["out_max"], d_out.max().item())
        worst["out_rms"] = max(
            worst["out_rms"], (d_out.pow(2).mean().sqrt() / ob.float().pow(2).mean().sqrt()).item()
        )
        worst["out_ne"] = max(worst["out_ne"], (d_out > 0).float().mean().item())
        worst["state_max"] = max(worst["state_max"], d_st.max().item())
        worst["state_rel"] = max(worst["state_rel"], (d_st.max() / lb.state[s].abs().max()).item())
        # Carry the same caches into the next round (the k cache may differ in a last BF16 bit).
        lb.restore(la.snapshot())
    print(
        f"\n[rec vs 3xTF32] H={H} num_spec={num_spec} B={B}: out max_abs={worst['out_max']:.3e} "
        f"rel_rms={worst['out_rms']:.3e} differing={worst['out_ne']:.2%}; state max_abs="
        f"{worst['state_max']:.3e} max_rel={worst['state_rel']:.3e}"
    )


@torch.no_grad()
def test_the_launcher_is_the_kernel():
    """``k31_kda_verify_rec`` (the launcher the op calls) writes exactly what the direct launch writes."""
    H, num_spec, B, slots = 12, 7, 4, 11
    la = RecLayer(H, B, num_spec, slots, seed=31)
    lb = RecLayer(H, B, num_spec, slots, seed=31)
    for rnd in range(2):
        slot_idx, num_acc = metadata(B, slots, num_spec, rnd, seed=7)
        la.new_tokens(seed=rnd)
        lb.new_tokens(seed=rnd)
        _run_rec(la, slot_idx, num_acc, out=la.out)
        rec.k31_kda_verify_rec(**lb.args(slot_idx, num_acc))
        torch.cuda.synchronize()
        assert torch.equal(la.out, lb.out)
        for a, b in zip(la.pools(), lb.pools()):
            assert torch.equal(a, b)


@torch.no_grad()
def test_cuda_graph_replay():
    """Captured once, replayed over rounds: the same results as eager calls."""
    H, num_spec, B, slots = 12, 7, 4, 9
    la = RecLayer(H, B, num_spec, slots, seed=21)
    lb = RecLayer(H, B, num_spec, slots, seed=21)
    slot_idx, num_acc = metadata(B, slots, num_spec, 1, seed=4)
    la.new_tokens(seed=0)
    lb.new_tokens(seed=0)
    snap = lb.snapshot()
    op(**lb.args(slot_idx, num_acc))  # compile outside the capture
    lb.restore(snap)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        op(**lb.args(slot_idx, num_acc))
    for _ in range(3):
        op(**la.args(slot_idx, num_acc))
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(la.out, lb.out)
        for a, b in zip(la.pools(), lb.pools()):
            assert torch.equal(a, b)


def _bits(t):
    return t.contiguous().view(torch.uint8)


def _round_metadata(B, slots, num_spec, rnd, seed):
    """Batch slots and accepted counts of one round: none pending in round 0, then every count from 0 to past
    ``num_spec`` (the kernel clamps it) across the requests and rounds."""
    slot_idx, num_acc = metadata(B, slots, num_spec, 0, seed=seed)
    if rnd:
        counts = (torch.arange(B) * 3 + 5 * rnd + seed) % (num_spec + 3)
        num_acc[slot_idx.long()] = counts.to(torch.int32).cuda()
    return slot_idx, num_acc


@torch.no_grad()
@pytest.mark.parametrize("use_pdl", [False, True])
@pytest.mark.parametrize("num_spec", [4, 7])
@pytest.mark.parametrize("H", [12, 6, 3])
@pytest.mark.parametrize("B", [1, 2, 3, 4, 5, 6, 8])
def test_specialized_kernel_is_bitwise_the_current_kernel(B, H, num_spec, use_pdl):
    """At the attention-TP head counts (TP8 / TP16 / TP32) and 1-8 requests of width 5 and 8, the output of
    ``kda_verify_rec_ws_kernel`` and every buffer it writes (state, replay caches, conv windows) are bitwise those of
    ``kda_verify_rec_kernel`` at the same cluster size, over chained rounds with mixed accepted counts."""
    C = rec.default_cluster(H, B, torch.cuda.get_device_properties(0).multi_processor_count)
    if not rec.use_specialized_kernel(C):
        pytest.skip(f"cluster {C}: kda_verify_rec_kernel runs this shape")
    slots = 2 * B + 3
    la = RecLayer(H, B, num_spec, slots, seed=13 * B + H + num_spec)
    lb = RecLayer(H, B, num_spec, slots, seed=13 * B + H + num_spec)
    for rnd in range(4):
        slot_idx, num_acc = _round_metadata(B, slots, num_spec, rnd, seed=B + H)
        la.new_tokens(seed=50 * rnd + B)
        lb.new_tokens(seed=50 * rnd + B)
        _run_rec(la, slot_idx, num_acc, out=la.out, use_pdl=use_pdl, specialized=False)
        _run_rec(lb, slot_idx, num_acc, out=lb.out, use_pdl=use_pdl)
        torch.cuda.synchronize()
        assert torch.equal(_bits(la.out), _bits(lb.out)), f"round {rnd}: output"
        for i, (a, b) in enumerate(zip(la.pools(), lb.pools())):
            assert torch.equal(_bits(a), _bits(b)), f"round {rnd}: written buffer {i}"


def test_supported():
    st = torch.zeros(4, 12, V, K, device="cuda")
    assert rec.supported(st, 7, K, V, W)
    assert not rec.supported(st.to(torch.bfloat16), 7, K, V, W)
    assert not rec.supported(st, 8, K, V, W)
    assert not rec.supported(st, 6, 64, V, W)
    assert not rec.supported(st, 6, K, V, 3)
    assert not rec.supported(st.cpu(), 6, K, V, W)
