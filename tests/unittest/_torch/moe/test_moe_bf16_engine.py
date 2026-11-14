# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The BF16 decode MoE engine (kernels/blackwell/moe_bf16_engine.py) against trtllm::bf16_moe_runner routed from
the router logits, un-finalized: per (token, k) the same pick and gemm2 row bit for bit and the BF16 top-k weight
within 1 ulp (the engine's softmax is ex2.approx / rcp.approx, the runner's expf / IEEE divide), through each op's own
expanded -> permuted index (the engine packs the local picks' rows), at the Qwen3.5 MTP draft layer's shapes (512
experts, top-10, hidden 4096, intermediate 512 per rank, 128 / 64 local experts) and at 256 experts, top-8."""

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._utils import is_blackwell_datacenter

pytestmark = pytest.mark.skipif(not is_blackwell_datacenter(), reason="needs SM 10.0 / 10.3")

H, E, TOPK, INTER = 4096, 512, 10, 512


def _assert_within_one_ulp(got: torch.Tensor, ref: torch.Tensor) -> None:
    """Positive BF16 weights at most one ulp apart (as int16 bit patterns, adjacent values differ by 1)."""
    assert (got > 0).all() and (ref > 0).all()
    ulps = (got.view(torch.int16).int() - ref.view(torch.int16).int()).abs()
    assert int(ulps.max()) <= 1, f"top-k weights {int(ulps.max())} ulp apart"


def _gen(seed):
    return torch.Generator(device="cuda").manual_seed(seed)


def _experts(e_local, seed):
    """BF16 experts in the trtllm-gen BlockMajorK layout both kernels read."""
    g = _gen(seed)
    w31 = (torch.randn((e_local, H // 64, 2 * INTER, 64), generator=g, device="cuda") * 0.02).to(
        torch.bfloat16
    )
    w2 = (torch.randn((e_local, INTER // 64, H, 64), generator=g, device="cuda") * 0.02).to(
        torch.bfloat16
    )
    return w31, w2


def _inputs(m, num_experts, local_start, e_local, seed):
    g = _gen(seed)
    x = torch.randn(m, H, generator=g, device="cuda").to(torch.bfloat16)
    lg = torch.randn(m, num_experts, generator=g, device="cuda")
    lg[:, local_start : local_start + e_local] += 1.0  # most picks local
    return x, lg.to(torch.bfloat16)


def _reference(x, lg, w31, w2, top_k, local_start, e_local):
    from tensorrt_llm._torch.moe.fused_moe.moe_op_backend import TRTLLMOpBackend
    from tensorrt_llm._torch.moe.fused_moe.routing import RoutingMethodType
    from tensorrt_llm._torch.utils import ActType_TrtllmGen

    return TRTLLMOpBackend().run_bf16_moe(
        lg,
        None,
        x,
        w31,
        w2,
        lg.shape[1],
        top_k,
        None,
        None,
        INTER,
        local_start,
        e_local,
        None,
        RoutingMethodType.Renormalize,
        gated_act_type=int(ActType_TrtllmGen.SwiGlu),
        do_finalize=False,
    )


# 512 / 256 local experts: the draft experts with no or 2-way expert parallelism.
@pytest.mark.parametrize(
    "num_experts,top_k,e_local",
    [(E, TOPK, 512), (E, TOPK, 256), (E, TOPK, 128), (E, TOPK, 64), (256, 8, 64)],
)
@pytest.mark.parametrize("m", [1, 2, 7, 14, 16, 17, 32])
def test_engine_matches_bf16_moe_runner(m, num_experts, top_k, e_local):
    from tensorrt_llm._torch.kernels.blackwell import moe_bf16_engine as eng

    local_start = 64
    w31, w2 = _experts(e_local, 9)
    state = eng.new_state(torch.device("cuda"))
    for seed in (51 + m, 53 + m):  # consecutive calls on one state (both counter parities)
        x, lg = _inputs(m, num_experts, local_start, e_local, seed)
        h = torch.empty((m * top_k, INTER), dtype=torch.bfloat16, device="cuda")
        out = torch.full((m * top_k + 4, H), -1, dtype=torch.int16, device="cuda")
        wts = torch.empty((m, top_k), dtype=torch.bfloat16, device="cuda")
        e2p = torch.empty((m, top_k), dtype=torch.int32, device="cuda")
        eng.moe_bf16_engine(
            x,
            lg,
            w31,
            w2,
            h,
            out[: m * top_k].view(torch.bfloat16),
            wts,
            e2p,
            state,
            top_k=top_k,
            local_start=local_start,
        )
        ref = _reference(x, lg, w31, w2, top_k, local_start, e_local)
        torch.cuda.synchronize()
        r_rows, r_wts, r_e2p = ref[0], ref[1], ref[2].reshape(m, top_k)
        _assert_within_one_ulp(wts, r_wts)
        valid = e2p >= 0
        assert torch.equal(valid, r_e2p >= 0), "local picks"
        rows = torch.sort(e2p[valid]).values
        assert torch.equal(
            rows, torch.arange(rows.numel(), dtype=rows.dtype, device=rows.device)
        ), "one packed row per local pick"
        assert int(valid.sum()) > 0
        got = out[e2p[valid].long()]
        assert torch.equal(got, r_rows[r_e2p[valid].long()].view(torch.int16)), "gemm2 rows"
        assert (out[m * top_k :] == -1).all(), "wrote past the pair rows"


def test_engine_takes_the_layer_shapes():
    from tensorrt_llm._torch.kernels.blackwell.moe_bf16_engine import (
        supports_local_experts,
        supports_routing,
    )

    assert supports_routing(512, 10) and supports_routing(256, 8) and supports_routing(512, 16)
    assert (
        not supports_routing(512, 17)
        and not supports_routing(128, 4)
        and not supports_routing(1024, 8)
    )
    assert (
        supports_local_experts(E, 128)
        and supports_local_experts(E, 64)
        and supports_local_experts(E, 32)
    )
    assert not supports_local_experts(E, 48) and not supports_local_experts(256, 512)


def _leaf_and_inputs(m, e_local, local_start, seed, **overrides):
    """A BF16 leaf's attributes that ``decode_moe_bf16`` reads, and its kernel inputs."""
    from tensorrt_llm._torch.moe.fused_moe.quantization import (
        TRTLLM_GEN_WEIGHT_LAYOUT_BLOCK_MAJOR_K,
    )
    from tensorrt_llm._torch.moe.fused_moe.routing import RoutingMethodType
    from tensorrt_llm._torch.utils import ActivationType

    w31, w2 = _experts(e_local, 9)
    x, lg = _inputs(m, E, local_start, e_local, seed)
    leaf = dict(
        activation_type=ActivationType.Swiglu,
        bias=False,
        act_alpha=None,
        act_beta=None,
        act_clamp=None,
        routing_method=SimpleNamespace(routing_method_type=RoutingMethodType.Renormalize),
        num_slots=E,
        expert_size_per_partition=e_local,
        slot_start=local_start,
        quant_method=SimpleNamespace(
            use_shuffled_weight=True, weight_layout=TRTLLM_GEN_WEIGHT_LAYOUT_BLOCK_MAJOR_K
        ),
        w3_w1_weight=w31,
        w2_weight=w2,
    )
    k = dict(do_finalize=False, router_logits=lg, routing_bias=None, top_k=TOPK, x=x)
    for name, value in overrides.items():
        (leaf if name in leaf else k)[name] = value
    return SimpleNamespace(**leaf), SimpleNamespace(**k)


def test_decode_moe_bf16_applies_to_the_engine_shapes():
    """``decode_moe_bf16.applies``: an un-finalized logits-routed Renormalize SwiGLU run of 1..M_MAX rows on the
    BlockMajorK weights; any other run keeps bf16_moe_runner."""
    from tensorrt_llm._torch.kernels.blackwell.moe_bf16_engine import M_MAX
    from tensorrt_llm._torch.moe.fused_moe.routing import RoutingMethodType
    from tensorrt_llm._torch.moe.fused_moe.trtllm_gen import decode_moe_bf16
    from tensorrt_llm._torch.utils import ActivationType

    assert decode_moe_bf16.applies(*_leaf_and_inputs(7, 128, 64, 1))
    assert decode_moe_bf16.applies(*_leaf_and_inputs(M_MAX, 64, 0, 1))
    assert not decode_moe_bf16.applies(*_leaf_and_inputs(M_MAX + 1, 128, 64, 1))
    x, lg = _inputs(7, E, 64, 128, 1)
    # 16-byte row strides on an 8-byte-offset start: the kernel's 16-byte loads would be misaligned.
    hidden = x.shape[1]
    x_pad = torch.zeros((7, hidden + 8), dtype=x.dtype, device=x.device)
    x_pad[:, 4 : 4 + hidden] = x
    lg_pad = torch.zeros(lg.numel() + 4, dtype=lg.dtype, device=lg.device)
    lg_pad[4:] = lg.reshape(-1)
    for overrides in (
        dict(x=x_pad[:, 4 : 4 + hidden]),
        dict(router_logits=lg_pad[4:].view(lg.shape)),
        dict(do_finalize=True),
        dict(router_logits=None),
        dict(routing_bias=torch.zeros(E, device="cuda")),
        dict(activation_type=ActivationType.Geglu),
        dict(act_clamp=torch.ones(128, device="cuda")),
        dict(routing_method=SimpleNamespace(routing_method_type=RoutingMethodType.DeepSeekV3)),
        dict(quant_method=SimpleNamespace(use_shuffled_weight=False, weight_layout=None)),
        dict(router_logits=lg.t().contiguous().t()),
        dict(router_logits=lg.float()),
        dict(expert_size_per_partition=48),
    ):
        assert not decode_moe_bf16.applies(*_leaf_and_inputs(7, 128, 64, 1, **overrides)), overrides


@pytest.mark.parametrize("m", [1, 7, 32])
def test_decode_moe_bf16_run_matches_bf16_moe_runner(m):
    """``decode_moe_bf16.run``: the un-finalized (rows, top-k weights, expanded -> permuted index) of bf16_moe_runner,
    per (token, k)."""
    from tensorrt_llm._torch.kernels.blackwell import moe_bf16_engine as eng
    from tensorrt_llm._torch.moe.fused_moe.trtllm_gen import decode_moe_bf16

    e_local, local_start = 128, 64
    leaf, k = _leaf_and_inputs(m, e_local, local_start, 70 + m)
    assert decode_moe_bf16.applies(leaf, k)
    rows, wts, e2p = decode_moe_bf16.run(leaf, k, eng.new_state(torch.device("cuda")))
    ref = _reference(
        k.x, k.router_logits, leaf.w3_w1_weight, leaf.w2_weight, TOPK, local_start, e_local
    )
    torch.cuda.synchronize()
    r_rows, r_wts, r_e2p = ref[0], ref[1], ref[2].reshape(m, TOPK)
    _assert_within_one_ulp(wts, r_wts)
    valid = e2p >= 0
    assert torch.equal(valid, r_e2p >= 0) and int(valid.sum()) > 0
    assert torch.equal(
        rows[e2p[valid].long()].view(torch.int16),
        r_rows[r_e2p[valid].long()].view(torch.int16),
    )


def test_leaf_returns_the_index_tokens_by_top_k(monkeypatch):
    """The BF16 leaf's un-finalized output has the expanded -> permuted index [tokens, top_k], like the top-k
    weights."""
    from tensorrt_llm._torch.moe.fused_moe.moe_op_backend import TRTLLMOpBackend
    from tensorrt_llm._torch.moe.fused_moe.trtllm_gen import trtllm_bf16

    m, e_local, local_start = 7, 128, 64
    leaf, k = _leaf_and_inputs(m, e_local, local_start, 90)
    k = SimpleNamespace(
        **vars(k),
        n_group=None,
        topk_group=None,
        routed_scaling_factor=None,
        token_final_scales=None,
        token_selected_experts=None,
        moe_output=None,
    )
    monkeypatch.setattr(trtllm_bf16, "prepare_kernel_inputs", lambda impl, ctx: k)
    impl = SimpleNamespace(
        **vars(leaf),
        _decode_engine_state=None,
        op_backend=TRTLLMOpBackend(),
        intermediate_size_per_partition=INTER,
        max_num_tokens=8192,
        use_dp=False,
        _unfinalized=lambda outputs: outputs,
    )
    rows, wts, e2p = trtllm_bf16.TrtllmTrtllmGenBf16Impl.run_moe(impl, None)
    ref = _reference(
        k.x, k.router_logits, leaf.w3_w1_weight, leaf.w2_weight, TOPK, local_start, e_local
    )
    torch.cuda.synchronize()
    assert wts.shape == (m, TOPK) and e2p.shape == (m, TOPK)
    assert torch.equal(e2p, ref[2])
