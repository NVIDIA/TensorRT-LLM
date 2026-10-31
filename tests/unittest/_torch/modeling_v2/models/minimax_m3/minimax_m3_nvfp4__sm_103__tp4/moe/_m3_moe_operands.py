# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""One tp4 rank's MiniMax-M3 MoE layer operands for the decode-engine entries' tests.

The routed experts are random checkpoint-format NVFP4 tensors ([w3 (up); w1 (gate)] rows, e4m3 block scales) put
through TRT-LLM's own TRTLLM-Gen NVFP4 loader shuffle (``NVFP4TRTLLMGenFusedMoEMethod._shuffle_all_experts``), which
writes the buffers the engines read in place. The per-expert scalars follow the loader's conventions (act_beta and
act_clamp divided by fc31_alpha, fc31_scale_c = fc2_input_scale x fc31_alpha) and are sized so the FC1 sums land
around the SwiGLU-OAI clamp and the outputs near unit scale. The shared expert's MXFP8 weights come from
mxfp8_quantize with the swizzled scale layout, as the MXFP8 linear method stores them.
"""

import functools
from types import SimpleNamespace

import torch

H, I_TP, NUM_EXPERTS, TOP_K = 6144, 768, 128, 4
SHARED_I = 768
SWIGLU_ALPHA, SWIGLU_LIMIT = 1.702, 7.0
ROUTED_SCALING = 2.0
MINIMAX2_ROUTING = 5  # RoutingMethodType.MiniMax2
# NVFP4 global scales: 448 x 6 over the amax each side is sized for.
FC31_INPUT_SCALE = 448.0 * 6.0 / 8.0
FC2_INPUT_SCALE = 448.0 * 6.0 / 24.0


def _rand_e4m3_scales(shape, gen) -> torch.Tensor:
    """E4M3 block scales in [0.5, 2): biased exponents 6 and 7, any mantissa."""
    return torch.randint(0x30, 0x40, shape, dtype=torch.uint8, device="cuda", generator=gen).view(
        torch.float8_e4m3fn
    )


@functools.lru_cache(maxsize=None)
def experts(seed: int = 20261013):
    """The routed experts and the router of one MoE layer, in the layout and conventions of the TRTLLM-Gen loader.

    Returns a SimpleNamespace: ``tensors`` (the ten expert operands in ``M3MoeState.layer``'s order), ``bias``,
    ``router_weight``, ``fc31_input_scale`` and ``shared`` (gate_up, its scales, down, its scales).
    """
    from tensorrt_llm._torch.moe.fused_moe.quantization import NVFP4TRTLLMGenFusedMoEMethod

    gen = torch.Generator(device="cuda").manual_seed(seed)
    e = NUM_EXPERTS
    w31 = torch.randint(
        0, 256, (e, 2 * I_TP, H // 2), dtype=torch.uint8, device="cuda", generator=gen
    )
    w2 = torch.randint(0, 256, (e, H, I_TP // 2), dtype=torch.uint8, device="cuda", generator=gen)
    w31s = _rand_e4m3_scales((e, 2 * I_TP, H // 16), gen)
    w2s = _rand_e4m3_scales((e, H, I_TP // 16), gen)
    module = SimpleNamespace(
        w3_w1_weight=w31,
        w2_weight=w2,
        w3_w1_weight_scale=w31s,
        w2_weight_scale=w2s,
        is_gated_activation=True,
        bias=False,
    )
    NVFP4TRTLLMGenFusedMoEMethod()._shuffle_all_experts(module)
    torch.cuda.synchronize()

    # Raw FC1 sums: rms(x) x FC31_INPUT_SCALE x rms(E2M1 value) x rms(block scale) x sqrt(H) ~ 1.0e5 at rms(x) = 1;
    # fc31_alpha brings them to ~1.5 in the dequantized domain, the FC2 alpha the outputs to ~1.
    base = 1.5 / (FC31_INPUT_SCALE * 2.93 * 1.3 * H**0.5)
    fc31_alpha = base * (0.8 + 0.4 * torch.rand(e, device="cuda", generator=gen))
    fc2_alpha = (0.8 + 0.4 * torch.rand(e, device="cuda", generator=gen)) / (
        FC2_INPUT_SCALE * 2.93 * 1.3 * I_TP**0.5
    )
    tensors = (
        w31,
        w31s,
        w2,
        w2s,
        fc31_alpha,
        FC2_INPUT_SCALE * fc31_alpha,  # fc31_scale_c
        torch.full((e,), SWIGLU_ALPHA, device="cuda"),  # act_alpha
        1.0 / fc31_alpha,  # act_beta: the (up + 1) offset over fc31_alpha
        SWIGLU_LIMIT / fc31_alpha,  # act_clamp over fc31_alpha
        fc2_alpha,
    )
    bias = (0.02 * torch.randn(e, device="cuda", generator=gen)).float()
    router_weight = (torch.randn(e, H, device="cuda", generator=gen) / H**0.5).float().contiguous()
    fc31_input_scale = torch.tensor([FC31_INPUT_SCALE], dtype=torch.float32, device="cuda")

    gu = (0.02 * torch.randn(2 * SHARED_I, H, device="cuda", generator=gen)).to(torch.bfloat16)
    dn = (0.05 * torch.randn(H, SHARED_I, device="cuda", generator=gen)).to(torch.bfloat16)
    gu_q, gu_sf = torch.ops.trtllm.mxfp8_quantize(gu, True)
    dn_q, dn_sf = torch.ops.trtllm.mxfp8_quantize(dn, True)
    shared = (gu_q.view(2 * SHARED_I, H), gu_sf, dn_q.view(H, SHARED_I), dn_sf)
    return SimpleNamespace(
        tensors=tensors,
        bias=bias,
        router_weight=router_weight,
        fc31_input_scale=fc31_input_scale,
        shared=shared,
    )


# Routing cases: random (up to 4 M distinct experts), and every token on the same four experts (the bias outweighs
# any sigmoid difference), which leaves later slot groups empty.
CONCENTRATED = (5, 40, 77, 126)


def layer_for(state, case: str, shared: bool = True):
    """One layer on ``state`` over :func:`experts` for a routing case. ``shared=False``: the shared expert's weights
    are zero, so the engine returns the routed part alone (bf16(0 + bf16(routed)))."""
    ops = experts()
    bias = ops.bias.clone()
    if case == "concentrated":
        bias[list(CONCENTRATED)] += 1.0
    shared_weights = ops.shared
    if not shared:
        gu, gu_sf, dn, dn_sf = ops.shared
        shared_weights = (torch.zeros_like(gu), gu_sf, torch.zeros_like(dn), dn_sf)
    return state.layer(
        *ops.tensors,
        bias,
        router_weight=ops.router_weight,
        fc31_input_scale=ops.fc31_input_scale,
        shared_weights=shared_weights,
    )


def stock_routed(
    hidden: torch.Tensor, topk_ids: torch.Tensor, topk_weights: torch.Tensor
) -> torch.Tensor:
    """The stock TRTLLM-Gen NVFP4 MoE (fp4_quantize, then fp4_block_scale_moe_runner with SwiGLU-OAI) on the given
    routing, over :func:`experts`: what the built-in MiniMaxM3MoE computes for the routed part."""
    import tensorrt_llm._torch.custom_ops  # noqa: F401 -- registers the stock ops

    ops = experts()
    w31, w31s, w2, w2s, fc31_alpha, scale_c, act_alpha, act_beta, act_clamp, fc2_alpha = ops.tensors
    xq, xsf = torch.ops.trtllm.fp4_quantize(hidden, ops.fc31_input_scale, 16, False, False)
    out = torch.ops.trtllm.fp4_block_scale_moe_runner(
        None, None, xq, xsf.view(torch.float8_e4m3fn).flatten(),
        w31, w31s, None, act_alpha, act_beta, act_clamp, w2, w2s, None,
        scale_c, fc31_alpha, fc2_alpha,
        NUM_EXPERTS, TOP_K, None, None, I_TP, 0, NUM_EXPERTS, ROUTED_SCALING, MINIMAX2_ROUTING, True, 0,
        topk_weights=topk_weights.to(torch.bfloat16), topk_ids=topk_ids.to(torch.int32),
    )  # fmt: skip
    return out[0] if isinstance(out, (list, tuple)) else out


def hidden_states(m: int, seed: int) -> torch.Tensor:
    gen = torch.Generator(device="cuda").manual_seed(seed)
    return torch.randn(m, H, device="cuda", generator=gen).to(torch.bfloat16)


def check_routed_part(op, state, m: int, seed: int) -> None:
    """Exact, independent of the reference: the engine's routing (its route_dbg record: ids, then the bf16 weights'
    fp32 bits) is the reference routing bit for bit, and its routed part is the stock TRTLLM-Gen MoE's output on that
    routing bit for bit -- the claim the engines are built on."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_nvfp4__sm_103__tp4.ops.moe.m3_moe_engine import (  # noqa: E501
        moe_route,
    )

    layer = layer_for(state, "random", shared=False)
    hidden = hidden_states(m, seed=seed)
    out = op(hidden, layer)
    ids = state.route_dbg[:, :4].long()
    weights = state.route_dbg[:, 4:].contiguous().view(torch.float32)
    ref_ids, ref_weights = moe_route(hidden.float(), layer.router_weight, layer.bias)
    assert torch.equal(ids, ref_ids) and torch.equal(weights, ref_weights), (
        "routing differs from the reference"
    )
    stock = stock_routed(hidden, ids, weights)
    assert torch.equal(out.view(torch.int16), stock.view(torch.int16)), (
        "routed part differs from TRTLLM-Gen's"
    )
