# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The routed experts of the Qwen3.8-Flash-Next target, from checkpoint-format tensors to output.

The CuteDSL op tests build their operands with layout code of their own. This file builds
the checkpoint's expert tensors (512 experts: packed NVFP4 `up_proj` / `gate_proj` /
`down_proj` codes, linear e4m3 block scales, one `weight_scale_2` per expert, per-expert
`input_scale`s) and runs them through the target's own load-time transforms
(`weights._expert_fc1_w` / `_expert_fc1_s` / `_expert_fc2_w` / `_expert_fc2_s`) and scalar
derivation (`modeling._moe_layer_scalars`), once per routed backend. The operands then go
through the call sequence of the target's `_moe_block` -- CuteDSL: sort, FC1 with SwiGLU,
FC2, unpermute; trtllm-gen: the one-call block-scale runner -- with explicit routing.

Two routings hit all 512 experts. In the first every token's only non-zero routing weight is
on its own expert, so each expert's output is checked alone: a layout error confined to one
expert (a permutation, a scale swizzle, an alpha) cannot hide in a sum over ten. On it both
backends are held to a float64 reference over the dequantized checkpoint tensors, with
`down_proj.input_scale` spread across the experts as a calibrated checkpoint's is. The second
is a weighted top-10: with a uniform `down_proj.input_scale` the two backends requantize the
SwiGLU output identically, so on it they are held to each other -- the area's one
cross-backend check.

The reference does not model the kernels' accumulation. It models the one rounding step
that is not exact: the SwiGLU output is requantized to NVFP4 before FC2, at the backend's
own global scale (the layer's single `global_sf` on CuteDSL, each expert's
`1 / down_proj.input_scale` on trtllm-gen).
"""

import functools
from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog.moe.fp4_block_scale_moe_runner import (
    fp4_block_scale_moe_runner,
    quantize_nvfp4,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.quantization.fp4_quantize import (
    fp4_quantize,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_103__tp1 import (  # noqa: E501
    modeling as M,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_103__tp1 import (  # noqa: E501
    weights as W,
)

assert torch.cuda.is_available(), "the CuteDSL experts require a CUDA device"

DEV = "cuda"
E, K, H, INTER, VEC = 512, 10, 2560, 640, 16
#: The activation's input scale, one number for all experts: |x| = 5 maps to the e4m3 maximum.
ISC1 = 5.0 / (448 * 6)
#: The SwiGLU output maximum `down_proj.input_scale` is calibrated to. With `spread` each
#: expert's own value ranges over 2^-1/2 .. 2^1/2 of it, so the layer's `max` is not every
#: expert's.
ISC2 = 64.0 / (448 * 6)
E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
#: The routed runners read activation scales as a linear buffer, not the 128x4-swizzled
#: one of a dense NVFP4 GEMM; byte counts match at multiples of 128 rows, so the wrong
#: one passes silently.
SF_LINEAR = False

#: Tokens of the two routings: one per expert, and a weighted top-10.
ONE_HOT_TOKENS = E
WEIGHTED_TOKENS = 1024

BACKENDS = ("CUTEDSL", "TRTLLM")
#: The float64 reference's gate, relative per token. Both backends sit at 2.5e-3 or below:
#: the bf16 rounding of the output is 1e-3 of it, the rest is the reference's fp32 / tie
#: differences. A wrong layout is ~1; a wrong per-expert scalar is its own relative error.
REFERENCE_TOLERANCE = 5e-3
#: The same, between the backends under a uniform `down_proj.input_scale`. Each backend is
#: within `REFERENCE_TOLERANCE` of the reference, so the two are within twice that of each
#: other. Both requantize the SwiGLU output to NVFP4, and an fp32 difference in the last
#: place moves an element at an e2m1 rounding boundary by a whole code: the token holding
#: it differs by ~5e-3 while the mean stays near 5e-6. A wrong layout is still ~1.
BACKEND_TOLERANCE = 2 * REFERENCE_TOLERANCE


# ── the checkpoint ────────────────────────────────────────────────────────


def _quantize(dense: torch.Tensor):
    """A bf16 `[R, C]` matrix as the checkpoint stores it: packed e2m1 codes `[R, C/2]`
    uint8, e4m3 block scales `[R, C/16]`, and the `weight_scale_2` that undoes the global
    scale the block scales were derived under."""
    rows, cols = dense.shape
    gs = (448 * 6 / dense.abs().amax().float()).reshape(1)
    codes, sf = torch.ops.trtllm.fp4_quantize(dense, gs, VEC, False, False)
    return (
        codes.view(torch.uint8).view(rows, cols // 2),
        sf.view(torch.uint8).view(rows, cols // VEC).view(torch.float8_e4m3fn),
        (1.0 / gs).squeeze(0),
    )


@functools.lru_cache(maxsize=None)
def _checkpoint(spread: bool) -> SimpleNamespace:
    """All 512 experts of one layer, with the checkpoint's tensor names and dtypes.

    `up_proj` and `gate_proj` are quantized together, so they share one `weight_scale_2`, as
    the fused FC1 requires. Expert magnitudes spread over 2^-1 .. 2^1/2 (FC1) and 2^-1 .. 2^1
    (FC2), so a wrong per-expert scalar shows."""
    g = torch.Generator(device=DEV).manual_seed(20261014)
    spread1 = torch.exp2(torch.rand(E, generator=g, device=DEV) * 1.5 - 1)
    spread2 = torch.exp2(torch.rand(E, generator=g, device=DEV) * 2 - 1)
    fields = {k: [] for k in ("up_w", "up_s", "gate_w", "gate_s", "down_w", "down_s")}
    ws2_gu, ws2_dn = [], []
    for e in range(E):
        gate_up = torch.randn(2 * INTER, H, generator=g, device=DEV) * (spread1[e] / H**0.5)
        down = torch.randn(H, INTER, generator=g, device=DEV) * (spread2[e] / INTER**0.5)
        codes, sf, ws2 = _quantize(gate_up.to(torch.bfloat16))
        fields["up_w"].append(codes[:INTER])
        fields["gate_w"].append(codes[INTER:])
        fields["up_s"].append(sf[:INTER])
        fields["gate_s"].append(sf[INTER:])
        ws2_gu.append(ws2)
        codes, sf, ws2 = _quantize(down.to(torch.bfloat16))
        fields["down_w"].append(codes)
        fields["down_s"].append(sf)
        ws2_dn.append(ws2)
    ws2_gu, ws2_dn = torch.stack(ws2_gu), torch.stack(ws2_dn)
    isc1 = torch.full((E,), ISC1, dtype=torch.float32, device=DEV)
    isc2 = torch.full((E,), ISC2, dtype=torch.float32, device=DEV)
    if spread:
        isc2 = isc2 * torch.exp2(torch.rand(E, generator=g, device=DEV) - 0.5)
    return SimpleNamespace(
        **fields,
        ws2_gu=ws2_gu,
        ws2_dn=ws2_dn,
        isc2=isc2,
        # The rows `modeling._moe_layer_scalars` reads, under layer 0.
        scalars={
            "l0_e_isc1": isc1,
            "l0_e_isc1_up": isc1.clone(),
            "l0_e_ws2_1": ws2_gu,
            "l0_e_ws2_1_up": ws2_gu.clone(),
            "l0_e_isc2": isc2,
            "l0_e_ws2_2": ws2_dn,
        },
    )


# ── what the target builds from it ────────────────────────────────────────


@functools.lru_cache(maxsize=None)
def _layer(backend: str, spread: bool) -> SimpleNamespace:
    """The layer's expert operands for `backend`, built as the weight table builds them:
    one transform call per expert and tensor, stacked, then viewed as `_layer_views`
    views them."""
    ck = _checkpoint(spread)
    core = SimpleNamespace(trunk_moe_backend=backend)

    def stack(fn, *tensors):
        return torch.stack([fn(tuple(t[e] for t in tensors), core) for e in range(E)])

    def stack1(fn, tensor):
        return torch.stack([fn(tensor[e], core) for e in range(E)])

    fc1_w = stack(W._expert_fc1_w, ck.up_w, ck.gate_w)
    fc1_s = stack(W._expert_fc1_s, ck.up_s, ck.gate_s)
    fc2_w = stack1(W._expert_fc2_w, ck.down_w)
    fc2_s = stack1(W._expert_fc2_s, ck.down_s)
    if backend == "CUTEDSL":
        w_dtype, s_dtype = torch.float4_e2m1fn_x2, torch.uint8
    else:
        w_dtype, s_dtype = torch.uint8, torch.float8_e4m3fn
    return SimpleNamespace(
        fc1_w=fc1_w.view(w_dtype),
        fc1_s=fc1_s.view(s_dtype),
        fc2_w=fc2_w.view(w_dtype),
        fc2_s=fc2_s.view(s_dtype),
        scalars=M._moe_layer_scalars(ck.scalars, 0, backend),
    )


def _cutedsl_routed(layer, x, ids, weights) -> torch.Tensor:
    """The CuteDSL branch of `_moe_block`, from the quantization to the unpermute."""
    sc = layer.scalars
    tokens = x.shape[0]
    xq, xsf = fp4_quantize(x, sc["global_scale"], M._SF_VEC, False, SF_LINEAR)
    tile_expert, tile_limit, expanded_to_permuted, permuted_to_expanded, _, live_tiles = M.moe_sort(
        token_selected_experts=ids,
        token_final_scales=weights,
        num_experts=E,
        top_k=K,
        local_expert_offset=0,
        local_num_experts=E,
        tile_tokens_dim=M._CUTEDSL_TILE_SIZE,
    )
    h, h_sf = M.cute_dsl_nvfp4_gather_grouped_gemm_act_fusion(
        input=xq.view(torch.float4_e2m1fn_x2),
        weight=layer.fc1_w,
        input_scale=xsf.view(tokens, H // M._SF_VEC),
        weight_scale=layer.fc1_s,
        alpha=sc["alpha1"],
        tile_idx_to_group_idx=tile_expert,
        tile_idx_to_mn_limit=tile_limit,
        permuted_idx_to_expanded_idx=permuted_to_expanded,
        num_non_exiting_tiles=live_tiles,
        global_sf=sc["global_sf"],
        num_experts=E,
        top_k=K,
        num_local_experts=E,
        local_expert_offset=0,
        tile_size=M._CUTEDSL_TILE_SIZE,
        scaling_vector_size=M._SF_VEC,
        activation_type=int(M.ActivationType.Swiglu),
    )
    y = M.cute_dsl_nvfp4_grouped_gemm(
        input=h,
        weight=layer.fc2_w,
        input_scale=h_sf,
        weight_scale=layer.fc2_s,
        alpha=sc["alpha2"],
        tile_idx_to_group_idx=tile_expert,
        num_non_exiting_tiles=live_tiles,
        num_experts=E,
        top_k=K,
        num_local_experts=E,
        local_expert_offset=0,
        tile_size=M._CUTEDSL_TILE_SIZE,
        output_dtype=torch.bfloat16,
        scaling_vector_size=M._SF_VEC,
    )
    return M.moe_unpermute_inplace(
        permuted_input=y,
        output=torch.empty(tokens, H, dtype=torch.bfloat16, device=x.device),
        expanded_idx_to_permuted_idx=expanded_to_permuted,
        topk_scales=weights,
    )


def _trtllm_gen_routed(layer, x, ids, weights) -> torch.Tensor:
    """The trtllm-gen branch of `_moe_block`."""
    sc = layer.scalars
    xq, xsf = fp4_quantize(x, sc["global_scale"], M._SF_VEC, False, SF_LINEAR)
    return fp4_block_scale_moe_runner(
        routing_logits=None,
        routing_bias=None,
        hidden_states=xq,
        hidden_states_scale=xsf.view(torch.float8_e4m3fn),
        gemm1_weights=layer.fc1_w,
        gemm1_weights_scale=layer.fc1_s,
        gemm1_bias=None,
        gemm1_alpha=None,
        gemm1_beta=None,
        gemm1_clamp_limit=None,
        gemm2_weights=layer.fc2_w,
        gemm2_weights_scale=layer.fc2_s,
        gemm2_bias=None,
        output1_scale_scalar=sc["output1_scale_scalar"],
        output1_scale_gate_scalar=sc["output1_scale_gate_scalar"],
        output2_scale_scalar=sc["output2_scale_scalar"],
        num_experts=E,
        top_k=K,
        n_group=None,
        topk_group=None,
        intermediate_size=INTER,
        local_expert_offset=0,
        local_num_experts=E,
        routed_scaling_factor=None,
        routing_method_type=M._ROUTING_METHOD_RENORM,
        do_finalize=True,
        act_type=M._ACT_TYPE_SWIGLU,
        topk_weights=weights.to(torch.bfloat16),
        topk_ids=ids,
    )[0]


# ── the inputs and the reference ──────────────────────────────────────────


@functools.lru_cache(maxsize=None)
def _problem(case: str):
    """`(x, ids, weights)`: bf16 activations, expert ids `[T, K]` int32 and routing weights
    `[T, K]` fp32 (exact in bf16, which is the dtype trtllm-gen takes them in)."""
    tokens = ONE_HOT_TOKENS if case == "one_hot" else WEIGHTED_TOKENS
    g = torch.Generator(device=DEV).manual_seed(7 if case == "one_hot" else 8)
    x = torch.randn(tokens, H, generator=g, device=DEV).to(torch.bfloat16)
    scores = torch.rand(tokens, E, generator=g, device=DEV)
    if case == "one_hot":
        # The forced expert has the top score, so it sits in slot 0.
        scores[torch.arange(tokens), torch.arange(tokens) % E] = 2.0
        weights = torch.zeros(tokens, K, device=DEV)
        weights[:, 0] = 1.0
    else:
        weights = torch.softmax(torch.randn(tokens, K, generator=g, device=DEV), dim=-1)
        weights = weights.to(torch.bfloat16).float()
    ids = scores.topk(K, dim=-1).indices.to(torch.int32)
    return x, ids, weights


def _dequant(codes: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """`[R, C/2]` packed e2m1 (element 2j in the low nibble) and `[R, C/16]` e4m3 scales
    -> float64 `[R, C]`."""
    magnitude = torch.tensor(E2M1, dtype=torch.float64, device=codes.device)
    table = torch.cat([magnitude, -magnitude])
    lo, hi = table[(codes & 0xF).long()], table[(codes >> 4).long()]
    values = torch.stack([lo, hi], dim=-1).reshape(codes.shape[0], -1)
    return values * scales.view(torch.float8_e4m3fn).double().repeat_interleave(VEC, dim=1)


def _reference(backend: str, case: str, spread: bool) -> torch.Tensor:
    """The routed output in float64, `[T, H]`.

    The activation is what the kernels see, the dequantized NVFP4 of the target's own
    quantization. Each expert's weights are dequantized from the checkpoint tensors, times
    its `weight_scale_2`. The SwiGLU output is requantized at `backend`'s global scale."""
    ck, layer = _checkpoint(spread), _layer(backend, spread)
    x, ids, weights = _problem(case)
    sc = layer.scalars
    xq, xsf = fp4_quantize(x, sc["global_scale"], M._SF_VEC, False, SF_LINEAR)
    x_dq = _dequant(xq.view(torch.uint8), xsf.view(x.shape[0], H // VEC)) * ISC1
    if backend == "CUTEDSL":
        g2 = [float(sc["global_sf"])] * E
    else:
        g2 = (1.0 / ck.isc2).tolist()
    out = torch.zeros(x.shape[0], H, dtype=torch.float64, device=DEV)
    for e in range(E):
        token, slot = (ids == e).nonzero(as_tuple=True)
        w = weights[token, slot].double()
        token, w = token[w != 0], w[w != 0]
        if token.numel() == 0:
            continue
        ws2 = ck.ws2_gu[e].double()
        xe = x_dq[token]
        up = xe @ (_dequant(ck.up_w[e], ck.up_s[e]) * ws2).T
        gate = xe @ (_dequant(ck.gate_w[e], ck.gate_s[e]) * ws2).T
        act = up * gate * torch.sigmoid(gate)
        act = quantize_nvfp4(act.float(), g2[e]).double() / g2[e]
        y = act @ (_dequant(ck.down_w[e], ck.down_s[e]) * ck.ws2_dn[e].double()).T
        out.index_add_(0, token, y * w[:, None])
    return out


@functools.lru_cache(maxsize=None)
def _routed(backend: str, case: str, spread: bool) -> torch.Tensor:
    x, ids, weights = _problem(case)
    run = _cutedsl_routed if backend == "CUTEDSL" else _trtllm_gen_routed
    return run(_layer(backend, spread), x, ids, weights).double()


def _token_error(out: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """Each token's `|out - ref| / |ref|` (L2 over the hidden width)."""
    return (out - ref).norm(dim=1) / ref.norm(dim=1).clamp_min(1e-12)


# ── tests ─────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("backend", BACKENDS)
def test_routed_output_matches_the_float64_reference(backend: str) -> None:
    case = "one_hot"
    _, ids, weights = _problem(case)
    assert torch.unique(ids[weights != 0]).numel() == E, "the routing misses an expert"
    err = _token_error(_routed(backend, case, True), _reference(backend, case, True))
    print(f"[err] {backend} {case}: mean {err.mean():.4e} max {err.max():.4e}")
    assert err.max() <= REFERENCE_TOLERANCE, (err.mean().item(), err.max().item())


def test_cutedsl_matches_trtllm_gen_under_a_uniform_input_scale() -> None:
    case = "weighted"
    _, ids, weights = _problem(case)
    assert torch.unique(ids[weights != 0]).numel() == E, "the routing misses an expert"
    err = _token_error(_routed("CUTEDSL", case, False), _routed("TRTLLM", case, False))
    print(f"[err] CUTEDSL vs TRTLLM {case}: mean {err.mean():.4e} max {err.max():.4e}")
    assert err.max() <= BACKEND_TOLERANCE, (err.mean().item(), err.max().item())
