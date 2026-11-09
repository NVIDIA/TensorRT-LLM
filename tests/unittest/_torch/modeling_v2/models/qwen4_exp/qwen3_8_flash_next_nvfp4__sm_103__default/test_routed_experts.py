# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The routed experts at one rank (tp1): the CuteDSL experts and the NVFP4 draft experts.

**CuteDSL.** The CuteDSL op tests build their operands with layout code of their own. Here the
checkpoint's expert tensors (512 experts: packed NVFP4 `up_proj` / `gate_proj` / `down_proj`
codes, linear e4m3 block scales, one `weight_scale_2` per expert, per-expert `input_scale`s)
go through the target's own load-time transforms (`weights._expert_fc1_w` / `_expert_fc1_s` /
`_expert_fc2_w` / `_expert_fc2_s`) and scalar derivation (`modeling._moe_layer_scalars`,
`_moe_layer_scalars_cd`), once per routed backend, then through the call sequence of the
target's routed experts -- CuteDSL (`_cute_dsl_routed`): sort, FC1 with SwiGLU, FC2,
unpermute; trtllm-gen: the one-call block-scale runner -- with explicit routing.

Two routings hit all 512 experts. In the first every token's only non-zero routing weight is
on its own expert, so each expert's output is checked alone: a layout error confined to one
expert (a permutation, a scale swizzle, an alpha) cannot hide in a sum over ten. On it both
backends are held to a float64 reference over the dequantized checkpoint tensors, with
`down_proj.input_scale` spread across the experts as a calibrated checkpoint's is. The second
is a weighted top-10: with a uniform `down_proj.input_scale` the two backends requantize the
SwiGLU output identically, so on it they are held to each other. The reference models the one
rounding step that is not exact: the SwiGLU output is requantized to NVFP4 before FC2, at the
backend's own global scale (the layer's single `global_sf` on CuteDSL, each expert's
`1 / down_proj.input_scale` on trtllm-gen). The scalars, the CuteDSL weight row orders and the
prefill copy's decision are also checked on the host.

**NVFP4 draft experts** (`TRTLLM_MODELING_V2_MTP_MOE_NVFP4`). `weights.mtp_experts_to_nvfp4`
requantizes FP8 128x128 block-scaled experts to NVFP4 in the trunk's trtllm-gen layout,
`modeling._mtp_moe_nvfp4` adds the runner's scalars and `modeling._mtp_routed_nvfp4` runs
them. Against the FP8 block-scale path the draft layer runs by default, on the same random
experts, activations and router, the NVFP4 output must stay inside a relative-error band of
the FP8 one, keep its magnitude (a wrong runner scalar moves the norm, which unbiased
quantization noise does not), and leave no FP8 copy behind.
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
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.ops.moe.fp8_block_scale_moe_runner import (  # noqa: E501
    fp8_block_scale_moe_runner,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.ops.moe.renorm_moe_routing_op import (  # noqa: E501
    renorm_moe_routing_op,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.ops.quantization.fp8_quantize_1x128 import (  # noqa: E501
    fp8_quantize_1x128,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_103__default import (  # noqa: E501
    modeling as M,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_103__default import (  # noqa: E501
    weights as W,
)

_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="the experts run on a CUDA device")
DEV = "cuda"


# ---- the scalars and the weight layout, on the host ---------------------------------------

SCALAR_EXPERTS = 8


def _scalar_rows(isc2: torch.Tensor, seed: int = 0) -> dict:
    gen = torch.Generator().manual_seed(seed)
    isc1 = torch.full((SCALAR_EXPERTS,), 0.0123)
    ws2_1 = torch.rand(SCALAR_EXPERTS, generator=gen) * 1e-3 + 1e-4
    return {
        "l0_e_isc1": isc1,
        "l0_e_isc1_up": isc1.clone(),
        "l0_e_ws2_1": ws2_1,
        "l0_e_ws2_1_up": ws2_1.clone(),
        "l0_e_isc2": isc2,
        "l0_e_ws2_2": torch.rand(SCALAR_EXPERTS, generator=gen) * 1e-3 + 1e-4,
    }


@pytest.mark.cpu_only
def test_uniform_isc2_keeps_the_per_expert_products_bitwise() -> None:
    w = _scalar_rows(torch.full((SCALAR_EXPERTS,), 0.0371))
    got = M._moe_layer_scalars_cd(w, 0)
    assert torch.equal(got["alpha2"], w["l0_e_isc2"] * w["l0_e_ws2_2"])
    assert torch.equal(got["global_sf"], 1.0 / w["l0_e_isc2"][:1])
    assert got["global_sf"].shape == (1,)
    assert torch.equal(got["alpha1"], w["l0_e_isc1"] * w["l0_e_ws2_1"])


@pytest.mark.cpu_only
def test_isc2_spread_is_folded_into_alpha2() -> None:
    """CuteDSL requantizes FC1's output at ONE global scale per layer, where trtllm-gen folds
    each expert's `down_proj.input_scale` into its own FC1 scalar. Whatever the spread (here
    40x), `global_sf * alpha2[e]` is expert e's own `ws2_2[e]`, the product trtllm-gen forms
    from `(1 / isc2[e]) * (isc2[e] * ws2_2[e])`."""
    isc2 = 0.01 * torch.linspace(1.0, 40.0, SCALAR_EXPERTS)
    w = _scalar_rows(isc2.flip(0))
    cute = M._moe_layer_scalars_cd(w, 0)
    gen = M._moe_layer_scalars(w, 0, 0, SCALAR_EXPERTS)
    isc2 = w["l0_e_isc2"]
    assert torch.equal(cute["global_sf"], 1.0 / isc2.max().reshape(1))
    # No expert's FC2 input is scaled above what the checkpoint calibrated for it.
    assert (cute["global_sf"] <= 1.0 / isc2).all()
    assert torch.allclose(
        cute["global_sf"] * cute["alpha2"], (1.0 / isc2) * gen["output2_scale_scalar"], rtol=1e-6
    )
    assert torch.allclose(cute["global_sf"] * cute["alpha2"], w["l0_e_ws2_2"], rtol=1e-6)


@pytest.mark.cpu_only
def test_gate_up_input_scale_mismatch_is_refused() -> None:
    w = _scalar_rows(torch.full((SCALAR_EXPERTS,), 0.0371))
    w["l0_e_isc1_up"] = w["l0_e_isc1_up"] * 1.5
    with pytest.raises(AssertionError, match="gate/up input_scale differ"):
        M._moe_layer_scalars(w, 0, 0, SCALAR_EXPERTS)


@pytest.mark.cpu_only
def test_cutedsl_fc1_pairs_up_and_gate_rows_in_64_row_groups() -> None:
    """Destination rows `[128k, 128k + 64)` hold up rows `[64k, 64k + 64)` and the next 64
    hold the matching gate rows, which sit `I` rows later in the `[up ; gate]` stack."""
    inter = 640
    perm = W._fc1_perm(2 * inter, "cpu", SimpleNamespace(trunk_moe_backend="CUTEDSL"))
    assert sorted(perm.tolist()) == list(range(2 * inter))
    for k in range(inter // 64):
        assert perm[128 * k : 128 * k + 64].tolist() == list(range(64 * k, 64 * k + 64))
        assert perm[128 * k + 64 : 128 * k + 128].tolist() == list(
            range(inter + 64 * k, inter + 64 * k + 64)
        )


@pytest.mark.cpu_only
def test_cutedsl_fc2_keeps_the_checkpoint_row_order_and_trtllm_gen_shuffles_it() -> None:
    """trtllm-gen's 32-row block shuffle moves source row `4u + v` to destination row
    `8v + u`; CuteDSL reads `down_proj` as the checkpoint stores it."""
    rows = torch.arange(64).unsqueeze(1).expand(64, 8).contiguous()
    kept = W._expert_fc2_w(rows, SimpleNamespace(trunk_moe_backend="CUTEDSL"))
    assert torch.equal(kept, rows)
    shuffled = W._expert_fc2_w(rows, SimpleNamespace(trunk_moe_backend="TRTLLM"))
    for u in range(8):
        for v in range(4):
            assert int(shuffled[8 * v + u, 0]) == 4 * u + v
            assert int(shuffled[32 + 8 * v + u, 0]) == 32 + 4 * u + v


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "one_rank, backend, max_num_tokens, env, kept",
    [
        (True, "TRTLLM", 4096, None, True),
        (True, "TRTLLM", 4095, None, False),
        (True, "TRTLLM", 16384, "0", False),
        (True, "CUTEDSL", 16384, None, False),
        (False, "TRTLLM", 16384, None, False),
    ],
)
def test_second_expert_copy_kept_only_where_its_launches_can_happen(
    monkeypatch, one_rank, backend, max_num_tokens, env, kept
):
    """The copy costs about 1.3 GiB a layer: it is kept only at one rank, under trtllm-gen,
    when a step's rows can reach the CuteDSL dispatch threshold, and not when the env var is
    off."""
    if env is None:
        monkeypatch.delenv(M._MOE_PREFILL_COPY_ENV, raising=False)
    else:
        monkeypatch.setenv(M._MOE_PREFILL_COPY_ENV, env)
    assert M._moe_prefill_copy(one_rank, backend, max_num_tokens) is kept


# ---- CuteDSL: the experts' geometry and gates ---------------------------------------------

#: The trunk's routed experts: count, top-k, hidden, intermediate, NVFP4 block.
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


# ---- CuteDSL: the checkpoint -------------------------------------------------


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
        # The rows `modeling._moe_layer_scalars` / `_moe_layer_scalars_cd` read, under layer 0.
        scalars={
            "l0_e_isc1": isc1,
            "l0_e_isc1_up": isc1.clone(),
            "l0_e_ws2_1": ws2_gu,
            "l0_e_ws2_1_up": ws2_gu.clone(),
            "l0_e_isc2": isc2,
            "l0_e_ws2_2": ws2_dn,
        },
    )


# ---- CuteDSL: what the target builds from it ---------------------------------


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
        scalars=_expert_scalars(ck.scalars, backend),
    )


def _expert_scalars(rows: dict, backend: str) -> dict:
    """Layer 0's routed-expert scalars as the target derives them for `backend`: the
    trtllm-gen runner's (with the FC1 activation's global scale both backends quantize at),
    plus on CuteDSL its FC1 / FC2 alphas and FC2 global scale."""
    scalars = M._moe_layer_scalars(rows, 0, 0, E)
    if backend == "CUTEDSL":
        scalars.update(M._moe_layer_scalars_cd(rows, 0))
    return scalars


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


# ---- CuteDSL: the inputs and the reference -----------------------------------


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


@_cuda
@pytest.mark.parametrize("backend", BACKENDS)
def test_routed_output_matches_the_float64_reference(backend: str) -> None:
    case = "one_hot"
    _, ids, weights = _problem(case)
    assert torch.unique(ids[weights != 0]).numel() == E, "the routing misses an expert"
    err = _token_error(_routed(backend, case, True), _reference(backend, case, True))
    print(f"[err] {backend} {case}: mean {err.mean():.4e} max {err.max():.4e}")
    assert err.max() <= REFERENCE_TOLERANCE, (err.mean().item(), err.max().item())


@_cuda
def test_cutedsl_matches_trtllm_gen_under_a_uniform_input_scale() -> None:
    case = "weighted"
    _, ids, weights = _problem(case)
    assert torch.unique(ids[weights != 0]).numel() == E, "the routing misses an expert"
    err = _token_error(_routed("CUTEDSL", case, False), _routed("TRTLLM", case, False))
    print(f"[err] CUTEDSL vs TRTLLM {case}: mean {err.mean():.4e} max {err.max():.4e}")
    assert err.max() <= BACKEND_TOLERANCE, (err.mean().item(), err.max().item())


# ---- the NVFP4 draft experts --------------------------------------------------------------

#: The draft experts' FP8 weight block.
BLK = 128
#: A trunk calibration to derive the static draft scales from: the checkpoint's last trunk layer
#: (FC1 input, FC2 input amax), the layer that feeds the draft.
TRUNK_AMAX = (11.1875, 57.75)
#: Relative RMS of NVFP4 - FP8 over the FP8 output, the floor on each row's cosine, and the
#: band on the output norm ratio. Measured here: 0.239-0.251 RMS, row cosine >= 0.961, at every
#: row count; on the checkpoint's real draft experts 0.222-0.231 and >= 0.964 (bench). That is
#: the NVFP4 chain's own noise -- four e2m1 roundings (activation, FC1 weight, FC1 output, FC2
#: weight) at ~0.1 each, compounded by the SwiGLU product -- not a bias: it is flat across
#: activation scales 0.5-4x, and FP8 sits at 0.07 from the fp32 reference of the same weights.
REL_RMS_BAND = 0.3
COS_FLOOR = 0.95
NORM_RATIO_BAND = 0.1


def _fp8_blocks(rows: int, cols: int, gen: torch.Generator):
    """`[E, rows, cols]` e4m3 codes and `[E, rows/128, cols/128]` fp32 scales of a random
    expert stack whose 128x128 blocks span 16x in magnitude, as trained experts' do."""
    w = torch.randn(E, rows, cols, generator=gen, device=DEV) * cols**-0.5
    spread = torch.exp2(
        torch.randint(-2, 3, (E, rows // BLK, 1, cols // BLK, 1), generator=gen, device=DEV)
    ).float()
    blocks = w.view(E, rows // BLK, BLK, cols // BLK, BLK) * spread
    scale = blocks.abs().amax(dim=(2, 4), keepdim=True) / 448.0
    codes = (blocks / scale).to(torch.float8_e4m3fn).view(E, rows, cols)
    return codes, scale.view(E, rows // BLK, cols // BLK).contiguous()


@pytest.fixture(scope="module")
def draft_experts():
    gen = torch.Generator(device=DEV).manual_seed(20261022)
    fc1_w, fc1_s = _fp8_blocks(2 * INTER, H, gen)
    fc2_w, fc2_s = _fp8_blocks(H, INTER, gen)
    router = (torch.randn(E, H, generator=gen, device=DEV) * H**-0.5).to(torch.bfloat16)
    fp8 = SimpleNamespace(fc1_w=fc1_w, fc1_s=fc1_s, fc2_w=fc2_w, fc2_s=fc2_s, router=router.t())
    p = "l1_mtp_"
    w = {p + k: getattr(fp8, k).clone() for k in ("fc1_w", "fc1_s", "fc2_w", "fc2_s")}
    w["l0_e_isc1"] = torch.full((E,), TRUNK_AMAX[0] / 2688.0, device=DEV)
    w["l0_e_isc2"] = torch.full((E,), TRUNK_AMAX[1] / 2688.0, device=DEV)
    # No packed FP8 draft projections: the shared expert runs its bf16 GEMMs.
    core = SimpleNamespace(
        w=w,
        mtp_layer_idx=1,
        num_layers=1,
        num_experts=E,
        topk=K,
        moe_inter=INTER,
        _mtp_gemv8={},
    )
    fields = M._mtp_moe_nvfp4(core)
    nvfp4 = SimpleNamespace(router=fp8.router, **fields)
    return core, fp8, nvfp4


def _fp8_routed(fp8, x):
    """The FP8 block-scale draft path, call for call as `_mtp_moe_block` makes it."""
    logits = M.cublas_mm(x, fp8.router)
    ids, wts = renorm_moe_routing_op(logits, topk=K, output_dtype=torch.bfloat16)
    xq, xsf = fp8_quantize_1x128(x, use_ue8m0=False)
    return fp8_block_scale_moe_runner(
        routing_logits=None,
        routing_bias=None,
        hidden_states=xq,
        hidden_states_scale=xsf,
        gemm1_weights=fp8.fc1_w,
        gemm1_weights_scale=fp8.fc1_s,
        gemm2_weights=fp8.fc2_w,
        gemm2_weights_scale=fp8.fc2_s,
        num_experts=E,
        top_k=K,
        num_fused_shared_experts=None,
        n_group=None,
        topk_group=None,
        intermediate_size=INTER,
        local_expert_offset=0,
        local_num_experts=E,
        routed_scaling_factor=None,
        routing_method_type=M._ROUTING_METHOD_RENORM,
        topk_weights=wts,
        topk_ids=ids,
        act_type=M._ACT_TYPE_SWIGLU,
        gemm1_clamp_limit=None,
    )


@_cuda
def test_draft_fp8_copies_freed_and_layout(draft_experts):
    core, _, nvfp4 = draft_experts
    for k in ("fc1_w", "fc1_s", "fc2_w", "fc2_s"):
        assert core.w[f"l1_mtp_{k}"].numel() == 0, k
    assert nvfp4.fc1_w.shape == (E, 2 * INTER, H // 2) and nvfp4.fc1_w.dtype == torch.uint8
    assert nvfp4.fc1_s.shape == (E, 2 * INTER, H // 16) and nvfp4.fc1_s.dtype == torch.float8_e4m3fn
    assert nvfp4.fc2_w.shape == (E, H, INTER // 2) and nvfp4.fc2_s.shape == (E, H, INTER // 16)
    for s in (nvfp4.out1_scale, nvfp4.out1_gate_scale, nvfp4.out2_scale):
        assert s.shape == (E,) and s.dtype == torch.float32 and torch.isfinite(s).all()
    assert nvfp4.moe_gscale.shape == (1,)


@_cuda
@pytest.mark.parametrize("rows", [1, 4, 32, 128, 1024])
def test_draft_nvfp4_tracks_fp8(draft_experts, rows):
    core, fp8, nvfp4 = draft_experts
    gen = torch.Generator(device=DEV).manual_seed(rows)
    x = torch.randn(rows, H, generator=gen, device=DEV).to(torch.bfloat16)
    ref = _fp8_routed(fp8, x).float()
    out = M._mtp_routed_nvfp4(core, nvfp4, x)
    assert out.shape == (rows, H) and out.dtype == torch.bfloat16
    out = out.float()
    rel = ((out - ref).norm() / ref.norm()).item()
    cos = torch.nn.functional.cosine_similarity(out, ref, dim=1).min().item()
    ratio = (out.norm() / ref.norm()).item()
    print(f"rows={rows}: rel RMS {rel:.4f}, min row cosine {cos:.4f}, norm ratio {ratio:.4f}")
    assert rel < REL_RMS_BAND, rel
    assert cos > COS_FLOOR, cos
    assert abs(ratio - 1.0) < NORM_RATIO_BAND, ratio


@_cuda
@pytest.mark.parametrize("rows", [1, 64, 512])
def test_draft_shared_expert_overlap_preserves_serial_output(draft_experts, rows):
    """`DecodeTarget._mtp_moe` over the NVFP4 experts runs the shared expert on the side stream
    beside them: its output is the serial composition's bitwise, eager and replayed from a
    CUDA graph over new inputs."""
    core, _, nvfp4 = draft_experts
    gen = torch.Generator(device=DEV).manual_seed(42)
    # The checkpoint's fused shared expert: gate|up rows, then the one gate-logit row.
    sh_gu = (torch.randn(2 * INTER + 1, H, generator=gen, device=DEV) / H**0.5).to(torch.bfloat16)
    sh_dn = (torch.randn(H, INTER, generator=gen, device=DEV) / INTER**0.5).to(torch.bfloat16)
    weights = SimpleNamespace(
        **vars(nvfp4),
        sh_gu=sh_gu[: 2 * INTER].t(),
        sh_dn=sh_dn.t(),
        sh_gate=sh_gu[2 * INTER :].t(),
    )
    target = object.__new__(M.DecodeTarget)
    target.core = SimpleNamespace(
        **vars(core), _mtp=weights, _side_stream=torch.cuda.Stream(), shared_inter=INTER
    )
    x = torch.randn(rows, H, generator=gen, device=DEV).to(torch.bfloat16)

    def serial():
        routed = M._mtp_routed_nvfp4(target.core, weights, x)
        up = M.silu_and_mul_rows(M.cublas_mm(x, weights.sh_gu), INTER)
        shared = M.cublas_mm(up, weights.sh_dn)
        gate = M.cublas_mm(x, weights.sh_gate)
        return M.fused_sigmoid_gate_mul_add(routed, gate, shared)

    expected = serial().clone()
    torch.testing.assert_close(target._mtp_moe(x), expected, rtol=0, atol=0)
    if rows == 64:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = target._mtp_moe(x)
        for _ in range(3):
            x.normal_(generator=gen)
            expected = serial().clone()
            graph.replay()
            torch.testing.assert_close(captured, expected, rtol=0, atol=0)
