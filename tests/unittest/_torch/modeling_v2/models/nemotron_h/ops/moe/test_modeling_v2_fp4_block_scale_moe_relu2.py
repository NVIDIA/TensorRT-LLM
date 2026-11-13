# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the nemotron_h family's fp4_block_scale_moe_relu2 op.

Each cell drives the op on a Relu2 layer at Nemotron-3 Ultra's latent-space geometry (512
experts, H 2048, I 5120) the way its NVFP4 targets call it -- pre-routed sigmoid-x5 weights,
per-expert `weight_scale_2`, the scalars mapped from the checkpoint's form -- and gates it
against the op's own fp32 reference in the shared runner's band (`H._compare`). The NVFP4
operands come from the shared test helpers; the Relu2 layout (no gate interleave), scalars
and routing are this family's and stay in this file.
"""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.nemotron_h.ops.moe.fp4_block_scale_moe_relu2 import (  # noqa: E501
    ACT_TYPE_RELU2,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.nemotron_h.ops.moe.fp4_block_scale_moe_relu2 import (  # noqa: E501
    fp4_block_scale_moe_relu2 as op,
)

__extra_import_path__ = ["../../../.."]
import _fp4_moe_test_utils as H  # noqa: E402
from _precision import true_fp32_matmul  # noqa: E402
from _validating import certifying, validating  # noqa: E402,F401 — path above; fixture by name

assert torch.cuda.is_available(), "fp4_block_scale_moe_relu2 requires a CUDA device"

DEV = "cuda"
G1 = 137.0  # activation global scale, as the shared runner's test draws it

# The routing arguments the NVFP4 targets pass with their pre-routed call.
_ROUTING_ARGS = dict(routing_method_type=2, n_group=1, topk_group=1, routed_scaling_factor=5.0)


def blk32_row_perm(rows: int, device=None) -> torch.Tensor:
    """Gather index of trtllm-gen's 32-row block shuffle (epilogue tile M 128).

    Within each block of 32 rows, source row `4u + v` lands at destination row `8v + u`.
    """
    assert rows % 32 == 0, f"{rows} rows is not a whole number of 32-row blocks"
    j = torch.arange(32, device=device)
    src = torch.empty_like(j)
    src[(j % 4) * 8 + j // 4] = j
    block_base = torch.arange(rows // 32, device=device).repeat_interleave(32) * 32
    return src.repeat(rows // 32) + block_base


def swizzle_128x4(sf: torch.Tensor) -> torch.Tensor:
    """trtllm-gen's 128x4 block-scale swizzle per expert, viewed back as `[E, M, C]`."""
    e, m, c = sf.shape
    assert m % 128 == 0 and c % 4 == 0, f"[{m}, {c}] block scales would need padding"
    v = sf.reshape(e, m // 128, 4, 32, c // 4, 4).permute(0, 1, 4, 3, 2, 5)
    return v.reshape(e, m, c).contiguous()


def single_layout(packed: torch.Tensor, block_scales: torch.Tensor):
    """An NVFP4 stack the kernel reads as one matrix (Relu2's FC1, every FC2): rows
    block-shuffled, the scales then 128x4-swizzled. No gate half to interleave."""
    perm = blk32_row_perm(packed.shape[1], packed.device)
    weights = packed.index_select(1, perm).contiguous()
    scales = block_scales.view(torch.uint8).index_select(1, perm)
    return weights, swizzle_128x4(scales).view(torch.float8_e4m3fn)


def relu2_scalars(up_input_scale, up_weight_scale_2, down_input_scale, down_weight_scale_2):
    """`(output1_scale_scalar, output1_scale_gate_scalar, output2_scale_scalar)`, each `[E]`
    fp32, from a ModelOpt checkpoint's layer-shared input scales and per-expert weight
    scales (see the op's note)."""
    up_is = up_input_scale.float().reshape(())
    down_is = down_input_scale.float().reshape(())
    num_experts = up_weight_scale_2.numel()
    s1 = (1.0 / down_is).expand(num_experts).contiguous()
    sg = (up_is * up_weight_scale_2.float().reshape(-1)).contiguous()
    s2 = (down_is * down_weight_scale_2.float().reshape(-1)).contiguous()
    return s1, sg, s2


def _rand_weight_scale_2(num_experts: int, gen: torch.Generator) -> torch.Tensor:
    """Per-expert fp32 global weight scales, log-uniform over a factor of 4 and never a
    power of two, so a scalar applied to the wrong expert shows."""
    return torch.exp2(torch.rand(num_experts, device=DEV, generator=gen) * 2.0 - 1.0) / 64.0


def _sigmoid_x5_routing(num_tokens, num_experts, top_k, gen):
    """Pre-routed ids/weights in the form Nemotron-3 Ultra's router emits: selection on
    sigmoid scores plus a large, nearly flat correction bias; the selected unbiased scores
    renormalized and scaled to sum to 5. Distinct ids per row."""
    scores = torch.sigmoid(
        torch.randn(num_tokens, num_experts, device=DEV, dtype=torch.float32, generator=gen)
    )
    bias = 25.25 + 0.125 * torch.rand(num_experts, device=DEV, generator=gen)
    idx = torch.topk(scores + bias, top_k, dim=-1).indices
    w = torch.gather(scores, 1, idx)
    w = w / w.sum(-1, keepdim=True) * 5.0
    return idx.to(torch.int32).contiguous(), w.to(torch.bfloat16).contiguous()


@pytest.fixture(scope="module")
def ultra_layer():
    """Nemotron-3 Ultra's 512 latent-space experts, built on first use and released at
    teardown: ~6 GiB of kernel-ready weights plus the reference operands."""
    cache = {}

    def get():
        if not cache:
            gen = torch.Generator(device=DEV).manual_seed(5122048)
            num_experts, hidden, inter = 512, 2048, 5120
            up_p, up_s, up_c = H._rand_nvfp4(num_experts, inter, hidden, gen)
            dn_p, dn_s, dn_c = H._rand_nvfp4(num_experts, hidden, inter, gen)
            up_ws2 = _rand_weight_scale_2(num_experts, gen)
            dn_ws2 = _rand_weight_scale_2(num_experts, gen)
            w1, s1 = single_layout(up_p, up_s)
            w2, s2 = single_layout(dn_p, dn_s)
            cache["args"] = dict(
                gemm1_weights=w1,
                gemm1_weights_scale=s1,
                gemm2_weights=w2,
                gemm2_weights_scale=s2,
            )
            cache["ref"] = dict(up=(up_c, up_s, up_ws2), down=(dn_c, dn_s, dn_ws2))
        return cache["args"], cache["ref"]

    yield get
    cache.clear()
    torch.cuda.empty_cache()


def _window(operand, off: int, n: int):
    return tuple(t[off : off + n] for t in operand)


def _calibrate_down_input_scale(x, ids, up, off: int, n: int) -> float:
    """`amax / (448 * 6)` of the true relu2 activation over the window: the FC2 input scale a
    checkpoint's `down_proj.input_scale` encodes."""
    amax = 0.0
    with true_fp32_matmul():
        for e in range(n):
            tok = (ids == off + e).nonzero(as_tuple=True)[0].unique()
            if tok.numel() == 0:
                continue
            u = x[tok] @ H._dequant(up[0][e], up[1][e]).t() * up[2][e].float()
            amax = max(amax, torch.relu(u).square().abs().max().item())
    return max(amax, 1e-6) / (H.E4M3_MAX * H.E2M1_MAX)


def _call(data, sf, args, spec, s1, sg, s2, ids, wts, **overrides):
    off, n = spec["local_expert_offset"], spec["local_num_experts"]
    kw = dict(
        routing_logits=None,
        routing_bias=None,
        hidden_states=data,
        hidden_states_scale=sf.view(torch.float8_e4m3fn),
        gemm1_weights=args["gemm1_weights"][off : off + n],
        gemm1_weights_scale=args["gemm1_weights_scale"][off : off + n],
        gemm1_bias=None,
        gemm1_alpha=None,
        gemm1_beta=None,
        gemm1_clamp_limit=None,
        gemm2_weights=args["gemm2_weights"][off : off + n],
        gemm2_weights_scale=args["gemm2_weights_scale"][off : off + n],
        gemm2_bias=None,
        output1_scale_scalar=s1[off : off + n],
        output1_scale_gate_scalar=sg[off : off + n],
        output2_scale_scalar=s2[off : off + n],
        num_experts=spec["num_experts"],
        top_k=spec["top_k"],
        intermediate_size=spec["intermediate"],
        local_expert_offset=off,
        local_num_experts=n,
        do_finalize=True,
        act_type=spec["act_type"],
        topk_weights=wts,
        topk_ids=ids,
        **_ROUTING_ARGS,
    )
    kw.update(overrides)
    with validating(op):
        return op(**kw)


def _operands(spec, seed, ultra_layer):
    args, ref = ultra_layer()
    gen = torch.Generator(device=DEV).manual_seed(seed)
    data, sf, x = H._rand_activation(spec["tokens"], spec["hidden"], G1, gen)
    ids, wts = _sigmoid_x5_routing(spec["tokens"], spec["num_experts"], spec["top_k"], gen)
    off, n = spec["local_expert_offset"], spec["local_num_experts"]
    down_is = _calibrate_down_input_scale(x, ids, ref["up"], off, n)
    up_is = torch.tensor(1.0 / G1, dtype=torch.float32, device=DEV)
    s1, sg, s2 = relu2_scalars(
        up_is, ref["up"][2], torch.tensor(down_is, device=DEV), ref["down"][2]
    )
    # The FC2-input global scale the kernel is handed, which the reference requantizes with.
    g2 = s1[0].item()
    return args, ref, data, sf, x, ids, wts, (s1, sg, s2), g2


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying, ultra_layer) -> None:  # noqa: F811
    certifying(op)
    spec = cell.spec
    assert spec["act_type"] == ACT_TYPE_RELU2 and spec["routing"] == "sigmoid_x5"
    seed = zlib.crc32(cell.why.encode()) % 2**31
    args, ref, data, sf, x, ids, wts, (s1, sg, s2), g2 = _operands(spec, seed, ultra_layer)
    off, n = spec["local_expert_offset"], spec["local_num_experts"]
    y = _call(data, sf, args, spec, s1, sg, s2, ids, wts)[0]
    assert y.shape == (spec["tokens"], spec["hidden"]) and y.dtype == torch.bfloat16
    with true_fp32_matmul():
        exp = op.reference(
            x,
            ids,
            wts,
            _window(ref["up"], off, n),
            _window(ref["down"], off, n),
            g2,
            local_expert_offset=off,
            local_num_experts=n,
        )
    H._compare(y, exp.to(torch.bfloat16))


def test_relu2_refuses_a_gated_fc1_stack(ultra_layer) -> None:
    """A 2I-row (SwiGlu-shaped) FC1 handed to Relu2 is read at I-row strides and answers
    with plausible rows from the wrong weights; the guard refuses it."""
    spec = dict(op.CELLS[0].spec, tokens=4)
    args, _, data, sf, _, ids, wts, (s1, sg, s2), _ = _operands(spec, 7, ultra_layer)
    off, n = spec["local_expert_offset"], spec["local_num_experts"]
    w1 = args["gemm1_weights"][off : off + n]
    gated = torch.cat([w1, w1], dim=1)
    sf1 = args["gemm1_weights_scale"][off : off + n]
    with pytest.raises(AssertionError, match="needs FC1 with"):
        _call(
            data,
            sf,
            args,
            spec,
            s1,
            sg,
            s2,
            ids,
            wts,
            gemm1_weights=gated,
            gemm1_weights_scale=torch.cat([sf1, sf1], dim=1),
        )
