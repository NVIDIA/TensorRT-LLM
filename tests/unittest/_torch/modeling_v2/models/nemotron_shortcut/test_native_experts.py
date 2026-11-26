# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compare the masked NVFP4 expert pipeline with independent per-expert projections."""

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.catalog.gemm.nvfp4_gemm import nvfp4_gemm
from tensorrt_llm._torch.modeling_v2.catalog.quantization.fp4_quantize import fp4_quantize
from tensorrt_llm._torch.modeling_v2.models.nemotron_shortcut.nvfp4__sm_103__default.ops import (
    grouped_experts,
)
from tensorrt_llm._torch.modeling_v2.models.nemotron_shortcut.nvfp4__sm_103__default.weights import (
    Projection,
    _kv_scale_keys,
    _load_kv_scales,
    _load_projection,
    projection_parts,
)


@pytest.mark.cpu_only
def test_mtp_kv_calibration_on_excluded_projections():
    prefix = "model.language_model.mtp.layers.0.block.self_attn."
    entries = [(prefix + f"{axis}_proj.weight", None, None, None) for axis in "qkv"]
    k, v = prefix + "k_proj.k_scale", prefix + "v_proj.v_scale"
    weights = {k: torch.tensor([0.25]), v: torch.tensor(0.5)}
    scales, inverse = torch.empty(3), torch.empty(3)

    def refit():
        keys = _kv_scale_keys(entries, weights, required=False)
        _load_kv_scales(scales, inverse, keys, weights)

    refit()
    torch.testing.assert_close(scales, torch.tensor([1.0, 0.25, 0.5]), rtol=0, atol=0)
    torch.testing.assert_close(inverse, torch.tensor([1.0, 4.0, 2.0]), rtol=0, atol=0)
    del weights[k]
    with pytest.raises(ValueError, match=r"Missing KV calibration: .*k_proj\.k_scale"):
        refit()
    del weights[v]
    refit()
    torch.testing.assert_close(scales, torch.ones(3), rtol=0, atol=0)
    torch.testing.assert_close(inverse, torch.ones(3), rtol=0, atol=0)


@pytest.mark.cpu_only
def test_unquantized_mtp_experts_use_per_expert_tp_windows():
    from tensorrt_llm.models.modeling_utils import QuantConfig

    prefix = "model.language_model.mtp.layers.0.block.moe.experts"
    excluded = QuantConfig(
        exclude_modules=["model.language_model.mtp*"]
    ).is_module_excluded_from_quantization
    eager = torch.arange(12, dtype=torch.bfloat16).reshape(3, 4)
    checkpoint = {f"{prefix}.{expert}.down_proj.weight": eager + expert * 12 for expert in (1, 3)}
    entries = [(f"{prefix}.down_proj", ([3, 1], slice(None), slice(2, 4)), None, None)]
    (part,) = projection_parts("mtp_experts_down", (2, 3, 2), entries, {}, excluded)
    target = {}
    part.declare(target, lambda *shape: torch.empty(shape, dtype=torch.bfloat16))
    _load_projection(part, target, checkpoint)
    expected = torch.stack([checkpoint[f"{prefix}.{i}.down_proj.weight"][:, 2:] for i in (3, 1)])
    torch.testing.assert_close(part.view(target), expected)
    assert part.suffixes == ("weight",)


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
    reason="native NVFP4 expert kernels require Blackwell SM100/SM103",
)
def test_expert_calibration_and_weighted_fc2_across_ep():
    # Experts 0, 2 and 3 receive different patterns. Expert 1 is empty. Expert
    # 2 receives 131 assignments, crossing the 128-row scratch tile boundary.
    device = torch.device("cuda")
    hidden = 256
    n_experts = 4
    n_tokens = 131
    channel = torch.arange(hidden, device=device)
    token = torch.arange(n_tokens, device=device)
    ids = torch.stack((torch.full_like(token, 2), torch.where(token % 3 == 0, 3, 0)), dim=1)
    scores = torch.stack((0.31 + (token % 5) * 0.08, 0.69 - (token % 5) * 0.08), dim=1)
    x = ((token[:, None] * 3 + channel) % 71 - 35).to(torch.bfloat16)

    checkpoint, decoded = {}, {}
    for proj in ("up_proj", "down_proj"):
        for expert in range(n_experts):
            prefix = f"moe.experts.{expert}.{proj}"
            # One nonzero per row makes accumulator rounding exact. Both nibble
            # positions and nonuniform block scales are exercised.
            col = (channel + expert + (proj == "down_proj")) % hidden
            packed = torch.zeros((hidden, hidden // 2), device=device, dtype=torch.uint8)
            code = 2 + 2 * (channel % 2)  # e2m1 values 1 or 2
            packed[channel, col // 2] = (code << (4 * (col % 2))).to(torch.uint8)
            scales = 0.5 + ((channel[:, None] + torch.arange(hidden // 16, device=device)) % 4) / 4
            scales = scales.to(torch.float8_e4m3fn)
            weight_scale = torch.tensor(0.5 + expert * 0.25, device=device)
            input_scale = torch.tensor(0.0625 + expert * 0.03125, device=device)
            checkpoint[prefix + ".weight"] = packed
            checkpoint[prefix + ".weight_scale"] = scales
            checkpoint[prefix + ".weight_scale_2"] = weight_scale
            checkpoint[prefix + ".input_scale"] = input_scale
            values = (1 + channel % 2) * scales.float()[channel, col // 16] * weight_scale
            decoded[proj, expert] = col, values, input_scale

    def oracle(rows, proj, expert):
        col, values, scale = decoded[proj, expert]
        packed, sf = fp4_quantize(rows.contiguous(), scale.reciprocal(), 16)
        act = nvfp4_gemm._dequantize(packed, sf) * scale
        return (act[:, col] * values).to(x.dtype)

    want = torch.zeros((n_tokens, 2, hidden), dtype=x.dtype, device=device)
    for expert in (0, 2, 3):
        tid, slot = (ids == expert).nonzero(as_tuple=True)
        up = oracle(x[tid], "up_proj", expert)
        limited = (16.0 * torch.tanh(up.float() / 16.0)).to(x.dtype)
        weighted = (limited.relu().square().float() * scores[tid, slot, None]).to(x.dtype)
        want[tid, slot] = oracle(weighted, "down_proj", expert)

    # Remap logical experts across ranks, with one replica on each rank.
    placements = ([3, 0, 2], [2, 1, 3])
    logical_to_physical = ids.new_tensor([1, 4, 3, 0])
    slots = logical_to_physical[ids]
    ranks = []
    for ep_rank in range(2):
        es = placements[ep_rank]
        banks = []
        for proj in ("up_proj", "down_proj"):
            part = Projection(
                proj,
                "moe.experts." + proj,
                "NVFP4",
                (len(es), hidden, hidden),
                [((es, slice(None)), None)],
            )
            parameters = {}
            part.declare(parameters, lambda *s, dtype: torch.empty(s, dtype=dtype, device=device))
            _load_projection(part, parameters, checkpoint)
            banks.append(part.view(parameters))
        up, down = banks
        ranks.append(grouped_experts(x, slots, scores, up, down, ep_rank, 16.0))
    torch.testing.assert_close(ranks[0] + ranks[1], want.float().sum(1), rtol=0, atol=0)


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
    reason="native mixed projection kernels require Blackwell SM100/SM103",
)
def test_separate_attention_scales_and_padded_segmented_projection():
    from types import SimpleNamespace

    from tensorrt_llm._torch.modeling_v2.models.nemotron_shortcut.nvfp4__sm_103__default.modeling import (
        Nvfp4Decoder,
    )

    device = torch.device("cuda")
    cols = 256
    x = (torch.arange(3 * cols, device=device).view(3, cols) % 61 - 30).to(torch.bfloat16)

    def allocate(*shape, dtype):
        return torch.empty(shape, device=device, dtype=dtype)

    def build(name, shape, entries, recipe, checkpoint):
        weights = {}
        views = []
        for part in projection_parts(name, shape, entries, recipe):
            part.declare(weights, allocate)
            _load_projection(part, weights, checkpoint)
            views.append(part.view(weights))
        return views

    rows = 64
    entries, recipe, checkpoint, want = [], {}, {}, []
    for i, key in enumerate(("q", "k", "v")):
        entries.append((key + ".weight", None, (slice(i * rows, (i + 1) * rows),), None))
        recipe[key] = SimpleNamespace(quant_algo="FP8")
        channel = torch.arange(rows, device=device)
        code = torch.zeros((rows, cols), device=device)
        code[channel, channel + i] = 1.0
        checkpoint[key + ".weight"] = code.to(torch.float8_e4m3fn)
        a, b = 0.015625 * 2**i, 0.25 * 2**i
        checkpoint[key + ".input_scale"] = torch.tensor(a, device=device)
        checkpoint[key + ".weight_scale"] = torch.tensor(b, device=device)
        xq = (x.float() / a).clamp(-448, 448).to(torch.float8_e4m3fn).float()
        want.append((xq[:, channel + i] * (a * b)).to(x.dtype))
    views = build("qkv", (3 * rows, cols), entries, recipe, checkpoint)
    torch.testing.assert_close(
        Nvfp4Decoder._project(x, views), torch.cat(want, dim=1), atol=0, rtol=0
    )

    # A two-window row shard with 43 output features exercises zero-padding and
    # ensures the scales go through the row permutation before the 128x4 swizzle.
    raw_rows = 96
    selected = torch.cat((torch.arange(5, 24), torch.arange(44, 68))).to(device)
    code = torch.zeros((raw_rows, cols // 2), dtype=torch.uint8, device=device)
    channel = torch.arange(raw_rows, device=device)
    code[channel, channel // 2] = (2 << (4 * (channel % 2))).to(torch.uint8)
    scale = 0.5 + ((channel[:, None] + torch.arange(cols // 16, device=device)) % 5) / 4
    inp_scale = torch.tensor(0.125, device=device)
    checkpoint = {
        "in.weight": code,
        "in.input_scale": inp_scale,
        "in.weight_scale": scale.to(torch.float8_e4m3fn),
        "in.weight_scale_2": torch.tensor(0.75, device=device),
    }
    entries = [
        ("in.weight", (slice(5, 24),), (slice(0, 19),), None),
        ("in.weight", (slice(44, 68),), (slice(19, 43),), None),
    ]
    (view,) = build(
        "gdp_in",
        (43, cols),
        entries,
        {"in": SimpleNamespace(quant_algo="NVFP4", group_size=16)},
        checkpoint,
    )
    packed, sf = fp4_quantize(x, inp_scale.reciprocal(), 16)
    dequantized = nvfp4_gemm._dequantize(packed, sf) * inp_scale
    want = dequantized[:, selected] * scale[selected, selected // 16] * 0.75
    torch.testing.assert_close(view.apply(x), want.to(x.dtype), rtol=0, atol=0)

    # An idle attention-DP rank still reaches the MoE collective with its
    # actual local width.
    for projection, width in ((views[0], rows), (views, 3 * rows), (view, 43)):
        got = Nvfp4Decoder._project(x[:0], projection)
        assert got.shape == (0, width) and got.dtype == x.dtype and got.device == x.device


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
    reason="masked grouped NVFP4 requires Blackwell SM100/SM103",
)
def test_expert_graph_replay_matches_eager_after_routing_change():
    from tensorrt_llm._torch.modeling_v2.models.nemotron_shortcut.nvfp4__sm_103__default.ops import (
        Nvfp4Projection,
    )

    # 1024 scratch rows / 37 tokens would give a 27-expert first batch.
    # Its successor's FP32 alpha would have a 108-byte offset.
    device = torch.device("cuda")
    tokens, experts, width = 37, 32, 128
    channel = torch.arange(width, device=device)
    packed = torch.zeros((experts, width, width // 2), device=device, dtype=torch.uint8)
    packed[:, channel, channel // 2] = (2 << (4 * (channel % 2))).to(torch.uint8)
    block = torch.full((experts, width * width // 16), 0x38, device=device, dtype=torch.uint8)
    scales = 0.125 + torch.arange(experts, device=device) / 256
    weight_scale = 1 + (torch.arange(experts, device=device) % 5) / 8
    mat = Nvfp4Projection(packed, block, scales.reciprocal(), scales * weight_scale, width)
    ids = torch.arange(tokens, device=device)[:, None] % experts
    ids = torch.cat((ids, (ids + 13) % experts), dim=1)
    scores = torch.tensor([0.7, 0.3], device=device).expand(tokens, 2).contiguous()
    x = (torch.arange(tokens * width, device=device).view(tokens, width) % 29 / 8).bfloat16()
    original = grouped_experts(x, ids, scores, mat, mat, 0, 16)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = grouped_experts(x, ids, scores, mat, mat, 0, 16)
    ids.copy_((ids + 7) % experts)
    expected = grouped_experts(x, ids, scores, mat, mat, 0, 16)
    graph.replay()
    assert not torch.equal(original, expected)
    torch.testing.assert_close(captured, expected, rtol=0, atol=0)
