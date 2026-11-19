# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""VDN geometry, checkpoint validation and independent attention/state oracles."""

import copy
import json
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from tensorrt_llm._torch.visual_gen.models.minimax_h3.vdn_attention import (
    VDNAlpha,
    VDNHybridAttention,
    VDNLayout,
    VDNLinearBranch,
    VDNShortConv,
)
from tensorrt_llm._torch.visual_gen.models.minimax_h3.vdn_checkpoint import merge_vdn_adapter

SPEC = {
    "anchor_frames": "both",
    "enable_softmax_gate": True,
    "softmax_attention": {"radius": 1, "chunk": 5},
    "linear_attention": {
        "delta_rule": "vdn_solve",
        "bridge": "alpha",
        "a_fp32": True,
        "enable_text_state": True,
        "linear_head_dim": 8,
        "short_conv": {"targets": ["k", "v"]},
    },
}


def test_layout_excludes_conditioning_video_rows() -> None:
    target = torch.cartesian_prod(torch.arange(17), torch.arange(2), torch.arange(3))
    positions = torch.cat((torch.zeros(8, 3), target))
    video = torch.cat((torch.tensor([3, 4]), torch.arange(8, len(positions))))
    layout = VDNLayout.from_indices(positions, video, torch.arange(3))
    assert (layout.video_start, layout.num_frames, layout.height, layout.width) == (8, 17, 2, 3)
    positions[-1, 2] = 100
    with pytest.raises(ValueError, match="rectangular|ordered"):
        VDNLayout.from_indices(positions, video, torch.arange(3))


def test_adapter_merges_sequentially_in_bf16_and_rejects_unknown_keys() -> None:
    a, b = torch.randn(2, 3), torch.randn(4, 2)
    original = torch.randn(4, 3)
    weights = {"transformer_blocks.0.attn.to_q.weight": original.clone()}
    adapter = {
        "transformer_blocks.0.attn.orig.to_q.lora_A.default.weight": a,
        "transformer_blocks.0.attn.orig.to_q.lora_B.default.weight": b,
    }
    merge_vdn_adapter(weights, adapter)
    expected = original.bfloat16() + (b @ a).bfloat16()
    torch.testing.assert_close(next(iter(weights.values())), expected, rtol=0, atol=0)
    with pytest.raises(ValueError, match="Incomplete"):
        merge_vdn_adapter(weights, {next(iter(adapter)): a})
    with pytest.raises(ValueError, match="Unrecognized"):
        merge_vdn_adapter(weights, {"unused": a})


def test_short_conv_matches_separable_conv3d() -> None:
    torch.manual_seed(3)
    conv = VDNShortConv(16, ("k",))
    x = torch.randn(7 * 2 * 3, 2, 8)
    actual = conv(x, "k", 7, 2, 3)
    kernel = conv.k_tm.weight[:, :, :, None, None] * conv.k_sp.weight[:, :, None]
    volume = x.reshape(7, 2, 3, 16).permute(3, 0, 1, 2).unsqueeze(0)
    expected = F.conv3d(volume, kernel, padding=2, groups=16)
    expected = expected.squeeze(0).permute(1, 2, 3, 0).reshape_as(x)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("text_seed", [False, True])
def test_linear_branch_matches_direct_joint_solves(text_seed: bool) -> None:
    torch.manual_seed(4)
    spec = copy.deepcopy(SPEC["linear_attention"])
    spec["short_conv"]["targets"] = []
    branch = VDNLinearBranch(12, 2, 8, spec)
    frames, rows = 7, 3
    layout = VDNLayout(2, frames, 1, rows, torch.arange(2))
    x = torch.randn(frames * rows, 12)
    qkv = tuple(torch.randn(frames * rows, 2, 8) for _ in range(3))
    tx = torch.randn(2, 12) if text_seed else None
    tqkv = tuple(torch.randn(2, 2, 8) for _ in range(3)) if text_seed else None
    bounds = [(f - 1, f + 1) for f in range(frames)]
    actual = branch(x, qkv, layout, bounds, tx, tqkv)
    inner_x = x[rows:-rows]
    q, k, v = (F.silu(t[rows:-rows]) for t in qkv)
    q, k = (F.normalize(t, dim=-1, eps=1e-6) for t in (q, k))
    q, k, v = (t.reshape(frames - 2, rows, 2, 8) for t in (q, k, v))
    beta = branch.beta_proj(inner_x).sigmoid().reshape(frames - 2, rows, 2)
    alpha = branch.alpha(inner_x.reshape(frames - 2, rows, 12).mean(1))
    eye = torch.eye(8).expand(2, 8, 8)
    a = torch.einsum("fshk,fsh,fshl->fhkl", k, beta, k)
    b = torch.einsum("fshv,fsh,fshk->fhvk", v, beta, k)
    seed = torch.zeros(2, 8, 8)
    if text_seed:
        kt = F.normalize(F.silu(tqkv[1]), dim=-1, eps=1e-6)
        vt = F.silu(tqkv[2])
        bt = branch.beta_proj(tx).sigmoid()
        at = torch.einsum("shk,sh,shl->hkl", kt, bt, kt)
        bt = torch.einsum("shv,sh,shk->hvk", vt, bt, kt)
        seed = torch.linalg.solve(eye + at, bt, left=False) * 0.5
    expected = []
    for frame in range(frames - 2):
        lo, hi = frame - 1, frame + 1
        before, after = seed, seed
        for j in range(max(lo, 0)):
            before = torch.linalg.solve(
                eye + a[j], before * alpha[j].unsqueeze(-2) + b[j], left=False
            )
        for j in reversed(range(min(hi + 1, frames - 2), frames - 2)):
            after = torch.linalg.solve(
                eye + a[j], after * alpha[j].unsqueeze(-2) + b[j], left=False
            )
        for j in range(max(lo, 0), frame + 1):
            before = before * alpha[j].unsqueeze(-2)
        for j in range(frame, min(hi + 1, frames - 2)):
            after = after * alpha[j].unsqueeze(-2)
        expected.append(torch.einsum("hvk,shk->shv", before + after, q[frame]))
    expected = torch.stack(expected).reshape(-1, 2, 8)
    expected = F.rms_norm(expected, (8,), branch.norm.weight, eps=1e-6) * branch.output_gate(
        inner_x
    )
    expected = F.pad(expected.flatten(1), (0, 0, rows, rows))
    torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-5)
    assert torch.count_nonzero(actual[:rows]) == torch.count_nonzero(actual[-rows:]) == 0


def test_alpha_preserves_fp32_under_autocast() -> None:
    alpha = VDNAlpha(12, 2, 8).bfloat16()
    x = torch.randn(4, 12)
    expected = alpha(x)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual = alpha(x)
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_vdn_checkpoint_detection(tmp_path: Path) -> None:
    from tensorrt_llm._torch.visual_gen.pipeline_registry import AutoPipeline

    path = tmp_path / "stage-dmd-step-250/diffusers"
    path.mkdir(parents=True)
    (path / "config.json").write_text(
        json.dumps({"_class_name": "VDNMiniMaxH3Transformer3DModel", "vdn": {}})
    )
    (tmp_path / "modular_model_index.json").write_text(
        json.dumps({"_class_name": "MiniMaxH3ModularPipeline"})
    )
    assert AutoPipeline._detect_from_checkpoint(str(tmp_path)) == "VDNMiniMaxH3Transformer3DModel"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="H3 QK normalization requires CUDA")
@pytest.mark.parametrize("frames", [1, 2, 7, 17])
@pytest.mark.parametrize("backend", ["VANILLA", "FA4"])
def test_window_attention_matches_dense_boolean_mask(frames: int, backend: str) -> None:
    from test_minimax_h3_transformer import _initialize_weights, _make_model_config

    from tensorrt_llm._torch.visual_gen.models.minimax_h3.transformer_minimax_h3 import (
        MiniMaxH3Attention,
    )

    if backend == "FA4" and torch.cuda.get_device_capability()[0] not in (9, 10, 11):
        pytest.skip("FA4 requires a supported GPU architecture")
    dim = 128 if backend == "FA4" else 8
    spec = copy.deepcopy(SPEC)
    spec["linear_attention"]["linear_head_dim"] = dim
    config = _make_model_config()
    config.attention.backend = backend
    orig = MiniMaxH3Attention(
        hidden_size=12,
        num_attention_heads=2,
        attention_head_dim=dim,
        qk_norm_eps=1e-5,
        model_config=config,
        layer_idx=0,
        module_name="attn",
    )
    attn = VDNHybridAttention(orig, 12, spec).cuda()
    attn.linear_attention.bfloat16()
    attn.softmax_gate.bfloat16()
    attn.to_out_linear.bfloat16()
    _initialize_weights(attn)
    with torch.no_grad():
        attn.to_out_linear.weight.zero_()
    rows, start = 3, 5
    length = start + frames * rows
    attn.layout = VDNLayout(start, frames, 1, rows, torch.arange(3, device="cuda"))
    x = torch.randn(1, length, 12, dtype=torch.bfloat16, device="cuda")
    raw = [t.view(1, length, 2, dim) for t in orig.get_qkv(x)]
    q, k = orig.apply_qk_norm(raw[0], raw[1])
    idx = torch.arange(length, device="cuda")
    frame = (idx - start) // rows
    chunk = frame // 5
    allowed = (
        (idx[:, None] < start)
        | (idx[None, :] < start)
        | ((chunk[:, None] - chunk[None, :]).abs() <= 1)
        | (frame[:, None] == 0)
        | (frame[:, None] == frames - 1)
        | (frame[None, :] == 0)
        | (frame[None, :] == frames - 1)
    )
    expected = F.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), raw[2].transpose(1, 2), attn_mask=allowed
    )
    expected = expected.transpose(1, 2) * attn.softmax_gate(x[0]).unsqueeze(0)
    expected = orig.to_out[0](expected.flatten(2))
    with torch.no_grad():
        actual = attn(x)
    torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.002)


def test_vdn_supports_meta_initialization() -> None:
    from test_minimax_h3_transformer import _make_model_config

    from tensorrt_llm._torch.models.modeling_utils import MetaInitMode
    from tensorrt_llm._torch.visual_gen.models.minimax_h3.transformer_vdn_h3 import (
        VDNMiniMaxH3Transformer3DModel,
    )

    with MetaInitMode():
        model = VDNMiniMaxH3Transformer3DModel(_make_model_config(num_layers=1, vdn_transform=SPEC))
    # MetaInit deliberately keeps constant-initialized vectors on the CPU.
    assert all(p.is_meta for p in model.parameters() if p.ndim > 1)
    assert (
        model.transformer_blocks[0].attn.linear_attention.beta_proj.weight.dtype == torch.bfloat16
    )


@pytest.mark.parametrize(
    ("stage", "grid_points"), [("stage-dmd-step-250", 9), ("stage-b-step-2000", 51)]
)
def test_vdn_default_sigma_grid(stage: str, grid_points: int) -> None:
    from tensorrt_llm._torch.visual_gen.models.minimax_h3.pipeline_vdn_h3 import (
        VDNMiniMaxH3Pipeline,
    )

    pipeline = object.__new__(VDNMiniMaxH3Pipeline)
    pipeline.stage = stage
    assert pipeline.default_generation_params["num_inference_steps"] == grid_points


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_vdn_noise_matches_upstream_device_generator() -> None:
    from tensorrt_llm._torch.visual_gen.models.minimax_h3.pipeline_vdn_h3 import (
        VDNMiniMaxH3Pipeline,
    )

    pipeline = object.__new__(VDNMiniMaxH3Pipeline)
    pipeline._device = torch.device("cuda")
    expected = torch.randn(64, device="cuda", generator=torch.Generator("cuda").manual_seed(42))
    actual = torch.randn(64, device="cuda", generator=pipeline._request_generator(42))
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("frames", [1, 7, 17])
@pytest.mark.parametrize(("activate", "l2norm"), [(False, False), (True, False), (True, True)])
def test_temporal_kernel_matches_fp32_conv1d(frames: int, activate: bool, l2norm: bool) -> None:
    from tensorrt_llm._torch.visual_gen.models.minimax_h3.vdn_kernels import temporal_conv

    torch.manual_seed(11)
    x = torch.randn(frames, 6, 32, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(32, 5, device="cuda", dtype=torch.bfloat16)
    actual = temporal_conv(x, weight, 16, activate=activate, l2norm=l2norm)
    expected = F.conv1d(
        x.float().permute(1, 2, 0), weight.float().unsqueeze(1), padding=2, groups=32
    ).permute(2, 0, 1)
    if activate:
        expected = F.silu(expected)
        if l2norm:
            expected = F.normalize(expected.reshape(frames, 6, 2, 16), dim=-1, eps=1e-6).reshape_as(
                x
            )
    torch.testing.assert_close(actual, expected.to(x.dtype), rtol=0.008, atol=0.002)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_fused_partial_rope_matches_fp32_reference() -> None:
    from tensorrt_llm._torch.visual_gen.models.minimax_h3.transformer_minimax_h3 import (
        apply_minimax_h3_rotary_emb,
    )
    from tensorrt_llm._torch.visual_gen.models.minimax_h3.vdn_attention import (
        _compiled_qk_norm_rope,
    )

    torch.manual_seed(12)
    packed = torch.randn(1, 19, 6, 16, device="cuda", dtype=torch.bfloat16)
    q = packed[:, :, :2]
    weight = torch.randn(16, device="cuda", dtype=torch.bfloat16)
    angles = torch.randn(19, 6, device="cuda").repeat(1, 2)
    cos, sin = angles.cos(), angles.sin()
    expected = apply_minimax_h3_rotary_emb(
        F.rms_norm(q.float(), (16,), weight.float(), 1e-5), cos, sin
    ).to(q.dtype)
    actual = _compiled_qk_norm_rope(q, weight, 1e-5, cos, sin)
    torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.02)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_fused_modulation_matches_fp32_reference() -> None:
    from tensorrt_llm._torch.visual_gen.models.minimax_h3.transformer_vdn_h3 import (
        _compiled_modulate,
        _compiled_residual,
    )

    torch.manual_seed(13)
    hidden = torch.randn(1, 19, 128, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(128, device="cuda", dtype=torch.bfloat16)
    scale, shift, gate = (
        torch.randn(3, 128, device="cuda", dtype=torch.bfloat16) for _ in range(3)
    )
    indices = torch.arange(19, device="cuda") % 3
    expected = F.rms_norm(hidden.float(), (128,), weight.float(), 1e-6)
    expected = (expected * (1 + scale.float()[indices]) + shift.float()[indices]).bfloat16()
    actual = _compiled_modulate(hidden, weight, 1e-6, scale, shift, indices)
    torch.testing.assert_close(actual, expected, rtol=0.008, atol=0.008)
    output = _compiled_residual(hidden, actual, gate, indices)
    expected_output = (hidden.float() + gate.float()[indices] * actual.float()).bfloat16()
    torch.testing.assert_close(output, expected_output, rtol=0, atol=0)
