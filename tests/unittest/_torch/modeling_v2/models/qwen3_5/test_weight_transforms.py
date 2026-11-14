# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The Qwen3.8-27B-NVFP4 target's checkpoint mapping, on the CPU with synthetic tensors.

* **manifest**: on a small config with the checkpoint's layer pattern, the table reads
  exactly the checkpoint names the ModelOpt MIXED_PRECISION export carries (per layer kind),
  and declares one parameter per manifest row;
* **transforms**: each load-time transform on small tensors, against the arithmetic it
  documents -- q/k/v and qkvz rescaled onto the larger weight scale, `[b ; a]`,
  `[gate ; up]` and its scale blocks concatenated before the swizzle, `1 / input_scale`,
  `alpha = input_scale * weight_scale_2`, and the refusals of shards that disagree;
* **load**: a whole synthetic checkpoint (plus MTP and vision keys) fills every declared
  parameter with the declared shape and dtype and covers the checkpoint.
"""

from __future__ import annotations

import re
import zlib
from types import SimpleNamespace

import pytest
import torch
from transformers import PretrainedConfig

from tensorrt_llm._torch._experimental.modeling_v2.models.qwen3_5.qwen3_8_27b_nvfp4__sm_120__tp1 import (  # noqa: E501
    weights as WT,
)

pytestmark = pytest.mark.cpu_only

e4m3 = torch.float8_e4m3fn
_LM = "model.language_model"

#: A 4-layer model with the checkpoint's pattern (every 4th layer full attention) and
#: widths small enough to build every tensor.
_DIMS = dict(
    num_hidden_layers=4,
    layer_types=["linear_attention"] * 3 + ["full_attention"],
    full_attention_interval=4,
    hidden_size=64,
    intermediate_size=64,
    vocab_size=32,
    num_attention_heads=2,
    num_key_value_heads=1,
    head_dim=16,
    linear_num_key_heads=2,
    linear_key_head_dim=16,
    linear_num_value_heads=4,
    linear_value_head_dim=16,
    linear_conv_kernel_dim=4,
)
_ATTN, _GDN = (3,), (0, 1, 2)


def _core():
    cfg = PretrainedConfig(**_DIMS)
    return SimpleNamespace(
        model_config=SimpleNamespace(pretrained_config=cfg, torch_dtype=torch.bfloat16)
    )


def _expected_checkpoint_keys() -> set:
    """The names the export carries (hf_quant_config.json quantized_layers plus the bf16
    tensors), for `_DIMS`."""
    keys = {
        f"{_LM}.embed_tokens.weight",
        f"{_LM}.norm.weight",
        *(f"lm_head.{s}" for s in ("weight", "weight_scale", "weight_scale_2", "input_scale")),
    }
    fp8 = ("weight", "weight_scale", "input_scale")
    nvfp4 = ("weight", "weight_scale", "input_scale", "weight_scale_2")
    for i in range(_DIMS["num_hidden_layers"]):
        p = f"{_LM}.layers.{i}"
        keys |= {f"{p}.input_layernorm.weight", f"{p}.post_attention_layernorm.weight"}
        keys |= {f"{p}.mlp.{m}.{s}" for m in ("gate_proj", "up_proj", "down_proj") for s in nvfp4}
        if i in _ATTN:
            keys |= {f"{p}.self_attn.{m}_proj.{s}" for m in "qkvo" for s in fp8}
            keys |= {f"{p}.self_attn.{n}_norm.weight" for n in "qk"}
        else:
            keys |= {
                f"{p}.linear_attn.{m}.{s}"
                for m in ("in_proj_qkv", "in_proj_z", "out_proj")
                for s in fp8
            }
            keys |= {
                f"{p}.linear_attn.{t}"
                for t in (
                    "in_proj_a.weight",
                    "in_proj_b.weight",
                    "conv1d.weight",
                    "A_log",
                    "dt_bias",
                    "norm.weight",
                )
            }
    return keys


# ------------------------------------------------------------------ manifest ---


def test_manifest_reads_exactly_the_export_names():
    table = WT.Qwen3827BTp1Weights()
    manifest = table.manifest(_core())
    read = {k for keys, _, _ in manifest.values() for k in keys}
    assert read == _expected_checkpoint_keys(), sorted(read ^ _expected_checkpoint_keys())


def test_manifest_rows_are_the_declared_parameters():
    table = WT.Qwen3827BTp1Weights()
    core = _core()
    declared = table.declare(table.dims(core))
    assert set(table.manifest(core)) == set(declared.keys())
    # Per layer kind: attention parameters only on layer 3, Gated-DeltaNet ones on 0..2.
    assert {k for k in declared if k.endswith("_qkv_w")} == {"l3_qkv_w"}
    assert {k for k in declared if k.endswith("_qkvz_w")} == {f"l{i}_qkvz_w" for i in _GDN}


def test_the_mtp_and_vision_keys_are_the_only_unconsumed_ones():
    table = WT.Qwen3827BTp1Weights()
    weights = dict.fromkeys(
        [*_expected_checkpoint_keys(), "mtp.fc.weight", "model.visual.patch_embed.proj.weight"]
    )
    assert table.expected_unconsumed(weights) == {
        "mtp.fc.weight",
        "model.visual.patch_embed.proj.weight",
    }


# ---------------------------------------------------------------- transforms ---


def _fp8(rows, cols, seed):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(rows, cols, generator=g) * 50).to(e4m3)


def _bytes(t):
    return t.contiguous().view(torch.uint8)


@pytest.mark.parametrize("z_larger", [True, False])
def test_qkvz_rescales_the_smaller_scaled_half_in_fp32(z_larger):
    qkv, z = _fp8(6, 8, 1), _fp8(4, 8, 2)
    small, large = torch.tensor(0.5), torch.tensor(2.0)
    qkv_ws, z_ws = (small, large) if z_larger else (large, small)
    out = WT._qkvz_weight((qkv, z, qkv_ws, z_ws))
    assert out.dtype == e4m3 and out.shape == (10, 8)
    rescaled = lambda w: (w.to(torch.float32) * 0.25).to(e4m3)  # noqa: E731 - 0.5 / 2.0
    want_qkv, want_z = (rescaled(qkv), z) if z_larger else (qkv, rescaled(z))
    assert torch.equal(_bytes(out[:6]), _bytes(want_qkv))
    assert torch.equal(_bytes(out[6:]), _bytes(want_z))
    # The fused scale is the larger one.
    assert WT._max_scalar((qkv_ws, z_ws)).item() == 2.0
    assert WT._max_scalar((qkv_ws, z_ws)).shape == ()


def test_qkvz_with_equal_scales_is_the_checkpoint_bytes():
    qkv, z = _fp8(6, 8, 3), _fp8(4, 8, 4)
    s = torch.tensor(0.75)
    out = WT._qkvz_weight((qkv, z, s, s.clone()))
    assert torch.equal(_bytes(out), _bytes(torch.cat([qkv, z], 0)))


def test_qkv_rescales_every_shard_through_bf16_onto_the_largest_scale():
    q, k, v = _fp8(8, 8, 5), _fp8(4, 8, 6), _fp8(4, 8, 7)
    scales = (torch.tensor(0.25), torch.tensor(1.5), torch.tensor(0.5))
    out = WT._qkv_weight((q, k, v, *scales))
    assert out.dtype == e4m3 and out.shape == (16, 8)
    smax = torch.tensor(1.5)
    want = [
        w.to(torch.bfloat16).mul_(s.to(torch.float32)).div_(smax).to(e4m3)
        for w, s in zip((q, k, v), scales)
    ]
    assert torch.equal(_bytes(out), _bytes(torch.cat(want, 0)))
    # In value terms: each shard keeps weight * scale (within e4m3 and bf16 rounding).
    deq = out.to(torch.float32) * smax
    ref = torch.cat([w.to(torch.float32) * s for w, s in zip((q, k, v), scales)], 0)
    # e4m3 keeps 3 mantissa bits (half-step 1/16 relative); subnormals near zero get an
    # absolute floor of a few of their 2^-9 steps at this scale.
    torch.testing.assert_close(deq, ref, rtol=0.07, atol=0.01)


def test_ba_is_b_then_a_and_gate_up_is_gate_then_up():
    b, a = torch.full((4, 8), 1.0), torch.full((4, 8), 2.0)
    assert torch.equal(WT._ba_weight((b, a)), torch.cat([b, a], 0))
    gate = torch.full((3, 4), 7, dtype=torch.uint8)
    up = torch.full((3, 4), 9, dtype=torch.uint8)
    assert torch.equal(WT._gate_up_weight((gate, up)), torch.cat([gate, up], 0))


def test_conv_weight_drops_the_unit_dim():
    w = torch.arange(24.0).reshape(6, 1, 4)
    out = WT._conv_weight(w)
    assert out.shape == (6, 4) and out.is_contiguous()
    assert torch.equal(out, w[:, 0, :])


def test_gate_up_scale_blocks_concatenate_before_the_swizzle():
    """Two 64-row blocks swizzle as one 128-row buffer (one row pad), not as two padded
    halves: the length is the fused weight's `_sf_bytes`."""
    rows, cols = 64, 64  # K = 64 -> 4 scale columns
    g = torch.Generator().manual_seed(8)
    gate_sf = torch.randint(0, 127, (rows, cols // WT.SF_VEC), generator=g, dtype=torch.uint8)
    up_sf = torch.randint(0, 127, (rows, cols // WT.SF_VEC), generator=g, dtype=torch.uint8)
    out = WT._swizzled_sf((gate_sf.view(e4m3), up_sf.view(e4m3)))
    assert out.dtype == torch.uint8 and out.dim() == 1
    assert out.numel() == WT._sf_bytes(2 * rows, cols) == 128 * 4
    want = torch.ops.trtllm.block_scale_interleave(torch.cat([gate_sf, up_sf], 0))
    assert torch.equal(out, want)


def test_act_scale_is_the_reciprocal_and_alpha_the_product():
    is_ = torch.tensor(0.004)
    ws2 = torch.tensor(0.125)
    act = WT._act_scale((is_, is_.clone()))
    assert act.shape == (1,) and act.dtype == torch.float32
    assert act.item() == pytest.approx(1.0 / 0.004, rel=1e-7)
    alpha = WT._alpha((is_, is_.clone(), ws2, ws2.clone()))
    assert alpha.shape == (1,) and alpha.dtype == torch.float32
    assert alpha.item() == pytest.approx(0.004 * 0.125, rel=1e-7)
    # A single (down_proj) shard takes the same path.
    assert WT._act_scale(is_).item() == act.item()
    assert WT._alpha((is_, ws2)).item() == alpha.item()


@pytest.mark.parametrize("which", ["input_scale", "weight_scale_2"])
def test_fused_shards_that_disagree_are_refused(which):
    a, b = torch.tensor(0.004), torch.tensor(0.008)
    if which == "input_scale":
        with pytest.raises(AssertionError, match="fused shards disagree"):
            WT._act_scale((a, b))
        with pytest.raises(AssertionError, match="fused shards disagree"):
            WT._alpha((a, b, a, a))
    else:
        with pytest.raises(AssertionError, match="fused shards disagree"):
            WT._alpha((a, a, a, b))


# ---------------------------------------------------------------------- load ---


def _source(key: str, d: SimpleNamespace) -> torch.Tensor:
    """A synthetic checkpoint tensor of the stored shape and dtype for `key`. gate_proj and
    up_proj share their scalars, as the export's fused pair must."""
    role = re.sub(r"(gate|up)_proj", "gate_up", key)
    g = torch.Generator().manual_seed(zlib.crc32(role.encode()))
    per_tensor = re.search(r"(self_attn|linear_attn)\.\w+\.weight_scale$", key)
    if key.endswith(("weight_scale_2", "input_scale")) or per_tensor:
        return torch.tensor(0.01 + torch.rand(1, generator=g).item())
    if key.endswith("lm_head.weight"):
        return torch.randint(0, 256, (d.vocab, d.hidden // 2), generator=g, dtype=torch.uint8)
    if key.endswith("lm_head.weight_scale"):
        return torch.rand(d.vocab, d.hidden // WT.SF_VEC, generator=g).to(e4m3)
    mlp = re.search(r"\.mlp\.(gate|up|down)_proj\.(weight|weight_scale)$", key)
    if mlp:
        rows, cols = (d.hidden, d.inter) if mlp.group(1) == "down" else (d.inter, d.hidden)
        if mlp.group(2) == "weight":
            return torch.randint(0, 256, (rows, cols // 2), generator=g, dtype=torch.uint8)
        return torch.rand(rows, cols // WT.SF_VEC, generator=g).to(e4m3)
    kv_width = _DIMS["num_key_value_heads"] * d.head_dim
    key_dim = _DIMS["linear_num_key_heads"] * _DIMS["linear_key_head_dim"]
    fp8_rows = {
        "self_attn.q_proj.weight": (2 * d.q_width, d.hidden),
        "self_attn.k_proj.weight": (kv_width, d.hidden),
        "self_attn.v_proj.weight": (kv_width, d.hidden),
        "self_attn.o_proj.weight": (d.hidden, d.q_width),
        "linear_attn.in_proj_qkv.weight": (2 * key_dim + d.value_dim, d.hidden),
        "linear_attn.in_proj_z.weight": (d.value_dim, d.hidden),
        "linear_attn.out_proj.weight": (d.hidden, d.value_dim),
    }
    for suffix, shape in fp8_rows.items():
        if key.endswith(suffix):
            return (torch.randn(*shape, generator=g) * 10).to(e4m3)
    bf16 = {
        "embed_tokens.weight": (d.vocab, d.hidden),
        "in_proj_a.weight": (d.num_v_heads, d.hidden),
        "in_proj_b.weight": (d.num_v_heads, d.hidden),
        "conv1d.weight": (d.conv_dim, 1, d.conv_kernel),
        "A_log": (d.num_v_heads,),
        "dt_bias": (d.num_v_heads,),
        "linear_attn.norm.weight": (d.head_v_dim,),
        "q_norm.weight": (d.head_dim,),
        "k_norm.weight": (d.head_dim,),
        "layernorm.weight": (d.hidden,),
        "language_model.norm.weight": (d.hidden,),
    }
    for suffix, shape in bf16.items():
        if key.endswith(suffix):
            return torch.randn(*shape, generator=g).to(torch.bfloat16)
    raise AssertionError(f"no synthetic source for {key}")


def test_load_fills_every_parameter_and_covers_the_checkpoint(monkeypatch):
    """`load` asserts each transformed source has its parameter's shape and dtype and that
    only the MTP and vision keys stay unread; then spot-check the stored values."""
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a, **k: None)
    table = WT.Qwen3827BTp1Weights()
    core = _core()
    d = table.dims(core)
    core.w = table.declare(d)
    weights = {k: _source(k, d) for k in _expected_checkpoint_keys()}
    weights["mtp.fc.weight"] = torch.zeros(2, 2)
    weights["model.visual.patch_embed.proj.weight"] = torch.zeros(2, 2)

    table.load(SimpleNamespace(model=core), weights)

    p = f"{_LM}.layers"
    gate_is = weights[f"{p}.0.mlp.gate_proj.input_scale"]
    gate_ws2 = weights[f"{p}.0.mlp.gate_proj.weight_scale_2"]
    assert core.w["l0_gate_up_is"].item() == pytest.approx(1.0 / gate_is.item(), rel=1e-6)
    assert core.w["l0_gate_up_alpha"].item() == pytest.approx(
        gate_is.item() * gate_ws2.item(), rel=1e-6
    )
    down_is = weights[f"{p}.0.mlp.down_proj.input_scale"]
    down_ws2 = weights[f"{p}.0.mlp.down_proj.weight_scale_2"]
    assert core.w["l0_down_is"].item() == pytest.approx(1.0 / down_is.item(), rel=1e-6)
    assert core.w["l0_down_alpha"].item() == pytest.approx(
        down_is.item() * down_ws2.item(), rel=1e-6
    )
    assert torch.equal(
        core.w["l1_ba_w"],
        torch.cat(
            [weights[f"{p}.1.linear_attn.in_proj_b.weight"],
             weights[f"{p}.1.linear_attn.in_proj_a.weight"]], 0
        ),
    )  # fmt: skip
    assert core.w["l3_qkv_is"].item() == max(
        weights[f"{p}.3.self_attn.{n}_proj.input_scale"].item() for n in "qkv"
    )
