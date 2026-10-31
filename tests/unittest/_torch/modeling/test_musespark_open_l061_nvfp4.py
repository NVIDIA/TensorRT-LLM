# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""NVFP4 experts for the layers the checkpoint stores in BF16 (MUSESPARK_OPEN_L061_EXPERTS_NVFP4).

The CPU tests pin the exclusion rewrite and the FC1 activation bound. The GPU
tests check the load-time quantizer against the dequantization the checkpoint's
quantized layers use, and the whole NVFP4 MoE layer against the BF16 one it
replaces. With MSO_CHECKPOINT_DIR (the NVFP4 checkpoint) they use the real
layer 0 / 61 experts; with MSO_L061_DUMP_DIR as well (saved expert inputs: files
holding ``expert_input`` and ``router_logits``) they use real expert inputs and report
the FC2-input amax those inputs reach.
"""

import glob
import json
import os

import pytest
import torch

from tensorrt_llm._torch.models.checkpoints.hf.musespark_open_weight_mapper import (
    NVFP4_GLOBAL_RANGE,
    WeightAccounting,
    build_expert_weights,
    build_nvfp4_expert_weights_from_bf16,
    nvfp4_quantize_weight,
    pre_expert_norm_amax,
)
from tensorrt_llm._torch.models.modeling_musespark_open import (
    normalize_exclude_modules,
    unexclude_layer_experts,
)
from tensorrt_llm.models.modeling_utils import QuantAlgo, QuantConfig

HIDDEN, INTERMEDIATE, EXPERTS, TOP_K = 4096, 4096, 256, 8
CHECKPOINT = os.environ.get("MSO_CHECKPOINT_DIR", "")
DUMPS = os.environ.get("MSO_L061_DUMP_DIR", "")
_PREFIX = "model.language_model.layers.{layer}.mlp."


def _checkpoint_reader():
    from safetensors import safe_open

    index = json.load(open(os.path.join(CHECKPOINT, "model.safetensors.index.json")))
    weight_map = index["weight_map"]

    def read(key):
        with safe_open(os.path.join(CHECKPOINT, weight_map[key]), framework="pt") as handle:
            return handle.get_tensor(key)

    return read


def test_unexclude_layer_experts_keeps_attention_excluded():
    config = QuantConfig(
        quant_algo=QuantAlgo.NVFP4,
        exclude_modules=normalize_exclude_modules(
            ["layers.0.", "layers.61.", "self_attn", "lm_head"]
        ),
    )
    assert config.is_module_excluded_from_quantization("model.layers.0.feed_forward.experts")
    assert unexclude_layer_experts(config, 62) == [0, 61]
    assert config.exclude_modules == ["*self_attn*", "lm_head"]
    for layer in (0, 1, 61):
        assert not config.is_module_excluded_from_quantization(
            f"model.layers.{layer}.feed_forward.experts"
        )
        assert config.is_module_excluded_from_quantization(f"model.layers.{layer}.self_attn")
    assert unexclude_layer_experts(config, 62) == []


def test_unexclude_layer_experts_one_layer():
    config = QuantConfig(
        quant_algo=QuantAlgo.NVFP4,
        exclude_modules=["*layers.0.*", "*layers.61.*", "*self_attn*", "lm_head"],
    )
    assert unexclude_layer_experts(config, 62, [61]) == [61]
    assert config.exclude_modules == ["*layers.0.*", "*self_attn*", "lm_head"]
    assert config.is_module_excluded_from_quantization("model.layers.0.feed_forward.experts")


def test_unexclude_layer_experts_refuses_to_requantize_attention():
    config = QuantConfig(quant_algo=QuantAlgo.NVFP4, exclude_modules=["*layers.0.*", "lm_head"])
    with pytest.raises(ValueError, match="format of more than"):
        unexclude_layer_experts(config, 62)


def test_pre_expert_norm_amax_bounds_every_input():
    torch.manual_seed(5)
    weight = torch.randn(HIDDEN).to(torch.bfloat16)
    bound = pre_expert_norm_amax(weight, HIDDEN)
    gain = (weight.float() + 1.0).to(torch.bfloat16).float()
    x = torch.randn(64, HIDDEN) * torch.rand(64, 1) * 10
    x[0] = 0.0
    x[0, int(gain.abs().argmax())] = 300.0
    y = x / torch.sqrt(x.pow(2).mean(-1, keepdim=True) + 1e-5) * gain
    assert y.abs().max() <= bound
    # A one-hot input on the largest gain reaches the bound.
    assert y[0].abs().max() >= bound * 0.999


def _dequantize(packed, block_scale, weight_scale_2):
    return torch.ops.tensorrt_llm.e2m1_and_ufp8sf_scale_to_float_v2(
        packed.cpu(),
        block_scale.view(torch.uint8).flatten().cpu(),
        weight_scale_2.float().cpu().reshape(1),
        16,
        1,
        False,
    )


_E2M1_MAGNITUDES = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
_E2M1 = torch.tensor(_E2M1_MAGNITUDES + [-v for v in _E2M1_MAGNITUDES])


def _dequantize_cuda(packed, block_scale, weight_scale_2):
    """GPU dequantization of :func:`nvfp4_quantize_weight`'s output (low nibble first)."""
    packed = packed.cuda()
    table = _E2M1.cuda()
    values = torch.stack((table[(packed & 0xF).long()], table[(packed >> 4).long()]), dim=-1)
    values = values.reshape(packed.shape[0], -1)
    scale = block_scale.cuda().float().repeat_interleave(16, dim=-1) * weight_scale_2.float().cuda()
    return values * scale


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_nvfp4_quantize_weight_round_trip():
    torch.manual_seed(7)
    weight = (torch.randn(512, HIDDEN) * 0.02).to(torch.bfloat16)
    weight[3, 17] = 0.9
    packed, block_scale, weight_scale_2 = nvfp4_quantize_weight(weight)
    assert packed.dtype == torch.uint8 and packed.shape == (512, HIDDEN // 2)
    assert block_scale.dtype == torch.float8_e4m3fn and block_scale.shape == (512, HIDDEN // 16)
    assert weight_scale_2.shape == () and weight_scale_2.dtype == torch.float32
    torch.testing.assert_close(
        weight_scale_2, weight[3, 17].float() / NVFP4_GLOBAL_RANGE, rtol=1e-6, atol=0.0
    )
    restored = _dequantize(packed, block_scale, weight_scale_2)
    relative = (restored - weight.float()).norm() / weight.float().norm()
    print(f"random weight round trip relative_l2={relative.item():.4f}")
    assert relative < 0.15
    # The amax element maps onto E2M1 6 x E4M3 448 exactly.
    assert abs(restored[3, 17].item() / weight[3, 17].float().item() - 1.0) < 1e-6
    torch.testing.assert_close(
        _dequantize_cuda(packed, block_scale, weight_scale_2).cpu(), restored, rtol=0.0, atol=0.0
    )


@pytest.mark.skipif(
    not torch.cuda.is_available() or not CHECKPOINT, reason="CUDA and MSO_CHECKPOINT_DIR required"
)
def test_nvfp4_quantize_weight_matches_checkpoint_layout():
    """Re-quantizing a dequantized checkpoint weight reproduces it.

    The values of a dequantized NVFP4 tensor are on that tensor's own grid, so a
    quantizer with the same nibble order, scale layout and rounding maps them
    back almost exactly; a swapped nibble order or a transposed scale layout
    would not.
    """
    read = _checkpoint_reader()
    prefix = _PREFIX.format(layer=1) + "experts."
    packed = read(prefix + "gate_up_proj")[5, :INTERMEDIATE]
    block_scale = read(prefix + "gate_up_proj_weight_scale")[5, :INTERMEDIATE]
    weight_scale_2 = read(prefix + "gate_up_proj_weight_scale_2")[5, 0]
    original = _dequantize(packed, block_scale, weight_scale_2)
    again = nvfp4_quantize_weight(original.to(torch.bfloat16))
    restored = _dequantize(*again)
    relative = (restored - original).norm() / original.norm()
    same_bytes = (again[0].cpu() == packed).float().mean()
    print(
        f"layer 1 expert 5 w1 re-quantized: relative_l2={relative.item():.3g} "
        f"identical_bytes={same_bytes.item():.4f}"
    )
    assert relative < 2e-2
    assert same_bytes > 0.9


def _layer_tensors(read, layer):
    mlp = _PREFIX.format(layer=layer)
    return {
        "w13": read(mlp + "experts.gate_up_proj"),
        "w2": read(mlp + "experts.down_proj"),
        "post_norm": read(mlp + "experts.post_expert_norm.weight"),
        "pre_norm": read(mlp + "pre_expert_norm.weight"),
        "bias": read(mlp + "gate.e_score_correction_bias"),
    }


def _inputs(layer, gain, count):
    """Real (expert input, router logits) calls of ``layer``, else synthetic ones."""
    calls = []
    for path in sorted(glob.glob(os.path.join(DUMPS, f"layer{layer}_call*.pt")))[:count]:
        record = torch.load(path)
        calls.append((record["expert_input"], record["router_logits"], os.path.basename(path)))
    if calls or DUMPS:
        return calls
    torch.manual_seed(11 + layer)
    x = torch.randn(2048, HIDDEN)
    x = x / torch.sqrt(x.pow(2).mean(-1, keepdim=True)) * gain
    return [(x.to(torch.bfloat16), torch.randn(2048, EXPERTS) * 2, "synthetic")]


def _make_moe(bias, quantized):
    from transformers import PretrainedConfig

    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm._torch.models.modeling_musespark_open import MuseSparkOpenRoutingMethod
    from tensorrt_llm._torch.moe.fused_moe import create_moe
    from tensorrt_llm.mapping import Mapping

    config = ModelConfig(
        pretrained_config=PretrainedConfig(
            hidden_size=HIDDEN,
            intermediate_size=INTERMEDIATE,
            num_experts=EXPERTS,
            torch_dtype=torch.bfloat16,
        ),
        mapping=Mapping(),
        moe_backend="TRTLLM",
    )
    routing = MuseSparkOpenRoutingMethod(
        top_k=TOP_K,
        num_experts=EXPERTS,
        callable_e_score_correction_bias=lambda: bias,
        renorm_eps=1e-15,
    )
    return create_moe(
        routing_method=routing,
        num_experts=EXPERTS,
        hidden_size=HIDDEN,
        intermediate_size=INTERMEDIATE,
        dtype=torch.bfloat16,
        reduce_results=False,
        model_config=config,
        override_quant_config=QuantConfig(quant_algo=QuantAlgo.NVFP4) if quantized else None,
        layer_idx=0,
    ).cuda()


def _fc2_input_amax(w13, x, selected):
    """max |silu(x w1^T) * (x w3^T)| over the (token, selected expert) pairs."""
    amax = torch.zeros(EXPERTS)
    for expert in torch.unique(selected).tolist():
        rows = (selected == expert).any(-1).nonzero().flatten()
        h = x[rows] @ w13[expert].T
        act = torch.nn.functional.silu(h[:, :INTERMEDIATE].float()) * h[:, INTERMEDIATE:].float()
        amax[expert] = act.abs().max().item()
    return amax


@pytest.mark.skipif(
    not torch.cuda.is_available() or not CHECKPOINT, reason="CUDA and MSO_CHECKPOINT_DIR required"
)
@pytest.mark.parametrize("layer", [0, 61])
@torch.inference_mode()
def test_excluded_layer_nvfp4_moe_matches_bf16(layer):
    """The NVFP4 MoE built from the BF16 experts, against the BF16 MoE, on one GPU."""
    from tensorrt_llm._torch.models.modeling_musespark_open import post_expert_norm_combine

    read = _checkpoint_reader()
    tensors = _layer_tensors(read, layer)
    gain = (tensors["pre_norm"].float() + 1.0).to(torch.bfloat16).float()
    fc1_amax = pre_expert_norm_amax(tensors["pre_norm"], HIDDEN)
    calls = _inputs(layer, gain, int(os.environ.get("MSO_L061_MAX_CALLS", "64")))
    assert calls, f"no layer {layer} dumps under {DUMPS}"
    bias = tensors["bias"].float().cuda()
    post_norm = tensors["post_norm"].cuda()
    checkpoint = {"e.w13_weight": tensors["w13"], "e.w2_weight": tensors["w2"]}

    bf16 = _make_moe(bias, quantized=False)
    bf16.load_weights(
        [
            build_expert_weights(
                checkpoint, "e", 0, EXPERTS, EXPERTS, INTERMEDIATE, False, WeightAccounting()
            )
        ]
    )
    bf16.post_load_weights()

    # Calibration: the FC2-input amax these inputs reach, before choosing it.
    w13_gpu = tensors["w13"].cuda()
    observed_fc1, observed_fc2 = 0.0, torch.zeros(EXPERTS)
    for x, logits, name in calls:
        x, logits = x.cuda(), logits.cuda()
        selected, _ = bf16.routing_method.apply(logits)
        per_expert = _fc2_input_amax(w13_gpu, x, selected)
        observed_fc2 = torch.maximum(observed_fc2, per_expert)
        observed_fc1 = max(observed_fc1, x.float().abs().max().item())
        distinct = torch.unique(x[:, :64].float(), dim=0).shape[0] / x.shape[0]
        print(
            f"layer {layer} {name}: rows={x.shape[0]} distinct_rows={distinct:.3f} "
            f"fc1_amax={x.float().abs().max().item():.4g} fc2_amax={per_expert.max().item():.4g}"
        )
    print(
        f"layer {layer} calibration over {len(calls)} calls: fc1 observed {observed_fc1:.4g} "
        f"(bound {fc1_amax:.4g}), fc2 observed max {observed_fc2.max().item():.4g}, "
        f"p50/p90/p99 over experts {observed_fc2.quantile(0.5).item():.4g}/"
        f"{observed_fc2.quantile(0.9).item():.4g}/{observed_fc2.quantile(0.99).item():.4g}"
    )
    assert observed_fc1 <= fc1_amax

    configured = os.environ.get("MUSESPARK_OPEN_L061_EXPERTS_NVFP4_FC2_AMAX", "")
    fc2_amax = {int(k): float(v) for k, v in (i.split("=") for i in configured.split(",") if i)}
    fc2_amax = fc2_amax.get(layer, 2.0 * observed_fc2.max().item())
    print(f"layer {layer} FC2 amax used: {fc2_amax:.4g}")
    assert fc2_amax >= observed_fc2.max().item()
    del w13_gpu

    def nvfp4_moe(fc1):
        moe = _make_moe(bias, quantized=True)
        moe.load_weights(
            [
                build_nvfp4_expert_weights_from_bf16(
                    checkpoint,
                    "e",
                    0,
                    EXPERTS,
                    EXPERTS,
                    INTERMEDIATE,
                    fc1,
                    fc2_amax,
                    WeightAccounting(),
                )
            ]
        )
        moe.post_load_weights()
        return moe

    # The served configuration, then two diagnostics: FC1 scaled from the observed amax
    # instead of the bound, and NVFP4 weights with BF16 activations (weight error only).
    variants = {"nvfp4": nvfp4_moe(fc1_amax)}
    if os.environ.get("MSO_L061_DIAGNOSTICS", "1") == "1":
        variants["nvfp4_fc1_observed"] = nvfp4_moe(2.0 * observed_fc1)
        dequantized = {}
        for expert in range(EXPERTS):
            for name, weight in (
                ("w1", tensors["w13"][expert, :INTERMEDIATE]),
                ("w3", tensors["w13"][expert, INTERMEDIATE:]),
                ("w2", tensors["w2"][expert]),
            ):
                dequantized[f"{expert}.{name}.weight"] = _dequantize_cuda(
                    *nvfp4_quantize_weight(weight)
                ).to(torch.bfloat16)
        weight_only = _make_moe(bias, quantized=False)
        weight_only.load_weights([dequantized])
        weight_only.post_load_weights()
        del dequantized
        variants["weights_only"] = weight_only

    worst = 1.0
    for x, logits, name in calls[: int(os.environ.get("MSO_L061_COMPARE_CALLS", "8"))]:
        x, logits = x.cuda(), logits.cuda()
        tokens = x.shape[0]
        _, routing = bf16.routing_method.apply(logits)

        def run(moe):
            gemm2, _, index = moe(x, logits, do_finalize=False)
            return post_expert_norm_combine(
                gemm2,
                index,
                routing,
                post_norm,
                1e-8,
                num_tokens=tokens,
                top_k=TOP_K,
                hidden_size=HIDDEN,
            ).float()

        reference = run(bf16)
        for label, moe in variants.items():
            actual = run(moe)
            relative = (actual - reference).norm() / reference.norm()
            cosine = torch.nn.functional.cosine_similarity(actual, reference, dim=-1)
            print(
                f"layer {layer} {name}: {label} vs BF16 relative_l2={relative.item():.4f} "
                f"token cosine mean={cosine.mean().item():.5f} "
                f"p1={cosine.quantile(0.01).item():.5f} min={cosine.min().item():.5f}"
            )
            assert torch.isfinite(actual).all()
            if label == "nvfp4":
                worst = min(worst, cosine.mean().item())
    assert worst > float(os.environ.get("MSO_L061_MIN_COSINE", "0.97"))
