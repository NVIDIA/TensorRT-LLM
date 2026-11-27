# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for building FLUX 3 from its config and loading Diffusers checkpoints."""

import re

import pytest
import torch
from flux3_test_utils import (
    block_names,
    diffusers_shapes,
    diffusers_weights,
    lora_adapter,
    merged_weights,
    model_config,
    tiny_config,
    to_module_name,
    write_checkpoint,
    write_lora,
)

from tensorrt_llm._torch.modules.linear import Linear
from tensorrt_llm._torch.visual_gen.models.flux3.packed import pack
from tensorrt_llm._torch.visual_gen.models.flux3.transformer_flux3 import Flux3Transformer2DModel
from tensorrt_llm._torch.visual_gen.runtime_lora import apply_runtime_lora
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization.mode import QuantAlgo
from tensorrt_llm.visual_gen.args import RuntimeLoRAConfig

pytestmark = pytest.mark.cpu_only

FLOW_MAP = [pytest.param(True, id="klein"), pytest.param(False, id="base")]


@pytest.fixture(autouse=True)
def cpu_weight_loading(monkeypatch: pytest.MonkeyPatch) -> None:
    """Build without probing CUDA and keep every Linear shard on the host."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    load_shard = Linear.load_shard

    def load_shard_on_cpu(self, weights, label=None, device=None, **kwargs):
        return load_shard(self, weights, label, device=torch.device("cpu"), **kwargs)

    monkeypatch.setattr(Linear, "load_shard", load_shard_on_cpu)


def build(flow_map: bool, **kwargs) -> Flux3Transformer2DModel:
    return Flux3Transformer2DModel(model_config(tiny_config(flow_map), **kwargs))


def build_and_load(flow_map: bool) -> tuple[Flux3Transformer2DModel, dict[str, torch.Tensor]]:
    model = build(flow_map)
    weights = diffusers_weights(tiny_config(flow_map))
    model.load_weights(weights)
    model.post_load_weights()
    return model, weights


def expected_parameters(weights: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Model parameters as Diffusers defines them: q, k, v stacked and linear_in as gate, up.

    Keys are TRT-LLM names, where per-mode image blocks are top-level lists.
    """
    expected = {}
    for key, value in weights.items():
        if key.endswith(".attn.to_q.weight"):
            prefix = key[: -len("to_q.weight")]
            expected[prefix + "qkv_proj.weight"] = torch.cat(
                [weights[f"{prefix}{name}.weight"] for name in ("to_q", "to_k", "to_v")]
            )
        elif key.endswith((".attn.to_k.weight", ".attn.to_v.weight")):
            continue
        elif ".ff.linear_in." in key:
            expected[key.replace(".ff.linear_in.", ".ff.gate_up_proj.")] = value
        elif ".ff.linear_out." in key:
            expected[key.replace(".ff.linear_out.", ".ff.down_proj.")] = value
        else:
            expected[key] = value
    return {to_module_name(name): value for name, value in expected.items()}


def packed_kwargs(model: Flux3Transformer2DModel, lengths: dict[str, tuple[int, ...]]) -> dict:
    """Packed zero inputs and timesteps with ``lengths[modality][sample]`` tokens."""
    channels = model.packed_channels
    samples = [
        {
            name: (torch.zeros(counts[index], channels[name]), torch.zeros(counts[index], 4))
            for name, counts in lengths.items()
            if counts[index]
        }
        for index in range(len(next(iter(lengths.values()))))
    ]
    return {**pack(samples, channels).kwargs(), "timestep": torch.zeros(len(samples))}


@pytest.mark.parametrize("flow_map", FLOW_MAP)
def test_every_checkpoint_tensor_loads_into_one_parameter(flow_map: bool) -> None:
    """No checkpoint tensor is missing or left over, and each lands unchanged."""
    model, weights = build_and_load(flow_map)
    expected = expected_parameters(weights)
    actual = dict(model.named_parameters())
    assert actual.keys() == expected.keys()
    for name, value in expected.items():
        assert torch.equal(actual[name], value), name


def test_attention_layer_indices_are_unique_across_stacks() -> None:
    config = tiny_config(flow_map=False)
    model = build(flow_map=False, skip_create_weights_in_init=True)
    blocks = [to_module_name(block) for block in block_names(config)]
    indices = [model.get_submodule(f"{block}.attn").layer_idx for block in blocks]
    assert indices == list(range(len(indices)))
    mlp_indices = [model.get_submodule(f"{block}.ff").layer_idx for block in blocks]
    assert mlp_indices == indices


def test_missing_and_unexpected_tensors_are_named() -> None:
    """Fused, plain-Linear and non-Linear tensors are each reported under checkpoint names."""
    weights = diffusers_weights(tiny_config(flow_map=True))
    del weights["single_transformer_blocks.1.ff.linear_in.weight"]
    del weights["mode_blocks.image_cond.0.attn.norm_k.weight"]
    del weights["target_timestep_embedder.linear_1.weight"]
    weights["single_transformer_blocks.3.attn.to_q.weight"] = torch.zeros(32, 32)
    with pytest.raises(ValueError) as error:
        build(flow_map=True).load_weights(weights)
    message = str(error.value)
    assert (
        "missing 3 [mode_blocks.image_cond.0.attn.norm_k.weight, "
        "single_transformer_blocks.1.ff.linear_in.weight, "
        "target_timestep_embedder.linear_1.weight]"
    ) in message
    assert "unexpected 1 [single_transformer_blocks.3.attn.to_q.weight]" in message


def test_static_fp8_rowwise_checkpoint_loads_strictly() -> None:
    """Quantized blocks carry weight_scale; fused q/k/v and linear_in keep their row order."""
    config = tiny_config(flow_map=True)
    unquantized = [
        "txt_mode_blocks.*",
        "image_embedder",
        "image_cond_embedder",
        "context_embedder",
        "timestep_embedder.*",
        "target_timestep_embedder.*",
        "norm_out.*",
        "proj_out",
    ]
    quant_config = QuantConfig(
        quant_algo=QuantAlgo.FP8_PER_CHANNEL_PER_TOKEN, exclude_modules=unquantized
    )
    weights = diffusers_weights(config)
    generator = torch.Generator().manual_seed(2)
    for name, shape in diffusers_shapes(config).items():
        module = name[: -len(".weight")]
        excluded = quant_config.is_module_excluded_from_quantization(to_module_name(module))
        if len(shape) != 2 or excluded:
            continue
        weights[name] = weights[name].to(torch.float8_e4m3fn)
        weights[f"{module}.weight_scale"] = torch.rand(shape[0], generator=generator) + 0.5
    model = build(flow_map=True, quant_config=quant_config, skip_create_weights_in_init=True)
    model.load_weights(weights)

    block = "single_transformer_blocks.0"
    qkv_proj = model.get_submodule(f"{block}.attn.qkv_proj")
    assert qkv_proj.weight.dtype == torch.float8_e4m3fn
    q_k_v = [weights[f"{block}.attn.to_{name}.weight_scale"] for name in "qkv"]
    torch.testing.assert_close(qkv_proj.weight_scale, torch.cat(q_k_v))
    gate_up = model.get_submodule(f"{block}.ff.gate_up_proj")
    torch.testing.assert_close(gate_up.weight_scale, weights[f"{block}.ff.linear_in.weight_scale"])
    assert model.get_submodule("txt_mode_blocks.0.attn.qkv_proj").weight.dtype == torch.bfloat16


def test_production_config_path_builds_and_loads(tmp_path) -> None:
    """The pipeline config reader and sharded weight loader feed the strict config and loader."""
    from tensorrt_llm._torch.visual_gen.checkpoints import WeightLoader
    from tensorrt_llm._torch.visual_gen.config import DiffusionPipelineConfig
    from tensorrt_llm.mapping import Mapping
    from tensorrt_llm.visual_gen.args import VisualGenArgs

    config = tiny_config(flow_map=True)
    weights = diffusers_weights(config)
    write_checkpoint(tmp_path, config, weights)
    pipeline_config = DiffusionPipelineConfig.from_pretrained(
        str(tmp_path), args=VisualGenArgs(model=str(tmp_path)), device="cpu"
    )
    model = Flux3Transformer2DModel(pipeline_config.model_configs["transformer"])
    assert model.flow_map
    loaded = WeightLoader().load_weights(str(tmp_path), Mapping(), prefetch=False)
    assert loaded.keys() == weights.keys()
    model.load_weights(loaded)


@pytest.mark.parametrize(
    "removed, changes, message",
    [
        pytest.param("num_layers", {}, "missing keys ['num_layers']", id="missing_key"),
        pytest.param(
            None, {"qkv_bias": False}, "unsupported keys ['qkv_bias']", id="unsupported_key"
        ),
        pytest.param(
            None,
            {"axes_dims_rope": [4, 4, 8]},
            "needs 4 even RoPE axis dims",
            id="three_rope_axes",
        ),
        pytest.param(
            None,
            {"axes_dims_rope": [4, 4, 4, 6]},
            "must sum to attention_head_dim",
            id="rope_axes_sum",
        ),
    ],
)
def test_config_is_read_strictly(removed: str | None, changes: dict, message: str) -> None:
    config = {**tiny_config(flow_map=True), **changes}
    if removed is not None:
        del config[removed]
    with pytest.raises(ValueError, match=re.escape(message)):
        Flux3Transformer2DModel(model_config(config))


def test_checkpoint_module_names_exclude_fused_linears() -> None:
    quant_config = QuantConfig(
        quant_algo=QuantAlgo.FP8,
        exclude_modules=[
            "single_transformer_blocks.0.attn.to_k",
            "single_transformer_blocks.0.ff.linear_in",
            "image_mode_blocks.1.ff.linear_out",
        ],
    )
    model = build(flow_map=False, quant_config=quant_config, skip_create_weights_in_init=True)
    excluded = {
        name
        for name, module in model.named_modules()
        if isinstance(module, Linear) and module.quant_config.quant_algo is None
    }
    assert excluded == {
        "single_transformer_blocks.0.attn.qkv_proj",
        "single_transformer_blocks.0.ff.gate_up_proj",
        "image_mode_blocks.1.ff.down_proj",
    }


ONE_SAMPLE = {"image": (4,), "image_cond": (0,), "text": (3,)}


def without(kwargs: dict, key: str) -> dict:
    return {name: value for name, value in kwargs.items() if name != key}


@pytest.mark.parametrize(
    "flow_map, edit, message",
    [
        pytest.param(False, lambda k: without(k, "x_image"), "need x_image", id="no_image"),
        pytest.param(
            False,
            lambda k: {**k, "layout": (("image", (4,)), ("image_cond", (0,)), ("text", (0,)))},
            "needs text tokens",
            id="no_text_tokens",
        ),
        pytest.param(
            False,
            lambda k: {**k, "layout": (("image", (0,)), ("image_cond", (0,)), ("text", (3,)))},
            "needs image tokens",
            id="no_image_tokens",
        ),
        pytest.param(
            False,
            lambda k: {**k, "layout": tuple(reversed(k["layout"]))},
            "must list",
            id="modality_order",
        ),
        pytest.param(False, lambda k: without(k, "timestep"), "need timestep", id="no_timestep"),
        pytest.param(
            False,
            lambda k: {**k, "timestep": torch.zeros(4)},
            "timestep must be a scalar or have shape (1,)",
            id="per_token_timesteps",
        ),
        pytest.param(True, lambda k: k, "required by, and only by", id="klein_without_target"),
        pytest.param(
            False,
            lambda k: {**k, "target_timestep": torch.zeros(1)},
            "required by, and only by",
            id="base_with_target",
        ),
        pytest.param(
            True,
            lambda k: {**k, "target_timestep": torch.zeros(2)},
            "target_timestep must have shape (1,)",
            id="target_shape",
        ),
    ],
)
def test_invalid_inputs_fail_before_compute(flow_map: bool, edit, message: str) -> None:
    model = build(flow_map, skip_create_weights_in_init=True)
    with pytest.raises(ValueError, match=re.escape(message)):
        model(**edit(packed_kwargs(model, ONE_SAMPLE)))


# =============================================================================
# Runtime LoRA
# =============================================================================

LORA_RANK, LORA_ALPHA = 4, 8.0


def parameters(model: Flux3Transformer2DModel) -> dict[str, torch.Tensor]:
    return {name: value.detach().clone() for name, value in model.named_parameters()}


@pytest.mark.parametrize("flow_map", FLOW_MAP)
def test_runtime_lora_in_both_formats_matches_the_merged_checkpoint(tmp_path, flow_map) -> None:
    """Diffusers and BFL adapters fuse identically into every Linear, as a merged checkpoint."""
    weights = diffusers_weights(tiny_config(flow_map))
    adapter = lora_adapter(weights, LORA_RANK)
    expected = expected_parameters(merged_weights(weights, adapter, LORA_ALPHA / LORA_RANK))
    fused = {}
    for form in ("diffusers", "bfl"):
        path = write_lora(tmp_path / form, adapter, LORA_ALPHA, bfl=form == "bfl")
        model, _ = build_and_load(flow_map)
        report = apply_runtime_lora(model, RuntimeLoRAConfig(path=str(path)))
        linears = [name for name, module in model.named_modules() if isinstance(module, Linear)]
        assert sorted(report.applied_modules) == sorted(linears)
        assert report.skipped_non_targets == report.skipped_incomplete == 0
        fused[form] = parameters(model)
    assert fused["diffusers"].keys() == expected.keys()
    for name, value in expected.items():
        assert torch.equal(fused["bfl"][name], fused["diffusers"][name]), name
        # Fusion rounds the BF16 delta and sum, far below the size of a misplaced delta.
        atol = value.abs().max().item() * 2**-6
        torch.testing.assert_close(fused["diffusers"][name].float(), value, rtol=0, atol=atol)
