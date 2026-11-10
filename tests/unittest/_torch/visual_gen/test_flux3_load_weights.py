# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for building FLUX.3 from its config and loading Diffusers checkpoints."""

import re
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
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
from safetensors.torch import save_file

from tensorrt_llm._torch.modules.linear import Linear
from tensorrt_llm._torch.visual_gen.attention import attention as attention_module
from tensorrt_llm._torch.visual_gen.attention.attention import Attention, QKVMode
from tensorrt_llm._torch.visual_gen.attention.backends.interface import AttentionTensorLayout
from tensorrt_llm._torch.visual_gen.models.flux3.packed import pack
from tensorrt_llm._torch.visual_gen.models.flux3.transformer_flux3 import (
    Flux3Attention,
    Flux3Transformer2DModel,
    Flux3TransformerConfig,
)
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


def test_norm_out_reads_scale_before_shift_per_sample() -> None:
    model, weights = build_and_load(flow_map=True)
    hidden_size = model.arch_config.hidden_size
    generator = torch.Generator().manual_seed(1)
    x = torch.randn(2, 3, hidden_size, generator=generator).to(torch.bfloat16)
    vec = torch.randn(2, hidden_size, generator=generator).to(torch.bfloat16)
    with torch.inference_mode():
        actual = model.norm_out(x, vec)
    scale, shift = F.linear(F.silu(vec), weights["norm_out.linear.weight"])[:, None].chunk(2, -1)
    normed = F.layer_norm(x.float(), (hidden_size,), eps=1e-6).to(torch.bfloat16)
    torch.testing.assert_close(actual, (1 + scale) * normed + shift)


def test_target_timestep_embedder_exists_only_with_flow_map() -> None:
    assert build(flow_map=True).target_timestep_embedder is not None
    base = build(flow_map=False)
    assert base.target_timestep_embedder is None
    assert not any(name.startswith("target_") for name, _ in base.named_parameters())


def test_attention_layer_indices_are_unique_across_stacks() -> None:
    config = tiny_config(flow_map=False)
    model = build(flow_map=False, skip_create_weights_in_init=True)
    blocks = [to_module_name(block) for block in block_names(config)]
    indices = [model.get_submodule(f"{block}.attn").layer_idx for block in blocks]
    assert indices == list(range(len(indices)))
    mlp_indices = [model.get_submodule(f"{block}.ff").layer_idx for block in blocks]
    assert mlp_indices == indices


@pytest.mark.parametrize(
    "model_flow_map, checkpoint_flow_map, message",
    [
        pytest.param(
            True,
            False,
            "missing 2 [target_timestep_embedder.linear_1.weight, "
            "target_timestep_embedder.linear_2.weight], unexpected 0 []",
            id="klein_model_base_checkpoint",
        ),
        pytest.param(
            False,
            True,
            "missing 0 [], unexpected 2 [target_timestep_embedder.linear_1.weight, "
            "target_timestep_embedder.linear_2.weight]",
            id="base_model_klein_checkpoint",
        ),
    ],
)
def test_flow_map_mismatch_fails_to_load(
    model_flow_map: bool, checkpoint_flow_map: bool, message: str
) -> None:
    model = build(model_flow_map)
    with pytest.raises(ValueError, match=re.escape(message)):
        model.load_weights(diffusers_weights(tiny_config(checkpoint_flow_map)))


def test_missing_and_unexpected_tensors_are_named() -> None:
    weights = diffusers_weights(tiny_config(flow_map=True))
    del weights["single_transformer_blocks.1.ff.linear_in.weight"]
    del weights["mode_blocks.image_cond.0.attn.norm_k.weight"]
    weights["single_transformer_blocks.3.attn.to_q.weight"] = torch.zeros(32, 32)
    with pytest.raises(ValueError) as error:
        build(flow_map=True).load_weights(weights)
    message = str(error.value)
    assert (
        "missing 2 [mode_blocks.image_cond.0.attn.norm_k.weight, "
        "single_transformer_blocks.1.ff.linear_in.weight]"
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
    actual = dict(model.named_parameters())
    for name, value in expected_parameters(weights).items():
        assert torch.equal(actual[name], value), name


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
        Flux3TransformerConfig.from_pretrained_config(SimpleNamespace(**config))
    with pytest.raises(ValueError, match=re.escape(message)):
        Flux3Transformer2DModel(model_config(config))


def test_config_defaults_flow_map_and_ignores_loader_keys() -> None:
    config = {**tiny_config(flow_map=False), "_name_or_path": "/ckpt", "quantization_config": {}}
    arch = Flux3TransformerConfig.from_pretrained_config(SimpleNamespace(**config))
    assert arch.flow_map is False
    assert arch.hidden_size == 32 and arch.mlp_hidden_dim == 96


NVFP4 = QuantConfig(quant_algo=QuantAlgo.NVFP4, group_size=16)


@pytest.mark.parametrize(
    "quant_config, dynamic_weight_quant, qkv_mode",
    [
        pytest.param(QuantConfig(), False, QKVMode.FUSE_QKV, id="bf16"),
        pytest.param(QuantConfig(quant_algo=QuantAlgo.FP8), False, QKVMode.FUSE_QKV, id="fp8"),
        pytest.param(NVFP4, True, QKVMode.FUSE_QKV, id="dynamic_nvfp4"),
        pytest.param(NVFP4, False, QKVMode.SEPARATE_QKV, id="static_nvfp4"),
        pytest.param(
            QuantConfig(
                quant_algo=QuantAlgo.NVFP4, group_size=16, exclude_modules=["txt_mode_blocks.0"]
            ),
            False,
            QKVMode.FUSE_QKV,
            id="static_nvfp4_excluded_block",
        ),
    ],
)
def test_qkv_topology_follows_quant_config_and_keeps_backend(
    monkeypatch: pytest.MonkeyPatch,
    quant_config: QuantConfig,
    dynamic_weight_quant: bool,
    qkv_mode: QKVMode,
) -> None:
    """Static NVFP4 splits quantized Q/K/V but stays self-attention, so TRTLLM is kept."""
    backends = []

    def create_attention(backend: str, **kwargs) -> SimpleNamespace:
        backends.append(backend)
        return SimpleNamespace(preferred_layout=AttentionTensorLayout.NHD)

    monkeypatch.setattr(attention_module, "create_attention", create_attention)
    config = model_config(
        tiny_config(flow_map=True),
        quant_config=quant_config,
        backend="TRTLLM",
        dynamic_weight_quant=dynamic_weight_quant,
        skip_create_weights_in_init=True,
    )
    attn = Flux3Attention(
        hidden_size=32,
        num_heads=2,
        eps=1e-6,
        config=config,
        layer_idx=7,
        module_name="txt_mode_blocks.0.attn",
    )
    assert attn.qkv_mode == qkv_mode
    assert attn.separate_qkv_is_self_attention
    assert attn.attn_backend == "TRTLLM"
    assert attn.layer_idx == 7
    assert backends == ["TRTLLM"]


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


def test_composite_modules_settle_after_their_linears(monkeypatch: pytest.MonkeyPatch) -> None:
    model = build(flow_map=False)
    calls = []
    monkeypatch.setattr(Linear, "post_load_weights", lambda self: calls.append("linear"))
    monkeypatch.setattr(Attention, "post_load_weights", lambda self: calls.append("composite"))
    model.post_load_weights()
    assert "composite" in calls
    assert calls.index("composite") > max(i for i, call in enumerate(calls) if call == "linear")


ONE_SAMPLE = {"image": (4,), "image_cond": (0,), "text": (3,)}
TWO_SAMPLES = {"image": (4, 4), "image_cond": (0, 0), "text": (3, 5)}


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


def test_prepared_text_streams_replace_raw_text_inputs() -> None:
    """Cached streams need no raw text tensors, so graph callers may drop them."""
    model = build(flow_map=False, skip_create_weights_in_init=True)
    kwargs = packed_kwargs(model, TWO_SAMPLES)
    cached = {"_flux3_text_tokens": torch.zeros(1), "_flux3_joint_cos": torch.zeros(1)}
    layout, timestep = kwargs["layout"], kwargs["timestep"]
    model._check_inputs(
        {**without(without(kwargs, "x_text"), "x_text_ids"), **cached}, layout, timestep
    )
    with pytest.raises(ValueError, match="need x_text"):
        model._check_inputs(without(kwargs, "x_text"), layout, timestep)


def test_cuda_graph_key_includes_packed_layout() -> None:
    model = build(flow_map=False, skip_create_weights_in_init=True)
    key_fns = {}
    runner = SimpleNamespace(register_extra_key_fn=key_fns.__setitem__)
    model.register_cuda_graph_extra_key_fns(runner)
    layout = packed_kwargs(model, TWO_SAMPLES)["layout"]
    assert key_fns["flux3_layout"](layout=layout) == layout
    assert key_fns["flux3_layout"]() is None


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


@pytest.mark.parametrize("bfl", [False, True], ids=["diffusers", "bfl"])
def test_runtime_lora_at_scale_zero_keeps_every_weight(tmp_path, bfl: bool) -> None:
    model, weights = build_and_load(flow_map=True)
    path = write_lora(tmp_path / "lora", lora_adapter(weights, LORA_RANK), LORA_ALPHA, bfl=bfl)
    before = parameters(model)
    apply_runtime_lora(model, RuntimeLoRAConfig(path=str(path), scale=0.0))
    for name, value in model.named_parameters():
        assert torch.equal(value, before[name]), name


def lora_tensors(prefix: str, module: str, a: torch.Tensor, b: torch.Tensor) -> dict:
    return {
        f"{prefix}{module}.lora_A.weight": a.clone(),
        f"{prefix}{module}.lora_B.weight": b.clone(),
    }


def test_runtime_lora_strict_mode_rejects_unknown_targets_before_fusing(tmp_path) -> None:
    model, weights = build_and_load(flow_map=True)
    a, b = lora_adapter(weights, LORA_RANK)["proj_out"]
    tensors = {
        **lora_tensors("diffusion_model.", "final_layer.image.linear", a, b),
        **lora_tensors("diffusion_model.", "single_blocks.0.gate_proj", a, b),
    }
    save_file(tensors, str(tmp_path / "lora.safetensors"))
    before = parameters(model)
    config = RuntimeLoRAConfig(path=str(tmp_path / "lora.safetensors"))
    with pytest.raises(ValueError, match="skipped 1 adapter target.*single_blocks.0.gate_proj"):
        apply_runtime_lora(model, config)
    for name, value in model.named_parameters():
        assert torch.equal(value, before[name]), name

    report = apply_runtime_lora(model, config.model_copy(update={"strict": False}))
    assert report.applied_modules == ("proj_out",)
    assert report.skipped_non_targets == 1


def test_runtime_lora_rejects_bfl_and_diffusers_names_for_one_module(tmp_path) -> None:
    model, weights = build_and_load(flow_map=True)
    a, b = lora_adapter(weights, LORA_RANK)["proj_out"]
    tensors = {
        **lora_tensors("diffusion_model.", "final_layer.image.linear", a, b),
        **lora_tensors("transformer.", "proj_out", a, b),
    }
    save_file(tensors, str(tmp_path / "lora.safetensors"))
    with pytest.raises(ValueError, match="both target FLUX.3 module 'proj_out'"):
        apply_runtime_lora(model, RuntimeLoRAConfig(path=str(tmp_path / "lora.safetensors")))
