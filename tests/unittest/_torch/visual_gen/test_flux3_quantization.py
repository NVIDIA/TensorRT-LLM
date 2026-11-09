# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FLUX.3 pre-quantized BFL variants and load-time FP8: CPU cases are cpu_only, the rest need a GPU."""

import json
import re
from pathlib import Path

import pytest
import torch
from flux3_test_utils import (
    KLEIN_BASE_CONFIG,
    KLEIN_CONFIG,
    FakeTextEncoder,
    assert_parity,
    bfl_hf_quant_config,
    bfl_variant_excludes,
    bfl_variant_tensor_names,
    diffusers_weights,
    gpu_transformer_config,
    model_config,
    placeholder_variant_tensors,
    quantize_variant,
    tiny_config,
    write_checkpoint,
    write_gpu_checkpoint,
    write_variant,
)

from tensorrt_llm._torch.models.modeling_utils import MetaInitMode
from tensorrt_llm._torch.modules.linear import (
    FP8RowwiseLinearMethod,
    Linear,
    NVFP4LinearMethod,
    UnquantizedLinearMethod,
)

# Importing the models package registers every pipeline.
from tensorrt_llm._torch.visual_gen import models  # noqa: F401
from tensorrt_llm._torch.visual_gen import pipeline_loader as pipeline_loader_module
from tensorrt_llm._torch.visual_gen.attention.attention import QKVMode
from tensorrt_llm._torch.visual_gen.config import DiffusionPipelineConfig
from tensorrt_llm._torch.visual_gen.models.flux3 import pack
from tensorrt_llm._torch.visual_gen.models.flux3.bfl_keys import (
    bfl_to_diffusers_key,
    bfl_to_diffusers_state_dict,
    diffusers_to_bfl_module,
)
from tensorrt_llm._torch.visual_gen.models.flux3.pipeline_flux3 import (
    Flux3DistilledPipeline,
    Flux3Pipeline,
)
from tensorrt_llm._torch.visual_gen.models.flux3.text_encoder import Flux3TextEncoder
from tensorrt_llm._torch.visual_gen.models.flux3.transformer_flux3 import (
    LOAD_TIME_QUANT_EXCLUDED_MODULES,
    Flux3Transformer2DModel,
    _checkpoint_module_names,
)
from tensorrt_llm._torch.visual_gen.models.flux3.variants import (
    check_variant,
    collapse_exclusions,
    variant_quantization_config,
)
from tensorrt_llm._torch.visual_gen.pipeline_loader import PipelineLoader
from tensorrt_llm._torch.visual_gen.pipeline_registry import AutoPipeline
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization.mode import QuantAlgo
from tensorrt_llm.visual_gen import VisualGen
from tensorrt_llm.visual_gen.args import (
    AttentionConfig,
    RuntimeLoRAConfig,
    TorchCompileConfig,
    VisualGenArgs,
)

VARIANTS = ["nvfp4", "fp8r"]
FLOW_MAP = [pytest.param(True, id="klein"), pytest.param(False, id="base")]
HF_IDS = {
    True: "black-forest-labs/flux-3-image-klein",
    False: "black-forest-labs/flux-3-image-klein-base",
}
DOWNLOAD_PATTERNS = [
    "modular_model_index.json",
    "transformer/*",
    "vae/*",
    "text_encoder/*",
    "LICENSE.md",
    "NOTICE.md",
    "README.md",
]
# Tensor counts of the released BF16, nvfp4 and fp8r exports (klein, klein-base).
RELEASED_TENSOR_COUNTS = {None: (359, 357), "nvfp4": (953, 951), "fp8r": (593, 591)}
# exclude_modules lengths of the released hf_quant_config.json files (klein, klein-base).
RELEASED_EXCLUDE_COUNTS = {"nvfp4": (75, 73), "fp8r": (39, 37)}
# Block-level exclusions the released variants translate to.
KLEIN_IGNORE = {
    "nvfp4": [
        "context_embedder",
        "early_stream_modulations",
        "mode_blocks.image.0",
        "norm_out",
        "proj_out",
        "single_stream_modulations",
        "single_transformer_blocks.24",
        "single_transformer_blocks.25",
        "single_transformer_blocks.26",
        "single_transformer_blocks.27",
        "target_timestep_embedder",
        "timestep_embedder",
        "txt_mode_blocks",
        "x_embedders",
    ],
    "fp8r": [
        "context_embedder",
        "norm_out",
        "proj_out",
        "target_timestep_embedder",
        "timestep_embedder",
        "txt_mode_blocks",
        "x_embedders",
    ],
}
QUANTIZED_METHODS = {"nvfp4": NVFP4LinearMethod, "fp8r": FP8RowwiseLinearMethod}
QUANTIZED_DTYPES = {"nvfp4": torch.uint8, "fp8r": torch.float8_e4m3fn}
FP8_ROWWISE = {"quant_algo": "FP8_PER_CHANNEL_PER_TOKEN"}

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")
requires_blackwell = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 10,
    reason="NVFP4 needs a Blackwell GPU",
)


# =============================================================================
# Helpers
# =============================================================================


def released_config(flow_map: bool) -> dict:
    return KLEIN_CONFIG if flow_map else KLEIN_BASE_CONFIG


def drop_transformer_weights(path: Path) -> None:
    """Keep only transformer/config.json, as a Hub variant download does."""
    for file in (path / "transformer").iterdir():
        if file.name != "config.json":
            file.unlink()


def write_variant_checkpoint(path: Path, flow_map: bool, variant: str) -> dict:
    """A tiny variant download whose tensors only have the right names; no BF16 transformer."""
    config = tiny_config(flow_map)
    write_checkpoint(path, config, diffusers_weights(config))
    write_variant(path, config, variant, placeholder_variant_tensors(config, variant))
    drop_transformer_weights(path)
    return config


def load_config(path: Path, **args) -> DiffusionPipelineConfig:
    return DiffusionPipelineConfig.from_pretrained(
        str(path), args=VisualGenArgs(model=str(path), **args), device="cpu"
    )


def build_pipeline(path: Path, **args) -> Flux3Pipeline:
    with MetaInitMode():
        return AutoPipeline.from_config(load_config(path, **args), str(path))


def bfl_linear_quantization(transformer: Flux3Transformer2DModel) -> dict[str, bool]:
    """Whether each BFL Linear module loads quantized into ``transformer``."""
    return {
        diffusers_to_bfl_module(module): quantized
        for module, quantized in transformer.checkpoint_quantization().items()
    }


def expected_bfl_quantization(config: dict, variant: str) -> dict[str, bool]:
    excluded = set(bfl_variant_excludes(config, variant))
    modules = [
        name[: -len(".weight")]
        for name in bfl_variant_tensor_names(config)
        if name.endswith(".weight")
    ]
    return {module: module not in excluded for module in modules}


def assert_methods_follow_quantization(transformer: Flux3Transformer2DModel, method: type) -> None:
    """Each Linear runs ``method`` exactly when its checkpoint module is quantized."""
    quantization = transformer.checkpoint_quantization()
    for name, module in transformer.named_modules():
        if isinstance(module, Linear):
            quantized = quantization[_checkpoint_module_names(name)[0]]
            expected = method if quantized else UnquantizedLinearMethod
            assert type(module.quant_method) is expected, name


@pytest.fixture
def no_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    """Build transformers without probing CUDA."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)


# =============================================================================
# Variant selection and download
# =============================================================================


@pytest.mark.cpu_only
@pytest.mark.parametrize("variant", [None, *VARIANTS])
def test_known_variants_are_accepted(variant):
    assert check_variant(variant) == variant


@pytest.mark.cpu_only
@pytest.mark.parametrize("variant", ["NVFP4", "fp8", "../nvfp4", "", 4, ["nvfp4"]])
def test_unknown_variants_are_rejected_before_any_download(monkeypatch, variant):
    def download(*args, **kwargs):
        raise AssertionError("an invalid variant must not download")

    monkeypatch.setattr(pipeline_loader_module, "download_hf_partial", download)
    monkeypatch.setattr(pipeline_loader_module, "download_hf_model", download)
    message = re.escape("pipeline_config.variant must be null or one of ['fp8r', 'nvfp4']")
    with pytest.raises(ValueError, match=message):
        check_variant(variant)
    args = VisualGenArgs(model=HF_IDS[True], pipeline_config={"variant": variant})
    with pytest.raises(ValueError, match=message):
        PipelineLoader(args, device="cpu")._resolve_checkpoint_dir(HF_IDS[True])


@pytest.mark.cpu_only
@pytest.mark.parametrize("variant", [None, *VARIANTS])
@pytest.mark.parametrize("flow_map", FLOW_MAP)
def test_hub_download_fetches_only_the_selected_variant(monkeypatch, tmp_path, flow_map, variant):
    """One download at the requested revision; a variant replaces the BF16 transformer weights."""
    hf_id = HF_IDS[flow_map]
    assert VisualGen.pipeline_config(hf_id) == {"variant": None}
    downloads = []

    def download_partial(model, allow_patterns, revision=None):
        downloads.append((model, allow_patterns, revision))
        return tmp_path

    def download_full(*args, **kwargs):
        raise AssertionError("FLUX.3 must not download the whole repository")

    monkeypatch.setattr(pipeline_loader_module, "download_hf_partial", download_partial)
    monkeypatch.setattr(pipeline_loader_module, "download_hf_model", download_full)
    pipeline_config = {} if variant is None else {"variant": variant}
    args = VisualGenArgs(model=hf_id, revision="0123abc", pipeline_config=pipeline_config)
    assert PipelineLoader(args, device="cpu")._resolve_checkpoint_dir(hf_id) == str(tmp_path)

    expected = DOWNLOAD_PATTERNS
    if variant is not None:
        expected = [p for p in DOWNLOAD_PATTERNS if p != "transformer/*"]
        expected += ["transformer/config.json", f"variants/{variant}/*"]
    assert downloads == [(hf_id, expected, "0123abc")]


# =============================================================================
# hf_quant_config translation on the released structure
# =============================================================================


@pytest.mark.cpu_only
@pytest.mark.parametrize("flow_map", FLOW_MAP)
def test_synthetic_variants_have_the_released_structure(flow_map):
    config, index = released_config(flow_map), 0 if flow_map else 1
    for variant, counts in RELEASED_TENSOR_COUNTS.items():
        assert len(bfl_variant_tensor_names(config, variant)) == counts[index], variant
    for variant, counts in RELEASED_EXCLUDE_COUNTS.items():
        assert len(bfl_variant_excludes(config, variant)) == counts[index], variant


@pytest.mark.cpu_only
@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("flow_map", FLOW_MAP)
def test_released_variants_translate_to_block_level_exclusions(no_cuda, flow_map, variant):
    """Every BFL-excluded Linear stays BF16 and every other one gets the variant's method."""
    config = released_config(flow_map)
    quantization_config = variant_quantization_config(
        variant, bfl_hf_quant_config(config, variant), bfl_variant_tensor_names(config, variant)
    )
    expected_ignore = KLEIN_IGNORE[variant]
    if not flow_map:
        expected_ignore = [name for name in expected_ignore if name != "target_timestep_embedder"]
    assert quantization_config["ignore"] == expected_ignore

    quant_config, _, dynamic_weights, dynamic_activations = (
        DiffusionPipelineConfig.load_diffusion_quant_config(quantization_config)
    )
    assert quant_config.quant_algo == QuantAlgo(quantization_config["quant_algo"])
    assert quant_config.group_size == (16 if variant == "nvfp4" else None)
    assert not dynamic_weights
    assert dynamic_activations is (variant == "fp8r")

    with MetaInitMode():
        transformer = Flux3Transformer2DModel(
            model_config(
                config,
                quant_config=quant_config,
                force_dynamic_quantization=dynamic_activations,
                skip_create_weights_in_init=True,
            )
        )
    assert bfl_linear_quantization(transformer) == expected_bfl_quantization(config, variant)
    assert_methods_follow_quantization(transformer, QUANTIZED_METHODS[variant])
    # Static NVFP4 q, k and v carry their own scales, so quantized blocks keep them separate.
    separate = QKVMode.SEPARATE_QKV if variant == "nvfp4" else QKVMode.FUSE_QKV
    assert transformer.single_transformer_blocks[0].attn.qkv_mode == separate
    assert transformer.txt_mode_blocks[0].attn.qkv_mode == QKVMode.FUSE_QKV


@pytest.mark.cpu_only
def test_exclusions_collapse_only_into_fully_excluded_ancestors():
    modules = ["a.0.x", "a.0.y", "a.1.x", "a.10.x", "b.x", "c"]
    assert collapse_exclusions(["a.0.x", "a.0.y", "b.x"], modules) == ["a.0", "b"]
    assert collapse_exclusions(["a.1.x"], modules) == ["a.1"]
    assert collapse_exclusions(["a.0.x", "c"], modules) == ["a.0.x", "c"]
    assert collapse_exclusions(modules, modules) == ["a", "b", "c"]
    with pytest.raises(ValueError, match=r"does not have: \['d'\]"):
        collapse_exclusions(["d"], modules)


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "variant, change, message",
    [
        pytest.param(
            "nvfp4", {"quant_algo": "FP8"}, "needs quant_algo NVFP4", id="nvfp4_wrong_algo"
        ),
        pytest.param(
            "fp8r",
            {"quant_algo": "NVFP4"},
            "needs quant_algo FP8_PER_CHANNEL_PER_TOKEN",
            id="fp8r_wrong_algo",
        ),
        pytest.param("nvfp4", {"group_size": 32}, "needs group_size 16", id="group_size"),
        pytest.param(
            "fp8r", {"exclude_modules": ["txt_in", "proj"]}, "module name: 'proj'", id="unknown"
        ),
    ],
)
def test_unsupported_hf_quant_configs_are_rejected(variant, change, message):
    config = tiny_config(flow_map=True)
    hf_quant_config = bfl_hf_quant_config(config, variant)
    hf_quant_config["quantization"].update(change)
    with pytest.raises(ValueError, match=re.escape(message)):
        variant_quantization_config(
            variant, hf_quant_config, bfl_variant_tensor_names(config, variant)
        )


# =============================================================================
# Variants through the checkpoint-config hook
# =============================================================================


@pytest.mark.cpu_only
@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("flow_map", FLOW_MAP)
def test_variant_quantization_reaches_the_transformer(no_cuda, tmp_path, flow_map, variant):
    config = write_variant_checkpoint(tmp_path, flow_map, variant)
    pipeline_config = load_config(tmp_path, pipeline_config={"variant": variant})
    transformer_config = pipeline_config.model_configs["transformer"]
    algo = QuantAlgo.NVFP4 if variant == "nvfp4" else QuantAlgo.FP8_PER_CHANNEL_PER_TOKEN
    assert transformer_config.quant_config.quant_algo == algo
    assert not transformer_config.dynamic_weight_quant
    assert transformer_config.force_dynamic_quantization is (variant == "fp8r")
    assert transformer_config.pretrained_config.quantization_config["quant_algo"] == algo.value
    assert pipeline_config.extra_attrs["variant"] == variant

    with MetaInitMode():
        pipeline = AutoPipeline.from_config(pipeline_config, str(tmp_path))
    assert type(pipeline) is (Flux3DistilledPipeline if flow_map else Flux3Pipeline)
    expected = expected_bfl_quantization(config, variant)
    assert bfl_linear_quantization(pipeline.transformer) == expected
    assert_methods_follow_quantization(pipeline.transformer, QUANTIZED_METHODS[variant])


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "quant_config",
    [
        pytest.param(FP8_ROWWISE, id="fp8_rowwise"),
        pytest.param({"quant_algo": "NVFP4", "dynamic": True}, id="dynamic_nvfp4"),
        pytest.param(QuantConfig(quant_algo=QuantAlgo.NVFP4), id="quant_config_object"),
    ],
)
@pytest.mark.parametrize("variant", VARIANTS)
def test_variant_rejects_an_explicit_quant_algo(tmp_path, variant, quant_config):
    write_variant_checkpoint(tmp_path, True, variant)
    message = f"FLUX.3 variant '{variant}' is pre-quantized; remove quant_config.quant_algo"
    with pytest.raises(ValueError, match=re.escape(message)):
        load_config(tmp_path, pipeline_config={"variant": variant}, quant_config=quant_config)


@pytest.mark.cpu_only
@pytest.mark.parametrize("variant", VARIANTS)
def test_variant_rejects_runtime_lora(tmp_path, variant):
    write_variant_checkpoint(tmp_path, True, variant)
    lora = RuntimeLoRAConfig(path=str(tmp_path / "adapter.safetensors"))
    with pytest.raises(ValueError, match="runtime_lora_config does not currently support"):
        load_config(tmp_path, pipeline_config={"variant": variant}, runtime_lora_config=lora)


@pytest.mark.cpu_only
@pytest.mark.parametrize("file_name", ["config.json", "hf_quant_config.json", "model.safetensors"])
def test_missing_variant_files_are_named(tmp_path, file_name):
    write_variant_checkpoint(tmp_path, True, "fp8r")
    (tmp_path / "variants" / "fp8r" / file_name).unlink()
    message = "variant file not found: .*" + re.escape(f"variants/fp8r/{file_name}")
    with pytest.raises(ValueError, match=message):
        load_config(tmp_path, pipeline_config={"variant": "fp8r"})


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "field, value, mismatch",
    [
        pytest.param("quantization", "nvfp4", ("nvfp4", "fp8r"), id="quantization"),
        pytest.param("hidden", 64, (64, 32), id="hidden"),
        pytest.param("heads", 4, (4, 2), id="heads"),
        pytest.param("depth", 3, (3, 2), id="depth"),
        pytest.param("depth_single", 4, (4, 3), id="depth_single"),
    ],
)
def test_variant_config_mismatches_are_named(tmp_path, field, value, mismatch):
    write_variant_checkpoint(tmp_path, True, "fp8r")
    path = tmp_path / "variants" / "fp8r" / "config.json"
    variant_config = json.loads(path.read_text())
    (variant_config if field == "quantization" else variant_config["dit"])[field] = value
    path.write_text(json.dumps(variant_config))
    message = "does not match transformer/config.json, as (variant, transformer): "
    with pytest.raises(ValueError, match=re.escape(message + str({field: mismatch}))):
        load_config(tmp_path, pipeline_config={"variant": "fp8r"})


@pytest.mark.cpu_only
def test_variant_of_another_model_is_rejected(tmp_path):
    """A klein variant beside a klein-base transformer fails on its config.json."""
    base = tiny_config(flow_map=False)
    write_checkpoint(tmp_path, base, diffusers_weights(base))
    klein = tiny_config(flow_map=True)
    write_variant(tmp_path, klein, "fp8r", placeholder_variant_tensors(klein, "fp8r"))
    message = "does not match transformer/config.json, as (variant, transformer): "
    with pytest.raises(ValueError, match=re.escape(message + "{'flow_map': (True, False)}")):
        load_config(tmp_path, pipeline_config={"variant": "fp8r"})


@pytest.mark.cpu_only
@pytest.mark.parametrize("variant", VARIANTS)
def test_variant_weights_load_with_diffusers_names_and_all_scales(no_cuda, tmp_path, variant):
    config = write_variant_checkpoint(tmp_path, True, variant)
    pipeline = build_pipeline(tmp_path, pipeline_config={"variant": variant})
    weights = pipeline.load_transformer_weights(str(tmp_path))
    names = bfl_variant_tensor_names(config, variant)
    assert weights.keys() == {bfl_to_diffusers_key(name) for name in names}
    assert "single_transformer_blocks.0.attn.to_q.weight_scale" in weights
    assert "txt_mode_blocks.0.attn.to_q.weight_scale" not in weights


@pytest.mark.cpu_only
@pytest.mark.parametrize("variant", VARIANTS)
def test_variant_weights_must_carry_exactly_their_scales(no_cuda, tmp_path, variant):
    config = tiny_config(flow_map=True)
    write_checkpoint(tmp_path, config, diffusers_weights(config))
    tensors = placeholder_variant_tensors(config, variant)
    dropped = "input_scale" if variant == "nvfp4" else "weight_scale"
    del tensors[f"single_blocks.1.mlp_in.{dropped}"]
    tensors["txt_mode_blocks.0.q_proj.weight_scale"] = torch.zeros(2)
    write_variant(tmp_path, config, variant, tensors)
    pipeline = build_pipeline(tmp_path, pipeline_config={"variant": variant})
    message = (
        f"quantized layers miss 1 [single_transformer_blocks.1.ff.linear_in.{dropped}], "
        "BF16 layers carry 1 [txt_mode_blocks.0.attn.to_q.weight_scale]"
    )
    with pytest.raises(ValueError, match=re.escape(message)):
        pipeline.load_transformer_weights(str(tmp_path))


# =============================================================================
# Load-time FP8 rowwise
# =============================================================================


@pytest.mark.cpu_only
@pytest.mark.parametrize("flow_map", FLOW_MAP)
def test_load_time_fp8_keeps_the_default_and_user_exclusions_in_bf16(no_cuda, tmp_path, flow_map):
    config = tiny_config(flow_map)
    write_checkpoint(tmp_path, config, diffusers_weights(config))
    user_quant = {**FP8_ROWWISE, "ignore": ["single_transformer_blocks.0"]}
    pipeline_config = load_config(tmp_path, quant_config=user_quant)
    assert pipeline_config.dynamic_weight_quant
    with MetaInitMode():
        pipeline = AutoPipeline.from_config(pipeline_config, str(tmp_path))

    assert sorted(LOAD_TIME_QUANT_EXCLUDED_MODULES) == KLEIN_IGNORE["fp8r"]
    excluded = tuple(f"{name}." for name in ("single_transformer_blocks.0", *KLEIN_IGNORE["fp8r"]))
    quantization = pipeline.transformer.checkpoint_quantization()
    assert quantization == {
        module: not f"{module}.".startswith(excluded) for module in quantization
    }
    assert any(quantization.values()) and "proj_out" in quantization
    assert_methods_follow_quantization(pipeline.transformer, FP8RowwiseLinearMethod)
    assert pipeline_config.quant_config.exclude_modules == ["single_transformer_blocks.0"]


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "quant_config, got",
    [
        pytest.param({"quant_algo": "FP8", "dynamic": True}, "FP8", id="fp8"),
        pytest.param(
            {"quant_algo": "FP8_BLOCK_SCALES", "dynamic": True}, "FP8_BLOCK_SCALES", id="blockwise"
        ),
        pytest.param({"quant_algo": "NVFP4", "dynamic": True}, "NVFP4", id="nvfp4"),
        pytest.param({"quant_algo": "W4A16_AWQ"}, "W4A16_AWQ", id="awq"),
        pytest.param(
            {**FP8_ROWWISE, "dynamic": False},
            "static FP8_PER_CHANNEL_PER_TOKEN",
            id="static_fp8_rowwise",
        ),
    ],
)
def test_other_quantization_is_rejected_naming_the_supported_options(tmp_path, quant_config, got):
    config = tiny_config(flow_map=True)
    write_checkpoint(tmp_path, config, diffusers_weights(config))
    message = (
        "FLUX.3 quantizes BF16 weights on load only with quant_config.quant_algo "
        f"FP8_PER_CHANNEL_PER_TOKEN, got {got}; for pre-quantized NVFP4 or FP8 rowwise weights, "
        "set pipeline_config.variant to 'nvfp4' or 'fp8r'."
    )
    with pytest.raises(ValueError, match=re.escape(message)):
        build_pipeline(tmp_path, quant_config=quant_config)


# =============================================================================
# GPU: tiny checkpoints through PipelineLoader
# =============================================================================

# Head size 64 gives the quantized GEMMs 128-wide inputs; the last single block is BF16 in nvfp4.
GPU_OVERRIDES = dict(attention_head_dim=64, axes_dims_rope=[16, 16, 16, 16], num_single_layers=3)
# Calibrated NVFP4 input scales leave headroom for activations that quantization shifts.
INPUT_AMAX_MARGIN = 2.0
# Measured worst cases over three input seeds: nvfp4 0.098 / 0.9952, FP8 rowwise 0.044 / 0.9990.
PARITY = {
    "nvfp4": dict(relative_l2=0.15, cosine=0.99),
    "fp8r": dict(relative_l2=0.08, cosine=0.997),
}


def gpu_positions(name: str, length: int) -> torch.Tensor:
    """(t, h, w, l) ids: text on l, images on a scaled h, w grid, references at l = -1."""
    index = torch.arange(length, device="cuda", dtype=torch.float32)
    ids = torch.zeros(length, 4, device="cuda")
    if name == "text":
        ids[:, 3] = index
    else:
        ids[:, 1], ids[:, 2] = (index // 4) * 0.5, (index % 4) * 0.5
        if name == "image_cond":
            ids[:, 3] = -1
    return ids


def gpu_inputs(transformer: Flux3Transformer2DModel) -> dict:
    """One packed sample with target, reference and text tokens."""
    generator = torch.Generator(device="cuda").manual_seed(0)
    channels = transformer.packed_channels
    sample = {}
    for name, length in (("image", 16), ("image_cond", 16), ("text", 8)):
        tokens = torch.randn(length, channels[name], generator=generator, device="cuda")
        sample[name] = (tokens.to(torch.bfloat16), gpu_positions(name, length))
    packed = pack([sample], channels)
    packed.timesteps["image"].fill_(0.6)
    kwargs = packed.kwargs()
    if transformer.flow_map:
        kwargs["target_timesteps"] = torch.tensor([0.3], device="cuda")
    return kwargs


def bf16_transformer(config: dict, weights: dict) -> Flux3Transformer2DModel:
    transformer = Flux3Transformer2DModel(model_config(config)).to("cuda")
    transformer.load_weights({name: value.cuda() for name, value in weights.items()})
    transformer.post_load_weights()
    return transformer.eval()


@torch.inference_mode()
def input_amax(transformer: Flux3Transformer2DModel, kwargs: dict) -> dict[str, float]:
    """Largest input of each Diffusers checkpoint module over one forward."""
    amax = {}

    def record(name):
        def hook(module, args):
            for source in _checkpoint_module_names(name):
                amax[source] = max(amax.get(source, 0.0), args[0].abs().max().item())

        return hook

    handles = [
        module.register_forward_pre_hook(record(name))
        for name, module in transformer.named_modules()
        if isinstance(module, Linear)
    ]
    try:
        transformer(**kwargs)
    finally:
        for handle in handles:
            handle.remove()
    return amax


def load_gpu_pipeline(path: Path, monkeypatch: pytest.MonkeyPatch, **args) -> Flux3Pipeline:
    """Load through PipelineLoader with only the text encoder faked; quantize flags restored."""
    for method in (NVFP4LinearMethod, FP8RowwiseLinearMethod):
        monkeypatch.setattr(method, "use_tunable_quantize", method.use_tunable_quantize)

    def from_pretrained(cls, *args, **kwargs):
        return FakeTextEncoder(device="cuda")

    monkeypatch.setattr(Flux3TextEncoder, "from_pretrained", classmethod(from_pretrained))
    visual_gen_args = VisualGenArgs(
        model=str(path),
        attention_config=AttentionConfig(backend="VANILLA"),
        torch_compile_config=TorchCompileConfig(enable=False, enable_autotune=False),
        **args,
    )
    return PipelineLoader(visual_gen_args).load(skip_warmup=True)


def assert_weight_dtypes(transformer: Flux3Transformer2DModel, quantized_dtype: torch.dtype):
    quantization = transformer.checkpoint_quantization()
    for name, module in transformer.named_modules():
        if isinstance(module, Linear):
            quantized = quantization[_checkpoint_module_names(name)[0]]
            expected = quantized_dtype if quantized else torch.bfloat16
            assert module.weight.dtype == expected, name


def assert_linears_hold_variant_tensors(
    transformer: Flux3Transformer2DModel, tensors: dict[str, torch.Tensor], variant: str
) -> None:
    """Each quantized Linear holds exactly the BFL ``tensors`` of its checkpoint modules."""
    source = bfl_to_diffusers_state_dict(tensors)
    quantization = transformer.checkpoint_quantization()
    for name, module in transformer.named_modules():
        if not isinstance(module, Linear):
            continue
        modules = _checkpoint_module_names(name)
        if not quantization[modules[0]]:
            continue
        device = module.weight.device
        weight = torch.cat([source[f"{m}.weight"] for m in modules]).to(device)
        weight_scale = torch.cat([source[f"{m}.weight_scale"] for m in modules]).to(device)
        if variant == "fp8r":
            assert torch.equal(module.weight.view(torch.uint8), weight.view(torch.uint8)), name
            assert torch.equal(module.weight_scale, weight_scale.reshape(-1)), name
            continue
        weight_scale_2 = source[f"{modules[0]}.weight_scale_2"].reshape(1).to(device)
        input_scale = source[f"{modules[0]}.input_scale"].reshape(1).to(device)
        assert torch.equal(module.weight_scale_2, weight_scale_2), name
        assert torch.equal(module.alpha, input_scale * weight_scale_2), name
        # gate_up_proj rows are reordered after loading for the SwiGLU kernel.
        if name.endswith(".gate_up_proj"):
            continue
        assert torch.equal(module.weight, weight), name
        interleaved = torch.ops.trtllm.block_scale_interleave(weight_scale.view(torch.uint8))
        assert torch.equal(module.weight_scale.view(torch.uint8), interleaved), name


@requires_blackwell
@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("flow_map", FLOW_MAP)
def test_variant_loads_through_pipeline_loader_and_tracks_bf16(
    tmp_path, monkeypatch, flow_map, variant
):
    weights = write_gpu_checkpoint(tmp_path, flow_map, num_layers=2, **GPU_OVERRIDES)
    config = gpu_transformer_config(flow_map, num_layers=2, **GPU_OVERRIDES)
    reference = bf16_transformer(config, weights)
    kwargs = gpu_inputs(reference)
    amax = {
        name: INPUT_AMAX_MARGIN * value for name, value in input_amax(reference, kwargs).items()
    }
    tensors = quantize_variant(config, weights, variant, amax)
    write_variant(tmp_path, config, variant, tensors)
    drop_transformer_weights(tmp_path)

    pipeline = load_gpu_pipeline(tmp_path, monkeypatch, pipeline_config={"variant": variant})
    transformer = pipeline.transformer
    assert bfl_linear_quantization(transformer) == expected_bfl_quantization(config, variant)
    assert_weight_dtypes(transformer, QUANTIZED_DTYPES[variant])
    separate = QKVMode.SEPARATE_QKV if variant == "nvfp4" else QKVMode.FUSE_QKV
    assert transformer.single_transformer_blocks[0].attn.qkv_mode == separate
    assert transformer.single_transformer_blocks[2].attn.qkv_mode == QKVMode.FUSE_QKV
    assert_linears_hold_variant_tensors(transformer, tensors, variant)

    with torch.inference_mode():
        actual = transformer(**kwargs)["x_image"]
        expected = reference(**kwargs)["x_image"]
    assert_parity(actual, expected, **PARITY[variant])

    output = pipeline.forward("a red fox", height=64, width=64, num_inference_steps=2, seed=0)
    assert output.image.shape == (1, 64, 64, 3) and output.image.dtype == torch.uint8


@requires_cuda
@pytest.mark.parametrize("flow_map", FLOW_MAP)
def test_load_time_fp8_pipeline_runs_and_keeps_exclusions_in_bf16(tmp_path, monkeypatch, flow_map):
    weights = write_gpu_checkpoint(tmp_path, flow_map, num_layers=2, **GPU_OVERRIDES)
    config = gpu_transformer_config(flow_map, num_layers=2, **GPU_OVERRIDES)
    reference = bf16_transformer(config, weights)

    pipeline = load_gpu_pipeline(tmp_path, monkeypatch, quant_config=FP8_ROWWISE)
    transformer = pipeline.transformer
    quantization = transformer.checkpoint_quantization()
    excluded = tuple(f"{name}." for name in LOAD_TIME_QUANT_EXCLUDED_MODULES)
    assert quantization == {
        module: not f"{module}.".startswith(excluded) for module in quantization
    }
    assert_weight_dtypes(transformer, torch.float8_e4m3fn)

    kwargs = gpu_inputs(reference)
    with torch.inference_mode():
        actual = transformer(**kwargs)["x_image"]
        expected = reference(**kwargs)["x_image"]
    assert_parity(actual, expected, **PARITY["fp8r"])

    output = pipeline.forward("a red fox", height=64, width=64, num_inference_steps=2, seed=0)
    assert output.image.shape == (1, 64, 64, 3) and output.image.dtype == torch.uint8


@requires_cuda
@pytest.mark.parametrize("flow_map", FLOW_MAP)
def test_fp8r_variant_matches_load_time_fp8_exactly(tmp_path, monkeypatch, flow_map):
    """Both quantize the same Linears with the same rowwise op, so BFL names and scales round-trip."""
    weights = write_gpu_checkpoint(tmp_path, flow_map, num_layers=2, **GPU_OVERRIDES)
    config = gpu_transformer_config(flow_map, num_layers=2, **GPU_OVERRIDES)
    write_variant(tmp_path, config, "fp8r", quantize_variant(config, weights, "fp8r"))
    variant = load_gpu_pipeline(tmp_path, monkeypatch, pipeline_config={"variant": "fp8r"})
    load_time = load_gpu_pipeline(tmp_path, monkeypatch, quant_config=FP8_ROWWISE)
    kwargs = gpu_inputs(variant.transformer)
    with torch.inference_mode():
        actual = variant.transformer(**kwargs)["x_image"]
        expected = load_time.transformer(**kwargs)["x_image"]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
