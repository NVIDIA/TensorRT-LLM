# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FLUX 3 pre-quantized BFL variants and load-time quantization.

CPU cases are cpu_only, the rest need a GPU.
"""

import contextlib
import json
import math
import re
from pathlib import Path

import pytest
import torch
from flux3_test_utils import (
    KLEIN_BASE_CONFIG,
    KLEIN_CONFIG,
    assert_cuda_graph_replays,
    assert_parity,
    bfl_hf_quant_config,
    bfl_variant_excludes,
    bfl_variant_tensor_names,
    diffusers_weights,
    gpu_transformer_config,
    load_gpu_pipeline,
    model_config,
    placeholder_variant_tensors,
    quantize_variant,
    tiny_config,
    to_diffusers_name,
    to_module_name,
    write_checkpoint,
    write_gpu_checkpoint,
    write_variant,
)

from tensorrt_llm._torch.models.modeling_utils import MetaInitMode
from tensorrt_llm._torch.modules.linear import (
    FP8BlockScalesLinearMethod,
    FP8QDQLinearMethod,
    FP8RowwiseLinearMethod,
    Linear,
    NVFP4LinearMethod,
    UnquantizedLinearMethod,
)
from tensorrt_llm._torch.utils import Fp8PerTokenQuantizedTensor, unswizzle_sf

# Importing the models package registers every pipeline.
from tensorrt_llm._torch.visual_gen import (
    models,  # noqa: F401
    prims_ts,
)
from tensorrt_llm._torch.visual_gen import pipeline_loader as pipeline_loader_module
from tensorrt_llm._torch.visual_gen.attention.attention import QKVMode, apply_rotary_emb
from tensorrt_llm._torch.visual_gen.config import DiffusionPipelineConfig
from tensorrt_llm._torch.visual_gen.models.flux3 import fused_gemm_common
from tensorrt_llm._torch.visual_gen.models.flux3.packed import pack
from tensorrt_llm._torch.visual_gen.models.flux3.pipeline_flux3 import (
    Flux3DistilledPipeline,
    Flux3Pipeline,
)
from tensorrt_llm._torch.visual_gen.models.flux3.sm120_nvfp4 import (
    Sm120NVFP4LinearMethod,
    repack_swiglu_rows,
)
from tensorrt_llm._torch.visual_gen.models.flux3.transformer_flux3 import (
    Flux3Attention,
    Flux3Transformer2DModel,
    _checkpoint_module_names,
    _Flux3Block,
    bfl_to_diffusers_key,
    bfl_to_diffusers_state_dict,
    diffusers_to_bfl_module,
)
from tensorrt_llm._torch.visual_gen.models.flux3.variants import (
    apply_variant_quantization,
    collapse_exclusions,
    variant_quantization_config,
)
from tensorrt_llm._torch.visual_gen.pipeline_loader import PipelineLoader
from tensorrt_llm._torch.visual_gen.pipeline_registry import AutoPipeline
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization.mode import QuantAlgo
from tensorrt_llm.visual_gen.args import VisualGenArgs

VARIANTS = ["nvfp4", "fp8r"]
FLOW_MAP = [pytest.param(True, id="klein"), pytest.param(False, id="base")]
DOWNLOAD_PATTERNS = [
    "modular_model_index.json",
    "transformer/*",
    "variants/*",
    "vae/*",
    "text_encoder/*",
    "LICENSE.md",
    "NOTICE.md",
    "README.md",
]
# exclude_modules lengths of the released hf_quant_config.json files (klein, klein-base).
RELEASED_EXCLUDE_COUNTS = {"nvfp4": (75, 73), "fp8r": (39, 37)}
# Block-level exclusions the released variants translate to.
KLEIN_IGNORE = {
    "nvfp4": [
        "context_embedder",
        "image_cond_early_modulation",
        "image_cond_embedder",
        "image_cond_single_modulation",
        "image_early_modulation",
        "image_embedder",
        "image_mode_blocks.0",
        "image_single_modulation",
        "norm_out",
        "proj_out",
        "single_transformer_blocks.24",
        "single_transformer_blocks.25",
        "single_transformer_blocks.26",
        "single_transformer_blocks.27",
        "target_timestep_embedder",
        "timestep_embedder",
        "txt_early_modulation",
        "txt_mode_blocks",
        "txt_single_modulation",
    ],
    "fp8r": [
        "context_embedder",
        "image_cond_embedder",
        "image_embedder",
        "norm_out",
        "proj_out",
        "target_timestep_embedder",
        "timestep_embedder",
        "txt_mode_blocks",
    ],
}
QUANTIZED_METHODS = {"nvfp4": NVFP4LinearMethod, "fp8r": FP8RowwiseLinearMethod}
QUANTIZED_DTYPES = {"nvfp4": torch.uint8, "fp8r": torch.float8_e4m3fn}

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
    """Keep only transformer/config.json, so a variant must load without the BF16 weights."""
    for file in (path / "transformer").iterdir():
        if file.name != "config.json":
            file.unlink()


def write_variant_checkpoint(path: Path, flow_map: bool, variant: str) -> dict:
    """A tiny variant checkpoint whose tensors only have the right names; no BF16 transformer."""
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
        diffusers_to_bfl_module(to_diffusers_name(module)): quantized
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
@pytest.mark.parametrize("variant", ["NVFP4", "fp8", "../nvfp4", "", 4, ["nvfp4"]])
def test_unknown_variants_are_rejected(variant):
    message = re.escape("pipeline_config.variant must be null or one of ['fp8r', 'nvfp4']")
    with pytest.raises(ValueError, match=message):
        apply_variant_quantization("", {}, variant, None)


@pytest.mark.cpu_only
def test_hub_download_fetches_the_registered_patterns(monkeypatch, tmp_path):
    """One download at the requested revision, holding every variant."""
    hf_id = "black-forest-labs/flux-3-image-klein"
    downloads = []

    def download_partial(model, allow_patterns, revision=None):
        downloads.append((model, allow_patterns, revision))
        return tmp_path

    def download_full(*args, **kwargs):
        raise AssertionError("FLUX 3 must not download the whole repository")

    monkeypatch.setattr(pipeline_loader_module, "download_hf_partial", download_partial)
    monkeypatch.setattr(pipeline_loader_module, "download_hf_model", download_full)
    args = VisualGenArgs(model=hf_id, revision="0123abc")
    assert PipelineLoader(args, device="cpu")._resolve_checkpoint_dir(hf_id) == str(tmp_path)
    assert downloads == [(hf_id, DOWNLOAD_PATTERNS, "0123abc")]


# =============================================================================
# hf_quant_config translation on the released structure
# =============================================================================


@pytest.mark.cpu_only
@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("flow_map", FLOW_MAP)
def test_released_variants_translate_to_block_level_exclusions(no_cuda, flow_map, variant):
    """Every BFL-excluded Linear stays BF16 and every other one gets the variant's method."""
    config = released_config(flow_map)
    released_excludes = RELEASED_EXCLUDE_COUNTS[variant][0 if flow_map else 1]
    assert len(bfl_variant_excludes(config, variant)) == released_excludes
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
    # Static NVFP4 q, k and v carry their own scales, so quantized blocks keep them separate,
    # still as self-attention so the requested backend is kept.
    attn = transformer.single_transformer_blocks[0].attn
    assert attn.qkv_mode == (QKVMode.SEPARATE_QKV if variant == "nvfp4" else QKVMode.FUSE_QKV)
    assert attn.separate_qkv_is_self_attention
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
# Variants through DiffusionPipelineConfig
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
        pytest.param({"quant_algo": "NVFP4", "dynamic": True}, id="dict"),
        pytest.param(QuantConfig(quant_algo=QuantAlgo.NVFP4), id="quant_config_object"),
    ],
)
def test_variant_rejects_an_explicit_quant_algo(tmp_path, quant_config):
    write_variant_checkpoint(tmp_path, True, "nvfp4")
    message = "FLUX 3 variant 'nvfp4' is pre-quantized; remove quant_config.quant_algo"
    with pytest.raises(ValueError, match=re.escape(message)):
        load_config(tmp_path, pipeline_config={"variant": "nvfp4"}, quant_config=quant_config)


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
    del tensors[f"content_mode_blocks.image_cond.1.mlp_in.{dropped}"]
    tensors["txt_mode_blocks.0.q_proj.weight_scale"] = torch.zeros(2)
    write_variant(tmp_path, config, variant, tensors)
    pipeline = build_pipeline(tmp_path, pipeline_config={"variant": variant})
    message = (
        f"quantized layers miss 1 [image_cond_mode_blocks.1.ff.linear_in.{dropped}], "
        "BF16 layers carry 1 [txt_mode_blocks.0.attn.to_q.weight_scale]"
    )
    with pytest.raises(ValueError, match=re.escape(message)):
        pipeline.load_transformer_weights(str(tmp_path))


# =============================================================================
# Load-time quantization
# =============================================================================

# Each load-time algorithm's Linear method and the BFL variant whose BF16 Linears it keeps.
LOAD_TIME = {
    "FP8": (FP8QDQLinearMethod, "fp8r"),
    "FP8_BLOCK_SCALES": (FP8BlockScalesLinearMethod, "fp8r"),
    "FP8_PER_CHANNEL_PER_TOKEN": (FP8RowwiseLinearMethod, "fp8r"),
    "NVFP4": (NVFP4LinearMethod, "nvfp4"),
}
# Per-rank widths of the block-scaled GEMMs and their TP shards.
BLOCK_ALIGNMENT = {"FP8_BLOCK_SCALES": 128, "NVFP4": 32}


def load_time_quant(quant_algo: str, **extra) -> dict:
    return {"quant_algo": quant_algo, "dynamic": True, **extra}


def load_time_transformer(config: dict, quant_algo: str, **kwargs) -> Flux3Transformer2DModel:
    """A meta-device transformer with ``quant_algo`` parsed as the loader parses it."""
    quant_config, _, dynamic_weights, dynamic_activations = (
        DiffusionPipelineConfig.load_diffusion_quant_config(load_time_quant(quant_algo))
    )
    assert dynamic_weights
    with MetaInitMode():
        return Flux3Transformer2DModel(
            model_config(
                config,
                quant_config=quant_config,
                dynamic_weight_quant=True,
                force_dynamic_quantization=dynamic_activations,
                skip_create_weights_in_init=True,
                **kwargs,
            )
        )


@pytest.mark.cpu_only
@pytest.mark.parametrize("quant_algo", list(LOAD_TIME))
@pytest.mark.parametrize("flow_map", FLOW_MAP)
def test_load_time_quant_keeps_the_bf16_linears_of_its_variant(no_cuda, flow_map, quant_algo):
    """On the released configs, FP8 algorithms keep fp8r's BF16 Linears and NVFP4 keeps nvfp4's."""
    config = released_config(flow_map)
    method, variant = LOAD_TIME[quant_algo]
    transformer = load_time_transformer(config, quant_algo)
    assert bfl_linear_quantization(transformer) == expected_bfl_quantization(config, variant)
    assert_methods_follow_quantization(transformer, method)
    # Load-time recipes keep QKV fused; NVFP4 quantizes q, k and v under one global scale.
    assert transformer.single_transformer_blocks[0].attn.qkv_mode == QKVMode.FUSE_QKV
    if flow_map:
        excluded = transformer.model_config.quant_config.exclude_modules
        assert sorted(excluded) == KLEIN_IGNORE[variant]


@pytest.mark.cpu_only
@pytest.mark.parametrize("quant_algo", list(LOAD_TIME))
def test_load_time_quant_adds_user_exclusions_to_the_defaults(no_cuda, tmp_path, quant_algo):
    config = tiny_config(flow_map=True)
    write_checkpoint(tmp_path, config, diffusers_weights(config))
    user_quant = load_time_quant(quant_algo, ignore=["single_transformer_blocks.0"])
    pipeline_config = load_config(tmp_path, quant_config=user_quant)
    assert pipeline_config.dynamic_weight_quant
    with MetaInitMode():
        pipeline = AutoPipeline.from_config(pipeline_config, str(tmp_path))

    method, variant = LOAD_TIME[quant_algo]
    expected = expected_bfl_quantization(config, variant)
    expected.update({module: False for module in expected if module.startswith("single_blocks.0.")})
    assert bfl_linear_quantization(pipeline.transformer) == expected
    assert_methods_follow_quantization(pipeline.transformer, method)
    assert pipeline_config.quant_config.exclude_modules == ["single_transformer_blocks.0"]


@pytest.mark.cpu_only
# Every tp > 1 builds the same Linears, and tp 2 and 4 shards are multiples of tp 8 shards.
@pytest.mark.parametrize("tp_size", [1, 8])
@pytest.mark.parametrize("quant_algo", list(BLOCK_ALIGNMENT))
@pytest.mark.parametrize("flow_map", FLOW_MAP)
def test_released_quantized_linears_align_at_every_tp_size(no_cuda, flow_map, quant_algo, tp_size):
    mapping = Mapping(world_size=tp_size, tp_size=tp_size, rank=tp_size - 1)
    transformer = load_time_transformer(released_config(flow_map), quant_algo, mapping=mapping)
    alignment = BLOCK_ALIGNMENT[quant_algo]
    shapes = {
        name: (module.in_features, module.out_features)
        for name, module in transformer.named_modules()
        if isinstance(module, Linear)
        and getattr(module, "quant_config", None) is not None
        and module.quant_config.quant_algo is not None
    }
    assert shapes
    misaligned = {
        name: shape for name, shape in shapes.items() if any(d % alignment for d in shape)
    }
    assert not misaligned


@pytest.mark.cpu_only
def test_unsupported_load_time_quant_is_rejected_naming_the_supported_algorithms(tmp_path):
    config = tiny_config(flow_map=True)
    write_checkpoint(tmp_path, config, diffusers_weights(config))
    message = (
        "FLUX 3 quantizes BF16 weights on load with quant_config.quant_algo FP8, "
        "FP8_BLOCK_SCALES, FP8_PER_CHANNEL_PER_TOKEN, NVFP4, got W4A16_AWQ."
    )
    with pytest.raises(ValueError, match=re.escape(message)):
        build_pipeline(tmp_path, quant_config=load_time_quant("W4A16_AWQ"))


# =============================================================================
# GPU: tiny checkpoints through PipelineLoader
# =============================================================================

# Head size 64 gives the quantized GEMMs 128-wide inputs; the last single block is BF16 in nvfp4.
GPU_OVERRIDES = dict(attention_head_dim=64, axes_dims_rope=[16, 16, 16, 16], num_single_layers=3)
# Calibrated NVFP4 input scales leave headroom for activations that quantization shifts.
INPUT_AMAX_MARGIN = 2.0
# Measured worst cases over three input seeds: nvfp4 0.098 / 0.9952, FP8 rowwise 0.044 / 0.9990;
# load-time NVFP4 0.096 / 0.9954, FP8 0.050 / 0.9988, FP8 blockwise 0.051 / 0.9987.
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
    kwargs = {**pack([sample], channels).kwargs(), "timestep": torch.tensor([0.6], device="cuda")}
    if transformer.flow_map:
        kwargs["target_timestep"] = torch.tensor([0.3], device="cuda")
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
            for source in map(to_diffusers_name, _checkpoint_module_names(name)):
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
    source = {to_module_name(k): v for k, v in bfl_to_diffusers_state_dict(tensors).items()}
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


@requires_cuda
@pytest.mark.parametrize("variant", [pytest.param("nvfp4", marks=requires_blackwell), "fp8r"])
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
@pytest.mark.parametrize(
    "quant_algo",
    [
        pytest.param("FP8", id="fp8"),
        pytest.param("FP8_BLOCK_SCALES", id="fp8_block_scales"),
        pytest.param("FP8_PER_CHANNEL_PER_TOKEN", id="fp8_rowwise"),
        pytest.param("NVFP4", id="nvfp4", marks=requires_blackwell),
    ],
)
@pytest.mark.parametrize("flow_map", FLOW_MAP)
def test_load_time_quant_pipeline_runs_and_keeps_its_bf16_linears(
    tmp_path, monkeypatch, flow_map, quant_algo
):
    weights = write_gpu_checkpoint(tmp_path, flow_map, num_layers=2, **GPU_OVERRIDES)
    config = gpu_transformer_config(flow_map, num_layers=2, **GPU_OVERRIDES)
    reference = bf16_transformer(config, weights)

    pipeline = load_gpu_pipeline(tmp_path, monkeypatch, quant_config=load_time_quant(quant_algo))
    transformer = pipeline.transformer
    variant = LOAD_TIME[quant_algo][1]
    assert bfl_linear_quantization(transformer) == expected_bfl_quantization(config, variant)
    assert_weight_dtypes(transformer, QUANTIZED_DTYPES[variant])

    kwargs = gpu_inputs(reference)
    with torch.inference_mode():
        actual = transformer(**kwargs)["x_image"]
        expected = reference(**kwargs)["x_image"]
    assert_parity(actual, expected, **PARITY[variant])

    output = pipeline.forward("a red fox", height=64, width=64, num_inference_steps=2, seed=0)
    assert output.image.shape == (1, 64, 64, 3) and output.image.dtype == torch.uint8


# =============================================================================
# Fused block GEMMs: Prims-TS on SM100/SM103/SM107, CuTe DSL on SM120/SM121
# =============================================================================

PRIMS_TS_ARCHES = ((10, 0), (10, 3), (10, 7))
SM120_ARCHES = ((12, 0), (12, 1))


def _gpu_in(arches) -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability() in arches


requires_prims_ts = pytest.mark.skipif(
    not _gpu_in(PRIMS_TS_ARCHES), reason="Prims-TS GEMMs need SM100, SM103 or SM107"
)
requires_sm120 = pytest.mark.skipif(
    not _gpu_in(SM120_ARCHES), reason="the FLUX 3 NVFP4 CuTe DSL kernels need SM120 or SM121"
)
# The fused QK-norm + RoPE epilogue needs head_dim 128; the FP8 kernels need N % 512 == 0.
PRIMS_TS_OVERRIDES = dict(
    attention_head_dim=128,
    num_attention_heads=4,
    axes_dims_rope=[32, 32, 32, 32],
    num_single_layers=3,
)


def _assert_prims_ts_installed():
    assert prims_ts.fp8_gemm_available() and prims_ts.nvfp4_qkv_available(), (
        "flashinfer.prims_ts.gemm is missing; requirements.txt pins flashinfer-python 0.7.3"
    )


def _write_variant_checkpoint(tmp_path, variant):
    """A calibrated klein ``variant`` checkpoint; returns its BF16 reference and inputs."""
    weights = write_gpu_checkpoint(tmp_path, True, num_layers=2, **PRIMS_TS_OVERRIDES)
    config = gpu_transformer_config(True, num_layers=2, **PRIMS_TS_OVERRIDES)
    reference = bf16_transformer(config, weights)
    kwargs = gpu_inputs(reference)
    amax = {
        name: INPUT_AMAX_MARGIN * value for name, value in input_amax(reference, kwargs).items()
    }
    write_variant(tmp_path, config, variant, quantize_variant(config, weights, variant, amax))
    drop_transformer_weights(tmp_path)
    return reference, kwargs


def _variant_pipeline(tmp_path, monkeypatch, variant):
    reference, kwargs = _write_variant_checkpoint(tmp_path, variant)
    pipeline = load_gpu_pipeline(tmp_path, monkeypatch, pipeline_config={"variant": variant})
    return pipeline, reference, kwargs


def _trtllm_transformer(tmp_path, monkeypatch, variant):
    """The same checkpoint loaded with every block GEMM on TRT-LLM."""
    with monkeypatch.context() as patch:
        patch.setattr(fused_gemm_common, "FUSED_GEMMS", False)
        return load_gpu_pipeline(
            tmp_path, monkeypatch, pipeline_config={"variant": variant}
        ).transformer


def _blocks(transformer, method):
    return [
        module
        for module in transformer.modules()
        if isinstance(module, _Flux3Block)
        and isinstance(module.ff.gate_up_proj.quant_method, method)
    ]


def _block_inputs(attn, tokens=96, batch=1):
    generator = torch.Generator(device="cuda").manual_seed(0)
    hidden = torch.randn(
        batch, tokens, attn.hidden_size, device="cuda", dtype=torch.bfloat16, generator=generator
    )
    angles = torch.rand(tokens, attn.head_dim // 2, device="cuda", generator=generator) * math.tau
    cos = angles.cos().repeat_interleave(2, dim=-1).view(1, tokens, 1, attn.head_dim)
    sin = angles.sin().repeat_interleave(2, dim=-1).view(1, tokens, 1, attn.head_dim)
    return hidden, (cos, sin)


@requires_prims_ts
def test_nvfp4_qkv_runs_fused_on_prims_ts_and_tracks_trtllm(tmp_path, monkeypatch):
    _assert_prims_ts_installed()
    pipeline, reference, kwargs = _variant_pipeline(tmp_path, monkeypatch, "nvfp4")
    transformer = pipeline.transformer
    separate = [
        module
        for module in transformer.modules()
        if isinstance(module, Flux3Attention) and module.qkv_mode == QKVMode.SEPARATE_QKV
    ]
    assert separate and all(attn.prims_ts_qkv is not None for attn in separate)
    # The TRT-LLM projections read the packed buffer rather than keeping their own copies.
    packed = separate[0].prims_ts_qkv.weight.untyped_storage().data_ptr()
    assert separate[0].to_k.weight.untyped_storage().data_ptr() == packed
    # Q/K/V and gate/up see the same input with the same scale, so it is quantized once.
    fused_blocks = _blocks(transformer, NVFP4LinearMethod)
    assert all(
        block.share_nvfp4_quant for block in fused_blocks if block.attn.prims_ts_qkv is not None
    )

    # One fused GEMM + QK-norm + RoPE matches the three TRT-LLM GEMMs and eager norm/RoPE.
    attn, tokens = separate[0], 96
    hidden, (cos, sin) = _block_inputs(attn, tokens)
    heads = attn.local_num_attention_heads
    with torch.inference_mode():
        fused = attn.prims_ts_qkv(hidden, cos, sin)
        q, k, v = (t.view(1, tokens, heads, attn.head_dim) for t in attn.get_qkv(hidden))
        q, k = attn.apply_qk_norm(q, k)
        expected = (apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin), v)
        # One sequence's RoPE table broadcasts over a batch.
        batched = attn.prims_ts_qkv(hidden.expand(2, -1, -1).contiguous(), cos, sin)
        assert_cuda_graph_replays(lambda: attn.prims_ts_qkv(hidden, cos, sin))
    for name, actual, reference_plane, batch in zip("qkv", fused, expected, batched):
        assert actual.shape == (1, tokens, heads * attn.head_dim), name
        assert_parity(
            actual, reference_plane.flatten(2).to(actual.dtype), relative_l2=0.02, cosine=0.9995
        )
        assert_parity(batch[1:], actual, relative_l2=1e-3, cosine=0.99999)

    with torch.inference_mode():
        actual = transformer(**kwargs)["x_image"]
        expected_image = reference(**kwargs)["x_image"]
    assert_parity(actual, expected_image, **PARITY["nvfp4"])


@requires_prims_ts
def test_fused_gemms_can_be_disabled(tmp_path, monkeypatch):
    _assert_prims_ts_installed()
    _write_variant_checkpoint(tmp_path, "nvfp4")
    modules = list(_trtllm_transformer(tmp_path, monkeypatch, "nvfp4").modules())
    for name in ("prims_ts_qkv", "prims_ts_fp8", "sm120_nvfp4"):
        assert not any(getattr(m, name, None) is not None for m in modules), name
    assert not any(getattr(m, "share_nvfp4_quant", False) for m in modules)


@requires_prims_ts
def test_fp8r_blocks_run_on_prims_ts_and_track_trtllm(tmp_path, monkeypatch):
    _assert_prims_ts_installed()
    pipeline, reference, kwargs = _variant_pipeline(tmp_path, monkeypatch, "fp8r")
    transformer = pipeline.transformer
    blocks = _blocks(transformer, FP8RowwiseLinearMethod)
    assert blocks and all(block.prims_ts_fp8 is not None for block in blocks)
    # The plain projections keep TRT-LLM's FP8-rowwise GEMM.
    for block in blocks:
        for layer in (block.attn.to_out[0], block.ff.down_proj):
            assert type(layer.quant_method) is FP8RowwiseLinearMethod
    # Prims-TS repacks gate/up in place, so the TRT-LLM reference loads separately.
    trtllm = _trtllm_transformer(tmp_path, monkeypatch, "fp8r")
    trtllm_blocks = _blocks(trtllm, FP8RowwiseLinearMethod)
    assert trtllm_blocks and all(block.prims_ts_fp8 is None for block in trtllm_blocks)

    # Fused QKV + QK-norm + RoPE and gate/up + SwiGLU (TRT-LLM out/down) match the TRT-LLM block.
    block, trtllm_block = blocks[0], trtllm_blocks[0]
    hidden, rope = _block_inputs(block.attn)
    with torch.inference_mode():
        actual = block.prims_ts_fp8(hidden, rope, None, None)
        attended = trtllm_block.attn.attend(hidden, rope)
        mlp = trtllm_block.ff(hidden.reshape(-1, hidden.shape[-1])).reshape_as(hidden)
        expected = trtllm_block.attn.to_out[0](attended) + mlp
        # One sequence's RoPE table broadcasts over a batch.
        batched = block.prims_ts_fp8(hidden.expand(2, -1, -1).contiguous(), rope, None, None)
        assert_cuda_graph_replays(lambda: block.prims_ts_fp8(hidden, rope, None, None))
    assert_parity(actual, expected, relative_l2=0.03, cosine=0.999)
    assert_parity(batched[1:], actual, relative_l2=1e-3, cosine=0.99999)

    with torch.inference_mode():
        image = transformer(**kwargs)["x_image"]
        expected_image = reference(**kwargs)["x_image"]
        trtllm_image = trtllm(**kwargs)["x_image"]
    assert_parity(image, expected_image, **PARITY["fp8r"])
    assert_parity(image, trtllm_image, relative_l2=0.05, cosine=0.998)


@requires_cuda
@pytest.mark.parametrize("fuse_qk_norm_rope", [True, False], ids=["fused_rope", "eager_rope"])
def test_fp8r_trtllm_blocks_quantize_once_for_qkv_and_gate_up(
    tmp_path, monkeypatch, fuse_qk_norm_rope
):
    """On TRT-LLM's GEMMs, Q/K/V and gate/up share one per-token FP8 quantize, bit for bit."""
    _, kwargs = _write_variant_checkpoint(tmp_path, "fp8r")
    transformer = _trtllm_transformer(tmp_path, monkeypatch, "fp8r")
    blocks = _blocks(transformer, FP8RowwiseLinearMethod)
    assert blocks and all(block.share_fp8_quant for block in blocks)
    assert not any(block.share_fp8_quant for block in transformer.txt_mode_blocks)
    inputs = []
    for block in blocks:
        block.attn.fuse_qk_norm_rope = fuse_qk_norm_rope
        for layer in (block.attn.qkv_proj, block.ff.gate_up_proj):
            layer.register_forward_pre_hook(lambda _, args: inputs.append(args[0]))

    hidden, rope = _block_inputs(blocks[0].attn)
    mod = tuple(torch.randn_like(hidden) * 0.1 for _ in range(3))

    def run():
        return blocks[0](hidden, rope, mod), transformer(**kwargs)["x_image"]

    with torch.inference_mode():
        shared = run()
        shared_inputs = inputs.copy()
        inputs.clear()
        for block in blocks:
            block.share_fp8_quant = False
        unshared = run()
    # Each block call hands Q/K/V and then gate/up one quantization; unshared, the BF16 input.
    assert len(shared_inputs) == 2 * (1 + len(blocks))
    for qkv_input, gate_up_input in zip(shared_inputs[::2], shared_inputs[1::2]):
        assert isinstance(qkv_input, Fp8PerTokenQuantizedTensor) and gate_up_input is qkv_input
    assert all(isinstance(x, torch.Tensor) for x in inputs)
    for actual, expected in zip(shared, unshared):
        assert torch.equal(actual, expected)


@pytest.mark.cpu_only
@pytest.mark.parametrize("tp_size", [1, 2])
def test_fp8r_blocks_share_one_quantize_under_tensor_parallelism(no_cuda, tp_size):
    """Column-parallel Q/K/V and gate/up both read the replicated block input."""
    mapping = Mapping(world_size=tp_size, tp_size=tp_size, rank=tp_size - 1)
    transformer = load_time_transformer(KLEIN_CONFIG, "FP8_PER_CHANNEL_PER_TOKEN", mapping=mapping)
    transformer._prepare_fused_gemms()
    blocks = [module for module in transformer.modules() if isinstance(module, _Flux3Block)]
    fp8_blocks = _blocks(transformer, FP8RowwiseLinearMethod)
    assert fp8_blocks and len(fp8_blocks) < len(blocks)
    for block in blocks:
        assert block.share_fp8_quant == (block in fp8_blocks)


@requires_prims_ts
@pytest.mark.parametrize("variant", ["nvfp4", "fp8r"])
def test_fused_blocks_compile_once(tmp_path, monkeypatch, variant):
    """Per-layer checkpoint scales stay out of the traced graph: blocks share one compile."""
    _assert_prims_ts_installed()
    pipeline, _, _ = _variant_pipeline(tmp_path, monkeypatch, variant)
    blocks = [
        block
        for block in pipeline.transformer.single_transformer_blocks
        if (block.attn.prims_ts_qkv if variant == "nvfp4" else block.prims_ts_fp8) is not None
    ]
    assert len(blocks) >= 2
    if variant == "nvfp4":
        assert len({float(b.attn.to_q.weight_scale_2) for b in blocks}) > 1
    hidden, rope = _block_inputs(blocks[0].attn)
    mod = tuple(torch.randn_like(hidden) * 0.1 for _ in range(3))
    from torch._dynamo.testing import CompileCounter

    torch._dynamo.reset()
    counter = CompileCounter()
    with torch.inference_mode():
        for index, block in enumerate(blocks):
            torch.compile(block, backend=counter)(hidden, rope, mod)
            if index == 0:
                first = counter.frame_count
    assert counter.frame_count == first


@pytest.mark.cpu_only
def test_prims_ts_autotune_is_opt_in(monkeypatch):
    """Pinned tiles by default; TRTLLM_FLUX3_PRIMS_TS_AUTOTUNE=1 leaves the FP8 tiles to FlashInfer."""
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: (10, 0))
    monkeypatch.delenv("TRTLLM_FLUX3_PRIMS_TS_AUTOTUNE", raising=False)
    device = torch.device("cuda")

    def configs():
        return (
            prims_ts.config_for(device, "fp8_e4m3", "swiglu", 4096, 3072, 18432),
            prims_ts.config_for(device, "fp8_e4m3", "qkv_qknorm_rope", 4096, 3072, 9216, 128),
            prims_ts.config_for(
                device, "nvfp4_e2m1", "qkv_qknorm_rope", 4096, 3072, 9216, 128, qkv_scale=True
            ),
        )

    # The klein tables' 8192-token rows.
    swiglu, qkv, planes = configs()
    assert (swiglu.cluster_shape, swiglu.ab_stages, swiglu.use_tma_store) == ((4, 4, 1), 6, False)
    assert qkv.cluster_shape == (2, 2, 1) and qkv.use_tma_store and not qkv.separate_qkv_output
    assert planes.separate_qkv_output and planes.tmem_overlap and not planes.use_tma_store
    assert isinstance(prims_ts.autotune_context(device, tune=True), contextlib.nullcontext)

    monkeypatch.setenv("TRTLLM_FLUX3_PRIMS_TS_AUTOTUNE", "1")
    # FlashInfer's tuned configs write packed Q|K|V, so the NVFP4 planes keep their tile.
    assert configs() == (None, None, planes)


@requires_prims_ts
def test_prims_ts_autotune_tunes_fp8r_blocks_and_tracks_trtllm(tmp_path, monkeypatch):
    """FP8R blocks tune, then load the saved tactics and replay them eagerly and in a graph.

    The NVFP4 Q/K/V planes keep their pinned tile under the same contexts.
    """
    _assert_prims_ts_installed()
    from flashinfer.autotuner import AutoTuner
    from flashinfer.prims_ts.gemm.runner import PrimsTsGemmRunner

    cache = tmp_path / "cache"
    monkeypatch.setenv("TRTLLM_FLUX3_PRIMS_TS_AUTOTUNE", "1")
    monkeypatch.setenv("TRTLLM_FLUX3_PRIMS_TS_CACHE_DIR", str(cache))
    # One bucket above the 96 test tokens and two candidates per GEMM keep compiles few.
    monkeypatch.setenv("TRTLLM_FLUX3_PRIMS_TS_TUNING_BUCKETS", "128")
    candidates = PrimsTsGemmRunner.get_valid_tactics
    monkeypatch.setattr(PrimsTsGemmRunner, "get_valid_tactics", lambda *args: candidates(*args)[:2])
    fp8r, _, _ = _variant_pipeline(tmp_path / "fp8r", monkeypatch, "fp8r")
    trtllm = _trtllm_transformer(tmp_path / "fp8r", monkeypatch, "fp8r")
    nvfp4, _, _ = _variant_pipeline(tmp_path / "nvfp4", monkeypatch, "nvfp4")
    block = _blocks(fp8r.transformer, FP8RowwiseLinearMethod)[0]
    trtllm_block = _blocks(trtllm, FP8RowwiseLinearMethod)[0]
    attn = next(m for m in nvfp4.transformer.modules() if getattr(m, "prims_ts_qkv", None))
    hidden, rope = _block_inputs(block.attn)
    nvfp4_hidden, (cos, sin) = _block_inputs(attn)

    def fused():
        planes = attn.prims_ts_qkv(nvfp4_hidden, cos, sin)
        return block.prims_ts_fp8(hidden, rope, None, None), *planes

    tuner = AutoTuner.get()
    choose_one, choices = tuner.choose_one, []

    def record(op, *args, **kwargs):
        runner, tactic = choose_one(op, *args, **kwargs)
        choices.append((op, tactic))
        return runner, tactic

    monkeypatch.setattr(tuner, "choose_one", record)
    with torch.inference_mode():
        with prims_ts.autotune_context(hidden.device, tune=True):
            fused()
        choices.clear()
        with prims_ts.autotune_context(hidden.device, load=True):
            actual = assert_cuda_graph_replays(fused)
        attended = trtllm_block.attn.attend(hidden, rope)
        mlp = trtllm_block.ff(hidden.reshape(-1, hidden.shape[-1])).reshape_as(hidden)
        expected = trtllm_block.attn.to_out[0](attended) + mlp
        shape = (*nvfp4_hidden.shape[:2], attn.local_num_attention_heads, attn.head_dim)
        q, k, v = (t.view(shape) for t in attn.get_qkv(nvfp4_hidden))
        q, k = attn.apply_qk_norm(q, k)
        expected_planes = (apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin), v)
    # Every lookup hits a tuned tactic; the NVFP4 planes never reach the tuner.
    assert {op for op, _ in choices} == {"prims_ts_fp8_swiglu", "prims_ts_fp8_qkv"}
    assert all(tactic != -1 for _, tactic in choices)
    (saved,) = cache.rglob("*.json")
    assert "prims_ts_fp8_swiglu" in saved.read_text()
    assert_parity(actual[0], expected, relative_l2=0.03, cosine=0.999)
    for plane, reference in zip(actual[1:], expected_planes):
        assert_parity(plane, reference.flatten(2).to(plane.dtype), relative_l2=0.02, cosine=0.9995)


@pytest.mark.cpu_only
def test_swiglu_repack_pairs_rows_and_block_scales() -> None:
    """[gate; up] NVFP4 rows and their 128x4 scales become [up, gate] pairs, once."""
    rows, reduction = 512, 256
    torch.manual_seed(0)
    layer = torch.nn.Module()
    weight = torch.randint(0, 256, (rows, reduction // 2), dtype=torch.uint8)
    scales = torch.randint(0, 127, (rows, reduction // 16), dtype=torch.uint8)
    layer.weight = torch.nn.Parameter(weight.clone(), requires_grad=False)
    layer.weight_scale = torch.nn.Parameter(
        torch.ops.trtllm.block_scale_interleave(scales).reshape(-1), requires_grad=False
    )
    repack_swiglu_rows(layer)
    half = rows // 2
    expected_rows = torch.stack((weight[half:], weight[:half]), dim=1).reshape_as(weight)
    expected_scales = torch.stack((scales[half:], scales[:half]), dim=1).reshape_as(scales)
    assert torch.equal(layer.weight, expected_rows)
    assert torch.equal(unswizzle_sf(layer.weight_scale, rows, reduction)[:rows], expected_scales)
    repack_swiglu_rows(layer)  # idempotent
    assert torch.equal(layer.weight, expected_rows)


@requires_sm120
def test_nvfp4_blocks_run_on_sm120_and_track_trtllm(tmp_path, monkeypatch):
    pipeline, reference, kwargs = _variant_pipeline(tmp_path, monkeypatch, "nvfp4")
    transformer = pipeline.transformer
    blocks = _blocks(transformer, NVFP4LinearMethod)
    assert blocks and all(block.sm120_nvfp4 is not None for block in blocks)
    # The TRT-LLM projections read the packed Q/K/V buffer rather than own copies.
    packed = blocks[0].sm120_nvfp4.qkv_weight.untyped_storage().data_ptr()
    assert blocks[0].attn.to_k.weight.untyped_storage().data_ptr() == packed
    # gate/up is repacked in place, so the TRT-LLM reference loads separately.
    trtllm = _trtllm_transformer(tmp_path, monkeypatch, "nvfp4")
    trtllm_blocks = _blocks(trtllm, NVFP4LinearMethod)
    assert trtllm_blocks and all(block.sm120_nvfp4 is None for block in trtllm_blocks)
    # Attention-out and down stay on Linear.forward, with the SM120 GEMM as their method.
    assert all(
        type(layer.quant_method) is Sm120NVFP4LinearMethod
        for block in blocks
        for layer in (block.attn.to_out[0], block.ff.down_proj)
    )
    assert all(
        type(layer.quant_method) is NVFP4LinearMethod
        for block in trtllm_blocks
        for layer in (block.attn.to_out[0], block.ff.down_proj)
    )

    block, trtllm_block = blocks[0], trtllm_blocks[0]
    hidden, rope = _block_inputs(block.attn)
    with torch.inference_mode():
        actual = block.sm120_nvfp4(hidden, rope, None, None)
        attended = trtllm_block.attn.attend(hidden, rope)
        mlp = trtllm_block.ff(hidden.reshape(-1, hidden.shape[-1])).reshape_as(hidden)
        expected = trtllm_block.attn.to_out[0](attended) + mlp
        batched = block.sm120_nvfp4(hidden.expand(2, -1, -1).contiguous(), rope, None, None)
        assert_cuda_graph_replays(lambda: block.sm120_nvfp4(hidden, rope, None, None))
    assert_parity(actual, expected, relative_l2=0.05, cosine=0.998)
    assert_parity(batched[1:], actual, relative_l2=1e-3, cosine=0.99999)

    with torch.inference_mode():
        image = transformer(**kwargs)["x_image"]
        expected_image = reference(**kwargs)["x_image"]
    assert_parity(image, expected_image, **PARITY["nvfp4"])
