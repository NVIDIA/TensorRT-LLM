# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import load_file, save_file

from tensorrt_llm._torch.visual_gen.checkpoint_adapter import ResolvedCheckpoint
from tensorrt_llm._torch.visual_gen.checkpoints.comfy_bundle import prepare_comfy_bundle
from tensorrt_llm._torch.visual_gen.checkpoints.comfy_flux import convert_comfy_flux_weights
from tensorrt_llm._torch.visual_gen.checkpoints.comfy_h3 import convert_comfy_h3_weights
from tensorrt_llm._torch.visual_gen.checkpoints.comfy_quant import (
    convert_comfy_nvfp4,
    inspect_comfy_quantization,
)
from tensorrt_llm._torch.visual_gen.checkpoints.weight_loader import WeightLoader
from tensorrt_llm._torch.visual_gen.config import DiffusionPipelineConfig
from tensorrt_llm._torch.visual_gen.models.flux import checkpoint as flux_checkpoint
from tensorrt_llm._torch.visual_gen.models.flux.joint_proj import _split_nvfp4_projection
from tensorrt_llm._torch.visual_gen.models.minimax_h3 import checkpoint as h3_checkpoint
from tensorrt_llm._torch.visual_gen.models.minimax_h3.adaln_curves import interpolate_adaln_curve
from tensorrt_llm._torch.visual_gen.models.minimax_h3.pipeline_minimax_h3 import MiniMaxH3Pipeline
from tensorrt_llm._torch.visual_gen.pipeline_loader import PipelineLoader
from tensorrt_llm._torch.visual_gen.pipeline_registry import PIPELINE_REGISTRY, AutoPipeline
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.quantization.mode import QuantAlgo
from tensorrt_llm.visual_gen import VisualGenArgs

pytestmark = pytest.mark.cpu_only


def test_select_one_checkpoint_file(tmp_path: Path) -> None:
    chosen = tmp_path / "chosen.safetensors"
    alternative = tmp_path / "alternative.safetensors"
    save_file({"weight": torch.ones(2, 2)}, chosen)
    save_file({"weight": torch.zeros(2, 2)}, alternative)
    loaded = WeightLoader().load_weights(str(chosen), Mapping())
    torch.testing.assert_close(loaded["weight"], torch.ones(2, 2))


def _metadata(config: dict | None = None) -> torch.Tensor:
    return torch.tensor(list(json.dumps(config or {"format": "nvfp4"}).encode()), dtype=torch.uint8)


def _checkpoint(rows: int, cols: int) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Construct cuBLAS scale tiles by scalar addressing, independently of the importer."""
    scale_cols = cols // 16
    padded_rows = (rows + 127) // 128 * 128
    padded_cols = (scale_cols + 3) // 4 * 4
    linear = (torch.arange(rows * scale_cols) % 120).to(torch.uint8).reshape(rows, scale_cols)
    tiled = torch.zeros(padded_rows * padded_cols, dtype=torch.uint8)
    for row in range(rows):
        for col in range(scale_cols):
            tile = (row // 128) * (padded_cols // 4) + col // 4
            offset = tile * 512 + (row % 32) * 16 + ((row % 128) // 32) * 4 + col % 4
            tiled[offset] = linear[row, col]
    return {
        "proj.weight": (torch.arange(rows * cols // 2) % 256)
        .to(torch.uint8)
        .reshape(rows, cols // 2),
        "proj.weight_scale": tiled.reshape(padded_rows, padded_cols).view(torch.float8_e4m3fn),
        "proj.weight_scale_2": torch.tensor(0.125),
        "proj.comfy_quant": _metadata(),
        "norm.weight": torch.ones(rows, dtype=torch.bfloat16),
    }, linear


@pytest.mark.parametrize("rows,cols", [(1, 16), (33, 80), (128, 64), (129, 144), (256, 256)])
def test_decoded_weights_preserved(rows: int, cols: int) -> None:
    weights, expected_scales = _checkpoint(rows, cols)
    original = {name: value.clone() for name, value in weights.items()}
    result = convert_comfy_nvfp4(weights)
    assert "proj.comfy_quant" not in result
    torch.testing.assert_close(result["proj.weight_scale"].view(torch.uint8), expected_scales)
    # Decode each convention independently. Include all signed E2M1 values, including -0.
    lut = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6])
    source = weights["proj.weight"].long()
    target = result["proj.weight"].long()
    before = torch.stack((lut[source // 16], lut[source % 16]), dim=-1).reshape(rows, cols)
    after = torch.stack((lut[target % 16], lut[target // 16]), dim=-1).reshape(rows, cols)
    scales = expected_scales.view(torch.float8_e4m3fn).float().repeat_interleave(16, dim=1)
    torch.testing.assert_close(before * scales * 0.125, after * scales * 0.125, rtol=0, atol=0)
    assert result["norm.weight"] is weights["norm.weight"]
    assert result["proj.weight_scale_2"] is weights["proj.weight_scale_2"]
    for name, value in original.items():
        torch.testing.assert_close(
            weights[name].reshape(-1).view(torch.uint8), value.reshape(-1).view(torch.uint8)
        )
    again = convert_comfy_nvfp4(result)
    assert all(again[name] is value for name, value in result.items())


def test_detection_across_shards(tmp_path: Path) -> None:
    weights, _ = _checkpoint(129, 80)
    first, second = tmp_path / "a.safetensors", tmp_path / "b.safetensors"
    save_file({key: value for key, value in weights.items() if key.endswith("weight")}, first)
    save_file({key: value for key, value in weights.items() if not key.endswith("weight")}, second)
    detected = inspect_comfy_quantization([first, second])
    assert detected is not None
    assert detected.nvfp4_layers == {"proj": (129, 80)}
    assert inspect_comfy_quantization([first]) is None
    with pytest.raises(ValueError, match="Duplicate checkpoint tensor"):
        inspect_comfy_quantization([first, second, first])


@pytest.mark.parametrize(
    "config",
    [
        {"format": "mxfp4"},
        {"format": "nvfp4", "full_precision_matrix_mult": True},
        {"format": "nvfp4", "rotation": "hadamard"},
    ],
)
def test_unsupported_recipe_rejected(config: dict, tmp_path: Path) -> None:
    weights, _ = _checkpoint(128, 64)
    weights["proj.comfy_quant"] = _metadata(config)
    with pytest.raises(ValueError):
        convert_comfy_nvfp4(weights)
    path = tmp_path / "unsupported.safetensors"
    save_file(weights, path)
    with pytest.raises(ValueError):
        inspect_comfy_quantization([path])


@pytest.mark.parametrize("key", ["proj.weight", "proj.weight_scale", "proj.weight_scale_2"])
def test_missing_tensor_rejected(key: str, tmp_path: Path) -> None:
    weights, _ = _checkpoint(128, 64)
    del weights[key]
    with pytest.raises(ValueError, match="missing"):
        convert_comfy_nvfp4(weights)
    path = tmp_path / "incomplete.safetensors"
    save_file(weights, path)
    with pytest.raises(ValueError):
        inspect_comfy_quantization([path])


@pytest.mark.parametrize("value", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_global_scale_rejected(value: float) -> None:
    weights, _ = _checkpoint(128, 64)
    weights["proj.weight_scale_2"] = torch.tensor(value)
    with pytest.raises(ValueError, match="finite and positive"):
        convert_comfy_nvfp4(weights)


@pytest.mark.parametrize("value", [-1.0, float("nan")])
def test_invalid_block_scale_rejected(value: float) -> None:
    weights, _ = _checkpoint(128, 64)
    weights["proj.weight_scale"] = torch.full((128, 4), value).to(torch.float8_e4m3fn)
    with pytest.raises(ValueError, match="finite and nonnegative"):
        convert_comfy_nvfp4(weights)


def test_uint8_without_metadata_is_not_detected(tmp_path: Path) -> None:
    weights = {"proj.weight": torch.zeros((128, 32), dtype=torch.uint8)}
    path = tmp_path / "nvfp4.safetensors"
    save_file(weights, path)
    assert inspect_comfy_quantization([path]) is None
    assert convert_comfy_nvfp4(weights)["proj.weight"] is weights["proj.weight"]


@pytest.mark.parametrize(
    "metadata",
    [
        torch.empty(0, dtype=torch.uint8),
        torch.zeros(4097, dtype=torch.uint8),
        torch.zeros((1, 19), dtype=torch.uint8),
        torch.ones(19, dtype=torch.float32),
        torch.tensor(list(b"not-json"), dtype=torch.uint8),
        torch.tensor(list(b"[]"), dtype=torch.uint8),
    ],
)
def test_malformed_metadata_rejected(metadata: torch.Tensor, tmp_path: Path) -> None:
    weights, _ = _checkpoint(128, 64)
    weights["proj.comfy_quant"] = metadata
    with pytest.raises(ValueError, match="comfy_quant"):
        convert_comfy_nvfp4(weights)
    path = tmp_path / "malformed.safetensors"
    save_file(weights, path)
    with pytest.raises(ValueError, match="comfy_quant"):
        inspect_comfy_quantization([path])


@pytest.mark.parametrize(
    "key,shape,dtype",
    [
        ("proj.weight", (128, 31), torch.uint8),
        ("proj.weight", (128, 32), torch.float16),
        ("proj.weight_scale", (127, 4), torch.float8_e4m3fn),
        ("proj.weight_scale", (128, 4), torch.float32),
        ("proj.weight_scale_2", (2,), torch.float32),
        ("proj.weight_scale_2", (), torch.float16),
    ],
)
def test_invalid_tensor_layout_rejected(
    key: str, shape: tuple[int, ...], dtype: torch.dtype, tmp_path: Path
) -> None:
    weights, _ = _checkpoint(128, 64)
    weights[key] = torch.ones(shape).to(dtype)
    with pytest.raises(ValueError):
        convert_comfy_nvfp4(weights)
    path = tmp_path / "bad-layout.safetensors"
    save_file(weights, path)
    with pytest.raises(ValueError):
        inspect_comfy_quantization([path])


@pytest.mark.parametrize("version", [1, 2])
@pytest.mark.parametrize(
    "modality,destinations",
    [
        ("img", ("to_q", "to_k", "to_v")),
        ("txt", ("add_q_proj", "add_k_proj", "add_v_proj")),
    ],
)
def test_flux_qkv_partition_preserves_quantization(
    version: int, modality: str, destinations: tuple[str, ...]
) -> None:
    weights, linear_scales = _checkpoint(192, 64)
    source = {
        name.replace("proj.", f"model.diffusion_model.double_blocks.0.{modality}_attn.qkv."): value
        for name, value in weights.items()
        if name.startswith("proj.")
    }
    result = convert_comfy_flux_weights(source, version=version, hidden_size=64)
    converted = convert_comfy_nvfp4(weights)
    for index, projection in enumerate(destinations):
        prefix = f"transformer_blocks.0.attn.{projection}"
        rows = slice(index * 64, (index + 1) * 64)
        torch.testing.assert_close(result[prefix + ".weight"], converted["proj.weight"][rows])
        torch.testing.assert_close(
            result[prefix + ".weight_scale"].view(torch.uint8), linear_scales[rows]
        )
        assert result[prefix + ".weight_scale_2"] is weights["proj.weight_scale_2"]


def test_flux1_single_stream_partition() -> None:
    weights, scales = _checkpoint(448, 64)
    source = {
        name.replace("proj.", "single_blocks.0.linear1."): value
        for name, value in weights.items()
        if name.startswith("proj.")
    }
    result = convert_comfy_flux_weights(source, version=1, hidden_size=64)
    expected = [
        ("attn.to_q", 0, 64),
        ("attn.to_k", 64, 128),
        ("attn.to_v", 128, 192),
        ("proj_mlp", 192, 448),
    ]
    for name, start, stop in expected:
        prefix = "single_transformer_blocks.0." + name
        assert result[prefix + ".weight"].shape == (stop - start, 32)
        torch.testing.assert_close(
            result[prefix + ".weight_scale"].view(torch.uint8), scales[start:stop]
        )
        assert result[prefix + ".weight_scale_2"] is weights["proj.weight_scale_2"]


@pytest.mark.parametrize("version", [1, 2])
def test_flux_modulation_reorders_scales_with_weight_rows(version: int) -> None:
    weights, scales = _checkpoint(128, 64)
    source = {
        name.replace("proj.", "final_layer.adaLN_modulation.1."): value
        for name, value in weights.items()
        if name.startswith("proj.")
    }
    result = convert_comfy_flux_weights(source, version=version, hidden_size=64)
    expected_weight = convert_comfy_nvfp4(weights)["proj.weight"]
    torch.testing.assert_close(
        result["norm_out.linear.weight"], torch.cat((expected_weight[64:], expected_weight[:64]))
    )
    torch.testing.assert_close(
        result["norm_out.linear.weight_scale"].view(torch.uint8),
        torch.cat((scales[64:], scales[:64])),
    )
    assert result["norm_out.linear.weight_scale_2"] is weights["proj.weight_scale_2"]


def test_flux2_single_stream_stays_fused() -> None:
    source = {"single_blocks.0.linear1.weight": torch.arange(288).reshape(36, 8)}
    result = convert_comfy_flux_weights(source, version=2, hidden_size=4, mlp_ratio=3)
    assert (
        result["single_transformer_blocks.0.attn.to_qkv_mlp_proj.weight"]
        is source["single_blocks.0.linear1.weight"]
    )


def test_flux_rejects_unknown_transforms() -> None:
    with pytest.raises(ValueError, match="Unsupported native FLUX"):
        convert_comfy_flux_weights(
            {"double_blocks.0.img_attn.qkv.rotation": torch.eye(4)}, version=1, hidden_size=4
        )


@pytest.mark.parametrize("version", [1, 2])
def test_pipeline_detects_comfy_recipe_before_construction(tmp_path: Path, version: int) -> None:
    model_name = "FluxTransformer2DModel" if version == 1 else "Flux2Transformer2DModel"
    component = tmp_path / "transformer"
    component.mkdir()
    (tmp_path / "model_index.json").write_text(
        json.dumps(
            {
                "_class_name": "FluxPipeline" if version == 1 else "Flux2Pipeline",
                "transformer": ["diffusers", model_name],
            }
        )
    )
    (component / "config.json").write_text(
        json.dumps(
            {
                "_class_name": model_name,
                "num_attention_heads": 1,
                "attention_head_dim": 64,
            }
        )
    )
    weights, _ = _checkpoint(192, 64)
    source = {
        name.replace("proj.", "double_blocks.0.img_attn.qkv."): value
        for name, value in weights.items()
        if name.startswith("proj.")
    }
    source["double_blocks.0.img_mlp.0.weight"] = torch.ones(128, 64, dtype=torch.bfloat16)
    source["single_blocks.0.linear2.weight"] = torch.ones(64, 256, dtype=torch.bfloat16)
    save_file(source, component / "model.safetensors")
    pipeline = DiffusionPipelineConfig.from_pretrained(str(tmp_path), device="cpu")
    config = pipeline.model_configs["transformer"]
    assert config.quant_config.quant_algo == QuantAlgo.NVFP4
    assert not config.dynamic_weight_quant
    assert config.force_dynamic_quantization
    assert not config.quant_config.is_module_excluded_from_quantization(
        "transformer_blocks.0.attn.qkv_proj"
    )
    mlp = "up_proj" if version == 1 else "gate_up_proj"
    assert config.quant_config.is_module_excluded_from_quantization(
        f"transformer_blocks.0.ff.{mlp}"
    )
    output = "proj_out" if version == 1 else "attn.to_out"
    assert config.quant_config.is_module_excluded_from_quantization(
        f"single_transformer_blocks.0.{output}.proj"
    )


def test_h3_gate_first_partition() -> None:
    weights, scales = _checkpoint(128, 64)
    source = {
        name.replace("proj.", "blocks.0.mlp.fc1."): value
        for name, value in weights.items()
        if name.startswith("proj.")
    }
    result = convert_comfy_h3_weights(source, attention_dim=64, ffn_dim=64)
    normalized = convert_comfy_nvfp4(weights)
    for index, name in enumerate(("gate", "up")):
        prefix = f"transformer_blocks.0.ff.{name}"
        rows = slice(index * 64, (index + 1) * 64)
        torch.testing.assert_close(result[prefix + ".weight"], normalized["proj.weight"][rows])
        torch.testing.assert_close(result[prefix + ".weight_scale"].view(torch.uint8), scales[rows])
        assert result[prefix + ".weight_scale_2"] is weights["proj.weight_scale_2"]


def test_h3_adaln_curve_interpolation() -> None:
    table = torch.tensor([[0.0, 2.0], [4.0, -2.0], [12.0, 6.0]])
    timesteps = torch.tensor([-1.0, 0.0, 0.25, 0.5, 0.75, 1.0, 2.0])
    expected = torch.tensor(
        [[0.0, 2.0], [0.0, 2.0], [2.0, 0.0], [4.0, -2.0], [8.0, 2.0], [12.0, 6.0], [12.0, 6.0]]
    )
    torch.testing.assert_close(interpolate_adaln_curve(table, timesteps), expected, rtol=0, atol=0)


def test_pipeline_detects_pruned_h3_recipe(tmp_path: Path) -> None:
    component = tmp_path / "transformer"
    component.mkdir()
    (tmp_path / "model_index.json").write_text(
        json.dumps(
            {
                "_class_name": "MiniMaxH3Pipeline",
                "transformer": ["diffusers", "MiniMaxH3Transformer3DModel"],
            }
        )
    )
    (component / "config.json").write_text(
        json.dumps(
            {
                "_class_name": "MiniMaxH3Transformer3DModel",
                "num_attention_heads": 1,
                "attention_head_dim": 64,
                "ffn_dim": 64,
                "time_embed_dim": 256,
            }
        )
    )
    weights, _ = _checkpoint(128, 64)
    source = {
        name.replace("proj.", "blocks.0.mlp.fc1."): value
        for name, value in weights.items()
        if name.startswith("proj.")
    }
    source["adaln_t_table"] = torch.ones(9, 8)
    source["blocks.0.adaln_proj.linear.weight"] = torch.ones(18 * 64, 8, dtype=torch.float16)
    source["token_refiner.blocks.0.mlp.fc1.weight"] = torch.ones(128, 64, dtype=torch.bfloat16)
    save_file(source, component / "model.safetensors")
    config = DiffusionPipelineConfig.from_pretrained(str(tmp_path), device="cpu").model_configs[
        "transformer"
    ]
    assert config.pretrained_config.time_embed_dim == 8
    assert config.extra_attrs["adaln_curve_grid"] == 9
    assert config.extra_attrs["comfy_nvfp4"]
    assert not config.dynamic_weight_quant
    assert config.force_dynamic_quantization
    assert not config.quant_config.is_module_excluded_from_quantization(
        "transformer_blocks.0.ff.gate_up_proj"
    )
    assert config.quant_config.is_module_excluded_from_quantization(
        "transformer_blocks.0.adaln_proj.linear"
    )
    assert config.quant_config.is_module_excluded_from_quantization(
        "token_refiner.refiner_blocks.0.ff.gate_up_proj"
    )


def test_prepare_bundle_selects_checkpoint_and_preserves_base(tmp_path: Path) -> None:
    base = tmp_path / "base"
    transformer = base / "transformer"
    transformer.mkdir(parents=True)
    (base / "vae").mkdir()
    (base / "vae" / "config.json").write_text("{}")
    (base / "model_index.json").write_text(
        json.dumps(
            {
                "_class_name": "FluxPipeline",
                "transformer": ["diffusers", "FluxTransformer2DModel"],
                "vae": ["diffusers", "AutoencoderKL"],
            }
        )
    )
    base_config = {
        "_class_name": "FluxTransformer2DModel",
        "num_attention_heads": 1,
        "attention_head_dim": 64,
        "quantization_config": {"quant_algo": "FP8"},
    }
    (transformer / "config.json").write_text(json.dumps(base_config))
    save_file({"original": torch.ones(1)}, transformer / "original.safetensors")
    weights, _ = _checkpoint(192, 64)
    source = {
        name.replace("proj.", "double_blocks.0.img_attn.qkv."): value
        for name, value in weights.items()
        if name.startswith("proj.")
    }
    selected = tmp_path / "selected.safetensors"
    save_file(source, selected)
    output = prepare_comfy_bundle(base, selected, tmp_path / "prepared")
    assert (output / "vae").resolve() == base / "vae"
    assert (output / "transformer" / "diffusion_pytorch_model.safetensors").resolve() == selected
    assert not (output / "transformer" / "original.safetensors").exists()
    assert json.loads((transformer / "config.json").read_text()) == base_config
    pipeline = DiffusionPipelineConfig.from_pretrained(str(output), device="cpu")
    assert pipeline.quant_config.quant_algo == QuantAlgo.NVFP4
    assert not pipeline.dynamic_weight_quant
    with pytest.raises(FileExistsError):
        prepare_comfy_bundle(base, selected, output)
    with pytest.raises(ValueError, match="outside"):
        prepare_comfy_bundle(base, selected, base / "recursive")


@pytest.mark.parametrize("dim,sizes", [(0, [32, 64, 32]), (1, [16, 48])])
def test_nvfp4_wrapper_splits_preserve_weights_and_scales(dim: int, sizes: list[int]) -> None:
    source, scales = _checkpoint(128, 64)
    normalized = convert_comfy_nvfp4(source)
    weights = {
        name.removeprefix("proj."): value
        for name, value in normalized.items()
        if name.startswith("proj.")
    }
    weights["bias"] = torch.arange(128, dtype=torch.bfloat16)
    weights["input_scale"] = torch.tensor(2.0)
    parts = _split_nvfp4_projection(weights, sizes, dim=dim)
    torch.testing.assert_close(
        torch.cat([part["weight"] for part in parts], dim=dim), weights["weight"]
    )
    torch.testing.assert_close(
        torch.cat([part["weight_scale"].view(torch.uint8) for part in parts], dim=dim), scales
    )
    for size, part in zip(sizes, parts):
        assert part["weight_scale_2"] is weights["weight_scale_2"]
        assert part["input_scale"] is weights["input_scale"]
        assert part["weight"].shape[dim] == size // (2 if dim == 1 else 1)
        assert part["weight_scale"].shape[dim] == size // (16 if dim == 1 else 1)
        if dim == 1:
            assert "bias" not in part
    if dim == 0:
        torch.testing.assert_close(torch.cat([part["bias"] for part in parts]), weights["bias"])


def test_nvfp4_wrapper_rejects_unaligned_column_split() -> None:
    source, _ = _checkpoint(128, 64)
    normalized = convert_comfy_nvfp4(source)
    weights = {
        name.removeprefix("proj."): value
        for name, value in normalized.items()
        if name.startswith("proj.")
    }
    with pytest.raises(ValueError, match="align to scale blocks"):
        _split_nvfp4_projection(weights, [8, 56], dim=1)


def test_legacy_header_detected_and_loaded(tmp_path: Path) -> None:
    source, scales = _checkpoint(128, 64)
    del source["proj.comfy_quant"]
    source["proj.input_scale"] = torch.tensor(0.25)
    path = tmp_path / "legacy.safetensors"
    save_file(
        source,
        path,
        metadata={
            "_quantization_metadata": json.dumps(
                {
                    "format_version": "1.0",
                    "layers": {"proj": {"format": "nvfp4"}},
                }
            )
        },
    )
    detected = inspect_comfy_quantization([path])
    assert detected is not None
    assert detected.nvfp4_layers == {"proj": (128, 64)}
    loaded = WeightLoader().load_weights(str(path), Mapping())
    assert "proj.comfy_quant" in loaded
    normalized = convert_comfy_nvfp4(loaded)
    torch.testing.assert_close(normalized["proj.weight_scale"].view(torch.uint8), scales)
    assert "proj.input_scale" not in normalized
    torch.testing.assert_close(normalized["proj.weight_scale_2"], source["proj.weight_scale_2"])


def test_legacy_header_equivalent_to_tensor_metadata_accepted(tmp_path: Path) -> None:
    source, _ = _checkpoint(128, 64)
    path = tmp_path / "conflicting.safetensors"
    save_file(
        source,
        path,
        metadata={
            "_quantization_metadata": json.dumps(
                {
                    "format_version": "1.0",
                    "layers": {"proj": {"format": "nvfp4", "full_precision_matrix_mult": False}},
                }
            )
        },
    )
    detected = inspect_comfy_quantization([path])
    assert detected is not None
    assert detected.nvfp4_layers == {"proj": (128, 64)}
    assert "proj.comfy_quant" in WeightLoader().load_weights(str(path), Mapping())


def test_prepare_h3_reference_component_routes_config_and_weights(tmp_path: Path) -> None:
    base = tmp_path / "base"
    config = {
        "_class_name": "MiniMaxH3Transformer3DModel",
        "num_attention_heads": 1,
        "attention_head_dim": 64,
        "ffn_dim": 64,
        "time_embed_dim": 256,
    }
    for name in ("transformer", "transformer_ref"):
        component = base / name
        component.mkdir(parents=True)
        (component / "config.json").write_text(json.dumps(config))
        save_file({"base_marker": torch.ones(1)}, component / "original.safetensors")
    (base / "model_index.json").write_text(
        json.dumps(
            {
                "_class_name": "MiniMaxH3Pipeline",
                "transformer": ["diffusers", "MiniMaxH3Transformer3DModel"],
                "transformer_ref": ["diffusers", "MiniMaxH3Transformer3DModel"],
            }
        )
    )
    weights, _ = _checkpoint(128, 64)
    source = {
        name.replace("proj.", "blocks.0.mlp.fc1."): value
        for name, value in weights.items()
        if name.startswith("proj.")
    }
    source["adaln_t_table"] = torch.ones(9, 8)
    selected = tmp_path / "ref2va.safetensors"
    save_file(source, selected)
    output = prepare_comfy_bundle(
        base, selected, tmp_path / "prepared", component="transformer_ref"
    )
    pipeline = DiffusionPipelineConfig.from_pretrained(
        str(output), device="cpu", pipeline_config={"workflow": "ref2va"}
    )
    assert pipeline.model_configs["transformer"].pretrained_config.time_embed_dim == 8
    assert pipeline.model_configs["transformer"].quant_config.quant_algo == QuantAlgo.NVFP4
    routed = MiniMaxH3Pipeline.load_transformer_weights(
        SimpleNamespace(workflow="ref2va", mapping=Mapping()), str(output)
    )
    assert "blocks.0.mlp.fc1.comfy_quant" in routed
    assert "base_marker" not in routed
    assert (output / "transformer" / "original.safetensors").exists()
    assert not (output / "transformer_ref" / "original.safetensors").exists()


@pytest.fixture
def native_h3_directory(tmp_path, monkeypatch):
    checkpoint_dir = tmp_path / "my-checkpoint"
    checkpoint_dir.mkdir()
    weights, _ = _checkpoint(128, 64)
    source = {
        name.replace("proj.", "blocks.0.mlp.fc1."): value
        for name, value in weights.items()
        if name.startswith("proj.")
    }
    source.update(
        {
            "video_patch_proj.weight": torch.ones(64, 96),
            "audio_patch_proj.weight": torch.ones(64, 32),
            "condition_proj.weight": torch.ones(64, 128),
            "adaln_t_table": torch.ones(9, 8),
        }
    )
    save_file(source, checkpoint_dir / "minimax_h3_ref2va.safetensors")
    base = tmp_path / "base-revision"
    base.mkdir()
    manifest = {"_class_name": "MiniMaxH3ModularPipeline"}
    config = {
        "_class_name": "MiniMaxH3Transformer3DModel",
        "num_attention_heads": 1,
        "attention_head_dim": 64,
        "hidden_size": 64,
        "ffn_dim": 64,
        "in_channels": 24,
        "audio_in_channels": 32,
        "patch_size": [1, 2, 2],
        "text_dim": 128,
        "time_embed_dim": 256,
    }
    for component in ("transformer", "transformer_ref"):
        folder = base / component
        folder.mkdir()
        (folder / "config.json").write_text(json.dumps(config))
        manifest[component] = ["diffusers", "MiniMaxH3Transformer3DModel"]
    (base / "modular_model_index.json").write_text(json.dumps(manifest))
    for component in h3_checkpoint._H3_COMPONENTS:
        (base / component).mkdir()
        (base / component / "marker").write_text(component)
    calls = []

    def download(model, allow_patterns, revision=None):
        calls.append((model, allow_patterns, revision))
        return base

    monkeypatch.setattr(h3_checkpoint, "download_hf_partial", download)
    monkeypatch.setattr(h3_checkpoint, "HF_HUB_CACHE", str(tmp_path / "cache" / "hub"))
    return checkpoint_dir, base, calls


def test_native_h3_directory_auto_resolution(native_h3_directory):
    directory, base, calls = native_h3_directory
    detected = h3_checkpoint.discover_native_h3(directory)
    assert detected.workflow == "ref2va"
    resolved = Path(h3_checkpoint.resolve_native_h3(detected))
    selected = resolved / "transformer_ref" / "diffusion_pytorch_model.safetensors"
    assert selected.resolve() == detected.path
    assert (resolved / "text_encoder" / "marker").read_text() == "text_encoder"
    assert calls[0][0] == "MiniMaxAI/MiniMax-H3"
    assert calls[1][2] == base.name
    assert not any(
        pattern in ("transformer/*", "transformer_ref/*")
        for _, patterns, _ in calls
        for pattern in patterns
    )
    assert h3_checkpoint.resolve_native_h3(detected) == str(resolved)
    config = DiffusionPipelineConfig.from_pretrained(
        str(resolved), device="cpu", pipeline_config={"workflow": detected.workflow}
    )
    assert config.model_configs["transformer"].quant_config.quant_algo == QuantAlgo.NVFP4
    assert config.model_configs["transformer"].pretrained_config.time_embed_dim == 8


def test_native_h3_normal_loader_routes_before_config(native_h3_directory, monkeypatch):
    directory, _, _ = native_h3_directory

    class ConfigReached(Exception):
        pass

    def inspect_config(path, args, pipeline_config):
        assert pipeline_config["workflow"] == "ref2va"
        assert Path(path, "transformer_ref", "diffusion_pytorch_model.safetensors").is_file()
        raise ConfigReached

    monkeypatch.setattr(DiffusionPipelineConfig, "from_pretrained", inspect_config)
    loader = PipelineLoader(VisualGenArgs(model=str(directory)), device="cpu")
    with pytest.raises(ConfigReached):
        loader.load(skip_warmup=True)


def test_native_h3_rejects_ambiguous_variants(native_h3_directory):
    directory, _, _ = native_h3_directory
    source = directory / "minimax_h3_ref2va.safetensors"
    (directory / "another_ref2va.safetensors").symlink_to(source)
    with pytest.raises(ValueError, match="Multiple H3 checkpoints"):
        h3_checkpoint.discover_native_h3(directory)


def test_native_h3_workflow_selection(native_h3_directory):
    directory, _, _ = native_h3_directory
    source = directory / "minimax_h3_ref2va.safetensors"
    (directory / "model_fl2va.safetensors").symlink_to(source)
    result = h3_checkpoint.discover_native_h3(directory, "fl2va")
    assert result.workflow == "fl2va"
    assert result.path == source.resolve()


def test_native_h3_requires_workflow_evidence(native_h3_directory):
    directory, _, _ = native_h3_directory
    (directory / "minimax_h3_ref2va.safetensors").rename(directory / "model.safetensors")
    with pytest.raises(ValueError, match="cannot be distinguished"):
        h3_checkpoint.discover_native_h3(directory)
    assert h3_checkpoint.discover_native_h3(directory, "ref2va").workflow == "ref2va"


def test_native_h3_workflow_conflict(native_h3_directory):
    directory, _, _ = native_h3_directory
    with pytest.raises(ValueError, match="do not match requested workflow"):
        h3_checkpoint.discover_native_h3(directory, "fl2va")


def test_native_h3_dimensions_checked_before_component_download(native_h3_directory):
    directory, base, calls = native_h3_directory
    config_path = base / "transformer_ref" / "config.json"
    config = json.loads(config_path.read_text())
    config["hidden_size"] = 123
    config_path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="dimensions do not match"):
        h3_checkpoint.resolve_native_h3(h3_checkpoint.discover_native_h3(directory))
    assert len(calls) == 1


def test_native_discovery_leaves_pipeline_and_unrelated_weights_alone(tmp_path):
    save_file({"not_h3.weight": torch.ones(2, 2)}, tmp_path / "h3_ref2va.safetensors")
    assert h3_checkpoint.discover_native_h3(tmp_path) is None
    (tmp_path / "model_index.json").write_text(json.dumps({"_class_name": "Model"}))
    assert h3_checkpoint.discover_native_h3(tmp_path) is None


def test_native_h3_invalid_workflow_does_not_download(native_h3_directory):
    directory, _, calls = native_h3_directory
    with pytest.raises(ValueError, match="workflow must be"):
        h3_checkpoint.discover_native_h3(directory, "ref2v")
    assert not calls


def test_native_h3_cache_changes_with_source(native_h3_directory):
    directory, _, _ = native_h3_directory
    detected = h3_checkpoint.discover_native_h3(directory)
    first = h3_checkpoint.resolve_native_h3(detected)
    stat = detected.path.stat()
    os.utime(detected.path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1000000000))
    second = h3_checkpoint.resolve_native_h3(detected)
    assert first != second


def test_native_h3_rejects_untagged_encoding(native_h3_directory):
    directory, _, calls = native_h3_directory
    path = directory / "minimax_h3_ref2va.safetensors"
    save_file(
        {
            "video_patch_proj.weight": torch.ones(64, 96),
            "audio_patch_proj.weight": torch.ones(64, 32),
            "condition_proj.weight": torch.ones(64, 128),
        },
        path,
    )
    with pytest.raises(ValueError, match="encoding is unsupported"):
        h3_checkpoint.discover_native_h3(directory)
    assert not calls


def test_legacy_fp8_metadata_remains_with_existing_loader(tmp_path):
    path = tmp_path / "fp8.safetensors"
    weight = torch.ones(32, 32).to(torch.float8_e4m3fn)
    save_file(
        {"proj.weight": weight},
        path,
        metadata={
            "config": json.dumps({"transformer": {"_class_name": "LTX2VideoTransformer3DModel"}}),
            "_quantization_metadata": json.dumps(
                {"format_version": "1.0", "layers": {"proj": {"format": "float8_e4m3fn"}}}
            ),
        },
    )
    assert inspect_comfy_quantization([path]) is None
    config = DiffusionPipelineConfig.from_pretrained(str(path), device="cpu")
    assert config.model_configs["transformer"].quant_config.quant_algo == QuantAlgo.FP8
    weights = WeightLoader().load_weights(str(path), Mapping())
    assert set(weights) == {"proj.weight"}
    torch.testing.assert_close(weights["proj.weight"].float(), weight.float())


def test_native_h3_quantized_signature_uses_logical_shape(native_h3_directory):
    directory, _, _ = native_h3_directory
    path = directory / "minimax_h3_ref2va.safetensors"
    weights = load_file(path)
    quantized, _ = _checkpoint(64, 128)
    weights.update(
        {
            name.replace("proj.", "condition_proj."): value
            for name, value in quantized.items()
            if name.startswith("proj.")
        }
    )
    save_file(weights, path)
    detected = h3_checkpoint.discover_native_h3(directory)
    assert detected.shapes["condition_proj.weight"] == (64, 128)
    assert Path(h3_checkpoint.resolve_native_h3(detected)).is_dir()


def test_registered_checkpoint_adapter_is_model_independent(tmp_path, monkeypatch):
    resolved_path = str(tmp_path / "resolved")
    resolutions = []

    class Adapter:
        def resolve(self):
            resolutions.append(True)
            return ResolvedCheckpoint(resolved_path, {"mode": "detected"})

    def probe(path, options):
        assert path == tmp_path
        assert options == {"mode": "requested"}
        with pytest.raises(TypeError):
            options["mode"] = "mutated"
        return Adapter()

    monkeypatch.setattr(
        "tensorrt_llm._torch.visual_gen.pipeline_registry.PIPELINE_REGISTRY",
        {
            "UnrelatedPipeline": SimpleNamespace(
                checkpoint_probe=probe, defaults={"mode": "default"}
            )
        },
    )
    result = AutoPipeline.resolve(str(tmp_path), {"mode": "requested"})
    assert result.path == resolved_path
    assert result.pipeline_config == {"mode": "requested"}
    assert result.class_name == "UnrelatedPipeline"
    assert resolutions == [True]


def test_adapter_ambiguity_rejected_before_resolution(tmp_path, monkeypatch):
    class Adapter:
        def resolve(self):
            pytest.fail("Ambiguous recognition must not download or mutate anything")

    entry = SimpleNamespace(checkpoint_probe=lambda path, options: Adapter())
    monkeypatch.setattr(
        "tensorrt_llm._torch.visual_gen.pipeline_registry.PIPELINE_REGISTRY",
        {"First": entry, "Second": entry},
    )
    with pytest.raises(ValueError, match="Multiple pipeline adapters"):
        AutoPipeline.resolve(str(tmp_path), {})


def test_standard_pipeline_bypasses_native_adapters(tmp_path, monkeypatch):
    (tmp_path / "model_index.json").write_text(json.dumps({"_class_name": "Model"}))
    entry = SimpleNamespace(
        checkpoint_probe=lambda *args: pytest.fail("Already resolved"),
        defaults={"workflow": "default"},
    )
    monkeypatch.setattr(
        "tensorrt_llm._torch.visual_gen.pipeline_registry.PIPELINE_REGISTRY", {"Model": entry}
    )
    assert AutoPipeline.resolve(str(tmp_path), {"workflow": "untouched"}).path == str(tmp_path)


def test_unrecognized_checkpoint_fails_during_resolution(tmp_path, monkeypatch):
    entry = SimpleNamespace(checkpoint_probe=lambda *args: None)
    monkeypatch.setattr(
        "tensorrt_llm._torch.visual_gen.pipeline_registry.PIPELINE_REGISTRY", {"Model": entry}
    )
    with pytest.raises(ValueError, match="Cannot detect pipeline type"):
        AutoPipeline.resolve(str(tmp_path), {})


def test_h3_adapter_registered_by_model():
    assert (
        PIPELINE_REGISTRY["MiniMaxH3ModularPipeline"].checkpoint_probe
        is h3_checkpoint.probe_native_h3
    )


def test_adapter_options_use_normal_validation_and_user_precedence(tmp_path, monkeypatch):
    (tmp_path / "model_index.json").write_text(
        json.dumps({"_class_name": "MiniMaxH3ModularPipeline"})
    )

    class Adapter:
        def resolve(self):
            return ResolvedCheckpoint(str(tmp_path), {"workflow": "ref2va"})

    entry = PIPELINE_REGISTRY["MiniMaxH3ModularPipeline"]
    monkeypatch.setattr(entry, "checkpoint_probe", lambda path, options: Adapter())
    (tmp_path / "model_index.json").unlink()
    result = AutoPipeline.resolve(str(tmp_path), {"workflow": "fl2va"})
    assert result.pipeline_config["workflow"] == "fl2va"
    with pytest.raises(ValueError, match="Unknown pipeline_config keys"):
        AutoPipeline.resolve(str(tmp_path), {"unknown_adapter_option": True})


@pytest.fixture(params=[1, 2])
def native_flux_directory(request, tmp_path, monkeypatch):
    version = request.param
    directory = tmp_path / "unlabelled"
    directory.mkdir()
    packed, _ = _checkpoint(192, 64)
    source = {
        name.replace("proj.", "double_blocks.0.img_attn.qkv."): value
        for name, value in packed.items()
        if name.startswith("proj.")
    }
    source.update(
        {
            "img_in.weight": torch.ones(64, 64),
            "txt_in.weight": torch.ones(64, 128),
            "final_layer.linear.weight": torch.ones(64, 64),
            "guidance_in.in_layer.weight": torch.ones(64, 256),
            "single_blocks.0.linear2.weight": torch.ones(64, 320),
        }
    )
    if version == 1:
        source["vector_in.in_layer.weight"] = torch.ones(64, 32)
    else:
        source.update(
            {
                "double_stream_modulation_img.lin.weight": torch.ones(384, 64),
                "double_stream_modulation_txt.lin.weight": torch.ones(384, 64),
                "single_stream_modulation.lin.weight": torch.ones(192, 64),
            }
        )
    # Packed signature projection verifies that matching uses logical dimensions.
    packed_img, _ = _checkpoint(64, 64)
    source.update(
        {
            name.replace("proj.", "img_in."): value
            for name, value in packed_img.items()
            if name.startswith("proj.")
        }
    )
    save_file(
        {"model.diffusion_model." + key: value for key, value in source.items()},
        directory / "weights.safetensors",
    )
    base = tmp_path / "base-revision"
    (base / "transformer").mkdir(parents=True)
    architecture = "FluxTransformer2DModel" if version == 1 else "Flux2Transformer2DModel"
    config = {
        "_class_name": architecture,
        "num_attention_heads": 1,
        "attention_head_dim": 64,
        "in_channels": 64,
        "out_channels": 64,
        "joint_attention_dim": 128,
        "pooled_projection_dim": 32,
        "num_layers": 1,
        "num_single_layers": 1,
        "guidance_embeds": True,
        "patch_size": 1,
    }
    (base / "transformer" / "config.json").write_text(json.dumps(config))
    (base / "model_index.json").write_text(
        json.dumps(
            {
                "_class_name": "FluxPipeline" if version == 1 else "Flux2Pipeline",
                "transformer": ["diffusers", architecture],
            }
        )
    )
    (base / "text_encoder").mkdir()
    (base / "text_encoder" / "marker").write_text("supporting component")
    calls = []

    def download(model, allow_patterns, revision=None):
        calls.append((model, allow_patterns, revision))
        return base

    monkeypatch.setattr(flux_checkpoint, "download_hf_partial", download)
    monkeypatch.setattr(flux_checkpoint, "HF_HUB_CACHE", str(tmp_path / "cache" / "hub"))
    return directory, base, calls, version


def test_native_flux_directory_resolution(native_flux_directory):
    directory, base, calls, version = native_flux_directory
    resolved = AutoPipeline.resolve(str(directory), {})
    selected = Path(resolved.path) / "transformer" / "diffusion_pytorch_model.safetensors"
    assert selected.resolve() == directory / "weights.safetensors"
    assert (Path(resolved.path) / "text_encoder" / "marker").read_text() == "supporting component"
    assert resolved.pipeline_config == {}
    assert calls[0][0] == f"black-forest-labs/FLUX.{version}-dev"
    assert calls[1][2] == base.name
    assert "transformer/*" not in calls[1][1]
    assert ("text_encoder_2/*" in calls[1][1]) == (version == 1)
    assert AutoPipeline.resolve(str(directory), {}).path == resolved.path
    config = DiffusionPipelineConfig.from_pretrained(resolved.path, device="cpu")
    assert config.model_configs["transformer"].quant_config.quant_algo == QuantAlgo.NVFP4


def test_native_flux_normal_loader_routes_before_config(native_flux_directory, monkeypatch):
    directory, _, _, version = native_flux_directory

    class ConfigReached(Exception):
        pass

    def inspect_config(path, args, pipeline_config):
        manifest = json.loads(Path(path, "model_index.json").read_text())
        assert manifest["_class_name"] == ("FluxPipeline" if version == 1 else "Flux2Pipeline")
        assert Path(path, "transformer", "diffusion_pytorch_model.safetensors").is_file()
        raise ConfigReached

    monkeypatch.setattr(DiffusionPipelineConfig, "from_pretrained", inspect_config)
    with pytest.raises(ConfigReached):
        PipelineLoader(VisualGenArgs(model=str(directory)), device="cpu").load(skip_warmup=True)


@pytest.mark.parametrize("field,value", [("in_channels", 32), ("num_layers", 2)])
def test_native_flux_rejects_wrong_base_before_component_download(
    native_flux_directory, field, value
):
    directory, base, calls, _ = native_flux_directory
    path = base / "transformer" / "config.json"
    config = json.loads(path.read_text())
    config[field] = value
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="do not match"):
        AutoPipeline.resolve(str(directory), {})
    assert len(calls) == 1


def test_native_flux_rejects_ambiguous_directory(native_flux_directory):
    directory, _, calls, _ = native_flux_directory
    (directory / "second.safetensors").symlink_to(directory / "weights.safetensors")
    with pytest.raises(ValueError, match="Multiple FLUX checkpoints"):
        AutoPipeline.resolve(str(directory), {})
    assert not calls


def test_native_flux_rejects_unsupported_variant(native_flux_directory):
    directory, _, calls, _ = native_flux_directory
    path = directory / "weights.safetensors"
    weights = load_file(path)
    del weights["model.diffusion_model.guidance_in.in_layer.weight"]
    save_file(weights, path)
    with pytest.raises(ValueError, match="not a supported dev variant"):
        AutoPipeline.resolve(str(directory), {})
    assert not calls


def test_native_flux_nested_directory(native_flux_directory):
    directory, _, _, _ = native_flux_directory
    nested = directory / "diffusion_models"
    nested.mkdir()
    (directory / "weights.safetensors").rename(nested / "weights.safetensors")
    resolved = AutoPipeline.resolve(str(directory), {})
    assert (
        Path(resolved.path, "transformer", "diffusion_pytorch_model.safetensors").resolve()
        == nested / "weights.safetensors"
    )


@pytest.mark.parametrize("native", [False, True])
def test_selected_pipeline_is_reused_for_construction(tmp_path, monkeypatch, native):
    class Pipeline:
        @classmethod
        def resolve_variant(cls, config):
            return cls

        def __init__(self, config):
            self.config = config

    entry = SimpleNamespace(
        pipeline_cls=Pipeline, checkpoint_probe=None, defaults={"mode": "default"}
    )
    monkeypatch.setattr(
        "tensorrt_llm._torch.visual_gen.pipeline_registry.PIPELINE_REGISTRY", {"Model": entry}
    )
    if native:
        entry.checkpoint_probe = lambda path, options: SimpleNamespace(
            resolve=lambda: ResolvedCheckpoint(str(path), {"mode": "inferred"})
        )
        monkeypatch.setattr(
            AutoPipeline, "_detect_from_checkpoint", lambda path: pytest.fail("Already recognized")
        )
    else:
        (tmp_path / "model_index.json").write_text(json.dumps({"_class_name": "Model"}))
    selected = AutoPipeline.resolve(str(tmp_path), {"mode": "explicit"})
    assert selected.entry is entry
    assert selected.pipeline_config == {"mode": "explicit"}
    monkeypatch.setattr(
        AutoPipeline, "_detect_from_checkpoint", lambda path: pytest.fail("Already selected")
    )
    config = SimpleNamespace(model_configs={}, vae_conv_quant_config=None)
    assert AutoPipeline.from_config(config, selected).config is config


def test_from_config_does_not_adapt_an_already_configured_model(native_h3_directory):
    directory, _, calls = native_h3_directory
    with pytest.raises(ValueError, match="Cannot detect pipeline type"):
        AutoPipeline.from_config(SimpleNamespace(), str(directory))
    assert not calls


def test_ltx_embedded_checkpoint_resolution_preserves_options(tmp_path):
    save_file(
        {"marker": torch.ones(1)},
        tmp_path / "weights.safetensors",
        metadata={"config": json.dumps({"transformer": {}, "vae": {}})},
    )
    options = {
        "spatial_upsampler_path": "/models/upsampler",
        "distilled_lora_path": "/models/lora",
    }
    selected = AutoPipeline.resolve(str(tmp_path), options)
    assert selected.class_name == "LTX2Pipeline"
    assert selected.path == str(tmp_path)
    assert selected.pipeline_config == {"text_encoder_path": "google/gemma-3-12b-it", **options}
    options["text_encoder_path"] = "/models/text"
    assert AutoPipeline.resolve(str(tmp_path), options).pipeline_config == options
