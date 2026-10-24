# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from tensorrt_llm._torch.visual_gen.checkpoints.comfy_bundle import prepare_comfy_bundle
from tensorrt_llm._torch.visual_gen.checkpoints.comfy_flux import convert_comfy_flux_weights
from tensorrt_llm._torch.visual_gen.checkpoints.comfy_h3 import convert_comfy_h3_weights
from tensorrt_llm._torch.visual_gen.checkpoints.comfy_quant import (
    convert_comfy_nvfp4,
    inspect_comfy_quantization,
)
from tensorrt_llm._torch.visual_gen.checkpoints.weight_loader import WeightLoader
from tensorrt_llm._torch.visual_gen.config import DiffusionPipelineConfig
from tensorrt_llm._torch.visual_gen.models.flux.joint_proj import _split_nvfp4_projection
from tensorrt_llm._torch.visual_gen.models.minimax_h3.adaln_curves import interpolate_adaln_curve
from tensorrt_llm._torch.visual_gen.models.minimax_h3.pipeline_minimax_h3 import MiniMaxH3Pipeline
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.quantization.mode import QuantAlgo

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
