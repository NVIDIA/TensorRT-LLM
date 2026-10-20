# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FLUX.3 package recognition and checkpoint validation without model assets."""

import json
import shutil
from pathlib import Path

import pytest
import torch
from flux3_test_utils import small_weights
from safetensors.torch import save_file

from tensorrt_llm._torch.visual_gen.config import DiffusionPipelineConfig
from tensorrt_llm._torch.visual_gen.models.flux3.checkpoint import Flux3Checkpoint
from tensorrt_llm._torch.visual_gen.pipeline_registry import AutoPipeline
from tensorrt_llm.visual_gen import VisualGenArgs

pytestmark = pytest.mark.cpu_only


@pytest.mark.parametrize("asset", ["text_encoder", "ae.safetensors"])
def test_hf_asset_downloads_only_referenced_content(package, asset, monkeypatch):
    cached = package / "cache"
    cached.mkdir()
    target = cached / asset
    if asset == "text_encoder":
        target.mkdir()
    else:
        target.write_bytes(b"vae")

    def download(model, *, allow_patterns):
        assert model == "test-owner/shared-model"
        assert allow_patterns == [asset, f"{asset}/**"]
        return cached

    monkeypatch.setattr(
        "tensorrt_llm._torch.visual_gen.models.flux3.checkpoint.download_hf_partial", download
    )
    checkpoint = Flux3Checkpoint(package)
    assert checkpoint.asset(f"hf://test-owner/shared-model/{asset}") == target
    target.rmdir() if target.is_dir() else target.unlink()
    with pytest.raises(FileNotFoundError, match="Missing FLUX.3 asset"):
        checkpoint.asset(f"hf://test-owner/shared-model/{asset}")


@pytest.mark.parametrize(
    "name", ["hf://owner/repo", "hf://owner/repo/../file", "hf://owner/repo/*"]
)
def test_invalid_hf_asset(package, name):
    with pytest.raises(ValueError, match="Invalid FLUX.3 HF asset"):
        Flux3Checkpoint(package).asset(name)


@pytest.mark.parametrize("variant", [None, "fp8r", "nvfp4"])
@pytest.mark.parametrize("flow_map", [False, True])
def test_split_klein_package(package: Path, variant: str | None, flow_map: bool) -> None:
    if flow_map:
        for precision in (None, "fp8r", "nvfp4"):
            source = package / "variants" / (f"sd-{precision}" if precision else "sd")
            target = package / "variants" / precision if precision else package
            for filename in ("config.json", "model.safetensors"):
                shutil.copyfile(source / filename, target / filename)
    class_name = "Flux3DistilledModularPipeline" if flow_map else "Flux3ModularPipeline"
    (package / "modular_model_index.json").write_text(json.dumps({"_class_name": class_name}))
    assert AutoPipeline._detect_from_checkpoint(str(package)) == "Flux3Pipeline"
    args = VisualGenArgs(model=str(package), pipeline_config={"variant": variant})
    config = DiffusionPipelineConfig.from_pretrained(str(package), args=args, device="cpu")
    assert config.primary_pretrained_config.flow_map is flow_map
    assert (
        config.quant_config.quant_algo
        == {
            None: None,
            "fp8r": "FP8_PER_CHANNEL_PER_TOKEN",
            "nvfp4": "NVFP4",
        }[variant]
    )
    assert config.extra_attrs["flux3_checkpoint"].directory == (
        package / "variants" / variant if variant else package
    )


@pytest.fixture
def package(tmp_path):
    config = {
        "format_version": 1,
        "model": "flux-3-image-klein",
        "torch_dtype": "bfloat16",
        "dit": {
            "hidden": 256,
            "heads": 2,
            "depth": 1,
            "depth_single": 2,
            "mlp_ratio": 3.0,
            "axes_dim": [32, 32, 32, 32],
            "theta": 10000,
            "in_channels": {"image": 128, "image_cond": 128, "text": 48},
            "out_modes": ["image"],
            "vec_in_dim": None,
            "image_rope_scale": 0.5,
        },
        "text_encoder": {
            "path": "text_encoder",
            "tokenizer": "tokenizer",
            "layers": [4],
            "pad_multiple": 80,
            "max_tokens": 8192,
            "dtype": "bfloat16",
        },
        "vae": {"path": "ae.safetensors", "dtype": "float32"},
        "sampler": {"steps": 4, "guidance": 3.5, "alpha": 6.93, "size": [64, 64]},
    }
    for variant in (None, "fp8r", "nvfp4", "sd", "sd-fp8r", "sd-nvfp4"):
        directory = tmp_path if variant is None else tmp_path / "variants" / variant
        directory.mkdir(parents=True, exist_ok=True)
        flow_map = variant is not None and variant.startswith("sd")
        quantization = variant.removeprefix("sd-") if variant else None
        if quantization == "sd":
            quantization = None
        config["quantization"] = quantization
        config["dit"]["flow_map"] = flow_map
        config["sampler"]["guidance"] = None if flow_map else 3.5
        (directory / "config.json").write_text(json.dumps(config))
        weights = small_weights(flow_map=flow_map)
        if quantization:
            for name, weight in list(weights.items()):
                if name.endswith(".weight") and name.startswith(
                    ("content_mode_blocks.", "single_blocks.")
                ):
                    prefix = name.removesuffix(".weight")
                    if quantization == "fp8r":
                        weights[name] = weight.to(torch.float8_e4m3fn)
                        weights[prefix + ".weight_scale"] = torch.ones(weight.shape[0])
                    else:
                        n, k = weight.shape
                        weights[name] = torch.zeros(n, k // 2, dtype=torch.uint8)
                        weights[prefix + ".weight_scale"] = torch.ones(n, k // 16).to(
                            torch.float8_e4m3fn
                        )
                        weights[prefix + ".weight_scale_2"] = torch.tensor(1.0)
                        weights[prefix + ".input_scale"] = torch.tensor(1.0)
        save_file(weights, directory / "model.safetensors")
    return tmp_path


@pytest.mark.parametrize(
    "variant,algo",
    [
        (None, None),
        ("fp8r", "FP8_PER_CHANNEL_PER_TOKEN"),
        ("nvfp4", "NVFP4"),
        ("sd", None),
        ("sd-fp8r", "FP8_PER_CHANNEL_PER_TOKEN"),
        ("sd-nvfp4", "NVFP4"),
    ],
)
def test_variant_controls_checkpoint_configuration(package, variant, algo):
    assert AutoPipeline._detect_from_checkpoint(str(package)) == "Flux3Pipeline"
    args = VisualGenArgs(model=str(package), pipeline_config={"variant": variant})
    config = DiffusionPipelineConfig.from_pretrained(str(package), args=args, device="cpu")
    assert config.quant_config.quant_algo == algo
    assert not config.dynamic_weight_quant
    if algo:
        excluded = Flux3Checkpoint(package, variant).transformer_config()["quantization_config"][
            "ignore"
        ]
        assert "txt_in" in excluded
        if variant.startswith("sd"):
            assert "ref_time_in.in_layer" in excluded
            assert "ref_time_in.out_layer" in excluded
        assert "single_blocks.0.q_proj" not in excluded
    assert config.primary_pretrained_config.out_modes == ["image"]
    assert config.primary_pretrained_config.flow_map == bool(variant and variant.startswith("sd"))
    selected = config.extra_attrs["flux3_checkpoint"]
    assert selected.root == package
    assert selected.directory == (package / "variants" / variant if variant else package)


@pytest.mark.parametrize("variant", ["../nvfp4", "a/b", 1])
def test_invalid_variant(package, variant):
    with pytest.raises(ValueError):
        Flux3Checkpoint(package, variant)


def test_missing_variant(package):
    with pytest.raises(FileNotFoundError):
        Flux3Checkpoint(package, "missing")


def test_quantization_override_is_rejected(package):
    args = VisualGenArgs(model=str(package), quant_config={"quant_algo": "FP8"})
    with pytest.raises(ValueError, match="variant"):
        DiffusionPipelineConfig.from_pretrained(str(package), args=args, device="cpu")


def test_invalid_tensor_shape_is_rejected(package):
    save_file({"txt_in.weight": torch.zeros(1)}, package / "model.safetensors")
    with pytest.raises(ValueError, match="tensor mismatch"):
        Flux3Checkpoint(package).transformer_config()


def test_quantization_metadata_must_match_tensors(package):
    config_path = package / "variants" / "fp8r" / "config.json"
    config = json.loads(config_path.read_text())
    config["quantization"] = "nvfp4"
    config_path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="tensor mismatch"):
        Flux3Checkpoint(package, "fp8r").transformer_config()


def test_manifest_cannot_override_variant_configuration(package):
    directory = package / "variants" / "nvfp4"
    (directory / "manifest.json").write_text(json.dumps({"quantization": "fp8r"}))
    config = Flux3Checkpoint(package, "nvfp4").transformer_config()
    assert config["quantization_config"]["quant_algo"] == "NVFP4"


def test_hub_cache_asset_symlinks(package, tmp_path):
    blob = tmp_path.parent / (tmp_path.name + "-blob")
    blob.write_bytes(b"cached asset")
    (package / "ae.safetensors").symlink_to(blob)
    assert Flux3Checkpoint(package).asset("ae.safetensors").read_bytes() == b"cached asset"
    with pytest.raises(ValueError, match="escapes"):
        Flux3Checkpoint(package).asset("../outside")


@pytest.mark.parametrize(
    "section,field",
    [
        ("dit", "vec_in_dim"),
        ("dit", "image_rope_scale"),
        ("text_encoder", "dtype"),
        ("vae", "dtype"),
        (None, "quantization"),
    ],
)
def test_package_fields_are_required(package, section, field):
    path = package / "config.json"
    raw = json.loads(path.read_text())
    del (raw[section] if section else raw)[field]
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match=field):
        Flux3Checkpoint(package)


@pytest.mark.parametrize(
    "field,value", [("heads", 0), ("hidden", 255), ("axes_dim", [2]), ("image_rope_scale", 0)]
)
def test_invalid_dit_config_fails_before_loading(package, field, value):
    path = package / "config.json"
    raw = json.loads(path.read_text())
    raw["dit"][field] = value
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError):
        Flux3Checkpoint(package)


@pytest.mark.parametrize("filename", ["model_index.json", "modular_model_index.json"])
@pytest.mark.parametrize("class_name", ["Flux3Pipeline", "Flux3ImagePipeline"])
def test_legacy_flux3_layout_has_actionable_error(tmp_path, filename, class_name):
    (tmp_path / filename).write_text(json.dumps({"_class_name": class_name}))
    with pytest.raises(ValueError, match="pipeline_config.variant"):
        AutoPipeline._detect_from_checkpoint(str(tmp_path))


@pytest.mark.parametrize("class_name", ["FluxPipeline", "Flux2Pipeline", "WanPipeline"])
def test_other_pipeline_layouts_still_detect(tmp_path, class_name):
    (tmp_path / "model_index.json").write_text(json.dumps({"_class_name": class_name}))
    assert AutoPipeline._detect_from_checkpoint(str(tmp_path)) == class_name


@pytest.mark.parametrize("manifest", ["model_index.json", "modular_model_index.json"])
@pytest.mark.parametrize(
    "root_config", [b"[]", b"null", b"{}", b'{"model":"other"}', b"{", b"\xff"]
)
def test_unrelated_root_config_preserves_diffusers_loading(tmp_path, manifest, root_config):
    """Both entry points ignore unrelated root config data and load the manifest."""
    (tmp_path / "config.json").write_bytes(root_config)
    (tmp_path / manifest).write_text(
        json.dumps(
            {"_class_name": "WanPipeline", "transformer": ["diffusers", "WanTransformer3DModel"]}
        )
    )
    transformer_dir = tmp_path / "transformer"
    transformer_dir.mkdir()
    (transformer_dir / "config.json").write_text(json.dumps({"hidden_size": 128}))

    assert AutoPipeline._detect_from_checkpoint(str(tmp_path)) == "WanPipeline"
    config = DiffusionPipelineConfig.from_pretrained(str(tmp_path), device="cpu")
    assert config.primary_pretrained_config.hidden_size == 128
    assert "flux3_checkpoint" not in config.extra_attrs


@pytest.mark.parametrize("root_config", [b"[]", b"{"])
def test_invalid_root_config_without_manifest_has_loader_error(tmp_path, root_config):
    """Unrecognized packages reach the normal unsupported-checkpoint errors."""
    (tmp_path / "config.json").write_bytes(root_config)
    with pytest.raises(ValueError):
        AutoPipeline._detect_from_checkpoint(str(tmp_path))
    with pytest.raises(ValueError, match="Config not found"):
        DiffusionPipelineConfig.from_pretrained(str(tmp_path), device="cpu")


@pytest.mark.parametrize("config", [{}, [], {"model": "flux-3-image-klein", "format_version": 2}])
def test_unsupported_package_has_actionable_error(tmp_path, config):
    (tmp_path / "config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match="format_version=1 package root"):
        Flux3Checkpoint(tmp_path)


def test_missing_package_has_actionable_error(tmp_path):
    with pytest.raises(FileNotFoundError, match=r"Missing FLUX.3 config: .*config\.json"):
        Flux3Checkpoint(tmp_path)


@pytest.mark.parametrize("flow_map,guidance", [(1, None), (True, 3.5), (False, None)])
def test_flow_map_config_contract(package, flow_map, guidance) -> None:
    path = package / "config.json"
    config = json.loads(path.read_text())
    config["dit"]["flow_map"] = flow_map
    config["sampler"]["guidance"] = guidance
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="flow_map|guidance"):
        Flux3Checkpoint(package)


def test_flow_map_requires_target_time_weights(package) -> None:
    path = package / "variants" / "sd" / "model.safetensors"
    save_file(small_weights(), path)
    with pytest.raises(ValueError, match="ref_time_in"):
        Flux3Checkpoint(package, "sd").transformer_config()


def test_missing_flow_map_flag_preserves_ordinary_checkpoint(package) -> None:
    path = package / "config.json"
    config = json.loads(path.read_text())
    del config["dit"]["flow_map"]
    path.write_text(json.dumps(config))
    assert Flux3Checkpoint(package).transformer_config()["flow_map"] is False


def test_flow_map_rejects_cfg_parallel_before_model_allocation() -> None:
    from types import SimpleNamespace

    from tensorrt_llm._torch.visual_gen.models.flux3.pipeline_flux3 import Flux3Pipeline

    pipeline = SimpleNamespace(
        pipeline_config=SimpleNamespace(
            model_configs={
                "transformer": SimpleNamespace(pretrained_config=SimpleNamespace(flow_map=True))
            },
            visual_gen_mapping=SimpleNamespace(cfg_size=2),
        )
    )
    with pytest.raises(ValueError, match="CFG parallelism"):
        Flux3Pipeline._init_transformer(pipeline)


@pytest.mark.parametrize("variant", ["fp8r", "nvfp4", "sd-fp8r", "sd-nvfp4"])
@pytest.mark.parametrize("tp_size", [1, 2])
def test_quantized_runtime_lora_configuration(package, variant, tp_size):
    """Quantized FLUX.3 variants follow the shared VisualGen LoRA rejection."""
    from tensorrt_llm.visual_gen.args import RuntimeLoRAConfig

    args = VisualGenArgs(
        model=str(package),
        pipeline_config={"variant": variant},
        parallel_config={"tp_size": tp_size},
        runtime_lora_config=RuntimeLoRAConfig(path="/tmp/adapter.safetensors"),
    )
    with pytest.raises(
        ValueError, match="does not currently support VisualGen weight quantization"
    ):
        DiffusionPipelineConfig.from_pretrained(package, args=args, device="cpu")
