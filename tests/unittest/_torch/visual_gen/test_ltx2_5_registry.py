# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Native checkpoint detection must not redirect existing LTX-2 checkpoints."""

import json
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from tensorrt_llm._torch.visual_gen.models.ltx2_5.loading import _build_ltx25_transformer
from tensorrt_llm._torch.visual_gen.pipeline_registry import AutoPipeline

pytestmark = pytest.mark.cpu_only


def test_split_pack_resolution_selects_only_bf16(tmp_path, monkeypatch):
    from tensorrt_llm._torch.visual_gen.models.ltx2_5 import checkpoint

    selected = tmp_path / checkpoint.BF16_TRANSFORMER
    selected.parent.mkdir()
    selected.write_bytes(b"checkpoint")
    assert checkpoint.resolve_ltx25_checkpoint(str(tmp_path)) == str(selected)
    selected.unlink()
    selected.with_name("ltx-2.5-22b-distilled-transformer-nvfp4.safetensors").touch()
    with pytest.raises(FileNotFoundError, match="BF16 distilled"):
        checkpoint.resolve_ltx25_checkpoint(str(tmp_path))


def test_hub_resolution_downloads_only_canonical_bf16(tmp_path, monkeypatch):
    from tensorrt_llm._torch.visual_gen.models.ltx2_5 import checkpoint

    selected = tmp_path / checkpoint.BF16_TRANSFORMER
    selected.parent.mkdir()
    selected.touch()
    calls = []

    def download(model, **kwargs):
        calls.append((model, kwargs))
        return str(tmp_path)

    monkeypatch.setattr(checkpoint, "download_hf_partial", download)
    assert checkpoint.resolve_ltx25_checkpoint("Lightricks/LTX-2.5", revision="fixed") == str(
        selected
    )
    assert calls == [
        (
            "Lightricks/LTX-2.5",
            {"allow_patterns": [checkpoint.BF16_TRANSFORMER], "revision": "fixed"},
        )
    ]


@pytest.mark.parametrize("ltx25", [False, True])
def test_native_model_family_detection(tmp_path, ltx25):
    config = {"transformer": {"_class_name": "AVTransformer3DModel"}, "vae": {}}
    if ltx25:
        config["transformer"].update(
            use_keyframes_abs_pos_embedding=True, cross_attention_adaln=True, ff_bias=False
        )
    path = tmp_path / "model.safetensors"
    save_file({"marker": torch.zeros(1)}, str(path), metadata={"config": json.dumps(config)})
    assert AutoPipeline._detect_from_single_safetensors(str(path)) == (
        "LTX25ICLoraPipeline" if ltx25 else "LTX2Pipeline"
    )


def test_keyframes_flag_alone_does_not_select_ltx25(tmp_path):
    path = tmp_path / "unknown.safetensors"
    save_file(
        {"marker": torch.zeros(1)},
        str(path),
        metadata={"config": json.dumps({"transformer": {"use_keyframes_abs_pos_embedding": True}})},
    )
    assert AutoPipeline._detect_from_single_safetensors(str(path)) is None


def test_wrong_transformer_metadata_fails_before_allocation():
    config = SimpleNamespace(
        model_configs={"transformer": SimpleNamespace(pretrained_config=SimpleNamespace())}
    )
    with pytest.raises(ValueError, match="_class_name"):
        _build_ltx25_transformer(config)


@pytest.mark.parametrize("unknown", ["iclora_scal", "reference_strength"])
def test_pipeline_rejects_unknown_config_before_allocating(unknown):
    from tensorrt_llm._torch.visual_gen.models.ltx2_5 import LTX25ICLoraPipeline

    config = SimpleNamespace(extra_attrs={unknown: 1.0})
    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        LTX25ICLoraPipeline(config)


def test_native_metadata_is_not_a_user_pipeline_option():
    from tensorrt_llm._torch.visual_gen.models.ltx2_5 import LTX25ICLoraPipeline

    config = SimpleNamespace(extra_attrs={"monolithic_safetensors_config": {}})
    # Native metadata is accepted; missing required component paths fail next.
    with pytest.raises(ValueError, match="requires pipeline_config.text_encoder_path"):
        LTX25ICLoraPipeline(config)


@pytest.mark.parametrize("option", ["torch_compile", "cuda_graph", "cache_backend"])
def test_unqualified_acceleration_fails_before_allocation(option):
    from tensorrt_llm._torch.visual_gen.models.ltx2_5 import LTX25ICLoraPipeline

    config = SimpleNamespace(
        extra_attrs={
            name: "component.safetensors"
            for name in (
                "text_encoder_path",
                "video_vae_path",
                "audio_vae_path",
                "spatial_upsampler_path",
                "iclora_path",
            )
        },
        torch_dtype=torch.bfloat16,
        attention=SimpleNamespace(
            backend="VANILLA", quant_attention_config=None, sparse_attention_config=None
        ),
        runtime_lora=None,
        parallel=SimpleNamespace(cfg_size=1, tp_size=1, seq_parallel_size=1, parallel_vae_size=1),
        cache_backend=None,
        torch_compile=SimpleNamespace(enable=False),
        cuda_graph=SimpleNamespace(enable=False),
    )
    if option == "cache_backend":
        config.cache_backend = "test"
    else:
        getattr(config, option).enable = True
    with pytest.raises(ValueError, match="Disable cache acceleration"):
        LTX25ICLoraPipeline(config)


def test_model_registration_does_not_require_pyav():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; sys.modules['av'] = None; "
            "import tensorrt_llm._torch.visual_gen.models.ltx2_5",
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("version", ["5.5.4", "5.9.0", "5.15.0"])
def test_gemma4_version_guard_precedes_checkpoint_access(monkeypatch, version):
    from tensorrt_llm._torch.visual_gen.models.ltx2_5 import text_encoder

    monkeypatch.setattr(text_encoder.transformers, "__version__", version)
    with pytest.raises(RuntimeError, match="model-specific environment"):
        text_encoder.load_gemma4_config("missing-checkpoint", "gemma4")
    with pytest.raises(RuntimeError, match="model-specific environment"):
        text_encoder.load_gemma4_model("missing-checkpoint", None, torch.bfloat16, "cpu")
