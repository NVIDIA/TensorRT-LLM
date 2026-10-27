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
