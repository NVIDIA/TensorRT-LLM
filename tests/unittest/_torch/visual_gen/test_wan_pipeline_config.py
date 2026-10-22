# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU tests for the Wan pipeline_config key enable_fp8_block_fusions."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

# Importing models applies the pipeline registration decorators.
from tensorrt_llm._torch.visual_gen import models  # noqa: F401
from tensorrt_llm._torch.visual_gen.config import DiffusionModelConfig, DiffusionPipelineConfig
from tensorrt_llm._torch.visual_gen.models.wan.transformer_wan import (
    WanBlock,
    WanTransformer3DModel,
)
from tensorrt_llm._torch.visual_gen.pipeline_loader import PipelineLoader
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.visual_gen.args import VisualGenArgs

pytestmark = pytest.mark.cpu_only


@pytest.fixture
def wan_checkpoint(tmp_path: Path) -> str:
    """Only the metadata that pipeline detection and configuration read."""
    (tmp_path / "model_index.json").write_text(
        json.dumps(
            {
                "_class_name": "WanPipeline",
                "transformer": ["diffusers", "WanTransformer3DModel"],
                "vae": ["diffusers", "AutoencoderKLWan"],
            }
        ),
        encoding="utf-8",
    )
    transformer_dir = tmp_path / "transformer"
    transformer_dir.mkdir()
    (transformer_dir / "config.json").write_text(
        json.dumps({"_class_name": "WanTransformer3DModel"}), encoding="utf-8"
    )
    return str(tmp_path)


def test_block_fusions_are_off_by_default(wan_checkpoint: str) -> None:
    args = VisualGenArgs(model=wan_checkpoint)
    resolved = PipelineLoader(args, device="cpu")._resolve_pipeline_config(wan_checkpoint)
    assert resolved == {"enable_fp8_block_fusions": False}
    assert args.pipeline_config == {}


def test_block_fusions_override_reaches_the_model_config(wan_checkpoint: str) -> None:
    args = VisualGenArgs(model=wan_checkpoint, pipeline_config={"enable_fp8_block_fusions": True})
    resolved = PipelineLoader(args, device="cpu")._resolve_pipeline_config(wan_checkpoint)
    assert resolved == {"enable_fp8_block_fusions": True}

    config = DiffusionPipelineConfig.from_pretrained(
        wan_checkpoint, args=args, pipeline_config=resolved, device="cpu"
    )
    assert config.extra_attrs["enable_fp8_block_fusions"] is True


def test_unknown_pipeline_config_key_is_rejected(wan_checkpoint: str) -> None:
    args = VisualGenArgs(model=wan_checkpoint, pipeline_config={"enable_fp8_block_fusion": True})
    with pytest.raises(ValueError, match="Unknown pipeline_config keys for WanPipeline"):
        PipelineLoader(args, device="cpu")._resolve_pipeline_config(wan_checkpoint)


@pytest.mark.parametrize("enabled", [False, True])
def test_post_load_weights_plans_block_fusions_only_when_enabled(monkeypatch, enabled) -> None:
    planned = []
    monkeypatch.setattr(
        WanBlock, "plan_fp8_fusions", lambda self, log=False: planned.append(self) or False
    )
    heads, head_dim = 2, 128
    config = DiffusionModelConfig(
        pretrained_config=SimpleNamespace(
            attention_head_dim=head_dim,
            cross_attn_norm=True,
            eps=1e-6,
            ffn_dim=512,
            freq_dim=256,
            in_channels=16,
            num_attention_heads=heads,
            num_layers=2,
            out_channels=16,
            patch_size=[1, 2, 2],
            qk_norm="rms_norm_across_heads",
            rope_max_seq_len=1024,
            text_dim=256,
            hidden_size=heads * head_dim,
            torch_dtype="bfloat16",
        ),
        quant_config=QuantConfig(),
        skip_create_weights_in_init=False,
    )
    config.extra_attrs["enable_fp8_block_fusions"] = enabled
    model = WanTransformer3DModel(model_config=config)
    model.post_load_weights()
    assert len(planned) == (2 if enabled else 0)
