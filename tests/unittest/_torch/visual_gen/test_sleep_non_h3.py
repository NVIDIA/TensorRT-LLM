# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Local/QA default-loader regressions with tiny real Wan and FLUX checkpoints."""

import json
from pathlib import Path
from unittest.mock import Mock

import pytest
import torch
from diffusers import FluxTransformer2DModel, WanTransformer3DModel

from tensorrt_llm import VisualGenArgs
from tensorrt_llm._torch.visual_gen.models.flux import FluxPipeline
from tensorrt_llm._torch.visual_gen.models.wan.pipeline_wan import WanPipeline
from tensorrt_llm._torch.visual_gen.pipeline_loader import PipelineLoader
from tensorrt_llm._torch.visual_gen.pipeline_registry import PipelineComponent
from tensorrt_llm._torch.visual_gen.sleep import PipelineSleepManager

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.fixture(params=["wan", "flux"])
def tiny_checkpoint(request: pytest.FixtureRequest, tmp_path: Path) -> tuple:
    family = request.param
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(7)
        if family == "wan":
            reference = WanTransformer3DModel(
                num_attention_heads=2,
                attention_head_dim=64,
                in_channels=4,
                out_channels=4,
                text_dim=32,
                freq_dim=32,
                ffn_dim=64,
                num_layers=1,
                patch_size=(1, 2, 2),
                rope_max_seq_len=32,
            )
            pipeline_cls = WanPipeline
            expected_weight = reference.proj_out.weight.detach().clone()
        else:
            reference = FluxTransformer2DModel(
                num_attention_heads=2,
                attention_head_dim=64,
                in_channels=4,
                out_channels=4,
                num_layers=1,
                num_single_layers=1,
                joint_attention_dim=32,
                pooled_projection_dim=32,
                axes_dims_rope=(16, 24, 24),
            )
            pipeline_cls = FluxPipeline
            expected_weight = reference.x_embedder.weight.detach().clone()
    reference.save_pretrained(tmp_path / "transformer")
    (tmp_path / "model_index.json").write_text(
        json.dumps(
            {
                "_class_name": pipeline_cls.__name__,
                "transformer": ["diffusers", type(reference).__name__],
            }
        )
    )
    return tmp_path, family, pipeline_cls, expected_weight


@pytest.mark.parametrize("explicit_none", [False, True])
@pytest.mark.parametrize("skip_warmup", [False, True])
def test_non_h3_default_load_does_not_enable_sleep(
    tiny_checkpoint: tuple,
    monkeypatch: pytest.MonkeyPatch,
    explicit_none: bool,
    skip_warmup: bool,
) -> None:
    checkpoint, family, pipeline_cls, expected_weight = tiny_checkpoint
    capture = Mock(side_effect=AssertionError("Default loading entered sleep allocation capture"))
    monkeypatch.setattr(PipelineSleepManager, "__init__", capture)
    warmup = Mock()
    monkeypatch.setattr(pipeline_cls, "warmup", warmup)
    config = VisualGenArgs(
        model=str(checkpoint),
        torch_compile_config={"enable": False},
        attention_config={"backend": "VANILLA"},
    )
    options = {"sleep_restore_mode": None} if explicit_none else {}
    pipeline = PipelineLoader(config).load(
        skip_warmup=skip_warmup,
        skip_components=[
            item for item in PipelineComponent if item != PipelineComponent.TRANSFORMER
        ],
        **options,
    )
    assert type(pipeline) is pipeline_cls
    for name in ("_sleep_manager", "sleep", "wake_up", "is_sleeping", "supports_sleep"):
        assert not hasattr(pipeline, name)
    capture.assert_not_called()
    assert warmup.call_count == int(not skip_warmup)
    transformer = pipeline.transformer
    actual_weight = (
        transformer.proj_out.weight if family == "wan" else transformer.x_embedder.weight
    )
    torch.testing.assert_close(
        actual_weight.cpu(), expected_weight.to(actual_weight.dtype), rtol=0, atol=0
    )
    assert all(not tensor.is_meta for tensor in pipeline.parameters())
    assert actual_weight.is_cuda

    with torch.inference_mode():
        if family == "wan":
            latents = torch.ones(1, 4, 1, 4, 4, device="cuda", dtype=actual_weight.dtype)
            result = transformer(
                hidden_states=latents,
                timestep=torch.tensor([0.5], device="cuda"),
                encoder_hidden_states=torch.ones(
                    1, 4, 32, device="cuda", dtype=actual_weight.dtype
                ),
            )
        else:
            latents = torch.ones(1, 4, 4, device="cuda", dtype=actual_weight.dtype)
            result = transformer(
                hidden_states=latents,
                timestep=torch.tensor([0.5], device="cuda"),
                encoder_hidden_states=torch.ones(
                    1, 4, 32, device="cuda", dtype=actual_weight.dtype
                ),
                pooled_projections=torch.ones(1, 32, device="cuda", dtype=actual_weight.dtype),
                img_ids=torch.zeros(4, 3, device="cuda"),
                txt_ids=torch.zeros(4, 3, device="cuda"),
                return_dict=False,
            )[0]
    assert result.shape == latents.shape
    assert torch.isfinite(result).all()
    assert result.abs().max() > 0
    capture.assert_not_called()


def test_non_h3_sleep_rejected_before_weight_loading(
    tiny_checkpoint: tuple,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint, _, pipeline_cls, _ = tiny_checkpoint
    load_weights = Mock(side_effect=AssertionError("Unsupported sleep request loaded weights"))
    monkeypatch.setattr(pipeline_cls, "load_transformer_weights", load_weights)
    loader = PipelineLoader(VisualGenArgs(model=str(checkpoint)))
    materialize = Mock(side_effect=AssertionError("Unsupported sleep request materialized tensors"))
    monkeypatch.setattr(loader, "_materialize_meta_tensors", materialize)
    with pytest.raises(ValueError, match="only for MiniMax-H3"):
        loader.load(sleep_restore_mode="PINNED")
    materialize.assert_not_called()
    load_weights.assert_not_called()
