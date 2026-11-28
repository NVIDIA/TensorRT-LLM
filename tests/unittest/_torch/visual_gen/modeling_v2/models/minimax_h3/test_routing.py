# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import pytest
import torch

from tensorrt_llm._torch.visual_gen.config import DiffusionPipelineConfig
from tensorrt_llm._torch.visual_gen.mapping import VisualGenMapping
from tensorrt_llm._torch.visual_gen.modeling_v2.models.minimax_h3.fasth3.pipeline import (
    FastH3Pipeline,
)
from tensorrt_llm._torch.visual_gen.modeling_v2.models.minimax_h3.h3_pipeline.pipeline import (
    MiniMaxH3Pipeline,
)
from tensorrt_llm._torch.visual_gen.modeling_v2.models.minimax_h3.pipeline_registry import route
from tensorrt_llm._torch.visual_gen.pipeline_registry import PIPELINE_REGISTRY, AutoPipeline
from tensorrt_llm.visual_gen.args import AttentionConfig, ParallelConfig, VisualGenArgs

pytestmark = pytest.mark.cpu_only


def _checkpoint(path, manifest=None):
    (path / "transformer").mkdir(parents=True)
    (path / "transformer/config.json").write_text(json.dumps({"_name_or_path": "upstream/model"}))
    (path / "modular_model_index.json").write_text(
        json.dumps(
            {
                "_class_name": "MiniMaxH3ModularPipeline",
                "transformer": ["diffusers", "MiniMaxH3Transformer3DModel"],
            }
        )
    )
    if manifest is not None:
        (path / "fastvideo_inference.json").write_text(manifest)


def test_h3_family_registers_v2_pipeline_with_its_router() -> None:
    entry = PIPELINE_REGISTRY["MiniMaxH3ModularPipeline"]
    assert entry.pipeline_cls is MiniMaxH3Pipeline
    assert entry.router is route


def test_router_selects_target_per_checkpoint(tmp_path: Path) -> None:
    _checkpoint(tmp_path / "base")
    _checkpoint(tmp_path / "fast", '{"schema_version":"fasth3-v2"}')

    def resolve(name):
        checkpoint = str(tmp_path / name)
        config = DiffusionPipelineConfig.from_pretrained(checkpoint)
        config.device = "cpu"
        assert config.primary_pretrained_config._name_or_path == "upstream/model"
        return AutoPipeline.resolve_class(config, checkpoint), config

    base, base_config = resolve("base")
    assert base is MiniMaxH3Pipeline
    assert base_config.attention.sparse_attention_config is None
    fast, fast_config = resolve("fast")
    assert fast is FastH3Pipeline
    assert fast_config.primary_pretrained_config._fasth3_manifest == {"schema_version": "fasth3-v2"}
    assert fast_config.attention.sparse_attention_config.algorithm == "vsa"


def test_fasth3_rejects_text_encoder_tp_on_one_gpu(tmp_path: Path) -> None:
    _checkpoint(tmp_path, json.dumps({"schema_version": "fasth3-inference-contract-v1"}))
    args = VisualGenArgs(
        model=str(tmp_path), parallel_config=ParallelConfig(text_encoder_tp_size=2)
    )
    config = DiffusionPipelineConfig.from_pretrained(str(tmp_path), args=args)
    with pytest.raises(ValueError, match=r"text_encoder_tp_size \(2\) exceeds world_size \(1\)"):
        AutoPipeline.resolve_class(config, str(tmp_path))


def test_fast_checkpoint_requires_a_single_gpu(tmp_path: Path) -> None:
    _checkpoint(tmp_path, '{"schema_version":"fasth3-v2"}')
    config = DiffusionPipelineConfig.from_pretrained(str(tmp_path))
    config.visual_gen_mapping = VisualGenMapping(world_size=2, rank=0, ulysses_size=2)
    with pytest.raises(ValueError, match="FastH3 VSA requires a single GPU"):
        AutoPipeline.resolve_class(config, str(tmp_path))


@pytest.mark.parametrize("recipe", ["NVFP4", "FP8_PER_CHANNEL_PER_TOKEN"])
def test_h3_quantization_keeps_other_linear_methods_unchanged(tmp_path, monkeypatch, recipe):
    from tensorrt_llm._torch.modules.linear import FP8RowwiseLinearMethod, NVFP4LinearMethod

    method = NVFP4LinearMethod if recipe == "NVFP4" else FP8RowwiseLinearMethod
    monkeypatch.setattr(method, "use_tunable_quantize", False)
    _checkpoint(tmp_path)
    (tmp_path / "transformer/config.json").write_text(
        json.dumps({"quantization_config": {"quant_algo": recipe}})
    )
    DiffusionPipelineConfig.from_pretrained(str(tmp_path))
    assert not method.use_tunable_quantize


def test_fast_checkpoint_rejects_reference_workflow_at_load(tmp_path):
    _checkpoint(tmp_path, '{"schema_version":"fasth3-v2"}')
    config = DiffusionPipelineConfig.from_pretrained(str(tmp_path))
    config.extra_attrs["workflow"] = "ref2va"
    with pytest.raises(ValueError, match="FastH3 VSA supports text-to-video only"):
        AutoPipeline.resolve_class(config, str(tmp_path))


@pytest.mark.parametrize(
    ("backend", "capability", "cute_available"),
    [
        ("TRTLLM", (9, 0), True),
        ("CUTEDSL", (9, 0), True),
        ("CUTEDSL", (12, 0), True),
        ("CUTEDSL", (10, 0), False),
    ],
)
def test_fast_checkpoint_rejects_unsupported_cuda_backend(
    tmp_path, monkeypatch, backend, capability, cute_available
):
    from tensorrt_llm._torch.visual_gen.modeling_v2.catalog.attention import block_sparse_attention

    _checkpoint(tmp_path, '{"schema_version":"fasth3-v2"}')
    config = DiffusionPipelineConfig.from_pretrained(str(tmp_path))
    config.attention = AttentionConfig(backend=backend)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _: capability)
    monkeypatch.setattr(block_sparse_attention, "CUTE_AVAILABLE", cute_available)
    with pytest.raises(NotImplementedError, match=f"FastH3 {backend} attention requires"):
        AutoPipeline.resolve_class(config, str(tmp_path))
