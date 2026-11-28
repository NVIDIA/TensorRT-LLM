# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import os
import sys
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from test_common.visual_gen.h3_rollout import H3_V2

from tensorrt_llm._torch.models.modeling_utils import MetaInitMode
from tensorrt_llm._torch.visual_gen.config import DiffusionPipelineConfig
from tensorrt_llm._torch.visual_gen.mapping import VisualGenMapping
from tensorrt_llm._torch.visual_gen.pipeline_registry import AutoPipeline
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


def test_fasth3_rejects_text_encoder_tp_on_one_gpu(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _checkpoint(tmp_path, json.dumps({"schema_version": "fasth3-inference-contract-v1"}))
    monkeypatch.setenv(H3_V2, "1")
    args = VisualGenArgs(
        model=str(tmp_path), parallel_config=ParallelConfig(text_encoder_tp_size=2)
    )
    config = DiffusionPipelineConfig.from_pretrained(str(tmp_path), args=args)
    with pytest.raises(ValueError, match=r"text_encoder_tp_size \(2\) exceeds world_size \(1\)"):
        AutoPipeline.resolve_class(config, str(tmp_path))


@pytest.mark.parametrize("recipe", ["NVFP4", "FP8_PER_CHANNEL_PER_TOKEN"])
def test_opt_in_quantization_keeps_other_linear_methods_unchanged(tmp_path, monkeypatch, recipe):
    from tensorrt_llm._torch.modules.linear import FP8RowwiseLinearMethod, NVFP4LinearMethod

    method = NVFP4LinearMethod if recipe == "NVFP4" else FP8RowwiseLinearMethod
    monkeypatch.setattr(method, "use_tunable_quantize", False)
    _checkpoint(tmp_path)
    (tmp_path / "transformer/config.json").write_text(
        json.dumps({"quantization_config": {"quant_algo": recipe}})
    )
    monkeypatch.setenv(H3_V2, "1")
    DiffusionPipelineConfig.from_pretrained(str(tmp_path))
    assert not method.use_tunable_quantize
    monkeypatch.delenv(H3_V2)
    DiffusionPipelineConfig.from_pretrained(str(tmp_path))
    assert method.use_tunable_quantize


def _check_rollout(_rank, root):
    from tensorrt_llm._torch.visual_gen.models.minimax_h3.pipeline_minimax_h3 import (
        MiniMaxH3Pipeline as LegacyPipeline,
    )

    def resolve(name):
        checkpoint = str(root / name)
        config = DiffusionPipelineConfig.from_pretrained(checkpoint)
        config.device = "cpu"
        assert config.primary_pretrained_config._name_or_path == "upstream/model"
        return AutoPipeline.resolve_class(config, checkpoint)

    assert H3_V2 not in os.environ
    assert resolve("base") is resolve("fast") is LegacyPipeline
    assert "tensorrt_llm._torch.visual_gen.modeling_v2.models.minimax_h3" not in sys.modules

    os.environ[H3_V2] = "1"
    base, fast = resolve("base"), resolve("fast")
    from tensorrt_llm._torch.visual_gen.modeling_v2.models.minimax_h3.fasth3.pipeline import (
        FastH3Pipeline,
    )
    from tensorrt_llm._torch.visual_gen.modeling_v2.models.minimax_h3.h3_pipeline.pipeline import (
        MiniMaxH3Pipeline,
    )

    assert base is MiniMaxH3Pipeline
    assert fast is FastH3Pipeline
    os.environ[H3_V2] = "0"
    assert resolve("base") is resolve("fast") is LegacyPipeline


def test_rollout_selects_v1_and_v2_per_checkpoint(tmp_path, monkeypatch):
    monkeypatch.delenv(H3_V2, raising=False)
    _checkpoint(tmp_path / "base")
    _checkpoint(tmp_path / "fast", '{"schema_version":"fasth3-v2"}')
    mp.spawn(_check_rollout, args=(tmp_path,), nprocs=1)


def test_family_without_opt_in_keeps_local_factory_path(tmp_path, monkeypatch):
    from tensorrt_llm._torch.visual_gen.models.wan import WanPipeline

    _checkpoint(tmp_path)
    index = tmp_path / "modular_model_index.json"
    index.write_text(
        json.dumps(
            {"_class_name": "WanPipeline", "transformer": ["diffusers", "WanTransformer3DModel"]}
        )
    )
    config = DiffusionPipelineConfig.from_pretrained(str(tmp_path))

    class Mesh:
        @property
        def world_size(self):
            pytest.fail("An unregistered rollout inspected the worker topology")

    config.visual_gen_mapping = Mesh()
    monkeypatch.setenv(H3_V2, "1")
    monkeypatch.setattr(WanPipeline, "__init__", lambda self, cfg: torch.nn.Module.__init__(self))
    with MetaInitMode():
        assert type(AutoPipeline.from_config(config, str(tmp_path))) is WanPipeline


def _check_rank_agreement(rank, root):
    from tensorrt_llm._torch.visual_gen.modeling_v2.models.minimax_h3.h3_pipeline.pipeline import (
        MiniMaxH3Pipeline,
    )

    visual_mapping = VisualGenMapping(world_size=2, rank=rank, ulysses_size=2)
    tp_mapping = visual_mapping.to_llm_mapping()
    assert tp_mapping.world_size == 1
    dist.init_process_group(
        "gloo",
        init_method=(root / "store").as_uri(),
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=30),
    )
    try:
        config = DiffusionPipelineConfig.from_pretrained(str(root / "base"))
        config.visual_gen_mapping = visual_mapping
        config.mapping = tp_mapping

        def construct(self, cfg):
            torch.nn.Module.__init__(self)
            self.weight = torch.nn.Parameter(torch.empty(1))

        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(MiniMaxH3Pipeline, "__init__", construct)
            with MetaInitMode():
                pipeline = AutoPipeline.from_config(config, str(root / "base"))
            assert type(pipeline) is MiniMaxH3Pipeline and pipeline.weight.is_meta
        with pytest.raises(
            ValueError if rank else RuntimeError, match="FastH3 VSA requires a single GPU"
        ):
            AutoPipeline.resolve_class(config, str(root / ("fast" if rank else "base")))
        if rank:
            with pytest.raises(json.JSONDecodeError):
                AutoPipeline.resolve_class(config, str(root / "bad"))
        else:
            with pytest.raises(RuntimeError, match="rank 1"):
                AutoPipeline.resolve_class(config, str(root / "base"))
        with pytest.MonkeyPatch.context() as patch:
            patch.setenv(H3_V2, str(rank))
            with pytest.raises(RuntimeError, match="differs across ranks"):
                AutoPipeline.resolve_class(config, str(root / "base"))
    finally:
        dist.destroy_process_group()


def test_ranks_agree_on_target_and_propagate_manifest_errors(tmp_path: Path):
    _checkpoint(tmp_path / "base")
    _checkpoint(tmp_path / "fast", '{"schema_version":"fasth3-v2"}')
    _checkpoint(tmp_path / "bad", "{")
    mp.spawn(_check_rank_agreement, args=(tmp_path,), nprocs=2)


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
