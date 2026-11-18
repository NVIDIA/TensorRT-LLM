# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from datetime import timedelta
from pathlib import Path

import pytest
import torch.distributed as dist
import torch.multiprocessing as mp

from tensorrt_llm._torch.visual_gen.config import DiffusionPipelineConfig
from tensorrt_llm._torch.visual_gen.models.minimax_h3.default.pipeline import MiniMaxH3Pipeline
from tensorrt_llm._torch.visual_gen.models.minimax_h3.fasth3.pipeline import FastH3Pipeline
from tensorrt_llm._torch.visual_gen.pipeline_registry import AutoPipeline
from tensorrt_llm.mapping import Mapping

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


@pytest.mark.parametrize("fast", [False, True])
def test_factory_routes_using_resolved_checkpoint(tmp_path, fast):
    _checkpoint(tmp_path, '{"schema_version":"fasth3-v2"}' if fast else None)
    config = DiffusionPipelineConfig.from_pretrained(str(tmp_path))
    assert config.primary_pretrained_config._name_or_path == "upstream/model"
    assert AutoPipeline.resolve_class(config, str(tmp_path)) is (
        FastH3Pipeline if fast else MiniMaxH3Pipeline
    )


def _check_rank_agreement(rank, root):
    dist.init_process_group(
        "gloo",
        init_method=(root / "store").as_uri(),
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=30),
    )
    try:
        config = DiffusionPipelineConfig.from_pretrained(str(root / "base"))
        config.mapping = Mapping(world_size=2, rank=rank, tp_size=2)
        assert AutoPipeline.resolve_class(config, str(root / "base")) is MiniMaxH3Pipeline
        with pytest.raises(RuntimeError, match="differs across ranks"):
            AutoPipeline.resolve_class(config, str(root / ("fast" if rank else "base")))
        if rank:
            with pytest.raises(json.JSONDecodeError):
                AutoPipeline.resolve_class(config, str(root / "bad"))
        else:
            with pytest.raises(RuntimeError, match="rank 1"):
                AutoPipeline.resolve_class(config, str(root / "base"))
    finally:
        dist.destroy_process_group()


def test_ranks_agree_on_target_and_propagate_manifest_errors(tmp_path: Path):
    _checkpoint(tmp_path / "base")
    _checkpoint(tmp_path / "fast", '{"schema_version":"fasth3-v2"}')
    _checkpoint(tmp_path / "bad", "{")
    mp.spawn(_check_rank_agreement, args=(tmp_path,), nprocs=2)
