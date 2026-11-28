# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pytest


def _record_worker_pipeline(directory: Path, **worker_args):
    import torch.distributed as dist

    from tensorrt_llm._torch.visual_gen import executor
    from tensorrt_llm.visual_gen.visual_gen import run_diffusion_worker

    class RecordingLoader(executor.PipelineLoader):
        def load(self, *args, **kwargs):
            pipeline = super().load(*args, **kwargs)
            target = type(pipeline)
            (directory / str(dist.get_rank())).write_text(
                f"{target.__module__}.{target.__qualname__}"
            )
            return pipeline

    with patch.object(executor, "PipelineLoader", RecordingLoader):
        run_diffusion_worker(**worker_args)


def load_default_h3_visual_gen(args):
    """Start the public executor and check the pipeline instance loaded by every rank."""
    from tensorrt_llm import VisualGen
    from tensorrt_llm._torch.visual_gen import executor
    from tensorrt_llm._torch.visual_gen.modeling_v2.models.minimax_h3.h3_pipeline.pipeline import (
        MiniMaxH3Pipeline,
    )

    expected = f"{MiniMaxH3Pipeline.__module__}.{MiniMaxH3Pipeline.__qualname__}"
    with TemporaryDirectory() as tmp:
        directory = Path(tmp)
        worker = partial(_record_worker_pipeline, directory)
        with patch.object(executor, "run_diffusion_worker", worker):
            engine = VisualGen(model=args.model, args=args)
        targets = {int(path.name): path.read_text() for path in directory.iterdir()}
        if targets != dict.fromkeys(range(args.parallel_config.n_workers), expected):
            engine.shutdown()
            pytest.fail(
                f"Expected {expected} on every H3 worker, loaded targets by rank: {targets}"
            )
    return engine
