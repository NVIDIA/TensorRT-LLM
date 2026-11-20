# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
from typing import TYPE_CHECKING, Optional, Type

from tensorrt_llm.logger import logger

from .pipeline_registry import AutoPipeline, _PipelineEntry, _PipelineResolver

if TYPE_CHECKING:
    from .config import DiffusionPipelineConfig
    from .pipeline import BasePipeline


def resolve_opt_in(
    config: "DiffusionPipelineConfig",
    checkpoint_dir: str,
    class_name: str,
    entry: _PipelineEntry,
    opt_in: tuple[str, _PipelineResolver],
) -> Optional[Type["BasePipeline"]]:
    env_var, resolver = opt_in
    enabled = os.environ.get(env_var) == "1"
    mapping = getattr(config, "visual_gen_mapping", None) or getattr(config, "mapping", None)
    multi_rank = getattr(mapping, "world_size", 1) > 1
    if not enabled and not multi_rank:
        return None

    import torch.distributed as dist

    distributed = multi_rank and dist.is_initialized()
    if not enabled and not distributed:
        return None

    from torch.utils._python_dispatch import _disable_current_modes

    # All ranks in an opt-in family agree, including ranks with the switch off.
    # The collective needs real tensors even inside the loader's MetaInitMode.
    with _disable_current_modes():
        pipeline_class = None
        error = None
        try:
            AutoPipeline._validate_vae_quantization(config, class_name, entry)
            pipeline_class = (
                resolver(config, checkpoint_dir)
                if enabled
                else entry.pipeline_cls.resolve_variant(config)
            )
        except Exception as exc:
            if not distributed:
                raise
            error = exc
        identity = (
            f"{pipeline_class.__module__}.{pipeline_class.__qualname__}"
            if pipeline_class is not None
            else None
        )
        if distributed:
            decisions = [None] * dist.get_world_size()
            dist.all_gather_object(decisions, (identity, str(error) if error is not None else None))
            if error is not None:
                raise error
            for rank, (_, failure) in enumerate(decisions):
                if failure is not None:
                    raise RuntimeError(f"VisualGen routing failed on rank {rank}: {failure}")
            if any(selected != identity for selected, _ in decisions):
                raise RuntimeError(
                    f"VisualGen pipeline selection differs across ranks: {decisions}"
                )

    assert pipeline_class is not None
    target = identity if enabled else pipeline_class.__name__
    logger.info(f"AutoPipeline: Creating {target} from {checkpoint_dir}")
    return pipeline_class
