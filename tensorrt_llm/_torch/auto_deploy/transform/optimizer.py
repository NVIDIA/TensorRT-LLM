"""High-level entrypoint to transform a model into an efficient inference model."""

import gc
import os
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn as nn

from ..distributed import common as dist_ad
from ..models.factory import ModelFactory
from ..shim.interface import CachedSequenceInterface
from .graph_module_visualizer import to_dot
from .interface import (
    InferenceOptimizerConfig,
    SharedConfig,
    Stages,
    StrictInferenceOptimizerConfig,
    TransformConfig,
    TransformRegistry,
)


class InferenceOptimizer:
    def __init__(self, factory: ModelFactory, config: InferenceOptimizerConfig):
        self.factory = factory
        self.config = self._clean_config(config)
        if not dist.is_initialized():
            local_rank, world_size = 0, 1
        else:
            local_rank, world_size = dist_ad.get_rank_world_size()
        self.shared_config = SharedConfig(local_rank=local_rank, world_size=world_size)

    def _clean_config(self, config: InferenceOptimizerConfig) -> StrictInferenceOptimizerConfig:
        """Get a typed checked ("strict") config with sorted keys according to stages."""
        # convert to nested kwargs, no TransformConfig objects allowed
        nested_kwargs = {
            k: v.model_dump() if isinstance(v, TransformConfig) else v for k, v in config.items()
        }
        # sort by stage
        keys_sorted = sorted(nested_kwargs.keys(), key=lambda k: Stages(nested_kwargs[k]["stage"]))
        # create strict config with correct config classes and correct order
        strict_config: StrictInferenceOptimizerConfig = {
            k: TransformRegistry.get_config_class(k)(**nested_kwargs[k]) for k in keys_sorted
        }
        # return strict config
        return strict_config

    def __call__(self, cm: CachedSequenceInterface, mod: Optional[nn.Module] = None) -> nn.Module:
        """Transform a model into an optimized inference model.

        Args:
            cm: The cached sequence interface defining the sequence interface.
            mod: The model to transform.

        Returns:
            A nn.Module representing the optimized inference model.
        """
        ############################################################################################
        # RUN THROUGH CONFIGURED TRANSFORMATIONS
        ############################################################################################

        # start with an empty model if not provided
        if mod is None:
            mod = nn.Module()

        # iterate over all transforms sorted by stage in the config
        for idx, (t_name, t_config) in enumerate(self.config.items()):
            # instantiate transform
            transform = TransformRegistry.get(t_name)(t_config)
            # run transform
            mod = transform(mod, cm, self.factory, self.shared_config)

            if isinstance(mod, torch.fx.GraphModule):
                # Generate a graphviz diagram if the environment variable AD_DEBUG_VISUALIZE_DIR is set
                visualize_dir = os.environ.get("AD_DEBUG_VISUALIZE_DIR", None)
                if visualize_dir:
                    if not os.path.exists(visualize_dir):
                        os.makedirs(visualize_dir)
                    name_stem = f"gm_{idx + 1:02d}_{t_name}"
                    visualize_path = os.path.join(visualize_dir, f"{name_stem}")
                    to_dot(mod, name=name_stem, save_path=visualize_path, format="svg")
                    print(
                        f"[{idx + 1:02d}/{len(self.config)}] Visualized {name_stem} to {visualize_path}"
                    )

        ############################################################################################
        # RETURN OPTIMIZED MODEL
        ############################################################################################
        torch.cuda.empty_cache()
        gc.collect()
        return mod
