"""High-level entrypoint to transform a model into an efficient inference model."""

import gc
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn as nn

from ..distributed import common as dist_ad
from ..models.factory import ModelFactory
from ..shim.interface import CachedSequenceInterface
from ..utils.logger import ad_logger
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
        # SETUP DEBUG DUMPING IF ENABLED
        ############################################################################################
        if ad_logger.debug_dump_enabled:
            # Save inputs for replay
            inputs_to_save = dict(cm.named_args) if hasattr(cm, "named_args") else None
            ad_logger.set_debug_inputs(inputs_to_save)

        ############################################################################################
        # RUN THROUGH CONFIGURED TRANSFORMATIONS
        ############################################################################################

        # start with an empty model if not provided
        if mod is None:
            mod = nn.Module()

        # iterate over all transforms sorted by stage in the config
        for t_name, t_config in self.config.items():
            # instantiate transform
            transform = TransformRegistry.get(t_name)(t_config)
            # run transform
            mod = transform(mod, cm, self.factory, self.shared_config)

            # Dump after export_to_gm (post-export state)
            if ad_logger.debug_dump_enabled and t_name == "export_to_gm":
                model_path = getattr(self.factory, "checkpoint_path", None)
                ad_logger.dump_debug_artifacts(mod, "post_export", model_path=model_path)
            if t_name == "export_to_gm":
                from tensorrt_llm._torch.auto_deploy.utils.dtype_metadata import dump_dtype_metadata

                dump_dtype_metadata(mod, "dtype_metadata.json")
        ############################################################################################
        # DUMP FINAL STATE IF DEBUG ENABLED
        ############################################################################################
        if ad_logger.debug_dump_enabled:
            model_path = getattr(self.factory, "checkpoint_path", None)
            ad_logger.dump_debug_artifacts(mod, "final", model_path=model_path)

            from tensorrt_llm._torch.auto_deploy.utils.graph_debug_compare import run_comparison

            run_comparison(mod, cm, self.factory, output_dir="1layer_subblock_debug_scatter_plots")
        ############################################################################################
        # RETURN OPTIMIZED MODEL
        ############################################################################################
        torch.cuda.empty_cache()
        gc.collect()
        return mod
