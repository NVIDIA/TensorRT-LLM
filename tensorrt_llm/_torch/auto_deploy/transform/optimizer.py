"""High-level entrypoint to transform a model into an efficient inference model."""

import gc
import time
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn as nn

from ..distributed import common as dist_ad
from ..models.factory import ModelFactory
from ..shim.interface import CachedSequenceInterface
from ..utils.logger import ad_logger
from .graph_cache import GraphCache
from .interface import (
    InferenceOptimizerConfig,
    SharedConfig,
    Stages,
    StrictInferenceOptimizerConfig,
    TransformConfig,
    TransformRegistry,
)


class InferenceOptimizer:
    def __init__(self, factory: ModelFactory, config: InferenceOptimizerConfig, mapping=None):
        self.factory = factory
        self.config = self._clean_config(config)
        if not dist.is_initialized():
            local_rank, world_size = 0, 1
        else:
            local_rank, world_size = dist_ad.get_rank_world_size()
        self.shared_config = SharedConfig(
            local_rank=local_rank, world_size=world_size, mapping=mapping
        )

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

        When graph caching is enabled (``AD_ENABLE_CACHING=1``), the optimizer
        will attempt to load a previously cached graph from disk.  On a cache
        hit every transform with ``stage < WEIGHT_LOAD`` is skipped and the
        pipeline resumes from the ``WEIGHT_LOAD`` stage.  On a cache miss the
        full pipeline runs from the beginning and the graph is saved to the
        cache at the ``WEIGHT_LOAD`` boundary for future reuse.

        Args:
            cm: The cached sequence interface defining the sequence interface.
            mod: The model to transform.

        Returns:
            A nn.Module representing the optimized inference model.
        """
        ############################################################################################
        # GRAPH CACHE LOOKUP
        ############################################################################################
        if mod is None:
            mod = nn.Module()

        cache = GraphCache()
        cache_hit = False
        cache_key: Optional[str] = None

        if cache.is_enabled():
            pre_weight_config = {
                k: v for k, v in self.config.items() if v.stage < Stages.WEIGHT_LOAD
            }
            cache_key = cache.compute_cache_key(
                pre_weight_config=pre_weight_config,
                model_id=self.factory.model,
                model_kwargs=self.factory.model_kwargs,
                world_size=self.shared_config.world_size,
                rank=self.shared_config.local_rank,
                max_seq_len=cm.info.max_seq_len,
                max_batch_size=cm.info.max_batch_size,
            )
            cached_mod = cache.load(cache_key)
            if cached_mod is not None:
                mod = cached_mod
                cache_hit = True
                ad_logger.info("[cache] Cache hit -- resuming from WEIGHT_LOAD stage")
            else:
                ad_logger.info("[cache] Cache miss -- running full pipeline from beginning")

        ############################################################################################
        # RUN THROUGH CONFIGURED TRANSFORMATIONS
        ############################################################################################
        start_time = time.time()
        prev_stage = None

        for idx, (t_name, t_config) in enumerate(self.config.items()):
            # CACHE HIT: skip all pre-weight-load transforms
            if cache_hit and t_config.stage < Stages.WEIGHT_LOAD:
                ad_logger.info(f"[cache] Skipping {t_name} (loaded from cache)")
                continue

            # CACHE MISS: save graph at the boundary before first WEIGHT_LOAD transform
            if (
                cache_key is not None
                and not cache_hit
                and prev_stage is not None
                and prev_stage < Stages.WEIGHT_LOAD
                and t_config.stage >= Stages.WEIGHT_LOAD
            ):
                cache.save(cache_key, mod)

            # run the transform
            transform = TransformRegistry.get(t_name)(t_config)
            mod = transform(mod, cm, self.factory, self.shared_config, idx)
            prev_stage = t_config.stage

        total_time = time.time() - start_time
        ad_logger.info(f"Total time for all transforms: {total_time:.2f}s")

        ############################################################################################
        # RETURN OPTIMIZED MODEL
        ############################################################################################
        torch.cuda.empty_cache()
        gc.collect()
        return mod
