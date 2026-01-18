"""High-level entrypoint to transform a model into an efficient inference model."""

import gc
from typing import Optional, Set

import torch
import torch.distributed as dist
import torch.nn as nn

from ..cache import CacheKey, ExportCacheConfig, GraphSerializer
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
    """Transform a model into an optimized inference model."""

    def __init__(
        self,
        factory: ModelFactory,
        config: InferenceOptimizerConfig,
    ):
        """Initialize the InferenceOptimizer.

        Args:
            factory: Model factory for building the model.
            config: Transform pipeline configuration.
        """
        self.factory = factory
        self.config = self._clean_config(config)

        # Setup distributed info
        if not dist.is_initialized():
            local_rank, world_size = 0, 1
        else:
            local_rank, world_size = dist_ad.get_rank_world_size()
        self.shared_config = SharedConfig(local_rank=local_rank, world_size=world_size)

        # Setup cache config from export_to_gm transform config
        self.cache_config = self._get_cache_config()
        self.cache_path = self._get_cache_path() if self.cache_config else None

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

    def _get_cache_config(self) -> Optional[ExportCacheConfig]:
        """Get cache config from export_to_gm transform."""
        export_config = self.config.get("export_to_gm")
        if export_config is None:
            return None

        enable_cache = getattr(export_config, "enable_cache", True)
        if not enable_cache:
            return None

        cache_dir = getattr(export_config, "cache_dir", None)
        return ExportCacheConfig(cache_dir=cache_dir, enabled=enable_cache)

    def _get_cache_path(self):
        """Get the cache path for this configuration."""
        if not self.cache_config or not self.cache_config.enabled:
            return None

        export_config = self.config.get("export_to_gm")
        model_config = {
            "strict": getattr(export_config, "strict", False),
            "patch_list": getattr(export_config, "patch_list", None),
            "model_kwargs": self.factory.model_kwargs,
        }

        cache_key = CacheKey.from_model_config(
            model=self.factory.model or "",
            model_config=model_config,
        )
        return cache_key.to_cache_path(self.cache_config.cache_dir)

    def _try_load_from_cache(self) -> Optional[nn.Module]:
        """Try to load from cache, return None if not found."""
        if self.cache_path is None:
            return None

        if not GraphSerializer.is_valid_cache(self.cache_path):
            return None

        try:
            gm, metadata = GraphSerializer.load(self.cache_path)
            ad_logger.info(f"Loaded from cache: {self.cache_path}")
            return gm
        except Exception as e:
            ad_logger.warning(f"Failed to load from cache: {e}")
            return None

    def _save_to_cache(self, gm: nn.Module) -> None:
        """Save the exported graph to cache."""
        if self.cache_path is None:
            return

        exported_program = getattr(gm, "_exported_program", None)
        if exported_program is None:
            ad_logger.warning("No ExportedProgram attached, cannot save to cache")
            return

        try:
            export_config = self.config.get("export_to_gm")
            metadata = {
                "strict": getattr(export_config, "strict", False),
                "patch_list": getattr(export_config, "patch_list", None),
            }
            GraphSerializer.save(exported_program, self.cache_path, metadata)
            ad_logger.info(f"Saved to cache: {self.cache_path}")
        except Exception as e:
            ad_logger.warning(f"Failed to save to cache: {e}")

    def __call__(self, cm: CachedSequenceInterface, mod: nn.Module = None) -> nn.Module:
        """Transform a model into an optimized inference model.

        Args:
            cm: The cached sequence interface defining the sequence interface.
            mod: The model to transform.

        Returns:
            A nn.Module representing the optimized inference model.
        """
        # Try to load from cache first - this skips FACTORY and EXPORT stages
        skip_stages: Set[Stages] = set()
        cached_mod = self._try_load_from_cache()
        if cached_mod is not None:
            mod = cached_mod
            skip_stages = {Stages.FACTORY, Stages.EXPORT}
            ad_logger.info(f"Loaded from cache, skipping stages: {[s.value for s in skip_stages]}")
        elif mod is None:
            mod = nn.Module()

        # Track if we need to save cache after export
        should_save_cache = self.cache_path is not None and not skip_stages
        cache_saved = False

        # iterate over all transforms sorted by stage in the config
        for t_name, t_config in self.config.items():
            current_stage = t_config.stage

            # Skip stages if loaded from cache
            if current_stage in skip_stages:
                ad_logger.debug(f"Skipping {t_name} (loaded from cache)")
                continue

            # instantiate transform
            transform = TransformRegistry.get(t_name)(t_config)
            # run transform
            mod = transform(mod, cm, self.factory, self.shared_config)

            # Save cache after EXPORT stage completes
            if should_save_cache and not cache_saved and current_stage == Stages.EXPORT:
                self._save_to_cache(mod)
                cache_saved = True

        torch.cuda.empty_cache()
        gc.collect()
        return mod
