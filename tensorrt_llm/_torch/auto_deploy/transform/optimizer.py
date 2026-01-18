"""High-level entrypoint to transform a model into an efficient inference model."""

import gc
from typing import Optional, Set

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.fx import GraphModule

from ..cache import CacheKey, CompilationCacheConfig, CompilationCacheManager
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

# With torch.export.save(), we can only cache at the export stage
# because transforms modify the graph in ways that can't be re-exported
CACHE_STAGE = Stages.EXPORT


class InferenceOptimizer:
    """Transform a model into an optimized inference model.

    This class orchestrates the transformation pipeline and optionally
    caches intermediate results to speed up subsequent runs.
    """

    def __init__(
        self,
        factory: ModelFactory,
        config: InferenceOptimizerConfig,
        cache_config: Optional[CompilationCacheConfig] = None,
    ):
        """Initialize the InferenceOptimizer.

        Args:
            factory: Model factory for building the model.
            config: Transform pipeline configuration.
            cache_config: Optional cache configuration. If None, caching is disabled.
        """
        self.factory = factory
        self.config = self._clean_config(config)

        # Setup distributed info
        if not dist.is_initialized():
            local_rank, world_size = 0, 1
        else:
            local_rank, world_size = dist_ad.get_rank_world_size()
        self.shared_config = SharedConfig(local_rank=local_rank, world_size=world_size)

        # Setup cache manager
        self.cache_config = cache_config
        self.cache_manager: Optional[CompilationCacheManager] = None

        if cache_config is not None and cache_config.enabled:
            self.cache_manager = CompilationCacheManager(cache_config)

            # Generate cache key from model and transforms config
            cache_key = CacheKey.from_config(
                model=factory.model or "",
                transforms_config={k: v.model_dump() for k, v in self.config.items()},
                world_size=world_size,
                local_rank=local_rank,
            )
            self.cache_manager.set_cache_key(cache_key)
            ad_logger.info(
                f"Compilation cache enabled: {cache_config.cache_dir}, "
                f"enabled_transforms={cache_key.enabled_transforms[:5]}..."
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

    def _save_cache(self, mod: nn.Module) -> None:
        """Save the graph to cache using torch.export.save().

        The ExportedProgram must be attached to the module as _exported_program.
        This is done by the export_to_gm transform.
        """
        if not isinstance(mod, GraphModule):
            ad_logger.warning(f"Cannot save cache: model is {type(mod).__name__}, not GraphModule")
            return

        # Get the ExportedProgram that was attached during export
        exported_program = getattr(mod, "_exported_program", None)
        if exported_program is None:
            ad_logger.warning("Cannot save cache: no ExportedProgram attached to module")
            return

        self.cache_manager.save_graph_to_cache(
            gm=mod,
            transform_history={},
            cached_stage=CACHE_STAGE.value,
            exported_program=exported_program,
        )
        ad_logger.info(f"Saved compilation cache (stage: {CACHE_STAGE.value})")

    def __call__(self, cm: CachedSequenceInterface, mod: Optional[nn.Module] = None) -> nn.Module:
        """Transform a model into an optimized inference model.

        Args:
            cm: The cached sequence interface defining the sequence interface.
            mod: The model to transform.

        Returns:
            A nn.Module representing the optimized inference model.
        """
        ############################################################################################
        # CHECK FOR CACHED GRAPH
        ############################################################################################

        # With torch.export.save(), we cache at the EXPORT stage only
        # On load, we skip FACTORY and EXPORT stages
        skip_stages: Set[Stages] = set()
        loaded_from_cache = False

        if self.cache_manager is not None and self.cache_manager.has_valid_cache():
            cached_result = self.cache_manager.load_cached_graph()
            if cached_result is not None:
                mod, metadata = cached_result
                loaded_from_cache = True

                # Skip FACTORY and EXPORT stages (covered by cache)
                skip_stages = {Stages.FACTORY, Stages.EXPORT}

                ad_logger.info(
                    f"Loaded from cache (stage: {CACHE_STAGE.value}), "
                    f"skipping stages: {[s.value for s in skip_stages]}"
                )

        ############################################################################################
        # RUN THROUGH CONFIGURED TRANSFORMATIONS
        ############################################################################################

        # start with an empty model if not provided
        if mod is None:
            mod = nn.Module()

        # Determine if we should save cache after export
        should_save_cache = (
            self.cache_manager is not None
            and not loaded_from_cache  # Don't re-save if we just loaded
        )
        cache_saved = False

        # iterate over all transforms sorted by stage in the config
        for t_name, t_config in self.config.items():
            current_stage = t_config.stage

            # Skip stages that are covered by cache
            if current_stage in skip_stages:
                ad_logger.debug(f"Skipping {t_name} (loaded from cache)")
                continue

            # instantiate transform
            transform = TransformRegistry.get(t_name)(t_config)
            # run transform
            mod = transform(mod, cm, self.factory, self.shared_config)

            # Save cache right after the export stage completes
            if should_save_cache and not cache_saved and current_stage == CACHE_STAGE:
                self._save_cache(mod)
                cache_saved = True

        ############################################################################################
        # RETURN OPTIMIZED MODEL
        ############################################################################################
        torch.cuda.empty_cache()
        gc.collect()
        return mod
