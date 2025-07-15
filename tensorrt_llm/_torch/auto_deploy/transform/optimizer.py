"""High-level entrypoint to transform a model into an efficient inference model."""

import torch.nn as nn
from torch.fx import Graph, GraphModule

from ..models.factory import ModelFactory
from ..shim.interface import CachedSequenceInterface
from .interface import (
    InferenceOptimizerConfig,
    Stages,
    TransformConfig,
    TransformRegistry,
    _StrictInferenceOptimizerConfig,
)


class InferenceOptimizer:
    def __init__(self, factory: ModelFactory, config: InferenceOptimizerConfig):
        self.factory = factory
        self.config = self._clean_config(config)

    def _clean_config(self, config: InferenceOptimizerConfig) -> _StrictInferenceOptimizerConfig:
        """Get a typed checked ("strict") config with sorted keys according to stages."""
        # convert to nested kwargs, no TransformConfig objects allowed
        nested_kwargs = {
            k: v.model_dump() if isinstance(v, TransformConfig) else v for k, v in config.items()
        }
        # sort by stage
        keys_sorted = sorted(nested_kwargs.keys(), key=lambda k: Stages(nested_kwargs[k]["stage"]))
        # create strict config with correct config classes and correct order
        strict_config: _StrictInferenceOptimizerConfig = {
            k: TransformRegistry.get_config_class(k)(**nested_kwargs[k]) for k in keys_sorted
        }
        # return strict config
        return strict_config

    @staticmethod
    def _init_gm() -> GraphModule:
        """Initialize a fake graph module.

        This is a dummy graph module that will be used to kick off the transforms.
        """
        return GraphModule(nn.Module(), Graph())

    def __call__(self, cm: CachedSequenceInterface) -> GraphModule:
        """Transform a model into an optimized inference model.

        Args:
            cm: The cached sequence interface defining the sequence interface.

        Returns:
            A GraphModule representing the optimized inference model.
        """
        ############################################################################################
        # RUN THROUGH CONFIGURED TRANSFORMATIONS
        ############################################################################################

        # start with an empty fake graph module
        gm = self._init_gm()

        # iterate over all transforms sorted by stage in the config
        for t_name, t_config in self.config.items():
            # instantiate transform
            transform = TransformRegistry.get(t_name)(t_config)
            # run transform
            gm = transform(gm, cm, self.factory)

        ############################################################################################
        # RETURN OPTIMIZED GRAPH
        ############################################################################################
        return gm
