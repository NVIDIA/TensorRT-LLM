"""The interface for all transforms.

This module defines the base classes and interfaces for all transforms.
"""

import time
from abc import ABC
from contextlib import nullcontext
from enum import Enum
from functools import total_ordering, wraps
from typing import Any, Callable, Dict, Mapping, Tuple, Type, Union, final

import torch.nn as nn
from pydantic import BaseModel, Field
from torch.fx import GraphModule

from ..models.factory import ModelFactory
from ..shim.interface import CachedSequenceInterface
from ..utils._graph import (
    canonicalize_graph,
    lift_to_meta,
    named_graphmodules,
    placeholders_on_meta,
    run_shape_prop,
)
from ..utils.logger import ad_logger
from ..utils.sharding_utils import ShardingConfig


class TransformError(Exception):
    """An exception raised when a transform fails."""

    pass


@total_ordering
class Stages(Enum):
    """Enumerated (ordered!) stages of the transformation pipeline.

    This is used to classify and pre-order transforms.
    """

    FACTORY = "factory"  # factory stage for building the model
    EXPORT = "export"  # export stage for exporting the model to a graph module
    POST_EXPORT = "post_export"  # low-level cleanups of the exported graph
    PATTERN_MATCHER = "pattern_matcher"  # high-level pattern matching to standardize graph
    SHARDING = "sharding"  # auto-sharding of the graph
    WEIGHT_LOAD = "weight_load"  # loading of the model weights
    POST_LOAD_FUSION = "post_load_fusion"  # post-loading fusion and perf optimizations of the graph
    CACHE_INIT = "cache_init"  # initialization of cached attention + (KV) cache initialization
    VISUALIZE = "visualize"  # visualization of the graph
    COMPILE = "compile"  # graph compilation stage using low-level compilers like torch.compile

    def __lt__(self, other):
        """Enable sorting by definition order."""
        if self.__class__ is other.__class__:
            return list(self.__class__).index(self) < list(other.__class__).index(other)
        return NotImplemented


class SharedConfig(BaseModel):
    """Global config shared between multiple transforms in the inference optimizer."""

    sharding_config: ShardingConfig = Field(default_factory=ShardingConfig)
    local_rank: int = Field(default=0)
    world_size: int = Field(default=1)


class TransformConfig(BaseModel):
    """A simple configuration class that can be extended by a transform for configurability."""

    model_config = {
        # to provide an easy way to do config validation of child config classes with more fields
        "extra": "allow",
    }

    ### MANDATORY CONFIG ###########################################################################
    stage: Stages = Field(
        description="The stage of the transformation pipeline where this transform should run.",
    )

    ### OPTIONAL CONFIG ###########################################################################
    run_per_gm: bool = Field(
        description="Whether to run the transform per graph (sub)module or on whole module.",
        default=True,
    )
    enabled: bool = Field(
        default=True,
        description="Whether to enable this transform.",
    )
    skip_on_error: bool = Field(
        default=False,
        description="Whether to skip the transform if an error occurs.",
    )

    run_graph_cleanup: bool = Field(
        default=True,
        description="Whether to run graph cleanup/canonicalization after this transform.",
    )
    run_shape_prop: bool = Field(
        default=False,
        description="Whether to run shape propagation after this transform.",
    )

    requires_clean_graph: bool = Field(
        default=True,
        description="Whether this transform requires the graph to be clean before it is applied.",
    )
    requires_shape_prop: bool = Field(
        default=False,
        description="Whether this transform requires shape propagation before it is applied.",
    )


AutodeployMeta = Dict[str, Any]
_UntypedInferenceOptimizerConfig = Dict[str, Any]
StrictInferenceOptimizerConfig = Dict[str, TransformConfig]
InferenceOptimizerConfig = Mapping[str, Union[TransformConfig, _UntypedInferenceOptimizerConfig]]


class TransformInfo(BaseModel):
    """Information about the result of a transform."""

    model_config = {
        "frozen": True,  # Make the model immutable after creation
    }

    skipped: bool = Field(
        default=True,
        description="Whether the transform was skipped.",
    )
    num_matches: int = Field(
        default=0,
        description="Number of matches found.",
    )
    is_clean: bool = Field(
        default=False,
        description="Whether the graph is clean after the transform. This can be set by the "
        "transform to indicate that the transform does not change the graph and it preserves the "
        "is_clean flag of the last transform.",
    )
    has_valid_shapes: bool = Field(
        default=False,
        description="Whether meta tensor shapes are valid after the transform. This can be set by "
        "the transform to indicate that the transform does not affect the shapes in the meta "
        "information of the graph. In other words, the transform does not change the shapes of the "
        "tensors in the graph and it preserves the has_valid_shapes flag of the last transform.",
    )

    @classmethod
    def from_last_info(cls, info: "TransformInfo") -> "TransformInfo":
        """Create a new TransformInfo from the last transform info."""
        return cls(
            is_clean=info.is_clean,
            has_valid_shapes=info.has_valid_shapes,
        )

    def __or__(self, other: "TransformInfo") -> "TransformInfo":
        """Merge two TransformInfo objects."""
        return TransformInfo(
            skipped=self.skipped and other.skipped,  # we only count skipped if both were skipped
            num_matches=self.num_matches + other.num_matches,
            is_clean=self.is_clean or other.is_clean,
            has_valid_shapes=self.has_valid_shapes or other.has_valid_shapes,
        )

    def __and__(self, other: "TransformInfo") -> "TransformInfo":
        """Merge two TransformInfo objects."""
        return TransformInfo(
            skipped=self.skipped and other.skipped,  # we only count skipped if both were skipped
            num_matches=self.num_matches + other.num_matches,
            is_clean=self.is_clean and other.is_clean,
            has_valid_shapes=self.has_valid_shapes and other.has_valid_shapes,
        )


TransformHistory = Dict[str, TransformInfo]


def with_transform_logging(call_fn: Callable) -> Callable:
    """Decorator to prepend transform-specific prefix to all ad_logger logs during __call__.

    Temporarily patches `ad_logger.log` so that any logs emitted within the call automatically
    include the `[stage=..., transform=...]` prefix that `_log_info` would otherwise add manually.
    The original logger behavior is restored after the call, even if an exception occurs.
    """

    @wraps(call_fn)
    def _wrapper(
        self,
        gm: nn.Module,
        cm: CachedSequenceInterface,
        factory: ModelFactory,
        shared_config: SharedConfig,
    ) -> nn.Module:
        prefix = f"[stage={self.config.stage.value}, transform={self.get_transform_key()}]"
        original_log = ad_logger.log

        def _patched_log(severity, *msg):
            if msg and isinstance(msg[0], str) and msg[0].startswith(prefix):
                return original_log(severity, *msg)
            return original_log(severity, prefix, *msg)

        ad_logger.log = _patched_log  # type: ignore[assignment]
        try:
            return call_fn(self, gm, cm, factory, shared_config)
        finally:
            ad_logger.log = original_log  # type: ignore[assignment]

    return _wrapper


class BaseTransform(ABC):
    """A base class for all transforms."""

    config: TransformConfig  # overwrite type hint if other config cls is used in subclass!
    _autodeploy_meta_key: str = "_autodeploy"
    _history_key: str = "transform_history"
    _transform_key: str  # Set by TransformRegistry.register() decorator

    @classmethod
    def get_transform_key(cls) -> str:
        """Get the short name of the transform.

        This is used to identify the transform in the transformation pipeline.
        """
        if hasattr(cls, "_transform_key"):
            return cls._transform_key
        raise NotImplementedError(
            f"Transform class {cls.__name__} must be registered with TransformRegistry.register() "
            "or manually implement get_transform_key()"
        )

    @classmethod
    def get_config_class(cls) -> Type[TransformConfig]:
        """Get the configuration class for the transform.

        This is used to validate the configuration of the transform.
        """
        return TransformConfig

    @final
    def __init__(self, config: TransformConfig):
        """Initialize the transform.

        Args:
            config: The configuration for the transform, either as base config object or the actual
                config object.

        To customize the initialization, override the `_post_init` method.
        """
        if not isinstance(config, self.get_config_class()):
            config = self.get_config_class()(**config.model_dump())
        self.config = config
        self._post_init()

    def _post_init(self):
        """Post-initialization hook that can be overridden by subclasses."""
        pass

    @final
    @classmethod
    def from_kwargs(cls, **kwargs) -> "BaseTransform":
        """Create a transform from kwargs.

        Args:
            **kwargs: The configuration for the transform.

        Returns:
            The transform instance.
        """
        config = cls.get_config_class()(**kwargs)
        return cls(config=config)

    @with_transform_logging
    @final
    def __call__(
        self,
        mod: nn.Module,
        cm: CachedSequenceInterface,
        factory: ModelFactory,
        shared_config: SharedConfig,
    ) -> nn.Module:
        """Apply the transform to the graph.

        Args:
            mod: The model to apply the transform to.
            cm: The cached sequence interface defining the sequence interface.
            factory: The model factory used to build the model.
            shared_config: Global info shared between multiple transforms.

        Returns:
            nn.Module: The transformed model.

        NOTE: The transform can/should modify the graph module in place if possible. Returning the
        graph is mostly to standardize the interface for transforms that cannot modify the graph
        in place (e.g. the factory or export transform).

        This method is the main entry point for any transforms and is called by the
        InferenceOptimizer pipeline.
        """

        # get the transform key
        t_name = self.get_transform_key()

        # retrieve autodeploy metadata from the graphmodule
        autodeploy_meta = self._get_autodeploy_meta(mod)

        # retrieve transform history and last transform info
        history: TransformHistory = autodeploy_meta.get(self._history_key, {})
        h_keys = list(history.keys())  # preserves order of insertion/transform execution
        info_last = history[h_keys[-1]] if h_keys else TransformInfo(skipped=False, num_matches=0)

        # initialize new info object
        info = TransformInfo.from_last_info(info_last)

        # show debug info for debug config
        ad_logger.debug(f"{t_name} config: {self.config}")

        # store some timing information
        elapsed_time_total = -time.time()
        elapsed_time_pre_cleanup = 0.0
        elapsed_time_apply = 0.0
        elapsed_time_post_cleanup = 0.0

        # run or skip the transform
        if self.config.enabled:
            # run graph pre-cleanup and update info object
            elapsed_time_pre_cleanup = -time.time()
            info = info | self._run_cleanup(
                mod,
                self.config.requires_clean_graph,
                self.config.requires_shape_prop,
                info.is_clean,
                info.has_valid_shapes,
            )
            elapsed_time_pre_cleanup += time.time()

            # run the transform in a error-handling wrapper if desired
            elapsed_time_apply = -time.time()
            if self.config.skip_on_error:
                try:
                    mod, info_apply = self._apply_per_gm_or_whole_model(
                        mod, cm, factory, shared_config
                    )
                except Exception as e:
                    error_msg = f"Transform {t_name} failed"
                    ad_logger.warning(f"{error_msg}: {e}")
                    info_apply = TransformInfo(skipped=True, num_matches=0)
            else:
                # handle this here normally to improve debugging and error message
                mod, info_apply = self._apply_per_gm_or_whole_model(mod, cm, factory, shared_config)
            elapsed_time_apply += time.time()

            # we cannot say it's clean if the previous wasn't clean even if this one is
            # create new info object with updated cleanup status
            info = info & info_apply

            # run graph post-cleanup
            elapsed_time_post_cleanup = -time.time()
            info = info | self._run_cleanup(
                mod,
                self.config.run_graph_cleanup,
                self.config.run_shape_prop,
                info.is_clean,
                info.has_valid_shapes,
            )
            elapsed_time_post_cleanup += time.time()

        elapsed_time_total += time.time()

        # log the result of the transform
        log_msgs = [
            f"enabled={self.config.enabled}",
            "skipped=True" if info.skipped else f"num_matches={info.num_matches}",
            f"is_clean={info.is_clean}",
            f"has_valid_shapes={info.has_valid_shapes}",
        ]
        self._log_info(", ".join(log_msgs))
        log_msgs_timing = [
            f"elapsed time: total={elapsed_time_total:.3f}s",
            f"pre_cleanup={elapsed_time_pre_cleanup:.3f}s",
            f"apply={elapsed_time_apply:.3f}s",
            f"post_cleanup={elapsed_time_post_cleanup:.3f}s",
        ]
        self._log_info(", ".join(log_msgs_timing))
        ad_logger.debug(f"Model after {t_name}: {mod}")

        # update + store new meta data
        history[t_name] = info
        autodeploy_meta[self._history_key] = history
        self._set_autodeploy_meta(mod, autodeploy_meta)

        # return the graph module
        return mod

    @final
    def _apply_per_gm_or_whole_model(
        self,
        mod: nn.Module,
        cm: CachedSequenceInterface,
        factory: ModelFactory,
        shared_config: SharedConfig,
    ) -> Tuple[nn.Module, TransformInfo]:
        if not self.config.run_per_gm:
            return self._apply_to_full_model(mod, cm, factory, shared_config)

        # just run it on first graph module we are encountering for now...
        info = TransformInfo()
        for k, graph_sub in named_graphmodules(mod):
            graph_sub, info_apply = self._apply(graph_sub, cm, factory, shared_config)
            if k == "":
                mod = graph_sub
            else:
                mod.set_submodule(k, graph_sub)
            info = info & info_apply
        return mod, info

    @final
    def _log_info(self, *args: any):
        """Log a message with the transform key."""
        ad_logger.info(*args)

    @final
    def _get_autodeploy_meta(self, mod: nn.Module) -> AutodeployMeta:
        """Get the autodeploy metadata from the graphmodule."""
        if not hasattr(mod, "meta"):
            mod.meta = {}
        return mod.meta.get(self._autodeploy_meta_key, {})

    @final
    def _set_autodeploy_meta(self, mod: nn.Module, autodeploy_meta: AutodeployMeta) -> None:
        """Set the autodeploy metadata in the graphmodule."""
        if not hasattr(mod, "meta"):
            mod.meta = {}
        mod.meta[self._autodeploy_meta_key] = autodeploy_meta

    @final
    def _run_cleanup(
        self,
        mod: nn.Module,
        clean_graph: bool,
        clean_shape: bool,
        is_clean: bool,
        has_valid_shapes: bool,
    ) -> TransformInfo:
        """Run graph cleanup before the transform.

        Args:
            mod: The model to run cleanup on.
            clean_graph: Whether we want a clean graph after the transform.
            clean_shape: Whether we want clean shapes after the transform.
            is_clean: The current cleanup status.
            has_valid_shapes: The current shape propagation status.

        Returns:
            An info object indicating the cleanup status after this function is called.
        """
        # check if run cleanup depending on the config and info
        if clean_shape and not (is_clean and has_valid_shapes):
            self._log_info("running graph cleanup (with shape_prop)")
            canonicalize_graph(mod)
            with lift_to_meta(mod) if placeholders_on_meta(mod) else nullcontext():
                run_shape_prop(mod)
            is_clean = True
            has_valid_shapes = True
        elif clean_graph and not is_clean:
            self._log_info("running graph cleanup (no shape_prop)")
            canonicalize_graph(mod)
            is_clean = True
            has_valid_shapes = False

        return TransformInfo(is_clean=is_clean, has_valid_shapes=has_valid_shapes)

    def _apply(
        self,
        gm: GraphModule,
        cm: CachedSequenceInterface,
        factory: ModelFactory,
        shared_config: SharedConfig,
    ) -> Tuple[GraphModule, TransformInfo]:
        """Apply the transform to the graph.

        This is the core method that should be implemented by subclasses.
        """
        raise NotImplementedError(
            f"Transform {self.get_transform_key()} only supports `run_per_gm=False`."
        )

    def _apply_to_full_model(
        self,
        model: nn.Module,
        cm: CachedSequenceInterface,
        factory: ModelFactory,
        shared_config: SharedConfig,
    ) -> Tuple[nn.Module, TransformInfo]:
        """Apply the transform to the full model."""
        raise NotImplementedError(
            f"Transform {self.get_transform_key()} only supports `run_per_gm=True`."
        )


class TransformRegistry:
    """A registry for all transforms."""

    _registry: Dict[str, Type[BaseTransform]] = {}

    @classmethod
    def register(cls, name: str) -> Callable[[Type[BaseTransform]], Type[BaseTransform]]:
        def inner(fn: Type[BaseTransform]) -> Type[BaseTransform]:
            cls._registry[name] = fn
            # Auto-store the transform key as a class attribute
            fn._transform_key = name
            return fn

        return inner

    @classmethod
    def get(cls, name: str) -> Type[BaseTransform]:
        """Get the transform class by name."""
        return cls._registry[name]

    @classmethod
    def get_config_class(cls, name: str) -> Type[TransformConfig]:
        """Get the configuration class for a transform by name."""
        return cls.get(name).get_config_class()

    @classmethod
    def has(cls, name: str) -> bool:
        """Check if a transform is registered."""
        return name in cls._registry
