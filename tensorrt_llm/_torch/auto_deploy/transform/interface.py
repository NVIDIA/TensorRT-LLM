"""The interface for all transforms.

This module defines the base classes and interfaces for all transforms.
"""

from abc import ABC, abstractmethod
from enum import Enum
from functools import total_ordering
from typing import Any, Callable, Dict, Mapping, Tuple, Type, Union, final

from pydantic import BaseModel, Field
from torch.fx import GraphModule

from ..models.factory import ModelFactory
from ..shim.interface import CachedSequenceInterface
from ..transformations._graph import canonicalize_graph, lift_to_meta
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
    attn_backend: str = Field(default="flashinfer", description="The attention backend to use.")


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
        description="Whether the transform was skipped.",
    )
    num_matches: int = Field(
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


TransformHistory = Dict[str, TransformInfo]


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

    @final
    def __call__(
        self,
        gm: GraphModule,
        cm: CachedSequenceInterface,
        factory: ModelFactory,
        shared_config: SharedConfig,
    ) -> GraphModule:
        """Apply the transform to the graph.

        Args:
            gm: The graph module to apply the transform to.
            cm: The cached sequence interface defining the sequence interface.
            factory: The model factory used to build the model.
            shared_config: Global info shared between multiple transforms.

        Returns:
            GraphModule: The transformed graph module.

        NOTE: The transform can/should modify the graph module in place if possible. Returning the
        graph is mostly to standardize the interface for transforms that cannot modify the graph
        in place (e.g. the factory or export transform).

        This method is the main entry point for any transforms and is called by the
        InferenceOptimizer pipeline.
        """

        # get the transform key
        t_name = self.get_transform_key()

        # retrieve autodeploy metadata from the graphmodule
        autodeploy_meta = self._get_autodeploy_meta(gm)

        # retrieve transform history and last transform info
        history: TransformHistory = autodeploy_meta.get(self._history_key, {})
        h_keys = list(history.keys())  # preserves order of insertion/transform execution
        info_last = history[h_keys[-1]] if h_keys else TransformInfo(skipped=False, num_matches=0)

        # show debug info for debug config
        ad_logger.debug(f"{t_name} config: {self.config}")

        # run or skip the transform
        if self.config.enabled:
            # run graph pre-cleanup
            is_clean_pre, has_valid_shapes_pre = self._run_pre_cleanup(gm, info_last)

            # run the transform in a error-handling wrapper if desired
            if self.config.skip_on_error:
                try:
                    gm, info = self._apply(gm, cm, factory, shared_config)
                except Exception as e:
                    error_msg = f"Transform {t_name} failed"
                    ad_logger.warning(f"{error_msg}: {e}")
                    info = TransformInfo(skipped=True, num_matches=0)
            else:
                # handle this here normally to improve debugging and error message
                gm, info = self._apply(gm, cm, factory, shared_config)

            # we cannot say it's clean if the previous wasn't clean even if this one is
            # create new info object with updated cleanup status
            info_dict = info.model_dump()
            info_dict["is_clean"] &= is_clean_pre
            info_dict["has_valid_shapes"] &= has_valid_shapes_pre
            info = TransformInfo(**info_dict)

            # run graph post-cleanup
            info = self._run_post_cleanup(gm, info)
        else:
            # skip the transform and set info object using the last transform info
            info_dict = info_last.model_dump()
            info_dict["skipped"] = True
            info_dict["num_matches"] = 0
            info = TransformInfo(**info_dict)

        # log the result of the transform
        log_msgs = [
            f"stage={self.config.stage.value}",
            f"transform={t_name}",
            "skipped=True" if info.skipped else f"num_matches={info.num_matches}",
            f"is_clean={info.is_clean}",
            f"has_valid_shapes={info.has_valid_shapes}",
        ]
        ad_logger.info(", ".join(log_msgs))
        ad_logger.debug(f"Graph after {t_name}: {gm}")

        # update + store new meta data
        history[t_name] = info
        autodeploy_meta[self._history_key] = history

        if isinstance(gm, GraphModule):
            # After compilation, gm becomes type CapturedGraph with no meta data.
            self._set_autodeploy_meta(gm, autodeploy_meta)

        # return the graph module
        return gm

    @final
    def _get_autodeploy_meta(self, gm: GraphModule) -> AutodeployMeta:
        """Get the autodeploy metadata from the graphmodule."""
        return gm.meta.get(self._autodeploy_meta_key, {})

    @final
    def _set_autodeploy_meta(self, gm: GraphModule, autodeploy_meta: AutodeployMeta) -> None:
        """Set the autodeploy metadata in the graphmodule."""
        gm.meta[self._autodeploy_meta_key] = autodeploy_meta

    @final
    def _run_pre_cleanup(self, gm: GraphModule, info: TransformInfo) -> Tuple[bool, bool]:
        """Run graph cleanup before the transform.

        Args:
            gm: The graph module to run cleanup on.
            info: The last transform info.

        Returns:
            A tuple of (is_clean, has_valid_shapes) indicating the cleanup status after the
            pre-cleanup.

        This is used to ensure the transform is applied to a clean graph as needed by the transform.
        """
        if not self.config.requires_clean_graph:
            return info.is_clean, info.has_valid_shapes

        is_clean = info.is_clean
        has_valid_shapes = is_clean and info.has_valid_shapes

        # check if run cleanup depending on the config and info
        if self.config.requires_shape_prop and not has_valid_shapes:
            with lift_to_meta(gm):
                canonicalize_graph(gm, shape_prop=True)
            is_clean = True
            has_valid_shapes = True
        elif self.config.requires_clean_graph and not is_clean:
            canonicalize_graph(gm)
            is_clean = True

        return is_clean, has_valid_shapes

    @final
    def _run_post_cleanup(self, gm: GraphModule, info: TransformInfo) -> TransformInfo:
        """Run graph cleanup after the transform.

        Cleanup is done as requested in the config and we will update the graph module and info
        accordingly.

        Returns:
            Updated TransformInfo with cleanup status.
        """
        if not self.config.run_graph_cleanup:
            return info

        # check if run cleanup depending on the config and info
        if self.config.run_shape_prop and not (info.is_clean and info.has_valid_shapes):
            with lift_to_meta(gm):
                canonicalize_graph(gm, shape_prop=True)
        elif self.config.run_graph_cleanup and not info.is_clean:
            canonicalize_graph(gm)

        # create new info object with updated cleanup status
        info_dict = info.model_dump()
        info_dict["is_clean"] |= self.config.run_graph_cleanup
        info_dict["has_valid_shapes"] |= self.config.run_shape_prop
        return TransformInfo(**info_dict)

    @abstractmethod
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
