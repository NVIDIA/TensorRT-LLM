"""The interface for all transforms.

This module defines the base classes and interfaces for all transforms.
"""

import os
import time
from abc import ABC
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from enum import Enum
from functools import total_ordering, wraps
from typing import Any, Callable, Dict, Mapping, Optional, Tuple, Type, Union, final

import torch
import torch.nn as nn
from pydantic import BaseModel, Field
from torch.fx import GraphModule, Node

from ..models.factory import ModelFactory
from ..shim.interface import CachedSequenceInterface
from ..utils._graph import (
    add_graph_input,
    canonicalize_graph,
    lift_to_meta,
    named_graphmodules,
    placeholders_on_meta,
    run_shape_prop,
)
from ..utils.cuda_mem_tracker import get_mem_info
from ..utils.logger import ad_logger
from .graph_module_visualizer import to_dot

# ANSI color codes for log formatting (set to False to disable colors)
# NOTE: colors disabled by default to make logging in CI/CD pipelines easier to read
_ENABLE_LOG_COLORS = False


class _Colors:
    RESET = "\033[0m" if _ENABLE_LOG_COLORS else ""
    BOLD = "\033[1m" if _ENABLE_LOG_COLORS else ""
    DIM = "\033[2m" if _ENABLE_LOG_COLORS else ""
    CYAN = "\033[36m" if _ENABLE_LOG_COLORS else ""
    MAGENTA = "\033[35m" if _ENABLE_LOG_COLORS else ""
    GREEN = "\033[32m" if _ENABLE_LOG_COLORS else ""
    YELLOW = "\033[33m" if _ENABLE_LOG_COLORS else ""
    ORANGE = "\033[38;5;208m" if _ENABLE_LOG_COLORS else ""


@dataclass
class MemStats:
    """Memory statistics snapshot for tracking CUDA memory usage."""

    tot: float
    free: float
    resv: float
    alloc: float
    frag: float

    def diff(self, other: "MemStats") -> "MemStats":
        """Calculate the difference (self - other)."""
        return MemStats(
            tot=self.tot - other.tot,
            free=self.free - other.free,
            resv=self.resv - other.resv,
            alloc=self.alloc - other.alloc,
            frag=self.frag - other.frag,
        )

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for serialization."""
        return {
            "tot": self.tot,
            "free": self.free,
            "resv": self.resv,
            "alloc": self.alloc,
            "frag": self.frag,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, float]) -> "MemStats":
        """Create from dictionary."""
        return cls(tot=d["tot"], free=d["free"], resv=d["resv"], alloc=d["alloc"], frag=d["frag"])


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
    EXPORT_ONNX = "export_onnx"  # export the graph to onnx
    COMPILE = "compile"  # graph compilation stage using low-level compilers like torch.compile

    def __lt__(self, other):
        """Enable sorting by definition order."""
        if self.__class__ is other.__class__:
            return list(self.__class__).index(self) < list(other.__class__).index(other)
        return NotImplemented


class SharedConfig(BaseModel):
    """Global config shared between multiple transforms in the inference optimizer."""

    model_config = {
        # to provide an easy way to do config validation of child config classes with more fields
        "extra": "allow",
    }
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
    debug_visualize_dir: Optional[str] = Field(
        default=None,
        description="Debug visualization directory. None to disable visualization, "
        "or a path string to specify the output directory.",
    )

    expect_mem_change: bool = Field(
        default=False,
        description="Whether this transform is expected to cause changes in CUDA memory stats.",
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

    # implement + addition operator for TransformInfo
    def __add__(self, other: "TransformInfo") -> "TransformInfo":
        return self.__and__(other)


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
        idx: int,
    ) -> nn.Module:
        prefix = f"[stage={self.config.stage.value}, transform={self.get_transform_key()}]"
        original_log = ad_logger.log

        def _patched_log(severity, *msg):
            if msg and isinstance(msg[0], str) and msg[0].startswith(prefix):
                return original_log(severity, *msg)
            return original_log(severity, prefix, *msg)

        ad_logger.log = _patched_log  # type: ignore[assignment]
        try:
            return call_fn(self, gm, cm, factory, shared_config, idx)
        finally:
            ad_logger.log = original_log  # type: ignore[assignment]

    return _wrapper


class BaseTransform(ABC):
    """A base class for all transforms."""

    config: TransformConfig  # overwrite type hint if other config cls is used in subclass!
    _autodeploy_meta_key: str = "_autodeploy"
    _history_key: str = "transform_history"
    _mem_history_key: str = "mem_history"
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
        idx: int,
    ) -> nn.Module:
        """Apply the transform to the graph.

        Args:
            mod: The model to apply the transform to.
            cm: The cached sequence interface defining the sequence interface.
            factory: The model factory used to build the model.
            shared_config: Global info shared between multiple transforms.
            idx: The index of the transform in the pipeline.

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

        # capture memory stats at the start
        mem_pre = self._get_mem_stats(empty_cache=True)

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
                phase="pre",
            )
            elapsed_time_pre_cleanup += time.time()

            # run the transform in a error-handling wrapper if desired
            elapsed_time_apply = -time.time()
            with self._apply_logging_context():
                self._log_info("applying transform...")
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
                    mod, info_apply = self._apply_per_gm_or_whole_model(
                        mod, cm, factory, shared_config
                    )
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
                phase="post",
            )
            elapsed_time_post_cleanup += time.time()

        elapsed_time_total += time.time()

        # capture memory stats at the end and log summary (only log if enabled)
        mem_post = self._get_mem_stats(empty_cache=True)
        if self.config.enabled:
            self._log_mem_summary(mem_pre, mem_post, self.config.expect_mem_change)

        # log the result of the transform
        self._log_transform_summary(
            enabled=self.config.enabled,
            skipped=info.skipped,
            num_matches=info.num_matches,
            elapsed_total=elapsed_time_total,
            elapsed_pre=elapsed_time_pre_cleanup,
            elapsed_apply=elapsed_time_apply,
            elapsed_post=elapsed_time_post_cleanup,
        )
        ad_logger.debug(f"Model after {t_name}: {mod}")

        # update + store new meta data (transform history and memory history)
        history[t_name] = info
        autodeploy_meta[self._history_key] = history

        # store memory history
        mem_history: Dict[str, Dict[str, Dict[str, float]]] = autodeploy_meta.get(
            self._mem_history_key, {}
        )
        mem_history[t_name] = {"pre": mem_pre.to_dict(), "post": mem_post.to_dict()}
        autodeploy_meta[self._mem_history_key] = mem_history

        self._set_autodeploy_meta(mod, autodeploy_meta)
        self._visualize_graph(mod, idx)

        # return the graph module
        return mod

    @final
    def _visualize_graph(self, mod: nn.Module, idx: int) -> None:
        """Visualize the graph if debug visualization is enabled.
        Args:
            mod: The graph module to visualize.
            idx: The index of the transform in the pipeline.
        Note:
            we may want to consider doing this for each subgraph.
            See https://github.com/NVIDIA/TensorRT-LLM/issues/10203
        """
        if not isinstance(mod, torch.fx.GraphModule):
            return
        visualize_dir = self.config.debug_visualize_dir
        if not visualize_dir:
            return
        if not os.path.exists(visualize_dir):
            os.makedirs(visualize_dir)
        name_stem = f"gm_{idx + 1:02d}_{self.get_transform_key()}"
        visualize_path = os.path.join(visualize_dir, f"{name_stem}")
        to_dot(mod, name=name_stem, save_path=visualize_path, format="svg")
        ad_logger.debug(f"[{idx + 1:02d}] Visualized {name_stem} to {visualize_path}")

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
        info = None
        for k, graph_sub in named_graphmodules(mod):
            graph_sub, info_apply = self._apply(graph_sub, cm, factory, shared_config)
            if k == "":
                mod = graph_sub
            else:
                mod.set_submodule(k, graph_sub)
            info = info & info_apply if info is not None else info_apply
        return mod, info

    @final
    def _log_warning(self, *args: any):
        """Log a warning message with the transform key."""
        ad_logger.warning(*args)

    @final
    def _log_info(self, *args: any):
        """Log a message with the transform key."""
        ad_logger.info(*args)

    @final
    def _log_debug(self, *args: any):
        """Log a message with the transform key."""
        ad_logger.debug(*args)

    @contextmanager
    def _apply_logging_context(self):
        """Context manager to add [APPLY] prefix to logs during transform execution."""
        original_log = ad_logger.log
        apply_label = "[APPLY]"

        def _patched_log(severity, *msg):
            # Prepend [APPLY] after any existing prefix
            if msg:
                first = msg[0] if isinstance(msg[0], str) else str(msg[0])
                return original_log(severity, f"{apply_label} {first}", *msg[1:])
            return original_log(severity, apply_label)

        ad_logger.log = _patched_log  # type: ignore[assignment]
        try:
            yield
        finally:
            ad_logger.log = original_log  # type: ignore[assignment]

    @final
    def _log_cleanup_status(self, phase: str, action: str, reason: str = "") -> None:
        """Log cleanup status with colored formatting.

        Args:
            phase: "pre" or "post"
            action: "ran" or "skipped"
            reason: Description of what ran or why skipped
        """
        label = f"{_Colors.CYAN}[{phase.upper()}-CLEANUP]{_Colors.RESET}"
        if action == "skipped":
            self._log_info(f"{label} {_Colors.DIM}skipped ({reason}){_Colors.RESET}")
        else:
            self._log_info(f"{label} {reason}")

    @final
    def _log_transform_summary(
        self,
        enabled: bool,
        skipped: bool,
        num_matches: int,
        elapsed_total: float,
        elapsed_pre: float,
        elapsed_apply: float,
        elapsed_post: float,
    ) -> None:
        """Log transform summary with colored formatting.

        Args:
            enabled: Whether the transform was enabled.
            skipped: Whether the transform was skipped.
            num_matches: Number of matches found.
            elapsed_total: Total elapsed time in seconds.
            elapsed_pre: Pre-cleanup elapsed time in seconds.
            elapsed_apply: Apply elapsed time in seconds.
            elapsed_post: Post-cleanup elapsed time in seconds.
        """
        label = f"{_Colors.GREEN}[SUMMARY]{_Colors.RESET}"
        timing_str = (
            f"{elapsed_total:.3f}s "
            f"(pre={elapsed_pre:.3f}s, apply={elapsed_apply:.3f}s, post={elapsed_post:.3f}s)"
        )

        if not enabled:
            self._log_info(f"{label} {_Colors.DIM}disabled{_Colors.RESET}")
        elif skipped:
            self._log_info(f"{label} {_Colors.DIM}skipped{_Colors.RESET} | time: {timing_str}")
        else:
            self._log_info(f"{label} matches={num_matches} | time: {timing_str}")

    @final
    def _get_mem_stats(self, empty_cache: bool = True) -> MemStats:
        """Get current CUDA memory statistics.

        Args:
            empty_cache: Whether to empty the memory cache before getting the memory stats.

        Returns:
            MemStats object with current memory values in GB.
        """
        tot, free, resv, alloc, frag = get_mem_info(empty_cache=empty_cache, unit="GB")
        return MemStats(tot=tot, free=free, resv=resv, alloc=alloc, frag=frag)

    @final
    def _log_mem_summary(self, pre: MemStats, post: MemStats, expect_mem_change: bool) -> None:
        """Log memory summary with diff between pre and post stats.

        Logs one of three cases:
        1. Expected mem change: info log, magenta color
        2. Unexpected mem change: warning log, yellow color
        3. No mem change: debug log, no colors

        Args:
            pre: Memory stats captured before the transform.
            post: Memory stats captured after the transform.
            expect_mem_change: Whether this transform is expected to cause memory changes.
        """
        diff = post.diff(pre)

        # Threshold for detecting significant memory changes (in GB)
        mem_change_threshold = 0.005

        # Check if there was a significant memory change
        has_mem_change = (
            abs(diff.resv) >= mem_change_threshold
            or abs(diff.alloc) >= mem_change_threshold
            or abs(diff.frag) >= mem_change_threshold
            or abs(diff.free) >= mem_change_threshold
        )

        def _fmt_val_with_delta(val: float, delta: float, color: str) -> str:
            """Format value with optional delta in the specified color."""
            val_str = f"{val:6.2f}GB"
            if abs(delta) < mem_change_threshold:
                return val_str
            sign = "+" if delta > 0 else ""
            if color:
                return f"{val_str} {_Colors.BOLD}{color}({sign}{delta:.2f}GB){_Colors.RESET}"
            return f"{val_str} ({sign}{delta:.2f}GB)"

        def _fmt_parts(color: str) -> str:
            """Format all memory parts with the specified color for deltas."""
            parts = [
                f"free: {_fmt_val_with_delta(post.free, diff.free, color)}",
                f"resv: {_fmt_val_with_delta(post.resv, diff.resv, color)}",
                f"alloc: {_fmt_val_with_delta(post.alloc, diff.alloc, color)}",
                f"frag: {_fmt_val_with_delta(post.frag, diff.frag, color)}",
            ]
            return " | ".join(parts)

        if has_mem_change and expect_mem_change:
            # Case 1: Expected mem change - info log, magenta
            label = f"{_Colors.MAGENTA}[CUDA MEM DIFF (EXPECTED)]{_Colors.RESET}"
            self._log_info(f"{label} {_fmt_parts(_Colors.MAGENTA)}")
        elif has_mem_change and not expect_mem_change:
            # Case 2: Unexpected mem change - warning log, yellow
            label = f"{_Colors.YELLOW}[CUDA MEM DIFF (UNEXPECTED)]{_Colors.RESET}"
            self._log_warning(f"{label} {_fmt_parts(_Colors.YELLOW)}")
        else:
            # Case 3: No mem change - debug log, no colors
            self._log_debug(f"[CUDA MEM] {_fmt_parts('')}")

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
        phase: str,
    ) -> TransformInfo:
        """Run graph cleanup before or after the transform.

        Args:
            mod: The model to run cleanup on.
            clean_graph: Whether we want a clean graph after the transform.
            clean_shape: Whether we want clean shapes after the transform.
            is_clean: The current cleanup status.
            has_valid_shapes: The current shape propagation status.
            phase: The phase of cleanup ("pre" or "post").

        Returns:
            An info object indicating the cleanup status after this function is called.
        """
        # check if run cleanup depending on the config and info
        if clean_shape and not (is_clean and has_valid_shapes):
            self._log_cleanup_status(phase, "ran", "graph canonicalization + shape_prop")
            canonicalize_graph(mod)
            with lift_to_meta(mod) if placeholders_on_meta(mod) else nullcontext():
                run_shape_prop(mod)
            is_clean = True
            has_valid_shapes = True
        elif clean_graph and not is_clean:
            self._log_cleanup_status(phase, "ran", "graph canonicalization")
            canonicalize_graph(mod)
            is_clean = True
            has_valid_shapes = False
        elif not clean_graph and not clean_shape:
            self._log_cleanup_status(phase, "skipped", "disabled")
        else:
            self._log_cleanup_status(phase, "skipped", "graph already clean")

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

    def _add_or_retrieve_input(
        self, gm: GraphModule, cm: CachedSequenceInterface, name: str
    ) -> Node:
        """Add or retrieve an input node from the graph."""
        input_nodes = gm.graph.find_nodes(op="placeholder", target=name)
        if len(input_nodes) == 0:
            cm.info.activate_arg(name)
            return add_graph_input(gm, name)
        elif len(input_nodes) == 1:
            return input_nodes[0]
        else:
            raise ValueError(f"Expected exactly one input node for {name=}, got {input_nodes=}")


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
