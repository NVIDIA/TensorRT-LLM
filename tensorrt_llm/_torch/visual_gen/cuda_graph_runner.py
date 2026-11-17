# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import contextlib
import contextvars
import functools
import gc
import inspect
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Dict,
    Hashable,
    List,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    TypeAlias,
)

import torch
from torch import nn
from torch.utils._pytree import keystr, tree_flatten_with_path, tree_unflatten

from tensorrt_llm.logger import logger

from ..nccl_window_graph import nccl_window_graph_capture
from ..nccl_window_tensor_scope import discard_nccl_window_tensor_outputs
from ..utils import make_weak_ref

# Extra-key values the runner resolved host-side for the call it is currently
# capturing. CUDA-graph capture forbids device-to-host syncs, so anything a
# forward would otherwise derive from a CUDA tensor with `.item()` -- the
# sparse-attention dense-prefix phase, for instance -- must be resolved before
# capture and read back from here inside it. The runner already computes these
# values to build the graph key; this just makes them visible to the callee.
_RESOLVED_EXTRA_KEYS: contextvars.ContextVar[Optional[Dict[str, Any]]] = contextvars.ContextVar(
    "cuda_graph_resolved_extra_keys", default=None
)


def resolved_extra_key(name: str) -> Any:
    """Return the extra-key value `name` resolved for the call being captured, else None.

    None means "not inside a runner-driven call" (or the key was omitted), and
    callers must fall back to deriving the value themselves.
    """
    resolved = _RESOLVED_EXTRA_KEYS.get()
    return None if resolved is None else resolved.get(name)


@contextlib.contextmanager
def resolved_extra_keys_scope(values: Dict[str, Any]):
    """Expose `values` through `resolved_extra_key` for the enclosed block."""
    token = _RESOLVED_EXTRA_KEYS.set(dict(values))
    try:
        yield
    finally:
        _RESOLVED_EXTRA_KEYS.reset(token)


@dataclass(frozen=True)
class _TensorKeyName:
    """Keep callback names distinct from tensor keyword names in graph keys."""

    name: str


# Tensor shape names and registered callback names use separate namespaces.
KeyPart: TypeAlias = Tuple[str | _TensorKeyName, Hashable]
# Full CUDA graph cache key, stored as a tuple so it can be used as a dict key.
KeyType: TypeAlias = Tuple[KeyPart, ...]


class ExtraKeyFn(Protocol):
    """Computes one optional non-shape graph-key value from model.forward inputs."""

    def __call__(self, *args: Any, **kwargs: Any) -> Optional[Hashable]: ...


@dataclass
class CUDAGraphRunnerConfig:
    """Configuration for the CUDAGraphRunner."""

    use_cuda_graph: bool
    """
    Master switch. Same three-path semantics as the LLM runner:

    1. False  → runner is dormant, maybe_get_cuda_graph always returns None
    2. True, ineligible → fallback to eager (e.g., first call with new shapes)
    3. True, eligible   → capture (if new) or replay
    """
    cuda_graph_mem_pool: Any = None


class SharedGraphPool:
    """Mutable container for sharing a CUDA graph memory pool across multiple runners.

    When multiple CUDAGraphRunners wrap different models that never execute
    concurrently (e.g., WAN 2.2 high-noise / low-noise transformers), sharing
    a pool lets CUDA alias memory between their graphs, avoiding duplication.
    """

    def __init__(self):
        self.handle = None


class CUDAGraphRunner:
    """
    Manages CUDA graph lifecycle for visual generation.

    Mirrors the LLM CUDAGraphRunner API:
      get_graph_key → maybe_get_cuda_graph → needs_capture → capture / replay

    Key differences from the LLM runner:
    - Keys are derived from model-forward tensor shapes plus optional explicit
      graph-visible state
    - Static buffers are per-key (allocated at capture time), not shared/pre-allocated
    - No batch padding, speculative decoding, or cross-rank coordination
    - Outputs returned via make_weak_ref (zero-copy, same as LLM runner)
    """

    WARMUP_STEPS = 2
    # Run gc.collect() and torch.cuda.empty_cache() after each eager warmup call,
    # so a large captured callable starts capture with the caching allocator trimmed.
    RELEASE_MEMORY_AFTER_WARMUP = True

    def __init__(
        self,
        config: CUDAGraphRunnerConfig,
        shared_pool: SharedGraphPool = None,
    ):
        self.config = config
        self.enabled = config.use_cuda_graph
        # When False, uncaptured keys run eagerly instead of triggering capture.
        self.allow_capture = True
        self._shared_pool = shared_pool
        self._extra_key_fns: Dict[str, ExtraKeyFn] = {}
        self._extra_key_capture_limits: Dict[str, int] = {}

        self.graphs: Dict[KeyType, torch.cuda.CUDAGraph] = {}
        self.graph_outputs: Dict[KeyType, Any] = {}  # weak refs
        # Keep the capture-created output tensors alive. Some pipelines keep
        # outputs from multiple graph keys live at once, for example true CFG
        # combines positive and negative Qwen-Image-Edit transformer outputs.
        # Without strong refs, a later graph capture using the same memory pool
        # can reuse the earlier output storage and make those outputs alias.
        self._graph_output_refs: Dict[KeyType, Any] = {}
        self.static_inputs: Dict[KeyType, Tuple[List[Any], Dict[str, Any]]] = {}
        self.memory_pool = config.cuda_graph_mem_pool

    def register_extra_key_fn(
        self, name: str, fn: ExtraKeyFn, *, max_graphs: Optional[int] = None
    ) -> None:
        """Register a graph-key extension computed from forward args/kwargs.

        The function is called with only the wrapped model.forward inputs.
        If max_graphs is set, at most that many graphs carrying this key part
        are retained. Uncached calls beyond the limit run eagerly; existing
        graphs still replay. Calls for which fn returns None do not count.
        """
        if self.graphs:
            raise RuntimeError(
                "CUDA graph extra key functions must be registered before graph capture."
            )
        if max_graphs is not None and max_graphs < 1:
            raise ValueError("CUDA graph capture limits must be positive")
        self._extra_key_fns[name] = fn
        if max_graphs is None:
            self._extra_key_capture_limits.pop(name, None)
        else:
            self._extra_key_capture_limits[name] = max_graphs

    def _within_capture_limits(self, key: KeyType) -> bool:
        # Keep captured outputs alive: callers may still hold a CFG branch's
        # weak output reference while invoking the other branch with a new key.
        names = {name for name, _ in key}
        for name, limit in self._extra_key_capture_limits.items():
            if (
                name in names
                and sum(any(part_name == name for part_name, _ in cached) for cached in self.graphs)
                >= limit
            ):
                return False
        return True

    def _get_tensor_shape_key(self, *args, **kwargs) -> KeyType:
        # Base VisualGen graphs are keyed by tensor shapes from the wrapped
        # model.forward call.
        return tuple(
            list(
                (_TensorKeyName(f"arg{i}"), tuple(arg.shape))
                for i, arg in enumerate(args)
                if isinstance(arg, torch.Tensor)
            )
            + list(
                (_TensorKeyName(k), tuple(kwargs[k].shape))
                for k in sorted(kwargs.keys())
                if isinstance(kwargs[k], torch.Tensor)
            )
        )

    def _get_extra_key(self, *args, **kwargs) -> KeyType:
        parts = []
        for name, fn in sorted(self._extra_key_fns.items()):
            value = fn(*args, **kwargs)
            if value is not None:
                parts.append((name, value))
        # Remembered so `capture()` can republish them to the callee; this runs
        # outside capture, where the `.item()` these callbacks need is legal.
        self._last_resolved_extra_keys = dict(parts)
        return tuple(parts)

    def get_graph_key(self, *args, **kwargs) -> KeyType:
        return self._get_tensor_shape_key(*args, **kwargs) + self._get_extra_key(*args, **kwargs)

    def _get_pool(self):
        """Return the best available pool: shared first, then own, then None."""
        if self._shared_pool is not None and self._shared_pool.handle is not None:
            return self._shared_pool.handle
        return self.memory_pool

    def _log_capture(self, key: KeyType) -> None:
        logger.info(f"Capturing graph for key: {key}")

    def capture(
        self, key: KeyType, fn: Callable, args: Tuple[Any, ...], kwargs: Dict[str, Any]
    ) -> None:
        self._log_capture(key)

        # One time clone of inputs
        static_args = [arg.clone() if isinstance(arg, torch.Tensor) else arg for arg in args]
        static_kwargs = {
            k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in kwargs.items()
        }

        graph = torch.cuda.CUDAGraph()
        # Warmup and capture both see the host-resolved extra keys, so a
        # forward that needs the dense-prefix phase reads it from here instead
        # of syncing a CUDA tensor -- which capture would reject.
        with resolved_extra_keys_scope(getattr(self, "_last_resolved_extra_keys", {})):
            for _ in range(self.WARMUP_STEPS):
                with discard_nccl_window_tensor_outputs((static_args, static_kwargs)):
                    fn(*static_args, **static_kwargs)
                torch.cuda.synchronize()
                if self.RELEASE_MEMORY_AFTER_WARMUP:
                    gc.collect()
                    torch.cuda.empty_cache()

            graph_pool = self._get_pool() or torch.cuda.graph_pool_handle()
            with nccl_window_graph_capture(graph, graph_pool):
                output = fn(*static_args, **static_kwargs)

        self.graphs[key] = graph
        self.static_inputs[key] = (static_args, static_kwargs)
        self._graph_output_refs[key] = output
        self.graph_outputs[key] = make_weak_ref(output)
        self.memory_pool = graph_pool

        # Publish pool so other runners sharing the same SharedGraphPool can reuse it
        if self._shared_pool is not None and self._shared_pool.handle is None:
            self._shared_pool.handle = self.memory_pool

    def replay(
        self,
        key: KeyType,
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
    ) -> Any:
        static_args, static_kwargs = self.static_inputs[key]

        for i, arg in enumerate(args):
            if isinstance(arg, torch.Tensor):
                static_args[i].copy_(arg)
            else:
                static_args[i] = arg

        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                static_kwargs[k].copy_(v)
            else:
                static_kwargs[k] = v

        self.graphs[key].replay()
        return self.graph_outputs[key]

    def wrap(self, fn):
        """Wrap a callable with CUDA graph capture/replay.

        Returns a drop-in replacement that:
        - On first call with a new graph key: captures a graph
        - On subsequent calls with the same graph key: replays the graph
        - Falls back to eager if disabled
        """

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            if not self.enabled:
                return fn(*args, **kwargs)

            key = self.get_graph_key(*args, **kwargs)

            if key not in self.graphs:
                if not self.allow_capture or not self._within_capture_limits(key):
                    return fn(*args, **kwargs)
                self.capture(key, fn, args, kwargs)
                return self.replay(key, args, kwargs)
            else:
                return self.replay(key, args, kwargs)

        return wrapper

    def clear(self):
        """Releases all captured graphs and the associated memory pool."""
        if not self.graphs:
            return
        for graph in self.graphs.values():
            graph.reset()
        self.graphs.clear()
        self.graph_outputs.clear()
        self._graph_output_refs.clear()
        self.static_inputs.clear()
        self.memory_pool = None


class ModelForwardExtraKeys:
    """Model-registered graph-key functions, evaluated against ``model.forward`` inputs.

    ``register_cuda_graph_extra_key_fns`` callbacks are written for a runner that
    wraps ``model.forward``; ``sparse_attn_phase``, for example, reads its
    ``timestep`` kwarg. A block runner only sees block inputs. This class stands
    in for a runner when the model registers its callbacks, records the inputs
    of the ``model.forward`` call in progress (:meth:`record_inputs`), and hands
    block runners callbacks that read those inputs (:meth:`block_key_fn`), so
    block graphs are keyed the same way a whole-forward graph would be.
    """

    def __init__(self) -> None:
        self.key_fns: Dict[str, ExtraKeyFn] = {}
        self._inputs: contextvars.ContextVar[Optional[Tuple[Tuple[Any, ...], Dict[str, Any]]]] = (
            contextvars.ContextVar("model_forward_inputs", default=None)
        )

    def register_extra_key_fn(self, name: str, fn: ExtraKeyFn) -> None:
        self.key_fns[name] = fn

    def record_inputs(self, forward: Callable) -> Callable:
        """Wrap ``forward`` so key callbacks can read the inputs of the current call."""

        @functools.wraps(forward)
        def wrapper(*args, **kwargs):
            token = self._inputs.set((args, kwargs))
            try:
                return forward(*args, **kwargs)
            finally:
                self._inputs.reset(token)

        return wrapper

    def block_key_fn(self, fn: ExtraKeyFn) -> ExtraKeyFn:
        """Adapt ``fn`` for a block runner: ignore block inputs, use ``model.forward`` inputs."""

        def key_fn(*_block_args: Any, **_block_kwargs: Any) -> Optional[Hashable]:
            inputs = self._inputs.get()
            if inputs is None:
                return None
            args, kwargs = inputs
            return fn(*args, **kwargs)

        return key_fn


class BlockCUDAGraphRunner(CUDAGraphRunner):
    """CUDA graph runner for one transformer block called from eager Python.

    Used when a step cache decides inside ``model.forward`` which blocks run
    (Cache-DiT), so the whole forward cannot be a single graph. Each block call
    is normalized before the base ``capture`` / ``replay``:

    - Arguments are bound to the block's ``forward`` signature, so positional
      and keyword calls with the same inputs share one graph.
    - Tensors nested in containers, such as FLUX's ``image_rotary_emb`` tuple,
      become top-level graph inputs, so they are keyed and copied on replay.
      The base runner only handles top-level tensors.
    - Tensor dtypes and all non-tensor values are part of the key, because the
      graph bakes them in. A call with an unhashable non-tensor value runs
      eagerly.

    A transformer has dozens of blocks, so the per-warmup ``gc.collect()`` and
    ``empty_cache()`` of the base runner would run hundreds of times at startup
    for little gain: one block's temporaries are small.
    """

    RELEASE_MEMORY_AFTER_WARMUP = False

    def __init__(
        self,
        config: CUDAGraphRunnerConfig,
        signature: inspect.Signature,
        shared_pool: Optional[SharedGraphPool] = None,
        name: str = "",
    ):
        super().__init__(config, shared_pool)
        self._signature = signature
        self.name = name

    def _log_capture(self, key: KeyType) -> None:
        logger.debug(f"{self.name}: capturing graph for key: {key}")

    def wrap(self, fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            if not self.enabled:
                return fn(*args, **kwargs)
            try:
                arguments = dict(self._signature.bind(*args, **kwargs).arguments)
            except TypeError:
                # Let the block report the bad call itself.
                return fn(*args, **kwargs)

            leaves_with_paths, spec = tree_flatten_with_path(arguments)
            leaves = [leaf for _, leaf in leaves_with_paths]
            tensor_positions = [
                i for i, leaf in enumerate(leaves) if isinstance(leaf, torch.Tensor)
            ]
            tensors = tuple(leaves[i] for i in tensor_positions)
            tensor_key = tuple(
                (keystr(leaves_with_paths[i][0]), tuple(leaves[i].shape), str(leaves[i].dtype))
                for i in tensor_positions
            )
            value_key = tuple(
                (keystr(path), leaf)
                for path, leaf in leaves_with_paths
                if not isinstance(leaf, torch.Tensor)
            )
            key = (
                ("tensors", tensor_key),
                ("values", value_key),
                ("structure", str(spec)),
            ) + self._get_extra_key(*args, **kwargs)
            try:
                hash(key)
            except TypeError:
                return fn(*args, **kwargs)

            if key not in self.graphs:
                if not self.allow_capture:
                    return fn(*args, **kwargs)

                def call_with_tensors(*graph_tensors: torch.Tensor) -> Any:
                    rebuilt = list(leaves)
                    for position, tensor in zip(tensor_positions, graph_tensors):
                        rebuilt[position] = tensor
                    bound = inspect.BoundArguments(self._signature, tree_unflatten(rebuilt, spec))
                    return fn(*bound.args, **bound.kwargs)

                self.capture(key, call_with_tensors, tensors, {})
            return self.replay(key, tensors, {})

        return wrapper


def wrap_blocks_with_cuda_graphs(
    model: nn.Module,
    block_attrs: Sequence[str],
    shared_pool: Optional[SharedGraphPool] = None,
    name: str = "transformer",
) -> Dict[str, BlockCUDAGraphRunner]:
    """Capture every block in ``model``'s block lists as its own CUDA graph.

    Each block's ``forward`` is wrapped in place, so block identity, module
    names, and references other code already holds (Cache-DiT's block lists)
    are unchanged. Call after ``torch.compile`` has replaced the block lists:
    capturing an eager block that is later compiled would make dynamo trace the
    graph wrapper. Model-registered extra keys are evaluated against the inputs
    of the enclosing ``model.forward`` call (see :class:`ModelForwardExtraKeys`).

    Returns the runners keyed by ``"{name}.{block_attr}.{index}"``.
    """
    extra_keys = ModelForwardExtraKeys()
    register_extra_keys = getattr(model, "register_cuda_graph_extra_key_fns", None)
    if register_extra_keys is not None:
        register_extra_keys(extra_keys)
    if extra_keys.key_fns:
        model.forward = extra_keys.record_inputs(model.forward)

    runners: Dict[str, BlockCUDAGraphRunner] = {}
    for block_attr in block_attrs:
        for index, block in enumerate(getattr(model, block_attr)):
            runner_name = f"{name}.{block_attr}.{index}"
            # torch.compile wrappers expose (*args, **kwargs); bind against the eager block.
            eager_block = getattr(block, "_orig_mod", block)
            runner = BlockCUDAGraphRunner(
                CUDAGraphRunnerConfig(use_cuda_graph=True),
                inspect.signature(eager_block.forward),
                shared_pool,
                name=runner_name,
            )
            for key_name, key_fn in extra_keys.key_fns.items():
                runner.register_extra_key_fn(key_name, extra_keys.block_key_fn(key_fn))
            block.forward = runner.wrap(block.forward)
            runners[runner_name] = runner
    return runners
