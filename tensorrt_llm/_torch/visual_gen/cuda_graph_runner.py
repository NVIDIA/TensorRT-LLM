# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import contextlib
import contextvars
import functools
import gc
import inspect
import weakref
from dataclasses import dataclass, field
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
from torch.utils._pytree import keystr, tree_flatten_with_path, tree_leaves, tree_unflatten

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

    def capture(
        self, key: KeyType, fn: Callable, args: Tuple[Any, ...], kwargs: Dict[str, Any]
    ) -> None:
        logger.info(f"Capturing graph for key: {key}")

        # One time clone of inputs
        static_args = [arg.clone() if isinstance(arg, torch.Tensor) else arg for arg in args]
        static_kwargs = {
            k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in kwargs.items()
        }
        self._capture_static(key, fn, static_args, static_kwargs, self.WARMUP_STEPS)

    def _capture_static(
        self,
        key: KeyType,
        fn: Callable,
        static_args: List[Any],
        static_kwargs: Dict[str, Any],
        warmup_steps: int,
        release_memory: bool = True,
    ) -> None:
        """Warm up ``fn`` on the static inputs, capture it, and record the graph.

        ``release_memory`` runs gc.collect() + empty_cache() after each warmup call.
        """
        graph = torch.cuda.CUDAGraph()
        # Warmup and capture both see the host-resolved extra keys, so a
        # forward that needs the dense-prefix phase reads it from here instead
        # of syncing a CUDA tensor -- which capture would reject.
        with resolved_extra_keys_scope(getattr(self, "_last_resolved_extra_keys", {})):
            for _ in range(warmup_steps):
                with discard_nccl_window_tensor_outputs((static_args, static_kwargs)):
                    fn(*static_args, **static_kwargs)
                torch.cuda.synchronize()
                if release_memory:
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


@dataclass
class _ForwardCall:
    """One ``model.forward`` call, as its block graphs see it."""

    extra_key: KeyType = ()
    # A block graph replayed in this call; see BlockCUDAGraphRunner._get_pool.
    replayed: bool = False
    capture_pool: Optional[Tuple[int, int]] = None
    # Input buffers of the block graphs captured in this call, by source tensor (data
    # pointer and layout), so blocks that receive the same tensor share one buffer.
    input_buffers: Dict[Tuple[Any, ...], torch.Tensor] = field(default_factory=dict)


class ModelForwardContext:
    """Model-registered extra graph keys of the ``model.forward`` call in progress.

    A forward pre-hook evaluates the keys once on the call's inputs, so block graphs
    are keyed like a whole-forward graph. Hooks, not a forward wrapper, because
    Cache-DiT's disable restores ``forward`` and would drop a wrapper.
    """

    def __init__(self) -> None:
        self._key_fns: Dict[str, ExtraKeyFn] = {}
        self._current = contextvars.ContextVar("model_forward_call", default=None)
        self._tokens: List[contextvars.Token] = []

    def register_extra_key_fn(
        self, name: str, fn: ExtraKeyFn, *, max_graphs: Optional[int] = None
    ) -> None:
        """Same signature as ``CUDAGraphRunner.register_extra_key_fn``.

        ``max_graphs`` is checked but not applied: block graphs are captured only until
        warmup ends (``BasePipeline._stop_block_graph_capture``).
        """
        if max_graphs is not None and max_graphs < 1:
            raise ValueError("CUDA graph capture limits must be positive")
        self._key_fns[name] = fn

    def track(self, model: nn.Module) -> None:
        """Install the forward hooks on ``model``."""
        model.register_forward_pre_hook(self._enter_call, with_kwargs=True)
        model.register_forward_hook(self._exit_call, always_call=True)

    def _enter_call(self, _module: nn.Module, args: Tuple[Any, ...], kwargs: Dict[str, Any]):
        call = _ForwardCall()
        # Push first: _exit_call also runs if a key callback raises.
        self._tokens.append(self._current.set(call))
        keys = ((name, fn(*args, **kwargs)) for name, fn in sorted(self._key_fns.items()))
        call.extra_key = tuple(k for k in keys if k[1] is not None)

    def _exit_call(self, _module: nn.Module, _args: Tuple[Any, ...], _output: Any):
        self._current.reset(self._tokens.pop())

    def current(self) -> Optional[_ForwardCall]:
        """The call in progress; None outside ``model.forward``."""
        return self._current.get()


def _unexpanded(tensor: torch.Tensor) -> torch.Tensor:
    """``tensor`` with each broadcast (stride-0) dimension reduced to size 1."""
    shape = [1 if stride == 0 else size for size, stride in zip(tensor.shape, tensor.stride())]
    return tensor.as_strided(shape, tensor.stride())


def _copy_into(buffer: torch.Tensor, tensor: torch.Tensor) -> None:
    if buffer.is_contiguous():
        buffer.copy_(tensor)
    else:
        _unexpanded(buffer).copy_(_unexpanded(tensor))


def _probe_copy(tensor: torch.Tensor) -> torch.Tensor:
    """A copy of ``tensor`` with its strides; a broadcast input stays broadcast.

    Only the unexpanded data is stored.
    """
    unexpanded = _unexpanded(tensor)
    probe = torch.empty_strided(
        unexpanded.shape, unexpanded.stride(), dtype=tensor.dtype, device=tensor.device
    ).expand(tensor.shape)
    _copy_into(probe, tensor)
    return probe


class BlockCUDAGraphRunner(CUDAGraphRunner):
    """CUDA graph runner for one transformer block, used with Cache-DiT.

    Arguments are bound to the block's signature and nested tensors are flattened, so
    positional and keyword calls share a graph and nested tensors (FLUX rotary) are
    keyed and copied. Dtypes, non-contiguous layouts and non-tensor values are part of
    the key.

    Block outputs are not kept alive: as in eager mode, the pool reuses an output's
    memory once the caller drops it. An input that is the previous block's output is
    read in place. Every other input is copied into a static buffer on each replay; the
    blocks of one ``model.forward`` call that receive the same tensor share its buffer.
    An input the block writes in place, or one passed at two positions of the capture
    call, gets a buffer of its own.
    """

    def __init__(
        self,
        config: CUDAGraphRunnerConfig,
        signature: inspect.Signature,
        block_outputs: "weakref.WeakValueDictionary[int, torch.Tensor]",
        forward_context: ModelForwardContext,
        shared_pool: Optional[SharedGraphPool] = None,
        name: str = "",
    ):
        super().__init__(config, shared_pool)
        self._signature = signature
        self.name = name
        # Output buffers of all block graphs of the model, by data pointer.
        self._block_outputs = block_outputs
        self._forward_context = forward_context

    def _get_extra_key(self, *args, **kwargs) -> KeyType:
        call = self._forward_context.current()  # keys evaluated once per model.forward call
        key = () if call is None else call.extra_key
        self._last_resolved_extra_keys = dict(key)
        return key

    def _get_pool(self):
        call = self._forward_context.current()
        if call is None or not call.replayed:
            return super()._get_pool()
        # Outputs of graphs replayed earlier in this call are in use, but the shared pool
        # has them as free: capture into the call's own pool so they are not overwritten.
        if call.capture_pool is None:
            call.capture_pool = torch.cuda.graph_pool_handle()
        return call.capture_pool

    def capture(
        self, key: KeyType, fn: Callable, args: Tuple[Any, ...], kwargs: Dict[str, Any]
    ) -> Any:
        """Capture ``fn`` for ``key``, run the graph on ``args`` and return its output.

        The output is handed to the caller, not kept, so its memory returns to the pool
        when the caller drops it.
        """
        logger.debug(f"{self.name}: capturing graph for key: {key}")

        # The first run uses copies of the real inputs: it finds the inputs the block
        # writes, and state a block builds on its first call for a shape (VC-Attention's
        # token grouping) comes from the same data as in eager mode.
        probe = [_probe_copy(arg) for arg in args]
        with resolved_extra_keys_scope(getattr(self, "_last_resolved_extra_keys", {})):
            with discard_nccl_window_tensor_outputs((probe, {})):
                fn(*probe)
        torch.cuda.synchronize()

        call = self._forward_context.current()
        shared_buffers = {} if call is None else call.input_buffers
        static_args: List[torch.Tensor] = []
        for index, (arg, tensor) in enumerate(zip(args, probe)):
            written = not torch.equal(_unexpanded(tensor), _unexpanded(arg))
            if written:
                input_name = key[0][1][index][0]
                logger.warning(
                    f"{self.name}: block writes input {input_name} in place; "
                    "keeping a private CUDA graph buffer for it"
                )
            previous = self._block_outputs.get(arg.data_ptr())
            chained = (
                not written
                and previous is not None
                and (previous.shape, previous.stride(), previous.dtype)
                == (arg.shape, arg.stride(), arg.dtype)
                and not any(previous is static for static in static_args)
            )
            source = (arg.data_ptr(), arg.shape, arg.stride(), arg.dtype)
            shared = shared_buffers.get(source)
            if chained:  # read the previous block's output buffer in place
                static_args.append(previous)
            elif written:
                static_args.append(tensor)
            elif shared is not None and not any(shared is static for static in static_args):
                # An earlier block of this call has a buffer for this tensor; each replay
                # still copies the tensor in, so a change between the blocks is seen.
                static_args.append(shared)
            else:
                shared_buffers.setdefault(source, tensor)
                static_args.append(tensor)
        del probe

        # Dozens of small blocks: skip the per-warmup gc.collect() + empty_cache().
        self._capture_static(key, fn, static_args, {}, self.WARMUP_STEPS - 1, release_memory=False)
        output = self._graph_output_refs.pop(key)
        for leaf in tree_leaves(self.graph_outputs[key]):
            if isinstance(leaf, torch.Tensor):
                self._block_outputs[leaf.data_ptr()] = leaf
        self.replay(key, args, {})
        return output

    def replay(self, key: KeyType, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Any:
        static_args, _ = self.static_inputs[key]
        copies = [  # a chained input already is the graph's buffer
            (static, arg) for arg, static in zip(args, static_args) if not arg.is_set_to(static)
        ]
        # A source that is also another input's buffer (one block output passed twice) is read
        # before any buffer is written.
        targets = {static.data_ptr() for static, _ in copies}
        sources = [
            _unexpanded(arg).clone().expand(arg.shape) if arg.data_ptr() in targets else arg
            for _, arg in copies
        ]
        for (static, _), source in zip(copies, sources):
            _copy_into(static, source)
        self.graphs[key].replay()
        return self.graph_outputs[key]

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
                (
                    keystr(leaves_with_paths[i][0]),
                    tuple(leaves[i].shape),
                    str(leaves[i].dtype),
                    None if leaves[i].is_contiguous() else leaves[i].stride(),
                )
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

                return self.capture(key, call_with_tensors, tensors, {})
            call = self._forward_context.current()
            if call is not None:
                call.replayed = True
            return self.replay(key, tensors, {})

        return wrapper


def wrap_blocks_with_cuda_graphs(
    model: nn.Module,
    block_attrs: Sequence[str],
    shared_pool: Optional[SharedGraphPool] = None,
    name: str = "transformer",
) -> Dict[str, BlockCUDAGraphRunner]:
    """Wrap each block's ``forward`` in place with a :class:`BlockCUDAGraphRunner`.

    Call after ``torch.compile``. Block identity is unchanged, so Cache-DiT's block
    references stay valid. Returns runners keyed ``"{name}.{block_attr}.{index}"``.
    """
    forward_context = ModelForwardContext()
    register_extra_keys = getattr(model, "register_cuda_graph_extra_key_fns", None)
    if register_extra_keys is not None:
        register_extra_keys(forward_context)
    forward_context.track(model)
    if shared_pool is None:  # one pool, so graphs can reuse the memory of earlier outputs
        shared_pool = SharedGraphPool()
    # Weak, so clearing the runners releases the buffers.
    block_outputs: "weakref.WeakValueDictionary[int, torch.Tensor]" = weakref.WeakValueDictionary()

    runners: Dict[str, BlockCUDAGraphRunner] = {}
    for block_attr in block_attrs:
        for index, block in enumerate(getattr(model, block_attr)):
            runner_name = f"{name}.{block_attr}.{index}"
            # torch.compile wrappers expose (*args, **kwargs); bind against the eager block.
            eager_block = getattr(block, "_orig_mod", block)
            runner = BlockCUDAGraphRunner(
                CUDAGraphRunnerConfig(use_cuda_graph=True),
                inspect.signature(eager_block.forward),
                block_outputs,
                forward_context,
                shared_pool,
                name=runner_name,
            )
            block.forward = runner.wrap(block.forward)
            runners[runner_name] = runner
    return runners
