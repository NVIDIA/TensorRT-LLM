# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import contextlib
import contextvars
import functools
import gc
import inspect
import itertools
import weakref
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
        self._capture_static(key, fn, static_args, static_kwargs, self.WARMUP_STEPS)

    def _capture_static(
        self,
        key: KeyType,
        fn: Callable,
        static_args: List[Any],
        static_kwargs: Dict[str, Any],
        warmup_steps: int,
    ) -> None:
        """Warm up ``fn`` on the static inputs, capture it, and record the graph."""
        graph = torch.cuda.CUDAGraph()
        # Warmup and capture both see the host-resolved extra keys, so a
        # forward that needs the dense-prefix phase reads it from here instead
        # of syncing a CUDA tensor -- which capture would reject.
        with resolved_extra_keys_scope(getattr(self, "_last_resolved_extra_keys", {})):
            for _ in range(warmup_steps):
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


class ModelForwardContext:
    """Tracks the ``model.forward`` call in progress for the model's block runners.

    Block runners only see block inputs. :meth:`track` installs forward hooks on
    the model that tell them two things about the enclosing call:

    - Its inputs. ``register_cuda_graph_extra_key_fns`` callbacks are written for
      a runner that wraps ``model.forward`` (``sparse_attn_phase``, for example,
      reads its ``timestep`` kwarg). This object stands in for that runner when
      the model registers them, and :meth:`block_key_fn` evaluates them against
      the recorded inputs, so block graphs are keyed like a whole-forward graph.
    - A call id, so a buffer shared by the blocks of one call is filled once per
      call (see :class:`BlockGraphBuffers`).

    Hooks rather than a ``model.forward`` wrapper: Cache-DiT replaces
    ``model.forward`` while enabled and restores the saved attribute as a method
    when disabled, which would break an instance-level wrapper.
    """

    def __init__(self) -> None:
        self.key_fns: Dict[str, ExtraKeyFn] = {}
        self._call_ids = itertools.count(1)
        self._current: contextvars.ContextVar[
            Optional[Tuple[int, Tuple[Any, ...], Dict[str, Any]]]
        ] = contextvars.ContextVar("model_forward_call", default=None)
        self._tokens: List[contextvars.Token] = []

    def register_extra_key_fn(self, name: str, fn: ExtraKeyFn) -> None:
        self.key_fns[name] = fn

    def track(self, model: nn.Module) -> None:
        """Make the inputs and id of ``model``'s forward call in progress visible."""
        model.register_forward_pre_hook(self._enter_call, with_kwargs=True)
        model.register_forward_hook(self._exit_call, always_call=True)

    def _enter_call(self, _module: nn.Module, args: Tuple[Any, ...], kwargs: Dict[str, Any]):
        self._tokens.append(self._current.set((next(self._call_ids), args, kwargs)))

    def _exit_call(self, _module: nn.Module, _args: Tuple[Any, ...], _output: Any):
        self._current.reset(self._tokens.pop())

    def call_id(self) -> Optional[int]:
        """Id of the ``model.forward`` call in progress, or None outside one."""
        current = self._current.get()
        return None if current is None else current[0]

    def block_key_fn(self, fn: ExtraKeyFn) -> ExtraKeyFn:
        """Adapt ``fn`` for a block runner: ignore block inputs, use ``model.forward`` inputs."""

        def key_fn(*_block_args: Any, **_block_kwargs: Any) -> Optional[Hashable]:
            current = self._current.get()
            if current is None:
                return None
            _, args, kwargs = current
            return fn(*args, **kwargs)

        return key_fn


class BlockGraphBuffers:
    """Static input buffers shared by the block graphs of one model.

    Most block inputs already sit in a buffer some graph reads, so a private
    copy per block wastes a device copy per input per replay and the memory:

    - **Chained**: an input that is another block graph's output buffer (the
      previous block's output) is read in place. A different tensor arriving at
      that input is copied into it, overwriting the earlier block's output
      buffer, whose value has been consumed by then.
    - **Shared**: inputs that the blocks of one ``model.forward`` call receive as
      the same tensor (``temb``, the rotary embedding) use one buffer, filled by
      the first block of each call.
    """

    def __init__(self, forward_context: ModelForwardContext) -> None:
        self._forward_context = forward_context
        self._graph_outputs: Dict[int, torch.Tensor] = {}
        # Capture time: id(source) -> (call id, source, shared buffer).
        self._shared: Dict[int, Tuple[int, weakref.ref, torch.Tensor]] = {}
        # id(shared buffer) -> (call id, source) whose contents it holds.
        self._filled: Dict[int, Tuple[int, weakref.ref]] = {}

    def register_outputs(self, outputs: Any) -> None:
        for leaf in tree_leaves(outputs):
            if isinstance(leaf, torch.Tensor):
                self._graph_outputs[id(leaf)] = leaf

    def graph_output(self, tensor: torch.Tensor) -> Optional[torch.Tensor]:
        """``tensor`` itself if it is a registered graph output buffer, else None."""
        found = self._graph_outputs.get(id(tensor))
        return found if found is tensor else None

    def shared_buffer(self, source: torch.Tensor) -> torch.Tensor:
        """A buffer holding ``source``; blocks of the same call get the same buffer."""
        call = self._forward_context.call_id()
        entry = self._shared.get(id(source))
        if call is not None and entry is not None and entry[0] == call and entry[1]() is source:
            return entry[2]
        buffer = source.clone()
        if call is not None:
            self._shared[id(source)] = (call, weakref.ref(source), buffer)
            self._filled[id(buffer)] = (call, weakref.ref(source))
        return buffer

    def needs_copy(self, buffer: torch.Tensor, source: torch.Tensor) -> bool:
        """Whether ``source`` must be copied into shared ``buffer`` for the current call.

        Records the fill when it returns True. Outside a ``model.forward`` call
        there is no call to fill once per, so it always returns True.
        """
        call = self._forward_context.call_id()
        filled = self._filled.get(id(buffer))
        if call is not None and filled is not None and filled[0] == call and filled[1]() is source:
            return False
        if call is None:
            self._filled.pop(id(buffer), None)
        else:
            self._filled[id(buffer)] = (call, weakref.ref(source))
        return True


# How a block graph's static input buffer relates to the tensor passed in.
_PRIVATE = "private"  # owned by this graph; copied into on every replay
_CHAINED = "chained"  # another block graph's output buffer
_SHARED = "shared"  # shared by the blocks of one model.forward call; filled once per call


def _is_same_tensor(a: torch.Tensor, b: torch.Tensor) -> bool:
    return a is b or (
        a.data_ptr() == b.data_ptr()
        and a.dtype == b.dtype
        and a.shape == b.shape
        and a.stride() == b.stride()
    )


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

    With ``buffers``, a graph reads chained and shared inputs in place (see
    :class:`BlockGraphBuffers`) instead of copying every input on every replay.
    At capture, the first eager run uses private copies: an input the block
    writes in place keeps a private buffer, because a chained buffer is also the
    previous block's output and a shared buffer is read by later blocks.

    A transformer has dozens of blocks, so the per-warmup ``gc.collect()`` and
    ``empty_cache()`` of the base runner would run hundreds of times at startup
    for little gain: one block's temporaries are small.
    """

    RELEASE_MEMORY_AFTER_WARMUP = False
    # Read chained and shared inputs in place. False gives every graph private
    # input buffers, copied on every replay (debugging aid).
    CHAIN_BUFFERS = True

    def __init__(
        self,
        config: CUDAGraphRunnerConfig,
        signature: inspect.Signature,
        shared_pool: Optional[SharedGraphPool] = None,
        name: str = "",
        buffers: Optional[BlockGraphBuffers] = None,
    ):
        super().__init__(config, shared_pool)
        self._signature = signature
        self.name = name
        self._buffers = buffers
        self._input_roles: Dict[KeyType, List[str]] = {}
        # Input copies made by the last replay; for tests and debugging.
        self.last_replay_copies = 0

    def _log_capture(self, key: KeyType) -> None:
        logger.debug(f"{self.name}: capturing graph for key: {key}")

    def capture(
        self, key: KeyType, fn: Callable, args: Tuple[Any, ...], kwargs: Dict[str, Any]
    ) -> None:
        if not (self.CHAIN_BUFFERS and self._buffers is not None):
            super().capture(key, fn, args, kwargs)
            return
        self._log_capture(key)

        private = [arg.clone() for arg in args]
        before = [tensor.clone() for tensor in private]
        with resolved_extra_keys_scope(getattr(self, "_last_resolved_extra_keys", {})):
            with discard_nccl_window_tensor_outputs((private, {})):
                fn(*private)
        torch.cuda.synchronize()

        static_args: List[torch.Tensor] = []
        roles: List[str] = []
        for index, (arg, tensor, original) in enumerate(zip(args, private, before)):
            if not torch.equal(tensor, original):
                input_name = key[0][1][index][0]
                logger.warning(
                    f"{self.name}: block writes input {input_name} in place; "
                    "keeping a private CUDA graph buffer for it"
                )
                static_args.append(tensor)
                roles.append(_PRIVATE)
            elif (output := self._buffers.graph_output(arg)) is not None:
                static_args.append(output)
                roles.append(_CHAINED)
            else:
                static_args.append(self._buffers.shared_buffer(arg))
                roles.append(_SHARED)
        del private, before

        self._capture_static(key, fn, static_args, {}, self.WARMUP_STEPS - 1)
        self._input_roles[key] = roles
        self._buffers.register_outputs(self.graph_outputs[key])

    def replay(self, key: KeyType, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Any:
        roles = self._input_roles.get(key)
        if roles is None:
            self.last_replay_copies = len(args)
            return super().replay(key, args, kwargs)

        static_args, _ = self.static_inputs[key]
        copies = 0
        for arg, static, role in zip(args, static_args, roles):
            if _is_same_tensor(arg, static):
                continue
            if role == _SHARED and not self._buffers.needs_copy(static, arg):
                continue
            static.copy_(arg)
            copies += 1
        self.last_replay_copies = copies
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
    graph wrapper. ``model``'s forward calls are tracked with hooks:
    model-registered extra keys are evaluated against the call's inputs, and
    inputs shared by its blocks are copied once per call (see
    :class:`ModelForwardContext` and :class:`BlockGraphBuffers`).

    Returns the runners keyed by ``"{name}.{block_attr}.{index}"``.
    """
    forward_context = ModelForwardContext()
    register_extra_keys = getattr(model, "register_cuda_graph_extra_key_fns", None)
    if register_extra_keys is not None:
        register_extra_keys(forward_context)
    forward_context.track(model)
    buffers = BlockGraphBuffers(forward_context)

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
                buffers=buffers,
            )
            for key_name, key_fn in forward_context.key_fns.items():
                runner.register_extra_key_fn(key_name, forward_context.block_key_fn(key_fn))
            block.forward = runner.wrap(block.forward)
            runners[runner_name] = runner
    return runners
