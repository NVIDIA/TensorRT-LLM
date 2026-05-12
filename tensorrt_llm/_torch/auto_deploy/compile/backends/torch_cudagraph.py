"""Compile backend with cudagraph.

1. Monolithic CUDA graph: captures entire model as one graph for decode-only.
2. Piecewise CUDA graph: splits model at dynamic ops, captures static segments
   individually. Used for prefill/mixed batches when piecewise_enabled=True.
3. Per-N-layers CUDA graph: splits the decode-only graph at every Nth fused
   allreduce-residual-rmsnorm op (one-per-half-layer for transformer blocks),
   captures one CUDA graph per segment.  Each ``cudaGraphLaunch`` boundary
   acts as an implicit cross-rank re-sync, which prevents per-layer skew on
   spin-wait collectives (e.g., lamport one-shot AR) from accumulating across
   the full iteration.  Selectable via ``AD_LAYERS_PER_CHUNK`` env var or the
   ``cuda_graph_config.layers_per_chunk`` YAML field.

When piecewise_enabled=True, a DualModeCapturedGraph is returned that dispatches:
  - Decode-only batches → monolithic CapturedGraph (fastest, single graph replay)
  - Prefill/mixed batches → PiecewiseCapturedGraph (per-segment replay + eager dynamic ops)
"""

import copy  # noqa: I001
import gc
import os
from collections import abc
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import torch
import torch.nn as nn
from torch.cuda import CUDAGraph
from torch.fx import GraphModule, Node
from torch.fx._pytree import tree_flatten_spec
from torch.fx.passes.split_module import split_module
from torch.utils._pytree import PyTree, TreeSpec, tree_flatten

try:
    from tensorrt_llm._torch.autotuner import autotune
except ModuleNotFoundError:
    from contextlib import contextmanager

    @contextmanager
    def autotune(*args, **kwargs):
        yield  # no-op in standalone mode


from ...utils.cuda_graph import CudaGraphWarmUpPhase
from ...utils.logger import ad_logger
from ..compiler import CompileBackendRegistry, CompilerBackend, GetArgsKwargsForBatchSize
from ..piecewise_runner import ADPiecewiseRunner, DynamicOpWrapper, MetadataWrapper, OutputInfo
from ..piecewise_utils import (
    SplitInfo,
    is_dynamic_cached_op,
    is_metadata_prep,
    needs_out_buffer,
    split_graph_at_dynamic_ops,
    submod_has_cuda_ops,
)


def _args_kwargs_flatten_spec(in_spec: TreeSpec, *args, **kwargs) -> List[Any]:
    """Flatten inputs according to provided in_spec."""
    all_args: PyTree = (args, kwargs)
    return tree_flatten_spec(all_args, in_spec)


def _args_kwargs_flatten(*args, **kwargs) -> Tuple[List[Any], TreeSpec]:
    """Flatten inputs and return flattened inputs together with the TreeSpec."""
    all_args: PyTree = (args, kwargs)
    return tree_flatten(all_args)


def _coalesce_output(
    op_result: Optional[torch.Tensor], out: Optional[torch.Tensor]
) -> torch.Tensor:
    """Pick the pre-allocated ``out`` buffer when available, else the op's return."""
    return out if out is not None else op_result


def _inject_out_param(submod: GraphModule) -> None:
    """Rewrite a dynamic submodule's FX graph to accept and forward an ``out`` kwarg.

    Adds an ``out`` placeholder (default ``None``) and wires it as a kwarg to
    the dynamic custom-op ``call_function`` node.  Because custom ops must not
    return a tensor that aliases an input, the op returns a 0-element dummy
    tensor when ``out`` is provided.  A ``_coalesce_output`` node is inserted
    after the op call to select ``out`` (the mutated buffer) over the dummy.
    """
    graph = submod.graph

    last_placeholder = None
    for node in graph.nodes:
        if node.op == "placeholder":
            last_placeholder = node
        else:
            break

    assert last_placeholder is not None, (
        "_inject_out_param: no placeholder nodes found — dynamic partition should have inputs"
    )

    target_node = None
    for node in graph.nodes:
        if node.op == "call_function" and is_dynamic_cached_op(node):
            target_node = node
            break

    assert target_node is not None, (
        "_inject_out_param: no dynamic cached op found — partition was classified as dynamic"
    )

    with graph.inserting_after(last_placeholder):
        out_placeholder = graph.placeholder("out", default_value=None)

    # If the target op has `out` in the *middle* of its schema (e.g.
    # trtllm_attention_mha_with_cache: out_scale, out, rotary_cos_sin, ...),
    # the cached-attn insertion in transform/library/kvcache.py will have
    # passed `None` for `out` positionally to keep the positional ordering of
    # the params after it. Setting `out=out_placeholder` as a kwarg on top of
    # that produces a duplicate binding (PyTorch reports "received N+1
    # arguments"). Convert any positional args at/after the schema `out`
    # position into kwargs, then bind `out` as a kwarg.
    schema_args = target_node.target._schema.arguments
    schema_names = [a.name for a in schema_args]
    try:
        out_idx = schema_names.index("out")
    except ValueError as e:
        raise RuntimeError(
            f"_inject_out_param: dynamic cached op {target_node.target} has no "
            "'out' parameter in its schema; cannot wire pre-allocated buffer."
        ) from e

    new_args = list(target_node.args)
    new_kwargs = dict(target_node.kwargs)
    if len(new_args) > out_idx:
        for i in range(out_idx, len(new_args)):
            name = schema_names[i]
            if name == "out":
                # Will be set explicitly below; drop the positional None.
                continue
            # Don't overwrite an existing kwarg if for some reason it's already set.
            new_kwargs.setdefault(name, new_args[i])
        new_args = new_args[:out_idx]

    new_kwargs["out"] = out_placeholder
    target_node.args = tuple(new_args)
    target_node.kwargs = new_kwargs

    with graph.inserting_after(target_node):
        coalesce_node = graph.call_function(_coalesce_output, args=(target_node, out_placeholder))

    for user in list(target_node.users):
        if user is not coalesce_node:
            user.replace_input_with(target_node, coalesce_node)

    # Validate graph structure, then regenerate forward() from the modified graph
    graph.lint()
    submod.recompile()


class CapturedGraph(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        num_batched_inputs: Optional[int] = None,  # number of batched, dynamic inputs...
        dynamic_dims: Optional[List[int]] = None,
    ):
        super().__init__()
        self.model = model
        self.num_batched_inputs = num_batched_inputs if num_batched_inputs is not None else 1
        self.dynamic_dims = dynamic_dims or [0] * self.num_batched_inputs
        assert len(self.dynamic_dims) == self.num_batched_inputs
        self.cudagraphs: Dict[Tuple[int, ...], CUDAGraph] = {}
        self._input_buffers: List[torch.Tensor] = [
            torch.empty(0, 1) for _ in range(self.num_batched_inputs)
        ]
        self._out_buffer_flat: List[torch.Tensor] = None
        self._output_dynamic_dim: int = 0
        self._args_hash: Optional[Tuple[int, ...]] = None
        self._cuda_graph_mem_pool = None

        # store the in_spec and out_spec during graph capture
        self._in_spec = None
        self._out_spec = None

    def _get_hash(self, flat_args: List[Any]) -> Tuple[int, ...]:
        return tuple(hash(a) for a in flat_args)

    def _capture_one_graph(
        self,
        args: Tuple,
        kwargs: Dict,
        refresh_args_static: Optional[Callable] = None,
    ) -> torch.cuda.CUDAGraph:
        """Capture and return one cuda graph."""
        # warm-up and invoke autotuner
        with autotune():
            for _ in range(3):
                with CudaGraphWarmUpPhase():
                    self.model(*args, **kwargs)
                if refresh_args_static is not None:
                    # model.forward() sometimes modifies the cache_seq_interface directly
                    # Example: Eagle switches the batch from extend-only to generate-only mode during the
                    # draft loop. We want to restore the args to satisfy invariants needed by model.forward()
                    refresh_args_static()

        # capture graph now
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        od = self._output_dynamic_dim
        with torch.cuda.graph(graph, pool=self._cuda_graph_mem_pool):
            # compute output
            out = self.model(*args, **kwargs)
            # write out into output buffer up to out batch size
            out_flat = tree_flatten_spec(out, self._out_spec)
            for o_buffer, o in zip(self._out_buffer_flat, out_flat):
                o_buffer.narrow(od, 0, o.shape[od]).copy_(o)
        torch.cuda.synchronize()
        self._cuda_graph_mem_pool = self._cuda_graph_mem_pool or graph.pool()
        return graph

    def capture_graph(self, get_args_kwargs: GetArgsKwargsForBatchSize, batch_sizes: List[int]):
        """Capture and pre-fetch the graph for desired batch sizes."""
        assert not self.cudagraphs, "Graphs already captured."

        # sort batch sizes in descending order
        batch_sizes = sorted(batch_sizes, reverse=True)

        # Probe with a smaller batch size first to detect dynamic dims. Some
        # backends reuse live SequenceInfo buffers, so we must fetch the
        # max-batch inputs again afterwards to avoid leaving metadata in the
        # probed (smaller) state for the warmup/capture path.
        probe_bs = max(1, batch_sizes[0] - 1)
        args_probe, kwargs_probe = get_args_kwargs(probe_bs)
        flat_probe, _ = _args_kwargs_flatten(*args_probe, **kwargs_probe)
        probe_shapes = [
            tuple(t.shape) if isinstance(t, torch.Tensor) else None
            for t in flat_probe[: self.num_batched_inputs]
        ]

        # Re-fetch args/kwargs for the largest batch size and use those as the
        # canonical inputs for warmup and capture.
        args, kwargs = get_args_kwargs(batch_sizes[0])

        # flatten args, kwargs for the first time and record in_spec
        all_args_flat, self._in_spec = _args_kwargs_flatten(*args, **kwargs)

        # extract the batched input tensors
        args_batched = all_args_flat[: self.num_batched_inputs]
        args_static = all_args_flat[self.num_batched_inputs :]

        # Auto-detect dynamic dims by comparing the max-batch shapes against
        # the probed smaller-batch shapes.
        detected_dims = []
        for t1, probe_shape in zip(args_batched, probe_shapes):
            dim_found = 0
            if probe_shape is not None:
                for d in range(t1.ndim):
                    if t1.shape[d] != probe_shape[d]:
                        dim_found = d
                        break
            detected_dims.append(dim_found)
        self.dynamic_dims = detected_dims

        # Detect output dynamic dim from the first batched input's dynamic dim
        self._output_dynamic_dim = self.dynamic_dims[0]

        # set the args hash --> this is used to compare the static inputs during graph replay
        self._args_hash = self._get_hash(args_static)

        # store the input buffers for the largest batch size
        self._input_buffers = [a.clone() for a in args_batched]

        # create new args, kwargs with the input buffers and static args
        args, kwargs = self._in_spec.unflatten(self._input_buffers + args_static)

        # capture output once with cuda_graph_max_batch_size to capture output buffers
        # store the out_spec at this point
        with CudaGraphWarmUpPhase():
            ad_logger.info(f"Warm up with max_batch_size={batch_sizes[0]} before graph capture")
            out = self.model(*args, **kwargs)
        self._out_buffer_flat, self._out_spec = tree_flatten(out)

        # capture graph now for a range of batch sizes
        for bs in batch_sizes:
            ad_logger.info(f"Capturing graph for batch size: {bs}")

            # get new args, kwargs for the current batch size
            args, kwargs = get_args_kwargs(bs)
            all_args_flat = _args_kwargs_flatten_spec(self._in_spec, *args, **kwargs)
            args_batched = all_args_flat[: self.num_batched_inputs]
            args_static = all_args_flat[self.num_batched_inputs :]

            # assert that static args match the stored hash
            assert self._args_hash == self._get_hash(args_static), (
                "Static args mismatch during capture"
            )

            # copy new inputs to input buffers along their respective dynamic dims
            input_sizes: List[int] = []
            for i, input_tensor in enumerate(args_batched):
                dim_i = self.dynamic_dims[i]
                size_i = input_tensor.shape[dim_i]
                input_sizes.append(size_i)
                self._input_buffers[i].narrow(dim_i, 0, size_i).copy_(
                    input_tensor, non_blocking=True
                )

            # truncate input buffers along their respective dynamic dims
            inputs_truncated = [
                buf.narrow(self.dynamic_dims[i], 0, input_sizes[i])
                for i, buf in enumerate(self._input_buffers)
            ]
            args, kwargs = self._in_spec.unflatten(inputs_truncated + args_static)

            # model.forward() sometimes modifies the cache_seq_interface directly (e.g. Eagle
            # switches the batch from extend-only to generate-only mode during the draft loop).
            # Always refresh the static input buffers before graph capture so the next warmup
            # iteration sees the original layout.
            def refresh_args_static(_bs: int = bs) -> None:
                get_args_kwargs(_bs)

            # capture graph for truncated inputs
            combined_shape = sum((tuple(input.shape) for input in inputs_truncated), start=())
            self.cudagraphs[combined_shape] = self._capture_one_graph(
                args=args,
                kwargs=kwargs,
                refresh_args_static=refresh_args_static,
            )

    def forward(self, *args, **kwargs) -> Any:
        """Run the compiled graph."""
        # flatten args, kwargs
        all_args_flat = _args_kwargs_flatten_spec(self._in_spec, *args, **kwargs)

        # extract the batched input tensors
        args_batched = all_args_flat[: self.num_batched_inputs]
        args_static = all_args_flat[self.num_batched_inputs :]

        # check if args_static match the stored hash
        if self._args_hash != self._get_hash(args_static):
            return self.model(*args, **kwargs)

        # Calculate combined shape tuple as hash for cudagraph lookup
        combined_shape = sum((arg.shape for arg in args_batched), start=())

        # regular forward for non-matching shapes
        if combined_shape not in self.cudagraphs:
            return self.model(*args, **kwargs)

        # copy inputs to input buffers along their respective dynamic dims
        for i, input_tensor in enumerate(args_batched):
            dim_i = self.dynamic_dims[i]
            size_i = input_tensor.shape[dim_i]
            self._input_buffers[i].narrow(dim_i, 0, size_i).copy_(input_tensor, non_blocking=True)

        # run forward pass via graph
        self.cudagraphs[combined_shape].replay()

        # retrieve output from buffer, cut to batch size, and unflatten
        od = self._output_dynamic_dim
        bs = args_batched[0].shape[self.dynamic_dims[0]]
        out_flat = [o_b.narrow(od, 0, bs) for o_b in self._out_buffer_flat]
        return self._out_spec.unflatten(out_flat)


# ============================================================================
# PerNLayersCudaGraphWrapper helpers
# ============================================================================


# Fused (allreduce + residual + RMSNorm) op inserted by
# ``fuse_allreduce_residual_rmsnorm``.  Each transformer layer with TP > 1
# normally produces two of these: one after attention, one after MoE/MLP.
# We anchor the per-N-layers split on this op so that splits land on
# half-layer boundaries, which is exactly where cross-rank skew on the
# spin-wait one-shot AR can accumulate.
def _get_ar_boundary_op():
    """Resolve the fused AR op lazily.

    The op is registered by ``trtllm_dist.py`` when AutoDeploy's distributed
    custom ops are imported; resolving lazily avoids import-order issues.
    Returns ``None`` if the op is not registered (e.g. world_size==1 or
    AutoDeploy distributed ops not loaded).
    """
    try:
        return torch.ops.dist.trtllm_fused_allreduce_residual_rmsnorm.default
    except (AttributeError, RuntimeError):
        return None


_AR_OP_NAME_PREFIX = "dist::trtllm_fused_allreduce_residual_rmsnorm"


def _is_ar_boundary_node(node: Node) -> bool:
    """Return True if ``node`` is a fused AR-residual-rmsnorm call.

    Tolerant of both ``OpOverload`` (e.g. ``...op.default``) and
    ``OpOverloadPacket`` (the bare ``torch.ops.dist.<name>``) targets, since
    different FX tracers record one or the other.
    """
    if node.op != "call_function":
        return False
    target = node.target

    # OpOverload: target.name() returns "ns::op.overload"
    name_attr = getattr(target, "name", None)
    if callable(name_attr):
        try:
            n = name_attr()
        except Exception:
            n = None
        if isinstance(n, str) and n.startswith(_AR_OP_NAME_PREFIX):
            return True

    # OpOverloadPacket: target._qualified_op_name returns "ns::op"
    qname = getattr(target, "_qualified_op_name", None)
    if isinstance(qname, str) and qname.startswith(_AR_OP_NAME_PREFIX):
        return True

    # Identity / overloadpacket comparison fall-back.
    boundary_op = _get_ar_boundary_op()
    if boundary_op is None:
        return False
    if target is boundary_op:
        return True
    bp_packet = getattr(boundary_op, "_overloadpacket", None)
    t_packet = getattr(target, "_overloadpacket", None)
    if bp_packet is not None and (target is bp_packet or t_packet is bp_packet):
        return True
    return False


def _split_gm_at_ar_boundaries(gm: GraphModule, layers_per_chunk: int) -> Tuple[GraphModule, int]:
    """Split ``gm`` into N submodules using fused AR ops as cut points.

    For gpt-oss-120b each transformer layer contains two AR ops (post-attn,
    post-MoE).  ``layers_per_chunk=N`` therefore produces a cut every ``2*N``
    AR ops, i.e. one CUDA graph per N transformer layers.

    Returns the split GraphModule and the number of submodules that the split
    produced.  Falls back to a single-partition (no-op) split when no AR
    boundary is found, which lets the wrapper degrade gracefully if the model
    has no AR fusion (e.g. world_size==1).
    """
    # Each transformer layer for TP>1 has 2 fused AR ops (attn + MoE/MLP).
    ops_per_layer = 2
    cut_every = max(1, layers_per_chunk * ops_per_layer)

    partition_counter = [0]
    ar_seen = [0]
    node_to_partition: Dict[Node, int] = {}

    for node in gm.graph.nodes:
        if node.op in ("placeholder", "output"):
            continue
        node_to_partition[node] = partition_counter[0]
        if _is_ar_boundary_node(node):
            ar_seen[0] += 1
            # After the Nth AR boundary, advance to the next partition so that
            # subsequent ops land in the next subgraph.  The AR op itself stays
            # in the current partition (i.e. it's the LAST op of its chunk).
            if ar_seen[0] % cut_every == 0:
                partition_counter[0] += 1

    if ar_seen[0] == 0:
        # No AR ops in the graph — degrade to a single-partition split.
        return gm, 1

    def partition_fn(node: Node) -> int:
        return node_to_partition.get(node, 0)

    split_gm = split_module(gm, gm, partition_fn, keep_original_order=True)

    submod_names = [n for n, _ in split_gm.named_children() if n.startswith("submod_")]
    return split_gm, len(submod_names)


class _SubgraphRunner(nn.Module):
    """Captures-and-replays a single FX submodule as one CUDAGraph per shape key.

    Stateful: in CAPTURE phase the runner captures the wrapped submodule into
    a new ``torch.cuda.CUDAGraph`` and stores it under the current shape key
    (set by the wrapper before each capture).  In REPLAY phase the runner
    replays the graph stored under the current shape key and returns the
    captured outputs.  All runners in a chain share one ``graph_pool`` so
    tensors flowing between subgraphs keep stable addresses across replays.

    NOTE: The runner returns the *exact same* output object captured during
    capture.  Because every subgraph in a chain replays into those same
    tensors before the next one consumes them, we don't need to clone here —
    callers must not mutate outputs between subgraph boundaries.
    """

    _PHASE_WARMUP = "warmup"
    _PHASE_CAPTURE = "capture"
    _PHASE_REPLAY = "replay"

    def __init__(self, submodule: nn.Module, graph_pool: Any, name: str):
        super().__init__()
        self.submodule = submodule
        self._graph_pool = graph_pool
        self._name = name
        self._phase = self._PHASE_WARMUP
        # Per-(combined_shape) capture state.
        self._cuda_graphs: Dict[Tuple[int, ...], CUDAGraph] = {}
        self._captured_outputs: Dict[Tuple[int, ...], Any] = {}
        self._current_key: Optional[Tuple[int, ...]] = None

    def set_phase(self, phase: str) -> None:
        assert phase in (self._PHASE_WARMUP, self._PHASE_CAPTURE, self._PHASE_REPLAY), phase
        self._phase = phase

    def set_current_key(self, key: Tuple[int, ...]) -> None:
        self._current_key = key

    def has_key(self, key: Tuple[int, ...]) -> bool:
        return key in self._cuda_graphs

    def get_graph(self, key: Tuple[int, ...]) -> Optional[CUDAGraph]:
        return self._cuda_graphs.get(key)

    def get_output(self, key: Tuple[int, ...]) -> Any:
        return self._captured_outputs.get(key)

    @property
    def graph_pool(self) -> Any:
        return self._graph_pool

    def forward(self, *args, **kwargs) -> Any:
        if self._phase == self._PHASE_WARMUP:
            return self.submodule(*args, **kwargs)

        key = self._current_key
        assert key is not None, (
            f"_SubgraphRunner({self._name}): forward called with no current key."
        )

        if self._phase == self._PHASE_CAPTURE:
            graph = CUDAGraph()
            with torch.cuda.graph(graph, pool=self._graph_pool):
                output = self.submodule(*args, **kwargs)
            self._cuda_graphs[key] = graph
            self._captured_outputs[key] = output
            # If we were the first runner in the chain, publish the new pool
            # handle so subsequent runners share it.
            if self._graph_pool is None:
                self._graph_pool = graph.pool()
            return output

        # _PHASE_REPLAY
        graph = self._cuda_graphs.get(key)
        assert graph is not None, f"_SubgraphRunner({self._name}): no captured graph for key={key}."
        graph.replay()
        return self._captured_outputs[key]


class PerNLayersCapturedGraph(CapturedGraph):
    """CapturedGraph variant that splits decode capture into N-layer chunks.

    Captures the decode-only forward pass into ``ceil(num_ar_boundaries /
    (2*layers_per_chunk))`` separate CUDA graphs, all backed by a single
    shared CUDA graph memory pool.  At replay time each sub-graph is launched
    via its own ``cudaGraphLaunch``, which acts as an implicit cross-rank
    re-sync barrier — this prevents per-layer micro-skew on spin-wait
    collectives (e.g. lamport one-shot allreduce) from accumulating across
    the full 36-layer iteration.

    Mirrors the per-bucket ``cudaGraphLaunch`` pattern that the PyTorch
    backend gets "for free" by capturing one graph per (num_tokens, beam)
    bucket.
    """

    def __init__(
        self,
        model: nn.Module,
        layers_per_chunk: int,
        layers_per_chunk_source: str = "yaml",
        num_batched_inputs: Optional[int] = None,
        dynamic_dims: Optional[List[int]] = None,
    ):
        super().__init__(
            model=model,
            num_batched_inputs=num_batched_inputs,
            dynamic_dims=dynamic_dims,
        )
        assert layers_per_chunk >= 1, (
            f"PerNLayersCapturedGraph: layers_per_chunk must be >= 1, got {layers_per_chunk}"
        )
        self.layers_per_chunk = layers_per_chunk
        self.layers_per_chunk_source = layers_per_chunk_source
        # The split GraphModule replaces ``self.model`` for capture/replay.
        # We keep the original ``self.model`` reference for the very first
        # warm-up + dynamic-dim probe (CapturedGraph.capture_graph uses
        # ``self.model(...)`` before our hooks are wired up).
        self._original_model = model
        self._split_gm: Optional[GraphModule] = None
        self._runners: List[_SubgraphRunner] = []
        self._split_done = False
        self._activation_logged = False

    def _ensure_split(self) -> bool:
        """Split the FX graph and wrap each submod in a _SubgraphRunner.

        Returns True if the split produced more than one chunk (per-N-layers
        capture is active).  Returns False if no AR boundaries were found —
        in that case the caller should fall back to monolithic capture.
        """
        if self._split_done:
            return len(self._runners) > 1

        if not isinstance(self._original_model, GraphModule):
            ad_logger.warning(
                f"[PerNLayersCudaGraphWrapper] underlying model is "
                f"{type(self._original_model).__name__}, not a "
                f"GraphModule — falling back to monolithic CUDA graph capture."
            )
            self._split_done = True
            return False

        # Deep-copy the FX graph so we don't mutate the user's GraphModule;
        # otherwise the split's submod additions persist on the original.
        gm = GraphModule(self._original_model, copy.deepcopy(self._original_model.graph))
        split_gm, num_submods = _split_gm_at_ar_boundaries(gm, self.layers_per_chunk)

        if num_submods <= 1:
            ad_logger.warning(
                f"[PerNLayersCudaGraphWrapper] split produced {num_submods} "
                f"submod(s) — no AR boundaries detected.  Falling back to "
                f"monolithic capture."
            )
            self._split_done = True
            return False

        # Wrap each submod_K in a _SubgraphRunner sharing one graph pool.
        # NOTE: the FIRST runner starts with graph_pool=None; after its capture
        # it publishes the pool handle and we propagate it to the rest.
        graph_pool: Any = None
        runners: List[_SubgraphRunner] = []
        for name, child in list(split_gm.named_children()):
            if not name.startswith("submod_"):
                continue
            runner = _SubgraphRunner(child, graph_pool=graph_pool, name=name)
            runners.append(runner)
            setattr(split_gm, name, runner)

        # Sort by partition index so replay order matches FX call order.
        runners.sort(key=lambda r: int(r._name.split("_")[1]))

        self._split_gm = split_gm
        self._runners = runners
        self.model = split_gm  # capture and forward will route through split_gm
        self._split_done = True

        if not self._activation_logged:
            # Count fused-AR boundary ops in the original (pre-split) graph so
            # we can give the user an honest layer estimate.  Each transformer
            # layer with TP > 1 contributes 2 AR ops (post-attn, post-MoE).
            ar_count = sum(1 for n in self._original_model.graph.nodes if _is_ar_boundary_node(n))
            n_layers_est = ar_count // 2
            ad_logger.info(
                f"[PerNLayersCudaGraphWrapper] activated "
                f"(layers_per_chunk={self.layers_per_chunk}, "
                f"source={self.layers_per_chunk_source}); "
                f"capturing {len(runners)} sub-graphs "
                f"(~{n_layers_est}/{self.layers_per_chunk})"
            )
            self._activation_logged = True

        return True

    def _current_combined_shape(self, args: Tuple, kwargs: Dict) -> Tuple[int, ...]:
        """Compute the combined-shape key for the per-runner per-key map."""
        flat = _args_kwargs_flatten_spec(self._in_spec, *args, **kwargs)
        batched = flat[: self.num_batched_inputs]
        return sum((tuple(t.shape) for t in batched), start=())

    def _capture_one_graph(
        self,
        args: Tuple,
        kwargs: Dict,
        refresh_args_static: Optional[Callable] = None,
    ) -> torch.cuda.CUDAGraph:
        """Capture the model split into per-chunk CUDA graphs.

        Returns the FIRST sub-graph as a sentinel for the parent class; the
        full chain is stored per-runner per-shape and is replayed in
        ``forward``.  ``self.cudagraphs`` only ever sees this sentinel — the
        real replay path iterates over the runners.
        """
        # Lazy split (one-time).  If the split degenerates to <=1 chunk we
        # delegate to the parent's monolithic capture path.
        if not self._ensure_split():
            return super()._capture_one_graph(args, kwargs, refresh_args_static)

        # Compute the per-shape capture key from the (already truncated) args.
        key = self._current_combined_shape(args, kwargs)

        # Set every runner to warmup so submodules execute eagerly.
        for r in self._runners:
            r.set_phase(_SubgraphRunner._PHASE_WARMUP)

        with autotune():
            for _ in range(3):
                with CudaGraphWarmUpPhase():
                    self._split_gm(*args, **kwargs)
                if refresh_args_static is not None:
                    refresh_args_static()

        # Capture phase: each runner captures its own CUDAGraph during the
        # next forward, and shares the memory pool with subsequent runners.
        torch.cuda.synchronize()
        for r in self._runners:
            r.set_phase(_SubgraphRunner._PHASE_CAPTURE)
            r.set_current_key(key)

        od = self._output_dynamic_dim
        # Run split_gm once.  Each _SubgraphRunner intercepts its submod call
        # and captures into its own CUDAGraph under the current shape key.
        # The FIRST runner allocates and publishes the shared pool handle;
        # we then thread it through the remaining runners so all subgraphs
        # share one CUDA mempool.
        out = self._split_gm(*args, **kwargs)
        shared_pool = self._runners[0]._graph_pool
        for r in self._runners[1:]:
            if r._graph_pool is None:
                r._graph_pool = shared_pool

        # Write the final captured output into the pre-allocated output
        # buffer.  This mirrors CapturedGraph._capture_one_graph so downstream
        # replay bookkeeping keeps working.  NOTE: ``tree_flatten_spec`` would
        # error here because ``self._split_gm(*args, **kwargs)`` returns a flat
        # tuple of tensors (FX split_module rebuilds the top-level graph
        # without the dataclass-wrapping output node), whereas ``_out_spec``
        # was captured from the original (un-split) gm and encodes the model's
        # dataclass output (e.g. ``GptOssCausalLMOutput(['logits'])``).  Use
        # plain ``tree_flatten`` to extract the flat tensors; the dataclass
        # is reconstructed at the end of ``forward`` via
        # ``self._out_spec.unflatten(...)``.
        out_flat, _ = tree_flatten(out)
        for o_buffer, o in zip(self._out_buffer_flat, out_flat):
            o_buffer.narrow(od, 0, o.shape[od]).copy_(o)

        torch.cuda.synchronize()

        # All runners flip to replay phase; key stays cached for replay.
        for r in self._runners:
            r.set_phase(_SubgraphRunner._PHASE_REPLAY)

        # Publish the shared pool to the parent's bookkeeping field so any
        # later monolithic-style capture (unlikely but possible) reuses it.
        self._cuda_graph_mem_pool = self._cuda_graph_mem_pool or shared_pool

        # Return the first runner's CUDAGraph as a "primary handle" so the
        # parent's ``self.cudagraphs[combined_shape] = self._capture_one_graph(...)``
        # bookkeeping still has a CUDAGraph object to record.  ``forward``
        # below ignores it and replays the full chain instead.
        return self._runners[0].get_graph(key)

    def forward(self, *args, **kwargs) -> Any:
        """Replay the per-chunk CUDA graph chain.

        The parent ``CapturedGraph.forward`` only handles the case of one
        CUDAGraph per batch-size combo; for the per-N-layers wrapper we have
        a *chain* of graphs (one per chunk) per batch-size combo, all
        keyed by combined_shape on each runner.  Replay routes through the
        runners and copies the chain's final output into the parent output
        buffer.
        """
        # If split degenerated (e.g. world_size==1, no AR ops), use the
        # parent's monolithic replay path.
        if not self._split_done or len(self._runners) <= 1:
            return super().forward(*args, **kwargs)

        # flatten args, kwargs
        all_args_flat = _args_kwargs_flatten_spec(self._in_spec, *args, **kwargs)
        args_batched = all_args_flat[: self.num_batched_inputs]
        args_static = all_args_flat[self.num_batched_inputs :]

        # check static-args hash; on mismatch, fall back to eager (split_gm).
        if self._args_hash != self._get_hash(args_static):
            return self._split_gm(*args, **kwargs)

        combined_shape = sum((arg.shape for arg in args_batched), start=())
        if combined_shape not in self.cudagraphs:
            return self._split_gm(*args, **kwargs)

        # If any runner is missing this shape key (shouldn't happen — capture
        # populates all runners atomically — but guard anyway), fall back.
        if not all(r.has_key(combined_shape) for r in self._runners):
            return self._split_gm(*args, **kwargs)

        # copy inputs to input buffers along their respective dynamic dims
        for i, input_tensor in enumerate(args_batched):
            dim_i = self.dynamic_dims[i]
            size_i = input_tensor.shape[dim_i]
            self._input_buffers[i].narrow(dim_i, 0, size_i).copy_(input_tensor, non_blocking=True)

        # Replay each subgraph in order.  Inputs are already in the static
        # input buffers (captured-time addresses); each replay refreshes the
        # tensors flowing between subgraphs in-place.
        for r in self._runners:
            r.get_graph(combined_shape).replay()

        # The final runner's captured output IS the model output.  Copy it
        # into the pre-allocated output buffer (dynamic-dim aware) so callers
        # observe the same buffer layout the parent class promises.  NOTE:
        # the runner's captured output is whatever ``submod_K`` returns —
        # FX split_module gives a flat tuple of tensors, NOT the original
        # dataclass.  Use ``tree_flatten`` (not ``tree_flatten_spec``) here;
        # the dataclass is reconstructed below by ``self._out_spec.unflatten``.
        final_output = self._runners[-1].get_output(combined_shape)
        final_flat, _ = tree_flatten(final_output)
        od = self._output_dynamic_dim
        bs = args_batched[0].shape[self.dynamic_dims[0]]
        for o_buffer, o in zip(self._out_buffer_flat, final_flat):
            o_buffer.narrow(od, 0, o.shape[od]).copy_(o.narrow(od, 0, o.shape[od]))

        out_flat = [o_b.narrow(od, 0, bs) for o_b in self._out_buffer_flat]
        return self._out_spec.unflatten(out_flat)


def _resolve_layers_per_chunk(yaml_value: Optional[int]) -> Tuple[Optional[int], str]:
    """Resolve the effective ``layers_per_chunk`` and its source.

    Precedence (highest to lowest):
      1. ``AD_LAYERS_PER_CHUNK`` env var (integer >= 1; "0" or unset disables).
      2. ``cuda_graph_config.layers_per_chunk`` from YAML (>= 1).
      3. None (use monolithic FullCudaGraphWrapper / CapturedGraph).

    Returns ``(layers_per_chunk, source)`` where ``source`` is "env",
    "yaml", or "" when unset.
    """
    env_val = os.environ.get("AD_LAYERS_PER_CHUNK", "").strip()
    if env_val:
        try:
            n = int(env_val)
        except ValueError:
            ad_logger.warning(
                f"[PerNLayersCudaGraphWrapper] AD_LAYERS_PER_CHUNK={env_val!r} "
                f"is not an integer; ignoring."
            )
            n = 0
        if n >= 1:
            return n, "env"

    if yaml_value is not None and yaml_value >= 1:
        return int(yaml_value), "yaml"

    return None, ""


class PiecewiseCapturedGraph(nn.Module):
    """Manages piecewise CUDA graph capture/replay for prefill/mixed batches.

    The model is split at dynamic op boundaries (attention, SSM, conv, delta).
    Static segments are wrapped in ADPiecewiseRunner for CUDA graph capture.
    Non-inplace dynamic ops are wrapped in DynamicOpWrapper, which passes a
    pre-allocated output buffer (out=) from the preceding static runner's graph pool.
    Metadata-prep ops are wrapped in MetadataWrapper to keep their output
    addresses stable across capture and replay (prevents CUDA graph crashes).
    Inplace dynamic ops (see _INPLACE_DYNAMIC_OPS) run eagerly without wrapping.
    """

    def __init__(
        self,
        model: nn.Module,
        piecewise_num_tokens: Optional[List[int]] = None,
        capture_lm_head: bool = False,
        max_batch_size: Optional[int] = None,
        out_spec: Optional[TreeSpec] = None,
    ):
        super().__init__()
        self.original_model = model
        self.piecewise_num_tokens = piecewise_num_tokens or []
        self.capture_lm_head = capture_lm_head
        self.max_batch_size = max_batch_size
        self.split_info: Optional[SplitInfo] = None
        self.split_gm: Optional[GraphModule] = None
        self._is_prepared = False
        self._wrapped_dynamic_indices: Set[int] = set()
        self._static_runners: Dict[int, ADPiecewiseRunner] = {}
        # Pre-allocated static buffers for kwargs whose addresses change between
        # calls.  Allocated during warmup_and_capture, used at runtime to ensure
        # CUDA graph replay sees stable addresses.
        # Format: {kwarg_name: (static_buffer, dynamic_dim_or_none)}
        self._static_input_buffers: Dict[str, Tuple[torch.Tensor, Optional[int]]] = {}
        # Output tree spec for reconstructing structured outputs (e.g.
        # ModelOutput) from flat tuples returned by split_gm.
        self._out_spec = out_spec

    def prepare(self) -> None:
        """Split the model, wrap static segments in runners, wrap Group 3 dynamic ops."""
        if self._is_prepared:
            return

        model = self.original_model
        if not isinstance(model, GraphModule):
            ad_logger.warning(
                "PiecewiseCapturedGraph: model is not a GraphModule, "
                "piecewise CUDA graph requires an FX GraphModule. "
                "Falling back to eager execution."
            )
            self._is_prepared = True
            return

        gm = GraphModule(model, copy.deepcopy(model.graph))

        self.split_info = split_graph_at_dynamic_ops(gm)

        # When multi-stream transforms reclassify ALL static partitions as
        # dynamic (e.g. multi_stream_moe + multi_stream_mla_attn on every
        # layer), there are zero capturable static segments.  Piecewise CUDA
        # graphs are impossible — fall back to eager execution for
        # prefill/mixed batches (monolithic CG still handles decode).
        if not self.split_info.static_submod_indices:
            ad_logger.warning(
                "PiecewiseCapturedGraph: no static partitions after splitting "
                "(%d dynamic). Piecewise CUDA graphs disabled — prefill/mixed "
                "batches will run eagerly.",
                len(self.split_info.dynamic_submod_indices),
            )
            self._is_prepared = True
            return

        self.split_gm = self.split_info.split_gm

        graph_pool = torch.cuda.graph_pool_handle()
        num_wrapped_static = 0
        num_skipped_static = 0
        num_wrapped_dynamic = 0

        # Phase 1: wrap static segments in ADPiecewiseRunner
        # Track which indices got a runner so we can link dynamic ops later.
        #
        # Optionally exclude the trailing static partition (final_norm + lm_head)
        # from capture.  The lm_head produces a [num_tokens, vocab_size] output
        # whose graph-pool cost is enormous (e.g. 4-6 GiB at nt=8192 for 256K
        # vocab).  Excluding it lets that partition run eagerly, matching the
        # PyTorch backend's piecewise behaviour.
        static_indices = list(self.split_info.static_submod_indices)
        if (
            not self.capture_lm_head
            and static_indices
            and self.split_info.dynamic_submod_indices
            and static_indices[-1] > max(self.split_info.dynamic_submod_indices)
        ):
            trailing_idx = static_indices.pop()
            ad_logger.info(
                "PiecewiseCapturedGraph: excluding trailing static submod_%d "
                "(lm_head) from capture — will run eagerly",
                trailing_idx,
            )

        runner_by_idx: Dict[int, ADPiecewiseRunner] = {}
        for idx in static_indices:
            submod_name = f"submod_{idx}"
            if not hasattr(self.split_gm, submod_name):
                continue
            original_submod = getattr(self.split_gm, submod_name)

            if not submod_has_cuda_ops(original_submod):
                ad_logger.info(
                    "PiecewiseCapturedGraph: skipping %s (no CUDA ops, will run eagerly)",
                    submod_name,
                )
                num_skipped_static += 1
                continue

            runner = ADPiecewiseRunner(
                submodule=original_submod,
                graph_pool=graph_pool,
            )
            setattr(self.split_gm, submod_name, runner)
            runner_by_idx[idx] = runner
            self._static_runners[idx] = runner
            num_wrapped_static += 1

        # Phase 2: wrap dynamic ops.
        # - Metadata-prep ops → MetadataWrapper (stable output addresses)
        # - Attention/SSM/delta/logits → DynamicOpWrapper (out= pre-alloc)
        # - Inplace ops (e.g. conv) → left unwrapped (mutate input, return None)
        # Iterate over all actual submod indices in order (not range(N), because
        # indices are partition IDs that may have gaps from empty partitions).
        dynamic_set = set(self.split_info.dynamic_submod_indices)
        all_submod_indices = sorted(
            self.split_info.static_submod_indices + self.split_info.dynamic_submod_indices
        )
        current_static_runner: Optional[ADPiecewiseRunner] = None
        # Fallback runner: the first available static runner.  When
        # multi-stream transforms reclassify the initial static partition(s)
        # as dynamic (e.g. record_event_passthrough from multi_stream_mla_attn)
        # AND the static partitions between metadata-prep and attention have
        # no CUDA ops (skipped), there is no *preceding* static runner for the
        # first attention op.  In that case we fall back to the nearest
        # *following* static runner — any runner in the shared graph pool can
        # host the pre-allocated output buffer.
        fallback_runner: Optional[ADPiecewiseRunner] = None
        if runner_by_idx:
            fallback_runner = runner_by_idx[min(runner_by_idx)]
        num_metadata_wrapped = 0
        for idx in all_submod_indices:
            if idx in runner_by_idx:
                current_static_runner = runner_by_idx[idx]
            elif idx in dynamic_set:
                submod_name = f"submod_{idx}"
                if not hasattr(self.split_gm, submod_name):
                    continue
                submod = getattr(self.split_gm, submod_name)

                if not needs_out_buffer(submod):
                    if is_metadata_prep(submod):
                        wrapper = MetadataWrapper(submod, max_batch_size=self.max_batch_size)
                        setattr(self.split_gm, submod_name, wrapper)
                        num_metadata_wrapped += 1
                        ad_logger.info(
                            "PiecewiseCapturedGraph: wrapped %s in "
                            "MetadataWrapper (stable output addresses)",
                            submod_name,
                        )
                    continue

                effective_runner = current_static_runner or fallback_runner
                assert effective_runner is not None, (
                    f"Dynamic {submod_name} has no static runner available — "
                    f"cannot allocate out= buffer for stable output addresses"
                )
                if current_static_runner is None:
                    ad_logger.info(
                        "PiecewiseCapturedGraph: %s has no preceding static "
                        "runner, using fallback runner (submod_%d)",
                        submod_name,
                        min(runner_by_idx),
                    )

                _inject_out_param(submod)
                wrapper = DynamicOpWrapper(
                    submod,
                    preceding_runner=effective_runner,
                    dynamic_submod_id=idx,
                )
                setattr(self.split_gm, submod_name, wrapper)
                self._wrapped_dynamic_indices.add(idx)
                num_wrapped_dynamic += 1

        self._is_prepared = True
        num_dynamic_eager = (
            len(self.split_info.dynamic_submod_indices) - num_wrapped_dynamic - num_metadata_wrapped
        )
        ad_logger.info(
            f"PiecewiseCapturedGraph: prepared with {self.split_info.num_submodules} submodules "
            f"({num_wrapped_static} static runners, {num_skipped_static} trivial skipped, "
            f"{num_wrapped_dynamic} dynamic wrapped, {num_metadata_wrapped} metadata wrapped, "
            f"{num_dynamic_eager} dynamic eager), "
            f"piecewise_num_tokens={self.piecewise_num_tokens}"
        )

    def _discover_dynamic_output_shapes(self, args: Tuple, kwargs: Dict) -> Dict[int, OutputInfo]:
        """Run one eager forward with hooks to discover output shapes of wrapped dynamic ops."""
        discovered: Dict[int, OutputInfo] = {}

        def _make_shape_hook(dynamic_idx: int):
            def hook(module, hook_args, output):
                if isinstance(output, torch.Tensor):
                    discovered[dynamic_idx] = OutputInfo(
                        shape=output.shape,
                        dtype=output.dtype,
                        device=output.device,
                    )
                else:
                    ad_logger.warning(
                        "Dynamic submod_%d returned %s (expected torch.Tensor), "
                        "cannot pre-allocate output buffer",
                        dynamic_idx,
                        type(output).__name__,
                    )

            return hook

        hooks: List[torch.utils.hooks.RemovableHandle] = []
        for idx in self._wrapped_dynamic_indices:
            wrapper = getattr(self.split_gm, f"submod_{idx}")
            if isinstance(wrapper, DynamicOpWrapper):
                h = wrapper.submodule.register_forward_hook(_make_shape_hook(idx))
                hooks.append(h)

        try:
            self.split_gm(*args, **kwargs)
        finally:
            for h in hooks:
                h.remove()

        return discovered

    def _set_dynamic_out_info_on_runners(self, discovered: Dict[int, OutputInfo]) -> None:
        """Pass discovered output shapes to the preceding static runners."""
        for dynamic_idx, info in discovered.items():
            wrapper = getattr(self.split_gm, f"submod_{dynamic_idx}")
            if isinstance(wrapper, DynamicOpWrapper):
                wrapper.preceding_runner.set_dynamic_out_info(dynamic_idx, info)

        missing = self._wrapped_dynamic_indices - set(discovered.keys())
        assert not missing, (
            f"Shape discovery did not capture outputs for dynamic submods: {sorted(missing)}. "
            f"Cannot pre-allocate out= buffers — downstream static runners require stable addresses."
        )

    def _allocate_static_input_buffers(
        self,
        get_args_kwargs: Callable[[int], Any],
    ) -> None:
        """Allocate static buffers for kwargs whose addresses change between calls.

        Calls `get_args_kwargs` twice with the largest bucket to check address
        stability (data_ptr), and once with a different size to detect the
        dynamic dimension by shape comparison.  Any kwarg with unstable
        addresses gets a pre-allocated static buffer.
        """
        max_bucket = max(self.piecewise_num_tokens)
        _, kw1 = get_args_kwargs(max_bucket)
        _, kw2 = get_args_kwargs(max_bucket)
        _, kw_probe = get_args_kwargs(max(1, max_bucket - 1))

        for key in kw1:
            v1, v2 = kw1.get(key), kw2.get(key)
            if (
                isinstance(v1, torch.Tensor)
                and isinstance(v2, torch.Tensor)
                and v1.data_ptr() != v2.data_ptr()
            ):
                v_probe = kw_probe.get(key)
                dyn_dim = None
                if isinstance(v_probe, torch.Tensor):
                    for d in range(v1.ndim):
                        if v1.shape[d] != v_probe.shape[d]:
                            dyn_dim = d
                            break
                if dyn_dim is not None or (
                    isinstance(v_probe, torch.Tensor) and v_probe.shape == v1.shape
                ):
                    # Static-shape kwargs still need buffering when their addresses
                    # change across calls. In that case dyn_dim stays None and we
                    # copy the full buffer at runtime.
                    self._static_input_buffers[key] = (torch.empty_like(v1), dyn_dim)
                else:
                    ad_logger.warning(
                        "PiecewiseCapturedGraph: kwarg '%s' has unstable address but "
                        "no dynamic dim found; leaving it unbuffered",
                        key,
                    )

        if self._static_input_buffers:
            ad_logger.info(
                "PiecewiseCapturedGraph: allocated %d static input buffer(s): %s",
                len(self._static_input_buffers),
                {k: (v[0].shape, f"dyn_dim={v[1]}") for k, v in self._static_input_buffers.items()},
            )

    def _copy_to_static_buffers(self, kwargs: Dict[str, Any]) -> None:
        """Copy kwargs into pre-allocated static buffers for address stability."""
        for key, (buf, dyn_dim) in self._static_input_buffers.items():
            src = kwargs.get(key)
            if src is not None and isinstance(src, torch.Tensor):
                if dyn_dim is None:
                    buf.copy_(src)
                    kwargs[key] = buf
                else:
                    buf_view = buf.narrow(dyn_dim, 0, src.shape[dyn_dim])
                    buf_view.copy_(src)
                    kwargs[key] = buf_view

    def warmup_and_capture(
        self,
        get_args_kwargs: Callable[[int], Any],
        warmup_iters: int = 3,
    ) -> None:
        """Warmup, discover dynamic output shapes, and capture CUDA graphs.

        For each num_tokens bucket (largest first):
          1. Warmup: run split_gm eagerly (warmup_iters - 1) times.
          2. Shape discovery: on the LAST warmup iteration, install forward hooks
             on wrapped dynamic ops to capture output shape/dtype.
          3. Inform runners: pass discovered OutputInfo to each ADPiecewiseRunner
             so it knows what to allocate inside its graph capture.
          4. Capture: run split_gm once more; runners capture CUDA graphs and
             allocate dynamic output buffers inside torch.cuda.graph().
          5. Cleanup: gc.collect() + empty_cache() between buckets.

        Before the per-bucket loop, calls get_args_kwargs twice to detect any
        kwargs whose tensor addresses are unstable.  Those kwargs are copied
        into static buffers for both capture and runtime replay.
        """
        if not self._is_prepared:
            self.prepare()

        if self.split_gm is None:
            return

        self._allocate_static_input_buffers(get_args_kwargs)

        num_tokens_list = sorted(self.piecewise_num_tokens, reverse=True)
        for nt in num_tokens_list:
            ad_logger.info(f"PiecewiseCapturedGraph: warming up for num_tokens={nt}")
            args, kwargs = get_args_kwargs(nt)
            self._copy_to_static_buffers(kwargs)

            ADPiecewiseRunner.set_current_num_tokens(nt)

            with CudaGraphWarmUpPhase():
                ADPiecewiseRunner.set_current_phase("warmup")
                for _ in range(warmup_iters - 1):
                    self.split_gm(*args, **kwargs)

                # Last warmup iteration: discover dynamic output shapes
                if self._wrapped_dynamic_indices:
                    discovered = self._discover_dynamic_output_shapes(args, kwargs)
                    self._set_dynamic_out_info_on_runners(discovered)
                else:
                    self.split_gm(*args, **kwargs)

                # Capture phase
                ADPiecewiseRunner.set_current_phase("capture")
                try:
                    self.split_gm(*args, **kwargs)
                finally:
                    for runner in self._static_runners.values():
                        runner.finalize_capture(nt)

            ad_logger.info(f"PiecewiseCapturedGraph: captured graphs for num_tokens={nt}")

            torch.cuda.synchronize()
            gc.collect()
            torch.cuda.empty_cache()

        ADPiecewiseRunner.set_current_num_tokens(None)
        ADPiecewiseRunner.set_current_phase("replay")

    def _reconstruct_output(self, result: Any) -> Any:
        """Reconstruct structured output from a flat tuple using the output tree spec."""
        if not isinstance(result, tuple) or self._out_spec is None:
            return result
        try:
            return self._out_spec.unflatten(list(result))
        except Exception as e:
            ad_logger.warning(
                "PiecewiseCapturedGraph._reconstruct_output: failed to unflatten output "
                "(%s); returning raw tuple",
                e,
            )
            return result

    def forward(
        self,
        *args,
        num_tokens: Optional[int] = None,
        **kwargs,
    ) -> Any:
        """Forward pass: static segments replay graphs, dynamic segments run eagerly."""
        if self.split_gm is not None:
            self._copy_to_static_buffers(kwargs)
            ADPiecewiseRunner.set_current_num_tokens(num_tokens)
            try:
                result = self.split_gm(*args, **kwargs)
                # Some captured kernels use internal CUDA streams, so wait for
                # graph-launched work to finish before returning to eager code.
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
            finally:
                ADPiecewiseRunner.set_current_num_tokens(None)
            return self._reconstruct_output(result)
        return self.original_model(*args, **kwargs)


class DualModeCapturedGraph(nn.Module):
    """Dispatches between monolithic CG (decode) and piecewise CG (prefill/mixed).

    At runtime:
    - If batch is decode-only (num_prefill == 0) -> use monolithic CapturedGraph
    - If batch has prefill/mixed tokens and total num_tokens <= largest pre-captured
      bucket -> use PiecewiseCapturedGraph with the smallest bucket >= num_tokens
    - Otherwise -> fall back to eager

    Padding contract for the piecewise path:
    - Input tensors (input_ids, position_ids) arrive at real size (total_num_tokens).
      The tail beyond total_num_tokens is zeroed via reset_val=0 in nest_sequences
      to prevent stale values from leaking into the padding region during graph replay.
    - batch_info reflects real counts so dynamic ops process only real tokens.
    - Output logits are truncated to total_num_tokens in forward().
    """

    def __init__(
        self,
        monolithic: CapturedGraph,
        piecewise: PiecewiseCapturedGraph,
        batch_info_kwarg_name: str = "batch_info_host",
        batched_input_names: Optional[List[str]] = None,
    ):
        super().__init__()
        self.monolithic = monolithic
        self.piecewise = piecewise
        self.batch_info_kwarg_name = batch_info_kwarg_name
        # Names of kwargs used to infer total num_tokens
        self.batched_input_names = batched_input_names or ["input_ids", "position_ids"]

        # Sorted list of pre-captured bucket sizes for nearest-bucket lookup
        self._captured_num_tokens_sorted: List[int] = sorted(piecewise.piecewise_num_tokens)

    def __getattr__(self, name: str):
        """Proxy attribute lookups to the underlying model.

        When this module replaces an inner submodule, the parent may access
        methods like get_input_embeddings() that live on the original model.
        """
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.monolithic.model, name)

    def _is_decode_only(self, **kwargs) -> bool:
        """Check if the current batch is decode-only using batch_info_host.

        batch_info_host is the serialized BatchInfo tensor.
        Decode-only means num_prefill == 0.
        """
        batch_info = kwargs.get(self.batch_info_kwarg_name)
        if batch_info is not None and isinstance(batch_info, torch.Tensor):
            # Under AD_BATCH_INFO_DEVICE=1 the kwarg is the device shadow; read
            # from the registered host buffer instead to avoid a CUDA sync.
            if batch_info.device.type == "cuda":
                from ...custom_ops.attention_interface import _get_current_batch_info_host

                host = _get_current_batch_info_host()
                if host is not None:
                    batch_info = host
            # batch_info_host[0] = num_prefill
            num_prefill = batch_info[0].item()
            return num_prefill == 0

        # Fallback heuristic: check if first batched input has sequence dim == 1
        # (decode = 1 token per sequence)
        for name in self.batched_input_names:
            v = kwargs.get(name)
            if v is not None and isinstance(v, torch.Tensor) and v.ndim >= 2:
                return v.shape[1] == 1

        # Default to monolithic (decode) path
        return True

    def _get_num_tokens(self, **kwargs) -> int:
        """Extract total num_tokens from batch_info_host or batched inputs."""
        batch_info = kwargs.get(self.batch_info_kwarg_name)
        if batch_info is not None and isinstance(batch_info, torch.Tensor):
            if batch_info.device.type == "cuda":
                from ...custom_ops.attention_interface import _get_current_batch_info_host

                host = _get_current_batch_info_host()
                if host is not None:
                    batch_info = host
            # batch_info_host layout: [0]=num_prefill, [1]=num_prefill_tokens,
            # [2]=num_extend, [3]=num_extend_tokens, [4]=num_decode, [5]=num_decode_tokens
            return int((batch_info[1] + batch_info[3] + batch_info[5]).item())
        for name in self.batched_input_names:
            v = kwargs.get(name)
            if v is not None and isinstance(v, torch.Tensor) and v.ndim >= 1:
                return int(v.numel())
        return 0

    def _find_nearest_bucket(self, num_tokens: int) -> Optional[int]:
        """Find smallest captured bucket >= num_tokens, or None."""
        for bucket in self._captured_num_tokens_sorted:
            if bucket >= num_tokens:
                return bucket
        return None

    def _truncate_output(self, result: Any, num_tokens: int, bucket: int) -> Any:
        """Slice padded outputs from bucket size to real num_tokens.

        Finds the token dimension by looking for the dim whose size equals
        the bucket size, then narrows it to num_tokens.
        """
        output_dynamic_dim = getattr(self.monolithic, "_output_dynamic_dim", None)

        def _narrow(v):
            if not isinstance(v, torch.Tensor):
                return v
            if (
                isinstance(output_dynamic_dim, int)
                and 0 <= output_dynamic_dim < v.ndim
                and v.shape[output_dynamic_dim] == bucket
            ):
                return v.narrow(output_dynamic_dim, 0, num_tokens)
            matching_dims = [d for d in range(v.ndim) if v.shape[d] == bucket]
            if not matching_dims:
                return v
            if len(matching_dims) > 1:
                ad_logger.warning(
                    "DualModeCapturedGraph._truncate_output: ambiguous token dim for shape %s, "
                    "bucket=%d; falling back to dim %d",
                    tuple(v.shape),
                    bucket,
                    matching_dims[0],
                )
            return v.narrow(matching_dims[0], 0, num_tokens)

        if isinstance(result, torch.Tensor):
            return _narrow(result)
        if hasattr(result, "to_tuple"):
            sliced = {k: _narrow(v) for k, v in result.items()}
            return type(result)(**sliced)
        elif isinstance(result, abc.Mapping):
            return {k: _narrow(v) for k, v in result.items()}
        else:
            return tuple(_narrow(r) for r in result)

    def forward(self, *args, **kwargs) -> Any:
        # NOTE: AD calls model(**named_args) so everything is in kwargs, args is empty
        if self._is_decode_only(**kwargs):
            ADPiecewiseRunner.set_current_num_tokens(None)
            return self.monolithic(*args, **kwargs)

        # ── PREFILL/MIXED PATH ──
        num_tokens = self._get_num_tokens(**kwargs)
        bucket = self._find_nearest_bucket(num_tokens)
        if bucket is not None:
            try:
                result = self.piecewise(
                    *args,
                    num_tokens=bucket,
                    **kwargs,
                )
            finally:
                ADPiecewiseRunner.set_current_num_tokens(None)
            if bucket > num_tokens:
                result = self._truncate_output(result, num_tokens, bucket)
            return result

        # No bucket large enough -- eager fallback
        ADPiecewiseRunner.set_current_num_tokens(None)
        ad_logger.debug(
            f"DualModeCapturedGraph: num_tokens={num_tokens} exceeds largest bucket "
            f"{self._captured_num_tokens_sorted[-1] if self._captured_num_tokens_sorted else 'N/A'}"
            f", falling back to eager"
        )
        return self.piecewise.original_model(*args, **kwargs)


def _setup_piecewise_mixed_batch(seq_info: Any, num_tokens: int) -> None:
    """Set up SequenceInfo for a mixed-batch with the given total num_tokens.

    Creates a mixed batch with at least 1 prefill + 1 decode to exercise both
    code paths in dynamic ops (attention, SSM). Each prefill sequence is capped
    to max_seq_len so page indices stay within block_offsets capacity.

    Args:
        seq_info: SequenceInfo object (duck-typed: needs max_seq_len, max_batch_size,
            tokens_per_block, and nest_sequences method).
        num_tokens: Total number of tokens for this piecewise bucket.
    """
    assert num_tokens >= 3, (
        f"Piecewise bucket {num_tokens} too small for mixed batch. "
        f"Minimum is 3 (1 prefill seq with len>=2 + 1 decode seq)."
    )
    max_seq = seq_info.max_seq_len
    max_batch = seq_info.max_batch_size

    seq_lens: List[int] = []
    remaining = num_tokens - 1
    while remaining > 0 and len(seq_lens) < max_batch - 1:
        sl = min(remaining, max_seq)
        seq_lens.append(sl)
        remaining -= sl
    seq_lens.append(1)  # decode token

    assert remaining == 0, (
        f"Piecewise bucket {num_tokens} exceeds batch capacity "
        f"({max_batch - 1} seqs * {max_seq} tokens + 1 decode). "
        f"Increase max_seq_len or max_batch_size."
    )

    bs = len(seq_lens)
    input_ids_flat = torch.ones(sum(seq_lens), dtype=torch.int)

    cu_seqlen = torch.zeros(bs + 1, dtype=torch.int)
    for i, sl in enumerate(seq_lens):
        cu_seqlen[i + 1] = cu_seqlen[i] + sl

    tpb = seq_info.tokens_per_block
    cu_num_pages = torch.zeros(bs + 1, dtype=torch.int)
    for i, sl in enumerate(seq_lens):
        cu_num_pages[i + 1] = cu_num_pages[i] + (sl + tpb - 1) // tpb
    cache_loc = torch.arange(cu_num_pages[-1].item())
    slot_idx = torch.arange(bs)

    seq_info.nest_sequences(
        input_ids=input_ids_flat,
        cu_seqlen=cu_seqlen,
        input_pos=0,
        cache_loc=cache_loc,
        cu_num_pages=cu_num_pages,
        slot_idx=slot_idx,
    )


def _capture_inner_kwargs(
    full_model: nn.Module,
    inner_module: nn.Module,
    top_level_kwargs: Dict[str, Any],
) -> Dict[str, Any]:
    """Run full model once and intercept kwargs passed to the inner module."""
    captured: Dict[str, Any] = {}

    def hook(module, args, kwargs):
        captured.update(kwargs)
        return args, kwargs

    handle = inner_module.register_forward_pre_hook(hook, with_kwargs=True)
    try:
        full_model(**top_level_kwargs)
    finally:
        handle.remove()
    return captured


@CompileBackendRegistry.register("torch-cudagraph")
class TorchCudagraphCompiler(CompilerBackend):
    """Compiler that uses CUDA graphs.

    Supports two modes:
    - piecewise_enabled=False (default): monolithic CG only (decode-only batches)
    - piecewise_enabled=True: dual-mode (monolithic for decode + piecewise for prefill/mixed)

    When the top-level model is a wrapper (not a GraphModule), the compiler
    auto-discovers the inner GraphModule (e.g. text model) and compiles it.
    The wrapper (e.g. vision tower, embed merge) runs eagerly.
    """

    def __init__(
        self,
        *args_for_init,
        cuda_graph_batch_sizes: Optional[List[int]] = None,
        num_batched_inputs: int = 1,
        get_args_kwargs_for_compile: GetArgsKwargsForBatchSize = None,
        piecewise_enabled: bool = False,
        piecewise_num_tokens: Optional[List[int]] = None,
        piecewise_seq_info: Any = None,
        piecewise_named_args_fn: Optional[Callable[[], Dict[str, Any]]] = None,
        full_model: Optional[nn.Module] = None,
        layers_per_chunk: Optional[int] = None,
        **kwargs_for_init,
    ):
        super().__init__(*args_for_init, **kwargs_for_init)
        self.num_batched_inputs = num_batched_inputs
        self.cuda_graph_batch_sizes = cuda_graph_batch_sizes or []
        self.get_args_kwargs_for_compile = get_args_kwargs_for_compile
        self.piecewise_enabled = piecewise_enabled
        self.piecewise_num_tokens = piecewise_num_tokens or []
        self.piecewise_seq_info = piecewise_seq_info
        self.piecewise_named_args_fn = piecewise_named_args_fn
        self.full_model = full_model
        # YAML-provided value; env var overrides this in compile().
        self.layers_per_chunk_yaml = layers_per_chunk

    def _get_inner_args_kwargs_fn(self, inner_gm: GraphModule) -> GetArgsKwargsForBatchSize:
        """Return a function that generates inner-model args for a given batch size.

        Runs the full model with top-level args and captures the kwargs that the
        wrapper passes to the inner GraphModule via a forward pre-hook.
        """
        assert self.full_model is not None

        def get_inner_args(batch_size: int):
            _, top_level_kwargs = self.get_args_kwargs_for_compile(batch_size)
            inner_kwargs = _capture_inner_kwargs(self.full_model, inner_gm, top_level_kwargs)
            return (), inner_kwargs

        return get_inner_args

    @torch.inference_mode()
    def compile(self) -> nn.Module:
        assert self.get_args_kwargs_for_compile is not None, (
            "get_args_kwargs_for_compile must be provided"
        )

        # Build args functions once — unified for both wrapper and direct GM cases.
        # self.model is always the target GM (compile_model.py extracts it).
        target_gm = self.model

        if self.full_model is not None:
            ad_logger.info("TorchCudagraphCompiler: wrapper detected, compiling inner GraphModule")
            get_capture_args_fn = self._get_inner_args_kwargs_fn(target_gm)
        else:
            get_capture_args_fn = self.get_args_kwargs_for_compile

        def get_capture_args_with_warmup(batch_size: int):
            with CudaGraphWarmUpPhase():
                return get_capture_args_fn(batch_size)

        # Select the decode-path wrapper.  Env var ``AD_LAYERS_PER_CHUNK``
        # takes precedence over the YAML field; both default off (None) and
        # fall back to the monolithic CapturedGraph (current behaviour).
        layers_per_chunk, lpc_source = _resolve_layers_per_chunk(self.layers_per_chunk_yaml)
        if layers_per_chunk is not None:
            monolithic: CapturedGraph = PerNLayersCapturedGraph(
                target_gm,
                layers_per_chunk=layers_per_chunk,
                layers_per_chunk_source=lpc_source,
                num_batched_inputs=self.num_batched_inputs,
            )
        else:
            monolithic = CapturedGraph(target_gm, num_batched_inputs=self.num_batched_inputs)
        monolithic.capture_graph(get_capture_args_with_warmup, self.cuda_graph_batch_sizes)

        piecewise = None
        if self.piecewise_enabled:
            ad_logger.info("TorchCudagraphCompiler: dual-mode enabled (monolithic + piecewise)")
            piecewise = PiecewiseCapturedGraph(
                model=target_gm,
                piecewise_num_tokens=self.piecewise_num_tokens,
                max_batch_size=(
                    self.piecewise_seq_info.max_batch_size
                    if self.piecewise_seq_info is not None
                    else None
                ),
                out_spec=monolithic._out_spec,
            )
            piecewise.prepare()

            if (
                self.piecewise_seq_info is not None
                and self.piecewise_named_args_fn is not None
                and self.piecewise_num_tokens
            ):

                def get_piecewise_args(num_tokens: int):
                    _setup_piecewise_mixed_batch(self.piecewise_seq_info, num_tokens)
                    top_level_kwargs = self.piecewise_named_args_fn()
                    if self.full_model is not None:
                        return (), _capture_inner_kwargs(
                            self.full_model, target_gm, top_level_kwargs
                        )
                    return (), top_level_kwargs

                piecewise.warmup_and_capture(get_piecewise_args)

        if piecewise is not None:
            return DualModeCapturedGraph(monolithic, piecewise)
        return monolithic
