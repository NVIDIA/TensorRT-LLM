"""Compile backend with cudagraph.

1. Monolithic CUDA graph: captures entire model as one graph for decode-only.
2. Piecewise CUDA graph: splits model at dynamic ops, captures static segments
   individually. Used for prefill/mixed batches when piecewise_enabled=True.

When piecewise_enabled=True, a DualModeCapturedGraph is returned that dispatches:
  - Decode-only batches → monolithic CapturedGraph (fastest, single graph replay)
  - Prefill/mixed batches → PiecewiseCapturedGraph (per-segment replay + eager dynamic ops)
"""

import copy  # noqa: I001
import gc
from collections import abc
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import torch
import torch.nn as nn
from torch.cuda import CUDAGraph
from torch.fx import GraphModule
from torch.fx._pytree import tree_flatten_spec
from torch.utils._pytree import PyTree, TreeSpec, tree_flatten

from tensorrt_llm._torch.autotuner import autotune

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

    target_node.kwargs = {**dict(target_node.kwargs), "out": out_placeholder}

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

    def _capture_one_graph(self, *args, **kwargs) -> torch.cuda.CUDAGraph:
        """Capture and return one cuda graph."""
        # warm-up and invoke autotuner
        with CudaGraphWarmUpPhase(), autotune():
            for _ in range(3):
                self.model(*args, **kwargs)

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

            # capture graph for truncated inputs
            combined_shape = sum((tuple(input.shape) for input in inputs_truncated), start=())
            self.cudagraphs[combined_shape] = self._capture_one_graph(*args, **kwargs)

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
            f"{num_dynamic_eager} dynamic eager), piecewise_num_tokens={self.piecewise_num_tokens}"
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
                self.split_gm(*args, **kwargs)

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

    def forward(self, *args, num_tokens: Optional[int] = None, **kwargs) -> Any:
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
                result = self.piecewise(*args, num_tokens=bucket, **kwargs)
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
