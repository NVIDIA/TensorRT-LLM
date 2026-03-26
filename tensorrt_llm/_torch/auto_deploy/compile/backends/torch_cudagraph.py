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
    ):
        super().__init__()
        self.model = model
        self.num_batched_inputs = num_batched_inputs if num_batched_inputs is not None else 1
        self.cudagraphs: Dict[Tuple[int, ...], CUDAGraph] = {}
        self._input_buffers: List[torch.Tensor] = [
            torch.empty(0, 1) for _ in range(self.num_batched_inputs)
        ]
        self._out_buffer_flat: List[torch.Tensor] = None
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
        with torch.cuda.graph(graph, pool=self._cuda_graph_mem_pool):
            # compute output
            out = self.model(*args, **kwargs)
            # write out into output buffer up to out batch size
            out_flat = tree_flatten_spec(out, self._out_spec)
            for o_buffer, o in zip(self._out_buffer_flat, out_flat):
                o_buffer[: o.shape[0]] = o
        torch.cuda.synchronize()
        self._cuda_graph_mem_pool = self._cuda_graph_mem_pool or graph.pool()
        return graph

    def capture_graph(self, get_args_kwargs: GetArgsKwargsForBatchSize, batch_sizes: List[int]):
        """Capture and pre-fetch the graph for desired batch sizes."""
        assert not self.cudagraphs, "Graphs already captured."

        # sort batch sizes in descending order
        batch_sizes = sorted(batch_sizes, reverse=True)

        # get args, kwargs for the first time for the largest batch size
        args, kwargs = get_args_kwargs(batch_sizes[0])

        # flatten args, kwargs for the first time and record in_spec
        all_args_flat, self._in_spec = _args_kwargs_flatten(*args, **kwargs)

        # extract the batched input tensors
        args_batched = all_args_flat[: self.num_batched_inputs]
        args_static = all_args_flat[self.num_batched_inputs :]

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

            # copy new inputs to input buffers
            for i, input_tensor in enumerate(args_batched):
                self._input_buffers[i][: input_tensor.shape[0]].copy_(
                    input_tensor, non_blocking=True
                )

            # setup args, kwargs
            inputs_truncated = [in_buffer[:bs] for in_buffer in self._input_buffers]
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

        # copy inputs to input buffers
        for i, input_tensor in enumerate(args_batched):
            self._input_buffers[i][: input_tensor.shape[0]].copy_(input_tensor, non_blocking=True)

        # run forward pass via graph
        self.cudagraphs[combined_shape].replay()

        # retrieve output from buffer, cut to batch size, and unflatten
        bs = args_batched[0].shape[0]
        out_flat = [o_b[:bs] for o_b in self._out_buffer_flat]
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

                assert current_static_runner is not None, (
                    f"Dynamic {submod_name} has no preceding static runner — "
                    f"cannot allocate out= buffer for stable output addresses"
                )

                _inject_out_param(submod)
                wrapper = DynamicOpWrapper(
                    submod,
                    preceding_runner=current_static_runner,
                    dynamic_submod_id=idx,
                )
                setattr(self.split_gm, submod_name, wrapper)
                self._wrapped_dynamic_indices.add(idx)
                num_wrapped_dynamic += 1

        self._is_prepared = True
        ad_logger.info(
            "PiecewiseCapturedGraph: prepared with %d submodules "
            "(%d static runners, %d trivial skipped, %d dynamic wrapped, "
            "%d metadata wrapped, %d dynamic eager), piecewise_num_tokens=%s",
            self.split_info.num_submodules,
            num_wrapped_static,
            num_skipped_static,
            num_wrapped_dynamic,
            num_metadata_wrapped,
            len(self.split_info.dynamic_submod_indices)
            - num_wrapped_dynamic
            - num_metadata_wrapped,
            self.piecewise_num_tokens,
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
        """
        if not self._is_prepared:
            self.prepare()

        if self.split_gm is None:
            return

        num_tokens_list = sorted(self.piecewise_num_tokens, reverse=True)
        for nt in num_tokens_list:
            ad_logger.info("PiecewiseCapturedGraph: warming up for num_tokens=%d", nt)
            args, kwargs = get_args_kwargs(nt)

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

            ad_logger.info("PiecewiseCapturedGraph: captured graphs for num_tokens=%d", nt)

            torch.cuda.synchronize()
            gc.collect()
            torch.cuda.empty_cache()

        ADPiecewiseRunner.set_current_num_tokens(None)
        ADPiecewiseRunner.set_current_phase("replay")

    def forward(self, *args, num_tokens: Optional[int] = None, **kwargs) -> Any:
        """Forward pass: static segments replay graphs, dynamic segments run eagerly."""
        if self.split_gm is not None:
            ADPiecewiseRunner.set_current_num_tokens(num_tokens)
            return self.split_gm(*args, **kwargs)
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

    def _is_decode_only(self, **kwargs) -> bool:
        """Check if the current batch is decode-only using batch_info_host.

        batch_info_host = [num_prefill, num_prefill_tokens, num_decode]
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
        """Extract total num_tokens from the batched inputs.

        For prefill/mixed with flattened layout: input_ids shape = [1, total_num_tokens]
        We use numel() which works for both [1, N] and [N] layouts.
        """
        for name in self.batched_input_names:
            v = kwargs.get(name)
            if v is not None and isinstance(v, torch.Tensor):
                return v.numel()
        return 0

    def _find_nearest_bucket(self, num_tokens: int) -> Optional[int]:
        """Find smallest captured bucket >= num_tokens, or None."""
        for bucket in self._captured_num_tokens_sorted:
            if bucket >= num_tokens:
                return bucket
        return None

    def forward(self, *args, **kwargs) -> Any:
        # NOTE: AD calls model(**named_args) so everything is in kwargs, args is empty
        if self._is_decode_only(**kwargs):
            ADPiecewiseRunner.set_current_num_tokens(None)
            return self.monolithic(*args, **kwargs)

        # ── PREFILL/MIXED PATH ──
        num_tokens = self._get_num_tokens(**kwargs)
        bucket = self._find_nearest_bucket(num_tokens)
        if bucket is not None:
            result = self.piecewise(*args, num_tokens=bucket, **kwargs)
            ADPiecewiseRunner.set_current_num_tokens(None)
            if bucket > num_tokens:
                # HF ModelOutput iterates over field names (e.g. "logits"), not
                # tensor values. Normalize to the payload tuple before slicing.
                if hasattr(result, "to_tuple"):
                    result = result.to_tuple()
                elif isinstance(result, abc.Mapping):
                    result = tuple(result.values())
                else:
                    result = tuple(result)
                result = tuple(r[:, :num_tokens] if r.ndim >= 2 else r for r in result)
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


@CompileBackendRegistry.register("torch-cudagraph")
class TorchCudagraphCompiler(CompilerBackend):
    """Compiler that uses CUDA graphs.

    Supports two modes:
    - piecewise_enabled=False (default): monolithic CG only (decode-only batches)
    - piecewise_enabled=True: dual-mode (monolithic for decode + piecewise for prefill/mixed)
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

    @torch.inference_mode()
    def compile(self) -> nn.Module:
        assert self.get_args_kwargs_for_compile is not None, (
            "get_args_kwargs_for_compile must be provided"
        )

        # wrap get_args_kwargs_for_compile with CudaGraphWarmUpPhase. Note that host-side prepare
        # functions may be called as part of get_args_kwargs. We want to let these functions know it's
        # a warm-up phase.
        def get_args_kwargs_warmup(batch_size: int):
            with CudaGraphWarmUpPhase():
                return self.get_args_kwargs_for_compile(batch_size)

        monolithic = CapturedGraph(self.model, num_batched_inputs=self.num_batched_inputs)
        monolithic.capture_graph(get_args_kwargs_warmup, self.cuda_graph_batch_sizes)

        piecewise = None
        if self.piecewise_enabled:
            ad_logger.info("TorchCudagraphCompiler: dual-mode enabled (monolithic + piecewise)")
            piecewise = PiecewiseCapturedGraph(
                model=self.model,
                piecewise_num_tokens=self.piecewise_num_tokens,
                max_batch_size=(
                    self.piecewise_seq_info.max_batch_size
                    if self.piecewise_seq_info is not None
                    else None
                ),
            )
            piecewise.prepare()

            if (
                self.piecewise_seq_info is not None
                and self.piecewise_named_args_fn is not None
                and self.piecewise_num_tokens
            ):

                def get_mixed_args_kwargs(num_tokens: int):
                    _setup_piecewise_mixed_batch(self.piecewise_seq_info, num_tokens)
                    return (), self.piecewise_named_args_fn()

                piecewise.warmup_and_capture(get_mixed_args_kwargs)

        if piecewise is not None:
            return DualModeCapturedGraph(monolithic, piecewise)
        return monolithic
