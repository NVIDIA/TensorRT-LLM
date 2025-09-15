"""Compile backend with cudagraph."""

from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.cuda import CUDAGraph
from torch.utils._pytree import TreeSpec, tree_flatten

from ...utils.cuda_graph import CudaGraphWarmUpPhase
from ...utils.logger import ad_logger
from ..compiler import BackendCompiler, BackendRegistry, _flatten_args


class CapturedGraph(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        in_spec: TreeSpec,
        out_spec: TreeSpec,
        cuda_graph_batch_sizes: List[int],
        num_batched_inputs: Optional[int] = 1,  # number of batched, dynamic inputs...
    ):
        super().__init__()
        self._in_spec = in_spec
        self._out_spec = out_spec
        self.model = model
        self.cuda_graph_max_batch_size = max(cuda_graph_batch_sizes)
        ad_logger.info(f"Setting {self.cuda_graph_max_batch_size=}")
        self.num_batched_inputs = num_batched_inputs if num_batched_inputs is not None else 1
        self.graphs: Dict[Tuple[int, ...], CUDAGraph] = {}
        self._input_buffers: List[torch.Tensor] = [
            torch.empty(0, 1) for _ in range(self.num_batched_inputs)
        ]
        self._out_buffer_flat: List[torch.Tensor] = None
        self._args_hash: Optional[Tuple[int, ...]] = None
        self.cuda_graph_batch_sizes = sorted(cuda_graph_batch_sizes, reverse=True)
        self._cuda_graph_mem_pool = None

    def _get_hash(self, flat_args: List[Any]) -> Tuple[int, ...]:
        return tuple(hash(a) for a in flat_args)

    @staticmethod
    def round_up_to_closest(batch_sizes: Iterable[int], bs: int) -> Optional[int]:
        """Return closest batch size larger or equal to bs."""
        if bs > max(batch_sizes, default=0):
            return None
        return min(batch_sizes, key=lambda x: (x < bs, abs(x - bs)), default=None)

    def round_to_cuda_batch_size(self, bs: int) -> int:
        """Round batch size to the nearest cuda batch size."""
        return self.round_up_to_closest(self.cuda_graph_batch_sizes, bs)

    def _capture_one_graph(self, *args, **kwargs) -> torch.cuda.CUDAGraph:
        """Capture and return one cuda graph."""
        # warm-up
        with CudaGraphWarmUpPhase():
            for _ in range(3):
                self.model(*args, **kwargs)

        # capture graph now
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=self._cuda_graph_mem_pool):
            # compute output
            out = self.model(*args, **kwargs)
            # write out into output buffer up to out batch size
            out_flat, out_spec = tree_flatten(out)
            assert out_spec == self._out_spec, "Output spec mismatch."
            for o_buffer, o in zip(self._out_buffer_flat, out_flat):
                o_buffer[: o.shape[0]] = o
        torch.cuda.synchronize()
        self._cuda_graph_mem_pool = self._cuda_graph_mem_pool or graph.pool()
        return graph

    def capture_graph(self, *args, **kwargs):
        """Capture and pre-fetch the graph for variable batch size."""
        # flatten args, kwargs
        all_args_flat = _flatten_args(self._in_spec, *args, **kwargs)

        # extract the batched input tensors
        args_batched = all_args_flat[: self.num_batched_inputs]
        args_static = all_args_flat[self.num_batched_inputs :]

        # set the args hash --> this is used to compare the static inputs during graph replay
        self._args_hash = self._get_hash(args_static)

        # sanity checks on the batched inputs
        msg_bs = (
            f"Input batch size exceeds maximum CUDA graph batch size. "
            f"CUDA graph max batch size: {self.cuda_graph_max_batch_size}, "
            f"but got input batch sizes: {[input.shape[0] for input in args_batched]}. "
            f"Did you intentionally set the maximal value of cuda_graph_batch_sizes lower "
            f"than the max_batch_size? It will fall back to non-CUDA graph forward pass for "
            f"batch sizes exceeding the max_batch_size."
        )
        msg_ndim = "Expecting at least a 2D for batched input tensors."
        if any(self.cuda_graph_max_batch_size < input.shape[0] for input in args_batched):
            ad_logger.info(msg_bs)
        assert all(input.ndim > 1 for input in args_batched), msg_ndim

        # repeat the batched input tensors to the cuda_graph_max_batch_size
        self._input_buffers = [
            input[:1].repeat_interleave(self.cuda_graph_max_batch_size, dim=0)
            for input in args_batched
        ]

        # create new args, kwargs with the input buffers and static args
        args, kwargs = self._in_spec.unflatten(self._input_buffers + args_static)

        # truncate input buffers to cuda_graph_max_batch_size
        inputs_truncated = [
            in_buffer[: self.cuda_graph_max_batch_size] for in_buffer in self._input_buffers
        ]
        args, kwargs = self._in_spec.unflatten(inputs_truncated + args_static)

        # capture output once with cuda_graph_max_batch_size to capture output buffers
        with CudaGraphWarmUpPhase():
            ad_logger.info(f"Warm up with {self.cuda_graph_max_batch_size=} before graph capture")
            out = self.model(*args, **kwargs)
        self._out_buffer_flat, out_spec = tree_flatten(out)
        assert out_spec == self._out_spec, "Output spec mismatch."

        # capture graph now for a range of batch sizes
        for bs in self.cuda_graph_batch_sizes:
            ad_logger.info(f"Capturing graph for batch size: {bs}")
            # capture graph for truncated inputs
            combined_shape = sum((input.shape for input in inputs_truncated), start=())
            self.graphs[combined_shape] = self._capture_one_graph(*args, **kwargs)

    def forward(self, *args, **kwargs) -> Any:
        """Run the compiled graph."""
        # flatten args, kwargs
        all_args_flat = _flatten_args(self._in_spec, *args, **kwargs)

        # extract the batched input tensors
        args_batched = all_args_flat[: self.num_batched_inputs]
        args_static = all_args_flat[self.num_batched_inputs :]

        # check if args_static match the stored hash
        if self._args_hash != self._get_hash(args_static):
            return self.model(*args, **kwargs)

        # Calculate rounded-up shapes for each input
        # If any batch size exceeds max cuda_graph_batch_sizes, fall back to regular forward
        rounded_batch_sizes = [
            self.round_to_cuda_batch_size(input.shape[0]) for input in args_batched
        ]

        # If any rounded batch size is None (exceeds max), use regular forward
        if any(bs is None for bs in rounded_batch_sizes):
            actual_batch_sizes = [input.shape[0] for input in args_batched]
            max_cuda_bs = max(self.cuda_graph_batch_sizes) if self.cuda_graph_batch_sizes else 0
            ad_logger.debug(
                f"Batch size {actual_batch_sizes} exceeds max CUDA graph batch size {max_cuda_bs}. "
                f"Falling back to regular forward pass."
            )
            return self.model(*args, **kwargs)

        rounded_shapes = [
            (rounded_bs,) + input.shape[1:]
            for rounded_bs, input in zip(rounded_batch_sizes, args_batched)
        ]
        combined_shape = sum(rounded_shapes, start=())

        # regular forward for non-matching shapes
        if combined_shape not in self.graphs:
            return self.model(*args, **kwargs)

        # copy inputs to input buffers
        for i, input_tensor in enumerate(args_batched):
            self._input_buffers[i][: input_tensor.shape[0]].copy_(input_tensor, non_blocking=True)

        # run forward pass via graph
        self.graphs[combined_shape].replay()

        # retrieve output from buffer, cut to batch size, and unflatten
        bs = args_batched[0].shape[0]
        out_flat = [o_b[:bs].detach().clone() for o_b in self._out_buffer_flat]
        return self._out_spec.unflatten(out_flat)


@BackendRegistry.register("torch-cudagraph")
class TorchCudagraphCompiler(BackendCompiler):
    """Compiler that uses only CUDA graphs."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        requested = self.compiler_kwargs.get("cuda_graph_batch_sizes")
        if not requested:
            # Use heuristic which includes commonly-used sizes like 1 and max_bs
            self.cuda_graph_batch_sizes = self._get_graph_batch_sizes(self.max_batch_size)
            ad_logger.info(f"Using heuristic cuda_graph_batch_sizes: {self.cuda_graph_batch_sizes}")
        else:
            # Sanitize user-provided sizes: clamp to [1, max_batch_size], dedupe, sort desc
            # No point capturing CUDA graphs for batch sizes larger than max_batch_size
            effective = {
                min(max(1, int(b)), int(self.max_batch_size))
                for b in requested
                if isinstance(b, (int, float)) and b > 0
            }
            self.cuda_graph_batch_sizes = sorted(effective, reverse=True)

            # Log if we clamped any values
            original_values = [int(b) for b in requested if isinstance(b, (int, float)) and b > 0]
            clamped_values = [v for v in original_values if v > self.max_batch_size]
            if clamped_values:
                ad_logger.info(
                    f"Clamped CUDA graph batch sizes {clamped_values} to max_batch_size={self.max_batch_size}"
                )

            ad_logger.info(
                f"Using explicit cuda_graph_batch_sizes: requested={requested}"
                f" -> effective={self.cuda_graph_batch_sizes}"
                f" (clamped to [1, {self.max_batch_size}])"
            )

    def _init_captured_graph(
        self, gm: nn.Module, in_spec: TreeSpec, out_spec: TreeSpec
    ) -> CapturedGraph:
        return CapturedGraph(
            gm,
            in_spec=in_spec,
            out_spec=out_spec,
            cuda_graph_batch_sizes=self.cuda_graph_batch_sizes,
            num_batched_inputs=self.compiler_kwargs.get("num_batched_inputs"),
        )

    @torch.inference_mode()
    def compile(self) -> CapturedGraph:
        captured_model = self._init_captured_graph(self.gm, self.gm._in_spec, self.gm._out_spec)

        # try capturing cudagraph
        if self.args is not None or self.kwargs is not None:
            captured_model.capture_graph(*self.args, **self.kwargs)

        return captured_model

    @staticmethod
    def _get_graph_batch_sizes(
        max_bs: int, extra: Optional[List[int]] = None, multiplier: int = 128
    ) -> List[int]:
        """Heuristic to set batch sizes for graph capture."""
        # do 1, max_bs, and extra as special batch sizes
        batch_sizes = {1, max_bs, *(extra or [])}

        # add all multiples of multiplier up to max_bs
        batch_sizes.update(range(multiplier, max_bs + 1, multiplier))

        # return as sorted list
        return sorted(batch_sizes, reverse=True)
