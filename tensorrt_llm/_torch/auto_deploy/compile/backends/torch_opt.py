"""Mixed backend with torch + Cudagraph."""

from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.cuda import CUDAGraph
from torch.fx import GraphModule
from torch.utils._pytree import TreeSpec, tree_flatten

from ...utils.cuda_graph import CudaGraphWarmUpPhase
from ...utils.logger import ad_logger
from ..compiler import BackendCompiler, BackendRegistry, _flatten_args


class CompiledGraph(nn.Module):
    def __init__(
        self,
        model: GraphModule,
        max_batch_size: int,
        cuda_graph_batch_sizes: List[int] = None,
        num_batched_inputs: Optional[int] = 1,  # number of batched, dynamic inputs...
    ):
        super().__init__()
        self._in_spec: TreeSpec = model._in_spec
        self._out_spec: TreeSpec = model._out_spec
        self.gm_compiled = torch.compile(model, dynamic=True)
        self.max_batch_size = max_batch_size
        self.num_batched_inputs = num_batched_inputs if num_batched_inputs is not None else 1
        self.graphs: Dict[Tuple[int, ...], CUDAGraph] = {}
        self._input_buffers: List[torch.Tensor] = [
            torch.empty(0, 1) for _ in range(self.num_batched_inputs)
        ]
        self._out_buffer_flat: List[torch.Tensor] = None
        self._args_hash: Optional[Tuple[int, ...]] = None
        self.cuda_graph_batch_sizes = (
            cuda_graph_batch_sizes
            if cuda_graph_batch_sizes is not None
            else self._get_graph_batch_sizes(self.max_batch_size)
        )

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
                self.gm_compiled(*args, **kwargs)

        # capture graph now
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            # compute output
            out = self.gm_compiled(*args, **kwargs)
            # write out into output buffer up to out batch size
            out_flat, out_spec = tree_flatten(out)
            assert out_spec == self._out_spec, "Output spec mismatch."
            for o_buffer, o in zip(self._out_buffer_flat, out_flat):
                o_buffer[: o.shape[0]] = o
        torch.cuda.synchronize()

        return graph

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
        return sorted(batch_sizes)

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
        msg_bs = "Max batch size too small."
        msg_ndim = "Expecting at least a 2D for batched input tensors."
        assert all(self.max_batch_size >= input.shape[0] for input in args_batched), msg_bs
        assert all(input.ndim > 1 for input in args_batched), msg_ndim

        # repeat the batched input tensors to the max batch size
        self._input_buffers = [
            input[:1].repeat_interleave(self.max_batch_size, dim=0) for input in args_batched
        ]

        # create new args, kwargs with the input buffers and static args
        args, kwargs = self._in_spec.unflatten(self._input_buffers + args_static)

        # capture output once with max batch size to capture output buffers
        with CudaGraphWarmUpPhase():
            out = self.gm_compiled(*args, **kwargs)
        self._out_buffer_flat, out_spec = tree_flatten(out)
        assert out_spec == self._out_spec, "Output spec mismatch."

        # capture graph now for a range of batch sizes
        for bs in self.cuda_graph_batch_sizes:
            ad_logger.info(f"Capturing graph for batch size: {bs}")

            # setup args, kwargs
            inputs_truncated = [in_buffer[:bs] for in_buffer in self._input_buffers]
            args, kwargs = self._in_spec.unflatten(inputs_truncated + args_static)

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
            return self.gm_compiled(*args, **kwargs)

        # Calculate rounded-up shapes for each input
        rounded_shapes = [
            (self.round_to_cuda_batch_size(input.shape[0]),) + input.shape[1:]
            for input in args_batched
        ]
        combined_shape = sum(rounded_shapes, start=())

        # regular forward for non-matching shapes
        if combined_shape not in self.graphs:
            return self.gm_compiled(*args, **kwargs)

        # copy inputs to input buffers
        for i, input_tensor in enumerate(args_batched):
            self._input_buffers[i][: input_tensor.shape[0]] = input_tensor

        # run forward pass via graph
        self.graphs[combined_shape].replay()

        # retrieve output from buffer, cut to batch size, and unflatten
        bs = args_batched[0].shape[0]
        out_flat = [o_b[:bs].detach().clone() for o_b in self._out_buffer_flat]
        return self._out_spec.unflatten(out_flat)


@BackendRegistry.register("torch-opt")
class TorchOptCompiler(BackendCompiler):
    @torch.inference_mode()
    def compile(self) -> CompiledGraph:
        compiled_gm = CompiledGraph(
            self.gm,
            max_batch_size=self.max_batch_size,
            cuda_graph_batch_sizes=self.compiler_kwargs.get("cuda_graph_batch_sizes"),
            num_batched_inputs=self.compiler_kwargs.get("num_batched_inputs"),
        )

        # try capturing cudagraph
        # if self.args is not None or self.kwargs is not None:
        #    compiled_gm.capture_graph(*self.args, **self.kwargs)

        return compiled_gm
