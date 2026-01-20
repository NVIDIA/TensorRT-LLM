"""Compile backend with cudagraph."""

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.cuda import CUDAGraph
from torch.fx._pytree import tree_flatten_spec
from torch.utils._pytree import PyTree, TreeSpec, tree_flatten

from tensorrt_llm._torch.autotuner import autotune

from ...utils.cuda_graph import CudaGraphWarmUpPhase
from ...utils.logger import ad_logger
from ..compiler import CompileBackendRegistry, CompilerBackend, GetArgsKwargsForBatchSize


def _args_kwargs_flatten_spec(in_spec: TreeSpec, *args, **kwargs) -> List[Any]:
    """Flatten inputs according to provided in_spec."""
    all_args: PyTree = (args, kwargs)
    return tree_flatten_spec(all_args, in_spec)


def _args_kwargs_flatten(*args, **kwargs) -> Tuple[List[Any], TreeSpec]:
    """Flatten inputs and return flattened inputs together with the TreeSpec."""
    all_args: PyTree = (args, kwargs)
    return tree_flatten(all_args)


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


@CompileBackendRegistry.register("torch-cudagraph")
class TorchCudagraphCompiler(CompilerBackend):
    """Compiler that uses only CUDA graphs."""

    def __init__(
        self,
        *args_for_init,
        cuda_graph_batch_sizes: Optional[List[int]] = None,
        num_batched_inputs: int = 1,
        get_args_kwargs_for_compile: GetArgsKwargsForBatchSize = None,
        **kwargs_for_init,
    ):
        super().__init__(*args_for_init, **kwargs_for_init)
        self.num_batched_inputs = num_batched_inputs
        self.cuda_graph_batch_sizes = cuda_graph_batch_sizes or []
        self.get_args_kwargs_for_compile = get_args_kwargs_for_compile

    @torch.inference_mode()
    def compile(self) -> CapturedGraph:
        captured_model = CapturedGraph(self.model, num_batched_inputs=self.num_batched_inputs)

        # try capturing cudagraph
        assert self.get_args_kwargs_for_compile is not None, (
            "get_args_kwargs_for_compile must be provided"
        )
        captured_model.capture_graph(self.get_args_kwargs_for_compile, self.cuda_graph_batch_sizes)

        return captured_model
