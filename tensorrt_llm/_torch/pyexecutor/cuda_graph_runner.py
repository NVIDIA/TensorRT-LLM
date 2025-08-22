import threading
from typing import Any, Callable, Dict, NamedTuple, Optional, Tuple

import torch

from tensorrt_llm.logger import logger
from tensorrt_llm.profiler import device_memory_info

from ..attention_backend.interface import AttentionMetadata
from ..speculative.interface import SpecMetadata
from ..utils import make_weak_ref, set_piecewise_cuda_graph_flag


class MemoryStats(NamedTuple):
    """Memory usage statistics"""
    inside_torch: int
    outside_torch_method1: int
    outside_torch_method2: int
    total_used: int
    total_free: int


def get_memory_stats() -> MemoryStats:
    """Get current GPU memory usage statistics"""
    # Get memory info from device_memory_info
    mem_used, mem_free, mem_total = device_memory_info()

    # Get PyTorch allocated memory
    mem_inside_torch = torch.cuda.memory_stats()["allocated_bytes.all.current"]

    # Method 1: using device_memory_info
    mem_outside_torch_method1 = (
        mem_used - mem_inside_torch) if mem_used > mem_inside_torch else 0

    # Method 2: using torch.cuda.mem_get_info
    mem_free_cuda, total_gpu_memory = torch.cuda.mem_get_info()
    total_used_bytes = total_gpu_memory - mem_free_cuda
    mem_outside_torch_method2 = (
        total_used_bytes -
        mem_inside_torch) if total_used_bytes > mem_inside_torch else 0

    return MemoryStats(inside_torch=mem_inside_torch,
                       outside_torch_method1=mem_outside_torch_method1,
                       outside_torch_method2=mem_outside_torch_method2,
                       total_used=mem_used,
                       total_free=mem_free)


def log_memory_stats(description: str, stats: MemoryStats):
    """Log memory statistics with a description"""
    logger.info(
        f"{description}: "
        f"inside torch: {stats.inside_torch / 1024**3:.2f} GB, "
        f"outside torch (method1): {stats.outside_torch_method1 / 1024**3:.2f} GB, "
        f"outside torch (method2): {stats.outside_torch_method2 / 1024**3:.2f} GB, "
        f"total: {stats.total_used / 1024**3:.2f} GB, "
        f"free: {stats.total_free / 1024**3:.2f} GB")


def log_memory_comparison(description: str, before: MemoryStats,
                          after: MemoryStats):
    """Log memory statistics comparison with current values and changes"""
    # Calculate changes
    inside_torch_change = after.inside_torch - before.inside_torch
    outside_torch_change_method1 = after.outside_torch_method1 - before.outside_torch_method1
    outside_torch_change_method2 = after.outside_torch_method2 - before.outside_torch_method2
    total_change = after.total_used - before.total_used

    logger.info(
        f"{description}: "
        f"CURRENT - inside torch: {after.inside_torch / 1024**3:.2f} GB, "
        f"outside torch (method1): {after.outside_torch_method1 / 1024**3:.2f} GB, "
        f"outside torch (method2): {after.outside_torch_method2 / 1024**3:.2f} GB, "
        f"total: {after.total_used / 1024**3:.2f} GB, "
        f"free: {after.total_free / 1024**3:.2f} GB | "
        f"CHANGES - inside torch: {inside_torch_change / 1024**3:.2f} GB, "
        f"outside torch (method1): {outside_torch_change_method1 / 1024**3:.2f} GB, "
        f"outside torch (method2): {outside_torch_change_method2 / 1024**3:.2f} GB, "
        f"total: {total_change / 1024**3:.2f} GB")


class graph_capturing_local(threading.local):

    def __init__(self):
        self.is_graph_capturing = False


_local = graph_capturing_local()


def set_graph_capturing(enable: bool):
    _local.is_graph_capturing = enable


def is_graph_capturing() -> bool:
    return _local.is_graph_capturing


class DecodingCUDAGraphRunner:

    def __init__(
        self,
        batch_size: int,
        device: str,
        attn_metadata: AttentionMetadata,
        spec_metadata: Optional[SpecMetadata] = None,
        use_mrope: bool = False,
    ) -> None:
        """
        Stores a CUDA graph and its associated input buffers.

        Each CUDA graph runner is associated with an AttentionMetadata object
        if flashinfer is being used. Make sure to call attn_metadata.prepare()
        before run()!

        Note that torch.compile w/ mode reduce-overhead supports CUDA graphs
        with memory pool sharing. However, we have our own manager here because,
        at the time of writing this, torch.compile takes way too long to warmup
        graphs compared to doing it manually (not to mention, custom ops from
        e.g. FlashInfer cause graph breaks).
        """
        self.batch_size = batch_size

        # [CUDA graph spec decode padding]
        # We pad input IDs/position IDs to the maximum draft length (token per request).
        # We're forced to do this because we cannot reallocate inputs over many graph runs.
        token_per_request = spec_metadata.max_draft_len + 1 if spec_metadata is not None else 1

        # Using ones instead of zeros prevents NaNs in e.g. Deepseek
        self.input_ids = torch.ones((batch_size * token_per_request, ),
                                    device=device,
                                    dtype=torch.int32)
        self.position_ids = torch.zeros((1, batch_size * token_per_request),
                                        device=device,
                                        dtype=torch.int32)
        self.mrope_position_deltas = torch.zeros(
            (batch_size,
             1), device=device, dtype=torch.int32) if use_mrope else None

        self.attn_metadata = attn_metadata
        self.spec_metadata = spec_metadata
        self._output = None
        self._graph = None
        self.optional_extra_model_inputs = ["mrope_position_deltas"]

    def __del__(self):
        self._graph.reset()

    def capture(
        self,
        forward_fn: Callable[[Dict[str, Any]], torch.Tensor],
        pool: Optional[Tuple[int, int]] = None,
    ) -> Tuple[int, int]:
        # Memory statistics at capture start
        stats_start = get_memory_stats()
        log_memory_stats(
            f"CUDA graph capture START (batch_size={self.batch_size})",
            stats_start)

        self._graph = torch.cuda.CUDAGraph()
        inputs = {
            "attn_metadata": self.attn_metadata,
            "input_ids": self.input_ids,
            "position_ids": self.position_ids,
            "inputs_embeds": None,
            "spec_metadata": self.spec_metadata,
            "mrope_position_deltas": self.mrope_position_deltas,
        }

        # Memory statistics before warmup
        stats_before_warmup = get_memory_stats()

        # We have to do warm up runs to initialize PyTorch's
        # internal states according to the docs:
        # https://pytorch.org/docs/stable/notes/cuda.html#cuda-graph-semantics
        # This also lets us initialize states in the attn_metadata.
        set_graph_capturing(True)
        set_piecewise_cuda_graph_flag(False)
        for _ in range(2):
            forward_fn(inputs)

        # Memory statistics after warmup
        stats_after_warmup = get_memory_stats()
        log_memory_comparison("CUDA graph warmup completed",
                              stats_before_warmup, stats_after_warmup)

        # Memory statistics before graph capture
        stats_before_capture = get_memory_stats()

        with torch.cuda.graph(self._graph, pool=pool):
            output = forward_fn(inputs)

        # Memory statistics after graph capture
        stats_after_capture = get_memory_stats()
        log_memory_comparison("CUDA graph capture completed",
                              stats_before_capture, stats_after_capture)

        set_graph_capturing(False)
        set_piecewise_cuda_graph_flag(True)
        # Mark weak ref here. The output tensor should be freed properly.
        self._output = make_weak_ref(output)

        # Memory statistics at capture end
        stats_end = get_memory_stats()
        log_memory_comparison(
            f"CUDA graph capture COMPLETE (batch_size={self.batch_size})",
            stats_start, stats_end)

        return self._graph.pool()

    def needs_capture(self) -> bool:
        return self._output is None

    def run(self, inputs: Dict[str, Any]) -> torch.Tensor:
        assert "input_ids" in inputs
        assert "position_ids" in inputs
        assert "attn_metadata" in inputs

        attn_metadata = inputs["attn_metadata"]
        assert attn_metadata is self.attn_metadata, (
            "attn_metadata does not match the attn_metadata instance that was used to "
            "capture this graph.")

        if "spec_metadata" in inputs:
            spec_metadata = inputs["spec_metadata"]
            assert spec_metadata is self.spec_metadata, (
                "spec_metadata does not match the spec_metadata instance that was used to "
                "capture this graph.")

        input_ids = inputs["input_ids"]
        position_ids = inputs["position_ids"]
        seqlen = input_ids.shape[0]
        self.input_ids[:seqlen].copy_(input_ids)
        self.position_ids[:, :seqlen].copy_(position_ids)
        if "mrope_position_deltas" in inputs:
            self.mrope_position_deltas[:self.batch_size].copy_(
                inputs["mrope_position_deltas"])

        assert self._output is not None and self._graph is not None
        self._graph.replay()
        return self._output
