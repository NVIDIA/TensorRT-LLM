import threading
from typing import Any, Callable, Dict, Optional, Tuple

import torch

from tensorrt_llm.logger import logger
from tensorrt_llm.profiler import device_memory_info

from ..attention_backend.interface import AttentionMetadata
from ..speculative.interface import SpecMetadata
from ..utils import make_weak_ref, set_piecewise_cuda_graph_flag


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
        mem_used_start, mem_free_start, mem_total = device_memory_info()
        mem_used_inside_torch_start = torch.cuda.memory_stats(
        )["allocated_bytes.all.current"]

        # Method 1: device_memory_info
        mem_used_outside_torch_method1_start = (
            mem_used_start - mem_used_inside_torch_start
        ) if mem_used_start > mem_used_inside_torch_start else 0

        # Method 2: torch.cuda.mem_get_info
        mem_free_cuda_start, total_gpu_memory_start = torch.cuda.mem_get_info()
        total_used_bytes_start = total_gpu_memory_start - mem_free_cuda_start
        mem_used_outside_torch_method2_start = (
            total_used_bytes_start - mem_used_inside_torch_start
        ) if total_used_bytes_start > mem_used_inside_torch_start else 0

        logger.info(
            f"CUDA graph capture START (batch_size={self.batch_size}): "
            f"inside torch: {mem_used_inside_torch_start / 1024**3:.2f} GB, "
            f"outside torch (method1): {mem_used_outside_torch_method1_start / 1024**3:.2f} GB, "
            f"outside torch (method2): {mem_used_outside_torch_method2_start / 1024**3:.2f} GB, "
            f"total: {mem_used_start / 1024**3:.2f} GB, free: {mem_free_start / 1024**3:.2f} GB"
        )

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
        mem_used_before_warmup, _, _ = device_memory_info()
        mem_used_inside_torch_before_warmup = torch.cuda.memory_stats(
        )["allocated_bytes.all.current"]

        # We have to do warm up runs to initialize PyTorch's
        # internal states according to the docs:
        # https://pytorch.org/docs/stable/notes/cuda.html#cuda-graph-semantics
        # This also lets us initialize states in the attn_metadata.
        set_graph_capturing(True)
        set_piecewise_cuda_graph_flag(False)
        for _ in range(2):
            forward_fn(inputs)

        # Memory statistics after warmup
        mem_used_after_warmup, mem_free_after_warmup, _ = device_memory_info()
        mem_used_inside_torch_after_warmup = torch.cuda.memory_stats(
        )["allocated_bytes.all.current"]

        # Method 1: device_memory_info
        mem_used_outside_torch_method1_after_warmup = (
            mem_used_after_warmup - mem_used_inside_torch_after_warmup
        ) if mem_used_after_warmup > mem_used_inside_torch_after_warmup else 0

        # Method 2: torch.cuda.mem_get_info
        mem_free_cuda_after_warmup, total_gpu_memory_after_warmup = torch.cuda.mem_get_info(
        )
        total_used_bytes_after_warmup = total_gpu_memory_after_warmup - mem_free_cuda_after_warmup
        mem_used_outside_torch_method2_after_warmup = (
            total_used_bytes_after_warmup - mem_used_inside_torch_after_warmup
        ) if total_used_bytes_after_warmup > mem_used_inside_torch_after_warmup else 0

        warmup_mem_change = mem_used_after_warmup - mem_used_before_warmup
        warmup_inside_torch_change = mem_used_inside_torch_after_warmup - mem_used_inside_torch_before_warmup

        logger.info(
            f"CUDA graph warmup completed: "
            f"CURRENT - inside torch: {mem_used_inside_torch_after_warmup / 1024**3:.2f} GB, "
            f"outside torch (method1): {mem_used_outside_torch_method1_after_warmup / 1024**3:.2f} GB, "
            f"outside torch (method2): {mem_used_outside_torch_method2_after_warmup / 1024**3:.2f} GB, "
            f"total: {mem_used_after_warmup / 1024**3:.2f} GB, free: {mem_free_after_warmup / 1024**3:.2f} GB | "
            f"CHANGES - total: {warmup_mem_change / 1024**3:.2f} GB, "
            f"inside torch: {warmup_inside_torch_change / 1024**3:.2f} GB")

        # Memory statistics before graph capture
        mem_used_before_capture, _, _ = device_memory_info()
        mem_used_inside_torch_before_capture = torch.cuda.memory_stats(
        )["allocated_bytes.all.current"]

        with torch.cuda.graph(self._graph, pool=pool):
            output = forward_fn(inputs)

        # Memory statistics after graph capture
        mem_used_after_capture, mem_free_after_capture, _ = device_memory_info()
        mem_used_inside_torch_after_capture = torch.cuda.memory_stats(
        )["allocated_bytes.all.current"]

        # Method 1: device_memory_info
        mem_used_outside_torch_method1_after_capture = (
            mem_used_after_capture - mem_used_inside_torch_after_capture
        ) if mem_used_after_capture > mem_used_inside_torch_after_capture else 0

        # Method 2: torch.cuda.mem_get_info
        mem_free_cuda_after_capture, total_gpu_memory_after_capture = torch.cuda.mem_get_info(
        )
        total_used_bytes_after_capture = total_gpu_memory_after_capture - mem_free_cuda_after_capture
        mem_used_outside_torch_method2_after_capture = (
            total_used_bytes_after_capture - mem_used_inside_torch_after_capture
        ) if total_used_bytes_after_capture > mem_used_inside_torch_after_capture else 0

        capture_mem_change = mem_used_after_capture - mem_used_before_capture
        capture_inside_torch_change = mem_used_inside_torch_after_capture - mem_used_inside_torch_before_capture

        logger.info(
            f"CUDA graph capture completed: "
            f"CURRENT - inside torch: {mem_used_inside_torch_after_capture / 1024**3:.2f} GB, "
            f"outside torch (method1): {mem_used_outside_torch_method1_after_capture / 1024**3:.2f} GB, "
            f"outside torch (method2): {mem_used_outside_torch_method2_after_capture / 1024**3:.2f} GB, "
            f"total: {mem_used_after_capture / 1024**3:.2f} GB, free: {mem_free_after_capture / 1024**3:.2f} GB | "
            f"CHANGES - total: {capture_mem_change / 1024**3:.2f} GB, "
            f"inside torch: {capture_inside_torch_change / 1024**3:.2f} GB")

        set_graph_capturing(False)
        set_piecewise_cuda_graph_flag(True)
        # Mark weak ref here. The output tensor should be freed properly.
        self._output = make_weak_ref(output)

        # Memory statistics at capture end
        mem_used_end, mem_free_end, mem_total = device_memory_info()
        mem_used_inside_torch_end = torch.cuda.memory_stats(
        )["allocated_bytes.all.current"]

        # Method 1: device_memory_info
        mem_used_outside_torch_method1_end = (
            mem_used_end - mem_used_inside_torch_end
        ) if mem_used_end > mem_used_inside_torch_end else 0

        # Method 2: torch.cuda.mem_get_info
        mem_free_cuda_end, total_gpu_memory_end = torch.cuda.mem_get_info()
        total_used_bytes_end = total_gpu_memory_end - mem_free_cuda_end
        mem_used_outside_torch_method2_end = (
            total_used_bytes_end - mem_used_inside_torch_end
        ) if total_used_bytes_end > mem_used_inside_torch_end else 0

        # Calculate total changes
        total_mem_change = mem_used_end - mem_used_start
        total_inside_torch_change = mem_used_inside_torch_end - mem_used_inside_torch_start
        total_outside_torch_change_method1 = mem_used_outside_torch_method1_end - mem_used_outside_torch_method1_start
        total_outside_torch_change_method2 = mem_used_outside_torch_method2_end - mem_used_outside_torch_method2_start

        logger.info(
            f"CUDA graph capture COMPLETE (batch_size={self.batch_size}): "
            f"FINAL CURRENT - inside torch: {mem_used_inside_torch_end / 1024**3:.2f} GB, "
            f"outside torch (method1): {mem_used_outside_torch_method1_end / 1024**3:.2f} GB, "
            f"outside torch (method2): {mem_used_outside_torch_method2_end / 1024**3:.2f} GB, "
            f"total: {mem_used_end / 1024**3:.2f} GB, free: {mem_free_end / 1024**3:.2f} GB | "
            f"TOTAL CHANGES - inside torch: {total_inside_torch_change / 1024**3:.2f} GB, "
            f"outside torch (method1): {total_outside_torch_change_method1 / 1024**3:.2f} GB, "
            f"outside torch (method2): {total_outside_torch_change_method2 / 1024**3:.2f} GB, "
            f"total memory: {total_mem_change / 1024**3:.2f} GB")

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
