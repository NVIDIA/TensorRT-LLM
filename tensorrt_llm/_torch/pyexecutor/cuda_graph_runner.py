import threading
from typing import Any, Callable, Dict, Optional, Tuple

import torch

from ..attention_backend.interface import AttentionMetadata
from ..pipeline_interface import PipelineInterface
from ..speculative.interface import SpecMetadata
from ..utils import make_weak_ref

_local = threading.local()


def set_graph_capturing(enable: bool):
    _local.is_graph_capturing = enable


def is_graph_capturing() -> bool:
    if not hasattr(_local, 'is_graph_capturing'):
        return False
    return _local.is_graph_capturing


class DecodingCUDAGraphRunner:

    def __init__(
        self,
        batch_size: int,
        device: str,
        attn_metadata: AttentionMetadata,
        spec_metadata: Optional[SpecMetadata] = None,
        pipeline_interface: Optional[PipelineInterface] = None,
        has_pp: bool = False,
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
        token_per_request = spec_metadata.max_draft_tokens + 1 if spec_metadata is not None else 1

        # Using ones instead of zeros prevents NaNs in e.g. Deepseek
        self.input_ids = torch.ones((batch_size * token_per_request, ),
                                    device=device,
                                    dtype=torch.int64)
        self.position_ids = torch.zeros((1, batch_size * token_per_request),
                                        device=device,
                                        dtype=torch.int64)

        self.extra_model_inputs = {}
        self.attn_metadata = attn_metadata
        self.spec_metadata = spec_metadata
        self.pipeline_interface = pipeline_interface
        self.has_pp = has_pp
        self._output = None
        self._graph = None

    def __del__(self):
        self._graph.reset()

    def capture(
        self,
        forward_fn: Callable[[Dict[str, Any]], torch.Tensor],
        pool: Optional[Tuple[int, int]] = None,
        extra_model_inputs: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[int, int]:
        """
        Captures a CUDA graph by calling forward_fn(inputs),
        where inputs is extra_model_inputs + this graph runner's
        input_ids, position_ids, spec_metadata and attn_metadata.

        Extra model inputs have the following semantics if
        the extra input is a tensor (or collection of
        tensors). The CUDA graph runner will create a buffer
        of the same shape/dtype/device, and subsequent calls to run() will
        require this extra model input. Input tensors will be
        copied into the buffer that this CUDA graph runner owns.
        This implies that these buffers *must* have static shapes for
        this CUDA graph's batch size.
        """
        self._graph = torch.cuda.CUDAGraph()
        inputs = {
            "attn_metadata": self.attn_metadata,
            "input_ids": self.input_ids,
            "position_ids": self.position_ids,
            "inputs_embeds": None,
            "spec_metadata": self.spec_metadata,
        }
        if extra_model_inputs is not None:
            for key, tensor in extra_model_inputs.items():
                new_tensor = tensor.clone()
                inputs[key] = new_tensor
                self.extra_model_inputs[key] = new_tensor

        if self.has_pp:
            inputs["pipeline_interface"] = self.pipeline_interface

        # We have to do warm up runs to initialize PyTorch's
        # internal states according to the docs:
        # https://pytorch.org/docs/stable/notes/cuda.html#cuda-graph-semantics
        # This also lets us initialize states in the attn_metadata.
        set_graph_capturing(True)
        for _ in range(2):
            forward_fn(inputs)
        with torch.cuda.graph(self._graph, pool=pool):
            output = forward_fn(inputs)
        set_graph_capturing(False)
        # Mark weak ref here. The output tensor should be freed properly.
        self._output = make_weak_ref(output)
        return self._graph.pool()

    def needs_capture(self) -> bool:
        return self._output is None

    def run(
        self,
        inputs: Dict[str, Any],
        extra_model_inputs: Optional[Dict[str, torch.Tensor]] = None
    ) -> torch.Tensor:
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

        if self.pipeline_interface is not None:
            assert "pipeline_interface" in inputs
            pipeline_interface = inputs["pipeline_interface"]
            for key in ["hidden_states", "residual"]:
                self.pipeline_interface[key].copy_(pipeline_interface[key])

        if self.extra_model_inputs:
            assert extra_model_inputs is not None, "Model was captured with extra model inputs, so extra_model_inputs must be provided to run()"
            for key in self.extra_model_inputs:
                assert key in extra_model_inputs, f"Graph runner is missing extra input {key}"
                dst_tensor = self.extra_model_inputs[key]
                dst_tensor.copy_(extra_model_inputs[key])

        assert self._output is not None and self._graph is not None
        self._graph.replay()
        return self._output
