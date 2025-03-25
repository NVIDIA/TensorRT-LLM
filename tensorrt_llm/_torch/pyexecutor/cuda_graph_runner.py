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

        token_per_request = spec_metadata.max_draft_tokens + 1 if spec_metadata is not None else 1

        # Using ones instead of zeros prevents NaNs in e.g. Deepseek
        self.input_ids = torch.ones((batch_size * token_per_request, ),
                                    device=device,
                                    dtype=torch.int64)
        self.position_ids = torch.zeros((1, batch_size * token_per_request),
                                        device=device,
                                        dtype=torch.int64)

        self.attn_metadata = attn_metadata
        self.spec_metadata = spec_metadata
        self.pipeline_interface = pipeline_interface
        self.has_pp = has_pp
        self._output = None
        self._graph = None

    def capture(
        self,
        forward_fn: Callable[[Dict[str, Any]], torch.Tensor],
        pool: Optional[Tuple[int, int]] = None,
    ) -> Tuple[int, int]:
        self._graph = torch.cuda.CUDAGraph()
        inputs = {
            "attn_metadata": self.attn_metadata,
            "input_ids": self.input_ids,
            "position_ids": self.position_ids,
            "inputs_embeds": None,
            "spec_metadata": self.spec_metadata,
        }
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
        self.input_ids.copy_(input_ids)
        self.position_ids.copy_(position_ids)

        if self.pipeline_interface is not None:
            assert "pipeline_interface" in inputs
            pipeline_interface = inputs["pipeline_interface"]
            for key in ["hidden_states", "residual"]:
                self.pipeline_interface[key].copy_(pipeline_interface[key])

        assert self._output is not None and self._graph is not None
        self._graph.replay()
        return self._output
