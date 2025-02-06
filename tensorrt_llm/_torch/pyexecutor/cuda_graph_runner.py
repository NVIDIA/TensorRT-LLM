from typing import Any, Callable, Dict, Optional, Tuple

import torch

from tensorrt_llm._torch.attention_backend.interface import AttentionMetadata


class DecodingCUDAGraphRunner:

    def __init__(
        self,
        batch_size: int,
        device: str,
        attn_metadata: AttentionMetadata,
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

        self.input_ids = torch.zeros((batch_size, ),
                                     device=device,
                                     dtype=torch.int64)
        self.position_ids = torch.zeros((1, batch_size),
                                        device=device,
                                        dtype=torch.int64)

        self.attn_metadata = attn_metadata

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
            "inputs_embeds": None
        }
        # We have to do warm up runs to initialize PyTorch's
        # internal states according to the docs:
        # https://pytorch.org/docs/stable/notes/cuda.html#cuda-graph-semantics
        # This also lets us initialize states in the attn_metadata.
        for _ in range(2):
            forward_fn(inputs)

        with torch.cuda.graph(self._graph, pool=pool):
            self._output = forward_fn(inputs)
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

        input_ids = inputs["input_ids"]
        position_ids = inputs["position_ids"]
        self.input_ids.copy_(input_ids)
        self.position_ids.copy_(position_ids)
        assert self._output is not None and self._graph is not None
        self._graph.replay()
        return self._output
