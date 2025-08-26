from typing import Any, Callable, Dict, Optional, Tuple

import torch

from ...inputs.multimodal import MultimodalParams
from ..attention_backend.interface import AttentionMetadata
from ..modules.multi_stream_utils import with_multi_stream
from ..speculative.interface import SpecMetadata
from ..utils import make_weak_ref, piecewise_cuda_graph


class DecodingCUDAGraphRunner:

    def __init__(
        self,
        batch_size: int,
        device: str,
        attn_metadata: AttentionMetadata,
        spec_metadata: Optional[SpecMetadata] = None,
        use_mrope: bool = False,
        max_beam_width: int = 1,
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
        self.max_beam_width = max_beam_width
        # [CUDA graph spec decode padding]
        # We pad input IDs/position IDs to the maximum draft length (token per request).
        # We're forced to do this because we cannot reallocate inputs over many graph runs.
        token_per_request = spec_metadata.max_draft_len + 1 if spec_metadata is not None else 1

        # Using ones instead of zeros prevents NaNs in e.g. Deepseek
        self.input_ids = torch.ones(
            (batch_size * max_beam_width * token_per_request, ),
            device=device,
            dtype=torch.int32)
        self.use_mrope = use_mrope

        if self.use_mrope:
            self.position_ids = torch.zeros(
                (3, 1, batch_size * max_beam_width * token_per_request),
                device=device,
                dtype=torch.int32)
        else:
            self.position_ids = torch.zeros(
                (1, batch_size * max_beam_width * token_per_request),
                device=device,
                dtype=torch.int32)

        self.multimodal_params = [
            MultimodalParams(
                multimodal_data={
                    "mrope_config": {
                        "mrope_position_deltas":
                        torch.zeros((1, 1), device=device, dtype=torch.int32)
                    }
                }) for _ in range(batch_size)
        ] if self.use_mrope else []
        self.attn_metadata = attn_metadata
        self.spec_metadata = spec_metadata
        self._output = None
        self._graph = None

    def __del__(self):
        self._graph.reset()

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
            "multimodal_params": self.multimodal_params,
        }

        # We have to do warm up runs to initialize PyTorch's
        # internal states according to the docs:
        # https://pytorch.org/docs/stable/notes/cuda.html#cuda-graph-semantics
        # This also lets us initialize states in the attn_metadata.
        with with_multi_stream(True), piecewise_cuda_graph(False):
            for _ in range(2):
                forward_fn(inputs)
            with torch.cuda.graph(self._graph, pool=pool):
                output = forward_fn(inputs)
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
        seqlen = input_ids.shape[0]
        self.input_ids[:seqlen].copy_(input_ids)
        if self.use_mrope:
            self.position_ids[:, :, :seqlen].copy_(position_ids)
            for i, multimodal_param in enumerate(inputs['multimodal_params']):
                # NOTE: Currently, we only need 'mrope_position_deltas' on generation phase for multimodal models.
                self.multimodal_params[i].multimodal_data['mrope_config'][
                    'mrope_position_deltas'].copy_(
                        multimodal_param.multimodal_data['mrope_config']
                        ['mrope_position_deltas'],
                        non_blocking=True)
        else:
            self.position_ids[:, :seqlen].copy_(position_ids)

        assert self._output is not None and self._graph is not None
        self._graph.replay()
        return self._output
