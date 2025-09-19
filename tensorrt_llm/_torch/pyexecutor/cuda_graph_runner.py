import bisect
import contextlib
from typing import Any, Callable, Dict, Optional, Tuple

import torch

from tensorrt_llm.mapping import Mapping

from ..attention_backend.interface import AttentionMetadata
from ..distributed import MPIDist
from ..expert_statistic import ExpertStatistic
from ..modules.multi_stream_utils import with_multi_stream
from ..speculative import SpecMetadata
from ..utils import make_weak_ref, piecewise_cuda_graph
from .resource_manager import ResourceManager, ResourceManagerType
from .scheduler import ScheduledRequests

# A large prime number used for dummy request IDs to avoid collisions
CUDA_GRAPH_DUMMY_REQUEST_ID = (1 << 64) - 1


class CUDAGraphRunner:
    """
    Manages the lifecycle and execution of CUDA graphs for the model engine.

    This unified class handles high-level orchestration (padding, eligibility)
    and low-level execution (capturing, resource management, replaying) for
    multiple graphs, keyed by (batch size, draft_len).
    """
    WARMUP_STEPS = 2

    def __init__(
        self,
        *,
        use_cuda_graph: bool,
        cuda_graph_padding_enabled: bool,
        supported_batch_sizes: list[int],
        max_supported_batch_size: int,
        max_batch_size: int,
        max_beam_width: int,
        max_draft_len: int,
        use_mrope: bool,
        spec_config: Optional["DecodingBaseConfig"],
        cuda_graph_mem_pool: Optional[int],
        enable_attention_dp: bool,
        mapping: Mapping,
        dist: Optional[MPIDist],
        kv_cache_manager_key: ResourceManagerType,
    ):
        # --- High-level configuration passed from the engine ---
        self.enabled = use_cuda_graph
        self.padding_enabled = cuda_graph_padding_enabled
        self.supported_batch_sizes = supported_batch_sizes
        self.max_supported_batch_size = max_supported_batch_size
        self.max_batch_size = max_batch_size
        self.max_beam_width = max_beam_width
        self.max_draft_len = max_draft_len
        self.use_mrope = use_mrope
        self.spec_config = spec_config
        self.enable_attention_dp = enable_attention_dp
        self.mapping = mapping
        self.dist = dist
        self.kv_cache_manager_key = kv_cache_manager_key

        # --- Internal state ---
        self.graphs: Dict[Tuple[int, int], torch.cuda.CUDAGraph] = {}
        self.graph_outputs: Dict[Tuple[int, int],
                                 Callable[[], Optional[torch.Tensor]]] = {}
        self.graph_metadata: Dict[Tuple[int, int], Dict[str, Any]] = {}
        self.memory_pool = cuda_graph_mem_pool
        self.padding_dummy_request: Optional["Request"] = None

    def __del__(self):
        self.clear()

    def maybe_get_cuda_graph(
        self,
        batch: ScheduledRequests,
        iter_counter: int,
        is_spec_decode: bool,
        attn_metadata: AttentionMetadata,
        spec_metadata: Optional[SpecMetadata],
        draft_tokens_cuda: torch.Tensor,
    ) -> Tuple[bool, Optional[AttentionMetadata], Optional[SpecMetadata]]:
        """
        Determines if the current batch can be run with a CUDA graph.

        Returns a tuple containing:
        - A boolean indicating if a graph can be used.
        - The attn_metadata for the graph, if applicable.
        - The spec_metadata for the graph, if applicable.
        """
        # disable when doing statistic
        if hasattr(self,
                   'iter_counter') and ExpertStatistic.set_iter(iter_counter):
            return False, None, None

        can_run_cuda_graph = batch.can_run_cuda_graph
        batch_size = batch.batch_size
        if self.enabled and self.enable_attention_dp and self.mapping.tp_size > 1:
            all_can_graph_batch = self.dist.tp_allgather(
                [can_run_cuda_graph, batch_size])
            is_all_gen_only = all(all_can_graph[0]
                                  for all_can_graph in all_can_graph_batch)
            all_batch_size_equal = all(
                all_gen_only[1] == all_can_graph_batch[0][1]
                for all_gen_only in all_can_graph_batch)

            if not is_all_gen_only or not all_batch_size_equal:
                return False, None, None

        if not self.enabled or not can_run_cuda_graph:
            return False, None, None

        draft_len = self.spec_config.max_draft_len if is_spec_decode else 0
        key = (batch_size, draft_len)

        if key in self.graphs:
            return True, self.graph_metadata[key][
                "attn_metadata"], self.graph_metadata[key]["spec_metadata"]

        if batch_size not in self.supported_batch_sizes:
            return False, None, None

        num_sequences_in_batch = batch_size * self.max_beam_width
        graph_attn_metadata = attn_metadata.create_cuda_graph_metadata(
            num_sequences_in_batch, False, draft_len)
        assert graph_attn_metadata.is_cuda_graph

        graph_spec_metadata = None
        if is_spec_decode and spec_metadata:
            graph_spec_metadata = spec_metadata.create_cuda_graph_metadata(
                num_sequences_in_batch)
            graph_spec_metadata.draft_tokens = draft_tokens_cuda

        return True, graph_attn_metadata, graph_spec_metadata

    def needs_capture(self, batch_size: int, is_spec_decode: bool) -> bool:
        draft_len = self.spec_config.max_draft_len if is_spec_decode else 0
        return (batch_size, draft_len) not in self.graph_outputs

    def capture(self,
                batch_size: int,
                is_spec_decode: bool,
                forward_fn: Callable,
                initial_inputs: Dict[str, Any],
                postprocess_fn: Optional[Callable] = None):
        """Captures the forward pass for a given batch size."""
        draft_len = self.spec_config.max_draft_len if is_spec_decode else 0
        key = (batch_size, draft_len)

        # [CUDA graph spec decode padding]
        # We pad input IDs/position IDs to the maximum draft length (token per request).
        # We're forced to do this because we cannot reallocate inputs over many graph runs.
        token_per_request = draft_len + 1
        num_tokens_for_capture = (batch_size * self.max_beam_width *
                                  token_per_request)

        sliced_static_tensors = {
            "input_ids":
            self.shared_static_tensors["input_ids"][:num_tokens_for_capture],
            "position_ids":
            self.shared_static_tensors["position_ids"]
            [:, :num_tokens_for_capture],
        }
        if self.use_mrope:
            sliced_static_tensors["mrope_position_deltas"] = torch.zeros(
                (batch_size, 1), device="cuda", dtype=torch.int32)

        # Use the sliced tensors for capture
        capture_inputs = initial_inputs.copy()
        capture_inputs.update(sliced_static_tensors)

        self.graph_metadata[key] = {
            "attn_metadata": initial_inputs["attn_metadata"],
            "spec_metadata": initial_inputs.get("spec_metadata", None),
        }

        # We have to do warm up runs to initialize PyTorch's
        # internal states according to the docs:
        # https://pytorch.org/docs/stable/notes/cuda.html#cuda-graph-semantics
        # This also lets us initialize states in the attn_metadata.
        graph = torch.cuda.CUDAGraph()
        with with_multi_stream(True), piecewise_cuda_graph(False):
            for _ in range(self.WARMUP_STEPS):
                forward_fn(capture_inputs)
                if postprocess_fn is not None:
                    postprocess_fn(capture_inputs)
            with torch.cuda.graph(graph, pool=self.memory_pool):
                output = forward_fn(capture_inputs)
            if postprocess_fn is not None:
                postprocess_fn(capture_inputs)

        self.graphs[key] = graph
        self.graph_outputs[key] = make_weak_ref(output)
        self.memory_pool = graph.pool()

    def replay(self, batch_size: int, is_spec_decode: bool,
               current_inputs: Dict[str, Any]) -> Optional[torch.Tensor]:
        """Replays a previously captured graph."""
        draft_len = self.spec_config.max_draft_len if is_spec_decode else 0
        key = (batch_size, draft_len)

        stored_meta = self.graph_metadata[key]
        assert current_inputs["attn_metadata"] is stored_meta["attn_metadata"]
        if stored_meta["spec_metadata"] is not None:
            assert current_inputs.get(
                "spec_metadata") is stored_meta["spec_metadata"]

        static_tensors = self.shared_static_tensors

        input_ids = current_inputs["input_ids"]
        seqlen = input_ids.shape[0]
        static_tensors["input_ids"][:seqlen].copy_(input_ids)

        position_ids = current_inputs["position_ids"]
        static_tensors["position_ids"][:, :seqlen].copy_(position_ids)

        if "mrope_position_deltas" in current_inputs:
            assert "mrope_position_deltas" in static_tensors
            mrope_num = current_inputs["mrope_position_deltas"].shape[0]
            static_tensors["mrope_position_deltas"][:mrope_num].copy_(
                current_inputs["mrope_position_deltas"])

        self.graphs[key].replay()
        output_ref = self.graph_outputs[key]

        return output_ref

    def _get_padded_batch(self, batch: ScheduledRequests,
                          resource_manager: ResourceManager) -> int:
        kv_cache_manager = resource_manager.get_resource_manager(
            self.kv_cache_manager_key)
        can_run_cuda_graph = batch.can_run_cuda_graph
        batch_size = batch.batch_size
        new_batch_size = batch_size

        if self.enabled and self.enable_attention_dp and self.mapping.tp_size > 1:
            graph_batch_size = self.dist.tp_allgather(
                [can_run_cuda_graph, batch_size])
            all_can_graph = all(graph_batch[0]
                                for graph_batch in graph_batch_size)
            if all_can_graph:
                new_batch_size = max(gen_only_batch[1]
                                     for gen_only_batch in graph_batch_size)

        if (not self.enabled or not self.padding_enabled
                or not can_run_cuda_graph
                or new_batch_size > self.max_supported_batch_size):
            return 0

        padded_batch_size = self._round_up_batch_size(new_batch_size)
        if batch_size == padded_batch_size:
            return 0

        padding_size = padded_batch_size - batch_size
        if padding_size + batch.batch_size > self.max_batch_size:
            return 0

        # No padding if it would create too many concurrent requests.
        # This is not strictly required, but we should probably
        # respect the requirement just in case that changes in the future.
        if self.padding_dummy_request is None:
            available_blocks = kv_cache_manager.get_num_free_blocks()
            # No padding if not enough KV cache space
            if available_blocks < 1:
                return 0

            self.padding_dummy_request = kv_cache_manager.add_dummy_requests(
                [CUDA_GRAPH_DUMMY_REQUEST_ID],
                is_gen=True,
                max_num_draft_tokens=self.max_draft_len,
                use_mrope=self.use_mrope,
                max_beam_width=self.max_beam_width)[0]
            self.padding_dummy_request.is_cuda_graph_dummy = True
            spec_res_mgr = resource_manager.get_resource_manager(
                ResourceManagerType.SPEC_RESOURCE_MANAGER)
            if spec_res_mgr:
                spec_res_mgr.add_dummy_requests([CUDA_GRAPH_DUMMY_REQUEST_ID])

        batch.generation_requests.extend([self.padding_dummy_request] *
                                         padding_size)
        return padding_size

    def _round_up_batch_size(self, batch_size: int) -> int:
        """Finds the smallest supported graph batch size >= the given size."""
        if not self.supported_batch_sizes:
            return 0
        idx = bisect.bisect_left(self.supported_batch_sizes, batch_size)
        if idx == len(self.supported_batch_sizes):
            return 0
        return self.supported_batch_sizes[idx]

    @contextlib.contextmanager
    def pad_batch(self, scheduled_requests: ScheduledRequests,
                  resource_manager: ResourceManager):
        """Context manager to pad a batch to a graph-compatible size."""

        padding_size = self._get_padded_batch(scheduled_requests,
                                              resource_manager)
        try:
            yield scheduled_requests
        finally:
            if padding_size > 0:
                scheduled_requests.generation_requests = scheduled_requests.generation_requests[:
                                                                                                -padding_size]

    def clear(self):
        """Releases all captured graphs and the associated memory pool."""
        for graph in self.graphs.values():
            graph.reset()
        self.graphs.clear()
        self.graph_outputs.clear()
        self.graph_metadata.clear()
        self.padding_dummy_request = None
        del self.memory_pool
        self.memory_pool = None
        torch.cuda.empty_cache()
