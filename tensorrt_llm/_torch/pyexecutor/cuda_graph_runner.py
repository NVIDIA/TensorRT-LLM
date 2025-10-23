import bisect
import contextlib
import weakref
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Tuple

import torch

from ...inputs.multimodal import MultimodalParams
from ..expert_statistic import ExpertStatistic
from ..memory_buffer_utils import get_memory_buffers
from ..modules.multi_stream_utils import with_multi_stream
from ..speculative.eagle3 import Eagle3ResourceManager
from ..utils import make_weak_ref, piecewise_cuda_graph
from .resource_manager import (BaseResourceManager, ResourceManager,
                               ResourceManagerType)
from .scheduler import ScheduledRequests

if TYPE_CHECKING:
    from .model_engine import PyTorchModelEngine

# A large prime number used for dummy request IDs to avoid collisions
CUDA_GRAPH_DUMMY_REQUEST_ID = (1 << 64) - 1


class CUDAGraphRunner:
    """
    Manages the lifecycle and execution of CUDA graphs for the model engine.

    This unified class handles high-level orchestration (padding, eligibility)
    and low-level execution (capturing, resource management, replaying) for
    multiple graphs, keyed by (batch size, draft_len, is_first_draft).
    """
    WARMUP_STEPS = 2

    def __init__(self, engine: "PyTorchModelEngine"):
        self.engine_ref = weakref.ref(engine)

        # High-level configuration
        config = engine.pytorch_backend_config
        self.enabled = config.use_cuda_graph
        self.padding_enabled = config.cuda_graph_padding_enabled
        self.supported_batch_sizes = engine._cuda_graph_batch_sizes
        self.max_supported_batch_size = engine._max_cuda_graph_batch_size
        self.max_beam_width = engine.max_beam_width
        self.spec_config = engine.spec_config

        self.graphs: Dict[Tuple[int, int, int], torch.cuda.CUDAGraph] = {}
        self.graph_outputs: Dict[Tuple[int, int, int],
                                 Callable[[], Optional[torch.Tensor]]] = {}
        self.graph_metadata: Dict[Tuple[int, int, int], Dict[str, Any]] = {}
        self.memory_pool = engine._cuda_graph_mem_pool
        self.padding_dummy_request: Optional["Request"] = None

        self.shared_static_tensors: Dict[str, torch.Tensor] = {}
        if self.enabled:
            self._create_shared_static_tensors()
        self.cuda_graph_meta_buffers = get_memory_buffers()

    def _create_shared_static_tensors(self):
        """Allocates static tensors sized for the largest possible batch."""
        engine = self._get_engine()

        token_per_request = self.max_possible_draft_len + 1
        max_total_tokens = (self.max_supported_batch_size *
                            self.max_beam_width * token_per_request)
        max_total_tokens = min(max_total_tokens, engine.max_num_tokens)

        self.shared_static_tensors = {
            "input_ids":
            torch.ones((max_total_tokens, ), device="cuda", dtype=torch.int32),
            "position_ids":
            torch.zeros((1, max_total_tokens), device="cuda",
                        dtype=torch.int32),
        }
        if engine.use_mrope:
            self.shared_static_tensors["position_ids"] = torch.zeros(
                (3, 1, max_total_tokens), device="cuda", dtype=torch.int32)
            self.shared_static_tensors["multimodal_params"] = [
                MultimodalParams(
                    multimodal_data={
                        "mrope_config": {
                            "mrope_position_deltas":
                            torch.zeros(
                                (1, 1), device="cuda", dtype=torch.int32)
                        }
                    }) for _ in range(max_total_tokens)
            ]

    @property
    def enable_spec_decode(self):
        return self._get_engine().enable_spec_decode

    @property
    def max_possible_draft_len(self):
        engine = self._get_engine()
        return (engine.original_max_total_draft_tokens
                if self.enable_spec_decode else 0)

    def get_graph_key(
            self,
            batch_size,
            spec_resource_manager: Optional[BaseResourceManager] = None):
        engine = self._get_engine()
        if engine.is_draft_model and spec_resource_manager is not None and isinstance(
                spec_resource_manager, Eagle3ResourceManager):
            # If 'is_first_draft' is True, even with tree decoding, the length of draft_len will only be 'max_draft_len', not 'max_total_draft_token'.
            # Because we will pad the input to 'max_draft_len' length for the first draft layer.
            draft_len = engine.original_max_draft_len if spec_resource_manager.is_first_draft else 0
            key = (batch_size, draft_len, spec_resource_manager.is_first_draft)
        else:
            draft_len = self.spec_config.max_total_draft_tokens if self.enable_spec_decode else 0
            key = (batch_size, draft_len, False)
        return key

    @property
    def spec_metadata(self):
        return self._get_engine().spec_metadata

    @property
    def draft_tokens_cuda(self):
        return self._get_engine().draft_tokens_cuda

    @property
    def attn_metadata(self):
        return self._get_engine().attn_metadata

    def __del__(self):
        self.clear()

    def _get_engine(self) -> "PyTorchModelEngine":
        """Safely dereferences the weak reference to the engine."""
        engine = self.engine_ref()
        if engine is None:
            raise RuntimeError(
                "The parent PyTorchModelEngine has been garbage collected.")
        return engine

    def maybe_get_cuda_graph(
            self,
            batch: ScheduledRequests,
            spec_resource_manager: Optional[BaseResourceManager] = None):
        """
        Determines if the current batch can be run with a CUDA graph.

        Returns a tuple containing:
        - A boolean indicating if a graph can be used.
        - The attn_metadata for the graph, if applicable.
        - The spec_metadata for the graph, if applicable.
        - The key for the graph.
        """
        engine = self._get_engine()

        # disable when doing statistic
        if hasattr(engine, 'iter_counter') and ExpertStatistic.set_iter(
                engine.iter_counter):
            return False, None, None, None

        can_run_cuda_graph = batch.can_run_cuda_graph
        batch_size = batch.batch_size
        if self.enabled and engine.enable_attention_dp and engine.mapping.tp_size > 1:
            all_can_graph_batch = engine.dist.tp_allgather(
                [can_run_cuda_graph, batch_size])
            is_all_gen_only = all(all_can_graph[0]
                                  for all_can_graph in all_can_graph_batch)
            all_batch_size_equal = all(
                all_gen_only[1] == all_can_graph_batch[0][1]
                for all_gen_only in all_can_graph_batch)

            if not is_all_gen_only or not all_batch_size_equal:
                return False, None, None, None

        if not self.enabled or not can_run_cuda_graph:
            return False, None, None, None
        key = self.get_graph_key(batch_size, spec_resource_manager)

        if key in self.graphs:
            return True, self.graph_metadata[key][
                "attn_metadata"], self.graph_metadata[key]["spec_metadata"], key

        if batch_size not in self.supported_batch_sizes:
            return False, None, None, None

        num_sequences_in_batch = batch_size * self.max_beam_width
        attn_metadata = self.attn_metadata.create_cuda_graph_metadata(
            num_sequences_in_batch, False, key[1], self.cuda_graph_meta_buffers)
        assert attn_metadata.is_cuda_graph

        if self.enable_spec_decode:
            spec_metadata = self.spec_metadata.create_cuda_graph_metadata(
                num_sequences_in_batch)
            spec_metadata.draft_tokens = self.draft_tokens_cuda
        else:
            spec_metadata = None
        return True, attn_metadata, spec_metadata, key

    def needs_capture(self, key: Tuple[int, int, int]):

        return key not in self.graph_outputs

    def get_graph_pool(self):
        """Returns the CUDA memory pool used by this graph runner.

        Returns:
            The CUDA memory pool associated with captured graphs, or None if
            no graphs have been captured yet.
        """
        return self.memory_pool

    def capture(self,
                key: Tuple[int, int, int],
                forward_fn: Callable,
                initial_inputs: Dict[str, Any],
                postprocess_fn: Optional[Callable] = None):
        """Captures the forward pass for a given batch size."""
        engine = self._get_engine()
        batch_size = key[0]
        # [CUDA graph spec decode padding]
        # We pad input IDs/position IDs to the maximum draft length (token per request).
        # We're forced to do this because we cannot reallocate inputs over many graph runs.
        max_draft_len = key[1]
        token_per_request = max_draft_len + 1
        num_tokens_for_capture = (batch_size * self.max_beam_width *
                                  token_per_request)

        sliced_static_tensors = {
            "input_ids":
            self.shared_static_tensors["input_ids"][:num_tokens_for_capture],
            "position_ids":
            self.shared_static_tensors["position_ids"]
            [:, :num_tokens_for_capture],
        }
        if engine.use_mrope:
            sliced_static_tensors["position_ids"] = self.shared_static_tensors[
                "position_ids"][:, :, :num_tokens_for_capture],
            sliced_static_tensors[
                "multimodal_params"] = self.shared_static_tensors[
                    "multimodal_params"][:batch_size * self.max_beam_width]

        capture_inputs = initial_inputs.copy()
        capture_inputs.update(sliced_static_tensors)

        self.graph_metadata[key] = {
            "attn_metadata": initial_inputs["attn_metadata"],
            "spec_metadata": initial_inputs.get("spec_metadata", None),
        }

        def _setup_spec_decoding_and_forward(key: Tuple[int, int, int],
                                             forward_fn: Callable,
                                             capture_inputs: Dict[str, Any]):
            engine = self._get_engine()
            # for the first inference of draft model, we need to set the use_spec_decoding to True when capture the graph for multiple runs.
            is_first_draft = key[2]
            needs_kv_cache_recompute = True if engine.enable_spec_decode and engine.spec_config.spec_dec_mode.needs_kv_cache_recompute(
            ) else False
            if is_first_draft and engine.is_draft_model and needs_kv_cache_recompute:
                capture_inputs['attn_metadata'].use_spec_decoding = True
            return forward_fn(capture_inputs)

        # We have to do warm up runs to initialize PyTorch's
        # internal states according to the docs:
        # https://pytorch.org/docs/stable/notes/cuda.html#cuda-graph-semantics
        # This also lets us initialize states in the attn_metadata.
        graph = torch.cuda.CUDAGraph()
        with with_multi_stream(True), piecewise_cuda_graph(False):
            for _ in range(self.WARMUP_STEPS):
                _setup_spec_decoding_and_forward(key, forward_fn,
                                                 capture_inputs)
                if postprocess_fn is not None:
                    postprocess_fn(capture_inputs)

            with torch.cuda.graph(graph, pool=self.memory_pool):
                output = _setup_spec_decoding_and_forward(
                    key, forward_fn, capture_inputs)
            if postprocess_fn is not None:
                postprocess_fn(capture_inputs)

        self.graphs[key] = graph
        self.graph_outputs[key] = make_weak_ref(output)
        self.memory_pool = graph.pool()

    def replay(self, key: Tuple[int, int, int],
               current_inputs: Dict[str, Any]) -> Optional[torch.Tensor]:
        """Replays a previously captured graph."""
        engine = self._get_engine()
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
        if engine.use_mrope and current_inputs.get(
                'multimodal_params') is not None:
            static_tensors["position_ids"][:, :, :seqlen].copy_(position_ids)
            for i, multimodal_param in enumerate(
                    current_inputs['multimodal_params']):
                # NOTE: Currently, we only need 'mrope_position_deltas' on generation phase for multimodal models.
                static_tensors['multimodal_params'][i].multimodal_data[
                    'mrope_config']['mrope_position_deltas'].copy_(
                        multimodal_param.multimodal_data['mrope_config']
                        ['mrope_position_deltas'],
                        non_blocking=True)
        else:
            static_tensors["position_ids"][:, :seqlen].copy_(position_ids)

        self.graphs[key].replay()
        output_ref = self.graph_outputs[key]

        return output_ref

    def _get_padded_batch(self, batch: ScheduledRequests,
                          resource_manager: ResourceManager) -> int:
        engine = self._get_engine()
        kv_cache_manager = resource_manager.get_resource_manager(
            engine.kv_cache_manager_key)
        can_run_cuda_graph = batch.can_run_cuda_graph
        batch_size = batch.batch_size
        new_batch_size = batch_size

        if self.enabled and engine.enable_attention_dp and engine.mapping.tp_size > 1:
            graph_batch_size = engine.dist.tp_allgather(
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
        if padding_size + batch.batch_size > engine.batch_size:
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
                max_num_draft_tokens=engine.runtime_draft_len,
                use_mrope=engine.use_mrope,
                max_beam_width=engine.max_beam_width)[0]
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
