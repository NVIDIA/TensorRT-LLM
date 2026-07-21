import bisect
import contextlib
from dataclasses import dataclass
from typing import (Any, Callable, Dict, Iterator, List, Optional, Tuple,
                    TypeAlias)

import torch

from tensorrt_llm._utils import prefer_pinned
from tensorrt_llm.llmapi.llm_args import (BaseSparseAttentionConfig,
                                          DecodingBaseConfig,
                                          SeqLenAwareSparseAttentionConfig)
from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping

from ..attention_backend.trtllm import TrtllmAttentionMetadata
from ..distributed import Distributed
from ..expert_statistic import ExpertStatistic
from ..memory_buffer_utils import get_memory_buffers
from ..modules.multi_stream_utils import with_multi_stream
from ..speculative.eagle3 import Eagle3ResourceManager
from ..speculative.interface import SpecMetadata
from ..speculative.spec_sampler_base import SampleStateTensorsSpec
from ..speculative.utils import get_draft_kv_cache_manager
from ..utils import make_weak_ref, piecewise_cuda_graph
from .llm_request import LlmRequest, get_draft_token_length
from .resource_manager import (BaseResourceManager, ResourceManager,
                               ResourceManagerType)
from .sampler import SampleStateTensors
from .scheduler import ScheduledRequests

# A large prime number used for dummy request IDs to avoid collisions
CUDA_GRAPH_DUMMY_REQUEST_ID = (1 << 64) - 1
# Gen dummies get prompt_len = token_num - 1. Before capturing enc-dec decode
# graphs, prepare_cross_batch temporarily runs each dummy generation request
# as a one-token context chunk to write its cross-KV cache, so enc-dec
# dummies need one prompt token plus one generated token.
ENC_DEC_CUDA_GRAPH_DUMMY_TOKEN_NUM = 2
KeyType: TypeAlias = Tuple[int, int, bool, bool, bool]


@dataclass
class CUDAGraphRunnerConfig:
    """Configuration for the CUDAGraphRunner, passed from the ModelEngine."""
    use_cuda_graph: bool
    """
    Master switch controlling the model's execution path.

    This flag determines one of three distinct execution paths for the
    model engine:

    1.  **`False` (Pure Eager Path):**
        * Forces all execution to be in eager mode.
        * The `CUDAGraphRunner` instance is mostly dormant
        * Methods like `maybe_get_cuda_graph` and `pad_batch`
            will return immediately, signaling the model engine to
            run in eager mode.

    2.  **`True` (Eager Fallback Path):**
        * The runner is active and checks for graph eligibility.
        * If a batch is ineligible (e.g., it's a prefill batch,
            stats collection is on, or it's an unsupported batch size),
            the runner signals a fallback to eager mode for that batch.

    3.  **`True` (CUDA Graph Path):**
        * The runner finds an eligible batch and a matching graph.
        * The graph is then captured (if new) or replayed.

    Note: As of this implementation, the model engine *always* calls
    `cuda_graph_runner.pad_batch` and `cuda_graph_runner.maybe_get_cuda_graph`
    even when this is `False`. This could be refactored in the future
    so that the engine bypasses the `CUDAGraphRunner` entirely in Case 1.
    """
    cuda_graph_padding_enabled: bool
    cuda_graph_batch_sizes: list[int]
    max_cuda_graph_batch_size: int
    max_beam_width: int
    max_num_tokens: int
    spec_config: Optional[DecodingBaseConfig]
    cuda_graph_mem_pool: Any
    use_mrope: bool
    original_max_draft_len: int
    original_max_total_draft_tokens: int
    is_draft_model: bool
    enable_attention_dp: bool
    is_encoder_decoder: bool
    batch_size: int
    mapping: Optional[Mapping]
    dist: Optional[Distributed]
    kv_cache_manager_key: Any
    dynamic_draft_len_mapping: Optional[Dict[int, int]] = None
    sparse_attention_config: Optional[BaseSparseAttentionConfig] = None


class CUDAGraphRunner:
    """
    Manages the lifecycle and execution of CUDA graphs for the model engine.

    This unified class handles high-level orchestration (padding, eligibility)
    and low-level execution (capturing, resource management, replaying) for
    multiple graphs, keyed by (batch size, draft_len, is_first_draft).
    """
    WARMUP_STEPS = 2

    def __init__(self, config: CUDAGraphRunnerConfig):
        self.config = config

        # High-level configuration from the config object
        self.enabled = config.use_cuda_graph
        self.padding_enabled = config.cuda_graph_padding_enabled
        self.supported_batch_sizes = config.cuda_graph_batch_sizes
        self.max_supported_batch_size = config.max_cuda_graph_batch_size
        self.max_beam_width = config.max_beam_width
        self.spec_config = config.spec_config
        self.sparse_config = config.sparse_attention_config
        self.is_encoder_decoder = config.is_encoder_decoder

        self.graphs: Dict[KeyType, torch.cuda.CUDAGraph] = {}
        self.graph_outputs: Dict[KeyType,
                                 Callable[[], Optional[torch.Tensor]]] = {}
        self.graph_metadata: Dict[KeyType, Dict[str, Any]] = {}
        self.memory_pool = config.cuda_graph_mem_pool
        self.padding_dummy_requests: Dict[int, LlmRequest] = {}
        self.dynamic_draft_len_mapping = config.dynamic_draft_len_mapping

        self.shared_static_tensors: Dict[str, torch.Tensor] = {}
        if self.enabled:
            self._create_shared_static_tensors()
        self.cuda_graph_meta_buffers = get_memory_buffers()

        # On-the-fly capture is disabled by default to prevent workspace
        # tensor reallocation from invalidating addresses baked into existing
        # CUDA graphs.  Use allow_capture() context manager during warmup.
        self._capture_allowed = False
        # Gates the once-only "first replay" hard-path log (see replay()).
        self._replay_logged = False

    def _create_shared_static_tensors(self):
        """Allocates static tensors sized for the largest possible batch."""
        runtime_draft_token_buffer_width = (
            self.config.original_max_total_draft_tokens
            if self.config.spec_config is not None else 0)
        token_per_request = runtime_draft_token_buffer_width + 1
        max_total_tokens = (self.max_supported_batch_size *
                            self.max_beam_width * token_per_request)
        max_total_tokens = min(max_total_tokens, self.config.max_num_tokens)

        self.shared_static_tensors = {
            "input_ids":
            torch.ones((max_total_tokens, ), device="cuda", dtype=torch.int32),
            "position_ids":
            torch.zeros((1, max_total_tokens), device="cuda",
                        dtype=torch.int32),
        }
        if self.config.use_mrope:
            self.shared_static_tensors["position_ids"] = torch.zeros(
                (3, 1, max_total_tokens), device="cuda", dtype=torch.int32)
            self.shared_static_tensors[
                "mrope_delta_read_seq_slots"] = torch.zeros(
                    (max_total_tokens, ), device="cuda", dtype=torch.long)

    def _get_seq_len_mode(
            self,
            batch: ScheduledRequests,
            new_tensors_device: Optional[SampleStateTensors] = None):
        if (isinstance(self.sparse_config, SeqLenAwareSparseAttentionConfig)
                and self.sparse_config.needs_separate_short_long_cuda_graphs()):
            # Some sparse attention algorithms need to use different forward paths for short and long sequences.
            # For example, the DSA can skip the MQA and Top-K in the indexer for short sequences to reduce the
            # computational overhead. To support this feature, we need to capture separate CUDA graphs for short
            # and long sequences. We need to first collect the sequence length of the requests and then determine
            # the sequence length mode. For long sequences, use the default maximum sequence length. For short
            # sequences, use the sequence length threshold as the maximum sequence length.
            total_seq_lens = []
            new_tokens_device, next_draft_tokens_device = None, None
            if new_tensors_device is not None:
                new_tokens_device = new_tensors_device.new_tokens
                if isinstance(new_tensors_device, SampleStateTensorsSpec):
                    next_draft_tokens_device = new_tensors_device.next_draft_tokens
            overlap_scheduler_enabled = new_tokens_device is not None
            for request in batch.generation_requests:
                is_spec_request = get_draft_token_length(
                    request) > 0 or next_draft_tokens_device is not None
                num_draft_tokens = self.spec_config.max_draft_len if is_spec_request else 0
                # First draft
                if request.py_is_first_draft:
                    total_seq_len = len(request.get_tokens(0))
                # With overlap scheduler disabled or dummy request or not assigned to a batch,
                elif not overlap_scheduler_enabled or request.is_dummy or request.py_batch_idx is None:
                    total_seq_len = request.max_beam_num_tokens + num_draft_tokens
                # Other cases
                else:
                    total_seq_len = request.max_beam_num_tokens + num_draft_tokens + 1
                total_seq_lens.append(total_seq_len)
            # Determine the sequence length mode.
            from ..speculative import get_num_extra_kv_tokens
            num_extra_kv_tokens = get_num_extra_kv_tokens(self.spec_config)
            max_seq_len = max(total_seq_lens)
            if max_seq_len <= self.sparse_config.seq_len_threshold - num_extra_kv_tokens:
                short_seq_len_mode = True
            else:
                short_seq_len_mode = False
        else:
            # For non-sparse attention or sparse attention that does not need separate short and long CUDA graphs,
            # use the default sequence length mode.
            short_seq_len_mode = False
        return short_seq_len_mode

    def get_graph_key(
            self,
            batch: ScheduledRequests,
            new_tensors_device: Optional[SampleStateTensors] = None,
            spec_resource_manager: Optional[BaseResourceManager] = None,
            spec_metadata: Optional[SpecMetadata] = None):
        batch_size = batch.batch_size

        # Get the sequence length mode.
        short_seq_len_mode = self._get_seq_len_mode(batch, new_tensors_device)

        # Spec one-engine sampler has two code paths (argmax fast-path vs
        # advanced sampling kernel). Include this in the key so we capture
        # both variants and dispatch at replay based on actual batch state.
        # Default to True (greedy fast-path) when the metadata doesn't carry
        # this field (non-one-engine paths or non-spec batches).
        is_all_greedy_sample = bool(
            getattr(spec_metadata, "is_all_greedy_sample", True))

        if self.config.is_draft_model and spec_resource_manager is not None and isinstance(
                spec_resource_manager, Eagle3ResourceManager):
            # If 'is_first_draft' is True, even with tree decoding, the length of draft_len will only be 'max_draft_len', not 'max_total_draft_token'.
            # Because we will pad the input to 'max_draft_len' length for the first draft layer.
            draft_len = self.config.original_max_draft_len if spec_resource_manager.is_first_draft else 0
            key = (batch_size, draft_len, spec_resource_manager.is_first_draft,
                   short_seq_len_mode, is_all_greedy_sample)
        else:
            # With dynamic spec decode, the draft length may be zero even when enable_spec_decode is True,
            # so we need to get the draft length from the batch instead of using enable_spec_decode.
            draft_len_list = []
            for request in batch.generation_requests:
                draft_len_list.append(len(request.py_draft_tokens))
            draft_len = max(draft_len_list)
            assert len(
                set(draft_len_list)) == 1, "All draft lengths must be the same"
            key = (batch_size, draft_len, False, short_seq_len_mode,
                   is_all_greedy_sample)
        return key

    @staticmethod
    def _get_mrope_position_delta(request: Any) -> Optional[Any]:
        mrope_position_delta = getattr(request, "py_mrope_position_delta", None)
        if mrope_position_delta is not None:
            return mrope_position_delta

        multimodal_data = getattr(request, "py_multimodal_data", None)
        if not multimodal_data:
            return None

        mrope_config = multimodal_data.get("mrope_config")
        if mrope_config is None:
            return None
        return mrope_config.get("mrope_position_deltas")

    @staticmethod
    def _needs_mrope_delta_cache_update(request: Any) -> bool:
        if request.py_seq_slot is None or request.is_dummy:
            return False

        if getattr(request, "py_mrope_delta_cache_slot",
                   None) == request.py_seq_slot:
            return False

        return CUDAGraphRunner._get_mrope_position_delta(request) is not None

    def __del__(self):
        self.clear()

    def maybe_get_cuda_graph(
        self,
        batch: ScheduledRequests,
        enable_spec_decode: bool,
        attn_metadata: Any,
        spec_metadata: Optional[SpecMetadata] = None,
        draft_tokens_cuda: Optional[torch.Tensor] = None,
        new_tensors_device: Optional[SampleStateTensors] = None,
        spec_resource_manager: Optional[BaseResourceManager] = None,
    ) -> Tuple[Optional[Any], Optional[Any], Optional[Tuple[int, int, bool]]]:
        """
        Determines if the current batch can be run with a CUDA graph.

        Returns a tuple containing:
        - The attn_metadata for the graph, if applicable.
        - The spec_metadata for the graph, if applicable.
        - The key for the graph, if applicable.
        """
        # disable when doing statistic
        if ExpertStatistic.should_record():
            return None, None, None

        can_run_cuda_graph = batch.can_run_cuda_graph
        batch_size = batch.batch_size
        if self.enabled and self.config.enable_attention_dp and self.config.mapping.tp_size > 1:
            all_can_graph_batch = self.config.dist.tp_allgather(
                [can_run_cuda_graph, batch_size])
            is_all_gen_only = all(all_can_graph[0]
                                  for all_can_graph in all_can_graph_batch)
            all_batch_size_equal = all(
                all_gen_only[1] == all_can_graph_batch[0][1]
                for all_gen_only in all_can_graph_batch)

            if not is_all_gen_only or not all_batch_size_equal:
                return None, None, None

        if not self.enabled or not can_run_cuda_graph:
            return None, None, None
        if self.config.use_mrope and any(
                self._needs_mrope_delta_cache_update(request)
                for request in batch.generation_requests):
            # Some MRoPE paths have no per-request delta (for example,
            # Qwen3.5 configs normalized to text-only decoding). Requests that
            # do carry a delta must first populate the model-side cache for their
            # current seq slot before graph replay.
            return None, None, None
        key = self.get_graph_key(batch, new_tensors_device,
                                 spec_resource_manager, spec_metadata)

        if key in self.graphs:
            return self.graph_metadata[key][
                "attn_metadata"], self.graph_metadata[key]["spec_metadata"], key

        # Graph doesn't exist yet.  If on-the-fly capture is not allowed,
        # fall back to eager so the caller doesn't need a separate check.
        if not self._capture_allowed:
            return None, None, None

        if batch_size not in self.supported_batch_sizes:
            return None, None, None

        num_sequences_in_batch = batch_size * self.max_beam_width
        graph_attn_metadata = attn_metadata.create_cuda_graph_metadata(
            num_sequences_in_batch, False, key[1], self.cuda_graph_meta_buffers)
        assert graph_attn_metadata.is_cuda_graph

        if enable_spec_decode:
            graph_spec_metadata = spec_metadata.create_cuda_graph_metadata(
                num_sequences_in_batch)
            graph_spec_metadata.draft_tokens = draft_tokens_cuda
        else:
            graph_spec_metadata = None
        return graph_attn_metadata, graph_spec_metadata, key

    def needs_capture(self, key: KeyType):
        return self._capture_allowed and key not in self.graph_outputs

    @contextlib.contextmanager
    def allow_capture(self):
        """Context manager that enables CUDA graph capture.

        Capture is disabled by default.  On-the-fly captures outside this
        context are prevented because they can resize the shared
        cuda_graph_workspace tensor, invalidating addresses baked into
        previously captured graphs.
        """
        self._capture_allowed = True
        try:
            yield
        finally:
            self._capture_allowed = False

    def get_graph_pool(self):
        """Returns the CUDA memory pool used by this graph runner.

        Returns:
            The CUDA memory pool associated with captured graphs, or None if
            no graphs have been captured yet.
        """
        return self.memory_pool

    def capture(self,
                key: KeyType,
                forward_fn: Callable,
                initial_inputs: Dict[str, Any],
                enable_spec_decode: bool = False,
                postprocess_fn: Optional[Callable] = None):
        """Captures the forward pass for a given batch size."""
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
        if self.config.use_mrope:
            sliced_static_tensors["position_ids"] = self.shared_static_tensors[
                "position_ids"][:, :, :num_tokens_for_capture]
            if "mrope_delta_read_seq_slots" in initial_inputs:
                sliced_static_tensors[
                    "mrope_delta_read_seq_slots"] = self.shared_static_tensors[
                        "mrope_delta_read_seq_slots"][:batch_size *
                                                      self.max_beam_width]

        capture_inputs = initial_inputs.copy()
        capture_inputs.update(sliced_static_tensors)

        self.graph_metadata[key] = {
            "attn_metadata": initial_inputs["attn_metadata"],
            "spec_metadata": initial_inputs.get("spec_metadata", None),
        }

        def _setup_spec_decoding_and_forward(key: KeyType, forward_fn: Callable,
                                             capture_inputs: Dict[str, Any]):
            is_first_draft = key[2]
            needs_kv_cache_recompute = True if enable_spec_decode and self.config.spec_config.spec_dec_mode.needs_kv_cache_recompute(
            ) else False
            if is_first_draft and self.config.is_draft_model and needs_kv_cache_recompute:
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
        # Hard-path evidence: an explicit, greppable record that the production
        # runtime actually CAPTURED a real CUDA graph for this decode key (not a
        # silent fallback to eager). Emitted from the worker rank during warmup,
        # so it survives the TP>1 proxy layout where the driver process cannot
        # introspect the engine's graph store directly.
        logger.info(
            f"[cuda-graph] CAPTURED generation graph key={key} "
            f"total_captured={len(self.graphs)}")

    def replay(self, key: KeyType,
               current_inputs: Dict[str, Any]) -> Optional[torch.Tensor]:
        """Replays a previously captured graph."""
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
        if self.config.use_mrope:
            static_tensors["position_ids"][:, :, :seqlen].copy_(position_ids)
            mrope_delta_read_seq_slots = current_inputs.get(
                'mrope_delta_read_seq_slots')
            if mrope_delta_read_seq_slots is not None:
                static_tensors[
                    'mrope_delta_read_seq_slots'][:mrope_delta_read_seq_slots.
                                                  shape[0]].copy_(
                                                      mrope_delta_read_seq_slots,
                                                      non_blocking=True)
        else:
            static_tensors["position_ids"][:, :seqlen].copy_(position_ids)

        self.graphs[key].replay()
        # Hard-path evidence (logged once): prove the captured graph was actually
        # REPLAYED on the decode path rather than the runtime silently falling
        # back to eager. Gated by a flag so a long generation logs a single line,
        # not one per decode step.
        if not self._replay_logged:
            logger.info(
                f"[cuda-graph] REPLAYED generation graph key={key} "
                f"(first replay; total_captured={len(self.graphs)})")
            self._replay_logged = True
        output_ref = self.graph_outputs[key]

        return output_ref

    def _get_padded_batch(self, batch: ScheduledRequests,
                          resource_manager: ResourceManager,
                          runtime_draft_len: int) -> int:
        kv_cache_manager = resource_manager.get_resource_manager(
            self.config.kv_cache_manager_key)
        can_run_cuda_graph = batch.can_run_cuda_graph
        batch_size = batch.batch_size
        new_batch_size = batch_size

        if self.enabled and self.config.enable_attention_dp and self.config.mapping.tp_size > 1:
            graph_batch_size = self.config.dist.tp_allgather(
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

        # When dynamic draft length is enabled (one-model path), we treat the determined runtime draft length
        # as the source of truth and pad the batch size up to the nearest existing graph
        # for that draft length.
        if (self.spec_config and self.spec_config.draft_len_schedule
                and self.spec_config.spec_dec_mode.support_dynamic_draft_len()):
            padded_batch_size = self._round_up_batch_size_with_draft_len(
                new_batch_size, runtime_draft_len)
        else:
            padded_batch_size = self._round_up_batch_size(new_batch_size)

        if batch_size == padded_batch_size:
            return 0

        padding_size = padded_batch_size - batch_size
        if padding_size <= 0:
            return 0
        if padding_size + batch.batch_size > self.config.batch_size:
            return 0

        runtime_tokens_per_gen_step = (
            self.spec_config.get_runtime_tokens_per_gen_step(runtime_draft_len)
            if self.spec_config is not None else 1 + runtime_draft_len)
        runtime_draft_token_buffer_width = runtime_tokens_per_gen_step - 1

        # No padding if it would create too many concurrent requests.
        # This is not strictly required, but we should probably
        # respect the requirement just in case that changes in the future.
        # Use per-draft-len dummy requests for dynamic draft length support.
        if runtime_draft_len not in self.padding_dummy_requests:
            dummy_encoder_output_len = None
            if self.is_encoder_decoder:
                cross_kv_cache_manager = resource_manager.get_resource_manager(
                    ResourceManagerType.CROSS_KV_CACHE_MANAGER)
                if cross_kv_cache_manager is None:
                    return 0
                dummy_encoder_output_len = self._get_padding_dummy_encoder_output_len(
                    cross_kv_cache_manager)

            # Get draft KV cache manager only for one-model speculative decoding.
            # In two-model mode, each model has its own KV cache manager, so
            # draft_kv_cache_manager should be None.
            draft_kv_cache_manager = get_draft_kv_cache_manager(
                self.spec_config, resource_manager)

            # Use unique dummy request ID per draft length
            dummy_request_id = CUDA_GRAPH_DUMMY_REQUEST_ID - runtime_draft_len
            dummy_request = kv_cache_manager.add_dummy_requests(
                [dummy_request_id],
                token_nums=[ENC_DEC_CUDA_GRAPH_DUMMY_TOKEN_NUM]
                if self.is_encoder_decoder else None,
                is_gen=True,
                max_num_draft_tokens=runtime_draft_token_buffer_width,
                use_mrope=self.config.use_mrope,
                max_beam_width=self.config.max_beam_width,
                encoder_output_lens=[dummy_encoder_output_len]
                if dummy_encoder_output_len is not None else None,
                draft_kv_cache_manager=draft_kv_cache_manager)

            if dummy_request is None:
                return 0
            else:
                dummy_request = dummy_request[0]
            dummy_request.is_cuda_graph_dummy = True
            if self.is_encoder_decoder:
                if not self._add_cross_dummy_request(
                        dummy_request, resource_manager,
                        dummy_encoder_output_len, draft_kv_cache_manager):
                    return 0

            spec_res_mgr = resource_manager.get_resource_manager(
                ResourceManagerType.SPEC_RESOURCE_MANAGER)
            if spec_res_mgr:
                spec_res_mgr.add_dummy_requests([dummy_request_id])
            self.padding_dummy_requests[runtime_draft_len] = dummy_request

        padding_dummy_request = self.padding_dummy_requests[runtime_draft_len]
        batch.generation_requests.extend([padding_dummy_request] * padding_size)
        return padding_size

    def _add_cross_dummy_request(
            self, dummy_request: LlmRequest, resource_manager: ResourceManager,
            encoder_output_len: int,
            draft_kv_cache_manager: Optional[BaseResourceManager]) -> bool:
        cross_kv_cache_manager = resource_manager.get_resource_manager(
            ResourceManagerType.CROSS_KV_CACHE_MANAGER)
        if cross_kv_cache_manager is None:
            return False

        dummy_request.py_encoder_output = None
        dummy_request.py_skip_cross_kv_projection = True

        encoder_output_lens = [encoder_output_len]
        cross_dummy_requests = cross_kv_cache_manager.add_dummy_requests(
            request_ids=[dummy_request.py_request_id],
            token_nums=encoder_output_lens,
            is_gen=True,
            max_beam_width=self.config.max_beam_width,
            encoder_output_lens=encoder_output_lens)
        if cross_dummy_requests is not None:
            return True

        kv_cache_manager = resource_manager.get_resource_manager(
            self.config.kv_cache_manager_key)
        kv_cache_manager.free_resources(dummy_request)
        if draft_kv_cache_manager is not None:
            draft_kv_cache_manager.free_resources(dummy_request)
        return False

    @staticmethod
    def _get_padding_dummy_encoder_output_len(
            cross_kv_cache_manager: Any) -> int:
        encoder_output_len = 1
        max_seq_len = getattr(cross_kv_cache_manager, "max_seq_len", None)
        if max_seq_len is not None:
            encoder_output_len = min(encoder_output_len, int(max_seq_len))
        return encoder_output_len

    def _round_up_batch_size(self, batch_size: int) -> int:
        """Finds the smallest supported graph batch size >= the given size."""
        if not self.supported_batch_sizes:
            return 0
        idx = bisect.bisect_left(self.supported_batch_sizes, batch_size)
        if idx == len(self.supported_batch_sizes):
            return 0
        return self.supported_batch_sizes[idx]

    def _round_up_batch_size_with_draft_len(self, batch_size: int,
                                            draft_len: int) -> int:
        """Finds the smallest graph batch size >= batch_size that also matches the given draft_len."""
        if not self.dynamic_draft_len_mapping:
            # Fallback to regular round up if no mapping
            return self._round_up_batch_size(batch_size)

        start_idx = bisect.bisect_left(self.supported_batch_sizes, batch_size)
        # Negate the list to make it non-decreasing for bisect
        # (draft_len decreases as batch_size increases in the schedule)
        draft_lens = [
            self.dynamic_draft_len_mapping.get(self.supported_batch_sizes[i], 0)
            for i in range(start_idx, len(self.supported_batch_sizes))
        ]
        idx = bisect.bisect_left(draft_lens, -draft_len, key=lambda x: -x)
        if idx < len(draft_lens) and draft_lens[idx] == draft_len:
            return self.supported_batch_sizes[start_idx + idx]
        # No suitable graph found
        return 0

    @contextlib.contextmanager
    def pad_batch(self,
                  scheduled_requests: ScheduledRequests,
                  resource_manager: ResourceManager,
                  runtime_draft_len: int = 0):
        """Context manager to pad a batch to a graph-compatible size."""
        padding_size = self._get_padded_batch(scheduled_requests,
                                              resource_manager,
                                              runtime_draft_len)
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
        self.padding_dummy_requests = {}
        del self.memory_pool
        self.memory_pool = None
        torch.cuda.empty_cache()


EncoderKeyType: TypeAlias = Tuple[int, int, int]


@dataclass
class EncoderCUDAGraphRunnerConfig:
    """Configuration for EncoderCUDAGraphRunner."""
    use_cuda_graph: bool
    cuda_graph_padding_enabled: bool
    cuda_graph_batch_sizes: List[int]
    cuda_graph_num_tokens: List[int]
    cuda_graph_seq_lens: List[int]
    max_cuda_graph_batch_size: int
    max_cuda_graph_num_tokens: int
    max_num_tokens: int
    max_seq_len: int
    cuda_graph_mem_pool: Any


class EncoderCUDAGraphRunner:
    """CUDA graph runner for no-cache encoder forward passes.

    Designed for encoder inputs with `input_ids` (flat [total_tokens]) and
    `seq_lens` ([batch_size]). Encoder CUDA graphs are keyed on the 3-tuple
    (padded_batch_size, padded_num_tokens, padded_max_seq_len).

    Restricted to `TrtllmAttentionMetadata` — FlashInfer's per-batch planner state is not compatible with CUDA graph capture/replay.
    """

    WARMUP_STEPS = 2

    def __init__(self, config: EncoderCUDAGraphRunnerConfig):
        self.config = config

        self.enabled = config.use_cuda_graph
        self.padding_enabled = config.cuda_graph_padding_enabled
        self.supported_batch_sizes = sorted(config.cuda_graph_batch_sizes)
        self.max_supported_batch_size = config.max_cuda_graph_batch_size
        self.supported_num_tokens = sorted(config.cuda_graph_num_tokens)
        self.max_supported_num_tokens = config.max_cuda_graph_num_tokens
        self.supported_seq_lens = sorted(config.cuda_graph_seq_lens)

        self.graphs: Dict[EncoderKeyType, torch.cuda.CUDAGraph] = {}
        self.graph_outputs: Dict[EncoderKeyType, Callable[[],
                                                          Optional[Any]]] = {}
        self.graph_metadata: Dict[EncoderKeyType, Dict[str, Any]] = {}
        self.memory_pool = config.cuda_graph_mem_pool

        self.shared_static_tensors: Dict[str, torch.Tensor] = {}
        self.shared_static_tensors_cpu: Dict[str, torch.Tensor] = {}
        if self.enabled:
            self._create_shared_static_tensors()
        self.cuda_graph_meta_buffers = get_memory_buffers()

        self._capture_allowed = False

        # CUDA graph H2D memcpy nodes require pinned host sources. In CC mode
        # prefer_pinned() is false: pageable host buffers are preferred, so the
        # H2D copies must be issued before graph replay instead of captured.
        self._capture_h2d_copy = prefer_pinned()

    def _create_shared_static_tensors(self):
        """Allocates static tensors sized for the largest supported num_tokens."""
        max_total_tokens = min(self.max_supported_num_tokens,
                               self.config.max_num_tokens)
        max_batch_size = self.max_supported_batch_size

        self.shared_static_tensors = {
            "input_ids":
            torch.ones((max_total_tokens, ), device="cuda", dtype=torch.int32),
            "position_ids":
            torch.zeros((1, max_total_tokens), device="cuda",
                        dtype=torch.int32),
        }
        self.shared_static_tensors_cpu = {
            "input_ids":
            torch.ones((max_total_tokens, ),
                       device="cpu",
                       dtype=torch.int32,
                       pin_memory=prefer_pinned()),
            "position_ids":
            torch.zeros((1, max_total_tokens),
                        device="cpu",
                        dtype=torch.int32,
                        pin_memory=prefer_pinned()),
            # Pinned static buffer for seq_lens. Each captured graph's attn_metadata._seq_lens
            # is reseated (in maybe_get_cuda_graph) to a stable slice of this buffer, and the
            # corresponding H2D copy into _seq_lens_cuda is captured inside the graph itself.
            "seq_lens":
            torch.ones((max_batch_size, ),
                       device="cpu",
                       dtype=torch.int32,
                       pin_memory=prefer_pinned()),
        }

        # Cached arange used by replay() to build packed position_ids in-place via slice copies.
        self._arange_max = torch.arange(max_total_tokens, dtype=torch.int32)

    @staticmethod
    def _round_up(value: int, supported: List[int]) -> int:
        """Smallest element of `supported` >= value, or 0 if none exists."""
        if not supported:
            return 0
        idx = bisect.bisect_left(supported, value)
        if idx == len(supported):
            return 0
        return supported[idx]

    def _get_valid_graph_key(self, batch_size: int, num_tokens: int,
                             max_seq_len: int) -> EncoderKeyType:
        num_tokens_idx = bisect.bisect_left(self.supported_num_tokens,
                                            num_tokens)
        seq_len_idx = bisect.bisect_left(self.supported_seq_lens, max_seq_len)

        while (num_tokens_idx < len(self.supported_num_tokens)
               and seq_len_idx < len(self.supported_seq_lens)):
            padded_num_tokens = self.supported_num_tokens[num_tokens_idx]
            padded_max_seq_len = self.supported_seq_lens[seq_len_idx]

            if padded_num_tokens > batch_size * padded_max_seq_len:
                seq_len_idx += 1
            elif padded_max_seq_len > padded_num_tokens:
                num_tokens_idx += 1
            else:
                return batch_size, padded_num_tokens, padded_max_seq_len

        return batch_size, 0, 0

    def get_graph_key(
            self, inputs: Dict[str, Any]) -> Tuple[EncoderKeyType, bool, bool]:
        """Compute the (bs, padded_num_tokens, padded_max_seq_len) bucket.

        `inputs['seq_lens']` must already be padded to padded_batch_size via
        `pad_batch(...)` before calling this. Dummy entries are 1-token each
        and do not raise max_seq_len since real requests dominate.
        """
        seq_lens = inputs['seq_lens']

        num_tokens = sum(
            seq_lens
        )  # Can't use len(inputs['input_ids']) because it's not padded
        batch_size = len(seq_lens)
        max_seq_len = max(seq_lens) if batch_size > 0 else 0

        key = self._get_valid_graph_key(batch_size, num_tokens, max_seq_len)
        _, padded_num_tokens, padded_max_seq_len = key

        is_padding_performed = (padded_num_tokens != num_tokens
                                or padded_max_seq_len != max_seq_len)
        is_padding_successful = (padded_num_tokens != 0
                                 and padded_max_seq_len != 0)

        return key, is_padding_performed, is_padding_successful

    @contextlib.contextmanager
    def allow_capture(self):
        """Context manager that enables CUDA graph capture.

        Capture is disabled by default.  On-the-fly captures outside this
        context are prevented — unseen keys fall back to eager instead of
        incurring a multi-millisecond capture latency spike at runtime.
        """
        self._capture_allowed = True
        try:
            yield
        finally:
            self._capture_allowed = False

    @contextlib.contextmanager
    def pad_batch(self, inputs: Dict[str, Any],
                  batch_size: int) -> Iterator[Dict[str, Any]]:
        if not self.enabled or not self.padding_enabled:
            yield inputs
            return

        padded_batch_size = self._round_up(batch_size,
                                           self.supported_batch_sizes)
        if padded_batch_size == 0 or padded_batch_size == batch_size:
            yield inputs
            return

        padding_size = padded_batch_size - batch_size
        # Should not pad inputs if it would exceed the max supported number of tokens
        # maybe_get_cuda_graph will check this and fall back to eager if batch size is not in the supported list
        if len(inputs['input_ids']
               ) + padding_size > self.max_supported_num_tokens:
            yield inputs
            return

        # Only seq_lens is padded — that's all the attention metadata needs.
        # Token-shaped inputs (input_ids, position_ids, ...) are padded implicitly
        # by zero-filling the static buffer in `replay`.
        padded_inputs = dict(inputs)
        padded_inputs['seq_lens'] = list(
            inputs['seq_lens']) + [1] * padding_size

        yield padded_inputs

    def maybe_get_cuda_graph(
        self,
        inputs: Dict[str, Any],
        attn_metadata: Any,
    ) -> Tuple[Optional[Any], Optional[EncoderKeyType]]:
        """
        Decide whether the batch can use a CUDA graph.

        Returns (graph_attn_metadata, key) when a graph can be used, else
        (None, None). On graph hit, the returned `attn_metadata` is the
        graph-resident metadata whose `_seq_lens` is permanently aliased to
        a slice of the runner's pinned `seq_lens` buffer; per-replay seq_lens
        updates are pure CPU memcpys into that buffer (the H2D copy that
        feeds `_seq_lens_cuda` is captured inside the graph).
        """
        if not self.enabled:
            return None, None

        # Only TRTLLM attention backend supports encoder CUDA graphs. Other
        # backends (FlashInfer) have per-batch planner state that breaks
        # graph replay.
        if not isinstance(attn_metadata, TrtllmAttentionMetadata):
            logger.warning_once(
                "Encoder CUDA graph only supports TrtllmAttentionMetadata; "
                "falling back to eager.",
                key="encoder_cuda_graph_backend_warning")
            return None, None

        if ExpertStatistic.should_record():
            return None, None

        seq_lens = inputs['seq_lens']
        padded_batch_size = len(seq_lens)
        if padded_batch_size not in self.supported_batch_sizes:
            return None, None

        key, is_padding_performed, is_padding_successful = self.get_graph_key(
            inputs)
        _, _, padded_max_seq_len = key
        if (not self.padding_enabled and is_padding_performed) \
                or not is_padding_successful:
            return None, None

        if key in self.graphs:
            return self.graph_metadata[key]["attn_metadata"], key

        # New key not yet captured. Only create metadata if capture is
        # allowed (warmup time); otherwise fall back to eager.
        if not self._capture_allowed:
            return None, None

        if "multi_item_part_lens" in inputs:
            # See model_engine.py for more details
            logger.warning_once(
                "Encoder CUDA graph does not support multi-item scoring; "
                "falling back to eager.",
                key="encoder_cuda_graph_multi_item_scoring_warning")
            return None, None

        if attn_metadata.has_cross_sub_metadata:
            logger.warning_once(
                "Encoder CUDA graph does not support cross-attention metadata; "
                "falling back to eager.",
                key="encoder_cuda_graph_cross_attention_warning")
            return None, None

        # First sighting of this key: create graph-resident metadata and bind
        # it to stable pinned seq_lens storage for future replays.
        graph_attn_metadata = attn_metadata.create_cuda_graph_metadata(
            padded_batch_size,
            False,
            0,
            self.cuda_graph_meta_buffers,
            encode_only=True,
        )
        assert graph_attn_metadata.is_cuda_graph

        # Lock FMHA kernel launch params to the padded max_seq_len so the
        # cubin + grid dims stay constant across replays for this key.
        graph_attn_metadata.max_context_q_len_override = padded_max_seq_len

        # Bind graph metadata to stable host seq_lens storage. The storage may
        # be pinned or pageable; only captured H2D copies require pinned memory.
        graph_attn_metadata.bind_encoder_cuda_graph_seq_lens(
            self.shared_static_tensors_cpu["seq_lens"], padded_batch_size)
        graph_attn_metadata.max_seq_len = self.config.max_seq_len
        graph_attn_metadata.request_ids = list(range(padded_batch_size))

        return graph_attn_metadata, key

    def _contains_nested_tensor(self, x: Any) -> bool:
        if isinstance(x, torch.Tensor):
            return x.is_nested
        if isinstance(x, dict):
            return any(self._contains_nested_tensor(v) for v in x.values())
        if isinstance(x, (list, tuple)):
            return any(self._contains_nested_tensor(v) for v in x)
        return False

    def needs_capture(self, key: EncoderKeyType) -> bool:
        return self._capture_allowed and key not in self.graphs

    def capture(
        self,
        key: EncoderKeyType,
        forward_fn: Callable[[Dict[str, Any]], Any],
        inputs: Dict[str, Any],
    ) -> None:
        """Capture a CUDA graph for the given key."""
        _, padded_num_tokens, _ = key

        sliced_static_tensors = {
            "input_ids":
            self.shared_static_tensors["input_ids"][:padded_num_tokens],
            "position_ids":
            self.shared_static_tensors["position_ids"][:, :padded_num_tokens],
        }
        sliced_static_tensors_cpu = {
            "input_ids":
            self.shared_static_tensors_cpu["input_ids"][:padded_num_tokens],
            "position_ids":
            self.shared_static_tensors_cpu["position_ids"]
            [:, :padded_num_tokens],
        }

        capture_inputs = dict(inputs)
        capture_inputs.update(sliced_static_tensors)

        attn_md = capture_inputs["attn_metadata"]

        self.graph_metadata[key] = {
            "attn_metadata": attn_md,
        }

        graph = torch.cuda.CUDAGraph()
        # Warmup runs required by CUDA graph semantics. See
        # https://pytorch.org/docs/stable/notes/cuda.html#cuda-graph-semantics
        with with_multi_stream(True), piecewise_cuda_graph(False):
            for _ in range(self.WARMUP_STEPS):
                forward_fn(capture_inputs)

            with torch.cuda.graph(graph, pool=self.memory_pool):
                if self._capture_h2d_copy:
                    # H2D copies for captured inside the graph: at replay
                    # time it re-issues from the pinned static buffer without
                    # an eager driver call.
                    capture_inputs["input_ids"].copy_(
                        sliced_static_tensors_cpu["input_ids"],
                        non_blocking=True)
                    capture_inputs["position_ids"].copy_(
                        sliced_static_tensors_cpu["position_ids"],
                        non_blocking=True)
                    attn_md._seq_lens_cuda.copy_(attn_md._seq_lens,
                                                 non_blocking=True)
                output = forward_fn(capture_inputs)

        if self._contains_nested_tensor(output):
            raise TypeError(
                "Encoder CUDA graph does not support nested tensor outputs. "
                "Disable encoder CUDA graphs for models with ragged outputs.")
        self.graphs[key] = graph
        self.graph_outputs[key] = make_weak_ref(output)
        self.memory_pool = graph.pool()

    def replay(
        self,
        key: EncoderKeyType,
        inputs: Dict[str, Any],
    ) -> Any:
        """Replay a captured graph with current inputs."""
        stored_meta = self.graph_metadata[key]
        assert inputs["attn_metadata"] is stored_meta["attn_metadata"]

        _, padded_num_tokens, _ = key

        # According to prefer_pinned(), CC forces most transfers to be synchronous.
        # So we don't put non_blocking=True here.
        static_tensors = self.shared_static_tensors_cpu if self._capture_h2d_copy else self.shared_static_tensors

        # input_ids: convert (if list) and write into pinned active region in
        # one allocation + one memcpy. Padding region is zero-filled below.
        input_ids = inputs["input_ids"]
        if isinstance(input_ids, list):
            actual_tokens = len(input_ids)
            static_tensors["input_ids"][:actual_tokens].copy_(
                torch.tensor(input_ids, dtype=torch.int32))
        elif isinstance(input_ids, torch.Tensor):
            actual_tokens = int(input_ids.shape[0])
            static_tensors["input_ids"][:actual_tokens].copy_(input_ids)
        else:
            raise TypeError(f"Unsupported input_ids type: {type(input_ids)}")
        static_tensors["input_ids"][actual_tokens:padded_num_tokens].fill_(0)

        # position_ids: pinned buffer is shape [1, max_total_tokens]; use the
        # 1-D row view. Auto-generate via the cached arange (zero allocations,
        # N small memcpys) or copy user-provided values.
        pinned_pos = static_tensors["position_ids"][0]
        position_ids = inputs.get("position_ids")
        if position_ids is None:
            # Pad entries (seq_len=1) get arange[:1] = [0], the correct
            # position for a 1-token dummy request.
            offset = 0
            for s in inputs["seq_lens"]:
                pinned_pos[offset:offset + s].copy_(self._arange_max[:s])
                offset += s
        else:
            if isinstance(position_ids, list):
                pinned_pos[:actual_tokens].copy_(
                    torch.tensor(position_ids, dtype=torch.int32))
            elif isinstance(position_ids, torch.Tensor):
                pinned_pos[:actual_tokens].copy_(position_ids.flatten())
            else:
                raise TypeError(
                    f"Unsupported position_ids type: {type(position_ids)}")
            offset = actual_tokens

        pinned_pos[offset:padded_num_tokens].fill_(0)

        if not self._capture_h2d_copy:
            stored_meta["attn_metadata"]._seq_lens_cuda.copy_(
                stored_meta["attn_metadata"]._seq_lens, non_blocking=True)

        self.graphs[key].replay()

        return self.graph_outputs[key]

    def get_graph_pool(self):
        return self.memory_pool

    def clear(self):
        for graph in self.graphs.values():
            graph.reset()
        self.graphs.clear()
        self.graph_outputs.clear()
        self.graph_metadata.clear()
        del self.memory_pool
        self.memory_pool = None
        torch.cuda.empty_cache()
