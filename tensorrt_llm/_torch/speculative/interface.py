import copy
import os
from collections import Counter
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import IntEnum, auto
from typing import TYPE_CHECKING, List, Optional, Type

import torch
from torch import nn

from tensorrt_llm.logger import logger

from ..._utils import get_sm_version, prefer_pinned
from ..attention_backend.trtllm import (AttentionBackend, TrtllmAttention,
                                        TrtllmAttentionMetadata)
from ..flashinfer_utils import IS_FLASHINFER_AVAILABLE
from ..pyexecutor.resource_manager import (BaseResourceManager,
                                           ResourceManagerType)

if TYPE_CHECKING:
    from ..pyexecutor.guided_decoder import CapturableGuidedDecoder
    from ..pyexecutor.llm_request import LlmRequest

if IS_FLASHINFER_AVAILABLE:
    import flashinfer

# Environment variable name for forcing the number of accepted tokens in speculative decoding
FORCE_NUM_ACCEPTED_TOKENS_ENV_VAR = "TLLM_SPEC_DECODE_FORCE_NUM_ACCEPTED_TOKENS"


def should_use_separate_draft_kv_cache(spec_config) -> bool:
    """
    Check if separate draft KV cache should be used for one-engine speculative decoding.
    """
    if spec_config is None:
        return False
    if not spec_config.spec_dec_mode.use_one_engine():
        return False
    return spec_config._allow_separate_draft_kv_cache


def prepare_attn_metadata_for_draft_replay(attn_metadata,
                                           draft_kv_cache_manager):
    """
    Prepare attention metadata for CUDA graph replay when using separate draft KV cache.
    Swaps to draft manager and (for DSA) re-prepares indexer slot mappings for the current
    batch. Call restore_attn_metadata_after_draft_replay after replay in a finally block.
    Returns saved state or None if no-op.
    """
    if draft_kv_cache_manager is None:
        return None
    if not isinstance(attn_metadata, TrtllmAttentionMetadata):
        return None
    draft_block_offsets = getattr(attn_metadata, 'draft_kv_cache_block_offsets',
                                  None)
    if draft_block_offsets is None:
        return None

    saved = {
        'target_kv_cache_manager':
        attn_metadata.kv_cache_manager,
        'target_kv_cache_block_offsets':
        attn_metadata.kv_cache_block_offsets,
        'target_host_kv_cache_block_offsets':
        attn_metadata.host_kv_cache_block_offsets,
    }
    attn_metadata.kv_cache_manager = draft_kv_cache_manager
    attn_metadata.kv_cache_block_offsets = attn_metadata.draft_kv_cache_block_offsets
    attn_metadata.host_kv_cache_block_offsets = (
        draft_kv_cache_manager.host_kv_cache_block_offsets)

    from ..attention_backend.sparse.dsa import (DSAtrtllmAttentionMetadata,
                                                Indexer)
    if (isinstance(attn_metadata, DSAtrtllmAttentionMetadata)
            and hasattr(draft_kv_cache_manager, 'index_head_dim')):
        m = attn_metadata
        saved['saved_dsa_state'] = {
            'host_indexer_k_cache_block_offsets':
            m.host_indexer_k_cache_block_offsets.clone(),
            'indexer_k_cache_block_offsets':
            m.indexer_k_cache_block_offsets.clone(),
            'host_slot_mapping_fp8':
            m.host_slot_mapping_fp8.clone(),
            'host_slot_mapping_scale':
            m.host_slot_mapping_scale.clone(),
            'slot_mapping_fp8':
            m.slot_mapping_fp8.clone(),
            'slot_mapping_scale':
            m.slot_mapping_scale.clone(),
        }
        # Derive pool indices from the draft manager's encoded block
        # offsets (via _get_pool_block_indices) instead of using raw block
        # IDs.  With host cache offload, block IDs can exceed
        # blocks_in_primary_pool after offload swaps (the block keeps its
        # original high ID even though its memory now lives in the primary
        # GPU pool).  Using raw block IDs as pool indices causes OOB access
        # in the indexer k-cache buffers.  _get_pool_block_indices correctly
        # decodes memPoolBlockIndex from the C++ encoded offsets.
        # Note: kv_cache_manager was already swapped to draft above (line 67).
        pool_indices = m._get_pool_block_indices()
        num_blocks = pool_indices.shape[1]
        m.host_indexer_k_cache_block_offsets[:m.num_seqs, :num_blocks].copy_(
            pool_indices)
        m.indexer_k_cache_block_offsets[:m.num_seqs].copy_(
            m.host_indexer_k_cache_block_offsets[:m.num_seqs],
            non_blocking=True)
        # Safety clamp: sanitize stale padding entries beyond num_seqs
        # that may contain negative or out-of-range values, matching the
        # regular DSA prepare() flow.
        m.indexer_k_cache_block_offsets.clamp_(min=0)
        Indexer.recompute_slot_mappings(m)
    return saved


def restore_attn_metadata_after_draft_replay(attn_metadata, saved_state):
    """Restore attention metadata after draft replay. No-op if saved_state is None."""
    if saved_state is None:
        return
    attn_metadata.kv_cache_manager = saved_state['target_kv_cache_manager']
    attn_metadata.kv_cache_block_offsets = (
        saved_state['target_kv_cache_block_offsets'])
    attn_metadata.host_kv_cache_block_offsets = (
        saved_state['target_host_kv_cache_block_offsets'])
    saved_dsa = saved_state.get('saved_dsa_state')
    if saved_dsa is not None:
        m = attn_metadata
        m.host_indexer_k_cache_block_offsets.copy_(
            saved_dsa['host_indexer_k_cache_block_offsets'], non_blocking=True)
        m.indexer_k_cache_block_offsets.copy_(
            saved_dsa['indexer_k_cache_block_offsets'], non_blocking=True)
        m.host_slot_mapping_fp8.copy_(saved_dsa['host_slot_mapping_fp8'])
        m.host_slot_mapping_scale.copy_(saved_dsa['host_slot_mapping_scale'])
        m.slot_mapping_fp8.copy_(saved_dsa['slot_mapping_fp8'])
        m.slot_mapping_scale.copy_(saved_dsa['slot_mapping_scale'])


def get_force_num_accepted_tokens() -> int:
    """
    Read and parse the TLLM_SPEC_DECODE_FORCE_NUM_ACCEPTED_TOKENS environment variable.

    Returns:
        int: The forced number of accepted tokens, or 0 if not set or invalid.
    """
    env_value = os.environ.get(FORCE_NUM_ACCEPTED_TOKENS_ENV_VAR, "0")
    try:
        return int(env_value)
    except ValueError:
        logger.warning(
            f"{FORCE_NUM_ACCEPTED_TOKENS_ENV_VAR} must be a valid integer, "
            f"got '{env_value}'. Using default value 0.")
        return 0


class SpeculativeDecodingMode(IntEnum):
    MTP = auto()
    MTP_EAGLE = auto()
    MTP_EAGLE_ONE_MODEL = auto()
    EAGLE3 = auto()
    EAGLE3_ONE_MODEL = auto()
    NGRAM = auto()
    SA = auto()
    DRAFT_TARGET = auto()
    DRAFT_TARGET_ONE_MODEL = auto()
    USER_PROVIDED = auto()
    SAVE_HIDDEN_STATES = auto()
    PARD = auto()
    NONE = auto()
    AUTO = auto()

    def is_mtp_one_model(self):
        return self == SpeculativeDecodingMode.MTP or self == SpeculativeDecodingMode.MTP_EAGLE_ONE_MODEL

    def is_mtp_eagle_one_model(self):
        return self == SpeculativeDecodingMode.MTP_EAGLE_ONE_MODEL

    def is_mtp_vanilla(self):
        return self == SpeculativeDecodingMode.MTP

    def is_mtp_eagle(self):
        return self == SpeculativeDecodingMode.MTP_EAGLE

    def is_eagle3(self):
        return self == SpeculativeDecodingMode.EAGLE3

    def use_one_engine(self):
        return self.is_eagle3_one_model() or self.is_mtp_one_model(
        ) or self.is_external_drafter() or self.is_sa()

    def is_eagle3_one_model(self):
        return self == SpeculativeDecodingMode.EAGLE3_ONE_MODEL

    def is_pard(self):
        return self == SpeculativeDecodingMode.PARD

    def is_ngram(self):
        return self == SpeculativeDecodingMode.NGRAM

    def is_sa(self):
        return self == SpeculativeDecodingMode.SA

    def is_user_provided(self):
        return self == SpeculativeDecodingMode.USER_PROVIDED

    def is_none(self):
        return self == SpeculativeDecodingMode.NONE

    def is_draft_target(self):
        return self == SpeculativeDecodingMode.DRAFT_TARGET

    def is_draft_target_one_model(self):
        return self == SpeculativeDecodingMode.DRAFT_TARGET_ONE_MODEL

    def is_save_hidden_states(self):
        return self == SpeculativeDecodingMode.SAVE_HIDDEN_STATES

    def is_external_drafter(self):
        return self.is_pard() or self.is_draft_target_one_model()

    def without_logits(self):
        return self.is_mtp_one_model() or self.is_eagle3_one_model(
        ) or self.is_external_drafter() or self.is_sa()

    def needs_kv_cache_rewind(self):
        return self.is_mtp_one_model() or self.is_eagle3_one_model(
        ) or self.is_ngram() or self.is_sa() or self.is_external_drafter()

    def support_overlap_scheduler(self):
        return self.is_mtp_one_model() or self.is_eagle3_one_model(
        ) or self.is_sa() or self.has_draft_model() or self.is_external_drafter(
        )

    def support_guided_decoder(self):
        return self.is_none() or self.has_spec_drafter()

    def support_capturable_guided_decoder(self):
        return self.is_mtp_one_model() or self.is_eagle3_one_model(
        ) or self.is_external_drafter() or self.is_sa()

    def support_dynamic_draft_len(self):
        # TODO: expand to all one-model algorithms
        return self.is_eagle3_one_model()

    def has_draft_model(self):
        return self.is_eagle3() or self.is_draft_target() or self.is_mtp_eagle()

    def needs_kv_cache_recompute(self):
        """
        Whether the draft model needs to recompute the kv cache.
        If true, the 1st draft model forward will recompute the kv cache for
        the accepted draft tokens.
        """
        return self.is_eagle3() or self.is_mtp_eagle()

    def need_load_draft_weights(self):
        """
        Whether the draft model and target model are in the same model engine,
        and the draft model needs to load weights from the separate checkpoint.
        """
        return self.is_eagle3_one_model() or self.is_external_drafter()

    def has_spec_decoder(self):
        return self.is_mtp_one_model() or self.is_mtp_eagle() or self.is_eagle3(
        ) or self.is_eagle3_one_model() or self.is_external_drafter(
        ) or self.is_sa()

    def has_spec_drafter(self):
        return self.is_eagle3() or self.is_draft_target() or self.is_ngram(
        ) or self.is_user_provided() or self.is_mtp_eagle()

    def extend_ctx(self, attention_backend: Type[AttentionBackend]):
        """
        If true, treat generation requests with draft tokens as
        chunked context requests at the kernel level.
        """

        if self.use_one_engine():
            # 1-model has separate logic for handling draft tokens
            return False

        xqa_supported = get_sm_version() < 120
        return not issubclass(attention_backend,
                              TrtllmAttention) or not xqa_supported

    def attention_need_spec_dec_mode(
            self,
            spec_resource_manager: Optional[BaseResourceManager],
            is_draft_model: bool,
            attention_backend: Type[AttentionBackend],
            use_chain_drafter: bool,  # CDL
    ):
        """
        If true, the attention backend kernel needs to run in spec-dec mode (multi-token query mode).
        Args:
            spec_resource_manager: the resource manager for the spec-dec mode.
            is_draft_model: whether the model is a draft model.
            attention_backend: the attention backend.
            use_chain_drafter: whether to use capturable drafting loops (CDL). For the target model, it is always False.
        """
        is_trtllm_attention = issubclass(attention_backend, TrtllmAttention)

        # Always use the multi-token query mode for 1-model if the kernels are available.
        use_case_1 = self.use_one_engine()
        # For 2-model, we need to enable it when we process multiple tokens at once. This occurs with
        # the target model (verification) or on the first draft for CDL based speculation.
        use_case_2 = not self.use_one_engine() and (
            not is_draft_model or
            (spec_resource_manager is not None
             and spec_resource_manager.is_first_draft
             and use_chain_drafter)) and is_trtllm_attention

        return use_case_1 or use_case_2

    @staticmethod
    def from_string(name: Optional[str]) -> "SpeculativeDecodingMode":
        if name is None:
            return SpeculativeDecodingMode.NONE
        return SpeculativeDecodingMode[name.upper()]


@dataclass
class SpecMetadata:
    """
    Metadata for speculative decoding.
    """
    # The max number of requests in a single batch.
    max_num_requests: int
    # The number of draft layers. (Also the number of draft tokens for the linear tree.)
    max_draft_len: int
    # The max number of draft tokens for the static tree and dynamic tree   .
    max_total_draft_tokens: int
    # Capacity for persistent sequence-slot indexed state. This can be larger
    # than max_num_requests when the executor has multiple sequence slots.
    max_num_sequence_slots: Optional[int] = None
    # The number of gen-phase sequences in the batch.
    num_generations: int = 0
    # Whether CUDA graph is enabled.
    is_cuda_graph: bool = field(default=False, repr=False)
    # The mode of speculative decoding.
    spec_dec_mode: SpeculativeDecodingMode = SpeculativeDecodingMode.NONE
    # Draft tokens.
    draft_tokens: Optional[torch.Tensor] = None
    # The length of the draft tokens.
    draft_lens: Optional[torch.Tensor] = None
    # The request ID of each sequence in the batch.
    # The shape is (batch_size).
    request_ids: Optional[List[int]] = None
    # Sequence length for each request.
    seq_lens: Optional[List[int]] = None
    # The gather ids for logits.
    gather_ids: Optional[torch.Tensor] = None
    # The number of accepted draft tokens for each request.
    num_accepted_draft_tokens: Optional[torch.Tensor] = None
    # The number of tokens for speculative model/layer
    num_tokens: int = 0
    # The number of tokens for speculative model/layer of different rank
    all_rank_num_tokens: Optional[List[int]] = None

    # The number of sequences for speculative model/layer of different rank
    all_rank_num_seqs: Optional[List[int]] = None
    # The number of extra kv tokens
    # Some speculative decoding methods need to use different kv lengths for the
    # draft/target layers. But KVCacheManager can only support kv caches with the
    # same kv lengths for different layers. Add extra kv token in kv cache manager
    # to handle this issue.
    num_extra_kv_tokens: Optional[int] = 0  # Number of layers in target model
    # The number of layers
    num_layers: int = 0

    # if spec-dec tree wouldn't be changed at all, the mask won't be computed every step.
    # NOTE: For the linear tree, though it can be treated as a special case of static tree.
    # NOTE: But we do not set `is_spec_dec_tree` to True for this cases.
    # NOTE: i.e., for the linear tree, is_spec_dec_tree == False and is_spec_dec_dynamic_tree == False.
    # whether the spec-dec mode is a tree (can be static tree or dynamic tree).
    is_spec_dec_tree: bool = False
    # whether the spec-dec mode is a dynamic tree.
    is_spec_dec_dynamic_tree: bool = False

    # The draft length used for the current iteration.
    # With dynamic draft length enabled, this varies per batch based on
    # draft_len_schedule.  Otherwise it equals max_draft_len (the static max).
    # Always set by model_engine.forward() before any downstream code reads it.
    runtime_draft_len: int = 0

    # For non-greedy sampling on 1-model.
    allow_advanced_sampling: bool = False
    # Sampling parameters for non-greedy sampling (per-request)
    temperatures: Optional[torch.Tensor] = None
    top_ks: Optional[torch.Tensor] = None
    top_ps: Optional[torch.Tensor] = None
    recent_penalty_token_ids: Optional[torch.Tensor] = field(default=None,
                                                             repr=False)
    recent_penalty_values: Optional[torch.Tensor] = field(default=None,
                                                          repr=False)
    recent_seq_penalty_token_ids: Optional[torch.Tensor] = field(default=None,
                                                                 repr=False)
    recent_seq_penalty_values: Optional[torch.Tensor] = field(default=None,
                                                              repr=False)
    draft_prefix_penalty_token_ids: Optional[torch.Tensor] = field(
        default=None, repr=False)
    draft_prefix_penalty_values: Optional[torch.Tensor] = field(default=None,
                                                                repr=False)
    draft_prefix_penalty_rows: Optional[torch.Tensor] = field(default=None,
                                                              repr=False)
    device_penalty_history_tokens: Optional[torch.Tensor] = field(
        default=None, repr=False)
    device_penalty_history_lens: Optional[torch.Tensor] = field(default=None,
                                                                repr=False)
    device_penalty_row_slots: Optional[torch.Tensor] = field(default=None,
                                                             repr=False)
    device_penalty_seq_slots: Optional[torch.Tensor] = field(default=None,
                                                             repr=False)
    device_frequency_penalties: Optional[torch.Tensor] = field(default=None,
                                                               repr=False)
    device_seq_frequency_penalties: Optional[torch.Tensor] = field(
        default=None, repr=False)
    device_penalty_history_capacity: int = 0
    use_device_penalty_history: bool = False
    device_penalty_token_counts: Optional[torch.Tensor] = field(default=None,
                                                                repr=False)
    device_penalty_sparse_token_ids: Optional[torch.Tensor] = field(
        default=None, repr=False)
    device_penalty_sparse_token_counts: Optional[torch.Tensor] = field(
        default=None, repr=False)
    device_penalty_sparse_count_lens: Optional[torch.Tensor] = field(
        default=None, repr=False)
    device_penalty_count_row_slots: Optional[torch.Tensor] = field(
        default=None, repr=False)
    device_penalty_count_seq_slots: Optional[torch.Tensor] = field(
        default=None, repr=False)
    device_count_frequency_penalties: Optional[torch.Tensor] = field(
        default=None, repr=False)
    device_count_seq_frequency_penalties: Optional[torch.Tensor] = field(
        default=None, repr=False)
    device_penalty_count_reset_slots: Optional[torch.Tensor] = field(
        default=None, repr=False)
    device_penalty_count_reset_count: int = 0
    device_penalty_count_prompt_tokens: Optional[torch.Tensor] = field(
        default=None, repr=False)
    device_penalty_count_prompt_token_counts: Optional[torch.Tensor] = field(
        default=None, repr=False)
    device_penalty_count_prompt_lens: Optional[torch.Tensor] = field(
        default=None, repr=False)
    device_penalty_count_prompt_seq_slots: Optional[torch.Tensor] = field(
        default=None, repr=False)
    device_penalty_count_prompt_count: int = 0
    device_penalty_count_prompt_capacity: int = 0
    device_penalty_sparse_count_capacity: int = 0
    device_penalty_count_vocab_size: int = 0
    device_penalty_count_mode: str = "dense"
    use_device_penalty_counts: bool = False
    device_penalty_count_slot_request_ids: dict[int, int] = field(
        default_factory=dict, repr=False)
    cuda_graph_source_metadata: Optional[object] = field(default=None,
                                                         repr=False)
    sampling_request_ids: Optional[list[int]] = field(default=None, repr=False)
    sampling_seq_slots: Optional[list[int]] = field(default=None, repr=False)

    def __post_init__(self):
        pass

    def prepare(self):
        """
        Hook to be called before the forward step of the model.
        """

    def create_cuda_graph_metadata(self, max_batch_size: int):
        """
        Creates metadata for CUDA graph execution.
        """
        if self.is_cuda_graph:
            return self

        cuda_graph_metadata = copy.copy(self)
        cuda_graph_metadata.is_cuda_graph = True
        cuda_graph_metadata.max_num_requests = max_batch_size
        cuda_graph_metadata.cuda_graph_source_metadata = self
        cuda_graph_metadata.device_penalty_count_slot_request_ids = (
            self.device_penalty_count_slot_request_ids)
        cuda_graph_metadata._sync_device_penalty_count_state_from_owner()
        cuda_graph_metadata.__post_init__()
        return cuda_graph_metadata

    def _device_penalty_count_state_owner(self):
        return self.cuda_graph_source_metadata or self

    def _sync_device_penalty_count_state_from_owner(self) -> None:
        owner = self._device_penalty_count_state_owner()
        if owner is self:
            return
        for name in (
                "device_penalty_token_counts",
                "device_penalty_sparse_token_ids",
                "device_penalty_sparse_token_counts",
                "device_penalty_sparse_count_lens",
                "device_penalty_sparse_count_capacity",
                "device_penalty_count_vocab_size",
        ):
            setattr(self, name, getattr(owner, name))

    def is_layer_capture(self, layer_id: int):
        """
        Whether the layer should be captured (eg for Eagle3).
        By default, does nothing.
        """
        return False

    def maybe_capture_hidden_states(self, layer_id: int,
                                    hidden_states: torch.Tensor,
                                    residual: torch.Tensor) -> None:
        """
        Some spec decode algorithms require hidden states from the target
        model. Use this method to record them. By default, does nothing.
        """

    @staticmethod
    def _sampling_config_value(config, name: str, default):
        value = getattr(config, name, None)
        if value is None:
            return default
        if isinstance(value, torch.Tensor):
            if value.numel() == 0:
                return default
            return value.flatten()[0].item()
        if isinstance(value, (list, tuple)):
            if len(value) == 0:
                return default
            value = value[0]
        return default if value is None else value

    @staticmethod
    def _effective_prompt_ignore_length(request: "LlmRequest",
                                        prompt_ignore_length: int) -> int:
        prompt_len = getattr(request, "py_orig_prompt_len", None)
        if prompt_len is None:
            prompt_len = getattr(request, "orig_prompt_len", None)
        if prompt_len is None:
            prompt_len = getattr(request, "py_prompt_len", None)
        if prompt_len is None:
            prompt_len = getattr(request, "prompt_len", 0)
        return min(max(prompt_ignore_length, 0), max(int(prompt_len), 0))

    @staticmethod
    def _prompt_len(request: "LlmRequest") -> int:
        for attr in ("py_orig_prompt_len", "orig_prompt_len", "py_prompt_len",
                     "prompt_len"):
            value = getattr(request, attr, None)
            if value is not None:
                return max(int(value), 0)
        return 0

    def _valid_seq_slot(self, slot: int) -> bool:
        return 0 <= slot < self._max_num_sequence_slots()

    def _max_num_sequence_slots(self) -> int:
        max_num_sequence_slots = self.max_num_sequence_slots
        if max_num_sequence_slots is None or max_num_sequence_slots <= 0:
            return self.max_num_requests
        return max(int(max_num_sequence_slots), self.max_num_requests)

    @staticmethod
    def _env_bool(value: Optional[str]) -> Optional[bool]:
        if value is None:
            return None
        normalized = value.strip().lower()
        if normalized in ("", "auto"):
            return None
        if normalized in ("1", "true", "yes", "on"):
            return True
        if normalized in ("0", "false", "no", "off"):
            return False
        return None

    @staticmethod
    def _is_disagg_generation_role() -> bool:
        role = os.environ.get("TRTLLM_DISAGG_ROLE", "").strip().lower()
        if role in ("generation", "gen", "decode"):
            return True
        return os.environ.get("TRTLLM_DISAGG_BENCHMARK_GEN_ONLY") == "1"

    def _force_graph_count_path_enabled(self) -> bool:
        override = self._env_bool(
            os.environ.get("TRTLLM_SPEC_FORCE_GRAPH_COUNT_PATH"))
        if override is not None:
            return self.is_cuda_graph and override
        return self.is_cuda_graph and self._is_disagg_generation_role()

    def _ensure_recent_penalty_buffers(self, width: int) -> None:
        max_rows = (self.max_draft_len + 1) * self.max_num_requests
        max_seqs = self.max_num_requests
        needs_alloc = (
            self.recent_penalty_token_ids is None
            or self.recent_penalty_values is None
            or self.recent_seq_penalty_token_ids is None
            or self.recent_seq_penalty_values is None
            or self.recent_penalty_token_ids.shape != (max_rows, width)
            or self.recent_seq_penalty_token_ids.shape != (max_seqs, width))
        if not needs_alloc:
            return

        self.recent_penalty_token_ids = torch.zeros((max_rows, width),
                                                    dtype=torch.long,
                                                    device="cuda")
        self.recent_penalty_values = torch.zeros((max_rows, width),
                                                 dtype=torch.float32,
                                                 device="cuda")
        self.recent_seq_penalty_token_ids = torch.zeros((max_seqs, width),
                                                        dtype=torch.long,
                                                        device="cuda")
        self.recent_seq_penalty_values = torch.zeros((max_seqs, width),
                                                     dtype=torch.float32,
                                                     device="cuda")

    def _ensure_draft_prefix_penalty_buffers(self, width: int) -> None:
        max_rows = (self.max_draft_len + 1) * self.max_num_requests
        needs_alloc = (
            self.draft_prefix_penalty_token_ids is None
            or self.draft_prefix_penalty_values is None
            or self.draft_prefix_penalty_rows is None
            or self.draft_prefix_penalty_token_ids.shape != (max_rows, width))
        if not needs_alloc:
            return

        self.draft_prefix_penalty_token_ids = torch.zeros((max_rows, width),
                                                          dtype=torch.long,
                                                          device="cuda")
        self.draft_prefix_penalty_values = torch.zeros((max_rows, width),
                                                       dtype=torch.float32,
                                                       device="cuda")
        self.draft_prefix_penalty_rows = torch.arange(max_rows,
                                                      dtype=torch.long,
                                                      device="cuda")

    def _ensure_device_penalty_history_buffers(self) -> None:
        max_rows = (self.max_draft_len + 1) * self.max_num_requests
        slot_capacity = self._max_num_sequence_slots()
        capacity = int(
            os.environ.get("TRTLLM_SPEC_PENALTY_HISTORY_TOKENS", "16384"))
        capacity = max(capacity, 0)
        if capacity == 0:
            self.use_device_penalty_history = False
            return

        needs_alloc = (
            self.device_penalty_history_tokens is None
            or self.device_penalty_history_lens is None
            or self.device_penalty_row_slots is None
            or self.device_penalty_seq_slots is None
            or self.device_frequency_penalties is None
            or self.device_seq_frequency_penalties is None
            or self.device_penalty_history_tokens.shape !=
            (slot_capacity, capacity)
            or self.device_penalty_row_slots.shape != (max_rows, ))
        if not needs_alloc:
            return

        self.device_penalty_history_capacity = capacity
        self.device_penalty_history_tokens = torch.zeros(
            (slot_capacity, capacity), dtype=torch.int32, device="cuda")
        self.device_penalty_history_lens = torch.zeros(
            (slot_capacity, ), dtype=torch.int32, device="cuda")
        self.device_penalty_row_slots = torch.zeros((max_rows, ),
                                                    dtype=torch.int32,
                                                    device="cuda")
        self.device_penalty_seq_slots = torch.zeros((self.max_num_requests, ),
                                                    dtype=torch.int32,
                                                    device="cuda")
        self.device_frequency_penalties = torch.zeros((max_rows, ),
                                                      dtype=torch.float32,
                                                      device="cuda")
        self.device_seq_frequency_penalties = torch.zeros(
            (self.max_num_requests, ), dtype=torch.float32, device="cuda")

    def _ensure_device_penalty_count_metadata_buffers(self) -> None:
        max_rows = (self.max_draft_len + 1) * self.max_num_requests
        needs_alloc = (
            self.device_penalty_count_row_slots is None
            or self.device_penalty_count_seq_slots is None
            or self.device_count_frequency_penalties is None
            or self.device_count_seq_frequency_penalties is None
            or self.device_penalty_count_reset_slots is None
            or self.device_penalty_count_row_slots.shape != (max_rows, ))
        if not needs_alloc:
            return

        self.device_penalty_count_row_slots = torch.zeros((max_rows, ),
                                                          dtype=torch.int32,
                                                          device="cuda")
        self.device_penalty_count_seq_slots = torch.zeros(
            (self.max_num_requests, ), dtype=torch.int32, device="cuda")
        self.device_count_frequency_penalties = torch.zeros((max_rows, ),
                                                            dtype=torch.float32,
                                                            device="cuda")
        self.device_count_seq_frequency_penalties = torch.zeros(
            (self.max_num_requests, ), dtype=torch.float32, device="cuda")
        self.device_penalty_count_reset_slots = torch.zeros(
            (self.max_num_requests, ), dtype=torch.int64, device="cuda")

    def ensure_device_penalty_count_buffers(self, vocab_size: int) -> None:
        if vocab_size <= 0:
            self.use_device_penalty_counts = False
            return
        owner = self._device_penalty_count_state_owner()
        if owner is not self:
            owner.device_penalty_count_mode = self.device_penalty_count_mode
            owner.use_device_penalty_counts = self.use_device_penalty_counts
            owner.ensure_device_penalty_count_buffers(vocab_size)
            self._sync_device_penalty_count_state_from_owner()
            return

        slot_capacity = self._max_num_sequence_slots()
        if self.device_penalty_count_mode == "dense":
            if (self.device_penalty_token_counts is not None
                    and self.device_penalty_count_vocab_size == vocab_size
                    and self.device_penalty_token_counts.shape ==
                    (slot_capacity, vocab_size)):
                return

            self.device_penalty_count_vocab_size = vocab_size
            self.device_penalty_token_counts = torch.zeros(
                (slot_capacity, vocab_size),
                dtype=torch.int32,
                device="cuda")
            return

        capacity_env = os.environ.get("TRTLLM_SPEC_SPARSE_COUNT_CAPACITY",
                                      "").strip()
        capacity = int(capacity_env) if capacity_env else 0
        if capacity <= 0:
            capacity = vocab_size
        else:
            capacity = min(capacity, vocab_size)
        if (self.device_penalty_sparse_token_ids is not None
                and self.device_penalty_sparse_token_counts is not None
                and self.device_penalty_sparse_count_lens is not None
                and self.device_penalty_count_vocab_size == vocab_size
                and self.device_penalty_sparse_count_capacity == capacity
                and self.device_penalty_sparse_token_ids.shape ==
                (slot_capacity, capacity)):
            return

        self.device_penalty_count_vocab_size = vocab_size
        self.device_penalty_sparse_count_capacity = capacity
        self.device_penalty_sparse_token_ids = torch.zeros(
            (slot_capacity, capacity), dtype=torch.int32, device="cuda")
        self.device_penalty_sparse_token_counts = torch.zeros(
            (slot_capacity, capacity), dtype=torch.int32, device="cuda")
        self.device_penalty_sparse_count_lens = torch.zeros(
            (slot_capacity, ), dtype=torch.int32, device="cuda")

    def reset_device_penalty_count_slots(self) -> None:
        if (not self.use_device_penalty_counts
                or self.device_penalty_count_reset_slots is None
                or self.device_penalty_count_reset_count == 0):
            return
        reset_slots = self.device_penalty_count_reset_slots[:
                                                            self.device_penalty_count_reset_count]
        reset_slots = reset_slots[(reset_slots >= 0)
                                  &
                                  (reset_slots < self._max_num_sequence_slots())]
        if reset_slots.numel() == 0:
            self.device_penalty_count_reset_count = 0
            return
        if self.device_penalty_count_mode == "dense":
            if self.device_penalty_token_counts is None:
                return
            self.device_penalty_token_counts.index_fill_(0, reset_slots, 0)
        else:
            if self.device_penalty_sparse_count_lens is None:
                return
            self.device_penalty_sparse_count_lens.index_fill_(0, reset_slots, 0)
        self.device_penalty_count_reset_count = 0

    def init_device_penalty_count_prompt_tokens(self) -> None:
        if (not self.use_device_penalty_counts
                or self.device_penalty_count_prompt_tokens is None
                or self.device_penalty_count_prompt_lens is None
                or self.device_penalty_count_prompt_seq_slots is None
                or self.device_penalty_count_prompt_count == 0):
            return

        count = self.device_penalty_count_prompt_count
        if self.device_penalty_count_mode == "dense":
            if self.device_penalty_token_counts is None:
                return
            from .one_model_sampler import append_accepted_tokens_to_counts
            append_accepted_tokens_to_counts(
                self.device_penalty_token_counts,
                self.device_penalty_count_prompt_seq_slots[:count],
                self.device_penalty_count_prompt_tokens[:count].contiguous(),
                self.device_penalty_count_prompt_lens[:count].contiguous())
        else:
            if (self.device_penalty_sparse_token_ids is None
                    or self.device_penalty_sparse_token_counts is None
                    or self.device_penalty_sparse_count_lens is None
                    or self.device_penalty_count_prompt_token_counts is None):
                return
            width = self.device_penalty_count_prompt_capacity
            from .one_model_sampler import init_sparse_token_counts
            init_sparse_token_counts(
                self.device_penalty_sparse_token_ids,
                self.device_penalty_sparse_token_counts,
                self.device_penalty_sparse_count_lens,
                self.device_penalty_count_prompt_tokens[:count, :
                                                        width].contiguous(),
                self.device_penalty_count_prompt_token_counts[:count, :
                                                              width].contiguous(),
                self.device_penalty_count_prompt_lens[:count].contiguous(),
                self.device_penalty_count_prompt_seq_slots[:count].contiguous(),
                self.device_penalty_count_vocab_size)
        self.device_penalty_count_prompt_count = 0

    def prepare_device_penalty_counts(self, vocab_size: int) -> None:
        if not self.use_device_penalty_counts:
            return
        self.ensure_device_penalty_count_buffers(vocab_size)
        self.reset_device_penalty_count_slots()
        self.init_device_penalty_count_prompt_tokens()

    def _ensure_device_penalty_count_prompt_buffers(
            self, max_prompt_tokens: int) -> None:
        if max_prompt_tokens <= 0:
            return
        if (self.device_penalty_count_prompt_tokens is not None
                and self.device_penalty_count_prompt_token_counts is not None
                and self.device_penalty_count_prompt_lens is not None
                and self.device_penalty_count_prompt_seq_slots is not None
                and self.device_penalty_count_prompt_capacity >= max_prompt_tokens):
            return

        self.device_penalty_count_prompt_capacity = max_prompt_tokens
        self.device_penalty_count_prompt_tokens = torch.zeros(
            (self.max_num_requests, max_prompt_tokens),
            dtype=torch.int32,
            device="cuda")
        self.device_penalty_count_prompt_token_counts = torch.zeros(
            (self.max_num_requests, max_prompt_tokens),
            dtype=torch.int32,
            device="cuda")
        self.device_penalty_count_prompt_lens = torch.zeros(
            (self.max_num_requests, ), dtype=torch.int32, device="cuda")
        self.device_penalty_count_prompt_seq_slots = torch.zeros(
            (self.max_num_requests, ), dtype=torch.int32, device="cuda")

    def _populate_device_count_frequency_penalties(
            self, requests: list["LlmRequest"]) -> bool:
        force_graph_count_path = self._force_graph_count_path_enabled()
        if (os.environ.get("TRTLLM_SPEC_USE_DEVICE_COUNTS", "0") != "1"
                and not force_graph_count_path):
            self.use_device_penalty_counts = False
            return False
        if not self.allow_advanced_sampling or not self.spec_dec_mode.use_one_engine(
        ):
            self.use_device_penalty_counts = False
            return False

        row_slots: list[int] = []
        seq_slots: list[int] = []
        frequency_penalties: list[float] = []
        seq_frequency_penalties: list[float] = []
        reset_slots: list[int] = []
        prompt_init_slots: list[int] = []
        prompt_init_tokens: list[list[int]] = []
        prompt_init_token_counts: list[list[int]] = []
        any_penalty = False
        can_use = True
        count_mode = os.environ.get("TRTLLM_SPEC_COUNT_MODE",
                                    "sparse").strip().lower()
        if count_mode not in ("dense", "sparse"):
            count_mode = "sparse"
        sparse_capacity_limit = int(
            os.environ.get("TRTLLM_SPEC_SPARSE_COUNT_CAPACITY", "") or "0")

        next_slot_request_ids: dict[int, int] = {}

        for request in requests:
            raw_slot = getattr(request, "py_seq_slot", None)
            slot = int(raw_slot) if raw_slot is not None else -1
            valid_slot = (self._valid_seq_slot(slot)
                          and not getattr(request, "is_dummy", False))
            effective_slot = slot if valid_slot else -1
            seq_slots.append(effective_slot)
            request_id = int(getattr(request, "py_request_id",
                                     getattr(request, "request_id", -1)))
            if valid_slot:
                next_slot_request_ids[slot] = request_id

            sampling_config = request.sampling_config
            frequency_penalty = float(
                self._sampling_config_value(sampling_config,
                                            "frequency_penalty", 0.0))
            presence_penalty = float(
                self._sampling_config_value(sampling_config,
                                            "presence_penalty", 0.0))
            prompt_ignore_length = int(
                self._sampling_config_value(sampling_config,
                                            "prompt_ignore_length", 0))

            if presence_penalty != 0.0:
                can_use = False
                break

            if not valid_slot:
                frequency_penalty = 0.0

            any_penalty = any_penalty or frequency_penalty != 0.0
            is_new_slot_request = valid_slot and (
                self.device_penalty_count_slot_request_ids.get(slot) != request_id)
            if is_new_slot_request:
                reset_slots.append(slot)
                if frequency_penalty != 0.0:
                    ignore_length = self._effective_prompt_ignore_length(
                        request, prompt_ignore_length)
                    tokens = request.get_tokens(0)
                    count_history = [
                        int(token) for token in tokens[ignore_length:]
                        if int(token) >= 0
                    ]
                    if count_history:
                        if count_mode == "sparse":
                            counts = Counter(count_history)
                            unique_tokens = list(counts.keys())
                            if (sparse_capacity_limit > 0
                                    and len(unique_tokens) >
                                    sparse_capacity_limit):
                                # Do not disable device-side generated-token
                                # counts for the whole batch just because one
                                # request history cannot fit in the sparse table.
                                # The request still starts with an empty count
                                # table, and accepted generated tokens are
                                # appended below by the sampler.
                                continue
                            prompt_init_slots.append(slot)
                            prompt_init_tokens.append(unique_tokens)
                            prompt_init_token_counts.append([
                                int(counts[token]) for token in unique_tokens
                            ])
                        else:
                            prompt_init_slots.append(slot)
                            prompt_init_tokens.append(count_history)
                            prompt_init_token_counts.append([])

            from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequestState
            num_rows = 1 + self.runtime_draft_len if request.state == LlmRequestState.GENERATION_IN_PROGRESS else 1
            row_slots.extend(effective_slot for _ in range(num_rows))
            frequency_penalties.extend(frequency_penalty for _ in range(num_rows))
            seq_frequency_penalties.append(frequency_penalty)

        if not can_use or not row_slots or (not any_penalty
                                            and not force_graph_count_path):
            self.use_device_penalty_counts = False
            return False

        self.device_penalty_count_slot_request_ids.update(
            next_slot_request_ids)

        self.device_penalty_count_mode = count_mode
        self._ensure_device_penalty_count_metadata_buffers()
        assert self.device_penalty_count_row_slots is not None
        assert self.device_penalty_count_seq_slots is not None
        assert self.device_count_frequency_penalties is not None
        assert self.device_count_seq_frequency_penalties is not None
        assert self.device_penalty_count_reset_slots is not None
        max_prompt_tokens = max((len(tokens) for tokens in prompt_init_tokens),
                                default=0)
        self._ensure_device_penalty_count_prompt_buffers(max_prompt_tokens)

        self.device_penalty_count_seq_slots[:len(seq_slots)].copy_(
            torch.tensor(seq_slots, dtype=torch.int32,
                         pin_memory=prefer_pinned()),
            non_blocking=True)
        self.device_penalty_count_row_slots[:len(row_slots)].copy_(
            torch.tensor(row_slots, dtype=torch.int32,
                         pin_memory=prefer_pinned()),
            non_blocking=True)
        self.device_count_frequency_penalties[:len(frequency_penalties)].copy_(
            torch.tensor(frequency_penalties,
                         dtype=torch.float32,
                         pin_memory=prefer_pinned()),
            non_blocking=True)
        self.device_count_seq_frequency_penalties[:len(seq_frequency_penalties
                                                       )].copy_(
                                                           torch.tensor(
                                                               seq_frequency_penalties,
                                                               dtype=torch.float32,
                                                               pin_memory=prefer_pinned(
                                                               )),
                                                           non_blocking=True)
        self.device_penalty_count_reset_count = len(reset_slots)
        if reset_slots:
            self.device_penalty_count_reset_slots[:len(reset_slots)].copy_(
                torch.tensor(reset_slots,
                             dtype=torch.int64,
                             pin_memory=prefer_pinned()),
                non_blocking=True)
        self.device_penalty_count_prompt_count = len(prompt_init_tokens)
        if prompt_init_tokens:
            assert self.device_penalty_count_prompt_tokens is not None
            assert self.device_penalty_count_prompt_token_counts is not None
            assert self.device_penalty_count_prompt_lens is not None
            assert self.device_penalty_count_prompt_seq_slots is not None
            prompt_tensor = torch.zeros(
                (len(prompt_init_tokens), max_prompt_tokens),
                dtype=torch.int32,
                pin_memory=prefer_pinned())
            prompt_counts_tensor = torch.zeros(
                (len(prompt_init_tokens), max_prompt_tokens),
                dtype=torch.int32,
                pin_memory=prefer_pinned())
            prompt_lens = []
            for row, tokens in enumerate(prompt_init_tokens):
                prompt_lens.append(len(tokens))
                prompt_tensor[row, :len(tokens)] = torch.tensor(
                    tokens, dtype=torch.int32)
                if count_mode == "sparse":
                    prompt_counts_tensor[row, :len(tokens)] = torch.tensor(
                        prompt_init_token_counts[row], dtype=torch.int32)
            self.device_penalty_count_prompt_tokens[:len(prompt_init_tokens),
                                                    :max_prompt_tokens].copy_(
                                                        prompt_tensor,
                                                        non_blocking=True)
            if count_mode == "sparse":
                self.device_penalty_count_prompt_token_counts[:len(
                    prompt_init_tokens), :max_prompt_tokens].copy_(
                        prompt_counts_tensor, non_blocking=True)
            self.device_penalty_count_prompt_lens[:len(prompt_lens)].copy_(
                torch.tensor(prompt_lens,
                             dtype=torch.int32,
                             pin_memory=prefer_pinned()),
                non_blocking=True)
            self.device_penalty_count_prompt_seq_slots[:len(
                prompt_init_slots)].copy_(torch.tensor(
                    prompt_init_slots,
                    dtype=torch.int32,
                    pin_memory=prefer_pinned()),
                                          non_blocking=True)
        self.use_device_penalty_counts = True
        return True

    def _populate_device_history_frequency_penalties(
            self, requests: list["LlmRequest"]) -> bool:
        if os.environ.get("TRTLLM_SPEC_USE_DEVICE_HISTORY", "0") != "1":
            self.use_device_penalty_history = False
            return False
        if not self.allow_advanced_sampling or not self.spec_dec_mode.use_one_engine(
        ):
            self.use_device_penalty_history = False
            return False

        row_slots: list[int] = []
        seq_slots: list[int] = []
        frequency_penalties: list[float] = []
        seq_frequency_penalties: list[float] = []
        reset_slots: list[int] = []
        can_use = True
        row_mode = os.environ.get("TRTLLM_SPEC_PENALTY_ROW_MODE",
                                  "all").strip().lower()
        if row_mode not in ("all", "root"):
            row_mode = "all"

        for request in requests:
            raw_slot = getattr(request, "py_seq_slot", None)
            slot = int(raw_slot) if raw_slot is not None else -1
            valid_slot = (self._valid_seq_slot(slot)
                          and not getattr(request, "is_dummy", False))
            effective_slot = slot if valid_slot else -1
            seq_slots.append(effective_slot)

            sampling_config = request.sampling_config
            frequency_penalty = float(
                self._sampling_config_value(sampling_config,
                                            "frequency_penalty", 0.0))
            presence_penalty = float(
                self._sampling_config_value(sampling_config,
                                            "presence_penalty", 0.0))
            prompt_ignore_length = int(
                self._sampling_config_value(sampling_config,
                                            "prompt_ignore_length", 0))
            raw_prompt_len = self._prompt_len(request)

            # The device-history fast path intentionally covers the current
            # NVBug workload: frequency penalty over generated tokens only.
            # Other token-history semantics fall back to the slower probe path.
            if presence_penalty != 0.0 or prompt_ignore_length < raw_prompt_len:
                can_use = False
                break

            if not valid_slot:
                frequency_penalty = 0.0
            seq_frequency_penalties.append(frequency_penalty)

            generated_len = max(request.get_num_tokens(0) - raw_prompt_len, 0)
            if valid_slot and generated_len == 0:
                reset_slots.append(slot)

            from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequestState
            num_rows = 1 + self.runtime_draft_len if request.state == LlmRequestState.GENERATION_IN_PROGRESS else 1
            row_slots.append(effective_slot)
            frequency_penalties.append(frequency_penalty)
            if num_rows > 1:
                if row_mode == "all":
                    row_slots.extend(effective_slot for _ in range(num_rows - 1))
                    frequency_penalties.extend(frequency_penalty
                                               for _ in range(num_rows - 1))
                else:
                    row_slots.extend(-1 for _ in range(num_rows - 1))
                    frequency_penalties.extend(0.0 for _ in range(num_rows - 1))

        if not can_use or not row_slots:
            self.use_device_penalty_history = False
            return False

        self._ensure_device_penalty_history_buffers()
        if not self.use_device_penalty_history and self.device_penalty_history_tokens is None:
            return False
        assert self.device_penalty_history_tokens is not None
        assert self.device_penalty_history_lens is not None
        assert self.device_penalty_row_slots is not None
        assert self.device_penalty_seq_slots is not None
        assert self.device_frequency_penalties is not None
        assert self.device_seq_frequency_penalties is not None

        if reset_slots:
            reset_slots_cuda = torch.tensor(reset_slots,
                                            dtype=torch.int64,
                                            pin_memory=prefer_pinned()).to(
                                                "cuda", non_blocking=True)
            self.device_penalty_history_lens.index_fill_(0, reset_slots_cuda, 0)

        self.device_penalty_seq_slots[:len(seq_slots)].copy_(
            torch.tensor(seq_slots, dtype=torch.int32,
                         pin_memory=prefer_pinned()),
            non_blocking=True)
        self.device_penalty_row_slots[:len(row_slots)].copy_(
            torch.tensor(row_slots, dtype=torch.int32,
                         pin_memory=prefer_pinned()),
            non_blocking=True)
        self.device_frequency_penalties[:len(frequency_penalties)].copy_(
            torch.tensor(frequency_penalties,
                         dtype=torch.float32,
                         pin_memory=prefer_pinned()),
            non_blocking=True)
        self.device_seq_frequency_penalties[:len(seq_frequency_penalties)].copy_(
            torch.tensor(seq_frequency_penalties,
                         dtype=torch.float32,
                         pin_memory=prefer_pinned()),
            non_blocking=True)
        self.use_device_penalty_history = True
        return True

    def append_accepted_tokens_to_penalty_history(
            self, accepted_tokens: torch.Tensor,
            num_accepted_tokens: torch.Tensor, batch_size: int) -> None:
        if (self.use_device_penalty_counts
                and self.device_penalty_count_seq_slots is not None):
            if (self.device_penalty_count_mode == "dense"
                    and self.device_penalty_token_counts is not None):
                from .one_model_sampler import append_accepted_tokens_to_counts
                append_accepted_tokens_to_counts(
                    self.device_penalty_token_counts,
                    self.device_penalty_count_seq_slots[:batch_size],
                    accepted_tokens[:batch_size].contiguous(),
                    num_accepted_tokens[:batch_size].contiguous())
                return
            if (self.device_penalty_count_mode == "sparse"
                    and self.device_penalty_sparse_token_ids is not None
                    and self.device_penalty_sparse_token_counts is not None
                    and self.device_penalty_sparse_count_lens is not None):
                from .one_model_sampler import append_accepted_tokens_to_sparse_counts
                append_accepted_tokens_to_sparse_counts(
                    self.device_penalty_sparse_token_ids,
                    self.device_penalty_sparse_token_counts,
                    self.device_penalty_sparse_count_lens,
                    self.device_penalty_count_seq_slots[:batch_size],
                    accepted_tokens[:batch_size].contiguous(),
                    num_accepted_tokens[:batch_size].contiguous(),
                    self.device_penalty_count_vocab_size)
                return

        if not self.use_device_penalty_history:
            return
        if self.device_penalty_history_tokens is None:
            return
        assert self.device_penalty_history_lens is not None
        assert self.device_penalty_seq_slots is not None

        from .one_model_sampler import append_accepted_tokens_to_history
        append_accepted_tokens_to_history(
            self.device_penalty_history_tokens,
            self.device_penalty_history_lens,
            self.device_penalty_seq_slots[:batch_size],
            accepted_tokens[:batch_size].contiguous(),
            num_accepted_tokens[:batch_size].contiguous())

    def _populate_recent_token_penalties_for_one_model(
            self, requests: list["LlmRequest"]) -> None:
        if not self.allow_advanced_sampling or not self.spec_dec_mode.use_one_engine(
        ):
            return

        width = int(os.environ.get("TRTLLM_SPEC_RECENT_PENALTY_TOKENS", "0"))
        width = max(width, 0)
        if width == 0:
            return

        row_token_ids: list[list[int]] = []
        row_penalty_values: list[list[float]] = []
        seq_token_ids: list[list[int]] = []
        seq_penalty_values: list[list[float]] = []
        any_penalty = False

        for request in requests:
            sampling_config = request.sampling_config
            frequency_penalty = float(
                self._sampling_config_value(sampling_config,
                                            "frequency_penalty", 0.0))
            presence_penalty = float(
                self._sampling_config_value(sampling_config,
                                            "presence_penalty", 0.0))
            prompt_ignore_length = int(
                self._sampling_config_value(sampling_config,
                                            "prompt_ignore_length", 0))

            from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequestState
            num_rows = 1 + self.runtime_draft_len if request.state == LlmRequestState.GENERATION_IN_PROGRESS else 1

            if frequency_penalty == 0.0 and presence_penalty == 0.0:
                ids = [0] * width
                penalties = [0.0] * width
            else:
                prompt_ignore_length = self._effective_prompt_ignore_length(
                    request, prompt_ignore_length)
                tokens = request.get_tokens(0)
                recent_start = max(prompt_ignore_length, len(tokens) - width)
                counts: dict[int, int] = {}
                for token in tokens[recent_start:]:
                    if token < 0:
                        continue
                    counts[token] = counts.get(token, 0) + 1
                items = list(counts.items())[:width]
                ids = [token for token, _ in items]
                penalties = [
                    presence_penalty + frequency_penalty * count
                    for _, count in items
                ]
                if penalties:
                    any_penalty = True
                pad = width - len(ids)
                if pad > 0:
                    ids.extend([0] * pad)
                    penalties.extend([0.0] * pad)

            for _ in range(num_rows):
                row_token_ids.append(ids)
                row_penalty_values.append(penalties)
            seq_token_ids.append(ids)
            seq_penalty_values.append(penalties)

        if not row_token_ids:
            return

        self._ensure_recent_penalty_buffers(width)
        assert self.recent_penalty_token_ids is not None
        assert self.recent_penalty_values is not None
        assert self.recent_seq_penalty_token_ids is not None
        assert self.recent_seq_penalty_values is not None

        num_rows = len(row_token_ids)
        num_seqs = len(seq_token_ids)
        if not any_penalty:
            self.recent_penalty_values[:num_rows].zero_()
            self.recent_seq_penalty_values[:num_seqs].zero_()
            return

        self.recent_penalty_token_ids[:num_rows].copy_(
            torch.tensor(row_token_ids,
                         dtype=torch.long,
                         pin_memory=prefer_pinned()),
            non_blocking=True)
        self.recent_penalty_values[:num_rows].copy_(
            torch.tensor(row_penalty_values,
                         dtype=torch.float32,
                         pin_memory=prefer_pinned()),
            non_blocking=True)
        self.recent_seq_penalty_token_ids[:num_seqs].copy_(
            torch.tensor(seq_token_ids,
                         dtype=torch.long,
                         pin_memory=prefer_pinned()),
            non_blocking=True)
        self.recent_seq_penalty_values[:num_seqs].copy_(
            torch.tensor(seq_penalty_values,
                         dtype=torch.float32,
                         pin_memory=prefer_pinned()),
            non_blocking=True)

    def populate_sampling_params_for_one_model(
            self, requests: list["LlmRequest"]) -> None:
        """
        Set up topp/topk/temperatures for 1-model sampler.
        """
        from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequestState
        from tensorrt_llm.sampling_params import SamplingParams

        if not self.allow_advanced_sampling or not self.spec_dec_mode.use_one_engine(
        ):
            return

        if self.temperatures is None:
            # Ensures determinism across ranks.
            torch.manual_seed(0)

        temperatures = []
        top_ks = []
        top_ps = []

        # Need to use a very small value for temperature when disabled to avoid division by 0
        DISABLE_TEMP_VAL = 1e-5
        # Very large values disable topk.
        DISABLE_TOPK_VAL = torch.iinfo(torch.int32).max
        DISABLE_TOPP_VAL = 1.0

        for request in requests:
            sampling_config = request.sampling_config
            temp = sampling_config.temperature
            temp_val = temp[0] if temp is not None and len(temp) > 0 else None

            tk = sampling_config.top_k
            tk_val = tk[0] if tk is not None and len(tk) > 0 else None

            tp = sampling_config.top_p
            tp_val = tp[0] if tp is not None and len(tp) > 0 else None

            # Context requests have no draft tokens yet.
            num_tokens = 1 + self.runtime_draft_len if request.state == LlmRequestState.GENERATION_IN_PROGRESS else 1

            is_greedy = SamplingParams.params_imply_greedy_decoding(
                temperature=temp_val,
                top_k=tk_val,
                top_p=tp_val,
                use_beam_search=False)

            temp_val = DISABLE_TEMP_VAL if is_greedy or temp_val is None or temp_val == 0 else temp_val
            tk_val = DISABLE_TOPK_VAL if is_greedy or tk_val is None or tk_val <= 0 else tk_val
            tp_val = DISABLE_TOPP_VAL if is_greedy or tp_val is None else tp_val

            temperatures.extend(temp_val for _ in range(num_tokens))
            top_ks.extend(tk_val for _ in range(num_tokens))
            top_ps.extend(tp_val for _ in range(num_tokens))

        if self.temperatures is None:
            self.temperatures = torch.ones(
                (self.max_draft_len + 1) * self.max_num_requests,
                dtype=torch.float32,
                device='cuda')
            self.top_ks = torch.zeros(
                (self.max_draft_len + 1) * self.max_num_requests,
                dtype=torch.int32,
                device='cuda')
            self.top_ps = torch.ones(
                (self.max_draft_len + 1) * self.max_num_requests,
                dtype=torch.float32,
                device='cuda')

        self.temperatures[:len(temperatures)].copy_(torch.tensor(
            temperatures, dtype=torch.float32, pin_memory=prefer_pinned()),
                                                    non_blocking=True)
        self.top_ks[:len(top_ks)].copy_(torch.tensor(
            top_ks, dtype=torch.int32, pin_memory=prefer_pinned()),
                                        non_blocking=True)
        self.top_ps[:len(top_ps)].copy_(torch.tensor(
            top_ps, dtype=torch.float32, pin_memory=prefer_pinned()),
                                        non_blocking=True)
        if self._populate_device_count_frequency_penalties(requests):
            return
        if not self._populate_device_history_frequency_penalties(requests):
            self._populate_recent_token_penalties_for_one_model(requests)


class SpecWorkerBase(nn.Module, ABC):
    """
    Base class for speculative decoding workers.
    Provides common functionality for sampling and token handling.
    """

    def __init__(self, use_separate_draft_kv_cache: bool = False):
        super().__init__()
        self.guided_decoder: Optional["CapturableGuidedDecoder"] = None
        self.force_num_accepted_tokens = get_force_num_accepted_tokens()
        self.use_flashinfer = IS_FLASHINFER_AVAILABLE and flashinfer.__version__ >= "0.6.4"
        self.seed: Optional[torch.Tensor] = None
        self.offset: Optional[torch.Tensor] = None
        self.use_separate_draft_kv_cache = use_separate_draft_kv_cache

    @property
    @abstractmethod
    def max_draft_len(self) -> int:
        """
        Returns the maximum draft length for this worker.
        Subclasses should override this property.
        """

    def skip_forward(
        self,
        input_ids,
        position_ids,
        hidden_states,
        logits,
        attn_metadata,
        spec_metadata,
        draft_model,
    ):
        """Skip spec dec for non-last rank (PP). Returns placeholder outputs."""
        batch_size = attn_metadata.num_seqs
        accepted_tokens = torch.empty((batch_size, (self.max_draft_len + 1)),
                                      dtype=torch.int,
                                      device=logits.device)
        num_accepted_tokens = torch.ones(batch_size,
                                         dtype=torch.int,
                                         device=logits.device)
        next_draft_tokens = torch.empty((batch_size, self.max_draft_len),
                                        dtype=torch.int,
                                        device=logits.device)
        next_new_tokens = torch.empty((batch_size, (self.max_draft_len + 1)),
                                      dtype=torch.int,
                                      device=logits.device)
        outputs = {
            'logits': logits,
            'new_tokens': accepted_tokens,
            'new_tokens_lens': num_accepted_tokens,
            'next_draft_tokens': next_draft_tokens,
            'next_new_tokens': next_new_tokens
        }
        return self._add_penalty_history_outputs(outputs, spec_metadata)

    def skip_drafting(
        self,
        input_ids,
        position_ids,
        hidden_states,
        logits,
        attn_metadata,
        spec_metadata,
        draft_model,
    ):
        """
        Used when speculation is disabled for dynamic draft length (e.g., large batch size).
        """
        batch_size = attn_metadata.num_seqs
        num_contexts = attn_metadata.num_contexts

        if self.guided_decoder is not None:
            self.guided_decoder.execute(logits)

        target_tokens = self._sample_tokens_for_batch(logits, spec_metadata,
                                                      num_contexts, batch_size)

        accepted_tokens = torch.zeros((batch_size, 1),
                                      dtype=torch.int,
                                      device=logits.device)
        accepted_tokens[:, 0] = target_tokens

        num_accepted_tokens = torch.ones(batch_size,
                                         dtype=torch.int,
                                         device=logits.device)
        next_draft_tokens = torch.zeros((batch_size, 0),
                                        dtype=torch.int,
                                        device=logits.device)

        next_new_tokens = torch.zeros((batch_size, 1),
                                      dtype=torch.int,
                                      device=logits.device)
        next_new_tokens[:, 0] = target_tokens

        outputs = {
            'logits': logits,
            'new_tokens': accepted_tokens,
            'new_tokens_lens': num_accepted_tokens,
            'next_draft_tokens': next_draft_tokens,
            'next_new_tokens': next_new_tokens
        }
        return self._add_penalty_history_outputs(outputs, spec_metadata)

    def set_guided_decoder(self,
                           guided_decoder: "CapturableGuidedDecoder") -> bool:
        self.guided_decoder = guided_decoder
        return True

    def _prepare_attn_metadata_for_spec_dec(self, attn_metadata):
        """
        Prepare attention metadata before speculative decoding draft token generation.
        Saves current state for later restoration.
        """
        attn_metadata.prepare_for_spec_dec("_seq_lens", "_seq_lens_cuda")

    def _restore_attn_metadata_from_spec_dec(self, attn_metadata):
        """
        Restore attention metadata after speculative decoding draft token generation.
        """
        attn_metadata.restore_from_spec_dec()
        attn_metadata.on_update()

    def _apply_force_accepted_tokens(self, num_accepted_tokens, num_contexts,
                                     runtime_draft_len: int):
        """
        Apply forced number of accepted tokens if environment variable is set.
        This is used for testing and debugging.

        Args:
            num_accepted_tokens: Tensor of shape [batch_size] with current accepted counts
            num_contexts: Number of context (prefill) requests
            runtime_draft_len: The draft length for the current iteration.

        Returns:
            Modified num_accepted_tokens tensor
        """
        if self.force_num_accepted_tokens != 0:
            # total tokens per iteration = accepted draft tokens + 1 target token
            force_total_tokens = min(self.force_num_accepted_tokens + 1,
                                     runtime_draft_len + 1)
            num_accepted_tokens[num_contexts:] = force_total_tokens
        return num_accepted_tokens

    def _sample_and_accept_draft_tokens_base(
        self,
        logits: torch.Tensor,
        draft_tokens: torch.Tensor,
        num_contexts: int,
        batch_size: int,
        spec_metadata,
    ):
        """
        Base implementation for sampling and accepting draft tokens.
        Uses strict acceptance (token equality with cumulative product).

        This is the common logic shared between Eagle3 and MTP (when relaxed
        acceptance is disabled).

        Args:
            logits: [num_tokens, vocab_size] - Target model logits
            draft_tokens: [num_gens, runtime_draft_len] - Previously predicted draft tokens
            num_contexts: Number of context requests
            batch_size: Total number of requests
            spec_metadata: Speculative decoding metadata

        Returns:
            accepted_tokens: [batch_size, runtime_draft_len + 1] - Accepted tokens (padded)
            num_accepted_tokens: [batch_size] - Number of accepted tokens per request
        """
        # Derive draft length from the actual draft_tokens shape rather than
        # spec_metadata.runtime_draft_len, because they can differ: PARD sets
        # runtime_draft_len = 2K-1 for input sizing but only passes K draft
        # tokens for acceptance;
        runtime_draft_len = draft_tokens.shape[-1]
        num_gens = batch_size - num_contexts

        if logits.dim() == 1:
            logits = logits.unsqueeze(0)

        # Allocate return buffers
        accepted_tokens = torch.empty((batch_size, runtime_draft_len + 1),
                                      dtype=torch.int,
                                      device=logits.device)
        num_accepted_tokens = torch.ones(batch_size,
                                         dtype=torch.int,
                                         device=logits.device)

        # Sample tokens using per-request sampling parameters
        target_tokens = self._sample_tokens_for_batch(
            logits,
            spec_metadata,
            num_contexts,
            batch_size,
            draft_tokens=draft_tokens)

        # Context requests: only accept the sampled token (no draft tokens yet)
        accepted_tokens[:num_contexts, 0] = target_tokens[:num_contexts]

        # Generation requests: verify draft tokens against target tokens
        gen_target_tokens = target_tokens[num_contexts:].reshape(
            num_gens, runtime_draft_len + 1)
        accepted_tokens[num_contexts:, :runtime_draft_len +
                        1] = gen_target_tokens

        # Compare draft tokens with target tokens using cumulative product
        # Counts consecutive matches from the start
        num_accepted_tokens[num_contexts:] += torch.cumprod(
            (draft_tokens == gen_target_tokens[:, :runtime_draft_len]).int(),
            dim=-1).sum(1)

        # Apply force override if set
        num_accepted_tokens = self._apply_force_accepted_tokens(
            num_accepted_tokens, num_contexts, runtime_draft_len)

        return accepted_tokens, num_accepted_tokens

    @staticmethod
    def _add_penalty_history_outputs(outputs: dict[str, torch.Tensor],
                                     spec_metadata: SpecMetadata):
        if spec_metadata.sampling_request_ids is not None:
            outputs["penalty_sampling_request_ids"] = spec_metadata.sampling_request_ids
        if spec_metadata.sampling_seq_slots is not None:
            outputs["penalty_sampling_seq_slots"] = spec_metadata.sampling_seq_slots
        if spec_metadata.use_device_penalty_counts:
            if spec_metadata.device_penalty_count_seq_slots is not None:
                outputs[
                    "penalty_count_seq_slots"] = spec_metadata.device_penalty_count_seq_slots
            if (spec_metadata.device_penalty_count_mode == "dense"
                    and spec_metadata.device_penalty_token_counts is not None):
                outputs["penalty_token_counts"] = spec_metadata.device_penalty_token_counts
            elif (spec_metadata.device_penalty_count_mode == "sparse"
                  and spec_metadata.device_penalty_sparse_token_ids is not None
                  and spec_metadata.device_penalty_sparse_token_counts is not None
                  and spec_metadata.device_penalty_sparse_count_lens is not None):
                outputs["penalty_sparse_token_ids"] = spec_metadata.device_penalty_sparse_token_ids
                outputs["penalty_sparse_token_counts"] = spec_metadata.device_penalty_sparse_token_counts
                outputs["penalty_sparse_count_lens"] = spec_metadata.device_penalty_sparse_count_lens
                outputs["penalty_count_vocab_size"] = spec_metadata.device_penalty_count_vocab_size
        if (spec_metadata.use_device_penalty_history
                and spec_metadata.device_penalty_history_tokens is not None
                and spec_metadata.device_penalty_history_lens is not None):
            if spec_metadata.device_penalty_seq_slots is not None:
                outputs[
                    "penalty_history_seq_slots"] = spec_metadata.device_penalty_seq_slots
            outputs["penalty_history_tokens"] = spec_metadata.device_penalty_history_tokens
            outputs["penalty_history_lens"] = spec_metadata.device_penalty_history_lens
        return outputs

    def _draft_sampler_greedy(self, logits: torch.Tensor, d2t=None):
        """
        Simple greedy draft token sampling using argmax.

        Args:
            logits: [num_tokens, vocab_size] - Draft model logits
            d2t: Optional dictionary offset tensor for vocab mapping

        Returns:
            draft_tokens: [num_tokens] - Sampled draft token ids (int32)
        """
        draft_tokens = torch.argmax(logits, dim=-1)

        # Apply d2t (offsets between draft and target model dictionaries)
        if d2t is not None:
            draft_tokens = d2t[draft_tokens] + draft_tokens

        return draft_tokens.type(torch.int32)

    @staticmethod
    def _draft_prefix_frequency_penalties(spec_metadata: SpecMetadata):
        if spec_metadata.device_count_seq_frequency_penalties is not None:
            return spec_metadata.device_count_seq_frequency_penalties
        if spec_metadata.device_seq_frequency_penalties is not None:
            return spec_metadata.device_seq_frequency_penalties
        return None

    def _apply_draft_prefix_penalty_values(
            self, logits: torch.Tensor, spec_metadata: SpecMetadata,
            row_token_ids: torch.Tensor, row_penalty_values: torch.Tensor,
            num_rows: int, width: int) -> torch.Tensor:
        if num_rows <= 0 or width <= 0:
            return logits
        if row_token_ids is None or row_penalty_values is None:
            return logits
        if os.environ.get("TRTLLM_SPEC_USE_PENALTY_OP", "0") == "1":
            from .one_model_sampler import apply_recent_token_penalties
            return apply_recent_token_penalties(
                logits, row_token_ids[:num_rows, :width],
                row_penalty_values[:num_rows, :width])
        # scatter_add handles duplicate prefix tokens in the same row correctly.
        logits.scatter_add_(1, row_token_ids[:num_rows, :width].long(),
                            -row_penalty_values[:num_rows, :width].to(
                                logits.dtype))
        return logits

    def _apply_target_draft_prefix_frequency_penalty(
            self, logits: torch.Tensor, spec_metadata: SpecMetadata,
            num_contexts: int, batch_size: int,
            draft_tokens: Optional[torch.Tensor]) -> torch.Tensor:
        if os.environ.get("TRTLLM_SPEC_APPLY_DRAFT_PREFIX_PENALTY",
                          "0") != "1":
            return logits
        if draft_tokens is None or draft_tokens.numel() == 0:
            return logits
        runtime_draft_len = int(draft_tokens.shape[-1])
        if runtime_draft_len <= 0:
            return logits
        frequency_penalties = self._draft_prefix_frequency_penalties(
            spec_metadata)
        if frequency_penalties is None:
            return logits
        num_gens = batch_size - num_contexts
        if num_gens <= 0:
            return logits

        num_tokens = num_contexts + num_gens * (runtime_draft_len + 1)
        spec_metadata._ensure_draft_prefix_penalty_buffers(runtime_draft_len)
        token_ids = spec_metadata.draft_prefix_penalty_token_ids
        penalty_values = spec_metadata.draft_prefix_penalty_values
        rows = spec_metadata.draft_prefix_penalty_rows
        assert token_ids is not None
        assert penalty_values is not None
        assert rows is not None

        token_ids[:num_tokens, :runtime_draft_len].zero_()
        penalty_values[:num_tokens, :runtime_draft_len].zero_()
        gen_frequency_penalties = frequency_penalties[
            num_contexts:batch_size].to(torch.float32)
        gen_rows = rows[:num_gens]
        row_stride = runtime_draft_len + 1
        for pos in range(1, runtime_draft_len + 1):
            target_rows = num_contexts + gen_rows * row_stride + pos
            token_ids[target_rows, :pos].copy_(draft_tokens[:, :pos])
            penalty_values[target_rows, :pos].copy_(
                gen_frequency_penalties.unsqueeze(1).expand(-1, pos))
        return self._apply_draft_prefix_penalty_values(
            logits, spec_metadata, token_ids, penalty_values, num_tokens,
            runtime_draft_len)

    def _apply_draft_step_prefix_frequency_penalty(
            self, logits: torch.Tensor, spec_metadata: SpecMetadata,
            batch_size: int,
            draft_prefix_tokens: Optional[torch.Tensor]) -> torch.Tensor:
        if os.environ.get("TRTLLM_SPEC_APPLY_DRAFT_PREFIX_PENALTY",
                          "0") != "1":
            return logits
        if draft_prefix_tokens is None or draft_prefix_tokens.numel() == 0:
            return logits
        prefix_len = int(draft_prefix_tokens.shape[-1])
        if prefix_len <= 0:
            return logits
        frequency_penalties = self._draft_prefix_frequency_penalties(
            spec_metadata)
        if frequency_penalties is None:
            return logits

        spec_metadata._ensure_draft_prefix_penalty_buffers(prefix_len)
        token_ids = spec_metadata.draft_prefix_penalty_token_ids
        penalty_values = spec_metadata.draft_prefix_penalty_values
        assert token_ids is not None
        assert penalty_values is not None

        token_ids[:batch_size, :prefix_len].copy_(
            draft_prefix_tokens[:batch_size, :prefix_len])
        penalty_values[:batch_size, :prefix_len].copy_(
            frequency_penalties[:batch_size].to(torch.float32).unsqueeze(1).
            expand(-1, prefix_len))
        return self._apply_draft_prefix_penalty_values(
            logits, spec_metadata, token_ids, penalty_values, batch_size,
            prefix_len)

    def _maybe_apply_history_penalty_to_draft_logits(
            self, logits: torch.Tensor, spec_metadata: SpecMetadata,
            batch_size: int, d2t=None,
            draft_prefix_tokens: Optional[torch.Tensor] = None):
        if os.environ.get("TRTLLM_SPEC_APPLY_HISTORY_TO_DRAFT", "0") != "1":
            return self._apply_draft_step_prefix_frequency_penalty(
                logits, spec_metadata, batch_size, draft_prefix_tokens)
        if d2t is not None:
            return logits
        if (spec_metadata.use_device_penalty_counts
                and spec_metadata.device_penalty_count_seq_slots is not None
                and spec_metadata.device_count_seq_frequency_penalties is not None):
            from .one_model_sampler import (apply_count_frequency_penalty,
                                            apply_sparse_count_frequency_penalty)
            spec_metadata.ensure_device_penalty_count_buffers(
                int(logits.shape[-1]))
            if spec_metadata.device_penalty_count_mode == "dense":
                if spec_metadata.device_penalty_token_counts is None:
                    return logits
                logits = apply_count_frequency_penalty(
                    logits, spec_metadata.device_penalty_token_counts,
                    spec_metadata.device_penalty_count_seq_slots[:batch_size],
                    spec_metadata.device_count_seq_frequency_penalties[:batch_size])
                return self._apply_draft_step_prefix_frequency_penalty(
                    logits, spec_metadata, batch_size, draft_prefix_tokens)
            if (spec_metadata.device_penalty_sparse_token_ids is None
                    or spec_metadata.device_penalty_sparse_token_counts is None
                    or spec_metadata.device_penalty_sparse_count_lens is None):
                return logits
            logits = apply_sparse_count_frequency_penalty(
                logits, spec_metadata.device_penalty_sparse_token_ids,
                spec_metadata.device_penalty_sparse_token_counts,
                spec_metadata.device_penalty_sparse_count_lens,
                spec_metadata.device_penalty_count_seq_slots[:batch_size],
                spec_metadata.device_count_seq_frequency_penalties[:batch_size])
            return self._apply_draft_step_prefix_frequency_penalty(
                logits, spec_metadata, batch_size, draft_prefix_tokens)
        if (spec_metadata.recent_seq_penalty_token_ids is not None
                and spec_metadata.recent_seq_penalty_values is not None):
            from .one_model_sampler import apply_recent_token_penalties
            logits = apply_recent_token_penalties(
                logits,
                spec_metadata.recent_seq_penalty_token_ids[:batch_size],
                spec_metadata.recent_seq_penalty_values[:batch_size])
            return self._apply_draft_step_prefix_frequency_penalty(
                logits, spec_metadata, batch_size, draft_prefix_tokens)
        if (not spec_metadata.use_device_penalty_history
                or logits.dtype != torch.float32
                or spec_metadata.device_penalty_history_tokens is None
                or spec_metadata.device_penalty_history_lens is None
                or spec_metadata.device_penalty_seq_slots is None
                or spec_metadata.device_seq_frequency_penalties is None):
            return logits

        from .one_model_sampler import apply_history_frequency_penalty
        logits = apply_history_frequency_penalty(
            logits, spec_metadata.device_penalty_history_tokens,
            spec_metadata.device_penalty_history_lens,
            spec_metadata.device_penalty_seq_slots[:batch_size],
            spec_metadata.device_seq_frequency_penalties[:batch_size])
        return self._apply_draft_step_prefix_frequency_penalty(
            logits, spec_metadata, batch_size, draft_prefix_tokens)

    def _execute_guided_decoder_if_present(self, logits):
        """Execute guided decoder on target model logits if available."""
        if self.guided_decoder is not None:
            self.guided_decoder.execute(logits)

    def _prepare_next_new_tokens(self, accepted_tokens, next_draft_tokens,
                                 batch_indices_cuda, batch_size,
                                 num_accepted_tokens):
        """
        Prepare next_new_tokens for overlap scheduler support.

        Args:
            accepted_tokens: [batch_size, max_draft_len + 1] - Accepted tokens
            next_draft_tokens: [batch_size, runtime_draft_len] - Predicted draft tokens (NOT padded)
            batch_indices_cuda: Batch indices tensor
            batch_size: Number of requests
            num_accepted_tokens: [batch_size] - Number of accepted tokens per request

        Returns:
            next_new_tokens: [batch_size, runtime_draft_len + 1] - Input tokens for next iteration
        """
        next_new_tokens = accepted_tokens[batch_indices_cuda[:batch_size],
                                          num_accepted_tokens - 1].unsqueeze(1)
        next_new_tokens = torch.concat([next_new_tokens, next_draft_tokens],
                                       dim=1)
        return next_new_tokens

    def _prepare_context_input_ids(self, input_ids, num_ctx_tokens, gather_ids,
                                   accepted_tokens, num_contexts):
        """
        Prepare context input IDs for draft model forward.
        Shifts input IDs left by 1 and places the first accepted token at gather positions.

        Args:
            input_ids: Original input IDs tensor
            num_ctx_tokens: Number of context tokens
            gather_ids: Indices for placing accepted tokens (last token positions)
            accepted_tokens: [batch_size, max_draft_len + 1] - Accepted tokens
            num_contexts: Number of context requests

        Returns:
            input_ids_ctx: Prepared context input IDs
        """
        if num_ctx_tokens > 0:
            input_prompt_ids = input_ids[:num_ctx_tokens]
            input_ids_ctx = torch.empty_like(input_prompt_ids,
                                             dtype=torch.int32,
                                             device="cuda")
            input_ids_ctx[:-1].copy_(input_prompt_ids[1:])
            input_ids_ctx[
                gather_ids[:num_contexts]] = accepted_tokens[:num_contexts, 0]
            return input_ids_ctx
        else:
            return torch.empty(0, dtype=torch.int32, device="cuda")

    def get_draft_kv_cache_manager(self, resource_manager):
        """
        Get the draft KV cache manager if using separate KV cache layouts.
        """
        if self.use_separate_draft_kv_cache and resource_manager is not None:
            return resource_manager.get_resource_manager(
                ResourceManagerType.DRAFT_KV_CACHE_MANAGER)
        return None

    @contextmanager
    def draft_kv_cache_context(self, attn_metadata, draft_kv_cache_manager):
        """
        Context manager to temporarily switch to draft KV cache manager in one-engine speculative decoding.

        This swaps both the kv_cache_manager reference AND the block offset tensors,
        since the target and draft KV caches have different block layouts.
        """

        # draft_kv_cache_manager is None if using two-engine speculative decoding or not enabling separate draft KV cache.
        if draft_kv_cache_manager is None:
            yield
            return

        # Only TrtllmAttentionMetadata supports separate draft KV cache layouts
        if not isinstance(attn_metadata, TrtllmAttentionMetadata):
            yield
            return

        # Check if draft KV cache block offsets are allocated
        draft_block_offsets = getattr(attn_metadata,
                                      'draft_kv_cache_block_offsets', None)
        if draft_block_offsets is None:
            # Draft KV cache block offsets not allocated, skip switching
            yield
            return

        # Save main KV cache manager and block offsets
        target_kv_cache_manager = attn_metadata.kv_cache_manager
        target_kv_cache_block_offsets = attn_metadata.kv_cache_block_offsets
        target_host_kv_cache_block_offsets = attn_metadata.host_kv_cache_block_offsets

        # Switch to draft KV cache manager and its block offsets
        attn_metadata.kv_cache_manager = draft_kv_cache_manager
        attn_metadata.kv_cache_block_offsets = attn_metadata.draft_kv_cache_block_offsets
        attn_metadata.host_kv_cache_block_offsets = draft_kv_cache_manager.host_kv_cache_block_offsets

        try:
            yield
        finally:
            # Restore main KV cache manager and block offsets
            attn_metadata.kv_cache_manager = target_kv_cache_manager
            attn_metadata.kv_cache_block_offsets = target_kv_cache_block_offsets
            attn_metadata.host_kv_cache_block_offsets = target_host_kv_cache_block_offsets

    def _sample_tokens_for_batch(
        self,
        logits: torch.Tensor,
        spec_metadata: SpecMetadata,
        num_contexts: int,
        batch_size: int,
        draft_tokens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Sample tokens from logits using per-request sampling parameters.
        Supports both greedy and non-greedy sampling.

        Args:
            logits: [num_tokens, vocab_size] - Logits to sample from
            spec_metadata: Metadata containing sampling parameters
            num_contexts: Number of context requests in the batch
            batch_size: Number of requests in the batch

        Returns:
            sampled_tokens: [num_tokens] - Sampled token ids
        """
        if spec_metadata.allow_advanced_sampling:
            from .one_model_sampler import (apply_count_frequency_penalty,
                                            apply_history_frequency_penalty,
                                            apply_recent_token_penalties,
                                            apply_sparse_count_frequency_penalty,
                                            sampling_batch_spec_dec_one_model)

            num_gens = batch_size - num_contexts
            num_tokens = num_contexts + num_gens * (
                spec_metadata.runtime_draft_len + 1)

            temperatures = spec_metadata.temperatures[:num_tokens]
            top_ks = spec_metadata.top_ks[:num_tokens]
            top_ps = spec_metadata.top_ps[:num_tokens]
            if (spec_metadata.use_device_penalty_counts
                    and spec_metadata.device_penalty_count_row_slots is not None
                    and spec_metadata.device_count_frequency_penalties is not None):
                spec_metadata.ensure_device_penalty_count_buffers(
                    int(logits.shape[-1]))
                if (spec_metadata.device_penalty_count_mode == "dense"
                        and spec_metadata.device_penalty_token_counts is not None):
                    logits = apply_count_frequency_penalty(
                        logits,
                        spec_metadata.device_penalty_token_counts,
                        spec_metadata.device_penalty_count_row_slots[:num_tokens],
                        spec_metadata.device_count_frequency_penalties[:
                                                                      num_tokens])
                elif (spec_metadata.device_penalty_count_mode == "sparse"
                      and spec_metadata.device_penalty_sparse_token_ids is not None
                      and spec_metadata.device_penalty_sparse_token_counts is not None
                      and spec_metadata.device_penalty_sparse_count_lens is not None):
                    logits = apply_sparse_count_frequency_penalty(
                        logits,
                        spec_metadata.device_penalty_sparse_token_ids,
                        spec_metadata.device_penalty_sparse_token_counts,
                        spec_metadata.device_penalty_sparse_count_lens,
                        spec_metadata.device_penalty_count_row_slots[:num_tokens],
                        spec_metadata.device_count_frequency_penalties[:
                                                                      num_tokens])
            elif (spec_metadata.use_device_penalty_history
                    and logits.dtype == torch.float32
                    and spec_metadata.device_penalty_history_tokens is not None
                    and spec_metadata.device_penalty_history_lens is not None
                    and spec_metadata.device_penalty_row_slots is not None
                    and spec_metadata.device_frequency_penalties is not None):
                logits = apply_history_frequency_penalty(
                    logits,
                    spec_metadata.device_penalty_history_tokens,
                    spec_metadata.device_penalty_history_lens,
                    spec_metadata.device_penalty_row_slots[:num_tokens],
                    spec_metadata.device_frequency_penalties[:num_tokens])
            else:
                recent_penalty_token_ids = (
                    None if spec_metadata.recent_penalty_token_ids is None else
                    spec_metadata.recent_penalty_token_ids[:num_tokens])
                recent_penalty_values = (
                    None if spec_metadata.recent_penalty_values is None else
                    spec_metadata.recent_penalty_values[:num_tokens])
                if recent_penalty_token_ids is not None and recent_penalty_values is not None:
                    logits = apply_recent_token_penalties(
                        logits, recent_penalty_token_ids, recent_penalty_values)

            logits = self._apply_target_draft_prefix_frequency_penalty(
                logits, spec_metadata, num_contexts, batch_size, draft_tokens)

            if self.use_flashinfer:
                top_ks = top_ks.clamp(min=1, max=logits.shape[-1] - 1)
                # Lazily initialize seed/offset tensors on correct device
                if self.seed is None:
                    self.seed = torch.tensor([0],
                                             dtype=torch.int64,
                                             device=logits.device)
                    self.offset = torch.tensor([0],
                                               dtype=torch.int64,
                                               device=logits.device)
                self.seed += 1
                self.seed %= (2**31)

            sampled_tokens = sampling_batch_spec_dec_one_model(
                logits,
                temperatures,
                top_ks,
                top_ps,
                use_flashinfer=self.use_flashinfer,
                seed=self.seed,
                offset=self.offset)
        else:
            sampled_tokens = torch.argmax(logits, dim=-1)

        return sampled_tokens
