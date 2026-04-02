import copy
import os
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

if IS_FLASHINFER_AVAILABLE:
    import flashinfer

from .one_model_sampler import (compute_probs_from_logits,
                                rejection_sampling_one_model,
                                sampling_batch_spec_dec_one_model)

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
        block_ids_per_seq = draft_kv_cache_manager.get_block_ids_per_seq(
            m.request_ids)
        num_blocks = block_ids_per_seq.shape[1]
        m.host_indexer_k_cache_block_offsets[:len(
            block_ids_per_seq), :num_blocks].copy_(block_ids_per_seq)
        m.indexer_k_cache_block_offsets[:m.num_seqs].copy_(
            m.host_indexer_k_cache_block_offsets[:m.num_seqs],
            non_blocking=True)
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
    # Whether to use rejection sampling (requires allow_advanced_sampling=True)
    use_rejection_sampling: bool = False
    # Sampling parameters for non-greedy sampling (per-request)
    temperatures: Optional[torch.Tensor] = None
    top_ks: Optional[torch.Tensor] = None
    top_ps: Optional[torch.Tensor] = None
    # Vocab size of the model, used for draft_probs buffer allocation in prepare().
    vocab_size: int = 0
    # Draft probabilities for rejection sampling, allocated in prepare().
    # Shape: [max_num_requests * max_draft_len * vocab_size] stored as flat, reshaped on use.
    draft_probs: Optional[torch.Tensor] = None
    draft_probs_vocab_size: int = 0
    # Whether draft_probs contains valid data. The buffer is pre-allocated with
    # uninitialized memory in prepare(); this flag gates _can_use_rejection_sampling
    # so that rejection sampling only runs after _compute_and_store_draft_probs has
    # written real probabilities.
    draft_probs_valid: bool = False

    def __post_init__(self):
        pass

    def prepare(self):
        """Hook to be called before the forward step of the model."""
        # Reset each step so stale probs from a previous iteration are never used
        # if _compute_and_store_draft_probs is not called this step.
        self.draft_probs_valid = False

        if self.use_rejection_sampling and self.draft_probs is None and self.vocab_size > 0:
            buffer_size = self.max_num_requests * self.max_draft_len * self.vocab_size
            self.draft_probs = torch.empty(buffer_size,
                                           dtype=torch.float32,
                                           device='cuda')
            self.draft_probs_vocab_size = self.vocab_size

    def create_cuda_graph_metadata(self, max_batch_size: int):
        """
        Creates metadata for CUDA graph execution.
        """
        if self.is_cuda_graph:
            return self

        cuda_graph_metadata = copy.copy(self)
        cuda_graph_metadata.is_cuda_graph = True
        cuda_graph_metadata.max_num_requests = max_batch_size
        cuda_graph_metadata.__post_init__()
        return cuda_graph_metadata

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
            # For dynamic tree, each request can have up to max_total_draft_tokens
            # draft tokens (= topK * max_draft_len), which is larger than max_draft_len.
            # Use max_total_draft_tokens so the buffer is large enough for both modes.
            tokens_per_req = self.max_total_draft_tokens + 1
            self.temperatures = torch.ones(tokens_per_req *
                                           self.max_num_requests,
                                           dtype=torch.float32,
                                           device='cuda')
            self.top_ks = torch.zeros(tokens_per_req * self.max_num_requests,
                                      dtype=torch.int32,
                                      device='cuda')
            self.top_ps = torch.ones(tokens_per_req * self.max_num_requests,
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
        self.seed = None
        self.offset = None
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
        return {
            'logits': logits,
            'new_tokens': accepted_tokens,
            'new_tokens_lens': num_accepted_tokens,
            'next_draft_tokens': next_draft_tokens,
            'next_new_tokens': next_new_tokens
        }

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

        return {
            'logits': logits,
            'new_tokens': accepted_tokens,
            'new_tokens_lens': num_accepted_tokens,
            'next_draft_tokens': next_draft_tokens,
            'next_new_tokens': next_new_tokens
        }

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
        target_tokens = self._sample_tokens_for_batch(logits, spec_metadata,
                                                      num_contexts, batch_size)

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

    def _accept_draft_tokens(self, logits, draft_tokens, num_contexts,
                             batch_size, spec_metadata):
        """Unified draft token acceptance with automatic rejection sampling support.

        Subclasses should call this instead of manually branching between
        _sample_and_accept_draft_tokens_base and _sample_and_accept_draft_tokens_rejection.

        Args:
            logits: Target model logits
            draft_tokens: [num_gens, draft_len] - already reshaped by the caller
            num_contexts: Number of context (prefill) requests
            batch_size: Total batch size
            spec_metadata: Speculative decoding metadata
        """
        if self._can_use_rejection_sampling(spec_metadata):
            num_gens = batch_size - num_contexts
            draft_len = draft_tokens.shape[1]
            vocab_size = spec_metadata.draft_probs_vocab_size
            draft_probs = spec_metadata.draft_probs[:num_gens * draft_len *
                                                    vocab_size].reshape(
                                                        num_gens, draft_len,
                                                        vocab_size)
            return self._sample_and_accept_draft_tokens_rejection(
                logits,
                draft_tokens,
                draft_probs,
                num_contexts,
                batch_size,
                spec_metadata,
                d2t=getattr(self, '_d2t', None))
        return self._sample_and_accept_draft_tokens_base(
            logits, draft_tokens, num_contexts, batch_size, spec_metadata)

    def _can_use_rejection_sampling(self, spec_metadata: SpecMetadata) -> bool:
        """
        Whether rejection sampling can be used for the given spec_metadata.
        """
        return spec_metadata.use_rejection_sampling and spec_metadata.draft_probs_valid

    def _sample_and_accept_draft_tokens_rejection(
        self,
        logits: torch.Tensor,
        draft_tokens: torch.Tensor,
        draft_probs: torch.Tensor,
        num_contexts: int,
        batch_size: int,
        spec_metadata,
        d2t: Optional[torch.Tensor] = None,
    ):
        """
        Rejection sampling implementation for accepting draft tokens.
        Uses flashinfer's chain_speculative_sampling for CUDA graph compatibility.

        This method performs proper rejection sampling where tokens are accepted
        with probability min(1, p_target / p_draft), providing lossless acceleration
        that matches the target model's distribution exactly.

        Supports mixed prefill+decode batches: context (prefill) requests are skipped
        and only generation requests (num_gens = batch_size - num_contexts) are processed.

        Args:
            logits: [num_tokens, vocab_size] - Target model logits (all tokens including context)
            draft_tokens: [num_gens, max_draft_len] - Previously predicted draft tokens
            draft_probs: [num_gens, max_draft_len, vocab_size] - Draft model probabilities
            num_contexts: Number of context (prefill) requests
            batch_size: Total batch size (num_contexts + num_gens)
            spec_metadata: Speculative decoding metadata containing sampling parameters

        Returns:
            accepted_tokens: [batch_size, max_draft_len + 1] - Accepted tokens
            num_accepted_tokens: [batch_size] - Number of accepted tokens per request
        """
        device = logits.device
        vocab_size = logits.shape[-1]
        num_gens = batch_size - num_contexts

        if logits.dim() == 1:
            logits = logits.unsqueeze(0)

        draft_vocab_size = draft_probs.shape[-1]

        # Extract only the generation-request logits (skip context tokens)
        num_ctx_tokens = logits.shape[0] - num_gens * (self.max_draft_len + 1)
        gen_logits = logits[num_ctx_tokens:]

        # Get sampling parameters for generation requests only
        gen_slice = slice(num_contexts * (self.max_draft_len + 1),
                          (num_contexts + num_gens) * (self.max_draft_len + 1))
        temperatures = spec_metadata.temperatures[gen_slice]
        top_ks = spec_metadata.top_ks[
            gen_slice] if spec_metadata.top_ks is not None else None
        top_ps = spec_metadata.top_ps[
            gen_slice] if spec_metadata.top_ps is not None else None

        # Compute target probs for generation requests
        target_probs_flat = compute_probs_from_logits(gen_logits.clone(),
                                                      temperatures, top_ks,
                                                      top_ps)
        target_probs = target_probs_flat.reshape(num_gens,
                                                 self.max_draft_len + 1,
                                                 vocab_size)

        # Prepare draft probs and tokens for rejection sampling
        # Handle vocab size mismatch: draft model may have different vocab than target
        if draft_vocab_size != vocab_size:
            assert draft_vocab_size < vocab_size
            full_draft_probs = torch.zeros(
                (num_gens, self.max_draft_len, vocab_size),
                dtype=torch.float32,
                device=device)
            if d2t is not None:
                # Use d2t offset mapping: draft token i maps to target token i + d2t[i]
                src_idx = torch.arange(draft_vocab_size, device=device)
                target_idx = src_idx + d2t[:draft_vocab_size].long()
                full_draft_probs.scatter_(
                    2, target_idx.expand(num_gens, self.max_draft_len, -1),
                    draft_probs)
            else:
                # Fallback: assume draft vocab is a contiguous prefix of target vocab
                full_draft_probs[:, :, :draft_vocab_size] = draft_probs
        else:
            full_draft_probs = draft_probs

        full_draft_tokens = draft_tokens.to(torch.int32).contiguous()

        # Lazily initialize seed/offset tensors on correct device
        if self.seed is None:
            self.seed = torch.tensor([0], dtype=torch.int64, device=device)
            self.offset = torch.tensor([0], dtype=torch.int64, device=device)

        # Increment seed for CUDA graph compatibility (use in-place ops)
        self.seed.add_(1).remainder_(2**31)

        # Perform rejection sampling
        gen_accepted_tokens, gen_num_accepted = rejection_sampling_one_model(
            draft_probs=full_draft_probs,
            draft_token_ids=full_draft_tokens,
            target_probs=target_probs,
            deterministic=True,
            seed=self.seed,
            offset=self.offset,
        )

        # Apply force override if set
        gen_num_accepted = self._apply_force_accepted_tokens(
            gen_num_accepted, 0, self.max_draft_len)

        # Build full-batch output tensors (context requests get zero-filled slots)
        accepted_tokens = torch.zeros(batch_size,
                                      self.max_draft_len + 1,
                                      dtype=gen_accepted_tokens.dtype,
                                      device=device)
        num_accepted_tokens = torch.zeros(batch_size,
                                          dtype=gen_num_accepted.dtype,
                                          device=device)
        accepted_tokens[num_contexts:batch_size] = gen_accepted_tokens
        num_accepted_tokens[num_contexts:batch_size] = gen_num_accepted

        # The draft probs have been consumed; mark them invalid so that a future
        # step that skips _compute_and_store_draft_probs cannot accidentally reuse
        # this iteration's stale data.
        spec_metadata.draft_probs_valid = False

        return accepted_tokens, num_accepted_tokens

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

    def _compute_and_store_draft_probs(
        self,
        draft_logits_list: List[torch.Tensor],
        spec_metadata: SpecMetadata,
        batch_size: int,
        num_contexts: int,
    ):
        """
        Compute draft probabilities from draft logits and store in spec_metadata
        for use in the next iteration's rejection sampling.

        Args:
            draft_logits_list: List of [batch_size, vocab_size] tensors, one per draft layer
            spec_metadata: Speculative decoding metadata
            batch_size: Total number of requests (contexts + gens)
            num_contexts: Number of context (prefill) requests
        """
        draft_tokens_per_request = len(draft_logits_list)
        vocab_size = draft_logits_list[0].shape[-1]
        device = draft_logits_list[0].device

        # Stack draft logits: [batch_size, draft_tokens_per_request, vocab_size] (contiguous)
        # then reshape to [batch_size * draft_tokens_per_request, vocab_size].
        draft_logits_flat = torch.stack(draft_logits_list,
                                        dim=1).reshape(-1, vocab_size)

        num_draft_tokens = batch_size * draft_tokens_per_request
        if spec_metadata.temperatures is not None:
            # The temperatures tensor layout (from populate_sampling_params_for_one_model)
            # is variable-stride per request type:
            #   - context request i:  1 slot  → index i
            #   - gen request j:      (1 + runtime_draft_len) slots
            #                         → start index num_contexts + j*(1+runtime_draft_len)
            # All slots within a request hold the same temperature value, so we extract
            # exactly one entry per request and then broadcast across draft steps.
            num_gens = batch_size - num_contexts
            stride = spec_metadata.runtime_draft_len + 1
            ctx_temps = spec_metadata.temperatures[:num_contexts]
            gen_temps = spec_metadata.temperatures[
                num_contexts::stride][:num_gens]
            per_request_temps = torch.cat([ctx_temps, gen_temps])
            draft_temps = per_request_temps.repeat_interleave(
                draft_tokens_per_request)

            if spec_metadata.top_ks is not None:
                ctx_top_ks = spec_metadata.top_ks[:num_contexts]
                gen_top_ks = spec_metadata.top_ks[
                    num_contexts::stride][:num_gens]
                per_request_top_ks = torch.cat([ctx_top_ks, gen_top_ks])
                draft_top_ks = per_request_top_ks.repeat_interleave(
                    draft_tokens_per_request)
            else:
                draft_top_ks = None

            if spec_metadata.top_ps is not None:
                ctx_top_ps = spec_metadata.top_ps[:num_contexts]
                gen_top_ps = spec_metadata.top_ps[
                    num_contexts::stride][:num_gens]
                per_request_top_ps = torch.cat([ctx_top_ps, gen_top_ps])
                draft_top_ps = per_request_top_ps.repeat_interleave(
                    draft_tokens_per_request)
            else:
                draft_top_ps = None
        else:
            # Default temperature of 1.0 if not set
            draft_temps = torch.ones(num_draft_tokens, device=device)
            draft_top_ks = None
            draft_top_ps = None

        draft_probs_flat = compute_probs_from_logits(draft_logits_flat,
                                                     draft_temps, draft_top_ks,
                                                     draft_top_ps)

        num_elements = batch_size * draft_tokens_per_request * vocab_size
        spec_metadata.draft_probs[:num_elements].copy_(
            draft_probs_flat.flatten())
        spec_metadata.draft_probs_valid = True

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
            num_gens = batch_size - num_contexts
            num_tokens = num_contexts + num_gens * (
                spec_metadata.runtime_draft_len + 1)

            temperatures = spec_metadata.temperatures[:num_tokens]
            top_ks = spec_metadata.top_ks[:num_tokens]
            top_ps = spec_metadata.top_ps[:num_tokens]

            if self.use_flashinfer:
                # Lazily initialize seed/offset tensors on correct device
                if self.seed is None:
                    self.seed = torch.tensor([0],
                                             dtype=torch.int64,
                                             device=logits.device)
                    self.offset = torch.tensor([0],
                                               dtype=torch.int64,
                                               device=logits.device)
                top_ks = top_ks.clamp(min=1, max=logits.shape[-1] - 1)
                # Use in-place operations for CUDA graph compatibility
                self.seed.add_(1).remainder_(2**31)

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
