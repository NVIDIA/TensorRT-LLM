import copy
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import IntEnum, auto
from typing import TYPE_CHECKING, List, Optional, Type

import torch
from torch import nn

from tensorrt_llm.logger import logger

from ..._utils import get_sm_version
from ..attention_backend.trtllm import AttentionBackend, TrtllmAttention
from ..cute_dsl_kernels.argmax import argmax as cute_argmax
from ..flashinfer_utils import IS_FLASHINFER_AVAILABLE
from ..pyexecutor.resource_manager import BaseResourceManager

if TYPE_CHECKING:
    from ..pyexecutor.guided_decoder import CapturableGuidedDecoder

if IS_FLASHINFER_AVAILABLE:
    import flashinfer

# Environment variable name for forcing the number of accepted tokens in speculative decoding
FORCE_NUM_ACCEPTED_TOKENS_ENV_VAR = "TLLM_SPEC_DECODE_FORCE_NUM_ACCEPTED_TOKENS"


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
    DRAFT_TARGET = auto()
    USER_PROVIDED = auto()
    SAVE_HIDDEN_STATES = auto()
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
        return self.is_eagle3_one_model() or self.is_mtp_one_model()

    def is_eagle3_one_model(self):
        return self == SpeculativeDecodingMode.EAGLE3_ONE_MODEL

    def is_ngram(self):
        return self == SpeculativeDecodingMode.NGRAM

    def is_user_provided(self):
        return self == SpeculativeDecodingMode.USER_PROVIDED

    def is_none(self):
        return self == SpeculativeDecodingMode.NONE

    def is_draft_target(self):
        return self == SpeculativeDecodingMode.DRAFT_TARGET

    def is_save_hidden_states(self):
        return self == SpeculativeDecodingMode.SAVE_HIDDEN_STATES

    def without_logits(self):
        return self.is_mtp_one_model() or self.is_eagle3_one_model()

    def needs_kv_cache_rewind(self):
        return self.is_mtp_one_model() or self.is_eagle3_one_model(
        ) or self.is_ngram()

    def support_overlap_scheduler(self):
        return self.is_mtp_one_model() or self.is_eagle3_one_model(
        ) or self.has_draft_model()

    def support_guided_decoder(self):
        return self.is_none() or self.has_spec_drafter()

    def support_capturable_guided_decoder(self):
        return self.is_mtp_one_model() or self.is_eagle3_one_model()

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
        return self.is_eagle3_one_model()

    def has_spec_decoder(self):
        return self.is_mtp_one_model() or self.is_mtp_eagle() or self.is_eagle3(
        ) or self.is_eagle3_one_model()

    def has_spec_drafter(self):
        return self.is_eagle3(
        ) or self.is_draft_target() or self.is_ngram() or self.is_user_provided(
        ) or self.is_mtp_eagle() or self.is_save_hidden_states()

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
        is_mla: bool,
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
        xqa_supported = not is_mla or get_sm_version() < 120
        use_case_1 = self.use_one_engine() and xqa_supported
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

    # For non-greedy sampling on 1-model.
    allow_advanced_sampling: bool = False
    # Sampling parameters for non-greedy sampling (per-request)
    temperatures: Optional[torch.Tensor] = None
    top_ks: Optional[torch.Tensor] = None
    top_ps: Optional[torch.Tensor] = None

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
            num_tokens = 1 + self.max_draft_len if request.state == LlmRequestState.GENERATION_IN_PROGRESS else 1

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
            temperatures, dtype=torch.float32, pin_memory=True),
                                                    non_blocking=True)
        self.top_ks[:len(top_ks)].copy_(torch.tensor(top_ks,
                                                     dtype=torch.int32,
                                                     pin_memory=True),
                                        non_blocking=True)
        self.top_ps[:len(top_ps)].copy_(torch.tensor(top_ps,
                                                     dtype=torch.float32,
                                                     pin_memory=True),
                                        non_blocking=True)


class SpecWorkerBase(nn.Module, ABC):
    """
    Base class for speculative decoding workers.
    Provides common functionality for sampling and token handling.
    """

    def __init__(self):
        super().__init__()
        self.guided_decoder: Optional["CapturableGuidedDecoder"] = None
        self.force_num_accepted_tokens = get_force_num_accepted_tokens()
        self.use_flashinfer = IS_FLASHINFER_AVAILABLE and flashinfer.__version__ >= "0.6.0"
        self.seed = 0
        self.offset = 0

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

    def _apply_force_accepted_tokens(self, num_accepted_tokens, num_contexts):
        """
        Apply forced number of accepted tokens if environment variable is set.
        This is used for testing and debugging.

        Args:
            num_accepted_tokens: Tensor of shape [batch_size] with current accepted counts
            num_contexts: Number of context (prefill) requests

        Returns:
            Modified num_accepted_tokens tensor

        Note:
            For MTPWorker, self.max_draft_len equals num_nextn_predict_layers (mtp_num_modules).
            For Eagle3OneModelWorker, self.max_draft_len equals spec_config.max_draft_len.
        """
        if self.force_num_accepted_tokens != 0:
            # total tokens per iteration = accepted draft tokens + 1 target token
            force_total_tokens = min(self.force_num_accepted_tokens + 1,
                                     self.max_draft_len + 1)
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
            draft_tokens: [num_gens, max_draft_len] - Previously predicted draft tokens
            num_contexts: Number of context requests
            batch_size: Total number of requests
            spec_metadata: Speculative decoding metadata

        Returns:
            accepted_tokens: [batch_size, max_draft_len + 1] - Accepted tokens
            num_accepted_tokens: [batch_size] - Number of accepted tokens per request
        """
        num_gens = batch_size - num_contexts

        if logits.dim() == 1:
            logits = logits.unsqueeze(0)

        # Allocate return buffers
        accepted_tokens = torch.empty((batch_size, self.max_draft_len + 1),
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
            num_gens, self.max_draft_len + 1)
        accepted_tokens[num_contexts:, :] = gen_target_tokens

        # Compare draft tokens with target tokens using cumulative product
        # Counts consecutive matches from the start
        num_accepted_tokens[num_contexts:] += torch.cumprod(
            (draft_tokens == gen_target_tokens[:, :self.max_draft_len]).int(),
            dim=-1).sum(1)

        # Apply force override if set
        num_accepted_tokens = self._apply_force_accepted_tokens(
            num_accepted_tokens, num_contexts)

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
        # cute_argmax returns (M, 2) where col 0 = max value, col 1 = argmax index
        draft_tokens = cute_argmax(logits)[:, 1].long()

        # Apply d2t (offsets between draft and target model dictionaries)
        if d2t is not None:
            draft_tokens = d2t[draft_tokens] + draft_tokens

        return draft_tokens.type(torch.int32)

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
            next_draft_tokens: [batch_size, max_draft_len] - Predicted draft tokens
            batch_indices_cuda: Batch indices tensor
            batch_size: Number of requests
            num_accepted_tokens: [batch_size] - Number of accepted tokens per request

        Returns:
            next_new_tokens: [batch_size, max_draft_len + 1] - Input tokens for next iteration
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
        input_prompt_ids = input_ids[:num_ctx_tokens]
        input_ids_ctx = torch.empty_like(input_prompt_ids,
                                         dtype=torch.int32,
                                         device="cuda")
        input_ids_ctx[:-1].copy_(input_prompt_ids[1:])
        input_ids_ctx[
            gather_ids[:num_contexts]] = accepted_tokens[:num_contexts, 0]
        return input_ids_ctx

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
            from .one_model_sampler import sampling_batch_spec_dec_one_model

            num_gens = batch_size - num_contexts
            num_tokens = num_contexts + num_gens * (self.max_draft_len + 1)

            temperatures = spec_metadata.temperatures[:num_tokens]
            top_ks = spec_metadata.top_ks[:num_tokens]
            top_ps = spec_metadata.top_ps[:num_tokens]

            if self.use_flashinfer:
                self.seed += 1

            sampled_tokens = sampling_batch_spec_dec_one_model(
                logits,
                temperatures,
                top_ks,
                top_ps,
                use_flashinfer=self.use_flashinfer,
                seed=self.seed,
                offset=self.offset)
        else:
            # cute_argmax returns (M, 2) where col 0 = max value, col 1 = argmax index
            sampled_tokens = cute_argmax(logits)[:, 1].long()

        return sampled_tokens
