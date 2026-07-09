# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import os
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import IntEnum, auto
from typing import TYPE_CHECKING, List, Optional, Type

import torch
from packaging.version import Version
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

from ..pyexecutor.sampler.sampling_utils import (
    compute_probs_from_logits, greedy, sampling_batch_spec_dec_one_model,
    sampling_batch_spec_dec_one_model_for_rejection)


def rejection_sampling_one_model(
    draft_probs: torch.Tensor,
    draft_token_ids: torch.Tensor,
    target_probs: torch.Tensor,
    deterministic: bool = True,
    seed: Optional[int] = None,
    offset: Optional[int] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    # chain_speculative_sampling requires flashinfer>=0.6.4. This entry point can
    # be reached independently of SpecWorkerBase.__init__ (e.g. via
    # _can_use_rejection_sampling), so re-check here to fail with a clear message
    # instead of a cryptic flashinfer error.
    if not IS_FLASHINFER_AVAILABLE or Version(
            flashinfer.__version__) < Version("0.6.4"):
        raise RuntimeError(
            "Rejection sampling for one-model speculative decoding requires flashinfer>=0.6.4"
        )
    batch_size = draft_token_ids.shape[0]
    device = draft_token_ids.device
    output_accepted_token_num = torch.zeros(batch_size,
                                            dtype=torch.int32,
                                            device=device)
    output_emitted_draft_token_num = torch.zeros(batch_size,
                                                 dtype=torch.int32,
                                                 device=device)
    accepted_tokens, _, output_emitted_draft_token_num = flashinfer.sampling.chain_speculative_sampling(
        draft_probs,
        draft_token_ids,
        target_probs,
        maybe_output_accepted_token_num=output_accepted_token_num,
        maybe_output_emitted_draft_token_num=output_emitted_draft_token_num,
        deterministic=deterministic,
        generator=None,
        seed=seed,
        offset=offset,
    )
    return accepted_tokens, output_emitted_draft_token_num + 1


# Environment variable name for forcing the number of accepted tokens in speculative decoding
FORCE_NUM_ACCEPTED_TOKENS_ENV_VAR = "TLLM_SPEC_DECODE_FORCE_NUM_ACCEPTED_TOKENS"

# RNG pool configuration for the fractional (probabilistic) component of the
# synthetic acceptance rate. Pool size MUST be a power of two so we can use
# a bitmask (`& (pool_size - 1)`) for wrap-around — this stays cheap and
# keeps tensor shapes static for CUDA graph capture. The fixed seed is what
# guarantees identical random draws on every TP rank (so all ranks accept the
# same number of tokens per iteration and downstream collectives stay in
# lock-step). The two stride primes mix the per-call counter with the
# per-slot index so consecutive calls / consecutive slots map to decorrelated
# pool entries.
_FORCE_ACCEPT_RNG_POOL_SIZE = 1 << 16  # 65536 entries (256 KiB float32)
_FORCE_ACCEPT_RNG_SEED = 0xACCE9D
_FORCE_ACCEPT_RNG_COUNTER_STRIDE = 6007
_FORCE_ACCEPT_RNG_SLOT_STRIDE = 1009


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
    if attn_metadata.enable_flash_mla:
        attn_metadata.prepare_flash_mla()

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
    if attn_metadata.enable_flash_mla:
        attn_metadata.prepare_flash_mla()
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
    Read and parse the TLLM_SPEC_DECODE_FORCE_NUM_ACCEPTED_TOKENS environment
    variable as an integer.

    Used by speculative decoding paths that operate on Python lists/slices and
    therefore require an integer count (e.g. the two-model path in
    ``TorchSampler``). For the one-model path, see
    :func:`get_force_num_accepted_tokens_float`, which supports fractional
    synthetic acceptance rates.

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


def get_force_num_accepted_tokens_float() -> float:
    """
    Read and parse the TLLM_SPEC_DECODE_FORCE_NUM_ACCEPTED_TOKENS environment
    variable as a (possibly fractional) float.

    Used by the one-model speculative decoding path to synthesize non-integer
    acceptance rates: the integer part is the number of draft tokens accepted
    on every iteration, and the fractional part is the probability of
    accepting one additional draft token. For example, "2.6" means always
    accept 2 draft tokens and accept one more with probability 0.6.

    Returns:
        float: The forced (possibly fractional) number of accepted draft
        tokens, or 0.0 if not set or invalid.
    """
    env_value = os.environ.get(FORCE_NUM_ACCEPTED_TOKENS_ENV_VAR, "0")
    try:
        return float(env_value)
    except ValueError:
        logger.warning(
            f"{FORCE_NUM_ACCEPTED_TOKENS_ENV_VAR} must be a valid number "
            f"(int or float), got '{env_value}'. Using default value 0.0.")
        return 0.0


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
    DFLASH = auto()
    NONE = auto()
    AUTO = auto()

    def is_mtp_one_model(self):
        # Union: covers vanilla MTP and MTP_EAGLE_ONE_MODEL. Use is_mtp_vanilla()
        # when only the vanilla MTP variant should match.
        return (self == SpeculativeDecodingMode.MTP
                or self == SpeculativeDecodingMode.MTP_EAGLE_ONE_MODEL)

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

    def is_dflash(self):
        return self == SpeculativeDecodingMode.DFLASH

    def is_parallel_draft(self):
        return self.is_pard() or self.is_dflash()

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
        return self.is_parallel_draft() or self.is_draft_target_one_model()

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
        return self.is_mtp_one_model() or self.is_eagle3_one_model(
        ) or self.is_mtp_eagle_one_model() or self.is_pard() or self.is_dflash(
        ) or self.is_draft_target_one_model() or self.is_sa()

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
    # Total runtime tokens per generation request for the current iteration,
    # Normally, it equals 1 + runtime_draft_len. But for PARD, it equals 2 * runtime_draft_len.
    runtime_tokens_per_gen_step: int = 1

    # Auto-detected per step from populated sampling params:
    # True if every request is greedy (no temp/top_k/top_p) and we can take
    # the argmax fast-path. False if any request needs sampling.
    # Used as part of the CUDA graph key so we capture two variants
    # (greedy fast-path vs advanced sampling) and dispatch at replay.
    # Defaults to True so non-one-engine paths (where populate is a no-op)
    # never accidentally select the advanced graph variant.
    is_all_greedy_sample: bool = True
    # Whether to use rejection sampling for one-model speculative decoding.
    use_rejection_sampling: bool = False
    # Sampling parameters for non-greedy sampling (per-request)
    temperatures: Optional[torch.Tensor] = None
    top_ks: Optional[torch.Tensor] = None
    top_ps: Optional[torch.Tensor] = None
    # Whether top-k/top-p/temperature are globally disabled for the current batch.
    skip_temperature: bool = False
    skip_top_k: bool = False
    skip_top_p: bool = False
    has_greedy_requests: bool = False
    # Pre-computed top_k_max scalar (CPU-side) to avoid CUDA-graph-incompatible
    # dynamic boolean tensor indexing inside verify_dynamic_tree_rejection_from_logits_out.
    top_k_max: int = 0
    # Sampling parameters indexed per request.
    request_temperatures: Optional[torch.Tensor] = None
    request_top_ks: Optional[torch.Tensor] = None
    request_top_ps: Optional[torch.Tensor] = None
    # Whether to use sampling parameters when sampling draft tokens.
    use_sampling_params_for_draft_tokens: bool = False
    # Vocab size used for draft_probs buffer allocation.
    vocab_size: int = 0
    # Draft probabilities buffer for rejection sampling, indexed by py_seq_slot
    # so per-request data is stable across iterations regardless of batch
    # composition shifts (chunking ctx, gen completion, new ctx joining).
    # Shape: [max_num_requests, max_draft_len, vocab_size].
    draft_probs: Optional[torch.Tensor] = None
    draft_probs_vocab_size: int = 0
    # Whether draft_probs contains valid data.
    draft_probs_valid: bool = False
    # Last dimension size of the draft logits/probs stored in draft_probs.
    draft_probs_last_dim: int = 0
    # Per-request slot ids (py_seq_slot) for the current batch, in batch order.
    # Used to scatter draft probs by slot at write time and gather them by slot
    # at the next iter's verify. Shape: [max_num_requests], dtype=long.
    batch_slot_ids: Optional[torch.Tensor] = None
    # Draft-to-target vocab offset tensor.
    d2t: Optional[torch.Tensor] = None
    # Pre-allocated scratch for draft probs expanded to the target vocab size.
    # Filled with zeros once at prepare(); each rejection iter only overwrites
    # the positions selected by d2t (or [:draft_vocab] when there is no d2t),
    # so the zeros outside those positions persist across iterations and we
    # avoid a per-iter 64 MB zero-fill on the (max_num_requests, max_draft_len,
    # vocab_size) tensor. Shape: [max_num_requests, max_draft_len, vocab_size].
    full_draft_probs: Optional[torch.Tensor] = None
    # Cached d2t-projected target vocab indices, computed once on first use
    # (d2t is a model-static tensor). Replaces the per-iter
    # arange + (source + d2t) % vocab_size kernel sequence inside the d2t
    # padding step. Shape: [draft_vocab_size], dtype long.
    d2t_target_indices: Optional[torch.Tensor] = None

    def __post_init__(self):
        pass

    def prepare(self):
        """
        Hook to be called before the forward step of the model.
        """
        if (self.use_rejection_sampling and self.draft_probs is None
                and self.vocab_size > 0):
            # 3D [slot, draft_step, vocab] so we can scatter/gather by slot id
            # and avoid the brittle "batch position == buffer position" mapping.
            self.draft_probs = torch.empty(
                (self.max_num_requests, self.max_draft_len, self.vocab_size),
                dtype=torch.float32,
                device='cuda')
            self.draft_probs_vocab_size = self.vocab_size
        if (self.use_rejection_sampling and self.batch_slot_ids is None
                and self.max_num_requests > 0):
            self.batch_slot_ids = torch.empty((self.max_num_requests, ),
                                              dtype=torch.long,
                                              device='cuda')
        if (self.use_rejection_sampling and self.full_draft_probs is None
                and self.vocab_size > 0):
            # Zero-fill once. Subsequent iters only overwrite the d2t-mapped
            # positions (constant across iters since d2t is model-static), so
            # untouched positions stay 0 forever — saves the per-iter 64 MB
            # zero-fill in _sample_and_accept_draft_tokens_rejection.
            self.full_draft_probs = torch.zeros(
                (self.max_num_requests, self.max_draft_len, self.vocab_size),
                dtype=torch.float32,
                device='cuda')

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

    def _scan_one_model_sampling(
        self, requests: list["LlmRequest"]
    ) -> tuple[list[tuple[float, int, float, int]], list[int]]:
        """Single source of truth for one-engine sampling-param detection.

        Scans the batch's sampling configs and sets skip_*/has_greedy_requests/
        is_all_greedy_sample (honoring the warmup capture override). Returns
        ``(per_request_normalized, per_request_slot_ids)`` for buffer
        population. Does NOT allocate or fill GPU buffers, so it is safe to call
        before the CUDA graph key is built.
        """
        from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequestState
        from tensorrt_llm.sampling_params import SamplingParams

        # Need to use a very small value for temperature when disabled to avoid division by 0
        DISABLE_TEMP_VAL = 1e-5
        # Very large values disable topk.
        DISABLE_TOPK_VAL = torch.iinfo(torch.int32).max
        DISABLE_TOPP_VAL = 1.0

        def _first_or_none(values):
            """Return the first sampling parameter value when present."""
            return values[0] if values is not None and len(values) > 0 else None

        def _normalize_request_sampling_params(
            *,
            temperature: Optional[float],
            top_k: Optional[int],
            top_p: Optional[float],
        ) -> tuple[float, int, float, bool, bool, bool, bool]:
            """Convert request sampling params into normalized per-request scalars."""
            is_greedy = SamplingParams.params_imply_greedy_decoding(
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                use_beam_search=False)

            use_temperature = (not is_greedy
                               and temperature not in (None, 0, 1))
            use_top_k = not is_greedy and top_k is not None and top_k > 0
            use_top_p = not is_greedy and top_p is not None and top_p < 1.0

            normalized_temperature = (DISABLE_TEMP_VAL
                                      if is_greedy or temperature is None
                                      or temperature == 0 else temperature)
            normalized_top_k = DISABLE_TOPK_VAL if not use_top_k else top_k
            normalized_top_p = (DISABLE_TOPP_VAL
                                if is_greedy or top_p is None else top_p)

            return (
                normalized_temperature,
                normalized_top_k,
                normalized_top_p,
                use_temperature,
                use_top_k,
                use_top_p,
                is_greedy,
            )

        # Phase 1: collect per-request flags and normalized values.
        per_request_normalized: list[tuple[float, int, float, int]] = []
        temperature_enabled = False
        top_k_enabled = False
        top_p_enabled = False
        has_greedy_requests = False
        per_request_slot_ids: list[int] = []

        for request in requests:
            sampling_config = request.sampling_config
            temp_val = _first_or_none(sampling_config.temperature)
            tk_val = _first_or_none(sampling_config.top_k)
            tp_val = _first_or_none(sampling_config.top_p)

            # Context requests have no draft tokens yet.
            num_tokens = 1 + self.runtime_draft_len if request.state == LlmRequestState.GENERATION_IN_PROGRESS else 1

            (
                temp_val,
                tk_val,
                tp_val,
                use_temperature,
                use_top_k,
                use_top_p,
                is_greedy,
            ) = _normalize_request_sampling_params(
                temperature=temp_val,
                top_k=tk_val,
                top_p=tp_val,
            )

            temperature_enabled |= use_temperature
            top_k_enabled |= use_top_k
            top_p_enabled |= use_top_p
            has_greedy_requests |= is_greedy

            per_request_normalized.append(
                (temp_val, tk_val, tp_val, num_tokens))
            # py_seq_slot is a stable per-request id used to scatter / gather
            # draft probs across iterations. Dummies / unallocated slots fall
            # back to 0 (any valid index is fine — the data at that slot will
            # be overwritten on the next real iteration before being read).
            per_request_slot_ids.append(
                request.py_seq_slot if request.py_seq_slot is not None else 0)

        self.skip_temperature = not temperature_enabled
        self.skip_top_k = not top_k_enabled
        self.skip_top_p = not top_p_enabled
        self.has_greedy_requests = has_greedy_requests
        # Used in the CUDA graph key to pick the argmax / advanced variant.
        self.is_all_greedy_sample = (self.skip_temperature and self.skip_top_k
                                     and self.skip_top_p)

        # Warmup-time override (set via runtime attribute by the model engine):
        # force the advanced-sampling code path so the CUDA graph for the
        # (is_all_greedy_sample=False) key gets captured. Dummy warmup requests
        # carry no sampling params, so the natural detection above always
        # returns True; this branch substitutes synthetic non-greedy scalars
        # into the per-request data and lets Phase 2 run normally to populate
        # the GPU buffers used by the captured kernels.
        if getattr(self, '_force_non_greedy_for_capture', False):
            self.skip_temperature = False
            self.skip_top_k = False
            self.skip_top_p = False
            self.is_all_greedy_sample = False
            per_request_normalized = [
                (0.7, 50, 0.9, num_tokens)
                for (_, _, _, num_tokens) in per_request_normalized
            ]

        return per_request_normalized, per_request_slot_ids

    def update_is_all_greedy_sample(self, requests: list["LlmRequest"]) -> None:
        """Refresh ``is_all_greedy_sample`` for the *current* batch.

        Must be called BEFORE the CUDA graph key is built (the key includes
        ``is_all_greedy_sample`` to choose the argmax vs advanced-sampling graph
        variant). ``populate_sampling_params_for_one_model`` runs later, inside
        ``_prepare_inputs``, and re-derives the same flag while filling the GPU
        sampling buffers. Computing the flag here first keeps the selected graph
        consistent with the buffers ``populate`` fills; otherwise the key would
        use the previous iteration's stale value and could replay the advanced
        graph against unpopulated (greedy) buffers, which can hang/corrupt the
        run (notably for MTP with num_nextn>=2).
        """
        if not self.spec_dec_mode.use_one_engine():
            return
        self._scan_one_model_sampling(requests)

    def populate_sampling_params_for_one_model(
            self, requests: list["LlmRequest"]) -> None:
        """
        Set up topp/topk/temperatures for 1-model sampler.

        Scans sampling configs to set skip_*/is_all_greedy_sample flags. When
        any request needs sampling, also builds per-token/per-request lists
        and copies them to GPU buffers; all-greedy batches skip this entirely.
        """
        if not self.spec_dec_mode.use_one_engine():
            return

        if self.temperatures is None:
            # Ensures determinism across ranks.
            torch.manual_seed(0)

        per_request_normalized, per_request_slot_ids = (
            self._scan_one_model_sampling(requests))

        tokens_per_request = (self.max_total_draft_tokens + 1 if
                              self.is_spec_dec_tree else self.max_draft_len + 1)
        # Warmup batches may exceed max_num_requests * tokens_per_request (e.g.
        # when CUDA-graph warmup passes use max_batch_size > max_num_requests).
        actual_flat_size = sum(
            num_tokens for _, _, _, num_tokens in per_request_normalized)
        required_flat_size = max(tokens_per_request * self.max_num_requests,
                                 actual_flat_size)

        if self.temperatures is None or self.temperatures.numel(
        ) < required_flat_size:
            # Allocate once; the captured graph reads from these stable addresses.
            self.temperatures = torch.ones(required_flat_size,
                                           dtype=torch.float32,
                                           device='cuda')
            self.top_ks = torch.zeros(required_flat_size,
                                      dtype=torch.int32,
                                      device='cuda')
            self.top_ps = torch.ones(required_flat_size,
                                     dtype=torch.float32,
                                     device='cuda')
            self.request_temperatures = torch.ones(self.max_num_requests,
                                                   dtype=torch.float32,
                                                   device='cuda')
            self.request_top_ks = torch.zeros(self.max_num_requests,
                                              dtype=torch.int32,
                                              device='cuda')
            self.request_top_ps = torch.ones(self.max_num_requests,
                                             dtype=torch.float32,
                                             device='cuda')

        # Always-populate the per-request slot id table when rejection sampling
        # is configured: it's tiny (max_num_requests longs) and needed at
        # draft-sampler time to scatter draft probs by slot.
        if self.use_rejection_sampling and self.batch_slot_ids is not None:
            self.batch_slot_ids[:len(per_request_slot_ids)].copy_(
                torch.tensor(per_request_slot_ids,
                             dtype=torch.long,
                             pin_memory=prefer_pinned()),
                non_blocking=True,
            )

        # All-greedy: sampler takes the argmax branch (and rejection sampling
        # is also bypassed for all-greedy), so the per-token buffers are never
        # read. Skip the heavier H->D copies.
        if self.is_all_greedy_sample:
            return

        # Phase 2: build per-token / per-request lists and copy to GPU.
        temperatures: list[float] = []
        top_ks: list[int] = []
        top_ps: list[float] = []
        request_temperatures: list[float] = []
        request_top_ks: list[int] = []
        request_top_ps: list[float] = []
        for temp_val, tk_val, tp_val, num_tokens in per_request_normalized:
            request_temperatures.append(temp_val)
            request_top_ks.append(tk_val)
            request_top_ps.append(tp_val)
            temperatures.extend(temp_val for _ in range(num_tokens))
            top_ks.extend(tk_val for _ in range(num_tokens))
            top_ps.extend(tp_val for _ in range(num_tokens))

        self.temperatures[:len(temperatures)].copy_(torch.tensor(
            temperatures, dtype=torch.float32, pin_memory=prefer_pinned()),
                                                    non_blocking=True)
        self.top_ks[:len(top_ks)].copy_(torch.tensor(
            top_ks, dtype=torch.int32, pin_memory=prefer_pinned()),
                                        non_blocking=True)
        self.top_ps[:len(top_ps)].copy_(torch.tensor(
            top_ps, dtype=torch.float32, pin_memory=prefer_pinned()),
                                        non_blocking=True)
        self.request_temperatures[:len(request_temperatures)].copy_(
            torch.tensor(request_temperatures,
                         dtype=torch.float32,
                         pin_memory=prefer_pinned()),
            non_blocking=True)
        self.request_top_ks[:len(request_top_ks)].copy_(
            torch.tensor(request_top_ks,
                         dtype=torch.int32,
                         pin_memory=prefer_pinned()),
            non_blocking=True,
        )
        self.request_top_ps[:len(request_top_ps)].copy_(
            torch.tensor(request_top_ps,
                         dtype=torch.float32,
                         pin_memory=prefer_pinned()),
            non_blocking=True,
        )

        # Pre-compute top_k_max on the CPU so CUDA-graph capture does not
        # encounter boolean-tensor indexing (dynamic size) or .item() calls.
        # DISABLE_TOPK_VAL (INT32_MAX) is the sentinel for "top-k disabled".
        _disable_topk = torch.iinfo(torch.int32).max
        self.top_k_max = max(
            (tk for tk in request_top_ks if 0 < tk < _disable_topk), default=0)


class SpecWorkerBase(nn.Module, ABC):
    """
    Base class for speculative decoding workers.
    Provides common functionality for sampling and token handling.
    """

    def __init__(self, use_separate_draft_kv_cache: bool = False):
        super().__init__()
        self.guided_decoder: Optional["CapturableGuidedDecoder"] = None
        self.force_num_accepted_tokens: float = get_force_num_accepted_tokens_float(
        )
        # One-model speculative sampling goes through flashinfer unconditionally
        # (sampling_batch_spec_dec_one_model), so flashinfer>=0.6.4 is a hard
        # dependency here. Fail at construction with a clear error instead of
        # crashing mid-inference on the first non-greedy sampling step.
        if not IS_FLASHINFER_AVAILABLE or Version(
                flashinfer.__version__) < Version("0.6.4"):
            raise ImportError(
                "Speculative decoding requires flashinfer>=0.6.4, please install "
                "the version pinned in requirements.txt.")
        self.seed: Optional[torch.Tensor] = None
        self.offset: Optional[torch.Tensor] = None
        self.use_separate_draft_kv_cache = use_separate_draft_kv_cache
        # Lazily-initialized state for the fractional synthetic acceptance
        # rate. The pool is a fixed-seed, rank-independent table of uniform
        # [0, 1) values; the counter is a device-side int64 advanced in-place
        # inside captured CUDA graphs (mirroring the existing flashinfer
        # seed/offset pattern in `_sample_tokens_for_batch`).
        self._force_accept_rng_pool: Optional[torch.Tensor] = None
        self._force_accept_rng_counter: Optional[torch.Tensor] = None

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
        resource_manager=None,
    ):
        """Skip spec dec for non-last rank (PP). Returns placeholder outputs.

        ``resource_manager`` is accepted but unused; it appears in the
        ``forward()`` signature of one-model workers (Eagle3 / MTP-Eagle) and
        the caller in ``modeling_speculative.py`` forwards it unconditionally,
        so the skip path must accept it as well.
        """
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

    def _ensure_force_accept_rng_state(self, device: torch.device) -> None:
        """
        Lazily build the deterministic RNG state used by
        :meth:`_apply_force_accepted_tokens` for fractional synthetic
        acceptance rates.

        The pool is filled from a CPU generator with a fixed seed so that
        every tensor-parallel rank produces the bit-for-bit identical pool
        (TP ranks must agree on the per-iteration accepted-token count, or
        downstream collectives expecting identical shapes will hang).

        First-call allocation must happen during eager warmup — never inside
        a captured CUDA graph. The CUDA-graph runner already runs warmup
        forwards before capture, which satisfies this in practice.
        """
        if self._force_accept_rng_pool is not None:
            return
        cpu_gen = torch.Generator(device="cpu")
        cpu_gen.manual_seed(_FORCE_ACCEPT_RNG_SEED)
        pool_cpu = torch.rand(_FORCE_ACCEPT_RNG_POOL_SIZE,
                              dtype=torch.float32,
                              generator=cpu_gen)
        self._force_accept_rng_pool = pool_cpu.to(device=device)
        self._force_accept_rng_counter = torch.zeros(1,
                                                     dtype=torch.int64,
                                                     device=device)

    def _apply_force_accepted_tokens(self,
                                     num_accepted_tokens,
                                     num_contexts,
                                     runtime_draft_len: int,
                                     spec_metadata=None):
        """
        Apply a forced (synthetic) number of accepted draft tokens if the
        ``TLLM_SPEC_DECODE_FORCE_NUM_ACCEPTED_TOKENS`` environment variable is
        set. This is used for testing and debugging speculative decoding.

        The forced value supports fractional synthetic acceptance rates: the
        integer part is the number of draft tokens accepted on every
        generation iteration, and the fractional part is the probability of
        accepting one additional draft token on that iteration. For example,
        a value of ``2.6`` means: always accept 2 draft tokens, and accept
        one more with probability 0.6 (per generation request).

        The implementation is CUDA-graph-compatible AND tensor-parallel
        deterministic. Randomness is sourced from a fixed-seed lookup pool
        plus a device-side counter that is advanced in place each call —
        the same pattern as the flashinfer seed/offset state used by
        :meth:`_sample_tokens_for_batch`. Because every rank seeds the pool
        identically and increments the counter on the same captured ops,
        every rank draws the same uniform values and therefore agrees on the
        accepted-token count for every request in every iteration.

        Args:
            num_accepted_tokens: Tensor of shape [batch_size] with current
                accepted counts (target token + accepted draft tokens).
            num_contexts: Number of context (prefill) requests in the batch.
            runtime_draft_len: The draft length for the current iteration.
            spec_metadata: Optional SpecMetadata. When provided, used to
                detect eager CUDA-graph warmup so the override is skipped
                there — warmup batches use dummy requests whose KV cache and
                draft buffers are not populated for an inflated accepted
                count, which would drive downstream MTP ops out-of-bounds.

        Returns:
            Modified num_accepted_tokens tensor.
        """
        if self.force_num_accepted_tokens == 0.0:
            return num_accepted_tokens

        if spec_metadata is not None:
            is_warmup = (spec_metadata.is_cuda_graph
                         and not torch.cuda.is_current_stream_capturing())
            if is_warmup:
                return num_accepted_tokens

        # Decompose into a deterministic integer part (always accepted) and a
        # probabilistic fractional part. ``int(...)`` truncates toward zero,
        # which matches floor for the supported non-negative range.
        int_part = int(self.force_num_accepted_tokens)
        frac_part = self.force_num_accepted_tokens - int_part

        # ``num_accepted_tokens`` counts the target token + accepted draft
        # tokens, so the maximum reachable value is ``runtime_draft_len + 1``.
        max_total = runtime_draft_len + 1
        base_total = min(int_part + 1, max_total)

        if frac_part > 0.0 and base_total < max_total:
            self._ensure_force_accept_rng_state(num_accepted_tokens.device)

            # ``num_gens`` is fixed at CUDA-graph capture time (graphs are
            # captured for a specific batch shape with ``num_contexts``
            # typically 0), so all of the ops below have static shapes.
            num_gens = num_accepted_tokens.shape[0] - num_contexts

            # In-place counter bump is captured by the graph and replayed on
            # every iteration, so each replay yields fresh draws from the
            # pool. All TP ranks bump in lock-step → identical indices.
            self._force_accept_rng_counter += 1

            slot_ids = torch.arange(num_gens,
                                    device=num_accepted_tokens.device,
                                    dtype=torch.int64)
            # Hash (counter, slot) → pool index. ``& (pool_size - 1)`` is a
            # cheap power-of-two modulo. The two stride primes are coprime
            # to ``pool_size`` so consecutive calls and consecutive slots
            # land on decorrelated pool entries.
            indices = (self._force_accept_rng_counter *
                       _FORCE_ACCEPT_RNG_COUNTER_STRIDE +
                       slot_ids * _FORCE_ACCEPT_RNG_SLOT_STRIDE) & (
                           _FORCE_ACCEPT_RNG_POOL_SIZE - 1)
            rand = self._force_accept_rng_pool[indices]
            extra = (rand < frac_part).to(num_accepted_tokens.dtype)
            # ``base_total + extra`` is at most ``int_part + 2``; clamp so we
            # never exceed the available draft slots.
            force_total_tokens = (base_total + extra).clamp_(max=max_total)
            num_accepted_tokens[num_contexts:] = force_total_tokens
        else:
            num_accepted_tokens[num_contexts:] = base_total

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
        # spec_metadata.runtime_draft_len, because callers may slice a wider
        # runtime token layout down to the K draft tokens used for acceptance.
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
            num_accepted_tokens,
            num_contexts,
            runtime_draft_len,
            spec_metadata=spec_metadata)

        return accepted_tokens, num_accepted_tokens

    def _accept_draft_tokens(self, logits, draft_tokens, num_contexts,
                             batch_size, spec_metadata):
        """
        Accept draft tokens with optional rejection sampling support.

        Mixed batches (num_contexts > 0) are supported: context rows take the
        first sampled target token via the base logic, and rejection sampling
        runs on the gen subset. Draft probs for the gen subset are gathered
        from the slot-indexed buffer by `py_seq_slot`.
        """
        num_gens = batch_size - num_contexts
        if num_gens > 0 and self._can_use_rejection_sampling(spec_metadata):
            draft_len = draft_tokens.shape[1]
            stored_vocab = (spec_metadata.draft_probs_last_dim
                            if spec_metadata.draft_probs_last_dim > 0 else
                            spec_metadata.draft_probs_vocab_size)
            # Gather the slot rows for the gen subset. The buffer was filled
            # at the previous draft step indexed by py_seq_slot, so each gen
            # request reads back exactly its own probs, regardless of batch
            # composition changes since then.
            gen_slot_ids = spec_metadata.batch_slot_ids[num_contexts:batch_size]
            draft_probs = spec_metadata.draft_probs[
                gen_slot_ids, :draft_len, :stored_vocab]
            return self._sample_and_accept_draft_tokens_rejection(
                logits, draft_tokens, draft_probs, num_contexts, batch_size,
                spec_metadata)
        return self._sample_and_accept_draft_tokens_base(
            logits, draft_tokens, num_contexts, batch_size, spec_metadata)

    def _can_use_rejection_sampling(self, spec_metadata: SpecMetadata) -> bool:
        # Skip rejection sampling when the whole batch is greedy: the accepted
        # result is identical to argmax and the base path is cheaper. Mixed
        # batches (context + gen) are handled via slot-indexed draft probs and
        # are split inside _sample_and_accept_draft_tokens_rejection.
        return (spec_metadata.use_rejection_sampling
                and spec_metadata.draft_probs_valid
                and not spec_metadata.is_all_greedy_sample)

    def _sample_and_accept_draft_tokens_rejection(
        self,
        logits: torch.Tensor,
        draft_tokens: torch.Tensor,
        draft_probs: torch.Tensor,
        num_contexts: int,
        batch_size: int,
        spec_metadata,
    ):
        """
        Rejection-sampling acceptance for one-model speculative decoding.

        Mixed batches are handled by treating the two subsets separately:
          - context rows (first `num_contexts`) take the target's sampled first
            token; no draft tokens to verify.
          - generation rows (`[num_contexts:batch_size]`) run the rejection
            sampling kernel on slot-gathered draft probs.

        Per-token sampling-parameter tensors (`temperatures / top_ks / top_ps`)
        are laid out as `[ctx (1 each), gen (draft_len+1 each)]`, matching the
        logits layout, so slicing is symmetric for both subsets.
        """
        device = logits.device
        vocab_size = logits.shape[-1]
        num_gens = batch_size - num_contexts
        runtime_draft_len = draft_tokens.shape[1]

        if logits.dim() == 1:
            logits = logits.unsqueeze(0)

        accepted_tokens = torch.empty((batch_size, runtime_draft_len + 1),
                                      dtype=torch.int,
                                      device=device)
        num_accepted_tokens = torch.ones(batch_size,
                                         dtype=torch.int,
                                         device=device)

        # === Context subset: sample target's first token directly ===
        if num_contexts > 0:
            ctx_target_tokens = self._sample_tokens_for_batch(
                logits[:num_contexts], spec_metadata, num_contexts,
                num_contexts)
            accepted_tokens[:num_contexts, 0] = ctx_target_tokens

        # === Generation subset: rejection sampling on the gen slice ===
        if num_gens > 0:
            num_gen_logits = num_gens * (runtime_draft_len + 1)
            gen_logits = logits[num_contexts:num_contexts + num_gen_logits]
            gen_start = num_contexts
            gen_end = num_contexts + num_gen_logits

            temperatures = spec_metadata.temperatures[gen_start:gen_end]
            # Pass None instead of an all-disabled tensor so the C++ op can short-circuit
            # on a host-side check rather than a `.item<bool>()` sync, which would break
            # CUDA graph capture.
            top_ks = (None if spec_metadata.skip_top_k else
                      spec_metadata.top_ks[gen_start:gen_end])
            top_ps = (None if spec_metadata.skip_top_p else
                      spec_metadata.top_ps[gen_start:gen_end])

            target_probs_flat = compute_probs_from_logits(
                gen_logits, temperatures, top_ks, top_ps)
            target_probs = target_probs_flat.reshape(num_gens,
                                                     runtime_draft_len + 1,
                                                     vocab_size)

            draft_vocab_size = draft_probs.shape[-1]
            assert draft_probs.shape[0] == num_gens, (
                f"draft_probs batch mismatch: {draft_probs.shape[0]} != "
                f"num_gens={num_gens}")
            assert draft_probs.shape[1] == runtime_draft_len, (
                f"draft_probs draft length mismatch: {draft_probs.shape[1]} != "
                f"{runtime_draft_len}")
            d2t = getattr(spec_metadata, "d2t", None)
            if draft_vocab_size != vocab_size:
                # Use the pre-allocated buffer from spec_metadata.prepare()
                # (zero-filled once at init; untouched positions stay 0).
                # Falls back to per-iter allocation if the buffer is not
                # configured, e.g. when use_rejection_sampling was off at
                # prepare() time.
                if spec_metadata.full_draft_probs is not None:
                    full_draft_probs = spec_metadata.full_draft_probs[:num_gens]
                else:
                    full_draft_probs = torch.zeros(
                        (num_gens, runtime_draft_len, vocab_size),
                        dtype=torch.float32,
                        device=device)
                if d2t is not None:
                    assert d2t.numel() == draft_vocab_size, (
                        f"d2t size mismatch: {d2t.numel()} != {draft_vocab_size}"
                    )
                    # d2t is model-static; compute target_indices once and
                    # cache on spec_metadata to skip the arange + add + mod
                    # kernel sequence on every iter.
                    target_indices = spec_metadata.d2t_target_indices
                    if target_indices is None:
                        source_indices = torch.arange(draft_vocab_size,
                                                      device=device,
                                                      dtype=torch.long)
                        target_indices = (source_indices +
                                          d2t.to(device=device)) % vocab_size
                        spec_metadata.d2t_target_indices = target_indices
                    full_draft_probs[:, :runtime_draft_len,
                                     target_indices] = draft_probs
                else:
                    assert draft_vocab_size < vocab_size
                    full_draft_probs[:, :runtime_draft_len, :
                                     draft_vocab_size] = (draft_probs)
            else:
                full_draft_probs = draft_probs

            full_draft_tokens = draft_tokens.to(torch.int32).contiguous()

            if self.seed is None:
                self.seed = torch.tensor([0], dtype=torch.int64, device=device)
            if self.offset is None:
                self.offset = torch.tensor([0],
                                           dtype=torch.int64,
                                           device=device)
            self.seed += 1
            self.seed %= 2**31

            gen_accepted, gen_num_accepted = rejection_sampling_one_model(
                draft_probs=full_draft_probs,
                draft_token_ids=full_draft_tokens,
                target_probs=target_probs,
                deterministic=True,
                seed=self.seed,
                offset=self.offset,
            )

            if self.force_num_accepted_tokens != 0.0:
                # Fill gen_accepted positions 1..runtime_draft_len with all draft tokens
                # so that when _apply_force_accepted_tokens inflates num_accepted_tokens
                # the decoder reads valid draft tokens instead of zeros.
                # Slice bounds are Python ints (static at CUDA-graph capture time).
                gen_accepted[:,
                             1:runtime_draft_len + 1].copy_(full_draft_tokens)

            accepted_tokens[num_contexts:] = gen_accepted
            num_accepted_tokens[num_contexts:] = gen_num_accepted

        num_accepted_tokens = self._apply_force_accepted_tokens(
            num_accepted_tokens,
            num_contexts,
            runtime_draft_len,
            spec_metadata=spec_metadata)
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
        draft_tokens = greedy(logits, return_probs=False)[0]

        # Apply d2t (offsets between draft and target model dictionaries)
        if d2t is not None:
            draft_tokens = d2t[draft_tokens] + draft_tokens

        return draft_tokens.type(torch.int32)

    def _draft_sampler_advanced(
        self,
        logits: torch.Tensor,
        spec_metadata: "SpecMetadata",
        batch_size: int,
        d2t: Optional[torch.Tensor] = None,
    ):
        """
        Draft token sampling using per-request sampling parameters from the
        target's sampling config. Falls back to argmax when the batch is
        all-greedy.

        Args:
            logits: [batch_size, vocab_size] - Draft model logits (one row per
                request, since each draft step emits one token per request).
            spec_metadata: Source of per-request temperatures / top_k / top_p
                tensors populated by populate_sampling_params_for_one_model.
            batch_size: Number of active requests in the batch.
            d2t: Optional dictionary offset tensor for vocab mapping.

        Returns:
            draft_tokens: [batch_size] - Sampled draft token ids (int32)
        """
        if spec_metadata.is_all_greedy_sample:
            return self._draft_sampler_greedy(logits, d2t)

        temperatures = spec_metadata.request_temperatures[:batch_size]
        top_ks = spec_metadata.request_top_ks[:batch_size]
        top_ps = spec_metadata.request_top_ps[:batch_size]

        if self.seed is None:
            self.seed = torch.tensor([0],
                                     dtype=torch.int64,
                                     device=logits.device)
            self.offset = torch.tensor([0],
                                       dtype=torch.int64,
                                       device=logits.device)
        self.seed += 1
        self.seed %= (2**31)

        draft_tokens = sampling_batch_spec_dec_one_model(logits,
                                                         temperatures,
                                                         top_ks,
                                                         top_ps,
                                                         seed=self.seed,
                                                         offset=self.offset)

        if d2t is not None:
            draft_tokens = d2t[draft_tokens] + draft_tokens

        return draft_tokens.type(torch.int32)

    def _draft_sampler_advanced_for_rejection(
        self,
        logits: torch.Tensor,
        spec_metadata: "SpecMetadata",
        batch_size: int,
        d2t: Optional[torch.Tensor] = None,
        draft_step: int = 0,
    ):
        """
        Rejection-sampling-aware variant of ``_draft_sampler_advanced``.

        Single-pass compute + sample + scatter: computes the per-request prob
        distribution once via TRT-LLM's fused ``compute_probs_from_logits``
        (temp + top_k + top_p + softmax + greedy override in one CUDA kernel),
        samples the draft token from that distribution, and scatters the same
        probs into the slot-indexed ``spec_metadata.draft_probs`` buffer for
        next-iter rejection verification. Replaces the previous two-stage path
        (flashinfer fused sampling kernel + a redundant softmax pass to store
        probs).

        All-greedy batches take the cheaper argmax path —
        ``_can_use_rejection_sampling`` will bypass rejection for those anyway.
        """
        if spec_metadata.is_all_greedy_sample:
            return self._draft_sampler_greedy(logits, d2t)

        temperatures = spec_metadata.request_temperatures[:batch_size]
        top_ks = spec_metadata.request_top_ks[:batch_size]
        top_ps = spec_metadata.request_top_ps[:batch_size]

        if self.seed is None:
            self.seed = torch.tensor([0],
                                     dtype=torch.int64,
                                     device=logits.device)
            self.offset = torch.tensor([0],
                                       dtype=torch.int64,
                                       device=logits.device)
        self.seed += 1
        self.seed %= (2**31)

        draft_tokens, probs = sampling_batch_spec_dec_one_model_for_rejection(
            logits,
            temperatures,
            top_ks,
            top_ps,
            seed=self.seed,
            offset=self.offset,
        )

        # Scatter probs into the slot-indexed buffer (shaped
        # [max_num_requests, max_draft_len, vocab_size]). Each request's data
        # always lands at its stable py_seq_slot row regardless of batch
        # composition shifts across iterations.
        assert spec_metadata.batch_slot_ids is not None, (
            "batch_slot_ids must be populated by "
            "populate_sampling_params_for_one_model before draft probs storage")
        batch_slots = spec_metadata.batch_slot_ids[:batch_size]
        vocab = probs.shape[-1]
        spec_metadata.draft_probs[batch_slots, draft_step, :vocab] = probs
        spec_metadata.draft_probs_last_dim = vocab

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
        if attn_metadata.enable_flash_mla:
            attn_metadata.prepare_flash_mla()

        try:
            yield
        finally:
            # Restore main KV cache manager and block offsets
            attn_metadata.kv_cache_manager = target_kv_cache_manager
            attn_metadata.kv_cache_block_offsets = target_kv_cache_block_offsets
            attn_metadata.host_kv_cache_block_offsets = target_host_kv_cache_block_offsets
            if attn_metadata.enable_flash_mla:
                attn_metadata.prepare_flash_mla()

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
        if not spec_metadata.is_all_greedy_sample:
            # Use logits.shape[0] directly: for PARD under CUDA graph capture
            # runtime_draft_len may reflect the PARD-max while the captured
            # graph was built for a shorter draft_len, causing a shape mismatch
            # in sampling_batch_spec_dec_one_model (which is torch.compiled).
            num_tokens = logits.shape[0]

            temperatures = spec_metadata.temperatures[:num_tokens]
            top_ks = spec_metadata.top_ks[:num_tokens]
            top_ps = spec_metadata.top_ps[:num_tokens]

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
                seed=self.seed,
                offset=self.offset)
        else:
            sampled_tokens = torch.argmax(logits, dim=-1)

        return sampled_tokens
