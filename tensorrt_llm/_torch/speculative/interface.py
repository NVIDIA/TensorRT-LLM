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
from ..attention_backend.interface import AttentionMetadata
from ..attention_backend.trtllm import (AttentionBackend, TrtllmAttention,
                                        TrtllmAttentionMetadata)
from ..flashinfer_utils import IS_FLASHINFER_AVAILABLE
from ..pyexecutor.resource_manager import ResourceManagerType

if TYPE_CHECKING:
    from ..pyexecutor.guided_decoder import CapturableGuidedDecoder
    from ..pyexecutor.llm_request import LlmRequest

if IS_FLASHINFER_AVAILABLE:
    import flashinfer

from tensorrt_llm.llmapi.llm_args import AdvancedSamplingMode

from ..pyexecutor.sampler import penalties as penalty_ops
from ..pyexecutor.sampler.ops.spec_dispatch import (
    spec_compute_probs_from_logits, spec_sample_from_logits,
    spec_sample_from_logits_with_probs)
from ..pyexecutor.sampler.ops.vanilla import greedy_search_sampling_batch


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
    if spec_config._use_shared_kv_cache:
        return False
    # The embedded DSpark draft owns a dedicated rolling-window cache in
    # DSv4DSparkWorker and never reads the paged draft KV cache that attention
    # metadata manages. A standalone DSpark drafter runs on DSparkWorker
    # (DFlash lineage), which does read it, so it keeps the default -- hence a
    # form check, not a mode check
    # (see DSparkDecodingConfig.draft_is_embedded_in_target).
    if (spec_config.spec_dec_mode.is_dspark()
            and spec_config.draft_is_embedded_in_target):
        return False
    return spec_config._allow_separate_draft_kv_cache


def prepare_attn_metadata_for_draft_replay(attn_metadata,
                                           draft_kv_cache_manager):
    """
    Prepare attention metadata for a draft forward or CUDA graph replay when using a
    separate draft KV cache. Swaps cache-layout-dependent buffers, refreshes FlashMLA
    block IDs outside capture, and (for DSA) re-prepares indexer slot mappings
    for the current batch.
    Call restore_attn_metadata_after_draft_replay in a finally block.
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
    if attn_metadata.enable_flash_mla:
        if (attn_metadata.draft_block_ids_per_seq is None
                or attn_metadata.draft_kv_block_ids_per_seq is None):
            raise RuntimeError(
                "FlashMLA separate draft KV cache requires dedicated draft block-ID buffers"
            )
        saved['target_block_ids_per_seq'] = attn_metadata.block_ids_per_seq
        saved[
            'target_kv_block_ids_per_seq'] = attn_metadata.kv_block_ids_per_seq
        attn_metadata.block_ids_per_seq = attn_metadata.draft_block_ids_per_seq
        attn_metadata.kv_block_ids_per_seq = (
            attn_metadata.draft_kv_block_ids_per_seq)
    attn_metadata.kv_cache_manager = draft_kv_cache_manager
    attn_metadata.kv_cache_block_offsets = attn_metadata.draft_kv_cache_block_offsets
    attn_metadata.host_kv_cache_block_offsets = (
        draft_kv_cache_manager.host_kv_cache_block_offsets)
    if attn_metadata.enable_flash_mla:
        attn_metadata.prepare_flash_mla()

    # Backends select any additional draft-forward state, such as native DSA
    # indexer buffers or DeepSeek-V4 sparse tables and pool pointers.
    backend_saved = attn_metadata.prepare_for_draft_forward()
    if backend_saved is not None:
        saved['saved_backend_state'] = backend_saved
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
        attn_metadata.block_ids_per_seq = saved_state[
            'target_block_ids_per_seq']
        attn_metadata.kv_block_ids_per_seq = saved_state[
            'target_kv_block_ids_per_seq']
        # Target and draft block-ID buffers are independent. Restoring only
        # needs to invalidate the scheduler metadata; refreshing the unchanged
        # target buffers would repeat request-specific H2D work.
        attn_metadata._flash_mla_metadata_valid = False
    attn_metadata.restore_after_draft_forward(
        saved_state.get('saved_backend_state'))


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
    DSPARK = auto()
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

    def is_dspark(self):
        return self == SpeculativeDecodingMode.DSPARK

    def is_parallel_draft(self):
        return self.is_pard() or self.is_dflash() or self.is_dspark()

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
        is_draft_model: bool,
        attention_backend: Type[AttentionBackend],
    ):
        """
        If true, the attention backend kernel needs to run in spec-dec mode (multi-token query mode).
        Args:
            is_draft_model: whether the model is a draft model.
            attention_backend: the attention backend.
        """
        is_trtllm_attention = issubclass(attention_backend, TrtllmAttention)

        # Always use the multi-token query mode for 1-model if the kernels are available.
        use_case_1 = self.use_one_engine()
        # For 2-model, only the target model (verification) processes multiple tokens at once.
        use_case_2 = (not self.use_one_engine() and not is_draft_model
                      and is_trtllm_attention)

        return use_case_1 or use_case_2

    @staticmethod
    def from_string(name: Optional[str]) -> "SpeculativeDecodingMode":
        if name is None:
            return SpeculativeDecodingMode.NONE
        return SpeculativeDecodingMode[name.upper()]


# Philox seed for requests that did not set ``SamplingParams.seed``. Fixed
# rather than advanced per step so a run is reproducible: a request's stream is
# separated from other rows' by the kernel's per-row subsequence and from its
# own earlier steps by the offset, which leaves the seed free to be a constant.
DEFAULT_SAMPLING_SEED = 42


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
    # The number of generation requests for the speculative model/layer of each
    # rank (num_seqs - num_contexts). Used by external drafters (e.g. DSpark)
    # whose draft forward processes only generation requests and must size a
    # FUSED_COMM MoE (DeepGEMM MegaMoE) chunk loop identically across EP ranks.
    all_rank_num_gens: Optional[List[int]] = None
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
    # Group-synchronized override for ``is_all_greedy_sample`` (AND over the
    # TP group's local flags; None = no group sync configured, use the local
    # value). Under ADP + LM-head TP with rejection sampling, the greedy-vs-
    # advanced choice gates group collectives, so all ranks must take the same
    # path even though their batches (and thus local flags) differ. Set by
    # ``_sync_group_all_greedy_sample`` before the CUDA graph key is built and
    # re-applied by ``_scan_one_model_sampling`` on every rescan. AND is safe:
    # a greedy rank pulled onto the advanced path still samples greedily via
    # its sentinel params.
    group_all_greedy_sample: Optional[bool] = None
    # Whether to use rejection sampling for one-model speculative decoding.
    use_rejection_sampling: bool = False
    # Advanced-sampling specialization (deploy-time; from DecodingBaseConfig.advanced_sampling_mode).
    advanced_sampling_mode: AdvancedSamplingMode = AdvancedSamplingMode.FULL
    # Whether the occurrence penalties are enabled (deploy-time; from
    # DecodingBaseConfig.enable_penalty). Gates the occurrence workspace allocation.
    enable_penalty: bool = False
    # Occurrence-penalty device state; None until prepare_penalty_buffers runs.
    # See penalty_ops.PenaltyState for what each buffer holds and why the prompt is
    # split from generated tokens.
    penalty_state: Optional["penalty_ops.PenaltyState"] = None
    # Whether any request in the current batch has a penalty. Diagnostic only: it
    # must NOT gate the apply pass, because decode steps replay a CUDA graph
    # captured during warmup, when the flag is necessarily False. Whether a row is
    # actually penalized is decided on device by ``penalty_active``.
    #
    # Held in a list rather than a plain bool because create_cuda_graph_metadata
    # shallow-copies this object and both views must observe the same value; same
    # reasoning as _sampling_params_signature below.
    _batch_uses_penalty: list = field(default_factory=lambda: [False],
                                      repr=False)

    @property
    def batch_uses_penalty(self) -> bool:
        return self._batch_uses_penalty[0]

    @batch_uses_penalty.setter
    def batch_uses_penalty(self, value: bool) -> None:
        self._batch_uses_penalty[0] = value

    # Sampling parameters for non-greedy sampling (per-request)
    temperatures: Optional[torch.Tensor] = None
    top_ks: Optional[torch.Tensor] = None
    top_ps: Optional[torch.Tensor] = None
    # Only read when advanced_sampling_mode is UNIVERSAL -- the other modes reject a
    # min_p request at admission, so the buffer stays at its neutral 0.0 for them.
    # Allocated regardless: it is one float per row, and making its existence conditional
    # would make the CUDA-graph capture depend on the deploy mode for no saving.
    min_ps: Optional[torch.Tensor] = None
    # Pre-computed top_k_max scalar (CPU-side) to avoid CUDA-graph-incompatible
    # dynamic boolean tensor indexing inside verify_dynamic_tree_rejection_from_logits_out.
    top_k_max: int = 0
    # Sampling parameters indexed per request.
    request_temperatures: Optional[torch.Tensor] = None
    request_top_ks: Optional[torch.Tensor] = None
    request_top_ps: Optional[torch.Tensor] = None
    request_min_ps: Optional[torch.Tensor] = None
    # Describe what the sampling-parameter buffers currently hold, so a step
    # that reproduces them can skip the refill. Two entries because the two
    # buffer groups depend on different things:
    #   [0] request_* -- the per-request values, in batch order.
    #   [1] the expanded per-token buffers, which additionally depend on each
    #       request's token count (a context request contributing one row
    #       instead of draft_len + 1 shifts every later request's offset).
    # A context->generation transition therefore invalidates [1] while leaving
    # [0] valid. See _sampling_params_buffers_need_update.
    #
    # Held in a list rather than plain fields because
    # create_cuda_graph_metadata shallow-copies this object: the graph views
    # and the eager view write the *same* tensors, so they must agree on what
    # those tensors hold. Plain fields would give each view its own stale
    # answer and let one skip a fill another view invalidated.
    _sampling_params_signature: list = field(
        default_factory=lambda: [None, None], repr=False)
    # Per-row Philox state for user-specified ``SamplingParams.seed``, laid out
    # to match the logits rows the sampling kernels consume.
    #
    # ``request_seeds`` carries the request's own seed (the engine-wide seed
    # for unseeded requests). ``request_offsets`` carries how far that
    # request's stream has advanced, which is what separates one step from the
    # next: with a fixed user seed the offset is the only thing that changes,
    # and taking it from the request's own progress -- rather than a global
    # step counter -- keeps a seeded request reproducible regardless of which
    # batch it lands in.
    #
    # NB: the pinned flashinfer reads only element 0 of each tensor, separating
    # rows by blockIdx.x, so these per-row values are carried end-to-end but
    # not yet honored per request. See
    # https://github.com/flashinfer-ai/flashinfer/pull/2345.
    request_seeds: Optional[torch.Tensor] = None
    request_offsets: Optional[torch.Tensor] = None
    # Per-slot count of RNG windows already handed out, keyed by py_seq_slot.
    #
    # This deliberately does NOT read request.py_decoding_iter: the overlap
    # scheduler runs _forward_step (where this is populated) before the
    # previous batch's _update_requests, which is what increments that field.
    # A request appearing in adjacent batches would therefore be seen at the
    # same iteration twice and replay the same offset window. Counting the
    # windows we hand out keeps the stream advancing once per sampling pass
    # under either scheduler.
    #
    # Held in a dict so create_cuda_graph_metadata's copy.copy keeps graph and
    # eager views sharing one counter; keyed by slot rather than batch position
    # because batch composition shifts between iterations. Bounded by the slot
    # pool, which SeqSlotManager frees and reuses on request completion.
    #
    # The counter is not reset when a slot is reused, so a new request on a
    # recycled slot starts partway into its stream. That is still a disjoint
    # region of it, so sampling stays correct; the cost is that a seeded
    # request reproduces bit-exactly only for a given slot history.
    _rng_window_counter: dict = field(default_factory=dict)
    # The same state expanded to one entry per logits row, mirroring the
    # temperatures / top_ks / top_ps layout, for the sampling calls that
    # consume rows rather than requests.
    seeds: Optional[torch.Tensor] = None
    offsets: Optional[torch.Tensor] = None
    # Whether to use sampling parameters when sampling draft tokens.
    use_sampling_params_for_draft_tokens: bool = False
    # Vocab size used for draft_probs buffer allocation.
    vocab_size: int = 0
    # Size of the SeqSlotManager pool. py_seq_slot values range over
    # [0, num_seq_slots); DeepSeek-V4 overlap can use 2 * max_batch_size,
    # larger than max_num_requests (== max_batch_size).
    # Slot-indexed buffers (draft_probs) must span this full range.
    # 0 falls back to max_num_requests.
    num_seq_slots: int = 0
    # Draft-model vocab size. full_draft_probs is allocated only when it differs
    # from vocab_size; 0 (unknown) or a value equal to vocab_size means shared
    # vocab and skips the buffer.
    draft_vocab_size: int = 0
    # Draft probabilities buffer for rejection sampling, indexed by py_seq_slot
    # so per-request data is stable across iterations regardless of batch
    # composition shifts (chunking ctx, gen completion, new ctx joining).
    # Shape: [num_seq_slots, max_draft_len, vocab_size].
    draft_probs: Optional[torch.Tensor] = None
    draft_probs_vocab_size: int = 0
    # Scratch row index that dummy/padding requests (py_seq_slot is None) route
    # to, captured when draft_probs is allocated so it tracks the buffer's real
    # size. It must NOT be re-derived from max_num_requests at use time:
    # create_cuda_graph_metadata shrinks max_num_requests to the graph bucket
    # size while sharing the full-size buffer, so a bucket-relative index would
    # collide with a real request's slot row and overwrite its draft probs.
    dummy_slot_row: int = 0
    # Last dimension size of the draft logits/probs stored in draft_probs.
    draft_probs_last_dim: int = 0
    # Per-request slot ids (py_seq_slot) for the current batch, in batch order.
    # Used to scatter draft probs by slot at write time and gather them by slot
    # at the next iter's verify. Shape: [max_num_requests], dtype=long.
    batch_slot_ids: Optional[torch.Tensor] = None
    # Draft probs expanded to the target vocab size. Zero-filled once at
    # prepare(); each rejection iter overwrites only the d2t-selected positions
    # (or [:draft_vocab] when there is no d2t).
    # Shape: [max_num_requests, max_draft_len, vocab_size].
    full_draft_probs: Optional[torch.Tensor] = None
    # Cached d2t-projected target vocab indices, computed once on first use.
    # Shape: [draft_vocab_size], dtype long.
    d2t_target_indices: Optional[torch.Tensor] = None

    def __post_init__(self):
        pass

    def _populate_request_rng_state(
        self, requests: list["LlmRequest"],
        per_request_normalized: list[tuple[float, int, float, float,
                                           int]]) -> None:
        """Fill the Philox seed/offset buffers for this batch.

        A request's seed is fixed for its lifetime, so the offset is what has
        to advance between steps -- otherwise every step of a seeded request
        would draw the same numbers. Taking it from the request's own window
        counter, rather than a global step counter, is what ties the stream to
        how far that request has decoded instead of to when it was scheduled.

        Both layouts are produced: ``request_*`` with one entry per request,
        and ``seeds`` / ``offsets`` expanded to one entry per logits row (the
        temperatures / top_ks / top_ps layout), because the sampling calls
        take one or the other.

        A request that specified no seed gets ``DEFAULT_SAMPLING_SEED``. Its
        stream is then separated from the other rows' by the kernel's per-row
        subsequence and from its own earlier steps by the offset, so unseeded
        requests still sample independently -- just reproducibly.
        """
        from tensorrt_llm._torch.pyexecutor.sampler.sampler_common import \
            request_random_seed

        request_seeds = [
            seed if (seed := request_random_seed(request)) is not None else
            DEFAULT_SAMPLING_SEED for request in requests
        ]
        # Base of this step's Philox offset window. Each sampling pass owns
        # max_draft_len + 1 consecutive offsets: the target sampler (or the
        # rejection kernel, which is its alternative) takes the first, and
        # draft step i takes base + 1 + i. Sizing the window by the static
        # max_draft_len rather than the runtime one keeps a step's offsets
        # disjoint from its neighbours' even when the draft length shrinks.
        #
        # The window index comes from _rng_window_counter, not from
        # py_decoding_iter, which is still stale here under the overlap
        # scheduler (see the field's comment).
        window = self.max_draft_len + 1
        request_offsets = []
        for request in requests:
            slot = request.py_seq_slot
            # Dummy/padding requests (no slot) never have their output kept,
            # so they share one counter rather than perturbing a real slot's.
            step = self._rng_window_counter.get(slot, 0)
            self._rng_window_counter[slot] = step + 1
            request_offsets.append(step * window)
        num_tokens_per_request = [n for *_, n in per_request_normalized]

        flat_seeds: list[int] = []
        flat_offsets: list[int] = []
        for seed, offset, num_tokens in zip(request_seeds, request_offsets,
                                            num_tokens_per_request):
            flat_seeds.extend(seed for _ in range(num_tokens))
            flat_offsets.extend(offset for _ in range(num_tokens))

        # A batch wider than the buffers would silently truncate the copies
        # below, so assert rather than grow: CUDA graph batch sizes are already
        # clamped to the executor's batch size (_filter_cuda_graph_batch_sizes),
        # and graph padding refuses to cross it, so exceeding it here means the
        # invariant broke upstream and should surface.
        assert len(requests) <= self.max_num_requests, (
            f"batch has {len(requests)} requests but max_num_requests is "
            f"{self.max_num_requests}")
        if (self.request_seeds is None
                or self.request_seeds.numel() < self.max_num_requests):
            self.request_seeds = torch.zeros(self.max_num_requests,
                                             dtype=torch.int64,
                                             device='cuda')
            self.request_offsets = torch.zeros(self.max_num_requests,
                                               dtype=torch.int64,
                                               device='cuda')
        if self.seeds is None or self.seeds.numel() < len(flat_seeds):
            # Match the per-token buffers' capacity so a later batch with more
            # rows does not reallocate mid-stream.
            capacity = max(
                len(flat_seeds),
                self.temperatures.numel()
                if self.temperatures is not None else 0)
            self.seeds = torch.zeros(capacity, dtype=torch.int64, device='cuda')
            self.offsets = torch.zeros(capacity,
                                       dtype=torch.int64,
                                       device='cuda')

        def _upload(dst: torch.Tensor, values: list[int]) -> None:
            dst[:len(values)].copy_(torch.tensor(values,
                                                 dtype=torch.int64,
                                                 pin_memory=prefer_pinned()),
                                    non_blocking=True)

        _upload(self.request_seeds, request_seeds)
        _upload(self.request_offsets, request_offsets)
        _upload(self.seeds, flat_seeds)
        _upload(self.offsets, flat_offsets)

    def prepare_rejection_sampling_buffers(self):
        """
        Allocate the slot-indexed buffers used by one-model rejection sampling.

        Idempotent and gated on ``use_rejection_sampling``.
        """
        if not self.use_rejection_sampling:
            return

        # Slot-indexed buffers span the full SeqSlotManager pool: py_seq_slot
        # can range over [0, num_seq_slots), which under DeepSeek-V4 overlap
        # exceeds max_num_requests. Fall back to max_num_requests when the pool
        # size is unknown (0). One extra scratch row at index ``slot_capacity``
        # absorbs CUDA-graph dummy/padding requests (``py_seq_slot is None``).
        slot_capacity = self.num_seq_slots or self.max_num_requests
        num_slot_rows = slot_capacity + 1

        if self.draft_probs is None and self.vocab_size > 0:
            # [slot, draft_step, vocab]: scatter/gather by stable slot id.
            self.draft_probs = torch.empty(
                (num_slot_rows, self.max_draft_len, self.vocab_size),
                dtype=torch.float32,
                device='cuda')
            self.draft_probs_vocab_size = self.vocab_size
            # Dummy requests route to the extra scratch row (the buffer's last
            # row at index ``slot_capacity``). Capture it here against the real
            # allocation size so it stays correct on graph copies that shrink
            # max_num_requests and under overlap where slot_capacity exceeds it.
            self.dummy_slot_row = slot_capacity
        if self.batch_slot_ids is None and self.max_num_requests > 0:
            self.batch_slot_ids = torch.empty((self.max_num_requests, ),
                                              dtype=torch.long,
                                              device='cuda')
        # full_draft_probs (d2t-expanded) is read only when draft and target
        # vocabularies differ; skip it otherwise. Zero-filled once.
        if (self.full_draft_probs is None and self.vocab_size > 0
                and self.draft_vocab_size not in (0, self.vocab_size)):
            self.full_draft_probs = torch.zeros(
                (num_slot_rows, self.max_draft_len, self.vocab_size),
                dtype=torch.float32,
                device='cuda')

    def prepare_penalty_buffers(self):
        """Allocate the occurrence-penalty state. Idempotent; no-op when disabled.

        Sized and indexed exactly like the rejection buffers -- see
        ``prepare_rejection_sampling_buffers`` for why the slot pool is
        ``num_seq_slots`` (not ``max_num_requests``) and why one scratch row is
        appended for dummy/padding requests.
        """
        if not self.enable_penalty or self.vocab_size <= 0:
            return
        slot_capacity = self.num_seq_slots or self.max_num_requests
        if slot_capacity <= 0:
            return

        if self.penalty_state is None:
            self.penalty_state = penalty_ops.PenaltyState.create(
                slot_capacity=slot_capacity, vocab_size=self.vocab_size)
        # The scratch row padding requests route to. Normally published by
        # prepare_rejection_sampling_buffers, which returns early when rejection
        # sampling is off -- leaving it at 0, a live request's row -- so publish it
        # here too. Both buffers append their scratch row at the same index, so the
        # single value stays correct for either consumer.
        self.dummy_slot_row = slot_capacity
        if self.batch_slot_ids is None and self.max_num_requests > 0:
            # Normally allocated by prepare_rejection_sampling_buffers; the penalties
            # need the same row -> slot table even when rejection sampling is off.
            self.batch_slot_ids = torch.empty((self.max_num_requests, ),
                                              dtype=torch.long,
                                              device='cuda')

    @staticmethod
    def _request_prompt_tokens(request):
        """The request's prompt token ids, or None when they are unavailable.

        Sliced to ``py_orig_prompt_len`` so only the prompt is taken, never tokens
        the model has already generated -- those belong to the output counts.
        """
        get_tokens = getattr(request, "get_tokens", None)
        if get_tokens is None:
            return None
        prompt_len = getattr(request, "py_orig_prompt_len", None)
        tokens = get_tokens(0)
        if prompt_len is not None:
            tokens = tokens[:prompt_len]
        if not len(tokens):
            return None
        return torch.tensor(tokens,
                            dtype=torch.int64,
                            pin_memory=prefer_pinned())

    @staticmethod
    def _penalty_value(sampling_config, name: str, default: float) -> float:
        """Read one penalty parameter, which the C++ SamplingConfig holds as an
        optional singleton list. Missing / empty / None all mean "not set"."""
        if sampling_config is None:
            return default
        values = getattr(sampling_config, name, None)
        if values is None:
            return default
        if isinstance(values, (list, tuple)):
            if len(values) == 0 or values[0] is None:
                return default
            return float(values[0])
        return float(values)

    def _populate_penalty_params(self, requests):
        """Write each request's penalty parameters into its slot row.

        Only the rows of the requests in this batch are touched; a slot keeps its
        values for the request's lifetime, and ``penalty_active`` is rewritten every
        step so a slot whose new occupant has no penalties stops being penalized
        (slot reuse would otherwise inherit the previous request's parameters).

        Counts for a newly admitted slot are zeroed here as well, for the same
        reason. Requests without a slot (CUDA-graph dummies) are skipped; their
        rows keep the no-op defaults.
        """
        state = self.penalty_state
        if not self.enable_penalty or state is None:
            return

        slots: list[int] = []
        repetition: list[float] = []
        presence: list[float] = []
        frequency: list[float] = []
        active: list[bool] = []
        reset_slots: list[int] = []
        seed_requests: list[tuple] = []
        num_rows = state.counts.size(0)

        for request in requests:
            slot = getattr(request, "py_seq_slot", None)
            # Live rows only: a negative slot would wrap onto another request's
            # row, and the last row is the CUDA-graph padding scratch row.
            if slot is None or not 0 <= slot < num_rows - 1:
                continue
            config = getattr(request, "sampling_config", None)
            rep = self._penalty_value(config, "repetition_penalty", 1.0)
            pre = self._penalty_value(config, "presence_penalty", 0.0)
            freq = self._penalty_value(config, "frequency_penalty", 0.0)
            ignore_len = int(
                self._penalty_value(config, "prompt_ignore_length", 0.0))
            slots.append(slot)
            repetition.append(rep)
            presence.append(pre)
            frequency.append(freq)
            active.append(rep != 1.0 or pre != 0.0 or freq != 0.0)
            # A context request is starting its sequence: drop whatever the slot's
            # previous occupant accumulated, then seed the prompt this one starts
            # from. Only penalized requests are seeded -- reading the prompt costs a
            # host-side copy that an unpenalized request would never consult.
            #
            # Gated on the LAST context chunk rather than merely on the context
            # state: under chunked prefill ``is_context_init_state`` stays true for
            # every chunk, so resetting on each one would repeatedly wipe the state
            # the earlier chunks built and re-seed the prompt several times.
            if getattr(request, "is_context_init_state", False) and getattr(
                    request, "is_last_context_chunk", True):
                reset_slots.append(slot)
                if active[-1]:
                    seed_requests.append((slot, request, ignore_len))

        # Host-side gate for the apply pass: with no penalized request in the batch,
        # the whole vocab-sized rewrite is a no-op worth skipping.
        self.batch_uses_penalty = any(active)

        if not slots:
            return

        slots_cuda = torch.tensor(slots,
                                  dtype=torch.long,
                                  pin_memory=prefer_pinned()).to(
                                      'cuda', non_blocking=True)
        params = torch.tensor([repetition, presence, frequency],
                              dtype=torch.float32,
                              pin_memory=prefer_pinned()).to('cuda',
                                                             non_blocking=True)
        state.repetition.index_copy_(0, slots_cuda, params[0])
        state.presence.index_copy_(0, slots_cuda, params[1])
        state.frequency.index_copy_(0, slots_cuda, params[2])
        state.active.index_copy_(
            0, slots_cuda,
            torch.tensor(active, dtype=torch.bool,
                         pin_memory=prefer_pinned()).to('cuda',
                                                        non_blocking=True))
        if reset_slots:
            reset_cuda = torch.tensor(reset_slots,
                                      dtype=torch.long,
                                      pin_memory=prefer_pinned()).to(
                                          'cuda', non_blocking=True)
            state.counts.index_fill_(0, reset_cuda, 0)
            # The prompt bitmask is per-sequence too: a reused slot must not inherit
            # the previous occupant's prompt.
            state.prompt_mask.index_fill_(0, reset_cuda, 0)

        # Seeded after the reset above, or the clear would wipe what we just wrote.
        for slot, request, ignore_len in seed_requests:
            prompt = self._request_prompt_tokens(request)
            if prompt is not None:
                penalty_ops.seed_prompt(self, slot, prompt, ignore_len)

    def write_padding_onehot_draft_probs(self, padding_slot_ids, draft_len):
        """Write a one-hot draft-prob row (prob 1.0 at draft-vocab token id 0,
        the placeholder token) into each padding gen request's stable slot row.

        Padding requests are gen requests that entered this iteration with 0 real
        draft tokens (e.g. a runtime_draft_len K->0->K dynamic-draft-len toggle).
        Their slot's draft_probs row was never scattered by the draft sampler, so
        the next iteration's (possibly CUDA-graph-captured) rejection kernel would
        read a stale/uninitialized distribution. Writing a legal one-hot row makes
        acceptance reject the placeholder and resample from the target (equivalent
        to strict acceptance) for those rows. Written eagerly before graph replay
        into the stable draft_probs buffer, so the replayed kernel reads it.

        Idempotent w.r.t. context->gen transitions whose row was already one-hot'd
        by write_context_onehot_draft_probs. The width matches the value already
        published in draft_probs_last_dim (what acceptance reads), so it is NOT
        overwritten here. No-op unless rejection is enabled and slots exist.
        Static shapes -> CUDA-graph safe.
        """
        if (not padding_slot_ids
                or not getattr(self, "use_rejection_sampling", False)
                or self.draft_probs is None):
            return
        onehot_vocab = (self.draft_probs_last_dim if self.draft_probs_last_dim
                        > 0 else self.draft_probs_vocab_size)
        slots = torch.tensor(padding_slot_ids,
                             dtype=torch.long,
                             device=self.draft_probs.device)
        self.draft_probs[slots, :draft_len, :onehot_vocab] = 0.0
        self.draft_probs[slots, :draft_len, 0] = 1.0

    def prepare(self):
        """
        Hook to be called before the forward step of the model.
        """
        self.prepare_rejection_sampling_buffers()

    def create_cuda_graph_metadata(self, max_batch_size: int):
        """
        Creates metadata for CUDA graph execution.
        """
        if self.is_cuda_graph:
            return self

        cuda_graph_metadata = copy.copy(self)
        cuda_graph_metadata.is_cuda_graph = True
        cuda_graph_metadata.max_num_requests = max_batch_size
        # NB: the shallow copy deliberately keeps sharing
        # _sampling_params_signature with this object. Both views write the
        # same sampling-parameter tensors, so the record of what those tensors
        # hold has to be shared too.
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
    ) -> tuple[list[tuple[float, int, float, float, int]], list[int]]:
        """Single source of truth for one-engine sampling-param detection.

        Scans the batch's sampling configs and sets skip_*/is_all_greedy_sample
        (honoring the group-synchronized value, see below). Returns
        ``(per_request_normalized, per_request_slot_ids)`` for buffer
        population. Does NOT allocate or fill GPU buffers, so it is safe to call
        before the CUDA graph key is built.
        """
        from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequestState
        from tensorrt_llm._torch.pyexecutor.sampler.ops.vanilla import \
            GREEDY_TEMPERATURE_THRESHOLD
        from tensorrt_llm.sampling_params import SamplingParams

        # Sentinel temperature for greedy / temperature-disabled rows. Must stay
        # strictly below GREEDY_TEMPERATURE_THRESHOLD so the sampling kernels
        # recognize these rows as greedy; small enough that even if a row were
        # (incorrectly) sampled, softmax(logits / val) is effectively one-hot,
        # and non-zero to avoid division by 0.
        DISABLE_TEMP_VAL = GREEDY_TEMPERATURE_THRESHOLD / 10
        # Very large values disable topk.
        DISABLE_TOPK_VAL = torch.iinfo(torch.int32).max
        DISABLE_TOPP_VAL = 1.0
        DISABLE_MINP_VAL = 0.0

        def _first_or_none(values):
            """Return the first sampling parameter value when present."""
            return values[0] if values is not None and len(values) > 0 else None

        def _normalize_request_sampling_params(
            *,
            temperature: Optional[float],
            top_k: Optional[int],
            top_p: Optional[float],
            min_p: Optional[float],
        ) -> tuple[float, int, float, float, bool]:
            """Convert request sampling params into normalized per-request scalars."""
            # min_p participates in the greedy classification and must: a request whose
            # ONLY knob is min_p is not greedy (params_imply_greedy_decoding treats
            # 0 < min_p < 1 as an active sampling knob), and min_p == 1.0 is explicit
            # greedy. Omitting it here would classify a min_p-only request as greedy, send
            # it down the argmax fast path, and drop min_p silently -- and since this flag
            # is part of the CUDA graph key, it would also pick the wrong graph variant.
            #
            # Requests only reach here carrying min_p when the deploy selected
            # advanced_sampling_mode=UNIVERSAL; every other mode rejects them at admission
            # (SpecSampler.validate_request). So this is unconditional: on the other modes
            # min_p is always None and the classification is unchanged.
            is_greedy = SamplingParams.params_imply_greedy_decoding(
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                min_p=min_p,
                use_beam_search=False)

            use_top_k = not is_greedy and top_k is not None and top_k > 0

            normalized_temperature = (DISABLE_TEMP_VAL
                                      if is_greedy or temperature is None
                                      or temperature == 0 else temperature)
            normalized_top_k = DISABLE_TOPK_VAL if not use_top_k else top_k
            normalized_top_p = (DISABLE_TOPP_VAL
                                if is_greedy or top_p is None else top_p)
            # Unlike top_p, min_p's neutral value is 0.0, not 1.0.
            normalized_min_p = (DISABLE_MINP_VAL
                                if is_greedy or min_p is None else min_p)

            return (
                normalized_temperature,
                normalized_top_k,
                normalized_top_p,
                normalized_min_p,
                is_greedy,
            )

        # Phase 1: collect per-request flags and normalized values.
        per_request_normalized: list[tuple[float, int, float, float, int]] = []
        has_non_greedy_requests = False
        per_request_slot_ids: list[int] = []

        for request in requests:
            sampling_config = request.sampling_config
            temp_val = _first_or_none(sampling_config.temperature)
            tk_val = _first_or_none(sampling_config.top_k)
            tp_val = _first_or_none(sampling_config.top_p)
            mp_val = _first_or_none(sampling_config.min_p)

            # Context requests have no draft tokens yet.
            num_tokens = 1 + self.runtime_draft_len if request.state == LlmRequestState.GENERATION_IN_PROGRESS else 1

            (
                temp_val,
                tk_val,
                tp_val,
                mp_val,
                is_greedy,
            ) = _normalize_request_sampling_params(
                temperature=temp_val,
                top_k=tk_val,
                top_p=tp_val,
                min_p=mp_val,
            )

            has_non_greedy_requests |= not is_greedy

            per_request_normalized.append(
                (temp_val, tk_val, tp_val, mp_val, num_tokens))
            # py_seq_slot is a stable per-request id used to scatter/gather draft
            # probs across iterations. Dummy/padding requests (py_seq_slot is
            # None) route to the scratch row captured at allocation time (the
            # buffer's real last row), not max_num_requests, which a graph copy
            # shrinks to the bucket size and would alias a real request's slot.
            per_request_slot_ids.append(
                request.py_seq_slot if request.
                py_seq_slot is not None else self.dummy_slot_row)

        # Used in the CUDA graph key to pick the argmax / advanced variant.
        # All-greedy iff EVERY request is greedy -- note a non-greedy request
        # may still enable no filter (e.g. temperature=1.0 with top_k/top_p
        # unset), so this cannot be derived from which filters are in use.
        self.is_all_greedy_sample = not has_non_greedy_requests

        # Apply the group-synchronized value last (semantics: see the
        # ``group_all_greedy_sample`` field comment). Local contract: rescans
        # (e.g. populate after the graph key) must converge to the synced
        # value rather than resurrect the local value.
        if self.group_all_greedy_sample is not None:
            self.is_all_greedy_sample = self.group_all_greedy_sample
        return per_request_normalized, per_request_slot_ids

    @property
    def wants_advanced_draft_sampling(self) -> bool:
        """Whether the current batch takes the advanced (rejection) draft
        path: rejection sampling enabled AND not an all-greedy batch.

        Single source of truth for the greedy-vs-advanced decision: the
        sampler branch (``sample_draft_tokens``) and the worker's LM-head-TP
        bypass (``_forward_linear_draft_loop``) must agree exactly -- a
        divergence feeds the wrong logits layout to the sampler -- so both
        read this property instead of re-deriving the predicate.
        """
        return self.use_rejection_sampling and not self.is_all_greedy_sample

    def update_is_all_greedy_sample(self, requests: list["LlmRequest"]) -> None:
        """Refresh ``is_all_greedy_sample`` for the *current* batch.

        Must be called BEFORE the CUDA graph key is built (the key includes
        ``is_all_greedy_sample`` to choose the argmax vs advanced-sampling graph
        variant), so the selected graph stays consistent with the buffers
        ``populate_sampling_params_for_one_model`` fills later.
        """
        if not self.spec_dec_mode.use_one_engine():
            return
        # The synchronized group decision belongs to the previous iteration.
        # Clear it before deriving this iteration's local flag; the caller
        # immediately recomputes the group decision before graph-key lookup.
        self.group_all_greedy_sample = None
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

        # Allocate the rejection buffers before copying py_seq_slot values into
        # batch_slot_ids below; this runs earlier than prepare() in the
        # model-engine flow. No-op unless use_rejection_sampling is set.
        self.prepare_rejection_sampling_buffers()
        # Likewise for the occurrence-penalty workspace. No-op unless enable_penalty.
        self.prepare_penalty_buffers()

        if self.temperatures is None:
            # Ensures determinism across ranks.
            torch.manual_seed(0)

        per_request_normalized, per_request_slot_ids = (
            self._scan_one_model_sampling(requests))

        tokens_per_request = (self.max_total_draft_tokens + 1 if
                              self.is_spec_dec_tree else self.max_draft_len + 1)
        # Warmup batches may exceed max_num_requests * tokens_per_request (e.g.
        # when CUDA-graph warmup passes use max_batch_size > max_num_requests).
        actual_flat_size = sum(num_tokens
                               for *_, num_tokens in per_request_normalized)
        required_flat_size = max(tokens_per_request * self.max_num_requests,
                                 actual_flat_size)

        if self.temperatures is None or self.temperatures.numel(
        ) < required_flat_size:
            # Fresh tensors hold none of the recorded values.
            self.invalidate_sampling_params_cache()
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
            # zeros, not ones: min_p's neutral value is 0.0 where top_p's is 1.0.
            self.min_ps = torch.zeros(required_flat_size,
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
            self.request_min_ps = torch.zeros(self.max_num_requests,
                                              dtype=torch.float32,
                                              device='cuda')

        self._populate_request_rng_state(requests, per_request_normalized)

        # Always-populate the per-request slot id table when rejection sampling
        # is configured: it's tiny (max_num_requests longs) and needed at
        # draft-sampler time to scatter draft probs by slot. The penalties need the
        # same table to map logits rows back to their slot, so they enable it too.
        if (self.use_rejection_sampling
                or self.enable_penalty) and self.batch_slot_ids is not None:
            self.batch_slot_ids[:len(per_request_slot_ids)].copy_(
                torch.tensor(per_request_slot_ids,
                             dtype=torch.long,
                             pin_memory=prefer_pinned()),
                non_blocking=True,
            )

        # Penalties are independent of the greedy/advanced split -- they rewrite the
        # logits before sampling, so an all-greedy batch is penalized too (its argmax
        # is taken over the penalized logits). Filled before the early return below.
        self._populate_penalty_params(requests)

        # All-greedy: sampler takes the argmax branch (and rejection sampling
        # is also bypassed for all-greedy), so the per-token buffers are never
        # read. Skip the heavier H->D copies.
        if self.is_all_greedy_sample:
            return

        # Phase 2: build per-token / per-request lists and copy to GPU.
        #
        # Sampling params are fixed for a request's lifetime, so a steady-state
        # decode batch reproduces the buffers it already holds. Both the host
        # expansion and the copies sit on the critical path ahead of the
        # forward, so skip whichever group is already current.
        need_update_sampler_param, need_update_expanded_sampler_param = (
            self._sampling_params_buffers_need_update(per_request_normalized))
        # Only UNIVERSAL reads the min_p buffers. Under every other mode a min_p
        # request is rejected at admission, so every entry would be the 0.0 sentinel
        # the buffers were allocated with -- the list build and the H2D copy would
        # write zeros over zeros. This runs on the host critical path ahead of the
        # forward, so the existing modes must not pay for a filter they cannot use.
        fill_min_p = self.advanced_sampling_mode.is_universal
        if not (need_update_sampler_param
                or need_update_expanded_sampler_param):
            return

        if need_update_sampler_param:
            request_temperatures: list[float] = []
            request_top_ks: list[int] = []
            request_top_ps: list[float] = []
            request_min_ps: list[float] = []
            for temp_val, tk_val, tp_val, mp_val, _ in per_request_normalized:
                request_temperatures.append(temp_val)
                request_top_ks.append(tk_val)
                request_top_ps.append(tp_val)
                if fill_min_p:
                    request_min_ps.append(mp_val)

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
            if fill_min_p:
                self.request_min_ps[:len(request_min_ps)].copy_(
                    torch.tensor(request_min_ps,
                                 dtype=torch.float32,
                                 pin_memory=prefer_pinned()),
                    non_blocking=True,
                )

            # Pre-compute top_k_max on the CPU so CUDA-graph capture does not
            # encounter boolean-tensor indexing (dynamic size) or .item()
            # calls. DISABLE_TOPK_VAL (INT32_MAX) is the "top-k disabled"
            # sentinel. Derived from the same values as the per-request
            # buffers, so it is refreshed exactly when they are.
            _disable_topk = torch.iinfo(torch.int32).max
            self.top_k_max = max(
                (tk for tk in request_top_ks if 0 < tk < _disable_topk),
                default=0)

        if need_update_expanded_sampler_param:
            temperatures: list[float] = []
            top_ks: list[int] = []
            top_ps: list[float] = []
            min_ps: list[float] = []
            for temp_val, tk_val, tp_val, mp_val, num_tokens in per_request_normalized:
                temperatures.extend(temp_val for _ in range(num_tokens))
                top_ks.extend(tk_val for _ in range(num_tokens))
                top_ps.extend(tp_val for _ in range(num_tokens))
                if fill_min_p:
                    min_ps.extend(mp_val for _ in range(num_tokens))

            self.temperatures[:len(temperatures)].copy_(torch.tensor(
                temperatures, dtype=torch.float32, pin_memory=prefer_pinned()),
                                                        non_blocking=True)
            self.top_ks[:len(top_ks)].copy_(torch.tensor(
                top_ks, dtype=torch.int32, pin_memory=prefer_pinned()),
                                            non_blocking=True)
            self.top_ps[:len(top_ps)].copy_(torch.tensor(
                top_ps, dtype=torch.float32, pin_memory=prefer_pinned()),
                                            non_blocking=True)
            if fill_min_p:
                self.min_ps[:len(min_ps)].copy_(torch.tensor(
                    min_ps, dtype=torch.float32, pin_memory=prefer_pinned()),
                                                non_blocking=True)

    def _sampling_params_buffers_need_update(
        self, per_request_normalized: list[tuple[float, int, float, float, int]]
    ) -> tuple[bool, bool]:
        """Report which sampling-parameter buffers this step has to refill.

        Returns ``(need_update_sampler_param,
        need_update_expanded_sampler_param)`` for the per-request buffers and
        the expanded per-token buffers respectively, recording the new
        signatures as a side effect.

        Both signatures are built from ``per_request_normalized``, which every
        consumer reads by batch position -- so its order already encodes the
        batch ordering and a reshuffle changes the signature on its own. Slot
        ids are deliberately absent: they index ``batch_slot_ids`` (copied
        separately for the rejection path), never these buffers, so including
        them would only force refills when a slot changes hands between
        requests that happen to sample identically.

        The expanded buffers additionally depend on each request's token count,
        which sets their layout, so they can need a refill while the
        per-request buffers stay valid -- a context request becoming a
        generation request grows its span from one row to ``draft_len + 1`` and
        shifts every later request. Whenever the per-request buffers need an
        update the expanded ones do too.

        ``top_k_max`` derives from the same values as the per-request buffers,
        so it stays valid for as long as they do.
        """
        values = tuple(
            (temp, top_k, top_p, min_p)
            for temp, top_k, top_p, min_p, _ in per_request_normalized)
        num_tokens = tuple(n for *_, n in per_request_normalized)

        request_signature = values
        expanded_signature = (values, num_tokens)

        need_update_sampler_param = (self._sampling_params_signature[0]
                                     != request_signature)
        need_update_expanded_sampler_param = (self._sampling_params_signature[1]
                                              != expanded_signature)

        self._sampling_params_signature[0] = request_signature
        self._sampling_params_signature[1] = expanded_signature
        return need_update_sampler_param, need_update_expanded_sampler_param

    def invalidate_sampling_params_cache(self) -> None:
        """Force the next populate call to refill both buffer groups.

        Needed whenever the buffers stop reflecting the recorded signatures,
        e.g. after reallocating them.
        """
        self._sampling_params_signature[0] = None
        self._sampling_params_signature[1] = None


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
        # (sample_from_logits_op), so flashinfer>=0.6.4 is a hard
        # dependency here. Fail at construction with a clear error instead of
        # crashing mid-inference on the first non-greedy sampling step.
        if not IS_FLASHINFER_AVAILABLE or Version(
                flashinfer.__version__) < Version("0.6.4"):
            raise ImportError(
                "Speculative decoding requires flashinfer>=0.6.4, please install "
                "the version pinned in requirements.txt.")
        self.use_separate_draft_kv_cache = use_separate_draft_kv_cache
        # Static draft->target vocab offset map, cached once the draft model is
        # loaded (see set_draft_model). None when draft and target share a vocab.
        self._d2t: Optional[torch.Tensor] = None
        # Lazily-initialized state for the fractional synthetic acceptance
        # rate. The pool is a fixed-seed, rank-independent table of uniform
        # [0, 1) values; the counter is a device-side int64 advanced in-place
        # inside captured CUDA graphs (mirroring the existing flashinfer
        # seed/offset pattern in `_sample_tokens_for_batch`).
        self._force_accept_rng_pool: Optional[torch.Tensor] = None
        self._force_accept_rng_counter: Optional[torch.Tensor] = None

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if "forward" in cls.__dict__:
            raise TypeError(
                f"{cls.__name__} must not override SpecWorkerBase.forward; "
                f"implement _forward_impl instead. SpecWorkerBase.forward "
                f"guarantees spec-dec attn-metadata cleanup when a forward "
                f"fails (https://nvbugs/6442074).")

    def forward(self, *args, **kwargs):
        """Run _forward_impl with guaranteed spec-dec metadata cleanup.

        Tolerated forward failures (e.g. an OOM during the max-shape general
        warmup, or an error-budget-tolerated serving exception) must not leak
        the attn-metadata state saved by prepare_for_spec_dec: a stale save
        fails every subsequent forward at the pairing assert.
        https://nvbugs/6442074
        """
        attn_metadata = kwargs.get("attn_metadata")
        spec_metadata = kwargs.get("spec_metadata")
        if attn_metadata is None or spec_metadata is None:
            for a in args:
                if attn_metadata is None and isinstance(a, AttentionMetadata):
                    attn_metadata = a
                elif spec_metadata is None and isinstance(a, SpecMetadata):
                    spec_metadata = a
        try:
            return self._forward_impl(*args, **kwargs)
        finally:
            self._ensure_spec_dec_state_restored(attn_metadata, spec_metadata)

    @abstractmethod
    def _forward_impl(self, *args, **kwargs):
        """Worker-specific forward logic, called by SpecWorkerBase.forward."""

    def _ensure_spec_dec_state_restored(self, attn_metadata, spec_metadata):
        """Restore attn-metadata spec-dec state if a failure skipped it.

        No-op on the success path: workers restore at their preferred point
        and this sees no saved state. Subclasses with extra transient state
        (e.g. the deferred kv_lens rewind in PARD/DFlash) extend this.
        """
        if attn_metadata is not None and attn_metadata.has_spec_dec_saved_state:
            logger.warning(
                "Spec-dec worker forward failed between prepare_for_spec_dec "
                "and restore_from_spec_dec; restoring attn metadata state.")
            self._restore_attn_metadata_from_spec_dec(attn_metadata)

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

    def set_draft_model(self, draft_model) -> None:
        """Cache the static draft->target vocab offset map (``d2t``) once the
        draft model is loaded. ``d2t`` is a model-static parameter present only
        when the draft and target vocabularies differ; stays None otherwise.
        """
        self._d2t = getattr(getattr(draft_model, "model", None), "d2t", None)

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

    def _apply_occurrence_penalties(
            self, logits: torch.Tensor, draft_tokens: torch.Tensor,
            num_contexts: int, batch_size: int,
            spec_metadata: SpecMetadata) -> torch.Tensor:
        """Return the target logits acceptance should read, penalized.

        No-op unless the deploy enabled the penalties and some request in the batch
        actually uses one. ``logits`` must already be in the normalized
        ``[ctx (1 row), gen (draft_len + 1 rows)]`` layout, i.e. after
        ``_reshape_logits_for_accept`` -- which is what makes PARD's wider raw
        layout fit the same mapping.

        Returns the logits acceptance should read: a penalized copy when the
        penalties apply, otherwise the caller's tensor unchanged.
        """
        if not getattr(spec_metadata, "enable_penalty", False):
            return logits
        # NB: deliberately NOT gated on batch_uses_penalty. Decode steps replay a
        # captured CUDA graph, so a host-side skip decided at capture time would be
        # baked in permanently -- and capture happens during warmup, when no real
        # request is resident and the flag is False. The penalty pass must always be
        # captured; whether it changes anything is decided on device by
        # ``penalty_active``, which the replayed kernel re-reads every step.
        draft_len = draft_tokens.shape[1] if draft_tokens.dim() > 1 else 0
        mapping = penalty_ops.build_row_mapping(spec_metadata, num_contexts,
                                                batch_size, draft_len,
                                                draft_tokens, logits.device)
        if mapping is None:
            return logits
        row_slots, intra_tokens, intra_valid = mapping
        if row_slots.numel() != logits.shape[0]:
            # The caller's row layout is not the one this mapping describes (tree
            # modes); penalizing against it would charge the wrong request.
            return logits
        # Copy first: apply_penalties rewrites in place, and the caller keeps this
        # tensor as the step's reported logits. Penalizing it directly would feed
        # the penalized scores back to logprobs and to the next step's consumers.
        penalized = logits.clone()
        penalty_ops.apply_penalties(penalized, spec_metadata, row_slots,
                                    intra_tokens, intra_valid)
        return penalized

    def _accept_draft_tokens(self, logits, draft_tokens, num_contexts,
                             batch_size, spec_metadata):
        """
        Accept draft tokens with optional rejection sampling support.

        Mixed batches (num_contexts > 0) are supported: context rows take the
        first sampled target token via the base logic, and rejection sampling
        runs on the gen subset. Draft probs for the gen subset are gathered
        from the slot-indexed buffer by `py_seq_slot`.

        Occurrence penalties are applied to the target logits first, so both the
        strict and the rejection branch verify against the penalized distribution.
        They are applied to a copy: the caller keeps a reference to this tensor and
        returns it as the step's ``logits`` output (Eagle3's ``raw_logits``), which
        feeds logprobs and other consumers that must see the model's own scores.
        The draft distribution is deliberately left unpenalized: rejection sampling
        stays unbiased either way (it only requires draft_probs to match how the
        draft tokens were actually drawn), so this costs acceptance rate rather
        than correctness.
        """
        logits = self._apply_occurrence_penalties(logits, draft_tokens,
                                                  num_contexts, batch_size,
                                                  spec_metadata)
        num_gens = batch_size - num_contexts
        if num_gens > 0 and self._can_use_rejection_sampling(spec_metadata):
            draft_len = draft_tokens.shape[1]
            stored_vocab = (spec_metadata.draft_probs_last_dim
                            if spec_metadata.draft_probs_last_dim > 0 else
                            spec_metadata.draft_probs_vocab_size)
            # Fail closed: run the rejection kernel only when every buffer is
            # present and correctly shaped; otherwise fall back to strict
            # acceptance.
            if self._rejection_buffers_valid(draft_tokens, draft_len,
                                             stored_vocab, num_contexts,
                                             batch_size, logits, spec_metadata):
                # Gather the gen subset's slot rows, filled at the previous draft
                # step indexed by py_seq_slot.
                gen_slot_ids = spec_metadata.batch_slot_ids[
                    num_contexts:batch_size]
                draft_probs = spec_metadata.draft_probs[
                    gen_slot_ids, :draft_len, :stored_vocab]
                accepted = self._sample_and_accept_draft_tokens_rejection(
                    logits, draft_tokens, draft_probs, num_contexts, batch_size,
                    spec_metadata)
                return self._commit_occurrence_counts(accepted, batch_size,
                                                      spec_metadata)
        accepted = self._sample_and_accept_draft_tokens_base(
            logits, draft_tokens, num_contexts, batch_size, spec_metadata)
        return self._commit_occurrence_counts(accepted, batch_size,
                                              spec_metadata)

    def _commit_occurrence_counts(
            self, accepted: tuple[torch.Tensor, torch.Tensor], batch_size: int,
            spec_metadata: SpecMetadata) -> tuple[torch.Tensor, torch.Tensor]:
        """Record the tokens this step accepted, so later steps penalize them.

        Rejected speculative tokens never entered the sequence and are excluded by
        ``num_accepted_tokens``. Passes ``accepted`` straight through so callers can
        keep returning in one expression.
        """
        if not getattr(spec_metadata, "enable_penalty", False):
            return accepted
        accepted_tokens, num_accepted_tokens = accepted
        slot_ids = spec_metadata.batch_slot_ids
        if slot_ids is None:
            return accepted
        penalty_ops.update_penalty_counts(spec_metadata,
                                          slot_ids[:batch_size].to(torch.int64),
                                          accepted_tokens, num_accepted_tokens)
        return accepted

    def _draft_logits_are_sharded(self, logits, spec_metadata):
        """Whether the draft logits are vocab-sharded and need a TP gather.

        Sharded when tp_size>1 and the logits' last dim is narrower than the
        DRAFT head's own full vocab -- either plain TP (no attention DP) or the
        ADP + LM-head-TP mode, both of which produce vocab-sharded draft logits.
        Replicated full-vocab logits (borrowed/gathered LM head, plain attention
        DP, or a single rank) are not sharded.

        The reference width is the draft head's own full vocab (``draft_vocab_size``,
        falling back to ``vocab_size`` when unknown/shared), NOT the target
        ``vocab_size``: an Eagle3 reduced-vocab draft head produces full,
        replicated ``[tokens, draft_vocab_size]`` logits that are narrower than the
        target vocab; comparing against the target vocab would misclassify those as
        sharded and gather identical copies (overflowing the d2t table).
        """
        mapping = self.mapping
        if mapping is None or getattr(mapping, "tp_size", 1) <= 1:
            return False
        # Under attention DP every rank is data-parallel -- it owns a distinct
        # set of requests -- so the draft logits must NOT be cross-rank gathered,
        # whether they are replicated full-vocab (plain ADP) or vocab-sharded
        # (ADP + LM-head TP). A per-rank argmax on the rank's own logits is the
        # correct proposal (verified later); a gather would splice in a
        # mismatched collective across ranks that hold different token counts,
        # desyncing them into a hang / DeepEP launch failure. Only plain TP
        # (tp>1 without ADP), where all ranks share the same tokens sharded over
        # the vocab dim, needs the gather.
        if getattr(mapping, "enable_attention_dp", False):
            return False
        draft_full_vocab = (getattr(spec_metadata, "draft_vocab_size", 0)
                            or getattr(spec_metadata, "vocab_size", 0) or 0)
        return bool(draft_full_vocab) and logits.shape[-1] < draft_full_vocab

    def maybe_gather_sharded_draft_logits(self,
                                          logits,
                                          spec_metadata,
                                          mapping_lm_head_tp=None):
        """All-gather TP-sharded draft logits to full vocab before advanced sampling.

        Advanced (non-greedy) draft sampling needs the full-vocab distribution.
        Gathers shards only for a non-greedy batch when the logits are sharded
        (see ``_draft_logits_are_sharded``); replicated full-vocab logits are
        returned unchanged.

        Plain TP gathers vocab shards over ``self.mapping``. The LM-head-TP
        stacked/sharded layout never reaches this path: an advanced-sampling
        batch bypasses the LM-head-TP fast path in the worker and computes
        full-vocab logits locally from the (ADP-replicated) lm_head weight, so
        ``mapping_lm_head_tp`` is only ever passed alongside greedy sampling.
        """
        assert mapping_lm_head_tp is None, (
            "Advanced draft sampling must not receive LM-head-TP "
            "stacked/sharded logits; the worker bypasses the LM-head-TP fast "
            "path for non-all-greedy batches (see _forward_linear_draft_loop)")
        if (spec_metadata is None or spec_metadata.is_all_greedy_sample
                or not self._draft_logits_are_sharded(logits, spec_metadata)):
            return logits

        # Only plain TP (no attention DP) reaches here -- ADP variants return
        # early via _draft_logits_are_sharded, since their ranks are
        # data-parallel and must not gather draft logits across ranks.
        from ..distributed.ops import allgather
        return allgather(logits, self.mapping, dim=-1)

    def advanced_sample_draft(self,
                              logits: torch.Tensor,
                              spec_metadata: "SpecMetadata",
                              batch_size: int,
                              draft_step: Optional[int] = None):
        """Per-step advanced (non-greedy) draft sampler for step workers
        (MTP, DraftTarget).

        With rejection enabled and a ``draft_step``, samples via
        ``sampling_batch_spec_dec_one_model_for_rejection`` and scatters this
        step's proposal distribution into the slot-indexed ``draft_probs``
        buffer; otherwise uses ``sample_from_logits_op`` (tokens
        only). Returns tokens in draft-vocab space (the caller applies d2t).
        Expects 2D ``[batch_size, vocab]`` logits (one row per request).
        """
        temperatures = spec_metadata.request_temperatures[:batch_size]
        top_ks = spec_metadata.request_top_ks[:batch_size]
        top_ps = spec_metadata.request_top_ps[:batch_size]
        min_ps = spec_metadata.request_min_ps[:batch_size]

        # One row per request here, matching the request_* slices above.
        # Slot 0 of the step's offset window belongs to the target sampler, so
        # draft step i takes 1 + i. Callers that do not pass a draft_step run
        # this sampler once per step and take the first draft slot.
        seed, offset = self._rng_state_per_request(spec_metadata,
                                                   end=batch_size,
                                                   step_offset=1 +
                                                   (draft_step or 0))
        if spec_metadata.use_rejection_sampling and draft_step is not None:
            draft_tokens, probs = spec_sample_from_logits_with_probs(
                spec_metadata.advanced_sampling_mode,
                logits,
                temperatures,
                top_ks,
                top_ps,
                min_ps,
                seed=seed,
                offset=offset)
            # Scatter probs into the slot-indexed buffer so each request's data
            # lands at its stable py_seq_slot row regardless of batch shifts.
            assert spec_metadata.batch_slot_ids is not None, (
                "batch_slot_ids must be populated by "
                "populate_sampling_params_for_one_model before draft probs "
                "storage")
            batch_slots = spec_metadata.batch_slot_ids[:batch_size]
            vocab = probs.shape[-1]
            spec_metadata.draft_probs[batch_slots, draft_step, :vocab] = probs
            spec_metadata.draft_probs_last_dim = vocab
        else:
            draft_tokens = spec_sample_from_logits(
                spec_metadata.advanced_sampling_mode,
                logits,
                temperatures,
                top_ks,
                top_ps,
                min_ps,
                seed=seed,
                offset=offset)

        return draft_tokens.type(torch.int32)

    def _reshape_draft_tokens_for_accept(self, spec_metadata, num_gens, device):
        """Reshape the stored draft tokens to ``[num_gens, runtime_draft_len]``
        for acceptance. Default assumes one draft token per step (DraftTarget,
        DFlash); workers with a different buffer layout (e.g. PARD's 2K-1
        entries) override this.
        """
        runtime_draft_len = spec_metadata.runtime_draft_len
        if spec_metadata.draft_tokens is None:
            return torch.zeros((num_gens, runtime_draft_len),
                               dtype=torch.int,
                               device=device)
        return spec_metadata.draft_tokens.reshape(num_gens, runtime_draft_len)

    def _reshape_logits_for_accept(self, logits, num_contexts, num_gens,
                                   spec_metadata):
        """Reshape target logits to the ``[num_contexts + num_gens*(K+1), vocab]``
        layout expected by acceptance. Default is identity (target already emits
        one logit per accepted position); workers that emit extra positions
        (e.g. PARD's 2K per gen request) override this.
        """
        return logits

    def sample_and_accept_draft_tokens(self, logits, attn_metadata,
                                       spec_metadata):
        """Sample the golden token and verify previously proposed draft tokens.

        Default implementation for one-model workers whose acceptance differs
        only in how draft tokens / target logits are reshaped (DraftTarget,
        PARD, DFlash): unpack batch sizes, reshape via the overridable hooks,
        then route through ``_accept_draft_tokens`` (strict or rejection). Workers
        with a materially different acceptance path (MTP, Eagle3: relaxed /
        THOP / extra ``input_ids``) override this method entirely.
        """
        batch_size = attn_metadata.num_seqs
        num_contexts = attn_metadata.num_contexts
        num_gens = batch_size - num_contexts
        draft_tokens = self._reshape_draft_tokens_for_accept(
            spec_metadata, num_gens, logits.device)
        logits = self._reshape_logits_for_accept(logits, num_contexts, num_gens,
                                                 spec_metadata)
        return self._accept_draft_tokens(logits, draft_tokens, num_contexts,
                                         batch_size, spec_metadata)

    def _rejection_buffers_valid(self, draft_tokens, draft_len, stored_vocab,
                                 num_contexts, batch_size, logits,
                                 spec_metadata) -> bool:
        """Fail-closed guard: return True only when the slot-indexed draft-prob
        buffers exist and every shape the rejection path dereferences is valid;
        otherwise the caller falls back to strict acceptance. Inspects only
        host-side tensor shapes -- no ``.item()`` / value read on CUDA tensors --
        so it stays CUDA-graph-capture safe.
        """
        draft_probs = spec_metadata.draft_probs
        batch_slot_ids = spec_metadata.batch_slot_ids
        if draft_probs is None or batch_slot_ids is None:
            return False
        if stored_vocab <= 0:
            return False
        num_gens = batch_size - num_contexts
        # draft_probs must cover the slice [:, :draft_len, :stored_vocab].
        if draft_probs.dim() != 3:
            return False
        if draft_probs.shape[1] < draft_len or draft_probs.shape[
                2] < stored_vocab:
            return False
        if draft_tokens.dim() != 2 or draft_tokens.shape[0] != num_gens:
            return False
        # logits must cover context rows (1 each) + gen rows (draft_len + 1 each).
        logits_rows = logits.shape[0] if logits.dim() > 1 else 1
        if logits_rows < num_contexts + num_gens * (draft_len + 1):
            return False
        # Slot ids for the gen subset must exist (range safety is guaranteed by
        # construction).
        if batch_slot_ids.shape[0] < batch_size:
            return False
        if batch_slot_ids[num_contexts:batch_size].shape[0] != num_gens:
            return False
        return True

    def _can_use_rejection_sampling(self, spec_metadata: SpecMetadata) -> bool:
        # Skip rejection sampling when the whole batch is greedy: the accepted
        # result is identical to argmax and the base path is cheaper. Mixed
        # batches (context + gen) are handled via slot-indexed draft probs and
        # are split inside _sample_and_accept_draft_tokens_rejection.
        return (spec_metadata.use_rejection_sampling
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
            # The target distribution the acceptance test divides by. It must be
            # filtered exactly as the draft probs were (see advanced_sample_draft),
            # which is why both go through the same dispatcher on the same mode --
            # a mismatch here corrupts acceptance silently rather than raising.
            target_probs_flat = spec_compute_probs_from_logits(
                spec_metadata.advanced_sampling_mode, gen_logits, temperatures,
                spec_metadata.top_ks[gen_start:gen_end],
                spec_metadata.top_ps[gen_start:gen_end],
                spec_metadata.min_ps[gen_start:gen_end])
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
            d2t = self._d2t.data if self._d2t is not None else None
            if draft_vocab_size != vocab_size:
                if spec_metadata.full_draft_probs is not None:
                    # Slice to runtime_draft_len so the max_draft_len buffer
                    # never passes stale extra rows to the rejection kernel.
                    full_draft_probs = spec_metadata.full_draft_probs[:
                                                                      num_gens, :
                                                                      runtime_draft_len]
                else:
                    # Buffer not pre-allocated (e.g. rejection off at prepare()):
                    # fall back to a per-iter allocation.
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

            # One entry per gen request; slot 0 of the step's offset window.
            seed, offset = self._rng_state_per_request(spec_metadata,
                                                       num_contexts, batch_size)

            gen_accepted, gen_num_accepted = rejection_sampling_one_model(
                draft_probs=full_draft_probs,
                draft_token_ids=full_draft_tokens,
                target_probs=target_probs,
                deterministic=True,
                seed=seed,
                offset=offset,
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

    def _rng_state_per_request(
        self,
        spec_metadata: SpecMetadata,
        start: int = 0,
        end: Optional[int] = None,
        repeat: int = 1,
        step_offset: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Philox (seed, offset) laid out one entry per request row.

        ``start`` / ``end`` select the request subset the caller samples (e.g.
        the gen slice); ``repeat`` expands each request to the ``K`` rows a
        block sampler flattens it into.

        ``step_offset`` picks a slot inside this decoding step's offset window
        (see ``_populate_request_rng_state``). The target sampler and the
        rejection kernel leave it at 0; the draft loop passes ``1 +
        draft_step`` so each of its launches draws a distinct stream -- with a
        fixed user seed the offset is the only thing separating them, since
        every draft launch restarts the kernel's per-row subsequence at 0.

        """
        seeds = spec_metadata.request_seeds[start:end]
        offsets = spec_metadata.request_offsets[start:end]
        if step_offset:
            offsets = offsets + step_offset
        if repeat > 1:
            seeds = seeds.repeat_interleave(repeat)
            offsets = offsets.repeat_interleave(repeat)
        return seeds, offsets

    def _rng_state_per_token(
        self,
        spec_metadata: SpecMetadata,
        num_tokens: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Philox (seed, offset) laid out one entry per logits row.

        Mirrors how ``temperatures`` / ``top_ks`` / ``top_ps`` are sliced at
        the same call sites.
        """
        return spec_metadata.seeds[:
                                   num_tokens], spec_metadata.offsets[:
                                                                      num_tokens]

    def _draft_sampler_greedy(self, logits: torch.Tensor):
        """
        Simple greedy draft token sampling using argmax.

        Args:
            logits: [num_tokens, vocab_size] - Draft model logits

        Returns:
            draft_tokens: [num_tokens] - Sampled draft token ids (int32)
        """
        draft_tokens = greedy_search_sampling_batch(logits,
                                                    return_probs=False)[0]

        # Apply the cached draft->target vocab offset map.
        if self._d2t is not None:
            draft_tokens = self._d2t[draft_tokens] + draft_tokens

        return draft_tokens.type(torch.int32)

    def _get_local_max_and_combined(self, logits, mapping_lm_tp=None):
        """Pack each rank's local (global_argmax_index, max_value) for a
        distributed argmax over a vocab-sharded draft LM head.
        """
        local_max_values, local_argmax = torch.max(logits, dim=-1, keepdim=True)
        vocab_per_rank = logits.shape[-1]
        mapping_lm_tp = mapping_lm_tp if mapping_lm_tp is not None else self.mapping
        max_index_per_rank = local_argmax.type(
            torch.int32) + (mapping_lm_tp.tp_rank * vocab_per_rank)
        max_index_per_rank_float = max_index_per_rank.float()
        local_max_values_float32 = local_max_values.float()
        # Interleaved layout: [idx0, val0, idx1, val1, ...] after all-gather.
        combined = torch.stack(
            [max_index_per_rank_float, local_max_values_float32],
            dim=-1).flatten(-2)
        return combined

    @torch.compile(options={"max-autotune": True})
    def _get_draft_tokens_from_gathered(self, gathered):
        """Pick the global-argmax token id from the all-gathered per-rank
        (index, value) pairs produced by ``_get_local_max_and_combined``.
        """
        gathered_indices_float = gathered[..., 0::2]
        gathered_values_float = gathered[..., 1::2]
        max_indices = torch.argmax(gathered_values_float, dim=-1, keepdim=True)
        draft_tokens = torch.gather(gathered_indices_float, -1,
                                    max_indices).squeeze(-1).type(torch.int32)
        return draft_tokens

    def greedy_sample_draft_with_tp_gather(self,
                                           logits: torch.Tensor,
                                           spec_metadata=None,
                                           mapping_lm_head_tp=None):
        """Greedy draft-token sampling with a TP all-gather of the argmax.

        When the draft LM head is vocab-sharded under plain tensor parallelism,
        a per-rank argmax disagrees across ranks and desyncs speculative
        decoding. Gather only each rank's local (index, value) and pick the
        global argmax. Falls back to plain argmax when the logits are not
        vocab-sharded (see ``_draft_logits_are_sharded``) -- e.g. a borrowed or
        gathered full-vocab draft head. Returns tokens in draft-vocab space (the
        caller applies d2t). Expects 2D ``[num_tokens, vocab_shard]`` logits.

        Under ADP + LM-head TP (``mapping_lm_head_tp`` given) the logits are the
        LM-head-TP group's row-stacked batch (``tp_size`` segments of
        ``max_num_requests`` padded rows, all-gathered along dim 0 by the MTP
        shared head) with the vocab sharded across the group. The global argmax
        must combine the group's vocab shards, and each rank must read its own
        row segment at offset ``tp_rank * max_num_requests`` -- NOT rows
        ``[:batch]``, which belong to group rank 0.
        """
        if (mapping_lm_head_tp is not None
                and getattr(mapping_lm_head_tp, "tp_size", 1) > 1):
            from ..distributed.ops import allgather
            combined = self._get_local_max_and_combined(logits,
                                                        mapping_lm_head_tp)
            gathered = allgather(combined, mapping_lm_head_tp, dim=-1)
            group_size = mapping_lm_head_tp.tp_size
            local_rows = logits.shape[0] // group_size
            own_segment = gathered.view(group_size, local_rows,
                                        -1)[mapping_lm_head_tp.tp_rank]
            return self._get_draft_tokens_from_gathered(own_segment)
        mapping = self.mapping
        sharded = self._draft_logits_are_sharded(logits, spec_metadata)
        if (sharded and mapping is not None
                and getattr(mapping, "tp_size", 1) > 1
                and not mapping.enable_attention_dp):
            from ..distributed.ops import allgather
            combined = self._get_local_max_and_combined(logits)
            gathered = allgather(combined, mapping, dim=-1)
            return self._get_draft_tokens_from_gathered(gathered)
        # No cross-rank gather for plain attention-DP: each rank owns its own
        # requests with replicated full-vocab logits, so a per-rank argmax is
        # the correct proposal and a gather would desync the ranks (see
        # _draft_logits_are_sharded). Plain argmax; caller applies d2t.
        return torch.argmax(logits, dim=-1).type(torch.int32)

    def advanced_sample_draft_block(self, gen_logits: torch.Tensor,
                                    spec_metadata: "SpecMetadata",
                                    num_contexts: int, batch_size: int):
        """Block counterpart of ``advanced_sample_draft`` for gen-only workers
        (PARD, DFLASH): produces all ``K`` draft positions per gen request in one
        forward from ``[num_gens, K, vocab]`` logits.

        With rejection enabled, samples via
        ``sampling_batch_spec_dec_one_model_for_rejection`` and scatters the K
        proposal rows into ``draft_probs[gen_slot_ids, 0:K, :]``; otherwise uses
        ``sample_from_logits_op`` (tokens only). Only called for a
        non-greedy batch (the all-greedy path is handled by the caller). Returns
        ``[num_gens, K]`` int32 tokens in draft-vocab space (the caller applies
        d2t); stored probs likewise stay in draft-vocab space.
        """
        num_gens, K, vocab = gen_logits.shape
        if num_gens == 0:
            return torch.empty((0, K),
                               dtype=torch.int32,
                               device=gen_logits.device)

        # Take the gen slice and repeat each request's value K times to line up
        # with the flattened [num_gens*K, vocab] logits (K rows per request).
        temps = spec_metadata.request_temperatures[
            num_contexts:batch_size].repeat_interleave(K)
        top_ks = spec_metadata.request_top_ks[
            num_contexts:batch_size].repeat_interleave(K)
        top_ps = spec_metadata.request_top_ps[
            num_contexts:batch_size].repeat_interleave(K)
        min_ps = spec_metadata.request_min_ps[
            num_contexts:batch_size].repeat_interleave(K)

        flat_logits = gen_logits.reshape(num_gens * K, vocab)
        # A block sampler emits all K draft positions in one launch, so the
        # kernel's per-row subsequence already separates them and they share
        # the first draft slot of the step's offset window.
        seed, offset = self._rng_state_per_request(spec_metadata,
                                                   num_contexts,
                                                   batch_size,
                                                   repeat=K,
                                                   step_offset=1)

        if getattr(spec_metadata, "use_rejection_sampling", False):
            flat_tokens, flat_probs = spec_sample_from_logits_with_probs(
                spec_metadata.advanced_sampling_mode,
                flat_logits,
                temps,
                top_ks,
                top_ps,
                min_ps,
                seed=seed,
                offset=offset)
            # Scatter the K prob rows per gen request into its stable slot row.
            if spec_metadata.draft_probs is not None:
                assert spec_metadata.batch_slot_ids is not None, (
                    "batch_slot_ids must be populated before block draft prob "
                    "storage")
                gen_slot_ids = spec_metadata.batch_slot_ids[
                    num_contexts:batch_size]
                probs = flat_probs.reshape(num_gens, K, vocab)
                spec_metadata.draft_probs[gen_slot_ids, :K, :vocab] = probs
                spec_metadata.draft_probs_last_dim = vocab
        else:
            flat_tokens = spec_sample_from_logits(
                spec_metadata.advanced_sampling_mode,
                flat_logits,
                temps,
                top_ks,
                top_ps,
                min_ps,
                seed=seed,
                offset=offset)

        return flat_tokens.reshape(num_gens, K).type(torch.int32)

    def write_context_onehot_draft_probs(self,
                                         spec_metadata,
                                         num_contexts,
                                         num_gens,
                                         draft_len,
                                         gen_vocab=None):
        """Write a one-hot draft-prob distribution (prob 1.0 at draft-vocab token
        id 0, the placeholder draft token) into each context request's stable
        slot row, so the row is a legal distribution when the context request
        becomes a generation request next iteration and rejection acceptance
        reads ``draft_probs[slot, :draft_len, :stored_vocab]``.

        Block workers (PARD/DFlash) do not draft context requests (they get a
        zero placeholder token), leaving their slot rows unwritten. Mixed
        iterations reuse the vocab width the gen scatter just set (``gen_vocab``);
        pure-context iterations have no gen logits, so use the buffer's full
        draft-vocab width and publish it via ``draft_probs_last_dim`` (acceptance
        next iter reads this scalar). No-op unless rejection is enabled and the
        slot-indexed buffers exist. Static shapes -> CUDA-graph safe.
        """
        if (num_contexts <= 0
                or not getattr(spec_metadata, "use_rejection_sampling", False)
                or spec_metadata.draft_probs is None):
            return
        ctx_slot_ids = spec_metadata.batch_slot_ids[:num_contexts]
        if num_gens > 0:
            onehot_vocab = gen_vocab  # matches the gen scatter's width
        else:
            onehot_vocab = spec_metadata.draft_probs_vocab_size
            spec_metadata.draft_probs_last_dim = onehot_vocab
        spec_metadata.draft_probs[ctx_slot_ids, :draft_len, :onehot_vocab] = 0.0
        spec_metadata.draft_probs[ctx_slot_ids, :draft_len, 0] = 1.0

    def sample_draft_tokens(self,
                            logits,
                            spec_metadata,
                            batch_size,
                            *,
                            num_contexts=0,
                            draft_step=None,
                            mapping_lm_head_tp=None):
        """Unified draft-token production entry for all one-model workers.

        Branches by logits rank: 3D ``[num_gens, K, vocab]`` is the block form
        (gen-only workers emitting all K positions in one forward, e.g.
        PARD/DFlash); 2D ``[num_tokens, vocab]`` is the per-step form
        (autoregressive workers called once per draft step, e.g. MTP/
        DraftTarget). ``draft_step`` applies only to the step form and
        ``num_contexts`` only to the block form. d2t is read from ``self._d2t``.

        In a mixed (context + generation) batch the block form receives
        gen-only logits (context requests draft from accepted tokens they do not
        have yet), so ``num_contexts`` slices the full-batch per-request metadata
        down to the gen segment. The step form receives full-batch logits
        (context requests draft from target hidden states already available), so
        no slicing is needed.
        """
        is_block = logits.dim() == 3

        # Draft tokens use argmax unless rejection sampling is engaged for a
        # non-greedy batch. Rejection sampling is the only path that needs the
        # draft's stochastic proposal distribution (stored in draft_probs); every
        # other path (all-greedy, or non-greedy with strict/exact-match
        # acceptance) accepts a draft token only when it equals the target's
        # choice, and there argmax maximizes the acceptance rate (E[accept] =
        # max_i p_i >= sum_i p_i^2 = E[accept] for a stochastic draft). This
        # matches sglang/vLLM, which draft with argmax/top-k by default and apply
        # sampling params only on the target/acceptance side.
        advanced = spec_metadata.wants_advanced_draft_sampling

        # All samplers below return tokens in draft-vocab space; d2t is applied
        # once after the branch.
        if not advanced:
            # greedy_sample_draft_with_tp_gather expects 2D [tokens, vocab];
            # flatten the block form and restore its shape (step form is 2D).
            batch_shape = logits.shape[:-1]
            tokens = self.greedy_sample_draft_with_tp_gather(
                logits.reshape(-1, logits.shape[-1]), spec_metadata,
                mapping_lm_head_tp)
            if mapping_lm_head_tp is None:
                tokens = tokens.reshape(batch_shape)
            # else: ADP+LM-head-TP (2D step form only) -- the sampler returned
            # this rank's own row segment, 1/tp_size of the stacked input rows,
            # so the input batch shape no longer applies. Keep as-is; the
            # caller trims the max_num_requests padding to token_count.
        else:
            # Advanced sampling gathers the vocab-sharded draft logits to full
            # vocab, then samples (scattering this step's proposal distribution
            # into draft_probs).
            logits = self.maybe_gather_sharded_draft_logits(
                logits, spec_metadata, mapping_lm_head_tp)
            if is_block:
                tokens = self.advanced_sample_draft_block(
                    logits, spec_metadata, num_contexts, batch_size).long()
            else:
                tokens = self.advanced_sample_draft(logits,
                                                    spec_metadata,
                                                    batch_size,
                                                    draft_step=draft_step)

        # Map draft-vocab token ids to target vocab (no-op for shared vocab).
        if self._d2t is not None:
            tokens = self._d2t[tokens] + tokens

        return tokens.type(torch.int32)

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
        Select draft attention metadata for one-engine speculative decoding.

        TRTLLM metadata temporarily swaps its manager and cache-layout-dependent
        buffers, including DSA indexer offsets and slot mappings.
        FlashInfer uses an independently planned metadata view because its page
        tables and kernel wrappers are manager-specific.
        """

        # draft_kv_cache_manager is None if using two-engine speculative decoding or not enabling separate draft KV cache.
        if draft_kv_cache_manager is None:
            yield attn_metadata
            return

        from ..attention_backend.flashinfer import FlashInferAttentionMetadata
        if isinstance(attn_metadata, FlashInferAttentionMetadata):
            yield attn_metadata.get_draft_metadata(draft_kv_cache_manager)
            return

        if not isinstance(attn_metadata, TrtllmAttentionMetadata):
            yield attn_metadata
            return

        saved_state = prepare_attn_metadata_for_draft_replay(
            attn_metadata, draft_kv_cache_manager)
        if saved_state is None:
            yield attn_metadata
            return

        try:
            yield attn_metadata
        finally:
            restore_attn_metadata_after_draft_replay(attn_metadata, saved_state)

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
            # in sample_from_logits_op (which is torch.compiled).
            num_tokens = logits.shape[0]

            temperatures = spec_metadata.temperatures[:num_tokens]
            top_ks = spec_metadata.top_ks[:num_tokens]
            top_ps = spec_metadata.top_ps[:num_tokens]
            min_ps = spec_metadata.min_ps[:num_tokens]

            # One row per logits row here, the same slice the per-token
            # sampling params above use.
            seed, offset = self._rng_state_per_token(spec_metadata, num_tokens)
            sampled_tokens = spec_sample_from_logits(
                spec_metadata.advanced_sampling_mode,
                logits,
                temperatures,
                top_ks,
                top_ps,
                min_ps,
                seed=seed,
                offset=offset)
        else:
            sampled_tokens = torch.argmax(logits, dim=-1)

        return sampled_tokens
