from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Set

import torch
import torch.nn.functional as F
from torch import nn

from tensorrt_llm._utils import prefer_pinned
from tensorrt_llm.mapping import Mapping

from ..attention_backend import AttentionMetadata
from ..distributed.ops import allgather
from ..model_config import ModelConfig
from ..pyexecutor.llm_request import LlmRequest
from ..pyexecutor.mamba_cache_manager import MambaHybridCacheManager
from ..pyexecutor.resource_manager import BaseResourceManager, SlotManager
from ..pyexecutor.sampler import TorchSampler
from ..pyexecutor.scheduler import ScheduledRequests
from .interface import SpecMetadata, SpecWorkerBase
from .mtp import MTPSampler
from .sa_enhancer import SADraftEnhancer
from .spec_tree_manager import SpecTreeManager

if TYPE_CHECKING:
    from ...llmapi.llm_args import EagleDecodingConfig


class Eagle3ResourceManager(BaseResourceManager):
    """
    Eagle3 needs to save the hidden states for the draft model. When using
    Eagle3TwoModel, there will be two model engines, one for the target model
    and one for the draft model. Use this class to manage the hidden states.
    """

    def __init__(self,
                 config: "EagleDecodingConfig",
                 dtype: torch.dtype,
                 hidden_size: int,
                 max_num_requests: int,
                 max_seq_len: int,
                 max_num_tokens: int,
                 sa_manager=None):
        self.dtype = dtype
        self.max_draft_len = config.max_draft_len
        self.hidden_size = hidden_size
        self.max_num_requests = max_num_requests
        self.max_seq_len = max_seq_len
        # Optional SA manager for EAGLE3+SA mode
        self.sa_manager = sa_manager
        # There could be dummy request for padding batch when using CUDA graph.
        # Reserve one more slot for the dummy request.
        slot_size = self.max_seq_len + 1
        self.slot_manager = SlotManager(slot_size)
        # This class is reused by MTP_EAGLE
        from ...llmapi.llm_args import EagleDecodingConfig

        if isinstance(config, EagleDecodingConfig):
            self.max_total_draft_tokens = config.tokens_per_gen_step - 1
        else:
            self.max_total_draft_tokens = self.max_draft_len

        # empty hidden states tensor
        max_num_tokens = min(max_num_tokens, max_num_requests *
                             self.max_seq_len) + (self.max_total_draft_tokens +
                                                  1) * max_num_requests

        num_capture_layers = getattr(config, 'num_capture_layers', 1)
        self.hidden_states = torch.empty(
            (max_num_tokens, self.hidden_size * num_capture_layers),
            dtype=self.dtype,
            device='cuda')
        # sequence length, only used for metadata preparation
        self.seq_lens = {i: 0 for i in range(slot_size)}

        self.use_relaxed_acceptance_for_thinking = getattr(
            config, 'use_relaxed_acceptance_for_thinking', False)
        if self.use_relaxed_acceptance_for_thinking:
            # Per-request delta pool tracking whether the request is in the
            # thinking phase; mirrors MTPHiddenStatesManager.mtp_relaxed_delta_pool.
            self.relaxed_delta_pool = torch.zeros((slot_size, ),
                                                  dtype=torch.float,
                                                  device='cuda')
        # start indices of each slot
        self.start_indices = {i: 0 for i in range(slot_size)}
        # whether the next draft forward is the first
        self.is_first_draft = True
        self.spec_tree_manager = None

        if isinstance(config,
                      EagleDecodingConfig) and config.eagle_choices is not None:
            self.spec_tree_manager = SpecTreeManager(
                max_num_requests=self.max_num_requests,
                use_dynamic_tree=config.use_dynamic_tree,
                max_draft_len=self.max_draft_len,
                max_total_draft_tokens=self.max_total_draft_tokens,
                eagle_choices=config.eagle_choices,
                dynamic_tree_max_topK=config.dynamic_tree_max_topK,
            )

    def prepare_resources(self, scheduled_batch: ScheduledRequests):
        context_batch = scheduled_batch.context_requests
        # allocate hidden state tensors and update slot ids
        self.slot_ids = []
        for req in context_batch:
            if req.is_first_context_chunk:
                slot_id = self.slot_manager.add_slot(req.request_id)
                self.slot_ids.append(slot_id)
                if self.use_relaxed_acceptance_for_thinking:
                    self.relaxed_delta_pool[slot_id].fill_(0)
        # reset the flag before model forward
        self.is_first_draft = True

    def update_resources(self, scheduled_batch: ScheduledRequests):
        pass

    def free_resources(self, request: LlmRequest):
        slot_id = self.slot_manager.get_slot(request.request_id)
        self.seq_lens[slot_id] = 0
        self.start_indices[slot_id] = 0
        if self.use_relaxed_acceptance_for_thinking:
            self.relaxed_delta_pool[slot_id].fill_(0)
        self.slot_manager.remove_slot(request.request_id)
        if self.sa_manager is not None:
            self.sa_manager.remove_request(request.request_id)

    def add_dummy_requests(self, request_ids: List[int]):
        for rid in request_ids:
            self.slot_manager.add_slot(rid)
        if self.sa_manager is not None:
            self.sa_manager.add_dummy_requests(request_ids)

    def shutdown(self):
        if self.sa_manager is not None:
            self.sa_manager.shutdown()

    def get_max_resource_count(self) -> int:
        return self.max_num_requests

    def get_needed_resource_to_completion(self, request: LlmRequest):
        return 0


def _get_eagle3_default_capture_layers(num_layers: int):
    return (1, num_layers // 2 - 1, num_layers - 4)


@dataclass
class Eagle3SpecMetadata(SpecMetadata):
    hidden_states: List[torch.Tensor] = field(default_factory=list)
    layers_to_capture: Optional[Set[int]] = None
    target_model_embed_tokens: Optional[torch.nn.Module] = None
    hidden_size: int = 0
    max_num_tokens: int = 0
    dtype: torch.dtype = torch.bfloat16
    is_draft_model: bool = False
    is_first_draft: bool = False
    eagle3_resource_manager: Optional[Eagle3ResourceManager] = None
    is_mtp_eagle: bool = False

    eagle_choices: Optional[List[List[int]]] = None
    max_total_draft_tokens: int = 0
    # This is to store the request type and accepted path for each request.
    # For each request, {key: request_ids, value: accepted_path}
    # 'accepted_path' is a list of accepted tokens indices.
    request_accepted_path: Optional[Dict[int, List[int]]] = None

    def __post_init__(self):
        if self.is_draft_model:
            self.layers_to_capture = (self.num_layers - 1, )
        elif self.layers_to_capture is None:
            if self.num_layers == 1 or self.is_mtp_eagle:
                self.layers_to_capture = (-1, )
            else:
                if self.num_layers <= 5:
                    raise ValueError(
                        "Not enough hidden layers for default EAGLE3 capture")
                self.layers_to_capture = _get_eagle3_default_capture_layers(
                    self.num_layers)
        else:
            self.layers_to_capture = sorted(list(self.layers_to_capture))
            if self.layers_to_capture[0] == -1:
                self.layers_to_capture = self.layers_to_capture[1:] + [
                    self.layers_to_capture.pop(0)
                ]
        self.num_capture_layers = len(self.layers_to_capture)

        # Initialize to 0 to avoid reading uninitialized memory during warmup
        self.hidden_states_read_indices = torch.zeros([self.max_num_tokens],
                                                      dtype=torch.long,
                                                      device='cuda')
        self.hidden_states_write_indices = torch.zeros([self.max_num_tokens],
                                                       dtype=torch.long,
                                                       device='cuda')
        self.hidden_states_read_indices_host = None
        self.hidden_states_write_indices_host = None

        if self.eagle_choices is not None:
            self.is_spec_dec_tree = True
            self.is_spec_dec_dynamic_tree = False

    def prepare(self):
        is_first_draft = self.eagle3_resource_manager.is_first_draft
        spec_tree_manager = self.eagle3_resource_manager.spec_tree_manager
        # Update start indices
        # Here, we assume the sequence lengths (seq_lens) during the draft model
        # forward will not exceed those of the target model. So pre-allocate
        # hidden state space before the target model forward.
        start_idx = 0
        if not self.is_draft_model:
            for req_id, seq_len in zip(self.request_ids, self.seq_lens):
                slot_id = self.eagle3_resource_manager.slot_manager.get_slot(
                    req_id)
                self.eagle3_resource_manager.start_indices[slot_id] = start_idx
                # Make sure that the space between two requests is at least max_total_draft_tokens + 1.
                start_idx += max(seq_len, self.max_total_draft_tokens + 1)
                assert start_idx < self.eagle3_resource_manager.hidden_states.shape[
                    0], f"start_idx {start_idx} is greater than hidden_states.shape[0] {self.eagle3_resource_manager.hidden_states.shape[0]}"

        # Prepare hidden states gather ids
        hidden_states_read_indices = []
        hidden_states_write_indices = []
        for req_id, seq_len in zip(self.request_ids, self.seq_lens):
            slot_id = self.eagle3_resource_manager.slot_manager.get_slot(req_id)
            start_idx = self.eagle3_resource_manager.start_indices[slot_id]
            # 1) target model or (is_first_draft and is_linear_tree)
            # If this is the first draft or the target model forward, we need to
            # read/write all of the hidden states
            if not self.is_draft_model or (is_first_draft
                                           and spec_tree_manager is None):
                hidden_states_read_indices.extend(
                    list(range(start_idx, start_idx + seq_len)))
                hidden_states_write_indices.extend(
                    list(range(start_idx, start_idx + seq_len)))
            # 2）is_first_draft and draft_token_tree
            # After target model forward, some draft tokens will be accepted.
            # These draft tokens' hidden states will be used for draft model's first drafter layer.
            elif is_first_draft and spec_tree_manager is not None:
                assert req_id in self.request_accepted_path.keys(
                ), f"Request {req_id} not found in request_accepted_path"
                # 'node_idx + 1' is because we '-1' in sampler.py for kv cache rewind. Now we add it back.
                accepted_path = [
                    node_idx + 1
                    for node_idx in self.request_accepted_path[req_id]
                ]

                if accepted_path == []:
                    # Case 1: This is a context request, We need to read all the hidden states.
                    # Case 2: This is a generation request, but no accepted tokens are accepted. Actually only the first token's hidden states is needed. The others are just padding tokens.
                    hidden_states_read_indices.extend(
                        list(range(start_idx, start_idx + seq_len)))
                else:
                    # This is a generation request. And there are draft tokens accepted.
                    # We only read the accepted tokens' hidden states.
                    accepted_path = [0] + accepted_path  # add the root node
                    accepted_path_pad = accepted_path + [0] * (
                        seq_len - len(accepted_path))
                    assert len(accepted_path_pad) == seq_len
                    hidden_states_read_indices.extend([
                        start_idx + accepted_draft_token_offset
                        for accepted_draft_token_offset in accepted_path_pad
                    ])

                # For the write indices, we just write all the hidden states.
                hidden_states_write_indices.extend(
                    list(range(start_idx, start_idx + seq_len)))
            # otherwise: only read the last token
            else:
                old_seq_len = self.eagle3_resource_manager.seq_lens[slot_id]
                hidden_states_read_indices.append(start_idx + old_seq_len - 1)
                hidden_states_write_indices.append(start_idx + seq_len - 1)
            self.eagle3_resource_manager.seq_lens[slot_id] = seq_len
        # Prepare hidden states gather ids
        self.hidden_states_read_indices_host = torch.tensor(
            hidden_states_read_indices,
            dtype=torch.long,
            pin_memory=prefer_pinned())
        self.hidden_states_write_indices_host = torch.tensor(
            hidden_states_write_indices,
            dtype=torch.long,
            pin_memory=prefer_pinned())
        self.is_first_draft = is_first_draft and self.is_draft_model
        if self.is_draft_model:
            self.eagle3_resource_manager.is_first_draft = False

        self.hidden_states_read_indices[:self.num_tokens].copy_(
            self.hidden_states_read_indices_host, non_blocking=True)
        self.hidden_states_write_indices[:self.num_tokens].copy_(
            self.hidden_states_write_indices_host, non_blocking=True)

    def is_layer_capture(self, layer_id: int):
        return layer_id in self.layers_to_capture

    def maybe_capture_hidden_states(
            self,
            layer_id: int,
            hidden_states: torch.Tensor,
            residual: Optional[torch.Tensor] = None) -> None:
        token_idx = self.hidden_states_write_indices[:self.num_tokens]
        eagle3_hidden_states = self.eagle3_resource_manager.hidden_states
        for i, captured_layer_id in enumerate(self.layers_to_capture):
            if captured_layer_id == layer_id:
                to_save = hidden_states + residual if residual is not None else hidden_states
                to_save = to_save.to(dtype=eagle3_hidden_states.dtype)
                eagle3_hidden_states[:, i * self.hidden_size:(i + 1) *
                                     self.hidden_size].index_copy_(
                                         0, token_idx,
                                         to_save[:self.num_tokens])
                break

    def get_hidden_states(self):
        hidden_states = self.eagle3_resource_manager.hidden_states[
            self.hidden_states_read_indices[:self.num_tokens], :]
        if not self.is_first_draft:
            hidden_states = hidden_states[:, :self.hidden_size]
        return hidden_states


@dataclass
class Eagle3OneModelSpecMetadata(SpecMetadata):
    # The hidden states
    hidden_states: Optional[torch.Tensor] = None
    # The layers to be captured
    layers_to_capture: Optional[Set[int]] = None
    # The hidden size of the hidden states
    hidden_size: int = 0
    # The max number of tokens
    max_num_tokens: int = 0
    # The dtype of the hidden states
    dtype: torch.dtype = torch.bfloat16
    # The index of the batch inputs
    batch_indices_cuda: Optional[torch.Tensor] = None
    # Optional resource manager (used to access SA manager and relaxed-acceptance
    # delta pool for Eagle3+SA / Eagle3+relaxed-thinking modes)
    spec_resource_manager: Optional[Eagle3ResourceManager] = None
    # Slot IDs for each request; populated in prepare() when spec_resource_manager
    # is present (required for relaxed acceptance, mirrors MTPSpecMetadata.slot_ids).
    slot_ids: Optional[torch.Tensor] = None
    # One-model speculative decoding uses the first draft forward token counts
    # for the first loop iteration and per-sequence token counts for
    # subsequent iterations.
    subseq_all_rank_num_tokens: Optional[List[int]] = None

    def __post_init__(self):
        if self.layers_to_capture is None:
            if self.spec_dec_mode.is_mtp_eagle_one_model(
            ) or self.num_layers == 1:
                self.layers_to_capture = (self.num_layers - 1, )
            else:
                if self.num_layers <= 5:
                    raise ValueError(
                        "Not enough hidden layers for default EAGLE3 capture")

                self.layers_to_capture = (1, self.num_layers // 2 - 1,
                                          self.num_layers - 4)
        else:
            self.layers_to_capture = sorted(list(self.layers_to_capture))
        self.num_capture_layers = len(self.layers_to_capture)
        self.hidden_states = torch.empty(
            (self.max_num_tokens,
             self.hidden_size * len(self.layers_to_capture)),
            dtype=self.dtype,
            device='cuda')
        self.batch_indices_cuda = torch.empty(
            [self.max_num_requests],
            dtype=torch.int,
            device='cuda',
        )
        # Pre-allocate slot_ids; filled in prepare() when spec_resource_manager
        # is present.  Mirrors MTPSpecMetadata.slot_ids allocation pattern.
        self.slot_ids = torch.empty(
            [self.max_num_requests],
            dtype=torch.long,
            device='cuda',
        )

        # currently Eagle3 only supports linear tree
        self.is_spec_dec_tree = False

        # currently Eagle3 only supports static tree
        self.is_spec_dec_dynamic_tree = False

    def is_layer_capture(self, layer_id: int):
        return layer_id in self.layers_to_capture

    def prepare(self):
        assert self.request_ids is not None
        # update batch indices
        num_seqs = len(self.request_ids)
        batch_indices = torch.arange(num_seqs,
                                     dtype=torch.int,
                                     device='cpu',
                                     pin_memory=prefer_pinned())
        self.batch_indices_cuda[:num_seqs].copy_(batch_indices,
                                                 non_blocking=True)
        if not self.spec_dec_mode.is_mtp_eagle_one_model():
            self.num_tokens -= (self.num_generations) * self.runtime_draft_len

        if self.spec_resource_manager is not None:
            # Populate slot_ids for all requests in this batch.  Used by relaxed
            # acceptance (relaxed_delta_pool indexing), mirroring the pattern
            # in MTPSpecMetadata.prepare().
            eagle_slot_ids = [
                self.spec_resource_manager.slot_manager.get_slot(rid)
                for rid in self.request_ids
            ]
            eagle_slot_ids_tensor = torch.tensor(eagle_slot_ids,
                                                 dtype=torch.int,
                                                 pin_memory=prefer_pinned())
            self.slot_ids[:num_seqs].copy_(eagle_slot_ids_tensor,
                                           non_blocking=True)

        sa_manager = getattr(self.spec_resource_manager, 'sa_manager', None)
        if sa_manager is not None:
            gen_request_ids = self.request_ids[num_seqs - self.num_generations:]
            if gen_request_ids:
                sa_manager.prepare(gen_request_ids, self.max_draft_len)

    def maybe_capture_hidden_states(
            self,
            layer_id: int,
            hidden_states: torch.Tensor,
            residual: Optional[torch.Tensor] = None) -> None:
        for i, captured_layer_id in enumerate(self.layers_to_capture):
            if captured_layer_id == layer_id:
                num_tokens = hidden_states.shape[0]
                to_save = hidden_states + residual if residual is not None else hidden_states
                self.hidden_states[:num_tokens, i * self.hidden_size:(i + 1) *
                                   self.hidden_size].copy_(to_save,
                                                           non_blocking=True)
                break


class Eagle3OneModelSampler(MTPSampler):

    def __init__(self, args: TorchSampler.Args):
        super().__init__(args, nextn=args.max_draft_len)


class Eagle3OneModelWorker(SpecWorkerBase):
    """
    Unified one-model worker for Eagle3 and MTP Eagle speculative decoding.

    The operating mode is determined by spec_config.spec_dec_mode:
      - EAGLE3_ONE_MODEL: multi-layer hidden states from Eagle3, apply_eagle3_fc projection,
        independent EAGLE draft model network.
      - MTP_EAGLE_ONE_MODEL: single last-layer hidden states, MTP layer called repeatedly,
        supports TP-aware sampling and Mamba hybrid cache.

    Where the two modes differ, self.is_mtp_eagle is used to branch.
    """

    def __init__(self,
                 spec_config,
                 mapping: Optional[Mapping] = None,
                 model_config: Optional[ModelConfig] = None,
                 use_separate_draft_kv_cache: bool = False):
        super().__init__(use_separate_draft_kv_cache)
        self.spec_config = spec_config
        self.mapping = mapping
        # model_config is required for MTP Eagle TP/ADP support
        self.model_config = model_config

        # Mode flag: True = MTP Eagle one-model, False = Eagle3 one-model
        self.is_mtp_eagle = spec_config.spec_dec_mode.is_mtp_eagle_one_model()

        # SA enhancer (common to both modes)
        self.sa_enhancer: Optional[SADraftEnhancer] = None
        if getattr(spec_config, 'use_sa_spec', False):
            self.sa_enhancer = SADraftEnhancer(spec_config.sa_spec_threshold)

        # MTP Eagle specific attributes
        if self.is_mtp_eagle:
            self._is_mamba_hybrid_cache = None

    @property
    def max_draft_len(self) -> int:
        return self.spec_config.max_draft_len

    def _prepare_attn_metadata_for_spec_dec(self, attn_metadata):
        super()._prepare_attn_metadata_for_spec_dec(attn_metadata)

    def _restore_attn_metadata_from_spec_dec(self, attn_metadata):
        super()._restore_attn_metadata_from_spec_dec(attn_metadata)

    def _prepare_flash_mla_generation_layout(self, attn_metadata, num_contexts,
                                             batch_size):
        if num_contexts <= 0 or not attn_metadata.enable_flash_mla:
            return
        reorder_block_ids_per_seq = torch.cat([
            attn_metadata.kv_block_ids_per_seq[num_contexts:batch_size],
            attn_metadata.kv_block_ids_per_seq[:num_contexts]
        ])
        attn_metadata.block_ids_per_seq[:batch_size, :].copy_(
            reorder_block_ids_per_seq, non_blocking=True)

    def _get_step_all_rank_num_tokens(self, spec_metadata, step_idx: int):
        return (spec_metadata.all_rank_num_tokens
                if step_idx == 0 else spec_metadata.subseq_all_rank_num_tokens)

    def _run_draft_forward(self, draft_model, inputs, spec_metadata,
                           step_idx: int):
        all_rank_num_tokens = self._get_step_all_rank_num_tokens(
            spec_metadata, step_idx)

        if self.is_mtp_eagle:
            hidden_states = draft_model.mtp_layers[0](
                embed_tokens=draft_model.embed_tokens,
                all_rank_num_tokens=all_rank_num_tokens,
                **inputs)
            return hidden_states, None

        inputs["all_rank_num_tokens"] = all_rank_num_tokens
        hidden_states, hidden_states_to_save = draft_model.model(**inputs)
        return hidden_states, hidden_states_to_save

    def sample_and_accept_draft_tokens(
        self,
        input_ids: torch.IntTensor,
        logits: torch.Tensor,
        attn_metadata: AttentionMetadata,
        spec_metadata,
    ):
        """
        Sample the golden token and verify previously proposed draft tokens.

        input_ids is scanned for thinking-phase tokens when relaxed acceptance
        is enabled (both Eagle3 and MTP Eagle); ignored otherwise.
        """
        batch_size = attn_metadata.num_seqs
        num_contexts = attn_metadata.num_contexts
        num_gens = batch_size - num_contexts

        runtime_draft_len = spec_metadata.runtime_draft_len

        if getattr(self.spec_config, 'use_relaxed_acceptance_for_thinking',
                   False):
            # ----------------------------------------------------------------
            # Relaxed acceptance — common path for Eagle3 and MTP Eagle.
            # Accepts draft tokens that fall within the top-K candidates of the
            # target distribution during the thinking phase.
            # ----------------------------------------------------------------
            if logits.dim() == 1:
                logits = logits.unsqueeze(0)

            accepted_tokens = torch.ones((batch_size, runtime_draft_len + 1),
                                         dtype=torch.int,
                                         device=logits.device)
            num_accepted_tokens = torch.ones(batch_size,
                                             dtype=torch.int,
                                             device=logits.device)

            resource_manager = spec_metadata.spec_resource_manager
            relaxed_delta_pool = resource_manager.relaxed_delta_pool

            # Context phase: detect thinking tokens and update the delta pool
            con_logits = logits[:num_contexts]
            con_target_tokens = torch.argmax(con_logits, dim=-1)
            accepted_tokens[:num_contexts, 0] = con_target_tokens[:num_contexts]
            last_tokens_idx_for_thinking = torch.cumsum(
                attn_metadata.seq_lens_cuda, dim=0, dtype=torch.long) - 1
            ctx_input_ids = input_ids[:attn_metadata.num_ctx_tokens]
            ctx_is_think = (ctx_input_ids ==
                            self.spec_config.begin_thinking_phase_token).int()
            ctx_is_think_cumsum = torch.cumsum(ctx_is_think, dim=0)
            ctx_last_cumsum = ctx_is_think_cumsum[
                last_tokens_idx_for_thinking[:num_contexts]]
            ctx_think_tokens_num = torch.diff(
                ctx_last_cumsum,
                dim=0,
                prepend=torch.zeros(1,
                                    dtype=torch.int,
                                    device=ctx_last_cumsum.device))
            ctx_delta = (ctx_think_tokens_num
                         >= 1).int() * self.spec_config.relaxed_delta
            ctx_slot_ids = spec_metadata.slot_ids[:num_contexts]
            relaxed_delta_pool.index_copy_(0, ctx_slot_ids, ctx_delta)

            # Generation phase: top-k logprobs + relaxed acceptance op
            gen_logprobs = self._process_generation_logits(logits, num_contexts)
            topk_value, topk_indices, draft_tokens = self._topk_kernel(
                gen_logprobs, num_gens, runtime_draft_len, spec_metadata)

            accepted_tokens, num_accepted_tokens = torch.ops.trtllm.mtp_relaxed_acceptance_op(
                spec_metadata.slot_ids, topk_value, topk_indices, draft_tokens,
                relaxed_delta_pool, num_accepted_tokens, accepted_tokens,
                runtime_draft_len, batch_size, num_contexts,
                self.spec_config.relaxed_topk, self.spec_config.relaxed_delta,
                self.spec_config.begin_thinking_phase_token,
                self.spec_config.end_thinking_phase_token)

            num_accepted_tokens = self._apply_force_accepted_tokens(
                num_accepted_tokens, num_contexts, runtime_draft_len)

        else:
            # ----------------------------------------------------------------
            # Strict acceptance — common path for Eagle3 and MTP Eagle.
            # Both modes now use runtime_draft_len for dynamic draft length
            # support.
            # ----------------------------------------------------------------
            if logits.dim() == 1:
                logits = logits.unsqueeze(0)
            draft_tokens = spec_metadata.draft_tokens.reshape(
                num_gens, runtime_draft_len)
            accepted_tokens, num_accepted_tokens = \
                self._sample_and_accept_draft_tokens_base(
                    logits, draft_tokens, num_contexts, batch_size,
                    spec_metadata)

        # SA enhancer (common)
        resource_manager = spec_metadata.spec_resource_manager
        sa_manager = getattr(resource_manager, 'sa_manager', None)
        if self.sa_enhancer is not None and sa_manager is not None:
            self.sa_enhancer.extend_and_prepare(
                sa_manager=sa_manager,
                request_ids=spec_metadata.request_ids,
                accepted_tokens=accepted_tokens,
                num_accepted_tokens=num_accepted_tokens,
                num_gens=num_gens,
                num_contexts=num_contexts,
                max_draft_len=self.max_draft_len,
            )

        return accepted_tokens, num_accepted_tokens

    def prepare_drafter_inputs(
        self,
        input_ids: torch.LongTensor,
        position_ids: torch.LongTensor,
        last_tokens_idx: torch.LongTensor,
        hidden_states: torch.Tensor,
        accepted_tokens: torch.Tensor,
        attn_metadata: AttentionMetadata,
        spec_metadata,
        draft_model: Optional[nn.Module] = None,
    ):
        """
        Prepare inputs for the first draft model forward.

        Branching:
          - Eagle3: applies apply_eagle3_fc on multi-layer concatenated hidden
            states; generation tokens limited to runtime_draft_len + 1.
          - MTP Eagle: uses hidden_states directly (single last layer);
            generation tokens include all accepted + 1 bonus token.
        """
        num_contexts = attn_metadata.num_contexts
        num_tokens = input_ids.shape[0]
        runtime_draft_len = spec_metadata.runtime_draft_len

        if self.is_mtp_eagle:
            # MTP Eagle: hidden_states come directly from the target model
            # (single last-layer capture), no FC projection needed
            pass
        else:
            # Eagle3: project the multi-layer concatenated hidden states
            hidden_size_up = spec_metadata.hidden_size * len(
                spec_metadata.layers_to_capture)
            hidden_states = spec_metadata.hidden_states[:num_tokens, :
                                                        hidden_size_up]
            hidden_states = draft_model.apply_eagle3_fc(hidden_states)

        # Context input ids (common logic)
        input_ids_ctx = self._prepare_context_input_ids(
            input_ids, attn_metadata.num_ctx_tokens, last_tokens_idx,
            accepted_tokens, num_contexts)

        input_ids_gen = accepted_tokens[num_contexts:, :runtime_draft_len +
                                        1].flatten()

        input_ids = torch.concat([input_ids_ctx, input_ids_gen], dim=0)

        inputs = {
            "input_ids": input_ids,
            "position_ids": position_ids,
            "hidden_states": hidden_states,
            "attn_metadata": attn_metadata,
            "spec_metadata": spec_metadata,
        }
        return inputs

    # Skip torch.compile for now since current Torch is not compatible with Triton 3.4
    # @torch.compile(options={"max-autotune": True})
    def forward(self,
                input_ids,
                position_ids,
                hidden_states,
                logits,
                attn_metadata,
                spec_metadata,
                draft_model,
                resource_manager=None):

        runtime_draft_len = spec_metadata.runtime_draft_len

        if runtime_draft_len == 0:
            return self.skip_drafting(input_ids, position_ids, hidden_states,
                                      logits, attn_metadata, spec_metadata,
                                      draft_model)

        batch_size = attn_metadata.num_seqs
        num_contexts = attn_metadata.num_contexts
        num_gens = batch_size - num_contexts

        raw_logits = logits

        self._execute_guided_decoder_if_present(logits)

        # Sample and accept draft tokens
        accepted_tokens, num_accepted_tokens = self.sample_and_accept_draft_tokens(
            input_ids, logits, attn_metadata, spec_metadata)

        # MTP Eagle only: Mamba hybrid models need state updates after token
        # acceptance because accepted token count affects which Mamba states
        # are valid; Eagle3 does not use Mamba layers.
        if self.is_mtp_eagle:
            if self._is_mamba_hybrid_cache is None:
                self._is_mamba_hybrid_cache = isinstance(
                    attn_metadata.kv_cache_manager, MambaHybridCacheManager)
            if num_gens > 0 and self._is_mamba_hybrid_cache:
                attn_metadata.kv_cache_manager.update_mamba_states(
                    attn_metadata=attn_metadata,
                    num_accepted_tokens=num_accepted_tokens)

        # Save attn_metadata state before draft loop
        self._prepare_attn_metadata_for_spec_dec(attn_metadata)

        # Compute last-token indices (used for gather_ids in the draft loop)
        position_ids = position_ids.squeeze(0)
        last_tokens_idx = torch.cumsum(
            attn_metadata.seq_lens_cuda, dim=0, dtype=torch.long) - 1

        # Prepare inputs for the first draft forward
        inputs = self.prepare_drafter_inputs(input_ids=input_ids,
                                             position_ids=position_ids,
                                             last_tokens_idx=last_tokens_idx,
                                             hidden_states=hidden_states,
                                             accepted_tokens=accepted_tokens,
                                             attn_metadata=attn_metadata,
                                             spec_metadata=spec_metadata,
                                             draft_model=draft_model)

        next_draft_tokens = []

        draft_kv_cache_manager = self.get_draft_kv_cache_manager(
            resource_manager)

        with self.draft_kv_cache_context(attn_metadata, draft_kv_cache_manager):
            for i in range(runtime_draft_len):
                hidden_states, hidden_states_to_save = self._run_draft_forward(
                    draft_model, inputs, spec_metadata, i)
                if not self.is_mtp_eagle:
                    # Disable spec-dec mode for subsequent iterations
                    attn_metadata.use_spec_decoding = False

                # Compute gather_ids: on the first draft step each generation
                # request may have accepted multiple tokens, so we index into
                # the flattened token sequence to find the last accepted one.
                # From step 1 onwards every sequence has length 1, so
                # batch_indices_cuda is sufficient.
                if i == 0:
                    start_ids_gen = (
                        spec_metadata.batch_indices_cuda[:num_gens] *
                        (runtime_draft_len + 1)).long()
                    gather_ids_gen = (start_ids_gen +
                                      num_accepted_tokens[num_contexts:] - 1 +
                                      attn_metadata.num_ctx_tokens)
                    gather_ids = torch.concat(
                        [last_tokens_idx[:num_contexts], gather_ids_gen], dim=0)
                else:
                    # All seq_lens are 1, use batch_indices_cuda directly
                    gather_ids = spec_metadata.batch_indices_cuda[:batch_size]

                if self.guided_decoder is not None:
                    new_tokens = inputs["input_ids"][gather_ids]
                    self.guided_decoder.add_draft_batch(new_tokens,
                                                        num_accepted_tokens,
                                                        draft_step=i)

                if self.is_mtp_eagle:
                    # MTP Eagle: shared head of the MTP layer, with LM head TP/ADP support
                    if (self.model_config is not None
                            and self.model_config.mapping.enable_attention_dp
                            and getattr(self.model_config.mapping,
                                        'enable_lm_head_tp_in_adp', False)):
                        hidden_states_gathered = hidden_states[gather_ids]
                        token_count = hidden_states_gathered.view(
                            -1, hidden_states_gathered.shape[-1]).shape[0]
                        max_num_requests = spec_metadata.max_num_requests
                        pad_len = max_num_requests - token_count
                        if pad_len > 0:
                            padded_hidden_states = F.pad(
                                hidden_states_gathered.view(
                                    -1, hidden_states_gathered.shape[-1]),
                                (0, 0, 0, pad_len),
                                mode="constant",
                                value=0)
                        elif pad_len == 0:
                            padded_hidden_states = hidden_states_gathered.view(
                                -1, hidden_states_gathered.shape[-1])
                        else:
                            raise ValueError(
                                "In Eagle3OneModelWorker (MTP Eagle mode): "
                                "token_count > max_num_requests, which is not supported"
                            )
                        logits = draft_model.mtp_layers[0].shared_head(
                            padded_hidden_states, draft_model.lm_head,
                            attn_metadata, True)
                    else:
                        logits = draft_model.mtp_layers[0].shared_head(
                            hidden_states[gather_ids], draft_model.lm_head,
                            attn_metadata, True)
                else:
                    # Eagle3: logits processor of the EAGLE draft model
                    logits = draft_model.logits_processor(
                        hidden_states[gather_ids], draft_model.lm_head,
                        attn_metadata, True)

                if self.guided_decoder is not None:
                    if self.is_mtp_eagle:
                        self.guided_decoder.execute_draft_batch(logits,
                                                                draft_step=i)
                    else:
                        d2t = getattr(draft_model.model, "d2t", None)
                        self.guided_decoder.execute_draft_batch(logits,
                                                                d2t,
                                                                draft_step=i)

                # Sample the next draft token.
                # MTP Eagle: TP-aware sampler; when ADP+LM-head-TP is active
                #   logits are padded to max_num_requests across TP ranks, so
                #   the result must be trimmed back to token_count.
                # Eagle3: simple greedy sampling; d2t remaps vocab indices
                #   when the draft model uses a compressed vocabulary.
                if self.is_mtp_eagle:
                    if (self.model_config is not None
                            and self.model_config.mapping.enable_attention_dp
                            and getattr(self.model_config.mapping,
                                        'enable_lm_head_tp_in_adp', False)):
                        mapping_lm_head_tp = draft_model.mtp_layers[
                            0].shared_head.mapping_lm_head_tp
                        new_draft_token = self.draft_sampler(
                            logits, mapping_lm_head_tp)
                        new_draft_token = new_draft_token[:token_count]
                    else:
                        new_draft_token = self.draft_sampler(logits)
                else:
                    # Eagle3: greedy sampling with optional token mapping (d2t)
                    d2t = getattr(draft_model.model, "d2t", None)
                    new_draft_token = self._draft_sampler_greedy(logits, d2t)

                next_draft_tokens.append(new_draft_token)

                # Update hidden states for the next iteration.
                # MTP Eagle: the MTP layer returns a single tensor; slice by
                #   gather_ids to get one hidden state per request.
                # Eagle3: the EAGLE draft model returns a secondary
                #   hidden_states_to_save specifically for this purpose.
                if self.is_mtp_eagle:
                    hidden_states = hidden_states[gather_ids]
                else:
                    hidden_states = hidden_states_to_save[gather_ids]

                position_ids = inputs["position_ids"][gather_ids] + 1

                if i == 0:
                    attn_metadata._seq_lens[:batch_size].fill_(1)
                    attn_metadata._seq_lens_cuda[:batch_size].fill_(1)
                    attn_metadata.on_update()
                    # Cannot run generation if there is no KV cache
                    has_kv_cache = inputs[
                        "attn_metadata"].kv_cache_manager is not None
                    if has_kv_cache:
                        attn_metadata.host_request_types[:attn_metadata.
                                                         num_contexts].fill_(1)
                        attn_metadata.num_contexts = 0
                    if hasattr(attn_metadata, 'kv_lens_cuda'):
                        attn_metadata.kv_lens_cuda[num_contexts:batch_size] -= (
                            runtime_draft_len -
                            num_accepted_tokens[num_contexts:])
                        attn_metadata.kv_lens_cuda[:num_contexts] += 1

                    if has_kv_cache:
                        self._prepare_flash_mla_generation_layout(
                            attn_metadata, num_contexts, batch_size)
                    if hasattr(attn_metadata, 'kv_lens_cuda'):
                        attn_metadata.update_for_spec_dec()

                    attn_metadata.use_spec_decoding = False
                else:
                    if hasattr(attn_metadata, 'kv_lens_cuda'):
                        attn_metadata.kv_lens_cuda[:batch_size] += 1
                        attn_metadata.update_for_spec_dec()

                inputs = {
                    "input_ids": new_draft_token,
                    "position_ids": position_ids,
                    "hidden_states": hidden_states,
                    "attn_metadata": attn_metadata,
                    "spec_metadata": spec_metadata,
                }
        next_draft_tokens = torch.stack(next_draft_tokens, dim=1)

        # SA draft token override (common)
        if self.sa_enhancer is not None:
            gen_draft_tokens = next_draft_tokens[num_contexts:]
            gen_draft_tokens = self.sa_enhancer.maybe_override_all_draft_tokens(
                gen_draft_tokens)
            next_draft_tokens[num_contexts:] = gen_draft_tokens

        # Restore attn_metadata
        self._restore_attn_metadata_from_spec_dec(attn_metadata)

        next_new_tokens = self._prepare_next_new_tokens(
            accepted_tokens, next_draft_tokens,
            spec_metadata.batch_indices_cuda, batch_size, num_accepted_tokens)

        attn_metadata.use_spec_decoding = True

        return {
            'logits': raw_logits,
            'new_tokens': accepted_tokens,
            'new_tokens_lens': num_accepted_tokens,
            'next_draft_tokens': next_draft_tokens,
            'next_new_tokens': next_new_tokens,
        }

    @torch.compile(options={"max-autotune": True})
    def _get_local_max_and_combined(self, logits, mapping_lm_tp=None):
        local_max_values, local_argmax = torch.max(logits, dim=-1, keepdim=True)
        vocab_per_rank = logits.shape[-1]
        mapping_lm_tp = mapping_lm_tp if mapping_lm_tp is not None else \
            self.model_config.mapping
        max_index_per_rank = local_argmax.type(
            torch.int32) + (mapping_lm_tp.tp_rank * vocab_per_rank)
        max_index_per_rank_float = max_index_per_rank.float()
        local_max_values_float32 = local_max_values.float()
        combined = torch.stack(
            [max_index_per_rank_float, local_max_values_float32],
            dim=-1).flatten(-2)
        return combined

    @torch.compile(options={"max-autotune": True})
    def _get_draft_tokens_from_gathered(self, gathered):
        gathered_indices_float = gathered[..., 0::2]
        gathered_values_float = gathered[..., 1::2]
        max_indices = torch.argmax(gathered_values_float, dim=-1, keepdim=True)
        draft_tokens = torch.gather(gathered_indices_float, -1,
                                    max_indices).squeeze(-1).type(torch.int32)
        return draft_tokens

    def draft_sampler(
        self,
        logits: torch.Tensor,
        mapping_lm_head_tp=None,
    ):
        """
        TP-aware greedy draft token sampler (MTP Eagle mode).

        Falls back to simple argmax when no tensor parallelism is active.
        """
        if (self.model_config is not None
                and hasattr(self.model_config, 'mapping')
                and self.model_config.mapping.tp_size > 1
                and not self.model_config.mapping.enable_attention_dp):
            combined = self._get_local_max_and_combined(logits)
            gathered = allgather(combined, self.model_config.mapping, dim=-1)
            return self._get_draft_tokens_from_gathered(gathered)
        elif (self.model_config is not None
              and hasattr(self.model_config, 'mapping')
              and self.model_config.mapping.tp_size > 1
              and self.model_config.mapping.enable_lm_head_tp_in_adp):
            combined = self._get_local_max_and_combined(logits,
                                                        mapping_lm_head_tp)
            gathered = allgather(combined, mapping_lm_head_tp, dim=-1)
            batch_size = logits.shape[0]
            local_batch_size = batch_size // mapping_lm_head_tp.tp_size
            gathered = gathered.view(mapping_lm_head_tp.tp_size,
                                     local_batch_size, -1)
            sliced_gathered = gathered[mapping_lm_head_tp.tp_rank]
            return self._get_draft_tokens_from_gathered(sliced_gathered)
        else:
            return self._draft_sampler_greedy(logits)

    @torch.compile(options={"max-autotune": True})
    def _topk_kernel(self, gen_logprobs, num_gens, mtp_num_modules,
                     spec_metadata):
        topk_value, topk_indices = torch.topk(gen_logprobs,
                                              k=self.spec_config.relaxed_topk,
                                              dim=-1)
        topk_indices = topk_indices.reshape(num_gens, mtp_num_modules + 1,
                                            self.spec_config.relaxed_topk)
        topk_value = topk_value.reshape(num_gens, mtp_num_modules + 1,
                                        self.spec_config.relaxed_topk)
        draft_tokens = spec_metadata.draft_tokens.reshape(
            num_gens, mtp_num_modules)
        return topk_value, topk_indices, draft_tokens

    @torch.compile(options={"max-autotune": True})
    def _process_generation_logits(self, logits, num_contexts):
        gen_logits = logits[num_contexts:]
        gen_logprobs = torch.softmax(gen_logits, dim=-1)
        return gen_logprobs


class MTPEagleWorker(Eagle3OneModelWorker):
    """
    Backward-compatible wrapper preserving the historical MTPEagleWorker
    constructor while delegating to the unified implementation.
    """

    def __init__(self,
                 spec_config,
                 model_config: Optional[ModelConfig] = None,
                 use_separate_draft_kv_cache: bool = False):
        super().__init__(
            spec_config,
            model_config=model_config,
            use_separate_draft_kv_cache=use_separate_draft_kv_cache)
        # Preserve the historical public attribute for callers and tests that
        # still expect it on MTPEagleWorker instances.
        self.is_thop = False
