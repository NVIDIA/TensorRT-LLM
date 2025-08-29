from dataclasses import dataclass, field
from typing import List, Optional, Set

import torch
from torch import nn

from tensorrt_llm.mapping import Mapping

from ..attention_backend import AttentionMetadata
from ..pyexecutor.llm_request import LlmRequest
from ..pyexecutor.resource_manager import BaseResourceManager, SlotManager
from ..pyexecutor.sampler import TorchSampler
from ..pyexecutor.scheduler import ScheduledRequests
from .interface import SpecMetadata
from .mtp import MTPSampler


class Eagle3ResourceManager(BaseResourceManager):
    """
    Eagle3 needs to save the hidden states for the draft model. When using
    Eagle3TwoModel, there will be two model engines, one for the target model
    and one for the draft model. Use this class to manage the hidden states.
    """

    def __init__(self, config: "EagleDecodingConfig", dtype: torch.dtype,
                 hidden_size: int, max_num_requests: int, max_seq_len: int,
                 max_num_tokens: int):
        self.dtype = dtype
        self.max_draft_len = config.max_draft_len
        self.hidden_size = hidden_size
        self.max_num_requests = max_num_requests
        self.max_seq_len = max_seq_len
        self.slot_manager = SlotManager(max_num_requests)

        # empty hidden states tensor
        max_num_tokens = min(max_num_tokens,
                             max_num_requests * self.max_seq_len)
        self.hidden_states = torch.empty(
            (max_num_tokens, self.hidden_size * config.num_capture_layers),
            dtype=self.dtype,
            device='cuda')
        # sequence length, only used for metadata preparation
        self.seq_lens = {i: 0 for i in range(max_num_requests)}
        # start indices of each slot
        self.start_indices = {i: 0 for i in range(max_num_requests)}
        # whether the next draft forward is the first
        self.is_first_draft = True

    def prepare_resources(self, scheduled_batch: ScheduledRequests):
        context_batch = scheduled_batch.context_requests
        # allocate hidden state tensors and update slot ids
        self.slot_ids = []
        for req in context_batch:
            if req.is_first_context_chunk:
                slot_id = self.slot_manager.add_slot(req.request_id)
                self.slot_ids.append(slot_id)
        # reset the flag before model forward
        self.is_first_draft = True

    def update_resources(self, scheduled_batch: ScheduledRequests):
        pass

    def free_resources(self, request: LlmRequest):
        self.slot_manager.remove_slot(request.request_id)

    def add_dummy_requests(self, request_ids: List[int]):
        for rid in request_ids:
            self.slot_manager.add_slot(rid)

    def shutdown(self):
        pass

    def get_max_resource_count(self) -> int:
        return self.max_num_requests

    def get_needed_resource_to_completion(self, request: LlmRequest):
        return 0


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

    def __post_init__(self):
        if self.layers_to_capture is None:
            if self.num_layers == 1:
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

        # Initialize to 0 to avoid reading uninitialized memory during warmup
        self.hidden_states_read_indices = torch.zeros([self.max_num_tokens],
                                                      dtype=torch.long,
                                                      device='cuda')
        self.hidden_states_write_indices = torch.zeros([self.max_num_tokens],
                                                       dtype=torch.long,
                                                       device='cuda')
        self.hidden_states_read_indices_host = None
        self.hidden_states_write_indices_host = None

    def prepare(self):
        is_first_draft = self.eagle3_resource_manager.is_first_draft
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
                start_idx += seq_len
        # Prepare hidden states gather ids
        hidden_states_read_indices = []
        hidden_states_write_indices = []
        for req_id, seq_len in zip(self.request_ids, self.seq_lens):
            slot_id = self.eagle3_resource_manager.slot_manager.get_slot(req_id)
            start_idx = self.eagle3_resource_manager.start_indices[slot_id]
            # If this is the first draft or the target model forward, we need to
            # read/write all of the hidden states, otherwise, only read the last token
            if is_first_draft or not self.is_draft_model:
                hidden_states_read_indices.extend(
                    list(range(start_idx, start_idx + seq_len)))
                hidden_states_write_indices.extend(
                    list(range(start_idx, start_idx + seq_len)))
            else:
                old_seq_len = self.eagle3_resource_manager.seq_lens[slot_id]
                hidden_states_read_indices.append(start_idx + old_seq_len - 1)
                hidden_states_write_indices.append(start_idx + seq_len - 1)
            self.eagle3_resource_manager.seq_lens[slot_id] = seq_len
        # Prepare hidden states gather ids
        self.hidden_states_read_indices_host = torch.tensor(
            hidden_states_read_indices, dtype=torch.long, pin_memory=True)
        self.hidden_states_write_indices_host = torch.tensor(
            hidden_states_write_indices, dtype=torch.long, pin_memory=True)
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
                                         0, token_idx, to_save)
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
    # The index of the batche inputs
    batch_indices_cuda: Optional[torch.Tensor] = None

    def __post_init__(self):
        if self.layers_to_capture is None:
            if self.num_layers == 1:
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

        # currently Eagle3 only supports linear tree
        self.is_spec_dec_tree = False

        # currently Eagle3 only supports static tree
        self.is_spec_dec_dynamic_tree = False

    def is_layer_capture(self, layer_id: int):
        return layer_id in self.layers_to_capture

    def prepare(self):
        assert self.request_ids is not None
        # update batch indeices
        num_seqs = len(self.request_ids)
        batch_indices = torch.arange(num_seqs,
                                     dtype=torch.int,
                                     device='cpu',
                                     pin_memory=True)
        self.batch_indices_cuda[:num_seqs].copy_(batch_indices,
                                                 non_blocking=True)
        self.num_tokens -= (self.num_generations) * self.max_draft_len

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


class Eagle3OneModelWorker(nn.Module):

    def __init__(self, spec_config: "EagleDecodingConfig", mapping: Mapping):
        super().__init__()
        self.spec_config = spec_config
        self.max_draft_len = self.spec_config.max_draft_len
        self.mapping = mapping

    # Skip torch.compile for now since current Torch is not compatible with Triton 3.4
    # @torch.compile(options={"max-autotune": True})
    def forward(self, input_ids, position_ids, hidden_states, logits,
                attn_metadata, spec_metadata, draft_model):
        batch_size = attn_metadata.num_seqs
        num_contexts = attn_metadata.num_contexts
        num_gens = batch_size - num_contexts

        raw_logits = logits

        # Sample and accept tokens
        accepted_tokens, num_accepted_tokens = self.sample_and_accept_draft_tokens(
            logits, attn_metadata, spec_metadata)

        # Save the old attn_metadata and spec_metadata
        if attn_metadata.is_cuda_graph:
            seq_len = attn_metadata._seq_lens[:batch_size].clone()
            seq_len_cuda = attn_metadata._seq_lens_cuda[:batch_size].clone()

        @torch.compile(options={"max-autotune": True})
        def calc_position_ids_and_last_tokens_idx(position_ids, attn_metadata):
            position_ids = position_ids.squeeze(0)
            last_tokens_idx = torch.cumsum(
                attn_metadata.seq_lens_cuda, dim=0, dtype=torch.long) - 1
            return position_ids, last_tokens_idx

        position_ids, last_tokens_idx = calc_position_ids_and_last_tokens_idx(position_ids, attn_metadata)

        inputs = self.prepare_1st_drafter_inputs(
            input_ids=input_ids,
            position_ids=position_ids,
            last_tokens_idx=last_tokens_idx,
            hidden_states=hidden_states,
            accepted_tokens=accepted_tokens,
            attn_metadata=attn_metadata,
            spec_metadata=spec_metadata,
            draft_model=draft_model)

        # Predict draft tokens
        next_draft_tokens = []
        for i in range(self.max_draft_len):
            hidden_states, hidden_states_to_save = draft_model.model(**inputs)

            # FIXME (jhaotingc): Currently we disable use_spec_decoding mode for Eagle engine nth steps except 1st step.
            # Eagle engine takes in draft_len tokens from the previous step, run spec-dec mode with those tokens,
            # then the following step can use regular decoding mode to generate 1 tokens per step.
            # Currently the spec-dec mask for chained tree is not implemented yet.
            # When token tree is supported, this can be removed and all steps may use spec-dec mode as well.
            attn_metadata.use_spec_decoding = False
            if i == 0:
                @torch.compile(options={"max-autotune": True}) # 7us saving
                def compute_gather_ids(spec_metadata, num_gens, self_max_draft_len, num_accepted_tokens, num_contexts, attn_metadata, last_tokens_idx):
                    start_ids_gen = (spec_metadata.batch_indices_cuda[:num_gens] *
                                     (self_max_draft_len + 1)).long()
                    gather_ids_gen = (start_ids_gen +
                                      num_accepted_tokens[num_contexts:] - 1 +
                                      attn_metadata.num_ctx_tokens)
                    gather_ids = torch.concat(
                        [last_tokens_idx[:num_contexts], gather_ids_gen], dim=0)
                    return gather_ids

                gather_ids = compute_gather_ids(
                    spec_metadata, num_gens, self.max_draft_len, num_accepted_tokens, num_contexts, attn_metadata, last_tokens_idx
                )
            else:
                # All of the seq_len are 1, use batch_indices_cuda as gather_ids
                @torch.compile(options={"max-autotune": True})
                def get_gather_ids(spec_metadata, batch_size):
                    return spec_metadata.batch_indices_cuda[:batch_size]
                gather_ids = get_gather_ids(spec_metadata, batch_size)
            logits = draft_model.logits_processor(hidden_states[gather_ids],
                                                  draft_model.lm_head,
                                                  attn_metadata, True)
            new_draft_token = self.draft_decoder(logits, draft_model)
            
            @torch.compile(options={"max-autotune": True})
            def update_draft_tokens_and_inputs(new_draft_token, hidden_states_to_save, gather_ids, inputs):
                next_draft_tokens.append(new_draft_token)
                hidden_states = hidden_states_to_save[gather_ids]
                position_ids = inputs["position_ids"][gather_ids] + 1
                return hidden_states, position_ids

            hidden_states, position_ids = update_draft_tokens_and_inputs(
                new_draft_token, hidden_states_to_save, gather_ids, inputs
            )
            # update attn_metadata
            if i == 0:
                attn_metadata._seq_lens[:batch_size].fill_(1)
                attn_metadata._seq_lens_cuda[:batch_size].fill_(1)
                attn_metadata.on_update()
                # cannot run generation if their is no kv cache
                if inputs["attn_metadata"].kv_cache_manager is not None:
                    attn_metadata.host_request_types[:attn_metadata.
                                                     num_contexts].fill_(1)
                    attn_metadata.num_contexts = 0
                # update kv_lens_cuda
                if hasattr(attn_metadata, 'kv_lens_cuda'):
                    attn_metadata.kv_lens_cuda[num_contexts:batch_size] -= (
                        self.max_draft_len - num_accepted_tokens[num_contexts:])
                    attn_metadata.kv_lens_cuda[:num_contexts] += 1
            elif hasattr(attn_metadata, 'kv_lens_cuda'):
                @torch.compile(options={"max-autotune": True})
                def update_kv_lens_cuda(attn_metadata, batch_size):
                    attn_metadata.kv_lens_cuda[:batch_size] += 1
                update_kv_lens_cuda(attn_metadata, batch_size)
            # support attention dp
            if spec_metadata.all_rank_num_tokens is not None:
                spec_metadata.all_rank_num_tokens = spec_metadata.all_rank_num_seqs
            inputs = {
                "input_ids": new_draft_token,
                "position_ids": position_ids,
                "hidden_states": hidden_states,
                "attn_metadata": attn_metadata,
                "spec_metadata": spec_metadata,
            }

        # restore attn_metadata to support cuda graph
        if attn_metadata.is_cuda_graph:
            attn_metadata._seq_lens[:batch_size].copy_(seq_len)
            attn_metadata._seq_lens_cuda[:batch_size].copy_(seq_len_cuda)
            attn_metadata.on_update()

        @torch.compile(options={"max-autotune": True})
        def prepare_next_tokens(next_draft_tokens, accepted_tokens, spec_metadata, batch_size, num_accepted_tokens):
            next_draft_tokens_stacked = torch.stack(next_draft_tokens, dim=1)
            next_new_tokens = accepted_tokens[
                spec_metadata.batch_indices_cuda[:batch_size],
                num_accepted_tokens - 1
            ].unsqueeze(1)
            next_new_tokens = torch.concat([next_new_tokens, next_draft_tokens_stacked], dim=1)
            return next_draft_tokens_stacked, next_new_tokens

        next_draft_tokens, next_new_tokens = prepare_next_tokens(
            next_draft_tokens, accepted_tokens, spec_metadata, batch_size, num_accepted_tokens
        )

        attn_metadata.use_spec_decoding = True

        return {
            'logits': raw_logits,
            'new_tokens': accepted_tokens,
            'new_tokens_lens': num_accepted_tokens,
            'next_draft_tokens': next_draft_tokens,
            'next_new_tokens': next_new_tokens,
        }

    # @torch.compile(options={"max-autotune": True}) # dont apply on all
    def sample_and_accept_draft_tokens(
        self,
        logits: torch.Tensor,
        attn_metadata: AttentionMetadata,
        spec_metadata: Eagle3OneModelSpecMetadata,
    ):
        @torch.compile(options={"max-autotune": True}) # this and below compile saves 8us ; torch.compile on argmax spoils the performance
        def get_num_gens_and_accepted_tokens(logits, attn_metadata, max_draft_len):
            batch_size = attn_metadata.num_seqs
            num_contexts = attn_metadata.num_contexts
            num_gens = batch_size - num_contexts

            if logits.dim() == 1:
                logits = logits.unsqueeze(0)

            # The return buffer
            accepted_tokens = torch.empty((batch_size, (max_draft_len + 1)),
                                        dtype=torch.int,
                                        device=logits.device)
            num_accepted_tokens = torch.ones(batch_size,
                                            dtype=torch.int,
                                            device=logits.device)
            
            return num_gens, num_contexts, num_accepted_tokens, accepted_tokens

        num_gens, num_contexts, num_accepted_tokens, accepted_tokens = get_num_gens_and_accepted_tokens(logits, attn_metadata, self.max_draft_len)

        # Do greedy sampling for the input logits
        target_tokens = torch.argmax(logits, dim=-1)
        
        @torch.compile(options={"max-autotune": True})
        def process_accepted_tokens(target_tokens, num_contexts, num_gens, accepted_tokens, gen_target_tokens, draft_tokens, num_accepted_tokens, max_draft_len):
            # context
            accepted_tokens[:num_contexts, 0] = target_tokens[:num_contexts]

            # generation
            gen_target_tokens_reshaped = gen_target_tokens.reshape(
                num_gens, max_draft_len + 1)
            accepted_tokens[num_contexts:, :] = gen_target_tokens_reshaped
            draft_tokens_reshaped = draft_tokens.reshape(
                num_gens, max_draft_len)
            num_accepted_tokens[num_contexts:] += torch.cumprod(
                (draft_tokens_reshaped == gen_target_tokens_reshaped[:, :max_draft_len]).int(),
                dim=-1).sum(1)
            return accepted_tokens, num_accepted_tokens

        accepted_tokens, num_accepted_tokens = process_accepted_tokens(
            target_tokens, num_contexts, num_gens, accepted_tokens,
            target_tokens[num_contexts:], spec_metadata.draft_tokens,
            num_accepted_tokens, self.max_draft_len
        )
        return accepted_tokens, num_accepted_tokens

    # @torch.compile(options={"max-autotune": True})
    def draft_decoder(
        self,
        logits: torch.Tensor,
        draft_model: nn.Module,
    ):
        '''
        Sampling draft tokens.

        Args:
            logits: torch.Tensor
                [num_tokens, vocab_size]
                Logits produced by the draft model.
            draft_model: nn.Module
                The draft model.

        Returns:
            draft_tokens: torch.Tensor
                [batch_size * max_draft_len]
                Draft token ids. Flattened.
        '''
        # print(f"DBG : logits.shape: {logits.shape} {logits.dtype} {logits.device} logits stride: {logits.stride()}")
        draft_tokens = torch.argmax(logits, dim=-1) # [num_tokens]
        # print(f"DBG : draft_tokens.shape: {draft_tokens.shape} {draft_tokens.dtype} {draft_tokens.device} draft_tokens stride: {draft_tokens.stride()}")

        # Apply d2t (offsets between draft model dictionary and main model dictionary).
        if hasattr(draft_model.model,
                   "d2t") and draft_model.model.d2t is not None:
            draft_tokens = draft_model.model.d2t[draft_tokens] + draft_tokens

        draft_tokens = draft_tokens.type(torch.int32)

        return draft_tokens

    @torch.compile(options={"max-autotune": True})
    def prepare_1st_drafter_inputs(
        self,
        input_ids: torch.LongTensor,
        position_ids: torch.LongTensor,
        last_tokens_idx: torch.LongTensor,
        hidden_states: torch.Tensor,
        accepted_tokens: torch.Tensor,
        attn_metadata: AttentionMetadata,
        spec_metadata: Eagle3OneModelSpecMetadata,
        draft_model: nn.Module,
    ):
        num_contexts = attn_metadata.num_contexts
        num_tokens = input_ids.shape[0]

        # prepare hidden states
        hidden_size_up = spec_metadata.hidden_size * len(
            spec_metadata.layers_to_capture)
        hidden_states = spec_metadata.hidden_states[:num_tokens, :
                                                    hidden_size_up]
        hidden_states = draft_model.apply_eagle3_fc(hidden_states)

        # context
        input_ctx_ids = input_ids[:attn_metadata.num_ctx_tokens]
        input_ids_ctx = torch.empty_like(input_ctx_ids,
                                         dtype=torch.int32,
                                         device="cuda")
        input_ids_ctx[:-1].copy_(input_ctx_ids[1:])
        input_ids_ctx[
            last_tokens_idx[:num_contexts]] = accepted_tokens[:num_contexts, 0]

        # generation
        input_ids_gen = accepted_tokens[num_contexts:, :].flatten()

        # get draft inputs
        input_ids = torch.concat([input_ids_ctx, input_ids_gen], dim=0)

        return {
            "input_ids": input_ids,
            "position_ids": position_ids,
            "hidden_states": hidden_states,
            "attn_metadata": attn_metadata,
            "spec_metadata": spec_metadata,
        }
