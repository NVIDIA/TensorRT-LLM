from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional

import torch
import torch.nn.functional as F
from torch import nn

from tensorrt_llm.mapping import Mapping

from ..attention_backend import AttentionMetadata
from ..distributed.ops import allgather
from ..model_config import ModelConfig
from ..pyexecutor.guided_decoder import CapturableGuidedDecoder
from ..pyexecutor.llm_request import LlmRequest, LlmRequestState
from ..pyexecutor.resource_manager import BaseResourceManager, SlotManager
from ..pyexecutor.sampler import (SampleState, SampleStateTensors, TorchSampler,
                                  add_token, int_tensor)
from ..pyexecutor.scheduler import ScheduledRequests
from .interface import SpecMetadata

if TYPE_CHECKING:
    from tensorrt_llm.llmapi.llm_args import MTPDecodingConfig


@dataclass(kw_only=True)
class SampleStateTensorsMTP(SampleStateTensors):
    new_tokens_lens: torch.Tensor
    next_draft_tokens: torch.Tensor


@dataclass(kw_only=True)
class SampleStateMTP(SampleState):
    device: SampleStateTensorsMTP
    host: SampleStateTensorsMTP


class MTPHiddenStatesManager(BaseResourceManager):

    def __init__(self, config: "MTPDecodingConfig", dtype: torch.dtype,
                 hidden_size: int, max_num_requests: int):
        self.dtype = dtype
        self.num_nextn_predict_layers = config.num_nextn_predict_layers
        self.hidden_size = hidden_size
        self.max_num_requests = max_num_requests
        self.use_relaxed_acceptance_for_thinking = config.use_relaxed_acceptance_for_thinking
        self.slot_manager = SlotManager(max_num_requests)

        # Since golden token's hidden state will always be generated after target model
        self.mtp_past_hidden_states_pool = torch.zeros(
            (max_num_requests, self.num_nextn_predict_layers, self.hidden_size),
            device='cuda',
            dtype=self.dtype,
        )
        self.mtp_past_tokens_pool = torch.zeros(
            (max_num_requests, self.num_nextn_predict_layers),
            device='cuda',
            dtype=torch.int,
        )
        if self.use_relaxed_acceptance_for_thinking:
            # The relaxed_delta for relaxed acceptance
            self.mtp_relaxed_delta_pool = torch.zeros(
                (self.max_num_requests),
                dtype=torch.float,
                device='cuda',
            )

    def prepare_resources(self, scheduled_batch: ScheduledRequests):
        context_batch = scheduled_batch.context_requests
        # allocate hidden state tensors
        for req in context_batch:
            if req.is_first_context_chunk:
                slot_id = self.slot_manager.add_slot(req.request_id)
                if self.use_relaxed_acceptance_for_thinking:
                    self.mtp_relaxed_delta_pool[slot_id].copy_(
                        0, non_blocking=True)

    def update_resources(self, scheduled_batch: ScheduledRequests):
        pass

    def free_resources(self, request: LlmRequest):
        free_slot_id = self.slot_manager.get_slot(request.request_id)
        if self.use_relaxed_acceptance_for_thinking:
            self.mtp_relaxed_delta_pool[free_slot_id].copy_(0,
                                                            non_blocking=True)
        self.slot_manager.remove_slot(request.request_id)

    def add_dummy_requests(self, request_ids: List[int]):
        for rid in request_ids:
            self.slot_manager.add_slot(rid)

    def shutdown(self):
        self.slot_manager.shutdown()

    def get_max_resource_count(self) -> int:
        return self.max_num_requests

    def get_needed_resource_to_completion(self, request: LlmRequest):
        return 0


@dataclass
class MTPSpecMetadata(SpecMetadata):
    """
    Metadata for MTP.
    """
    # The number of MTP modules in the model
    mtp_num_modules: int = 1
    # The hidden states manager for MTP
    mtp_hidden_states_manager: Optional[MTPHiddenStatesManager] = None
    # The slot ids for each request.
    slot_ids: Optional[torch.Tensor] = None
    # The index of the batche inputs
    batch_indices_cuda: Optional[torch.Tensor] = None
    # The number of sequences for speculative model/layer of different rank
    _all_rank_num_seqs: Optional[List[int]] = None
    # This is used for attention dp in the MTP Eagle worker. The numbers of input
    # tokens varies between the 1st draft forward and subsequent ones. To support
    # CUDA graph, we use this tensor to store the number of input tokens for the
    # subsequence draft forward.
    subseq_all_rank_num_tokens: Optional[List[int]] = None

    def __post_init__(self) -> None:
        if self.mtp_hidden_states_manager is not None:
            # mtp_hidden_states_ptrs is a pointer tensor
            self.mtp_hidden_states_ptrs = torch.empty(
                [self.max_num_requests],
                dtype=torch.int64,
                device='cuda',
            )

            self.mtp_past_tokens_ptrs = torch.empty(
                [self.max_num_requests],
                dtype=torch.int64,
                device='cuda',
            )
            self.slot_ids = torch.empty(
                [self.max_num_requests],
                dtype=torch.long,
                device='cuda',
            )
        self.batch_indices_cuda = torch.empty(
            [self.max_num_requests],
            dtype=torch.int,
            device='cuda',
        )
        self.draft_token_indices_cuda = torch.arange(
            self.mtp_num_modules,
            device='cuda',
        )

    @property
    def all_rank_num_seqs(self):
        return self._all_rank_num_seqs

    @all_rank_num_seqs.setter
    def all_rank_num_seqs(self, value: List[int]):
        self._all_rank_num_seqs = value
        if self.spec_dec_mode.is_mtp_eagle_one_model():
            self.subseq_all_rank_num_tokens = value

    def prepare(self):
        assert self.request_ids is not None
        num_seqs = len(self.request_ids)
        # update batch indeices
        batch_indices = torch.arange(num_seqs,
                                     dtype=torch.int,
                                     device='cpu',
                                     pin_memory=True)
        self.batch_indices_cuda[:num_seqs].copy_(batch_indices,
                                                 non_blocking=True)
        # MTP vanilla worker uses total max_draft_len input tokens in generation phase,
        # while MTP Eagle worker uses (max_draft_len + 1) input tokens in the 1st draft
        # forward and only one input token in the following draft forward.
        # This num_tokens is used to set the all_rank_num_tokens for attention dp.
        if not self.spec_dec_mode.is_mtp_eagle_one_model():
            self.num_tokens -= self.num_generations

        if self.mtp_hidden_states_manager is not None:  # MTP vanilla or use relaxed acceptance
            mtp_slot_ids = []
            for rid in self.request_ids:
                slot_id = self.mtp_hidden_states_manager.slot_manager.get_slot(
                    rid)
                mtp_slot_ids.append(slot_id)

            # MTP Vanilla: Update mtp hidden states and past tokens
            if self.spec_dec_mode.is_mtp_one_model():
                mtp_hidden_states_ptrs = []
                mtp_past_tokens_ptrs = []
                for slot_id in mtp_slot_ids:
                    mtp_hidden_states_ptrs.append(
                        self.mtp_hidden_states_manager.
                        mtp_past_hidden_states_pool[slot_id].data_ptr())
                    mtp_past_tokens_ptrs.append(
                        self.mtp_hidden_states_manager.
                        mtp_past_tokens_pool[slot_id].data_ptr())
                mtp_hidden_states_ptrs = torch.tensor(mtp_hidden_states_ptrs,
                                                      dtype=torch.int64,
                                                      pin_memory=True)
                mtp_past_tokens_ptrs = torch.tensor(mtp_past_tokens_ptrs,
                                                    dtype=torch.int64,
                                                    pin_memory=True)
                self.mtp_hidden_states_ptrs[:num_seqs].copy_(
                    mtp_hidden_states_ptrs, non_blocking=True)
                self.mtp_past_tokens_ptrs[:num_seqs].copy_(mtp_past_tokens_ptrs,
                                                           non_blocking=True)
            mtp_slot_ids = torch.tensor(mtp_slot_ids,
                                        dtype=torch.int,
                                        pin_memory=True)
            self.slot_ids[:num_seqs].copy_(mtp_slot_ids, non_blocking=True)


class MTPSampler(TorchSampler):
    """
    MTP sampler.
    """

    SampleState = SampleStateMTP

    def __init__(self, args: TorchSampler.Args, *, nextn: int):
        self.mapping = None
        self.draft_len = nextn
        super().__init__(args)

    @dataclass(frozen=True, kw_only=True)
    class Store(TorchSampler.Store):
        next_new_tokens: torch.Tensor
        next_draft_tokens: torch.Tensor
        new_tokens_lens: torch.Tensor
        max_total_draft_tokens: torch.Tensor

    def create_store(self) -> Store:
        num_tokens, seq_slots, _ = self.NEW_TOKENS_SHAPE
        draft_len = num_tokens - 1
        assert draft_len == self.draft_len
        return self.Store(
            new_tokens=int_tensor(self.NEW_TOKENS_SHAPE),
            next_new_tokens=int_tensor(self.NEW_TOKENS_SHAPE),
            next_draft_tokens=int_tensor((seq_slots, draft_len)),
            new_tokens_lens=int_tensor((seq_slots, )),
            max_total_draft_tokens=int_tensor((seq_slots, draft_len)),
        )

    def _request_common_handling(self, request: LlmRequest,
                                 next_draft_tokens: list[list[int]]):
        assert not request.py_return_context_logits, "return_context_logits not implemented for MTPSampler"
        assert not request.py_return_generation_logits, "return_generation_logits not implemented for MTPSampler"
        assert not request.py_return_log_probs, "return_log_probs not implemented for MTPSampler"
        request.py_draft_tokens = next_draft_tokens[request.py_seq_slot]
        request.py_decoding_iter += 1

    def update_requests(
            self,
            state: SampleStateMTP,
            resource_manager: Optional[BaseResourceManager] = None) -> None:
        # resource_manager will be not be used in this function
        assert isinstance(state, SampleStateMTP)

        state.sampler_event.synchronize()
        new_tokens = state.host.new_tokens.tolist()
        new_tokens_lens_list = state.host.new_tokens_lens.tolist()
        next_draft_tokens_list = state.host.next_draft_tokens.tolist()
        beam_idx = self.BEAM
        for req in state.scheduled_requests.context_requests:
            if req.state == LlmRequestState.GENERATION_COMPLETE or req.context_remaining_length != 0:
                continue
            new_token = add_token(req, new_tokens, beam=beam_idx)
            self._handle_stop_criteria(req, new_token)
            self._request_common_handling(req, next_draft_tokens_list)

        for req in state.scheduled_requests.generation_requests:
            if req.state == LlmRequestState.GENERATION_COMPLETE:
                continue
            num_new_tokens = new_tokens_lens_list[req.py_seq_slot]
            for i in range(num_new_tokens):
                new_token = add_token(req, new_tokens, beam=beam_idx, step=i)
                if self._handle_stop_criteria(req, new_token):
                    break
            req.py_num_accepted_draft_tokens = num_new_tokens - 1
            req.py_rewind_len = self.draft_len - req.py_num_accepted_draft_tokens
            self._request_common_handling(req, next_draft_tokens_list)

    def sample_async(
            self, scheduled_requests: ScheduledRequests,
            outputs: dict[str, torch.Tensor],
            num_context_logits_prefix_sum: list[int]) -> SampleStateMTP:
        # new_tokens_device: accepted tokens, device tensor, shape: batch_size, nextn + 1
        # new_tokens_lens_device: accepted lengths, device tensor, shape: batch_size
        # next_draft_tokens_device: predicted draft tokens, device tensor, shape: batch_size, nextn
        # next_new_tokens_device: input tokens for the next iteration, device tensor, shape: batch_size, nextn + 1

        requests = scheduled_requests.all_requests()
        slots = torch.as_tensor([r.py_seq_slot for r in requests])
        slots = slots.to(device="cuda", non_blocking=True)

        o_new_tokens = outputs['new_tokens'][:len(requests)]
        o_new_tokens_lens = outputs['new_tokens_lens'][:len(requests)]
        o_next_draft_tokens = outputs['next_draft_tokens'][:len(requests)]
        o_next_new_tokens = outputs['next_new_tokens'][:len(requests)]

        new_tokens = self.store.new_tokens
        next_new_tokens = self.store.next_new_tokens
        new_tokens_lens = self.store.new_tokens_lens
        next_draft_tokens = self.store.next_draft_tokens

        new_tokens.squeeze(-1).T.index_copy_(0, slots, o_new_tokens)
        next_new_tokens.squeeze(-1).T.index_copy_(0, slots, o_next_new_tokens)
        new_tokens_lens.index_copy_(0, slots, o_new_tokens_lens)
        next_draft_tokens.index_copy_(0, slots, o_next_draft_tokens)

        device = SampleStateTensorsMTP(
            new_tokens=next_new_tokens,
            new_tokens_lens=new_tokens_lens,
            next_draft_tokens=next_draft_tokens,
        )
        host = SampleStateTensorsMTP(
            new_tokens=new_tokens.to('cpu', non_blocking=True),
            new_tokens_lens=new_tokens_lens.to('cpu', non_blocking=True),
            next_draft_tokens=next_draft_tokens.to('cpu', non_blocking=True),
        )
        sampler_event = torch.cuda.Event()
        sampler_event.record()
        # add dummy draft tokens to context requests to prepare kv cache in advance
        # with the max draft token length
        for request in scheduled_requests.context_requests:
            request.py_draft_tokens = [1] * self.draft_len
        return SampleStateMTP(scheduled_requests=scheduled_requests,
                              device=device,
                              host=host,
                              sampler_event=sampler_event)


class MTPWorker(nn.Module):

    def __init__(self, spec_config: "MTPDecodingConfig", model_config=None):
        super().__init__()
        self.spec_config = spec_config
        self.model_config = model_config
        self.is_thop = False
        self.guided_decoder: Optional[CapturableGuidedDecoder] = None

    def forward(
        self,
        input_ids,
        position_ids,
        hidden_states,
        logits,
        attn_metadata,
        spec_metadata,
        draft_model,
    ):
        '''
        Example:
            Assume there are 3 MTP layers
            Notation:
                - H_t: token t's hidden state, generated by the target model
                - h_t: token t's hidden state, generated by the draft model

            Prompt: ABCD

            Context phase:
            Target model:
                - input tokens: ABCD + []
                - sampling tokens: E
                - accepted tokens: E
                - KV cache: ABCD
                - hidden states: H_A, H_B, H_C, H_D
            Draft model:
                MTP1:
                    # For context request, prompt[1:] + new generated goloden token is the input.
                    - input tokens: BCDE
                    - input hidden states: H_A, H_B, H_C, H_D
                    # '()' means historical KV cache
                    - KV cache: () + BCDE
                    - output hidden states: h_B, h_C, h_D, h_E
                    - output next draft token: F
                MTP2:
                    - input token: CDEF
                    - input hidden states: H_B, H_C, H_D, h_E
                    - KV cache: () + CDEF
                    - output hidden states: h_C, h_D, h_E, h_F
                    - output next draft token: G
                MTP3:
                    - input tokens: DEFG
                    - input hidden states: H_C, H_D, h_E, h_F
                    - KV cache: () + DEFG
                    - output hidden states: h_D, h_E, h_F, h_G
                    - output next draft token: H
                After 3 MTP layers:
                    - new generated draft tokens: FGH

            Generation phase 1: accept partial draft tokens
            Target model:
                - input tokens: E + FGH
                - sampling tokens: FGXY
                - accepted tokens: FGX
                - KV cache: (ABCD) + EFGH (H's KV cache is invalid)
                - hidden states: H_E, H_F, H_G, H_H (H_H is invalid)
            Draft model:
                MPT1:
                    # For generation request, `mtp_num_modules` of tokens will be used as input.
                    - input tokens: FGX
                    - input hidden states: H_E, H_F, H_G
                    - KV cache: (BCDE) + FGX
                    - output hidden states: h_F, h_G, h_X
                    - output next draft token: N
                MPT2:
                    - input tokens: GXN
                    - input hidden states: H_F, H_G, h_X
                    - KV cache: (CDEF) + GXN
                    - output hidden states: h_G, h_X, h_N
                    - output next draft token: O
                MPT3:
                    - input tokens: XNO
                    - input hidden states: H_G, H_X, h_N
                    - KV cache: (DEFG) + XNO
                    - output hidden states: h_X, h_N, h_O
                    - output next draft token: P
                After 3 MTP layers:
                    - new generated draft tokens: NOP

            Generation 2: accept none draft tokens
            Target model:
                - input tokens: X + NOP
                - sampling tokens: KMZY
                - accepted tokens: K
                - KV cache: (ABCDEFG) + NOP (NOP's KV cache is invalid)
                - hidden states: H_X, H_N, H_O, H_P (H_N, H_O, H_P is invalid)
            Draft model:
                MTP1:
                    - input tokens: GXK
                    - input hidden states: H_F, H_G, H_X
                    - KV cache: (BCDE + F) + GXK
                    - output hidden states: h_G, h_X, h_K
                    - output next draft token: U
                MTP2:
                    - input tokens: XKU
                    - input hidden states: H_G, H_X, h_K
                    - KV cache: (CDEF + G) + XKU
                    - output hidden states: h_X, h_K, h_U
                    - output next draft token: V
                MTP3:
                    - input tokens: KUV
                    - input hidden states: H_X, h_K, h_U
                    - KV cache: (DEFG + X) + KUV
                    - output hidden states: h_K, h_U, h_V
                    - output next draft token: Q
                After 3 MTP layers:
                    - new generated draft tokens: UVQ
        '''

        batch_size = attn_metadata.num_seqs

        raw_logits = logits

        if self.guided_decoder is not None:
            self.guided_decoder.execute(logits)

        # Sample and verify draft tokens
        accepted_tokens, num_accepted_tokens = self.sample_and_accept_draft_tokens(
            input_ids, logits, spec_metadata, attn_metadata)

        # Update MTP past hidden states
        self.update_mtp_hidden_states(input_ids=input_ids,
                                      hidden_states=hidden_states,
                                      num_accepted_tokens=num_accepted_tokens,
                                      spec_metadata=spec_metadata,
                                      attn_metadata=attn_metadata)

        # prepare draft layer inputs
        position_ids = position_ids.squeeze(0)
        draft_inputs = self.prepare_drafter_inputs(
            input_ids=input_ids,
            position_ids=position_ids,
            hidden_states=hidden_states,
            accepted_tokens=accepted_tokens,
            num_accepted_tokens=num_accepted_tokens,
            spec_metadata=spec_metadata,
            attn_metadata=attn_metadata)

        # update attn metadata
        if attn_metadata is not None:
            self.change_attn_metadata(num_accepted_tokens, attn_metadata)
            draft_inputs.update(attn_metadata=attn_metadata)

        # Run MTP layers to predict draft tokens
        next_draft_tokens = []
        last_tokens_idx = torch.cumsum(
            attn_metadata.seq_lens_cuda, dim=0, dtype=torch.long) - 1
        for i, mtp_layer in enumerate(draft_model.mtp_layers):
            if self.guided_decoder is not None:
                new_tokens = draft_inputs['input_ids'][last_tokens_idx]
                self.guided_decoder.add_draft_batch(new_tokens,
                                                    num_accepted_tokens,
                                                    draft_step=i)

            hidden_states = mtp_layer(embed_tokens=draft_model.embed_tokens,
                                      **draft_inputs)
            logits = mtp_layer.shared_head(hidden_states, draft_model.lm_head,
                                           attn_metadata).float()
            if self.guided_decoder is not None:
                self.guided_decoder.execute_draft_batch(logits, draft_step=i)

            new_draft_token = self.draft_sampler(logits)
            next_draft_tokens.append(new_draft_token)
            # shift input_ids and hidden_states
            input_ids = draft_inputs["input_ids"]
            input_ids[:-1] = input_ids[1:].clone()
            input_ids[last_tokens_idx] = new_draft_token
            draft_hidden_states = draft_inputs["hidden_states"]
            draft_hidden_states[:-1] = draft_hidden_states[1:].clone()
            draft_hidden_states[last_tokens_idx] = hidden_states[
                last_tokens_idx, :]
            draft_inputs = {
                "input_ids": input_ids,
                "position_ids": draft_inputs["position_ids"],
                "hidden_states": draft_hidden_states,
                "attn_metadata": draft_inputs["attn_metadata"],
            }
        next_draft_tokens = torch.stack(next_draft_tokens, dim=1)

        # restore attn metadata
        if attn_metadata is not None:
            self.restore_attn_metadata(attn_metadata=attn_metadata)

        # prepare next new tokens to support overlap scheduler
        next_new_tokens = accepted_tokens[
            spec_metadata.batch_indices_cuda[:batch_size],
            num_accepted_tokens - 1].unsqueeze(1)
        next_new_tokens = torch.concat([next_new_tokens, next_draft_tokens],
                                       dim=1)

        return {
            'logits': raw_logits,
            'new_tokens': accepted_tokens,
            'new_tokens_lens': num_accepted_tokens,
            'next_draft_tokens': next_draft_tokens,
            'next_new_tokens': next_new_tokens
        }

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
        mtp_num_modules = self.spec_config.num_nextn_predict_layers
        accepted_tokens = torch.empty((batch_size, (mtp_num_modules + 1)),
                                      dtype=torch.int,
                                      device=logits.device)
        num_accepted_tokens = torch.ones(batch_size,
                                         dtype=torch.int,
                                         device=logits.device)
        next_draft_tokens = torch.empty((batch_size, mtp_num_modules),
                                        dtype=torch.int,
                                        device=logits.device)
        next_new_tokens = torch.empty((batch_size, (mtp_num_modules + 1)),
                                      dtype=torch.int,
                                      device=logits.device)
        return {
            'logits': logits,
            'new_tokens': accepted_tokens,
            'new_tokens_lens': num_accepted_tokens,
            'next_draft_tokens': next_draft_tokens,
            'next_new_tokens': next_new_tokens
        }

    def update_mtp_hidden_states(
        self,
        input_ids: torch.IntTensor,
        hidden_states: torch.Tensor,
        num_accepted_tokens: torch.Tensor,
        spec_metadata: MTPSpecMetadata,
        attn_metadata: AttentionMetadata,
    ):
        '''
        Update the past hidden states and past tokens in spec_metadata base on
        the newly accepted tokens and historical hidden states.
        These past hidden states and past tokens will be use in MTP module.

        Args:
            input_ids: torch.IntTensor
                [num_tokens]
                The input ids of all requests. Flatten.

            hidden_states: torch.Tensor
                [num_tokens, hidden_size]
                Target model's hidden states.

            num_accepted_tokens: torch.Tensor
                [batch_size]
                Number of accepted tokens per request.

            spec_metadata: MTPSpecMetadata
                MTP speculative decoding metadata

            attn_metadata: AttentionMetadata
                Attention metadata

        Returns:
            None
        '''

        def unpack_sequence(packed_seq_cuda, seq_lens_cuda, seq_lens_cpu):
            # max_length is used as tensor shape, so it should be from host;
            # otherwise, an implicit D2H copy will be triggered.
            max_length = seq_lens_cpu.max().item()
            num_sequences = seq_lens_cuda.shape[0]
            # initialize a zero tensor to store the result
            result = torch.zeros(
                (num_sequences, max_length, packed_seq_cuda.shape[1]),
                dtype=packed_seq_cuda.dtype,
                device=packed_seq_cuda.device)
            # get mask
            seq_indices = torch.arange(
                max_length, device=seq_lens_cuda.device).unsqueeze(0).expand(
                    num_sequences, -1)
            mask = seq_indices < seq_lens_cuda.unsqueeze(1)
            # unpack
            result[mask] = packed_seq_cuda
            return result

        batch_size = attn_metadata.num_seqs
        num_contexts = attn_metadata.num_contexts
        num_ctx_tokens = attn_metadata.num_ctx_tokens
        num_gens = batch_size - num_contexts
        seq_lens = attn_metadata.seq_lens_cuda
        seq_lens_cpu = attn_metadata.seq_lens
        hidden_size = hidden_states.shape[-1]
        mtp_num_modules = self.spec_config.num_nextn_predict_layers

        if self.is_thop:
            _, _ = torch.ops.trtllm.mtp_update_hidden_states_op(
                input_ids, seq_lens, hidden_states,
                spec_metadata.mtp_hidden_states_ptrs,
                spec_metadata.mtp_past_tokens_ptrs, num_accepted_tokens,
                mtp_num_modules, batch_size, num_contexts, hidden_size)
        else:
            assert len(spec_metadata.request_ids) == batch_size
            mtp_past_hidden_states_pool = spec_metadata.mtp_hidden_states_manager.mtp_past_hidden_states_pool
            mtp_past_tokens_pool = spec_metadata.mtp_hidden_states_manager.mtp_past_tokens_pool

            slot_ids = spec_metadata.slot_ids[:batch_size]
            mtp_tokens = mtp_past_tokens_pool[slot_ids]
            mtp_hidden_states = mtp_past_hidden_states_pool[slot_ids]

            new_mtp_past_tokens, new_mtp_past_hidden_states = [], []
            # context
            if num_contexts > 0:
                seq_lens_ctx = seq_lens[:num_contexts]
                seq_lens_ctx_cpu = seq_lens_cpu[:num_contexts]
                unpacked_input_ids_ctx = unpack_sequence(
                    input_ids[:num_ctx_tokens].unsqueeze(1), seq_lens_ctx,
                    seq_lens_ctx_cpu).squeeze(2)
                unpacked_hidden_states_ctx = unpack_sequence(
                    hidden_states[:num_ctx_tokens], seq_lens_ctx,
                    seq_lens_ctx_cpu)
                cat_tokens_ctx = torch.cat(
                    (mtp_tokens[:num_contexts], unpacked_input_ids_ctx), dim=1)
                cat_hidden_states_ctx = torch.cat(
                    (mtp_hidden_states[:num_contexts],
                     unpacked_hidden_states_ctx),
                    dim=1)
                ctx_batch_idx = spec_metadata.batch_indices_cuda[:num_contexts]
                row_indices_ctx = ctx_batch_idx.unsqueeze(1).expand(
                    -1, mtp_num_modules)
                col_indices_ctx = (seq_lens_ctx.unsqueeze(1) +
                                   spec_metadata.draft_token_indices_cuda)
                new_mtp_past_tokens.append(cat_tokens_ctx[row_indices_ctx,
                                                          col_indices_ctx])
                new_mtp_past_hidden_states.append(
                    cat_hidden_states_ctx[row_indices_ctx, col_indices_ctx, :])

            # generation
            if num_gens > 0:
                unpacked_input_ids_gen = input_ids[num_ctx_tokens:].reshape(
                    num_gens, mtp_num_modules + 1).int()
                hidden_states_gen = hidden_states[num_ctx_tokens:, :]
                unpacked_hidden_states_gen = hidden_states_gen.reshape(
                    num_gens, mtp_num_modules + 1, hidden_size)
                cat_tokens_gen = torch.cat(
                    (mtp_tokens[num_contexts:], unpacked_input_ids_gen), dim=1)
                cat_hidden_states_gen = torch.cat(
                    (mtp_hidden_states[num_contexts:],
                     unpacked_hidden_states_gen),
                    dim=1)
                gen_batch_idx = spec_metadata.batch_indices_cuda[:num_gens]
                row_indices_gen = gen_batch_idx.unsqueeze(1).expand(
                    -1, mtp_num_modules)
                col_indices_gen = (
                    num_accepted_tokens[num_contexts:].unsqueeze(1) +
                    spec_metadata.draft_token_indices_cuda)
                new_mtp_past_tokens.append(cat_tokens_gen[row_indices_gen,
                                                          col_indices_gen])
                new_mtp_past_hidden_states.append(
                    cat_hidden_states_gen[row_indices_gen, col_indices_gen, :])

            # update past tokens and hidden states
            new_mtp_past_tokens = torch.cat(new_mtp_past_tokens, dim=0)
            new_mtp_past_hidden_states = torch.cat(new_mtp_past_hidden_states,
                                                   dim=0)
            mtp_past_tokens_pool.index_copy_(0, slot_ids, new_mtp_past_tokens)
            mtp_past_hidden_states_pool.index_copy_(0, slot_ids,
                                                    new_mtp_past_hidden_states)

    @torch.compile(options={"max-autotune": True})
    def topk_kernel(self, gen_logprobs, num_gens, mtp_num_modules,
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
    def process_generation_logits(self, logits, num_contexts):
        gen_logits = logits[num_contexts:]
        gen_logprobs = torch.softmax(gen_logits, dim=-1)
        return gen_logprobs

    def sample_and_accept_draft_tokens(
        self,
        input_ids: torch.IntTensor,
        logits: torch.Tensor,
        spec_metadata: MTPSpecMetadata,
        attn_metadata: AttentionMetadata,
    ):
        '''
        Takes input logits and samples golden token + predictions from draft tokens.
        Runs acceptance algorithm to accept draft tokens.
        Currently only support greedy sampling. All decoding is done using Top1 and token equality is used
        for acceptance.

        Args:
            input_ids: torch.IntTensor
                [num_tokens]
                The input ids of all requests. Flatten.

            logits: torch.Tensor
                [num_tokens, vocab_size]
                Logits produced by the target model.

            spec_metadata: MTPSpecMetadata
                MTP speculative decoding metadata

            attn_metadata: AttentionMetadata
                Attention metadata

        Returns:
            accepted_tokens: torch.Tensor
                [batch_size, (max_draft_len + 1)]
                Accepted token ids. Flattened.

            num_accepted_tokens: torch.Tensor
                [batch_size]
                Number of accepted tokens per request.

        Example:
            Assume there are 3 MTP layers
            Prompt: ABCD

            Context phase:
            Target model:
                - input tokens: ABCD + []
                - sampling tokens: E
                - accepted tokens: E
            Draft model:
                - input tokens: BCDE
                - new generated draft tokens: FGH
            Current sequence: ABCD E`FGH   -> Whitespace separates tokens produced by each phase
                                           -> Backtick separates accepted and draft tokens

            Generation phase 1:
            Target model:
                - input tokens: E + FGH
                - sampling tokens: FGXY    -> Sample with E's logit and get 'F'; Sample with F's logit, ...
                - accepted tokens: FGX     -> 'X' will be treat as the accepted token
            Draft model:
                - input tokens: FGX
                - new generated draft tokens: PQR
            Current sequence: ABCD EFG X`PQR

            Generation phase 2:
            Target model:
                - input tokens: X + PQR
                - sampling tokens: PYST
                - accepted token: PY
            Draft model:
                - input tokens: PY
                - new generated draft tokens: UVW
            Current sequence: ABCD EFG XP Y`UVW
        '''

        batch_size = attn_metadata.num_seqs
        num_contexts = attn_metadata.num_contexts
        num_gens = batch_size - num_contexts
        mtp_num_modules = self.spec_config.num_nextn_predict_layers

        if logits.dim() == 1:
            logits = logits.unsqueeze(0)

        # The return buffer
        if self.spec_config.use_relaxed_acceptance_for_thinking or not self.is_thop:
            accepted_tokens = torch.ones((batch_size, (mtp_num_modules + 1)),
                                         dtype=torch.int,
                                         device=logits.device)
            num_accepted_tokens = torch.ones(batch_size,
                                             dtype=torch.int,
                                             device=logits.device)
        if self.spec_config.use_relaxed_acceptance_for_thinking:
            mtp_relaxed_delta_pool = spec_metadata.mtp_hidden_states_manager.mtp_relaxed_delta_pool

            # context
            con_logits = logits[:num_contexts]
            con_target_tokens = torch.argmax(con_logits, dim=-1)
            accepted_tokens[:num_contexts, 0] = con_target_tokens[:num_contexts]
            last_tokens_idx = torch.cumsum(
                attn_metadata.seq_lens_cuda, dim=0, dtype=torch.long) - 1
            ctx_input_ids = input_ids[:attn_metadata.num_ctx_tokens]
            ctx_is_think = (ctx_input_ids ==
                            self.spec_config.BEGIN_THINKING_PHASE_TOKEN).int()
            ctx_is_think_cumsum = torch.cumsum(ctx_is_think, dim=0)
            ctx_last_cumsum = ctx_is_think_cumsum[
                last_tokens_idx[:num_contexts]]
            ctx_think_tokens_num = torch.diff(
                ctx_last_cumsum,
                dim=0,
                prepend=torch.zeros(1,
                                    dtype=torch.int,
                                    device=ctx_last_cumsum.device))

            ctx_delta = (ctx_think_tokens_num
                         >= 1).int() * self.spec_config.relaxed_delta
            ctx_slot_ids = spec_metadata.slot_ids[:num_contexts]
            mtp_relaxed_delta_pool.index_copy_(0, ctx_slot_ids, ctx_delta)

            # generation
            gen_logprobs = self.process_generation_logits(logits, num_contexts)
            topk_value, topk_indices, draft_tokens = self.topk_kernel(
                gen_logprobs, num_gens, mtp_num_modules, spec_metadata)

            accepted_tokens, num_accepted_tokens = torch.ops.trtllm.mtp_relaxed_acceptance_op(
                spec_metadata.slot_ids, topk_value, topk_indices, draft_tokens,
                mtp_relaxed_delta_pool, num_accepted_tokens, accepted_tokens,
                mtp_num_modules, batch_size, num_contexts,
                self.spec_config.relaxed_topk, self.spec_config.relaxed_delta,
                self.spec_config.BEGIN_THINKING_PHASE_TOKEN,
                self.spec_config.END_THINKING_PHASE_TOKEN)

        # Strict acceptance
        else:
            if self.is_thop:
                # Temporary buffer
                target_tokens_cache = torch.zeros(batch_size *
                                                  (mtp_num_modules + 1),
                                                  dtype=torch.int,
                                                  device=logits.device)
                accepted_tokens, num_accepted_tokens = torch.ops.trtllm.mtp_sampling_and_accepted_draft_tokens_op(
                    logits, spec_metadata.draft_tokens, target_tokens_cache,
                    mtp_num_modules, batch_size, num_contexts, logits.shape[-1])
            else:
                # Do greedy sampling for the input logits
                target_tokens = torch.argmax(logits, dim=-1)

                # context
                accepted_tokens[:num_contexts, 0] = target_tokens[:num_contexts]

                # generation
                gen_target_tokens = target_tokens[num_contexts:].reshape(
                    num_gens, mtp_num_modules + 1)
                accepted_tokens[num_contexts:, :] = gen_target_tokens
                draft_tokens = spec_metadata.draft_tokens.reshape(
                    num_gens, mtp_num_modules)
                num_accepted_tokens[num_contexts:] += torch.cumprod(
                    (draft_tokens == gen_target_tokens[:, :mtp_num_modules]
                     ).int(),
                    dim=-1).sum(1)

        return accepted_tokens, num_accepted_tokens

    def change_attn_metadata(self, num_accepted_tokens: torch.Tensor,
                             attn_metadata: AttentionMetadata):
        attn_metadata.prepare_for_spec_dec("_seq_lens", "_seq_lens_cuda")
        batch_size = attn_metadata.num_seqs
        mtp_num_modules = self.spec_config.num_nextn_predict_layers

        num_contexts = attn_metadata.num_contexts
        attn_metadata._seq_lens[num_contexts:batch_size] -= 1
        attn_metadata._seq_lens_cuda[num_contexts:batch_size] -= 1
        attn_metadata.on_update()

        if hasattr(attn_metadata, 'kv_lens_cuda'):
            # Note that it's important to not free the seq_lens_cuda
            # buffer once the graph has been captured also - this will invalidate
            # the graph and force an expensive recapture.
            attn_metadata.kv_lens_cuda[num_contexts:batch_size] -= (
                mtp_num_modules + 1 -
                num_accepted_tokens[num_contexts:batch_size])

        if attn_metadata.kv_cache_params is not None and not attn_metadata.is_cuda_graph:
            for i in range(num_contexts, batch_size):
                # used for vanilla MLA, list on cpu
                attn_metadata.kv_cache_params.num_cached_tokens_per_seq[
                    i] -= mtp_num_modules + 1 - num_accepted_tokens[i].item()

    def restore_attn_metadata(self, attn_metadata: AttentionMetadata):
        attn_metadata.restore_from_spec_dec()
        attn_metadata.on_update()

    def prepare_drafter_inputs(
        self,
        input_ids: torch.IntTensor,
        position_ids: torch.IntTensor,
        hidden_states: torch.Tensor,
        accepted_tokens: torch.Tensor,
        num_accepted_tokens: torch.Tensor,
        spec_metadata: MTPSpecMetadata,
        attn_metadata: AttentionMetadata,
    ):
        '''
        Parepare the input of the draft model.

        Args:
            input_ids: torch.IntTensor
                [num_tokens]
                The input ids of all requests. Flatten.
                num_tokens = sum(all prompts) + num_generation * (mtp_num_modules + 1)

            position_ids: torch.IntTensor
                [1][num_tokens]
                The position id of all requests. Flatten.

            hidden_states: torch.Tensor
                [num_tokens, hidden_size]
                Target model's hidden states.

            accepted_tokens: torch.Tensor
                [batch_size, max_draft_len + 1]
                Accepted token ids. Flattened.

            num_accepted_tokens: torch.Tensor
                [batch_size]
                Number of accepted draft tokens. Will be used for the first MTP layer.

            spec_metadata: MTPSpecMetadata
                MTP speculative decoding metadata

            attn_metadata: AttentionMetadata
                Attention metadata

        Returns: draft_inputs
            input_ids: torch.Tensor
                [num_tokens]
                The new input ids of all requests. Flatten.
                num_tokens = sum(all prompts) + num_generation * (mtp_num_modules)

            position_ids: torch.Tensor
                [1][[num_tokens]]
                The new position ids of all requests. Flatten.
                Directly use the input position ids.

            hidden_states: torch.Tensor
                [num_tokens][hidden_size]
                Continuous hidden states buffer.

            attn_metadata: AttentionMetadata
                Attention metadata

            spec_metadata: MTPSpecMetadata
                MTP speculative decoding metadata

        '''
        batch_size = attn_metadata.num_seqs
        num_contexts = attn_metadata.num_contexts
        num_ctx_tokens = attn_metadata.num_ctx_tokens
        num_gens = batch_size - num_contexts
        mtp_past_hidden_states_pool = spec_metadata.mtp_hidden_states_manager.mtp_past_hidden_states_pool
        mtp_past_tokens_pool = spec_metadata.mtp_hidden_states_manager.mtp_past_tokens_pool
        mtp_num_modules = self.spec_config.num_nextn_predict_layers

        if self.is_thop:
            # Temporary buffer
            hidden_size = hidden_states.shape[-1]

            # generation requests' golden tokens
            num_tokens = input_ids.shape[0] - num_gens
            return_input_ids = torch.empty(num_tokens,
                                           dtype=torch.int,
                                           device="cuda")

            return_hidden_states = torch.empty((num_tokens, hidden_size),
                                               dtype=hidden_states.dtype,
                                               device="cuda")

            (return_input_ids, return_hidden_states
             ) = torch.ops.trtllm.mtp_prepare_drafter_inputs_op(
                 input_ids, attn_metadata.seq_lens_cuda,
                 spec_metadata.mtp_hidden_states_ptrs,
                 spec_metadata.mtp_past_tokens_ptrs, hidden_states,
                 accepted_tokens, num_accepted_tokens, return_input_ids,
                 return_hidden_states, mtp_num_modules, batch_size,
                 num_contexts, hidden_size)

        else:
            return_input_ids_list = []
            return_hidden_states_list = []
            # Calculate cumulative sequence lengths for indexing
            last_tokens_idx = torch.cumsum(
                attn_metadata.seq_lens_cuda, dim=0, dtype=torch.long) - 1
            # context
            if num_contexts > 0:
                hidden_states_ctx = hidden_states[:num_ctx_tokens, :]
                input_prompt_ids = input_ids[:num_ctx_tokens]
                input_ids_ctx = torch.empty_like(input_prompt_ids,
                                                 dtype=torch.int32,
                                                 device="cuda")
                input_ids_ctx[:-1].copy_(input_prompt_ids[1:])
                input_ids_ctx[last_tokens_idx[:num_contexts]] = \
                    accepted_tokens[:num_contexts, 0]
                return_input_ids_list.append(input_ids_ctx)
                return_hidden_states_list.append(hidden_states_ctx)
            # generation
            if num_gens > 0:
                slot_ids = spec_metadata.slot_ids[num_contexts:batch_size]
                gen_batch_idx = spec_metadata.batch_indices_cuda[:num_gens]
                gen_token_idx = num_accepted_tokens[num_contexts:] - 1
                accepted_tokens_gen = accepted_tokens[num_contexts:, :]
                input_ids_gen = accepted_tokens_gen[gen_batch_idx,
                                                    gen_token_idx].unsqueeze(1)
                input_ids_gen = torch.concat(
                    [mtp_past_tokens_pool[slot_ids][:, 1:], input_ids_gen],
                    dim=1)
                hidden_states_gen = mtp_past_hidden_states_pool[
                    slot_ids].flatten(0, 1)
                return_input_ids_list.append(input_ids_gen.flatten(0, 1))
                return_hidden_states_list.append(hidden_states_gen)
            # Concatenate into continuous buffers
            return_input_ids = torch.concat(return_input_ids_list, dim=0)
            return_hidden_states = torch.concat(return_hidden_states_list,
                                                dim=0)

        # update position_ids
        position_ids_list = []
        if num_contexts > 0:
            position_ids_list.append(position_ids[:num_ctx_tokens])
        if num_gens > 0:
            position_ids_gen = position_ids[num_ctx_tokens:].reshape(
                num_gens, mtp_num_modules + 1)[:, -mtp_num_modules:]
            position_ids_gen = position_ids_gen - (
                1 + mtp_num_modules -
                num_accepted_tokens[num_contexts:].unsqueeze(1))
            position_ids_list.append(position_ids_gen.flatten())
        return_position_ids = torch.concat(position_ids_list, dim=-1)

        return {
            "input_ids": return_input_ids,
            "position_ids": return_position_ids,
            "hidden_states": return_hidden_states,
            "attn_metadata": attn_metadata,
        }

    @torch.compile(options={"max-autotune": True})
    def get_local_max_and_combined(self, logits, mapping_lm_tp=None):
        local_max_values, local_argmax = torch.max(logits, dim=-1, keepdim=True)
        # Adjust indices based on TP rank and size
        vocab_per_rank = logits.shape[-1]
        mapping_lm_tp = mapping_lm_tp if mapping_lm_tp is not None else self.model_config.mapping
        max_index_per_rank = local_argmax.type(
            torch.int32) + (mapping_lm_tp.tp_rank * vocab_per_rank)
        # Use torch.stack and flatten instead of view+cat to avoid torch.compile issues
        # Convert both to float32 to ensure consistent dtype
        max_index_per_rank_float = max_index_per_rank.float()
        local_max_values_float32 = local_max_values.float()

        # Stack and flatten to get interleaved layout: [idx0, val0, idx1, val1, ...]
        combined = torch.stack(
            [max_index_per_rank_float, local_max_values_float32],
            dim=-1).flatten(-2)
        return combined

    @torch.compile(options={"max-autotune": True})
    def get_draft_tokens_from_gathered(self, gathered):
        gathered_indices_float = gathered[..., 0::2]  # Even positions: indices
        gathered_values_float = gathered[..., 1::2]  # Odd positions: values

        # Find the rank with maximum value
        max_indices = torch.argmax(gathered_values_float, dim=-1, keepdim=True)

        # Get the corresponding token indices and convert back to int32
        draft_tokens = torch.gather(gathered_indices_float, -1,
                                    max_indices).squeeze(-1).type(torch.int32)
        return draft_tokens

    def draft_sampler(
        self,
        logits: torch.Tensor,
        mapping_lm_head_tp: Mapping = None,
    ):
        '''
        Sampling draft tokens.

        Args:
            logits: torch.Tensor
                [num_tokens, vocab_size]
                Logits produced by the draft model.

        Returns:
            draft_tokens: torch.Tensor
                [batch_size * max_draft_len]
                Draft token ids. Flattened.
        '''
        if (self.model_config is not None
                and hasattr(self.model_config, 'mapping')
                and self.model_config.mapping.tp_size
                > 1) and not (self.model_config.mapping.enable_attention_dp):
            combined = self.get_local_max_and_combined(logits)
            gathered = allgather(combined, self.model_config.mapping, dim=-1)
            draft_tokens = self.get_draft_tokens_from_gathered(gathered)
        elif (self.model_config is not None
              and hasattr(self.model_config, 'mapping')
              and self.model_config.mapping.tp_size
              > 1) and self.model_config.mapping.enable_lm_head_tp_in_adp:
            # For ADP + LM head TP mode, we need to find the global argmax across all TP ranks
            combined = self.get_local_max_and_combined(logits,
                                                       mapping_lm_head_tp)
            gathered = allgather(combined, mapping_lm_head_tp, dim=-1)
            batch_size = logits.shape[0]
            local_batch_size = batch_size // mapping_lm_head_tp.tp_size
            gathered = gathered.view(mapping_lm_head_tp.tp_size,
                                     local_batch_size, -1)
            sliced_gathered = gathered[mapping_lm_head_tp.tp_rank]
            draft_tokens = self.get_draft_tokens_from_gathered(sliced_gathered)
        else:
            # Simple argmax if no TP or no model config
            draft_tokens = torch.argmax(logits, dim=-1).type(torch.int32)

        return draft_tokens

    def set_guided_decoder(self,
                           guided_decoder: CapturableGuidedDecoder) -> bool:
        self.guided_decoder = guided_decoder
        return True


class MTPEagleWorker(MTPWorker):

    def __init__(self,
                 spec_config: "MTPDecodingConfig",
                 model_config: Optional[ModelConfig] = None):
        super().__init__(spec_config, model_config)
        self.model_config = model_config
        self.mtp_num_modules = spec_config.num_nextn_predict_layers

    @torch.compile(options={"max-autotune": True})
    def update_draft_tokens(self, next_draft_tokens, new_draft_token,
                            hidden_states, gather_ids, inputs):
        next_draft_tokens.append(new_draft_token)
        # update inputs
        hidden_states = hidden_states[gather_ids]
        position_ids = inputs["position_ids"][gather_ids] + 1
        return hidden_states, position_ids

    def forward(
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
        num_contexts = attn_metadata.num_contexts
        num_gens = batch_size - num_contexts

        raw_logits = logits

        if self.guided_decoder is not None:
            self.guided_decoder.execute(logits)

        # Sample and verify draft tokens
        accepted_tokens, num_accepted_tokens = self.sample_and_accept_draft_tokens(
            input_ids, logits, spec_metadata, attn_metadata)

        # Save the old attn_metadata and spec_metadata
        attn_metadata.prepare_for_spec_dec("_seq_lens", "_seq_lens_cuda")

        # Prepare inputs for the 1st MTP layer
        @torch.compile(options={"max-autotune": True})
        def prepare_position_ids_and_last_tokens(position_ids, attn_metadata):
            position_ids = position_ids.squeeze(0)
            last_tokens_idx = torch.cumsum(
                attn_metadata.seq_lens_cuda, dim=0, dtype=torch.long) - 1
            return position_ids, last_tokens_idx

        position_ids, last_tokens_idx = prepare_position_ids_and_last_tokens(
            position_ids, attn_metadata)
        inputs = self.prepare_drafter_inputs(input_ids=input_ids,
                                             position_ids=position_ids,
                                             last_tokens_idx=last_tokens_idx,
                                             hidden_states=hidden_states,
                                             accepted_tokens=accepted_tokens,
                                             attn_metadata=attn_metadata,
                                             spec_metadata=spec_metadata)

        # Predict draft tokens
        next_draft_tokens = []
        for i in range(self.mtp_num_modules):
            if i == 0:
                hidden_states = draft_model.mtp_layers[0](
                    embed_tokens=draft_model.embed_tokens,
                    all_rank_num_tokens=spec_metadata.all_rank_num_tokens,
                    **inputs)
                start_ids_gen = (spec_metadata.batch_indices_cuda[:num_gens] *
                                 (self.mtp_num_modules + 1)).long()
                gather_ids_gen = (start_ids_gen +
                                  num_accepted_tokens[num_contexts:] - 1 +
                                  attn_metadata.num_ctx_tokens)
                gather_ids = torch.concat(
                    [last_tokens_idx[:num_contexts], gather_ids_gen], dim=0)
            else:
                hidden_states = draft_model.mtp_layers[0](
                    embed_tokens=draft_model.embed_tokens,
                    all_rank_num_tokens=spec_metadata.
                    subseq_all_rank_num_tokens,
                    **inputs)
                # All of the seq_len are 1, use batch_indices_cuda as gather_ids
                gather_ids = spec_metadata.batch_indices_cuda[:batch_size]

            if self.guided_decoder is not None:
                new_tokens = inputs["input_ids"][gather_ids]
                self.guided_decoder.add_draft_batch(new_tokens,
                                                    num_accepted_tokens,
                                                    draft_step=i)
            if self.model_config.mapping.enable_attention_dp and \
                getattr(self.model_config.mapping, 'enable_lm_head_tp_in_adp', False):
                hidden_states_gathered = hidden_states[gather_ids]
                token_count = hidden_states_gathered.view(
                    -1, hidden_states_gathered.shape[-1]).shape[0]
                max_num_requests = spec_metadata.max_num_requests
                pad_len = max_num_requests - token_count
                if pad_len > 0:
                    padded_hidden_states = F.pad(hidden_states_gathered.view(
                        -1, hidden_states_gathered.shape[-1]),
                                                 (0, 0, 0, pad_len),
                                                 mode="constant",
                                                 value=0)
                elif pad_len == 0:
                    padded_hidden_states = hidden_states_gathered.view(
                        -1, hidden_states_gathered.shape[-1])
                else:
                    raise ValueError(
                        f"In MTPEagleWorker.forward(), token_count < max_num_requests, which is not supported"
                    )
                logits = draft_model.mtp_layers[0].shared_head(
                    padded_hidden_states, draft_model.lm_head, attn_metadata,
                    True)
            else:
                logits = draft_model.mtp_layers[0].shared_head(
                    hidden_states[gather_ids], draft_model.lm_head,
                    attn_metadata, True)
            if self.guided_decoder is not None:
                self.guided_decoder.execute_draft_batch(logits, draft_step=i)

            if self.model_config.mapping.enable_attention_dp and \
                getattr(self.model_config.mapping, 'enable_lm_head_tp_in_adp', False):
                mapping_lm_head_tp = draft_model.mtp_layers[
                    0].shared_head.mapping_lm_head_tp
                new_draft_token = self.draft_sampler(logits, mapping_lm_head_tp)
                new_draft_token = new_draft_token[:token_count]
            else:
                new_draft_token = self.draft_sampler(logits)

            hidden_states, position_ids = self.update_draft_tokens(
                next_draft_tokens, new_draft_token, hidden_states, gather_ids,
                inputs)
            # update attn_metadata
            if i == 0:
                attn_metadata._seq_lens[:batch_size].fill_(1)
                attn_metadata._seq_lens_cuda[:batch_size].fill_(1)
                attn_metadata.on_update()
                # cannot run generation if their is no kv cache
                has_kv_cache = inputs[
                    "attn_metadata"].kv_cache_manager is not None
                if has_kv_cache:
                    attn_metadata.host_request_types[:attn_metadata.
                                                     num_contexts].fill_(1)
                    attn_metadata.num_contexts = 0
                # update kv_lens_cuda
                if hasattr(attn_metadata, 'kv_lens_cuda'):
                    attn_metadata.kv_lens_cuda[num_contexts:batch_size] -= (
                        self.mtp_num_modules -
                        num_accepted_tokens[num_contexts:])
                    attn_metadata.kv_lens_cuda[:num_contexts] += 1
                # update metadata for flash mla
                if has_kv_cache and num_contexts > 0 and attn_metadata.enable_flash_mla:
                    reorder_block_ids_per_seq = torch.cat([
                        attn_metadata.
                        kv_block_ids_per_seq[num_contexts:batch_size],
                        attn_metadata.kv_block_ids_per_seq[:num_contexts]
                    ])
                    attn_metadata.block_ids_per_seq[:batch_size, :].copy_(
                        reorder_block_ids_per_seq, non_blocking=True)
                # update metadata
                # some attention metadata needs to be updated when changing seq_lens/kv_lens
                attn_metadata.update_for_spec_dec()
            elif hasattr(attn_metadata, 'kv_lens_cuda'):

                @torch.compile(options={"max-autotune": True})
                def update_kv_lens(kv_lens_cuda, batch_size):
                    kv_lens_cuda[:batch_size] += 1

                update_kv_lens(attn_metadata.kv_lens_cuda, batch_size)
                # update metadata
                # some attention metadata needs to be updated when changing kv_lens
                attn_metadata.update_for_spec_dec()
            inputs = {
                "input_ids": new_draft_token,
                "position_ids": position_ids,
                "hidden_states": hidden_states,
                "attn_metadata": attn_metadata,
            }

        # restore attn_metadata to support cuda graph
        attn_metadata.restore_from_spec_dec()
        attn_metadata.on_update()

        @torch.compile(options={"max-autotune": True})
        def prepare_next_tokens(next_draft_tokens, accepted_tokens,
                                spec_metadata, batch_size, num_accepted_tokens):
            next_draft_tokens = torch.stack(next_draft_tokens, dim=1)
            # prepare next new tokens to support overlap scheduler
            next_new_tokens = accepted_tokens[
                spec_metadata.batch_indices_cuda[:batch_size],
                num_accepted_tokens - 1].unsqueeze(1)
            next_new_tokens = torch.concat([next_new_tokens, next_draft_tokens],
                                           dim=1)
            return next_draft_tokens, next_new_tokens

        next_draft_tokens, next_new_tokens = prepare_next_tokens(
            next_draft_tokens, accepted_tokens, spec_metadata, batch_size,
            num_accepted_tokens)

        return {
            'logits': raw_logits,
            'new_tokens': accepted_tokens,
            'new_tokens_lens': num_accepted_tokens,
            'next_draft_tokens': next_draft_tokens,
            'next_new_tokens': next_new_tokens
        }

    @torch.compile(options={"max-autotune": True})
    def prepare_drafter_inputs(
        self,
        input_ids: torch.IntTensor,
        position_ids: torch.IntTensor,
        last_tokens_idx: torch.LongTensor,
        hidden_states: torch.Tensor,
        accepted_tokens: torch.Tensor,
        attn_metadata: AttentionMetadata,
        spec_metadata: MTPSpecMetadata,
    ):
        num_contexts = attn_metadata.num_contexts

        # context
        input_prompt_ids = input_ids[:attn_metadata.num_ctx_tokens]
        input_ids_ctx = torch.empty_like(input_prompt_ids,
                                         dtype=torch.int32,
                                         device="cuda")
        input_ids_ctx[:-1].copy_(input_prompt_ids[1:])
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
        }
