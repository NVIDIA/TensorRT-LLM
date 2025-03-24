from dataclasses import dataclass
from typing import Dict, Optional

import torch
from torch import nn

from ..attention_backend import AttentionMetadata
from ..pyexecutor.decoder import TorchDecoder
from ..pyexecutor.llm_request import *
from ..pyexecutor.llm_request import LlmRequest
from ..pyexecutor.resource_manager import BaseResourceManager, SlotManager
from ..pyexecutor.scheduler import ScheduledRequests
from .interface import SpecConfig, SpecMetadata, SpeculativeDecodingMode


@dataclass
class MTPConfig(SpecConfig):
    """
    Configuration for MTP.
    """
    # The name of speculative decoding.
    spec_dec_name = "MTP"
    # The number of MTP modules
    num_nextn_predict_layers: int = 1
    # The number of max batch size
    max_batch_size: int = 8

    def __post_init__(self) -> None:
        self.spec_dec_mode = SpeculativeDecodingMode.from_string(
            self.spec_dec_name)
        self.max_draft_tokens = self.num_nextn_predict_layers

    def update_from_model_config(self, model_config):
        assert self.num_nextn_predict_layers > 0
        if model_config.num_nextn_predict_layers == 1:
            self.spec_dec_mode = SpeculativeDecodingMode.MTP_EAGLE
            self.num_extra_kv_tokens = self.num_nextn_predict_layers - 1


class MTPHiddenStatesManager(BaseResourceManager):

    def __init__(self, config: MTPConfig, dtype: torch.dtype, hidden_size: int,
                 max_num_requests: int):
        self.dtype = dtype
        self.num_nextn_predict_layers = config.num_nextn_predict_layers
        self.hidden_size = hidden_size
        self.max_num_requests = max_num_requests
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

    def prepare_resources(self, scheduled_batch: ScheduledRequests):
        context_batch = scheduled_batch.context_requests
        # allocate hidden state tensors
        for req in context_batch:
            if req.is_first_context_chunk():
                self.slot_manager.add_slot(req.request_id)

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
class MTPSpecMetadata(SpecMetadata):
    """
    Metadata for MTP.
    """
    # The number of MTP modules in the model
    mtp_num_modules: int = 1,
    # The hidden states manager for MTP
    mtp_hidden_states_manager: Optional[MTPHiddenStatesManager] = None
    # The slot ids for each request.
    slot_ids: Optional[torch.Tensor] = None
    # The index of the batche inputs
    batch_indices_cuda: Optional[torch.Tensor] = None

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

    def prepare(self):
        assert self.request_ids is not None
        num_seqs = len(self.request_ids)
        # update batch indeices
        batch_indices = torch.arange(num_seqs, dtype=torch.int, device='cpu')
        self.batch_indices_cuda[:num_seqs].copy_(batch_indices,
                                                 non_blocking=True)
        # MTP module need different number of input tokens in generation phase
        if self.spec_dec_mode.is_mtp_eagle():
            self.num_tokens -= (self.num_generations) * self.mtp_num_modules
        else:
            self.num_tokens -= self.num_generations
        # update mtp hidden states and past tokens
        if self.mtp_hidden_states_manager is not None:
            mtp_hidden_states_ptrs = []
            mtp_past_tokens_ptrs = []
            mtp_slot_ids = []
            for rid in self.request_ids:
                slot_id = self.mtp_hidden_states_manager.slot_manager.get_slot(
                    rid)
                mtp_hidden_states_ptrs.append(
                    self.mtp_hidden_states_manager.
                    mtp_past_hidden_states_pool[slot_id].data_ptr())
                mtp_past_tokens_ptrs.append(
                    self.mtp_hidden_states_manager.
                    mtp_past_tokens_pool[slot_id].data_ptr())
                mtp_slot_ids.append(slot_id)
            mtp_hidden_states_ptrs = torch.tensor(mtp_hidden_states_ptrs,
                                                  dtype=torch.int64)
            mtp_past_tokens_ptrs = torch.tensor(mtp_past_tokens_ptrs,
                                                dtype=torch.int64)
            mtp_slot_ids = torch.tensor(mtp_slot_ids, dtype=torch.int)

            self.mtp_hidden_states_ptrs[:num_seqs].copy_(mtp_hidden_states_ptrs,
                                                         non_blocking=True)
            self.mtp_past_tokens_ptrs[:num_seqs].copy_(mtp_past_tokens_ptrs,
                                                       non_blocking=True)
            self.slot_ids[:num_seqs].copy_(mtp_slot_ids, non_blocking=True)


class MTPDecoder(TorchDecoder):
    """
    MTP decoder.
    """

    def __init__(self, max_seq_len: int, config: MTPConfig):
        super().__init__(max_seq_len, False)
        self.mapping = None
        self.draft_len = config.num_nextn_predict_layers

    def _draft_meet_max_token_stop_criteria(self, request: LlmRequest,
                                            num_tokens: int, beam_idx: int):
        if self._meet_max_token_stop_criteria(request,
                                              num_tokens + self.draft_len):
            request.state = LlmRequestState.GENERATION_COMPLETE
            request.set_finished_reason(tllm_executor.FinishReason.LENGTH,
                                        beam_idx)

    def update_requests(self, scheduled_requests: ScheduledRequests,
                        new_tensors_host: Dict[str, torch.Tensor],
                        decoder_event: torch.cuda.Event):
        decoder_event.synchronize()
        new_tokens_list = new_tensors_host["new_tokens_host"].tolist()
        new_tokens_lens_list = new_tensors_host["new_tokens_lens_host"].tolist()
        next_draft_tokens_list = new_tensors_host[
            "next_draft_tokens_host"].tolist()

        idx = 0
        beam_idx = 0
        for request in scheduled_requests.context_requests:
            if request.get_context_remaining_length() != 0:
                idx += 1
                continue

            if request.state != LlmRequestState.GENERATION_COMPLETE:
                new_token = new_tokens_list[idx][0]
                num_tokens = request.add_new_token(new_token, beam_idx)
                should_stop = self._handle_stop_criteria(
                    request, new_token, num_tokens, beam_idx)
                if self._draft_meet_max_token_stop_criteria(
                        request, num_tokens, beam_idx):
                    should_stop = True
                if not should_stop:
                    request.py_draft_tokens = next_draft_tokens_list[idx]
            idx += 1

        for request in scheduled_requests.generation_requests:
            if request.state != LlmRequestState.GENERATION_COMPLETE:
                new_tokens = new_tokens_list[idx]
                num_new_tokens = new_tokens_lens_list[idx]
                should_stop = False
                for i in range(num_new_tokens):
                    new_token = new_tokens[i]
                    num_tokens = request.add_new_token(new_token, beam_idx)
                    should_stop = self._handle_stop_criteria(
                        request, new_token, num_tokens, beam_idx)
                    if should_stop:
                        break
                if self._draft_meet_max_token_stop_criteria(
                        request, num_tokens, beam_idx):
                    should_stop = True
                if not should_stop:
                    request.py_draft_tokens = next_draft_tokens_list[idx]
                request.py_rewind_len = self.draft_len - (num_new_tokens - 1)
            idx += 1

    def decode_async(self, scheduled_requests: ScheduledRequests,
                     model_outputs):
        # new_tokens_device: all of the accepted tokens, device tensor
        # new_tokens_lens_device: the accepted lengths, device tensor
        # next_draft_tokens_device: predicted draft tokens, device tensor
        # next_new_tokens_device: input tokens for the next iteration, device tensor
        new_tokens_device = model_outputs[0]
        new_tokens_lens_device = model_outputs[1]
        next_draft_tokens_device = model_outputs[2]
        next_new_tokens_device = model_outputs[3]
        new_tokens_host = new_tokens_device.to('cpu', non_blocking=True)
        new_tokens_lens_host = new_tokens_lens_device.to('cpu',
                                                         non_blocking=True)
        next_draft_tokens_host = next_draft_tokens_device.to('cpu',
                                                             non_blocking=True)

        decoder_event = torch.cuda.Event()
        decoder_event.record()
        new_tensors_device = {
            "new_tokens_device": next_new_tokens_device,
            "new_tokens_lens_device": new_tokens_lens_device,
            "next_draft_tokens_device": next_draft_tokens_device,
        }
        new_tensors_host = {
            "new_tokens_host": new_tokens_host,
            "new_tokens_lens_host": new_tokens_lens_host,
            "next_draft_tokens_host": next_draft_tokens_host,
        }
        # add dummy draft tokens to context requests to prepare kv cache in advance
        # with the max draft token length
        for request in scheduled_requests.context_requests:
            request.py_draft_tokens = [1] * self.draft_len
        return new_tensors_device, new_tensors_host, decoder_event


class MTPWorker(nn.Module):

    def __init__(self, spec_config: MTPConfig):
        super().__init__()
        self.spec_config = spec_config
        self.is_thop = True

    def forward(
        self,
        input_ids,
        position_ids,
        hidden_states,
        logits,
        lm_head,
        embed_tokens,
        attn_metadata,
        spec_metadata,
        mtp_layers,
    ):
        batch_size = attn_metadata.num_seqs

        # Sample and verify draft tokens
        accepted_tokens, num_accepted_tokens = self.sample_and_accept_draft_tokens(
            logits, spec_metadata, attn_metadata)

        # Update MTP past hidden states
        self.update_mtp_hidden_states(input_ids=input_ids,
                                      target_model_hidden_states=hidden_states,
                                      num_accepted_tokens=num_accepted_tokens,
                                      accepted_tokens=accepted_tokens,
                                      spec_metadata=spec_metadata,
                                      attn_metadata=attn_metadata)

        # Predict draft tokens
        next_draft_tokens = []
        # will not be used in the first MTP, just a placeholder to avoid Nonetype
        previous_layer_draft_tokens = torch.empty(1,
                                                  dtype=torch.int,
                                                  device='cpu')
        for mtp_layer_idx, mtp_layer in enumerate(mtp_layers):
            draft_inputs = self.prepare_drafter_inputs(
                mtp_layer_idx=mtp_layer_idx,
                input_ids=input_ids,
                position_ids=position_ids,
                previous_layer_hidden_states=hidden_states,
                previous_layer_draft_tokens=previous_layer_draft_tokens,
                num_accepted_tokens=num_accepted_tokens,
                spec_metadata=spec_metadata,
                attn_metadata=attn_metadata)
            hidden_states, logits = mtp_layer(lm_head=lm_head,
                                              embed_tokens=embed_tokens,
                                              **draft_inputs)
            previous_layer_draft_tokens = self.draft_decoder(logits)
            next_draft_tokens.append(previous_layer_draft_tokens)

            input_ids = draft_inputs["input_ids"]
            position_ids = draft_inputs["position_ids"]
            attn_metadata = draft_inputs["attn_metadata"]
        next_draft_tokens = torch.stack(next_draft_tokens, dim=1)
        if attn_metadata.is_cuda_graph and attn_metadata is not None:
            self.restore_attn_metadata(num_accepted_tokens=num_accepted_tokens,
                                       attn_metadata=attn_metadata)

        # prepare next new tokens to support overlap scheduler
        next_new_tokens = accepted_tokens[
            spec_metadata.batch_indices_cuda[:batch_size],
            num_accepted_tokens - 1].unsqueeze(1)
        next_new_tokens = torch.concat([next_new_tokens, next_draft_tokens],
                                       dim=1)

        return accepted_tokens, num_accepted_tokens, next_draft_tokens, next_new_tokens

    def update_mtp_hidden_states(
        self,
        input_ids: torch.LongTensor,
        target_model_hidden_states: torch.Tensor,
        num_accepted_tokens: torch.Tensor,
        accepted_tokens: torch.Tensor,
        spec_metadata: MTPSpecMetadata,
        attn_metadata: AttentionMetadata,
    ):
        '''
        Update the past hidden states and past tokens in spec_metadata base on
        the newly accepted tokens and historical hidden states.
        These past hidden states and past tokens will be use in MTP module.
        Also update the seq_len and kv_lens in attention metadata.

        Args:
            input_ids: torch.LongTensor
                [num_tokens]
                The input ids of all requests. Flatten.

            target_model_hidden_states: torch.Tensor
                [num_tokens, hidden_size]
                Target model's hidden states.

            num_accepted_tokens: torch.Tensor
                [batch_size]
                Number of accepted tokens per request.

            accepted_tokens: torch.Tensor
                [batch_size, max_draft_tokens + 1]
                Accepted token ids. Flattened.

            spec_metadata: MTPSpecMetadata
                MTP speculative decoding metadata

            attn_metadata: AttentionMetadata
                Attention metadata

        Returns:
            None

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
                    - input tokens: BCDE                            # For context request, prompt[2: -1] + new generated goloden token is the input.
                    - input hidden states: H_A, H_B, H_C, H_D       # Therefore, the input and output tokens/hidden_states will have the dimension of `prompt_length`.
                    - KV cache: () + BCDE                           '()' means historical KV cache
                    - output hidden states: h_B, h_C, h_D, h_E
                    - output next draft token: F
                MTP2:
                    - input token: CDEF                             # Also with the `prompt_length`.
                    - input hidden states: h_B, h_C, h_D, h_E
                    - KV cache: () + CDEF
                    - output hidden states: h_C, h_D, h_E, h_F
                    - output next draft token: G
                MTP3:
                    - input tokens: DEFG
                    - input hidden states: h_C, h_D, h_E, h_F
                    - KV cache: () + DEFG
                    - output hidden states: h_D, h_E, h_F, h_G
                    - output next draft token: H
                After 3 MTP layers:
                    - input tokens: BCDE
                    - new generated draft tokens: FGH

            Generation phase 1: accept partial draft tokens
            Target model:
                - input tokens: E + FGH
                - sampling tokens: FGXY
                - accepted tokens: FGX
                - KV cache: (ABCD) + EFGH (H's KV cache is useless)
                - hidden states: H_E, H_F, H_G, H_H (H_H is useless)
            Draft model:
                MPT1:
                    - input tokens: FGX                            # For generation request, `mtp_num_modules` + 1 of tokens will be used as input.
                    - input hidden states: H_E, H_F, H_G
                    - KV cache: (BCDE) + FGX
                    - output hidden states: h_F, h_G, h_X
                    - output next draft token: N
                MPT2:
                    - input tokens: GXN
                    - input hidden states: h_F, h_G, h_X
                    - KV cache: (CDEF) + GXN
                    - output hidden states: h_G, h_X, h_N
                    - output next draft token: O
                MPT3:
                    - input tokens: XNO
                    - input hidden states: h_G, h_X, h_N
                    - KV cache: (DEFG) + XNO
                    - output hidden states: h_X, h_N, h_O
                    - output next draft token: P
                After 3 MTP layers:
                    - input tokens: FGX
                    - new generated draft tokens: NOP

            Generation 2: accept none draft tokens
            Target model:
                - input tokens: X + NOP
                - sampling tokens: KMZY
                - accepted tokens: K
                - KV cache: (ABCDEFG) + NOP (NOP's KV cache is useless)
                - hidden states: H_X, H_N, H_O, H_P (H_N, H_O, H_P is useless)
            Draft model:
                MTP1:
                    - input tokens: GXK
                    - input hidden states: H_F, H_G, H_X
                    - KV cache: (BCDE + FGX) + FGX
                    - output hidden states: h_G, h_X, h_K
                    - output next draft token: U
                MTP2:
                    - input tokens: XKU
                    - input hidden states: h_G, h_X, h_K
                    - KV cache: (CDEF + GXN) + XKU
                    - output hidden states: h_X, h_K, h_U
                    - output next draft token: V
                MTP3:
                    - input tokens: KUV
                    - input hidden states: h_X, h_K, h_U
                    - KV cache: (DEFG + XNO) + KUV
                    - output hidden states: h_K, h_U, h_V
                    - output next draft token: Q
                After 3 MTP layers:
                    - input tokens: GXK
                    - new generated draft tokens: UVQ
        '''
        batch_size = attn_metadata.num_seqs
        num_contexts = attn_metadata.num_contexts
        seq_lens = attn_metadata.seq_lens_cuda
        hidden_size = target_model_hidden_states.shape[-1]
        mtp_num_modules = self.spec_config.num_nextn_predict_layers

        if self.is_thop:
            _, _ = torch.ops.trtllm.mtp_update_hidden_states_op(
                input_ids, seq_lens, target_model_hidden_states,
                spec_metadata.mtp_hidden_states_ptrs,
                spec_metadata.mtp_past_tokens_ptrs, num_accepted_tokens,
                accepted_tokens, mtp_num_modules, batch_size, num_contexts,
                hidden_size)
        else:
            assert len(spec_metadata.request_ids) == batch_size
            mtp_past_hidden_states_pool = spec_metadata.mtp_hidden_states_manager.mtp_past_hidden_states_pool
            mtp_past_tokens_pool = spec_metadata.mtp_hidden_states_manager.mtp_past_tokens_pool

            input_ids_offset = 0

            for bix in range(batch_size):
                slot_id_cuda = spec_metadata.slot_ids[bix]
                # [num_nextn_predict_layers - 1, hidden_states]
                mtp_hidden_states = torch.index_select(
                    mtp_past_hidden_states_pool, 0, slot_id_cuda).squeeze(0)
                # [num_nextn_predict_layers]
                mtp_tokens = torch.index_select(mtp_past_tokens_pool, 0,
                                                slot_id_cuda).squeeze(0)
                cur_accepted_len = num_accepted_tokens[bix]
                cur_seq_len = seq_lens[bix]

                # Update MTP tokens
                cur_accepted_tokens = accepted_tokens[bix, 0:cur_accepted_len]
                if bix < num_contexts:
                    past_input_ids = input_ids[
                        input_ids_offset:input_ids_offset + cur_seq_len]
                else:
                    past_input_ids = mtp_tokens

                cat_past_tokens = torch.cat(
                    (past_input_ids, cur_accepted_tokens), dim=0)
                # shape: [mtp_num_modules]
                new_mtp_past_tokens = cat_past_tokens[
                    -mtp_num_modules:,
                ]
                # Update the buffer, but keep the pointer unchanged.
                mtp_past_tokens_pool.index_copy_(
                    0, slot_id_cuda, new_mtp_past_tokens.unsqueeze(0))

                # Update MTP hidden states
                past_hidden_states = mtp_hidden_states
                # For context, we need to slice prompt length
                # For generation, we only need to slice accepted length
                num_slice_tokens = cur_seq_len if bix < num_contexts else cur_accepted_len
                accepted_tokens_hidden_states = target_model_hidden_states[
                    input_ids_offset:input_ids_offset + num_slice_tokens]
                cat_hidden_states = torch.cat(
                    (past_hidden_states, accepted_tokens_hidden_states), dim=0)
                # shape: [mtp_num_modules, hidden_states]
                new_mtp_hidden_states = cat_hidden_states[
                    -mtp_num_modules:,
                ]
                # Update the buffer, but keep the pointer unchanged.
                mtp_past_hidden_states_pool.index_copy_(
                    0, slot_id_cuda, new_mtp_hidden_states.unsqueeze(0))

                # Update offset
                input_ids_offset += cur_seq_len

    def sample_and_accept_draft_tokens(
        self,
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
            logits: torch.Tensor
                [num_tokens, vocab_size]
                Logits produced by the target model.

            spec_metadata: MTPSpecMetadata
                MTP speculative decoding metadata

            attn_metadata: AttentionMetadata
                Attention metadata

        Returns:
            accepted_tokens: torch.Tensor
                [batch_size, (max_draft_tokens + 1)]
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
            Current sequence: ABCDE

            Generation phase 1:
            Target model:
                - input tokens: E + FGH
                - sampling tokens: FGXY  -> Sample with E's logit and get 'F'; Sample with F's logit, ...
                - accepted tokens: FGX   -> 'X' will be treat as the accepted token
            Draft model:
                - input tokens: FGX
                - new generated draft tokens: NOP
            Current sequence: ABCDEFGX

            Generation phase 2:
            Target model:
                - input tokens: X + NOP
                - sampling tokens: NYQC
                - accepted token: NY
            Draft model:
                - input tokens: NY
                - new generated draft tokens: XYZ
            Current sequence: ABCDEFGXNY
        '''

        batch_size = attn_metadata.num_seqs
        num_contexts = attn_metadata.num_contexts
        num_gens = batch_size - num_contexts
        mtp_num_modules = self.spec_config.num_nextn_predict_layers

        if logits.dim() == 1:
            logits = logits.unsqueeze(0)

        # The return buffer
        accepted_tokens = torch.empty((batch_size, (mtp_num_modules + 1)),
                                      dtype=torch.int,
                                      device=logits.device)
        num_accepted_tokens = torch.ones(batch_size,
                                         dtype=torch.int,
                                         device=logits.device)
        if self.is_thop:
            # Temporary buffer
            target_tokens_cache = torch.zeros(batch_size *
                                              (mtp_num_modules + 1),
                                              dtype=torch.int,
                                              device=logits.device)
            accepted_tokens, num_accepted_tokens = torch.ops.trtllm.mtp_sampling_and_accepted_draft_tokens_op(
                logits, spec_metadata.draft_tokens, target_tokens_cache,
                accepted_tokens, num_accepted_tokens, mtp_num_modules,
                batch_size, num_contexts, logits.shape[-1])
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
                (draft_tokens == gen_target_tokens[:, :mtp_num_modules]).int(),
                dim=-1).sum(1)

        return accepted_tokens, num_accepted_tokens

    def change_attn_metadata(self, num_accepted_tokens: torch.Tensor,
                             attn_metadata: AttentionMetadata):
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

    def restore_attn_metadata(self, num_accepted_tokens: torch.Tensor,
                              attn_metadata: AttentionMetadata):
        batch_size = attn_metadata.num_seqs
        mtp_num_modules = self.spec_config.num_nextn_predict_layers

        num_contexts = attn_metadata.num_contexts
        attn_metadata._seq_lens[num_contexts:batch_size] += 1
        attn_metadata._seq_lens_cuda[num_contexts:batch_size] += 1
        attn_metadata.on_update()

        if hasattr(attn_metadata, 'kv_lens_cuda'):
            # Note that it's important to not free the seq_lens_cuda
            # buffer once the graph has been captured also - this will invalidate
            # the graph and force an expensive recapture.
            attn_metadata.kv_lens_cuda[num_contexts:batch_size] += (
                mtp_num_modules + 1 -
                num_accepted_tokens[num_contexts:batch_size])

    def prepare_drafter_inputs(
        self,
        mtp_layer_idx: int,
        input_ids: torch.LongTensor,
        position_ids: torch.LongTensor,
        previous_layer_hidden_states: torch.Tensor,
        previous_layer_draft_tokens: torch.Tensor,
        num_accepted_tokens: torch.Tensor,
        spec_metadata: MTPSpecMetadata,
        attn_metadata: AttentionMetadata,
    ):
        '''
        Parepare the input of the draft model.

        Args:
            mtp_layer_idx: int
                The index number of the current MTP layer.

            input_ids: torch.LongTensor
                [num_tokens]
                The input ids of all requests. Flatten.
                When mtp_layer_idx == 0: num_tokens = sum(all prompts) + num_generation * (mtp_num_modules + 1)
                When mtp_layer_idx > 0: num_tokens = sum(all prompts) + num_generation * (mtp_num_modules)

            position_ids: torch.LongTensor
                [1][num_tokens]
                The position id of all requests. Flatten.

            previous_layer_hidden_states: torch.Tensor
                [num_tokens, hidden_size]
                Target model's hidden states.

            previous_layer_draft_tokens: torch.Tensor
                [batch_size]
                Privous layer's draft tokens.

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

        '''
        batch_size = attn_metadata.num_seqs
        num_contexts = attn_metadata.num_contexts
        num_gens = batch_size - num_contexts
        mtp_past_hidden_states_pool = spec_metadata.mtp_hidden_states_manager.mtp_past_hidden_states_pool
        mtp_past_tokens_pool = spec_metadata.mtp_hidden_states_manager.mtp_past_tokens_pool
        mtp_num_modules = self.spec_config.num_nextn_predict_layers

        if self.is_thop:
            # Temporary buffer
            hidden_size = previous_layer_hidden_states.shape[-1]

            # generation requests' golden tokens
            num_tokens = input_ids.shape[
                0] - num_gens if mtp_layer_idx == 0 else input_ids.shape[0]
            return_input_ids = torch.empty(num_tokens,
                                           dtype=torch.int,
                                           device="cuda")

            if (mtp_layer_idx == 0):
                return_hidden_states = torch.empty(
                    (num_tokens, hidden_size),
                    dtype=previous_layer_hidden_states.dtype,
                    device="cuda")
            else:
                return_hidden_states = torch.empty(
                    1, dtype=previous_layer_hidden_states.dtype,
                    device="cuda")  # Useless, placeholder

            (return_input_ids, return_hidden_states
             ) = torch.ops.trtllm.mtp_prepare_drafter_inputs_op(
                 input_ids, attn_metadata.seq_lens_cuda,
                 spec_metadata.mtp_hidden_states_ptrs,
                 spec_metadata.mtp_past_tokens_ptrs,
                 previous_layer_hidden_states, previous_layer_draft_tokens,
                 return_input_ids, return_hidden_states, mtp_num_modules,
                 mtp_layer_idx, batch_size, num_contexts, hidden_size)

        else:
            return_input_ids_list = []
            return_hidden_states_list = []
            if mtp_layer_idx == 0:  # The first MTP layer
                input_ids_offset = 0
                for bix in range(batch_size):
                    slot_id_cuda = spec_metadata.slot_ids[bix]
                    cur_seq_len = attn_metadata.seq_lens_cuda[bix]
                    past_tokens = torch.index_select(mtp_past_tokens_pool, 0,
                                                     slot_id_cuda).squeeze(0)
                    past_hidden_states = torch.index_select(
                        mtp_past_hidden_states_pool, 0, slot_id_cuda).squeeze(0)

                    if bix < num_contexts:
                        # Context request
                        # MTP past tokens
                        # cuda Graph should not run this part since has context request
                        prompt_tokens = input_ids[
                            input_ids_offset:input_ids_offset + cur_seq_len]
                        cat_tensor = torch.cat(
                            (prompt_tokens[1:], past_tokens[-1:]), dim=0)
                        return_input_ids_list.append(cat_tensor)

                        # MTP past hidden states
                        prompt_hidden_states = previous_layer_hidden_states[
                            input_ids_offset:input_ids_offset + cur_seq_len]
                        return_hidden_states_list.append(prompt_hidden_states)
                    else:
                        # Generation request
                        # Directly append
                        return_input_ids_list.append(past_tokens)
                        return_hidden_states_list.append(past_hidden_states)

                    # Update offset
                    input_ids_offset += cur_seq_len

                    # Concat into a continuous buffer
                return_input_ids = torch.cat(return_input_ids_list, dim=0)
                return_hidden_states = torch.cat(return_hidden_states_list,
                                                 dim=0)
            else:
                # this else part should be CUDA Graph supported
                input_ids_offset = 0
                for bix in range(batch_size):
                    # For the generation request, the 'cur_seq_len' already been update to 'num_nextn_predict_layers'.
                    cur_seq_len = attn_metadata.seq_lens_cuda[bix]

                    # The 'input_ids' come from the prvious layer
                    previous_layer_tokens = input_ids[
                        input_ids_offset:input_ids_offset + cur_seq_len]

                    # MTP past tokens
                    previous_draft_tokens = previous_layer_draft_tokens[bix:(
                        bix + 1)]
                    cat_tensor = torch.cat(
                        (previous_layer_tokens, previous_draft_tokens), dim=0)
                    return_input_ids_list.append(cat_tensor[1:])

                    # Update offset
                    input_ids_offset += cur_seq_len

                return_input_ids = torch.cat(return_input_ids_list, dim=0)
                # Directly use previous_layer_hidden_states as this layer's input hidden states
                return_hidden_states = previous_layer_hidden_states

        if mtp_layer_idx == 0 and attn_metadata is not None:
            self.change_attn_metadata(num_accepted_tokens, attn_metadata)

        return {
            "input_ids": return_input_ids,
            "position_ids": position_ids,
            "hidden_states": return_hidden_states,
            "attn_metadata": attn_metadata,
            "spec_metadata": spec_metadata,
        }

    def draft_decoder(
        self,
        logits: torch.Tensor,
    ):
        '''
        Sampling draft tokens.

        Args:
            logits: torch.Tensor
                [num_tokens, vocab_size]
                Logits produced by the draft model.

        Returns:
            draft_tokens: torch.Tensor
                [batch_size * max_draft_tokens]
                Draft token ids. Flattened.
        '''

        draft_tokens = torch.argmax(logits, dim=-1).type(torch.int32)
        return draft_tokens


class MTPEagleWorker(MTPWorker):

    def __init__(self, spec_config: MTPConfig):
        super().__init__(spec_config)
        self.mtp_num_modules = spec_config.num_nextn_predict_layers

    def forward(
        self,
        input_ids,
        position_ids,
        hidden_states,
        logits,
        lm_head,
        embed_tokens,
        attn_metadata,
        spec_metadata,
        mtp_layers,
    ):
        batch_size = attn_metadata.num_seqs
        num_contexts = attn_metadata.num_contexts

        # Sample and verify draft tokens
        accepted_tokens, num_accepted_tokens = self.sample_and_accept_draft_tokens(
            logits, spec_metadata, attn_metadata)

        # Save the old attn_metadata and spec_metadata
        if attn_metadata.is_cuda_graph:
            seq_len = attn_metadata._seq_lens[:batch_size].clone()
            seq_len_cuda = attn_metadata._seq_lens_cuda[:batch_size].clone()
            spec_all_rank_num_tokens = spec_metadata.all_rank_num_tokens
            req_types = attn_metadata.host_request_types[:batch_size].clone()
            if hasattr(attn_metadata, 'kv_lens_cuda'):
                kv_lens_cuda = attn_metadata.kv_lens_cuda[:batch_size].clone()

        # Prepare inputs for the 1st MTP layer
        position_ids = position_ids.squeeze(0)
        inputs = self.prepare_drafter_inputs(
            input_ids=input_ids,
            position_ids=position_ids,
            hidden_states=hidden_states,
            accepted_tokens=accepted_tokens,
            num_accepted_tokens=num_accepted_tokens,
            attn_metadata=attn_metadata,
            spec_metadata=spec_metadata)

        # Predict draft tokens
        next_draft_tokens = []
        for i in range(self.mtp_num_modules):
            hidden_states, logits = mtp_layers[0](lm_head=lm_head,
                                                  embed_tokens=embed_tokens,
                                                  **inputs)
            new_draft_token = self.draft_decoder(logits)
            next_draft_tokens.append(new_draft_token)
            # update inputs
            last_tokens = torch.cumsum(
                attn_metadata.seq_lens_cuda,
                dim=0,
                dtype=torch.long,
            ) - 1
            position_ids = inputs["position_ids"][last_tokens] + 1
            hidden_states = hidden_states[last_tokens]
            attn_metadata._seq_lens[:attn_metadata.num_contexts].fill_(1)
            attn_metadata._seq_lens_cuda[:attn_metadata.num_contexts].fill_(1)
            attn_metadata.on_update()
            # cannot run generation if their is no kv cache
            if inputs["attn_metadata"].kv_cache_manager is not None:
                attn_metadata.host_request_types[:attn_metadata.
                                                 num_contexts].fill_(1)
                attn_metadata.num_contexts = 0
            if hasattr(attn_metadata, 'kv_lens_cuda'):
                attn_metadata.kv_lens_cuda[:batch_size] += 1
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
        next_draft_tokens = torch.stack(next_draft_tokens, dim=1)

        # restore attn_metadata to support cuda graph
        if attn_metadata.is_cuda_graph:
            attn_metadata.num_contexts = num_contexts
            attn_metadata._seq_lens[:batch_size].copy_(seq_len)
            attn_metadata._seq_lens_cuda[:batch_size].copy_(seq_len_cuda)
            attn_metadata.on_update()
            attn_metadata.host_request_types[:batch_size].copy_(req_types)
            spec_metadata.all_rank_num_tokens = spec_all_rank_num_tokens
            if hasattr(attn_metadata, 'kv_lens_cuda'):
                attn_metadata.kv_lens_cuda[:batch_size].copy_(kv_lens_cuda)

        # prepare next new tokens to support overlap scheduler
        next_new_tokens = accepted_tokens[
            spec_metadata.batch_indices_cuda[:batch_size],
            num_accepted_tokens - 1].unsqueeze(1)
        next_new_tokens = torch.concat([next_new_tokens, next_draft_tokens],
                                       dim=1)

        return accepted_tokens, num_accepted_tokens, next_draft_tokens, next_new_tokens

    def prepare_drafter_inputs(
        self,
        input_ids: torch.LongTensor,
        position_ids: torch.LongTensor,
        hidden_states: torch.Tensor,
        accepted_tokens: torch.Tensor,
        num_accepted_tokens: torch.Tensor,
        attn_metadata: AttentionMetadata,
        spec_metadata: MTPSpecMetadata,
    ):
        batch_size = attn_metadata.num_seqs
        num_contexts = attn_metadata.num_contexts
        num_gens = batch_size - num_contexts
        num_ctx_tokens = attn_metadata.num_ctx_tokens
        hidden_size = hidden_states.shape[1]
        last_tokens_idx = torch.cumsum(
            attn_metadata.seq_lens_cuda, dim=0, dtype=torch.long) - 1

        # context
        hidden_states_ctx = hidden_states[:attn_metadata.num_ctx_tokens, :]
        input_ctx_ids = input_ids[:attn_metadata.num_ctx_tokens]
        input_ids_ctx = torch.empty_like(input_ctx_ids,
                                         dtype=torch.int32,
                                         device="cuda")
        input_ids_ctx[:-1].copy_(input_ctx_ids[1:])
        input_ids_ctx[
            last_tokens_idx[:num_contexts]] = accepted_tokens[:num_contexts, 0]
        position_ids_ctx = position_ids[:num_ctx_tokens]

        # generation
        gen_batch_idx = spec_metadata.batch_indices_cuda[:num_gens]
        gen_token_idx = num_accepted_tokens[num_contexts:] - 1
        hidden_states_gen = hidden_states[attn_metadata.num_ctx_tokens:, :]
        hidden_states_gen = hidden_states_gen.reshape(num_gens,
                                                      self.mtp_num_modules + 1,
                                                      hidden_size)
        hidden_states_gen = hidden_states_gen[gen_batch_idx, gen_token_idx, :]
        accepted_tokens_gen = accepted_tokens[num_contexts:, :]
        input_ids_gen = accepted_tokens_gen[gen_batch_idx, gen_token_idx]
        position_ids_gen = position_ids[num_ctx_tokens:].reshape(
            num_gens, self.mtp_num_modules + 1)
        position_ids_gen = position_ids_gen[gen_batch_idx, gen_token_idx]

        # get draft inputs
        input_ids = torch.concat([input_ids_ctx, input_ids_gen], dim=0)
        hidden_states = torch.concat([hidden_states_ctx, hidden_states_gen],
                                     dim=0)
        position_ids = torch.concat([position_ids_ctx, position_ids_gen], dim=0)

        # change attn_metadata
        attn_metadata._seq_lens[num_contexts:batch_size].fill_(1)
        attn_metadata._seq_lens_cuda[num_contexts:batch_size].fill_(1)
        attn_metadata.on_update()
        if hasattr(attn_metadata, 'kv_lens_cuda'):
            # Note that it's important to not free the seq_lens_cuda
            # buffer once the graph has been captured also - this will invalidate
            # the graph and force an expensive recapture.
            attn_metadata.kv_lens_cuda[num_contexts:batch_size] -= (
                self.mtp_num_modules + 1 - num_accepted_tokens[num_contexts:])

        return {
            "input_ids": input_ids,
            "position_ids": position_ids,
            "hidden_states": hidden_states,
            "attn_metadata": attn_metadata,
            "spec_metadata": spec_metadata,
        }
