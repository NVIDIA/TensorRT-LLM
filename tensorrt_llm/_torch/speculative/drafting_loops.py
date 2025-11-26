"""
This module contains capturable drafting loops for speculative decoding.

These are torch modules wrap another draft model. The wrapped module
is supposed to invoke the draft model autoregressively and invoke
a sampling algorithm to obtain draft tokens. By structuring the code
like this, we are able to avoid host overhead: the entire drafting process
for speculation can be launched as a single CUDA graph.
"""

from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Optional, final

import torch
import torch.nn as nn

from tensorrt_llm._torch.attention_backend.interface import AttentionMetadata
from tensorrt_llm._torch.speculative.eagle3 import Eagle3SpecMetadata
from tensorrt_llm._torch.speculative.interface import SpecMetadata
from tensorrt_llm._torch.speculative.spec_tree_manager import SpecTreeManager


class BaseDraftingLoopWrapper(ABC, torch.nn.Module):

    @abstractmethod
    def forward(self, input_ids: torch.Tensor, position_ids: torch.Tensor,
                attn_metadata: AttentionMetadata, spec_metadata: SpecMetadata,
                **kwargs) -> dict[str, torch.Tensor]:
        raise NotImplementedError

    @abstractmethod
    def sample(self,
               logits: torch.Tensor,
               max_top_k: Optional[int] = None) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def prepare_for_generation(
        self,
        attn_metadata: AttentionMetadata,
        spec_metadata: SpecMetadata,
        position_ids: torch.Tensor,
        spec_tree_manager: Optional[SpecTreeManager] = None
    ) -> torch.Tensor | None:
        raise NotImplementedError

    @final
    def load_weights_from_target_model(self, target_model) -> None:
        loader = getattr(self.draft_model, "load_weights_from_target_model",
                         None)
        if callable(loader):
            self.draft_model.load_weights_from_target_model(target_model)


@contextmanager
def save_metadata_state(attn_metadata: AttentionMetadata,
                        spec_metadata: SpecMetadata) -> None:
    attn_metadata.prepare_for_spec_dec("_seq_lens", "_seq_lens_cuda")
    batch_size = attn_metadata.num_seqs
    # Do not use prepare_for_spec_dec for this special field.
    # TRTLLM attention uses views of this tensor internally and prepare_for_spec_dec
    # creates a copy. If you write to the copy, TRTLLM attention won't see the updates.
    kv_lens = attn_metadata.kv_lens_cuda[:batch_size].clone()

    if attn_metadata.is_cuda_graph:
        assert spec_metadata.is_cuda_graph
        num_tokens = spec_metadata.num_tokens
        if isinstance(spec_metadata, Eagle3SpecMetadata):
            read_indices = spec_metadata.hidden_states_read_indices[:
                                                                    batch_size].clone(
                                                                    )
            write_indices = spec_metadata.hidden_states_write_indices[:
                                                                      batch_size].clone(
                                                                      )

    try:
        yield
    finally:
        attn_metadata.restore_from_spec_dec()
        attn_metadata.kv_lens_cuda[:batch_size].copy_(kv_lens)
        attn_metadata.on_update()
        if attn_metadata.is_cuda_graph:
            spec_metadata.num_tokens = num_tokens
            if isinstance(spec_metadata, Eagle3SpecMetadata):
                spec_metadata.hidden_states_read_indices[:batch_size].copy_(
                    read_indices)
                spec_metadata.hidden_states_write_indices[:batch_size].copy_(
                    write_indices)

        # This restore has to happen even if the spec_metadata is not being used
        # for CUDA graphs. It won't be reset by spec_metadata.prepare().
        if isinstance(spec_metadata, Eagle3SpecMetadata):
            spec_metadata.is_first_draft = True
            spec_metadata.eagle3_resource_manager.is_first_draft = True


class LinearDraftingLoopWrapper(BaseDraftingLoopWrapper):

    def __init__(self, max_draft_len: int, max_total_draft_tokens: int,
                 draft_model: torch.nn.Module):
        super().__init__()
        self.draft_model = draft_model
        self.config = self.draft_model.config
        self.model_config = self.draft_model.model_config
        self.max_draft_len = max_draft_len
        self.max_total_draft_tokens = max_total_draft_tokens

    def forward(self, input_ids: torch.Tensor, position_ids: torch.Tensor,
                attn_metadata: AttentionMetadata, spec_metadata: SpecMetadata,
                **kwargs) -> dict[str, torch.Tensor]:
        logits = self.draft_model.forward(input_ids=input_ids,
                                          position_ids=position_ids,
                                          attn_metadata=attn_metadata,
                                          spec_metadata=spec_metadata,
                                          return_context_logits=True)
        logits = logits[spec_metadata.gather_ids]

        new_draft_tokens = [self.sample(logits)]
        draft_logits = [logits]
        with save_metadata_state(attn_metadata, spec_metadata):
            batch_size = attn_metadata.num_seqs

            new_position_ids = self.prepare_for_generation(
                attn_metadata, spec_metadata, position_ids)
            for i in range(self.max_draft_len - 1):
                logits = self.draft_model.forward(
                    input_ids=new_draft_tokens[-1],
                    position_ids=new_position_ids,
                    attn_metadata=attn_metadata,
                    spec_metadata=spec_metadata)
                new_draft_tokens.append(self.sample(logits))
                draft_logits.append(logits)
                new_position_ids += 1
                attn_metadata.kv_lens_cuda[:batch_size] += 1
                if i == 0 and isinstance(spec_metadata, Eagle3SpecMetadata):
                    spec_metadata.hidden_states_read_indices[:batch_size].copy_(
                        spec_metadata.hidden_states_write_indices[:batch_size])

        return {
            "new_draft_tokens": torch.stack(new_draft_tokens),
            "draft_logits": torch.stack(draft_logits)
        }

    def sample(self, logits: torch.Tensor) -> torch.Tensor:
        # TODO: inject the sampler here so we can support non-greedy
        tokens = torch.argmax(logits, dim=-1)
        if hasattr(self.draft_model.model, "d2t"):
            d2t = self.draft_model.model.d2t.data
            return tokens + d2t[tokens]

        return tokens

    @torch.compile(options={'max-autotune': True})
    def prepare_for_generation(self, attn_metadata: AttentionMetadata,
                               spec_metadata: SpecMetadata,
                               position_ids: torch.Tensor) -> torch.Tensor:
        batch_size = attn_metadata.num_seqs
        num_accepted_draft_tokens = spec_metadata.num_accepted_draft_tokens[:
                                                                            batch_size]
        # Using attn_metadata.seq_lens_cuda[:batch_size] to get the max_draft_len + 1
        seq_lens = attn_metadata.seq_lens_cuda[:batch_size]
        attn_metadata.kv_lens_cuda[:
                                   batch_size] -= seq_lens - num_accepted_draft_tokens - 1

        # Calculate last accepted token indices
        last_tokens_idx = torch.cumsum(
            seq_lens, dim=0,
            dtype=torch.long) - seq_lens + num_accepted_draft_tokens
        new_position_ids = position_ids[0, last_tokens_idx] + 1

        attn_metadata._seq_lens[:batch_size].fill_(1)
        attn_metadata._seq_lens_cuda[:batch_size].fill_(1)
        attn_metadata.on_update()
        attn_metadata.kv_lens_cuda[:batch_size] += 1

        attn_metadata.host_request_types[:attn_metadata.num_contexts].fill_(1)
        attn_metadata.num_contexts = 0
        # The next inference of draft model will not use spec decoding and the number of input tokens is 1
        attn_metadata.use_spec_decoding = False

        spec_metadata.num_tokens = batch_size

        if isinstance(spec_metadata, Eagle3SpecMetadata):
            spec_metadata.eagle3_resource_manager.is_first_draft = False
            spec_metadata.is_first_draft = False

            old_write_indices = spec_metadata.hidden_states_write_indices

            spec_metadata.hidden_states_read_indices[:batch_size].copy_(
                old_write_indices[last_tokens_idx])
            spec_metadata.hidden_states_write_indices[:batch_size].copy_(
                torch.arange(
                    batch_size,
                    dtype=spec_metadata.hidden_states_write_indices.dtype,
                    device=spec_metadata.hidden_states_write_indices.device))

        return new_position_ids


class StaticTreeDraftingLoopWrapper(BaseDraftingLoopWrapper):

    def __init__(self, max_draft_len: int, max_total_draft_tokens: int,
                 max_batch_size: int, draft_model: torch.nn.Module):
        super().__init__()
        self.draft_model = draft_model
        self.config = self.draft_model.config
        self.model_config = self.draft_model.model_config
        self.max_draft_len = max_draft_len
        self.max_total_draft_tokens = max_total_draft_tokens
        self.max_batch_size = max_batch_size

        self.draft_tokens_buffer = torch.zeros(
            (max_batch_size, max_total_draft_tokens + 1),
            dtype=torch.int64,
            device='cuda')
        self.position_ids_buffer = torch.zeros(
            (max_batch_size, max_total_draft_tokens + 1),
            dtype=torch.int64,
            device='cuda')

    def forward(self, input_ids: torch.Tensor, position_ids: torch.Tensor,
                attn_metadata: AttentionMetadata, spec_metadata: SpecMetadata,
                **kwargs) -> dict[str, torch.Tensor]:
        assert isinstance(spec_metadata, Eagle3SpecMetadata)
        spec_tree_manager = spec_metadata.eagle3_resource_manager.spec_tree_manager

        logits = self.draft_model.forward(input_ids=input_ids,
                                          position_ids=position_ids,
                                          attn_metadata=attn_metadata,
                                          spec_metadata=spec_metadata,
                                          return_context_logits=True)
        batch_size = attn_metadata.num_seqs
        vocab_size = logits.shape[-1]
        logits = logits[spec_metadata.gather_ids]  # [batch_size, vocab_size]

        # new_draft_tokens: [batch_size * max_top_k]
        new_draft_tokens = self.sample(logits=logits,
                                       max_top_k=spec_tree_manager.max_top_k)

        self.extract_real_draft_tokens(
            cur_draft_idx=0,
            batch_size=batch_size,
            new_draft_tokens=new_draft_tokens,
            use_cuda_graph=attn_metadata.is_cuda_graph,
            spec_tree_manager=spec_tree_manager)

        return_draft_logits = None
        with save_metadata_state(attn_metadata, spec_metadata):
            batch_size = attn_metadata.num_seqs

            self.prepare_for_generation(attn_metadata=attn_metadata,
                                        spec_metadata=spec_metadata,
                                        spec_tree_manager=spec_tree_manager,
                                        position_ids=position_ids)

            for layer_idx in range(1, self.max_draft_len):
                # input_ids: [batch_size * (max_total_draft_tokens + 1)]
                # position_ids: [batch_size * (max_total_draft_tokens + 1)]
                # logits: [batch_size * (max_total_draft_tokens + 1), vocab_size]
                logits = self.draft_model.forward(
                    input_ids=self.draft_tokens_buffer[:batch_size, :].reshape(
                        -1),
                    position_ids=self.position_ids_buffer[:batch_size, :].
                    reshape(-1),
                    attn_metadata=attn_metadata,
                    spec_metadata=spec_metadata,
                    return_context_logits=True)

                # new_draft_tokens: [batch_size * (max_total_draft_tokens + 1) * max_top_k]
                new_draft_tokens = self.sample(
                    logits=logits, max_top_k=spec_tree_manager.max_top_k)
                # Keep updating
                self.extract_real_draft_tokens(
                    cur_draft_idx=layer_idx,
                    batch_size=batch_size,
                    new_draft_tokens=new_draft_tokens,
                    use_cuda_graph=attn_metadata.is_cuda_graph,
                    spec_tree_manager=spec_tree_manager)

                if layer_idx == self.max_draft_len - 1:
                    return_draft_logits = logits

        # self.draft_tokens_buffer[:batch_size, :]: [batch_size, max_total_draft_tokens + 1]
        # return_new_draft_tokens: [max_total_draft_tokens, batch_size]
        return_new_draft_tokens = torch.transpose(
            self.draft_tokens_buffer[:batch_size, :-1], 0, 1)

        # return_draft_logits: [batch_size, max_total_draft_tokens + 1, vocab_size] -> [max_total_draft_tokens, batch_size, vocab_size]
        return_draft_logits = return_draft_logits.reshape(
            batch_size, self.max_total_draft_tokens + 1, vocab_size)
        return_draft_logits = torch.transpose(return_draft_logits[:, :-1, :], 0,
                                              1)

        assert return_new_draft_tokens.shape == (self.max_total_draft_tokens,
                                                 batch_size)
        assert return_draft_logits.shape == (self.max_total_draft_tokens,
                                             batch_size, vocab_size)

        return {
            "new_draft_tokens": return_new_draft_tokens,
            "draft_logits": return_draft_logits
        }

    def sample(self, logits: torch.Tensor, max_top_k: int) -> torch.Tensor:
        # TODO: inject the sampler here so we can support non-greedy

        # for draft_layer_idx == 0, logits is of shape [batch_size, vocab_size]
        # for draft_layer_idx > 0, logits is of shape [batch_size * (max_total_draft_tokens + 1), vocab_size]
        indices = torch.topk(
            logits, k=max_top_k, dim=-1
        ).indices  # [batch_size, max_top_k] or [batch_size * max_total_draft_tokens, max_top_k]
        tokens = indices.reshape(-1)

        if hasattr(self.draft_model.model, "d2t"):
            d2t = self.draft_model.model.d2t.data
            tokens = tokens + d2t[tokens]

        return tokens

    def extract_real_draft_tokens(self, cur_draft_idx: int, batch_size: int,
                                  new_draft_tokens: torch.Tensor,
                                  use_cuda_graph: bool,
                                  spec_tree_manager: SpecTreeManager):
        '''
        Extract the real draft tokens from the new draft tokens to self.draft_tokens_buffer.
        '''
        # After the first drafter layer, new_draft_tokens: [batch_size * max_top_k]
        # For other drafter layers, new_draft_tokens: [batch_size * (max_total_draft_tokens + 1) * max_top_k]
        if cur_draft_idx == 0:
            assert new_draft_tokens.shape[0] == (batch_size *
                                                 spec_tree_manager.max_top_k)
        else:
            assert new_draft_tokens.shape[0] == (
                batch_size * (self.max_total_draft_tokens + 1) *
                spec_tree_manager.max_top_k)

        # reshape the new_draft_tokens to [batch_size, -1, spec_tree_manager.max_top_k]
        new_draft_tokens = new_draft_tokens.reshape(batch_size, -1,
                                                    spec_tree_manager.max_top_k)

        # If using cuda graph, we need to use a torch op to implement this logic
        if use_cuda_graph:
            torch.ops.trtllm.extract_real_draft_tokens_op(
                new_draft_tokens, self.draft_tokens_buffer, spec_tree_manager.
                tokens_gather_idx_for_drafter_model[cur_draft_idx],
                spec_tree_manager.top_k_list_cuda[cur_draft_idx],
                spec_tree_manager.draft_tokens_indices_cumsum, cur_draft_idx,
                batch_size, self.max_draft_len, self.max_total_draft_tokens,
                spec_tree_manager.max_top_k)
        else:
            # 1) Gather the real tokens processed by this layer
            process_tokens = new_draft_tokens[:, spec_tree_manager.
                                              tokens_gather_idx_for_drafter_model[
                                                  cur_draft_idx], :]  # [batch_size, num_tokens_process_this_layer, max_top_k]
            process_tokens = process_tokens.reshape(
                -1, spec_tree_manager.max_top_k
            )  # [batch_size * num_tokens_process_this_layer, max_top_k]

            # 2) Gather the real draft tokens samples by these processed tokens' logits
            top_k_list = spec_tree_manager.top_k_list_cuda[
                cur_draft_idx].repeat(
                    batch_size)  # [batch_size * num_tokens_process_this_layer]
            assert top_k_list.shape[0] == process_tokens.shape[0]

            # [batch_size * num_tokens_process_this_layer, spec_tree_manager.max_top_k]
            col_indices = torch.arange(
                spec_tree_manager.max_top_k,
                device=new_draft_tokens.device).unsqueeze(0).repeat(
                    top_k_list.shape[0], 1)

            mask = col_indices < top_k_list.unsqueeze(
                1
            )  # [batch_size * num_tokens_process_this_layer, spec_tree_manager.max_top_k]

            real_new_draft_tokens = process_tokens[
                mask]  # [batch_size * sum(spec_tree_manager.top_k_list_cuda[cur_draft_idx])]
            real_new_draft_tokens = real_new_draft_tokens.reshape(
                batch_size, -1
            )  # [batch_size, sum(spec_tree_manager.top_k_list_cuda[cur_draft_idx])]

            self.draft_tokens_buffer[:batch_size, spec_tree_manager.
                                     draft_tokens_indices_cumsum[cur_draft_idx]:
                                     spec_tree_manager.
                                     draft_tokens_indices_cumsum[
                                         cur_draft_idx +
                                         1]] = real_new_draft_tokens[:, :]

    def prepare_for_generation(self, attn_metadata: AttentionMetadata,
                               spec_metadata: SpecMetadata,
                               spec_tree_manager: SpecTreeManager,
                               position_ids: torch.Tensor):
        '''
        Prepare the inputs for the subsequent draft layers.
        Note: Except for the 0th drafter layer, in each subsequent drafter layer,
        we take 'max_total_drafter_tokens + 1' draft tokens as input.
        Only the first part of the draft tokens is meaningful, and the later tokens can be regarded as padding
        until we continuously write the correct value.

        This introduces additional redundant computation, but it makes it compatible with cuda graphs.

        What we need to prepare are:
            1) position_ids
            2) attn_metadata
                2.1) kv_lens_cuda
                2.2) _seq_lens, _seq_lens_cuda
                2.3) host_request_types
                2.4) num_contexts
                2.5) use_spec_decoding
                2.6) spec_decoding_position_offsets
                2.7) spec_decoding_packed_mask
                2.8) spec_decoding_generation_lengths
            3) spec_metadata
                3.1) num_tokens
                3.2) hidden_states_read_indices, hidden_states_write_indices
                3.3) is_first_draft
        '''
        batch_size = attn_metadata.num_seqs

        # 1) Prepare the position_ids
        num_accepted_draft_tokens = spec_metadata.num_accepted_draft_tokens[:
                                                                            batch_size]
        seq_lens = attn_metadata.seq_lens_cuda[:batch_size]
        # Calculate last accepted token indices
        last_tokens_idx = torch.cumsum(
            seq_lens, dim=0,
            dtype=torch.long) - seq_lens + num_accepted_draft_tokens
        position_start_idx = position_ids[0,
                                          last_tokens_idx] + 1  # [batch_size]
        self.position_ids_buffer[:batch_size, :-1] = position_start_idx.unsqueeze(
            1) + spec_tree_manager.spec_dec_position_offsets[0, 1:].unsqueeze(
                0) - 1  # exclude the root node

        # 2) Prepare the attn_metadata
        ## 2.1) kv_lens_cuda
        attn_metadata.kv_lens_cuda[:
                                   batch_size] -= seq_lens - num_accepted_draft_tokens - 1
        attn_metadata.kv_lens_cuda[:batch_size] += (
            self.max_total_draft_tokens + 1)

        ## 2.2) _seq_lens, _seq_lens_cuda
        attn_metadata._seq_lens[:batch_size].fill_(self.max_total_draft_tokens +
                                                   1)
        attn_metadata._seq_lens_cuda[:batch_size].fill_(
            self.max_total_draft_tokens + 1)
        attn_metadata.on_update()

        ## 2.3) host_request_types
        attn_metadata.host_request_types[:attn_metadata.num_contexts].fill_(1)

        ## 2.4) num_contexts
        attn_metadata.num_contexts = 0

        ## 2.5) use_spec_decoding
        attn_metadata.use_spec_decoding = True

        ## 2.6) spec_decoding_position_offsets
        ### attn_metadata.spec_decoding_position_offsets: [max_num_requests, max_total_draft_tokens + 1]
        attn_metadata.spec_decoding_position_offsets[:batch_size, :self.
                                                     max_total_draft_tokens] = spec_tree_manager.spec_dec_position_offsets[
                                                         0, 1:].unsqueeze(
                                                             0
                                                         ) - 1  # exclude the root node
        attn_metadata.spec_decoding_position_offsets[:batch_size, self.
                                                     max_total_draft_tokens] = 0  # padding

        ## 2.7) spec_decoding_packed_mask
        ### attn_metadata.spec_decoding_packed_mask: [max_num_requests, max_total_draft_tokens + 1, math.ceil((max_total_draft_tokens + 1) / 32)]
        attn_metadata.spec_decoding_packed_mask[:
                                                batch_size, :, :] = spec_tree_manager.spec_dec_packed_mask_for_drafter_model

        ## 2.8) spec_decoding_generation_lengths
        ### attn_metadata.spec_decoding_generation_lengths: [max_num_requests]
        attn_metadata.spec_decoding_generation_lengths[:
                                                       batch_size] = self.max_total_draft_tokens + 1

        # 3) Update spec_metadata
        ## 3.1) num_tokens
        spec_metadata.num_tokens = batch_size * (self.max_total_draft_tokens +
                                                 1)
        ## 3.2) hidden_states_read_indices, hidden_states_write_indices
        ### spec_metadata.hidden_states_read_indices: [self.max_num_tokens]
        ### spec_metadata.hidden_states_write_indices: [self.max_num_tokens]
        old_write_indices = spec_metadata.hidden_states_write_indices
        start_idx = old_write_indices[
            last_tokens_idx]  # [batch_size], already take the accepted tokens into account.

        ### shape: [batch_size, self.max_total_draft_tokens + 1]
        hidden_states_read_indices_offset = spec_tree_manager.hidden_states_read_indices_offset_for_drafter_model.repeat(
            batch_size).reshape(batch_size, self.max_total_draft_tokens + 1)
        hidden_states_read_indices_offset = hidden_states_read_indices_offset + start_idx.unsqueeze(
            1)
        spec_metadata.hidden_states_read_indices[:batch_size * (
            self.max_total_draft_tokens +
            1)] = hidden_states_read_indices_offset.reshape(-1)

        hidden_states_write_offset = torch.arange(
            1, self.max_total_draft_tokens + 1 + 1,
            device=position_ids.device).unsqueeze(0).repeat(
                batch_size, 1) + start_idx.unsqueeze(1)
        spec_metadata.hidden_states_write_indices[:batch_size * (
            self.max_total_draft_tokens +
            1)] = hidden_states_write_offset.reshape(-1)

        ## 3.3) is_first_draft
        spec_metadata.eagle3_resource_manager.is_first_draft = False
        spec_metadata.is_first_draft = False

        return

class DynamicTreeDraftingLoopWrapper(BaseDraftingLoopWrapper):

    def __init__(self, max_draft_len: int, max_total_draft_tokens: int,
                 max_batch_size: int, dynamic_tree_max_topK, draft_model: torch.nn.Module):
        super().__init__()
        self.draft_model = draft_model
        self.config = self.draft_model.config
        self.model_config = self.draft_model.model_config
        self.max_draft_len = max_draft_len
        self.max_total_draft_tokens = max_total_draft_tokens
        self.max_batch_size = max_batch_size
        self.dynamic_tree_max_topK = dynamic_tree_max_topK
        self.logsoftmax = nn.LogSoftmax(dim=-1)

        self.draft_tokens_buffer = torch.zeros(
            (max_batch_size, max_total_draft_tokens + 1),
            dtype=torch.int64,
            device='cuda')
        self.position_ids_buffer = torch.zeros(
            (max_batch_size, max_total_draft_tokens + 1),
            dtype=torch.int64,
            device='cuda')
        self.history_draft_tokens_buffer = torch.zeros(
            (max_batch_size, dynamic_tree_max_topK + dynamic_tree_max_topK * dynamic_tree_max_topK * (max_draft_len - 1)),
            dtype=torch.int64,
            device='cuda')
        self.history_score_buffer = torch.zeros(
            (max_batch_size, dynamic_tree_max_topK + dynamic_tree_max_topK * dynamic_tree_max_topK * (max_draft_len - 1)),
            dtype=torch.float32,
            device='cuda')
        self.history_draft_tokens_parent_buffer = torch.ones(
            (max_batch_size, dynamic_tree_max_topK + dynamic_tree_max_topK * dynamic_tree_max_topK * (max_draft_len - 1)),
            dtype=torch.int64,
            device='cuda') * -1

    def forward(self, input_ids: torch.Tensor, position_ids: torch.Tensor,
                attn_metadata: AttentionMetadata, spec_metadata: SpecMetadata,
                **kwargs) -> dict[str, torch.Tensor]:
        
        assert isinstance(spec_metadata, Eagle3SpecMetadata)
        spec_tree_manager = spec_metadata.eagle3_resource_manager.spec_tree_manager

        logits = self.draft_model.forward(input_ids=input_ids,
                                          position_ids=position_ids,
                                          attn_metadata=attn_metadata,
                                          spec_metadata=spec_metadata,
                                          return_context_logits=True)
        batch_size = attn_metadata.num_seqs
        vocab_size = logits.shape[-1]
        logits = logits[spec_metadata.gather_ids]  # [batch_size, vocab_size]

        # new_draft_tokens: [batch_size * dynamic_tree_max_topK]
        # new_draft_scores: [batch_size * dynamic_tree_max_topK]
        new_draft_tokens, new_draft_scores = self.sample(logits=logits, max_top_k=self.dynamic_tree_max_topK)

        cur_scores = self.update_draft_tokens_and_scores(
            cur_draft_idx=0,
            batch_size=batch_size,
            new_draft_tokens=new_draft_tokens,
            new_draft_scores=new_draft_scores,
            previous_draft_scores=None,
            attn_metadata=attn_metadata,
            spec_metadata=spec_metadata)

        return_draft_logits = None
        with save_metadata_state(attn_metadata, spec_metadata):
            batch_size = attn_metadata.num_seqs

            self.prepare_for_generation(
                attn_metadata=attn_metadata,
                spec_metadata=spec_metadata,
                spec_tree_manager=spec_tree_manager,
                position_ids=position_ids)

            for layer_idx in range(1, self.max_draft_len):
                # input_ids: [batch_size * (max_total_draft_tokens + 1)]
                # position_ids: [batch_size * (max_total_draft_tokens + 1)]
                # logits: [batch_size * (max_total_draft_tokens + 1), vocab_size]
                logits = self.draft_model.forward(
                    input_ids=self.draft_tokens_buffer[:batch_size, :].reshape(
                        -1),
                    position_ids=self.position_ids_buffer[:batch_size, :].
                    reshape(-1),
                    attn_metadata=attn_metadata,
                    spec_metadata=spec_metadata,
                    return_context_logits=True)

                # new_draft_tokens: [batch_size * (max_total_draft_tokens + 1) * dynamic_tree_max_topK]
                # new_draft_scores: [batch_size * (max_total_draft_tokens + 1) * dynamic_tree_max_topK]
                new_draft_tokens, new_draft_scores = self.sample(
                    logits=logits, max_top_k=spec_tree_manager.dynamic_tree_max_topK)
                # Keep updating
                cur_scores = self.update_draft_tokens_and_scores(
                    cur_draft_idx=layer_idx,
                    batch_size=batch_size,
                    new_draft_tokens=new_draft_tokens,
                    new_draft_scores=new_draft_scores,
                    previous_draft_scores=cur_scores,
                    attn_metadata=attn_metadata,
                    spec_metadata=spec_metadata)

                if layer_idx == self.max_draft_len - 1:
                    # FIXME: Actually the logits is incorrect; we don't have compatibility with that yet.
                    return_draft_logits = logits

        # Resampling the final draft tokens
        # real_draft_tokens: [batch_size, self.max_total_draft_tokens]
        # topk_score_indices: [batch_size, self.max_total_draft_tokens]
        real_draft_tokens, topk_score_indices = self.resampling_final_draft_tokens(batch_size=batch_size)

        # return_new_draft_tokens: [max_total_draft_tokens, batch_size]
        return_new_draft_tokens = torch.transpose(real_draft_tokens, 0, 1)

        # return_draft_logits: [batch_size, max_total_draft_tokens + 1, vocab_size] -> [max_total_draft_tokens, batch_size, vocab_size]
        return_draft_logits = return_draft_logits.reshape(
            batch_size, self.max_total_draft_tokens + 1, vocab_size)
        return_draft_logits = torch.transpose(return_draft_logits[:, :-1, :], 0,
                                              1)

        assert return_new_draft_tokens.shape == (self.max_total_draft_tokens,
                                                 batch_size)
        assert return_draft_logits.shape == (self.max_total_draft_tokens,
                                             batch_size, vocab_size)

        print(f"======= return_new_draft_tokens: {return_new_draft_tokens} ========")

        return {
            "new_draft_tokens": return_new_draft_tokens,
            "draft_logits": return_draft_logits,
            "dynamic_tree_buffers": {
                "topk_score_indices": topk_score_indices,
                "history_draft_tokens_parent_buffer": self.history_draft_tokens_parent_buffer[:batch_size, :]}
        }

    def sample(self, logits: torch.Tensor, max_top_k: int) -> torch.Tensor:
        # TODO: inject the sampler here so we can support non-greedy

        # for draft_layer_idx == 0, logits is of shape [batch_size, vocab_size]
        # for draft_layer_idx > 0, logits is of shape [batch_size * (max_total_draft_tokens + 1), vocab_size]
        last_p = self.logsoftmax(logits)
        topk_values, topk_indices = torch.topk(last_p, k=max_top_k, dim=-1) # [batch_size, max_top_k] or [batch_size * max_total_draft_tokens, max_top_k]

        tokens = topk_indices.reshape(-1)
        scores = topk_values.reshape(-1)

        if hasattr(self.draft_model.model, "d2t"):
            d2t = self.draft_model.model.d2t.data
            tokens = tokens + d2t[tokens]

        return tokens, scores

    def update_draft_tokens_and_scores(self, cur_draft_idx: int, batch_size: int,
                                  new_draft_tokens: torch.Tensor,
                                  new_draft_scores: torch.Tensor,
                                  previous_draft_scores: torch.Tensor,
                                  attn_metadata: AttentionMetadata,
                                  spec_metadata: SpecMetadata):
        '''
        Args:
            cur_draft_idx: int, already finished forward.
            batch_size: int
            new_draft_tokens: 
                when cur_draft_idx == 0: [batch_size * dynamic_tree_max_topK]
                when cur_draft_idx > 0: [batch_size * (max_total_draft_tokens + 1) * dynamic_tree_max_topK]
            previous_draft_scores:
                when cur_draft_idx == 0: None
                when cur_draft_idx > 0: [batch_size, dynamic_tree_max_topK]
        '''

        print(f"======= cur_draft_idx: {cur_draft_idx} ========")
        # import pdb; pdb.set_trace()
        '''
        What this function does:
        1) Update the scores (exclude the first drafter layer)
        2) Extract the real draft tokens this layer
        3) Save the draft tokens and scores to self.history_draft_tokens_buffer and self.history_score_buffer, respectively.
        4) Update the attn_metadata.spec_decoding_packed_mask for the subsequent drafter layer.
        5) Update the spec_metadata.hidden_states_read_indices for the subsequent drafter layer.
        6) Update the parent nodes of the next layer's new nodes in advance.
        '''
        # After the first drafter layer, new_draft_tokens: [batch_size * dynamic_tree_max_topK]
        # For other drafter layers, new_draft_tokens: [batch_size * (max_total_draft_tokens + 1) * dynamic_tree_max_topK]
        if cur_draft_idx == 0:
            assert new_draft_tokens.shape[0] == (batch_size * self.dynamic_tree_max_topK)
            assert new_draft_scores.shape[0] == (batch_size * self.dynamic_tree_max_topK)
        else:
            assert new_draft_tokens.shape[0] == (batch_size * (self.max_total_draft_tokens + 1) * self.dynamic_tree_max_topK)
            assert new_draft_scores.shape[0] == (batch_size * (self.max_total_draft_tokens + 1) * self.dynamic_tree_max_topK)

        if cur_draft_idx == 0:
            # new_draft_tokens: [batch_size, self.dynamic_tree_max_topK]
            # new_draft_scores: [batch_size, self.dynamic_tree_max_topK]
            new_draft_tokens = new_draft_tokens.reshape(batch_size, self.dynamic_tree_max_topK)
            new_draft_scores = new_draft_scores.reshape(batch_size, self.dynamic_tree_max_topK)
            
            # 2) & 3) Update draft tokens and scores buffer.
            self.draft_tokens_buffer[:batch_size, :self.dynamic_tree_max_topK] = new_draft_tokens[:, :]
            self.history_draft_tokens_buffer[:batch_size, :self.dynamic_tree_max_topK] = new_draft_tokens[:, :]
            self.history_score_buffer[:batch_size, :self.dynamic_tree_max_topK] = new_draft_scores[:, :]

            # 4) Update the attn_metadata.spec_decoding_packed_mask
            attn_metadata.spec_decoding_packed_mask[:batch_size, :, :].fill_(0)
            dummy_idx = torch.arange(self.dynamic_tree_max_topK, dtype=torch.int32, device='cuda')
            packed_mask = torch.pow(2, dummy_idx) # [self.dynamic_tree_max_topK]
            attn_metadata.spec_decoding_packed_mask[:batch_size, :self.dynamic_tree_max_topK, :] = packed_mask.unsqueeze(1)
            print(f"======= attn_metadata.spec_decoding_packed_mask: {attn_metadata.spec_decoding_packed_mask} ========")

            # 5) Update the attn_metadata.hidden_states_read_indices
            ## Will be updated in the prepare_for_generation function. Because it will need the information of the old_write_indices and so on.

            # 6) Process the parent buffer.
            self.history_draft_tokens_parent_buffer[:batch_size, :self.dynamic_tree_max_topK] = -1 # Use -1 to represent the root node
            # These selected nodes will expand into new nodes at the next layer. 
            # We update the parent nodes of these new nodes in advance.
            parents_indices_for_next_layer_draft_tokens = torch.repeat_interleave(torch.arange(0, self.dynamic_tree_max_topK, dtype=torch.int32, device='cuda'), self.dynamic_tree_max_topK, dim=0) # [self.dynamic_tree_max_topK * self.dynamic_tree_max_topK]
            self.history_draft_tokens_parent_buffer[:batch_size, self.dynamic_tree_max_topK : self.dynamic_tree_max_topK + self.dynamic_tree_max_topK * self.dynamic_tree_max_topK] = parents_indices_for_next_layer_draft_tokens

            print(f"======= new_draft_tokens.shape: {new_draft_tokens.shape}, new_draft_tokens: {new_draft_tokens} ========")
            print(f"======= new_draft_scores.shape: {new_draft_scores.shape}, new_draft_scores: {new_draft_scores} ========")
            
            print(f"======= self.draft_tokens_buffer: {self.draft_tokens_buffer} ========")
            print(f"======= self.history_draft_tokens_buffer: {self.history_draft_tokens_buffer} ========")
            print(f"======= self.parents_indices_for_next_layer_draft_tokens: {parents_indices_for_next_layer_draft_tokens} ========")
            print(f"======= self.history_draft_tokens_parent_buffer: {self.history_draft_tokens_parent_buffer} ========")
            print(f"======= self.history_score_buffer: {self.history_score_buffer} ========")
            return new_draft_scores # [batch_size, self.dynamic_tree_max_topK]
        else:
            # new_draft_tokens: [batch_size * (self.max_total_draft_tokens + 1) * self.dynamic_tree_max_topK]
            # new_draft_scores: [batch_size * (self.max_total_draft_tokens + 1) * self.dynamic_tree_max_topK]

            new_draft_tokens = new_draft_tokens.reshape(batch_size, (self.max_total_draft_tokens + 1), self.dynamic_tree_max_topK)
            new_draft_scores = new_draft_scores.reshape(batch_size, (self.max_total_draft_tokens + 1), self.dynamic_tree_max_topK)
            print(f"======= new_draft_tokens.shape: {new_draft_tokens.shape}, new_draft_tokens: {new_draft_tokens} ========")
            print(f"======= new_draft_scores.shape: {new_draft_scores.shape}, new_draft_scores: {new_draft_scores} ========")

            # We process 'self.max_total_draft_tokens + 1' draft tokens, but we only need specific draft tokens for each layer.
            gather_draft_tokens_start_offset = (cur_draft_idx - 1) * self.dynamic_tree_max_topK
            gather_draft_tokens_end_offset = gather_draft_tokens_start_offset + self.dynamic_tree_max_topK
            gather_new_draft_tokens = new_draft_tokens[:, gather_draft_tokens_start_offset:gather_draft_tokens_end_offset, :].reshape(batch_size, self.dynamic_tree_max_topK * self.dynamic_tree_max_topK) # [batch_size, self.dynamic_tree_max_topK * self.dynamic_tree_max_topK]
            gather_new_draft_scores = new_draft_scores[:, gather_draft_tokens_start_offset:gather_draft_tokens_end_offset, :] # [batch_size, self.dynamic_tree_max_topK, self.dynamic_tree_max_topK]
            print(f"======= gather_draft_tokens_start_offset: {gather_draft_tokens_start_offset}, gather_draft_tokens_end_offset: {gather_draft_tokens_end_offset} ========")
            print(f"======= gather_new_draft_tokens.shape: {gather_new_draft_tokens.shape}, gather_new_draft_tokens: {gather_new_draft_tokens} ========")
            print(f"======= gather_new_draft_scores.shape: {gather_new_draft_scores.shape}, gather_new_draft_scores: {gather_new_draft_scores} ========")

            # 1) Update the scores with the previous layer's scores
            assert previous_draft_scores.shape == (batch_size, self.dynamic_tree_max_topK)
            gather_new_draft_scores = gather_new_draft_scores + previous_draft_scores.unsqueeze(2) # [batch_size, self.dynamic_tree_max_topK, self.dynamic_tree_max_topK]
            gather_new_draft_scores = gather_new_draft_scores.reshape(batch_size, self.dynamic_tree_max_topK * self.dynamic_tree_max_topK) # [batch_size, self.dynamic_tree_max_topK * self.dynamic_tree_max_topK]
            print(f"======= previous_draft_scores.shape: {previous_draft_scores.shape}, previous_draft_scores: {previous_draft_scores} ========")
            print(f"======= gather_new_draft_scores.shape: {gather_new_draft_scores.shape}, gather_new_draft_scores: {gather_new_draft_scores} ========")

            # 2) Extract the real draft tokens this layer, topk again.
            # topk_values: [batch_size, self.dynamic_tree_max_topK], the output scores of this layer
            # topk_indices: [batch_size, self.dynamic_tree_max_topK]
            topk_values, topk_indices = torch.topk(gather_new_draft_scores, k=self.dynamic_tree_max_topK, dim=-1) 
            real_draft_tokens = torch.gather(gather_new_draft_tokens, dim=1, index=topk_indices) # [batch_size, self.dynamic_tree_max_topK]
            write_back_real_draft_tokens_start_offset = cur_draft_idx * self.dynamic_tree_max_topK
            write_back_real_draft_tokens_end_offset = write_back_real_draft_tokens_start_offset + self.dynamic_tree_max_topK
            self.draft_tokens_buffer[:batch_size, write_back_real_draft_tokens_start_offset:write_back_real_draft_tokens_end_offset] = real_draft_tokens[:, :]
            print(f"======= write_back_real_draft_tokens_start_offset: {write_back_real_draft_tokens_start_offset}, write_back_real_draft_tokens_end_offset: {write_back_real_draft_tokens_end_offset} ========")
            print(f"======= topk_values: {topk_values}, topk_indices: {topk_indices} ========")
            print(f"======= real_draft_tokens.shape: {real_draft_tokens.shape}, real_draft_tokens: {real_draft_tokens} ========")
            print(f"======= self.draft_tokens_buffer: {self.draft_tokens_buffer} ========")

            # 3) Save the draft tokens and scores to self.history_draft_tokens_buffer and self.history_score_buffer.
            write_history_start_offset = self.dynamic_tree_max_topK + (cur_draft_idx - 1) * self.dynamic_tree_max_topK * self.dynamic_tree_max_topK
            write_history_end_offset = write_history_start_offset + self.dynamic_tree_max_topK * self.dynamic_tree_max_topK
            self.history_draft_tokens_buffer[:batch_size, write_history_start_offset:write_history_end_offset] = gather_new_draft_tokens[:, :]
            self.history_score_buffer[:batch_size, write_history_start_offset:write_history_end_offset] = gather_new_draft_scores[:, :]
            print(f"======= write_history_start_offset: {write_history_start_offset}, write_history_end_offset: {write_history_end_offset} ========")
            print(f"======= self.history_draft_tokens_buffer: {self.history_draft_tokens_buffer} ========")
            print(f"======= self.history_score_buffer: {self.history_score_buffer} ========")

            # 4) Update the attn_metadata.spec_decoding_packed_mask, shape: [max_num_requests, max_total_draft_tokens + 1, math.ceil(max_total_draft_tokens + 1 / 32)]
            selected_parents = topk_indices // self.dynamic_tree_max_topK # [batch_size, self.dynamic_tree_max_topK]
            # For simplicity, we will only consider the case where math.ceil(max_total_draft_tokens + 1 / 32) == 1.
            parents_packed_mask = torch.gather(attn_metadata.spec_decoding_packed_mask[:batch_size, gather_draft_tokens_start_offset:gather_draft_tokens_end_offset, :].squeeze(-1), dim=1, index=selected_parents) # [batch_size, self.dynamic_tree_max_topK]
            child_packed_mask = torch.pow(2, torch.arange(cur_draft_idx * self.dynamic_tree_max_topK, cur_draft_idx * self.dynamic_tree_max_topK + self.dynamic_tree_max_topK, dtype=torch.int32, device='cuda')) # [self.dynamic_tree_max_topK]
            print(f"======= child_packed_mask1111.shape: {child_packed_mask.shape}, child_packed_mask1111: {child_packed_mask} ========")
            child_packed_mask = child_packed_mask + parents_packed_mask # [batch_size, self.dynamic_tree_max_topK]
            attn_metadata.spec_decoding_packed_mask[:batch_size, write_back_real_draft_tokens_start_offset:write_back_real_draft_tokens_end_offset, :] = child_packed_mask.unsqueeze(-1)
            print(f"======= selected_parents.shape: {selected_parents.shape}, selected_parents: {selected_parents} ========")
            print(f"======= parents_packed_mask.shape: {parents_packed_mask.shape}, parents_packed_mask: {parents_packed_mask} ========")
            print(f"======= child_packed_mask2222.shape: {child_packed_mask.shape}, child_packed_mask22222: {child_packed_mask} ========")
            print(f"======= attn_metadata.spec_decoding_packed_mask: {attn_metadata.spec_decoding_packed_mask} ========")

            # 5) Update the spec_metadata.hidden_states_read_indices, shape: [max_num_tokens], but we save as [:batch_size * (max_total_draft_tokens + 1)]
            selected_parents_write_indices = selected_parents + gather_draft_tokens_start_offset # [batch_size, self.dynamic_tree_max_topK]
            hidden_states_write_indices_view = spec_metadata.hidden_states_write_indices[:batch_size * (self.max_total_draft_tokens + 1)] # [batch_size, self.max_total_draft_tokens + 1]
            hidden_states_write_indices_view = hidden_states_write_indices_view.view(batch_size, self.max_total_draft_tokens + 1) # [batch_size, self.max_total_draft_tokens + 1]
            child_hidden_states_read_indices = torch.gather(hidden_states_write_indices_view, dim=1, index=selected_parents_write_indices) # [batch_size, self.dynamic_tree_max_topK]
            print(f"======= selected_parents_write_indices.shape: {selected_parents_write_indices.shape}, selected_parents_write_indices: {selected_parents_write_indices} ========")
            print(f"======= hidden_states_write_indices_view.shape: {hidden_states_write_indices_view.shape}, hidden_states_write_indices_view: {hidden_states_write_indices_view} ========")
            print(f"======= child_hidden_states_read_indices.shape: {child_hidden_states_read_indices.shape}, child_hidden_states_read_indices: {child_hidden_states_read_indices} ========")
            print(f"======= spec_metadata.hidden_states_write_indices: {spec_metadata.hidden_states_write_indices} ========")

            hidden_states_read_indices_view = spec_metadata.hidden_states_read_indices[:batch_size * (self.max_total_draft_tokens + 1)] # [batch_size * (max_total_draft_tokens + 1)]
            hidden_states_read_indices_view = hidden_states_read_indices_view.view(batch_size, self.max_total_draft_tokens + 1) # [batch_size, self.max_total_draft_tokens + 1]
            hidden_states_read_indices_view[:, write_back_real_draft_tokens_start_offset:write_back_real_draft_tokens_end_offset] = child_hidden_states_read_indices[:, :] 
            print(f"======= hidden_states_read_indices_view.shape: {hidden_states_read_indices_view.shape}, hidden_states_read_indices_view: {hidden_states_read_indices_view} ========")
            print(f"======= spec_metadata.hidden_states_read_indices: {spec_metadata.hidden_states_read_indices} ========")

            if cur_draft_idx < self.max_draft_len - 1:
                # 6) Update the parent nodes of the next layer's new nodes in advance.
                # We need to know next layer's draft tokens are expaned from which parents. 
                # i.e. calculate the index of the selected draft tokens in the entire tree (including all historical nodes, for subsequent reconstruction of the entire tree).
                parents_indices = topk_indices + (self.dynamic_tree_max_topK + (cur_draft_idx - 1) * self.dynamic_tree_max_topK * self.dynamic_tree_max_topK) # [batch_size, self.dynamic_tree_max_topK]
                parents_indices = torch.repeat_interleave(parents_indices, self.dynamic_tree_max_topK, dim=1) # [batch_size, self.dynamic_tree_max_topK * self.dynamic_tree_max_topK]
                next_layer_draft_tokens_start_offset = self.dynamic_tree_max_topK + cur_draft_idx * self.dynamic_tree_max_topK * self.dynamic_tree_max_topK
                next_layer_draft_tokens_end_offset = next_layer_draft_tokens_start_offset + self.dynamic_tree_max_topK * self.dynamic_tree_max_topK
                self.history_draft_tokens_parent_buffer[:batch_size, next_layer_draft_tokens_start_offset:next_layer_draft_tokens_end_offset] = parents_indices[:, :]
                print(f"======= parents_indices.shape: {parents_indices.shape}, parents_indices: {parents_indices} ========")
                print(f"======= next_layer_draft_tokens_start_offset: {next_layer_draft_tokens_start_offset}, next_layer_draft_tokens_end_offset: {next_layer_draft_tokens_end_offset} ========")
                print(f"======= self.history_draft_tokens_parent_buffer: {self.history_draft_tokens_parent_buffer} ========")

            return topk_values # [batch_size, self.dynamic_tree_max_topK]
    

    def resampling_final_draft_tokens(self, batch_size: int):
        '''
        Restruct the tree based on the self.history_draft_tokens_buffer, self.history_draft_tokens_parent_buffer and self.history_score_buffer.
        '''
        # self.history_score_buffer[:batch_size, :] shape: [batch_size, dynamic_tree_max_topK + dynamic_tree_max_topK * dynamic_tree_max_topK * (max_draft_len - 1)]
        topk_score_indices = torch.topk(self.history_score_buffer[:batch_size, :], k=self.max_total_draft_tokens, dim=-1).indices
        topk_score_indices = torch.sort(topk_score_indices).values # [batch_size, self.max_total_draft_tokens]

        # The final output draft tokens
        real_draft_tokens = torch.gather(self.history_draft_tokens_buffer[:batch_size, :], dim=1, index=topk_score_indices) # [batch_size, self.max_total_draft_tokens]

        # self.history_draft_tokens_parent_buffer[:batch_size, :] shape: [batch_size, dynamic_tree_max_topK + dynamic_tree_max_topK * dynamic_tree_max_topK * (max_draft_len - 1)]
        # real_draft_tokens_parents = torch.gather(self.history_draft_tokens_parent_buffer[:batch_size, :], dim=1, index=topk_score_indices) # [batch_size, self.max_total_draft_tokens]


        # return real_draft_tokens, topk_score_indices, real_draft_tokens_parents
        return real_draft_tokens, topk_score_indices



    def prepare_for_generation(self, attn_metadata: AttentionMetadata,
                               spec_metadata: SpecMetadata,
                               spec_tree_manager: SpecTreeManager,
                               position_ids: torch.Tensor):
        '''
        Setup the attn_metadata and spec_metadata for the subsequent drafter layer. Therefore, only call once after the first drafter layer.
        To the subsequent drafter layer, we take 'max_total_drafter_tokens + 1' draft tokens as input.
        Only the first part of the draft tokens is meaningful, and the later tokens can be regarded as padding
        until we continuously write the correct value.

        This introduces additional redundant computation, but it makes it compatible with cuda graphs.

        What we need to prepare are:
            1) position_ids
            2) attn_metadata
                2.1) kv_lens_cuda
                2.2) _seq_lens, _seq_lens_cuda
                2.3) host_request_types
                2.4) num_contexts
                2.5) use_spec_decoding
                2.6) spec_decoding_position_offsets
                2.7) spec_decoding_packed_mask
                2.8) spec_decoding_generation_lengths
            3) spec_metadata
                3.1) num_tokens
                3.2) hidden_states_read_indices, hidden_states_write_indices
                3.3) is_first_draft
        '''
        batch_size = attn_metadata.num_seqs

        # 1) Prepare the position_ids
        num_accepted_draft_tokens = spec_metadata.num_accepted_draft_tokens[:
                                                                            batch_size]
        seq_lens = attn_metadata.seq_lens_cuda[:batch_size]
        # Calculate last accepted token indices
        last_tokens_idx = torch.cumsum(
            seq_lens, dim=0,
            dtype=torch.long) - seq_lens + num_accepted_draft_tokens
        position_start_idx = position_ids[0,
                                          last_tokens_idx] + 1  # [batch_size]
        self.position_ids_buffer[:batch_size, :] = position_start_idx.unsqueeze(
            1) + spec_tree_manager.spec_dec_position_offsets_for_drafter_model[0, :].unsqueeze(0) # [batch_size, max_total_draft_tokens + 1]

        # 2) Prepare the attn_metadata
        ## 2.1) kv_lens_cuda
        attn_metadata.kv_lens_cuda[:
                                   batch_size] -= seq_lens - num_accepted_draft_tokens - 1
        attn_metadata.kv_lens_cuda[:batch_size] += (
            self.max_total_draft_tokens + 1)

        ## 2.2) _seq_lens, _seq_lens_cuda
        attn_metadata._seq_lens[:batch_size].fill_(self.max_total_draft_tokens +
                                                   1)
        attn_metadata._seq_lens_cuda[:batch_size].fill_(
            self.max_total_draft_tokens + 1)
        attn_metadata.on_update()

        ## 2.3) host_request_types
        attn_metadata.host_request_types[:attn_metadata.num_contexts].fill_(1)

        ## 2.4) num_contexts
        attn_metadata.num_contexts = 0

        ## 2.5) use_spec_decoding
        attn_metadata.use_spec_decoding = True

        ## 2.6) spec_decoding_position_offsets
        ### attn_metadata.spec_decoding_position_offsets: [max_num_requests, max_total_draft_tokens + 1]
        attn_metadata.spec_decoding_position_offsets[:batch_size, :] = spec_tree_manager.spec_dec_position_offsets_for_drafter_model[0, :].unsqueeze(0) # [batch_size, max_total_draft_tokens + 1]

        ## 2.7) spec_decoding_packed_mask 
        ### NOTE: spec_decoding_packed_mask will be updated for each drafter layer in 'update_draft_tokens_and_scores'
        # attn_metadata.spec_decoding_packed_mask[:batch_size, :, :].fill_(0)
        # dummy_idx = torch.arange(self.dynamic_tree_max_topK, dtype=torch.int32, device='cuda')
        # packed_mask = torch.pow(2, dummy_idx + 1) - 1
        # attn_metadata.spec_decoding_packed_mask[:batch_size, :self.dynamic_tree_max_topK, :] = packed_mask.unsqueeze(1)

        ## 2.8) spec_decoding_generation_lengths
        ### attn_metadata.spec_decoding_generation_lengths: [max_num_requests]
        attn_metadata.spec_decoding_generation_lengths[:batch_size] = self.max_total_draft_tokens + 1

        # 3) Update spec_metadata
        ## 3.1) num_tokens
        spec_metadata.num_tokens = batch_size * (self.max_total_draft_tokens +
                                                 1)
        ## 3.2) hidden_states_read_indices, hidden_states_write_indices
        old_write_indices = spec_metadata.hidden_states_write_indices
        start_idx = old_write_indices[
            last_tokens_idx]  # [batch_size], already take the accepted tokens into account.

        ### spec_metadata.hidden_states_read_indices: [max_num_tokens], but we save as [:batch_size * (max_total_draft_tokens + 1)]
        ### NOTE: spec_metadata.hidden_states_read_indices needs to be updated for each drafter layer
        hidden_states_read_offset = start_idx.unsqueeze(1).repeat(1, self.max_total_draft_tokens + 1) # [batch_size, max_total_draft_tokens + 1]
        spec_metadata.hidden_states_read_indices[:batch_size * (self.max_total_draft_tokens + 1)] = hidden_states_read_offset.reshape(-1)

        ### spec_metadata.hidden_states_write_indices: [max_num_tokens], but we save as [:batch_size * (max_total_draft_tokens + 1)]
        hidden_states_write_offset = torch.arange(
            1, self.max_total_draft_tokens + 1 + 1,
            device=position_ids.device).unsqueeze(0).repeat(
                batch_size, 1) + start_idx.unsqueeze(1)
        spec_metadata.hidden_states_write_indices[:batch_size * (
            self.max_total_draft_tokens +
            1)] = hidden_states_write_offset.reshape(-1)

        ## 3.3) is_first_draft
        spec_metadata.eagle3_resource_manager.is_first_draft = False
        spec_metadata.is_first_draft = False

        return
