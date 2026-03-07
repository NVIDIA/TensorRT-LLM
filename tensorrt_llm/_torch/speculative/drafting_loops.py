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
from tensorrt_llm._torch.speculative.dynamic_tree_ops import \
    create_dynamic_tree_ops_converter
from tensorrt_llm._torch.speculative.eagle3 import Eagle3SpecMetadata
from tensorrt_llm._torch.speculative.interface import SpecMetadata
from tensorrt_llm._torch.speculative.spec_tree_manager import SpecTreeManager

# Enable capture_scalar_outputs to avoid graph breaks from Tensor.item() calls
torch._dynamo.config.capture_scalar_outputs = True


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
        if self.max_draft_len > 1:
            is_eagle3 = isinstance(spec_metadata, Eagle3SpecMetadata)
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
                    if i == 0 and is_eagle3:
                        spec_metadata.hidden_states_read_indices[:batch_size].copy_(
                            spec_metadata.
                            hidden_states_write_indices[:batch_size])

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

    def prepare_for_generation(self, attn_metadata: AttentionMetadata,
                               spec_metadata: SpecMetadata,
                               position_ids: torch.Tensor) -> torch.Tensor:
        batch_size = attn_metadata.num_seqs
        num_accepted_draft_tokens = spec_metadata.num_accepted_draft_tokens[:
                                                                            batch_size]
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
                    input_ids=self.draft_tokens_buffer[:batch_size, :self.
                                                       max_total_draft_tokens +
                                                       1].reshape(-1),
                    position_ids=self.
                    position_ids_buffer[:batch_size, :self.
                                        max_total_draft_tokens + 1].reshape(-1),
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
        if return_draft_logits is None:
            # When max_draft_len == 1, the loop doesn't execute.
            # Expand the initial logits to match the expected shape.
            return_draft_logits = logits.unsqueeze(1).expand(
                batch_size, self.max_total_draft_tokens + 1,
                vocab_size).reshape(-1, vocab_size)

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
                                                         0, 1:self.
                                                         max_total_draft_tokens +
                                                         1].unsqueeze(
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
        hidden_states_read_indices_offset = spec_tree_manager.hidden_states_read_indices_offset_for_drafter_model[:self.max_total_draft_tokens + 1].repeat(
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
                 max_batch_size: int, dynamic_tree_max_topK,
                 draft_model: torch.nn.Module):
        super().__init__()
        self.draft_model = draft_model
        self.config = self.draft_model.config
        self.model_config = self.draft_model.model_config
        self.max_draft_len = max_draft_len
        self.max_total_draft_tokens = max_total_draft_tokens
        self.max_batch_size = max_batch_size
        self.dynamic_tree_max_topK = dynamic_tree_max_topK
        self.logsoftmax = nn.LogSoftmax(dim=-1)

        # 1D buffers: store tokens contiguously
        self.draft_tokens_buffer = torch.zeros(
            (max_batch_size * (max_total_draft_tokens)),
            dtype=torch.int64,
            device='cuda')
        self.position_ids_buffer = torch.zeros(
            (max_batch_size * (max_total_draft_tokens + 1)),
            dtype=torch.int64,
            device='cuda')
        self.history_draft_tokens_buffer = torch.zeros(
            (max_batch_size, (dynamic_tree_max_topK +
                              dynamic_tree_max_topK * dynamic_tree_max_topK *
                              (max_draft_len - 1))),
            dtype=torch.int64,
            device='cuda')
        self.history_score_buffer = torch.zeros(
            (max_batch_size, dynamic_tree_max_topK +
             dynamic_tree_max_topK * dynamic_tree_max_topK *
             (max_draft_len - 1)),
            dtype=torch.float32,
            device='cuda')
        self.history_draft_tokens_parent_buffer = torch.zeros(
            (max_batch_size, self.dynamic_tree_max_topK *
             (self.max_draft_len - 1) + 1),
            dtype=torch.int64,
            device='cuda')
        self.tree_mask_buffer = torch.zeros(
            (max_batch_size * (max_total_draft_tokens + 1) *
             (max_total_draft_tokens + 1)),
            dtype=torch.int32,
            device='cuda')
        self.tree_mask_init_buffer = torch.eye(
            dynamic_tree_max_topK, dtype=torch.int32,
            device='cuda').unsqueeze(0).repeat(max_batch_size, 1, 1)
        self.tree_mask_padding_zeros = torch.zeros(
            (max_batch_size, max_total_draft_tokens,
             max_total_draft_tokens + 1),
            dtype=torch.int32,
            device='cuda')

        # Initialize dynamic tree ops converter for CUDA kernel integration
        self.tree_ops_converter = create_dynamic_tree_ops_converter(
            dynamic_tree_max_topK=dynamic_tree_max_topK,
            max_draft_len=max_draft_len,
            max_total_draft_tokens=max_total_draft_tokens,
            max_batch_size=max_batch_size,
            device=torch.device('cuda'),
        )

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
        logits = logits[spec_metadata.gather_ids]
        new_draft_tokens, new_draft_scores = self.sample(
            logits=logits, max_top_k=self.dynamic_tree_max_topK)

        cur_scores = self.update_draft_tokens_and_scores(
            cur_draft_idx=0,
            new_draft_tokens=new_draft_tokens,
            new_draft_scores=new_draft_scores,
            previous_draft_scores=None,
            attn_metadata=attn_metadata,
            spec_metadata=spec_metadata,
            spec_tree_manager=spec_tree_manager)

        return_draft_logits = None
        with save_metadata_state(attn_metadata, spec_metadata):
            batch_size = attn_metadata.num_seqs

            self.prepare_for_generation(attn_metadata=attn_metadata,
                                        spec_metadata=spec_metadata,
                                        position_ids=position_ids,
                                        cur_draft_idx=0)
            for layer_idx in range(1, self.max_draft_len):
                num_infer_tokens = batch_size * (
                    self.dynamic_tree_max_topK) * layer_idx

                logits = self.draft_model.forward(
                    input_ids=self.draft_tokens_buffer[0:num_infer_tokens],
                    position_ids=self.position_ids_buffer[0:num_infer_tokens],
                    attn_metadata=attn_metadata,
                    spec_metadata=spec_metadata,
                    return_context_logits=True)
                logits = logits.reshape(batch_size,
                                        self.dynamic_tree_max_topK * layer_idx,
                                        vocab_size)

                selected_logits = logits[:, -self.dynamic_tree_max_topK:, :]
                new_draft_tokens, new_draft_scores = self.sample(
                    logits=selected_logits,
                    max_top_k=spec_tree_manager.dynamic_tree_max_topK)

                cur_scores = self.update_draft_tokens_and_scores(
                    cur_draft_idx=layer_idx,
                    new_draft_tokens=new_draft_tokens,
                    new_draft_scores=new_draft_scores,
                    previous_draft_scores=cur_scores,
                    attn_metadata=attn_metadata,
                    spec_metadata=spec_metadata,
                    spec_tree_manager=spec_tree_manager)
                self.prepare_for_generation(
                    attn_metadata=attn_metadata,
                    spec_metadata=spec_metadata,
                    cur_draft_idx=layer_idx,
                    position_ids=None,
                )

                if layer_idx == self.max_draft_len - 1:
                    # FIXME: The logits here contain all accumulated tokens from all layers,
                    # but after resampling we need logits corresponding to the selected tokens.
                    # This requires storing logits per-layer and gathering them after resampling.
                    # For now, we use the last layer's logits as an approximation.
                    return_draft_logits = logits

        # Resampling the final draft tokens
        # real_draft_tokens: [batch_size, self.max_total_draft_tokens]
        # topk_score_indices: [batch_size, self.max_total_draft_tokens]
        real_draft_tokens, topk_score_indices = self.resampling_final_draft_tokens(
            batch_size=batch_size)

        tree_structure = None
        if self.tree_ops_converter is not None:
            try:
                # Write directly to spec_tree_manager buffers (target model reads from there).
                # use_packed_mask=True: kernel outputs bit-packed mask directly,
                # skipping the unpacked tree_mask_buffer and compute_spec_dec_packed_mask.
                tree_structure = self.tree_ops_converter.build_dynamic_tree(
                    history_draft_tokens_parent_buffer=self.
                    history_draft_tokens_parent_buffer[:batch_size],
                    topk_score_indices=topk_score_indices,
                    tree_mask=spec_tree_manager.
                    spec_dec_packed_mask[:batch_size],
                    positions=spec_tree_manager.
                    spec_dec_position_offsets[:batch_size],
                    retrieve_index=spec_tree_manager.
                    retrieve_index[:batch_size],
                    retrieve_next_token=spec_tree_manager.
                    retrieve_next_token[:batch_size],
                    retrieve_next_sibling=spec_tree_manager.
                    retrieve_next_sibling[:batch_size],
                    use_packed_mask=True,
                )

            except Exception as e:
                raise RuntimeError(
                    f"Dynamic tree CUDA kernel failed: {e}") from e

        # return_new_draft_tokens: [max_total_draft_tokens, batch_size]
        return_new_draft_tokens = torch.transpose(real_draft_tokens, 0, 1)

        # return_draft_logits: [batch_size, max_total_draft_tokens + 1, vocab_size] -> [max_total_draft_tokens, batch_size, vocab_size]
        # FIXME: For dynamic tree, logits from last layer don't match max_total_draft_tokens
        # because we only process tokens_accumulated = batch_size * layer_idx * K tokens
        # For now, create dummy logits. Proper fix requires collecting logits per-layer and gathering.
        # Fallback: create dummy logits
        return_draft_logits = torch.zeros(self.max_total_draft_tokens,
                                          batch_size,
                                          vocab_size,
                                          dtype=torch.float32,
                                          device='cuda')

        assert return_new_draft_tokens.shape == (self.max_total_draft_tokens,
                                                 batch_size)
        assert return_draft_logits.shape == (self.max_total_draft_tokens,
                                             batch_size, vocab_size)

        return {
            "new_draft_tokens": return_new_draft_tokens,
            "draft_logits": return_draft_logits,
            "dynamic_tree_buffers": {
                "topk_score_indices":
                topk_score_indices,
                "history_draft_tokens_parent_buffer":
                self.history_draft_tokens_parent_buffer[:batch_size, :],
                "tree_structure":
                tree_structure,  # CUDA kernel output or None
            }
        }

    def sample(self, logits: torch.Tensor, max_top_k: int) -> torch.Tensor:
        # TODO: inject the sampler here so we can support non-greedy

        # for draft_layer_idx == 0, logits is of shape [batch_size, vocab_size]
        # for draft_layer_idx > 0, logits is of shape [batch_size * (max_total_draft_tokens + 1), vocab_size]
        last_p = self.logsoftmax(logits)
        topk_values, topk_indices = torch.topk(
            last_p, k=max_top_k, dim=-1
        )  # [batch_size, max_top_k] or [batch_size * max_total_draft_tokens, max_top_k]
        if hasattr(self.draft_model.model, "d2t"):
            d2t = self.draft_model.model.d2t.data
            topk_indices = topk_indices + d2t[topk_indices]

        return topk_indices, topk_values

    def update_draft_tokens_and_scores(
            self, cur_draft_idx: int, new_draft_tokens: torch.Tensor,
            new_draft_scores: torch.Tensor, previous_draft_scores: torch.Tensor,
            attn_metadata: AttentionMetadata, spec_metadata: SpecMetadata,
            spec_tree_manager: 'SpecTreeManager'):
        '''
        Update draft tokens and scores, write contiguously to buffer.

        Args:
            cur_draft_idx: int, current draft layer index (0 for Layer 0, 1 for Layer 1...)
            batch_size: int, batch size
            new_draft_tokens:
                Layer 0: [batch_size * K]  - K tokens sampled from root node
                Layer i>0: [batch_size * K * K]  - K candidates sampled from each of K parent nodes
            new_draft_scores:
                Same shape as new_draft_tokens, corresponding log probability scores
            previous_draft_scores:
                Layer 0: None
                Layer i>0: [batch_size, K]  - Cumulative scores of K selected tokens from previous layer
            attn_metadata: AttentionMetadata
            spec_metadata: SpecMetadata
        '''
        '''
        What this function does:
        1) Update the scores (exclude the first drafter layer)
        2) Extract the real draft tokens this layer
        3) Save the draft tokens and scores to self.history_draft_tokens_buffer and self.history_score_buffer, respectively.
        4) Update the attn_metadata.spec_decoding_packed_mask for the subsequent drafter layer.
        5) Update the spec_metadata.hidden_states_read_indices for the subsequent drafter layer.
        6) Update the parent nodes of the next layer's new nodes in advance.
        '''
        return_draft_scores = None
        batch_size = attn_metadata.num_seqs
        if cur_draft_idx == 0:

            new_draft_scores = new_draft_scores.reshape(
                batch_size, self.dynamic_tree_max_topK)

            num_tokens_layer0 = batch_size * self.dynamic_tree_max_topK
            self.draft_tokens_buffer[:
                                     num_tokens_layer0] = new_draft_tokens.reshape(
                                         -1)

            self.history_draft_tokens_buffer[:batch_size, :self.
                                             dynamic_tree_max_topK] = new_draft_tokens.reshape(
                                                 batch_size,
                                                 self.dynamic_tree_max_topK)
            self.history_score_buffer[:batch_size, :self.
                                      dynamic_tree_max_topK] = new_draft_scores[:, :]

            self.history_draft_tokens_parent_buffer[:batch_size, :self.
                                                    dynamic_tree_max_topK +
                                                    1] = torch.arange(
                                                        -1,
                                                        self.
                                                        dynamic_tree_max_topK,
                                                        device='cuda',
                                                        dtype=torch.int32
                                                    ).unsqueeze(0).expand(
                                                        batch_size, -1
                                                    )  # Use -1 to represent the root node

            self.prepare_tree_mask_and_position_offset(cur_draft_idx,
                                                       attn_metadata,
                                                       spec_tree_manager, None)

            self.update_hidden_states_indices(cur_draft_idx, attn_metadata,
                                              spec_metadata)

            return_draft_scores = new_draft_scores
        else:

            new_draft_tokens = new_draft_tokens.reshape(
                batch_size,
                self.dynamic_tree_max_topK * self.dynamic_tree_max_topK)

            # 1) Update the scores with the previous layer's scores
            assert previous_draft_scores.shape == (batch_size,
                                                   self.dynamic_tree_max_topK)
            new_draft_scores = new_draft_scores + previous_draft_scores.unsqueeze(
                2)
            new_draft_scores = new_draft_scores.reshape(
                batch_size,
                self.dynamic_tree_max_topK * self.dynamic_tree_max_topK)

            # 2) Extract the real draft tokens this layer, topk again.
            # topk_values: [batch_size, self.dynamic_tree_max_topK], the output scores of this layer
            # topk_indices: [batch_size, self.dynamic_tree_max_topK]
            topk_values, topk_indices = torch.topk(new_draft_scores,
                                                   k=self.dynamic_tree_max_topK,
                                                   dim=-1)
            real_draft_tokens = torch.gather(
                new_draft_tokens, dim=1,
                index=topk_indices)  # [batch_size, self.dynamic_tree_max_topK]
            num_tokens_previous_layer = cur_draft_idx * self.dynamic_tree_max_topK
            num_tokens_current_layer = (cur_draft_idx +
                                        1) * self.dynamic_tree_max_topK
            old_tokens = self.draft_tokens_buffer[:batch_size *
                                                  num_tokens_previous_layer].reshape(
                                                      batch_size,
                                                      num_tokens_previous_layer)
            self.draft_tokens_buffer[:batch_size *
                                     num_tokens_current_layer] = torch.cat(
                                         [old_tokens, real_draft_tokens],
                                         dim=1).reshape(-1)

            # 3) Save the draft tokens and scores to self.history_draft_tokens_buffer and self.history_score_buffer.
            write_history_start_offset = self.dynamic_tree_max_topK + (
                cur_draft_idx -
                1) * self.dynamic_tree_max_topK * self.dynamic_tree_max_topK
            write_history_end_offset = write_history_start_offset + self.dynamic_tree_max_topK * self.dynamic_tree_max_topK
            self.history_draft_tokens_buffer[:batch_size,
                                             write_history_start_offset:
                                             write_history_end_offset] = new_draft_tokens
            self.history_score_buffer[:batch_size, write_history_start_offset:
                                      write_history_end_offset] = new_draft_scores

            # 4) Update the attn_metadata.spec_decoding_packed_mask
            # Shape: [max_num_requests, max_total_draft_tokens + 1, math.ceil((max_total_draft_tokens + 1) / 32)]
            selected_parents = topk_indices // self.dynamic_tree_max_topK  # [batch_size, self.dynamic_tree_max_topK]
            self.prepare_tree_mask_and_position_offset(cur_draft_idx,
                                                       attn_metadata,
                                                       spec_tree_manager,
                                                       selected_parents)

            self.update_hidden_states_indices(cur_draft_idx, attn_metadata,
                                              spec_metadata, selected_parents)

            # 6) Update the parent nodes of the next layer's new nodes in advance.
            # We need to know next layer's draft tokens are expanded from which parents.
            if cur_draft_idx < self.max_draft_len - 1:
                next_layer_draft_tokens_start_offset = (
                    cur_draft_idx) * self.dynamic_tree_max_topK + 1
                next_layer_draft_tokens_end_offset = next_layer_draft_tokens_start_offset + self.dynamic_tree_max_topK
                parents_relative_indices = topk_indices + self.dynamic_tree_max_topK**2 * (
                    cur_draft_idx - 1) + self.dynamic_tree_max_topK
                self.history_draft_tokens_parent_buffer[:batch_size,
                                                        next_layer_draft_tokens_start_offset:
                                                        next_layer_draft_tokens_end_offset] = parents_relative_indices

            return_draft_scores = topk_values
        return return_draft_scores

    def prepare_tree_mask_and_position_offset(
            self,
            cur_draft_idx: int,
            attn_metadata: AttentionMetadata,
            spec_tree_manager: SpecTreeManager,
            selected_parents: torch.Tensor = None):
        '''
        Prepare the mask for the next layer.
        '''
        batch_size = attn_metadata.num_seqs
        num_tokens_current_layer = self.dynamic_tree_max_topK * (cur_draft_idx +
                                                                 1)
        num_tokens_previous_layer = self.dynamic_tree_max_topK * cur_draft_idx
        if cur_draft_idx == 0:
            attn_metadata.spec_decoding_packed_mask.fill_(0)
            spec_tree_manager.compute_spec_dec_packed_mask(
                self.tree_mask_init_buffer[:batch_size],
                attn_metadata.spec_decoding_packed_mask[:batch_size])
            self.tree_mask_buffer[:batch_size * num_tokens_current_layer *
                                  num_tokens_current_layer].copy_(
                                      self.tree_mask_init_buffer[:batch_size].
                                      view(-1))
            attn_metadata.spec_decoding_position_offsets.fill_(0)
            attn_metadata.spec_decoding_generation_lengths[:
                                                           batch_size] = num_tokens_current_layer
        else:
            num_parent_mask = batch_size * cur_draft_idx * self.dynamic_tree_max_topK * cur_draft_idx * self.dynamic_tree_max_topK
            parant_mask = self.tree_mask_buffer[:num_parent_mask].reshape(
                batch_size, cur_draft_idx * self.dynamic_tree_max_topK,
                cur_draft_idx * self.dynamic_tree_max_topK)

            selected_parents_expanded = selected_parents.unsqueeze(-1).expand(
                batch_size, self.dynamic_tree_max_topK, parant_mask.size(-1))
            # get the valid last K tokens of the parent mask
            parant_mask_selected = torch.gather(
                parant_mask[:, -self.dynamic_tree_max_topK:, :],
                dim=1,
                index=selected_parents_expanded)
            current_mask = torch.cat(
                [parant_mask_selected, self.tree_mask_init_buffer[:batch_size]],
                dim=2)
            mask_padding = self.tree_mask_padding_zeros[:batch_size, :
                                                        num_tokens_previous_layer, :
                                                        num_tokens_current_layer]
            # [batch_size, (cur_draft_idx+ 1)*self.dynamic_tree_max_topK,(cur_draft_idx+ 1)*self.dynamic_tree_max_topK ]
            current_mask = torch.cat([mask_padding, current_mask], dim=1)
            spec_tree_manager.compute_spec_dec_packed_mask(
                current_mask,
                attn_metadata.spec_decoding_packed_mask[:batch_size])
            self.tree_mask_buffer[:batch_size * num_tokens_current_layer *
                                  num_tokens_current_layer].copy_(
                                      current_mask.view(-1))

            attn_metadata.spec_decoding_generation_lengths[:
                                                           batch_size] = num_tokens_current_layer

            previous_position_offsets = attn_metadata.spec_decoding_position_offsets[:
                                                                                     batch_size
                                                                                     *
                                                                                     num_tokens_previous_layer]
            previous_position_offsets = previous_position_offsets.view(
                batch_size, num_tokens_previous_layer)
            # Last K positions + 1 to get new K positions
            new_position_offsets = torch.cat([
                previous_position_offsets,
                previous_position_offsets[:, -self.dynamic_tree_max_topK:] + 1
            ],
                                             dim=1)
            attn_metadata.spec_decoding_position_offsets[:batch_size *
                                                         num_tokens_current_layer] = new_position_offsets.reshape(
                                                             -1)

    def update_hidden_states_indices(self,
                                     cur_draft_idx: int,
                                     attn_metadata: AttentionMetadata,
                                     spec_metadata: SpecMetadata,
                                     selected_parents: torch.Tensor = None):
        """
        Update hidden states read indices so the next draft layer reads correct hidden states.

        Flow:
        1. Current layer (cur_draft_idx) inference is done, produced (cur_draft_idx+1)*K tokens
        2. From last K write_indices of current layer, select based on selected_parents
        3. Update next layer's read_indices to point to selected parent hidden states

        Args:
            cur_draft_idx: Current draft layer index (0-based), the just-completed layer
            spec_metadata: Metadata containing hidden_states_read/write_indices
            selected_parents: [batch_size, K], which of the K candidates to use as parents
                              for the next layer. Values in [0, K-1], relative to last K tokens.
        """
        K = self.dynamic_tree_max_topK
        batch_size = attn_metadata.num_seqs
        if cur_draft_idx == 0:
            spec_metadata.num_tokens = batch_size * self.dynamic_tree_max_topK
            last_tokens_idx = self.get_last_accepted_token_indices(
                attn_metadata, spec_metadata)
            old_write_indices = spec_metadata.hidden_states_write_indices
            start_idx = old_write_indices[
                last_tokens_idx]  # [batch_size], already take the accepted tokens into account.
            hidden_states_read_offset = start_idx.unsqueeze(1).repeat(
                1, self.dynamic_tree_max_topK
            )  # [batch_size, dynamic_tree_max_topK]
            spec_metadata.hidden_states_read_indices[:batch_size * self.
                                                     dynamic_tree_max_topK] = hidden_states_read_offset.reshape(
                                                         -1)

            ### spec_metadata.hidden_states_write_indices: [max_num_tokens], but we save as [:batch_size * (max_total_draft_tokens + 1)]
            hidden_states_write_offset = torch.arange(
                1, self.dynamic_tree_max_topK + 1,
                device='cuda').unsqueeze(0).repeat(batch_size,
                                                   1) + start_idx.unsqueeze(1)
            spec_metadata.hidden_states_write_indices[:batch_size * self.
                                                      dynamic_tree_max_topK] = hidden_states_write_offset.reshape(
                                                          -1)

        else:

            # 1. Compute positions of "last K" tokens in the global sequence

            last_K_start_for_previous_layer = (cur_draft_idx - 1) * K
            last_K_end_for_previous_layer = last_K_start_for_previous_layer + K
            last_K_start_for_current_layer = cur_draft_idx * K
            last_K_end_for_current_layer = last_K_start_for_current_layer + K
            num_tokens_previous_layer = cur_draft_idx * K
            num_tokens_current_layer = (cur_draft_idx + 1) * K

            # 2. Get 2D view of hidden_states_write_indices
            hidden_states_write_indices_view = spec_metadata.hidden_states_write_indices[:
                                                                                         batch_size
                                                                                         *
                                                                                         num_tokens_previous_layer]  # [batch_size * (max_total_draft_tokens + 1)]
            hidden_states_write_indices_view = hidden_states_write_indices_view.view(
                batch_size, num_tokens_previous_layer)

            # 3. Extract write_indices of "last K" tokens from current layer
            last_K_write_indices = hidden_states_write_indices_view[:,
                                                                    last_K_start_for_previous_layer:
                                                                    last_K_end_for_previous_layer]

            # 4. Use selected_parents to pick from last K, yielding next layer's read_indices
            child_hidden_states_read_indices = torch.gather(
                last_K_write_indices, dim=1, index=selected_parents)

            # 5. Update next layer's read_indices
            # Next layer needs (cur_draft_idx+1)*K positions; first cur_draft_idx*K are unchanged
            # Last K positions are updated to the selected parent positions
            hidden_states_read_indices_view = spec_metadata.hidden_states_read_indices[:
                                                                                       batch_size
                                                                                       *
                                                                                       num_tokens_current_layer]  # [batch_size * (max_total_draft_tokens + 1)]
            hidden_states_read_indices_view = hidden_states_read_indices_view.view(
                batch_size, num_tokens_current_layer)
            hidden_states_read_indices_view[:, last_K_start_for_current_layer:
                                            last_K_end_for_current_layer] = child_hidden_states_read_indices

            # 6. Update next layer's write_indices
            # First num_tokens_previous_layer values unchanged; append K new incrementing values
            hidden_states_write_indices_view_current = spec_metadata.hidden_states_write_indices[:
                                                                                                 batch_size
                                                                                                 *
                                                                                                 num_tokens_current_layer]
            hidden_states_write_indices_view_current = hidden_states_write_indices_view_current.view(
                batch_size, num_tokens_current_layer)

            # Get last write_index from previous layer
            last_write_index = hidden_states_write_indices_view[:, -1]

            # Generate K new incrementing write_indices: [last+1, last+2, ..., last+K]
            new_write_indices = last_write_index.unsqueeze(1) + torch.arange(
                1, K + 1, device='cuda', dtype=torch.long).unsqueeze(0)

            # Write to current layer's last K positions
            hidden_states_write_indices_view_current[:,
                                                     last_K_start_for_current_layer:
                                                     last_K_end_for_current_layer] = new_write_indices

    def resampling_final_draft_tokens(self, batch_size: int):
        '''
        Reconstruct the tree based on history_draft_tokens_buffer, history_draft_tokens_parent_buffer, and history_score_buffer.
        '''

        topk_score_indices = torch.topk(
            self.history_score_buffer[:batch_size, :],
            k=self.max_total_draft_tokens,
            dim=-1).indices
        topk_score_indices = torch.sort(topk_score_indices).values

        # The final output draft tokens
        real_draft_tokens = torch.gather(
            self.history_draft_tokens_buffer[:batch_size, :],
            dim=1,
            index=topk_score_indices)
        return real_draft_tokens, topk_score_indices

    def get_last_accepted_token_indices(self, attn_metadata: AttentionMetadata,
                                        spec_metadata: SpecMetadata):
        '''
        Get the last accepted token indices.
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
        return last_tokens_idx

    def prepare_for_generation(
        self,
        attn_metadata: AttentionMetadata,
        spec_metadata: SpecMetadata,
        cur_draft_idx: int,
        position_ids: torch.Tensor = None,
    ):
        '''
        Setup the attn_metadata and spec_metadata for the subsequent drafter layer.

        '''
        num_tokens_current_layer = self.dynamic_tree_max_topK * (cur_draft_idx +
                                                                 1)
        batch_size = attn_metadata.num_seqs
        if cur_draft_idx == 0:

            seq_lens = attn_metadata.seq_lens_cuda[:batch_size]
            num_accepted_draft_tokens = spec_metadata.num_accepted_draft_tokens[:
                                                                                batch_size]

            last_tokens_idx = self.get_last_accepted_token_indices(
                attn_metadata, spec_metadata)
            new_position_ids = position_ids[0,
                                            last_tokens_idx] + 1  # [batch_size]
            self.position_ids_buffer[:batch_size * self.
                                     dynamic_tree_max_topK] = new_position_ids.repeat_interleave(
                                         self.dynamic_tree_max_topK)

            attn_metadata.kv_lens_cuda[:
                                       batch_size] -= seq_lens - num_accepted_draft_tokens - 1
            attn_metadata._seq_lens[:batch_size].fill_(num_tokens_current_layer)
            attn_metadata._seq_lens_cuda[:batch_size].fill_(
                num_tokens_current_layer)
            attn_metadata.on_update()
            attn_metadata.kv_lens_cuda[:
                                       batch_size] += self.dynamic_tree_max_topK

            attn_metadata.host_request_types[:attn_metadata.num_contexts].fill_(
                1)

            ## 2.4) num_contexts
            attn_metadata.num_contexts = 0
            # attn_metadata.use_spec_decoding = True

            spec_metadata.eagle3_resource_manager.is_first_draft = False
            spec_metadata.is_first_draft = False
        else:
            num_tokens_previous_layer = cur_draft_idx * self.dynamic_tree_max_topK
            previous_position_ids = self.position_ids_buffer[:batch_size *
                                                             num_tokens_previous_layer]
            previous_position_ids = previous_position_ids.view(
                batch_size, num_tokens_previous_layer)
            new_position_ids = torch.cat([
                previous_position_ids,
                previous_position_ids[:, -self.dynamic_tree_max_topK:] + 1
            ],
                                         dim=1)
            self.position_ids_buffer[:batch_size *
                                     num_tokens_current_layer] = new_position_ids.reshape(
                                         -1)
            attn_metadata._seq_lens[:batch_size].fill_(num_tokens_current_layer)
            attn_metadata._seq_lens_cuda[:batch_size].fill_(
                num_tokens_current_layer)
            attn_metadata.on_update()
            attn_metadata.kv_lens_cuda[:
                                       batch_size] += self.dynamic_tree_max_topK
            spec_metadata.num_tokens = batch_size * num_tokens_current_layer

            return
