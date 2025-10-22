"""
This module contains capturable drafting loops for speculative decoding.

These are torch modules wrap another draft model. The wrapped module
is supposed to invoke the draft model autoregressively and invoke
a sampling algorithm to obtain draft tokens. By structuring the code
like this, we are able to avoid host overhead: the entire drafting process
for speculation can be launched as a single CUDA graph.
"""

from contextlib import contextmanager
from typing import List, Optional, Tuple

import torch

from tensorrt_llm._torch.attention_backend.interface import AttentionMetadata
from tensorrt_llm._torch.speculative.eagle3 import Eagle3SpecMetadata
from tensorrt_llm._torch.speculative.interface import SpecMetadata
from tensorrt_llm._torch.speculative.spec_tree_manager import SpecTreeManager


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


def prepare_for_generation(attn_metadata: AttentionMetadata,
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


def prepare_for_generation_with_tree_decoding(
        prepare_for_layer_idx: int, new_draft_tokens: List[torch.Tensor],
        attn_metadata: AttentionMetadata, spec_metadata: SpecMetadata,
        spec_tree_manager: SpecTreeManager,
        position_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    '''
    Prepare the inputs for the next draft layer. What we need to prepare are:
        1) inputs_ids
        2) position_ids
        3) attn_metadata
            3.1) kv_lens_cuda
            3.2) _seq_lens, _seq_lens_cuda
            3.3) host_request_types
            3.4) num_contexts
            3.5) use_spec_decoding
            3.6) spec_decoding_position_offsets
            3.7) spec_decoding_packed_mask
            3.8) spec_decoding_generation_lengths
        4) spec_metadata
            4.1) num_tokens
            4.2) gather_ids
            4.3) hidden_states_read_indices, hidden_states_write_indices
            4.4) is_first_draft

    '''

    batch_size = attn_metadata.num_seqs
    next_layer_gen_len_per_req = spec_tree_manager.spec_dec_generation_lengths_for_drafter_model[
        prepare_for_layer_idx - 1]

    # 1) Prepare the inputs_ids
    all_draft_tokens = torch.cat(
        new_draft_tokens,
        dim=-1)  # [batch_size, num_draft_tokens_has_been_generated]
    cur_tokens_gather_idx = spec_tree_manager.tokens_gather_idx[
        prepare_for_layer_idx -
        1] - 1  # shape: [next_layer_gen_len_per_req]. -1 is toshift the root node
    new_input_ids = all_draft_tokens[:, cur_tokens_gather_idx].reshape(
        -1)  # [batch_size * next_layer_gen_len_per_req]

    num_accepted_draft_tokens = spec_metadata.num_accepted_draft_tokens[:
                                                                        batch_size]
    seq_lens = attn_metadata.seq_lens_cuda[:batch_size]
    last_tokens_idx = None

    # 2) Prepare the position_ids
    if prepare_for_layer_idx == 1:
        last_tokens_idx = torch.cumsum(
            seq_lens, dim=0,
            dtype=torch.long) - seq_lens + num_accepted_draft_tokens
        new_position_ids = position_ids[0, last_tokens_idx] + 1  # [batch_size]
        assert new_position_ids.shape == (batch_size, )
        # For the layer_idx == 1, the input tokens are both expanded from root node.
        # Therefore, their position ids are the same.
        new_position_ids = torch.repeat_interleave(
            new_position_ids, repeats=next_layer_gen_len_per_req,
            dim=0)  # [batch_size * next_layer_gen_len_per_req]
    else:
        position_ids = position_ids.reshape(batch_size, -1)
        position_ids_start_idx = position_ids[:, 0]  # [batch_size]
        assert position_ids_start_idx.shape == (batch_size, )

        new_position_ids = spec_tree_manager.spec_dec_position_offsets_for_drafter_model[
            prepare_for_layer_idx - 1].unsqueeze(0).repeat(
                batch_size, 1)  # [batch_size, num_next_layer_input_tokens]
        new_position_ids = new_position_ids + position_ids_start_idx.unsqueeze(
            1)  # [batch_size, num_next_layer_input_tokens]
        new_position_ids = new_position_ids.reshape(
            -1)  # [batch_size * num_next_layer_input_tokens]

    assert new_position_ids.shape == new_input_ids.shape

    # 3) Prepare the attn_metadata
    ## 3.1) kv_lens_cuda
    if prepare_for_layer_idx == 1:
        attn_metadata.kv_lens_cuda[:
                                   batch_size] -= seq_lens - num_accepted_draft_tokens - 1
        attn_metadata.kv_lens_cuda[:batch_size] += next_layer_gen_len_per_req
    else:
        prev_layer_gen_len_per_req = spec_tree_manager.spec_dec_generation_lengths_for_drafter_model[
            prepare_for_layer_idx - 2]
        attn_metadata.kv_lens_cuda[:
                                   batch_size] -= prev_layer_gen_len_per_req  # reset to original length before the drafter loop.
        attn_metadata.kv_lens_cuda[:batch_size] += next_layer_gen_len_per_req

    ## 3.2) _seq_lens, _seq_lens_cuda
    attn_metadata._seq_lens[:batch_size].fill_(next_layer_gen_len_per_req)
    attn_metadata._seq_lens_cuda[:batch_size].fill_(next_layer_gen_len_per_req)
    attn_metadata.on_update()

    # Update once is enough
    if prepare_for_layer_idx == 1:
        ## 3.3) host_request_types
        attn_metadata.host_request_types[:attn_metadata.num_contexts].fill_(1)
        ## 3.4) num_contexts
        attn_metadata.num_contexts = 0
        ## 3.5) use_spec_decoding
        attn_metadata.use_spec_decoding = True

    ## 3.6) spec_decoding_position_offsets
    attn_metadata.spec_decoding_position_offsets[:, :
                                                 next_layer_gen_len_per_req] = spec_tree_manager.spec_dec_position_offsets_for_drafter_model[
                                                     prepare_for_layer_idx -
                                                     1].unsqueeze(0)
    attn_metadata.spec_decoding_position_offsets[:,
                                                 next_layer_gen_len_per_req:] = 0

    ## 3.7) spec_decoding_packed_mask
    attn_metadata.spec_decoding_packed_mask[:, :
                                            next_layer_gen_len_per_req, :] = spec_tree_manager.spec_dec_packed_mask_for_drafter_model[
                                                prepare_for_layer_idx -
                                                1].unsqueeze(0)
    attn_metadata.spec_decoding_packed_mask[:,
                                            next_layer_gen_len_per_req:, :] = 0

    ## 3.8) spec_decoding_generation_lengths
    attn_metadata.spec_decoding_generation_lengths[:] = next_layer_gen_len_per_req

    # 4) spec_metadata
    ## 4.1) num_tokens
    spec_metadata.num_tokens = batch_size * next_layer_gen_len_per_req

    ## 4.2) gather_ids
    offset = torch.arange(
        batch_size,
        device=position_ids.device) * next_layer_gen_len_per_req  # [batch_size]
    spec_metadata.gather_ids = spec_tree_manager.logits_gather_idx[
        prepare_for_layer_idx - 1].unsqueeze(0).repeat(
            batch_size, 1)  # [1, num_tokens_has_children]
    spec_metadata.gather_ids = spec_metadata.gather_ids + offset.unsqueeze(
        1)  # [batch_size, num_tokens_has_children]
    spec_metadata.gather_ids = spec_metadata.gather_ids.reshape(
        -1)  # [batch_size * num_tokens_has_children]

    ## 4.3) hidden_states_read_indices, hidden_states_write_indices
    if isinstance(spec_metadata, Eagle3SpecMetadata):
        start_idx = None
        if prepare_for_layer_idx == 1:
            old_write_indices = spec_metadata.hidden_states_write_indices
            start_idx = old_write_indices[
                last_tokens_idx]  # [batch_size], already take the accepted tokens into account.
        else:
            prev_layer_gen_len_per_req = spec_tree_manager.spec_dec_generation_lengths_for_drafter_model[
                prepare_for_layer_idx - 2]
            last_tokens_idx = torch.arange(
                batch_size,
                device=position_ids.device) * prev_layer_gen_len_per_req
            old_read_indices = spec_metadata.hidden_states_read_indices
            start_idx = old_read_indices[last_tokens_idx]  # [batch_size]

        start_idx = start_idx.unsqueeze(1)  # [batch_size, 1]

        start_read_idx = start_idx + spec_tree_manager.hidden_states_read_indices_offset_for_drafter_model[
            prepare_for_layer_idx -
            1]  # [batch_size, next_layer_gen_len_per_req]
        spec_metadata.hidden_states_read_indices[:batch_size *
                                                 next_layer_gen_len_per_req].copy_(
                                                     start_read_idx.reshape(-1)
                                                 )  # [batch_size * next_layer_gen_len_per_req]

        start_write_idx = start_idx + spec_tree_manager.hidden_states_write_indices_offset_for_drafter_model[
            prepare_for_layer_idx -
            1]  # [batch_size, next_layer_gen_len_per_req]
        spec_metadata.hidden_states_write_indices[:batch_size *
                                                  next_layer_gen_len_per_req].copy_(
                                                      start_write_idx.reshape(
                                                          -1)
                                                  )  # [batch_size * next_layer_gen_len_per_req]

        if prepare_for_layer_idx == 1:
            ## 4.4) is_first_draft
            spec_metadata.eagle3_resource_manager.is_first_draft = False
            spec_metadata.is_first_draft = False

    return new_input_ids, new_position_ids


class ChainDrafter(torch.nn.Module):

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
        spec_tree_manager = None
        if isinstance(spec_metadata, Eagle3SpecMetadata):
            spec_tree_manager = spec_metadata.eagle3_resource_manager.spec_tree_manager

        logits = self.draft_model.forward(input_ids=input_ids,
                                          position_ids=position_ids,
                                          attn_metadata=attn_metadata,
                                          spec_metadata=spec_metadata,
                                          return_context_logits=True)
        batch_size = attn_metadata.num_seqs
        vocab_size = logits.shape[-1]
        logits = logits[spec_metadata.gather_ids]  # [batch_size, vocab_size]

        new_draft_tokens = [
            self.sample(draft_layer_idx=0,
                        batch_size=batch_size,
                        logits=logits,
                        spec_tree_manager=spec_tree_manager)
        ]
        assert logits.shape == (batch_size, vocab_size)
        # When using tree decoding, the first layer's draft tokens are all from the root node's logits.
        # Therefore, we repeat the logits and collect them.
        draft_logits = [
            logits if spec_tree_manager is None else logits.repeat(
                spec_tree_manager.top_k_list_cuda[0], 1).reshape(
                    batch_size, -1, vocab_size)
        ]

        with save_metadata_state(attn_metadata, spec_metadata):
            batch_size = attn_metadata.num_seqs
            if spec_tree_manager is None:
                new_input_ids = new_draft_tokens[-1]
                new_position_ids = prepare_for_generation(
                    attn_metadata, spec_metadata, position_ids)
            else:
                new_input_ids, new_position_ids = prepare_for_generation_with_tree_decoding(
                    prepare_for_layer_idx=
                    1,  # prepare for the 1st layer, start from the 0-th layer.
                    new_draft_tokens=new_draft_tokens,
                    attn_metadata=attn_metadata,
                    spec_metadata=spec_metadata,
                    spec_tree_manager=spec_tree_manager,
                    position_ids=position_ids)

            for layer_idx in range(1, self.max_draft_len):
                logits = self.draft_model.forward(
                    input_ids=new_input_ids,
                    position_ids=new_position_ids,
                    attn_metadata=attn_metadata,
                    spec_metadata=spec_metadata,
                    return_context_logits=False
                    if spec_tree_manager is None else True)
                if spec_tree_manager is not None:
                    # if using tree decoding, only the last 'num_tokens_has_children' tokens need to be sampled.
                    logits = logits[spec_metadata.gather_ids]

                new_draft_tokens.append(
                    self.sample(draft_layer_idx=layer_idx,
                                batch_size=batch_size,
                                logits=logits,
                                spec_tree_manager=spec_tree_manager))

                if spec_tree_manager is None:
                    draft_logits.append(logits)
                else:
                    # logits: [batch_size * num_tokens_has_children, vocab_size]
                    cur_top_k_list = spec_tree_manager.top_k_list_cuda[
                        layer_idx]  # [num_tokens_has_children]
                    cur_top_k_list = cur_top_k_list.repeat(
                        batch_size)  # [batch_size * num_tokens_has_children]
                    logits = torch.repeat_interleave(
                        logits, repeats=cur_top_k_list, dim=0
                    )  # [batch_size * num_tokens_has_children, vocab_size]
                    logits = logits.reshape(
                        batch_size, -1, vocab_size
                    )  # [batch_size, next_layer_draft_tokens, vocab_size]
                    draft_logits.append(logits)

                if spec_tree_manager is None:
                    new_input_ids = new_draft_tokens[-1]
                    new_position_ids += 1
                    attn_metadata.kv_lens_cuda[:batch_size] += 1
                    if layer_idx == 0 and isinstance(spec_metadata,
                                                     Eagle3SpecMetadata):
                        spec_metadata.hidden_states_read_indices[:batch_size].copy_(
                            spec_metadata.
                            hidden_states_write_indices[:batch_size])
                elif layer_idx < spec_tree_manager.max_draft_len - 1:
                    new_input_ids, new_position_ids = prepare_for_generation_with_tree_decoding(
                        prepare_for_layer_idx=layer_idx + 1,
                        new_draft_tokens=new_draft_tokens,
                        attn_metadata=attn_metadata,
                        spec_metadata=spec_metadata,
                        spec_tree_manager=spec_tree_manager,
                        position_ids=new_position_ids)

        if spec_tree_manager is None:
            return {
                "new_draft_tokens": torch.stack(new_draft_tokens),
                "draft_logits": torch.stack(draft_logits)
            }
        else:
            # new_draft_tokens: List[torch.Tensor], each tensor is of shape [batch_size, num_draft_tokens_each_layers]
            # len(new_draft_tokens) == max_draft_len
            return_new_draft_tokens = torch.cat(
                new_draft_tokens,
                dim=-1)  # [batch_size, max_total_draft_tokens]
            return_new_draft_tokens = torch.transpose(
                return_new_draft_tokens, 0,
                1)  # [max_total_draft_tokens, batch_size]

            # draft_logits: List[torch.Tensor], each tensor is of shape [batch_size, num_draft_tokens_each_layers, vocab_size]
            return_draft_logits = torch.cat(
                draft_logits,
                dim=1)  # [batch_size, max_total_draft_tokens, vocab_size]
            return_draft_logits = torch.transpose(
                return_draft_logits, 0,
                1)  # [max_total_draft_tokens, batch_size, vocab_size]

            assert return_new_draft_tokens.shape[
                0] == return_draft_logits.shape[0]
            assert return_new_draft_tokens.shape[
                1] == return_draft_logits.shape[1]

            return {
                "new_draft_tokens": return_new_draft_tokens,
                "draft_logits": return_draft_logits
            }

    def sample(
            self,
            draft_layer_idx: int,
            batch_size: int,
            logits: torch.Tensor,
            spec_tree_manager: Optional[SpecTreeManager] = None
    ) -> torch.Tensor:
        # TODO: inject the sampler here so we can support non-greedy

        if spec_tree_manager is None:
            tokens = torch.argmax(logits, dim=-1)
            if hasattr(self.draft_model.model, "d2t"):
                d2t = self.draft_model.model.d2t.data
                return tokens + d2t[tokens]
        else:
            max_topk_list = spec_tree_manager.max_top_k_list_cuda[
                draft_layer_idx]
            indices = torch.topk(logits, k=max_topk_list, dim=-1).indices
            top_k_list = spec_tree_manager.top_k_list_cuda[draft_layer_idx]
            top_k_list = top_k_list.repeat(batch_size)
            rows = torch.arange(top_k_list.shape[0],
                                dtype=torch.int32,
                                device=logits.device)
            row_indices = torch.repeat_interleave(rows, repeats=top_k_list)
            col_indices = torch.cat([torch.arange(c) for c in top_k_list])
            tokens = indices[
                row_indices,
                col_indices]  # [batch_size * num_draft_tokens_this_layer]

            if hasattr(self.draft_model.model, "d2t"):
                d2t = self.draft_model.model.d2t.data
                tokens = tokens + d2t[tokens]

            # reshape, for better gather later.
            tokens = tokens.reshape(
                batch_size, -1)  # [batch_size, num_draft_tokens_this_layer]

        return tokens

    def load_weights_from_target_model(self,
                                       target_model: torch.nn.Module) -> None:
        loader = getattr(self.draft_model, "load_weights_from_target_model",
                         None)
        if callable(loader):
            self.draft_model.load_weights_from_target_model(target_model)
