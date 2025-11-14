"""
This module contains capturable drafting loops for speculative decoding.

These are torch modules wrap another draft model. The wrapped module
is supposed to invoke the draft model autoregressively and invoke
a sampling algorithm to obtain draft tokens. By structuring the code
like this, we are able to avoid host overhead: the entire drafting process
for speculation can be launched as a single CUDA graph.
"""

from contextlib import contextmanager

import torch

from tensorrt_llm._torch.attention_backend.interface import AttentionMetadata
from tensorrt_llm._torch.speculative.eagle3 import Eagle3SpecMetadata
from tensorrt_llm._torch.speculative.interface import SpecMetadata


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

            new_position_ids = prepare_for_generation(attn_metadata,
                                                      spec_metadata,
                                                      position_ids)
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

    def load_weights_from_target_model(self,
                                       target_model: torch.nn.Module) -> None:
        loader = getattr(self.draft_model, "load_weights_from_target_model",
                         None)
        if callable(loader):
            self.draft_model.load_weights_from_target_model(target_model)
