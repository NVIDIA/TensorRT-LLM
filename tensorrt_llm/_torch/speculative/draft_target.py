"""
DraftTarget One-Model Speculative Decoding Implementation.

This module implements a one-model approach for DraftTarget speculative decoding,
where the draft and target models share the same model engine. The draft model
layers are integrated into the target model's KV cache and run in a single forward pass.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import torch
from torch import nn

from tensorrt_llm.mapping import Mapping

from ..attention_backend import AttentionMetadata
from ..pyexecutor.sampler import TorchSampler
from .interface import SpecMetadata, SpecWorkerBase
from .mtp import MTPSampler

if TYPE_CHECKING:
    from ...llmapi.llm_args import DraftTargetDecodingConfig


@dataclass
class DraftTargetOneModelSpecMetadata(SpecMetadata):
    """
    Metadata for DraftTarget one-model speculative decoding.

    This class manages the batch information needed for the one-model DraftTarget
    approach where draft and target models share the same model engine.
    Unlike Eagle3/MTP, DraftTarget does not require capturing hidden states
    from the target model to pass to the draft model.
    """

    # The max number of tokens
    max_num_tokens: int = 0
    # The index of the batch inputs
    batch_indices_cuda: Optional[torch.Tensor] = None
    # Per-request flags indicating if each context request is in its final chunk
    # Shape: (max_num_requests,), only first num_contexts entries are meaningful
    is_last_context_chunk: Optional[torch.Tensor] = None

    def __post_init__(self):
        self.batch_indices_cuda = torch.empty(
            [self.max_num_requests],
            dtype=torch.int,
            device="cuda",
        )
        # Initialize is_last_context_chunk flags
        # TODO: This should be populated from LlmRequest.is_last_context_chunk
        # by the executor/scheduler when preparing the batch
        self.is_last_context_chunk = torch.zeros(
            [self.max_num_requests],
            dtype=torch.bool,
            device="cuda",
        )

    def set_last_context_chunk_flags(self, context_requests):
        """
        Helper method to populate is_last_context_chunk flags from context requests.
        Should be called by the executor/scheduler when preparing the batch.

        Args:
            context_requests: List of LlmRequest objects for context phase
        """
        if not context_requests:
            return

        is_last_chunk_flags = []
        for req in context_requests:
            # A context is on its last chunk if there's no remaining context to process
            # Check if context_remaining_length attribute exists and is 0
            is_last = getattr(req, 'context_remaining_length', 0) == 0
            is_last_chunk_flags.append(is_last)

        if is_last_chunk_flags:
            is_last_chunk_tensor = torch.tensor(
                is_last_chunk_flags, dtype=torch.bool, device="cpu", pin_memory=True)
            self.is_last_context_chunk[:len(is_last_chunk_flags)].copy_(
                is_last_chunk_tensor, non_blocking=True)

    def prepare(self):
        """Prepare the metadata before model forward."""
        assert self.request_ids is not None
        # Update batch indices
        num_seqs = len(self.request_ids)
        batch_indices = torch.arange(num_seqs, dtype=torch.int, device="cpu", pin_memory=True)
        self.batch_indices_cuda[:num_seqs].copy_(batch_indices, non_blocking=True)

        # Note: is_last_context_chunk should be populated by calling
        # set_last_context_chunk_flags() before prepare() is called.
        # If not set, all contexts default to True (assuming they are last chunks).


class DraftTargetOneModelSampler(MTPSampler):
    """
    Sampler for DraftTarget one-model speculative decoding.

    Inherits from MTPSampler to reuse the speculative decoding sampling logic.
    """

    def __init__(self, args: TorchSampler.Args):
        super().__init__(args, nextn=args.max_draft_len)


class DraftTargetOneModelWorker(SpecWorkerBase):
    """
    Worker for DraftTarget one-model speculative decoding.

    This worker handles the draft token generation and verification process
    within a single model engine. It:
    1. Samples and accepts draft tokens from the target model's logits
    2. Prepares inputs for the draft model forward
    3. Runs the draft model to generate new draft tokens
    4. Returns the results for the next iteration
    """

    def __init__(self, spec_config: "DraftTargetDecodingConfig", mapping: Mapping):
        super().__init__()
        self.spec_config = spec_config
        self.mapping = mapping

    @property
    def max_draft_len(self) -> int:
        return self.spec_config.max_draft_len

    def forward(self, input_ids, position_ids, hidden_states, logits,
                attn_metadata, spec_metadata, draft_model):
        """
        Forward pass for Draft Target One Model speculative decoding.

        Key differences from Eagle3:
        - No hidden states passed to draft model
        - Full tokens passed (0..n+1 with positions 0..n+1)
        - Extra token only added on last chunk for chunked context
        """
        batch_size = attn_metadata.num_seqs
        num_contexts = attn_metadata.num_contexts
        num_ctx_tokens = attn_metadata.num_ctx_tokens
        num_gens = batch_size - num_contexts

        raw_logits = logits

        self._execute_guided_decoder_if_present(logits)

        # Sample and accept tokens
        accepted_tokens, num_accepted_tokens = self.sample_and_accept_draft_tokens(
            logits, attn_metadata, spec_metadata)

        # Save the old attn_metadata and spec_metadata
        self._prepare_attn_metadata_for_spec_dec(attn_metadata)

        # Prepare inputs for the 1st draft model forward
        position_ids = position_ids.squeeze(0)
        inputs = self.prepare_1st_drafter_inputs(
            input_ids=input_ids,
            position_ids=position_ids,
            accepted_tokens=accepted_tokens,
            attn_metadata=attn_metadata,
            spec_metadata=spec_metadata,
            draft_model=draft_model)

        # CRITICAL: Compute gather_ids BEFORE modifying seq_lens
        # For context requests: compute gather_ids accounting for variable layout
        # Last chunks: [ctx_tokens, appended_token]
        # Non-last chunks: [ctx_tokens] (no appended token)
        if num_contexts > 0:
            is_last_chunk = spec_metadata.is_last_context_chunk[:num_contexts]

            # Compute indices accounting for variable layout
            # Last chunks occupy (seq_len + 1) slots, non-last chunks occupy seq_len slots
            seq_lens_with_appends = attn_metadata.seq_lens_cuda[:num_contexts] + is_last_chunk.long()
            ctx_cumsum = torch.cumsum(seq_lens_with_appends, dim=0, dtype=torch.long)

            # For each context:
            # - If last chunk: appended token is at cumsum - 1, original last is at cumsum - 2
            # - If non-last chunk: original last token is at cumsum - 1
            gather_ids_ctx = ctx_cumsum - 1  # For last chunks, this is the appended token
                                             # For non-last chunks, this is the original last token

        # Adjust seq_lens for the first draft forward to account for appended tokens
        # For last chunk contexts, we appended an extra token, so increment seq_lens
        # This is crucial for accuracy - the attention mechanism needs correct seq_lens
        if num_contexts > 0:
            is_last_chunk = spec_metadata.is_last_context_chunk[:num_contexts]
            # Update CUDA version (used by attention kernels)
            attn_metadata._seq_lens_cuda[:num_contexts] += is_last_chunk.long()
            # Update CPU version (used for control flow)
            # This loop is acceptable as num_contexts is typically small
            is_last_chunk_cpu = is_last_chunk.cpu()
            for i in range(num_contexts):
                if is_last_chunk_cpu[i]:
                    attn_metadata._seq_lens[i] += 1
            attn_metadata.on_update()

        # Predict draft tokens
        next_draft_tokens = []
        original_all_rank_num_tokens = attn_metadata.all_rank_num_tokens
        original_num_contexts = num_contexts  # Save before it gets set to 0

        for i in range(self.max_draft_len):
            if i == 0:
                # First iteration: gather from last accepted token position
                start_ids_gen = (spec_metadata.batch_indices_cuda[:num_gens] *
                                 (self.max_draft_len + 1)).long()
                # For generation, tokens start after context tokens + appended tokens (only for last chunks)
                # ctx_tokens_total = num_ctx_tokens + number of last chunks
                if num_contexts > 0:
                    num_last_chunks = spec_metadata.is_last_context_chunk[:num_contexts].sum()
                    ctx_tokens_total = num_ctx_tokens + num_last_chunks
                else:
                    ctx_tokens_total = 0
                gather_ids_gen = (start_ids_gen +
                                  num_accepted_tokens[num_contexts:] - 1 +
                                  ctx_tokens_total)

                if num_contexts > 0:
                    gather_ids = torch.cat([gather_ids_ctx, gather_ids_gen], dim=0)
                else:
                    gather_ids = gather_ids_gen
            else:
                # All of the seq_len are 1, use batch_indices_cuda as gather_ids
                gather_ids = spec_metadata.batch_indices_cuda[:batch_size]

            if self.guided_decoder is not None:
                new_tokens = inputs["input_ids"][gather_ids]
                self.guided_decoder.add_draft_batch(new_tokens,
                                                    num_accepted_tokens,
                                                    draft_step=i)

            # Update attn_metadata.all_rank_num_tokens for attention DP
            if original_all_rank_num_tokens is not None:
                if i == 0:
                    attn_metadata.all_rank_num_tokens = original_all_rank_num_tokens
                elif spec_metadata.all_rank_num_seqs is not None:
                    attn_metadata.all_rank_num_tokens = spec_metadata.all_rank_num_seqs

            hidden_states = draft_model.model(**inputs)

            # Disable spec-dec mode after first iteration
            attn_metadata.use_spec_decoding = False

            logits = draft_model.logits_processor(hidden_states[gather_ids],
                                                  draft_model.lm_head,
                                                  attn_metadata, True)
            if self.guided_decoder is not None:
                d2t = getattr(draft_model.model, "d2t", None)
                self.guided_decoder.execute_draft_batch(logits,
                                                        d2t,
                                                        draft_step=i)

            new_draft_token = self.draft_decoder(logits, draft_model)
            next_draft_tokens.append(new_draft_token)
            # update inputs
            hidden_states = hidden_states[gather_ids]
            position_ids = inputs["position_ids"][gather_ids] + 1
            # update attn_metadata
            if i == 0:
                # After first draft iteration, all sequences process 1 token at a time
                attn_metadata._seq_lens[:batch_size].fill_(1)
                attn_metadata._seq_lens_cuda[:batch_size].fill_(1)
                attn_metadata.on_update()
                # cannot run generation if their is no kv cache
                if inputs["attn_metadata"].kv_cache_manager is not None:
                    # Mark all requests as generation for subsequent iterations
                    attn_metadata.host_request_types[:attn_metadata.
                                                     num_contexts].fill_(1)
                    attn_metadata.num_contexts = 0
                # update kv_lens_cuda
                if hasattr(attn_metadata, 'kv_lens_cuda'):
                    # For generation: rewind KV cache by removing rejected draft tokens
                    attn_metadata.kv_lens_cuda[num_contexts:batch_size] -= (
                        self.max_draft_len - num_accepted_tokens[num_contexts:])
                    # BUGFIX: Don't manually increment kv_lens for contexts
                    # When seq_lens was modified to N+1 for last chunks (line 192), the attention kernel
                    # already updated kv_lens during the forward pass based on the increased seq_lens.
                    # Manually incrementing here causes double-increment: kv_lens becomes N+2 instead of N+1,
                    # corrupting KV cache state and causing accuracy degradation.
                    #
                    # Eagle3 doesn't have this bug because it doesn't modify seq_lens, so the attention
                    # kernel doesn't auto-increment kv_lens, requiring the manual increment.
                    #
                    # For Draft Target: seq_lens modification → attention auto-increments kv_lens → no manual increment needed
            elif hasattr(attn_metadata, 'kv_lens_cuda'):
                # BUGFIX: Only increment kv_lens for generation sequences, not original contexts
                # After setting num_contexts=0, we lose track of which sequences were contexts.
                # Original context sequences should not have their kv_lens incremented in subsequent
                # draft iterations, as they're done processing. Only generation sequences need updates.
                attn_metadata.kv_lens_cuda[original_num_contexts:batch_size] += 1
            # support attention dp
            inputs = {
                "input_ids": new_draft_token,
                "position_ids": position_ids,
                "attn_metadata": attn_metadata,
                "spec_metadata": spec_metadata,
            }
        next_draft_tokens = torch.stack(next_draft_tokens, dim=1)

        # CRITICAL: Save kv_lens_cuda before restore (these are permanent KV cache updates)
        if hasattr(attn_metadata, 'kv_lens_cuda'):
            kv_lens_cuda_after_draft = attn_metadata.kv_lens_cuda.clone()

        # restore attn_metadata to support cuda graph
        self._restore_attn_metadata_from_spec_dec(attn_metadata)
        # restore all_rank_num_tokens for attention DP
        if original_all_rank_num_tokens is not None:
            attn_metadata.all_rank_num_tokens = original_all_rank_num_tokens

        # CRITICAL: Restore kv_lens_cuda to post-draft values (permanent updates, not temporary)
        if hasattr(attn_metadata, 'kv_lens_cuda'):
            attn_metadata.kv_lens_cuda.copy_(kv_lens_cuda_after_draft)

        # prepare next new tokens to support overlap scheduler
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

    def sample_and_accept_draft_tokens(
        self,
        logits: torch.Tensor,
        attn_metadata: AttentionMetadata,
        spec_metadata: DraftTargetOneModelSpecMetadata,
    ):
        """
        Sample and accept draft tokens for Draft Target.
        Uses the base implementation for strict acceptance.
        """
        batch_size = attn_metadata.num_seqs
        num_contexts = attn_metadata.num_contexts
        num_gens = batch_size - num_contexts

        # Reshape draft tokens for base implementation
        draft_tokens = spec_metadata.draft_tokens.reshape(
            num_gens, self.max_draft_len)

        # Use base implementation for strict acceptance
        return self._sample_and_accept_draft_tokens_base(
            logits, draft_tokens, num_contexts, batch_size, spec_metadata)

    def draft_decoder(
        self,
        logits: torch.Tensor,
        draft_model: nn.Module,
    ):
        """
        Sample draft tokens using greedy decoding.
        """
        d2t = getattr(draft_model.model, "d2t", None)
        return self._draft_sampler_greedy(logits, d2t)

    def prepare_1st_drafter_inputs(
        self,
        input_ids: torch.LongTensor,
        position_ids: torch.LongTensor,
        accepted_tokens: torch.Tensor,
        attn_metadata: AttentionMetadata,
        spec_metadata: DraftTargetOneModelSpecMetadata,
        draft_model: nn.Module,
    ):
        """
        Prepare inputs for the first draft model forward.

        Key difference from Eagle3: Pass full tokens (0..n+1 with positions 0..n+1)
        instead of Eagle3's approach (tokens 1..n+1 with positions 0..n).

        For chunked context, only add the extra token on the last chunk.
        All operations stay on GPU to maintain CUDA graph compatibility.
        """
        num_contexts = attn_metadata.num_contexts
        num_ctx_tokens = attn_metadata.num_ctx_tokens

        # Context: Keep ALL original tokens (no shift like Eagle3)
        # Eagle3 does: input_ids_ctx[:-1] = input_ids[1:] (shift left)
        # Draft Target: input_ids_ctx = input_ids (no shift)
        if num_contexts > 0:
            # CRITICAL: Build tensor with each context's tokens contiguous
            # For last chunks: [ctx_tokens, appended_token]
            # For non-last chunks: [ctx_tokens] (no appended token)

            is_last_chunk = spec_metadata.is_last_context_chunk[:num_contexts]
            is_last_chunk_cpu = is_last_chunk.cpu()  # Move to CPU to avoid implicit syncs in loop
            accepted_ctx_tokens = accepted_tokens[:num_contexts, 0]
            seq_lens_cpu = attn_metadata.seq_lens[:num_contexts]

            # Build token list by interleaving context tokens with appended tokens (only for last chunks)
            input_ids_list = []
            position_ids_list = []
            token_offset = 0

            for ctx_idx in range(num_contexts):
                seq_len = seq_lens_cpu[ctx_idx]
                # Add original context tokens
                ctx_tokens = input_ids[token_offset:token_offset + seq_len]
                ctx_positions = position_ids[token_offset:token_offset + seq_len]
                input_ids_list.append(ctx_tokens)
                position_ids_list.append(ctx_positions)

                # Only add appended token for last chunks
                if is_last_chunk_cpu[ctx_idx]:
                    input_ids_list.append(accepted_ctx_tokens[ctx_idx:ctx_idx+1])
                    # Position = last position + 1
                    next_pos = ctx_positions[-1] + 1
                    position_ids_list.append(next_pos.unsqueeze(0))

                token_offset += seq_len

            input_ids_ctx = torch.cat(input_ids_list, dim=0)
            position_ids_ctx = torch.cat(position_ids_list, dim=0)
        else:
            input_ids_ctx = torch.empty(0, dtype=input_ids.dtype, device=input_ids.device)
            position_ids_ctx = torch.empty(0, dtype=position_ids.dtype, device=position_ids.device)

        # Generation: Use all accepted tokens (full sequence)
        # This is the main path for speculative decoding
        input_ids_gen = accepted_tokens[num_contexts:, :].flatten()

        # Position IDs for generation: use the positions from target model
        if num_contexts < attn_metadata.num_seqs:
            gen_position_ids_start = position_ids[num_ctx_tokens:]
            num_gens = attn_metadata.num_seqs - num_contexts
            position_ids_gen = gen_position_ids_start.reshape(num_gens, -1).flatten()
        else:
            position_ids_gen = torch.empty(0, dtype=position_ids.dtype, device=position_ids.device)

        # Concatenate context and generation
        input_ids_draft = torch.cat([input_ids_ctx, input_ids_gen], dim=0)
        position_ids_draft = torch.cat([position_ids_ctx, position_ids_gen], dim=0)

        return {
            "input_ids": input_ids_draft,
            "position_ids": position_ids_draft,
            "attn_metadata": attn_metadata,
            "spec_metadata": spec_metadata,
        }