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

    def prepare(self):
        """Prepare the metadata before model forward."""
        assert self.request_ids is not None
        # Update batch indices
        num_seqs = len(self.request_ids)
        batch_indices = torch.arange(num_seqs, dtype=torch.int, device="cpu", pin_memory=True)
        self.batch_indices_cuda[:num_seqs].copy_(batch_indices, non_blocking=True)
        # Adjust num_tokens for generation phase
        self.num_tokens -= (self.num_generations) * self.max_draft_len


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
        # For saving/restoring spec_metadata fields
        self._saved_gather_ids = None
        self._saved_num_contexts = None
        self._saved_num_tokens = None
        # For saving/restoring attn_metadata fields
        self._saved_seq_lens = None
        self._saved_seq_lens_cuda = None
        self._saved_num_ctx_tokens = None

    @property
    def max_draft_len(self) -> int:
        """Returns the maximum draft length for this worker."""
        return self.spec_config.max_draft_len

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
        """
        Forward pass for DraftTarget one-model speculative decoding.

        Args:
            input_ids: Input token ids
            position_ids: Position ids for the tokens
            hidden_states: Hidden states from the target model
            logits: Logits from the target model
            attn_metadata: Attention metadata
            spec_metadata: Speculative decoding metadata
            draft_model: The draft model module

        Returns:
            Dictionary containing:
                - logits: Raw logits from target model
                - new_tokens: Accepted tokens
                - new_tokens_lens: Number of accepted tokens per sequence
                - next_draft_tokens: Draft tokens for next iteration
                - next_new_tokens: Combined accepted and draft tokens
        """
        batch_size = attn_metadata.num_seqs
        num_contexts = attn_metadata.num_contexts
        num_gens = batch_size - num_contexts

        raw_logits = logits

        self._execute_guided_decoder_if_present(logits)

        # Sample and accept draft tokens
        accepted_tokens, num_accepted_tokens = self.sample_and_accept_draft_tokens(
            logits, attn_metadata, spec_metadata
        )

        # Save the old attn_metadata for restoration later
        self._prepare_attn_metadata_for_spec_dec(attn_metadata)

        # Prepare inputs for the draft model forward
        position_ids = position_ids.squeeze(0)
        inputs = self.prepare_drafter_inputs(
            input_ids=input_ids,
            position_ids=position_ids,
            accepted_tokens=accepted_tokens,
            num_accepted_tokens=num_accepted_tokens,
            attn_metadata=attn_metadata,
            spec_metadata=spec_metadata,
            draft_model=draft_model,
        )

        # Generate draft tokens
        next_draft_tokens = []
        original_all_rank_num_tokens = attn_metadata.all_rank_num_tokens

        for i in range(self.max_draft_len):
            if i == 0:
                start_ids_gen = (
                    spec_metadata.batch_indices_cuda[:num_gens] * (self.max_draft_len + 1)
                ).long()
                gather_ids_gen = (
                    start_ids_gen
                    + num_accepted_tokens[num_contexts:]
                    - 1
                    + attn_metadata.num_ctx_tokens
                )
                gather_ids = torch.concat(
                    [spec_metadata.gather_ids[:num_contexts], gather_ids_gen], dim=0
                )
            else:
                # All seq_len are 1, use batch_indices_cuda as gather_ids
                gather_ids = spec_metadata.batch_indices_cuda[:batch_size]

            if self.guided_decoder is not None:
                new_tokens = inputs["input_ids"][gather_ids]
                self.guided_decoder.add_draft_batch(new_tokens, num_accepted_tokens, draft_step=i)

            # Update attn_metadata.all_rank_num_tokens for attention DP
            if original_all_rank_num_tokens is not None:
                if i == 0:
                    attn_metadata.all_rank_num_tokens = original_all_rank_num_tokens
                elif spec_metadata.all_rank_num_seqs is not None:
                    attn_metadata.all_rank_num_tokens = spec_metadata.all_rank_num_seqs

            # Run the draft model forward
            draft_hidden_states = draft_model.model(**inputs)

            # Disable spec-dec mode after first step
            attn_metadata.use_spec_decoding = False

            # Get logits from draft model
            draft_logits = draft_model.logits_processor(
                draft_hidden_states[gather_ids], draft_model.lm_head, attn_metadata, True
            )

            if self.guided_decoder is not None:
                self.guided_decoder.execute_draft_batch(draft_logits, draft_step=i)

            # Sample new draft token
            new_draft_token = self.draft_decoder(draft_logits)
            next_draft_tokens.append(new_draft_token)

            # Update inputs for next iteration
            position_ids = inputs["position_ids"][gather_ids] + 1

            # Update attn_metadata for next draft step
            if i == 0:
                attn_metadata._seq_lens[:batch_size].fill_(1)
                attn_metadata._seq_lens_cuda[:batch_size].fill_(1)

                if inputs["attn_metadata"].kv_cache_manager is not None:
                    attn_metadata.host_request_types[: attn_metadata.num_contexts].fill_(1)

                # Always reset _num_ctx_tokens and num_contexts when transitioning to generation-only mode
                # This must happen regardless of kv_cache_manager to maintain consistent metadata
                attn_metadata._num_ctx_tokens = 0
                attn_metadata.num_contexts = 0

                # Call on_update AFTER all metadata changes to ensure consistency
                attn_metadata.on_update()

                if hasattr(attn_metadata, "kv_lens_cuda"):
                    attn_metadata.kv_lens_cuda[num_contexts:batch_size] -= (
                        self.max_draft_len - num_accepted_tokens[num_contexts:]
                    )
                    # Only increment kv_lens for final chunks (that got the extra n+1 token)
                    if num_contexts > 0 and spec_metadata.is_last_context_chunk is not None:
                        is_final_chunk = spec_metadata.is_last_context_chunk[:num_contexts]
                        attn_metadata.kv_lens_cuda[:num_contexts] += is_final_chunk.to(attn_metadata.kv_lens_cuda.dtype)
                    else:
                        # Fallback: increment all contexts (backward compatibility)
                        attn_metadata.kv_lens_cuda[:num_contexts] += 1
            elif hasattr(attn_metadata, "kv_lens_cuda"):
                attn_metadata.kv_lens_cuda[:batch_size] += 1

            inputs = {
                "input_ids": new_draft_token,
                "position_ids": position_ids,
                "attn_metadata": attn_metadata,
                "spec_metadata": spec_metadata,
            }

        next_draft_tokens = torch.stack(next_draft_tokens, dim=1)

        # Restore attn_metadata
        self._restore_attn_metadata_from_spec_dec(attn_metadata)
        if original_all_rank_num_tokens is not None:
            attn_metadata.all_rank_num_tokens = original_all_rank_num_tokens

        # Restore spec_metadata fields if they were modified
        if self._saved_gather_ids is not None and self._saved_num_contexts is not None:
            spec_metadata.gather_ids[:self._saved_num_contexts] = self._saved_gather_ids
            self._saved_gather_ids = None
            self._saved_num_contexts = None
        if self._saved_num_tokens is not None:
            spec_metadata.num_tokens = self._saved_num_tokens
            self._saved_num_tokens = None

        # Restore attn_metadata fields if they were modified
        if self._saved_seq_lens is not None and self._saved_seq_lens_cuda is not None:
            batch_size_saved = len(self._saved_seq_lens)
            attn_metadata._seq_lens[:batch_size_saved] = self._saved_seq_lens
            attn_metadata._seq_lens_cuda[:batch_size_saved] = self._saved_seq_lens_cuda
            attn_metadata._num_ctx_tokens = self._saved_num_ctx_tokens
            # Call on_update to recompute derived fields from restored seq_lens
            attn_metadata.on_update()
            self._saved_seq_lens = None
            self._saved_seq_lens_cuda = None
            self._saved_num_ctx_tokens = None

        # Prepare next new tokens for overlap scheduler
        next_new_tokens = self._prepare_next_new_tokens(
            accepted_tokens,
            next_draft_tokens,
            spec_metadata.batch_indices_cuda,
            batch_size,
            num_accepted_tokens,
        )

        attn_metadata.use_spec_decoding = True

        return {
            "logits": raw_logits,
            "new_tokens": accepted_tokens,
            "new_tokens_lens": num_accepted_tokens,
            "next_draft_tokens": next_draft_tokens,
            "next_new_tokens": next_new_tokens,
        }

    def sample_and_accept_draft_tokens(
        self,
        logits: torch.Tensor,
        attn_metadata: AttentionMetadata,
        spec_metadata: DraftTargetOneModelSpecMetadata,
    ):
        """
        Sample tokens and verify draft tokens against target model logits.

        Args:
            logits: Logits from the target model
            attn_metadata: Attention metadata
            spec_metadata: Speculative decoding metadata

        Returns:
            Tuple of:
                - accepted_tokens: Tensor of accepted token ids
                - num_accepted_tokens: Number of accepted tokens per sequence
        """
        batch_size = attn_metadata.num_seqs
        num_contexts = attn_metadata.num_contexts
        num_gens = batch_size - num_contexts

        draft_tokens = spec_metadata.draft_tokens.reshape(num_gens, self.max_draft_len)

        return self._sample_and_accept_draft_tokens_base(
            logits, draft_tokens, num_contexts, batch_size, spec_metadata
        )

    def draft_decoder(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Sample draft tokens from logits.

        Uses greedy decoding for draft tokens as it's faster and doesn't
        significantly affect acceptance rate.

        Args:
            logits: Logits from the draft model

        Returns:
            Sampled draft token ids
        """
        return self._draft_sampler_greedy(logits)

    def _prepare_context_input_ids(self, input_ids, num_ctx_tokens, gather_ids,
                                   accepted_tokens, num_contexts, spec_metadata):
        """
        Prepare context input IDs for draft model forward (DraftTarget variant).

        Unlike Eagle3/MTP which use tokens 1..n+1 (shift left, remove token 0),
        DraftTarget uses tokens 0..n+1 (keep token 0, add token n+1).

        For chunked context:
        - Non-final chunks: Do NOT append n+1 token (still processing prompt)
        - Final chunks: Append n+1 token (ready for first prediction)

        Args:
            input_ids: Original input IDs tensor (flattened context tokens)
            num_ctx_tokens: Number of context tokens in the batch
            gather_ids: Indices of last token position for each context request
            accepted_tokens: [batch_size, max_draft_len + 1] - Accepted tokens
            num_contexts: Number of context requests
            spec_metadata: Speculative decoding metadata with is_last_context_chunk flags

        Returns:
            input_ids_ctx: Prepared context input IDs with shape [num_ctx_tokens + num_final_chunks]
        """
        input_prompt_ids = input_ids[:num_ctx_tokens]

        # Determine which contexts are in their final chunk
        # TODO: Populate is_last_context_chunk from LlmRequest.is_last_context_chunk
        # in the executor/scheduler when creating spec_metadata
        if spec_metadata.is_last_context_chunk is not None:
            is_final_chunk = spec_metadata.is_last_context_chunk[:num_contexts]
            num_final_chunks = is_final_chunk.sum()  # Keep as tensor to avoid host sync
        else:
            # Fallback: Assume all contexts are either non-chunked or final chunks
            # This maintains backward compatibility but won't work correctly with
            # multi-chunk prefill where intermediate chunks shouldn't add n+1 token
            num_final_chunks = num_contexts
            is_final_chunk = torch.ones(num_contexts, dtype=torch.bool, device="cuda")

        # Allocate buffer: original context tokens + n+1 tokens for final chunks only
        # Convert num_final_chunks to int for allocation (unavoidable host sync for allocation)
        num_final_chunks_int = num_final_chunks.item() if isinstance(num_final_chunks, torch.Tensor) else num_final_chunks
        input_ids_ctx = torch.empty(num_ctx_tokens + num_final_chunks_int,
                                    dtype=torch.int32,
                                    device="cuda")

        # Copy all original context tokens (no left shift - this is key difference from Eagle/MTP)
        input_ids_ctx[:num_ctx_tokens].copy_(input_prompt_ids)

        # Append n+1 tokens only for final chunks
        # The n+1 token is the first accepted token (accepted_tokens[:, 0])
        if num_final_chunks_int > 0:
            # Use vectorized operations to avoid loops and host syncs
            # Find indices of final chunks
            final_chunk_indices = torch.nonzero(is_final_chunk, as_tuple=True)[0]

            # Calculate append positions for each final chunk
            # append_pos = gather_ids[i] + 1 + offset (where offset is the cumulative count)
            cumulative_offsets = torch.arange(len(final_chunk_indices), dtype=torch.long, device="cuda")
            append_positions = gather_ids[final_chunk_indices] + 1 + cumulative_offsets

            # Place accepted tokens at append positions
            input_ids_ctx[append_positions] = accepted_tokens[final_chunk_indices, 0]

        return input_ids_ctx, num_final_chunks_int

    def prepare_drafter_inputs(
        self,
        input_ids: torch.LongTensor,
        position_ids: torch.LongTensor,
        accepted_tokens: torch.Tensor,
        num_accepted_tokens: torch.Tensor,
        attn_metadata: AttentionMetadata,
        spec_metadata: DraftTargetOneModelSpecMetadata,
        draft_model: nn.Module,
    ):
        """
        Prepare inputs for the draft model forward.

        Unlike Eagle3/MTP, DraftTarget does not pass hidden states from the
        target model to the draft model. The draft model operates independently
        with its own embeddings and forward pass.

        DraftTarget uses tokens 0..n+1 (not 1..n+1 like Eagle/MTP), meaning:
        - Context phase: Keep all original tokens + append n+1 (first prediction)
        - Generation phase: Use all accepted_tokens columns (0..max_draft_len)

        Args:
            input_ids: Input token ids
            position_ids: Position ids
            accepted_tokens: Accepted token ids [batch_size, max_draft_len + 1]
            num_accepted_tokens: Number of accepted tokens per sequence
            attn_metadata: Attention metadata
            spec_metadata: Speculative decoding metadata
            draft_model: The draft model module

        Returns:
            Dictionary of inputs for draft model forward
        """
        num_contexts = attn_metadata.num_contexts
        batch_size = attn_metadata.num_seqs
        num_gens = batch_size - num_contexts

        # Reset saved state at the beginning of each call
        self._saved_gather_ids = None
        self._saved_num_contexts = None
        self._saved_num_tokens = None
        self._saved_seq_lens = None
        self._saved_seq_lens_cuda = None
        self._saved_num_ctx_tokens = None

        # Prepare context phase: use updated _prepare_context_input_ids
        if num_contexts > 0:
            # Debug: log original state before any modifications
            if num_gens > 0:
                import sys
                print(f"[DraftTarget Debug] BEFORE updates: num_contexts={num_contexts}, num_gens={num_gens}", file=sys.stderr)
                print(f"[DraftTarget Debug] BEFORE seq_lens: {attn_metadata._seq_lens[:batch_size]}", file=sys.stderr)
                print(f"[DraftTarget Debug] BEFORE _num_ctx_tokens: {attn_metadata.num_ctx_tokens}", file=sys.stderr)

            # Use gather_ids from spec_metadata (populated by executor)
            # These point to the last token position for each context
            gather_ids = spec_metadata.gather_ids[:num_contexts]

            # Save original num_ctx_tokens before updating metadata
            original_num_ctx_tokens = attn_metadata.num_ctx_tokens

            input_ids_ctx, num_final_chunks = self._prepare_context_input_ids(
                input_ids,
                original_num_ctx_tokens,
                gather_ids,
                accepted_tokens,
                num_contexts,
                spec_metadata
            )

            # Save original spec_metadata fields before modifications
            self._saved_num_tokens = spec_metadata.num_tokens

            # Save original attn_metadata fields before modifications (deep copy)
            self._saved_seq_lens = attn_metadata._seq_lens[:batch_size].clone()
            self._saved_seq_lens_cuda = attn_metadata._seq_lens_cuda[:batch_size].clone()
            self._saved_num_ctx_tokens = attn_metadata._num_ctx_tokens

            # Update attention metadata to account for extra tokens
            self._update_attn_metadata_for_extra_tokens(attn_metadata, num_contexts, num_gens, num_final_chunks, spec_metadata)

            # Update spec metadata
            self._update_spec_metadata_for_extra_tokens(spec_metadata, num_final_chunks)

            # Build position IDs for context with extra positions for n+1 tokens
            position_ids_ctx = self._build_context_position_ids(
                position_ids,
                original_num_ctx_tokens,
                num_contexts,
                gather_ids,
                num_final_chunks,
                spec_metadata
            )

            # Update gather_ids in spec_metadata to account for position shifts from inserted tokens
            # Save original gather_ids so we can restore them later (in forward method)
            self._saved_gather_ids = spec_metadata.gather_ids[:num_contexts].clone()
            self._saved_num_contexts = num_contexts
            updated_gather_ids = self._update_gather_ids_for_extra_tokens(
                gather_ids,
                num_contexts,
                spec_metadata
            )
            # Update the first num_contexts entries in spec_metadata.gather_ids
            spec_metadata.gather_ids[:num_contexts] = updated_gather_ids
        else:
            input_ids_ctx = None
            position_ids_ctx = None
            num_final_chunks = 0

        # Prepare generation phase: use accepted tokens for generation phase.
        # Generation requests are laid out contiguously after all context tokens,
        # each with fixed length (max_draft_len + 1). Replace that region in one
        # vectorized write to keep CUDA-graph compatibility.
        if num_gens > 0:
            # Note: accepted_tokens[:, :] uses ALL columns (0..max_draft_len), which is correct for DraftTarget
            gen_tokens_flat = accepted_tokens[num_contexts:, :].reshape(-1)
            gen_tokens_flat = gen_tokens_flat.to(input_ids.dtype)

            # Start position for generation tokens
            # Note: attn_metadata.num_ctx_tokens already includes num_final_chunks (updated at line 428)
            start = attn_metadata.num_ctx_tokens
            end = start + gen_tokens_flat.numel()

            # Clone and update input_ids
            if input_ids_ctx is not None:
                # Combine context and generation tokens
                draft_input_ids = torch.empty(end, dtype=input_ids.dtype, device="cuda")
                draft_input_ids[:len(input_ids_ctx)] = input_ids_ctx
                draft_input_ids[start:end] = gen_tokens_flat

                # Build position IDs for generation tokens
                # Extract generation position IDs from original position_ids and increment by 1
                # (since we're preparing to generate the next token)
                # Read from original range, not the updated range
                gen_end = original_num_ctx_tokens + gen_tokens_flat.numel()
                original_gen_position_ids = position_ids[original_num_ctx_tokens:gen_end]
                gen_position_ids = original_gen_position_ids + 1

                # Combine context and generation position IDs
                draft_position_ids = torch.empty(end, dtype=position_ids.dtype, device="cuda")
                draft_position_ids[:len(position_ids_ctx)] = position_ids_ctx
                draft_position_ids[start:end] = gen_position_ids
            else:
                # Generation only
                draft_input_ids = input_ids.clone()
                draft_input_ids[start:end] = gen_tokens_flat

                # Increment generation position IDs by 1
                draft_position_ids = position_ids.clone()
                draft_position_ids[start:end] += 1
        else:
            # Context only
            draft_input_ids = input_ids_ctx
            draft_position_ids = position_ids_ctx

        return {
            "input_ids": draft_input_ids,
            "position_ids": draft_position_ids,
            "attn_metadata": attn_metadata,
            "spec_metadata": spec_metadata,
        }

    def _update_attn_metadata_for_extra_tokens(self, attn_metadata, num_contexts, num_gens, num_final_chunks, spec_metadata):
        """
        Update attention metadata to account for extra n+1 tokens in final context chunks.

        Args:
            attn_metadata: Attention metadata to update
            num_contexts: Number of context requests
            num_final_chunks: Number of contexts in their final chunk (that get n+1 token)
            spec_metadata: Spec metadata with is_last_context_chunk flags
        """
        if num_final_chunks > 0:
            # Increment sequence lengths for final chunk contexts
            # on_update() will recompute _num_ctx_tokens from these seq_lens
            if num_final_chunks == num_contexts:
                # Simple case: all contexts are final chunks
                attn_metadata._seq_lens[:num_contexts] += 1
                attn_metadata._seq_lens_cuda[:num_contexts] += 1
            else:
                # Complex case: selectively update only final chunks
                # Use vectorized operations to avoid host sync
                if spec_metadata.is_last_context_chunk is not None:
                    is_final_chunk = spec_metadata.is_last_context_chunk[:num_contexts]
                    # Increment only the final chunks
                    attn_metadata._seq_lens[:num_contexts] += is_final_chunk.cpu().to(attn_metadata._seq_lens.dtype)
                    attn_metadata._seq_lens_cuda[:num_contexts] += is_final_chunk.to(attn_metadata._seq_lens_cuda.dtype)
                else:
                    # Fallback: assume all are final chunks
                    attn_metadata._seq_lens[:num_contexts] += 1
                    attn_metadata._seq_lens_cuda[:num_contexts] += 1

            # Recompute derived fields (_num_ctx_tokens will be computed from seq_lens)
            attn_metadata.on_update()

            # Debug logging for mixed batch scenario
            if num_gens > 0:
                import sys
                print(f"[DraftTarget Debug] AFTER updates: num_contexts={num_contexts}, batch_size={attn_metadata.num_seqs}", file=sys.stderr)
                print(f"[DraftTarget Debug] AFTER seq_lens: {attn_metadata._seq_lens[:attn_metadata.num_seqs]}", file=sys.stderr)
                print(f"[DraftTarget Debug] AFTER _num_ctx_tokens: {attn_metadata._num_ctx_tokens}", file=sys.stderr)
                print(f"[DraftTarget Debug] Context seq_lens: {attn_metadata._seq_lens[:num_contexts]}", file=sys.stderr)
                print(f"[DraftTarget Debug] Generation seq_lens: {attn_metadata._seq_lens[num_contexts:attn_metadata.num_seqs]}", file=sys.stderr)

    def _update_spec_metadata_for_extra_tokens(self, spec_metadata, num_final_chunks):
        """
        Update spec metadata to account for extra tokens in context phase.

        Args:
            spec_metadata: Speculative decoding metadata to update
            num_final_chunks: Number of contexts in their final chunk
        """
        if num_final_chunks > 0:
            spec_metadata.num_tokens += num_final_chunks

    def _update_gather_ids_for_extra_tokens(self, gather_ids, num_contexts, spec_metadata):
        """
        Update gather_ids to account for position shifts from inserted tokens.

        When we insert n+1 tokens for final chunks, all subsequent gather_ids need to
        be shifted by the cumulative number of insertions before them.

        Args:
            gather_ids: Original gather_ids (last token position for each context)
            num_contexts: Number of context requests
            spec_metadata: Spec metadata with is_last_context_chunk flags

        Returns:
            Updated gather_ids accounting for position shifts
        """
        if spec_metadata.is_last_context_chunk is None:
            # Fallback: assume all contexts are final chunks
            is_final_chunk = torch.ones(num_contexts, dtype=torch.bool, device="cuda")
        else:
            is_final_chunk = spec_metadata.is_last_context_chunk[:num_contexts]

        # Check if any tokens were inserted
        num_final_chunks = is_final_chunk.sum()
        if num_final_chunks == 0:
            return gather_ids

        # For each position i, calculate how many tokens were inserted at positions <= i
        # This is the cumulative sum of is_final_chunk
        cumulative_insertions = torch.cumsum(is_final_chunk.to(torch.long), dim=0)

        # Update gather_ids: each position shifts by the number of insertions before it
        # For final chunks, they also get their own insertion, so use cumulative_insertions directly
        # For non-final chunks, they shift by the insertions before them
        updated_gather_ids = gather_ids + cumulative_insertions

        return updated_gather_ids

    def _build_context_position_ids(self, position_ids, original_num_ctx_tokens,
                                    num_contexts, gather_ids, num_final_chunks,
                                    spec_metadata):
        """
        Build position IDs for context phase with extra positions for n+1 tokens.

        Args:
            position_ids: Original position IDs
            original_num_ctx_tokens: Original number of context tokens (before adding extra tokens)
            num_contexts: Number of context requests
            gather_ids: Last token position for each context
            num_final_chunks: Number of contexts in their final chunk
            spec_metadata: Spec metadata with is_last_context_chunk flags

        Returns:
            Position IDs with extra positions for n+1 tokens
        """
        if num_final_chunks == 0:
            return position_ids[:original_num_ctx_tokens]

        # Allocate buffer for context position IDs with extra positions
        position_ids_ctx = torch.empty(original_num_ctx_tokens + num_final_chunks,
                                       dtype=torch.int32,
                                       device="cuda")

        # Copy original context position IDs
        position_ids_ctx[:original_num_ctx_tokens] = position_ids[:original_num_ctx_tokens]

        # Add position IDs for n+1 tokens (one position after each final chunk)
        if spec_metadata.is_last_context_chunk is not None:
            is_final_chunk = spec_metadata.is_last_context_chunk[:num_contexts]
        else:
            is_final_chunk = torch.ones(num_contexts, dtype=torch.bool, device="cuda")

        # Use vectorized operations to avoid loops and host syncs
        final_chunk_indices = torch.nonzero(is_final_chunk, as_tuple=True)[0]

        if len(final_chunk_indices) > 0:
            # Calculate append positions and last positions vectorized
            cumulative_offsets = torch.arange(len(final_chunk_indices), dtype=torch.long, device="cuda")
            append_positions = gather_ids[final_chunk_indices] + 1 + cumulative_offsets

            # Get last positions for final chunks (no .item() call)
            last_positions = position_ids[gather_ids[final_chunk_indices]]

            # Set position IDs to last_position + 1
            position_ids_ctx[append_positions] = last_positions + 1

        return position_ids_ctx
