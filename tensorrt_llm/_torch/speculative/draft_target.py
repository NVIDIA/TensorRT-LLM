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

    def __post_init__(self):
        self.batch_indices_cuda = torch.empty(
            [self.max_num_requests],
            dtype=torch.int,
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
                attn_metadata.on_update()
                if inputs["attn_metadata"].kv_cache_manager is not None:
                    attn_metadata.host_request_types[: attn_metadata.num_contexts].fill_(1)
                    attn_metadata.num_contexts = 0
                if hasattr(attn_metadata, "kv_lens_cuda"):
                    attn_metadata.kv_lens_cuda[num_contexts:batch_size] -= (
                        self.max_draft_len - num_accepted_tokens[num_contexts:]
                    )
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

        Args:
            input_ids: Input token ids
            position_ids: Position ids
            accepted_tokens: Accepted token ids
            num_accepted_tokens: Number of accepted tokens per sequence
            attn_metadata: Attention metadata
            spec_metadata: Speculative decoding metadata
            draft_model: The draft model module

        Returns:
            Dictionary of inputs for draft model forward
        """
        num_contexts = attn_metadata.num_contexts
        batch_size = attn_metadata.num_seqs

        # Prepare input tokens - use accepted tokens for generation phase.
        # Generation requests are laid out contiguously after all context tokens,
        # each with fixed length (max_draft_len + 1). Replace that region in one
        # vectorized write to keep CUDA-graph compatibility.
        draft_input_ids = input_ids.clone()
        num_gens = batch_size - num_contexts
        if num_gens > 0:
            gen_tokens_flat = accepted_tokens[num_contexts:, :].reshape(-1)
            gen_tokens_flat = gen_tokens_flat.to(draft_input_ids.dtype)
            start = attn_metadata.num_ctx_tokens
            end = start + gen_tokens_flat.numel()
            draft_input_ids[start:end] = gen_tokens_flat

        return {
            "input_ids": draft_input_ids,
            "position_ids": position_ids,
            "attn_metadata": attn_metadata,
            "spec_metadata": spec_metadata,
        }
