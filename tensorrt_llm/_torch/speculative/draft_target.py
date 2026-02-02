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
        self.num_tokens -= self.num_generations * self.max_draft_len
        self.is_spec_dec_tree = False
        self.is_spec_dec_dynamic_tree = False


class DraftTargetOneModelSampler(MTPSampler):
    """
    Sampler for DraftTarget one-model speculative decoding.

    Inherits from MTPSampler to reuse the speculative decoding sampling logic.
    """

    def __init__(self, args: TorchSampler.Args):
        super().__init__(args, nextn=args.max_draft_len)


class DraftTargetOneModelWorker(SpecWorkerBase):
    def __init__(
        self,
        spec_config: "DraftTargetDecodingConfig",
        mapping: Mapping,
        use_separate_draft_kv_cache: bool = False,
    ):
        super().__init__(use_separate_draft_kv_cache)
        self.spec_config = spec_config
        self.mapping = mapping

    @property
    def max_draft_len(self) -> int:
        return self.spec_config.max_draft_len

    def forward(
        self,
        input_ids,
        position_ids,
        hidden_states,
        logits,
        attn_metadata: AttentionMetadata,
        spec_metadata: DraftTargetOneModelSpecMetadata,
        draft_model: nn.Module,
        resource_manager=None,
    ):
        """
        Technically incorrect at the moment.
        Leverages Eagle3/MTP setup that does this for the context
        input_ids_ctx[:-1].copy_(input_prompt_ids[1:])
        In DraftTarget, we do not want to shift, which necessitates increasing the final chunk of each request by 1
        for the final accepted token.  This creates a big headache since then the kv lens, seq_lens, token counts all
        have to be updated and then reverted when heading back to the target.  TODO: non trivially fix this issue.
        """

        batch_size = attn_metadata.num_seqs
        num_contexts = attn_metadata.num_contexts
        num_gens = batch_size - num_contexts

        raw_logits = logits

        self._execute_guided_decoder_if_present(logits)

        accepted_tokens, num_accepted_tokens = self.sample_and_accept_draft_tokens(
            logits, attn_metadata, spec_metadata
        )

        # Prepare attention metadata for speculative decoding and save state for restore
        self._prepare_attn_metadata_for_spec_dec(attn_metadata)

        # Prepare inputs for the first draft forward
        position_ids = position_ids.squeeze(0)
        inputs = self.prepare_1st_drafter_inputs(
            input_ids=input_ids,
            position_ids=position_ids,
            accepted_tokens=accepted_tokens,
            attn_metadata=attn_metadata,
            spec_metadata=spec_metadata,
        )

        next_draft_tokens = []
        original_all_rank_num_tokens = attn_metadata.all_rank_num_tokens

        # Get the draft KV cache manager if using separate layouts
        draft_kv_cache_manager = self.get_draft_kv_cache_manager(resource_manager)

        with self.draft_kv_cache_context(attn_metadata, draft_kv_cache_manager):
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
                    gather_ids = spec_metadata.batch_indices_cuda[:batch_size]

                if self.guided_decoder is not None:
                    new_tokens = inputs["input_ids"][gather_ids]
                    self.guided_decoder.add_draft_batch(
                        new_tokens, num_accepted_tokens, draft_step=i
                    )

                if original_all_rank_num_tokens is not None:
                    if i == 0:
                        attn_metadata.all_rank_num_tokens = original_all_rank_num_tokens
                    elif spec_metadata.all_rank_num_seqs is not None:
                        attn_metadata.all_rank_num_tokens = spec_metadata.all_rank_num_seqs

                hidden_states = draft_model.model(**inputs)
                if isinstance(hidden_states, tuple):
                    hidden_states = hidden_states[0]

                # Disable spec-dec mode for chained draft steps
                attn_metadata.use_spec_decoding = False

                logits = draft_model.logits_processor(
                    hidden_states[gather_ids], draft_model.lm_head, attn_metadata, True
                )
                if self.guided_decoder is not None:
                    d2t = getattr(draft_model.model, "d2t", None)
                    self.guided_decoder.execute_draft_batch(logits, d2t, draft_step=i)

                new_draft_token = self.draft_decoder(logits, draft_model)
                next_draft_tokens.append(new_draft_token)

                # Update inputs and metadata for next draft step
                position_ids = inputs["position_ids"][gather_ids] + 1
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

        # Restore attention metadata to original state
        self._restore_attn_metadata_from_spec_dec(attn_metadata)
        if original_all_rank_num_tokens is not None:
            attn_metadata.all_rank_num_tokens = original_all_rank_num_tokens

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
        batch_size = attn_metadata.num_seqs
        num_contexts = attn_metadata.num_contexts
        num_gens = batch_size - num_contexts

        if spec_metadata.draft_tokens is None:
            draft_tokens = torch.zeros(
                (num_gens, self.max_draft_len), dtype=torch.int, device=logits.device
            )
        else:
            draft_tokens = spec_metadata.draft_tokens.reshape(num_gens, self.max_draft_len)

        return self._sample_and_accept_draft_tokens_base(
            logits, draft_tokens, num_contexts, batch_size, spec_metadata
        )

    def draft_decoder(
        self,
        logits: torch.Tensor,
        draft_model: nn.Module,
    ):
        d2t = getattr(draft_model.model, "d2t", None)
        return self._draft_sampler_greedy(logits, d2t)

    def prepare_1st_drafter_inputs(
        self,
        input_ids: torch.LongTensor,
        position_ids: torch.LongTensor,
        accepted_tokens: torch.Tensor,
        attn_metadata: AttentionMetadata,
        spec_metadata: DraftTargetOneModelSpecMetadata,
    ):
        num_contexts = attn_metadata.num_contexts

        input_ids_ctx = self._prepare_context_input_ids(
            input_ids,
            attn_metadata.num_ctx_tokens,
            spec_metadata.gather_ids,
            accepted_tokens,
            num_contexts,
        )

        input_ids_gen = accepted_tokens[num_contexts:, :].flatten()
        input_ids = torch.concat([input_ids_ctx, input_ids_gen], dim=0)

        return {
            "input_ids": input_ids,
            "position_ids": position_ids,
            "attn_metadata": attn_metadata,
            "spec_metadata": spec_metadata,
        }
