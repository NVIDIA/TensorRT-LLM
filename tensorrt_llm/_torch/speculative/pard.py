from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import torch
from torch import nn

from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping

from ..attention_backend import AttentionMetadata
from .interface import SpecMetadata, SpecWorkerBase

if TYPE_CHECKING:
    from ...llmapi.llm_args import PARDDecodingConfig


@dataclass
class PARDSpecMetadata(SpecMetadata):
    """Metadata for PARD speculative decoding."""

    batch_indices_cuda: Optional[torch.Tensor] = None

    def __post_init__(self):
        self.batch_indices_cuda = torch.empty(
            [self.max_num_requests],
            dtype=torch.int,
            device="cuda",
        )

        self.is_spec_dec_tree = False
        self.is_spec_dec_dynamic_tree = False

    def prepare(self):
        assert self.request_ids is not None

        num_seqs = len(self.request_ids)
        batch_indices = torch.arange(num_seqs, dtype=torch.int, device="cpu", pin_memory=True)
        self.batch_indices_cuda[:num_seqs].copy_(batch_indices, non_blocking=True)


class PARDWorker(SpecWorkerBase):
    """
    Worker for PARD (PARallel Draft) speculative decoding.

    PARD is a target-independent method: the draft model relies only on its
    own embeddings and mask tokens, not target hidden states.  All K draft
    tokens are predicted in a single forward pass:
        Input:  [accepted_tokens, mask_0, ..., mask_(K-1)]
        Output: K draft tokens from K positions in parallel.

    Reference: https://arxiv.org/pdf/2504.18583
    """

    def __init__(
        self,
        spec_config: "PARDDecodingConfig",
        mapping: Mapping,
        use_separate_draft_kv_cache: bool = False,
    ):
        super().__init__(use_separate_draft_kv_cache)
        self.spec_config = spec_config
        self.mapping = mapping
        logger.info(
            f"PARDWorker initialized with use_separate_draft_kv_cache={use_separate_draft_kv_cache}"
        )

    @property
    def max_draft_len(self) -> int:
        return self.spec_config.max_draft_len

    @property
    def _draft_tokens_per_req(self) -> int:
        """Total tokens per gen request in the draft forward.

        Uses 2K to fit all accepted tokens (up to K+1) plus K-1 mask tokens,
        ensuring K unique predictions regardless of how many tokens were accepted.
        """
        return 2 * self.max_draft_len

    def _prepare_attn_metadata_for_pard(self, attn_metadata, spec_metadata):
        """
        Save attn_metadata fields that PARD modifies during forward.

        CUDA graph handling: During warmup, save and restore kv_lens_cuda to
        prevent accumulation. During capture, don't save kv_lens_cuda so the
        modification is captured but restoration is not — ensuring the
        modification persists correctly during replay.
        """
        is_capturing = torch.cuda.is_current_stream_capturing()

        if spec_metadata.is_cuda_graph and not is_capturing:
            attn_metadata.prepare_for_spec_dec("_seq_lens", "_seq_lens_cuda", "kv_lens_cuda")
        else:
            attn_metadata.prepare_for_spec_dec("_seq_lens", "_seq_lens_cuda")

    def _prepare_kv_for_draft_forward(
        self,
        attn_metadata,
        num_accepted_tokens: torch.Tensor,
        num_contexts: int,
        batch_size: int,
    ):
        """
        Adjust kv_lens_cuda so the draft model sees correct RoPE positions.

        The attention kernel computes positions as
        ``position_i = kv_lens - seq_lens + i``.  The batch manager sets
        ``kv_lens = past_seen + 2K`` and the draft uses ``seq_lens = 2K``,
        giving ``position_0 = past_seen``.  The target's bonus token sits
        at past_seen, so the draft's first accepted token (acc_0) needs
        position past_seen + 1.  We add 1 to kv_lens_cuda to achieve this.

        The corresponding rewind is deferred to _apply_kv_rewind_after_draft.
        """
        if hasattr(attn_metadata, "kv_lens_cuda"):
            # Rewind amount: after draft forward, subtract (1 - num_accepted)
            # so kv_lens lands at past_seen + 2K + num_accepted for the next
            # target iteration.
            self._kv_rewind_amount = 1 - num_accepted_tokens[num_contexts:batch_size]
            self._kv_rewind_nc = num_contexts
            self._kv_rewind_bs = batch_size

            if batch_size > num_contexts:
                attn_metadata.kv_lens_cuda[num_contexts:batch_size] += 1

            attn_metadata.update_for_spec_dec()

    def _apply_kv_rewind_after_draft(self, attn_metadata, spec_metadata):
        """
        Apply the deferred kv_lens rewind after the draft forward.

        Skipped during CUDA graph warmup (where kv_lens_cuda is saved/restored
        by prepare_for_spec_dec) to avoid cumulative shrinkage.  Applied during
        capture and normal inference.
        """
        is_warmup = spec_metadata.is_cuda_graph and not torch.cuda.is_current_stream_capturing()
        if is_warmup:
            return

        if hasattr(self, "_kv_rewind_amount") and hasattr(attn_metadata, "kv_lens_cuda"):
            nc = self._kv_rewind_nc
            bs = self._kv_rewind_bs
            attn_metadata.kv_lens_cuda[nc:bs] -= self._kv_rewind_amount
            attn_metadata.kv_lens_cuda[nc:bs].clamp_(min=0)

    def forward(
        self,
        input_ids,
        position_ids,
        hidden_states,
        logits,
        attn_metadata,
        spec_metadata,
        draft_model,
        resource_manager=None,
    ):
        batch_size = attn_metadata.num_seqs
        num_contexts = attn_metadata.num_contexts
        num_gens = batch_size - num_contexts

        raw_logits = logits
        K = self.max_draft_len

        self._execute_guided_decoder_if_present(logits)

        # draft_tokens buffer has (2K-1) entries per gen request; extract the K real drafts
        if num_gens > 0:
            draft_tokens = spec_metadata.draft_tokens.reshape(num_gens, 2 * K - 1)[:, :K]
        else:
            draft_tokens = spec_metadata.draft_tokens.reshape(0, K)

        # logits have 2K entries per gen request; extract K+1 for acceptance
        if num_gens > 0:
            ctx_logits = logits[:num_contexts]
            vocab_size = logits.shape[-1]
            gen_logits_2k = logits[num_contexts:].reshape(num_gens, 2 * K, vocab_size)
            gen_logits_kp1 = gen_logits_2k[:, : K + 1, :].reshape(-1, vocab_size)
            logits_for_accept = torch.cat([ctx_logits, gen_logits_kp1], dim=0)
        else:
            logits_for_accept = logits

        accepted_tokens, num_accepted_tokens = self._sample_and_accept_draft_tokens_base(
            logits_for_accept, draft_tokens, num_contexts, batch_size, spec_metadata
        )

        # Pad accepted_tokens from (batch, K+1) to (batch, 2K) to match sampler buffer
        if K > 1:
            acc_padding = torch.zeros(
                (batch_size, K - 1), dtype=accepted_tokens.dtype, device=accepted_tokens.device
            )
            accepted_tokens = torch.cat([accepted_tokens, acc_padding], dim=1)

        self._prepare_attn_metadata_for_pard(attn_metadata, spec_metadata)
        self._prepare_kv_for_draft_forward(
            attn_metadata, num_accepted_tokens, num_contexts, batch_size
        )

        position_ids = position_ids.squeeze(0)
        inputs = self.prepare_1st_drafter_inputs(
            input_ids=input_ids,
            position_ids=position_ids,
            hidden_states=hidden_states,
            accepted_tokens=accepted_tokens,
            num_accepted_tokens=num_accepted_tokens,
            attn_metadata=attn_metadata,
            spec_metadata=spec_metadata,
            draft_model=draft_model,
        )

        draft_kv_cache_manager = self.get_draft_kv_cache_manager(resource_manager)

        if num_gens > 0:
            with self.draft_kv_cache_context(attn_metadata, draft_kv_cache_manager):
                hidden_states_out = draft_model.model(**inputs)

                # Gather K logits per gen request starting at the bonus position.
                # Layout: [acc_0..acc_M, masks] (2K total). Positions M..M+K-1
                # produce K unique draft predictions (bonus + K-1 masks).
                gen_start_idx = attn_metadata.num_ctx_tokens

                request_bases = (
                    torch.arange(num_gens, dtype=torch.long, device="cuda")
                    * self._draft_tokens_per_req
                    + gen_start_idx
                )

                gen_num_accepted = num_accepted_tokens[num_contexts:batch_size].long()
                base_offsets = gen_num_accepted - 1  # M = bonus position
                offsets = torch.arange(self.max_draft_len, dtype=torch.long, device="cuda")

                gen_gather_ids = (
                    request_bases.unsqueeze(1) + base_offsets.unsqueeze(1) + offsets.unsqueeze(0)
                ).flatten()
                gen_gather_ids = gen_gather_ids.clamp(max=hidden_states_out.shape[0] - 1)

                gen_logits = draft_model.logits_processor(
                    hidden_states_out[gen_gather_ids], draft_model.lm_head, attn_metadata, True
                )

                vocab_size = gen_logits.shape[-1]
                gen_logits = gen_logits.reshape(num_gens, self.max_draft_len, vocab_size)

                # Use torch.argmax directly to avoid cute_argmax stride issues
                d2t = getattr(draft_model.model, "d2t", None)
                gen_draft_tokens = torch.argmax(gen_logits, dim=-1, keepdim=False).long()

                if d2t is not None:
                    gen_draft_tokens = d2t[gen_draft_tokens] + gen_draft_tokens

                gen_draft_tokens = gen_draft_tokens.type(torch.int32)

                # Pad from (num_gens, K) to (num_gens, 2K-1).
                if K > 1:
                    pad = torch.zeros((num_gens, K - 1), dtype=torch.int32, device="cuda")
                    gen_draft_tokens = torch.cat([gen_draft_tokens, pad], dim=1)

        elif num_contexts > 0 and self.use_separate_draft_kv_cache:
            # Pure context batch: populate the draft KV cache so it's
            # ready when generation starts.
            with self.draft_kv_cache_context(attn_metadata, draft_kv_cache_manager):
                draft_model.model(**inputs)
            gen_draft_tokens = torch.empty((0, 2 * K - 1), dtype=torch.int32, device="cuda")

        else:
            gen_draft_tokens = torch.empty((0, 2 * K - 1), dtype=torch.int32, device="cuda")

        if num_contexts > 0 and num_gens > 0:
            ctx_draft_tokens = torch.zeros(
                (num_contexts, 2 * K - 1), dtype=torch.int32, device="cuda"
            )
            next_draft_tokens = torch.cat([ctx_draft_tokens, gen_draft_tokens], dim=0)
        elif num_contexts > 0:
            next_draft_tokens = torch.zeros(
                (num_contexts, 2 * K - 1), dtype=torch.int32, device="cuda"
            )
        else:
            next_draft_tokens = gen_draft_tokens

        self._restore_attn_metadata_from_spec_dec(attn_metadata)

        # Deferred kv_lens rewind (must happen after restore so it persists).
        self._apply_kv_rewind_after_draft(attn_metadata, spec_metadata)

        next_new_tokens = self._prepare_next_new_tokens(
            accepted_tokens,
            next_draft_tokens,
            spec_metadata.batch_indices_cuda,
            batch_size,
            num_accepted_tokens,
        )

        return {
            "logits": raw_logits,
            "new_tokens": accepted_tokens,
            "new_tokens_lens": num_accepted_tokens,
            "next_draft_tokens": next_draft_tokens,
            "next_new_tokens": next_new_tokens,
        }

    def draft_decoder(
        self,
        logits: torch.Tensor,
        draft_model: nn.Module,
    ):
        """
        Sample draft tokens using greedy decoding.

        Args:
            logits: [num_tokens, vocab_size] from the draft model.
            draft_model: The draft model (used to read the d2t mapping).

        Returns:
            draft_tokens: [batch_size * max_draft_len] flattened token ids.
        """
        d2t = getattr(draft_model.model, "d2t", None)
        return self._draft_sampler_greedy(logits, d2t)

    def prepare_1st_drafter_inputs(
        self,
        input_ids: torch.LongTensor,
        position_ids: torch.LongTensor,
        hidden_states: torch.Tensor,
        accepted_tokens: torch.Tensor,
        num_accepted_tokens: torch.Tensor,
        attn_metadata: AttentionMetadata,
        spec_metadata: PARDSpecMetadata,
        draft_model: nn.Module,
    ):
        """
        Prepare inputs for PARD draft model.

        Gen request layout: [acc_0, ..., acc_M, masks] (2K total).
        All accepted tokens provide in-sequence context for the mask tokens,
        ensuring K unique predictions from positions M..M+K-1.
        """
        num_contexts = attn_metadata.num_contexts
        batch_size = attn_metadata.num_seqs
        num_gens = batch_size - num_contexts

        if (
            hasattr(self.spec_config, "mask_token_id")
            and self.spec_config.mask_token_id is not None
        ):
            mask_token_id = self.spec_config.mask_token_id
        elif hasattr(draft_model, "mask_token_id"):
            mask_token_id = draft_model.mask_token_id
        elif hasattr(draft_model.model, "mask_token_id"):
            mask_token_id = draft_model.model.mask_token_id
        else:
            raise ValueError(
                "PARD requires mask_token_id to be set. Please set it in PARDDecodingConfig "
                "or ensure the draft model config has 'pard_token' or 'mask_token_id'."
            )

        if num_contexts > 0:
            # No left-shift: PARD uses its own embeddings, not target hidden states
            input_ids_ctx = input_ids[: attn_metadata.num_ctx_tokens].to(torch.int32).clone()
            position_ids_ctx = position_ids[: attn_metadata.num_ctx_tokens]
        else:
            input_ids_ctx = torch.empty(0, dtype=torch.int32, device="cuda")
            position_ids_ctx = torch.empty(0, dtype=torch.int32, device="cuda")

        if num_gens > 0:
            gen_num_accepted = num_accepted_tokens[num_contexts : num_contexts + num_gens]
            gen_accepted_tokens = accepted_tokens[num_contexts : num_contexts + num_gens, :]

            total_tokens_per_req = self._draft_tokens_per_req  # 2K

            # Start with all mask tokens
            request_ids_2d = torch.full(
                (num_gens, total_tokens_per_req),
                mask_token_id,
                dtype=torch.int32,
                device="cuda",
            )

            # Place all accepted tokens at positions 0..M (including bonus).
            # The remaining slots stay as mask tokens. This ensures:
            # 1) Accepted tokens provide in-sequence context via attention
            # 2) K unique mask predictions from positions M..M+K-1
            max_acc_cols = gen_accepted_tokens.shape[1]  # K+1
            col_range = torch.arange(max_acc_cols, dtype=torch.int32, device="cuda")
            place_mask = col_range.unsqueeze(0) < gen_num_accepted.unsqueeze(1)
            request_ids_2d[:, :max_acc_cols] = torch.where(
                place_mask,
                gen_accepted_tokens[:, :max_acc_cols].to(torch.int32),
                request_ids_2d[:, :max_acc_cols],
            )

            input_ids_gen = request_ids_2d.flatten()

            # Update seq_lens for gen requests to 2K
            attn_metadata._seq_lens_cuda[num_contexts : num_contexts + num_gens] = (
                total_tokens_per_req
            )
            attn_metadata._seq_lens[num_contexts : num_contexts + num_gens] = total_tokens_per_req

            # Position IDs: base = kv_lens - 2K = past_seen (correct for acc_0).
            # With TrtllmAttention (rope_fusion=True) these are ignored in
            # favor of kv_lens - seq_lens, but they're set for other backends.
            if hasattr(attn_metadata, "kv_lens_cuda"):
                gen_pos_starts = (
                    attn_metadata.kv_lens_cuda[num_contexts : num_contexts + num_gens].int()
                    - total_tokens_per_req
                )
            else:
                gen_pos_starts = position_ids[
                    attn_metadata.num_ctx_tokens :: self._draft_tokens_per_req
                ][:num_gens]

            offsets = torch.arange(total_tokens_per_req, dtype=torch.int32, device="cuda")
            position_ids_gen = (gen_pos_starts.unsqueeze(1) + offsets.unsqueeze(0)).flatten()
        else:
            input_ids_gen = torch.empty(0, dtype=torch.int32, device="cuda")
            position_ids_gen = torch.empty(0, dtype=torch.int32, device="cuda")

        input_ids_final = torch.cat([input_ids_ctx, input_ids_gen], dim=0)
        position_ids_final = torch.cat([position_ids_ctx, position_ids_gen], dim=0).int()

        return {
            "input_ids": input_ids_final,
            "position_ids": position_ids_final,
            "attn_metadata": attn_metadata,
            "spec_metadata": spec_metadata,
        }
