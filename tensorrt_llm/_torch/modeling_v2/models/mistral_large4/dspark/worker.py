# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The DSpark worker the Mistral Large 4 targets build for the checkpoint's own drafter.

The drafter (``params.json`` ``mtp.dspark``) differs from a standalone DSpark drafter in three
ways, which this worker routes to it:

* It embeds its input block itself, with learned position-specific mask embeddings
  (``embed_draft_block``). The fused paged-latent preparation, which embeds the block from the
  token embedding, does not serve it.
* Its block forward picks the prefill or the decode implementation for the step, so it reads
  the step's attention metadata (``dflash_forward_with_metadata``).
* Its Markov head is recurrent over the block's draft hidden states
  (``apply_markov_chain_logits``). The engine's DSpark chains compute the vanilla head only, so
  the drafts are the DFlash greedy-chained block with this bias: never the vocab-parallel
  shard chain, the fused sampled chain or the ancestral sampler.
"""

from typing import Optional

import torch

from tensorrt_llm._torch.speculative.dspark import DSparkWorker
from tensorrt_llm._torch.speculative.interface import SpecMetadata


class MistralLarge4DSparkWorker(DSparkWorker):
    """``DSparkWorker`` for the Mistral Large 4 drafter; see the module docstring."""

    def _uses_fused_paged_latent_prepare(self, draft_model, kv_factor: int) -> bool:
        return False

    def _draft_block_embedding(
        self,
        draft_model,
        bonus: torch.Tensor,
        mask_token_id: int,
        block_size: int,
        trained_mask_embedding: Optional[torch.Tensor],
    ) -> torch.Tensor:
        return draft_model.embed_draft_block(bonus, block_size)

    def _draft_block_forward(self, draft_model, attn_metadata, **block_inputs) -> torch.Tensor:
        return draft_model.dflash_forward_with_metadata(attn_metadata=attn_metadata, **block_inputs)

    def _vocab_parallel_draft_tokens(self, draft_model):
        return None

    def _sample_block_draft_tokens(
        self,
        draft_model: torch.nn.Module,
        gen_logits: torch.Tensor,
        gen_hidden_states: torch.Tensor,
        inputs: dict,
        spec_metadata: SpecMetadata,
        num_contexts: int,
        batch_size: int,
    ) -> tuple[torch.Tensor, int]:
        num_gens, block_len = gen_logits.shape[:2]
        draft_hidden = (
            None
            if gen_hidden_states is None
            else gen_hidden_states.reshape(num_gens, block_len, -1)
        )
        gen_logits = draft_model.apply_markov_chain_logits(
            gen_logits, inputs["first_prev_tokens"], hidden_states=draft_hidden
        )
        tokens = self.sample_draft_tokens(
            gen_logits, spec_metadata, batch_size, num_contexts=num_contexts
        )
        return tokens, gen_logits.shape[-1]
