# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Standalone DSpark worker, Kimi K3 Rubin local copy.

The 3-way merge of ``DSparkWorker`` in ``tensorrt_llm/_torch/speculative/dspark.py``
(main) with the K3 Rubin agentic optimizations. Only the
standalone worker (DFlash lineage) is here: the embedded DeepSeek-V4 worker
(``DSv4DSparkWorker``), its ``DSparkSpecMetadata`` and helpers stay main's.

The K3 Rubin switches: ``TLLM_SPEC_VOCAB_PARALLEL=1`` (the Markov chain through the
private head, see :meth:`DSparkWorker._draft_vocab_parallel_head`) and
``TLLM_DSPARK_FUSED_MARKOV=1`` (the fused full-vocab chain in place of the
drafter's ``apply_markov_chain_logits``). Unset, the worker is main's.
"""

from typing import Optional

import torch

from tensorrt_llm._torch.distributed import AllReduceStrategy
from tensorrt_llm._torch.modules.linear import Linear, TensorParallelMode
from tensorrt_llm._torch.speculative.vocab_parallel import tp_pair_exchange

from .dflash import DFlashWorker, VocabParallelDraftTokens, dflash_draft_slot_ids
from .dspark_markov_fused import (
    fused_markov_chain_logits,
    fused_markov_enabled,
    fused_markov_supported,
)


class DSparkWorker(DFlashWorker):
    """Worker for a *standalone* DSpark drafter (DFlash lineage).

    DSpark is DFlash plus two extra heads, so the drafting plumbing is
    inherited wholesale from :class:`DFlashWorker` -- paged context K/V,
    slot management, the mask-token block forward -- and only the two
    head-driven policies are overridden here: the block-output slot
    convention (``shift_label``) and the Markov intra-block logit bias.

    Mirrors the model side, where ``GQADSparkForCausalLM`` extends
    ``DFlashForCausalLM`` with the same two heads.

    Naming: this is the unqualified DSpark worker because a separately
    shipped drafter is the ordinary case; :class:`DSv4DSparkWorker` carries
    the qualifier because a draft embedded in the target checkpoint is the
    special one. Workers are classified by *deployment form*, never by draft
    backbone -- so there is no ``Qwen3DSparkWorker``. Note the name meant the
    embedded worker before this split; both the rebind and the rename to
    ``DSv4DSparkWorker`` land in one commit so the swap reads as a unit.

    A worker is agnostic to the draft backbone: everything backbone-shaped is
    supplied by the draft model, which reports its own shapes
    (``_num_attn_layers``, ``_num_heads``, ``_num_kv_heads``, ``_head_dim``)
    and owns the operators (``_build_fused_kv_buffers``,
    ``precompute_context_kv``, ``dflash_forward``,
    ``apply_markov_chain_logits``, ``project_target_hidden``). The worker only
    allocates against the reported shapes and sequences the calls. An MLA
    drafter therefore reuses this class unchanged; its differences (fused-QKV
    assumptions, a 576-latent K/V layout) land in its own draft-model
    subclass. Naming workers by backbone would produce N classes with
    identical bodies.

    Deployment form is the axis the runtime state actually splits on: paged
    draft K/V here, a worker-owned rolling window in
    :class:`DSv4DSparkWorker`.
    """

    def set_draft_model(self, draft_model) -> None:
        """Reject an unsupported vocab mapping here rather than mid-decode.

        ``d2t`` is model-static, so a config mistake should surface at load and
        not as a ``NotImplementedError`` raised per decode step, possibly during
        CUDA-graph capture.
        """
        super().set_draft_model(draft_model)
        if self._d2t is not None and getattr(draft_model, "has_markov_head", False):
            raise NotImplementedError(
                "DSpark Markov head requires a shared draft/target vocab "
                "(d2t vocab mapping is not supported); drafter "
                f"{type(draft_model).__name__} declares one."
            )
        # Built by bind_draft_lm_head when the drafter's head is known.
        self._markov_tp_exchange = None

    def bind_draft_lm_head(self, draft_model) -> None:
        """Also build the Markov chain's TP exchange where it is used.

        A Markov drafter with its own column-parallel LM head that does not
        gather its output drafts from vocab-shard logits; its chain exchanges
        each step's argmax over the TP group (see _apply_dspark_markov_bias).
        NCCL: the path is rare and needs no all-reduce workspace.
        """
        super().bind_draft_lm_head(draft_model)
        head = draft_model.lm_head
        mapping = self.mapping
        if (
            self._markov_tp_exchange is None
            and getattr(draft_model, "has_markov_head", False)
            and isinstance(head, Linear)
            and head.tp_mode == TensorParallelMode.COLUMN
            and not head.gather_output
            and mapping is not None
            and mapping.tp_size > 1
            and not mapping.enable_attention_dp
        ):
            self._markov_tp_exchange = tp_pair_exchange(mapping, AllReduceStrategy.NCCL)

    def _draft_block_width(self, draft_model) -> int:
        """Block width under the dspark ``shift_label`` convention.

        shift_label reads slots 0..K-1, so K draft tokens fit in K slots and
        the base class' K+1 over-demands by one -- enough to reject a block-7
        checkpoint at max_draft_len=7, which is how both published DSpark
        drafters are meant to run.
        """
        if getattr(draft_model, "_dspark_shift_label", False):
            return self.max_draft_len
        return super()._draft_block_width(draft_model)

    def _draft_first_slot(self, draft_model) -> int:
        return 0 if getattr(draft_model, "_dspark_shift_label", False) else 1

    def _draft_slot_ids(
        self, draft_model, num_gens: int, block_size: int, num_draft_tokens: int
    ) -> torch.Tensor:
        """Block-output slots under the dspark ``shift_label`` convention.

        The drafter checkpoint declares the convention, so it is read off the
        draft model rather than assumed: a DSpark drafter trained with the
        legacy DFlash slot layout keeps the base class' slots 1..K.
        """
        shift_label = getattr(draft_model, "_dspark_shift_label", False)
        return dflash_draft_slot_ids(
            num_gens, block_size, num_draft_tokens, shift_label, device="cuda"
        )

    def _vocab_parallel_draft_tokens(self, draft_model) -> Optional[VocabParallelDraftTokens]:
        """A drafter with a Markov head chains the greedy block through it."""
        draft_tokens = super()._vocab_parallel_draft_tokens(draft_model)
        if draft_tokens is None or not getattr(draft_model, "has_markov_head", False):
            return draft_tokens
        return self._markov_block_tokens

    def _markov_block_tokens(
        self, draft_model, local_block: torch.Tensor, inputs: dict
    ) -> torch.Tensor:
        return self._vocab_parallel.head.markov_tokens(
            local_block, inputs["first_prev_tokens"], draft_model.markov_w1, draft_model.markov_w2
        )

    def _draft_vocab_parallel_head(self, draft_model, spec_metadata):
        """Also require Markov weights the vocab-parallel chain can index:
        contiguous ``[vocab, rank]`` tensors of one dtype
        (``TLLM_SPEC_VOCAB_PARALLEL=1``)."""
        head = super()._draft_vocab_parallel_head(draft_model, spec_metadata)
        if head is None or not getattr(draft_model, "has_markov_head", False):
            return head
        w1, w2 = draft_model.markov_w1, draft_model.markov_w2
        if (
            w1.is_contiguous()
            and w2.is_contiguous()
            and w1.shape == w2.shape
            and w1.dtype == w2.dtype
            and w2.shape[0] == head.vocab_size
        ):
            return head
        return None

    def _vocab_parallel_markov_weights(self, draft_model):
        if not getattr(draft_model, "has_markov_head", False):
            return None, None
        return draft_model.markov_w1, draft_model.markov_w2

    def _refine_block_logits(
        self,
        draft_model,
        gen_logits: torch.Tensor,
        inputs: dict,
        spec_metadata,
    ) -> torch.Tensor:
        """Add the greedy-chained Markov intra-block bias to the block logits.

        A DSpark drafter checkpoint may omit the Markov head (``markov_rank``
        0), which loads as a drafter without one; that case falls through to
        the unmodified backbone logits.
        """
        if not getattr(draft_model, "has_markov_head", False):
            return gen_logits
        return self._apply_dspark_markov_bias(
            draft_model, gen_logits, inputs["first_prev_tokens"], spec_metadata
        )

    def _apply_dspark_markov_bias(
        self,
        draft_model,
        gen_logits: torch.Tensor,
        first_prev_tokens: torch.Tensor,
        spec_metadata=None,
    ) -> torch.Tensor:
        """Apply the dspark vanilla-Markov intra-block bias to block logits.

        Reference (DeepSpec VanillaMarkov.sample_block_tokens, temperature 0):
        step i adds bias = markov_w2 @ markov_w1[prev_i] to the shared-lm_head
        logits, where prev_0 is the anchor (last accepted) token and prev_{i>0}
        is the greedy token from step i-1's biased logits. Greedy per-position
        argmax of the returned logits therefore reproduces the reference
        sampled chain; the rejection-sampling path samples from the same
        biased distributions (proposal conditioned on the greedy chain).

        A TP vocab-sharded draft lm_head yields this rank's vocab shard of the
        logits; the chain then takes each step's global argmax over the TP
        group.

        ``TLLM_DSPARK_FUSED_MARKOV=1`` chains full-vocab logits through the K3 Rubin
        fused kernel (``dspark_markov_fused``) instead of the drafter's
        ``apply_markov_chain_logits``; the same chain bit for bit.
        """
        # The d2t guard lives in set_draft_model: it is model-static, so raising
        # it here would surface a load-time config error per decode step.
        # Unlike the d2t guard this one cannot move to set_draft_model: it
        # keys on the runtime logits width, and reproducing that at init would
        # duplicate the draft head's sharding rules. A standalone drafter
        # borrows the target lm_head, whose gather_output defaults to True, so
        # the logits normally arrive full-vocab.
        full_vocab = draft_model.markov_w2.shape[0]
        shard = gen_logits.shape[-1]
        if shard == full_vocab:
            if (
                fused_markov_enabled()
                and (
                    spec_metadata is None
                    or not self._draft_logits_are_sharded(gen_logits, spec_metadata)
                )
                and fused_markov_supported(gen_logits, draft_model.markov_w1, draft_model.markov_w2)
            ):
                return fused_markov_chain_logits(
                    gen_logits, first_prev_tokens, draft_model.markov_w1, draft_model.markov_w2
                )
            return draft_model.apply_markov_chain_logits(gen_logits, first_prev_tokens)
        exchange = self._markov_tp_exchange
        if exchange is None or shard * exchange.tp_size != full_vocab:
            raise NotImplementedError(
                f"DSpark Markov head: draft logits width {shard} does not "
                f"match the drafter vocab {full_vocab} and is not a plain "
                "TP column shard of it."
            )
        return draft_model.apply_markov_chain_logits(
            gen_logits, first_prev_tokens, tp_exchange=exchange
        )
