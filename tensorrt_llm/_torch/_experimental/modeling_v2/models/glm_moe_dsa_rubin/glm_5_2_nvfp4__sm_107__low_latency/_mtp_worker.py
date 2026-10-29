# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GLM-owned MTP worker and optimized draft-loop dispatch."""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, Optional, Tuple

import torch

from tensorrt_llm._torch.attention.backends.sparse.params import MTPIndexShareMetadata
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.speculative import MTPEagleWorker
from tensorrt_llm._torch.speculative.eagle3 import _reset_mtp_index_share
from tensorrt_llm._torch.speculative.interface import (
    _FORCE_ACCEPT_RNG_COUNTER_STRIDE,
    _FORCE_ACCEPT_RNG_POOL_SIZE,
    _FORCE_ACCEPT_RNG_SLOT_STRIDE,
)
from tensorrt_llm.llmapi.llm_args import AdvancedSamplingMode

from ._small_batch_draft import DRAFT_SHAPES, FirstDraftRowSelect
from .ops.spec.verify_head_accept import glm_verify_head_accept

if TYPE_CHECKING:
    from .modeling import DecodeTarget


class MTPDraftWorker(MTPEagleWorker):
    """The one-model MTP-Eagle worker, preserving main's draft metadata lifecycle.

    The TP16 single-request inline path runs first; eligible TP4/TP8 small batches
    then use owned draft kernels. Other steps use the built-in loop.

    The loop keeps the built-in loop's attention-metadata handling: the MTP index-share flags,
    the switch to one row per request after draft step 0, and the cleanup. The draft-step
    kernels advance the sequence and KV lengths in place, and the draft tokens and the
    next-new-tokens rows come from them.
    """

    def __init__(
        self,
        spec_config,
        model_config: ModelConfig,
        use_separate_draft_kv_cache: bool,
        *,
        mapping,
        decode_target,
    ) -> None:
        super().__init__(spec_config, model_config, use_separate_draft_kv_cache, mapping=mapping)
        # The core's per-step lookup of the decode target (a bound method rather than the
        # core itself, which belongs to the shell and is no submodule of this worker).
        self._decode_target = decode_target
        # The draft tokens of the last draft loop on the decode target and the next-new-tokens
        # rows its last step wrote, for _prepare_next_new_tokens.
        self._drafted_rows: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        self._verify_sampler = None

    def prepare_1st_drafter_inputs(
        self,
        input_ids,
        position_ids,
        hidden_states,
        accepted_tokens,
        attn_metadata,
        spec_metadata,
        draft_model,
    ):
        """Without context requests draft step 0's input ids are the accepted-token rows as
        they are, flattened, rather than a copy behind an empty context part."""
        if not self.is_mtp_eagle or attn_metadata.num_contexts > 0:
            return super().prepare_1st_drafter_inputs(
                input_ids,
                position_ids,
                hidden_states,
                accepted_tokens,
                attn_metadata,
                spec_metadata,
                draft_model,
            )
        return {
            "input_ids": accepted_tokens[:, : spec_metadata.runtime_draft_len + 1].flatten(),
            "position_ids": position_ids,
            "hidden_states": hidden_states,
            "attn_metadata": attn_metadata,
            "spec_metadata": spec_metadata,
        }

    def _prepare_next_new_tokens(
        self,
        accepted_tokens,
        next_draft_tokens,
        batch_indices_cuda,
        batch_size,
        num_accepted_tokens,
    ):
        """The rows the last draft-step kernel wrote when these draft tokens are that loop's."""
        drafted, self._drafted_rows = self._drafted_rows, None
        if drafted is not None and drafted[0] is next_draft_tokens:
            return drafted[1]
        return super()._prepare_next_new_tokens(
            accepted_tokens,
            next_draft_tokens,
            batch_indices_cuda,
            batch_size,
            num_accepted_tokens,
        )

    def _sample_tokens_for_batch(self, logits, spec_metadata, num_contexts, batch_size):
        """Sample gathered generation logits with the owned FULL sampler when eligible."""
        sampler = self._verify_sampler
        if (
            sampler is not None
            and num_contexts == 0
            and spec_metadata is not None
            and not spec_metadata.is_ragged_verify
            and not spec_metadata.is_all_greedy_sample
            and spec_metadata.advanced_sampling_mode == AdvancedSamplingMode.FULL
            and isinstance(logits, torch.Tensor)
            and not torch.compiler.is_compiling()
        ):
            rows = logits.shape[0]
            parameters = (
                spec_metadata.temperatures[:rows],
                spec_metadata.top_ks[:rows],
                spec_metadata.top_ps[:rows],
                spec_metadata.min_ps[:rows],
                *self._rng_state_per_token(spec_metadata, rows),
            )
            if sampler.supported(logits, *parameters):
                return sampler.sample(logits, *parameters)
        return super()._sample_tokens_for_batch(logits, spec_metadata, num_contexts, batch_size)

    def _sample_and_accept_draft_tokens_base(
        self,
        logits,
        draft_tokens,
        num_contexts,
        batch_size,
        spec_metadata,
    ):
        """Strict acceptance and the forced accepted-token count in one kernel
        (``glm_verify_head_accept``) for a step of generation requests only, with the worker's
        target-token sampling; ragged verification and context requests take the built-in
        chain. The forced count follows ``_apply_force_accepted_tokens``: none during eager
        CUDA-graph warm-up, the integer part plus one, and with a fractional part the same pool
        draw and counter step."""
        if spec_metadata is None or spec_metadata.is_ragged_verify or num_contexts != 0:
            return super()._sample_and_accept_draft_tokens_base(
                logits, draft_tokens, num_contexts, batch_size, spec_metadata
            )
        if logits.dim() == 1:
            logits = logits.unsqueeze(0)
        tokens_per_request = draft_tokens.shape[-1] + 1
        target_tokens = self._sample_tokens_for_batch(logits, spec_metadata, 0, batch_size)
        mode, base, frac = 0, 0, 0.0
        warmup = spec_metadata.is_cuda_graph and not torch.cuda.is_current_stream_capturing()
        if self.force_num_accepted_tokens != 0.0 and not warmup:
            int_part = int(self.force_num_accepted_tokens)
            frac = self.force_num_accepted_tokens - int_part
            base = min(int_part + 1, tokens_per_request)
            mode = 2 if frac > 0.0 and base < tokens_per_request else 1
        pool = counter = None
        if mode == 2:
            self._ensure_force_accept_rng_state(logits.device)
            pool, counter = self._force_accept_rng_pool, self._force_accept_rng_counter
        accepted_tokens = torch.empty(
            (batch_size, tokens_per_request), dtype=torch.int, device=logits.device
        )
        num_accepted_tokens = glm_verify_head_accept(
            target_tokens,
            draft_tokens,
            0,
            accepted_tokens,
            pool,
            counter,
            mode,
            base,
            tokens_per_request,
            frac,
            _FORCE_ACCEPT_RNG_COUNTER_STRIDE,
            _FORCE_ACCEPT_RNG_SLOT_STRIDE,
            _FORCE_ACCEPT_RNG_POOL_SIZE,
        )
        return accepted_tokens, num_accepted_tokens

    def _forward_linear_draft_loop(
        self,
        inputs,
        attn_metadata,
        spec_metadata,
        draft_model,
        draft_kv_cache_manager,
        num_contexts,
        batch_size,
        num_accepted_tokens,
        original_all_rank_num_tokens,
    ):
        self._drafted_rows = None
        decode = self._drafting_target(
            inputs,
            attn_metadata,
            spec_metadata,
            draft_kv_cache_manager,
            num_contexts,
            batch_size,
            num_accepted_tokens,
        )
        if decode is not None:
            return self._draft_on_target(
                decode,
                inputs,
                attn_metadata,
                spec_metadata,
                draft_kv_cache_manager,
                batch_size,
                num_accepted_tokens,
            )
        helper = self._small_batch_target(
            inputs,
            attn_metadata,
            spec_metadata,
            draft_model,
            num_contexts,
            batch_size,
            num_accepted_tokens,
        )
        if helper is not None:
            drafted = self._forward_small_batch_draft_loop(
                helper,
                inputs,
                attn_metadata,
                spec_metadata,
                draft_model,
                draft_kv_cache_manager,
                batch_size,
                num_accepted_tokens,
            )
            if drafted is not None:
                return drafted
        return super()._forward_linear_draft_loop(
            inputs,
            attn_metadata,
            spec_metadata,
            draft_model,
            draft_kv_cache_manager,
            num_contexts,
            batch_size,
            num_accepted_tokens,
            original_all_rank_num_tokens,
        )

    def _small_batch_target(
        self,
        inputs,
        attn_metadata,
        spec_metadata,
        draft_model,
        num_contexts,
        batch_size,
        num_accepted_tokens,
    ):
        """Select the small-batch bridge before any draft layer or collective."""
        draft_len = spec_metadata.runtime_draft_len
        rows = batch_size * (draft_len + 1)
        ids, positions, hidden = (
            inputs["input_ids"],
            inputs["position_ids"],
            inputs["hidden_states"],
        )
        batch_indices = spec_metadata.batch_indices_cuda
        max_batch = DRAFT_SHAPES.get((self.mapping.tp_size, self.max_draft_len), 0)
        if not (
            self.is_mtp_eagle
            and num_contexts == 0
            and 1 <= batch_size <= max_batch
            and draft_len == self.max_draft_len
            and getattr(self.spec_config, "draft_len_schedule", None) is None
            and self.guided_decoder is None
            and self.sa_enhancer is None
            and self._d2t is None
            and not spec_metadata.wants_advanced_draft_sampling
            and not spec_metadata.is_ragged_verify
            and not torch.compiler.is_compiling()
            and isinstance(attn_metadata, MTPIndexShareMetadata)
            and attn_metadata.num_tokens == rows
            and ids.dtype == torch.int32
            and tuple(ids.shape) == (rows,)
            and ids.is_cuda
            and ids.is_contiguous()
            and positions.dtype in (torch.int32, torch.int64)
            and tuple(positions.shape) == (rows,)
            and positions.is_contiguous()
            and positions.device == ids.device
            and hidden.dtype == torch.bfloat16
            and tuple(hidden.shape) == (rows, 6144)
            and hidden.device == ids.device
            and hidden.is_contiguous()
            and hidden.data_ptr() % 32 == 0
            and num_accepted_tokens.dtype == torch.int32
            and num_accepted_tokens.dim() == 1
            and num_accepted_tokens.numel() >= batch_size
            and num_accepted_tokens.device == ids.device
            and num_accepted_tokens.is_contiguous()
            and batch_indices.dtype == torch.int32
            and batch_indices.dim() == 1
            and batch_indices.numel() >= batch_size
            and batch_indices.device == ids.device
            and batch_indices.is_contiguous()
        ):
            return None
        decode = self._decode_target(attn_metadata)
        helper = getattr(decode, "_small_batch_draft", None)
        if (
            helper is None
            or helper.DRAFT_LEN != draft_len
            or batch_size > helper.MAX_BATCH
            or len(draft_model.mtp_layers) != 1
            or helper.layer is not draft_model.mtp_layers[0]
            or helper.lm_head is not draft_model.lm_head
        ):
            return None
        return helper

    def _forward_small_batch_draft_loop(
        self,
        helper,
        inputs,
        attn_metadata,
        spec_metadata,
        draft_model,
        draft_kv_cache_manager,
        batch_size,
        num_accepted_tokens,
    ):
        """Keep main's metadata lifecycle around owned full-attention draft steps.

        None means only the preflight declined: no draft forward has executed.
        Once the first layer starts, failures propagate without replaying it.
        """
        from tensorrt_llm._torch.attention.backends.sparse.dsa import is_dsa_cache_manager

        draft_len = spec_metadata.runtime_draft_len
        first_rows = batch_size * (draft_len + 1)
        accepted_tokens = inputs["input_ids"].view(batch_size, draft_len + 1)
        batch_indices = spec_metadata.batch_indices_cuda[:batch_size]
        num_accepted = num_accepted_tokens[:batch_size]
        selector = FirstDraftRowSelect(batch_indices, num_accepted, draft_len + 1)
        # Do not modify the caller's inputs if the rebound-manager preflight declines.
        inputs = dict(inputs)
        with contextlib.ExitStack() as scope:
            attn_metadata.set_in_mtp_draft_loop(True)
            attn_metadata.set_mtp_num_accepted(num_accepted_tokens)
            scope.callback(_reset_mtp_index_share, attn_metadata)
            attn_metadata = scope.enter_context(
                self.draft_kv_cache_context(attn_metadata, draft_kv_cache_manager)
            )
            # A V2 draft view can differ from the target manager without a
            # separately configured draft cache. Validate the actual rebound views.
            if not helper.supports(attn_metadata, first_rows, batch_size):
                return None
            if is_dsa_cache_manager(draft_kv_cache_manager):
                attn_metadata.on_update_kv_lens()
            inputs["attn_metadata"] = attn_metadata
            inputs["owned_draft"] = helper
            inputs["row_select"] = selector
            draft_tokens = accepted_tokens.new_empty((batch_size, draft_len))
            next_new_tokens = None
            for step in range(draft_len):
                attn_metadata.set_skip_topk(step > 0)
                hidden, _ = self._run_draft_forward(draft_model, inputs, spec_metadata, step)
                # The layer already selected both hidden and residual BEFORE MoE.
                # A second gather here would select the wrong request (or go OOB).
                if step == 0 and selector.gather_ids is None:
                    raise RuntimeError("small-batch MTP did not select its first-step rows")
                logits, hidden_half = helper.head(hidden, next_step=step + 1 < draft_len)
                if step == 0:
                    attn_metadata._seq_lens[:batch_size].fill_(1)
                    attn_metadata.on_update()
                    attn_metadata.num_contexts = 0
                next_projection, next_positions, next_new_tokens = helper.sample_step(
                    logits,
                    hidden,
                    draft_tokens,
                    step,
                    inputs["position_ids"],
                    selector.gather_ids if step == 0 else None,
                    attn_metadata.seq_lens_cuda[:batch_size],
                    attn_metadata.kv_lens_cuda[:batch_size],
                    num_accepted,
                    accepted_tokens,
                    batch_indices,
                    hidden_half=hidden_half,
                )
                # The fused op just wrote seq/KV lengths. Main owns derived DSA
                # slots, schedules and dense selection; update even the final step.
                attn_metadata.update_for_spec_dec()
                if step == 0:
                    attn_metadata.use_spec_decoding = False
                if step + 1 < draft_len:
                    inputs = {
                        "input_ids": draft_tokens[:, step],
                        "position_ids": next_positions,
                        "hidden_states": hidden,
                        "attn_metadata": attn_metadata,
                        "spec_metadata": spec_metadata,
                        "owned_draft": helper,
                        "eh_proj_output" if helper.use_table else "eh_proj_input": next_projection,
                    }
        self._drafted_rows = (draft_tokens, next_new_tokens)
        return draft_tokens

    def _drafting_target(
        self,
        inputs,
        attn_metadata,
        spec_metadata,
        draft_kv_cache_manager,
        num_contexts,
        batch_size,
        num_accepted_tokens,
    ) -> Optional[DecodeTarget]:
        """The decode target when its kernels draft this step, else None.

        They draft a generation step of one request whose draft layer shares the target's KV
        pool, through the configured draft length with no draft-length schedule (the draft
        steps after the first skip the MTP layer's indexer K write: the next step's draft
        step 0 writes those positions before it reads them), greedily (argmax drafts), with
        nothing that rewrites the drafts (guided decoding, SA, a draft vocabulary map), and
        when the decode target's kernels serve the rows.
        """
        draft_len = spec_metadata.runtime_draft_len
        rows = batch_size * (draft_len + 1)
        input_ids = inputs["input_ids"]
        position_ids = inputs["position_ids"]
        if not (
            self.is_mtp_eagle
            and num_contexts == 0
            and batch_size == 1
            and draft_kv_cache_manager is None
            and draft_len == self.max_draft_len
            and getattr(self.spec_config, "draft_len_schedule", None) is None
            and self.guided_decoder is None
            and self.sa_enhancer is None
            and self._d2t is None
            and not spec_metadata.wants_advanced_draft_sampling
            and not spec_metadata.is_ragged_verify
            and attn_metadata.num_tokens == rows
            and inputs["hidden_states"].shape[0] == rows
            and input_ids.dtype == torch.int32
            and tuple(input_ids.shape) == (rows,)
            and input_ids.is_contiguous()
            and position_ids.dtype in (torch.int32, torch.int64)
            and tuple(position_ids.shape) == (rows,)
            and position_ids.is_contiguous()
            and num_accepted_tokens.dtype == torch.int32
            and num_accepted_tokens.is_contiguous()
            and spec_metadata.batch_indices_cuda.dtype == torch.int32
        ):
            return None
        decode = self._decode_target(attn_metadata)
        if decode is None or not decode.drafts(rows):
            return None
        return decode

    def _draft_on_target(
        self,
        decode: DecodeTarget,
        inputs,
        attn_metadata,
        spec_metadata,
        draft_kv_cache_manager,
        batch_size,
        num_accepted_tokens,
    ) -> torch.Tensor:
        """The draft loop on the decode target's kernels: the draft tokens [batch, draft_len].

        Each step's last kernel writes the next step's sequence and KV lengths in place. The
        steps after the first run no indexer and read nothing else of the DSA decode metadata,
        so the built-in loop's per-step rebuild of it (``update_for_spec_dec``) is left out;
        with a context short enough that the metadata skips the indexer, only the next step's
        dense selection is refreshed from the new lengths. The host side of draft step 0's
        switch to one row per request comes before that.
        """
        draft_len = spec_metadata.runtime_draft_len
        # The accepted tokens' rows: draft step 0's input ids (prepare_1st_drafter_inputs).
        accepted_tokens = inputs["input_ids"].view(batch_size, draft_len + 1)
        batch_indices = spec_metadata.batch_indices_cuda[:batch_size]
        num_accepted = num_accepted_tokens[:batch_size]
        draft_tokens = accepted_tokens.new_empty((batch_size, draft_len))
        positions = inputs["position_ids"]
        eh_proj_output = gather_ids = pool_rows = next_new_tokens = None
        uses_mtp_index_share = isinstance(attn_metadata, MTPIndexShareMetadata)
        with contextlib.ExitStack() as draft_scope:
            if uses_mtp_index_share:
                attn_metadata.set_in_mtp_draft_loop(True)
                attn_metadata.set_mtp_num_accepted(num_accepted_tokens)
                draft_scope.callback(_reset_mtp_index_share, attn_metadata)
            attn_metadata = draft_scope.enter_context(
                self.draft_kv_cache_context(attn_metadata, draft_kv_cache_manager)
            )
            # The draft loop's copy of the sequence lengths (prepare_for_spec_dec) and the KV
            # lengths, which the draft-step kernels advance.
            seq_lens = attn_metadata.seq_lens_cuda[:batch_size]
            kv_lens = attn_metadata.kv_lens_cuda[:batch_size]
            for i in range(draft_len):
                if uses_mtp_index_share:
                    attn_metadata.set_skip_topk(i > 0)
                if i == 0:
                    hidden_states, gather_ids, pool_rows = decode.draft_first_step(
                        attn_metadata,
                        inputs["input_ids"],
                        positions,
                        inputs["hidden_states"],
                        batch_indices,
                        num_accepted,
                        draft_len,
                    )
                else:
                    hidden_states = decode.draft_next_step(
                        attn_metadata, eh_proj_output, pool_rows, join_selection=i == 1
                    )
                last = i == draft_len - 1
                logits, hidden_half = decode.draft_head(hidden_states, next_step=not last)
                rewind_accepted = num_accepted if i == 0 else None
                if i == 0:
                    # One row per request from here on, on the host (no context requests to
                    # turn into generation ones); the kernel below fills seq_lens.
                    attn_metadata._seq_lens[:batch_size].fill_(1)
                    attn_metadata.on_update()
                if last:
                    next_new_tokens = decode.draft_sample_glue(
                        logits,
                        draft_tokens,
                        i,
                        seq_lens,
                        kv_lens,
                        rewind_accepted,
                        accepted_tokens,
                        batch_indices,
                        num_accepted,
                    )
                else:
                    eh_proj_output, positions = decode.draft_ehproj_table(
                        logits,
                        hidden_half,
                        draft_tokens,
                        i,
                        positions,
                        gather_ids if i == 0 else None,
                        seq_lens,
                        kv_lens,
                        rewind_accepted,
                    )
                if not last and attn_metadata.skip_indexer_for_gen_reqs:
                    attn_metadata.prepare_dense_topk_indices(
                        attn_metadata.kv_lens_cuda, device=True
                    )
                if i == 0:
                    attn_metadata.use_spec_decoding = False
        self._drafted_rows = (draft_tokens, next_new_tokens)
        return draft_tokens
