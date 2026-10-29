# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The speculative worker of the glm_moe_dsa sm_103 tp8 target: the one-model MTP-Eagle worker with the
draft-loop fast paths.

``GlmMtpWorker.adopt`` re-classes the built-in ``MTPEagleWorker`` the shell built in place, so
the worker the engine and the shell's epilogue hold stays the same object. The class adds
methods and a class-level default only; nothing is built at adoption, every fast path of the
draft loop is decided per forward.

Each override runs the target's draft loop when the step is a decode step
(``dispatch.Phase.DECODE``) and the built-in method otherwise:

* ``_forward_impl`` accepts the target's tokens and prepares the first draft input and the next
  new tokens through the draft-loop operations (``draft_loop``);
* ``_forward_linear_draft_loop`` runs the draft steps with the draft tail (``draft_tail``), the
  step-0 live rows of the draft layer (``draft_live_rows``) and the draft-loop step updates;
* ``_run_draft_forward`` calls the draft layer with the draft-loop embedding.
"""

import contextlib

import torch
import torch.nn.functional as F

from tensorrt_llm._torch.attention.backends.sparse.params import MTPIndexShareMetadata
from tensorrt_llm._torch.pyexecutor.kv_cache.mamba_cache_manager import MambaHybridCacheManager
from tensorrt_llm._torch.speculative.eagle3 import MTPEagleWorker, _reset_mtp_index_share
from tensorrt_llm._torch.speculative.mtp import _select_mtp_position_ids

from . import dispatch
from .attention import reclass


class GlmMtpWorker(MTPEagleWorker):
    """The one-model MTP-Eagle worker, with the target's draft loop on decode steps."""

    # The draft-loop operations of the running forward; None keeps the built-in acceptance,
    # first draft inputs, step updates and draft-token stacking.
    _draft_loop_ops = None

    @classmethod
    def adopt(cls, worker: MTPEagleWorker) -> "GlmMtpWorker":
        """Re-class the built-in MTP-Eagle worker.

        Raises:
            TypeError: if ``worker`` is not exactly the built-in ``MTPEagleWorker``.
        """
        reclass(worker, MTPEagleWorker, cls)
        return worker

    def _forward_impl(
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
        if dispatch.phase() is not dispatch.Phase.DECODE:
            return super()._forward_impl(
                input_ids=input_ids,
                position_ids=position_ids,
                hidden_states=hidden_states,
                logits=logits,
                attn_metadata=attn_metadata,
                spec_metadata=spec_metadata,
                draft_model=draft_model,
                resource_manager=resource_manager,
            )

        runtime_draft_len = spec_metadata.runtime_draft_len
        # skip the draft forward if the runtime draft length is 0
        if runtime_draft_len == 0:
            return self.skip_drafting(
                input_ids,
                position_ids,
                hidden_states,
                logits,
                attn_metadata,
                spec_metadata,
                draft_model,
            )

        batch_size = attn_metadata.num_seqs
        num_contexts = attn_metadata.num_contexts
        num_gens = batch_size - num_contexts

        raw_logits = logits

        self._execute_guided_decoder_if_present(logits)

        # The draft loop's fused bookkeeping (draft_loop), decided on the host before any
        # device work; None keeps the native code below.
        draft_loop_ops = None
        if self.is_mtp_eagle and dispatch.is_enabled("draft_loop"):
            draft_loop_ops = dispatch.get_impl("draft_loop", "draft_loop_ops")(
                self, attn_metadata, spec_metadata
            )
        self._draft_loop_ops = draft_loop_ops

        # Sample and accept tokens. ``input_ids`` is required by the relaxed-
        # acceptance path (scans for thinking-phase tokens); ignored otherwise.
        if draft_loop_ops is not None:
            accepted_tokens, num_accepted_tokens = draft_loop_ops.accept(logits)
        else:
            accepted_tokens, num_accepted_tokens = self.sample_and_accept_draft_tokens(
                input_ids, logits, attn_metadata, spec_metadata
            )

        # Mamba hybrid models need state updates after token acceptance because
        # the accepted token count affects which Mamba states are valid. The
        # isinstance check below naturally no-ops on non-Mamba kv_cache_managers,
        # so this is safe to run unconditionally regardless of spec mode (Eagle3
        # over a Mamba-style draft is plausible, even if no such draft exists today).
        if self._is_mamba_hybrid_cache is None:
            self._is_mamba_hybrid_cache = isinstance(
                attn_metadata.kv_cache_manager, MambaHybridCacheManager
            )
        if num_gens > 0 and self._is_mamba_hybrid_cache:
            attn_metadata.kv_cache_manager.update_mamba_states(
                attn_metadata=attn_metadata,
                num_accepted_tokens=num_accepted_tokens,
                state_indices=attn_metadata.mamba_metadata.state_indices,
            )

        sa_manager = getattr(spec_metadata.spec_resource_manager, "sa_manager", None)
        if self.sa_enhancer is not None and sa_manager is not None:
            self.sa_enhancer.extend_and_prepare(
                sa_manager=sa_manager,
                request_ids=spec_metadata.request_ids,
                accepted_tokens=accepted_tokens,
                num_accepted_tokens=num_accepted_tokens,
                num_gens=num_gens,
                num_contexts=num_contexts,
                max_draft_len=runtime_draft_len,
            )

        if self._uses_external_shared_target_kv:
            next_draft_tokens = self._forward_external_shared_target_kv_draft_loop(
                position_ids=position_ids,
                hidden_states=hidden_states,
                attn_metadata=attn_metadata,
                spec_metadata=spec_metadata,
                draft_model=draft_model,
                accepted_tokens=accepted_tokens,
                num_accepted_tokens=num_accepted_tokens,
                num_contexts=num_contexts,
                batch_size=batch_size,
            )
        else:
            # Save the old attn_metadata and spec_metadata
            self._prepare_attn_metadata_for_spec_dec(attn_metadata)

            # Prepare inputs for the 1st draft model forward
            position_ids = position_ids.squeeze(0)
            if draft_loop_ops is not None:
                inputs = draft_loop_ops.first_drafter_inputs(position_ids, hidden_states)
            else:
                inputs = self.prepare_1st_drafter_inputs(
                    input_ids=input_ids,
                    position_ids=position_ids,
                    hidden_states=hidden_states,
                    accepted_tokens=accepted_tokens,
                    attn_metadata=attn_metadata,
                    spec_metadata=spec_metadata,
                    draft_model=draft_model,
                )

            # Predict draft tokens. ``original_all_rank_num_tokens`` is saved here
            # so the post-loop restore (below) can put attn_metadata back into a
            # state the target model expects.
            original_all_rank_num_tokens = attn_metadata.all_rank_num_tokens

            # Get the draft KV cache manager if using separate layouts
            draft_kv_cache_manager = self.get_draft_kv_cache_manager(resource_manager)

            next_draft_tokens = self._forward_draft_loop(
                inputs,
                attn_metadata,
                spec_metadata,
                draft_model,
                draft_kv_cache_manager,
                num_contexts,
                num_gens,
                batch_size,
                num_accepted_tokens,
                original_all_rank_num_tokens,
                resource_manager,
            )
            # restore attn_metadata to support cuda graph
            self._restore_attn_metadata_from_spec_dec(attn_metadata)
            # restore all_rank_num_tokens for attention DP
            if original_all_rank_num_tokens is not None:
                attn_metadata.all_rank_num_tokens = original_all_rank_num_tokens

        # prepare next new tokens to support overlap scheduler
        if draft_loop_ops is not None and draft_loop_ops.next_new_tokens is not None:
            next_new_tokens = draft_loop_ops.next_new_tokens
        else:
            next_new_tokens = self._prepare_next_new_tokens(
                accepted_tokens,
                next_draft_tokens,
                spec_metadata.batch_indices_cuda,
                batch_size,
                num_accepted_tokens,
            )
        self._draft_loop_ops = None

        attn_metadata.use_spec_decoding = True

        # Commit model-owned target state only after draft preparation has
        # succeeded; SpecWorkerBase aborts the still-pending snapshots on any
        # exception above this point.
        if self._auxiliary_state_handlers:
            self.commit_auxiliary_speculative_states(
                num_accepted_tokens,
                attn_metadata.mamba_metadata.state_indices[:batch_size],
                num_contexts,
            )

        return {
            "logits": raw_logits,
            "new_tokens": accepted_tokens,
            "new_tokens_lens": num_accepted_tokens,
            "next_draft_tokens": next_draft_tokens,
            "next_new_tokens": next_new_tokens,
        }

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
        """Linear draft loop, unified for Eagle3 and MTP Eagle."""
        if dispatch.phase() is not dispatch.Phase.DECODE:
            return super()._forward_linear_draft_loop(
                inputs=inputs,
                attn_metadata=attn_metadata,
                spec_metadata=spec_metadata,
                draft_model=draft_model,
                draft_kv_cache_manager=draft_kv_cache_manager,
                num_contexts=num_contexts,
                batch_size=batch_size,
                num_accepted_tokens=num_accepted_tokens,
                original_all_rank_num_tokens=original_all_rank_num_tokens,
            )
        from tensorrt_llm._torch.attention.backends.sparse.dsa import is_dsa_cache_manager

        runtime_draft_len = spec_metadata.runtime_draft_len
        num_gens = batch_size - num_contexts
        next_draft_tokens = []
        # The draft-loop operations run only without context rows, so the last-token scan is
        # unused with them.
        draft_loop_ops = getattr(self, "_draft_loop_ops", None)
        if draft_loop_ops is None:
            last_tokens_idx = torch.cumsum(attn_metadata.seq_lens_cuda, dim=0, dtype=torch.long) - 1
        position_ids = inputs["position_ids"]

        # Sparse backends that can reuse one indexer selection across the draft
        # loop expose this setter; each indexer still decides whether its own
        # config turns the reuse on.
        uses_mtp_index_share = self.is_mtp_eagle and isinstance(
            attn_metadata, MTPIndexShareMetadata
        )
        # The draft tail (draft_tail): one launch for the draft LM head, the vocab-shard max
        # and the cross-rank argmax. Decided once per forward, before any device work,
        # identically on every TP rank.
        draft_tail = None
        if self.is_mtp_eagle and dispatch.is_enabled("draft_tail"):
            draft_tail = dispatch.get_impl("draft_tail", "select_draft_tail")(
                self, draft_model, spec_metadata
            )
        # The step-0 live rows (draft_live_rows): draft step 0 runs the MTP layer's MoE and
        # post-MoE all-reduce on the live rows the draft-loop operations selected only, and
        # the layer returns those rows. Decided once per forward, before any device work,
        # identically on every TP rank.
        live_rows_step0 = None
        if (
            self.is_mtp_eagle
            and draft_loop_ops is not None
            and dispatch.is_enabled("draft_live_rows")
        ):
            live_rows_step0 = dispatch.get_impl("draft_live_rows", "select_step0")(
                self, draft_model, spec_metadata, draft_loop_ops
            )
        with contextlib.ExitStack() as draft_scope:
            if uses_mtp_index_share:
                attn_metadata.set_in_mtp_draft_loop(True)
                # Accepted counts let the indexer stash each gen's last-accepted row.
                attn_metadata.set_mtp_num_accepted(num_accepted_tokens)
                # The metadata outlives this call, so clear the state even when a
                # draft step raises: a stale draft-loop flag would make the next
                # target forward reuse these selections. The loop writes
                # set_skip_topk on the metadata yielded below, which is this same
                # object: draft_kv_cache_context swaps buffers in place for every
                # TrtllmAttentionMetadata, and sparse metadata all derive from it.
                draft_scope.callback(_reset_mtp_index_share, attn_metadata)
            draft_attn_metadata = draft_scope.enter_context(
                self.draft_kv_cache_context(attn_metadata, draft_kv_cache_manager)
            )
            attn_metadata = draft_attn_metadata
            inputs["attn_metadata"] = draft_attn_metadata
            if self.is_mtp_eagle and is_dsa_cache_manager(draft_kv_cache_manager):
                # Overlap scheduling corrects kv_lens_cuda from the runtime
                # accepted-token counts inside the captured graph. The target
                # forward refreshes target slot mappings during that correction;
                # refresh once more after rebinding so the first draft forward
                # writes the separate DSA indexer cache at the same positions.
                attn_metadata.on_update_kv_lens()
            for i in range(runtime_draft_len):
                if uses_mtp_index_share:
                    attn_metadata.set_skip_topk(i > 0)
                # Run draft model (mode-specific via helper). The helper
                # passes ``all_rank_num_tokens`` as a kwarg so the draft model
                # handles save/restore internally (Eagle3DraftModel.forward
                # uses try/finally); attn_metadata is left untouched here.
                live_rows_compacted = False
                if i == 0 and live_rows_step0 is not None:
                    with live_rows_step0.armed() as live_rows:
                        hidden_states, hidden_states_to_save = self._run_draft_forward(
                            draft_model, inputs, spec_metadata, i
                        )
                    live_rows_compacted = live_rows.compacted
                else:
                    hidden_states, hidden_states_to_save = self._run_draft_forward(
                        draft_model, inputs, spec_metadata, i
                    )

                # Compute gather_ids: on the first draft step each generation
                # request may have accepted multiple tokens, so we index into
                # the flattened token sequence to find the last accepted one.
                # From step 1 onwards the loop feeds exactly one token per
                # request, in request order, so every per-request row
                # selection is the prefix slice [0, batch_size); each
                # ``[gather_ids]`` site below takes a slice as readily as an
                # index tensor.
                draft_rows = None
                if live_rows_compacted:
                    # The MTP layer already returned the live rows.
                    gather_ids, draft_rows = draft_loop_ops.gather_ids, hidden_states
                elif i == 0 and draft_loop_ops is not None:
                    gather_ids, draft_rows = draft_loop_ops.step0_rows(hidden_states)
                elif i == 0:
                    start_ids_gen = (
                        spec_metadata.batch_indices_cuda[:num_gens] * (runtime_draft_len + 1)
                    ).long()
                    gather_ids_gen = (
                        start_ids_gen
                        + num_accepted_tokens[num_contexts:]
                        - 1
                        + attn_metadata.num_ctx_tokens
                    )
                    gather_ids = torch.concat(
                        [last_tokens_idx[:num_contexts], gather_ids_gen], dim=0
                    )
                else:
                    gather_ids = slice(0, batch_size)

                if self.guided_decoder is not None:
                    new_tokens = inputs["input_ids"][gather_ids]
                    self.guided_decoder.add_draft_batch(
                        new_tokens, num_accepted_tokens, draft_step=i
                    )

                # Compute logits.
                # MTP Eagle: shared_head of the MTP layer, with optional
                #   ADP+LM-head-TP padding to ``max_num_requests`` so every TP
                #   rank produces logits of the same shape.
                # Eagle3: logits_processor of the EAGLE draft model.
                #
                # The LM-head-TP fast path (group-stacked rows, vocab-sharded
                # weight) is only usable when the consumer is an argmax: the
                # distributed greedy sampler recovers the global argmax from
                # vocab-shard maxima without materializing full distributions.
                # Advanced (rejection) sampling needs each request's full-vocab
                # distribution, so a batch headed for the advanced path
                # bypasses LM-head-TP and computes full-vocab logits locally --
                # under ADP the lm_head weight is replicated (the sliced-shard
                # trick is a runtime optimization), exactly like the target
                # head. With rejection off, non-greedy batches keep the
                # LM-head-TP argmax path unconditionally, where the per-rank
                # greedy flag never enters control flow. This branch is safe
                # to take group-uniformly because is_all_greedy_sample is
                # group-synchronized whenever rejection+ADP+LM-head-TP are
                # combined -- see SpecMetadata.group_all_greedy_sample (anchor
                # for the group-sync semantics).
                advanced_draft_sampling = spec_metadata.wants_advanced_draft_sampling
                # enable_lm_head_tp_in_adp implies enable_attention_dp
                # (asserted in Mapping.__init__); no separate ADP check.
                lm_head_tp_in_adp_configured = (
                    self.is_mtp_eagle
                    and self.model_config is not None
                    and self.model_config.mapping.enable_lm_head_tp_in_adp
                )
                use_lm_head_tp_in_adp = lm_head_tp_in_adp_configured and not advanced_draft_sampling
                draft_tokens = None
                if self.is_mtp_eagle:
                    # One row per request: the LM-head input on this step and
                    # the draft input on the next.
                    request_hidden_states = (
                        draft_rows if draft_rows is not None else hidden_states[gather_ids]
                    )
                    if draft_tail is not None:
                        draft_tokens = draft_tail(request_hidden_states)
                    if draft_tokens is not None:
                        logits = None
                    elif use_lm_head_tp_in_adp:
                        token_count = request_hidden_states.view(
                            -1, request_hidden_states.shape[-1]
                        ).shape[0]
                        max_num_requests = spec_metadata.max_num_requests
                        pad_len = max_num_requests - token_count
                        if pad_len > 0:
                            padded_hidden_states = F.pad(
                                request_hidden_states.view(-1, request_hidden_states.shape[-1]),
                                (0, 0, 0, pad_len),
                                mode="constant",
                                value=0,
                            )
                        elif pad_len == 0:
                            padded_hidden_states = request_hidden_states.view(
                                -1, request_hidden_states.shape[-1]
                            )
                        else:
                            raise ValueError(
                                "Eagle3OneModelWorker (MTP Eagle mode): "
                                "token_count > max_num_requests, which is not supported"
                            )
                        logits = draft_model.mtp_layers[0].shared_head(
                            padded_hidden_states, draft_model.lm_head, attn_metadata, True
                        )
                    elif lm_head_tp_in_adp_configured:
                        # Advanced-sampling bypass: the model's shared_head
                        # would re-apply the LM-head-TP stacked/sharded path
                        # from config on its own. Under ADP the LMHead weight
                        # is replicated, so project this rank's own rows to the
                        # full vocabulary. Model-specific shared heads may need
                        # preprocessing before that local projection.
                        shared_head = draft_model.mtp_layers[0].shared_head
                        local_full_vocab_forward = getattr(
                            shared_head, "forward_local_full_vocab", None
                        )
                        if local_full_vocab_forward is None:
                            logits = draft_model.lm_head(request_hidden_states)
                        else:
                            logits = local_full_vocab_forward(
                                request_hidden_states,
                                draft_model.lm_head,
                                attn_metadata,
                                True,
                            )
                    else:
                        logits = draft_model.mtp_layers[0].shared_head(
                            request_hidden_states, draft_model.lm_head, attn_metadata, True
                        )
                else:
                    logits = draft_model.logits_processor(
                        hidden_states[gather_ids], draft_model.lm_head, attn_metadata, True
                    )

                if self.guided_decoder is not None:
                    if self.is_mtp_eagle:
                        self.guided_decoder.execute_draft_batch(logits, draft_step=i)
                    else:
                        self.guided_decoder.execute_draft_batch(logits, self._d2t, draft_step=i)

                # ADP+LM-head-TP logits are the LM-head-TP group's row-stacked
                # batch (each rank's rows padded to max_num_requests, then
                # all-gathered along dim 0) with the vocab sharded across the
                # group. Rows [:token_count] would be group rank 0's requests,
                # not this rank's, and a per-rank argmax would return a
                # shard-local index -- so keep the full stacked logits and let
                # greedy_sample_draft_with_tp_gather combine the group's vocab
                # shards and slice this rank's own row segment; only then trim
                # the max_num_requests padding down to token_count.
                mapping_lm_head_tp = None
                if use_lm_head_tp_in_adp:
                    # The MTP head built this per-forward mapping when producing
                    # the vocab-sharded logits; the sampler needs it to gather.
                    mapping_lm_head_tp = getattr(
                        draft_model.mtp_layers[0].shared_head, "mapping_lm_head_tp", None
                    )
                if draft_tokens is not None:
                    new_draft_token = draft_tokens
                else:
                    new_draft_token = self.sample_draft_tokens(
                        logits,
                        spec_metadata,
                        batch_size,
                        draft_step=i,
                        mapping_lm_head_tp=mapping_lm_head_tp,
                    )
                if use_lm_head_tp_in_adp:
                    new_draft_token = new_draft_token[:token_count]
                next_draft_tokens.append(new_draft_token)

                # Build the next iteration's inputs; the last iteration has
                # none to build.
                # MTP Eagle: the MTP layer returns a single tensor; the rows
                #   selected above for the LM head carry over as they are.
                # Eagle3: the EAGLE draft model returns a secondary
                #   ``hidden_states_to_save`` specifically for this purpose.
                # The draft-loop step update: one launch for the position advance
                # and the metadata update below; None keeps the native update.
                draft_step = None
                if draft_loop_ops is not None:
                    draft_step = draft_loop_ops.step_update(
                        attn_metadata,
                        i,
                        num_accepted_tokens,
                        inputs["position_ids"] if i + 1 < runtime_draft_len else None,
                        gather_ids if i == 0 else None,
                    )
                if i + 1 < runtime_draft_len:
                    if self.is_mtp_eagle:
                        hidden_states = request_hidden_states
                    else:
                        hidden_states = hidden_states_to_save[gather_ids]
                    if draft_step is not None:
                        position_ids = draft_step.position_ids
                    else:
                        position_ids = (
                            _select_mtp_position_ids(inputs["position_ids"], gather_ids) + 1
                        )
                    inputs = {
                        "input_ids": new_draft_token,
                        "position_ids": position_ids,
                        "hidden_states": hidden_states,
                        "attn_metadata": attn_metadata,
                        "spec_metadata": spec_metadata,
                    }

                # Update attn_metadata for the next iteration.
                if draft_step is not None:
                    pass  # The draft-loop step update updated the metadata above.
                elif i == 0:
                    attn_metadata._seq_lens[:batch_size].fill_(1)
                    attn_metadata._seq_lens_cuda[:batch_size].fill_(1)
                    attn_metadata.on_update()
                    has_kv_cache = inputs["attn_metadata"].kv_cache_manager is not None
                    if has_kv_cache and hasattr(attn_metadata, "host_request_types"):
                        attn_metadata.host_request_types[: attn_metadata.num_contexts].fill_(1)
                    if has_kv_cache:
                        attn_metadata.num_contexts = 0
                    if hasattr(attn_metadata, "kv_lens_cuda"):
                        attn_metadata.kv_lens_cuda[num_contexts:batch_size] -= (
                            runtime_draft_len - num_accepted_tokens[num_contexts:]
                        )
                        attn_metadata.kv_lens_cuda[:num_contexts] += 1

                    if has_kv_cache:
                        self._prepare_flash_mla_generation_layout(
                            attn_metadata, num_contexts, batch_size
                        )
                    if hasattr(attn_metadata, "kv_lens_cuda"):
                        attn_metadata.update_for_spec_dec()

                    # Both Eagle3 and MTP Eagle drafters take ``draft_len + 1``
                    # tokens in the first draft step (attention runs in spec-dec
                    # mode), then 1 token per step in subsequent iterations.
                    # Disable spec_decoding here so the masks/positions stay
                    # correct on subsequent iters.
                    attn_metadata.use_spec_decoding = False
                else:
                    if hasattr(attn_metadata, "kv_lens_cuda"):
                        attn_metadata.kv_lens_cuda[:batch_size] += 1
                        attn_metadata.update_for_spec_dec()
        stacked = draft_loop_ops.finalize(next_draft_tokens) if draft_loop_ops is not None else None
        next_draft_tokens = (
            stacked if stacked is not None else torch.stack(next_draft_tokens, dim=1)
        )

        # Override with SA draft tokens after all draft layers have run,
        # so that draft layers never see SA tokens in their inputs.
        if self.sa_enhancer is not None:
            gen_draft_tokens = next_draft_tokens[num_contexts:]
            gen_draft_tokens = self.sa_enhancer.maybe_override_all_draft_tokens(gen_draft_tokens)
            next_draft_tokens[num_contexts:] = gen_draft_tokens

        return next_draft_tokens

    def _run_draft_forward(self, draft_model, inputs, spec_metadata, step_idx: int):
        """Invoke the draft model for one iteration, branching on mode.

        ``all_rank_num_tokens`` is passed as a kwarg in both modes. For MTP
        Eagle it goes directly to ``mtp_layers[0]``; for Eagle3 it goes to
        ``Eagle3DraftModel.forward`` which guards it with a try/finally so
        attn_metadata sees the original value on return.
        """
        if dispatch.phase() is not dispatch.Phase.DECODE:
            return super()._run_draft_forward(
                draft_model=draft_model,
                inputs=inputs,
                spec_metadata=spec_metadata,
                step_idx=step_idx,
            )
        all_rank_num_tokens = self._get_step_all_rank_num_tokens(spec_metadata, step_idx)

        if self.is_mtp_eagle:
            embed_tokens = draft_model.embed_tokens
            if dispatch.is_enabled("draft_loop"):
                embed_tokens = dispatch.get_impl("draft_loop", "draft_embedding")(embed_tokens)
            hidden_states = draft_model.mtp_layers[0](
                embed_tokens=embed_tokens, all_rank_num_tokens=all_rank_num_tokens, **inputs
            )
            return hidden_states, None

        inputs["all_rank_num_tokens"] = all_rank_num_tokens
        hidden_states, hidden_states_to_save = draft_model.model(**inputs)
        return hidden_states, hidden_states_to_save
