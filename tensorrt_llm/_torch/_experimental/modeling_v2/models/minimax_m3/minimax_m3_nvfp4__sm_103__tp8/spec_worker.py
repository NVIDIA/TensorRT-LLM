# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Eagle3 one-model speculative decoding of the MiniMax-M3 tp8 target, at stock numerics.

``M3Eagle3Worker`` is the in-tree ``Eagle3OneModelWorker`` with a fused verify and a fused draft loop:

* Verify: on decode-only batches whose verify rows the fused kernels cover, the target's BF16 lm_head computes this
  rank's vocabulary shard of the logits (the in-tree LM head's own local GEMM) and one kernel replaces the logits
  all-gather, the argmax (or the temperature + top-p draw) and the strict acceptance.
* Draft: the in-tree Eagle3 drafter's modules and weights at their precision (BF16 weights and activations, the
  full-vocabulary BF16 head, greedy drafts). The first pass may take the per-layer fc_norm RMSNorms in one launch (or
  folded into the hidden-state capture) and run its fc beside the target head GEMM. Its BF16 linears run
  FlashInfer's fixed-tactic low-M kernels at the rows ``dispatch.DRAFT_LOWM`` covers (``prebuild_drafter_linears``).
  On a decode-only step of at most 8 requests (``dispatch.DRAFT_STEP``) the draft loop is fused
  (``_m3_fused_draft_loop``): each pass's draft input in one launch, the drafter layer's in-tree attention and MLP
  with their all-reduces fused with the residual add and the next RMSNorm, and one kernel for the BF16 head, the
  cross-rank argmax and the pass's bookkeeping.

Every other batch takes the in-tree paths.

The target's outer CausalLM drives the worker through ``M3SpecDecForwardMixin.forward`` (or the same calls made from
its own forward), and its layer loop captures the Eagle3 hidden states through ``capture_hidden_states``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

import torch
import triton
import triton.language as tl
from torch import nn

from tensorrt_llm._torch.distributed import AllReduce, AllReduceParams
from tensorrt_llm._torch.models.modeling_speculative import _slice_spec_position_ids
from tensorrt_llm._torch.modules.gated_mlp import GatedMLP
from tensorrt_llm._torch.modules.linear import TensorParallelMode
from tensorrt_llm._torch.pyexecutor.kv_cache.mamba_cache_manager import MambaHybridCacheManager
from tensorrt_llm._torch.speculative.eagle3 import Eagle3OneModelWorker
from tensorrt_llm._torch.speculative.interface import (
    _FORCE_ACCEPT_RNG_COUNTER_STRIDE,
    _FORCE_ACCEPT_RNG_POOL_SIZE,
    _FORCE_ACCEPT_RNG_SLOT_STRIDE,
)
from tensorrt_llm.mapping import Mapping

from . import dispatch
from .kernels import m3_draft_lowm, m3_draft_step, m3_sampled_accept, m3_shard_accept, m3_spec_tail

if TYPE_CHECKING:
    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm.llmapi.llm_args import EagleDecodingConfig

# Id of the target's final hidden state in ``layers_to_capture`` (SpecDecOneEngineForCausalLM.layer_idx).
FINAL_HIDDEN_LAYER_ID = -1


@triton.jit
def _m3_force_vec_kernel(
    pool,
    counter,
    out,
    batch_size,
    base_total,
    max_total,
    frac,
    counter_stride,
    slot_stride,
    pool_mask,
    BLOCK: tl.constexpr,
):
    c = tl.load(counter) + 1
    slots = tl.arange(0, BLOCK)
    tl.store(counter, c)
    idx = (c * counter_stride + slots.to(tl.int64) * slot_stride) & pool_mask
    live = slots < batch_size
    r = tl.load(pool + idx, mask=live, other=1.0)
    v = tl.minimum(base_total + (r < frac).to(tl.int32), max_total)
    tl.store(out + slots, v, mask=live)


def draft_ar_norm(
    all_reduce: AllReduce, x: torch.Tensor, residual: torch.Tensor, norm: nn.Module
) -> tuple[torch.Tensor, torch.Tensor]:
    """One draft-layer boundary in stock Eagle3's structure: the projection's plain all-reduce (BF16 out), then the
    norm module with the residual (FlashInfer's fused_add_rmsnorm: FP32 add, norm of the FP32 sum). Returns the
    normalized rows and the residual sum."""
    return norm(all_reduce(x), residual)


def capture_hidden_states(
    spec_metadata: Any,
    layer_id: int,
    hidden_states: torch.Tensor,
    residual: Optional[torch.Tensor],
    capture_norm: Optional[tuple[torch.Tensor, float]],
) -> None:
    """The Eagle3 hidden-state capture of one target layer.

    Mirrors ``Eagle3OneModelSpecMetadata.maybe_capture_hidden_states``. With ``capture_norm`` = (the drafter's
    stacked fc_norm weights [layers, H], eps) (``M3Eagle3Worker.capture_norm``), the capture writes
    rmsnorm(bf16(h + residual)) * fc_norm[i] in one kernel (the in-tree capture's bf16 add, then the in-tree fc_norm
    arithmetic), which the fused first drafter inputs read as is; otherwise it is the in-tree capture.
    """
    if capture_norm is None:
        spec_metadata.maybe_capture_hidden_states(layer_id, hidden_states, residual)
        return
    for i, captured_layer_id in enumerate(spec_metadata.layers_to_capture):
        if captured_layer_id == layer_id:
            assert spec_metadata.hidden_states is not None
            num_tokens = spec_metadata.num_capture_tokens
            if num_tokens > spec_metadata.hidden_states.shape[0]:
                raise RuntimeError(
                    "EAGLE3 hidden-state capture token count exceeds "
                    f"buffer capacity: num_tokens={num_tokens}, "
                    f"capacity={spec_metadata.hidden_states.shape[0]}"
                )
            if num_tokens > hidden_states.shape[0]:
                raise RuntimeError(
                    "EAGLE3 hidden-state capture token count exceeds "
                    f"available hidden states: num_tokens={num_tokens}, "
                    f"hidden_states={hidden_states.shape[0]}"
                )
            hidden_size = spec_metadata.hidden_size
            m3_spec_tail.capture_norm(
                hidden_states[:num_tokens],
                residual[:num_tokens] if residual is not None else None,
                capture_norm[0][i],
                capture_norm[1],
                spec_metadata.hidden_states[:num_tokens, i * hidden_size : (i + 1) * hidden_size],
            )
            break


class M3Eagle3Worker(Eagle3OneModelWorker):
    """``Eagle3OneModelWorker`` with the MiniMax-M3 fused verify (see the module docstring).

    Methods whose docstring says "Mirrors" are copies of the named in-tree method with the fused-verify branches
    inserted; compare them with that method when it changes.
    """

    def __init__(
        self,
        spec_config: "EagleDecodingConfig",
        mapping: Optional[Mapping] = None,
        model_config: Optional["ModelConfig"] = None,
        use_separate_draft_kv_cache: bool = False,
    ):
        super().__init__(
            spec_config,
            mapping=mapping,
            model_config=model_config,
            use_separate_draft_kv_cache=use_separate_draft_kv_cache,
        )
        self._init_fused_verify()

    @classmethod
    def adopt(cls, worker: Eagle3OneModelWorker) -> "M3Eagle3Worker":
        """Turn the in-tree worker the one-engine shell built (``get_spec_worker``) into this worker, in place."""
        if type(worker) is not Eagle3OneModelWorker:
            raise TypeError(f"expected an Eagle3OneModelWorker, got {type(worker).__name__}")
        worker.__class__ = cls
        worker._init_fused_verify()
        return worker

    def _init_fused_verify(self) -> None:
        # Per-step state, reset at the top of every _forward_impl: the first draft pass's gather rows when this step
        # takes the fused first drafter inputs, and the fc output a fork computed beside the target head.
        self._m3_first_gather_ids: Optional[torch.Tensor] = None
        self._m3_fc_hidden: Optional[torch.Tensor] = None
        self._m3_target_lm_head: Optional[nn.Module] = None
        # (fc_norm[0].weight, stacked fc_norm weights [layers, H]), built eagerly on the first fused first pass.
        self._m3_fc_norm_w: Optional[tuple[torch.Tensor, torch.Tensor]] = None
        # Armed per step by arm_capture_norm: (stacked fc_norm weights, eps) or None.
        self._m3_capture_norm: Optional[tuple[torch.Tensor, float]] = None
        self._m3_aux_stream: Optional[torch.cuda.Stream] = None
        # Armed per step by arm_capture_norm: the drafter whose first fc is forked at the last captured target layer
        # (capture_hidden_states), and the fork that did it, joined by _m3_target_accept.
        self._m3_fc_early_drafter: Optional[nn.Module] = None
        self._m3_fc_early_stream: Optional[torch.cuda.Stream] = None
        self._m3_force_vec_warm = False
        # The fragment-ordered copy of the drafter's BF16 head shard the fused draft loop streams, built before the
        # executor's warmup (prebuild_draft_head).
        self._m3_draft_head: Optional[m3_draft_step.DraftHead] = None
        # Per step: the next step's input tokens [nreq, D + 1], column 0 written by the fused verify (or the seed) and
        # columns 1.. by the fused draft passes; set as _m3_next_new_tokens once the fused draft loop filled it.
        self._m3_next_new: Optional[torch.Tensor] = None
        self._m3_next_new_tokens: Optional[torch.Tensor] = None
        # Per step: the first draft inputs are the fused ones (accepted tokens as they lie, fc output of every row).
        self._m3_fused_first_inputs = False

    @property
    def capture_norm(self) -> Optional[tuple[torch.Tensor, float]]:
        """What ``capture_hidden_states`` needs for this step's captures (set by ``arm_capture_norm``)."""
        return self._m3_capture_norm

    def capture_hidden_states(
        self,
        spec_metadata: Any,
        layer_id: int,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> None:
        """``capture_hidden_states`` with this step's capture norm; on a step armed for the early fc fork
        (``arm_capture_norm``), the last captured layer then forks the drafter's first fc onto the aux stream, where
        it overlaps the remaining target layers and the head (joined by ``_m3_target_accept``)."""
        capture_hidden_states(
            spec_metadata, layer_id, hidden_states, residual, self._m3_capture_norm
        )
        drafter = self._m3_fc_early_drafter
        layers = spec_metadata.layers_to_capture
        if drafter is not None and layers and layer_id == layers[-1]:
            self._m3_fc_early_drafter = None
            self._m3_fc_early_stream = self._m3_fc_fork(
                spec_metadata, spec_metadata.num_capture_tokens, drafter
            )

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
        """Mirrors Eagle3OneModelWorker._forward_impl, with the fused verify branch."""
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

        self._m3_first_gather_ids = None
        self._m3_next_new = None
        self._m3_next_new_tokens = None
        self._m3_fused_first_inputs = False
        if logits is None:
            # tail_eligible() skipped the gathered target logits: verify from this rank's logits shard.
            accepted_tokens, num_accepted_tokens, self._m3_first_gather_ids = (
                self._m3_target_accept(
                    hidden_states, spec_metadata, batch_size, runtime_draft_len, draft_model
                )
            )
        else:
            self._execute_guided_decoder_if_present(logits)

            # Sample and accept tokens. ``input_ids`` is required by the relaxed-
            # acceptance path (scans for thinking-phase tokens); ignored otherwise.
            accepted_tokens, num_accepted_tokens = self.sample_and_accept_draft_tokens(
                input_ids, logits, attn_metadata, spec_metadata
            )
            if self.draft_only_eligible(attn_metadata, spec_metadata, draft_model):
                self._m3_seed_first_pass(num_accepted_tokens, spec_metadata, batch_size)

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
        if self._m3_next_new_tokens is not None:
            # Filled in place by the fused verify or the seed (column 0) and the fused draft passes (columns 1..).
            next_new_tokens = self._m3_next_new_tokens
        else:
            next_new_tokens = self._prepare_next_new_tokens(
                accepted_tokens,
                next_draft_tokens,
                spec_metadata.batch_indices_cuda,
                batch_size,
                num_accepted_tokens,
            )

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

    def prepare_1st_drafter_inputs(
        self,
        input_ids: torch.LongTensor,
        position_ids: torch.LongTensor,
        hidden_states: torch.Tensor,
        accepted_tokens: torch.Tensor,
        attn_metadata: Any,
        spec_metadata: Any,
        draft_model: nn.Module,
    ):
        """The fused first draft inputs on a step that verified with the fused kernels (or was seeded for them),
        else the in-tree ones."""
        fused = self._m3_first_drafter_inputs(
            position_ids,
            accepted_tokens,
            input_ids.shape[0],
            attn_metadata.num_contexts,
            spec_metadata,
            draft_model,
        )
        self._m3_fused_first_inputs = fused is not None
        if fused is not None:
            fused.update(attn_metadata=attn_metadata, spec_metadata=spec_metadata)
            return fused
        return super().prepare_1st_drafter_inputs(
            input_ids=input_ids,
            position_ids=position_ids,
            hidden_states=hidden_states,
            accepted_tokens=accepted_tokens,
            attn_metadata=attn_metadata,
            spec_metadata=spec_metadata,
            draft_model=draft_model,
        )

    def _m3_structural_ok(self, attn_metadata: Any, spec_metadata: Any) -> bool:
        """The batch conditions shared by the fused verify and the fused first drafter inputs: a decode-only plain
        Eagle3 step without attention DP, guided decoding, relaxed acceptance or a suffix-automaton enhancer, of at
        most DECODE_MAX_TOKENS verify rows. Host state only, so the TP ranks agree."""
        if self.is_mtp_eagle or self.use_dynamic_tree:
            return False
        if self._uses_external_shared_target_kv or self.sa_enhancer is not None:
            return False
        if (
            self.guided_decoder is not None
            or self.mapping is None
            or self.mapping.enable_attention_dp
        ):
            return False
        if getattr(self.spec_config, "use_relaxed_acceptance_for_thinking", False):
            return False
        if attn_metadata.num_contexts != 0 or attn_metadata.num_seqs == 0:
            return False
        draft_len = spec_metadata.runtime_draft_len
        if draft_len == 0 or attn_metadata.num_seqs * (draft_len + 1) > dispatch.DECODE_MAX_TOKENS:
            return False
        return not spec_metadata.wants_advanced_draft_sampling

    def tail_eligible(self, attn_metadata: Any, spec_metadata: Any, lm_head: nn.Module) -> bool:
        """Whether this step verifies with the fused kernels instead of the gathered target logits.

        Decode-only Eagle3 batches (``_m3_structural_ok``) over an unpadded vocab-sharded bias-free target head,
        all greedy at a request count ``dispatch.SHARD_ACCEPT`` covers, or sampled under the temperature + top-p
        contract (see _m3_sampled_verify_ok), with nothing else reading the target logits. Every input is host state
        shared by all TP ranks, so the ranks agree.
        """
        if not self._m3_structural_ok(attn_metadata, spec_metadata):
            return False
        if getattr(spec_metadata, "enable_penalty", False):
            return False
        if not self._m3_sharded_head(lm_head) or getattr(lm_head, "bias", None) is not None:
            return False
        draft_len = spec_metadata.runtime_draft_len
        if spec_metadata.is_all_greedy_sample:
            key = (draft_len, attn_metadata.num_seqs)
            if not dispatch.SHARD_ACCEPT.covers(key):
                dispatch.uncovered(
                    dispatch.SHARD_ACCEPT.path,
                    f"draft length {draft_len} at {attn_metadata.num_seqs} requests",
                )
                return False
        elif not self._m3_sampled_verify_ok(spec_metadata):
            return False
        self._m3_target_lm_head = lm_head
        return True

    def draft_only_eligible(
        self, attn_metadata: Any, spec_metadata: Any, draft_model: Optional[nn.Module]
    ) -> bool:
        """Whether a step verified by the in-tree path can still take the fused first drafter inputs: the
        structural conditions of tail_eligible, without the target-sampling ones."""
        return self._m3_structural_ok(attn_metadata, spec_metadata) and self._m3_fc_norm_ok(
            draft_model
        )

    def _m3_seed_first_pass(
        self, num_accepted_tokens: torch.Tensor, spec_metadata: Any, batch_size: int
    ) -> None:
        """The first draft pass's gather rows (the last accepted row of each request) from the in-tree acceptance,
        as _m3_target_accept leaves them."""
        k1 = spec_metadata.runtime_draft_len + 1
        n = num_accepted_tokens[:batch_size].to(torch.int64)
        rows = spec_metadata.batch_indices_cuda[:batch_size].to(torch.int64)
        self._m3_first_gather_ids = rows * k1 + n - 1

    def _m3_first_drafter_inputs(
        self,
        position_ids: torch.Tensor,
        accepted_tokens: torch.Tensor,
        num_tokens: int,
        num_contexts: int,
        spec_metadata: Any,
        draft_model: nn.Module,
    ) -> Optional[dict[str, Any]]:
        """The in-tree first draft inputs of a decode-only step, with the per-layer fc_norm RMSNorms of
        ``apply_eagle3_fc`` in one kernel (unless the capture already applied them) before the in-tree fc; the input
        ids are the accepted tokens as they lie. None to use the in-tree path."""
        if (
            self._m3_first_gather_ids is None
            or self.is_mtp_eagle
            or num_contexts != 0
            or not self._m3_fc_norm_ok(draft_model)
        ):
            return None
        dm = draft_model.model
        layers = len(spec_metadata.layers_to_capture)
        if dm.hidden_size_in != spec_metadata.hidden_size or layers != len(dm.fc_norm):
            return None
        cache = self._m3_fc_norm_w
        if cache is None or cache[0] is not dm.fc_norm[0].weight:
            if torch.cuda.is_current_stream_capturing():
                dispatch.capture_missed("spec_tail", None, "the stacked Eagle3 fc_norm weights")
                return None
            cache = (
                dm.fc_norm[0].weight,
                torch.stack([n.weight.detach() for n in dm.fc_norm]).contiguous(),
            )
            self._m3_fc_norm_w = cache
        hidden = self._m3_fc_hidden
        self._m3_fc_hidden = None
        if hidden is None or hidden.shape[0] != num_tokens:
            hs = spec_metadata.hidden_states[:num_tokens, : spec_metadata.hidden_size * layers]
            if self._m3_capture_norm is None:
                hs = m3_spec_tail.chunk_norm(hs, cache[1], dm.fc_norm[0].variance_epsilon)
            hidden = dm.fc(hs)
        input_ids = accepted_tokens[:, : spec_metadata.runtime_draft_len + 1].reshape(-1)
        return {"input_ids": input_ids, "position_ids": position_ids, "hidden_states": hidden}

    @staticmethod
    def _m3_fc_norm_ok(draft_model: Optional[nn.Module]) -> bool:
        """An Eagle3 drafter with an fc over several captured layers, each normalized by its own fc_norm (one
        shared epsilon): what the fused first drafter inputs compute."""
        dm = getattr(draft_model, "model", None)
        fc_norm = getattr(dm, "fc_norm", None)
        return (
            fc_norm is not None
            and getattr(dm, "fc", None) is not None
            and len({n.variance_epsilon for n in fc_norm}) == 1
        )

    def _m3_fc_fork(
        self,
        spec_metadata: Any,
        num_tokens: int,
        draft_model: Optional[nn.Module],
    ) -> Optional[torch.cuda.Stream]:
        """Launch the first draft pass's fc_norm + fc on a forked stream beside the target head GEMM (after everything
        queued on the current stream); the result is picked up by
        _m3_first_drafter_inputs. Returns the stream to join, or None when the fc path does not apply (the first
        drafter inputs then compute it)."""
        self._m3_fc_hidden = None
        if draft_model is None or not self._m3_fc_norm_ok(draft_model):
            return None
        dm = draft_model.model
        layers = len(spec_metadata.layers_to_capture)
        if (
            dm.hidden_size_in != spec_metadata.hidden_size
            or layers != len(dm.fc_norm)
            or self._m3_fc_norm_w is None
        ):
            return None
        aux = self._m3_aux_stream
        if aux is None:
            aux = self._m3_aux_stream = torch.cuda.Stream()
        aux.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(aux):
            hs = spec_metadata.hidden_states[:num_tokens, : spec_metadata.hidden_size * layers]
            if self._m3_capture_norm is None:
                hs = m3_spec_tail.chunk_norm(
                    hs, self._m3_fc_norm_w[1], dm.fc_norm[0].variance_epsilon, use_pdl=False
                )
            self._m3_fc_hidden = dm.fc(hs)
        return aux

    def _m3_fc_join(self, fork: torch.cuda.Stream) -> None:
        """Join the fc fork onto the current stream and keep its output's memory alive for that stream.

        _m3_fc_hidden is allocated in fork (aux) stream order and read on the current stream. Without the record,
        dropping it returns its block to the aux stream's pool, where later aux work can take it while the current
        stream's reads are still queued (eager warm-up / mixed steps). Nothing is recorded under graph capture or
        torch.compile, as in maybe_execute_in_parallel: a captured graph relies on the next fork's
        aux.wait_stream(main) to order the reuse."""
        main = torch.cuda.current_stream()
        main.wait_stream(fork)
        hidden = self._m3_fc_hidden
        if (
            hidden is not None
            and hidden.is_cuda
            and not (torch.compiler.is_compiling() or torch.cuda.is_current_stream_capturing())
        ):
            hidden.record_stream(main)

    def arm_capture_norm(
        self,
        attn_metadata: Any,
        spec_metadata: Any,
        lm_head: nn.Module,
        draft_model: Optional[nn.Module],
    ) -> None:
        """Before every target forward: whether this step's Eagle3 captures apply the drafter's fc_norm themselves.

        On a fused-verify step whose fc_norm weights are already stacked (by an earlier fused first pass), the
        capture writes rmsnorm(h + r) * fc_norm[i] (one kernel per captured layer instead of add + copy) and the
        fused first drafter inputs read it without their own chunk_norm. A fused-verify step of at most
        ``dispatch.FC_EARLY_MAX_ROWS`` verify rows is also armed to fork the drafter's first fc at the last captured
        layer (``capture_hidden_states``).
        """
        self._m3_capture_norm = None
        self._m3_fc_early_drafter = None
        self._m3_fc_early_stream = None
        if draft_model is None:
            return
        if not self.tail_eligible(attn_metadata, spec_metadata, lm_head):
            return
        # The fork reads every captured column at the last captured layer, so a final-state capture, which is
        # written after the model returns, keeps the fc at its stock place.
        if (
            attn_metadata.num_seqs * (spec_metadata.runtime_draft_len + 1)
            <= dispatch.FC_EARLY_MAX_ROWS
            and FINAL_HIDDEN_LAYER_ID not in spec_metadata.layers_to_capture
        ):
            self._m3_fc_early_drafter = draft_model
        fc_norm_w = self._m3_fc_norm_w
        if (
            fc_norm_w is not None
            and self._m3_fc_norm_ok(draft_model)
            and fc_norm_w[0] is draft_model.model.fc_norm[0].weight
            and len(spec_metadata.layers_to_capture) == fc_norm_w[1].shape[0]
        ):
            self._m3_capture_norm = (fc_norm_w[1], draft_model.model.fc_norm[0].variance_epsilon)

    @staticmethod
    def _m3_sampled_verify_ok(spec_metadata: Any) -> bool:
        """Whether sampled verify rows can use the fused sampled op: only under advanced_sampling_mode=no_topk, the
        deploy-time guarantee that no row uses top-k or min-p (that op applies neither)."""
        mode = getattr(spec_metadata, "advanced_sampling_mode", None)
        return mode is not None and mode.skips_top_k and not mode.skips_top_p

    def _m3_target_accept(
        self,
        hidden_states: torch.Tensor,
        spec_metadata: Any,
        batch_size: int,
        draft_len: int,
        draft_model: Optional[nn.Module] = None,
    ):
        """Fused verification (see tail_eligible); same outputs as the strict acceptance path."""
        force_total = 0
        force_vec = None
        if self.force_num_accepted_tokens != 0.0:
            is_warmup = spec_metadata.is_cuda_graph and not torch.cuda.is_current_stream_capturing()
            if not torch.cuda.is_current_stream_capturing():
                # The draw pool is built by an H2D copy and the draw kernel JIT-compiles per block width, so both
                # happen here, before the first captured step (a scratch counter keeps the real draw sequence).
                self._ensure_force_accept_rng_state(hidden_states.device)
                if not self._m3_force_vec_warm:
                    scratch = torch.zeros(1, dtype=torch.int64, device=hidden_states.device)
                    dummy = torch.empty(64, dtype=torch.int32, device=hidden_states.device)
                    for block in (16, 32, 64):
                        _m3_force_vec_kernel[(1,)](
                            self._force_accept_rng_pool,
                            scratch,
                            dummy,
                            block,
                            1,
                            2,
                            0.5,
                            _FORCE_ACCEPT_RNG_COUNTER_STRIDE,
                            _FORCE_ACCEPT_RNG_SLOT_STRIDE,
                            _FORCE_ACCEPT_RNG_POOL_SIZE - 1,
                            BLOCK=block,
                        )
                    self._m3_force_vec_warm = True
            if not is_warmup:
                force_total = min(int(self.force_num_accepted_tokens) + 1, draft_len + 1)
                force_vec = self._m3_force_vec(
                    batch_size, force_total, draft_len + 1, hidden_states.device
                )
        draft_tokens = spec_metadata.draft_tokens.view(batch_size, draft_len)
        if draft_tokens.dtype != torch.int32:
            draft_tokens = draft_tokens.to(torch.int32)
        # Column 0 receives the last accepted token, the next step's first input (the fused draft loop adds the rest).
        self._m3_next_new = torch.empty(
            (batch_size, draft_len + 1), dtype=torch.int32, device=hidden_states.device
        )
        rows = hidden_states[: batch_size * (draft_len + 1)]
        lm_head = self._m3_target_lm_head
        batch_idx = spec_metadata.batch_indices_cuda[:batch_size]
        # The Eagle3 fc on the aux stream: the fork made at the last captured layer (none at FC_EARLY_MAX_ROWS = 0),
        # else one beside the head.
        fork = self._m3_fc_early_stream
        self._m3_fc_early_stream = None
        if fork is None:
            fork = self._m3_fc_fork(spec_metadata, rows.shape[0], draft_model)
        # This rank's vocabulary shard of the target logits: the in-tree LM head's own local BF16 GEMM, whose
        # all-gather the fused kernels replace (they exchange per-row records across ranks, not logits).
        logits = lm_head.apply_linear(rows, None)
        vocab_offset = self.mapping.tp_rank * lm_head.weight.shape[0]
        if spec_metadata.is_all_greedy_sample:
            result = m3_shard_accept.shard_accept(
                logits,
                vocab_offset,
                self.mapping,
                draft_tokens,
                batch_idx,
                force_total,
                self._m3_next_new,
                force_vec=force_vec,
            )
        else:
            n_rows = rows.shape[0]
            seeds, offsets = self._rng_state_per_token(spec_metadata, n_rows)
            result = m3_sampled_accept.sampled_accept(
                logits,
                vocab_offset,
                self.mapping,
                draft_tokens,
                batch_idx,
                spec_metadata.temperatures[:n_rows],
                spec_metadata.top_ks[:n_rows],
                spec_metadata.top_ps[:n_rows],
                None,
                seeds,
                offsets,
                force_total,
                self._m3_next_new,
                force_vec=force_vec,
            )
        if fork is not None:
            self._m3_fc_join(fork)
        if result is None:
            raise RuntimeError(
                "M3 fused verify unavailable after tail_eligible() accepted the step"
            )
        return result

    def _m3_force_vec(
        self, batch_size: int, base_total: int, max_total: int, device: torch.device
    ) -> Optional[torch.Tensor]:
        """Per-request forced accepted counts for a fractional forced acceptance, else None.

        Draws exactly as _apply_force_accepted_tokens does for a decode-only batch (same fixed-seed pool, one counter
        bump per step, the same (counter, slot) hash), so every TP rank and the generic path agree on the counts.
        """
        frac = self.force_num_accepted_tokens - int(self.force_num_accepted_tokens)
        if frac <= 0.0 or base_total >= max_total:
            return None
        self._ensure_force_accept_rng_state(device)
        out = torch.empty(batch_size, dtype=torch.int32, device=device)
        # One launch: bump the shared counter and draw, with _apply_force_accepted_tokens' exact arithmetic.
        _m3_force_vec_kernel[(1,)](
            self._force_accept_rng_pool,
            self._force_accept_rng_counter,
            out,
            batch_size,
            base_total,
            max_total,
            float(frac),
            _FORCE_ACCEPT_RNG_COUNTER_STRIDE,
            _FORCE_ACCEPT_RNG_SLOT_STRIDE,
            _FORCE_ACCEPT_RNG_POOL_SIZE - 1,
            BLOCK=max(16, 1 << (batch_size - 1).bit_length()),
        )
        return out

    @staticmethod
    def _m3_sharded_head(lm_head: nn.Module) -> bool:
        """A vocab-sharded TP head without padding (the fused kernels read the local shard)."""
        return (
            getattr(lm_head, "tp_mode", None) == TensorParallelMode.COLUMN
            and getattr(lm_head, "padding_size", 0) == 0
            and getattr(lm_head, "tp_size", 1) > 1
        )

    # ---------------------------------------------------------------------------------------------- fused draft loop

    def prebuild_draft_head(self, draft_model: Optional[nn.Module]) -> bool:
        """Build the fragment-ordered copy of the drafter's BF16 head shard that the fused draft loop streams (rebuilt
        from the current weight values on every call). Called by the target's ``cache_derived_state`` once the weights
        are real, before the executor's warmup, on every rank. False when the drafter is not one the loop serves."""
        self._m3_draft_head = None
        if draft_model is None or not self._m3_drafter_supported(draft_model):
            return False
        self._m3_draft_head = m3_draft_step.prepare_head(draft_model.lm_head, self.mapping.tp_rank)
        return True

    def prebuild_drafter_linears(self, draft_model: Optional[nn.Module]) -> int:
        """Put the drafter's BF16 linears on the fixed-tactic low-M kernels (``dispatch.DRAFT_LOWM``) and compile every
        covered call, and cap the drafter attention's trtllm-gen KV split per request count
        (``dispatch.DRAFT_DECODE_MAX_SEQ_LEN_BY_BATCH``). Called by the target's ``cache_derived_state`` once the
        weights are real, before the executor's warmup, on every rank. Returns the number of low-M calls compiled."""
        if draft_model is None:
            return 0
        dm = getattr(draft_model, "model", None)
        attn = getattr(getattr(getattr(dm, "midlayer", None), "self_attn", None), "attn", None)
        if attn is not None:
            attn.dense_decode_max_seq_len_by_batch = dispatch.DRAFT_DECODE_MAX_SEQ_LEN_BY_BATCH
        m3_draft_lowm.install(draft_model)
        return m3_draft_lowm.prebuild(draft_model)

    def _m3_drafter_supported(self, draft_model: nn.Module) -> bool:
        """The drafter the fused draft loop runs: one plain (non-MLA) Eagle3 layer with a gated MLP over the
        target-vocabulary embedding (whole table on every rank), a draft vocabulary equal to the target's (no d2t),
        all-reduced TP projections, and a bias-free vocab-sharded BF16 head without padding."""
        dm = getattr(draft_model, "model", None)
        if (
            dm is None
            or getattr(dm, "num_layers", 0) != 1
            or dm._use_mla
            or dm._eh_proj_before_attn
        ):
            return False
        layer = dm.midlayer
        if layer._next_layer_regular or not isinstance(layer.mlp, GatedMLP):
            return False
        embed = dm.embed_tokens
        if embed is None or getattr(embed, "tp_size", 1) != 1:
            return False
        if self.mapping is None or self.mapping.tp_size <= 1 or self.mapping.enable_attention_dp:
            return False
        if getattr(layer.self_attn.o_proj, "all_reduce", None) is None:
            return False
        if getattr(layer.mlp.down_proj, "all_reduce", None) is None:
            return False
        if self._d2t is not None or getattr(dm, "d2t", None) is not None:
            return False
        lm_head = draft_model.lm_head
        return self._m3_sharded_head(lm_head) and m3_draft_step.head_supported(lm_head)

    def _m3_fused_loop_ok(
        self,
        attn_metadata: Any,
        spec_metadata: Any,
        batch_size: int,
        draft_model: Optional[nn.Module],
    ) -> bool:
        """Whether this step drafts with the fused loop: a decode-only step the fused verify or the seed prepared
        (``_m3_first_gather_ids``) with the fused first draft inputs, a supported drafter, a (draft length, requests)
        ``dispatch.DRAFT_STEP`` covers, the prebuilt head and the exchange workspace. Host state only, so the TP
        ranks agree."""
        if (
            self._m3_first_gather_ids is None
            or not self._m3_fused_first_inputs
            or draft_model is None
        ):
            return False
        path = dispatch.DRAFT_STEP.path
        if not self._m3_drafter_supported(draft_model) or not hasattr(
            attn_metadata, "kv_lens_cuda"
        ):
            dispatch.uncovered(
                path, "the drafter or its attention metadata is not one the fused loop serves"
            )
            return False
        draft_len = spec_metadata.runtime_draft_len
        if not dispatch.DRAFT_STEP.covers((draft_len, batch_size)):
            dispatch.uncovered(path, f"draft length {draft_len} at {batch_size} requests")
            return False
        head = self._m3_draft_head
        if head is None or head.weight is not draft_model.lm_head.weight:
            dispatch.missed(
                path,
                batch_size,
                "the drafter head's fragment-ordered copy is not built for its current weight (prebuild_draft_head)",
            )
            return False
        return m3_draft_step.workspace(self.mapping, head.wf.device) is not None

    def _forward_draft_loop(
        self,
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
    ):
        """The fused draft loop on a step it covers (``_m3_fused_loop_ok``), else the in-tree loop."""
        if self._m3_fused_loop_ok(attn_metadata, spec_metadata, batch_size, draft_model):
            return self._m3_fused_draft_loop(
                inputs,
                attn_metadata,
                spec_metadata,
                draft_model,
                draft_kv_cache_manager,
                batch_size,
                num_accepted_tokens,
            )
        return super()._forward_draft_loop(
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

    def _m3_fused_draft_loop(
        self,
        inputs: dict[str, Any],
        attn_metadata: Any,
        spec_metadata: Any,
        draft_model: nn.Module,
        draft_kv_cache_manager: Any,
        batch_size: int,
        num_accepted_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """Mirrors Eagle3OneModelWorker._forward_linear_draft_loop for a decode-only Eagle3 step, with each pass's glue
        fused (see ``_m3_fused_loop_ok``):

        * the draft input -- the embedding lookup, the layer's input_layernorm and hidden_norm and their concat -- in
          one launch (``m3_spec_tail.draft_input``; the second pass gathers the last accepted rows there);
        * the drafter layer's in-tree attention and MLP, each followed by the in-tree all-reduce with the residual add
          and the next RMSNorm fused into it (post_attention_layernorm, then the drafter's final norm);
        * one kernel for the BF16 head over the head rows (pass 0: the last accepted row of each request), the
          cross-rank argmax and the bookkeeping the in-tree loop does between passes: the draft token, the next
          pass's input id and position, the draft KV lengths (and pass 0's sequence lengths) (``m3_draft_step``).

        The host metadata transitions are the in-tree loop's. The drafts land in columns 1.. of the step's next-input
        tokens (column 0 is the last accepted token), which ``_forward_impl`` returns as they lie.
        """
        draft_len = spec_metadata.runtime_draft_len
        dm = draft_model.model
        layer = dm.midlayer
        attn = layer.self_attn
        mlp = layer.mlp
        attn_all_reduce = attn.o_proj.all_reduce
        mlp_all_reduce = mlp.down_proj.all_reduce
        table = dm.embed_tokens.weight
        dev = table.device
        head = self._m3_draft_head
        gather_ids = self._m3_first_gather_ids
        ids = inputs["input_ids"]
        hidden = inputs["hidden_states"]
        pos = inputs["position_ids"]
        if ids.dtype != torch.int32:
            ids = ids.to(torch.int32)
        if pos.dtype != torch.int32:
            pos = pos.to(torch.int32)
        num_acc = num_accepted_tokens[:batch_size]
        if num_acc.dtype != torch.int32:
            num_acc = num_acc.to(torch.int32)
        next_new = self._m3_next_new
        if next_new is None:
            # Seeded from the in-tree acceptance: the last accepted token is the first-pass input at the gather row.
            next_new = torch.empty((batch_size, draft_len + 1), dtype=torch.int32, device=dev)
            next_new[:, 0] = ids[gather_ids]
        draft_tokens = next_new[:, 1:]

        no_all_reduce = AllReduceParams(enable_allreduce=False)
        with self.draft_kv_cache_context(attn_metadata, draft_kv_cache_manager) as md:
            for i in range(draft_len):
                x, residual = m3_spec_tail.draft_input(
                    ids,
                    table,
                    hidden,
                    gather_ids if i == 1 else None,
                    layer.input_layernorm.weight,
                    layer.input_layernorm.variance_epsilon,
                    layer.hidden_norm.weight,
                    layer.hidden_norm.variance_epsilon,
                )
                h = attn(
                    position_ids=pos,
                    hidden_states=x,
                    attn_metadata=md,
                    all_reduce_params=no_all_reduce,
                )
                h, residual = draft_ar_norm(
                    attn_all_reduce, h, residual, layer.post_attention_layernorm
                )
                h = mlp(h, final_all_reduce_params=no_all_reduce)
                h, to_save = draft_ar_norm(mlp_all_reduce, h, residual, dm.norm)
                if dm._return_hidden_post_norm:
                    to_save = h
                ids_next = torch.empty(batch_size, dtype=torch.int32, device=dev)
                pos_next = torch.empty(batch_size, dtype=torch.int32, device=dev)
                if not m3_draft_step.draft_step(
                    head,
                    h,
                    batch_size,
                    gather_ids if i == 0 else None,
                    self.mapping,
                    draft_tokens,
                    i,
                    ids_next,
                    pos,
                    pos_next,
                    md.kv_lens_cuda,
                    num_acc if i == 0 else None,
                    md._seq_lens_cuda if i == 0 else None,
                    draft_len,
                ):
                    raise RuntimeError(
                        "M3 fused draft step unavailable after _m3_fused_loop_ok() accepted the step"
                    )
                ids, pos, hidden = ids_next, pos_next, to_save
                if i == 0:
                    # The draft step set the device sequence lengths to 1 and rolled the KV lengths back.
                    md._seq_lens[:batch_size].fill_(1)
                    md.on_update()
                    has_kv_cache = md.kv_cache_manager is not None
                    if has_kv_cache and hasattr(md, "host_request_types"):
                        md.host_request_types[: md.num_contexts].fill_(1)
                    if has_kv_cache:
                        md.num_contexts = 0
                        self._prepare_flash_mla_generation_layout(md, 0, batch_size)
                    md.update_for_spec_dec()
                    md.use_spec_decoding = False
                else:
                    md.update_for_spec_dec()
        self._m3_next_new_tokens = next_new
        return draft_tokens


class M3SpecDecForwardMixin:
    """The target CausalLM's step forward with the fused-verify hooks of the speculative wrapper.

    Mirrors ``SpecDecOneEngineForCausalLM.forward`` (tensorrt_llm/_torch/models/modeling_speculative.py) with the
    fused-verify branches: before the target forward the worker arms the capture norm, and a step the worker verifies
    from the logits shard (``M3Eagle3Worker.tail_eligible``) skips the gathered target logits.

    The class it is mixed into (listed before ``DecoderModelForCausalLM``) provides ``model`` (called like the in-tree
    decoder model, capturing through ``M3Eagle3Worker.capture_hidden_states``), ``lm_head``, ``logits_processor``,
    ``model_config``, ``draft_model`` and ``spec_worker`` (an ``M3Eagle3Worker``, or None without speculation).
    """

    def forward(
        self,
        attn_metadata: Any,
        input_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        return_context_logits: bool = False,
        spec_metadata: Optional[Any] = None,
        resource_manager: Optional[Any] = None,
        **kwargs,
    ) -> Any:
        spec_worker: Optional[M3Eagle3Worker] = self.spec_worker
        if spec_worker is not None and spec_metadata is not None:
            spec_worker.arm_capture_norm(
                attn_metadata, spec_metadata, self.lm_head, self.draft_model
            )
        hidden_states = self.model(
            input_ids=input_ids,
            attn_metadata=attn_metadata,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            spec_metadata=spec_metadata,
            **kwargs,
        )
        if getattr(attn_metadata, "csa2_remote_tail_mode", None) == "source":
            # A context worker stopped at the encoder/decoder boundary: a disposable token for the context-only
            # completion, and no draft worker.
            return self.logits_processor.forward(hidden_states, self.lm_head, attn_metadata, False)
        if spec_metadata is not None and spec_metadata.is_layer_capture(FINAL_HIDDEN_LAYER_ID):
            if spec_worker is not None:
                spec_worker.capture_hidden_states(
                    spec_metadata, FINAL_HIDDEN_LAYER_ID, hidden_states, None
                )
            else:
                capture_hidden_states(
                    spec_metadata, FINAL_HIDDEN_LAYER_ID, hidden_states, None, None
                )
        if attn_metadata.padded_num_tokens is not None:
            hidden_states = hidden_states[: attn_metadata.num_tokens]

        if spec_worker is not None:
            # The fused verify computes this rank's logits shard itself and needs no gathered logits.
            if spec_worker.tail_eligible(attn_metadata, spec_metadata, self.lm_head):
                logits = None
            else:
                logits = self.logits_processor.forward(
                    hidden_states[spec_metadata.gather_ids],
                    self.lm_head,
                    attn_metadata,
                    True,
                )
            spec_input_ids = input_ids if input_ids is not None else kwargs.get("orig_input_ids")
            spec_position_ids = position_ids
            if attn_metadata.padded_num_tokens is not None:
                if spec_input_ids is not None:
                    spec_input_ids = spec_input_ids[: attn_metadata.num_tokens]
                if position_ids is not None:
                    spec_position_ids = _slice_spec_position_ids(
                        position_ids, attn_metadata.num_tokens
                    )
            return spec_worker(
                input_ids=spec_input_ids,
                position_ids=spec_position_ids,
                hidden_states=hidden_states,
                logits=logits,
                attn_metadata=attn_metadata,
                spec_metadata=spec_metadata,
                draft_model=self.draft_model,
                resource_manager=resource_manager,
            )

        return self.logits_processor.forward(
            hidden_states,
            self.lm_head,
            attn_metadata,
            return_context_logits,
        )
