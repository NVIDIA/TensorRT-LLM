# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import functools
import inspect
from typing import Optional

import torch
from torch import nn

from tensorrt_llm._torch.attention.backends import AttentionMetadata
from tensorrt_llm._torch.pyexecutor.kv_cache.mamba_cache_manager import MambaHybridCacheManager
from tensorrt_llm._torch.speculative.dflash import (
    DFlashSpecMetadata,
    DFlashWorker,
    dflash_allocated_ctx_limit,
    dflash_noise_block_embedding,
)
from tensorrt_llm._torch.speculative.dspark import DSparkWorker
from tensorrt_llm._torch.speculative.interface import SpecMetadata, SpecWorkerBase
from tensorrt_llm.logger import logger


@functools.lru_cache(maxsize=None)
def _takes_ctx_rows_start_cls(cls) -> bool:
    return "ctx_rows_start" in inspect.signature(cls.dflash_forward).parameters


def _takes_ctx_rows_start(draft_model) -> bool:
    """Whether the draft model's ``dflash_forward`` accepts ``ctx_rows_start``."""
    return _takes_ctx_rows_start_cls(type(draft_model))


def dflash_ctx_rows_kwargs(draft_model, ctx_rows_start: Optional[int]) -> dict:
    """The ``ctx_rows_start`` keyword (from ``prepare_1st_drafter_inputs``) for a draft model whose ``dflash_forward``
    takes it; no keyword for another draft model or without a start."""
    if ctx_rows_start is None or not _takes_ctx_rows_start(draft_model):
        return {}
    return {"ctx_rows_start": ctx_rows_start}


def capture_view(
    spec_metadata: Optional[SpecMetadata], layer_id: int, num_tokens: int
) -> Optional[torch.Tensor]:
    """``layer_id``'s slot of the DFlash capture buffer for ``num_tokens`` rows (a strided view a kernel can write the
    tap into), or None when ``spec_metadata`` is not DFlash's or the layer is not captured."""
    if (
        not isinstance(spec_metadata, DFlashSpecMetadata)
        or spec_metadata.captured_hidden_states is None
    ):
        return None
    i = spec_metadata._layer_to_idx.get(layer_id)
    if i is None:
        return None
    return spec_metadata.captured_hidden_states[
        :num_tokens, i * spec_metadata.hidden_size : (i + 1) * spec_metadata.hidden_size
    ]


class KimiK3DFlashWorker(DFlashWorker):
    # Set by a Kimi K3 target running its decode kernels: decode steps then run the acceptance and the drafter's
    # inputs as trtllm::k3_spec_accept, keep the target logits vocabulary-sharded and write the drafter's context
    # K/V with trtllm::k3_ctx_kv.
    k3_decode = False

    def _mask_token_id(self, draft_model) -> int:
        """The drafter's mask token id, resolved once."""
        if self._resolved_mask_token_id is None:
            if (
                hasattr(self.spec_config, "mask_token_id")
                and self.spec_config.mask_token_id is not None
            ):
                self._resolved_mask_token_id = self.spec_config.mask_token_id
            elif hasattr(draft_model, "mask_token_id"):
                self._resolved_mask_token_id = draft_model.mask_token_id
            elif hasattr(draft_model.model, "mask_token_id"):
                self._resolved_mask_token_id = draft_model.model.mask_token_id
            else:
                raise ValueError(
                    "DFlash requires mask_token_id to be set. Please set it in DFlashDecodingConfig "
                    "or ensure the draft model config has 'dflash_config.mask_token_id' or 'mask_token_id'."
                )
        return self._resolved_mask_token_id

    @staticmethod
    def _trained_mask_embedding(draft_model, mask_token_id: int) -> Optional[torch.Tensor]:
        """The drafter's own trained mask row, if it kept one for the mask id in use."""
        if mask_token_id != getattr(draft_model, "mask_token_id", None):
            return None
        return getattr(draft_model, "mask_token_embedding", None)

    def _k3_accept_applies(
        self, logits, attn_metadata, spec_metadata, draft_model, num_contexts: int, num_gens: int
    ) -> bool:
        """Whether this step's acceptance and the drafter's inputs run as ``trtllm::k3_spec_accept``.

        That mode: a decode step (no context requests, at most 8 gen requests) with greedy strict acceptance (no
        rejection sampling, penalties or guided decoding, the base draft-token and logits layouts), fp32 target
        logits of a vocabulary the kernel splits, the V2 Mamba manager's KDA replay record, the draft pool's block
        table, and an unsharded bf16 draft embedding. Any other step keeps the torch path.
        """
        if not self._k3_accept_step_applies(
            attn_metadata, spec_metadata, draft_model, num_contexts, num_gens
        ):
            return False
        K = spec_metadata.runtime_draft_len
        if (
            logits.dim() != 2
            or logits.dtype != torch.float32
            or logits.shape[0] != num_gens * (K + 1)
        ):
            return False
        from tensorrt_llm._torch.cute_dsl_kernels.k3_spec_accept import op as accept_op

        embed = draft_model.draft_model_full.model.embed_tokens
        return accept_op.supports(
            logits.shape[1], num_gens, self._compute_block_size, K, embed.weight.shape[1]
        )

    def _k3_accept_step_applies(
        self, attn_metadata, spec_metadata, draft_model, num_contexts: int, num_gens: int
    ) -> bool:
        """Everything ``_k3_accept_applies`` checks but the target logits."""
        if not self.k3_decode:
            return False
        from tensorrt_llm._torch.pyexecutor.kv_cache.mamba_cache_manager import (
            MambaHybridCacheManagerV2,
        )

        if num_contexts != 0 or not 0 < num_gens <= 8 or self.guided_decoder is not None:
            return False
        if (
            not spec_metadata.is_all_greedy_sample
            or self._can_use_rejection_sampling(spec_metadata)
            or getattr(spec_metadata, "enable_penalty", False)
            or type(self)._reshape_draft_tokens_for_accept
            is not SpecWorkerBase._reshape_draft_tokens_for_accept
            or type(self)._reshape_logits_for_accept
            is not SpecWorkerBase._reshape_logits_for_accept
        ):
            return False
        K = spec_metadata.runtime_draft_len
        draft_tokens = spec_metadata.draft_tokens
        if (
            K <= 0
            or draft_tokens is None
            or draft_tokens.dtype != torch.int32
            or draft_tokens.numel() != num_gens * K
        ):
            return False
        mgr = attn_metadata.kv_cache_manager
        mamba_metadata = getattr(attn_metadata, "mamba_metadata", None)
        if (
            not isinstance(mgr, MambaHybridCacheManagerV2)
            or type(mgr).update_mamba_states is not MambaHybridCacheManagerV2.update_mamba_states
            or not mgr.use_kda_replay_update
            or getattr(mgr, "prev_num_accepted_tokens", None) is None
            or getattr(mgr, "_dummy_request_mask", None) is None
            or mamba_metadata is None
            or mamba_metadata.state_indices.dtype != torch.int32
        ):
            return False
        if (
            self._ctx_block_tables is None
            or getattr(attn_metadata, "draft_kv_cache_block_offsets", None) is None
            or getattr(attn_metadata, "kv_lens_cuda", None) is None
        ):
            return False
        embed = draft_model.draft_model_full.model.embed_tokens
        return getattr(embed, "tp_size", 1) == 1 and embed.weight.dtype == torch.bfloat16

    def target_logits(
        self, hidden_states, lm_head, logits_processor, attn_metadata, spec_metadata, draft_model
    ) -> torch.Tensor:
        """This rank's bf16 vocabulary shard of the target logits, [rows, vocab / TP] (``_k3_head_shard``: the head
        GEMM without its all-gather and fp32 cast), for a step whose acceptance runs as the sharded
        ``trtllm::k3_spec_accept`` (see ``_k3_logits_shard``); the gathered fp32 logits otherwise. At such a step the
        acceptance is their only reader: the one-model spec sampler takes the worker's tokens and rejects requests for
        logits or log probabilities, so the shard is also what the step returns as ``"logits"``."""
        self._k3_step_shard = None
        shard = self._k3_logits_shard(
            hidden_states, lm_head, attn_metadata, spec_metadata, draft_model
        )
        if shard is None:
            return super().target_logits(
                hidden_states, lm_head, logits_processor, attn_metadata, spec_metadata, draft_model
            )
        logits = self._k3_head_shard(logits_processor, lm_head, hidden_states)
        self._k3_step_shard = (logits, shard)
        return logits

    @staticmethod
    def _k3_head_shard(logits_processor, lm_head, rows: torch.Tensor) -> torch.Tensor:
        """This rank's bf16 vocabulary shard of ``lm_head(rows)``, without the head's all-gather: from the logits
        processor's own head kernel where it has one that takes the rows (``lm_head_shard``; the Kimi K3 target's
        ``gemm/k3_head_gemv``), so the shard holds the values of the processor's gathered logits; else from the head's
        own GEMM."""
        head_shard = getattr(logits_processor, "lm_head_shard", None)
        logits = None if head_shard is None else head_shard(rows, lm_head)
        if logits is None:
            logits = lm_head.apply_linear(rows, lm_head.bias)
        return logits

    def _k3_logits_shard(self, hidden_states, lm_head, attn_metadata, spec_metadata, draft_model):
        """``(workspace, first column)`` when this step's target logits stay vocabulary-sharded, else None.

        That needs a step ``_k3_accept_step_applies`` takes, plain TP whose all-reduces own an MNNVL workspace for
        this mapping, and an unquantized, bias-free, unpadded, evenly split column-parallel bf16 lm_head with shards
        the kernel splits. The exchange is collective, so every rank must decide alike: every condition is the
        batch's or the configuration's. The workspace is allocated (collectively) and the kernel compiled outside
        CUDA-graph capture only, so a graph whose warmup allocated no workspace is captured on the gathered path.
        """
        if not self.k3_decode:
            return None
        from tensorrt_llm._torch.cute_dsl_kernels.k3_spec_accept import op as accept_op
        from tensorrt_llm._torch.distributed.ops import MNNVLAllReduce

        mapping = self.mapping
        if getattr(self, "_k3_shard_head", None) is not lm_head:
            reason = self._k3_logits_shard_reason(lm_head)
            if reason is None and MNNVLAllReduce.allreduce_mnnvl_workspaces.get(mapping) is None:
                # Not kept: re-checked on later steps, since the model's MNNVL workspace may not exist yet.
                logger.info_once(
                    "DFlash: the target logits are all-gathered (the TP all-reduces are not MNNVL)",
                    key="dflash_k3_logits_no_mnnvl",
                )
                return None
            if mapping is not None and mapping.tp_size > 1:
                if reason is None:
                    logger.info(
                        f"DFlash: decode-step target logits stay vocabulary-sharded ({lm_head.weight.shape[0]} "
                        f"columns per rank); trtllm::k3_spec_accept exchanges the row maxima, and a request's "
                        f"logits post-processors get them gathered"
                    )
                else:
                    logger.info_once(
                        f"DFlash: the target logits are all-gathered ({reason})", key=reason
                    )
            self._k3_shard_head, self._k3_shard_ok = lm_head, reason is None
        if not self._k3_shard_ok:
            return None
        num_contexts = attn_metadata.num_contexts
        num_gens = attn_metadata.num_seqs - num_contexts
        K = spec_metadata.runtime_draft_len
        if hidden_states.shape[0] != num_gens * (K + 1) or not self._k3_accept_step_applies(
            attn_metadata, spec_metadata, draft_model, num_contexts, num_gens
        ):
            return None
        embed = draft_model.draft_model_full.model.embed_tokens
        columns = lm_head.weight.shape[0]
        if not accept_op.supports(
            columns, num_gens, self._compute_block_size, K, embed.weight.shape[1], mapping.tp_size
        ):
            return None
        if torch.cuda.is_current_stream_capturing():
            ws = accept_op.existing_workspace(mapping)
            if ws is None:
                return None
        else:
            ws = accept_op.workspace(mapping)
        return ws, mapping.tp_rank * columns

    def _k3_logits_shard_reason(self, lm_head):
        """Why the target logits cannot stay vocabulary-sharded under this configuration and head (None if they can,
        given an MNNVL workspace for the mapping, which ``_k3_logits_shard`` checks)."""
        from tensorrt_llm._torch.cute_dsl_kernels.k3_spec_accept import op as accept_op
        from tensorrt_llm._torch.modules.linear import TensorParallelMode

        mapping = self.mapping
        if mapping is None or mapping.tp_size <= 1:
            return "no TP"
        if mapping.enable_attention_dp or mapping.has_cp() or mapping.has_pp():
            return "attention DP, CP or PP"
        if (
            getattr(lm_head, "tp_mode", None) != TensorParallelMode.COLUMN
            or not getattr(lm_head, "gather_output", False)
            or getattr(lm_head, "padding_size", 0) != 0
            or getattr(lm_head, "gather_output_sizes", None) is not None
        ):
            return "the lm_head is not an evenly split, unpadded column-parallel head"
        if (
            lm_head.bias is not None
            or lm_head.has_any_quant
            or lm_head.weight.dtype != torch.bfloat16
        ):
            return "the lm_head has a bias, is quantized or is not bf16"
        if not accept_op.supports_columns(lm_head.weight.shape[0], mapping.tp_size):
            columns = lm_head.weight.shape[0]
            return f"trtllm::k3_spec_accept cannot split {columns} columns over {mapping.tp_size} ranks"
        return None

    def _k3_static_arange(self, lo: int, hi: int) -> torch.Tensor:
        """``arange(lo, hi)`` (int64), built once outside CUDA-graph capture."""
        cache = self.__dict__.setdefault("_k3_aranges", {})
        t = cache.get((lo, hi))
        if t is None:
            t = torch.arange(lo, hi, dtype=torch.long, device="cuda")
            if not torch.cuda.is_current_stream_capturing():
                cache[(lo, hi)] = t
        return t

    def _k3_mask_row(self, draft_model) -> torch.Tensor:
        """The noise block's mask embedding, bf16 [hidden], cached: the drafter's own trained mask row if it kept one
        for the mask id in use, else the embedding's row (``dflash_noise_block_embedding`` reads the same row)."""
        row = getattr(self, "_k3_mask_embedding", None)
        if row is None:
            embed = draft_model.draft_model_full.model.embed_tokens
            mask_token_id = self._mask_token_id(draft_model)
            trained = self._trained_mask_embedding(draft_model, mask_token_id)
            if trained is not None:
                row = trained.to(embed.weight.dtype).reshape(-1).contiguous()
            else:
                row = embed.weight[mask_token_id].contiguous()
            if torch.cuda.is_current_stream_capturing():
                return row
            self._k3_mask_embedding = row
        return row

    def _k3_ctx_pool_key(self):
        """The identity of the pool the ctx cache is bound to now (KV cache estimation rebinds it to a new one)."""
        return (
            tuple(t.data_ptr() for t in self._ctx_kv_buf) if self._ctx_kv_buf is not None else None
        )

    def _k3_ctx_kv_applies(self, draft_model, projected: torch.Tensor) -> bool:
        """Whether ``trtllm::k3_ctx_kv`` writes this step's context K/V: the manager-bound paged pool with K and V,
        bf16, the fused K/V weight without bias or context input norm, k_norm, NeoX RoPE from flashinfer's fp32
        cache, a context the op can split (up to 64 tokens: B <= 8 requests of K + 1 <= 8, ``ctx_op.pick_split``), one
        allocation for every layer's pool. Checked once per token count and
        bound pool; the pool view is kept for the bound pool only, so a rebind never leaves the kernel writing the
        replaced pool."""
        pool_key = self._k3_ctx_pool_key()
        if getattr(self, "_k3_ctx_pool_bound", None) != pool_key:
            self._k3_ctx_pool_bound = pool_key
            self._k3_ctx_pool = None
            self._k3_ctx_kv_ok = {}
        cache = self._k3_ctx_kv_ok
        n = projected.shape[0]
        ok = cache.get(n)
        if ok is not None:
            return ok
        from tensorrt_llm._torch.cute_dsl_kernels.k3_ctx_kv import op as ctx_op
        from tensorrt_llm._torch.models import modeling_dflash

        reason = None
        if not (self._ctx_paged and self._ctx_block_tables is not None):
            reason = "the context pool is not the manager-bound paged pool"
        elif projected.dtype != torch.bfloat16 or self._ctx_kv_buf[0].dtype != torch.bfloat16:
            reason = "not bf16"
        else:
            if draft_model._fused_kv_weight is None:
                draft_model._build_fused_kv_buffers()
            view = ctx_op.pool_view(list(self._ctx_kv_buf))
            if (
                draft_model._fused_kv_bias is not None
                or getattr(draft_model, "_input_ln_eps", None) is not None
            ):
                reason = "K/V bias or context input norm"
            elif draft_model._k_norm_stacked is None or not draft_model._is_neox:
                reason = "no k_norm or not NeoX RoPE"
            elif (
                modeling_dflash._flashinfer_rope is None
                or draft_model._get_cos_sin_cache().shape[1] != 64
            ):
                reason = "RoPE is not flashinfer's full-rotary fp32 cache"
            elif view is None or self._ctx_kv_buf[0].size(1) != 2:
                reason = "the layers' pools are not K/V views of one allocation"
            elif not ctx_op.pick_split(
                draft_model._fused_kv_weight.shape[0],
                projected.shape[1],
                draft_model._num_kv_heads,
                self._draft_tokens_per_req,
                n,
                projected.device,
            ):
                reason = f"{n} context tokens (at most 64) or the K/V weight shape"
            else:
                self._k3_ctx_pool = view
        ok = reason is None
        if not ok:
            logger.info(f"DFlash: context K/V on the Python path ({reason})")
        if not torch.cuda.is_current_stream_capturing():
            cache[n] = ok
        return ok

    def _k3_ctx_kv(
        self, draft_model, projected, ctx_positions, num_accepted, slots, rows
    ) -> torch.Tensor:
        """``trtllm::k3_ctx_kv``: every drafter layer's context K/V of this step into the paged pool and
        ``_ctx_len += num_accepted`` (clamped); returns the context length each request may advertise."""
        if self._k3_ctx_pool is None or self._k3_ctx_pool_bound != self._k3_ctx_pool_key():
            raise RuntimeError("k3_ctx_kv: the context pool was rebound after the path was chosen")
        flat, layer_off, page_stride, kv_stride, head_stride = self._k3_ctx_pool
        return torch.ops.trtllm.k3_ctx_kv(
            projected.contiguous(),
            draft_model._fused_kv_weight,
            draft_model._k_norm_stacked,
            draft_model._get_cos_sin_cache(),
            ctx_positions,
            num_accepted,
            self._ctx_len,
            slots,
            rows,
            self._ctx_block_tables,
            self._ctx_block_counts,
            flat,
            layer_off,
            page_stride,
            kv_stride,
            head_stride,
            draft_model._k_norm_eps,
            self._max_ctx,
            self._ctx_page_size,
            self._compute_block_size,
            draft_model._num_kv_heads,
        )

    def _on_acceptance(
        self, accepted_tokens, num_accepted_tokens, attn_metadata, spec_metadata
    ) -> None:
        """Called with this step's acceptance when ``k3_spec_accept`` produced it (a hook for drafter families)."""

    def _k3_accept(self, logits, attn_metadata, spec_metadata, draft_model, shard=None):
        """``trtllm::k3_spec_accept`` for a step ``_k3_accept_applies`` accepted: the acceptance, the block table,
        the KDA replay record, kv_lens + 1 and the drafter's inputs in one launch. ``shard``: the exchange
        workspace and first column when ``logits`` are this rank's vocabulary shard (``_k3_logits_shard``).
        Returns (accepted tokens, num accepted, {rewind, bonus, qpos, cpos, noise})."""
        from tensorrt_llm._torch.cute_dsl_kernels.k3_spec_accept import op as accept_op

        num_gens = attn_metadata.num_seqs
        K = spec_metadata.runtime_draft_len
        mgr = attn_metadata.kv_cache_manager
        is_warmup = spec_metadata.is_cuda_graph and not torch.cuda.is_current_stream_capturing()
        force = 0.0 if is_warmup else float(self.force_num_accepted_tokens)
        mode, _, _ = accept_op.force_mode(force, K)
        if mode == accept_op._kernel_module().FORCE_FRAC:
            self._ensure_force_accept_rng_state(logits.device)
            pool, counter = self._force_accept_rng_pool, self._force_accept_rng_counter
        else:
            dummy_rng = getattr(self, "_k3_dummy_rng", None)
            if dummy_rng is None:
                dummy_rng = self._k3_dummy_rng = (
                    torch.zeros(4, dtype=torch.float32, device=logits.device),
                    torch.zeros(2, dtype=torch.int64, device=logits.device),
                )
            pool, counter = dummy_rng
        embed = draft_model.draft_model_full.model.embed_tokens
        accepted, num_acc, rewind, bonus, qpos, cpos, noise = torch.ops.trtllm.k3_spec_accept(
            logits,
            spec_metadata.draft_tokens.reshape(num_gens, K),
            attn_metadata.draft_kv_cache_block_offsets,
            self._ctx_pool_idx,
            self._ctx_block_divisor,
            self._ctx_block_counts,
            self._ctx_block_tables,
            mgr.prev_num_accepted_tokens,
            attn_metadata.mamba_metadata.state_indices,
            mgr._dummy_request_mask,
            attn_metadata.kv_lens_cuda,
            self._batch_to_slot,
            self._ctx_len,
            self._max_ctx,
            embed.weight,
            self._k3_mask_row(draft_model),
            pool,
            counter,
            force,
            self._compute_block_size,
            *self._k3_shard_args(shard),
        )
        self._on_acceptance(accepted, num_acc, attn_metadata, spec_metadata)
        return (
            accepted,
            num_acc,
            dict(rewind=rewind, bonus=bonus, qpos=qpos, cpos=cpos, noise=noise),
        )

    @staticmethod
    def _k3_shard_args(shard) -> tuple:
        """``trtllm::k3_spec_accept``'s exchange arguments for a vocabulary-sharded step (none otherwise)."""
        if shard is None:
            return ()
        ws, first_column = shard
        return (
            ws["uc"],
            ws["mc"],
            ws["flags"],
            ws["rank"],
            ws["slots"],
            ws["push_copies"],
            first_column,
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
        batch_size = attn_metadata.num_seqs
        num_contexts = attn_metadata.num_contexts
        num_gens = batch_size - num_contexts
        # Set when target_logits kept this step's logits vocabulary-sharded (only k3_spec_accept reads them).
        step_shard, self._k3_step_shard = getattr(self, "_k3_step_shard", None), None
        shard = None
        if step_shard is not None:
            if step_shard[0] is not logits:
                raise RuntimeError(
                    "DFlash: the target logits are not the vocabulary shard target_logits returned"
                )
            shard = step_shard[1]

        raw_logits = logits
        K = spec_metadata.runtime_draft_len

        if K == 0:
            return self.skip_drafting(
                input_ids,
                position_ids,
                hidden_states,
                logits,
                attn_metadata,
                spec_metadata,
                draft_model,
            )

        # Lazy init buffers and attach worker reference for prepare()
        draft_kv_cache_manager = self.get_draft_kv_cache_manager(resource_manager)
        self._lazy_init_ctx_buffers(
            draft_model, spec_metadata, attn_metadata, draft_kv_cache_manager
        )
        spec_metadata._dflash_worker = self
        # A decode step of the supported mode runs the acceptance and the drafter's inputs as one kernel
        # (trtllm::k3_spec_accept), which also decodes the block table below.
        if shard is None:
            fused_accept = self._k3_accept_applies(
                logits, attn_metadata, spec_metadata, draft_model, num_contexts, num_gens
            )
        elif self._k3_accept_step_applies(
            attn_metadata, spec_metadata, draft_model, num_contexts, num_gens
        ):
            fused_accept = True
        else:
            raise RuntimeError(
                "DFlash: the target logits are vocabulary-sharded, but the fused acceptance no longer applies"
            )
        # Before any store: prefill and decode both address pages through it.
        # Returning False here means an empty batch -- the missing-offsets case
        # raises inside, with the metadata type in the message.
        if not fused_accept:
            self._refresh_ctx_block_tables(attn_metadata, batch_size)

        # Save context lengths so both warmup and a failed forward can roll
        # back the in-place _ctx_len updates made during drafting.
        is_warmup = spec_metadata.is_cuda_graph and not torch.cuda.is_current_stream_capturing()
        if not torch.cuda.is_current_stream_capturing():
            # Never allocate the snapshot while capturing a CUDA graph: the
            # clone would live in the graph memory pool, and its replay-time
            # writes could alias blocks reused by later captures. Rollback is
            # only meaningful for eager/warmup forwards anyway; a failure
            # during capture aborts the graph itself, and captured ops do not
            # mutate _ctx_len until replay.
            self._saved_ctx_len = self._ctx_len.clone()
            self._saved_ctx_len_host = list(self._ctx_len_host)
            self._saved_req_ctx_pos = dict(self._req_ctx_pos)
            self._ctx_len_restore_pending = True

        self._execute_guided_decoder_if_present(logits)

        prep = None
        if fused_accept:
            # Saved before the kernel updates kv_lens (warmup also snapshots kv_lens_cuda here).
            self._prepare_attn_metadata_for_dflash(attn_metadata, spec_metadata)
            accepted_tokens, num_accepted_tokens, prep = self._k3_accept(
                logits, attn_metadata, spec_metadata, draft_model, shard
            )
        else:
            accepted_tokens, num_accepted_tokens = self.sample_and_accept_draft_tokens(
                logits, attn_metadata, spec_metadata
            )

        # Opt-in acceptance recording (env-gated; eager-mode measurement
        # runs only). Skipped for CUDA-graph batches (capture/replay/warmup
        # use synthetic requests and forbid the host sync).
        if (
            self._accept_stats is not None
            and num_gens > 0
            and not spec_metadata.is_cuda_graph
            and not torch.cuda.is_current_stream_capturing()
            and spec_metadata.request_ids is not None
        ):
            self._accept_stats.on_accept(
                spec_metadata.request_ids[num_contexts:batch_size],
                num_accepted_tokens[num_contexts:batch_size].tolist(),
            )

        if fused_accept:
            # The kernel recorded the KDA replay acceptance and advanced kv_lens_cuda.
            self._kv_rewind_amount = prep["rewind"]
            self._kv_rewind_nc = num_contexts
            self._kv_rewind_bs = batch_size
            self._kv_rewind_pending = True
            attn_metadata.update_for_spec_dec()
        else:
            # Update GDN/Mamba recurrent states to the accepted token's state.
            if num_gens > 0 and isinstance(attn_metadata.kv_cache_manager, MambaHybridCacheManager):
                attn_metadata.kv_cache_manager.update_mamba_states(
                    attn_metadata=attn_metadata,
                    num_accepted_tokens=num_accepted_tokens,
                    state_indices=attn_metadata.mamba_metadata.state_indices,
                )

            self._prepare_attn_metadata_for_dflash(attn_metadata, spec_metadata)
            self._prepare_kv_for_draft_forward(
                attn_metadata, num_accepted_tokens, num_contexts, batch_size
            )

        # Collapse mrope [3, 1, N] to 1D by taking the first (temporal) dimension.
        # The draft model uses standard 1D RoPE, so only scalar positions are needed.
        if position_ids.ndim == 3:
            position_ids = position_ids[0, 0]
        else:
            position_ids = position_ids.squeeze(0)

        # Get total tokens processed by target model (for hidden state extraction)
        total_target_tokens = input_ids.shape[0]

        # Capture prefill (context) hidden states for future gen steps.
        # This gives the draft model the full prompt context, not just gen tokens.
        if num_contexts > 0:
            self._store_prefill_context(
                draft_model, spec_metadata, attn_metadata, position_ids, total_target_tokens
            )
            # Rebuild batch_to_slot after prefill assigns new slots
            if self._ctx_buf_inited and spec_metadata.request_ids:
                self._rebuild_batch_to_slot(spec_metadata.request_ids)

        inputs = self.prepare_1st_drafter_inputs(
            input_ids=input_ids,
            position_ids=position_ids,
            hidden_states=hidden_states,
            accepted_tokens=accepted_tokens,
            num_accepted_tokens=num_accepted_tokens,
            attn_metadata=attn_metadata,
            spec_metadata=spec_metadata,
            draft_model=draft_model,
            total_target_tokens=total_target_tokens,
            prep=prep,
        )

        if num_gens > 0:
            with self.draft_kv_cache_context(attn_metadata, draft_kv_cache_manager):
                hidden_states_out = draft_model.dflash_forward(
                    noise_embedding=inputs["noise_embedding"],
                    query_positions=inputs["query_positions"],
                    num_ctx_per_req=inputs["num_ctx_per_req"],
                    ctx_k_cache=inputs["ctx_k_cache"],
                    ctx_v_cache=inputs["ctx_v_cache"],
                    ctx_cache_batch_idx=inputs["ctx_cache_batch_idx"],
                    ctx_kv_cache=inputs["ctx_kv_cache"],
                    ctx_page_table=inputs["ctx_page_table"],
                    **dflash_ctx_rows_kwargs(draft_model, inputs["ctx_rows_start"]),
                )

                # Gather K logits per gen request from the block outputs.
                # hidden_states_out is flat: [num_gens * block_size, hidden_dim].
                # Which block slots carry them is a drafter-family convention,
                # resolved through _draft_slot_ids.
                block_size = self._compute_block_size
                gen_hidden_states = self._draft_block_hidden_states(
                    draft_model, hidden_states_out, num_gens, block_size, K
                )
                gen_logits = self._draft_block_logits(
                    draft_model, gen_hidden_states, attn_metadata, spec_metadata
                )

                vocab_size = gen_logits.shape[-1]
                gen_logits = gen_logits.reshape(num_gens, K, vocab_size)

                gen_logits = self._refine_block_logits(
                    draft_model, gen_logits, inputs, spec_metadata
                )

                # DFlash 2: replace the independent per-position picks with one
                # coherent path through the block (absent for plain DFlash).
                if getattr(draft_model, "has_candidate_selector", False):
                    gen_logits = self._apply_dflash2_selector(
                        draft_model,
                        gen_logits,
                        gen_hidden_states.reshape(num_gens, K, -1),
                        inputs["first_prev_tokens"],
                        spec_metadata,
                    )
                    vocab_size = gen_logits.shape[-1]

                gen_draft_tokens = self.sample_draft_tokens(
                    gen_logits,
                    spec_metadata,
                    batch_size,
                    num_contexts=num_contexts,
                )

                # Opt-in confidence-calibration recording (env-gated). The
                # confidence values come from an optional provider on the
                # recorder (None here -> no calibration rows collected; the
                # DSpark confidence-scheduled verification MR supplies the
                # real provider, see accept_stats.py). Guarded off for
                # CUDA-graph batches and d2t vocab-mapped drafters.
                if (
                    self._accept_stats is not None
                    and self._accept_stats.confidence_provider is not None
                    and not spec_metadata.is_cuda_graph
                    and not torch.cuda.is_current_stream_capturing()
                    and spec_metadata.request_ids is not None
                    and self._d2t is None
                ):
                    self._accept_stats.record_draft_confidence(
                        spec_metadata.request_ids[num_contexts:batch_size],
                        draft_model,
                        gen_hidden_states.reshape(num_gens, K, -1),
                        inputs["first_prev_tokens"],
                        gen_draft_tokens,
                    )

        else:
            gen_draft_tokens = torch.empty((0, K), dtype=torch.int32, device="cuda")

        # Context requests are not drafted by the block worker (zero placeholder
        # token); fill their draft-prob slot rows with a one-hot placeholder so
        # they are a legal distribution when they become gen requests next iter.
        gen_vocab = vocab_size if num_gens > 0 else None
        self.write_context_onehot_draft_probs(spec_metadata, num_contexts, num_gens, K, gen_vocab)

        if num_contexts > 0 and num_gens > 0:
            ctx_draft_tokens = torch.zeros((num_contexts, K), dtype=torch.int32, device="cuda")
            next_draft_tokens = torch.cat([ctx_draft_tokens, gen_draft_tokens], dim=0)
        elif num_contexts > 0:
            next_draft_tokens = torch.zeros((num_contexts, K), dtype=torch.int32, device="cuda")
        else:
            next_draft_tokens = gen_draft_tokens

        self._restore_attn_metadata_from_spec_dec(attn_metadata)
        self._apply_kv_rewind_after_draft(attn_metadata, spec_metadata)

        self._rollback_guided_decoder_after_verify(num_accepted_tokens)

        next_new_tokens = self._prepare_next_new_tokens(
            accepted_tokens,
            next_draft_tokens,
            spec_metadata.batch_indices_cuda,
            batch_size,
            num_accepted_tokens,
        )

        # Restore context lengths after warmup; real runs keep the updates.
        if is_warmup:
            self._ctx_len.copy_(self._saved_ctx_len)
            self._restore_ctx_len_host()
        self._ctx_len_restore_pending = False

        outputs = {
            "logits": raw_logits,
            "new_tokens": accepted_tokens,
            "new_tokens_lens": num_accepted_tokens,
            "next_draft_tokens": next_draft_tokens,
            "next_new_tokens": next_new_tokens,
        }
        if shard is not None:
            # "logits" is this rank's vocabulary shard: the engine gathers it for any logits post-processor.
            outputs["logits_vocab_shard"] = True
        return outputs

    def _draft_block_hidden_states(
        self,
        draft_model,
        hidden_states_out: torch.Tensor,
        num_gens: int,
        block_size: int,
        num_draft_tokens: int,
    ) -> torch.Tensor:
        """The block-output rows that produce the K draft logits per gen request (``_draft_slot_ids``).

        The slot ids depend only on the shapes, so they are built once per shape, outside CUDA-graph
        capture, and cached. When they are one contiguous run of rows (a single request, or the
        shift_label convention with a block of K slots, whose requests' slots follow each other) the
        rows are returned as a view, without a gather; otherwise they are gathered.
        """
        rows = hidden_states_out.shape[0]
        key = (num_gens, block_size, num_draft_tokens, rows)
        cache = self.__dict__.setdefault("_draft_block_rows", {})
        entry = cache.get(key)
        if entry is None:
            ids = self._draft_slot_ids(draft_model, num_gens, block_size, num_draft_tokens)
            # Shields only the last request: at block_size == K with
            # shift_label off, slots run 1..K, so every request reads the
            # next one's slot 0 and the last overruns. Degrades, never raises.
            ids = ids.clamp(max=rows - 1)
            if torch.cuda.is_current_stream_capturing():
                return hidden_states_out[ids]
            host = ids.tolist()
            if host and host == list(range(host[0], host[0] + len(host))):
                entry = host[0]
            else:
                entry = ids
            cache[key] = entry
        if isinstance(entry, int):
            return hidden_states_out[entry : entry + num_gens * num_draft_tokens]
        return hidden_states_out[entry]

    def _draft_block_logits(
        self,
        draft_model,
        gen_hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        spec_metadata,
    ) -> torch.Tensor:
        """fp32 logits of the block positions over the full vocabulary (the TP lm_head all-gathers its
        shards). A drafter family whose refinement and draft sampler reduce across ranks themselves may
        return this rank's vocabulary shard instead."""
        return draft_model.logits_processor(
            gen_hidden_states, draft_model.lm_head, attn_metadata, True
        )

    def _ctx_rows_start(self, num_contexts: int, num_gens: int) -> Optional[int]:
        """The first row of the gen requests' page-table rows where those rows are one contiguous run, else None.

        The manager's block table is keyed by batch position, so the gen requests' rows are [num_contexts,
        num_contexts + num_gens): a drafter can view them instead of gathering ``ctx_cache_batch_idx``'s rows. The
        private arena's table is keyed by slot, which need not be contiguous.
        """
        if num_gens > 0 and self._ctx_block_tables is not None:
            return num_contexts
        return None

    def prepare_1st_drafter_inputs(
        self,
        input_ids: torch.LongTensor,
        position_ids: torch.LongTensor,
        hidden_states: torch.Tensor,
        accepted_tokens: torch.Tensor,
        num_accepted_tokens: torch.Tensor,
        attn_metadata: AttentionMetadata,
        spec_metadata: DFlashSpecMetadata,
        draft_model: nn.Module,
        total_target_tokens: int = 0,
        prep: Optional[dict] = None,
    ):
        """Prepare inputs for DFlash's draft forward.

        ``prep``: ``trtllm::k3_spec_accept``'s bonus tokens, positions and noise embedding for a decode step it
        took (see ``_k3_accept``); computed here otherwise.

        For gen requests, builds:
        - noise_embedding: token embeddings for [accepted + mask] tokens
        - query_positions: position IDs for the query tokens
        - num_ctx_per_req: per-request context length in the pool
        - ctx_k_cache / ctx_v_cache / ctx_cache_batch_idx: slot-indexed
          views of the persistent per-layer K/V pool.
        - ctx_rows_start: where ctx_cache_batch_idx is the contiguous rows
          [ctx_rows_start, ctx_rows_start + num_gens) of ctx_page_table, their
          start (``_ctx_rows_start``), else None.
        """
        num_contexts = attn_metadata.num_contexts
        batch_size = attn_metadata.num_seqs
        num_gens = batch_size - num_contexts

        mask_token_id = self._mask_token_id(draft_model)

        # Get the embed_tokens layer from the draft model
        embed_tokens = draft_model.draft_model_full.model.embed_tokens
        hidden_dim = (
            spec_metadata.hidden_size if spec_metadata.hidden_size > 0 else hidden_states.shape[-1]
        )

        if num_gens > 0:
            gen_num_accepted = num_accepted_tokens[num_contexts : num_contexts + num_gens]
            gen_accepted_tokens = accepted_tokens[num_contexts : num_contexts + num_gens, :]

            total_tokens_per_req = self._draft_tokens_per_req  # K+1
            K = spec_metadata.runtime_draft_len

            # Get captured multi-layer hidden states from spec_metadata
            captured_hs = spec_metadata.get_hidden_states(total_target_tokens)
            has_target_features = (
                captured_hs is not None
                and hasattr(draft_model, "fc")
                and hasattr(draft_model, "hidden_norm")
            )

            # Use cached block_size (resolved once on first call)
            block_size = self._compute_block_size
            query_tokens_per_req = block_size

            # Get slots for gen requests from pre-computed mapping
            slots = self._batch_to_slot[num_contexts : num_contexts + num_gens]
            K_plus_1 = K + 1

            if prep is not None:
                gen_rows_out = self._k3_static_arange(num_contexts, num_contexts + num_gens)
                offsets_kp1 = self._k3_static_arange(0, K_plus_1)
                bonus = prep["bonus"]
                query_position_ids = prep["qpos"]
                ctx_position_ids = prep["cpos"]
                noise_embed_2d = prep["noise"]
            else:
                gen_rows_out = torch.arange(
                    num_contexts, num_contexts + num_gens, dtype=torch.long, device="cuda"
                )

                bonus_idx = (gen_num_accepted - 1).clamp_min(0).long().unsqueeze(1)
                bonus = gen_accepted_tokens.gather(1, bonus_idx).squeeze(1).long()

                ctx_len_gen = self._ctx_len[slots]
                j_block = torch.arange(query_tokens_per_req, dtype=torch.long, device="cuda")
                offsets_kp1 = torch.arange(K_plus_1, dtype=torch.long, device="cuda")

                # _ctx_len is clamped to _max_ctx only AFTER this step's accepted
                # tokens are folded in (see the update below), so the running length
                # used here has to be clamped on its own -- otherwise a request that
                # already sits at the ceiling indexes num_accepted positions past
                # any position the sequence can legitimately reach.
                ctx_len_now = (ctx_len_gen + gen_num_accepted.long()).clamp_(max=self._max_ctx)
                query_position_ids = ctx_len_now.unsqueeze(1) + j_block.unsqueeze(0)
                ctx_position_ids = ctx_len_gen.unsqueeze(1) + offsets_kp1.unsqueeze(0)

                noise_embed_2d = dflash_noise_block_embedding(
                    embed_tokens,
                    bonus,
                    mask_token_id,
                    query_tokens_per_req,
                    self._trained_mask_embedding(draft_model, mask_token_id),
                )

            # Accumulate new accepted features into context buffers
            fused_ctx_kv = False
            if has_target_features:
                gen_start = attn_metadata.num_ctx_tokens
                # Target now processes exactly K+1 tokens per gen req, so the
                # captured slice is already the full set we need to project.
                gen_hs = captured_hs[gen_start : gen_start + num_gens * total_tokens_per_req]
                gen_hs_to_project = gen_hs.reshape(-1, gen_hs.shape[-1])
                projected_to_store = draft_model.project_target_hidden(gen_hs_to_project)
                fused_ctx_kv = prep is not None and self._k3_ctx_kv_applies(
                    draft_model, projected_to_store
                )
                if fused_ctx_kv:
                    num_ctx_fused = self._k3_ctx_kv(
                        draft_model,
                        projected_to_store,
                        ctx_position_ids,
                        gen_num_accepted,
                        slots,
                        gen_rows_out,
                    )
                    # k3_ctx_kv adds the accepted tokens in the kernel; empty the dummy slot after
                    # it, as _advance_ctx_len does after the add on the torch path.
                    self._ctx_len[self._dummy_slot].zero_()
            if has_target_features and not fused_ctx_kv:
                gen_num_accepted_long = gen_num_accepted.long()
                col_idx = self._ctx_len[slots].unsqueeze(1) + offsets_kp1.unsqueeze(0)
                write_mask = offsets_kp1.unsqueeze(0) < gen_num_accepted_long.unsqueeze(1)
                if self._ctx_block_tables is not None:
                    # Bound by what the manager handed THIS request: the masked
                    # entries below are still written (fixed-size, for graph
                    # safety) and a placeholder page resolves to another's.
                    gen_capacity = self._ctx_block_counts[gen_rows_out] * self._ctx_page_size
                    col_idx = torch.minimum(col_idx, (gen_capacity - 1).unsqueeze(1)).clamp_(min=0)
                else:
                    col_idx = col_idx.clamp(max=self._max_ctx - 1)

                # Fixed-size writes for CUDA graph compatibility:
                # Write ALL entries but zero out invalid ones. Invalid
                # writes land at clamped column indices beyond valid
                # _ctx_len range, so they're harmless.
                slot_flat = slots.unsqueeze(1).expand(-1, K + 1).reshape(-1)
                col_flat = col_idx.reshape(-1)
                proj_flat = projected_to_store  # already [num_gens*(K+1), proj_dim]
                pos_flat = ctx_position_ids.long().reshape(-1)
                mask_1d = write_mask.reshape(-1)

                # Fast path: store the pre-projected/pre-RoPE'd K/V.
                # dflash_forward reads these directly via cache_batch_idx.
                cache_dtype = (
                    self._ctx_kv_buf[0].dtype if self._ctx_paged else self._ctx_k_buf.dtype
                )
                k_new, v_new = draft_model.precompute_context_kv(
                    proj_flat.to(cache_dtype), pos_flat
                )
                mask_bc = mask_1d.view(-1, 1, 1, 1).to(k_new.dtype)
                k_new.mul_(mask_bc)
                if v_new is not None:
                    v_new.mul_(mask_bc)
                slot_long = slot_flat.long()
                col_long = col_flat.long()
                if self._ctx_paged:
                    if self._ctx_block_tables is not None:
                        # Batch positions of the gen requests, matching the
                        # per-request block table's row order.
                        rows_long = gen_rows_out.unsqueeze(1).expand(-1, K + 1).reshape(-1)
                    else:
                        rows_long = slot_long
                    self._store_context_kv_paged(k_new, v_new, rows_long, col_long)
                else:  # VANILLA DFlash backend (FlashAttention)
                    self._ctx_k_buf[slot_long, :, col_long] = k_new
                    if v_new is not None:
                        self._ctx_v_buf[slot_long, :, col_long] = v_new

                self._advance_ctx_len(slots, gen_num_accepted_long)

            if fused_ctx_kv:
                num_ctx_per_req_t = num_ctx_fused
            else:
                num_ctx_per_req_t = self._ctx_len[slots]
            if not fused_ctx_kv and self._ctx_block_tables is not None:
                # The write above clamps columns to the request's allocation, so
                # a context that outruns it has its tail written over the last
                # valid slot. Truncate what is advertised to match, or the read
                # attends that clobbered value -- or another request's page.
                allocated = dflash_allocated_ctx_limit(
                    self._ctx_block_counts, self._ctx_page_size, block_size
                )
                num_ctx_per_req_t = torch.minimum(num_ctx_per_req_t, allocated[gen_rows_out])
            noise_embedding = noise_embed_2d
            query_positions = query_position_ids.long()

            # Update seq_lens for gen requests to K+1 (the number of tokens
            # the target forward actually processed).
            attn_metadata._seq_lens_cuda[num_contexts : num_contexts + num_gens] = (
                total_tokens_per_req
            )
            attn_metadata._seq_lens[num_contexts : num_contexts + num_gens] = total_tokens_per_req
        else:
            noise_embedding = hidden_states.new_empty(0, 0, hidden_dim)
            query_positions = torch.empty(0, 0, dtype=torch.long, device="cuda")
            num_ctx_per_req_t = torch.empty(0, dtype=torch.long, device="cuda")
            slots = torch.empty(0, dtype=torch.long, device="cuda")
            gen_rows_out = slots
            bonus = torch.empty(0, dtype=torch.long, device="cuda")

        return {
            "noise_embedding": noise_embedding,
            "query_positions": query_positions,
            "num_ctx_per_req": num_ctx_per_req_t,
            "ctx_k_cache": self._ctx_k_buf,
            "ctx_v_cache": self._ctx_v_buf,
            # Slots index the private arena; the manager's block table is keyed
            # by batch position, so the drafter reads its own rows there.
            "ctx_cache_batch_idx": gen_rows_out if self._ctx_block_tables is not None else slots,
            # Anchor token per gen request (block slot 0): last accepted
            # token. The dspark Markov chain conditions its first step on it.
            "first_prev_tokens": bonus,
            "ctx_kv_cache": self._ctx_kv_buf,
            "ctx_page_table": (
                self._ctx_block_tables
                if self._ctx_block_tables is not None
                else self._ctx_page_table
            ),
            "ctx_rows_start": self._ctx_rows_start(num_contexts, num_gens),
        }


class KimiK3DSparkWorker(KimiK3DFlashWorker, DSparkWorker):
    # Whether this step's block logits are the Kimi K3 decode path's vocabulary shard (see ``_draft_block_logits``).
    _k3_sharded_block_logits = False

    def _apply_dspark_markov_bias(
        self,
        draft_model,
        gen_logits: torch.Tensor,
        first_prev_tokens: torch.Tensor,
        spec_metadata,
    ) -> torch.Tensor:
        """Apply the dspark vanilla-Markov intra-block bias to block logits.

        Reference (DeepSpec VanillaMarkov.sample_block_tokens, temperature 0):
        step i adds bias = markov_w2 @ markov_w1[prev_i] to the shared-lm_head
        logits, where prev_0 is the anchor (last accepted) token and prev_{i>0}
        is the greedy token from step i-1's biased logits. Greedy per-position
        argmax of the returned logits therefore reproduces the reference
        sampled chain; the rejection-sampling path samples from the same
        biased distributions (proposal conditioned on the greedy chain).

        Block logits the Kimi K3 decode path kept vocab-sharded (see ``_draft_block_logits``) run the
        whole chain as ``trtllm::k3_markov``, which also returns the greedy draft tokens and
        next_new_tokens. Other TP-sharded logits slice markov_w2's rows to this rank's contiguous
        shard and chain through the TP-aware global argmax.
        """
        # The d2t guard lives in set_draft_model: it is model-static, so raising
        # it here would surface a load-time config error per decode step.
        # Unlike the d2t guard this one cannot move to set_draft_model: it
        # keys on the runtime logits width, and reproducing that at init would
        # duplicate the draft head's sharding rules. A standalone drafter
        # borrows the target lm_head, whose gather_output defaults to True, so
        # the logits normally arrive full-vocab and this branch is skipped.
        self._k3_markov = None
        k3_sharded, self._k3_sharded_block_logits = self._k3_sharded_block_logits, False
        full_vocab = draft_model.markov_w2.shape[0]
        shard = gen_logits.shape[-1]
        vocab_slice = None
        if shard != full_vocab:
            mapping = self.mapping
            if (
                mapping is None
                or getattr(mapping, "enable_attention_dp", False)
                or shard * mapping.tp_size != full_vocab
            ):
                raise NotImplementedError(
                    f"DSpark Markov head: draft logits width {shard} does not "
                    f"match the drafter vocab {full_vocab} and is not a plain "
                    "TP column shard of it."
                )
            vocab_slice = slice(mapping.tp_rank * shard, (mapping.tp_rank + 1) * shard)
            if k3_sharded:
                return self._k3_markov_chain(
                    draft_model, gen_logits, first_prev_tokens, vocab_slice
                )

        def argmax_fn(step_logits):
            # Full-vocab token ids (TP-aware when sharded); tokens stay in
            # draft-vocab space, which is what markov_w1 indexes.
            return self.greedy_sample_draft_with_tp_gather(step_logits, spec_metadata).long()

        return draft_model.apply_markov_chain_logits(
            gen_logits,
            first_prev_tokens,
            argmax_fn=argmax_fn,
            vocab_slice=vocab_slice,
        )

    def _keep_draft_logits_sharded(self, draft_model, spec_metadata, num_gens: int) -> bool:
        """Keep the draft logits vocab-sharded: ``trtllm::k3_markov`` reduces across the ranks itself.

        The chain's global argmaxes are the greedy draft tokens, so the lm_head's all-gather of the
        full logits and the draft sampler's gather are both skipped. Needs a Kimi K3 target running its
        decode kernels (``k3_decode``), plain TP, greedy drafting (rejection sampling reads
        full-vocabulary probabilities), an unquantized bias-free column-parallel bf16 lm_head whose
        shards tile the Markov vocabulary, bf16 Markov weights of the kernel's rank, MNNVL (the model's
        all-reduces own an MNNVL workspace for this mapping), and a block, shard and batch of
        ``num_gens`` requests the kernel can split.
        """
        if not self.k3_decode:
            return False
        from tensorrt_llm._torch.cute_dsl_kernels.k3_markov import op as k3_markov_op
        from tensorrt_llm._torch.distributed.ops import MNNVLAllReduce
        from tensorrt_llm._torch.modules.linear import TensorParallelMode

        mapping = self.mapping
        lm_head = getattr(draft_model, "lm_head", None)
        if (
            lm_head is None
            or not getattr(draft_model, "has_markov_head", False)
            or mapping is None
            or mapping.tp_size <= 1
            or mapping.enable_attention_dp
            or spec_metadata.wants_advanced_draft_sampling
            or getattr(lm_head, "tp_mode", None) != TensorParallelMode.COLUMN
            or not getattr(lm_head, "gather_output", False)
            or getattr(lm_head, "bias", None) is not None
            or lm_head.weight.dtype != torch.bfloat16
            or draft_model.markov_w1.dtype != torch.bfloat16
            or draft_model.markov_w2.dtype != torch.bfloat16
            or lm_head.weight.shape[0] * mapping.tp_size != draft_model.markov_w2.shape[0]
            or MNNVLAllReduce.allreduce_mnnvl_workspaces.get(mapping) is None
        ):
            return False
        shard = lm_head.weight.shape[0]
        block = spec_metadata.runtime_draft_len
        rank = k3_markov_op._kernel_module().MARKOV_RANK
        if (
            not 0 < block <= k3_markov_op.WORKSPACE_MAX_BLOCK
            or draft_model.markov_w1.dim() != 2
            or draft_model.markov_w1.shape[1] != rank
            or tuple(draft_model.markov_w2.shape[1:]) != (rank,)
        ):
            return False
        if k3_markov_op.pick_grid(shard, block, num_gens) == 0:
            logger.warning_once(
                f"DSpark Markov head: trtllm::k3_markov cannot split a {shard}-row vocab shard for "
                f"{num_gens} requests; the draft logits are all-gathered and the chain runs unfused.",
                key=f"dspark_k3_markov_shard_{num_gens}",
            )
            return False
        return True

    def _draft_block_logits(
        self,
        draft_model,
        gen_hidden_states: torch.Tensor,
        attn_metadata,
        spec_metadata,
    ) -> torch.Tensor:
        """This rank's bf16 shard of the block logits when ``k3_markov`` takes them (it converts them to fp32
        exactly, as ``.float()`` would), from the drafter's logits processor's head kernel where it takes the rows,
        else the head's own GEMM (``_k3_head_shard``); otherwise the base class' fp32 logits."""
        num_gens = gen_hidden_states.shape[0] // max(spec_metadata.runtime_draft_len, 1)
        self._k3_sharded_block_logits = self._keep_draft_logits_sharded(
            draft_model, spec_metadata, num_gens
        )
        if self._k3_sharded_block_logits:
            return self._k3_head_shard(
                draft_model.logits_processor, draft_model.lm_head, gen_hidden_states
            )
        return super()._draft_block_logits(
            draft_model, gen_hidden_states, attn_metadata, spec_metadata
        )

    def sample_and_accept_draft_tokens(self, logits, attn_metadata, spec_metadata):
        """The base acceptance; under ``k3_decode`` its outputs are kept for ``k3_markov``'s next_new_tokens."""
        accepted_tokens, num_accepted_tokens = super().sample_and_accept_draft_tokens(
            logits, attn_metadata, spec_metadata
        )
        if self.k3_decode:
            self._on_acceptance(accepted_tokens, num_accepted_tokens, attn_metadata, spec_metadata)
        return accepted_tokens, num_accepted_tokens

    def _on_acceptance(
        self, accepted_tokens, num_accepted_tokens, attn_metadata, spec_metadata
    ) -> None:
        """Keeps this step's acceptance for ``k3_markov``'s next_new_tokens (and the metadata whose KV lengths it
        rewinds)."""
        self._k3_acceptance = (
            accepted_tokens, num_accepted_tokens, attn_metadata.num_contexts, spec_metadata, attn_metadata
        )  # fmt: skip

    def _k3_markov_chain(
        self,
        draft_model,
        gen_logits: torch.Tensor,
        first_prev_tokens: torch.Tensor,
        vocab_slice: slice,
    ) -> torch.Tensor:
        """The Markov chain as ``trtllm::k3_markov`` on this rank's shard of the block logits.

        Returns the corrected logits (fp32); keeps the kernel's greedy tokens for
        ``sample_draft_tokens`` and its next_new_tokens (built from this step's acceptance) for
        ``_prepare_next_new_tokens``.
        """
        from tensorrt_llm._torch.cute_dsl_kernels.k3_markov import op as k3_markov_op

        acceptance = getattr(self, "_k3_acceptance", None)
        if acceptance is None:
            raise RuntimeError("DSpark Markov head: k3_markov needs this step's acceptance first")
        accepted_tokens, num_accepted_tokens, num_contexts, spec_metadata, attn_metadata = (
            acceptance
        )
        num_gens = gen_logits.shape[0]
        # The draft forward's pending KV-length rewind (see _apply_kv_rewind_after_draft) runs in the kernel: every
        # reader of kv_lens_cuda in the draft forward precedes it. In warmup the restore of kv_lens_cuda that
        # follows overwrites it, as it would the Python rewind.
        rewind = getattr(self, "_kv_rewind_amount", None)
        fold = (
            getattr(self, "_kv_rewind_pending", False)
            and rewind is not None
            and getattr(attn_metadata, "kv_lens_cuda", None) is not None
            and rewind.dtype == torch.int32
            and self._kv_rewind_bs - self._kv_rewind_nc == rewind.numel() <= num_gens
        )
        corrected, tokens, next_new = k3_markov_op.markov_chain(
            self.mapping,
            gen_logits.contiguous(),
            first_prev_tokens.long(),
            draft_model.markov_w1,
            draft_model.markov_w2[vocab_slice],
            vocab_slice.start,
            accepted_tokens,
            num_accepted_tokens[num_contexts : num_contexts + num_gens],
            spec_metadata.batch_indices_cuda[num_contexts : num_contexts + num_gens],
            kv_lens=attn_metadata.kv_lens_cuda if fold else None,
            rewind=rewind if fold else None,
            rewind_first=self._kv_rewind_nc if fold else 0,
        )
        if fold:
            self._kv_rewind_amount = None
            self._kv_rewind_pending = False
        self._k3_markov = (corrected, tokens, next_new, accepted_tokens, num_accepted_tokens)
        return corrected

    def sample_draft_tokens(
        self,
        logits,
        spec_metadata,
        batch_size,
        *,
        num_contexts=0,
        draft_step=None,
        mapping_lm_head_tp=None,
    ):
        """Greedy block drafts straight from ``k3_markov``.

        When ``logits`` are the corrected logits the kernel just returned, its per-position global
        argmax (first maximum, lowest vocabulary index among equal values) is the token the
        TP-gathered greedy sampler would pick. Anything else goes to the base sampler.
        """
        chain = getattr(self, "_k3_markov", None)
        if (
            chain is not None
            and chain[0] is logits
            and mapping_lm_head_tp is None
            and not spec_metadata.wants_advanced_draft_sampling
        ):
            self._k3_markov_next = (chain[1], chain[2], chain[3], chain[4])
            return chain[1]
        self._k3_markov_next = None
        return super().sample_draft_tokens(
            logits,
            spec_metadata,
            batch_size,
            num_contexts=num_contexts,
            draft_step=draft_step,
            mapping_lm_head_tp=mapping_lm_head_tp,
        )

    def _prepare_next_new_tokens(
        self,
        accepted_tokens,
        next_draft_tokens,
        batch_indices_cuda,
        batch_size,
        num_accepted_tokens,
    ):
        """``k3_markov``'s next_new_tokens when the drafts are its tokens for the whole batch (no context
        requests); otherwise the base assembly."""
        chain = getattr(self, "_k3_markov_next", None)
        # The step ends here: release its acceptance (which holds the step's attention and spec metadata) and the
        # kernel's outputs, so they do not keep a captured graph's pool or metadata alive after the step.
        self._k3_markov_next = None
        self._k3_acceptance = None
        self._k3_markov = None
        if (
            chain is not None
            and chain[0] is next_draft_tokens
            and chain[2] is accepted_tokens
            and chain[3] is num_accepted_tokens
            and next_draft_tokens.shape[0] == batch_size
        ):
            return chain[1]
        return super()._prepare_next_new_tokens(
            accepted_tokens, next_draft_tokens, batch_indices_cuda, batch_size, num_accepted_tokens
        )
