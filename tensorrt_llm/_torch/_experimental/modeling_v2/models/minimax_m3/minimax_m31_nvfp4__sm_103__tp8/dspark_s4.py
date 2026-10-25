# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MiniMax-M3.1 TP8 implementation."""

from __future__ import annotations

import atexit
import os
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from tensorrt_llm._torch._experimental.modeling_v2.catalog.spec_decode import (
    dspark_markov_bias as markov_bias,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.spec_decode import (
    dspark_markov_chain as chain,
)
from tensorrt_llm._torch.models.modeling_speculative import dspark_markov_step_bias
from tensorrt_llm.logger import logger

from .m31_switches import m31_disabled

MARKOV_RANK = 256
MAX_FAST_BATCH = 8
DISABLE_ENV = "TRTLLM_M31_DRAFT_DISABLE_S4"
S4_COUNT_LOG_EVERY = 1000
MARKOV_LL = not m31_disabled("MARKOV_LL")
TAILBK_SPEC = not m31_disabled("TAILBK_SPEC")
_DISABLED = os.environ.get(DISABLE_ENV, "").strip() not in ("", "0")


def disabled() -> bool:
    return _DISABLED


class _S4State:
    def __init__(
        self,
        mapping,
        rows: int,
        shard: int,
        offset: int,
        w1: torch.Tensor,
        w2s: torch.Tensor,
        block: int,
        ws: dict,
    ) -> None:
        self.mapping = mapping
        self.ws = ws
        self.rows = rows
        self.shard = shard
        self.offset = offset
        self.w1 = w1
        self.w2s = w2s
        self.block = block
        self.bufs: Dict[Tuple[int, int], torch.Tensor] = {}

    def buffer(self, batch: int) -> torch.Tensor:
        return self.bufs[batch, self.block]


S4_LOGITS_BATCHED = True


def _biased_shard_logits(
    shard_logits: torch.Tensor,
    first_prev: torch.Tensor,
    tokens: torch.Tensor,
    w1: torch.Tensor,
    w2_shard: torch.Tensor,
    fp32: bool = True,
) -> torch.Tensor:
    """The in-tree chain's biased block logits on this rank's vocab shard, given the chain's tokens: step k adds
    bf16(w2 @ w1[prev_k]) with prev_0 = the anchor and prev_k = token k-1 (dspark_markov_step_bias). With ``fp32``
    (the in-tree arithmetic: LogitsProcessor hands the chain fp32 logits) the bf16 shard logits and the bias are
    widened and added in fp32; without it, one bf16 add."""
    if (
        fp32
        and S4_LOGITS_BATCHED
        and TAILBK_SPEC
        and markov_bias.supports(shard_logits, first_prev, tokens, w1)
    ):
        B, K, N = shard_logits.shape
        bias = F.linear(markov_bias.prev_embeddings(first_prev, tokens, w1), w2_shard)
        return markov_bias.widen_add(shard_logits, bias.view(B, K, N))
    if fp32:
        shard_logits = shard_logits.float()
    B, K, N = shard_logits.shape
    prevs = torch.cat([first_prev.view(B, 1), tokens[:, : K - 1].long()], dim=1)
    if S4_LOGITS_BATCHED:
        bias = dspark_markov_step_bias(prevs.reshape(-1), w1, w2_shard).view(B, K, N)
        return shard_logits + bias.to(shard_logits.dtype)
    out = torch.empty_like(shard_logits)
    for k in range(K):
        out[:, k] = shard_logits[:, k] + dspark_markov_step_bias(prevs[:, k], w1, w2_shard).to(
            shard_logits.dtype
        )
    return out


class _ShardedDraftLogits(nn.Module):
    """The step's draft logits processor: this rank's vocab shard of ``lm_head`` over the gathered block rows,
    copied into the padded buffer; any other call goes to the drafter's own processor."""

    def __init__(self, inner: nn.Module, state: _S4State, batch: int) -> None:
        super().__init__()
        self.inner = inner
        self._state = [state]
        self.batch = batch

    def forward(self, hidden_states, lm_head, attn_metadata, return_context_logits: bool = False):
        state = self._state[0]
        if hidden_states.dim() != 2 or hidden_states.shape[0] != self.batch * state.block:
            return self.inner(hidden_states, lm_head, attn_metadata, return_context_logits)
        logits = lm_head.apply_linear(hidden_states, None)
        buf = state.buffer(self.batch)
        buf[:, :, : state.shard].copy_(logits.view(self.batch, state.block, state.shard))
        return logits


def _head_is_sharded(lm_head: nn.Module, vocab: int, tp: int) -> bool:
    from tensorrt_llm._torch.modules.linear import TensorParallelMode

    return (
        tp > 1
        and getattr(lm_head, "tp_mode", None) == TensorParallelMode.COLUMN
        and (getattr(lm_head, "padding_size", 0) == 0)
        and (getattr(lm_head, "bias", None) is None)
        and (lm_head.out_features * tp == vocab)
        and (lm_head.weight.dtype == torch.bfloat16)
    )


class S4WorkerMixin:
    """Mixed into the target's ``DSparkWorker`` subclass (before it in the bases)."""

    _s4: Optional[_S4State] = None
    _s4_step: Optional[_S4State] = None
    _s4_tokens: Optional[torch.Tensor] = None
    _s4_step_meta: Optional[Tuple[int, int, object]] = None
    _s4_draft: Optional[nn.Module] = None
    _s4_served: int = 0
    _s4_gen_steps: int = 0
    _s4_ll_logged: bool = False
    _s4_step_attn = None
    _tailbk_rewound: bool = False
    _tailbk_h_logged: bool = False
    _tailbk_s4l_logged: bool = False
    _tailbk_c_logged: bool = False

    def prebuild_s4(self, lm_head: nn.Module, draft_model) -> bool:
        """Set up the one-launch chain for this worker; False (and nothing set up) when it does not apply."""
        self._s4 = None
        if disabled():
            logger.info(f"M3.1 DSpark S4: off ({DISABLE_ENV} set); the in-tree Markov chain runs")
            return False
        if not getattr(draft_model, "has_markov_head", False):
            return False
        w1 = draft_model.markov_w1
        w2 = draft_model.markov_w2
        mapping = self.mapping
        tp = mapping.tp_size
        vocab = w1.shape[0]
        if (
            w1.dtype != torch.bfloat16
            or w2.dtype != torch.bfloat16
            or w1.shape[1] != MARKOV_RANK
            or (tuple(w2.shape) != tuple(w1.shape))
            or mapping.enable_attention_dp
            or (not _head_is_sharded(lm_head, vocab, tp))
        ):
            return False
        block = int(self.max_draft_len)
        shard = vocab // tp
        rows = chain.padded_rows(shard, MAX_FAST_BATCH, w1.device)
        if not self._s4_ranks_agree(rows != 0 and block <= chain.MAX_BLOCK):
            return False
        offset = mapping.tp_rank * shard
        w2s = chain.pad_w2_shard(w2[offset : offset + shard].contiguous(), rows)
        state = _S4State(
            mapping, rows, shard, offset, w1, w2s, block, chain.workspace_args(mapping)
        )
        for batch in range(1, MAX_FAST_BATCH + 1):
            state.bufs[batch, block] = chain.pad_logits_buffer(batch, block, rows, w1.device)
        s4b8 = chain.enable_batch1_variant(not m31_disabled("S4B8"))
        markov_ll = chain.enable_ll_exchange(MARKOV_LL)
        tailbk_c = chain.enable_cached_inputs(TAILBK_SPEC)
        self._s4_fp32 = chain.enable_fp32_sum(not m31_disabled("S4_FP32SUM"))
        self._s4_logits_on = not m31_disabled("S4_LOGITS")
        chain.precompile(mapping, rows, w1, w2s, offset, blocks=(block,), max_batch=MAX_FAST_BATCH)
        self._s4 = state
        self._s4_draft = draft_model
        atexit.register(self._s4_log_count)
        logger.info(
            (
                "M3.1 DSpark S4: one-launch Markov chain set up on rank "
                f"{mapping.tp_rank}"
                "/"
                f"{tp}"
                " (shard "
                f"{shard}"
                " rows padded to "
                f"{rows}"
                ", K="
                f"{block}"
                ", batches 1.."
                f"{MAX_FAST_BATCH}"
                "; S4B8 batch-1 kernel "
                f"{('on' if s4b8 else 'off')}"
                "; MARKOV_LL "
                f"{('on' if markov_ll and s4b8 else 'off')}"
                "; TAILBK_SPEC cached inputs "
                f"{('on' if tailbk_c else 'off')}"
                "; S4-logits "
                f"{('on' if self._s4_logits_on else 'off')}"
                "; fp32 sum "
                f"{('on' if self._s4_fp32 else 'off')}"
                ")"
            )
        )
        return True

    def _s4_ranks_agree(self, local: bool) -> bool:
        """``local`` if every rank of the TP group decided the same; raises when they differ (a rank going on into
        the collective workspace alone would hang its peers)."""
        from tensorrt_llm._torch.distributed.ops import (
            _get_mnnvl_workspace_comm,
            _mnnvl_workspace_all_succeeded,
        )

        comm = _get_mnnvl_workspace_comm(self.mapping)
        all_on = _mnnvl_workspace_all_succeeded(comm, local)
        all_off = _mnnvl_workspace_all_succeeded(comm, not local)
        if not (all_on or all_off):
            raise RuntimeError(
                "m31 DSpark S4: the TP ranks disagree on setting up the one-launch Markov chain"
            )
        return all_on

    def _s4_serves(self, num_gens: int, spec_metadata, draft_model) -> bool:
        return self._s4_mode(num_gens, spec_metadata, draft_model) is not None

    def _s4_mode(self, num_gens: int, spec_metadata, draft_model):
        """ "tokens" (greedy drafting: the chain's tokens are the draft), "logits" (advanced draft sampling with
        S4-logits on: the chain's corrected logits go to the stock sampler), or None (the in-tree chain runs)."""
        if not self._s4_common(num_gens, draft_model):
            return None
        if not getattr(spec_metadata, "wants_advanced_draft_sampling", False):
            return "tokens"
        return "logits" if getattr(self, "_s4_logits_on", False) else None

    def _s4_common(self, num_gens: int, draft_model) -> bool:
        return (
            self._s4 is not None
            and 1 <= num_gens <= MAX_FAST_BATCH
            and (getattr(self, "guided_decoder", None) is None)
            and (getattr(self, "_d2t", None) is None)
            and (getattr(draft_model, "lilicorr", None) is None)
            and (not getattr(draft_model, "has_candidate_selector", False))
        )

    def _host_step_hook(self, spec_metadata) -> None:
        """The engagement counter (host only, no CUDA): called by the spec metadata's prepare() on every iteration,
        outside any captured graph, it evaluates the same eligibility the captured step was routed by and counts
        the generation steps the chain serves. One line per rank every S4_COUNT_LOG_EVERY generation steps and at
        exit: "M3.1 DSpark S4: served <x>/<y> generation steps". The counts start at server start, so they include
        warmup and padding steps: read engagement as the difference of two lines logged after the server is ready."""
        if self._s4 is None:
            return
        num_gens = int(getattr(spec_metadata, "num_generations", 0) or 0)
        if num_gens <= 0:
            return
        self._s4_gen_steps += 1
        self._s4_served += int(self._s4_serves(num_gens, spec_metadata, self._s4_draft))
        if self._s4_gen_steps % S4_COUNT_LOG_EVERY == 0:
            self._s4_log_count()

    def _s4_log_count(self) -> None:
        if self._s4 is not None and self._s4_gen_steps:
            logger.info(
                (
                    "M3.1 DSpark S4: served "
                    f"{self._s4_served}"
                    "/"
                    f"{self._s4_gen_steps}"
                    " generation steps on rank "
                    f"{self.mapping.tp_rank}"
                )
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
        num_gens = attn_metadata.num_seqs - attn_metadata.num_contexts
        mode = self._s4_mode(num_gens, spec_metadata, draft_model)
        served = mode is not None
        if not served:
            return super()._forward_impl(
                input_ids,
                position_ids,
                hidden_states,
                logits,
                attn_metadata,
                spec_metadata,
                draft_model,
                resource_manager,
            )
        inner = draft_model.logits_processor
        draft_model.logits_processor = _ShardedDraftLogits(inner, self._s4, num_gens)
        self._s4_step = self._s4
        self._s4_step_mode = mode
        self._s4_step_meta = (attn_metadata.num_contexts, attn_metadata.num_seqs, spec_metadata)
        self._s4_step_attn = attn_metadata
        try:
            return super()._forward_impl(
                input_ids,
                position_ids,
                hidden_states,
                logits,
                attn_metadata,
                spec_metadata,
                draft_model,
                resource_manager,
            )
        finally:
            draft_model.logits_processor = inner
            self._s4_step = None
            self._s4_step_mode = None
            self._s4_tokens = None
            self._s4_step_meta = None
            self._s4_step_attn = None
            self._tailbk_rewound = False

    def _refine_block_logits(self, draft_model, gen_logits, inputs, spec_metadata):
        state = self._s4_step
        if state is not None and gen_logits.dim() == 3 and (gen_logits.shape[-1] == state.shard):
            base = state.buffer(gen_logits.shape[0])
            if (
                chain.supports(base, state.w1, state.w2s)
                and getattr(self, "_s4_step_mode", None) == "logits"
            ):
                first_prev = inputs["first_prev_tokens"].long()
                tokens = self._s4_chain(state, base, first_prev, spec_metadata)
                self._s4_log_ll(base.shape[0])
                biased = _biased_shard_logits(
                    gen_logits,
                    first_prev,
                    tokens,
                    state.w1,
                    state.w2s[: state.shard],
                    self._s4_fp32,
                )
                self._s4l_log(gen_logits, first_prev, tokens, state.w1)
                return biased
            if chain.supports(base, state.w1, state.w2s):
                tokens = self._s4_chain(
                    state, base, inputs["first_prev_tokens"].long(), spec_metadata
                )
                self._s4_log_ll(base.shape[0])
                self._s4_tokens = tokens
                return gen_logits
        return super()._refine_block_logits(draft_model, gen_logits, inputs, spec_metadata)

    def _s4_chain(
        self, state: _S4State, base: torch.Tensor, first_prev: torch.Tensor, spec_metadata
    ):
        """The chain launch; under TAILBK_SPEC (site H) it also applies the step's deferred KV-length rewind when
        the stock _apply_kv_rewind_after_draft would (a real step: not the eager warmup of a graph batch, whose
        kv_lens is restored wholesale), and marks it done for that method."""
        rewind = self._tailbk_rewind_args(base.shape[0], spec_metadata)
        if rewind is None:
            tokens = chain.dspark_markov_chain(
                base, first_prev, state.w1, state.w2s, state.offset, **state.ws
            )
            self._tailbk_c_log()
            return tokens
        kv_lens, amount, first = rewind
        tokens = chain.dspark_markov_chain_rewind(
            base, first_prev, state.w1, state.w2s, state.offset, kv_lens, amount, first, **state.ws
        )
        self._tailbk_rewound = True
        self._tailbk_c_log()
        if not self._tailbk_h_logged:
            self._tailbk_h_logged = True
            logger.info(
                (
                    "[M3 dispatch] TAILBK_SPEC H: KV-length rewind in the Markov chai"
                    "n launch on rank "
                    f"{self.mapping.tp_rank}"
                )
            )
        return tokens

    def _tailbk_c_log(self) -> None:
        """The TAILBK_SPEC C engagement line: once per rank, after the first chain call that took its accepted inputs
        from the precompile cache (host only; a captured step logs it while it is captured)."""
        if self._tailbk_c_logged or not getattr(chain, "cached_inputs_taken", lambda: False)():
            return
        self._tailbk_c_logged = True
        logger.info(
            (
                "[M3 dispatch] TAILBK_SPEC C: chain accepted inputs from precompi"
                "le cache (taken) on rank "
                f"{self.mapping.tp_rank}"
            )
        )

    def _tailbk_rewind_args(self, batch: int, spec_metadata):
        """(kv_lens, rewind amounts, first row) for the chain to apply, or None (the stock rewind runs)."""
        if not TAILBK_SPEC or not getattr(self, "_kv_rewind_pending", False):
            return None
        amount = getattr(self, "_kv_rewind_amount", None)
        attn = self._s4_step_attn
        if amount is None or attn is None or (not hasattr(attn, "kv_lens_cuda")):
            return None
        if spec_metadata.is_cuda_graph and (not torch.cuda.is_current_stream_capturing()):
            return None
        nc, bs = (self._kv_rewind_nc, self._kv_rewind_bs)
        if bs - nc != batch or amount.numel() != batch:
            return None
        kv_lens = attn.kv_lens_cuda
        if not chain.supports_rewind(kv_lens, amount, nc):
            return None
        return (kv_lens, amount, nc)

    def _apply_kv_rewind_after_draft(self, attn_metadata, spec_metadata):
        if self._tailbk_rewound:
            self._tailbk_rewound = False
            self._kv_rewind_pending = False
            return
        super()._apply_kv_rewind_after_draft(attn_metadata, spec_metadata)

    def _s4l_log(self, shard_logits, first_prev, tokens, w1) -> None:
        """The TAILBK_SPEC S4L engagement line: once per rank, on the first step whose biased logits took the
        two-launch glue (host only; a captured step logs it while it is captured)."""
        if self._tailbk_s4l_logged or not (self._s4_fp32 and S4_LOGITS_BATCHED and TAILBK_SPEC):
            return
        if markov_bias.supports(shard_logits, first_prev, tokens, w1):
            self._tailbk_s4l_logged = True
            logger.info(
                (
                    "[M3 dispatch] TAILBK_SPEC S4L: biased block logits glue in two l"
                    "aunches on rank "
                    f"{self.mapping.tp_rank}"
                )
            )

    def _s4_log_ll(self, batch: int) -> None:
        """The MARKOV_LL engagement line: once per rank, on the first served step whose chain call took the LL
        exchange kernel (host only; a captured step logs it while it is captured)."""
        if not self._s4_ll_logged and chain.ll_exchange_engaged(batch):
            self._s4_ll_logged = True
            logger.info(
                (
                    "[M3 dispatch] MARKOV_LL: k3_markov LL flag-in-payload exchange ("
                    "batch 1) on rank "
                    f"{self.mapping.tp_rank}"
                )
            )

    def sample_draft_tokens(self, logits, spec_metadata, batch_size, **kwargs):
        tokens = self._s4_tokens
        if tokens is not None and logits.dim() == 3 and (tokens.shape == logits.shape[:2]):
            self._s4_tokens = None
            return tokens if tokens.dtype == torch.int32 else tokens.to(torch.int32)
        return super().sample_draft_tokens(logits, spec_metadata, batch_size, **kwargs)
