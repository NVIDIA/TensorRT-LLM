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
"""S4: the catalog's one-launch Markov chain (``ops/dspark/dspark_markov_chain``) in this target's DSpark
worker (each M3.1 target keeps its own copy of this glue: targets share only the catalog).

``S4WorkerMixin`` goes before ``DSparkWorker`` in the target's worker. For a generation step it serves (switch not
set, 1 <= generation requests <= ``MAX_FAST_BATCH``, greedy drafting, no d2t / guided decoder / DFlash-2 path, a
vocab-sharded borrowed head):

- the draft head computes this rank's vocab shard of the block logits (the head's local GEMM, no gather) and copies
  it into a padded buffer whose tail holds -inf (the step's logits processor, swapped in for the step);
- ``_refine_block_logits`` runs the catalog op over the shards in one launch and keeps the tokens;
- ``sample_draft_tokens`` returns them.

The tokens are bitwise the in-tree chain's on the gathered logits: the gathered logits are the concatenation of the
same per-rank GEMMs, and the op reproduces the chain's bf16 bias, bf16 add and first-maximum argmax. When the op
does not support a step after the logits were sharded, the in-tree sharded path (``vocab_slice`` + the TP-aware
argmax) runs instead, with the same tokens.

Setup (``prebuild_s4``, from the worker's ``prebuild``, outside capture, on every rank): the padded ``markov_w2``
shard, one padded logits buffer per (batch, block), the exchange workspace (collective) and the compiles.

Every routing decision is Python, so it is fixed into each decode graph when the graph is captured:
``TRTLLM_M31_DISABLE_DRAFT_S4`` is read once, at import, not a switch that can be toggled at run time. The set-up
decision must be the same on every rank (the workspace is collective); it follows from the config and the
environment, and ``prebuild_s4`` checks that the ranks agree before the collective.
"""

from __future__ import annotations

import atexit
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.native import family_ops
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.ops.dspark import (
    dspark_markov_bias as markov_bias,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.ops.dspark import (
    dspark_markov_chain as chain,
)
from tensorrt_llm._torch.models.modeling_speculative import dspark_markov_step_bias
from tensorrt_llm._torch.speculative.interface import _RNG_SLOT_SPAN
from tensorrt_llm.logger import logger

from ..kernels.m3ll.prob1 import draft_prob1 as prob1
from .m31_switches import m31_disabled
from .ops.sampling import m31_topp_det as _topp_det

MARKOV_RANK = 256
MAX_FAST_BATCH = 8  # capacity: fixed: the chain's batch bound
DISABLE_ENV = "TRTLLM_M31_DISABLE_DRAFT_S4"
S4_COUNT_LOG_EVERY = 1000  # generation steps between the per-rank "served x/y" engagement lines

# TAILBK_SPEC (DSpark-worker sites): bookkeeping launches
# of the draft tail removed exactly. Here, site C: the chain op's three read-only accepted inputs (ones, arange,
# zeros; 3 launches per step) come from a per-shape cache filled at precompile. Site H: the target's KV-length rewind
# after the draft forward (kv_lens[nc:bs] -= rewind; clamp_(min=0): 2 launches per step) runs in the chain's launch
# (its rewind path, after the grid-dependency wait; nothing between the chain and the stock rewind reads kv_lens).
# Site S4L (the Markov chain's logits path under advanced draft sampling): the biased block logits' glue around the bias
# GEMM -- the prevs cat / casts, get_masked_input_and_mask, F.embedding, masked_fill, the two widening copies and the
# add (14 launches per step) -- in two launches (catalog dspark_markov_bias; the GEMM unchanged).
# Bitwise the same tokens, kv_lens and biased logits. Default ON; kill switch TRTLLM_M31_DISABLE_TAIL_BOOKKEEP_SPEC=1,
# read at import.
TAILBK_SPEC = not m31_disabled("TAIL_BOOKKEEP_SPEC")
# MARKOV_LL: the batch-1 kernel's per-position exchange as one 16-byte {value, tag, index,
# tag} multicast store per CTA (no fence, no flag word), polled and reduced by warp 0 alone (no CTA barrier after the
# poll). Bitwise the batch-1 kernel's (MARKOV_CHAIN_BATCH1_KERNEL) tokens (same entries, same total argmax order).
# Default ON; kill switch TRTLLM_M31_DISABLE_MARKOV_LL=1, read at import.
MARKOV_LL = not m31_disabled("MARKOV_LL")
# TAIL16_G3: the block draft probabilities stored into their request slots as one vectorized row copy
# (mv2_minimax_m3::minimax_m31_scatter_rows) instead of the stock index_put. Bitwise the stock store. Default ON; kill
# switch
# TRTLLM_M31_DISABLE_TAIL_DRAFT_PROBS_SLOT_COPY=1, read at import.
TAIL16_G3 = not m31_disabled("TAIL_DRAFT_PROBS_SLOT_COPY")
TAIL16_G3_MAX_REQUESTS = 8  # capacity: fixed: the requests the row copy serves
_TAIL16_G3_LOGGED: set = set()
# FOLD_F5B: the draft head GEMM writes this rank's vocab shard straight into the chain's padded logits buffer (the
# buffer's row stride as the GEMM's output leading dimension) instead of a contiguous output and a strided copy; taken
# only where lm_head would run the default F.linear GEMM. Bitwise the copied logits. Default ON; kill switch
# TRTLLM_M31_DISABLE_FOLD_DRAFT_HEAD_STRIDED_OUT=1, read at import.
FOLD_F5B = not m31_disabled("FOLD_DRAFT_HEAD_STRIDED_OUT")
_FOLD_F5B_LOGGED: set = set()
# The catalog's strided-logits arm (dspark_markov_bias ``strided_by``): F5B's logits are a view of the padded buffer.
_F5B_ARM = "minimax_m3_1_flash_nvfp4__sm_107__tp8 FOLD_F5B" if FOLD_F5B else None
# TP8_DRAFT_HEAD_MXFP8: the drafter head's Markov-chain shard GEMM on an MXFP8 copy of the head (the DHEAD_MX4 module,
# dspark_draft_head.DraftHeadMx4, on this target's shard), written into the padded buffer through FOLD_F5B's strided
# arm; set up at the end of prebuild_s4 on every rank or none. Drafter-only precision (the verify head stays BF16):
# changes draft tokens, not the target's distribution. Default ON (needs FOLD_F5B); kill switch
# TRTLLM_M31_DISABLE_TP8_DRAFT_HEAD_MXFP8=1, read at import.
DHEAD8 = FOLD_F5B and not m31_disabled("TP8_DRAFT_HEAD_MXFP8")
#: Decode batches whose B x K block rows the MXFP8 copy serves (the TP8 decode worker runs batch 1).
DHEAD8_BATCHES = (1,)
# FOLD_F1: the draft block's top-p renorm (TOPP_DET) also writes its output rows into their request slots
# (mv2_minimax_m3::minimax_m31_topp_renorm_det_v3_rows), so the store after the sampler is skipped for exactly those
# rows. Armed only while CUDA-graph padding rows keep the renorm's output (interface._CHEAP_PADDING_ROWS off, as the
# sm_107 targets arm it). Bitwise the stored rows. Default ON; kill switch TRTLLM_M31_DISABLE_FOLD_TOPP_SLOT_WRITE=1,
# read at import.
FOLD_F1 = not m31_disabled("FOLD_TOPP_SLOT_WRITE")
_FOLD_F1_LOGGED: set = set()

# Read once, at import (the target's switch registry): every rank and every step of a process sees the same setting.
_DISABLED = m31_disabled("DRAFT_S4")


def disabled() -> bool:
    return _DISABLED


class _S4State:
    # TP8_DRAFT_HEAD_MXFP8 (dspark_draft_head.DraftHeadMx4, set by prebuild_s4): None keeps the head's own BF16 GEMM
    # (FOLD_F5B, else GEMM + copy).
    head4 = None

    def __init__(self, mapping, rows: int, shard: int, offset: int, w1: torch.Tensor, w2s: torch.Tensor,
                 block: int, ws: dict) -> None:  # fmt: skip
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
        return self.bufs[(batch, self.block)]


# S4-logits bias form: True = all K positions' bias in one GEMM over B*K rows (S4LC: bitwise the per-step GEMMs);
# False = the in-tree per-step loop over the chain's tokens (exact by construction).
S4_LOGITS_BATCHED = True


def _biased_shard_logits(
    shard_logits: torch.Tensor,
    first_prev: torch.Tensor,
    tokens: torch.Tensor,
    w1: torch.Tensor,
    w2_shard: torch.Tensor,
    fp32: bool,
) -> torch.Tensor:
    """The in-tree chain's biased block logits on this rank's vocab shard, given the chain's tokens: step k adds
    w2 @ w1[prev_k] (dspark_markov_step_bias) with prev_0 = the anchor and prev_k = token k-1, cast to the in-tree
    logits dtype: fp32 under the S4 fp32 sum (LogitsProcessor returns logits.float()), else the shard's bf16."""
    if (
        fp32
        and S4_LOGITS_BATCHED
        and TAILBK_SPEC
        and markov_bias.supports(shard_logits, first_prev, tokens, w1, strided_by=_F5B_ARM)
    ):
        # TAILBK_SPEC S4L: the same prevs, embedding rows, GEMM and fp32 sum, in two launches around the GEMM.
        B, K, N = shard_logits.shape
        bias = F.linear(markov_bias.prev_embeddings(first_prev, tokens, w1), w2_shard)
        return markov_bias.widen_add(shard_logits, bias.view(B, K, N), strided_by=_F5B_ARM)
    B, K, N = shard_logits.shape
    base = shard_logits.float() if fp32 else shard_logits
    prevs = torch.cat([first_prev.view(B, 1), tokens[:, : K - 1].long()], dim=1)
    if S4_LOGITS_BATCHED:
        bias = dspark_markov_step_bias(prevs.reshape(-1), w1, w2_shard).view(B, K, N)
        return base + bias.to(base.dtype)
    out = torch.empty_like(base)
    for k in range(K):
        out[:, k] = base[:, k] + dspark_markov_step_bias(prevs[:, k], w1, w2_shard).to(base.dtype)
    return out


def _head_into(buf: torch.Tensor, lm_head: nn.Module, hidden_states: torch.Tensor, shard: int):
    """FOLD_F5B: lm_head's shard logits computed into ``buf``'s first ``shard`` columns and returned as that
    [rows, shard] view, or None where lm_head would not run the default F.linear GEMM (the caller then copies)."""
    from tensorrt_llm._torch.modules.linear import UnquantizedLinearMethod
    from tensorrt_llm._torch.modules.low_m_gemm import _should_apply_low_m_gemm

    weight = getattr(lm_head, "weight", None)
    if (
        not isinstance(getattr(lm_head, "quant_method", None), UnquantizedLinearMethod)
        or getattr(lm_head, "_locality_domain_weight_shards", None) is not None
        or getattr(lm_head, "use_cute_dsl_bf16_gemm", False)
        or getattr(lm_head, "use_custom_cublas_mm", False)
        or getattr(lm_head, "bias", None) is not None
        or weight is None
        or weight.dtype != torch.bfloat16
        or weight.dim() != 2
        or tuple(weight.shape) != (shard, hidden_states.shape[1])
        or hidden_states.dtype != torch.bfloat16
        or buf.dtype != torch.bfloat16
        or not buf.is_contiguous()
        or buf.shape[-1] < shard
        or buf.numel() // buf.shape[-1] != hidden_states.shape[0]
        or _should_apply_low_m_gemm(hidden_states)
    ):
        return None
    out = buf.view(-1, buf.shape[-1])[:, :shard]
    torch.mm(hidden_states, weight.t(), out=out)
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
        buf = state.buffer(self.batch)
        if state.head4 is not None:
            # TP8_DRAFT_HEAD_MXFP8: the MXFP8 copy writes this rank's shard logits into the padded buffer (its -inf tail
            # untouched) and returns that [B x K, shard] view, as FOLD_F5B does; None (a call it does not take) keeps
            # FOLD_F5B.
            logits = state.head4(hidden_states, lm_head, buf, fallback="FOLD_F5B / BF16 draft head")
            if logits is not None:
                return logits
        if FOLD_F5B:
            logits = _head_into(buf, lm_head, hidden_states, state.shard)
            if logits is not None:
                if self.batch not in _FOLD_F5B_LOGGED:
                    _FOLD_F5B_LOGGED.add(self.batch)
                    logger.info(f"[M3.1 sm_107 tp8] FOLD_F5B taken at {self.batch} requests")
                return logits
        logits = lm_head.apply_linear(hidden_states, None)
        buf[:, :, : state.shard].copy_(logits.view(self.batch, state.block, state.shard))
        return logits


def _head_is_sharded(lm_head: nn.Module, vocab: int, tp: int) -> bool:
    from tensorrt_llm._torch.modules.linear import TensorParallelMode

    return (
        tp > 1
        and getattr(lm_head, "tp_mode", None) == TensorParallelMode.COLUMN
        and getattr(lm_head, "padding_size", 0) == 0
        and getattr(lm_head, "bias", None) is None
        and lm_head.out_features * tp == vocab
        and lm_head.weight.dtype == torch.bfloat16
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
    _s4_step_attn = (
        None  # the step's attention metadata (the KV-length rewind's kv_lens), while S4 serves it
    )
    _tailbk_rewound: bool = (
        False  # TAILBK_SPEC H: this step's chain launch applied the KV-length rewind
    )
    _tailbk_h_logged: bool = False
    _tailbk_s4l_logged: bool = False
    _tailbk_c_logged: bool = False
    _s4_ll_logged: bool = False
    # DRAFT_PROB1: the probabilistic Markov draft (kernels/m3ll/prob1, adapted from an NVIDIA reference
    # implementation), set up by prebuild_s4.
    _p1: Optional["prob1.DraftProb1"] = None

    def prebuild_s4(self, lm_head: nn.Module, draft_model) -> bool:
        """Set up the one-launch chain for this worker; False (and nothing set up) when it does not apply."""
        self._s4 = None
        if not getattr(draft_model, "has_markov_head", False):
            return False
        # The kill switch and the config both decide through the agreement: a rank leaving before it alone would hang
        # the ranks waiting in it.
        if disabled():
            logger.info(f"M3.1 DSpark S4: off ({DISABLE_ENV} set); the in-tree Markov chain runs")
            self._s4_ranks_agree(False)
            return False
        w1 = draft_model.markov_w1
        w2 = draft_model.markov_w2
        mapping = self.mapping
        tp = mapping.tp_size
        vocab = w1.shape[0]
        eligible = not (
            w1.dtype != torch.bfloat16
            or w2.dtype != torch.bfloat16
            or w1.shape[1] != MARKOV_RANK
            or tuple(w2.shape) != tuple(w1.shape)
            or mapping.enable_attention_dp
            or not _head_is_sharded(lm_head, vocab, tp)
        )
        block = int(self.max_draft_len)
        shard = vocab // tp
        rows = chain.padded_rows(shard, MAX_FAST_BATCH, w1.device) if eligible else 0
        if not self._s4_ranks_agree(eligible and rows != 0 and block <= chain.MAX_BLOCK):
            return False
        offset = mapping.tp_rank * shard
        w2s = chain.pad_w2_shard(w2[offset : offset + shard].contiguous(), rows)
        state = _S4State(
            mapping, rows, shard, offset, w1, w2s, block, chain.workspace_args(mapping)
        )
        for batch in range(1, MAX_FAST_BATCH + 1):
            state.bufs[(batch, block)] = chain.pad_logits_buffer(batch, block, rows, w1.device)
        # S4 fp32 sum: the chain compares base + bias as the in-tree chain does, the unrounded fp32 sum (its logits are
        # fp32 from LogitsProcessor's .float()). Kill switch TRTLLM_M31_DISABLE_S4_FP32SUM (bf16-rounded sums).
        self._s4_fp32 = chain.enable_fp32_sum(not m31_disabled("S4_FP32SUM"))
        # MARKOV_CHAIN_BATCH1_KERNEL: batch-1 calls take the chain's batch-1 kernel (bitwise the same tokens).
        # Kill switch TRTLLM_M31_DISABLE_MARKOV_CHAIN_BATCH1_KERNEL, read at import.
        s4b8 = chain.enable_batch1_variant(not m31_disabled("MARKOV_CHAIN_BATCH1_KERNEL"))
        # TAILBK_SPEC site C: the chain's read-only accepted inputs from a per-shape cache (bitwise; see TAILBK_SPEC).
        tailbk_c = chain.enable_cached_inputs(TAILBK_SPEC)
        # MARKOV_LL: those batch-1 calls exchange through the LL flag-in-payload kernel (bitwise; see MARKOV_LL).
        markov_ll = chain.enable_ll_exchange(MARKOV_LL)
        # S4-logits: under advanced (rejection) draft sampling the chain still computes the greedy chain's tokens; the
        # biased block logits are rebuilt from them (_biased_shard_logits) and the stock sampler gathers and samples
        # them as it would the in-tree chain's. Kill switch TRTLLM_M31_DISABLE_S4_LOGITS, read at import.
        self._s4_logits_on = not m31_disabled("S4_LOGITS")
        chain.precompile(mapping, rows, w1, w2s, offset, blocks=(block,), max_batch=MAX_FAST_BATCH)
        self._prebuild_prob1(mapping, w1, w2s[:shard], offset, vocab, block)
        self._prebuild_dhead8(lm_head, state, block)
        self._s4 = state
        self._s4_draft = draft_model
        atexit.register(self._s4_log_count)
        # The engagement line: one per rank, at server start.
        logger.info(
            f"M3.1 DSpark S4: one-launch Markov chain set up on rank {mapping.tp_rank}/{tp} "
            f"(shard {shard} rows padded to {rows}, K={block}, batches 1..{MAX_FAST_BATCH}; fp32 sum "
            f"{'on' if self._s4_fp32 else 'off'}; S4-logits {'on' if self._s4_logits_on else 'off'}; "
            f"S4B8 batch-1 kernel {'on' if s4b8 else 'off'}; MARKOV_LL {'on' if markov_ll and s4b8 else 'off'}; "
            f"TAILBK_SPEC cached inputs {'on' if tailbk_c else 'off'})"
        )
        return True

    def _prebuild_prob1(
        self, mapping, w1: torch.Tensor, w2_shard: torch.Tensor, offset: int, vocab: int, block: int
    ) -> None:
        """DRAFT_PROB1 (TP8): under rejection-sampling acceptance the S4-logits step
        runs the dk_mk_prob1_w8 draft (adapted from an NVIDIA reference implementation) + the draft_probs adapter
        instead of the greedy chain + biased logits + the stock gather / filter / draw (the FOLD_F1 renorm row store and
        TAIL16_G3 store are not reached: the adapter writes the slots). Default ON; kill switch
        TRTLLM_M31_DISABLE_DRAFT_PROB1=1 (a family switch, read at import).
        Collective: the ranks agree first (shapes / environment only), as the chain's workspace."""
        from tensorrt_llm._torch.distributed.ops import _get_mnnvl_workspace_comm

        self._p1 = None
        why = None if self._s4_logits_on else "S4-logits off"
        if why is None:
            why = prob1.supported(mapping.tp_size, block, vocab, w1, w2_shard)
        if not self._s4_ranks_agree(why is None):
            logger.info(
                f"{prob1.TAG} not engaged on rank {mapping.tp_rank}: {why or 'another rank declined'}"
            )
            return
        self._p1 = prob1.DraftProb1.create(
            mapping, _get_mnnvl_workspace_comm(mapping), w1, w2_shard, offset, vocab, block
        )
        logger.info(self._p1.describe())

    def _p1_engages(self, spec_metadata, num_gens: int, state) -> bool:
        return self._p1 is not None and prob1.engages(
            spec_metadata, num_gens, self._p1.k, state.block, self._p1.max_rows
        )

    def _p1_step(self, state, base: torch.Tensor, inputs, spec_metadata) -> torch.Tensor:
        """One DRAFT_PROB1 draft: the tokens [G, K] int32 (sample_draft_tokens returns them) and draft_probs[slot]
        for the G gen requests; publishes draft_probs_last_dim (the context one-hot width and acceptance read it)."""
        nc, bs, _ = self._s4_step_meta
        g = bs - nc
        assert getattr(spec_metadata, "use_rejection_sampling", False), (
            "DRAFT_PROB1 engaged without rejection-sampling acceptance"
        )
        slots = spec_metadata.batch_slot_ids[nc:bs]
        tokens = self._p1.run(
            base.view(g * state.block, base.shape[-1]),
            inputs["first_prev_tokens"].view(-1)[:g].long(),
            slots if slots.dtype == torch.int64 else slots.long(),
            int(getattr(spec_metadata, "dummy_slot_row", 0) or 0),
            spec_metadata.request_offsets[nc:bs],
            int(spec_metadata.rng_draft_slot(0)) * _RNG_SLOT_SPAN,
            _RNG_SLOT_SPAN,
            spec_metadata.request_temperatures[nc:bs],
            spec_metadata.request_seeds[nc:bs],
            spec_metadata.draft_probs,
        )
        spec_metadata.draft_probs_last_dim = self._p1.vocab
        line = self._p1.taken_line(g)
        if line is not None:
            logger.info(line)
        return tokens

    def _dhead8_all_ok(self, local: bool) -> bool:
        """Whether ``local`` holds on every rank of the TP group (a host-side allreduce, usable after a CUDA fault)."""
        agree = getattr(self, "_a1_all_ok", None)
        if callable(agree):
            return agree(local)
        from tensorrt_llm._torch.distributed.ops import (
            _get_mnnvl_workspace_comm,
            _mnnvl_workspace_all_succeeded,
        )

        return _mnnvl_workspace_all_succeeded(_get_mnnvl_workspace_comm(self.mapping), local)

    def _prebuild_dhead8(self, lm_head: nn.Module, state: _S4State, block: int) -> bool:
        """Set up the MXFP8 draft head (TP8_DRAFT_HEAD_MXFP8) on ``state`` (every rank or none); False, with one line
        naming why, when it is off. The steps of the DHEAD_MX4 set-up (dspark_draft_head) on this target's shard: a
        malformed or unrunnable pinned tactic (TRTLLM_M31_DHEAD_MX4_TACTIC) raises on every rank; a set-up failure that
        left the CUDA context faulted fails the boot on every rank; any other failure declines on every rank."""
        from . import dspark_draft_head as _dhead4

        rank = getattr(self.mapping, "tp_rank", None)
        rows = [b * block for b in DHEAD8_BATCHES]
        tactic, source = _dhead4.pinned_tactic() if DHEAD8 else (None, "default")
        # Off by switch still goes through the agreements below (a rank leaving here alone would hang the others).
        if not DHEAD8:
            why = "TRTLLM_M31_DISABLE_TP8_DRAFT_HEAD_MXFP8 set" if FOLD_F5B else "FOLD_F5B off"
        else:
            why = _dhead4.unfit(lm_head, state.shard, block, state.rows)
        if why is None and source != "default":
            bad = _dhead4.tactic_invalid(tactic, state.shard, rows, int(lm_head.weight.shape[1]))
            if bad is not None:
                raise ValueError(f"{_dhead4.TACTIC_ENV}={_dhead4.tactic_name(tactic)}: {bad}")
        head4, fault = None, None
        if why is None:
            try:
                head4 = _dhead4.DraftHeadMx4(lm_head, rows, state.rows, rank=rank, tactic=tactic)
            except Exception as e:  # noqa: BLE001 - declines on every rank through the agreement below
                try:
                    _dhead4.raise_if_cuda_fault(e)
                except RuntimeError as f:
                    fault = f
                why = f"set-up failed: {type(e).__name__}: {e}"
        if not self._dhead8_all_ok(fault is None):
            if fault is not None:
                raise fault
            raise RuntimeError(
                f"DHEAD8 set-up left the CUDA context faulted on another rank (rank {rank} is healthy)"
            )
        if not self._dhead8_all_ok(why is None):
            logger.info(
                f"M3.1 DSpark TP8_DRAFT_HEAD_MXFP8: off on rank {rank} ({why or 'off on another rank'}); "
                "BF16 draft head"
            )
            return False
        state.head4 = head4
        logger.info(
            f"M3.1 DSpark TP8_DRAFT_HEAD_MXFP8: set up on rank {rank}/{self.mapping.tp_size} (shard {head4.n} rows, "
            f"MXFP8 copy {head4.np} rows, K {head4.k}, rows {head4.rows}; GEMM at logical N {head4.n} into the S4 "
            f"buffer, row stride {head4.row_stride}; tactic {_dhead4.tactic_name(head4.tactic)} ({source}); PDL "
            f"{'on' if head4.pdl else 'off'}; copy {head4.bytes / 1e6:.1f} MB; self-check {head4.check}; set-up "
            f"{head4.setup_s:.2f} s)"
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
            and getattr(self, "guided_decoder", None) is None
            and getattr(self, "_d2t", None) is None
            and getattr(draft_model, "lilicorr", None) is None
            and not getattr(draft_model, "has_candidate_selector", False)
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
        if self._p1 is not None:
            try:  # synchronizes: host step hook / exit only, outside capture; at exit CUDA may be torn down
                line = self._p1.fault_line()
            except Exception:  # noqa: BLE001
                line = None
            if line is not None:
                logger.warning(line)
        if self._s4 is not None and self._s4_gen_steps:
            logger.info(
                f"M3.1 DSpark S4: served {self._s4_served}/{self._s4_gen_steps} generation steps on rank "
                f"{self.mapping.tp_rank}"
            )

    def _forward_impl(self, input_ids, position_ids, hidden_states, logits, attn_metadata, spec_metadata, draft_model,
                      resource_manager=None):  # fmt: skip
        num_gens = attn_metadata.num_seqs - attn_metadata.num_contexts
        mode = self._s4_mode(num_gens, spec_metadata, draft_model)
        served = mode is not None
        if not served:
            return super()._forward_impl(input_ids, position_ids, hidden_states, logits, attn_metadata,
                                         spec_metadata, draft_model, resource_manager)  # fmt: skip
        inner = draft_model.logits_processor
        draft_model.logits_processor = _ShardedDraftLogits(inner, self._s4, num_gens)
        self._s4_step = self._s4
        self._s4_step_mode = mode
        self._s4_step_meta = (attn_metadata.num_contexts, attn_metadata.num_seqs, spec_metadata)
        self._s4_step_attn = attn_metadata
        try:
            return super()._forward_impl(input_ids, position_ids, hidden_states, logits, attn_metadata,
                                         spec_metadata, draft_model, resource_manager)  # fmt: skip
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
        if state is not None and gen_logits.dim() == 3 and gen_logits.shape[-1] == state.shard:
            base = state.buffer(gen_logits.shape[0])
            if getattr(self, "_s4_step_mode", None) == "logits" and self._p1_engages(
                spec_metadata, gen_logits.shape[0], state
            ):
                # DRAFT_PROB1: the draft tokens and their proposal (draft_probs) in one op; the stock advanced sampler
                # is skipped (sample_draft_tokens returns the stashed tokens), so gen_logits is returned unchanged.
                self._s4_tokens = self._p1_step(state, base, inputs, spec_metadata)
                return gen_logits
            if (
                chain.supports(base, state.w1, state.w2s)
                and getattr(self, "_s4_step_mode", None) == "logits"
            ):
                # S4-logits: the chain's tokens are the in-tree greedy chain's (bitwise); the biased shard logits are
                # rebuilt from them with the in-tree bias arithmetic and handed to the stock advanced sampler.
                first_prev = inputs["first_prev_tokens"].long()
                tokens = self._s4_chain(state, base, first_prev, spec_metadata)
                self._s4_log_ll(base.shape[0])
                biased = _biased_shard_logits(
                    gen_logits,
                    first_prev,
                    tokens,
                    state.w1,
                    state.w2s[: state.shard],
                    getattr(self, "_s4_fp32", True),
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
        tokens = chain.dspark_markov_chain_rewind(base, first_prev, state.w1, state.w2s, state.offset, kv_lens,
                                                  amount, first, **state.ws)  # fmt: skip
        self._tailbk_rewound = True
        self._tailbk_c_log()
        if not self._tailbk_h_logged:
            self._tailbk_h_logged = True
            logger.info(
                "[M3 dispatch] TAILBK_SPEC H: KV-length rewind in the Markov chain launch "
                f"on rank {self.mapping.tp_rank}"
            )
        return tokens

    def _tailbk_c_log(self) -> None:
        """The TAILBK_SPEC C engagement line: once per rank, after the first chain call that took its accepted inputs
        from the precompile cache (host only; a captured step logs it while it is captured)."""
        if self._tailbk_c_logged or not getattr(chain, "cached_inputs_taken", lambda: False)():
            return
        self._tailbk_c_logged = True
        logger.info(
            "[M3 dispatch] TAILBK_SPEC C: chain accepted inputs from precompile cache (taken) "
            f"on rank {self.mapping.tp_rank}"
        )

    def _tailbk_rewind_args(self, batch: int, spec_metadata):
        """(kv_lens, rewind amounts, first row) for the chain to apply, or None (the stock rewind runs)."""
        if not TAILBK_SPEC or not getattr(self, "_kv_rewind_pending", False):
            return None
        amount = getattr(self, "_kv_rewind_amount", None)
        attn = self._s4_step_attn
        if amount is None or attn is None or not hasattr(attn, "kv_lens_cuda"):
            return None
        if spec_metadata.is_cuda_graph and not torch.cuda.is_current_stream_capturing():
            return None  # the warmup: the stock method skips the rewind (kv_lens is restored wholesale)
        nc, bs = self._kv_rewind_nc, self._kv_rewind_bs
        if bs - nc != batch or amount.numel() != batch:
            return None
        kv_lens = attn.kv_lens_cuda
        if not chain.supports_rewind(kv_lens, amount, nc):
            return None
        return kv_lens, amount, nc

    def _apply_kv_rewind_after_draft(self, attn_metadata, spec_metadata):
        if self._tailbk_rewound:
            # TAILBK_SPEC H: the chain launch applied it (see _s4_chain).
            self._tailbk_rewound = False
            self._kv_rewind_pending = False
            return
        super()._apply_kv_rewind_after_draft(attn_metadata, spec_metadata)

    def _s4l_log(self, shard_logits, first_prev, tokens, w1) -> None:
        """The TAILBK_SPEC S4L engagement line: once per rank, on the first step whose biased logits took the
        two-launch glue (host only; a captured step logs it while it is captured)."""
        if self._tailbk_s4l_logged or not (
            getattr(self, "_s4_fp32", True) and S4_LOGITS_BATCHED and TAILBK_SPEC
        ):
            return
        if markov_bias.supports(shard_logits, first_prev, tokens, w1, strided_by=_F5B_ARM):
            self._tailbk_s4l_logged = True
            logger.info(
                "[M3 dispatch] TAILBK_SPEC S4L: biased block logits glue in two launches "
                f"on rank {self.mapping.tp_rank}"
            )

    def _s4_log_ll(self, batch: int) -> None:
        """The MARKOV_LL engagement line: once per rank, on the first served step whose chain call took the LL
        exchange kernel (host only; a captured step logs it while it is captured)."""
        if not self._s4_ll_logged and chain.ll_exchange_engaged(batch):
            self._s4_ll_logged = True
            logger.info(
                "[M3 dispatch] MARKOV_LL: k3_markov LL flag-in-payload exchange (batch 1) "
                f"on rank {self.mapping.tp_rank}"
            )

    def advanced_sample_draft_block(self, gen_logits, spec_metadata, num_contexts, batch_size):
        """FOLD_F1: the block's renorm is armed to write its rows into their slots as well (_store_block_draft_probs
        then skips them); unarmed, or when the renorm does not take the arm, the block samples and stores as before."""
        from tensorrt_llm._torch.speculative import interface

        armed = (
            FOLD_F1
            and not getattr(interface, "_CHEAP_PADDING_ROWS", False)
            and getattr(spec_metadata, "use_rejection_sampling", False)
            and getattr(spec_metadata, "draft_probs", None) is not None
            and getattr(spec_metadata, "batch_slot_ids", None) is not None
            and gen_logits.dim() == 3
        )
        if armed:
            _topp_det.arm_row_store(
                spec_metadata.draft_probs,
                spec_metadata.batch_slot_ids[num_contexts:batch_size],
                gen_logits.shape[1],
            )
        try:
            return super().advanced_sample_draft_block(
                gen_logits, spec_metadata, num_contexts, batch_size
            )
        finally:
            if armed:
                _topp_det.disarm_row_store()

    @staticmethod
    def _store_block_draft_probs(draft_probs, slot_ids, probs):
        """FOLD_F1: nothing to store when the renorm already wrote these rows into their slots. TAIL16_G3: the stock
        store as one vectorized row copy for up to TAIL16_G3_MAX_REQUESTS requests; any other layout or size keeps the
        stock index_put."""
        if (
            FOLD_F1
            and isinstance(probs, torch.Tensor)
            and _topp_det.rows_stored(draft_probs, slot_ids, probs)
        ):
            if probs.shape[0] not in _FOLD_F1_LOGGED:
                _FOLD_F1_LOGGED.add(probs.shape[0])
                logger.info(
                    f"[M3.1 sm_107 tp8] FOLD_F1: block draft probs stored by the renorm at "
                    f"{probs.shape[0]} requests"
                )
            return
        op = getattr(family_ops(), "minimax_m31_scatter_rows", None) if TAIL16_G3 else None
        if (
            op is not None
            and isinstance(draft_probs, torch.Tensor)
            and isinstance(probs, torch.Tensor)
            and isinstance(slot_ids, torch.Tensor)
            and draft_probs.is_cuda
            and probs.is_cuda
            and slot_ids.is_cuda
            and draft_probs.dtype == torch.float32
            and probs.dtype == torch.float32
            and draft_probs.dim() == 3
            and probs.dim() == 3
            and draft_probs.stride(2) == 1
            and draft_probs.shape[1] >= probs.shape[1]
            and draft_probs.shape[2] >= probs.shape[2]
            and slot_ids.dtype == torch.int64
            and slot_ids.dim() == 1
            and slot_ids.shape[0] == probs.shape[0]
            and 1 <= probs.shape[0] <= TAIL16_G3_MAX_REQUESTS
            and draft_probs.device == probs.device == slot_ids.device
        ):
            op(draft_probs, slot_ids.contiguous(), probs.contiguous())
            key = (int(probs.shape[0]), int(probs.shape[1]))
            if key not in _TAIL16_G3_LOGGED:
                _TAIL16_G3_LOGGED.add(key)
                logger.info(f"[M3.1] TAIL16_G3 taken at {key[0]} requests ({key[1]} rows)")
            return
        # Each request's block proposal at its stable slot.
        _, K, vocab = probs.shape
        draft_probs[slot_ids, :K, :vocab] = probs

    def sample_draft_tokens(self, logits, spec_metadata, batch_size, **kwargs):
        tokens = self._s4_tokens
        if tokens is not None and logits.dim() == 3 and tokens.shape == logits.shape[:2]:
            self._s4_tokens = None
            return tokens if tokens.dtype == torch.int32 else tokens.to(torch.int32)
        return super().sample_draft_tokens(logits, spec_metadata, batch_size, **kwargs)
