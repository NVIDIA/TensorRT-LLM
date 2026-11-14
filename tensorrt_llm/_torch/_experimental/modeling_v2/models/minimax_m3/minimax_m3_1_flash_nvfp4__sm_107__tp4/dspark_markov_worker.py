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
from typing import Callable, Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.ops.dspark import (
    dspark_markov_bias as markov_bias,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.ops.dspark import (
    dspark_markov_chain as chain,
)
from tensorrt_llm._torch.models.modeling_speculative import dspark_markov_step_bias
from tensorrt_llm.logger import logger

from . import dspark_markov_grid as markov_grid
from .m31_switches import m31_disabled, m31_enabled

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
# PADFOLD: on a greedy ("tokens") step the draft head's vocab-shard GEMM writes
# straight into the padded logits buffer -- torch.mm(hidden, W.t(), out=buf[..., :shard] viewed [B*K, shard] with the
# buffer's row stride), the ATen mm that F.linear runs, handed the padded row stride as cuBLAS's ldc -- instead of
# F.linear into a fresh tensor and a copy into the buffer (one launch and 2 x B*K*shard*2 bytes fewer per step).
# Nothing writes the columns past the shard, so they keep their -inf. The returned logits are that strided view (the
# step's reshape is a view, _refine_block_logits reads only its last dim). Bitwise F.linear + copy as long as cuBLAS
# keeps the same algorithm for the wider ldc (the UT checks the bytes at the M3.1 TP4 shapes). Only the head's plain
# F.linear path folds (unquantized bf16, no bias, no low-M / CuTe DSL / custom cuBLAS route). Unlike the original
# PADFOLD, S4-logits steps fold too (as the sm_107 TP8 target's FOLD_F5B does): the S4L glue reads the strided rows
# through the catalog's strided-logits arm (_S4L_ARM), bitwise its contiguous read. DHEAD_MX4 (dspark.DHEAD_MX4), when
# set up, is offered every call first: a call it takes is its own write of the buffer and does not fold here; PADFOLD
# serves the calls it declines, and every call when it is off or killed. Default ON; kill switch
# TRTLLM_M31_DISABLE_PADFOLD=1, read at import.
PADFOLD = not m31_disabled("PADFOLD")
# The catalog's strided-logits arm (dspark_markov_bias ``strided_by``): PADFOLD's and DHEAD_MX4's block logits are the
# first ``shard`` columns of the padded buffer, so the S4L glue takes them at the buffer's row stride (contiguous logits
# are taken as before). Always armed: a strided view only reaches S4L from those two writers.
_S4L_ARM = "minimax_m3_1_flash_nvfp4__sm_107__tp4 PADFOLD/DHEAD_MX4"
# MARKOV_GRID: batch-1 chain calls (S4B8, and MARKOV_LL's exchange) launch the catalog's batch-1 kernel on
# this target's wider grid (dspark_markov_grid: on SM107 198 CTAs x 256 markov_w2 rows instead of the catalog's
# 132 x 384; 4 row blocks per warp instead of 6, 792 exchange entries per position instead of 528). Bitwise the same
# tokens and KV-length rewind. Set up only when every rank plans the same wider grid. Opt-in until its micro bench
# (>= 1.5 us per call) and a disaggregated GEN run (198 of 212 SMs co-resident beside the transfer kernels) back it:
# TRTLLM_M31_ENABLE_MARKOV_GRID=1; TRTLLM_M31_DISABLE_MARKOV_GRID=1 still wins (the catalog grid). Read at import.
MARKOV_GRID = m31_enabled("MARKOV_GRID") and not m31_disabled("MARKOV_GRID")

# Read once, at import (the target's switch registry): every rank and every step of a process sees the same setting.
_DISABLED = m31_disabled("DRAFT_S4")


def disabled() -> bool:
    return _DISABLED


class _S4State:
    # DHEAD_MX4 (dspark_draft_head.DraftHeadMx4, set by the worker's prebuild): the shard GEMM on an MXFP8 copy of the
    # head, written straight into the padded buffer; None keeps the head's own BF16 GEMM (PADFOLD, else GEMM + copy).
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
        self.padfold_logged = False
        self.grid_b1 = 0  # MARKOV_GRID: the batch-1 calls' grid (0: the catalog op's)

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
        and markov_bias.supports(shard_logits, first_prev, tokens, w1, strided_by=_S4L_ARM)
    ):
        # TAILBK_SPEC S4L: the same prevs, embedding rows, GEMM and fp32 sum, in two launches around the GEMM.
        B, K, N = shard_logits.shape
        bias = F.linear(markov_bias.prev_embeddings(first_prev, tokens, w1), w2_shard)
        return markov_bias.widen_add(shard_logits, bias.view(B, K, N), strided_by=_S4L_ARM)
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


def padfold_applies(lm_head: nn.Module, hidden_states: torch.Tensor, shard: int) -> bool:
    """Whether ``lm_head.apply_linear(hidden_states, None)`` is exactly ``F.linear(hidden_states, lm_head.weight)``
    (UnquantizedLinearMethod's default route: no low-M dispatcher, no CuTe DSL or custom cuBLAS GEMM, no bias, no
    locality-domain shards) on a bf16 [shard, K] weight, so PADFOLD may run that GEMM into the padded buffer."""
    from tensorrt_llm._torch.modules.linear import UnquantizedLinearMethod
    from tensorrt_llm._torch.modules.low_m_gemm import _should_apply_low_m_gemm

    w = getattr(lm_head, "weight", None)
    return (
        isinstance(w, torch.Tensor)
        and type(getattr(lm_head, "quant_method", None)) is UnquantizedLinearMethod
        and getattr(lm_head, "bias", None) is None
        and not getattr(lm_head, "use_custom_cublas_mm", False)
        and not getattr(lm_head, "use_cute_dsl_bf16_gemm", False)
        and getattr(lm_head, "_locality_domain_weight_shards", None) is None
        and w.is_cuda
        and w.dtype == torch.bfloat16
        and w.dim() == 2
        and w.shape[0] == shard
        and hidden_states.is_cuda
        and hidden_states.device == w.device
        and hidden_states.dtype == torch.bfloat16
        and hidden_states.dim() == 2
        and hidden_states.shape[1] == w.shape[1]
        and not _should_apply_low_m_gemm(hidden_states)
    )


def padfold_linear(
    hidden_states: torch.Tensor, weight: torch.Tensor, buf: torch.Tensor, shard: int
) -> torch.Tensor:
    """PADFOLD's GEMM: ``hidden_states @ weight.T`` written into ``buf[..., :shard]`` ([B, K, rows] contiguous, B * K
    = the hidden rows) through its [B * K, shard] view (row stride rows), which it returns. ATen's mm takes a result
    whose last dim is contiguous and whose row stride is at least its width in place (cuBLAS ldc = rows); the columns
    past ``shard`` are not written."""
    out = buf.view(-1, buf.shape[-1])[:, :shard]
    return torch.mm(hidden_states, weight.t(), out=out)


# capacity: fixed: ranks_agree_value compares values below 2**16 (a batch-1 grid is at most 256 CTAs).
AGREE_VALUE_BITS = 16


def ranks_agree_value(all_ranks: Callable[[bool], bool], value: int) -> bool:
    """Whether every rank holds the same nonzero ``value``, given ``all_ranks(local)``: a collective that returns, on
    every rank, whether ``local`` is True on every rank. The answer is the same on every rank, and so is every call
    each rank makes (each call's result is), so the collective rounds stay matched: one for "nonzero everywhere", then
    per bit "1 everywhere" or else "0 everywhere", stopping at the first bit the ranks disagree on. A value outside
    [0, 2**AGREE_VALUE_BITS) counts as 0 (disagreement), never as an exception that would leave the peers waiting."""
    if not 0 <= value < (1 << AGREE_VALUE_BITS):
        value = 0
    if not all_ranks(value != 0):
        return False
    for bit in range(AGREE_VALUE_BITS):
        one = (value >> bit) & 1 == 1
        if not (all_ranks(one) or all_ranks(not one)):
            return False
    return True


class _ShardedDraftLogits(nn.Module):
    """The step's draft logits processor: this rank's vocab shard of ``lm_head`` over the gathered block rows, written
    into the padded buffer by exactly one of, in this order: the DHEAD_MX4 copy (``state.head4``, when it is set up and
    takes the call), PADFOLD (``fold``: the head's own GEMM straight into the buffer), the head's GEMM + a copy. None of
    them writes past the shard, so the buffer's -inf tail holds on every path. Any other call goes to the drafter's own
    processor."""

    def __init__(self, inner: nn.Module, state: _S4State, batch: int, fold: bool = False) -> None:
        super().__init__()
        self.inner = inner
        self._state = [state]
        self.batch = batch
        self.fold = fold

    def forward(self, hidden_states, lm_head, attn_metadata, return_context_logits: bool = False):
        state = self._state[0]
        if state.head4 is not None:
            # DHEAD_MX4: the MXFP8 copy writes this rank's shard logits into the padded buffer (its -inf tail untouched)
            # and returns that [B x K, shard] view; None (a call it does not take) keeps the path below, which its
            # decline line names.
            block_call = (
                hidden_states.dim() == 2 and hidden_states.shape[0] == self.batch * state.block
            )
            logits = state.head4(
                hidden_states,
                lm_head,
                state.bufs.get((self.batch, state.block)),
                fallback="BF16 draft head" if block_call else "drafter processor",
            )
            if logits is not None:
                return logits
        if hidden_states.dim() != 2 or hidden_states.shape[0] != self.batch * state.block:
            return self.inner(hidden_states, lm_head, attn_metadata, return_context_logits)
        buf = state.buffer(self.batch)
        if self.fold and padfold_applies(lm_head, hidden_states, state.shard):
            logits = padfold_linear(hidden_states, lm_head.weight, buf, state.shard)
            if not state.padfold_logged:
                state.padfold_logged = True
                logger.info(
                    f"[M3 dispatch] PADFOLD taken: draft head GEMM into the S4 padded buffer ({hidden_states.shape[0]} "
                    f"rows, ldc {buf.shape[-1]}) on rank {getattr(state.mapping, 'tp_rank', None)}"
                )
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
        # MARKOV_GRID: the batch-1 calls' wider grid, when every rank plans the same one (collective).
        grid_b1 = markov_grid.plan(rows, block, w1.device) if MARKOV_GRID else 0
        if grid_b1 and not markov_grid.fits_workspace(block, grid_b1, state.ws):
            grid_b1 = 0
        if self._s4_ranks_agree_value(grid_b1):
            markov_grid.precompile(rows, w1, w2s, offset, block, grid_b1, state.ws)
            state.grid_b1 = grid_b1
        grid_note = (
            f" x {rows // state.grid_b1} rows (catalog {markov_grid.catalog_grid(rows, block, w1.device)})"
            if state.grid_b1
            else ""
        )
        self._s4 = state
        self._s4_draft = draft_model
        atexit.register(self._s4_log_count)
        # The engagement line: one per rank, at server start.
        logger.info(
            f"M3.1 DSpark S4: one-launch Markov chain set up on rank {mapping.tp_rank}/{tp} "
            f"(shard {shard} rows padded to {rows}, K={block}, batches 1..{MAX_FAST_BATCH}; fp32 sum "
            f"{'on' if self._s4_fp32 else 'off'}; S4-logits {'on' if self._s4_logits_on else 'off'}; "
            f"S4B8 batch-1 kernel {'on' if s4b8 else 'off'}; MARKOV_LL {'on' if markov_ll and s4b8 else 'off'}; "
            f"TAILBK_SPEC cached inputs {'on' if tailbk_c else 'off'}; PADFOLD {'on' if PADFOLD else 'off'}; "
            f"MARKOV_GRID batch-1 grid {state.grid_b1 or 'off'}{grid_note})"
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

    def _s4_ranks_agree_value(self, value: int) -> bool:
        """Whether every rank of the TP group holds the same nonzero ``value`` (MARKOV_GRID's batch-1 grid: the
        exchange layout follows it, so a rank may launch it only if all do); False otherwise, on every rank."""
        from tensorrt_llm._torch.distributed.ops import (
            _get_mnnvl_workspace_comm,
            _mnnvl_workspace_all_succeeded,
        )

        comm = _get_mnnvl_workspace_comm(self.mapping)
        return ranks_agree_value(lambda local: _mnnvl_workspace_all_succeeded(comm, local), value)

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
        draft_model.logits_processor = _ShardedDraftLogits(inner, self._s4, num_gens, fold=PADFOLD)
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
            if not gen_logits.is_contiguous():
                # PADFOLD's or DHEAD_MX4's strided view of the padded buffer, declined by the op: the in-tree path gets
                # the same values in a contiguous tensor (the layout the head's own GEMM returns).
                gen_logits = gen_logits.contiguous()
        return super()._refine_block_logits(draft_model, gen_logits, inputs, spec_metadata)

    def _s4_chain(
        self, state: _S4State, base: torch.Tensor, first_prev: torch.Tensor, spec_metadata
    ):
        """The chain launch; under TAILBK_SPEC (site H) it also applies the step's deferred KV-length rewind when
        the stock _apply_kv_rewind_after_draft would (a real step: not the eager warmup of a graph batch, whose
        kv_lens is restored wholesale), and marks it done for that method."""
        rewind = self._tailbk_rewind_args(base.shape[0], spec_metadata)
        grid = state.grid_b1 if base.shape[0] == 1 else 0
        if grid and markov_grid.supports(base, state.w1, state.w2s, grid):
            # MARKOV_GRID: the same batch-1 call on this target's wider grid (with the rewind when it applies).
            kv_lens, amount, first = rewind if rewind is not None else (None, None, 0)
            tokens = markov_grid.markov_chain_b1(base, first_prev, state.w1, state.w2s, state.offset, grid,
                                                 kv_lens=kv_lens, rewind=amount, rewind_first=first,
                                                 **state.ws)  # fmt: skip
            self._markov_grid_log(grid, base.shape[-1] // grid)
            if rewind is None:
                self._tailbk_c_log()
                return tokens
        elif rewind is None:
            tokens = chain.dspark_markov_chain(
                base, first_prev, state.w1, state.w2s, state.offset, **state.ws
            )
            self._tailbk_c_log()
            return tokens
        else:
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

    _markov_grid_logged: bool = False

    def _markov_grid_log(self, grid: int, rows: int) -> None:
        """The MARKOV_GRID engagement line: once per rank, on the first chain call launched on this target's grid
        (host only; a captured step logs it while it is captured)."""
        if self._markov_grid_logged:
            return
        self._markov_grid_logged = True
        logger.info(
            f"[M3 dispatch] MARKOV_GRID taken: k3_markov batch-1 grid {grid} x {rows} rows "
            f"({'LL' if chain.ll_exchange_engaged(1) else 'Lamport'} exchange) on rank {self.mapping.tp_rank}"
        )

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
        if markov_bias.supports(shard_logits, first_prev, tokens, w1, strided_by=_S4L_ARM):
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

    def sample_draft_tokens(self, logits, spec_metadata, batch_size, **kwargs):
        tokens = self._s4_tokens
        if tokens is not None and logits.dim() == 3 and tokens.shape == logits.shape[:2]:
            self._s4_tokens = None
            return tokens if tokens.dtype == torch.int32 else tokens.to(torch.int32)
        return super().sample_draft_tokens(logits, spec_metadata, batch_size, **kwargs)
