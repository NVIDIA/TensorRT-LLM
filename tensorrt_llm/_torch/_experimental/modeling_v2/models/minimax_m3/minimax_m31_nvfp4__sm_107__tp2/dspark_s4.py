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
"""S4: the catalog's one-launch Markov chain (``catalog/spec_decode/dspark_markov_chain``) in this target's DSpark
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
``TRTLLM_M31_DRAFT_DISABLE_S4`` is read once, at import, not a switch that can be toggled at run time. The set-up
decision must be the same on every rank (the workspace is collective); it follows from the config and the
environment, and ``prebuild_s4`` checks that the ranks agree before the collective.

Gate only: with ``TRTLLM_M31_DSPARK_GATE_DUMP=<dir>`` set at prebuild (eager runs), each served step also runs the
in-tree chain on the gathered logits and logs both token sets per rank to ``<dir>/rank<r>.jsonl``. It synchronizes;
never set it in a perf arm.
"""

from __future__ import annotations

import atexit
import json
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

from .kernels import m3_report
from .m31_switches import m31_disabled
from .ops.spec.m31_markov_epilogue import m31_markov_chain_fused, markov_epilogue_op

MARKOV_RANK = 256
# S4CAP12: generation steps of up to 12 requests take the chain (the kernel takes up to 16). Measured on VR200 TP2
# (net of the S4-logits rebuild): it saves 115 / 42 us per step over the in-tree chain
# at 8 / 12 requests and loses 46.5 at 16. Kill switch TRTLLM_M31_DISABLE_S4CAP12 (back to 8 requests), read at import.
S4CAP12 = not m31_disabled("S4CAP12")
MAX_FAST_BATCH = 12 if S4CAP12 else 8  # capacity: measured (above)
DISABLE_ENV = "TRTLLM_M31_DRAFT_DISABLE_S4"
S4_COUNT_LOG_EVERY = 1000  # generation steps between the per-rank "served x/y" engagement lines
GATE_DUMP_ENV = "TRTLLM_M31_DSPARK_GATE_DUMP"  # gate-only: a directory; never set in a perf arm (it synchronizes)

# TAILBK_SPEC (draft-tail bookkeeping, DSpark-worker sites): bookkeeping launches
# of the draft tail removed exactly. Here, site C: the chain op's three read-only accepted inputs (ones, arange,
# zeros; 3 launches per step) come from a per-shape cache filled at precompile. Site H: the target's KV-length rewind
# after the draft forward (kv_lens[nc:bs] -= rewind; clamp_(min=0): 2 launches per step) runs in the chain's launch
# (its rewind path, after the grid-dependency wait; nothing between the chain and the stock rewind reads kv_lens).
# Site S4L (S4-logits, the advanced-sampling contract path): the biased block logits' glue around the bias GEMM -- the
# prevs cat / casts, get_masked_input_and_mask, F.embedding, masked_fill, the two widening copies and the add (14
# launches per step) -- in two launches (catalog dspark_markov_bias; the GEMM unchanged).
# Bitwise the same tokens, kv_lens and biased logits. Default ON; kill switch TRTLLM_M31_DISABLE_TAILBK_SPEC=1, read
# at import.
TAILBK_SPEC = not m31_disabled("TAILBK_SPEC")
# S4FE: the steps the chain declines (above MAX_FAST_BATCH requests; the agg TP2
# c46 / c56 / c64 steps) run the in-tree full-vocab Markov chain with its elementwise tail fused per position
# (ops/spec/m31_markov_epilogue.py): the same embedding and F.linear, then one C++ pass for the add, the corrected
# write, torch's argmax and the next embedding. Bitwise the in-tree corrected logits. VR TP2 K3: the
# in-tree chain costs 357 / 401 / 444 us per step at 46 / 56 / 64 requests, about 30-45 us of it the GEMM.
# Default ON; kill switch TRTLLM_M31_DISABLE_S4FE=1, read at import.
S4FE = not m31_disabled("S4FE")
# Zero uniform CUDA-graph padding rows in place. Real rows are untouched; padding rows become +0.0.
# Ragged verify layouts retain the stock path. Default ON; kill switch TRTLLM_M31_DISABLE_TAIL16_G2=1.
TAIL16_G2 = not m31_disabled("TAIL16_G2")
_TAIL16_G2_LOGGED: set = set()  # (rows, rows per request) already logged
# Copy block draft probabilities into request slots. Default ON; kill switch TRTLLM_M31_DISABLE_TAIL16_G3=1.
TAIL16_G3 = not m31_disabled("TAIL16_G3")
_TAIL16_G3_LOGGED: set = set()
# TAIL_BK3: post-layer bookkeeping in fewer launches, bitwise the stock ops:
# (B) the S4FE position-0 Markov embedding as one gather (ops/spec/m31_markov_epilogue.py), (C) the block sampler's
# per-row parameters and Philox state as one kernel (_block_sampling_params), (D) no int32 -> int64 -> int32 round trip
# of the block tokens. Default ON; kill switch TRTLLM_M31_DISABLE_TAIL_BK3=1, read at import.
TAIL_BK3 = not m31_disabled("TAIL_BK3")
_TAIL_BK3_LOGGED: set = set()

# Read once, at import: every rank and every step of a process sees the same setting. Any value but "" or "0" turns
# the item off, as the targets' m31_switches reads its switches.
_DISABLED = os.environ.get(DISABLE_ENV, "").strip() not in ("", "0")


def disabled() -> bool:
    return _DISABLED


class _S4State:
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
        and markov_bias.supports(shard_logits, first_prev, tokens, w1)
    ):
        # TAILBK_SPEC S4L: the same prevs, embedding rows, GEMM and fp32 sum, in two launches around the GEMM.
        B, K, N = shard_logits.shape
        bias = F.linear(markov_bias.prev_embeddings(first_prev, tokens, w1), w2_shard)
        return markov_bias.widen_add(shard_logits, bias.view(B, K, N))
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
        and getattr(lm_head, "padding_size", 0) == 0
        and getattr(lm_head, "bias", None) is None
        and lm_head.out_features * tp == vocab
        and lm_head.weight.dtype == torch.bfloat16
    )


def _capacity_reason(shard: int) -> str:
    """The decline reason when no padded shard fits the chain kernel: its CTAs keep their markov_w2 rows in shared
    memory up to SMEM_ROW_BUDGET rows and read the rest from global memory (S4STREAM), up to MAX_ROWS_PER_CTA rows per
    CTA, and must all be co-resident (the TP2 shard of 100,032 rows is padded to 101,376 = 144 CTAs x 704 rows)."""
    kern = chain._kernel_module()
    budget = int(getattr(kern, "MAX_ROWS_PER_CTA", getattr(kern, "SMEM_ROW_BUDGET", 0)))
    return (
        f"vocab shard {shard} rows past the chain kernel's capacity of {chain.MAX_GRID} co-resident CTAs x {budget} "
        f"markov_w2 rows = {chain.MAX_GRID * budget} rows"
    )


class S4WorkerMixin:
    """Mixed into the target's ``DSparkWorker`` subclass (before it in the bases)."""

    _block_tokens_keep_int32: bool = TAIL_BK3
    _s4: Optional[_S4State] = None
    _s4_step: Optional[_S4State] = None
    _s4_tokens: Optional[torch.Tensor] = None
    _s4_step_meta: Optional[Tuple[int, int, object]] = None
    _s4_dump_step: int = 0
    _s4_dump_dir: Optional[str] = None
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

    def prebuild_s4(self, lm_head: nn.Module, draft_model) -> bool:
        """Set up the one-launch chain for this worker; False (and nothing set up) when it does not apply."""
        self._s4 = None
        self._s4_dump_dir = os.environ.get(GATE_DUMP_ENV) or None
        if disabled():
            logger.info(f"M3.1 DSpark S4: off ({DISABLE_ENV} set); the in-tree Markov chain runs")
            return False
        if not getattr(draft_model, "has_markov_head", False):
            return self._s4_decline("the drafter has no Markov head")
        w1 = draft_model.markov_w1
        w2 = draft_model.markov_w2
        mapping = self.mapping
        tp = mapping.tp_size
        vocab = w1.shape[0]
        # Config and weights only, the same on every rank: no rank goes on to the collective agreement below alone.
        off = [
            why
            for why, bad in (
                (f"markov_w1 {w1.dtype} / markov_w2 {w2.dtype}, not bf16",
                 w1.dtype != torch.bfloat16 or w2.dtype != torch.bfloat16),
                (f"Markov rank {w1.shape[1]}, not {MARKOV_RANK}", w1.shape[1] != MARKOV_RANK),
                (f"markov_w2 {tuple(w2.shape)} is not markov_w1's {tuple(w1.shape)}",
                 tuple(w2.shape) != tuple(w1.shape)),
                ("attention DP", bool(mapping.enable_attention_dp)),
                ("the draft head is not a vocab-sharded, unpadded, bias-free bf16 column-parallel head",
                 not _head_is_sharded(lm_head, vocab, tp)),
            )
            if bad
        ]  # fmt: skip
        if off:
            return self._s4_decline("; ".join(off))
        block = int(self.max_draft_len)
        shard = vocab // tp
        rows = chain.padded_rows(shard, MAX_FAST_BATCH, w1.device)
        local = rows != 0 and block <= chain.MAX_BLOCK
        if not self._s4_ranks_agree(local):
            if rows == 0:
                return self._s4_decline(_capacity_reason(shard))
            if block > chain.MAX_BLOCK:
                return self._s4_decline(f"draft length {block} past the chain's {chain.MAX_BLOCK}")
            return self._s4_decline("off on another rank")
        offset = mapping.tp_rank * shard
        w2s = chain.pad_w2_shard(w2[offset : offset + shard].contiguous(), rows)
        state = _S4State(
            mapping, rows, shard, offset, w1, w2s, block, chain.workspace_args(mapping)
        )
        # One padded buffer for every batch: batch b takes its leading b rows (contiguous, at fixed addresses).
        whole = chain.pad_logits_buffer(MAX_FAST_BATCH, block, rows, w1.device)
        for batch in range(1, MAX_FAST_BATCH + 1):
            state.bufs[(batch, block)] = whole[:batch]
        # S4 fp32 sum: the chain compares base + bias as the in-tree chain does, the unrounded fp32 sum (its logits are
        # fp32 from LogitsProcessor's .float()). Kill switch TRTLLM_M31_DISABLE_S4_FP32SUM (bf16-rounded sums).
        self._s4_fp32 = chain.enable_fp32_sum(not m31_disabled("S4_FP32SUM"))
        # S4B8: batch-1 calls take the chain's batch-1 kernel (bitwise the same tokens; faster at one request only).
        # Kill switch TRTLLM_M31_DISABLE_S4B8, read at import.
        s4b8 = chain.enable_batch1_variant(not m31_disabled("S4B8"))
        self._s4b8 = bool(s4b8)
        # TAILBK_SPEC site C: the chain's read-only accepted inputs from a per-shape cache (bitwise; see TAILBK_SPEC).
        tailbk_c = chain.enable_cached_inputs(TAILBK_SPEC)
        # S4-logits: under advanced (rejection) draft sampling the chain still computes the greedy chain's tokens; the
        # biased block logits are rebuilt from them (_biased_shard_logits) and the stock sampler gathers and samples
        # them as it would the in-tree chain's. ON by default with MCAG off (the TP8 recipe: the stock all-gather of the
        # shard logits keeps two runs of the same requests identical). Kill switch TRTLLM_M31_DISABLE_S4_LOGITS.
        self._s4_logits_on = not m31_disabled("S4_LOGITS")
        chain.precompile(mapping, rows, w1, w2s, offset, blocks=(block,), max_batch=MAX_FAST_BATCH)
        self._s4 = state
        self._s4_draft = draft_model
        atexit.register(self._s4_log_count)
        # The engagement line: one per rank, at server start.
        logger.info(
            f"M3.1 DSpark S4: one-launch Markov chain set up on rank {mapping.tp_rank}/{tp} "
            f"(shard {shard} rows padded to {rows}, K={block}, batches 1..{MAX_FAST_BATCH}; fp32 sum "
            f"{'on' if self._s4_fp32 else 'off'}; S4-logits {'on' if self._s4_logits_on else 'off'}; "
            f"S4B8 batch-1 kernel {'on' if s4b8 else 'off'}; TAILBK_SPEC cached inputs {'on' if tailbk_c else 'off'})"
        )
        return True

    def _s4_decline(self, reason: str) -> bool:
        """The S4 decline line (one per rank, at prebuild), then False: the in-tree Markov chain runs, with the
        TAILBK_SPEC sites the chain launch carries (C, H, S4L) not taken."""
        mapping = getattr(self, "mapping", None)
        logger.info(
            f"M3.1 DSpark S4: one-launch Markov chain off on rank {getattr(mapping, 'tp_rank', None)}/"
            f"{getattr(mapping, 'tp_size', None)} ({reason}); the in-tree Markov chain runs (TAILBK_SPEC chain sites "
            "C / H / S4L not taken)"
        )
        return False

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
        if self._s4_dump_dir and not torch.cuda.is_current_stream_capturing():
            # The gate's independent denominator: every generation step, its eligibility by batch and sampling
            # alone, and whether the chain served it.
            eligible = 1 <= num_gens <= MAX_FAST_BATCH and not getattr(
                spec_metadata, "wants_advanced_draft_sampling", False
            )
            self._s4_dump_line({"kind": "step", "num_gens": num_gens, "eligible": bool(eligible),
                                "served": bool(served)})  # fmt: skip
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
            if (
                chain.supports(base, state.w1, state.w2s)
                and getattr(self, "_s4_step_mode", None) == "logits"
            ):
                # S4-logits: the chain's tokens are the in-tree greedy chain's (bitwise); the biased shard logits are
                # rebuilt from them with the in-tree bias arithmetic and handed to the stock advanced sampler.
                first_prev = inputs["first_prev_tokens"].long()
                tokens = self._s4_chain(state, base, first_prev, spec_metadata)
                biased = _biased_shard_logits(
                    gen_logits,
                    first_prev,
                    tokens,
                    state.w1,
                    state.w2s[: state.shard],
                    getattr(self, "_s4_fp32", True),
                )
                self._s4l_log(gen_logits, first_prev, tokens, state.w1)
                if self._s4_dump_dir and not torch.cuda.is_current_stream_capturing():
                    self._s4_logits_gate_dump(
                        draft_model, gen_logits, inputs, spec_metadata, biased
                    )
                return biased
            if chain.supports(base, state.w1, state.w2s):
                tokens = self._s4_chain(
                    state, base, inputs["first_prev_tokens"].long(), spec_metadata
                )
                self._s4_tokens = tokens
                if self._s4_dump_dir and not torch.cuda.is_current_stream_capturing():
                    self._s4_gate_dump(draft_model, gen_logits, inputs, spec_metadata, tokens)
                return gen_logits
        if S4FE and state is None:
            fused = self._s4fe(draft_model, gen_logits, inputs, spec_metadata)
            if fused is not None:
                return fused
        return super()._refine_block_logits(draft_model, gen_logits, inputs, spec_metadata)

    @staticmethod
    def _store_block_draft_probs(draft_probs, slot_ids, probs):
        """TAIL16_G3: the stock store as one vectorized row copy; any other layout keeps the stock index_put."""
        loaded = markov_epilogue_op() if TAIL16_G3 else None
        op = (
            getattr(torch.ops.trtllm, "minimax_m31_scatter_rows", None)
            if loaded is not None
            else None
        )
        if (
            op is not None
            and draft_probs is not None
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
            and draft_probs.device == probs.device == slot_ids.device
            and probs.shape[0] * probs.shape[1] <= 65535
            and probs.shape[2] <= 2147483647
        ):
            op(draft_probs, slot_ids.contiguous(), probs.contiguous())
            key = (int(probs.shape[0]), int(probs.shape[1]))
            if key not in _TAIL16_G3_LOGGED:
                _TAIL16_G3_LOGGED.add(key)
                logger.info(f"[M3.1] TAIL16_G3 taken at {key[0]} requests ({key[1]} rows)")
            return
        from tensorrt_llm._torch.speculative.interface import SpecWorkerBase

        SpecWorkerBase._store_block_draft_probs(draft_probs, slot_ids, probs)

    def _block_sampling_params(self, spec_metadata, num_contexts: int, batch_size: int, K: int):
        """TAIL_BK3 (C): the stock per-row repeat_interleave of the request sampling parameters and the Philox state
        (_rng_state_per_request, repeat=K) in one launch; any other layout keeps the stock ops."""
        from tensorrt_llm._torch.speculative.interface import _RNG_SLOT_SPAN

        loaded = markov_epilogue_op() if TAIL_BK3 else None
        op = (
            getattr(torch.ops.trtllm, "minimax_m31_expand_block_sampling", None)
            if loaded is not None
            else None
        )
        bufs = tuple(
            getattr(spec_metadata, name, None)
            for name in (
                "request_temperatures",
                "request_top_ks",
                "request_top_ps",
                "request_min_ps",
                "request_seeds",
                "request_offsets",
            )
        )
        dtypes = (
            torch.float32,
            torch.int32,
            torch.float32,
            torch.float32,
            torch.int64,
            torch.int64,
        )
        if (
            op is not None
            and K >= 1
            and batch_size > num_contexts
            and all(b is not None and b.is_cuda and b.dim() == 1 and b.stride(0) == 1 for b in bufs)
            and all(b.dtype == dt for b, dt in zip(bufs, dtypes))
            and len({b.device for b in bufs}) == 1
        ):
            slot = int(spec_metadata.rng_draft_slot(0))
            out = op(
                *(b[num_contexts:batch_size] for b in bufs),
                int(K),
                slot * _RNG_SLOT_SPAN,
                _RNG_SLOT_SPAN,
            )
            key = (int(batch_size - num_contexts), int(K))
            if key not in _TAIL_BK3_LOGGED:
                _TAIL_BK3_LOGGED.add(key)
                logger.info(
                    f"[M3.1] TAIL_BK3 sampling params taken at {key[0]} requests (K{key[1]})"
                )
            return tuple(out)
        return super()._block_sampling_params(spec_metadata, num_contexts, batch_size, K)

    @staticmethod
    def _zero_padding_rows(
        logits, spec_metadata, num_contexts, batch_size, rows_per_request, *, packed_verify=False
    ):
        """TAIL16_G2: the stock zeroing of the CUDA-graph padding requests' rows, in place and in one launch, for the
        uniform layout; every other case (ragged verify, a layout mismatch, the op or switch off) is the stock code."""
        slot_ids = getattr(spec_metadata, "batch_slot_ids", None)
        dummy = getattr(spec_metadata, "dummy_slot_row", 0)
        num_gens = batch_size - num_contexts
        op = markov_epilogue_op() if TAIL16_G2 else None
        if (
            op is not None
            and slot_ids is not None
            and dummy > 0
            and num_gens > 0
            and not (packed_verify and spec_metadata.is_ragged_verify)
            and logits.dim() == 2
            and logits.shape[0] == num_gens * rows_per_request
            and logits.dtype == torch.float32
            and logits.stride(1) == 1
            and slot_ids.dtype == torch.int64
        ):
            torch.ops.trtllm.minimax_m31_zero_padding_rows(
                logits,
                slot_ids[num_contexts:batch_size].contiguous(),
                int(dummy),
                int(rows_per_request),
            )
            key = (int(logits.shape[0]), int(rows_per_request))
            if key not in _TAIL16_G2_LOGGED:
                _TAIL16_G2_LOGGED.add(key)
                logger.info(f"[M3.1] TAIL16_G2 taken at {key[0]} rows ({key[1]} per request)")
            return logits
        from tensorrt_llm._torch.speculative.interface import SpecWorkerBase

        return SpecWorkerBase._zero_padding_rows(
            logits,
            spec_metadata,
            num_contexts,
            batch_size,
            rows_per_request,
            packed_verify=packed_verify,
        )

    def _s4fe(self, draft_model, gen_logits, inputs, spec_metadata):
        """S4FE: the in-tree chain's corrected logits with the fused tail, or None (the in-tree path runs): no Markov
        head, logits not the full vocab or TP-sharded (the in-tree argmax would gather), or outside the op."""
        if not getattr(draft_model, "has_markov_head", False) or gen_logits.dim() != 3:
            return None
        if gen_logits.shape[-1] != draft_model.markov_w2.shape[0] or self._draft_logits_are_sharded(
            gen_logits[:, 0], spec_metadata
        ):
            return None
        out = m31_markov_chain_fused(
            gen_logits, inputs["first_prev_tokens"], draft_model.markov_w1, draft_model.markov_w2
        )
        key = (tuple(gen_logits.shape[:2]), out is not None)
        logged = self.__dict__.setdefault("_s4fe_logged", set())
        if key not in logged:
            logged.add(key)
            what = "taken" if out is not None else "stock"
            logger.info(
                f"[M3.1] S4FE {what} at {gen_logits.shape[0]} requests (K{gen_logits.shape[1]}) on rank "
                f"{self.mapping.tp_rank}"
            )
        return out

    def _s4_chain(
        self, state: _S4State, base: torch.Tensor, first_prev: torch.Tensor, spec_metadata
    ):
        """The chain launch; under TAILBK_SPEC (site H) it also applies the step's deferred KV-length rewind when
        the stock _apply_kv_rewind_after_draft would (a real step: not the eager warmup of a graph batch, whose
        kv_lens is restored wholesale), and marks it done for that method."""
        rewind = self._tailbk_rewind_args(base.shape[0], spec_metadata)
        self._s4_work(base.shape[0], rewind is not None)
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

    def _s4_work(self, batch: int, rewind: bool) -> None:
        """Gate e: one chain launch's work counters (S4, and the variants and TAILBK_SPEC sites it carries)."""
        m3_report.work("S4", f"Markov chain launch, batch {batch}, rank {self.mapping.tp_rank}")
        if getattr(self, "_s4_fp32", False):
            m3_report.work("S4_FP32SUM", f"batch {batch}")
        if getattr(self, "_s4b8", False) and batch == 1:
            m3_report.work("S4B8", "batch-1 chain kernel")
        if TAILBK_SPEC and (rewind or getattr(chain, "cached_inputs_taken", lambda: False)()):
            m3_report.work("TAILBK_SPEC", "chain site " + ("H (KV rewind) + C" if rewind else "C"))

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
        if markov_bias.supports(shard_logits, first_prev, tokens, w1):
            self._tailbk_s4l_logged = True
            logger.info(
                "[M3 dispatch] TAILBK_SPEC S4L: biased block logits glue in two launches "
                f"on rank {self.mapping.tp_rank}"
            )

    def _s4_gate_dump(self, draft_model, shard_logits, inputs, spec_metadata, tokens) -> None:
        """Gate-only (eager): the in-tree chain on the gathered logits of this same step, logged beside the
        one-launch chain's tokens, one JSON line per step and rank."""
        from tensorrt_llm._torch.distributed import allgather

        full = allgather(shard_logits.contiguous(), self.mapping, dim=-1)
        if getattr(self, "_s4_fp32", True):
            full = full.float()  # the in-tree chain's input: LogitsProcessor returns logits.float()
        corrected = super()._refine_block_logits(draft_model, full, inputs, spec_metadata)
        stock = torch.argmax(corrected, dim=-1)
        num_contexts, num_seqs, meta = self._s4_step_meta or (0, tokens.shape[0], None)
        ids = getattr(meta, "request_ids", None)
        ids = list(ids[num_contexts:num_seqs]) if ids is not None else None
        self._s4_dump_line({"kind": "tokens", "step": self._s4_dump_step, "request_ids": ids, "s4": tokens.tolist(),
                            "stock": stock.tolist(), "equal": bool(torch.equal(tokens.long(), stock))})  # fmt: skip
        self._s4_dump_step += 1

    def _s4_logits_gate_dump(
        self, draft_model, shard_logits, inputs, spec_metadata, biased
    ) -> None:
        """Gate-only (eager): the in-tree chain on the gathered logits of this same step (full-vocab bias), compared
        bitwise with S4-logits' gathered biased logits, one JSON line per step and rank."""
        from tensorrt_llm._torch.distributed import allgather

        full = allgather(shard_logits.contiguous(), self.mapping, dim=-1)
        if getattr(self, "_s4_fp32", True):
            full = full.float()  # the in-tree chain's input: LogitsProcessor returns logits.float()
        stock = super()._refine_block_logits(draft_model, full, inputs, spec_metadata)
        ours = allgather(biased.contiguous(), self.mapping, dim=-1)
        num_contexts, num_seqs, meta = self._s4_step_meta or (0, biased.shape[0], None)
        ids = getattr(meta, "request_ids", None)
        ids = list(ids[num_contexts:num_seqs]) if ids is not None else None
        mism = int((ours != stock).sum())
        self._s4_dump_line({"kind": "logits", "step": self._s4_dump_step, "request_ids": ids,
                            "shape": list(ours.shape), "dtype": str(ours.dtype), "mismatch": mism,
                            "equal": mism == 0 and ours.dtype == stock.dtype,
                            "batched": S4_LOGITS_BATCHED})  # fmt: skip
        self._s4_dump_step += 1

    def _s4_dump_line(self, rec: dict) -> None:
        rec["rank"] = self.mapping.tp_rank
        path = os.path.join(self._s4_dump_dir, f"rank{self.mapping.tp_rank}.jsonl")
        with open(path, "a") as f:
            f.write(json.dumps(rec) + "\n")

    def sample_draft_tokens(self, logits, spec_metadata, batch_size, **kwargs):
        tokens = self._s4_tokens
        if tokens is not None and logits.dim() == 3 and tokens.shape == logits.shape[:2]:
            self._s4_tokens = None
            return tokens if tokens.dtype == torch.int32 else tokens.to(torch.int32)
        return super().sample_draft_tokens(logits, spec_metadata, batch_size, **kwargs)
