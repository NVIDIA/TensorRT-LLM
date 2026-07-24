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
"""TensorRT-LLM PyTorch bring-up of the Inkling text tower (``InklingCausalLLM``).

Scope: the text decoder that drives the GSM8K/MMLU accuracy gates on the NVFP4
checkpoint. Audio / vision / MTP are intentionally deferred (their weights are
accounted as unused). See ``configs/inkling.py`` for the config and
``checkpoints/hf/inkling_weight_mapper.py`` for the HF→TRT weight mapping.

Numeric ground truth is the HF reference
(``codes/transformers/.../models/inkling/modeling_inkling.py``); the NVFP4
serving/quant path mirrors the SGLang reference
(``codes/sglang/.../models/inkling*``).

Architecture summary (all primary-source verified against the checkpoint):
  * RoPE-free attention with per-head q/k RMSNorm and score scale ``1/head_dim``.
  * Learned relative-position bias (``RelLogitsProj``), added pre-softmax as a
    ``score_mod`` inside the Inkling Triton attention kernels (prefill + paged
    decode); see ``attention_backend/inkling_triton.py``.
  * Hybrid layers: 55 local sliding-window (win=512, 16 kv-heads) + 11 global
    full-causal (8 kv-heads). Global layers apply log-scaling tau (a no-op below
    128k tokens, still implemented for correctness).
  * Four causal short convolutions per layer (k, v inside attention before the
    k/q norm; one post-attention and one post-MLP on the residual stream).
  * Sigmoid-gated MoE, top-6 of 256 routed experts with an additive selection
    bias, log-sigmoid renorm over the selected-routed *plus* two shared logits,
    scaled by ``route_scale * global_scale``. Layers 0/1 are dense MLP.
  * Routed experts for layers 3..65 are NVFP4; layer-2 experts and everything
    else are bf16.
  * muP: divide hidden states by ``logits_mup_width_multiplier`` before the head;
    slice logits to ``unpadded_vocab_size``. ``embed_norm`` folds onto embeddings.
"""

import copy
import os
from collections import namedtuple
from dataclasses import dataclass
from typing import List, Optional

import torch
from torch import nn

from tensorrt_llm._torch.attention_backend import AttentionMetadata
from tensorrt_llm._torch.attention_backend.inkling_triton import (
    build_page_table, inkling_decode_attention, inkling_prefill_attention,
    write_kv_cache_hnd)
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm.logger import logger
from tensorrt_llm._torch.models.modeling_utils import (DecoderModel,
                                                       DecoderModelForCausalLM,
                                                       filter_weights,
                                                       register_auto_model)
from tensorrt_llm._torch.modules.embedding import Embedding
from tensorrt_llm._torch.modules.fused_moe import (BaseMoeRoutingMethod,
                                                   RoutingMethodType,
                                                   create_moe)
from tensorrt_llm._torch.modules.linear import (Linear, TensorParallelMode,
                                                WeightMode,
                                                WeightsLoadingConfig)
from tensorrt_llm._torch.modules.mamba.causal_conv1d import (
    causal_conv1d_fn, causal_conv1d_update)
from tensorrt_llm._torch.modules.qk_norm_attention import QKNormRoPEAttention
from tensorrt_llm._torch.modules.rms_norm import RMSNorm
from tensorrt_llm._utils import prefer_pinned

from ..configs.inkling import InklingConfig, InklingTextConfig

# Per-layer, per-request short-conv state carried across decode steps: the four
# causal short convolutions of one Inkling decoder layer. Each field is a
# ``[num_req, channels, sconv_kernel_size - 1]`` buffer holding the previous
# ``kernel_size - 1`` pre-conv inputs (oldest first): ``k``/``v`` for the
# attention k/v short-convs (channels = num_kv_heads * head_dim, TP-sharded) and
# ``attn``/``mlp`` for the post-attention / post-MLP residual-stream short-convs
# (channels = hidden_size, replicated). The generation phase reads these,
# convolves the one new token, and rolls the window forward IN PLACE (``copy_``),
# so the buffers keep stable addresses across decode steps and CUDA-graph replay;
# the context phase passes ``conv_state=None`` and every short-conv runs its
# stateless full-sequence causal conv. This is exactly the state the runtime
# short-conv cache carries per request alongside the paged KV cache (crit8).
InklingConvState = namedtuple("InklingConvState", ["k", "v", "attn", "mlp"])


# ---------------------------------------------------------------------------
# Batched-decode divergence localizer (env-gated, zero cost when unset)
# ---------------------------------------------------------------------------
# Set INKLING_DIVERGE_CHECK=1 to localize the served batched-decode corruption
# (fair served GSM8K nc=4 = 0.60 vs nc=1 correct; the fixed-batch repro shows 4
# IDENTICAL requests forking at decode step ~14+). On a batch of identical
# requests every generation row MUST stay bit-identical, so this walks the decode
# stack and reports the first (decode step, layer, sub-op) where identical-input
# rows first produce a DIVERGENT output. It also reads the carried short-conv pool
# state at the batch's generation slots, so a divergence can be attributed to a
# genuine per-slot STATE bug (divergent carried conv state) vs benign fused-kernel
# non-determinism (a tiny 1-2 ULP diff in a stateless op that reads identical
# state). Prints only from tp_rank 0. Reset each new prefill so the step index is
# per generation episode.
_INK_DIVERGE = {"on": None, "step": 0, "reported": False}


def _ink_diverge_on() -> bool:
    d = _INK_DIVERGE
    if d["on"] is None:
        d["on"] = bool(os.environ.get("INKLING_DIVERGE_CHECK"))
    return d["on"]


def _ink_rowdiff(t: Optional[torch.Tensor], ctx=None) -> float:
    """Max abs difference of any request from request 0 (0.0 => bit-identical).

    Decode mode (``ctx is None``): ``t`` is ``[num_req, ...]`` (one row per
    generation request); compare each row to row 0.

    Prefill/context mode (``ctx=(num_req, seqlen)``): ``t`` is the packed
    ``[num_req*seqlen, ...]`` context activation for ``num_req`` IDENTICAL
    prompts of equal length ``seqlen``; reshape to ``[num_req, seqlen, ...]`` and
    compare each request's whole span to request 0's. This is the invariant a
    batch of identical prefills must hold, and the packed varlen layout means a
    plain dim-0 rowdiff would wrongly compare token 0 to token 1 within a
    sequence -- the reshape compares request-vs-request instead.
    """
    if t is None:
        return 0.0
    if ctx is not None:
        num_req, seqlen = ctx
        if num_req < 2 or t.shape[0] != num_req * seqlen:
            return 0.0
        r = t.detach().float().reshape(num_req, seqlen, *t.shape[1:])
        return (r - r[0:1]).abs().max().item()
    if t.shape[0] < 2:
        return 0.0
    r0 = t[0:1].detach().float()
    return (t.detach().float() - r0).abs().max().item()


# feedback #17 op-level B2 bisection: intra-layer op boundaries in forward order.
# The op fingerprint (``InklingModel._ink_fp_ops``) is a
# ``[num_layers, N_INK_FP_OPS, 4]`` capture-safe STATS buffer -- for every op it
# stores ``[nonfinite_count, max_abs, 0, numel]`` computed element-wise on-device
# and ``copy_``'d INTO the decode graph. Stats (not the raw vector) so the buffer
# is shape-agnostic: it fingerprints the non-hidden-width attention internals
# (q/k/v/rel_logits) alongside the hidden-width residual points, and
# ``nonfinite_count`` names the FIRST tensor that is finite in eager but
# NON-FINITE under CUDA graph regardless of shape. The chain walks the feedback
# #17 order: attn_norm -> QK(q,k,v,rel) -> softmax/PV/KV-page(attn_kernel) ->
# TP all-reduce(o_proj_out) -> attn_sconv -> h_attn. Order MUST match
# inkling_fp_ops_analyze.py OP_NAMES.
#
# HEISENBUG GUARD (iter-79 -- fixes the iter-75/76/77/78 NO_NONFINITE + onset-
# unprobed rejects). Job data proved the op probe is NOT a passive observer:
# op-probing a global-attention layer PROTECTS it, moving the B2 nan onset to the
# first UNPROBED global layer (no probe -> onset L5, job 5558469; probe globals
# 5/11/17 -> onset L23, job 5560480; probe ALL globals -> fully SUPPRESSED, job
# 5561260 -- onset pushed past layer 65). Cause: the earlier fingerprint
# allocated ~3 numel-sized temporaries per op (a reshape-copy of the possibly
# non-contiguous q/k/v slice, ``.to(fp32)``, the ``isfinite`` bool, the ``abs``
# float) IN THE MIDDLE of the global-attention kernel path, shifting the
# CUDA-graph memory pool exactly where B2's capture-baked stale-buffer read
# lives. So chasing the onset with a layer cap (probe layer <= N) is futile --
# the probe always moves the onset off the probed set. The iter-79 fix instead
# makes the fingerprint ALLOCATION-FREE (see ``_ink_fp_stat``): a pre-allocated
# fp32 scratch + in-place ``abs_`` + a scalar ``amax``, allocating NOTHING
# numel-sized -- strictly lighter than the per-layer residual ``_ink_fp``
# ``.to(fp32)`` copy that is already proven CUDA-graph-transparent (5558469
# reproduces at natural onset L5 with it active). Transparent => no protection =>
# B2 reproduces at its NATURAL onset layer, which is itself op-probed, so ALL 11
# global-attention layers can be probed at once (any global onset is captured).
# Retained from the lean iter-77 probe: GLOBAL-attention layers only (55
# local/SWA skipped), the ``o_proj_local`` extra GEMM (op 6) left UNMEASURED
# (matmul-vs-all-reduce discriminated via cross-rank attn_kernel/o_proj_out
# finiteness in FP_OPS_XRANK), and mlp-side ops 10/11/12 UNMEASURED (B2 is
# attention-born; the ``_ink_fp`` residual covers the post-mlp stream).
_INK_FP_OP_NAMES = (
    "attn_norm",     # 0 decoder: pre-attention RMSNorm output
    "attn_q",        # 1 attention: q after qkv-proj + qk-norm
    "attn_k",        # 2 attention: k after kv-sconv + qk-norm
    "attn_v",        # 3 attention: v after kv-sconv
    "attn_rel",      # 4 attention: relative-position bias rel_logits
    "attn_kernel",   # 5 attention: paged decode-kernel output (softmax/PV/KV-page read)
    "o_proj_local",  # 6 UNMEASURED (extra GEMM dropped -- Heisenbug); use XRANK
    "o_proj_out",    # 7 attention: o_proj output POST all-reduce (= h_core)
    "attn_sconv",    # 8 decoder: attention short-conv output (h_asc)
    "h_attn",        # 9 decoder: post-attention residual (residual + h_asc)
    "mlp_norm",      # 10 UNMEASURED (mlp-side; residual buffer covers downstream)
    "moe_out",       # 11 UNMEASURED (mlp-side; residual buffer covers downstream)
    "mlp_sconv",     # 12 UNMEASURED (mlp-side; residual buffer covers downstream)
)
N_INK_FP_OPS = len(_INK_FP_OP_NAMES)
_INK_FP_STAT_W = 4  # [nonfinite_count, max_abs, l2, numel]


def _ink_fp_stat(slot: torch.Tensor, t: torch.Tensor,
                 scratch: Optional[torch.Tensor] = None) -> None:
    """Capture-safe finiteness fingerprint of ``t`` written into ``slot`` (a
    length-4 fp32 view): ``[nonfinite_flag, max_abs, 0, numel]``.

    ALLOCATION-FREE when ``scratch`` (a pre-allocated fp32 1-D buffer sized
    ``>= t.numel()``) is supplied: the per-op work is ``copy_`` (cast+gather into
    the scratch) -> in-place ``abs_`` -> a scalar ``amax`` reduction, so NOTHING
    numel-sized is allocated inside the captured decode graph. Only tiny scalars
    (the amax result + its ``isfinite``) are created -- strictly lighter than the
    per-layer residual ``_ink_fp`` ``.to(fp32)`` copy that is already proven
    CUDA-graph-transparent (job 5558469 reproduces B2 at its natural onset layer 5
    with that residual probe active). See the HEISENBUG GUARD note above for why
    allocation-freedom is load-bearing: the earlier ~3-numel-temp-per-op
    fingerprint shifted the CUDA-graph memory pool and PROTECTED whichever global
    layer it probed, moving the nan onset off the probed set.

    ``nonfinite_flag`` (>0) is the FIRST-op nan detector for ANY tensor shape:
    ``max_abs = amax(|t|)`` is nan if ``t`` has any nan and +inf if ``t`` has any
    inf, so ``~isfinite(max_abs)`` flags any nonfinite element regardless of
    shape. Everything is device-side (no ``.item()``/``.cpu()`` host sync) so it
    is CUDA-graph-capture-safe. Pass the already-sliced last row (decode: the last
    generation request)."""
    tv = t.detach()
    n = tv.numel()
    if scratch is not None and n <= scratch.numel():
        sv = scratch[:n].view(tv.shape)   # view of pre-alloc scratch (no alloc)
        sv.copy_(tv)                       # cast->fp32 + gather (no numel alloc)
        sv.abs_()                          # in-place abs (no alloc)
        m = sv.amax()                      # scalar reduction (no host sync)
        slot[0].copy_((~torch.isfinite(m)).to(torch.float32))
        slot[1].copy_(torch.nan_to_num(m, nan=0.0, posinf=0.0, neginf=0.0))
        slot[3].fill_(float(n))
        return
    # Fallback (no scratch supplied, or op larger than the scratch -- does not
    # happen for the probed Inkling ops): the original small-alloc path.
    tf = tv.reshape(-1).to(torch.float32)
    slot.copy_(
        torch.stack([
            (~torch.isfinite(tf)).sum().to(torch.float32),
            tf.abs().amax(),
            tf.new_zeros(()),
            tf.new_full((), float(tf.numel())),
        ]))


# ===========================================================================
# feedback #18 -- NO-PROBE capture-time buffer bisection for B2 (Stage 6).
#
# WHY (supersedes the feedback #17 op-probe as the FIRST tactic). The in-graph
# op fingerprint above (``INKLING_FP_OPS``) is a Heisenbug: every element-wise
# stat it records into the decode graph perturbs the CUDA-graph memory pool and
# MOVES/SUPPRESSES the nan (no probe -> onset L5, job 5558469; probe 5/11/17 ->
# L23, 5560480; probe all globals -> suppressed, 5561260). That layout
# sensitivity is itself the tell: B2 is a CAPTURE-TIME fault from a
# stale/UNINITIALIZED buffer or a captured pointer/offset in the global-attention
# decode path (nan present from the FIRST replay at kv_pos=5, BEFORE any KV-page
# boundary; the captured graph HOLDS it; eager is finite at identical metadata).
#
# So this route carries NO in-graph fingerprint. The pass/fail signal is the real
# free-run output itself (B2 = collapse to a token-0 repeat; fixed = coherent
# text -- driven by ``inkling_fp_localize_test.py`` with ``INKLING_FP`` UNSET).
# ``INKLING_B2_FIX`` selects ONE candidate init/refresh toggle per run; each is a
# distinct hypothesis about which captured buffer is stale, and each is a
# candidate FIX if it flips collapse -> coherent while eager stays finite.
#
# CAPTURED-BUFFER ENUMERATION (global-attention decode, ``_run_generation`` +
# ``inkling_decode_attention``). Every tensor the captured decode kernel reads:
#   * ``q`` / ``rel_logits``          transient, RE-computed in-graph each replay
#                                     (qkv_proj / einsum) -> not stale.
#   * ``k_cache`` / ``v_cache``       persistent KVCacheManagerV2 pool views;
#                                     stable pointer, contents read LIVE. If the
#                                     pool is allocated uninitialized (poisoned),
#                                     a decode read of an unwritten (page,slot)
#                                     returns nan -> candidate ``zero_kvpool``.
#   * ``meta.seq_lens``/``page_table``stable per-layer buffers (InklingDecodeMeta)
#                                     refreshed EAGERLY each step; only ``[:num_gen]``
#                                     is written, so a stale padding/prior-batch
#                                     row could drive the in-graph KV scatter to a
#                                     wrong page -> candidate ``full_meta`` (memset
#                                     the whole cap each refresh). Their H2D copy
#                                     is ``non_blocking`` -> a capture/replay read
#                                     could race it -> candidate ``sync_meta``.
#   * decode kernel out ``o``         ``torch.empty_like(q)`` -- a FRESH transient
#                                     grabbed from the graph pool each forward, so
#                                     its address moves with the pool layout (the
#                                     exact Heisenbug knob). Candidate
#                                     ``persist_out`` gives it a stable,
#                                     eagerly-zeroed persistent buffer instead.
# The four above were all RULED OUT (feedback #19). The leading unexplored
# feedback #19 candidate class is the CUDA-graph PADDING metadata:
#   * padding/dummy generation row     the ``max_batch_size=8`` decode graph pads
#     KV scatter                       the single real request up to the bucket with
#                                      dummy rows. The DIAG (job 5567187) MEASURED a
#                                      padded replay batch as {row0: real, sl=511,
#                                      16 pages} + {row1..7: dummy, sl=1, one page
#                                      each, e.g. 99/110/121/.. and sometimes page 0}.
#                                      Each dummy still runs the in-graph KV scatter
#                                      ``k_cache[pages,:,offs,:] = k`` at slot 0 of
#                                      its page, writing its pool-transient (garbage,
#                                      possibly non-finite) k/v there. If a real
#                                      request ever reads one of those pages a
#                                      capture-time nan results -- at kv_pos<=5,
#                                      before any page boundary, memory-layout-
#                                      sensitive (the dummy's hidden state is
#                                      pool-transient garbage), and INVISIBLE to
#                                      compute-sanitizer initcheck (the slot was
#                                      written, so the read is of initialised-but-
#                                      corrupted memory, not an uninitialised read).
#                                      Candidate ``pad_scatter`` redirects the dummy
#                                      rows to a scratch page that no real row uses,
#                                      EAGERLY in :meth:`refresh` (zero in-graph op,
#                                      so a collapse->coherent flip is a real fix not
#                                      a pool-perturbation artifact). CORRECTED
#                                      detector: a dummy is a ``seq_len==1`` row in a
#                                      PADDED CUDA-graph batch (``is_graph and
#                                      num_gen>1``); iter-88 keyed on ``seq_len==1``
#                                      alone and so also mis-redirected the sole real
#                                      row of an eager ``num_gen==1`` warmup step
#                                      (sl==1 there too), which made its NO_LEAD
#                                      invalid (iter-90 confound analysis).
# Toggles are env-gated and BYTE-UNCHANGED when ``INKLING_B2_FIX`` is unset.
_B2_FIX_NAMES = ("zero_kvpool", "persist_out", "full_meta", "sync_meta",
                 "pad_scatter")


def _b2_fix_active(name: str) -> bool:
    """True iff the feedback #18 no-probe B2 candidate ``name`` is selected via
    ``INKLING_B2_FIX`` (comma-separated). Single ``dict`` lookup -> zero cost when
    the env is unset (the production path is byte-unchanged)."""
    v = os.environ.get("INKLING_B2_FIX")
    if not v:
        return False
    return name in v.split(",")


def _ink_report_divergence(num_layers: int, dsink: dict,
                           inputs_embeds: torch.Tensor,
                           out: torch.Tensor) -> None:
    """Print the first (step, layer, sub-op) where identical decode rows fork.

    ``dsink[i]`` holds per-sub-op row divergences for layer ``i`` (see
    :meth:`InklingDecoderLayer.forward`). Called only from tp_rank 0 on a
    pure-generation forward of a batch of identical requests. Emits one greppable
    TRACE line per step and one ONSET line the first time the final hidden state
    diverges, attributing it to the first diverging sub-op and reporting whether
    the carried short-conv pool state had already diverged coming in.
    """
    step = _INK_DIVERGE["step"]
    d_embed = _ink_rowdiff(inputs_embeds)
    final_d = _ink_rowdiff(out)
    # First layer + sub-op whose output diverged (sub-ops in execution order,
    # so the first nonzero pins the origin op within the stack).
    first = None
    for i in range(num_layers):
        rec = dsink.get(i)
        if rec is None:
            continue
        for op in ("attn_core", "attn_sconv", "mlp_core", "mlp_sconv"):
            if rec[op] > 0.0:
                first = (i, op, rec)
                break
        if first is not None:
            break
    # First layer whose carried pre-update conv-pool state already diverged.
    state_first = None
    for i in range(num_layers):
        rec = dsink.get(i)
        if rec is not None and any(v > 0.0 for v in rec["pre_state"]):
            state_first = i
            break
    op_str = ("L%d/%s=%.2e" % (first[0], first[1], first[2][first[1]])
              if first else "none")
    if step <= 80 or final_d > 0.0:
        print("INKLING_DIVERGE_TRACE step=%d nrows=%d d_embed=%.3e final_d=%.3e "
              "state_first=%s op_first=%s"
              % (step, out.shape[0], d_embed, final_d,
                 ("L%d" % state_first) if state_first is not None else "none",
                 op_str),
              flush=True)
    if final_d > 0.0 and not _INK_DIVERGE["reported"]:
        _INK_DIVERGE["reported"] = True
        if first is not None:
            rec = first[2]
            print("INKLING_DIVERGE_ONSET step=%d d_embed=%.3e final_d=%.3e "
                  "first_layer=%d first_subop=%s d_in=%.3e attn_core=%.3e "
                  "attn_sconv=%.3e mlp_core=%.3e mlp_sconv=%.3e pre_state=%s"
                  % (step, d_embed, final_d, first[0], first[1], rec["d_in"],
                     rec["attn_core"], rec["attn_sconv"], rec["mlp_core"],
                     rec["mlp_sconv"],
                     ",".join("%.2e" % v for v in rec["pre_state"])),
                  flush=True)
        else:
            print("INKLING_DIVERGE_ONSET step=%d d_embed=%.3e final_d=%.3e "
                  "first_layer=-1 (final diverged but no per-layer sub-op did; "
                  "final-norm / logits / gather path)"
                  % (step, d_embed, final_d),
                  flush=True)


def _ink_report_prefill(num_layers: int, dsink: dict,
                        inputs_embeds: torch.Tensor, out: torch.Tensor,
                        ctx) -> None:
    """Print the first (layer, sub-op) where identical PROMPTS diverge in prefill.

    The decode localizer (:func:`_ink_report_divergence`) only fires on pure
    generation forwards, so a residual divergence seeded during the context
    forward (identical prompts prefilling to different logits) shows up there
    only as a nonzero step-1 ``d_embed``. This reports it directly: ``ctx`` is
    ``(num_req, seqlen)`` for the identical-prompt context batch, ``dsink`` holds
    per-sub-op request-span divergences (already computed context-aware in
    :meth:`InklingDecoderLayer.forward`). The first nonzero sub-op pins the
    prefill origin -- dense-layer (0/1) ``attn_core`` => the prefill attention
    kernel; layer>=2 ``mlp_core`` => the context-phase MoE.
    """
    num_req, seqlen = ctx
    d_embed = _ink_rowdiff(inputs_embeds, ctx)
    final_d = _ink_rowdiff(out, ctx)
    first = None
    for i in range(num_layers):
        rec = dsink.get(i)
        if rec is None:
            continue
        for op in ("attn_core", "attn_sconv", "mlp_core", "mlp_sconv"):
            if rec[op] > 0.0:
                first = (i, op, rec)
                break
        if first is not None:
            break
    op_str = ("L%d/%s=%.2e" % (first[0], first[1], first[2][first[1]])
              if first else "none")
    print("INKLING_PREFILL_DIVERGE num_req=%d seqlen=%d d_embed=%.3e final_d=%.3e "
          "op_first=%s" % (num_req, seqlen, d_embed, final_d, op_str),
          flush=True)
    if first is not None:
        rec = first[2]
        print("INKLING_PREFILL_ONSET first_layer=%d first_subop=%s d_in=%.3e "
              "attn_core=%.3e attn_sconv=%.3e mlp_core=%.3e mlp_sconv=%.3e"
              % (first[0], first[1], rec["d_in"], rec["attn_core"],
                 rec["attn_sconv"], rec["mlp_core"], rec["mlp_sconv"]),
              flush=True)


class InklingConvStateCache:
    """Runtime-owned per-request short-conv state pool for the whole decoder.

    This is the runtime cache contract the plan calls for (Design Choice 5):
    the four causal short-convs of every decoder layer, carried per request
    across decode steps with the same lifetime as the paged KV cache -- NOT a
    model-local Python dict that would drift from scheduler/cache ownership.

    Per layer it allocates the four short-conv state buffers
    (:class:`InklingConvState`), each ``[max_batch, channels, kernel_size - 1]``
    holding the previous ``kernel_size - 1`` pre-conv inputs (oldest first). The
    k/v conv channels follow the fused-qkv k/v split (TP-sharded like
    ``InklingShortConv(tp_shard=True)``); the post-attention / post-MLP convs run
    on the full (all-reduced) hidden stream and are replicated. The buffers keep
    stable device addresses for their whole lifetime -- the fused
    ``causal_conv1d_update`` / ``causal_conv1d_fn`` ops mutate them IN PLACE at
    the per-request ``state_indices`` slots, so a captured CUDA graph replays
    cleanly (no realloc, no gather/scatter). ``state_indices`` is a single stable
    ``[max_batch]`` int32 CUDA buffer written in place per forward (the
    Mamba2Metadata stable-pointer pattern), so the runtime can alias it under
    graph capture.

    Slot ownership (request-id -> row) is a thin allocator here so the model side
    is validated end to end now; wrapping this pool in a ``BaseResourceManager``
    that shares the KV-cache request lifetime is the remaining runtime-
    provisioning step (registration in ``get_kv_cache_manager_cls`` + pyexecutor
    construction + TP=4 launch).
    """

    def __init__(self,
                 model_config: "ModelConfig[InklingTextConfig]",
                 max_batch_size: int,
                 device: torch.device,
                 dtype: torch.dtype = torch.bfloat16):
        # Accept either the text ``ModelConfig`` (runtime: InklingModel's own
        # config) or the top-level multimodal one (tests build from the full
        # checkpoint config); resolve to the text sub-config either way.
        config = model_config.pretrained_config
        config = getattr(config, "text_config", config)
        tp_size = model_config.mapping.tp_size
        kwin = config.sconv_kernel_size - 1
        self.max_batch_size = max_batch_size
        self.kwin = kwin

        def buf(channels):
            return torch.zeros(max_batch_size,
                               channels,
                               kwin,
                               device=device,
                               dtype=dtype)

        self._layers: List[InklingConvState] = []
        for i in range(config.num_hidden_layers):
            kv_dim = (config.layer_num_kv_heads(i) *
                      config.layer_head_dim(i)) // tp_size
            hidden = config.hidden_size
            self._layers.append(
                InklingConvState(k=buf(kv_dim),
                                 v=buf(kv_dim),
                                 attn=buf(hidden),
                                 mlp=buf(hidden)))
        # Stable per-request slot-index buffer (int32, CUDA). Refreshed in place
        # per forward -- EAGERLY, from input preparation, before CUDA-graph
        # capture/replay (see :meth:`write_state_indices`) -- so a captured
        # decode graph aliases it and every replay reads the current batch's
        # rows (Mamba2Metadata stable-pointer pattern).
        self.state_indices = torch.arange(max_batch_size,
                                          dtype=torch.int32,
                                          device=device)
        # Pinned host staging for that per-forward write: the eager input-prep
        # phase fills this and issues ONE async H2D copy into ``state_indices``.
        # Pinned so the copy is cheap and legal even under graph capture; kept in
        # lock-step size with ``state_indices`` across :meth:`_grow`.
        self.state_indices_cpu = torch.zeros(max_batch_size,
                                             dtype=torch.int32,
                                             pin_memory=prefer_pinned())
        self._slot_of = {}
        self._free = list(range(max_batch_size - 1, -1, -1))

    def layer_state(self, layer_idx: int) -> InklingConvState:
        """The four short-conv state buffers for ``layer_idx`` (pool views)."""
        return self._layers[layer_idx]

    def reset(self):
        """Zero every state buffer and release all slots (fresh sequences)."""
        for st in self._layers:
            for t in st:
                t.zero_()
        self._slot_of.clear()
        self._free = list(range(self.max_batch_size - 1, -1, -1))

    def slots_for(self, request_ids: List[int]) -> List[int]:
        """Map request ids to their (stable) pool rows, allocating new ones.

        Fresh requests get a zero-initialised slot; existing requests keep their
        row so their carried short-conv windows persist across decode steps.

        If a single forward presents more *fresh* requests than the pool has
        free rows, the pool grows to fit (see :meth:`_grow`). Steady-state
        serving is bounded by ``max_batch_size`` (+1 CUDA-graph pad row) and
        never triggers growth, but the one-time KV-cache estimation forward can
        exceed it: that dummy batch is sized to saturate ``max_num_tokens`` (and
        is replicated ``x tp_size`` under attention DP), independent of
        ``max_batch_size``. Growing there (instead of ``IndexError`` on an empty
        free list) lets estimation profile memory correctly, and because growth
        only happens in that eager estimation/warmup window the buffers a later
        CUDA graph captures are the final, pointer-stable ones.
        """
        num_new = sum(1 for r in request_ids if r not in self._slot_of)
        if num_new > len(self._free):
            self._grow(num_new - len(self._free))
        slots = []
        for r in request_ids:
            if r not in self._slot_of:
                slot = self._free.pop()
                self._slot_of[r] = slot
                for st in self._layers:
                    for t in st:
                        t[slot].zero_()
            slots.append(self._slot_of[r])
        return slots

    def _grow(self, extra: int):
        """Append ``extra`` fresh (zeroed) rows to every per-request buffer.

        Reallocates each layer's four short-conv state tensors and the shared
        ``state_indices`` scratch to ``max_batch_size + extra`` rows, copying the
        existing rows forward so any in-flight request keeps its carried window,
        and returns the new rows to the free list. Called only from
        :meth:`slots_for` when a batch needs more rows than the pool owns; see
        there for why that happens (KV-cache estimation / attention-DP), and why
        it is safe w.r.t. CUDA-graph pointer stability.
        """
        old = self.max_batch_size
        new = old + extra
        for i, st in enumerate(self._layers):
            grown = []
            for t in st:
                buf = torch.zeros(new,
                                  t.shape[1],
                                  t.shape[2],
                                  device=t.device,
                                  dtype=t.dtype)
                buf[:old].copy_(t)
                grown.append(buf)
            self._layers[i] = InklingConvState(*grown)
        self.state_indices = torch.arange(new,
                                          dtype=torch.int32,
                                          device=self.state_indices.device)
        # Keep the pinned host-staging buffer sized in lock-step, else the eager
        # H2D write in write_state_indices would index past its end.
        self.state_indices_cpu = torch.zeros(new,
                                             dtype=torch.int32,
                                             pin_memory=prefer_pinned())
        # New rows old..new-1 join the free list, popped ascending like __init__.
        self._free = list(range(new - 1, old - 1, -1)) + self._free
        self.max_batch_size = new

    def write_state_indices(self, request_ids: List[int],
                            is_graph: bool) -> List[int]:
        """Resolve ``request_ids`` to pool rows and publish them into the stable
        ``state_indices`` CUDA buffer -- the EAGER, pre-capture slot write.

        Returns the resolved slot list (context requests first, then
        generation, matching the packed batch order). The host->device copy goes
        through the pinned ``state_indices_cpu`` staging buffer so it is legal
        under CUDA-graph capture and non-blocking. Because a captured decode
        graph aliases ``state_indices`` (via the ``gen_indices`` view built in
        :meth:`InklingConvRuntime.from_metadata`), this MUST run every forward
        from eager input-prep -- NOT inside the captured ``model.forward`` --
        so each replay reads the current batch's rows rather than the stale
        capture-time ones.

        ``is_graph`` (``attn_metadata.is_cuda_graph``) guards pool-pointer
        stability. Growth reallocates ``state_indices`` and would strand a
        captured graph's aliased pointer, so it may only happen in the eager
        estimation/warmup window (``is_graph`` False). The pool is sized
        ``max_batch_size + 1`` >= any graph batch, so a graph forward never needs
        to grow; assert it to turn a latent pointer bug into a loud failure
        instead of silent decode corruption.
        """
        before = self.state_indices.data_ptr()
        slots = self.slots_for(request_ids)
        if is_graph and self.state_indices.data_ptr() != before:
            raise RuntimeError(
                "Inkling short-conv pool grew during CUDA graph capture/replay; "
                "the pool must be sized to the max graph batch up front (a grown "
                "pool strands the captured state_indices pointer).")
        n = len(slots)
        self.state_indices_cpu[:n].copy_(torch.tensor(slots,
                                                      dtype=torch.int32))
        self.state_indices[:n].copy_(self.state_indices_cpu[:n],
                                     non_blocking=True)
        return slots

    def free(self, request_ids: List[int]):
        for r in request_ids:
            slot = self._slot_of.pop(r, None)
            if slot is not None:
                self._free.append(slot)


class InklingConvStateManager:
    """Request-lifetime resource manager wrapping the short-conv state pool.

    This is the runtime-provisioning piece of Design Choice 5: it owns one
    :class:`InklingConvStateCache` and is registered in the executor's resource
    dict under ``ResourceManagerType.CONV_STATE_MANAGER`` (see
    ``pyexecutor/py_executor_creator.py``), so the four short-conv states of
    every decoder layer are carried per request with the same lifetime as the
    paged KV cache. The model fetches the pool from this manager each forward
    (:meth:`InklingForCausalLM.forward`).

    Pool rows are allocated lazily on first sight of a request id inside
    :meth:`InklingConvRuntime.from_metadata` (the model calls it per forward with
    the batch's request ids) and released in :meth:`free_resources` when a
    request completes -- keyed on the same ``LlmRequest.py_request_id`` the KV
    cache and ``attn_metadata.request_ids`` use, so slot ownership stays in
    lock-step with the KV cache. This is a plain duck-typed manager -- the
    ``ResourceManager`` container dispatches ``prepare_resources`` /
    ``free_resources`` / ``update_resources`` via ``hasattr`` -- so it needs no
    ``BaseResourceManager`` import at model-load time (and no import cycle
    through ``pyexecutor``).
    """

    def __init__(self,
                 model_config: "ModelConfig[InklingConfig]",
                 max_batch_size: int,
                 device: torch.device,
                 dtype: torch.dtype = torch.bfloat16):
        # +1 row for a CUDA-graph padding / dummy-request slot (mamba pattern):
        # padded decode batches admit up to max_batch_size real requests plus a
        # shared dummy row.
        self.cache = InklingConvStateCache(model_config, max_batch_size + 1,
                                           device, dtype)
        self.max_batch_size = max_batch_size

    # ---- BaseResourceManager duck-typed interface (container uses hasattr) ----
    def get_max_resource_count(self) -> int:
        return self.max_batch_size

    def get_needed_resource_to_completion(self, request) -> int:
        # One pool row per request for its whole lifetime; the KV cache is the
        # binding admission constraint, so a flat 1 keeps this from over-gating
        # the capacity scheduler.
        return 1

    def prepare_resources(self, scheduled_batch):
        # Rows are allocated per forward from the padded batch's request ids in
        # prepare_conv_runtime (called by the model engine's eager input-prep,
        # which sees the CUDA-graph padding this hook does not), so there is
        # nothing to pre-allocate here.
        pass

    def update_resources(self, scheduled_batch):
        pass

    def add_dummy_requests(self, request_ids):
        # CUDA-graph dummy / padding requests get zero-initialised rows on first
        # sight via slots_for (in write_state_indices), like real requests.
        pass

    def free_resources(self, request):
        rid = getattr(request, "py_request_id", None)
        if rid is not None:
            self.cache.free([rid])

    def shutdown(self):
        pass

    # ---- Model-facing eager entry point ---------------------------------------
    def prepare_conv_runtime(self, attn_metadata):
        """Resolve this batch's conv pool rows and build its per-forward split.

        Called EAGERLY by the model engine (``_prepare_tp_inputs``) -- before
        CUDA-graph capture/replay and before ``model.forward`` -- so the H2D
        state_indices write happens outside the captured region and each replay
        reads the current (padded) batch's rows. Returns ``(pool, conv_rt)`` for
        the model to consume via its ``conv_cache`` / ``conv_rt`` kwargs; the
        captured forward then performs no host->device slot copy.
        """
        conv_rt = InklingConvRuntime.build(attn_metadata, self.cache)
        return self.cache, conv_rt


@dataclass
class InklingConvRuntime:
    """Per-forward short-conv plumbing for the pool path (all layers share it).

    Splits the packed ``[context tokens | one-token generation]`` batch at the
    context boundary so each of the four short-convs seeds the pool for context
    requests (varlen ``causal_conv1d_fn``) and updates it in place for generation
    requests (``causal_conv1d_update``), exactly like the paged attention split
    in :meth:`InklingAttention._attention`. ``None`` selects the stateless
    full-sequence conv (focused replays without a cache).
    """

    num_ctx_tokens: int
    ctx_indices: Optional[torch.Tensor]  # int32 pool slots, context requests
    gen_indices: Optional[torch.Tensor]  # int32 pool slots, generation requests
    query_start_loc: Optional[torch.Tensor]  # int32 [n_ctx+1] varlen offsets
    has_initial_state: Optional[torch.Tensor]  # bool [n_ctx]

    @classmethod
    def build(cls, attn_metadata,
              cache: InklingConvStateCache) -> "InklingConvRuntime":
        """Eager entry point: publish this batch's slots, then build the split.

        Resolves the batch's request ids to pool rows and writes them into the
        stable ``state_indices`` buffer (:meth:`InklingConvStateCache.write_state_indices`),
        then builds the context/generation views (:meth:`from_metadata`). This
        is the single place that does BOTH steps; the runtime path reaches it via
        :meth:`InklingConvStateManager.prepare_conv_runtime` from the model
        engine's eager input-prep, so the host->device slot write lands outside
        the captured ``model.forward``.
        """
        is_graph = bool(getattr(attn_metadata, "is_cuda_graph", False))
        slots = cache.write_state_indices(list(attn_metadata.request_ids),
                                          is_graph)
        return cls.from_metadata(attn_metadata, cache, slots)

    @classmethod
    def from_metadata(cls, attn_metadata, cache: InklingConvStateCache,
                      slots: List[int]) -> "InklingConvRuntime":
        """Build the context/generation split from ALREADY-published pool rows.

        ``slots`` are the request-id -> pool-row assignments already written into
        ``cache.state_indices`` by :meth:`InklingConvStateCache.write_state_indices`;
        this method only slices views of that stable buffer and (for prefill)
        builds the varlen offset tensors. It performs NO host->device copy of
        ``state_indices``, so it is safe to run inside the captured
        ``model.forward``. The context/generation split mirrors the attention
        split: context requests first (each with its full new-token span), then
        one-token generation requests. Prefill-only tensors
        (``query_start_loc`` / ``has_initial_state``) are built only when
        ``num_contexts > 0`` -- never during decode-graph capture -- so no
        host->device copy is ever captured.
        """
        seq_lens = attn_metadata.seq_lens.tolist()
        num_contexts = attn_metadata.num_contexts
        state_indices = cache.state_indices
        device = state_indices.device
        num_ctx_tokens = sum(seq_lens[:num_contexts])
        ctx_indices = state_indices[:num_contexts] if num_contexts else None
        gen_indices = (state_indices[num_contexts:len(slots)]
                       if num_contexts < len(slots) else None)
        query_start_loc = has_initial_state = None
        if num_contexts:
            cu = torch.zeros(num_contexts + 1, dtype=torch.int32, device=device)
            cu[1:] = torch.tensor(seq_lens[:num_contexts],
                                  dtype=torch.int32,
                                  device=device).cumsum(0)
            query_start_loc = cu
            # Fresh prefill carries no prior conv window (chunked-prefill reuse
            # would set this per request from cached-token counts).
            has_initial_state = torch.zeros(num_contexts,
                                            dtype=torch.bool,
                                            device=device)
        return cls(num_ctx_tokens=num_ctx_tokens,
                   ctx_indices=ctx_indices,
                   gen_indices=gen_indices,
                   query_start_loc=query_start_loc,
                   has_initial_state=has_initial_state)


def _resolve_conv_runtime(resource_manager, attn_metadata):
    """Fallback fetch of the runtime short-conv pool + this forward's split.

    Looks up the registered :class:`InklingConvStateManager` in the executor's
    ``ResourceManager`` container and, when present, returns ``(pool, conv_rt)``
    via :meth:`InklingConvStateManager.prepare_conv_runtime`. Returns
    ``(None, None)`` when no conv manager is registered, leaving the model on its
    stateless focused-replay behavior.

    The runtime decode path pre-builds ``conv_cache`` / ``conv_rt`` EAGERLY in
    the model engine (so the captured ``model.forward`` does no host->device slot
    copy); this fallback only fires for eager, never-captured warmup paths that
    reach ``model.forward`` without the engine having pre-built the split.
    """
    from tensorrt_llm._torch.pyexecutor.resource_manager import \
        ResourceManagerType
    mgr = resource_manager.get_resource_manager(
        ResourceManagerType.CONV_STATE_MANAGER)
    if mgr is None:
        return None, None
    return mgr.prepare_conv_runtime(attn_metadata)


def _apply_sconv(sconv: "InklingShortConv", x: torch.Tensor,
                 pool_buf: Optional[torch.Tensor],
                 rt: Optional[InklingConvRuntime]) -> torch.Tensor:
    """Run one short-conv over a (possibly mixed) batch through the state pool.

    ``rt is None`` -> stateless full-sequence causal conv (focused replays).
    Otherwise the context slice seeds ``pool_buf`` (varlen prefill) and the
    generation slice updates it in place at ``rt.gen_indices`` (decode), then the
    two outputs are concatenated in packed order. ``pool_buf`` is this conv's
    ``[max_batch, channels, kernel-1]`` state buffer from
    :class:`InklingConvStateCache`.
    """
    if rt is None:
        return sconv(x)
    parts = []
    nctx = rt.num_ctx_tokens
    if nctx > 0:
        parts.append(
            sconv.forward(x[:nctx],
                          conv_state=pool_buf,
                          cache_indices=rt.ctx_indices,
                          query_start_loc=rt.query_start_loc,
                          has_initial_state=rt.has_initial_state,
                          is_decode=False))
    if x.shape[0] > nctx:
        parts.append(
            sconv.forward(x[nctx:],
                          conv_state=pool_buf,
                          cache_indices=rt.gen_indices,
                          is_decode=True))
    return parts[0] if len(parts) == 1 else torch.cat(parts, dim=0)


def _module_excluded_from_quant(model_config: ModelConfig, name: str) -> bool:
    """True if ``name`` (or an ancestor) is bf16, not NVFP4.

    This plain-NVFP4 checkpoint lists its bf16 modules in
    ``hf_quant_config.json`` ``quantization.exclude_modules`` (read into
    ``quant_config.exclude_modules`` by ``from_pretrained``) rather than in
    ``per_layer_quant_configs`` (only populated for MIXED_PRECISION checkpoints).
    ``QuantConfig.is_module_excluded_from_quantization`` walks the dotted
    ancestry, so a listed ``model.llm.layers.5.attn`` covers the qkv/o
    projections under it. Used to build attention (all ``.attn`` excluded) and
    layer-2 routed experts (``.mlp.experts`` excluded) as bf16.
    """
    qc = model_config.quant_config
    return (qc is not None and qc.exclude_modules is not None
            and qc.is_module_excluded_from_quantization(name))


# ----------------------------------------------------------------------------
# Routing method
# ----------------------------------------------------------------------------
def _inkling_trtllm_moe_backend() -> bool:
    """True when the routed NVFP4 experts should run on the trtllm-gen
    (blockScaleMoe) MoE kernel instead of the default CUTLASS backend.

    Gated by ``INKLING_MOE_BACKEND=TRTLLM``. The trtllm-gen kernel is the same
    family SGLang runs (``flashinfer_trtllm_routed``) and has a deterministic
    finalize/combine, which removes the CUTLASS fused-combine cross-row
    non-determinism that craters served nc>1 GSM8K. When the env is unset the
    default CUTLASS path is byte-for-byte unchanged.
    """
    return os.environ.get("INKLING_MOE_BACKEND", "").upper() == "TRTLLM"


def _moe_config_with_trtllm_backend(model_config: ModelConfig) -> ModelConfig:
    """Return a shallow ``ModelConfig`` copy whose ``moe_backend`` is ``TRTLLM``.

    ``ModelConfig`` freezes itself after construction (``_frozen=True``) and its
    ``__setattr__`` rejects every field except a small documented allowlist
    (``_frozen``/``extra_attrs``/``pretrained_config``/``quant_config``), so the
    naive ``mc = copy.copy(model_config); mc.moe_backend = "TRTLLM"`` raises
    ``AttributeError: Cannot modify ModelConfig.'moe_backend' - instance is
    frozen`` (the iter64 failure). Use the sanctioned escape hatch named in
    ``ModelConfig.__setattr__``: unfreeze the *copy*, retarget only the scalar
    ``moe_backend``, then re-freeze. ``copy.copy`` is a shallow copy, so the
    ``_frozen`` bool on the copy is independent of the original -- the global
    config the rest of the model shares stays frozen and byte-unchanged on the
    default CUTLASS backend; only the routed-expert ``create_moe`` build below
    sees the trtllm-gen selection.
    """
    moe_config = copy.copy(model_config)
    # '_frozen' is explicitly writable even on a frozen instance (see
    # ModelConfig.__setattr__); flip it on the copy, set the field, re-freeze.
    moe_config._frozen = False
    moe_config.moe_backend = "TRTLLM"
    moe_config._frozen = True
    return moe_config


class InklingMoeRoutingMethod(BaseMoeRoutingMethod):
    """Sigmoid gate + additive-bias top-k selection + log-sigmoid renorm.

    The renorm denominator spans the selected routed logits *and* the shared
    logits together (``shared_expert_sink``), so this cannot be expressed by the
    stock sigmoid/MiniMax routing methods. ``apply`` returns only the routed
    ``(topk_ids, topk_weights)`` needed by the fused MoE; the shared gammas come
    from the same joint renorm and are recomputed in :class:`InklingMoE` for the
    shared-expert branch (see :func:`inkling_joint_renorm`).
    """

    def __init__(self, top_k: int, num_experts: int, n_shared_experts: int,
                 callable_gate_bias, callable_global_scale, route_scale: float):
        super().__init__()
        self.top_k = top_k
        self.num_experts = num_experts
        self.n_shared_experts = n_shared_experts
        self._callable_gate_bias = callable_gate_bias
        self._callable_global_scale = callable_global_scale
        self.route_scale = route_scale
        # When the trtllm-gen MoE backend is selected, the routed experts run the
        # blockScaleMoe kernel with EXTERNALLY precomputed routing, so the routing
        # must be computed here (separated routing) and tagged with the Inkling
        # routing enum. CUTLASS (default) also computes routing via ``apply`` but
        # keeps ``Unspecified`` and integrated (non-separated) dispatch, so its
        # behavior is unchanged.
        self._trtllm_backend = _inkling_trtllm_moe_backend()

    def apply(self,
              router_logits: torch.Tensor,
              input_ids=None) -> tuple[torch.Tensor, torch.Tensor]:
        # router_logits: [num_tokens, num_experts + n_shared] in fp32.
        routed_w, topk_idx, _ = inkling_joint_renorm(
            router_logits.float(),
            gate_bias=self._callable_gate_bias(),
            global_scale=self._callable_global_scale(),
            route_scale=self.route_scale,
            top_k=self.top_k,
            num_routed=self.num_experts,
            n_shared=self.n_shared_experts,
        )
        return topk_idx.to(torch.int32), routed_w.to(torch.float32)

    @property
    def routing_method_type(self):
        # TRTLLM backend (INKLING_MOE_BACKEND=TRTLLM): tag the routing with the
        # dedicated Inkling enum so the blockScaleMoe kernel dispatches its
        # precomputed-routing branch (added in iter64: runner.h/runner.cu
        # ``InklingSinkRenorm``) -- permute + fp4 GEMM + deterministic finalize on
        # the topk_ids/topk_weights this method precomputes. Default (CUTLASS):
        # keep ``Unspecified`` so CUTLASS/VANILLA compute routing torch-side via
        # :meth:`apply` exactly as before (byte-unchanged).
        if self._trtllm_backend:
            return RoutingMethodType.InklingSinkRenorm
        return RoutingMethodType.Unspecified

    @property
    def requires_separated_routing(self) -> bool:
        # Force the trtllm-gen backend to compute routing here (via
        # :meth:`apply`) and pass precomputed (topk_ids, topk_weights) to the
        # kernel rather than a routing_logits tensor. Only when the trtllm-gen
        # backend is selected; CUTLASS keeps the default (False).
        return self._trtllm_backend


def inkling_joint_renorm(router_logits: torch.Tensor, gate_bias: torch.Tensor,
                         global_scale: torch.Tensor, route_scale: float,
                         top_k: int, num_routed: int, n_shared: int):
    """Exact Inkling router math (fp32). Mirrors HF ``InklingTopkRouter``.

    Returns ``(routed_weights [T, top_k], topk_idx [T, top_k], shared_gammas
    [T, n_shared])``. Selection uses ``sigmoid(routed) + bias``; the weights are
    a softmax over ``logsigmoid`` of the selected-routed-plus-shared *logits*,
    scaled by ``route_scale * global_scale``.
    """
    routed_logits = router_logits[..., :num_routed]
    shared_logits = router_logits[..., num_routed:num_routed + n_shared]

    scores = routed_logits.sigmoid()
    scores_for_choice = scores + gate_bias
    topk_idx = torch.topk(scores_for_choice, top_k, dim=-1, sorted=False)[1]

    topk_logits = torch.cat([routed_logits.gather(-1, topk_idx), shared_logits],
                            dim=-1)
    topk_log_probs = torch.nn.functional.logsigmoid(topk_logits)
    weights = torch.exp(topk_log_probs -
                        torch.logsumexp(topk_log_probs, dim=-1, keepdim=True))
    weights = weights * route_scale * global_scale

    routed_weights = weights[..., :top_k].contiguous()
    shared_gammas = weights[..., top_k:top_k + n_shared].contiguous()
    return routed_weights, topk_idx, shared_gammas


# ----------------------------------------------------------------------------
# Short convolution (four per layer)
# ----------------------------------------------------------------------------
class InklingShortConv(nn.Module):
    """Causal depthwise short convolution (kernel 4) with an internal residual.

    The weight matches the checkpoint layout ``[channels, 1, kernel]``. At
    prefill this runs :func:`causal_conv1d_fn`; at cached decode it runs
    :func:`causal_conv1d_update` against the per-request conv state carried by
    the state cache manager. ``conv_state`` (and the runtime metadata that
    selects the per-request slot) is threaded in by the caller; when it is
    ``None`` the module falls back to a self-contained causal convolution over
    the provided sequence (used by focused replay tests without a cache).

    Reference: HF ``InklingShortConvolution`` (fp32 conv, cast back) and SGLang
    ``inkling_common/sconv.py``.

    TP sharding (``tp_shard=True``): the k/v short convs act on the per-rank
    slice of the k/v stream produced by the fused qkv projection, so their
    channels are sharded by kv-head exactly like that projection. The checkpoint
    stores the *full* (unsharded) conv weight, so :meth:`load_weights` slices the
    rank's contiguous channel block -- the same pattern as the mamba mixer, which
    stores its depthwise conv in a column-parallel ``Linear``. The
    post-attention / post-MLP convs run on the full (all-reduced) hidden stream
    and are replicated (``tp_shard=False``).
    """

    def __init__(self,
                 channels: int,
                 kernel_size: int,
                 mapping=None,
                 tp_shard: bool = False):
        super().__init__()
        self.kernel_size = kernel_size
        self.tp_size = mapping.tp_size if (mapping is not None
                                           and tp_shard) else 1
        self.tp_rank = mapping.tp_rank if (mapping is not None
                                           and tp_shard) else 0
        assert channels % self.tp_size == 0, (channels, self.tp_size)
        self.channels_full = channels
        # Local (this rank's) channel count -- what the forward actually sees.
        self.channels = channels // self.tp_size
        # Depthwise conv weight, one filter per (local) channel: [channels,1,kernel].
        self.weight = nn.Parameter(torch.empty(self.channels, 1, kernel_size))
        self.register_parameter("bias", None)

    def load_weights(self, weights, allow_partial_loading: bool = False):
        """Copy the (full) checkpoint conv weight, slicing this rank's channels.

        The loader routes here (``hasattr(module, 'load_weights')``) with a
        one-element list of ``{'weight': [channels_full, 1, kernel]}``. For the
        replicated post-attn/post-MLP convs ``tp_size == 1`` and the full tensor
        is copied; for the sharded k/v convs the rank's contiguous channel block
        is taken (kv-head aligned, matching the fused qkv k/v split).
        """
        w = weights[0]["weight"]
        if self.tp_size > 1:
            w = w.chunk(self.tp_size, dim=0)[self.tp_rank]
        self.weight.data.copy_(w[:])

    def forward(self,
                x: torch.Tensor,
                conv_state: Optional[torch.Tensor] = None,
                cache_indices: Optional[torch.Tensor] = None,
                query_start_loc: Optional[torch.Tensor] = None,
                has_initial_state: Optional[torch.Tensor] = None,
                is_decode: bool = False) -> torch.Tensor:
        """x: [num_tokens, channels]; internal residual ``y = conv(x) + x``.

        The stateless (no-cache) branch runs the conv in fp32 (per the source);
        the fused cached branches run in the input dtype (the ``causal_conv1d``
        ops require ``weight.dtype == x.dtype``, so the fp32 conv Parameter is
        cast to ``x.dtype`` and ``conv_state`` -- the bf16 state pool -- matches).
        Output is cast back to the input dtype. ``conv_state`` is updated in place
        by the fused ops.
        """
        in_dtype = x.dtype
        residual = x
        # Fused ops need weight and state in the input dtype (bf16); the fp32
        # conv Parameter is cast here (the stateless branch below uses fp32).
        w = self.weight.squeeze(1).to(x.dtype)  # [channels, kernel]
        if conv_state is not None and is_decode:
            # Cached single/short-step decode: [num_tokens, channels] -> op.
            # ``causal_conv1d_update`` writes its output IN PLACE into its ``x``
            # argument (and returns that same tensor), so it must be given a
            # COPY -- otherwise it clobbers ``residual`` (which aliases ``x``)
            # and the internal residual becomes ``conv(x) + conv(x)`` instead of
            # ``conv(x) + x``. (The prefill branch is safe: ``transpose().
            # contiguous()`` already copies. This decode-only aliasing was the
            # multi-step-decode K/V divergence.)
            y = causal_conv1d_update(x.clone(),
                                     conv_state,
                                     w,
                                     self.bias,
                                     activation=None,
                                     conv_state_indices=cache_indices)
        elif conv_state is not None:
            # Prefill with cache: varlen [channels, total_tokens].
            xt = x.transpose(0, 1).contiguous()
            y = causal_conv1d_fn(xt,
                                 w,
                                 self.bias,
                                 query_start_loc=query_start_loc,
                                 cache_indices=cache_indices,
                                 has_initial_state=has_initial_state,
                                 conv_states=conv_state,
                                 activation=None)
            y = y.transpose(0, 1).contiguous()
        else:
            # No cache: self-contained causal depthwise conv over the sequence.
            xt = x.float().transpose(0, 1).unsqueeze(0)  # [1, channels, T]
            y = torch.nn.functional.conv1d(xt,
                                           self.weight.float(),
                                           bias=None,
                                           padding=self.kernel_size - 1,
                                           groups=self.channels)
            y = y[..., :x.shape[0]].squeeze(0).transpose(0, 1)
        return (y.to(in_dtype) + residual).to(in_dtype)

    def forward_decode(self, x_new: torch.Tensor,
                       conv_state: torch.Tensor) -> tuple:
        """Single-step decode short conv with an explicit conv-state window.

        ``x_new`` is ``[num_req, channels]`` (one new token per request);
        ``conv_state`` is ``[num_req, channels, kernel_size-1]`` holding the
        previous ``kernel_size-1`` pre-conv inputs (oldest first). Returns
        ``(y_new [num_req, channels], updated_state)`` where the causal depthwise
        conv over ``[state | x_new]`` plus the internal residual is computed in
        fp32 (matching the prefill path), and the state is rolled forward.

        This is the generation-phase equivalent of the no-cache prefill conv:
        for the new token at position ``p`` it produces exactly the same output
        the full-sequence conv would at ``p`` (the last ``kernel_size`` inputs
        are ``[x[p-K+1..p]]``). The runtime carries ``conv_state`` in the state
        cache; focused replay tests seed it from the prefill's tail tokens.
        """
        in_dtype = x_new.dtype
        w = self.weight.squeeze(1).float()  # [channels, K]
        window = torch.cat([conv_state.float(),
                            x_new.float().unsqueeze(-1)],
                           dim=-1)  # [R,C,K]
        y = (window * w.unsqueeze(0)).sum(dim=-1)  # [R, C]
        new_state = window[..., 1:].to(in_dtype)  # [R, C, K-1]
        return (y.to(in_dtype) + x_new).to(in_dtype), new_state


# ----------------------------------------------------------------------------
# Attention
# ----------------------------------------------------------------------------
class InklingDecodeMeta:
    """Per-attention-layer STABLE GPU buffers for the generation-step decode
    metadata, refreshed EAGERLY (before CUDA-graph capture/replay) so the
    captured decode forward reads them with zero host->device copy.

    The Inkling Triton decode kernel needs, per generation request: the total KV
    length (``num_cached + 1``) and the physical page table. The stateless focused
    replays pass these explicitly; the real runtime used to build them from host
    lists INSIDE ``model.forward`` (``torch.tensor(..., device=cuda)`` +
    ``build_page_table``), which raises ``Cannot copy between CPU and CUDA tensors
    during CUDA graph capture`` under the enabled config. This class holds the two
    tensors in fixed-pointer GPU buffers and :meth:`refresh` overwrites their
    contents each step (mirroring :class:`InklingConvStateCache`'s
    ``state_indices`` publish and the base metadata's ``seq_lens_cuda`` in-place
    ``copy_``). One instance per :class:`InklingAttention` because
    ``KVCacheManagerV2.get_batch_cache_indices`` is per-layer (per pool_id /
    index_scale).

    ``page_table`` width is sized once to ``mgr.max_blocks_per_seq`` (the runtime
    KV-config bound), so a real sequence never overflows it. The row capacity
    grows elastically only when a batch oversubscribes it (the one-time KV-cache
    estimation forward can), and NEVER under CUDA graph -- growth would strand the
    captured pointer, so it raises loudly there instead.
    """

    def __init__(self, layer_idx: int):
        self.layer_idx = layer_idx
        self.max_pages: Optional[int] = None
        self.cap = 0
        self.seq_lens: Optional[torch.Tensor] = None  # [cap] int32 GPU total-KV
        self.page_table: Optional[torch.Tensor] = None  # [cap, max_pages] int32
        self.ready = False
        # feedback #18 no-probe B2 candidates. ``_owner`` is the InklingAttention
        # (set right after construction) so :meth:`refresh` can size the persistent
        # decode-output buffer eagerly. ``out_buf`` is the ``persist_out`` candidate
        # -- a stable, eagerly-zeroed replacement for the decode kernel's fresh
        # ``torch.empty_like`` output (whose pool-transient address is the Heisenbug
        # knob). Both stay ``None`` unless the matching toggle is selected.
        self._owner = None
        self.out_buf: Optional[torch.Tensor] = None
        # feedback #19 padding-metadata candidate ``pad_scatter``: total physical
        # page count of this layer's KV pool, resolved LAZILY and ONLY when the
        # toggle is selected (production path never calls this), so a CUDA-graph
        # padding/dummy generation row can be scattered to a reserved scratch page
        # instead of aliasing the real request's physical page 0.
        self.num_pages: Optional[int] = None
        # Guards for the INKLING_B2_DIAG batch-composition dump (eager, gated OFF
        # by default -> production byte-unchanged). Two independent guards: one for
        # the first eager/warmup refresh and one that fires up to 4x on the PADDED
        # CUDA-graph replay (is_graph & num_gen>1). Job 5566938 proved a single
        # one-shot guard logged only the num_gen=1 warmup and MISSED the padded
        # batch=8 replay -- the batch composition B2 actually depends on.
        self._diag_eager_logged = False
        self._diag_graph_n = 0

    def _ensure(self, num_gen: int, mgr, device, is_graph: bool) -> None:
        if self.max_pages is None:
            self.max_pages = max(1, int(mgr.max_blocks_per_seq))
        if self.seq_lens is not None and num_gen <= self.cap:
            return
        if is_graph and self.seq_lens is not None:
            raise RuntimeError(
                f"InklingDecodeMeta(layer={self.layer_idx}) would grow its stable "
                f"decode buffers during CUDA graph capture/replay (num_gen="
                f"{num_gen} > cap={self.cap}); the buffers are sized to the "
                f"scheduler batch, so this signals a capture-shape mismatch")
        self.cap = max(num_gen, self.cap)
        self.seq_lens = torch.ones(self.cap, dtype=torch.int32, device=device)
        self.page_table = torch.zeros((self.cap, self.max_pages),
                                      dtype=torch.int32,
                                      device=device)

    def _b2_fix_eager(self, device) -> None:
        """Apply the feedback #18 no-probe B2 candidates that must run EAGERLY (the
        captured forward only READS these stable buffers):

        * ``full_meta`` -- memset the WHOLE cap (not just ``[:num_gen]``) so a stale
          padding / prior-batch page-table row cannot drive the in-graph KV scatter
          to a wrong page.
        * ``persist_out`` -- size the stable, zeroed decode-output buffer HERE
          (never inside the captured forward), so the candidate adds no in-graph
          allocation or zero of its own that would itself perturb the pool layout
          and could "fix" B2 by perturbation (a false positive).

        No-op unless the matching toggle is selected. Split out of :meth:`refresh`
        so it is unit-testable without a full ``attn_metadata``."""
        if _b2_fix_active("full_meta"):
            self.seq_lens.fill_(1)
            self.page_table.zero_()
        if _b2_fix_active("persist_out") and self._owner is not None and (
                self.out_buf is None or self.out_buf.shape[0] < self.cap):
            o = self._owner
            self.out_buf = torch.zeros(self.cap,
                                       o.local_num_heads,
                                       o.head_dim,
                                       dtype=o.qkv_proj.weight.dtype,
                                       device=device)

    def refresh(self, attn_metadata, device) -> bool:
        """Publish this batch's generation decode metadata into the stable
        buffers. Returns whether a generation slice was prepared (``ready``)."""
        self.ready = False
        request_ids = attn_metadata.request_ids
        mgr = getattr(attn_metadata, "kv_cache_manager", None)
        if request_ids is None or mgr is None:
            return False
        num_contexts = attn_metadata.num_contexts
        num_gen = len(request_ids) - num_contexts
        if num_gen <= 0:
            return False
        gen_ids = request_ids[num_contexts:]
        num_cached = attn_metadata.kv_cache_params.num_cached_tokens_per_seq[
            num_contexts:]
        # Host-side block table for THIS layer (per-pool). This is the SAME call
        # the previous host path made inside forward; relocating it here (eager,
        # outside any captured region) is what makes the copy legal.
        block_ids = mgr.get_batch_cache_indices(gen_ids, self.layer_idx)
        is_graph = bool(getattr(attn_metadata, "is_cuda_graph", False))
        self._ensure(num_gen, mgr, device, is_graph)
        self._b2_fix_eager(device)
        # ``sync_meta``: force the metadata H2D copies synchronous (production is
        # ``non_blocking``) to test whether a captured/replayed decode read races
        # the async copy of the eagerly-published seq_lens/page_table.
        nb = not _b2_fix_active("sync_meta")
        # ``pad_scatter`` (feedback #19 padding-metadata hypothesis; detector
        # CORRECTED per iter-90's confound analysis): a CUDA-graph padded/dummy
        # generation row (num_cached==0 -> seq_len==1) still runs the in-graph KV
        # scatter, writing its pool-transient (garbage) k/v into whatever physical
        # page its page-table row names; if a real request later reads that page a
        # capture-time nan results. Redirect every dummy row to a scratch page that
        # NO real row in this batch uses.
        #
        # RELIABLE dummy detector (fixes the iter-88 confound): dummy rows exist
        # ONLY in a PADDED CUDA-graph batch (``is_graph and num_gen > 1``); the DIAG
        # (job 5567187) proved such a batch is always {1 real row sl=511} + {N dummy
        # rows sl=1}. In an eager or ``num_gen==1`` step the sole row is the real
        # request even when its sl==1 at graph warmup -- so gating on
        # ``is_graph and num_gen > 1`` never redirects a real row (the exact
        # mis-redirect that made iter-88's NO_LEAD invalid). Lazy + toggle-gated ->
        # the production path never calls ``get_buffers`` and stays byte-unchanged.
        scratch_page = None
        if _b2_fix_active("pad_scatter") and is_graph and num_gen > 1:
            if self.num_pages is None:
                try:
                    kvb = mgr.get_buffers(self.layer_idx, kv_layout="HND")
                    self.num_pages = int(kvb.shape[0])
                except Exception:  # noqa: BLE001 - diagnostic; fall back to no-op
                    self.num_pages = None
            if self.num_pages and self.num_pages > 1:
                # Pages the REAL rows (num_cached>0 -> sl>1) hold this step; a dummy
                # must never be scattered onto one of them. Pick the highest FREE
                # page as scratch (num_pages-1 itself can be a real page -- the DIAG
                # real row used page 242).
                real_pages = set()
                for j in range(num_gen):
                    if int(num_cached[j]) > 0:
                        real_pages.update(int(b) for b in block_ids[j]
                                          if int(b) >= 0)
                for p in range(self.num_pages - 1, -1, -1):
                    if p not in real_pages:
                        scratch_page = p
                        break
        # Build host staging (pinned) then ONE async copy into each stable buffer.
        pin = prefer_pinned()
        sl_host = torch.empty(num_gen, dtype=torch.int32, pin_memory=pin)
        for i in range(num_gen):
            sl_host[i] = int(num_cached[i]) + 1
        pt_host = torch.zeros((num_gen, self.max_pages),
                              dtype=torch.int32,
                              pin_memory=pin)
        for i, blocks in enumerate(block_ids):
            # ``pad_scatter``: ``scratch_page`` is non-None only in a padded
            # CUDA-graph batch (is_graph and num_gen>1), so ``sl_host[i]==1`` here
            # reliably means a dummy row (a real gen row has num_cached>=1 ->
            # seq_len>=2). Point its whole page-table row at the real-row-free
            # scratch page so the in-graph KV scatter cannot write the dummy's
            # (pool-transient) k/v into any physical page a real request reads.
            # Eager (no in-graph op) so a collapse->coherent flip is a real fix,
            # not a pool-perturbation artifact.
            if scratch_page is not None and int(sl_host[i]) == 1:
                pt_host[i, :] = scratch_page
                continue
            valid = [int(b) for b in blocks if int(b) >= 0]
            if valid:
                w = min(len(valid), self.max_pages)
                pt_host[i, :w] = torch.tensor(valid[:w], dtype=torch.int32)
        # INKLING_B2_DIAG (eager, gated OFF by default -> production byte-unchanged):
        # dump this batch's decode composition at the first global layer (5) so the
        # CUDA-graph padding structure B2 depends on is MEASURED, not guessed.
        # Per-row request id + seq_len + allocated pages reveal whether padding/dummy
        # rows exist, their seq_len (validating the pad_scatter seq_len==1 detector),
        # and whether they alias the real request's physical page 0 (row0). Log the
        # first eager refresh AND up to 4 PADDED graph refreshes (is_graph & num_gen>1)
        # -- the padded batch=8 replay is the composition that must be captured; a
        # single one-shot guard fired only on the num_gen=1 warmup (job 5566938).
        if os.environ.get("INKLING_B2_DIAG") == "1" and self.layer_idx == 5:
            is_pad = is_graph and num_gen > 1
            do_log = False
            if is_pad and self._diag_graph_n < 4:
                self._diag_graph_n += 1
                do_log = True
            elif not is_pad and not self._diag_eager_logged:
                self._diag_eager_logged = True
                do_log = True
            if do_log:
                parts = []
                for i in range(num_gen):
                    pgs = [int(b) for b in block_ids[i] if int(b) >= 0]
                    parts.append(
                        f"row{i}:id={gen_ids[i]!s} sl={int(sl_host[i])} pages={pgs}")
                print(
                    f"INKLING_B2_DIAG layer=5 num_gen={num_gen} is_graph={is_graph} "
                    f"num_pages={self.num_pages} | " + " | ".join(parts),
                    flush=True)
        self.seq_lens[:num_gen].copy_(sl_host, non_blocking=nb)
        self.page_table[:num_gen].copy_(pt_host, non_blocking=nb)
        self.ready = True
        return True


class InklingAttention(QKNormRoPEAttention):
    """RoPE-free attention with per-head q/k RMSNorm, k/v short-conv, and a
    learned relative-position bias applied as a Triton ``score_mod``.

    Reuses :class:`QKNormRoPEAttention` for the fused qkv/o projections and
    per-head q/k RMSNorm (``skip_rope=True`` gives qk-norm without RoPE), and
    owns the extra ``r`` projection, the k/v short convolutions, and the
    relative-logit projection. The attention *compute* itself runs through the
    Inkling Triton attention path (``attention_backend/inkling_triton.py``)
    rather than the base backend, because Inkling's learned relative bias is a
    per-(query,head,relative-distance) additive ``score_mod`` that no fused,
    CUDA-graph-safe TensorRT-LLM backend exposes:
      * ``cpp/.../common/attentionOp.cpp`` disables context FMHA for
        ``position_embedding_type == kRELATIVE`` (unfused MHA fallback);
      * the trtllm-gen decode kernel rejects a relative attention bias;
      * FlashInfer has no additive per-token bias hook.
    Mirroring the SGLang reference (``inkling_common/attn.py`` +
    ``kernels/ops/attention/{extend,decode}_attention.py``), the bias is
    precomputed on the torch side as a contiguous ``rel_logits`` aux tensor
    ``[num_query_tokens, local_heads, rel_extent]`` (``einsum('thd,de->the', r,
    proj)`` with the global-layer ``tau`` folded in), and the Triton prefill /
    paged-decode kernels gather+add it: ``bias = rel_logits[q_idx, head,
    clamp(q_pos-k_pos, 0, rel_extent-1)]`` where ``0 <= q_pos-k_pos <
    rel_extent``. Because ``rel_logits`` is a static-shape tensor (its first dim
    equals the batch in the decode phase), the paged-decode kernel captures and
    replays cleanly under CUDA graph -- the launch grid ``(batch, heads)`` is
    fixed and per-request sequence lengths are read from a GPU tensor. Local
    layers apply the sliding window natively inside the kernel
    (``window_left = sliding_window_size - 1``); global layers apply the
    log-scaling ``tau`` folded into ``rel_logits`` (a no-op below the
    ``log_scaling_n_floor`` = 128k positions the bring-up stays under).

    KV read/write goes through ``KVCacheManagerV2`` in the HND paged layout: the
    context phase writes new K/V to the cache (for later reuse) and attends over
    the contiguous extend tensors; the generation phase writes the one new
    token's K/V and attends over the paged cache. ``self.attn`` (the base
    backend) is built but unused -- only its runtime-assigned ``local_layer_idx``
    (the KV-cache layer offset) is read here.
    """

    def __init__(self, model_config: ModelConfig[InklingTextConfig],
                 layer_idx: int):
        config = model_config.pretrained_config
        self.is_local = config.is_local_layer(layer_idx)
        head_dim = config.layer_head_dim(layer_idx)
        num_heads = config.layer_num_heads(layer_idx)
        num_kv_heads = config.layer_num_kv_heads(layer_idx)
        self.attention_window_size = config.layer_window(layer_idx)
        self.d_rel = config.d_rel
        self.rel_extent = (config.sliding_window_size
                           if self.is_local else config.rel_extent)
        self.log_scaling_n_floor = (None if self.is_local else
                                    config.log_scaling_n_floor)
        self.log_scaling_alpha = config.log_scaling_alpha

        # Attention (q/k/v/o projections + KV cache) is bf16, not NVFP4: the
        # checkpoint excludes ``model.llm.layers.{i}.attn``. The base Attention
        # builds qkv_proj/o_proj from ``config.get_quant_config()`` (the global
        # NVFP4 config, which packs the input dim to hidden/2 and demands scale
        # sidecars the bf16 checkpoint does not have), so hand it a shallow
        # ModelConfig copy whose ``quant_config`` is empty for this layer. r_proj
        # below is already unquantized (no quant_config passed).
        # (ModelConfig.__setattr__ whitelists ``quant_config`` for exactly this
        # per-module-quant override, so the shallow copy needs no unfreeze.)
        attn_model_config = model_config
        if _module_excluded_from_quant(model_config,
                                       f"model.llm.layers.{layer_idx}.attn"):
            from tensorrt_llm.models.modeling_utils import QuantConfig
            attn_model_config = copy.copy(model_config)
            attn_model_config.quant_config = QuantConfig()

        super().__init__(
            hidden_size=config.hidden_size,
            num_attention_heads=num_heads,
            num_key_value_heads=num_kv_heads,
            max_position_embeddings=config.max_position_embeddings,
            bias=False,
            # No RoPE: this model overrides forward to run the Inkling Triton
            # attention (qk-norm + sconv + relative-bias score_mod) directly, so
            # pos_embd_params=None keeps the base from building an unused
            # RotaryEmbedding. The base backend ``self.attn`` is still
            # constructed but unused for compute (only its runtime-assigned
            # ``local_layer_idx`` -- the KV-cache layer offset -- is read).
            pos_embd_params=None,
            layer_idx=layer_idx,
            dtype=config.torch_dtype,
            config=attn_model_config,
            # q/k are per-head RMS-normalized, so the score scale is 1/head_dim
            # rather than 1/sqrt(head_dim). The backend uses
            # 1/(sqrt(head_dim) * q_scaling); q_scaling = sqrt(head_dim) yields
            # the required 1/head_dim.
            q_scaling=float(head_dim)**0.5,
            skip_rope=True,
            fuse_qk_norm_rope=False,
            is_qk_norm=True,
        )
        # head_dim is uniform (128) across local/global layers and differs from
        # hidden_size // num_heads (96), so the base Attention must read it from
        # config.head_dim (QKNormRoPEAttention does not accept a head_dim kwarg).
        assert self.head_dim == head_dim, (self.head_dim, head_dim)

        # Inkling score scale is 1/head_dim (per-head q/k RMSNorm replaces the
        # usual 1/sqrt(head_dim)), applied directly by the Triton kernels. The
        # sliding window is applied natively inside the kernel for local layers
        # (inclusive radius = window - 1: query p attends to keys [p-(w-1), p]).
        self.sm_scale = 1.0 / float(head_dim)
        self.window_left = (self.attention_window_size -
                            1) if self.is_local else -1

        tp_size = model_config.mapping.tp_size
        # r projection: per-head relative states (num_heads * d_rel), sharded by
        # head like q. Output is not gathered (consumed locally to build bias).
        self.r_proj = Linear(
            config.hidden_size,
            num_heads * self.d_rel,
            bias=False,
            dtype=config.torch_dtype,
            mapping=model_config.mapping,
            tensor_parallel_mode=TensorParallelMode.COLUMN,
            gather_output=False,
        )
        # Learned relative-logit profiles, replicated across TP ranks. The
        # profile length is per-layer: local layers store only the
        # sliding-window extent (512), global layers the full rel_extent (1024)
        # -- so the parameter must use ``self.rel_extent``, not the global
        # ``config.rel_extent`` (mismatch here is the 1024-vs-512 load crash).
        self.rel_logits_proj = nn.Parameter(
            torch.empty(self.d_rel, self.rel_extent))
        # k/v short convs act on the k/v stream from the fused qkv projection,
        # so they are sharded by kv-head like that projection. Pass the FULL
        # channel count and let InklingShortConv slice this rank's block at load.
        full_kv_dim = num_kv_heads * head_dim
        self.k_sconv = InklingShortConv(full_kv_dim,
                                        config.sconv_kernel_size,
                                        mapping=model_config.mapping,
                                        tp_shard=True)
        self.v_sconv = InklingShortConv(full_kv_dim,
                                        config.sconv_kernel_size,
                                        mapping=model_config.mapping,
                                        tp_shard=True)
        self.local_num_heads = num_heads // tp_size
        # Stable GPU buffers for the CUDA-graph-safe runtime decode metadata,
        # refreshed eagerly (before capture/replay) by the model engine via
        # InklingForCausalLM.prepare_inkling_attn_decode -> _decode_meta.refresh.
        self._decode_meta = InklingDecodeMeta(layer_idx)
        # feedback #18 no-probe B2 candidates: give the decode-meta a back-ref so
        # ``refresh`` can size the ``persist_out`` buffer from this layer's head
        # geometry; ``_b2_kvpool_zeroed`` gates the one-time ``zero_kvpool`` pool
        # memset (see ``_attention``). Both are inert unless ``INKLING_B2_FIX`` is
        # set, so the production path is byte-unchanged.
        self._decode_meta._owner = self
        self._b2_kvpool_zeroed = False

    def _project(self,
                 hidden_states,
                 conv_states,
                 conv_pool_kv=None,
                 conv_rt=None):
        """Fused qkv projection -> split -> k/v short-conv -> per-head qk RMSNorm.

        Returns ``(q, k, v, new_kv_state)`` with q/k/v shaped
        ``[T, local_heads, head_dim]`` / ``[T, local_kv_heads, head_dim]``.
        The k/v short-conv path is selected by (in priority order):

        * ``conv_pool_kv=(pool_k, pool_v)`` + ``conv_rt`` -- the RUNTIME state
          pool path: seed the pool for context tokens and update it in place at
          the per-request slots for generation tokens (fused ops, CUDA-graph
          safe, supports mixed batches). ``new_kv_state`` is ``None`` (the pool
          is mutated in place, not returned).
        * ``conv_states=(state_k, state_v)`` -- the explicit-window single-step
          decode path used by the focused replays (crit4/crit8/global-source);
          ``new_kv_state`` is the rolled ``(state_k', state_v')`` window.
        * neither -- the stateless full-sequence causal conv (context phase /
          focused prefill); ``new_kv_state`` is ``None``.
        """
        D = self.head_dim
        num_tokens = hidden_states.shape[0]
        qkv = self.qkv_proj(hidden_states)
        q, k, v = self.split_qkv(qkv, None, None)
        # k/v short convolution before the q/k norm (source order).
        if conv_pool_kv is not None:
            pool_k, pool_v = conv_pool_kv
            k = _apply_sconv(self.k_sconv, k, pool_k, conv_rt)
            v = _apply_sconv(self.v_sconv, v, pool_v, conv_rt)
            new_kv_state = None
        elif conv_states is None:
            k = self.k_sconv(k)
            v = self.v_sconv(v)
            new_kv_state = None
        else:
            state_k, state_v = conv_states
            k, new_sk = self.k_sconv.forward_decode(k, state_k)
            v, new_sv = self.v_sconv.forward_decode(v, state_v)
            new_kv_state = (new_sk, new_sv)
        q, k = self.apply_qk_norm(q, k)
        nh = self.q_size // D
        nkv = self.kv_size // D
        return (q.view(num_tokens, nh,
                       D), k.view(num_tokens, nkv,
                                  D), v.view(num_tokens, nkv, D), new_kv_state)

    def _build_rel_logits(self, hidden_states: torch.Tensor,
                          position_ids: Optional[torch.Tensor]) -> torch.Tensor:
        """Contiguous relative-bias aux tensor ``[T, local_heads, rel_extent]``.

        ``rel_logits[t, h, e] = sum_d r[t, h, d] * proj[d, e]`` (fp32), mirroring
        HF ``InklingRelativeLogits`` / SGLang ``RelLogitsProj``. For global layers
        the log-scaling ``tau`` (a no-op below ``log_scaling_n_floor`` = 128k, so
        exactly 1.0 across the bring-up regime) is folded in per query token. The
        Triton kernels index this by ``clamp(q_pos-k_pos, 0, rel_extent-1)`` and
        zero it outside ``[0, rel_extent)`` -- the exact source score_mod.
        """
        r = self.r_proj(hidden_states).view(-1, self.local_num_heads,
                                            self.d_rel)
        rel = torch.einsum("thd,de->the", r.float(),
                           self.rel_logits_proj.float())  # [T, H, rel_extent]
        # DIAGNOSTIC ablation (env-gated, default OFF -> production byte-unchanged):
        # INKLING_ABLATE_RELBIAS=1 zeros the learned relative-position bias so the
        # attention runs on the core QK/PV path alone. Used by the iter90 MMLU
        # B-token-bias localizer to test whether TRT's relative-bias implementation
        # (vs SGLang's flashinfer score_mod) injects the systematic 'B' answer bias.
        if os.environ.get("INKLING_ABLATE_RELBIAS", "0") == "1":
            rel = torch.zeros_like(rel)
        if self.log_scaling_n_floor is not None and position_ids is not None:
            pos = position_ids.reshape(-1).float()
            tau = 1.0 + self.log_scaling_alpha * torch.log(
                ((pos + 1.0) / self.log_scaling_n_floor).clamp(min=1.0))
            rel = rel * tau[:, None, None]
        return rel.contiguous()

    def _attention(self,
                   q,
                   k,
                   v,
                   rel_logits,
                   attn_metadata,
                   *,
                   decode_seq_lens,
                   decode_page_table,
                   skip_kv_write,
                   allow_mixed=False):
        """Dispatch prefill / decode over the paged cache, supporting mixed
        context+generation batches.

        The runtime packs context requests first (each with its full new-token
        span) then one-token generation requests (``seq_lens == 1``). We slice
        the packed q/k/v/rel_logits + per-request metadata at that boundary and
        run the context slice through the prefill kernel and the generation
        slice through the paged-decode kernel, concatenating the outputs. Pure
        context (``num_contexts == num_seqs``) and pure generation
        (``num_contexts == 0``) fall out as the single-slice cases.
        """
        # ``KVCacheManagerV2.get_buffers`` / ``get_batch_cache_indices`` take the
        # GLOBAL layer index and map it through ``layer_offsets`` themselves
        # (identity for single-node TP-only, the pp-local offset under PP). Use
        # ``self.layer_idx`` (the model's global decoder layer index) directly:
        # the base backend's ``self.attn.local_layer_idx`` is only primed inside
        # the base attention forward, which Inkling bypasses, so it stays ``None``
        # at real runtime (the focused replays set it by hand, masking this).
        cache_layer = self.layer_idx
        kv = attn_metadata.kv_cache_manager.get_buffers(cache_layer,
                                                        kv_layout="HND")
        # kv: [num_pages, 2, num_kv_heads, page_size, head_dim]
        k_cache, v_cache = kv[:, 0], kv[:, 1]
        page_size = kv.shape[3]
        mgr = attn_metadata.kv_cache_manager
        request_ids = attn_metadata.request_ids
        num_cached = attn_metadata.kv_cache_params.num_cached_tokens_per_seq
        seq_lens = attn_metadata.seq_lens.tolist()
        num_contexts = attn_metadata.num_contexts
        num_seqs = len(seq_lens)
        ctx_tokens = sum(seq_lens[:num_contexts])

        # feedback #18 no-probe B2 candidate ``zero_kvpool``: zero this layer's
        # paged KV pool ONCE, eagerly, on its first context (prefill) pass -- before
        # any K/V is written -- so a later decode read of an unwritten (page, slot)
        # sees a finite 0 instead of poisoned/uninitialized pool memory (a candidate
        # capture-time B2 nan source). Fires at most once per layer lifetime (the
        # first prefill / KV-estimation forward, never under decode-graph capture),
        # so it cannot wipe a live request's cache. Diagnostic-only (the single-
        # request B2 repro); default-off, production byte-unchanged.
        if (num_contexts > 0 and not self._b2_kvpool_zeroed
                and _b2_fix_active("zero_kvpool")):
            kv.zero_()
            self._b2_kvpool_zeroed = True

        # A mixed context+generation batch needs ``_project`` to apply the
        # prefill short-conv to the context tokens and the decode short-conv to
        # the generation tokens -- which only the per-request short-conv state
        # pool (``InklingConvStateCache``, the ``conv_rt`` runtime path) does
        # correctly. The stateless / explicit-window paths convolve one token
        # group at a time, so a mixed batch there would convolve across the
        # context/generation boundary; refuse it explicitly unless the pool path
        # is active (``allow_mixed``). The focused replays issue pure-context and
        # pure-generation forwards, so they never hit this.
        if 0 < num_contexts < num_seqs and not allow_mixed:
            raise NotImplementedError(
                "InklingAttention: mixed context+generation batch needs the "
                "short-conv state pool (pass conv_cache/conv_rt); the stateless "
                "and explicit-window short-conv paths cannot mix a batch")

        outs = []
        if num_contexts > 0:
            outs.append(
                self._run_context(q[:ctx_tokens], k[:ctx_tokens],
                                  v[:ctx_tokens], rel_logits[:ctx_tokens],
                                  seq_lens[:num_contexts],
                                  num_cached[:num_contexts],
                                  request_ids[:num_contexts], mgr, cache_layer,
                                  k_cache, v_cache, page_size, skip_kv_write))
        if num_contexts < num_seqs:
            outs.append(
                self._run_generation(q[ctx_tokens:], k[ctx_tokens:],
                                     v[ctx_tokens:], rel_logits[ctx_tokens:],
                                     num_cached[num_contexts:],
                                     request_ids[num_contexts:], mgr,
                                     cache_layer, k_cache, v_cache, page_size,
                                     decode_seq_lens, decode_page_table,
                                     skip_kv_write))
        return outs[0] if len(outs) == 1 else torch.cat(outs, dim=0)

    def _run_context(self, q, k, v, rel_logits, seq_lens, num_cached,
                     request_ids, mgr, cache_layer, k_cache, v_cache, page_size,
                     skip_kv_write):
        device = q.device
        # Persist new K/V to the paged cache for later generation reuse.
        if not skip_kv_write:
            block_ids = mgr.get_batch_cache_indices(request_ids, cache_layer)
            off = 0
            for i, sl in enumerate(seq_lens):
                write_kv_cache_hnd(k_cache, v_cache, k[off:off + sl],
                                   v[off:off + sl], block_ids[i],
                                   int(num_cached[i]), page_size)
                off += sl
        cu = torch.zeros(len(seq_lens) + 1, dtype=torch.int32, device=device)
        cu[1:] = torch.tensor(seq_lens, dtype=torch.int32,
                              device=device).cumsum(0)
        max_seqlen = max(seq_lens)
        return inkling_prefill_attention(q, k, v, cu, max_seqlen, self.sm_scale,
                                         rel_logits, self.rel_extent,
                                         self.window_left)

    def _run_generation(self, q, k, v, rel_logits, num_cached, request_ids, mgr,
                        cache_layer, k_cache, v_cache, page_size,
                        decode_seq_lens, decode_page_table, skip_kv_write):
        device = q.device
        # --- Runtime CUDA-graph-safe path. ---------------------------------
        # When the model engine has eagerly published this batch's decode
        # metadata into the layer's stable GPU buffers (``_decode_meta.ready``)
        # and no explicit static tensors were passed (the focused-replay path),
        # the captured forward performs ZERO host->device copy: it reads the
        # stable ``seq_lens``/``page_table`` buffers and persists the new token's
        # K/V into the paged cache with an in-graph GPU scatter whose (page,
        # offset) indices are derived on-GPU from those buffers. This replaces the
        # host ``write_kv_cache_hnd`` loop + ``torch.tensor(..., device=cuda)``
        # build that raised ``Cannot copy between CPU and CUDA tensors during CUDA
        # graph capture``. Padding rows carry their own (dummy) registered request
        # slots -- ``attn_metadata.request_ids`` is padded after ``prepare()`` --
        # so the scatter never corrupts a real request's page 0.
        meta = self._decode_meta
        if decode_seq_lens is None and decode_page_table is None and meta.ready:
            num_req = q.shape[0]
            sl = meta.seq_lens[:num_req]
            pt = meta.page_table[:num_req]
            pos = (sl - 1).long()  # write slot = total_kv_len - 1 = num_cached
            page_row = torch.div(pos, page_size, rounding_mode="floor")
            offs = pos - page_row * page_size
            pages = pt.gather(1, page_row.unsqueeze(1)).squeeze(1).long()
            # HND paged cache: [num_pages, num_kv_heads, page_size, head_dim];
            # paired advanced indices (pages, offs) select one (page, slot) per
            # request -> [num_req, num_kv_heads, head_dim], matching new k/v.
            k_cache[pages, :, offs, :] = k.to(k_cache.dtype)
            v_cache[pages, :, offs, :] = v.to(v_cache.dtype)
            # feedback #18 no-probe B2 candidate ``persist_out``: write the decode
            # kernel into the stable, eagerly-zeroed per-layer buffer (sized in
            # ``_decode_meta.refresh``) instead of a fresh pool-transient
            # ``torch.empty_like`` output whose address moves with the CUDA-graph
            # pool layout -- the exact Heisenbug knob. ``None`` -> production path
            # (fresh output), byte-unchanged when the toggle is off.
            ob = (meta.out_buf[:num_req]
                  if (meta.out_buf is not None
                      and _b2_fix_active("persist_out")) else None)
            return inkling_decode_attention(q, k_cache, v_cache, sl, pt,
                                            page_size, self.sm_scale, rel_logits,
                                            self.rel_extent, self.window_left,
                                            out=ob)
        # The write and the ragged->dense block-id work are host-side; under
        # CUDA-graph replay ``skip_kv_write`` is set and static ``decode_*``
        # tensors are supplied, so this whole block is skipped and only GPU ops
        # (projection, einsum, decode kernel, o_proj) enter the captured graph.
        if (not skip_kv_write) or decode_seq_lens is None \
                or decode_page_table is None:
            num_req = len(request_ids)
            block_ids = mgr.get_batch_cache_indices(request_ids, cache_layer)
            if not skip_kv_write:
                for i in range(num_req):
                    write_kv_cache_hnd(k_cache, v_cache, k[i:i + 1],
                                       v[i:i + 1], block_ids[i],
                                       int(num_cached[i]), page_size)
            if decode_seq_lens is None:
                total = [int(num_cached[i]) + 1 for i in range(num_req)]
                decode_seq_lens = torch.tensor(total,
                                               dtype=torch.int32,
                                               device=device)
            if decode_page_table is None:
                max_pages = max(len(b) for b in block_ids)
                decode_page_table = build_page_table(block_ids, max_pages,
                                                     device)
        return inkling_decode_attention(q, k_cache, v_cache, decode_seq_lens,
                                        decode_page_table, page_size,
                                        self.sm_scale, rel_logits,
                                        self.rel_extent, self.window_left)

    def forward(self,
                position_ids: Optional[torch.IntTensor],
                hidden_states: torch.Tensor,
                attn_metadata: AttentionMetadata,
                *,
                conv_states=None,
                conv_pool_kv=None,
                conv_rt=None,
                decode_seq_lens=None,
                decode_page_table=None,
                skip_kv_write: bool = False,
                return_conv_state: bool = False,
                ops_fp: Optional[torch.Tensor] = None,
                ops_scratch: Optional[torch.Tensor] = None,
                **kwargs):
        """Inkling attention through the Triton score_mod path.

        Short-conv path (priority): ``conv_pool_kv=(pool_k, pool_v)`` + ``conv_rt``
        drives the RUNTIME state pool (seed on context, in-place update at the
        per-request slots on generation, mixed-batch capable, CUDA-graph safe);
        ``conv_states=(state_k, state_v)`` drives the explicit-window single-step
        decode used by the focused replays; neither runs the stateless conv.
        ``decode_seq_lens``/``decode_page_table`` are precomputed static GPU
        tensors for CUDA-graph replay; ``skip_kv_write`` skips the paged-cache
        write (cache pre-populated before capture). When ``return_conv_state`` is
        set (explicit-window carry path) returns ``(o_proj_out, new_kv_state)``;
        otherwise just ``o_proj_out`` (default, so the crit4 replays and the pool
        path -- which mutates the pool in place -- are unchanged).
        """
        num_tokens = hidden_states.shape[0]
        # The pre-attention RMSNorm can emit fp32 (the residual-stream norm
        # path), but the attention/r projections are bf16 (``.attn`` is excluded
        # from NVFP4). Cast once so the decoder-layer forward is robust to the
        # norm's output dtype -- the isolated crit4 replay fed a pre-cast bf16
        # tensor, which hid this until the stacked forward / runtime.
        hidden_states = hidden_states.to(self.qkv_proj.weight.dtype)
        q, k, v, new_kv_state = self._project(hidden_states, conv_states,
                                              conv_pool_kv, conv_rt)
        # feedback #17 op-level B2 bisection -- walk INTO the global-attention
        # kernel. q/k/v (post qkv-proj + kv-sconv + qk-norm) and rel_logits are
        # the paged-decode kernel's INPUTS; if they are finite but attn_kernel is
        # non-finite, the nan is born in the softmax / PV / KV-page read, not
        # upstream. Written capture-safely (element-wise stats, D2D into the graph)
        # only under INKLING_FP_OPS. Fingerprint the last generation row.
        if ops_fp is not None:
            _ink_fp_stat(ops_fp[1], q[-1], ops_scratch)  # attn_q
            _ink_fp_stat(ops_fp[2], k[-1], ops_scratch)  # attn_k
            _ink_fp_stat(ops_fp[3], v[-1], ops_scratch)  # attn_v
        rel_logits = self._build_rel_logits(hidden_states, position_ids)
        if ops_fp is not None:
            _ink_fp_stat(ops_fp[4], rel_logits[-1], ops_scratch)  # attn_rel
        attn_out = self._attention(q,
                                   k,
                                   v,
                                   rel_logits,
                                   attn_metadata,
                                   decode_seq_lens=decode_seq_lens,
                                   decode_page_table=decode_page_table,
                                   skip_kv_write=skip_kv_write,
                                   allow_mixed=conv_rt is not None)
        attn_out = attn_out.reshape(num_tokens, self.q_size)
        if ops_fp is not None:
            # attn_kernel = softmax/PV/KV-page-read output (pre out-proj).
            _ink_fp_stat(ops_fp[5], attn_out[-1], ops_scratch)
            # op index 6 (o_proj_local, the per-rank o_proj matmul PRE all-reduce)
            # is INTENTIONALLY left UNMEASURED: the extra F.linear GEMM per global
            # layer is the heaviest in-graph op and is what shifted the CUDA-graph
            # pool and SUPPRESSED B2 in iter-75. The matmul-vs-all-reduce question
            # is answered instead from the CROSS-RANK finiteness of attn_kernel
            # (op 5, this rank's pre-o_proj input) vs o_proj_out (op 7, post
            # all-reduce) in FP_OPS_XRANK: attn_kernel finite on every rank while
            # o_proj_out is non-finite implicates the TP all-reduce collective (or
            # a remote rank's nan it summed), not this rank's kernel.
        out = self.o_proj(attn_out)
        if ops_fp is not None:
            _ink_fp_stat(ops_fp[7], out[-1], ops_scratch)  # o_proj_out (post all-reduce)
        if return_conv_state:
            return out, new_kv_state
        return out


# ----------------------------------------------------------------------------
# Dense MLP (layers 0, 1) and MoE (layers 2..65)
# ----------------------------------------------------------------------------
class InklingDenseMLP(nn.Module):
    """SwiGLU MLP with a learned scalar ``global_scale`` (layers 0, 1).

    Fused gate+up (``w13_dn``) column-parallel, down (``w2_md``) row-parallel.
    """

    def __init__(self, model_config: ModelConfig[InklingTextConfig]):
        super().__init__()
        config = model_config.pretrained_config
        inter = config.dense_intermediate_size
        self.gate_up_proj = Linear(
            config.hidden_size,
            2 * inter,
            bias=False,
            dtype=config.torch_dtype,
            mapping=model_config.mapping,
            tensor_parallel_mode=TensorParallelMode.COLUMN,
            weights_loading_config=WeightsLoadingConfig(
                weight_mode=WeightMode.FUSED_GATE_UP_LINEAR),
        )
        self.down_proj = Linear(
            inter,
            config.hidden_size,
            bias=False,
            dtype=config.torch_dtype,
            mapping=model_config.mapping,
            tensor_parallel_mode=TensorParallelMode.ROW,
        )
        self.global_scale = nn.Parameter(torch.ones(1))
        self.act_fn = torch.nn.functional.silu

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        # ``global_scale`` is an fp32 scalar Parameter; multiplying promotes the
        # output to fp32. Cast back to the input dtype so the bf16 residual
        # stream (and the next layer's bf16 projections) stay bf16.
        out = self.down_proj(self.act_fn(gate) * up) * self.global_scale
        return out.to(x.dtype)


class InklingGate(nn.Module):
    """fp32 router: logits over 256 routed + 2 shared experts, plus the additive
    selection bias and the learned global scale. Feeds
    :class:`InklingMoeRoutingMethod`.
    """

    def __init__(self, config: InklingTextConfig):
        super().__init__()
        self.num_routed = config.n_routed_experts
        self.n_shared = config.n_shared_experts
        self.top_k = config.num_experts_per_tok
        self.route_scale = config.route_scale
        n_total = self.num_routed + self.n_shared
        self.weight = nn.Parameter(
            torch.empty(n_total, config.hidden_size, dtype=torch.float32))
        self.bias = nn.Parameter(
            torch.empty(self.num_routed, dtype=torch.float32))
        self.global_scale = nn.Parameter(torch.ones(1, dtype=torch.float32))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.linear(hidden_states.float(), self.weight)

    @property
    def routing_method(self) -> InklingMoeRoutingMethod:
        return InklingMoeRoutingMethod(
            top_k=self.top_k,
            num_experts=self.num_routed,
            n_shared_experts=self.n_shared,
            callable_gate_bias=lambda: self.bias,
            callable_global_scale=lambda: self.global_scale,
            route_scale=self.route_scale,
        )


class InklingSharedExperts(nn.Module):
    """Two shared SwiGLU experts, each weighted by a per-token gamma and summed.

    Reference: HF ``InklingSharedExperts`` (batched 2-expert SwiGLU, fp32 sum).
    """

    def __init__(self, config: InklingTextConfig):
        super().__init__()
        self.n_shared = config.n_shared_experts
        inter = config.intermediate_size
        hidden = config.hidden_size
        # [n_shared, 2*inter, hidden] fused gate+up; [n_shared, hidden, inter] down.
        # Must be created in the model dtype (bf16): the shared experts run as raw
        # bmms against the bf16 hidden stream, so an untyped (default-fp32) param
        # dtype-mismatches the bmm ("expected BFloat16 but found Float"). The
        # checkpoint stores these bf16 (shared_experts is in exclude_modules).
        self.shared_w13 = nn.Parameter(
            torch.empty(self.n_shared,
                        2 * inter,
                        hidden,
                        dtype=config.torch_dtype))
        self.shared_w2 = nn.Parameter(
            torch.empty(self.n_shared, hidden, inter, dtype=config.torch_dtype))
        self.act_fn = torch.nn.functional.silu

    def forward(self, hidden_states: torch.Tensor,
                gammas: torch.Tensor) -> torch.Tensor:
        # hidden_states: [T, hidden] (bf16); gammas: [T, n_shared] fp32 (from the
        # joint renorm). Keep both bmms in the activation dtype and apply the
        # per-token gamma in fp32 AFTER the (linear) down projection, where it
        # commutes: gamma * (act @ w2) == (act * gamma) @ w2. This avoids
        # upcasting the down-proj bmm's LHS to fp32 (which mismatches the bf16
        # shared_w2 -- an "expected BFloat16 but found Float" bmm error) while
        # keeping gamma at full fp32 precision and matching the source fp32
        # gamma-weighted sum.
        x = hidden_states.unsqueeze(0).expand(self.n_shared, -1, -1)
        gate_up = torch.bmm(x, self.shared_w13.transpose(1, 2))
        # ``shared_w13`` loads RAW: gate/up are Inkling-INTERLEAVED [g0,u0,...]
        # along its 2*inter output dim (SGLang inference_moe_w13_interleaved), so
        # the bmm output channels are interleaved -- gate = even, up = odd. A
        # contiguous chunk(2) here would pair the wrong channels (silu(mix)*mix).
        gate, up = gate_up[..., 0::2], gate_up[..., 1::2]
        activated = self.act_fn(gate) * up
        out = torch.bmm(activated,
                        self.shared_w2.transpose(1, 2))  # [S, T, hidden]
        out = out.float() * gammas.transpose(0, 1).unsqueeze(-1).float()
        return out.sum(dim=0).to(hidden_states.dtype)


class InklingMoE(nn.Module):
    """Router + routed experts (fused MoE) + two shared experts.

    Routed experts run through :func:`create_moe` (NVFP4 for layers 3..65, bf16
    for layer 2 via a per-layer quant override). Shared experts and the router
    stay bf16/fp32. The routed output already reduces over the top-6 experts; the
    gamma-weighted shared output is added on top (source ``h + shared``).
    """

    def __init__(self, model_config: ModelConfig[InklingTextConfig],
                 layer_idx: int):
        super().__init__()
        config = model_config.pretrained_config
        self.gate = InklingGate(config)
        self.num_routed = config.n_routed_experts
        self.n_shared = config.n_shared_experts
        self.top_k = config.num_experts_per_tok
        self.route_scale = config.route_scale

        experts_quant_config = self._experts_quant_config(
            model_config, layer_idx)
        # reduce_results=True: all-reduce the routed-expert output across the TP
        # group. Under TP each rank holds a shard of the 256 experts and produces
        # only a PARTIAL routed sum, so the full routed output is the sum across
        # ranks. Without this all-reduce the TP=4 runtime adds a per-rank partial
        # routed output to the (replicated, full) shared-expert output and the
        # whole model produces garbage from the first token (the dense layers 0/1
        # are correct because their row-parallel down_proj already all-reduces;
        # the all-TP=1 focused MoE replay never exercised this). The shared
        # experts stay replicated and are added AFTER this reduce (full + full),
        # so they are not double-counted.
        #
        # NOTE (iter29): an fp32 routed partial + fp32 all-reduce (reduce_results=
        # False + output_dtype=fp32) was TESTED and REVERTED -- it did NOT fix the
        # TP=4 garbage. The TP-localizer showed the seed (layer3 isolated cos
        # 0.9977) is unchanged by fp32 reduction, so it is NOT bf16 all-reduce
        # cancellation; the sharded NVFP4 routed GEMM itself diverges ~0.3% from
        # the full TP=1 GEMM (magnitude-dependent, in the FP4 weight/scale
        # sharding, not the reduce). See progress.yaml iter29.
        # Select the routed-expert MoE backend. Default: whatever
        # ``model_config.moe_backend`` resolves to (CUTLASS for this checkpoint).
        # When ``INKLING_MOE_BACKEND=TRTLLM`` route the NVFP4 routed experts
        # (layers 3..65) through the trtllm-gen blockScaleMoe kernel via a
        # frozen-safe shallow model_config copy so the global config is not
        # mutated (see ``_moe_config_with_trtllm_backend``). The bf16 layer-2
        # experts (no quant) auto-fall back to CUTLASS inside ``resolve_moe_cls``.
        moe_model_config = model_config
        if _inkling_trtllm_moe_backend():
            moe_model_config = _moe_config_with_trtllm_backend(model_config)
        self.experts = create_moe(
            routing_method=self.gate.routing_method,
            num_experts=self.num_routed,
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            dtype=config.torch_dtype,
            reduce_results=True,
            model_config=moe_model_config,
            override_quant_config=experts_quant_config,
            layer_idx=layer_idx,
        )
        # Proof marker: log the RESOLVED routed-expert backend so runs can confirm
        # the trtllm-gen (blockScaleMoe) kernel is actually selected under
        # INKLING_MOE_BACKEND=TRTLLM rather than silently falling back to CUTLASS.
        # ``create_moe`` returns a ``ConfigurableMoE`` wrapper whose ``.backend`` is
        # the resolved backend instance (``TRTLLMGenFusedMoE`` for the NVFP4 layers
        # 3..65, ``CutlassFusedMoE`` for the bf16 layer-2), so the wrapper class
        # name alone does NOT reveal the backend -- introspect ``.backend`` for the
        # true ``backend_cls``. NOTE: the TRT-LLM ``logger`` appends the args
        # instead of %-interpolating (a ``"...%s..."`` call prints the literal
        # format string with the values dumped at the end -- unreadable AND
        # ungreppable, which is why the iter65 TGMOE_BACKEND grep matched 0 lines),
        # so pre-format with an f-string.
        backend_cls = type(getattr(self.experts, "backend", self.experts)).__name__
        logger.info(
            f"INKLING_MOE_SELECT layer={layer_idx} "
            f"requested_backend={moe_model_config.moe_backend} "
            f"experts_cls={type(self.experts).__name__} "
            f"backend_cls={backend_cls} "
            f"routing_type={self.gate.routing_method.routing_method_type.name} "
            f"separated={self.gate.routing_method.requires_separated_routing}")
        self.shared_experts = InklingSharedExperts(config)

    @staticmethod
    def _experts_quant_config(model_config: ModelConfig, layer_idx: int):
        """Per-layer expert quant: NVFP4 unless the checkpoint excludes it.

        The checkpoint lists its bf16 modules in ``hf_quant_config.json``
        ``quantization.exclude_modules`` (read into
        ``quant_config.exclude_modules`` by ``from_pretrained``). Layer-2 routed
        experts are excluded (bf16 MoE) while layers 3..65 routed experts are
        NVFP4. ``quant_config_dict`` / ``per_layer_quant_configs`` are only
        populated for MIXED_PRECISION checkpoints, so for this plain-NVFP4
        checkpoint the authoritative per-layer signal is ``exclude_modules``.
        Return an empty (no-quant) ``QuantConfig`` for an excluded expert module
        so ``create_moe`` builds an unquantized bf16 MoE; otherwise the NVFP4
        base config.
        """
        if _module_excluded_from_quant(
                model_config, f"model.llm.layers.{layer_idx}.mlp.experts"):
            from tensorrt_llm.models.modeling_utils import QuantConfig
            return QuantConfig()
        return model_config.quant_config

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        router_logits = self.gate(hidden_states)  # [T, 258] fp32
        routed = self.experts(hidden_states, router_logits)
        _, _, shared_gammas = inkling_joint_renorm(
            router_logits,
            gate_bias=self.gate.bias,
            global_scale=self.gate.global_scale,
            route_scale=self.route_scale,
            top_k=self.top_k,
            num_routed=self.num_routed,
            n_shared=self.n_shared,
        )
        shared = self.shared_experts(hidden_states, shared_gammas)
        # Keep the bf16 residual-stream dtype (fp32 scales in the routed/shared
        # paths can promote the sum) so the next layer's projections stay bf16.
        return (routed + shared).to(hidden_states.dtype)


# ----------------------------------------------------------------------------
# Decoder layer / model / causal LM
# ----------------------------------------------------------------------------
class InklingDecoderLayer(nn.Module):
    """Pre-norm attention + MLP, each followed by a short-conv with an internal
    residual, then the residual add (HF ``InklingDecoderLayer`` order).
    """

    def __init__(self, model_config: ModelConfig[InklingTextConfig],
                 layer_idx: int):
        super().__init__()
        config = model_config.pretrained_config
        self.layer_idx = layer_idx
        self.attn_norm = RMSNorm(hidden_size=config.hidden_size,
                                 eps=config.rms_norm_eps,
                                 dtype=config.torch_dtype)
        self.attn = InklingAttention(model_config, layer_idx)
        self.attn_sconv = InklingShortConv(config.hidden_size,
                                           config.sconv_kernel_size)
        self.mlp_norm = RMSNorm(hidden_size=config.hidden_size,
                                eps=config.rms_norm_eps,
                                dtype=config.torch_dtype)
        if config.is_dense_layer(layer_idx):
            self.mlp = InklingDenseMLP(model_config)
        else:
            self.mlp = InklingMoE(model_config, layer_idx)
        self.mlp_sconv = InklingShortConv(config.hidden_size,
                                          config.sconv_kernel_size)

    def forward(self,
                position_ids: torch.IntTensor,
                hidden_states: torch.Tensor,
                attn_metadata: AttentionMetadata,
                *,
                conv_state: Optional[InklingConvState] = None,
                conv_rt: Optional[InklingConvRuntime] = None,
                dump_sink: Optional[dict] = None,
                diverge_sink: Optional[dict] = None,
                ops_fp: Optional[torch.Tensor] = None,
                ops_scratch: Optional[torch.Tensor] = None,
                **kwargs) -> torch.Tensor:
        """Pre-norm attention + MLP, each followed by a short-conv (internal
        residual), then the residual add.

        ``dump_sink`` (env-gated localizer, runtime pool path only): when a dict
        is passed, the layer stashes its two sub-block intermediates -- the
        post-attention residual ``h_attn`` (isolates the attention TP transform)
        and the pure MLP/MoE transform output ``moe_out`` (pre-sconv, pre-residual;
        isolates the routed/shared expert TP transform when replayed on this SAME
        ``h_attn``). Zero cost when ``None``.

        Three short-conv modes, by argument:

        * ``conv_rt`` given -- the RUNTIME state-pool path: ``conv_state`` holds
          this layer's four ``[max_batch, C, K-1]`` pool buffers
          (:meth:`InklingConvStateCache.layer_state`); each short-conv seeds the
          pool for context tokens and updates it in place at the per-request
          slots for generation tokens (fused ops, mixed-batch + CUDA-graph safe).
        * ``conv_state`` an explicit-window :class:`InklingConvState`, no
          ``conv_rt`` -- the focused single-step decode carry (crit8): convolve
          the new token against the carried window and roll it forward in place.
        * neither -- the stateless full-sequence causal conv (context phase /
          focused prefill replays).
        """
        if conv_rt is not None:
            # --- Runtime state-pool path (prefill-seed / decode / mixed). ---
            # Divergence localizer: read the carried short-conv pool state at the
            # generation slots BEFORE this step updates it. For a batch of
            # identical requests these rows must be bit-identical; a nonzero diff
            # here means the per-slot state carried from a prior step already
            # diverged (a genuine state bug), vs a stateless op introducing the
            # divergence THIS step (non-determinism / divergent KV read).
            if diverge_sink is not None and conv_rt.gen_indices is not None:
                gi = conv_rt.gen_indices
                pre_state = (_ink_rowdiff(conv_state.k[gi]),
                             _ink_rowdiff(conv_state.v[gi]),
                             _ink_rowdiff(conv_state.attn[gi]),
                             _ink_rowdiff(conv_state.mlp[gi]))
            else:
                pre_state = (0.0, 0.0, 0.0, 0.0)

            residual = hidden_states
            h = self.attn_norm(hidden_states)
            # feedback #17 op-level B2 bisection (env INKLING_FP_OPS, zero cost
            # off): capture-safe element-wise finiteness stats of this layer's
            # intra-layer op boundaries recorded INTO the decode graph, so the
            # analyzer names the FIRST op inside the first divergent layer that
            # goes non-finite under CUDA graph. ``self.attn`` writes its OWN
            # internals (indices 1..7: attn_q/k/v, attn_rel, attn_kernel,
            # o_proj_local, o_proj_out) so the bisection walks INTO the
            # global-attention kernel / paged-KV read / TP all-reduce; this layer
            # writes the boundaries around it. Order MUST match _INK_FP_OP_NAMES /
            # inkling_fp_ops_analyze.py.
            if ops_fp is not None:
                _ink_fp_stat(ops_fp[0], h[-1], ops_scratch)  # attn_norm
            h_core = self.attn(position_ids,
                               h,
                               attn_metadata,
                               conv_pool_kv=(conv_state.k, conv_state.v),
                               conv_rt=conv_rt,
                               ops_fp=ops_fp,
                               ops_scratch=ops_scratch,
                               **kwargs)
            # (h_core == o_proj_out, written as ops_fp[7] inside self.attn.)
            h_asc = _apply_sconv(self.attn_sconv, h_core, conv_state.attn,
                                 conv_rt)
            if ops_fp is not None:
                _ink_fp_stat(ops_fp[8], h_asc[-1], ops_scratch)  # attn_sconv
            h = residual + h_asc
            if ops_fp is not None:
                _ink_fp_stat(ops_fp[9], h[-1], ops_scratch)  # h_attn

            residual = h
            hn = self.mlp_norm(h)
            hmlp = self.mlp(hn)
            # ops 10/11/12 (mlp_norm/moe_out/mlp_sconv) are INTENTIONALLY not
            # fingerprinted here: B2 is attention-born (first divergent layer is a
            # global-ATTENTION layer) and the per-layer ``_ink_fp`` residual buffer
            # already captures the post-mlp stream, so skipping the mlp-side probes
            # trims in-graph reductions (the iter-75 Heisenbug lever) at no cost to
            # the attention op-localization verdict.
            if dump_sink is not None:
                # Sub-block localizer split: the post-attention residual isolates
                # the attention TP transform; the pure MLP/MoE transform output
                # (pre-sconv, pre-residual) isolates the routed/shared expert TP
                # transform when the reference replays it on this SAME h_attn.
                # ``_answer_pos_only`` (feedback #4 all-66-layer module dump) slices
                # the last (answer) token on-device before the host copy so the dump
                # stays O(2*H)/layer instead of O(2*T*H); default path unchanged.
                _ap = bool(dump_sink.get("_answer_pos_only"))

                def _pick(_t):
                    return (_t[-1] if _ap else _t).detach().float().cpu()

                dump_sink["h_attn"] = _pick(h)
                dump_sink["moe_out"] = _pick(hmlp)
                # feedback #7 finer-grain L0-L4 split: capture the intra-layer
                # module boundaries so an L3 divergence can be attributed to the
                # router vs attention-core vs the short-convs. The DECISIVE point is
                # the MoE gate -- router logits + top-k expert ids + post-renorm
                # routing weights -- which separates "router top-k chaos" from an
                # "expert GEMM/quant" bug. Recomputed from this layer's gate on the
                # SAME ``hn`` (mlp_norm output) the fused MoE consumed: ``self.mlp.
                # gate`` + ``inkling_joint_renorm`` ARE the exact routing math the
                # CUTLASS path runs, and the trtllm-gen kernel consumes the SAME
                # ``router_logits``, so this recompute is a faithful,
                # backend-independent readout of the intended top-k. Zero cost when
                # ``_finegrain`` is unset.
                if dump_sink.get("_finegrain"):
                    dump_sink["attn_core"] = _pick(h_core)
                    dump_sink["attn_sconv"] = _pick(h_asc)
                    dump_sink["mlp_norm"] = _pick(hn)
                    if isinstance(self.mlp, InklingMoE):
                        _rl = self.mlp.gate(hn)  # [T, num_routed+n_shared] fp32
                        _rw, _ti, _ = inkling_joint_renorm(
                            _rl,
                            gate_bias=self.mlp.gate.bias,
                            global_scale=self.mlp.gate.global_scale,
                            route_scale=self.mlp.route_scale,
                            top_k=self.mlp.top_k,
                            num_routed=self.mlp.num_routed,
                            n_shared=self.mlp.n_shared,
                        )
                        dump_sink["router_logits"] = _pick(_rl)
                        dump_sink["topk_idx"] = (
                            _ti[-1] if _ap else _ti).detach().to(
                                torch.int32).cpu()
                        dump_sink["routed_w"] = _pick(_rw)
            hm = _apply_sconv(self.mlp_sconv, hmlp, conv_state.mlp, conv_rt)
            # op 12 (mlp_sconv) intentionally not fingerprinted (see the mlp-side
            # note above -- attention-born B2, residual buffer covers downstream).
            if dump_sink is not None and dump_sink.get("_finegrain"):
                # point 6: mlp short-conv output (computed after the dump block).
                dump_sink["mlp_sconv"] = (
                    hm[-1] if dump_sink.get("_answer_pos_only") else hm
                ).detach().float().cpu()
            out = residual + hm
            if diverge_sink is not None:
                # Per-sub-op row divergence (identical requests -> must be 0.0).
                # The sub-ops run in this order, so the first nonzero pins the
                # origin: attn_core (paged/prefill attention + k/v sconv) ->
                # attn_sconv -> mlp_core (dense/MoE) -> mlp_sconv. ``_ctx`` is set
                # for a context/prefill forward of identical prompts (compare
                # per-request spans); None for decode (compare per-row).
                _ctx = diverge_sink.get("_ctx")
                diverge_sink[self.layer_idx] = {
                    "d_in": _ink_rowdiff(hidden_states, _ctx),
                    "attn_core": _ink_rowdiff(h_core, _ctx),
                    "attn_sconv": _ink_rowdiff(h_asc, _ctx),
                    "mlp_core": _ink_rowdiff(hmlp, _ctx),
                    "mlp_sconv": _ink_rowdiff(hm, _ctx),
                    "d_out": _ink_rowdiff(out, _ctx),
                    "pre_state": pre_state,
                }
            # Prefill-residual confirmer (env INKLING_PREFILL_RERUN): the 4
            # identical prompts prefill as SEPARATE num_contexts=1 forwards, so
            # the batched-span localizer can't see the residual -- it is run-to-run
            # non-determinism in a single prefill. self.mlp is a pure function of
            # its input, so rerun it on the SAME hn and compare: a nonzero diff
            # pins the context-phase MoE (layers >=2) as the non-deterministic op,
            # while the dense MLP (layers 0/1) must stay 0.0. Zero cost when unset.
            import os as _os_rr
            if (_os_rr.environ.get("INKLING_PREFILL_RERUN")
                    and getattr(conv_rt, "num_ctx_tokens", 0) > 0
                    and conv_rt.gen_indices is None
                    and self.layer_idx in (0, 1, 2, 3, 4, 5)):
                _hmlp2 = self.mlp(hn)
                _d = (hmlp.detach().float()
                      - _hmlp2.detach().float()).abs().max().item()
                print("INKLING_MOE_RERUN layer=%d ntok=%d d=%.3e" %
                      (self.layer_idx, hn.shape[0], _d),
                      flush=True)
            return out

        if conv_state is None:
            residual = hidden_states
            hidden_states = self.attn_norm(hidden_states)
            hidden_states = self.attn(position_ids, hidden_states,
                                      attn_metadata)
            hidden_states = self.attn_sconv(hidden_states)  # internal residual
            hidden_states = residual + hidden_states

            residual = hidden_states
            hidden_states = self.mlp_norm(hidden_states)
            hidden_states = self.mlp(hidden_states)
            hidden_states = self.mlp_sconv(hidden_states)  # internal residual
            hidden_states = residual + hidden_states
            return hidden_states

        # --- Generation phase: carry all four short-conv states. ---
        residual = hidden_states
        h = self.attn_norm(hidden_states)
        h, new_kv = self.attn(position_ids,
                              h,
                              attn_metadata,
                              conv_states=(conv_state.k, conv_state.v),
                              return_conv_state=True,
                              **kwargs)
        # Roll the k/v short-conv windows forward in place for the next step.
        conv_state.k.copy_(new_kv[0])
        conv_state.v.copy_(new_kv[1])
        h, new_attn = self.attn_sconv.forward_decode(h, conv_state.attn)
        conv_state.attn.copy_(new_attn)
        h = residual + h

        residual = h
        hm = self.mlp_norm(h)
        hm = self.mlp(hm)
        hm, new_mlp = self.mlp_sconv.forward_decode(hm, conv_state.mlp)
        conv_state.mlp.copy_(new_mlp)
        return residual + hm


class InklingModel(DecoderModel):
    """The Inkling text decoder stack. ``embed_norm`` folds onto the token
    embeddings before the layers (``use_embed_norm``)."""

    def __init__(self, model_config: ModelConfig[InklingTextConfig]):
        super().__init__(model_config)
        config = model_config.pretrained_config
        self.embed_tokens = Embedding(
            config.vocab_size,
            config.hidden_size,
            dtype=config.torch_dtype,
            mapping=model_config.mapping,
            tensor_parallel_mode=TensorParallelMode.COLUMN,
            gather_output=True,
        )
        self.embed_norm = RMSNorm(hidden_size=config.hidden_size,
                                  eps=config.rms_norm_eps,
                                  dtype=config.torch_dtype)
        self.layers = nn.ModuleList([
            InklingDecoderLayer(model_config, i)
            for i in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(hidden_size=config.hidden_size,
                            eps=config.rms_norm_eps,
                            dtype=config.torch_dtype)
        # --- B2 CUDA-graph decode localizer (env INKLING_FP, zero cost off) ---
        # Capture-SAFE per-layer decode fingerprint: a persistent, stable-pointer
        # GPU buffer [num_layers+1, hidden_size] holding the last generation row's
        # residual-stream hidden state after each decoder layer (and the final
        # norm). The per-layer write is a pure device->device ``copy_`` recorded
        # INTO the captured decode graph, so it re-runs every replay -- unlike the
        # existing ``dump_sink``/``diverge_sink`` localizers, which ``.cpu()``
        # (a D2H copy illegal under CUDA-graph capture) and therefore only ever
        # saw the EAGER path. The buffer is allocated EAGERLY (pre-capture) by
        # ``InklingForCausalLM.prepare_inkling_attn_decode`` and its contents are
        # read out eagerly there too (see that method). ``None`` => feature off.
        self._ink_fp: Optional[torch.Tensor] = None
        self._ink_fp_ops: Optional[torch.Tensor] = None
        # iter-79: pre-allocated fp32 scratch for the ALLOCATION-FREE op
        # fingerprint (``_ink_fp_stat``), so the per-op probe never allocates a
        # numel-sized temporary inside the captured decode graph.
        self._ink_fp_ops_scratch: Optional[torch.Tensor] = None
        self._ink_fp_step = 0
        self._ink_fp_prev_decode = False
        # Op-probe layer cap (Heisenbug safety knob): fingerprint global-attention
        # layers with index <= this. Default unlimited (all 11 globals). Set
        # INKLING_FP_OPS_MAXLAYER small (e.g. 17 => globals 5/11/17) to shrink the
        # in-graph op count when confirming B2 still reproduces under the probe --
        # the all-layer ``_ink_fp`` residual finds the true onset layer regardless.
        self._ink_fp_ops_maxlayer = 10_000

    def _ensure_fp_buffer(self, device) -> None:
        """Allocate the capture-safe decode-fingerprint buffer once, EAGERLY,
        before any CUDA-graph capture (a realloc under capture would strand the
        graph's aliased pointer). ``[num_layers+1, hidden_size]`` fp32; the +1 row
        holds the final-norm output. Called from ``prepare_inkling_attn_decode``
        when ``INKLING_FP`` is set."""
        if self._ink_fp is not None:
            return
        import os
        h = self.norm.weight.shape[0]
        self._ink_fp = torch.zeros(len(self.layers) + 1, h,
                                   dtype=torch.float32, device=device)
        # feedback #17 op-level B2 bisection: per-layer intra-layer op boundaries
        # (N_INK_FP_OPS=13, walking attn_norm -> QK(q,k,v,rel) ->
        # softmax/PV/KV-page(attn_kernel) -> out-proj(o_proj_local) ->
        # TP all-reduce(o_proj_out) -> sconv -> downstream) as capture-safe
        # element-wise stats [nonfinite_count, max_abs, l2, numel], so the analyzer
        # names the FIRST op inside the first divergent layer that goes non-finite
        # under CUDA graph -- even for the non-hidden-width attention internals.
        # Allocated EAGERLY (pre-capture) like _ink_fp. Zero cost unless
        # INKLING_FP_OPS is set.
        if os.environ.get("INKLING_FP_OPS"):
            self._ink_fp_ops = torch.zeros(len(self.layers), N_INK_FP_OPS,
                                           _INK_FP_STAT_W, dtype=torch.float32,
                                           device=device)
            self._ink_fp_ops_maxlayer = int(
                os.environ.get("INKLING_FP_OPS_MAXLAYER", "10000"))
            # iter-79 ALLOCATION-FREE op fingerprint: a fp32 scratch sized well
            # above the largest probed op numel (global rel_logits[-1] =
            # local_heads*rel_extent <= 64*1024 at TP=1; hidden=6144; q<=8192).
            # 1<<17 (512 KiB) gives >=8x margin at TP=4 and is allocated EAGERLY
            # (pre-capture) like the buffers above, so ``_ink_fp_stat`` reuses it
            # with copy_/abs_/amax and allocates nothing numel-sized in-graph.
            self._ink_fp_ops_scratch = torch.zeros(1 << 17, dtype=torch.float32,
                                                   device=device)

    def forward(self,
                attn_metadata: AttentionMetadata,
                input_ids: Optional[torch.IntTensor] = None,
                position_ids: Optional[torch.IntTensor] = None,
                inputs_embeds: Optional[torch.Tensor] = None,
                conv_cache: Optional[InklingConvStateCache] = None,
                conv_rt: Optional[InklingConvRuntime] = None,
                **kwargs) -> torch.Tensor:
        """Decoder stack. ``conv_cache`` (+ ``conv_rt``) is the runtime short-conv
        state pool: each layer reads its own four ``[max_batch, C, K-1]`` buffers
        and the shared per-forward ``conv_rt`` split, so the four short-convs of
        every layer carry per-request state across decode steps exactly like the
        paged KV cache. ``conv_cache=None`` keeps the stateless focused-replay
        behavior."""
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        hidden_states = self.embed_norm(inputs_embeds)
        # Debug-only: env-gated per-layer PREFILL activation dump, to localize the
        # full-model TP=4 vs TP=1 divergence (the focused replays are all TP=1).
        # Zero cost when INKLING_DUMP_PREFILL is unset. Dumps once, on the first
        # context forward, one file per rank (all-reduced hidden is identical
        # across ranks, so the comparison reads rank 0).
        import os as _os
        _dump_path = _os.environ.get("INKLING_DUMP_PREFILL")
        # Only the real short prompt, not the ~max_num_tokens KV-cache estimation
        # prefill: gate on a context-token count window and overwrite (the last
        # matching prefill wins). Default window (1..64) preserves the original
        # short-prompt TP-divergence localizer behavior. INKLING_DUMP_MINTOK /
        # INKLING_DUMP_MAXTOK widen it for the per-layer TRT-vs-SGLang residual
        # localizer (set both to the exact prompt token count to gate precisely and
        # skip the warmup prefill); INKLING_DUMP_ALLLAYERS=1 records every decoder
        # layer's residual stream instead of just the first eight.
        _dump_min = int(_os.environ.get("INKLING_DUMP_MINTOK", "1"))
        _dump_max = int(_os.environ.get("INKLING_DUMP_MAXTOK", "64"))
        _dump_all = _os.environ.get("INKLING_DUMP_ALLLAYERS") == "1"
        # feedback #4 Stage B: in all-layers mode ALSO capture the per-layer module
        # split -- the post-attention residual ``h_attn`` (attention kernel) and the
        # pure MLP/MoE transform ``moe_out`` (routed/shared-expert kernel) -- for
        # EVERY one of the 66 layers, answer-position only. This is what cleanly
        # separates an "attention kernel" divergence from a "MoE kernel" divergence
        # against the SGLang-triton reference (the raw residual stream carries the
        # pre/post-sconv convention offset that fakes a sharp drop). Zero cost unless
        # both INKLING_DUMP_ALLLAYERS=1 and INKLING_DUMP_MODULES=1 are set.
        _dump_modules = _os.environ.get("INKLING_DUMP_MODULES") == "1"
        # feedback #7 finer-grain L0-L4 mode: on top of the feedback-#4 module dump,
        # capture the intra-layer boundaries (attn_core, attn_sconv, mlp_norm,
        # moe_out, mlp_sconv) plus the MoE gate (router_logits, top-k ids, routing
        # weights) for layers 0..INKLING_DUMP_MAXLAYER only, to split the L3 MoE
        # jump into router-chaos vs expert-GEMM. Zero cost unless _finegrain is set.
        _dump_finegrain = _os.environ.get("INKLING_DUMP_FINEGRAIN") == "1"
        _dump_maxlayer = int(_os.environ.get("INKLING_DUMP_MAXLAYER", "999"))
        _ctx_tok = (int(attn_metadata.seq_lens[:attn_metadata.num_contexts].sum())
                    if attn_metadata.num_contexts > 0 else 0)
        _do_dump = bool(_dump_path) and _dump_min <= _ctx_tok <= _dump_max
        _rec = None
        if _do_dump:
            try:
                _rank = int(self.model_config.mapping.tp_rank)
            except Exception:
                _rank = 0
            _rec = {
                "rank": _rank,
                "input_ids":
                (input_ids.detach().cpu() if input_ids is not None else None),
                "position_ids":
                (position_ids.detach().cpu()
                 if position_ids is not None else None),
                "num_contexts": int(attn_metadata.num_contexts),
                "seq_lens": attn_metadata.seq_lens.detach().cpu(),
                "inputs_embeds": inputs_embeds.detach().float().cpu(),
                "embed_norm": hidden_states.detach().float().cpu(),
                "layers": {},
            }
        # Batched-decode divergence localizer (env-gated). Reset the per-episode
        # step counter on any context/prefill forward; probe pure-generation
        # forwards of a batch (>1 row) on the runtime state-pool path (decode),
        # AND context/prefill forwards of >=2 IDENTICAL prompts (residual).
        _dv_on = _ink_diverge_on()
        if _dv_on and attn_metadata.num_contexts > 0:
            _INK_DIVERGE["step"] = 0
            _INK_DIVERGE["reported"] = False
        _dv = (_dv_on and attn_metadata.num_contexts == 0
               and hidden_states.shape[0] > 1 and conv_rt is not None)
        # Prefill residual localizer: a PURE context forward of >=2 identical
        # prompts of equal length. The decode localizer cannot see a divergence
        # seeded here (it only fires on pure-gen forwards); this compares each
        # request's whole prefill span to request 0's.
        _pf = None
        if _dv_on and attn_metadata.num_contexts >= 2 and conv_rt is not None:
            _slens = attn_metadata.seq_lens.tolist()
            _nc = attn_metadata.num_contexts
            _clens = _slens[:_nc]
            _L = _clens[0]
            _ctok = sum(_clens)
            if (_L > 0 and all(x == _L for x in _clens)
                    and hidden_states.shape[0] == _ctok
                    and input_ids is not None
                    and input_ids.shape[0] >= _ctok):
                _ii = input_ids[:_ctok].reshape(_nc, _L)
                if bool((_ii == _ii[0:1]).all()):
                    _pf = (_nc, _L)
        _dsink = {} if _dv else None
        _pfsink = {"_ctx": _pf} if _pf else None
        _active_sink = _dsink if _dv else _pfsink
        if _dv:
            _INK_DIVERGE["step"] += 1
        # B2 localizer: fingerprint pure-generation (decode) forwards only. The
        # copy_ ops below are recorded into the captured decode graph and re-run
        # on every replay (num_contexts is 0 both at capture and replay for the
        # decode graph), so the buffer holds the LAST replay's per-layer decode
        # output -- read out eagerly in prepare_inkling_attn_decode.
        _fp_on = (self._ink_fp is not None and attn_metadata.num_contexts == 0
                  and hidden_states.shape[0] >= 1)
        for i, layer in enumerate(self.layers):
            layer_state = (conv_cache.layer_state(i)
                           if conv_cache is not None else None)
            # Sublayer detail (h_attn/moe_out): full-position for the original
            # short-prompt mode (i<8), OR answer-position-only for EVERY layer in the
            # feedback #4 all-layers module-split mode. Plain all-layers mode keeps
            # just the answer-position residual to bound the dump to ~66*H floats.
            if _do_dump and _dump_all and _dump_modules:
                if _dump_finegrain:
                    # feedback #7: intra-layer boundaries + gate for L0..maxlayer
                    # only; layers beyond maxlayer keep just the residual stream.
                    _sink = ({"_answer_pos_only": True, "_finegrain": True}
                             if i <= _dump_maxlayer else None)
                else:
                    _sink = {"_answer_pos_only": True}
            elif _do_dump and not _dump_all and i < 8:
                _sink = {}
            else:
                _sink = None
            hidden_states = layer(position_ids,
                                  hidden_states,
                                  attn_metadata,
                                  conv_state=layer_state,
                                  conv_rt=conv_rt,
                                  dump_sink=_sink,
                                  diverge_sink=_active_sink,
                                  ops_fp=(self._ink_fp_ops[i]
                                          if (_fp_on
                                              and self._ink_fp_ops is not None
                                              and not layer.attn.is_local
                                              and i <= self._ink_fp_ops_maxlayer)
                                          else None),
                                  ops_scratch=self._ink_fp_ops_scratch)
            if _fp_on:
                # Last generation row's residual after layer i (device->device,
                # captured -> replays). fp32 cast is a transient graph-pool alloc.
                self._ink_fp[i].copy_(hidden_states[-1].to(torch.float32))
            if _do_dump and (_dump_all or i < 8):
                # residual stream after layer i (non-fused: hidden_states IS the
                # stream). All-layers mode stores the answer-position (last) token
                # only, matching the SGLang forward-hook's rs[-1] capture.
                _rec["layers"][i] = (hidden_states[-1] if _dump_all
                                     else hidden_states).detach().float().cpu()
                if _sink:
                    _rec.setdefault("h_attn", {})[i] = _sink.get("h_attn")
                    _rec.setdefault("moe_out", {})[i] = _sink.get("moe_out")
                    if _sink.get("_finegrain"):
                        # feedback #7 fine-grain points (present only for L0..
                        # maxlayer; router_* only on MoE layers >= dense_mlp_idx).
                        for _k in ("attn_core", "attn_sconv", "mlp_norm",
                                   "mlp_sconv", "router_logits", "topk_idx",
                                   "routed_w"):
                            _v = _sink.get(_k)
                            if _v is not None:
                                _rec.setdefault(_k, {})[i] = _v
        out = self.norm(hidden_states)
        if _fp_on:
            self._ink_fp[len(self.layers)].copy_(out[-1].to(torch.float32))
        if _dv or _pf:
            try:
                _dv_rank = int(self.model_config.mapping.tp_rank)
            except Exception:
                _dv_rank = 0
            if _dv_rank == 0:
                if _dv:
                    _ink_report_divergence(len(self.layers), _dsink,
                                           inputs_embeds, out)
                if _pf:
                    _ink_report_prefill(len(self.layers), _pfsink,
                                        inputs_embeds, out, _pf)
        if _do_dump:
            _rec["final_norm"] = (out[-1] if _dump_all
                                  else out).detach().float().cpu()
            import torch as _torch
            # All-layers mode keys the file by context-token count so several
            # teacher-forced prompts (distinct lengths) written under one fixed
            # INKLING_DUMP_PREFILL base (set in the launcher env, seen by every TP
            # worker) land in distinct files instead of overwriting each other.
            _suffix = (f".n{_ctx_tok}.rank{_rec['rank']}" if _dump_all
                       else f".rank{_rec['rank']}")
            _torch.save(_rec, f"{_dump_path}{_suffix}")
            print(f"[inkling-dump] wrote prefill activations to "
                  f"{_dump_path}{_suffix}",
                  flush=True)
        return out


class InklingForCausalLM(DecoderModelForCausalLM[InklingModel,
                                                 InklingTextConfig]):
    """Text CausalLM: muP logit scaling + unpadded-vocab slice.

    ``embed`` and ``unembed`` are separate checkpoint tensors (never tied). The
    ``LMHead`` is built at the unpadded vocab size so its forward slices off the
    padding automatically; hidden states are divided by
    ``logits_mup_width_multiplier`` before the head (accuracy-critical).
    """

    def __init__(self, model_config: ModelConfig[InklingTextConfig]):
        config = model_config.pretrained_config
        self.mup_multiplier = float(config.logits_mup_width_multiplier)
        super().__init__(
            InklingModel(model_config),
            config=model_config,
            hidden_size=config.hidden_size,
            vocab_size=config.unpadded_vocab_size,
        )

    def prepare_inkling_attn_decode(self, attn_metadata) -> None:
        """Eagerly refresh every attention layer's stable decode-metadata buffers
        (total-KV seq_lens + per-layer page table) for this batch, BEFORE the
        model engine captures/replays the decode CUDA graph. Called from
        ``PyTorchModelEngine._prepare_tp_inputs`` alongside the short-conv pool
        publish, so the captured ``model.forward`` decode path does no host->device
        copy. Cheap and side-effect-free when there is no generation slice."""
        device = self.model.embed_tokens.weight.device
        for layer in self.model.layers:
            layer.attn._decode_meta.refresh(attn_metadata, device)
        # feedback #18 no-probe B2 candidate ``sync_meta``: block until every
        # layer's eager metadata H2D copy has landed before the captured/replayed
        # decode forward reads it. Legal here (``prepare`` runs EAGERLY, outside any
        # CUDA-graph capture); one device sync per decode step. Tests whether a
        # captured/replayed read races the async seq_lens/page_table copy. No-op
        # unless the toggle is selected.
        if _b2_fix_active("sync_meta"):
            torch.cuda.synchronize(device)
        # --- B2 CUDA-graph decode localizer (env INKLING_FP, zero cost off) ---
        # The captured decode forward runs no Python at replay, so we (1) allocate
        # the capture-safe fingerprint buffer EAGERLY here (before capture) and
        # (2) read out the PREVIOUS decode step's fingerprint HERE (eager) -- the
        # buffer was filled by that step's graph replay's device->device copies.
        # prepare(decode_k) dumps decode_{k-1}, one file per rank per step; the
        # driver compares step 0 (first decode, whose INPUT token matches the
        # cg=off run since prefill logits are identical) to pin the first layer
        # where CUDA graph corrupts, plus cross-rank residual consistency.
        import os
        fp_path = os.environ.get("INKLING_FP")
        if fp_path:
            self.model._ensure_fp_buffer(device)
            req_ids = getattr(attn_metadata, "request_ids", None)
            num_ctx = int(getattr(attn_metadata, "num_contexts", 0) or 0)
            num_gen = (len(req_ids) - num_ctx) if req_ids is not None else 0
            if num_ctx > 0:
                # A new episode's prefill (incl. the KV-estimation/warmup prefill):
                # reset the per-episode step counter so the first REAL decode is
                # step 0 and warmup dummy-decode fills before it are discarded.
                self.model._ink_fp_step = 0
                self.model._ink_fp_prev_decode = False
            else:
                if self.model._ink_fp_prev_decode:
                    try:
                        rank = int(self.model.model_config.mapping.tp_rank)
                    except Exception:  # noqa: BLE001
                        rank = 0
                    step = self.model._ink_fp_step
                    torch.save(
                        self.model._ink_fp.detach().to("cpu"),
                        f"{fp_path}.rank{rank}.step{step}")
                    # feedback #17 op-level stats dump. The [L,13,4] stats tensor
                    # is tiny (~14 KB), so dump EVERY step (no MAXSTEP truncation):
                    # the analyzer needs the full onset trace -- including steps
                    # PAST a KV-page boundary -- to classify capture (nan from the
                    # first replay, before any page boundary) vs replay-metadata
                    # (nan onset AT a page-boundary-crossing step).
                    if self.model._ink_fp_ops is not None:
                        torch.save(
                            self.model._ink_fp_ops.detach().to("cpu"),
                            f"{fp_path}.ops.rank{rank}.step{step}")
                        # One-time page_size sidecar (rank 0, first dumped step):
                        # the KV-cache page/token size, so the analyzer can place
                        # each step's KV position relative to a page boundary for
                        # the capture-vs-replay call. Best-effort; the verdict
                        # degrades to pos-only if this is absent.
                        if rank == 0 and step == 0:
                            try:
                                import json as _json
                                _kv = attn_metadata.kv_cache_manager.get_buffers(
                                    0, kv_layout="HND")
                                with open(f"{fp_path}.page_size.json",
                                          "w") as _pf:
                                    _json.dump({"page_size": int(_kv.shape[3])},
                                               _pf)
                            except Exception:  # noqa: BLE001
                                pass
                    self.model._ink_fp_step += 1
                self.model._ink_fp_prev_decode = (num_gen > 0)

    def forward(self,
                attn_metadata: AttentionMetadata,
                input_ids: Optional[torch.IntTensor] = None,
                position_ids: Optional[torch.IntTensor] = None,
                inputs_embeds: Optional[torch.Tensor] = None,
                return_context_logits: bool = False,
                conv_cache: Optional[InklingConvStateCache] = None,
                conv_rt: Optional[InklingConvRuntime] = None,
                resource_manager=None,
                **kwargs) -> torch.Tensor:
        # Real-runtime path: the short-conv state pool is owned by the registered
        # InklingConvStateManager (request lifetime shared with the KV cache).
        # The model engine's eager input-prep pre-builds ``conv_cache``/``conv_rt``
        # for this batch (so the captured forward does no host->device slot copy);
        # they arrive here as kwargs. The focused replays likewise pass them
        # explicitly. This fallback only fires for eager, never-captured warmup
        # paths that reach forward without a pre-built split.
        if conv_cache is None and resource_manager is not None:
            conv_cache, conv_rt = _resolve_conv_runtime(resource_manager,
                                                        attn_metadata)
        hidden_states = self.model(
            attn_metadata=attn_metadata,
            input_ids=input_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            conv_cache=conv_cache,
            conv_rt=conv_rt,
        )
        hidden_states = hidden_states / self.mup_multiplier
        return self.logits_processor.forward(hidden_states, self.lm_head,
                                             attn_metadata,
                                             return_context_logits)


@register_auto_model("InklingForConditionalGeneration")
class InklingForConditionalGeneration(InklingForCausalLM):
    """Registered entry point for the multimodal ``inkling_mm_model`` checkpoint.

    For the text-only GSM8K/MMLU bring-up this routes straight to the text
    :class:`InklingForCausalLM` over the ``text_config`` sub-config and consumes
    only ``model.llm.*`` weights; audio / vision / MTP keys are intentionally
    unused (Phase 3). See ``checkpoints/hf/inkling_weight_mapper.py`` for the
    HF→TRT name mapping and consumed/deferred accounting.
    """

    @classmethod
    def get_model_defaults(cls, llm_args: 'TorchLlmArgs') -> dict:
        # Inkling's hybrid per-layer KV-head split (local sliding-window layers
        # carry 16 KV heads, global layers 8) structurally requires
        # KVCacheManagerV2's per-layer ``num_kv_heads`` geometry -- V1's unified
        # pool would coerce it to a single value and mis-size the per-layer KV
        # bytes (a correctness bug, not just efficiency). The concrete manager
        # class is already forced to V2 for Inkling in
        # ``_util._non_hybrid_kv_cache_manager_cls`` (the ``is_inkling`` branch),
        # and ``_fallback_if_unsupported_kv_cache_manager_v2`` raises rather than
        # silently downgrading. Declaring the default here makes the *resolved*
        # ``kv_cache_config.use_kv_cache_manager_v2`` flag agree with that reality
        # across every launch path (LLM API, trtllm-serve, trtllm-eval), so the
        # flag's readers -- ``model_loader`` startup log, ``get_server_info``'s
        # ``kv_cache_hash_algo`` report, and the KV-cache-event hash algo -- no
        # longer report 'auto -> False' while the engine actually runs V2.
        #
        # NOTE (iter59 tried, iter61 REVERTED, iter63 diagnosis corrected):
        # a ``moe_config.disable_finalize_fusion=True`` default was added on the
        # theory that the CUTLASS FUSED FC2+finalize kernel (non-deterministic for
        # top-k > 2; Inkling routes top-6) drove the served nc>1 GSM8K `!!!!`/EMPTY
        # collapse. Fused STAYS the default: disabling finalize fusion is a
        # CORRECTNESS REGRESSION -- a single greedy chat ("What is 3+4?") collapses
        # to a `!!!!` loop at num_concurrent=1 with unfused finalize (job 5489709
        # ARM B, resolved_dff=True, bang_run=6), while FUSED answers `7` cleanly
        # (ARM A). The unfused-finalize path has its OWN separate bug.
        #
        # BUT the nc>1 corruption IS the fused MoE combine's cross-row
        # non-determinism -- NOT a phantom per-slot short-conv/KV state bug (this
        # CORRECTS the iter61 note). Decisive evidence: job 5489709 ran the
        # divergence localizer on a batch of 8 IDENTICAL decode rows with
        # ``d_embed=0`` (identical inputs) and ``pre_state=0`` (identical carried
        # conv state), and the fused path STILL forks first at ``L2/mlp_core`` --
        # the first (bf16) MoE layer -- by ~3.1e-2 (~1 ULP at that activation
        # scale), which the 64-layer residual stack amplifies ~200x to
        # ``final_d=6.4`` in ONE step, flipping the greedy argmax. Layers 0/1
        # (dense) and attention stay bit-identical, so the origin is the MoE
        # combine, not short-conv/KV. The earlier nrows=4 ``d_embed~2.05`` is the
        # amplified DOWNSTREAM state many steps later, not a prefill divergence. At
        # nc=1 there is one row so no cross-row fork (correct); at nc>1 identical
        # rows diverge and a large fraction go off-track -> served 0.60. SGLang
        # runs flashinfer_trtllm_routed (deterministic combine) and holds 0.955 at
        # nc=4, so the FIX DIRECTION is a deterministic MoE combine that KEEPS nc=1
        # correct -- exactly the human "confirm CUTLASS-vs-flashinfer kernel"
        # guidance; pursue a Python/config-level deterministic combine first.
        return {
            "kv_cache_config": {
                "use_kv_cache_manager_v2": True
            },
        }

    def __init__(self, model_config: ModelConfig[InklingConfig]):
        text_model_config = _text_sub_model_config(model_config)
        super().__init__(text_model_config)
        self._top_model_config = model_config

    def load_weights(self, weights: dict, weight_mapper=None):
        from tensorrt_llm._torch.models.checkpoints.hf.inkling_weight_mapper import \
            InklingHfWeightMapper
        if weight_mapper is None:
            weight_mapper = InklingHfWeightMapper()
            weight_mapper.init_model_and_config(self, self.model_config)
        # Keep only the text tower; drop audio/vision/mtp (intentionally unused),
        # then remap the checkpoint's SGLang-style keys to the TRT module tree
        # (fuse q/k/v, split dense w13, unfuse NVFP4 experts). This preprocess
        # step must run here (like modeling_nemotron_h) -- the base
        # _load_weights_impl_v2 assumes already-mapped names.
        text_weights = filter_weights("model.llm", weights)
        text_weights = weight_mapper.preprocess_weights(text_weights)
        super().load_weights(text_weights, weight_mapper=weight_mapper)


def _text_sub_model_config(
        model_config: ModelConfig[InklingConfig]
) -> ModelConfig[InklingTextConfig]:
    """Build a text-only ``ModelConfig`` from the multimodal one, preserving the
    mapping / quant config so NVFP4 expert loading and TP sharding are intact."""
    import copy
    text_config = model_config.pretrained_config.text_config
    text_model_config = copy.copy(model_config)
    text_model_config.pretrained_config = text_config
    return text_model_config
