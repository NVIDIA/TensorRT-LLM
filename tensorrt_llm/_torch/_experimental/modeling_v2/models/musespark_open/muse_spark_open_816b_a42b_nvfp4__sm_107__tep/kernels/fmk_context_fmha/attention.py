# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``fmk_context_fmha``: the context rows of the full-attention layers' mixed steps on the
multi-instance paged context FMHA (``ops/attention/context_fmha_fmk``), sm_107.

A lever (``levers.LEVERS``): lossless-numeric, on under ``auto`` on aggregated sm107 TEP4
servers, where its accuracy gate holds; the plan installs it (``layers.adopt_model``). Its
pure context sibling ``fmk_pure_context_fmha`` (last section) is a lever of its own.

Where it acts. ``mixed_step_gen_rows`` puts a ``ThopContextPhaseFmha`` before the thop
fallback of every full-attention layer's FMHA manager; a mixed step with
``MIN_GENERATIONS``..``MAX_GENERATIONS`` generation requests runs its context rows through
it (the thop op with ``attention_input_type=context_only``). ``install`` replaces that
library with ``FmkContextPhaseFmha``, a subclass offering the same steps (so the manager's
selection and its cache key are unchanged), whose ``run_context`` runs:

1. ``thop.trtllm_gen_context_preprocess``: the decoder-info kernel (cumulative Q / KV
   lengths, the BMM scales) and the QKV preprocessing kernel (RoPE as configured, K / V
   appended to the paged cache, Q quantized to E4M3) with the parameters the thop op's
   context phase uses ahead of its FMHA; it returns the layer's flat KV pool view and its
   block table. Its post-process counterpart acts on sparse KV only, so none runs.
2. ``plan_context_fmha`` on the step's host lengths (``prompt_lens_cpu_runtime``,
   ``kv_lens_runtime``): planned by the first full-attention layer of a step and reused by
   the others (their lengths are the same); one pinned non-blocking copy to the device.
3. ``paged_context_fmha_fmk`` into the context rows of the layer's output, with the softmax
   scale (``bmm1[1]``) and the output scale (``bmm2[0]``) read on the device.

No host synchronisation: every device value is read in stream order. A call this path
cannot serve runs the thop op as before (``ThopContextPhaseFmha.run_context``), logged
once per reason: a stream capture in progress, a mask other than causal, attention sinks,
softmax statistics requested, cross attention, M-RoPE, chunked attention, a sliding window
shorter than the context, no paged context FMHA, a page size other than 32, a Q-only input,
an output that is not contiguous BF16, host lengths that disagree with the call, or a
workspace smaller than the preprocess needs. Mixed steps outside the band do not reach
it; pure context steps reach it through its ``PURE_CONTEXT`` param (below) or through
``fmk_pure_context_fmha`` (last section).

Numerics: the same precision as the thop op's cubin with another summation order and P
rounding point (the same E4M3 Q / K / V and P, fp32 accumulation, BF16 O, the same
scales), so not bitwise; accuracy-gated (``levers.LEVERS``). Q, the appended K / V and the
scales are those of the thop op's preprocessing.

Install after ``mixed_step_gen_rows`` and before any CUDA graph is captured: it reserves
the partial-result workspace (``max_slots`` slots, about 57 MB at 212 SMs) and compiles
the kernel and its merge on a small synthetic call.

Pure context steps (``PURE_CONTEXT``, off by default). With the lever's ``PURE_CONTEXT`` param at 1
the library is also offered on steps that hold context rows only (no generation request, beam
width 1, any input type but ``generation_only``): the FMHA manager then hands such a step's whole
batch to it (a context-phase library and no generation phase), so every full-attention layer of a
ctx-only / prefill step runs the multi-instance kernel instead of the thop op's trtllm-gen cubin. The
cache key separates these steps (generation batch size 0) from the mixed band. Same numerics as on
mixed steps; the accuracy gate covers mixed steps only, so the param stays 0 until a pure-context
gate (GSM8K-long + SPEED-Bench AL on prefill-heavy traffic) passes.

Pure context steps (``fmk_pure_context_fmha``, a separate lever). A disaggregated context
server runs only steps with context requests and no generation request. Those never reach
``ThopContextPhaseFmha``: the FMHA manager hands such a step's whole batch to the thop
fallback. ``install_pure_context`` puts an ``FmkPureContextFmha`` just before that fallback
on every full-attention layer. It offers the context phase of a step ``takes_pure_context``
accepts (context requests, no generation request, beam width 1), which the manager then
runs alone, and serves it with the three steps above. It first grows the attention
workspace to the preprocess's layout in place, as the thop op grows it for its own call. A
call it declines runs the fallback with the manager's own arguments: the call the thop op
served before. It does not need
``mixed_step_gen_rows``; with ``fmk_context_fmha`` installed too, both share one workspace
and one step plan.

Kernel build. Every call and the warm-up run the build ``kernel_kwargs`` names: with
``context_fmha_fmk.KF`` (``MUSESPARK_OPEN_FMK_KF``; on unless unset with an early-QK or fp16
exponent knob) the KF ctx-fmha build ``KERNEL_ID`` (``KERNEL_KWARGS``: Rubin K=64 FP8 MMAs,
pair-shared S), else the sm100 build with the ``context_fmha_fmk`` knobs (early QK, fp16
exponent). The two are order-only against each other.
"""

from __future__ import annotations

import time
from typing import Optional, Sequence, Tuple

import torch
from torch.utils._python_dispatch import _disable_current_modes

from tensorrt_llm._torch.attention.backends.fmha.fallback import FallbackFmha
from tensorrt_llm._torch.attention.backends.fmha.interface import FmhaPhase
from tensorrt_llm._torch.attention.backends.fmha.phased import FmhaParams, PhasedFmha
from tensorrt_llm._torch.attention.backends.fmha.utils import (
    get_attention_chunk_size,
    get_bmm1_scale,
)
from tensorrt_llm._torch.attention.backends.interface import AttentionInputType
from tensorrt_llm.bindings.internal import thop
from tensorrt_llm.functional import AttentionMaskType
from tensorrt_llm.logger import logger
from tensorrt_llm.quantization.mode import QuantMode

from ...dispatch import sm_of
from ...ops.attention import context_fmha_fmk as _fmk
from ..mixed_step_gen_rows import attention as _mixed

#: The SM version the kernel is written for.
SERVED_SM = 107
#: Query instances per CTA and key tile of the build this path runs.
NUM_INST = 4
TILE_N = 64
#: Kernel build of this overlay (ctx lane port, 2026-11-15): KF ctx-fmha-vr leader f086a525 (Rubin-native K=64 FP8 MMA,
#: pair-shared S buffers), launched with these keyword arguments by every call and by the warm-up when
#: ``context_fmha_fmk.KF`` (``kernel_kwargs``).
KERNEL_ID = "kf-f086a525"
KERNEL_KWARGS = dict(kv_stage=12, s_stages=1, turn_groups=0, self_issue=False, pair_s=True)
#: Engagement log (checklist rule 7): one line per ENGAGEMENT_EVERY context phases (16 full-attention layers per step).
ENGAGEMENT_EVERY = 1600
_ENGAGEMENT = {"ran": 0, "declined": 0, "since": 0}


def kernel_kwargs() -> dict:
    """The build arguments of every call and the warm-up: ``KERNEL_KWARGS`` (the KF build, its
    early-QK and fp16-exponent options off) under ``context_fmha_fmk.KF``, else none (the sm100
    build with the ``context_fmha_fmk`` knobs)."""
    if _fmk.KF:
        return dict(KERNEL_KWARGS, p_smem=False, exp_f16x2=False, kf=True)
    return {}


def kernel_id() -> str:
    """The build ``kernel_kwargs`` selects, for log lines."""
    return KERNEL_ID if _fmk.KF else "sm100"


def _log_engagement(ran: bool) -> None:
    e = _ENGAGEMENT
    e["ran" if ran else "declined"] += 1
    e["since"] += 1
    if e["since"] >= ENGAGEMENT_EVERY:
        e["since"] = 0
        logger.info(
            f"[musespark_open] fmk_context_fmha engagement ({kernel_id()}): ran {e['ran']} declined {e['declined']} "
            f"context phases so far ({e['ran'] / max(e['ran'] + e['declined'], 1) * 100:.2f} % on the multi-instance "
            f"kernel)"
        )


#: The per-rank (query, KV) head counts the planner sizes its work units for
#: (``plan_context_fmha``'s defaults, which ``StepPlans`` and the warm-up plan with).
PLANNED_HEADS = (32, 4)


def _workspace_bytes(
    dtype: torch.dtype, batch: int, num_tokens: int, num_heads: int, head_dim: int, rope_dim: int
) -> int:
    """Bytes of the preprocess's context workspace layout (no trtllm-gen FMHA workspace)."""
    layout = thop.get_trtllm_gen_context_workspace_layout(
        dtype, batch, num_tokens, num_heads, head_dim, rope_dim, True, True, True
    )
    return int(layout["total_size"])


class StepPlans:
    """The plan of the latest lengths, shared by the full-attention layers of a step.

    A step's full-attention layers run one after another with the same lengths, so the
    first plans and the rest reuse the plan's device buffer (read in stream order). A
    step with other lengths replaces it; the replaced buffer returns to the caching
    allocator on the same stream.
    """

    def __init__(self, num_ctas: int):
        self.num_ctas = num_ctas
        self._key: Optional[Tuple] = None
        self._plan: Optional[_fmk.FmkPlan] = None
        #: Plans built, and the host time they took (read by the served-path log).
        self.planned = 0
        self.plan_seconds = 0.0

    def get(self, q_lens: Tuple[int, ...], kv_lens: Tuple[int, ...], device) -> _fmk.FmkPlan:
        key = (q_lens, kv_lens, device)
        if key != self._key:
            start = time.perf_counter()
            self._plan = _fmk.plan_context_fmha(
                q_lens,
                kv_lens,
                num_ctas=self.num_ctas,
                num_inst=NUM_INST,
                tile_n=TILE_N,
                device=device,
            )
            self.plan_seconds += time.perf_counter() - start
            self.planned += 1
            self._key = key
        return self._plan


class _FmkContextPhase:
    """The context phase on the multi-instance context FMHA: the part ``FmkContextPhaseFmha``
    (mixed steps) and ``FmkPureContextFmha`` (pure context steps) share.

    A library mixes it in front of a ``PhasedFmha`` holding ``_fallback`` and
    ``context_calls``, calls ``_init_fmk`` and says in ``_run_declined`` what serves a call
    ``_decline`` refuses.
    """

    #: The lever the library serves and the steps it takes (its log lines).
    LEVER = ""
    STEP = ""

    def _init_fmk(self, plans: StepPlans, workspace: _fmk.FmkWorkspace) -> None:
        self._plans = plans
        self._workspace = workspace
        #: Context phases run on the multi-instance kernel, and declined (thop op).
        self.fmk_calls = 0
        self.declined_calls = 0

    def _run_declined(self, params: FmhaParams) -> None:
        raise NotImplementedError

    def _step_kind(self, params: FmhaParams) -> Tuple[str, str]:
        """The step a call belongs to: its text in the log lines and its log key part."""
        return f"{self.STEP} context rows", "step"

    def run_context(self, params: FmhaParams) -> None:
        text, kind = self._step_kind(params)
        lengths = self._host_lengths(params)
        reason = self._decline(params, lengths)
        if reason is not None:
            self.declined_calls += 1
            _log_engagement(False)
            logger.info_once(
                f"[musespark_open] {self.LEVER}: {text} ran the thop op "
                f"(layer {self.attn.layer_idx}): {reason}",
                key=f"musespark_open.{self.LEVER}.decline.{kind}.{reason}",
            )
            self._run_declined(params)
            return
        self._run_fmk(params, *lengths)
        self.fmk_calls += 1
        _log_engagement(True)
        self.context_calls += 1
        logger.info_once(
            f"[musespark_open] {self.LEVER}: {text} "
            f"({params.batch_size} requests, {params.num_tokens} rows, layer "
            f"{self.attn.layer_idx}) ran the multi-instance context FMHA",
            key=f"musespark_open.{self.LEVER}.ran.{kind}",
        )

    @staticmethod
    def _host_lengths(params: FmhaParams) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
        meta, batch = params.meta, params.batch_size
        q_lens = tuple(meta.prompt_lens_cpu_runtime[:batch].tolist())
        kv_lens = tuple(meta.kv_lens_runtime[:batch].tolist())
        return q_lens, kv_lens

    def _decline(
        self, params: FmhaParams, lengths: Tuple[Sequence[int], Sequence[int]]
    ) -> Optional[str]:
        if torch.cuda.is_current_stream_capturing():
            return "a stream capture is in progress"
        attn, fwd = params.attn, params.fwd
        if params.is_cross or fwd.cross_kv is not None:
            return "cross attention"
        if AttentionMaskType(fwd.mask_type) != AttentionMaskType.causal:
            return "a mask other than causal"
        if fwd.attention_sinks is not None:
            return "attention sinks"
        if fwd.softmax_stats_tensor is not None:
            return "softmax statistics requested"
        if fwd.mrope_rotary_cos_sin is not None:
            return "M-RoPE"
        if get_attention_chunk_size(attn) != 0:
            return "chunked attention"
        if 0 < params.cyclic_attention_window_size < params.max_past_kv_length:
            return "a sliding window shorter than the context"
        if not params.meta.use_paged_context_fmha:
            return "no paged context FMHA"
        if params.tokens_per_block != _fmk.PAGE:
            return f"a page size of {params.tokens_per_block}"
        qkv = params.qkv_input
        if qkv is None:
            return "a Q-only input"
        out = params.output
        if out is None or out.dtype != torch.bfloat16 or not out.is_contiguous():
            return "an output that is not contiguous BF16"
        q_lens, kv_lens = lengths
        if (
            len(q_lens) != params.batch_size
            or sum(q_lens) != params.num_tokens
            or any(q <= 0 or k < q for q, k in zip(q_lens, kv_lens))
        ):
            return "host lengths that disagree with the call"
        if params.total_num_blocks * attn.num_kv_heads >= 2**31:
            return "a KV pool of more than 2^31 pages"
        need = _workspace_bytes(
            qkv.dtype,
            params.batch_size,
            params.num_tokens,
            attn.num_heads,
            attn.head_dim,
            attn.rope_params.dim,
        )
        if params.workspace.numel() * params.workspace.element_size() < need:
            return "a workspace smaller than the preprocess needs"
        return None

    def _run_fmk(self, params: FmhaParams, q_lens, kv_lens) -> None:
        attn, meta, fwd = params.attn, params.meta, params.fwd
        rope = attn.rope_params
        batch, num_tokens = params.batch_size, params.num_tokens
        heads, head_dim = attn.num_heads, attn.head_dim
        (q, kv_pool, block_tables, _, bmm1_scale, bmm2_scale, _, cu_q, *_) = (
            thop.trtllm_gen_context_preprocess(
                qkv_input=params.qkv_input,
                workspace=params.workspace,
                sequence_lengths=params.sequence_lengths,
                context_lengths=params.context_lengths,
                kv_cache_block_offsets=meta.kv_cache_block_offsets,
                host_kv_cache_pool_pointers=meta.host_kv_cache_pool_pointers,
                host_kv_cache_pool_mapping=meta.host_kv_cache_pool_mapping,
                kv_scale_orig_quant=fwd.kv_scale_orig_quant,
                kv_scale_quant_orig=fwd.kv_scale_quant_orig,
                attention_output_orig_quant=fwd.out_scale,
                rotary_inv_freq=attn.rotary_inv_freq,
                rotary_cos_sin=attn.rotary_cos_sin,
                mrope_rotary_cos_sin=None,
                layer_idx=attn.local_layer_idx,
                num_heads=heads,
                num_kv_heads=attn.num_kv_heads,
                head_size=head_dim,
                tokens_per_block=params.tokens_per_block,
                mask_type=fwd.mask_type,
                kv_cache_quant_mode=attn.quant_mode,
                max_attention_window_size=params.max_attention_window_size,
                cyclic_attention_window_size=params.cyclic_attention_window_size,
                num_tokens=num_tokens,
                batch_size=batch,
                input_seq_length=params.input_seq_length,
                max_past_kv_length=params.max_past_kv_length,
                rotary_embedding_dim=rope.dim,
                rotary_embedding_base=rope.theta,
                rotary_embedding_scale_type=int(rope.scale_type),
                rotary_embedding_scale=rope.scale,
                rotary_embedding_max_positions=rope.max_positions,
                position_embedding_type=attn.position_embedding_type,
                bmm1_scale=get_bmm1_scale(attn),
                bmm2_scale=1.0,
                attention_chunk_size=0,
                fp8_context_fmha=True,
                paged_context_fmha=True,
                is_mla_enable=False,
                multi_processor_count=self._plans.num_ctas,
                total_num_blocks=params.total_num_blocks,
                kv_factor=params.kv_factor,
                need_build_kv_cache_metadata=True,
                cross_kv=None,
                cross_attention=False,
                skip_fmha_workspace=True,
            )
        )
        q = (
            q.view(torch.uint8)
            .flatten()[: num_tokens * heads * head_dim]
            .view(torch.float8_e4m3fn)
            .view(num_tokens, heads, head_dim)
        )
        plan = self._plans.get(q_lens, kv_lens, q.device)
        _fmk.paged_context_fmha_fmk(
            q,
            kv_pool,
            block_tables,
            cu_q,
            params.sequence_lengths[:batch],
            plan,
            bmm1_scale[1:2],
            bmm2_scale[0:1],
            params.output.view(num_tokens, heads, head_dim),
            workspace=self._workspace,
            **kernel_kwargs(),
        )


class FmkContextPhaseFmha(_FmkContextPhase, _mixed.ThopContextPhaseFmha):
    """``ThopContextPhaseFmha`` whose context phase runs the multi-instance context FMHA.

    Offered on the same steps as its base, and with ``pure_context`` (the lever's
    ``PURE_CONTEXT`` param) also on pure context steps; a call it cannot serve runs the
    base's thop context phase.
    """

    LEVER = "fmk_context_fmha"
    STEP = "a mixed step's"

    def __init__(
        self,
        attn,
        fallback,
        plans: StepPlans,
        workspace: _fmk.FmkWorkspace,
        pure_context: bool = False,
    ):
        _mixed.ThopContextPhaseFmha.__init__(self, attn, fallback)
        self._init_fmk(plans, workspace)
        #: Also offered on pure context steps (the lever's ``PURE_CONTEXT`` param).
        self.pure_context = bool(pure_context)

    def _is_supported(self, q, k, v, metadata, forward_args, *, phase=None) -> bool:
        if self.pure_context and phase == FmhaPhase.CONTEXT and takes_pure_context(metadata):
            if forward_args.attention_input_type == AttentionInputType.generation_only:
                return False
            if forward_args.output is None or forward_args.output_sf is not None:
                return False
            return self._fallback.is_supported(q, k, v, metadata, forward_args)
        return super()._is_supported(q, k, v, metadata, forward_args, phase=phase)

    def _step_kind(self, params: FmhaParams) -> Tuple[str, str]:
        if takes_pure_context(params.meta):
            return "a pure context step", "pure"
        return "a mixed step's context rows", "mixed"

    def _run_declined(self, params: FmhaParams) -> None:
        """The thop op's context phase, for a call ``_decline`` refused."""
        _mixed.ThopContextPhaseFmha.run_context(self, params)


def takes_pure_context(attn_metadata) -> bool:
    """Whether a step is a pure context step: context requests, no generation request, beam
    width 1. Host-side reads only; both counts are FMHA cache key fields (zero is a grid
    point of its own), so a cached selection never crosses this test."""
    return (
        getattr(attn_metadata, "num_contexts", 0) > 0
        and getattr(attn_metadata, "num_generations", 0) == 0
        and getattr(attn_metadata, "beam_width", 1) == 1
    )


class FmkPureContextFmha(_FmkContextPhase, PhasedFmha):
    """The multi-instance context FMHA for pure context steps (``takes_pure_context``).

    Sits just before the thop fallback. The manager selects its context phase with no
    generation phase and runs it alone; a call ``_decline`` refuses runs the fallback with
    the arguments the manager passed (the whole batch, as the thop op served it before).
    """

    LEVER = "fmk_pure_context_fmha"
    STEP = "a pure context step's"
    supports_workspace_reclamation = True

    def __init__(
        self, attn, fallback: FallbackFmha, plans: StepPlans, workspace: _fmk.FmkWorkspace
    ):
        PhasedFmha.__init__(self, attn)
        self._fallback = fallback
        #: Context phases served (multi-instance kernel or declined to the fallback).
        self.context_calls = 0
        self._init_fmk(plans, workspace)
        self._call: Optional[Tuple] = None

    def _is_supported(self, q, k, v, metadata, forward_args, *, phase=None) -> bool:
        if phase != FmhaPhase.CONTEXT or not takes_pure_context(metadata):
            return False
        if forward_args.attention_input_type not in (
            AttentionInputType.mixed,
            AttentionInputType.context_only,
        ):
            return False
        if forward_args.output is None or forward_args.output_sf is not None:
            return False
        return self._fallback.is_supported(q, k, v, metadata, forward_args)

    def forward(self, q, k, v, metadata, forward_args) -> None:
        self._call = (q, k, v, metadata, forward_args)
        try:
            super().forward(q, k, v, metadata, forward_args)
        finally:
            self._call = None

    def run_context(self, params: FmhaParams) -> None:
        self._grow_workspace(params)
        super().run_context(params)

    @staticmethod
    def _grow_workspace(params: FmhaParams) -> None:
        """Grow the attention workspace to the preprocess's layout in place, as the thop op
        grows it for its own call. A step's first attention call can find it empty: the
        eager workspace reclaimer shrinks it to zero elements (storage kept) at the start of
        a forward, and the thop op grows it on demand."""
        qkv, workspace, attn = params.qkv_input, params.workspace, params.attn
        if qkv is None or workspace is None or torch.cuda.is_current_stream_capturing():
            return
        need = _workspace_bytes(
            qkv.dtype,
            params.batch_size,
            params.num_tokens,
            attn.num_heads,
            attn.head_dim,
            attn.rope_params.dim,
        )
        have = workspace.numel() * workspace.element_size()
        if have < need:
            workspace.resize_(-(-need // workspace.element_size()))

    def _run_declined(self, params: FmhaParams) -> None:
        """The fallback on the manager's arguments: the thop op's call before this path."""
        self._fallback.forward(*self._call)
        self.context_calls += 1

    def run_generation(self, params: FmhaParams) -> None:
        raise RuntimeError("FmkPureContextFmha serves pure context steps only")


def pure_context_library(attn) -> Optional[FmkPureContextFmha]:
    """The ``FmkPureContextFmha`` in ``attn``'s FMHA manager, or None."""
    for fmha in attn._fmha_manager.fmha_libs:
        if isinstance(fmha, FmkPureContextFmha):
            return fmha
    return None


def _refusal(attention, sm: Optional[int], pure: bool = False) -> Optional[str]:
    attn = attention.attn
    if sm != SERVED_SM:
        return f"the kernel is written for sm{SERVED_SM}, not sm{sm}"
    if pure:
        if not any(isinstance(fmha, FallbackFmha) for fmha in attn._fmha_manager.fmha_libs):
            return "no thop fallback among its FMHA libraries"
    elif _mixed.context_library(attn) is None:
        return "no ThopContextPhaseFmha in its FMHA libraries (mixed_step_gen_rows not installed)"
    if getattr(attn, "is_mla_enable", False):
        return "MLA"
    if attn.head_dim != _fmk.HEAD_DIM:
        return f"head dim {attn.head_dim}"
    if attn.num_heads % attn.num_kv_heads or (attn.num_heads // attn.num_kv_heads) % NUM_INST:
        return f"{attn.num_heads} query heads over {attn.num_kv_heads} KV heads"
    if (attn.num_heads, attn.num_kv_heads) != PLANNED_HEADS:
        return (
            f"{attn.num_heads} query heads over {attn.num_kv_heads} KV heads per rank (the "
            f"planner is sized for {PLANNED_HEADS[0]} over {PLANNED_HEADS[1]})"
        )
    if not QuantMode(attn.quant_mode).has_fp8_kv_cache():
        return "a KV cache that is not FP8"
    if not hasattr(thop, "trtllm_gen_context_preprocess"):
        return "the binding lacks trtllm_gen_context_preprocess"
    return None


def _warm_up(device: torch.device, num_heads: int, num_kv_heads: int, num_ctas: int, workspace):
    """Compile the kernel and its merge on a small synthetic call (one request of one row over
    12,800 zero keys, which the planner cuts) and check that it ran."""
    pages = 400
    q = torch.zeros(1, num_heads, _fmk.HEAD_DIM, dtype=torch.float8_e4m3fn, device=device)
    pool = torch.zeros(
        2 * pages, num_kv_heads, _fmk.PAGE, _fmk.HEAD_DIM, dtype=torch.float8_e4m3fn, device=device
    )
    ids = torch.arange(pages, dtype=torch.int32, device=device)
    table = torch.stack([2 * ids, 2 * ids + 1]).unsqueeze(0).contiguous()
    cu_q = torch.tensor([0, 1], dtype=torch.int32, device=device)
    kv_lens = torch.tensor([pages * _fmk.PAGE], dtype=torch.int32, device=device)
    scales = torch.tensor([1.0, 1.0], dtype=torch.float32, device=device)
    out = torch.full(
        (1, num_heads, _fmk.HEAD_DIM), float("nan"), dtype=torch.bfloat16, device=device
    )
    plan = _fmk.plan_context_fmha(
        (1,),
        (pages * _fmk.PAGE,),
        num_ctas=num_ctas,
        num_inst=NUM_INST,
        tile_n=TILE_N,
        device=device,
    )
    _fmk.paged_context_fmha_fmk(
        q,
        pool,
        table,
        cu_q,
        kv_lens,
        plan,
        scales[0:1],
        scales[1:2],
        out,
        workspace=workspace,
        **kernel_kwargs(),
    )
    logger.info(
        f"[musespark_open] fmk_context_fmha: kernel build {kernel_id()} {kernel_kwargs()} warm-up OK"
    )
    if plan.merge_meta is None or not bool(torch.isfinite(out).all()):
        raise ValueError(
            "[musespark_open] fmk_context_fmha: the warm-up call did not run as planned "
            f"(merges {0 if plan.merge_meta is None else plan.merge_meta.shape[0]})"
        )


def _prepare(
    device: torch.device, num_heads: int, num_kv_heads: int, num_ctas: int
) -> _fmk.FmkWorkspace:
    """The reserved partial-result workspace, after a warm-up call on it.

    The loader builds the model under ``MetaInitMode``, which turns ``torch.empty`` into
    meta tensors (no storage). The workspace and the warm-up are device-side setup, not
    model construction, so the active dispatch modes are popped while they run.
    """
    with _disable_current_modes():
        workspace = _fmk.FmkWorkspace()
        workspace.reserve(_fmk.max_slots(num_ctas), NUM_INST, device)
        if workspace.part_o.is_meta or workspace.part_ml.is_meta:
            raise ValueError("[musespark_open] fmk_context_fmha: the workspace has no storage")
        _warm_up(device, num_heads, num_kv_heads, num_ctas, workspace)
    return workspace


def install(model: torch.nn.Module, compute_capability, pure_context: bool = False) -> int:
    """Replace the ``ThopContextPhaseFmha`` of every full-attention layer's FMHA manager with
    an ``FmkContextPhaseFmha`` (also offered on pure context steps when ``pure_context``).
    Returns the number of layers changed.

    Run after ``mixed_step_gen_rows.install`` and before any CUDA graph is captured. Every
    layer is checked before any changes; idempotent.

    Raises:
        ValueError: if a full-attention layer has no ``ThopContextPhaseFmha``, or its
            attention or the device is not one the kernel serves.
    """
    sm = sm_of(compute_capability)
    chosen = [layer.self_attn for layer in model.layers if not layer.self_attn.is_sliding]
    refused = [
        f"layer {attention.layer_idx}: {reason}"
        for attention in chosen
        if (reason := _refusal(attention, sm)) is not None
    ]
    if refused:
        raise ValueError(
            "[musespark_open] fmk_context_fmha cannot take the mixed steps' context rows: "
            + "; ".join(refused)
        )
    device = torch.device("cuda", torch.cuda.current_device())
    num_ctas = torch.cuda.get_device_properties(device).multi_processor_count
    first = chosen[0].attn
    workspace = _prepare(device, first.num_heads, first.num_kv_heads, num_ctas)
    plans = StepPlans(num_ctas)
    for attention in chosen:
        attn = attention.attn
        libs = attn._fmha_manager.fmha_libs
        index = next(
            i for i, fmha in enumerate(libs) if isinstance(fmha, _mixed.ThopContextPhaseFmha)
        )
        if isinstance(libs[index], FmkContextPhaseFmha):
            libs[index].pure_context = bool(pure_context)
            attn._fmha_manager._cache.clear()
            continue
        libs[index] = FmkContextPhaseFmha(
            attn, libs[index]._fallback, plans, workspace, pure_context=pure_context
        )
        attn._fmha_manager._cache.clear()
    logger.info(
        f"[musespark_open] fmk_context_fmha: the context rows of mixed steps with "
        f"{_mixed.MIN_GENERATIONS}..{_mixed.MAX_GENERATIONS} generation requests"
        f"{' and every pure context step' if pure_context else ''} run the "
        f"multi-instance context FMHA on {len(chosen)} full-attention layers ({num_ctas} CTAs, "
        f"workspace {workspace.nbytes / 2**20:.1f} MiB; {_fmk.build_note()})"
    )
    return len(chosen)


def install_pure_context(model: torch.nn.Module, compute_capability) -> int:
    """Put an ``FmkPureContextFmha`` just before the thop fallback of every full-attention
    layer's FMHA manager. Returns the number of layers changed.

    Run after ``install`` when both are planned (the two then share its workspace and step
    plan), and before any CUDA graph is captured. Every layer is checked before any
    changes; idempotent.

    Raises:
        ValueError: if a full-attention layer has no thop fallback, or its attention or the
            device is not one the kernel serves.
    """
    sm = sm_of(compute_capability)
    chosen = [layer.self_attn for layer in model.layers if not layer.self_attn.is_sliding]
    refused = [
        f"layer {attention.layer_idx}: {reason}"
        for attention in chosen
        if (reason := _refusal(attention, sm, pure=True)) is not None
    ]
    if refused:
        raise ValueError(
            "[musespark_open] fmk_pure_context_fmha cannot take the pure context steps: "
            + "; ".join(refused)
        )
    device = torch.device("cuda", torch.cuda.current_device())
    num_ctas = torch.cuda.get_device_properties(device).multi_processor_count
    shared = next(
        (lib for attention in chosen if (lib := context_library(attention.attn)) is not None),
        None,
    )
    if shared is not None:
        plans, workspace = shared._plans, shared._workspace
    else:
        first = chosen[0].attn
        workspace = _prepare(device, first.num_heads, first.num_kv_heads, num_ctas)
        plans = StepPlans(num_ctas)
    for attention in chosen:
        attn = attention.attn
        if pure_context_library(attn) is not None:
            continue
        libs = attn._fmha_manager.fmha_libs
        index = next(i for i, fmha in enumerate(libs) if isinstance(fmha, FallbackFmha))
        libs.insert(index, FmkPureContextFmha(attn, libs[index], plans, workspace))
        attn._fmha_manager._cache.clear()
    logger.info(
        f"[musespark_open] fmk_pure_context_fmha: pure context steps run the multi-instance "
        f"context FMHA on {len(chosen)} full-attention layers ({num_ctas} CTAs, workspace "
        f"{workspace.nbytes / 2**20:.1f} MiB{', shared with fmk_context_fmha' if shared else ''}; "
        f"{_fmk.build_note()})"
    )
    return len(chosen)


def context_library(attn) -> Optional[FmkContextPhaseFmha]:
    """The ``FmkContextPhaseFmha`` in ``attn``'s FMHA manager, or None."""
    for fmha in attn._fmha_manager.fmha_libs:
        if isinstance(fmha, FmkContextPhaseFmha):
            return fmha
    return None


__all__ = [
    "KERNEL_ID",
    "KERNEL_KWARGS",
    "kernel_kwargs",
    "FmkContextPhaseFmha",
    "FmkPureContextFmha",
    "StepPlans",
    "context_library",
    "install",
    "install_pure_context",
    "pure_context_library",
    "takes_pure_context",
]
