# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``fmk_context_fmha``: the context rows of mixed steps on the multi-instance paged context FMHA
(``ops/attention/context_fmha_fmk``), sm_103.

Opt-in: ``MUSESPARK_OPEN_FMK_CONTEXT_FMHA=1``. Lossless-numeric, not on under any default.

Where it acts. A mixed step (context rows and generation rows in one batch) of a full-attention
layer is served by the layer's ``FlashInferTrtllmGenFmha``: its ``run_context`` preprocesses the
context rows (RoPE, K / V appended to the paged cache, Q quantized to E4M3) and launches
flashinfer's trtllm-gen context FMHA, then ``run_generation`` attends the generation rows.
``install`` re-classes that library with ``FmkContextMixin``, whose ``run_context`` on a mixed
step with ``MIN_GENERATIONS``..``MAX_GENERATIONS`` generation requests runs:

1. ``thop.trtllm_gen_context_preprocess`` with the parameters the library's own context phase
   uses (without its FMHA workspace); it returns the layer's flat KV pool view and its block
   table. Its post-process counterpart acts on sparse KV only, so none runs.
2. ``plan_context_fmha`` on the step's host lengths (``prompt_lens_cpu_runtime``,
   ``kv_lens_runtime``): planned by the first full-attention layer of a step and reused by the
   others (their lengths are the same); one pinned non-blocking copy to the device.
3. ``paged_context_fmha_fmk`` into the context rows of the layer's output, with the softmax
   scale (``bmm1[1]``) and the output scale (``bmm2[0]``) read on the device.

The generation rows, pure context steps, decode steps and mixed steps outside the band run what
they ran before. No host synchronisation: every device value is read in stream order. A call
this path cannot serve runs the library's own context phase, logged once per reason: a stream
capture in progress, a mask other than causal, attention sinks, softmax statistics requested,
cross attention, M-RoPE, chunked attention, a sliding window shorter than the context, no paged
context FMHA, a cache block other than 32 or 64 tokens, a Q-only input, an output that is not
contiguous BF16, host lengths that disagree with the call, or a workspace smaller than the
preprocess needs.

Cache blocks: 32 or 64 tokens. The kernel reads a 64-token block as its two 32-token halves.

Numerics: the same precision as the trtllm-gen cubin (the same E4M3 Q / K / V and P, fp32
accumulation, BF16 O, the same scales) with another summation order and P rounding point, so not
bitwise.

Install after every other attention install (it only replaces the context phase of the FMHA
library) and before any CUDA graph is captured: it reserves the partial-result workspace
(about 38 MB at 152 SMs) and compiles the kernel builds and the merge on small synthetic calls.
"""

from __future__ import annotations

import functools
import os
import time
from typing import Optional, Sequence, Tuple

import torch
from torch.utils._python_dispatch import _disable_current_modes

from tensorrt_llm._torch.attention.backends.fmha import flashinfer_trtllm_gen as _fi_gen
from tensorrt_llm._torch.attention.backends.fmha.phased import FmhaParams
from tensorrt_llm._torch.attention.backends.fmha.utils import (
    get_attention_chunk_size,
    get_bmm1_scale,
)
from tensorrt_llm.bindings.internal import thop
from tensorrt_llm.functional import AttentionMaskType
from tensorrt_llm.logger import logger
from tensorrt_llm.quantization.mode import QuantMode

from ...ops.attention import context_fmha_fmk as _fmk

#: The knob that turns the path on (``1``).
ENV = "MUSESPARK_OPEN_FMK_CONTEXT_FMHA"
#: ``lo-hi``: the generation requests of the mixed steps this path takes (inclusive).
BAND_ENV = "MUSESPARK_OPEN_FMK_CONTEXT_FMHA_GENERATIONS"
DEFAULT_BAND = (1, 1024)
#: The SM version the kernel is built for here.
SERVED_SM = (10, 3)
#: Query instances per CTA and key tile of the build this path runs.
NUM_INST = 4
TILE_N = 64


def enabled() -> bool:
    return os.environ.get(ENV, "0") == "1"


def band() -> Tuple[int, int]:
    """The ``(min, max)`` generation requests of a mixed step this path takes."""
    raw = os.environ.get(BAND_ENV, "").strip()
    if not raw:
        return DEFAULT_BAND
    lo, _, hi = raw.partition("-")
    try:
        lo_n, hi_n = int(lo), int(hi)
    except ValueError:
        raise ValueError(f"{BAND_ENV} expects 'lo-hi' (two integers), got {raw!r}") from None
    if not 1 <= lo_n <= hi_n:
        raise ValueError(f"{BAND_ENV} expects 1 <= lo <= hi, got {raw!r}")
    return lo_n, hi_n


def takes_step(meta, lo: int, hi: int) -> bool:
    """Whether a step is a mixed step this path takes. Host-side reads only."""
    return (
        getattr(meta, "num_contexts", 0) > 0
        and lo <= getattr(meta, "num_generations", 0) <= hi
        and getattr(meta, "beam_width", 1) == 1
    )


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

    A step's full-attention layers run one after another with the same lengths, so the first
    plans and the rest reuse the plan's device buffer (read in stream order). A step with other
    lengths replaces it; the replaced buffer returns to the caching allocator on the same stream.
    """

    def __init__(self, num_ctas: int, num_heads: int, num_kv_heads: int):
        self.num_ctas = num_ctas
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
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
                num_heads=self.num_heads,
                num_kv_heads=self.num_kv_heads,
                num_inst=NUM_INST,
                tile_n=TILE_N,
                device=device,
            )
            self.plan_seconds += time.perf_counter() - start
            self.planned += 1
            self._key = key
        return self._plan


class FmkContextMixin:
    """Members ``install`` adds to a ``FlashInferTrtllmGenFmha`` (see ``_fmk_class``): the context
    phase of the mixed steps ``takes_step`` accepts runs the multi-instance context FMHA; every
    other call is the library's own. Adds plain attributes only (set by ``install``)."""

    _fmk_plans: Optional[StepPlans] = None
    _fmk_workspace: Optional[_fmk.FmkWorkspace] = None
    _fmk_band: Tuple[int, int] = DEFAULT_BAND
    #: Context phases run on the multi-instance kernel, and declined (the library's own).
    fmk_calls = 0
    fmk_declined = 0

    def run_context(self, params: FmhaParams) -> None:
        lo, hi = self._fmk_band
        if not takes_step(params.meta, lo, hi):
            self._fmk_shipped_run_context(params)
            return
        lengths = self._fmk_host_lengths(params)
        reason = self._fmk_decline(params, lengths)
        if reason is not None:
            self.fmk_declined += 1
            logger.info_once(
                f"[musespark_open] fmk_context_fmha: a mixed step's context rows ran the "
                f"trtllm-gen context FMHA (layer {self.attn.layer_idx}): {reason}",
                key=f"musespark_open.fmk_context_fmha.decline.{reason}",
            )
            self._fmk_shipped_run_context(params)
            return
        self._fmk_run(params, *lengths)
        self.fmk_calls += 1
        logger.info_once(
            f"[musespark_open] fmk_context_fmha: a mixed step's context rows "
            f"({params.batch_size} requests, {params.num_tokens} rows, layer "
            f"{self.attn.layer_idx}, {params.tokens_per_block}-token cache blocks) ran the "
            f"multi-instance context FMHA",
            key="musespark_open.fmk_context_fmha.ran",
        )

    @staticmethod
    def _fmk_host_lengths(params: FmhaParams) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
        meta, batch = params.meta, params.batch_size
        q_lens = tuple(meta.prompt_lens_cpu_runtime[:batch].tolist())
        kv_lens = tuple(meta.kv_lens_runtime[:batch].tolist())
        return q_lens, kv_lens

    def _fmk_decline(
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
        if params.tokens_per_block not in _fmk.CACHE_PAGES:
            return f"a cache block of {params.tokens_per_block} tokens"
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
        boxes_per_block = params.tokens_per_block // _fmk.PAGE
        if params.total_num_blocks * attn.num_kv_heads * boxes_per_block >= 2**31:
            return "a KV pool of more than 2^31 TMA boxes"
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

    def _fmk_run(self, params: FmhaParams, q_lens, kv_lens) -> None:
        attn, meta, fwd = params.attn, params.meta, params.fwd
        rope = attn.rope_params
        batch, num_tokens = params.batch_size, params.num_tokens
        heads, head_dim = attn.num_heads, attn.head_dim
        plans = self._fmk_plans
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
                multi_processor_count=plans.num_ctas,
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
        plan = plans.get(q_lens, kv_lens, q.device)
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
            workspace=self._fmk_workspace,
        )


@functools.lru_cache(maxsize=None)
def _fmk_class(base: type) -> type:
    """A subclass of ``base`` holding ``FmkContextMixin``'s members, one class per base.

    ``_fmk_shipped_run_context`` is the library's own ``run_context``. The class is not
    ``type(name, (FmkContextMixin, base), {})``: ``lib.__class__ = ...`` refuses it (the instance
    layout differs, as the ABC base of the library adds the instance dict below the mixin's).
    """
    members = {
        name: member
        for name, member in vars(FmkContextMixin).items()
        if name not in ("__dict__", "__weakref__", "__module__", "__doc__")
    }
    members["_fmk_shipped_run_context"] = base.run_context
    return type(f"{base.__name__}FmkContext", (base,), members)


def _refusal(attention) -> Optional[str]:
    attn = attention.attn
    if _fi_gen_library(attn) is None:
        return "no flashinfer trtllm-gen FMHA library in its FMHA manager"
    if getattr(attn, "is_mla_enable", False):
        return "MLA"
    if attn.head_dim != _fmk.HEAD_DIM:
        return f"head dim {attn.head_dim}"
    if attn.num_heads % attn.num_kv_heads or (attn.num_heads // attn.num_kv_heads) % NUM_INST:
        return f"{attn.num_heads} query heads over {attn.num_kv_heads} KV heads"
    if not QuantMode(attn.quant_mode).has_fp8_kv_cache():
        return "a KV cache that is not FP8"
    if not hasattr(thop, "trtllm_gen_context_preprocess"):
        return "the binding lacks trtllm_gen_context_preprocess"
    return None


def _fi_gen_library(attn) -> Optional[_fi_gen.FlashInferTrtllmGenFmha]:
    """The flashinfer trtllm-gen FMHA library in ``attn``'s FMHA manager, or None."""
    for fmha in getattr(getattr(attn, "_fmha_manager", None), "fmha_libs", ()):
        if isinstance(fmha, _fi_gen.FlashInferTrtllmGenFmha):
            return fmha
    return None


def _warm_up(
    device: torch.device,
    num_heads: int,
    num_kv_heads: int,
    num_ctas: int,
    cache_page: int,
    workspace,
):
    """Compile the kernel build of ``cache_page`` and its merge on a small synthetic call (one
    request of one row over 12,800 zero keys, which the planner cuts) and check that it ran."""
    keys = 12800
    blocks = keys // cache_page
    q = torch.zeros(1, num_heads, _fmk.HEAD_DIM, dtype=torch.float8_e4m3fn, device=device)
    pool = torch.zeros(
        2 * blocks,
        num_kv_heads,
        cache_page,
        _fmk.HEAD_DIM,
        dtype=torch.float8_e4m3fn,
        device=device,
    )
    ids = torch.arange(blocks, dtype=torch.int32, device=device)
    table = torch.stack([2 * ids, 2 * ids + 1]).unsqueeze(0).contiguous()
    cu_q = torch.tensor([0, 1], dtype=torch.int32, device=device)
    kv_lens = torch.tensor([keys], dtype=torch.int32, device=device)
    scales = torch.tensor([1.0, 1.0], dtype=torch.float32, device=device)
    out = torch.full(
        (1, num_heads, _fmk.HEAD_DIM), float("nan"), dtype=torch.bfloat16, device=device
    )
    plan = _fmk.plan_context_fmha(
        (1,),
        (keys,),
        num_ctas=num_ctas,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        num_inst=NUM_INST,
        tile_n=TILE_N,
        device=device,
    )
    _fmk.paged_context_fmha_fmk(
        q, pool, table, cu_q, kv_lens, plan, scales[0:1], scales[1:2], out, workspace=workspace
    )
    if plan.merge_meta is None or not bool(torch.isfinite(out).all()):
        raise ValueError(
            "[musespark_open] fmk_context_fmha: the warm-up call did not run as planned "
            f"(cache blocks of {cache_page}, merges "
            f"{0 if plan.merge_meta is None else plan.merge_meta.shape[0]})"
        )


def _prepare(
    device: torch.device, num_heads: int, num_kv_heads: int, num_ctas: int
) -> _fmk.FmkWorkspace:
    """The reserved partial-result workspace, after a warm-up call of every cache-block build.

    The loader builds the model under ``MetaInitMode``, which turns ``torch.empty`` into meta
    tensors (no storage). The workspace and the warm-up are device-side setup, not model
    construction, so the active dispatch modes are popped while they run.
    """
    with _disable_current_modes():
        workspace = _fmk.FmkWorkspace()
        workspace.reserve(_fmk.max_slots(num_ctas), NUM_INST, device)
        if workspace.part_o.is_meta or workspace.part_ml.is_meta:
            raise ValueError("[musespark_open] fmk_context_fmha: the workspace has no storage")
        for cache_page in _fmk.CACHE_PAGES:
            _warm_up(device, num_heads, num_kv_heads, num_ctas, cache_page, workspace)
    return workspace


def install(model: torch.nn.Module) -> int:
    """Re-class the ``FlashInferTrtllmGenFmha`` of every full-attention layer's FMHA manager with
    ``FmkContextMixin`` in front, and compile the kernel builds. Returns the number of layers
    changed.

    Run after every other attention install and before any CUDA graph is captured. Every layer
    is checked before any changes; idempotent.

    Raises:
        ValueError: if the device is not sm_103, or a full-attention layer has no trtllm-gen FMHA
            library or is not a layer the kernel serves (a plain FP8-KV causal layer of 64-wide
            heads, query heads over KV heads in multiples of 4), or the layers differ in rank
            shape.
    """
    chosen = [layer.self_attn for layer in model.layers if not layer.self_attn.is_sliding]
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != SERVED_SM:
        raise ValueError("[musespark_open] the fmk_context_fmha kernel is built for sm_103 only")
    refused = [
        f"layer {attention.layer_idx}: {reason}"
        for attention in chosen
        if (reason := _refusal(attention)) is not None
    ]
    shapes = {(a.attn.num_heads, a.attn.num_kv_heads) for a in chosen}
    if len(shapes) > 1:
        refused.append(f"the layers differ in query / KV heads per rank: {sorted(shapes)}")
    if refused:
        raise ValueError(
            "[musespark_open] fmk_context_fmha cannot take the mixed steps' context rows: "
            + "; ".join(refused)
        )
    lo, hi = band()
    device = torch.device("cuda", torch.cuda.current_device())
    num_ctas = torch.cuda.get_device_properties(device).multi_processor_count
    heads, kv_heads = next(iter(shapes))
    workspace = _prepare(device, heads, kv_heads, num_ctas)
    plans = StepPlans(num_ctas, heads, kv_heads)
    for attention in chosen:
        lib = _fi_gen_library(attention.attn)
        if not hasattr(lib, "_fmk_shipped_run_context"):
            lib.__class__ = _fmk_class(type(lib))
        lib._fmk_plans = plans
        lib._fmk_workspace = workspace
        lib._fmk_band = (lo, hi)
    logger.info(
        f"[musespark_open] fmk_context_fmha: the context rows of mixed steps with {lo}..{hi} "
        f"generation requests run the multi-instance context FMHA on {len(chosen)} full-attention "
        f"layers ({heads} query / {kv_heads} KV heads per rank, {num_ctas} CTAs, cache blocks of "
        f"{list(_fmk.CACHE_PAGES)} tokens, workspace {workspace.nbytes / 2**20:.1f} MiB)"
    )
    return len(chosen)


__all__ = [
    "BAND_ENV",
    "ENV",
    "FmkContextMixin",
    "StepPlans",
    "band",
    "enabled",
    "install",
    "takes_step",
]
