# Adapted from https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py
# Adapted from https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/configs/qwen3_next.py
# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import functools
import os
import weakref
from types import SimpleNamespace
from typing import NamedTuple, Optional, Tuple

import torch
import triton
import triton.language as tl
from torch import nn
from transformers import Qwen3NextConfig

from tensorrt_llm._torch.modules.fla.cached_replay import (
    CACHED_REPLAY_PARTITION_MIN_BATCH_SIZE,
    fused_recurrent_gated_delta_rule_cached_replay_update,
)
from tensorrt_llm._torch.modules.fla.fused_recurrent import fused_recurrent_gated_delta_rule_update
from tensorrt_llm._torch.modules.fla.flashinfer_chunk import chunk_gated_delta_rule_indexed_direct
from tensorrt_llm._torch.modules.fla.fused_sigmoid_gating_recurrent import (
    _can_use_flashinfer_gdn_decode,
    _can_use_flashinfer_gdn_verify,
    _flashinfer_gdn_verify,
    can_use_gdn_decode_pdl_pair,
    flashinfer_gdn_bf16_state_available,
    flashinfer_gdn_decode_direct_available,
    flashinfer_gdn_decode_t1,
    flashinfer_gdn_tail_recurrent,
    fused_sigmoid_gating_delta_rule_update,
    gdn_decode_pdl_update,
)
from tensorrt_llm._utils import is_flashinfer_gdn_supported_arch, is_sm_100f
from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping

from ...attention.backends import AttentionMetadata
from ...distributed import AllReduceParams
from ...model_config import ModelConfig
from ...pyexecutor.breakable_cuda_graph import eager_on_graph, is_in_breakable_cuda_graph
from ...speculative import SpecMetadata
from ...utils import EventType, get_model_extra_attrs, is_gdn_replay_enabled, is_torch_compiling
from ..linear import FP8QDQLinearMethod, Linear, TensorParallelMode, UnquantizedLinearMethod
from ..multi_stream_utils import maybe_execute_in_parallel
from .causal_conv1d import causal_conv1d_fn, causal_conv1d_update
from .causal_conv1d_triton import causal_conv1d_update as causal_conv1d_update_triton
from .fuse_elementwise_ops import (
    extract_transpose_prefill_slice,
    fused_gdn_post_conv,
    pack_gdn_decode_qkv,
)
from .layernorm_gated import RMSNorm as RMSNormGated
from .layernorm_gated import rms_norm_gated_token_major
from .mamba2_metadata import Mamba2Metadata
from .mamba2_mixer import _cached_arange
from .recurrent_state_cache import reset_recurrent_state_rows

# One GEMM for both GDN input projections ([Q|K|V|Z] and [b|a]) instead of two
# column-parallel Linears run side by side: at decode the pair is two cuBLAS
# split-K launches on two streams that slow each other down and a cross-stream
# join before the GDN kernel; the fused [qkvz | ba] weight makes it one launch.
# Only taken for plain bf16 (unquantized, no bias) projections; TLLM_GDN_FUSED_IN_PROJ=0
# restores the two-Linear path.
_GDN_FUSED_IN_PROJ = os.environ.get("TLLM_GDN_FUSED_IN_PROJ", "1") != "0"


# FlashInfer GDN prefill is ON by default; set TLLM_USE_FLASHINFER_GDN_PREFILL=0
# to force the vendored Triton chunk_gated_delta_rule everywhere. FlashInfer only
# ships the GDN prefill kernel for Hopper (SM90) and datacenter Blackwell
# (SM100/SM103); on consumer Blackwell (SM120) and other archs it aborts at
# launch, so we fall back to Triton there. Resolution is deferred to first call
# (and cached) so importing this module never initializes CUDA.
def _use_flashinfer_gdn_prefill() -> bool:
    """Check the prefill backend setting and supported GPU architecture."""
    return (
        os.getenv("TLLM_USE_FLASHINFER_GDN_PREFILL", "1") == "1"
        and is_flashinfer_gdn_supported_arch()
    )


@functools.lru_cache(maxsize=1)
def _resolve_chunk_gated_delta_rule():
    """Resolve and cache the selected prefill implementation."""
    if _use_flashinfer_gdn_prefill():
        from tensorrt_llm._torch.modules.fla.flashinfer_chunk import chunk_gated_delta_rule as impl
    else:
        from tensorrt_llm._torch.modules.fla.chunk import chunk_gated_delta_rule as impl
    return impl


@torch.compiler.disable
def chunk_gated_delta_rule(
    *args, state_workspace: Optional[tuple[torch.Tensor, torch.Tensor]] = None, **kwargs
):
    """Dispatch prefill, omitting unused FlashInfer-only workspace arguments."""
    if state_workspace is not None:
        kwargs["state_workspace"] = state_workspace
    return _resolve_chunk_gated_delta_rule()(*args, **kwargs)


@functools.lru_cache(maxsize=1)
def _gdn_prefill_uses_flashinfer() -> bool:
    """Whether ``chunk_gated_delta_rule`` resolves to the FlashInfer adapter.

    Decides the gate convention handed to it: the FlashInfer kernel consumes
    ``alpha = exp(g)``, which ``fused_gdn_post_conv(g_linear=True)`` emits
    directly (no torch.exp launch per scan); the Triton kernels want log space.
    """
    return _resolve_chunk_gated_delta_rule().__module__.rsplit(".", 1)[-1] == "flashinfer_chunk"


# Packed prefill tokens per iteration up to which the main scan pins FlashInfer's
# plain chunked kernel (``use_cp=False``): one launch with indexed pool I/O
# instead of the chunk-parallel kernel's four launches plus gather and scatter.
# Measured on VR200 / flashinfer 0.6.18 (one TP8 rank of Qwen3.5-397B): the
# plain kernel is also faster on the GPU up to 4096 tokens and ~35 us slower at
# 8192 (results/_analysis_scratch/gdn_prefill_cp_vs_plain_bench.py). Above the
# threshold the adapter's default ("auto") leaves the choice to FlashInfer.
_GDN_PREFILL_CP_MIN_TOKENS = int(os.environ.get("TLLM_GDN_PREFILL_CP_MIN_TOKENS", "4096"))


def _gdn_prefill_use_cp(num_prefill_tokens: int):
    """``use_cp`` for the main prefill scan of an iteration with this many packed tokens."""
    return "auto" if num_prefill_tokens > _GDN_PREFILL_CP_MIN_TOKENS else False


# Mixed iterations (context chunks + decode requests in one batch): run the
# decode requests' tokens through the recurrent decode kernel, as pure-decode
# iterations do, instead of scanning them as one-token sequences in the chunked
# prefill kernel. The scan pays 70-110 us per layer for the ~190 one-token
# sequences a typical mixed iteration carries (gdn_prefill_mixedbatch_bench.py:
# one 512-token chunk 69 us alone, 143-175 us with 192-256 decode sequences),
# the decode kernel ~10 us. TLLM_GDN_MIXED_DECODE_RECURRENT=0 restores the
# single scan for A/B.
_MIXED_DECODE_RECURRENT = os.environ.get("TLLM_GDN_MIXED_DECODE_RECURRENT", "1") == "1"


def _mixed_scan_prefill_layout(
    num_prefill: int, fold, scan_state_indices: torch.Tensor, scan_cu_seqlens: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """The prefill-only prefix of a mixed scan layout.

    Both the plain layout (``query_start_loc`` / ``cache_indices``) and the folded
    one (``build_fold_segments``) list the context requests' segments first and
    the decode requests, one token each, after them; a folded context request
    contributes two segments (A -> S1, B -> S2).
    """
    n_seg = num_prefill + (fold.fold_count if fold is not None else 0)
    return scan_state_indices[:n_seg], scan_cu_seqlens[: n_seg + 1]


def _extract_gdn_extra_attrs(layer_idx: str):
    extra_attrs = get_model_extra_attrs()
    assert extra_attrs is not None, "Model extra attrs is not set"

    metadata_ref = extra_attrs.get("attention_metadata", None)
    assert metadata_ref is not None, "Attention metadata is not set"
    metadata = metadata_ref()
    assert isinstance(metadata, AttentionMetadata)

    gdn_layers = extra_attrs.get("gdn_layers", None)
    assert gdn_layers is not None, "GDN layer is not registered"
    gdn_layer_ref = gdn_layers.get(layer_idx, None)
    assert gdn_layer_ref is not None, f"Cannot find GDN layer for layer {layer_idx}"
    gdn_layer = gdn_layer_ref()
    assert isinstance(gdn_layer, Qwen3NextGatedDeltaNet)

    return metadata, gdn_layer, extra_attrs.get("spec_metadata", None)


@triton.jit
def _reset_gdn_states_all_layers_kernel(
    ssm_states,
    conv_states,
    state_indices,
    has_initial_states,
    ssm_layer_stride,
    ssm_slot_stride,
    conv_layer_stride,
    conv_slot_stride,
    NUM_CACHE_LINES: tl.constexpr,
    SSM_STATE_SIZE: tl.constexpr,
    CONV_STATE_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """One launch clears the slots of every local GDN layer: grid (request, layer, block)."""
    request_idx = tl.program_id(0)
    layer = tl.program_id(1).to(tl.int64)
    offsets = tl.program_id(2) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    state_idx = tl.load(state_indices + request_idx).to(tl.int64)
    needs_reset = ~tl.load(has_initial_states + request_idx).to(tl.int1)
    valid_state = (state_idx >= 0) & (state_idx < NUM_CACHE_LINES)
    ssm_base = layer * ssm_layer_stride.to(tl.int64) + state_idx * ssm_slot_stride.to(tl.int64)
    conv_base = layer * conv_layer_stride.to(tl.int64) + state_idx * conv_slot_stride.to(tl.int64)
    tl.store(
        ssm_states + ssm_base + offsets,
        0.0,
        mask=needs_reset & valid_state & (offsets < SSM_STATE_SIZE),
    )
    tl.store(
        conv_states + conv_base + offsets,
        0.0,
        mask=needs_reset & valid_state & (offsets < CONV_STATE_SIZE),
    )


def _all_layer_state_pools(kv_cache_manager) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """``(ssm [layers, slots, ...], conv [layers, slots, ...])`` spanning every local GDN layer.

    The C++ hybrid manager keeps both as regular-stride views of one pool
    (``all_ssm_states`` / ``all_conv_states``); the Python manager keeps them as
    the ``temporal`` / ``conv`` tensors of its state container. ``None`` when the
    manager exposes neither, in which case the caller falls back to the
    per-layer reset.
    """
    ssm = getattr(kv_cache_manager, "all_ssm_states", None)
    conv = getattr(kv_cache_manager, "all_conv_states", None)
    if isinstance(ssm, torch.Tensor) and isinstance(conv, torch.Tensor):
        return ssm, conv
    impl = getattr(kv_cache_manager, "_impl", None)
    if impl is not None and impl is not kv_cache_manager:
        pools = _all_layer_state_pools(impl)
        if pools is not None:
            return pools
    cache = getattr(kv_cache_manager, "mamba_cache", None)
    ssm = getattr(cache, "temporal", None)
    conv = getattr(cache, "conv", None)
    if (
        isinstance(ssm, torch.Tensor)
        and isinstance(conv, torch.Tensor)
        and ssm.dim() == 5
        and conv.dim() == 4
        and ssm.shape[0] == conv.shape[0]
    ):
        return ssm, conv
    return None


def _reset_gdn_states_all_layers(
    ssm_states: torch.Tensor,
    conv_states: torch.Tensor,
    state_indices: torch.Tensor,
    has_initial_states: torch.Tensor,
) -> None:
    """Zero the slots of ``state_indices`` whose request has no initial state, in every layer.

    ``ssm_states`` is ``[layers, slots, heads, head_dim, d_state]`` and
    ``conv_states`` ``[layers, slots, conv_dim, d_conv - 1]``; both may be
    strided along the layer and slot dims (pool views) but each slot's state
    must be contiguous.
    """
    num_requests = state_indices.shape[0]
    if num_requests == 0:
        return
    num_layers = ssm_states.shape[0]
    assert conv_states.shape[0] == num_layers, (ssm_states.shape, conv_states.shape)
    assert ssm_states.shape[1] == conv_states.shape[1], (ssm_states.shape, conv_states.shape)
    assert ssm_states[0, 0].is_contiguous() and conv_states[0, 0].is_contiguous()
    ssm_state_size = ssm_states[0, 0].numel()
    conv_state_size = conv_states[0, 0].numel()
    block_size = 1024
    grid = (
        num_requests,
        num_layers,
        triton.cdiv(max(ssm_state_size, conv_state_size), block_size),
    )
    _reset_gdn_states_all_layers_kernel[grid](
        ssm_states,
        conv_states,
        state_indices,
        has_initial_states,
        ssm_states.stride(0),
        ssm_states.stride(1),
        conv_states.stride(0),
        conv_states.stride(1),
        ssm_states.shape[1],
        ssm_state_size,
        conv_state_size,
        block_size,
    )


# Fused bookkeeping of the folded save-last prefill (TLLM_MAMBA_FOLD_FUSED=0
# restores the per-layer torch path). The fold used to cost ~24 small host
# launches per GDN layer and iteration (arange/sub/add for the conv-tail
# index, two .long() casts, index_select + 2x index_copy on the conv state,
# five row gathers + an index_copy around the tail scan); on the aggregated
# engine, whose mixed iterations are host-launch bound, that was ~20 ms per
# mixed iteration (2026-09-09 fold on/off A/B). Here the per-iteration
# constants come pre-built from Mamba2Metadata, the conv-state move is one
# Triton launch, and a single folded chunk slices instead of gathering.
class _PackedPostConv(tuple):
    """``(q3, k3, v3, g2, beta2)``: the ``[T, H, D]`` / ``[T, HV]`` views FlashInfer's
    launchers take of the post-conv scratch buffers (same storage as the
    ``[1, T, ...]`` views ``fused_gdn_post_conv`` fills), plus the row slices of
    the folded tails cut from them. The folds of an iteration are the same for
    all of its layers, so each set of five slices is built once per iteration
    instead of once per layer (chunk_gated_delta_rule_indexed_direct)."""

    def __new__(cls, views):
        self = super().__new__(cls, views)
        self.slices = {}
        return self

    def tail(self, start: int, n: int):
        key = (start, n)
        views = self.slices.get(key)
        if views is None:
            if len(self.slices) >= 512:
                self.slices.clear()
            views = tuple(t[start : start + n] for t in self)
            self.slices[key] = views
        return views


class _GdnPrefillScratch:
    """Buffers of the eager GDN prefill path, shared by the layers of one iteration.

    Every GDN layer of a mixed / prefill iteration allocated its transposed conv
    input and the five post-conv outputs (q, k, v, g, beta) per call, six
    allocator round trips (~25 us of host time per layer, ~1 ms per iteration
    on the 45-layer engine). The layers run back to back on one stream and a
    layer's buffers are dead once its scans and its decode update have been
    issued, so one set per device, grown on demand, serves all of them. Flat
    storage viewed as a prefix keeps every view contiguous.
    """

    def __init__(self) -> None:
        self._bufs: dict = {}
        # Views handed out for the last shapes: the 45 layers of one iteration
        # ask for the same shapes, so each set of views is built once per
        # iteration instead of once per layer. Cleared whenever a buffer grows.
        self._views: dict = {}

    def _flat(self, key, numel: int, dtype: torch.dtype, device) -> torch.Tensor:
        buf = self._bufs.get(key)
        if buf is None or buf.numel() < numel:
            buf = torch.empty(numel, dtype=dtype, device=device)
            self._bufs[key] = buf
            self._views.clear()
        return buf

    def conv_input(self, conv_dim: int, num_tokens: int, dtype: torch.dtype, device) -> torch.Tensor:
        """``[conv_dim, num_tokens]`` for extract_transpose_prefill_slice / the in-place conv."""
        key = ("conv", conv_dim, num_tokens, dtype, device)
        view = self._views.get(key)
        if view is None:
            buf = self._flat(("conv", device, dtype), conv_dim * num_tokens, dtype, device)
            view = buf[: conv_dim * num_tokens].view(conv_dim, num_tokens)
            self._views[key] = view
        return view

    def conv_out(self, conv_dim: int, num_tokens: int, dtype: torch.dtype, device) -> torch.Tensor:
        """``[num_tokens, conv_dim]`` token-major output of the channel-last prefill conv."""
        key = ("convout", conv_dim, num_tokens, dtype, device)
        view = self._views.get(key)
        if view is None:
            buf = self._flat(("convout", device, dtype), conv_dim * num_tokens, dtype, device)
            view = buf[: conv_dim * num_tokens].view(num_tokens, conv_dim)
            self._views[key] = view
        return view

    def conv_tail(self, num_folds: int, conv_dim: int, width: int, dtype: torch.dtype, device) -> torch.Tensor:
        """``[num_folds, conv_dim, width]``: the fold-point conv states written by the extract launch."""
        key = ("tail", num_folds, conv_dim, width, dtype, device)
        view = self._views.get(key)
        if view is None:
            n = num_folds * conv_dim * width
            buf = self._flat(("tail", device, dtype), n, dtype, device)
            view = buf[:n].view(num_folds, conv_dim, width)
            self._views[key] = view
        return view

    def tail_rows(self, num_rows: int, q: torch.Tensor, v: torch.Tensor, g: torch.Tensor, beta: torch.Tensor):
        """Gathered tail rows ``(q, k, v, g, beta, out)`` for ``num_rows`` packed tokens (fold_scan_tails)."""
        dev = q.device
        hq, dk = q.shape[2], q.shape[3]
        hv, dv = v.shape[2], v.shape[3]
        key = ("rows", num_rows, hq, dk, hv, dv, q.dtype, v.dtype, g.dtype, beta.dtype, dev)
        views = self._views.get(key)
        if views is None:
            nq, nv, ng = num_rows * hq * dk, num_rows * hv * dv, num_rows * hv
            qb = self._flat(("tq", dev, q.dtype), nq, q.dtype, dev)[:nq].view(1, num_rows, hq, dk)
            kb = self._flat(("tk", dev, q.dtype), nq, q.dtype, dev)[:nq].view(1, num_rows, hq, dk)
            vb = self._flat(("tv", dev, v.dtype), nv, v.dtype, dev)[:nv].view(1, num_rows, hv, dv)
            gb = self._flat(("tg", dev, g.dtype), ng, g.dtype, dev)[:ng].view(1, num_rows, hv)
            bb = self._flat(("tb", dev, beta.dtype), ng, beta.dtype, dev)[:ng].view(1, num_rows, hv)
            ob = self._flat(("to", dev, v.dtype), nv, v.dtype, dev)[:nv].view(1, num_rows, hv, dv)
            views = (qb, kb, vb, gb, bb, ob)
            self._views[key] = views
            self._views[("rows3",) + key[1:]] = tuple(t.squeeze(0) for t in views)
        return views

    def tail_rows_packed(self, num_rows: int, q: torch.Tensor, v: torch.Tensor, g: torch.Tensor, beta: torch.Tensor):
        """The packed ``[rows, ...]`` views of the buffers ``tail_rows`` returned for the same arguments (call it first)."""
        hq, dk = q.shape[2], q.shape[3]
        hv, dv = v.shape[2], v.shape[3]
        return self._views.get(("rows3", num_rows, hq, dk, hv, dv, q.dtype, v.dtype, g.dtype, beta.dtype, q.device))

    def post_conv(
        self,
        num_tokens: int,
        num_k_heads: int,
        head_k_dim: int,
        num_v_heads: int,
        head_v_dim: int,
        dtype: torch.dtype,
        device,
        beta_dtype: torch.dtype = torch.float32,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """``(q, k, v, g, beta)`` of fused_gdn_post_conv's shapes for ``num_tokens``."""
        key = ("post", num_tokens, num_k_heads, head_k_dim, num_v_heads, head_v_dim, dtype, beta_dtype, device)
        views = self._views.get(key)
        if views is None:
            nk = num_tokens * num_k_heads * head_k_dim
            nv = num_tokens * num_v_heads * head_v_dim
            ng = num_tokens * num_v_heads
            q = self._flat(("q", device, dtype), nk, dtype, device)[:nk].view(1, num_tokens, num_k_heads, head_k_dim)
            k = self._flat(("k", device, dtype), nk, dtype, device)[:nk].view(1, num_tokens, num_k_heads, head_k_dim)
            v = self._flat(("v", device, dtype), nv, dtype, device)[:nv].view(1, num_tokens, num_v_heads, head_v_dim)
            g = self._flat(("g", device), ng, torch.float32, device)[:ng].view(1, num_tokens, num_v_heads)
            beta = self._flat(("beta", device, beta_dtype), ng, beta_dtype, device)[:ng].view(1, num_tokens, num_v_heads)
            views = (q, k, v, g, beta)
            self._views[key] = views
            self._views[("packed",) + key[1:]] = _PackedPostConv(t.squeeze(0) for t in views)
        return views

    def post_conv_packed(
        self,
        num_tokens: int,
        num_k_heads: int,
        head_k_dim: int,
        num_v_heads: int,
        head_v_dim: int,
        dtype: torch.dtype,
        device,
        beta_dtype: torch.dtype = torch.float32,
    ) -> Optional[_PackedPostConv]:
        """The ``_PackedPostConv`` views of the buffers ``post_conv`` returned for the same arguments (call it first)."""
        return self._views.get(
            ("packed", num_tokens, num_k_heads, head_k_dim, num_v_heads, head_v_dim, dtype, beta_dtype, device)
        )


_GDN_PREFILL_SCRATCH = os.environ.get("TLLM_GDN_PREFILL_SCRATCH", "1") == "1"
_gdn_prefill_scratch = _GdnPrefillScratch()
# Run the prefill conv on the channel-last (token-major) projection slice and let
# it write a token-major result: no extract-transpose launch, no transposed
# copy, and the fold-point conv tail is read from the untouched input by the
# seed launch instead of being staged. TLLM_GDN_CONV_CHANNEL_LAST=0 restores the
# channel-major kernel with the transpose in front of it.
_GDN_CONV_CHANNEL_LAST = os.environ.get("TLLM_GDN_CONV_CHANNEL_LAST", "1") == "1"
# TLLM_PWCG_NVTX_PROBE=1 brackets the eager GDN op body in an NVTX range (host attribution).
_GDN_NVTX_PROBE = os.environ.get("TLLM_PWCG_NVTX_PROBE", "0") == "1"


class _ConvTailFromInput(NamedTuple):
    """The fold-point conv tails as ``d_conv - 1`` token rows of the token-major
    pre-conv input ``x`` [tokens, conv_dim] before each fold point ``conv_tok``."""

    x: torch.Tensor
    conv_tok: torch.Tensor


class _FoldTailRecurrentInputs(NamedTuple):
    """What the recurrent fold-tail path reads per tail row range: the token-major
    raw conv output ``conv_out`` [tokens, conv_dim] and the raw gate columns
    ``a`` / ``b`` [tokens, HV] of the prefill rows, plus the layer's gate
    parameters and head geometry ``(H, DK, HV, DV)``."""

    conv_out: torch.Tensor
    a: torch.Tensor
    b: torch.Tensor
    A_log: torch.Tensor
    dt_bias: torch.Tensor
    heads: Tuple[int, int, int, int]


# Run each folded save-last tail (<= tokens_per_block tokens from the snapshot
# slot S1 into the terminal slot S2) through FlashInfer's recurrent kernel with
# split-pool state I/O instead of a second chunked scan seeded by an S2 <- S1
# copy: 12-14 us of GPU time instead of ~35 per folded layer (~1 ms per mixed
# iteration of the 45-layer engine at one fold per layer), terminal state within
# one bf16 ulp of the two-chunk schedule. The conv-state commit of the folds
# rides in the post-conv launch (fused_gdn_post_conv fold_conv), so the seed
# launch disappears too: 136 -> 124 us of host per folded layer on one TP8
# rank (gdn_forward_core_host_bench.py). Up to this many folds per iteration
# take one recurrent launch each; more fall back to the gathered chunked scan.
# TLLM_GDN_FOLD_TAIL_RECURRENT=0 restores the chunked tail.
_FOLD_TAIL_RECURRENT = os.environ.get("TLLM_GDN_FOLD_TAIL_RECURRENT", "1") == "1"
_FOLD_TAIL_RECURRENT_MAX_FOLDS = int(os.environ.get("TLLM_GDN_FOLD_TAIL_RECURRENT_MAX_FOLDS", "4"))
# Folded tails scan one launch per fold (slices, no gathers) up to this many
# folds; above it the tails are gathered into one scan. Each FlashInfer launch
# costs ~35-40 us of host and ~10 us of GPU time whatever its size, so the
# per-fold form loses to gather + one scan + index_copy beyond a couple of
# folds (measured: k folds cost 39k us host / 11k us GPU per layer against
# ~80 / ~20 for the gathered form).
_FOLD_TAIL_SLICE_MAX_FOLDS = int(os.environ.get("TLLM_GDN_FOLD_TAIL_SLICE_MAX_FOLDS", "1"))


_FOLD_FUSED_BOOKKEEPING = os.environ.get("TLLM_MAMBA_FOLD_FUSED", "1") == "1"


@triton.jit
def _fold_commit_conv_kernel(
    conv_states,
    tail,
    s1,
    s2,
    conv_state_stride,
    tail_stride_r,
    tail_stride_c,
    tail_stride_w,
    WIDTH: tl.constexpr,
    STATE_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # One program per (folded request, block of the flat [conv_dim, WIDTH]
    # state): conv_states[S2] <- conv_states[S1] (chunk-end state written by
    # the conv kernel), then conv_states[S1] <- tail (fold-point state). Every
    # element is read from S1 before it is overwritten by the same program.
    r = tl.program_id(0)
    offs = tl.program_id(1) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < STATE_SIZE
    slot1 = tl.load(s1 + r).to(tl.int64)
    slot2 = tl.load(s2 + r).to(tl.int64)
    src = tl.load(conv_states + slot1 * conv_state_stride.to(tl.int64) + offs, mask=mask)
    tl.store(conv_states + slot2 * conv_state_stride.to(tl.int64) + offs, src, mask=mask)
    c = offs // WIDTH
    w = offs - c * WIDTH
    t = tl.load(
        tail + r * tail_stride_r.to(tl.int64) + c * tail_stride_c + w * tail_stride_w,
        mask=mask,
    )
    tl.store(
        conv_states + slot1 * conv_state_stride.to(tl.int64) + offs,
        t.to(src.dtype),
        mask=mask,
    )


@triton.jit
def _gather_tail_rows_kernel(
    rows_ptr,
    q_ptr,
    k_ptr,
    v_ptr,
    g_ptr,
    b_ptr,
    qo_ptr,
    ko_ptr,
    vo_ptr,
    go_ptr,
    bo_ptr,
    q_stride,
    k_stride,
    v_stride,
    g_stride,
    b_stride,
    QK_ROW: tl.constexpr,
    V_ROW: tl.constexpr,
    G_ROW: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # One program per gathered token row: copies its q, k, v, g and beta rows
    # (token-major layouts, contiguous within a token) into the packed outputs.
    i = tl.program_id(0)
    src = tl.load(rows_ptr + i).to(tl.int64)
    offs = tl.arange(0, BLOCK)
    for start in range(0, QK_ROW, BLOCK):
        o = start + offs
        m = o < QK_ROW
        tl.store(qo_ptr + i * QK_ROW + o, tl.load(q_ptr + src * q_stride + o, mask=m), mask=m)
        tl.store(ko_ptr + i * QK_ROW + o, tl.load(k_ptr + src * k_stride + o, mask=m), mask=m)
    for start in range(0, V_ROW, BLOCK):
        o = start + offs
        m = o < V_ROW
        tl.store(vo_ptr + i * V_ROW + o, tl.load(v_ptr + src * v_stride + o, mask=m), mask=m)
    og = tl.arange(0, BLOCK)
    mg = og < G_ROW
    tl.store(go_ptr + i * G_ROW + og, tl.load(g_ptr + src * g_stride + og, mask=mg), mask=mg)
    tl.store(bo_ptr + i * G_ROW + og, tl.load(b_ptr + src * b_stride + og, mask=mg), mask=mg)


def gather_tail_rows(q, k, v, g, beta, rows: torch.Tensor, out) -> None:
    """``out = (q, k, v, g, beta)`` buffers of ``[1, len(rows), ...]`` receive the
    packed token rows ``rows`` of the five ``[1, T, ...]`` tensors in one launch
    (the torch path spent one index_select per tensor)."""
    qo, ko, vo, go, bo = out
    n = rows.shape[0]
    if n == 0:
        return
    hq, dk = q.shape[2], q.shape[3]
    hv, dv = v.shape[2], v.shape[3]
    for t in (q, k, v, g, beta):
        assert t.dim() >= 3 and t.shape[0] == 1 and t.stride(-1) == 1
    assert q.stride(2) == dk and k.stride(2) == dk and v.stride(2) == dv, "heads contiguous within a token"
    assert all(t.is_contiguous() for t in out)
    _gather_tail_rows_kernel[(n,)](
        rows,
        q,
        k,
        v,
        g,
        beta,
        qo,
        ko,
        vo,
        go,
        bo,
        q.stride(1),
        k.stride(1),
        v.stride(1),
        g.stride(1),
        beta.stride(1),
        QK_ROW=hq * dk,
        V_ROW=hv * dv,
        G_ROW=hv,
        BLOCK=1024,
    )


@triton.jit
def _fold_seed_terminal_slots_kernel(
    pool,
    conv_states,
    tail,
    s1,
    s2,
    pool_stride_n,
    pool_stride_h,
    pool_stride_v,
    pool_stride_k,
    conv_state_stride,
    tail_stride_r,
    tail_stride_c,
    tail_stride_w,
    conv_tok,
    H: tl.constexpr,
    V: tl.constexpr,
    K: tl.constexpr,
    BLOCK_V: tl.constexpr,
    BLOCK_K: tl.constexpr,
    HAS_CONV: tl.constexpr,
    TAIL_FROM_INPUT: tl.constexpr,
    WIDTH: tl.constexpr,
    CONV_STATE_SIZE: tl.constexpr,
    CONV_BLOCK: tl.constexpr,
    HAS_SSM: tl.constexpr = True,
):
    # Programs (fold r, j): j < H * VK_BLOCKS copy one [BLOCK_V, BLOCK_K] tile of
    # the recurrent state pool[S1] -> pool[S2]; the following programs move the
    # conv state conv_states[S2] <- conv_states[S1] and conv_states[S1] <- tail
    # (every element read from S1 before the same program overwrites it). The
    # tail is either a staged [fold, conv_dim, WIDTH] buffer or, with
    # TAIL_FROM_INPUT, the WIDTH token rows before conv_tok[r] of the token-major
    # pre-conv input (tail_stride_r / _w = its token stride, _c = 1).
    r = tl.program_id(0)
    j = tl.program_id(1)
    slot1 = tl.load(s1 + r).to(tl.int64)
    slot2 = tl.load(s2 + r).to(tl.int64)
    num_k_blocks: tl.constexpr = (K + BLOCK_K - 1) // BLOCK_K
    num_v_blocks: tl.constexpr = (V + BLOCK_V - 1) // BLOCK_V
    ssm_programs: tl.constexpr = (H * num_v_blocks * num_k_blocks) if HAS_SSM else 0
    if j < ssm_programs:
        h = j // (num_v_blocks * num_k_blocks)
        vk = j % (num_v_blocks * num_k_blocks)
        v_offs = (vk // num_k_blocks) * BLOCK_V + tl.arange(0, BLOCK_V)
        k_offs = (vk % num_k_blocks) * BLOCK_K + tl.arange(0, BLOCK_K)
        tile_mask = (v_offs < V)[:, None] & (k_offs < K)[None, :]
        tile_offs = h * pool_stride_h + v_offs[:, None] * pool_stride_v + k_offs[None, :] * pool_stride_k
        data = tl.load(pool + slot1 * pool_stride_n + tile_offs, mask=tile_mask, other=0.0)
        tl.store(pool + slot2 * pool_stride_n + tile_offs, data, mask=tile_mask)
    elif HAS_CONV:
        coffs = (j - ssm_programs) * CONV_BLOCK + tl.arange(0, CONV_BLOCK)
        cmask = coffs < CONV_STATE_SIZE
        src = tl.load(conv_states + slot1 * conv_state_stride.to(tl.int64) + coffs, mask=cmask)
        tl.store(conv_states + slot2 * conv_state_stride.to(tl.int64) + coffs, src, mask=cmask)
        c = coffs // WIDTH
        w = coffs - c * WIDTH
        if TAIL_FROM_INPUT:
            base = (tl.load(conv_tok + r) - WIDTH).to(tl.int64) * tail_stride_r
        else:
            base = r * tail_stride_r.to(tl.int64)
        t = tl.load(tail + base + c * tail_stride_c + w * tail_stride_w, mask=cmask)
        tl.store(
            conv_states + slot1 * conv_state_stride.to(tl.int64) + coffs,
            t.to(src.dtype),
            mask=cmask,
        )


def fold_seed_terminal_slots(
    ssm_states: torch.Tensor,
    fold,
    conv_states: Optional[torch.Tensor] = None,
    tail: Optional[torch.Tensor] = None,
    copy_ssm: bool = True,
) -> None:
    """One launch per layer for every folded request: pool[S2] <- pool[S1] (the
    snapshot seeds the terminal slot the tail scan then updates in place) and,
    with ``conv_states`` / ``tail``, the conv-state commit of
    fold_commit_conv_states (S2 <- chunk-end state, S1 <- fold-point state).
    ``copy_ssm=False`` runs the conv-state commit alone (the recurrent tail
    reads S1 and writes S2 itself)."""
    n = fold.fold_count
    if n == 0:
        return
    has_conv = conv_states is not None
    if not copy_ssm and not has_conv:
        return
    tail_from_input = isinstance(tail, _ConvTailFromInput)
    conv_tok = fold.fold_s1  # placeholder pointer when unused
    if has_conv:
        assert tail is not None and conv_states.dim() == 3
        assert conv_states.stride(2) == 1 and conv_states.stride(1) == conv_states.shape[2]
        width = conv_states.shape[2]
        conv_state_size = conv_states.shape[1] * width
        conv_block = 1024
        conv_programs = triton.cdiv(conv_state_size, conv_block)
        if tail_from_input:
            x, conv_tok = tail.x, tail.conv_tok
            assert x.dim() == 2 and x.stride(1) == 1 and x.shape[1] == conv_states.shape[1]
            assert conv_tok.shape[0] == n
            tail = x
            tail_strides = (x.stride(0), 1, x.stride(0))
        else:
            assert tail.dim() == 3
            tail_strides = (tail.stride(0), tail.stride(1), tail.stride(2))
    else:
        conv_states, tail = ssm_states, ssm_states
        width, conv_state_size, conv_block, conv_programs = 1, 1, 1, 0
        tail_strides = (0, 0, 0)
    _, h, v, k = ssm_states.shape
    block_v, block_k = min(v, 128), min(k, 128)
    ssm_programs = h * triton.cdiv(v, block_v) * triton.cdiv(k, block_k) if copy_ssm else 0
    grid = (n, ssm_programs + conv_programs)
    _fold_seed_terminal_slots_kernel[grid](
        ssm_states,
        conv_states,
        tail,
        fold.fold_s1,
        fold.fold_s2,
        ssm_states.stride(0),
        ssm_states.stride(1),
        ssm_states.stride(2),
        ssm_states.stride(3),
        conv_states.stride(0),
        *tail_strides,
        conv_tok,
        H=h,
        V=v,
        K=k,
        BLOCK_V=block_v,
        BLOCK_K=block_k,
        HAS_CONV=has_conv,
        TAIL_FROM_INPUT=tail_from_input,
        WIDTH=width,
        CONV_STATE_SIZE=conv_state_size,
        CONV_BLOCK=conv_block,
        HAS_SSM=copy_ssm,
    )


def _fold_tail_recurrent_ready(inputs: Optional["_FoldTailRecurrentInputs"], ssm_states: torch.Tensor, fold) -> bool:
    """Whether the folds of this call take the recurrent tail path: knob on, few
    enough folds, the FlashInfer bf16-state kernels available for this pool, a
    token-major conv output and 32-byte aligned gate columns."""
    if inputs is None or not _FOLD_TAIL_RECURRENT or fold.fold_count > _FOLD_TAIL_RECURRENT_MAX_FOLDS:
        return False
    if fold.fold_count > 1 and not hasattr(fold, "fold_s1_aligned"):
        return False  # one aligned slot index per fold (Mamba2Metadata); a fold_s1[i:i + 1] slice is not
    _, dk, _, dv = inputs.heads
    if not flashinfer_gdn_bf16_state_available(ssm_states.dtype, dk, dv):
        return False
    x, a, b = inputs.conv_out, inputs.a, inputs.b
    if x.dim() != 2 or x.stride(1) != 1 or (x.stride(0) * x.element_size()) % 32 or x.data_ptr() % 32:
        return False
    for t in (a, b):
        if t.stride(1) != 1 or t.data_ptr() % 32 or (t.stride(0) * t.element_size()) % 32:
            return False
    return True


def fold_conv_tail(
    x_t: torch.Tensor, fold, width: int, fused: Optional[bool] = None
) -> torch.Tensor:
    """Raw pre-conv inputs of the ``width`` tokens before each fold point.

    ``x_t`` is the transposed packed prefill input ``[conv_dim, T]`` BEFORE the
    in-place conv; the returned ``[n_fold, conv_dim, width]`` is exactly what
    the conv kernel would have left in the snapshot slot had the chunk ended
    at the fold point (segment A is >= tokens_per_block > width tokens long,
    so no initial-state carry is involved).
    """
    if fused is None:
        fused = _FOLD_FUSED_BOOKKEEPING
    if fused:
        idx = fold.conv_tail_index(width)
    else:
        idx = (
            fold.fold_conv_tok.unsqueeze(1)
            - width
            + torch.arange(width, device=x_t.device, dtype=fold.fold_conv_tok.dtype)
        ).reshape(-1)
    tail = x_t.index_select(1, idx)
    return tail.view(x_t.shape[0], -1, width).permute(1, 0, 2)


def fold_commit_conv_states(
    conv_states: torch.Tensor, fold, tail: torch.Tensor, fused: Optional[bool] = None
) -> None:
    """After the conv over the whole chunk (cache index S1): move the chunk-end
    conv state S1 -> S2 and put the fold-point state ``tail`` into S1 (a staged
    [fold, conv_dim, width] tensor or the rows of the pre-conv input)."""
    if fused is None:
        fused = _FOLD_FUSED_BOOKKEEPING
    n = fold.fold_count
    if isinstance(tail, _ConvTailFromInput):
        width = conv_states.shape[2]
        rows = tail.x.index_select(0, fold.conv_tail_index(width))
        tail = rows.view(n, width, -1).permute(0, 2, 1)
    if (
        fused
        and n > 0
        and conv_states.dim() == 3
        and conv_states.stride(2) == 1
        and conv_states.stride(1) == conv_states.shape[2]
        and tail.dim() == 3
    ):
        width = conv_states.shape[2]
        state_size = conv_states.shape[1] * width
        block_size = 1024
        grid = (n, triton.cdiv(state_size, block_size))
        _fold_commit_conv_kernel[grid](
            conv_states,
            tail,
            fold.fold_s1,
            fold.fold_s2,
            conv_states.stride(0),
            tail.stride(0),
            tail.stride(1),
            tail.stride(2),
            width,
            state_size,
            block_size,
        )
        return
    s1 = fold.fold_s1.long()
    s2 = fold.fold_s2.long()
    conv_states.index_copy_(0, s2, conv_states.index_select(0, s1))
    conv_states.index_copy_(0, s1, tail.to(conv_states.dtype))


@functools.lru_cache(maxsize=None)
def _fold_tail_indexed_scan_available() -> bool:
    """The per-fold tail scans run as indexed single-launch FlashInfer calls only
    when the FlashInfer adapter is the resolved prefill scan and its indexed
    pool I/O path applies to a one-sequence scan (process-wide constants)."""
    if not _gdn_prefill_uses_flashinfer():
        return False
    from ..fla.flashinfer_chunk import indexed_state_io_available

    return indexed_state_io_available(1)


def fold_scan_tails(
    q,
    k,
    v,
    g,
    beta,
    ssm_states,
    fold,
    out,
    fused: Optional[bool] = None,
    g_is_linear: bool = False,
    conv_commit: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    packed: Optional[_PackedPostConv] = None,
    recurrent: Optional[_FoldTailRecurrentInputs] = None,
) -> None:
    """Second scan launch: every folded tail [fold point, chunk end) from its
    snapshot slot S1 (written by the first launch) into the terminal slot S2;
    the outputs replace the discarded first-launch rows.

    Every tail is one contiguous row range of the packed prefill tokens, so the
    fused path never gathers: one launch seeds S2 with the snapshot (S2 <- S1
    for all folds), then each tail scans its q/k/v/g/beta slices on S2 in place
    through the adapter's indexed pool I/O and writes its rows straight into
    ``out``. Two launches for one fold, ``1 + fold_count`` for several, against
    the gather / gather-cast / scan / cast-scatter / index_copy chain (8 + for
    two folds) of the torch path, with the identical state handoff through the
    pool dtype. Where the indexed path is unavailable the single-fold slice
    form and the gather form below are kept.

    ``conv_commit = (conv_states, tail)`` folds the conv-state commit of
    fold_commit_conv_states into the seed launch (the conv states are not read
    again before the next iteration); the fallback forms run it first.

    ``packed`` (the ``_PackedPostConv`` of the scratch buffers behind q..beta, with
    linear-space ``g``) lets the indexed launches go straight to FlashInfer's
    launcher from cached tail slices; without it they take the generic adapter.

    ``recurrent`` (raw conv output rows, raw gate columns, gate parameters) with
    TLLM_GDN_FOLD_TAIL_RECURRENT=1 runs each tail through FlashInfer's recurrent
    kernel from S1 into S2 instead (see _FOLD_TAIL_RECURRENT); the seed launch
    then only commits the conv states."""
    if fused is None:
        fused = _FOLD_FUSED_BOOKKEEPING
    ranges = getattr(fold, "fold_b_ranges_host", None)
    tail_cu = getattr(fold, "fold_tail_cu_seqlens_long", None)
    sliceable = fused and ranges and out.is_contiguous() and out.dtype == q.dtype
    if sliceable and _fold_tail_recurrent_ready(recurrent, ssm_states, fold):
        if conv_commit is not None:
            fold_seed_terminal_slots(ssm_states, fold, conv_commit[0], conv_commit[1], copy_ssm=False)
        x, a, b, A_log, dt_bias, (hk, dk, hv, dv) = recurrent
        kd = hk * dk
        xs, as_, bs, os_ = x.stride(0), a.stride(0), b.stride(0), out.stride(1)
        # as_strided offsets are absolute in the storage: add each view's own offset
        # (a / b are column slices of the projection, x and out may be slices too).
        xbase, abase, bbase, obase = x.storage_offset(), a.storage_offset(), b.storage_offset(), out.storage_offset()
        scale = dk**-0.5
        aligned = fold.fold_count > 1
        for i, (start, n) in enumerate(ranges):
            # [1, n, ...] views of the packed rows in one op each (as_strided
            # instead of a slice plus a view; the row strides carry the padding).
            xo = xbase + start * xs
            flashinfer_gdn_tail_recurrent(
                A_log,
                dt_bias,
                x.as_strided((1, n, hk, dk), (0, xs, dk, 1), xo),
                x.as_strided((1, n, hk, dk), (0, xs, dk, 1), xo + kd),
                x.as_strided((1, n, hv, dv), (0, xs, dv, 1), xo + 2 * kd),
                a.as_strided((1, n, hv), (0, as_, 1), abase + start * as_),
                b.as_strided((1, n, hv), (0, bs, 1), bbase + start * bs),
                ssm_states,
                fold.fold_s1_aligned(i) if aligned else fold.fold_s1[:1],
                fold.fold_s2_aligned(i) if aligned else fold.fold_s2[:1],
                out.as_strided((1, n, hv, dv), (0, os_, dv, 1), obase + start * os_),
                scale,
            )
        return
    if sliceable and tail_cu is not None and _fold_tail_indexed_scan_available():
        if conv_commit is not None:
            fold_seed_terminal_slots(ssm_states, fold, conv_commit[0], conv_commit[1])
        else:
            fold_seed_terminal_slots(ssm_states, fold)
        if fold.fold_count > _FOLD_TAIL_SLICE_MAX_FOLDS:
            # Many folds: one scan over the gathered tail rows (one gather
            # launch, one scan, one index_copy) instead of a launch per fold;
            # the scan still runs in place on the seeded S2 slots.
            rows = fold.fold_b_rows
            nb = rows.shape[0]
            bufs = _gdn_prefill_scratch.tail_rows(nb, q, v, g, beta)
            gather_tail_rows(q, k, v, g, beta, rows, bufs[:5])
            qb, kb, vb, gb, bb, ob = bufs
            direct = (
                g_is_linear
                and packed is not None
                and chunk_gated_delta_rule_indexed_direct(
                    *_gdn_prefill_scratch.tail_rows_packed(nb, q, v, g, beta),
                    fold.fold_b_cu_seqlens_long,
                    ssm_states,
                    fold.fold_s2,
                )
            )
            if not direct:
                chunk_gated_delta_rule(
                    q=qb,
                    k=kb,
                    v=vb,
                    g=gb,
                    beta=bb,
                    initial_state=ssm_states,
                    initial_state_indices=fold.fold_s2,
                    inplace_indexed_state_update=True,
                    output_final_state=False,
                    cu_seqlens=fold.fold_b_cu_seqlens_long,
                    head_first=False,
                    use_qk_l2norm_in_kernel=False,
                    use_cp=False,
                    output=ob,
                    g_is_linear=g_is_linear,
                )
            out.index_copy_(1, rows, ob)
            return
        out3 = out.squeeze(0) if packed is not None and g_is_linear else None
        for i, (start, n) in enumerate(ranges):
            if out3 is not None and chunk_gated_delta_rule_indexed_direct(
                *packed.tail(start, n),
                out3[start : start + n],
                tail_cu(i),
                ssm_states,
                fold.fold_s2[i : i + 1],
            ):
                continue
            sl = slice(start, start + n)
            chunk_gated_delta_rule(
                q=q[:, sl],
                k=k[:, sl],
                v=v[:, sl],
                g=g[:, sl],
                beta=beta[:, sl],
                initial_state=ssm_states,
                initial_state_indices=fold.fold_s2[i : i + 1],
                inplace_indexed_state_update=True,
                output_final_state=False,
                cu_seqlens=tail_cu(i),
                head_first=False,
                use_qk_l2norm_in_kernel=False,
                # <= tokens_per_block tokens: the plain kernel, indexed on S2.
                use_cp=False,
                output=out[:, sl],
                g_is_linear=g_is_linear,
            )
        return
    if conv_commit is not None:
        fold_commit_conv_states(conv_commit[0], fold, conv_commit[1], fused=fused)
    if sliceable and fold.fold_count == 1:
        start, n = ranges[0]
        sl = slice(start, start + n)
        chunk_gated_delta_rule(
            q=q[:, sl],
            k=k[:, sl],
            v=v[:, sl],
            g=g[:, sl],
            beta=beta[:, sl],
            initial_state=ssm_states,
            initial_state_indices=fold.fold_s1,
            inplace_indexed_state_update=True,
            output_final_state=False,
            cu_seqlens=fold.fold_b_cu_seqlens_long,
            head_first=False,
            use_qk_l2norm_in_kernel=False,
            # <= tokens_per_block tokens per tail: the chunk-parallel path costs
            # four launches plus fix-up passes for nothing here.
            use_cp=False,
            output=out[:, sl],
            output_state_indices=fold.fold_s2,
            g_is_linear=g_is_linear,
        )
        return
    rows = fold.fold_b_rows
    out_b, _ = chunk_gated_delta_rule(
        q=q.index_select(1, rows),
        k=k.index_select(1, rows),
        v=v.index_select(1, rows),
        g=g.index_select(1, rows),
        beta=beta.index_select(1, rows),
        initial_state=ssm_states,
        initial_state_indices=fold.fold_s1,
        inplace_indexed_state_update=True,
        output_final_state=False,
        cu_seqlens=fold.fold_b_cu_seqlens_long,
        head_first=False,
        use_qk_l2norm_in_kernel=False,
        use_cp=False,
        output_state_indices=fold.fold_s2,
        g_is_linear=g_is_linear,
    )
    out.index_copy_(1, rows, out_b.to(out.dtype))


@torch.library.custom_op("trtllm::gdn_custom_op_inplace", mutates_args=("output",))
def gdn_custom_op_inplace(
    mixed_qkv: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    layer_idx: str,
    output: torch.Tensor,
) -> None:
    if _GDN_NVTX_PROBE:
        torch.cuda.nvtx.range_push("gdn_op")
    attn_metadata, gdn_layer, spec_metadata = _extract_gdn_extra_attrs(layer_idx)
    num_tokens = attn_metadata.num_tokens
    gdn_layer.forward_core(
        mixed_qkv[:num_tokens],
        a[:num_tokens],
        b[:num_tokens],
        attn_metadata,
        attn_metadata.mamba_metadata,
        spec_metadata=spec_metadata,
        output=output[:, :num_tokens, :, :],
    )
    if _GDN_NVTX_PROBE:
        torch.cuda.nvtx.range_pop()


maybe_bcg_gdn_custom_op_inplace = eager_on_graph(gdn_custom_op_inplace)


def ensure_divisibility(numerator, denominator):
    """Ensure that numerator is divisible by the denominator."""
    assert numerator % denominator == 0, "{} is not divisible by {}".format(numerator, denominator)


def divide(numerator, denominator):
    """Ensure that numerator is divisible by the denominator and return
    the division value."""
    ensure_divisibility(numerator, denominator)
    return numerator // denominator


# g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
@triton.jit
def fused_gdn_gating_kernel(
    g,
    A_log,
    a,
    dt_bias,
    stride_a_row,
    NUM_HEADS: tl.constexpr,
    beta: tl.constexpr,
    threshold: tl.constexpr,
    BLK_HEADS: tl.constexpr,
):
    i_b, i_d = tl.program_id(0), tl.program_id(1)
    head_off = i_d * BLK_HEADS + tl.arange(0, BLK_HEADS)
    # a may be a row-strided view sliced out of the packed ba projection;
    # g is always allocated packed.
    off_a = i_b * stride_a_row + head_off
    off_g = i_b * NUM_HEADS + head_off
    mask = head_off < NUM_HEADS
    blk_A_log = tl.load(A_log + head_off, mask=mask)
    blk_a = tl.load(a + off_a, mask=mask)
    blk_bias = tl.load(dt_bias + head_off, mask=mask)
    x = blk_a.to(tl.float32) + blk_bias.to(tl.float32)
    softplus_x = tl.where(beta * x <= threshold, (1 / beta) * tl.log(1 + tl.exp(beta * x)), x)
    blk_g = -tl.exp(blk_A_log.to(tl.float32)) * softplus_x
    tl.store(g + off_g, blk_g.to(g.dtype.element_ty), mask=mask)


def fused_gdn_gating(
    A_log: torch.Tensor,
    a: torch.Tensor,
    dt_bias: torch.Tensor,
    beta: float = 1.0,
    threshold: float = 20.0,
) -> torch.Tensor:
    batch, num_heads = a.shape
    grid = (batch, triton.cdiv(num_heads, 8))
    g = torch.empty(batch, num_heads, dtype=torch.float32, device=a.device)
    fused_gdn_gating_kernel[grid](
        g, A_log, a, dt_bias, a.stride(0), num_heads, beta, threshold, 8, num_warps=1
    )
    return g


# The verify path writes each decode request's per-draft-token states into the
# batch-scoped ``intermediate_conv_window`` / ``intermediate_ssm`` buffers at rows
# ``[0, num_decodes)``; ``MambaCacheManager.update_mamba_states()`` reads them
# back from the same rows. The row-index vector is therefore ``arange(num_decodes)``
# for every GDN layer of a step, so build it once per process instead of once
# per layer: prefer the cache-manager-owned arange (allocated together with the
# intermediate buffers), else a cached arange sized to the pool. Slicing a
# persistent buffer is a view, i.e. no allocation and no kernel launch inside the
# CUDA graph. `_cached_arange` is shared with Mamba2Mixer so that both mixers
# hit the same `functools.cache` entry instead of each holding its own copy.


def _verify_intermediate_state_indices(
    kv_cache_manager, num_decodes: int, device: torch.device
) -> torch.Tensor:
    indices = getattr(kv_cache_manager, "intermediate_state_indices", None)
    if indices is None or indices.shape[0] < num_decodes:
        indices = _cached_arange(
            max(kv_cache_manager.get_max_resource_count(), num_decodes), device
        )
    return indices[:num_decodes]


class Qwen3NextGatedDeltaNet(nn.Module):
    def __init__(
        self,
        model_config: ModelConfig[Qwen3NextConfig],
        aux_stream: torch.cuda.Stream,
        layer_idx: Optional[int] = None,
    ):
        super().__init__()
        config = model_config.pretrained_config
        self.model_config = model_config
        self.pretrained_config = config
        replay_enabled = is_gdn_replay_enabled()
        if replay_enabled:
            logger.info_once(
                "GDN MTP replay is requested; set TRTLLM_USE_GDN_REPLAY=0 to disable it",
                key="gdn_mtp_replay_cached",
            )
        else:
            logger.info_once(
                "GDN MTP replay is disabled; set TRTLLM_USE_GDN_REPLAY=1 to enable it",
                key="gdn_mtp_replay_disabled",
            )

        # tensor parallel
        tp_size = model_config.mapping.tp_size
        pp_size = model_config.mapping.pp_size
        if model_config.mapping.enable_attention_dp:
            tp_size = 1

        mapping = Mapping(
            world_size=tp_size * pp_size,
            tp_size=tp_size,
            pp_size=pp_size,
            rank=model_config.mapping.rank,
            gpus_per_node=model_config.mapping.gpus_per_node,
            enable_attention_dp=model_config.mapping.enable_attention_dp,
        )
        self.mapping = mapping

        self.attn_tp_rank = mapping.tp_rank
        self.attn_tp_size = mapping.tp_size
        self.hidden_size = config.hidden_size
        self.num_v_heads = config.linear_num_value_heads
        self.num_k_heads = config.linear_num_key_heads
        self.head_k_dim = config.linear_key_head_dim
        self.head_v_dim = config.linear_value_head_dim
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads
        self.num_k_heads_per_tp = divide(self.num_k_heads, self.attn_tp_size)
        self.num_v_heads_per_tp = divide(self.num_v_heads, self.attn_tp_size)
        self.key_dim_per_tp = self.head_k_dim * self.num_k_heads_per_tp
        self.value_dim_per_tp = self.head_v_dim * self.num_v_heads_per_tp
        self.conv_dim_per_tp = self.key_dim_per_tp * 2 + self.value_dim_per_tp

        self.conv_kernel_size = config.linear_conv_kernel_dim
        self.layer_idx = layer_idx
        self.layer_idx_str = str(layer_idx)
        self.activation = config.hidden_act
        self.layer_norm_epsilon = config.rms_norm_eps

        self.register_to_config = False
        if model_config is not None:
            if "gdn_layers" not in model_config.extra_attrs:
                model_config.extra_attrs["gdn_layers"] = {}
            suffix = 0
            while self.layer_idx_str in model_config.extra_attrs["gdn_layers"]:
                self.layer_idx_str = str(layer_idx) + f"_{suffix}"
                suffix += 1
            model_config.extra_attrs["gdn_layers"][self.layer_idx_str] = weakref.ref(self)
            self.register_to_config = True

        # QKV
        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.conv1d = Linear(
            self.conv_kernel_size,
            self.conv_dim,
            bias=False,
            dtype=config.torch_dtype,
            mapping=mapping,
            tensor_parallel_mode=TensorParallelMode.COLUMN,
            quant_config=model_config.get_quant_config(),
            reduce_output=False,
            skip_create_weights_in_init=model_config.skip_create_weights_in_init,
            allreduce_strategy=model_config.allreduce_strategy,
            force_dynamic_quantization=model_config.force_dynamic_quantization,
            use_cute_dsl_blockscaling_mm=False,
        )

        self.in_proj_qkvz = Linear(
            self.hidden_size,
            self.key_dim * 2 + self.value_dim * 2,
            bias=False,
            dtype=config.torch_dtype,
            mapping=mapping,
            tensor_parallel_mode=TensorParallelMode.COLUMN,
            quant_config=model_config.get_quant_config(),
            reduce_output=False,
            skip_create_weights_in_init=model_config.skip_create_weights_in_init,
            allreduce_strategy=model_config.allreduce_strategy,
            force_dynamic_quantization=model_config.force_dynamic_quantization,
            use_cute_dsl_blockscaling_mm=False,
        )
        self.in_proj_ba = Linear(
            self.hidden_size,
            self.num_v_heads * 2,
            bias=False,
            dtype=config.torch_dtype,
            mapping=mapping,
            tensor_parallel_mode=TensorParallelMode.COLUMN,
            quant_config=model_config.get_quant_config(),
            reduce_output=False,
            skip_create_weights_in_init=model_config.skip_create_weights_in_init,
            allreduce_strategy=model_config.allreduce_strategy,
            force_dynamic_quantization=model_config.force_dynamic_quantization,
            use_cute_dsl_blockscaling_mm=False,
        )

        # time step projection (discretization)
        # instantiate once and copy inv_dt in init_weights of PretrainedModel
        # Fused [qkvz | b | pad | a | pad] input projection weight, built in
        # cache_derived_state(); column offsets of the b and a gate slices.
        self._in_proj_fused_weight: Optional[torch.Tensor] = None
        self._in_proj_qkvz_rows: int = 0
        self._in_proj_b_col: int = 0
        self._in_proj_a_col: int = 0
        self.dt_bias = nn.Parameter(
            torch.ones(
                (self.num_v_heads // self.attn_tp_size),
                dtype=torch.float32,
            ),
            requires_grad=False,
        )

        A = torch.empty(divide(self.num_v_heads, self.attn_tp_size), dtype=torch.float32).uniform_(
            0, 16
        )
        self.A_log = nn.Parameter(
            torch.log(A),
            requires_grad=False,
        )
        self.A_log._no_weight_decay = True

        self.norm = RMSNormGated(
            self.head_v_dim,
            eps=self.layer_norm_epsilon,
            group_size=None,
            norm_before_gate=True,
            device=torch.cuda.current_device(),
            dtype=config.torch_dtype,
        )

        # gemmaNorm is not supported in fused_all_reduce kernel.
        # So, we need to do allReduce in Linear and do gemmaNorm in separate kernel.
        self.out_proj = Linear(
            self.value_dim,
            self.hidden_size,
            bias=False,
            dtype=config.torch_dtype,
            mapping=mapping,
            tensor_parallel_mode=TensorParallelMode.ROW,
            quant_config=model_config.get_quant_config(),
            reduce_output=True,
            skip_create_weights_in_init=model_config.skip_create_weights_in_init,
            allreduce_strategy=model_config.allreduce_strategy,
            force_dynamic_quantization=model_config.force_dynamic_quantization,
            use_cute_dsl_blockscaling_mm=False,
        )

        self.event_dict = {key: torch.cuda.Event() for key in [EventType.Main, EventType.Attention]}
        self.aux_stream = aux_stream

    def cache_derived_state(self) -> None:
        """Attach downstream static quantization state after loading weights."""
        self.norm.fp8_scale = None
        if isinstance(self.out_proj.quant_method, FP8QDQLinearMethod):
            scale = self.out_proj.quant_method.get_static_input_scale(self.out_proj)
            # detach() strips the Parameter wrapper so nn.Module.__setattr__
            # does not register the derived scale into norm's state_dict; the
            # detached view still shares storage with out_proj.input_scale.
            self.norm.fp8_scale = scale.detach() if scale is not None else None
        self._refresh_fused_in_proj()

    def _fused_in_proj_eligible(self) -> bool:
        """The fused weight reproduces exactly the plain bf16 F.linear path of the
        two projections: unquantized, no bias, no LoRA, no cuBLAS/CuTe override,
        no locality-domain shards, both weights of one dtype and K."""
        if not _GDN_FUSED_IN_PROJ:
            return False
        lins = (getattr(self, "in_proj_qkvz", None), getattr(self, "in_proj_ba", None))
        if not all(isinstance(lin, Linear) for lin in lins):
            return False
        for lin in lins:
            if not isinstance(lin.quant_method, UnquantizedLinearMethod):
                return False
            if lin.bias is not None or lin.lora is not None:
                return False
            if lin.use_custom_cublas_mm or lin.use_cute_dsl_bf16_gemm:
                return False
            if getattr(lin, "_locality_domain_weight_shards", None) is not None:
                return False
            weight = getattr(lin, "weight", None)
            if weight is None or weight.dim() != 2:
                return False
        wq, wb = self.in_proj_qkvz.weight, self.in_proj_ba.weight
        return (wq.dtype == wb.dtype and wq.device == wb.device
                and wq.shape[1] == wb.shape[1])

    def _refresh_fused_in_proj(self) -> None:
        """Keep one ``[qkvz | b | pad | a | pad]`` weight for the two input projections.

        ``in_proj_qkvz.weight`` aliases the leading row slab, so in-place weight
        (re)loads land in the fused buffer, which keeps one stable address for
        captured CUDA graphs; a parameter whose storage was replaced is copied
        back in and re-aliased. The ``b`` and ``a`` gate rows are copied into
        two slabs that start at 32-byte-aligned columns of the projection: the
        FlashInfer GDN decode kernel takes ``a`` and ``b`` as 32-byte-aligned
        tensors and the wrapper clones a misaligned slice on every call, which
        adjacent 16-byte ``b`` / ``a`` column blocks (8 heads per rank in bf16)
        forced for ``a`` in every decode graph and every mixed iteration. The
        copy (a few hundred KB) reruns on every ``post_load_weights``, i.e. after
        the checkpoint load and after each RL refit.
        """
        if not self._fused_in_proj_eligible():
            self._in_proj_fused_weight = None
            return
        wq, wb = self.in_proj_qkvz.weight, self.in_proj_ba.weight
        rows_q = wq.shape[0]
        n_gate = wb.shape[0] // 2
        assert wb.shape[0] == 2 * n_gate, f"in_proj_ba rows {wb.shape[0]} are not [b | a]"
        align = max(1, 32 // wq.element_size())  # columns per 32 bytes

        def _up(x: int) -> int:
            return -(-x // align) * align

        b_col = _up(rows_q)
        a_col = b_col + _up(n_gate)
        shape = (a_col + _up(n_gate), wq.shape[1])
        fused = self._in_proj_fused_weight
        if (fused is None or fused.shape != shape or fused.dtype != wq.dtype
                or fused.device != wq.device):
            fused = torch.zeros(shape, dtype=wq.dtype, device=wq.device)
            self._in_proj_fused_weight = fused
        slab = fused[:rows_q]
        if wq.data.data_ptr() != slab.data_ptr() or wq.data.stride() != slab.stride():
            slab.copy_(wq.data)
            wq.data = slab
        fused[b_col:b_col + n_gate].copy_(wb.data[:n_gate])
        fused[a_col:a_col + n_gate].copy_(wb.data[n_gate:])
        self._in_proj_qkvz_rows = rows_q
        self._in_proj_b_col = b_col
        self._in_proj_a_col = a_col

    def post_load_weights(self) -> None:
        self.cache_derived_state()

    def _compute_tokenwise_inputs(self, hidden_states: torch.Tensor):
        fused_weight = self._in_proj_fused_weight
        if fused_weight is not None:
            projected = torch.nn.functional.linear(hidden_states, fused_weight)
            num_tokens = projected.shape[0]
            n_gate = self.num_v_heads_per_tp
            mixed_qkv = projected[:, : self.conv_dim_per_tp]
            z = projected[:, self.conv_dim_per_tp : self._in_proj_qkvz_rows].view(
                num_tokens, n_gate, self.head_v_dim
            )
            # 32-byte-aligned column blocks (see _refresh_fused_in_proj).
            b = projected[:, self._in_proj_b_col : self._in_proj_b_col + n_gate]
            a = projected[:, self._in_proj_a_col : self._in_proj_a_col + n_gate]
            return mixed_qkv, z, a, b
        else:

            def _compute_projected_states_qkvz():
                return self.in_proj_qkvz(hidden_states)

            def _compute_projected_states_ba():
                return self.in_proj_ba(hidden_states)

            projected_states_qkvz, projected_states_ba = maybe_execute_in_parallel(
                _compute_projected_states_qkvz,
                _compute_projected_states_ba,
                self.event_dict[EventType.Main],
                self.event_dict[EventType.Attention],
                self.aux_stream,
                disable_on_compile=True,
            )

        # The weight mapper reorders in_proj rows into the dense per-rank
        # layouts [Q|K|V|Z] and [b|a] (see grouped_to_dense_in_proj_qkvz_perm),
        # so every component is a plain column slice of the projection —
        # no split/reshape kernel. Downstream consumers (causal_conv1d,
        # the GDN decode kernels, the gated norm) read these row-strided
        # views in place.
        num_tokens = projected_states_qkvz.shape[0]
        mixed_qkv = projected_states_qkvz[:, : self.conv_dim_per_tp]
        z = projected_states_qkvz[:, self.conv_dim_per_tp :].view(
            num_tokens, self.num_v_heads_per_tp, self.head_v_dim
        )
        b = projected_states_ba[:, : self.num_v_heads_per_tp]
        a = projected_states_ba[:, self.num_v_heads_per_tp :]

        return mixed_qkv, z, a, b

    def _postprocess_gdn_output(
        self,
        attn_out: torch.Tensor,
        z: torch.Tensor,
        all_reduce_params: Optional[AllReduceParams] = None,
    ):
        # z is a [num_tokens, num_v_heads, head_v_dim] view of the in_proj
        # output whose (heads, head_dim) block is contiguous per token; the
        # gated norm reads it through its token stride instead of packing a
        # copy.
        attn_out = rms_norm_gated_token_major(
            attn_out.reshape(-1, self.head_v_dim),
            z,
            self.norm.weight,
            self.norm.eps,
            fp8_scale=self.norm.fp8_scale,
        )
        attn_out = attn_out.view(-1, self.value_dim_per_tp)
        return self.out_proj(attn_out, all_reduce_params=all_reduce_params)

    def _replay_verify_recurrent(
        self,
        query,
        key,
        value,
        a,
        b,
        ssm_states,
        state_indices_d,
        num_decodes,
        draft_token_num,
        replay_metadata,
        layer_cache,
        replay_work_items,
        replay_n_writes,
        output_d=None,
        packed_qkv=None,
        use_all_layer_commit=False,
    ):
        """Run MTP target verification via the replay kernel.

        This avoids intermediate-state writes and accepted-state copies; commits
        are deferred via the compact history cache. Cached replay reuses the
        Mamba2 fields as old_x<->U, old_B<->normalized k, and
        old_dt<->cumulative G.
        """
        assert replay_metadata is not None, (
            "GDN replay enabled but replay metadata was not allocated."
        )
        assert draft_token_num == replay_metadata.replay_step_width, (
            "GDN replay does not support dynamic draft length yet: "
            f"{draft_token_num} != {replay_metadata.replay_step_width}"
        )
        if draft_token_num > 8 or replay_metadata.replay_history_size > 16:
            raise RuntimeError(
                "GDN cached replay requires draft_token_num <= 8 and replay_history_size <= 16."
            )
        return fused_recurrent_gated_delta_rule_cached_replay_update(
            q=query,
            k=key,
            v=value,
            g=a,
            beta=b,
            A_log=self.A_log,
            dt_bias=self.dt_bias,
            launch_with_pdl=True,
            ssm_states=ssm_states,
            state_indices=state_indices_d[:num_decodes],
            old_u=layer_cache.old_x,
            old_k=layer_cache.old_B,
            old_G=layer_cache.old_dt,
            old_beta=layer_cache.old_dA_cumsum,
            cache_buf_idx=layer_cache.cache_buf_idx,
            prev_num_accepted_tokens=layer_cache.prev_num_accepted_tokens,
            history_size=replay_metadata.replay_history_size,
            replay_work_items=replay_work_items,
            n_writes=replay_n_writes,
            use_qk_l2norm_in_kernel=True,
            packed_qkv=packed_qkv,
            use_all_layer_commit=use_all_layer_commit,
            output=output_d,
        )

    def forward_decode(
        self,
        conv_states,
        ssm_states,
        query_start_loc_long,
        spec_metadata: Optional[SpecMetadata] = None,
        intermediate_conv_states: Optional[torch.Tensor] = None,
        intermediate_ssm_states: Optional[torch.Tensor] = None,
        is_target_verify: bool = False,
        output: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        mixed_qkv = kwargs["mixed_qkv"]
        a = kwargs["a"]
        b = kwargs["b"]
        cache_indices = kwargs["cache_indices"]
        num_decodes = kwargs["num_decodes"]

        if is_target_verify:
            draft_token_num = spec_metadata.runtime_draft_len + 1
            assert num_decodes > 0
            assert mixed_qkv.shape[0] == num_decodes * draft_token_num
            assert a.shape[0] == num_decodes * draft_token_num
            assert b.shape[0] == num_decodes * draft_token_num
            assert intermediate_conv_states is not None
            assert kwargs.get("use_replay", False) or intermediate_ssm_states is not None

            # Speculative verification path:
            # 1. run conv update with per-step intermediate cache writes
            # 2. run recurrent delta rule with intermediate SSM-state cache writes
            # 3. defer final state selection to kv_cache_manager.update_mamba_states()
            intermediate_state_indices = kwargs["intermediate_state_indices"]

            mixed_qkv_reshaped = mixed_qkv.reshape(num_decodes, draft_token_num, -1).transpose(1, 2)
            mixed_qkv_processed = causal_conv1d_update_triton(
                mixed_qkv_reshaped,
                conv_states,
                self.conv1d.weight,
                self.conv1d.bias,
                self.activation,
                conv_state_indices=cache_indices[:num_decodes],
                intermediate_conv_window=intermediate_conv_states,
                intermediate_state_indices=intermediate_state_indices,
                # PDL chain: conv1d -> replay verify kernel (replay only)
                launch_dependent_kernels=kwargs.get("use_replay", False),
            )
            mixed_qkv = mixed_qkv_processed.transpose(1, 2).reshape(
                num_decodes * draft_token_num, -1
            )

            key_size = self.key_dim // self.attn_tp_size
            query = mixed_qkv[..., :key_size]
            key = mixed_qkv[..., key_size : key_size * 2]
            value = mixed_qkv[..., key_size * 2 :]

            query = query.reshape(
                num_decodes, draft_token_num, self.num_k_heads // self.attn_tp_size, self.head_k_dim
            )
            key = key.reshape(
                num_decodes, draft_token_num, self.num_k_heads // self.attn_tp_size, self.head_k_dim
            )
            value = value.reshape(
                num_decodes, draft_token_num, self.num_v_heads // self.attn_tp_size, self.head_v_dim
            )

            a = a.reshape(num_decodes, draft_token_num, -1)
            b = b.reshape(num_decodes, draft_token_num, -1)

            # Prefer the FlashInfer MTP kernel (raw a/b gating in-kernel,
            # initial state gathered from the pool via cache indices, per-step
            # intermediate states written to the batch-scoped [:num_decodes]
            # prefix consumed by update_mamba_states()); fall back to the
            # Triton recurrent kernel when unavailable.
            if kwargs.get("use_replay", False):
                output_d = None
                if output is not None:
                    output_d = output.view(
                        num_decodes,
                        draft_token_num,
                        self.num_v_heads // self.attn_tp_size,
                        self.head_v_dim,
                    )
                return self._replay_verify_recurrent(
                    query,
                    key,
                    value,
                    a,
                    b,
                    ssm_states,
                    cache_indices,
                    num_decodes,
                    draft_token_num,
                    kwargs.get("replay_metadata"),
                    kwargs.get("layer_cache"),
                    kwargs.get("replay_work_items"),
                    kwargs.get("replay_n_writes"),
                    output_d,
                    packed_qkv=mixed_qkv,
                    use_all_layer_commit=kwargs.get("use_cached_replay_all_layer_commit", False),
                ).view(
                    1,
                    num_decodes * draft_token_num,
                    self.num_v_heads // self.attn_tp_size,
                    self.head_v_dim,
                )

            if _can_use_flashinfer_gdn_verify(
                ssm_states, self.head_k_dim, self.head_v_dim, draft_token_num
            ):
                output_d = None
                if output is not None:
                    output_d = output.view(
                        num_decodes,
                        draft_token_num,
                        self.num_v_heads // self.attn_tp_size,
                        self.head_v_dim,
                    )
                return _flashinfer_gdn_verify(
                    A_log=self.A_log,
                    a=a,
                    dt_bias=self.dt_bias,
                    softplus_beta=1.0,
                    softplus_threshold=20.0,
                    q=query,
                    k=key,
                    v=value,
                    b=b,
                    initial_state_source=ssm_states,
                    initial_state_indices=cache_indices[:num_decodes],
                    intermediate_states_buffer=intermediate_ssm_states[:num_decodes],
                    scale=self.head_k_dim**-0.5,
                    use_qk_l2norm_in_kernel=True,
                    output=output_d,
                ).view(
                    1,
                    num_decodes * draft_token_num,
                    self.num_v_heads // self.attn_tp_size,
                    self.head_v_dim,
                )

            beta = b.sigmoid()
            g = fused_gdn_gating(
                self.A_log,
                a.view(num_decodes * draft_token_num, -1),
                self.dt_bias,
            ).reshape(num_decodes, draft_token_num, -1)

            # Keep intermediate-state indexing consistent with Mamba2Mixer:
            # cache slots [0..num_decodes-1] are consumed by
            # MambaCacheManager.update_mamba_states(), while initial states are
            # gathered from real slot indices.
            recurrent_state_source = ssm_states[cache_indices[:num_decodes]]
            recurrent_state_indices = kwargs["intermediate_state_indices"]

            output_d = None
            if output is not None:
                output_d = output.view(
                    num_decodes,
                    draft_token_num,
                    self.num_v_heads // self.attn_tp_size,
                    self.head_v_dim,
                )

            attn_out = fused_recurrent_gated_delta_rule_update(
                q=query,
                k=key,
                v=value,
                g=g,
                beta=beta,
                initial_state_source=recurrent_state_source,
                initial_state_indices=recurrent_state_indices,
                use_qk_l2norm_in_kernel=True,
                disable_state_update=True,
                intermediate_states_buffer=intermediate_ssm_states,
                cache_steps=draft_token_num,
                output=output_d,
            )
            return attn_out.view(
                1,
                num_decodes * draft_token_num,
                self.num_v_heads // self.attn_tp_size,
                self.head_v_dim,
            )

        # Standard decode (1 token/seq): when the FlashInfer kernel is not
        # taking this batch, run the conv update and the delta-rule update as a
        # PDL-overlapped pair -- the update kernel's dominant latency (the SSM
        # state-pool read) is independent of the conv output, so it executes
        # concurrently with the conv kernel (measured -1.2 us/layer at bs=64 on
        # VR200). Disable with TRTLLM_GDN_DISABLE_PDL_DECODE_PAIR=1.
        num_seqs = query_start_loc_long.numel() - 1
        if not _can_use_flashinfer_gdn_decode(
            ssm_states, self.head_k_dim, self.head_v_dim, mixed_qkv.shape[0], num_seqs
        ) and can_use_gdn_decode_pdl_pair(
            ssm_states,
            mixed_qkv.shape[0],
            num_seqs,
            self.head_k_dim,
            self.head_v_dim,
            self.activation,
        ):
            mixed_qkv = causal_conv1d_update_triton(
                mixed_qkv,
                conv_states,
                self.conv1d.weight,
                self.conv1d.bias,
                self.activation,
                conv_state_indices=cache_indices,
                launch_dependent_kernels=True,
            )
            return gdn_decode_pdl_update(
                A_log=self.A_log,
                a=a,
                dt_bias=self.dt_bias,
                softplus_beta=1.0,
                softplus_threshold=20.0,
                mixed_qkv=mixed_qkv,
                b=b,
                initial_state_source=ssm_states,
                initial_state_indices=cache_indices,
                num_k_heads=self.num_k_heads_per_tp,
                num_v_heads=self.num_v_heads_per_tp,
                head_k_dim=self.head_k_dim,
                head_v_dim=self.head_v_dim,
                use_qk_l2norm_in_kernel=True,
                output=output,
            )

        mixed_qkv = causal_conv1d_update(
            mixed_qkv,
            conv_states,
            self.conv1d.weight,
            self.conv1d.bias,
            self.activation,
            conv_state_indices=cache_indices,
        )

        # Keep q/k/v as views over mixed_qkv so the fused decode kernel can
        # consume their native strides without forcing packed copies.
        query = mixed_qkv[..., : self.key_dim_per_tp]
        key = mixed_qkv[..., self.key_dim_per_tp : self.key_dim_per_tp * 2]
        value = mixed_qkv[..., self.key_dim_per_tp * 2 :]
        seq_len = query.shape[0]
        query = query.view(1, seq_len, self.num_k_heads_per_tp, self.head_k_dim)
        key = key.view(1, seq_len, self.num_k_heads_per_tp, self.head_k_dim)
        value = value.view(1, seq_len, self.num_v_heads_per_tp, self.head_v_dim)

        core_attn_out = fused_sigmoid_gating_delta_rule_update(
            A_log=self.A_log,
            dt_bias=self.dt_bias,
            q=query,
            k=key,
            v=value,
            a=a,
            b=b,
            initial_state_source=ssm_states,
            initial_state_indices=cache_indices,
            cu_seqlens=query_start_loc_long,
            use_qk_l2norm_in_kernel=True,
            softplus_beta=1.0,
            softplus_threshold=20.0,
            output=output,
        )

        return core_attn_out

    def forward_extend(
        self,
        conv_states,
        ssm_states,
        spec_metadata: Optional[SpecMetadata] = None,
        intermediate_conv_states: Optional[torch.Tensor] = None,
        intermediate_ssm_states: Optional[torch.Tensor] = None,
        is_target_verify: bool = False,
        output: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        mixed_qkv = kwargs["mixed_qkv"]
        a = kwargs["a"]
        b = kwargs["b"]
        batch_size = kwargs["batch_size"]
        has_initial_states = kwargs.get("has_initial_states_bs")
        if has_initial_states is None:
            has_initial_states = kwargs["has_initial_states"][:batch_size]
        cache_indices = kwargs["cache_indices"]
        query_start_loc = kwargs["query_start_loc"]
        query_start_loc_long = kwargs["query_start_loc_long"]
        num_prefill_tokens = kwargs["num_prefill_tokens"]
        num_decode_tokens = kwargs["num_decode_tokens"]
        state_indices_p = kwargs["state_indices_p"]
        state_indices_d = kwargs["state_indices_d"]
        num_prefill = kwargs["num_prefill"]
        num_decodes = kwargs["num_decodes"]
        fold = kwargs.get("fold")
        if fold is not None and not kwargs.get("fold_checked", False):
            self._check_fold_supported(is_target_verify)
        # Gate convention of the resolved prefill scan (see _gdn_prefill_uses_flashinfer).
        g_linear = kwargs.get("g_linear")
        if g_linear is None:
            g_linear = _gdn_prefill_uses_flashinfer()
        # (post-conv decode rows, a, b) when they take the recurrent kernel instead of the scan.
        decode_recurrent: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None
        # FlashInfer's packed views of the scratch-backed post-conv outputs (None off the scratch).
        packed: Optional[_PackedPostConv] = None
        # Recurrent fold-tail inputs and the conv commit fused into the post-conv launch (None unless taken).
        recurrent: Optional[_FoldTailRecurrentInputs] = None
        fold_conv = None

        conv_states_to_use = conv_states

        if num_decode_tokens > 0:
            mixed_qkv_p = mixed_qkv[:num_prefill_tokens]
            mixed_qkv_d = mixed_qkv[num_prefill_tokens:]
            query_start_loc_p = kwargs.get("query_start_loc_p")
            if query_start_loc_p is None:
                query_start_loc_p = query_start_loc[: num_prefill + 1]
            has_initial_states_p = kwargs.get("has_initial_states_p")
            if has_initial_states_p is None:
                has_initial_states_p = has_initial_states[:num_prefill]

            mixed_qkv_p_t, fold_conv_tail = self._prefill_conv(
                mixed_qkv_p, conv_states_to_use, has_initial_states_p, state_indices_p, query_start_loc_p, fold
            )

            if is_target_verify:
                a_d = a[num_prefill_tokens:]
                b_d = b[num_prefill_tokens:]
                draft_token_num = spec_metadata.runtime_draft_len + 1
                assert num_decodes > 0
                assert mixed_qkv_d.shape[0] == num_decodes * draft_token_num
                assert a_d.shape[0] == num_decodes * draft_token_num
                assert b_d.shape[0] == num_decodes * draft_token_num
                assert intermediate_conv_states is not None
                assert kwargs.get("use_replay", False) or intermediate_ssm_states is not None

                intermediate_state_indices = kwargs["intermediate_state_indices"]
                mixed_qkv_d = mixed_qkv_d.reshape(num_decodes, draft_token_num, -1).transpose(1, 2)
                mixed_qkv_d = causal_conv1d_update_triton(
                    mixed_qkv_d,
                    conv_states_to_use,
                    self.conv1d.weight,
                    self.conv1d.bias,
                    activation=self.activation,
                    conv_state_indices=state_indices_d,
                    intermediate_conv_window=intermediate_conv_states,
                    intermediate_state_indices=intermediate_state_indices,
                    # PDL chain: conv1d -> replay verify kernel (replay only)
                    launch_dependent_kernels=kwargs.get("use_replay", False),
                )
                mixed_qkv_d = mixed_qkv_d.transpose(1, 2).reshape(num_decode_tokens, -1)
            else:
                mixed_qkv_d = causal_conv1d_update(
                    mixed_qkv_d,
                    conv_states_to_use,
                    self.conv1d.weight,
                    self.conv1d.bias,
                    activation=self.activation,
                    conv_state_indices=state_indices_d,
                )
            if is_target_verify:
                if num_prefill_tokens > 0:
                    query_p, key_p, value_p, g_p, beta_p = fused_gdn_post_conv(
                        mixed_qkv_p_t,
                        None,
                        a[:num_prefill_tokens],
                        b[:num_prefill_tokens],
                        self.A_log,
                        self.dt_bias,
                        self.num_k_heads_per_tp,
                        self.head_k_dim,
                        self.num_v_heads_per_tp,
                        self.head_v_dim,
                        beta_dtype=b.dtype,
                        g_linear=g_linear,
                    )
                query_d, key_d, value_d = pack_gdn_decode_qkv(
                    mixed_qkv_d,
                    self.num_k_heads_per_tp,
                    self.head_k_dim,
                    self.num_v_heads_per_tp,
                    self.head_v_dim,
                )
            elif _MIXED_DECODE_RECURRENT:
                # Prefill rows only; the decode rows go through the recurrent
                # kernel after the scan (see _mixed_decode_recurrent).
                decode_recurrent = (mixed_qkv_d, a[num_prefill_tokens:], b[num_prefill_tokens:])
                recurrent, fold_conv = self._fold_tail_recurrent_setup(
                    fold, mixed_qkv_p_t, mixed_qkv_p, a[:num_prefill_tokens], b[:num_prefill_tokens],
                    conv_states_to_use, fold_conv_tail, ssm_states,
                )
                query, key, value, g, beta, packed = self._post_conv(
                    mixed_qkv_p_t, None, a[:num_prefill_tokens], b[:num_prefill_tokens], g_linear, fold_conv
                )
            else:
                query, key, value, g, beta, packed = self._post_conv(mixed_qkv_p_t, mixed_qkv_d, a, b, g_linear)
        else:
            mixed_qkv_t, fold_conv_tail = self._prefill_conv(
                mixed_qkv, conv_states_to_use, has_initial_states, cache_indices, query_start_loc, fold
            )
            recurrent, fold_conv = self._fold_tail_recurrent_setup(
                fold, mixed_qkv_t, mixed_qkv, a, b, conv_states_to_use, fold_conv_tail, ssm_states
            )
            query, key, value, g, beta, packed = self._post_conv(mixed_qkv_t, None, a, b, g_linear, fold_conv)

        if is_target_verify and num_decode_tokens > 0:
            attn_out_prefill = None
            if num_prefill_tokens > 0:
                output_p = output[:, :num_prefill_tokens, :, :] if output is not None else None
                # Same indexed, in-place state I/O as the non-verify prefill path
                # below: read each request's initial state from its pool slot and
                # write the final state back to it inside the kernel, instead of
                # gathering a [num_prefill, H, V, K] copy and scattering it back.
                # Preconditions match that path: fresh slots were zeroed in
                # forward_core, and prefill slots never alias the decode slots
                # verified in the same step.
                attn_out_prefill, _ = chunk_gated_delta_rule(
                    q=query_p,
                    k=key_p,
                    v=value_p,
                    g=g_p,
                    beta=beta_p,
                    initial_state=ssm_states,
                    initial_state_indices=state_indices_p,
                    inplace_indexed_state_update=True,
                    output_final_state=False,
                    cu_seqlens=query_start_loc_long[: num_prefill + 1],
                    head_first=False,
                    use_qk_l2norm_in_kernel=False,
                    output=output_p,
                    state_workspace=kwargs.get("state_workspace"),
                    g_is_linear=g_linear,
                )

            draft_token_num = spec_metadata.runtime_draft_len + 1
            query_d = query_d.reshape(
                num_decodes, draft_token_num, self.num_k_heads // self.attn_tp_size, self.head_k_dim
            )
            key_d = key_d.reshape(
                num_decodes, draft_token_num, self.num_k_heads // self.attn_tp_size, self.head_k_dim
            )
            value_d = value_d.reshape(
                num_decodes, draft_token_num, self.num_v_heads // self.attn_tp_size, self.head_v_dim
            )

            a_d = a_d.reshape(num_decodes, draft_token_num, -1)
            b_d = b_d.reshape(num_decodes, draft_token_num, -1)
            out_v_heads = self.num_v_heads // self.attn_tp_size

            output_d = None
            if output is not None:
                output_d = output[:, num_prefill_tokens:, :, :].view(
                    num_decodes,
                    draft_token_num,
                    out_v_heads,
                    self.head_v_dim,
                )

            if kwargs.get("use_replay", False):
                attn_out_decode = self._replay_verify_recurrent(
                    query_d,
                    key_d,
                    value_d,
                    a_d,
                    b_d,
                    ssm_states,
                    state_indices_d,
                    num_decodes,
                    draft_token_num,
                    kwargs.get("replay_metadata"),
                    kwargs.get("layer_cache"),
                    kwargs.get("replay_work_items"),
                    kwargs.get("replay_n_writes"),
                    output_d,
                    use_all_layer_commit=kwargs.get("use_cached_replay_all_layer_commit", False),
                ).reshape(1, num_decode_tokens, out_v_heads, self.head_v_dim)
            elif _can_use_flashinfer_gdn_verify(
                ssm_states, self.head_k_dim, self.head_v_dim, draft_token_num
            ):
                # FI gathers the initial state from the pool via state_indices_d
                # (no host gather) and writes batch-scoped intermediate states;
                # the [:num_decodes] prefix matches update_mamba_states()'s rows.
                attn_out_decode = _flashinfer_gdn_verify(
                    A_log=self.A_log,
                    a=a_d,
                    dt_bias=self.dt_bias,
                    softplus_beta=1.0,
                    softplus_threshold=20.0,
                    q=query_d,
                    k=key_d,
                    v=value_d,
                    b=b_d,
                    initial_state_source=ssm_states,
                    initial_state_indices=state_indices_d,
                    intermediate_states_buffer=intermediate_ssm_states[:num_decodes],
                    scale=self.head_k_dim**-0.5,
                    use_qk_l2norm_in_kernel=True,
                    output=output_d,
                ).reshape(1, num_decode_tokens, out_v_heads, self.head_v_dim)
            else:
                beta_d = b_d.sigmoid()
                g_d = fused_gdn_gating(
                    self.A_log,
                    a_d.view(num_decodes * draft_token_num, -1),
                    self.dt_bias,
                ).reshape(num_decodes, draft_token_num, -1)

                recurrent_state_source = ssm_states[state_indices_d]
                recurrent_state_indices = kwargs["intermediate_state_indices"]

                attn_out_decode = fused_recurrent_gated_delta_rule_update(
                    q=query_d,
                    k=key_d,
                    v=value_d,
                    g=g_d,
                    beta=beta_d,
                    initial_state_source=recurrent_state_source,
                    initial_state_indices=recurrent_state_indices,
                    use_qk_l2norm_in_kernel=True,
                    disable_state_update=True,
                    intermediate_states_buffer=intermediate_ssm_states,
                    cache_steps=draft_token_num,
                    output=output_d,
                ).view(1, num_decode_tokens, out_v_heads, self.head_v_dim)

            if output is not None:
                return output
            if attn_out_prefill is None:
                return attn_out_decode
            return torch.cat((attn_out_prefill, attn_out_decode), dim=1)

        if kwargs.get("scan_state_indices") is not None and kwargs.get("scan_cu_seqlens") is not None:
            scan_state_indices = kwargs["scan_state_indices"]
            scan_cu_seqlens = kwargs["scan_cu_seqlens"]
        elif fold is None:
            scan_state_indices = cache_indices
            scan_cu_seqlens = query_start_loc_long
        else:
            # Folded save-last chunks are split into segment A (-> snapshot slot
            # S1, in place) and a discarded segment B (-> terminal slot S2); the
            # real tail runs in _fold_scan_tails from S1 into S2.
            scan_state_indices = fold.scan_state_indices
            scan_cu_seqlens = fold.scan_cu_seqlens_long
        output_p = output
        if decode_recurrent is not None:
            # The scan covers the context rows only; its output lands in the
            # leading rows of ``output`` (context tokens are packed first).
            layout = kwargs.get("scan_prefill_layout")
            if layout is None:
                layout = _mixed_scan_prefill_layout(
                    num_prefill, fold, scan_state_indices, scan_cu_seqlens
                )
            scan_state_indices, scan_cu_seqlens = layout
            if output is not None:
                output_p = output[:, :num_prefill_tokens]
        # Plain chunked kernel with indexed pool I/O for short prefills (one
        # launch); FlashInfer's own heuristic above the threshold.
        use_cp = _gdn_prefill_use_cp(num_prefill_tokens)
        core_attn_out = None
        if packed is not None and g_linear and use_cp is False and output_p is not None:
            # FlashInfer's layouts straight from the scratch views: the launch
            # only, none of the adapter's layout / dtype / validation work.
            if chunk_gated_delta_rule_indexed_direct(
                *packed, output_p.squeeze(0), scan_cu_seqlens, ssm_states, scan_state_indices
            ):
                core_attn_out = output_p
        if core_attn_out is None:
            core_attn_out, _ = chunk_gated_delta_rule(
                q=query,
                k=key,
                v=value,
                g=g,
                beta=beta,
                initial_state=ssm_states,
                initial_state_indices=scan_state_indices,
                # This path writes recurrent state directly back into the shared
                # pool; callers **must** ensure cache_indices do not alias live slots.
                inplace_indexed_state_update=True,
                output_final_state=False,
                cu_seqlens=scan_cu_seqlens,
                head_first=False,
                use_qk_l2norm_in_kernel=False,
                output=output_p,
                use_cp=use_cp,
                state_workspace=kwargs.get("state_workspace"),
                g_is_linear=g_linear,
            )
        if fold is not None:
            # When the caller pre-allocated `output`, write the tail rows into it
            # as well (the custom-op path discards the return value). The conv
            # state commit (S2 <- chunk end, S1 <- fold point) rides in the same
            # seed launch: nothing reads the two conv slots again this iteration.
            # With the recurrent tail path the conv-state commit already rode
            # in the post-conv launch (fold_conv); the seed launch is skipped.
            self._fold_scan_tails(
                query,
                key,
                value,
                g,
                beta,
                ssm_states,
                fold,
                output_p if output is not None else core_attn_out,
                g_is_linear=g_linear,
                conv_commit=None if fold_conv is not None else (conv_states_to_use, fold_conv_tail),
                packed=packed,
                recurrent=recurrent,
            )
        if decode_recurrent is not None:
            mixed_qkv_d, a_d, b_d = decode_recurrent
            decode_slots = kwargs.get("decode_state_indices")
            if decode_slots is None:
                decode_slots = state_indices_d
            elif decode_slots.shape[0] != num_decodes:
                decode_slots = decode_slots[:num_decodes]
            attn_out_decode = self._mixed_decode_recurrent(
                mixed_qkv_d,
                a_d,
                b_d,
                ssm_states,
                decode_slots,
                kwargs["decode_cu_seqlens_long"],
                output[:, num_prefill_tokens:] if output is not None else None,
            )
            if output is not None:
                return output
            return torch.cat((core_attn_out, attn_out_decode), dim=1)

        return core_attn_out

    def _get_prefill_state_workspace(
        self, attn_metadata: AttentionMetadata, ssm_states: torch.Tensor
    ) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
        """Reserve maximum state scratch on first use and reuse it across layers."""
        # Blackwell uses indexed pool I/O directly, without state scratch.
        if not _use_flashinfer_gdn_prefill() or is_sm_100f():
            return None
        key = (ssm_states.device, ssm_states.shape[1:])
        # First use during warmup reserves the configured maximum, not the current batch.
        # Sequential layers share this persistent input/output scratch.
        workspaces = self.model_config.extra_attrs.setdefault("gdn_state_workspaces", {})
        workspace = workspaces.get(key)
        capacity = min(
            attn_metadata.max_num_sequences or attn_metadata.max_num_requests,
            attn_metadata.max_num_tokens,
        )
        if workspace is None or workspace[0].shape[0] < capacity:
            state_in = torch.empty(
                (capacity, *ssm_states.shape[1:]), dtype=torch.float32, device=ssm_states.device
            )
            workspace = workspaces[key] = (state_in, torch.empty_like(state_in))
        return workspace

    def _mixed_decode_recurrent(
        self,
        mixed_qkv_d: torch.Tensor,
        a_d: torch.Tensor,
        b_d: torch.Tensor,
        ssm_states: torch.Tensor,
        state_indices_d: torch.Tensor,
        cu_seqlens_long: torch.Tensor,
        output_d: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """The decode rows of a mixed iteration through the standard decode kernel.

        Same kernel and same raw inputs (post-conv q/k/v views, raw ``a``/``b``
        gates, in-kernel L2 norm) as ``forward_decode``'s standard path, so the
        decode requests of a mixed iteration follow exactly the numerics of a
        pure-decode iteration. Updates the pool slots ``state_indices_d`` in place.

        With a preallocated output, a bf16 pool and 32-byte aligned gate columns
        and slot buffer (the fused in_proj layout and Mamba2Metadata's decode slot
        buffer provide both) the batched views go straight to FlashInfer's
        launcher (flashinfer_gdn_decode_t1); the same kernel, without the two
        adapter layers.
        """
        n = mixed_qkv_d.shape[0]
        kd = self.key_dim_per_tp
        hk, dk, hv, dv = self.num_k_heads_per_tp, self.head_k_dim, self.num_v_heads_per_tp, self.head_v_dim
        if (
            output_d is not None
            and flashinfer_gdn_decode_direct_available(ssm_states.dtype, dk, dv)
            and state_indices_d.dtype == torch.int32
            and a_d.data_ptr() % 32 == 0
            and b_d.data_ptr() % 32 == 0
            and state_indices_d.data_ptr() % 32 == 0
        ):
            flashinfer_gdn_decode_t1(
                self.A_log,
                self.dt_bias,
                mixed_qkv_d[:, :kd].view(n, 1, hk, dk),
                mixed_qkv_d[:, kd : 2 * kd].view(n, 1, hk, dk),
                mixed_qkv_d[:, 2 * kd :].view(n, 1, hv, dv),
                a_d.view(n, 1, hv),
                b_d.view(n, 1, hv),
                ssm_states,
                state_indices_d,
                output_d.view(n, 1, hv, dv),
                dk**-0.5,
            )
            return output_d
        query = mixed_qkv_d[..., : self.key_dim_per_tp].view(
            1, n, self.num_k_heads_per_tp, self.head_k_dim
        )
        key = mixed_qkv_d[..., self.key_dim_per_tp : self.key_dim_per_tp * 2].view(
            1, n, self.num_k_heads_per_tp, self.head_k_dim
        )
        value = mixed_qkv_d[..., self.key_dim_per_tp * 2 :].view(
            1, n, self.num_v_heads_per_tp, self.head_v_dim
        )
        return fused_sigmoid_gating_delta_rule_update(
            A_log=self.A_log,
            dt_bias=self.dt_bias,
            q=query,
            k=key,
            v=value,
            a=a_d,
            b=b_d,
            initial_state_source=ssm_states,
            initial_state_indices=state_indices_d,
            cu_seqlens=cu_seqlens_long,
            use_qk_l2norm_in_kernel=True,
            softplus_beta=1.0,
            softplus_threshold=20.0,
            output=output_d,
        )

    # ------------------------------------------------------------------
    # Folded save-last prefill: the save-last snapshot point ``reachable``
    # sits inside the (single) context chunk instead of ending it. The conv
    # kernel and the first scan launch treat the chunk as today's chunk 1
    # would (state slot S1 = the snapshot block); the tail after the fold
    # point is then re-run from S1 into the terminal slot S2, and S1 receives
    # the conv state at the fold point. Numerically this reproduces the
    # two-chunk schedule exactly (same segment boundaries, same pool-dtype
    # state handoff) without the second ADP iteration.
    # ------------------------------------------------------------------
    def _check_fold_supported(self, is_target_verify: bool) -> None:
        if is_target_verify:
            raise RuntimeError(
                "TLLM_MAMBA_FOLD_SAVE_LAST=1 is not supported together with speculative "
                "decoding target verification"
            )
        impl = _resolve_chunk_gated_delta_rule()
        if impl.__module__.rsplit(".", 1)[-1] != "flashinfer_chunk":
            raise RuntimeError(
                "TLLM_MAMBA_FOLD_SAVE_LAST=1 requires the FlashInfer GDN prefill path "
                "(TLLM_USE_FLASHINFER_GDN_PREFILL=1 on a supported architecture)"
            )

    def _fold_conv_tail(self, x_t: torch.Tensor, fold) -> torch.Tensor:
        return fold_conv_tail(x_t, fold, self.conv_kernel_size - 1)

    def _prefill_conv_input(
        self, mixed_qkv_p: torch.Tensor, fold=None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Transposed ``[conv_dim, num_prefill_tokens]`` conv input of the prefill
        rows and, on a folded iteration, the ``[fold_count, conv_dim, d_conv - 1]``
        fold-point conv tails, both written by the one extract launch."""
        num_tokens, width = mixed_qkv_p.shape
        out = tail = None
        conv_tok = getattr(fold, "fold_conv_tok", None) if fold is not None else None
        if _GDN_PREFILL_SCRATCH:
            out = _gdn_prefill_scratch.conv_input(width, num_tokens, mixed_qkv_p.dtype, mixed_qkv_p.device)
            if conv_tok is not None:
                tail = _gdn_prefill_scratch.conv_tail(
                    fold.fold_count, width, self.conv_kernel_size - 1, mixed_qkv_p.dtype, mixed_qkv_p.device
                )
        elif conv_tok is not None:
            tail = torch.empty(
                fold.fold_count, width, self.conv_kernel_size - 1, dtype=mixed_qkv_p.dtype, device=mixed_qkv_p.device
            )
        x_t = extract_transpose_prefill_slice(
            mixed_qkv_p, num_tokens, 0, width, out=out, tail=tail, conv_tok=conv_tok, check=out is None
        )
        if fold is not None and tail is None:
            tail = self._fold_conv_tail(x_t, fold)
        return x_t, tail

    def _prefill_conv(
        self,
        mixed_qkv_p: torch.Tensor,
        conv_states: torch.Tensor,
        has_initial_state: torch.Tensor,
        cache_indices: torch.Tensor,
        query_start_loc: torch.Tensor,
        fold=None,
    ):
        """The prefill causal conv. Channel-last: the [tokens, conv_dim] projection
        slice is handed to the kernel as its transposed view and the result is
        written token-major into the scratch (no extract-transpose launch, no
        copy); the fold-point conv tails are then read from the untouched input
        by the seed launch. Returns the ``[conv_dim, tokens]`` conv output view
        and the fold tails (``None`` without a fold)."""
        if _GDN_CONV_CHANNEL_LAST and _GDN_PREFILL_SCRATCH and mixed_qkv_p.stride(1) == 1:
            num_tokens, width = mixed_qkv_p.shape
            out = _gdn_prefill_scratch.conv_out(width, num_tokens, mixed_qkv_p.dtype, mixed_qkv_p.device)
            y = causal_conv1d_fn(
                mixed_qkv_p.t(),
                self.conv1d.weight,
                self.conv1d.bias,
                activation=self.activation,
                conv_states=conv_states,
                has_initial_state=has_initial_state,
                cache_indices=cache_indices,
                query_start_loc=query_start_loc,
                out=out.t(),
            )
            tail = _ConvTailFromInput(mixed_qkv_p, fold.fold_conv_tok) if fold is not None else None
            return y, tail
        x_t, tail = self._prefill_conv_input(mixed_qkv_p, fold)
        y = causal_conv1d_fn(
            x_t,
            self.conv1d.weight,
            self.conv1d.bias,
            activation=self.activation,
            conv_states=conv_states,
            has_initial_state=has_initial_state,
            cache_indices=cache_indices,
            query_start_loc=query_start_loc,
        )
        return y, tail

    def _post_conv(self, prefill_t: torch.Tensor, decode, a, b, g_linear: bool, fold_conv=None):
        """fused_gdn_post_conv on scratch-backed outputs (fp32 beta; the scan paths):
        ``(q, k, v, g, beta, packed)`` with ``packed`` the ``_PackedPostConv`` of
        the same buffers (None without the scratch). ``fold_conv`` commits the
        folded requests' conv states in the same launch (fused_gdn_post_conv)."""
        out = packed = None
        if _GDN_PREFILL_SCRATCH:
            num_tokens = prefill_t.shape[1] + (0 if decode is None else decode.shape[0])
            shape = (
                num_tokens,
                self.num_k_heads_per_tp,
                self.head_k_dim,
                self.num_v_heads_per_tp,
                self.head_v_dim,
                prefill_t.dtype,
                prefill_t.device,
            )
            out = _gdn_prefill_scratch.post_conv(*shape)
            packed = _gdn_prefill_scratch.post_conv_packed(*shape)
        q, k, v, g, beta = fused_gdn_post_conv(
            prefill_t,
            decode,
            a,
            b,
            self.A_log,
            self.dt_bias,
            self.num_k_heads_per_tp,
            self.head_k_dim,
            self.num_v_heads_per_tp,
            self.head_v_dim,
            g_linear=g_linear,
            out=out,
            check=out is None,
            fold_conv=fold_conv,
        )
        return q, k, v, g, beta, packed

    def _fold_commit_conv_states(self, conv_states: torch.Tensor, fold, tail: torch.Tensor) -> None:
        fold_commit_conv_states(conv_states, fold, tail)

    @torch.no_grad()
    def warmup_fold_kernels(self, kv_cache_manager=None) -> None:
        """Compile the Triton variants of the folded save-last prefill path on dummy
        data: the channel-last conv + post-conv layout, the S2 seed + conv commit
        (tail from the input and staged), the tail row gather and the indexed tail
        scans; with TLLM_GDN_FOLD_TAIL_RECURRENT the recurrent tail kernel for
        every tail length. The engine warm-up never folds a chunk (folds need a
        save-last snapshot inside the chunk), so without this the first folded
        iteration of a serving run JIT-compiled them in rank lockstep (~0.7 s on
        Qwen3.5-397B; ~1 s per tail length for the recurrent kernel).

        With ``kv_cache_manager`` the layer's real state pools are used (slots
        1-5, free at warm-up time): FlashInfer compiles its CuTe kernels per pool
        geometry -- size and slot strides of a non-contiguous pool, which the
        per-layer view of the all-layer pool is -- so dummy pools would compile
        variants serving never uses. Without it, dummy pools."""
        dev = self.A_log.device
        dtype = self.in_proj_qkvz.weight.dtype
        width = self.conv_kernel_size - 1
        conv_dim = self.conv_dim_per_tp
        fused = self._in_proj_fused_weight
        row = fused.shape[0] if fused is not None else conv_dim + self.num_v_heads_per_tp * 2
        lens = [24, 24]
        total = sum(lens)
        proj = torch.randn(total, row, dtype=dtype, device=dev) * 0.1
        mixed_qkv = proj[:, :conv_dim]
        conv_states = ssm_states = None
        layer_cache = (
            kv_cache_manager.mamba_layer_cache(self.layer_idx)
            if kv_cache_manager is not None and hasattr(kv_cache_manager, "mamba_layer_cache")
            else None
        )
        if layer_cache is not None:
            conv_pool = getattr(layer_cache, "conv", None)
            ssm_pool = getattr(layer_cache, "temporal", None)
            if (
                isinstance(conv_pool, torch.Tensor)
                and isinstance(ssm_pool, torch.Tensor)
                and conv_pool.shape[0] >= 6
                and ssm_pool.shape[0] >= 6
                and conv_pool.shape[1:] == (conv_dim, width)
                and ssm_pool.shape[1:] == (self.num_v_heads_per_tp, self.head_v_dim, self.head_k_dim)
            ):
                conv_states, ssm_states = conv_pool, ssm_pool
                dtype = ssm_pool.dtype if ssm_pool.dtype in (torch.bfloat16, torch.float16) else dtype
        if conv_states is None:
            conv_states = torch.zeros(6, conv_dim, width, dtype=dtype, device=dev)
            ssm_states = torch.zeros(
                6, self.num_v_heads_per_tp, self.head_v_dim, self.head_k_dim, dtype=dtype, device=dev
            )
        query_start_loc = torch.tensor([0, lens[0], total], dtype=torch.int32, device=dev)
        cache_indices = torch.tensor([1, 4], dtype=torch.int32, device=dev)
        has_initial_state = torch.tensor([False, True], device=dev)
        n_gate = self.num_v_heads_per_tp
        a = torch.randn(total, n_gate, dtype=dtype, device=dev)
        b = torch.randn(total, n_gate, dtype=dtype, device=dev)
        # two folded requests: tails [16, 24) and [40, 48)
        b_rows = list(range(16, 24)) + list(range(40, 48))
        fold = SimpleNamespace(
            fold_count=2,
            fold_s1=torch.tensor([1, 4], dtype=torch.int32, device=dev),
            fold_s2=torch.tensor([2, 5], dtype=torch.int32, device=dev),
            fold_conv_tok=torch.tensor([16, 40], dtype=torch.long, device=dev),
            fold_b_rows=torch.tensor(b_rows, dtype=torch.long, device=dev),
            fold_b_cu_seqlens_long=torch.tensor([0, 8, 16], dtype=torch.long, device=dev),
            fold_b_ranges_host=[(16, 8), (40, 8)],
        )
        tail_cu = torch.tensor([0, 8, 0, 8], dtype=torch.long, device=dev)
        fold.fold_tail_cu_seqlens_long = lambda i: tail_cu[2 * i : 2 * i + 2]
        fold.conv_tail_index = lambda w: torch.tensor(
            [tok - w + j for tok in (16, 40) for j in range(w)], dtype=torch.long, device=dev
        )
        g_linear = _gdn_prefill_uses_flashinfer()
        x_t, tail = self._prefill_conv(mixed_qkv, conv_states, has_initial_state, cache_indices, query_start_loc, fold)
        q, k, v, g, beta, packed = self._post_conv(x_t, None, a, b, g_linear)
        out = torch.empty(1, total, self.num_v_heads_per_tp, self.head_v_dim, dtype=dtype, device=dev)
        # tails: seed + commit from the input rows, gather + indexed scan (2 folds) ...
        fold_scan_tails(
            q, k, v, g, beta, ssm_states, fold, out, g_is_linear=g_linear, conv_commit=(conv_states, tail), packed=packed
        )
        # ... the single-fold slice scan, and the staged-tail commit variant
        fold1 = SimpleNamespace(**{k_: v_ for k_, v_ in vars(fold).items()})
        fold1.fold_count, fold1.fold_b_ranges_host = 1, [(16, 8)]
        fold1.fold_s1, fold1.fold_s2 = fold.fold_s1[:1], fold.fold_s2[:1]
        fold1.fold_conv_tok = fold.fold_conv_tok[:1]
        fold1.fold_b_rows, fold1.fold_b_cu_seqlens_long = fold.fold_b_rows[:8], fold.fold_b_cu_seqlens_long[:2]
        fold1.fold_tail_cu_seqlens_long = fold.fold_tail_cu_seqlens_long
        fold1.conv_tail_index = lambda w: fold.conv_tail_index(w)[:w]
        fold_scan_tails(q, k, v, g, beta, ssm_states, fold1, out, g_is_linear=g_linear, packed=packed)
        staged = torch.zeros(1, conv_dim, width, dtype=dtype, device=dev)
        fold_seed_terminal_slots(ssm_states, fold1, conv_states, staged)
        if _FOLD_TAIL_RECURRENT and flashinfer_gdn_bf16_state_available(
            ssm_states.dtype, self.head_k_dim, self.head_v_dim
        ):
            # FlashInfer compiles its recurrent kernel per tail length (~1 s
            # each); every length up to the block size occurs in serving.
            fold_seed_terminal_slots(ssm_states, fold1, conv_states, staged, copy_ssm=False)
            # the post-conv launch with the fused conv commit (one and two folds)
            for f in (fold, fold1):
                self._post_conv(
                    x_t, None, a, b, g_linear,
                    fold_conv=(mixed_qkv, conv_states, f.fold_s1, f.fold_s2, f.fold_conv_tok),
                )
            xr = x_t.t() if x_t.stride(0) == 1 else torch.randn(total, conv_dim, dtype=dtype, device=dev)
            a32 = torch.randn(total, 32, dtype=dtype, device=dev)[:, :n_gate] if n_gate % 16 else a
            b32 = torch.randn(total, 32, dtype=dtype, device=dev)[:, :n_gate] if n_gate % 16 else b
            hk, dk, hv, dv = self.num_k_heads_per_tp, self.head_k_dim, self.num_v_heads_per_tp, self.head_v_dim
            kd = hk * dk
            for n in range(1, min(32, total) + 1):
                xs = xr[:n]
                flashinfer_gdn_tail_recurrent(
                    self.A_log,
                    self.dt_bias,
                    xs[:, :kd].view(1, n, hk, dk),
                    xs[:, kd : 2 * kd].view(1, n, hk, dk),
                    xs[:, 2 * kd :].view(1, n, hv, dv),
                    a32[:n].view(1, n, hv),
                    b32[:n].view(1, n, hv),
                    ssm_states,
                    fold1.fold_s1,
                    fold1.fold_s2,
                    out[:, :n].view(1, n, hv, dv),
                    dk**-0.5,
                )
        torch.cuda.synchronize()

    def _fold_tail_recurrent_setup(self, fold, conv_out_t, x, a, b, conv_states, fold_conv_tail, ssm_states):
        """``(recurrent inputs, fold_conv)`` for a folded iteration on the recurrent
        tail path, ``(None, None)`` otherwise. ``fold_conv`` lets the post-conv launch
        commit the conv states (S2 <- chunk end, S1 <- fold point from the pre-conv
        rows ``x``) so the seed launch is not needed; only when the fold tails come
        from the input rows (channel-last conv), as the seed kernel would read them."""
        if fold is None or not _FOLD_TAIL_RECURRENT:
            return None, None
        recurrent = _FoldTailRecurrentInputs(
            conv_out_t.t(),
            a,
            b,
            self.A_log,
            self.dt_bias,
            (self.num_k_heads_per_tp, self.head_k_dim, self.num_v_heads_per_tp, self.head_v_dim),
        )
        if not _fold_tail_recurrent_ready(recurrent, ssm_states, fold):
            return None, None
        fold_conv = None
        if isinstance(fold_conv_tail, _ConvTailFromInput) and conv_states is not None:
            fold_conv = (fold_conv_tail.x, conv_states, fold.fold_s1, fold.fold_s2, fold_conv_tail.conv_tok)
        return recurrent, fold_conv

    def _fold_scan_tails(
        self,
        q,
        k,
        v,
        g,
        beta,
        ssm_states,
        fold,
        out,
        g_is_linear: bool = False,
        conv_commit=None,
        packed=None,
        recurrent=None,
    ) -> None:
        fold_scan_tails(
            q,
            k,
            v,
            g,
            beta,
            ssm_states,
            fold,
            out,
            g_is_linear=g_is_linear,
            conv_commit=conv_commit,
            packed=packed,
            recurrent=recurrent,
        )

    def _reset_prefill_states(
        self,
        attn_metadata: AttentionMetadata,
        mamba_metadata,
        ssm_states: torch.Tensor,
        conv_states: torch.Tensor,
        state_indices_p: torch.Tensor,
        has_initial_states_p: torch.Tensor,
    ) -> None:
        """Zero the recurrent and conv state of context requests that start without one.

        Host-gated and hoisted: ``Mamba2Metadata.prepare`` records whether any
        context request of the iteration needs a reset (continuation chunks
        under prefix reuse never do), and the first GDN layer that runs clears
        the slots of every local layer in one launch. Each layer used to launch
        its own masked kernel unconditionally: 45 launches and ~5 ms of host time
        per mixed iteration on the aggregated engine, almost all of it for nothing.
        Managers that expose no all-layer pool keep the per-layer reset.
        """
        if not getattr(mamba_metadata, "prefill_needs_state_reset", True):
            return
        if getattr(mamba_metadata, "state_reset_done", False):
            return
        pools = _all_layer_state_pools(attn_metadata.kv_cache_manager)
        if pools is None:
            reset_recurrent_state_rows(
                ssm_states, state_indices_p, has_initial_states_p, conv_states
            )
            return
        _reset_gdn_states_all_layers(pools[0], pools[1], state_indices_p, has_initial_states_p)
        mamba_metadata.state_reset_done = True

    def forward(
        self,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        mamba_metadata: Mamba2Metadata,
        spec_metadata: Optional[SpecMetadata] = None,
        all_reduce_params: Optional[AllReduceParams] = None,
    ):
        mixed_qkv, z, a, b = self._compute_tokenwise_inputs(hidden_states)

        use_breakable_cuda_graph = not is_torch_compiling() and is_in_breakable_cuda_graph()
        if self.register_to_config and (is_torch_compiling() or use_breakable_cuda_graph):
            attn_out = mixed_qkv.new_empty(
                (1, mixed_qkv.shape[0], self.num_v_heads_per_tp, self.head_v_dim)
            )
            maybe_bcg_gdn_custom_op_inplace(mixed_qkv, a, b, self.layer_idx_str, attn_out)
        else:
            attn_out = self.forward_core(
                mixed_qkv,
                a,
                b,
                attn_metadata,
                mamba_metadata,
                spec_metadata=spec_metadata,
            )

        return self._postprocess_gdn_output(attn_out, z, all_reduce_params)

    def _iteration_kwargs(
        self,
        attn_metadata: AttentionMetadata,
        mamba_metadata: Mamba2Metadata,
        spec_metadata: Optional[SpecMetadata],
        has_layer_cache: bool,
    ) -> dict:
        """The metadata-derived arguments of forward_extend / forward_decode for
        one iteration: batch split, request-slot and initial-state views, scan
        layout, replay bookkeeping. Built once per iteration by the first GDN
        layer (each tensor view costs a few microseconds of host time, and the
        45 layers of a mixed iteration used to rebuild all of them)."""
        num_prefills = attn_metadata.num_contexts
        batch_size = attn_metadata.seq_lens.shape[0]
        num_decodes = batch_size - num_prefills
        num_prefill_tokens = attn_metadata.num_ctx_tokens
        num_decode_tokens = attn_metadata.num_tokens - num_prefill_tokens
        has_initial_states = mamba_metadata.has_initial_states
        state_indices = mamba_metadata.state_indices[:batch_size]
        state_indices_p, state_indices_d = torch.split(state_indices, [num_prefills, num_decodes])
        kv_cache_manager = attn_metadata.kv_cache_manager
        is_target_verify = (
            num_decodes > 0
            and spec_metadata is not None
            and kv_cache_manager.is_speculative()
            and has_layer_cache
        )
        intermediate_state_indices = (
            _verify_intermediate_state_indices(kv_cache_manager, num_decodes, state_indices.device)
            if is_target_verify
            else None
        )
        use_replay = is_target_verify and getattr(kv_cache_manager, "use_replay_state_update", False)
        replay_metadata = (
            kv_cache_manager.get_replay_state_update_metadata() if use_replay else None
        )
        use_cached_replay_all_layer_commit = (
            num_decodes >= CACHED_REPLAY_PARTITION_MIN_BATCH_SIZE
            and getattr(kv_cache_manager, "use_gdn_cached_replay_all_layer_commit", False)
        )
        fold = mamba_metadata if getattr(mamba_metadata, "fold_count", 0) > 0 else None
        if fold is not None:
            # Once per iteration instead of once per layer (resolves the scan implementation).
            self._check_fold_supported(is_target_verify)
        decode_state_indices = getattr(mamba_metadata, "state_indices_decode", None)
        if decode_state_indices is not None:
            decode_state_indices = decode_state_indices[:num_decodes]
        # Decode-only iterations carry no prefill layout (Mamba2Metadata.prepare
        # leaves query_start_loc None); the prefill-side views exist only when
        # there are context requests.
        query_start_loc = mamba_metadata.query_start_loc
        query_start_loc_long = mamba_metadata.query_start_loc_long
        query_start_loc_p = scan_state_indices = scan_cu_seqlens = scan_prefill_layout = None
        if num_prefills > 0:
            if query_start_loc is not None:
                query_start_loc_p = query_start_loc[: num_prefills + 1]
            if fold is None:
                scan_state_indices, scan_cu_seqlens = state_indices, query_start_loc_long
            else:
                scan_state_indices, scan_cu_seqlens = fold.scan_state_indices, fold.scan_cu_seqlens_long
            if scan_cu_seqlens is not None:
                scan_prefill_layout = _mixed_scan_prefill_layout(
                    num_prefills, fold, scan_state_indices, scan_cu_seqlens
                )
        has_initial_states_bs = has_initial_states_p = None
        if has_initial_states is not None:
            has_initial_states_bs = has_initial_states[:batch_size]
            has_initial_states_p = has_initial_states[:num_prefills]
        return {
            "has_initial_states": has_initial_states,
            "has_initial_states_bs": has_initial_states_bs,
            "has_initial_states_p": has_initial_states_p,
            "cache_indices": state_indices,
            "query_start_loc": query_start_loc,
            "query_start_loc_p": query_start_loc_p,
            "query_start_loc_long": query_start_loc_long,
            "batch_size": batch_size,
            "num_prefill_tokens": num_prefill_tokens,
            "num_decode_tokens": num_decode_tokens,
            "state_indices_p": state_indices_p,
            "state_indices_d": state_indices_d,
            "num_prefill": num_prefills,
            "num_decodes": num_decodes,
            "is_target_verify": is_target_verify,
            "intermediate_state_indices": intermediate_state_indices,
            "use_replay": use_replay,
            "replay_metadata": replay_metadata,
            "replay_work_items": mamba_metadata.replay_work_items[:num_decodes]
            if use_replay and num_decodes >= CACHED_REPLAY_PARTITION_MIN_BATCH_SIZE
            else None,
            "replay_n_writes": mamba_metadata.replay_n_writes
            if use_replay and num_decodes >= CACHED_REPLAY_PARTITION_MIN_BATCH_SIZE
            else None,
            "use_cached_replay_all_layer_commit": use_cached_replay_all_layer_commit,
            # Folded save-last prefill layout (None on iterations without one),
            # already checked against the resolved scan implementation.
            "fold": fold,
            "fold_checked": fold is not None,
            # The main scan's layout: the context segments (two per folded
            # request) first, then the decode requests as one-token sequences;
            # its prefill-only prefix serves the mixed-iteration scan.
            "scan_state_indices": scan_state_indices,
            "scan_cu_seqlens": scan_cu_seqlens,
            "scan_prefill_layout": scan_prefill_layout,
            # cu_seqlens of the decode requests as one-token sequences and their
            # slots in an aligned buffer (mixed iterations, see _mixed_decode_recurrent).
            "decode_cu_seqlens_long": mamba_metadata.arange_long(num_decodes),
            "decode_state_indices": decode_state_indices,
            # Gate convention of the resolved prefill scan (see _gdn_prefill_uses_flashinfer).
            "g_linear": _gdn_prefill_uses_flashinfer(),
        }

    def forward_core(
        self,
        mixed_qkv: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        attn_metadata: AttentionMetadata,
        mamba_metadata: Mamba2Metadata,
        spec_metadata: Optional[SpecMetadata] = None,
        output: Optional[torch.Tensor] = None,
    ):
        ### sglang linear attn
        # has_initial_states = None
        # if forward_batch.extend_prefix_lens is not None:
        #     has_initial_states = forward_batch.extend_prefix_lens > 0

        ### mamba2_mixer layer
        layer_cache = attn_metadata.kv_cache_manager.mamba_layer_cache(self.layer_idx)
        conv_states = layer_cache.conv
        ssm_states = layer_cache.temporal

        # Everything below that derives from the metadata alone is the same for
        # every GDN layer of the iteration: the first layer builds it, the
        # others reuse it (Mamba2Metadata.prepare clears the slot).
        common = getattr(mamba_metadata, "gdn_iteration_kwargs", None)
        if common is None:
            common = self._iteration_kwargs(
                attn_metadata, mamba_metadata, spec_metadata, layer_cache is not None
            )
            if hasattr(mamba_metadata, "gdn_iteration_kwargs"):
                mamba_metadata.gdn_iteration_kwargs = common
        num_prefills = common["num_prefill"]
        if num_prefills > 0:
            # PyExecutor guarantees prefill requests are placed before decode requests
            self._reset_prefill_states(
                attn_metadata,
                mamba_metadata,
                ssm_states,
                conv_states,
                common["state_indices_p"],
                common["has_initial_states_p"],
            )
        kwargs = dict(common)
        # forward_extend / forward_decode take is_target_verify as a named argument.
        is_target_verify = kwargs.pop("is_target_verify")
        intermediate_conv_states = (
            layer_cache.intermediate_conv_window if is_target_verify else None
        )
        intermediate_ssm_states = layer_cache.intermediate_ssm if is_target_verify else None
        kwargs["mixed_qkv"] = mixed_qkv
        kwargs["a"] = a
        kwargs["b"] = b
        kwargs["layer_cache"] = layer_cache
        if num_prefills > 0:
            attn_out = self.forward_extend(
                conv_states,
                ssm_states,
                spec_metadata=spec_metadata,
                intermediate_conv_states=intermediate_conv_states,
                intermediate_ssm_states=intermediate_ssm_states,
                is_target_verify=is_target_verify,
                output=output,
                state_workspace=self._get_prefill_state_workspace(attn_metadata, ssm_states),
                **kwargs,
            )
        else:
            attn_out = self.forward_decode(
                conv_states,
                ssm_states,
                spec_metadata=spec_metadata,
                intermediate_conv_states=intermediate_conv_states,
                intermediate_ssm_states=intermediate_ssm_states,
                is_target_verify=is_target_verify,
                output=output,
                **kwargs,
            )

        return attn_out
