# Adapted from https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py
# Adapted from https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/configs/qwen3_next.py
# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import functools
import os
import weakref
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl
from torch import nn
from transformers import Qwen3NextConfig

from tensorrt_llm._torch.modules.fla.cached_replay import (
    CACHED_REPLAY_PARTITION_MIN_BATCH_SIZE,
    fused_recurrent_gated_delta_rule_cached_replay_update,
)
from tensorrt_llm._torch.modules.fla.flashinfer_cached_replay import flashinfer_cached_replay_update
from tensorrt_llm._torch.modules.fla.flashinfer_chunk import chunk_gated_delta_rule_indexed_direct
from tensorrt_llm._torch.modules.fla.fused_recurrent import fused_recurrent_gated_delta_rule_update
from tensorrt_llm._torch.modules.fla.fused_sigmoid_gating_recurrent import (
    _can_use_flashinfer_gdn_verify,
    _flashinfer_gdn_verify,
    flashinfer_gdn_decode_direct_available,
    flashinfer_gdn_decode_t1,
    fused_sigmoid_gating_delta_rule_update,
)
from tensorrt_llm._torch.modules.fla.utils import tensor_cache
from tensorrt_llm._utils import (
    is_flashinfer_gdn_decode_supported_arch,
    is_flashinfer_gdn_prefill_supported_arch,
    is_sm_100f,
)
from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping

from ...attention.backends import AttentionMetadata
from ...distributed import AllReduceParams
from ...model_config import ModelConfig
from ...pyexecutor.breakable_cuda_graph import eager_on_graph, is_in_breakable_cuda_graph
from ...pyexecutor.kv_cache.mamba_cache_manager.common import _stack_state_views
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
# Only taken for plain bf16 (unquantized, no bias) projections.


# FlashInfer GDN prefill is ON by default; set TLLM_USE_FLASHINFER_GDN_PREFILL=0
# to force the vendored Triton chunk_gated_delta_rule everywhere. FlashInfer
# ships the GDN prefill kernel for Hopper (SM90), datacenter Blackwell
# (SM100/SM103) and consumer Blackwell (SM120/SM121); on other archs it aborts
# at launch, so we fall back to Triton there. Resolution is deferred to first call
# (and cached) so importing this module never initializes CUDA.
def _use_flashinfer_gdn_prefill() -> bool:
    """Check the prefill backend setting and supported GPU architecture."""
    return (
        os.getenv("TLLM_USE_FLASHINFER_GDN_PREFILL", "1") == "1"
        and is_flashinfer_gdn_prefill_supported_arch()
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


@tensor_cache
def _prefill_query_start_loc(query_start_loc: torch.Tensor, num_prefill: int) -> torch.Tensor:
    # FLA caches chunk metadata by tensor identity. Keep this view stable across
    # warmup and CUDA graph capture so preparing the metadata needs no host copy.
    return query_start_loc[: num_prefill + 1]


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
# The plain kernel is also faster on the GPU up to this size. Above the
# threshold the adapter's default ("auto") leaves the choice to FlashInfer. The
# pin applies on SM90/SM100/SM103 only; FlashInfer's prefill path also runs on
# SM120/SM121, which keep its own "auto" routing.
_GDN_PREFILL_CP_MIN_TOKENS = 4096


@functools.lru_cache(maxsize=1)
def _gdn_prefill_plain_pin_applies() -> bool:
    """Whether this GPU is one the plain-kernel pin was chosen for."""
    return is_flashinfer_gdn_decode_supported_arch()


def _gdn_prefill_use_cp(num_prefill_tokens: int):
    """``use_cp`` for the main prefill scan of an iteration with this many packed tokens."""
    if num_prefill_tokens > _GDN_PREFILL_CP_MIN_TOKENS or not _gdn_prefill_plain_pin_applies():
        return "auto"
    return False


# Mixed iterations (context chunks + decode requests in one batch): run the
# decode requests' tokens through the recurrent decode kernel, as pure-decode
# iterations do, instead of scanning them as one-token sequences in the chunked
# prefill kernel. The scan pays 70-110 us per layer for the ~190 one-token
# sequences a typical mixed iteration carries (gdn_prefill_mixedbatch_bench.py:
# one 512-token chunk 69 us alone, 143-175 us with 192-256 decode sequences),
# the decode kernel ~10 us.


def _mixed_scan_prefill_layout(
    num_prefill: int, scan_state_indices: torch.Tensor, scan_cu_seqlens: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """The prefill-only prefix of a mixed scan layout.

    The layout (``query_start_loc`` / ``cache_indices``) lists the context
    requests' segments first and the decode requests, one token each, after them.
    """
    return scan_state_indices[:num_prefill], _prefill_query_start_loc(scan_cu_seqlens, num_prefill)


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
    (``all_ssm_states`` / ``all_conv_states``); the V2 manager keeps them as lists of
    per-layer views, stacked here when their layer spacing is constant; the Python
    manager keeps them as the ``temporal`` / ``conv`` tensors of its state container.
    ``None`` when the manager exposes none of these, in which case the caller falls
    back to the per-layer reset.
    """
    ssm = getattr(kv_cache_manager, "all_ssm_states", None)
    conv = getattr(kv_cache_manager, "all_conv_states", None)
    if isinstance(ssm, torch.Tensor) and isinstance(conv, torch.Tensor):
        return ssm, conv
    if isinstance(ssm, list) and isinstance(conv, list) and ssm and len(ssm) == len(conv):
        # The V2 manager keeps one logical-slot-indexed view per layer; stack them when affine.
        key = (len(ssm), ssm[0].data_ptr(), conv[0].data_ptr())
        cached = getattr(kv_cache_manager, "_gdn_stacked_pools", None)
        if cached is None or cached[0] != key:
            stacked_ssm, stacked_conv = _stack_state_views(ssm), _stack_state_views(conv)
            pools = (
                (stacked_ssm, stacked_conv)
                if stacked_ssm is not None and stacked_conv is not None
                else None
            )
            cached = kv_cache_manager._gdn_stacked_pools = (key, pools)
        return cached[1]
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


class _PackedPostConv(tuple):
    """``(q3, k3, v3, g2, beta2)``: the ``[T, H, D]`` / ``[T, HV]`` views FlashInfer's
    launchers take of the post-conv scratch buffers (same storage as the
    ``[1, T, ...]`` views ``fused_gdn_post_conv`` fills) for
    chunk_gated_delta_rule_indexed_direct."""


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

    def conv_input(
        self, conv_dim: int, num_tokens: int, dtype: torch.dtype, device
    ) -> torch.Tensor:
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
        key = (
            "post",
            num_tokens,
            num_k_heads,
            head_k_dim,
            num_v_heads,
            head_v_dim,
            dtype,
            beta_dtype,
            device,
        )
        views = self._views.get(key)
        if views is None:
            nk = num_tokens * num_k_heads * head_k_dim
            nv = num_tokens * num_v_heads * head_v_dim
            ng = num_tokens * num_v_heads
            q = self._flat(("q", device, dtype), nk, dtype, device)[:nk].view(
                1, num_tokens, num_k_heads, head_k_dim
            )
            k = self._flat(("k", device, dtype), nk, dtype, device)[:nk].view(
                1, num_tokens, num_k_heads, head_k_dim
            )
            v = self._flat(("v", device, dtype), nv, dtype, device)[:nv].view(
                1, num_tokens, num_v_heads, head_v_dim
            )
            g = self._flat(("g", device), ng, torch.float32, device)[:ng].view(
                1, num_tokens, num_v_heads
            )
            beta = self._flat(("beta", device, beta_dtype), ng, beta_dtype, device)[:ng]
            beta = beta.view(1, num_tokens, num_v_heads)
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
        """The ``_PackedPostConv`` views of the buffers ``post_conv`` returned for the same arguments.

        Call ``post_conv`` first."""
        return self._views.get(
            (
                "packed",
                num_tokens,
                num_k_heads,
                head_k_dim,
                num_v_heads,
                head_v_dim,
                dtype,
                beta_dtype,
                device,
            )
        )


_gdn_prefill_scratch = _GdnPrefillScratch()
# TLLM_PWCG_NVTX_PROBE=1 brackets the eager GDN op body in an NVTX range (host attribution).
_GDN_NVTX_PROBE = os.environ.get("TLLM_PWCG_NVTX_PROBE", "0") == "1"


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
    # Token-major prefill conv, shared with main's Mamba2 switch (default 1).
    _token_major_conv = os.environ.get("TRTLLM_MAMBA_TOKEN_MAJOR_CONV", "1") == "1"

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

    def setup_aliases(self) -> None:
        """Move ``in_proj_qkvz.weight`` into the fused projection buffer.

        Done before any weight address is published, so the address stays
        stable through ``cache_derived_state`` and later reloads.
        """
        self._refresh_fused_in_proj()

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
        projections = (getattr(self, "in_proj_qkvz", None), getattr(self, "in_proj_ba", None))
        if not all(isinstance(lin, Linear) for lin in projections):
            return False
        for lin in projections:
            # Exact type: FP8QDQLinearMethod subclasses UnquantizedLinearMethod.
            if type(lin.quant_method) is not UnquantizedLinearMethod:
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
        return wq.dtype == wb.dtype and wq.device == wb.device and wq.shape[1] == wb.shape[1]

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
        if (
            fused is None
            or fused.shape != shape
            or fused.dtype != wq.dtype
            or fused.device != wq.device
        ):
            fused = torch.zeros(shape, dtype=wq.dtype, device=wq.device)
            self._in_proj_fused_weight = fused
        slab = fused[:rows_q]
        if wq.data.data_ptr() != slab.data_ptr() or wq.data.stride() != slab.stride():
            slab.copy_(wq.data)
            wq.data = slab
        fused[b_col : b_col + n_gate].copy_(wb.data[:n_gate])
        fused[a_col : a_col + n_gate].copy_(wb.data[n_gate:])
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
        With FlashInfer ring replay, the history lives in per-layer rings and
        each slot's ring start (cache_base) instead.
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
        state_indices = state_indices_d[:num_decodes]
        if layer_cache.cache_base is not None:
            # FlashInfer reads the history length and ring start per batch row.
            slots = state_indices.long()
            return flashinfer_cached_replay_update(
                q=query,
                k=key,
                v=value,
                a=a,
                b=b,
                A_log=self.A_log,
                dt_bias=self.dt_bias,
                ssm_states=ssm_states,
                state_indices=state_indices,
                k_ring=layer_cache.k_ring,
                u_ring=layer_cache.u_ring,
                g_ring=layer_cache.g_ring,
                hist_len=layer_cache.prev_num_accepted_tokens[slots],
                cache_base=layer_cache.cache_base[slots],
                history_size=replay_metadata.replay_history_size,
                output=output_d,
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
            state_indices=state_indices,
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

        mixed_qkv = causal_conv1d_update(
            mixed_qkv,
            conv_states,
            self.conv1d.weight,
            self.conv1d.bias,
            self.activation,
            conv_state_indices=cache_indices,
        )

        return self._recurrent_update(
            mixed_qkv, a, b, ssm_states, cache_indices, query_start_loc_long, output
        )

    def _recurrent_update(
        self,
        mixed_qkv: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
        query_start_loc_long: torch.Tensor,
        output: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
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
        # Gate convention of the resolved prefill scan (see _gdn_prefill_uses_flashinfer).
        g_linear = kwargs.get("g_linear")
        if g_linear is None:
            g_linear = _gdn_prefill_uses_flashinfer()
        # (post-conv decode rows, a, b) when they take the recurrent kernel instead of the scan.
        decode_recurrent: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None
        # FlashInfer's packed views of the scratch-backed post-conv outputs.
        packed: Optional[_PackedPostConv] = None

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

            mixed_qkv_p_t = self._prefill_conv(
                mixed_qkv_p,
                conv_states_to_use,
                has_initial_states_p,
                state_indices_p,
                query_start_loc_p,
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
                layer_cache = kwargs.get("layer_cache")
                if (
                    kwargs.get("use_replay", False)
                    and layer_cache is not None
                    and layer_cache.cache_base is not None
                ):
                    # FlashInfer ring replay needs q/k/v as slices of one buffer
                    # (strided mode), so slice the conv output like forward_decode.
                    key_size = self.key_dim // self.attn_tp_size
                    query_d = mixed_qkv_d[..., :key_size]
                    key_d = mixed_qkv_d[..., key_size : key_size * 2]
                    value_d = mixed_qkv_d[..., key_size * 2 :]
                else:
                    # Copies q/k/v into separate contiguous tensors.
                    query_d, key_d, value_d = pack_gdn_decode_qkv(
                        mixed_qkv_d,
                        self.num_k_heads_per_tp,
                        self.head_k_dim,
                        self.num_v_heads_per_tp,
                        self.head_v_dim,
                    )
            else:
                # Prefill rows only; the decode rows go through the recurrent
                # kernel after the scan (see _mixed_decode_recurrent).
                decode_recurrent = (mixed_qkv_d, a[num_prefill_tokens:], b[num_prefill_tokens:])
                query, key, value, g, beta, packed = self._post_conv(
                    mixed_qkv_p_t, None, a[:num_prefill_tokens], b[:num_prefill_tokens], g_linear
                )
        else:
            mixed_qkv_t = self._prefill_conv(
                mixed_qkv, conv_states_to_use, has_initial_states, cache_indices, query_start_loc
            )
            query, key, value, g, beta, packed = self._post_conv(mixed_qkv_t, None, a, b, g_linear)

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

        if (
            kwargs.get("scan_state_indices") is not None
            and kwargs.get("scan_cu_seqlens") is not None
        ):
            scan_state_indices = kwargs["scan_state_indices"]
            scan_cu_seqlens = kwargs["scan_cu_seqlens"]
        else:
            scan_state_indices = cache_indices
            scan_cu_seqlens = query_start_loc_long
        output_p = output
        if decode_recurrent is not None:
            # The scan covers the context rows only; its output lands in the
            # leading rows of ``output`` (context tokens are packed first).
            layout = kwargs.get("scan_prefill_layout")
            if layout is None:
                layout = _mixed_scan_prefill_layout(
                    num_prefill, scan_state_indices, scan_cu_seqlens
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
        if decode_recurrent is not None:
            mixed_qkv_d, a_d, b_d = decode_recurrent
            decode_slots = kwargs.get("decode_state_indices")
            if decode_slots is None:
                decode_slots = state_indices_d
            elif decode_slots.shape[0] != num_decodes:
                decode_slots = decode_slots[:num_decodes]
            decode_cu_seqlens_long = kwargs.get("decode_cu_seqlens_long")
            attn_out_decode = self._mixed_decode_recurrent(
                mixed_qkv_d,
                a_d,
                b_d,
                ssm_states,
                decode_slots,
                decode_cu_seqlens_long
                if decode_cu_seqlens_long is not None
                else query_start_loc_long[num_prefill:] - num_prefill_tokens,
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
        hk, dk, hv, dv = (
            self.num_k_heads_per_tp,
            self.head_k_dim,
            self.num_v_heads_per_tp,
            self.head_v_dim,
        )
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

    def _prefill_conv_input(self, mixed_qkv_p: torch.Tensor) -> torch.Tensor:
        """Transposed ``[conv_dim, num_prefill_tokens]`` conv input of the prefill rows."""
        num_tokens, width = mixed_qkv_p.shape
        out = _gdn_prefill_scratch.conv_input(
            width, num_tokens, mixed_qkv_p.dtype, mixed_qkv_p.device
        )
        return extract_transpose_prefill_slice(
            mixed_qkv_p, num_tokens, 0, width, out=out, check=False
        )

    def _prefill_conv(
        self,
        mixed_qkv_p: torch.Tensor,
        conv_states: torch.Tensor,
        has_initial_state: torch.Tensor,
        cache_indices: torch.Tensor,
        query_start_loc: torch.Tensor,
    ):
        """The prefill causal conv. Token-major (``TRTLLM_MAMBA_TOKEN_MAJOR_CONV``): the
        [tokens, conv_dim] projection slice is handed to the kernel as its transposed
        view and the result is written token-major into the scratch (no extract-transpose
        launch, no copy). Returns the ``[conv_dim, tokens]`` conv output view."""
        if self._token_major_conv and mixed_qkv_p.stride(1) == 1:
            num_tokens, width = mixed_qkv_p.shape
            out = _gdn_prefill_scratch.conv_out(
                width, num_tokens, mixed_qkv_p.dtype, mixed_qkv_p.device
            )
            return causal_conv1d_fn(
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
        x_t = self._prefill_conv_input(mixed_qkv_p)
        return causal_conv1d_fn(
            x_t,
            self.conv1d.weight,
            self.conv1d.bias,
            activation=self.activation,
            conv_states=conv_states,
            has_initial_state=has_initial_state,
            cache_indices=cache_indices,
            query_start_loc=query_start_loc,
        )

    def _post_conv(self, prefill_t: torch.Tensor, decode, a, b, g_linear: bool):
        """fused_gdn_post_conv on scratch-backed outputs (fp32 beta; the scan paths):
        ``(q, k, v, g, beta, packed)`` with ``packed`` the ``_PackedPostConv`` of
        the same buffers."""
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
            check=False,
        )
        return q, k, v, g, beta, packed

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
        # Have prefill and decode write into disjoint slices of the same buffer, avoiding
        # concatenation when eager execution mixes both kinds of requests.
        attn_out = mixed_qkv.new_empty(
            (1, mixed_qkv.shape[0], self.num_v_heads_per_tp, self.head_v_dim)
        )

        use_breakable_cuda_graph = not is_torch_compiling() and is_in_breakable_cuda_graph()
        if self.register_to_config and (is_torch_compiling() or use_breakable_cuda_graph):
            maybe_bcg_gdn_custom_op_inplace(mixed_qkv, a, b, self.layer_idx_str, attn_out)
        else:
            attn_out = self.forward_core(
                mixed_qkv,
                a,
                b,
                attn_metadata,
                mamba_metadata,
                spec_metadata=spec_metadata,
                output=attn_out,
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
        replay_metadata = (
            kv_cache_manager.get_replay_state_update_metadata() if is_target_verify else None
        )
        use_replay = replay_metadata is not None
        # Work items only feed Triton's large-batch replay + all-layer commit;
        # the FlashInfer ring kernel always folds in-kernel.
        pass_replay_work_items = (
            use_replay
            and num_decodes >= CACHED_REPLAY_PARTITION_MIN_BATCH_SIZE
            and not getattr(kv_cache_manager, "use_gdn_flashinfer_ring_replay", False)
        )
        use_cached_replay_all_layer_commit = (
            num_decodes >= CACHED_REPLAY_PARTITION_MIN_BATCH_SIZE
            and getattr(kv_cache_manager, "use_gdn_cached_replay_all_layer_commit", False)
        )
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
            scan_state_indices, scan_cu_seqlens = state_indices, query_start_loc_long
            if scan_cu_seqlens is not None:
                scan_prefill_layout = _mixed_scan_prefill_layout(
                    num_prefills, scan_state_indices, scan_cu_seqlens
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
            if pass_replay_work_items
            else None,
            "replay_n_writes": mamba_metadata.replay_n_writes if pass_replay_work_items else None,
            "use_cached_replay_all_layer_commit": use_cached_replay_all_layer_commit,
            # The main scan's layout: the context segments first, then the decode
            # requests as one-token sequences; its prefill-only prefix serves the
            # mixed-iteration scan.
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
