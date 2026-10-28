# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""This target's changes to the DFlash drafter, installed after the drafter loads.

The drafter (``llm.draft_model``, a built-in ``DFlashForCausalLM``) is built by
the built-in speculative causal LM and gets its DFlash 2 convolutions only when
its checkpoint loads, so the shell installs these from ``load_draft_weights``,
after the built-in load returns and before any forward is captured. A selected
fast path that cannot be installed on the drafter raises (``install``); every
install is idempotent, and is logged only by the call that changed something
(the shell installs from every load hook).

``drafter_block_conv``: each built-in ``DFlash2BlockConv`` is re-classed in place as
``FusedDFlash2BlockConv``, as the core's ``adopt`` does: the module tree, the
parameters and their names stay the built-in ones; only ``_convolve`` changes.

``drafter_block_decode``, first part: the VANILLA block decode calls its attention through one attribute,
``_dflash_flash_attention`` (FlashAttention-2's ``flash_attn_with_kvcache``,
set when the drafter is built), after it has appended the block's K/V to the
arena. The install replaces that attribute with ``WindowedBlockAttention``, which
sends the calls its kernel takes -- non-causal, a bounded left window -- to
``dflash_window_attention`` and every other call, argument for argument, to the
built-in function it wraps. The block decode, the arena and the append stay
the built-in ones.

``drafter_block_decode``, second part: the block decode itself is this module's
``windowed_block_decode`` -- the built-in ``dflash_forward`` written out for
the one path this drafter takes (fused QK-norm + RoPE, DFlash 2 block
convolutions, non-causal windowed layers, no output gate) over the built-in
modules -- with the block's K/V append done inside the attention op instead of
the built-in's per-layer append, and without the built-in's unused RoPE table
gather. The op reads the context where the DFlash worker keeps it: the dense
arena, or the worker's pages through its page table. This part is installed
only when every layer's attention geometry is one the op takes (it calls the op
directly; selected for a drafter outside that path, the install raises); a step
outside that path runs the built-in forward.

``wide_drafter_window_attention`` (tep4 on sm_103 only) binds a ``WideWindowAttention``
to the drafter (``kernels/wide_drafter_window_attention``); a block decode of the
rows it takes calls it (``ops/spec/dflash_window_attention_wide``, the same
attention and append on a multi-request split geometry) instead of
``dflash_window_attention``.

The fast paths compose. ``windowed_block_decode``, like the built-in forward,
reaches the block convolutions through the modules' ``prepare`` / ``finish``,
which call ``_convolve``; with ``drafter_block_conv`` installed that is the fused convolution
whichever of the two forwards runs.
"""

from __future__ import annotations

import types
from typing import Callable

import torch
from torch import nn

from tensorrt_llm._torch.models.modeling_dflash import DFlash2BlockConv, DFlashForCausalLM
from tensorrt_llm.logger import logger

from ...dispatch import LAUNCH_PDL
from ...ops.spec.dflash2_block_conv import dflash2_block_conv
from ...ops.spec.dflash_window_attention import (
    dflash_window_attention,
    geometry_unsupported_reason,
    unsupported_reason,
)

#: Operand dtypes the fused convolution takes; anything else runs the built-in.
SUPPORTED_DTYPES = (torch.bfloat16, torch.float16, torch.float32)

#: Block rows on which the PDL chain's conv / merge entry triggers apply (``chain.ROWS``):
#: one request of K + 1 = 8 rows, whose successor GEMMs run the ``drafter_fixed_gemms`` tactic.
_ENTRY_ROWS = 8


class FusedDFlash2BlockConv(DFlash2BlockConv):
    """The built-in DFlash 2 block convolution with ``_convolve`` as one kernel.

    ``prepare`` and ``finish`` are the built-in ones: the coefficient
    projection stays the built-in ``kernel_projection`` GEMM, and only the
    elementwise convolution behind it changes, to ``dflash2_block_conv``,
    which is bitwise the built-in ``dflash2_grouped_conv``. An operand the
    kernel does not take (mixed dtypes, a non-unit column stride) goes to the
    built-in convolution.
    """

    def _convolve(
        self, hidden_states: torch.Tensor, delta: torch.Tensor, side: int, block_size: int
    ) -> torch.Tensor:
        base_kernel = self.base_kernel[side]
        if (
            hidden_states.dim() != 2
            or hidden_states.stride(-1) != 1
            or not (hidden_states.dtype == delta.dtype == base_kernel.dtype)
            or hidden_states.dtype not in SUPPORTED_DTYPES
        ):
            return super()._convolve(hidden_states, delta, side, block_size)
        if side == 0 and hidden_states.shape[0] == _ENTRY_ROWS and self.__dict__.get("_pdl_conv"):
            # PDL chain conv trigger (drafter_pdl_chain/chain.py): the prepare side of an 8-row block, whose
            # successor GEMM loads only its weight before its own wait.
            return dflash2_block_conv(
                hidden_states, delta, base_kernel, block_size, self.group_size, True
            )
        return dflash2_block_conv(hidden_states, delta, base_kernel, block_size, self.group_size)


def install_fused_block_conv(draft_model: nn.Module) -> int:
    """Re-class every built-in ``DFlash2BlockConv`` of ``draft_model``; return how many.

    Idempotent: a module already re-classed is counted and left alone. A
    drafter without block convolutions (plain DFlash) has none to install.

    Raises:
        TypeError: on a block convolution that is neither the built-in class
            nor this one (a subclass whose ``_convolve`` would be replaced).
    """
    installed = 0
    for convs in (
        getattr(draft_model, "attention_convs", None),
        getattr(draft_model, "mlp_convs", None),
    ):
        for conv in convs or ():
            if type(conv) is DFlash2BlockConv:
                conv.__class__ = FusedDFlash2BlockConv
            elif type(conv) is not FusedDFlash2BlockConv:
                raise TypeError(
                    "drafter_block_conv fuses the built-in DFlash2BlockConv, got "
                    + type(conv).__name__
                )
            installed += 1
    return installed


class WindowedBlockAttention:
    """``flash_attn_with_kvcache`` for the drafter, windowed calls on the window kernel.

    Takes the keywords the block decode passes. A call with any other keyword,
    a causal mask, an unbounded left window, or operands the kernel does not
    take (``unsupported_reason``) goes to ``builtin`` unchanged.
    """

    def __init__(self, builtin: Callable[..., torch.Tensor], enable_pdl: bool) -> None:
        self.builtin = builtin
        self.enable_pdl = enable_pdl

    def __call__(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        cache_seqlens: torch.Tensor | None = None,
        cache_batch_idx: torch.Tensor | None = None,
        causal: bool = False,
        window_size: tuple[int, int] = (-1, -1),
        softmax_scale: float | None = None,
        **kwargs,
    ) -> torch.Tensor:
        window = tuple(window_size)
        if (
            kwargs
            or causal
            or cache_seqlens is None
            or cache_batch_idx is None
            or unsupported_reason(q, k_cache, v_cache, cache_seqlens, cache_batch_idx, window)
            is not None
        ):
            return self.builtin(
                q=q,
                k_cache=k_cache,
                v_cache=v_cache,
                cache_seqlens=cache_seqlens,
                cache_batch_idx=cache_batch_idx,
                causal=causal,
                window_size=window_size,
                softmax_scale=softmax_scale,
                **kwargs,
            )
        return dflash_window_attention(
            q,
            k_cache,
            v_cache,
            cache_seqlens,
            cache_batch_idx,
            window,
            softmax_scale,
            self.enable_pdl,
        )


def install_windowed_block_attention(draft_model: nn.Module, enable_pdl: bool) -> bool:
    """Wrap the drafter's VANILLA block attention; return whether it is now wrapped.

    Idempotent: an attention already wrapped is left alone (and counts). A
    drafter on another attention backend (TRTLLM, FA4) has no
    ``_dflash_flash_attention`` to wrap and is left unchanged.
    """
    attention = getattr(draft_model, "_dflash_flash_attention", None)
    if attention is None or getattr(draft_model, "dflash_attention_backend", None) != "VANILLA":
        return False
    if not isinstance(attention, WindowedBlockAttention):
        draft_model._dflash_flash_attention = WindowedBlockAttention(attention, enable_pdl)
    return True


def block_decode_unsupported_reason(draft_model: nn.Module) -> str | None:
    """Why ``windowed_block_decode`` cannot stand in for this drafter's forward, or None.

    Besides the class checks, every layer's attention geometry -- a block of
    ``draft_model.block_size`` rows, its query / KV head counts and head width,
    its window, bf16 operands -- must be one the attention op takes:
    ``windowed_block_decode`` calls the op directly, with no per-call fallback
    to the built-in attention.
    """
    if not isinstance(draft_model, DFlashForCausalLM):
        return f"not a DFlashForCausalLM ({type(draft_model).__name__})"
    if type(draft_model).dflash_forward is not DFlashForCausalLM.dflash_forward:
        return f"{type(draft_model).__name__} overrides dflash_forward"
    if type(draft_model)._post_attention_gate is not DFlashForCausalLM._post_attention_gate:
        return f"{type(draft_model).__name__} gates the attention output"
    if not getattr(draft_model, "_uses_worker_attention_backend", False):
        return "the drafter brings its own attention"
    block_size = getattr(draft_model, "block_size", None)
    if not block_size:
        return "the drafter declares no block size"
    for layer_idx, layer in enumerate(draft_model.model.layers):
        causal, window = draft_model._resolve_block_attention(layer_idx)
        if causal or window[0] < 0:
            return f"layer {layer_idx} is causal or unwindowed"
        attn = layer.self_attn
        reason = geometry_unsupported_reason(
            block_size,
            attn.num_heads,
            attn.num_key_value_heads,
            attn.head_dim,
            torch.bfloat16,
            tuple(window),
        )
        if reason is not None:
            return f"layer {layer_idx}: {reason}"
    return None


def windowed_block_decode(
    self: DFlashForCausalLM,
    noise_embedding: torch.Tensor,
    query_positions: torch.Tensor,
    num_ctx_per_req: torch.Tensor,
    ctx_k_cache: torch.Tensor,
    ctx_v_cache: torch.Tensor,
    ctx_cache_batch_idx: torch.Tensor,
    ctx_kv_cache: torch.Tensor | None = None,
    ctx_page_table: torch.Tensor | None = None,
) -> torch.Tensor:
    """The built-in ``DFlashForCausalLM.dflash_forward``, fused-RoPE path, append fused.

    Bound to the drafter as ``dflash_forward`` by ``install_block_decode``.
    Same modules, same order, same context state afterwards; the per-layer
    append of the block K/V and the block attention are one
    ``dflash_window_attention`` call with ``k_new`` / ``v_new``, over the dense
    arena or, given ``ctx_page_table``, the pages of ``ctx_kv_cache``. Steps the
    written-out path does not cover (fp16/fp32 activations, a drafter without
    the fused QK-norm + RoPE, a block wider than the one ``install_block_decode``
    checked the attention geometry for) run the built-in forward.
    """
    if self._fused_kv_weight is None:
        self._build_fused_kv_buffers()
    attn0 = self.model.layers[0].self_attn
    if ctx_page_table is None:
        caches = [(ctx_k_cache[:, i], ctx_v_cache[:, i]) for i in range(ctx_k_cache.shape[1])]
    else:
        # Each layer's [pages, K/V, Hkv, page, D] pool as two pools of [page, Hkv, D] rows.
        caches = [(pool[:, 0].transpose(1, 2), pool[:, 1].transpose(1, 2)) for pool in ctx_kv_cache]
    k0, v0 = caches[0]
    if not (
        self._use_fused_qk_norm_rope
        and noise_embedding.dtype == k0.dtype == v0.dtype == torch.bfloat16
        and k0.stride(-1) == 1
        and v0.stride(-1) == 1
        and noise_embedding.shape[1] <= self.block_size
    ):
        return DFlashForCausalLM.dflash_forward(
            self,
            noise_embedding,
            query_positions,
            num_ctx_per_req,
            ctx_k_cache,
            ctx_v_cache,
            ctx_cache_batch_idx,
            ctx_kv_cache,
            ctx_page_table,
        )
    # The built-in forward initializes RoPE through its (here unused) table gather.
    if not self._rope_initialized:
        self._init_rope()
    enable_pdl = self._block_decode_enable_pdl

    q_size = attn0.q_size
    kv_size = attn0.kv_size
    head_dim = attn0.head_dim
    num_heads_per_rank = attn0.num_heads
    num_kv_heads_per_rank = attn0.num_key_value_heads

    B = noise_embedding.shape[0]
    block_size = noise_embedding.shape[1]
    hidden_states = noise_embedding

    cache_seqlens_i32 = num_ctx_per_req[:B].to(torch.int32)
    cache_batch_idx_i32 = ctx_cache_batch_idx.to(torch.int32)
    seq_lens_after = cache_seqlens_i32 + block_size
    query_positions_flat_i32 = query_positions.reshape(-1).to(torch.int32)

    residual = None
    has_block_conv = self.has_block_conv
    # ``drafter_hops``: every hop site of this forward on the fused drafter hop, or none.
    hops = self.__dict__.get("_drafter_hop")
    if hops is not None and not (
        has_block_conv and hops.takes(B * block_size, block_size, noise_embedding.dtype)
    ):
        hops = None
    if hops is not None:
        hops.begin_forward()
    layers = self.model.layers
    prenormed = None
    # PDL chain merge trigger (drafter_pdl_chain/chain.py): the merge releases o_proj's GEMM at its entry.
    merge_entry = bool(self.__dict__.get("_pdl_merge")) and B * block_size == _ENTRY_ROWS
    # wide_drafter_window_attention: a block decode of several requests on the wide op.
    wide = self.__dict__.get("_wide_window_attention")
    if wide is not None and not wide.takes(B * block_size):
        wide = None
    for layer_idx, layer in enumerate(layers):
        attn_mod = layer.self_attn

        hs_flat = hidden_states.reshape(-1, hidden_states.shape[-1])
        if prenormed is not None:
            # The previous layer's hop already applied this layer's input norm.
            hs_normed_flat = prenormed
        elif residual is None:
            residual = (
                hidden_states.clone(memory_format=torch.contiguous_format)
                if hops is not None
                else hidden_states.clone()
            )
            hs_normed_flat = layer.input_layernorm(hs_flat)
        else:
            res_flat = residual.reshape(-1, residual.shape[-1])
            hs_normed_flat, res_flat = layer.input_layernorm(hs_flat, res_flat)
            residual = res_flat.reshape(B, block_size, -1)

        attn_conv_coefficients = None
        if has_block_conv:
            hs_normed_flat, attn_conv_coefficients = self.attention_convs[layer_idx].prepare(
                hs_normed_flat, block_size
            )

        qkv_query = attn_mod.qkv_proj(hs_normed_flat)
        attn_mod.apply_qk_norm_rope(qkv_query, query_positions_flat_i32)
        q_bshd = qkv_query[:, :q_size].reshape(B, block_size, num_heads_per_rank, head_dim)
        k_new = qkv_query[:, q_size : q_size + kv_size].reshape(-1, num_kv_heads_per_rank, head_dim)
        v_new = qkv_query[:, q_size + kv_size :].reshape(-1, num_kv_heads_per_rank, head_dim)

        _, window_size = self._resolve_block_attention(layer_idx)
        k_cache, v_cache = caches[layer_idx]
        if wide is not None:
            out = wide(
                q_bshd,
                k_cache,
                v_cache,
                seq_lens_after,
                cache_batch_idx_i32,
                window_size,
                enable_pdl,
                k_new,
                v_new,
                ctx_page_table,
            )
        else:
            out = dflash_window_attention(
                q_bshd,
                k_cache,
                v_cache,
                seq_lens_after,
                cache_batch_idx_i32,
                window_size,
                None,
                enable_pdl,
                k_new,
                v_new,
                merge_entry,
                ctx_page_table,
            )
        attn_output = out.reshape(B * block_size, q_size)

        res_flat = residual.reshape(-1, residual.shape[-1])
        if hops is not None:
            # o_proj, its all-reduce, the attention finish convolution and
            # post_attention_layernorm in one hop; the residual is updated in place.
            hidden_out = hops.attention(
                layer_idx,
                attn_mod.o_proj,
                attn_output,
                res_flat,
                attn_conv_coefficients,
                self.attention_convs[layer_idx],
                layer.post_attention_layernorm,
                block_size,
            )
        else:
            hidden_out = attn_mod.o_proj(attn_output)
            if has_block_conv:
                hidden_out = self.attention_convs[layer_idx].finish(
                    hidden_out, attn_conv_coefficients, block_size
                )
            hidden_out, res_flat = layer.post_attention_layernorm(hidden_out, res_flat)
        if has_block_conv:
            hidden_out, mlp_conv_coefficients = self.mlp_convs[layer_idx].prepare(
                hidden_out, block_size
            )
        if hops is not None:
            # The MLP, its all-reduce, the MLP finish convolution and the next
            # norm (the next layer's input_layernorm, or model.norm) in one hop.
            next_norm = (
                layers[layer_idx + 1].input_layernorm
                if layer_idx + 1 < len(layers)
                else self.model.norm
            )
            prenormed = hops.mlp(
                layer_idx,
                layer.mlp,
                hidden_out,
                res_flat,
                mlp_conv_coefficients,
                self.mlp_convs[layer_idx],
                next_norm,
                block_size,
            )
            continue
        hidden_out = layer.mlp(hidden_out)
        if has_block_conv:
            hidden_out = self.mlp_convs[layer_idx].finish(
                hidden_out, mlp_conv_coefficients, block_size
            )

        hidden_states = hidden_out.reshape(B, block_size, -1)
        residual = res_flat.reshape(B, block_size, -1)

    if hops is not None:
        return prenormed
    hidden_states_out, _ = self.model.norm(
        hidden_states.reshape(-1, hidden_states.shape[-1]),
        residual.reshape(-1, residual.shape[-1]),
    )
    return hidden_states_out


def install_block_decode(draft_model: nn.Module, enable_pdl: bool) -> str | None:
    """Bind ``windowed_block_decode`` as the drafter's ``dflash_forward``.

    Returns None when installed (or already installed), else why not
    (``block_decode_unsupported_reason``, which includes the attention op's
    geometry check on every layer); the drafter is then left with its built-in
    forward. Idempotent.
    """
    reason = block_decode_unsupported_reason(draft_model)
    if reason is not None:
        return reason
    draft_model._block_decode_enable_pdl = enable_pdl
    bound = draft_model.__dict__.get("dflash_forward")
    if not (isinstance(bound, types.MethodType) and bound.__func__ is windowed_block_decode):
        draft_model.dflash_forward = types.MethodType(windowed_block_decode, draft_model)
    return None


def install(draft_model: nn.Module | None, block_conv: bool, block_decode: bool) -> None:
    """Install the fused block convolution and the windowed block decode on ``draft_model``.

    Each installs independently of the other; the order does not matter
    (``block_conv`` re-classes the convolution modules, ``block_decode`` rebinds
    the block attention and the block decode). Each is logged only by the call
    that changed something: the shell installs from every load hook, and the
    repeated calls find them in place.

    Raises:
        ValueError: if one asked for cannot be installed on this drafter
            (``block_conv``: no DFlash 2 block convolutions; ``block_decode``:
            ``block_decode_unsupported_reason``).
        TypeError: from ``install_fused_block_conv``.
    """
    if block_conv:
        _install_fused_block_conv(draft_model)
    if block_decode:
        _install_windowed_attention(draft_model, True)


def _builtin_block_convs(draft_model: nn.Module) -> int:
    """How many of ``draft_model``'s block convolutions are still the built-in class."""
    return sum(
        type(conv) is DFlash2BlockConv
        for convs in (
            getattr(draft_model, "attention_convs", None),
            getattr(draft_model, "mlp_convs", None),
        )
        for conv in convs or ()
    )


def _install_fused_block_conv(draft_model: nn.Module | None) -> None:
    fresh = 0 if draft_model is None else _builtin_block_convs(draft_model)
    count = 0 if draft_model is None else install_fused_block_conv(draft_model)
    if not count:
        raise ValueError(
            "[musespark_open] drafter_block_conv cannot run here: the drafter has no DFlash 2 block convolutions"
        )
    if fresh:
        logger.info(
            f"[musespark_open] drafter_block_conv: fused DFlash 2 block convolution on {count} modules"
        )


def _install_windowed_attention(draft_model: nn.Module | None, block_decode: bool) -> None:
    enable_pdl = LAUNCH_PDL
    wrapped = isinstance(
        getattr(draft_model, "_dflash_flash_attention", None), WindowedBlockAttention
    )
    if (
        draft_model is not None
        and install_windowed_block_attention(draft_model, enable_pdl)
        and not wrapped
    ):
        logger.info(
            f"[musespark_open] drafter_block_decode: windowed DFlash block attention on the drafter (pdl={enable_pdl})"
        )
    if block_decode:
        bound = draft_model.__dict__.get("dflash_forward")
        decoding = isinstance(bound, types.MethodType) and bound.__func__ is windowed_block_decode
        reason = install_block_decode(draft_model, enable_pdl)
        if reason is not None:
            raise ValueError(f"[musespark_open] drafter_block_decode cannot run here: {reason}")
        if not decoding:
            logger.info(
                "[musespark_open] drafter_block_decode: DFlash block decode with the K/V append fused"
            )
