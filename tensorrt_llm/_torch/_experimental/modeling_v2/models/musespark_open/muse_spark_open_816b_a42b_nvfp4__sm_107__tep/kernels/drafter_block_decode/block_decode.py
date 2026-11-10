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

``wide_drafter_window_attention`` (tep4) binds a ``WideWindowAttention``
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
from ..context_kv_fork.fork import join_is_pending as _join_is_pending
from ..context_kv_fork.fork import join_pending as _join_pending

#: Operand dtypes the fused convolution takes; anything else runs the built-in.
SUPPORTED_DTYPES = (torch.bfloat16, torch.float16, torch.float32)


# ---- The drafter preamble computed by the prepare (drafter_preamble_from_commit) ---------
class DrafterPreamble:
    """The integer tensors and the residual copy ``windowed_block_decode`` would derive itself.

    Published by the verify fold's prepare (``verify_fold/fold.py``, ``_fold_preamble``)
    right before the drafter forward, taken by that forward when its inputs are these very
    tensors (identity), otherwise ignored: ``i32s`` = (``num_ctx.to(int32)``,
    that ``+ block``, ``ctx_cache_batch_idx.to(int32)`` (the slots, or the batch rows over the
    manager's block tables), ``query_positions.reshape(-1).to(int32)``),
    ``residual`` a copy of ``noise``'s bytes (or None).

    ``join`` (``drafter_preamble_late_join``): the context-K/V side chain's join that the
    prepare left to this forward, a callable, or None when the prepare joined. The side chain
    writes ``num_ctx``, ``i32s``, the arena and ``_seq_lens_cuda``; the forward's first read of
    any of them is layer 0's QK norm + RoPE (the query positions), so ``windowed_block_decode``
    runs the join right before it. A preamble the forward does not take has its join run when it
    is dropped (``_take_preamble``, ``settle_preamble``).

    ``normed`` (not installed by any composition): layer 0's input norm of ``noise``'s rows
    (``[rows, hidden]``), written by the prepare's fused gather (``ox_entry.EntryGatherNorm``,
    the drafter's own layer-0 ``input_layernorm`` weight and eps), or None.

    ``tt`` (``drafter_folds``' layer-0 table): ``(qkv, coefficients)``, layer 0's pre-RoPE
    ``qkv_proj`` output (``[rows, q + 2 kv]``) and ``prepare``'s side-1 coefficients
    (``[rows, taps, groups]``) for ``noise``'s rows, written by the prepare's table gather
    (``tt_table.drafter_l0``) as the installed layer-0 front computes them, or None.
    """

    __slots__ = (
        "num_ctx",
        "slots",
        "query_positions",
        "noise",
        "i32s",
        "residual",
        "join",
        "normed",
        "tt",
    )

    def __init__(
        self,
        num_ctx,
        slots,
        query_positions,
        noise,
        i32s,
        residual,
        join=None,
        normed=None,
        tt=None,
    ):
        self.num_ctx = num_ctx
        self.slots = slots
        self.query_positions = query_positions
        self.noise = noise
        self.i32s = i32s
        self.residual = residual
        self.join = join
        self.normed = normed
        self.tt = tt

    def run_join(self) -> None:
        """Run the deferred join once; a no-op when there is none or it has run."""
        join, self.join = self.join, None
        if join is not None:
            join()


_PUBLISHED_PREAMBLE = None


def publish_preamble(memo: DrafterPreamble) -> None:
    global _PUBLISHED_PREAMBLE
    _PUBLISHED_PREAMBLE = memo


def settle_preamble() -> None:
    """Drop a published preamble no forward will take, running its deferred join."""
    global _PUBLISHED_PREAMBLE
    memo, _PUBLISHED_PREAMBLE = _PUBLISHED_PREAMBLE, None
    if memo is not None:
        memo.run_join()


def defers_join(draft_model: nn.Module) -> bool:
    """Whether a preamble's deferred join reaches ``draft_model``'s forward
    (``drafter_preamble_late_join``).

    True when its ``dflash_forward`` is ``windowed_block_decode``.
    """
    bound = draft_model.__dict__.get("dflash_forward")
    return isinstance(bound, types.MethodType) and bound.__func__ is windowed_block_decode


def _take_preamble(
    noise_embedding, query_positions, num_ctx_per_req, ctx_cache_batch_idx, B, block_size
):
    """The published preamble when this forward's inputs are its tensors, else None; consumed either way.

    A preamble not taken has its deferred join run here.
    """
    global _PUBLISHED_PREAMBLE
    memo, _PUBLISHED_PREAMBLE = _PUBLISHED_PREAMBLE, None
    if memo is None:
        return None
    if (
        memo.noise is not noise_embedding
        or memo.query_positions is not query_positions
        or memo.num_ctx is not num_ctx_per_req
        or memo.slots is not ctx_cache_batch_idx
        or num_ctx_per_req.shape[0] != B
        or query_positions.numel() != B * block_size
    ):
        memo.run_join()
        return None
    return memo


#: Block rows on which the PDL chain's conv / merge entry triggers apply when the chain's
#: install did not record their widths (``_pdl_entry_rows`` / ``_pdl_merge_rows``,
#: ``chain.entry_rows`` / ``chain.MERGE_ENTRY_ROWS``): one request of K + 1 = 8 rows, whose
#: successor GEMMs run the ``drafter_fixed_gemms`` tactic.
_ENTRY_ROWS = (8,)


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
        if (
            side == 0
            and self.__dict__.get("_pdl_conv")
            and hidden_states.shape[0] in self.__dict__.get("_pdl_entry_rows", _ENTRY_ROWS)
        ):
            # PDL chain conv trigger (drafter_pdl_chain/chain.py): the prepare side of a block whose
            # successor GEMM runs a fixed tactic, which loads only its weight before its own wait.
            return dflash2_block_conv(
                hidden_states, delta, base_kernel, block_size, self.group_size, True
            )
        return dflash2_block_conv(hidden_states, delta, base_kernel, block_size, self.group_size)

    def prepare(
        self, hidden_states: torch.Tensor, block_size: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # The conv fold (``kernels/dr3_folds``, ``drafter_folds``): the projection weight's side-0
        # rows are permuted and the 8-row prepare runs in the projection GEMV's epilogue.
        tactic = self.__dict__.get("_dr3_cf")
        if tactic is None:
            return super().prepare(hidden_states, block_size)
        from ..dr3_folds import folds as _dr3_folds

        return _dr3_folds.folded_prepare(self, tactic, hidden_states, block_size)


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
        settle_preamble()
        _join_pending(self)
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

    preamble = _take_preamble(
        noise_embedding, query_positions, num_ctx_per_req, ctx_cache_batch_idx, B, block_size
    )
    if preamble is not None:
        # The preamble's lengths come from the folded commit: join first.
        _join_pending(self)
        cache_seqlens_i32, seq_lens_after, cache_batch_idx_i32, query_positions_flat_i32 = (
            preamble.i32s
        )
    elif _join_is_pending(self):
        # ``context_kv_late_join``: the lengths the context-K/V fork's commit wrote are
        # read right before the first block attention, after the join the prepare left
        # pending (``join_pending``).
        cache_batch_idx_i32 = ctx_cache_batch_idx.to(torch.int32)
        query_positions_flat_i32 = query_positions.reshape(-1).to(torch.int32)
        seq_lens_after = None
    else:
        cache_seqlens_i32 = num_ctx_per_req[:B].to(torch.int32)
        cache_batch_idx_i32 = ctx_cache_batch_idx.to(torch.int32)
        seq_lens_after = cache_seqlens_i32 + block_size
        query_positions_flat_i32 = query_positions.reshape(-1).to(torch.int32)
    # drafter_preamble_late_join: the side chain's join, run before layer 0's QK norm + RoPE
    # (None: already joined).
    pending_join = preamble.run_join if preamble is not None and preamble.join is not None else None

    residual = None
    has_block_conv = self.has_block_conv
    # drafter_folds' layer-0 table: layer 0's qkv and side-1 coefficient rows, written by
    # the prepare's table gather, when they fit this block (else layer 0 runs its own front).
    tt_rows = preamble.tt if preamble is not None else None
    if tt_rows is not None and not (
        has_block_conv
        and preamble.residual is not None
        and tuple(tt_rows[0].shape) == (B * block_size, q_size + 2 * kv_size)
        and tt_rows[1].shape[0] == B * block_size
    ):
        tt_rows = None
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
    merge_entry = bool(self.__dict__.get("_pdl_merge")) and B * block_size in self.__dict__.get(
        "_pdl_merge_rows", _ENTRY_ROWS
    )
    # wide_drafter_window_attention: a block decode of several requests on the wide op.
    wide = self.__dict__.get("_wide_window_attention")
    if wide is not None and not wide.takes(B * block_size):
        wide = None
    # ``drafter_attn_pdl`` (``drafter_folds``): QK norm + RoPE on the PDL copy and the split's pre-wait
    # loads (PDL only; not with the wide op, which has no pre-wait split).
    attn_pdl = self.__dict__.get("_pt_attn_pdl") if enable_pdl and wide is None else None
    # The QK fold (``kernels/dr3_folds``, not installed by any composition; on attn_pdl's path only): the window
    # attention's split normalizes and rotates the raw q / k_new instead of a QK norm + RoPE launch.
    qk_fold = self.__dict__.get("_dr3_qf") if attn_pdl is not None else None
    if qk_fold is not None:
        from ..dr3_folds import folds as _dr3_folds
    for layer_idx, layer in enumerate(layers):
        attn_mod = layer.self_attn

        if layer_idx == 0 and tt_rows is not None:
            # drafter_folds' layer-0 table: layer 0's input norm, coefficient GEMV, prepare conv and
            # qkv_proj are the table gather's rows (the installed front's bits for this block's bonus id).
            residual = preamble.residual
            qkv_query, attn_conv_coefficients = tt_rows
        else:
            hs_flat = hidden_states.reshape(-1, hidden_states.shape[-1])
            if prenormed is not None:
                # The previous layer's hop already applied this layer's input norm.
                hs_normed_flat = prenormed
            elif residual is None:
                if preamble is not None and preamble.residual is not None:
                    # The noise gather's second output, the same bytes as the clone.
                    residual = preamble.residual
                else:
                    residual = (
                        hidden_states.clone(memory_format=torch.contiguous_format)
                        if hops is not None
                        else hidden_states.clone()
                    )
                # drafter_folds' entry-trigger norm: the same norm, releasing its dependent at entry.
                entry_norm = self.__dict__.get("_ox_entry_norm")
                normed = preamble.normed if preamble is not None else None
                if normed is not None and normed.shape == hs_flat.shape:
                    # The prepare's fused gather + norm (``install_gather``, which no composition installs)
                    # applied this norm.
                    hs_normed_flat = normed
                elif entry_norm is not None and entry_norm.takes(hs_flat):
                    hs_normed_flat = entry_norm(hs_flat)
                else:
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
        if pending_join is not None:
            # drafter_preamble_late_join: the first read of the side chain's outputs (positions,
            # lengths, arena) follows.
            pending_join()
            pending_join = None
        layer_qk_fold = None
        if qk_fold is not None:
            layer_qk_fold = _dr3_folds.layer_qk_fold(
                qk_fold, layer_idx, attn_mod, query_positions_flat_i32
            )
        elif attn_pdl is not None:
            attn_pdl.qk_norm_rope(layer_idx, attn_mod, qkv_query, query_positions_flat_i32)
        else:
            attn_mod.apply_qk_norm_rope(qkv_query, query_positions_flat_i32)
        q_bshd = qkv_query[:, :q_size].reshape(B, block_size, num_heads_per_rank, head_dim)
        k_new = qkv_query[:, q_size : q_size + kv_size].reshape(-1, num_kv_heads_per_rank, head_dim)
        v_new = qkv_query[:, q_size + kv_size :].reshape(-1, num_kv_heads_per_rank, head_dim)

        if seq_lens_after is None:
            _join_pending(self)
            seq_lens_after = num_ctx_per_req[:B].to(torch.int32) + block_size
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
                attn_pdl is not None,
                qk_fold=layer_qk_fold,
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
