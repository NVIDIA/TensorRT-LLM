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

Three levers refine them, each resolved by the plan and passed to ``install``:

* ``drafter_conv_epilogue`` (on ``drafter_block_conv`` and ``drafter_fixed_gemms``): on blocks of
  9..32 rows whose coefficient projection runs a ``drafter_fixed_gemms`` tactic, ``prepare`` is one
  launch, the coefficient GEMM with the input-side convolution in its epilogue
  (``ops/gemm/conv_coef_gemm.py``, ``ConvEpilogue``) over a copy of the projection weight with
  side 0's two taps per group on adjacent rows (32 MiB per convolution), bitwise the fixed-tactic
  GEMM and ``dflash2_block_conv``.
* ``drafter_finish_norm`` (on ``drafter_block_decode``): on blocks the drafter
  hops do not take, each hop's finish convolution and the next norm's fused residual add + RMSNorm
  are one launch (``ops/spec/drafter_finish_norm.py``, ``FinishNorm``), bitwise the two.
* ``drafter_silu_epilogue`` (on ``drafter_block_decode``): on blocks of 9..32
  rows whose ``gate_up_proj`` runs a ``drafter_fixed_gemms`` tactic, the MLP's ``gate_up_proj`` and
  ``silu(gate) * up`` are one launch (``ops/gemm/gate_up_silu_gemm.py``, ``SiluEpilogue``) over a
  copy of the gate_up weight with gate and up rows interleaved (112 MiB per layer), bitwise the
  fixed-tactic GEMM and the built-in ``silu_and_mul``.

The fast paths compose. ``windowed_block_decode``, like the built-in forward,
reaches the block convolutions through the modules' ``prepare`` / ``finish``,
which call ``_convolve``; with ``drafter_block_conv`` installed that is the fused convolution
whichever of the two forwards runs.
"""

from __future__ import annotations

import types
from typing import Callable, Optional

import torch
from torch import nn

from tensorrt_llm._torch.models.modeling_dflash import DFlash2BlockConv, DFlashForCausalLM
from tensorrt_llm._torch.modules.rms_norm import RMSNorm
from tensorrt_llm.logger import logger

from ...dispatch import LAUNCH_PDL
from ...ops.spec.dflash2_block_conv import dflash2_block_conv
from ...ops.spec.dflash_window_attention import (
    dflash_window_attention,
    geometry_unsupported_reason,
    unsupported_reason,
)
from ...ops.spec.drafter_finish_norm import drafter_finish_norm
from ...ops.spec.drafter_finish_norm import takes as _finish_norm_takes

#: Operand dtypes the fused convolution takes; anything else runs the built-in.
SUPPORTED_DTYPES = (torch.bfloat16, torch.float16, torch.float32)


#: Block rows on which the PDL chain's conv / merge entry triggers apply when the chain's
#: install did not record their widths (``_pdl_entry_rows`` / ``_pdl_merge_rows``,
#: ``chain.entry_rows`` / ``chain.MERGE_ENTRY_ROWS_BY_SM``): one request of K + 1 = 8 rows, whose
#: successor GEMMs run the ``drafter_fixed_gemms`` tactic.
_ENTRY_ROWS = (8,)


#: Block rows ``SiluEpilogue`` serves: two to four requests' blocks (the one-request block keeps
#: the PDL chain's silu trigger).
SILU_EPILOGUE_ROWS = (9, 32)
#: Block rows ``ConvEpilogue`` serves: two to four requests' blocks (the one-request block keeps
#: the PDL chain's convolution trigger).
CONV_EPILOGUE_ROWS = (9, 32)


class ConvEpilogue:
    """One convolution's ``prepare`` as the coefficient GEMM with the convolution in its epilogue.

    Holds the ``interleave``d copy of the module's ``kernel_projection`` weight. ``prepare``
    returns None for a call it does not serve: rows outside ``CONV_EPILOGUE_ROWS``, operands the
    kernel does not take, or a projection not on a ``drafter_fixed_gemms`` tactic at that row
    count (whose bits the kernel reproduces); the module then runs the built-in ``prepare``.
    """

    logged = False

    def __init__(self, weight: torch.Tensor) -> None:
        from ...ops.gemm import conv_coef_gemm as _gemm

        self.weight = _gemm.interleave(weight.detach())
        self.source = weight

    def prepare(self, conv: "FusedDFlash2BlockConv", x: torch.Tensor, block_size: int):
        rows = x.shape[0] if x.dim() == 2 else 0
        if not CONV_EPILOGUE_ROWS[0] <= rows <= CONV_EPILOGUE_ROWS[1]:
            return None
        proj = conv.kernel_projection
        tactics = getattr(proj, "_fixed_gemm_tactics", None)
        if (
            tactics is None
            or proj.weight is not self.source
            or x.dtype != torch.bfloat16
            or not x.is_contiguous()
            or x.data_ptr() % 32
            or torch.is_grad_enabled()
        ):
            return None
        from ...ops.gemm import batch_splitk_gemm as _bsk
        from ...ops.gemm import conv_coef_gemm as _gemm
        from ...ops.gemm.fixed_low_m_gemm import tactic_for

        tactic = tactic_for(rows, x.shape[1], self.weight.shape[0], tactics)
        if tactic is None:
            return None
        conv_out, coefficients = _gemm.run(
            x,
            self.weight,
            conv.base_kernel[0],
            block_size,
            _bsk.BatchSplitKTactic(*tactic),
            bool(getattr(proj, "_fixed_gemm_pdl", True)),
        )
        if not ConvEpilogue.logged:
            ConvEpilogue.logged = True
            logger.info(
                f"[musespark_open] drafter_conv_epilogue: the first fused prepare ran ({rows} rows, "
                f"split-K tactic {tuple(tactic)}): coefficient GEMM + input-side convolution in one launch"
            )
        return conv_out, coefficients.view(rows, 2, conv.taps, conv.num_groups)[:, 1]


def _conv_epilogue_reason(conv: nn.Module) -> Optional[str]:
    weight = conv.kernel_projection.weight
    if (
        conv.taps != 2
        or conv.group_size != 16
        or tuple(weight.shape) != (2048, 8192)
        or weight.dtype != torch.bfloat16
        or conv.base_kernel.dtype != torch.bfloat16
        or conv.kernel_projection.bias is not None
    ):
        return "not a two-tap, group-16, BF16 [2048, 8192] coefficient projection"
    return None


class FinishNorm:
    """A hop's finish convolution and the next norm's fused add + RMSNorm in one launch.

    ``__call__`` returns the normed rows and updates ``residual`` in place, as the built-in
    ``finish`` then ``norm(hidden, residual)`` do; operands the fused op does not take run those
    two.
    """

    logged = False

    def __init__(self, enable_pdl: bool, entry_trigger: bool = True) -> None:
        self.enable_pdl = enable_pdl
        #: Release the successor at entry (its next coefficient GEMM streams its weight under this
        #: kernel), as the fused drafter hop releases the same successors early; False: after the
        #: stores. Every successor reads this kernel's output and residual after its own wait.
        self.entry_trigger = entry_trigger

    def __call__(
        self,
        hidden: torch.Tensor,
        coefficients: torch.Tensor,
        conv: "FusedDFlash2BlockConv",
        norm: nn.Module,
        residual: torch.Tensor,
        block_size: int,
    ) -> torch.Tensor:
        base = conv.base_kernel[1]
        if not _finish_norm_takes(hidden, coefficients, base, residual, norm.weight):
            hidden = conv.finish(hidden, coefficients, block_size)
            normed, _ = norm(hidden, residual)
            return normed
        normed = drafter_finish_norm(
            hidden,
            coefficients,
            base,
            residual,
            norm.weight,
            norm.variance_epsilon,
            block_size,
            self.enable_pdl,
            self.entry_trigger,
        )
        if not FinishNorm.logged:
            FinishNorm.logged = True
            logger.info(
                f"[musespark_open] drafter_finish_norm: the first fused hop tail ran ({hidden.shape[0]} "
                "rows): finish convolution + residual add + RMSNorm in one launch"
            )
        return normed


def _finish_norm_reason(draft_model: nn.Module) -> Optional[str]:
    """Why ``FinishNorm`` cannot stand in for this drafter's finish + norm pairs, or None."""
    if not getattr(draft_model, "has_block_conv", False):
        return "the drafter has no DFlash 2 block convolutions"
    layers = draft_model.model.layers
    norms = [("model.norm", draft_model.model.norm)]
    for index, layer in enumerate(layers):
        norms.append((f"layers.{index}.input_layernorm", layer.input_layernorm))
        norms.append((f"layers.{index}.post_attention_layernorm", layer.post_attention_layernorm))
    for where, norm in norms:
        if type(norm) is not RMSNorm:
            return f"{where} is {type(norm).__name__}, not the built-in RMSNorm"
        if norm.use_gemma or norm.use_cuda_tile or norm.return_hp_output:
            return f"{where} is not the plain flashinfer RMSNorm"
        if norm.is_nvfp4 and norm.nvfp4_scale is not None:
            return f"{where} quantizes its output"
        weight = getattr(norm, "weight", None)
        if weight is None or weight.dtype != torch.bfloat16 or tuple(weight.shape) != (8192,):
            return f"{where} has no BF16 weight of 8192"
    for kind in ("attention_convs", "mlp_convs"):
        for index, conv in enumerate(getattr(draft_model, kind, None) or ()):
            if (
                conv.taps != 2
                or conv.group_size != 16
                or tuple(conv.base_kernel.shape) != (2, 2, 8192)
                or conv.base_kernel.dtype != torch.bfloat16
            ):
                return f"{kind}.{index} is not a two-tap, group-16, BF16 convolution over 8192"
            if type(conv).finish is not DFlash2BlockConv.finish:
                return f"{kind}.{index} overrides finish"
    return None


class SiluEpilogue:
    """One drafter MLP with ``gate_up_proj`` and ``silu(gate) * up`` as one launch.

    Holds the ``interleave``d copy of the MLP's ``gate_up_proj`` weight. Called as the MLP is
    (``x``, ``final_all_reduce_params``); a call it does not serve -- rows outside
    ``SILU_EPILOGUE_ROWS``, operands the kernel does not take, a ``gate_up_proj`` not on a
    ``drafter_fixed_gemms`` tactic at that row count (whose bits the kernel reproduces) or no longer
    holding the copied weight -- runs the MLP itself.
    """

    logged = False

    def __init__(self, mlp: nn.Module) -> None:
        from ...ops.gemm import gate_up_silu_gemm as _gemm

        weight = mlp.gate_up_proj.weight
        self.mlp = mlp
        self.weight = _gemm.interleave(weight.detach())
        self.source = weight
        self._errors: dict = {}

    def _tactic(self, rows: int, k: int):
        """The ``drafter_fixed_gemms`` tactic ``gate_up_proj`` runs at ``rows``, if the fused kernel
        takes it, else None (cached per row count and tactic)."""
        from ...spec_worker import FixedTacticLinearMethod

        proj = self.mlp.gate_up_proj
        method = proj.quant_method
        if type(method) is not FixedTacticLinearMethod or proj.weight is not self.source:
            return None
        if getattr(proj, "use_custom_cublas_mm", False):
            return None
        tactic = method.tactic_for(rows, k, self.weight.shape[0])
        if tactic is None:
            return None
        key = (rows, k, tuple(tactic))
        error = self._errors.get(key, False)
        if error is False:
            from ...ops.gemm import batch_splitk_gemm as _bsk
            from ...ops.gemm import gate_up_silu_gemm as _gemm

            error = _gemm.tactic_error(
                _bsk.BatchSplitKTactic(*tactic), rows, self.weight.shape[0], k
            )
            self._errors[key] = error
        return None if error is not None else (tactic, method.enable_pdl)

    def __call__(self, x: torch.Tensor, final_all_reduce_params=None) -> torch.Tensor:
        rows = x.shape[0] if x.dim() == 2 else 0
        picked = None
        if (
            SILU_EPILOGUE_ROWS[0] <= rows <= SILU_EPILOGUE_ROWS[1]
            and x.dtype == torch.bfloat16
            and x.is_cuda
            and x.is_contiguous()
            and x.data_ptr() % 32 == 0
            and not torch.is_grad_enabled()
        ):
            picked = self._tactic(rows, x.shape[1])
        if picked is None:
            return self.mlp(x, final_all_reduce_params=final_all_reduce_params)
        from ...ops.gemm import batch_splitk_gemm as _bsk
        from ...ops.gemm import gate_up_silu_gemm as _gemm

        tactic, enable_pdl = picked
        activated = _gemm.run(x, self.weight, _bsk.BatchSplitKTactic(*tactic), enable_pdl)
        if not SiluEpilogue.logged:
            SiluEpilogue.logged = True
            logger.info(
                f"[musespark_open] drafter_silu_epilogue: the first fused gate_up ran ({rows} rows, "
                f"split-K tactic {tuple(tactic)}): gate_up GEMM + silu(gate) * up in one launch"
            )
        mlp = self.mlp
        return mlp.down_proj(
            activated, all_reduce_params=final_all_reduce_params, layer_idx=mlp.layer_idx
        )


def _silu_epilogue_reason(draft_model: nn.Module) -> Optional[str]:
    """Why ``SiluEpilogue`` cannot stand in for this drafter's MLPs, or None.

    The MLP must run exactly ``down_proj(silu_and_mul(gate_up_proj(x)))``: the built-in
    ``GatedMLP`` (or the PDL chain's re-class of it), a plain SwiGLU on a fused, bias-free BF16
    ``gate_up_proj`` and a BF16 ``down_proj`` input.
    """
    from tensorrt_llm._torch.modules.gated_mlp import GatedMLP

    for index, layer in enumerate(draft_model.model.layers):
        mlp = layer.mlp
        where = f"layers.{index}.mlp"
        if not isinstance(mlp, GatedMLP) or type(mlp).forward is not GatedMLP.forward:
            return f"{where} is {type(mlp).__name__}, not the built-in GatedMLP"
        if (
            mlp.activation is not torch.nn.functional.silu
            or not mlp._is_plain_swiglu()
            or mlp.swiglu_limit
        ):
            return f"{where} is not a plain SwiGLU"
        if mlp.split_gate_up:
            return f"{where} splits its gate/up projection"
        if mlp._can_fuse_gate_up_swiglu() or mlp._can_fuse_swiglu_fp8_quant():
            return f"{where} fuses its activation into a GEMM"
        down = mlp.down_proj
        if getattr(down, "has_fp8_qdq", False) or getattr(down, "has_w4a8_nvfp4_fp8", False):
            return f"{where} quantizes the activation for down_proj"
        weight = getattr(mlp.gate_up_proj, "weight", None)
        if (
            weight is None
            or weight.dtype != torch.bfloat16
            or weight.dim() != 2
            or weight.shape[0] % 16
            or mlp.gate_up_proj.bias is not None
        ):
            return f"{where}.gate_up_proj is not a bias-free BF16 [2d, K] projection"
    return None


def install_silu_epilogue(draft_model: nn.Module) -> int:
    """Give every drafter MLP its ``SiluEpilogue``; return how many are new.

    Idempotent (an MLP whose epilogue holds the current gate_up weight is left alone).

    Raises:
        ValueError: on an MLP the epilogue cannot stand in for (``_silu_epilogue_reason``), on
            every rank alike.
    """
    reason = _silu_epilogue_reason(draft_model)
    if reason is not None:
        raise ValueError(f"[musespark_open] drafter_silu_epilogue cannot run here: {reason}")
    added, total_bytes = 0, 0
    for layer in draft_model.model.layers:
        mlp = layer.mlp
        fused = mlp.__dict__.get("_silu_epilogue")
        if fused is None or fused.source is not mlp.gate_up_proj.weight:
            mlp._silu_epilogue = SiluEpilogue(mlp)
            added += 1
            total_bytes += mlp._silu_epilogue.weight.numel() * 2
    if added:
        logger.info(
            f"[musespark_open] drafter_silu_epilogue: drafter blocks of {SILU_EPILOGUE_ROWS[0]}.."
            f"{SILU_EPILOGUE_ROWS[1]} rows run each MLP's gate_up_proj + silu(gate) * up as one launch "
            f"on {len(draft_model.model.layers)} layers ({total_bytes / 2**20:.0f} MiB of interleaved "
            "gate_up weights)"
        )
    return added


class FusedDFlash2BlockConv(DFlash2BlockConv):
    """The built-in DFlash 2 block convolution with ``_convolve`` as one kernel.

    ``prepare`` and ``finish`` are the built-in ones: the coefficient
    projection stays the built-in ``kernel_projection`` GEMM, and only the
    elementwise convolution behind it changes, to ``dflash2_block_conv``,
    which is bitwise the built-in ``dflash2_grouped_conv``. An operand the
    kernel does not take (mixed dtypes, a non-unit column stride) goes to the
    built-in convolution.
    """

    def prepare(
        self, hidden_states: torch.Tensor, block_size: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The built-in ``prepare``, or ``ConvEpilogue``'s one launch where it serves the call."""
        fused = self.__dict__.get("_conv_epilogue")
        if fused is not None:
            prepared = fused.prepare(self, hidden_states, block_size)
            if prepared is not None:
                return prepared
        return super().prepare(hidden_states, block_size)

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
    merge_entry = bool(self.__dict__.get("_pdl_merge")) and B * block_size in self.__dict__.get(
        "_pdl_merge_rows", _ENTRY_ROWS
    )
    # ``drafter_finish_norm``: each hop's finish convolution + next norm in one launch on
    # blocks the drafter hops do not take.
    finish_norm = self.__dict__.get("_finish_norm")
    if finish_norm is not None and (
        hops is not None or not has_block_conv or noise_embedding.dtype != torch.bfloat16
    ):
        finish_norm = None
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
        elif finish_norm is not None:
            # The finish convolution and post_attention_layernorm in one launch; the residual is
            # updated in place.
            hidden_out = finish_norm(
                attn_mod.o_proj(attn_output),
                attn_conv_coefficients,
                self.attention_convs[layer_idx],
                layer.post_attention_layernorm,
                res_flat,
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
        # ``drafter_silu_epilogue``: gate_up_proj + silu(gate) * up in one launch.
        mlp = layer.mlp.__dict__.get("_silu_epilogue") or layer.mlp
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
                mlp,
                hidden_out,
                res_flat,
                mlp_conv_coefficients,
                self.mlp_convs[layer_idx],
                next_norm,
                block_size,
            )
            continue
        hidden_out = mlp(hidden_out)
        if finish_norm is not None:
            # The MLP finish convolution and the next norm (the next layer's input_layernorm,
            # or model.norm) in one launch, as the hop does; the residual is updated in place.
            next_norm = (
                layers[layer_idx + 1].input_layernorm
                if layer_idx + 1 < len(layers)
                else self.model.norm
            )
            prenormed = finish_norm(
                hidden_out,
                mlp_conv_coefficients,
                self.mlp_convs[layer_idx],
                next_norm,
                res_flat,
                block_size,
            )
            residual = res_flat.reshape(B, block_size, -1)
            continue
        if has_block_conv:
            hidden_out = self.mlp_convs[layer_idx].finish(
                hidden_out, mlp_conv_coefficients, block_size
            )

        hidden_states = hidden_out.reshape(B, block_size, -1)
        residual = res_flat.reshape(B, block_size, -1)

    if hops is not None or finish_norm is not None:
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


def install(
    draft_model: nn.Module | None,
    block_conv: bool,
    block_decode: bool,
    *,
    conv_epilogue: bool = False,
    finish_norm: bool = False,
    silu_epilogue: bool = False,
) -> None:
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
        _install_fused_block_conv(draft_model, conv_epilogue)
    if block_decode:
        _install_windowed_attention(draft_model, True, finish_norm, silu_epilogue)


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


def _install_fused_block_conv(draft_model: nn.Module | None, conv_epilogue: bool) -> None:
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
    if conv_epilogue:
        install_conv_epilogue(draft_model)


def install_conv_epilogue(draft_model: nn.Module) -> int:
    """Give every fused convolution of ``draft_model`` its ``ConvEpilogue``; return how many are new.

    Idempotent (a convolution whose epilogue holds the current projection weight is left alone);
    logged by the call that added any.

    Raises:
        ValueError: on a convolution the epilogue cannot take (``_conv_epilogue_reason``), on every
            rank alike.
    """
    convs = [
        (f"{kind}.{index}", conv)
        for kind in ("attention_convs", "mlp_convs")
        for index, conv in enumerate(getattr(draft_model, kind, None) or ())
    ]
    for where, conv in convs:
        reason = _conv_epilogue_reason(conv)
        if reason is not None:
            raise ValueError(
                f"[musespark_open] drafter_conv_epilogue cannot run here: {where} is {reason}"
            )
    added = 0
    for _, conv in convs:
        fused = conv.__dict__.get("_conv_epilogue")
        if fused is None or fused.source is not conv.kernel_projection.weight:
            conv._conv_epilogue = ConvEpilogue(conv.kernel_projection.weight)
            added += 1
    if added:
        logger.info(
            f"[musespark_open] drafter_conv_epilogue: drafter blocks of {CONV_EPILOGUE_ROWS[0]}.."
            f"{CONV_EPILOGUE_ROWS[1]} rows run each convolution's prepare as one launch (coefficient "
            f"GEMM on its fixed split-K tactic + input-side convolution) on {len(convs)} modules "
            f"({added * 2048 * 8192 * 2 / 2**20:.0f} MiB of tap-interleaved projection weights)"
        )
    return added


def _install_windowed_attention(
    draft_model: nn.Module | None, block_decode: bool, finish_norm: bool, silu_epilogue: bool
) -> None:
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
        if finish_norm:
            reason = _finish_norm_reason(draft_model)
            if reason is not None:
                raise ValueError(f"[musespark_open] drafter_finish_norm cannot run here: {reason}")
            if not isinstance(draft_model.__dict__.get("_finish_norm"), FinishNorm):
                draft_model._finish_norm = FinishNorm(enable_pdl)
                logger.info(
                    "[musespark_open] drafter_finish_norm: drafter blocks the drafter hops do not "
                    "take run each hop's finish convolution + residual add + RMSNorm as one launch "
                    f"(pdl={enable_pdl}, dependents released at entry)"
                )
        if silu_epilogue:
            install_silu_epilogue(draft_model)
