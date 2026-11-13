# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""drafter_attn_pdl: the drafter's QK norm + RoPE and window attention as programmatic dependents.

In the block decode (``kernels/drafter_block_decode``) each drafter layer runs ``qkv_proj`` (an 8-row split-K
GEMV launched with PDL), ``apply_qk_norm_rope`` (``torch.ops.trtllm.fused_qk_norm_rope``, a plain launch) and
``dflash_window_attention`` (a split kernel that waits at its entry, then the merge). The plain launch is a
full dependency on each side: the QK norm + RoPE starts only after the GEMV grid has drained, and the split
kernel, although launched with PDL, is released only when the QK norm + RoPE grid has completed, so its
loads of the request length, the slot and the window's context K/V all wait behind it.

With this fast path (part of ``drafter_folds``; PDL on):

* ``qk_norm_rope_pdl`` replaces the call with a copy of the shipped kernel (``_qk_norm_rope_pdl/``, BF16 in
  place, plain RoPE) launched with PDL. Every CTA runs ``griddepcontrol.wait`` before its first global read
  and then ``griddepcontrol.launch_dependents``. Built with the shipped library's nvcc flags (no
  ``--use_fast_math``), it writes the shipped kernel's bits; the unit test checks that on random inputs.
* The split kernel runs with ``split_pre_wait``: it loads the request length, the slot, over pages the
  request's page-table row, and the context part of its first key tile before its wait, and the block's keys
  (``k_new`` / ``v_new``) and the queries after it.

Why the pre-wait loads read final values: the split is released when every CTA of the QK norm + RoPE has
passed its wait, i.e. when the qkv GEMV grid has completed. Each kernel of the chain waits for its own
predecessor in every CTA that reads anything, so that grid's completion implies the completion of every
earlier kernel on the stream, including the drafter preamble that wrote the lengths and the slot. The
context K/V columns below the block, and over pages the step's page table, were written by earlier kernels
too (the context-K/V side chain, joined into the main stream before the drafter, or with
``drafter_preamble``'s deferred join right before layer 0's QK norm + RoPE, whose CTAs wait before their
first read and release the split only after it; and earlier decode steps); the block columns ``[L - M, L)``
the split reads from ``k_new`` / ``v_new`` and writes to the context only after its wait. The values every
kernel reads are those of the plain chain, so the drafter's output and its context are bitwise those of the
step without this fast path.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from torch import nn

from tensorrt_llm.logger import logger

#: The CUDA source of the library.
SOURCE_DIR = Path(__file__).resolve().parent / "_qk_norm_rope_pdl"
SOURCE = "qk_norm_rope_pdl.cu"
#: The shipped library's nvcc flags that change device code: none of ``--use_fast_math``'s.
NVCC_FLAGS = (
    "-O3",
    "--expt-relaxed-constexpr",
    "--expt-extended-lambda",
    "-lineinfo",
)
#: Prefix of the torch extension name (one per device architecture).
LIBRARY_NAME = "musespark_open_qk_norm_rope_pdl"

_LOCK = threading.Lock()
_LOADED: Optional[str] = None


def arch_target() -> str:
    """The arch-specific target of the current device (``107a``); raises on any other device."""
    major, minor = torch.cuda.get_device_capability()
    if (major, minor) == (10, 7):
        return "107a"
    raise RuntimeError(
        f"the drafter attention PDL library is built for sm_107a, not sm_{major}{minor}"
    )


def build(verbose: bool = False) -> str:
    """Compile (if needed) and load the library under torch's extension build root; its path."""
    from torch.utils.cpp_extension import load

    target = arch_target()
    path = load(
        name=f"{LIBRARY_NAME}_sm{target}",
        sources=[str(SOURCE_DIR / SOURCE)],
        extra_cuda_cflags=list(NVCC_FLAGS) + [f"-gencode=arch=compute_{target},code=sm_{target}"],
        is_python_module=False,
        verbose=verbose,
    )
    return str(path) if path else LIBRARY_NAME


def load_library() -> str:
    """Build and load the library once per process; returns its path."""
    global _LOADED
    with _LOCK:
        if _LOADED is None:
            _LOADED = build()
        return _LOADED


@dataclass(frozen=True)
class QkNormRopeArgs:
    """The scalar arguments ``apply_qk_norm_rope`` passes for one attention module (plain RoPE)."""

    num_heads_q: int
    num_heads_k: int
    head_dim: int
    rotary_dim: int
    eps: float
    base: float
    is_neox: bool
    factor: float
    low: float
    high: float
    attention_factor: float
    is_qk_norm: bool
    use_gemma: bool


def module_args(attn_mod: nn.Module) -> QkNormRopeArgs:
    """``attn_mod.apply_qk_norm_rope``'s arguments, as ``qk_norm_attention.QKNormRoPEAttention`` computes them.

    Raises:
        ValueError: if the module uses interleaved mRoPE (which the copy does not take).
    """
    from tensorrt_llm._torch.attention.qk_norm_attention import compute_yarn_parameters

    pos = attn_mod.pos_embd_params
    if (
        bool(getattr(pos, "mrope_interleaved", False))
        and getattr(pos, "mrope_section", None) is not None
    ):
        raise ValueError("interleaved mRoPE is not taken")
    factor, low, high, attention_factor = compute_yarn_parameters(attn_mod.pretrained_config)
    return QkNormRopeArgs(
        num_heads_q=int(attn_mod.num_heads),
        num_heads_k=int(attn_mod.num_key_value_heads),
        head_dim=int(attn_mod.head_dim),
        rotary_dim=int(attn_mod._get_qk_norm_rotary_dim()),
        eps=float(attn_mod.q_norm.variance_epsilon),
        base=float(pos.rope.theta),
        is_neox=bool(pos.is_neox),
        factor=float(factor),
        low=float(low),
        high=float(high),
        attention_factor=float(attention_factor),
        is_qk_norm=bool(attn_mod.is_qk_norm),
        use_gemma=bool(attn_mod.use_gemma_rms_norm),
    )


def qk_norm_rope_pdl(
    attn_mod: nn.Module,
    args: QkNormRopeArgs,
    qkv: torch.Tensor,
    positions_i32: torch.Tensor,
    pdl: bool = True,
) -> None:
    """``attn_mod.apply_qk_norm_rope(qkv, positions)`` on the PDL copy, in place on ``qkv``."""
    torch.ops.musespark_open_qk_norm_rope_pdl.run(
        qkv,
        args.num_heads_q,
        args.num_heads_k,
        args.num_heads_k,
        args.head_dim,
        args.rotary_dim,
        args.eps,
        attn_mod.q_norm.weight,
        attn_mod.k_norm.weight,
        args.base,
        args.is_neox,
        positions_i32,
        args.factor,
        args.low,
        args.high,
        args.attention_factor,
        args.is_qk_norm,
        args.use_gemma,
        pdl,
    )


class AttnPdl:
    """The per-layer arguments of the copy, held on the drafter as ``_pt_attn_pdl``."""

    def __init__(self, layer_args):
        self.layer_args = tuple(layer_args)

    def qk_norm_rope(self, layer_idx: int, attn_mod: nn.Module, qkv, positions_i32) -> None:
        qk_norm_rope_pdl(attn_mod, self.layer_args[layer_idx], qkv, positions_i32)


def unsupported_reason(draft_model: nn.Module) -> Optional[str]:
    """Why the fast path cannot run on ``draft_model``, or None.

    The block decode takes the copy only on its own path (the drafter's fused QK norm + RoPE, BF16) and only
    when it launches with PDL; other steps keep ``apply_qk_norm_rope``.
    """
    if not hasattr(draft_model, "model") or not hasattr(draft_model.model, "layers"):
        return "no drafter layers"
    for attn_mod in (layer.self_attn for layer in draft_model.model.layers):
        try:
            args = module_args(attn_mod)
        except (AttributeError, ValueError) as exc:
            return f"attention module: {exc}"
        if args.head_dim not in (64, 128, 256):
            return f"head_dim {args.head_dim}"
        if attn_mod.q_norm.weight.dtype != torch.bfloat16:
            return "the QK norm weights are not BF16"
    return None


def install(draft_model: nn.Module) -> str:
    """Build and load the library and set ``draft_model._pt_attn_pdl``; a one-line summary.

    Part of the ``drafter_folds`` fast path. Idempotent: a drafter that has it keeps it (the
    shell installs from every drafter load hook).

    Raises:
        ValueError: ``unsupported_reason``.
    """
    existing = draft_model.__dict__.get("_pt_attn_pdl") if draft_model is not None else None
    if existing is not None:
        return f"{len(existing.layer_args)} layers"
    reason = unsupported_reason(draft_model)
    if reason is not None:
        raise ValueError(f"[musespark_open] drafter_attn_pdl cannot run here: {reason}")
    load_library()
    layer_args = [module_args(layer.self_attn) for layer in draft_model.model.layers]
    draft_model.__dict__["_pt_attn_pdl"] = AttnPdl(layer_args)
    summary = f"{len(layer_args)} layers"
    logger.info(
        f"[musespark_open] drafter_attn_pdl: QK norm + RoPE on the PDL copy, split pre-wait loads ({summary})"
    )
    return summary


def uninstall(draft_model: nn.Module) -> None:
    draft_model.__dict__.pop("_pt_attn_pdl", None)
