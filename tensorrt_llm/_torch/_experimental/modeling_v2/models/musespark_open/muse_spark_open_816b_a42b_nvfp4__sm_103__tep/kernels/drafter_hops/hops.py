# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The DFlash 2 drafter's hop sites on the fused drafter hop (fast path ``drafter_hops``).

A drafter layer has two hops, both in ``draft.windowed_block_decode`` (the
``drafter_block_decode`` block decode): after the row-parallel ``o_proj`` (all-reduce, the
attention convolution's ``finish``, ``post_attention_layernorm``) and after
the row-parallel ``down_proj`` (all-reduce, the MLP convolution's ``finish``,
the next layer's ``input_layernorm`` or, after the last layer,
``model.norm``). With this fast path each hop is one
``ops/allreduce/drafter_hop.py`` launch, bit for bit the built-in three.

The projection runs as before but returns its partial unreduced
(``AllReduceParams(enable_allreduce=False)``) and the hop publishes it; the
hop's dependent-launch trigger fires right after its last transport access.

The site order (``2 * layer`` after ``o_proj``, ``2 * layer + 1`` after
``down_proj``) is the same on every rank and every forward; a forward takes
the fused hops at all its sites or at none, decided on the block's row count
(``DrafterHopCall.takes``), which every rank shares.

With ``wide_drafter_hops`` (``install_wide``) a forward of ``MAX_ROWS + 1`` ..
``wide_served_rows()`` rows (the blocks of several requests at K = 7, per device in
``WIDE_SERVED_ROWS_BY_SM``) takes the same hops on a second, wide workspace; a forward of at
most ``MAX_ROWS`` rows keeps the first one.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from tensorrt_llm._torch.flashinfer_utils import IS_FLASHINFER_AVAILABLE
from tensorrt_llm._torch.models.modeling_dflash import DFlash2BlockConv, DFlashForCausalLM
from tensorrt_llm._torch.modules.linear import Linear, TensorParallelMode
from tensorrt_llm._torch.modules.rms_norm import RMSNorm
from tensorrt_llm.functional import AllReduceParams
from tensorrt_llm.logger import logger

from ... import dispatch as _tp_agreement
from ...ops.allreduce import drafter_hop as _hop

#: The projection keyword of a hop site: return the partial, the hop all-reduces it.
_NO_ALLREDUCE = AllReduceParams(enable_allreduce=False)


class DrafterHopCall:
    """What the drafter holds for its fused hops: the workspace, the per-site modules, the mode."""

    def __init__(self, workspace: "_hop.DrafterHopWorkspace") -> None:
        self.workspace = workspace
        #: The wide workspace (``wide_drafter_hops``), or None.
        self.wide: Optional["_hop.DrafterHopWorkspace"] = None
        #: The workspace of the forward ``takes`` admitted last.
        self.active = workspace
        #: Hops issued by this rank (a test counts them to see a forward took the fused path).
        self.calls = 0

    def takes(self, rows: int, block_size: int, dtype: torch.dtype) -> bool:
        """Whether a forward of ``rows`` rows runs every site on the fused hop, and on which
        workspace. Rank-uniform."""
        if dtype != torch.bfloat16 or block_size < 1 or rows < 1:
            return False
        if rows <= _hop.MAX_ROWS:
            self.active = self.workspace
            return True
        if self.wide is not None and rows <= self.wide.max_rows:
            self.active = self.wide
            return True
        return False

    def begin_forward(self) -> None:
        """A drafter forward starts; its hops name the sites from 0 (see the workspace's)."""
        self.active.begin_forward()

    def _site(self, site, partial, residual, coefficients, conv, norm, block_size):
        # ``coefficients`` is what ``prepare`` returned for ``finish``: [rows, taps, groups].
        delta = coefficients
        base = conv.base_kernel[1]
        if not _hop.hop_layout_ok(partial, residual, delta, base, norm.weight):
            # Rank-local layout repair, only of operands the hop reads after its
            # dependency wait (the copy is then a predecessor it waits for). The base
            # kernel and the norm weight are read before the wait and are taken as
            # they are: install checked them, and the op refuses them otherwise.
            partial = None if partial is None else partial.contiguous()
            delta = delta if delta.stride(-1) == 1 else delta.contiguous()
        self.calls += 1
        return _hop.drafter_hop(
            site,
            partial,
            residual,
            delta,
            base,
            norm.weight,
            norm.variance_epsilon,
            block_size,
            self.active,
        )

    def attention(
        self,
        layer_idx: int,
        o_proj: Linear,
        attn_output: torch.Tensor,
        residual: torch.Tensor,
        coefficients: torch.Tensor,
        conv: DFlash2BlockConv,
        norm: RMSNorm,
        block_size: int,
    ) -> torch.Tensor:
        """``o_proj`` -> all-reduce -> ``finish`` -> ``post_attention_layernorm``; ``residual`` in place."""
        site = 2 * layer_idx
        partial = o_proj(attn_output, all_reduce_params=_NO_ALLREDUCE)
        return self._site(site, partial, residual, coefficients, conv, norm, block_size)

    def mlp(
        self,
        layer_idx: int,
        mlp: nn.Module,
        hidden: torch.Tensor,
        residual: torch.Tensor,
        coefficients: torch.Tensor,
        conv: DFlash2BlockConv,
        norm: RMSNorm,
        block_size: int,
    ) -> torch.Tensor:
        """``mlp`` -> all-reduce -> ``finish`` -> the next norm; ``residual`` in place."""
        site = 2 * layer_idx + 1
        partial = mlp(hidden, final_all_reduce_params=_NO_ALLREDUCE)
        return self._site(site, partial, residual, coefficients, conv, norm, block_size)


def _mnnvl_requestable(mapping) -> bool:
    """Whether an MNNVL BF16 all-reduce requested by name can be built on ``mapping``'s TP group."""
    from tensorrt_llm._torch.distributed.ops import MNNVLAllReduce

    return bool(MNNVLAllReduce.is_mnnvl(mapping, torch.bfloat16, explicitly_requested=True))


def _all_reduce_reason(linear: nn.Module, where: str, tp_size: int) -> Optional[str]:
    if not isinstance(linear, Linear) or linear.tp_mode != TensorParallelMode.ROW:
        return f"{where} is not a row-parallel Linear"
    if not linear.reduce_output or linear.all_reduce is None:
        return f"{where} does not all-reduce its output"
    if linear.bias is not None:
        return f"{where} has a bias"
    if linear.weight.dtype != torch.bfloat16 or linear.weight.shape[0] != _hop.HIDDEN:
        return f"{where} is not a BF16 projection to {_hop.HIDDEN}"
    if linear.mapping.tp_size != tp_size:
        return f"{where} is on a TP group of {linear.mapping.tp_size}, the drafter's is {tp_size}"
    all_reduce = linear.all_reduce
    if getattr(all_reduce, "symm_mem_allreduce", None) is not None:
        return f"{where} all-reduces with symmetric memory, not the MNNVL one-shot"
    if tp_size == 4:
        # One tray: the built-in AUTO strategy builds the MNNVL all-reduce only across
        # nodes, so the install requests it by name (``request_mnnvl``) when it is missing.
        if getattr(all_reduce, "mnnvl_allreduce", None) is None and not _mnnvl_requestable(
            linear.mapping
        ):
            return f"{where} cannot all-reduce with the MNNVL one-shot on this TP group"
    elif tp_size == _hop.MAX_RANKS:
        if getattr(all_reduce, "mnnvl_allreduce", None) is None:
            return f"{where} does not all-reduce with the MNNVL one-shot"
    else:
        return f"{where} is on a TP group of {tp_size}, the hop serves 4 or 8"
    # A LoRA layer (attention modules always build one for o_proj) applies only with
    # lora_params, which the block decode never passes, on either path.
    return None


def row_linears(draft_model: nn.Module) -> list[Linear]:
    """The drafter's row-parallel projections the hops all-reduce: every layer's ``o_proj`` and ``down_proj``."""
    return [
        linear
        for layer in draft_model.model.layers
        for linear in (layer.self_attn.o_proj, layer.mlp.down_proj)
    ]


def request_mnnvl(draft_model: nn.Module) -> int:
    """Give every drafter row projection the MNNVL one-shot all-reduce, requested by name.

    For a TP group on one tray, where the built-in ``AUTO`` strategy leaves it
    off: each ``o_proj`` / ``down_proj`` all-reduce without one gets an
    ``MNNVLAllReduce`` (the one ``AllReduce(strategy=MNNVL)`` builds), so the calls
    the hop does not take (more than ``MAX_ROWS`` rows) reduce with the same
    one-shot arithmetic as the hop. Collective (the MNNVL workspace); returns how
    many all-reduces it changed.
    """
    from tensorrt_llm._torch.distributed.ops import MNNVLAllReduce

    changed = 0
    for linear in row_linears(draft_model):
        all_reduce = linear.all_reduce
        if getattr(all_reduce, "mnnvl_allreduce", None) is None:
            all_reduce.mnnvl_allreduce = MNNVLAllReduce(linear.mapping, torch.bfloat16)
            changed += 1
    return changed


def _norm_reason(norm: nn.Module, where: str) -> Optional[str]:
    if type(norm) is not RMSNorm:
        return f"{where} is {type(norm).__name__}, not the built-in RMSNorm"
    if norm.use_gemma or norm.use_cuda_tile or norm.return_hp_output:
        return f"{where} is not the plain flashinfer RMSNorm"
    if norm.is_nvfp4 and norm.nvfp4_scale is not None:
        return f"{where} quantizes its output"
    weight = getattr(norm, "weight", None)
    if weight is None or weight.dtype != torch.bfloat16 or tuple(weight.shape) != (_hop.HIDDEN,):
        return f"{where} has no BF16 weight of {_hop.HIDDEN}"
    return None


def _conv_reason(conv: nn.Module, where: str) -> Optional[str]:
    if not isinstance(conv, DFlash2BlockConv):
        return f"{where} is not a DFlash 2 block convolution"
    if (
        conv.taps != _hop.TAPS
        or conv.group_size != _hop.GROUP_SIZE
        or tuple(conv.base_kernel.shape) != (2, _hop.TAPS, _hop.HIDDEN)
        or conv.base_kernel.dtype != torch.bfloat16
    ):
        return f"{where} is not a two-tap, group-16, BF16 convolution over {_hop.HIDDEN}"
    if type(conv).finish is not DFlash2BlockConv.finish:
        return f"{where} overrides finish"
    return None


def pre_wait_operands(draft_model: nn.Module) -> list[tuple[str, torch.Tensor]]:
    """The parameters the drafter's hops read before their dependency wait, named.

    The finish-side base kernel of every block convolution and the weight of
    every norm a hop applies (``post_attention_layernorm``, the next layer's
    ``input_layernorm``, ``model.norm``). Written at load time only, so the hop
    may read them before ``griddepcontrol.wait``; a call must pass them as they
    are, never a fresh copy.
    """
    layers = draft_model.model.layers
    named = []
    for index, layer in enumerate(layers):
        named.append(
            (
                f"attention conv {index} finish base kernel",
                draft_model.attention_convs[index].base_kernel[1],
            )
        )
        named.append(
            (f"MLP conv {index} finish base kernel", draft_model.mlp_convs[index].base_kernel[1])
        )
        named.append(
            (
                f"layer {index} post_attention_layernorm weight",
                layer.post_attention_layernorm.weight,
            )
        )
        if index:
            named.append((f"layer {index} input_layernorm weight", layer.input_layernorm.weight))
    named.append(("model.norm weight", draft_model.model.norm.weight))
    return named


def unsupported_reason(draft_model: nn.Module) -> Optional[str]:
    """Why the drafter's hops cannot run on the fused hop, or None.

    Every fact checked here is the same on every rank of the group (module
    classes, shapes, dtypes, strides); the pointer alignment of the parameters
    the hops read before their dependency wait is checked by the workspace,
    under its group agreement.
    """
    from ..drafter_block_decode import block_decode as _draft

    if not IS_FLASHINFER_AVAILABLE:
        return "flashinfer (the built-in add + RMSNorm the hop reproduces) is unavailable"
    if not isinstance(draft_model, DFlashForCausalLM):
        return f"not a DFlashForCausalLM ({type(draft_model).__name__})"
    bound = draft_model.__dict__.get("dflash_forward")
    if getattr(bound, "__func__", None) is not _draft.windowed_block_decode:
        return "the drafter does not run the windowed block decode"
    if not draft_model.has_block_conv:
        return "the drafter has no DFlash 2 block convolutions"
    layers = draft_model.model.layers
    tp_size = layers[0].self_attn.o_proj.mapping.tp_size
    for index, layer in enumerate(layers):
        for reason in (
            _all_reduce_reason(layer.self_attn.o_proj, f"layer {index} o_proj", tp_size),
            _all_reduce_reason(layer.mlp.down_proj, f"layer {index} down_proj", tp_size),
            _norm_reason(layer.post_attention_layernorm, f"layer {index} post_attention_layernorm"),
            _norm_reason(layer.input_layernorm, f"layer {index} input_layernorm")
            if index
            else None,
            _conv_reason(draft_model.attention_convs[index], f"attention conv {index}"),
            _conv_reason(draft_model.mlp_convs[index], f"MLP conv {index}"),
        ):
            if reason is not None:
                return reason
    reason = _norm_reason(draft_model.model.norm, "model.norm")
    if reason is not None:
        return reason
    for where, tensor in pre_wait_operands(draft_model):
        if not tensor.is_contiguous():
            return (
                f"{where} is not contiguous (the hop reads it as it is, before its dependency wait)"
            )
    return None


def installed(draft_model: Optional[nn.Module]) -> bool:
    return isinstance(getattr(draft_model, "_drafter_hop", None), DrafterHopCall)


def _drafter_mapping(draft_model: Optional[nn.Module]):
    """The parallel mapping of the drafter's first ``o_proj``, or None if it has none."""
    try:
        return draft_model.model.layers[0].self_attn.o_proj.mapping
    except (AttributeError, IndexError, TypeError):
        return None


def install(draft_model: Optional[nn.Module], mapping=None) -> None:
    """Give the drafter its fused hop, triggering its dependents early. Collective (workspace).

    Called from the shell's drafter install after ``draft.install`` (``drafter_block_decode``)
    and ``drafter_worker.install`` (``drafter_fixed_gemms``), with the shell's ``mapping``
    (the drafter shares it; without one, the drafter's own ``o_proj`` mapping).
    The refusal is agreed over that TP group (collective) before the
    workspace's collective construction. On a TP group of 4 (one tray) the
    drafter's row projections are given the MNNVL one-shot all-reduce first
    (``request_mnnvl``, collective). Idempotent.

    Raises:
        ValueError: if the hop cannot run on this drafter on some rank (on every
            rank alike).
    """
    if installed(draft_model):
        return
    early = True
    if draft_model is None:
        reason = "no drafter"
    else:
        reason = unsupported_reason(draft_model)
    reason = _tp_agreement.agreed_refusal(
        mapping if mapping is not None else _drafter_mapping(draft_model),
        None if reason is None else f"[musespark_open] drafter_hops cannot run here: {reason}",
    )
    if reason is not None:
        raise ValueError(reason)
    layers = draft_model.model.layers
    tp_size = layers[0].self_attn.o_proj.mapping.tp_size
    if tp_size == 4:
        requested = request_mnnvl(draft_model)
        if requested:
            logger.info(
                f"[musespark_open] drafter_hops: MNNVL one-shot all-reduce requested by name on {requested} "
                "drafter row projections (one tray)"
            )
    norms = [layer.post_attention_layernorm for layer in layers]
    norms += [layer.input_layernorm for layer in layers[1:]] + [draft_model.model.norm]
    eps = sorted({float(norm.variance_epsilon) for norm in norms})
    mapping = layers[0].self_attn.o_proj.mapping
    workspace = _hop.DrafterHopWorkspace(
        mapping,
        sites=2 * len(layers),
        eps=eps,
        early_trigger=early,
        pre_wait=[tensor for _, tensor in pre_wait_operands(draft_model)],
    )
    draft_model._drafter_hop = DrafterHopCall(workspace)
    _hop.log_once_enabled(2 * len(layers), False, early)
    logger.info("[musespark_open] drafter_hops: fused drafter hops installed")


def install_wide(draft_model: Optional[nn.Module], mapping=None) -> None:
    """``wide_drafter_hops``: the fused hops also on drafter forwards of ``MAX_ROWS + 1`` ..
    ``wide_served_rows()`` rows of this device, on a wide workspace of their own. Collective;
    after ``install``.

    The calls the wide hops replace are the same three kernels per site as at 8 rows (the
    MNNVL one-shot all-reduce ``request_mnnvl`` gave every row projection on one tray, the
    finish convolution, flashinfer's fused add + RMSNorm); the hop reproduces them per row,
    and the tap-1 mask per block (``row % block``). Idempotent.

    Raises:
        ValueError: when ``drafter_hops`` is not installed on this drafter.
    """
    if not installed(draft_model):
        raise ValueError("[musespark_open] wide_drafter_hops needs drafter_hops installed first")
    call: DrafterHopCall = draft_model._drafter_hop
    if call.wide is not None:
        return
    narrow = call.workspace
    layers = draft_model.model.layers
    norms = [layer.post_attention_layernorm for layer in layers]
    norms += [layer.input_layernorm for layer in layers[1:]] + [draft_model.model.norm]
    eps = sorted({float(norm.variance_epsilon) for norm in norms})
    rows = _hop.wide_served_rows()
    call.wide = _hop.DrafterHopWorkspace(
        layers[0].self_attn.o_proj.mapping,
        sites=narrow.sites,
        eps=eps,
        early_trigger=narrow.early_trigger,
        pre_wait=[tensor for _, tensor in pre_wait_operands(draft_model)],
        max_rows=rows,
    )
    logger.info(
        f"[musespark_open] wide_drafter_hops: fused drafter hops on forwards of {_hop.MAX_ROWS + 1}.."
        f"{rows} rows ({call.wide.areas} areas, "
        f"{call.wide.nbytes() / 2**20:.1f} MiB of fabric memory on this rank)"
    )
