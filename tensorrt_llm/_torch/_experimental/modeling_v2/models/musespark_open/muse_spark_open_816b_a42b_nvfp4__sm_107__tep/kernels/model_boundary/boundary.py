# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fast path ``model_boundary``: the decode step's model input and output boundaries, fewer launches.

Input. ``DecodeTarget`` runs, before layer 0's QKV projection, the replicated embedding gather,
``embed_norm``, the FP32 residual copy, its BF16 copy and layer 0's ``pre_qkv_proj_norm``: five
launches. With this fast path a decode step of at most ``replicated_embedding.MAX_ROWS`` ids
runs ``input_boundary`` instead -- one launch, the same bits -- and hands its two outputs to
layer 0 as the residual stream and ``precomputed_attn_input``, as every later layer receives
them from the previous layer's fused boundary.

Output. Layer 61 has no next norm, so its post-FFN join is the built-in
``add_after_rmsnorm`` (the Triton ``_gated_residual_after_rmsnorm_kernel``, FP32 out), then
``DecodeTarget`` casts the stream to BF16 for the final norm: a separate copy kernel. While
``ModelBoundary.armed`` holds, that join's kernel stores BF16 directly -- the same kernel,
its output pointer BF16, so Triton's store rounds the FP32 value to nearest even exactly as
``.to(torch.bfloat16)`` does -- and the cast is a no-op. The FP32 stream after layer 61 has
no reader: it is not a DFlash capture layer (``install`` refuses otherwise) and the final
norm reads its BF16 cast only.

What this module does:

* ``install`` (after ``replicated_embedding``, from ``load_weights``) checks the model it acts
  on -- the replicated table, ``m_emb == 1``, the historical pre-QKV ordering, flashinfer's
  RMSNorm geometry, layer 61 not captured -- builds the kernel (agreed over the TP group: a
  refusal on one rank makes every rank raise) and re-classes layer 61's post-FFN join.
* ``ModelBoundary.takes`` / ``input`` serve the input boundary per call: shapes and dtypes only,
  so every rank takes the same branch; ``inputs_embeds`` calls and other row counts keep the
  built-in chain.
* ``ModelBoundary.armed`` arms layer 61's BF16 join for one decode forward and disarms it on
  any exit; prefill never arms it.
"""

from __future__ import annotations

import contextlib
from typing import FrozenSet, Optional

import torch

from tensorrt_llm._torch.models import modeling_musespark_open as _builtin
from tensorrt_llm.logger import logger

from ... import dispatch as _tp_agreement
from ..replicated_embedding import embedding as _replicated
from . import input_boundary as _input


class Bf16OutputJoin:
    """Mixin for layer 61's post-FFN join: ``add_after_rmsnorm`` stores BF16 while armed.

    ``_bf16_out`` is set by ``ModelBoundary.armed`` for one decode forward. Unarmed, the
    join is the class it was before ``install``.
    """

    _bf16_out: bool = False

    def add_after_rmsnorm(
        self,
        residual: torch.Tensor,
        branch: torch.Tensor,
        eps: float,
        gain: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if not self._bf16_out:
            return super().add_after_rmsnorm(residual, branch, eps, gain)
        num_tokens = residual.shape[0]
        hidden_size = residual.shape[-1]
        if not (
            self._fused_enabled
            and residual.is_cuda
            and num_tokens > 0
            and residual.is_contiguous()
            and branch.is_contiguous()
            and branch.dtype == torch.bfloat16
            and (gain is None or gain.dtype == branch.dtype)
        ):
            # The built-in composition, then the cast DecodeTarget would have made.
            return super().add_after_rmsnorm(residual, branch, eps, gain).to(torch.bfloat16)
        alpha, beta = self.coefficients()
        out = torch.empty(residual.shape, dtype=torch.bfloat16, device=residual.device)
        # The built-in launch (MuseSparkOpenGatedResidual.add_after_rmsnorm), with a BF16 output.
        _builtin._gated_residual_after_rmsnorm_kernel[(num_tokens,)](
            out,
            residual,
            branch,
            gain if gain is not None else alpha,
            alpha,
            beta,
            eps,
            out,
            alpha,
            eps,
            HIDDEN=hidden_size,
            BLOCK_H=_builtin.triton.next_power_of_2(hidden_size),
            HAS_GAIN=gain is not None,
            FUSE_NEXT_NORM=False,
            NORM_FP32_RESIDUAL=False,
            num_warps=_builtin._RESIDUAL_NUM_WARPS,
        )
        return out


_JOIN_CLASSES: dict = {}


def _bf16_join_class(base: type) -> type:
    cls = _JOIN_CLASSES.get(base)
    if cls is None:
        cls = type(f"ModelBoundaryBf16{base.__name__}", (Bf16OutputJoin, base), {})
        _JOIN_CLASSES[base] = cls
    return cls


class ModelBoundary:
    """What ``DecodeTarget`` holds under ``model_boundary`` (``core._model_boundary``)."""

    def __init__(self, model) -> None:
        self.table = model.embed_tokens.__dict__[_replicated.TABLE_ATTR]
        self.embed_norm = model.embed_norm
        self.qkv_norm = model.layers[0].pre_qkv_proj_norm
        self.last_join = model.layers[-1].add_ffn

    def takes(
        self, input_ids: Optional[torch.Tensor], inputs_embeds: Optional[torch.Tensor]
    ) -> bool:
        """Whether this call's input boundary runs fused (shapes and dtypes only)."""
        return (
            inputs_embeds is None
            and input_ids is not None
            and input_ids.dim() == 1
            and 1 <= input_ids.shape[0] <= _replicated.MAX_ROWS
            and _input.takes(self.table, input_ids, self.embed_norm.weight, self.qkv_norm.weight)
        )

    def input(self, input_ids: torch.Tensor):
        """``(FP32 residual, layer 0's pre-QKV norm output)`` of the step, one launch."""
        return _input.input_boundary(
            self.table,
            input_ids,
            self.embed_norm.weight,
            self.embed_norm.variance_epsilon,
            self.qkv_norm.weight,
            self.qkv_norm.variance_epsilon,
        )

    @contextlib.contextmanager
    def armed(self):
        """Layer 61's join stores BF16 for one decode forward; disarmed on any exit."""
        self.last_join._bf16_out = True
        try:
            yield
        finally:
            self.last_join._bf16_out = False


def _check(model, captured: FrozenSet[int]) -> None:
    """``install``'s checks. Raises ValueError naming the first that fails."""
    reason = _input.available()
    if reason is not None:
        raise ValueError(f"[musespark_open] model_boundary: {reason}")
    if _replicated.TABLE_ATTR not in model.embed_tokens.__dict__:
        raise ValueError(
            "[musespark_open] model_boundary gathers from the replicated table; replicated_embedding "
            "is not installed"
        )
    if model.m_emb != 1.0:
        raise ValueError(
            f"[musespark_open] model_boundary is built for m_emb == 1, got {model.m_emb}"
        )
    layers = list(model.layers)
    if layers[0].add_attn._preqkv_norm_fp32:
        raise ValueError(
            "[musespark_open] model_boundary computes layer 0's pre-QKV norm on the BF16 residual; "
            "MUSESPARK_OPEN_PREQKV_NORM_FP32_RESIDUAL selects the FP32 ordering"
        )
    for name, norm in (
        ("embed_norm", model.embed_norm),
        ("pre_qkv_proj_norm", layers[0].pre_qkv_proj_norm),
    ):
        if norm.use_cuda_tile or norm.is_nvfp4 or norm.weight.dtype != torch.bfloat16:
            raise ValueError(
                f"[musespark_open] model_boundary: {name} is not flashinfer's BF16 RMSNorm here"
            )
    if model.embed_norm.use_gemma or not layers[0].pre_qkv_proj_norm.use_gemma:
        raise ValueError(
            "[musespark_open] model_boundary expects a plain embed_norm and a gemma pre_qkv_proj_norm"
        )
    last = len(layers) - 1
    if last in captured:
        raise ValueError(
            f"[musespark_open] model_boundary: layer {last} is a DFlash capture layer; its FP32 "
            "residual stream has a reader"
        )
    join = layers[-1].add_ffn
    if not isinstance(join, _builtin.MuseSparkOpenGatedResidual):
        raise ValueError(
            f"[musespark_open] model_boundary: layer {last}'s post-FFN join is {type(join).__name__}"
        )


def install(model, captured: FrozenSet[int], mapping=None) -> ModelBoundary:
    """Build the input-boundary kernel and re-class layer 61's join (agreed over ``mapping``'s TP group)."""
    reason = None
    try:
        _check(model, captured)
        # Compile now, with the step's id dtype (int32), so a compile failure is agreed here.
        _input._compile(_input.HIDDEN, _input.EXIT, "int32", torch.cuda.current_device())
    except Exception as error:  # noqa: BLE001 - every failure must reach the agreement
        reason = str(error)
    reason = _tp_agreement.agreed_refusal(mapping, reason)
    if reason is not None:
        raise ValueError(reason)
    join = model.layers[-1].add_ffn
    join.__class__ = _bf16_join_class(type(join))
    join._bf16_out = False
    boundary = ModelBoundary(model)
    model._model_boundary = boundary
    logger.info_once(
        "[musespark_open] model_boundary: decode input boundary in one launch, layer "
        f"{len(model.layers) - 1} join stores BF16",
        key="musespark_open_model_boundary",
    )
    return boundary
