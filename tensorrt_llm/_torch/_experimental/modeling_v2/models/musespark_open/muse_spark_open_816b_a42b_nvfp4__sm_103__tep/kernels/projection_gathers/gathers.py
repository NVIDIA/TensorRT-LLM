# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""projection_gathers: the sharded projections' column gathers, fused into their consumers.

With ``MUSESPARK_OPEN_SHARDED_PROJECTIONS=1`` the built-in MoE computes its
pool (8192 -> 4096) and unpool (4096 -> 8192) projections as column slices per
rank of its TP8 (two trays) or TP4 (one tray) group and gathers each slice with
a separate fabric kernel. On decode steps
of at most 8 rows this fast path hands the slices straight to their consumers:

* unpool: ``ops/comm/unpool_gather_boundary`` gathers the slice
  inside the post-FFN boundary kernel (residual join + next pre-QKV norm);
* pool: ``ops/comm/pool_gather_fp4quant`` gathers the slice inside
  flashinfer's pre-expert RMSNorm + NVFP4 quantization kernel.

It works by re-classing three built-in modules of every
layer in place -- methods only, the module tree and parameters stay the
built-in model's:

* the decoder layer accepts ``fused_unpool_gather`` / ``fused_pool_gather`` from ``DecodeTarget``
  and, for the call, lets its MoE return those slices ungathered (the unpool
  not on the last layer, whose residual join has no next norm and keeps the
  built-in path);
* the MoE's ``_replicated_projection`` returns the slice (the same GEMM the
  built-in runs) instead of gathering it, and its ``prepare_expert_input``
  runs the fused pool op when the built-in would have run flashinfer's fused
  quantization, or gathers with the built-in workspace and runs the built-in
  otherwise;
* the post-FFN gated residual does the same for the unpool slice and the
  built-in cluster boundary.

Either way every value is the built-in path's, bit for bit. Prefill steps
never pass the flags and run the built-in path unchanged.

``wide_projection_gathers`` (``kernels/wide_projection_gathers``) widens the
same three classes to decode steps of 9..32 rows while ``DecodeTarget`` arms it
(``_wide``): there the projections return the rank's slice too, and the fused
ops run on the path's wide inboxes (``_wide_pool_inbox``, ``_wide_gather_inbox``,
32 rows per slot); the 8-row inboxes above serve only calls of at most 8 rows.
At those rows the built-in multiplies the whole replicated weights, so the
slice GEMM is a different kernel from the built-in's (FP32 summation order);
the gathers and their consumers are the built-in kernels' bits on the gathered
rows, as at 1..8.

``projection_publish`` (``kernels/projection_publish``) takes the same deferred
projections on decode steps of 33..512 rows while ``DecodeTarget`` arms it
(``_published``): the slice GEMM writes into a multicast transport slot and the
pre-expert quantization and the post-FFN join consume the published slot.

Every rank must take the same path at every call: the fused ops and the
built-in gathers are collectives over different memory, so a rank that took
the other one would leave its peers waiting forever. The per-call choice
therefore reads only shapes, dtypes, contiguity and construction-time flags,
never pointer values; the fused path then requires its operands' 16-byte
alignment (fresh allocations, so always true) and raises before launching
anything if one is not aligned. Whether every rank can build each site's kernel
is agreed over the TP group at ``install``.

The classes below copy built-in code: the decoder-layer and MoE forwards that
route each projection's output to its one consumer
(``MuseSparkOpenDecoderLayer.forward``, ``MuseSparkOpenMoE.forward``,
``MuseSparkOpenMoE.prepare_moe_inputs``), the sharded projection whose GEMM
``_replicated_projection`` repeats, the gather the fused ops replace
(``musespark_open_projection_gather``), and the consumers' kernel choice
(``MuseSparkOpenGatedResidual._add_after_rmsnorm_and_norm``,
``MuseSparkOpenMoE.prepare_expert_input``).
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch

from tensorrt_llm._torch.distributed.communicator import Distributed
from tensorrt_llm._torch.models.modeling_musespark_open import (
    MuseSparkOpenDecoderLayer,
    MuseSparkOpenGatedResidual,
    MuseSparkOpenModel,
    MuseSparkOpenMoE,
)
from tensorrt_llm._torch.modules.rms_norm import RMSNorm
from tensorrt_llm._torch.utils import Fp4QuantizedTensor
from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping

from ...ops.comm import pool_gather_fp4quant as _pool
from ...ops.comm import unpool_gather_boundary as _unpool
from ...ops.comm.projection_inbox import MAX_ROWS, WIDE_ROWS, ProjectionInbox


def fused_rows(owner) -> int:
    """The most rows a call of ``owner`` (a re-classed MoE or post-FFN join) may fuse: 32 while
    ``wide_projection_gathers`` is armed for the forward (``owner._wide``), else 8."""
    wide = getattr(owner, "_wide", None)
    return WIDE_ROWS if wide is not None and wide.is_armed else MAX_ROWS


def _require_aligned(**tensors: torch.Tensor) -> None:
    """Raise unless every operand starts on a 16-byte boundary (the fused ops' vector accesses).

    Checked after the fused path was chosen -- a choice that must not depend on
    pointer values, so that every rank makes it alike -- and before anything is
    launched, so a failure raises on the host instead of leaving peers waiting.
    """
    for name, tensor in tensors.items():
        if tensor.data_ptr() % 16:
            raise RuntimeError(
                f"[musespark_open] projection_gathers: fused-path operand {name} at {tensor.data_ptr():#x} is not "
                "16-byte aligned; its producers allocate fresh tensors, so a caller changed"
            )


class MuseSparkOpenTepGatherMoE(MuseSparkOpenMoE):
    """The built-in MoE, whose pool and unpool may hand their slices on ungathered."""

    _defer_unpool: bool
    _defer_pool: bool
    _pool_inbox: Optional[ProjectionInbox]
    #: ``wide_projection_gathers``: its arming (``WideProjections``) and the pool site's
    #: 32-row inbox; None without the path.
    _wide = None
    _wide_pool_inbox: Optional[ProjectionInbox] = None
    #: ``projection_publish``: its arming (``PublishedProjections``) and this layer's buffer of
    #: each site's transport (None: the site does not publish); None without the path.
    _published = None
    _published_pool_buffer: Optional[int] = None
    _published_unpool_buffer: Optional[int] = None

    def _replicated_projection(self, module, hidden_states, workspace):
        defer = (self._defer_unpool and module is self.unpooler) or (
            self._defer_pool and module is self.pooler
        )
        published = self._published
        if defer and published is not None and published.is_armed:
            # 33..512 rows: the slice GEMM publishes into the site's transport.
            slot = published.project(self, module, hidden_states)
            if slot is not None:
                return slot
        if (
            not defer
            or workspace is None
            or torch.compiler.is_compiling()
            or hidden_states.dtype != torch.bfloat16
            or hidden_states.ndim != 2
            or not 0 < hidden_states.shape[0] <= fused_rows(self)
            or not hidden_states.is_contiguous()
        ):
            return super()._replicated_projection(module, hidden_states, workspace)
        # The built-in sharded projection, minus its gather.
        from tensorrt_llm._torch.modules.low_m_gemm import apply_low_m_gemm

        columns = module.weight.shape[0] // self.mapping.tp_size
        begin = self.mapping.tp_rank * columns
        weight = module.weight[begin : begin + columns]
        projected = apply_low_m_gemm(module, hidden_states, weight, None, force_active=True)
        if projected is None:
            projected = torch.nn.functional.linear(hidden_states, weight)
        return projected

    def prepare_expert_input(self, pooled: torch.Tensor):
        published = self._published
        if published is not None and published.is_armed:
            quantized = published.consume_pool(self, pooled)
            if quantized is not None:
                return quantized
        if (
            pooled.dim() == 2
            # The built-in workspace's TP size is the one the slices are cut for.
            and pooled.shape[1] == _pool.local_columns(self._pool_gather_workspace.world)
            and self.expert_hidden_size == _pool.HIDDEN
        ):
            args = self._pool_quant_args(pooled)
            if args is not None:
                gain, global_scale = args
                _require_aligned(pooled=pooled, gain=gain)
                narrow = pooled.shape[0] <= MAX_ROWS
                # At 1..8 rows the gathered row is written although nothing reads it:
                # the op with that store runs better beside the 8-row decode router (the
                # pool site's real neighbour) than the op without it.
                fp4, scale, _ = _pool.pool_gather_rmsnorm_fp4quant(
                    pooled,
                    self._pool_inbox if narrow else self._wide_pool_inbox,
                    gain,
                    global_scale,
                    self.pre_expert_norm.variance_epsilon,
                    return_gathered=narrow,
                )
                if not narrow:
                    self._wide.report("pool", pooled.shape[0], True)
                return Fp4QuantizedTensor(fp4, scale, is_sf_swizzled=False)
            if pooled.shape[0] > MAX_ROWS and self._wide is not None:
                self._wide.report("pool", pooled.shape[0], False)
            # Gather as the built-in would have, then prepare as it would have.
            pooled = self._pool_gather_workspace.gather(pooled)
        return super().prepare_expert_input(pooled)

    def _pool_quant_args(self, pooled: torch.Tensor):
        """(gain, global scale) exactly when the built-in would quantize the gathered row
        with flashinfer's fused ``rmsnorm_fp4quant``; ``None`` otherwise.

        The built-in conditions of ``prepare_expert_input``, evaluated for the
        gathered row it would have seen (same rows, BF16, CUDA, contiguous).
        Nothing here can differ between ranks.
        """
        from tensorrt_llm._torch.modules.fused_ops.rmsnorm_fp4_quant import (
            rmsnorm_fp4_quant_available,
        )
        from tensorrt_llm._torch.moe.fused_moe.moe_op_backend import TRTLLMOpBackend

        backend = getattr(self.experts, "backend", self.experts)
        rows = pooled.shape[0]
        inbox = self._pool_inbox if rows <= MAX_ROWS else self._wide_pool_inbox
        if (
            inbox is None
            or not getattr(self, "_pre_expert_quant_fused", True)
            or not pooled.is_cuda
            or pooled.dtype != torch.bfloat16
            or not pooled.is_contiguous()
            or torch.compiler.is_compiling()
            or getattr(self.experts, "comm", None) is not None
            or not isinstance(getattr(backend, "op_backend", None), TRTLLMOpBackend)
            or not getattr(backend, "has_nvfp4", False)
            or getattr(backend, "fc31_act_scale", None) is not None
            or not 1 <= rows <= fused_rows(self)
            or rows > getattr(self.experts, "moe_max_num_tokens", rows)
            or not rmsnorm_fp4_quant_available()
        ):
            return None
        scale = getattr(backend, "fc31_input_scale", None)
        if scale is None or scale.numel() != 1 or scale.dtype != torch.float32:
            return None
        if self._pre_expert_quant_gain is None:
            # The built-in's lazily cached gain, computed the built-in's way.
            self._pre_expert_quant_gain = (self.pre_expert_norm.weight.float() + 1).to(pooled.dtype)
        gain = self._pre_expert_quant_gain
        if (
            gain.dtype != torch.bfloat16
            or tuple(gain.shape) != (_pool.HIDDEN,)
            or not gain.is_contiguous()
        ):
            return None
        return gain, scale.reshape(1)


class MuseSparkOpenTepGatedResidual(MuseSparkOpenGatedResidual):
    """The built-in gated residual join, which also takes the unpool's local slice.

    ``_boundary_trigger`` is when the fused kernel lets its dependent (the next
    layer's QKV projection) launch: ``"exit"`` as the built-in boundary does,
    unless ``boundary_early_release`` sets ``"entry"`` (``pdl_triggers.install_boundary_triggers``).
    ``_capture_fold`` (set by ``capture_fold`` for one call) is a
    DFlash capture view the fused kernel also writes ``updated.to(bfloat16)``
    into; ``_capture_stored`` records that it did. Unset, the join is the plain fused gather join.
    """

    _gather_inbox: Optional[ProjectionInbox]
    _gather: Optional[object]
    #: ``wide_projection_gathers``: its arming and the unpool site's 32-row inbox; None
    #: without the path.
    _wide = None
    _wide_gather_inbox: Optional[ProjectionInbox] = None
    #: ``projection_publish``: its arming and this layer's buffer of the unpool transport.
    _published = None
    _published_unpool_buffer: Optional[int] = None
    _boundary_trigger: str = "exit"
    _capture_fold: Optional[torch.Tensor] = None
    _capture_stored: bool = False

    def _add_after_rmsnorm_and_norm(
        self,
        residual: torch.Tensor,
        branch: torch.Tensor,
        eps: float,
        next_norm: RMSNorm,
        gain: Optional[torch.Tensor] = None,
    ):
        published = self._published
        if published is not None and published.is_armed:
            joined = published.consume_unpool(self, residual, branch, eps, next_norm, gain)
            if joined is not None:
                return joined
        if (
            branch.dim() == 2
            # The built-in workspace's TP size is the one the slices are cut for.
            and branch.shape[1] == _unpool.local_columns(self._gather.world)
            and residual.shape[-1] == _unpool.HIDDEN
        ):
            alpha, beta = self.coefficients()
            fused = self._fits_fused(residual, branch, next_norm, gain, alpha, beta)
            if branch.shape[0] > MAX_ROWS and self._wide is not None:
                self._wide.report("unpool", branch.shape[0], fused)
            if fused:
                _require_aligned(
                    residual=residual,
                    branch=branch,
                    gain=next_norm.weight if gain is None else gain,
                    alpha=alpha,
                    beta=beta,
                    weight=next_norm.weight,
                )
                inbox = (
                    self._gather_inbox if residual.shape[0] <= MAX_ROWS else self._wide_gather_inbox
                )
                capture = self._capture_fold
                if capture is not None and not _unpool.capture_fits(
                    capture, residual.shape[0], residual.device
                ):
                    raise RuntimeError(
                        f"[musespark_open] capture_fold: capture view {tuple(capture.shape)} stride "
                        f"{tuple(capture.stride())} {capture.dtype} does not fit a "
                        f"{residual.shape[0]}-row boundary call"
                    )
                if capture is None:
                    return _unpool.unpool_gather_boundary(
                        branch,
                        inbox,
                        residual,
                        gain,
                        alpha,
                        beta,
                        next_norm.weight,
                        eps,
                        next_norm.variance_epsilon,
                        self._preqkv_norm_fp32,
                        self._boundary_trigger,
                    )
                result = _unpool.unpool_gather_boundary(
                    branch,
                    inbox,
                    residual,
                    gain,
                    alpha,
                    beta,
                    next_norm.weight,
                    eps,
                    next_norm.variance_epsilon,
                    self._preqkv_norm_fp32,
                    self._boundary_trigger,
                    capture=capture,
                )
                self._capture_stored = True
                return result
            # Gather as the built-in would have, then join as it would have.
            branch = self._gather.gather(branch)
        return super()._add_after_rmsnorm_and_norm(residual, branch, eps, next_norm, gain)

    def _fits_fused(self, residual, branch, next_norm, gain, alpha, beta) -> bool:
        """Exactly when the built-in join would run the cluster boundary kernel.

        ``_add_after_rmsnorm_and_norm`` takes the cluster kernel when its fused
        arm is on, the shapes and dtypes are the boundary's, and
        ``cluster_boundary_supported`` holds; the gathered branch it would have
        seen is a fresh ``[rows, 8192]`` BF16 tensor. The built-in predicate's
        16-byte alignment terms are not evaluated here -- pointer values are the
        one input that could differ between ranks -- but asserted by the caller
        once the fused path is chosen.
        """
        from tensorrt_llm._torch.models.musespark_open_boundary import cluster_boundary_available

        hidden = _unpool.HIDDEN
        rows = residual.shape[0] if residual.dim() == 2 else 0
        inbox = self._gather_inbox if rows <= MAX_ROWS else self._wide_gather_inbox
        if not (
            inbox is not None
            and self._fused_enabled
            and not torch.compiler.is_compiling()
            and 1 <= rows <= min(fused_rows(self), _unpool.max_rows(inbox))
            and branch.shape[0] == rows
            and residual.is_cuda
            and residual.dtype == torch.float32
            and branch.dtype == torch.bfloat16
            and residual.is_contiguous()
            and branch.is_contiguous()
            and next_norm.use_gemma
            and next_norm.weight.dtype == torch.bfloat16
            and next_norm.weight.is_contiguous()
            and (gain is None or (gain.dtype == torch.bfloat16 and gain.is_contiguous()))
            and cluster_boundary_available(residual.device)
        ):
            return False
        gain_arg = next_norm.weight if gain is None else gain
        vectors = (gain_arg, alpha, beta, next_norm.weight)
        return (
            all(
                tuple(t.shape) == (hidden,) and t.device == residual.device and t.is_contiguous()
                for t in vectors
            )
            and alpha.dtype == torch.float32
            and beta.dtype == torch.float32
        )


class MuseSparkOpenTepDecoderLayer(MuseSparkOpenDecoderLayer):
    """The built-in decoder layer, which can defer its projection gathers to their consumers."""

    def forward(
        self,
        position_ids: torch.IntTensor,
        hidden_states: torch.Tensor,
        attn_metadata,
        precomputed_attn_input: Optional[torch.Tensor] = None,
        next_layer_norm: Optional[RMSNorm] = None,
        sp_num_tokens: Optional[int] = None,
        fused_unpool_gather: bool = False,
        fused_pool_gather: bool = False,
        **kwargs,
    ):
        moe = self.feed_forward
        moe._defer_unpool = (
            bool(fused_unpool_gather) and next_layer_norm is not None and sp_num_tokens is None
        )
        moe._defer_pool = bool(fused_pool_gather) and sp_num_tokens is None
        try:
            return super().forward(
                position_ids,
                hidden_states,
                attn_metadata,
                precomputed_attn_input=precomputed_attn_input,
                next_layer_norm=next_layer_norm,
                sp_num_tokens=sp_num_tokens,
                **kwargs,
            )
        finally:
            moe._defer_unpool = False
            moe._defer_pool = False


#: The MoE classes ``install`` takes, and what it re-classes each as.
_MOE_CLASSES = {
    MuseSparkOpenMoE: MuseSparkOpenTepGatherMoE,
}


def fusable_sites(mapping: Mapping, *, unpool: bool, pool: bool) -> Tuple[bool, bool]:
    """The selected sites, once every TP rank has agreed it can build their kernels.

    Collective over the TP group.

    Raises:
        ValueError: if a selected site's kernel cannot be built on some rank (on
            every rank alike).
    """
    local = {}
    if unpool:
        local["unpool"] = _unpool.available()
    if pool:
        local["pool"] = _pool.available()
    everyone = Distributed.get(mapping).tp_allgather(local, small_payload=True)
    for site, needs in (
        ("unpool", "CuTe DSL and the built-in cluster boundary"),
        ("pool", "CuTe DSL and flashinfer's RMSNormFP4QuantKernel"),
    ):
        if site in local and not all(ranks[site] for ranks in everyone):
            raise ValueError(
                f"[musespark_open] projection_gathers ({site}) needs {needs} on every TP rank"
            )
    return unpool, pool


def install(model: MuseSparkOpenModel, *, unpool: bool, pool: bool) -> dict:
    """Re-class every layer's decoder, MoE and post-FFN join; build the inboxes.

    Returns the keyword arguments ``DecodeTarget`` passes to every layer (empty
    when no site is fused, in which case nothing is re-classed). Collective over
    the TP group (``fusable_sites`` and the inboxes, which are fabric memory).

    Raises:
        ValueError: if the sharded projections are off, a kernel cannot be
            built on some rank, or a module is not exactly the built-in class.
    """
    layers = list(model.layers)
    for layer in layers:
        moe = layer.feed_forward
        if moe._pool_gather_workspace is None or moe._unpool_gather_workspace is None:
            raise ValueError(
                "[musespark_open] projection_gathers fuses the sharded projections' gathers; it needs "
                "MUSESPARK_OPEN_SHARDED_PROJECTIONS=1 (TP8/EP8 or single-node TP4/EP4; on "
                "sm107 with MUSESPARK_OPEN_SM107_FUSED_KERNELS=1)"
            )
        for module, classes in (
            (layer, (MuseSparkOpenDecoderLayer,)),
            (moe, tuple(_MOE_CLASSES)),
            (layer.add_ffn, (MuseSparkOpenGatedResidual,)),
        ):
            if type(module) not in classes:
                raise ValueError(
                    f"[musespark_open] projection_gathers re-classes the built-in {classes[0].__name__}, "
                    f"got {type(module).__name__} in layer {layer.layer_idx}"
                )
    mapping = layers[0].mapping
    world = mapping.tp_size
    unpool, pool = fusable_sites(mapping, unpool=unpool, pool=pool)
    if not (unpool or pool):
        logger.warning(
            "[musespark_open] projection_gathers: no site is fused; every layer keeps the built-in path"
        )
        return {}
    unpool_inbox = (
        ProjectionInbox(mapping, row_words=_unpool.local_columns(world) // 2) if unpool else None
    )
    pool_inbox = (
        ProjectionInbox(mapping, row_words=_pool.local_columns(world) // 2) if pool else None
    )
    for layer in layers:
        layer.__class__ = MuseSparkOpenTepDecoderLayer
        moe = layer.feed_forward
        moe.__class__ = _MOE_CLASSES[type(moe)]
        moe._defer_unpool = False
        moe._defer_pool = False
        moe._pool_inbox = pool_inbox
        layer.add_ffn.__class__ = MuseSparkOpenTepGatedResidual
        layer.add_ffn._gather_inbox = unpool_inbox
        layer.add_ffn._gather = moe._unpool_gather_workspace
    sites = [
        name
        for name, on in (
            ("unpool -> post-FFN boundary", unpool),
            ("pool -> pre-expert FP4 quant", pool),
        )
        if on
    ]
    logger.info(
        f"[musespark_open] projection_gathers: TP{world} sharded-projection gathers fused on the "
        f"decode path ({'; '.join(sites)}), steps of at most {MAX_ROWS} rows"
    )
    layer_kwargs = {}
    if unpool:
        layer_kwargs["fused_unpool_gather"] = True
    if pool:
        layer_kwargs["fused_pool_gather"] = True
    return layer_kwargs
