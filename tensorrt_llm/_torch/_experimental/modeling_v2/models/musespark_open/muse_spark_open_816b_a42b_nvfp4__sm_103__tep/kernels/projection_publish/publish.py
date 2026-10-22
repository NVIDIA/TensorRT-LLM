# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fast path ``projection_publish``: decode steps of 33..448 rows (5 to 56 requests at K = 7)
on column-sharded pool / unpool projections that publish from the GEMM, sm103 TEP4.

From 33 rows the built-in MoE multiplies the whole replicated pool (8192 -> 4096) and
unpool (4096 -> 8192) weights on every rank (67 MB each per rank), and ``projection_gathers``
/ ``wide_projection_gathers`` stop at 8 / 32 rows. While ``DecodeTarget`` arms this path for
a decode forward of 33..448 rows (``PublishedProjections.armed``):

* the pool and unpool projections of every publishing layer multiply the rank's column
  slice (16.8 MB) with the slot of the site's multicast transport as the GEMM's output
  (``ops/comm/projection_publish``): the GEMM's epilogue stores are the publication;
* the pre-expert RMSNorm + NVFP4 quantization (``ops/comm/pool_publish_fp4quant``, on the
  MoE preparation stream where the built-in runs it) and the post-FFN join
  (``projection_publish.unpool_publish_join``) poll their rows of the transport, reset
  them, and compute the built-in kernels' arithmetic on the gathered row.

Publishing layers: the pool site of every layer whose built-in pre-expert input is
flashinfer's fused NVFP4 quantization (the NVFP4-expert layers; the BF16-expert layers
keep the built-in path), the unpool site of every layer whose post-FFN join feeds the next
layer's norm (every layer but the last). Each site's calls of a step form one cycle on its
transport's buffers (``buffer_sequence``).

The transports (11.0 MB pool + 22.0 MB unpool per rank at 448 rows; the build logs the
device memory they take, which the multicast allocation rounds up) are built at the first
armed forward, which the engine runs eagerly (its CUDA-graph warm-up): a deployment whose
decode steps never reach 33 rows never allocates them. A forward captured before they
exist runs unarmed, on every rank alike. The build refuses, on every rank alike, and the
steps run the built-in projections when the cuBLAS producer is not the one the census
certified (``projection_publish.producer_refusal``) or a transport cannot be allocated.

``dense_gemm_prefetch`` composes: no dispatcher has a kernel at 33..448 rows, so none
forks a prefetch on these steps, and the slice GEMM first joins the plan's outstanding
forks as the site's dispatcher would (``_join_named_forks``).

Numerics: the consumers are the built-in kernels' bits on the gathered rows. The slice
GEMMs are another cuBLAS kernel than the replicated GEMM's (FP32 accumulation, rounded
once): their outputs differ from the built-in's by the summation order only.

Every rank must take the same path at every call: a rank whose GEMM published while a
peer's did not would leave the peers' consumers polling. Arming reads only the step's row
count; the per-layer choice is fixed at the build, agreed over the TP group; a call that
does not fit the armed shape raises on every rank alike instead of falling back.
"""

from __future__ import annotations

import contextlib
from typing import Iterator, List, Optional, Tuple

import torch

from tensorrt_llm._torch.distributed.communicator import Distributed
from tensorrt_llm._torch.utils import Fp4QuantizedTensor
from tensorrt_llm.logger import logger

from ... import dispatch as _tp_agreement
from ...ops.comm import pool_publish_fp4quant as _pool
from ...ops.comm import projection_publish as _publish
from ..projection_gathers.gathers import MuseSparkOpenTepGatedResidual, MuseSparkOpenTepGatherMoE

#: The rows of the decode steps this path arms. Above 448 rows the published join costs more
#: than the slice GEMMs save (generation-worker decode steps of 57..64 requests), so those
#: steps keep the built-in projections; the transports are sized to this bound.
MIN_ROWS, MAX_ROWS = _publish.MIN_ROWS, 448


def _pool_args(moe) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """(gain, global scale) exactly when the built-in ``prepare_expert_input`` would quantize a
    BF16 ``[33..448, 4096]`` pool output with flashinfer's fused ``rmsnorm_fp4quant``; else None.

    The built-in's conditions that do not depend on the call's tensor (which is a fresh,
    contiguous BF16 CUDA tensor of 33..448 rows on the armed path). Reads nothing that
    differs between ranks.
    """
    from tensorrt_llm._torch.modules.fused_ops.rmsnorm_fp4_quant import rmsnorm_fp4_quant_available
    from tensorrt_llm._torch.moe.fused_moe.moe_op_backend import TRTLLMOpBackend

    backend = getattr(moe.experts, "backend", moe.experts)
    if (
        not getattr(moe, "_pre_expert_quant_fused", True)
        or getattr(moe.experts, "comm", None) is not None
        or not isinstance(getattr(backend, "op_backend", None), TRTLLMOpBackend)
        or not getattr(backend, "has_nvfp4", False)
        or getattr(backend, "fc31_act_scale", None) is not None
        or MAX_ROWS > getattr(moe.experts, "moe_max_num_tokens", MAX_ROWS)
        or moe.expert_hidden_size != _pool.HIDDEN
        or not rmsnorm_fp4_quant_available()
    ):
        return None
    scale = getattr(backend, "fc31_input_scale", None)
    if scale is None or scale.numel() != 1 or scale.dtype != torch.float32:
        return None
    if moe._pre_expert_quant_gain is None:
        # The built-in's lazily cached gain, computed the built-in's way.
        moe._pre_expert_quant_gain = (moe.pre_expert_norm.weight.float() + 1).to(torch.bfloat16)
    gain = moe._pre_expert_quant_gain
    if (
        gain.dtype != torch.bfloat16
        or tuple(gain.shape) != (_pool.HIDDEN,)
        or not gain.is_contiguous()
    ):
        return None
    return gain, scale.reshape(1)


def _join_fits(join, next_norm, gain) -> bool:
    """Whether the built-in join of 33..448 rows runs its Triton kernel (``_fused_enabled`` and
    the operands' dtypes); the call's residual is checked per call."""
    return (
        bool(getattr(join, "_fused_enabled", False))
        and next_norm.use_gemma
        and next_norm.weight.dtype == torch.bfloat16
        and next_norm.weight.is_contiguous()
        and tuple(next_norm.weight.shape) == (_publish.UNPOOL_HIDDEN,)
        and (
            gain is None
            or (
                gain.dtype == torch.bfloat16
                and gain.is_contiguous()
                and tuple(gain.shape) == (_publish.UNPOOL_HIDDEN,)
            )
        )
    )


def _join_named_forks(module) -> None:
    """Join the dense GEMM plan's forks the site's dispatcher would have joined before its GEMM."""
    dispatcher = module.__dict__.get("_low_m_dispatcher")
    site = getattr(dispatcher, "site", None)
    if site is None:
        return
    site.plan.join()
    if site.plan.named_forks:
        site.plan.join_role(site.role)


class PublishedProjections:
    """The decode steps of 33..448 rows on the published projections: armed per forward by
    ``DecodeTarget``.

    Logs once per row count and site that an armed call published, so a served log shows
    what the captured decode steps run.
    """

    def __init__(self, model, comm) -> None:
        self._layers = list(model.layers)
        self.mapping = self._layers[0].mapping
        self._comm = comm
        self.pool: Optional[_publish.PublishTransport] = None
        self.unpool: Optional[_publish.PublishTransport] = None
        self.pool_layers: Tuple[int, ...] = ()
        self.unpool_layers: Tuple[int, ...] = ()
        self._built = False
        self._refused: Optional[str] = None
        self._armed = False
        self._rows = 0
        self._reported = set()
        #: Slots published in the current forward and not consumed yet.
        self._unconsumed = 0

    @property
    def is_armed(self) -> bool:
        return self._armed

    def takes(self, rows: int) -> bool:
        """Whether a decode forward of ``rows`` rows runs on the published projections.

        33..448 rows, unless the build refused, or the forward is captured before the first
        eager armed forward built the transports (every rank captures the same graphs, so
        every rank answers alike).
        """
        if not MIN_ROWS <= rows <= MAX_ROWS or self._refused is not None:
            return False
        if not self._built and (
            torch.cuda.is_current_stream_capturing() or torch.compiler.is_compiling()
        ):
            logger.warning_once(
                "[musespark_open] projection_publish: a decode forward of 33..448 rows was captured "
                "before an eager one built the transports; it runs the built-in projections",
                key="musespark_open.projection_publish.unbuilt",
            )
            return False
        return True

    @contextlib.contextmanager
    def armed(self, rows: int) -> Iterator[None]:
        """Arm every publishing layer for one decode forward of ``rows`` rows (building the
        transports first if this is the first armed forward); disarm on any exit.

        Raises:
            RuntimeError: if the forward returned with a published slot that no consumer took
                (a slot is never read through the multicast address, and an unconsumed one
                would hold its data into the buffer's next call).
        """
        if not self._built:
            self._build()
        if self._refused is not None:
            yield
            return
        self._armed = True
        self._rows = int(rows)
        self._unconsumed = 0
        try:
            yield
            if self._unconsumed:
                raise RuntimeError(
                    f"[musespark_open] projection_publish: a {rows}-row decode forward returned with "
                    f"{self._unconsumed} published slot(s) no consumer took"
                )
        finally:
            self._armed = False
            self._rows = 0
            self._unconsumed = 0

    def _build(self) -> None:
        """Choose the publishing layers, build both transports, assign buffers (collective).

        Every refusal is agreed over the TP group: the layer choice and the producer check
        through ``agreed_refusal``, a transport that cannot be built because its
        construction raises on every rank alike.
        """
        self._built = True
        layers = self._layers
        pool_layers: List[int] = []
        unpool_layers: List[int] = []
        for index, layer in enumerate(layers):
            moe = layer.feed_forward
            if isinstance(moe, MuseSparkOpenTepGatherMoE) and _pool_args(moe) is not None:
                pool_layers.append(index)
            if index + 1 < len(layers) and isinstance(layer.add_ffn, MuseSparkOpenTepGatedResidual):
                gain = (
                    layer.post_ffn_norm.gain()
                    if callable(getattr(layer.post_ffn_norm, "gain", None))
                    else None
                )
                if _join_fits(layer.add_ffn, layers[index + 1].pre_qkv_proj_norm, gain):
                    unpool_layers.append(index)
        local = (tuple(pool_layers), tuple(unpool_layers))
        everyone = Distributed.get(self.mapping).tp_allgather(local, small_payload=True)
        reason = None
        if any(tuple(map(tuple, other)) != local for other in everyone):
            reason = "the TP ranks chose different publishing layers"
        elif len(pool_layers) < 2 or len(unpool_layers) < 2:
            reason = (
                f"fewer than two publishing layers (pool {len(pool_layers)}, unpool "
                f"{len(unpool_layers)})"
            )
        else:
            reason = _publish.producer_refusal(torch.device("cuda", torch.cuda.current_device()))
        reason = _tp_agreement.agreed_refusal(self.mapping, reason)
        if reason is None:
            try:
                self.pool = _publish.PublishTransport(
                    self.mapping,
                    _publish.local_columns(_publish.POOL_HIDDEN),
                    MAX_ROWS,
                    comm=self._comm,
                )
                self.unpool = _publish.PublishTransport(
                    self.mapping,
                    _publish.local_columns(_publish.UNPOOL_HIDDEN),
                    MAX_ROWS,
                    comm=self._comm,
                )
            except RuntimeError as error:
                # PublishTransport agrees its outcome over the group before it raises, so
                # every rank lands here alike.
                self.pool = self.unpool = None
                reason = f"a transport could not be built: {error}"
        if reason is not None:
            self._refused = reason
            logger.warning(
                f"[musespark_open] projection_publish: not armed ({reason}); 33..448-row decode "
                "steps run the built-in projections"
            )
            return
        for index, buffer in zip(pool_layers, _publish.buffer_sequence(len(pool_layers))):
            layers[index].feed_forward._published_pool_buffer = buffer
        for index, buffer in zip(unpool_layers, _publish.buffer_sequence(len(unpool_layers))):
            layers[index].feed_forward._published_unpool_buffer = buffer
            layers[index].add_ffn._published_unpool_buffer = buffer
        self.pool_layers = tuple(pool_layers)
        self.unpool_layers = tuple(unpool_layers)
        self.pool.start_guard_watch("pool")
        self.unpool.start_guard_watch("unpool")
        logger.info(
            f"[musespark_open] projection_publish: decode steps of {MIN_ROWS}..{MAX_ROWS} rows publish "
            f"the pool slice from {len(pool_layers)} layers and the unpool slice from "
            f"{len(unpool_layers)} layers into multicast transports "
            f"({(self.pool.nbytes + self.unpool.nbytes) / 2**20:.1f} MiB per rank, "
            f"{(self.pool.allocated + self.unpool.allocated) / 2**20:.1f} MiB of device memory)"
        )

    def _report(self, site: str, rows: int) -> None:
        if (site, rows) in self._reported:
            return
        self._reported.add((site, rows))
        logger.info(
            f"[musespark_open] projection_publish: a {rows}-row decode step publishes the {site} "
            "slice from its GEMM"
        )

    def _check_rows(self, tensor: torch.Tensor, what: str) -> int:
        rows = tensor.shape[0] if tensor.dim() == 2 else -1
        if rows != self._rows or tensor.dtype != torch.bfloat16 or not tensor.is_contiguous():
            raise RuntimeError(
                f"[musespark_open] projection_publish: armed for {self._rows} rows, the {what} got "
                f"{tuple(tensor.shape)} {tensor.dtype} contiguous={tensor.is_contiguous()}"
            )
        return rows

    def project(self, moe, module, hidden_states: torch.Tensor) -> Optional[torch.Tensor]:
        """The slot the rank's slice GEMM of ``module`` just published into, or None when this
        layer's site does not publish."""
        if module is moe.pooler:
            site, transport = "pool", self.pool
            buffer = getattr(moe, "_published_pool_buffer", None)
        elif module is moe.unpooler:
            site, transport = "unpool", self.unpool
            buffer = getattr(moe, "_published_unpool_buffer", None)
        else:
            return None
        if buffer is None:
            return None
        rows = self._check_rows(hidden_states, f"{site} projection")
        slot = transport.slot(buffer, rows)
        columns = transport.columns
        begin = self.mapping.tp_rank * columns
        _join_named_forks(module)
        _publish.publish_projection(hidden_states, module.weight[begin : begin + columns], slot)
        self._unconsumed += 1
        self._report(site, rows)
        return slot

    def consume_pool(self, moe, pooled: torch.Tensor) -> Optional[Fp4QuantizedTensor]:
        """The pre-expert NVFP4 input from the published pool slot ``pooled``, or None when
        ``pooled`` is not this layer's slot."""
        buffer = getattr(moe, "_published_pool_buffer", None)
        if buffer is None or not self.pool.is_slot(pooled, buffer, self._rows):
            return None
        args = _pool_args(moe)
        if args is None:
            raise RuntimeError(
                f"[musespark_open] projection_publish: layer {moe.layer_idx}'s pool slice was "
                "published but its pre-expert quantization no longer fits"
            )
        gain, global_scale = args
        fp4, scale = _pool.pool_publish_rmsnorm_fp4quant(
            self.pool, buffer, self._rows, gain, global_scale, moe.pre_expert_norm.variance_epsilon
        )
        self._unconsumed -= 1
        return Fp4QuantizedTensor(fp4, scale, is_sf_swizzled=False)

    def consume_unpool(
        self,
        join,
        residual: torch.Tensor,
        branch: torch.Tensor,
        eps: float,
        next_norm,
        gain: Optional[torch.Tensor],
    ):
        """The post-FFN join's ``(updated, next_input)`` from the published unpool slot
        ``branch``, or None when ``branch`` is not this layer's slot."""
        buffer = getattr(join, "_published_unpool_buffer", None)
        if buffer is None or not self.unpool.is_slot(branch, buffer, self._rows):
            return None
        rows = self._rows
        if not (
            _join_fits(join, next_norm, gain)
            and residual.dim() == 2
            and tuple(residual.shape) == (rows, _publish.UNPOOL_HIDDEN)
            and residual.dtype == torch.float32
            and residual.is_contiguous()
        ):
            raise RuntimeError(
                "[musespark_open] projection_publish: the post-FFN join was handed a published "
                f"unpool slot but the call is not the built-in Triton join's (residual "
                f"{tuple(residual.shape)} {residual.dtype}); the unpool GEMM already published"
            )
        alpha, beta = join.coefficients()
        joined = _publish.unpool_publish_join(
            self.unpool,
            buffer,
            rows,
            residual,
            gain,
            alpha,
            beta,
            next_norm.weight,
            eps,
            next_norm.variance_epsilon,
            join._preqkv_norm_fp32,
        )
        self._unconsumed -= 1
        return joined


def check(model, layer_kwargs: dict) -> None:
    """Whether ``install`` may run on ``model``; changes nothing.

    Raises:
        ValueError: if the model is not TP4, the kernels cannot be built here, or the
            projection gathers did not fuse both sites (this path rides their classes).
    """
    layers = list(model.layers)
    mapping = layers[0].mapping
    if mapping.tp_size != _publish.WORLD:
        raise ValueError(
            f"[musespark_open] projection_publish serves TP{_publish.WORLD} (one tray), got TP{mapping.tp_size}"
        )
    if not (_pool.available() and _publish.available()):
        raise ValueError(
            "[musespark_open] projection_publish needs CuTe DSL, flashinfer's RMSNormFP4QuantKernel "
            "and Triton"
        )
    if not (layer_kwargs.get("fused_unpool_gather") and layer_kwargs.get("fused_pool_gather")):
        raise ValueError(
            "[musespark_open] projection_publish defers both projections through projection_gathers, "
            "which did not fuse both sites here"
        )
    for layer in layers:
        if not isinstance(layer.feed_forward, MuseSparkOpenTepGatherMoE) or not isinstance(
            layer.add_ffn, MuseSparkOpenTepGatedResidual
        ):
            raise ValueError(
                f"[musespark_open] projection_publish rides projection_gathers' MoE and join classes, "
                f"which layer {layer.layer_idx} does not run"
            )


def install(model) -> PublishedProjections:
    """Hand one ``PublishedProjections`` to every layer's MoE and post-FFN join.

    Collective over the TP group (the transports' communicator split); call ``check``
    first, agreed over the group. The transports themselves are built at the first armed
    forward.
    """
    from tensorrt_llm._torch.distributed.ops import _get_mnnvl_workspace_comm

    layers = list(model.layers)
    published = PublishedProjections(model, _get_mnnvl_workspace_comm(layers[0].mapping))
    for layer in layers:
        layer.feed_forward._published = published
        layer.add_ffn._published = published
    logger.info(
        f"[musespark_open] projection_publish: decode steps of {MIN_ROWS}..{MAX_ROWS} rows in "
        f"{len(layers)} layers will publish the pool / unpool slices from their GEMMs (transports "
        "built at the first armed forward)"
    )
    return published
