# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``decode_qkv_mxfp8``: the QKV projections of published decode steps on an MXFP8 GEMM (sm107).

A lossy lever (``levers.RULES``), on only by its knob. Needs ``context_fp8_dense`` with its
QKV copies (``qkv`` among its PROJECTIONS), ``projection_publish`` and ``batch_dense_gemms``.

On a decode step that ``projection_publish`` serves (33..448 rows), the post-FFN join of layer
``l`` hands layer ``l + 1`` its BF16 attention input. With this path that join is
``unpool_publish_join_mxfp8``: the served join, which also writes the MXFP8 quantization of
the row it hands on (E4M3 values, one UE8M0 scale per 1x32 block, 128x4-swizzled).
``DecodeQkvMxfp8.offer`` records those operands for layer ``l + 1``'s QKV module. When
``batch_dense_gemms`` routes that module's call (``BatchDenseGemms.projection``) with the
very BF16 tensor the join returned, ``DecodeQkvMxfp8.take`` runs cuBLASLt's block-scaled
MXFP8 GEMM (``torch._scaled_mm``) on them with ``context_fp8_dense``'s MXFP8 copy of the
weight, in place of the BF16 GEMM. A call that does not match its record (another tensor or
shape) runs the BF16 GEMM; a record that no call took is dropped when the forward ends
(``discard``) and counted. Layer 0's QKV projection (its input comes from the embedding
norm), every prefill step and every decode step outside the published rows keep the BF16 GEMM.

At these rows the QKV GEMM is latency-bound: MXFP8 halves its weight bytes and doubles its
MMA rate, while a separate activation quantization pass in the chain would cost most of
that. The join already holds the row; re-reading its own BF16 store after a CTA barrier
keeps the join's registers, so two of its programs still share an SM above 212 rows.
cuBLASLt's MXFP8 kernels launch as PDL dependents under stream capture, as its BF16 ones
do, and need no output memset.

Numerics: lossy (MXFP8 activations and weights on these QKV calls); accuracy-gated, never
on under ``auto``. The join's residual and BF16 output are unchanged bit for bit, and so
is every other projection. No weight memory is added (the copies are ``context_fp8_dense``'s);
a call's values and scales live from its join to its GEMM.

Ownership: one join program per row writes that row's values and scales, which nothing else
writes; the GEMM is launched after the join on the same stream and reads them only after the
join grid completed. The records are host state of one forward (under a CUDA graph, of its
capture): ``offer`` / ``take`` pair layer ``l``'s join with layer ``l + 1``'s QKV call by module
identity and by the BF16 tensor's address and shape. Agreed over the TP group
(``agreed_install``): every rank installs it or none does, so every rank's joins publish and
consume alike.

Install after ``context_fp8_dense`` and before any CUDA graph is captured.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch

from tensorrt_llm.logger import logger

from ... import dispatch as _tp_agreement
from ...ops.comm import projection_publish as _publish

#: The name this path logs and raises under.
NAME = "decode_qkv_mxfp8"
#: The SM version this path is written for (the MXFP8 join and its GEMM operands).
SERVED_SM = 107
#: A TP4 rank's QKV projection: ``[4608, 8192]``.
QKV_SHAPE = (4608, _publish.UNPOOL_HIDDEN)
#: Rows of the eager GEMM ``install`` runs before any capture.
PROBE_ROWS = 136


def gemm(
    values: torch.Tensor, scales: torch.Tensor, weight: torch.Tensor, weight_scales: torch.Tensor
) -> torch.Tensor:
    """``values @ weight^T`` (cuBLASLt block-scaled MXFP8), BF16 ``[M, N]``.

    ``values`` E4M3 ``[M, K]`` and ``weight`` E4M3 ``[N, K]``, each with its 128x4-swizzled
    UE8M0 scales of 1x32 blocks along K.
    """
    return torch._scaled_mm(
        values,
        weight.t(),
        scales.view(torch.float8_e8m0fnu),
        weight_scales.view(torch.float8_e8m0fnu),
        out_dtype=torch.bfloat16,
    )


class DecodeQkvMxfp8:
    """The records between the MXFP8 joins and the next layers' QKV calls of a forward.

    Host state only: under a CUDA graph each record is made and taken during the capture.
    """

    def __init__(self) -> None:
        self._pending: Dict[int, Tuple[int, Tuple[int, ...], torch.Tensor, torch.Tensor]] = {}
        #: Calls per kind: records offered, taken, dropped, declined.
        self.counts: Dict[str, int] = {}
        self._reported = set()

    def _count(self, key: str, n: int = 1) -> None:
        self.counts[key] = self.counts.get(key, 0) + n

    def offer(
        self,
        module: torch.nn.Module,
        next_input: torch.Tensor,
        values: torch.Tensor,
        scales: torch.Tensor,
    ) -> None:
        """Record the MXFP8 operands of ``next_input`` for ``module``'s next call."""
        self._pending[id(module)] = (
            next_input.data_ptr(),
            tuple(next_input.shape),
            values,
            scales,
        )
        self._count("offered")

    def take(self, module: torch.nn.Module, input_tensor) -> Optional[torch.Tensor]:
        """The MXFP8 GEMM of ``module``'s call on ``input_tensor`` from its record, or None.

        None when ``module`` has no record, or when ``input_tensor`` is not the tensor the
        record was made for (the call then runs the BF16 GEMM; the record is consumed either way).
        """
        record = self._pending.pop(id(module), None)
        if record is None:
            return None
        address, shape, values, scales = record
        if (
            not isinstance(input_tensor, torch.Tensor)
            or input_tensor.data_ptr() != address
            or tuple(input_tensor.shape) != shape
        ):
            self._count("declined")
            if "declined" not in self._reported:
                self._reported.add("declined")
                logger.warning(
                    f"[musespark_open] {NAME}: a QKV call whose input is not the join's next input "
                    f"({tuple(getattr(input_tensor, 'shape', ()))} vs {shape}); it runs the BF16 GEMM"
                )
            return None
        copy = module.__dict__["_context_fp8"]
        out = gemm(values, scales, copy.weight, copy.scale)
        self._count("taken")
        if shape[0] not in self._reported:
            self._reported.add(shape[0])
            logger.info(
                f"[musespark_open] {NAME}: a {shape[0]}-row decode step runs its QKV projections "
                "on the MXFP8 GEMM"
            )
        return out

    def discard(self) -> None:
        """Drop the records no call took (end of a forward)."""
        if not self._pending:
            return
        self._count("dropped", len(self._pending))
        if "dropped" not in self._reported:
            self._reported.add("dropped")
            logger.warning(
                f"[musespark_open] {NAME}: {len(self._pending)} MXFP8 join output(s) of a forward "
                "reached no QKV call; those layers ran the BF16 GEMM"
            )
        self._pending.clear()

    def census(self) -> str:
        """One line of what the forwards did (tests)."""
        return ", ".join(f"{k}={v}" for k, v in sorted(self.counts.items()))


def _check_copy(index: int, module) -> object:
    copy = module.__dict__.get("_context_fp8")
    if (
        copy is None
        or (copy.out_features, copy.in_features) != QKV_SHAPE
        or copy.weight.dtype != torch.float8_e4m3fn
        or not copy.weight.is_contiguous()
        or copy.scale.numel() != _publish.mxfp8_scale_bytes(QKV_SHAPE[0], QKV_SHAPE[1])
    ):
        raise ValueError(
            f"[musespark_open] {NAME}: layer {index}'s QKV projection carries no "
            f"{QKV_SHAPE} MXFP8 copy; it needs context_fp8_dense with qkv among its PROJECTIONS"
        )
    return copy


def install(text_model, mapping, compute_capability: Optional[Tuple[int, int]]):
    """Route the published decode steps' QKV projections of layers 1.. through the MXFP8 GEMM.

    Returns the ``DecodeQkvMxfp8`` it installed. Collective over the TP group (the outcome
    is agreed): raises ValueError on every rank alike when some rank cannot serve it
    (not sm107; no ``projection_publish`` or
    ``batch_dense_gemms``; a QKV projection without ``context_fp8_dense``'s MXFP8 copy; or an
    eager probe GEMM that fails or disagrees with the CuTe MXFP8 GEMM).
    """
    sm = _tp_agreement.sm_of(compute_capability)

    def build() -> DecodeQkvMxfp8:
        if sm != SERVED_SM:
            raise ValueError(f"[musespark_open] {NAME}: written for sm{SERVED_SM}, got sm{sm}")
        layers = list(text_model.layers)
        published = getattr(layers[0].add_ffn, "_published", None)
        batch = getattr(layers[0].self_attn, "_batch_dense_gemms", None)
        if published is None or batch is None:
            raise ValueError(
                f"[musespark_open] {NAME}: needs projection_publish and batch_dense_gemms "
                f"(installed: projection_publish={published is not None}, "
                f"batch_dense_gemms={batch is not None})"
            )
        targets = []
        for index in range(len(layers) - 1):
            qkv = layers[index + 1].self_attn.qkv_proj
            targets.append((layers[index].add_ffn, qkv, _check_copy(index + 1, qkv)))
        copy = targets[0][2]
        gen = torch.Generator(device=copy.weight.device).manual_seed(PROBE_ROWS)
        x = torch.randn((PROBE_ROWS, QKV_SHAPE[1]), generator=gen, device=copy.weight.device).to(
            torch.bfloat16
        )
        try:
            values, scales = _publish.mxfp8_reference(x)
            out = gemm(values, scales, copy.weight, copy.scale).float()
            ref = copy.project(values, scales).float()
            torch.cuda.synchronize()
        except Exception as error:  # noqa: BLE001 - the failure must reach the agreement
            raise RuntimeError(
                f"[musespark_open] {NAME}: the MXFP8 GEMM probe failed: {error!r}"
            ) from error
        cosine = float((out * ref).sum() / (out.norm() * ref.norm()))
        if not math.isfinite(cosine) or 1.0 - cosine > 1e-4:
            raise RuntimeError(
                f"[musespark_open] {NAME}: cuBLASLt's MXFP8 GEMM disagrees with the CuTe MXFP8 GEMM "
                f"on the probe (cosine {cosine})"
            )
        switch = DecodeQkvMxfp8()
        for join, qkv, _ in targets:
            join._decode_qkv_target = qkv
        published.decode_qkv = switch
        batch.decode_qkv = switch
        logger.info(
            f"[musespark_open] {NAME}: the published decode steps' post-FFN joins of "
            f"{len(targets)} layers quantize the next input to MXFP8 and layers 1..{len(layers) - 1} "
            "run their QKV projections on cuBLASLt's MXFP8 GEMM"
        )
        return switch

    return _tp_agreement.agreed_install(mapping, build, (ValueError, RuntimeError))
