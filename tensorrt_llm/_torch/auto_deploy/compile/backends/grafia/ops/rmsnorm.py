# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Grafia RMSNorm support checks and CTM emission."""

from __future__ import annotations

import importlib
from functools import cache
from typing import Any

import torch
from torch.fx import Node

from ....lowering import (
    LoweringContext,
    ModeContext,
    OpArgumentResolver,
    ProgramData,
    SupportDecision,
    ValueType,
)
from ..constants import RMSNORM_OP_KIND, SUPPORTED_RMSNORM_HIDDEN, SUPPORTED_RMSNORM_ROWS
from ..errors import GrafiaCompileError, GrafiaUnsupportedError
from ..metadata import _ctm_dtype_to_torch, _dtype_from_meta, _is_contiguous_meta, _shape_from_meta
from .base import GrafiaOpLowering

_COMPILE_RESOURCE_ID = "grafia.rmsnorm.cubin"


class RmsNormLowering(GrafiaOpLowering):
    """Grafia lowering for canonical AutoDeploy RMSNorm."""

    @property
    def source_ops(self) -> tuple[Any, ...]:
        return (torch.ops.auto_deploy.torch_rmsnorm.default,)

    def classify_node(
        self,
        node: Node,
        mode: ModeContext,
        _program: ProgramData,
        args: OpArgumentResolver,
    ) -> SupportDecision:
        if not node.users:
            return SupportDecision.eager_only(
                "canonical torch_rmsnorm has no external users and remains eager"
            )
        try:
            self.validate_node_contract(node, args)
        except GrafiaUnsupportedError as exc:
            return SupportDecision.eager_only(str(exc))
        return SupportDecision.supported(
            f"canonical torch_rmsnorm is supported in Grafia {mode.name}"
        )

    def lower(self, ctx: LoweringContext, node: Node) -> Any:
        x, weight, eps = ctx.args.get(node, "input", "weight", "eps")
        return self.emit(
            ctx.adapter,
            ctx.resolve(x),
            ctx.resolve(weight),
            eps=float(eps),
            result_meta=ctx.result_type(node),
            loc=ctx.loc(node),
        )

    def validate_node_contract(self, node: Node, args: OpArgumentResolver) -> None:
        x_node, weight_node, eps = args.get(node, "input", "weight", "eps")
        if not isinstance(x_node, Node) or not isinstance(weight_node, Node):
            raise GrafiaUnsupportedError(
                f"torch_rmsnorm node {node.name!r} requires tensor node inputs"
            )
        if isinstance(eps, bool) or not isinstance(eps, (float, int)):
            raise GrafiaUnsupportedError(
                f"torch_rmsnorm node {node.name!r} requires numeric eps, got {eps!r}"
            )

        _validate_tensor_contract(x_node, weight_node, node)

    def emit(
        self,
        adapter: Any,
        x: Any,
        weight: Any,
        *,
        eps: float,
        result_meta: ValueType,
        loc: Any | None = None,
    ) -> Any:
        adapter.register_compile_resource(
            _COMPILE_RESOURCE_ID,
            cache_key=_default_rmsnorm_cubin_path,
            configure_backend=_configure_rmsnorm_cubin_path,
        )
        shape, dtype = adapter._shape_dtype_from_result_type(result_meta, loc)
        _validate_ctm_specs(adapter, x, weight, shape, dtype, loc)
        if len(shape) < 1:
            raise GrafiaUnsupportedError(
                f"{adapter._region_id}: RMSNorm result {loc!r} must have rank >= 1"
            )
        hidden_size = shape[-1]
        op_id = len(adapter.ops)
        output = adapter._tensor_spec(
            name=str(loc or f"rms_norm_{op_id}"),
            shape=shape,
            dtype=dtype,
            producer_id=op_id,
        )
        attrs = {"hidden_size": hidden_size, "eps": float(eps)}
        adapter.ops.append(
            adapter.spec_mod.CTMOpSpec(
                op_kind=RMSNORM_OP_KIND,
                id=op_id,
                inputs=[x, weight],
                outputs=[output],
                attrs=attrs,
            )
        )
        adapter.op_kinds.append(RMSNORM_OP_KIND)
        adapter._op_cache_records.append(
            (
                RMSNORM_OP_KIND,
                tuple(getattr(tensor, "name", "") for tensor in (x, weight)),
                output.name,
                tuple(shape),
                adapter._dtype_name(dtype),
                tuple(sorted(attrs.items())),
            )
        )
        return output


RMSNORM_LOWERING = RmsNormLowering()


@cache
def _default_rmsnorm_cubin_path() -> str:
    cubin_path = _import_rmsnorm_factory()._default_cubin_path()
    if cubin_path is None:
        raise GrafiaCompileError(
            "compile_backend='grafia' could not find the rmsnorm_rts cubin. "
            "Set GRAFIA_RMSNORM_RTS_CUBIN or DKG_HOME."
        )
    return str(cubin_path)


def _configure_rmsnorm_cubin_path(backend: Any) -> None:
    kernel_config = getattr(backend, "kernel_config", None)
    if isinstance(kernel_config, dict):
        cubin_path = _default_rmsnorm_cubin_path()
        kernel_config.setdefault(RMSNORM_OP_KIND, {})["cubin_path"] = cubin_path


def _import_rmsnorm_factory():
    try:
        return importlib.import_module("backends.ctm.factories.rmsnorm_rts")
    except ImportError as exc:
        raise GrafiaCompileError(
            "compile_backend='grafia' RMSNorm requires the CTM rmsnorm_rts "
            "factory to be importable. Add $GRAFIA_ARM/thin_ir to PYTHONPATH."
        ) from exc


def _validate_tensor_contract(x: Node, weight: Node, out: Node) -> None:
    _verify_grafia_rms_norm(
        input_shape=_shape_from_meta(x),
        weight_shape=_shape_from_meta(weight),
        output_shape=_shape_from_meta(out),
        input_dtype=_dtype_from_meta(x),
        weight_dtype=_dtype_from_meta(weight),
        output_dtype=_dtype_from_meta(out),
        input_contiguous=_is_contiguous_meta(x),
        weight_contiguous=_is_contiguous_meta(weight),
        prefix="compile_backend='grafia' RMSNorm",
    )


def _validate_ctm_specs(
    adapter: Any,
    x: Any,
    weight: Any,
    output_shape: tuple[int, ...],
    output_dtype: torch.dtype,
    loc: Any | None,
) -> None:
    _verify_grafia_rms_norm(
        input_shape=tuple(int(dim) for dim in x.spec.shape),
        weight_shape=tuple(int(dim) for dim in weight.spec.shape),
        output_shape=output_shape,
        input_dtype=_ctm_dtype_to_torch(x.spec.dtype),
        weight_dtype=_ctm_dtype_to_torch(weight.spec.dtype),
        output_dtype=output_dtype,
        prefix=f"{adapter._region_id}: RMSNorm {loc!r}",
    )


def _verify_grafia_rms_norm(
    *,
    input_shape: tuple[int, ...],
    weight_shape: tuple[int, ...],
    output_shape: tuple[int, ...],
    input_dtype: torch.dtype,
    weight_dtype: torch.dtype,
    output_dtype: torch.dtype,
    prefix: str,
    input_contiguous: bool | None = None,
    weight_contiguous: bool | None = None,
) -> None:
    if input_dtype is not torch.bfloat16 or weight_dtype is not torch.bfloat16:
        raise GrafiaUnsupportedError(
            f"{prefix} supports only BF16 input and BF16 weight; "
            f"got input={input_dtype}, weight={weight_dtype}"
        )
    if output_dtype is not torch.bfloat16:
        raise GrafiaUnsupportedError(f"{prefix} output must be BF16, got {output_dtype}")
    if len(input_shape) < 1:
        raise GrafiaUnsupportedError(f"{prefix} input must be ranked")
    if input_shape[-1] != SUPPORTED_RMSNORM_HIDDEN:
        raise GrafiaUnsupportedError(
            f"{prefix} supports hidden size {SUPPORTED_RMSNORM_HIDDEN}, got {input_shape[-1]}"
        )

    rows = 1
    for dim in input_shape[:-1]:
        rows *= dim
    if rows != SUPPORTED_RMSNORM_ROWS:
        raise GrafiaUnsupportedError(
            f"{prefix} supports flattened rows {SUPPORTED_RMSNORM_ROWS}, got {rows}"
        )
    if weight_shape != (SUPPORTED_RMSNORM_HIDDEN,):
        raise GrafiaUnsupportedError(
            f"{prefix} weight shape must be ({SUPPORTED_RMSNORM_HIDDEN},), got {weight_shape}"
        )
    if output_shape != input_shape:
        raise GrafiaUnsupportedError(
            f"{prefix} output shape {output_shape} must match input shape {input_shape}"
        )
    if input_contiguous is False or weight_contiguous is False:
        raise GrafiaUnsupportedError(f"{prefix} requires contiguous input and weight metadata")
