# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""NATIVE_ADD_RMSNORM: the M3.1 residual add + RMSNorm of BF16 rows of width 6144 by the CUDA C++ op
``mv2_minimax_m3::minimax_m31_add_rmsnorm``, with the call signatures of the CuTe DSL kernels it replaces
(``kernels.megamoe.add_rmsnorm_mxfp8``: ``add_rmsnorm_mxfp8`` / ``add_rmsnorm_replicated`` /
``add_rmsnorm_replicated_mxfp8``). Bit for bit those kernels and FlashInfer's ``fused_add_rmsnorm`` (the plain
form).

A call the op does not take (a width other than 6144, rows that are not contiguous 16-byte-aligned BF16, more than
65535 rows, a device other than sm_107, torch.compile, the op not built) or ``TRTLLM_M31_DISABLE_NATIVE_ADD_RMSNORM=1``
(read at import) runs the CuTe DSL kernel (FlashInfer for the plain form), as before.

``install_plain`` routes chosen ``RMSNorm`` instances' fused add + RMSNorm (``RMSNorm.forward``'s FlashInfer branch)
to the plain form, per instance; a target picks the instances (the DEP target: its model's norms that run that branch,
after its own hand-off installs).

NORM_L2PF: ``add_rmsnorm_mxfp8(..., prefetch=...)`` also has the op's CTAs pull the first bytes of up to two
ranges (the consumer GEMM's weight and weight scales) into L2 before its PDL wait, so the GEMM after it reads them from
L2. The outputs do not change (bitwise). ``l2pf_ranges`` is the band rule (L2PF_MIN_ROWS..L2PF_MAX_ROWS rows, 1 /
L2PF_DIVISOR of each range); a target owns the switch and picks the sites.
"""

from __future__ import annotations

import types
from typing import Callable, Iterable, Optional

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
from tensorrt_llm._torch._experimental.modeling_v2.catalog.norm.flashinfer_fused_add_rmsnorm import (
    FlashinferFusedAddRmsnorm,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.m31_switches import (
    m31_disabled,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.native import family_ops
from tensorrt_llm._torch.modules.rms_norm import RMSNorm
from tensorrt_llm._torch.utils import is_torch_compiling

NATIVE_ADD_RMSNORM = not m31_disabled("NATIVE_ADD_RMSNORM")
HIDDEN = 6144
#: NORM_L2PF band: the norm -> GEMM chain gains at 88 / 128 rows and not at <= 64.
L2PF_MIN_ROWS = 65
L2PF_MAX_ROWS = 128
#: The share of each range prefetched: the read's f25 (the first quarter of the weight and of its scales).
L2PF_DIVISOR = 4
_OPS: list = []
_SM107: dict = {}
_PREFETCH: list = []


def add_rmsnorm_op():
    """``mv2_minimax_m3::minimax_m31_add_rmsnorm`` when it is on and this build carries it, else None (looked up
    once)."""
    if not _OPS:
        _OPS.append(
            getattr(family_ops(), "minimax_m31_add_rmsnorm", None) if NATIVE_ADD_RMSNORM else None
        )
    return _OPS[0]


def takes_prefetch() -> bool:
    """Whether the op carries the NORM_L2PF prefetch arguments (looked up once)."""
    if not _PREFETCH:
        op = add_rmsnorm_op()
        _PREFETCH.append(op is not None and "prefetch0" in str(op.default._schema))
    return _PREFETCH[0]


def l2pf_ranges(rows: int, *ranges: torch.Tensor) -> Optional[tuple]:
    """The ``prefetch`` argument of ``add_rmsnorm_mxfp8`` for a call of ``rows`` rows whose consumer reads ``ranges``
    (at most two tensors: its weight and weight scales): the first 1 / L2PF_DIVISOR of each (16-byte multiple) when
    L2PF_MIN_ROWS <= rows <= L2PF_MAX_ROWS and the op takes it, else None (no prefetch)."""
    if (
        not L2PF_MIN_ROWS <= rows <= L2PF_MAX_ROWS
        or not 0 < len(ranges) <= 2
        or not takes_prefetch()
    ):
        return None
    return tuple((t, (t.numel() * t.element_size() // L2PF_DIVISOR) & ~15) for t in ranges)


def _sm107(device: torch.device) -> bool:
    if device not in _SM107:
        _SM107[device] = torch.cuda.get_device_capability(device) == (10, 7)
    return _SM107[device]


def takes(input: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor) -> bool:
    """Whether the op serves the add + RMSNorm of these operands (see the module docstring)."""
    if add_rmsnorm_op() is None or is_torch_compiling():
        return False
    if (
        input.dim() != 2
        or input.shape[1] != HIDDEN
        or input.shape[0] > 65535
        or residual.shape != input.shape
    ):
        return False
    if weight.shape != (HIDDEN,) or not input.is_cuda or not _sm107(input.device):
        return False
    return all(
        t.dtype == torch.bfloat16 and t.is_contiguous() and t.data_ptr() % 16 == 0
        for t in (input, residual, weight)
    )


def _cute():
    from ..kernels.megamoe import add_rmsnorm_mxfp8 as cute

    return cute


def add_rmsnorm_mxfp8(
    input, residual, weight, eps, q, sf, *, enable_pdl: bool, prefetch=None
) -> None:
    """``kernels.megamoe.add_rmsnorm_mxfp8``, bit for bit; ``prefetch`` (``l2pf_ranges``' value, or None) warms L2
    for the consumer GEMM on the op's path (the CuTe DSL fallback ignores it)."""
    if takes(input, residual, weight):
        if prefetch:
            ranges = [arg for t, nbytes in prefetch for arg in (t, nbytes)]
            ranges += [None, 0] * (2 - len(prefetch))
            add_rmsnorm_op()(input, residual, weight, eps, q, sf, None, enable_pdl, *ranges)
        else:
            add_rmsnorm_op()(input, residual, weight, eps, q, sf, None, enable_pdl)
    else:
        _cute().add_rmsnorm_mxfp8(input, residual, weight, eps, q, sf, enable_pdl=enable_pdl)


def add_rmsnorm_replicated(input, residual, weight, eps, replicas, *, enable_pdl: bool) -> None:
    """``kernels.megamoe.add_rmsnorm_replicated``, bit for bit."""
    if takes(input, residual, weight):
        add_rmsnorm_op()(input, residual, weight, eps, None, None, replicas, enable_pdl)
    else:
        _cute().add_rmsnorm_replicated(
            input, residual, weight, eps, replicas, enable_pdl=enable_pdl
        )


def add_rmsnorm_replicated_mxfp8(
    input, residual, weight, eps, replicas, q, sf, *, enable_pdl: bool
) -> None:
    """``kernels.megamoe.add_rmsnorm_replicated_mxfp8``, bit for bit."""
    if takes(input, residual, weight):
        add_rmsnorm_op()(input, residual, weight, eps, q, sf, replicas, enable_pdl)
    else:
        _cute().add_rmsnorm_replicated_mxfp8(
            input, residual, weight, eps, replicas, q, sf, enable_pdl=enable_pdl
        )


def fused_add_rmsnorm(
    input: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    enable_pdl: Optional[bool] = None,
) -> bool:
    """The plain form (FlashInfer's ``fused_add_rmsnorm``, bit for bit) in place when the op takes the operands;
    False (nothing written) otherwise, and the caller runs FlashInfer."""
    if not takes(input, residual, weight):
        return False
    if enable_pdl is None:
        from tensorrt_llm._torch.flashinfer_utils import get_env_enable_pdl

        enable_pdl = get_env_enable_pdl()
    add_rmsnorm_op()(input, residual, weight, eps, None, None, None, enable_pdl)
    return True


def _plain_forward(
    self: RMSNorm,
    hidden_states: torch.Tensor,
    residual=RMSNorm._ARGUMENT_NOT_SPECIFIED_SENTINEL,
    *,
    return_norm_out: bool = False,
):
    """``RMSNorm.forward``, the fused add + RMSNorm by the plain form where the op takes it (bit for bit the FlashInfer
    branch it replaces, which ignores ``return_norm_out`` too)."""
    if (
        isinstance(residual, torch.Tensor)
        and not (self.is_nvfp4 and self.nvfp4_scale is not None)
        and fused_add_rmsnorm(hidden_states, residual, self.weight, self.variance_epsilon)
    ):
        return hidden_states, residual
    return RMSNorm.forward(self, hidden_states, residual, return_norm_out=return_norm_out)


def is_plain_site(norm: torch.nn.Module) -> bool:
    """Whether ``install_plain`` wrapped ``norm``."""
    return getattr(norm.__dict__.get("forward"), "__func__", None) is _plain_forward


def uninstall_plain(modules: Iterable[torch.nn.Module]) -> None:
    """Remove ``install_plain``'s wrappers (before a target re-runs its own forward-replacing installs)."""
    for module in modules:
        if is_plain_site(module):
            del module.__dict__["forward"]


def install_plain(
    modules: Iterable[torch.nn.Module], decline_reason: Callable[[torch.nn.Module], Optional[str]]
) -> int:
    """Wrap the forward of every ``RMSNorm`` of width 6144 in ``modules`` whose forward is not already replaced and for
    which ``decline_reason`` (the target's "is this FlashInfer's fused add + RMSNorm" check) returns None; returns how
    many (0 with TRTLLM_M31_DISABLE_NATIVE_ADD_RMSNORM=1 or without the op)."""
    if add_rmsnorm_op() is None:
        return 0
    count = 0
    for module in modules:
        if (
            type(module) is not RMSNorm
            or module.weight.numel() != HIDDEN
            or "forward" in module.__dict__
        ):
            continue
        if decline_reason(module) is not None:
            continue
        module.forward = types.MethodType(_plain_forward, module)
        count += 1
    return count


class M31FusedAddRmsnorm(FlashinferFusedAddRmsnorm):
    """The catalog's fused add + RMSNorm for the M3.1 targets: BF16 rows of width 6144 on sm_107 run the family op
    (:func:`fused_add_rmsnorm`, bit for bit FlashInfer's kernel); every other call, and
    TRTLLM_M31_DISABLE_NATIVE_ADD_RMSNORM=1, runs the catalog op. Same contract, cells and reference as the catalog
    entry."""

    def __call__(
        self, x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if x.dim() == 2 and x.shape[-1] == HIDDEN and fused_add_rmsnorm(x, residual, weight, eps):
            return x, residual
        return super().__call__(x, residual, weight, eps)


m31_fused_add_rmsnorm = M31FusedAddRmsnorm()
