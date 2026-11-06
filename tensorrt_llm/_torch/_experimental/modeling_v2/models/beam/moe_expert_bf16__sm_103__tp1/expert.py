# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""One Beam MoE routed expert of decoder block 3 / sm_103 / tp1, constructed directly.

A component, not a routed target. There is no ``routing.py``, no router-index entry and
no ``ForCausalLM``: nothing in the engine builds this class, and it serves no request.
A caller constructs it, loads one expert's three tensors and passes it the
already-normalized hidden states of the tokens routed to that expert. Which expert it is
decides only which tensors are loaded. It routes no token and applies no mixing weight,
routed scaling factor, residual or normalization, and it makes no collective: weighting
and combining expert outputs, and any reduction across ranks, belong to the caller. It
does not call the built-in Beam modeling.

The expert, bias-free, with the projections in bf16 and the activation in fp32::

    gate    = hidden @ gate_proj^T          [T, 2560] bf16    catalog cublas_mm
    up      = hidden @ up_proj^T            [T, 2560] bf16    catalog cublas_mm
    silu    = silu(gate.float())            [T, 2560] fp32    ops/activation/swiglu_fp32.py
    product = silu * up.float()             [T, 2560] fp32
    act     = product.to(bfloat16)          [T, 2560] bf16    the one rounding before down
    out     = act @ down_proj^T             [T, 5120] bf16    catalog cublas_mm
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Mapping

import torch
from torch import nn

from tensorrt_llm._torch._experimental.modeling_v2.catalog.activation.swiglu_fp32 import swiglu_fp32
from tensorrt_llm._torch._experimental.modeling_v2.catalog.gemm.cublas_mm import cublas_mm

HIDDEN_SIZE = 5120
INTERMEDIATE_SIZE = 2560

#: The expert's three tensors by checkpoint name (relative to ``mlp.experts.<e>.``), [out, in].
WEIGHT_SHAPES = OrderedDict(
    [
        ("gate_proj.weight", (INTERMEDIATE_SIZE, HIDDEN_SIZE)),
        ("up_proj.weight", (INTERMEDIATE_SIZE, HIDDEN_SIZE)),
        ("down_proj.weight", (HIDDEN_SIZE, INTERMEDIATE_SIZE)),
    ]
)

_SM = (10, 3)


def _attribute(name: str) -> str:
    return name.removesuffix(".weight")


class BeamMoeExpertBf16Sm103Tp1(nn.Module):
    """One routed expert: BF16 hidden states in, the expert's unweighted BF16 output out.

    State is exactly three bias-free BF16 matrices, loaded with ``load_weights``:
    ``gate_proj`` and ``up_proj`` ``[2560, 5120]`` and ``down_proj`` ``[5120, 2560]``, stored
    as given, [out, in].

    ``forward(hidden_states)`` takes ``[T, 5120]`` ``torch.bfloat16`` on the expert's
    device, T >= 1 tokens already normalized by the caller, and returns ``[T, 5120]``
    ``torch.bfloat16`` on that device, one row per input row. Inputs outside that are
    rejected before any kernel runs. Measured at T = 1, 17 and 129; zero tokens (an expert
    no token was routed to) is rejected, not computed.

    Op ownership: the three projections are the catalog's ``cublas_mm``, each with the
    stored weight's ``.t()`` as its column-major operand; the fp32 SiLU, the fp32 product
    and the bf16 rounding are this component's ``swiglu_fp32``.
    """

    def __init__(self, device: str | torch.device = "cuda"):
        super().__init__()
        device = torch.device(device)
        if device.type != "cuda" or torch.cuda.get_device_capability(device) != _SM:
            raise ValueError(
                f"this expert is built for sm_{_SM[0]}{_SM[1]} CUDA devices, got {device}"
            )
        for name, shape in WEIGHT_SHAPES.items():
            tensor = torch.empty(shape, dtype=torch.bfloat16, device=device)
            self.register_parameter(_attribute(name), nn.Parameter(tensor, requires_grad=False))
        self._loaded = False

    def checkpoint_tensors(self) -> "OrderedDict[str, torch.Tensor]":
        """The expert's tensors by checkpoint name, in checkpoint order."""
        return OrderedDict((name, getattr(self, _attribute(name))) for name in WEIGHT_SHAPES)

    @torch.no_grad()
    def load_weights(self, weights: Mapping[str, torch.Tensor]) -> None:
        """Copy exactly the expert's three tensors, by checkpoint name: BF16, [out, in].

        Values are copied as given, without transposing or converting; host tensors are
        copied to the expert's device. Every tensor is checked, and copied to the device,
        before any parameter changes, so a rejected load leaves the expert as it was.
        """
        if not isinstance(weights, Mapping):
            raise TypeError(f"weights must be a mapping, got {type(weights).__name__}")
        missing = [n for n in WEIGHT_SHAPES if n not in weights]
        unexpected = sorted(str(n) for n in weights if n not in WEIGHT_SHAPES)
        if missing or unexpected:
            raise ValueError(f"expert weights mismatch: missing {missing}, unexpected {unexpected}")
        for name, shape in WEIGHT_SHAPES.items():
            tensor = weights[name]
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"{name} must be a torch.Tensor, got {type(tensor).__name__}")
            if tensor.dtype != torch.bfloat16:
                raise TypeError(f"{name} must be torch.bfloat16, got {tensor.dtype}")
            if tuple(tensor.shape) != shape:
                raise ValueError(f"{name} must be {list(shape)}, got {list(tensor.shape)}")
            if tensor.layout != torch.strided or tensor.device.type not in ("cpu", "cuda"):
                raise ValueError(
                    f"{name} must be a dense host or CUDA tensor, got {tensor.layout} "
                    f"on {tensor.device}"
                )
        device = self.gate_proj.device
        staged = {name: weights[name].to(device=device, copy=True) for name in WEIGHT_SHAPES}
        for name, tensor in staged.items():
            getattr(self, _attribute(name)).copy_(tensor)
        self._loaded = True

    def _check_input(self, hidden_states: torch.Tensor) -> None:
        """Raise on anything this expert would otherwise compute wrongly."""
        if not self._loaded:
            raise RuntimeError("load_weights must be called before the expert runs")
        if not isinstance(hidden_states, torch.Tensor):
            raise TypeError(
                f"hidden_states must be a torch.Tensor, got {type(hidden_states).__name__}"
            )
        if hidden_states.dtype != torch.bfloat16:
            raise TypeError(f"hidden_states must be torch.bfloat16, got {hidden_states.dtype}")
        if hidden_states.dim() != 2 or hidden_states.shape[1] != HIDDEN_SIZE:
            raise ValueError(
                f"hidden_states must be [tokens, {HIDDEN_SIZE}], got {list(hidden_states.shape)}"
            )
        if hidden_states.shape[0] < 1:
            raise ValueError("hidden_states must hold at least one token")
        if hidden_states.device != self.gate_proj.device:
            raise ValueError(
                f"hidden_states must be on the expert's device {self.gate_proj.device}, "
                f"got {hidden_states.device}"
            )

    @torch.no_grad()
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        self._check_input(hidden_states)
        x = hidden_states.contiguous()
        gate = cublas_mm(x, self.gate_proj.t())
        up = cublas_mm(x, self.up_proj.t())
        act = swiglu_fp32(gate, up)["act"]
        return cublas_mm(act, self.down_proj.t())
