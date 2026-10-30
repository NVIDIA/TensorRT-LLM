# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Beam MoE feed-forward of the sparse decoder blocks 3-5 / sm_103 / tp1, constructed directly.

A component, not a routed target. There is no ``routing.py``, no router-index entry and
no ``ForCausalLM``: nothing in the engine builds this class, and it serves no request.
A caller constructs it from the checkpoint's configuration, loads the MoE's 770 tensors
and passes it the already-normalized hidden states that enter its sparse block's
feed-forward sublayer; ``layer_idx`` (3, 4 or 5, as the router accepts) names that block,
and each block owns its own router and experts. It composes three accepted components and
adds no arithmetic of its own::

    weights, ids = router(hidden)                  ../moe_router_bf16__sm_103__tp1
    out = dispatch_combine(hidden, ids, weights,   ../moe_dispatch_combine_bf16__sm_103__tp1
                           lambda e, rows: experts[e](rows))
                                                   ../moe_expert_bf16__sm_103__tp1, 256 of them

That is, as the source model computes it: fp32 grouped sigmoid top-k routing with a
selection-only correction bias (8 of 256 experts per token, from 4 of 8 groups) and
unbiased mixing weights; each selected expert, in ascending id order, runs its own bf16
SwiGLU on its tokens' rows; each expert output times its fp32 weight is rounded to bf16,
the routes are summed in fp32, scaled by the routed scaling factor 2.5 once and rounded
to bf16. No residual, normalization or collective is applied here.

Host synchronization: routing runs on the GPU without one. Dispatch reads the selected
experts and each expert's tokens back to the host -- two synchronizations for the expert
list and two per selected expert -- so a call is not capturable in a CUDA graph. Token
rows, products, roundings, sums and the scaling run on the GPU.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Mapping

import torch
from torch import nn

from ..full_model_nvfp4__sm_103__tp4 import dispatch_combine
from ..moe_expert_bf16__sm_103__tp1.expert import WEIGHT_SHAPES as EXPERT_WEIGHT_SHAPES
from ..moe_expert_bf16__sm_103__tp1.expert import BeamMoeExpertBf16Sm103Tp1
from ..moe_router_bf16__sm_103__tp1.router import BeamMoeRouterBf16Sm103Tp1
from ..moe_router_bf16__sm_103__tp1.router import check_config as check_router_config

#: Fields of the checkpoint's configuration this component reads beyond the router's.
SUPPORTED_CONFIG = {
    "moe_intermediate_size": 2560,
    "hidden_act": "silu",
    "routed_scaling_factor": 2.5,
}
SUPPORTED_LAYER_IDX = 3

_SM = (10, 3)


def check_config(config, layer_idx: int) -> None:
    """Raise ValueError naming the first field outside what this component computes."""
    check_router_config(config, layer_idx)
    for field, expected in SUPPORTED_CONFIG.items():
        actual = getattr(config, field, None)
        if actual != expected or type(actual) is not type(expected):
            raise ValueError(f"unsupported {field}={actual!r}; this MoE computes {expected!r}")


def weight_shapes(config) -> "OrderedDict[str, tuple[tuple[int, ...], torch.dtype]]":
    """The MoE's tensors by checkpoint name (relative to ``mlp.``), in order, with dtypes."""
    shapes = OrderedDict(
        [
            ("gate.weight", ((config.num_experts, config.hidden_size), torch.float32)),
            ("gate.e_score_correction_bias", ((config.num_experts,), torch.float32)),
        ]
    )
    for expert_id in range(config.num_experts):
        for name, shape in EXPERT_WEIGHT_SHAPES.items():
            shapes[f"experts.{expert_id}.{name}"] = (shape, torch.bfloat16)
    return shapes


def check_weights(expected: Mapping, weights: Mapping[str, torch.Tensor], what: str) -> None:
    """Raise unless ``weights`` holds exactly ``expected``'s names, dtypes and shapes.

    Reads tensor metadata only, never values, and copies nothing.
    """
    if not isinstance(weights, Mapping):
        raise TypeError(f"weights must be a mapping, got {type(weights).__name__}")
    missing = [n for n in expected if n not in weights]
    unexpected = sorted(str(n) for n in weights if n not in expected)
    if missing or unexpected:
        raise ValueError(
            f"{what} weights mismatch: missing {missing[:8]}{'...' if len(missing) > 8 else ''}, "
            f"unexpected {unexpected[:8]}{'...' if len(unexpected) > 8 else ''}"
        )
    for name, (shape, dtype) in expected.items():
        tensor = weights[name]
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor, got {type(tensor).__name__}")
        if tensor.dtype != dtype:
            raise TypeError(f"{name} must be {dtype}, got {tensor.dtype}")
        if tuple(tensor.shape) != shape:
            raise ValueError(f"{name} must be {list(shape)}, got {list(tensor.shape)}")
        if tensor.layout != torch.strided or tensor.device.type not in ("cpu", "cuda"):
            raise ValueError(
                f"{name} must be a dense host or CUDA tensor, got {tensor.layout} on {tensor.device}"
            )


class BeamMoeBf16Sm103Tp1(nn.Module):
    """A sparse block's MoE: normalized BF16 hidden states in, the combined BF16 MoE output out.

    State: ``gate``, the router (FP32 ``weight [256, 5120]`` and selection-only
    ``e_score_correction_bias [256]``), and ``experts``, 256 routed experts, each with its
    own bias-free BF16 ``gate_proj``/``up_proj [2560, 5120]`` and ``down_proj [5120, 2560]``.
    All are loaded together with ``load_weights``.

    ``forward(hidden_states)`` takes ``[T, 5120]`` ``torch.bfloat16`` on the MoE's device,
    T >= 1 tokens already normalized by the caller, and returns ``[T, 5120]``
    ``torch.bfloat16`` on that device, row ``t`` for token ``t``. Inputs outside that are
    rejected before any kernel runs, and the input is never modified. Measured at T = 1, 4,
    17 and 129.
    """

    def __init__(
        self, config, layer_idx: int = SUPPORTED_LAYER_IDX, device: str | torch.device = "cuda"
    ):
        super().__init__()
        check_config(config, layer_idx)
        device = torch.device(device)
        if device.type != "cuda" or torch.cuda.get_device_capability(device) != _SM:
            raise ValueError(
                f"this MoE is built for sm_{_SM[0]}{_SM[1]} CUDA devices, got {device}"
            )
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self._shapes = weight_shapes(config)
        self.gate = BeamMoeRouterBf16Sm103Tp1(config, layer_idx, device=device)
        self.experts = nn.ModuleList(
            BeamMoeExpertBf16Sm103Tp1(device=device) for _ in range(config.num_experts)
        )
        self._loaded = False

    def checkpoint_tensors(self) -> "OrderedDict[str, torch.Tensor]":
        """The MoE's tensors by checkpoint name (relative to ``mlp.``), in checkpoint order."""
        tensors = OrderedDict(
            [
                ("gate.weight", self.gate.weight),
                ("gate.e_score_correction_bias", self.gate.e_score_correction_bias),
            ]
        )
        for expert_id, expert in enumerate(self.experts):
            for name, tensor in expert.checkpoint_tensors().items():
                tensors[f"experts.{expert_id}.{name}"] = tensor
        return tensors

    def check_weights(self, weights: Mapping[str, torch.Tensor]) -> None:
        """Raise unless ``weights`` is exactly the MoE's 770 tensors by name, dtype and shape."""
        check_weights(self._shapes, weights, "MoE")

    @torch.no_grad()
    def load_weights(self, weights: Mapping[str, torch.Tensor]) -> None:
        """Copy exactly the MoE's 770 tensors, by checkpoint name relative to ``mlp.``.

        ``gate.weight`` and ``gate.e_score_correction_bias`` are FP32; each
        ``experts.<e>.{gate,up,down}_proj.weight`` is BF16 and loads into expert ``e``.
        Values are copied as given, [out, in], without transposing or converting; host
        tensors are copied to the MoE's device. Every tensor is checked before any
        parameter changes, so a rejected load leaves the MoE as it was. The MoE runs only
        after a load completed: a failure while copying leaves it unloaded.
        """
        self.check_weights(weights)
        self._loaded = False
        self.gate.load_weights(weights["gate.weight"], weights["gate.e_score_correction_bias"])
        for expert_id, expert in enumerate(self.experts):
            prefix = f"experts.{expert_id}."
            expert.load_weights({name: weights[prefix + name] for name in EXPERT_WEIGHT_SHAPES})
        self._loaded = True

    def _check_input(self, hidden_states: torch.Tensor) -> None:
        """Raise on anything this MoE would otherwise compute wrongly."""
        if not self._loaded:
            raise RuntimeError("load_weights must be called before the MoE runs")
        if not isinstance(hidden_states, torch.Tensor):
            raise TypeError(
                f"hidden_states must be a torch.Tensor, got {type(hidden_states).__name__}"
            )
        if hidden_states.dtype != torch.bfloat16:
            raise TypeError(f"hidden_states must be torch.bfloat16, got {hidden_states.dtype}")
        if hidden_states.layout != torch.strided:
            raise ValueError(f"hidden_states must be a dense tensor, got {hidden_states.layout}")
        if hidden_states.dim() != 2 or hidden_states.shape[1] != self.hidden_size:
            raise ValueError(
                f"hidden_states must be [tokens, {self.hidden_size}], "
                f"got {list(hidden_states.shape)}"
            )
        if hidden_states.shape[0] < 1:
            raise ValueError("hidden_states must hold at least one token")
        if hidden_states.device != self.gate.weight.device:
            raise ValueError(
                f"hidden_states must be on the MoE's device {self.gate.weight.device}, "
                f"got {hidden_states.device}"
            )

    def _expert(self, expert_id: int, rows: torch.Tensor) -> torch.Tensor:
        """The dispatch provider: expert ``expert_id``'s own unweighted output for ``rows``."""
        return self.experts[expert_id](rows)

    @torch.no_grad()
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        self._check_input(hidden_states)
        mixing_weights, expert_ids = self.gate(hidden_states)
        return dispatch_combine.beam_dispatch_combine(
            hidden_states, expert_ids, mixing_weights, self._expert
        )
