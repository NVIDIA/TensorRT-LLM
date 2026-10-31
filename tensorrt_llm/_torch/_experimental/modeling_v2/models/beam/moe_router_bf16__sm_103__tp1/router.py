# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Beam MoE router of the sparse decoder blocks 3-5 / sm_103 / tp1, constructed directly.

A component, not a routed target. There is no ``routing.py``, no router-index entry and
no ``ForCausalLM``: nothing in the engine builds this class, and it serves no request.
A caller constructs it from the checkpoint's configuration, loads the router's two
tensors and passes it the already-normalized hidden states that enter its sparse
block's feed-forward sublayer; each sparse block owns its own router. It returns the
eight experts each token is routed to and their mixing weights. It runs no expert,
dispatches no token and combines no expert output; the routed scaling factor belongs to
that combine and is not applied here. It does not call the built-in Beam modeling.

The routing, every step fp32 (``ops/moe/beam_grouped_topk_router.py``)::

    logits  = hidden.float() @ weight^T                        [T, 256]
    scores  = sigmoid(logits)                                  unbiased: the weights use these
    choice  = scores + e_score_correction_bias                 biased: only selection uses these
    groups  = top-4 of (sum of the 2 largest choice in each of 8 groups of 32), descending
    ids     = top-8 choice among the experts of those groups, descending      [T, 8] int64
    weights = scores[ids] / (sum(scores[ids]) + 1e-20)                        [T, 8] fp32
"""

from __future__ import annotations

import torch
from torch import nn

from tensorrt_llm._torch._experimental.modeling_v2.catalog.moe.beam_grouped_topk_router import (
    beam_grouped_topk_router,
)

#: The configuration this component computes, as the checkpoint's config.json states it.
#: Anything else is rejected at construction rather than computed approximately.
SUPPORTED_CONFIG = {
    "hidden_size": 5120,
    "num_experts": 256,
    "num_experts_per_tok": 8,
    "n_group": 8,
    "topk_group": 4,
    "norm_topk_prob": True,
}
SUPPORTED_LAYER_IDX = 3
#: The sparse blocks this router has been built for: the checkpoint's first three.
SUPPORTED_LAYER_INDICES = (3, 4, 5)

#: Added to the selected scores' sum before dividing, as the source model does.
NORM_EPS = 1e-20

_SM = (10, 3)


def check_config(config, layer_idx: int) -> None:
    """Raise ValueError naming the first field outside what this component computes."""
    if type(layer_idx) is not int or layer_idx not in SUPPORTED_LAYER_INDICES:
        raise ValueError(
            f"only the routers of sparse blocks {SUPPORTED_LAYER_INDICES} are supported, "
            f"got layer_idx={layer_idx!r}"
        )
    for field, expected in SUPPORTED_CONFIG.items():
        actual = getattr(config, field, None)
        if actual != expected or type(actual) is not type(expected):
            raise ValueError(f"unsupported {field}={actual!r}; this router computes {expected!r}")
    mlp_types = getattr(config, "mlp_layer_types", None) or []
    if len(mlp_types) <= layer_idx or mlp_types[layer_idx] != "sparse":
        raise ValueError(
            f"block {layer_idx} must have a sparse (MoE) feed-forward: "
            f"mlp_layer_types[{layer_idx}]="
            f"{mlp_types[layer_idx] if len(mlp_types) > layer_idx else None!r}"
        )


class BeamMoeRouterBf16Sm103Tp1(nn.Module):
    """A sparse block's router: BF16 hidden states in, expert ids and FP32 mixing weights out.

    State is exactly two FP32 tensors, loaded with ``load_weights``: ``weight``
    ``[256, 5120]`` (checkpoint ``mlp.gate.weight``) and ``e_score_correction_bias``
    ``[256]`` (``mlp.gate.e_score_correction_bias``).

    ``forward(hidden_states)`` takes ``[T, 5120]`` ``torch.bfloat16`` on the router's
    device, T >= 1 tokens already normalized by the caller, and returns
    ``(mixing_weights, expert_ids)`` -- the source model's order -- both ``[T, 8]`` on that
    device: ``mixing_weights`` ``torch.float32``, ``expert_ids`` ``torch.int64`` in
    ``[0, 256)``, in descending order of the biased selection score. Inputs outside that
    are rejected before any kernel runs. Routing runs entirely on the GPU and makes no
    host synchronization.
    """

    def __init__(
        self, config, layer_idx: int = SUPPORTED_LAYER_IDX, device: str | torch.device = "cuda"
    ):
        super().__init__()
        check_config(config, layer_idx)
        device = torch.device(device)
        if device.type != "cuda" or torch.cuda.get_device_capability(device) != _SM:
            raise ValueError(
                f"this router is built for sm_{_SM[0]}{_SM[1]} CUDA devices, got {device}"
            )
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.n_group = config.n_group
        self.topk_group = config.topk_group
        self.weight = nn.Parameter(
            torch.empty(self.num_experts, self.hidden_size, dtype=torch.float32, device=device),
            requires_grad=False,
        )
        self.register_buffer(
            "e_score_correction_bias",
            torch.empty(self.num_experts, dtype=torch.float32, device=device),
        )
        self._loaded = False

    @torch.no_grad()
    def load_weights(self, weight: torch.Tensor, correction_bias: torch.Tensor) -> None:
        """Copy the router's two tensors: ``weight [256, 5120]`` and ``correction_bias [256]``.

        Both must already be ``torch.float32``: the router computes in fp32 and does not
        accept parameters that were rounded below it. Both are checked before either is
        copied, so a rejected load leaves the router as it was.
        """
        expected = (
            ("weight", weight, (self.num_experts, self.hidden_size)),
            ("correction_bias", correction_bias, (self.num_experts,)),
        )
        for name, tensor, shape in expected:
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"{name} must be a torch.Tensor, got {type(tensor).__name__}")
            if tensor.dtype != torch.float32:
                raise TypeError(f"{name} must be torch.float32, got {tensor.dtype}")
            if tuple(tensor.shape) != shape:
                raise ValueError(f"{name} must be {list(shape)}, got {list(tensor.shape)}")
        self.weight.copy_(weight)
        self.e_score_correction_bias.copy_(correction_bias)
        self._loaded = True

    def _check_input(self, hidden_states: torch.Tensor) -> None:
        """Raise on anything this router would otherwise compute wrongly or move to the host."""
        if not self._loaded:
            raise RuntimeError("load_weights must be called before routing")
        if not isinstance(hidden_states, torch.Tensor):
            raise TypeError(
                f"hidden_states must be a torch.Tensor, got {type(hidden_states).__name__}"
            )
        if hidden_states.dtype != torch.bfloat16:
            raise TypeError(f"hidden_states must be torch.bfloat16, got {hidden_states.dtype}")
        if hidden_states.dim() != 2 or hidden_states.shape[1] != self.hidden_size:
            raise ValueError(
                f"hidden_states must be [tokens, {self.hidden_size}], "
                f"got {list(hidden_states.shape)}"
            )
        if hidden_states.shape[0] < 1:
            raise ValueError("hidden_states must hold at least one token")
        if hidden_states.device != self.weight.device:
            raise ValueError(
                f"hidden_states must be on the router's device {self.weight.device}, "
                f"got {hidden_states.device}"
            )

    def route(self, hidden_states: torch.Tensor) -> dict[str, torch.Tensor]:
        """Every intermediate of one routing call, by the op's names; see the op's docstring."""
        self._check_input(hidden_states)
        return beam_grouped_topk_router(
            hidden_states,
            self.weight,
            self.e_score_correction_bias,
            n_group=self.n_group,
            topk_group=self.topk_group,
            top_k=self.top_k,
            eps=NORM_EPS,
        )

    @torch.no_grad()
    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        routing = self.route(hidden_states)
        return routing["weights"], routing["ids"]
