# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Inputs of the glm5_next fp8 tp4 mHC op tests, at GLM-5.3-Flash's scale.

The checkpoint stores each sublayer's mixer ``fn [24, 16384]`` as BF16 values (standard deviation
0.02 to 0.03 per layer) and its ``scale`` (0.04 to 0.6) and ``base`` in FP32; the target holds
``fn`` as FP32. The ``fn`` drawn here is BF16-representable like the checkpoint's, which TF32 holds
exactly, so the tensor-core backends differ from the references in summation order only, as they
do on the real weights. ``base`` is drawn wider than the checkpoint's (whose combination logits sit
near a diagonal), so that every combination entry and every stream carries weight and a transposed
or misplaced coefficient shows.
"""

import torch

MULT, HIDDEN, MIX = 4, 4096, 24

#: The op arguments after the tensors, in call order.
SCALAR_NAMES = (
    "sinkhorn_iters",
    "norm_eps",
    "hc_eps",
    "sinkhorn_eps",
    "post_mult",
    "layer_input_eps",
)


def generator(seed: int) -> torch.Generator:
    return torch.Generator(device="cuda").manual_seed(seed)


def scalars(spec) -> list:
    """A cell's mHC scalars in call order."""
    return [spec[name] for name in SCALAR_NAMES]


def streams(rows: int, gen: torch.Generator, std: float = 1.0) -> torch.Tensor:
    """``[rows, 4, 4096]`` BF16 residual streams."""
    return (torch.randn(rows, MULT, HIDDEN, generator=gen, device="cuda") * std).to(torch.bfloat16)


def block_output(rows: int, gen: torch.Generator, std: float = 1.0) -> torch.Tensor:
    """``[rows, 4096]`` BF16: a sublayer's reduced output."""
    return (torch.randn(rows, HIDDEN, generator=gen, device="cuda") * std).to(torch.bfloat16)


def coefficients(rows: int, gen: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    """A sublayer's ``post [rows, 4]`` in ``[0, 2]`` and column-normalized ``comb [rows, 4, 4]``."""
    post = torch.rand(rows, MULT, generator=gen, device="cuda") * 2
    comb = torch.rand(rows, MULT, MULT, generator=gen, device="cuda") + 0.05
    return post, comb / comb.sum(-2, keepdim=True)


def mixer(gen: torch.Generator) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """A sublayer's ``(fn [24, 16384], scale [3], base [24])``, FP32."""
    fn = torch.randn(MIX, MULT * HIDDEN, generator=gen, device="cuda") * 0.03
    scale = torch.tensor([0.1, 0.08, 0.3], device="cuda")
    base = torch.randn(MIX, generator=gen, device="cuda") * 2
    return fn.to(torch.bfloat16).float(), scale, base


def norm_weight(gen: torch.Generator) -> torch.Tensor:
    """A sublayer's input RMSNorm weight ``[4096]``, BF16."""
    return (1 + 0.1 * torch.randn(HIDDEN, generator=gen, device="cuda")).to(torch.bfloat16)
