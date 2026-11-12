# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""xIELU activation (https://arxiv.org/abs/2411.13010)."""

import math

import torch
import torch.nn.functional as F
from torch import nn


def xielu_reference(
    x: torch.Tensor, a_p: float, a_n: float, beta: float, eps: float
) -> torch.Tensor:
    """xIELU on post-softplus coefficients, computed in fp32 and rounded once to ``x.dtype``."""
    xf = x.float()
    pos = a_p * xf * xf + beta * xf
    neg = (torch.expm1(torch.clamp_max(xf, eps)) - xf) * a_n + beta * xf
    return torch.where(xf > 0, pos, neg).to(x.dtype)


class XIELU(nn.Module):
    """xIELU activation with per-layer learnable coefficients.

    ``alpha_p`` and ``alpha_n`` are stored in the checkpoint's pre-softplus
    form. The effective coefficients ``a_p = softplus(alpha_p)`` and
    ``a_n = beta + softplus(alpha_n)`` are constant at inference time, so they
    are derived once in fp32 by :meth:`cache_derived_state` and passed to the
    kernels as plain floats.
    """

    def __init__(
        self,
        alpha_p_init: float = 0.8,
        alpha_n_init: float = 0.8,
        beta: float = 0.5,
        eps: float = -1e-6,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.alpha_p = nn.Parameter(
            torch.tensor([math.log(math.expm1(alpha_p_init))], dtype=dtype), requires_grad=False
        )
        self.alpha_n = nn.Parameter(
            torch.tensor([math.log(math.expm1(alpha_n_init - beta))], dtype=dtype),
            requires_grad=False,
        )
        self.register_buffer("beta", torch.tensor(beta, dtype=dtype))
        self.register_buffer("eps", torch.tensor(eps, dtype=dtype))
        self.a_p = alpha_p_init
        self.a_n = alpha_n_init
        self.beta_value = beta
        self.eps_value = eps

    def load_weights(self, weights: list[dict], allow_partial_loading: bool = False):
        assert len(weights) == 1
        weights = weights[0]
        for name in ("alpha_p", "alpha_n", "beta", "eps"):
            if name not in weights:
                assert allow_partial_loading or name in ("beta", "eps"), (
                    f"Missing xIELU weight '{name}'"
                )
                continue
            value = weights[name]
            if not isinstance(value, torch.Tensor):
                value = value[:]
            target = getattr(self, name)
            target.data.copy_(value.reshape(target.shape))
        self.cache_derived_state()

    def cache_derived_state(self) -> None:
        beta = self.beta.float().item()
        self.a_p = F.softplus(self.alpha_p.float()).item()
        self.a_n = beta + F.softplus(self.alpha_n.float()).item()
        self.beta_value = beta
        self.eps_value = self.eps.float().item()

    def post_load_weights(self) -> None:
        self.cache_derived_state()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return xielu_reference(x, self.a_p, self.a_n, self.beta_value, self.eps_value)

    def extra_repr(self) -> str:
        return (
            f"a_p={self.a_p:.6g}, a_n={self.a_n:.6g}, beta={self.beta_value:.6g}, "
            f"eps={self.eps_value:.6g}"
        )
