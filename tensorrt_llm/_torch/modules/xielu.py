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

import functools
import math
import os
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn


def xielu_reference(x: torch.Tensor, a_p, a_n, beta, eps) -> torch.Tensor:
    """xIELU on post-softplus coefficients, computed in fp32 and rounded once to ``x.dtype``.

    The coefficients are Python floats or fp32 0-d tensors on ``x``'s device.
    """
    xf = x.float()
    pos = a_p * xf * xf + beta * xf
    neg = (torch.expm1(torch.clamp_max(xf, eps)) - xf) * a_n + beta * xf
    return torch.where(xf > 0, pos, neg).to(x.dtype)


@functools.lru_cache(maxsize=1)
def _xielu_op_available() -> bool:
    # Set TRTLLM_XIELU_CUDA=0 to use the PyTorch expression instead of the kernel.
    return os.environ.get("TRTLLM_XIELU_CUDA", "1") == "1" and hasattr(torch.ops.trtllm, "xielu")


class XIELU(nn.Module):
    """xIELU activation with per-layer learnable coefficients.

    ``alpha_p`` and ``alpha_n`` are stored in the checkpoint's pre-softplus
    form. The effective coefficients ``a_p = softplus(alpha_p)`` and
    ``a_n = beta + softplus(alpha_n)`` are constant at inference time, so
    :meth:`cache_derived_state` derives them once, in fp32 and on the weights'
    device, into ``_coeffs = [a_p, a_n, beta, eps]``. Both the kernel and the
    PyTorch expression read that tensor on the device, and later calls update it
    in place, so a captured CUDA graph follows an in-place weight update without
    recapture. ``_coeffs`` is a plain attribute rather than a buffer, so
    ``Module.to(dtype)`` keeps it in fp32 and read-only weight sharing does not
    import it.
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
        self._coeffs: Optional[torch.Tensor] = None

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

    @torch.no_grad()
    def cache_derived_state(self) -> None:
        if self.alpha_p.is_meta:
            return
        beta = self.beta.float().reshape(1)
        coeffs = torch.cat(
            [
                F.softplus(self.alpha_p.float()).reshape(1),
                beta + F.softplus(self.alpha_n.float()).reshape(1),
                beta,
                self.eps.float().reshape(1).to(beta.device),
            ]
        )
        if self._coeffs is not None and self._coeffs.device == coeffs.device:
            self._coeffs.copy_(coeffs)
        else:
            self._coeffs = coeffs

    @property
    def coeffs(self) -> torch.Tensor:
        """fp32 ``[a_p, a_n, beta, eps]`` on the weights' device."""
        if self._coeffs is None or self._coeffs.device != self.alpha_p.device:
            self.cache_derived_state()
        return self._coeffs

    def post_load_weights(self) -> None:
        self.cache_derived_state()

    # Host copies of the coefficients, for inspection. Each read synchronizes with the device.
    @property
    def a_p(self) -> float:
        return self.coeffs[0].item()

    @property
    def a_n(self) -> float:
        return self.coeffs[1].item()

    @property
    def beta_value(self) -> float:
        return self.coeffs[2].item()

    @property
    def eps_value(self) -> float:
        return self.coeffs[3].item()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.is_cuda and x.dtype in (torch.float16, torch.bfloat16) and _xielu_op_available():
            return torch.ops.trtllm.xielu(x, self.coeffs)
        return xielu_reference(x, *self.coeffs.unbind())

    def extra_repr(self) -> str:
        if self.alpha_p.is_meta:
            return ""
        a_p, a_n, beta, eps = self.coeffs.tolist()
        return f"a_p={a_p:.6g}, a_n={a_n:.6g}, beta={beta:.6g}, eps={eps:.6g}"
