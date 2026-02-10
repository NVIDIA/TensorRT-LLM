# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Custom ops corresponding to l2norm."""

import torch

from tensorrt_llm._torch.modules.fla.l2norm import l2norm_fwd


@torch.library.custom_op("auto_deploy::torch_l2norm", mutates_args=())
def _torch_l2norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    x: (..., D)
    returns:
      y: (..., D)  # normalized
      rstd: (...,) # 1/sqrt(sum(x^2)+eps) along `dim`
    """
    x_f32 = x.float()
    s = (x_f32 * x_f32).sum(dim=-1, keepdim=True)  # (..., 1)
    rstd = torch.rsqrt(s + eps)  # (..., 1)
    y = (x_f32 * rstd).to(x.dtype)  # cast back
    return y


@_torch_l2norm.register_fake
def _torch_l2norm_fake(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return torch.empty_like(x)


@torch.library.custom_op("auto_deploy::fla_l2norm", mutates_args=())
def fla_l2norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    y = l2norm_fwd(x, eps)
    return y


@fla_l2norm.register_fake
def fla_l2norm_fake(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return torch.empty_like(x)
