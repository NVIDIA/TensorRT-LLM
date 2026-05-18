# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from typing import Tuple, Union

import torch


def all_close(
    t1: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
    t2: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
    atol: float = 1e-3,
    rtol: float = 1e-3,
):
    if isinstance(t1, torch.Tensor):
        t1 = (t1,)
    if isinstance(t2, torch.Tensor):
        t2 = (t2,)

    all_close = True
    for idx, (a, b) in enumerate(zip(t1, t2)):
        print(f"tensor {idx=}: {a.shape=} {b.shape=}, {a=}, {b=}")
        all_close &= torch.allclose(a, b, atol=atol, rtol=rtol)
    return all_close


def reset_parameters(model: torch.nn.Module):
    for p in model.parameters():
        p.data = torch.randn_like(p.data, dtype=torch.float16).to(p)


def fp8_compatible():
    return torch.cuda.get_device_capability(0) >= (8, 9)


def fp4_compatible():
    return torch.cuda.get_device_capability(0) >= (10, 0)


def trtllm_ops_available():
    # `torch.ops.<name>` is a lazy `_OpNamespace`, so `hasattr(torch.ops, "trtllm")`
    # is always True even when no TRT-LLM ops are registered (e.g. the auto_deploy
    # standalone package). Probe for a specific op instead.
    try:
        return hasattr(torch.ops.trtllm, "fp8_quantize_1x128")
    except (AttributeError, RuntimeError):
        return False
