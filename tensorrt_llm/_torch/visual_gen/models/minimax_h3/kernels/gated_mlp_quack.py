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
"""MiniMax-H3's BF16 gate/up GEMM + SwiGLU epilogue at a fixed QuACK config.

``GatedMLP``'s opt-in path runs QuACK's SM100 ``gemm_act`` at QuACK's default config; H3 runs it at
``gemm_act_config``: 256x256 tiles on a 2x1 cluster, the dynamic persistent scheduler and
``max_swizzle_size`` 16. The output is the same SwiGLU, rounded to BF16 once.
"""

import functools

import torch


@functools.cache
def gemm_act_config(device_capacity: int):
    """QuACK config of H3's gated GEMMs, here and in the VAE decoder's FFN (``vae_swiglu_quack``)."""
    from quack.gemm_config import GemmConfig

    return GemmConfig(
        tile_m=256,
        tile_n=256,
        cluster_m=2,
        cluster_n=1,
        pingpong=False,
        is_dynamic_persistent=True,
        max_swizzle_size=16,
        device_capacity=device_capacity,
    )


def _gemm_gated(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    from quack.gemm_interface import gemm_act

    if x.shape[0] == 0:
        return x.new_empty(0, weight.shape[0] // 2)
    _, out = gemm_act(
        x,
        weight.t(),
        activation="swiglu",
        store_preact=False,
        config=gemm_act_config(torch.cuda.get_device_capability(x.device)[0]),
        concat_layout=("B",),
    )
    return out


@torch.library.custom_op(
    "trtllm::minimax_h3_gate_up_swiglu_quack", mutates_args=(), device_types="cuda"
)
def gate_up_swiglu_quack_h3(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """``silu(x @ W_gate^T) * (x @ W_up^T)`` for ``weight = [W_gate; W_up]``."""
    return _gemm_gated(x.contiguous(), weight.contiguous())


@gate_up_swiglu_quack_h3.register_fake
def _(x, weight):
    return x.new_empty((x.shape[0], weight.shape[0] // 2))
