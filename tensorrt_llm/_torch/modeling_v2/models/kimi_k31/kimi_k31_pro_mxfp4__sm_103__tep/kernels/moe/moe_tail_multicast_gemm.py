# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause
# Vendored from vLLM (Apache-2.0), with TensorRT-LLM changes:
# https://github.com/vllm-project/vllm/blob/8c1557a79c539ffe82d004d2a0c8d7b5e71159ce/vllm/models/kimi_k3/nvidia/ops/cute_dsl/latent_moe_tail/fused_add_multicast_gemm.py

# Upstream file notice: Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Upstream file license: BSD-3-Clause

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""Up-projection dispatcher of the fused latent-MoE tail: the static-M skinny SIMT GEMM with a shared-shard add
and a multicast epilogue into the symmetric mailbox."""

import torch

from .moe_tail_skinny_gemm import FusedAddMulticastSkinnyGemmKernel


class AdaptiveUpProjectionKernel:
    """Dispatch the static-M skinny up-projection into one mailbox."""

    def __init__(
        self,
        *,
        rank: int,
        tp_size: int,
        latent_dim: int,
        hidden_dim: int,
        max_m: int,
        skinny_max_m: int,
        mailbox: torch.Tensor,
        mailbox_multicast_ptr: int,
    ) -> None:
        """``mailbox`` is this rank's unicast view ``[1, max_m, hidden_dim]`` of
        the symmetric mailbox and ``mailbox_multicast_ptr`` its multicast
        address. The caller fills the mailbox with the empty marker (BF16 -0)
        and fences all ranks before the first launch."""
        if hidden_dim % tp_size:
            raise ValueError("hidden_dim must be divisible by TP size")
        if not 0 <= skinny_max_m <= min(1, max_m):
            raise ValueError("skinny_max_m must be in [0, min(1, max_m)]")
        if tuple(mailbox.shape) != (1, max_m, hidden_dim) or mailbox.dtype != torch.bfloat16:
            raise ValueError(f"mailbox must be BF16 [1, {max_m}, {hidden_dim}]")
        if not mailbox_multicast_ptr:
            raise ValueError("mailbox multicast address is unavailable")
        self.rank = rank
        self.tp_size = tp_size
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.shard_dim = hidden_dim // tp_size
        self.max_m = max_m
        self.skinny_max_m = skinny_max_m
        self._device = mailbox.device
        self._skinny_by_m: dict[int, FusedAddMulticastSkinnyGemmKernel] = {}
        if skinny_max_m and self.latent_dim % (224 * 8):
            raise ValueError("Skinny up-projection requires latent_dim divisible by 1792.")

        self._mailbox = mailbox
        self._mailbox_multicast_ptr = int(mailbox_multicast_ptr)

    @staticmethod
    def init_mailbox(mailbox: torch.Tensor) -> None:
        """Every BF16 lane -0 (the Lamport copy checks each lane)."""
        mailbox.view(torch.int32).fill_(0x80008000 - (1 << 32))

    def compile_skinny(self, m: int) -> None:
        if not 1 <= m <= self.skinny_max_m:
            raise ValueError(f"Skinny up-projection requires M in [1, {self.skinny_max_m}].")
        if m in self._skinny_by_m:
            return
        with torch.accelerator.device_index(self._device.index):
            self._skinny_by_m[m] = FusedAddMulticastSkinnyGemmKernel(
                rank=self.rank,
                tp_size=self.tp_size,
                latent_dim=self.latent_dim,
                hidden_dim=self.hidden_dim,
                num_rows=m,
            )

    def __call__(
        self,
        latent: torch.Tensor,
        weight: torch.Tensor,
        shared_shard: torch.Tensor,
    ) -> torch.Tensor:
        if latent.ndim != 2:
            raise ValueError("latent must be rank-2")
        m = latent.shape[0]
        device = self._mailbox.device
        expected = (
            (latent, (m, self.latent_dim), "latent"),
            (
                weight,
                (self.shard_dim, self.latent_dim),
                "weight",
            ),
            (
                shared_shard,
                (self.max_m, self.shard_dim),
                "shared_shard",
            ),
        )
        for tensor, shape, name in expected:
            if tensor.shape != shape or tensor.dtype != torch.bfloat16 or tensor.device != device:
                raise ValueError(f"{name} must be CUDA torch.bfloat16 {list(shape)}")
        if (
            not latent.is_contiguous()
            or not weight.is_contiguous()
            or shared_shard.stride() != (self.hidden_dim, 1)
        ):
            raise ValueError("up-projection inputs have unsupported strides")
        if not 1 <= m <= self.skinny_max_m:
            raise ValueError(f"runtime M={m} must be in [1, {self.skinny_max_m}]")

        skinny = self._skinny_by_m.get(m)
        if skinny is None:
            raise RuntimeError(f"Skinny up-projection M={m} was not compiled before launch.")
        return skinny(
            latent,
            weight,
            shared_shard,
            self._mailbox,
            self._mailbox_multicast_ptr,
        )
