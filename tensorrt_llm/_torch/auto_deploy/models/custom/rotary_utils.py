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

"""RoPE table helpers for AutoDeploy custom model implementations."""

import torch
from torch import nn


class RotaryEmbeddingBase(nn.Module):
    """Base class for RoPE modules that keep inv_freq in FP32."""

    def _apply(self, fn):
        super()._apply(fn)
        inv_freq = getattr(self, "inv_freq", None)
        if isinstance(inv_freq, torch.Tensor) and inv_freq.is_floating_point():
            self.inv_freq = inv_freq.float()
        return self


def build_rope_cos_sin_cache(
    inv_freq: torch.Tensor,
    max_position_embeddings: int,
    target: torch.Tensor,
    attention_scaling: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build full RoPE cos/sin tables from a small inv_freq buffer.

    This intentionally returns graph-computed tensors instead of registering large
    module buffers. Later AutoDeploy RoPE transforms can materialize a fused cache
    after the pipeline-cache boundary.
    """
    inv_freq = inv_freq.to(device=target.device)
    positions = torch.arange(
        max_position_embeddings,
        dtype=inv_freq.dtype,
        device=inv_freq.device,
    )
    freqs = torch.outer(positions, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos() * attention_scaling
    sin = emb.sin() * attention_scaling
    return cos.to(dtype=target.dtype), sin.to(dtype=target.dtype)


def build_rope_complex_cache(
    inv_freq: torch.Tensor,
    max_position_embeddings: int,
    target: torch.Tensor,
    attention_scaling: float = 1.0,
) -> torch.Tensor:
    """Build full complex RoPE frequencies from a small inv_freq buffer."""
    inv_freq = inv_freq.to(device=target.device)
    positions = torch.arange(
        max_position_embeddings,
        dtype=inv_freq.dtype,
        device=inv_freq.device,
    )
    freqs = torch.outer(positions, inv_freq)
    return torch.polar(torch.ones_like(freqs), freqs) * attention_scaling
