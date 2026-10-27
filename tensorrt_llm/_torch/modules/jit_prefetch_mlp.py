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
"""Triton variant provider for the MLP activation kernels (see
``tensorrt_llm/_torch/jit_prefetch.py``).

An ``MLP`` with the relu2 activation (Nemotron-H's MLP layers and shared
experts) runs ``fused_relu2`` on its up projection's output, a contiguous
``[tokens, intermediate]`` tensor. The Triton variant follows from that
output's dtype and the integer class of its element count, so every variant
a model reaches is known once it is constructed; they are planned before
warmup's first forward (``early_batches``).
"""

from types import SimpleNamespace
from typing import Any, Iterator, List, Optional

import torch

from ..jit_prefetch import KernelCall, shadow_launches


class Relu2Provider:
    """Plans the ``fused_relu2`` launches of the model's relu2 MLPs."""

    kernel_modules = ("tensorrt_llm._torch.fused_relu2_triton",)
    # A row count of each integer class Triton specializes on.
    ROWS = (1, 16, 17)

    def __init__(self, model):
        from ..utils import relu2
        from .mlp import MLP

        shapes = set()
        for m in model.modules():
            if not isinstance(m, MLP) or m.activation is not relu2:
                continue
            up_proj = getattr(m, "up_proj", None)
            width = getattr(up_proj, "out_features", None)
            dtype = getattr(up_proj, "dtype", None)
            if not isinstance(width, int) or not isinstance(dtype, torch.dtype):
                continue
            shapes.add((width, dtype))
        self.shapes = sorted(shapes, key=lambda s: (s[0], str(s[1])))

    def __bool__(self):
        return bool(self.shapes)

    def early_batches(self) -> Iterator[Any]:
        if self.shapes:
            yield SimpleNamespace(relu2=True, ctx_chunk_lens=[])

    def __call__(self, batch_ctx: Any, seen: Optional[set] = None) -> List[KernelCall]:
        if not getattr(batch_ctx, "relu2", False):
            return []
        from ..fused_relu2_triton import fused_relu2

        with shadow_launches() as rec:
            for width, dtype in self.shapes:
                for rows in self.ROWS:
                    fused_relu2(torch.empty(rows, width, dtype=dtype, device="meta"))
        return list(rec)


def register_providers(prefetcher, engine) -> None:
    """Register the provider of the relu2 MLP activations of ``engine``'s
    model. Once per process: later calls change nothing."""
    provider = Relu2Provider(engine.model)
    if provider:
        prefetcher.register("relu2", provider)
