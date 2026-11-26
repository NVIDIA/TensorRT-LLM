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

# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Public entry point for continuous MQA TopK."""

from __future__ import annotations

from typing import Literal

import torch


class FP8MQATopKWrapper:
    """Selective prefill over contiguous, precompressed K.

    Prepare bindings before capture and serialize calls on each wrapper.
    See README.md for dtype, planning and lifetime contracts."""

    is_experimental = True

    def __init__(
        self,
        *,
        max_context_length: int,
        max_chunk_size: int,
        max_batch_size: int,
        device,
        strategy: Literal["selective"] = "selective",
        backend: Literal["cutedsl"] = "cutedsl",
        num_heads: int,
        head_dim=128,
        top_k=512,
        math_mode=None,
        compression_ratio=1,
        num_keys=None,
        qkv_dtype="fp8",
        logits_dtype=torch.float32,
        weights_dtype=torch.float32,
        model_config=None,
    ):
        if strategy != "selective":
            raise ValueError("continuous TopK supports selective only")
        if backend != "cutedsl":
            raise ValueError("backend must be 'cutedsl'")
        from .adapter import _continuous_plan_type

        options = dict(
            max_context_length=max_context_length,
            max_chunk_size=max_chunk_size,
            max_batch_size=max_batch_size,
            device=device,
            num_heads=num_heads,
            head_dim=head_dim,
            top_k=top_k,
            compression_ratio=compression_ratio,
            math_mode=math_mode,
            num_keys=num_keys,
        )
        options.update(
            qkv_dtype=qkv_dtype,
            logits_dtype=logits_dtype,
            weights_dtype=weights_dtype,
            model_config=model_config,
        )
        self._capacity_plan = _continuous_plan_type("cutedsl")(**options)
        self._compression_ratio = compression_ratio
        self._backend = backend
        self._strategy = strategy

    def plan(
        self,
        qo_indptr: torch.Tensor,
        kv_indptr: torch.Tensor,
        *,
        num_query_rows: int | None = None,
        next_n: int | None = None,
        host_length_hints: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> None:
        """Stage CUDA int32[2] query/KV prefixes in original tokens.

        next_n asserts the query count; CPU host_length_hints bound live lengths.
        Omit host hints under capture; inspect error flags outside capture."""
        return self._capacity_plan.plan(
            qo_indptr,
            kv_indptr,
            num_query_rows=num_query_rows,
            next_n=next_n,
            host_length_hints=host_length_hints,
        )

    def run(
        self,
        q: torch.Tensor,
        weights: torch.Tensor,
        k: torch.Tensor,
        *,
        k_scale: torch.Tensor,
        out: torch.Tensor | None = None,
        q_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return request-local INT32 indices with -1 padding.

        Block-scaled inputs require q_scale. Outputs share storage unless out is
        supplied. See README.md for shapes and scale encodings."""
        return self._capacity_plan.run(
            q, weights, k, k_scale=k_scale, out=out, q_scale=q_scale
        )

    def prepare_pool(self, k, *, k_scale):
        """Prepare a K/scale binding outside caller graph capture."""
        return self._capacity_plan.prepare_pool(k, k_scale=k_scale)

    def clear_pools(self):
        return self._capacity_plan.clear_pools()

    def input_buffers(self):
        """Stable Q/weight/prefix buffers (plus q_scale for MXFP4/MXFP8)."""
        return self._capacity_plan.input_buffers()

    @property
    def strategy(self):
        return self._strategy

    @property
    def math_mode(self):
        return self._capacity_plan.math_mode

    @property
    def compression_ratio(self):
        return self._compression_ratio

    @property
    def error(self):
        return self._capacity_plan.error

    @property
    def repair_error(self):
        return self._capacity_plan.repair_error

    @property
    def active_shape(self):
        return self._capacity_plan.active_shape


__all__ = ["FP8MQATopKWrapper"]
