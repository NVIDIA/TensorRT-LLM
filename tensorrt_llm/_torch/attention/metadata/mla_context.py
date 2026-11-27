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
"""Cached, uncached and total KV boundaries of the MLA context rows."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from tensorrt_llm._utils import prefer_pinned

from .base import Metadata, _Registry
from .core import TrtllmCoreMetadata

if TYPE_CHECKING:
    from .batch_inputs import BatchInputs


class MlaContextMetadata(Metadata):
    """Per-context-request token boundaries the MLA context kernels read."""

    requires = (TrtllmCoreMetadata,)

    def __init__(self, registry: _Registry, config: "BatchInputs") -> None:
        super().__init__(registry, config)
        # Whether MLA context attention reads KV that is already in the cache
        # (KV-cache reuse or chunked prefill).
        self.enable_context_mla_with_cached_kv = config.enable_context_mla_with_cached_kv
        size = (config.max_num_requests + 1,)
        # Exclusive prefix sums over the context requests of the cached, the
        # new and the total KV tokens.
        self.ctx_cached_token_indptr = self.get_buffer(size, torch.int64, "ctx_cached_token_indptr")
        self.ctx_uncached_token_indptr = self.get_buffer(
            size, torch.int64, "ctx_uncached_token_indptr"
        )
        self.ctx_kv_indptr = self.get_buffer(size, torch.int64, "ctx_kv_indptr")
        self.host_ctx_cached_token_indptr = self._host_like(self.ctx_cached_token_indptr)
        self.host_ctx_uncached_token_indptr = self._host_like(self.ctx_uncached_token_indptr)
        self.host_ctx_kv_indptr = self._host_like(self.ctx_kv_indptr)
        self.num_ctx_cached_tokens = 0
        self.max_ctx_cached_token_len = 0
        self.max_ctx_kv_len = 0
        self.max_ctx_seq_len = 0

    @staticmethod
    def _host_like(buffer: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(buffer, device="cpu", pin_memory=prefer_pinned())

    def prepare(self, inputs: "BatchInputs") -> None:
        core = self.get_metadata(TrtllmCoreMetadata)
        num_contexts = core.num_contexts
        if num_contexts == 0:
            self.num_ctx_cached_tokens = 0
            self.max_ctx_cached_token_len = 0
            self.max_ctx_kv_len = 0
            self.max_ctx_seq_len = 0
            return
        if not inputs.stage_host:
            return
        cached_token_lens = torch.tensor(
            inputs.kv_cache_params.num_cached_tokens_per_seq[:num_contexts],
            dtype=torch.int,
            device="cpu",
        )
        seq_lens = core.seq_lens[:num_contexts]
        kv_lens = cached_token_lens + seq_lens
        self.num_ctx_cached_tokens = cached_token_lens.sum().item()
        self.max_ctx_cached_token_len = cached_token_lens.max().item()
        self.max_ctx_kv_len = kv_lens.max().item()
        self.max_ctx_seq_len = seq_lens.max().item()
        end = num_contexts + 1
        for lens, host, device in (
            (cached_token_lens, self.host_ctx_cached_token_indptr, self.ctx_cached_token_indptr),
            (seq_lens, self.host_ctx_uncached_token_indptr, self.ctx_uncached_token_indptr),
            (kv_lens, self.host_ctx_kv_indptr, self.ctx_kv_indptr),
        ):
            torch.cumsum(lens, dim=0, dtype=torch.int64, out=host[1:end])
            device[:end].copy_(host[:end], non_blocking=True)
