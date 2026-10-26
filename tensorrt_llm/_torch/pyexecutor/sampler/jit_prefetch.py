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
"""Triton variant provider for the sampler's kernels (see
``tensorrt_llm/_torch/jit_prefetch.py``).

``TorchSampler`` writes the end-ID and maximum-length finish reasons of a
step with ``fused_write_finish_reasons`` (``ops/triton.py``). The launch reads
the sampler's own buffers, whose layouts are fixed when it is constructed,
and two per-step tensors (the sequence slots and lengths); the request count
is only the grid. So its variants are known before warmup, which samples the
first time in its first forwards: the provider plans them then
(``early_batches``).
"""

from types import SimpleNamespace
from typing import Any, Iterator, List, Optional

import torch

from ...jit_prefetch import JitPrefetcher, KernelCall, prefetch_enabled, shadow_launches


class FinishReasonsProvider:
    """Plans the ``fused_write_finish_reasons`` launches of a ``TorchSampler``."""

    kernel_modules = ("tensorrt_llm._torch.pyexecutor.sampler.ops.triton",)
    # The sequence lengths are built as int32 or int64 tensors, depending on
    # the sampler path; the sequence slots are always int64.
    SEQ_LEN_DTYPES = (torch.int32, torch.int64)

    def __init__(self, sampler):
        handler = getattr(sampler, "_finish_reasons_handler", None)
        new_tokens = getattr(getattr(sampler, "store", None), "new_tokens", None)
        self._handler = handler if isinstance(new_tokens, torch.Tensor) else None
        self._new_tokens = new_tokens

    def __bool__(self):
        return self._handler is not None

    def early_batches(self) -> Iterator[Any]:
        if self:
            yield SimpleNamespace(finish_reasons=True, ctx_chunk_lens=[])

    def __call__(self, batch_ctx: Any, seen: Optional[set] = None) -> List[KernelCall]:
        if not getattr(batch_ctx, "finish_reasons", False) or not self:
            return []
        from tensorrt_llm.bindings.executor import FinishReason

        from .ops.triton import fused_write_finish_reasons

        handler = self._handler
        store = handler.store
        meta = torch.device("meta")
        with shadow_launches() as rec:
            for seq_len_dtype in self.SEQ_LEN_DTYPES:
                fused_write_finish_reasons(
                    finish_reasons=store.finish_reasons_cuda,
                    new_tokens=self._new_tokens,
                    seq_slots=torch.empty(2, dtype=torch.int64, device=meta),
                    seq_lens=torch.empty(2, dtype=seq_len_dtype, device=meta),
                    max_lengths=store.max_lengths_cuda,
                    end_ids=store.end_ids_cuda,
                    max_tokens=handler._max_tokens,
                    beam_width=handler._max_beam_width,
                    not_finished_value=FinishReason.NOT_FINISHED.value,
                    length_value=FinishReason.LENGTH.value,
                    end_id_value=FinishReason.END_ID.value,
                )
        return list(rec)


def register_sampler_providers(sampler, rank: int) -> bool:
    """Register the providers of ``sampler``'s Triton kernels with this
    process's JIT prefetcher, before warmup: the engine queues their variants
    before warmup's first forward (``JitPrefetcher.queue_early_batches``).
    Returns False if compile ahead is off or there is nothing to plan."""
    if not prefetch_enabled():
        return False
    provider = FinishReasonsProvider(sampler)
    if not provider:
        return False
    prefetcher = JitPrefetcher.init(rank)
    return prefetcher is not None and prefetcher.register("sampler_finish_reasons", provider)
