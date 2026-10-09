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
"""Triton variant provider for the fused DSA decode-metadata kernel.

``_fused_dsa_decode_metadata_kernel`` (DeepSeek V3.2 / V4, GLM-5) runs on
pure-decode batches. Its only batch-dependent specialization is
``BLOCK_S = next_power_of_2(num_seqs)`` (``num_seqs`` and ``num_tokens`` are
``do_not_specialize``), so a process needs at most one variant per power of
two up to ``max_batch_size``. Decode batches at a captured CUDA-graph size
compile theirs during warmup; eager decode batches (above the largest graph,
or with graphs off) would compile the rest on the executor.

The fixed launch parameters are taken from the first real call (warmup
always issues one when the fused path is enabled), so nothing here
duplicates the metadata object's buffer layout. Every BLOCK_S variant is
then planned once, in the background.
"""

from types import SimpleNamespace
from typing import Any, Iterator, List, Optional

import torch

from .....jit_prefetch import KernelCall, shadow_launches
from . import kernels


class DsaDecodeMetadataProvider:
    def __init__(self, max_batch_size: int):
        self.max_batch_size = int(max_batch_size)
        self._done = False

    def __bool__(self):
        # Known only after warmup ran the fused kernel once.
        return bool(kernels.FUSED_DSA_DECODE_TEMPLATE) and self.max_batch_size > 0

    def __call__(self, batch_ctx: Any, seen: Optional[set] = None) -> List[KernelCall]:
        # Every variant is planned by the background enumeration; a scheduled
        # batch adds nothing it does not already cover.
        if not getattr(batch_ctx, "dsa_all_block_s", False):
            return []
        return self.plan_all()

    def enumerate_batches(self) -> Iterator[Any]:
        if self:
            yield SimpleNamespace(dsa_all_block_s=True, ctx_chunk_lens=[])

    def num_seqs_classes(self) -> List[int]:
        out, n = [], 1
        while n <= self.max_batch_size:
            out.append(n)
            n *= 2
        if out and out[-1] < self.max_batch_size:
            out.append(self.max_batch_size)
        return out

    def plan_all(self) -> List[KernelCall]:
        t = dict(kernels.FUSED_DSA_DECODE_TEMPLATE)
        if not t:
            return []
        meta = torch.device("meta")
        calls: List[KernelCall] = []
        q = t["max_query_len"]
        s0, s1 = t["block_offsets_stride"]
        for n in self.num_seqs_classes():
            tokens = n * q
            i32 = dict(dtype=torch.int32, device=meta)
            i64 = dict(dtype=torch.int64, device=meta)
            block_offsets = torch.empty_strided((n, t["max_blocks"]), (s0, s1), **i32)
            with shadow_launches() as rec:
                kernels.fused_dsa_decode_metadata(
                    torch.empty(n, **i32),
                    torch.empty(n, **i32),
                    block_offsets,
                    torch.empty(tokens, **i32),
                    torch.empty(tokens, **i64),
                    torch.empty(tokens, **i64),
                    torch.empty(n + 1, **i64),
                    torch.empty(n + 1, **i64),
                    num_tokens=tokens,
                    max_query_len=q,
                    tokens_per_block=t["tokens_per_block"],
                    index_head_dim=t["index_head_dim"],
                    quant_block_size=t["quant_block_size"],
                    data_bytes_per_token=t["data_bytes_per_token"],
                )
            calls.extend(rec)
        return calls
