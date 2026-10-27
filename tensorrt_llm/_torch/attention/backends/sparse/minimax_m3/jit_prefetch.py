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
"""Triton variant provider for MiniMax-M3's Triton sparse attention.

The Triton sparse backend (``triton_backend.py``) runs two Triton kernels in
every sparse layer, prefill and decode alike, on tensors it allocates itself
(so every pointer is aligned and every tensor contiguous):

* ``_block_max_score_kernel`` over ``qk`` ``[total_q, index_heads, max_k]``,
  writing ``[total_q, index_heads, n_blocks]``, ``n_blocks = ceil(max_k /
  block_size)``;
* ``_sparse_softmax_kernel`` over ``qk`` ``[rows, kv_heads, group, max_k]``,
  with ``BLOCK_K`` the largest power of two in [16, 128] not above ``max_k``.

What varies from batch to batch is the row count (``total_q``, or a chunk of
it) and ``max_k`` (the longest sequence of the batch). Triton specializes each
integer argument on whether it is 1 or a multiple of 16, and the strides are
the head counts times ``max_k`` or ``n_blocks``, so a variant is fixed by
whether the row count is 1 or a multiple of 16, by ``BLOCK_K``, and by the
power of two dividing ``max_k`` and ``n_blocks`` (up to 16). Every such class
up to the engine's longest sequence is planned once, in the background, by
calling the real wrappers on ``meta`` tensors.
"""

from types import SimpleNamespace
from typing import Any, Iterator, List, Optional

import torch

from .....jit_prefetch import KernelCall, shadow_launches

# Row counts covering Triton's integer classes: == 1, multiple of 16, other.
_ROWS = (1, 16, 17)


def _two_adic(n: int) -> int:
    """The exponent of the largest power of two dividing ``n``, capped at 4
    (a multiple of 16 times anything is a multiple of 16)."""
    return min((n & -n).bit_length() - 1, 4)


def _softmax_block_k(max_k: int) -> int:
    """``BLOCK_K`` as ``triton_sparse_softmax`` picks it."""
    block = 128
    while block > max_k and block > 16:
        block //= 2
    return max(block, 16)


class MiniMaxM3TritonProvider:
    """Plans every variant of the Triton sparse backend's two kernels."""

    kernel_modules = ("tensorrt_llm._torch.attention.backends.sparse.minimax_m3.triton_kernels",)

    def __init__(self, model, max_seq_len: int):
        from .triton_backend import MiniMaxM3SparseRuntimeBackend

        configs = {}
        for mod in model.modules() if model is not None else ():
            backend = mod.__dict__.get("attn")
            if isinstance(backend, MiniMaxM3SparseRuntimeBackend):
                cfg = backend.m3_config
                key = (cfg.num_index_heads, cfg.num_kv_heads, cfg.num_q_heads, cfg.block_size)
                configs.setdefault(key, cfg)
        self.configs = list(configs.values())
        self.max_seq_len = max(1, int(max_seq_len))

    def __bool__(self):
        return bool(self.configs)

    def __call__(self, batch_ctx: Any, seen: Optional[set] = None) -> List[KernelCall]:
        # Every variant is planned by the background enumeration.
        if not getattr(batch_ctx, "minimax_m3_triton_all", False):
            return []
        return self.plan_all()

    def enumerate_batches(self) -> Iterator[Any]:
        if self:
            yield SimpleNamespace(minimax_m3_triton_all=True, ctx_chunk_lens=[])

    def max_k_values(self, block_size: int) -> List[int]:
        """One ``max_k`` per class of the arguments it determines, up to the
        longest sequence. The classes repeat with a period of 16 blocks once
        ``BLOCK_K`` stops growing, so only that far is scanned."""
        top = min(self.max_seq_len, 128 + 32 * block_size)
        seen, out = set(), []
        for m in range(1, top + 1):
            n = -(-m // block_size)
            sig = (m == 1, _two_adic(m), _softmax_block_k(m), n == 1, _two_adic(n))
            if sig not in seen:
                seen.add(sig)
                out.append(m)
        return out

    def plan_all(self) -> List[KernelCall]:
        from .triton_kernels import triton_block_max_score, triton_sparse_softmax

        meta = torch.device("meta")
        with shadow_launches() as rec:
            for cfg in self.configs:
                g = cfg.num_q_heads // cfg.num_kv_heads
                for max_k in self.max_k_values(cfg.block_size):
                    for rows in _ROWS:
                        triton_block_max_score(
                            torch.empty(rows, cfg.num_index_heads, max_k, device=meta),
                            cfg.block_size,
                        )
                        triton_sparse_softmax(
                            torch.empty(rows, cfg.num_kv_heads, g, max_k, device=meta),
                            torch.empty(
                                rows, cfg.num_kv_heads, max_k, dtype=torch.bool, device=meta
                            ),
                        )
        return list(rec)


def register_providers(prefetcher, engine) -> None:
    """Register MiniMax-M3's sparse-attention Triton provider and its MSA
    decode provider at the end of warmup."""
    from .jit_prefetch_msa import MiniMaxM3MsaDecodeProvider

    m3_provider = MiniMaxM3TritonProvider(engine.model, engine.max_seq_len)
    if m3_provider:
        prefetcher.register("minimax_m3_triton", m3_provider)
    msa_provider = MiniMaxM3MsaDecodeProvider(engine.batch_size)
    if msa_provider:
        prefetcher.register("minimax_m3_msa", msa_provider)
