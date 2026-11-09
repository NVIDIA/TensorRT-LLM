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

import contextlib
from typing import Callable, Iterator

import torch

import tensorrt_llm
from tensorrt_llm._torch.attention.backends.fmha.interface import Fmha, FmhaPhase
from tensorrt_llm._torch.attention.backends.fmha.phased import FmhaParams, PhasedFmha
from tensorrt_llm._torch.attention.backends.interface import (
    AttentionForwardArgs,
    AttentionRuntimeFeatures,
)
from tensorrt_llm._torch.attention.backends.trtllm import TrtllmAttention, TrtllmAttentionMetadata
from tensorrt_llm._torch.metadata import KVCacheParams
from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager
from tensorrt_llm._utils import torch_dtype_to_binding
from tensorrt_llm.llmapi.llm_args import KvCacheConfig
from tensorrt_llm.mapping import Mapping


class FakeAttention:
    def __init__(self, local_layer_idx: int = 0) -> None:
        self.is_mla_enable = False
        self.kv_lora_rank = None
        self.v_head_dim = None
        self.head_dim = 4
        self.qk_rope_head_dim = 0
        self.rope_append = None
        self.num_heads = 1
        self.num_kv_heads = 1
        self.predicted_tokens_per_seq = 1
        self.has_fp8_kv_cache = False
        self.skip_correction_threshold = 0.0
        self.local_layer_idx = local_layer_idx

    def out_head_size(self, is_gen_only: bool) -> int:
        """Extents the fake layer reports, tolerant of its unset MLA dimensions."""
        if not is_gen_only:
            return self.v_head_dim if self.is_mla_enable and self.v_head_dim else self.head_dim
        kv_lora_rank = self.kv_lora_rank or 0
        if not (self.is_mla_enable and kv_lora_rank):
            return self.head_dim
        return kv_lora_rank if self.rope_append else kv_lora_rank + self.qk_rope_head_dim


class FakePhasedFmha(PhasedFmha):
    def __init__(
        self,
        attn: FakeAttention,
        supported_phases: set[FmhaPhase | None],
        name: str,
        events: list[tuple],
        workspace_size: int = 0,
        support_predicate: Callable[[object, FmhaPhase | None], bool] | None = None,
    ) -> None:
        super().__init__(attn)
        self._supported_phases = supported_phases
        self._name = name
        self._events = events
        self._workspace_size = workspace_size
        self._support_predicate = support_predicate

    def _is_supported(
        self,
        q: torch.Tensor,
        k: torch.Tensor | None,
        v: torch.Tensor | None,
        metadata: object,
        forward_args: AttentionForwardArgs,
        *,
        phase: FmhaPhase | None = None,
    ) -> bool:
        self._events.append(("support", self._name, phase))
        return phase in self._supported_phases and (
            self._support_predicate is None or self._support_predicate(metadata, phase)
        )

    def prepare_workspace(
        self,
        q: torch.Tensor,
        k: torch.Tensor | None,
        v: torch.Tensor | None,
        metadata: object,
        forward_args: AttentionForwardArgs,
        workspace: torch.Tensor,
    ) -> None:
        self._events.append(("prepare", self._name))
        if workspace.numel() < self._workspace_size:
            workspace.resize_(self._workspace_size)

    def run_context(self, params: FmhaParams) -> None:
        self._events.append(
            (
                "run",
                self._name,
                FmhaPhase.CONTEXT,
                params.num_tokens,
                params.batch_size,
                params.num_requests,
            )
        )

    def run_generation(self, params: FmhaParams) -> None:
        self._events.append(
            (
                "run",
                self._name,
                FmhaPhase.GENERATION,
                params.num_tokens,
                params.batch_size,
                params.num_requests,
            )
        )


class FakeFmha(Fmha):
    def __init__(
        self,
        attn: FakeAttention,
        name: str,
        events: list[tuple],
        support_predicate: Callable[[AttentionForwardArgs], bool] | None = None,
        request_support_predicate: Callable[[torch.Tensor, object], bool] | None = None,
    ) -> None:
        super().__init__(attn)
        self._name = name
        self._events = events
        self._support_predicate = support_predicate
        self._request_support_predicate = request_support_predicate

    def _is_supported(
        self,
        q: torch.Tensor,
        k: torch.Tensor | None,
        v: torch.Tensor | None,
        metadata: object,
        forward_args: AttentionForwardArgs,
        *,
        phase: FmhaPhase | None = None,
    ) -> bool:
        self._events.append(("support", self._name, phase))
        return (self._support_predicate is None or self._support_predicate(forward_args)) and (
            self._request_support_predicate is None or self._request_support_predicate(q, metadata)
        )

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor | None,
        v: torch.Tensor | None,
        metadata: object,
        forward_args: AttentionForwardArgs,
    ) -> None:
        self._events.append(("forward", self._name))


@contextlib.contextmanager
def native_batch(
    dtype: torch.dtype,
    *,
    num_contexts: int,
    num_generations: int,
    chunked_prefill: bool = False,
    tokens_per_block: int = 32,
    extra_page_sizes: frozenset[int] = frozenset(),
) -> Iterator[tuple[TrtllmAttention, TrtllmAttentionMetadata, torch.Tensor]]:
    """A TrtllmAttention layer and one prepared batch over a real KV cache.

    Yields the layer, the metadata and the batch's fused QKV rows: ``num_contexts`` fresh
    8-token contexts, then ``num_generations`` one-token steps over an 8-token past.
    """
    num_heads, num_kv_heads, head_dim, length = 4, 1, 64, 8
    manager = KVCacheManager(
        KvCacheConfig(max_tokens=1024, enable_block_reuse=False),
        tensorrt_llm.bindings.internal.batch_manager.CacheType.SELF,
        num_layers=1,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        tokens_per_block=tokens_per_block,
        max_seq_len=1024,
        max_batch_size=8,
        mapping=Mapping(world_size=1, tp_size=1, rank=0),
        dtype=torch_dtype_to_binding(dtype),
    )
    manager.trtllm_gen_extra_tokens_per_block = extra_page_sizes
    try:
        num_seqs = num_contexts + num_generations
        is_gen = [i >= num_contexts for i in range(num_seqs)]
        manager.add_dummy_requests(list(range(num_seqs)), [length + g for g in is_gen])
        seq_lens = [1 if g else length for g in is_gen]
        metadata = TrtllmAttentionMetadata(
            max_num_requests=8,
            max_num_tokens=1024,
            kv_cache_manager=manager,
            runtime_features=AttentionRuntimeFeatures(chunked_prefill=chunked_prefill),
            num_contexts=num_contexts,
            seq_lens=torch.tensor(seq_lens, dtype=torch.int),
            request_ids=list(range(num_seqs)),
            prompt_lens=[length] * num_seqs,
            kv_cache_params=KVCacheParams(
                use_cache=True, num_cached_tokens_per_seq=[length if g else 0 for g in is_gen]
            ),
        )
        metadata.prepare()
        attn = TrtllmAttention(
            layer_idx=0, num_heads=num_heads, head_dim=head_dim, num_kv_heads=num_kv_heads
        )
        qkv = torch.randn(
            sum(seq_lens), (num_heads + 2 * num_kv_heads) * head_dim, device="cuda"
        ).to(dtype)
        yield attn, metadata, qkv
    finally:
        manager.shutdown()


def native_forward_args(
    attn: TrtllmAttention,
    metadata: TrtllmAttentionMetadata,
    qkv: torch.Tensor,
    **overrides: object,
) -> AttentionForwardArgs:
    """Forward arguments of a fused-QKV self-attention call writing a fresh output, as
    ``TrtllmAttention.forward`` prepares them."""
    attn.local_layer_idx = attn.get_local_layer_idx(metadata)
    values = {
        "attention_window_size": metadata.max_seq_len,
        "output": torch.empty(
            (qkv.shape[0], attn.num_heads * attn.head_dim), dtype=qkv.dtype, device=qkv.device
        ),
        "is_fused_qkv": True,
    }
    values.update(overrides)
    return AttentionForwardArgs(**values)
