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

from types import SimpleNamespace
from typing import Callable

import torch

from tensorrt_llm._torch.attention.backends.fmha.interface import Fmha, FmhaPhase
from tensorrt_llm._torch.attention.backends.fmha.phased import FmhaParams, PhasedFmha
from tensorrt_llm._torch.attention.backends.interface import AttentionForwardArgs, RopeParams
from tensorrt_llm.functional import PositionEmbeddingType
from tensorrt_llm.quantization.mode import QuantMode


def make_fake_metadata(**overrides: object) -> SimpleNamespace:
    """Attention metadata with every field ``PhasedFmha._build_params`` reads.

    That method lowers the whole per-forward contract in one pass, so a partial
    namespace fails on whichever field it happens to reach first. Defaults describe an
    unquantized, non-speculative, non-sparse batch; pass overrides for what a test is
    actually about.
    """
    defaults = dict(
        # Batch shape.
        num_contexts=0,
        num_generations=0,
        num_ctx_tokens=0,
        beam_width=1,
        max_num_requests=1,
        max_num_sequences=1,
        max_context_length=1,
        max_seq_len=1,
        max_context_q_len_override=None,
        # Sequence lengths. `host_total_kv_lens` is indexed per phase, so it needs two
        # entries even when the test does not care about the values.
        kv_lens_cuda_runtime=None,
        kv_lens_runtime=None,
        prompt_lens_cuda_runtime=None,
        prompt_lens_cpu_runtime=None,
        host_total_kv_lens=torch.zeros(2, dtype=torch.int32),
        # KV cache.
        kv_cache_manager=None,
        kv_cache_block_offsets=None,
        host_kv_cache_pool_pointers=None,
        host_kv_cache_pool_mapping=None,
        block_ids_per_seq=None,
        tokens_per_block=32,
        cache_indirection=None,
        use_paged_context_fmha=False,
        effective_workspace=None,
        # Speculative decoding.
        is_spec_decoding_enabled=False,
        use_spec_decoding=False,
        is_spec_dec_tree=False,
        force_prepare_spec_dec_tree_mask=False,
        max_total_draft_tokens=0,
        spec_decoding_packed_mask=None,
        spec_decoding_generation_lengths=None,
        spec_decoding_position_offsets_for_cpp=None,
        spec_decoding_bl_tree_mask=None,
        spec_decoding_bl_tree_mask_offset=None,
        spec_bl_tree_first_sparse_mask_offset_kv=None,
        # Sparse attention, helix, FlashMLA.
        num_sparse_topk=0,
        helix_position_offsets=None,
        helix_is_inactive_rank=None,
        flash_mla_tile_scheduler_metadata=None,
        flash_mla_num_splits=None,
        # Misc.
        is_cross=False,
        trtllm_gen_jit_warmup=False,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


class FakeAttention:
    def __init__(self) -> None:
        self.is_mla_enable = False
        self.kv_lora_rank = None
        self.qk_rope_head_dim = None
        self.v_head_dim = None
        self.head_dim = 4
        self.num_heads = 1
        self.num_kv_heads = 1
        self.predicted_tokens_per_seq = 1
        self.layer_idx = 0
        self.flashinfer_mla_backend = None
        self.has_fp8_kv_cache = False
        self.rope_append = True
        # `PhasedFmha._build_params` lowers the whole attention configuration, so the
        # fake has to answer for all of it. Values follow `TrtllmAttention`'s non-MLA
        # defaults; MLA tests override what they need.
        self.q_lora_rank = None
        self.qk_nope_head_dim = None
        self.q_scaling = 1.0
        self.attention_chunk_size = None
        self.position_embedding_type = PositionEmbeddingType.learned_absolute
        self.quant_mode = QuantMode(0)
        self.rope_params = RopeParams()
        self.rotary_inv_freq = None
        self.rotary_cos_sin = None
        self.sparse_params = None

    def get_local_layer_idx(self, metadata: object) -> int:
        return self.layer_idx

    def out_head_size(self, is_gen_only: bool) -> int:
        """Mirror ``TrtllmAttention.out_head_size``; ``PhasedFmha`` reads it at init."""
        if not self.is_mla_enable:
            return self.head_dim
        if not is_gen_only:
            return self.v_head_dim
        return self.kv_lora_rank if self.rope_append else self.kv_lora_rank + self.qk_rope_head_dim


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

    def is_supported(
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

    def prepare_workspace(self, params: FmhaParams, metadata: object) -> None:
        self._events.append(("prepare", self._name))
        workspace = params.workspace
        if workspace is not None and workspace.numel() < self._workspace_size:
            workspace.resize_(self._workspace_size)

    def run_context(self, params: FmhaParams) -> None:
        self._events.append(("run", self._name, FmhaPhase.CONTEXT, params.num_tokens))

    def run_generation(self, params: FmhaParams) -> None:
        self._events.append(("run", self._name, FmhaPhase.GENERATION, params.num_tokens))


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

    def is_supported(
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
