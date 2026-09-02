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
from unittest.mock import patch

import pytest
import torch

from tensorrt_llm._torch.attention_backend import trtllm as trtllm_backend
from tensorrt_llm._torch.attention_backend.fmha.combined import CombinedFmha
from tensorrt_llm._torch.attention_backend.fmha.flashinfer_trtllm_gen import FlashInferTrtllmGenFmha
from tensorrt_llm._torch.attention_backend.fmha.interface import Fmha, FmhaPhase
from tensorrt_llm._torch.attention_backend.fmha.phased import FmhaParams, PhasedFmha
from tensorrt_llm._torch.attention_backend.fmha.triton_custom_mask import TritonCustomMaskFmha
from tensorrt_llm._torch.attention_backend.interface import (
    AttentionForwardArgs,
    AttentionInputType,
    PredefinedAttentionMask,
)
from tensorrt_llm._torch.attention_backend.trtllm import TrtllmAttention
from tensorrt_llm.quantization.mode import QuantMode


class _TestAttention:
    def __init__(self) -> None:
        self.is_mla_enable = False
        self.kv_lora_rank = None
        self.v_head_dim = None
        self.head_dim = 4
        self.num_heads = 1
        self.num_kv_heads = 1
        self.predicted_tokens_per_seq = 1
        self.flashinfer_mla_backend = None
        self.has_fp8_kv_cache = False


class _TestPhasedFmha(PhasedFmha):
    def __init__(
        self,
        attn: _TestAttention,
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
        self._events.append(("run", self._name, FmhaPhase.CONTEXT, params.num_tokens))

    def run_generation(self, params: FmhaParams) -> None:
        self._events.append(("run", self._name, FmhaPhase.GENERATION, params.num_tokens))


class _TestFmha(Fmha):
    def __init__(
        self,
        attn: _TestAttention,
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


def test_select_non_mla_fmha_combines_supported_phases() -> None:
    events: list[tuple] = []
    attn = _TestAttention()
    context_fmha = _TestPhasedFmha(attn, {FmhaPhase.CONTEXT}, "context", events)
    generation_fmha = _TestPhasedFmha(attn, {FmhaPhase.GENERATION}, "generation", events)
    attn.fmha_libs = [context_fmha, generation_fmha]
    attn.phased_fmha_libs = [context_fmha, generation_fmha]

    selected = TrtllmAttention._select_non_mla_fmha(
        attn,
        torch.empty((2, 4)),
        None,
        None,
        SimpleNamespace(num_contexts=1, num_generations=1),
        AttentionForwardArgs(attention_input_type=AttentionInputType.mixed),
    )

    assert isinstance(selected, CombinedFmha)
    assert selected._get_context_impl() is context_fmha
    assert selected._get_generation_impl() is generation_fmha
    assert events == [
        ("support", "context", None),
        ("support", "context", FmhaPhase.CONTEXT),
        ("support", "context", FmhaPhase.GENERATION),
        ("support", "generation", None),
        ("support", "generation", FmhaPhase.GENERATION),
    ]


def test_select_non_mla_fmha_checks_followup_support() -> None:
    events: list[tuple] = []
    attn = _TestAttention()
    context_fmha = _TestPhasedFmha(attn, {FmhaPhase.CONTEXT}, "context", events)
    unsupported_followup = _TestPhasedFmha(attn, set(), "followup", events)
    backend = SimpleNamespace(
        fmha_libs=[context_fmha, unsupported_followup],
        phased_fmha_libs=[context_fmha, unsupported_followup],
    )

    selected = TrtllmAttention._select_non_mla_fmha(
        backend,
        torch.empty((2, 4)),
        None,
        None,
        SimpleNamespace(num_contexts=1, num_generations=1),
        AttentionForwardArgs(attention_input_type=AttentionInputType.mixed),
    )

    assert selected is None
    assert events == [
        ("support", "context", None),
        ("support", "context", FmhaPhase.CONTEXT),
        ("support", "context", FmhaPhase.GENERATION),
        ("support", "followup", None),
        ("support", "followup", FmhaPhase.GENERATION),
    ]


def test_select_non_mla_fmha_reuses_one_implementation() -> None:
    events: list[tuple] = []
    attn = _TestAttention()
    fmha = _TestPhasedFmha(
        attn,
        {None, FmhaPhase.CONTEXT, FmhaPhase.GENERATION},
        "both",
        events,
    )
    backend = SimpleNamespace(
        fmha_libs=[fmha],
        phased_fmha_libs=[fmha],
    )

    selected = TrtllmAttention._select_non_mla_fmha(
        backend,
        torch.empty((2, 4)),
        None,
        None,
        SimpleNamespace(num_contexts=1, num_generations=1),
        AttentionForwardArgs(attention_input_type=AttentionInputType.mixed),
    )

    assert selected is fmha
    assert events == [("support", "both", None)]


def test_select_non_mla_fmha_skips_same_phased_implementation_after_full_rejection() -> None:
    events: list[tuple] = []
    attn = _TestAttention()
    phased_fmha = _TestPhasedFmha(
        attn,
        {FmhaPhase.CONTEXT, FmhaPhase.GENERATION},
        "phased",
        events,
    )
    fallback_fmha = _TestFmha(attn, "fallback", events)
    backend = SimpleNamespace(
        fmha_libs=[phased_fmha, fallback_fmha],
        phased_fmha_libs=[phased_fmha],
    )

    selected = TrtllmAttention._select_non_mla_fmha(
        backend,
        torch.empty((2, 4)),
        None,
        None,
        SimpleNamespace(num_contexts=1, num_generations=1),
        AttentionForwardArgs(attention_input_type=AttentionInputType.mixed),
    )

    assert selected is fallback_fmha
    assert events == [
        ("support", "phased", None),
        ("support", "phased", FmhaPhase.CONTEXT),
        ("support", "phased", FmhaPhase.GENERATION),
        ("support", "fallback", None),
    ]


def test_select_non_mla_fmha_uses_context_only_phased_implementation() -> None:
    events: list[tuple] = []
    attn = _TestAttention()
    context_fmha = _TestPhasedFmha(attn, {FmhaPhase.CONTEXT}, "context", events)
    backend = SimpleNamespace(
        fmha_libs=[context_fmha],
        phased_fmha_libs=[context_fmha],
    )

    selected = TrtllmAttention._select_non_mla_fmha(
        backend,
        torch.empty((2, 4)),
        None,
        None,
        SimpleNamespace(num_contexts=1, num_generations=0),
        AttentionForwardArgs(attention_input_type=AttentionInputType.mixed),
    )

    assert selected is context_fmha
    assert events == [
        ("support", "context", None),
        ("support", "context", FmhaPhase.CONTEXT),
    ]


def test_select_non_mla_fmha_preserves_registry_order() -> None:
    events: list[tuple] = []
    attn = _TestAttention()
    non_phased_fmha = _TestFmha(attn, "non-phased", events)
    phased_fmha = _TestPhasedFmha(
        attn,
        {FmhaPhase.CONTEXT, FmhaPhase.GENERATION},
        "phased",
        events,
    )
    backend = SimpleNamespace(
        fmha_libs=[non_phased_fmha, phased_fmha],
        phased_fmha_libs=[phased_fmha],
    )

    selected = TrtllmAttention._select_non_mla_fmha(
        backend,
        torch.empty((2, 4)),
        None,
        None,
        SimpleNamespace(num_contexts=1, num_generations=1),
        AttentionForwardArgs(attention_input_type=AttentionInputType.mixed),
    )

    assert selected is non_phased_fmha
    assert events == [("support", "non-phased", None)]


def test_combined_fmha_delegates_phases_and_prepares_max_workspace() -> None:
    events: list[tuple] = []
    attn = _TestAttention()
    context_fmha = _TestPhasedFmha(
        attn,
        {FmhaPhase.CONTEXT},
        "context",
        events,
        workspace_size=8,
    )
    generation_fmha = _TestPhasedFmha(
        attn,
        {FmhaPhase.GENERATION},
        "generation",
        events,
        workspace_size=4,
    )
    combined_fmha = CombinedFmha(attn)
    combined_fmha.set_fmha_impls(context_fmha, generation_fmha)
    metadata = SimpleNamespace(
        kv_cache_block_offsets=object(),
        effective_workspace=torch.empty(0, dtype=torch.uint8),
        num_contexts=1,
        num_ctx_tokens=2,
        num_generations=1,
        kv_lens_cuda_runtime=torch.tensor([2, 5], dtype=torch.int32),
        kv_lens_runtime=torch.tensor([2, 5], dtype=torch.int32),
        prompt_lens_cuda_runtime=torch.tensor([2, 1], dtype=torch.int32),
        prompt_lens_cpu_runtime=torch.tensor([2, 1], dtype=torch.int32),
        beam_width=1,
        cache_indirection=None,
        tokens_per_block=32,
        kv_cache_manager=None,
        is_cross=False,
        is_spec_decoding_enabled=False,
    )
    forward_args = AttentionForwardArgs(
        output=torch.empty((3, 4)),
        attention_input_type=AttentionInputType.mixed,
        attention_window_size=8,
    )

    combined_fmha.forward(torch.empty((3, 4)), None, None, metadata, forward_args)

    assert events == [
        ("prepare", "context"),
        ("prepare", "generation"),
        ("run", "context", FmhaPhase.CONTEXT, 2),
        ("run", "generation", FmhaPhase.GENERATION, 1),
    ]
    assert metadata.effective_workspace.numel() == 8


def test_combined_fmha_uses_flattened_v2_page_bound() -> None:
    attn = _TestAttention()
    combined_fmha = CombinedFmha(attn)
    kv_cache_manager = SimpleNamespace(
        impl=SimpleNamespace(get_page_index_upper_bound=lambda: 23),
        blocks_in_primary_pool=23,
        num_local_layers=4,
    )

    assert (
        combined_fmha._get_total_num_blocks(SimpleNamespace(kv_cache_manager=kv_cache_manager))
        == 23
    )


def test_flashinfer_fp8_mode_remains_implementation_local() -> None:
    attn = _TestAttention()
    attn.quant_mode = int(QuantMode.from_description(use_fp8_kv_cache=True))
    fmha = FlashInferTrtllmGenFmha(attn)
    output = torch.empty(1, dtype=torch.bfloat16)

    assert fmha._use_fp8_context_fmha(output, AttentionInputType.context_only)
    assert fmha._use_fp8_context_fmha(output, AttentionInputType.mixed)
    assert not fmha._use_fp8_context_fmha(output, AttentionInputType.generation_only)


def test_triton_custom_mask_rejects_whole_request_probe() -> None:
    fmha = object.__new__(TritonCustomMaskFmha)

    assert not fmha.is_supported(
        torch.empty((1, 4)),
        None,
        None,
        SimpleNamespace(),
        AttentionForwardArgs(),
    )


@pytest.mark.parametrize("is_fused_qkv", [True, False], ids=["fused_qkv", "q_only"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("sm_version", [100, 103])
@pytest.mark.parametrize("tokens_per_block", [32, 64])
@pytest.mark.parametrize("num_contexts", [1, 4, 5])
@pytest.mark.parametrize("head_dim", [64, 256, 512])
def test_flashinfer_context_fallback_scope(
    monkeypatch: pytest.MonkeyPatch,
    dtype: torch.dtype,
    sm_version: int,
    tokens_per_block: int,
    num_contexts: int,
    head_dim: int,
    is_fused_qkv: bool,
) -> None:
    monkeypatch.setattr(
        "tensorrt_llm._torch.attention_backend.fmha.flashinfer_trtllm_gen.get_sm_version",
        lambda: sm_version,
    )
    fmha = object.__new__(FlashInferTrtllmGenFmha)
    fmha.kv_factor = 2
    attn = _TestAttention()
    attn.sparse_params = None
    attn.position_embedding_type = 0
    attn.head_dim = head_dim
    q_hidden_size = attn.num_heads * attn.head_dim
    # Both layouts the trtllm-gen self-attention path accepts, wired the way
    # TrtllmAttention.forward wires them: fused QKV carries K/V inline and writes
    # the cache, while a Q-only input only reads an already-populated cache.
    input_hidden_size = q_hidden_size
    if is_fused_qkv:
        input_hidden_size += 2 * attn.num_kv_heads * attn.head_dim
    q = torch.empty((num_contexts, input_hidden_size), dtype=dtype)
    metadata = SimpleNamespace(
        num_contexts=num_contexts,
        helix_position_offsets=None,
        num_sparse_topk=0,
        use_spec_decoding=False,
        is_spec_dec_tree=False,
        kv_cache_block_offsets=object(),
        kv_cache_manager=None,
        is_cross=False,
        is_spec_decoding_enabled=False,
        tokens_per_block=tokens_per_block,
        beam_width=1,
    )
    forward_args = AttentionForwardArgs(
        output=torch.empty((num_contexts, q_hidden_size), dtype=dtype),
        attention_input_type=AttentionInputType.mixed,
        is_fused_qkv=is_fused_qkv,
        update_kv_cache=is_fused_qkv,
    )

    small_bf16_fallback = (
        dtype == torch.bfloat16 and num_contexts <= 4 and is_fused_qkv and head_dim != 512
    )
    expected_fallback = small_bf16_fallback or sm_version == 103
    for phase in (None, FmhaPhase.CONTEXT):
        supported, reason = fmha._is_supported_with_reason(
            q,
            None,
            None,
            attn,
            metadata,
            forward_args,
            phase=phase,
        )
        if expected_fallback:
            assert not supported
            assert "fallback FMHA" in reason
        else:
            assert supported, reason
            assert reason == ""

    generation_supported, generation_reason = fmha._is_supported_with_reason(
        q,
        None,
        None,
        attn,
        metadata,
        forward_args,
        phase=FmhaPhase.GENERATION,
    )
    assert generation_supported, generation_reason
    assert generation_reason == ""


def _make_fmha_cache_backend(*, sanity_check_enabled: bool = False) -> TrtllmAttention:
    backend = object.__new__(TrtllmAttention)
    backend.is_mla_enable = False
    backend.kv_lora_rank = None
    backend.head_dim = 4
    backend.v_head_dim = None
    backend.fmha_libs = []
    backend.phased_fmha_libs = []
    backend.non_phased_fmha_libs = []
    backend._fmha_cache = {}
    backend._fmha_cache_inputs = {}
    backend._fmha_cache_sanity_check_enabled = sanity_check_enabled
    return backend


def _make_fmha_cache_metadata(
    *,
    num_contexts: int,
    num_generations: int,
    num_ctx_tokens: int = 0,
    use_spec_decoding: bool = False,
) -> SimpleNamespace:
    return SimpleNamespace(
        num_contexts=num_contexts,
        num_generations=num_generations,
        num_ctx_tokens=num_ctx_tokens,
        use_spec_decoding=use_spec_decoding,
    )


@pytest.mark.parametrize(("lower_q_length", "upper_q_length"), [(3, 4), (7, 8)])
@pytest.mark.parametrize("upper_first", [False, True])
def test_fmha_cache_separates_generation_q_boundaries(
    lower_q_length: int,
    upper_q_length: int,
    upper_first: bool,
) -> None:
    events: list[tuple] = []
    backend = _make_fmha_cache_backend()
    backend.is_mla_enable = True
    upper_boundary_fmha = _TestFmha(
        backend,
        "upper-boundary",
        events,
        request_support_predicate=lambda q, metadata: q.shape[0] // metadata.num_generations
        == upper_q_length,
    )
    fallback = _TestFmha(backend, "fallback", events)
    backend.fmha_libs = [upper_boundary_fmha, fallback]
    batch_size = 128
    metadata = _make_fmha_cache_metadata(num_contexts=0, num_generations=batch_size)
    forward_args = AttentionForwardArgs(attention_input_type=AttentionInputType.generation_only)
    request_order = (
        (upper_q_length, lower_q_length) if upper_first else (lower_q_length, upper_q_length)
    )

    with patch.object(trtllm_backend, "_is_fmha_cache_enabled", return_value=True):
        selected_by_q_length = {
            q_length: backend._select_fmha(
                torch.empty((batch_size * q_length, 4)), None, None, metadata, forward_args
            )
            for q_length in request_order
        }
        for q_length in request_order:
            assert (
                backend._select_fmha(
                    torch.empty((batch_size * q_length, 4)), None, None, metadata, forward_args
                )
                is selected_by_q_length[q_length]
            )

    assert selected_by_q_length == {
        lower_q_length: fallback,
        upper_q_length: upper_boundary_fmha,
    }
    assert len(backend._fmha_cache) == 2
    assert events.count(("support", "upper-boundary", None)) == 2
    assert events.count(("support", "fallback", None)) == 1


@pytest.mark.parametrize(
    ("lower_batch_size", "upper_batch_size", "q_length"),
    [(31, 32, 4), (63, 64, 1)],
)
@pytest.mark.parametrize("upper_first", [False, True])
def test_fmha_cache_separates_generation_batch_boundaries(
    lower_batch_size: int,
    upper_batch_size: int,
    q_length: int,
    upper_first: bool,
) -> None:
    events: list[tuple] = []
    backend = _make_fmha_cache_backend()
    backend.is_mla_enable = True
    upper_boundary_fmha = _TestFmha(
        backend,
        "upper-boundary",
        events,
        request_support_predicate=lambda _q, metadata: metadata.num_generations == upper_batch_size,
    )
    fallback = _TestFmha(backend, "fallback", events)
    backend.fmha_libs = [upper_boundary_fmha, fallback]
    forward_args = AttentionForwardArgs(attention_input_type=AttentionInputType.generation_only)
    request_order = (
        (upper_batch_size, lower_batch_size)
        if upper_first
        else (lower_batch_size, upper_batch_size)
    )

    with patch.object(trtllm_backend, "_is_fmha_cache_enabled", return_value=True):
        selected_by_batch_size = {
            batch_size: backend._select_fmha(
                torch.empty((batch_size * q_length, 4)),
                None,
                None,
                _make_fmha_cache_metadata(num_contexts=0, num_generations=batch_size),
                forward_args,
            )
            for batch_size in request_order
        }
        for batch_size in request_order:
            assert (
                backend._select_fmha(
                    torch.empty((batch_size * q_length, 4)),
                    None,
                    None,
                    _make_fmha_cache_metadata(num_contexts=0, num_generations=batch_size),
                    forward_args,
                )
                is selected_by_batch_size[batch_size]
            )

    assert selected_by_batch_size == {
        lower_batch_size: fallback,
        upper_batch_size: upper_boundary_fmha,
    }
    assert len(backend._fmha_cache) == 2
    assert events.count(("support", "upper-boundary", None)) == 2
    assert events.count(("support", "fallback", None)) == 1


def test_fmha_cache_reuses_grid_cell() -> None:
    events: list[tuple] = []
    backend = _make_fmha_cache_backend()
    boundary_fmha = _TestFmha(
        backend,
        "boundary",
        events,
        request_support_predicate=lambda q, metadata: (
            metadata.num_generations == 64 and q.shape[0] // metadata.num_generations == 8
        ),
    )
    fallback = _TestFmha(backend, "fallback", events)
    backend.fmha_libs = [boundary_fmha, fallback]
    forward_args = AttentionForwardArgs(attention_input_type=AttentionInputType.generation_only)

    with patch.object(trtllm_backend, "_is_fmha_cache_enabled", return_value=True):
        for batch_size, q_length in ((57, 5), (63, 7)):
            assert (
                backend._select_fmha(
                    torch.empty((batch_size * q_length, 4)),
                    None,
                    None,
                    _make_fmha_cache_metadata(num_contexts=0, num_generations=batch_size),
                    forward_args,
                )
                is fallback
            )
        assert (
            backend._select_fmha(
                torch.empty((64 * 8, 4)),
                None,
                None,
                _make_fmha_cache_metadata(num_contexts=0, num_generations=64),
                forward_args,
            )
            is boundary_fmha
        )

    assert len(backend._fmha_cache) == 2
    assert events == [
        ("support", "boundary", None),
        ("support", "fallback", None),
        ("support", "boundary", None),
    ]


def test_context_fmha_cache_uses_batch_grid_only() -> None:
    events: list[tuple] = []
    backend = _make_fmha_cache_backend()
    fmha = _TestFmha(backend, "fmha", events)
    backend.fmha_libs = [fmha]
    forward_args = AttentionForwardArgs(attention_input_type=AttentionInputType.context_only)

    with patch.object(trtllm_backend, "_is_fmha_cache_enabled", return_value=True):
        first = backend._select_fmha(
            torch.empty((57 * 3, 4)),
            None,
            None,
            _make_fmha_cache_metadata(num_contexts=57, num_generations=0, num_ctx_tokens=57 * 3),
            forward_args,
        )
        second = backend._select_fmha(
            torch.empty((63 * 17, 4)),
            None,
            None,
            _make_fmha_cache_metadata(num_contexts=63, num_generations=0, num_ctx_tokens=63 * 17),
            forward_args,
        )
        same_batch_different_q_length = backend._select_fmha(
            torch.empty((63 * 3, 4)),
            None,
            None,
            _make_fmha_cache_metadata(num_contexts=63, num_generations=0, num_ctx_tokens=63 * 3),
            forward_args,
        )

    assert first is fmha
    assert second is fmha
    assert same_batch_different_q_length is fmha
    assert len(backend._fmha_cache) == 1
    assert events == [("support", "fmha", None)]


@pytest.mark.parametrize("generation_first", [False, True])
def test_fmha_cache_separates_compacted_mla_phases(generation_first: bool) -> None:
    events: list[tuple] = []
    backend = _make_fmha_cache_backend()
    backend.is_mla_enable = True
    context_fmha = _TestPhasedFmha(backend, {FmhaPhase.CONTEXT}, "context", events)
    generation_fmha = _TestPhasedFmha(backend, {FmhaPhase.GENERATION}, "generation", events)
    backend.fmha_libs = [context_fmha, generation_fmha]
    metadata = _make_fmha_cache_metadata(num_contexts=3, num_generations=2, num_ctx_tokens=6)
    requests = {
        FmhaPhase.CONTEXT: (
            torch.empty((6, 4)),
            AttentionForwardArgs(attention_input_type=AttentionInputType.context_only),
        ),
        FmhaPhase.GENERATION: (
            torch.empty((2, 4)),
            AttentionForwardArgs(attention_input_type=AttentionInputType.generation_only),
        ),
    }
    request_order = (
        (FmhaPhase.GENERATION, FmhaPhase.CONTEXT)
        if generation_first
        else (FmhaPhase.CONTEXT, FmhaPhase.GENERATION)
    )

    with patch.object(trtllm_backend, "_is_fmha_cache_enabled", return_value=True):
        selected_by_phase = {
            phase: backend._select_fmha(
                requests[phase][0], None, None, metadata, requests[phase][1]
            )
            for phase in request_order
        }
        for phase in request_order:
            assert (
                backend._select_fmha(requests[phase][0], None, None, metadata, requests[phase][1])
                is selected_by_phase[phase]
            )

    assert selected_by_phase == {
        FmhaPhase.CONTEXT: context_fmha,
        FmhaPhase.GENERATION: generation_fmha,
    }
    assert len(backend._fmha_cache) == 2
    assert events.count(("support", "context", FmhaPhase.CONTEXT)) == 1
    assert events.count(("support", "context", FmhaPhase.GENERATION)) == 1
    assert events.count(("support", "generation", FmhaPhase.GENERATION)) == 1


def test_fmha_cache_tracks_attention_mask_data() -> None:
    for mask_data_first in (True, False):
        events: list[tuple] = []
        backend = _make_fmha_cache_backend()
        implicit_mask_only = _TestFmha(
            backend,
            "implicit-mask-only",
            events,
            support_predicate=lambda forward_args: forward_args.attention_mask_data is None,
        )
        fallback = _TestFmha(backend, "fallback", events)
        backend.fmha_libs = [implicit_mask_only, fallback]
        metadata = _make_fmha_cache_metadata(num_contexts=0, num_generations=1)
        q = torch.empty((1, 4))
        implicit_mask_args = AttentionForwardArgs(
            attention_input_type=AttentionInputType.generation_only,
            attention_mask=PredefinedAttentionMask.CAUSAL,
        )
        mask_data_args = AttentionForwardArgs(
            attention_input_type=AttentionInputType.generation_only,
            attention_mask=PredefinedAttentionMask.CAUSAL,
            attention_mask_data=torch.empty((1, 1)),
        )
        request_order = (
            (mask_data_args, implicit_mask_args)
            if mask_data_first
            else (implicit_mask_args, mask_data_args)
        )

        with patch.object(trtllm_backend, "_is_fmha_cache_enabled", return_value=True):
            for forward_args in request_order:
                fresh = backend._select_fmha_uncached(q, None, None, metadata, forward_args)
                cached = backend._select_fmha(q, None, None, metadata, forward_args)

                assert cached is fresh

        assert (
            backend._select_fmha(q, None, None, metadata, implicit_mask_args) is implicit_mask_only
        )
        assert backend._select_fmha(q, None, None, metadata, mask_data_args) is fallback
        assert len(backend._fmha_cache) == 2


@pytest.mark.parametrize("speculative_first", [False, True])
def test_fmha_cache_separates_speculative_decoding(speculative_first: bool) -> None:
    events: list[tuple] = []
    backend = _make_fmha_cache_backend()
    regular_only = _TestFmha(
        backend,
        "regular-only",
        events,
        request_support_predicate=lambda _q, metadata: not metadata.use_spec_decoding,
    )
    fallback = _TestFmha(backend, "fallback", events)
    backend.fmha_libs = [regular_only, fallback]
    q = torch.empty((4, 4))
    forward_args = AttentionForwardArgs(attention_input_type=AttentionInputType.generation_only)
    metadata_by_mode = {
        False: _make_fmha_cache_metadata(
            num_contexts=0,
            num_generations=1,
            use_spec_decoding=False,
        ),
        True: _make_fmha_cache_metadata(
            num_contexts=0,
            num_generations=1,
            use_spec_decoding=True,
        ),
    }
    request_order = (True, False) if speculative_first else (False, True)

    with patch.object(trtllm_backend, "_is_fmha_cache_enabled", return_value=True):
        selected_by_mode = {
            use_spec_decoding: backend._select_fmha(
                q,
                None,
                None,
                metadata_by_mode[use_spec_decoding],
                forward_args,
            )
            for use_spec_decoding in request_order
        }
        for use_spec_decoding in request_order:
            assert (
                backend._select_fmha(
                    q,
                    None,
                    None,
                    metadata_by_mode[use_spec_decoding],
                    forward_args,
                )
                is selected_by_mode[use_spec_decoding]
            )

    assert selected_by_mode == {False: regular_only, True: fallback}
    assert len(backend._fmha_cache) == 2
    assert events.count(("support", "regular-only", None)) == 2
    assert events.count(("support", "fallback", None)) == 1


def test_fmha_cache_tracks_lora_output_representation() -> None:
    for lora_first in (True, False):
        events: list[tuple] = []
        backend = _make_fmha_cache_backend()
        unpacked_only = _TestFmha(
            backend,
            "unpacked-only",
            events,
            support_predicate=lambda forward_args: forward_args.output is not None
            and forward_args.output.dtype == torch.bfloat16,
        )
        fallback = _TestFmha(backend, "fallback", events)
        backend.fmha_libs = [unpacked_only, fallback]
        metadata = _make_fmha_cache_metadata(num_contexts=0, num_generations=4)
        q = torch.empty((4, 4), dtype=torch.bfloat16)
        lora_args = AttentionForwardArgs(
            output=torch.empty((4, 4), dtype=torch.bfloat16),
            attention_input_type=AttentionInputType.generation_only,
        )
        base_args = AttentionForwardArgs(
            output=torch.empty((4, 2), dtype=torch.uint8),
            output_sf=torch.empty(4, dtype=torch.uint8),
            attention_input_type=AttentionInputType.generation_only,
        )
        request_order = (lora_args, base_args) if lora_first else (base_args, lora_args)

        with patch.object(trtllm_backend, "_is_fmha_cache_enabled", return_value=True):
            for forward_args in request_order:
                fresh = backend._select_fmha_uncached(q, None, None, metadata, forward_args)
                cached = backend._select_fmha(q, None, None, metadata, forward_args)

                assert cached is fresh

        assert backend._select_fmha(q, None, None, metadata, lora_args) is unpacked_only
        assert backend._select_fmha(q, None, None, metadata, base_args) is fallback
        assert len(backend._fmha_cache) == 2


def test_fmha_cache_sanity_check_logs_mismatched_inputs() -> None:
    events: list[tuple] = []
    backend = _make_fmha_cache_backend(sanity_check_enabled=True)
    preferred = _TestFmha(
        backend,
        "preferred",
        events,
        request_support_predicate=lambda q, metadata: q.shape[1] == 4,
    )
    fallback = _TestFmha(backend, "fallback", events)
    backend.fmha_libs = [preferred, fallback]
    cached_metadata = _make_fmha_cache_metadata(num_contexts=0, num_generations=1)
    cached_metadata.beam_width = 1
    uncached_metadata = _make_fmha_cache_metadata(num_contexts=0, num_generations=1)
    uncached_metadata.beam_width = 2
    cached_args = AttentionForwardArgs(
        attention_input_type=AttentionInputType.generation_only,
        relative_attention_max_distance=0,
    )
    uncached_args = AttentionForwardArgs(
        attention_input_type=AttentionInputType.generation_only,
        relative_attention_max_distance=7,
    )

    with (
        patch.object(trtllm_backend, "_is_fmha_cache_enabled", return_value=True),
        patch.object(trtllm_backend.logger, "error") as log_error,
    ):
        assert (
            backend._select_fmha(
                torch.empty((1, 4)),
                torch.empty((1, 2)),
                None,
                cached_metadata,
                cached_args,
            )
            is preferred
        )
        with pytest.raises(RuntimeError, match="FMHA cache sanity check failed"):
            backend._select_fmha(
                torch.empty((1, 8)),
                torch.empty((1, 3)),
                torch.empty((1, 3)),
                uncached_metadata,
                uncached_args,
            )

    log_error.assert_called_once()
    message = log_error.call_args.args[0]
    assert "cached=_TestFmha, uncached=_TestFmha" in message
    assert "q: cached=shape=(1, 4), uncached=shape=(1, 8)" in message
    assert "k: cached=shape=(1, 2), uncached=shape=(1, 3)" in message
    assert "v: cached=None, uncached=shape=(1, 3)" in message
    assert "metadata.beam_width: cached=1, uncached=2" in message
    assert "forward_args.relative_attention_max_distance: cached=0, uncached=7" in message


def test_fmha_cache_sanity_check_accepts_equivalent_combined_fmha() -> None:
    events: list[tuple] = []
    backend = _make_fmha_cache_backend(sanity_check_enabled=True)
    context_fmha = _TestPhasedFmha(backend, {FmhaPhase.CONTEXT}, "context", events)
    generation_fmha = _TestPhasedFmha(backend, {FmhaPhase.GENERATION}, "generation", events)
    cached_combined = CombinedFmha(backend)
    cached_combined.set_fmha_impls(context_fmha, generation_fmha)
    equivalent_combined = CombinedFmha(backend)
    equivalent_combined.set_fmha_impls(context_fmha, generation_fmha)
    metadata = _make_fmha_cache_metadata(num_contexts=1, num_generations=1, num_ctx_tokens=1)
    forward_args = AttentionForwardArgs(attention_input_type=AttentionInputType.mixed)
    q = torch.empty((2, 4))

    with (
        patch.object(trtllm_backend, "_is_fmha_cache_enabled", return_value=True),
        patch.object(
            backend,
            "_select_fmha_uncached",
            side_effect=[cached_combined, equivalent_combined],
        ) as select_uncached,
    ):
        first = backend._select_fmha(q, None, None, metadata, forward_args)
        second = backend._select_fmha(q, None, None, metadata, forward_args)

    assert first is cached_combined
    assert second is cached_combined
    assert select_uncached.call_count == 2


def test_fmha_cache_keeps_combined_selections_immutable() -> None:
    events: list[tuple] = []
    backend = _make_fmha_cache_backend()
    context_small = _TestPhasedFmha(
        backend,
        {FmhaPhase.CONTEXT},
        "context-small",
        events,
        support_predicate=lambda metadata, phase: metadata.num_contexts == 1,
    )
    context_large = _TestPhasedFmha(
        backend,
        {FmhaPhase.CONTEXT},
        "context-large",
        events,
        support_predicate=lambda metadata, phase: metadata.num_contexts > 1,
    )
    generation_small = _TestPhasedFmha(
        backend,
        {FmhaPhase.GENERATION},
        "generation-small",
        events,
        support_predicate=lambda metadata, phase: metadata.num_generations == 1,
    )
    generation_large = _TestPhasedFmha(
        backend,
        {FmhaPhase.GENERATION},
        "generation-large",
        events,
        support_predicate=lambda metadata, phase: metadata.num_generations > 1,
    )
    backend.fmha_libs = [
        context_small,
        context_large,
        generation_small,
        generation_large,
    ]
    backend.phased_fmha_libs = list(backend.fmha_libs)
    forward_args = AttentionForwardArgs(attention_input_type=AttentionInputType.mixed)
    small_metadata = _make_fmha_cache_metadata(num_contexts=1, num_generations=1, num_ctx_tokens=2)
    large_metadata = _make_fmha_cache_metadata(
        num_contexts=26, num_generations=26, num_ctx_tokens=52
    )

    with patch.object(trtllm_backend, "_is_fmha_cache_enabled", return_value=True):
        small = backend._select_fmha(torch.empty((3, 4)), None, None, small_metadata, forward_args)
        large = backend._select_fmha(torch.empty((78, 4)), None, None, large_metadata, forward_args)
        num_events_after_misses = len(events)
        small_again = backend._select_fmha(
            torch.empty((3, 4)), None, None, small_metadata, forward_args
        )

    assert isinstance(small, CombinedFmha)
    assert isinstance(large, CombinedFmha)
    assert small is small_again
    assert small is not large
    assert small._get_context_impl() is context_small
    assert small._get_generation_impl() is generation_small
    assert large._get_context_impl() is context_large
    assert large._get_generation_impl() is generation_large
    assert len(events) == num_events_after_misses


def test_fmha_cache_is_bypassed_while_autotuning() -> None:
    cases = (
        (False, AttentionInputType.mixed, 1, 1, 1, 2),
        (True, AttentionInputType.context_only, 1, 0, 1, 1),
        (True, AttentionInputType.generation_only, 0, 1, 0, 1),
    )
    for (
        is_mla_enable,
        attention_input_type,
        num_contexts,
        num_generations,
        num_ctx_tokens,
        num_q_tokens,
    ) in cases:
        events: list[tuple] = []
        backend = _make_fmha_cache_backend()
        backend.is_mla_enable = is_mla_enable
        fmha = _TestFmha(backend, "fmha", events)
        backend.fmha_libs = [fmha]
        metadata = _make_fmha_cache_metadata(
            num_contexts=num_contexts,
            num_generations=num_generations,
            num_ctx_tokens=num_ctx_tokens,
        )
        forward_args = AttentionForwardArgs(attention_input_type=attention_input_type)
        q = torch.empty((num_q_tokens, 4))

        with patch.object(trtllm_backend, "_is_fmha_cache_enabled", return_value=False):
            assert backend._select_fmha(q, None, None, metadata, forward_args) is fmha
            assert backend._select_fmha(q, None, None, metadata, forward_args) is fmha
        assert not backend._fmha_cache
        with patch.object(trtllm_backend, "_is_fmha_cache_enabled", return_value=True):
            assert backend._select_fmha(q, None, None, metadata, forward_args) is fmha
            assert backend._select_fmha(q, None, None, metadata, forward_args) is fmha
        with patch.object(trtllm_backend, "_is_fmha_cache_enabled", return_value=False):
            assert backend._select_fmha(q, None, None, metadata, forward_args) is fmha
        with patch.object(trtllm_backend, "_is_fmha_cache_enabled", return_value=True):
            assert backend._select_fmha(q, None, None, metadata, forward_args) is fmha

        assert events == [
            ("support", "fmha", None),
            ("support", "fmha", None),
            ("support", "fmha", None),
            ("support", "fmha", None),
        ]


def test_fmha_cache_does_not_cache_failed_selection() -> None:
    events: list[tuple] = []
    backend = _make_fmha_cache_backend()
    unsupported = _TestPhasedFmha(backend, set(), "unsupported", events)
    backend.fmha_libs = [unsupported]
    metadata = _make_fmha_cache_metadata(num_contexts=1, num_generations=0, num_ctx_tokens=1)
    forward_args = AttentionForwardArgs(attention_input_type=AttentionInputType.context_only)
    q = torch.empty((1, 4))

    with patch.object(trtllm_backend, "_is_fmha_cache_enabled", return_value=True):
        assert backend._select_fmha(q, None, None, metadata, forward_args) is None
        assert backend._select_fmha(q, None, None, metadata, forward_args) is None

    assert events == [
        ("support", "unsupported", FmhaPhase.CONTEXT),
        ("support", "unsupported", FmhaPhase.CONTEXT),
    ]


def test_create_fmha_libs_invalidates_fmha_cache() -> None:
    events: list[tuple] = []

    class _OldFmha(_TestFmha):
        def __init__(self, attn: _TestAttention) -> None:
            super().__init__(attn, "old", events)

    class _NewFmha(_TestFmha):
        def __init__(self, attn: _TestAttention) -> None:
            super().__init__(attn, "new", events)

    backend = _make_fmha_cache_backend()
    backend.sparse_params = None
    backend.kv_cache_dtype = "auto"
    metadata = _make_fmha_cache_metadata(num_contexts=0, num_generations=1)
    forward_args = AttentionForwardArgs(attention_input_type=AttentionInputType.generation_only)
    q = torch.empty((1, 4))

    with (
        patch.object(trtllm_backend, "get_enabled_fmha_lib_classes", return_value=[_OldFmha]),
        patch.object(trtllm_backend, "_is_fmha_cache_enabled", return_value=True),
    ):
        backend.create_fmha_libs()
        old = backend._select_fmha(q, None, None, metadata, forward_args)

    with (
        patch.object(trtllm_backend, "get_enabled_fmha_lib_classes", return_value=[_NewFmha]),
        patch.object(trtllm_backend, "_is_fmha_cache_enabled", return_value=True),
    ):
        backend.create_fmha_libs()
        new = backend._select_fmha(q, None, None, metadata, forward_args)

    assert isinstance(old, _OldFmha)
    assert isinstance(new, _NewFmha)
    assert events == [
        ("support", "old", None),
        ("support", "new", None),
    ]
