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

import torch

from tensorrt_llm._torch.attention_backend.fmha.combined import CombinedFmha
from tensorrt_llm._torch.attention_backend.fmha.flashinfer_trtllm_gen import FlashInferTrtllmGenFmha
from tensorrt_llm._torch.attention_backend.fmha.interface import Fmha, FmhaPhase
from tensorrt_llm._torch.attention_backend.fmha.phased import FmhaParams, PhasedFmha
from tensorrt_llm._torch.attention_backend.interface import AttentionForwardArgs, AttentionInputType
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


class _TestPhasedFmha(PhasedFmha):
    def __init__(
        self,
        attn: _TestAttention,
        supported_phases: set[FmhaPhase],
        name: str,
        events: list[tuple],
        workspace_size: int = 0,
    ) -> None:
        super().__init__(attn)
        self._supported_phases = supported_phases
        self._name = name
        self._events = events
        self._workspace_size = workspace_size

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
        return phase in self._supported_phases

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
    def __init__(self, attn: _TestAttention, name: str, events: list[tuple]) -> None:
        super().__init__(attn)
        self._name = name
        self._events = events

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
        return True

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
    combined_fmha = CombinedFmha(attn)
    backend = SimpleNamespace(
        fmha_libs=[context_fmha, generation_fmha],
        phased_fmha_libs=[context_fmha, generation_fmha],
        combined_fmha=combined_fmha,
    )

    selected = TrtllmAttention._select_non_mla_fmha(
        backend,
        torch.empty((2, 4)),
        None,
        None,
        SimpleNamespace(num_contexts=1, num_generations=1),
        AttentionForwardArgs(attention_input_type=AttentionInputType.mixed),
    )

    assert selected is combined_fmha
    assert events == [
        ("support", "context", FmhaPhase.CONTEXT),
        ("support", "context", FmhaPhase.GENERATION),
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
        combined_fmha=CombinedFmha(attn),
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
    assert events[-1] == ("support", "followup", FmhaPhase.GENERATION)


def test_select_non_mla_fmha_reuses_one_implementation() -> None:
    events: list[tuple] = []
    attn = _TestAttention()
    fmha = _TestPhasedFmha(
        attn,
        {FmhaPhase.CONTEXT, FmhaPhase.GENERATION},
        "both",
        events,
    )
    backend = SimpleNamespace(
        fmha_libs=[fmha],
        phased_fmha_libs=[fmha],
        combined_fmha=CombinedFmha(attn),
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
        combined_fmha=CombinedFmha(attn),
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
