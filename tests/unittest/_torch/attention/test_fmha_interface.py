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

from contextlib import nullcontext
from types import SimpleNamespace
from typing import cast
from unittest.mock import Mock, patch

import pytest
import torch
from fmha_test_utils import FakeAttention

from tensorrt_llm._torch.attention.backends.fmha.fallback import FallbackFmha
from tensorrt_llm._torch.attention.backends.fmha.interface import Fmha, FmhaPhase
from tensorrt_llm._torch.attention.backends.fmha.registry import FMHA_LIBS
from tensorrt_llm._torch.attention.backends.interface import (
    AttentionForwardArgs,
    AttentionInputType,
)
from tensorrt_llm._torch.attention.backends.sparse.params import (
    BlockSparseForwardInputs,
    SparseRuntimeParams,
)
from tensorrt_llm._torch.attention.backends.trtllm import TrtllmAttention, TrtllmAttentionMetadata


class _MinimalFmha(Fmha):
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor | None,
        v: torch.Tensor | None,
        metadata: TrtllmAttentionMetadata,
        forward_args: AttentionForwardArgs,
    ) -> None:
        pass


@pytest.mark.parametrize(
    "is_cross,update_kv_cache,expected",
    [(False, False, False), (False, True, True), (True, False, True)],
)
def test_fallback_support_matches_thop_kv_update_contract(
    is_cross: bool, update_kv_cache: bool, expected: bool
) -> None:
    attn = FakeAttention()
    fmha = FallbackFmha(attn)
    metadata = SimpleNamespace(is_cross=is_cross, helix_position_offsets=None, num_contexts=0)
    forward_args = AttentionForwardArgs(is_fused_qkv=True, update_kv_cache=update_kv_cache)
    assert fmha.is_supported(None, None, None, metadata, forward_args) is expected


def test_fallback_rejects_raw_fp8_input() -> None:
    attn = FakeAttention()
    fmha = FallbackFmha(attn)
    metadata = SimpleNamespace(is_cross=False, helix_position_offsets=None, num_contexts=0)
    q = torch.empty((1, 128), dtype=torch.float8_e4m3fn)
    assert not fmha.is_supported(q, None, None, metadata, AttentionForwardArgs(is_fused_qkv=True))


@pytest.mark.parametrize("mode", ["ordinary", "mla", "cross", "sage"])
def test_fallback_separate_qkv_support(mode: str) -> None:
    attn = FakeAttention()
    attn.is_mla_enable = mode == "mla"
    fmha = FallbackFmha(attn)
    metadata = SimpleNamespace(
        is_cross=mode == "cross", helix_position_offsets=None, num_contexts=0
    )
    forward_args = AttentionForwardArgs(
        is_fused_qkv=False, sage_attn_num_elts_per_blk_q=2 if mode == "sage" else 0
    )
    q = torch.empty((1, 128), dtype=torch.bfloat16)
    assert fmha.is_supported(q, q, q, metadata, forward_args) is (mode != "ordinary")


@pytest.mark.parametrize("initial_fused_qkv", [False, True])
@pytest.mark.parametrize(
    "input_type,sparse",
    [
        (AttentionInputType.context_only, False),
        (AttentionInputType.context_only, True),
        (AttentionInputType.generation_only, False),
    ],
)
def test_mla_forward_clears_fused_qkv_before_fmha_selection(
    monkeypatch: pytest.MonkeyPatch,
    input_type: AttentionInputType,
    sparse: bool,
    initial_fused_qkv: bool,
) -> None:
    sparse_params = SparseRuntimeParams(
        sparse_attn_indices=torch.zeros(1, dtype=torch.int32) if sparse else None
    )
    monkeypatch.setattr(
        "tensorrt_llm._torch.attention.backends.trtllm.prepare_sparse_runtime_params",
        lambda *args: sparse_params,
    )
    attn = Mock(spec=TrtllmAttention)
    attn.sparse_params = None
    attn.is_mla_enable = True
    attn.num_heads = 2
    attn.kv_lora_rank = 8
    attn.qk_nope_head_dim = 4
    attn.qk_rope_head_dim = 2
    attn.print_skip_softmax_stat = False
    attn.kv_scale_orig_quant = None
    attn.kv_scale_quant_orig = None
    attn.get_local_layer_idx.return_value = 0
    attn._ensure_rope_table_size = Mock()
    fmha = Mock()
    attn._fmha_manager = Mock()
    attn._fmha_manager.select.return_value = fmha

    metadata = Mock(spec=TrtllmAttentionMetadata)
    metadata.token_major_gen_view.return_value = None
    metadata.presented_token_major.return_value = nullcontext()
    metadata.is_cross = False
    metadata.enable_flash_mla = False
    metadata.spec_bl_tree_first_sparse_mask_offset_kv = None
    metadata.spec_decoding_bl_tree_mask = None
    metadata.max_context_q_len_override = None
    metadata.kv_cache_manager = None
    lengths = torch.ones(1, dtype=torch.int32)
    metadata.kv_lens_cuda_runtime = lengths
    metadata.kv_lens_runtime = lengths
    metadata.prompt_lens_cuda_runtime = lengths
    metadata.prompt_lens_cpu_runtime = lengths
    metadata.host_request_types_runtime = lengths
    metadata.max_seq_len = 8

    has_kv = input_type == AttentionInputType.context_only and not sparse
    q_head_dim = (attn.qk_nope_head_dim if has_kv else attn.kv_lora_rank) + attn.qk_rope_head_dim
    q = torch.empty((1, attn.num_heads * q_head_dim))
    k = torch.empty_like(q) if has_kv else None
    v = torch.empty_like(q) if has_kv else None
    forward_args = AttentionForwardArgs(
        output=torch.empty_like(q),
        attention_input_type=input_type,
        is_fused_qkv=initial_fused_qkv,
    )

    output = TrtllmAttention.forward(attn, q, k, v, metadata, forward_args)

    metadata.presented_token_major.assert_called_once_with(None)
    attn._fmha_manager.select.assert_called_once_with(attn, q, k, v, metadata, forward_args)
    fmha.forward.assert_called_once_with(q, k, v, metadata, forward_args)
    assert forward_args.sparse_runtime_params is sparse_params
    assert not forward_args.is_fused_qkv
    assert forward_args.update_kv_cache
    assert output is forward_args.output


@pytest.mark.parametrize("fmha_cls", FMHA_LIBS.values(), ids=FMHA_LIBS.keys())
@pytest.mark.parametrize("threshold", [0.0, 0.1])
@pytest.mark.parametrize("implementation_available", [False, True])
def test_availability_checks_capabilities_before_implementation(
    fmha_cls: type[Fmha],
    threshold: float,
    implementation_available: bool,
) -> None:
    attn = cast(TrtllmAttention, FakeAttention())
    attn.skip_correction_threshold = threshold
    capability_supported = threshold == 0.0 or fmha_cls is FallbackFmha

    with patch.object(fmha_cls, "_is_available", return_value=implementation_available) as hook:
        assert fmha_cls.is_available(attn) is (capability_supported and implementation_available)
        if capability_supported:
            hook.assert_called_once_with(attn)
        else:
            hook.assert_not_called()


@pytest.mark.parametrize("phase", [None, FmhaPhase.CONTEXT, FmhaPhase.GENERATION])
@pytest.mark.parametrize("supported", [False, True])
def test_support_forwards_request_and_phase(phase: FmhaPhase | None, supported: bool) -> None:
    attn = cast(TrtllmAttention, FakeAttention())
    fmha = _MinimalFmha(attn)
    q, k, v = (torch.empty((2, 4)) for _ in range(3))
    metadata = Mock(spec=TrtllmAttentionMetadata)
    forward_args = AttentionForwardArgs()

    with patch.object(fmha, "_is_supported", return_value=supported) as hook:
        assert fmha.is_supported(q, k, v, metadata, forward_args, phase=phase) is supported
        hook.assert_called_once_with(q, k, v, metadata, forward_args, phase=phase)


def test_default_hooks_accept_requests() -> None:
    attn = cast(TrtllmAttention, FakeAttention())
    fmha = _MinimalFmha(attn)

    assert _MinimalFmha.is_available(attn)
    assert fmha.is_supported(
        torch.empty((2, 4)), None, None, Mock(spec=TrtllmAttentionMetadata), AttentionForwardArgs()
    )


def test_inherited_availability_hook_uses_subclass_capabilities() -> None:
    class _SkipCorrectionFmha(_MinimalFmha):
        supports_skip_correction = True

        @classmethod
        def _is_available(cls, attn: TrtllmAttention) -> bool:
            return super()._is_available(attn)

    attn = cast(TrtllmAttention, FakeAttention())
    attn.skip_correction_threshold = 0.1

    assert not _MinimalFmha.is_available(attn)
    assert _SkipCorrectionFmha.is_available(attn)


@pytest.mark.parametrize(
    ("capability", "request_args"),
    [
        (
            "supports_block_sparse_inputs",
            {
                "sparse_runtime_params": SparseRuntimeParams(
                    block_sparse_inputs=BlockSparseForwardInputs(
                        q_block_size=64,
                        kv_block_size=64,
                        exact_block_bits=torch.zeros((1, 1, 1, 1), dtype=torch.int32),
                    )
                )
            },
        ),
        ("supports_per_tensor_fp8", {"qk_fp8_per_tensor": True}),
        ("supports_per_tensor_fp8", {"v_fp8_per_tensor": True}),
    ],
)
@pytest.mark.parametrize("requested", [False, True])
@pytest.mark.parametrize("supported", [False, True])
def test_support_checks_capabilities_before_implementation(
    capability: str, request_args: dict, requested: bool, supported: bool
) -> None:
    _CapableFmha = type("_CapableFmha", (_MinimalFmha,), {capability: True})

    attn = cast(TrtllmAttention, FakeAttention())
    q, k, v = (torch.empty((2, 4)) for _ in range(3))
    metadata = Mock(spec=TrtllmAttentionMetadata)
    forward_args = AttentionForwardArgs(**request_args) if requested else AttentionForwardArgs()

    for fmha in (_MinimalFmha(attn), _CapableFmha(attn)):
        capability_supported = not requested or getattr(fmha, capability)
        with patch.object(fmha, "_is_supported", return_value=supported) as hook:
            assert fmha.is_supported(q, k, v, metadata, forward_args) is (
                capability_supported and supported
            )
            if capability_supported:
                hook.assert_called_once_with(q, k, v, metadata, forward_args, phase=None)
            else:
                hook.assert_not_called()
