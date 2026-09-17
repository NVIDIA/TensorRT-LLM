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

from typing import cast
from unittest.mock import Mock, patch

import pytest
import torch
from fmha_test_utils import FakeAttention

from tensorrt_llm._torch.attention.backends.fmha.fallback import FallbackFmha
from tensorrt_llm._torch.attention.backends.fmha.interface import Fmha, FmhaPhase
from tensorrt_llm._torch.attention.backends.fmha.registry import FMHA_LIBS
from tensorrt_llm._torch.attention.backends.interface import AttentionForwardArgs
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


@pytest.mark.parametrize("has_block_sparse_inputs", [False, True])
@pytest.mark.parametrize("supported", [False, True])
def test_support_checks_block_sparse_capability_before_implementation(
    has_block_sparse_inputs: bool, supported: bool
) -> None:
    class _BlockSparseFmha(_MinimalFmha):
        supports_block_sparse_inputs = True

    attn = cast(TrtllmAttention, FakeAttention())
    q, k, v = (torch.empty((2, 4)) for _ in range(3))
    metadata = Mock(spec=TrtllmAttentionMetadata)
    forward_args = AttentionForwardArgs()
    if has_block_sparse_inputs:
        forward_args.sparse_runtime_params = SparseRuntimeParams(
            block_sparse_inputs=BlockSparseForwardInputs(
                q_block_size=64,
                kv_block_size=64,
                exact_block_bits=torch.zeros((1, 1, 1, 1), dtype=torch.int32),
            )
        )

    for fmha in (_MinimalFmha(attn), _BlockSparseFmha(attn)):
        capability_supported = not has_block_sparse_inputs or fmha.supports_block_sparse_inputs
        with patch.object(fmha, "_is_supported", return_value=supported) as hook:
            assert fmha.is_supported(q, k, v, metadata, forward_args) is (
                capability_supported and supported
            )
            if capability_supported:
                hook.assert_called_once_with(q, k, v, metadata, forward_args, phase=None)
            else:
                hook.assert_not_called()
