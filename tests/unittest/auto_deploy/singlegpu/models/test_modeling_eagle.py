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

"""Unit tests for AutoDeploy Eagle modeling helpers."""

from unittest.mock import patch

import pytest
import torch

from tensorrt_llm._torch.auto_deploy.models.custom.modeling_eagle import (
    EagleRMSNorm,
    EagleWrapper,
    EagleWrapperConfig,
)
from tensorrt_llm._torch.auto_deploy.shim.interface import SpeculativeDecodingModelArgs
from tensorrt_llm.llmapi import SAEnhancerConfig


def test_eagle_rmsnorm_keeps_fp32_weights():
    norm = EagleRMSNorm(hidden_size=16)

    assert norm.weight.dtype == torch.float32


def test_eagle_wrapper_instantiates_sa_enhancer():
    wrapper = EagleWrapper(
        EagleWrapperConfig(
            max_draft_len=3,
            load_embedding_from_target=True,
            load_lm_head_from_target=True,
            sa_config=SAEnhancerConfig(threshold=2),
        ),
        target_model=torch.nn.Module(),
        draft_model=torch.nn.Module(),
    )

    assert wrapper.sa_enhancer is not None
    assert wrapper.sa_enhancer.threshold == 2


def test_eagle_wrapper_skips_sa_enhancer_without_sa_config():
    wrapper = EagleWrapper(
        EagleWrapperConfig(
            max_draft_len=3,
            load_embedding_from_target=True,
            load_lm_head_from_target=True,
        ),
        target_model=torch.nn.Module(),
        draft_model=torch.nn.Module(),
    )

    assert wrapper.sa_enhancer is None


def test_eagle_wrapper_sa_override_updates_next_new_tokens():
    class FakeSAEnhancer:
        def __init__(self):
            self.seen_draft_tokens = None

        def maybe_override_all_draft_tokens(self, draft_tokens):
            self.seen_draft_tokens = draft_tokens.clone()
            return draft_tokens + 100

    wrapper = EagleWrapper(
        EagleWrapperConfig(
            max_draft_len=2,
            load_embedding_from_target=True,
            load_lm_head_from_target=True,
        ),
        target_model=torch.nn.Module(),
        draft_model=torch.nn.Module(),
    )
    fake_sa_enhancer = FakeSAEnhancer()
    wrapper.sa_enhancer = fake_sa_enhancer
    next_new_tokens = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.int32)

    wrapper._maybe_apply_sa_draft_override(
        next_new_tokens,
        num_prefill=1,
        sa_manager=object(),
    )

    torch.testing.assert_close(
        next_new_tokens,
        torch.tensor([[1, 2, 3], [4, 105, 106]], dtype=torch.int32),
    )
    torch.testing.assert_close(
        fake_sa_enhancer.seen_draft_tokens,
        torch.tensor([[5, 6]], dtype=torch.int32),
    )


def test_eagle_wrapper_sa_override_requires_manager():
    wrapper = EagleWrapper(
        EagleWrapperConfig(
            max_draft_len=2,
            load_embedding_from_target=True,
            load_lm_head_from_target=True,
            sa_config=SAEnhancerConfig(threshold=2),
        ),
        target_model=torch.nn.Module(),
        draft_model=torch.nn.Module(),
    )
    next_new_tokens = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.int32)

    with pytest.raises(RuntimeError):
        wrapper._maybe_apply_sa_draft_override(next_new_tokens, num_prefill=1, sa_manager=None)

    with pytest.raises(RuntimeError):
        wrapper._maybe_apply_sa_draft_override(next_new_tokens, num_prefill=2, sa_manager=None)


def test_eagle_wrapper_cached_forward_requires_manager_when_sa_enabled():
    wrapper = EagleWrapper(
        EagleWrapperConfig(
            max_draft_len=2,
            load_embedding_from_target=True,
            load_lm_head_from_target=True,
            sa_config=SAEnhancerConfig(threshold=2),
        ),
        target_model=torch.nn.Module(),
        draft_model=torch.nn.Module(),
    )

    with pytest.raises(RuntimeError):
        wrapper._forward_with_kv_cache(csi=object(), sa_manager=None)


def test_eagle_wrapper_forward_unpacks_spec_dec_args():
    """forward() routes spec-dec inputs through SpeculativeDecodingModelArgs.

    The struct is the executor->model contract: forward must unpack it and call the cached
    path with the struct's cache_seq_interface and sa_manager. With no struct (export time),
    it must fall back to the prefill-only path.
    """
    wrapper = EagleWrapper(
        EagleWrapperConfig(
            max_draft_len=2,
            load_embedding_from_target=True,
            load_lm_head_from_target=True,
        ),
        target_model=torch.nn.Module(),
        draft_model=torch.nn.Module(),
    )

    csi = object()
    sa_manager = object()
    with (
        patch.object(wrapper, "_forward_with_kv_cache") as mock_cached,
        patch.object(wrapper, "_forward_prefill_only") as mock_prefill,
    ):
        wrapper.forward(
            spec_dec_args=SpeculativeDecodingModelArgs(
                cache_seq_interface=csi, sa_manager=sa_manager
            )
        )
        mock_cached.assert_called_once_with(
            csi,
            sa_manager=sa_manager,
        )
        mock_prefill.assert_not_called()

    with (
        patch.object(wrapper, "_forward_with_kv_cache") as mock_cached,
        patch.object(wrapper, "_forward_prefill_only") as mock_prefill,
    ):
        wrapper.forward(input_ids="tokens", position_ids="pos")
        mock_prefill.assert_called_once_with(input_ids="tokens", position_ids="pos")
        mock_cached.assert_not_called()
