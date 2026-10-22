# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Separate-draft-KV policy follows the DFlash backend's storage ownership.

TRTLLM borrows the managed paged pool. VANILLA and FA4 own private context
buffers and must not allocate a redundant manager or be gated on its capacity.
"""

import pytest

from tensorrt_llm._torch.speculative.interface import should_use_separate_draft_kv_cache
from tensorrt_llm.llmapi import DFlashDecodingConfig, EagleDecodingConfig

pytestmark = pytest.mark.cpu_only


@pytest.mark.parametrize(
    ("backend", "uses_managed_pool"),
    [("VANILLA", False), ("FA4", False), ("TRTLLM", True)],
)
def test_dflash_draft_cache_matches_backend_storage(backend: str, uses_managed_pool: bool) -> None:
    config = DFlashDecodingConfig(max_draft_len=7, attention_backend=backend)
    assert config.spec_dec_mode.use_one_engine()
    assert not config._use_shared_kv_cache
    assert config._allow_separate_draft_kv_cache
    assert should_use_separate_draft_kv_cache(config) is uses_managed_pool


def test_default_dflash_has_no_redundant_mirror_or_target_draft_layers() -> None:
    from tensorrt_llm._torch.speculative.utils import get_num_spec_layers

    config = DFlashDecodingConfig(max_draft_len=7)
    assert config.attention_backend == "VANILLA"
    assert should_use_separate_draft_kv_cache(config) is False
    assert get_num_spec_layers(config) == 0


@pytest.mark.parametrize("backend", ["VANILLA", "FA4", "TRTLLM"])
def test_dflash_explicit_cache_opt_out_is_preserved(backend: str) -> None:
    config = DFlashDecodingConfig(max_draft_len=7, attention_backend=backend)
    config._allow_separate_draft_kv_cache = False
    assert should_use_separate_draft_kv_cache(config) is False


def test_eagle3_one_model_still_gets_a_separate_draft_kv_cache():
    """The DFlash exemption must not leak into other one-engine drafters."""
    config = EagleDecodingConfig(max_draft_len=3, speculative_model="/tmp/eagle3-draft")
    assert config.spec_dec_mode.is_eagle3_one_model()
    assert should_use_separate_draft_kv_cache(config) is True
