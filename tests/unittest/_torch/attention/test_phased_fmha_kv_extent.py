# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The KV-pool page extent PhasedFmha libraries pass to their kernels."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from tensorrt_llm._torch.attention.backends.fmha.phased import PhasedFmha
from tensorrt_llm._torch.attention.backends.fmha.utils import (
    MAX_KV_POOL_INDEX,
    get_kv_pool_growth_bound_bytes,
)
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")


def _v2_extent(upper_bound: int, page_stride: int) -> int:
    impl = SimpleNamespace(
        get_page_index_upper_bound=lambda layer, role: upper_bound,
        get_page_stride=lambda layer, role: page_stride,
    )
    manager = MagicMock(spec=KVCacheManagerV2)
    manager.impl = impl
    fmha = SimpleNamespace(attn=SimpleNamespace(local_layer_idx=0))
    meta = SimpleNamespace(kv_cache_manager=manager)
    return PhasedFmha._get_total_num_blocks(fmha, meta)


def test_v2_extent_does_not_follow_pool_growth() -> None:
    """KV pool rebalancing grows pools in place after decode CUDA graphs captured the extent.

    The extent must not be the current page bound, or the pages a later grow adds
    lie beyond what the captured kernels read. It is a bound no pool can outgrow,
    so it stays the same as the pool grows.
    """
    stride = 64 * 1024
    before = _v2_extent(upper_bound=1000, page_stride=stride)
    after = _v2_extent(upper_bound=4000, page_stride=stride)
    assert before == after
    assert before >= get_kv_pool_growth_bound_bytes(torch.cuda.current_device()) // stride
    assert before >= 4000


def test_v2_extent_keeps_a_larger_current_bound_and_fits_int32() -> None:
    assert (
        _v2_extent(upper_bound=MAX_KV_POOL_INDEX - 1, page_stride=1 << 30) == MAX_KV_POOL_INDEX - 1
    )
    assert _v2_extent(upper_bound=1, page_stride=1) == MAX_KV_POOL_INDEX
