# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Host-side cache retention validation; no model weights or GPU allocations."""

from types import SimpleNamespace

import pytest

from tensorrt_llm._torch._experimental.modeling_v2.models.beam.full_model_nvfp4__sm_103__tp4.model import (
    BeamFullModelNvfp4Sm103Tp4 as Component,
)
from tensorrt_llm.bindings import DataType

pytestmark = pytest.mark.cpu_only


@pytest.mark.parametrize("dtype", [DataType.BF16, DataType.FP8])
@pytest.mark.parametrize(
    "length,layer,window,accepted",
    [
        (8192, 0, None, True),
        (8192, 0, 512, True),
        (8192, 0, 1024, True),
        (8192, 0, 8192, True),
        (128, 0, 128, True),
        (8192, 0, 511, False),
        (128, 0, 127, False),
        (8192, 51, 8192, True),
        (8192, 51, 512, False),
    ],
)
def test_cache_retention(length, layer, window, accepted, dtype):
    component = SimpleNamespace(
        head_dim=128,
        local_kv_heads=2,
        window=512,
        kv_cache_dtype=dtype,
        schedule=SimpleNamespace(is_sliding=lambda index: index < 51),
    )
    manager = SimpleNamespace(
        layer_offsets=dict(enumerate(range(52))),
        dtype=dtype,
        head_dim=128,
        num_kv_heads_per_layer=[2] * 52,
        max_attention_window_vec=[None] * 52,
        max_seq_len=length,
    )
    manager.max_attention_window_vec[layer] = window
    if accepted:
        Component.check_cache_manager(component, manager)
        assert Component.attention_windows(component, length) == [512] * 51 + [length]
    else:
        with pytest.raises(ValueError, match="cache keeps a window"):
            Component.check_cache_manager(component, manager)
