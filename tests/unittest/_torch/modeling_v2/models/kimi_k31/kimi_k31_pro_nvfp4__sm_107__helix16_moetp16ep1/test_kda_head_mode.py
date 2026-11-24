# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check the fused KDA head's SM-count gate and launch order on a real SM10x layer."""

from types import SimpleNamespace
from unittest import mock

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
    reason="the fused KDA head needs compute capability 10.x",
)

_FAMILY = "tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31"
_PKG = f"{_FAMILY}.kimi_k31_pro_nvfp4__sm_107__helix16_moetp16ep1"


def _layer(local_heads: int = 6):
    from importlib import import_module

    weights = import_module(f"{_PKG}.weights")
    pitch = weights.kda_pitch(local_heads)
    packed = weights.DequantizedMxfp8(torch.zeros(pitch, 7168, dtype=torch.bfloat16, device="cuda"))
    return SimpleNamespace(
        inputs=weights.Bf16KdaInput(packed=packed, head_rank=0, local_heads=local_heads)
    )


@pytest.mark.parametrize("sms,expected", [(152, False), (153, True), (212, True)])
def test_head_gates_on_the_device_sm_count(sms, expected) -> None:
    from importlib import import_module

    modeling = import_module(f"{_PKG}.modeling")
    assert modeling.HEAD_MIN_SMS == 152
    layer = _layer()
    real = torch.cuda.get_device_properties(0)
    props = SimpleNamespace(major=real.major, minor=real.minor, multi_processor_count=sms)
    with mock.patch.object(torch.cuda, "get_device_properties", return_value=props):
        assert modeling.FlatKda.head_supported(layer, 8) is expected
        assert modeling.FlatKda.head_supported(layer, 1) is False


@pytest.mark.parametrize("sms,expected", [(152, "event"), (153, "sibling"), (212, "sibling")])
def test_head_order_resolves_on_the_device_sm_count(sms, expected) -> None:
    from importlib import import_module

    modeling = import_module(f"{_PKG}.modeling")
    props = SimpleNamespace(multi_processor_count=sms)
    with mock.patch.object(torch.cuda, "get_device_properties", return_value=props):
        assert modeling.FlatKda.head_order(torch.device("cuda", 0)) == expected
