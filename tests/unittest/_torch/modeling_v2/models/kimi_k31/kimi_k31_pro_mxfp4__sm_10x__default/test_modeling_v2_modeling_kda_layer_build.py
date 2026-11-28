# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The Kimi K3.1 KDA layer builds under ``MetaInitMode`` with the checkpoint's parameter names, shapes and dtypes.

The model is built inside ``MetaInitMode``, which allows only ``empty`` / ``empty_like`` and the random-init ops on
meta tensors. The default target's KDA layer must build there, with the dtypes the checkpoint loader copies into:
BF16 projections, convolutions and output-norm weight, FP32 ``A_log`` / ``dt_bias``. CPU only, no GPU required.
"""

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.kimi_k31.kimi_k31_pro_mxfp4__sm_10x__default.modeling import (
    K31KdaLayer,
)
from tensorrt_llm._torch.models.modeling_utils import MetaInitMode

# The L0 CPU stages run `pytest -m cpu_only` and collect only files that carry
# this marker.
pytestmark = pytest.mark.cpu_only

_HIDDEN, _HEADS, _HEAD_DIM, _WIDTH = 256, 4, 128, 4
_KDA = dict(
    num_heads=_HEADS,
    head_dim=_HEAD_DIM,
    short_conv_kernel_size=_WIDTH,
    use_full_rank_gate=True,
    gate_lower_bound=-5.0,
)


def _cfg():
    return SimpleNamespace(
        hidden_size=_HIDDEN,
        rms_norm_eps=1e-5,
        linear_attn_config=dict(_KDA),
    )


def _expected() -> dict:
    d = _HEADS * _HEAD_DIM
    bf16, fp32 = torch.bfloat16, torch.float32
    return {
        "q_proj.weight": ((d, _HIDDEN), bf16),
        "k_proj.weight": ((d, _HIDDEN), bf16),
        "v_proj.weight": ((d, _HIDDEN), bf16),
        "q_conv1d.weight": ((d, 1, _WIDTH), bf16),
        "k_conv1d.weight": ((d, 1, _WIDTH), bf16),
        "v_conv1d.weight": ((d, 1, _WIDTH), bf16),
        "A_log": ((_HEADS,), fp32),
        "f_a_proj.weight": ((_HEAD_DIM, _HIDDEN), bf16),
        "f_b_proj.weight": ((d, _HEAD_DIM), bf16),
        "dt_bias": ((d,), fp32),
        "b_proj.weight": ((_HEADS, _HIDDEN), bf16),
        "g_proj.weight": ((d, _HIDDEN), bf16),
        "o_norm.weight": ((_HEAD_DIM,), bf16),
        "o_proj.weight": ((_HIDDEN, d), bf16),
    }


def test_kda_layer_builds_under_meta_init(monkeypatch) -> None:
    # The layer's fork / join events need no device until they are recorded.
    monkeypatch.setattr(torch.cuda, "Event", lambda *args, **kwargs: None)
    with MetaInitMode():
        layer = K31KdaLayer(_cfg(), layer_idx=0)

    params = {name: (tuple(p.shape), p.dtype) for name, p in layer.named_parameters()}
    assert params == _expected()
    # Everything the checkpoint fills stays on meta; the output-norm weight is initialized to ones.
    assert all(p.is_meta for name, p in layer.named_parameters() if name != "o_norm.weight")
    assert torch.equal(layer.o_norm.weight, torch.ones(_HEAD_DIM, dtype=torch.bfloat16))
