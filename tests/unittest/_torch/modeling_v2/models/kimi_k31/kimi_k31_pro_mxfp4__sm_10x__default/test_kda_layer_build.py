# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The Kimi K3.1 KDA layers build under ``MetaInitMode`` with the checkpoint's parameter names, shapes and dtypes.

The model is built inside ``MetaInitMode``, which allows only ``empty`` / ``empty_like`` and the random-init ops on
meta tensors. Each target's KDA layer and the tep / dep subclasses must build there, with the dtypes the checkpoint
loader copies into: BF16 projections, convolutions and output-norm weight, FP32 ``A_log`` / ``dt_bias``. Any other KDA
configuration is refused at construction. CPU only, no GPU required.
"""

import importlib
from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.models.modeling_utils import MetaInitMode

# The L0 CPU stages run `pytest -m cpu_only` and collect only files that carry
# this marker.
pytestmark = pytest.mark.cpu_only

_FAMILY = "tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31"
#: (module, class) of each target's KDA layer and of the tep / dep subclasses.
_LAYERS = (
    ("kimi_k31_pro_mxfp4__sm_10x__default.modeling", "K31KdaLayer"),
    ("kimi_k31_pro_mxfp4__sm_103__tep.modeling", "K31KdaLayer"),
    ("kimi_k31_pro_mxfp4__sm_103__dep.modeling", "K31KdaLayer"),
    ("kimi_k31_pro_mxfp4__sm_103__tep.modeling", "K31KDALinearAttention"),
    ("kimi_k31_pro_mxfp4__sm_103__dep.modeling", "K31KDALinearAttention"),
)
_HIDDEN, _HEADS, _HEAD_DIM, _WIDTH = 256, 4, 128, 4
_KDA = dict(
    num_heads=_HEADS,
    head_dim=_HEAD_DIM,
    short_conv_kernel_size=_WIDTH,
    use_full_rank_gate=True,
    gate_lower_bound=-5.0,
)


def _layer_class(module: str, name: str):
    return getattr(importlib.import_module(f"{_FAMILY}.{module}"), name)


def _cfg(kda_fp32_gate_params=True, **change):
    return SimpleNamespace(
        hidden_size=_HIDDEN,
        rms_norm_eps=1e-5,
        linear_attn_config={**_KDA, **change},
        kda_fp32_gate_params=kda_fp32_gate_params,
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


@pytest.mark.parametrize("module, name", _LAYERS)
def test_kda_layer_builds_under_meta_init(module, name, monkeypatch) -> None:
    # The layer's fork / join events need no device until they are recorded.
    monkeypatch.setattr(torch.cuda, "Event", lambda *args, **kwargs: None)
    layer_cls = _layer_class(module, name)
    with MetaInitMode():
        layer = layer_cls(_cfg(), layer_idx=0)

    params = {name: (tuple(p.shape), p.dtype) for name, p in layer.named_parameters()}
    assert params == _expected()
    # Everything the checkpoint fills stays on meta; the output-norm weight is initialized to ones.
    assert all(p.is_meta for name, p in layer.named_parameters() if name != "o_norm.weight")
    assert torch.equal(layer.o_norm.weight, torch.ones(_HEAD_DIM, dtype=torch.bfloat16))


@pytest.mark.parametrize("module, name", _LAYERS)
@pytest.mark.parametrize(
    "cfg",
    [
        _cfg(head_dim=64),
        _cfg(short_conv_kernel_size=3),
        _cfg(use_full_rank_gate=False),
        _cfg(gate_lower_bound=None),
        _cfg(gate_lower_bound=0.5),
        _cfg(kda_fp32_gate_params=False),
    ],
    ids=[
        "head_dim",
        "conv_width",
        "low_rank_gate",
        "no_lower_bound",
        "positive_lower_bound",
        "bf16_gates",
    ],
)
def test_kda_layer_refuses_other_kda_configs(module, name, cfg, monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "Event", lambda *args, **kwargs: None)
    with pytest.raises(NotImplementedError):
        _layer_class(module, name)(cfg, layer_idx=0)
