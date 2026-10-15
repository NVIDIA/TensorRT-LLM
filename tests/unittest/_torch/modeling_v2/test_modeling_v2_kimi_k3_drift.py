# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""K3 targets share the model lifecycle while preserving each checkpoint's layout."""

import importlib
from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl import modeling
from tensorrt_llm.quantization.mode import QuantAlgo

pytestmark = pytest.mark.cpu_only


@pytest.mark.parametrize(
    "sm,fmt,moe_tp,moe_ep",
    [
        (100, "mxfp4", 4, 4),
        (100, "mxfp4", 16, 1),
        (103, "mxfp4", 4, 4),
        (103, "mxfp4", 16, 1),
        (103, "nvfp4", 4, 4),
        (107, "nvfp4", 4, 4),
    ],
)
def test_targets_share_the_model_lifecycle(sm: int, fmt: str, moe_tp: int, moe_ep: int) -> None:
    module = importlib.import_module(
        "tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl."
        f"kimi_k3_{fmt}__sm_{sm}__tp16_moetp{moe_tp}ep{moe_ep}.modeling"
    )
    target = getattr(module, f"ModelingV2KimiK3{fmt.capitalize()}Sm{sm}Tp16Moetp{moe_tp}ep{moe_ep}")
    assert target.forward is modeling.ModelingV2KimiK3.forward
    assert target.load_weights is modeling.ModelingV2KimiK3.load_weights
    assert target.cache_derived_state is modeling.ModelingV2KimiK3.cache_derived_state
    assert target.post_load_weights is modeling.ModelingV2KimiK3.post_load_weights
    assert target.target_spec.sm == (sm // 10, sm % 10)
    assert (target.target_spec.moe_tp, target.target_spec.moe_ep) == (moe_tp, moe_ep)


@pytest.mark.parametrize("algo", [QuantAlgo.W4A8_MXFP4_MXFP8, QuantAlgo.NVFP4])
def test_streaming_loader_preserves_checkpoint_tensors(algo: QuantAlgo) -> None:
    """Each loader passes the exact packed tensors and scales to its quant method."""
    calls = []

    def record(backend: object, **kwargs: object) -> None:
        calls.append(kwargs)

    backend = SimpleNamespace(
        quant_method=SimpleNamespace(
            load_packed_mxfp4_expert=record,
            load_streaming_nvfp4_expert=record,
        )
    )
    spec = modeling._k3_expert_ckpt_spec(algo)
    tensors = {
        f"experts.7.{weight}.{kind}": object()
        for weight in ("w1", "w2", "w3")
        for kind in spec.kinds
    }
    spec.loader(backend, "experts", 7, 2, tensors.__getitem__)
    assert len(calls) == 1
    loaded = calls[0]
    assert loaded.pop("global_expert_id") == 7
    assert loaded.pop("local_slot_id") == 2
    for weight in ("w1", "w2", "w3"):
        for kind in spec.kinds:
            argument = "weight" if kind == "weight_packed" else kind
            assert loaded.pop(f"{weight}_{argument}") is tensors[f"experts.7.{weight}.{kind}"]
    assert not loaded
    assert spec.needs_layer_finalize == (algo == QuantAlgo.NVFP4)


@pytest.mark.parametrize("sm", [(10, 0), (10, 3), (10, 7)])
def test_mxfp4_fused_kernel_gates_hardware_and_scale_layout(monkeypatch, sm) -> None:
    from tensorrt_llm._torch.cute_dsl_kernels.k3_fused_moe.op import is_supported

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: sm)
    weights = torch.empty(224, 1536, 1792, dtype=torch.uint8, device="meta")
    fc1_scales = torch.empty(224, 1536, 112, dtype=torch.uint8, device="meta")
    down = torch.empty(224, 3584, 384, dtype=torch.uint8, device="meta")
    fc2_scales = torch.empty(224, 3584, 24, dtype=torch.uint8, device="meta")
    supported, _ = is_supported(weights, fc1_scales, down, fc2_scales, 224)
    assert supported == (sm in ((10, 0), (10, 3)))
    # Group-16 scales cannot be consumed by this group-32 MXFP4 kernel.
    nvfp4_scales = torch.empty(224, 1536, 224, dtype=torch.uint8, device="meta")
    supported, _ = is_supported(weights, nvfp4_scales, down, fc2_scales, 224)
    assert not supported
