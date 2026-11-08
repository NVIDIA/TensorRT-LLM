# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import tensorrt_llm._torch.peft.lora.manager as lora_manager
from tensorrt_llm._torch.pyexecutor._util import _get_initial_lora_data_type

pytestmark = pytest.mark.cpu_only


def test_missing_native_fp8_lora_capability_query_warns_once(caplog, monkeypatch):
    monkeypatch.setattr(
        torch.ops.trtllm,
        "lora_grouped_gemm_supports_fp8",
        None,
        raising=False,
    )
    lora_manager._warn_native_fp8_lora_capability_query_unavailable.cache_clear()

    with caplog.at_level("WARNING", logger=lora_manager.__name__):
        assert not lora_manager._native_fp8_lora_kernels_available((9, 0))
        assert not lora_manager._native_fp8_lora_kernels_available((10, 0))

    assert caplog.messages == [
        "Native FP8 LoRA capability query is unavailable; adapter weights "
        "will fall back to the model compute dtype. Check that the "
        "TensorRT-LLM libraries match the Python package and are loaded."
    ]


@pytest.mark.parametrize("device_capability", [(9, 0), (10, 0), (10, 3)])
def test_native_fp8_lora_initializes_fp8_cache(device_capability, monkeypatch):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: device_capability)
    monkeypatch.setattr(lora_manager, "_native_fp8_lora_kernels_available", lambda _: True)

    assert _get_initial_lora_data_type(torch.float8_e4m3fn) == torch.float8_e4m3fn


@pytest.mark.parametrize("device_capability", [(8, 0), (12, 0)])
def test_device_without_native_kernels_does_not_initialize_fp8_cache(
    device_capability, monkeypatch
):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: device_capability)
    monkeypatch.setattr(lora_manager, "_native_fp8_lora_kernels_available", lambda _: False)

    assert _get_initial_lora_data_type(torch.float8_e4m3fn) is None


def test_missing_native_fp8_lora_kernels_do_not_initialize_fp8_cache(monkeypatch):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (10, 0))
    monkeypatch.setattr(lora_manager, "_native_fp8_lora_kernels_available", lambda _: False)

    assert _get_initial_lora_data_type(torch.float8_e4m3fn) is None


def test_non_fp8_lora_does_not_query_device_capability(monkeypatch):
    def fail_if_called():
        raise AssertionError("device capability should not be queried for non-FP8 LoRA")

    monkeypatch.setattr(torch.cuda, "get_device_capability", fail_if_called)

    assert _get_initial_lora_data_type(torch.bfloat16) is None
