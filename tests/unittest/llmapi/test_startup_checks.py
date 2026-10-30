# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Preflight checks that run before weights are loaded (no GPU; devices are faked)."""

import sys
from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._errors import ResourceError
from tensorrt_llm._startup_errors import classify_startup_error
from tensorrt_llm.llmapi import startup_checks
from tensorrt_llm.llmapi.llm_args import Nvfp4GemmConfig, _ParallelConfig
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization.mode import QuantAlgo

pytestmark = pytest.mark.cpu_only


@pytest.fixture
def fake_gpus(monkeypatch):
    """Fake the visible device count, NVML and GPU 0's compute capability."""

    def set_gpus(count: int = 1, sm: int = 90):
        monkeypatch.setattr(startup_checks, "get_device_count", lambda: count)
        monkeypatch.setattr(startup_checks, "get_sm_version", lambda: sm)
        monkeypatch.setattr(torch.cuda, "get_device_name", lambda device=None: "Fake GPU")
        monkeypatch.setattr(
            startup_checks,
            "_nvml_facts",
            lambda: "NVML sees 4 GPU(s), driver 570.00 supports CUDA up to 12.8",
        )

    return set_gpus


def test_no_visible_gpu_lists_facts_and_causes(fake_gpus, monkeypatch):
    fake_gpus(count=0)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")

    with pytest.raises(RuntimeError) as excinfo:
        startup_checks.check_visible_gpus(_ParallelConfig(), all_ranks_local=True)

    message = str(excinfo.value)
    assert message.startswith("No GPU is visible to PyTorch")
    assert "CUDA_VISIBLE_DEVICES=," in message
    assert "NVML sees 4 GPU(s), driver 570.00 supports CUDA up to 12.8" in message
    assert "(3) the NVIDIA driver is too old" in message
    info = classify_startup_error(excinfo.value)
    assert (info.category, info.reason) == ("resource", "no_gpu")
    assert info.hint.startswith("Possible causes:")


def test_no_visible_gpu_error_is_a_runtime_error(fake_gpus):
    fake_gpus(count=0)
    error = startup_checks.no_visible_gpu_error()
    assert isinstance(error, ResourceError)
    assert isinstance(error, RuntimeError)


def test_nvml_facts_report_unavailable(monkeypatch):
    monkeypatch.setitem(sys.modules, "pynvml", None)  # import fails
    assert startup_checks._nvml_facts() == "NVML unavailable (ModuleNotFoundError)"


def test_too_few_gpus_for_local_ranks(fake_gpus):
    # gpus_per_node defaults to the visible count, which alone would read as a
    # 2-node world; ranks spawned by the LLM's own pool all land on this node.
    fake_gpus(count=1)

    with pytest.raises(RuntimeError, match=r"Only 1 GPUs are available, but 2 are required") as e:
        startup_checks.check_visible_gpus(
            _ParallelConfig(tp_size=2, gpus_per_node=1), all_ranks_local=True
        )

    assert "2 x 1 x 1 = 2 ranks on this node" in str(e.value)
    assert classify_startup_error(e.value).category == "resource"


def test_too_few_gpus_for_this_nodes_share(fake_gpus):
    fake_gpus(count=2)

    with pytest.raises(RuntimeError, match=r"Only 2 GPUs are available, but 4 are required") as e:
        startup_checks.check_visible_gpus(
            _ParallelConfig(tp_size=8, pp_size=2, gpus_per_node=4), all_ranks_local=False
        )

    assert "8 x 2 x 1 = 16 ranks over 4 node(s)" in str(e.value)


@pytest.mark.parametrize(
    "count, tp_size, gpus_per_node, all_ranks_local",
    [
        (1, 1, 8, True),
        (4, 4, 8, True),
        (8, 4, 8, True),
        # Launched across nodes: this node only needs its share.
        (1, 2, 1, False),
        (4, 8, 4, False),
    ],
)
def test_enough_gpus_pass(fake_gpus, count, tp_size, gpus_per_node, all_ranks_local):
    fake_gpus(count=count)
    startup_checks.check_visible_gpus(
        _ParallelConfig(tp_size=tp_size, gpus_per_node=gpus_per_node), all_ranks_local
    )


def _quant_args(quant_algo=None, kv_cache_quant_algo=None, nvfp4_backends=None):
    nvfp4 = (
        Nvfp4GemmConfig()
        if nvfp4_backends is None
        else Nvfp4GemmConfig(allowed_backends=nvfp4_backends)
    )
    return SimpleNamespace(
        quant_config=QuantConfig(quant_algo=quant_algo, kv_cache_quant_algo=kv_cache_quant_algo),
        nvfp4_gemm_config=nvfp4,
    )


@pytest.mark.parametrize(
    "sm, quant_algo, kv_cache_quant_algo, backends",
    [
        (80, QuantAlgo.FP8, None, None),
        (89, QuantAlgo.FP8_BLOCK_SCALES, None, None),
        (89, QuantAlgo.FP8_PER_CHANNEL_PER_TOKEN, None, None),
        (90, QuantAlgo.NVFP4, None, None),
        (80, QuantAlgo.NVFP4, None, ["marlin"]),
        (90, QuantAlgo.W4A8_MXFP4_MXFP8, None, None),
    ],
)
def test_quantization_below_minimum_is_unsupported(
    fake_gpus, sm, quant_algo, kv_cache_quant_algo, backends
):
    fake_gpus(sm=sm)

    with pytest.raises(ValueError, match=r"requires SM\d+ or newer") as e:
        startup_checks.check_quantization_compute_capability(
            _quant_args(quant_algo, kv_cache_quant_algo, backends)
        )

    assert f"(Fake GPU) is SM{sm}." in str(e.value)
    info = classify_startup_error(e.value)
    assert info.category == "unsupported"
    assert "quantization.html" in info.hint


@pytest.mark.parametrize(
    "sm, quant_algo, kv_cache_quant_algo, backends",
    [
        (89, QuantAlgo.FP8, None, None),
        (90, QuantAlgo.FP8_BLOCK_SCALES, None, None),
        (100, QuantAlgo.NVFP4, None, None),
        (90, QuantAlgo.NVFP4, None, ["cutlass", "marlin"]),
        (120, QuantAlgo.NVFP4, QuantAlgo.FP8, None),
        # KV cache formats are not checked: kernels dequantize them before the MMA.
        (80, None, QuantAlgo.FP8, None),
        (90, None, QuantAlgo.NVFP4, None),
        # Formats outside the documented matrix are not checked.
        (80, QuantAlgo.W4A16_AWQ, None, None),
        (80, QuantAlgo.MIXED_PRECISION, None, None),
        (80, None, None, None),
    ],
)
def test_quantization_at_or_above_minimum_passes(
    fake_gpus, sm, quant_algo, kv_cache_quant_algo, backends
):
    fake_gpus(sm=sm)
    startup_checks.check_quantization_compute_capability(
        _quant_args(quant_algo, kv_cache_quant_algo, backends)
    )


def test_nvfp4_on_hopper_suggests_marlin(fake_gpus):
    fake_gpus(sm=90)
    with pytest.raises(ValueError, match="add 'marlin' to nvfp4_gemm_config.allowed_backends"):
        startup_checks.check_quantization_compute_capability(_quant_args(QuantAlgo.NVFP4))


def test_quantization_check_skipped_without_gpu(fake_gpus):
    fake_gpus(sm=-1)
    startup_checks.check_quantization_compute_capability(_quant_args(QuantAlgo.NVFP4))


def _arch_args(checkpoint_format=None, mm_encoder_only=False, encode_only=False):
    return SimpleNamespace(
        checkpoint_format=checkpoint_format,
        mm_encoder_only=mm_encoder_only,
        encode_only=encode_only,
    )


def test_unregistered_architecture_is_unsupported():
    config = SimpleNamespace(architectures=["FooBarForCausalLM"])

    with pytest.raises(ValueError, match="Unknown model architecture: FooBarForCausalLM") as e:
        startup_checks.check_architecture_registered(_arch_args(), config)

    info = classify_startup_error(e.value)
    assert info.category == "unsupported"
    assert "supported-models.html" in info.hint


@pytest.mark.parametrize(
    "args, architectures",
    [
        (_arch_args(), ["LlamaForCausalLM"]),
        (_arch_args(), None),
        (_arch_args(), []),
        (_arch_args(checkpoint_format="mistral"), ["FooBarForCausalLM"]),
        (_arch_args(mm_encoder_only=True), ["FooBarForCausalLM"]),
        (_arch_args(encode_only=True), ["FooBarForCausalLM"]),
    ],
)
def test_architecture_check_passes_or_is_skipped(args, architectures):
    startup_checks.check_architecture_registered(args, SimpleNamespace(architectures=architectures))


def test_architecture_check_skipped_without_config():
    startup_checks.check_architecture_registered(_arch_args(), None)
