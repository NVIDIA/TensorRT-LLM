# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for FLUX request orchestration."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from tensorrt_llm._torch.modules.linear import (
    FP8BlockScalesLinearMethod,
    FP8QDQLinearMethod,
    FP8RowwiseLinearMethod,
    NVFP4LinearMethod,
    W4A16NVFP4LinearMethod,
    W8A16FP8LinearMethod,
)
from tensorrt_llm._torch.visual_gen.batching import BatchInfo
from tensorrt_llm._torch.visual_gen.models.flux import Flux2Pipeline, FluxPipeline
from tensorrt_llm.visual_gen.args import AttentionConfig

pytestmark = pytest.mark.cpu_only


@pytest.mark.parametrize("pipeline_cls", [FluxPipeline, Flux2Pipeline])
@pytest.mark.parametrize(
    "method_cls,force_dynamic,input_scale,expected",
    [
        (NVFP4LinearMethod, True, torch.tensor(1.0), False),
        (NVFP4LinearMethod, False, None, False),
        (NVFP4LinearMethod, False, torch.tensor(1.0), False),
        (FP8QDQLinearMethod, True, torch.tensor(1.0), False),
        (FP8QDQLinearMethod, False, None, False),
        (FP8QDQLinearMethod, False, torch.tensor(1.0), True),
        (W4A16NVFP4LinearMethod, True, None, True),
        (W8A16FP8LinearMethod, True, None, True),
        (FP8RowwiseLinearMethod, True, None, True),
        (FP8BlockScalesLinearMethod, True, None, True),
    ],
    ids=[
        "nvfp4-forced-dynamic",
        "nvfp4-missing-scale",
        "nvfp4-static-scale",
        "fp8-forced-dynamic",
        "fp8-missing-scale",
        "fp8-static-scale",
        "nvfp4-weight-only",
        "fp8-weight-only",
        "fp8-rowwise",
        "fp8-block-scales",
    ],
)
def test_batching_quantization_eligibility(
    pipeline_cls, method_cls, force_dynamic, input_scale, expected
):
    layer = torch.nn.Module()
    layer.quant_method = method_cls()
    layer.force_dynamic_quantization = force_dynamic
    layer.input_scale = input_scale
    pipeline = pipeline_cls.__new__(pipeline_cls)
    torch.nn.Module.__init__(pipeline)
    # A nested quantized layer among unquantized layers also gates the pipeline.
    pipeline.transformer = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Sequential(layer))
    pipeline.transformer.model_config = SimpleNamespace(attention=AttentionConfig())
    assert pipeline.supports_dynamic_batching is expected


@pytest.mark.parametrize("pipeline_cls", [FluxPipeline, Flux2Pipeline])
@pytest.mark.parametrize(
    "backend,qk_dtype,v_dtype,expected",
    [
        ("CUDNN", None, None, True),
        ("CUDNN", "fp8", "fp8", False),
        ("CUDNN", "mxfp8", "mxfp8", True),
        ("FLASHINFER", None, None, True),
        ("FLASHINFER", "mxfp8", "fp8", False),
        ("FLASHINFER", "nvfp4", "fp8", False),
        ("FLASHINFER", "nvfp4", "nvfp4", False),
        ("TRTLLM", None, None, True),
    ],
)
def test_batching_attention_quantization_eligibility(
    pipeline_cls: type[FluxPipeline] | type[Flux2Pipeline],
    backend: str,
    qk_dtype: str | None,
    v_dtype: str | None,
    expected: bool,
) -> None:
    pipeline = pipeline_cls.__new__(pipeline_cls)
    torch.nn.Module.__init__(pipeline)
    pipeline.transformer = torch.nn.Linear(2, 2)
    pipeline.transformer.model_config = SimpleNamespace(
        attention=AttentionConfig(
            backend=backend,
            quant_attention_config=(
                {"qk_dtype": qk_dtype, "v_dtype": v_dtype} if qk_dtype is not None else None
            ),
        )
    )
    assert pipeline.supports_dynamic_batching is expected


@pytest.mark.parametrize("pipeline_cls", [FluxPipeline, Flux2Pipeline])
@pytest.mark.parametrize(
    "qk_dtype,block_sizes,expected",
    [
        ("bf16", (0, 0, 0), False),
        ("fp8", (0, 0, 0), False),
        ("int8", (1, 4, 1), True),
        ("fp8", (1, 4, 1), True),
    ],
    ids=["per-tensor-v", "per-tensor-qkv", "sage-int8", "sage-fp8"],
)
def test_batching_trtllm_attention_recipe_eligibility(
    pipeline_cls: type[FluxPipeline] | type[Flux2Pipeline],
    qk_dtype: str,
    block_sizes: tuple[int, int, int],
    expected: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Recipe validation is hardware-specific; these are valid SM100 recipes.
    monkeypatch.setattr("tensorrt_llm.visual_gen.args.get_sm_version", lambda: 100)
    pipeline = pipeline_cls.__new__(pipeline_cls)
    torch.nn.Module.__init__(pipeline)
    pipeline.transformer = torch.nn.Linear(2, 2)
    pipeline.transformer.model_config = SimpleNamespace(
        attention=AttentionConfig(
            backend="TRTLLM",
            quant_attention_config={
                "qk_dtype": qk_dtype,
                "v_dtype": "fp8",
                "q_block_size": block_sizes[0],
                "k_block_size": block_sizes[1],
                "v_block_size": block_sizes[2],
            },
        )
    )
    assert pipeline.supports_dynamic_batching is expected


@pytest.mark.parametrize("pipeline_cls", [FluxPipeline, Flux2Pipeline])
@pytest.mark.parametrize("batch_info", [None, BatchInfo((1, 2), (2, 2), (42, 17))])
def test_infer_forwards_num_images_per_prompt(
    pipeline_cls: type[FluxPipeline] | type[Flux2Pipeline],
    batch_info: BatchInfo | None,
) -> None:
    pipeline = pipeline_cls.__new__(pipeline_cls)
    pipeline.forward = Mock(return_value="image")
    request = SimpleNamespace(
        prompt="a cat",
        prepared_inputs={},
        batch_info=batch_info,
        params=SimpleNamespace(
            height=256,
            width=256,
            num_inference_steps=4,
            guidance_scale=3.5,
            seed=42,
            max_sequence_length=512,
            num_images_per_prompt=2,
            image_reference=[SimpleNamespace(content=b"reference")],
        ),
    )

    result = pipeline.infer(request)

    assert result == "image"
    pipeline.forward.assert_called_once()
    assert pipeline.forward.call_args.kwargs["num_images_per_prompt"] == 2
    assert pipeline.forward.call_args.kwargs["_batch_info"] is batch_info
    if pipeline_cls is Flux2Pipeline:
        assert pipeline.forward.call_args.kwargs["image"] == [b"reference"]
    else:
        assert "image" not in pipeline.forward.call_args.kwargs


@pytest.mark.parametrize("pipeline_cls", [FluxPipeline, Flux2Pipeline])
def test_prepare_latents_preserves_each_requests_noise(pipeline_cls) -> None:
    pipe = SimpleNamespace(
        vae_scale_factor=8,
        device=torch.device("cpu"),
        dtype=torch.float32,
        vae=SimpleNamespace(config=SimpleNamespace(latent_channels=16)),
        transformer=SimpleNamespace(config=SimpleNamespace(in_channels=128)),
        _prepare_latent_ids=lambda height, width: torch.zeros(4, 4),
    )
    pipe._pack_latents = pipeline_cls._pack_latents.__get__(pipe)
    prepare = pipeline_cls._prepare_latents.__get__(pipe)
    first, _ = prepare(2, 32, 32, torch.Generator().manual_seed(42))
    second, _ = prepare(1, 32, 32, torch.Generator().manual_seed(17))
    merged, _ = prepare(
        3,
        32,
        32,
        torch.Generator().manual_seed(42),
        batch_info=BatchInfo((1, 2), (2, 1), (42, 17)),
    )
    assert torch.equal(merged, torch.cat([first, second]))
