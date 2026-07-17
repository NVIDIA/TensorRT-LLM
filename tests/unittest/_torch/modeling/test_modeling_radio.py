# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
from unittest import mock

import pytest
import torch
from transformers import PretrainedConfig

from tensorrt_llm._torch import model_config as model_config_lib
from tensorrt_llm._torch.models import modeling_radio
from tensorrt_llm._torch.models.modeling_multimodal_encoder import MultimodalEncoderMixin
from tensorrt_llm._torch.models.modeling_radio import RADIOVisionModel
from tensorrt_llm.llmapi.llm_args import MultimodalEncoderCudaGraphConfig
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization.mode import QuantAlgo

_TINY_VIT = modeling_radio.VITTIMMConfig(
    embed_dim=64,
    depth=2,
    num_attention_heads=2,
    intermediate_size=128,
    img_size=32,
)

# Mirror the engine's encoder runtime sizes (``get_encoder_runtime_sizes`` ->
# ``encoder_max_batch_size`` / ``encoder_max_num_tokens``, defaulting to
# ``max_batch_size`` / ``max_num_tokens``). The encoder ``AttentionMetadata`` is
# sized once at load to this max budget; each forward re-preps it with the real
# per-image seq lens. Two distinct axes: requests = image/sequence count budget,
# tokens = total patch budget.
_ENCODER_TEST_MAX_NUM_REQUESTS = 2048
_ENCODER_TEST_MAX_NUM_TOKENS = 8192


def _make_vision_config():
    """Minimal PretrainedConfig mimicking Nemotron-Nano-V3's `vision_config`.

    Patterned on the `vision_config` block of config.json shipped with newer nemotron nano
    multimodal models; fields pared down to what RADIOVisionModel reads.
    """
    config = PretrainedConfig()
    config.patch_size = 16
    config.adaptor_names = None
    config.feature_normalizer_config = None
    config.inter_feature_normalizer_config = None
    config.max_resolution = 64
    config.vitdet_window_size = None
    config.preferred_resolution = (32, 32)
    config.video_temporal_patch_size = 1
    config.separate_video_embedder = True
    config.torch_dtype = torch.bfloat16
    config.args = {
        "model": "vit_tiny_test",
        "in_chans": None,
        "input_size": None,
        "drop": 0.0,
        "cpe_max_size": 64,
        "cls_token_per_teacher": False,
        "teachers": [{"name": "dummy"}],
        "register_multiple": None,
        "cpe_num_registers": None,
    }
    return config


@pytest.fixture
def tiny_vit_config():
    with mock.patch.dict(
        modeling_radio.VIT_TIMM_CONFIG_BY_NAME,
        {"vit_tiny_test": _TINY_VIT},
    ):
        yield


def _make_fp8_model_config():
    vision_config = _make_vision_config()
    quant_config = QuantConfig(
        quant_algo=QuantAlgo.FP8,
        kv_cache_quant_algo=QuantAlgo.FP8,
    )
    return model_config_lib.ModelConfig(
        pretrained_config=vision_config,
        quant_config=quant_config,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_radio_fp8_parent_kv_cache_does_not_leak_into_vit(tiny_vit_config):
    """When the parent LLM uses FP8 KV cache, the RADIO vision encoder must not inherit it.

    A ViT has no KV cache (kv_cache_manager=None). If `kv_cache_quant_algo=FP8` leaks into the
    vision tower, FlashInfer raises at forward time about it not being supported.
    """
    vision_model = RADIOVisionModel(_make_fp8_model_config(), disable_quantization=True)
    # Engine normally calls this after model load by walking `model.modules()`
    # for `MultimodalEncoderMixin` instances (the mixin lives on the inner
    # `VisionTransformer`, not the `RADIOVisionModel` wrapper); standalone tests
    # must mirror that themselves.
    for module in vision_model.modules():
        if isinstance(module, MultimodalEncoderMixin):
            module.setup_attn_metadata(
                max_num_requests=_ENCODER_TEST_MAX_NUM_REQUESTS,
                max_num_tokens=_ENCODER_TEST_MAX_NUM_TOKENS,
            )

    device = torch.device("cuda")
    dtype = torch.bfloat16
    vision_model = vision_model.to(device).to(dtype)

    # 32x32 image: multiple of `patch_size=16`, so `min_resolution_step` is satisfied.
    pixel_values = torch.randn(1, 3, 32, 32, device=device, dtype=dtype)

    with torch.inference_mode():
        features = vision_model.forward(pixel_values)

    assert features.shape[0] == 1
    assert features.shape[-1] == _TINY_VIT.embed_dim


def _make_bf16_model_config():
    """Plain bf16 ModelConfig — keeps the runner's metadata factory simple."""
    return model_config_lib.ModelConfig(
        pretrained_config=_make_vision_config(),
        quant_config=QuantConfig(),
    )


def _init_finite_weights(model: torch.nn.Module) -> None:
    """Re-initialize all parameters so the toy ViT produces finite outputs.

    The tiny config in this file uses random PyTorch init which routinely produces NaN/inf through
    the attention kernel on bf16. A small-std normal init + zero bias makes the eager forward
    numerically stable so the graph-vs-eager comparison below can do a bit-exact check.
    """
    with torch.no_grad():
        for param in model.parameters():
            if param.dim() >= 2:
                torch.nn.init.normal_(param, std=0.02)
            else:
                param.zero_()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_radio_blocks_cuda_graph_matches_eager(tiny_vit_config):
    """Block-loop CUDA graph wiring must produce the same output as eager.

    Runs the tiny ViT once in eager mode to capture a reference, then enables the encoder graph
    runner and re-runs the same input. The post-merger output must match bit-exactly: capture /
    replay just records the same kernel launches the eager path runs.
    """
    device = torch.device("cuda")
    dtype = torch.bfloat16

    torch.manual_seed(0)
    pixel_values = torch.randn(1, 3, 32, 32, device=device, dtype=dtype)

    model = (
        RADIOVisionModel(
            _make_bf16_model_config(),
            disable_quantization=True,
            encoder_cuda_graph_config=MultimodalEncoderCudaGraphConfig(
                buckets=[(16, 1)],
                enable_padding=True,
                warmup_steps=2,
            ),
        )
        .to(device)
        .to(dtype)
    )
    model.eval()
    _init_finite_weights(model)
    assert model.model_config.attn_backend == "FLASHINFER"

    with torch.no_grad():
        eager_out = model.forward(pixel_values).clone()
    assert not eager_out.isnan().any(), "eager fixture produced NaN; tighten init"

    # Bucket sized comfortably above the toy seq_lengths so dummy-context padding is exercised: a
    # 32x32 image with patch_size=16 yields four patches plus the CLS token, so 16 tokens leaves
    # ample room.
    model.enable_blocks_cuda_graph(device=device)
    vision_tower = model.radio_model.model
    runner = vision_tower._blocks_graph_runner
    assert runner is not None
    assert runner._captured

    with (
        mock.patch.object(runner, "maybe_run", wraps=runner.maybe_run) as maybe_run,
        mock.patch.object(
            vision_tower,
            "_run_blocks_eager",
            side_effect=AssertionError("CUDA graph replay fell back to the eager block loop."),
        ),
        torch.no_grad(),
    ):
        graph_out = model.forward(pixel_values)

    maybe_run.assert_called_once()

    torch.testing.assert_close(graph_out, eager_out, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_radio_blocks_cuda_graph_falls_back_when_no_bucket(tiny_vit_config):
    """A configured-but-unmatched request must take the eager fallback path.

    When no bucket has enough capacity, `maybe_run` returns `None` and
    the existing eager block loop runs against `self.attn_metadata`.
    """
    device = torch.device("cuda")
    dtype = torch.bfloat16
    pixel_values = torch.randn(1, 3, 32, 32, device=device, dtype=dtype)

    model = (
        RADIOVisionModel(
            _make_bf16_model_config(),
            disable_quantization=True,
            encoder_cuda_graph_config=MultimodalEncoderCudaGraphConfig(
                buckets=[(1, 1)],
                enable_padding=False,
            ),
        )
        .to(device)
        .to(dtype)
    )
    model.eval()

    # Bucket too small for even a single real token — every request falls back.
    model.enable_blocks_cuda_graph(device=device)
    with torch.no_grad():
        features = model.forward(pixel_values)

    assert features.shape[0] == 1
    assert features.shape[-1] == _TINY_VIT.embed_dim
    # The runner is installed and the configured bucket is captured at
    # startup, but the request shape did not match any bucket, so the
    # no-bucket-match fallback warning fired.
    runner = model.radio_model.model._blocks_graph_runner
    assert runner is not None
    assert runner._warned_no_bucket_match


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_radio_blocks_cuda_graph_matches_eager_multi_context(tiny_vit_config):
    """Dynamic-resolution inputs must replay through the graph runner with multiple contexts."""
    device = torch.device("cuda")
    dtype = torch.bfloat16

    image_sizes = [(32, 32), (16, 16)]
    patch_dim = 3 * 16 * 16
    total_patches = sum(modeling_radio.calc_seq_len(size, patch_size=16) for size in image_sizes)

    torch.manual_seed(0)
    pixel_values = torch.randn(1, total_patches, patch_dim, device=device, dtype=dtype)

    model = (
        RADIOVisionModel(
            _make_bf16_model_config(),
            disable_quantization=True,
            encoder_cuda_graph_config=MultimodalEncoderCudaGraphConfig(
                buckets=[(16, 2)],
                enable_padding=True,
                warmup_steps=2,
            ),
        )
        .to(device)
        .to(dtype)
    )
    model.eval()
    _init_finite_weights(model)

    with torch.no_grad():
        eager_out = model.forward(pixel_values, image_sizes=image_sizes).clone()
    assert not eager_out.isnan().any(), "eager fixture produced NaN; tighten init"

    model.enable_blocks_cuda_graph(device=device)
    vision_tower = model.radio_model.model
    runner = vision_tower._blocks_graph_runner
    assert runner is not None
    assert runner._captured

    with (
        mock.patch.object(runner, "maybe_run", wraps=runner.maybe_run) as maybe_run,
        mock.patch.object(
            vision_tower,
            "_run_blocks_eager",
            side_effect=AssertionError("CUDA graph replay fell back to the eager block loop."),
        ),
        torch.no_grad(),
    ):
        graph_out = model.forward(pixel_values, image_sizes=image_sizes)

    maybe_run.assert_called_once()

    torch.testing.assert_close(graph_out, eager_out, rtol=0, atol=0)
