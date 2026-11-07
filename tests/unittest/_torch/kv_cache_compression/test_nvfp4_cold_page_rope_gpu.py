# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Single-GPU QA regression with real model configs and synthetic KV pages."""

import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.kv_cache_compression.quantization_for_cold_page.nvfp4_quantization import (
    Nvfp4ColdPageQuantizationCompression,
)
from tensorrt_llm._torch.pyexecutor.config_utils import load_pretrained_config
from tensorrt_llm._torch.pyexecutor.resource_manager import DataType
from tensorrt_llm._utils import is_sm_100f
from tensorrt_llm.llmapi.llm_args import ColdPageQuantizationCompressionConfig
from tensorrt_llm.runtime.kv_cache_manager_v2 import AttentionLayerConfig, BufferConfig


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("model_name", ("GLM-5-NVFP4", "Qwen3.5-4B"))
def test_real_config_cold_page_roundtrip_preserves_rope_bytes(model_name: str) -> None:
    """Check the Attention-defined RoPE slice, independently of codec metadata."""
    if not is_sm_100f():
        pytest.skip("NVFP4 cold-page kernels require an SM100-family GPU")
    models_root = os.environ.get("LLM_MODELS_ROOT")
    if not models_root or not (Path(models_root) / model_name / "config.json").is_file():
        pytest.skip(f"LLM_MODELS_ROOT must contain {model_name}/config.json")
    config = load_pretrained_config(str(Path(models_root) / model_name), trust_remote_code=True)
    text_config = config.get_text_config()
    if model_name == "GLM-5-NVFP4":
        # MLA Attention stores [latent NoPE][RoPE] in its key-only cache.
        head_dim = text_config.kv_lora_rank + text_config.qk_rope_head_dim
        rope_slice = slice(text_config.kv_lora_rank, head_dim)
        nope_slice = slice(0, text_config.kv_lora_rank)
        num_heads, roles = 1, ("key",)
    else:
        # Partial-rotary GQA Attention rotates the leading K head elements.
        head_dim = text_config.head_dim
        rope_dim = int(head_dim * text_config.partial_rotary_factor)
        rope_slice, nope_slice = slice(0, rope_dim), slice(rope_dim, head_dim)
        num_heads, roles = text_config.num_key_value_heads, ("key", "value")
    assert 0 < rope_slice.stop - rope_slice.start < head_dim

    tokens_per_page = 64
    generator = torch.Generator(device="cuda").manual_seed(42)
    buffers = {
        role: torch.randn(
            (4, num_heads, tokens_per_page, head_dim), generator=generator, device="cuda"
        ).to(torch.bfloat16)
        for role in roles
    }
    original_key = buffers["key"][[0, 2]].clone()
    raw_page_bytes = buffers["key"][0].numel() * buffers["key"].element_size()
    cache_config = SimpleNamespace(
        tokens_per_block=tokens_per_page,
        layers=(
            AttentionLayerConfig(
                layer_id=0,
                buffers=[BufferConfig(role=role, size=raw_page_bytes) for role in roles],
            ),
        ),
    )
    provider = Nvfp4ColdPageQuantizationCompression(
        ColdPageQuantizationCompressionConfig(skip_rope_quantization=True),
        pretrained_config=config,
    )
    state = provider.build_codec_state(
        cache_config,
        runtime_dtype=DataType.BF16,
        pp_layers=(0,),
        num_kv_heads_per_layer=(num_heads,),
        head_dim_per_layer=(head_dim,),
    )
    lifecycle = SimpleNamespace(
        layers={
            0: {
                role: SimpleNamespace(
                    raw_base=buffer.data_ptr(),
                    raw_slot_bytes=raw_page_bytes,
                    raw_bytes=raw_page_bytes,
                )
                for role, buffer in buffers.items()
            }
        }
    )
    provider.configure(state, [lifecycle])
    # Device cold scratch isolates the codec roundtrip; this is not a Host-migration test.
    cold = torch.empty(
        2 * state.lifecycle_metadata[0].cold_page_bytes, dtype=torch.uint8, device="cuda"
    )
    encode_pairs = torch.tensor([[1, 0], [0, 2]], dtype=torch.int32, device="cpu")
    decode_pairs = torch.tensor([[1, 1], [3, 0]], dtype=torch.int32, device="cpu")
    stream = torch.cuda.current_stream()
    provider.encode_cold_pages(
        state, 0, cold.data_ptr(), encode_pairs.data_ptr(), 2, stream.cuda_stream
    )
    provider.decode_cold_pages(
        state, 0, cold.data_ptr(), decode_pairs.data_ptr(), 2, stream.cuda_stream
    )
    stream.synchronize()
    restored_key = buffers["key"][[1, 3]]
    assert torch.equal(
        restored_key[..., rope_slice].view(torch.uint8),
        original_key[..., rope_slice].view(torch.uint8),
    )
    assert not torch.equal(restored_key[..., nope_slice], original_key[..., nope_slice])
