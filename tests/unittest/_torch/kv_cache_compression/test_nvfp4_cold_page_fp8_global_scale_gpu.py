# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Single-GPU round trips of FP8 Pages through NVFP4 cold Pages for each global-scale policy."""

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.kv_cache_compression.quantization_for_cold_page.nvfp4_quantization import (
    Nvfp4ColdPageQuantizationCompression,
)
from tensorrt_llm._torch.pyexecutor.resource_manager import DataType
from tensorrt_llm._utils import is_sm_100f
from tensorrt_llm.llmapi.llm_args import ColdPageQuantizationCompressionConfig
from tensorrt_llm.runtime.kv_cache_manager_v2 import AttentionLayerConfig, BufferConfig

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not is_sm_100f(),
    reason="NVFP4 cold-page kernels require an SM100-family GPU",
)

_HEADS, _TOKENS, _HEAD_DIM, _PAGES = 4, 32, 64, 4


def _fp8_pages(block_amax_log2: tuple[float, float], seed: int = 0) -> torch.Tensor:
    """FP8 Pages [pages, heads, tokens, head_dim] whose 16-value blocks span a range of magnitudes."""

    generator = torch.Generator(device="cuda").manual_seed(seed)
    shape = (_PAGES, _HEADS, _TOKENS, _HEAD_DIM // 16, 16)
    low, high = block_amax_log2
    block_amax = torch.exp2(
        torch.empty(shape[:-1] + (1,), device="cuda").uniform_(low, high, generator=generator)
    )
    values = torch.empty(shape, device="cuda").uniform_(-1.0, 1.0, generator=generator)
    values = values / values.abs().amax(-1, keepdim=True) * block_amax
    return values.reshape(_PAGES, _HEADS, _TOKENS, _HEAD_DIM).to(torch.float8_e4m3fn)


class _Codec:
    """One FP8 K/V layer, encoded to and decoded from device cold scratch."""

    def __init__(self, global_scale: str, key: torch.Tensor, value: torch.Tensor) -> None:
        self.buffers = {"key": key, "value": value}
        page_bytes = key[0].numel()
        cache_config = SimpleNamespace(
            tokens_per_block=_TOKENS,
            layers=(
                AttentionLayerConfig(
                    layer_id=0,
                    buffers=[BufferConfig(role=role, size=page_bytes) for role in self.buffers],
                ),
            ),
        )
        self.provider = Nvfp4ColdPageQuantizationCompression(
            ColdPageQuantizationCompressionConfig(global_scale=global_scale),
            pretrained_config=SimpleNamespace(model_type="muse_spark"),
        )
        self.state = self.provider.build_codec_state(
            cache_config,
            runtime_dtype=DataType.FP8,
            pp_layers=(0,),
            num_kv_heads_per_layer=(_HEADS,),
            head_dim_per_layer=(_HEAD_DIM,),
        )
        lifecycle = SimpleNamespace(
            layers={
                0: {
                    role: SimpleNamespace(
                        raw_base=buffer.data_ptr(), raw_slot_bytes=page_bytes, raw_bytes=page_bytes
                    )
                    for role, buffer in self.buffers.items()
                }
            }
        )
        self.provider.configure(self.state, [lifecycle])
        self.cold_page_bytes = self.state.lifecycle_metadata[0].cold_page_bytes
        self.cold = torch.zeros(_PAGES * self.cold_page_bytes, dtype=torch.uint8, device="cuda")
        self.stream = torch.cuda.current_stream()

    def encode(self, gpu_to_cold: list[tuple[int, int]]) -> None:
        pairs = torch.tensor([[cold, gpu] for gpu, cold in gpu_to_cold], dtype=torch.int32)
        self.provider.encode_cold_pages(
            self.state,
            0,
            self.cold.data_ptr(),
            pairs.data_ptr(),
            len(pairs),
            self.stream.cuda_stream,
        )
        self.stream.synchronize()

    def decode(self, cold_to_gpu: list[tuple[int, int]]) -> None:
        pairs = torch.tensor([[gpu, cold] for cold, gpu in cold_to_gpu], dtype=torch.int32)
        self.provider.decode_cold_pages(
            self.state,
            0,
            self.cold.data_ptr(),
            pairs.data_ptr(),
            len(pairs),
            self.stream.cuda_stream,
        )
        self.stream.synchronize()

    def cold_page(self, index: int) -> torch.Tensor:
        return self.cold[index * self.cold_page_bytes : (index + 1) * self.cold_page_bytes].clone()


def _round_trip(
    global_scale: str, source: dict[str, torch.Tensor]
) -> tuple[dict[str, torch.Tensor], _Codec]:
    """Pages 0-1 hold the source; they are encoded to cold slots 0-1 and decoded into Pages 2-3."""

    buffers = {role: torch.zeros_like(pages) for role, pages in source.items()}
    for role, pages in source.items():
        buffers[role][:2] = pages[:2]
    codec = _Codec(global_scale, buffers["key"], buffers["value"])
    codec.encode([(0, 0), (1, 1)])
    codec.decode([(0, 2), (1, 3)])
    return {role: buffer[2:].clone() for role, buffer in buffers.items()}, codec


def _relative_error(restored: torch.Tensor, source: torch.Tensor) -> float:
    return ((restored.float() - source.float()).norm() / source.float().norm()).item()


def test_fp8_block_amax_restores_each_block_maximum_exactly() -> None:
    source = {"key": _fp8_pages((-9.0, 8.8), seed=1), "value": _fp8_pages((-9.0, 8.8), seed=2)}
    restored, _ = _round_trip("fp8_block_amax", source)
    for role, pages in source.items():
        original_amax = pages[:2].float().reshape(-1, 16).abs().amax(-1)
        restored_amax = restored[role].float().reshape(-1, 16).abs().amax(-1)
        assert torch.equal(restored_amax, original_amax), role


def test_fp8_block_amax_is_more_accurate_than_identity_and_keeps_small_blocks() -> None:
    source = {"key": _fp8_pages((-9.0, 8.8), seed=3), "value": _fp8_pages((-9.0, 8.8), seed=4)}
    identity, _ = _round_trip("identity", source)
    tuned, _ = _round_trip("fp8_block_amax", source)
    for role, pages in source.items():
        assert _relative_error(tuned[role], pages[:2]) < _relative_error(identity[role], pages[:2])

    # Identity block scales flush to zero below an amax of about 6 * 2^-10; an exact amax scale never does.
    small = {"key": _fp8_pages((-8.5, -8.0), seed=5), "value": _fp8_pages((-8.5, -8.0), seed=6)}
    identity, _ = _round_trip("identity", small)
    tuned, _ = _round_trip("fp8_block_amax", small)
    for role in small:
        assert torch.count_nonzero(identity[role].float()) == 0
        assert torch.count_nonzero(tuned[role].float().reshape(-1, 16).abs().amax(-1)) == 2 * (
            _HEADS * _TOKENS * _HEAD_DIM // 16
        )


def test_fp8_block_amax_second_host_trip_reproduces_the_first() -> None:
    """A Page onboarded from host and evicted again is re-encoded from its decoded FP8 values."""

    source = {"key": _fp8_pages((-2.0, 8.8), seed=7), "value": _fp8_pages((-2.0, 8.8), seed=8)}
    first, codec = _round_trip("fp8_block_amax", source)
    first_cold = [codec.cold_page(0), codec.cold_page(1)]
    codec.encode([(2, 2), (3, 3)])
    assert torch.equal(codec.cold_page(2), first_cold[0])
    assert torch.equal(codec.cold_page(3), first_cold[1])
    codec.decode([(2, 0), (3, 1)])
    for role, buffer in codec.buffers.items():
        assert torch.equal(buffer[:2].view(torch.uint8), first[role].view(torch.uint8)), role
