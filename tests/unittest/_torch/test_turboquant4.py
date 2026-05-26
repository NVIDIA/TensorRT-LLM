# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from types import SimpleNamespace

import pytest
import torch

import tensorrt_llm._torch.attention_backend.trtllm as trtllm_backend
from tensorrt_llm._torch.attention_backend import (
    AttentionInputType,
    TrtllmAttention,
    TrtllmAttentionMetadata,
    VanillaAttention,
    trtllm_gen,
)
from tensorrt_llm._torch.attention_backend.interface import PredefinedAttentionMask
from tensorrt_llm._torch.attention_backend.vanilla import repeat_kv
from tensorrt_llm._torch.modules.attention import Attention
from tensorrt_llm._torch.modules.turboquant4 import (
    TURBOQUANT4_CENTROIDS,
    fwht,
    turboquant4_attention,
    turboquant4_batch_attention,
    turboquant4_dequantize,
    turboquant4_dequantize_cache,
    turboquant4_dequantize_value_cache,
    turboquant4_quantize,
    turboquant4_quantize_dequantize,
    turboquant4_update_cache,
    read_turboquant4_dense_key_cache,
)
from tensorrt_llm._torch.pyexecutor import resource_manager as resource_manager_mod
from tensorrt_llm._torch.pyexecutor._util import (
    _sync_turboquant4_kv_cache_config,
    get_kv_cache_manager_cls,
)
from tensorrt_llm._torch.pyexecutor.mamba_cache_manager import MambaHybridCacheManager
from tensorrt_llm._torch.pyexecutor.model_engine import PyTorchModelEngine
from tensorrt_llm._torch.pyexecutor.resource_manager import (
    CacheTypeCpp,
    DataType,
    KVCacheManagerV2,
    Role,
)
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantAlgo, QuantConfig


class TestTurboQuant4CacheManagerRouting:
    @staticmethod
    def _model_config(pretrained_config=None, sparse_attention_config=None, quant_config=None):
        return SimpleNamespace(
            pretrained_config=pretrained_config or SimpleNamespace(),
            sparse_attention_config=sparse_attention_config,
            quant_config=quant_config,
        )

    @staticmethod
    def _kv_config(
        dtype="auto",
        use_kv_cache_manager_v2=False,
        max_attention_window=None,
        event_buffer_max_size=0,
        enable_block_reuse=True,
    ):
        return SimpleNamespace(
            dtype=dtype,
            use_kv_cache_manager_v2=use_kv_cache_manager_v2,
            max_attention_window=max_attention_window,
            event_buffer_max_size=event_buffer_max_size,
            enable_block_reuse=enable_block_reuse,
        )

    def test_turboquant4_dtype_uses_kv_cache_manager_v2(self):
        kv_config = self._kv_config(dtype="turboquant4")
        manager_cls = get_kv_cache_manager_cls(
            self._model_config(),
            kv_config,
        )
        assert manager_cls is KVCacheManagerV2
        assert kv_config.use_kv_cache_manager_v2
        assert not kv_config.enable_block_reuse

    def test_turboquant4_quant_config_uses_kv_cache_manager_v2(self):
        kv_config = self._kv_config(dtype="auto")
        manager_cls = get_kv_cache_manager_cls(
            self._model_config(quant_config=QuantConfig(kv_cache_quant_algo=QuantAlgo.TURBOQUANT4)),
            kv_config,
        )
        assert manager_cls is KVCacheManagerV2
        assert kv_config.dtype == "turboquant4"
        assert kv_config.use_kv_cache_manager_v2
        assert not kv_config.enable_block_reuse

    def test_turboquant4_quant_config_normalizes_kv_cache_dtype(self):
        kv_config = self._kv_config(dtype="auto")
        assert _sync_turboquant4_kv_cache_config(
            kv_config,
            QuantConfig(kv_cache_quant_algo=QuantAlgo.TURBOQUANT4),
        )
        assert kv_config.dtype == "turboquant4"
        assert kv_config.use_kv_cache_manager_v2
        assert not kv_config.enable_block_reuse

    def test_turboquant4_raw_quant_algo_normalizes_kv_cache_dtype(self):
        kv_config = self._kv_config(dtype="auto")
        assert _sync_turboquant4_kv_cache_config(
            kv_config,
            SimpleNamespace(kv_cache_quant_algo=QuantAlgo.TURBOQUANT4),
        )
        assert kv_config.dtype == "turboquant4"
        assert kv_config.use_kv_cache_manager_v2
        assert not kv_config.enable_block_reuse

    def test_turboquant4_raw_string_quant_algo_normalizes_kv_cache_dtype(self):
        kv_config = self._kv_config(dtype="auto")
        assert _sync_turboquant4_kv_cache_config(
            kv_config,
            SimpleNamespace(kv_cache_quant_algo="turboquant4"),
        )
        assert kv_config.dtype == "turboquant4"
        assert kv_config.use_kv_cache_manager_v2
        assert not kv_config.enable_block_reuse

    def test_turboquant4_kv_cache_does_not_require_calibration(self):
        quant_config = QuantConfig(kv_cache_quant_algo=QuantAlgo.TURBOQUANT4)
        assert not quant_config._requires_calibration

    def test_turboquant4_rejects_sparse_attention_manager(self):
        with pytest.raises(ValueError, match="sparse attention"):
            get_kv_cache_manager_cls(
                self._model_config(sparse_attention_config=SimpleNamespace()),
                self._kv_config(dtype="turboquant4"),
            )

    def test_turboquant4_rejects_mla_manager(self):
        config = SimpleNamespace(kv_lora_rank=64, qk_rope_head_dim=64)
        with pytest.raises(ValueError, match="MLA"):
            get_kv_cache_manager_cls(
                self._model_config(pretrained_config=config),
                self._kv_config(dtype="turboquant4"),
            )

    def test_turboquant4_rejects_sliding_window_manager(self):
        with pytest.raises(ValueError, match="sliding-window"):
            get_kv_cache_manager_cls(
                self._model_config(),
                self._kv_config(dtype="turboquant4", max_attention_window=[128]),
            )

    def test_turboquant4_rejects_event_buffer_manager(self):
        with pytest.raises(ValueError, match="event buffers"):
            get_kv_cache_manager_cls(
                self._model_config(),
                self._kv_config(dtype="turboquant4", event_buffer_max_size=1),
            )

    def test_turboquant4_rejects_hybrid_mamba_manager(self):
        config = SimpleNamespace(hybrid_override_pattern="M-M-")
        with pytest.raises(ValueError, match="hybrid Mamba"):
            get_kv_cache_manager_cls(
                self._model_config(pretrained_config=config),
                self._kv_config(dtype="turboquant4"),
            )

    def test_non_turboquant4_hybrid_still_uses_mamba_hybrid_manager(self):
        config = SimpleNamespace(hybrid_override_pattern="M-M-")
        manager_cls = get_kv_cache_manager_cls(
            self._model_config(pretrained_config=config),
            self._kv_config(dtype="auto"),
        )
        assert manager_cls is MambaHybridCacheManager


class _FakeKVCacheImpl:
    def __init__(self, num_pages=4):
        self.num_pages = num_pages
        self.page_strides = {
            Role.KEY: 1024,
            Role.VALUE: 256,
            Role.VALUE_BLOCK_SCALE: 64,
        }

    def get_mem_pool_base_address(self, layer_offset, role):
        del layer_offset
        if role == Role.KEY:
            return 1000
        if role == Role.VALUE:
            return 2000
        if role == Role.VALUE_BLOCK_SCALE:
            return 3000
        raise AssertionError(f"unexpected role: {role}")

    def get_page_stride(self, layer_offset, role):
        del layer_offset
        return self.page_strides[role]

    def get_page_index_upper_bound(self, layer_offset, role):
        del layer_offset, role
        return self.num_pages

    def get_page_index_scale(self, layer_offset, role):
        del layer_offset, role
        return 1


class _CapturedTensorWrapper:
    def __init__(self, data_ptr, dtype, shape):
        self.data_ptr = data_ptr
        self.dtype = dtype
        self.shape = tuple(shape)


class TestTurboQuant4KVCacheManagerV2Buffers:
    @staticmethod
    def _manager():
        manager = KVCacheManagerV2.__new__(KVCacheManagerV2)
        manager.is_turboquant4 = True
        manager.dtype = DataType.HALF
        manager.layer_offsets = [0]
        manager.impl = _FakeKVCacheImpl()
        manager.kv_cache_type = object()
        manager.kv_factor = 2
        manager.tokens_per_block = 4
        manager.num_kv_heads_per_layer = [3]
        manager.head_dim = 128
        manager.num_local_layers = 1
        manager.kv_cache_map = {}
        return manager

    def test_get_turboquant4_value_scale_buffers_shape(self, monkeypatch):
        monkeypatch.setattr(resource_manager_mod, "TensorWrapper", _CapturedTensorWrapper)
        monkeypatch.setattr(resource_manager_mod, "convert_to_torch_tensor", lambda tensor: tensor)
        manager = self._manager()

        scales = manager.get_turboquant4_value_scale_buffers(0)

        assert scales.data_ptr == 3000
        assert scales.dtype == DataType.FLOAT
        assert scales.shape == (4, 4, 3, 1)

    def test_get_turboquant4_value_scale_buffers_hnd_shape(self, monkeypatch):
        monkeypatch.setattr(resource_manager_mod, "TensorWrapper", _CapturedTensorWrapper)
        monkeypatch.setattr(resource_manager_mod, "convert_to_torch_tensor", lambda tensor: tensor)
        manager = self._manager()

        scales = manager.get_turboquant4_value_scale_buffers(0, kv_layout="HND")

        assert scales.shape == (4, 3, 4, 1)

    def test_get_turboquant4_key_and_value_buffers(self, monkeypatch):
        monkeypatch.setattr(resource_manager_mod, "TensorWrapper", _CapturedTensorWrapper)
        monkeypatch.setattr(resource_manager_mod, "convert_to_torch_tensor", lambda tensor: tensor)
        manager = self._manager()

        key_cache = manager.get_turboquant4_key_buffers(0)
        value_cache = manager.get_turboquant4_value_buffers(0)

        assert key_cache.data_ptr == 1000
        assert key_cache.dtype == DataType.HALF
        assert key_cache.shape == (4, 4, 3, 128)
        assert value_cache.data_ptr == 2000
        assert value_cache.dtype == torch.uint8
        assert value_cache.shape == (4, 4, 3, 64)

    def test_get_buffers_rejects_turboquant4(self):
        manager = self._manager()

        with pytest.raises(ValueError, match="asymmetric key/value buffers"):
            manager.get_buffers(0)

    def test_get_num_free_blocks_uses_key_block_count(self):
        manager = self._manager()

        assert manager.get_num_free_blocks() == 4

    def test_cache_bytes_include_fp32_vector_scales(self):
        manager = self._manager()
        key_bytes = resource_manager_mod.get_size_in_bytes(3 * 128, DataType.HALF)
        value_bytes = resource_manager_mod.get_size_in_bytes(3 * 128, DataType.NVFP4)
        scale_bytes = 3 * 4

        assert manager.get_layer_bytes_per_token(0, Role.KEY) == key_bytes
        assert manager.get_layer_bytes_per_token(0, Role.VALUE) == value_bytes
        assert manager.get_layer_bytes_per_token(0, Role.VALUE_BLOCK_SCALE) == scale_bytes
        assert manager.get_layer_bytes_per_token(0, Role.ALL) == key_bytes + value_bytes + scale_bytes
        assert manager.get_cache_bytes_per_token() == key_bytes + value_bytes + scale_bytes

    def test_draft_token_relocation_rejects_turboquant4(self):
        manager = self._manager()
        request = SimpleNamespace(
            state=resource_manager_mod.LlmRequestState.GENERATION_IN_PROGRESS,
            py_num_accepted_draft_tokens=1,
            py_num_accepted_draft_tokens_indices=[0],
        )
        scheduled_batch = SimpleNamespace(
            generation_requests=[request],
            all_requests=lambda: [request],
        )

        with pytest.raises(RuntimeError, match="draft-token relocation"):
            resource_manager_mod._update_kv_cache_draft_token_location(
                manager, scheduled_batch, attn_metadata=None, kv_cache_dtype_byte_size=0.5
            )

    def test_turboquant4_rejects_key_only_cache_type(self):
        with pytest.raises(ValueError, match="SELFKONLY"):
            KVCacheManagerV2._validate_turboquant4_cache_type(
                SimpleNamespace(dtype="turboquant4"),
                CacheTypeCpp.SELFKONLY,
            )

    @pytest.mark.parametrize("torch_dtype", [torch.float32, "float32"])
    def test_model_engine_turboquant4_kv_byte_size_uses_model_dtype(
            self, torch_dtype):
        engine = PyTorchModelEngine.__new__(PyTorchModelEngine)
        engine.model = SimpleNamespace(
            config=SimpleNamespace(torch_dtype=torch_dtype),
            model_config=SimpleNamespace(
                quant_config=QuantConfig(
                    kv_cache_quant_algo=QuantAlgo.TURBOQUANT4)),
        )

        assert engine.get_kv_cache_dtype_byte_size() == 4


class _FakeModelConfig:
    def __init__(self, pretrained_config, quant_config):
        self.pretrained_config = pretrained_config
        self.quant_config = quant_config

    def get_num_attention_layers(self):
        return self.pretrained_config.num_hidden_layers


def test_turboquant4_v2_static_cache_size_uses_ceil_tp_kv_heads():
    config = SimpleNamespace(
        hidden_size=1024,
        num_attention_heads=16,
        num_key_value_heads=5,
        head_dim=64,
        num_hidden_layers=2,
    )
    model_config = _FakeModelConfig(
        config,
        QuantConfig(kv_cache_quant_algo=QuantAlgo.TURBOQUANT4),
    )
    mapping = Mapping(world_size=2, tp_size=2, rank=0)

    bytes_per_token = KVCacheManagerV2.get_cache_size_per_token(
        model_config,
        mapping,
    )

    local_kv_heads = 3
    num_layers = 2
    expected_key = num_layers * local_kv_heads * 64 * 2
    expected_value = num_layers * local_kv_heads * (64 // 2)
    expected_scales = num_layers * local_kv_heads * 4
    assert bytes_per_token == expected_key + expected_value + expected_scales


def test_turboquant4_v2_static_cache_size_uses_model_dtype_for_dense_keys():
    config = SimpleNamespace(
        hidden_size=1024,
        num_attention_heads=16,
        num_key_value_heads=5,
        head_dim=64,
        num_hidden_layers=2,
        torch_dtype=torch.float32,
    )
    model_config = _FakeModelConfig(
        config,
        QuantConfig(kv_cache_quant_algo=QuantAlgo.TURBOQUANT4),
    )
    mapping = Mapping(world_size=2, tp_size=2, rank=0)

    bytes_per_token = KVCacheManagerV2.get_cache_size_per_token(
        model_config,
        mapping,
    )

    local_kv_heads = 3
    num_layers = 2
    expected_key = num_layers * local_kv_heads * 64 * 4
    expected_value = num_layers * local_kv_heads * (64 // 2)
    expected_scales = num_layers * local_kv_heads * 4
    assert bytes_per_token == expected_key + expected_value + expected_scales


def test_turboquant4_v2_static_cache_size_uses_per_layer_kv_heads():
    config = SimpleNamespace(
        hidden_size=1024,
        num_attention_heads=16,
        num_key_value_heads=[1, 3],
        head_dim=64,
        num_hidden_layers=2,
    )
    model_config = _FakeModelConfig(
        config,
        QuantConfig(kv_cache_quant_algo=QuantAlgo.TURBOQUANT4),
    )
    mapping = Mapping(world_size=2, tp_size=2, rank=0)

    bytes_per_token = KVCacheManagerV2.get_cache_size_per_token(
        model_config,
        mapping,
    )

    local_kv_heads_sum = 1 + 2
    expected_key = local_kv_heads_sum * 64 * 2
    expected_value = local_kv_heads_sum * (64 // 2)
    expected_scales = local_kv_heads_sum * 4
    assert bytes_per_token == expected_key + expected_value + expected_scales


class TestFWHT:
    def test_self_inverse(self):
        """fwht(fwht(x)) should return x (normalized WHT is its own inverse)."""
        x = torch.randn(4, 128)
        result = fwht(fwht(x))
        torch.testing.assert_close(result, x, atol=1e-5, rtol=1e-5)

    def test_self_inverse_bf16(self):
        """Self-inverse property should hold for bfloat16 inputs."""
        x = torch.randn(4, 128, dtype=torch.bfloat16)
        result = fwht(fwht(x))
        torch.testing.assert_close(result, x, atol=2e-2, rtol=2e-2)

    def test_preserves_norm(self):
        """WHT is orthogonal, so it preserves L2 norms."""
        x = torch.randn(8, 128)
        expected = x.clone()
        transformed = fwht(x)
        torch.testing.assert_close(x, expected)
        x_norms = expected.norm(dim=-1)
        t_norms = transformed.norm(dim=-1)
        torch.testing.assert_close(x_norms, t_norms, atol=1e-5, rtol=1e-5)

    def test_does_not_mutate_float32_input(self):
        """WHT should not mutate float32 inputs in place."""
        x = torch.randn(4, 128)
        expected = x.clone()
        _ = fwht(x)
        torch.testing.assert_close(x, expected)

    def test_non_contiguous_input(self):
        """WHT should work for non-contiguous tensors."""
        x = torch.randn(128, 4).transpose(0, 1)
        assert not x.is_contiguous()
        result = fwht(fwht(x))
        torch.testing.assert_close(result, x, atol=1e-5, rtol=1e-5)

    def test_small_dims(self):
        """WHT should work for small power-of-2 dimensions."""
        for dim in [2, 4, 8, 16, 32, 64]:
            x = torch.randn(3, dim)
            result = fwht(fwht(x))
            torch.testing.assert_close(result, x, atol=1e-5, rtol=1e-5)

    def test_non_power_of_2_raises(self):
        """Non-power-of-2 dimensions should raise a ValueError."""
        x = torch.randn(3, 100)
        with pytest.raises(ValueError, match="power-of-2"):
            fwht(x)

    def test_batched(self):
        """WHT should work with arbitrary batch dimensions."""
        x = torch.randn(2, 3, 4, 128)
        result = fwht(fwht(x))
        torch.testing.assert_close(result, x, atol=1e-5, rtol=1e-5)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda(self):
        """WHT should work on CUDA tensors."""
        x = torch.randn(4, 128, device="cuda")
        result = fwht(fwht(x))
        torch.testing.assert_close(result, x, atol=1e-5, rtol=1e-5)


class TestCentroids:
    def test_symmetry(self):
        """Centroids should be symmetric around zero."""
        n = len(TURBOQUANT4_CENTROIDS)
        for i in range(n // 2):
            assert TURBOQUANT4_CENTROIDS[i] == pytest.approx(
                -TURBOQUANT4_CENTROIDS[n - 1 - i], abs=1e-6
            )

    def test_count(self):
        """Should have exactly 16 centroids for 4-bit quantization."""
        assert len(TURBOQUANT4_CENTROIDS) == 16

    def test_sorted(self):
        """Centroids should be in ascending order."""
        for i in range(len(TURBOQUANT4_CENTROIDS) - 1):
            assert TURBOQUANT4_CENTROIDS[i] < TURBOQUANT4_CENTROIDS[i + 1]


class TestQuantizeDequantize:
    def test_round_trip_mse(self):
        """Quantize-dequantize MSE should be small for Gaussian-like inputs."""
        torch.manual_seed(42)
        # Simulate typical KV cache vectors (head_dim=128).
        x = torch.randn(64, 128) * 0.1
        result = turboquant4_quantize_dequantize(x)

        mse = (x - result).pow(2).mean().item()
        rel_mse = mse / x.pow(2).mean().item()
        # turbo4-resurrection reports +0.23% PPL; relative MSE should be small.
        assert rel_mse < 0.05, f"Relative MSE {rel_mse:.4f} too high"

    def test_preserves_shape_and_dtype(self):
        """Output should match input shape and dtype."""
        x = torch.randn(4, 8, 128, dtype=torch.bfloat16)
        result = turboquant4_quantize_dequantize(x)
        assert result.shape == x.shape
        assert result.dtype == x.dtype

    def test_zero_vector(self):
        """Zero vector should remain zero (or near-zero) after round-trip."""
        x = torch.zeros(1, 128)
        result = turboquant4_quantize_dequantize(x)
        assert result.abs().max().item() < 1e-6

    def test_quantize_rejects_unsupported_dtype(self):
        x = torch.ones(1, 128, dtype=torch.int32)

        with pytest.raises(NotImplementedError, match="FP16"):
            turboquant4_quantize(x)

    def test_dequantize_rejects_invalid_dtypes(self):
        codes = torch.zeros(1, 64, dtype=torch.int8)
        scales = torch.ones(1, 1, dtype=torch.float32)

        with pytest.raises(ValueError, match="uint8"):
            turboquant4_dequantize(codes, scales, dtype=torch.float32)

        with pytest.raises(ValueError, match="float32"):
            turboquant4_dequantize(
                codes.to(torch.uint8), scales.to(torch.float16), dtype=torch.float32
            )

        with pytest.raises(NotImplementedError, match="FP16"):
            turboquant4_dequantize(codes.to(torch.uint8), scales, dtype=torch.int32)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_round_trip(self):
        """Round-trip should work on CUDA."""
        x = torch.randn(16, 128, device="cuda") * 0.1
        result = turboquant4_quantize_dequantize(x)
        mse = (x - result).pow(2).mean().item()
        rel_mse = mse / x.pow(2).mean().item()
        assert rel_mse < 0.05

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_native_cuda_ops_match_reference(self):
        """Native CUDA ops should match the Python reference implementation."""
        if not hasattr(torch.ops.trtllm, "turboquant4_quantize"):
            pytest.skip("TurboQuant4 native torch ops are not available")

        x = torch.randn(8, 128, device="cuda") * 0.1
        native_codes, native_scales = turboquant4_quantize(x)
        native_result = turboquant4_dequantize(native_codes, native_scales, dtype=x.dtype)

        ref_codes, ref_scales = turboquant4_quantize(x.cpu())
        ref_result = turboquant4_dequantize(ref_codes, ref_scales, dtype=x.dtype)

        torch.testing.assert_close(native_codes.cpu(), ref_codes)
        torch.testing.assert_close(native_scales.cpu(), ref_scales, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(native_result.cpu(), ref_result, atol=1e-5, rtol=1e-5)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_update_rejects_out_of_range_block_id_before_native(self):
        """Python wrapper should reject invalid page ids before native dispatch."""
        x = torch.randn(1, 1, 32, device="cuda")
        codes = torch.empty(1, 2, 4, 1, 16, dtype=torch.uint8, device="cuda")
        scales = torch.empty(1, 2, 4, 1, 1, dtype=torch.float32, device="cuda")

        with pytest.raises(RuntimeError, match="out of range"):
            turboquant4_update_cache(x, codes, scales, [1], 0, 0, 4)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_native_update_rejects_out_of_range_block_id(self):
        if not hasattr(torch.ops.trtllm, "turboquant4_update_cache"):
            pytest.skip("TurboQuant4 native torch ops are not available")

        x = torch.randn(1, 1, 128, device="cuda")
        codes = torch.empty(1, 2, 4, 1, 64, dtype=torch.uint8, device="cuda")
        scales = torch.empty(1, 2, 4, 1, 1, dtype=torch.float32, device="cuda")
        block_ids = torch.tensor([1], dtype=torch.int32, device="cuda")

        with pytest.raises(RuntimeError, match="block ids must be in"):
            torch.ops.trtllm.turboquant4_update_cache(
                x, codes, scales, block_ids, 0, 0, 4
            )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_native_batch_attention_rejects_out_of_range_block_id(self):
        if not hasattr(torch.ops.trtllm, "turboquant4_batch_attention"):
            pytest.skip("TurboQuant4 native torch ops are not available")

        q = torch.randn(1, 2, 128, device="cuda")
        codes = torch.empty(1, 2, 4, 1, 64, dtype=torch.uint8, device="cuda")
        scales = torch.empty(1, 2, 4, 1, 1, dtype=torch.float32, device="cuda")
        block_ids = torch.tensor([[1]], dtype=torch.int32, device="cuda")
        q_batch_indices = torch.tensor([0], dtype=torch.int32, device="cuda")
        query_positions = torch.tensor([0], dtype=torch.int32, device="cuda")
        seq_lens = torch.tensor([1], dtype=torch.int32, device="cuda")

        with pytest.raises(RuntimeError, match="batch attention block ids"):
            torch.ops.trtllm.turboquant4_batch_attention(
                q,
                codes,
                scales,
                block_ids,
                q_batch_indices,
                query_positions,
                seq_lens,
                1,
                4,
                1.0,
                True,
                0,
            )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_native_batch_attention_rejects_invalid_query_metadata(self):
        if not hasattr(torch.ops.trtllm, "turboquant4_batch_attention"):
            pytest.skip("TurboQuant4 native torch ops are not available")

        q = torch.randn(1, 2, 128, device="cuda")
        codes = torch.empty(1, 2, 4, 1, 64, dtype=torch.uint8, device="cuda")
        scales = torch.empty(1, 2, 4, 1, 1, dtype=torch.float32, device="cuda")
        block_ids = torch.tensor([[0]], dtype=torch.int32, device="cuda")
        seq_lens = torch.tensor([1], dtype=torch.int32, device="cuda")

        with pytest.raises(RuntimeError, match="batch indices"):
            torch.ops.trtllm.turboquant4_batch_attention(
                q,
                codes,
                scales,
                block_ids,
                torch.tensor([1], dtype=torch.int32, device="cuda"),
                torch.tensor([0], dtype=torch.int32, device="cuda"),
                seq_lens,
                1,
                4,
                1.0,
                True,
                0,
            )

        with pytest.raises(RuntimeError, match="query positions"):
            torch.ops.trtllm.turboquant4_batch_attention(
                q,
                codes,
                scales,
                block_ids,
                torch.tensor([0], dtype=torch.int32, device="cuda"),
                torch.tensor([1], dtype=torch.int32, device="cuda"),
                seq_lens,
                1,
                4,
                1.0,
                True,
                0,
            )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_native_batch_attention_allows_padded_ragged_block_ids(self):
        if not hasattr(torch.ops.trtllm, "turboquant4_batch_attention"):
            pytest.skip("TurboQuant4 native torch ops are not available")

        tokens_per_block = 4
        q = torch.randn(2, 2, 128, device="cuda")
        codes = torch.zeros(3, 2, tokens_per_block, 1, 64, dtype=torch.uint8, device="cuda")
        scales = torch.ones(3, 2, tokens_per_block, 1, 1, dtype=torch.float32, device="cuda")
        block_ids = torch.tensor([[0, 1], [2, -1]], dtype=torch.int32, device="cuda")
        q_batch_indices = torch.tensor([0, 1], dtype=torch.int32, device="cuda")
        query_positions = torch.tensor([4, 0], dtype=torch.int32, device="cuda")
        seq_lens = torch.tensor([5, 1], dtype=torch.int32, device="cuda")

        output = torch.ops.trtllm.turboquant4_batch_attention(
            q,
            codes,
            scales,
            block_ids,
            q_batch_indices,
            query_positions,
            seq_lens,
            5,
            tokens_per_block,
            1.0,
            True,
            0,
        )

        assert output.shape == q.shape
        assert torch.isfinite(output).all()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_native_non_tiled_attention_matches_python_reference(self):
        """Short decode should use the native non-tiled attention path."""
        if not hasattr(torch.ops.trtllm, "turboquant4_attention"):
            pytest.skip("TurboQuant4 native torch ops are not available")

        torch.manual_seed(0)
        tokens_per_block = 16
        seq_len = 64
        num_heads = 4
        num_kv_heads = 2
        head_dim = 128
        num_blocks = (seq_len + tokens_per_block - 1) // tokens_per_block
        block_ids = list(range(num_blocks))

        q = torch.randn(3, num_heads, head_dim, device="cuda")
        k = torch.randn(seq_len, num_kv_heads, head_dim, device="cuda")
        v = torch.randn(seq_len, num_kv_heads, head_dim, device="cuda")
        native_codes = torch.empty(
            num_blocks,
            2,
            tokens_per_block,
            num_kv_heads,
            head_dim // 2,
            dtype=torch.uint8,
            device="cuda",
        )
        native_scales = torch.empty(
            num_blocks,
            2,
            tokens_per_block,
            num_kv_heads,
            1,
            dtype=torch.float32,
            device="cuda",
        )
        turboquant4_update_cache(k, native_codes, native_scales, block_ids, 0, 0, tokens_per_block)
        turboquant4_update_cache(v, native_codes, native_scales, block_ids, 1, 0, tokens_per_block)
        native = turboquant4_attention(
            q,
            native_codes,
            native_scales,
            block_ids,
            seq_len,
            seq_len - q.shape[0],
            tokens_per_block,
            1.0,
            True,
            None,
        )

        ref_codes = torch.empty(
            num_blocks,
            2,
            tokens_per_block,
            num_kv_heads,
            head_dim // 2,
            dtype=torch.uint8,
        )
        ref_scales = torch.empty(num_blocks, 2, tokens_per_block, num_kv_heads, 1)
        turboquant4_update_cache(k.cpu(), ref_codes, ref_scales, block_ids, 0, 0, tokens_per_block)
        turboquant4_update_cache(v.cpu(), ref_codes, ref_scales, block_ids, 1, 0, tokens_per_block)
        expected = turboquant4_attention(
            q.cpu(),
            ref_codes,
            ref_scales,
            block_ids,
            seq_len,
            seq_len - q.shape[0],
            tokens_per_block,
            1.0,
            True,
            None,
        )

        torch.testing.assert_close(native.cpu(), expected, atol=1e-4, rtol=1e-4)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_native_tiled_attention_matches_python_reference(self):
        """Long decode should use the native tiled attention path."""
        if not hasattr(torch.ops.trtllm, "turboquant4_attention"):
            pytest.skip("TurboQuant4 native torch ops are not available")

        torch.manual_seed(0)
        tokens_per_block = 16
        seq_len = 260
        num_heads = 4
        num_kv_heads = 2
        head_dim = 128
        num_blocks = (seq_len + tokens_per_block - 1) // tokens_per_block
        block_ids = list(range(num_blocks))

        q = torch.randn(2, num_heads, head_dim, device="cuda")
        k = torch.randn(seq_len, num_kv_heads, head_dim, device="cuda")
        v = torch.randn(seq_len, num_kv_heads, head_dim, device="cuda")
        native_codes = torch.empty(
            num_blocks,
            2,
            tokens_per_block,
            num_kv_heads,
            head_dim // 2,
            dtype=torch.uint8,
            device="cuda",
        )
        native_scales = torch.empty(
            num_blocks,
            2,
            tokens_per_block,
            num_kv_heads,
            1,
            dtype=torch.float32,
            device="cuda",
        )
        turboquant4_update_cache(k, native_codes, native_scales, block_ids, 0, 0, tokens_per_block)
        turboquant4_update_cache(v, native_codes, native_scales, block_ids, 1, 0, tokens_per_block)
        native = turboquant4_attention(
            q,
            native_codes,
            native_scales,
            block_ids,
            seq_len,
            seq_len - q.shape[0],
            tokens_per_block,
            1.0,
            True,
            None,
        )

        ref_codes = torch.empty(
            num_blocks,
            2,
            tokens_per_block,
            num_kv_heads,
            head_dim // 2,
            dtype=torch.uint8,
        )
        ref_scales = torch.empty(num_blocks, 2, tokens_per_block, num_kv_heads, 1)
        turboquant4_update_cache(k.cpu(), ref_codes, ref_scales, block_ids, 0, 0, tokens_per_block)
        turboquant4_update_cache(v.cpu(), ref_codes, ref_scales, block_ids, 1, 0, tokens_per_block)
        expected = turboquant4_attention(
            q.cpu(),
            ref_codes,
            ref_scales,
            block_ids,
            seq_len,
            seq_len - q.shape[0],
            tokens_per_block,
            1.0,
            True,
            None,
        )

        torch.testing.assert_close(native.cpu(), expected, atol=1e-4, rtol=1e-4)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_native_non_tiled_batch_attention_matches_python_reference(self):
        """Short batched native attention should match the Python reference."""
        if not hasattr(torch.ops.trtllm, "turboquant4_batch_attention"):
            pytest.skip("TurboQuant4 native torch ops are not available")

        torch.manual_seed(0)
        tokens_per_block = 16
        num_heads = 4
        num_kv_heads = 2
        head_dim = 128
        seq_lens = [64, 32]
        q_lens = [2, 1]
        block_counts = [
            (seq_len + tokens_per_block - 1) // tokens_per_block for seq_len in seq_lens
        ]
        block_ids_per_request = [
            list(range(block_counts[0])),
            list(range(block_counts[0], block_counts[0] + block_counts[1])),
        ]
        num_blocks = sum(block_counts)

        native_codes = torch.empty(
            num_blocks,
            2,
            tokens_per_block,
            num_kv_heads,
            head_dim // 2,
            dtype=torch.uint8,
            device="cuda",
        )
        native_scales = torch.empty(
            num_blocks,
            2,
            tokens_per_block,
            num_kv_heads,
            1,
            dtype=torch.float32,
            device="cuda",
        )
        ref_codes = torch.empty(
            num_blocks, 2, tokens_per_block, num_kv_heads, head_dim // 2, dtype=torch.uint8
        )
        ref_scales = torch.empty(
            num_blocks, 2, tokens_per_block, num_kv_heads, 1, dtype=torch.float32
        )

        q_chunks = []
        q_batch_indices = []
        query_positions = []
        for batch_idx, (seq_len, q_len, block_ids) in enumerate(
            zip(seq_lens, q_lens, block_ids_per_request)
        ):
            k = torch.randn(seq_len, num_kv_heads, head_dim, device="cuda")
            v = torch.randn(seq_len, num_kv_heads, head_dim, device="cuda")
            turboquant4_update_cache(
                k, native_codes, native_scales, block_ids, 0, 0, tokens_per_block
            )
            turboquant4_update_cache(
                v, native_codes, native_scales, block_ids, 1, 0, tokens_per_block
            )
            turboquant4_update_cache(
                k.cpu(), ref_codes, ref_scales, block_ids, 0, 0, tokens_per_block
            )
            turboquant4_update_cache(
                v.cpu(), ref_codes, ref_scales, block_ids, 1, 0, tokens_per_block
            )

            q_chunks.append(torch.randn(q_len, num_heads, head_dim, device="cuda"))
            q_batch_indices.extend([batch_idx] * q_len)
            query_positions.extend(range(seq_len - q_len, seq_len))

        q = torch.cat(q_chunks, dim=0)
        native = turboquant4_batch_attention(
            q,
            native_codes,
            native_scales,
            block_ids_per_request,
            torch.tensor(q_batch_indices, dtype=torch.int32),
            torch.tensor(query_positions, dtype=torch.int32),
            torch.tensor(seq_lens, dtype=torch.int32),
            tokens_per_block,
            1.0,
            True,
            None,
        )
        expected = turboquant4_batch_attention(
            q.cpu(),
            ref_codes,
            ref_scales,
            block_ids_per_request,
            torch.tensor(q_batch_indices, dtype=torch.int32),
            torch.tensor(query_positions, dtype=torch.int32),
            torch.tensor(seq_lens, dtype=torch.int32),
            tokens_per_block,
            1.0,
            True,
            None,
        )

        torch.testing.assert_close(native.cpu(), expected, atol=1e-4, rtol=1e-4)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_native_batch_attention_matches_python_reference(self):
        """Batched native attention should match the Python reference implementation."""
        if not hasattr(torch.ops.trtllm, "turboquant4_batch_attention"):
            pytest.skip("TurboQuant4 native torch ops are not available")

        torch.manual_seed(0)
        tokens_per_block = 16
        num_heads = 4
        num_kv_heads = 2
        head_dim = 128
        seq_lens = [260, 130]
        q_lens = [2, 1]
        block_counts = [
            (seq_len + tokens_per_block - 1) // tokens_per_block for seq_len in seq_lens
        ]
        block_ids_per_request = [
            list(range(block_counts[0])),
            list(range(block_counts[0], block_counts[0] + block_counts[1])),
        ]
        num_blocks = sum(block_counts)

        native_codes = torch.empty(
            num_blocks,
            2,
            tokens_per_block,
            num_kv_heads,
            head_dim // 2,
            dtype=torch.uint8,
            device="cuda",
        )
        native_scales = torch.empty(
            num_blocks,
            2,
            tokens_per_block,
            num_kv_heads,
            1,
            dtype=torch.float32,
            device="cuda",
        )
        ref_codes = torch.empty(
            num_blocks, 2, tokens_per_block, num_kv_heads, head_dim // 2, dtype=torch.uint8
        )
        ref_scales = torch.empty(
            num_blocks, 2, tokens_per_block, num_kv_heads, 1, dtype=torch.float32
        )

        q_chunks = []
        q_batch_indices = []
        query_positions = []
        for batch_idx, (seq_len, q_len, block_ids) in enumerate(
            zip(seq_lens, q_lens, block_ids_per_request)
        ):
            k = torch.randn(seq_len, num_kv_heads, head_dim, device="cuda")
            v = torch.randn(seq_len, num_kv_heads, head_dim, device="cuda")
            turboquant4_update_cache(
                k, native_codes, native_scales, block_ids, 0, 0, tokens_per_block
            )
            turboquant4_update_cache(
                v, native_codes, native_scales, block_ids, 1, 0, tokens_per_block
            )
            turboquant4_update_cache(
                k.cpu(), ref_codes, ref_scales, block_ids, 0, 0, tokens_per_block
            )
            turboquant4_update_cache(
                v.cpu(), ref_codes, ref_scales, block_ids, 1, 0, tokens_per_block
            )

            q_chunks.append(torch.randn(q_len, num_heads, head_dim, device="cuda"))
            q_batch_indices.extend([batch_idx] * q_len)
            query_positions.extend(range(seq_len - q_len, seq_len))

        q = torch.cat(q_chunks, dim=0)
        native = turboquant4_batch_attention(
            q,
            native_codes,
            native_scales,
            block_ids_per_request,
            torch.tensor(q_batch_indices, dtype=torch.int32),
            torch.tensor(query_positions, dtype=torch.int32),
            torch.tensor(seq_lens, dtype=torch.int32),
            tokens_per_block,
            1.0,
            True,
            None,
        )
        expected = turboquant4_batch_attention(
            q.cpu(),
            ref_codes,
            ref_scales,
            block_ids_per_request,
            torch.tensor(q_batch_indices, dtype=torch.int32),
            torch.tensor(query_positions, dtype=torch.int32),
            torch.tensor(seq_lens, dtype=torch.int32),
            tokens_per_block,
            1.0,
            True,
            None,
        )

        torch.testing.assert_close(native.cpu(), expected, atol=1e-4, rtol=1e-4)

    def test_native_ops_have_fake_shapes(self):
        """TurboQuant4 native ops should expose fake/meta implementations."""
        if not hasattr(torch.ops.trtllm, "turboquant4_quantize"):
            pytest.skip("TurboQuant4 native torch ops are not available")

        fake_tensor = pytest.importorskip("torch._subclasses.fake_tensor")
        with fake_tensor.FakeTensorMode():
            x = torch.empty(2, 3, 128, device="cuda")
            codes, scales = torch.ops.trtllm.turboquant4_quantize(x)
            assert codes.shape == (2, 3, 64)
            assert codes.dtype == torch.uint8
            assert scales.shape == (2, 3, 1)
            assert scales.dtype == torch.float32

            dequantized = torch.ops.trtllm.turboquant4_dequantize(codes, scales, torch.float32)
            assert dequantized.shape == x.shape
            assert dequantized.dtype == torch.float32

            cache = torch.empty(5, 2, 4, 2, 64, dtype=torch.uint8, device="cuda")
            scale_cache = torch.empty(5, 2, 4, 2, 1, dtype=torch.float32, device="cuda")
            block_ids = torch.empty(2, 3, dtype=torch.int32, device="cuda")
            q = torch.empty(4, 4, 128, device="cuda")
            q_batch_indices = torch.empty(4, dtype=torch.int32, device="cuda")
            query_positions = torch.empty(4, dtype=torch.int32, device="cuda")
            seq_lens = torch.empty(2, dtype=torch.int32, device="cuda")

            cached = torch.ops.trtllm.turboquant4_dequantize_cache(
                cache, scale_cache, block_ids[0], 0, 7, 4, torch.float32
            )
            assert cached.shape == (7, 2, 128)
            assert cached.dtype == torch.float32

            single_out = torch.ops.trtllm.turboquant4_attention(
                q, cache, scale_cache, block_ids[0], 7, 3, 4, 1.0, True, 0
            )
            assert single_out.shape == q.shape
            with pytest.raises(RuntimeError, match="divisible"):
                torch.ops.trtllm.turboquant4_attention(
                    torch.empty(4, 3, 128, device="cuda"),
                    cache,
                    scale_cache,
                    block_ids[0],
                    7,
                    3,
                    4,
                    1.0,
                    True,
                    0,
                )

            batch_out = torch.ops.trtllm.turboquant4_batch_attention(
                q,
                cache,
                scale_cache,
                block_ids,
                q_batch_indices,
                query_positions,
                seq_lens,
                7,
                4,
                1.0,
                True,
                0,
            )
            assert batch_out.shape == q.shape
            with pytest.raises(RuntimeError, match="divisible"):
                torch.ops.trtllm.turboquant4_batch_attention(
                    torch.empty(4, 3, 128, device="cuda"),
                    cache,
                    scale_cache,
                    block_ids,
                    q_batch_indices,
                    query_positions,
                    seq_lens,
                    7,
                    4,
                    1.0,
                    True,
                    0,
                )

            with pytest.raises(RuntimeError, match="even"):
                torch.ops.trtllm.turboquant4_quantize(torch.empty(1, 127, device="cuda"))
            with pytest.raises(NotImplementedError, match="FP16"):
                torch.ops.trtllm.turboquant4_dequantize(codes, scales, torch.int32)
            with pytest.raises(RuntimeError, match="shape"):
                torch.ops.trtllm.turboquant4_dequantize_cache(
                    torch.empty(5, 2, 4, dtype=torch.uint8, device="cuda"),
                    scale_cache,
                    block_ids[0],
                    0,
                    7,
                    4,
                    torch.float32,
                )
            with pytest.raises(RuntimeError, match="2D tensor"):
                torch.ops.trtllm.turboquant4_batch_attention(
                    q,
                    cache,
                    scale_cache,
                    block_ids[0],
                    q_batch_indices,
                    query_positions,
                    seq_lens,
                    7,
                    4,
                    1.0,
                    True,
                    0,
                )
            with pytest.raises(RuntimeError, match="kv_index"):
                torch.ops.trtllm.turboquant4_update_cache(
                    torch.empty(1, 2, 128, device="cuda"),
                    cache,
                    scale_cache,
                    block_ids[0],
                    2,
                    0,
                    4,
                )
            with pytest.raises(RuntimeError, match="shorter"):
                torch.ops.trtllm.turboquant4_update_cache(
                    torch.empty(5, 2, 128, device="cuda"),
                    cache,
                    scale_cache,
                    block_ids[0, :1],
                    0,
                    3,
                    4,
                )
            with pytest.raises(RuntimeError, match="tokens_per_block"):
                torch.ops.trtllm.turboquant4_attention(
                    q, cache, scale_cache, block_ids[0], 7, 3, 8, 1.0, True, 0
                )
            with pytest.raises(RuntimeError, match="shorter"):
                torch.ops.trtllm.turboquant4_dequantize_cache(
                    cache,
                    scale_cache,
                    block_ids[0, :1],
                    0,
                    7,
                    4,
                    torch.float32,
                )
            with pytest.raises(RuntimeError, match="within the KV sequence"):
                torch.ops.trtllm.turboquant4_attention(
                    q, cache, scale_cache, block_ids[0], 7, 4, 4, 1.0, True, 0
                )

    def test_kv_shaped_input(self):
        """Should work with typical KV cache tensor shapes."""
        # [batch, seq_len, num_kv_heads, head_dim]
        x = torch.randn(1, 32, 8, 128) * 0.1
        result = turboquant4_quantize_dequantize(x)
        assert result.shape == x.shape

    def test_non_128_head_dim_round_trip(self):
        """Centroids should scale to non-128 power-of-2 head dimensions."""
        torch.manual_seed(42)
        x = torch.randn(64, 64) * 0.1
        result = turboquant4_quantize_dequantize(x)

        mse = (x - result).pow(2).mean().item()
        rel_mse = mse / x.pow(2).mean().item()
        assert rel_mse < 0.05, f"Relative MSE {rel_mse:.4f} too high"


class TestPackedCache:
    def test_packed_shape_and_dtype(self):
        """Packed representation should store two 4-bit codes per byte."""
        x = torch.randn(2, 3, 4, 128, dtype=torch.bfloat16)

        nibbles, scales = turboquant4_quantize(x)

        assert nibbles.shape == (2, 3, 4, 64)
        assert nibbles.dtype == torch.uint8
        assert scales.shape == (2, 3, 4, 1)
        assert scales.dtype == torch.float32
        assert ((nibbles & 0x0F) <= 15).all()
        assert (((nibbles >> 4) & 0x0F) <= 15).all()

    def test_packed_round_trip_matches_qdq(self):
        """The packed path should match the convenience QDQ helper exactly."""
        x = torch.randn(4, 8, 128, dtype=torch.bfloat16)

        nibbles, scales = turboquant4_quantize(x)
        packed_result = turboquant4_dequantize(nibbles, scales, dtype=x.dtype)
        qdq_result = turboquant4_quantize_dequantize(x)

        torch.testing.assert_close(packed_result, qdq_result)

    def test_odd_head_dim_raises(self):
        """Packing requires an even head dimension."""
        x = torch.randn(2, 3, 7)

        with pytest.raises(ValueError, match="even head_dim"):
            turboquant4_quantize(x)

    def test_incremental_packed_cache_update(self):
        """Packed cache writes should compose across prefill and decode steps."""
        num_blocks = 1
        kv_factor = 2
        tokens_per_block = 8
        num_kv_heads = 2
        head_dim = 128
        cache_shape = (
            num_blocks,
            kv_factor,
            tokens_per_block,
            num_kv_heads,
            head_dim,
        )
        codes = torch.zeros((*cache_shape[:-1], head_dim // 2), dtype=torch.uint8)
        scales = torch.zeros((*cache_shape[:-1], 1), dtype=torch.float32)

        prefill = torch.randn(1, 4, num_kv_heads, head_dim)
        decode = torch.randn(1, 1, num_kv_heads, head_dim)

        prefill_codes, prefill_scales = turboquant4_quantize(prefill)
        decode_codes, decode_scales = turboquant4_quantize(decode)

        cache_position = torch.arange(0, 4)
        codes[0, 0].unsqueeze(0).index_copy_(1, cache_position, prefill_codes)
        scales[0, 0].unsqueeze(0).index_copy_(1, cache_position, prefill_scales)

        cache_position = torch.arange(4, 5)
        codes[0, 0].unsqueeze(0).index_copy_(1, cache_position, decode_codes)
        scales[0, 0].unsqueeze(0).index_copy_(1, cache_position, decode_scales)

        cached = turboquant4_dequantize(
            codes[0, 0].unsqueeze(0)[:, :5],
            scales[0, 0].unsqueeze(0)[:, :5],
            dtype=prefill.dtype,
        )
        expected = turboquant4_quantize_dequantize(torch.cat([prefill, decode], dim=1))

        torch.testing.assert_close(cached, expected)

    def test_cache_helpers_write_and_read_paged_blocks(self):
        """Cache helpers should write across block boundaries and dequantize back."""
        tokens_per_block = 3
        num_kv_heads = 2
        head_dim = 128
        codes = torch.zeros(3, 2, tokens_per_block, num_kv_heads, head_dim // 2, dtype=torch.uint8)
        scales = torch.zeros(3, 2, tokens_per_block, num_kv_heads, 1, dtype=torch.float32)
        block_ids = [0, 2]
        x = torch.randn(4, num_kv_heads, head_dim)

        turboquant4_update_cache(
            x,
            codes,
            scales,
            block_ids,
            kv_index=1,
            start_pos=2,
            tokens_per_block=tokens_per_block,
        )
        result = turboquant4_dequantize_cache(
            codes,
            scales,
            block_ids,
            kv_index=1,
            seq_len=6,
            tokens_per_block=tokens_per_block,
            dtype=x.dtype,
        )

        expected_codes, expected_scales = turboquant4_quantize(x.unsqueeze(0))
        torch.testing.assert_close(codes[0, 1, 2:], expected_codes[0, :1])
        torch.testing.assert_close(codes[2, 1, :3], expected_codes[0, 1:])
        torch.testing.assert_close(scales[0, 1, 2:], expected_scales[0, :1])
        torch.testing.assert_close(scales[2, 1, :3], expected_scales[0, 1:])
        torch.testing.assert_close(
            result[2:],
            turboquant4_quantize_dequantize(x.unsqueeze(0)).squeeze(0))

    def test_cache_helpers_validate_shape_contract(self):
        codes = torch.zeros(1, 2, 4, 2, 64, dtype=torch.uint8)
        scales = torch.zeros(1, 2, 4, 2, 2, dtype=torch.float32)
        x = torch.randn(1, 2, 128)

        with pytest.raises(ValueError, match="last dimension must be 1"):
            turboquant4_update_cache(
                x,
                codes,
                scales,
                [0],
                kv_index=0,
                start_pos=0,
                tokens_per_block=4,
            )

    def test_cache_helpers_reject_invalid_dtypes(self):
        codes = torch.zeros(1, 2, 4, 2, 64, dtype=torch.uint8)
        scales = torch.zeros(1, 2, 4, 2, 1, dtype=torch.float32)
        x = torch.randn(1, 2, 128)

        with pytest.raises(ValueError, match="cache must be uint8"):
            turboquant4_update_cache(
                x,
                codes.to(torch.int8),
                scales,
                [0],
                kv_index=0,
                start_pos=0,
                tokens_per_block=4,
            )

        with pytest.raises(ValueError, match="scale cache must be float32"):
            turboquant4_update_cache(
                x,
                codes,
                scales.to(torch.float16),
                [0],
                kv_index=0,
                start_pos=0,
                tokens_per_block=4,
            )

        with pytest.raises(NotImplementedError, match="FP16"):
            turboquant4_update_cache(
                x.to(torch.int32),
                codes,
                scales,
                [0],
                kv_index=0,
                start_pos=0,
                tokens_per_block=4,
            )

        with pytest.raises(ValueError, match="block ids must be int32"):
            turboquant4_update_cache(
                x,
                codes,
                scales,
                torch.tensor([0], dtype=torch.int64),
                kv_index=0,
                start_pos=0,
                tokens_per_block=4,
            )

    def test_cache_helpers_reject_mismatched_dense_head_dim(self):
        codes = torch.zeros(1, 2, 4, 2, 64, dtype=torch.uint8)
        scales = torch.zeros(1, 2, 4, 2, 1, dtype=torch.float32)
        x = torch.randn(1, 2, 129)

        with pytest.raises(ValueError, match="head_dim mismatch"):
            turboquant4_update_cache(
                x,
                codes,
                scales,
                [0],
                kv_index=0,
                start_pos=0,
                tokens_per_block=4,
            )

    def test_cache_helpers_reject_invalid_block_ids(self):
        codes = torch.zeros(1, 2, 4, 2, 64, dtype=torch.uint8)
        scales = torch.zeros(1, 2, 4, 2, 1, dtype=torch.float32)
        x = torch.randn(1, 2, 128)

        with pytest.raises(RuntimeError, match="shorter"):
            turboquant4_update_cache(
                x,
                codes,
                scales,
                [],
                kv_index=0,
                start_pos=0,
                tokens_per_block=4,
            )

        with pytest.raises(RuntimeError, match="out of range"):
            turboquant4_update_cache(
                x,
                codes,
                scales,
                [1],
                kv_index=0,
                start_pos=0,
                tokens_per_block=4,
            )

        with pytest.raises(ValueError, match="1D tensor"):
            turboquant4_dequantize_cache(
                codes,
                scales,
                torch.tensor([[0]], dtype=torch.int32),
                kv_index=0,
                seq_len=1,
                tokens_per_block=4,
                dtype=x.dtype,
            )

    @pytest.mark.parametrize("block_id", [-1, 1])
    def test_cache_helpers_reject_invalid_cuda_block_ids(self, block_id):
        if not torch.cuda.is_available():
            pytest.skip("CUDA is required for CUDA block-id validation")
        codes = torch.zeros(1, 2, 4, 2, 64, dtype=torch.uint8)
        scales = torch.zeros(1, 2, 4, 2, 1, dtype=torch.float32)
        x = torch.randn(1, 2, 128)
        block_ids = torch.tensor([block_id], dtype=torch.int32, device="cuda")

        with pytest.raises(RuntimeError, match="out of range"):
            turboquant4_update_cache(
                x,
                codes,
                scales,
                block_ids,
                kv_index=0,
                start_pos=0,
                tokens_per_block=4,
            )

    @pytest.mark.parametrize("block_id", [-1, 1])
    def test_batch_attention_rejects_invalid_cuda_block_ids(self, block_id):
        if not torch.cuda.is_available():
            pytest.skip("CUDA is required for CUDA block-id validation")
        cache = torch.zeros(1, 2, 4, 2, 64, dtype=torch.uint8)
        scale_cache = torch.zeros(1, 2, 4, 2, 1, dtype=torch.float32)
        q = torch.randn(1, 2, 128)
        block_ids = torch.tensor([[block_id]], dtype=torch.int32, device="cuda")

        with pytest.raises(RuntimeError, match="out of range"):
            turboquant4_batch_attention(
                q,
                cache,
                scale_cache,
                block_ids,
                torch.tensor([0], dtype=torch.int32),
                torch.tensor([0], dtype=torch.int32),
                torch.tensor([1], dtype=torch.int32),
                tokens_per_block=4,
                q_scaling=1.0,
                is_causal=True,
                attention_window_size=None,
            )

    def test_cache_helpers_reject_invalid_positions(self):
        codes = torch.zeros(1, 2, 4, 2, 64, dtype=torch.uint8)
        scales = torch.zeros(1, 2, 4, 2, 1, dtype=torch.float32)
        x = torch.randn(1, 2, 128)

        with pytest.raises(ValueError, match="start_pos"):
            turboquant4_update_cache(
                x,
                codes,
                scales,
                [0],
                kv_index=0,
                start_pos=-1,
                tokens_per_block=4,
            )

        with pytest.raises(ValueError, match="seq_len"):
            turboquant4_dequantize_cache(
                codes,
                scales,
                [0],
                kv_index=0,
                seq_len=-1,
                tokens_per_block=4,
                dtype=x.dtype,
            )

        with pytest.raises(ValueError, match="q_scaling"):
            turboquant4_attention(
                x,
                codes,
                scales,
                [0],
                seq_len=1,
                q_start_pos=0,
                tokens_per_block=4,
                q_scaling=0.0,
                is_causal=True,
                attention_window_size=None,
            )

        with pytest.raises(ValueError, match="query positions"):
            turboquant4_attention(
                x,
                codes,
                scales,
                [0],
                seq_len=1,
                q_start_pos=1,
                tokens_per_block=4,
                q_scaling=1.0,
                is_causal=True,
                attention_window_size=None,
            )

    def test_attention_helper_matches_sdpa_reference(self):
        tokens_per_block = 4
        num_heads = 4
        num_kv_heads = 2
        head_dim = 128
        codes = torch.zeros(1, 2, tokens_per_block, num_kv_heads, head_dim // 2, dtype=torch.uint8)
        scales = torch.zeros(1, 2, tokens_per_block, num_kv_heads, 1, dtype=torch.float32)
        block_ids = [0]
        k = torch.randn(3, num_kv_heads, head_dim)
        v = torch.randn(3, num_kv_heads, head_dim)
        q = torch.randn(3, num_heads, head_dim)
        turboquant4_update_cache(k, codes, scales, block_ids, 0, 0, tokens_per_block)
        turboquant4_update_cache(v, codes, scales, block_ids, 1, 0, tokens_per_block)

        result = turboquant4_attention(
            q,
            codes,
            scales,
            block_ids,
            seq_len=3,
            q_start_pos=0,
            tokens_per_block=tokens_per_block,
            q_scaling=1.0,
            is_causal=True,
            attention_window_size=None,
        )

        key_states = turboquant4_dequantize_cache(
            codes, scales, block_ids, 0, 3, tokens_per_block, q.dtype
        ).unsqueeze(0).transpose(1, 2)
        value_states = turboquant4_dequantize_cache(
            codes, scales, block_ids, 1, 3, tokens_per_block, q.dtype
        ).unsqueeze(0).transpose(1, 2)
        key_states = repeat_kv(key_states, num_heads // num_kv_heads)
        value_states = repeat_kv(value_states, num_heads // num_kv_heads)
        expected = (
            torch.nn.functional.scaled_dot_product_attention(
                q.unsqueeze(0).transpose(1, 2),
                key_states,
                value_states,
                is_causal=True,
                scale=1 / (head_dim**0.5),
            )
            .transpose(1, 2)
            .squeeze(0)
        )
        torch.testing.assert_close(result, expected)

    def test_attention_helper_matches_sliding_window_decode_reference(self):
        tokens_per_block = 4
        num_heads = 4
        num_kv_heads = 2
        head_dim = 128
        seq_len = 6
        codes = torch.zeros(2, 2, tokens_per_block, num_kv_heads, head_dim // 2, dtype=torch.uint8)
        scales = torch.zeros(2, 2, tokens_per_block, num_kv_heads, 1, dtype=torch.float32)
        block_ids = [0, 1]
        k = torch.randn(seq_len, num_kv_heads, head_dim)
        v = torch.randn(seq_len, num_kv_heads, head_dim)
        q = torch.randn(1, num_heads, head_dim)
        turboquant4_update_cache(k, codes, scales, block_ids, 0, 0, tokens_per_block)
        turboquant4_update_cache(v, codes, scales, block_ids, 1, 0, tokens_per_block)

        result = turboquant4_attention(
            q,
            codes,
            scales,
            block_ids,
            seq_len=seq_len,
            q_start_pos=5,
            tokens_per_block=tokens_per_block,
            q_scaling=1.0,
            is_causal=False,
            attention_window_size=3,
        )

        key_states = turboquant4_dequantize_cache(
            codes, scales, block_ids, 0, seq_len, tokens_per_block, q.dtype
        ).unsqueeze(0).transpose(1, 2)
        value_states = turboquant4_dequantize_cache(
            codes, scales, block_ids, 1, seq_len, tokens_per_block, q.dtype
        ).unsqueeze(0).transpose(1, 2)
        key_states = repeat_kv(key_states, num_heads // num_kv_heads)
        value_states = repeat_kv(value_states, num_heads // num_kv_heads)
        allowed = torch.tensor([[[[False, False, False, True, True, True]]]])
        expected = (
            torch.nn.functional.scaled_dot_product_attention(
                q.unsqueeze(0).transpose(1, 2),
                key_states,
                value_states,
                attn_mask=allowed,
                scale=1 / (head_dim**0.5),
            )
            .transpose(1, 2)
            .squeeze(0)
        )
        torch.testing.assert_close(result, expected)

    def test_attention_helpers_reject_empty_head_counts(self):
        tokens_per_block = 4
        head_dim = 128
        q = torch.randn(1, 4, head_dim)
        no_kv_head_codes = torch.zeros(1, 2, tokens_per_block, 0, head_dim // 2, dtype=torch.uint8)
        no_kv_head_scales = torch.zeros(1, 2, tokens_per_block, 0, 1, dtype=torch.float32)

        with pytest.raises(ValueError, match="at least one KV head"):
            turboquant4_attention(
                q,
                no_kv_head_codes,
                no_kv_head_scales,
                [0],
                seq_len=1,
                q_start_pos=0,
                tokens_per_block=tokens_per_block,
                q_scaling=1.0,
                is_causal=True,
                attention_window_size=None,
            )

        one_kv_head_codes = torch.zeros(1, 2, tokens_per_block, 1, head_dim // 2, dtype=torch.uint8)
        one_kv_head_scales = torch.zeros(1, 2, tokens_per_block, 1, 1, dtype=torch.float32)
        with pytest.raises(ValueError, match="at least one query head"):
            turboquant4_batch_attention(
                torch.randn(1, 0, head_dim),
                one_kv_head_codes,
                one_kv_head_scales,
                [[0]],
                torch.tensor([0], dtype=torch.int32),
                torch.tensor([0], dtype=torch.int32),
                torch.tensor([1], dtype=torch.int32),
                tokens_per_block,
                1.0,
                True,
                None,
            )

    def test_batch_attention_helper_matches_single_request_reference(self):
        tokens_per_block = 4
        num_heads = 4
        num_kv_heads = 2
        head_dim = 128
        codes = torch.zeros(3, 2, tokens_per_block, num_kv_heads, head_dim // 2, dtype=torch.uint8)
        scales = torch.zeros(3, 2, tokens_per_block, num_kv_heads, 1, dtype=torch.float32)
        block_ids_per_request = [[0, 2], [1]]

        k0 = torch.randn(5, num_kv_heads, head_dim)
        v0 = torch.randn(5, num_kv_heads, head_dim)
        q0 = torch.randn(2, num_heads, head_dim)
        k1 = torch.randn(3, num_kv_heads, head_dim)
        v1 = torch.randn(3, num_kv_heads, head_dim)
        q1 = torch.randn(3, num_heads, head_dim)
        turboquant4_update_cache(
            k0, codes, scales, block_ids_per_request[0], 0, 0, tokens_per_block
        )
        turboquant4_update_cache(
            v0, codes, scales, block_ids_per_request[0], 1, 0, tokens_per_block
        )
        turboquant4_update_cache(
            k1, codes, scales, block_ids_per_request[1], 0, 0, tokens_per_block
        )
        turboquant4_update_cache(
            v1, codes, scales, block_ids_per_request[1], 1, 0, tokens_per_block
        )

        q = torch.cat([q0, q1], dim=0)
        result = turboquant4_batch_attention(
            q,
            codes,
            scales,
            block_ids_per_request,
            torch.tensor([0, 0, 1, 1, 1], dtype=torch.int32),
            torch.tensor([4, 3, 0, 1, 2], dtype=torch.int32),
            torch.tensor([5, 3], dtype=torch.int32),
            tokens_per_block,
            1.0,
            True,
            None,
        )

        expected = torch.cat(
            [
                turboquant4_attention(
                    q0[:1],
                    codes,
                    scales,
                    block_ids_per_request[0],
                    5,
                    4,
                    tokens_per_block,
                    1.0,
                    True,
                    None,
                ),
                turboquant4_attention(
                    q0[1:],
                    codes,
                    scales,
                    block_ids_per_request[0],
                    5,
                    3,
                    tokens_per_block,
                    1.0,
                    True,
                    None,
                ),
                turboquant4_attention(
                    q1,
                    codes,
                    scales,
                    block_ids_per_request[1],
                    3,
                    0,
                    tokens_per_block,
                    1.0,
                    True,
                    None,
                ),
            ],
            dim=0,
        )
        torch.testing.assert_close(result, expected)

    def test_batch_attention_rejects_invalid_required_block_id(self):
        tokens_per_block = 4
        num_heads = 4
        num_kv_heads = 2
        head_dim = 128
        codes = torch.zeros(3, 2, tokens_per_block, num_kv_heads, head_dim // 2, dtype=torch.uint8)
        scales = torch.zeros(3, 2, tokens_per_block, num_kv_heads, 1, dtype=torch.float32)
        q = torch.randn(3, num_heads, head_dim)

        with pytest.raises(RuntimeError, match="batch 0"):
            turboquant4_batch_attention(
                q,
                codes,
                scales,
                [[0, -1], [1]],
                torch.tensor([0, 0, 1], dtype=torch.int32),
                torch.tensor([0, 4, 0], dtype=torch.int32),
                torch.tensor([5, 3], dtype=torch.int32),
                tokens_per_block,
                1.0,
                True,
                None,
            )

        with pytest.raises(RuntimeError, match="query position"):
            turboquant4_batch_attention(
                q[:1],
                codes,
                scales,
                [[0], [1]],
                torch.tensor([0], dtype=torch.int32),
                torch.tensor([3], dtype=torch.int32),
                torch.tensor([3, 3], dtype=torch.int32),
                tokens_per_block,
                1.0,
                True,
                None,
            )

    @pytest.mark.parametrize(
        ("batch_idx", "query_position", "match"),
        [
            (-1, 0, "batch index"),
            (2, 0, "batch index"),
            (0, -1, "negative"),
            (0, 3, "query position"),
        ],
    )
    def test_batch_attention_rejects_invalid_cuda_query_metadata(
        self, batch_idx, query_position, match
    ):
        if not torch.cuda.is_available():
            pytest.skip("CUDA is required for CUDA query metadata validation")
        tokens_per_block = 4
        num_heads = 4
        num_kv_heads = 2
        head_dim = 128
        codes = torch.zeros(1, 2, tokens_per_block, num_kv_heads, head_dim // 2, dtype=torch.uint8)
        scales = torch.zeros(1, 2, tokens_per_block, num_kv_heads, 1, dtype=torch.float32)
        q = torch.randn(1, num_heads, head_dim)

        with pytest.raises(RuntimeError, match=match):
            turboquant4_batch_attention(
                q,
                codes,
                scales,
                [[0], [0]],
                torch.tensor([batch_idx], dtype=torch.int32, device="cuda"),
                torch.tensor([query_position], dtype=torch.int32, device="cuda"),
                torch.tensor([3, 3], dtype=torch.int32),
                tokens_per_block,
                1.0,
                True,
                None,
            )

    def test_batch_attention_rejects_invalid_seq_lens(self):
        tokens_per_block = 4
        num_heads = 4
        num_kv_heads = 2
        head_dim = 128
        codes = torch.zeros(2, 2, tokens_per_block, num_kv_heads, head_dim // 2, dtype=torch.uint8)
        scales = torch.zeros(2, 2, tokens_per_block, num_kv_heads, 1, dtype=torch.float32)
        q = torch.randn(1, num_heads, head_dim)

        with pytest.raises(ValueError, match="seq_lens must be non-negative"):
            turboquant4_batch_attention(
                q,
                codes,
                scales,
                [[0], [1]],
                torch.tensor([0], dtype=torch.int32),
                torch.tensor([0], dtype=torch.int32),
                torch.tensor([1, -1], dtype=torch.int32),
                tokens_per_block,
                1.0,
                True,
                None,
            )

        with pytest.raises(ValueError, match="max_seq_len"):
            turboquant4_batch_attention(
                q,
                codes,
                scales,
                [[0], [1]],
                torch.tensor([0], dtype=torch.int32),
                torch.tensor([0], dtype=torch.int32),
                torch.tensor([5, 1], dtype=torch.int32),
                tokens_per_block,
                1.0,
                True,
                None,
                max_seq_len=4,
            )

    def test_batch_attention_rejects_non_int32_metadata(self):
        tokens_per_block = 4
        num_heads = 4
        num_kv_heads = 2
        head_dim = 128
        codes = torch.zeros(1, 2, tokens_per_block, num_kv_heads, head_dim // 2, dtype=torch.uint8)
        scales = torch.zeros(1, 2, tokens_per_block, num_kv_heads, 1, dtype=torch.float32)
        q = torch.randn(1, num_heads, head_dim)

        with pytest.raises(ValueError, match="q_batch_indices must be int32"):
            turboquant4_batch_attention(
                q,
                codes,
                scales,
                [[0]],
                torch.tensor([0], dtype=torch.int64),
                torch.tensor([0], dtype=torch.int32),
                torch.tensor([1], dtype=torch.int32),
                tokens_per_block,
                1.0,
                True,
                None,
            )

        with pytest.raises(ValueError, match="query_positions must be int32"):
            turboquant4_batch_attention(
                q,
                codes,
                scales,
                [[0]],
                torch.tensor([0], dtype=torch.int32),
                torch.tensor([0], dtype=torch.int64),
                torch.tensor([1], dtype=torch.int32),
                tokens_per_block,
                1.0,
                True,
                None,
            )

        with pytest.raises(ValueError, match="seq_lens must be int32"):
            turboquant4_batch_attention(
                q,
                codes,
                scales,
                [[0]],
                torch.tensor([0], dtype=torch.int32),
                torch.tensor([0], dtype=torch.int32),
                torch.tensor([1], dtype=torch.int64),
                tokens_per_block,
                1.0,
                True,
                None,
            )


class TestVanillaPackedCache:
    def test_vanilla_update_uses_native_packed_cache(self):
        """VANILLA TurboQuant4 should preserve dense K and quantize V."""

        num_kv_heads = 2
        head_dim = 128
        key_cache = torch.empty(1, 8, num_kv_heads, head_dim)
        value_cache = torch.empty(1, 8, num_kv_heads, head_dim // 2, dtype=torch.uint8)
        value_scales = torch.empty(1, 8, num_kv_heads, 1)
        attention = VanillaAttention(
            layer_idx=0,
            num_heads=4,
            head_dim=head_dim,
            num_kv_heads=num_kv_heads,
            quant_config=QuantConfig(kv_cache_quant_algo=QuantAlgo.TURBOQUANT4),
        )

        prefill_k = torch.randn(1, 4, num_kv_heads, head_dim)
        prefill_v = torch.randn(1, 4, num_kv_heads, head_dim)
        attention._single_request_update_turboquant4_kv_cache(
            prefill_k,
            prefill_v,
            (key_cache, value_cache),
            value_scales,
            past_seen_token=0,
            kv_len=4,
            cache_idx=([0], [0], [0]),
            dtype=prefill_k.dtype,
        )

        decode_k = torch.randn(1, 1, num_kv_heads, head_dim)
        decode_v = torch.randn(1, 1, num_kv_heads, head_dim)
        key_states, value_states = attention._single_request_update_turboquant4_kv_cache(
            decode_k,
            decode_v,
            (key_cache, value_cache),
            value_scales,
            past_seen_token=4,
            kv_len=1,
            cache_idx=([0], [0], [0]),
            dtype=decode_k.dtype,
        )

        assert key_cache.dtype == torch.float32
        assert key_cache.shape[-1] == head_dim
        assert value_cache.dtype == torch.uint8
        assert value_cache.shape[-1] == head_dim // 2
        assert value_scales.dtype == torch.float32
        assert value_scales.shape[-1] == 1

        expected_k = torch.cat([prefill_k, decode_k], dim=1)
        expected_v = torch.cat(
            [turboquant4_quantize_dequantize(prefill_v), decode_v], dim=1)
        torch.testing.assert_close(key_states, expected_k)
        torch.testing.assert_close(value_states, expected_v)

    def test_vanilla_update_uses_all_paged_blocks(self):
        num_kv_heads = 2
        head_dim = 128
        tokens_per_block = 4
        block_ids = [0, 1]
        key_cache = torch.empty(2, tokens_per_block, num_kv_heads, head_dim)
        value_cache = torch.empty(2, tokens_per_block, num_kv_heads, head_dim // 2, dtype=torch.uint8)
        value_scales = torch.empty(2, tokens_per_block, num_kv_heads, 1)
        attention = VanillaAttention(
            layer_idx=0,
            num_heads=4,
            head_dim=head_dim,
            num_kv_heads=num_kv_heads,
            quant_config=QuantConfig(kv_cache_quant_algo=QuantAlgo.TURBOQUANT4),
        )

        prefill_k = torch.randn(1, 6, num_kv_heads, head_dim)
        prefill_v = torch.randn(1, 6, num_kv_heads, head_dim)
        key_states, value_states = attention._single_request_update_turboquant4_kv_cache(
            prefill_k,
            prefill_v,
            (key_cache, value_cache),
            value_scales,
            past_seen_token=0,
            kv_len=6,
            cache_idx=(block_ids, block_ids, block_ids),
            dtype=prefill_k.dtype,
        )

        expected_k = prefill_k
        expected_v = prefill_v
        torch.testing.assert_close(key_states, expected_k)
        torch.testing.assert_close(value_states, expected_v)

    def test_vanilla_forward_uses_asymmetric_cache_buffers(self):
        num_heads = 4
        num_kv_heads = 2
        head_dim = 128
        q_len = 2
        tokens_per_block = 4
        key_cache = torch.empty(1, tokens_per_block, num_kv_heads, head_dim)
        value_cache = torch.empty(1,
                                  tokens_per_block,
                                  num_kv_heads,
                                  head_dim // 2,
                                  dtype=torch.uint8)
        value_scales = torch.empty(1, tokens_per_block, num_kv_heads, 1)
        manager = _FakeTurboQuant4KVCacheManager(
            key_cache, value_cache, value_scales, [0])
        attention = VanillaAttention(
            layer_idx=0,
            num_heads=num_heads,
            head_dim=head_dim,
            num_kv_heads=num_kv_heads,
            quant_config=QuantConfig(kv_cache_quant_algo=QuantAlgo.TURBOQUANT4),
        )
        q = torch.randn(q_len, num_heads * head_dim)
        k = torch.randn(q_len, num_kv_heads * head_dim)
        v = torch.randn(q_len, num_kv_heads * head_dim)
        metadata = SimpleNamespace(
            kv_cache_manager=manager,
            kv_cache_params=SimpleNamespace(num_cached_tokens_per_seq=[0]),
            block_ids_per_seq=[[0]],
            seq_lens=torch.tensor([q_len], dtype=torch.int32),
            seq_lens_kv=torch.tensor([q_len], dtype=torch.int32),
            request_ids=[11],
        )

        output = attention.forward(q, k, v, metadata)

        assert output.shape == (q_len, num_heads * head_dim)
        expected_k = k.view(1, q_len, num_kv_heads, head_dim).squeeze(0)
        torch.testing.assert_close(key_cache[0, :q_len], expected_k)

    def test_vanilla_forward_uses_layer_specific_block_ids(self):
        num_heads = 4
        head_dim = 128
        manager = _LayerAwareFakeTurboQuant4KVCacheManager()
        attention = VanillaAttention(
            layer_idx=1,
            num_heads=num_heads,
            head_dim=head_dim,
            num_kv_heads=2,
            quant_config=QuantConfig(kv_cache_quant_algo=QuantAlgo.TURBOQUANT4),
        )
        seen_cache_indices = []

        def fake_single_request_forward(
            q,
            k,
            v,
            attention_mask,
            kv_cache_tensor,
            kv_cache_scales,
            past_seen_token,
            cache_idx,
            sample_idx,
            metadata,
            attention_window_size=None,
            **kwargs,
        ):
            seen_cache_indices.append(cache_idx)
            return torch.zeros(num_heads, q.size(0), head_dim)

        attention._single_request_forward = fake_single_request_forward
        metadata = SimpleNamespace(
            kv_cache_manager=manager,
            kv_cache_params=SimpleNamespace(num_cached_tokens_per_seq=[0, 0]),
            block_ids_per_seq=[[0], [1]],
            seq_lens=torch.tensor([1, 1], dtype=torch.int32),
            seq_lens_kv=torch.tensor([1, 1], dtype=torch.int32),
            request_ids=[11, 12],
        )

        output = attention.forward(
            torch.zeros(2, num_heads * head_dim),
            torch.zeros(2, 2 * head_dim),
            torch.zeros(2, 2 * head_dim),
            metadata,
        )

        assert manager.calls == [
            ((11, 12), 1),
            ((11, 12), 1),
            ((11, 12), 1),
        ]
        assert seen_cache_indices == [
            ([110], [110], [110]),
            ([111], [111], [111]),
        ]
        assert output.shape == (2, num_heads * head_dim)


class _FakeTurboQuant4KVCacheManager:
    def __init__(self, key_cache, value_cache, value_scales, block_ids):
        self.key_cache = key_cache
        self.value_cache = value_cache
        self.value_scales = value_scales
        self.block_ids = block_ids

    def get_turboquant4_key_buffers(self, layer_idx):
        return self.key_cache

    def get_turboquant4_value_buffers(self, layer_idx):
        return self.value_cache

    def get_turboquant4_value_scale_buffers(self, layer_idx):
        return self.value_scales

    def get_batch_cache_indices_for_role(self, request_ids, layer_idx, data_role):
        del data_role
        if self.block_ids and isinstance(self.block_ids[0], list):
            return self.block_ids
        return [self.block_ids for _ in request_ids]


class _LayerAwareFakeTurboQuant4KVCacheManager:
    def __init__(self):
        self.calls = []

    def get_turboquant4_key_buffers(self, layer_idx):
        return torch.empty(2, 4, 2, 128)

    def get_turboquant4_value_buffers(self, layer_idx):
        return torch.empty(2, 4, 2, 64, dtype=torch.uint8)

    def get_turboquant4_value_scale_buffers(self, layer_idx):
        return torch.empty(2, 4, 2, 1)

    def get_batch_cache_indices_for_role(self, request_ids, layer_idx, data_role):
        del data_role
        self.calls.append((tuple(request_ids), layer_idx))
        return [[100 + layer_idx * 10 + idx] for idx, _ in enumerate(request_ids)]


class _FakeRotaryEmbedding:
    def __init__(self):
        self.position_ids = None

    def __call__(self, position_ids, targets):
        self.position_ids = position_ids.clone()
        q, k = targets
        return [q + 0.25, k - 0.125]


class TestTrtllmPackedCache:
    @staticmethod
    def _make_attention(num_heads=4, num_kv_heads=2, head_dim=128):
        attention = TrtllmAttention.__new__(TrtllmAttention)
        attention.layer_idx = 0
        attention.num_heads = num_heads
        attention.num_kv_heads = num_kv_heads
        attention.head_dim = head_dim
        attention.is_mla_enable = False
        attention.sparse_attention_config = None
        attention.attention_chunk_size = None
        attention.turboquant4_rotary_emb = None
        attention.wrapper = SimpleNamespace(q_scaling=1.0)
        return attention

    def test_trtllm_fallback_uses_native_packed_cache(self):
        num_heads = 4
        num_kv_heads = 2
        head_dim = 128
        tokens_per_block = 4
        key_cache = torch.empty(2, tokens_per_block, num_kv_heads, head_dim)
        value_cache = torch.empty(2, tokens_per_block, num_kv_heads, head_dim // 2, dtype=torch.uint8)
        value_scales = torch.empty(2, tokens_per_block, num_kv_heads, 1, dtype=torch.float32)
        block_ids = [0, 1]
        kv_manager = _FakeTurboQuant4KVCacheManager(
            key_cache, value_cache, value_scales, block_ids
        )
        attention = self._make_attention(num_heads, num_kv_heads, head_dim)

        prefill_q = torch.randn(4, num_heads * head_dim)
        prefill_k = torch.randn(4, num_kv_heads * head_dim)
        prefill_v = torch.randn(4, num_kv_heads * head_dim)
        prefill_metadata = SimpleNamespace(
            beam_width=1,
            enable_helix=False,
            is_spec_decoding_enabled=False,
            use_spec_decoding=False,
            kv_cache_manager=kv_manager,
            seq_lens=torch.tensor([4], dtype=torch.int32),
            seq_lens_kv=torch.tensor([4], dtype=torch.int32),
            num_tokens=4,
            request_ids=[0],
            tokens_per_block=tokens_per_block,
            position_ids=None,
            kv_cache_params=SimpleNamespace(num_cached_tokens_per_seq=[0]),
        )

        _ = attention._turboquant4_forward(
            prefill_q,
            prefill_k,
            prefill_v,
            prefill_metadata,
            output=None,
            output_sf=None,
            attention_mask=PredefinedAttentionMask.CAUSAL,
            attention_window_size=None,
        )

        decode_q = torch.randn(1, num_heads * head_dim)
        decode_k = torch.randn(1, num_kv_heads * head_dim)
        decode_v = torch.randn(1, num_kv_heads * head_dim)
        decode_metadata = SimpleNamespace(
            beam_width=1,
            enable_helix=False,
            is_spec_decoding_enabled=False,
            use_spec_decoding=False,
            kv_cache_manager=kv_manager,
            seq_lens=torch.tensor([1], dtype=torch.int32),
            seq_lens_kv=torch.tensor([1], dtype=torch.int32),
            num_tokens=1,
            request_ids=[0],
            tokens_per_block=tokens_per_block,
            position_ids=None,
            kv_cache_params=SimpleNamespace(num_cached_tokens_per_seq=[4]),
        )

        output = attention._turboquant4_forward(
            decode_q,
            decode_k,
            decode_v,
            decode_metadata,
            output=None,
            output_sf=None,
            attention_mask=PredefinedAttentionMask.CAUSAL,
            attention_window_size=None,
        )

        assert key_cache.dtype == torch.float32
        assert key_cache.shape[-1] == head_dim
        assert value_cache.dtype == torch.uint8
        assert value_cache.shape[-1] == head_dim // 2
        assert value_scales.dtype == torch.float32
        assert value_scales.shape[-1] == 1

        cached_k = read_turboquant4_dense_key_cache(
            key_cache, block_ids, 5, tokens_per_block, decode_q.dtype
        )
        cached_v = turboquant4_dequantize_value_cache(
            value_cache,
            value_scales,
            block_ids,
            block_ids,
            5,
            tokens_per_block,
            decode_q.dtype,
        )
        expected_k = torch.cat(
            [
                prefill_k.view(1, 4, num_kv_heads, head_dim),
                decode_k.view(1, 1, num_kv_heads, head_dim),
            ],
            dim=1,
        )
        expected_v = turboquant4_quantize_dequantize(
            torch.cat(
                [
                    prefill_v.view(1, 4, num_kv_heads, head_dim),
                    decode_v.view(1, 1, num_kv_heads, head_dim),
                ],
                dim=1,
            )
        )
        torch.testing.assert_close(cached_k, expected_k)
        torch.testing.assert_close(cached_v, expected_v)

        q = decode_q.view(1, 1, num_heads, head_dim).transpose(1, 2)
        expected_key = repeat_kv(expected_k.transpose(1, 2), 2)
        expected_attention_v = expected_v.clone()
        expected_attention_v[:, -1:] = decode_v.view(1, 1, num_kv_heads,
                                                      head_dim)
        expected_value = repeat_kv(expected_attention_v.transpose(1, 2), 2)
        expected_output = (
            torch.nn.functional.scaled_dot_product_attention(
                q,
                expected_key,
                expected_value,
                is_causal=False,
            )
            .transpose(1, 2)
            .contiguous()
            .view(1, -1)
        )
        torch.testing.assert_close(output, expected_output)

    def test_trtllm_fallback_handles_batched_requests(self):
        num_heads = 4
        num_kv_heads = 2
        head_dim = 128
        tokens_per_block = 4
        key_cache = torch.empty(2, tokens_per_block, num_kv_heads, head_dim)
        value_cache = torch.empty(2, tokens_per_block, num_kv_heads, head_dim // 2, dtype=torch.uint8)
        value_scales = torch.empty(2, tokens_per_block, num_kv_heads, 1, dtype=torch.float32)
        block_ids_per_request = [[0], [1]]
        kv_manager = _FakeTurboQuant4KVCacheManager(
            key_cache, value_cache, value_scales, block_ids_per_request
        )
        attention = self._make_attention(num_heads, num_kv_heads, head_dim)

        q = torch.randn(3, num_heads * head_dim)
        k = torch.randn(3, num_kv_heads * head_dim)
        v = torch.randn(3, num_kv_heads * head_dim)
        metadata = SimpleNamespace(
            beam_width=1,
            enable_helix=False,
            is_spec_decoding_enabled=False,
            use_spec_decoding=False,
            kv_cache_manager=kv_manager,
            seq_lens=torch.tensor([2, 1], dtype=torch.int32),
            seq_lens_kv=torch.tensor([2, 1], dtype=torch.int32),
            num_tokens=3,
            request_ids=[0, 1],
            tokens_per_block=tokens_per_block,
            position_ids=None,
            kv_cache_params=SimpleNamespace(num_cached_tokens_per_seq=[0, 0]),
        )

        output = attention._turboquant4_forward(
            q,
            k,
            v,
            metadata,
            output=None,
            output_sf=None,
            attention_mask=PredefinedAttentionMask.CAUSAL,
            attention_window_size=None,
        )

        expected_chunks = []
        offset = 0
        for q_len, block_ids in zip([2, 1], block_ids_per_request):
            cached_k = read_turboquant4_dense_key_cache(
                key_cache, block_ids, q_len, tokens_per_block, q.dtype
            )
            cached_v = turboquant4_dequantize_value_cache(
                value_cache,
                value_scales,
                block_ids,
                block_ids,
                q_len,
                tokens_per_block,
                q.dtype,
            )
            expected_k = k[offset : offset + q_len].view(1, q_len, num_kv_heads, head_dim)
            expected_v = turboquant4_quantize_dequantize(
                v[offset : offset + q_len].view(1, q_len, num_kv_heads, head_dim)
            )
            torch.testing.assert_close(cached_k, expected_k)
            torch.testing.assert_close(cached_v, expected_v)

            q_states = q[offset:offset + q_len].view(1, q_len, num_heads,
                                                     head_dim).transpose(1, 2)
            dense_k = k[offset:offset + q_len].view(1, q_len, num_kv_heads,
                                                    head_dim)
            dense_v = v[offset:offset + q_len].view(1, q_len, num_kv_heads,
                                                    head_dim)
            key_states = repeat_kv(dense_k.transpose(1, 2),
                                   num_heads // num_kv_heads)
            value_states = repeat_kv(dense_v.transpose(1, 2),
                                     num_heads // num_kv_heads)
            expected_chunks.append(
                torch.nn.functional.scaled_dot_product_attention(
                    q_states,
                    key_states,
                    value_states,
                    is_causal=True,
                )
                .transpose(1, 2)
                .contiguous()
                .view(q_len, -1)
            )
            offset += q_len

        torch.testing.assert_close(output, torch.cat(expected_chunks, dim=0))

    def test_trtllm_fallback_keeps_mixed_context_requests_dense(self):
        num_heads = 4
        num_kv_heads = 2
        head_dim = 128
        tokens_per_block = 4
        key_cache = torch.empty(3, tokens_per_block, num_kv_heads, head_dim)
        value_cache = torch.empty(3, tokens_per_block, num_kv_heads, head_dim // 2, dtype=torch.uint8)
        value_scales = torch.empty(3, tokens_per_block, num_kv_heads, 1, dtype=torch.float32)
        block_ids_per_request = [[0, 1], [2]]
        kv_manager = _FakeTurboQuant4KVCacheManager(
            key_cache, value_cache, value_scales, block_ids_per_request
        )
        attention = self._make_attention(num_heads, num_kv_heads, head_dim)

        prefill_q = torch.randn(4, num_heads * head_dim)
        prefill_k = torch.randn(4, num_kv_heads * head_dim)
        prefill_v = torch.randn(4, num_kv_heads * head_dim)
        prefill_metadata = SimpleNamespace(
            beam_width=1,
            enable_helix=False,
            is_spec_decoding_enabled=False,
            use_spec_decoding=False,
            kv_cache_manager=kv_manager,
            seq_lens=torch.tensor([4], dtype=torch.int32),
            seq_lens_kv=torch.tensor([4], dtype=torch.int32),
            num_tokens=4,
            request_ids=[0],
            tokens_per_block=tokens_per_block,
            position_ids=None,
            kv_cache_params=SimpleNamespace(num_cached_tokens_per_seq=[0]),
        )

        attention._turboquant4_forward(
            prefill_q,
            prefill_k,
            prefill_v,
            prefill_metadata,
            output=None,
            output_sf=None,
            attention_mask=PredefinedAttentionMask.CAUSAL,
            attention_window_size=None,
        )

        mixed_q = torch.randn(3, num_heads * head_dim)
        mixed_k = torch.randn(3, num_kv_heads * head_dim)
        mixed_v = torch.randn(3, num_kv_heads * head_dim)
        mixed_metadata = SimpleNamespace(
            beam_width=1,
            enable_helix=False,
            is_spec_decoding_enabled=False,
            use_spec_decoding=False,
            kv_cache_manager=kv_manager,
            seq_lens=torch.tensor([1, 2], dtype=torch.int32),
            seq_lens_kv=torch.tensor([1, 2], dtype=torch.int32),
            num_tokens=3,
            request_ids=[0, 1],
            tokens_per_block=tokens_per_block,
            position_ids=None,
            kv_cache_params=SimpleNamespace(num_cached_tokens_per_seq=[4, 0]),
        )

        output = attention._turboquant4_forward(
            mixed_q,
            mixed_k,
            mixed_v,
            mixed_metadata,
            output=None,
            output_sf=None,
            attention_mask=PredefinedAttentionMask.CAUSAL,
            attention_window_size=None,
        )

        cached_k_decode = read_turboquant4_dense_key_cache(
            key_cache, block_ids_per_request[0], 5, tokens_per_block, mixed_q.dtype
        )
        cached_v_decode = turboquant4_dequantize_value_cache(
            value_cache,
            value_scales,
            block_ids_per_request[0],
            block_ids_per_request[0],
            5,
            tokens_per_block,
            mixed_q.dtype,
        )
        cached_v_decode_for_attention = cached_v_decode.clone()
        cached_v_decode_for_attention[:, -1:] = mixed_v[:1].view(
            1, 1, num_kv_heads, head_dim)
        decode_q = mixed_q[:1].view(1, 1, num_heads, head_dim).transpose(1, 2)
        decode_key = repeat_kv(cached_k_decode.transpose(1, 2), num_heads // num_kv_heads)
        decode_value = repeat_kv(cached_v_decode_for_attention.transpose(1, 2),
                                 num_heads // num_kv_heads)
        expected_decode = (
            torch.nn.functional.scaled_dot_product_attention(
                decode_q,
                decode_key,
                decode_value,
                is_causal=False,
            )
            .transpose(1, 2)
            .contiguous()
            .view(1, -1)
        )

        cached_k_context = read_turboquant4_dense_key_cache(
            key_cache, block_ids_per_request[1], 2, tokens_per_block, mixed_q.dtype
        )
        cached_v_context = turboquant4_dequantize_value_cache(
            value_cache,
            value_scales,
            block_ids_per_request[1],
            block_ids_per_request[1],
            2,
            tokens_per_block,
            mixed_q.dtype,
        )
        expected_cached_k_context = mixed_k[1:].view(1, 2, num_kv_heads, head_dim)
        expected_cached_v_context = turboquant4_quantize_dequantize(
            mixed_v[1:].view(1, 2, num_kv_heads, head_dim)
        )
        torch.testing.assert_close(cached_k_context, expected_cached_k_context)
        torch.testing.assert_close(cached_v_context, expected_cached_v_context)

        context_q = mixed_q[1:].view(1, 2, num_heads, head_dim).transpose(1, 2)
        dense_k = mixed_k[1:].view(1, 2, num_kv_heads, head_dim)
        dense_v = mixed_v[1:].view(1, 2, num_kv_heads, head_dim)
        context_key = repeat_kv(dense_k.transpose(1, 2), num_heads // num_kv_heads)
        context_value = repeat_kv(dense_v.transpose(1, 2), num_heads // num_kv_heads)
        expected_context = (
            torch.nn.functional.scaled_dot_product_attention(
                context_q,
                context_key,
                context_value,
                is_causal=True,
            )
            .transpose(1, 2)
            .contiguous()
            .view(2, -1)
        )

        torch.testing.assert_close(output, torch.cat([expected_decode, expected_context], dim=0))

    def test_trtllm_fallback_rejects_query_kv_token_count_mismatch(self):
        num_heads = 4
        num_kv_heads = 2
        head_dim = 128
        tokens_per_block = 4
        key_cache = torch.empty(2, tokens_per_block, num_kv_heads, head_dim)
        value_cache = torch.empty(2, tokens_per_block, num_kv_heads, head_dim // 2, dtype=torch.uint8)
        value_scales = torch.empty(2, tokens_per_block, num_kv_heads, 1, dtype=torch.float32)
        block_ids = [0, 1]
        kv_manager = _FakeTurboQuant4KVCacheManager(
            key_cache, value_cache, value_scales, block_ids
        )
        attention = self._make_attention(num_heads, num_kv_heads, head_dim)
        q = torch.randn(2, num_heads * head_dim)
        k = torch.randn(1, num_kv_heads * head_dim)
        v = torch.randn(1, num_kv_heads * head_dim)
        metadata = SimpleNamespace(
            beam_width=1,
            enable_helix=False,
            is_spec_decoding_enabled=False,
            use_spec_decoding=False,
            kv_cache_manager=kv_manager,
            seq_lens=torch.tensor([2], dtype=torch.int32),
            seq_lens_kv=torch.tensor([1], dtype=torch.int32),
            num_tokens=1,
            request_ids=[0],
            tokens_per_block=tokens_per_block,
            position_ids=None,
            kv_cache_params=SimpleNamespace(num_cached_tokens_per_seq=[4]),
        )
        with pytest.raises(NotImplementedError, match="matching query"):
            attention._turboquant4_forward(
                q,
                k,
                v,
                metadata,
                output=torch.empty(4, num_heads * head_dim),
                output_sf=None,
                attention_mask=PredefinedAttentionMask.CAUSAL,
                attention_window_size=None,
            )

    def test_attention_impl_uses_query_token_count_for_output_slice(self):
        num_heads = 4
        num_kv_heads = 2
        head_dim = 128
        module = Attention.__new__(Attention)
        module.quant_config = QuantConfig(kv_cache_quant_algo=QuantAlgo.TURBOQUANT4)
        module.mapping = SimpleNamespace(has_cp_helix=lambda: False)
        module._use_quantize_output = lambda: False
        seen = {}

        def fake_forward(q, k, v, *args, **kwargs):
            seen["q_shape"] = q.shape
            seen["k_shape"] = k.shape
            seen["v_shape"] = v.shape
            seen["output_shape"] = kwargs["output"].shape
            return torch.full_like(q, 2.0)

        module.attn = SimpleNamespace(forward=fake_forward)
        metadata = SimpleNamespace(
            num_tokens=1,
            seq_lens=torch.tensor([2], dtype=torch.int32),
            seq_lens_kv=torch.tensor([2], dtype=torch.int32),
            num_contexts=0,
        )
        q = torch.randn(2, num_heads * head_dim)
        k = torch.randn(2, num_kv_heads * head_dim)
        v = torch.randn(2, num_kv_heads * head_dim)
        output_buffer = torch.empty(4, num_heads * head_dim)

        output, output_sf = module._attn_impl(
            q,
            k,
            v,
            metadata,
            PredefinedAttentionMask.CAUSAL,
            mrope_rotary_cos_sin=None,
            mrope_position_deltas=None,
            attention_window_size=None,
            attention_mask_data=None,
            output=output_buffer,
        )

        assert seen["q_shape"] == q.shape
        assert seen["k_shape"] == k.shape
        assert seen["v_shape"] == v.shape
        assert seen["output_shape"] == q.shape
        assert output_sf is None
        torch.testing.assert_close(output, torch.full_like(q, 2.0))

    def test_attention_impl_rejects_turboquant4_helix_before_backend(self):
        num_heads = 4
        num_kv_heads = 2
        head_dim = 128
        module = Attention.__new__(Attention)
        module.quant_config = QuantConfig(kv_cache_quant_algo=QuantAlgo.TURBOQUANT4)
        module.mapping = SimpleNamespace(has_cp_helix=lambda: True)
        module._use_quantize_output = lambda: False
        module.attn = SimpleNamespace(
            forward=lambda *args, **kwargs: pytest.fail(
                "TurboQuant4 Helix path should reject before backend forward."
            )
        )
        metadata = SimpleNamespace(
            num_tokens=1,
            seq_lens=torch.tensor([1], dtype=torch.int32),
            seq_lens_kv=torch.tensor([1], dtype=torch.int32),
            num_contexts=0,
        )

        with pytest.raises(NotImplementedError, match="Helix"):
            module._attn_impl(
                torch.randn(1, num_heads * head_dim),
                torch.randn(1, num_kv_heads * head_dim),
                torch.randn(1, num_kv_heads * head_dim),
                metadata,
                PredefinedAttentionMask.CAUSAL,
                mrope_rotary_cos_sin=None,
                mrope_position_deltas=None,
                attention_window_size=None,
                attention_mask_data=None,
            )

    def test_attention_impl_rejects_turboquant4_cuda_graph_before_backend(self):
        num_heads = 4
        num_kv_heads = 2
        head_dim = 128
        module = Attention.__new__(Attention)
        module.quant_config = QuantConfig(kv_cache_quant_algo=QuantAlgo.TURBOQUANT4)
        module.mapping = SimpleNamespace(has_cp_helix=lambda: False)
        module._use_quantize_output = lambda: False
        module.attn = SimpleNamespace(
            forward=lambda *args, **kwargs: pytest.fail(
                "TurboQuant4 CUDA graph path should reject before backend forward."
            )
        )
        metadata = SimpleNamespace(
            num_tokens=1,
            seq_lens=torch.tensor([1], dtype=torch.int32),
            seq_lens_kv=torch.tensor([1], dtype=torch.int32),
            num_contexts=0,
            is_cuda_graph=True,
        )

        with pytest.raises(NotImplementedError, match="CUDA graph"):
            module._attn_impl(
                torch.randn(1, num_heads * head_dim),
                torch.randn(1, num_kv_heads * head_dim),
                torch.randn(1, num_kv_heads * head_dim),
                metadata,
                PredefinedAttentionMask.CAUSAL,
                mrope_rotary_cos_sin=None,
                mrope_position_deltas=None,
                attention_window_size=None,
                attention_mask_data=None,
            )

    def test_trtllm_fallback_uses_forward_position_ids_for_rope(self):
        num_heads = 4
        num_kv_heads = 2
        head_dim = 128
        tokens_per_block = 4
        key_cache = torch.empty(1, tokens_per_block, num_kv_heads, head_dim)
        value_cache = torch.empty(1, tokens_per_block, num_kv_heads, head_dim // 2, dtype=torch.uint8)
        value_scales = torch.empty(1, tokens_per_block, num_kv_heads, 1, dtype=torch.float32)
        block_ids = [0]
        kv_manager = _FakeTurboQuant4KVCacheManager(
            key_cache, value_cache, value_scales, block_ids
        )
        attention = self._make_attention(num_heads, num_kv_heads, head_dim)
        rotary = _FakeRotaryEmbedding()
        attention.turboquant4_rotary_emb = rotary

        q = torch.randn(3, num_heads * head_dim)
        k = torch.randn(3, num_kv_heads * head_dim)
        v = torch.randn(3, num_kv_heads * head_dim)
        fused_qkv = torch.cat([q, k, v], dim=-1)
        position_ids = torch.tensor([[0, 1, 2]], dtype=torch.int32)
        metadata = SimpleNamespace(
            beam_width=1,
            enable_helix=False,
            is_spec_decoding_enabled=False,
            use_spec_decoding=False,
            kv_cache_manager=kv_manager,
            seq_lens=torch.tensor([3], dtype=torch.int32),
            seq_lens_kv=torch.tensor([3], dtype=torch.int32),
            num_tokens=3,
            request_ids=[0],
            tokens_per_block=tokens_per_block,
            kv_cache_params=SimpleNamespace(num_cached_tokens_per_seq=[0]),
        )

        _ = attention._turboquant4_forward(
            fused_qkv,
            None,
            None,
            metadata,
            output=None,
            output_sf=None,
            attention_mask=PredefinedAttentionMask.CAUSAL,
            attention_window_size=None,
            position_ids=position_ids,
        )

        torch.testing.assert_close(rotary.position_ids, position_ids)
        cached_k = read_turboquant4_dense_key_cache(
            key_cache, block_ids, 3, tokens_per_block, q.dtype
        )
        expected_k = (k - 0.125).view(1, 3, num_kv_heads, head_dim)
        torch.testing.assert_close(cached_k, expected_k)

    def test_trtllm_fallback_rejects_attention_sinks(self):
        num_heads = 4
        num_kv_heads = 2
        head_dim = 128
        tokens_per_block = 4
        key_cache = torch.empty(1, tokens_per_block, num_kv_heads, head_dim)
        value_cache = torch.empty(1, tokens_per_block, num_kv_heads, head_dim // 2, dtype=torch.uint8)
        value_scales = torch.empty(1, tokens_per_block, num_kv_heads, 1, dtype=torch.float32)
        kv_manager = _FakeTurboQuant4KVCacheManager(
            key_cache, value_cache, value_scales, [0]
        )
        attention = self._make_attention(num_heads, num_kv_heads, head_dim)
        metadata = SimpleNamespace(
            beam_width=1,
            enable_helix=False,
            is_spec_decoding_enabled=False,
            use_spec_decoding=False,
            kv_cache_manager=kv_manager,
            seq_lens=torch.tensor([1], dtype=torch.int32),
            seq_lens_kv=torch.tensor([1], dtype=torch.int32),
            num_tokens=1,
            request_ids=[0],
            tokens_per_block=tokens_per_block,
            kv_cache_params=SimpleNamespace(num_cached_tokens_per_seq=[0]),
        )

        with pytest.raises(NotImplementedError, match="attention sinks"):
            attention._turboquant4_forward(
                torch.randn(1, num_heads * head_dim),
                torch.randn(1, num_kv_heads * head_dim),
                torch.randn(1, num_kv_heads * head_dim),
                metadata,
                output=None,
                output_sf=None,
                attention_mask=PredefinedAttentionMask.CAUSAL,
                attention_window_size=None,
                attention_sinks=torch.zeros(num_heads),
            )

    def test_trtllm_fallback_rejects_mrope_config(self):
        num_heads = 4
        num_kv_heads = 2
        head_dim = 128
        tokens_per_block = 4
        key_cache = torch.empty(1, tokens_per_block, num_kv_heads, head_dim)
        value_cache = torch.empty(1, tokens_per_block, num_kv_heads, head_dim // 2, dtype=torch.uint8)
        value_scales = torch.empty(1, tokens_per_block, num_kv_heads, 1, dtype=torch.float32)
        kv_manager = _FakeTurboQuant4KVCacheManager(
            key_cache, value_cache, value_scales, [0]
        )
        attention = self._make_attention(num_heads, num_kv_heads, head_dim)
        metadata = SimpleNamespace(
            beam_width=1,
            enable_helix=False,
            is_spec_decoding_enabled=False,
            use_spec_decoding=False,
            kv_cache_manager=kv_manager,
            seq_lens=torch.tensor([1], dtype=torch.int32),
            seq_lens_kv=torch.tensor([1], dtype=torch.int32),
            num_tokens=1,
            request_ids=[0],
            tokens_per_block=tokens_per_block,
            kv_cache_params=SimpleNamespace(num_cached_tokens_per_seq=[0]),
        )

        with pytest.raises(NotImplementedError, match="MRoPE"):
            attention._turboquant4_forward(
                torch.randn(1, num_heads * head_dim),
                torch.randn(1, num_kv_heads * head_dim),
                torch.randn(1, num_kv_heads * head_dim),
                metadata,
                output=None,
                output_sf=None,
                attention_mask=PredefinedAttentionMask.CAUSAL,
                attention_window_size=None,
                mrope_config={"mrope_rotary_cos_sin": torch.empty(1)},
            )

    def test_trtllm_fallback_rejects_mrope_embedding_type(self):
        num_heads = 4
        num_kv_heads = 2
        head_dim = 128
        tokens_per_block = 4
        key_cache = torch.empty(1, tokens_per_block, num_kv_heads, head_dim)
        value_cache = torch.empty(1, tokens_per_block, num_kv_heads, head_dim // 2, dtype=torch.uint8)
        value_scales = torch.empty(1, tokens_per_block, num_kv_heads, 1, dtype=torch.float32)
        kv_manager = _FakeTurboQuant4KVCacheManager(
            key_cache, value_cache, value_scales, [0]
        )
        attention = self._make_attention(num_heads, num_kv_heads, head_dim)
        attention.pos_embd_params = SimpleNamespace(
            type=SimpleNamespace(is_mrope=lambda: True, is_rope=lambda: True))
        metadata = SimpleNamespace(
            beam_width=1,
            enable_helix=False,
            is_spec_decoding_enabled=False,
            use_spec_decoding=False,
            kv_cache_manager=kv_manager,
            seq_lens=torch.tensor([1], dtype=torch.int32),
            seq_lens_kv=torch.tensor([1], dtype=torch.int32),
            num_tokens=1,
            request_ids=[0],
            tokens_per_block=tokens_per_block,
            kv_cache_params=SimpleNamespace(num_cached_tokens_per_seq=[0]),
        )

        with pytest.raises(NotImplementedError, match="MRoPE"):
            attention._turboquant4_forward(
                torch.randn(1, num_heads * head_dim),
                torch.randn(1, num_kv_heads * head_dim),
                torch.randn(1, num_kv_heads * head_dim),
                metadata,
                output=None,
                output_sf=None,
                attention_mask=PredefinedAttentionMask.CAUSAL,
                attention_window_size=None,
            )

    def test_trtllm_fallback_rejects_cuda_graph_capture(self):
        num_heads = 4
        num_kv_heads = 2
        head_dim = 128
        attention = self._make_attention(num_heads, num_kv_heads, head_dim)
        metadata = SimpleNamespace(
            beam_width=1,
            enable_helix=False,
            is_spec_decoding_enabled=False,
            use_spec_decoding=False,
            is_cuda_graph=True,
        )

        with pytest.raises(NotImplementedError, match="CUDA graph"):
            attention._turboquant4_forward(
                torch.randn(1, num_heads * head_dim),
                torch.randn(1, num_kv_heads * head_dim),
                torch.randn(1, num_kv_heads * head_dim),
                metadata,
                output=None,
                output_sf=None,
                attention_mask=PredefinedAttentionMask.CAUSAL,
                attention_window_size=None,
            )

    def test_trtllm_fallback_rejects_chunked_attention(self):
        num_heads = 4
        num_kv_heads = 2
        head_dim = 128
        attention = self._make_attention(num_heads, num_kv_heads, head_dim)
        attention.attention_chunk_size = 128
        metadata = SimpleNamespace(
            beam_width=1,
            enable_helix=False,
            is_spec_decoding_enabled=False,
            use_spec_decoding=False,
            is_cuda_graph=False,
        )

        with pytest.raises(NotImplementedError, match="chunked attention"):
            attention._turboquant4_forward(
                torch.randn(1, num_heads * head_dim),
                torch.randn(1, num_kv_heads * head_dim),
                torch.randn(1, num_kv_heads * head_dim),
                metadata,
                output=None,
                output_sf=None,
                attention_mask=PredefinedAttentionMask.CAUSAL,
                attention_window_size=None,
            )

    def test_turboquant4_rotary_embedding_is_created_lazily(self, monkeypatch):
        calls = []

        class FakeRotaryEmbedding:
            def __init__(self, rope, head_dim, is_neox):
                calls.append((rope, head_dim, is_neox))

        attention = self._make_attention(head_dim=128)
        rope = object()
        attention.pos_embd_params = SimpleNamespace(
            rope=rope,
            is_neox=True,
            type=SimpleNamespace(is_mrope=lambda: False, is_rope=lambda: True),
        )
        monkeypatch.setattr(trtllm_backend, "RotaryEmbedding",
                            FakeRotaryEmbedding)

        assert attention.turboquant4_rotary_emb is None
        assert calls == []
        rotary_emb = attention._get_turboquant4_rotary_emb()

        assert isinstance(rotary_emb, FakeRotaryEmbedding)
        assert attention._get_turboquant4_rotary_emb() is rotary_emb
        assert calls == [(rope, 128, True)]

    def test_turboquant4_rotary_embedding_rejects_non_rope(self):
        attention = self._make_attention(head_dim=128)
        attention.pos_embd_params = SimpleNamespace(
            type=SimpleNamespace(is_mrope=lambda: False, is_rope=lambda: False))

        with pytest.raises(NotImplementedError, match="RoPE positional"):
            attention._get_turboquant4_rotary_emb()

    def test_trtllm_forward_intercepts_before_fused_attention(self):
        num_heads = 4
        num_kv_heads = 2
        head_dim = 128
        attention = self._make_attention(num_heads, num_kv_heads, head_dim)
        attention.has_turboquant4_kv_cache = True
        sentinel = torch.randn(1, num_heads * head_dim)
        seen = {}

        def fake_turboquant4_forward(*args, **kwargs):
            seen["mrope_config"] = kwargs["mrope_config"]
            return sentinel

        attention._turboquant4_forward = fake_turboquant4_forward
        attention.wrapper = SimpleNamespace(
            create_output=lambda *args, **kwargs: pytest.fail(
                "TurboQuant4 should bypass regular TRTLLM output allocation"
            ),
            plan=lambda *args, **kwargs: pytest.fail(
                "TurboQuant4 should bypass fused TRTLLM attention planning"
            ),
        )

        metadata = TrtllmAttentionMetadata.__new__(TrtllmAttentionMetadata)
        metadata._seq_lens = torch.tensor([1], dtype=torch.int32)
        metadata._seq_lens_kv = None
        metadata.runtime_features = None

        result = attention.forward(
            torch.randn(1, num_heads * head_dim),
            None,
            None,
            metadata,
            output=None,
            output_sf=None,
            attention_mask=PredefinedAttentionMask.CAUSAL,
            attention_input_type=AttentionInputType.context_only,
            mrope_config={"mrope_rotary_cos_sin": torch.empty(1)},
        )

        assert result is sentinel
        assert seen["mrope_config"] is not None

    def test_trtllm_forward_rejects_turboquant4_quantized_output(self):
        num_heads = 4
        num_kv_heads = 2
        head_dim = 128
        attention = self._make_attention(num_heads, num_kv_heads, head_dim)
        attention.has_turboquant4_kv_cache = True
        attention.wrapper = SimpleNamespace()

        metadata = TrtllmAttentionMetadata.__new__(TrtllmAttentionMetadata)
        metadata._seq_lens = torch.tensor([1], dtype=torch.int32)
        metadata._seq_lens_kv = None
        metadata.runtime_features = None

        with pytest.raises(NotImplementedError, match="quantized attention output"):
            attention.forward(
                torch.randn(1, num_heads * head_dim),
                None,
                None,
                metadata,
                output=None,
                output_sf=None,
                out_scale=torch.ones(1),
                attention_mask=PredefinedAttentionMask.CAUSAL,
                attention_input_type=AttentionInputType.context_only,
            )

    def test_trtllm_gen_support_probe_rejects_turboquant4(self):
        supported, reason = trtllm_gen.is_supported(
            q=torch.randn(1, 4 * 128),
            num_heads=4,
            num_kv_heads=2,
            head_size=128,
            quant_config=QuantConfig(kv_cache_quant_algo=QuantAlgo.TURBOQUANT4),
        )

        assert not supported
        assert "TurboQuant4" in reason
