# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Unit tests for LoraManager._retain_device_tensors behavior.

Verifies that GPU tensors are not accumulated in _lora_weights when the
PyTorch backend's C++ PeftCacheManager is provided, preventing OOM with
many unique LoRA adapters.
"""

import json
import tempfile
import unittest
from dataclasses import dataclass, field
from pathlib import Path
from unittest.mock import MagicMock, patch

import torch
from safetensors.torch import save_file

from tensorrt_llm._torch.peft.lora.loaders import HfLoraLoader
from tensorrt_llm._torch.peft.lora.manager import LoraManager, supports_native_fp8_lora
from tensorrt_llm.mapping import Mapping


@dataclass
class MockModelConfig:
    """Minimal model config for LoraManager tests."""

    lora_target_modules: list = field(default_factory=lambda: ["attn_q", "attn_k", "attn_v"])
    trtllm_modules_to_hf_modules: dict = field(
        default_factory=lambda: {
            "attn_q": "q_proj",
            "attn_k": "k_proj",
            "attn_v": "v_proj",
        }
    )
    hidden_size: int = 64
    dtype: str = "float16"
    swap_gate_up_proj_lora_b_weight: bool = True


def _create_dummy_hf_lora_adapter(
    adapter_dir: Path,
    hidden_size: int = 64,
    output_size: int | None = None,
    rank: int = 8,
    num_layers: int = 2,
    dtype: torch.dtype = torch.float16,
    use_dora: bool = False,
    input_dtype: torch.dtype | None = None,
    lora_alpha: float | None = None,
    weight_value: float | None = None,
    modules: list[str] | None = None,
):
    """Create a minimal HF-format LoRA adapter on disk."""
    output_size = hidden_size if output_size is None else output_size
    modules = ["q_proj", "k_proj", "v_proj"] if modules is None else modules
    config = {
        "r": rank,
        "lora_alpha": rank if lora_alpha is None else lora_alpha,
        "target_modules": modules,
        "bias": "none",
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "use_dora": use_dora,
    }
    with open(adapter_dir / "adapter_config.json", "w") as f:
        json.dump(config, f)

    weights = {}
    input_dtype = dtype if input_dtype is None else input_dtype

    def make_weight(shape, target_dtype):
        if weight_value is None:
            return torch.randn(*shape, dtype=torch.float16).to(target_dtype)
        return torch.full(shape, weight_value, dtype=torch.float16).to(target_dtype)

    for layer_idx in range(num_layers):
        for module in modules:
            prefix = f"base_model.model.model.layers.{layer_idx}.self_attn.{module}"
            weights[f"{prefix}.lora_A.weight"] = make_weight((rank, hidden_size), input_dtype)
            weights[f"{prefix}.lora_B.weight"] = make_weight((output_size, rank), dtype)
            if use_dora:
                weights[f"{prefix}.lora_magnitude_vector"] = torch.ones(
                    output_size, dtype=torch.bfloat16
                )

    save_file(weights, str(adapter_dir / "adapter_model.safetensors"))


def _create_dummy_hf_moe_lora_adapter(
    adapter_dir: Path,
    hidden_size: int = 64,
    output_size: int = 128,
    rank: int = 16,
    num_experts: int = 2,
    dtype: torch.dtype = torch.float8_e4m3fn,
    include_dense: bool = False,
):
    """Create a minimal expert-indexed HF-format LoRA adapter on disk."""
    config = {
        "r": rank,
        "lora_alpha": rank,
        "target_modules": ["gate_proj", "q_proj"] if include_dense else ["gate_proj"],
        "bias": "none",
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
    }
    with open(adapter_dir / "adapter_config.json", "w") as f:
        json.dump(config, f)

    weights = {}
    for expert_idx in range(num_experts):
        prefix = f"base_model.model.model.layers.0.mlp.experts.{expert_idx}.gate_proj"
        weights[f"{prefix}.lora_A.weight"] = torch.randn(rank, hidden_size, dtype=torch.float16).to(
            dtype
        )
        weights[f"{prefix}.lora_B.weight"] = torch.randn(output_size, rank, dtype=torch.float16).to(
            dtype
        )
    if include_dense:
        prefix = "base_model.model.model.layers.0.self_attn.q_proj"
        weights[f"{prefix}.lora_A.weight"] = torch.randn(rank, hidden_size, dtype=torch.float16).to(
            dtype
        )
        weights[f"{prefix}.lora_B.weight"] = torch.randn(hidden_size, rank, dtype=torch.float16).to(
            dtype
        )

    save_file(weights, str(adapter_dir / "adapter_model.safetensors"))


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestLoraManagerRetainDeviceTensors(unittest.TestCase):
    """Tests for the _retain_device_tensors flag that prevents GPU memory leaks."""

    def _create_manager(self, cpp_peft_cache_manager=None):
        mapping = Mapping(world_size=1, rank=0, tp_size=1)
        model_config = MockModelConfig()
        return LoraManager(
            mapping=mapping,
            model_config=model_config,
            cpp_peft_cache_manager=cpp_peft_cache_manager,
        )

    def test_retain_device_tensors_true_when_no_cpp_cache(self):
        """Legacy TRT path: cpp_peft_cache_manager=None retains GPU tensors."""
        manager = self._create_manager(cpp_peft_cache_manager=None)
        self.assertTrue(manager._retain_device_tensors)

    def test_retain_device_tensors_false_when_cpp_cache_provided(self):
        """PyTorch path: cpp_peft_cache_manager provided skips GPU tensor retention."""
        mock_cache = MagicMock()
        manager = self._create_manager(cpp_peft_cache_manager=mock_cache)
        self.assertFalse(manager._retain_device_tensors)

    def test_lora_weights_empty_with_cpp_cache(self):
        """With cpp_peft_cache_manager, _lora_weights stays empty after loading."""
        mock_cache = MagicMock()
        manager = self._create_manager(cpp_peft_cache_manager=mock_cache)

        with tempfile.TemporaryDirectory() as tmpdir:
            adapter_dir = Path(tmpdir) / "adapter_0"
            adapter_dir.mkdir()
            _create_dummy_hf_lora_adapter(adapter_dir)

            model_config = MockModelConfig()
            manager.load_from_hf(
                model_dirs=[str(adapter_dir)],
                model_config=model_config,
                uids=["test-uid-0"],
            )

        self.assertEqual(len(manager._lora_weights), 0)
        self.assertIn("test-uid-0", manager._cpp_lora_weights)

    def test_lora_weights_populated_without_cpp_cache(self):
        """Without cpp_peft_cache_manager (TRT), _lora_weights has GPU tensors."""
        manager = self._create_manager(cpp_peft_cache_manager=None)

        with tempfile.TemporaryDirectory() as tmpdir:
            adapter_dir = Path(tmpdir) / "adapter_0"
            adapter_dir.mkdir()
            _create_dummy_hf_lora_adapter(adapter_dir)

            model_config = MockModelConfig()
            manager.load_from_hf(
                model_dirs=[str(adapter_dir)],
                model_config=model_config,
                uids=["test-uid-0"],
            )

        self.assertGreater(len(manager._lora_weights), 0)
        self.assertTrue(all(t.is_cuda for t in manager._lora_weights))
        self.assertIn("test-uid-0", manager._lora_weights_pointers_list)

    def test_many_adapters_no_gpu_accumulation(self):
        """Loading many adapters with cpp_cache does not accumulate GPU tensors."""
        mock_cache = MagicMock()
        manager = self._create_manager(cpp_peft_cache_manager=mock_cache)
        model_config = MockModelConfig()

        num_adapters = 20
        with tempfile.TemporaryDirectory() as tmpdir:
            for i in range(num_adapters):
                adapter_dir = Path(tmpdir) / f"adapter_{i}"
                adapter_dir.mkdir()
                _create_dummy_hf_lora_adapter(adapter_dir)

                manager.load_from_hf(
                    model_dirs=[str(adapter_dir)],
                    model_config=model_config,
                    uids=[f"uid-{i}"],
                )

        self.assertEqual(len(manager._lora_weights), 0)
        self.assertEqual(len(manager._cpp_lora_weights), num_adapters)


@unittest.skipUnless(
    torch.cuda.is_available() and supports_native_fp8_lora(torch.cuda.get_device_capability()),
    "Native FP8 LoRA requires SM90 or SM100",
)
class TestLoraManagerFp8(unittest.TestCase):
    def test_hf_loader_reports_fp8_dtype(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            adapter_dir = Path(tmpdir) / "adapter"
            adapter_dir.mkdir()
            _create_dummy_hf_lora_adapter(
                adapter_dir,
                rank=16,
                num_layers=1,
                dtype=torch.float8_e4m3fn,
            )

            loader = HfLoraLoader([str(adapter_dir)])

        self.assertEqual(loader.get_lora_dtype(), torch.float8_e4m3fn)

    def test_hf_loader_reports_expert_only_fp8_adapter_dtype(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            adapter_dir = Path(tmpdir) / "adapter"
            adapter_dir.mkdir()
            _create_dummy_hf_moe_lora_adapter(adapter_dir)

            loader = HfLoraLoader([str(adapter_dir)])

        self.assertEqual(loader.get_lora_dtype(), torch.float8_e4m3fn)

    def test_fp8_dora_is_rejected(self):
        model_config = MockModelConfig()
        manager = LoraManager(
            mapping=Mapping(world_size=1, rank=0, tp_size=1),
            model_config=model_config,
            cpp_peft_cache_manager=MagicMock(),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            adapter_dir = Path(tmpdir) / "adapter"
            adapter_dir.mkdir()
            _create_dummy_hf_lora_adapter(
                adapter_dir,
                rank=16,
                num_layers=1,
                dtype=torch.float8_e4m3fn,
                use_dora=True,
            )

            with self.assertRaisesRegex(
                NotImplementedError,
                "DoRA is not supported with FP8 LoRA weights on SM90/SM100",
            ):
                manager.load_from_hf(
                    model_dirs=[str(adapter_dir)],
                    model_config=model_config,
                    uids=["fp8-dora"],
                )

    def test_fp8_moe_weights_remain_fp8(self):
        model_config = MockModelConfig(
            lora_target_modules=["moe_h_to_4h"],
            trtllm_modules_to_hf_modules={"moe_h_to_4h": "mlp.experts.gate_proj"},
            dtype="bfloat16",
        )
        manager = LoraManager(
            mapping=Mapping(world_size=1, rank=0, tp_size=1),
            model_config=model_config,
            cpp_peft_cache_manager=MagicMock(),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            adapter_dir = Path(tmpdir) / "adapter"
            adapter_dir.mkdir()
            _create_dummy_hf_moe_lora_adapter(adapter_dir)

            manager.load_from_hf(
                model_dirs=[str(adapter_dir)],
                model_config=model_config,
                uids=["fp8-moe"],
            )

        self.assertEqual(manager.cpp_lora_weights["fp8-moe"].dtype, torch.float8_e4m3fn)

    def test_mixed_fp8_dense_and_moe_modules_use_one_fp8_cache_dtype(self):
        model_config = MockModelConfig(
            lora_target_modules=["attn_q", "moe_h_to_4h"],
            trtllm_modules_to_hf_modules={
                "attn_q": "q_proj",
                "attn_k": "k_proj",
                "attn_v": "v_proj",
                "moe_h_to_4h": "mlp.experts.gate_proj",
            },
            dtype="bfloat16",
        )
        manager = LoraManager(
            mapping=Mapping(world_size=1, rank=0, tp_size=1),
            model_config=model_config,
            cpp_peft_cache_manager=MagicMock(),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            adapter_dir = Path(tmpdir) / "adapter"
            adapter_dir.mkdir()
            _create_dummy_hf_moe_lora_adapter(adapter_dir, include_dense=True)

            manager.load_from_hf(
                model_dirs=[str(adapter_dir)],
                model_config=model_config,
                uids=["mixed-fp8"],
            )

        self.assertEqual(manager.cpp_lora_weights["mixed-fp8"].dtype, torch.float8_e4m3fn)

    def test_partial_qkv_fp8_adapter_uses_fp8_placeholders(self):
        model_config = MockModelConfig(dtype="bfloat16")
        manager = LoraManager(
            mapping=Mapping(world_size=1, rank=0, tp_size=1),
            model_config=model_config,
            cpp_peft_cache_manager=MagicMock(),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            adapter_dir = Path(tmpdir) / "adapter"
            adapter_dir.mkdir()
            _create_dummy_hf_lora_adapter(
                adapter_dir,
                rank=16,
                num_layers=1,
                dtype=torch.float8_e4m3fn,
                modules=["q_proj", "v_proj"],
            )

            manager.load_from_hf(
                model_dirs=[str(adapter_dir)],
                model_config=model_config,
                uids=["partial-qkv-fp8"],
            )

        self.assertEqual(
            manager.cpp_lora_weights["partial-qkv-fp8"].dtype,
            torch.float8_e4m3fn,
        )

    def test_fp8_e5m2_weights_are_converted_to_model_dtype(self):
        model_config = MockModelConfig(dtype="bfloat16")
        manager = LoraManager(
            mapping=Mapping(world_size=1, rank=0, tp_size=1),
            model_config=model_config,
            cpp_peft_cache_manager=MagicMock(),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            adapter_dir = Path(tmpdir) / "adapter"
            adapter_dir.mkdir()
            _create_dummy_hf_lora_adapter(
                adapter_dir,
                rank=8,
                num_layers=1,
                dtype=torch.float8_e5m2,
            )

            manager.load_from_hf(
                model_dirs=[str(adapter_dir)],
                model_config=model_config,
                uids=["fp8-e5m2"],
            )

        self.assertEqual(manager.cpp_lora_weights["fp8-e5m2"].dtype, torch.bfloat16)

    def test_fp8_e4m3_weights_on_sm100_remain_fp8(self):
        model_config = MockModelConfig(dtype="bfloat16")
        manager = LoraManager(
            mapping=Mapping(world_size=1, rank=0, tp_size=1),
            model_config=model_config,
            cpp_peft_cache_manager=MagicMock(),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            adapter_dir = Path(tmpdir) / "adapter"
            adapter_dir.mkdir()
            _create_dummy_hf_lora_adapter(
                adapter_dir,
                rank=16,
                num_layers=1,
                dtype=torch.float8_e4m3fn,
            )

            with (
                patch("torch.cuda.get_device_capability", return_value=(10, 0)),
                patch(
                    "tensorrt_llm._torch.peft.lora.manager._native_fp8_lora_kernels_available",
                    return_value=True,
                ),
            ):
                manager.load_from_hf(
                    model_dirs=[str(adapter_dir)],
                    model_config=model_config,
                    uids=["fp8-e4m3-sm100"],
                )

        self.assertEqual(
            manager.cpp_lora_weights["fp8-e4m3-sm100"].dtype,
            torch.float8_e4m3fn,
        )

    def test_fp8_e4m3_weights_on_sm100_without_kernels_use_model_dtype(self):
        model_config = MockModelConfig(dtype="bfloat16")
        manager = LoraManager(
            mapping=Mapping(world_size=1, rank=0, tp_size=1),
            model_config=model_config,
            cpp_peft_cache_manager=MagicMock(),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            adapter_dir = Path(tmpdir) / "adapter"
            adapter_dir.mkdir()
            _create_dummy_hf_lora_adapter(
                adapter_dir,
                rank=16,
                num_layers=1,
                dtype=torch.float8_e4m3fn,
            )

            with (
                patch("torch.cuda.get_device_capability", return_value=(10, 0)),
                patch(
                    "tensorrt_llm._torch.peft.lora.manager._native_fp8_lora_kernels_available",
                    return_value=False,
                ),
            ):
                manager.load_from_hf(
                    model_dirs=[str(adapter_dir)],
                    model_config=model_config,
                    uids=["fp8-e4m3-sm100-fallback"],
                )

        self.assertEqual(
            manager.cpp_lora_weights["fp8-e4m3-sm100-fallback"].dtype,
            torch.bfloat16,
        )

    def test_fp8_e4m3_weights_on_sm120_are_converted_to_model_dtype(self):
        model_config = MockModelConfig(dtype="bfloat16")
        manager = LoraManager(
            mapping=Mapping(world_size=1, rank=0, tp_size=1),
            model_config=model_config,
            cpp_peft_cache_manager=MagicMock(),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            adapter_dir = Path(tmpdir) / "adapter"
            adapter_dir.mkdir()
            _create_dummy_hf_lora_adapter(
                adapter_dir,
                rank=16,
                num_layers=1,
                dtype=torch.float8_e4m3fn,
            )

            with patch("torch.cuda.get_device_capability", return_value=(12, 0)):
                manager.load_from_hf(
                    model_dirs=[str(adapter_dir)],
                    model_config=model_config,
                    uids=["fp8-e4m3-sm120"],
                )

        self.assertEqual(manager.cpp_lora_weights["fp8-e4m3-sm120"].dtype, torch.bfloat16)

    def test_fp8_e4m3_input_output_dtype_mismatch_is_rejected(self):
        model_config = MockModelConfig()
        manager = LoraManager(
            mapping=Mapping(world_size=1, rank=0, tp_size=1),
            model_config=model_config,
            cpp_peft_cache_manager=MagicMock(),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            adapter_dir = Path(tmpdir) / "adapter"
            adapter_dir.mkdir()
            _create_dummy_hf_lora_adapter(
                adapter_dir,
                rank=16,
                num_layers=1,
                dtype=torch.float8_e4m3fn,
                input_dtype=torch.float8_e5m2,
            )
            self.assertIsNone(HfLoraLoader([str(adapter_dir)]).get_lora_dtype())

            with self.assertRaisesRegex(
                ValueError, "FP8 LoRA input and output weights must have the same dtype"
            ):
                manager.load_from_hf(
                    model_dirs=[str(adapter_dir)],
                    model_config=model_config,
                    uids=["fp8-mismatched-dtype"],
                )

            uid = "fp8-mismatched-dtype"
            self.assertNotIn(uid, manager._cpp_lora_weights)
            self.assertNotIn(uid, manager._cpp_lora_config)
            self.assertNotIn(uid, manager._lora_uid_to_low_ranks)
            self.assertNotIn(uid, manager._lora_weights_pointers_list)

            _create_dummy_hf_lora_adapter(
                adapter_dir,
                rank=16,
                num_layers=1,
                dtype=torch.float8_e4m3fn,
            )
            manager.load_from_hf(
                model_dirs=[str(adapter_dir)],
                model_config=model_config,
                uids=[uid],
            )

            self.assertIn(uid, manager._cpp_lora_weights)
            self.assertIn(uid, manager._cpp_lora_config)
            self.assertIn(uid, manager._lora_uid_to_low_ranks)
            self.assertIn(uid, manager._lora_weights_pointers_list)

    def test_scaled_fp8_e4m3_weights_are_clamped_before_cast(self):
        model_config = MockModelConfig()
        manager = LoraManager(
            mapping=Mapping(world_size=1, rank=0, tp_size=1),
            model_config=model_config,
            cpp_peft_cache_manager=MagicMock(),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            adapter_dir = Path(tmpdir) / "adapter"
            adapter_dir.mkdir()
            _create_dummy_hf_lora_adapter(
                adapter_dir,
                rank=16,
                num_layers=1,
                dtype=torch.float8_e4m3fn,
                lora_alpha=32,
                weight_value=torch.finfo(torch.float8_e4m3fn).max,
            )

            manager.load_from_hf(
                model_dirs=[str(adapter_dir)],
                model_config=model_config,
                uids=["scaled-fp8"],
            )

        weights = manager.cpp_lora_weights["scaled-fp8"].float()
        self.assertTrue(torch.isfinite(weights).all())
        self.assertEqual(weights.abs().max(), torch.finfo(torch.float8_e4m3fn).max)


class TestLoraManagerFp8Alignment(unittest.TestCase):
    @staticmethod
    def _create_manager(model_config):
        return LoraManager(
            mapping=Mapping(world_size=1, rank=0, tp_size=1),
            model_config=model_config,
            cpp_peft_cache_manager=MagicMock(),
        )

    def test_misaligned_fp8_adapter_is_rejected_before_cuda_transfer(self):
        cases = [
            {"rank": 8, "hidden_size": 64, "output_size": 64, "match": "rank=8"},
            {
                "rank": 16,
                "hidden_size": 72,
                "output_size": 64,
                "match": "input size=72",
            },
            {
                "rank": 16,
                "hidden_size": 64,
                "output_size": 72,
                "match": "output size=72",
            },
        ]

        for case in cases:
            with self.subTest(case=case), tempfile.TemporaryDirectory() as tmpdir:
                adapter_dir = Path(tmpdir) / "adapter"
                adapter_dir.mkdir()
                _create_dummy_hf_lora_adapter(
                    adapter_dir,
                    hidden_size=case["hidden_size"],
                    output_size=case["output_size"],
                    rank=case["rank"],
                    num_layers=1,
                    dtype=torch.float8_e4m3fn,
                )
                model_config = MockModelConfig(hidden_size=case["hidden_size"])
                manager = self._create_manager(model_config)

                with (
                    patch(
                        "tensorrt_llm._torch.peft.lora.manager.supports_native_fp8_lora",
                        return_value=True,
                    ),
                    self.assertRaisesRegex(ValueError, case["match"]),
                ):
                    manager.load_from_hf(
                        model_dirs=[str(adapter_dir)],
                        model_config=model_config,
                        uids=["misaligned-fp8"],
                    )


if __name__ == "__main__":
    unittest.main()
