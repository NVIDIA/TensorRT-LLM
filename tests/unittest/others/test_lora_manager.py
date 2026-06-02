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
from unittest.mock import MagicMock

import torch
from safetensors.torch import save_file

from tensorrt_llm.lora_helper import all_false_moe_shared_flags
from tensorrt_llm.lora_manager import LoraManager
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
    adapter_dir: Path, hidden_size: int = 64, rank: int = 8, num_layers: int = 2
):
    """Create a minimal HF-format LoRA adapter on disk."""
    config = {
        "r": rank,
        "lora_alpha": rank,
        "target_modules": ["q_proj", "k_proj", "v_proj"],
        "bias": "none",
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
    }
    with open(adapter_dir / "adapter_config.json", "w") as f:
        json.dump(config, f)

    weights = {}
    for layer_idx in range(num_layers):
        for module in ["q_proj", "k_proj", "v_proj"]:
            prefix = f"base_model.model.model.layers.{layer_idx}.self_attn.{module}"
            weights[f"{prefix}.lora_A.weight"] = torch.randn(rank, hidden_size, dtype=torch.float16)
            weights[f"{prefix}.lora_B.weight"] = torch.randn(hidden_size, rank, dtype=torch.float16)

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


@dataclass
class MockMoeModelConfig:
    """Minimal model config for MoE LoRA tests.

    Mirrors the fields `LoraManager.load_from_hf` reads, with a single MoE
    up-projection target so the fixture stays small.
    """

    lora_target_modules: list = field(
        default_factory=lambda: ["moe_h_to_4h", "moe_gate", "moe_4h_to_h"]
    )
    trtllm_modules_to_hf_modules: dict = field(
        default_factory=lambda: {
            "moe_h_to_4h": "w1",
            "moe_4h_to_h": "w2",
            "moe_gate": "w3",
        }
    )
    hidden_size: int = 64
    dtype: str = "float16"
    swap_gate_up_proj_lora_b_weight: bool = False


def _create_dummy_moe_hf_lora_adapter(
    adapter_dir: Path,
    hidden_size: int = 64,
    intermediate_size: int = 128,
    rank: int = 8,
    num_layers: int = 2,
    num_experts: int = 4,
    make_shared_outer: bool = False,
):
    """Create a minimal HF-format MoE LoRA adapter on disk.

    The on-disk weights are always materialized per-expert, as the C++ LoRA
    cache row size requires. When `make_shared_outer` is True, the "outer" side
    of each module is identical across experts (A for the up-projections w1/w3,
    B for the down-projection w2), as a replicated shared-outer adapter looks on
    disk.
    """
    config = {
        "r": rank,
        "lora_alpha": rank,
        "target_modules": ["w1", "w2", "w3"],
        "bias": "none",
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
    }
    with open(adapter_dir / "adapter_config.json", "w") as f:
        json.dump(config, f)

    weights = {}
    for layer_idx in range(num_layers):
        # Per-layer shared "outer" matrices, cloned per expert so the on-disk
        # slices are bit-identical without sharing storage.
        if make_shared_outer:
            shared_a = {
                "w1": torch.randn(rank, hidden_size, dtype=torch.float16),
                "w3": torch.randn(rank, hidden_size, dtype=torch.float16),
            }
            shared_b_w2 = torch.randn(hidden_size, rank, dtype=torch.float16)
        for expert_idx in range(num_experts):
            base = f"base_model.model.model.layers.{layer_idx}.mlp.experts.{expert_idx}"
            # Up-projections (w1, w3): in_dim=hidden, out_dim=intermediate.
            for module in ("w1", "w3"):
                if make_shared_outer:
                    a = shared_a[module].clone()
                else:
                    a = torch.randn(rank, hidden_size, dtype=torch.float16)
                weights[f"{base}.{module}.lora_A.weight"] = a
                weights[f"{base}.{module}.lora_B.weight"] = torch.randn(
                    intermediate_size, rank, dtype=torch.float16
                )
            # Down-projection (w2): in_dim=intermediate, out_dim=hidden.
            weights[f"{base}.w2.lora_A.weight"] = torch.randn(
                rank, intermediate_size, dtype=torch.float16
            )
            if make_shared_outer:
                b_w2 = shared_b_w2.clone()
            else:
                b_w2 = torch.randn(hidden_size, rank, dtype=torch.float16)
            weights[f"{base}.w2.lora_B.weight"] = b_w2

    save_file(weights, str(adapter_dir / "adapter_model.safetensors"))


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestLoraManagerMoeSharedFlags(unittest.TestCase):
    """Tests for MoE shared-outer detection in the HF LoRA loader."""

    def _create_manager(self):
        # cpp_peft_cache_manager mocked so the manager skips the
        # legacy-TRT retain-GPU-tensors path; matches PyTorch runtime usage.
        mapping = Mapping(world_size=1, rank=0, tp_size=1)
        model_config = MockMoeModelConfig()
        return LoraManager(
            mapping=mapping,
            model_config=model_config,
            cpp_peft_cache_manager=MagicMock(),
        )

    def test_per_expert_weights_yield_all_false_moe_shared_flags(self):
        # Per-expert random weights: detection finds no shared side, so the
        # flags stay all-False (the default per-expert kernel path).
        manager = self._create_manager()
        model_config = MockMoeModelConfig()
        with tempfile.TemporaryDirectory() as tmpdir:
            adapter_dir = Path(tmpdir) / "adapter_per_expert"
            adapter_dir.mkdir()
            _create_dummy_moe_hf_lora_adapter(adapter_dir)
            manager.load_from_hf(
                model_dirs=[str(adapter_dir)],
                model_config=model_config,
                uids=["uid-per-expert"],
            )
        self.assertEqual(
            manager.get_moe_shared_flags("uid-per-expert"), all_false_moe_shared_flags()
        )

    def test_unknown_uid_returns_all_false(self):
        manager = self._create_manager()
        self.assertEqual(manager.get_moe_shared_flags("never-loaded"), all_false_moe_shared_flags())

    def test_shared_outer_flags_auto_detected_from_weights(self):
        # The loader detects the shared-outer side from the replicated expert
        # slices and builds the kernel flags.
        manager = self._create_manager()
        model_config = MockMoeModelConfig()
        with tempfile.TemporaryDirectory() as tmpdir:
            adapter_dir = Path(tmpdir) / "adapter_shared_outer_auto"
            adapter_dir.mkdir()
            _create_dummy_moe_hf_lora_adapter(adapter_dir, make_shared_outer=True)
            manager.load_from_hf(
                model_dirs=[str(adapter_dir)],
                model_config=model_config,
                uids=["uid-auto"],
            )
        # Up-projections (w1=moe_h_to_4h, w3=moe_gate) share A -> gated/fc1
        # _shared_a; down-projection (w2=moe_4h_to_h) shares B -> fc2_shared_b.
        self.assertEqual(
            manager.get_moe_shared_flags("uid-auto"),
            {
                "gated_shared_a": True,
                "fc1_shared_a": True,
                "fc2_shared_b": True,
            },
        )


if __name__ == "__main__":
    unittest.main()
