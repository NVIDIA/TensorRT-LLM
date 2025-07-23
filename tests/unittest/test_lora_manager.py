# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import io
import json
import os
import tarfile
import tempfile
import unittest
import warnings
from unittest import mock

import torch
import yaml
from safetensors import torch as safetensors_torch

# Import the modules to test
from tensorrt_llm import lora_manager, mapping

# Constants
DEFAULT_HIDDEN_SIZE = 4096
DEFAULT_RANK = 32
DEFAULT_NUM_LAYERS = 4
DEFAULT_TEST_RANK = 16


class TestLoraManagerBase(unittest.TestCase):
    """Base class with common functionality for LoRA manager tests."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.lora_manager = lora_manager.LoraManager()

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _create_incomplete_hf_checkpoint(self, missing_matrices=None):
        """
        Create an incomplete HF LoRA checkpoint for testing.

        Args:
            missing_matrices: List of matrix types to exclude (e.g., ['q_proj.lora_A'])
                            If None, defaults to ['q_proj.lora_A', 'v_proj.lora_A']

        Returns:
            str: Path to the created checkpoint directory
        """
        if missing_matrices is None:
            missing_matrices = ['q_proj.lora_A', 'v_proj.lora_A']

        # Create adapter_config.json
        adapter_config = {
            "r": 32,
            "lora_alpha": 32,
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
            "lora_dropout": 0.05,
            "bias": "none",
            "task_type": "CAUSAL_LM"
        }

        config_path = os.path.join(self.temp_dir, "adapter_config.json")
        with open(config_path, 'w') as f:
            json.dump(adapter_config, f)

        # Create incomplete weight tensors
        weights = {}
        hidden_size = DEFAULT_HIDDEN_SIZE
        rank = DEFAULT_RANK
        num_layers = DEFAULT_NUM_LAYERS  # Use fewer layers for faster tests

        for layer_idx in range(num_layers):
            layer_prefix = f"base_model.model.model.layers.{layer_idx}.self_attn"

            # Add weights for all modules except missing ones
            for module in ["q_proj", "k_proj", "v_proj", "o_proj"]:
                for matrix_type in ["lora_A", "lora_B"]:
                    key = f"{layer_prefix}.{module}.{matrix_type}.weight"

                    # Skip missing matrices
                    if any(missing in key for missing in missing_matrices):
                        continue

                    if matrix_type == "lora_A":
                        shape = (rank, hidden_size)
                    else:  # lora_B
                        shape = (hidden_size, rank)

                    weights[key] = torch.randn(shape, dtype=torch.float16)

        # Save to safetensors
        safetensors_path = os.path.join(self.temp_dir,
                                        "adapter_model.safetensors")
        safetensors_torch.save_file(weights, safetensors_path)

        return self.temp_dir

    def _create_hf_model_config(self):
        """Create a LoraModelConfig for HF testing."""
        return self._create_model_config()

    def _create_model_config(self, target_modules=None):
        """Create a LoraModelConfig for testing (backward compatibility)."""
        if target_modules is None:
            target_modules = ['attn_q', 'attn_k', 'attn_v', 'attn_dense']

        return lora_manager.LoraModelConfig(lora_target_modules=target_modules,
                                            trtllm_modules_to_hf_modules={
                                                'attn_q': 'q_proj',
                                                'attn_k': 'k_proj',
                                                'attn_v': 'v_proj',
                                                'attn_dense': 'o_proj'
                                            },
                                            hidden_size=DEFAULT_HIDDEN_SIZE,
                                            dtype='float16')

    def _create_incomplete_nemo_checkpoint(self,
                                           missing_matrices=None,
                                           include_rank_in_config=True):
        """
        Create a NeMo LoRA checkpoint (.nemo archive) for testing.

        Args:
            missing_matrices: Dict mapping layer_idx to list of missing matrices
                            e.g., {0: ['in'], 1: ['out']}
            include_rank_in_config: Whether to include adapter_dim in the config

        Returns:
            str: Path to the created .nemo file
        """
        if missing_matrices is None:
            missing_matrices = {}

        # Create model config
        model_config = {
            "target_modules": ["self_attention.adapter_layer.lora_kqv_adapter"],
            "hidden_size": DEFAULT_HIDDEN_SIZE,
            "num_layers": DEFAULT_NUM_LAYERS
        }

        # Conditionally add lora_tuning with adapter_dim
        if include_rank_in_config:
            model_config["lora_tuning"] = {
                "adapter_dim": DEFAULT_RANK,  # This will be used as the rank
                "target_modules": ["attention_qkv"],
                "alpha": DEFAULT_RANK
            }
        else:
            # Create lora_tuning without adapter_dim to test default rank fallback
            model_config["lora_tuning"] = {
                "target_modules": ["attention_qkv"],
                "alpha": DEFAULT_RANK
            }

        # Create model weights
        model_weights = {}
        rank = DEFAULT_RANK
        hidden_size = DEFAULT_HIDDEN_SIZE

        for layer_idx in range(DEFAULT_NUM_LAYERS):
            layer_missing = missing_matrices.get(layer_idx, [])

            # Add 'in' matrix unless it's marked as missing
            if 'in' not in layer_missing:
                key = f"model.layers.{layer_idx}.self_attention.adapter_layer.lora_kqv_adapter.linear_in.weight"
                model_weights[key] = torch.randn(rank,
                                                 hidden_size,
                                                 dtype=torch.float16)

            # Add 'out' matrix unless it's marked as missing
            if 'out' not in layer_missing:
                key = f"model.layers.{layer_idx}.self_attention.adapter_layer.lora_kqv_adapter.linear_out.weight"
                # NeMo fused QKV is 3x larger
                model_weights[key] = torch.randn(3 * hidden_size,
                                                 rank,
                                                 dtype=torch.float16)

        # Create .nemo archive
        nemo_path = os.path.join(self.temp_dir, "test_lora.nemo")

        with tarfile.open(nemo_path, 'w') as tar:
            # Add model_config.yaml
            config_str = yaml.dump(model_config)
            config_info = tarfile.TarInfo('model_config.yaml')
            config_info.size = len(config_str.encode())
            tar.addfile(config_info, io.BytesIO(config_str.encode()))

            # Add model_weights.ckpt
            weights_buffer = io.BytesIO()
            torch.save(model_weights, weights_buffer)
            weights_data = weights_buffer.getvalue()

            weights_info = tarfile.TarInfo('model_weights.ckpt')
            weights_info.size = len(weights_data)
            tar.addfile(weights_info, io.BytesIO(weights_data))

        return nemo_path

    def _create_nemo_model_config(self):
        """Create a LoraModelConfig for NeMo testing."""
        return lora_manager.LoraModelConfig(
            lora_target_modules=['attn_qkv'],
            trtllm_modules_to_hf_modules={'attn_qkv': 'attn_qkv'},
            hidden_size=DEFAULT_HIDDEN_SIZE,
            dtype='float16')

    def test_missing_matrices_graceful_handling(self):
        """Test for graceful handling of missing matrices across checkpoint formats."""
        test_cases = [
            # HF test cases
            ("hf", ["q_proj.lora_A",
                    "v_proj.lora_A"], ["q_proj", "v_proj", "missing", "in"]),
            ("hf", ["k_proj.lora_B"], ["k_proj", "missing", "out"]),
            ("hf", ["q_proj.lora_A", "k_proj.lora_B",
                    "v_proj.lora_A"], ["missing"]),
            # NeMo test cases
            ("nemo", {
                0: ["in"]
            }, ["Layer 0", "missing", "in"]),
            ("nemo", {
                1: ["out"]
            }, ["Layer 1", "missing", "out"]),
            ("nemo", {
                0: ["in", "out"]
            }, ["missing"]),
        ]

        for ckpt_source, missing_matrices, expected_warnings in test_cases:
            with self.subTest(ckpt_source=ckpt_source,
                              missing_matrices=missing_matrices):
                if ckpt_source == "hf":
                    checkpoint_path = self._create_incomplete_hf_checkpoint(
                        missing_matrices)
                    model_config = self._create_hf_model_config()
                else:  # nemo
                    checkpoint_path = self._create_incomplete_nemo_checkpoint(
                        missing_matrices)
                    model_config = self._create_nemo_model_config()

                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")

                    uids = self.lora_manager.load_from_ckpt(
                        model_dirs_or_files=[checkpoint_path],
                        model_config=model_config,
                        runtime_mapping=mapping.Mapping(),
                        ckpt_source=ckpt_source)

                    # Should successfully return UIDs
                    self.assertEqual(len(uids), 1)

                    # Should have generated appropriate warnings
                    warning_text = ' '.join(
                        [str(warning.message) for warning in w])
                    for expected in expected_warnings:
                        self.assertIn(
                            expected, warning_text,
                            f"Expected '{expected}' in warning text for {ckpt_source} checkpoint"
                        )

    def test_complete_checkpoints_no_warnings(self):
        """Test that complete checkpoints load without warnings."""
        test_cases = ["hf", "nemo"]

        for ckpt_source in test_cases:
            with self.subTest(ckpt_source=ckpt_source):
                if ckpt_source == "hf":
                    checkpoint_path = self._create_incomplete_hf_checkpoint(
                        [])  # No missing matrices
                    model_config = self._create_hf_model_config()
                else:  # nemo
                    checkpoint_path = self._create_incomplete_nemo_checkpoint(
                        {})  # No missing matrices
                    model_config = self._create_nemo_model_config()

                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")

                    uids = self.lora_manager.load_from_ckpt(
                        model_dirs_or_files=[checkpoint_path],
                        model_config=model_config,
                        runtime_mapping=mapping.Mapping(),
                        ckpt_source=ckpt_source)

                    self.assertEqual(len(uids), 1)

                    # Should not have any warnings about missing matrices
                    missing_warnings = [
                        warning for warning in w
                        if 'missing' in str(warning.message)
                    ]
                    self.assertEqual(
                        len(missing_warnings), 0,
                        f"Complete {ckpt_source} checkpoint should not generate missing matrix warnings"
                    )


class TestLoraManagerSpecificFeatures(TestLoraManagerBase):
    """Tests for specific features that are unique to each checkpoint format."""

    def test_hf_zero_tensor_dimensions(self):
        """Test HF-specific zero tensor dimensions (separate Q/K/V modules)."""
        checkpoint_dir = self._create_incomplete_hf_checkpoint(
            ['q_proj.lora_A'])
        model_config = self._create_model_config(['attn_q'
                                                  ])  # Only test one module

        # Mock the zero tensor creation to verify dimensions
        original_zeros = torch.zeros
        created_tensors = []

        def mock_zeros(*args, **kwargs):
            tensor = original_zeros(*args, **kwargs)
            created_tensors.append(tensor.shape)
            return tensor

        with mock.patch('torch.zeros', side_effect=mock_zeros):
            self.lora_manager.load_from_ckpt(
                model_dirs_or_files=[checkpoint_dir],
                model_config=model_config,
                runtime_mapping=mapping.Mapping(),
                ckpt_source='hf')

        # Should have created zero tensors for missing matrices
        self.assertGreater(
            len(created_tensors), 0,
            "Should have created zero tensors for missing matrices")

        # Verify HF tensor dimensions (rank=32, hidden_size=4096 for lora_A)
        expected_shape = (DEFAULT_RANK, DEFAULT_HIDDEN_SIZE
                          )  # lora_A dimensions
        self.assertIn(
            expected_shape, created_tensors,
            f"Expected HF lora_A tensor shape {expected_shape} to be created")

    def test_nemo_zero_tensor_dimensions(self):
        """Test NeMo-specific zero tensor dimensions (fused QKV - 3x larger output)."""
        # Create checkpoint without rank in config to use default rank (64)
        nemo_path = self._create_incomplete_nemo_checkpoint(
            {0: ['in', 'out']}, include_rank_in_config=False)
        model_config = self._create_nemo_model_config()

        # Mock the zero tensor creation to verify dimensions
        original_zeros = torch.zeros
        created_tensors = []

        def mock_zeros(*args, **kwargs):
            tensor = original_zeros(*args, **kwargs)
            created_tensors.append(tensor.shape)
            return tensor

        with mock.patch('torch.zeros', side_effect=mock_zeros):
            self.lora_manager.load_from_ckpt(model_dirs_or_files=[nemo_path],
                                             model_config=model_config,
                                             runtime_mapping=mapping.Mapping(),
                                             ckpt_source='nemo')

        # Should have created zero tensors
        self.assertGreater(
            len(created_tensors), 0,
            "Should have created zero tensors for missing matrices")

        # Verify NeMo tensor dimensions (rank=32 from other layers, hidden_size=4096, 3x for fused QKV)
        expected_in_shape = (DEFAULT_RANK, DEFAULT_HIDDEN_SIZE
                             )  # 'in' matrix (lora_A equivalent)
        expected_out_shape = (3 * DEFAULT_HIDDEN_SIZE, DEFAULT_RANK
                              )  # 'out' matrix (3x larger for fused QKV)

        self.assertIn(
            expected_in_shape, created_tensors,
            f"Expected NeMo 'in' tensor shape {expected_in_shape} to be created"
        )
        self.assertIn(
            expected_out_shape, created_tensors,
            f"Expected NeMo 'out' tensor shape {expected_out_shape} to be created"
        )

    def test_nemo_rank_derivation_from_config_and_tensors(self):
        """Test NeMo-specific rank derivation: from config first, then from existing tensors."""
        # Create checkpoint with custom rank where only 'in' is missing
        rank = DEFAULT_TEST_RANK
        hidden_size = DEFAULT_HIDDEN_SIZE

        # Manually create model weights with custom rank
        model_weights = {
            f"model.layers.0.self_attention.adapter_layer.lora_kqv_adapter.linear_out.weight":
            torch.randn(3 * hidden_size, rank, dtype=torch.float16)
        }

        # Create .nemo archive
        nemo_path = os.path.join(self.temp_dir, "custom_rank.nemo")
        model_config_dict = {
            "lora_tuning": {
                "adapter_dim": rank,  # This should be used as primary source
                "target_modules": ["attention_qkv"]
            },
            "hidden_size": hidden_size
        }

        with tarfile.open(nemo_path, 'w') as tar:
            # Add config
            config_str = yaml.dump(model_config_dict)
            config_info = tarfile.TarInfo('model_config.yaml')
            config_info.size = len(config_str.encode())
            tar.addfile(config_info, io.BytesIO(config_str.encode()))

            # Add weights
            weights_buffer = io.BytesIO()
            torch.save(model_weights, weights_buffer)
            weights_data = weights_buffer.getvalue()

            weights_info = tarfile.TarInfo('model_weights.ckpt')
            weights_info.size = len(weights_data)
            tar.addfile(weights_info, io.BytesIO(weights_data))

        model_config = self._create_nemo_model_config()

        # Mock zero tensor creation to verify correct rank is used
        created_tensors = []
        original_zeros = torch.zeros

        def mock_zeros(*args, **kwargs):
            tensor = original_zeros(*args, **kwargs)
            created_tensors.append(tensor.shape)
            return tensor

        with mock.patch('torch.zeros', side_effect=mock_zeros):
            self.lora_manager.load_from_ckpt(model_dirs_or_files=[nemo_path],
                                             model_config=model_config,
                                             runtime_mapping=mapping.Mapping(),
                                             ckpt_source='nemo')

        # Should have created 'in' tensor with rank from config (not derived from existing tensor)
        expected_in_shape = (rank, hidden_size)
        self.assertIn(
            expected_in_shape, created_tensors,
            f"Expected 'in' tensor with config rank {rank} to be created")

    def test_hf_original_typerror_regression(self):
        """Test HF-specific: Ensures original TypeError bug doesn't regress."""
        checkpoint_dir = self._create_incomplete_hf_checkpoint(
            ['q_proj.lora_A'])
        model_config = self._create_model_config(['attn_q'])

        # This test verifies that the current implementation handles the case gracefully
        # Before the fix, this would have raised: TypeError: new(): invalid data type 'str'
        try:
            uids = self.lora_manager.load_from_ckpt(
                model_dirs_or_files=[checkpoint_dir],
                model_config=model_config,
                runtime_mapping=mapping.Mapping(),
                ckpt_source='hf')
            # Should succeed with the fix in place
            self.assertEqual(len(uids), 1)
        except TypeError as e:
            if "invalid data type 'str'" in str(e):
                self.fail(
                    "The original TypeError bug has regressed - the fix is not working"
                )
            else:
                # Some other TypeError, re-raise
                raise

    def test_nemo_default_rank_fallback(self):
        """Test NeMo-specific: Fallback to default rank when both config and tensors unavailable."""
        # Create checkpoint without rank in config and ALL layers missing matrices to trigger default rank fallback
        missing_all_layers = {
            i: ['in', 'out']
            for i in range(DEFAULT_NUM_LAYERS)
        }
        nemo_path = self._create_incomplete_nemo_checkpoint(
            missing_all_layers, include_rank_in_config=False)
        model_config = self._create_nemo_model_config()

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            uids = self.lora_manager.load_from_ckpt(
                model_dirs_or_files=[nemo_path],
                model_config=model_config,
                runtime_mapping=mapping.Mapping(),
                ckpt_source='nemo')

            self.assertEqual(len(uids), 1)

            # Should have warnings for both missing matrices AND default rank usage
            missing_warnings = [
                warning for warning in w if 'missing' in str(warning.message)
            ]
            self.assertGreaterEqual(
                len(missing_warnings), 2,
                "Expected warnings for both missing matrices")

            # Should also have a warning about using default rank
            rank_warnings = [
                warning for warning in w
                if 'default rank' in str(warning.message)
            ]
            self.assertGreater(len(rank_warnings), 0,
                               "Expected warning about using default rank")


if __name__ == '__main__':
    unittest.main()
