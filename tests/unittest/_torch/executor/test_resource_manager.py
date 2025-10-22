import os
import pathlib
import subprocess
import sys
import unittest
from typing import NamedTuple, Tuple
from unittest.mock import patch

import numpy as np
import torch

import tensorrt_llm
import tensorrt_llm.bindings
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest
from tensorrt_llm._torch.pyexecutor.resource_manager import (KVCacheManager,
                                                             PeftCacheManager)
from tensorrt_llm.bindings import LayerType
from tensorrt_llm.bindings import ModelConfig as ModelConfigCpp
from tensorrt_llm.bindings import executor as tllm
from tensorrt_llm.bindings.internal.batch_manager import \
    PeftTaskNotCachedException
from tensorrt_llm.llmapi.llm_args import KvCacheConfig, PeftCacheConfig
from tensorrt_llm.lora_helper import LoraConfig
from tensorrt_llm.mapping import Mapping

DataType = tensorrt_llm.bindings.DataType
LoraModule = tensorrt_llm.bindings.LoraModule
LoraModuleType = tensorrt_llm.bindings.LoraModuleType
current_dir = pathlib.Path(__file__).parent.resolve()
root_dir = current_dir.parent.parent.parent.parent

sys.path.append(str(root_dir / "tests" / "integration"))


class TestResourceManager(unittest.TestCase):
    CPP_RESOURCES_DIR = os.path.join(str(root_dir), "cpp", "tests", "resources")
    CPP_DATA_DIR = os.path.join(CPP_RESOURCES_DIR, "data")
    LORA_TEST_WEIGHTS_TP1 = "lora-test-weights-tp1"
    TP1_WEIGHTS_PATH = os.path.join(CPP_DATA_DIR, LORA_TEST_WEIGHTS_TP1,
                                    "source.npy")
    TP1_CONFIG_PATH = os.path.join(CPP_DATA_DIR, LORA_TEST_WEIGHTS_TP1,
                                   "config.npy")

    @classmethod
    def setUpClass(cls):
        """
        Setup the lora test data resources
        """
        cpp_script_dir = os.path.join(cls.CPP_RESOURCES_DIR, "scripts")

        # No reason to run this script for each test.
        # TODO: move this to a fixture that runs once.
        generate_lora_data_args_tp1 = [
            sys.executable,
            f"{cpp_script_dir}/generate_test_lora_weights.py",
            f"--out-dir={cls.CPP_DATA_DIR}/{cls.LORA_TEST_WEIGHTS_TP1}",
            "--tp-size=1",
        ]

        subprocess.check_call(generate_lora_data_args_tp1,
                              cwd=root_dir,
                              shell=False,
                              env=None,
                              timeout=100)

    class MockModelConfig:
        """
        Mock model config for testing purposes. Using values defined in peftCacheManagerTest.cpp
        """

        def __init__(self):
            self.vocab_size = 0
            self.num_hidden_layers = 2
            self.num_attention_layers = 2
            self.num_rnn_layers = 0
            self.num_attention_heads = 1
            self.hidden_size = 16
            self.data_type = DataType.HALF

        @property
        def num_kv_heads_per_layer(self):
            return [self.num_attention_heads] * self.num_attention_layers

        @property
        def head_size(self):
            return self.hidden_size // self.num_attention_heads

    class MockPeftCacheManagerConfig:
        """
        Mock PeftCacheManagerConfig that mirrors the C++ test configuration.
        C++ code used:
        """

        def __init__(self):
            self.num_host_module_layer = 2 * 8 * 128
            self.num_device_module_layer = 2 * 8 * 92
            self.max_pages_per_block_host = 8
            self.max_pages_per_block_device = 8
            self.max_adapter_size = 64
            self.put_thread_count = 1
            self.ensure_thread_count = 1
            self.optimal_adapter_size = 8

    def setUp(self):
        mock_config = self.MockModelConfig()
        self.model_config = ModelConfigCpp(
            vocab_size=mock_config.vocab_size,
            num_layers=mock_config.num_hidden_layers,
            num_attention_layers=mock_config.num_attention_layers,
            num_rnn_layers=mock_config.num_rnn_layers,
            num_heads=mock_config.num_attention_heads,
            hidden_size=mock_config.hidden_size,
            data_type=mock_config.data_type)
        self._set_lora_modules()
        self.model_config.use_lora_plugin = True
        self.model_config.max_lora_rank = 64
        self.max_lora_rank = 64
        self.max_cpu_loras = 4
        self.max_loras = 4
        self.num_lora_modules = (mock_config.num_hidden_layers *
                                 len(self.model_config.lora_modules))

    def _set_lora_modules(self):
        lora_modules = [
            LoraModule(module_type=LoraModuleType.ATTN_QKV,
                       in_dim=16,
                       out_dim=3 * 16,
                       in_dim_first=False,
                       out_dim_first=True,
                       in_tp_split_dim=-1,
                       out_tp_split_dim=0),
            LoraModule(module_type=LoraModuleType.ATTN_Q,
                       in_dim=16,
                       out_dim=16,
                       in_dim_first=False,
                       out_dim_first=True,
                       in_tp_split_dim=-1,
                       out_tp_split_dim=0),
            LoraModule(module_type=LoraModuleType.ATTN_K,
                       in_dim=16,
                       out_dim=16,
                       in_dim_first=False,
                       out_dim_first=True,
                       in_tp_split_dim=-1,
                       out_tp_split_dim=0),
            LoraModule(module_type=LoraModuleType.ATTN_V,
                       in_dim=16,
                       out_dim=16,
                       in_dim_first=False,
                       out_dim_first=True,
                       in_tp_split_dim=-1,
                       out_tp_split_dim=0),
            LoraModule(module_type=LoraModuleType.ATTN_DENSE,
                       in_dim=16,
                       out_dim=16,
                       in_dim_first=False,
                       out_dim_first=True,
                       in_tp_split_dim=1,
                       out_tp_split_dim=-1),
            LoraModule(module_type=LoraModuleType.MLP_H_TO_4H,
                       in_dim=16,
                       out_dim=32,
                       in_dim_first=False,
                       out_dim_first=True,
                       in_tp_split_dim=-1,
                       out_tp_split_dim=0),
            LoraModule(module_type=LoraModuleType.MLP_4H_TO_H,
                       in_dim=32,
                       out_dim=16,
                       in_dim_first=False,
                       out_dim_first=True,
                       in_tp_split_dim=1,
                       out_tp_split_dim=-1),
            LoraModule(module_type=LoraModuleType.MLP_GATE,
                       in_dim=16,
                       out_dim=32,
                       in_dim_first=False,
                       out_dim_first=True,
                       in_tp_split_dim=-1,
                       out_tp_split_dim=0),
            LoraModule(module_type=LoraModuleType.CROSS_ATTN_QKV,
                       in_dim=16,
                       out_dim=3 * 16,
                       in_dim_first=False,
                       out_dim_first=True,
                       in_tp_split_dim=-1,
                       out_tp_split_dim=0),
            LoraModule(module_type=LoraModuleType.CROSS_ATTN_Q,
                       in_dim=16,
                       out_dim=16,
                       in_dim_first=False,
                       out_dim_first=True,
                       in_tp_split_dim=-1,
                       out_tp_split_dim=0),
            LoraModule(module_type=LoraModuleType.CROSS_ATTN_K,
                       in_dim=16,
                       out_dim=16,
                       in_dim_first=False,
                       out_dim_first=True,
                       in_tp_split_dim=-1,
                       out_tp_split_dim=0),
            LoraModule(module_type=LoraModuleType.CROSS_ATTN_V,
                       in_dim=16,
                       out_dim=16,
                       in_dim_first=False,
                       out_dim_first=True,
                       in_tp_split_dim=-1,
                       out_tp_split_dim=0),
            LoraModule(module_type=LoraModuleType.CROSS_ATTN_DENSE,
                       in_dim=16,
                       out_dim=16,
                       in_dim_first=False,
                       out_dim_first=True,
                       in_tp_split_dim=1,
                       out_tp_split_dim=-1),
        ]

        self.model_config.lora_modules = lora_modules

    def create_peft_cache_config(self) -> PeftCacheConfig:
        # Use the exact same values from C++ test

        mock_config = self.MockPeftCacheManagerConfig()

        # Create the PeftCacheConfig with parameter names that match the expected API
        peft_cache_config = tllm.PeftCacheConfig(
            num_host_module_layer=mock_config.num_host_module_layer,
            num_device_module_layer=mock_config.num_device_module_layer,
            optimal_adapter_size=mock_config.optimal_adapter_size,
            max_pages_per_block_host=mock_config.max_pages_per_block_host,
            max_adapter_size=mock_config.max_adapter_size,
            num_put_workers=mock_config.put_thread_count,
            num_ensure_workers=mock_config.ensure_thread_count,
        )

        return PeftCacheConfig.from_pybind(peft_cache_config)

    def _create_request(self,
                        request_id,
                        task_id=None,
                        lora_weights=None,
                        lora_config=None,
                        max_new_tokens=1):
        """Create a properly structured LlmRequest with optional task_id."""
        sampling_params = tensorrt_llm.sampling_params.SamplingParams()
        sampling_config = tensorrt_llm.bindings.SamplingConfig(
            sampling_params._get_sampling_config())

        # Convert NumPy arrays to PyTorch tensors if they are provided
        if lora_weights is not None:
            lora_weights = torch.from_numpy(lora_weights)

        if lora_config is not None:
            lora_config = torch.from_numpy(lora_config)

        input_tokens = [i + 1 for i in range(max_new_tokens)]
        request = LlmRequest(
            request_id=request_id,
            max_new_tokens=max_new_tokens,
            input_tokens=input_tokens,
            sampling_config=sampling_config,
            is_streaming=False,
            lora_task_id=task_id,
            lora_weights=lora_weights,
            lora_config=lora_config,
        )

        return request

    def get_lora_data(self):
        """Create mock LoRA weights and config.

        Returns:
            tuple: (weights tensor, config tensor).
        """
        lora_weights = np.load(self.TP1_WEIGHTS_PATH).astype(np.float16)
        lora_config = np.load(self.TP1_CONFIG_PATH)
        return lora_weights, lora_config

    def test_successful_mocked_peft_cache_manager_initialization(self):
        peft_cache_config = self.create_peft_cache_config()

        peft_cache_manager = PeftCacheManager(
            peft_cache_config=peft_cache_config,
            lora_config=LoraConfig(),
            model_config=self.model_config,
        )

        self.assertTrue(peft_cache_manager.impl.enabled)
        self.assertGreaterEqual(peft_cache_manager.impl.max_host_pages, 1)
        self.assertGreaterEqual(peft_cache_manager.impl.max_device_pages, 1)

    def test_add_request_peft_empty_weights_config(self):
        """Test adding a request with empty LoRA task."""
        peft_cache_config = self.create_peft_cache_config()

        peft_cache_manager = PeftCacheManager(
            peft_cache_config=peft_cache_config,
            lora_config=LoraConfig(),
            model_config=self.model_config,
        )

        request_id = 0
        task_id = 1
        request = self._create_request(request_id=request_id, task_id=task_id)

        with self.assertRaises(PeftTaskNotCachedException):
            print("Adding request without LoRA task ID")
            peft_cache_manager.add_request_peft(request)

    def test_add_request_peft_empty_batch(self):
        """Test adding a request with empty batch."""
        peft_cache_config = self.create_peft_cache_config()

        peft_cache_manager = PeftCacheManager(
            peft_cache_config=peft_cache_config,
            lora_config=LoraConfig(),
            model_config=self.model_config,
        )

        empty_context = []
        empty_generation = []
        expected_empty_table = peft_cache_manager.ensure_batch(
            empty_context, empty_generation)
        self.assertEqual(expected_empty_table, {})

    def test_add_request_peft(self):
        """Test adding a request with properly configured LoRA weights and config."""
        peft_cache_config = self.create_peft_cache_config()

        peft_cache_manager = PeftCacheManager(
            peft_cache_config=peft_cache_config,
            lora_config=LoraConfig(),
            model_config=self.model_config,
        )

        request_id = 3
        task_id = 1
        weights, config = self.get_lora_data()
        lora_request = self._create_request(request_id=request_id,
                                            task_id=task_id,
                                            lora_weights=weights,
                                            lora_config=config)

        peft_cache_manager.add_request_peft(lora_request)

        context_batch = []
        context_batch.append(lora_request)
        generation_batch = []

        result = peft_cache_manager.ensure_batch(context_batch,
                                                 generation_batch)

        self.assertIsNotNone(result)
        self.assertIn(request_id, result)

        peft_table = result[request_id]

        self.assertEqual(len(peft_table), self.num_lora_modules)

    def test_put_get(self):
        """Test adding a request with properly configured LoRA weights and config."""
        peft_cache_config = self.create_peft_cache_config()

        peft_cache_manager = PeftCacheManager(
            peft_cache_config=peft_cache_config,
            lora_config=LoraConfig(),
            model_config=self.model_config,
        )

        request_id = 0
        task_id = 1234
        weights, config = self.get_lora_data()
        lora_request = self._create_request(
            request_id=request_id,
            task_id=task_id,
            lora_weights=weights,
            lora_config=config,
            max_new_tokens=4,
        )

        peft_cache_manager.add_request_peft(lora_request)

        context_batch = []
        context_batch.append(lora_request)
        generation_batch = []

        result = peft_cache_manager.ensure_batch(context_batch,
                                                 generation_batch)

        self.assertIsNotNone(result)
        self.assertIn(request_id, result)

        peft_table = result[request_id]

        expected_values = [
            # Format: (pageId, slotIdx, inSize, outSize, moduleId, layerId, adapterSize, numSlots)
            (0, 0, 128, 384, 0, 0, 8, 16),
            (0, 16, 128, 384, 0, 1, 8, 16),
            (0, 32, 64, 64, 1, 0, 4, 4),
            (0, 36, 64, 64, 1, 1, 4, 4),
            (0, 40, 64, 64, 2, 0, 4, 4),
            (0, 44, 64, 64, 2, 1, 4, 4),
            (0, 48, 64, 64, 3, 0, 4, 4),
            (0, 52, 64, 64, 3, 1, 4, 4),
            (0, 56, 128, 128, 4, 0, 8, 8),
            (0, 64, 128, 128, 4, 1, 8, 8),
            (0, 72, 128, 256, 5, 0, 8, 12),
            (0, 84, 128, 256, 5, 1, 8, 12),
            (0, 96, 256, 128, 6, 0, 8, 12),
            (0, 108, 256, 128, 6, 1, 8, 12),
            (0, 120, 128, 256, 7, 0, 8, 12),
            (0, 132, 128, 256, 7, 1, 8, 12),
            (0, 144, 128, 384, 8, 0, 8, 16),
            (0, 160, 128, 384, 8, 1, 8, 16),
            (0, 176, 64, 64, 9, 0, 4, 4),
            (0, 180, 64, 64, 9, 1, 4, 4),
            (0, 184, 64, 64, 10, 0, 4, 4),
            (0, 188, 64, 64, 10, 1, 4, 4),
            (0, 192, 64, 64, 11, 0, 4, 4),
            (0, 196, 64, 64, 11, 1, 4, 4),
            (0, 200, 128, 128, 12, 0, 8, 8),
            (0, 208, 128, 128, 12, 1, 8, 8),
        ]

        # Verify number of entries matches expected
        self.assertEqual(len(peft_table), len(expected_values))

        for i, entry in enumerate(peft_table):
            self.assertEqual(entry.page_id, expected_values[i][0])
            self.assertEqual(entry.slot_idx, expected_values[i][1])
            self.assertEqual(entry.in_size, expected_values[i][2])
            self.assertEqual(entry.out_size, expected_values[i][3])
            self.assertEqual(entry.module_id, expected_values[i][4])
            self.assertEqual(entry.layer_id, expected_values[i][5])
            self.assertEqual(entry.adapter_size, expected_values[i][6])
            self.assertEqual(entry.num_slots, expected_values[i][7])

    def test_adjust_window_sizes_for_vswa(self):
        window_size_to_layers = {
            100: [0, 1, 2, 3],
            200: [4, 5, 6],
            7000: [7, 8],
        }
        max_attention_window_vec = [100] * 4 + [200] * 3 + [7000] * 2

        model_config = self.MockModelConfig()
        model_config.num_attention_heads = 2
        model_config.hidden_size = 2
        model_config.data_type = DataType.HALF

        total_layers = [
            i for layers in window_size_to_layers.values() for i in layers
        ]

        model_config.num_hidden_layers = len(total_layers)
        model_config.num_attention_layers = len(total_layers)

        kv_factor = 2
        cache_bytes_per_token_per_layer = 8

        # Define test cases:
        #    (memory_bytes, expected_window_sizes, max_tokens, description)
        #    If max_tokens is None, then it will use the default value of KvCacheConfig.
        test_cases = [
            (
                # Case 1: Limited memory - windows get clamped
                cache_bytes_per_token_per_layer * (100 * 9 + 30 * 5) + 4,
                {
                    100: [0, 1, 2, 3],
                    130: [4, 5, 6, 7, 8],
                },
                [100] * 4 + [130] * 5,
                None,
                "limited_memory_clamped_windows"),
            (
                # Case 2: Less limited memory - the largest window get clamped
                cache_bytes_per_token_per_layer *
                (100 * 9 + 100 * 5 + 817 * 2) + 4,
                {
                    100: [0, 1, 2, 3],
                    200: [4, 5, 6],
                    1017: [7, 8],
                },
                [100] * 4 + [200] * 3 + [1017] * 2,
                None,
                "less_limited_memory_clamped_windows"),
            (
                # Case 3: Sufficient memory - no clamping needed
                cache_bytes_per_token_per_layer *
                (100 * 4 + 200 * 3 + 7000 * 2) + 9402,
                {
                    100: [0, 1, 2, 3],
                    200: [4, 5, 6],
                    7000: [7, 8],
                },
                [100] * 4 + [200] * 3 + [7000] * 2,
                None,
                "sufficient_memory_no_clamping"),
            (
                # Case 4: Very limited memory - all windows get small values
                cache_bytes_per_token_per_layer * (51 * 9) + 1,
                {
                    51: [0, 1, 2, 3, 4, 5, 6, 7, 8],
                },
                [51] * 9,
                None,
                "very_limited_memory_all_clamped"),
            (
                # Case 5: Less limited memory but max_tokens is given.
                # memory is enough for 1017 tokens, it will be clamped by max_tokens=134.
                cache_bytes_per_token_per_layer *
                (100 * 9 + 100 * 5 + 817 * 2) + 4,
                {
                    100: [0, 1, 2, 3],
                    134: [4, 5, 6, 7, 8],
                },
                [100] * 4 + [134] * 5,
                134,
                "less_limited_memory_but_clamped_by_max_tokens"),
        ]

        for memory_bytes, expected_window_sizes, expected_max_attention_window_vec, max_tokens, description in test_cases:
            with self.subTest(case=description, memory_bytes=memory_bytes):
                kv_cache_config = tllm.KvCacheConfig(max_tokens=max_tokens)
                adjusted, adjusted_max_attention_window_vec = KVCacheManager.adjust_window_sizes_for_vswa(
                    window_size_to_layers=window_size_to_layers,
                    max_attention_window_vec=max_attention_window_vec,
                    model_config=model_config,
                    kv_cache_config=kv_cache_config,
                    pool_memory_bytes=memory_bytes,
                    kv_factor=kv_factor,
                    dtype=model_config.data_type,
                    is_cross_attention=False,
                )

                self.assertEqual(
                    adjusted, expected_window_sizes,
                    f"Test case '{description}' failed.\n"
                    f"Memory bytes: {memory_bytes}\n"
                    f"Actual: {adjusted}\n"
                    f"Expected: {expected_window_sizes}")
                self.assertEqual(
                    adjusted_max_attention_window_vec,
                    expected_max_attention_window_vec,
                    f"Test case '{description}' failed.\n"
                    f"Memory bytes: {memory_bytes}\n"
                    f"Actual: {adjusted_max_attention_window_vec}\n"
                    f"Expected: {expected_max_attention_window_vec}")

    @staticmethod
    def _create_model_config_for_kv_cache_manager() -> ModelConfigCpp:
        """
        Create a simple model config for KVCacheManager test.
        """

        model_config_params = {
            "vocab_size": 0,
            "num_layers": 4,
            "num_attention_layers": 4,
            "num_rnn_layers": 0,
            "num_heads": 64,
            "hidden_size": 64,
            "data_type": DataType.HALF
        }
        num_kv_heads = 8

        model_config = ModelConfigCpp(**model_config_params)
        model_config.layer_types = [LayerType.ATTENTION
                                    ] * model_config.num_attention_layers()
        model_config.set_num_kv_heads(num_kv_heads)

        return model_config

    @staticmethod
    def _create_kv_cache_config_for_kv_cache_manager(
            params: dict) -> KvCacheConfig:
        """
        Create a KV cache config for KVCacheManager test.
        """
        return KvCacheConfig(**params)

    def test_calculate_max_num_blocks_from_cpp(self):
        # Construct a minimal mapping (single-rank, no TP/PP)
        mapping = Mapping(world_size=1, tp_size=1, pp_size=1)

        # Construct model config
        model_config = TestResourceManager._create_model_config_for_kv_cache_manager(
        )

        # Construct KV cache config
        free_gpu_memory_fraction = 0.1
        max_attention_window = [64, 128]
        max_gpu_total_bytes = 32 * 1024 * 1024  # 32MB
        enable_block_reuse = False
        host_cache_size = 32 * 1024 * 1024  # 32MB

        # mock values for torch.cuda.mem_get_info to return a fixed value
        fixed_free_mem = 128 * 1024 * 1024  # 128MB
        fixed_total_mem = 256 * 1024 * 1024  # 256MB

        class MemTestCase(NamedTuple):
            case_name: str
            kv_cache_config_params: dict
            expected_memory_bytes: Tuple[
                int,
                int]  # (primary_pool_memory_bytes, secondary_pool_memory_bytes)

        test_cases = [
            # Case 1:
            # max_gpu_total_bytes is set, even if free_gpu_memory_fraction is set, we will use max_gpu_total_bytes
            # host_cache_size is set, we will use host_cache_size
            MemTestCase(
                case_name="max_gpu_total_bytes is set, host_cache_size is set",
                kv_cache_config_params={
                    "max_attention_window": max_attention_window,
                    "free_gpu_memory_fraction": free_gpu_memory_fraction,
                    "max_gpu_total_bytes": max_gpu_total_bytes,
                    "enable_block_reuse": enable_block_reuse,
                    "host_cache_size": host_cache_size,
                },
                expected_memory_bytes=(max_gpu_total_bytes, host_cache_size),
            ),

            # Case 2:
            # max_gpu_total_bytes is not set, we will use free_gpu_memory_fraction
            # host_cache_size is not set, we will use 0
            MemTestCase(
                case_name=
                "max_gpu_total_bytes is not set, host_cache_size is not set",
                kv_cache_config_params={
                    "max_attention_window": max_attention_window,
                    "free_gpu_memory_fraction": free_gpu_memory_fraction,
                    "enable_block_reuse": enable_block_reuse,
                },
                expected_memory_bytes=(int(fixed_free_mem *
                                           free_gpu_memory_fraction), 0),
            ),
        ]

        tokens_per_block = 32
        model_config.tokens_per_block = tokens_per_block
        max_seq_len = max(max_attention_window)
        max_batch_size = 1
        max_beam_width = 1

        for case_name, kv_cache_config_params, expected_memory_bytes in test_cases:
            with self.subTest(case=case_name):
                kv_cache_config = TestResourceManager._create_kv_cache_config_for_kv_cache_manager(
                    kv_cache_config_params)
                with patch('torch.cuda.mem_get_info',
                           return_value=(fixed_free_mem, fixed_total_mem)):
                    # Create a real KVCacheManager, it will run calculate_max_num_blocks_from_cpp in __init__
                    manager = KVCacheManager(
                        kv_cache_config=kv_cache_config,
                        kv_cache_type=tensorrt_llm.bindings.internal.
                        batch_manager.CacheType.SELF,
                        num_layers=model_config.num_attention_layers(),
                        num_kv_heads=model_config.num_kv_heads(
                            0
                        ),  # NOTE: assume same number of kv heads for all layers
                        head_dim=model_config.head_size,
                        tokens_per_block=tokens_per_block,
                        max_seq_len=max_seq_len,
                        max_batch_size=max_batch_size,
                        mapping=mapping,
                        dtype=model_config.data_type,
                        model_config=model_config,
                        max_beam_width=max_beam_width,
                    )
                    try:
                        expected_primary, expected_secondary = expected_memory_bytes
                        self.assertEqual(
                            manager._primary_pool_memory_bytes,
                            expected_primary,
                            f"Test case '{case_name}' failed.\n"
                            f"Expected primary pool memory bytes: {expected_primary}\n"
                            f"Actual primary pool memory bytes: {manager._primary_pool_memory_bytes}"
                        )
                        self.assertEqual(
                            manager._secondary_pool_memory_bytes,
                            expected_secondary,
                            f"Test case '{case_name}' failed.\n"
                            f"Expected secondary pool memory bytes: {expected_secondary}\n"
                            f"Actual secondary pool memory bytes: {manager._secondary_pool_memory_bytes}"
                        )
                    except Exception as e:
                        self.fail(f"Test case '{case_name}' failed: {e}")
                    finally:
                        manager.shutdown()


if __name__ == "__main__":
    unittest.main()
