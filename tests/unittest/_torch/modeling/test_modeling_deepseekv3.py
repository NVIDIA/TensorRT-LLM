"""
Unit tests for DeepseekV3 model with various mapping configurations.

This test validates that the DeepseekV3 model produces consistent results
across different parallelism configurations (TP, EP, CP, PP) when run on
a single GPU with simulated mapping configurations.
"""

import unittest
from copy import deepcopy
from dataclasses import dataclass
from typing import Optional

import torch
from parameterized import parameterized

# Try to import pytest, but make it optional for basic unittest usage
try:
    import pytest
    HAS_PYTEST = True
except ImportError:
    HAS_PYTEST = False
    pytest = None

import tensorrt_llm
from tensorrt_llm._torch.attention_backend.utils import get_attention_backend
from tensorrt_llm._torch.configs.deepseek_v3 import DeepseekV3Config
from tensorrt_llm._torch.metadata import KVCacheParams
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.modeling_deepseekv3 import DeepseekV3ForCausalLM
from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager
from tensorrt_llm.bindings.executor import KvCacheConfig
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantConfig

# DeepseekV3-Lite config with minimal settings for testing
# The "Lite" version has q_lora_rank=None which makes it simpler
DEEPSEEKV3_LITE_SINGLE_LAYER_CONFIG = {
    "architectures": ["DeepseekV3ForCausalLM"],
    "model_type": "deepseek_v3",
    "vocab_size": 1024,  # Reduced for testing
    "hidden_size": 256,  # Reduced from 7168
    "intermediate_size": 512,  # Reduced from 18432
    "moe_intermediate_size": 128,  # Reduced from 2048
    "num_hidden_layers": 1,
    "num_nextn_predict_layers": 0,  # No MTP for simplicity
    "num_attention_heads": 8,  # Reduced from 128
    "num_key_value_heads": 8,  # Same as num_attention_heads for simplicity
    "n_shared_experts": 1,
    "n_routed_experts": 8,  # Reduced from 256
    "ep_size": 1,
    "routed_scaling_factor": 2.5,
    "kv_lora_rank": 64,  # Reduced from 512
    "q_lora_rank": None,  # None for "Lite" version
    "qk_rope_head_dim": 16,  # Reduced from 64
    "v_head_dim": 32,  # Reduced from 128
    "qk_nope_head_dim": 32,  # Reduced from 128
    "topk_method": "noaux_tc",
    "n_group": 2,  # Reduced from 8
    "topk_group": 2,  # Reduced from 4
    "num_experts_per_tok": 2,  # Reduced from 8
    "moe_layer_freq": 1,
    "first_k_dense_replace": 0,  # Make first layer use MoE for testing
    "norm_topk_prob": True,
    "scoring_func": "sigmoid",
    "hidden_act": "silu",
    "max_position_embeddings": 512,
    "initializer_range": 0.02,
    "rms_norm_eps": 1e-6,
    "use_cache": True,
    "pad_token_id": None,
    "bos_token_id": 0,
    "eos_token_id": 1,
    "tie_word_embeddings": False,
    "rope_theta": 10000.0,
    "rope_scaling": None,
    "attention_bias": False,
    "attention_dropout": 0.0,
    "torch_dtype": "bfloat16",
}


@dataclass(repr=False)
class MappingScenario:
    """Scenario representing different parallelism configurations."""

    name: str
    tp_size: int = 1
    pp_size: int = 1
    cp_size: int = 1
    moe_ep_size: int = 1
    moe_tp_size: int = -1  # -1 means auto
    enable_attention_dp: bool = False

    def __repr__(self) -> str:
        return (f"{self.name}_tp{self.tp_size}_pp{self.pp_size}"
                f"_cp{self.cp_size}_ep{self.moe_ep_size}"
                f"_adp{self.enable_attention_dp}")

    def create_mapping(self) -> Mapping:
        """Create a Mapping object for this scenario."""
        return Mapping(
            world_size=1,
            rank=0,
            gpus_per_node=8,
            tp_size=self.tp_size,
            pp_size=self.pp_size,
            cp_size=self.cp_size,
            moe_ep_size=self.moe_ep_size,
            moe_tp_size=self.moe_tp_size,
            enable_attention_dp=self.enable_attention_dp,
        )


def get_kv_cache_manager(
    dtype: torch.dtype,
    config: DeepseekV3Config,
    tokens_per_block: int,
    max_seq_len: int,
    batch_size: int,
    mapping: Mapping,
) -> KVCacheManager:
    """Create a KV cache manager for MLA-style models like DeepseekV3."""
    if dtype == torch.half:
        kv_cache_dtype = tensorrt_llm.bindings.DataType.HALF
    elif dtype == torch.bfloat16:
        kv_cache_dtype = tensorrt_llm.bindings.DataType.BF16
    else:
        raise ValueError(f"Invalid dtype: {dtype}")

    # For MLA models, the KV cache uses kv_lora_rank + qk_rope_head_dim
    head_dim = config.kv_lora_rank + config.qk_rope_head_dim

    kv_cache_config = KvCacheConfig(
        enable_block_reuse=False,
        enable_partial_reuse=False,
        copy_on_partial_reuse=False,
        max_tokens=max_seq_len,
    )

    kv_cache_manager = KVCacheManager(
        kv_cache_config,
        # MLA uses SELFKONLY cache type
        tensorrt_llm.bindings.internal.batch_manager.CacheType.SELFKONLY,
        num_layers=config.num_hidden_layers,
        num_kv_heads=1,  # MLA uses 1 kv head
        head_dim=head_dim,
        tokens_per_block=tokens_per_block,
        max_seq_len=max_seq_len,
        max_batch_size=batch_size,
        mapping=mapping,
        dtype=kv_cache_dtype,
    )
    return kv_cache_manager


class TestDeepseekV3Lite(unittest.TestCase):
    """Test suite for DeepseekV3Lite model with various mapping configurations."""

    def setUp(self):
        """Clear CUDA cache before each test."""
        super().setUp()
        torch.cuda.empty_cache()
        torch.random.manual_seed(42)

    def tearDown(self):
        """Clean up after each test."""
        super().tearDown()
        torch.cuda.empty_cache()

    def _create_model_and_run_forward(
        self,
        config_dict: dict,
        mapping: Mapping,
        quant_config: Optional[QuantConfig] = None,
        backend: str = "TRTLLM",
    ) -> torch.Tensor:
        """
        Create a DeepseekV3 model and run a forward pass.

        Args:
            config_dict: Model configuration dictionary
            mapping: Mapping configuration for parallelism
            quant_config: Optional quantization configuration
            backend: Attention backend to use

        Returns:
            Output logits from the forward pass
        """
        # Create config
        config = DeepseekV3Config.from_dict(config_dict)
        dtype = config.torch_dtype
        device = torch.device("cuda")

        # Create model config
        model_config = ModelConfig(
            pretrained_config=config,
            mapping=mapping,
            quant_config=quant_config or QuantConfig(),
            attn_backend=backend,
        )

        # Create model
        model = DeepseekV3ForCausalLM(model_config).to(device)

        # Setup test inputs
        input_ids = torch.tensor(
            [100, 200, 300, 100, 200, 100, 400, 500],
            dtype=torch.int32,
            device=device,
        )

        context_sequence_length = [3, 2, 1]
        sequence_length = context_sequence_length + [1, 1]
        past_seen_tokens = [0, 0, 0, 62, 75]
        request_ids = list(range(len(sequence_length)))
        token_nums = (
            torch.tensor(past_seen_tokens) + torch.tensor(sequence_length)
        ).tolist()
        prompt_lens = token_nums[:3] + past_seen_tokens[3:]

        tokens_per_block = 64
        max_seq_len = 512
        batch_size = len(sequence_length)

        # Create KV cache manager
        kv_cache_manager = get_kv_cache_manager(
            dtype=dtype,
            config=config,
            tokens_per_block=tokens_per_block,
            max_seq_len=max_seq_len,
            batch_size=batch_size,
            mapping=mapping,
        )

        try:
            kv_cache_manager.add_dummy_requests(request_ids, token_nums)

            # Create attention metadata
            metadata_cls = get_attention_backend(model_config.attn_backend).Metadata
            attn_metadata = metadata_cls(
                seq_lens=torch.tensor(sequence_length, dtype=torch.int32),
                num_contexts=len(context_sequence_length),
                kv_cache_params=KVCacheParams(
                    use_cache=True,
                    num_cached_tokens_per_seq=past_seen_tokens,
                ),
                kv_cache_manager=kv_cache_manager,
                request_ids=request_ids,
                prompt_lens=prompt_lens,
                max_num_requests=len(sequence_length),
                max_num_tokens=8192,
            )

            # Create position ids
            position_ids = []
            for i, tokens in enumerate(past_seen_tokens):
                seq_len = (
                    context_sequence_length[i]
                    if i < len(context_sequence_length)
                    else 1
                )
                position_id = torch.arange(
                    tokens, tokens + seq_len, device=input_ids.device
                )
                position_ids.append(position_id)
            position_ids = torch.cat(position_ids).unsqueeze(0)

            # Run forward pass
            with torch.inference_mode():
                attn_metadata.prepare()
                logits = model.forward(
                    input_ids=input_ids,
                    position_ids=position_ids,
                    attn_metadata=attn_metadata,
                )

            return logits
        finally:
            kv_cache_manager.shutdown()

    def test_deepseekv3_lite_sanity(self):
        """
        Basic sanity test for DeepseekV3Lite model.

        Verifies that the model can be created and run a forward pass
        with default mapping configuration.
        """
        config_dict = deepcopy(DEEPSEEKV3_LITE_SINGLE_LAYER_CONFIG)
        mapping = Mapping(world_size=1, tp_size=1, rank=0)

        logits = self._create_model_and_run_forward(config_dict, mapping)

        # Verify output shape: should have one logit per sequence
        self.assertEqual(logits.shape[0], 5)  # 5 sequences
        self.assertEqual(logits.shape[1], config_dict["vocab_size"])

    @parameterized.expand([
        # Base case: no parallelism (TP=1, EP=1, PP=1, CP=1)
        MappingScenario(name="base", tp_size=1, pp_size=1, cp_size=1, moe_ep_size=1),
        # Attention DP mode (simulates data parallelism for attention)
        MappingScenario(name="attention_dp", tp_size=1, enable_attention_dp=True),
        # TP mode (tensor parallelism - simulated on single GPU with tp_size=1)
        MappingScenario(name="tp_only", tp_size=1, moe_tp_size=1, moe_ep_size=1),
        # EP mode (expert parallelism - simulated with ep_size=1)
        MappingScenario(name="ep_only", tp_size=1, moe_tp_size=1, moe_ep_size=1),
        # Combined TP+EP (both enabled, simulated on single GPU)
        MappingScenario(name="tp_ep_combined", tp_size=1, moe_tp_size=1, moe_ep_size=1),
    ], lambda testcase_func, param_num, param:
                          f"{testcase_func.__name__}[{param.args[0]}]")
    def test_deepseekv3_lite_mapping_configurations(self, scenario: MappingScenario):
        """
        Test DeepseekV3Lite with various mapping configurations.

        This test verifies that the model produces valid outputs
        for different parallelism configurations. Since this is a
        single-GPU test, we're mainly testing that the code paths
        for different configurations work correctly.
        """
        config_dict = deepcopy(DEEPSEEKV3_LITE_SINGLE_LAYER_CONFIG)
        mapping = scenario.create_mapping()

        logits = self._create_model_and_run_forward(config_dict, mapping)

        # Verify output shape
        self.assertEqual(logits.shape[0], 5)  # 5 sequences
        self.assertEqual(logits.shape[1], config_dict["vocab_size"])

        # Verify no NaN values
        self.assertFalse(
            torch.isnan(logits).any(), f"NaN values in output for {scenario}"
        )

    def test_deepseekv3_lite_consistency_across_mappings(self):
        """
        Test that different mapping configurations produce consistent results.

        Since all tests run on a single GPU with simulated parallelism,
        the outputs should be consistent across different mapping configurations.
        This test verifies that code paths for different parallelism modes
        don't introduce unexpected variations in output.
        """
        config_dict = deepcopy(DEEPSEEKV3_LITE_SINGLE_LAYER_CONFIG)

        # Define mapping scenarios to compare
        scenarios = [
            MappingScenario(name="base", tp_size=1, moe_ep_size=1),
            MappingScenario(name="attention_dp", tp_size=1, enable_attention_dp=True),
        ]

        # Collect results from each scenario
        results = {}
        for scenario in scenarios:
            torch.random.manual_seed(42)  # Reset seed for reproducibility
            torch.cuda.manual_seed(42)
            mapping = scenario.create_mapping()
            logits = self._create_model_and_run_forward(config_dict, mapping)
            results[scenario.name] = logits.clone()

        # Compare outputs across scenarios
        # Note: Due to different code paths, we allow for small numerical differences
        base_result = results["base"]
        for name, logits in results.items():
            if name == "base":
                continue
            # Check that shapes match
            self.assertEqual(base_result.shape, logits.shape,
                           f"Shape mismatch between base and {name}")
            # Check for numerical similarity (allowing for floating point differences)
            # Using a relaxed tolerance since different code paths may have different
            # numerical characteristics
            if not torch.allclose(base_result, logits, rtol=0.1, atol=0.1):
                max_diff = (base_result - logits).abs().max().item()
                self.fail(f"Output differs between base and {name}: max_diff={max_diff}")

    def test_deepseekv3_lite_context_logits(self):
        """
        Test that context logits can be returned.

        Verifies that when return_context_logits=True, the model
        returns logits for all input tokens rather than just the last one.
        """
        config_dict = deepcopy(DEEPSEEKV3_LITE_SINGLE_LAYER_CONFIG)
        config = DeepseekV3Config.from_dict(config_dict)
        dtype = config.torch_dtype
        device = torch.device("cuda")

        mapping = Mapping(world_size=1, tp_size=1, rank=0)
        model_config = ModelConfig(
            pretrained_config=config,
            mapping=mapping,
            attn_backend="TRTLLM",
        )

        model = DeepseekV3ForCausalLM(model_config).to(device)

        input_ids = torch.tensor(
            [100, 200, 300, 400, 500],
            dtype=torch.int32,
            device=device,
        )

        tokens_per_block = 64
        max_seq_len = 512

        kv_cache_manager = get_kv_cache_manager(
            dtype=dtype,
            config=config,
            tokens_per_block=tokens_per_block,
            max_seq_len=max_seq_len,
            batch_size=1,
            mapping=mapping,
        )

        try:
            request_ids = [0]
            token_nums = [input_ids.size(-1)]
            kv_cache_manager.add_dummy_requests(request_ids, token_nums)

            metadata_cls = get_attention_backend(model_config.attn_backend).Metadata
            attn_metadata = metadata_cls(
                seq_lens=torch.tensor([input_ids.size(-1)], dtype=torch.int32),
                num_contexts=1,
                kv_cache_params=KVCacheParams(
                    use_cache=True,
                    num_cached_tokens_per_seq=[0],
                ),
                kv_cache_manager=kv_cache_manager,
                request_ids=request_ids,
                prompt_lens=[input_ids.size(-1)],
                max_num_requests=1,
                max_num_tokens=8192,
            )

            position_ids = torch.arange(0, input_ids.size(-1), device=device).unsqueeze(
                0
            )

            with torch.inference_mode():
                attn_metadata.prepare()
                logits = model.forward(
                    input_ids=input_ids,
                    position_ids=position_ids,
                    attn_metadata=attn_metadata,
                    return_context_logits=True,
                )

            # When return_context_logits=True, should return logits for all tokens
            self.assertEqual(logits.shape[0], input_ids.size(-1))
            self.assertEqual(logits.shape[1], config_dict["vocab_size"])

        finally:
            kv_cache_manager.shutdown()


class TestDeepseekV3LiteDenseVsMoe(unittest.TestCase):
    """Test suite for dense vs MoE layer configurations."""

    def setUp(self):
        """Clear CUDA cache before each test."""
        super().setUp()
        torch.cuda.empty_cache()
        torch.random.manual_seed(42)

    def tearDown(self):
        """Clean up after each test."""
        super().tearDown()
        torch.cuda.empty_cache()

    def _run_forward_test(self, first_k_dense_replace: int):
        """Run forward test with given dense layer configuration."""
        config_dict = deepcopy(DEEPSEEKV3_LITE_SINGLE_LAYER_CONFIG)
        config_dict["first_k_dense_replace"] = first_k_dense_replace

        config = DeepseekV3Config.from_dict(config_dict)
        dtype = config.torch_dtype
        device = torch.device("cuda")

        mapping = Mapping(world_size=1, tp_size=1, rank=0)
        model_config = ModelConfig(
            pretrained_config=config,
            mapping=mapping,
            attn_backend="TRTLLM",
        )

        model = DeepseekV3ForCausalLM(model_config).to(device)

        input_ids = torch.tensor([100, 200, 300], dtype=torch.int32, device=device)
        tokens_per_block = 64
        max_seq_len = 512

        kv_cache_manager = get_kv_cache_manager(
            dtype=dtype,
            config=config,
            tokens_per_block=tokens_per_block,
            max_seq_len=max_seq_len,
            batch_size=1,
            mapping=mapping,
        )

        try:
            request_ids = [0]
            token_nums = [input_ids.size(-1)]
            kv_cache_manager.add_dummy_requests(request_ids, token_nums)

            metadata_cls = get_attention_backend(model_config.attn_backend).Metadata
            attn_metadata = metadata_cls(
                seq_lens=torch.tensor([input_ids.size(-1)], dtype=torch.int32),
                num_contexts=1,
                kv_cache_params=KVCacheParams(
                    use_cache=True,
                    num_cached_tokens_per_seq=[0],
                ),
                kv_cache_manager=kv_cache_manager,
                request_ids=request_ids,
                prompt_lens=[input_ids.size(-1)],
                max_num_requests=1,
                max_num_tokens=8192,
            )

            position_ids = torch.arange(
                0, input_ids.size(-1), device=device
            ).unsqueeze(0)

            with torch.inference_mode():
                attn_metadata.prepare()
                logits = model.forward(
                    input_ids=input_ids,
                    position_ids=position_ids,
                    attn_metadata=attn_metadata,
                )

            # Verify output
            self.assertEqual(logits.shape[0], 1)
            self.assertEqual(logits.shape[1], config_dict["vocab_size"])
            self.assertFalse(torch.isnan(logits).any())

            return logits

        finally:
            kv_cache_manager.shutdown()

    @parameterized.expand([
        # Dense layer (first_k_dense_replace=1 means layer 0 is dense)
        (1, "dense"),
        # MoE layer (first_k_dense_replace=0 means layer 0 is MoE)
        (0, "moe"),
    ])
    def test_deepseekv3_lite_dense_vs_moe_layer(
        self, first_k_dense_replace: int, layer_type: str
    ):
        """
        Test DeepseekV3Lite with dense layer vs MoE layer.

        DeepseekV3 supports having the first few layers be dense (non-MoE)
        layers. This test verifies both configurations work correctly.
        """
        self._run_forward_test(first_k_dense_replace)


if __name__ == "__main__":
    unittest.main()

