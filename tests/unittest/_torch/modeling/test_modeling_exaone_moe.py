import unittest
from copy import deepcopy
from dataclasses import dataclass

import torch
from _torch.helpers import create_mock_cuda_graph_runner
from parameterized import parameterized

import tensorrt_llm
from tensorrt_llm._torch.attention_backend.utils import get_attention_backend
from tensorrt_llm._torch.metadata import KVCacheParams
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.modeling_exaone_moe import ExaoneMoeForCausalLM
from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager
from tensorrt_llm.bindings.executor import KvCacheConfig
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantConfig

from utils.util import getSMVersion  # isort: skip

# fmt: off
# TODO: Remove this once we have a proper transformers package
from tensorrt_llm._torch.models.modeling_exaone_moe import ExaoneMoEConfig  # isort: skip

SKIP_EXAONE_MOE_HF_ACCURACY_TEST = False
try:
    from transformers.models.exaone_moe.modeling_exaone_moe import (
        ExaoneMoEForCausalLM as HFExaoneMoEForCausalLM,
    )
except ImportError:
    # TODO: Remove this once we have a proper config for EXAONE-MoE
    SKIP_EXAONE_MOE_HF_ACCURACY_TEST = True
# fmt: on

WINDOW_SIZE = 4
NUM_HIDDEN_LAYERS = 4

EXAONE_MOE_CONFIG = {
    "architectures": ["ExaoneMoEForCausalLM"],
    "attention_dropout": 0.0,
    "bos_token_id": 1,
    "dtype": "bfloat16",
    "eos_token_id": 53,
    "first_last_k_dense_replace": 1,
    "head_dim": 128,
    "hidden_act": "silu",
    "hidden_size": 6144,
    "initializer_range": 0.02,
    "intermediate_size": 18432,
    "is_moe_layer": [False] + [True] * (NUM_HIDDEN_LAYERS - 1),
    "layer_types": [
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "full_attention",
    ],
    "max_position_embeddings": 262144,
    "model_type": "exaone_moe",
    "moe_intermediate_size": 2048,
    "n_group": 1,
    "norm_topk_prob": True,
    "num_attention_heads": 64,
    "num_experts": 128,
    "num_experts_per_tok": 8,
    "num_hidden_layers": NUM_HIDDEN_LAYERS,
    "num_key_value_heads": 8,
    "num_shared_experts": 1,
    "pad_token_id": 0,
    "rms_norm_eps": 1e-05,
    "rope_scaling": None,
    "rope_theta": 1000000,
    "routed_scaling_factor": 2.5,
    "scoring_func": "sigmoid",
    "sliding_window": WINDOW_SIZE,
    "sliding_window_pattern": "LLLG",
    "tie_word_embeddings": False,
    "tokenizer_class": "GPT2Tokenizer",
    "topk_group": 1,
    "topk_method": "noaux_tc",
    "transformers_version": "5.0.0.dev0",
    "use_cache": True,
    "vocab_size": 153600,
}


@dataclass(repr=False)
class Scenario:
    attention_backend: str
    input_len: int = WINDOW_SIZE - 1
    use_cuda_graph: bool = False

    def __repr__(self) -> str:
        return (
            f"attention_backend:{self.attention_backend.lower()}-"
            f"input_len:{self.input_len}-"
            f"use_cuda_graph:{self.use_cuda_graph}"
        )


class TestExaoneMoe(unittest.TestCase):
    @parameterized.expand([None, "FP8"])
    def test_exaone_moe_sanity(self, quant_algo):
        """Test basic EXAONE-MoE model forward pass with optional quantization."""

        config_dict = deepcopy(EXAONE_MOE_CONFIG)
        exaone_moe_config = ExaoneMoEConfig.from_dict(config_dict)

        if quant_algo:
            quant_config = QuantConfig(quant_algo=quant_algo)
        else:
            quant_config = QuantConfig()

        if quant_algo == "FP8" and getSMVersion() < 89:
            self.skipTest("This test is not supported in pre-Ada architecture")

        dtype = exaone_moe_config.torch_dtype
        device = torch.device("cuda")

        model_config = ModelConfig(pretrained_config=exaone_moe_config, quant_config=quant_config)
        exaone_moe = ExaoneMoeForCausalLM(model_config).to(device)

        input_ids = torch.tensor(
            [100, 200, 300, 100, 200, 100, 400, 500], dtype=torch.int, device=device
        )

        context_sequence_lengths = [3, 2, 1]
        sequence_lengths = context_sequence_lengths + [1, 1]
        past_seen_tokens = [0, 0, 0, 62, 75]
        request_ids = list(range(len(sequence_lengths)))
        token_nums = (torch.tensor(past_seen_tokens) + torch.tensor(sequence_lengths)).tolist()
        prompt_lens = token_nums[:3] + past_seen_tokens[3:]

        num_blocks = 100
        tokens_per_block = 128
        head_dim = exaone_moe.config.hidden_size // exaone_moe.config.num_attention_heads
        num_layers = exaone_moe.config.num_hidden_layers
        num_kv_heads = exaone_moe.config.num_key_value_heads
        max_seq_len = num_blocks * tokens_per_block
        batch_size = len(context_sequence_lengths) + 2

        if dtype == torch.half:
            kv_cache_dtype = tensorrt_llm.bindings.DataType.HALF
        elif dtype == torch.bfloat16:
            kv_cache_dtype = tensorrt_llm.bindings.DataType.BF16
        else:
            raise ValueError("Invalid dtype")

        mapping = Mapping(world_size=1, tp_size=1, rank=0)
        kv_cache_config = KvCacheConfig(max_tokens=num_blocks * tokens_per_block)
        kv_cache_manager = KVCacheManager(
            kv_cache_config,
            tensorrt_llm.bindings.internal.batch_manager.CacheType.SELF,
            num_layers=num_layers,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            tokens_per_block=tokens_per_block,
            max_seq_len=max_seq_len,
            max_batch_size=batch_size,
            mapping=mapping,
            dtype=kv_cache_dtype,
        )
        kv_cache_manager.add_dummy_requests(request_ids, token_nums)

        metadata_cls = get_attention_backend(model_config.attn_backend).Metadata
        attn_metadata = metadata_cls(
            seq_lens=torch.tensor(sequence_lengths, dtype=torch.int),
            num_contexts=len(context_sequence_lengths),
            kv_cache_params=KVCacheParams(
                use_cache=True,
                num_cached_tokens_per_seq=past_seen_tokens,
            ),
            kv_cache_manager=kv_cache_manager,
            request_ids=request_ids,
            prompt_lens=prompt_lens,
            max_num_requests=len(context_sequence_lengths) + 2,
            max_num_tokens=8192,
        )

        position_ids = []
        for i, tokens in enumerate(past_seen_tokens):
            seq_len = context_sequence_lengths[i] if i < len(context_sequence_lengths) else 1
            position_id = torch.arange(tokens, tokens + seq_len, device=input_ids.device)
            position_ids.append(position_id)

        position_ids = torch.cat(position_ids).unsqueeze(0)

        with torch.inference_mode():
            attn_metadata.prepare()
            logits = exaone_moe.forward(
                input_ids=input_ids, position_ids=position_ids, attn_metadata=attn_metadata
            )

        self.assertEqual(len(past_seen_tokens), logits.shape[0])

        with torch.inference_mode():
            attn_metadata.prepare()
            logits = exaone_moe.forward(
                input_ids=input_ids,
                position_ids=position_ids,
                attn_metadata=attn_metadata,
                return_context_logits=True,
            )
        self.assertEqual(input_ids.shape, logits.shape[:-1])

        kv_cache_manager.shutdown()

    def test_exaone_moe_moe_layer_config(self):
        """Test that MoE layers are correctly configured."""
        config_dict = deepcopy(EXAONE_MOE_CONFIG)
        exaone_moe_config = ExaoneMoEConfig.from_dict(config_dict)

        device = torch.device("cuda")
        model_config = ModelConfig(pretrained_config=exaone_moe_config)
        exaone_moe = ExaoneMoeForCausalLM(model_config).to(device)

        # Verify MoE layer configuration
        is_moe_layer = config_dict["is_moe_layer"]
        self.assertEqual(len(is_moe_layer), NUM_HIDDEN_LAYERS)
        self.assertFalse(is_moe_layer[0])  # First layer should be dense
        for i in range(1, NUM_HIDDEN_LAYERS):
            self.assertTrue(is_moe_layer[i])  # Rest should be MoE

        # Verify model has correct number of layers
        self.assertEqual(len(exaone_moe.model.layers), NUM_HIDDEN_LAYERS)

    @parameterized.expand(
        [
            Scenario(attention_backend="TRTLLM", input_len=WINDOW_SIZE - 2),
            Scenario(attention_backend="TRTLLM", input_len=WINDOW_SIZE - 2, use_cuda_graph=True),
        ],
        lambda testcase_func, param_num, param: f"{testcase_func.__name__}[{param.args[0]}]",
    )
    @torch.no_grad()
    def test_exaone_moe_allclose_to_hf(self, scenario: Scenario) -> None:
        """Compare output to HuggingFace implementation."""
        if SKIP_EXAONE_MOE_HF_ACCURACY_TEST:
            self.skipTest("EXAONE-MoE HF model is not available in this environment")

        attention_backend = scenario.attention_backend
        metadata_cls = get_attention_backend(attention_backend).Metadata

        torch.random.manual_seed(0)
        config_dict = deepcopy(EXAONE_MOE_CONFIG)
        exaone_moe_config = ExaoneMoEConfig.from_dict(config_dict)
        dtype = exaone_moe_config.torch_dtype
        device = torch.device("cuda")

        hf_exaone_moe = HFExaoneMoEForCausalLM(exaone_moe_config).to(dtype).to(device).eval()

        model_config = ModelConfig(
            pretrained_config=exaone_moe_config, attn_backend=attention_backend
        )
        exaone_moe = ExaoneMoeForCausalLM(model_config).to(dtype).to(device)
        exaone_moe.load_weights(hf_exaone_moe.state_dict())
        exaone_moe.post_load_weights()

        num_blocks = 1
        tokens_per_block = 128
        head_dim = getattr(
            exaone_moe.config,
            "head_dim",
            exaone_moe.config.hidden_size // exaone_moe.config.num_attention_heads,
        )
        num_layers = exaone_moe.config.num_hidden_layers
        num_kv_heads = exaone_moe.config.num_key_value_heads
        max_seq_len = num_blocks * tokens_per_block
        batch_size = 1

        if dtype == torch.half:
            kv_cache_dtype = tensorrt_llm.bindings.DataType.HALF
        elif dtype == torch.bfloat16:
            kv_cache_dtype = tensorrt_llm.bindings.DataType.BF16
        else:
            raise ValueError("Invalid dtype")

        mapping = Mapping(world_size=1, tp_size=1, rank=0)
        kv_cache_config = KvCacheConfig(
            enable_block_reuse=False,
            enable_partial_reuse=False,
            copy_on_partial_reuse=False,
            max_attention_window=[int(exaone_moe_config.sliding_window)],
            max_tokens=num_blocks * tokens_per_block,
        )
        kv_cache_manager = KVCacheManager(
            kv_cache_config,
            tensorrt_llm.bindings.internal.batch_manager.CacheType.SELF,
            num_layers=num_layers,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            tokens_per_block=tokens_per_block,
            max_seq_len=max_seq_len,
            max_batch_size=batch_size,
            mapping=mapping,
            dtype=kv_cache_dtype,
        )

        # Context phase
        input_ids = torch.tensor(
            [i * 100 for i in range(1, scenario.input_len + 1)], dtype=torch.int32, device=device
        )

        num_cached_tokens_per_seq = [0]
        request_ids = [1]
        token_nums = [input_ids.size(-1)]
        prompt_lens = [input_ids.size(-1)]
        kv_cache_manager.add_dummy_requests(request_ids, token_nums)

        attn_metadata = metadata_cls(
            seq_lens=torch.tensor([input_ids.size(-1)], dtype=torch.int),
            num_contexts=1,
            kv_cache_params=KVCacheParams(
                use_cache=True,
                num_cached_tokens_per_seq=num_cached_tokens_per_seq,
            ),
            max_num_requests=1,
            max_num_tokens=8192,
            kv_cache_manager=kv_cache_manager,
            request_ids=request_ids,
            prompt_lens=prompt_lens,
        )

        position_ids = [torch.arange(0, input_ids.size(-1), dtype=torch.int32)]
        position_ids = torch.cat(position_ids).unsqueeze(0).cuda()

        with torch.inference_mode():
            attn_metadata.prepare()
            logits = exaone_moe.forward(
                input_ids=input_ids, position_ids=position_ids, attn_metadata=attn_metadata
            )
            ref = hf_exaone_moe.forward(
                input_ids=input_ids.unsqueeze(0), position_ids=position_ids, use_cache=True
            )

        # MoE models may have slightly higher tolerance due to expert routing
        torch.testing.assert_close(logits, ref.logits[:, -1].float(), atol=0.5, rtol=0.5)

        # Generation phase
        gen_input_ids = torch.tensor([600], dtype=torch.int32, device=device)
        num_cached_tokens_per_seq = [input_ids.size(-1)]

        attn_metadata = metadata_cls(
            seq_lens=torch.tensor([gen_input_ids.size(-1)], dtype=torch.int),
            num_contexts=0,
            kv_cache_params=KVCacheParams(
                use_cache=True,
                num_cached_tokens_per_seq=num_cached_tokens_per_seq,
            ),
            max_num_requests=1,
            max_num_tokens=8192,
            kv_cache_manager=kv_cache_manager,
            request_ids=request_ids,
            prompt_lens=prompt_lens,
        )

        gen_position_ids = [
            torch.arange(
                input_ids.size(-1), input_ids.size(-1) + gen_input_ids.size(-1), dtype=torch.int32
            )
        ]
        gen_position_ids = torch.cat(gen_position_ids).unsqueeze(0).cuda()

        graph_runner = create_mock_cuda_graph_runner(1) if scenario.use_cuda_graph else None

        def run_forward(input_ids, position_ids, attn_metadata):
            attn_metadata.prepare()
            if not scenario.use_cuda_graph:
                return exaone_moe.forward(
                    input_ids=input_ids, position_ids=position_ids, attn_metadata=attn_metadata
                )
            else:
                inputs = {
                    "input_ids": input_ids,
                    "position_ids": position_ids,
                    "attn_metadata": attn_metadata,
                }
                key = (1, 0, False)
                graph_runner.capture(key, lambda inputs: exaone_moe.forward(**inputs), inputs)

                for _ in range(2):
                    attn_metadata.prepare()
                    logits = graph_runner.replay(key, inputs)
                return logits

        if scenario.use_cuda_graph:
            attn_metadata = attn_metadata.create_cuda_graph_metadata(1)

        with torch.inference_mode():
            logits = run_forward(
                input_ids=gen_input_ids, position_ids=gen_position_ids, attn_metadata=attn_metadata
            )
            ref = hf_exaone_moe.forward(
                input_ids=gen_input_ids.unsqueeze(0),
                position_ids=gen_position_ids,
                past_key_values=ref.past_key_values,
                use_cache=True,
            )

        torch.testing.assert_close(logits, ref.logits[:, -1].float(), atol=0.5, rtol=0.5)

        if graph_runner is not None:
            graph_runner.clear()
        kv_cache_manager.shutdown()


if __name__ == "__main__":
    unittest.main()
