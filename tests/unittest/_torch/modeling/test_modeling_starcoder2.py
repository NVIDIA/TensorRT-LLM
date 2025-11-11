import unittest
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import torch
from parameterized import parameterized
from transformers import AutoTokenizer, Starcoder2Config
from transformers import Starcoder2ForCausalLM as HFStarcoder2ForCausalLM
from utils.util import default_dtype

import tensorrt_llm
from tensorrt_llm._torch.attention_backend.utils import get_attention_backend
from tensorrt_llm._torch.metadata import KVCacheParams
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.modeling_starcoder2 import Starcoder2ForCausalLM
from tensorrt_llm._torch.pyexecutor.cuda_graph_runner import CUDAGraphRunner
from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager
from tensorrt_llm.bindings.executor import KvCacheConfig
from tensorrt_llm.mapping import Mapping

# StarCoder2-3B config (reduced for testing)
STARCODER2_3B_CONFIG = {
    "architectures": ["Starcoder2ForCausalLM"],
    "attention_dropout": 0.0,
    "bos_token_id": 50256,
    "eos_token_id": 50256,
    "hidden_act": "gelu",
    "hidden_size": 2048,
    "initializer_range": 0.02,
    "intermediate_size": 8192,
    "max_position_embeddings": 16384,
    "model_type": "starcoder2",
    "num_attention_heads": 16,
    "num_hidden_layers": 6,  # Reduced from original for testing
    "num_key_value_heads": 2,
    "rope_theta": 10000.0,
    "sliding_window": 4096,
    "torch_dtype": "bfloat16",
    "transformers_version": "4.50.0.dev0",
    "use_bias": True,
    "use_cache": True,
    "vocab_size": 49152,
    "norm_epsilon": 1e-5,
    "mlp_type": "default",
}

# StarCoder2-7B config (reduced for testing)
STARCODER2_7B_CONFIG = {
    "architectures": ["Starcoder2ForCausalLM"],
    "attention_dropout": 0.0,
    "bos_token_id": 50256,
    "eos_token_id": 50256,
    "hidden_act": "gelu",
    "hidden_size": 3072,
    "initializer_range": 0.02,
    "intermediate_size": 12288,
    "max_position_embeddings": 16384,
    "model_type": "starcoder2",
    "num_attention_heads": 24,
    "num_hidden_layers": 6,  # Reduced from original for testing
    "num_key_value_heads": 2,
    "rope_theta": 10000.0,
    "sliding_window": 4096,
    "torch_dtype": "bfloat16",
    "transformers_version": "4.50.0.dev0",
    "use_bias": True,
    "use_cache": True,
    "vocab_size": 49152,
    "norm_epsilon": 1e-5,
    "mlp_type": "default",
}

# StarCoder2-15B config (reduced for testing)
STARCODER2_15B_CONFIG = {
    "architectures": ["Starcoder2ForCausalLM"],
    "attention_dropout": 0.1,
    "residual_dropout": 0.1,
    "embedding_dropout": 0.1,
    "bos_token_id": 0,
    "eos_token_id": 0,
    "hidden_act": "gelu_pytorch_tanh",
    "hidden_size": 6144,
    "initializer_range": 0.01275,
    "intermediate_size": 24576,
    "max_position_embeddings": 16384,
    "model_type": "starcoder2",
    "num_attention_heads": 48,
    "num_hidden_layers": 6,  # Reduced from 40 for testing
    "num_key_value_heads": 4,
    "rope_theta": 100000,
    "sliding_window": 4096,
    "torch_dtype": "bfloat16",
    "transformers_version": "4.50.0.dev0",
    "use_bias": True,
    "use_cache": True,
    "vocab_size": 49152,
    "norm_epsilon": 1e-5,
    "mlp_type": "default",
}


@dataclass(repr=False)
class Scenario:
    backend: str
    config_name: str
    use_cuda_graph: bool = False

    def __repr__(self) -> str:
        return f"backend:{self.backend.lower()}_config:{self.config_name.lower()}_cuda_graph:{self.use_cuda_graph}"


def reduce_starcoder2_config(
    mem_for_full_model: int, config_dict: dict[str, Any], default_num_layers: int = 6
):
    """Reduce model size if GPU memory is low."""
    _, total_mem = torch.cuda.mem_get_info()
    # scale model down if gpu memory is low
    if total_mem < mem_for_full_model:
        model_fraction = total_mem / mem_for_full_model
        num_layers = int(config_dict["num_hidden_layers"] * model_fraction)
        num_layers = min(num_layers, default_num_layers)
        config_dict["num_hidden_layers"] = num_layers


class TestStarcoder2(unittest.TestCase):

    def get_kv_cache_manager(
        self,
        dtype: torch.dtype,
        config: Starcoder2Config,
        tokens_per_block: int,
        max_seq_len: int,
        batch_size: int,
        num_blocks: int,
    ):
        """Helper to create KV cache manager."""
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
            max_tokens=num_blocks * tokens_per_block,
        )
        
        head_dim = config.hidden_size // config.num_attention_heads
        kv_cache_manager = KVCacheManager(
            kv_cache_config,
            tensorrt_llm.bindings.internal.batch_manager.CacheType.SELF,
            num_layers=config.num_hidden_layers,
            num_kv_heads=config.num_key_value_heads,
            head_dim=head_dim,
            tokens_per_block=tokens_per_block,
            max_seq_len=max_seq_len,
            max_batch_size=batch_size,
            mapping=mapping,
            dtype=kv_cache_dtype,
        )
        return kv_cache_manager

    def test_starcoder2_sanity(self):
        """Basic sanity test for StarCoder2 model."""
        config_dict = deepcopy(STARCODER2_3B_CONFIG)
        # 3B * sizeof(float16) plus some extra for activations
        mem_for_full_model = int((2 + 1) * 3 * 2 ** (30))
        reduce_starcoder2_config(mem_for_full_model, config_dict)
        
        if config_dict["num_hidden_layers"] <= 0:
            self.skipTest("Insufficient memory for a single StarCoder2 layer")
            
        starcoder2_config = Starcoder2Config.from_dict(config_dict)

        dtype = starcoder2_config.torch_dtype
        device = torch.device("cuda")

        with torch.device(device), default_dtype(dtype):
            model_config = ModelConfig(pretrained_config=starcoder2_config)
            starcoder2 = Starcoder2ForCausalLM(model_config).to(device)

        input_ids = torch.tensor(
            [100, 200, 300, 400, 500, 600, 700, 800],
            dtype=torch.int,
            device=device,
        )

        context_sequence_lengths = [3, 2, 1]
        sequence_lengths = context_sequence_lengths + [1, 1]
        past_seen_tokens = [0, 0, 0, 62, 75]
        request_ids = list(range(len(sequence_lengths)))
        token_nums = (
            torch.tensor(past_seen_tokens) + torch.tensor(sequence_lengths)
        ).tolist()
        prompt_lens = token_nums[:3] + past_seen_tokens[3:]

        num_blocks = 100
        tokens_per_block = 128
        max_seq_len = num_blocks * tokens_per_block
        batch_size = len(context_sequence_lengths) + 2
        
        kv_cache_manager = self.get_kv_cache_manager(
            dtype=dtype,
            config=starcoder2_config,
            tokens_per_block=tokens_per_block,
            max_seq_len=max_seq_len,
            batch_size=batch_size,
            num_blocks=num_blocks,
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
            seq_len = (
                context_sequence_lengths[i]
                if i < len(context_sequence_lengths)
                else 1
            )
            position_id = torch.arange(
                tokens, tokens + seq_len, device=input_ids.device
            )
            position_ids.append(position_id)

        position_ids = torch.cat(position_ids).unsqueeze(0)

        with torch.inference_mode():
            attn_metadata.prepare()
            logits = starcoder2.forward(
                input_ids=input_ids,
                position_ids=position_ids,
                attn_metadata=attn_metadata,
            )

        self.assertEqual(len(past_seen_tokens), logits.shape[0])

        with torch.inference_mode():
            attn_metadata.prepare()
            logits = starcoder2.forward(
                input_ids=input_ids,
                position_ids=position_ids,
                attn_metadata=attn_metadata,
                return_context_logits=True,
            )
        self.assertEqual(input_ids.shape, logits.shape[:-1])

        kv_cache_manager.shutdown()

    @parameterized.expand(
        [
            # 3B model tests
            Scenario(backend="TRTLLM", config_name="3B"),
            # 7B model tests
            Scenario(backend="TRTLLM", config_name="7B"),
            # 15B model tests
            Scenario(backend="TRTLLM", config_name="15B"),
        ],
        lambda testcase_func, param_num, param: f"{testcase_func.__name__}[{param.args[0]}]",
    )
    @torch.no_grad()
    def test_starcoder2_allclose_to_hf(self, scenario: Scenario) -> None:
        """
        Compare TensorRT-LLM StarCoder2 logits to HuggingFace.
        
        This test compares raw logit outputs (numerical correctness) using full 
        pretrained models from HuggingFace. It tests single forward passes for 
        context and generation phases.
        """
        backend = scenario.backend
        config_name = scenario.config_name
        metadata_cls = get_attention_backend(backend).Metadata

        torch.random.manual_seed(0)

        # Skip if insufficient memory for full pretrained model
        _, total_mem = torch.cuda.mem_get_info()
        min_mem_required = {
            "3B": 10 * (2**30),   # 10 GB
            "7B": 20 * (2**30),   # 20 GB  
            "15B": 40 * (2**30),  # 40 GB
        }
        
        if total_mem < min_mem_required[config_name]:
            self.skipTest(f"Insufficient memory for StarCoder2-{config_name}")

        # Load full pretrained model from HuggingFace
        model_name = f"bigcode/starcoder2-{config_name.lower()}"
        hf_starcoder2 = HFStarcoder2ForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        
        dtype = torch.bfloat16
        device = torch.device("cuda")

        # Build TRT-LLM model from the same pretrained config
        with torch.device(device), default_dtype(dtype):
            model_config = ModelConfig(
                pretrained_config=hf_starcoder2.config, 
                attn_backend=backend
            )
            starcoder2 = Starcoder2ForCausalLM(model_config).to(dtype).to(device)
            starcoder2.load_weights(hf_starcoder2.state_dict())

        num_blocks = 1
        tokens_per_block = 128
        max_seq_len = num_blocks * tokens_per_block
        batch_size = 1

        kv_cache_manager = self.get_kv_cache_manager(
            dtype=dtype,
            config=hf_starcoder2.config,
            tokens_per_block=tokens_per_block,
            max_seq_len=max_seq_len,
            batch_size=batch_size,
            num_blocks=num_blocks,
        )

        # Context phase.
        input_ids = torch.tensor(
            [100, 200, 300, 400, 500, 600, 700, 800],
            dtype=torch.int32,
            device=device,
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
            logits = starcoder2.forward(
                input_ids=input_ids,
                position_ids=position_ids,
                attn_metadata=attn_metadata,
            )
            ref = hf_starcoder2.forward(
                input_ids=input_ids.unsqueeze(0),
                position_ids=position_ids,
                use_cache=True,
            )
            torch.testing.assert_close(
                logits, ref.logits[:, -1].float(), atol=0.4, rtol=0.4
            )

        # Generation phase.
        gen_input_ids = torch.tensor([900], dtype=torch.int, device=device)
        num_cached_tokens_per_seq = [input_ids.size(-1)]
        attn_metadata = metadata_cls(
            seq_lens=torch.tensor([gen_input_ids.size(-1)], dtype=torch.int),
            num_contexts=0,
            kv_cache_params=KVCacheParams(
                use_cache=True,
                num_cached_tokens_per_seq=num_cached_tokens_per_seq,
            ),
            kv_cache_manager=kv_cache_manager,
            request_ids=request_ids,
            prompt_lens=prompt_lens,
            max_num_requests=1,
            max_num_tokens=8192,
        )

        gen_position_ids = [
            torch.arange(
                input_ids.size(-1), input_ids.size(-1) + gen_input_ids.size(-1)
            )
        ]
        gen_position_ids = torch.cat(gen_position_ids).unsqueeze(0).cuda()
        
        with torch.inference_mode():
            attn_metadata.prepare()
            logits = starcoder2.forward(
                input_ids=gen_input_ids,
                position_ids=gen_position_ids,
                attn_metadata=attn_metadata,
            )
            ref = hf_starcoder2.forward(
                input_ids=gen_input_ids.unsqueeze(0),
                position_ids=gen_position_ids,
                past_key_values=ref.past_key_values,
                use_cache=True,
            )
            torch.testing.assert_close(
                logits, ref.logits[:, -1].float(), atol=0.4, rtol=0.4
            )

        kv_cache_manager.shutdown()

    @parameterized.expand(
        [
            # 3B CUDA graph tests
            Scenario(backend="TRTLLM", config_name="3B", use_cuda_graph=True),
            # 7B CUDA graph tests
            Scenario(backend="TRTLLM", config_name="7B", use_cuda_graph=True),
            # 15B CUDA graph tests
            Scenario(backend="TRTLLM", config_name="15B", use_cuda_graph=True),
        ],
        lambda testcase_func, param_num, param: f"{testcase_func.__name__}[{param.args[0]}]",
    )
    @torch.no_grad()
    def test_starcoder2_with_cuda_graph(self, scenario: Scenario) -> None:
        """Test StarCoder2 with CUDA graphs for generation."""
        backend = scenario.backend
        use_cuda_graph = scenario.use_cuda_graph
        config_name = scenario.config_name
        metadata_cls = get_attention_backend(backend).Metadata

        torch.random.manual_seed(0)

        if config_name == "3B":
            config_dict = deepcopy(STARCODER2_3B_CONFIG)
            mem_for_full_model = int((2 + 1) * 3 * 2 ** (30) * 2)
        elif config_name == "15B":
            config_dict = deepcopy(STARCODER2_15B_CONFIG)
            mem_for_full_model = int((2 + 1) * 15 * 2 ** (30) * 2)
        else:
            raise ValueError(f"Unknown config_name: {config_name}")

        reduce_starcoder2_config(mem_for_full_model, config_dict)
        
        if config_dict["num_hidden_layers"] <= 0:
            self.skipTest(f"Insufficient memory for StarCoder2 {config_name} layer")
            
        starcoder2_config = Starcoder2Config.from_dict(config_dict)

        dtype = starcoder2_config.torch_dtype
        device = torch.device("cuda")

        with torch.device(device), default_dtype(dtype):
            hf_starcoder2 = HFStarcoder2ForCausalLM(starcoder2_config).eval()

            model_config = ModelConfig(
                pretrained_config=starcoder2_config, 
                attn_backend=backend
            )
            starcoder2 = Starcoder2ForCausalLM(model_config).to(dtype).to(device)
            starcoder2.load_weights(hf_starcoder2.state_dict())

        num_blocks = 1
        tokens_per_block = 128
        max_seq_len = num_blocks * tokens_per_block
        batch_size = 1

        kv_cache_manager = self.get_kv_cache_manager(
            dtype=dtype,
            config=starcoder2_config,
            tokens_per_block=tokens_per_block,
            max_seq_len=max_seq_len,
            batch_size=batch_size,
            num_blocks=num_blocks,
        )

        # Context phase.
        input_ids = torch.tensor(
            [100, 200, 300, 400, 500, 600, 700, 800],
            dtype=torch.int,
            device=device,
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

        position_ids = [torch.arange(0, input_ids.size(-1))]
        position_ids = torch.cat(position_ids).unsqueeze(0).cuda()
        
        # Note: no CUDA graphs for prefill
        with torch.inference_mode():
            attn_metadata.prepare()
            logits = starcoder2.forward(
                input_ids=input_ids,
                position_ids=position_ids,
                attn_metadata=attn_metadata,
            )
            ref = hf_starcoder2.forward(
                input_ids=input_ids.unsqueeze(0),
                position_ids=position_ids,
                use_cache=True,
            )

        torch.testing.assert_close(
            logits, ref.logits[:, -1].float(), atol=0.4, rtol=0.4
        )

        # Generation phase with optional CUDA graph
        gen_input_ids = torch.tensor([900], dtype=torch.int, device=device)
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
                input_ids.size(-1), input_ids.size(-1) + gen_input_ids.size(-1)
            )
        ]
        gen_position_ids = torch.cat(gen_position_ids).unsqueeze(0).cuda()

        graph_runner = None
        if use_cuda_graph:
            from _torch.helpers import create_mock_engine
            mock_engine = create_mock_engine(1)
            graph_runner = CUDAGraphRunner(mock_engine)

        def run_forward(input_ids, position_ids, attn_metadata):
            attn_metadata.prepare()
            if not use_cuda_graph:
                return starcoder2.forward(
                    input_ids=input_ids,
                    position_ids=position_ids,
                    attn_metadata=attn_metadata,
                )
            else:
                inputs = {
                    "input_ids": input_ids,
                    "position_ids": position_ids,
                    "attn_metadata": attn_metadata,
                }
                key = (1, 0, False)
                graph_runner.capture(
                    key, lambda inputs: starcoder2.forward(**inputs), inputs
                )

                for _ in range(2):
                    # Run it twice to catch buffer reallocation issues
                    attn_metadata.prepare()
                    logits = graph_runner.replay(key, inputs)
                return logits

        if use_cuda_graph:
            attn_metadata = attn_metadata.create_cuda_graph_metadata(1)

        with torch.inference_mode():
            logits = run_forward(
                input_ids=gen_input_ids,
                position_ids=gen_position_ids,
                attn_metadata=attn_metadata,
            )
            ref = hf_starcoder2.forward(
                input_ids=gen_input_ids.unsqueeze(0),
                position_ids=gen_position_ids,
                past_key_values=ref.past_key_values,
                use_cache=True,
            )

        torch.testing.assert_close(
            logits, ref.logits[:, -1].float(), atol=0.4, rtol=0.4
        )
        
        if graph_runner is not None:
            graph_runner.clear()

        kv_cache_manager.shutdown()

    @parameterized.expand(
        [
            # Test token-level generation for different model sizes
            Scenario(backend="TRTLLM", config_name="3B"),
            Scenario(backend="TRTLLM", config_name="7B"),
            Scenario(backend="TRTLLM", config_name="15B"),
        ],
        lambda testcase_func, param_num, param: f"{testcase_func.__name__}[{param.args[0]}]",
    )
    @torch.no_grad()
    def test_starcoder2_generated_tokens_match_hf(self, scenario: Scenario) -> None:
        """
        Compare generated tokens from TRT-LLM PyTorch backend to HuggingFace.
        
        This is the PyTorch backend equivalent of test_engine_allclose_to_hf in 
        test_starcoder2_engine.py. Both tests use the same full pretrained models 
        from HuggingFace and compare token-level generation output.
        
        The difference is:
        - This test uses the PyTorch execution backend (TRTLLM attention)
        - The engine test uses the compiled TensorRT engine
        """
        backend = scenario.backend
        config_name = scenario.config_name
        
        torch.random.manual_seed(0)

        # Skip if insufficient memory for full pretrained model
        _, total_mem = torch.cuda.mem_get_info()
        min_mem_required = {
            "3B": 10 * (2**30),   # 10 GB
            "7B": 20 * (2**30),   # 20 GB  
            "15B": 40 * (2**30),  # 40 GB
        }
        
        if total_mem < min_mem_required[config_name]:
            self.skipTest(f"Insufficient memory for StarCoder2-{config_name}")

        # Load full pretrained model from HuggingFace (same as engine test)
        model_name = f"bigcode/starcoder2-{config_name.lower()}"
        hf_starcoder2 = HFStarcoder2ForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        
        dtype = torch.bfloat16
        device = torch.device("cuda")

        # Build TRT-LLM model from the same pretrained config
        with torch.device(device), default_dtype(dtype):
            model_config = ModelConfig(
                pretrained_config=hf_starcoder2.config, 
                attn_backend=backend
            )
            starcoder2 = Starcoder2ForCausalLM(model_config).to(dtype).to(device)
            starcoder2.load_weights(hf_starcoder2.state_dict())

        # Test prompt - same as engine test
        test_prompt = "def fibonacci(n):"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Encode test prompt
        input_ids = torch.tensor(
            tokenizer.encode(test_prompt),
            dtype=torch.int32,
            device=device,
        )
        
        # Setup KV cache for TRT-LLM generation
        num_blocks = 2
        tokens_per_block = 128
        max_seq_len = num_blocks * tokens_per_block
        batch_size = 1

        kv_cache_manager = self.get_kv_cache_manager(
            dtype=dtype,
            config=hf_starcoder2.config,
            tokens_per_block=tokens_per_block,
            max_seq_len=max_seq_len,
            batch_size=batch_size,
            num_blocks=num_blocks,
        )

        # Generate tokens with TRT-LLM (manual generation loop)
        max_new_tokens = 20
        trt_output_ids = []
        num_cached_tokens = 0
        request_ids = [1]
        prompt_lens = [input_ids.size(-1)]
        metadata_cls = get_attention_backend(backend).Metadata

        # Context phase - process initial prompt
        token_nums = [input_ids.size(-1)]
        kv_cache_manager.add_dummy_requests(request_ids, token_nums)
        
        attn_metadata = metadata_cls(
            seq_lens=torch.tensor([input_ids.size(-1)], dtype=torch.int),
            num_contexts=1,
            kv_cache_params=KVCacheParams(
                use_cache=True,
                num_cached_tokens_per_seq=[0],
            ),
            kv_cache_manager=kv_cache_manager,
            request_ids=request_ids,
            prompt_lens=prompt_lens,
            max_num_requests=1,
            max_num_tokens=8192,
        )

        position_ids = torch.arange(0, input_ids.size(-1), dtype=torch.int32, device=device).unsqueeze(0)

        with torch.inference_mode():
            attn_metadata.prepare()
            logits = starcoder2.forward(
                input_ids=input_ids,
                position_ids=position_ids,
                attn_metadata=attn_metadata,
            )

        # Get first token
        next_token_id = torch.argmax(logits, dim=-1).item()
        trt_output_ids.append(next_token_id)
        num_cached_tokens = input_ids.size(-1)

        # Generation phase - generate remaining tokens
        for step in range(1, max_new_tokens):
            gen_input_ids = torch.tensor([next_token_id], dtype=torch.int32, device=device)
            
            attn_metadata = metadata_cls(
                seq_lens=torch.tensor([1], dtype=torch.int),
                num_contexts=0,
                kv_cache_params=KVCacheParams(
                    use_cache=True,
                    num_cached_tokens_per_seq=[num_cached_tokens],
                ),
                kv_cache_manager=kv_cache_manager,
                request_ids=request_ids,
                prompt_lens=prompt_lens,
                max_num_requests=1,
                max_num_tokens=8192,
            )

            gen_position_ids = torch.arange(
                num_cached_tokens, num_cached_tokens + 1,
                dtype=torch.int32,
                device=device
            ).unsqueeze(0)

            with torch.inference_mode():
                attn_metadata.prepare()
                logits = starcoder2.forward(
                    input_ids=gen_input_ids,
                    position_ids=gen_position_ids,
                    attn_metadata=attn_metadata,
                )

            # Greedy sampling: take argmax
            next_token_id = torch.argmax(logits, dim=-1).item()
            trt_output_ids.append(next_token_id)
            num_cached_tokens += 1

        # Generate with HuggingFace for comparison
        with torch.inference_mode():
            hf_output = hf_starcoder2.generate(
                input_ids.unsqueeze(0),
                max_new_tokens=max_new_tokens,
                do_sample=False,  # Greedy decoding
                pad_token_id=tokenizer.eos_token_id,
                use_cache=True,
            )
        
        hf_output_ids = hf_output[0, len(input_ids):].cpu().tolist()
        
        # Decode for debugging
        trt_text = tokenizer.decode(trt_output_ids)
        hf_text = tokenizer.decode(hf_output_ids)
        
        # Compare outputs - allow some tolerance for minor differences
        min_len = min(len(trt_output_ids), len(hf_output_ids))
        matches = sum(
            1 for i in range(min_len)
            if trt_output_ids[i] == hf_output_ids[i]
        )
        match_ratio = matches / min_len if min_len > 0 else 0.0
        
        # Print for debugging
        print(f"\n{config_name}/{backend} TRT output: {trt_text}")
        print(f"{config_name}/{backend} HF output:  {hf_text}")
        print(f"Match ratio: {match_ratio:.2%} ({matches}/{min_len} tokens)")
        
        # Should match at least 80% of tokens
        self.assertGreater(
            match_ratio, 0.8,
            f"TRT-LLM and HF token outputs differ significantly: {match_ratio:.2%} match"
        )
        
        kv_cache_manager.shutdown()


if __name__ == "__main__":
    unittest.main()

