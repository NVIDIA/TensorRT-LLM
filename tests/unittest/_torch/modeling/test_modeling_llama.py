import os
import sys
import unittest
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import torch
from parameterized import parameterized
from transformers import LlamaConfig
from transformers import LlamaForCausalLM as HFLlamaForCausalLM

import tensorrt_llm
from tensorrt_llm._torch.attention_backend.utils import get_attention_backend
from tensorrt_llm._torch.metadata import KVCacheParams
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.modeling_llama import LlamaForCausalLM
from tensorrt_llm._torch.pyexecutor.cuda_graph_runner import \
    DecodingCUDAGraphRunner
from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager
from tensorrt_llm.bindings.executor import KvCacheConfig
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantConfig

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from utils.util import getSMVersion

LLAMA_3_1_8B_CONFIG = {
    "architectures": ["LlamaForCausalLM"],
    "attention_bias": False,
    "attention_dropout": 0.0,
    "bos_token_id": 128000,
    "eos_token_id": 128001,
    "hidden_act": "silu",
    "hidden_size": 4096,
    "initializer_range": 0.02,
    "intermediate_size": 14336,
    "max_position_embeddings": 131072,
    "mlp_bias": False,
    "model_type": "llama",
    "num_attention_heads": 32,
    "num_hidden_layers": 32,
    "num_key_value_heads": 8,
    "pretraining_tp": 1,
    "rms_norm_eps": 1e-05,
    "rope_scaling": {
        "factor": 8.0,
        "low_freq_factor": 1.0,
        "high_freq_factor": 4.0,
        "original_max_position_embeddings": 8192,
        "rope_type": "llama3"
    },
    "rope_theta": 500000.0,
    "tie_word_embeddings": False,
    "torch_dtype": "bfloat16",
    "transformers_version": "4.43.0.dev0",
    "use_cache": True,
    "vocab_size": 128256
}


@dataclass(repr=False)
class Scenario:
    backend: str
    use_cuda_graph: bool = False

    def __repr__(self) -> str:
        return f"backend:{self.backend.lower()}-use_cuda_graph:{self.use_cuda_graph}"


def reduce_llama_config(mem_for_full_model: int,
                        config_dict: dict[str, Any],
                        default_num_layers: int = 32):
    _, total_mem = torch.cuda.mem_get_info()
    # scale model down if gpu memory is low
    if total_mem < mem_for_full_model:
        model_fraction = total_mem / mem_for_full_model
        num_layers = int(config_dict["num_hidden_layers"] * model_fraction)
        num_layers = min(num_layers, default_num_layers)
        config_dict["num_hidden_layers"] = num_layers


class TestLlama(unittest.TestCase):

    @parameterized.expand([None, "FP8"])
    def test_llama_sanity(self, quant_algo):
        config_dict = deepcopy(LLAMA_3_1_8B_CONFIG)
        # 8B * sizeof(float16) plus some extra for activations
        mem_for_full_model = (2 + 1) * 8 * 2**(30)
        reduce_llama_config(mem_for_full_model, config_dict)
        if config_dict["num_hidden_layers"] <= 0:
            self.skipTest("Insufficient memory for a single Llama layer")
        llama_config = LlamaConfig.from_dict(config_dict)
        if quant_algo:
            quant_config = QuantConfig(quant_algo=quant_algo)
        else:
            quant_config = None
        if quant_algo == "FP8" and getSMVersion() < 89:
            self.skipTest("This test is not supported in pre-Ada architecture")

        dtype = llama_config.torch_dtype
        device = torch.device('cuda')

        model_config = ModelConfig(pretrained_config=llama_config,
                                   quant_config=quant_config)
        llama = LlamaForCausalLM(model_config).to(device)

        input_ids = torch.tensor([100, 200, 300, 100, 200, 100, 400, 500],
                                 dtype=torch.int,
                                 device=device)

        context_sequence_lengths = [3, 2, 1]
        sequence_lengths = context_sequence_lengths + [1, 1]
        past_seen_tokens = [0, 0, 0, 62, 75]
        request_ids = list(range(len(sequence_lengths)))
        token_nums = (torch.tensor(past_seen_tokens) +
                      torch.tensor(sequence_lengths)).tolist()
        prompt_lens = token_nums[:3] + past_seen_tokens[3:]

        num_blocks = 100
        tokens_per_block = 128
        head_dim = llama.config.hidden_size // llama.config.num_attention_heads
        num_layers = llama.config.num_hidden_layers
        num_kv_heads = llama.config.num_key_value_heads
        max_seq_len = num_blocks * tokens_per_block
        batch_size = len(context_sequence_lengths) + 2

        if dtype == torch.half:
            kv_cache_dtype = tensorrt_llm.bindings.DataType.HALF
        elif dtype == torch.bfloat16:
            kv_cache_dtype = tensorrt_llm.bindings.DataType.BF16
        else:
            raise ValueError("Invalid dtype")

        mapping = Mapping(world_size=1, tp_size=1, rank=0)
        kv_cache_config = KvCacheConfig(max_tokens=num_blocks *
                                        tokens_per_block)
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
            seq_len = context_sequence_lengths[i] if i < len(
                context_sequence_lengths) else 1
            position_id = torch.arange(tokens,
                                       tokens + seq_len,
                                       device=input_ids.device)
            position_ids.append(position_id)

        position_ids = torch.cat(position_ids).unsqueeze(0)

        with torch.inference_mode():
            attn_metadata.prepare()
            logits = llama.forward(input_ids=input_ids,
                                   position_ids=position_ids,
                                   attn_metadata=attn_metadata)

        self.assertEqual(len(past_seen_tokens), logits.shape[0])

        with torch.inference_mode():
            attn_metadata.prepare()
            logits = llama.forward(input_ids=input_ids,
                                   position_ids=position_ids,
                                   attn_metadata=attn_metadata,
                                   return_context_logits=True)
        self.assertEqual(input_ids.shape, logits.shape[:-1])

        kv_cache_manager.shutdown()

    @parameterized.expand([
        Scenario(backend="VANILLA"),
        Scenario(backend="FLASHINFER"),
        Scenario(backend="FLASHINFER", use_cuda_graph=True),
        Scenario(backend="TRTLLM"),
        Scenario(backend="TRTLLM", use_cuda_graph=True),
    ], lambda testcase_func, param_num, param:
                          f"{testcase_func.__name__}[{param.args[0]}]")
    @torch.no_grad()
    def test_llama_allclose_to_hf(self, scenario: Scenario) -> None:
        """
        Compare output to HF
        """
        backend = scenario.backend
        metadata_cls = get_attention_backend(backend).Metadata

        torch.random.manual_seed(0)
        config_dict = deepcopy(LLAMA_3_1_8B_CONFIG)
        # 8B * sizeof(float16) plus some extra for activations
        # times 2, since we'll need 2 of these
        mem_for_full_model = (2 + 1) * 8 * 2**(30) * 4
        reduce_llama_config(mem_for_full_model, config_dict)
        if config_dict["num_hidden_layers"] <= 0:
            self.skipTest("Insufficient memory for a single Llama layer")
        llama_config = LlamaConfig.from_dict(config_dict)
        dtype = llama_config.torch_dtype
        device = torch.device('cuda')

        hf_llama = HFLlamaForCausalLM(llama_config).to(dtype).to(device).eval()

        model_config = ModelConfig(pretrained_config=llama_config,
                                   attn_backend=backend)
        llama = LlamaForCausalLM(model_config).to(dtype).to(device)
        llama.load_weights(hf_llama.state_dict())

        num_blocks = 1
        tokens_per_block = 128
        head_dim = llama.config.hidden_size // llama.config.num_attention_heads
        num_layers = llama.config.num_hidden_layers
        num_kv_heads = llama.config.num_key_value_heads
        max_seq_len = num_blocks * tokens_per_block
        batch_size = 1

        if dtype == torch.half:
            kv_cache_dtype = tensorrt_llm.bindings.DataType.HALF
        elif dtype == torch.bfloat16:
            kv_cache_dtype = tensorrt_llm.bindings.DataType.BF16
        else:
            raise ValueError("Invalid dtype")

        mapping = Mapping(world_size=1, tp_size=1, rank=0)
        kv_cache_config = KvCacheConfig(max_tokens=num_blocks *
                                        tokens_per_block)
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

        # context
        input_ids = torch.tensor([100, 200, 300, 100, 200, 100, 400, 500],
                                 dtype=torch.int,
                                 device=device)

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

        # Note: no CUDA graphs for prefill, the graph runner is built for
        # decoding only.
        position_ids = [torch.arange(0, input_ids.size(-1))]
        position_ids = torch.cat(position_ids).unsqueeze(0).cuda()
        with torch.inference_mode():
            attn_metadata.prepare()
            logits = llama.forward(input_ids=input_ids,
                                   position_ids=position_ids,
                                   attn_metadata=attn_metadata)
            ref = hf_llama.forward(input_ids=input_ids.unsqueeze(0),
                                   position_ids=position_ids,
                                   use_cache=True)

        torch.testing.assert_close(logits,
                                   ref.logits[:, -1].float(),
                                   atol=0.4,
                                   rtol=0.4)

        # gen
        gen_input_ids = torch.tensor([600], dtype=torch.int, device=device)

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
            torch.arange(input_ids.size(-1),
                         input_ids.size(-1) + gen_input_ids.size(-1))
        ]
        gen_position_ids = torch.cat(gen_position_ids).unsqueeze(0).cuda()

        def run_forward(input_ids, position_ids, attn_metadata):
            attn_metadata.prepare()
            if not scenario.use_cuda_graph:
                return llama.forward(input_ids=input_ids,
                                     position_ids=position_ids,
                                     attn_metadata=attn_metadata)
            else:
                graph_runner = DecodingCUDAGraphRunner(
                    attn_metadata.max_num_requests, "cuda", attn_metadata)
                graph_runner.capture(lambda inputs: llama.forward(**inputs))

                for _ in range(2):
                    # Run it twice. This helps us catch problems if buffers are accidentally reallocated
                    # in prepare().
                    attn_metadata.prepare()
                    logits = graph_runner.run({
                        "input_ids": input_ids,
                        "position_ids": position_ids,
                        "attn_metadata": attn_metadata,
                    })
                return logits

        if scenario.use_cuda_graph:
            attn_metadata = attn_metadata.create_cuda_graph_metadata(1)

        with torch.inference_mode():
            logits = run_forward(input_ids=gen_input_ids,
                                 position_ids=gen_position_ids,
                                 attn_metadata=attn_metadata)
            ref = hf_llama.forward(input_ids=gen_input_ids.unsqueeze(0),
                                   position_ids=gen_position_ids,
                                   past_key_values=ref.past_key_values,
                                   use_cache=True)

        torch.testing.assert_close(logits,
                                   ref.logits[:, -1].float(),
                                   atol=0.4,
                                   rtol=0.4)

        kv_cache_manager.shutdown()
