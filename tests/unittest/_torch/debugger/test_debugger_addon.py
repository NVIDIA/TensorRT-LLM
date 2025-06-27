import unittest
from copy import deepcopy

import pytest
import torch
from transformers import LlamaConfig

import tensorrt_llm
from tensorrt_llm._torch.attention_backend.utils import get_attention_backend
from tensorrt_llm._torch.debug.debug_hook import (DebuggerContext, Filter,
                                                  debug_mode,
                                                  get_current_debug_ctx)
from tensorrt_llm._torch.metadata import KVCacheParams
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.modeling_llama import LlamaForCausalLM
from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager
from tensorrt_llm.bindings.executor import KvCacheConfig
from tensorrt_llm.mapping import Mapping

LLAMA_3_1_8B_2_layer_CONFIG = {
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
    "num_hidden_layers": 2,
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


class EmptyFilter(Filter):

    def __init__(self):
        pass

    def filter(self, module: torch.nn.Module, debug_ctx: DebuggerContext):
        return True


def print_module_name(module: torch.nn.Module, data_tensor,
                      debug_ctx: DebuggerContext):
    name = module.name if hasattr(module, "name") else module.__class__.__name__
    is_pre_forward = debug_ctx.check_in_pre_forward()
    phase = "pre_forward" if is_pre_forward else "after_forward"
    print(f"module {name} is running at {phase}.")


def register_module_name_dump_hook():
    debug_ctx = get_current_debug_ctx()
    assert debug_ctx is not None, ""
    debug_ctx.register_pre_forward_action(EmptyFilter(), print_module_name)
    debug_ctx.register_after_forward_action(EmptyFilter(), print_module_name)


class TestDebugger(unittest.TestCase):

    def test_with_2_layer_llama(self):
        pytest.skip(
            f"skip because this is only demo and unit test for debughook but not functionality."
        )
        config_dict = deepcopy(LLAMA_3_1_8B_2_layer_CONFIG)
        if config_dict["num_hidden_layers"] <= 0:
            self.skipTest("Insufficient memory for a single Llama layer")
        llama_config = LlamaConfig.from_dict(config_dict)
        quant_config = None

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

        with torch.inference_mode(), debug_mode(llama):
            register_module_name_dump_hook()
            attn_metadata.prepare()
            logits = llama.forward(input_ids=input_ids,
                                   position_ids=position_ids,
                                   attn_metadata=attn_metadata,
                                   return_context_logits=True)
        self.assertEqual(input_ids.shape, logits.shape[:-1])

        kv_cache_manager.shutdown()
