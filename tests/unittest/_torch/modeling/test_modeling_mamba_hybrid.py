import os
import sys
import unittest
from copy import deepcopy

import torch

import tensorrt_llm
from tensorrt_llm._torch.attention_backend.utils import get_attention_backend
from tensorrt_llm._torch.metadata import KVCacheParams
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.modeling_mamba_hybrid import (
    MambaHybridConfig, MambaHybridForCausalLM)
from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager
from tensorrt_llm.bindings.executor import KvCacheConfig
from tensorrt_llm.mapping import Mapping

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

MAMBA_HYBRID_CONFIG = {
    "architectures": ["MambaHybridForCausalLM"],
    "attention_bias": False,
    "attention_dropout": 0.0,
    "attention_head_dim": 128,
    "bos_token_id": 1,
    "chunk_size": 256,
    "conv_kernel": 4,
    "eos_token_id": 2,
    "expand": 2,
    "hidden_dropout": 0.0,
    "hidden_size": 4096,
    "hybrid_override_pattern":
    "M-M-M-M*-M-M-M-M-M*-M-M-M-M-M*-M-M-M-M-M*-M-M-M-M-M-",
    "initializer_range": 0.02,
    "intermediate_size": 21504,
    "layer_norm_epsilon": 1e-05,
    "mamba_d_state": 128,
    "mamba_head_dim": 64,
    "mamba_hidden_act": "silu",
    "mamba_num_heads": 128,
    "mamba_proj_bias": False,
    "max_position_embeddings": 8192,
    "mlp_bias": False,
    "mlp_hidden_act": "relu2",
    "model_type": "mamba_hybrid",
    "n_groups": 8,
    "num_attention_heads": 32,
    "num_hidden_layers": 52,
    "num_key_value_heads": 8,
    "num_logits_to_keep": 1,
    "pad_token_id": 0,
    "rescale_prenorm_residual": True,
    "residual_in_fp32": True,
    "rms_norm_eps": 1e-05,
    "sliding_window": None,
    "ssm_states_size": 128,
    "tie_word_embeddings": False,
    "time_step_floor": 0.0001,
    "time_step_max": 0.1,
    "time_step_min": 0.001,
    "time_step_rank": 256,
    "torch_dtype": "bfloat16",
    "transformers_version": "4.48.0.dev0",
    "use_bias": False,
    "use_cache": True,
    "use_conv_bias": True,
    "use_mamba_kernels": True,
    "vocab_size": 131072,
}


class TestMambaHybrid(unittest.TestCase):

    def test_mamba_hybrid_sanity(self):
        config_dict = deepcopy(MAMBA_HYBRID_CONFIG)
        mamba_hybrid_config = MambaHybridConfig.from_dict(config_dict)

        dtype = mamba_hybrid_config.torch_dtype
        device = torch.device('cuda')

        model_config = ModelConfig(pretrained_config=mamba_hybrid_config)
        mamba_hybrid = MambaHybridForCausalLM(model_config).to(device)

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
        head_dim = mamba_hybrid.config.hidden_size // mamba_hybrid.config.num_attention_heads
        num_layers = mamba_hybrid.config.hybrid_override_pattern.count("*")
        num_heads = mamba_hybrid.config.num_attention_heads
        num_kv_heads = mamba_hybrid.config.num_key_value_heads
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
            num_heads=num_heads,
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
            logits = mamba_hybrid.forward(input_ids=input_ids,
                                          position_ids=position_ids,
                                          attn_metadata=attn_metadata)

        self.assertEqual(len(past_seen_tokens), logits.shape[0])

        with torch.inference_mode():
            attn_metadata.prepare()
            logits = mamba_hybrid.forward(input_ids=input_ids,
                                          position_ids=position_ids,
                                          attn_metadata=attn_metadata,
                                          return_context_logits=True)
        self.assertEqual(input_ids.shape, logits.shape[:-1])

        kv_cache_manager.shutdown()
