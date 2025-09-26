import unittest
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import torch
import transformers
from parameterized import parameterized
from transformers import AutoConfig
from transformers.dynamic_module_utils import get_class_from_dynamic_module
from utils.llm_data import llm_models_root

import tensorrt_llm
from tensorrt_llm._torch.attention_backend.utils import get_attention_backend
from tensorrt_llm._torch.metadata import KVCacheParams
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.modeling_nemotron_nas import \
    NemotronNASForCausalLM
from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager
from tensorrt_llm.bindings.executor import KvCacheConfig
from tensorrt_llm.mapping import Mapping

# Setup NEED_SETUP_CACHE_CLASSES_MAPPING to an empty dict for modeling_nemotron_nas.py
transformers.generation.utils.NEED_SETUP_CACHE_CLASSES_MAPPING = dict()

NEMOTRON_NAS_MINI_CONFIG = {
    "architectures": ["DeciLMForCausalLM"],
    "attention_bias":
    False,
    "block_configs": [{
        "attention": {
            "n_heads_in_group": 8,
            "no_op": False,
            "replace_with_linear": False
        },
        "ffn": {
            "ffn_mult": 2.0,
            "no_op": False,
            "replace_with_linear": False
        }
    }, {
        "attention": {
            "n_heads_in_group": 16,
            "no_op": False,
            "replace_with_linear": False
        },
        "ffn": {
            "ffn_mult": 2.0,
            "no_op": False,
            "replace_with_linear": False
        }
    }, {
        "attention": {
            "n_heads_in_group": None,
            "no_op": False,
            "replace_with_linear": True
        },
        "ffn": {
            "ffn_mult": 2.0,
            "no_op": False,
            "replace_with_linear": False
        }
    }, {
        "attention": {
            "n_heads_in_group": None,
            "no_op": True,
            "replace_with_linear": False
        },
        "ffn": {
            "ffn_mult": 2.0,
            "no_op": False,
            "replace_with_linear": False
        }
    }, {
        "attention": {
            "n_heads_in_group": 8,
            "no_op": False,
            "replace_with_linear": False
        },
        "ffn": {
            "ffn_mult": None,
            "no_op": False,
            "replace_with_linear": True
        }
    }, {
        "attention": {
            "n_heads_in_group": 4,
            "no_op": False,
            "replace_with_linear": False
        },
        "ffn": {
            "ffn_mult": None,
            "no_op": True,
            "replace_with_linear": False
        }
    }, {
        "attention": {
            "n_heads_in_group": 8,
            "no_op": False,
            "replace_with_linear": False
        },
        "ffn": {
            "ffn_mult": 2.0,
            "no_op": False,
            "replace_with_linear": False
        }
    }, {
        "attention": {
            "n_heads_in_group": 8,
            "no_op": False,
            "replace_with_linear": False
        },
        "ffn": {
            "ffn_mult": 2.0,
            "no_op": False,
            "replace_with_linear": False
        }
    }, {
        "attention": {
            "n_heads_in_group": 16,
            "no_op": False,
            "replace_with_linear": False
        },
        "ffn": {
            "ffn_mult": 2.0,
            "no_op": False,
            "replace_with_linear": False
        }
    }, {
        "attention": {
            "n_heads_in_group": None,
            "no_op": False,
            "replace_with_linear": True
        },
        "ffn": {
            "ffn_mult": 2.0,
            "no_op": False,
            "replace_with_linear": False
        }
    }, {
        "attention": {
            "n_heads_in_group": None,
            "no_op": True,
            "replace_with_linear": False
        },
        "ffn": {
            "ffn_mult": 2.0,
            "no_op": False,
            "replace_with_linear": False
        }
    }, {
        "attention": {
            "n_heads_in_group": 8,
            "no_op": False,
            "replace_with_linear": False
        },
        "ffn": {
            "ffn_mult": None,
            "no_op": False,
            "replace_with_linear": True
        }
    }, {
        "attention": {
            "n_heads_in_group": 4,
            "no_op": False,
            "replace_with_linear": False
        },
        "ffn": {
            "ffn_mult": None,
            "no_op": True,
            "replace_with_linear": False
        }
    }, {
        "attention": {
            "n_heads_in_group": 8,
            "no_op": False,
            "replace_with_linear": False
        },
        "ffn": {
            "ffn_mult": 2.0,
            "no_op": False,
            "replace_with_linear": False
        }
    }],
    "bos_token_id":
    1,
    "eos_token_id":
    2,
    "hidden_act":
    "silu",
    "hidden_size":
    2048,
    "initializer_range":
    0.02,
    "intermediate_size":
    None,
    "max_position_embeddings":
    2048,
    "model_type":
    "deci",
    "num_attention_heads":
    32,
    "num_hidden_layers":
    14,
    "num_key_value_heads":
    None,
    "rms_norm_eps":
    1e-06,
    "rope_scaling":
    None,
    "rope_theta":
    10000.0,
    "tie_word_embeddings":
    False,
    "torch_dtype":
    "bfloat16",
    "use_cache":
    True,
    "vocab_size":
    32128
}


@dataclass(repr=False)
class Scenario:
    backend: str

    def __repr__(self) -> str:
        return f"backend:{self.backend.lower()}"


def reduce_nemotron_nas_config(mem_for_full_model: int, config_dict: dict[str,
                                                                          Any]):
    _, total_mem = torch.cuda.mem_get_info()
    # scale model down if gpu memory is low
    if total_mem < mem_for_full_model:
        model_fraction = total_mem / mem_for_full_model
        num_layers = int(config_dict["num_hidden_layers"] * model_fraction)
        num_layers = min(num_layers, 32)
        config_dict["num_hidden_layers"] = num_layers
        config_dict["block_configs"] = config_dict["block_configs"][:num_layers]


class TestNemotronNAS(unittest.TestCase):

    def test_nemotron_nas_sanity(self):
        config_dict = deepcopy(NEMOTRON_NAS_MINI_CONFIG)
        # 8B * sizeof(float16) plus some extra for activations
        mem_for_full_model = (2 + 1) * 8 * 2**(30)
        reduce_nemotron_nas_config(mem_for_full_model, config_dict)
        if config_dict["num_hidden_layers"] <= 0:
            self.skipTest("Insufficient memory for a single NemotronNAS layer")
        nemotron_nas_config = AutoConfig.from_pretrained(
            llm_models_root() / "nemotron-nas/Llama-3_1-Nemotron-51B-Instruct",
            trust_remote_code=True,
        )
        nemotron_nas_config = nemotron_nas_config.from_dict(config_dict)

        dtype = nemotron_nas_config.torch_dtype
        device = torch.device('cuda')

        model_config = ModelConfig(pretrained_config=nemotron_nas_config)
        nemotron_nas = NemotronNASForCausalLM(model_config).to(dtype).to(device)

        input_ids = torch.tensor([100, 200, 300, 100, 200, 100, 400, 500],
                                 dtype=torch.int,
                                 device=device)

        num_blocks = 1000
        tokens_per_block = 128

        if dtype == torch.half:
            kv_cache_dtype = tensorrt_llm.bindings.DataType.HALF
        elif dtype == torch.bfloat16:
            kv_cache_dtype = tensorrt_llm.bindings.DataType.BF16
        else:
            raise ValueError("Invalid dtype")

        mapping = Mapping(world_size=1, tp_size=1, rank=0)
        kv_cache_config = KvCacheConfig(max_tokens=num_blocks *
                                        tokens_per_block)

        num_layers = nemotron_nas.config.num_hidden_layers
        num_kv_heads = nemotron_nas.config.num_key_value_heads
        num_heads = nemotron_nas.config.num_attention_heads
        head_dim = nemotron_nas.config.hidden_size // num_heads
        max_seq_len = num_blocks * tokens_per_block

        context_sequence_lengths = [3, 2, 1]
        sequence_lengths = context_sequence_lengths + [1, 1]
        batch_size = len(sequence_lengths)
        past_seen_tokens = [0, 0, 0, 62, 75]
        request_ids = list(range(len(sequence_lengths)))
        token_nums = (torch.tensor(past_seen_tokens) +
                      torch.tensor(sequence_lengths)).tolist()
        prompt_lens = token_nums[:3] + past_seen_tokens[3:]

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
            logits = nemotron_nas.forward(input_ids=input_ids,
                                          position_ids=position_ids,
                                          attn_metadata=attn_metadata)

        self.assertEqual(len(past_seen_tokens), logits.shape[0])

        with torch.inference_mode():
            attn_metadata.prepare()
            logits = nemotron_nas.forward(input_ids=input_ids,
                                          position_ids=position_ids,
                                          attn_metadata=attn_metadata,
                                          return_context_logits=True)
        self.assertEqual(input_ids.shape, logits.shape[:-1])

        kv_cache_manager.shutdown()

    @parameterized.expand([
        Scenario(backend="VANILLA"),
        Scenario(backend="FLASHINFER"),
        Scenario(backend="TRTLLM"),
    ], lambda testcase_func, param_num, param:
                          f"{testcase_func.__name__}[{param.args[0]}]")
    @torch.no_grad()
    @unittest.skip("https://nvbugspro.nvidia.com/bug/5439817")
    def test_nemotron_nas_allclose_to_hf(self, scenario: Scenario) -> None:
        """
        Compare output to HF
        """
        backend = scenario.backend
        metadata_cls = get_attention_backend(backend).Metadata

        torch.random.manual_seed(0)
        config_dict = deepcopy(NEMOTRON_NAS_MINI_CONFIG)
        # 8B * sizeof(float16) plus some extra for activations
        # times 2, since we'll need 2 of these
        mem_for_full_model = (2 + 1) * 8 * 2**(30) * 4
        reduce_nemotron_nas_config(mem_for_full_model, config_dict)
        if config_dict["num_hidden_layers"] <= 0:
            self.skipTest("Insufficient memory for a single NemotronNAS layer")

        nemotron_nas_ckpt = llm_models_root(
        ) / "nemotron-nas/Llama-3_1-Nemotron-51B-Instruct"
        nemotron_nas_config = AutoConfig.from_pretrained(
            nemotron_nas_ckpt,
            trust_remote_code=True,
        )
        class_ref = nemotron_nas_config.auto_map["AutoModelForCausalLM"]
        nemotron_nas_config = nemotron_nas_config.from_dict(config_dict)
        dtype = nemotron_nas_config.torch_dtype
        device = torch.device('cuda')

        model_class = get_class_from_dynamic_module(class_ref,
                                                    nemotron_nas_ckpt)
        hf_nemotron_nas = model_class(nemotron_nas_config).to(dtype).to(
            device).eval()
        # This line populates the "variable" field in the NEED_SETUP_CACHE_CLASSES_MAPPING dict
        hf_nemotron_nas._prepare_generation_config(None)
        # And this line is the only way to access the only concrete Cache class DeciLMForCausalLM accepts
        VariableCache = transformers.generation.utils.NEED_SETUP_CACHE_CLASSES_MAPPING[
            "variable"]

        model_config = ModelConfig(pretrained_config=nemotron_nas_config,
                                   attn_backend=backend)
        nemotron_nas = NemotronNASForCausalLM(model_config).to(dtype).to(device)
        nemotron_nas.load_weights(hf_nemotron_nas.state_dict())

        num_blocks = 1
        tokens_per_block = 128

        kv_cache_config = KvCacheConfig(max_tokens=num_blocks *
                                        tokens_per_block)

        num_layers = nemotron_nas.config.num_hidden_layers
        num_kv_heads = nemotron_nas.config.num_key_value_heads
        num_heads = nemotron_nas.config.num_attention_heads
        head_dim = nemotron_nas.config.hidden_size // num_heads
        max_seq_len = num_blocks * tokens_per_block
        batch_size = 1

        mapping = Mapping(world_size=1, tp_size=1, rank=0)
        if dtype == torch.half:
            kv_cache_dtype = tensorrt_llm.bindings.DataType.HALF
        elif dtype == torch.bfloat16:
            kv_cache_dtype = tensorrt_llm.bindings.DataType.BF16
        else:
            raise ValueError("Invalid dtype")

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
            kv_cache_manager=kv_cache_manager,
            request_ids=request_ids,
            prompt_lens=prompt_lens,
            max_num_requests=1,
            max_num_tokens=8192,
        )

        position_ids = [torch.arange(0, input_ids.size(-1))]
        position_ids = torch.cat(position_ids).unsqueeze(0).cuda()
        # And, lastly, this is the simplest way of creating a Cache that `hf_nemotron_nas` will accept
        past_key_values = VariableCache(config=nemotron_nas_config,
                                        dtype=dtype,
                                        batch_size=1)
        with torch.inference_mode():
            attn_metadata.prepare()
            logits = nemotron_nas.forward(input_ids=input_ids,
                                          position_ids=position_ids,
                                          attn_metadata=attn_metadata)
            ref = hf_nemotron_nas.forward(input_ids=input_ids.unsqueeze(0),
                                          position_ids=position_ids,
                                          past_key_values=past_key_values,
                                          use_cache=True)

        torch.testing.assert_close(logits,
                                   ref.logits[:, -1].float(),
                                   atol=0.1,
                                   rtol=0.1)

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
            kv_cache_manager=kv_cache_manager,
            request_ids=request_ids,
            prompt_lens=prompt_lens,
            max_num_requests=1,
            max_num_tokens=8192,
        )

        gen_position_ids = [
            torch.arange(input_ids.size(-1),
                         input_ids.size(-1) + gen_input_ids.size(-1))
        ]
        gen_position_ids = torch.cat(gen_position_ids).unsqueeze(0).cuda()
        with torch.inference_mode():
            attn_metadata.prepare()
            logits = nemotron_nas.forward(input_ids=gen_input_ids,
                                          position_ids=gen_position_ids,
                                          attn_metadata=attn_metadata)
            ref = hf_nemotron_nas.forward(input_ids=gen_input_ids.unsqueeze(0),
                                          position_ids=gen_position_ids,
                                          past_key_values=ref.past_key_values,
                                          use_cache=True)

        torch.testing.assert_close(logits,
                                   ref.logits[:, -1].float(),
                                   atol=0.1,
                                   rtol=0.1)

        kv_cache_manager.shutdown()
