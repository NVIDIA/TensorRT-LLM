import unittest
from copy import deepcopy
from dataclasses import dataclass

import torch
from parameterized import parameterized

try:
    from transformers import Exaone4Config
except ImportError:
    # TODO: Remove this once we have a proper transformers package
    from transformers import PretrainedConfig

    class Exaone4Config(PretrainedConfig):
        model_type = "exaone4"


SKIP_EXAONE4_HF_ACCURACY_TEST = False
try:
    from transformers import Exaone4ForCausalLM as HFExaone4ForCausalLM
except ImportError:
    # TODO: Remove this once we have a proper config for Exaone4
    SKIP_EXAONE4_HF_ACCURACY_TEST = True

from _torch.helpers import create_mock_engine
from transformers.cache_utils import HybridCache
from utils.util import getSMVersion

import tensorrt_llm
from tensorrt_llm._torch.attention_backend.utils import get_attention_backend
from tensorrt_llm._torch.metadata import KVCacheParams
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.modeling_exaone4 import Exaone4ForCausalLM
from tensorrt_llm._torch.pyexecutor.cuda_graph_runner import CUDAGraphRunner
from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager
from tensorrt_llm.bindings.executor import KvCacheConfig
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantConfig

WINDOW_SIZE = 4
EXAONE4_SINGLE_LAYER_CONFIG = {
    "architectures": ["Exaone4ForCausalLM"],
    "attention_dropout": 0.0,
    "bos_token_id": 1,
    "eos_token_id": 361,
    "head_dim": 128,
    "hidden_act": "silu",
    "hidden_size": 5120,
    "initializer_range": 0.02,
    "intermediate_size": 27392,
    "max_position_embeddings": 131072,
    "model_type": "exaone4",
    "num_attention_heads": 40,
    "num_hidden_layers":
    4,  #NOTE: For testing, we use 4 instead of 64(all layers)
    "num_key_value_heads": 8,
    "pad_token_id": 0,
    "rms_norm_eps": 1e-05,
    "rope_scaling": {
        "factor": 16.0,
        "high_freq_factor": 4.0,
        "low_freq_factor": 1.0,
        "original_max_position_embeddings": 8192,
        "rope_type": "llama3"
    },
    "rope_theta": 1000000,
    "sliding_window": 4,  # NOTE: For testing, we use 4 instead of 4096
    "sliding_window_pattern": 4,
    "tie_word_embeddings": False,
    "torch_dtype": "bfloat16",
    "transformers_version": "4.54.0.dev0",
    "use_cache": True,
    "vocab_size": 102400,
    "attn_implementation": "flash_attention_2"
}


@dataclass(repr=False)
class Scenario:
    backend: str
    input_len: int = WINDOW_SIZE - 1
    use_cuda_graph: bool = False

    def __repr__(self) -> str:
        return f"backend:{self.backend.lower()}-input_len:{self.input_len}-use_cuda_graph:{self.use_cuda_graph}"


class TestEXAONE4(unittest.TestCase):

    @parameterized.expand([None, "FP8"])
    def test_exaone4_sanity(self, quant_algo):
        config_dict = deepcopy(EXAONE4_SINGLE_LAYER_CONFIG)
        # TODO: Change to PretrainedConfig if we don't have the transformers version
        exaone4_config = Exaone4Config.from_dict(config_dict)
        if quant_algo:
            quant_config = QuantConfig(quant_algo=quant_algo)
        else:
            quant_config = None
        if quant_algo == "FP8" and getSMVersion() < 89:
            self.skipTest("This test is not supported in pre-Ada architecture")

        dtype = exaone4_config.torch_dtype
        device = torch.device('cuda')

        model_config = ModelConfig(pretrained_config=exaone4_config,
                                   quant_config=quant_config)
        exaone4 = Exaone4ForCausalLM(model_config).to(device)

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
        head_dim = exaone4.config.hidden_size // exaone4.config.num_attention_heads
        num_layers = exaone4.config.num_hidden_layers
        num_kv_heads = exaone4.config.num_key_value_heads
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
            logits = exaone4.forward(input_ids=input_ids,
                                     position_ids=position_ids,
                                     attn_metadata=attn_metadata)

        self.assertEqual(len(past_seen_tokens), logits.shape[0])

        with torch.inference_mode():
            attn_metadata.prepare()
            logits = exaone4.forward(input_ids=input_ids,
                                     position_ids=position_ids,
                                     attn_metadata=attn_metadata,
                                     return_context_logits=True)
        self.assertEqual(input_ids.shape, logits.shape[:-1])

        kv_cache_manager.shutdown()

    @parameterized.expand([
        Scenario(backend="TRTLLM", input_len=WINDOW_SIZE - 2),
        Scenario(
            backend="TRTLLM", input_len=WINDOW_SIZE - 2, use_cuda_graph=True),
    ], lambda testcase_func, param_num, param:
                          f"{testcase_func.__name__}[{param.args[0]}]")
    @torch.no_grad()
    def test_exaone4_allclose_to_hf(self, scenario: Scenario) -> None:
        """
        Compare output to HF
        """
        # TODO: Remove this once we have a proper transformers version for Exaone4
        if SKIP_EXAONE4_HF_ACCURACY_TEST:
            self.skipTest("Exaone4 is not supported in this environment")

        backend = scenario.backend
        metadata_cls = get_attention_backend(backend).Metadata

        torch.random.manual_seed(0)
        config_dict = deepcopy(EXAONE4_SINGLE_LAYER_CONFIG)
        exaone4_config = Exaone4Config.from_dict(config_dict)
        dtype = exaone4_config.torch_dtype
        device = torch.device('cuda')

        # TODO: Or change to PreTrainedModel
        hf_exaone4 = HFExaone4ForCausalLM(exaone4_config).to(dtype).to(
            device).eval()

        model_config = ModelConfig(pretrained_config=exaone4_config,
                                   attn_backend=backend)
        exaone4 = Exaone4ForCausalLM(model_config).to(dtype).to(device)
        exaone4.load_weights(hf_exaone4.state_dict())

        num_blocks = 1
        tokens_per_block = 128
        head_dim = getattr(
            exaone4.config, "head_dim",
            exaone4.config.hidden_size // exaone4.config.num_attention_heads)
        num_layers = exaone4.config.num_hidden_layers
        num_kv_heads = exaone4.config.num_key_value_heads
        max_seq_len = num_blocks * tokens_per_block
        batch_size = 1
        hf_cache = HybridCache(config=exaone4_config,
                               max_batch_size=batch_size,
                               max_cache_len=max_seq_len,
                               device=device,
                               dtype=dtype)
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
            max_attention_window=[int(exaone4_config.sliding_window)],
            max_tokens=num_blocks * tokens_per_block)
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
        input_ids = torch.tensor(
            [i * 100 for i in range(1, scenario.input_len + 1)],
            dtype=torch.int32,
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
        position_ids = [torch.arange(0, input_ids.size(-1), dtype=torch.int32)]
        position_ids = torch.cat(position_ids).unsqueeze(0).cuda()
        with torch.inference_mode():
            attn_metadata.prepare()
            logits = exaone4.forward(input_ids=input_ids,
                                     position_ids=position_ids,
                                     attn_metadata=attn_metadata)
            ref = hf_exaone4.forward(input_ids=input_ids.unsqueeze(0),
                                     position_ids=position_ids,
                                     past_key_values=hf_cache,
                                     use_cache=True)

        torch.testing.assert_close(logits,
                                   ref.logits[:, -1].float(),
                                   atol=0.4,
                                   rtol=0.4)

        # gen
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
            torch.arange(input_ids.size(-1),
                         input_ids.size(-1) + gen_input_ids.size(-1),
                         dtype=torch.int32)
        ]
        gen_position_ids = torch.cat(gen_position_ids).unsqueeze(0).cuda()

        graph_runner = None
        if scenario.use_cuda_graph:
            mock_engine = create_mock_engine(1)
            graph_runner = CUDAGraphRunner(mock_engine)

        def run_forward(input_ids, position_ids, attn_metadata):
            attn_metadata.prepare()
            if not scenario.use_cuda_graph:
                return exaone4.forward(input_ids=input_ids,
                                       position_ids=position_ids,
                                       attn_metadata=attn_metadata)
            else:
                inputs = {
                    "input_ids": input_ids,
                    "position_ids": position_ids,
                    "attn_metadata": attn_metadata,
                }
                key = (1, 0, False)
                graph_runner.capture(key,
                                     lambda inputs: exaone4.forward(**inputs),
                                     inputs)

                for _ in range(2):
                    # Run it twice. This helps us catch problems if buffers are accidentally reallocated
                    # in prepare().
                    attn_metadata.prepare()
                    logits = graph_runner.replay(key, inputs)
                return logits

        if scenario.use_cuda_graph:
            attn_metadata = attn_metadata.create_cuda_graph_metadata(1)

        with torch.inference_mode():
            logits = run_forward(input_ids=gen_input_ids,
                                 position_ids=gen_position_ids,
                                 attn_metadata=attn_metadata)
            ref = hf_exaone4.forward(
                input_ids=gen_input_ids.unsqueeze(0),  #hf_gen_input_ids,
                position_ids=gen_position_ids,
                past_key_values=ref.past_key_values,
                use_cache=True,
                cache_position=torch.LongTensor([input_ids.size(-1)
                                                 ]).to(device),
                last_cache_position=input_ids.size(-1) + 1)

        torch.testing.assert_close(logits,
                                   ref.logits[:, -1].float(),
                                   atol=0.4,
                                   rtol=0.4)
        if graph_runner is not None:
            graph_runner.clear()
        kv_cache_manager.shutdown()
