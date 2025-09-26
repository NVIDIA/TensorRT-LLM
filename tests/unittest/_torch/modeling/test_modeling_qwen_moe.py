import unittest
from copy import deepcopy
from dataclasses import dataclass

import torch
from _torch.helpers import create_mock_engine
from parameterized import parameterized
from transformers import Qwen2MoeConfig
from transformers import Qwen2MoeForCausalLM as HFQwen2MoeForCausalLM
from utils.util import getSMVersion

import tensorrt_llm
from tensorrt_llm._torch.attention_backend.utils import get_attention_backend
from tensorrt_llm._torch.metadata import KVCacheParams
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.checkpoints.hf.qwen2_moe_weight_mapper import \
    Qwen2MoeHfWeightMapper
from tensorrt_llm._torch.models.modeling_qwen_moe import Qwen2MoeForCausalLM
from tensorrt_llm._torch.pyexecutor.cuda_graph_runner import CUDAGraphRunner
from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager
from tensorrt_llm.bindings.executor import KvCacheConfig
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantConfig

QWEN2_57B_A14B_CONFIG = {
    "architectures": ["Qwen2MoeForCausalLM"],
    "attention_dropout": 0.0,
    "bos_token_id": 151643,
    "decoder_sparse_step": 1,
    "eos_token_id": 151643,
    "hidden_act": "silu",
    "hidden_size": 3584,
    "initializer_range": 0.02,
    "intermediate_size": 18944,
    "max_position_embeddings": 131072,
    "max_window_layers": 28,
    "model_type": "qwen2_moe",
    "moe_intermediate_size": 2560,
    "norm_topk_prob": False,
    "num_attention_heads": 28,
    "num_experts": 64,
    "num_experts_per_tok": 8,
    "num_hidden_layers": 28,
    "num_key_value_heads": 4,
    "output_router_logits": False,
    "rms_norm_eps": 1e-06,
    "rope_theta": 1000000.0,
    "router_aux_loss_coef": 0.001,
    "shared_expert_intermediate_size": 20480,
    "sliding_window": 131072,
    "tie_word_embeddings": False,
    "torch_dtype": "bfloat16",
    "transformers_version": "4.40.1",
    "use_cache": True,
    "use_sliding_window": False,
    "vocab_size": 151936
}


@dataclass(repr=False)
class Scenario:
    backend: str
    use_cuda_graph: bool = False

    def __repr__(self) -> str:
        return f"backend:{self.backend.lower()}-use_cuda_graph:{self.use_cuda_graph}"


class TestQwenMoe(unittest.TestCase):

    @parameterized.expand([None, "FP8", "NVFP4"])
    def test_qwen_moe_sanity(self, quant_algo):
        config_dict = deepcopy(QWEN2_57B_A14B_CONFIG)
        # Run a single layer
        config_dict["num_hidden_layers"] = 1
        qwen_moe_config = Qwen2MoeConfig.from_dict(config_dict)
        if quant_algo:
            quant_config = QuantConfig(quant_algo=quant_algo)
        else:
            quant_config = None
        if quant_algo == "FP8" and getSMVersion() < 90:
            self.skipTest(
                "This test is not supported in pre-Hopper architecture")
        if quant_algo == "NVFP4" and getSMVersion() < 100:
            self.skipTest(
                "This test is not supported in pre-Blackwell architecture")

        dtype = qwen_moe_config.torch_dtype
        device = torch.device("cuda")

        model_config = ModelConfig(pretrained_config=qwen_moe_config,
                                   quant_config=quant_config)
        qwen_moe = Qwen2MoeForCausalLM(model_config).to(device)

        input_ids = torch.tensor([100, 200, 300, 100, 200, 100, 400, 500],
                                 dtype=torch.int32,
                                 device=device)

        context_sequence_length = [3, 2, 1]
        sequence_length = context_sequence_length + [1, 1]
        past_seen_tokens = [0, 0, 0, 62, 75]
        request_ids = list(range(len(sequence_length)))
        token_nums = (torch.tensor(past_seen_tokens) +
                      torch.tensor(sequence_length)).tolist()
        prompt_lens = token_nums[:3] + past_seen_tokens[3:]

        num_blocks = 100
        tokens_per_block = 128
        head_dim = qwen_moe.config.hidden_size // qwen_moe.config.num_attention_heads
        num_layers = qwen_moe.config.num_hidden_layers
        qwen_moe.config.num_attention_heads
        num_kv_heads = qwen_moe.config.num_key_value_heads
        max_seq_len = num_blocks * tokens_per_block
        batch_size = len(sequence_length)

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

        position_ids = []
        for i, tokens in enumerate(past_seen_tokens):
            seq_len = context_sequence_length[i] if i < len(
                context_sequence_length) else 1
            position_id = torch.arange(tokens,
                                       tokens + seq_len,
                                       device=input_ids.device)
            position_ids.append(position_id)

        position_ids = torch.cat(position_ids).unsqueeze(0)

        with torch.inference_mode():
            attn_metadata.prepare()
            logits = qwen_moe.forward(input_ids=input_ids,
                                      position_ids=position_ids,
                                      attn_metadata=attn_metadata)
        self.assertEqual(len(past_seen_tokens), logits.shape[0])

        with torch.inference_mode():
            attn_metadata.prepare()
            logits = qwen_moe.forward(input_ids=input_ids,
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
    def test_qwen_moe_allclose_to_hf(self, scenario: Scenario):
        """
        Compare output to HF
        """
        backend = scenario.backend
        metadata_cls = get_attention_backend(backend).Metadata

        torch.random.manual_seed(0)

        config_dict = deepcopy(QWEN2_57B_A14B_CONFIG)
        # Run a single layer
        config_dict["num_hidden_layers"] = 1

        qwen_moe_config = Qwen2MoeConfig.from_dict(config_dict)
        dtype = qwen_moe_config.torch_dtype
        device = torch.device("cuda")

        hf_qwen_moe = HFQwen2MoeForCausalLM(qwen_moe_config).to(dtype).to(
            device).eval()

        model_config = ModelConfig(pretrained_config=qwen_moe_config,
                                   attn_backend=backend)
        qwen_moe = Qwen2MoeForCausalLM(model_config).to(device)
        weight_mapper = Qwen2MoeHfWeightMapper()
        weight_mapper.init_model_and_config(qwen_moe, qwen_moe_config)
        qwen_moe.load_weights(hf_qwen_moe.state_dict(), weight_mapper)

        num_blocks = 1
        tokens_per_block = 128
        head_dim = qwen_moe.config.hidden_size // qwen_moe.config.num_attention_heads
        num_layers = qwen_moe.config.num_hidden_layers
        qwen_moe.config.num_attention_heads
        num_kv_heads = qwen_moe.config.num_key_value_heads
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

        # context
        position_ids = [torch.arange(0, input_ids.size(-1))]
        position_ids = torch.cat(position_ids).unsqueeze(0).cuda()
        with torch.inference_mode():
            attn_metadata.prepare()
            logits = qwen_moe.forward(input_ids=input_ids,
                                      position_ids=position_ids,
                                      attn_metadata=attn_metadata)
            ref = hf_qwen_moe.forward(input_ids=input_ids.unsqueeze(0),
                                      position_ids=position_ids,
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

        graph_runner = None
        if scenario.use_cuda_graph:
            mock_engine = create_mock_engine(1)
            graph_runner = CUDAGraphRunner(mock_engine)

        def run_forward(input_ids, position_ids, attn_metadata):
            attn_metadata.prepare()
            if not scenario.use_cuda_graph:
                return qwen_moe.forward(input_ids=input_ids,
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
                                     lambda inputs: qwen_moe.forward(**inputs),
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
            ref = hf_qwen_moe.forward(input_ids=gen_input_ids.unsqueeze(0),
                                      position_ids=gen_position_ids,
                                      past_key_values=ref.past_key_values,
                                      use_cache=True)

        torch.testing.assert_close(logits,
                                   ref.logits[:, -1].float(),
                                   atol=0.1,
                                   rtol=0.1)
        if graph_runner is not None:
            graph_runner.clear()
        kv_cache_manager.shutdown()
