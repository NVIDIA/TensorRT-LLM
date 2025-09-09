import unittest
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import torch
from parameterized import parameterized
from transformers import AutoModel, Qwen2Config
from transformers import Qwen2ForCausalLM as HFQwenForCausalLM

# isort: off
import tensorrt_llm
from tensorrt_llm._torch.attention_backend.utils import get_attention_backend
from tensorrt_llm._torch.metadata import KVCacheParams
from tensorrt_llm._torch.model_config import ModelConfig
# yapf: disable
from tensorrt_llm._torch.models.modeling_qwen import (
    Qwen2ForCausalLM, Qwen2ForProcessRewardModel)
# yapf: enable
from _torch.helpers import create_mock_engine
from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager
from tensorrt_llm.bindings.executor import KvCacheConfig
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm._torch.pyexecutor.cuda_graph_runner import CUDAGraphRunner

from utils.llm_data import llm_models_root
from utils.util import getSMVersion
# isort: on

Qwen_2_7B_CONFIG = {
    "architectures": ["Qwen2ForCausalLM"],
    "attention_dropout": 0.0,
    "bos_token_id": 151643,
    "eos_token_id": 151643,
    "hidden_act": "silu",
    "hidden_size": 3584,
    "initializer_range": 0.02,
    "intermediate_size": 18944,
    "max_position_embeddings": 131072,
    "max_window_layers": 28,
    "model_type": "qwen2",
    "num_attention_heads": 28,
    "num_hidden_layers": 28,
    "num_key_value_heads": 4,
    "rms_norm_eps": 1e-06,
    "rope_theta": 1000000.0,
    "sliding_window": 131072,
    "tie_word_embeddings": False,
    "torch_dtype": "bfloat16",
    "transformers_version": "4.37.2",
    "use_cache": True,
    "use_sliding_window": False,
    "vocab_size": 152064
}

# Qwen2.5-Math-PRM-7B
Qwen_2_5_PRM_7B_CONFIG = deepcopy(Qwen_2_7B_CONFIG)
Qwen_2_5_PRM_7B_CONFIG.update({
    "architectures": ["Qwen2ForProcessRewardModel"],
    "eos_token_id": 151645,
    "max_position_embeddings": 4096,
    "rms_norm_eps": 1e-05,
    "rope_theta": 10000.0,
    "use_mrope": False
})

INPUT_IDS = torch.tensor([100, 200, 300, 100, 200, 100, 400, 500],
                         dtype=torch.int)

MODEL_CONFIGS = {"qwen": Qwen_2_7B_CONFIG, "qwen_prm": Qwen_2_5_PRM_7B_CONFIG}

MODEL_CLASS = {"qwen": Qwen2ForCausalLM, "qwen_prm": Qwen2ForProcessRewardModel}


@dataclass(repr=False)
class Scenario:
    backend: str
    model_name: str = "qwen"
    use_cuda_graph: bool = False

    def __repr__(self) -> str:
        return f"model:{self.model_name}-backend:{self.backend.lower()}-use_cuda_graph:{self.use_cuda_graph}"


def reduce_qwen_config(mem_for_full_model: int, config_dict: dict[str, Any]):
    _, total_mem = torch.cuda.mem_get_info()
    # scale model down if gpu memory is low
    if total_mem < mem_for_full_model:
        model_fraction = total_mem / mem_for_full_model
        num_layers = int(config_dict["num_hidden_layers"] * model_fraction)
        num_layers = min(num_layers, 32)
        config_dict["num_hidden_layers"] = num_layers


class TestQwen(unittest.TestCase):

    @parameterized.expand([None])  # TODO add FP8 Linear + Bias
    def test_qwen_sanity(self, quant_algo):
        model, input_ids, position_ids, past_seen_tokens, attn_metadata, kv_cache_manager = \
            self._prepare_sanity_test("qwen", quant_algo)

        with torch.inference_mode():
            attn_metadata.prepare()
            logits = model.forward(input_ids=input_ids,
                                   position_ids=position_ids,
                                   attn_metadata=attn_metadata)

        self.assertEqual(len(past_seen_tokens), logits.shape[0])

        with torch.inference_mode():
            attn_metadata.prepare()
            logits = model.forward(input_ids=input_ids,
                                   position_ids=position_ids,
                                   attn_metadata=attn_metadata,
                                   return_context_logits=True)
        self.assertEqual(input_ids.shape, logits.shape[:-1])

        kv_cache_manager.shutdown()

    @parameterized.expand([None])  # TODO add FP8 Linear + Bias
    def test_qwen_rm_sanity(self, quant_algo):
        model, input_ids, position_ids, _, attn_metadata, _ = \
            self._prepare_sanity_test("qwen_prm", quant_algo)

        with torch.inference_mode():
            attn_metadata.max_seq_len = input_ids.size(-1)
            attn_metadata.prepare()
            scores_logits = model.forward(input_ids=input_ids,
                                          position_ids=position_ids,
                                          attn_metadata=attn_metadata)
            scores_logits = scores_logits[0]

        self.assertEqual((attn_metadata.seq_lens[0], 2), scores_logits.shape)

    @parameterized.expand([
        Scenario(backend="VANILLA"),
        Scenario(backend="FLASHINFER"),
        Scenario(backend="FLASHINFER", use_cuda_graph=True),
        Scenario(backend="TRTLLM"),
        Scenario(backend="TRTLLM", use_cuda_graph=True),
    ], lambda testcase_func, param_num, param:
                          f"{testcase_func.__name__}[{param.args[0]}]")
    @torch.no_grad()
    def test_qwen_allclose_to_hf(self, scenario: Scenario) -> None:
        """
        Compare output to HF
        """
        backend = scenario.backend
        metadata_cls = get_attention_backend(backend).Metadata

        torch.random.manual_seed(0)
        config_dict = deepcopy(Qwen_2_7B_CONFIG)
        # 8B * sizeof(float16) plus some extra for activations
        # times 2, since we'll need 2 of these
        mem_for_full_model = (2 + 1) * 8 * 2**(30) * 4
        reduce_qwen_config(mem_for_full_model, config_dict)
        if config_dict["num_hidden_layers"] <= 0:
            self.skipTest("Insufficient memory for a single Qwen layer")
        qwen_config = Qwen2Config.from_dict(config_dict)
        dtype = qwen_config.torch_dtype
        device = torch.device('cuda')

        hf_qwen = HFQwenForCausalLM(qwen_config).to(dtype).to(device).eval()

        model_config = ModelConfig(pretrained_config=qwen_config,
                                   attn_backend=backend)
        qwen = Qwen2ForCausalLM(model_config).to(dtype).to(device)
        qwen.load_weights(hf_qwen.state_dict())

        num_blocks = 1
        tokens_per_block = 128
        head_dim = qwen.config.hidden_size // qwen.config.num_attention_heads
        num_layers = qwen.config.num_hidden_layers
        num_kv_heads = qwen.config.num_key_value_heads
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
        input_ids = INPUT_IDS.clone().to(device)

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
            logits = qwen.forward(input_ids=input_ids,
                                  position_ids=position_ids,
                                  attn_metadata=attn_metadata)
            ref = hf_qwen.forward(input_ids=input_ids.unsqueeze(0),
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

        graph_runner = None
        if scenario.use_cuda_graph:
            mock_engine = create_mock_engine(1)
            graph_runner = CUDAGraphRunner(mock_engine)

        def run_forward(input_ids, position_ids, attn_metadata):
            attn_metadata.prepare()
            if not scenario.use_cuda_graph:
                return qwen.forward(input_ids=input_ids,
                                    position_ids=position_ids,
                                    attn_metadata=attn_metadata)
            else:
                inputs = {
                    "input_ids": input_ids,
                    "position_ids": position_ids,
                    "attn_metadata": attn_metadata,
                }
                key = (1, 0, False)
                graph_runner.capture(key, lambda inputs: qwen.forward(**inputs),
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
            ref = hf_qwen.forward(input_ids=gen_input_ids.unsqueeze(0),
                                  position_ids=gen_position_ids,
                                  past_key_values=ref.past_key_values,
                                  use_cache=True)

        torch.testing.assert_close(logits,
                                   ref.logits[:, -1].float(),
                                   atol=0.4,
                                   rtol=0.4)
        if graph_runner is not None:
            graph_runner.clear()
        kv_cache_manager.shutdown()

    @parameterized.expand(
        [
            Scenario(backend="VANILLA", model_name="qwen_prm"
                     )  # Currently, only Vanilla Attn supports no KV Cache
        ],
        lambda testcase_func, param_num, param:
        f"{testcase_func.__name__}[{param.args[0]}]")
    @torch.no_grad()
    def test_qwen_rm_allclose_to_hf(self, scenario: Scenario) -> None:
        """
        Compare output to HF for Qwen reward models
        """
        _, total_men = torch.cuda.mem_get_info()
        if total_men < 80 * 1000**3:
            self.skipTest("Only test with 80GB nodes")

        backend = scenario.backend
        metadata_cls = get_attention_backend(backend).Metadata

        torch.random.manual_seed(0)
        config_dict = deepcopy(Qwen_2_5_PRM_7B_CONFIG)

        qwen_config = Qwen2Config.from_dict(config_dict)
        dtype = qwen_config.torch_dtype
        device = torch.device('cuda')

        model_dir = str(llm_models_root() / "Qwen2.5-Math-PRM-7B")

        # Qwen2ForProcessRewardModel definition is in the modeling file of ckpt
        # instead of in transformers, so trust_remote_code must be set to True.
        hf_qwen = AutoModel.from_pretrained(
            model_dir, trust_remote_code=True).to(dtype).to(device).eval()

        model_config = ModelConfig(pretrained_config=hf_qwen.config,
                                   attn_backend=backend)
        qwen = Qwen2ForProcessRewardModel(model_config).to(dtype).to(device)
        qwen.load_weights(hf_qwen.state_dict())

        # context
        input_ids = INPUT_IDS.clone().to(device)
        prompt_lens = [input_ids.size(-1)]

        attn_metadata = metadata_cls(
            seq_lens=torch.tensor([input_ids.size(-1)], dtype=torch.int),
            num_contexts=1,
            max_num_requests=1,
            max_num_tokens=8192,
            kv_cache_manager=None,
            request_ids=[1],
            prompt_lens=prompt_lens,
        )

        position_ids = [torch.arange(0, input_ids.size(-1))]
        position_ids = torch.cat(position_ids).unsqueeze(0).cuda()
        with torch.inference_mode():
            logits = qwen.forward(input_ids=input_ids,
                                  position_ids=position_ids,
                                  attn_metadata=attn_metadata)
            logits = logits[0]
            ref = hf_qwen.forward(input_ids=input_ids.unsqueeze(0),
                                  position_ids=position_ids,
                                  use_cache=False)

        # with actual weights, QwenPRM can pass atol/rtol=1e-1.
        # Here we use a loosen threshold to avoid falky.
        torch.testing.assert_close(logits,
                                   ref.logits.squeeze(0),
                                   atol=0.4,
                                   rtol=0.4)

    def _prepare_sanity_test(self, model_name, quant_algo):
        config_dict = deepcopy(MODEL_CONFIGS[model_name])
        model_class = MODEL_CLASS[model_name]

        # 7B * sizeof(float16) plus some extra for activations
        mem_for_full_model = (2 + 1) * 7 * 2**(30)
        reduce_qwen_config(mem_for_full_model, config_dict)
        if config_dict["num_hidden_layers"] <= 0:
            self.skipTest("Insufficient memory for a single Qwen layer")
        qwen_config = Qwen2Config.from_dict(config_dict)
        if quant_algo:
            quant_config = QuantConfig(quant_algo=quant_algo)
        else:
            quant_config = None
        if quant_algo == "FP8" and getSMVersion() < 90:
            self.skipTest(
                "This test is not supported in pre-Hopper architecture")

        dtype = qwen_config.torch_dtype
        device = torch.device('cuda')

        model_config = ModelConfig(pretrained_config=qwen_config,
                                   quant_config=quant_config)
        model = model_class(model_config).to(device)

        input_ids = INPUT_IDS.clone().to(device)

        context_sequence_lengths = [3, 2, 1]
        sequence_lengths = context_sequence_lengths + [1, 1]
        past_seen_tokens = [0, 0, 0, 62, 75]
        request_ids = list(range(len(sequence_lengths)))
        token_nums = (torch.tensor(past_seen_tokens) +
                      torch.tensor(sequence_lengths)).tolist()
        prompt_lens = token_nums[:3] + past_seen_tokens[3:]

        if model_config.is_generation:
            num_blocks = 100
            tokens_per_block = 128
            head_dim = model.config.hidden_size // model.config.num_attention_heads
            num_layers = model.config.num_hidden_layers
            num_kv_heads = model.config.num_key_value_heads
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

            attn_metadata_args = {
                "kv_cache_params":
                KVCacheParams(
                    use_cache=True,
                    num_cached_tokens_per_seq=past_seen_tokens,
                )
            }
        else:
            kv_cache_manager = None
            attn_metadata_args = {}

        metadata_cls = get_attention_backend(model_config.attn_backend).Metadata
        attn_metadata = metadata_cls(
            seq_lens=torch.tensor(sequence_lengths, dtype=torch.int),
            num_contexts=len(context_sequence_lengths)
            if model_config.is_generation else len(sequence_lengths),
            kv_cache_manager=kv_cache_manager,
            request_ids=request_ids,
            prompt_lens=prompt_lens,
            max_num_requests=len(context_sequence_lengths) + 2,
            max_num_tokens=8192,
            **attn_metadata_args,
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

        return model, input_ids, position_ids, past_seen_tokens, attn_metadata, kv_cache_manager
