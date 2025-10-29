import unittest
from copy import deepcopy
from dataclasses import dataclass

import torch
import transformers
from _torch.helpers import create_mock_engine
from parameterized import parameterized
from transformers import Llama4Config
from transformers import \
    Llama4ForConditionalGeneration as HFLlama4ForConditionalGeneration
from transformers.cache_utils import DynamicCache
from utils.util import default_dtype, getSMVersion

import tensorrt_llm
from tensorrt_llm._torch.attention_backend.utils import get_attention_backend
from tensorrt_llm._torch.metadata import KVCacheParams
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.checkpoints.hf.llama4_weight_mapper import \
    Llama4HfWeightMapper
from tensorrt_llm._torch.models.modeling_llama import \
    Llama4ForConditionalGeneration
from tensorrt_llm._torch.pyexecutor.config import PyTorchConfig
from tensorrt_llm._torch.pyexecutor.cuda_graph_runner import CUDAGraphRunner
from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager
from tensorrt_llm.bindings.executor import KvCacheConfig
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantConfig

# This is Llama4 Maverick config but with only 2 layers.
# 2 layers are needed to cover both MLP layer and MoE layer, as well as both
# RoPE and no-RoPE layers.
LLAMA_4_MAVERICK_TWO_LAYER_CONFIG = {
    "architectures": ["Llama4ForConditionalGeneration"],
    "boi_token_index": 200080,
    "eoi_token_index": 200081,
    "image_token_index": 200092,
    "model_type": "llama4",
    "text_config": {
        "_attn_implementation_autoset": True,
        "attention_bias": False,
        "attention_chunk_size": 8192,
        "attention_dropout": 0.0,
        "bos_token_id": 200000,
        "eos_token_id": [200001, 200007, 200008],
        "for_llm_compressor": False,
        "head_dim": 128,
        "hidden_act": "silu",
        "hidden_size": 5120,
        "initializer_range": 0.02,
        "interleave_moe_layer_step": 2,
        "intermediate_size": 8192,
        "intermediate_size_mlp": 16384,
        "max_position_embeddings": 1048576,
        "model_type": "llama4_text",
        # Added so that both RoPE and no-RoPE layers are present.
        "no_rope_layer_interval": 2,
        "num_attention_heads": 40,
        "num_experts_per_tok": 1,
        # Reduced to 2 layers from 48 layers.
        "num_hidden_layers": 2,
        "num_key_value_heads": 8,
        "num_local_experts": 128,
        "output_router_logits": False,
        "pad_token_id": 200018,
        "rms_norm_eps": 1e-05,
        "rope_scaling": None,
        "rope_theta": 500000.0,
        "router_aux_loss_coef": 0.001,
        "router_jitter_noise": 0.0,
        "torch_dtype": "bfloat16",
        "use_cache": True,
        "use_qk_norm": False,
        "vocab_size": 202048
    },
    "torch_dtype": "bfloat16",
    "transformers_version": "4.51.0.dev0",
    "vision_config": {
        "_attn_implementation_autoset": True,
        "attention_dropout": 0.0,
        "hidden_act": "gelu",
        "hidden_size": 1408,
        "image_size": 336,
        "initializer_range": 0.02,
        "intermediate_size": 5632,
        "model_type": "llama4_vision_model",
        "multi_modal_projector_bias": False,
        "norm_eps": 1e-05,
        "num_attention_heads": 16,
        "num_channels": 3,
        # Override to 0 because we don't need the vision model.
        "num_hidden_layers": 0,
        "patch_size": 14,
        "pixel_shuffle_ratio": 0.5,
        "projector_dropout": 0.0,
        "projector_input_dim": 4096,
        "projector_output_dim": 4096,
        "rope_theta": 10000,
        "vision_feature_layer": -1,
        "vision_feature_select_strategy": "default",
        "vision_output_dim": 4096
    }
}


@dataclass(repr=False)
class SanityScenario:
    quant_algo: str = None
    enable_min_latency: bool = False

    def __repr__(self) -> str:
        if self.quant_algo is not None:
            return f"quant_algo:{self.quant_algo}-enable_min_latency:{self.enable_min_latency}"

        return f"enable_min_latency:{self.enable_min_latency}"


@dataclass(repr=False)
class AllCloseScenario:
    use_cuda_graph: bool = False
    enable_min_latency: bool = False

    def __repr__(self) -> str:
        return f"use_cuda_graph:{self.use_cuda_graph}-enable_min_latency:{self.enable_min_latency}"


class TestLlama4MinLatency(unittest.TestCase):

    @parameterized.expand([
        SanityScenario(quant_algo=None, enable_min_latency=True),
        SanityScenario(quant_algo=None, enable_min_latency=False),
        SanityScenario(quant_algo="FP8", enable_min_latency=True),
        SanityScenario(quant_algo="FP8", enable_min_latency=False),
    ], lambda testcase_func, param_num, param:
                          f"{testcase_func.__name__}[{param.args[0]}]")
    def test_llama_sanity(self, scenario: SanityScenario):
        quant_algo = scenario.quant_algo
        enable_min_latency = scenario.enable_min_latency

        config_dict = deepcopy(LLAMA_4_MAVERICK_TWO_LAYER_CONFIG)
        # 17B * sizeof(float16) plus some extra for activations
        mem_for_full_model = (2 + 1) * 17 * 2**(30)
        _, total_mem = torch.cuda.mem_get_info()
        if total_mem < mem_for_full_model:
            self.skipTest("Insufficient memory for a two-layer Llama4 model")

        llama_config = Llama4Config.from_dict(config_dict)
        if quant_algo:
            quant_config = QuantConfig(quant_algo=quant_algo)
        else:
            quant_config = None
        if quant_algo == "FP8" and getSMVersion() < 89:
            self.skipTest("This test is not supported in pre-Ada architecture")

        dtype = llama_config.torch_dtype
        device = torch.device('cuda')

        with torch.device(device), default_dtype(dtype):
            model_config = ModelConfig(pretrained_config=llama_config,
                                       quant_config=quant_config)
            model_config.pytorch_backend_config = PyTorchConfig(
                enable_min_latency=enable_min_latency)
            llama = Llama4ForConditionalGeneration(model_config)

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

        if quant_algo == "FP8":
            kv_cache_dtype = tensorrt_llm.bindings.DataType.FP8
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
        AllCloseScenario(use_cuda_graph=False, enable_min_latency=False),
        AllCloseScenario(use_cuda_graph=False, enable_min_latency=True),
        AllCloseScenario(use_cuda_graph=True, enable_min_latency=False),
        AllCloseScenario(use_cuda_graph=True, enable_min_latency=True),
    ], lambda testcase_func, param_num, param:
                          f"{testcase_func.__name__}[{param.args[0]}]")
    @torch.no_grad()
    def test_llama_allclose_to_hf(self, scenario: AllCloseScenario) -> None:
        """
        Compare output to HF
        """
        enable_min_latency = scenario.enable_min_latency
        attention_backend = "TRTLLM"
        metadata_cls = get_attention_backend(attention_backend).Metadata

        if transformers.__version__ >= "4.55.0" \
            and transformers.__version__ < "4.56.1":
            self.skipTest(
                "The transformers between 4.55.0 and 4.56.1 have accuracy "
                "issues for Llama4. See: "
                "https://github.com/huggingface/transformers/pull/40609")

        torch.random.manual_seed(0)
        config_dict = deepcopy(LLAMA_4_MAVERICK_TWO_LAYER_CONFIG)
        # 17B * sizeof(float16) plus some extra for activations
        # times 2, since we'll need 2 of these
        mem_for_full_model = (2 + 1) * 17 * 2**(30) * 2
        _, total_mem = torch.cuda.mem_get_info()
        if total_mem < mem_for_full_model:
            self.skipTest("Insufficient memory for a two-layer Llama4 model")

        llama_config = Llama4Config.from_dict(config_dict)
        dtype = llama_config.torch_dtype
        device = torch.device('cuda')

        with torch.device(device), default_dtype(dtype):
            hf_llama = HFLlama4ForConditionalGeneration(llama_config).eval()

            model_config = ModelConfig(pretrained_config=llama_config,
                                       attn_backend=attention_backend)
            model_config.pytorch_backend_config = PyTorchConfig(
                enable_min_latency=enable_min_latency)
            llama = Llama4ForConditionalGeneration(model_config)
            weight_mapper = Llama4HfWeightMapper()
            weight_mapper.init_model_and_config(llama, model_config)
            llama.load_weights(hf_llama.state_dict(),
                               weight_mapper=weight_mapper)

        num_blocks = 1
        tokens_per_block = 128
        head_dim = llama.config.hidden_size // llama.config.num_attention_heads
        num_layers = llama.config.num_hidden_layers
        num_kv_heads = llama.config.num_key_value_heads
        max_seq_len = num_blocks * tokens_per_block
        batch_size = 1

        if dtype == torch.bfloat16:
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
            past_key_values = DynamicCache()
            ref = hf_llama.forward(input_ids=input_ids.unsqueeze(0),
                                   position_ids=position_ids,
                                   past_key_values=past_key_values,
                                   use_cache=True)

        # Allow up to 2% of mismatched values since BF16 has accuracy issues.
        mismatch_threshold = 0.02
        atol = 0.4
        rtol = 0.4
        ref_logits = ref.logits[:, -1].float()
        mismatch_count = torch.sum(
            torch.abs(logits - ref_logits) > (atol +
                                              rtol * torch.abs(ref_logits)))
        mismatch_ratio = mismatch_count / logits.numel()

        assert mismatch_ratio < mismatch_threshold, \
            f"Mismatch ratio {mismatch_ratio} exceeds threshold {mismatch_threshold}"

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
                return llama.forward(input_ids=input_ids,
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
                                     lambda inputs: llama.forward(**inputs),
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
            ref = hf_llama.forward(input_ids=gen_input_ids.unsqueeze(0),
                                   position_ids=gen_position_ids,
                                   past_key_values=ref.past_key_values,
                                   use_cache=True)

        # Allow up to 2% of mismatched values since BF16 has accuracy issues.
        mismatch_threshold = 0.02
        atol = 0.4
        rtol = 0.4
        ref_logits = ref.logits[:, -1].float()
        mismatch_count = torch.sum(
            torch.abs(logits - ref_logits) > (atol +
                                              rtol * torch.abs(ref_logits)))
        mismatch_ratio = mismatch_count / logits.numel()

        assert mismatch_ratio < mismatch_threshold, \
            f"Mismatch ratio {mismatch_ratio} exceeds threshold {mismatch_threshold}"

        kv_cache_manager.shutdown()
