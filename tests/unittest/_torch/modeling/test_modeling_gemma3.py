import unittest
from copy import deepcopy
from dataclasses import dataclass

import torch
from parameterized import parameterized
from transformers import Gemma3Config
from transformers import Gemma3ForCausalLM as HFGemma3ForCausalLM
from transformers import Gemma3TextConfig
from transformers.cache_utils import HybridCache

import tensorrt_llm
from tensorrt_llm._torch.attention_backend import (AttentionMetadata,
                                                   FlashInferAttentionMetadata)
from tensorrt_llm._torch.attention_backend.utils import get_attention_backend
from tensorrt_llm._torch.metadata import KVCacheParams
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.checkpoints.hf.gemma3_weight_mapper import \
    Gemma3HfWeightMapper
from tensorrt_llm._torch.models.modeling_gemma3 import Gemma3ForCausalLM
from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager
from tensorrt_llm.bindings.executor import KvCacheConfig
from tensorrt_llm.mapping import Mapping

GEMMA3_1B_CONFIG = {
    "architectures": ["Gemma3ForCausalLM"],
    "attention_bias": False,
    "attention_dropout": 0.0,
    "attn_logit_softcapping": None,
    "bos_token_id": 2,
    "cache_implementation": "hybrid",
    "eos_token_id": [1, 106],
    "final_logit_softcapping": None,
    "head_dim": 256,
    "hidden_activation": "gelu_pytorch_tanh",
    "hidden_size": 1152,
    "initializer_range": 0.02,
    "intermediate_size": 6912,
    "max_position_embeddings": 32768,
    "model_type": "gemma3_text",
    "num_attention_heads": 4,
    "num_hidden_layers": 6,
    "num_key_value_heads": 1,
    "pad_token_id": 0,
    "query_pre_attn_scalar": 256,
    "rms_norm_eps": 1e-06,
    "rope_local_base_freq": 10000,
    "rope_scaling": None,
    "rope_theta": 1000000,
    "sliding_window": 4,
    "sliding_window_pattern": 6,
    "torch_dtype": "bfloat16",
    "transformers_version": "4.50.0.dev0",
    "use_cache": True,
    "vocab_size": 262144
}

GEMMA3_27B_CONFIG = {
    "architectures": ["Gemma3ForConditionalGeneration"],
    "boi_token_index": 255999,
    "eoi_token_index": 256000,
    "eos_token_id": [1, 106],
    "image_token_index": 262144,
    "initializer_range": 0.02,
    "mm_tokens_per_image": 256,
    "model_type": "gemma3",
    "text_config": {
        "head_dim": 128,
        "hidden_size": 5376,
        "intermediate_size": 21504,
        "model_type": "gemma3_text",
        "num_attention_heads": 32,
        "num_hidden_layers": 6,
        "num_key_value_heads": 16,
        "query_pre_attn_scalar": 168,
        "rope_scaling": {
            "factor": 8.0,
            "rope_type": "linear"
        },
        "sliding_window": 4,
        "sliding_window_pattern": 6,
    },
    "torch_dtype": "bfloat16",
    "transformers_version": "4.50.0.dev0",
    "vision_config": {
        "hidden_size": 1152,
        "image_size": 896,
        "intermediate_size": 4304,
        "model_type": "siglip_vision_model",
        "num_attention_heads": 16,
        "num_hidden_layers": 27,
        "patch_size": 14,
        "vision_use_head": False
    }
}


@dataclass(repr=False)
class Scenario:
    backend: str
    config_name: str

    def __repr__(self) -> str:
        return f"backend:{self.backend.lower()}_config:{self.config_name.lower()}"


class TestGemma3(unittest.TestCase):

    def get_kv_cache_manager(self, dtype: torch.dtype, config: Gemma3TextConfig,
                             tokens_per_block: int, max_seq_len: int,
                             batch_size: int, num_blocks: int):
        if dtype == torch.half:
            kv_cache_dtype = tensorrt_llm.bindings.DataType.HALF
        elif dtype == torch.bfloat16:
            kv_cache_dtype = tensorrt_llm.bindings.DataType.BF16
        else:
            raise ValueError("Invalid dtype")

        mapping = Mapping(world_size=1, tp_size=1, rank=0)
        kv_cache_config = KvCacheConfig(enable_block_reuse=False,
                                        enable_partial_reuse=False,
                                        copy_on_partial_reuse=False,
                                        max_tokens=num_blocks *
                                        tokens_per_block)
        kv_cache_manager = KVCacheManager(
            kv_cache_config,
            tensorrt_llm.bindings.internal.batch_manager.CacheType.SELF,
            num_layers=config.num_hidden_layers,
            num_kv_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            tokens_per_block=tokens_per_block,
            max_seq_len=max_seq_len,
            max_batch_size=batch_size,
            mapping=mapping,
            dtype=kv_cache_dtype,
        )
        return kv_cache_manager

    def test_gemma3_sanity(self):

        # Using 1B config for sanity test.
        config_dict = deepcopy(GEMMA3_1B_CONFIG)
        gemma3_config = Gemma3TextConfig.from_dict(config_dict)

        dtype = gemma3_config.torch_dtype
        device = torch.device('cuda')

        model_config = ModelConfig(pretrained_config=gemma3_config)
        gemma3 = Gemma3ForCausalLM(model_config).to(device)

        input_ids = torch.tensor([100, 200, 300, 400, 500, 600, 700, 800],
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
        max_seq_len = num_blocks * tokens_per_block
        batch_size = len(context_sequence_lengths) + 2
        kv_cache_manager = self.get_kv_cache_manager(
            dtype=dtype,
            config=gemma3_config,
            tokens_per_block=tokens_per_block,
            max_seq_len=max_seq_len,
            batch_size=batch_size,
            num_blocks=num_blocks)
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
            logits = gemma3.forward(input_ids=input_ids,
                                    position_ids=position_ids,
                                    attn_metadata=attn_metadata)

        self.assertEqual(len(past_seen_tokens), logits.shape[0])

        with torch.inference_mode():
            attn_metadata.prepare()
            logits = gemma3.forward(input_ids=input_ids,
                                    position_ids=position_ids,
                                    attn_metadata=attn_metadata,
                                    return_context_logits=True)
        self.assertEqual(input_ids.shape, logits.shape[:-1])

        kv_cache_manager.shutdown()

    def _verify_params_flushed_upon_prepare(self,
                                            attn_metadata: AttentionMetadata):
        # This check is valid only for FlashInferAttentionMetadata. It checks that the PlanParams specific
        # to forward call with custom mask exist right after the forward call and are flushed upon prepare.
        if isinstance(attn_metadata, FlashInferAttentionMetadata):
            # Right after forward call with custom mask, plan_params will have non-trivial attention_mask_data.
            # One for global-prefill, other for local-prefill.
            self.assertEqual(len(attn_metadata._plan_params_to_wrappers), 2)
            for plan_params in attn_metadata._plan_params_to_wrappers.keys():
                assert plan_params.attention_mask_data is not None
            # Prepare should flush the params with non-trivial attention_mask_data.
            attn_metadata.prepare()
            self.assertEqual(len(attn_metadata._plan_params_to_wrappers), 0)

    # Allow room for small fraction of elements to fail. This is to mitigate flakiness.
    def _assert_most_elems_close(self, actual_value, ref_value, atol, rtol,
                                 max_failed_fraction):
        matches = torch.isclose(actual_value, ref_value, atol=atol, rtol=rtol)
        failed_fraction = (~matches).float().mean().item()
        assert failed_fraction <= max_failed_fraction, (
            f"Exceeded tolerance: {failed_fraction*100:.2f}% of elements differ more than allowed "
            f"(max allowed {max_failed_fraction*100:.2f}%)")

    @parameterized.expand([
        Scenario(backend="TRTLLM", config_name="1B"),
        Scenario(backend="VANILLA", config_name="1B"),
        Scenario(backend="FLASHINFER", config_name="1B"),
        Scenario(backend="TRTLLM", config_name="27B"),
        Scenario(backend="VANILLA", config_name="27B"),
        Scenario(backend="FLASHINFER", config_name="27B"),
    ], lambda testcase_func, param_num, param:
                          f"{testcase_func.__name__}[{param.args[0]}]")
    @torch.no_grad()
    def test_gemma3_allclose_to_hf(self, scenario: Scenario) -> None:
        """
        Compare output to HF.
        """
        backend = scenario.backend
        config_name = scenario.config_name
        metadata_cls = get_attention_backend(backend).Metadata

        torch.random.manual_seed(0)

        # Select the appropriate config based on the scenario
        if config_name == "1B":
            config_dict = deepcopy(GEMMA3_1B_CONFIG)
        elif config_name == "27B":
            config_dict = deepcopy(GEMMA3_27B_CONFIG)
        else:
            raise ValueError(f"Unknown config_name: {config_name}")

        if config_name == "27B":
            gemma3_config = Gemma3Config.from_dict(config_dict)
            gemma3_config.text_config.torch_dtype = gemma3_config.torch_dtype
            gemma3_config = gemma3_config.text_config
        else:
            gemma3_config = Gemma3TextConfig.from_dict(config_dict)

        dtype = gemma3_config.torch_dtype
        device = torch.device('cuda')

        num_blocks = 1
        tokens_per_block = 128
        max_seq_len = num_blocks * tokens_per_block
        batch_size = 1

        hf_gemma3 = HFGemma3ForCausalLM(gemma3_config).to(dtype).to(
            device).eval()
        hf_cache = HybridCache(config=gemma3_config,
                               max_batch_size=batch_size,
                               max_cache_len=10,
                               device=device,
                               dtype=dtype)

        model_config = ModelConfig(pretrained_config=gemma3_config,
                                   attn_backend=backend)
        gemma3 = Gemma3ForCausalLM(model_config).to(dtype).to(device)
        weight_mapper = Gemma3HfWeightMapper()
        weight_mapper.init_model_and_config(gemma3, model_config)
        gemma3.load_weights(hf_gemma3.state_dict(), weight_mapper)

        kv_cache_manager = self.get_kv_cache_manager(
            dtype=dtype,
            config=gemma3_config,
            tokens_per_block=tokens_per_block,
            max_seq_len=max_seq_len,
            batch_size=batch_size,
            num_blocks=num_blocks)

        # Context phase.
        input_ids = torch.tensor([100, 200, 300, 400, 500, 600, 700, 800],
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
        position_ids = [torch.arange(0, input_ids.size(-1), dtype=torch.int32)]
        position_ids = torch.cat(position_ids).unsqueeze(0).cuda()

        # This helps us better test the custom masking utils for Gemma3 VLM as well
        # as SWA plumbing for FlashInfer. All tokens being text tokens should yield
        # same results as using global or local attention for appropriate layers.
        if backend == "FLASHINFER":
            image_token_mask = torch.tensor(
                [False, False, False, False, False, False, False, False],
                device=device)
        else:
            image_token_mask = None

        with torch.inference_mode():
            attn_metadata.prepare()
            logits = gemma3.forward(input_ids=input_ids,
                                    position_ids=position_ids,
                                    attn_metadata=attn_metadata,
                                    image_token_mask=image_token_mask)
            ref = hf_gemma3.forward(input_ids=input_ids.unsqueeze(0),
                                    position_ids=position_ids,
                                    past_key_values=hf_cache,
                                    use_cache=True)
            self._assert_most_elems_close(actual_value=logits,
                                          ref_value=ref.logits[:, -1].float(),
                                          atol=0.4,
                                          rtol=0.4,
                                          max_failed_fraction=0.001)
            self._verify_params_flushed_upon_prepare(attn_metadata)

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
            torch.arange(input_ids.size(-1),
                         input_ids.size(-1) + gen_input_ids.size(-1))
        ]
        gen_position_ids = torch.cat(gen_position_ids).unsqueeze(0).cuda()
        with torch.inference_mode():
            attn_metadata.prepare()
            logits = gemma3.forward(input_ids=gen_input_ids,
                                    position_ids=gen_position_ids,
                                    attn_metadata=attn_metadata)
            ref = hf_gemma3.forward(input_ids=gen_input_ids.unsqueeze(0),
                                    position_ids=gen_position_ids,
                                    past_key_values=hf_cache,
                                    use_cache=True,
                                    cache_position=torch.LongTensor(
                                        [input_ids.size(-1)]).to(device),
                                    last_cache_position=input_ids.size(-1) + 1)
            self._assert_most_elems_close(actual_value=logits,
                                          ref_value=ref.logits[:, -1].float(),
                                          atol=0.4,
                                          rtol=0.4,
                                          max_failed_fraction=0.001)

        kv_cache_manager.shutdown()

    def test_gemma3_flashinfer_mask(self):
        config_dict = deepcopy(GEMMA3_1B_CONFIG)
        gemma3_config = Gemma3TextConfig.from_dict(config_dict)

        dtype = gemma3_config.torch_dtype
        device = torch.device('cuda')

        model_config = ModelConfig(pretrained_config=gemma3_config,
                                   attn_backend="FLASHINFER")
        gemma3 = Gemma3ForCausalLM(model_config).to(device)

        input_ids = torch.tensor([100, 200, 300, 400, 500, 600, 700, 800],
                                 dtype=torch.int,
                                 device=device)

        # This initial setup is to populate KV cache so that attn_metadata has the info
        # needed for generating mask.
        context_sequence_lengths = [3, 2, 1]
        sequence_lengths = context_sequence_lengths + [1, 1]
        past_seen_tokens = [0, 0, 0, 2, 1]
        request_ids = list(range(len(sequence_lengths)))
        token_nums = (torch.tensor(past_seen_tokens) +
                      torch.tensor(sequence_lengths)).tolist()
        prompt_lens = token_nums[:3] + past_seen_tokens[3:]

        num_blocks = 100
        tokens_per_block = 128
        max_seq_len = num_blocks * tokens_per_block
        batch_size = len(context_sequence_lengths) + 2
        kv_cache_manager = self.get_kv_cache_manager(
            dtype=dtype,
            config=gemma3_config,
            tokens_per_block=tokens_per_block,
            max_seq_len=max_seq_len,
            batch_size=batch_size,
            num_blocks=num_blocks)
        kv_cache_manager.add_dummy_requests(request_ids, token_nums)

        metadata_cls = get_attention_backend(model_config.attn_backend).Metadata
        assert metadata_cls == FlashInferAttentionMetadata
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
        attn_metadata.prepare()

        # First sample has 2 image tokens, second sample has 2 image tokens, third sample has none.
        image_token_mask = torch.tensor(
            [True, True, False, True, True, True, False, False], device=device)
        causal_mask = gemma3.get_flashinfer_attention_mask(
            image_token_mask=image_token_mask, attn_metadata=attn_metadata)
        # Causal mask for context request 1.
        ctx_request_1_mask = torch.tensor(
            [[True, True, False], [True, True, False], [True, True, True]],
            device=device)
        # Causal mask for context request 2.
        ctx_request_2_mask = torch.tensor([[True, True], [True, True]],
                                          device=device)
        # Causal mask for context request 3.
        ctx_request_3_mask = torch.tensor([[True]], device=device)

        expected_causal_mask = torch.cat([
            ctx_request_1_mask.flatten(),
            ctx_request_2_mask.flatten(),
            ctx_request_3_mask.flatten(),
        ],
                                         dim=0)
        torch.testing.assert_close(causal_mask, expected_causal_mask)

    def single_image_mask_test_helper(
            self, effective_sliding_window: int) -> torch.Tensor:
        config_dict = deepcopy(GEMMA3_1B_CONFIG)
        gemma3_config = Gemma3TextConfig.from_dict(config_dict)
        device = torch.device('cuda')
        model_config = ModelConfig(pretrained_config=gemma3_config,
                                   attn_backend="FLASHINFER")
        gemma3 = Gemma3ForCausalLM(model_config).to(device)

        image_token_mask = torch.tensor(
            [False, False, True, True, True, True, False, False], device=device)
        return gemma3.get_context_mask(
            image_token_mask=image_token_mask,
            effective_sliding_window=effective_sliding_window)

    @parameterized.expand([
        "global",
        "sliding_window_larger_than_seq",
        "sliding_window_equal_to_seq",
    ], lambda testcase_func, param_num, param:
                          f"{testcase_func.__name__}[{param.args[0]}]")
    def test_gemma3_global_context_mask_single_image(self,
                                                     test_name: str) -> None:
        device = torch.device('cuda')
        effecive_sliding_window_map = {
            # For global mask, don't mention sliding window size.
            "global": None,
            # For a sliding window larger than sequence length, the local mask is the same as the global mask.
            "sliding_window_larger_than_seq": 10,
            # For a sliding window same as sequence length, the local mask is the same as the global mask.
            "sliding_window_equal_to_seq": 8,
        }
        effective_sliding_window = effecive_sliding_window_map[test_name]
        attention_mask = self.single_image_mask_test_helper(
            effective_sliding_window=effective_sliding_window)
        # Text tokens attend to each other in causal fashion. Image tokens attend in causal fashion
        # as well as to all other image tokens.
        expected_attention_mask = torch.tensor(
            [[1, 0, 0, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0, 0],
             [1, 1, 1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 1, 0, 0],
             [1, 1, 1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 1, 0, 0],
             [1, 1, 1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 1, 1, 1, 1]],
            device=device).bool()
        torch.testing.assert_close(attention_mask, expected_attention_mask)

    def test_gemma3_local_context_mask_single_image(self) -> None:
        device = torch.device('cuda')
        attention_mask = self.single_image_mask_test_helper(
            effective_sliding_window=2)
        expected_attention_mask = torch.tensor(
            [[1, 0, 0, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0, 0],
             [0, 1, 1, 1, 1, 1, 0, 0], [0, 0, 1, 1, 1, 1, 0, 0],
             [0, 0, 1, 1, 1, 1, 0, 0], [0, 0, 1, 1, 1, 1, 0, 0],
             [0, 0, 0, 0, 0, 1, 1, 0], [0, 0, 0, 0, 0, 0, 1, 1]],
            device=device).bool()
        torch.testing.assert_close(attention_mask, expected_attention_mask)

    def multi_image_mask_test_helper(
            self, effective_sliding_window: int) -> torch.Tensor:
        config_dict = deepcopy(GEMMA3_1B_CONFIG)
        gemma3_config = Gemma3TextConfig.from_dict(config_dict)
        device = torch.device('cuda')
        model_config = ModelConfig(pretrained_config=gemma3_config,
                                   attn_backend="FLASHINFER")
        gemma3 = Gemma3ForCausalLM(model_config).to(device)

        # 4 images with 4, 3, 2, 2 tokens respectively.
        image_token_mask = torch.tensor(
            [
                # text blob.
                False,
                False,
                # image1 blob.
                True,
                True,
                True,
                True,
                # text blob.
                False,
                False,
                # image2 blob.
                True,
                True,
                True,
                # text blob.
                False,
                False,
                # image3 blob.
                True,
                True,
                # text blob.
                False,
                False,
                # image4 blob.
                True,
                True,
                # text blob.
                False,
            ],
            device=device)
        return gemma3.get_context_mask(
            image_token_mask=image_token_mask,
            effective_sliding_window=effective_sliding_window)

    @parameterized.expand([
        "global",
        "sliding_window_larger_than_seq",
        "sliding_window_equal_to_seq",
    ], lambda testcase_func, param_num, param:
                          f"{testcase_func.__name__}[{param.args[0]}]")
    def test_gemma3_global_context_mask_multi_image(self,
                                                    test_name: str) -> None:
        device = torch.device('cuda')
        effecive_sliding_window_map = {
            # Don't mention sliding window size for global mask.
            "global": None,
            # For a sliding window larger than sequence length, the local mask is the same as the global mask.
            "sliding_window_larger_than_seq": 25,
            # For a sliding window same as sequence length, the local mask is the same as the global mask.
            "sliding_window_equal_to_seq": 20,
        }
        effective_sliding_window = effecive_sliding_window_map[test_name]

        # Text tokens attend to each other in causal fashion. Image tokens attend in causal fashion
        # as well as to all other image tokens.
        expected_attention_mask = torch.tensor(
            [
                # text blob.
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                # image1 blob.
                [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                # text blob.
                [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                # image2 blob.
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                # text blob.
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                # image3 blob.
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                # text blob.
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
                # image4 blob.
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                # text blob.
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            ],
            device=device).bool()
        attention_mask = self.multi_image_mask_test_helper(
            effective_sliding_window=effective_sliding_window)
        torch.testing.assert_close(attention_mask, expected_attention_mask)

    def test_gemma3_local_context_mask_multi_image(self) -> None:
        device = torch.device('cuda')
        attention_mask = self.multi_image_mask_test_helper(
            effective_sliding_window=3)

        # Text tokens attend to each other in causal fashion. Image tokens attend in causal fashion
        # as well as to all other image tokens.
        expected_attention_mask = torch.tensor(
            [
                # text blob.
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                # image1 blob.
                [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                # text blob.
                [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                # image2 blob.
                [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                # text blob.
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                # image3 blob.
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
                # text blob.
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
                # image4 blob.
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0],
                # text blob.
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
            ],
            device=device).bool()
        torch.testing.assert_close(attention_mask, expected_attention_mask)
