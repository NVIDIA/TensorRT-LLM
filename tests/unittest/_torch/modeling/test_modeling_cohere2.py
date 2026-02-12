import unittest
from copy import deepcopy

import torch
from transformers import Cohere2Config
from transformers import Cohere2ForCausalLM as HFCohere2ForCausalLM
from transformers.cache_utils import HybridCache

import tensorrt_llm
from tensorrt_llm._torch.attention_backend.utils import get_attention_backend
from tensorrt_llm._torch.metadata import KVCacheParams
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models import Cohere2ForCausalLM
from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager
from tensorrt_llm.bindings.executor import KvCacheConfig
from tensorrt_llm.mapping import Mapping

# Using a dummy configuration due to the large size of public models.
# Key parameter differences from 'CohereLabs/c4ai-command-a-03-2025':
# config = Cohere2Config(
#     hidden_size = 512,
#     intermediate_size = 1024,
#     num_attention_heads = 4,
#     num_key_value_heads = 2,
#     vocab_size = 256000,  # same as the proper model's to support its tokenizer
# )
COHERE2_SMALL_CONFIG = {
    "_sliding_window_pattern": 4,
    "attention_bias": False,
    "attention_dropout": 0.0,
    "bos_token_id": 5,
    "eos_token_id": 255001,
    "head_dim": 128,
    "hidden_act": "silu",
    "hidden_size": 512,
    "initializer_range": 0.02,
    "intermediate_size": 1024,
    "layer_norm_eps": 1e-05,
    "layer_types": [
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "full_attention",
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "full_attention",
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "full_attention",
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "full_attention",
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "full_attention",
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "full_attention",
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "full_attention",
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "full_attention",
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "full_attention",
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "full_attention",
    ],
    "logit_scale": 0.0625,
    "max_position_embeddings": 8192,
    "model_type": "cohere2",
    "num_attention_heads": 4,
    "num_hidden_layers": 40,
    "num_key_value_heads": 2,
    "pad_token_id": 0,
    "rope_scaling": None,
    "rope_theta": 10000.0,
    "sliding_window": 4096,
    "transformers_version": "4.56.0",
    "use_cache": True,
    "vocab_size": 256000,
}


class TestCohere2(unittest.TestCase):
    def get_kv_cache_manager(
        self,
        dtype: torch.dtype,
        config: Cohere2Config,
        tokens_per_block: int,
        max_seq_len: int,
        batch_size: int,
        num_blocks: int,
    ):
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

    def _assert_most_elems_close(self, actual_value, ref_value, atol, rtol, max_failed_fraction):
        matches = torch.isclose(actual_value, ref_value, atol=atol, rtol=rtol)
        failed_fraction = (~matches).float().mean().item()
        assert failed_fraction <= max_failed_fraction, (
            f"Exceeded tolerance: {failed_fraction * 100:.2f}% of elements differ more than allowed "
            f"(max allowed {max_failed_fraction * 100:.2f}%)"
        )

    @torch.no_grad()
    def test_cohere2_allclose_to_hf(self) -> None:
        """
        Compare output to HF
        """

        torch.random.manual_seed(0)
        config_dict = deepcopy(COHERE2_SMALL_CONFIG)

        cohere2_config = Cohere2Config.from_dict(config_dict)

        dtype = torch.bfloat16
        device = torch.device("cuda")

        # Inference parameters:
        num_blocks = 1
        tokens_per_block = 128
        max_seq_len = num_blocks * tokens_per_block
        batch_size = 1

        # Initialize the hugging face model
        hf_cohere2 = HFCohere2ForCausalLM(cohere2_config).to(dtype).to(device).eval()
        hf_cache = HybridCache(
            config=cohere2_config,
            max_batch_size=batch_size,
            max_cache_len=10,
            device=device,
            dtype=dtype,
        )

        # Initialize the TRT-LLM model
        model_config = ModelConfig(pretrained_config=cohere2_config)
        cohere2 = Cohere2ForCausalLM(model_config).to(dtype).to(device)
        cohere2.load_weights(hf_cohere2.state_dict())

        kv_cache_manager = self.get_kv_cache_manager(
            dtype=dtype,
            config=cohere2_config,
            tokens_per_block=tokens_per_block,
            max_seq_len=max_seq_len,
            batch_size=batch_size,
            num_blocks=num_blocks,
        )

        try:
            # Prefill phase
            input_ids = torch.tensor(
                [100, 200, 300, 400, 500, 600, 700, 800], dtype=torch.int32, device=device
            )
            num_cached_tokens_per_seq = [0]
            request_ids = [1]
            token_nums = [input_ids.size(-1)]
            prompt_lens = [input_ids.size(-1)]
            kv_cache_manager.add_dummy_requests(request_ids, token_nums)

            metadata_cls = get_attention_backend(model_config.attn_backend).Metadata
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
                logits = cohere2.forward(
                    input_ids=input_ids, position_ids=position_ids, attn_metadata=attn_metadata
                )
                ref = hf_cohere2.forward(
                    input_ids=input_ids.unsqueeze(0),
                    position_ids=position_ids,
                    past_key_values=hf_cache,
                    use_cache=True,
                )
                self._assert_most_elems_close(
                    actual_value=logits,
                    ref_value=ref.logits[:, -1].float(),
                    atol=0.4,
                    rtol=0.4,
                    max_failed_fraction=0.001,
                )

            # Generation phase
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
                torch.arange(input_ids.size(-1), input_ids.size(-1) + gen_input_ids.size(-1))
            ]
            gen_position_ids = torch.cat(gen_position_ids).unsqueeze(0).cuda()
            with torch.inference_mode():
                attn_metadata.prepare()
                logits = cohere2.forward(
                    input_ids=gen_input_ids,
                    position_ids=gen_position_ids,
                    attn_metadata=attn_metadata,
                )
                ref = hf_cohere2.forward(
                    input_ids=gen_input_ids.unsqueeze(0),
                    position_ids=gen_position_ids,
                    past_key_values=hf_cache,
                    use_cache=True,
                    cache_positions=torch.tensor(
                        [input_ids.size(-1)],
                        dtype=torch.long,
                    ).to(device),
                )
                self._assert_most_elems_close(
                    actual_value=logits,
                    ref_value=ref.logits[:, -1].float(),
                    atol=0.4,
                    rtol=0.4,
                    max_failed_fraction=0.001,
                )
        finally:
            kv_cache_manager.shutdown()
