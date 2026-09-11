# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterator
from types import SimpleNamespace

import pytest
import torch
from transformers import PretrainedConfig

import tensorrt_llm
from tensorrt_llm._torch.attention.backends.utils import get_attention_backend
from tensorrt_llm._torch.metadata import KVCacheParams
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.modeling_nemotron_nas import \
    NemotronNASForCausalLM
from tensorrt_llm._torch.models.modeling_utils import get_registered_model_class
from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager
from tensorrt_llm.bindings.executor import KvCacheConfig
from tensorrt_llm.mapping import Mapping


def _block_config(n_heads_in_group: int,
                  ffn_mult: float,
                  *,
                  no_op_ffn: bool = False) -> SimpleNamespace:
    return SimpleNamespace(
        attention=SimpleNamespace(
            no_op=n_heads_in_group == 0,
            n_heads_in_group=n_heads_in_group,
            replace_with_linear=False,
        ),
        ffn=SimpleNamespace(no_op=no_op_ffn,
                            ffn_mult=ffn_mult,
                            replace_with_linear=False),
    )


def _make_config() -> PretrainedConfig:
    # Exercise MHA, an FFN-only layer, an attention-only layer, and GQA.
    return PretrainedConfig(
        architectures=["DeciLMForCausalLM"],
        hidden_size=256,
        num_attention_heads=4,
        num_hidden_layers=4,
        vocab_size=128,
        torch_dtype=torch.float16,
        rms_norm_eps=1e-6,
        hidden_act="silu",
        max_position_embeddings=128,
        rope_theta=10000.0,
        rope_scaling=None,
        tie_word_embeddings=False,
        block_configs=[
            _block_config(1, 1.0),
            _block_config(0, 2.0),
            _block_config(2, 1.0, no_op_ffn=True),
            _block_config(4, 1.0),
        ],
    )


@pytest.fixture
def decilm_model() -> NemotronNASForCausalLM:
    config = _make_config()
    # DeciLM is the HF architecture name; NemotronNAS is its native implementation.
    assert get_registered_model_class(
        config.architectures[0]) is NemotronNASForCausalLM
    model_config = ModelConfig(pretrained_config=config, attn_backend="TRTLLM")
    model = NemotronNASForCausalLM(model_config).cuda().eval()

    generator = torch.Generator(device="cuda").manual_seed(0)
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.ndim == 1:
                # RMSNorm scales must preserve activations rather than suppress them.
                parameter.fill_(1.0)
            else:
                parameter.normal_(mean=0.0, std=0.02, generator=generator)
    model.post_load_weights()
    return model


@pytest.fixture
def decilm_cache(
        decilm_model: NemotronNASForCausalLM) -> Iterator[KVCacheManager]:
    config = decilm_model.config
    cache = KVCacheManager(
        KvCacheConfig(max_tokens=512, enable_block_reuse=False),
        tensorrt_llm.bindings.internal.batch_manager.CacheType.SELF,
        num_layers=config.num_hidden_layers,
        num_kv_heads=config.num_key_value_heads,
        head_dim=config.hidden_size // config.num_attention_heads,
        tokens_per_block=128,
        max_seq_len=128,
        max_batch_size=3,
        mapping=Mapping(world_size=1, tp_size=1, rank=0),
        dtype=tensorrt_llm.bindings.DataType.HALF,
    )
    try:
        # Each request has room for the complete eight-token sequence.
        cache.add_dummy_requests([0, 1, 2], [8, 8, 8])
        yield cache
    finally:
        cache.shutdown()


@torch.inference_mode()
def _forward(
    model: NemotronNASForCausalLM,
    cache: KVCacheManager,
    input_ids: torch.Tensor,
    *,
    request_id: int,
    num_cached_tokens: int = 0,
    prompt_len: int = 8,
) -> torch.Tensor:
    metadata_cls = get_attention_backend(
        model.model_config.attn_backend).Metadata
    metadata = metadata_cls(
        seq_lens=torch.tensor([input_ids.numel()], dtype=torch.int),
        num_contexts=int(num_cached_tokens == 0),
        kv_cache_params=KVCacheParams(
            use_cache=True, num_cached_tokens_per_seq=[num_cached_tokens]),
        kv_cache_manager=cache,
        request_ids=[request_id],
        prompt_lens=[prompt_len],
        max_num_requests=1,
        max_num_tokens=128,
    )
    position_ids = torch.arange(num_cached_tokens,
                                num_cached_tokens + input_ids.numel(),
                                device=input_ids.device).unsqueeze(0)
    metadata.prepare()
    return model(
        input_ids=input_ids,
        position_ids=position_ids,
        attn_metadata=metadata,
        return_context_logits=True,
    )


class TestDeciLMForCausalLM:

    def test_construction(self, decilm_model: NemotronNASForCausalLM) -> None:
        config = decilm_model.config
        layers = decilm_model.model.layers
        assert config.architectures == ["DeciLMForCausalLM"]
        assert len(layers) == 4
        assert config.num_key_value_heads == [4, 0, 2, 1]
        assert decilm_model.model.embed_tokens.weight.shape == (128, 256)
        assert decilm_model.lm_head.weight.shape == (128, 256)

        assert layers[0].self_attn.num_key_value_heads == 4
        assert layers[0].mlp.gate_up_proj.weight.shape == (512, 256)
        assert not hasattr(layers[1], "self_attn")
        assert not hasattr(layers[1], "input_layernorm")
        assert layers[1].mlp.gate_up_proj.weight.shape == (1024, 256)
        assert layers[2].self_attn.num_key_value_heads == 2
        assert not hasattr(layers[2], "mlp")
        assert not hasattr(layers[2], "post_attention_layernorm")
        assert layers[3].self_attn.num_key_value_heads == 1
        assert hasattr(layers[3], "mlp")

    def test_forward(self, decilm_model: NemotronNASForCausalLM,
                     decilm_cache: KVCacheManager) -> None:
        input_ids = torch.tensor([3, 5, 7, 11, 13, 17, 19, 23],
                                 dtype=torch.int,
                                 device="cuda")
        full_logits = _forward(decilm_model,
                               decilm_cache,
                               input_ids,
                               request_id=0)
        assert full_logits.shape == (8, decilm_model.config.vocab_size)
        assert full_logits.dtype == torch.float32
        assert torch.isfinite(full_logits).all()

        prefix_len = 5
        prefix_logits = _forward(
            decilm_model,
            decilm_cache,
            input_ids[:prefix_len],
            request_id=1,
            prompt_len=prefix_len,
        )
        torch.testing.assert_close(prefix_logits,
                                   full_logits[:prefix_len],
                                   atol=5e-3,
                                   rtol=5e-3)

        # Decode through the real paged cache, including layers with different KV-head counts.
        for position in range(prefix_len, input_ids.numel()):
            decode_logits = _forward(
                decilm_model,
                decilm_cache,
                input_ids[position:position + 1],
                request_id=1,
                num_cached_tokens=position,
                prompt_len=prefix_len,
            )
            torch.testing.assert_close(decode_logits,
                                       full_logits[position:position + 1],
                                       atol=5e-3,
                                       rtol=5e-3)

        # Future tokens must not alter prefix logits, and the model must react to changed inputs.
        changed_ids = input_ids.clone()
        changed_ids[prefix_len:] = torch.tensor([29, 31, 37],
                                                dtype=torch.int,
                                                device="cuda")
        changed_logits = _forward(decilm_model,
                                  decilm_cache,
                                  changed_ids,
                                  request_id=2)
        torch.testing.assert_close(changed_logits[:prefix_len],
                                   full_logits[:prefix_len],
                                   atol=5e-3,
                                   rtol=5e-3)
        assert not torch.allclose(changed_logits[prefix_len:],
                                  full_logits[prefix_len:],
                                  atol=5e-3,
                                  rtol=5e-3)
