# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os

import pytest
import torch

from tensorrt_llm._torch.attention_backend.utils import get_attention_backend
from tensorrt_llm._torch.metadata import KVCacheParams
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.modeling_chatglm import ChatGLMForCausalLM
from tensorrt_llm._torch.models.modeling_utils import MODEL_CLASS_MAPPING
from tensorrt_llm._torch.pyexecutor.config_utils import load_pretrained_config
from tensorrt_llm.mapping import Mapping

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="ChatGLM3 parity tests require CUDA"
)


def _model_dir() -> str:
    d = os.environ.get("CHATGLM3_6B_MODEL_DIR")
    if d:
        return d
    from utils.llm_data import llm_models_root

    return str(llm_models_root() / "chatglm3-6b")


def _load_hf_chatglm(model_dir, dtype, device):
    from transformers import AutoConfig, AutoModelForCausalLM
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    cls = get_class_from_dynamic_module(
        "modeling_chatglm.ChatGLMForConditionalGeneration", model_dir
    )
    if not hasattr(cls, "all_tied_weights_keys"):
        cls.all_tied_weights_keys = {}

    config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    if getattr(config, "max_length", None) is None:
        config.max_length = config.seq_length
    return (
        AutoModelForCausalLM.from_pretrained(
            model_dir, trust_remote_code=True, torch_dtype=dtype, config=config
        )
        .to(device)
        .eval()
    )


def _trt_model_config(model_dir, backend="TRTLLM"):
    pretrained_config = load_pretrained_config(model_dir, trust_remote_code=True)
    return ModelConfig(pretrained_config=pretrained_config, attn_backend=backend)


def _build_kv_cache_manager(config, num_blocks=4, tokens_per_block=64, batch_size=1):
    import tensorrt_llm
    from tensorrt_llm._torch.pyexecutor.kv_cache_manager_v2 import KVCacheManagerV2
    from tensorrt_llm.llmapi.llm_args import KvCacheConfig

    kv_dtype = (
        tensorrt_llm.bindings.DataType.HALF
        if config.torch_dtype == torch.half
        else tensorrt_llm.bindings.DataType.BF16
    )
    max_seq_len = num_blocks * tokens_per_block
    kv_cache_config = KvCacheConfig(max_tokens=max_seq_len, enable_block_reuse=False)
    return KVCacheManagerV2(
        kv_cache_config,
        tensorrt_llm.bindings.internal.batch_manager.CacheType.SELF,
        num_layers=config.num_hidden_layers,
        num_kv_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
        tokens_per_block=tokens_per_block,
        max_seq_len=max_seq_len,
        max_batch_size=batch_size,
        mapping=Mapping(world_size=1, tp_size=1, rank=0),
        dtype=kv_dtype,
    )


def test_chatglm_config_recognition_and_normalization():
    config = load_pretrained_config(_model_dir(), trust_remote_code=True)

    assert config.model_type == "chatglm"
    assert config.architectures == ["ChatGLMModel"]
    assert config.num_hidden_layers == 28
    assert config.num_attention_heads == 32
    assert config.num_key_value_heads == 2
    assert config.head_dim == 128
    assert config.intermediate_size == 13696
    assert config.max_position_embeddings == 8192
    assert config.rms_norm_eps == 1e-5
    assert config.vocab_size == 65024
    assert config.tie_word_embeddings is False
    assert config.partial_rotary_factor == 0.5

    assert MODEL_CLASS_MAPPING["ChatGLMModel"] is ChatGLMForCausalLM


@torch.no_grad()
def test_chatglm_weight_loading_accounting():
    model_dir = _model_dir()
    dtype = torch.float16
    device = torch.device("cuda")

    hf = _load_hf_chatglm(model_dir, dtype, device)
    hf_sd = hf.state_dict()

    model_config = _trt_model_config(model_dir)
    model = ChatGLMForCausalLM(model_config).to(dtype).to(device)
    model.load_weights(hf_sd)

    learned_keys = [k for k in hf_sd if k != "transformer.rotary_pos_emb.inv_freq" and "." in k]
    assert len(learned_keys) == 199, f"unexpected key count: {len(learned_keys)}"

    layer0 = "transformer.encoder.layers.0."
    torch.testing.assert_close(
        model.model.layers[0].self_attn.qkv_proj.weight,
        hf_sd[layer0 + "self_attention.query_key_value.weight"].to(device),
    )
    torch.testing.assert_close(
        model.model.layers[0].self_attn.qkv_proj.bias,
        hf_sd[layer0 + "self_attention.query_key_value.bias"].to(device),
    )
    torch.testing.assert_close(
        model.model.layers[0].mlp.gate_up_proj.weight,
        hf_sd[layer0 + "mlp.dense_h_to_4h.weight"].to(device),
    )
    torch.testing.assert_close(
        model.model.layers[0].self_attn.o_proj.weight,
        hf_sd[layer0 + "self_attention.dense.weight"].to(device),
    )
    torch.testing.assert_close(
        model.lm_head.weight, hf_sd["transformer.output_layer.weight"].to(device)
    )
    assert model.lm_head.weight.data_ptr() != model.model.embed_tokens.weight.data_ptr()

    for name, p in model.named_parameters():
        assert torch.isfinite(p).all(), f"non-finite weights in {name}"


@torch.no_grad()
def test_chatglm_source_contracts():
    model_dir = _model_dir()
    dtype = torch.float16
    device = torch.device("cuda")
    backend = "TRTLLM"

    hf = _load_hf_chatglm(model_dir, dtype, device)
    model_config = _trt_model_config(model_dir, backend=backend)
    config = model_config.pretrained_config
    model = ChatGLMForCausalLM(model_config).to(dtype).to(device)
    model.load_weights(hf.state_dict())

    assert len(model.model.layers) == 28
    attn = model.model.layers[0].self_attn
    assert attn.num_heads == 32
    assert attn.num_key_value_heads == 2
    assert attn.head_dim == 128
    assert model.model.layers[0].input_layernorm.variance_epsilon == 1e-5
    assert attn.qkv_proj.bias is not None
    assert attn.o_proj.bias is None
    assert model.model.layers[0].mlp.gate_up_proj.bias is None
    assert attn.rotary_emb is not None
    assert attn.rotary_emb.is_neox is False
    assert attn.pos_embd_params.rope.dim == 64

    kv_cache_manager = _build_kv_cache_manager(config)
    try:
        metadata_cls = get_attention_backend(backend).Metadata

        input_ids = torch.tensor(
            [64790, 64792, 790, 30951, 517, 30910, 30939, 30996], dtype=torch.int32, device=device
        )
        request_ids = [1]
        prompt_lens = [input_ids.size(-1)]
        kv_cache_manager.add_dummy_requests(request_ids, [input_ids.size(-1)])

        attn_metadata = metadata_cls(
            seq_lens=torch.tensor([input_ids.size(-1)], dtype=torch.int),
            num_contexts=1,
            kv_cache_params=KVCacheParams(use_cache=True, num_cached_tokens_per_seq=[0]),
            max_num_requests=1,
            max_num_tokens=8192,
            kv_cache_manager=kv_cache_manager,
            request_ids=request_ids,
            prompt_lens=prompt_lens,
        )
        position_ids = torch.arange(
            0, input_ids.size(-1), dtype=torch.int32, device=device
        ).unsqueeze(0)

        with torch.inference_mode():
            attn_metadata.prepare()
            trt_logits = model.forward(
                input_ids=input_ids, position_ids=position_ids, attn_metadata=attn_metadata
            )
            hf_out = hf.forward(
                input_ids=input_ids.unsqueeze(0), position_ids=position_ids, use_cache=True
            )
            hf_logits = hf_out.logits[:, -1].float()

        assert trt_logits.argmax(-1).item() == hf_logits.argmax(-1).item()
        cos = torch.nn.functional.cosine_similarity(
            trt_logits.float().flatten(), hf_logits.flatten(), dim=0
        )
        assert cos > 0.99, f"context logit cosine {cos.item()}"

        gen_input_ids = torch.tensor([30910], dtype=torch.int32, device=device)
        attn_metadata = metadata_cls(
            seq_lens=torch.tensor([1], dtype=torch.int),
            num_contexts=0,
            kv_cache_params=KVCacheParams(
                use_cache=True, num_cached_tokens_per_seq=[input_ids.size(-1)]
            ),
            max_num_requests=1,
            max_num_tokens=8192,
            kv_cache_manager=kv_cache_manager,
            request_ids=request_ids,
            prompt_lens=prompt_lens,
        )
        gen_position_ids = torch.tensor([[input_ids.size(-1)]], dtype=torch.int32, device=device)
        with torch.inference_mode():
            attn_metadata.prepare()
            trt_logits = model.forward(
                input_ids=gen_input_ids, position_ids=gen_position_ids, attn_metadata=attn_metadata
            )
            hf_out = hf.forward(
                input_ids=gen_input_ids.unsqueeze(0),
                position_ids=gen_position_ids,
                past_key_values=hf_out.past_key_values,
                use_cache=True,
            )
            hf_logits = hf_out.logits[:, -1].float()

        assert trt_logits.argmax(-1).item() == hf_logits.argmax(-1).item()
    finally:
        kv_cache_manager.shutdown()
