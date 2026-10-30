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

import torch
from _deepseek_v41_test_utils import _init_random_weights, cache_case
from utils.util import skip_blackwell_geforce, skip_pre_blackwell

from tensorrt_llm._torch.configs.deepseek_v41 import (
    DeepseekV41Config,
    DeepseekV41TextConfig,
    DeepseekV41VisionConfig,
)
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.modeling_deepseekv41_multimodal import (
    DeepseekV41ForConditionalGeneration,
)
from tensorrt_llm._torch.models.modeling_utils import get_registered_model_class
from tensorrt_llm.inputs.multimodal import MultimodalParams
from tensorrt_llm.llmapi.llm_args import CSA2SparseAttentionConfig
from tensorrt_llm.mapping import Mapping


@skip_pre_blackwell
@skip_blackwell_geforce
def test_conditional_generation_image_forward(monkeypatch):
    monkeypatch.setenv("TRTLLM_V41_DECODER_BOUNDED_REPLAY", "0")
    text = DeepseekV41TextConfig(
        vocab_size=1024,
        hidden_size=1024,
        moe_intermediate_size=256,
        num_hidden_layers=1,
        num_attention_heads=64,
        head_dim=512,
        qk_rope_head_dim=64,
        q_lora_rank=128,
        o_lora_rank=128,
        o_groups=8,
        n_routed_experts=256,
        num_experts_per_tok=6,
        compress_ratios=[0],
        kv_source_layer_ids=[],
        index_source_layer_ids=[],
        candidate_source_layer_id=None,
        engram_layer_ids=[],
        num_nextn_predict_layers=0,
        dspark_noise_token_id=3,
        engram_pad_token_id=2,
        pad_token_id=None,
        bos_token_id=1,
        eos_token_id=2,
        max_position_embeddings=2048,
        torch_dtype=torch.bfloat16,
    )
    config = DeepseekV41Config(
        text_config=text,
        vision_config=DeepseekV41VisionConfig(
            num_hidden_layers=1,
            hidden_size=16,
            num_attention_heads=2,
            intermediate_size=12,
            patch_size=2,
            downsample_ratio=3,
        ),
        image_token_id=127,
        architectures=["DeepseekV41ForConditionalGeneration"],
        pad_token_id=None,
        bos_token_id=1,
        eos_token_id=2,
    )
    config.dtype = torch.bfloat16
    config.tie_word_embeddings = False
    config.mapping = Mapping(world_size=1, tp_size=1, rank=0)
    sparse = CSA2SparseAttentionConfig()
    config.sparse_attention_config = sparse
    model_config = ModelConfig(
        pretrained_config=config,
        mapping=config.mapping,
        sparse_attention_config=sparse,
        attn_backend="TRTLLM",
    )
    assert (
        get_registered_model_class("DeepseekV41ForConditionalGeneration")
        is DeepseekV41ForConditionalGeneration
    )
    model = DeepseekV41ForConditionalGeneration(model_config)
    model.cuda().eval()
    _init_random_weights(model)
    assert len(model.model.layers) == 1
    assert len(model.vision.blocks) == 1
    assert model.aligner is not None

    with (
        torch.inference_mode(),
        cache_case(model_config, sparse, "auto", [6], [0], 1) as case,
    ):
        metadata, _, positions, buffers = case
        ids = torch.tensor([3, 127, 127, 127, 127, 4], dtype=torch.int32, device="cuda")
        pixels = torch.linspace(-1, 1, 108, device="cuda").reshape(9, 3, 2, 2)
        pixels = pixels.to(model.embedding_dtype)
        raw = MultimodalParams(
            multimodal_data={
                "image": {
                    "pixel_values": [pixels],
                    "image_grid_hw": [(3, 3)],
                    "llm_grid_hw": [(1, 1)],
                    "num_tokens": [4],
                }
            }
        )
        features = model.aligner(model.vision(pixels, 3, 3), 3, 3)
        image_embeddings = torch.cat(
            (
                model.image_start[None],
                features,
                model.image_newline[None],
                model.image_end[None],
            )
        )
        cached = MultimodalParams(multimodal_data={"multimodal_embedding": image_embeddings})
        mm_indices = torch.arange(1, 5, device="cuda")
        text_indices = torch.tensor([0, 5], device="cuda")

        outputs = []
        for params in ([raw], [cached], []):
            # Every comparison starts from the same cold context cache.
            for buffer in buffers:
                buffer.zero_()
            metadata.prepare()
            output = model(
                attn_metadata=metadata,
                input_ids=ids,
                position_ids=positions,
                multimodal_params=params,
                mm_token_indices=mm_indices,
                text_token_indices=text_indices,
                return_context_logits=True,
            )
            outputs.append(output.clone())

        raw_logits, cached_logits, text_logits = outputs
        assert raw_logits.shape == (6, text.vocab_size)
        assert raw_logits.dtype == torch.float32
        assert raw_logits.is_cuda
        assert torch.isfinite(raw_logits).all()
        torch.testing.assert_close(raw_logits, cached_logits, atol=0.02, rtol=0.02)

        fused = model.text_embedding_layer(ids).clone()
        fused[mm_indices] = image_embeddings
        bypassed_logits = model.lm_head(model.model.norm(fused)).float()
        assert not torch.allclose(raw_logits, bypassed_logits, atol=0.02, rtol=0.02)
        assert torch.isfinite(text_logits).all()
        assert not torch.allclose(raw_logits[-1], text_logits[-1], atol=1e-6, rtol=1e-5)

    del model
    torch.cuda.empty_cache()
