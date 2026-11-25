# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from transformers import PretrainedConfig

from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm.llmapi.llm_args import MiniMaxM3SparseAttentionConfig
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization.mode import QuantAlgo


@pytest.fixture
def make_m3_config():
    def make_config(
        tp: int,
        *,
        moe_layers: tuple[int, ...] = (0, 1),
        sparse_layers: tuple[int, ...] = (0, 1),
        head_dim: int = 128,
        fused_projection: bool = False,
    ) -> ModelConfig[PretrainedConfig]:
        return ModelConfig(
            pretrained_config=PretrainedConfig(
                dtype=torch.bfloat16,
                vocab_size=128,
                hidden_size=128,
                num_hidden_layers=len(moe_layers),
                num_attention_heads=tp,
                num_key_value_heads=tp,
                head_dim=head_dim,
                max_position_embeddings=32,
                dense_intermediate_size=256,
                shared_intermediate_size=256,
                intermediate_size=256,
                num_local_experts=8,
                num_experts_per_tok=2,
                n_shared_experts=1,
                rms_norm_eps=1e-6,
                moe_layer_freq=moe_layers,
                sparse_attention_config={
                    "sparse_attention_freq": sparse_layers,
                    "sparse_num_index_heads": tp,
                    "sparse_index_dim": 128,
                },
            ),
            mapping=Mapping(world_size=tp, tp_size=tp),
            quant_config=QuantConfig(kv_cache_quant_algo=QuantAlgo.FP8),
            sparse_attention_config=MiniMaxM3SparseAttentionConfig(
                implementation="msa",
                indexer_kv_dtype="fp8",
                sparse_num_index_heads=tp,
                fuse_qkv_index_projection=fused_projection,
            ),
            allreduce_strategy="NCCL",
            max_num_tokens=32,
        )

    return make_config


@pytest.fixture
def make_cuda_m3_config(make_m3_config):
    if not torch.cuda.is_available():
        pytest.skip("MiniMax-M3 constructors allocate CUDA resources")
    return make_m3_config
