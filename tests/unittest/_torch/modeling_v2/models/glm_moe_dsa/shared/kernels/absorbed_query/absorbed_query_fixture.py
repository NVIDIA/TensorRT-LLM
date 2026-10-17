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
"""Real-weight native fixtures for the absorbed_query/projection package.

``build_native_attention`` constructs the built-in ``DeepseekV32Attention``
module that the GLM-5.2 decode engine uses for one target layer at TP8 rank
``tp_rank``, adopts it as the target's ``GlmMLA`` right after construction (as the
target's model does; on a decode step it runs the selected fast paths, and with none
selected or on any other step it computes the built-in attention) and places the
checkpoint tensors exactly as ``DeepseekV3WeightLoader`` does for the BF16 attention
weights:

* ``kv_a_proj_with_mqa.weight = cat([q_a_proj, kv_a_proj_with_mqa])``
* ``q_b_proj`` through ``Linear.load_weights`` (column-parallel slicing)
* ``k_b_proj_trans`` / ``v_b_proj`` / ``kv_b_proj`` from the TP slice of
  ``kv_b_proj`` (heads ``[r*8, r*8+8)``)
* indexer ``wq_b``/``wk``/``weights_proj``/``k_norm`` and
  ``Indexer.post_load_weights`` (fused ``wk``+``weights_proj``).

The attention linears of this checkpoint are excluded from NVFP4, so the
module is built with a KV-cache-only ``QuantConfig`` (FP8 KV), which is the
configuration ``apply_quant_config_exclude_modules`` leaves on them.

Fixture labels: weights are **real** (checkpoint tensors); activations are
**synthetic** unless a caller supplies captured ones.
"""

import json
import os
from typing import Dict, List, Optional

import torch
from utils.llm_data import llm_models_root

CKPT_DIR = os.environ.get(
    "GLM52_CKPT_DIR",
    os.path.join(str(llm_models_root()), "GLM-5.2-NVFP4"),
)
TP_SIZE = 8

# Full (own indexer) and shared (reused top-k) target layers of GLM-5.2.
FULL_LAYER = 6
SHARED_LAYER = 7


# Feature names of the neighbour packages every arm of a comparison runs on
# top of, the native reference included (e.g. "rope_kv_append" on an
# integration image that contains rope_kv_append).
BASE_FEATURES_ENV = "GLM_MOE_DSA_TEST_BASE_FEATURES"


def with_base_features(features: str) -> str:
    """Return the fast paths ``features`` as a comma-separated list.

    The fast paths of ``GLM_MOE_DSA_TEST_BASE_FEATURES`` follow them.
    """
    names = [f.strip() for f in features.split(",") if f.strip()]
    base = [f.strip() for f in os.environ.get(BASE_FEATURES_ENV, "").split(",") if f.strip()]
    return ",".join(names + [b for b in base if b not in names])


def checkpoint_available() -> bool:
    return os.path.isfile(os.path.join(CKPT_DIR, "model.safetensors.index.json"))


def load_layer_tensors(
    layer_idx: int, names: Optional[List[str]] = None
) -> Dict[str, torch.Tensor]:
    """Load ``model.layers.<layer_idx>.self_attn.*`` tensors (CPU)."""
    from safetensors import safe_open

    with open(os.path.join(CKPT_DIR, "model.safetensors.index.json")) as f:
        weight_map = json.load(f)["weight_map"]
    prefix = f"model.layers.{layer_idx}.self_attn."
    keys = [k for k in weight_map if k.startswith(prefix)]
    if names is not None:
        keys = [prefix + n for n in names if prefix + n in weight_map]
    by_shard: Dict[str, List[str]] = {}
    for k in keys:
        by_shard.setdefault(weight_map[k], []).append(k)
    out = {}
    for shard, shard_keys in by_shard.items():
        with safe_open(os.path.join(CKPT_DIR, shard), framework="pt") as f:
            for k in shard_keys:
                out[k[len(prefix) :]] = f.get_tensor(k)
    return out


def build_model_config(tp_rank: int = 0):
    """Native ``ModelConfig`` for GLM-5.2 at TP8 with the served DSA flags."""
    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm.llmapi.llm_args import DeepSeekSparseAttentionConfig, MTPDecodingConfig
    from tensorrt_llm.mapping import Mapping
    from tensorrt_llm.models.modeling_utils import QuantConfig
    from tensorrt_llm.quantization.mode import QuantAlgo

    mapping = Mapping(world_size=TP_SIZE, tp_size=TP_SIZE, rank=tp_rank, gpus_per_node=4)
    sparse = DeepSeekSparseAttentionConfig(
        enable_heuristic_topk=True, use_cute_dsl_paged_mqa_logits=True, use_cute_dsl_topk=True
    )
    model_config = ModelConfig.from_pretrained(
        CKPT_DIR,
        trust_remote_code=True,
        mapping=mapping,
        sparse_attention_config=sparse,
        spec_config=MTPDecodingConfig(max_draft_len=5),
        attn_backend="TRTLLM",
    )
    model_config._frozen = False
    # Attention linears are excluded from NVFP4 in this checkpoint; the
    # KV cache stays FP8 as in the served configuration.
    model_config.quant_config = QuantConfig(kv_cache_quant_algo=QuantAlgo.FP8)
    if model_config.pretrained_config.model_type == "glm_moe_dsa":
        model_config.pretrained_config.model_type = "deepseek_v32"
    model_config._frozen = True
    return model_config


def build_native_attention(
    layer_idx: int,
    tp_rank: int = 0,
    model_config=None,
    tensors: Optional[Dict[str, torch.Tensor]] = None,
    aux_stream: Optional[torch.cuda.Stream] = None,
):
    """Build the built-in ``DeepseekV32Attention``, adopted as ``GlmMLA``, with real weights on GPU."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.attention import (  # noqa: E501
        GlmMLA,
    )
    from tensorrt_llm._torch.models.modeling_deepseekv3 import DeepseekV32Attention
    from tensorrt_llm._torch.utils import AuxStreamType

    if model_config is None:
        model_config = build_model_config(tp_rank)
    if tensors is None:
        tensors = load_layer_tensors(layer_idx)
    if aux_stream is None:
        aux_stream = torch.cuda.Stream()
    attn = DeepseekV32Attention(
        model_config, layer_idx=layer_idx, aux_stream_dict={AuxStreamType.Attention: aux_stream}
    )
    GlmMLA.adopt(attn)
    # The model builds the FMHA libraries in ``create_weights`` after adoption; a module built
    # alone rebuilds them for the library adoption declared.
    for backend in (attn.mha, attn.mqa):
        if backend is not None and backend.prepended_fmha_lib_classes:
            backend.update_quant_config(backend.quant_config)
    attn = attn.cuda()

    cfg = model_config.pretrained_config
    num_heads = cfg.num_attention_heads
    nope, vdim = cfg.qk_nope_head_dim, cfg.v_head_dim
    local_heads = num_heads // TP_SIZE

    with torch.no_grad():
        fused_a = torch.cat(
            [tensors["q_a_proj.weight"], tensors["kv_a_proj_with_mqa.weight"]], dim=0
        )
        attn.kv_a_proj_with_mqa.weight.data.copy_(fused_a)
        attn.q_a_layernorm.weight.data.copy_(tensors["q_a_layernorm.weight"])
        attn.kv_a_layernorm.weight.data.copy_(tensors["kv_a_layernorm.weight"])
        attn.q_b_proj.load_weights(weights=[{"weight": tensors["q_b_proj.weight"]}])

        kv_b = tensors["kv_b_proj.weight"].unflatten(0, [num_heads, nope + vdim])
        kv_b = torch.chunk(kv_b, TP_SIZE, dim=0)[tp_rank].contiguous()
        k_nope, v_w = kv_b.split([nope, vdim], dim=1)
        attn.k_b_proj_trans.data.copy_(k_nope.transpose(2, 1).contiguous())
        attn.kv_b_proj.weight.data.copy_(
            torch.cat(
                [k_nope.reshape(local_heads * nope, -1), v_w.reshape(local_heads * vdim, -1)], dim=0
            )
        )
        attn.v_b_proj = torch.nn.Parameter(v_w.contiguous().cuda(), requires_grad=False)

        indexer = attn.mqa.indexer
        if indexer is not None:
            indexer.wq_b.load_weights(weights=[{"weight": tensors["indexer.wq_b.weight"]}])
            indexer.wk.load_weights(weights=[{"weight": tensors["indexer.wk.weight"]}])
            indexer.weights_proj.load_weights(
                weights=[{"weight": tensors["indexer.weights_proj.weight"]}]
            )
            indexer.k_norm.weight.data.copy_(tensors["indexer.k_norm.weight"])
            indexer.k_norm.bias.data.copy_(tensors["indexer.k_norm.bias"])
            indexer.post_load_weights()
    attn.post_load_weights()
    return attn


def make_hidden(
    num_rows: int, hidden_size: int = 6144, seed: int = 0, gamma: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Synthetic post-RMSNorm hidden state ``[R, hidden]`` (BF16).

    Unit-RMS Gaussian rows scaled by a real ``input_layernorm`` gain when one
    is given, so activation magnitudes match what the layer sees.
    """
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(num_rows, hidden_size, generator=g, device="cuda", dtype=torch.float32)
    if gamma is not None:
        x = x * gamma.float().cuda()
    return x.to(torch.bfloat16)


def load_input_layernorm(layer_idx: int) -> torch.Tensor:
    from safetensors import safe_open

    with open(os.path.join(CKPT_DIR, "model.safetensors.index.json")) as f:
        weight_map = json.load(f)["weight_map"]
    key = f"model.layers.{layer_idx}.input_layernorm.weight"
    with safe_open(os.path.join(CKPT_DIR, weight_map[key]), framework="pt") as f:
        return f.get_tensor(key)
