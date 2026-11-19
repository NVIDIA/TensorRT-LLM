# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The sm_107 targets' weight tables against the ModelOpt NVFP4 export, on the CPU.

Each target's weights.py owns its table (dep16, helix8, tep8). The table (``w1_rows`` and
``w2_rows``) of a generation server on a layout's last rank, against a snapshot of the export's
safetensors index and shard headers, no tensor bytes (the ``snapshot`` fixture): every source key
exists and every key is read or an expected leftover.
"""

from __future__ import annotations

import copy
import importlib
from types import SimpleNamespace

import pytest
import torch

#: K3.1's MLA (DSA) layers, numbered from 0; every other layer is KDA.
MLA_LAYERS = (*range(3, 88, 4), 91, 92)


#: K3.1's text config, as the export's config.json has it (what the weight tables read).
K31_TEXT = dict(
    hidden_size=7168,
    num_hidden_layers=93,
    vocab_size=163840,
    num_attention_heads=96,
    num_key_value_heads=96,
    q_lora_rank=1536,
    kv_lora_rank=512,
    qk_nope_head_dim=128,
    qk_rope_head_dim=64,
    v_head_dim=128,
    intermediate_size=33792,
    moe_intermediate_size=3072,
    num_experts=896,
    num_experts_per_token=16,
    num_shared_experts=2,
    first_k_dense_replace=1,
    moe_layer_freq=1,
    routed_expert_hidden_size=3584,
    attn_res_block_size=12,
    rms_norm_eps=1e-5,
    linear_attn_config={
        "full_attn_layers": [i + 1 for i in MLA_LAYERS],
        "kda_layers": [i + 1 for i in range(93) if i not in MLA_LAYERS],
        "head_dim": 128,
        "num_heads": 96,
        "short_conv_kernel_size": 4,
    },
    dsa_config={
        "dsa_layers": [i + 1 for i in MLA_LAYERS],
        "index_head_dim": 128,
        "index_k_norm_type": "layernorm",
        "index_num_heads": 32,
        "index_rope_dim": 0,
        "index_skip_topk_offset": 1,
        "index_topk": 2048,
        "index_topk_freq": 1,
    },
)


_MAPPING_ARGS = {
    # Attention DP over 16 ranks, the routed experts EP16.
    "dep16": dict(
        world_size=16, tp_size=16, moe_tp_size=1, moe_ep_size=16, enable_attention_dp=True
    ),
    # Helix CP over 8 ranks for MLA, the same ranks as TP8 elsewhere, MoE TP2 x EP4.
    "helix8": dict(
        world_size=8,
        tp_size=1,
        cp_size=8,
        cp_config={"cp_type": "HELIX"},
        moe_tp_size=2,
        moe_ep_size=4,
    ),
    # Attention TP over 8 ranks, the routed experts EP8.
    "tep8": dict(world_size=8, tp_size=8, moe_tp_size=1, moe_ep_size=8),
}


def mapping_args(layout: str) -> dict:
    """``layout``'s ``Mapping`` keyword arguments (the rank aside), a copy of its own: ``Mapping``
    converts the ``cp_type`` name in place."""
    return copy.deepcopy(_MAPPING_ARGS[layout])


def weights_core(dims, layout: str, role: str, rank: int = 0, vision_config=None, **model_config):
    """A core as a weight table's ``dims`` reads it: the server ``role``, the composite's
    ``vision_config`` and a model config of K3.1's text config, ``layout``'s mapping at ``rank``,
    BF16 and the fields ``model_config``; ``core.dims`` is ``dims(core)``."""
    import torch

    from tensorrt_llm._torch.configs import KimiLinearConfig
    from tensorrt_llm.mapping import Mapping

    core = SimpleNamespace(
        role=role,
        vision_config=vision_config,
        model_config=SimpleNamespace(
            pretrained_config=KimiLinearConfig(**copy.deepcopy(K31_TEXT)),
            mapping=Mapping(rank=rank, **mapping_args(layout)),
            torch_dtype=torch.bfloat16,
            **model_config,
        ),
    )
    core.dims = dims(core)
    return core


pytestmark = pytest.mark.cpu_only

_FAMILY = "tensorrt_llm._torch.modeling_v2.models.kimi_k31"

# --- the export, as its index and shard headers state it ---

#: K3.1's layers by kind: layer 0 has the dense MLP, 1-92 the latent MoE; the MLA (DSA) layers,
#: and the KDA layers (every other one).
_LAYERS = {
    "all": tuple(range(93)),
    "dense": (0,),
    "moe": tuple(range(1, 93)),
    "mla": MLA_LAYERS,
    "kda": tuple(i for i in range(93) if i not in MLA_LAYERS),
}
_EXPERTS = 896

# nvidia/kimi_k31_pro_nvfp4 rev 1d82032: its safetensors index and the headers of its 96 shards
# (992,691 tensors). The text model's: (key under "language_model.", dtype, shape, layers), {i}
# the layer and {e} the routed expert.
_L = "model.layers.{i}."
_ATTN = _L + "self_attn."
_MOE = _L + "block_sparse_moe."
_EXP = _MOE + "experts.{e}."
_CKPT = (
    ("lm_head.weight", "BF16", (163840, 7168), None),
    ("model.embed_tokens.weight", "BF16", (163840, 7168), None),
    (_EXP + "w1.input_scale", "F32", (), "moe"),
    (_EXP + "w1.weight", "U8", (3072, 1792), "moe"),
    (_EXP + "w1.weight_scale", "F8_E4M3", (3072, 224), "moe"),
    (_EXP + "w1.weight_scale_2", "F32", (), "moe"),
    (_EXP + "w2.input_scale", "F32", (), "moe"),
    (_EXP + "w2.weight", "U8", (3584, 1536), "moe"),
    (_EXP + "w2.weight_scale", "F8_E4M3", (3584, 192), "moe"),
    (_EXP + "w2.weight_scale_2", "F32", (), "moe"),
    (_EXP + "w3.input_scale", "F32", (), "moe"),
    (_EXP + "w3.weight", "U8", (3072, 1792), "moe"),
    (_EXP + "w3.weight_scale", "F8_E4M3", (3072, 224), "moe"),
    (_EXP + "w3.weight_scale_2", "F32", (), "moe"),
    (_MOE + "gate.e_score_correction_bias", "F32", (896,), "moe"),
    (_MOE + "gate.weight", "BF16", (896, 7168), "moe"),
    (_MOE + "routed_expert_down_proj.weight", "BF16", (3584, 7168), "moe"),
    (_MOE + "routed_expert_norm.weight", "BF16", (3584,), "moe"),
    (_MOE + "routed_expert_up_proj.weight", "BF16", (7168, 3584), "moe"),
    (_MOE + "shared_experts.down_proj.weight", "F8_E4M3", (7168, 6144), "moe"),
    (_MOE + "shared_experts.down_proj.weight_scale", "U8", (7168, 192), "moe"),
    (_MOE + "shared_experts.gate_proj.weight", "F8_E4M3", (6144, 7168), "moe"),
    (_MOE + "shared_experts.gate_proj.weight_scale", "U8", (6144, 224), "moe"),
    (_MOE + "shared_experts.up_proj.weight", "F8_E4M3", (6144, 7168), "moe"),
    (_MOE + "shared_experts.up_proj.weight_scale", "U8", (6144, 224), "moe"),
    (_L + "input_layernorm.weight", "BF16", (7168,), "all"),
    (_L + "mlp.down_proj.weight", "BF16", (7168, 33792), "dense"),
    (_L + "mlp.gate_proj.weight", "BF16", (33792, 7168), "dense"),
    (_L + "mlp.up_proj.weight", "BF16", (33792, 7168), "dense"),
    (_L + "mlp_res_norm.weight", "BF16", (7168,), "all"),
    (_L + "mlp_res_proj.weight", "BF16", (1, 7168), "all"),
    (_L + "post_attention_layernorm.weight", "BF16", (7168,), "all"),
    (_L + "self_attention_res_norm.weight", "BF16", (7168,), "all"),
    (_L + "self_attention_res_proj.weight", "BF16", (1, 7168), "all"),
    (_ATTN + "A_log", "F32", (128,), "kda"),
    (_ATTN + "b_proj.weight", "F8_E4M3", (96, 7168), "kda"),
    (_ATTN + "b_proj.weight_scale", "U8", (96, 224), "kda"),
    (_ATTN + "dt_bias", "F32", (12288,), "kda"),
    (_ATTN + "f_a_proj.weight", "F8_E4M3", (128, 7168), "kda"),
    (_ATTN + "f_a_proj.weight_scale", "U8", (128, 224), "kda"),
    (_ATTN + "f_b_proj.weight", "BF16", (12288, 128), "kda"),
    (_ATTN + "g_proj.weight", "BF16", (12288, 7168), "mla"),
    (_ATTN + "g_proj.weight", "F8_E4M3", (12288, 7168), "kda"),
    (_ATTN + "g_proj.weight_scale", "U8", (12288, 224), "kda"),
    (_ATTN + "indexer.k_norm.bias", "BF16", (128,), "mla"),
    (_ATTN + "indexer.k_norm.weight", "BF16", (128,), "mla"),
    (_ATTN + "indexer.weights_proj.weight", "BF16", (32, 7168), "mla"),
    (_ATTN + "indexer.wk.weight", "BF16", (128, 7168), "mla"),
    (_ATTN + "indexer.wq_b.weight", "BF16", (4096, 1536), "mla"),
    (_ATTN + "k_conv1d.weight", "F32", (12288, 1, 4), "kda"),
    (_ATTN + "k_proj.weight", "F8_E4M3", (12288, 7168), "kda"),
    (_ATTN + "k_proj.weight_scale", "U8", (12288, 224), "kda"),
    (_ATTN + "kv_a_layernorm.weight", "BF16", (512,), "mla"),
    (_ATTN + "kv_a_proj_with_mqa.weight", "BF16", (576, 7168), "mla"),
    (_ATTN + "kv_b_proj.weight", "BF16", (24576, 512), "mla"),
    (_ATTN + "o_norm.weight", "F32", (128,), "kda"),
    (_ATTN + "o_proj.weight", "BF16", (7168, 12288), "mla"),
    (_ATTN + "o_proj.weight", "F8_E4M3", (7168, 12288), "kda"),
    (_ATTN + "o_proj.weight_scale", "U8", (7168, 384), "kda"),
    (_ATTN + "q_a_layernorm.weight", "BF16", (1536,), "mla"),
    (_ATTN + "q_a_proj.weight", "BF16", (1536, 7168), "mla"),
    (_ATTN + "q_b_proj.weight", "BF16", (18432, 1536), "mla"),
    (_ATTN + "q_conv1d.weight", "F32", (12288, 1, 4), "kda"),
    (_ATTN + "q_proj.weight", "F8_E4M3", (12288, 7168), "kda"),
    (_ATTN + "q_proj.weight_scale", "U8", (12288, 224), "kda"),
    (_ATTN + "v_conv1d.weight", "F32", (12288, 1, 4), "kda"),
    (_ATTN + "v_proj.weight", "F8_E4M3", (12288, 7168), "kda"),
    (_ATTN + "v_proj.weight_scale", "U8", (12288, 224), "kda"),
    ("model.norm.weight", "BF16", (7168,), None),
    ("model.output_attn_res_norm.weight", "BF16", (7168,), None),
    ("model.output_attn_res_proj.weight", "BF16", (1, 7168), None),
)
# The vision tower's and projector's tensors: only their number matters here, never their names.
_VISION = {"vision_tower.": 165, "mm_projector.": 3}
_DTYPES = {
    "BF16": torch.bfloat16,
    "F32": torch.float32,
    "U8": torch.uint8,
    "F8_E4M3": torch.float8_e4m3fn,
}


@pytest.fixture(scope="module")
def snapshot() -> dict[str, tuple[torch.dtype, tuple[int, ...]]]:
    """Every checkpoint key -> (dtype, shape)."""
    keys = {}
    for pattern, dtype, shape, layers in _CKPT:
        spec = (_DTYPES[dtype], shape)
        template = "language_model." + pattern
        for i in _LAYERS[layers] if layers else (None,):
            key = template if i is None else template.replace("{i}", str(i))
            if "{e}" in key:
                keys.update({key.replace("{e}", str(e)): spec for e in range(_EXPERTS)})
            else:
                keys[key] = spec
    for prefix, count in _VISION.items():
        keys.update({f"{prefix}{j}": (torch.bfloat16, (1,)) for j in range(count)})
    assert len(keys) == 992_691
    return keys


def _keys(source) -> tuple[str, ...]:
    return source if isinstance(source, tuple) else (source,)


@pytest.mark.parametrize("layout", ["dep16", "helix8", "tep8"])
def test_table_covers_the_checkpoint(snapshot, layout):
    """Every source key exists, and every key is read or one of the leftovers the rank predicts."""
    W = importlib.import_module(f"{_FAMILY}.kimi_k31_pro_nvfp4__sm_107__{layout}.weights")
    core = weights_core(W.dims, layout, "generation", mapping_args(layout)["world_size"] - 1)
    consumed = {W.CKPT_LM_HEAD}
    rows = (*W.w1_rows(core.dims), *W.w2_rows(core.dims))
    for name, sources in W.manifest(core, rows).items():
        for key, _, _, _ in sources:
            missing = [k for k in _keys(key) if k not in snapshot]
            assert not missing, (name, missing)
            consumed.update(_keys(key))
    leftover = set(snapshot) - consumed
    expected = W.expected_unconsumed(core, snapshot)
    assert leftover == expected, (sorted(leftover - expected)[:5], sorted(expected - leftover)[:5])
