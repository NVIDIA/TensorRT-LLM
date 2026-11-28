# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The build check of each sm_107 NVFP4 target (CPU).

``kimi_k31_pro_nvfp4__sm_107__{dep16,helix8,tep8}`` build no engine attention layer for their DSA
layers: each DSA layer passes the FP8 KV recipe's quantization state to the attention op itself, as
layer constants. ``_check_build`` admits a deployment of the target's model, layout and recipe (its
module constants) over the FP8 KV cache those constants are written for, and refuses the same
deployment over an NVFP4 cache. It runs on a stand-in config.
"""

import importlib
from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm.quantization.mode import QuantAlgo

pytestmark = pytest.mark.cpu_only

_PKG = "tensorrt_llm._torch.modeling_v2.models.kimi_k31"


@pytest.mark.parametrize("layout", ["dep16", "helix8", "tep8"])
def test_the_core_refuses_another_kv_cache(layout, monkeypatch):
    M = importlib.import_module(f"{_PKG}.kimi_k31_pro_nvfp4__sm_107__{layout}.modeling")
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a: M._COMPUTE_CAPABILITY)
    S, world, cp = SimpleNamespace, M._WORLD_SIZE, M._HELIX_CP
    text = S(**M._K31, linear_attn_config=dict(M._K31_KDA), dsa_config=dict(M._K31_DSA))
    mapping = S(
        world_size=world,
        tp_size=world // cp,
        pp_size=1,
        cp_size=cp,
        has_cp_helix=lambda: cp > 1,
        enable_attention_dp=M._ATTENTION_DP,
        moe_tp_size=M._MOE_TP,
        moe_ep_size=M._MOE_EP,
        moe_tp_ep_user_specified=True,
    )
    quant = S(quant_algo=M._QUANT_ALGO, kv_cache_quant_algo=QuantAlgo.FP8)
    quant.mamba_ssm_cache_dtype = M._STATE_DTYPE
    sparse = S(algorithm="dsa", indexer_k_dtype=M._INDEXER_K_DTYPE)
    spec = S(spec_dec_mode=S(is_dflash=lambda: True, use_one_engine=lambda: True))
    config = S(
        pretrained_config=text,
        mapping=mapping,
        quant_config=quant,
        sparse_attention_config=sparse,
        spec_config=spec,
    )
    M._check_build(config)
    quant.kv_cache_quant_algo = QuantAlgo.NVFP4
    with pytest.raises(ValueError, match="not FP8"):
        M._check_build(config)
