# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""kimi_k3_nvfp4__sm_107__dep16: the MLA context dispatch of an FP4-cache chunked prefill
(``forward_context`` of the target's ``modeling.py``).

The split-K path it hands the prefill to (layer selection, the partition schedule, the MXFP8
cache gather and the chunked prefill against an FP32 reference) is the same code in every target
and is tested in ``kimi_k3_nvfp4__sm_107__tp1/test_k3_mla_ctx_split_k.py``."""

from types import SimpleNamespace

import pytest

from tensorrt_llm._torch.modeling_v2.models.kimi_k3_vl.kimi_k3_nvfp4__sm_107__dep16 import (
    modeling as context,
)

H, NOPE, ROPE, V, LORA = 96, 128, 64, 128, 512


def _layer(mha, kv_b_proj=None):
    """The attributes of the MLA layer the split-K context reads."""
    return SimpleNamespace(
        mha=mha,
        kv_b_proj=kv_b_proj,
        out_scale=None,
        kv_lora_rank=LORA,
        qk_rope_head_dim=ROPE,
        qk_nope_head_dim=NOPE,
        v_head_dim=V,
        num_heads_tp=H,
    )


@pytest.mark.cpu_only
def test_fp4_cache_chunked_prefill_runs_split_k_or_raises(monkeypatch):
    """The context dispatch hands an FP4-cache chunked prefill to the split-K context FMHA, and
    raises where it does not apply (there is no other FP4 partition loop)."""
    monkeypatch.setattr(context, "get_sm_version", lambda: 107)
    applies = {"value": True}
    monkeypatch.setattr(
        context, "uses_k3_split_k_context", lambda layer, q, output: applies["value"]
    )
    monkeypatch.setattr(
        context, "forward_context_split_k_fp4", lambda layer, q, ckv, latent, md, out: "split_k"
    )
    mha = SimpleNamespace(
        uses_fp4_mla_attention=True,
        has_cached_kv_for_mla_context_warmup=lambda md: False,
        is_chunked_prefill_for_mla_context=lambda md: True,
        has_cached_kv_for_mla_context=lambda md: True,
    )
    layer = _layer(mha)
    md = SimpleNamespace()
    run = lambda: context.forward_context(layer, None, None, md, None, None)  # noqa: E731
    assert run() == "split_k"
    applies["value"] = False
    with pytest.raises(NotImplementedError, match="split-K context FMHA"):
        run()
    mha.is_chunked_prefill_for_mla_context = lambda md: False
    with pytest.raises(RuntimeError, match="enable chunked prefill"):
        run()
