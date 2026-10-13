# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""deepseek's weight table declares what the hand-written code did.

The declaration reference below is a transcription of the __init__ block the table
replaced. The checkpoint-key side was proven differently: while the hand-written
`_manifest` still existed, a test compared the generated manifest against it, per key
and per source, with MTP on and off -- green in CI job 1081689 -- and both the
function and that test were deleted together. One renaming was deliberate: the
hand-written manifest called the MTP module's rows `mtp_*`; the table addresses it as
layer `num_hidden_layers`, which is what the checkpoint calls it, so they are `l61_*`.

No GPU: nothing here allocates, and `dims` reads `model_config` alone.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

__extra_import_path__ = [".."]  # noqa: F841 -- repo's file-scoped import hook

from tensorrt_llm._torch._experimental.modeling_v2.models.deepseek_v3.r1_0528_nvfp4__sm_103__dep4 import (  # noqa: E501
    weights as W,
)

# deepseek-r1-0528's own numbers, so the comparison is against the shapes this target
# actually ships rather than a toy configuration.
_CFG = dict(
    num_hidden_layers=61,
    hidden_size=7168,
    num_attention_heads=128,
    qk_nope_head_dim=128,
    qk_rope_head_dim=64,
    v_head_dim=128,
    kv_lora_rank=512,
    q_lora_rank=1536,
    vocab_size=129280,
    first_k_dense_replace=3,
    n_routed_experts=256,
    moe_intermediate_size=2048,
    n_shared_experts=1,
    intermediate_size=18432,
)
_EP_SIZE, _EP_RANK = 4, 1


def _core(mtp: bool):
    """The configuration surface both manifests read.

    `local_experts` and `expert_offset` are set on the core by its `__init__` and read
    by the hand-written manifest; `dims` derives the same two from the mapping, so the
    comparison also checks that those two derivations agree.
    """
    cfg = SimpleNamespace(**_CFG)
    mapping = SimpleNamespace(moe_ep_size=_EP_SIZE, moe_ep_rank=_EP_RANK)
    local = _CFG["n_routed_experts"] // _EP_SIZE
    return SimpleNamespace(
        model_config=SimpleNamespace(
            pretrained_config=cfg,
            mapping=mapping,
            torch_dtype=torch.bfloat16,
            spec_config=object() if mtp else None,
        ),
        local_experts=local,
        expert_offset=local * _EP_RANK,
        mtp_enabled=mtp,
    )


@pytest.mark.parametrize("mtp", [False, True], ids=["no_mtp", "mtp"])
def test_every_declared_weight_has_a_source(mtp):
    """A declared weight with no source allocates memory nothing ever fills -- silently
    zero rather than an error, which is what `load`'s coverage assert catches from the
    other side."""
    core = _core(mtp)
    manifest = W.MODEL_WEIGHTS.manifest(core)
    for key, sources in manifest.items():
        assert sources, key


def test_mtp_rows_appear_only_when_mtp_is_live():
    """The MTP module's parameters are allocated only under a speculative config; the
    table has to keep that conditional, or a non-speculative engine carries layer 61's
    weights for nothing."""
    layer = _CFG["num_hidden_layers"]
    off = set(W.MODEL_WEIGHTS.manifest(_core(False)))
    on = set(W.MODEL_WEIGHTS.manifest(_core(True)))
    assert not {k for k in off if k.startswith(f"l{layer}_")}
    assert {k for k in on if k.startswith(f"l{layer}_")}
    assert off < on


def _reference_declaration(mtp: bool) -> dict:
    """The declaration block from the core's __init__, transcribed.

    Deliberately a copy rather than an import: the block is deleted in the switch to
    the table, and the point is to hold the table to what the model used to say. The
    MTP rows carry the table's `l61_` keys; the hand block called them `mtp_`, which
    the module docstring records as the one deliberate renaming.
    """
    import torch.nn as nn

    c = SimpleNamespace(**_CFG)
    heads, nope, rope_dim = c.num_attention_heads, c.qk_nope_head_dim, c.qk_rope_head_dim
    qk_dim, lat_dim = nope + rope_dim, c.kv_lora_rank + rope_dim
    hidden, q_lora, kv_lora, v_dim = c.hidden_size, c.q_lora_rank, c.kv_lora_rank, c.v_head_dim
    sf_vec = 16
    local = c.n_routed_experts // _EP_SIZE

    def P(*shape, dtype=torch.bfloat16):
        return nn.Parameter(torch.empty(*shape, dtype=dtype), requires_grad=False)

    u8, f32 = torch.uint8, torch.float32
    w = {}
    for i in range(c.num_hidden_layers):
        w[f"l{i}_norm1"] = P(hidden)
        w[f"l{i}_qa"] = P(q_lora, hidden)
        w[f"l{i}_q_norm"] = P(q_lora)
        w[f"l{i}_qb"] = P(heads * qk_dim, q_lora)
        w[f"l{i}_kva"] = P(lat_dim, hidden)
        w[f"l{i}_kv_norm"] = P(kv_lora)
        w[f"l{i}_kvb"] = P(heads * (nope + v_dim), kv_lora)
        w[f"l{i}_o"] = P(hidden, heads * v_dim)
        w[f"l{i}_k_scale"] = P(1, dtype=f32)
        w[f"l{i}_v_scale"] = P(1, dtype=f32)
        w[f"l{i}_norm2"] = P(hidden)
        inter = (
            c.intermediate_size
            if i < c.first_k_dense_replace
            else (c.moe_intermediate_size * c.n_shared_experts)
        )
        w[f"l{i}_mlp_gu_w"] = P(2 * inter, hidden // 2, dtype=u8)
        w[f"l{i}_mlp_gu_s"] = P(2 * inter * (hidden // sf_vec), dtype=u8)
        w[f"l{i}_mlp_dn_w"] = P(hidden, inter // 2, dtype=u8)
        w[f"l{i}_mlp_dn_s"] = P(hidden * (inter // sf_vec), dtype=u8)
        for name in ("isc1", "isc1_up", "ws2_1", "ws2_1_up", "isc2", "ws2_2"):
            w[f"l{i}_mlp_{name}"] = P(1, dtype=f32)
        if i < c.first_k_dense_replace:
            continue
        mi = c.moe_intermediate_size
        w[f"l{i}_router"] = P(c.n_routed_experts, hidden)
        w[f"l{i}_router_bias"] = P(c.n_routed_experts, dtype=f32)
        w[f"l{i}_fc1_w"] = P(local, 2 * mi, hidden // 2, dtype=u8)
        w[f"l{i}_fc1_s"] = P(local, 2 * mi, hidden // sf_vec, dtype=u8)
        w[f"l{i}_fc2_w"] = P(local, hidden, mi // 2, dtype=u8)
        w[f"l{i}_fc2_s"] = P(local, hidden, mi // sf_vec, dtype=u8)
        for name in ("isc1", "isc1_up", "ws2_1", "ws2_1_up", "isc2", "ws2_2"):
            w[f"l{i}_e_{name}"] = P(c.n_routed_experts, dtype=f32)
    w["final_norm"] = P(hidden)
    w["embed"] = P(c.vocab_size, hidden)
    if mtp:
        j, mi = c.num_hidden_layers, c.moe_intermediate_size
        shared_inter = c.moe_intermediate_size * c.n_shared_experts
        w[f"l{j}_enorm"] = P(hidden)
        w[f"l{j}_hnorm"] = P(hidden)
        w[f"l{j}_eh"] = P(hidden, 2 * hidden)
        w[f"l{j}_norm1"] = P(hidden)
        w[f"l{j}_qa"] = P(q_lora, hidden)
        w[f"l{j}_q_norm"] = P(q_lora)
        w[f"l{j}_qb"] = P(heads * qk_dim, q_lora)
        w[f"l{j}_kva"] = P(lat_dim, hidden)
        w[f"l{j}_kv_norm"] = P(kv_lora)
        w[f"l{j}_kvb"] = P(heads * (nope + v_dim), kv_lora)
        w[f"l{j}_o"] = P(hidden, heads * v_dim)
        w[f"l{j}_k_scale"] = P(1, dtype=f32)
        w[f"l{j}_v_scale"] = P(1, dtype=f32)
        w[f"l{j}_norm2"] = P(hidden)
        w[f"l{j}_router"] = P(c.n_routed_experts, hidden)
        w[f"l{j}_router_bias"] = P(c.n_routed_experts, dtype=f32)
        w[f"l{j}_fc1"] = P(local, 2 * mi, hidden)
        w[f"l{j}_fc2"] = P(local, hidden, mi)
        w[f"l{j}_sh_gu"] = P(2 * shared_inter, hidden)
        w[f"l{j}_sh_dn"] = P(hidden, shared_inter)
        w[f"l{j}_head_norm"] = P(hidden)
    return w


@pytest.mark.parametrize("mtp", [False, True], ids=["no_mtp", "mtp"])
def test_the_table_declares_what_the_init_block_did(mtp):
    """Shape and dtype per key, so a failure names the weight that drifted."""
    core = _core(mtp)
    got = W.MODEL_WEIGHTS.declare(W.MODEL_WEIGHTS.dims(core))
    want = _reference_declaration(mtp)
    assert set(got.keys()) == set(want.keys())
    for key in sorted(want):
        assert tuple(got[key].shape) == tuple(want[key].shape), key
        assert got[key].dtype == want[key].dtype, key
