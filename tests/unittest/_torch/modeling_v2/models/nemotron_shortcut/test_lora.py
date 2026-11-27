# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.nemotron_shortcut.lora import (
    HF_PROJECTIONS,
    ShortcutLoraOwner,
)

pytestmark = pytest.mark.cpu_only


def test_indexer_adapter_cannot_alias_attention_query():
    from tensorrt_llm._torch.peft.lora.loaders import get_all_hf_lora_weights

    owner = ShortcutLoraOwner(module_path="model", head_size=8, attention_tp=4)
    key = "base_model.model.model.language_model.layers.5.self_attn.indexer.q_proj.lora_A.weight"
    with pytest.raises(ValueError, match="Unsupported decoder LoRA projection"):
        get_all_hf_lora_weights(
            {key: torch.empty(8, 16)}, set(HF_PROJECTIONS.values()), owner=owner
        )


def test_kv_adapter_rows_follow_attention_tp():
    from tensorrt_llm._torch.peft.lora.loaders import get_all_hf_lora_weights

    dim, rank, tp = 8, 2, 4
    owner = ShortcutLoraOwner(module_path="model", head_size=dim, attention_tp=tp)
    proj = "self_attn.k_proj"
    prefix = f"base_model.model.model.layers.0.{proj}"
    a = torch.arange(rank * 16).view(rank, 16)
    b = torch.arange(dim * rank).view(dim, rank)
    rows = get_all_hf_lora_weights(
        {f"{prefix}.lora_A.weight": a, f"{prefix}.lora_B.weight": b}, {proj}, owner=owner
    )[0][proj]
    assert rows["in"] is a
    for shard in rows["out"].chunk(tp):
        torch.testing.assert_close(shard, b)
    assert owner.prepare_hf_weight(proj, "out", rows["out"]) is rows["out"]
