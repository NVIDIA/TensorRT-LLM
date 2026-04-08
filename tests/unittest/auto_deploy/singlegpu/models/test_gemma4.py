# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch

import tensorrt_llm._torch.auto_deploy.custom_ops  # noqa: F401
from tensorrt_llm._torch.auto_deploy.export import torch_export_to_gm
from tensorrt_llm._torch.auto_deploy.models.custom.modeling_gemma4 import (
    Gemma4ForCausalLM,
    Gemma4TextConfig,
)
from tensorrt_llm._torch.auto_deploy.utils.node_utils import is_op


@torch.inference_mode()
def test_gemma4_text_export_uses_semantic_multimodal_mask():
    config = Gemma4TextConfig(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        global_head_dim=8,
        num_global_key_value_heads=1,
        sliding_window=4,
        layer_types=["sliding_attention"],
        enable_moe_block=False,
        final_logit_softcapping=None,
    )
    model = Gemma4ForCausalLM(config).eval()

    input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.int64)
    position_ids = torch.arange(input_ids.shape[1], dtype=torch.int64).unsqueeze(0)
    mm_item_cu_seqlen = torch.tensor([0, 1], dtype=torch.int32)
    mm_item_types = torch.tensor([0], dtype=torch.int32)
    mm_token_positions = torch.tensor([1], dtype=torch.int32)
    mm_token_lengths = torch.tensor([2], dtype=torch.int32)
    mm_special_offsets_cu_seqlen = torch.tensor([0, 2], dtype=torch.int32)
    mm_special_offsets = torch.tensor([0, 1], dtype=torch.int32)

    gm = torch_export_to_gm(
        model,
        args=(),
        kwargs={
            "input_ids": input_ids,
            "position_ids": position_ids,
            "mm_item_cu_seqlen": mm_item_cu_seqlen,
            "mm_item_types": mm_item_types,
            "mm_token_positions": mm_token_positions,
            "mm_token_lengths": mm_token_lengths,
            "mm_special_offsets_cu_seqlen": mm_special_offsets_cu_seqlen,
            "mm_special_offsets": mm_special_offsets,
        },
        clone=True,
    )

    semantic_nodes = [
        node for node in gm.graph.nodes if is_op(node, torch.ops.auto_deploy.gemma4_multimodal_mask)
    ]
    attention_nodes = [
        node for node in gm.graph.nodes if is_op(node, torch.ops.auto_deploy.torch_attention)
    ]

    assert len(semantic_nodes) == 1
    assert len(attention_nodes) == 1

    attention_mask_arg = attention_nodes[0].kwargs.get("attn_mask")
    if attention_mask_arg is None and len(attention_nodes[0].args) > 3:
        attention_mask_arg = attention_nodes[0].args[3]
    assert attention_mask_arg is semantic_nodes[0]
