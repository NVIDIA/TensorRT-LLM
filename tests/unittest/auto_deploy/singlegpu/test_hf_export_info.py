# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import operator

import torch
from torch import nn
from torch.fx import symbolic_trace

from tensorrt_llm._torch.auto_deploy.models.hf import TextModelExportInfo


class _DummyTextModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(32, 8)

    def get_input_embeddings(self):
        return self.embed_tokens

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)


def test_text_model_export_info_uses_scalar_embedding_keepalive_assert():
    model = _DummyTextModel()
    gm = symbolic_trace(model)

    export_info = TextModelExportInfo("dummy")
    export_info.post_process(model, gm)
    gm.recompile()
    gm.graph.lint()

    assert_node = next(
        node
        for node in gm.graph.nodes
        if node.op == "call_function" and node.target == torch._assert
    )
    cond_node = assert_node.args[0]

    assert cond_node.op == "call_function"
    assert cond_node.target == operator.ge
    assert cond_node.args[0].op == "call_function"
    assert cond_node.args[0].target == torch.ops.aten.sym_size.int
