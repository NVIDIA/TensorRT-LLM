# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import operator
import types

import torch
from torch import nn
from torch.fx import symbolic_trace

from tensorrt_llm._torch.auto_deploy.models.hf import (
    TextModelExportInfo,
    expose_graph_module_accessor,
)


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


def test_expose_graph_module_accessor_tolerates_duplicate_exposure():
    """Re-exposing the same accessor is harmless aside from a redundant sentinel.

    For a VLM target, ``TargetModelExportInfo.post_process`` exposes
    ``get_input_embeddings`` twice: once via the inner text export-info and once
    directly. The helper does not guard against this; the resulting duplicate
    keepalive sentinel is intentional and harmless. This test locks in that the
    graph stays valid, the accessor still resolves to the real embedding, and the
    module still runs.
    """
    model = _DummyTextModel()
    gm = symbolic_trace(model)

    expose_graph_module_accessor(model, gm, "get_input_embeddings", "missing embedding")
    expose_graph_module_accessor(model, gm, "get_input_embeddings", "missing embedding")
    gm.recompile()
    gm.graph.lint()

    assert gm.get_input_embeddings() is model.get_input_embeddings()
    out = gm(torch.randint(0, 32, (2, 4)))
    assert out.shape == (2, 4, 8)


def test_expose_graph_module_accessor_binds_resolved_submodule_path():
    """Unit test the helper path binding without going through export-info wrappers."""
    backbone = _DummyTextModel()
    sub_mod = nn.Module()
    sub_mod.add_module("backbone", backbone)
    sub_mod.get_input_embeddings = types.MethodType(
        lambda self: self.backbone.get_input_embeddings(), sub_mod
    )

    # Build a GraphModule the way export does: a plain ``nn.Module`` root with a
    # plain intermediate container (not the original ``_DummyTextModel`` class),
    # so the delegating accessor cannot be re-run on it.
    graph = torch.fx.Graph()
    placeholder = graph.placeholder("input_ids")
    graph.output(placeholder)
    sub_gm = torch.fx.GraphModule(nn.Module(), graph)
    sub_gm.add_module("backbone", nn.Module())

    expose_graph_module_accessor(sub_mod, sub_gm, "get_input_embeddings", "missing embedding")
    sub_gm.recompile()
    sub_gm.graph.lint()

    exposed = sub_gm.get_input_embeddings()
    assert exposed is sub_gm.get_submodule("backbone.embed_tokens")
    assert exposed is sub_mod.get_input_embeddings()
    assert isinstance(exposed, nn.Embedding)
