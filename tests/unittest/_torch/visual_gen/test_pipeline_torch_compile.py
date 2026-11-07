# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for BasePipeline per-block torch.compile and NCCL hook discovery."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

import tensorrt_llm._torch.visual_gen.pipeline as pipeline_module
from tensorrt_llm._torch.visual_gen.pipeline import BasePipeline

pytestmark = pytest.mark.cpu_only


def _blocks(n):
    return nn.ModuleList(nn.Linear(4, 4) for _ in range(n))


class _FlatModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = _blocks(3)
        self.single = _blocks(1)
        self.proj = nn.Linear(4, 4)


class _NestedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.text_blocks = _blocks(2)
        self.mode_blocks = nn.ModuleDict(
            {"image": _blocks(2), "image_cond": _blocks(3), "tiny": _blocks(1)}
        )
        self.heads = nn.ModuleDict({"out": nn.Linear(4, 4)})


class _OptionsPipeline(BasePipeline):
    def torch_compile_options(self):
        return {"emulate_precision_casts": True}


def _pipeline(model, cls=BasePipeline):
    pipeline = object.__new__(cls)
    nn.Module.__init__(pipeline)
    pipeline.transformer = model
    pipeline.pipeline_config = SimpleNamespace(
        torch_compile=SimpleNamespace(enable_fullgraph=False)
    )
    return pipeline


def _nested_blocks(model):
    return [
        *model.text_blocks,
        *model.mode_blocks["image"],
        *model.mode_blocks["image_cond"],
    ]


class _FakeCompiled(nn.Module):
    def __init__(self, module, kwargs):
        super().__init__()
        self.module = module
        self.kwargs = kwargs


@pytest.fixture
def fake_compile(monkeypatch):
    def compile_(module, **kwargs):
        return _FakeCompiled(module, kwargs)

    monkeypatch.setattr(torch, "compile", compile_)


def test_flat_models_keep_attribute_names():
    assert BasePipeline._find_transformer_blocks(_FlatModel()) == ["blocks"]


def test_block_lists_inside_module_dict_are_found():
    assert BasePipeline._find_transformer_blocks(_NestedModel()) == [
        "text_blocks",
        "mode_blocks.image",
        "mode_blocks.image_cond",
    ]


@pytest.mark.parametrize("model_cls", [_FlatModel, _NestedModel])
def test_every_block_is_compiled_in_default_mode(fake_compile, model_cls):
    model = model_cls()
    paths = BasePipeline._find_transformer_blocks(model)
    originals = [list(model.get_submodule(path)) for path in paths]

    _pipeline(model).torch_compile()

    assert BasePipeline._find_transformer_blocks(model) == paths
    for path, before in zip(paths, originals):
        after = model.get_submodule(path)
        assert isinstance(after, nn.ModuleList)
        assert [compiled.module for compiled in after] == before
        for compiled in after:
            assert compiled.kwargs == {"dynamic": None, "fullgraph": False, "mode": "default"}


def test_options_replace_mode(fake_compile):
    model = _NestedModel()
    _pipeline(model, _OptionsPipeline).torch_compile()

    for compiled in _nested_blocks(model):
        assert compiled.kwargs["options"] == {"emulate_precision_casts": True}
        assert "mode" not in compiled.kwargs


@pytest.mark.parametrize("cls", [BasePipeline, _OptionsPipeline])
def test_real_torch_compile_accepts_arguments(cls):
    model = _NestedModel()
    _pipeline(model, cls).torch_compile()

    assert all(
        isinstance(block, torch._dynamo.eval_frame.OptimizedModule)
        for block in _nested_blocks(model)
    )


def test_nccl_hooks_cover_nested_blocks(monkeypatch):
    captured = []

    def install(blocks):
        captured.extend(blocks)
        return []

    monkeypatch.setattr(pipeline_module, "install_module_nccl_window_tensor_scopes", install)
    model = _NestedModel()
    pipeline = _pipeline(model)
    pipeline._uses_nccl_window_tensors = lambda: True

    pipeline._install_nccl_window_tensor_scope_hooks()

    assert captured == _nested_blocks(model)
