# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the memory-tag weight allocation path in ``ModelLoader``.

The memory-tag pool slot starts as ``empty_like`` garbage; when meta-init
fails (the non-meta-init fallback) every non-meta constructor tensor must be
copied into the pool, or ctor values that nothing re-fills -- calibration
scalars, the DSv4 ``attn_sink`` ``-inf`` -- are silently destroyed under
AUTO/DUMMY load formats.
"""

from contextlib import contextmanager, nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from torch import nn

from tensorrt_llm._torch.pyexecutor import model_loader as model_loader_mod
from tensorrt_llm._torch.pyexecutor.model_loader import ModelLoader
from tensorrt_llm.llmapi.llm_args import LoadFormat


class _CtorValueModel(nn.Module):
    """Tiny model carrying ctor values only a copy-through allocation keeps."""

    def __init__(self):
        super().__init__()
        inner = nn.Module()
        inner.attn_sink = nn.Parameter(
            torch.full((4,), float("-inf"), dtype=torch.float32), requires_grad=False
        )
        inner.input_scale = nn.Parameter(torch.ones(1), requires_grad=False)
        inner.weight = nn.Parameter(torch.zeros(4, 4), requires_grad=False)
        self.self_attn = inner


@contextmanager
def _fake_memory_scope(_tag, _restore_mode):
    yield MagicMock(name="weight_pool_proxy")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="allocation targets CUDA")
def test_memory_tag_alloc_preserves_ctor_values_on_non_meta_init(monkeypatch):
    """Non-meta-init fallback + memory tag + DUMMY: the pool copy must be
    unconditional for non-meta sources. Skipping it for AUTO/DUMMY leaves
    ``empty_like`` garbage in params the dummy initializer deliberately
    skips (``.attn_sink``, ``.input_scale``) and nothing later re-fills."""
    model = _CtorValueModel()

    llm_args = SimpleNamespace(load_format=LoadFormat.DUMMY)
    loader = ModelLoader(
        llm_args=llm_args,
        mapping=MagicMock(name="mapping"),
        spec_config=None,
        sparse_attention_config=None,
        max_num_tokens=8,
        max_seq_len=8,
        model_weights_memory_tag=model_loader_mod.ExecutorMemoryType.MODEL_WEIGHTS_MAIN,
    )
    loader._load_and_validate_config = MagicMock(
        return_value=SimpleNamespace(name="config", mapping=SimpleNamespace())
    )

    monkeypatch.setattr(model_loader_mod, "timing", lambda *_a, **_k: nullcontext())
    monkeypatch.setattr(
        model_loader_mod,
        "maybe_create_moe_load_balancer",
        lambda *_a, **_k: nullcontext(),
    )
    monkeypatch.setattr(model_loader_mod, "virtual_memory_scope", _fake_memory_scope)
    # Force the non-meta-init fallback: the first (MetaInitMode) construction
    # raises, the second returns the real CPU-constructed model.
    monkeypatch.setattr(
        model_loader_mod.AutoModelForCausalLM,
        "from_config",
        MagicMock(side_effect=[RuntimeError("no meta init"), model]),
    )

    checkpoint_loader = MagicMock(name="checkpoint_loader")
    checkpoint_loader.checkpoint_format = "HF"

    loaded_model, _ = loader.load("/ckpt", checkpoint_loader)
    assert loaded_model is model

    sink = model.self_attn.attn_sink
    scale = model.self_attn.input_scale
    weight = model.self_attn.weight
    assert sink.data.is_cuda and scale.data.is_cuda and weight.data.is_cuda
    # Ctor values survived the pool allocation (skipped by dummy init).
    assert torch.isneginf(sink.data).all()
    assert torch.equal(scale.data.cpu(), torch.ones(1))
    # Dummy init still randomized ordinary weights inside the pool.
    assert not torch.equal(weight.data.cpu(), torch.zeros(4, 4))
