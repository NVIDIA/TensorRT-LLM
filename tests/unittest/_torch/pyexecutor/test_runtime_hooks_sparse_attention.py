# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``RuntimeHooks.sparse_attention``: a model's own subclasses of the sparse attention registry's classes.

Off by default; when set, the engine builds the step metadata from the model's backend class, the KV cache from its
KV cache manager class, and runs its warmup scope and indexer prewarm. A class that does not derive from the
registry's is refused.
"""

import contextlib
from types import SimpleNamespace

import pytest

from tensorrt_llm._torch.pyexecutor.engine import model_setup
from tensorrt_llm._torch.pyexecutor.engine.runners.decoder.warmup import DecoderWarmup
from tensorrt_llm._torch.pyexecutor.runtime_hooks import (
    RuntimeHooks,
    SparseAttentionClasses,
    override_sparse_attention_class,
)

pytestmark = pytest.mark.cpu_only


class _Stock:
    pass


class _Sub(_Stock):
    pass


class _Unrelated:
    pass


def test_override_keeps_the_stock_class_unless_a_subclass_is_named():
    assert override_sparse_attention_class(_Stock, None, "backend") is _Stock
    assert override_sparse_attention_class(_Stock, _Stock, "backend") is _Stock
    assert override_sparse_attention_class(_Stock, _Sub, "backend") is _Sub
    with pytest.raises(TypeError, match="backend must derive from _Stock"):
        override_sparse_attention_class(_Stock, _Unrelated, "backend")


def _model(hooks, sparse_config):
    pretrained = object()
    return SimpleNamespace(
        runtime_hooks=hooks,
        model_config=SimpleNamespace(
            sparse_attention_config=sparse_config, pretrained_config=pretrained
        ),
    )


@pytest.mark.parametrize("sparse", [False, True])
@pytest.mark.parametrize("declared", [False, True])
def test_resolve_attention_backend_takes_the_model_backend_only_for_sparse_attention(
    monkeypatch, sparse, declared
):
    monkeypatch.setattr(model_setup, "get_attention_backend", lambda name, sparse_params: _Stock)
    sparse_config = (
        SimpleNamespace(to_sparse_params=lambda pretrained_config: "params") if sparse else None
    )
    hooks = (
        RuntimeHooks(sparse_attention=SparseAttentionClasses(backend=_Sub)) if declared else None
    )
    model = _model(hooks, sparse_config)
    got = model_setup.resolve_attention_backend(model, SimpleNamespace(attn_backend="TRTLLM"))
    assert got is (_Sub if sparse and declared else _Stock)


def _warmup(hooks):
    warmup = DecoderWarmup.__new__(DecoderWarmup)
    warmup.model = SimpleNamespace(runtime_hooks=hooks)
    return warmup


def test_the_model_indexer_prewarm_replaces_the_stock_one():
    calls = []
    hooks = RuntimeHooks(
        sparse_attention=SparseAttentionClasses(
            indexer_prewarm=lambda runner, kv_cache_manager: calls.append(
                (runner, kv_cache_manager)
            )
        )
    )
    warmup = _warmup(hooks)
    warmup._prewarm_minimax_m3_msa_indexer("kv")
    assert calls == [(warmup, "kv")]


def test_the_model_warmup_scope_wraps_the_timed_warmup(monkeypatch):
    events = []

    @contextlib.contextmanager
    def scope(runner):
        events.append("enter")
        yield
        events.append("exit")

    hooks = RuntimeHooks(sparse_attention=SparseAttentionClasses(warmup_scope=scope))
    warmup = _warmup(hooks)

    @contextlib.contextmanager
    def timer():
        events.append("timer")
        yield

    warmup._warmup_timer = timer()
    monkeypatch.setattr(
        "tensorrt_llm._torch.pyexecutor.engine.runners.decoder.warmup.warmup_named_allreduce_topologies",
        lambda model: None,
    )
    warmup._warmup_scheduled = lambda resource_manager, kv_cache_manager: events.append("run")
    resource_manager = SimpleNamespace(get_resource_manager=lambda key: "kv")
    warmup.run(resource_manager)
    assert events == ["enter", "timer", "run", "exit"]
