# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A model-declared KV cache manager class (``RuntimeHooks.kv_cache_manager_cls``).

The engine resolves and validates the manager class for the model's KV cache as before; a
model that declares a subclass of that class gets the subclass, a model that declares any
other class is rejected, and a model that declares nothing keeps the resolved class.
"""

from types import SimpleNamespace

import pytest

from tensorrt_llm._torch.pyexecutor import _util
from tensorrt_llm._torch.pyexecutor.runtime_hooks import RuntimeHooks

pytestmark = pytest.mark.cpu_only


class _Resolved:
    """Stands in for the class the engine resolves."""


class _Subclass(_Resolved):
    pass


class _Unrelated:
    pass


class _Declares:
    def __init__(self, cls):
        self.runtime_hooks = RuntimeHooks(kv_cache_manager_cls=cls)


class _DeclaresNothing:
    pass


@pytest.fixture
def info(monkeypatch):
    lines = []
    monkeypatch.setattr(_util.logger, "info_once", lambda msg, key: lines.append(msg))
    return lines


def test_the_default_is_none():
    assert RuntimeHooks().kv_cache_manager_cls is None


@pytest.mark.parametrize("model", [_DeclaresNothing(), _Declares(None)], ids=["absent", "none"])
def test_without_a_declaration_the_resolved_class_is_kept(model, info):
    assert _util.model_declared_kv_cache_manager_cls(model, _Resolved) is _Resolved
    assert info == []


def test_a_declared_subclass_is_used(info):
    assert _util.model_declared_kv_cache_manager_cls(_Declares(_Subclass), _Resolved) is _Subclass
    assert info == ["Using the model-declared KV cache manager _Subclass (a subclass of _Resolved)"]


def test_the_resolved_class_itself_may_be_declared(info):
    assert _util.model_declared_kv_cache_manager_cls(_Declares(_Resolved), _Resolved) is _Resolved


@pytest.mark.parametrize(
    "declared, resolved",
    [(_Unrelated, _Resolved), (_Resolved, _Subclass), ("_Subclass", _Resolved)],
    ids=["unrelated", "base_class", "not_a_class"],
)
def test_a_class_that_is_not_a_subclass_is_rejected(declared, resolved, info):
    with pytest.raises(ValueError) as raised:
        _util.model_declared_kv_cache_manager_cls(_Declares(declared), resolved)
    message = str(raised.value)
    assert f"not a subclass of {resolved.__qualname__}" in message
    assert (declared if isinstance(declared, str) else declared.__qualname__) in message
    assert info == []


def _creator(model, monkeypatch, resolved=_Resolved):
    """A KvCacheCreator with just what ``_get_model_kv_cache_manager_cls`` reads."""
    monkeypatch.setattr(_util, "get_kv_cache_manager_cls", lambda *args, **kwargs: resolved)
    monkeypatch.setattr(
        _util.KvCacheCreator,
        "_validate_or_fallback_kv_cache_manager_v2",
        lambda self, cls, model_config, kv_cache_config: cls,
    )
    monkeypatch.setattr(_util, "is_hybrid_linear", lambda config: False)
    model.model_config = SimpleNamespace(pretrained_config=object())
    creator = object.__new__(_util.KvCacheCreator)
    creator._model_engine = SimpleNamespace(model=model)
    creator._kv_cache_config = SimpleNamespace(enable_block_reuse=False)
    creator._is_disagg = False
    creator._cache_transceiver_config = None
    creator._speculative_config = None
    return creator


@pytest.mark.parametrize(
    "model, expected",
    [
        (_DeclaresNothing(), _Resolved),
        (_Declares(None), _Resolved),
        (_Declares(_Subclass), _Subclass),
    ],
    ids=["absent", "none", "subclass"],
)
def test_the_kv_cache_creator_builds_the_declared_subclass(model, expected, monkeypatch, info):
    assert _creator(model, monkeypatch)._get_model_kv_cache_manager_cls() is expected


def test_the_kv_cache_creator_rejects_a_class_that_is_not_a_subclass(monkeypatch, info):
    creator = _creator(_Declares(_Unrelated), monkeypatch)
    with pytest.raises(ValueError, match="_Unrelated .*not a subclass of _Resolved"):
        creator._get_model_kv_cache_manager_cls()
