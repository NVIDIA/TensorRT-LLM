# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The draft samplers' contract between SpecWorkerBase and its overrides.

``sample_draft_tokens`` hands ``tokens_out`` to ``greedy_sample_draft_with_tp_gather`` and the
draft loops hand ``out`` to ``sample_draft_tokens``; an override with an older parameter list
raises TypeError at the first draft step. The signature check reads the source of every
override in ``tensorrt_llm/`` (modeling_v2 targets included, whose GPU-only modules it does not
import).
"""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from tensorrt_llm._torch.speculative.interface import SpecWorkerBase

pytestmark = pytest.mark.cpu_only

_PKG = Path(__file__).resolve().parents[5] / "tensorrt_llm"
_BASE = _PKG / "_torch/speculative/interface.py"
_METHODS = ("greedy_sample_draft_with_tp_gather", "sample_draft_tokens")


def _params(fn: ast.FunctionDef) -> list[tuple[str, str]]:
    a = fn.args
    params = [(p.arg, "positional") for p in a.posonlyargs + a.args]
    params += [(p.arg, "keyword") for p in a.kwonlyargs]
    if a.vararg is not None:
        params.append((a.vararg.arg, "*"))
    if a.kwarg is not None:
        params.append((a.kwarg.arg, "**"))
    return params


def _definitions(path: Path) -> list[tuple[str, str, ast.FunctionDef]]:
    """(class, method, def) for every draft-sampler method defined in a class of ``path``."""
    found = []
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.ClassDef):
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name in _METHODS:
                    found.append((node.name, item.name, item))
    return found


def _defaulted(fn: ast.FunctionDef) -> set[str]:
    """Parameters of ``fn`` that callers may leave out (a default, or ``*args`` / ``**kwargs``)."""
    a = fn.args
    positional = a.posonlyargs + a.args
    names = {p.arg for p in positional[len(positional) - len(a.defaults) :]}
    names |= {p.arg for p, d in zip(a.kwonlyargs, a.kw_defaults) if d is not None}
    names |= {p.arg for p in (a.vararg, a.kwarg) if p is not None}
    return names


def _takes_the_base_call(fn: ast.FunctionDef, base: list[tuple[str, str]]) -> bool:
    """Every base parameter, by name and kind, with the base positional ones first and in the base order; any
    extra parameter is defaulted. Then every call the base takes reaches the override unchanged."""
    params = _params(fn)
    positional = [name for name, kind in params if kind == "positional"]
    base_positional = [name for name, kind in base if kind == "positional"]
    extra = {name for name, _ in params} - {name for name, _ in base}
    return (
        set(base) <= set(params)
        and positional[: len(base_positional)] == base_positional
        and extra <= _defaulted(fn)
    )


def test_every_draft_sampler_override_takes_the_base_parameters() -> None:
    base = {
        method: _params(fn) for cls, method, fn in _definitions(_BASE) if cls == "SpecWorkerBase"
    }
    assert set(base) == set(_METHODS)
    overrides = []
    for path in sorted(_PKG.rglob("*.py")):
        for cls, method, fn in _definitions(path):
            if path == _BASE and cls == "SpecWorkerBase":
                continue
            overrides.append((path.relative_to(_PKG).as_posix(), cls))
            assert _takes_the_base_call(fn, base[method]), (path, cls, method)
    # Guard against a scan that finds nothing.
    assert overrides


_BASE_SIGNATURE = "(self, logits, meta=None, tokens_out=None)"


@pytest.mark.parametrize(
    "signature,compatible",
    [
        (_BASE_SIGNATURE, True),
        ("(self, logits, meta=None, tokens_out=None, lane=0)", True),  # extra defaulted parameter
        ("(self, logits, meta=None, tokens_out=None, **kw)", True),
        ("(self, logits, meta=None)", False),  # no tokens_out
        ("(self, logits, meta=None, tokens_out=None, *, lane)", False),  # extra required parameter
        ("(self, meta, logits, tokens_out=None)", False),  # reordered
    ],
)
def test_the_signature_check_accepts_compatible_extensions(signature, compatible) -> None:
    """An override may add defaulted parameters (or ``**kwargs``) to the base's, not drop or reorder them."""

    def fn(sig: str) -> ast.FunctionDef:
        return ast.parse(f"def f{sig}: pass").body[0]

    assert _takes_the_base_call(fn(signature), _params(fn(_BASE_SIGNATURE))) is compatible


def _worker(cls: type) -> SpecWorkerBase:
    w = cls.__new__(cls)
    nn.Module.__init__(w)  # attribute storage only; the worker's own init needs a model config
    w.mapping = SimpleNamespace(tp_size=1, tp_rank=0, enable_attention_dp=False)
    w._d2t = None
    return w


class _Worker(SpecWorkerBase):
    """The base samplers on a concrete worker."""

    def _forward_impl(self, *args, **kwargs):
        raise NotImplementedError

    @property
    def max_draft_len(self) -> int:
        return 1


class _OwnTensorWorker(_Worker):
    """An override that ignores ``tokens_out`` and returns a tensor of its own."""

    def greedy_sample_draft_with_tp_gather(
        self, logits, spec_metadata=None, mapping_lm_head_tp=None, tokens_out=None
    ):
        return logits.argmax(dim=-1).to(torch.int32)


class _ThreeArgWorker(_Worker):
    """An override without the ``tokens_out`` parameter (out-of-tree subclasses)."""

    def greedy_sample_draft_with_tp_gather(
        self, logits, spec_metadata=None, mapping_lm_head_tp=None
    ):
        return logits.argmax(dim=-1).to(torch.int32)


_META = SimpleNamespace(wants_advanced_draft_sampling=False, is_cuda_graph=False)


@pytest.mark.parametrize("cls", [_Worker, _OwnTensorWorker, _ThreeArgWorker])
def test_the_greedy_step_fills_out_on_every_override(cls) -> None:
    w = _worker(cls)
    logits = torch.randn(5, 64)
    want = logits.argmax(dim=-1).to(torch.int32)
    assert torch.equal(w.sample_draft_tokens(logits, _META, 5, draft_step=0), want)
    if cls is _ThreeArgWorker:
        return  # takes no tokens_out: only calls without ``out`` reach it unchanged
    out = torch.full((5,), -1, dtype=torch.int32)
    tokens = w.sample_draft_tokens(logits, _META, 5, draft_step=0, out=out)
    assert tokens.data_ptr() == out.data_ptr()  # the tokens are out's storage
    assert torch.equal(out, want)
