# Copyright (c) 2025-2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Static sync test for ``TrtllmAttention._call_thop_attention``.

The wrapper is the single explicit-kwarg call site for the C++
``thop.attention`` binding. This test parses its AST and enforces:

1. Every C++ binding kwarg appears in the wrapper (and nothing extra).
2. Every wrapper kwarg sourced as ``source.attribute`` resolves on exactly
   one of ``self`` / ``metadata`` / ``fwd`` / ``sparse``.
3. Every ``AttentionForwardArgs`` field is either consumed in the wrapper
   (directly or via a @property of ``AttentionForwardArgs``) or listed in
   ``_THOP_EXCLUDED_FIELDS``.
4. Every ``AttentionSparseArgs`` field is consumed in the wrapper.
5. Every kwarg passed as literal ``None`` is allowlisted in
   ``_THOP_LITERAL_NONE``.
"""

import ast
import inspect
import textwrap
from dataclasses import fields

import pytest

from tensorrt_llm._torch.attention_backend.interface import (
    _THOP_EXCLUDED_FIELDS,
    _THOP_LITERAL_NONE,
    AttentionForwardArgs,
    AttentionSparseArgs,
)
from tensorrt_llm._torch.attention_backend.trtllm import TrtllmAttention, TrtllmAttentionMetadata

# Module names used as the LHS of source.attr expressions in the wrapper.
_SOURCE_CLASSES = {
    "self": TrtllmAttention,
    "metadata": TrtllmAttentionMetadata,
    "fwd": AttentionForwardArgs,
    "sparse": AttentionSparseArgs,
}


def _parse_call_thop_attention() -> ast.Call:
    """Locate the single ``thop.attention(...)`` call inside the wrapper."""
    src = textwrap.dedent(inspect.getsource(TrtllmAttention._call_thop_attention))
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "attention"
        ):
            return node
    raise AssertionError("Could not find thop.attention(...) call in _call_thop_attention")


def _classify_kwargs() -> tuple[dict[str, tuple[str, str]], set[str], set[str]]:
    """Split the wrapper's kwargs into three buckets:

    - ``attr_kwargs``: ``kwarg=source.attr`` (or ``kwarg=int(source.attr)``)
      → ``{kwarg: (source_name, attr_name)}``.
    - ``literal_none_kwargs``: ``kwarg=None`` → set of kwarg names.
    - ``other_kwargs``: kwargs whose value is anything else (a local Name,
      a literal ``0``, …) → set of kwarg names.
    """
    call = _parse_call_thop_attention()
    attr_kwargs: dict[str, tuple[str, str]] = {}
    literal_none_kwargs: set[str] = set()
    other_kwargs: set[str] = set()
    for kw in call.keywords:
        v = kw.value
        # int(source.attr) → equivalent to source.attr for sync purposes.
        if (
            isinstance(v, ast.Call)
            and isinstance(v.func, ast.Name)
            and v.func.id == "int"
            and len(v.args) == 1
            and isinstance(v.args[0], ast.Attribute)
            and isinstance(v.args[0].value, ast.Name)
        ):
            inner = v.args[0]
            attr_kwargs[kw.arg] = (inner.value.id, inner.attr)
        elif isinstance(v, ast.Attribute) and isinstance(v.value, ast.Name):
            attr_kwargs[kw.arg] = (v.value.id, v.attr)
        elif isinstance(v, ast.Constant) and v.value is None:
            literal_none_kwargs.add(kw.arg)
        else:
            other_kwargs.add(kw.arg)
    return attr_kwargs, literal_none_kwargs, other_kwargs


def _has_attr_or_field(cls, name: str) -> bool:
    """True if ``cls`` exposes ``name`` as a class attribute (incl. @property
    descriptor) or as a dataclass field."""
    if hasattr(cls, name):
        return True
    try:
        return name in {f.name for f in fields(cls)}
    except TypeError:
        return False


# ---- Tests ------------------------------------------------------------------


def test_wrapper_kwargs_match_binding_kwargs():
    """The wrapper's keyword set must equal the C++ binding's keyword set."""
    pytest.importorskip("tensorrt_llm.bindings.internal")
    from tensorrt_llm.bindings.internal import thop

    cpp_kwargs = set(inspect.signature(thop.attention).parameters)
    attr_kwargs, literal_none_kwargs, other_kwargs = _classify_kwargs()
    wrapper_kwargs = set(attr_kwargs) | literal_none_kwargs | other_kwargs
    assert wrapper_kwargs == cpp_kwargs, (
        f"missing in wrapper: {cpp_kwargs - wrapper_kwargs}, "
        f"unknown to C++ binding: {wrapper_kwargs - cpp_kwargs}"
    )


def test_each_source_attr_kwarg_resolves_uniquely():
    """``source.attr`` kwargs must resolve to ``attr`` on exactly the named
    source class (and only that one)."""
    attr_kwargs, _, _ = _classify_kwargs()
    for thop_kwarg, (src_name, attr) in attr_kwargs.items():
        assert src_name in _SOURCE_CLASSES, (
            f"thop kwarg `{thop_kwarg}` sources from unknown object "
            f"`{src_name}` — wrapper must use one of {set(_SOURCE_CLASSES)}."
        )
        owners = [n for n, cls in _SOURCE_CLASSES.items() if _has_attr_or_field(cls, attr)]
        assert owners == [src_name], (
            f"thop kwarg `{thop_kwarg}` sources from `{src_name}.{attr}`, "
            f"but attribute `{attr}` exists on {owners}. "
            "Rename one side to remove ambiguity."
        )


def test_literal_none_kwargs_are_allowlisted():
    """``kwarg=None`` is only legal if the kwarg name is in
    ``_THOP_LITERAL_NONE``."""
    _, literal_none_kwargs, _ = _classify_kwargs()
    leaked = literal_none_kwargs - _THOP_LITERAL_NONE
    assert not leaked, (
        f"kwargs passed as literal None but not in _THOP_LITERAL_NONE: "
        f"{sorted(leaked)}. Either source from one of "
        f"{set(_SOURCE_CLASSES)} or add to the allowlist."
    )


def test_every_forward_args_field_is_consumed():
    """Every dataclass field on ``AttentionForwardArgs`` is either accessed
    in the wrapper (directly or via a @property on the class), or listed in
    ``_THOP_EXCLUDED_FIELDS``."""
    src = textwrap.dedent(inspect.getsource(TrtllmAttention._call_thop_attention))
    fwd_accesses_in_wrapper = {
        node.attr
        for node in ast.walk(ast.parse(src))
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "fwd"
    }
    # Properties / class attrs on AttentionForwardArgs are also valid consumers.
    fwd_properties = {
        name for name, obj in vars(AttentionForwardArgs).items() if isinstance(obj, property)
    }
    consumed = fwd_accesses_in_wrapper | fwd_properties
    field_names = {f.name for f in fields(AttentionForwardArgs)}
    unused = field_names - consumed - _THOP_EXCLUDED_FIELDS
    assert not unused, (
        f"AttentionForwardArgs fields not consumed by the thop wrapper: "
        f"{sorted(unused)}. Either source them in the wrapper, expose a "
        f"@property that does, or add them to _THOP_EXCLUDED_FIELDS."
    )


def test_every_sparse_args_field_is_consumed():
    """Every dataclass field on ``AttentionSparseArgs`` must be sourced in
    the wrapper (no exclusion list — this is what the type exists for)."""
    attr_kwargs, _, _ = _classify_kwargs()
    sparse_attrs_used = {attr for src, attr in attr_kwargs.values() if src == "sparse"}
    field_names = {f.name for f in fields(AttentionSparseArgs)}
    unused = field_names - sparse_attrs_used
    assert not unused, (
        f"AttentionSparseArgs fields not consumed by the thop wrapper: "
        f"{sorted(unused)}. Drop them from the dataclass."
    )


def test_no_unexpected_other_kwargs():
    """Catch wrapper kwargs whose value is neither ``source.attr``, literal
    ``None``, nor a clearly-named local. Local-Name kwargs are allowed but
    we still want the set to match expectations."""
    _, _, other_kwargs = _classify_kwargs()
    # Wrapper-local names — function parameters and computed locals whose
    # ``Name`` form the AST treats the same as a direct identifier reference.
    expected_locals = {
        # Function parameters
        "q",
        "k",
        "v",
        # Locals computed from rich objects at the top of the wrapper
        "is_fused_qkv",
        "update_kv_cache",
        "layer_idx",
        "attention_window_size",
        "spec_decoding_tensor_params",
        "kv_scale_orig_quant",
        "kv_scale_quant_orig",
        # Literal 0 (sink_token_length=0)
        "sink_token_length",
    }
    unexpected = other_kwargs - expected_locals
    assert not unexpected, (
        f"thop kwargs with unexpected expression form: {sorted(unexpected)}. "
        f"Allowed: source.attr, int(source.attr), literal None, or one of "
        f"the known wrapper locals {sorted(expected_locals)}."
    )
