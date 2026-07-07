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
"""Static sync test for the fallback ``thop.attention(...)`` call in
``FallbackFmha.forward``.

That call is the single explicit-kwarg call site for the C++ ``thop.attention``
binding. This test parses both the call site (Python AST) and the C++
function declaration in ``attentionOp.h`` (text/regex), and enforces:

1. Every C++ parameter name appears at the call site (and nothing extra).
2. Every call-site kwarg sourced as ``root.attr[.attr...]`` resolves on
   exactly one of ``attn`` / ``metadata`` / ``forward_args``, and its
   declared C++ type matches the source attribute's Python type at a
   coarse-category level (tensor / int / bool / float / list-of-X).
3. Every dataclass field reachable from ``AttentionForwardArgs`` (including
   nested dataclass sub-bags like ``SparsePrediction``) is consumed at
   the call site — directly, transitively via a @property of the
   containing class, or listed in ``_THOP_EXCLUDED_FIELDS``.
4. Every kwarg passed as a literal constant matches an entry in
   ``_THOP_LITERALS`` (both name and value).

The test is AST-only (no kernel run) so it fails fast and runs without a GPU.
"""

import ast
import dataclasses
import inspect
import pathlib
import re
import textwrap
import typing
from dataclasses import fields

from tensorrt_llm._torch.attention_backend.fmha.fallback import (
    _THOP_EXCLUDED_FIELDS,
    _THOP_LITERALS,
    FallbackFmha,
)
from tensorrt_llm._torch.attention_backend.interface import AttentionForwardArgs
from tensorrt_llm._torch.attention_backend.trtllm import TrtllmAttention, TrtllmAttentionMetadata

# Roots used as the LHS of attribute chains at the call site. Match the
# names inside ``FallbackFmha.forward``.
_SOURCE_CLASSES = {
    "attn": TrtllmAttention,
    "metadata": TrtllmAttentionMetadata,
    "forward_args": AttentionForwardArgs,
}

_THOP_KWARG_SOURCE_ALIASES: dict[str, tuple[str, tuple[str, ...]]] = {
    "context_lengths": ("metadata", ("prompt_lens_cuda_runtime",)),
    "head_size": ("attn", ("head_dim",)),
    "host_context_lengths": ("metadata", ("prompt_lens_cpu_runtime",)),
    "host_past_key_value_lengths": ("metadata", ("kv_lens_runtime",)),
    "host_request_types": ("metadata", ("host_request_types_runtime",)),
    "sequence_length": ("metadata", ("kv_lens_cuda_runtime",)),
    "skip_softmax_threshold_scale_factor_decode": (
        "skip_softmax_kernel_params",
        ("threshold_scale_factor_decode",),
    ),
    "skip_softmax_threshold_scale_factor_prefill": (
        "skip_softmax_kernel_params",
        ("threshold_scale_factor_prefill",),
    ),
    "spec_decoding_target_max_draft_tokens": (
        "metadata",
        ("max_total_draft_tokens",),
    ),
    "skip_softmax_threshold_scale_factor_decode": (
        "forward_args",
        (
            "skip_softmax_kernel_params",
            "threshold_scale_factor_decode",
        ),
    ),
    "skip_softmax_threshold_scale_factor_prefill": (
        "forward_args",
        (
            "skip_softmax_kernel_params",
            "threshold_scale_factor_prefill",
        ),
    ),
    "workspace_": ("metadata", ("effective_workspace",)),
}

# The C++ attention() declaration is the single source of truth for kwarg
# names, ordering, and types.
_HEADER = pathlib.Path(__file__).resolve().parents[4] / ("cpp/tensorrt_llm/thop/attentionOp.h")


# ---- C++ declaration parser -------------------------------------------------


_TENSOR_RE = re.compile(r"\btorch::Tensor\b")
_QUANT_MODE_RE = re.compile(r"\bcommon::QuantMode\b")
_INT64_RE = re.compile(r"\bint64_t\b")
_DOUBLE_RE = re.compile(r"\bdouble\b")
_BOOL_RE = re.compile(r"\bbool\b")


def _split_top_level(s: str, sep: str = ",") -> list[str]:
    """Split ``s`` by ``sep`` at angle-bracket depth 0."""
    out: list[str] = []
    depth = 0
    buf: list[str] = []
    for ch in s:
        if ch == "<":
            depth += 1
            buf.append(ch)
        elif ch == ">":
            depth -= 1
            buf.append(ch)
        elif ch == sep and depth == 0:
            out.append("".join(buf).strip())
            buf = []
        else:
            buf.append(ch)
    if buf:
        out.append("".join(buf).strip())
    return out


def _strip_inner_type(cpp_type: str) -> str:
    """Strip ``std::optional<...>`` and ``const`` / references to expose the
    inner element type. Idempotent."""
    t = cpp_type.replace("const", "").replace("&", "").strip()
    m = re.fullmatch(r"std::optional<\s*(.+)\s*>", t)
    if m:
        t = m.group(1).strip()
    return t


def _cpp_category(cpp_type: str) -> str:
    """Coarse Python-side category for a C++ parameter type.

    Returns one of ``tensor`` / ``int`` / ``bool`` / ``float`` /
    ``list_tensor`` / ``list_int`` / ``list_bool`` / ``list_float`` /
    ``unknown``. Optional/const wrappers and references are ignored.
    """
    bare = _strip_inner_type(cpp_type)
    if bare.startswith("std::vector<"):
        inner_match = re.fullmatch(r"std::vector<\s*(.+)\s*>", bare)
        if inner_match:
            inner_cat = _cpp_category(inner_match.group(1))
            return f"list_{inner_cat}"
    if _TENSOR_RE.search(bare):
        return "tensor"
    if _BOOL_RE.fullmatch(bare):
        return "bool"
    if _INT64_RE.fullmatch(bare) or _QUANT_MODE_RE.fullmatch(bare):
        return "int"
    if _DOUBLE_RE.fullmatch(bare):
        return "float"
    return "unknown"


def _parse_attention_decl() -> list[tuple[str, str]]:
    """Parse the ``void attention(...)`` declaration in ``attentionOp.h``
    and return ``[(name, cpp_type), ...]`` in declaration order."""
    src = _HEADER.read_text()
    # Strip line comments to keep the regex tidy.
    src = re.sub(r"//[^\n]*", "", src)
    m = re.search(r"void\s+attention\s*\(", src)
    if not m:
        raise AssertionError(f"Could not find void attention(...) in {_HEADER}")
    # Walk forward, matching the opening paren at m.end()-1 to its close.
    start = m.end() - 1
    depth = 0
    end = None
    for i in range(start, len(src)):
        if src[i] == "(":
            depth += 1
        elif src[i] == ")":
            depth -= 1
            if depth == 0:
                end = i
                break
    if end is None:
        raise AssertionError("Unbalanced parens in attention() declaration")
    body = src[start + 1 : end]

    params: list[tuple[str, str]] = []
    for raw in _split_top_level(body):
        # Drop ``= default`` suffix at top level only.
        param = _split_top_level(raw, sep="=")[0].strip()
        # The name is the trailing identifier. Strip trailing '&' or '*'
        # attached to the type, not the name.
        m = re.match(r"(.*?)([A-Za-z_]\w*)\s*$", param, re.DOTALL)
        if not m:
            raise AssertionError(f"Could not parse param: {param!r}")
        cpp_type = m.group(1).strip()
        name = m.group(2)
        params.append((name, cpp_type))
    return params


def _binding_kwargs() -> set[str]:
    """Set of parameter names declared on ``void attention(...)``."""
    return {name for name, _ in _parse_attention_decl()}


def _binding_types() -> dict[str, str]:
    """Map parameter name → declared C++ type."""
    return dict(_parse_attention_decl())


# ---- Call-site AST helpers --------------------------------------------------


def _parse_thop_attention_call() -> ast.Call:
    """Locate the single ``thop.attention(...)`` call inside
    ``FallbackFmha.forward``."""
    src = textwrap.dedent(inspect.getsource(FallbackFmha.forward))
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "attention"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "thop"
        ):
            return node
    raise AssertionError("Could not find thop.attention(...) call in FallbackFmha.forward")


def _attribute_path(node: ast.AST) -> tuple[str, tuple[str, ...]] | None:
    """If ``node`` is a pure attribute chain ``Name.attr1.attr2...``, return
    ``(root_name_id, (attr1, attr2, ...))``. Otherwise return ``None``.
    """
    if not isinstance(node, ast.Attribute):
        return None
    attrs: list[str] = []
    current: ast.AST = node
    while isinstance(current, ast.Attribute):
        attrs.append(current.attr)
        current = current.value
    if not isinstance(current, ast.Name):
        return None
    return current.id, tuple(reversed(attrs))


def _getattr_path(node: ast.AST) -> tuple[str, tuple[str, ...]] | None:
    """If ``node`` is ``getattr(Name.attr..., "leaf", <default>)``, return
    ``(root_name_id, (attr1, ..., leaf))``. Otherwise return ``None``.
    """
    if (
        not isinstance(node, ast.Call)
        or not isinstance(node.func, ast.Name)
        or node.func.id != "getattr"
        or len(node.args) not in (2, 3)
        or not isinstance(node.args[1], ast.Constant)
        or not isinstance(node.args[1].value, str)
    ):
        return None
    root_path: tuple[str, tuple[str, ...]]
    if isinstance(node.args[0], ast.Name):
        root_path = (node.args[0].id, ())
    else:
        path = _attribute_path(node.args[0])
        if path is None:
            return None
        root_path = path
    root, attrs = root_path
    return root, (*attrs, node.args[1].value)


def _classify_kwargs() -> tuple[
    dict[str, tuple[str, tuple[str, ...]]], dict[str, object], set[str]
]:
    """Split the call site's kwargs into three buckets:

    - ``attr_kwargs``: ``kwarg=source.attr[...]``,
      ``kwarg=int(source.attr)``, or ``kwarg=getattr(source, "attr", ...)``
      → ``{kwarg: (root, path)}``.
    - ``literal_kwargs``: ``kwarg=<constant>`` → ``{kwarg: value}``.
    - ``other_kwargs``: kwargs whose value is anything else (e.g. a bare
      Name like ``q``).
    """
    call = _parse_thop_attention_call()
    attr_kwargs: dict[str, tuple[str, tuple[str, ...]]] = {}
    literal_kwargs: dict[str, object] = {}
    other_kwargs: set[str] = set()
    for kw in call.keywords:
        v = kw.value
        # int(source.attr) → equivalent to source.attr for sync purposes.
        if (
            isinstance(v, ast.Call)
            and isinstance(v.func, ast.Name)
            and v.func.id == "int"
            and len(v.args) == 1
        ):
            path = _attribute_path(v.args[0])
            if path is not None:
                attr_kwargs[kw.arg] = path
                continue
            other_kwargs.add(kw.arg)
            continue
        if isinstance(v, ast.Constant):
            literal_kwargs[kw.arg] = v.value
            continue
        path = _getattr_path(v)
        if path is not None:
            attr_kwargs[kw.arg] = path
            continue
        path = _attribute_path(v)
        if path is not None:
            attr_kwargs[kw.arg] = path
            continue
        other_kwargs.add(kw.arg)
    return attr_kwargs, literal_kwargs, other_kwargs


# ---- Python-side attribute & type resolution --------------------------------


def _runtime_instance_attrs(cls) -> set[str]:
    """Names assigned as ``self.<name> = ...`` anywhere in ``cls`` or any of
    its base classes."""
    cache = _runtime_instance_attrs._cache  # type: ignore[attr-defined]
    if cls in cache:
        return cache[cls]

    def _walk_target(tgt: ast.AST, names: set[str]) -> None:
        if isinstance(tgt, (ast.Tuple, ast.List)):
            for elt in tgt.elts:
                _walk_target(elt, names)
        elif (
            isinstance(tgt, ast.Attribute)
            and isinstance(tgt.value, ast.Name)
            and tgt.value.id == "self"
        ):
            names.add(tgt.attr)

    names: set[str] = set()
    for base in cls.__mro__:
        if base is object:
            continue
        try:
            src = textwrap.dedent(inspect.getsource(base))
        except (OSError, TypeError):
            continue
        for node in ast.walk(ast.parse(src)):
            if isinstance(node, ast.Assign):
                for tgt in node.targets:
                    _walk_target(tgt, names)
            elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
                _walk_target(node.target, names)
    cache[cls] = names
    return names


_runtime_instance_attrs._cache = {}  # type: ignore[attr-defined]


def _has_attr_or_field(cls, name: str) -> bool:
    if hasattr(cls, name):
        return True
    try:
        if name in {f.name for f in fields(cls)}:
            return True
    except TypeError:
        pass
    return name in _runtime_instance_attrs(cls)


def _dataclass_field_type(cls, name: str):
    try:
        f = cls.__dataclass_fields__.get(name)  # type: ignore[attr-defined]
    except AttributeError:
        return None
    if f is None:
        return None
    return f.type if not isinstance(f.type, str) else None


def _resolve_path(root_cls, path: tuple[str, ...]):
    """Walk ``path[:-1]`` from ``root_cls`` and return ``(leaf_cls,
    leaf_attr)``. Returns ``(None, leaf_attr)`` if an intermediate step is
    not a dataclass field."""
    cls = root_cls
    for step in path[:-1]:
        nxt = _dataclass_field_type(cls, step)
        if nxt is None:
            return None, path[-1]
        cls = nxt
    return cls, path[-1]


def _python_category(py_type) -> str:
    """Coarse classification of a Python annotation, mirroring
    ``_cpp_category``. Returns ``unknown`` for anything we can't classify
    confidently (the type check is then skipped for that kwarg)."""
    # Unwrap Optional[X] / Union[X, None].
    origin = typing.get_origin(py_type)
    if origin is typing.Union:
        args = [a for a in typing.get_args(py_type) if a is not type(None)]
        if len(args) == 1:
            return _python_category(args[0])
        return "unknown"
    if origin in (list, typing.List):
        args = typing.get_args(py_type)
        if args:
            return f"list_{_python_category(args[0])}"
        return "list_unknown"
    if py_type is bool:
        return "bool"
    if py_type is int:
        return "int"
    if py_type is float:
        return "float"
    if isinstance(py_type, type):
        if py_type.__name__ == "Tensor":
            return "tensor"
    return "unknown"


# ---- Tests ------------------------------------------------------------------


def test_call_site_kwargs_match_binding_kwargs():
    """The call site's keyword set must equal the C++ binding's keyword set."""
    cpp_kwargs = _binding_kwargs()
    attr_kwargs, literal_kwargs, other_kwargs = _classify_kwargs()
    call_site_kwargs = set(attr_kwargs) | set(literal_kwargs) | other_kwargs
    assert call_site_kwargs == cpp_kwargs, (
        f"missing in call site: {cpp_kwargs - call_site_kwargs}, "
        f"unknown to C++ binding: {call_site_kwargs - cpp_kwargs}"
    )


def test_each_source_attr_kwarg_resolves_uniquely():
    """``root.attr[.attr...]`` kwargs must resolve along the chain to an
    attribute that exists on exactly one of the source classes, and the
    leaf's Python-category must match the C++ kwarg's category (where
    both are unambiguously known)."""
    attr_kwargs, _, _ = _classify_kwargs()
    cpp_types = _binding_types()
    for thop_kwarg, (root, path) in attr_kwargs.items():
        assert root in _SOURCE_CLASSES, (
            f"thop kwarg `{thop_kwarg}` sources from unknown root `{root}` — "
            f"call site must use one of {set(_SOURCE_CLASSES)}."
        )
        leaf_cls, leaf_attr = _resolve_path(_SOURCE_CLASSES[root], path)
        chain = ".".join((root, *path))
        assert leaf_cls is not None, (
            f"thop kwarg `{thop_kwarg}` reads `{chain}` but an intermediate "
            f"step is not a dataclass field of its parent."
        )
        assert _has_attr_or_field(leaf_cls, leaf_attr), (
            f"thop kwarg `{thop_kwarg}` reads `{chain}` but `{leaf_attr}` "
            f"is not exposed on {leaf_cls.__name__}."
        )
        # Coarse type cross-check (skipped when either side is unknown).
        cpp_cat = _cpp_category(cpp_types[thop_kwarg])
        leaf_py_type = _dataclass_field_type(leaf_cls, leaf_attr)
        if cpp_cat != "unknown" and leaf_py_type is not None:
            py_cat = _python_category(leaf_py_type)
            if py_cat != "unknown":
                assert cpp_cat == py_cat, (
                    f"thop kwarg `{thop_kwarg}` (C++ {cpp_types[thop_kwarg]!r}"
                    f" → {cpp_cat}) bound to `{chain}` "
                    f"({leaf_py_type!r} → {py_cat})."
                )


def test_attr_kwarg_names_match_source_leaf_attrs_except_allowlisted_aliases():
    """Most ``thop.attention`` kwargs should bind to a source attribute with
    the same name. Existing aliases must stay explicit so new semantic
    mismatches cannot slip in under a broad type-compatible mapping.
    """
    attr_kwargs, _, _ = _classify_kwargs()
    aliases = {kwarg: source for kwarg, source in attr_kwargs.items() if kwarg != source[1][-1]}
    assert aliases == _THOP_KWARG_SOURCE_ALIASES, (
        "Unexpected thop kwarg/source attribute aliases.\n"
        f"new or changed aliases: {aliases.items() - _THOP_KWARG_SOURCE_ALIASES.items()}\n"
        f"stale allowlist entries: {_THOP_KWARG_SOURCE_ALIASES.items() - aliases.items()}"
    )


def test_literal_kwargs_match_allowlist():
    """Every literal-constant kwarg at the call site must appear in
    ``_THOP_LITERALS`` with the matching value, and every entry in
    ``_THOP_LITERALS`` must be used at the call site."""
    _, literal_kwargs, _ = _classify_kwargs()
    unknown = set(literal_kwargs) - set(_THOP_LITERALS)
    assert not unknown, (
        f"kwargs passed as literals but not in _THOP_LITERALS: "
        f"{sorted(unknown)}. Source from one of {set(_SOURCE_CLASSES)} or "
        f"add to the allowlist."
    )
    stale = set(_THOP_LITERALS) - set(literal_kwargs)
    assert not stale, (
        f"_THOP_LITERALS entries no longer passed as literals: "
        f"{sorted(stale)}. Drop them or restore the call-site literal."
    )
    for kwarg, value in literal_kwargs.items():
        assert value == _THOP_LITERALS[kwarg], (
            f"thop kwarg `{kwarg}` passed as literal {value!r} but "
            f"_THOP_LITERALS expects {_THOP_LITERALS[kwarg]!r}."
        )


def _self_attrs_in_property(prop: property) -> set[str]:
    """``self.<attr>`` names read inside ``prop``'s getter body."""
    src = textwrap.dedent(inspect.getsource(prop.fget))
    return {
        node.attr
        for node in ast.walk(ast.parse(src))
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "self"
    }


def _collect_chains(root: str) -> set[tuple[str, ...]]:
    """All attribute paths in ``FallbackFmha.forward`` that start with
    ``Name(root).``."""
    src = textwrap.dedent(inspect.getsource(FallbackFmha.forward))
    chains: set[tuple[str, ...]] = set()
    for node in ast.walk(ast.parse(src)):
        if not isinstance(node, ast.Attribute):
            continue
        path: list[str] = []
        cur: ast.AST = node
        while isinstance(cur, ast.Attribute):
            path.insert(0, cur.attr)
            cur = cur.value
        if isinstance(cur, ast.Name) and cur.id == root:
            chains.add(tuple(path))
    return chains


def _verify_consumed(cls, chains: set[tuple[str, ...]], excluded=frozenset()):
    """Recursively assert that every field on ``cls`` is consumed by some
    chain in ``chains`` (or excluded). For nested dataclass fields, recurse
    into the sub-bag with chain tails. Properties on ``cls`` accessed at
    the call site transitively consume the fields they read on ``self``.
    """
    direct = {p[0] for p in chains if p}
    transitive: set[str] = set()
    for name in direct:
        obj = vars(cls).get(name)
        if isinstance(obj, property):
            transitive |= _self_attrs_in_property(obj)
    consumed = direct | transitive
    for f in fields(cls):
        if f.name in excluded:
            continue
        ftype = f.type if not isinstance(f.type, str) else None
        if ftype is not None and dataclasses.is_dataclass(ftype):
            sub = {p[1:] for p in chains if len(p) >= 2 and p[0] == f.name}
            assert sub, (
                f"Nested dataclass field `{f.name}` on {cls.__name__} is "
                f"declared but `{f.name}.<subfield>` is never read at the "
                f"call site."
            )
            _verify_consumed(ftype, sub)
        else:
            assert f.name in consumed, (
                f"Field `{f.name}` on {cls.__name__} not consumed by the "
                f"thop call site (directly or via @property). Source it at "
                f"the call site or add it to _THOP_EXCLUDED_FIELDS."
            )


def test_every_forward_args_field_is_consumed():
    """Recursively check that every dataclass field reachable from
    ``AttentionForwardArgs`` (including nested sub-bags such as
    ``SparsePrediction``) is consumed at the call site, transitively
    via @property where applicable, or listed in ``_THOP_EXCLUDED_FIELDS``.
    """
    _verify_consumed(
        AttentionForwardArgs,
        _collect_chains("forward_args"),
        excluded=_THOP_EXCLUDED_FIELDS,
    )


def test_no_unexpected_other_kwargs():
    """The only call-site kwargs that aren't ``source.attr`` chains or
    allowlisted literals are the ``FallbackFmha.forward`` parameters."""
    _, _, other_kwargs = _classify_kwargs()
    expected = {"q", "k", "v"}
    unexpected = other_kwargs - expected
    assert not unexpected, (
        f"thop kwargs with unexpected expression form: {sorted(unexpected)}. "
        f"Allowed: source.attr[.attr...], int(source.attr), literal "
        f"constant, or one of {sorted(expected)}."
    )


def _all_forward_args_field_names() -> set[str]:
    """All dataclass field names reachable from ``AttentionForwardArgs``,
    recursively descending into nested dataclass sub-bags."""
    seen: set[str] = set()

    def _walk(cls) -> None:
        for f in fields(cls):
            seen.add(f.name)
            ftype = f.type if not isinstance(f.type, str) else None
            if ftype is not None and dataclasses.is_dataclass(ftype):
                _walk(ftype)

    _walk(AttentionForwardArgs)
    return seen


def test_excluded_fields_match_real_fields():
    """Every entry in ``_THOP_EXCLUDED_FIELDS`` must name a real field on
    ``AttentionForwardArgs`` (or a nested sub-bag). Dead entries (typos,
    fields removed in a refactor) silently allow newly-added real fields
    to slip past ``test_every_forward_args_field_is_consumed``."""
    stale = set(_THOP_EXCLUDED_FIELDS) - _all_forward_args_field_names()
    assert not stale, (
        f"_THOP_EXCLUDED_FIELDS entries that don't match any "
        f"AttentionForwardArgs field: {sorted(stale)}. Drop them or "
        f"restore the field."
    )


def test_no_sequence_kwargs_at_thop_attention_boundary():
    """``thop.attention`` must not accept ``std::vector<...>`` /
    ``c10::ArrayRef<...>`` / ``std::array<...>`` parameters.

    Sequence params couple list position to semantic meaning, which the
    other sync tests in this file cannot verify element-by-element. Flat
    named params let every slot be checked individually (name, type,
    source). If a new sequence param creeps back in, flatten it the same
    way ``rotary_embedding_scales`` / ``helix_tensor_params`` /
    ``spec_decoding_*_params`` were flattened.
    """
    sequence_params = []
    for name, cpp_type in _parse_attention_decl():
        bare = cpp_type.strip()
        # _strip_inner_type only strips trailing const/&/*; we want to
        # detect outer container types regardless of qualifiers, so look
        # at the raw type string with leading qualifiers stripped.
        outer = re.sub(r"^(const\s+|volatile\s+)+", "", bare)
        outer = re.sub(r"\s*(const|&|\*)\s*$", "", outer)
        if (
            outer.startswith("std::vector<")
            or outer.startswith("c10::ArrayRef<")
            or outer.startswith("std::array<")
        ):
            sequence_params.append((name, cpp_type))
    assert not sequence_params, (
        "thop.attention must not accept Sequence-typed kwargs. Flatten "
        "the following params into their named scalar/tensor components:\n"
        + "\n".join(f"  - {name}: {t}" for name, t in sequence_params)
    )
