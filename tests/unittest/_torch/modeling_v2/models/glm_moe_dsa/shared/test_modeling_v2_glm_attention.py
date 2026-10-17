# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The GLM-5.2 tp8 target's attention modules: their adoption, the FMHA libraries the attention
backends declare, the phase gate of every override, the indexer's top-k region and bridge, and
the target's DSA op 2.

Host-only: the built-in modules are created without their constructors and carry only the
attributes adoption and the overrides read. The kernel factories are stand-ins, and the built-in
methods and the decode functions an override calls are recorders.
"""

from __future__ import annotations

import ast
import inspect
import types
from pathlib import Path

import pytest
import torch
from torch import nn

from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared import (  # noqa: E501
    attention,
    attention_forward,
    dispatch,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.indexer_topk import (  # noqa: E501
    region,
)
from tensorrt_llm._torch.attention.backends.fmha import manager as fmha_manager
from tensorrt_llm._torch.attention.backends.interface import AttentionInputType
from tensorrt_llm._torch.attention.backends.sparse.dsa.backend import DSATrtllmAttention
from tensorrt_llm._torch.attention.backends.sparse.dsa.indexer import Indexer
from tensorrt_llm._torch.attention.backends.sparse.dsa.module import DSAHooks
from tensorrt_llm._torch.attention.backends.trtllm import TrtllmAttention
from tensorrt_llm._torch.attention.mla import MLA
from tensorrt_llm._torch.models.modeling_deepseekv3 import DeepseekV32Attention

pytestmark = pytest.mark.cpu_only

PREFILL, DECODE = dispatch.Phase.PREFILL, dispatch.Phase.DECODE


# --- built-in modules without their constructors ------------------------------------------


def _new(cls, **attributes):
    obj = cls.__new__(cls)
    if isinstance(obj, nn.Module):
        nn.Module.__init__(obj)
    for name, value in attributes.items():
        setattr(obj, name, value)
    return obj


def _attention(layer_idx=3, full_indexer=True, dense_mha=False):
    backend = _new(DSATrtllmAttention, indexer=_new(Indexer) if full_indexer else None)
    return _new(
        DeepseekV32Attention,
        layer_idx=layer_idx,
        sparse_attn_hooks=_new(DSAHooks, short_seq_mha_threshold=0, need_dense_mha=dense_mha),
        mqa=backend,
        mha=_new(TrtllmAttention) if dense_mha else None,
        indexer=backend.indexer,
    )


class _SparseMlaDecodeFmha:
    """Stands in for the sparse-MLA decode FMHA library."""


class _Tail:
    def __init__(self, mla) -> None:
        self.mla = mla


class _FastPaths:
    """``dispatch.is_enabled`` / ``get_impl`` for the adoption tests."""

    def __init__(self, enabled):
        self.enabled = frozenset(enabled)
        self.events = []
        events = self.events

        def make_attn_tail(mla):
            events.append(("make_attn_tail", mla.layer_idx))
            return _Tail(mla)

        def wrap_attn_tail(mla, tail):
            events.append(("wrap_attn_tail", tail))
            return ("staged", tail)

        self.entries = {
            ("attention_output_tail", "make_attn_tail"): make_attn_tail,
            ("oproj_weight_staging", "wrap_attn_tail"): wrap_attn_tail,
            ("sparse_mla_decode", "fmha_lib"): _SparseMlaDecodeFmha,
        }

    def is_enabled(self, path):
        assert path in dispatch.FAST_PATH_NAMES, path
        return path in self.enabled

    def get_impl(self, path, name):
        return self.entries[(path, name)]


@pytest.fixture
def fast_paths(monkeypatch):
    def install(enabled=dispatch.FAST_PATH_NAMES):
        paths = _FastPaths(enabled)
        monkeypatch.setattr(dispatch, "is_enabled", paths.is_enabled)
        monkeypatch.setattr(dispatch, "get_impl", paths.get_impl)
        return paths

    return install


def _keys(module):
    return (
        [name for name, _ in module.named_parameters()],
        [name for name, _ in module.named_buffers()],
        [name for name, _ in module.named_modules()],
    )


# --- adoption -----------------------------------------------------------------------------


def test_adoption_reclasses_the_attention_modules_and_builds_the_tail(fast_paths):
    paths = fast_paths()
    mla = _attention()
    hooks, backend, indexer = mla.sparse_attn_hooks, mla.mqa, mla.mqa.indexer
    before = _keys(mla)

    assert attention.GlmMLA.adopt(mla) is mla

    assert type(mla) is attention.GlmMLA
    assert type(hooks) is attention.GlmDsaHooks and mla.sparse_attn_hooks is hooks
    assert type(backend) is attention.GlmDsaBackend and mla.mqa is backend
    assert type(indexer) is attention.GlmIndexer and mla.indexer is indexer
    assert hooks.short_seq_mha_threshold == 0 and hooks.need_dense_mha is False
    # The tail is made for the layer, then wrapped by the o_proj weight staging.
    staged, tail = mla._attn_tail
    assert staged == "staged" and isinstance(tail, _Tail) and tail.mla is mla
    assert paths.events == [("make_attn_tail", 3), ("wrap_attn_tail", tail)]
    assert mla._tail_prefetch is True
    assert _keys(mla) == before


@pytest.mark.parametrize(
    "path", ["attention_output_tail", "attention_output_tail_prefetch", "attention_output_into_hop"]
)
def test_any_tail_path_builds_the_tail(fast_paths, path):
    paths = fast_paths({path})
    mla = attention.GlmMLA.adopt(_attention())
    assert isinstance(mla._attn_tail, _Tail) and paths.events == [("make_attn_tail", 3)]
    assert mla._tail_prefetch is (path == "attention_output_tail_prefetch")


def test_without_the_attention_paths_adoption_attaches_nothing(fast_paths):
    paths = fast_paths(enabled=())
    mla = attention.GlmMLA.adopt(_attention(full_indexer=False))
    assert paths.events == []
    assert mla._attn_tail is None and mla._tail_prefetch is False
    assert mla.mqa.prepended_fmha_lib_classes == ()
    assert "prepended_fmha_lib_classes" not in vars(mla.mqa)
    assert mla.mqa.indexer is None and type(mla.mqa) is attention.GlmDsaBackend


@pytest.mark.parametrize("dense_mha", [False, True], ids=["mqa only", "mha and mqa"])
def test_the_sparse_mla_decode_library_is_declared_on_each_backend(fast_paths, dense_mha):
    fast_paths({"sparse_mla_decode"})
    mla = attention.GlmMLA.adopt(_attention(dense_mha=dense_mha))
    backends = [mla.mqa] + ([mla.mha] if dense_mha else [])
    for backend in backends:
        assert backend.prepended_fmha_lib_classes == (_SparseMlaDecodeFmha,)
    assert TrtllmAttention.prepended_fmha_lib_classes == ()


def _other(cls):
    return type(f"_Other{cls.__name__}", (cls,), {})


def _refusal(case):
    mla = _attention()
    if case == "an attention subclass":
        mla.__class__ = _other(DeepseekV32Attention)
    elif case == "the MLA base class":
        mla.__class__ = MLA
    elif case == "a DSA hooks subclass":
        mla.sparse_attn_hooks.__class__ = _other(DSAHooks)
    elif case == "a DSA backend subclass":
        mla.mqa.__class__ = _other(DSATrtllmAttention)
    elif case == "an indexer subclass":
        mla.mqa.indexer.__class__ = _other(Indexer)
    elif case == "an adopted attention":
        mla = attention.GlmMLA.adopt(mla)
    return mla


@pytest.mark.parametrize(
    "case, got",
    [
        ("an attention subclass", "_OtherDeepseekV32Attention"),
        ("the MLA base class", "MLA"),
        ("a DSA hooks subclass", "_OtherDSAHooks"),
        ("a DSA backend subclass", "_OtherDSATrtllmAttention"),
        ("an indexer subclass", "_OtherIndexer"),
        ("an adopted attention", "GlmMLA"),
    ],
)
def test_adoption_refuses_anything_but_the_exact_builtin_class(fast_paths, case, got):
    fast_paths(enabled=())
    mla = _refusal(case)
    with pytest.raises(TypeError, match=f"adopts the built-in .*, got {got}$"):
        attention.GlmMLA.adopt(mla)


def test_the_classes_declare_no_state_of_their_own():
    for cls in (
        attention.GlmMLA,
        attention.GlmDsaHooks,
        attention.GlmDsaBackend,
        attention.GlmIndexer,
    ):
        assert "__slots__" not in vars(cls) and "__init__" not in vars(cls), cls
    assert attention.GlmIndexer._topk_bridge is None
    assert attention.GlmMLA._attn_tail is None and attention.GlmMLA._tail_prefetch is False


# --- the FMHA libraries -------------------------------------------------------------------


def _library(name, calls):
    def is_available(cls, attn):
        calls.append((name, attn.name))
        return True

    return type(
        name,
        (),
        {"is_available": classmethod(is_available), "__init__": lambda self, attn: None},
    )


def _managers(monkeypatch, registry, declared):
    """The libraries an mha and an mqa backend build, in the order ``create_weights`` builds
    their managers, and the order their availability is asked in."""
    calls = []
    libraries = {name: _library(name, calls) for name in ("sparse", "a", "b", "fallback")}
    monkeypatch.setattr(
        fmha_manager, "get_enabled_fmha_lib_classes", lambda: [libraries[n] for n in registry]
    )
    built = []
    for name in ("mha", "mqa"):
        attn = types.SimpleNamespace(name=name)
        if declared:
            attn.prepended_fmha_lib_classes = tuple(libraries[n] for n in declared)
        manager = fmha_manager.FmhaManager(attn)
        built.append([type(lib).__name__ for lib in manager.fmha_libs])
    return built, calls


def test_the_declared_library_reproduces_a_registry_with_it_first(monkeypatch):
    """A backend that declares the library builds the list and asks the libraries in the order a
    registry with the library first would; one that declares nothing builds the registry's."""
    first = _managers(monkeypatch, ["sparse", "a", "b", "fallback"], declared=())
    declared = _managers(monkeypatch, ["a", "b", "fallback"], declared=("sparse",))
    assert declared == first
    assert declared[0] == [["sparse", "a", "b", "fallback"]] * 2
    plain = _managers(monkeypatch, ["a", "b", "fallback"], declared=())
    assert plain[0] == [["a", "b", "fallback"]] * 2


# --- the phase gate -----------------------------------------------------------------------

# Each override and the built-in method it falls back to.
OVERRIDES = [
    (attention.GlmMLA, DeepseekV32Attention, "forward"),
    (attention.GlmMLA, DeepseekV32Attention, "forward_absorption_generation"),
    (attention.GlmDsaHooks, DSAHooks, "forward"),
    (attention.GlmDsaHooks, DSAHooks, "forward_custom_op"),
    (attention.GlmDsaBackend, DSATrtllmAttention, "sparse_attn_predict"),
    (attention.GlmIndexer, Indexer, "_fill_decode_topk"),
]


def _method(cls, name):
    tree = ast.parse(Path(attention.__file__).read_text())
    return next(
        n
        for c in tree.body
        if isinstance(c, ast.ClassDef) and c.name == cls.__name__
        for n in c.body
        if isinstance(n, ast.FunctionDef) and n.name == name
    )


@pytest.mark.parametrize("cls, builtin, name", OVERRIDES, ids=lambda x: getattr(x, "__name__", x))
def test_each_override_starts_with_the_phase_gate(cls, builtin, name):
    """``if dispatch.phase() is not DECODE: return super().<name>(<every built-in parameter by
    name>)``; the parameters only the decode composition takes are dropped."""
    node = _method(cls, name)
    docstring = isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant)
    gate = node.body[1] if docstring else node.body[0]
    assert ast.unparse(gate.test) == "dispatch.phase() is not dispatch.Phase.DECODE"
    assert len(gate.body) == 1 and not gate.orelse
    call = gate.body[0].value
    assert ast.unparse(call.func) == f"super().{name}" and not call.args
    parameters = inspect.signature(getattr(builtin, name)).parameters
    expected = [
        p.name if p.kind is not p.VAR_KEYWORD else None
        for p in parameters.values()
        if p.name != "self"
    ]
    assert [k.arg for k in call.keywords] == expected
    # The override keeps the built-in parameters, in order, and may add keywords after them.
    named = [p for p in parameters.values() if p.kind is not p.VAR_KEYWORD]
    own = inspect.signature(getattr(cls, name)).parameters.values()
    assert [p.name for p in own][: len(named)] == [p.name for p in named]
    assert any(p.kind is p.VAR_KEYWORD for p in own) == (len(named) < len(parameters))


def _record(monkeypatch, owner, name, calls, result=None):
    def recorder(*args, **kwargs):
        calls.append((name, args, kwargs))
        return f"{getattr(owner, '__name__', owner)}.{name}" if result is None else result

    monkeypatch.setattr(owner, name, recorder)


def _adopted(monkeypatch):
    monkeypatch.setattr(dispatch, "is_enabled", lambda path: False)
    return attention.GlmMLA.adopt(_attention())


@pytest.mark.parametrize("phase", [PREFILL, None], ids=["prefill", "outside a step"])
def test_off_decode_every_override_runs_the_builtin_method(monkeypatch, phase):
    mla = _adopted(monkeypatch)
    calls = []
    for owner, name in [
        (MLA, "forward"),
        (MLA, "forward_absorption_generation"),
        (DSAHooks, "forward"),
        (DSAHooks, "forward_custom_op"),
        (DSATrtllmAttention, "sparse_attn_predict"),
    ]:
        _record(monkeypatch, owner, name, calls)
    for name in (
        "forward_absorption_generation",
        "forward_sparse_attn",
        "forward_sparse_attn_custom_op",
    ):
        monkeypatch.setattr(attention_forward, name, lambda *a, **k: pytest.fail())
    monkeypatch.setattr(region, "cross_rank_topk_decode", lambda *a, **k: pytest.fail())

    def run():
        assert mla.forward("pos", "h", "md", "params", "latent") == "MLA.forward"
        assert (
            mla.forward_absorption_generation(
                "q", "kv", "pe", "md", "out", "pos", "lc", "absorbed", "deferred", extra=1
            )
            == "MLA.forward_absorption_generation"
        )
        assert mla.sparse_attn_hooks.forward(mla, "pos", "h", "md", ["o"]) == "DSAHooks.forward"
        assert (
            mla.sparse_attn_hooks.forward_custom_op(mla, "h", "pos", ["o"], None)
            == "DSAHooks.forward_custom_op"
        )
        assert (
            mla.mqa.sparse_attn_predict("q", None, "md", "args")
            == "DSATrtllmAttention.sparse_attn_predict"
        )
        assert mla.mqa.indexer._fill_decode_topk("md", "q", "w", "buf", 0, 6) is False

    if phase is None:
        run()
    else:
        with dispatch.step(phase):
            run()
    assert [(name, kwargs) for name, _, kwargs in calls] == [
        (
            "forward",
            dict(
                position_ids="pos",
                hidden_states="h",
                attn_metadata="md",
                all_reduce_params="params",
                latent_cache_gen="latent",
            ),
        ),
        (
            "forward_absorption_generation",
            dict(
                q="q",
                compressed_kv="kv",
                k_pe="pe",
                attn_metadata="md",
                output="out",
                position_ids="pos",
                latent_cache="lc",
                extra=1,
            ),
        ),
        (
            "forward",
            dict(
                mla=mla,
                position_ids="pos",
                hidden_states="h",
                attn_metadata="md",
                attn_output=["o"],
            ),
        ),
        (
            "forward_custom_op",
            dict(
                mla=mla,
                hidden_states="h",
                position_ids="pos",
                attn_output=["o"],
                latent_cache_gen=None,
            ),
        ),
        ("sparse_attn_predict", dict(q="q", k=None, metadata="md", forward_args="args")),
    ]


def test_on_a_decode_step_each_override_runs_the_decode_function(monkeypatch):
    mla = _adopted(monkeypatch)
    calls = []
    for name in (
        "forward_absorption_generation",
        "forward_sparse_attn",
        "forward_sparse_attn_custom_op",
    ):
        _record(monkeypatch, attention_forward, name, calls)
    for owner, name in [
        (MLA, "forward_absorption_generation"),
        (DSAHooks, "forward"),
        (DSAHooks, "forward_custom_op"),
    ]:
        monkeypatch.setattr(owner, name, lambda *a, **k: pytest.fail())

    with dispatch.step(DECODE):
        mla.forward_absorption_generation(
            "q", "kv", "pe", "md", "out", "pos", "lc", "absorbed", "deferred", extra=1
        )
        assert mla.sparse_attn_hooks.forward(mla, "pos", "h", "md", ["o"]) is None
        assert mla.sparse_attn_hooks.forward_custom_op(mla, "h", "pos", ["o"], None) is True

    assert calls == [
        (
            "forward_absorption_generation",
            (mla, "q", "kv", "pe", "md", "out"),
            dict(
                position_ids="pos",
                latent_cache="lc",
                absorbed_fused_q="absorbed",
                deferred_indexer_k="deferred",
                extra=1,
            ),
        ),
        ("forward_sparse_attn", (mla, "pos", "h", "md", ["o"]), {}),
        ("forward_sparse_attn_custom_op", (mla, "h", "pos", ["o"], None), {}),
    ]


# --- the indexer's top-k region and bridge ------------------------------------------------


def _indexer(gvr_deterministic=False):
    indexer = attention.GlmIndexer.adopt(_new(Indexer))
    indexer.top_k = types.SimpleNamespace(gvr_deterministic=gvr_deterministic)
    return indexer


@pytest.mark.parametrize("cross_rank", [True, False])
@pytest.mark.parametrize("gvr_deterministic", [False, True], ids=["default", "deterministic"])
def test_the_decode_topk_region_tries_the_cross_rank_topk_first(
    monkeypatch, cross_rank, gvr_deterministic
):
    calls = []
    buffer = torch.zeros(8, 4, dtype=torch.int32)

    def cross_rank_topk_decode(indexer, md, q, w, offset, rows, out, bridge):
        calls.append(("cross_rank_topk", offset, rows, out.data_ptr(), bridge))
        return cross_rank

    def indexer_topk_decode(indexer, md, q, w, offset, rows, out):
        calls.append(("indexer_topk", offset, rows, out.data_ptr()))
        return True

    monkeypatch.setattr(region, "cross_rank_topk_decode", cross_rank_topk_decode)
    monkeypatch.setattr(region, "indexer_topk_decode", indexer_topk_decode)
    indexer = _indexer(gvr_deterministic)
    indexer._topk_bridge = "bridge"

    with dispatch.step(DECODE):
        assert indexer._fill_decode_topk("md", "q", "w", buffer, 2, 6) is True

    rows = buffer[2:8].data_ptr()
    expected = [] if gvr_deterministic else [("cross_rank_topk", 2, 6, rows, "bridge")]
    if gvr_deterministic or not cross_rank:
        expected.append(("indexer_topk", 2, 6, rows))
    assert calls == expected


def _rope_kv_append(monkeypatch, served, calls):
    def entry(name):
        def run(*args):
            calls.append((name,) + args)
            return served

        return run

    monkeypatch.setattr(dispatch, "get_impl", lambda path, name: entry((path, name)))


@pytest.mark.parametrize(
    "bridge, has_prefill, generations, served",
    [
        ("bridge", False, 3, True),
        ("bridge", False, 3, False),
        ("bridge", True, 3, True),
        ("bridge", False, 1, True),
        (None, False, 3, True),
    ],
)
def test_the_reused_topk_copy_defers_to_the_bridge(
    monkeypatch, bridge, has_prefill, generations, served
):
    """Three generation rows; the bridge is asked only for a pure decode call with one row per
    request, and the built-in copy runs unless it takes the copy."""
    calls = []
    _rope_kv_append(monkeypatch, served, calls)
    indexer = _indexer()
    indexer._topk_bridge = bridge
    shared = torch.arange(12, dtype=torch.int32).view(3, 4)
    metadata = types.SimpleNamespace(shared_topk_indices=shared)
    buffer = torch.full((4, 4), -1, dtype=torch.int32)

    indexer._copy_reused_topk(metadata, buffer, 1, 3, generations, has_prefill, None)

    asked = bridge is not None and not has_prefill and generations == 3
    assert [c[0] for c in calls] == ([("rope_kv_append", "defer_reuse_copy")] if asked else [])
    if asked:
        _, bridge_arg, buffer_arg, offset, rows, has_prior = calls[0]
        assert (bridge_arg, buffer_arg, offset, has_prior) == ("bridge", buffer, 1, False)
        assert torch.equal(rows, shared)
    if asked and served:
        assert (buffer == -1).all()
    else:
        assert torch.equal(buffer[1:], shared[:generations].expand(3, 4))
        assert (buffer[0] == -1).all()


@pytest.mark.parametrize("served", [True, False])
def test_the_mtp_row_stash_defers_to_the_bridge(monkeypatch, served):
    calls, stock = [], []
    _rope_kv_append(monkeypatch, served, calls)
    monkeypatch.setattr(Indexer, "_store_mtp_shared_rows", lambda self, *a: stock.append(a))
    indexer = _indexer()
    indexer._topk_bridge = "bridge"
    accepted = torch.tensor([9, 1, 2, 3])
    metadata = types.SimpleNamespace(shared_topk_indices="shared", mtp_num_accepted=accepted)
    args = ("buffer", metadata, 1, 3, 18, 0, False, True, True)

    indexer._store_mtp_shared_rows(*args)

    (call,) = calls
    assert call[:6] == (("rope_kv_append", "defer_mtp_row_stash"), "bridge", "shared", 1, 3, 6)
    assert torch.equal(call[6], accepted[1:4])
    assert stock == ([] if served else [args])


@pytest.mark.parametrize(
    "bridge, has_decode, has_prefill, is_generation",
    [
        (None, True, False, True),
        ("bridge", False, False, True),
        ("bridge", True, True, True),
        ("bridge", True, False, False),
    ],
)
def test_the_mtp_row_stash_without_a_bridge_call_is_the_builtin_one(
    monkeypatch, bridge, has_decode, has_prefill, is_generation
):
    calls, stock = [], []
    _rope_kv_append(monkeypatch, True, calls)
    monkeypatch.setattr(Indexer, "_store_mtp_shared_rows", lambda self, *a: stock.append(a))
    indexer = _indexer()
    indexer._topk_bridge = bridge
    args = ("buffer", "md", 1, 3, 18, 0, has_prefill, has_decode, is_generation)
    indexer._store_mtp_shared_rows(*args)
    assert calls == [] and stock == [args]


def test_the_builtin_seams_are_the_builtin_statements():
    """The stock bodies of the three ``sparse_attn_indexer`` seams: the reuse copy, no decode
    top-k of their own, and the MTP row stash."""
    indexer = _new(Indexer)
    shared = torch.arange(12, dtype=torch.int32).view(3, 4)
    metadata = types.SimpleNamespace(shared_topk_indices=shared, mtp_num_accepted=None)
    buffer = torch.full((4, 4), -1, dtype=torch.int32)
    indexer._copy_reused_topk(metadata, buffer, 1, 3, 3, False, None)
    assert torch.equal(buffer[1:], shared) and (buffer[0] == -1).all()
    assert indexer._fill_decode_topk(metadata, None, None, buffer, 0, 4) is False

    # Two generation requests at next_n = 2: the last row of each is stashed after row 1.
    rows = torch.arange(16, dtype=torch.int32).view(4, 4)
    metadata.shared_topk_indices = torch.zeros(3, 4, dtype=torch.int32)
    indexer._store_mtp_shared_rows(rows, metadata, 1, 2, 4, 0, False, True, True)
    assert torch.equal(metadata.shared_topk_indices[1:], rows[[1, 3]])
    metadata.shared_topk_indices = None
    indexer._store_mtp_shared_rows(rows, metadata, 0, 2, 4, 0, False, True, True)
    assert torch.equal(metadata.shared_topk_indices, rows[[1, 3]])


class _Bridged(Exception):
    pass


@pytest.mark.parametrize("fails", [False, True], ids=["returns", "raises"])
def test_the_backend_hands_the_bridge_to_its_indexer_for_one_call(monkeypatch, fails):
    seen = []
    bridge_paths = {"rope_kv_append", "cross_rank_topk_global_emit"}
    monkeypatch.setattr(dispatch, "is_enabled", lambda path: path in bridge_paths)

    def get_impl(path, name):
        assert (path, name) in {
            ("cross_rank_topk_global_emit", "new_bridge_packet"),
            ("cross_rank_topk_global_emit", "remap_topk_to_global"),
        }, (path, name)
        if name == "new_bridge_packet":
            return lambda metadata, is_generation: ("bridge", is_generation)
        return lambda backend, topk, metadata, bridge: ("global", topk, bridge)

    monkeypatch.setattr(dispatch, "get_impl", get_impl)
    backend = _attention().mqa
    backend.__class__ = attention.GlmDsaBackend
    indexer = backend.indexer
    indexer.__class__ = attention.GlmIndexer
    indexer.mtp_index_share = False
    backend.get_local_layer_idx = lambda metadata: 0

    def forward_from_projected(metadata, q, intermediates, is_generation):
        seen.append((indexer._topk_bridge, is_generation))
        if fails:
            raise _Bridged
        return "topk"

    indexer.forward_from_projected = forward_from_projected
    metadata = types.SimpleNamespace(
        num_ctx_tokens=0,
        num_tokens=6,
        shared_topk_indices=None,
        in_mtp_draft_loop=False,
        kv_cache_manager=None,
    )
    args = types.SimpleNamespace(
        attention_input_type=AttentionInputType.generation_only,
        sparse_backend_args=types.SimpleNamespace(indexer_intermediates="intermediates"),
    )

    with dispatch.step(DECODE):
        if fails:
            with pytest.raises(_Bridged):
                backend.sparse_attn_predict("q", None, metadata, args)
        else:
            out = backend.sparse_attn_predict("q", None, metadata, args)
            assert out == (("global", "topk", ("bridge", True)), None)

    assert seen == [(("bridge", True), True)]
    assert indexer._topk_bridge is None


# --- the attention-tail adoption and the absorbed-query module check ----------------------


def test_the_absorbed_query_packet_takes_the_adopted_dsa_hooks(monkeypatch):
    query_proj = pytest.importorskip(
        f"{dispatch.__package__}.kernels.absorbed_query.query_proj",
        reason="the absorbed_query kernels do not import here",
    )
    import tensorrt_llm._utils as utils

    monkeypatch.setattr(utils, "get_sm_version", lambda: 103)
    mla = types.SimpleNamespace(
        sparse_attn_hooks=_new(attention.GlmDsaHooks), is_lite=True, fuse_qkv_a_proj=True
    )
    assert query_proj.module_eligibility(mla) == (False, "needs the fused [q_a | kv_a] projection")
    mla.sparse_attn_hooks = _new(DSAHooks)
    assert query_proj.module_eligibility(mla) == (False, "needs the fused [q_a | kv_a] projection")
    mla.sparse_attn_hooks = _new(_other(DSAHooks))
    assert query_proj.module_eligibility(mla) == (False, "not a DSA MLA module")


# --- the target's DSA op 2 ----------------------------------------------------------------


def _op_args():
    t = torch.zeros(2, 4)
    return (t, t, t, t, [t], None, "7", torch.zeros(2, 4))


def test_the_op_runs_the_targets_op2_function_on_the_registered_layer(monkeypatch):
    calls = []
    monkeypatch.setattr(
        attention_forward, "extract_extra_attrs", lambda layer, kind: ("md", f"mla{layer}{kind}")
    )
    monkeypatch.setattr(attention_forward, "_forward_dsa_attn", lambda *args: calls.append(args))
    args = _op_args()

    torch.ops.trtllm.glm_moe_dsa_mla_dsa_attn_inplace(*args)
    attention_forward.maybe_bcg_glm_moe_dsa_mla_dsa_attn_inplace(*args)

    assert len(calls) == 2
    for call in calls:
        assert call[0] == "mla7mla" and call[7] == "md" and call[6] is None
        assert call[8] is args[-1]


def test_the_op_mutates_only_its_output():
    schema = torch.ops.trtllm.glm_moe_dsa_mla_dsa_attn_inplace.default._schema
    assert [a.name for a in schema.arguments if a.alias_info and a.alias_info.is_write] == [
        "output"
    ]
    assert len(schema.returns) == 0


def test_the_ops_fake_runs_nothing(monkeypatch):
    from torch._subclasses.fake_tensor import FakeTensorMode

    monkeypatch.setattr(attention_forward, "_forward_dsa_attn", lambda *a: pytest.fail())
    with FakeTensorMode():
        args = _op_args()
        torch.ops.trtllm.glm_moe_dsa_mla_dsa_attn_inplace(*args)


def test_only_the_target_registers_the_op():
    """The op is this target's own: no other module of the package names it, and the target's
    op-2 module registers it once, so it exists only after the target is imported.
    """
    name = "glm_moe_dsa_mla_dsa_attn_inplace"
    module = Path(attention_forward.__file__).resolve()
    package = next(parent for parent in module.parents if parent.name == "tensorrt_llm")
    naming = [path for path in package.rglob("*.py") if name in path.read_text(encoding="utf-8")]
    assert naming == [module]
    source = module.read_text(encoding="utf-8")
    assert source.count(f'@torch.library.custom_op("trtllm::{name}"') == 1
    assert source.count(f"@{name}.register_fake") == 1
    assert hasattr(torch.ops.trtllm, name)
