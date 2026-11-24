# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Cache-layout compatibility of specialized SM107 context and generation target pairs."""

from __future__ import annotations

import ast
import copy
import functools
import importlib.util
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import pytest

pytestmark = pytest.mark.cpu_only

_PKG_ROOT = Path(importlib.util.find_spec("tensorrt_llm").submodule_search_locations[0]).resolve()
_FAMILY = _PKG_ROOT / "_torch/_experimental/modeling_v2/models/kimi_k31"
_TARGETS = {
    "cep": "kimi_k31_pro_nvfp4__sm_107__cep",
    "tep8": "kimi_k31_pro_nvfp4__sm_107__tep8",
    "dep16": "kimi_k31_pro_nvfp4__sm_107__dep16",
    "helix8": "kimi_k31_pro_nvfp4__sm_107__helix8",
}
# (context target, generation target), TEP first.
_PAIRS = (
    ("tep8", "tep8"),
    ("dep16", "dep16"),
    ("dep16", "helix8"),
)
# The specialized pairs: the default target's, and the context-parallel context server feeding a
# TEP8 generation server (the default target has no cep layout).
_SPECIALIZED_PAIRS = (*_PAIRS, ("cep", "tep8"))
_MANAGER = _FAMILY / "ops/ssm/rubin_kda_dsa_cache_manager.py"
_INDEX_CACHE = _FAMILY / "ops/attention/rubin_dsa_cache.py"
# The one bound argument that shapes no page: the pools are zeroed once at construction.
_NOT_LAYOUT = frozenset({"zero_gpu_pools"})
# Recipe and model constants of modeling.py that shape a page on either side.
_RECIPE = (
    "_INDEXER_K_DTYPE",
    "_STATE_DTYPE",
    "_DRAFT_HISTORY_DTYPE",
)
_MODEL = ("_K31", "_K31_KDA", "_K31_DSA")
# Engine defaults that shape a page.
_KV_DEFAULTS = ("tokens_per_block", "dtype", "mamba_ssm_cache_dtype")
# Shell hooks whose answer must not depend on the side.
_HOOKS = (
    "select_kv_cache_manager_cls",
    "requires_kv_cache_manager_v2",
    "get_preferred_transceiver_runtime",
)
# An E4M3 index key's scale group (the manager's quant_block_size).
_INDEX_QUANT_BLOCK = 128


@functools.lru_cache(maxsize=None)
def _parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text(), filename=str(path))


def _modeling(layout: str) -> ast.Module:
    return _parse(_FAMILY / _TARGETS[layout] / "modeling.py")


class _Torch:
    """Stands in for ``torch`` when a constant names a dtype: ``torch.float32`` -> its name."""

    def __getattr__(self, name: str) -> str:
        return f"torch.{name}"


@functools.lru_cache(maxsize=None)
def _constants(layout: str) -> dict:
    """The module-level constants of modeling.py that evaluate from literals, dtypes and tables."""
    ns: dict = {"MappingProxyType": MappingProxyType, "torch": _Torch()}
    for node in _modeling(layout).body:
        if isinstance(node, ast.Assign) and all(isinstance(t, ast.Name) for t in node.targets):
            try:
                value = eval(compile(ast.Expression(node.value), "<constant>", "eval"), ns)
            except Exception:
                continue
            for t in node.targets:
                ns[t.id] = value
    return ns


def _shell(layout: str) -> ast.ClassDef:
    found = [
        n
        for n in _modeling(layout).body
        if isinstance(n, ast.ClassDef)
        and any("DecoderModelForCausalLM" in ast.unparse(b) for b in n.bases)
    ]
    assert len(found) == 1, [n.name for n in found]
    return found[0]


def _method(cls: ast.ClassDef, name: str) -> ast.FunctionDef:
    found = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name]
    assert found, f"{cls.name} has no {name}"
    return found[0]


def _bound_arguments() -> set[str]:
    """The keyword-only parameters of the manager's ``bind_kv_cache_args``."""
    fn = next(
        n
        for n in _parse(_MANAGER).body
        if isinstance(n, ast.FunctionDef) and n.name == "bind_kv_cache_args"
    )
    return {a.arg for a in fn.args.kwonlyargs}


def _defaults(layout: str) -> dict:
    """The literal ``kv_cache_config`` defaults ``get_model_defaults`` returns (whatever the
    server role: a context server may add MoE defaults beside them)."""
    fn = _method(_shell(layout), "get_model_defaults")
    found = [
        ast.literal_eval(value)
        for node in ast.walk(fn)
        if isinstance(node, ast.Dict)
        for key, value in zip(node.keys, node.values)
        if isinstance(key, ast.Constant) and key.value == "kv_cache_config"
    ]
    assert len(found) == 1, "get_model_defaults names its kv_cache_config defaults once"
    return found[0]


def _hook_answer(layout: str, name: str) -> str:
    """What a hook returns, as source: its return expressions (calls by their callee)."""
    fn = _method(_shell(layout), name)
    out = []
    for node in ast.walk(fn):
        if isinstance(node, ast.Return) and node.value is not None:
            out.append(ast.unparse(node.value))
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
            out.append(ast.unparse(node.value.func))
    return " | ".join(out)


@pytest.mark.parametrize("layout", tuple(_TARGETS))
def test_each_role_binds_exactly_the_managers_arguments(layout):
    """Every role's table holds exactly the manager's bound arguments (a new argument must be
    classified here), and the shell binds its role's table unchanged."""
    table = _constants(layout)["_CACHE_ARGS"]
    roles = _constants(layout)["_ROLES"]
    assert set(table) == set(roles), (set(table), roles)
    for role in roles:
        assert set(table[role]) == _bound_arguments(), (role, sorted(table[role]))
    calls = [
        n
        for n in ast.walk(_method(_shell(layout), "__init__"))
        if isinstance(n, ast.Call) and ast.unparse(n.func).endswith("bind_kv_cache_args")
    ]
    assert len(calls) == 1, "the shell binds its cache arguments once"
    (call,) = calls
    assert [k.arg for k in call.keywords] == [None], ast.unparse(call)
    assert ast.unparse(call.keywords[0].value).startswith("_CACHE_ARGS["), ast.unparse(call)


@pytest.mark.parametrize(
    "ctx, gen", _SPECIALIZED_PAIRS, ids=[f"{c}-ctx-to-{g}-gen" for c, g in _SPECIALIZED_PAIRS]
)
def test_context_and_generation_sides_build_the_same_cache(ctx, gen):
    """The two sides of a disaggregated pair agree on everything that shapes a cache page."""
    c, g = _constants(ctx), _constants(gen)
    assert "context" in c["_ROLES"], f"{ctx} serves no context role"
    assert "generation" in g["_ROLES"], f"{gen} serves no generation role"
    c_args, g_args = c["_CACHE_ARGS"]["context"], g["_CACHE_ARGS"]["generation"]
    layout_keys = _bound_arguments() - _NOT_LAYOUT
    differ = {k: (c_args[k], g_args[k]) for k in layout_keys if c_args[k] != g_args[k]}
    assert not differ, f"cache arguments differ (context, generation): {differ}"
    for name in _RECIPE + _MODEL:
        assert name in c and name in g, f"{name} is not a constant of both targets"
        assert c[name] == g[name], f"{name}: {c[name]} vs {g[name]}"
    for key in _KV_DEFAULTS:
        assert _defaults(ctx).get(key) == _defaults(gen).get(key), f"kv_cache_config.{key}"
    for hook in _HOOKS:
        assert _hook_answer(ctx, hook) == _hook_answer(gen, hook), hook
    c_meta = [n for n in _shell(ctx).body if _shapes_a_page(n)]
    g_meta = [n for n in _shell(gen).body if _shapes_a_page(n)]
    assert [ast.unparse(n) for n in c_meta] == [ast.unparse(n) for n in g_meta], (
        "class attributes of the shells (mamba_metadata_cls)"
    )


def _shapes_a_page(node: ast.AST) -> bool:
    """A shell class attribute other than ``supports_cp_ulysses``, the parallel layout cep runs."""
    return isinstance(node, ast.Assign) and not any(
        isinstance(t, ast.Name) and t.id == "supports_cp_ulysses" for t in node.targets
    )


def _latent_bytes_per_token(kv_dtype: str, latent_dim: int) -> int:
    """An MLA layer's latent KV bytes per token: one byte per value in FP8 or BF16 x 2; NVFP4
    adds an E4M3 scale per 16 values to its half byte per value."""
    return {
        "fp8": latent_dim,
        "bf16": 2 * latent_dim,
        "nvfp4": latent_dim // 2 + latent_dim // 16,
    }[kv_dtype]


def _index_bytes_per_token(index_head_dim: int, indexer_k_dtype: str) -> int:
    """The manager's index-key bytes per token, run from its source."""
    ns: dict = {}
    for node in _parse(_INDEX_CACHE).body:
        if isinstance(node, ast.FunctionDef) and node.name in (
            "get_indexer_k_data_bytes_per_token",
            "get_indexer_k_cache_bytes_per_token",
        ):
            node = copy.deepcopy(node)
            node.returns = None
            for arg in node.args.args:
                arg.annotation = None
            module = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
            exec(compile(module, str(_INDEX_CACHE), "exec"), ns)
    return ns["get_indexer_k_cache_bytes_per_token"](
        index_head_dim, _INDEX_QUANT_BLOCK, indexer_k_dtype
    )


# --- the ULL generation target's pairs: what the transfer moves ---

_ULL = "kimi_k31_pro_nvfp4__sm_107__helix16_moetp16ep1"
_ULL_FEATURES = _FAMILY / "ops/common/rubin_ull_features.py"
# The context servers in front of the ULL generation server (cep through its DKV hand-off to the
# Helix ranks).
_ULL_CONTEXTS = ("dep16", "tep8", "cep")
# The drafter history's identity in the transfer (the engine's draft transfer identity): the
# container's constants, and the drafter architecture its weights.py declares.
_DRAFT_CONSTANTS = ("_kv_factor", "_num_kv_heads", "dflash_attention_backend")
_DRAFT_GEOMETRY = ("num_hidden_layers", "kv_lora_rank", "qk_rope_head_dim", "sliding_window")


def _module_constants(path: Path) -> dict:
    """The module-level constants of ``path`` that evaluate from literals, dtypes and tables."""
    ns: dict = {"MappingProxyType": MappingProxyType, "torch": _Torch()}
    for node in _parse(path).body:
        if isinstance(node, ast.Assign) and all(isinstance(t, ast.Name) for t in node.targets):
            try:
                value = eval(compile(ast.Expression(node.value), "<constant>", "eval"), ns)
            except Exception:
                continue
            for t in node.targets:
                ns[t.id] = value
    return ns


def _class_constants(path: Path, name: str) -> dict:
    """The literal class attributes of ``name`` in ``path``."""
    cls = next(n for n in _parse(path).body if isinstance(n, ast.ClassDef) and n.name == name)
    out = {}
    for node in cls.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            try:
                out[ast.unparse(node.targets[0])] = ast.literal_eval(node.value)
            except ValueError:
                continue
    return out


def _ull_shell() -> ast.ClassDef:
    found = [
        n
        for n in _parse(_FAMILY / _ULL / "modeling.py").body
        if isinstance(n, ast.ClassDef)
        and any("DecoderModelForCausalLM" in ast.unparse(b) for b in n.bases)
    ]
    assert len(found) == 1, [n.name for n in found]
    return found[0]


@pytest.mark.parametrize(
    "ctx", _ULL_CONTEXTS, ids=[f"{c}-ctx-to-helix16-gen" for c in _ULL_CONTEXTS]
)
def test_context_and_ull_generation_sides_agree_on_the_transfer(ctx):
    """A dep16, tep8 or cep context server and the ULL generation server agree on what the
    transfer moves: the latent KV, index-key and KDA state dtypes (the context side's cache
    arguments against the ULL recipe and constants), an FP8 latent KV on every MLA layer, the KV
    page size, KV cache manager V2 and the Python transceiver, and the drafter history's identity
    (layers, one KV head of kv_lora + rope values, FP8, the drafter's own window attention, one
    latent per token, the drafter's attention window). How much of the history each side's cache
    keeps is the next tests'."""
    c, g = _constants(ctx), _module_constants(_FAMILY / _ULL / "modeling.py")
    ull = _module_constants(_ULL_FEATURES)
    assert "context" in c["_ROLES"], f"{ctx} serves no context role"
    assert g["_ROLES"] == ("generation",), g["_ROLES"]
    c_args = c["_CACHE_ARGS"]["context"]
    assert c_args["kv_dtype"] == ull["KV_CACHE_QUANT_ALGO"].lower(), c_args["kv_dtype"]
    assert c_args["fp8_kv_layers"] is None, "an FP8 latent KV on every MLA layer"
    assert c_args["indexer_k_dtype"] == g["_INDEXER_K_DTYPE"] == ull["INDEXER_K_DTYPE"]
    assert c_args["state_dtype"] == g["_STATE_DTYPE"], (c_args["state_dtype"], g["_STATE_DTYPE"])
    for name in ("_INDEXER_K_DTYPE", "_STATE_DTYPE", "_DRAFT_HISTORY_DTYPE"):
        assert c[name] == g[name], f"{name}: {c[name]} vs {g[name]}"
    ull_defaults = ast.literal_eval(
        next(
            n
            for n in ast.walk(_method(_ull_shell(), "get_model_defaults"))
            if isinstance(n, ast.Return)
        ).value
    )["kv_cache_config"]
    assert _defaults(ctx).get("tokens_per_block") == ull_defaults.get("tokens_per_block")
    for hook in ("get_preferred_transceiver_runtime",):
        assert _hook_answer(ctx, hook) == ast.unparse(
            next(
                n for n in ast.walk(_method(_ull_shell(), hook)) if isinstance(n, ast.Return)
            ).value
        ), hook
    requires_v2 = _method(_ull_shell(), "requires_kv_cache_manager_v2")
    assert [ast.unparse(n.value) for n in ast.walk(requires_v2) if isinstance(n, ast.Return)] == [
        "True"
    ]
    c_draft = _class_constants(_FAMILY / _TARGETS[ctx] / "modeling.py", "DraftModel")
    g_draft = _class_constants(_FAMILY / _ULL / "modeling.py", "DraftModel")
    for name in _DRAFT_CONSTANTS:
        assert c_draft[name] == g_draft[name], (
            f"DraftModel.{name}: {c_draft[name]} vs {g_draft[name]}"
        )
    c_shape = _module_constants(_FAMILY / _TARGETS[ctx] / "weights.py")
    g_shape = _module_constants(_FAMILY / _ULL / "weights.py")
    for name in _DRAFT_GEOMETRY:
        assert c_shape["DRAFT_SHAPE"][name] == g_shape["DRAFT_SHAPE"][name], name
    assert c_shape["DRAFT_DFLASH"]["attention_mode"] == g_shape["DRAFT_DFLASH"]["attention_mode"]


# --- the drafter history window: every target's cache keeps the drafter's attention window ---

_ULL_MANAGER = _FAMILY / "ops/ssm/rubin_ull_kda_dsa_cache_manager.py"
_DRAFT_WINDOW_CACHE = _FAMILY / "ops/ssm/draft_window_cache.py"
# The helix16 generation targets of this tree; each builds the ULL manager.
_ULL_GENERATIONS = tuple(
    target
    for target in (_ULL, "kimi_k31_pro_nvfp4__sm_107__helix16_wide")
    if (_FAMILY / target / "modeling.py").is_file()
)
# The targets that build the Rubin context targets' manager: every context server, and the dep16
# and helix8 generation servers.
_RUBIN_MANAGER_TARGETS = ("cep", "tep8", "dep16", "helix8")
# The drafter's unified draft layers (global ids 93..97), its page size and one request.
_DRAFT_LAYER_IDS = tuple(range(93, 98))
_DRAFT_LATENT = 576
_DRAFT_PAGE = 64
_REQUEST = 7
_OPTION_CALL = "draft_cache_option(model_config, DRAFT_WINDOW_ATTR, True)"


def _shell_of(target: str) -> ast.ClassDef:
    """The ``DecoderModelForCausalLM`` shell of a target directory's ``modeling.py``."""
    found = [
        n
        for n in _parse(_FAMILY / target / "modeling.py").body
        if isinstance(n, ast.ClassDef)
        and any("DecoderModelForCausalLM" in ast.unparse(b) for b in n.bases)
    ]
    assert len(found) == 1, [n.name for n in found]
    return found[0]


def _top_level(path: Path, kind: type, name: str):
    """The top-level ``kind`` node (a class or function) named ``name`` in ``path``."""
    return next(n for n in _parse(path).body if isinstance(n, kind) and n.name == name)


def _calls(node: ast.AST, name: str) -> list:
    """The calls of ``name`` (a plain name or a dotted callee) under ``node``."""
    return [n for n in ast.walk(node) if isinstance(n, ast.Call) and ast.unparse(n.func) == name]


def _assert_manager_windows_by_default(path: Path, cls_name: str) -> None:
    """``cls_name`` in ``path`` keeps the drafter's window of its history unless
    ``kimi_k31_draft_window`` is off: the engine passes ``draft_window`` from
    ``draft_cache_option(model_config, DRAFT_WINDOW_ATTR, True)``; under it the constructor makes
    the drafter checkpoint's window its ``draft_window_size`` (``set_draft_window``) before the
    base construction; ``DraftWindowCacheMixin`` (first base) follows the window's history; the
    static estimate costs the draft pages at the window (``window_draft_cost``) under the same
    option."""
    cls = _top_level(path, ast.ClassDef, cls_name)
    assert ast.unparse(cls.bases[0]) == "DraftWindowCacheMixin", (cls_name, cls.bases)
    (kwargs,) = [
        n.value
        for n in ast.walk(_method(cls, "extra_init_kwargs"))
        if isinstance(n, ast.Return) and isinstance(n.value, ast.Dict)
    ]
    entries = {
        k.value: ast.unparse(v)
        for k, v in zip(kwargs.keys, kwargs.values)
        if isinstance(k, ast.Constant)
    }
    assert entries["draft_window"] == _OPTION_CALL, (cls_name, entries.get("draft_window"))
    init = _method(cls, "__init__")
    guarded = [
        n
        for n in ast.walk(init)
        if isinstance(n, ast.If)
        and ast.unparse(n.test) == "draft_window"
        and any(_calls(statement, "set_draft_window") for statement in n.body)
    ]
    assert len(guarded) == 1, f"{cls_name} sets the drafter's window under draft_window"
    base_inits = _calls(init, "super().__init__")
    assert base_inits, f"{cls_name} builds no base"
    assert guarded[0].lineno < min(n.lineno for n in base_inits), (
        f"{cls_name} sets the window after the base construction"
    )
    costed = [
        n
        for n in ast.walk(_method(cls, "get_cache_size_per_token"))
        if isinstance(n, ast.If)
        and ast.unparse(n.test) == _OPTION_CALL
        and any(_calls(statement, "window_draft_cost") for statement in n.body)
    ]
    assert len(costed) == 1, f"{cls_name} costs its draft pages at the window"


def _imported(path: Path, module_suffix: str) -> set[str]:
    """The names ``path`` imports from a module whose dotted name ends with ``module_suffix``."""
    return {
        alias.name
        for n in _parse(path).body
        if isinstance(n, ast.ImportFrom) and (n.module or "").endswith(module_suffix)
        for alias in n.names
    }


def _draft_cache_option():
    """The context manager's ``draft_cache_option`` (run from its source) and ``DRAFT_WINDOW_ATTR``."""
    fn = copy.deepcopy(_top_level(_MANAGER, ast.FunctionDef, "draft_cache_option"))
    module = ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[]))
    ns: dict = {}
    exec(compile(module, str(_MANAGER), "exec"), ns)
    return ns["draft_cache_option"], _module_constants(_MANAGER)["DRAFT_WINDOW_ATTR"]


def _set_draft_window():
    """``set_draft_window`` run from its source; the drafter checkpoint's window
    (``dflash2_draft_window``) is the spec config's ``checkpoint_window`` here."""
    fn = copy.deepcopy(_top_level(_DRAFT_WINDOW_CACHE, ast.FunctionDef, "set_draft_window"))
    fn.returns = None
    for arg in fn.args.args:
        arg.annotation = None
    module = ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[]))
    ns: dict = {"dflash2_draft_window": lambda spec_config: spec_config.checkpoint_window}
    exec(compile(module, str(_DRAFT_WINDOW_CACHE), "exec"), ns)
    return ns["set_draft_window"]


def _model_config(model_kwargs: dict) -> SimpleNamespace:
    """A model config whose checkpoint config took ``model_kwargs``: a ``text_config`` entry updates
    the text sub-config (as ``ModelConfig.from_pretrained`` merges it), any other key the top level."""
    text = SimpleNamespace(**model_kwargs.get("text_config", {}))
    top = {key: value for key, value in model_kwargs.items() if key != "text_config"}
    return SimpleNamespace(pretrained_config=SimpleNamespace(text_config=text, **top))


def _draft_window(target: str) -> int:
    """The drafter's attention window a target decodes with (its weights.py ``DRAFT_SHAPE``)."""
    return _module_constants(_FAMILY / target / "weights.py")["DRAFT_SHAPE"]["sliding_window"]


def _draft_manager(window, prompt: int, *, released: int = 0, scratch: int = 8):
    """A ``KVCacheManagerV2`` with only what its draft-history methods read: the drafter's five FP8
    layers of one 576-value latent, ``window`` as its ``draft_window_size`` and one request of
    ``prompt`` tokens whose draft pages (64 tokens each) cover the prompt and ``scratch``. The first
    ``released`` pages are released (``BAD_PAGE_INDEX``), as a windowed cache leaves the pages that
    fell out of its window."""
    from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2
    from tensorrt_llm.bindings import DataType
    from tensorrt_llm.runtime.kv_cache_manager_v2 import BAD_PAGE_INDEX

    pages = -(-(prompt + scratch) // _DRAFT_PAGE)
    indices = [BAD_PAGE_INDEX] * released + list(range(100, 100 + pages - released))
    manager = object.__new__(KVCacheManagerV2)
    manager._standalone_draft_layer_ids = _DRAFT_LAYER_IDS
    manager.layer_offsets = {layer: i for i, layer in enumerate(_DRAFT_LAYER_IDS)}
    manager.head_dim_per_layer = [_DRAFT_LATENT] * len(_DRAFT_LAYER_IDS)
    manager._draft_num_kv_heads = 1
    manager._draft_total_num_kv_heads = 1
    manager._draft_dtype = DataType.FP8
    manager._draft_kv_factor = 1
    manager.draft_window_size = window
    manager.draft_history = {}
    manager.tokens_per_block = _DRAFT_PAGE
    manager.max_blocks_per_seq = pages
    manager._dkv = None
    manager._helix_cp_size = 1
    manager._draft_cp_size = 1
    manager.kv_cache_map = {
        _REQUEST: SimpleNamespace(is_active=True, capacity=pages * _DRAFT_PAGE, num_blocks=pages)
    }
    manager.get_batch_cache_indices = lambda request_ids, layer_idx, raw_indices=False: [
        list(indices) for _ in request_ids
    ]
    return manager


def _released_pages(window: int, prompt: int) -> int:
    """Draft pages a windowed cache has released at history ``prompt``: those wholly before its
    retention (the window and the next write, ``draft_retention_window_size``)."""
    return max(0, prompt - (window + 1)) // _DRAFT_PAGE


@pytest.mark.parametrize("gen", _ULL_GENERATIONS)
def test_every_target_keeps_the_drafter_window_by_default(gen):
    """Every context target, the dep16 and helix8 generation servers and every ULL helix16
    generation server keep the drafter's window of its history by default, read from one option
    (``kimi_k31_draft_window``, default on) and one window (the drafter checkpoint's
    ``sliding_window``), so a context server hands over the history its generation server keeps."""
    for target in _RUBIN_MANAGER_TARGETS:
        assert _calls(
            _method(_shell(target), "select_kv_cache_manager_cls"),
            "select_rubin_kv_cache_manager_cls",
        ), target
    selector = _top_level(_MANAGER, ast.FunctionDef, "select_rubin_kv_cache_manager_cls")
    returns = [ast.unparse(n.value) for n in ast.walk(selector) if isinstance(n, ast.Return)]
    assert "RubinKdaDsaHybridCacheManager" in returns
    _assert_manager_windows_by_default(_MANAGER, "RubinKdaDsaHybridCacheManager")

    select = _method(_shell_of(gen), "select_kv_cache_manager_cls")
    imported = {
        alias.name
        for n in ast.walk(select)
        if isinstance(n, ast.ImportFrom)
        and (n.module or "").endswith("rubin_ull_kda_dsa_cache_manager")
        for alias in n.names
    }
    assert imported == {"KimiK3DSAHybridCacheManager"}, (gen, imported)
    assert ast.unparse(select.body[-1]) == "return KimiK3DSAHybridCacheManager", gen
    _assert_manager_windows_by_default(_ULL_MANAGER, "KimiK3DSAHybridCacheManager")
    # One option and one window rule on both sides.
    assert {"DRAFT_WINDOW_ATTR", "draft_cache_option"} <= _imported(
        _ULL_MANAGER, "rubin_kda_dsa_cache_manager"
    )
    for path in (_MANAGER, _ULL_MANAGER):
        assert {"DraftWindowCacheMixin", "set_draft_window", "window_draft_cost"} <= _imported(
            path, "draft_window_cache"
        ), path.name
    # Both sides decode with the same attention window.
    assert {_draft_window(_TARGETS[t]) for t in _RUBIN_MANAGER_TARGETS} == {_draft_window(gen)}


@pytest.mark.parametrize(
    "model_kwargs",
    [{}, {"text_config": {"kimi_k31_draft_window": True}}],
    ids=["default", "text-config-on"],
)
def test_the_window_is_the_drafter_checkpoints(model_kwargs):
    """By default (or with the option on) the manager's ``draft_window_size`` is the drafter
    checkpoint's ``sliding_window`` once the engine appends the drafter's layers; with the option
    off, or without draft layers, it stays unset (the whole history)."""
    option, attr = _draft_cache_option()
    window = _draft_window(_ULL)
    spec_config = SimpleNamespace(checkpoint_window=window)
    set_draft_window = _set_draft_window()
    assert option(_model_config(model_kwargs), attr, True) is True
    kwargs = {"draft_num_layers": len(_DRAFT_LAYER_IDS), "spec_config": spec_config}
    set_draft_window(kwargs)
    assert kwargs["draft_window_size"] == window
    no_draft = {"spec_config": spec_config}
    set_draft_window(no_draft)
    assert "draft_window_size" not in no_draft
    off = _model_config({"text_config": {"kimi_k31_draft_window": False}})
    assert option(off, attr, True) is False


@pytest.mark.parametrize("prompt", [700, 1024, 1025, 4133, 190000])
@pytest.mark.parametrize("ctx", _ULL_CONTEXTS, ids=[f"{c}-ctx" for c in _ULL_CONTEXTS])
def test_ull_generation_restores_a_windowed_context_history(ctx, prompt):
    """A context server's windowed history (the last ``min(prompt, window)`` positions before the
    prompt's end) restores into a windowed ULL generation cache whose pages before the window are
    released: KV cache manager V2's draft-history export / restore, with the receiver's page-table
    check. A whole-history receiver refuses it (the hand-off error of a windowed context server in
    front of a whole-history generation server), a windowed receiver refuses a whole history, and a
    page missing inside the window is refused."""
    window = _draft_window(_TARGETS[ctx])
    assert window == _draft_window(_ULL)
    held = min(prompt, window)

    context = _draft_manager(window, prompt, released=_released_pages(window, prompt))
    context.set_draft_history(_REQUEST, held, prompt)
    metadata = context.export_draft_history(_REQUEST)
    assert metadata["layout"]["window_size"] == window
    assert (metadata["valid_length"], metadata["position"]) == (held, prompt)

    generation = _draft_manager(window, prompt, released=_released_pages(window, prompt))
    generation.restore_draft_history(_REQUEST, metadata)
    history = generation.get_draft_history(_REQUEST)
    assert (history.valid_length, history.position) == (held, prompt)
    table = generation.get_draft_block_table([_REQUEST])
    first_held = (prompt - held) // _DRAFT_PAGE
    assert bool((table[0, first_held : -(-prompt // _DRAFT_PAGE)] >= 0).all())

    layout_error = "Standalone draft transfer layout does not match the receiving worker"
    whole = _draft_manager(None, prompt)
    with pytest.raises(ValueError, match=layout_error):
        whole.restore_draft_history(_REQUEST, metadata)
    assert whole.get_draft_history(_REQUEST) is None
    whole_context = _draft_manager(None, prompt)
    whole_context.set_draft_history(_REQUEST, prompt, prompt)
    windowed = _draft_manager(window, prompt, released=_released_pages(window, prompt))
    with pytest.raises(ValueError, match=layout_error):
        windowed.restore_draft_history(_REQUEST, whole_context.export_draft_history(_REQUEST))

    if first_held > 0:
        short = _draft_manager(window, prompt, released=first_held + 1)
        with pytest.raises(ValueError, match="missing pages in its required history"):
            short.restore_draft_history(_REQUEST, metadata)
        assert short.get_draft_history(_REQUEST) is None
