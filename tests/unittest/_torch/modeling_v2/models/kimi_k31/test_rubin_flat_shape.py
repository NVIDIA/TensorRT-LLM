# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Flat-forward source contracts of the specialized SM107 targets and their family operations."""

from __future__ import annotations

import ast
import copy
import functools
import importlib.util
import re
from pathlib import Path

import pytest
from _rubin_ast import block_rows

pytestmark = pytest.mark.cpu_only

_PKG_ROOT = Path(importlib.util.find_spec("tensorrt_llm").submodule_search_locations[0]).resolve()
_MV2_PKG = "tensorrt_llm._torch._experimental.modeling_v2"
_FAMILY_PKG = f"{_MV2_PKG}.models.kimi_k31"
_FAMILY = _PKG_ROOT / "_torch/_experimental/modeling_v2/models/kimi_k31"
_TESTS_ROOT = Path(__file__).resolve().parents[5]
_KT = "unittest/_torch/modeling_v2/models/kimi_k31"

_SPECIALIZED = {
    "cep": "kimi_k31_pro_nvfp4__sm_107__cep",
    "tep8": "kimi_k31_pro_nvfp4__sm_107__tep8",
    "dep16": "kimi_k31_pro_nvfp4__sm_107__dep16",
    "helix8": "kimi_k31_pro_nvfp4__sm_107__helix8",
}
# The default target: the three layouts in one modeling.py, one bound at construction through
# its ``_BLOCKS`` table (layout-specific code under a layout suffix: ``mla_generate_tep8``).
_TARGETS = dict(_SPECIALIZED)
_LAYOUTS = tuple(_SPECIALIZED)
_ALL = tuple(_TARGETS)

# The default target's ``_Blocks`` fields. A per-step field's functions run every step (roots of
# the per-step bodies); a class field's listed methods do; the other fields are bound at
# construction or reached by name from a root. A new field must be classified here.
_STEP_FIELDS = frozenset(
    {
        "step_args",
        "mix_in",
        "mix_out",
        "tap",
        "kda_prefill",
        "kda_verify",
        "kda_decode",
        "mla_prefill",
        "mla_generate",
        "latent_moe",
        "dense_mlp",
        "step_prologue",
        # The one-token decode without a drafter, where a layout binds its own.
        "kda_decode_nospec",
    }
)
_STEP_CLASS_FIELDS = {"DFlashWorker": ("forward", "_forward_impl"), "prefill_target": ("forward",)}
_NOT_ROOT_FIELDS = frozenset(
    {
        "build_runtime",
        "layer_views",
        "head_view",
        "draft_config_for",
        "DraftModel",
        "check_lever_env",
        "tp_mapping",
        "place_experts",
    }
)
# A layout or recipe-set suffix of a default-target name (``mla_generate_tep8_ik4``).
_SUFFIX = re.compile(r"(_(tep8|dep16|helix8)(_[a-z0-9]+)*|(Tep8|Dep16|Helix8)([A-Z][a-z0-9]*)*)$")

# The runtime helpers the engine forces, relative to the family's ops/. A builder module
# also holds entries; only its module-level functions are builders. The default target's DSA
# carriers are DSA backend files, its metadata classes (with their index-key and latent-KV
# hook mixins) DSA metadata files.
_DSA_BACKEND_FILES = ("attention/rubin_dsa_backend.py", "attention/rubin_dsa_default_backend.py")
_DSA_METADATA_FILES = (
    "attention/rubin_dsa_metadata.py",
    "attention/rubin_dsa_default_metadata.py",
    "attention/rubin_dsa_default_fp4.py",
    "attention/rubin_dsa_default_nvfp4.py",
)
# Each DSA metadata module's test, which lists the methods its classes override.
_METADATA_OVERRIDE_TESTS = {
    "attention/rubin_dsa_metadata.py": "ops/attention/test_rubin_dsa_metadata_overrides.py",
    "attention/rubin_dsa_default_metadata.py": "ops/attention/test_rubin_dsa_default_metadata.py",
}
_CACHE_MANAGER_FILES = (
    "ssm/rubin_kda_dsa_cache_manager.py",
    "ssm/rubin_cache_policy.py",
    "attention/rubin_dsa_cache.py",
)
_KDA_METADATA_FILES = ("ssm/rubin_kda_metadata.py",)
# Host plans a target computes once a step: cep's row slices (pure host arithmetic, no launch), its
# DKV plan (the new-token pages, each history token's row in the other ranks' pools), and the
# sequence-split KDA's hand-off plan with the layer that runs it (entry calls and torch glue).
_STEP_PLAN_FILES = (
    "attention/rubin_cp_mla_rows.py",
    "attention/rubin_cp_dkv.py",
    "ssm/rubin_kda_handoff.py",
)
_BUILDER_FILES = (
    "comm/rubin_comm.py",
    "comm/rubin_push_allreduce.py",
    "attention/rubin_dsa_key_shard.py",
    "moe/rubin_megamoe.py",
    "moe/rubin_moe_a2a.py",
    "moe/rubin_moe_replica_plan.py",
    "moe/rubin_fc12.py",
    "moe/rubin_ull_cp_tail.py",
)
# The ULL generation target's runtime helpers: its DSA metadata class, its recipe (the features the
# target and its kernels read), the push all-reduce's buffers, the KDA + DSA cache manager and the
# headtp drafter's window ring (its buffers, targets, stores and first-GEN fill).
_ULL_HELPER_FILES = (
    "attention/rubin_ull_dsa_metadata.py",
    "common/rubin_ull_features.py",
    "comm/rubin_ull_push_allreduce.py",
    "ssm/rubin_ull_kda_dsa_cache_manager.py",
    "spec/rubin_ull_dflash2_headtp_ring.py",
)
_HELPERS = frozenset(
    _DSA_BACKEND_FILES
    + _DSA_METADATA_FILES
    + _CACHE_MANAGER_FILES
    + _KDA_METADATA_FILES
    + _BUILDER_FILES
    + _ULL_HELPER_FILES
    + _STEP_PLAN_FILES
)

# Environment variables that pick a kernel or its schedule (levers). The flat path takes each as
# a constant; no file on it may name one.
_LEVER_ENV = (
    "TRTLLM_K3_FC12_SM_RESERVE",
    "TRTLLM_K3_FC12_DIE_SCHED",
    "TRTLLM_K3_FC12_DIE_SCHED_MIN_TOKENS",
    "TRTLLM_K3_FC12_SCHED_LOOKAHEAD",
    "TRTLLM_K3_FC12_DIE_LAYOUT",
    "TRTLLM_CUTEDSL_FC2_N_TILE_SIZE",
    "TLLM_MOE_ASYNC_INPUT_QUANT",
    "MEGAMOE_COMBINE_FORMAT",
    "MEGAMOE_TACTIC_AUTOTUNE",
    "MEGAMOE_PRIME_LADDER",
    "TRTLLM_NVLINK_ONE_SIDED_A2A_WORKSPACE_MB",
    "TRTLLM_NVLINK_ONE_SIDED_A2A_FORCE_CFT",
    "TRTLLM_NVLINK_ONE_SIDED_A2A_CFT_MAX_BATCH_FOR_DISPATCH",
    "TRTLLM_NVLINK_ONE_SIDED_A2A_CFT_MAX_BATCH_FOR_COMBINE",
    "TLLM_K31_DFLASH2_VERIFY",
    "TRTLLM_DISABLE_FUSED_DSA_METADATA",
    "TRTLLM_FORCE_MNNVL_AR",
)

# Compute that must go through an entry: torch / functional calls and tensor methods.
_COMPUTE = frozenset(
    {
        "matmul",
        "mm",
        "bmm",
        "baddbmm",
        "addmm",
        "einsum",
        "linear",
        "layer_norm",
        "rms_norm",
        "group_norm",
        "softmax",
        "log_softmax",
        "silu",
        "gelu",
        "relu",
        "sigmoid",
        "topk",
        "conv1d",
        "scaled_dot_product_attention",
    }
)
_COMPUTE_ROOTS = ("torch.", "F.", "nn.functional.", "torch.nn.functional.")

# Module packages a target may not import (engine modules, engine MoE, the engine DSA package,
# the engine drafters, the die-locality helper).
_ENGINE_MODULE_PACKAGES = (
    "tensorrt_llm._torch.modules",
    "tensorrt_llm._torch.moe",
    "tensorrt_llm._torch.attention.backends.sparse.dsa",
    "tensorrt_llm._torch.models.modeling_dflash",
    "tensorrt_llm._torch.models.modeling_dspark",
    "tensorrt_llm._torch.die_locality",
)
_DRAFTER_MODULES = ("modeling_dflash", "modeling_dspark")

# Stock names modeling.py imports that run no computation (types, enums, the logger): they stay
# out of UNCERTIFIED_GENERIC_CALLS.
# get_env_enable_pdl reads the engine-wide PDL switch (a launch attribute, no computation).
_NON_COMPUTE = frozenset(
    {"AttentionMetadata", "Mapping", "ModelConfig", "QuantAlgo", "get_env_enable_pdl", "logger"}
)

# Bodies that run per step start here; everything they reach by name is per step too.
_STEP_ROOTS = (
    ("PrefillTarget", "forward"),
    ("DecodeTarget", "forward"),
    ("{core}", "forward"),
    ("{core}", "_select_target"),
    ("{shell}", "forward"),
    ("DFlashWorker", "forward"),
    ("DFlashWorker", "_forward_impl"),
    (None, "draft_step"),
)
# Construction-time functions a per-step body may name without running them per step.
_NOT_PER_STEP = frozenset({"__init__", "post_load_weights", "_build_runtime", "_layer_views"})
# What a per-step body may not read: a mapping, the environment, the launch-constant tables.
_STEP_NEEDLES = ("mapping.", "os.environ", "getenv", "environ.get")
_CONSTANT_TABLES = ("_CONSTANTS", "_MOE_ARGS", "_CACHE_ARGS")

# Known rule breaks: (check, scope, pattern of what) -> the change that removes it.
_PENDING: dict[tuple[str, str, str], str] = {
    # The ULL generation target's vendored MoE route tail and KDA MTP decode register torch ops;
    # their entries call the ops.
    **{
        ("op-registration", "family", rf".*/kernels/{f}\.py: "): (
            "the entries call direct launchers instead of the registered ops"
        )
        for f in ("moe/rubin_ull_k3_route_tail", "ssm/rubin_ull_kda_cute_dsl_kimi_k3_kda_mtp_ops")
    },
}

# Rule findings the design allows, by name, with the reason (not pending: they stay).
_ALLOWED: dict[tuple[str, str, str], str] = {
    # The all-to-all builder sets its workspace up with the engine's three setup ops,
    # once, outside capture; the dispatch and combine run through the two entries.
    (
        "entry-calls",
        "moe/rubin_moe_a2a.py",
        r"(rubin_moe_a2a_workspace|workspace_layout): torch\.ops(\.trtllm(\.moe_a2a_"
        r"(initialize|cft_initialize|get_workspace_layout))?)?$",
    ): "the all-to-all workspace's one-time setup ops",
    # The construction checks name the lever variables to assert them unset or equal to the
    # target's constants; they take the environment as an argument and read none.
    **{
        ("environment", "family", rf".*/ops/moe/{f}\.py names [A-Z0-9_]+$"): (
            "the construction checks of the lever variables"
        )
        for f in ("rubin_fc12", "rubin_cutedsl_fc2_finalize", "rubin_moe_a2a", "rubin_megamoe")
    },
}


# --- source readers ---


@functools.lru_cache(maxsize=None)
def _parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text(), filename=str(path))


def _target_dir(layout: str) -> Path:
    return _FAMILY / _TARGETS[layout]


def _modeling(layout: str) -> ast.Module:
    return _parse(_target_dir(layout) / "modeling.py")


def _rel(path: Path) -> str:
    return path.resolve().relative_to(_PKG_ROOT.parent).as_posix()


def _package_of(path: Path) -> str:
    parts = list(path.resolve().relative_to(_PKG_ROOT.parent).with_suffix("").parts)
    return ".".join(parts[:-1])


def _resolve(module: str | None, level: int, package: str) -> str:
    if level == 0:
        return module or ""
    parts = package.split(".")
    base = parts[: len(parts) - level + 1]
    return ".".join(base + ([module] if module else []))


def _imports(path: Path, tree: ast.AST | None = None) -> list[tuple[str, str | None, int]]:
    """``(module, name, line)`` of each import; ``name`` is None for ``import module``."""
    package = _package_of(path)
    found = []
    for node in ast.walk(tree if tree is not None else _parse(path)):
        if isinstance(node, ast.Import):
            found.extend((a.name, None, node.lineno) for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            module = _resolve(node.module, node.level, package)
            found.extend((module, a.name, node.lineno) for a in node.names)
    return found


def _base_names(cls: ast.ClassDef) -> list[str]:
    return [ast.unparse(b.value if isinstance(b, ast.Subscript) else b) for b in cls.bases]


def _classes(tree: ast.Module) -> dict[str, ast.ClassDef]:
    return {n.name: n for n in tree.body if isinstance(n, ast.ClassDef)}


def _methods(cls: ast.ClassDef) -> dict[str, ast.FunctionDef]:
    return {n.name: n for n in cls.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}


def _core_and_shell(tree: ast.Module) -> tuple[ast.ClassDef, ast.ClassDef]:
    found = _classes(tree)
    cores = [c for c in found.values() if "DecoderModel" in _base_names(c)]
    shells = [c for c in found.values() if "DecoderModelForCausalLM" in _base_names(c)]
    assert len(cores) == 1 and len(shells) == 1, (
        f"one core and one shell, found {[c.name for c in cores]}, {[c.name for c in shells]}"
    )
    return cores[0], shells[0]


def _chain(node: ast.AST) -> str | None:
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return None


def _base_name(name: str) -> str:
    """A name without its layout / recipe-set suffix (the default target's naming rule)."""
    return _SUFFIX.sub("", name)


def _is_constant_table(name: str) -> bool:
    return _base_name(name) in _CONSTANT_TABLES


def _top_level(tree: ast.Module) -> dict[str, ast.AST]:
    """Module-level functions and classes by name."""
    return {
        n.name: n
        for n in tree.body
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }


def _block_field_problems(tree: ast.Module) -> set[str]:
    """Every ``_Blocks`` field is classified (per step, class, or neither); every name a
    ``_BLOCKS`` row binds is a module-level definition or import (none is made in a function), of
    the row's layout or shared; every row binds its layout's drafter container and the one worker
    (``DraftModel<Layout>``, ``DFlashWorker``; the DFlash2 worker contract checks both on this
    target)."""
    rows = block_rows(tree)
    found = set()
    known = _STEP_FIELDS | set(_STEP_CLASS_FIELDS) | _NOT_ROOT_FIELDS
    defined = set(_top_level(tree))
    for node in tree.body:
        if isinstance(node, ast.Assign):
            defined |= {t.id for t in node.targets if isinstance(t, ast.Name)}
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            defined.add(node.target.id)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            defined |= {(a.asname or a.name).split(".")[0] for a in node.names}
    for key, row in rows.items():
        for field, name in row.items():
            if field not in known:
                found.add(f"_Blocks field {field} is not classified as per step or not")
            if name != "None" and name not in defined:
                found.add(f"{key}.{field} = {name} is not a module-level definition")
            layout = (suffix := _SUFFIX.search(name)) and (suffix[2] or suffix[4].lower())
            if layout not in (None, key[0]):
                found.add(f"{key}.{field} = {name} is another layout's")
        drafter = (row.get("DraftModel"), row.get("DFlashWorker"))
        if drafter != (f"DraftModel{key[0].capitalize()}", "DFlashWorker"):
            found.add(f"{key} binds the drafter {drafter}")
    return found


def _family_ops(pattern: str) -> list[Path]:
    return sorted(p for p in (_FAMILY / "ops").glob(pattern) if "__pycache__" not in p.parts)


def _family_kernels(pattern: str) -> list[Path]:
    return sorted(p for p in (_FAMILY / "kernels").glob(pattern) if "__pycache__" not in p.parts)


def _check(rule: str, scope: str, found: set[str]) -> None:
    """Fail on a rule break no pending entry covers, and on a pending entry that matches nothing."""
    allowed = [what for (r, s, what) in _ALLOWED if r == rule and s == scope]
    found = {f for f in found if not any(re.match(p, f) for p in allowed)}
    pending = {what: item for (r, s, what), item in _PENDING.items() if r == rule and s == scope}
    new = sorted(f for f in found if not any(re.match(p, f) for p in pending))
    gone = sorted(
        f"{p!r} ({item})" for p, item in pending.items() if not any(re.match(p, f) for f in found)
    )
    assert not new, f"{rule} {scope}: " + "; ".join(new)
    assert not gone, f"{rule} {scope}: fixed, remove from _PENDING: " + "; ".join(gone)


# --- the classes of modeling.py ---


def _module_classes(tree: ast.Module) -> set[str]:
    """Classes of modeling.py that are torch modules: a module base, or a class of this set."""
    mods: set[str] = set()
    cls = _classes(tree)
    changed = True
    while changed:
        changed = False
        for name, node in cls.items():
            if name in mods:
                continue
            bases = _base_names(node)
            if any(
                b.split(".")[-1] in ("Module", "DecoderModel", "DecoderModelForCausalLM")
                or b.startswith("_Engine")
                or b in mods
                or b.split(".")[-1].endswith("Worker")
                for b in bases
            ):
                mods.add(name)
                changed = True
    return mods


def _builds_module(value: ast.AST, module_classes: set[str]) -> bool:
    """Whether an assigned value constructs a torch module or parameter."""
    if not isinstance(value, ast.Call):
        return False
    callee = _chain(value.func) or ""
    leaf = callee.split(".")[-1]
    if callee.startswith(("nn.", "torch.nn.")) and leaf[:1].isupper():
        return True
    return leaf in module_classes


def _weight_rows(core: ast.ClassDef) -> bool:
    """Whether the core holds its layout's weight row (``self._wb = _weights._WBLOCKS[...]``):
    the default target's weight table is that row's."""
    for node in ast.walk(core):
        if isinstance(node, ast.Assign) and any(_chain(t) == "self._wb" for t in node.targets):
            if ast.unparse(node.value).startswith("_weights._WBLOCKS["):
                return True
    return False


@pytest.mark.parametrize("layout", _ALL)
def test_the_core_holds_one_parameter_dict(layout):
    """the core's only module attribute is ``w``; the shell's are the engine's and the
    drafter's; modeling.py declares no parameter or buffer of its own. The default target's
    core takes ``w`` from its layout's weight row."""
    tree = _modeling(layout)
    core, shell = _core_and_shell(tree)
    mods = _module_classes(tree)
    found = set()
    tables = ("_weights.declare(", "nn.ParameterDict(")
    if _weight_rows(core):
        tables += ("self._wb.declare(",)
    for cls, allowed in (
        (core, {"w"}),
        (shell, {"model", "lm_head", "logits_processor", "spec_worker", "draft_model"}),
    ):
        for fn in _methods(cls).values():
            for node in ast.walk(fn):
                if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                    continue
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for t in targets:
                    attr = _chain(t)
                    if not (attr and attr.startswith("self.") and attr.count(".") == 1):
                        continue
                    name = attr[5:]
                    if name == "w" and cls is core:
                        value = ast.unparse(node.value)
                        if not value.startswith(tables):
                            found.add(f"{cls.name}.w is {value}, not the weight table")
                    elif _builds_module(node.value, mods) and name not in allowed:
                        found.add(f"{cls.name}.{name} is a module ({ast.unparse(node.value)[:60]})")
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            callee = _chain(node.func) or ""
            if callee.split(".")[-1] in (
                "Parameter",
                "ParameterList",
                "ParameterDict",
                "register_parameter",
                "register_buffer",
            ):
                found.add(f"modeling.py makes a parameter or buffer: {callee}")
    _check("core-modules", layout, found)


@pytest.mark.parametrize("layout", _ALL)
def test_two_whole_targets_dispatched_on_num_contexts(layout):
    """``PrefillTarget`` / ``DecodeTarget`` are whole (no shared base, own forward);
    ``_select_target`` reads only ``num_contexts``; the decode target never reads the context
    counts; helix8's prefill target refuses (its layout serves generation steps only)."""
    tree = _modeling(layout)
    found = set()
    cls = _classes(tree)
    core, _ = _core_and_shell(tree)
    for name in ("PrefillTarget", "DecodeTarget"):
        node = cls.get(name)
        if node is None:
            found.add(f"no {name}")
            continue
        if [b for b in _base_names(node) if b != "object"]:
            found.add(f"{name} derives from {_base_names(node)}")
        if "forward" not in _methods(node):
            found.add(f"{name} has no forward of its own")
    select = _methods(core).get("_select_target")
    if select is None:
        found.add("the core has no _select_target")
    else:
        param = select.args.args[1].arg
        reads = {
            n.attr
            for n in ast.walk(select)
            if isinstance(n, ast.Attribute)
            and isinstance(n.value, ast.Name)
            and n.value.id == param
        }
        if reads != {"num_contexts"}:
            found.add(f"_select_target reads {sorted(reads)}")
    decode = cls.get("DecodeTarget")
    if decode is not None:
        text = ast.unparse(decode)
        for field in ("num_contexts", "num_ctx_tokens"):
            if re.search(rf"\.{field}\b", text):
                found.add(f"DecodeTarget reads .{field}")
    prefill = cls.get("PrefillTarget")
    if layout == "helix8" and prefill is not None and not _refuses(prefill):
        found.add("helix8 PrefillTarget.forward does not refuse")
    # The default target binds the prefill target per layout: helix8's refuses, the others' is
    # the whole PrefillTarget.
    for key, row in sorted(block_rows(tree).items()):
        name = row.get("prefill_target")
        target = cls.get(name or "")
        if target is None:
            found.add(f"{key} binds no prefill target class ({name})")
        elif key[0] == "helix8" and not _refuses(target):
            found.add(f"{key}: {name}.forward does not refuse")
        elif key[0] != "helix8" and name != "PrefillTarget":
            found.add(f"{key} binds {name}, not PrefillTarget")
        elif [b for b in _base_names(target) if b != "object"]:
            found.add(f"{name} derives from {_base_names(target)}")
    _check("targets", layout, found)


def _refuses(cls: ast.ClassDef) -> bool:
    """Whether a target class's ``forward`` starts by raising (docstring aside)."""
    fwd = _methods(cls).get("forward")
    body = [s for s in fwd.body if not isinstance(s, ast.Expr)] if fwd is not None else []
    return bool(body) and isinstance(body[0], ast.Raise)


@pytest.mark.parametrize("layout", _ALL)
def test_no_per_layer_module_and_no_engine_module_import(layout):
    """the only module classes of modeling.py are the core, the shell, the drafter worker
    and (if it is a module) the drafter container; no engine module package is imported."""
    tree = _modeling(layout)
    core, shell = _core_and_shell(tree)
    allowed = {core.name, shell.name, "DFlashWorker", "DraftModel"}
    # The default target: each layout's drafter container and worker, as its rows bind them.
    allowed |= {
        row.get(f) for row in block_rows(tree).values() for f in ("DraftModel", "DFlashWorker")
    } - {None}
    found = {f"module class {c}" for c in _module_classes(tree) - allowed}
    path = _target_dir(layout) / "modeling.py"
    for module, name, _ in _imports(path):
        full = f"{module}.{name}" if name else module
        for banned in _ENGINE_MODULE_PACKAGES:
            if full == banned or full.startswith(banned + "."):
                found.add(f"imports {full}")
    _check("module-classes", layout, found)


# --- kernel launches are entry calls ---


def _launch_breaks(path: Path, scope: list[ast.AST]) -> set[str]:
    """``torch.ops`` use, DeepGEMM or family-kernel imports, and compute outside an entry."""
    found = set()
    kernels = f"{_FAMILY_PKG}.kernels"
    for module, name, _ in _imports(path):
        full = f"{module}.{name}" if name else module
        if "deep_gemm" in full:
            found.add(f"imports {full}")
        if full == kernels or full.startswith(kernels + "."):
            found.add(f"imports the family kernel {full}")
    for root in scope:
        where = getattr(root, "name", path.name)
        for node in ast.walk(root):
            chain = _chain(node) if isinstance(node, ast.Attribute) else None
            if chain and (chain.startswith("torch.ops.") or chain == "torch.ops"):
                found.add(f"{where}: {'.'.join(chain.split('.')[:4])}")
            if isinstance(node, ast.BinOp) and isinstance(node.op, ast.MatMult):
                found.add(f"{where}: the @ operator")
            if isinstance(node, ast.Call):
                callee = _chain(node.func)
                if callee is None and isinstance(node.func, ast.Attribute):
                    callee = "<tensor>." + node.func.attr
                if not callee:
                    continue
                leaf = callee.split(".")[-1]
                if leaf in _COMPUTE and (
                    callee.startswith(_COMPUTE_ROOTS) or callee.startswith("<tensor>.")
                ):
                    found.add(f"{where}: {callee}")
    return found


@pytest.mark.parametrize("layout", _ALL)
def test_every_kernel_launch_in_modeling_is_an_entry_call(layout):
    """Every kernel launch in modeling.py is an OpWrapper call, as in the deepseek_v3 target: no
    ``torch.ops``, no DeepGEMM, no family kernel import, no matrix product, norm, activation,
    softmax or Top-K outside an entry."""
    path = _target_dir(layout) / "modeling.py"
    tree = _parse(path)
    _check("entry-calls", layout, _launch_breaks(path, [tree]))


@pytest.mark.parametrize(
    "helper", sorted(_DSA_BACKEND_FILES + _DSA_METADATA_FILES + _BUILDER_FILES)
)
def test_every_kernel_launch_in_the_runtime_helpers_is_an_entry_call(helper):
    """On the DSA backend carrier and metadata (whole files) and the runtime builders (their
    module-level functions; the entries beside them call their launchers)."""
    path = _FAMILY / "ops" / helper
    if not path.is_file():
        _check("entry-calls", helper, {"missing"})
        return
    tree = _parse(path)
    if helper in _BUILDER_FILES:
        found = _launch_breaks(path, _builders(tree))
        found = {f for f in found if not f.startswith("imports")}
    else:
        found = _launch_breaks(path, [tree])
    _check("entry-calls", helper, found)


def _builders(tree: ast.Module) -> list[ast.FunctionDef]:
    """The module-level functions of a builder module that no entry reaches: the builders. An
    entry's ``__call__``, ``reference`` and ``is_valid`` and what they call are the entry's."""
    top = {n.name: n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    reached: set[str] = set()
    todo = [
        m
        for c in _classes(tree).values()
        if any(b.split(".")[-1] == "OpWrapper" for b in _base_names(c))
        for m in _methods(c).values()
    ]
    while todo:
        fn = todo.pop()
        for node in ast.walk(fn):
            if isinstance(node, ast.Name) and node.id in top and node.id not in reached:
                reached.add(node.id)
                todo.append(top[node.id])
    return [fn for name, fn in top.items() if name not in reached and "reference" not in name]


@pytest.mark.parametrize("layout", _ALL)
def test_the_indexer_is_position_free(layout):
    """The checkpoint's DSA indexer has no RoPE (``index_rope_dim`` 0): the target states it, and
    modeling.py builds no indexer rope table and calls no rotary embedding."""
    tree = _modeling(layout)
    table = None
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "_K31_DSA" for t in node.targets
        ):
            table = ast.literal_eval(
                node.value.args[0] if isinstance(node.value, ast.Call) else node.value
            )
    found = set()
    if table is None or table.get("index_rope_dim") != 0:
        found.add(f"_K31_DSA does not state index_rope_dim 0: {table}")
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    names |= {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    for banned in ("RotaryEmbedding", "rubin_indexer_rope", "idx_rope", "indexer_rope"):
        if banned in names:
            found.add(f"modeling.py names {banned}")
    _check("indexer-rope", layout, found)


# --- per-step bodies ---


def _per_step_functions(tree: ast.Module) -> dict[str, ast.FunctionDef]:
    """Every function a step runs: the forwards and what they reach by name in modeling.py.

    In the default target a root also matches its layout copies (``draft_step_tep8``), the
    shell's forward without a drafter (``forward_logits``) is one, and so is every function a
    ``_BLOCKS`` row binds to a per-step field and the listed methods of a class it binds to a
    class field: the targets call those through the bound row, not by name."""
    core, shell = _core_and_shell(tree)
    cls = _classes(tree)
    top = {n.name: n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    methods = {}
    for c in cls.values():
        for m, fn in _methods(c).items():
            methods.setdefault(m, []).append((c.name, fn))
    rows = block_rows(tree)
    roots = list(_STEP_ROOTS)
    if rows:
        roots.append(("{shell}", "forward_logits"))
    todo = []
    for owner, name in roots:
        owner = {"{core}": core.name, "{shell}": shell.name}.get(owner, owner)
        if owner is None:
            todo += [(n, fn) for n, fn in top.items() if _base_name(n) == name]
        else:
            for c in cls.values():
                if (c.name == owner or _base_name(c.name) == owner) and name in _methods(c):
                    todo.append((f"{c.name}.{name}", _methods(c)[name]))
    for key, row in sorted(rows.items()):
        for field, name in row.items():
            if field in _STEP_FIELDS and name in top:
                todo.append((name, top[name]))
            elif field in _STEP_CLASS_FIELDS and name in cls:
                for m in _STEP_CLASS_FIELDS[field]:
                    if m in _methods(cls[name]):
                        todo.append((f"{name}.{m}", _methods(cls[name])[m]))
    seen: dict[str, ast.FunctionDef] = {}
    while todo:
        key, fn = todo.pop()
        if key in seen:
            continue
        seen[key] = fn
        for node in ast.walk(fn):
            if not isinstance(node, ast.Call):
                continue
            if isinstance(node.func, ast.Name) and node.func.id in top:
                todo.append((node.func.id, top[node.func.id]))
            elif isinstance(node.func, ast.Attribute):
                name = node.func.attr
                if name in _NOT_PER_STEP:
                    continue
                for owner, m in methods.get(name, ()):
                    if owner not in (core.name, shell.name) or name in (
                        "forward",
                        "_select_target",
                    ):
                        todo.append((f"{owner}.{name}", m))
    return seen


def _step_body_breaks(tree: ast.Module) -> set[str]:
    """Reads of a mapping, the environment or a launch-constant table (the default target's
    layout copies of the tables included) in what a step runs."""
    found = set()
    for key, fn in _per_step_functions(tree).items():
        text = ast.unparse(fn)
        for needle in _STEP_NEEDLES:
            if needle in text:
                found.add(f"{key} reads {needle}")
        for node in ast.walk(fn):
            if isinstance(node, ast.Name) and _is_constant_table(node.id):
                found.add(f"{key} reads {node.id}")
    return found


@pytest.mark.parametrize("layout", _ALL)
def test_per_step_bodies_take_no_feature_branch(layout):
    """what a step runs (everything a forward reaches) reads no mapping, no environment and no
    launch-constant table; as in the deepseek_v3 target, the layout is bound at construction, in
    closures and constants."""
    _check("step-bodies", layout, _step_body_breaks(_modeling(layout)))


def _nested_classes(tree: ast.Module) -> set[str]:
    """Classes defined inside a function (made per call, not once at import)."""
    found = set()
    for fn in ast.walk(tree):
        if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for node in ast.walk(fn):
                if isinstance(node, ast.ClassDef):
                    found.add(f"{fn.name} defines the class {node.name}")
    return found


@pytest.mark.parametrize("layout", _ALL)
def test_no_class_is_defined_inside_a_function(layout):
    """every class of modeling.py is a module-level class: the core, the targets, the shell and
    the drafter's classes exist once, at import, never per construction or per call."""
    _check("nested-classes", layout, _nested_classes(_modeling(layout)))


def _seed(tree: ast.Module, function: str, statement: str) -> ast.Module:
    """A copy of ``tree`` with ``statement`` first in ``function`` (a module-level function, or
    ``Class.method``)."""
    tree = copy.deepcopy(tree)
    owner, _, name = function.rpartition(".")
    fn = _methods(_classes(tree)[owner])[name] if owner else _top_level(tree)[function]
    fn.body.insert(0, ast.parse(statement).body[0])
    return ast.fix_missing_locations(tree)


def _seed_row(tree: ast.Module, layout: str | None) -> dict[str, str]:
    """A ``_BLOCKS`` row of ``layout``: a recipe set's row when the tree has one.
    No layout: the shell's methods, as ``{"forward_logits": "<shell>.forward_logits"}``."""
    if layout is None:
        shell = _core_and_shell(tree)[1].name
        return {m: f"{shell}.{m}" for m in _methods(_core_and_shell(tree)[1])}
    rows = block_rows(tree)
    for kv, ik in (("nvfp4", "fp8"), ("fp8", "fp4"), ("fp8", "fp8")):
        if (layout, kv, ik) in rows:
            return rows[(layout, kv, ik)]
    raise AssertionError(f"no {layout} row")


# The per-step rule, broken once where only the default target's roots reach: the shell's forward
# without a drafter, and a block function only a ``_BLOCKS`` row binds, reading a layout's copy of
# a constant table: (case, layout, field, statement, what the rule must report).
_SEEDS = (
    (
        "no-drafter-forward",
        None,
        "forward_logits",
        "rank = attn_metadata.mapping.rank",
        "forward_logits reads mapping.",
    ),
    (
        "constant-table",
        "tep8",
        "mla_generate",
        "chunk = _CONSTANTS_tep8['generation']",
        "reads _CONSTANTS_tep8",
    ),
)


# --- registration, imports, the drafter, stock calls, the environment ---


def _ours() -> list[Path]:
    """The files the Rubin targets add to the family: its kernels and entries, and the targets
    (the three specialized ones and the default one)."""
    files = _family_ops("**/rubin_*.py") + _family_ops("**/_rubin_*.py")
    files += _family_kernels("**/rubin_*.py") + _family_kernels("**/_rubin_*.py")
    # The two family kernels the targets add without the ``rubin_`` prefix.
    files += [
        _FAMILY / "kernels" / name
        for name in ("activation/situ_and_mul.py", "ssm/reset_recurrent_state_rows.py")
    ]
    for layout in _ALL:
        files += sorted(_target_dir(layout).rglob("*.py"))
    return files


_REGISTERS = frozenset(
    {"custom_op", "register_fake", "register_kernel", "impl_abstract", "Library"}
)


def _docstrings(tree: ast.Module) -> set[int]:
    """The ids of the docstring constants of a module and its classes and functions."""
    ids = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = node.body
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
                ids.add(id(body[0].value))
    return ids


def _strings(tree: ast.Module) -> list[str]:
    """String constants of the code, docstrings left out."""
    docs = _docstrings(tree)
    return [
        n.value
        for n in ast.walk(tree)
        if isinstance(n, ast.Constant) and isinstance(n.value, str) and id(n) not in docs
    ]


def test_our_files_register_no_torch_op():
    """a file the Rubin targets add uses no ``torch.library`` registration (``custom_op``,
    ``register_fake``, ``Library``) and embeds no ``TORCH_LIBRARY`` source: an entry's ``__call__``
    calls its launcher."""
    found = set()
    for path in _ours():
        tree = _parse(path)
        for module, name, _ in _imports(path):
            if module.startswith("torch.library") or (module == "torch" and name == "library"):
                found.add(f"{_rel(path)}: imports {module}.{name}")
        for node in ast.walk(tree):
            chain = _chain(node) if isinstance(node, (ast.Attribute, ast.Name)) else None
            if chain and (chain.startswith("torch.library") or chain.split(".")[-1] in _REGISTERS):
                found.add(f"{_rel(path)}: {chain}")
        for text in _strings(tree):
            for needle in ("TORCH_LIBRARY",):
                if needle in text:
                    found.add(f"{_rel(path)}: a string names {needle}")
    _check("op-registration", "family", found)


@pytest.mark.parametrize("layout", _ALL)
def test_a_target_imports_family_code_only_from_ops_and_single_files(layout):
    """family imports of modeling.py are ``ops/`` entries and helpers and single-file family
    modules (``config``); weights.py may add the family kernels for its transforms."""
    found = set()
    own = f"{_FAMILY_PKG}.{_TARGETS[layout]}"
    for product, roots in (("modeling.py", ("ops",)), ("weights.py", ("ops", "kernels"))):
        path = _target_dir(layout) / product
        for module, name, _ in _imports(path):
            full = f"{module}.{name}" if name else module
            if not full.startswith(_FAMILY_PKG + ".") or full.startswith(own + ".") or full == own:
                continue
            first = full[len(_FAMILY_PKG) + 1 :].split(".")[0]
            single = (_FAMILY / f"{first}.py").is_file() and not (_FAMILY / first).is_dir()
            if first not in roots and not single:
                found.add(f"{product} imports {full}")
    _check("family-imports", layout, found)


@pytest.mark.parametrize("layout", _ALL)
def test_the_drafter_lives_in_modeling(layout):
    """the drafter container, its worker and the draft step are modeling.py's; no engine
    drafter model is imported."""
    tree = _modeling(layout)
    found = set()
    names = set(_classes(tree)) | {
        n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    rows = block_rows(tree)
    if not rows:
        for name in ("DraftModel", "DFlashWorker", "draft_step"):
            if name not in names:
                found.add(f"no {name}")
    # The default target: each layout's rows bind a container and a worker of modeling.py, and
    # each layout has a draft step (its own copy or the shared one).
    for key, row in sorted(rows.items()):
        for field in ("DraftModel", "DFlashWorker"):
            if row.get(field) not in _classes(tree):
                found.add(f"{key}.{field} is not a class of modeling.py: {row.get(field)}")
    for layout_name in sorted({key[0] for key in rows}):
        if not {"draft_step", f"draft_step_{layout_name}"} & names:
            found.add(f"no draft step of {layout_name}")
    for module, name, _ in _imports(_target_dir(layout) / "modeling.py"):
        full = f"{module}.{name}" if name else module
        if any(d in full.split(".") for d in _DRAFTER_MODULES):
            found.add(f"imports {full}")
    if "install_k31_dflash2" in ast.unparse(tree):
        found.add("names install_k31_dflash2")
    _check("drafter", layout, found)


def _is_entry_module(module: str, layout: str) -> bool:
    roots = (
        f"{_MV2_PKG}.catalog",
        f"{_MV2_PKG}._native",
        f"{_FAMILY_PKG}.ops",
        f"{_FAMILY_PKG}.{_TARGETS[layout]}.ops",
    )
    return any(module == r or module.startswith(r + ".") for r in roots)


@pytest.mark.parametrize("layout", _ALL)
def test_uncertified_generic_calls_name_every_stock_compute_import(layout):
    """``UNCERTIFIED_GENERIC_CALLS`` is exactly the stock (non-entry) names modeling.py imports,
    minus the ones that run no computation (``_NON_COMPUTE``); this test is the list's check."""
    tree = _modeling(layout)
    declared = None
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "UNCERTIFIED_GENERIC_CALLS" for t in node.targets
        ):
            declared = set(ast.literal_eval(node.value))
    assert declared is not None, "modeling.py declares no UNCERTIFIED_GENERIC_CALLS"
    imported = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.ImportFrom)
            and node.level == 0
            and node.module
            and node.module.split(".")[0] == "tensorrt_llm"
            and not _is_entry_module(node.module, layout)
        ):
            imported.update(
                f"{node.module}.{a.name}" for a in node.names if a.name not in _NON_COMPUTE
            )
    found = {f"undeclared {n}" for n in imported - declared}
    found |= {f"declared, not imported {n}" for n in declared - imported}
    _check("uncertified", layout, found)


def _env_reads(tree: ast.Module) -> set[str]:
    """Reads of ``environ`` / ``getenv`` off a module name (``os``, or ``os`` imported under
    another name) and imports of them from ``os``."""
    found = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Attribute)
            and node.attr in ("environ", "getenv", "putenv", "unsetenv")
            and isinstance(node.value, ast.Name)
        ):
            found.add(f"{node.value.id}.{node.attr}")
        if isinstance(node, ast.ImportFrom) and node.module == "os":
            found.update(f"from os import {a.name}" for a in node.names if "env" in a.name)
    return found


def test_no_environment_read_on_the_flat_path():
    """the family's ``rubin_*`` kernels and entries, the runtime helpers and the targets read
    no environment variable (``get_env_enable_pdl()``, the engine-wide PDL switch, is allowed),
    import no engine ``*_ENV`` name and name no lever variable outside a docstring: each lever is
    a constant of the target."""
    files = set(_ours()) | {
        _FAMILY / "ops" / h for h in _HELPERS if (_FAMILY / "ops" / h).is_file()
    }
    found = set()
    for path in sorted(files):
        tree = _parse(path)
        for read in _env_reads(tree):
            found.add(f"{_rel(path)} reads the environment ({read})")
        for text in _strings(tree):
            for name in _LEVER_ENV:
                if re.search(rf"\b{name}\b", text):
                    found.add(f"{_rel(path)} names {name}")
        for module, name, _ in _imports(path):
            if name and name.endswith("_ENV"):
                found.add(f"{_rel(path)} imports {module}.{name}")
    _check("environment", "family", found)


# --- the family's ops/ files ---


def test_family_ops_files_are_entries_or_named_helpers():
    """each ``ops/`` file the Rubin targets add holds an entry (an OpWrapper subclass and its
    module-level instance) or is a named runtime helper; a private ``_rubin_*`` support module
    holds functions only. An entry's public methods are ``reference`` and ``is_valid`` (and its
    ``__call__``)."""
    found = set()
    for path in _family_ops("**/rubin_*.py") + _family_ops("**/_rubin_*.py"):
        key = path.relative_to(_FAMILY / "ops").as_posix()
        tree = _parse(path)
        wrappers = [
            c
            for c in _classes(tree).values()
            if any(b.split(".")[-1] == "OpWrapper" for b in _base_names(c))
        ]
        instances = {
            ast.unparse(n.value.func)
            for n in tree.body
            if isinstance(n, ast.Assign) and isinstance(n.value, ast.Call)
        }
        if path.name.startswith("_"):
            if _classes(tree):
                found.add(f"{key}: a support module with classes {sorted(_classes(tree))}")
        elif not wrappers and key not in _HELPERS:
            found.add(f"{key}: neither an entry nor a named runtime helper")
        for c in wrappers:
            if c.name not in instances:
                found.add(f"{key}: {c.name} has no module-level instance")
            public = {m for m in _methods(c) if not m.startswith("_")} - {"reference", "is_valid"}
            if public:
                found.add(f"{key}: {c.name} has public methods {sorted(public)}")
    _check("ops-files", "family", found)


def test_the_dsa_metadata_names_its_overrides():
    """each DSA metadata module's own test lists the methods its classes override and resolves
    every other one to the class below; it ships with the metadata."""
    found = set()
    for module, test in sorted(_METADATA_OVERRIDE_TESTS.items()):
        if not (_TESTS_ROOT / _KT / test).is_file():
            found.add(f"{module}: no {test}")
    _check("metadata-overrides", "family", found)


# --- the SM107 test list ---


def _listed(stage: str) -> set[str]:
    import yaml

    data = yaml.safe_load((_TESTS_ROOT / f"integration/test_lists/test-db/{stage}.yml").read_text())
    entries = set()
    for blocks in data.values():
        for block in blocks if isinstance(blocks, list) else ():
            for entry in block.get("tests") or ():
                entries.add(entry.split()[0].split("::")[0])
    return entries


def _module_mark(path: Path) -> str:
    for node in _parse(path).body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "pytestmark" for t in node.targets
        ):
            return ast.unparse(node.value)
    return ""


def _no_ci_stage() -> set[str]:
    claims = _TESTS_ROOT / "unittest/_torch/modeling_v2/test_modeling_v2_claims.py"
    for node in _parse(claims).body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "_NO_CI_STAGE" for t in node.targets
        ):
            return {ast.literal_eval(k) for k in node.value.keys}
    return set()


def test_the_sm107_stage_lists_each_single_gpu_rubin_test():
    """Each GPU ``test_rubin_*`` file of the family's tests, at their top level and under
    ``kernels/`` and ``ops/``, has its own ``l0_gr100`` entry (a directory entry would run
    uncertified family tests); a ``cpu_only`` file is l0_cpu's, whose directory entry collects
    only files marked ``cpu_only``. The multi-rank tests are under ``rubin_multirank/``, which the
    claims' ``_NO_CI_STAGE`` holds until a multi-GPU SM107 stage exists."""
    listed = _listed("l0_gr100")
    found = set()
    paths = sorted((_TESTS_ROOT / _KT).glob("test_rubin_*.py"))
    for sub in ("kernels", "ops"):
        paths += sorted((_TESTS_ROOT / _KT / sub).rglob("test_rubin_*.py"))
    for path in paths:
        entry = path.relative_to(_TESTS_ROOT).as_posix()
        if "cpu_only" in _module_mark(path):
            continue
        if entry not in listed:
            found.add(f"not in l0_gr100: {entry}")
    if f"{_KT}/rubin_multirank" not in _no_ci_stage():
        found.add("_NO_CI_STAGE lacks rubin_multirank")
    _check("test-lists", "l0_gr100", found)
