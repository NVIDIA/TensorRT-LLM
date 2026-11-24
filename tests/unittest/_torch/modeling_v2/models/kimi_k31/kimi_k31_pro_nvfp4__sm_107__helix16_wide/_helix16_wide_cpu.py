# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU helpers of the ``kimi_k31_pro_nvfp4__sm_107__helix16_wide`` tests (no GPU, no built engine):
``weights.py`` and the family's FC12 reference loaded by path, ``modeling.py``'s objects run as
extracted from its source (``source_objects``) with the caller's stand-ins."""

from __future__ import annotations

import ast
import enum
import importlib
import importlib.util
import sys
import types
from contextlib import contextmanager
from pathlib import Path

import torch

FAMILY_PKG = "tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31"
TARGET = "kimi_k31_pro_nvfp4__sm_107__helix16_wide"


def package_root() -> Path:
    """``tensorrt_llm/`` of the tree under test."""
    spec = importlib.util.find_spec("tensorrt_llm")
    return Path(spec.submodule_search_locations[0]).resolve()


def family_dir() -> Path:
    return package_root() / "_torch/_experimental/modeling_v2/models/kimi_k31"


def modeling_path() -> Path:
    return family_dir() / TARGET / "modeling.py"


def _defined(node: ast.stmt) -> set:
    """The top-level names a statement of a module defines (functions, classes, assignments)."""
    if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
        return {node.name}
    if isinstance(node, ast.Assign):
        names = set()
        for target in node.targets:
            elts = target.elts if isinstance(target, ast.Tuple) else [target]
            names.update(e.id for e in elts if isinstance(e, ast.Name))
        return names
    return set()


def source_objects(path: Path, names, namespace: dict) -> dict:
    """The top-level definitions ``names`` of ``path`` (parsed, never imported), run in source
    order in ``namespace`` (the caller's stand-ins) under ``from __future__ import annotations``;
    returns ``namespace``. A name the source does not define raises."""
    tree = ast.parse(path.read_text(), str(path))
    wanted = set(names)
    picked = [node for node in tree.body if _defined(node) & wanted]
    missing = wanted - set().union(*(_defined(node) for node in picked))
    if missing:
        raise LookupError(f"{path.name} defines no {sorted(missing)}")
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    module = ast.fix_missing_locations(ast.Module(body=[future, *picked], type_ignores=[]))
    namespace.setdefault("__name__", f"_k31_helix16_wide_source_{path.stem}")
    exec(compile(module, str(path), "exec"), namespace)
    return namespace


def function_node(path: Path, qualname: str) -> ast.FunctionDef:
    """The ``def`` of ``qualname`` (``f`` or ``Class.method``) in ``path``'s source."""
    scope = ast.parse(path.read_text(), str(path))
    for part in qualname.split("."):
        scope = next(
            node
            for node in scope.body
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name == part
        )
    return scope


def literal(path: Path, name: str):
    """The literal a top-level assignment of ``path`` gives ``name``."""
    for node in ast.parse(path.read_text(), str(path)).body:
        if isinstance(node, ast.Assign) and name in _defined(node):
            return ast.literal_eval(node.value)
    raise LookupError(f"{path.name} assigns no {name}")


def load_weights_module():
    """The target's ``weights.py`` as a module of its own, registered under a private name."""
    path = family_dir() / TARGET / "weights.py"
    name = "_k31_helix16_wide_weights_under_test"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        del sys.modules[name]
        raise
    return module


@contextmanager
def tiny_experts(module, *, experts=16, inter=256, latent=128):
    """``module``'s routed-expert dimensions shrunk (the TP2 x EP8 split unchanged)."""
    names = ("EXPERTS", "INTER", "HALF", "LOCAL", "LATENT")
    saved = {n: getattr(module, n) for n in names}
    values = dict(
        EXPERTS=experts,
        INTER=inter,
        HALF=inter // module.MOE_TP,
        LOCAL=experts // module.MOE_EP,
        LATENT=latent,
    )
    for n, v in values.items():
        setattr(module, n, v)
    try:
        yield values
    finally:
        for n, v in saved.items():
            setattr(module, n, v)


def e4m3_bytes(shape, generator) -> torch.Tensor:
    """Positive finite E4M3 block scales (as uint8): exponents 120..135 of a random mantissa."""
    values = torch.rand(shape, generator=generator) * 1.5 + 0.25
    return values.to(torch.float8_e4m3fn).view(torch.uint8)


def checkpoint(module, layer: int, *, prefix: str = "language_model.", seed: int = 0, scale=1.0):
    """A ModelOpt-shaped routed-expert checkpoint of one layer at ``module``'s dimensions."""
    g = torch.Generator().manual_seed(seed)
    e, inter, latent, sf = module.EXPERTS, module.INTER, module.LATENT, module.SF_VEC
    out = {}
    for expert in range(e):
        for w, shape in (("w1", (inter, latent)), ("w3", (inter, latent)), ("w2", (latent, inter))):
            key = lambda kind: module.expert_key(prefix, layer, expert, w, kind)  # noqa: E731
            rows, cols = shape
            out[key("weight")] = torch.randint(0, 256, (rows, cols // 2), generator=g).to(
                torch.uint8
            )
            out[key("weight_scale")] = e4m3_bytes((rows, cols // sf), g).view(torch.float8_e4m3fn)
            ws2 = torch.rand((), generator=g) * 0.01 + 0.001
            out[key("weight_scale_2")] = ws2.to(torch.float32)
            out[key("input_scale")] = torch.tensor(scale, dtype=torch.float32)
    # The gate and up global scales of one expert match (the export's).
    for expert in range(e):
        up = module.expert_key(prefix, layer, expert, "w3", "weight_scale_2")
        gate = module.expert_key(prefix, layer, expert, "w1", "weight_scale_2")
        out[up] = out[gate].clone()
    return out


def unswizzle(s: torch.Tensor) -> torch.Tensor:
    """Row-major ``[M, C]`` of 128x4-swizzled block-scale bytes."""
    m, c = s.shape
    return s.reshape(m // 128, c // 4, 32, 4, 4).permute(0, 3, 2, 1, 4).reshape(m, c)


def ungate_up16(rows: torch.Tensor):
    """``(gate, up)`` of FC1 rows in 16 gate / 16 up order."""
    two_r, c = rows.shape
    pairs = rows.reshape(two_r // 32, 2, 16, c)
    return pairs[:, 0].reshape(two_r // 2, c), pairs[:, 1].reshape(two_r // 2, c)


class _ActivationType(enum.IntEnum):
    """The values ``rubin_fc12`` reads of ``tensorrt_llm._torch.utils.ActivationType``."""

    Swiglu = 5
    SiTu = 9


@contextmanager
def fc12_reference_module():
    """``ops/moe/rubin_fc12.py`` loaded on CPU (launcher and ``ActivationType`` stood in), yielded;
    afterwards the family's modules it added and the stand-ins are removed."""
    before = set(sys.modules)
    saved = {}

    def put(name, module):
        saved.setdefault(name, sys.modules.get(name))
        sys.modules[name] = module

    root = family_dir()
    bare = (
        ("tensorrt_llm._torch._experimental.modeling_v2.models", root.parent),
        (FAMILY_PKG, root),
        (f"{FAMILY_PKG}.ops", root / "ops"),
        (f"{FAMILY_PKG}.ops.moe", root / "ops/moe"),
        (f"{FAMILY_PKG}.kernels", root / "kernels"),
        (f"{FAMILY_PKG}.kernels.moe", root / "kernels/moe"),
    )
    try:
        for name, path in bare:
            if name not in sys.modules:
                package = types.ModuleType(name)
                package.__path__ = [str(path)]
                put(name, package)
        launch = types.ModuleType(f"{FAMILY_PKG}.kernels.moe.rubin_fc12_launch")
        for attr in ("DIE_CLAIMS", "LOOKAHEADS", "MMA_N", "TUNER_KEY"):
            setattr(launch, attr, ())
        for attr in ("check_levers", "fc12_tactics", "launch_rubin_fc12"):
            setattr(launch, attr, None)
        put(launch.__name__, launch)
        try:
            importlib.import_module("tensorrt_llm._torch.utils")
        except Exception:  # noqa: BLE001 (the CPU host lacks the engine's imports)
            utils = types.ModuleType("tensorrt_llm._torch.utils")
            utils.ActivationType = _ActivationType
            put(utils.__name__, utils)
        name = f"{FAMILY_PKG}.ops.moe.rubin_fc12"
        spec = importlib.util.spec_from_file_location(name, root / "ops/moe/rubin_fc12.py")
        module = importlib.util.module_from_spec(spec)
        put(name, module)
        spec.loader.exec_module(module)
        yield module
    finally:
        for name in set(sys.modules) - before:
            if name == FAMILY_PKG or name.startswith(FAMILY_PKG + ".") or name in saved:
                sys.modules.pop(name, None)
        for name, module in saved.items():
            if module is not None:
                sys.modules[name] = module
            else:
                sys.modules.pop(name, None)
