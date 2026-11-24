# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""dep16's moe_back_fused_norm lever (CPU, from source).

A generation server's MegaMoE launches leave their combine plane unreduced (the generation
kernel's ``defer_topk_reduce``, ``kernels/moe/megamoe_token_path.py``), and one kernel reduces it
and applies the latent RMSNorm before latent_up (``rubin_moe_back_norm``, whose test holds its bits
against the engine's TopkReduce and FlashInfer's RMSNorm). Checked here:

1. the generation constant is on, a context server has none;
2. the runtime build binds the deferral before the MegaMoE workspace and collects the buckets'
   planes after it;
3. the product's chunk code, over recording stand-ins: with the lever on, a launch's rows are
   normalized from its bucket's plane with the routing weights that launch combined them with (the
   bucket's on the bucket path, the staged ones on the staging path); a bucket without a plane
   (another kernel, whose launch reduces) and the lever off take FlashInfer's RMSNorm of the rows;
4. the binding: off leaves the class alone, on sets the constant and installs the class, and a
   process cannot turn it off again; ``combine_plane`` views the recorded offset or returns None.
"""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import pytest
import torch

pytestmark = pytest.mark.cpu_only

_PKG_ROOT = Path(importlib.util.find_spec("tensorrt_llm").submodule_search_locations[0]).resolve()
_FAMILY = _PKG_ROOT / "_torch/_experimental/modeling_v2/models/kimi_k31"
_TARGET = _FAMILY / "kimi_k31_pro_nvfp4__sm_107__dep16"
_DEP16 = _TARGET / "modeling.py"
_TOKEN_PATH = _TARGET / "kernels/moe/megamoe_token_path.py"
_FUNCS = (
    "_moe_megamoe",
    "_moe_megamoe_chunk",
    "_moe_bucket_rows",
    "_moe_bucket_route",
    "_moe_bucket_quantize",
    "_moe_bucket_launch",
    "_moe_split",
    "_moe_chunk_plane",
    "_moe_routed_norm",
)
_LATENT, _EXPERTS, _TOP_K = 64, 32, 16


def _tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"))


def _function(path: Path, name: str) -> ast.FunctionDef:
    (fn,) = [n for n in ast.walk(_tree(path)) if isinstance(n, ast.FunctionDef) and n.name == name]
    return fn


def _calls(node: ast.AST, name: str) -> list:
    return [n for n in ast.walk(node) if isinstance(n, ast.Call) and ast.unparse(n.func) == name]


def _run(source: Path, names, ns: dict) -> dict:
    nodes = [n for n in _tree(source).body if getattr(n, "name", None) in set(names)]
    assert sorted(n.name for n in nodes) == sorted(names)
    exec(compile(ast.fix_missing_locations(ast.Module(nodes, [])), source.name, "exec"), ns)
    return ns


def _constants() -> dict:
    ns = {"MappingProxyType": MappingProxyType}
    for node in _tree(_DEP16).body:
        if isinstance(node, ast.Assign) and [getattr(t, "id", None) for t in node.targets] == [
            "_CONSTANTS"
        ]:
            exec(compile(ast.Module([node], []), _DEP16.name, "exec"), ns)
            return ns["_CONSTANTS"]
    raise KeyError("_CONSTANTS")


def test_a_generation_server_runs_the_lever_and_a_context_server_has_none():
    consts = _constants()
    assert consts["generation"]["moe_back_fused_norm"] is True
    assert "moe_back_fused_norm" not in consts["context"]


def test_the_runtime_defers_before_the_workspace_and_collects_the_planes_after_it():
    build = _function(_DEP16, "_build_runtime")
    (bind,) = _calls(build, "bind_deferred_topk_reduce")
    assert [ast.unparse(a) for a in bind.args] == ["c['moe_back_fused_norm']"]
    (workspace,) = _calls(build, "rubin_megamoe_workspace")
    (planes,) = _calls(build, "_moe_back_planes")
    assert bind.lineno < workspace.lineno < planes.lineno
    (field,) = [k for k in _calls(build, "_Rt")[0].keywords if k.arg == "moe_back_planes"]
    assert ast.unparse(field.value) == "moe_back_planes"


def _chunk_code(log: list) -> dict:
    """The product's chunk code over recording stand-ins (buckets of 32 / 64 / 128 rows)."""
    buckets = {
        t: SimpleNamespace(
            max_tokens=t,
            topk_idx=torch.zeros(t, _TOP_K, dtype=torch.int32),
            topk_weights=torch.full((t, _TOP_K), float(t)),
            activation=torch.zeros(t, _LATENT // 2, dtype=torch.uint8),
            activation_sf=torch.zeros(t, _LATENT // 16, dtype=torch.uint8),
        )
        for t in (32, 64, 128)
    }

    def megamoe_bucket(ws, chunk_max, rows):
        return buckets[min(t for t in buckets if t >= chunk_max)]

    def rubin_megamoe_in_bucket(b, *weights_ws_rows):
        rows = weights_ws_rows[-1]
        log.append(("in_bucket", b.max_tokens, rows))
        return torch.full((rows, _LATENT), -1.0)

    def rubin_megamoe(x4, sf, ids, weights, *rest):
        log.append(("staged", rest[-2], rest[-1]))
        return torch.full((rest[-1], _LATENT), -1.0)

    def rubin_moe_back_norm(plane, scores, weight, eps, out):
        log.append(("fused", plane, scores))
        out.fill_(2.0)
        return out

    def flashinfer_rmsnorm(x, weight, eps):
        log.append(("flashinfer", x.shape[0]))
        return torch.full_like(x, 3.0)

    def _moe_route(layer, logits):
        rows = logits.shape[0]
        return torch.zeros(rows, _TOP_K, dtype=torch.int32), torch.full((rows, _TOP_K), 7.0)

    def _moe_quantize(layer, latent_in):
        rows = latent_in.shape[0]
        return (
            torch.zeros(rows, _LATENT // 2, dtype=torch.uint8),
            torch.zeros(rows, _LATENT // 16, dtype=torch.uint8),
        )

    ns = dict(
        torch=torch,
        megamoe_bucket=megamoe_bucket,
        rubin_route_topk_out=lambda *a, **k: None,
        rubin_megamoe_quantize=lambda *a, **k: None,
        rubin_megamoe_in_bucket=rubin_megamoe_in_bucket,
        rubin_megamoe=rubin_megamoe,
        rubin_moe_back_norm=rubin_moe_back_norm,
        flashinfer_rmsnorm=flashinfer_rmsnorm,
        _moe_route=_moe_route,
        _moe_quantize=_moe_quantize,
        _MOE_ROUTE_MAX_M=128,
        _MOE_ROUTE_SKIP_M=frozenset({32}),
        _MOE_SCALE=1.0,
        _MOE_TOP_K=_TOP_K,
        _MOE_SF_VEC=16,
        _MOE_EPS=1e-5,
    )
    return _run(_DEP16, _FUNCS, ns), buckets


def _step(rows: int, planes, counts=None):
    log = []
    ns, buckets = _chunk_code(log)
    layer = SimpleNamespace(
        router_bias=None, fc31_in_scale=None, fc1_w=None, fc1_s=None, fc2_w=None, fc2_s=None,
        fc31_alpha=None, fc2_alpha=None, fc1_norm_const=None, latent_norm=None,
    )  # fmt: skip
    plane_of = None if planes is None else {t: torch.zeros(t, _TOP_K, _LATENT) for t in planes}
    rt = SimpleNamespace(
        megamoe=SimpleNamespace(ep_rank=0, chunk_tokens=128), moe_back_planes=plane_of
    )
    st = SimpleNamespace(all_rank_num_tokens=counts or [rows] * 16)
    out = ns["_moe_megamoe"](layer, torch.zeros(rows, _EXPERTS), torch.zeros(rows, _LATENT), st, rt)
    return out, log, buckets, plane_of


def test_the_bucket_path_normalizes_from_the_plane_with_the_buckets_weights():
    out, log, buckets, planes = _step(96, (32, 64, 128))
    (launch, (kind, plane, scores)) = log
    assert launch == ("in_bucket", 128, 96) and kind == "fused"
    assert plane is planes[128] and scores is buckets[128].topk_weights
    assert tuple(out.shape) == (96, _LATENT) and bool((out == 2.0).all())


def test_the_staging_path_normalizes_with_the_weights_it_staged():
    out, log, _, planes = _step(32, (32, 64, 128))
    (launch, (kind, plane, scores)) = log
    assert launch == ("staged", 32, 32) and kind == "fused"
    assert plane is planes[32] and bool((scores == 7.0).all()) and scores.shape == (32, _TOP_K)


def test_a_bucket_whose_launch_reduces_and_the_lever_off_take_flashinfer():
    _, log, _, _ = _step(96, (32, 64))  # bucket 128's kernel reduces
    assert log == [("in_bucket", 128, 96), ("flashinfer", 96)]
    out, log, _, _ = _step(96, None)
    assert log == [("in_bucket", 128, 96), ("flashinfer", 96)] and bool((out == 3.0).all())


def test_a_step_of_several_chunks_normalizes_each_chunk():
    out, log, _, _ = _step(13, (32, 64, 128), counts=[13] + [200] * 15)
    launches = [e for e in log if e[0] in ("in_bucket", "staged")]
    fused = [e for e in log if e[0] == "fused"]
    assert len(launches) == len(fused) == 2 and tuple(out.shape) == (13, _LATENT)
    assert bool((out == 2.0).all())


class _Kernel:
    defer_topk_reduce = False
    combine_plane_offsets: dict = {}


def _binding(installed: list) -> dict:
    ns = {
        "torch": torch,
        "RubinTokenPathMegaMoE": _Kernel,
        "_install_generation_kernel": installed.append,
    }
    return _run(_TOKEN_PATH, ("bind_deferred_topk_reduce", "combine_plane", "_no_topk_reduce"), ns)


def test_the_binding_turns_the_deferral_on_once():
    installed = []
    ns = _binding(installed)
    _Kernel.defer_topk_reduce = False
    ns["bind_deferred_topk_reduce"](False)
    assert installed == [] and not _Kernel.defer_topk_reduce
    ns["bind_deferred_topk_reduce"](True)
    assert installed == [_Kernel] and _Kernel.defer_topk_reduce
    with pytest.raises(ValueError, match="already runs"):
        ns["bind_deferred_topk_reduce"](False)
    assert ns["_no_topk_reduce"](1, None, 2, 3, 4) is None
    _Kernel.defer_topk_reduce = False


def test_the_plane_is_a_view_at_the_recorded_offset():
    ns = _binding([])
    ws = torch.zeros(4096 + 2 * 4 * 8 * 2, dtype=torch.uint8)
    _Kernel.combine_plane_offsets = {(16, 2, 4, 8): 4096}
    plane = ns["combine_plane"](ws, 16, 2, 4, 8)
    assert plane.shape == (2, 4, 8) and plane.dtype == torch.bfloat16
    assert plane.data_ptr() == ws.data_ptr() + 4096
    assert ns["combine_plane"](ws, 16, 4, 4, 8) is None
    _Kernel.combine_plane_offsets = {(16, 2, 4, 8): ws.numel() - 16}
    with pytest.raises(ValueError, match="outside"):
        ns["combine_plane"](ws, 16, 2, 4, 8)
    _Kernel.combine_plane_offsets = {}
