# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""dep16's start-up compile of the context server's long-chunk prefill Top-K (CPU, from source).

The core's post-load step launches the target's prefill Top-K once per (rows, row stride) of
``_GVR_PREFILL_PRECOMPILE``, so the GVR engine's variants past its bitmap limit compile before the
first request instead of at a step's first long chunk. With stand-ins for the entry, the logger
and the GPU tensors:

1. the shapes cover the engine's three prefill tiers once each, every row stride past its tier's
   bitmap limit (8 x 32 x BLK), the tier-0 one in the envelope bucket long chunks use; the tier
   rules, the routing and the limit are the engine's own source, executed or read here;
2. each launch is the entry on FP32 zero logits with full-row windows, Top-2048 int32 rows and
   the deterministic order the indexer's prefill Top-K runs;
3. a context server launches the three and logs one line per process; a generation server
   launches and logs nothing;
4. out of memory on one shape skips it with a warning and still launches the others;
5. the core's post-load step calls it with the core's role, once the targets are built.
"""

from __future__ import annotations

import ast
import importlib.util
import math
import time
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import pytest
import torch

pytestmark = pytest.mark.cpu_only

_PKG_ROOT = Path(importlib.util.find_spec("tensorrt_llm").submodule_search_locations[0]).resolve()
_TOP_K_DIR = _PKG_ROOT / "_torch/kernels/blackwell/top_k"
_HOST = _TOP_K_DIR / "gvr_topk_decode_self_sampling_host.py"
_KERNEL = _TOP_K_DIR / "gvr_topk_decode_self_sampling.py"
_DEP16 = (
    _PKG_ROOT
    / "_torch/_experimental/modeling_v2/models/kimi_k31/kimi_k31_pro_nvfp4__sm_107__dep16"
    / "modeling.py"
)
_TOP_K = 2048


def _engine() -> dict:
    """The engine's prefill tier, bucket and routing functions, run from its source (the module
    needs the GPU stack to import)."""
    wanted = {
        "SNB",
        "_PREFILL_TIER_ROWS",
        "_PREFILL_T1_MARGIN",
        "_PREFILL_T1_MAX",
        "_prefill_scpb_tier1",
        "_prefill_tier",
        "_prefill_bucket",
        "route_streaming",
    }
    nodes = []
    for node in ast.parse(_HOST.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.FunctionDef):
            names = {node.name}
        elif isinstance(node, ast.Assign):
            names = {t.id for t in node.targets if isinstance(t, ast.Name)}
        else:
            continue
        if names & wanted:
            nodes.append(node)
            wanted -= names
    assert not wanted, f"the engine no longer defines {sorted(wanted)}"
    ns: dict = {"math": math}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(_HOST), "exec"), ns)
    return ns


def _bitmap_limit(blk: int) -> int:
    """The deterministic dispatch's bitmap limit for a kernel of ``blk`` threads, as the engine
    writes it: a row stride above it takes the bitmap-free variant."""
    src = _KERNEL.read_text(encoding="utf-8")
    assert "nmax = 8 * 32 * int(tpl[0])" in src and "lbm = nn <= nmax" in src, (
        "the engine's bitmap limit moved: re-derive _GVR_PREFILL_PRECOMPILE"
    )
    return 8 * 32 * blk


class _Log:
    def __init__(self) -> None:
        self.info: list[str] = []
        self.warn: list[str] = []
        self._keys: set = set()

    def info_once(self, msg: str, *, key) -> None:
        if key not in self._keys:
            self._keys.add(key)
            self.info.append(msg)

    def warning(self, msg: str) -> None:
        self.warn.append(msg)


class _Oom(RuntimeError):
    pass


def _torch(fail_rows: int | None = None) -> SimpleNamespace:
    """CPU tensors for the ``device="cuda"`` allocations; ``fail_rows`` raises out of memory on
    the logits of that many rows."""

    def zeros(shape, dtype, device):
        assert device == "cuda"
        if fail_rows is not None and len(shape) == 2 and shape[0] == fail_rows:
            raise _Oom("CUDA out of memory")
        return torch.zeros(shape, dtype=dtype)

    def full(shape, value, dtype, device):
        assert device == "cuda"
        return torch.full(shape, value, dtype=dtype)

    def empty(shape, dtype, device):
        assert device == "cuda"
        return torch.empty(shape, dtype=dtype)

    cuda = SimpleNamespace(OutOfMemoryError=_Oom, synchronize=lambda: None)
    return SimpleNamespace(
        zeros=zeros, full=full, empty=empty, float32=torch.float32, int32=torch.int32, cuda=cuda
    )


def _definitions(names) -> list:
    wanted, nodes = set(names), []
    for node in ast.parse(_DEP16.read_text()).body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            found = {node.name}
        elif isinstance(node, ast.Assign):
            found = {t.id for t in node.targets if isinstance(t, ast.Name)}
        else:
            continue
        if found & wanted:
            nodes.append(node)
            wanted -= found
    assert not wanted, f"the product no longer defines {sorted(wanted)}"
    return nodes


def _precompile(fail_rows: int | None = None):
    """dep16's ``_GVR_PREFILL_PRECOMPILE`` and ``_gvr_prefill_precompile`` over stand-ins; returns
    (shapes, function, launches, log)."""
    launches: list[dict] = []

    def rubin_gvr_topk_prefill(logits, starts, ends, out, deterministic):
        launches.append(
            dict(
                rows=logits.shape[0],
                width=logits.shape[1],
                stride=logits.stride(0),
                logits=logits,
                starts=starts,
                ends=ends,
                out=out,
                deterministic=deterministic,
            )
        )
        return out

    log = _Log()
    ns = {
        "torch": _torch(fail_rows),
        "time": time,
        "logger": log,
        "MappingProxyType": MappingProxyType,
        "rubin_gvr_topk_prefill": rubin_gvr_topk_prefill,
        "_TARGET": "kimi_k31_pro_nvfp4__sm_107__dep16",
    }
    names = (
        "_K31_DSA",
        "_IDX_GVR_DETERMINISTIC",
        "_GVR_PREFILL_PRECOMPILE",
        "_gvr_prefill_precompile",
    )
    module = ast.Module(body=_definitions(names), type_ignores=[])
    exec(compile(module, str(_DEP16), "exec"), ns)
    return ns["_GVR_PREFILL_PRECOMPILE"], ns["_gvr_prefill_precompile"], launches, log


def test_shapes_pass_each_tiers_bitmap_limit():
    eng = _engine()
    shapes, _, _, _ = _precompile()
    tiers = {}
    for rows, stride in shapes:
        tier = eng["_prefill_tier"](rows, stride, _TOP_K)
        bucket = eng["_prefill_bucket"](stride)
        plan = eng["route_streaming"](
            eng["_PREFILL_TIER_ROWS"][tier],
            max(bucket, _TOP_K + 1),
            max(bucket, _TOP_K + 1),
            _TOP_K,
            force_main=True,
        )
        assert plan["kernel"] == "main"
        blk = plan["tpl"][0]
        assert stride > _bitmap_limit(blk), (rows, stride, blk)
        assert stride <= _bitmap_limit(blk) + 512, "just past the limit: small start-up buffers"
        assert tier not in tiers, f"tier {tier} twice"
        tiers[tier] = (rows, stride, bucket, blk)
    assert set(tiers) == {0, 1, 2}, tiers
    # Tier 0 passes its limit only in the largest envelope bucket (a window wider than 262,144).
    assert tiers[0][2] == eng["_prefill_bucket"](1 << 30)
    assert [tiers[t][3] for t in (0, 1, 2)] == [1024, 512, 256]


def test_launches_on_a_context_server():
    shapes, fn, launches, log = _precompile()
    fn("context")
    assert [(x["rows"], x["stride"]) for x in launches] == list(shapes)
    for x in launches:
        assert x["width"] == x["stride"], "contiguous logits: the row stride is the width"
        assert x["logits"].dtype == torch.float32 and not bool(x["logits"].any())
        assert x["starts"].dtype == torch.int32 and x["ends"].dtype == torch.int32
        assert x["starts"].tolist() == [0] * x["rows"]
        assert x["ends"].tolist() == [x["width"]] * x["rows"], "full-row windows, wider than K"
        assert tuple(x["out"].shape) == (x["rows"], _TOP_K) and x["out"].dtype == torch.int32
        assert x["deterministic"] is True, "the indexer's prefill Top-K runs deterministic"
    assert len(log.info) == 1 and not log.warn
    assert log.info[0].startswith(
        "Kimi K3.1 Rubin prefill Top-K pre-compiled: kimi_k31_pro_nvfp4__sm_107__dep16: "
        "3 of 3 GVR launches past the bitmap limit (75x262400, 149x131328, 297x65792) in "
    ), log.info[0]
    fn("context")  # a second core in the process: the launches again, no second line
    assert len(launches) == 2 * len(shapes) and len(log.info) == 1


def test_a_generation_server_skips_it():
    _, fn, launches, log = _precompile()
    fn("generation")
    assert launches == [] and log.info == [] and log.warn == []


def test_out_of_memory_skips_one_shape():
    shapes, fn, launches, log = _precompile(fail_rows=149)
    fn("context")
    assert [(x["rows"], x["stride"]) for x in launches] == [s for s in shapes if s[0] != 149]
    assert len(log.warn) == 1 and "149x131328 out of memory" in log.warn[0]
    assert "2 of 3 GVR launches past the bitmap limit (75x262400, 297x65792)" in log.info[0]


def test_the_post_load_step_calls_it_once_built():
    body = ast.parse(_DEP16.read_text()).body
    core = next(n for n in body if isinstance(n, ast.ClassDef) and n.name.endswith("Dep16Core"))
    post = next(
        n for n in core.body if isinstance(n, ast.FunctionDef) and n.name == "post_load_weights"
    )
    calls = [
        (i, s)
        for i, s in enumerate(post.body)
        if isinstance(s, ast.Expr)
        and isinstance(s.value, ast.Call)
        and getattr(s.value.func, "id", None) == "_gvr_prefill_precompile"
    ]
    assert len(calls) == 1, "one call in the core's post-load step"
    i, stmt = calls[0]
    assert ast.unparse(stmt.value) == "_gvr_prefill_precompile(self.role)"
    built = [
        j
        for j, s in enumerate(post.body)
        if isinstance(s, ast.Assign)
        and ast.unparse(s.targets[0]) in ("self._prefill", "self._decode")
    ]
    assert built and i > max(built), "after the targets are built (and after the early return)"
    # Nothing else calls it, and it launches with the flag the indexer's prefill Top-K passes.
    source = ast.unparse(ast.Module(body=body, type_ignores=[]))
    assert source.count("_gvr_prefill_precompile(") == 2  # the definition, the call
    assert source.count("rubin_gvr_topk_prefill(logits, ks, ke, out, _IDX_GVR_DETERMINISTIC)") == 1
