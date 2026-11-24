# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Programmatic dependent launch (PDL) in the family's SM107 dense block-scaled GEMM
(``rubin_dense_blockscaled_gemm``).

The base kernel launches with ``use_pdl=True``. Its TMA warp runs ``griddepcontrol.wait`` before
its persistent loop, so no load reads what the previous grid may still be writing; its MMA warp
runs ``griddepcontrol.launch_dependents`` after its loop, so a PDL kernel after the GEMM can start
during the epilogue. Without those three statements the file is the kernel without PDL: the
"plain" build.

* CPU (the source only): each statement sits where it must and nowhere else, and the plain build
  differs from the kernel by those statements alone.
* SM107: the plain build is loaded from a copy of the kernel file. On the GEMMs the specialized
  targets run (KDA ``[q|k|v|g|f_a|b]`` and o_proj, shared experts gate_up and down, per target),
  the two builds give the same bits for every tactic tried; a CUDA graph of PDL kernels around
  the GEMM keeps their order; the kernel's host-side split-K refusals hold; and the PTX of the
  two builds differs by the two ``griddepcontrol`` lines only.
"""

import ast
import importlib
import importlib.util
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

REPO = Path(__file__).resolve().parents[8]
MODULE = "tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kernels.gemm.rubin_dense_blockscaled_gemm"
KERNEL = REPO / (MODULE.replace(".", "/") + ".py")
BASE = "Sm107BlockScaledPersistentDenseGemmKernel"
MIXED = "Sm107BlockScaledPersistentDenseGemmMixedClustersKernel"
WAIT, TRIGGER = "griddepcontrol_wait", "griddepcontrol_launch_dependents"
#: Where the two helpers come from: inline ``griddepcontrol`` PTX with side effects.
HELPERS = "tensorrt_llm._torch.kernels.blackwell.utils"
#: The three PDL statements; the plain build drops the one line that holds each.
PDL_LINES = (f"{WAIT}()", f"{TRIGGER}()", "use_pdl=True,")

#: ``(rows, N, K)`` of each GEMM this kernel runs for a target, at one rank's share
#: (``rubin_mxfp8_linear``'s cells): rows are a generation step's verify rows, or a prefill chunk.
SHAPES = {
    # TP8 attention and shared experts; generation b1 k7 (8 rows).
    "tep8": {
        "kda_qkvgfab": (8, 6288, 7168),
        "kda_o_proj": (8, 7168, 1536),
        "shared_gate_up": (8, 1536, 7168),
        "shared_down": (8, 7168, 768),
        "kda_qkvgfab_prefill": (2954, 6288, 7168),
    },
    # Attention data-parallel (96 KDA heads per rank), shared experts whole; generation b12 k7
    # (96 rows).
    "dep16": {
        "kda_qkvgfab": (96, 49376, 7168),
        "kda_o_proj": (96, 7168, 12288),
        "shared_gate_up": (96, 12288, 7168),
        "shared_down": (96, 7168, 6144),
        "kda_qkvgfab_prefill": (2954, 49376, 7168),
    },
    # KDA and shared experts at TP8 over the helix ranks (12 heads); generation b16 k5 (96 rows),
    # no prefill.
    "helix8": {
        "kda_qkvgfab": (96, 6288, 7168),
        "kda_o_proj": (96, 7168, 1536),
        "shared_gate_up": (96, 1536, 7168),
        "shared_down": (96, 7168, 768),
    },
}

#: The tactics tried on every shape the kernel can run them on: the runner's fallback tile, small-M
#: swapped tiles on one and two CTAs, the dynamic tile scheduler, an unswapped tile with raster N,
#: split-K into FP32 planes, and a mixed-cluster tile (launched without PDL).
TACTICS = (
    -1,
    ("base", (128, 64, 128), (128, 64, 64), (1, 1), True, False, "static", "m", 1),
    ("base", (256, 64, 128), (256, 64, 64), (2, 1), True, True, "static", "m", 1),
    ("base", (256, 128, 128), (256, 128, 64), (2, 1), True, False, "clc_dynamic", "m", 1),
    ("base", (128, 128, 128), (128, 128, 64), (1, 1), False, False, "static", "n", 1),
    ("base", (128, 64, 128), (128, 64, 64), (1, 1), True, False, "static", "m", 2),
    ("mixed_clusters", (256, 128, 128), (256, 128, 64), (4, 2), (2, 1), True, False, "static", "m"),
)
#: The PTX comparison: one small GEMM, three tactics (one and two CTAs, the dynamic scheduler).
PTX_SHAPE = (8, 7168, 1536)
PTX_TACTICS = TACTICS[1:4]
_THREADS = 256

requires_sm107 = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7),
    reason="needs an SM107 GPU",
)


# The source


def plain_source(text: str) -> str:
    """``text`` without the line of each PDL statement: the kernel without PDL."""
    for statement in PDL_LINES:
        line = re.compile(rf"^[ \t]*{re.escape(statement)}[^\n]*\n", re.MULTILINE)
        text, count = line.subn("", text)
        assert count == 1, f"{statement!r} is on {count} lines, expected one"
    return text


def _class(tree: ast.Module, name: str) -> ast.ClassDef:
    return next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name)


def _method(cls: ast.ClassDef, name: str) -> ast.FunctionDef:
    return next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name)


def _calls_of(node: ast.AST, name: str):
    """The calls of the bare name ``name`` under ``node``."""
    return [
        c
        for c in ast.walk(node)
        if isinstance(c, ast.Call) and isinstance(c.func, ast.Name) and c.func.id == name
    ]


def _is_call(statement: ast.stmt, name: str) -> bool:
    return isinstance(statement, ast.Expr) and bool(_calls_of(statement, name))


def _role(fn: ast.FunctionDef, warp: str) -> ast.If:
    """The ``if warp_idx == self.<warp>:`` branch of ``fn`` that holds the role's persistent loop."""
    for node in ast.walk(fn):
        if (
            isinstance(node, ast.If)
            and ast.unparse(node.test) == f"warp_idx == self.{warp}"
            and any(isinstance(s, ast.While) for s in node.body)
        ):
            return node
    raise AssertionError(f"no {warp} branch with a persistent loop")


def _launches(tree: ast.Module):
    """``(enclosing class, launch call)`` of every ``.launch(...)`` call of the file."""
    found = []
    for node in tree.body:
        owner = node.name if isinstance(node, ast.ClassDef) else None
        for call in ast.walk(node):
            if isinstance(call, ast.Call) and getattr(call.func, "attr", None) == "launch":
                found.append((owner, call))
    return found


class _DropPdl(ast.NodeTransformer):
    """Removes the two griddepcontrol statements and the ``use_pdl`` keyword."""

    def visit_Expr(self, node):
        if _is_call(node, WAIT) or _is_call(node, TRIGGER):
            return None
        return self.generic_visit(node)

    def visit_Call(self, node):
        self.generic_visit(node)
        node.keywords = [kw for kw in node.keywords if kw.arg != "use_pdl"]
        return node


@pytest.mark.cpu_only
def test_the_tma_warp_waits_before_its_first_load():
    """The wait is the TMA warp's first statement, ahead of its persistent loop: every TMA load
    and L2 prefetch of the operands (A, B and their scales) comes after it, also on a CTA with no
    tile."""
    tree = ast.parse(KERNEL.read_text())
    branch = _role(_method(_class(tree, BASE), "kernel_impl"), "tma_warp_id")
    assert _is_call(branch.body[0], WAIT)
    wait_line = branch.body[0].lineno
    loads = [
        c
        for c in ast.walk(branch)
        if isinstance(c, ast.Call) and ast.unparse(c.func) in ("cute.copy", "cute.prefetch")
    ]
    assert sum(ast.unparse(c.func) == "cute.copy" for c in loads) == 4, "A, B and their scales"
    assert all(c.lineno > wait_line for c in loads)


@pytest.mark.cpu_only
def test_alpha_is_the_one_tensor_element_read_ahead_of_the_wait():
    """Every thread reads ``alpha[0]`` at kernel entry, ahead of the wait. No kernel writes it: the
    entries make it once per device, outside CUDA graph capture. The kernel body reads no other
    tensor argument element by element (the operands go through TMA)."""
    tree = ast.parse(KERNEL.read_text())
    impl = _method(_class(tree, BASE), "kernel_impl")
    tensors = {
        a.arg
        for a in impl.args.args
        if a.annotation is not None and ast.unparse(a.annotation) == "cute.Tensor"
    }
    assert {"mA_mkl", "mB_nkl", "mSFA_mkl", "mSFB_nkl", "mC_mnl", "alpha"} <= tensors
    reads = [
        ast.unparse(node)
        for node in ast.walk(impl)
        if isinstance(node, ast.Subscript)
        and isinstance(node.value, ast.Name)
        and node.value.id in tensors
    ]
    assert reads == ["alpha[0]"]


@pytest.mark.cpu_only
def test_the_mma_warp_triggers_once_every_mma_is_issued():
    """The trigger follows the MMA warp's persistent loop and comes before its accumulator tail."""
    tree = ast.parse(KERNEL.read_text())
    body = _role(_method(_class(tree, BASE), "kernel_impl"), "mma_warp_id").body
    loop = max(i for i, s in enumerate(body) if isinstance(s, ast.While))
    assert _is_call(body[loop + 1], TRIGGER)
    assert ast.unparse(body[loop + 2]) == "acc_pipeline.producer_tail(acc_producer_state)"


@pytest.mark.cpu_only
def test_each_pdl_statement_once_and_only_the_base_launch_is_pdl():
    """One wait and one trigger in the file, both in the base kernel's body; the base launch is the
    one PDL launch (the mixed-cluster kernel and the split-K reduction launch without it); the
    helpers are the engine's inline-PTX ones."""
    tree = ast.parse(KERNEL.read_text())
    impl = _method(_class(tree, BASE), "kernel_impl")
    for name in (WAIT, TRIGGER):
        assert len(_calls_of(tree, name)) == len(_calls_of(impl, name)) == 1, name
    pdl = [
        (owner, call)
        for owner, call in _launches(tree)
        if any(kw.arg == "use_pdl" for kw in call.keywords)
    ]
    assert len(pdl) == 1 and pdl[0][0] == BASE
    (keyword,) = [kw for kw in pdl[0][1].keywords if kw.arg == "use_pdl"]
    assert isinstance(keyword.value, ast.Constant) and keyword.value.value is True
    owners = sorted(str(owner) for owner, _ in _launches(tree))
    assert owners == sorted([BASE, MIXED, "None"]), owners
    imported = {
        alias.name
        for node in tree.body
        if isinstance(node, ast.ImportFrom) and node.module == HELPERS
        for alias in node.names
    }
    assert {WAIT, TRIGGER} <= imported


@pytest.mark.cpu_only
def test_the_plain_build_is_the_kernel_without_the_pdl_statements():
    """Dropping the three lines leaves the kernel's syntax tree without exactly those three nodes:
    the bitwise comparisons below compare the change and nothing else."""
    text = KERNEL.read_text()
    plain = plain_source(text)
    assert len(text.splitlines()) - len(plain.splitlines()) == len(PDL_LINES)
    plain_tree = ast.parse(plain)
    assert not _calls_of(plain_tree, WAIT) and not _calls_of(plain_tree, TRIGGER)
    assert "use_pdl" not in plain
    assert ast.dump(_DropPdl().visit(ast.parse(text))) == ast.dump(plain_tree)


# SM107: the two builds


def _require_rubin_dsl():
    from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_RUBIN_AVAILABLE

    if not IS_CUTLASS_DSL_RUBIN_AVAILABLE:
        pytest.skip("needs a Rubin-capable CuTe DSL")


def _load(name: str, path: Path):
    """The module of ``path`` (a file, so the DSL can read its kernels' source) under ``name``."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def builds(tmp_path_factory):
    _require_rubin_dsl()
    pdl = importlib.import_module(MODULE)
    path = tmp_path_factory.mktemp("plain_build") / "rubin_dense_blockscaled_gemm_plain.py"
    path.write_text(plain_source(Path(pdl.__file__).read_text()))
    return SimpleNamespace(pdl=pdl, plain=_load("rubin_dense_blockscaled_gemm_plain", path))


def _pad(value: int, multiple: int) -> int:
    return (value + multiple - 1) // multiple * multiple


def _cdiv(a: int, b: int) -> int:
    return -(-a // b)


def _inputs(m: int, n: int, k: int, seed: int):
    """E4M3 operands with UE8M0 scales in the R128c4 layout, and the FP32 one the kernel reads."""
    g = torch.Generator(device="cuda")
    g.manual_seed(seed)
    a = (torch.randn(m, k, generator=g, device="cuda") * 2).clamp(-8, 8).to(torch.float8_e4m3fn)
    b = (torch.randn(n, k, generator=g, device="cuda") / 2).to(torch.float8_e4m3fn)
    sf_k = _pad(k // 32, 4)
    a_sf = torch.randint(
        124, 129, (_pad(m, 128) * sf_k,), generator=g, dtype=torch.uint8, device="cuda"
    )
    b_sf = torch.randint(
        120, 125, (_pad(n, 128) * sf_k,), generator=g, dtype=torch.uint8, device="cuda"
    )
    return [a, b, a_sf, b_sf, torch.ones((), dtype=torch.float32, device="cuda")]


def _runs(module, runner, tactic, m: int, n: int, k: int) -> bool:
    """Whether the kernel can run ``tactic`` on ``(m, n, k)`` (the runner's own filter)."""
    import cutlass

    if tactic == -1:
        return True
    swap = tactic[5] if tactic[0] == "mixed_clusters" else tactic[4]
    kernel_m, kernel_n, c_major = (n, m, "m") if swap else (m, n, "n")
    common = (
        (kernel_m, kernel_n, k, 1),
        cutlass.Float8E4M3FN,
        cutlass.Float8E4M3FN,
        cutlass.Float8E8M0FNU,
    )
    if tactic[0] == "mixed_clusters":
        return getattr(module, MIXED).can_implement(
            *common, cutlass.BFloat16, "k", "k", c_major, 32, *tactic[1:5]
        )
    split_k = tactic[8]
    c_dtype = cutlass.Float32 if runner._fp32_split_k_partials(split_k) else cutlass.BFloat16
    return getattr(module, BASE).can_implement(
        *common, c_dtype, "k", "k", c_major, 32, *tactic[1:4], split_k
    )


def _bytes(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().view(torch.uint8)


@requires_sm107
def test_can_implement_refuses_the_illegal_split_k(builds):
    """The kernel's ``can_implement`` refuses split-K with FP8 output and split-K that does not
    divide the K tiles; split 1 is legal. It runs before every test of this file that launches
    the GEMM."""
    import cutlass

    f8, bf16, e8 = cutlass.Float8E4M3FN, cutlass.BFloat16, cutlass.Float8E8M0FNU
    tile = ((128, 128, 128), (128, 128, 64), (1, 1))
    problem = ((7168, 96, 7168, 1), f8, f8, e8)  # 56 K tiles of 128
    kernel = getattr(builds.pdl, BASE)
    assert not kernel.can_implement(*problem, f8, "k", "k", "m", 32, *tile, 2)
    assert not kernel.can_implement(*problem, bf16, "k", "k", "m", 32, *tile, 3)
    assert kernel.can_implement(*problem, bf16, "k", "k", "m", 32, *tile, 1)


_CASES = [(target, site) for target, sites in SHAPES.items() for site in sites]


@requires_sm107
@pytest.mark.parametrize("target, site", _CASES)
def test_pdl_and_plain_builds_give_the_same_bits(builds, target, site):
    m, n, k = SHAPES[target][site]
    inputs = _inputs(m, n, k, seed=m * 7 + n * 3 + k)
    pdl = builds.pdl.RubinMxfp8GemmRunner(torch.bfloat16)
    plain = builds.plain.RubinMxfp8GemmRunner(torch.bfloat16)
    ran = []
    for tactic in TACTICS:
        if not _runs(builds.pdl, pdl, tactic, m, n, k):
            continue
        got = pdl(inputs, tactic=tactic)
        want = plain(inputs, tactic=tactic)
        torch.cuda.synchronize()
        assert torch.equal(_bytes(got), _bytes(want)), (target, site, tactic)
        assert torch.isfinite(want.float()).all() and bool(want.abs().amax() > 0), tactic
        ran.append(tactic)
    assert len(ran) >= 3, (target, site, ran)


def _chain_kernels():
    """Stand-ins of the PDL kernels around the GEMM (CuTe kernels launched with PDL)."""
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute

    from tensorrt_llm._torch.kernels.blackwell.utils import (
        griddepcontrol_launch_dependents,
        griddepcontrol_wait,
    )

    class Producer:
        """Waits for its own predecessor, lets the GEMM launch at once, then writes its A."""

        def __init__(self, blocks: int, per: int):
            self.blocks, self.per = blocks, per

        @cute.kernel
        def kernel(self, src: cute.Tensor, dst: cute.Tensor):
            tidx, _, _ = cute.arch.thread_idx()
            bidx, _, _ = cute.arch.block_idx()
            gdim, _, _ = cute.arch.grid_dim()
            griddepcontrol_wait()
            griddepcontrol_launch_dependents()
            t = gdim * _THREADS
            g = bidx * _THREADS + tidx
            for j in cutlass.range_constexpr(self.per):
                dst[j * t + g] = src[j * t + g]

        @cute.jit
        def run(self, src: cute.Tensor, dst: cute.Tensor, stream: cuda.CUstream):
            self.kernel(src, dst).launch(
                grid=[self.blocks, 1, 1], block=[_THREADS, 1, 1], stream=stream, use_pdl=True
            )

    class Consumer:
        """Reads an unrelated buffer before its wait, the GEMM's output after it; one sum per
        thread."""

        def __init__(self, blocks: int, per_pre: int, per_c: int):
            self.blocks, self.per_pre, self.per_c = blocks, per_pre, per_c

        @cute.kernel
        def kernel(self, pre: cute.Tensor, c: cute.Tensor, o: cute.Tensor):
            tidx, _, _ = cute.arch.thread_idx()
            bidx, _, _ = cute.arch.block_idx()
            gdim, _, _ = cute.arch.grid_dim()
            t = gdim * _THREADS
            g = bidx * _THREADS + tidx
            acc = cutlass.Int32(0)
            for j in cutlass.range_constexpr(self.per_pre):
                acc = acc + pre[j * t + g]
            griddepcontrol_wait()
            for j in cutlass.range_constexpr(self.per_c):
                acc = acc + c[j * t + g]
            o[g] = acc

        @cute.jit
        def run(self, pre: cute.Tensor, c: cute.Tensor, o: cute.Tensor, stream: cuda.CUstream):
            self.kernel(pre, c, o).launch(
                grid=[self.blocks, 1, 1], block=[_THREADS, 1, 1], stream=stream, use_pdl=True
            )

    def stream():
        return cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    return Producer, Consumer, cute, stream


_GEN_CASES = [(target, site) for target, site in _CASES if not site.endswith("_prefill")]


@requires_sm107
@pytest.mark.parametrize("target, site", _GEN_CASES)
def test_pdl_kernels_around_the_gemm_keep_their_order_in_a_cuda_graph(builds, target, site):
    """Producer -> GEMM -> consumer, all three launched with PDL, captured in one CUDA graph. The
    producer lets the GEMM launch before it writes the GEMM's A; the consumer reads the GEMM's
    output after its wait. Every replay gives the GEMM output and the consumer's sums of the same
    chain run one kernel at a time."""
    from cutlass.cute.runtime import from_dlpack

    m, n, k = SHAPES[target][site]
    Producer, Consumer, cute, stream = _chain_kernels()
    runners = {
        name: type("InPlace", (module.RubinMxfp8GemmRunner,), {"allow_unpartitioned_output": True})(
            torch.bfloat16
        )
        for name, module in (("pdl", builds.pdl), ("plain", builds.plain))
    }
    tactic = next(
        t
        for t in TACTICS[1:]
        if t[0] == "base" and t[8] == 1 and _runs(builds.pdl, runners["pdl"], t, m, n, k)
    )

    # Buffers as int32 words, padded so that the stand-ins need no bounds checks.
    per = 4
    p_threads = _pad(_cdiv(m * k // 4, per), _THREADS)
    a_buf = torch.zeros(p_threads * per, dtype=torch.int32, device="cuda")
    a_src = torch.zeros_like(a_buf)
    a_values, w, a_sf, w_sf, alpha = _inputs(m, n, k, seed=2)
    a_src.view(torch.uint8)[: m * k].copy_(_bytes(a_values).view(-1))
    a = a_buf.view(torch.uint8)[: m * k].view(torch.float8_e4m3fn).view(m, k)
    c_words, pre_words = m * n // 2, m * 7168 // 2
    q_threads = _pad(max(_cdiv(pre_words, 16), _cdiv(c_words, 16), _THREADS), _THREADS)
    per_pre, per_c = _cdiv(pre_words, q_threads), _cdiv(c_words, q_threads)
    pre = torch.randint(-1000, 1000, (q_threads * per_pre,), dtype=torch.int32, device="cuda")
    c_buf = torch.zeros(q_threads * per_c, dtype=torch.int32, device="cuda")
    out = c_buf.view(torch.bfloat16)[: m * n].view(m, n)
    sums = torch.zeros(q_threads, dtype=torch.int32, device="cuda")
    inputs = [a, w, a_sf, w_sf, alpha, out]

    src_c, a_c = from_dlpack(a_src), from_dlpack(a_buf)
    pre_c, c_c, sums_c = from_dlpack(pre), from_dlpack(c_buf), from_dlpack(sums)
    producer = cute.compile(Producer(p_threads // _THREADS, per).run, src_c, a_c, stream())
    consumer = cute.compile(
        Consumer(q_threads // _THREADS, per_pre, per_c).run, pre_c, c_c, sums_c, stream()
    )

    # The chain one kernel at a time.
    a_buf.zero_()
    c_buf.zero_()
    producer(src_c, a_c, stream())
    torch.cuda.synchronize()
    runners["plain"](inputs, tactic=tactic)
    torch.cuda.synchronize()
    consumer(pre_c, c_c, sums_c, stream())
    torch.cuda.synchronize()
    want_out, want_sums = out.clone(), sums.clone()
    assert bool(want_out.abs().amax() > 0)

    def chain(runner):
        a_buf.zero_()
        c_buf.zero_()
        producer(src_c, a_c, stream())
        runner(inputs, tactic=tactic)
        consumer(pre_c, c_c, sums_c, stream())

    for name, runner in runners.items():
        chain(runner)  # eager, with PDL
        torch.cuda.synchronize()
        assert torch.equal(_bytes(out), _bytes(want_out)), (name, "eager")
        assert torch.equal(sums, want_sums), (name, "eager")
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            chain(runner)
        for replay in range(4):
            sums.zero_()
            graph.replay()
            torch.cuda.synchronize()
            assert torch.equal(_bytes(out), _bytes(want_out)), (name, replay)
            assert torch.equal(sums, want_sums), (name, replay)
        del graph


# SM107: the PTX of the two builds


def _normalized_ptx(text: str) -> str:
    """PTX without comment lines, the kernel's symbol names replaced."""
    lines = [line.rstrip() for line in text.splitlines() if not line.lstrip().startswith("//")]
    symbol = r"[A-Za-z_$][\w$]*(?:rubin_dense_blockscaled_gemm|Sm107BlockScaledPersistentDenseGemm)[\w$]*"
    return re.sub(symbol, "<sym>", "\n".join(lines))


def _files(root: Path):
    return {
        p.relative_to(root).as_posix(): (p.stat().st_mtime_ns, p.stat().st_size)
        for p in root.rglob("*")
        if p.is_file()
    }


def _compile_ptx(build: str, out: Path) -> None:
    """Child process: compile the ``PTX_TACTICS`` the kernel can run with one build, and write
    their normalized PTX.

    The plain build is loaded under the kernel module's own name, so both builds give their
    kernels the same symbols. The parent sets the CuTe DSL dump and cache directories.
    """
    dump = Path(os.environ["CUTE_DSL_DUMP_DIR"])
    dump.mkdir(parents=True, exist_ok=True)
    origin = Path(importlib.util.find_spec(MODULE).origin)  # imports the parent packages only
    if build == "plain":
        assert MODULE not in sys.modules
        path = out.parent / "plain_src" / origin.name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(plain_source(origin.read_text()))
        module = _load(MODULE, path)
    else:
        module = importlib.import_module(MODULE)
    runner = module.RubinMxfp8GemmRunner(torch.bfloat16)
    inputs = _inputs(*PTX_SHAPE, seed=3)
    found = {}
    for tactic in PTX_TACTICS:
        if not _runs(module, runner, tactic, *PTX_SHAPE):
            continue
        before = _files(dump)
        runner(inputs, tactic=tactic)
        torch.cuda.synchronize()
        new = sorted(
            f for f, v in _files(dump).items() if before.get(f) != v and f.endswith(".ptx")
        )
        found[repr(tactic)] = [_normalized_ptx((dump / f).read_text(errors="replace")) for f in new]
    out.write_text(json.dumps(found))


@requires_sm107
def test_the_two_builds_compile_to_ptx_two_griddepcontrol_lines_apart(tmp_path):
    """Each build compiles in a child process with an empty CuTe DSL cache and dumps its PTX. Per
    tactic, the PDL build's PTX holds one ``griddepcontrol.wait`` and one
    ``griddepcontrol.launch_dependents`` and, without those two lines, is the plain build's."""
    _require_rubin_dsl()
    ptx = {}
    for build in ("pdl", "plain"):
        root = tmp_path / build
        out = root / "ptx.json"
        env = dict(
            os.environ,
            CUTE_DSL_KEEP_PTX="1",
            CUTE_DSL_DUMP_DIR=str(root / "dump"),
            CUTE_DSL_CACHE_DIR=str(root / "cache"),
            XDG_CACHE_HOME=str(root / "xdg"),
        )
        root.mkdir()
        done = subprocess.run(
            [sys.executable, __file__, "--compile-ptx", build, str(out)],
            env=env,
            capture_output=True,
            text=True,
            timeout=1800,
        )
        assert done.returncode == 0, (build, done.stdout[-3000:], done.stderr[-3000:])
        ptx[build] = json.loads(out.read_text())
    assert set(ptx["pdl"]) == set(ptx["plain"]) and len(ptx["plain"]) >= 2, list(ptx["plain"])
    for tactic, plain in ptx["plain"].items():
        pdl = ptx["pdl"][tactic]
        assert len(pdl) == len(plain) == 1, (tactic, len(pdl), len(plain))
        pdl_lines, plain_lines = pdl[0].splitlines(), plain[0].splitlines()
        marked = sorted(line.strip() for line in pdl_lines if "griddepcontrol" in line)
        assert marked == ["griddepcontrol.launch_dependents;", "griddepcontrol.wait;"], tactic
        assert not any("griddepcontrol" in line for line in plain_lines), tactic
        assert [line for line in pdl_lines if "griddepcontrol" not in line] == plain_lines, tactic


if __name__ == "__main__" and sys.argv[1:2] == ["--compile-ptx"]:
    _compile_ptx(sys.argv[2], Path(sys.argv[3]))
