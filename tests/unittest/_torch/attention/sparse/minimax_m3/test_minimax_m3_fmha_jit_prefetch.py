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
"""The MiniMax-M3 MSA fmha_sm100 JIT-prefetch planner and its CPU-only builder.

The planner must name the variant every real ``fmha_sm100`` call selects, for
any context rows, at each MSA call site: the indexer's proxy max-score pass
(also its segmented, KV-split plan), the dense layers' attention, and the
block-sparse attention of a BF16/FP8 cache. The helper, with no visible GPU, must build a kernel where the real
call loads it, so the real call compiles nothing.
"""

import json
import os
import subprocess
import sys

import pytest
import torch

import tensorrt_llm._torch as trtllm_torch
from tensorrt_llm._torch.attention.backends.sparse.minimax_m3 import jit_prefetch_fmha as jpf

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
    reason="needs an SM100-family GPU",
)

PAGE = 128
_HELPER = os.path.join(
    os.path.dirname(os.path.abspath(trtllm_torch.__file__)), "jit_prefetch_fmha_sm100_helper.py"
)


def _fmha():
    from tensorrt_llm._torch.attention.backends.sparse.minimax_m3.kernels.msa_utils import (
        msa_package_available,
        require_msa_module,
    )

    if not msa_package_available():
        pytest.skip("fmha_sm100 is not installed")
    return require_msa_module()


class _NoLaunch:
    """Stands in for a compiled variant: records nothing, launches nothing."""

    def run(self, *args):
        return None


def _selected_variants(fmha, dtype, heads, kv_heads, mode, qo_lens, kv_len=4 * PAGE):
    """The ``get_fmha_variant`` keys the real plan and run of this batch
    select, in order (a split plan selects one per half). Variants are
    neither built nor launched, nor is the CuTe DSL sparse prefill a long
    block-sparse half is planned for."""
    import fmha_sm100.api as api

    seen = []

    def record(*key):
        seen.append(tuple(key))
        return _NoLaunch()

    def no_sparse_prefill(*args, **kwargs):
        return kwargs.get("out"), None

    qo = torch.tensor(qo_lens, dtype=torch.int32)
    kv = torch.full_like(qo, kv_len)
    plan_kwargs = dict(
        num_kv_heads=kv_heads,
        qo_offset=kv - qo,
        page_size=PAGE,
        num_kv_splits=1,
        causal=True,
        use_fp8_kvcache=dtype == torch.float8_e4m3fn,
    )
    run_kwargs = dict(sm_scale=0.1)
    total_q = int(qo.sum())
    if mode == jpf._ONLY_SCORE:
        plan_kwargs["output_maxscore"] = True
        run_kwargs.update(output_o=False, output_maxscore=True)
    else:
        run_kwargs.update(output_maxscore=False)
    if mode == jpf._SPARSE:
        plan_kwargs["kv_block_num"] = 16
    plan = fmha.fmha_sm100_plan(qo, kv, heads, **plan_kwargs)
    num_pages = (kv_len // PAGE) * len(qo_lens)
    q = torch.zeros(total_q, heads, 128, dtype=dtype, device="cuda")
    k = torch.zeros(num_pages, kv_heads, PAGE, 128, dtype=dtype, device="cuda")
    run_kwargs["kv_indices"] = torch.arange(num_pages, dtype=torch.int32, device="cuda")
    if mode == jpf._SPARSE:
        run_kwargs["kv_block_indexes"] = torch.zeros(
            total_q, kv_heads, 16, dtype=torch.int32, device="cuda"
        )
    if mode != jpf._ONLY_SCORE:
        run_kwargs["out"] = torch.empty(total_q, heads, 128, dtype=torch.bfloat16, device="cuda")
    orig_variant, orig_sparse = api.get_fmha_variant, api.sparse_fmha
    api.get_fmha_variant, api.sparse_fmha = record, no_sparse_prefill
    try:
        fmha.fmha_sm100(q, k, k, plan, **run_kwargs)
    finally:
        api.get_fmha_variant, api.sparse_fmha = orig_variant, orig_sparse
    return seen


# (q dtype, query heads, KV heads, sparse mode) per rank, for MiniMax-M3's
# 64 q / 4 KV / 4 index heads at TP1, TP2 and TP4: the indexer's proxy pass
# (MQA over the index-K cache), the dense layers, the block-sparse attention.
SITES = [
    (torch.float8_e4m3fn, 4, 1, jpf._ONLY_SCORE),
    (torch.float8_e4m3fn, 2, 1, jpf._ONLY_SCORE),
    (torch.bfloat16, 1, 1, jpf._ONLY_SCORE),
    (torch.float8_e4m3fn, 64, 4, jpf._OFF),
    (torch.float8_e4m3fn, 32, 2, jpf._OFF),
    (torch.bfloat16, 16, 1, jpf._OFF),
    (torch.float8_e4m3fn, 32, 2, jpf._SPARSE),
    (torch.bfloat16, 16, 1, jpf._SPARSE),
]
# Longest rows across every class (1, packed heads, the 64/128 packed-row
# limits, the 256-row tile) and batches mixing short and long rows.
ROWS = [[1], [2], [5], [9], [16], [17], [31], [32], [33], [64], [65], [128], [129], [700],
        [3, 1], [40, 7], [2, 300], [300, 2]]  # fmt: skip


def _planned(dtype, heads, kv_heads, mode):
    name = "bfloat16" if dtype == torch.bfloat16 else "float8_e4m3fn"
    return set(jpf.planned_variant_keys([(name, heads, kv_heads, mode)], PAGE))


@pytest.mark.parametrize("site", SITES, ids=lambda s: f"{str(s[0])[6:]}-{s[1]}x{s[2]}-mode{s[3]}")
@pytest.mark.parametrize("qo_lens", ROWS, ids=lambda r: "rows" + "_".join(map(str, r)))
def test_planned_variants_cover_real_calls(site, qo_lens):
    fmha = _fmha()
    dtype, heads, kv_heads, mode = site
    real = _selected_variants(fmha, dtype, heads, kv_heads, mode, qo_lens)
    if mode != jpf._SPARSE or min(qo_lens) <= jpf._SPARSE_PREFILL_ROWS:
        assert real, "no variant selected"
    missing = set(real) - _planned(dtype, heads, kv_heads, mode)
    assert not missing, f"rows={qo_lens}: unplanned {sorted(missing)}"


def _selected_split_proxy_variants(fmha, dtype, heads, rows, num_kv_splits):
    """The ``get_fmha_variant`` keys the indexer's segmented proxy plan of
    these (queries, cached prefix) rows selects; nothing is built or
    launched."""
    import fmha_sm100.api as api

    from tensorrt_llm._torch.attention.backends.sparse.minimax_m3 import msa_indexer

    seen = []

    def record(*key):
        seen.append(tuple(key))
        return _NoLaunch()

    qo_lens = [n for n, _ in rows]
    num_pages = sum((n + p + PAGE - 1) // PAGE for n, p in rows)
    kv_indices = torch.arange(num_pages, dtype=torch.int32, device="cuda")
    plan, plan_kv_indices = msa_indexer._segmented_proxy_plan(
        fmha,
        qo_lens,
        [p for _, p in rows],
        num_index_heads=heads,
        page_size=PAGE,
        num_kv_splits=num_kv_splits,
        kv_indices=kv_indices,
    )
    q = torch.zeros(sum(qo_lens), heads, 128, dtype=dtype, device="cuda")
    k = torch.zeros(num_pages, 1, PAGE, 128, dtype=dtype, device="cuda")
    orig_variant = api.get_fmha_variant
    api.get_fmha_variant = record
    try:
        fmha.fmha_sm100(
            q,
            k,
            k,
            plan,
            kv_indices=plan_kv_indices,
            output_o=False,
            output_maxscore=True,
            sm_scale=0.1,
        )
    finally:
        api.get_fmha_variant = orig_variant
    return seen


# Rows (queries, cached prefix) the indexer plans segmented: warmup's prewarm
# (one 128-query segment), and serving rows past 64 queries over a long prefix.
SPLIT_ROWS = [((128, 0),), ((65, 8192),), ((1025, 131072),), ((1500, 60000), (50, 20000), (300, 0))]


@pytest.mark.parametrize(
    "site",
    [s for s in SITES if s[3] == jpf._ONLY_SCORE],
    ids=lambda s: f"{str(s[0])[6:]}-{s[1]}x{s[2]}",
)
@pytest.mark.parametrize("rows", SPLIT_ROWS, ids=lambda r: "rows" + "_".join(str(n) for n, _ in r))
@pytest.mark.parametrize("num_kv_splits", [1, 2, 16])
def test_planned_variants_cover_split_proxy_plans(site, rows, num_kv_splits):
    fmha = _fmha()
    dtype, heads, kv_heads, mode = site
    real = _selected_split_proxy_variants(fmha, dtype, heads, rows, num_kv_splits)
    assert real, "no variant selected"
    missing = set(real) - _planned(dtype, heads, kv_heads, mode)
    assert not missing, f"rows={rows} splits={num_kv_splits}: unplanned {sorted(missing)}"


def _probe(env):
    """This environment's fmha_sm100 package dir and cache base."""
    code = (
        "import json, os\n"
        "from tensorrt_llm._torch.attention.backends.sparse.minimax_m3.jit_prefetch_fmha import "
        "fmha_jit_module\n"
        "jit = fmha_jit_module()\n"
        "print(json.dumps([os.path.dirname(os.path.abspath(jit.__file__)), str(jit.CACHE_BASE)]))\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], env=env, check=True, capture_output=True, text=True
    )
    return json.loads(out.stdout.strip().splitlines()[-1])


def _helper(env, pkg_dir, cache_base, specs, timeout):
    stdin = "".join(json.dumps({"tag": i, "spec": s}) + "\n" for i, s in enumerate(specs))
    out = subprocess.run(
        [sys.executable, _HELPER, str(os.getpid()), pkg_dir, cache_base],
        input=stdin,
        env=dict(env, CUDA_VISIBLE_DEVICES=""),
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    lines = [json.loads(line) for line in out.stdout.splitlines() if line.strip()]
    assert lines and lines[0] == {"ready": True}, out.stderr[-3000:]
    return lines[1:], out.stderr


_REAL_CALL = """
import json, sys, torch
from tensorrt_llm._torch.attention.backends.sparse.minimax_m3.kernels.msa_utils import (
    require_msa_module)
fmha = require_msa_module(); jit = fmha.jit
built = []
orig_variant = jit.FMHAVariantManager._compile_only
def compile_only(self, name, params):
    if not (jit.CACHE_BASE / name / (name + ".so")).exists():
        built.append(name)
    return orig_variant(self, name, params)
jit.FMHAVariantManager._compile_only = compile_only
orig_plan = jit._do_compile_plan
def compile_plan():
    if not (jit.CACHE_BASE / "plan" / "fmha_sm100_plan.so").exists():
        built.append("plan")
    return orig_plan()
jit._do_compile_plan = compile_plan
qo = torch.tensor([129], dtype=torch.int32); kv = torch.tensor([512], dtype=torch.int32)
plan = fmha.fmha_sm100_plan(qo, kv, 32, num_kv_heads=2, qo_offset=kv - qo, page_size=128,
                            num_kv_splits=1, causal=True, use_fp8_kvcache=True)
q = torch.zeros(129, 32, 128, dtype=torch.float8_e4m3fn, device="cuda")
k = torch.zeros(4, 2, 128, 128, dtype=torch.float8_e4m3fn, device="cuda")
idx = torch.arange(4, dtype=torch.int32, device="cuda")
fmha.fmha_sm100(q, k, k, plan, kv_indices=idx, sm_scale=0.1, output_maxscore=False)
torch.cuda.synchronize()
print("BUILT " + json.dumps(built))
"""


def test_helper_builds_what_the_real_call_loads(tmp_path):
    """The GPU-less helper builds the planner and one variant into an empty
    cache; a process that then runs the matching real call builds nothing."""
    _fmha()
    env = dict(os.environ, MINFER_FMHA_CACHE_DIR=str(tmp_path / "minfer"))
    pkg_dir, cache_base = _probe(env)
    key = jpf.row_variant("float8_e4m3fn", 32, 2, jpf._OFF, PAGE, 129)
    specs = [jpf.module_spec("plan"), jpf.variant_spec(*key)]
    replies, err = _helper(env, pkg_dir, cache_base, specs, timeout=900)
    assert [r["tag"] for r in replies] == [0, 1]
    assert all(r["ok"] and r["built"] for r in replies), err[-3000:]
    # A second request for the same kernels finds them built.
    replies, _ = _helper(env, pkg_dir, cache_base, specs, timeout=120)
    assert all(r["ok"] and not r["built"] for r in replies)
    res = subprocess.run(
        [sys.executable, "-c", _REAL_CALL], env=env, check=True, capture_output=True, text=True
    )
    built = [line for line in res.stdout.splitlines() if line.startswith("BUILT ")]
    assert built == ["BUILT []"], res.stdout[-2000:] + res.stderr[-2000:]


def test_helper_refuses_another_cache(tmp_path):
    """A helper whose environment resolves another cache base reports every
    request as failed instead of building where the parent never looks."""
    _fmha()
    env = dict(os.environ, MINFER_FMHA_CACHE_DIR=str(tmp_path / "minfer"))
    pkg_dir, _ = _probe(env)
    replies, _ = _helper(
        env, pkg_dir, str(tmp_path / "elsewhere"), [jpf.module_spec("plan")], timeout=120
    )
    assert len(replies) == 1 and not replies[0]["ok"] and "cache base" in replies[0]["err"]
    assert not (tmp_path / "minfer").exists()
