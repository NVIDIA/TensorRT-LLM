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
"""A KDA K123 variant exported by the CPU-only helper is what the real op runs.

The helper compiles from fake tensors, with no visible GPU, and exports the
variant. The real ``trtllm::kda_prefill`` then loads it in place of
``cute.compile`` and must give bitwise the same output and state as an
in-process compile, on batches of several shapes, without compiling.
"""

import json
import os
import subprocess
import sys

import pytest
import torch

from tensorrt_llm._torch import jit_prefetch_cute_dsl as jcd
from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE
from tensorrt_llm._utils import get_sm_version

pytestmark = pytest.mark.skipif(
    not IS_CUTLASS_DSL_AVAILABLE or get_sm_version() not in (100, 103),
    reason="needs SM100/SM103 and CuTe DSL",
)

H, K = 96, 128
_HELPER = os.path.join(
    os.path.dirname(os.path.abspath(jcd.__file__)), "jit_prefetch_cute_dsl_helper.py"
)


def _inputs(lengths, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)

    def rn(*s, dtype=torch.bfloat16, scale=0.05):
        return (torch.randn(*s, generator=g, dtype=torch.float32, device="cuda") * scale).to(dtype)

    T = sum(lengths)
    cu = torch.tensor(
        [0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.long, device="cuda"
    )
    q = torch.nn.functional.normalize(rn(1, T, H, K).float(), dim=-1).to(torch.bfloat16)
    k = torch.nn.functional.normalize(rn(1, T, H, K).float(), dim=-1).to(torch.bfloat16)
    return dict(
        q=q,
        k=k,
        v=rn(1, T, H, K),
        g=rn(1, T, H, K),
        beta=rn(1, T, H, dtype=torch.float32),
        state_pool=rn(len(lengths), H, K, K, dtype=torch.float32, scale=0.01),
        state_indices=torch.arange(len(lengths), dtype=torch.int32, device="cuda"),
        scale=K**-0.5,
        cu_seqlens=cu,
        safe_gate=True,
        lower_bound=-5.0,
        use_gate_in_kernel=True,
        use_beta_sigmoid_in_kernel=True,
        A_log=torch.randn(H, generator=g, device="cuda") * 0.1,
        dt_bias=torch.randn(H * K, generator=g, device="cuda") * 0.1,
    )


def _run(args):
    out = torch.ops.trtllm.kda_prefill(**args).clone()
    torch.cuda.synchronize()
    return out, args["state_pool"].clone()


@pytest.mark.parametrize("varlen_pure", [False, True])
def test_helper_export_is_what_the_op_runs(tmp_path, varlen_pure):
    from tensorrt_llm._torch.custom_ops import cute_dsl_kimi_k3_custom_ops as ops

    spec = jcd.k123_spec(H, True, True, varlen_pure, "int64", "int64", True, "float32")
    out_dir = str(tmp_path / "aot")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONNOUSERSITE="1")
    res = subprocess.run(
        [sys.executable, "-u", _HELPER, jcd.gpu_arch(), out_dir],
        input=json.dumps({"tag": 0, "spec": spec}) + "\n",
        capture_output=True,
        text=True,
        env=env,
        timeout=900,
    )
    assert res.returncode == 0, res.stderr[-4000:]
    res = [line for line in res.stdout.splitlines() if line.startswith("{")]
    assert json.loads(res[0]) == {"ready": True}
    r = json.loads(res[-1])
    assert r["ok"] and r["built"], r

    # Multi-sequence batches select varlen_pure from their lengths; a single
    # sequence is always the aligned variant.
    batches = [[64, 128, 64], [512, 64]] if varlen_pure else [[100, 37, 200], [513, 64, 3, 90]]
    ref = []
    ops.k123_loader = None
    ops._fused_k123_cache.clear()
    for i, lengths in enumerate(batches):
        ref.append(_run(_inputs(lengths, 10 + i)))
    keys = [k for k in ops._fused_k123_cache if k[5] is varlen_pure]
    assert len(keys) == 1, list(ops._fused_k123_cache)
    assert jcd.k123_spec_from_cache_key(keys[0]) == spec

    loads = []

    def loader(cache_key):
        fn = jcd.load(jcd.k123_spec_from_cache_key(cache_key), out_dir)
        loads.append(fn is not None)
        return fn

    ops._fused_k123_cache.clear()
    ops.k123_loader = loader
    try:
        for i, lengths in enumerate(batches):
            out, state = _run(_inputs(lengths, 10 + i))
            assert torch.equal(out, ref[i][0]), f"output differs at {lengths}"
            assert torch.equal(state, ref[i][1]), f"state differs at {lengths}"
    finally:
        ops.k123_loader = None
    assert loads == [True], f"op should load the export exactly once, got {loads}"


def _helper_build(out_dir, specs):
    """Run the CPU-only helper on ``specs``; return its replies."""
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONNOUSERSITE="1")
    lines = "".join(json.dumps({"tag": i, "spec": s}) + "\n" for i, s in enumerate(specs))
    res = subprocess.run(
        [sys.executable, "-u", _HELPER, jcd.gpu_arch(), out_dir, json.dumps(jcd.device_props())],
        input=lines,
        capture_output=True,
        text=True,
        env=env,
        timeout=900,
    )
    assert res.returncode == 0, res.stderr[-4000:]
    replies = [json.loads(line) for line in res.stdout.splitlines() if line.startswith("{")]
    assert replies[0] == {"ready": True}
    return replies[1:]


@pytest.mark.parametrize("contiguous", [True, False])
def test_flashinfer_norm_export_is_what_flashinfer_runs(tmp_path, contiguous):
    """FlashInfer's RMSNorm and fused add + RMSNorm, compiled by the helper
    (no visible GPU) and loaded through the wrapped getters, match FlashInfer's
    own compile bitwise; the getters compile nothing."""
    fi_norm = pytest.importorskip("flashinfer.norm")
    from flashinfer.norm.kernels import fused_add_rmsnorm as fa_mod
    from flashinfer.norm.kernels import rmsnorm as rms_mod

    if fi_norm._use_cuda_norm(torch.device("cuda")):
        pytest.skip("FlashInfer uses its CUDA norm kernels here")
    from flashinfer.norm.utils import get_sm_version

    H, eps = 7168, 1e-6
    sm = get_sm_version(torch.device("cuda"))
    specs = [
        jcd.fi_norm_spec(op, "bfloat16", H, 0.0, True, sm, contiguous)
        for op in ("rmsnorm", "fused_add_rmsnorm")
    ]
    out_dir = str(tmp_path / "aot")
    replies = _helper_build(out_dir, specs)
    assert all(r["ok"] and r["built"] for r in replies), replies

    g = torch.Generator(device="cuda").manual_seed(0)
    rows = 37
    width = H if contiguous else 2 * H  # a strided row view: the first H columns

    def rows_of(seed_tensor):
        return seed_tensor.to(torch.bfloat16)[:, :H]

    x_src = torch.randn(rows, width, generator=g, device="cuda")
    r_src = torch.randn(rows, width, generator=g, device="cuda")
    w = torch.randn(H, generator=g, device="cuda").to(torch.bfloat16)
    assert rows_of(x_src).is_contiguous() is contiguous

    def run():
        out = fi_norm.rmsnorm(rows_of(x_src), w, eps, enable_pdl=True)
        x, r = rows_of(x_src), rows_of(r_src)  # fresh copies, updated in place
        fi_norm.fused_add_rmsnorm(x, r, w, eps, enable_pdl=True)
        torch.cuda.synchronize()
        return out, x, r

    saved = (rms_mod._get_compiled_rmsnorm_kernel, fa_mod._get_compiled_fused_add_rmsnorm_kernel)
    try:
        ref = run()
        loads = []

        def loader(spec):
            fn = jcd.load(spec, out_dir)
            loads.append(fn is not None)
            return fn

        assert len(jcd.install_flashinfer_norm_loader(loader)) == 2
        got = run()
    finally:
        rms_mod._get_compiled_rmsnorm_kernel, fa_mod._get_compiled_fused_add_rmsnorm_kernel = saved
    assert loads == [True, True], loads
    for a, b in zip(got, ref):
        assert torch.equal(a, b)


def test_topk_export_is_what_the_op_runs(tmp_path):
    """Radix top-k variants enumerated by TopKProvider, compiled by the
    helper and loaded through the wrapped runners, give the same indices as
    an in-process compile, for decode batches of several row bands and
    widths, and the runners compile nothing themselves."""
    from tensorrt_llm._torch.custom_ops import cute_dsl_custom_ops as ops
    from tensorrt_llm._torch.modules.top_k import TopK, TopKImplementation

    top_k, max_cols = 512, 4096
    model = torch.nn.Module()
    model.top_k = TopK(top_k, decode_implementation=TopKImplementation.CUTE_DSL_RADIX)
    provider = jcd.TopKProvider(model, max_num_tokens=200, max_seq_len=max_cols)
    specs = provider.enumerate_specs((1,))
    assert specs, "no top-k variants enumerated"
    out_dir = str(tmp_path / "aot")
    replies = _helper_build(out_dir, specs)
    assert all(r["ok"] for r in replies), [r for r in replies if not r["ok"]][:3]

    g = torch.Generator(device="cuda").manual_seed(0)
    cases = [(1, 600), (8, 1000), (40, 4096), (160, 2049)]

    def run():
        outs = []
        for rows, cols in cases:
            scores = torch.randn(rows, cols, generator=g, device="cuda")
            lens = torch.full((rows,), cols, dtype=torch.int32, device="cuda")
            out = torch.empty(rows, top_k, dtype=torch.int32, device="cuda")
            torch.ops.trtllm.cute_dsl_indexer_topk_decode(scores, lens, out, top_k, 1)
            outs.append(out.sort(dim=1).values)
        torch.cuda.synchronize()
        return outs

    runners = {name: getattr(ops, name) for name in jcd.TOPK_RUNNERS}
    saved = {name: r.__dict__["_compile"] for name, r in runners.items()}
    caches = {name: dict(r.kernel_cache) for name, r in runners.items()}
    try:
        for r in runners.values():
            r.kernel_cache.clear()
        g.manual_seed(0)
        ref = run()
        for r in runners.values():
            r.kernel_cache.clear()
        loads = []

        def loader(spec):
            fn = jcd.load(spec, out_dir)
            loads.append(fn is not None)
            return fn

        assert jcd.install_topk_loader(loader)
        g.manual_seed(0)
        got = run()
    finally:
        for name, r in runners.items():
            setattr(r, "_compile", saved[name])
            r.kernel_cache.clear()
            r.kernel_cache.update(caches[name])
    assert loads and all(loads), f"some variants were not prefetched: {loads}"
    for a, b in zip(got, ref):
        assert torch.equal(a, b)


def test_provider_plans_the_variant_the_op_keys_on():
    p = jcd.KdaPrefillProvider(torch.nn.Module())
    p.cfgs = [(H, True, True)]
    ragged, aligned = p.specs(False), p.specs(True)
    assert p.plan([100, 37, 200]) == ragged
    assert p.plan([64, 128]) == aligned
    assert p.plan([100]) == aligned  # single sequence: sentinel-padded onto aligned
    assert set(p.enumerate_specs()) == set(ragged + aligned)
