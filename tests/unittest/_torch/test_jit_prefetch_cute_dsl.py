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

import inspect
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
    # A runner may inherit _compile; restore exactly what each class held.
    saved = {name: r.__dict__.get("_compile") for name, r in runners.items()}
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
            if saved[name] is None:
                if "_compile" in r.__dict__:
                    delattr(r, "_compile")
            else:
                setattr(r, "_compile", saved[name])
            r.kernel_cache.clear()
            r.kernel_cache.update(caches[name])
    assert loads and all(loads), f"some variants were not prefetched: {loads}"
    for a, b in zip(got, ref):
        assert torch.equal(a, b)


def test_topk_provider_finds_topk_inside_attention_backends():
    """A sparse attention backend (a plain object, not an nn.Module) owns its
    indexer and that indexer's TopK; the provider must find it there."""
    from tensorrt_llm._torch.modules.top_k import TopK, TopKImplementation

    class Backend:  # like an AttentionBackend: not an nn.Module
        def __init__(self):
            self.indexer = torch.nn.Module()
            self.indexer.top_k = TopK(512, decode_implementation=TopKImplementation.CUTE_DSL_RADIX)

    model = torch.nn.Module()
    model.layer = torch.nn.Module()
    model.layer.attn = Backend()
    # Attributes of the same names that are not backends are skipped: an
    # object without a __dict__ (an extension type) and a plain value.
    model.other = torch.nn.Module()
    model.other.attn = 3
    model.other.backend = object()
    provider = jcd.TopKProvider(model, max_num_tokens=64, max_seq_len=4096)
    assert provider.cfgs == {("decode", 512)}


def test_csa2_publish_export_is_what_the_op_runs(tmp_path):
    """CSA2 candidate publications compiled by the helper (no visible GPU)
    from the provider's specs, and loaded by CSA2's glue-kernel launcher,
    publish exactly what an in-process compile does, and the launcher
    compiles nothing."""
    from tensorrt_llm._torch.attention.backends.sparse.csa2 import kernel as csa2
    from tensorrt_llm._torch.attention.backends.sparse.csa2.indexer import CSA2Indexer
    from tensorrt_llm._torch.attention.backends.sparse.csa2.params import CSA2Layout

    block_size, topk_blocks = 8, 4
    layout = CSA2Layout(
        (1, 1),
        (0,),
        (0, 1),
        0,
        candidate_topk_blocks=topk_blocks,
        candidate_block_size=block_size,
        index_topk=32,
    )
    source = CSA2Indexer(layout, 0, 32, 128)
    model = torch.nn.Module()
    model.indexer = source
    provider = jcd.Csa2PublishProvider(model, max_seq_len=1024)
    assert provider.geometries == {(block_size, source.sparse_block)}
    specs = provider.enumerate_specs()
    replies = _helper_build(str(tmp_path / "aot"), specs)
    assert all(r["ok"] for r in replies), [r for r in replies if not r["ok"]][:3]

    def run():
        g = torch.Generator(device="cuda").manual_seed(0)
        outs = []
        for rows, width, start in [(5, 101, 0), (3, 13, 1), (2, 1000, 3)]:
            scores = torch.rand(rows + start, width, generator=g, device="cuda")[start:]
            visible = torch.full((rows + start,), width, dtype=torch.int64, device="cuda")[start:]
            published = min(topk_blocks, -(-width // block_size)) * block_size
            out = torch.full((rows + start, published), -1, dtype=torch.int32, device="cuda")
            ids = counts = None
            sb = source.sparse_block
            if sb:
                ids = torch.zeros(rows + start, published // sb, dtype=torch.int32, device="cuda")
                counts = torch.zeros(rows + start, dtype=torch.int32, device="cuda")
                ids, counts = ids[start:], counts[start:]
            assert csa2.publish_candidates_(
                scores, visible, out[start:], topk_blocks, block_size, ids, counts, sb
            )
            outs.append(out.clone())
        torch.cuda.synchronize()
        return outs

    saved = dict(csa2._COMPILED)
    try:
        csa2._COMPILED.clear()
        csa2.csa2_loader = None
        ref = run()
        csa2._COMPILED.clear()
        loads = []

        def loader(spec):
            fn = jcd.load(spec, str(tmp_path / "aot"))
            loads.append((fn is not None, spec))
            return fn

        assert jcd.install_csa2_loader(loader)
        got = run()
    finally:
        csa2.csa2_loader = None
        csa2._COMPILED.clear()
        csa2._COMPILED.update(saved)
    missing = [spec for ok, spec in loads if not ok]
    assert loads and not missing, f"publications not prefetched: {missing}"
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


def test_k4_export_serves_every_request_count(tmp_path):
    """One K4 export, built by the GPU-less helper, is what the op runs for
    staged state pools of any row count (the KDA mixer stages one row per
    context request of a BF16 cache): bitwise the same output and state as
    an in-process compile, and the op compiles nothing."""
    from tensorrt_llm._torch.custom_ops import cute_dsl_kimi_k3_custom_ops as ops

    out_dir = str(tmp_path / "aot")
    num_sm = inspect.signature(ops._launch_k4_persistent).parameters["num_sm"].default
    spec = jcd.k4_spec(num_sm, H, K, K, True)
    replies = _helper_build(out_dir, [spec])
    assert replies[0]["ok"] and replies[0]["built"], replies

    # A single sequence must be a chunk multiple here (the dispatcher pads it).
    batches = [[256], [100, 37], [64, 128, 77], [17] * 5, [130] * 16]
    ref = []
    ops.k4_loader = None
    ops._k4p_cache.clear()
    for i, lengths in enumerate(batches):
        ref.append(_run(_inputs(lengths, 30 + i)))
    keys = list(ops._k4p_cache)
    # The op keys on the pool's row count: one compile per request count.
    assert len(keys) == len({len(b) for b in batches}), keys
    assert {jcd.k4_spec_from_cache_key(k) for k in keys} == {spec}

    loads = []

    def loader(cache_key):
        fn = jcd.load(jcd.k4_spec_from_cache_key(cache_key), out_dir)
        loads.append(fn is not None)
        return fn

    compiles = []
    orig = ops.cute.compile

    class _Count:
        def __call__(self, fn, *a, **kw):
            if getattr(fn, "__qualname__", "").startswith("make_host_fn"):
                compiles.append(fn)
            return orig(fn, *a, **kw)

        def __getitem__(self, opts):
            return orig[opts]

    ops._k4p_cache.clear()
    ops.k4_loader = loader
    ops.cute.compile = _Count()
    try:
        for i, lengths in enumerate(batches):
            out, state = _run(_inputs(lengths, 30 + i))
            assert torch.equal(out, ref[i][0]), f"output differs at {lengths}"
            assert torch.equal(state, ref[i][1]), f"state differs at {lengths}"
    finally:
        ops.cute.compile = orig
        ops.k4_loader = None
    assert not compiles, f"the op compiled K4 {len(compiles)} time(s)"
    assert loads and all(loads), loads


def test_kda_provider_lists_one_k4_variant_per_layer_shape():
    p = jcd.KdaPrefillProvider(torch.nn.Module(), num_sm=148)
    assert p.k4_specs == [] and not p
    p.cfgs = [(H, True, True)]
    p.k4_specs = [jcd.k4_spec(148, H, K, K, True)]
    assert p.k4_specs[0] in p.plan([100, 37]) and p.k4_specs[0] in p.enumerate_specs()
    # Keys of pools of any row count map to that one spec; other layouts to none.
    for rows in (1, 3, 16):
        key = (0, 148, H, K, K, (torch.float32, (rows, H, K, K), (H * K * K, K * K, K, 1), True),
               (torch.int32, (1,), True))  # fmt: skip
        assert jcd.k4_spec_from_cache_key(key) == p.k4_specs[0]
    bf16_pool = (0, 148, H, K, K, (torch.bfloat16, (3, H, K, K), (H * K * K, K * K, K, 1), True),
                 (torch.int32, (1,), True))  # fmt: skip
    assert jcd.k4_spec_from_cache_key(bf16_pool) is None


def test_fp8_gemm_export_is_what_the_runner_runs(tmp_path):
    """Every FP8 blockwise GEMM tactic Fp8GemmProvider enumerates for a
    model's block-FP8 Linear layers (those that opt out of DeepGEMM), compiled
    by the helper (no visible GPU) and loaded through the runner's loader,
    gives bitwise the runner's own in-process output on GEMMs of several
    shapes, and the runner compiles nothing."""
    from tensorrt_llm._torch.custom_ops import cute_dsl_custom_ops as ops
    from tensorrt_llm._torch.modules.linear import Linear
    from tensorrt_llm.models.modeling_utils import QuantConfig
    from tensorrt_llm.quantization.mode import QuantAlgo

    qc = QuantConfig(quant_algo=QuantAlgo.FP8_BLOCK_SCALES)
    shapes = [(1024, 512), (2048, 1536)]  # (n, k)
    model = torch.nn.Module()
    for i, (n, k) in enumerate(shapes):
        setattr(
            model,
            f"proj{i}",
            Linear(k, n, bias=False, dtype=torch.bfloat16, quant_config=qc, disable_deep_gemm=True),
        )
    # A DeepGEMM layer is not this runner's.
    model.dg = Linear(512, 4096, bias=False, dtype=torch.bfloat16, quant_config=qc)
    provider = jcd.Fp8GemmProvider(model)
    assert provider.shapes == sorted(shapes)
    specs = provider.enumerate_specs()
    runner = ops.CuteDSLFp8BlackwellRunner(output_dtype=torch.bfloat16, use_tvm_ffi=True)
    g = torch.Generator(device="cuda").manual_seed(0)

    def inputs(m, n, k):
        x = torch.randn(m, k, generator=g, device="cuda").to(torch.bfloat16)
        a, a_sf = torch.ops.trtllm.fp8_quantize_1x128(x)
        w = (torch.randn(n, k, generator=g, device="cuda") * 0.5).to(torch.float8_e4m3fn)
        w_sf = torch.rand(n // 128, k // 128, generator=g, device="cuda") + 0.5
        return [a, w, a_sf, w_sf]

    tactics = set()
    data = {}
    for n, k in shapes:
        for m in (1, 37, 1000):
            data[(m, n, k)] = inputs(m, n, k)
            tactics.update(map(tuple, runner.get_valid_tactics(data[(m, n, k)], None)))
    # The provider covers every tactic the autotuner benchmarks for these shapes.
    planned = {
        (s["use_2cta"], tuple(s["mma_tiler"]), tuple(s["cluster"])) for s in map(json.loads, specs)
    }
    assert tactics == planned, tactics ^ planned
    replies = _helper_build(str(tmp_path / "aot"), specs)
    assert all(r["ok"] and r["built"] for r in replies), [r for r in replies if not r["ok"]][:3]

    def run():
        out = {}
        for t in sorted(tactics):
            for c, x in data.items():
                out[(t, c)] = runner(x, tactic=t).clone()
        torch.cuda.synchronize()
        return out

    saved = dict(ops.CuteDSLFp8BlackwellRunner.kernel_cache)
    try:
        ops.CuteDSLFp8BlackwellRunner.kernel_cache.clear()
        ops.fp8_gemm_loader = None
        ref = run()
        ops.CuteDSLFp8BlackwellRunner.kernel_cache.clear()
        loads = []

        def loader(spec):
            fn = jcd.load(spec, str(tmp_path / "aot"))
            loads.append((fn is not None, spec))
            return fn

        assert jcd.install_fp8_gemm_loader(loader)
        got = run()
    finally:
        ops.fp8_gemm_loader = None
        ops.CuteDSLFp8BlackwellRunner.kernel_cache.clear()
        ops.CuteDSLFp8BlackwellRunner.kernel_cache.update(saved)
    missing = [spec for ok, spec in loads if not ok]
    assert len(loads) == len(tactics) and not missing, f"tactics not prefetched: {missing}"
    for key, value in ref.items():
        assert torch.equal(got[key], value), key


def test_fp8_bmm_loads_the_gemm_tactic_exports(tmp_path):
    """The grouped FP8 O-LoRA projection of DeepSeek-V4 / V4.1 attention (an
    FP8 ``o_a_proj`` of shape (groups, o_lora_rank, group_dim)) runs the FP8
    blockwise GEMM's kernel as a batched GEMM: Fp8GemmProvider plans its
    (n, k) = (o_lora_rank, group_dim), and every tactic the BMM runner
    benchmarks, compiled by the helper and loaded through the GEMM loader,
    gives bitwise the runner's own output for several batch and row counts,
    and the runner compiles nothing."""
    from tensorrt_llm._torch.custom_ops import cute_dsl_custom_ops as ops

    groups, n, k = 4, 1024, 512  # (groups, o_lora_rank, group_dim)
    model = torch.nn.Module()
    model.attn = torch.nn.Module()
    model.attn.o_a_proj = torch.nn.Parameter(
        torch.empty(groups, n, k, dtype=torch.float8_e4m3fn, device="cuda"), requires_grad=False
    )
    # A BF16 O-LoRA projection runs no FP8 BMM.
    model.bf16 = torch.nn.Module()
    model.bf16.o_a_proj = torch.nn.Parameter(torch.empty(groups, 256, k), requires_grad=False)
    provider = jcd.Fp8GemmProvider(model)
    assert provider.shapes == [(n, k)]
    specs = provider.enumerate_specs()
    runner = ops.CuteDSLFp8BlackwellBmmRunner(output_dtype=torch.bfloat16, use_tvm_ffi=True)
    g = torch.Generator(device="cuda").manual_seed(0)

    def inputs(b, m):
        x = torch.randn(m, b, k, generator=g, device="cuda").to(torch.bfloat16)
        a, a_sf = torch.ops.trtllm.fp8_batched_quantize_1x128_permute102(x)
        w = (torch.randn(b, n, k, generator=g, device="cuda") * 0.5).to(torch.float8_e4m3fn)
        w_sf = torch.rand(b, n // 128, k // 128, generator=g, device="cuda") + 0.5
        out = torch.empty(m, b, n, dtype=torch.bfloat16, device="cuda")
        return [a, w, a_sf, w_sf, out.transpose(0, 1)]

    tactics = set()
    data = {}
    for b in (1, groups):
        for m in (1, 37, 1000):
            data[(b, m)] = inputs(b, m)
            tactics.update(map(tuple, runner.get_valid_tactics(data[(b, m)], None)))
    planned = {
        (s["use_2cta"], tuple(s["mma_tiler"]), tuple(s["cluster"])) for s in map(json.loads, specs)
    }
    assert tactics == planned, tactics ^ planned
    replies = _helper_build(str(tmp_path / "aot"), specs)
    assert all(r["ok"] and r["built"] for r in replies), [r for r in replies if not r["ok"]][:3]

    def run():
        out = {}
        for t in sorted(tactics):
            for c, x in data.items():
                x[4].zero_()
                runner(x, tactic=t)
                out[(t, c)] = x[4].clone()
        torch.cuda.synchronize()
        return out

    saved = dict(ops.CuteDSLFp8BlackwellBmmRunner.kernel_cache)
    try:
        ops.CuteDSLFp8BlackwellBmmRunner.kernel_cache.clear()
        ops.fp8_gemm_loader = None
        ref = run()
        ops.CuteDSLFp8BlackwellBmmRunner.kernel_cache.clear()
        loads = []

        def loader(spec):
            fn = jcd.load(spec, str(tmp_path / "aot"))
            loads.append((fn is not None, spec))
            return fn

        assert jcd.install_fp8_gemm_loader(loader)
        got = run()
    finally:
        ops.fp8_gemm_loader = None
        ops.CuteDSLFp8BlackwellBmmRunner.kernel_cache.clear()
        ops.CuteDSLFp8BlackwellBmmRunner.kernel_cache.update(saved)
    missing = [spec for ok, spec in loads if not ok]
    assert len(loads) == len(tactics) and not missing, f"tactics not prefetched: {missing}"
    for key, value in ref.items():
        assert torch.equal(got[key], value), key


def test_start_model_builds_queues_the_gemm_tactics_before_warmup(monkeypatch):
    """Right after model construction (no weights, no quant method yet) the
    FP8 GEMM provider already sees the block-FP8 Linear layers that opt out of
    DeepGEMM, and with prefetch on ``start_model_builds`` hands every tactic
    to the CuTe DSL helpers and installs the runner's loader."""
    from tensorrt_llm._torch import jit_prefetch as jp
    from tensorrt_llm._torch.custom_ops import cute_dsl_custom_ops as ops
    from tensorrt_llm._torch.modules.linear import Linear
    from tensorrt_llm.models.modeling_utils import QuantConfig
    from tensorrt_llm.quantization.mode import QuantAlgo

    qc = QuantConfig(quant_algo=QuantAlgo.FP8_BLOCK_SCALES)
    model = torch.nn.Module()
    model.proj = Linear(
        512, 1024, bias=False, dtype=torch.bfloat16, quant_config=qc, disable_deep_gemm=True,
        skip_create_weights_in_init=True,
    )  # fmt: skip
    assert getattr(model.proj, "quant_method", None) is None  # no weights yet
    provider = jcd.Fp8GemmProvider(model)
    assert provider.shapes == [(1024, 512)] and provider.specs

    queued = []

    class _Prefetcher:
        def enable_fp8_gemm(self, p):
            queued.extend(p.enumerate_specs())
            return True

        def enable_cute_dsl(self, p):
            return False

    monkeypatch.setenv("TLLM_JIT_PREFETCH", "1")
    monkeypatch.setattr(jp.JitPrefetcher, "init", classmethod(lambda cls, rank: _Prefetcher()))
    assert jp.start_model_builds(model)
    assert queued == provider.specs
    monkeypatch.setenv("TLLM_JIT_PREFETCH", "0")
    queued.clear()
    assert not jp.start_model_builds(model)
    assert queued == [] and ops.fp8_gemm_loader is None


def test_lowm_direct_export_is_what_flashinfer_runs(tmp_path):
    """FlashInfer's low-M BF16 direct GEMM, compiled by the helper (no
    visible GPU) for the variants the low-M AutoTuner race benchmarks and
    loaded through the wrapped getter, matches FlashInfer's own compile
    bitwise; the getter compiles nothing."""
    import dataclasses

    d = pytest.importorskip("flashinfer.gemm.kernels.dense_bf16_gemm_direct")

    cases = [(1, 16896, 7168), (7, 768, 7168), (14, 7168, 8448)]
    variants = [(m, n, k, t) for m, n, k in cases for t in d.autotune_tactics(m, n, k)[:2]]
    specs = [jcd.lowm_direct_spec(m, n, k, dataclasses.astuple(t), True) for m, n, k, t in variants]
    out_dir = str(tmp_path / "aot")
    replies = _helper_build(out_dir, specs)
    assert all(r["ok"] and r["built"] for r in replies), replies

    g = torch.Generator(device="cuda").manual_seed(0)
    inputs = {}
    for m, n, k, _ in variants:
        if (m, n, k) not in inputs:
            a = torch.randn(m, k, generator=g, device="cuda").to(torch.bfloat16)
            b = torch.randn(n, k, generator=g, device="cuda").to(torch.bfloat16)
            inputs[(m, n, k)] = (a, b)

    def run():
        outs = []
        for m, n, k, t in variants:
            a, b = inputs[(m, n, k)]
            out = torch.empty(m, n, dtype=torch.bfloat16, device="cuda")
            d.run_direct_dense(a, b.t(), out, True, t)
            outs.append(out)
        torch.cuda.synchronize()
        return outs

    saved = d._get_compiled_direct_kernel
    try:
        ref = run()
        loads = []

        def loader(spec):
            fn = jcd.load(spec, out_dir)
            loads.append(fn is not None)
            return fn

        assert jcd.install_lowm_direct_loader(loader)
        got = run()
    finally:
        d._get_compiled_direct_kernel = saved
    assert loads == [True] * len(variants), loads
    for a, b in zip(got, ref):
        assert torch.equal(a, b)


def test_lowm_provider_lists_the_race_variants(monkeypatch):
    """With the low-M path on, the provider sees the bias-free BF16 Linears
    of the constructed model and lists every direct tactic at every M the
    race profiles (the tuning buckets and 1..max_num_tokens); quantized and
    biased Linears never race the direct kernel."""
    import dataclasses

    d = pytest.importorskip("flashinfer.gemm.kernels.dense_bf16_gemm_direct")
    from tensorrt_llm._torch.modules import low_m_gemm as lm
    from tensorrt_llm._torch.modules.linear import Linear
    from tensorrt_llm.models.modeling_utils import QuantConfig
    from tensorrt_llm.quantization.mode import QuantAlgo

    monkeypatch.setattr(lm, "LOW_M_GEMM_ACTIVE", True)
    model = torch.nn.Module()
    model.a = Linear(7168, 768, bias=False, dtype=torch.bfloat16)
    model.b = Linear(7168, 768, bias=True, dtype=torch.bfloat16)
    model.c = Linear(
        7168, 1024, bias=False, dtype=torch.bfloat16,
        quant_config=QuantConfig(quant_algo=QuantAlgo.FP8_BLOCK_SCALES),
    )  # fmt: skip
    from tensorrt_llm._torch.flashinfer_utils import get_env_enable_pdl

    p = jcd.LowMDirectGemmProvider(model, max_num_tokens=14)
    assert p.shapes == [(768, 7168)]
    ms = sorted(set(lm._M_TUNING_BUCKETS) | set(range(1, 15)))
    pdl = bool(get_env_enable_pdl())
    want = {
        jcd.lowm_direct_spec(m, 768, 7168, dataclasses.astuple(t), pdl)
        for m in ms
        for t in [d.default_tactic(m, 768, 7168), *d.autotune_tactics(m, 768, 7168)]
    }
    assert set(p.specs) == want and len(p.specs) == len(want)
    monkeypatch.setattr(lm, "LOW_M_GEMM_ACTIVE", False)
    assert not jcd.LowMDirectGemmProvider(model, max_num_tokens=14)
