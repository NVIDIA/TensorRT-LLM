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
"""A CuTe DSL variant exported by the CPU-only helper is what the real op runs.

The helper compiles from fake tensors, with no visible GPU, and exports the
variant. The real op (FlashInfer's RMSNorm, the NVFP4 dense GEMM runner) then
loads it in place of ``cute.compile`` and must give bitwise the same output
as an in-process compile, on several shapes, without compiling.
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

_HELPER = os.path.join(
    os.path.dirname(os.path.abspath(jcd.__file__)), "jit_prefetch_cute_dsl_helper.py"
)


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


def test_nvfp4_gemm_export_is_what_the_runner_runs(tmp_path):
    """Every NVFP4 dense GEMM tactic Nvfp4GemmProvider enumerates for a
    model's NVFP4 Linear layers (those whose allowed backends include CuTe
    DSL), compiled by the helper (no visible GPU) and loaded through the
    runner's loader, gives bitwise the runner's own in-process output on
    GEMMs of several shapes, and the runner compiles nothing."""
    from tensorrt_llm._torch.custom_ops import cute_dsl_custom_ops as ops
    from tensorrt_llm._torch.modules.linear import Linear
    from tensorrt_llm.models.modeling_utils import QuantConfig
    from tensorrt_llm.quantization.mode import QuantAlgo

    qc = QuantConfig(quant_algo=QuantAlgo.NVFP4)
    shapes = [(1024, 512), (2048, 1536)]  # (n, k)
    max_num_tokens = 256
    model = torch.nn.Module()
    for i, (n, k) in enumerate(shapes):
        setattr(model, f"proj{i}", Linear(k, n, bias=False, dtype=torch.bfloat16, quant_config=qc))
    # A layer without the CuTe DSL backend never runs this runner.
    model.other = Linear(
        512, 4096, bias=False, dtype=torch.bfloat16, quant_config=qc,
        nvfp4_allowed_backends=["cutlass"],
    )  # fmt: skip
    provider = jcd.Nvfp4GemmProvider(model, max_num_tokens)
    assert provider.shapes == sorted((n, k // 2) for n, k in shapes)
    specs = provider.enumerate_specs()
    runner = ops.CuteDSLNVFP4BlackwellRunner(torch.bfloat16)
    g = torch.Generator(device="cuda").manual_seed(0)

    def inputs(m, n, k):
        x = torch.randn(m, k, generator=g, device="cuda").to(torch.bfloat16)
        w = torch.randn(n, k, generator=g, device="cuda").to(torch.bfloat16)
        x_global = (448 * 6) / x.abs().max().float()
        w_global = (448 * 6) / w.abs().max().float()
        x_fp4, x_sf = torch.ops.trtllm.fp4_quantize(x, x_global, 16, False)
        w_fp4, w_sf = torch.ops.trtllm.fp4_quantize(w, w_global, 16, False)
        alpha = (1.0 / (x_global * w_global)).reshape(1).float()
        return [x_fp4, w_fp4, x_sf, w_sf, alpha]

    # M buckets the autotuner profiles (powers of two up to max_num_tokens).
    data, valid = {}, {}
    for n, k in shapes:
        for m in (1, 32, max_num_tokens):
            data[(m, n, k)] = inputs(m, n, k)
            valid[(m, n, k)] = [
                runner._unpack_tactic(t) for t in runner.get_valid_tactics(data[(m, n, k)], None)
            ]
    tactics = {t for ts in valid.values() for t in ts}
    planned = set()
    for spec in map(json.loads, specs):
        planned.add((tuple(spec["mma_tiler"]), tuple(spec["cluster"]), spec["swap_ab"],
                     spec["use_prefetch"], spec["swizzle_size"], spec["raster_along_m"]))  # fmt: skip
    # The provider covers every tactic the autotuner benchmarks for these shapes.
    assert tactics <= planned, tactics - planned
    replies = _helper_build(str(tmp_path / "aot"), specs)
    assert all(r["ok"] and r["built"] for r in replies), [r for r in replies if not r["ok"]][:3]

    def run():
        out = {}
        for c, x in data.items():
            for t in valid[c]:
                out[(t, c)] = runner(x, tactic=t).clone()
        torch.cuda.synchronize()
        return out

    saved = dict(ops.CuteDSLNVFP4BlackwellRunner.kernel_cache)
    try:
        ops.CuteDSLNVFP4BlackwellRunner.kernel_cache.clear()
        ops.nvfp4_gemm_loader = None
        ref = run()
        ops.CuteDSLNVFP4BlackwellRunner.kernel_cache.clear()
        loads = []

        def loader(spec):
            fn = jcd.load(spec, str(tmp_path / "aot"))
            loads.append((fn is not None, spec))
            return fn

        assert jcd.install_nvfp4_gemm_loader(loader)
        got = run()
    finally:
        ops.nvfp4_gemm_loader = None
        ops.CuteDSLNVFP4BlackwellRunner.kernel_cache.clear()
        ops.CuteDSLNVFP4BlackwellRunner.kernel_cache.update(saved)
    missing = [spec for ok, spec in loads if not ok]
    assert len(loads) == len(tactics) and not missing, f"tactics not prefetched: {missing}"
    for key, value in ref.items():
        assert torch.equal(got[key], value), key


def test_start_model_builds_queues_the_nvfp4_gemm_tactics_before_warmup(monkeypatch):
    """Right after model construction (no quant method yet) the NVFP4 GEMM
    provider already sees the NVFP4 Linear layers, and with compile ahead on
    ``start_model_builds`` hands every tactic to the CuTe DSL helpers; with
    it off, nothing is queued and no loader is installed."""
    from tensorrt_llm._torch import jit_prefetch as jp
    from tensorrt_llm._torch import jit_prefetch_engine
    from tensorrt_llm._torch.custom_ops import cute_dsl_custom_ops as ops
    from tensorrt_llm._torch.modules.linear import Linear
    from tensorrt_llm.models.modeling_utils import QuantConfig
    from tensorrt_llm.quantization.mode import QuantAlgo

    qc = QuantConfig(quant_algo=QuantAlgo.NVFP4)
    model = torch.nn.Module()
    model.proj = Linear(512, 1024, bias=False, dtype=torch.bfloat16, quant_config=qc)
    provider = jcd.Nvfp4GemmProvider(model, 256)
    assert provider.shapes == [(1024, 256)] and provider.specs

    queued = []

    class _Prefetcher:
        def enable_nvfp4_gemm(self, p):
            queued.extend(p.enumerate_specs())
            return True

        def enable_flashinfer_norm(self, p):
            return False

    monkeypatch.setattr(jit_prefetch_engine, "FLASHINFER_MODULE_PROVIDERS", ())
    monkeypatch.setattr(
        jit_prefetch_engine, "warmup_builds_sampling_module", lambda model, llm_args: False
    )
    monkeypatch.setenv("TLLM_JIT_AHEAD", "1")
    monkeypatch.setattr(jp.JitPrefetcher, "init", classmethod(lambda cls, rank: _Prefetcher()))
    llm_args = type("Args", (), {"max_num_tokens": 256})()
    assert jit_prefetch_engine.start_model_builds(model, llm_args=llm_args)
    assert queued == provider.specs
    monkeypatch.setenv("TLLM_JIT_AHEAD", "0")
    queued.clear()
    assert not jit_prefetch_engine.start_model_builds(model, llm_args=llm_args)
    assert queued == [] and ops.nvfp4_gemm_loader is None


def test_occupancy_queries_compile_the_probe_kernel_once(monkeypatch):
    """With the cache installed, HardwareInfo's cluster-occupancy queries give
    what uncached ones give, and its probe kernel compiles once per device
    instead of at every query."""
    import cutlass.cute as cute
    from cutlass.utils.hardware_info import HardwareInfo

    # From CuTe DSL's own lookup (an earlier test may have installed the
    # cache), restored at teardown.
    current = HardwareInfo._get_device_function
    monkeypatch.setattr(
        HardwareInfo, "_get_device_function", getattr(current, "_jitp_inner", current)
    )
    sizes = (1, 2, 4, 8)
    want = [HardwareInfo().get_max_active_clusters(n) for n in sizes]
    compiles = []
    real_compile = cute.compile

    def counting_compile(*args, **kwargs):
        compiles.append(args[0])
        return real_compile(*args, **kwargs)

    monkeypatch.setattr(cute, "compile", counting_compile)
    assert jcd.install_occupancy_kernel_cache()
    got = [HardwareInfo().get_max_active_clusters(n) for n in sizes + sizes]
    assert got == want + want
    assert len(compiles) == 1, compiles
