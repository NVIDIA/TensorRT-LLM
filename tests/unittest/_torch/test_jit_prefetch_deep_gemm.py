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
"""The CPU-only DeepGEMM helper builds exactly the cubin the real op loads.

Each case runs in a fresh subprocess (DeepGEMM keeps an in-process runtime
cache): the helper compiles one (M, N, K) into an empty ``DG_JIT_CACHE_DIR``,
then the real ``trtllm::fp8_swap_ab_gemm`` runs the same shape against that
cache; it must not add a cache entry.
"""

import json
import os
import subprocess
import sys
import textwrap

import pytest
import torch

from tensorrt_llm._torch import jit_prefetch_deep_gemm as jdg

pytestmark = pytest.mark.skipif(not jdg.supported(), reason="needs SM100 and the patched DeepGEMM")

_HELPER = os.path.join(os.path.dirname(os.path.abspath(jdg.__file__)), "jit_prefetch_dg_helper.py")

_REAL = textwrap.dedent("""
    import sys, torch
    import tensorrt_llm  # noqa: F401  (registers trtllm ops)
    from tensorrt_llm.quantization.utils.fp8_utils import (
        resmooth_to_fp8_e8m0, transform_sf_into_required_layout)
    m, n, k = map(int, sys.argv[1:4])
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(n, k, device="cuda").to(torch.float8_e4m3fn)
    ws = torch.ones((n + 127) // 128, (k + 127) // 128, device="cuda")
    # The layout FP8BlockScalesLinearMethod gives the weight on SM100.
    w, ws = resmooth_to_fp8_e8m0(w, ws)
    ws = transform_sf_into_required_layout(ws, mn=n, k=k,
                                           recipe=(1, 128, 128), is_sfa=False)
    torch.ops.trtllm.fp8_swap_ab_gemm(x, w, ws, disable_ue8m0_cast=True)
    torch.cuda.synchronize()
""")


_REAL_MOE = textwrap.dedent("""
    import sys, torch
    from tensorrt_llm import deep_gemm
    from tensorrt_llm.quantization.utils.fp8_utils import transform_sf_into_required_layout
    g, m, n, k, expected_m = map(int, sys.argv[1:6])
    a = torch.randn(g, m, k, device="cuda").to(torch.float8_e4m3fn)
    b = torch.randn(g, n, k, device="cuda").to(torch.float8_e4m3fn)
    # Packed UE8M0 scales, as DeepGemmFusedMoE passes them: FP32 scales would
    # make DeepGEMM JIT a separate SF-packing kernel the MoE path never runs.
    sfa = transform_sf_into_required_layout(
        torch.ones(g, m, (k + 127) // 128, device="cuda"), mn=m, k=k,
        recipe=(1, 128, 128), num_groups=g, is_sfa=True)
    sfb = transform_sf_into_required_layout(
        torch.ones(g, (n + 127) // 128, (k + 127) // 128, device="cuda"), mn=n, k=k,
        recipe=(1, 128, 128), num_groups=g, is_sfa=False)
    d = torch.empty(g, m, n, device="cuda", dtype=torch.bfloat16)
    masked_m = torch.full((g,), min(m, expected_m), device="cuda", dtype=torch.int32)
    deep_gemm.fp8_m_grouped_gemm_nt_masked((a, sfa), (b, sfb), d, masked_m, expected_m,
                                           disable_ue8m0_cast=True)
    torch.cuda.synchronize()
""")


def _entries(cache):
    root = os.path.join(cache, "cache")
    return set(os.listdir(root)) if os.path.isdir(root) else set()


def _helper_compile(cache, specs):
    major, minor, sms = jdg.target()
    env = dict(os.environ, DG_JIT_CACHE_DIR=cache, CUDA_VISIBLE_DEVICES="", PYTHONNOUSERSITE="1")
    inp = "".join(json.dumps({"tag": i, "spec": s}) + "\n" for i, s in enumerate(specs))
    out = subprocess.run(
        [sys.executable, "-u", _HELPER, jdg.package_dir(), str(major), str(minor), str(sms)],
        input=inp,
        capture_output=True,
        text=True,
        env=env,
        timeout=900,
        check=True,
    ).stdout.splitlines()
    assert json.loads(out[0]) == {"ready": True}
    return [json.loads(line) for line in out[1:]]


# Small, ragged and large M; N/K of DeepSeek-V3 FP8 projections.
@pytest.mark.parametrize("m", [1, 17, 139, 2305])
@pytest.mark.parametrize("n,k", [(2112, 7168), (7168, 2048)])
def test_helper_cubin_is_what_the_real_op_loads(tmp_path, m, n, k):
    cache = str(tmp_path / "dg")
    (r,) = _helper_compile(cache, [jdg.spec_for(m, n, k)])
    assert r["ok"] and r["built"], r
    built = _entries(cache)
    assert len(built) == 1, built

    # A second request for the same variant is a cache hit in the helper.
    (r2,) = _helper_compile(cache, [jdg.spec_for(m, n, k)])
    assert r2["ok"] and not r2["built"], r2

    env = dict(os.environ, DG_JIT_CACHE_DIR=cache)
    subprocess.run(
        [sys.executable, "-c", _REAL, str(m), str(n), str(k)], env=env, check=True, timeout=900
    )
    assert _entries(cache) == built, "real op compiled a different variant"


def test_provider_enumeration_windows():
    p = jdg.FP8LinearDeepGemmProvider(torch.nn.Module(), max_num_tokens=100)
    p.shapes = [(128, 512)]
    ms = [json.loads(s)["m"] for s in p.enumerate_specs()]
    assert ms[:16] == list(range(1, 17))
    # One M per 16-row window up to max_num_tokens, which is included.
    assert ms[16:] == [32, 48, 64, 80, 96, 100]
    assert p.plan_tokens(37) and not p.plan_tokens(37)


# A DeepSeek-V3-like expert slice (G experts per rank, gate+up / down), at
# batch sizes far apart: one helper compile must serve all of them.
@pytest.mark.parametrize("g,n,k", [(32, 4096, 7168), (32, 7168, 2048)])
def test_moe_helper_cubin_serves_every_batch(tmp_path, g, n, k):
    cache = str(tmp_path / "dg")
    (r,) = _helper_compile(cache, [jdg.moe_spec_for(g, 128, n, k, 1)])
    assert r["ok"] and r["built"], r
    built = _entries(cache)
    assert len(built) == 1, built

    env = dict(os.environ, DG_JIT_CACHE_DIR=cache)
    for m, expected_m in ((128, 3), (256, 40), (1024, 600)):
        subprocess.run(
            [sys.executable, "-c", _REAL_MOE, str(g), str(m), str(n), str(k), str(expected_m)],
            env=env,
            check=True,
            timeout=900,
        )
        assert _entries(cache) == built, f"real op compiled another variant at m={m}"
