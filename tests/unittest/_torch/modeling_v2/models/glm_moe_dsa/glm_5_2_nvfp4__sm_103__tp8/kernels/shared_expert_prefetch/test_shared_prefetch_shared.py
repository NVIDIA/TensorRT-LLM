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
"""Single-GPU tests of shared_expert_prefetch's two-kernel BF16 shared expert
(``shared_prefetch_se.py``, an alternative to the native shared expert) on real GLM-5.2 weights.

* ``test_silu_variants``: every candidate SiLU-and-mul instruction sequence
  (``shared_prefetch_se_ptx.SILU_VARIANTS``) against the native Triton ``silu_and_mul`` over
  all 65,536 BF16 gate values times 24 up values; the default must be
  bit-identical.
* ``test_shared_expert``: the shared_expert_prefetch kernels against the native ``GatedMLP`` and
  the FP64 reference with native rounding points (``shared_prefetch_fixtures.reference``)
  on real hidden states (captured windows of layers 10, 30, 50, 77, all
  rows R = 1..8) for the layers in ``GLM_MOE_DSA_SHARED_EXPERT_PREFETCH_LAYERS`` (default
  3,10,30,50,77; ``all`` = every target layer) and the TP ranks in
  ``GLM_MOE_DSA_SHARED_EXPERT_PREFETCH_RANKS`` (default 0). Prints the error statistics per layer
  (``GLM_MOE_DSA_SHARED_EXPERT_PREFETCH_SE_STATS``) and checks the frozen error budget of
  ``se-reorder``.
* ``test_eligibility``: declines (host-only, nothing launched).
"""

import json
import os
import sys

import pytest
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "routed_moe"))
import routed_moe_fixtures as fx  # noqa: E402
import shared_prefetch_fixtures as zf  # noqa: E402

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


def _silu_probe(variant: int):
    import cutlass
    import cutlass.cute as cute
    import shared_prefetch_se_ptx as ptx
    from cutlass.cutlass_dsl import Int32, Int64

    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tp8.kernels.routed_moe import (  # noqa: E501
        ptx as wptx,
    )

    class SiluProbe:
        @cute.jit
        def run(self, g_addr: Int64, u_addr: Int64, o_addr: Int64, n: Int32, stream):
            self.kernel(g_addr, u_addr, o_addr, n).launch(
                grid=[(n + 255) // 256, 1, 1], block=[256, 1, 1], stream=stream
            )

        @cute.kernel
        def kernel(self, g_addr: Int64, u_addr: Int64, o_addr: Int64, n: Int32):
            tidx, _, _ = cute.arch.thread_idx()
            bidx, _, _ = cute.arch.block_idx()
            i = Int32(bidx) * 256 + Int32(tidx)
            if i < n:
                gv = wptx.bf16x2_to_f32(wptx.ld_nc_u16(g_addr + Int64(i) * 2), False)
                uv = wptx.bf16x2_to_f32(wptx.ld_nc_u16(u_addr + Int64(i) * 2), False)
                r = ptx.silu_mul_triton(gv, uv, variant)
                ptx.st_u16(o_addr + Int64(i) * 2, wptx.bf16_bits(r))

    from cuda.bindings import driver as cuda

    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    z = cutlass.Int64(0)
    fn = cute.compile(SiluProbe().run, z, z, z, cutlass.Int32(1), stream)

    def call(g, u):
        out = torch.empty_like(g)
        fn(
            cutlass.Int64(g.data_ptr()),
            cutlass.Int64(u.data_ptr()),
            cutlass.Int64(out.data_ptr()),
            cutlass.Int32(g.numel()),
            cuda.CUstream(torch.cuda.current_stream().cuda_stream),
        )
        return out

    return call


def test_silu_variants():
    import shared_prefetch_se_ptx as ptx

    import tensorrt_llm  # noqa: F401  (registers trtllm ops)

    gates = torch.arange(-32768, 32768, dtype=torch.int32).to(torch.int16).view(torch.bfloat16)
    gen = torch.Generator().manual_seed(7)
    ups = [1.0, -1.0, 0.5, 2.0, 3.140625, -0.0078125, 1e-3, 1e-20, 300.0, -6e4] + list(
        (torch.randn(14, generator=gen) * 4).tolist()
    )
    ups = torch.tensor(ups, dtype=torch.float32).to(torch.bfloat16)
    rows = ups.numel()
    d = gates.numel()
    x = torch.empty(rows, 2 * d, dtype=torch.bfloat16)
    x[:, :d] = gates[None, :]
    x[:, d:] = ups[:, None]
    x = x.cuda()
    native = torch.ops.trtllm.silu_and_mul(x)
    g = x[:, :d].contiguous().flatten()
    u = x[:, d:].contiguous().flatten()
    nat = native.flatten()
    nat_nan = torch.isnan(nat)
    results = {}
    for v in range(len(ptx.SILU_VARIANTS)):
        out = _silu_probe(v)(g, u)
        same = (out.view(torch.int16) == nat.view(torch.int16)) | (torch.isnan(out) & nat_nan)
        results[v] = int((~same).sum().item())
        print(
            f"GLM_MOE_DSA_SHARED_EXPERT_PREFETCH_SILU variant {v} {ptx.SILU_VARIANTS[v]}: {results[v]} of "
            f"{nat.numel()} differ",
            flush=True,
        )
    assert results[ptx.SILU_DEFAULT] == 0, results


def _windows():
    out = []
    for layer in (10, 30, 50, 77):
        cap = fx.load_real_capture(layer)
        for rows in range(1, 9):
            for x, _ in fx.real_windows(cap, rows, stride=97)[:12]:
                out.append(x)
    return out


def _layers():
    raw = os.environ.get("GLM_MOE_DSA_SHARED_EXPERT_PREFETCH_LAYERS", "3,10,30,50,77")
    return list(zf.TARGET_LAYERS) if raw == "all" else [int(v) for v in raw.split(",")]


def _ranks():
    return [
        int(v) for v in os.environ.get("GLM_MOE_DSA_SHARED_EXPERT_PREFETCH_RANKS", "0").split(",")
    ]


def _stats(y, ref):
    d = y.float() - ref.float()
    nrms = (d.pow(2).mean().sqrt() / ref.float().pow(2).mean().sqrt()).item()
    ulp = zf.ulp_distance(y, ref)
    return {
        "nrms": nrms,
        "max_ulp": int(ulp.max().item()),
        "frac_diff": (ulp > 0).float().mean().item(),
        "max_abs": d.abs().max().item(),
    }


def test_shared_expert():
    from shared_prefetch_se import shared_output

    wins = [w.cuda() for w in _windows()]
    failures = []
    for rank in _ranks():
        for layer in _layers():
            mlp = zf.build_shared_mlp(layer, rank)
            acc = {k: ([], []) for k in ("shared_expert_prefetch", "native", "native64")}
            repeat_ok = True
            for x in wins:
                ref = zf.reference(mlp, x)
                y7 = shared_output(mlp, x)
                assert y7 is not None
                y7b = shared_output(mlp, x)
                repeat_ok = repeat_ok and torch.equal(y7.view(torch.int16), y7b.view(torch.int16))
                for k, y in (
                    ("shared_expert_prefetch", y7),
                    ("native", mlp(x)),
                    ("native64", zf.native_in_batch(mlp, x)),
                ):
                    acc[k][0].append(y.flatten())
                    acc[k][1].append(ref.flatten())
            st = {k: _stats(torch.cat(v[0]), torch.cat(v[1])) for k, v in acc.items()}
            shared_prefetch_vs_native = _stats(
                torch.cat(acc["shared_expert_prefetch"][0]), torch.cat(acc["native"][0])
            )
            nat_spread = _stats(torch.cat(acc["native64"][0]), torch.cat(acc["native"][0]))
            row = {
                "layer": layer,
                "rank": rank,
                "windows": len(wins),
                "deterministic": repeat_ok,
                "vs_ref": st,
                "shared_prefetch_vs_native": shared_prefetch_vs_native,
                "native64_vs_native": nat_spread,
            }
            print("GLM_MOE_DSA_SHARED_EXPERT_PREFETCH_SE_STATS " + json.dumps(row), flush=True)
            # Frozen se-reorder budget: against the FP64 reference with native rounding points,
            # shared_expert_prefetch's normalized RMS error is at most 1.1x the larger of the two
            # native algorithms' and its largest ulp distance at most theirs + 1; the output is
            # run-to-run deterministic.
            nat_nrms = max(st["native"]["nrms"], st["native64"]["nrms"])
            nat_ulp = max(st["native"]["max_ulp"], st["native64"]["max_ulp"])
            if not (
                repeat_ok
                and st["shared_expert_prefetch"]["nrms"] <= 1.1 * nat_nrms
                and st["shared_expert_prefetch"]["max_ulp"] <= nat_ulp + 1
            ):
                failures.append(row)
            del mlp
    assert not failures, failures


@pytest.mark.parametrize("trig", ["end", "early"])
def test_staged_bit_identical(trig, monkeypatch):
    """Weights staged in shared memory (bulk copies before the dependency wait) give the unstaged outputs bit for
    bit: same fragments, same ``mma.sync`` order."""
    from shared_prefetch_se import shared_output

    wins = [w.cuda() for w in _windows()]
    bad = []
    for layer in _layers()[:2]:
        mlp = zf.build_shared_mlp(layer, _ranks()[0])
        for i, x in enumerate(wins):
            monkeypatch.delenv(
                "TRTLLM_STAIRCASE_GLM_LL_SHARED_EXPERT_PREFETCH_TWO_KERNEL_STAGE_WEIGHTS",
                raising=False,
            )
            want = shared_output(mlp, x)
            for down in ("48", "24"):
                monkeypatch.setenv(
                    "TRTLLM_STAIRCASE_GLM_LL_SHARED_EXPERT_PREFETCH_TWO_KERNEL_STAGE_WEIGHTS", "1"
                )
                monkeypatch.setenv(
                    "TRTLLM_STAIRCASE_GLM_LL_SHARED_EXPERT_PREFETCH_TWO_KERNEL_TRIGGER", trig
                )
                monkeypatch.setenv(
                    "TRTLLM_STAIRCASE_GLM_LL_SHARED_EXPERT_PREFETCH_TWO_KERNEL_DOWN_CTAS", down
                )
                got = shared_output(mlp, x)
                monkeypatch.delenv(
                    "TRTLLM_STAIRCASE_GLM_LL_SHARED_EXPERT_PREFETCH_TWO_KERNEL_DOWN_CTAS"
                )
                torch.cuda.synchronize()
                if not torch.equal(got.view(torch.int16), want.view(torch.int16)):
                    bad.append((layer, i, x.shape[0], down))
        del mlp
    print(f"GLM_MOE_DSA_SHARED_EXPERT_PREFETCH_SE_STAGED trig={trig} mismatches {bad}", flush=True)
    assert not bad


def test_eligibility():
    from shared_prefetch_se import eligibility

    mlp = zf.build_shared_mlp(10, 0)
    x = torch.zeros(6, fx.HIDDEN, dtype=torch.bfloat16, device="cuda")
    assert eligibility(mlp, x) == (True, "")
    assert not eligibility(mlp, torch.zeros(9, fx.HIDDEN, dtype=torch.bfloat16, device="cuda"))[0]
    assert not eligibility(mlp, x.float())[0]
    assert not eligibility(mlp, x[:, :-8])[0]
    assert not eligibility(
        mlp, torch.zeros(6, 2 * fx.HIDDEN, dtype=torch.bfloat16, device="cuda")[:, ::2]
    )[0]
    assert not eligibility(mlp, x.cpu())[0]
