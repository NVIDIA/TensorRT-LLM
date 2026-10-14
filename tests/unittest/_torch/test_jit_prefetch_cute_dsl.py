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


def test_provider_plans_the_variant_the_op_keys_on():
    p = jcd.KdaPrefillProvider(torch.nn.Module())
    p.cfgs = [(H, True, True)]
    ragged, aligned = p.specs(False), p.specs(True)
    assert p.plan([100, 37, 200]) == ragged
    assert p.plan([64, 128]) == aligned
    assert p.plan([100]) == aligned  # single sequence: sentinel-padded onto aligned
    assert set(p.enumerate_specs()) == set(ragged + aligned)
