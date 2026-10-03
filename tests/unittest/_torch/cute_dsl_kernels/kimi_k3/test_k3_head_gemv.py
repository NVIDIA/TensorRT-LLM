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
"""trtllm::k3_head_gemv (the drafter's lm_head vocab shard, stream-K, M <= 8) at the Kimi K3 TP16 shard
[10240, 7168], at every M in 1..8 as the DSpark worker calls it (default schedule): error against an fp64 product and
cuBLAS F.linear, run-to-run identical bits, each M's rows bit-identical to the same rows of the 8-row call, and calls of
different M back to back leaving the workspace counters at zero (the next call's bits unchanged). Stream-K weights
with fewer 128 x 128 tiles than SMs, or exactly as many, complete and match an fp64 product at every M."""

import functools
import os
import subprocess
import sys

import pytest
import torch
import torch.nn.functional as F


def _is_sm100() -> bool:
    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability()
    return major == 10


pytestmark = pytest.mark.skipif(not _is_sm100(), reason="needs an SM100-family GPU")

M_ALL = list(range(1, 9))
TOL = 8e-3
VOCAB_SHARD, HIDDEN = 10240, 7168  # 163840 / 16 rows per rank
# [N, K] weights with fewer 128 x 128 tiles than an SM100 GPU has SMs (56, 108, 120 and 144 tiles).
FEW_TILES = [(128, 7168), (1152, 1536), (1280, 1536), (2304, 1024)]
# For the few-tile child; import, compile and the calls take about a minute.
FEW_TILES_TIMEOUT_S = 300

# The few-tile shapes ("NxK" arguments) one after another at every M: prints each call's error against an fp64 product
# once the call has completed.
_FEW_TILES_CHILD = r"""
import sys
import torch
import tensorrt_llm  # noqa: F401
from tensorrt_llm._torch.cute_dsl_kernels.k3_head_gemv import op  # noqa: F401
gen = torch.Generator(device="cuda").manual_seed(20261002)
for shape in sys.argv[1:]:
    n, k = map(int, shape.split("x"))
    w = (torch.randn(n, k, generator=gen, device="cuda") * 0.02).bfloat16()
    x8 = torch.randn(8, k, generator=gen, device="cuda").bfloat16()
    for m in range(1, 9):
        x = x8[:m].contiguous()
        y = torch.ops.trtllm.k3_head_gemv(x, w)
        torch.cuda.synchronize()
        ref = x.double() @ w.double().t()
        print("REL", n, k, m, ((y.double() - ref).abs().max() / ref.abs().max()).item(), flush=True)
"""


def _ops():
    import tensorrt_llm  # noqa: F401
    from tensorrt_llm._torch.cute_dsl_kernels.k3_head_gemv import op  # noqa: F401

    return torch.ops.trtllm


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().view(torch.int16)


def _rel(y: torch.Tensor, ref: torch.Tensor) -> float:
    return (y.double() - ref.double()).abs().max().item() / ref.double().abs().max().item()


@functools.lru_cache(maxsize=None)
def _inputs():
    gen = torch.Generator(device="cuda").manual_seed(20261001)
    w = (torch.randn(VOCAB_SHARD, HIDDEN, generator=gen, device="cuda") * 0.02).bfloat16()
    x8 = torch.randn(8, HIDDEN, generator=gen, device="cuda").bfloat16()
    return w, x8


@pytest.mark.parametrize("m", M_ALL)
def test_k3_head_gemv(m):
    ops = _ops()
    from tensorrt_llm._torch.cute_dsl_kernels.k3_head_gemv import op

    w, x8 = _inputs()
    x = x8[:m].contiguous()
    assert op.supports(x, w)
    y = ops.k3_head_gemv(x, w)
    y8 = ops.k3_head_gemv(x8, w)
    again = ops.k3_head_gemv(x, w)
    ref = x.double() @ w.double().t()
    stock = F.linear(x, w)
    det = torch.equal(_bits(y), _bits(again))
    minv = torch.equal(_bits(y), _bits(y8[:m]))
    abs_err = (y.double() - ref).abs().max().item()
    print(f"OPCHECK op=k3_head_gemv case=lm_head_shard M={m} abs={abs_err:.3e} rel={_rel(y, ref):.3e} "
          f"vs_stock={_rel(y, stock):.2e} det={det} rows_as_m8={minv}")  # fmt: skip
    assert y.shape == (m, VOCAB_SHARD)
    assert _rel(y, ref) <= TOL and _rel(y, stock) <= TOL
    assert det and minv


def test_k3_head_gemv_mixed_m_sequence():
    """M 8, 1, 5, 8, 3 back to back on one stream (one workspace per shape): each the bits of its own call."""
    ops = _ops()
    w, x8 = _inputs()
    single = {m: ops.k3_head_gemv(x8[:m].contiguous(), w) for m in (1, 3, 5, 8)}
    seq = [ops.k3_head_gemv(x8[:m].contiguous(), w) for m in (8, 1, 5, 8, 3)]
    for m, y in zip((8, 1, 5, 8, 3), seq):
        assert torch.equal(_bits(y), _bits(single[m]))


def _tiles_equal_to_sms(sms):
    """An [N, K] weight with exactly ``sms`` 128 x 128 tiles, each 128-row tile split over at least 2 k-tiles."""
    k_tiles = max([d for d in range(2, 17) if sms % d == 0] or [1])
    return 128 * (sms // k_tiles), 128 * k_tiles


def test_k3_head_gemv_few_tiles():
    """Stream-K weights with fewer 128 x 128 tiles than SMs, and one with exactly as many: every call completes and
    matches an fp64 product. A stuck call would block its process, so the shapes run in a child process with a
    deadline."""
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    shapes = [(n, k) for n, k in FEW_TILES if (n // 128) * (k // 128) < sms]
    assert shapes
    shapes.append(_tiles_equal_to_sms(sms))
    # Without a compute-sanitizer target's injection variables: they would load the tool into the child too, and
    # there CUDA initialization can stall.
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith(("NV_SANITIZER_", "NVIDIA_PROCESS_INJECTION_"))
        and k != "NVTX_INJECTION64_PATH"
    }
    child = subprocess.Popen(
        [sys.executable, "-c", _FEW_TILES_CHILD, *(f"{n}x{k}" for n, k in shapes)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )
    try:
        out = child.communicate(timeout=FEW_TILES_TIMEOUT_S)[0]
    except subprocess.TimeoutExpired:
        child.kill()
        out = child.communicate()[0] + f"\nno result after {FEW_TILES_TIMEOUT_S} s"
    rel = {shape: {} for shape in shapes}
    for _, n, k, m, r in (ln.split() for ln in out.splitlines() if ln.startswith("REL ")):
        rel[(int(n), int(k))][int(m)] = float(r)
    for (n, k), by_m in rel.items():
        print(f"OPCHECK op=k3_head_gemv case=few_tiles N={n} K={k} tiles={(n // 128) * (k // 128)} sms={sms} "
              f"rel={by_m}")  # fmt: skip
    done = all(set(by_m) == set(M_ALL) for by_m in rel.values())
    assert child.returncode == 0 and done, out[-3000:]
    assert max(max(by_m.values()) for by_m in rel.values()) <= TOL, rel


@pytest.mark.parametrize("m", [0, 9, 16])
def test_token_limit(m):
    _ops()
    from tensorrt_llm._torch.cute_dsl_kernels.k3_head_gemv import op

    w, _ = _inputs()
    assert not op.supports(torch.zeros(m, HIDDEN, dtype=torch.bfloat16, device="cuda"), w)
