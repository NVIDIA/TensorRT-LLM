# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Exactness tests for the block-skip fast path of the self-sampling GVR
top-K decode kernel (`run_bsk` / `bsk_gate` in
`gvr_topk_decode_self_sampling.py`).

The block-skip path consumes a per-32-element block-maxima side tensor
(`bmax`) and only touches logit blocks whose maximum can clear the sampled
threshold; everything downstream (emission, verify, retry ladder) is shared
with the baseline row pass, so the contract is unchanged: exact
(tie-interchangeable) top-K indices of ``logits[:, :n_valid]`` per row.

Checks:
  - gate-open shapes: `run_bsk` output value-multiset equals both the
    baseline `run` output and ``torch.topk`` bitwise;
  - gate semantics: `bsk_gate` is shape-only, with the working-set and
    N/K-thinness terms cutting exactly where documented, and `run_bsk`
    falls back to the baseline (still exact) on gated shapes;
  - compact-list overflow: an adversarial input where every block survives
    exceeds the fixed candidate-list capacity and must take the
    identity-walk fallback, still exactly.
"""

import pytest
import torch
from utils.util import getSMVersion

import tensorrt_llm  # noqa: F401
from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE

if not torch.cuda.is_available():
    pytest.skip("CUDA is required for gvr blockskip tests", allow_module_level=True)

if not IS_CUTLASS_DSL_AVAILABLE:
    pytest.skip("cutlass DSL is required for gvr blockskip tests", allow_module_level=True)

if getSMVersion() not in (100, 103):
    pytest.skip(
        "self-sampling GVR kernels target datacenter Blackwell (sm_100/103)",
        allow_module_level=True,
    )

from tensorrt_llm._torch.cute_dsl_kernels.blackwell.top_k import (
    gvr_topk_decode_self_sampling as ss_dev,
)

_DEV = "cuda"


def _ws():
    return torch.zeros(ss_dev.WS_BYTES // 4, dtype=torch.int32, device=_DEV)


def _bmax_of(logits):
    b, npad = logits.shape
    return torch.amax(logits.view(b, npad // 32, 32), dim=2).contiguous()


def _make_case(batch_size, n_valid, top_k, seed, hit_ratio=0.6):
    """Decode-like fp32 logits + prev-step hint; padded tail poisoned with
    +3e38 so any unmasked read past n_valid corrupts the top-K values (the
    bmax superset property must hold under hostile pads)."""
    gen = torch.Generator(device=_DEV).manual_seed(seed)
    npad = (n_valid + 63) // 64 * 64
    logits = torch.randn((batch_size, npad), generator=gen, dtype=torch.float32, device=_DEV) - 2.0
    logits[:, n_valid:] = 3e38
    ref_vals, ref_idx = torch.topk(logits[:, :n_valid].float(), top_k, dim=1, largest=True)
    n_hits = int(top_k * hit_ratio)
    pre_idx = torch.randint(
        0, n_valid, (batch_size, top_k), generator=gen, dtype=torch.int32, device=_DEV
    )
    pre_idx[:, :n_hits] = ref_idx[:, :n_hits].to(torch.int32)
    return logits, pre_idx, ref_vals


def _check_exact(logits, indices, n_valid, ref_vals):
    top_k = indices.shape[1]
    idx64 = indices.to(torch.int64)
    assert int(idx64.min()) >= 0, "negative output index"
    assert int(idx64.max()) < n_valid, "output index past n_valid"
    for row in range(indices.shape[0]):
        assert int(torch.unique(idx64[row]).numel()) == top_k, f"row {row}: duplicate indices"
    got = torch.gather(logits, 1, idx64)
    got_sorted = torch.sort(got + 0.0, dim=1, descending=True).values
    ref_sorted = torch.sort(ref_vals + 0.0, dim=1, descending=True).values
    assert torch.equal(got_sorted, ref_sorted), (
        "top-K value multiset mismatch (inexact or padding read)"
    )


# Gate-open shapes: batch working set >= 128 MiB and n >= 72*k.
_OPEN_CASES = [
    (256, 512, 131075),
    (256, 2048, 163775),
]


@pytest.mark.parametrize(
    "batch_size,top_k,n_valid", _OPEN_CASES, ids=[f"bs{b}_k{k}_n{n}" for b, k, n in _OPEN_CASES]
)
def test_blockskip_exactness_gate_open(batch_size, top_k, n_valid):
    logits, pre_idx, ref_vals = _make_case(batch_size, n_valid, top_k, seed=n_valid + top_k)
    npad = logits.shape[1]
    assert ss_dev.bsk_gate(batch_size, n_valid, npad, top_k), "case must be gate-open"
    ws = _ws()
    out_base = torch.full((batch_size, top_k), -1, dtype=torch.int32, device=_DEV)
    out_bsk = torch.full((batch_size, top_k), -1, dtype=torch.int32, device=_DEV)
    ss_dev.run(logits, pre_idx, n_valid, out_base, ws)
    ss_dev.run_bsk(logits, pre_idx, n_valid, out_bsk, ws, _bmax_of(logits))
    torch.cuda.synchronize()
    _check_exact(logits, out_base, n_valid, ref_vals)
    _check_exact(logits, out_bsk, n_valid, ref_vals)


def test_blockskip_gate_fallback_exact():
    """Gated shape (working set below the L2/DRAM cut): run_bsk must route to
    the baseline and stay exact."""
    batch_size, top_k, n_valid = 4, 2048, 163775
    logits, pre_idx, ref_vals = _make_case(batch_size, n_valid, top_k, seed=7)
    npad = logits.shape[1]
    assert not ss_dev.bsk_gate(batch_size, n_valid, npad, top_k)
    out = torch.full((batch_size, top_k), -1, dtype=torch.int32, device=_DEV)
    ss_dev.run_bsk(logits, pre_idx, n_valid, out, _ws(), _bmax_of(logits))
    torch.cuda.synchronize()
    _check_exact(logits, out, n_valid, ref_vals)


def test_blockskip_gate_boundaries():
    """Shape-only gate: 128 MiB working-set cut and the n >= 72*k thinness
    cut, both edges exact."""
    k = 2048
    n = 72 * k  # 147456
    npad = (n + 63) // 64 * 64
    ws_rows = (128 * 1024 * 1024) // (npad * 4)
    assert ss_dev.bsk_gate(ws_rows + 1, n, npad, k)
    assert not ss_dev.bsk_gate(ws_rows - 1, n, npad, k)
    assert ss_dev.bsk_gate(256, n, npad, k)
    assert not ss_dev.bsk_gate(256, n - 1, npad, k)


def test_blockskip_capacity_overflow_identity_walk():
    """Adversarial tie storm: one +1e30 spike per 32-element block makes every
    block survive the sampled threshold, overflowing the fixed compact-list
    capacity (BSK_CAPL); the kernel must fall back to the identity walk and
    stay exact (top-K = K of the tied spikes)."""
    batch_size, top_k = 256, 2048
    n_valid = 72 * top_k + 3  # gate-open; npad/32 blocks > BSK_CAPL
    npad = (n_valid + 63) // 64 * 64
    assert npad // 32 > ss_dev.BSK_CAPL
    gen = torch.Generator(device=_DEV).manual_seed(11)
    logits = torch.randn((batch_size, npad), generator=gen, dtype=torch.float32, device=_DEV) - 2.0
    logits[:, n_valid:] = 3e38
    spike_cols = torch.arange(0, n_valid - 32, 32, device=_DEV)
    logits[:, spike_cols] = 1e30
    assert ss_dev.bsk_gate(batch_size, n_valid, npad, top_k)
    ref_vals = torch.topk(logits[:, :n_valid].float(), top_k, dim=1, largest=True).values
    pre_idx = torch.randint(
        0, n_valid, (batch_size, top_k), generator=gen, dtype=torch.int32, device=_DEV
    )
    out = torch.full((batch_size, top_k), -1, dtype=torch.int32, device=_DEV)
    ss_dev.run_bsk(logits, pre_idx, n_valid, out, _ws(), _bmax_of(logits))
    torch.cuda.synchronize()
    _check_exact(logits, out, n_valid, ref_vals)
