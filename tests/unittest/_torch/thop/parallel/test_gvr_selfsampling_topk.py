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
"""Exactness tests for the standalone self-sampling GVR top-K decode kernels
(`gvr_topk_decode_self_sampling[_host].py`).

Contract under test (see the host module docstring): batch-uniform host-int
``n_valid`` in compressed index space, fp32 logits with a 64-element-multiple
row stride, output = exact (tie-interchangeable) top-K indices of
``logits[:, :n_valid]`` per row.

Checks per case:
  - tie-aware exactness: the multiset of gathered output values equals the
    ``torch.topk`` value multiset bitwise (signed zeros normalized);
  - output indices are unique and within ``[0, n_valid)``;
  - padding immunity: the padded tail ``[n_valid, npad)`` is filled with huge
    values — any kernel read past ``n_valid`` fails the value comparison.
"""

import pytest
import torch
from utils.util import getSMVersion

import tensorrt_llm  # noqa: F401
from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE

if not torch.cuda.is_available():
    pytest.skip("CUDA is required for gvr_selfsampling_topk tests", allow_module_level=True)

if not IS_CUTLASS_DSL_AVAILABLE:
    pytest.skip("cutlass DSL is required for gvr_selfsampling_topk tests", allow_module_level=True)

if getSMVersion() != 100:
    pytest.skip("self-sampling GVR kernels target Blackwell sm_100", allow_module_level=True)

from tensorrt_llm._torch.cute_dsl_kernels.blackwell.top_k import (
    gvr_topk_decode_self_sampling_host as ss_host,
)

_DEV = "cuda"


def _make_case(batch_size, n_valid, top_k, seed, hit_ratio=0.6):
    """Decode-like fp32 logits + prev-step hint. The padded tail is poisoned
    with +3e38 so any read past n_valid corrupts the top-K values."""
    gen = torch.Generator(device=_DEV).manual_seed(seed)
    npad = (n_valid + 63) // 64 * 64
    logits = torch.randn((batch_size, npad), generator=gen, dtype=torch.float32, device=_DEV) - 2.0
    logits[:, n_valid:] = 3e38

    ref_vals, ref_idx = torch.topk(logits[:, :n_valid].float(), top_k, dim=1, largest=True)
    # hint: argmax first (anchor), then a hit_ratio slice of the true top-K,
    # the rest random valid indices — mirrors the decode-step temporal hint.
    n_hits = int(top_k * hit_ratio)
    rand_fill = torch.randint(
        0, n_valid, (batch_size, top_k), generator=gen, dtype=torch.int32, device=_DEV
    )
    pre_idx = rand_fill.clone()
    pre_idx[:, :n_hits] = ref_idx[:, :n_hits].to(torch.int32)
    indices = torch.full((batch_size, top_k), -1, dtype=torch.int32, device=_DEV)
    return logits, pre_idx, indices, ref_vals


def _check_exact(logits, indices, n_valid, ref_vals):
    top_k = indices.shape[1]
    idx64 = indices.to(torch.int64)
    assert int(idx64.min()) >= 0, "negative output index"
    assert int(idx64.max()) < n_valid, "output index past n_valid"
    for row in range(indices.shape[0]):
        assert int(torch.unique(idx64[row]).numel()) == top_k, f"row {row}: duplicate indices"
    got = torch.gather(logits, 1, idx64)
    # +0.0 maps -0.0 to +0.0 so signed zeros compare equal bitwise
    got_sorted = torch.sort(got + 0.0, dim=1, descending=True).values
    ref_sorted = torch.sort(ref_vals + 0.0, dim=1, descending=True).values
    assert torch.equal(got_sorted, ref_sorted), (
        "top-K value multiset mismatch (inexact or padding read)"
    )


# (top_k, n_valid) — gate-edge (131075/131076 straddle the K=2048 hint-band
# gate), the small-N floor, the mid band, and the deployment-envelope top.
_CASES = [
    (512, 4099),
    (512, 65536),
    (512, 262143),
    (1024, 16387),
    (1024, 131072),
    (2048, 4111),
    (2048, 131075),
    (2048, 131076),
    (2048, 262144),
]


@pytest.mark.parametrize("batch_size", [1, 4], ids=lambda b: f"bs{b}")
@pytest.mark.parametrize("top_k,n_valid", _CASES, ids=[f"k{k}_n{n}" for k, n in _CASES])
def test_selfsampling_topk_exactness(batch_size, top_k, n_valid):
    logits, pre_idx, indices, ref_vals = _make_case(
        batch_size, n_valid, top_k, seed=n_valid * 31 + top_k + batch_size
    )
    ss_host.run(logits, pre_idx, n_valid, indices)
    torch.cuda.synchronize()
    _check_exact(logits, indices, n_valid, ref_vals)


def test_selfsampling_topk_run_ws_explicit_workspace():
    """run_ws with a caller-owned workspace must agree with run()."""
    top_k, n_valid = 1024, 65536
    logits, pre_idx, indices, ref_vals = _make_case(2, n_valid, top_k, seed=7)
    ws = torch.zeros(ss_host.workspace_bytes(), dtype=torch.uint8, device=_DEV)
    ss_host.run_ws(logits, pre_idx, n_valid, indices, ws)
    torch.cuda.synchronize()
    _check_exact(logits, indices, n_valid, ref_vals)


def test_selfsampling_topk_guards():
    logits, pre_idx, indices, _ = _make_case(1, 8192, 512, seed=3)
    with pytest.raises(RuntimeError, match="float32"):
        ss_host.run(logits.to(torch.bfloat16), pre_idx, 8192, indices)
    with pytest.raises(RuntimeError, match="non-negative"):
        ss_host.run(logits, pre_idx, -1, indices)
    with pytest.raises(RuntimeError, match="batch dims"):
        ss_host.run(logits, pre_idx[:0], 8192, indices)


def test_selfsampling_dispatch_is_pure_and_total():
    """route(b, n, npad, k) must return a plan for every in-envelope shape."""
    for k in (512, 1024, 2048):
        for n in (k + 1, 4111, 65536, 131075, 131076, 262144):
            npad = (n + 63) // 64 * 64
            r = ss_host.route(4, n, npad, k)
            assert r["kernel"] in ("main", "reg", "clus", "reg_clus")
            assert r["block"] >= 128 and r["grid"][0] >= 1
