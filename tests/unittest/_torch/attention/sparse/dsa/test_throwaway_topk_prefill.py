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
"""Correctness tests for the fused throwaway Top-K used by the DSA prefill indexer.

GPU tests compare ``ThrowawayPrefillTopK.run_chunk`` with a PyTorch reference for token-scaled
FP8 and packed MXFP4 inputs. Exact Top-K implementations may pick different indices among tied
scores, so a selection is valid when it has the right size, lies inside the causal window, has no
duplicates and every selected score is at least the k-th largest reference score (scores are
rounded to the logits storage dtype first). The dispatch tests run on CPU with a fake runner.

``THROWAWAY_TOPK_TEST_LARGE=1`` adds the 64K/128K-key shapes, which need minutes of JIT time.
"""

import gc
import itertools
import os
import weakref
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Optional

import pytest
import torch

from tensorrt_llm._torch.attention.backends.sparse.dsa import indexer as indexer_module
from tensorrt_llm._torch.attention.backends.sparse.dsa import (
    throwaway_topk as throwaway_topk_module,
)
from tensorrt_llm._torch.attention.backends.sparse.dsa.indexer import (
    Indexer,
    IndexerPrefillChunkMetadata,
    throwaway_unfused_logits_elem_bound,
)
from tensorrt_llm._torch.attention.backends.sparse.dsa.throwaway_topk import (
    THROWAWAY_LOGITS_DTYPE_ENV,
    THROWAWAY_MAX_CHUNK_ENV,
    THROWAWAY_MIN_CONTEXT_ENV,
    ThrowawayPrefillTopK,
    get_throwaway_prefill_topk,
    throwaway_topk_launch_rows,
    throwaway_topk_logits_dtype,
    throwaway_topk_max_chunk,
    throwaway_topk_min_context,
    throwaway_topk_supported,
    validate_throwaway_topk_env,
)
from tensorrt_llm._utils import get_sm_version

HEAD_DIM = 128
# Score tensor budget (elements) of one reference block: rows_per_block * heads * keys.
REF_BLOCK_ELEMS = 1 << 27
MAX_REF_ROWS = 256
# Relative slack on the k-th largest score: ~1 ulp of the logits storage dtype.
SCORE_SLACK = {torch.float32: 1e-3, torch.bfloat16: 8e-3, torch.float16: 2e-3}
# E2M1 code 0b0010 is +1.0; two of them per byte.
FP4_ONE_BYTE = 0x22
UE8M0_ONE = 127
LARGE = os.environ.get("THROWAWAY_TOPK_TEST_LARGE", "0") == "1"

skip_not_sm100 = pytest.mark.skipif(
    get_sm_version() not in (100, 103),
    reason=f"throwaway Top-K supports SM 100/103 only, got SM {get_sm_version()}",
)

# name: (num_heads, top_k, compress_ratio) pairs that have a kernel policy
CONFIGS = {
    "ds_v32": (64, 2048, 1),
    "glm_5_3": (32, 2048, 1),
    "dsv4_flash": (64, 512, 4),
    "dsv4_pro": (64, 1024, 4),
    # Qwen3.8-Flash-Next: FP8, fixed weights 128 ** -0.5 and unit key scales (kernel contract)
    "qwen": (4, 512, 4),
}
QWEN_MODEL_CONFIG = "Qwen3.8-Flash-Next"
FP8_DISTRIBUTIONS = ["all_equal", "all_zero", "few_high", "monotonic", "heavy_tail"]
FP4_DISTRIBUTIONS = ["all_equal", "heavy_tail"]

_RUNNERS = {}


def _runner(
    config,
    max_context_length,
    max_chunk=None,
    monkeypatch=None,
    logits_dtype=torch.float32,
    qkv="fp8",
):
    """Runners JIT-compile on first use, so reuse them across tests."""
    key = (config, max_context_length, max_chunk, logits_dtype, qkv)
    if key not in _RUNNERS:
        heads, top_k, ratio = CONFIGS[config]
        if max_chunk is not None:
            monkeypatch.setenv(THROWAWAY_MAX_CHUNK_ENV, str(max_chunk))
        _RUNNERS[key] = ThrowawayPrefillTopK(
            n_heads=heads,
            index_topk=top_k,
            compress_ratio=ratio,
            max_context_length=max_context_length,
            device=torch.device("cuda"),
            logits_dtype=logits_dtype,
            qkv_dtype=qkv,
            model_config=QWEN_MODEL_CONFIG if config == "qwen" else None,
        )
    return _RUNNERS[key]


@dataclass
class Inputs:
    q: torch.Tensor
    k: torch.Tensor
    k_scale: torch.Tensor
    weights: torch.Tensor
    q_scale: Optional[torch.Tensor]
    qkv: str

    def reference_operands(self, rows):
        """Dequantized (q rows, all keys, per-key scale or None) in FP32."""
        if self.qkv == "fp8":
            return self.q[rows].float(), self.k.float(), self.k_scale
        if not hasattr(self, "_k_decoded"):
            self._k_decoded = Indexer._decode_mxfp4(self.k, self.k_scale)
        return Indexer._decode_mxfp4(self.q[rows], self.q_scale[rows]), self._k_decoded, None


def _fp8(x):
    return x.to(torch.float8_e4m3fn)


def _make_inputs(config, num_q, kv_prefix, seed, qkv="fp8", dist="random"):
    heads, _, ratio = CONFIGS[config]
    num_k = (kv_prefix + num_q) // ratio
    gen = torch.Generator(device="cuda").manual_seed(seed)

    def randn(*shape):
        return torch.randn(*shape, device="cuda", generator=gen)

    def rand(*shape):
        return torch.rand(*shape, device="cuda", generator=gen)

    weights = rand(num_q, heads)
    if qkv == "mxfp4":
        half = HEAD_DIM // 2

        def packed(*shape):
            return torch.randint(0, 256, shape, dtype=torch.uint8, device="cuda", generator=gen)

        q = packed(num_q, heads, half)
        k = packed(num_k, half)
        # UE8M0 exponent bytes near 127 keep the dot products well inside FP32 range.
        q_scale_bytes = torch.randint(
            125, 130, (num_q, heads, 4), dtype=torch.uint8, device="cuda", generator=gen
        )
        k_scale = torch.randint(
            125, 130, (num_k, 4), dtype=torch.uint8, device="cuda", generator=gen
        )
        if dist == "all_equal":
            q.fill_(FP4_ONE_BYTE)
            k.fill_(FP4_ONE_BYTE)
            q_scale_bytes.fill_(UE8M0_ONE)
            k_scale.fill_(UE8M0_ONE)
            weights.fill_(1.0)
        elif dist == "heavy_tail":
            q_scale_bytes = torch.randint(
                110, 141, (num_q, heads, 4), dtype=torch.uint8, device="cuda", generator=gen
            )
            k_scale = torch.randint(
                110, 141, (num_k, 4), dtype=torch.uint8, device="cuda", generator=gen
            )
        elif dist != "random":
            raise ValueError(f"unsupported MXFP4 distribution {dist}")
        q_scale = q_scale_bytes.view(torch.int32).reshape(num_q, heads)
        return Inputs(q, k, k_scale, weights, q_scale, qkv)

    k_scale = rand(num_k) * 0.5 + 0.5
    if config == "qwen":
        if dist != "random":
            raise ValueError("the Qwen contract fixes the weights; only 'random' data is defined")
        weights.fill_(HEAD_DIM**-0.5)
        k_scale.fill_(1.0)
    if dist == "random":
        q, k = _fp8(randn(num_q, heads, HEAD_DIM)), _fp8(randn(num_k, HEAD_DIM))
    elif dist == "all_equal":  # every valid key has the same score
        q = _fp8(torch.ones(num_q, heads, HEAD_DIM, device="cuda"))
        k = _fp8(torch.ones(num_k, HEAD_DIM, device="cuda"))
        k_scale.fill_(1.0)
        weights.fill_(1.0)
    elif dist == "all_zero":  # negative dot products: ReLU zeroes every score
        q = _fp8(torch.ones(num_q, heads, HEAD_DIM, device="cuda"))
        k = _fp8(-torch.ones(num_k, HEAD_DIM, device="cuda"))
        k_scale.fill_(1.0)
    else:  # positive dot products with a controlled per-key magnitude
        q = _fp8(randn(num_q, heads, HEAD_DIM).abs())
        k = _fp8(randn(num_k, HEAD_DIM).abs())
        if dist == "few_high":  # a handful of keys far above everything else
            k_scale = torch.full((num_k,), 1e-3, device="cuda")
            high = torch.randperm(num_k, device="cuda", generator=gen)[:16]
            k_scale[high] = 1.0
        elif dist == "monotonic":  # scores grow with the key position
            k_scale = torch.linspace(0.1, 1.0, num_k, device="cuda")
        elif dist == "heavy_tail":  # log-normal key magnitudes
            k_scale = torch.exp((randn(num_k) * 2).clamp(-8, 8))
        else:
            raise ValueError(f"unknown distribution {dist}")
    return Inputs(q, k, k_scale, weights, None, qkv)


def _causal_ends(num_q, kv_prefix, ratio):
    return (kv_prefix + torch.arange(1, num_q + 1, device="cuda")) // ratio


def _reference_logits(inputs, ends, rows):
    """logits[i, j] = sum_h w[i, h] * relu(q[i, h] . k[j] * scale[j]); -inf outside [0, end)."""
    q, k, scale = inputs.reference_operands(rows)
    dots = torch.einsum("qhd,kd->qhk", q, k)
    if scale is not None:
        dots = dots * scale[None, None, :]
    logits = (dots.relu() * inputs.weights[rows][:, :, None]).sum(1)
    cols = torch.arange(k.shape[0], device=k.device)[None, :]
    return logits.masked_fill(cols >= ends[rows][:, None], float("-inf"))


def _sample_rows(ends, top_k):
    """Rows checked against the reference: all if few, else an even spread plus the rows around
    the top_k boundary (where the valid-key count crosses top_k)."""
    num_q = ends.shape[0]
    if num_q <= MAX_REF_ROWS:
        return torch.arange(num_q, device="cuda")
    spread = torch.linspace(0, num_q - 1, MAX_REF_ROWS - 16, device="cuda").long()
    near = (ends - top_k).abs() <= 2
    boundary = near.nonzero().flatten()[:16]
    return torch.unique(torch.cat((spread, boundary)))


def _assert_valid_topk(out, inputs, ends, top_k, context, logits_dtype=torch.float32):
    """Counts, window, padding and duplicates on every row; the score check on sampled rows."""
    sel = out.long()
    valid = sel >= 0
    expected = ends.clamp(max=top_k)
    assert torch.equal(valid.sum(1), expected), (
        f"{context}: selected counts {valid.sum(1).tolist()[:8]} != {expected.tolist()[:8]}"
    )
    assert bool(((sel == -1) | valid).all()), f"{context}: padding is not -1"
    assert bool(((sel < ends[:, None]) | ~valid).all()), f"{context}: index outside the window"
    ordered = sel.masked_fill(~valid, -1).sort(dim=1).values
    assert not bool(((ordered[:, 1:] == ordered[:, :-1]) & (ordered[:, 1:] >= 0)).any()), (
        f"{context}: duplicate indices"
    )

    rows_all = _sample_rows(ends, top_k)
    num_keys = inputs.k.shape[0]
    heads = inputs.weights.shape[1]
    block = max(1, min(64, REF_BLOCK_ELEMS // (heads * num_keys)))
    for begin in range(0, rows_all.numel(), block):
        rows = rows_all[begin : begin + block]
        long_rows = ends[rows] > top_k
        if not bool(long_rows.any()):
            continue  # all valid keys are selected; the checks above already prove it
        ref = _reference_logits(inputs, ends, rows)
        if logits_dtype != torch.float32:
            ref = ref.to(logits_dtype).float()
        kth = ref.topk(top_k, dim=1).values[:, -1]
        picked = ref.gather(1, sel[rows].clamp(min=0))
        slack = SCORE_SLACK[logits_dtype] * kth.abs().clamp(min=1.0)
        bad = valid[rows] & long_rows[:, None] & (picked < (kth - slack)[:, None])
        assert not bool(bad.any()), (
            f"{context}: selected a score below the k-th largest (rows {rows[:4].tolist()})"
        )


def _run(runner, inputs, kv_prefix, top_k, plan_cache=None):
    out = torch.full((inputs.q.shape[0], top_k), -2, dtype=torch.int32, device="cuda")
    runner.run_chunk(
        inputs.q,
        inputs.weights,
        inputs.k,
        inputs.k_scale,
        out,
        kv_prefix_len=kv_prefix,
        plan_cache={} if plan_cache is None else plan_cache,
        q_scale=inputs.q_scale,
    )
    return out


def _check(
    config,
    num_q,
    kv_prefix,
    *,
    seed=0,
    qkv="fp8",
    dist="random",
    logits_dtype=torch.float32,
    max_context_length=16384,
    runner=None,
    plan_cache=None,
):
    _, top_k, ratio = CONFIGS[config]
    runner = runner or _runner(config, max_context_length, logits_dtype=logits_dtype, qkv=qkv)
    inputs = _make_inputs(config, num_q, kv_prefix, seed, qkv, dist)
    out = _run(runner, inputs, kv_prefix, top_k, plan_cache)
    ends = _causal_ends(num_q, kv_prefix, ratio)
    context = f"{config} {qkv} {dist} q={num_q} prefix={kv_prefix} {logits_dtype}"
    _assert_valid_topk(out, inputs, ends, top_k, context, logits_dtype)


def _boundary_cases(config):
    """(num_q, prefix) whose rows' valid-key counts sit around top_k, incl. unaligned prefixes."""
    _, top_k, ratio = CONFIGS[config]
    base = top_k * ratio
    cases = [(8, base - 4 + shift) for shift in range(ratio if ratio > 1 else 1)]
    cases += [(3, base - 1), (3, base - 2), (6, base + ratio - 3)]
    return cases


# ---------------------------------------------------------------------------------------------
# GPU: negative controls -- the checker above must reject wrong selections
# ---------------------------------------------------------------------------------------------


def _reference_selection(inputs, ends, top_k):
    """An exact selection computed with torch.topk, -1 padded."""
    num_q = inputs.q.shape[0]
    out = torch.full((num_q, top_k), -1, dtype=torch.int32, device="cuda")
    for begin in range(0, num_q, 16):
        rows = torch.arange(begin, min(begin + 16, num_q), device="cuda")
        ref = _reference_logits(inputs, ends, rows)
        indices = ref.topk(top_k, dim=1).indices
        keep = ref.gather(1, indices) > float("-inf")
        out[rows] = torch.where(keep, indices, torch.full_like(indices, -1)).int()
    return out


@skip_not_sm100
@pytest.mark.parametrize(
    "qkv,corruption",
    [
        ("fp8", "worst_key"),
        ("fp8", "duplicate"),
        ("fp8", "outside_window"),
        ("fp8", "short_count"),
        ("fp8", "stale_padding"),
        ("mxfp4", "worst_key"),  # the only check that reads the dequantized MXFP4 reference
    ],
)
def test_checker_rejects_corrupted_selection(qkv, corruption):
    num_q, kv_prefix, top_k = 64, 4096, 2048
    inputs = _make_inputs("ds_v32", num_q, kv_prefix, seed=21, qkv=qkv)
    ends = _causal_ends(num_q, kv_prefix, 1)
    out = _reference_selection(inputs, ends, top_k)
    _assert_valid_topk(out, inputs, ends, top_k, "control")  # the untouched selection passes
    row = 5
    if corruption == "worst_key":
        ref = _reference_logits(inputs, ends, torch.tensor([row], device="cuda"))[0]
        worst = ref.masked_fill(ref == float("-inf"), float("inf")).argmin()
        assert not bool((out[row] == worst).any())
        out[row, 0] = worst
    elif corruption == "duplicate":
        out[row, 1] = out[row, 0]
    elif corruption == "outside_window":
        out[row, 0] = int(ends[row])
    elif corruption == "short_count":
        out[row, 0] = -1
    else:
        out[row, 0] = -2
    with pytest.raises(AssertionError):
        _assert_valid_topk(out, inputs, ends, top_k, corruption)


# ---------------------------------------------------------------------------------------------
# GPU: reference comparison
# ---------------------------------------------------------------------------------------------

SMALL_SHAPES = [
    (128, 0),  # first chunk: fewer valid keys than top_k for every row
    (256, 4096),  # crosses top_k inside the chunk
    (300, 5000),  # sizes that are not tile multiples
    (2048, 6144),  # many query rows
]


@skip_not_sm100
@pytest.mark.parametrize("qkv", ["fp8", "mxfp4"])
@pytest.mark.parametrize("num_q,kv_prefix", SMALL_SHAPES)
def test_matches_reference_ds_v32(qkv, num_q, kv_prefix):
    _check("ds_v32", num_q, kv_prefix, qkv=qkv, seed=1)


@skip_not_sm100
@pytest.mark.parametrize(
    "config,qkv",
    [
        ("glm_5_3", "fp8"),
        ("glm_5_3", "mxfp4"),
        ("dsv4_flash", "fp8"),
        ("dsv4_flash", "mxfp4"),
        ("dsv4_pro", "mxfp4"),
    ],
)
@pytest.mark.parametrize("num_q,kv_prefix", [(256, 4096), (1024, 4096)])
def test_matches_reference_other_configs(config, qkv, num_q, kv_prefix):
    _check(config, num_q, kv_prefix, qkv=qkv, seed=2)


@skip_not_sm100
@pytest.mark.parametrize("qkv", ["fp8", "mxfp4"])
@pytest.mark.parametrize("config", ["ds_v32", "glm_5_3", "dsv4_flash", "dsv4_pro"])
def test_top_k_boundary(config, qkv):
    """Valid-key counts equal to, just below and just above top_k, with unaligned prefixes."""
    for num_q, prefix in _boundary_cases(config):
        _check(config, num_q, prefix, qkv=qkv, seed=3, max_context_length=32768)


# The DSA indexer does not run the H4 kernels (see _VALIDATED_GEOMETRIES); these keep the
# vendored Qwen path (FP8, fixed-weight contract) from breaking unnoticed.
@skip_not_sm100
def test_matches_reference_qwen():
    """H4 / top_k 512 / ratio 4 with the fixed-weight Qwen contract (FP8)."""
    _check("qwen", 300, 5000, seed=14)


@skip_not_sm100
def test_qwen_top_k_boundary():
    for num_q, prefix in _boundary_cases("qwen"):
        _check("qwen", num_q, prefix, seed=15, max_context_length=32768)


@skip_not_sm100
@pytest.mark.parametrize("qkv", ["fp8", "mxfp4"])
@pytest.mark.parametrize("num_q,kv_prefix", [(1, 0), (1, 5000), (3, 100), (7, 9000)])
def test_tiny_query_counts(qkv, num_q, kv_prefix):
    _check("ds_v32", num_q, kv_prefix, qkv=qkv, seed=4)


@skip_not_sm100
@pytest.mark.parametrize("logits_dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "config,qkv",
    [
        ("ds_v32", "fp8"),
        ("ds_v32", "mxfp4"),
        ("glm_5_3", "fp8"),
        ("glm_5_3", "mxfp4"),
        ("dsv4_flash", "mxfp4"),
        ("dsv4_pro", "mxfp4"),
    ],
)
def test_low_precision_logits(config, logits_dtype, qkv):
    """Selecting over BF16 / FP16 stored scores must still be an exact top-k of those scores."""
    runner = _runner(config, 16384, logits_dtype=logits_dtype, qkv=qkv)
    assert runner.logits_dtype == logits_dtype
    _check(config, 256, 4096, qkv=qkv, seed=5, logits_dtype=logits_dtype, runner=runner)


@skip_not_sm100
@pytest.mark.parametrize("qkv", ["fp8", "mxfp4"])
def test_bf16_weights_are_accepted(qkv):
    runner = _runner("ds_v32", 16384, qkv=qkv)
    num_q, kv_prefix = 128, 2048
    inputs = _make_inputs("ds_v32", num_q, kv_prefix, 7, qkv)
    inputs.weights = inputs.weights.bfloat16()
    out = _run(runner, inputs, kv_prefix, 2048)
    inputs.weights = inputs.weights.float()
    _assert_valid_topk(out, inputs, _causal_ends(num_q, kv_prefix, 1), 2048, f"bf16 weights {qkv}")


# ---------------------------------------------------------------------------------------------
# GPU: score distributions that stress the selection paths
# ---------------------------------------------------------------------------------------------


@skip_not_sm100
@pytest.mark.parametrize("dist", FP8_DISTRIBUTIONS)
@pytest.mark.parametrize("config,num_q,kv_prefix", [("ds_v32", 256, 12288), ("glm_5_3", 128, 6000)])
def test_distributions_fp8(config, dist, num_q, kv_prefix):
    _check(config, num_q, kv_prefix, qkv="fp8", dist=dist, seed=8)


@skip_not_sm100
@pytest.mark.parametrize("dist", FP4_DISTRIBUTIONS)
@pytest.mark.parametrize(
    "config,num_q,kv_prefix", [("ds_v32", 256, 12288), ("dsv4_flash", 256, 8192)]
)
def test_distributions_mxfp4(config, dist, num_q, kv_prefix):
    _check(config, num_q, kv_prefix, qkv="mxfp4", dist=dist, seed=9)


# ---------------------------------------------------------------------------------------------
# GPU: tiling, plan reuse and chunk interleaving
# ---------------------------------------------------------------------------------------------


@skip_not_sm100
@pytest.mark.parametrize("qkv", ["fp8", "mxfp4"])
def test_query_tiling_matches_reference(monkeypatch, qkv):
    """A chunk larger than max_chunk_size is processed in tiles with shifted K prefixes."""
    runner = _runner("ds_v32", 16384, max_chunk=256, monkeypatch=monkeypatch, qkv=qkv)
    _check("ds_v32", 700, 3000, qkv=qkv, seed=3, runner=runner)


@skip_not_sm100
@pytest.mark.parametrize("qkv", ["fp8", "mxfp4"])
def test_full_launch_rows_bind_the_caller_tensors_without_staging(monkeypatch, qkv):
    """Tiles that fill the launch read Q, weights and Q scales and write the selection in place;
    only the shorter last tile goes through the staging buffers and the output copy."""
    runner = _runner("ds_v32", 16384, max_chunk=256, monkeypatch=monkeypatch, qkv=qkv)
    plan = runner._wrapper._capacity_plan
    staged, direct = [], []
    real_stage, real_direct = plan._state.stage_payload, plan._run_direct

    def stage_spy(q, *args, **kwargs):
        staged.append(q.shape[0])
        return real_stage(q, *args, **kwargs)

    def direct_spy(*args, **kwargs):
        direct.append(True)
        return real_direct(*args, **kwargs)

    monkeypatch.setattr(plan._state, "stage_payload", stage_spy)
    monkeypatch.setattr(plan, "_run_direct", direct_spy)
    _check("ds_v32", 256, 3000, qkv=qkv, seed=41, runner=runner)
    assert (staged, len(direct)) == ([], 1)
    # Two full tiles, then a 188-row tile; the staged buffers hold stale rows from before.
    _check("ds_v32", 700, 3000, qkv=qkv, seed=42, runner=runner)
    assert (staged, len(direct)) == ([188], 3)
    _check("ds_v32", 256, 5000, qkv=qkv, seed=43, runner=runner)
    assert (staged, len(direct)) == ([188], 4)


@skip_not_sm100
@pytest.mark.parametrize("qkv", ["fp8", "mxfp4"])
def test_staging_buffers_passed_back_take_the_staged_path(monkeypatch, qkv):
    """Inputs inside the plan's own storage (here the staging buffers themselves) are not
    bound in place; the staged path runs them without a copy and gets the same selection."""
    runner = _runner("ds_v32", 16384, max_chunk=256, monkeypatch=monkeypatch, qkv=qkv)
    num_q, kv_prefix, top_k = 256, 3000, CONFIGS["ds_v32"][1]
    inputs = _make_inputs("ds_v32", num_q, kv_prefix, 45, qkv)
    expected = _run(runner, inputs, kv_prefix, top_k)
    plan = runner._wrapper._capacity_plan
    state = plan._state
    q = inputs.q.view(torch.uint8) if qkv == "mxfp4" else inputs.q
    state.q.copy_(q)
    state.weights.copy_(inputs.weights)
    q_scale = None
    if qkv == "mxfp4":
        q_scale = plan._q_scale
        q_scale.copy_(inputs.q_scale.contiguous().view(torch.int32).reshape(q_scale.shape))
    direct = []
    real_direct = plan._run_direct
    monkeypatch.setattr(
        plan, "_run_direct", lambda *a, **k: direct.append(1) or real_direct(*a, **k)
    )
    out = torch.full_like(expected, -2)
    runner._wrapper.run(
        state.q,
        state.weights,
        runner._k_stage,
        k_scale=runner._k_scale_stage,
        out=out,
        q_scale=q_scale,
    )
    assert direct == []
    # Row order within a selection is not part of the contract.
    assert torch.equal(out.sort(dim=1).values, expected.sort(dim=1).values)


@skip_not_sm100
@pytest.mark.parametrize("qkv", ["fp8", "mxfp4"])
def test_plan_is_issued_once_per_chunk_and_reissued_when_the_chunk_changes(monkeypatch, qkv):
    """Layers of one chunk share a plan; a different chunk (or returning to one) needs a new
    plan, and every layer's result stays correct either way."""
    runner = _runner("ds_v32", 16384, qkv=qkv)
    planned = []
    real_plan = runner._wrapper.plan

    def spy(*args, **kwargs):
        planned.append(kwargs.get("num_query_rows"))
        return real_plan(*args, **kwargs)

    monkeypatch.setattr(runner._wrapper, "plan", spy)
    runner._planned_key = None
    chunks = {"a": (256, 4096), "b": (192, 6000)}
    caches = {"a": {}, "b": {}}
    for step, which in enumerate(["a", "a", "a", "b", "b", "a"]):
        num_q, kv_prefix = chunks[which]
        _check(
            "ds_v32",
            num_q,
            kv_prefix,
            qkv=qkv,
            seed=200 + step,
            runner=runner,
            plan_cache=caches[which],
        )
    assert planned == [256, 192, 256], planned


@skip_not_sm100
@pytest.mark.parametrize("qkv", ["fp8", "mxfp4"])
def test_wrapper_rejects_inputs_that_alias_its_staging_buffers(qkv):
    """The staging copies must refuse a source overlapping a staging buffer, except the
    buffer itself (no copy needed). Guards the cached owner-range alias check."""
    runner = _runner("ds_v32", 16384, qkv=qkv)
    num_q, kv_prefix = 64, 4096
    _check("ds_v32", num_q, kv_prefix, qkv=qkv, seed=30, runner=runner)  # plans num_q rows
    state = runner._wrapper._capacity_plan._state
    inputs = _make_inputs("ds_v32", num_q, kv_prefix, 30, qkv)
    q_scale = None if qkv == "fp8" else inputs.q_scale.contiguous()

    def run(q):
        runner._wrapper.run(
            q,
            inputs.weights,
            runner._k_stage,
            k_scale=runner._k_scale_stage,
            q_scale=q_scale,
        )

    run(state.q[:num_q])  # the staging buffer itself: allowed
    with pytest.raises(ValueError, match="cross-alias"):
        run(state.q[1 : num_q + 1])  # overlaps the staging buffer at an offset


@skip_not_sm100
@pytest.mark.parametrize("qkv,num_keys", [("fp8", 10112), ("mxfp4", 10240)])
def test_key_storage_is_padded_to_whole_tiles(qkv, num_keys):
    """A max length that is not a tile multiple rounds the K storage up (FP8 128, MXFP4 256)."""
    runner = ThrowawayPrefillTopK(
        n_heads=64,
        index_topk=2048,
        compress_ratio=1,
        max_context_length=10000,
        device=torch.device("cuda"),
        qkv_dtype=qkv,
    )
    assert runner.num_keys == num_keys
    assert runner.can_run(10000) and not runner.can_run(num_keys + 1)
    _check("ds_v32", 256, 8700, qkv=qkv, seed=11, runner=runner)


# ---------------------------------------------------------------------------------------------
# GPU: Q = 4096 / 8192 with large K (reference rows are sampled)
# ---------------------------------------------------------------------------------------------

LARGE_SHAPES = [(8192, 0), (4096, 28672)]
if LARGE:
    LARGE_SHAPES += [(8192, 57344), (8192, 122880)]


@skip_not_sm100
@pytest.mark.parametrize("num_q,kv_prefix", LARGE_SHAPES)
@pytest.mark.parametrize(
    "config,qkv", [("ds_v32", "fp8"), ("glm_5_3", "fp8"), ("dsv4_flash", "mxfp4")]
)
def test_large_shapes(config, qkv, num_q, kv_prefix):
    _check(config, num_q, kv_prefix, qkv=qkv, seed=12, max_context_length=131072)


# Kernel variants are chosen from the runner capacity (max length // compress ratio), so each
# model is also checked at its own maximum length, with the query rows at the end of the window.
PRODUCTION_LENGTHS = [
    ("ds_v32", "fp8", 163840),
    ("glm_5_3", "fp8", 202752),
    ("dsv4_flash", "mxfp4", 1048576),
    ("dsv4_pro", "mxfp4", 1048576),
]


@skip_not_sm100
@pytest.mark.parametrize("num_q", [1024, 8192] if LARGE else [1024])
@pytest.mark.parametrize("config,qkv,max_context_length", PRODUCTION_LENGTHS)
def test_production_max_lengths(config, qkv, max_context_length, num_q):
    _check(
        config,
        num_q,
        max_context_length - num_q,
        qkv=qkv,
        seed=18,
        max_context_length=max_context_length,
    )


@skip_not_sm100
def test_large_shape_bf16_logits():
    _check(
        "ds_v32",
        8192,
        24576,
        seed=13,
        logits_dtype=torch.bfloat16,
        max_context_length=131072,
    )


@skip_not_sm100
def test_workspace_oom_disables_the_runner(monkeypatch):
    """Running out of memory while building the kernel workspace falls back instead of failing."""
    runner = ThrowawayPrefillTopK(
        n_heads=64,
        index_topk=2048,
        compress_ratio=1,
        max_context_length=16384,
        device=torch.device("cuda"),
    )
    calls = []

    def prepare_pool(*args, **kwargs):
        calls.append(args)
        raise torch.OutOfMemoryError("CUDA out of memory")

    monkeypatch.setattr(runner._wrapper, "prepare_pool", prepare_pool)
    inputs = _make_inputs("ds_v32", 64, 4096, seed=19)
    out = torch.full((64, 2048), -2, dtype=torch.int32, device="cuda")
    for _ in range(2):
        handled = runner.run_chunk(
            inputs.q,
            inputs.weights,
            inputs.k,
            inputs.k_scale,
            out,
            kv_prefix_len=4096,
            plan_cache={},
        )
        assert handled is False
    assert len(calls) == 1  # disabled after the first failure
    assert bool((out == -2).all())  # nothing written
    assert runner._k_stage is None  # buffers released
    assert not runner.ready


@skip_not_sm100
@pytest.mark.parametrize("config,qkv", [("ds_v32", "fp8"), ("dsv4_flash", "mxfp4")])
def test_warmup_builds_the_kernels_before_the_first_chunk(monkeypatch, config, qkv):
    """warmup() compiles the kernels and allocates the workspace, so the first real chunk does
    neither, and its selection is still exact."""
    heads, top_k, ratio = CONFIGS[config]
    runner = ThrowawayPrefillTopK(
        n_heads=heads,
        index_topk=top_k,
        compress_ratio=ratio,
        max_context_length=65536,
        device=torch.device("cuda"),
        qkv_dtype=qkv,
    )
    assert runner.warmup()
    assert runner.warmup()  # already warm: no second build

    def prepare_pool(*args, **kwargs):
        pytest.fail("the first chunk after warmup must not build the kernels")

    monkeypatch.setattr(runner._wrapper, "prepare_pool", prepare_pool)
    _check(config, 256, 12288, qkv=qkv, seed=41, runner=runner)


# ---------------------------------------------------------------------------------------------
# Dispatch (CPU only): Indexer._run_throwaway_chunk with a fake runner.
# ---------------------------------------------------------------------------------------------


class _FakeRunner:
    num_keys = 16
    compress_ratio = 1

    def __init__(self, can_run=True, ran=True):
        self._can_run = can_run
        self._ran = ran
        self.calls = []

    def can_run(self, num_k_rows):
        return self._can_run

    def run_chunk(self, q, weights, k, k_scale, out, *, kv_prefix_len, plan_cache, q_scale=None):
        self.calls.append((q, weights, k, k_scale, out, kv_prefix_len, plan_cache, q_scale))
        return self._ran


def _fake_indexer(runner, enabled=True, failed=False, min_context=0):
    return SimpleNamespace(
        _use_throwaway_topk=enabled,
        _throwaway_topk_failed=failed,
        _throwaway_topk_min_context=min_context,
        _throwaway_topk=runner,
        n_heads=4,
        index_topk=8,
    )


def _chunk(kv_prefix_len, token_start=3):
    return IndexerPrefillChunkMetadata(
        cu_seqlen_ks=None,
        cu_seqlen_ke=None,
        token_start=token_start,
        token_end=token_start + 10,
        k_token_start=0,
        k_token_end=20,
        kv_prefix_len=kv_prefix_len,
    )


def _call(indexer, chunk, q_begin=0, q_end=10, q_scale=None, compress_ratios=(1,)):
    q = torch.arange(32 * 4 * 2, dtype=torch.float32).reshape(32, 4, 2)
    weights = torch.arange(32 * 4, dtype=torch.float32).reshape(32, 4)
    k = torch.zeros(20, 2)
    k_scale = torch.ones(20)
    out = torch.zeros(32, 8, dtype=torch.int32)
    metadata = SimpleNamespace(compress_ratios=list(compress_ratios))
    handled = Indexer._run_throwaway_chunk(
        indexer, metadata, chunk, q, weights, k, k_scale, q_begin, q_end, out, q_scale=q_scale
    )
    return handled, q, weights, out


@pytest.mark.parametrize(
    "indexer_kwargs,kv_prefix_len",
    [
        ({"enabled": False}, 5),  # not opted in
        ({"failed": True}, 5),  # creation failed earlier
        ({}, None),  # multi-request chunk
    ],
)
def test_dispatch_falls_back(indexer_kwargs, kv_prefix_len):
    runner = _FakeRunner()
    handled, *_ = _call(_fake_indexer(runner, **indexer_kwargs), _chunk(kv_prefix_len))
    assert handled is False
    assert runner.calls == []


def test_dispatch_falls_back_when_the_runner_fails(monkeypatch):
    keys = []
    monkeypatch.setattr(indexer_module.logger, "info_once", lambda *msg, key: keys.append(key))
    runner = _FakeRunner(ran=False)
    handled, *_ = _call(_fake_indexer(runner), _chunk(5))
    assert handled is False
    assert len(runner.calls) == 1
    assert keys == []  # not reported as fused


def test_dispatch_falls_back_when_keys_do_not_fit():
    runner = _FakeRunner(can_run=False)
    handled, *_ = _call(_fake_indexer(runner), _chunk(5))
    assert handled is False
    assert runner.calls == []


@pytest.mark.parametrize(
    "kv_prefix_len,q_end,handled",
    [
        (93, 7, True),  # context of the last row == min_context
        (92, 7, False),  # one token short
        (0, 10, False),  # first chunk of a short request
    ],
)
def test_dispatch_routes_short_context_to_unfused_path(kv_prefix_len, q_end, handled):
    runner = _FakeRunner()
    result, *_ = _call(
        _fake_indexer(runner, min_context=100), _chunk(kv_prefix_len), q_begin=2, q_end=q_end
    )
    assert result is handled
    assert len(runner.calls) == int(handled)


class _WarmableRunner:
    def __init__(self, ready=True):
        self._ready = ready
        self.warmups = 0

    def warmup(self):
        self.warmups += 1
        return self._ready


@pytest.mark.parametrize(
    "enabled,failed,ready,expected,warmups",
    [
        (True, False, True, True, 1),
        (True, False, False, False, 1),  # the workspace did not fit
        (False, False, True, False, 0),  # not opted in
        (True, True, True, False, 0),  # creation failed earlier
    ],
)
def test_prewarm_builds_the_shared_runner_and_warms_it(
    monkeypatch, enabled, failed, ready, expected, warmups
):
    runner = _WarmableRunner(ready=ready)
    built = []

    def get_runner(**kwargs):
        built.append(kwargs)
        return runner

    monkeypatch.setattr(indexer_module, "get_throwaway_prefill_topk", get_runner)
    indexer = _fake_indexer(None, enabled=enabled, failed=failed)
    indexer.compress_ratio = 4
    indexer.use_fp4 = True
    indexer._build_throwaway_topk = Indexer._build_throwaway_topk.__get__(indexer)
    indexer._throwaway_query_rows = Indexer._throwaway_query_rows.__get__(indexer)
    indexer._indexer_max_chunk_size, indexer._q_split_threshold = None, 8192
    prewarmed = Indexer.prewarm_throwaway_topk(
        indexer, 1 << 20, torch.device("cuda", 0), 8448, _mapping(8, 2)
    )
    assert prewarmed is expected
    assert runner.warmups == warmups
    if warmups:
        assert built[0]["compress_ratio"] == 4
        assert built[0]["max_context_length"] == 1 << 20
        assert built[0]["qkv_dtype"] == "mxfp4"
        assert built[0]["device"] == torch.device("cuda", 0)
        assert built[0]["max_query_rows"] == 1056  # this rank's share of a full chunk
        assert indexer._throwaway_topk is runner  # the forward reuses the warmed runner


def test_dispatch_rejects_a_runner_built_for_another_compress_ratio():
    runner = _FakeRunner()
    with pytest.raises(RuntimeError, match="compress_ratio"):
        _call(_fake_indexer(runner), _chunk(5), compress_ratios=(4,))
    assert runner.calls == []


def test_tuning_env_is_validated_up_front(monkeypatch):
    for name in (THROWAWAY_MAX_CHUNK_ENV, THROWAWAY_LOGITS_DTYPE_ENV, THROWAWAY_MIN_CONTEXT_ENV):
        monkeypatch.delenv(name, raising=False)
    validate_throwaway_topk_env()
    assert throwaway_topk_max_chunk() == 8192
    assert throwaway_topk_logits_dtype() == torch.float32
    monkeypatch.setenv(THROWAWAY_MAX_CHUNK_ENV, "4096")
    monkeypatch.setenv(THROWAWAY_LOGITS_DTYPE_ENV, "bf16")
    assert throwaway_topk_max_chunk() == 4096
    assert throwaway_topk_logits_dtype() == torch.bfloat16
    for name, bad in (
        (THROWAWAY_MAX_CHUNK_ENV, "0"),
        (THROWAWAY_MAX_CHUNK_ENV, "8K"),
        (THROWAWAY_LOGITS_DTYPE_ENV, "fp8"),
        (THROWAWAY_MIN_CONTEXT_ENV, "-1"),
    ):
        with monkeypatch.context() as m:
            m.setenv(name, bad)
            with pytest.raises(ValueError, match=name):
                validate_throwaway_topk_env()


def test_supported_heads_match_validated_geometries(monkeypatch):
    monkeypatch.setattr(throwaway_topk_module, "IS_CUTLASS_DSL_AVAILABLE", True)
    assert throwaway_topk_supported(64, HEAD_DIM)
    assert throwaway_topk_supported(32, HEAD_DIM)
    # The H4 kernels are validated only under the Qwen fixed-weight contract.
    assert not throwaway_topk_supported(4, HEAD_DIM)
    assert not throwaway_topk_supported(64, 64)


def test_min_context_env(monkeypatch):
    monkeypatch.delenv(THROWAWAY_MIN_CONTEXT_ENV, raising=False)
    assert throwaway_topk_min_context() == 65536
    monkeypatch.setenv(THROWAWAY_MIN_CONTEXT_ENV, "0")
    assert throwaway_topk_min_context() == 0
    for bad in ("-1", "64K"):
        monkeypatch.setenv(THROWAWAY_MIN_CONTEXT_ENV, bad)
        with pytest.raises(ValueError, match=THROWAWAY_MIN_CONTEXT_ENV):
            throwaway_topk_min_context()


_RUNNER_ARGS = dict(
    n_heads=64,
    index_topk=2048,
    compress_ratio=1,
    max_context_length=8192,
    device=torch.device("cuda", 0),
)


def _fresh_runner_cache(monkeypatch):
    monkeypatch.setattr(throwaway_topk_module, "_RUNNERS", weakref.WeakValueDictionary())
    monkeypatch.setattr(throwaway_topk_module, "_FAILED", set())


def test_runner_is_shared_per_configuration(monkeypatch):
    built = []

    class _Runner:
        def __init__(self, **kwargs):
            built.append(kwargs)

    monkeypatch.setattr(throwaway_topk_module, "ThrowawayPrefillTopK", _Runner)
    _fresh_runner_cache(monkeypatch)
    first = get_throwaway_prefill_topk(**_RUNNER_ARGS)
    assert get_throwaway_prefill_topk(**_RUNNER_ARGS) is first  # every layer gets one runner
    other = get_throwaway_prefill_topk(**{**_RUNNER_ARGS, "qkv_dtype": "mxfp4"})
    assert other is not first
    assert len(built) == 2


def test_runner_is_released_with_the_model(monkeypatch):
    """The cache does not keep a runner (and its GPU buffers) alive once no layer holds it."""
    built = []

    class _Runner:
        def __init__(self, **kwargs):
            built.append(kwargs)

    monkeypatch.setattr(throwaway_topk_module, "ThrowawayPrefillTopK", _Runner)
    _fresh_runner_cache(monkeypatch)
    runner = get_throwaway_prefill_topk(**_RUNNER_ARGS)
    released = weakref.ref(runner)
    del runner  # the model and its indexers are gone
    gc.collect()
    assert released() is None
    assert get_throwaway_prefill_topk(**_RUNNER_ARGS) is not None  # a new model builds anew
    assert len(built) == 2


@pytest.mark.parametrize(
    "geometry",
    [
        dict(n_heads=4, index_topk=512, compress_ratio=4),  # needs the Qwen weight contract
        dict(n_heads=32, index_topk=512, compress_ratio=4),  # GLM-5.3-Flash, not validated
        dict(n_heads=64, index_topk=2048, compress_ratio=4),
    ],
)
def test_runner_rejects_unvalidated_geometries(monkeypatch, geometry):
    built, warnings = [], []
    monkeypatch.setattr(
        throwaway_topk_module, "ThrowawayPrefillTopK", lambda **kwargs: built.append(kwargs)
    )
    _fresh_runner_cache(monkeypatch)
    monkeypatch.setattr(throwaway_topk_module.logger, "warning", warnings.append)
    args = {**_RUNNER_ARGS, **geometry}
    assert get_throwaway_prefill_topk(**args) is None
    assert get_throwaway_prefill_topk(**args) is None
    assert built == []
    assert len(warnings) == 1


def test_runner_build_failure_is_cached(monkeypatch):
    built, warnings = [], []

    class _Unsupported:
        def __init__(self, **kwargs):
            built.append(kwargs)
            raise ValueError("unsupported geometry")

    monkeypatch.setattr(throwaway_topk_module, "ThrowawayPrefillTopK", _Unsupported)
    _fresh_runner_cache(monkeypatch)
    monkeypatch.setattr(throwaway_topk_module.logger, "warning", warnings.append)
    assert get_throwaway_prefill_topk(**_RUNNER_ARGS) is None
    assert get_throwaway_prefill_topk(**_RUNNER_ARGS) is None
    assert len(built) == 1
    assert len(warnings) == 1


def test_runner_build_oom_falls_back(monkeypatch):
    """Running out of memory while allocating the runner's buffers keeps the unfused path."""
    built, warnings = [], []

    class _TooLarge:
        def __init__(self, **kwargs):
            built.append(kwargs)
            raise torch.OutOfMemoryError("CUDA out of memory")

    monkeypatch.setattr(throwaway_topk_module, "ThrowawayPrefillTopK", _TooLarge)
    _fresh_runner_cache(monkeypatch)
    monkeypatch.setattr(throwaway_topk_module.logger, "warning", warnings.append)
    assert get_throwaway_prefill_topk(**_RUNNER_ARGS) is None
    assert get_throwaway_prefill_topk(**_RUNNER_ARGS) is None  # not retried by later chunks
    assert len(built) == 1
    assert len(warnings) == 1


def test_runner_build_oom_releases_the_partial_build_before_emptying_the_cache(monkeypatch):
    """The cache is emptied only once the partially built runner has been released."""
    partial, alive_at_empty_cache = [], []

    class _PartialBuild:
        def __init__(self, **kwargs):
            partial.append(weakref.ref(self))
            raise torch.OutOfMemoryError("CUDA out of memory")

    monkeypatch.setattr(throwaway_topk_module, "ThrowawayPrefillTopK", _PartialBuild)
    _fresh_runner_cache(monkeypatch)
    monkeypatch.setattr(throwaway_topk_module.logger, "warning", lambda message: None)
    monkeypatch.setattr(
        throwaway_topk_module.torch.cuda,
        "empty_cache",
        lambda: alive_at_empty_cache.append(partial[0]() is not None),
    )
    assert get_throwaway_prefill_topk(**_RUNNER_ARGS) is None
    assert alive_at_empty_cache == [False]


class _ReadyRunner:
    def __init__(self, compress_ratio=4, num_keys=1 << 18, ready=True):
        self.compress_ratio = compress_ratio
        self.num_keys = num_keys
        self.ready = ready

    def can_run(self, num_k_rows):
        return 0 < num_k_rows <= self.num_keys


def _model_with_indexers(*specs):
    """A module tree holding Indexers with only the fields the logits bound reads.

    Each spec is ``(enabled, failed, runner, min_context[, local])``; ``local=False`` marks an
    indexer of another pipeline stage.
    """
    model = torch.nn.Module()
    for i, (enabled, failed, runner, min_context, *local) in enumerate(specs):
        indexer = Indexer.__new__(Indexer)
        torch.nn.Module.__init__(indexer)
        # Layers of other pipeline stages keep their modules but not their parameters.
        indexer.wq_b = SimpleNamespace(weight=torch.empty(0) if local != [False] else None)
        indexer._use_throwaway_topk = enabled
        indexer._throwaway_topk_failed = failed
        indexer._throwaway_topk = runner
        indexer._throwaway_topk_min_context = min_context
        model.add_module(f"indexer{i}", indexer)
    return model


_MAX_SEQ_LEN = 1 << 20  # 1M tokens, 256K compressed keys at ratio 4


@pytest.mark.parametrize(
    "max_num_tokens,expected",
    [
        # 8192 rows x (64K + 8K) / 4 keys = 151M elements; below the 2^28 packed cap.
        (8192, 1 << 28),
        # 32768 rows x (64K + 32K) / 4 keys = 768M elements; above the packed cap.
        (32768, 32768 * (((1 << 16) + 32768) // 4)),
    ],
)
def test_unfused_logits_bound_covers_short_and_packed_chunks(max_num_tokens, expected):
    model = _model_with_indexers(
        (True, False, _ReadyRunner(), 1 << 16), (True, False, _ReadyRunner(), 1 << 16)
    )
    # One decode token: 256K keys, far below the prefill terms.
    bound = throwaway_unfused_logits_elem_bound(model, max_num_tokens, _MAX_SEQ_LEN, 1)
    assert bound == expected


def test_unfused_logits_bound_caps_prefill_terms_at_the_full_tile():
    model = _model_with_indexers((True, False, _ReadyRunner(compress_ratio=1), 1 << 16))
    # 64 rows x 4096 tokens = 262144 elements, below the short-context term.
    assert throwaway_unfused_logits_elem_bound(model, 64, 4096, 1) == 64 * 4096


def test_unfused_logits_bound_covers_decode_logits():
    """Decode logits are not tiled by the budget, so they bound the reserve from below."""
    model = _model_with_indexers((True, False, _ReadyRunner(), 1 << 16))
    # 2048 decode tokens x (256K compressed keys + column padding), above the 2^28 prefill term.
    assert throwaway_unfused_logits_elem_bound(model, 8192, _MAX_SEQ_LEN, 2048) == 2048 * (
        (1 << 18) + indexer_module._PAGED_LOGITS_COLUMN_SLACK
    )


def test_unfused_logits_bound_ignores_indexers_of_other_pipeline_stages():
    """Indexers of other stages have no runner on this rank and allocate no logits here."""
    model = _model_with_indexers(
        (True, False, _ReadyRunner(), 1 << 16), (True, False, None, 1 << 16, False)
    )
    assert throwaway_unfused_logits_elem_bound(model, 8192, _MAX_SEQ_LEN, 1) == 1 << 28


@pytest.mark.parametrize(
    "specs",
    [
        [],  # no DSA indexer
        [(False, False, None, 0)],  # fused path not requested
        [(True, True, None, 0)],  # runner build failed
        [(True, False, None, 0)],  # runner not built yet (e.g. no profiling run)
        [(True, False, _ReadyRunner(ready=False), 0)],  # workspace not allocated / disabled
        [(True, False, _ReadyRunner(num_keys=1024), 0)],  # keys of long chunks do not fit
        [(True, False, _ReadyRunner(), 0), (True, True, None, 0)],  # one layer failed
        [(True, False, _ReadyRunner(), 0), (False, False, None, 0)],  # one layer stays unfused
        [(False, False, None, 0, False)],  # only indexers of other pipeline stages
    ],
)
def test_unfused_logits_bound_needs_a_ready_runner_on_every_layer(specs):
    model = _model_with_indexers(*specs)
    assert throwaway_unfused_logits_elem_bound(model, 8192, _MAX_SEQ_LEN, 32) is None


def _chunk_metadata(cached, new_tokens, compress_ratio):
    """Host-side fields of the attention metadata that prepare_one_prefill_chunk reads."""

    def indptr(values):
        return torch.tensor([0] + list(itertools.accumulate(values)), dtype=torch.int32)

    return SimpleNamespace(
        cu_seqlen_ks=torch.zeros(1, dtype=torch.int32),
        compress_ratios=[compress_ratio],
        host_ctx_cached_token_indptr=indptr(cached),
        host_ctx_kv_indptr=indptr([c + n for c, n in zip(cached, new_tokens)]),
    )


def test_prepare_one_prefill_chunk_sets_kv_prefix_len():
    """kv_prefix_len counts original (uncompressed) tokens, cached ones included, and matches the
    compressed causal window the unfused path derives from cu_seqlen_ke."""
    ratio, cached, new_tokens = 4, [37, 1000], [50, 300]
    metadata = _chunk_metadata(cached, new_tokens, ratio)
    # Q block [120, 250) of request 1, whose tokens start at 50 in the batch.
    chunk = Indexer.prepare_one_prefill_chunk(metadata, [(1, 120, 250, 50)])
    assert chunk.kv_prefix_len == 1000 + 120
    rows = torch.arange(250 - 120)
    assert torch.equal(chunk.cu_seqlen_ke.cpu(), ((chunk.kv_prefix_len + rows + 1) // ratio).int())
    assert chunk.k_token_end - chunk.k_token_start == (1000 + 250) // ratio
    # Several whole requests in one chunk: not eligible for the fused path.
    multi = Indexer.prepare_one_prefill_chunk(metadata, [(0, 0, 50, 0), (1, 0, 300, 50)])
    assert multi.kv_prefix_len is None


def test_dispatch_slices_rows_and_shifts_prefix():
    runner = _FakeRunner()
    chunk = _chunk(kv_prefix_len=100, token_start=3)
    handled, q, weights, out = _call(_fake_indexer(runner), chunk, q_begin=2, q_end=7)
    assert handled is True
    assert len(runner.calls) == 1
    q_arg, w_arg, _, _, out_arg, prefix, plan_cache, q_scale_arg = runner.calls[0]
    # chunk rows start at token_start=3, local rows [2, 7) -> global rows [5, 10)
    assert torch.equal(q_arg, q[5:10])
    assert torch.equal(w_arg, weights[5:10])
    assert out_arg.data_ptr() == out[5:10].data_ptr()
    assert prefix == 100 + 2
    assert plan_cache is chunk.throwaway_plans
    assert q_scale_arg is None


def test_dispatch_logs_each_outcome_once(monkeypatch):
    keys = []
    monkeypatch.setattr(indexer_module.logger, "info_once", lambda *msg, key: keys.append(key))
    _call(_fake_indexer(_FakeRunner(), enabled=False), _chunk(5))  # not opted in: silent
    assert keys == []
    _call(_fake_indexer(_FakeRunner()), _chunk(None))
    _call(_fake_indexer(_FakeRunner(), min_context=100), _chunk(5))
    _call(_fake_indexer(_FakeRunner(can_run=False)), _chunk(5))
    _call(_fake_indexer(_FakeRunner()), _chunk(5))
    assert keys == [
        "throwaway_topk_multi_request",
        "throwaway_topk_short_context",
        "throwaway_topk_too_many_keys",
        "throwaway_topk_fused",
    ]


def test_dispatch_slices_fp4_query_scales_with_the_rows():
    runner = _FakeRunner()
    q_scale = torch.arange(32 * 4, dtype=torch.int32).reshape(32, 4)
    handled, *_ = _call(
        _fake_indexer(runner), _chunk(kv_prefix_len=0, token_start=3), 2, 7, q_scale=q_scale
    )
    assert handled is True
    assert torch.equal(runner.calls[0][7], q_scale[5:10])


def test_launch_rows_follow_the_full_chunk_rows(monkeypatch):
    monkeypatch.delenv(THROWAWAY_MAX_CHUNK_ENV, raising=False)
    assert throwaway_topk_launch_rows(64, None) == 8192  # unknown chunk rows: the cap
    assert throwaway_topk_launch_rows(64, 4096) == 4096
    assert throwaway_topk_launch_rows(64, 1056) == 1056
    assert throwaway_topk_launch_rows(64, 16384) == 8192  # larger chunks run in tiles
    assert throwaway_topk_launch_rows(64, 1057) == 1060  # whole row tiles of 4 rows
    assert throwaway_topk_launch_rows(32, 1003) == 1008  # 8 rows
    assert throwaway_topk_launch_rows(4, 100) == 128  # 64 rows
    monkeypatch.setenv(THROWAWAY_MAX_CHUNK_ENV, "2048")
    assert throwaway_topk_launch_rows(64, 4096) == 2048


def _mapping(tp_size, tp_rank=0, attention_dp=False):
    return SimpleNamespace(tp_size=tp_size, tp_rank=tp_rank, enable_attention_dp=attention_dp)


@pytest.mark.parametrize(
    "max_num_tokens,chunk_size,mapping,rows",
    [
        (4096, None, None, 4096),
        (4096, None, _mapping(8), 4096),  # below the split threshold: every rank has all rows
        (4096, 2048, _mapping(1), 2048),  # the indexer chunk size bounds a chunk
        (8448, None, _mapping(8, 3), 1056),  # split over the attention-TP ranks
        (8450, None, _mapping(8, 7), 1057),
        (8448, None, _mapping(8, 3, attention_dp=True), 8448),
    ],
)
def test_throwaway_query_rows(max_num_tokens, chunk_size, mapping, rows):
    indexer = SimpleNamespace(_indexer_max_chunk_size=chunk_size, _q_split_threshold=8192)
    assert Indexer._throwaway_query_rows(indexer, max_num_tokens, mapping) == rows
