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
"""DSA indexer prefill logits with BF16 head reduce (MXFP4 keys, DeepGEMM's BF16-reduce epilogue on the DeepGEMM
kernel or on the CuTe DSL FP4 MQA logits kernel)."""

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.attention.backends.sparse.dsa import indexer as idx
from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE
from tensorrt_llm._utils import get_sm_version

HEADS, DIM, TOPK = 32, 128, 2048


def _skip_unless_sm100():
    if not torch.cuda.is_available() or get_sm_version() not in (100, 103):
        pytest.skip("MXFP4 MQA logits require SM100 / SM103")


def _problem(rows, keys, seed):
    """MXFP4 Q [rows, 32, 64 B] + per-32 UE8M0 scales, MXFP4 K [keys, 64 B] + scales, FP32 head weights, causal
    windows: row i sees keys [0, keys - rows + i + 1)."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    q = torch.randint(
        -128, 128, (rows, HEADS, DIM // 2), generator=g, device="cuda", dtype=torch.int32
    )
    k = torch.randint(-128, 128, (keys, DIM // 2), generator=g, device="cuda", dtype=torch.int32)
    qs = torch.randint(124, 130, (rows, HEADS, 4), generator=g, device="cuda", dtype=torch.int32)
    ks = torch.randint(124, 130, (keys, 4), generator=g, device="cuda", dtype=torch.int32)
    w = torch.randn(rows, HEADS, generator=g, device="cuda") * 0.1
    ke = torch.arange(keys - rows + 1, keys + 1, device="cuda", dtype=torch.int32)
    kst = torch.zeros(rows, device="cuda", dtype=torch.int32)
    return (
        q.to(torch.int8).view(torch.float8_e4m3fn),
        qs.to(torch.uint8),
        k.to(torch.int8).view(torch.float8_e4m3fn),
        ks.to(torch.uint8).view(torch.int32),
        w,
        kst,
        ke,
    )


def _packed_problem(rows, keys, reqs, seed):
    """``_problem`` with ``reqs`` requests packed into the chunk: request r owns an equal share of the rows and of the
    keys (a cached prefix, then one key per row) and row i of it sees keys [request start, ...prefix + i + 1)."""
    q, qs, k, ks, w, kst, ke = _problem(rows, keys, seed)
    kst, ke = kst.cpu(), ke.cpu()
    for r in range(reqs):
        r0, r1 = r * rows // reqs, (r + 1) * rows // reqs
        k0, k1 = r * keys // reqs, (r + 1) * keys // reqs
        kst[r0:r1] = k0
        ke[r0:r1] = torch.arange(k1 - (r1 - r0) + 1, k1 + 1, dtype=torch.int32)
    return q, qs, k, ks, w, kst.cuda(), ke.cuda()


def _logits(prob, bf16):
    q, qs, k, ks, w, kst, ke = prob
    owner = SimpleNamespace(use_bf16=False, use_fp4=True, head_dim=DIM)
    return idx.Indexer._call_mqa_logits(
        owner, q, k, ks, w, kst, ke, qs, clean_logits=False, bf16_logits=bf16
    )


def _window_topk(logits, ke):
    cols = torch.arange(logits.shape[1], device=logits.device)[None, :]
    masked = logits.float().masked_fill(cols >= ke[:, None].long(), -torch.inf)
    return masked.topk(TOPK, dim=1).indices


@pytest.mark.parametrize("rows,keys", [(256, 8192), (512, 24576)])
def test_bf16_logits_off_is_the_fp32_kernel(rows, keys):
    """Knob off: the FP32-weight, FP32-reduce, FP32-logits DeepGEMM path, bit for bit (and the knob's default is
    off)."""
    _skip_unless_sm100()
    prob = _problem(rows, keys, seed=rows)
    a = _logits(prob, bf16=False)
    q, qs, k, ks, w, kst, ke = prob
    b = idx.fp8_fp4_mqa_logits(
        (q.view(torch.int8), qs.contiguous().view(torch.int32).reshape(rows, HEADS)),
        (k.view(torch.int8), ks.contiguous().view(torch.int32).reshape(-1)),
        w,
        kst,
        ke,
        clean_logits=False,
    )
    assert a.dtype == torch.float32
    cols = torch.arange(keys, device="cuda")[None, :]
    inside = cols < ke[:, None].long()
    assert torch.equal(a[inside].view(torch.int32), b[inside].view(torch.int32))


@pytest.mark.parametrize("rows,keys", [(256, 8192), (512, 24576)])
@pytest.mark.parametrize("cute", [False, True], ids=["deepgemm", "cute_dsl"])
def test_bf16_logits_close_to_fp32_and_topk_overlap(rows, keys, cute, monkeypatch):
    """Knob on: BF16-reduce logits (DeepGEMM's BF16 logits, or the CuTe DSL kernel's FP32 store of them) within the
    BF16-reduce error of the FP32 kernel; most of each row's Top-K survives."""
    _skip_unless_sm100()
    if cute and not IS_CUTLASS_DSL_AVAILABLE:
        pytest.skip("CuTe DSL not available")
    monkeypatch.setattr(idx, "_INDEXER_BF16_LOGITS_CUTE", cute)
    prob = _problem(rows, keys, seed=rows + 1)
    ref = _logits(prob, bf16=False)
    got = _logits(prob, bf16=True)
    assert got.dtype == (torch.float32 if cute else torch.bfloat16)
    ke = prob[-1]
    cols = torch.arange(keys, device="cuda")[None, :]
    inside = cols < ke[:, None].long()
    err = (got.float() - ref).abs().masked_fill(~inside, 0)
    scale = ref.abs().masked_fill(~inside, 0).amax(dim=1, keepdim=True)
    # 32 BF16 products accumulated in BF16 pairs: a few BF16 ulps of the row's largest logit.
    assert (err <= 3e-2 * scale).all(), float((err / scale).max())
    a, b = _window_topk(ref, ke), _window_topk(got, ke)
    overlap = torch.tensor(
        [len(set(x.tolist()) & set(y.tolist())) / TOPK for x, y in zip(a, b)], dtype=torch.float64
    )
    print(f"Top-K overlap BF16 vs FP32 logits: mean {overlap.mean():.4f} min {overlap.min():.4f}")
    assert overlap.mean() >= 0.9


@pytest.mark.parametrize(
    "rows,keys,reqs",
    [
        (8192, 124096, 1),  # an 8K-row chunk of a 124K-key request
        (2048, 600000, 1),  # a long request (the 600K-key chunk size)
        (1237, 24576, 1),  # a short tail chunk (rows not a multiple of the row group)
        (3001, 12000, 3),  # three requests packed into one chunk
    ],
)
def test_bf16_logits_cute_dsl_is_the_deepgemm_bf16_kernel(rows, keys, reqs, monkeypatch):
    """Knob on with the CuTe DSL: FP32 logits equal to DeepGEMM's BF16-reduce BF16 logits widened, bit for bit, in
    every row's window."""
    _skip_unless_sm100()
    if not IS_CUTLASS_DSL_AVAILABLE:
        pytest.skip("CuTe DSL not available")
    monkeypatch.setattr(idx, "_INDEXER_BF16_LOGITS_CUTE", True)
    prob = _packed_problem(rows, keys, reqs, seed=rows + reqs)
    q, qs, k, ks, w, kst, ke = prob
    got = _logits(prob, bf16=True)
    assert got.dtype == torch.float32
    ref = idx.fp8_fp4_mqa_logits(
        (q.view(torch.int8), qs.contiguous().view(torch.int32).reshape(rows, HEADS)),
        (k.view(torch.int8), ks.contiguous().view(torch.int32).reshape(-1)),
        w.to(torch.bfloat16),
        kst,
        ke,
        clean_logits=False,
        logits_dtype=torch.bfloat16,
    )
    cols = torch.arange(keys, device="cuda")[None, :]
    inside = (cols >= kst[:, None].long()) & (cols < ke[:, None].long())
    ref_bits = ref.float()[inside].view(torch.int32)
    assert torch.equal(got[inside].view(torch.int32), ref_bits)
    if reqs == 1:
        # The caller's hint that every row starts at key 0 (one request) picks the instantiation without column
        # offsets: same scores.
        owner = SimpleNamespace(use_bf16=False, use_fp4=True, head_dim=DIM)
        hinted = idx.Indexer._call_mqa_logits(
            owner,
            q,
            k,
            ks,
            w,
            kst,
            ke,
            qs,
            clean_logits=False,
            bf16_logits=True,
            row_starts_zero=True,
        )
        assert torch.equal(hinted[inside].view(torch.int32), ref_bits)
