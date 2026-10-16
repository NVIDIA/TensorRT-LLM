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
"""trtllm::k3_ctm_gemv_wide (up to 64 tokens in one MMA of 16 / 32 / 64 token columns) at the Kimi K3 TP16 per-rank
projection shapes, at every M in 1..64: error against an fp64 product and against cuBLAS (F.linear), run-to-run
identical bits, each M's rows bit-identical to the same rows of the 64-row call, the sigmoid columns torch.sigmoid of
the op's own plain output, and, where the split matches the call site's k3_ctm_gemv_long at most 8 tokens, the same
bits as that kernel. Both split-K transports give the same bits. 0 and 65 rows are refused."""

import functools

import pytest
import torch
import torch.nn.functional as F


def _is_sm100() -> bool:
    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability()
    return major == 10


pytestmark = pytest.mark.skipif(not _is_sm100(), reason="needs an SM100-family GPU")

M_ALL = list(range(1, 65))
TOL = 8e-3  # bf16 output: max |y - ref| / max |ref|
TOL_FP32 = 1e-4  # fp32 output

# (N, K, sig_col0, out_fp32): the per-rank TP16 shapes of the projections of a wide decode step.
WIDE = {
    "kda_qkvg": (3208, 7168, -1, False),  # KDA q/k/v/g/f_a/b: 25 whole tiles and an 8-row last one
    "mla_qkv_a_gate": (2880, 7168, 2112, False),  # [W_a; W_g], the gate rows as bf16(sigmoid)
    "o_proj": (7168, 768, -1, False),  # KDA / MLA o_proj
    "moe_head": (280, 7168, -1, True),  # [latent down slice; router rows], fp32 (router logits)
    "moe_tail": (7168, 640, -1, False),  # [latent up slice | padding | shared down], 5 k-tiles
    "shared_gate_up": (768, 7168, -1, False),
    "dense_gate_up": (4224, 7168, -1, False),
    "dense_down": (7168, 2112, -1, False),  # K 2112 ends in a half k-tile
    "drafter_qkv": (512, 7168, -1, False),
    "drafter_gate_up": (1792, 7168, -1, False),
    "drafter_o_proj": (7168, 384, -1, False),
}
# The k3_ctm_gemv_long call (split, ring, push) of the shapes that run it at most 8 tokens.
LONG_SITES = {
    "mla_qkv_a_gate": (6, 6, True),
    "dense_gate_up": (4, 5, False),
    "dense_down": (2, 6, False),
    "drafter_qkv": (8, 6, True),
    "drafter_gate_up": (8, 6, True),
}


def _ops():
    import tensorrt_llm  # noqa: F401  (registers the trtllm ops)
    from tensorrt_llm._torch.cute_dsl_kernels.k3_ctm_gemv import op  # noqa: F401

    return torch.ops.trtllm


def _ctm():
    from tensorrt_llm._torch.cute_dsl_kernels.k3_ctm_gemv import op

    return op


def _bits(t: torch.Tensor) -> torch.Tensor:
    t = t.contiguous()
    return t.view(torch.int32) if t.dtype == torch.float32 else t.view(torch.int16)


def _rel(y: torch.Tensor, ref: torch.Tensor) -> float:
    return (y.double() - ref.double()).abs().max().item() / ref.double().abs().max().item()


@functools.lru_cache(maxsize=None)
def _weight(n: int, k: int, seed: int) -> torch.Tensor:
    gen = torch.Generator(device="cuda").manual_seed(seed)
    return (torch.randn(n, k, generator=gen, device="cuda") * 0.02).bfloat16()


@functools.lru_cache(maxsize=None)
def _rows(k: int, seed: int) -> torch.Tensor:
    gen = torch.Generator(device="cuda").manual_seed(seed)
    return torch.randn(64, k, generator=gen, device="cuda").bfloat16()


@functools.lru_cache(maxsize=None)
def _full_call(name: str) -> torch.Tensor:
    """The 64-row call of a case (its rows are what every M's call must reproduce)."""
    n, k, sig, fp32 = WIDE[name]
    return _ops().k3_ctm_gemv_wide(_rows(k, 2), _weight(n, k, 1), sig, fp32)


@pytest.mark.parametrize("m", M_ALL)
@pytest.mark.parametrize("name", list(WIDE))
def test_k3_ctm_gemv_wide(name, m):
    ops = _ops()
    n, k, sig, fp32 = WIDE[name]
    w = _weight(n, k, 1)
    x = _rows(k, 2)[:m].contiguous()
    assert _ctm().supports_wide(x, w, sig, fp32)
    y = ops.k3_ctm_gemv_wide(x, w, sig, fp32)
    det = torch.equal(_bits(y), _bits(ops.k3_ctm_gemv_wide(x, w, sig, fp32)))
    rows_as_m64 = torch.equal(_bits(y), _bits(_full_call(name)[:m]))
    plain = ops.k3_ctm_gemv_wide(x, w, -1, fp32) if sig >= 0 else y
    ref = x.double() @ w.double().t()
    cols = slice(0, sig if sig >= 0 else n)
    # Columns >= sig_col0 hold bf16(sigmoid(bf16(x @ W^T))): torch.sigmoid of the op's own plain output.
    sigmoid_ok = sig < 0 or (
        torch.equal(_bits(y[:, :sig]), _bits(plain[:, :sig]))
        and torch.equal(_bits(y[:, sig:]), _bits(plain[:, sig:].sigmoid()))
    )
    eq_long = None
    if name in LONG_SITES and m <= 8:
        split, ring, push = LONG_SITES[name]
        from tensorrt_llm._torch.cute_dsl_kernels.k3_ctm_gemv import k3_ctm_gemv_kernel as kernel

        wide_split = _ctm().wide_config(
            n, k, kernel.wide_tile(m), torch.cuda.get_device_properties(0).multi_processor_count
        )[0]
        long_y = ops.k3_ctm_gemv_long(x, w, sig, split, ring, True, push)
        eq_long = torch.equal(_bits(y), _bits(long_y)) if wide_split == split else None
    stock = F.linear(x.float(), w.float()) if fp32 else F.linear(x, w)
    print(
        f"OPCHECK op=k3_ctm_gemv_wide case={name} M={m} dtype={str(y.dtype)[6:]} "
        f"abs={(plain.double() - ref).abs().max().item():.3e} rel={_rel(plain, ref):.3e} "
        f"vs_stock={_rel(plain, stock):.2e} det={det} rows_as_m64={rows_as_m64} sigmoid_cols={sigmoid_ok} "
        f"eq_long={eq_long}"
    )  # fmt: skip
    assert y.shape == (m, n) and y.dtype == (torch.float32 if fp32 else torch.bfloat16)
    tol = TOL_FP32 if fp32 else TOL
    assert _rel(plain, ref) <= tol and _rel(plain, stock) <= tol
    assert _rel(y[:, cols], ref[:, cols]) <= tol
    assert det and rows_as_m64 and sigmoid_ok
    assert eq_long is not False


@pytest.mark.parametrize("m", list(range(16, 65, 8)))
@pytest.mark.parametrize("name", list(LONG_SITES))
def test_k3_ctm_gemv_wide_rows_as_long(name, m):
    """A wide step of R x 8 tokens: every token's row is the bits k3_ctm_gemv_long gives that token at <= 8 tokens (the
    call sites' split), so wide steps and batch 1 agree on these projections."""
    ops = _ops()
    n, k, sig, _ = WIDE[name]
    split, ring, push = LONG_SITES[name]
    w = _weight(n, k, 1)
    x = _rows(k, 2)[:m].contiguous()
    y = ops.k3_ctm_gemv_wide(x, w, sig)

    def long_rows(r):
        return ops.k3_ctm_gemv_long(x[r : r + 8].contiguous(), w, sig, split, ring, True, push)

    same = [torch.equal(_bits(y[r : r + 8]), _bits(long_rows(r))) for r in range(0, m, 8)]
    print(
        f"OPCHECK op=k3_ctm_gemv_wide case={name} M={m} rows_as_long_per_8={sum(same)}/{len(same)}"
    )
    assert all(same)


@pytest.mark.parametrize("m", [0, 65])
def test_token_limit(m):
    _ops()
    x = torch.zeros(m, 7168, dtype=torch.bfloat16, device="cuda")
    w = _weight(3208, 7168, 1)
    assert not _ctm().supports_wide(x, w)
    with pytest.raises(ValueError):
        torch.ops.trtllm.k3_ctm_gemv_wide(x, w)
