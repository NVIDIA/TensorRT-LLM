# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the target's SM120 e4m3 head-256 context FMHA (csrc/fmha/).

The kernel is driven through the test-only op ``context_fmha_sm120_test``, which fills the
kernel parameters as the attention op does for this target: FP8 Q ``[T, 24, 256]``, FP8
paged KV (4 heads, 32 tokens per page, pages in a shuffled order), causal attention over a
cached prefix (query ``i`` of a sequence with KV length ``L`` and ``Lq`` new tokens sees
keys ``[0, L - Lq + i]``), BF16 output.

Two references, both in fp32 from the dequantized FP8 inputs:

* ``exact``: the plain softmax. The kernel quantizes the probabilities to e4m3 before the
  second GEMM (3 mantissa bits, a relative error up to 2^-4 per probability), so this one is
  checked only by the relative RMS over the whole output (< 0.05; measured 0.010 to 0.017,
  a wrong mask, page, head or scale is off by order 1).
* ``emulated``: the kernel's arithmetic: raw Q K^T, a running row max per 128-key tile
  (reset at each KV piece of the last q-tile when the split runs), P * 256 =
  2^((s - running max) * scale * log2(e) + 8) cast to e4m3, the tiles rescaled to the final
  max, the sum taken over the unquantized P. The output differs from it by its BF16 rounding
  (relative RMS ~1.6e-3, bound 4e-3) and by rare e4m3 rounding flips of single probabilities
  (the QMMA accumulation order and ex2.approx move P across a rounding midpoint; measured
  <= 2e-5 of the elements outside 2^-7 |ref| + 2^-10, bound 1e-4, each at most one e4m3 step
  of one probability: bound 2^-5 of the largest |ref|, measured <= 1%).

On top: the rows of q-tiles 0 .. Q-2 are bitwise the same with and without the last q-tile
split; ``skip_leading`` leaves them unwritten (NaN, the op's preset) and writes the last
q-tile bitwise as without it.

The whole attention op, KV-cache append included, is driven over real paged pools in
``test_modeling_v2_attention_sm120.py``.
"""

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from tensorrt_llm._torch._experimental.modeling_v2.models.qwen3_5 import (
    qwen3_8_27b_nvfp4__sm_120__tp1 as target,
)
from tensorrt_llm._utils import get_sm_version

pytestmark = pytest.mark.skipif(
    get_sm_version() != 120, reason="the qwen3_5 family library is built for sm_120f only"
)

DEV = "cuda"
H, HKV, D = 24, 4, 256
TPB = 32
STEP_Q, STEP_KV = 64, 128
LOG2E = 1.4426950408889634
DEFAULT_BMM1 = 1.0 / 16.0  # 1 / sqrt(256)


@pytest.fixture(autouse=True)
def _fp32_matmul():
    """The references need true fp32 GEMMs (no TF32)."""
    prev = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    yield
    torch.backends.cuda.matmul.allow_tf32 = prev


def _choose_num_pieces(b: int, s: int) -> int:
    """contextFmhaLastTileSplit.cuh choose_num_pieces, 2 CTAs/SM (48 KB smem, 128 threads)."""
    slots = torch.cuda.get_device_properties(0).multi_processor_count * 2
    if b <= 0 or b > 3:
        return 0
    q_tiles = (s + STEP_Q - 1) // STEP_Q
    if q_tiles < 2:
        return 0
    ctas, ctas_a = q_tiles * H * b, (q_tiles - 1) * H * b
    if (ctas_a + slots - 1) // slots >= (ctas + slots - 1) // slots:
        return 0
    k = min(slots // (H * b), 16, 384 // (H * b))
    return k if k >= 2 else 0


class _Case:
    def __init__(self, q_lens, prefixes, seed):
        assert len(q_lens) == len(prefixes)
        self.q_lens = list(q_lens)
        self.kv_lens = [p + q for p, q in zip(prefixes, q_lens)]
        self.b = len(q_lens)
        self.s = max(q_lens)
        g = torch.Generator(device=DEV).manual_seed(seed)
        g_cpu = torch.Generator().manual_seed(seed)
        t = sum(q_lens)
        self.q = (torch.randn(t, H, D, generator=g, device=DEV) * 1.5).to(torch.float8_e4m3fn)
        self.k = [
            (torch.randn(L, HKV, D, generator=g, device=DEV) * 1.5).to(torch.float8_e4m3fn)
            for L in self.kv_lens
        ]
        self.v = [
            torch.randn(L, HKV, D, generator=g, device=DEV).to(torch.float8_e4m3fn)
            for L in self.kv_lens
        ]
        pages = [(L + TPB - 1) // TPB for L in self.kv_lens]
        max_blocks = max(pages)
        num_blocks = 2 * sum(pages) + 3
        order = torch.randperm(num_blocks, generator=g_cpu)
        self.pool = torch.zeros(num_blocks, HKV, TPB, D, dtype=torch.uint8, device=DEV)
        self.block_offsets = torch.zeros(self.b, 2, max_blocks, dtype=torch.int32)
        nxt = 0
        for i, L in enumerate(self.kv_lens):
            for kv, x in enumerate((self.k[i], self.v[i])):
                ids = order[nxt : nxt + pages[i]]
                nxt += pages[i]
                self.block_offsets[i, kv, : pages[i]] = ids.to(torch.int32)
                # The unused tail of the last page is zero, as the KV append leaves it.
                padded = torch.zeros(pages[i] * TPB, HKV, D, dtype=torch.uint8, device=DEV)
                padded[:L] = x.view(torch.uint8)
                self.pool[ids.to(DEV)] = padded.view(pages[i], TPB, HKV, D).permute(0, 2, 1, 3)
        self.block_offsets = self.block_offsets.to(DEV)
        self.cu_q = torch.tensor([0] + list(_cumsum(q_lens)), dtype=torch.int32, device=DEV)
        self.cu_kv = torch.tensor([0] + list(_cumsum(self.kv_lens)), dtype=torch.int32, device=DEV)

    def run(self, split, skip_leading=False, bmm1=DEFAULT_BMM1, bmm2=1.0):
        out = target.native_ops.context_fmha_sm120_test(
            self.q.view(torch.uint8),
            self.pool,
            self.block_offsets,
            self.cu_q,
            self.cu_kv,
            self.s,
            bmm1,
            bmm2,
            split,
            skip_leading,
        )
        torch.cuda.synchronize()
        assert out.shape == (sum(self.q_lens), H, D) and out.dtype == torch.bfloat16
        return out

    def pieces(self, split):
        return _choose_num_pieces(self.b, self.s) if split else 0

    def last_tile_rows(self):
        """Per sequence: the rows (in the sequence) of the last q-tile of the grid."""
        first = STEP_Q * ((self.s + STEP_Q - 1) // STEP_Q - 1)
        return first

    def references(self, split, bmm1=DEFAULT_BMM1, bmm2=1.0):
        """(exact, emulated) fp32 outputs [T, 24, 256]."""
        pieces = self.pieces(split)
        first_last = self.last_tile_rows()
        s2 = torch.tensor(bmm1 * LOG2E, dtype=torch.float32).item()
        q32 = self.q.float()
        exact = torch.empty(q32.shape, dtype=torch.float32, device=DEV)
        emul = torch.empty_like(exact)
        row0 = 0
        for i, (lq, L) in enumerate(zip(self.q_lens, self.kv_lens)):
            k32, v32 = self.k[i].float(), self.v[i].float()
            pos = (L - lq) + torch.arange(lq, device=DEV)
            masked = torch.arange(L, device=DEV)[None, :] > pos[:, None]
            n_t = (L + STEP_KV - 1) // STEP_KV
            # The piece ranges of the last q-tile, per row of that tile (as the kernel computes them).
            segments = None
            if pieces and lq > first_last:
                q_start = first_last + L - lq
                valid = min(q_start + STEP_Q, L)
                n = (valid + STEP_KV - 1) // STEP_KV
                active = max(1, min(pieces, n // 2))
                segments = [(p * n // active, (p + 1) * n // active) for p in range(active)]
            for h in range(H):
                qh, kh, vh = (
                    q32[row0 : row0 + lq, h],
                    k32[:, h // (H // HKV)],
                    v32[:, h // (H // HKV)],
                )
                s = qh @ kh.T
                s.masked_fill_(masked, float("-inf"))
                exact[row0 : row0 + lq, h] = torch.softmax(s * bmm1, dim=-1) @ vh * bmm2

                st = torch.full((lq, n_t * STEP_KV), float("-inf"), device=DEV)
                st[:, :L] = s
                st = st.view(lq, n_t, STEP_KV)
                tmax = st.amax(-1)
                rm = torch.cummax(tmax, dim=-1).values
                if segments is not None:
                    rows = slice(first_last, lq)
                    for a, b in segments:
                        rm[rows, a:b] = torch.cummax(tmax[rows, a:b], dim=-1).values
                top = tmax.amax(-1, keepdim=True)
                p256 = torch.exp2((st - rm[..., None]) * s2 + 8.0)
                p256 = torch.where(st == float("-inf"), torch.zeros_like(p256), p256)
                pq = p256.to(torch.float8_e4m3fn).float()
                w = torch.exp2((rm - top) * s2)
                w = torch.where(rm == float("-inf"), torch.zeros_like(w), w)
                num = (pq * w[..., None]).view(lq, -1)[:, :L] @ vh
                den = (p256.sum(-1) * w).sum(-1, keepdim=True)
                emul[row0 : row0 + lq, h] = num / den * bmm2
            row0 += lq
        return exact, emul

    def leading_mask(self):
        """[T] bool: rows in q-tiles 0 .. Q-2 of their sequence."""
        first_last = self.last_tile_rows()
        return torch.cat([torch.arange(lq, device=DEV) < first_last for lq in self.q_lens])


def _cumsum(xs):
    total = 0
    for x in xs:
        total += x
        yield total


def _check(out, exact, emul, what):
    o = out.float()
    assert torch.isfinite(o).all(), f"{what}: non-finite output"
    err = (o - emul).abs()
    tol = emul.abs() * 2.0**-7 + 2.0**-10
    frac_off = (err > tol).float().mean().item()
    rel_rms_emul = (err.pow(2).mean() / emul.pow(2).mean()).sqrt().item()
    rel_rms = ((o - exact).pow(2).mean() / exact.pow(2).mean()).sqrt().item()
    assert frac_off < 1e-4, f"{what}: {frac_off:.2e} of the elements off the emulated reference"
    assert err.max().item() <= 2.0**-5 * emul.abs().max().item(), (
        f"{what}: max error {err.max().item():.3e} against the emulated reference"
    )
    assert rel_rms_emul < 4e-3, (
        f"{what}: relative RMS {rel_rms_emul:.2e} against the emulated reference"
    )
    assert rel_rms < 0.05, f"{what}: relative RMS {rel_rms:.4f} against the exact softmax"


def _seq_shapes(nseq, q_len, prefix):
    # The first sequence has (q_len, prefix); the others are shorter, with other prefixes.
    q_lens = [max(1, q_len - 3 * i) for i in range(nseq)]
    prefixes = [prefix + (13 * i if prefix else 0) for i in range(nseq)]
    return q_lens, prefixes


@pytest.mark.parametrize("prefix", [0, 32768])
@pytest.mark.parametrize("q_len", [1, 37, 64, 65, 2048])
@pytest.mark.parametrize("nseq", [1, 2, 3])
def test_matches_reference(nseq, q_len, prefix):
    q_lens, prefixes = _seq_shapes(nseq, q_len, prefix)
    case = _Case(q_lens, prefixes, seed=1000 * nseq + 7 * q_len + prefix)
    outs = {}
    for split in (False, True):
        outs[split] = case.run(split)
        exact, emul = case.references(split)
        _check(outs[split], exact, emul, f"split={split} pieces={case.pieces(split)}")
    lead = case.leading_mask()
    if case.pieces(True):
        # Rows of q-tiles 0 .. Q-2 come from the same kernel with s - 64.
        assert torch.equal(outs[True][lead].view(torch.int16), outs[False][lead].view(torch.int16))
    else:
        assert torch.equal(outs[True].view(torch.int16), outs[False].view(torch.int16))


def test_split_runs_for_long_single_prefill():
    # 1 x 2048 with 24 heads is 768 CTAs: a tail wave on 188 SMs x 2.
    assert _choose_num_pieces(1, 2048) > 0
    assert _choose_num_pieces(4, 2048) == 0


def test_batch_of_eight_mixed_lengths():
    q_lens = [1, 37, 64, 65, 300, 129, 2, 512]
    prefixes = [0, 5000, 31, 64, 0, 777, 1, 2048]
    case = _Case(q_lens, prefixes, seed=88)
    bmm1, bmm2 = DEFAULT_BMM1 * 0.75, 1.25
    exact, emul = case.references(split=False, bmm1=bmm1, bmm2=bmm2)
    for split in (False, True):  # B > 3: the split never runs
        out = case.run(split, bmm1=bmm1, bmm2=bmm2)
        _check(out, exact, emul, f"batch 8 split={split}")


@pytest.mark.parametrize(
    "q_lens,prefixes", [([2048], [32768]), ([2048, 2048, 2048], [0, 100, 32768])]
)
def test_skip_leading_writes_only_the_last_q_tile(q_lens, prefixes):
    case = _Case(q_lens, prefixes, seed=5)
    assert case.pieces(True) > 0
    full = case.run(split=True)
    skipped = case.run(split=True, skip_leading=True)
    lead = case.leading_mask()
    assert torch.isnan(skipped[lead].float()).all()
    assert torch.equal(skipped[~lead].view(torch.int16), full[~lead].view(torch.int16))
    exact, emul = case.references(split=True)
    _check(full, exact, emul, "split with leading rows")


def test_fake_matches_the_op_output_shape():
    with FakeTensorMode():
        q = torch.empty(100, H, D, dtype=torch.uint8, device=DEV)
        pool = torch.empty(16, HKV, TPB, D, dtype=torch.uint8, device=DEV)
        offsets = torch.empty(2, 2, 4, dtype=torch.int32, device=DEV)
        cu = torch.empty(3, dtype=torch.int32, device=DEV)
        out = target.native_ops.context_fmha_sm120_test(
            q, pool, offsets, cu, cu, 64, DEFAULT_BMM1, 1.0, True, False
        )
        assert out.shape == (100, H, D) and out.dtype == torch.bfloat16
