# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The ULL target's MLA query entries on their ``.CELLS``, each under ``validating``:

* ``k3_mla_q_fused_ull`` (SM107): every FP4 group of the written query rows within one FP4 step of
  the entry's plain-torch reference, codes equal in nearly every group, bytes outside the tokens'
  rows untouched (the band of ``kernels/k3r_mla_q/test_k3_mla_q_fused.py``);
* ``k3_mla_latent_row_ull``: the norm within one BF16 rounding of the FP32 reference, the K-pe
  columns copied exactly, and the row bitwise the family sibling ``k3_mla_latent_row``;
* ``rmsnorm_release_at_launch``: within one BF16 rounding of the FP32 reference.
"""

import functools
import importlib
import math
import zlib
from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.catalog._op import Arch

__extra_import_path__ = ["../../../.."]
from _validating import assert_within_ulp, validating  # noqa: E402

K3 = "tensorrt_llm._torch.modeling_v2.models.kimi_k3_vl"
H, QL, LORA, ROPE = 96, 1536, 512, 64
EPS = 1e-6

q_fused_module = importlib.import_module(f"{K3}.ops.attention.k3_mla_q_fused_ull")
k3_mla_q_fused_ull = q_fused_module.k3_mla_q_fused
k3_mla_latent_row_ull = importlib.import_module(
    f"{K3}.ops.attention.k3_mla_latent_row_ull"
).k3_mla_latent_row
rmsnorm_release_at_launch = importlib.import_module(
    f"{K3}.ops.attention.rmsnorm_release_at_launch"
).rmsnorm_release_at_launch

_ARCH = {(10, 0): Arch.SM_100, (10, 3): Arch.SM_103, (10, 7): Arch.SM_107}


def _on_archs(op) -> bool:
    return torch.cuda.is_available() and _ARCH.get(torch.cuda.get_device_capability()) in op.ARCHS


def _certified(op):
    return pytest.mark.skipif(not _on_archs(op), reason=f"{type(op).__name__}: not an ARCHS GPU")


def _cells(op):
    return pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:48] for c in op.CELLS])


def _seed(cell) -> int:
    return zlib.crc32(cell.why.encode()) % 2**31


@functools.lru_cache(maxsize=1)
def _weights():
    lowm = importlib.import_module(f"{K3}.kernels.k3r_gemm.k3_fp8w_lowm_gemm")
    fused = importlib.import_module(f"{K3}.kernels.k3r_mla_q.k3_mla_q_fused_ull")
    dev = torch.device("cuda")
    gen = torch.Generator().manual_seed(7)
    w_qb = (torch.randn(H * 192, QL, generator=gen) * 2).to(torch.float8_e4m3fn)
    grid = (math.ceil(H * 192 / 128), math.ceil(QL / 128))
    s_qb = torch.exp2(torch.randint(-8, -6, grid, generator=gen).float())
    w_qb, _ = lowm.prepare_k3_fp8w_lowm_weight(w_qb.to(dev), s_qb.to(dev))
    return SimpleNamespace(
        w_qb=w_qb,
        qb_exp=fused.block_exponents(s_qb.to(dev), H * 192, QL),
        w_uk=(torch.randn(H, LORA, 128, generator=gen) * 0.06).to(torch.bfloat16).to(dev),
        q_norm=(1 + 0.1 * torch.randn(QL, generator=gen)).bfloat16().to(dev),
    )


def _identity_rotary_table(max_pos):
    rot = torch.zeros((max_pos, ROPE, 2), dtype=torch.float32, device="cuda")
    rot[..., 0] = 1.0
    return rot.view(max_pos, ROPE * 2).contiguous()


def _flashinfer_norm(x, weight):
    from flashinfer.norm import rmsnorm

    out = torch.empty_like(x)
    rmsnorm(x, weight, EPS, out=out)
    return out


def _dequant_rows(q_fp4, q_sf, rows, gs):
    """FP32 ``[rows, 576]`` values of FP4 query rows (RoPE: main + residual) and their FP32
    ``[rows, 40]`` group scales, in units of the unscaled query."""
    cols = 40
    col = torch.arange(cols, device=q_fp4.device)
    offsets = q_fused_module.swizzled_sf_offset(rows[:, None], col[None, :], cols)
    sf = q_sf.view(torch.uint8).view(-1)[offsets.flatten()].view(-1, cols)
    sf = sf.view(torch.float8_e4m3fn).float()
    codes = q_fused_module.unpack_e2m1(q_fp4[rows])
    values = q_fused_module.e2m1_values(codes).view(-1, cols, 16) * sf.unsqueeze(-1)
    prefix = values[:, :32].flatten(1)
    rope = values[:, 32:].view(-1, 4, 2, 16)
    return torch.cat((prefix, (rope[:, :, 0] + rope[:, :, 1]).flatten(1)), -1) / gs, sf / gs


@_certified(k3_mla_q_fused_ull)
@_cells(k3_mla_q_fused_ull)
def test_k3_mla_q_fused_ull_cells(cell):
    """Every FP4 group within one FP4 step (two units of its scale) of the reference's; codes
    equal in nearly every group (the q_b MMA's summation order moves a BF16 value by an ulp now
    and then); every byte outside the tokens' rows untouched."""
    from tensorrt_llm._torch.attention.backends.fp4_mla import FP4_MLA_Q_GLOBAL_SCALE
    from tensorrt_llm._torch.attention.backends.fp4_mla.config import (
        FP4_MLA_Q_LOGICAL_DIM,
        _fp4_mla_packed_rows_per_seq,
    )
    from tensorrt_llm._torch.attention.backends.fp4_mla.layout import (
        _get_fp4_mla_swizzled_scale_size,
    )

    op = k3_mla_q_fused_ull
    spec = cell.spec
    w = _weights()
    requests, gen_len = spec["requests"], spec["gen_len"]
    tokens = requests * gen_len
    gen = torch.Generator(device="cuda").manual_seed(_seed(cell))
    x = torch.randn(tokens, QL, device="cuda", generator=gen).bfloat16()
    rows_per_seq = _fp4_mla_packed_rows_per_seq(H, gen_len)
    num_rows = requests * rows_per_seq
    q_fp4 = torch.zeros((num_rows, 320), dtype=torch.uint8, device="cuda")
    q_sf = torch.zeros(
        _get_fp4_mla_swizzled_scale_size(num_rows, FP4_MLA_Q_LOGICAL_DIM),
        dtype=torch.uint8,
        device="cuda",
    )
    gs = torch.tensor([FP4_MLA_Q_GLOBAL_SCALE], dtype=torch.float32, device="cuda")
    rot = _identity_rotary_table(4096)
    positions = (1000 + torch.arange(tokens, device="cuda")).to(torch.int32)
    norm = (w.q_norm, EPS) if spec["norm"] else None
    if norm is None:
        x = _flashinfer_norm(x, w.q_norm)
    args = (x, w.w_qb, w.qb_exp, w.w_uk, gs, rot, positions, q_fp4, q_sf, gen_len, rows_per_seq,
            True, norm, spec["early_trigger"])  # fmt: skip
    ref_q, ref_sf = op.reference(*args)
    with validating(op):
        op(*args)
    torch.cuda.synchronize()

    t = torch.arange(tokens, device="cuda")
    token_rows = t // gen_len * rows_per_seq + t % gen_len * H
    rows = (token_rows[:, None] + torch.arange(H, device="cuda")[None, :]).flatten()
    written = torch.zeros(num_rows, dtype=torch.bool, device="cuda")
    written[rows] = True
    assert torch.equal(q_fp4[~written], ref_q[~written]), "bytes outside the tokens' rows moved"
    got, got_sf = _dequant_rows(q_fp4, q_sf, rows, float(gs))
    ref, ref_sf = _dequant_rows(ref_q, ref_sf, rows, float(gs))
    step = 2 * torch.maximum(got_sf, ref_sf)
    step = torch.cat((step[:, :32].repeat_interleave(16, -1),
                      step[:, 32::2].repeat_interleave(16, -1)), -1)  # fmt: skip
    assert bool(((got - ref).abs() <= step).all()), cell.why
    groups_equal = torch.eq(q_fp4[rows].view(-1, 40, 8), ref_q[rows].view(-1, 40, 8)).all(-1)
    assert float(groups_equal.float().mean()) >= 0.95, cell.why


@_certified(k3_mla_latent_row_ull)
@_cells(k3_mla_latent_row_ull)
@pytest.mark.parametrize("use_pdl", [True, False])
def test_k3_mla_latent_row_ull_cells(cell, use_pdl):
    from tensorrt_llm._torch.modeling_v2.models.kimi_k3_vl.ops.attention.k3_mla_latent_row import (  # noqa: E501
        k3_mla_latent_row as sibling,
    )

    op = k3_mla_latent_row_ull
    spec = cell.spec
    tokens, stride, eps = spec["tokens"], spec["row_stride"], spec["eps"]
    gen = torch.Generator(device="cuda").manual_seed(_seed(cell))
    proj = torch.randn(tokens, stride, device="cuda", generator=gen).to(torch.bfloat16)
    c_kv = proj[:, stride - LORA - ROPE : stride - ROPE]
    k_pe = proj[:, stride - ROPE :]
    weight = (1 + 0.1 * torch.randn(LORA, device="cuda", generator=gen)).to(torch.bfloat16)
    with validating(op):
        out = op(c_kv, k_pe, weight, eps, use_pdl=use_pdl)
    assert out.shape == (tokens, LORA + ROPE) and out.is_contiguous()
    expected = op.reference(c_kv, k_pe, weight, eps, use_pdl=use_pdl)
    assert_within_ulp(out[:, :LORA], expected[:, :LORA], element_ulp=1, rms_ulp=0.5)
    assert torch.equal(out[:, LORA:], expected[:, LORA:]), cell.why
    want = sibling(c_kv, k_pe, weight, eps, use_pdl=use_pdl)
    torch.cuda.synchronize()
    assert torch.equal(out.view(torch.int16), want.view(torch.int16)), cell.why


@_certified(rmsnorm_release_at_launch)
@_cells(rmsnorm_release_at_launch)
def test_rmsnorm_release_at_launch_ull_cells(cell):
    op = rmsnorm_release_at_launch
    spec = cell.spec
    gen = torch.Generator(device="cuda").manual_seed(_seed(cell))
    rows = torch.randn(spec["tokens"], spec["row_stride"], device="cuda", generator=gen)
    x = rows.to(torch.bfloat16)[:, : spec["width"]]
    weight = (1 + 0.1 * torch.randn(spec["width"], device="cuda", generator=gen)).to(torch.bfloat16)
    eps = spec["eps"]
    with validating(op):
        y = op(x, weight, eps)
    assert y.is_contiguous() and y.shape == x.shape and y.dtype == torch.bfloat16
    assert_within_ulp(y, op.reference(x, weight, eps), element_ulp=1, rms_ulp=0.5)
