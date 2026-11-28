# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The shared MLA drafter RMSNorm + RoPE entries on their ``.CELLS``.

Check the ULL inputs against each entry's FP64 reference and verify that the inputs
are untouched.
"""

import importlib
import zlib

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.catalog._op import Arch

pytest.importorskip("cutlass")

__extra_import_path__ = ["../../../.."]
from _validating import (  # noqa: E402, F401 — path above; fixture by name
    assert_within_ulp,
    certifying,
)

_OPS = "tensorrt_llm._torch.modeling_v2.models.kimi_k3_vl.ops.spec"
KV_LORA = 512
ROPE = 64
HEAD_DIM = KV_LORA + ROPE
EPS = 1e-5

rmsnorm_rope = importlib.import_module(f"{_OPS}.k3_dspark_rmsnorm_rope").k3_dspark_rmsnorm_rope
ctx_norm_rope = importlib.import_module(
    f"{_OPS}.k3_dspark_mla_ctx_norm_rope"
).k3_dspark_mla_ctx_norm_rope

_ARCH = {(10, 0): Arch.SM_100, (10, 3): Arch.SM_103, (10, 7): Arch.SM_107}


def _on_archs(op) -> bool:
    return torch.cuda.is_available() and _ARCH.get(torch.cuda.get_device_capability()) in op.ARCHS


def _cells(op):
    return pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:48] for c in op.CELLS])


def _cell_gen(cell) -> torch.Generator:
    return torch.Generator(device="cuda").manual_seed(zlib.crc32(cell.why.encode()) % 2**31)


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().view(torch.int16)


def _freqs_table(num_positions: int, gen) -> torch.Tensor:
    """A ``[P, ROPE // 2, 2]`` FP32 (cos, sin) table with YaRN-like magnitudes."""
    angle = torch.rand(num_positions, ROPE // 2, device="cuda", generator=gen) * 6.3
    scale = 1.0 + 0.3 * torch.rand(num_positions, 1, device="cuda", generator=gen)
    return torch.stack([angle.cos() * scale, angle.sin() * scale], dim=-1).contiguous()


def _rows(n, width, gen, row_stride=None):
    """BF16 rows with a spread of magnitudes; optionally a row-strided view."""
    row_stride = row_stride or width
    base = torch.randn(n, row_stride, device="cuda", generator=gen)
    base = base * torch.logspace(-2, 2, n, device="cuda").unsqueeze(1)
    return base.to(torch.bfloat16)[:, :width]


def _norm_weight(n, gen, layers=None):
    shape = (n,) if layers is None else (layers, n)
    return (1.0 + 0.5 * torch.randn(*shape, device="cuda", generator=gen)).to(torch.bfloat16)


def _compare(out: torch.Tensor, ref: torch.Tensor) -> None:
    """2 ulp per element against the row scale, 0.5 ulp relative RMS: one BF16 rounding on each
    side of an FP32 (kernel) against an FP64 (reference) RMSNorm and rotation."""
    assert_within_ulp(out, ref, element_ulp=2.0, rms_ulp=0.5)


@pytest.mark.skipif(not _on_archs(rmsnorm_rope), reason="not an ARCHS architecture")
@_cells(rmsnorm_rope)
def test_k3_dspark_rmsnorm_rope_ull_cells(cell, certifying):  # noqa: F811 — fixture by name
    certifying(rmsnorm_rope)
    gen = _cell_gen(cell)
    rows = cell.spec["rows"]
    x = _rows(rows, HEAD_DIM, gen, row_stride=cell.spec["row_stride"])
    weight = _norm_weight(KV_LORA, gen)
    table = _freqs_table(rows, gen)
    before = x.clone()
    args = (weight, table, 1, ROPE, EPS, True, True, False, KV_LORA)
    out = rmsnorm_rope(x, *args)
    assert out.is_contiguous() and out.shape == x.shape and out.dtype == torch.bfloat16
    assert torch.equal(_bits(x), _bits(before)), "the op must not write its input"
    _compare(out, rmsnorm_rope.reference(x, *args))


@pytest.mark.skipif(not _on_archs(ctx_norm_rope), reason="not an ARCHS architecture")
@_cells(ctx_norm_rope)
def test_k3_dspark_mla_ctx_norm_rope_ull_cells(cell, certifying):  # noqa: F811 — fixture by name
    certifying(ctx_norm_rope)
    gen = _cell_gen(cell)
    rows, layers = cell.spec["rows"], cell.spec["layers"]
    latent = _rows(rows, layers * HEAD_DIM, gen).view(rows, layers, HEAD_DIM)
    weight = torch.nn.Parameter(_norm_weight(KV_LORA, gen, layers), requires_grad=False)
    table = _freqs_table(16384, gen)
    positions = torch.randint(0, 16384, (rows,), device="cuda", generator=gen)
    before = latent.clone()
    out = ctx_norm_rope(latent, weight, positions, table, EPS)
    assert out.shape == (layers, rows, HEAD_DIM) and out.dtype == torch.bfloat16
    assert torch.equal(_bits(latent), _bits(before)), "the op must not write its input"
    _compare(out, ctx_norm_rope.reference(latent, weight, positions, table, EPS))
