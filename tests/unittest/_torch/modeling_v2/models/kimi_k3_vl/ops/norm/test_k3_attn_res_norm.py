# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test of the k3_attn_res_norm entry (``kernels/k3r_attn_res/k3_attn_res.py``, one GPU).

Every cell of ``k3_attn_res_norm.CELLS`` runs under ``validating``: the updated prefix sum bit for bit
against the reference's FP32 adds rounded to BF16, the normed output against the reference (the family
``attn_res_fwd`` mixture and the twice-rounded trailing norm) under the cosine / relative-L2 metric of
the family attn_res tests, identical bits on a repeated call, untouched inputs, and the MXFP8 output
bit for bit the quantization rule of the low-M FP8 GEMM applied to the stored BF16 output (row-major
scales, or the mid-M GEMM's scale atoms).
"""

import importlib
import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch

__extra_import_path__ = ["../../../.."]
from _validating import validating  # noqa: E402

_ENTRY = importlib.import_module(
    "tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.ops.norm.k3_attn_res_norm"
)
k3_attn_res_norm = _ENTRY.k3_attn_res_norm

_ARCH = {(10, 0): Arch.SM_100, (10, 3): Arch.SM_103, (10, 7): Arch.SM_107}
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or _ARCH.get(torch.cuda.get_device_capability()) not in k3_attn_res_norm.ARCHS,
    reason="k3_attn_res_norm is certified on its ARCHS",
)

# The thresholds of the family attn_res tests (main's op test metric).
MIN_COSINE = 0.9999
MAX_RELATIVE_L2 = 5e-3


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().view(torch.uint8)


def _similarity(actual: torch.Tensor, expected: torch.Tensor) -> tuple[float, float]:
    a, e = actual.float().flatten(), expected.float().flatten()
    cosine = torch.nn.functional.cosine_similarity(a, e, dim=0).item()
    return cosine, ((a - e).norm() / (e.norm() + 1e-12)).item()


def _inputs(spec: dict, seed: int):
    gen = torch.Generator(device="cuda").manual_seed(seed)

    def rnd(*shape, scale=1.0, shift=0.0):
        x = torch.randn(*shape, generator=gen, device="cuda") * scale + shift
        return x.to(torch.bfloat16)

    m, n, h = spec["rows"], spec["candidates"], spec["hidden"]
    layer = rnd(m, h, scale=0.5)
    addends = tuple(rnd(m, h, scale=0.25) for _ in range(spec["addends"]))
    block = rnd(n - 1, m, h, scale=0.5)
    weights = (rnd(h, scale=0.02), rnd(h, scale=0.02, shift=1.0), rnd(h, scale=0.02, shift=1.0))
    return layer, addends, block, weights


@pytest.mark.parametrize(
    "cell", k3_attn_res_norm.CELLS, ids=[c.why[:48] for c in k3_attn_res_norm.CELLS]
)
def test_certified_cells(cell) -> None:
    spec = cell.spec
    layer, addends, block, weights = _inputs(spec, zlib.crc32(cell.why.encode()) % 2**31)
    flags = dict(
        mxfp8=spec["mxfp8"],
        entry_trigger=spec["entry_trigger"],
        mx_atom=spec["mx_atom"],
        early_trigger=spec["early_trigger"],
    )
    args = (layer, addends, block, *weights, spec["rms_eps"], spec["out_eps"])
    before = [t.clone() for t in (layer, *addends, block, *weights)]
    with validating(k3_attn_res_norm):
        got = k3_attn_res_norm(*args, **flags)
        again = k3_attn_res_norm(*args, **flags)
    ref = k3_attn_res_norm.reference(*args, **flags)
    if not addends:
        got, again, ref = (got,), (again,), (ref,)
    assert len(got) == len(ref), cell.why
    atom_offsets = (
        _ENTRY.mx_atom_offsets(layer.shape[0], layer.shape[1], layer.device)
        if spec["mxfp8"] and spec["mx_atom"]
        else None
    )
    for i, (a, b) in enumerate(zip(got, again)):
        if atom_offsets is not None and i == 3:
            # Atom rows past M are not stored: compare the scales the kernel writes.
            a, b = a[atom_offsets], b[atom_offsets]
        assert torch.equal(_bits(a), _bits(b)), f"{cell.why}: a repeated call changed bits"
    index = 1 if addends else 0
    output, ref_output = got[index], ref[index]
    assert output.shape == layer.shape and output.dtype == torch.bfloat16, cell.why
    cosine, rel_l2 = _similarity(output, ref_output)
    assert cosine > MIN_COSINE and rel_l2 < MAX_RELATIVE_L2, (
        f"{cell.why}: cosine {cosine:.6f}, relative L2 {rel_l2:.3e}"
    )
    if addends:
        assert torch.equal(_bits(got[0]), _bits(ref[0])), f"{cell.why}: updated prefix differs"
    if spec["mxfp8"]:
        xq, xs = got[2], got[3]
        want_q, want_s = _ENTRY.mxfp8_rows_reference(output)
        assert torch.equal(_bits(xq), _bits(want_q)), f"{cell.why}: E4M3 codes differ"
        if spec["mx_atom"]:
            offsets = _ENTRY.mx_atom_offsets(layer.shape[0], layer.shape[1], layer.device)
            assert tuple(xs.shape) == tuple(ref[3].shape), cell.why
            assert torch.equal(xs[offsets], want_s.reshape(-1)), f"{cell.why}: atom scales differ"
            assert torch.equal(
                ref[3][offsets], _ENTRY.mxfp8_rows_reference(ref_output)[1].reshape(-1)
            )
        else:
            assert torch.equal(xs, want_s), f"{cell.why}: scales differ"
    for old, new in zip(before, (layer, *addends, block, *weights)):
        assert torch.equal(_bits(old), _bits(new)), f"{cell.why}: an input was written"


def test_is_valid_refuses_strided_rows() -> None:
    """A row-strided layer is read as dense rows by the kernel; the guard refuses it."""
    spec = k3_attn_res_norm.CELLS[0].spec
    layer, addends, block, weights = _inputs(spec, 7)
    wide = torch.cat([layer, layer], dim=1)[:, : layer.shape[1]]
    with pytest.raises(AssertionError), validating(k3_attn_res_norm):
        k3_attn_res_norm(wide, addends, block, *weights, 1e-6, 1e-6)


def test_mxfp8_without_moe_addends_raises() -> None:
    spec = k3_attn_res_norm.CELLS[3].spec
    layer, addends, block, weights = _inputs(spec, 8)
    with pytest.raises(ValueError):
        k3_attn_res_norm(layer, addends, block, *weights, 1e-6, 1e-6, mxfp8=True)
