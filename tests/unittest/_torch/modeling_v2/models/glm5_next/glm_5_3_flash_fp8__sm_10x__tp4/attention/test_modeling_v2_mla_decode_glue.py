# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU tests for the glm5_next fp8 tp4 MLA decode entries mla_prologue, kpool_select and
latent_values."""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.glm5_next.glm_5_3_flash_fp8__sm_10x__tp4.ops.attention.kpool_select import (  # noqa: E501
    kpool_select,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.glm5_next.glm_5_3_flash_fp8__sm_10x__tp4.ops.attention.latent_values import (  # noqa: E501
    latent_values,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.glm5_next.glm_5_3_flash_fp8__sm_10x__tp4.ops.attention.mla_prologue import (  # noqa: E501
    mla_prologue,
)
from tensorrt_llm._utils import get_sm_version

__extra_import_path__ = ["../../../.."]
from _validating import certifying  # noqa: E402,F401

pytestmark = pytest.mark.skipif(
    get_sm_version() != 100, reason="the glm5_next fp8 tp4 entries are certified on sm_100"
)

_TPB, _SLOTS = 32, 512


def _prologue_args(spec: dict, seed: int):
    g = torch.Generator(device="cuda").manual_seed(seed)
    n, t = spec["n"], spec["t"]

    def rnd(*shape, scale=1.0):
        return torch.randn(*shape, generator=g, device="cuda") * scale

    qkv = rnd(n * t, 2048).to(torch.bfloat16)
    idx = rnd(n * t, 288).to(torch.bfloat16)
    weights = [(rnd(d) * 0.2 + 1.0).to(torch.bfloat16) for d in (1536, 512, 128)]
    bias = rnd(128, scale=0.1).to(torch.bfloat16)
    positions = torch.randint(64, 600, (n, 1), generator=g, device="cuda") + torch.arange(
        t, device="cuda"
    )
    tables = torch.randperm(_SLOTS, generator=g, device="cuda")[: n * 20].view(n, 20).int()
    latent_pool = torch.zeros(_SLOTS, _TPB, 512, device="cuda", dtype=torch.float8_e4m3fn)
    index_pool = torch.zeros(_SLOTS, _TPB, 288, device="cuda", dtype=torch.bfloat16)
    scale = torch.tensor(0.7, device="cuda")
    return [
        qkv,
        idx,
        *weights,
        bias,
        1e-5,
        1e-6,
        positions,
        tables,
        latent_pool,
        index_pool,
        _TPB,
        scale,
    ]


@pytest.mark.parametrize("cell", mla_prologue.CELLS, ids=[c.why[:44] for c in mla_prologue.CELLS])
def test_prologue_matches_reference(cell, certifying):  # noqa: F811
    """The rows, the latent cache and the indexer cache equal the reference's (same FP32 math)."""
    certifying(mla_prologue)
    args = _prologue_args(cell.spec, zlib.crc32(cell.why.encode()) % 2**31)
    mla_prologue.is_valid(*args)
    ref_args = list(args)
    ref_args[10], ref_args[11] = args[10].clone(), args[11].clone()
    out = mla_prologue(*args)
    ref = mla_prologue.reference(*ref_args)
    torch.testing.assert_close(
        out.float(), ref.float(), rtol=0.0, atol=2**-7 * ref.abs().max().item()
    )
    assert torch.equal(args[10].view(torch.uint8) != 0, ref_args[10].view(torch.uint8) != 0)
    diff = (args[10].float() - ref_args[10].float()).abs()
    assert diff.max().item() <= 0.125 * ref_args[10].float().abs().max().item()
    torch.testing.assert_close(args[11].float(), ref_args[11].float(), rtol=0.0, atol=2**-6)


@pytest.mark.parametrize("cell", latent_values.CELLS, ids=[c.why[:44] for c in latent_values.CELLS])
def test_latent_values_matches_reference(cell, certifying):  # noqa: F811
    """FP32 accumulation of BF16 products, one BF16 rounding; rows without a key are zero, NaN
    latents included."""
    certifying(latent_values)
    g = torch.Generator(device="cuda").manual_seed(zlib.crc32(cell.why.encode()) % 2**31)
    rows, heads, lat, v = (cell.spec[k] for k in ("r", "h", "l", "v"))
    latent = torch.randn(rows, heads, lat, generator=g, device="cuda").to(torch.bfloat16)
    w_v_t = (torch.randn(heads, lat, v, generator=g, device="cuda") * 0.05).to(torch.bfloat16)
    keep = torch.ones(rows, dtype=torch.bool, device="cuda")
    keep[-1] = False
    latent[-1] = float("nan")
    latent_values.is_valid(latent, w_v_t, keep)
    out = latent_values(latent, w_v_t, keep)
    ref = latent_values.reference(latent, w_v_t, keep)
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out.float(), ref.float(), rtol=2**-7, atol=1e-3)


@pytest.mark.parametrize("cell", kpool_select.CELLS, ids=[c.why[:44] for c in kpool_select.CELLS])
def test_kpool_select_matches_reference(cell, certifying):  # noqa: F811
    """Packed rows, lengths and the nonempty mask equal top-k + expand + compaction of the pools
    sorted by index; the last row sees no key."""
    certifying(kpool_select)
    spec = cell.spec
    g = torch.Generator(device="cuda").manual_seed(zlib.crc32(cell.why.encode()) % 2**31)
    rows, tpb, kpool = spec["rows"], 32, 4
    blocks = spec["pools"] * kpool // tpb
    tables = torch.randperm(2 * blocks, generator=g, device="cuda")[:blocks].repeat(rows, 1)
    kv_lens = spec["kv"] - torch.arange(rows, device="cuda").flip(0) % 4
    kv_lens[-1] = 0
    scores = torch.randn(rows, spec["pools"], generator=g, device="cuda")
    args = (scores, kv_lens, tables, tpb)
    kwargs = dict(select_k=512, kpool=kpool, base_row=3, rows_per_slot=384, out_width=2051)
    kpool_select.is_valid(*args, **kwargs)
    out = kpool_select(*args, **kwargs)
    ref = kpool_select.reference(*args, **kwargs)
    for got, want in zip(out, ref):
        assert torch.equal(got, want)
