# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the silu_mul_nvfp4_permuted entry.

Cells against the reference (un-permute, torch's SiLU, native-torch NVFP4) in the band
of `_nvfp4_checks.assert_nvfp4_matches_reference`, on gate_up outputs with zeros,
negative zeros, outliers, all-zero blocks and subnormal products. The exactness claim
on its own: bitwise `silu_and_mul` + `fp4_quantize` of the un-permuted columns, the
whole scale buffer included (under a poisoned allocation, so unwritten padding shows),
for both kernels; which kernel runs at which rows; graph replay; the scale guard.
"""

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.qwen3_5.qwen3_8_27b_nvfp4__sm_120__tp1.ops.activation import (  # noqa: E501
    _silu_mul_nvfp4 as kernels,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen3_5.qwen3_8_27b_nvfp4__sm_120__tp1.ops.activation.silu_mul_nvfp4_permuted import (  # noqa: E501
    silu_mul_nvfp4_permuted as op,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen3_5.qwen3_8_27b_nvfp4__sm_120__tp1.ops.quantization._nvfp4_recipes import (  # noqa: E501
    gate_up_swiglu_row_permutation,
)
from tensorrt_llm._utils import get_sm_version

__extra_import_path__ = ["../../../..", ".."]
from _nvfp4_checks import assert_nvfp4_matches_reference, capture, seed_of  # noqa: E402
from _validating import certifying, validating  # noqa: E402,F401

pytestmark = pytest.mark.skipif(
    get_sm_version() != 120, reason="the Qwen3.8-27B sm_120 tp1 entries are certified on sm_120"
)

DEV = "cuda"

#: Block scales allowed to differ from the reference: Triton's sigmoid and torch's
#: differ by a few fp32 ulps, which moves the BF16 product in roughly 1e-4 of the
#: elements; a moved element changes its block's scale only as the block maximum
#: crossing an E4M3 rounding boundary (~1/16 of those).
MAX_SF_FRAC = 1e-3


def _gate_up(m: int, d: int, seed: int) -> torch.Tensor:
    """A gate_up output `[m, 2d]` (unpermuted order) with the corner values: 1% zeros,
    1% negative zeros, 0.5% x60 outliers, whole zero 16-blocks, and a few gate/up pairs
    whose product is subnormal."""
    g = torch.Generator(device=DEV).manual_seed(seed)
    h = torch.randn(m, 2 * d, generator=g, device=DEV) * 2.0
    u = torch.rand(m, 2 * d, generator=g, device=DEV)
    h = torch.where(u < 0.01, torch.zeros_like(h), h)
    h = torch.where((u >= 0.01) & (u < 0.02), torch.full_like(h, -0.0), h)
    h = torch.where((u >= 0.02) & (u < 0.025), h * 60.0, h)
    h[:, 32:48] = 0.0  # gate block of outputs 32..47 zero
    h[:, d + 64 : d + 80] = -0.0  # up block of outputs 64..79 negative zero
    h[0, 96:112] = 2.0
    h[0, d + 96 : d + 112] = 1e-38  # subnormal products
    return h.to(torch.bfloat16)


def _permuted(h: torch.Tensor) -> torch.Tensor:
    perm = gate_up_swiglu_row_permutation(h.shape[1]).to(h.device)
    return h[:, perm].contiguous()


def _scale(seed: int) -> torch.Tensor:
    return torch.tensor([(33.6, 0.37, 112.0)[seed % 3]], dtype=torch.float32, device=DEV)


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811
    certifying(op)
    s = cell.spec
    seed = seed_of(cell.why)
    x = _permuted(_gate_up(s["m"], s["d"], seed))
    gs = _scale(seed)
    out = op(x, gs)
    assert out[0].shape == (s["m"], s["d"] // 2)
    ref = op.reference(x, gs)
    # The reference computes SiLU with torch's fp32 sigmoid; the kernel rounds the gated
    # product to BF16 exactly as the in-tree silu_and_mul does (see the bitwise test below),
    # so an intermediate an ulp away from torch's crosses an E2M1 boundary for ~1% of the
    # codes (measured 1.03% to 1.14% over these cells on sm_120); the band is twice that.
    assert_nvfp4_matches_reference(
        out, ref, s["m"], s["d"], max_sf_frac=MAX_SF_FRAC, max_code_frac=0.02
    )


@pytest.mark.parametrize("m", [1, 4, 5, 17, 32, 33, 40, 129], ids=lambda m: f"m{m}")
def test_bitwise_silu_and_mul_then_fp4_quantize(m) -> None:
    """Both kernels: the same FP4 bytes and the same full scale buffer as the two-op
    path on the un-permuted columns. The allocator is poisoned first, so a padding
    byte either kernel failed to zero would show."""
    d = 17408
    h = _gate_up(m, d, seed=m)
    gs = _scale(m)
    torch.empty(64 << 20, dtype=torch.uint8, device=DEV).fill_(0xAB)
    fp4, sf = op(_permuted(h), gs)
    want_fp4, want_sf = torch.ops.trtllm.fp4_quantize(
        torch.ops.trtllm.silu_and_mul(h), gs, 16, False, True
    )
    assert torch.equal(fp4, want_fp4.view(torch.uint8))
    assert torch.equal(sf, want_sf.view(torch.uint8))


def test_decode_kernel_runs_exactly_at_rows_5_to_32() -> None:
    d = 17408
    assert [m for m in range(1, 64) if kernels.uses_v2(m, d)] == list(range(5, 33))


def test_graph_replay_matches_eager() -> None:
    x = _permuted(_gate_up(8, 17408, seed=1))
    gs = _scale(1)
    graph, (fp4, sf) = capture(lambda: op(x, gs))
    for trial in range(3):
        x.copy_(_permuted(_gate_up(8, 17408, seed=20 + trial)))
        fp4.fill_(0xFF)
        sf.fill_(0xFF)
        graph.replay()
        torch.cuda.synchronize()
        want = op(x, gs)
        assert torch.equal(fp4, want[0]) and torch.equal(sf, want[1])


def test_guard_refuses_a_per_token_scale() -> None:
    x = _permuted(_gate_up(4, 17408, seed=2))
    with validating(op), pytest.raises(AssertionError, match="exactly one element"):
        op(x, _scale(2).repeat(4))
