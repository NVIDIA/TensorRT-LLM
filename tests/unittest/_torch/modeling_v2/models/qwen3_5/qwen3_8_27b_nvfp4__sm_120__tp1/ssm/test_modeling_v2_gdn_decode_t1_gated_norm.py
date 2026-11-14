# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the qwen3.8-27b sm_120 tp1 gdn_decode_t1_gated_norm entry.

Two gates per cell:

1. Bitwise against the composition it replaces, on copies of the pool:
   `gdn_decode_t1` then `rms_norm_gated_token_major` -- `y` and the whole pool. This
   holds for the Triton build whose single-warp reduction order `_warp32x4_sum_sq`
   replays; a Triton upgrade that moves it fails here first.
2. Against the entry's reference (the float64 step from the dequantized states, `o`
   rounded to bf16, then the gated norm's fp32 reference): `y` in torch's default bf16
   band -- the two `o` differ by the fp32 sums' reorder, which flips a bf16 rounding of
   `o` now and then, and the norm carries that one ulp into `y`; the pool as in
   `gdn_decode_t1`'s test (two int16 roundings plus 1e-6). Slots the call did not name,
   the pad row's included, stay bitwise.

Then the guard on z's layout.
"""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.qwen3_5.qwen3_8_27b_nvfp4__sm_120__tp1.ops.norm.rms_norm_gated_token_major import (  # noqa: E501
    rms_norm_gated_token_major,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen3_5.qwen3_8_27b_nvfp4__sm_120__tp1.ops.ssm.gdn_decode_t1 import (  # noqa: E501
    POOL_SLOTS,
    gdn_decode_t1,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen3_5.qwen3_8_27b_nvfp4__sm_120__tp1.ops.ssm.gdn_decode_t1_gated_norm import (  # noqa: E501
    EPS,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen3_5.qwen3_8_27b_nvfp4__sm_120__tp1.ops.ssm.gdn_decode_t1_gated_norm import (  # noqa: E501
    gdn_decode_t1_gated_norm as op,
)
from tensorrt_llm._utils import get_sm_version

__extra_import_path__ = ["../../../..", ".."]
from _gdn_checks import (  # noqa: E402
    DEV,
    HV,
    V,
    assert_q8_states_close,
    assert_untouched,
    coalesced_slots,
    decode_inputs,
    q8_pool,
    scattered_slots,
)
from _validating import certifying, validating  # noqa: E402,F401 — path above; fixture by name

pytestmark = pytest.mark.skipif(
    get_sm_version() != 120, reason="the qwen3.8-27b sm_120 tp1 entries are certified on sm_120"
)


def _build(spec, seed):
    """`(args, extras)`: `__call__`'s positional arguments and the buffers they view."""
    gen = torch.Generator(device=DEV).manual_seed(seed)
    rows = spec["rows"]
    if spec.get("pool") == "coalesced":
        slots = coalesced_slots(rows)
        backing, pool = q8_pool(102, "coalesced", slots + [0, 101], gen)
    else:
        slots = scattered_slots(rows, POOL_SLOTS, gen, pad=spec.get("pad", False))
        backing, pool = q8_pool(POOL_SLOTS, "compact", [s for s in slots if s >= 0] + [0], gen)
    inp = decode_inputs(rows, gen)
    slot_t = torch.tensor(slots, dtype=torch.int32, device=DEV)
    args = [inp["q"], inp["k"], inp["v"], inp["a"], inp["b"], inp["A_log"], inp["dt_bias"],
            pool, slot_t, inp["z"], inp["norm_weight"], EPS]  # fmt: skip
    return args, dict(backing=backing, neighbors=backing[:, 1:].clone(), slots=slots)


def _composition(args):
    """`gdn_decode_t1` then `rms_norm_gated_token_major`, on a copy of the pool:
    `(y, pool_after)`."""
    q, k, v, a, b, A_log, dt_bias, pool, slots, z, norm_weight, eps = args
    pool_copy = pool.clone()
    o = gdn_decode_t1(q, k, v, a, b, A_log, dt_bias, pool_copy, slots, None)
    y = rms_norm_gated_token_major(o.reshape(-1, V), z, norm_weight, eps)
    return y, pool_copy


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — fixture by name
    certifying(op)
    args, extras = _build(cell.spec, seed=zlib.crc32(cell.why.encode()))
    pool = args[7]
    pool_before = pool.clone()
    want_y, want_pool = _composition(args)
    ref_y, ref_pool = op.reference(*args)

    y = op(*args)

    rows = cell.spec["rows"]
    assert y.shape == (rows * HV, V) and y.dtype == torch.bfloat16 and y.is_contiguous()
    # Gate 1: bitwise the composition.
    assert torch.equal(y.view(torch.int16), want_y.view(torch.int16)), "y differs from T1 + norm"
    assert torch.equal(pool, want_pool), "the pool differs from gdn_decode_t1's"
    # Gate 2: the reference.
    torch.testing.assert_close(y, ref_y)
    assert_q8_states_close(pool, ref_pool, extras["slots"])
    assert_untouched(pool, pool_before, extras["slots"])
    if cell.spec.get("pool") == "coalesced":
        assert torch.equal(extras["backing"][:, 1:], extras["neighbors"]), (
            "another layer's states were written"
        )


def test_guard_refuses_a_packed_z_of_the_wrong_head_stride() -> None:
    """The kernel reads head i of token n at `n * stride(0) + i * V`: a z whose heads are
    not packed V apart (here the transposed `[N, V, HV]` storage seen as `[N, HV, V]`) is
    read as the wrong gate, silently; the guard refuses it."""
    args, _ = _build(dict(rows=5), seed=9)
    z = args[9]
    args[9] = z.contiguous().transpose(1, 2).contiguous().transpose(1, 2)
    assert args[9].stride(1) != V
    with pytest.raises(AssertionError, match="contiguous per token"):
        with validating(op):
            op(*args)
