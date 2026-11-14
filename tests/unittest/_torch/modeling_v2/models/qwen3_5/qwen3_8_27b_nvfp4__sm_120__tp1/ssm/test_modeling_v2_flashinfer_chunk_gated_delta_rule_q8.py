# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the qwen3.8-27b sm_120 tp1 flashinfer_chunk_gated_delta_rule_q8 entry.

Each cell against the entry's float64 sequential reference from the same dequantized
states, inputs in the domain the one-pass front end emits (unit-norm bf16 q/k, log-space
g < 0, beta a sigmoid):

* `o` in the band the chunked kernels are certified at (rtol 1.6e-2, atol 2e-3): a
  64-token WY reformulation with bf16 MMA operands, the same class of sum on every
  architecture (measured against fp64 on sm_103 at 1 x 8192 tokens: max abs 1.26e-3).
* the final states per value row: each pool within `sqrt(K) * scale / 2` of
  its own fp32 row (one int16 rounding a side), and the two fp32 rows within
  `1.5e-2 * |row| + 1e-5` -- the chunk-versus-sequential slack of the bf16-MMA chunked
  recurrence against the fp64 sequential reference. Measured on sm_120 over these cells:
  0.71-0.94% of the row norm beyond the int16 rounding. (The earlier `1e-3` never held;
  the int8 pool's rounding band, ~2.2% of the row norm, hid it.)
* slots the call did not name bitwise; `out` written in place.

At 188 SMs the cell's kernel (CP / V-split / fully fused) is asserted from the host
plan. Then the properties: V-split bitwise the fully fused kernel, the plan's cut points
(host only), and the guard on shared slot ids.
"""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.qwen3_5.qwen3_8_27b_nvfp4__sm_120__tp1.ops.ssm._fi_gdn_prefill import (  # noqa: E501
    plan,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen3_5.qwen3_8_27b_nvfp4__sm_120__tp1.ops.ssm.flashinfer_chunk_gated_delta_rule_q8 import (  # noqa: E501
    flashinfer_chunk_gated_delta_rule_q8 as op,
)
from tensorrt_llm._utils import get_sm_version

__extra_import_path__ = ["../../../..", ".."]
from _gdn_checks import (  # noqa: E402
    DEV,
    HV,
    H,
    K,
    V,
    assert_q8_rows_close,
    assert_untouched,
    q8_pool,
)
from _validating import certifying, validating  # noqa: E402,F401 — path above; fixture by name

pytestmark = pytest.mark.skipif(
    get_sm_version() != 120, reason="the qwen3.8-27b sm_120 tp1 entries are certified on sm_120"
)

RTOL, ATOL_OUTPUT = 1.6e-2, 2e-3
#: The card the routes are written for.
_RTX_PRO_6000 = dict(
    num_sms=188,
    device_name="NVIDIA RTX PRO 6000 Blackwell Server Edition",
    device_capability=(12, 0),
)
_POOL_SLOTS = 102


def _build(spec, seed):
    """`(args, extras)`: `__call__`'s positional arguments, the step output `out` views, and
    the pool backing."""
    gen = torch.Generator(device=DEV).manual_seed(seed)
    seqlens = spec["seqlens"]
    n, total = len(seqlens), sum(seqlens)
    q = torch.randn(1, total, H, K, generator=gen, device=DEV)
    k = torch.randn(1, total, H, K, generator=gen, device=DEV)
    q = (q / torch.sqrt(q.pow(2).sum(-1, keepdim=True) + 1e-6)).to(torch.bfloat16)
    k = (k / torch.sqrt(k.pow(2).sum(-1, keepdim=True) + 1e-6)).to(torch.bfloat16)
    v = torch.randn(1, total, HV, V, generator=gen, device=DEV).to(torch.bfloat16)
    a_log = torch.log(torch.rand(HV, generator=gen, device=DEV) * 15 + 1)
    x = torch.randn(1, total, HV, generator=gen, device=DEV) + torch.randn(
        HV, generator=gen, device=DEV
    )
    g = (-torch.exp(a_log) * torch.nn.functional.softplus(x)).float()
    beta = torch.rand(1, total, HV, generator=gen, device=DEV).sigmoid().float()
    if spec.get("pool") == "coalesced":
        slots = [100] + [7 * i + 2 for i in range(n - 1)]
    else:
        slots = torch.randperm(_POOL_SLOTS, generator=gen, device=DEV)[:n].tolist()
    zero = {slots[0]} if spec.get("zero_slot") else set()
    filled = [s for s in slots if s not in zero] + [0, _POOL_SLOTS - 1]
    filled = [s for s in filled if s not in zero]
    backing, pool = q8_pool(_POOL_SLOTS, spec.get("pool", "compact"), filled, gen, std=0.1)
    cu = torch.tensor([0] + torch.tensor(seqlens).cumsum(0).tolist(), dtype=torch.long, device=DEV)
    slot_t = torch.tensor(slots, dtype=torch.int32, device=DEV)
    # The step output: these context rows, then 3 rider rows the call must not touch.
    step_out = torch.full((1, total + 3, HV, V), -7.0, dtype=torch.bfloat16, device=DEV)
    out = step_out[:, :total]
    args = [q, k, v, g, beta, pool, slot_t, cu, n, out]
    return args, dict(
        backing=backing, neighbors=backing[:, 1:].clone(), step_out=step_out, slots=slots
    )


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — fixture by name
    certifying(op)
    args, extras = _build(cell.spec, seed=zlib.crc32(cell.why.encode()))
    pool, out = args[5], args[9]
    seqlens = cell.spec["seqlens"]
    if torch.cuda.get_device_properties(0).multi_processor_count == _RTX_PRO_6000["num_sms"]:
        assert (
            plan(len(seqlens), sum(seqlens), len(seqlens), HV, **_RTX_PRO_6000)
            == cell.spec["route"]
        )
    pool_before = pool.clone()
    tail_before = extras["step_out"][:, sum(seqlens) :].clone()
    ref_o, ref_pool = op.reference(*args)

    o = op(*args)

    assert o is out, "out is written in place and returned"
    assert torch.equal(extras["step_out"][:, sum(seqlens) :], tail_before), "rider rows written"
    assert torch.isfinite(o.float()).all()
    torch.testing.assert_close(o.float(), ref_o.float(), rtol=RTOL, atol=ATOL_OUTPUT)
    assert_q8_rows_close(pool, ref_pool, extras["slots"], row_rel=1.5e-2, abs_slack=1e-5)
    assert_untouched(pool, pool_before, extras["slots"])
    if cell.spec.get("pool") == "coalesced":
        assert torch.equal(extras["backing"][:, 1:], extras["neighbors"]), (
            "another layer's states were written"
        )


@pytest.mark.parametrize("seqlens", [(2048,), (100,)], ids=["1x2048", "1x100"])
def test_vsplit_is_bitwise_the_fully_fused_kernel(seqlens) -> None:
    """One long sequence runs the V-split x2 kernel; `num_long_seqs=0` sends the same call
    to the fully fused kernel. V is never a reduction dimension, so `o` and the pool are
    bitwise equal."""
    args, _ = _build(dict(seqlens=seqlens), seed=zlib.crc32(str(seqlens).encode()))
    pool = args[5]
    pristine = pool.clone()
    vsplit_o = op(*args).clone()
    vsplit_pool = pool.clone()
    pool.copy_(pristine)
    args[9].fill_(-7.0)  # rows the fully fused kernel leaves unwritten cannot pass as V-split's
    args[8] = 0
    fused_o = op(*args)
    assert torch.equal(vsplit_o.view(torch.int16), fused_o.view(torch.int16)), "o differs"
    assert torch.equal(pool, vsplit_pool), "the pool differs"


@pytest.mark.parametrize(
    "seqlens, route",
    [
        ((100,), "vsplit"),
        ((2048,), "vsplit"),  # 2 CP chunks of 1024 would not pay
        ((3072,), "cp"),  # 3 chunks of 1024
        ((8192,), "cp"),
        ((2048, 2048), "fused"),  # 2 * 48 * 3 >= 188: no CP; 2 * 48 * 2 > 188: no V-split
        ((2048, 2048, 2048, 1920), "fused"),
        ((700, 1300, 64, 1), "fused"),
    ],
)
def test_plan_cut_points(seqlens, route) -> None:
    """The host plan at 188 SMs (pure host arithmetic over FlashInfer's heuristics)."""
    n = len(seqlens)
    assert plan(n, sum(seqlens), n, HV, **_RTX_PRO_6000) == route


def test_guard_refuses_shared_slot_ids() -> None:
    """Two sequences scattering into one slot leave its final state nondeterministic
    (and both start from it); the guard refuses it."""
    args, _ = _build(dict(seqlens=(64, 64)), seed=11)
    args[6] = torch.tensor([4, 4], dtype=torch.int32, device=DEV)
    with pytest.raises(AssertionError, match="sharing a slot"):
        with validating(op):
            op(*args)
