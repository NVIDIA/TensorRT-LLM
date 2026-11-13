# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the ssm/fused_recurrent_kda op (FLA's ``fused_recurrent_kda`` with Kimi K3's flags).

Inputs as Kimi K3's sequential verify passes them: q / k / v the short-conv outputs (contiguous bf16
``[B, T, H, 128]``, not normalized), g the f_b gate and beta the fp32 logits as token slices of the round's
``[B, 1 + num_spec, ...]`` tensors (strided views), A_log / dt_bias the module's bf16 parameters, the dense fp32
V-first states gathered from the pool rows, lower bound -5. The reference is the entry's fp32 delta rule: outputs
within 2e-2 (bf16) and states within 1e-3 (fp32) of its largest entry; the kernel's exp / sigmoid and summation
order are the only differences.
"""

import importlib
import zlib

import pytest
import torch

__extra_import_path__ = ["../../.."]
from _validating import validating

fused_recurrent_kda = importlib.import_module(
    "tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl."
    "kimi_k3_mxfp4__sm_100__tp16_moetp4ep4.ops.ssm.fused_recurrent_kda"
).fused_recurrent_kda

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
    reason="certified on SM100 and SM103",
)

D = 128
LOWER_BOUND = -5.0
ROUND = 8  # tokens of one verify round (1 + 7 drafts); the call takes a slice of them
TOL_OUT, TOL_STATE = 2e-2, 1e-3


def _rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return ((a.float() - b.float()).abs().max() / b.float().abs().max().clamp_min(1e-6)).item()


def _make(rows: int, heads: int, gen: torch.Generator) -> dict:
    def rnd(*s, scale=1.0):
        return torch.randn(*s, generator=gen, device="cuda") * scale

    return {
        "q": rnd(rows, ROUND, heads * D).bfloat16().view(rows, ROUND, heads, D),
        "k": rnd(rows, ROUND, heads * D).bfloat16().view(rows, ROUND, heads, D),
        "v": rnd(rows, ROUND, heads * D, scale=0.5).bfloat16().view(rows, ROUND, heads, D),
        "g": rnd(rows, ROUND, heads, D).bfloat16(),
        "beta": rnd(rows, ROUND, heads),
        "A_log": rnd(heads, scale=0.5).bfloat16(),
        "dt_bias": rnd(heads * D, scale=0.5).bfloat16(),
        "state": rnd(rows, heads, D, D, scale=0.05),
    }


def _args(t: dict, start: int, tokens: int, state: torch.Tensor) -> tuple:
    """Tokens ``[start, start + tokens)`` of the round: q / k / v contiguous (the conv's fresh outputs), g and beta
    the round tensors' slices."""
    sl = slice(start, start + tokens)
    q, k, v = (t[n][:, sl].contiguous() for n in ("q", "k", "v"))
    return (q, k, v, t["g"][:, sl], t["beta"][:, sl], t["A_log"], t["dt_bias"], state, LOWER_BOUND)


@pytest.mark.parametrize(
    "cell", fused_recurrent_kda.CELLS, ids=[c.why[:40] for c in fused_recurrent_kda.CELLS]
)
def test_call_against_reference(cell) -> None:
    """One call: o and the final state against fp32; initial_state and every input untouched; the final state a new
    fp32 tensor; a second call gives the same bits."""
    rows, heads, tokens = cell.spec["rows"], cell.spec["heads"], cell.spec["tokens"]
    gen = torch.Generator(device="cuda").manual_seed(zlib.crc32(cell.why.encode()) % 2**31)
    t = _make(rows, heads, gen)
    args = _args(t, 1, tokens, t["state"])
    if rows > 1:  # a one-row slice is contiguous by torch's rule
        assert not args[3].is_contiguous() and not args[4].is_contiguous()
    before = [a.clone() if isinstance(a, torch.Tensor) else a for a in args]
    with torch.inference_mode():
        with validating(fused_recurrent_kda):
            o, final = fused_recurrent_kda(*args)
        o2, final2 = fused_recurrent_kda(*args)
        o_ref, final_ref = fused_recurrent_kda.reference(*args)
        torch.cuda.synchronize()
    assert o.shape == (rows, tokens, heads, D) and o.dtype == torch.bfloat16
    assert final.shape == (rows, heads, D, D) and final.dtype == torch.float32
    assert final.data_ptr() != t["state"].data_ptr()
    assert torch.isfinite(o).all() and torch.isfinite(final).all()
    assert _rel(o, o_ref) <= TOL_OUT, _rel(o, o_ref)
    assert _rel(final, final_ref) <= TOL_STATE, _rel(final, final_ref)
    for a, b in zip(args, before):
        if isinstance(a, torch.Tensor):
            assert torch.equal(a, b), "an input was written"
    assert torch.equal(o, o2) and torch.equal(final, final2)


def test_sequential_verify_chain() -> None:
    """The sequential verify of one round: ROUND single-token calls, each from the previous final state, against the
    reference chained the same way; the starting state untouched."""
    rows, heads = 4, 6
    t = _make(rows, heads, torch.Generator(device="cuda").manual_seed(17))
    start = t["state"].clone()
    with torch.inference_mode():
        state, ref_state = t["state"], t["state"]
        for step in range(ROUND):
            o, state = fused_recurrent_kda(*_args(t, step, 1, state))
            o_ref, ref_state = fused_recurrent_kda.reference(*_args(t, step, 1, ref_state))
            torch.cuda.synchronize()
            assert _rel(o, o_ref) <= TOL_OUT, (step, _rel(o, o_ref))
            assert _rel(state, ref_state) <= TOL_STATE, (step, _rel(state, ref_state))
    assert torch.equal(t["state"], start)
