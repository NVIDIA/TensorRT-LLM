# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the kda_mtp_verify entry of the Kimi K3.1 family.

Every certified cell runs the op (through its guard) and the entry's plain-torch reference on
identical inputs and compares the returned core and every tensor the call writes in place.
"""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.ssm.kda_mtp_verify import (
    kda_mtp_verify as op,
)

__extra_import_path__ = ["../../../..", "../../kernels/ssm"]
from _cells import certified_cells  # noqa: E402 — needs the path declared above
from _kda_verify_inputs import build  # noqa: E402 — needs the path declared above
from _validating import validating  # noqa: E402 — needs the path declared above

assert torch.cuda.is_available(), "kda_mtp_verify requires a CUDA device"

# The target's fused verify is the only caller; its shapes are the entry's own cells.
TARGET_CELLS: dict = {}

CELLS = certified_cells(op, TARGET_CELLS)


def _report(name, out, ref):
    diff = (out.float() - ref.float()).abs()
    rel = (diff.pow(2).mean().sqrt() / ref.float().pow(2).mean().sqrt().clamp_min(1e-12)).item()
    print(f"  {name}: max_abs={diff.max().item():.3e} rel_rms={rel:.3e}")
    return diff, rel


@pytest.mark.parametrize("cell", CELLS, ids=[c.why[:44] for c in CELLS])
def test_certified_cells(cell) -> None:
    spec = dict(cell.spec)
    use_pdl = spec.pop("use_pdl", False)
    state_evict_last = spec.pop("state_evict_last", False)
    seed = zlib.crc32(cell.why.encode()) % 2**31
    op_args = build(seed=seed, **spec)
    ref_args = build(seed=seed, **spec)
    untouched = [
        s
        for s in range(op_args["recurrent_state"].shape[0])
        if s not in op_args["ssm_state_indices"].tolist()
    ]
    before = op_args["recurrent_state"][untouched].clone()
    with validating(op):
        out = op(**op_args, use_pdl=use_pdl, state_evict_last=state_evict_last)
    ref = op.reference(**ref_args)
    torch.cuda.synchronize()

    # The kernel runs the recurrence in chunk form with 3xTF32 tensor-core products and the
    # reference one fp32 token step at a time: the two agree to fp32 rounding of a few
    # accumulation orders on the state (GB300, all cells: rel RMS <= 3e-6, max abs <= 3e-5), which
    # the bf16 rounding of the core and of the normed output turns into one-ulp flips of single
    # elements (max abs 1.6e-2 = one ulp in [2, 4), rel RMS <= 1.1e-3 over the cells). The gates
    # leave 2x on the RMS and one more ulp per element.
    assert out.dtype is torch.bfloat16 and out.shape == ref.shape
    assert torch.isfinite(out.float()).all()
    diff, rel = _report("out", out, ref)
    assert rel < 2.5e-3, f"normed core relative RMS {rel:.3e}"
    torch.testing.assert_close(out.float(), ref.float(), atol=3.2e-2, rtol=8e-3)

    slots = op_args["ssm_state_indices"].long()
    _, rel = _report("state", op_args["recurrent_state"][slots], ref_args["recurrent_state"][slots])
    assert rel < 3e-5, f"committed state relative RMS {rel:.3e}"
    torch.testing.assert_close(
        op_args["recurrent_state"][slots], ref_args["recurrent_state"][slots], atol=2e-3, rtol=1e-3
    )
    # Slots of the pool the batch does not own stay as they were.
    assert torch.equal(op_args["recurrent_state"][untouched], before)

    # Raw conv inputs are copied, not computed: exact.
    for name in ("cs_q", "cs_k", "cs_v"):
        assert torch.equal(op_args[name], ref_args[name]), name
    # Cached k / v / beta are bf16 roundings of values that agree to fp32 rounding (one ulp at
    # most); the gate is fp32 (fast exp / sigmoid on the device).
    for name in ("k_cache", "v_cache", "beta_cache"):
        torch.testing.assert_close(
            op_args[name].float(), ref_args[name].float(), atol=1e-2, rtol=8e-3
        )
    torch.testing.assert_close(op_args["g_cache"], ref_args["g_cache"], atol=2e-5, rtol=1e-5)


def test_guard_refuses_an_accepted_count_past_num_spec() -> None:
    """A count above num_spec replays rows past the cached drafts."""
    args = build(N=2, H=12, num_spec=4)
    args["num_accepted_tokens"][args["ssm_state_indices"][0].long()] = 5
    with pytest.raises(AssertionError, match="num_accepted_tokens"):
        with validating(op):
            op(**args)


def test_guard_refuses_a_repeated_slot() -> None:
    """Two requests on one slot race on the in-place state commit."""
    args = build(N=2, H=12, num_spec=4, pool=4)
    args["ssm_state_indices"][1] = args["ssm_state_indices"][0]
    with pytest.raises(AssertionError, match="distinct"):
        with validating(op):
            op(**args)


def test_guard_refuses_a_slot_outside_the_pool() -> None:
    args = build(N=2, H=12, num_spec=4, pool=2)
    args["ssm_state_indices"][1] = 2
    with pytest.raises(AssertionError, match=r"\[0, pool\)"):
        with validating(op):
            op(**args)
