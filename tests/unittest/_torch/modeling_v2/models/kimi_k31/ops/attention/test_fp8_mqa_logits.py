# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the Kimi K3.1 fp8_mqa_logits entry (DeepGEMM prefill logits).

Each cell is a prefill chunk: per request a cached prefix and new query rows, the chunk's keys
back to back, each row's window its request's causal keys. Only the windows are compared (the
K3.1 indexer passes clean_logits=False); with clean_logits=True the rest must be -inf. A scheduled
cell passes DeepGEMM's schedule of the call. Every cell also runs with and without a schedule,
and the two must agree bit for bit inside the windows.
"""

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.attention.fp8_mqa_logits import (  # noqa: E501
    fp8_mqa_logits as op,
)

__extra_import_path__ = ["../../../..", "."]
import _k31_dsa_inputs as inputs  # noqa: E402
from _validating import certifying, validating  # noqa: E402,F401

pytestmark = inputs.on_sm103


def _schedule(args: dict) -> torch.Tensor:
    """DeepGEMM's schedule of the call ``args`` describe."""
    from tensorrt_llm.deep_gemm import get_mqa_logits_metadata

    return get_mqa_logits_metadata(
        args["cu_seqlen_ks"], args["cu_seqlen_ke"], args["k"].shape[0], args["q"].shape[1]
    )


def _build(spec, seed: int) -> dict:
    gen = torch.Generator().manual_seed(seed)
    ks, ke, num_keys = inputs.chunk_windows(spec["requests"], spec["q_rows"])
    rows = ks.shape[0]
    k, k_scale = inputs.keys(num_keys, spec["data"], gen)
    args = dict(
        q=inputs.queries(rows, spec["data"], gen),
        k=k.cuda(),
        k_scale=k_scale.cuda(),
        weights=inputs.weights(rows, spec["data"], gen),
        cu_seqlen_ks=ks,
        cu_seqlen_ke=ke,
        clean_logits=spec["clean_logits"],
    )
    if spec["scheduled"]:
        args["schedule_meta"] = _schedule(args)
    return args


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:48] for c in op.CELLS])
def test_certified_cells(cell, certifying):  # noqa: F811
    certifying(op)
    args = _build(cell.spec, inputs.seed_of(cell))
    out = op(**args)
    again = op(**args)
    ref = op.reference(**args)
    torch.cuda.synchronize()
    assert out.shape == ref.shape and out.dtype == torch.float32
    window = torch.isfinite(ref)
    assert torch.equal(out[window], again[window]), "two identical calls disagree"
    if cell.spec["clean_logits"]:
        assert torch.isneginf(out[~window]).all(), "clean_logits left a finite value outside"
    if cell.spec["data"] == "int":
        assert torch.equal(out[window], ref[window])
    else:
        kf = args["k"].double() * args["k_scale"].double()[:, None]
        band = inputs.accumulation_band(args["q"], kf, args["weights"], window)
        err = (out.double() - ref.double()).masked_fill(~window, 0).abs()
        assert (err <= band).all(), f"max excess {float((err - band).max()):.3e}"


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:48] for c in op.CELLS])
def test_a_schedule_places_work_and_changes_no_logit(cell):
    """The call with DeepGEMM's schedule against the same call without one: equal bit for bit in
    every row's window, and -inf outside when the call cleans."""
    args = _build(cell.spec, inputs.seed_of(cell))
    args.pop("schedule_meta", None)
    plain = op(**args)
    scheduled = op(**args, schedule_meta=_schedule(args))
    torch.cuda.synchronize()
    cols = torch.arange(plain.shape[1], device="cuda")[None, :]
    ks, ke = args["cu_seqlen_ks"], args["cu_seqlen_ke"]
    window = (cols >= ks.long()[:, None]) & (cols < ke.long()[:, None])
    assert torch.equal(plain[window], scheduled[window]), "the schedule changed a logit"
    if cell.spec["clean_logits"]:
        assert torch.isneginf(scheduled[~window]).all(), "clean_logits left a finite value outside"


def test_guard_refuses_a_window_past_the_keys():
    args = _build(op.CELLS[1].spec, seed=7)
    ke = args["cu_seqlen_ke"].clone()
    ke[-1] = args["k"].shape[0] + 1
    with pytest.raises(AssertionError, match="past the key matrix"):
        with validating(op):
            op(**dict(args, cu_seqlen_ke=ke))


def test_guard_refuses_a_schedule_built_for_another_call():
    """A schedule of the first half of the rows is not this call's."""
    args = _build(op.CELLS[3].spec, seed=11)
    half = args["cu_seqlen_ks"].shape[0] // 2
    other = _schedule(
        dict(
            args,
            cu_seqlen_ks=args["cu_seqlen_ks"][:half].contiguous(),
            cu_seqlen_ke=args["cu_seqlen_ke"][:half].contiguous(),
        )
    )
    with pytest.raises(AssertionError, match="another call"):
        with validating(op):
            op(**args, schedule_meta=other)
