# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the causal_conv1d_update entry."""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_103__tp1.ops.ssm.causal_conv1d_update import (  # noqa: E501
    ENGINE_SLOTS,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_103__tp1.ops.ssm.causal_conv1d_update import (  # noqa: E501
    causal_conv1d_update as op,
)

__extra_import_path__ = ["../../../.."]
from _validating import certifying, validating  # noqa: E402,F401 — path above; fixture by name

assert torch.cuda.is_available(), "causal_conv1d_update requires a CUDA device"


def _compare(
    out: tuple[torch.Tensor, torch.Tensor], ref: tuple[torch.Tensor, torch.Tensor]
) -> None:
    """`(x, conv_state)` pairs: the returned `x` with the pool as the call
    left it, against `reference`'s pair.

    Activation at torch's default band: bf16/fp16 were measured bitwise
    equal to the fp32 reference in every cell; fp32 differs by tap order
    (max 1.43e-6, inside the default fp32 atol). The pool is a copy of
    input values, so it is gated bitwise -- over the whole pool, which is
    the lock on every slot the batch did not name.
    """
    (act, state), (ref_act, ref_state) = out, ref
    assert act.dtype == ref_act.dtype and act.shape == ref_act.shape
    torch.testing.assert_close(act, ref_act)
    assert torch.equal(state, ref_state), "the advanced state must be bitwise exact"


PAD_SLOT_ID = -1
#: Rows of prefill ahead of the decode rows in the in_proj output.
_PREFILL_ROWS = 37
#: Columns past q|k|v in the target's one GDN input-projection output: z | a | b.
_IN_PROJ_TAIL = 6144 + 2 * 48


def _build(spec, seed):
    """`__call__`'s arguments for one spec, plus the buffers x and the pool view."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    dim, batch, layers = spec["dim"], spec["batch"], spec.get("pool_layers", 1)

    def randn(*shape):
        return torch.randn(*shape, generator=g, device="cuda").to(torch.bfloat16)

    buf = randn(_PREFILL_ROWS + batch, dim + _IN_PROJ_TAIL)
    x = buf[_PREFILL_ROWS:, :dim]
    assert x.stride() == (dim + _IN_PROJ_TAIL, 1)
    pool_buf = randn(spec["pool_slots"], layers, dim, 3)
    conv_state = pool_buf[:, layers - 1]
    weight = randn(dim, 4)
    slots = spec["slots"]
    if slots == "random":
        indices = torch.randperm(spec["pool_slots"], generator=g, device="cuda")[:batch]
        indices = indices.to(torch.int32)
    else:
        indices = torch.tensor(slots, dtype=torch.int32, device="cuda")
    args = [x, conv_state, weight, None, True, None, indices, PAD_SLOT_ID]
    return args, dict(buf=buf, pool_buf=pool_buf)


def _run(args, extras):
    """Drive the op on copies of x and the pool, through the same views."""
    got = list(args)
    got_buf, got_pool = extras["buf"].clone(), extras["pool_buf"].clone()
    got[0] = got_buf[_PREFILL_ROWS:, : args[0].shape[1]]
    got[1] = got_pool[:, got_pool.shape[1] - 1]
    returned = op(*got)
    assert returned is got[0], "returns the x it was handed"
    assert torch.equal(got_buf[:_PREFILL_ROWS], extras["buf"][:_PREFILL_ROWS])
    dim = args[0].shape[1]
    assert torch.equal(got_buf[:, dim:], extras["buf"][:, dim:]), "columns past x written"
    assert torch.equal(got_pool[:, :-1], extras["pool_buf"][:, :-1]), "another layer written"
    return returned, got[1]


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — pytest resolves the fixture by this name
    certifying(op)
    args, extras = _build(cell.spec, seed=zlib.crc32(cell.why.encode()))
    weight = args[2].clone()
    ref = op.reference(*args)
    returned, state = _run(args, extras)
    assert torch.equal(args[2], weight), "weight is read-only"
    _compare((returned, state), ref)
    if cell.spec["pool_slots"] == ENGINE_SLOTS:
        # Non-vacuity for the large pool: every named slot moved.
        for slot in cell.spec["slots"]:
            if slot != PAD_SLOT_ID:
                assert not torch.equal(state[slot], args[1][slot]), slot


def test_op_rejects_metadata_violations_loudly() -> None:
    """Why the entry has no `is_valid`: the op raises on metadata violations
    itself (probed here at int64 slot indices)."""
    spec = dict(dim=512, batch=4, pool_slots=16, slots=(0, 1, 2, 3))
    args = _build(spec, seed=14)[0]
    args[6] = args[6].long()
    with pytest.raises(RuntimeError, match="conv_state_indices.scalar_type"):
        with validating(op):
            op(*args)
