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
#: Rows of prefill ahead of the decode rows in the mixed-batch cell.
_PREFILL_ROWS = 37


def _build(spec, seed):
    """`__call__`'s arguments for one spec, plus the buffers some layouts view."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    dim, batch = spec["dim"], spec["batch"]
    width, state_len, seqlen = spec.get("width", 4), spec.get("state_len", 3), spec.get("seqlen", 1)
    dtype, x_form = spec.get("dtype", torch.bfloat16), spec.get("x_form", "3d")

    def randn(*shape):
        return torch.randn(*shape, generator=g, device="cuda").to(dtype)

    extras = {}
    if x_form == "2d":
        x = randn(batch, dim)
    elif x_form == "2d_row_slice":
        extras["buf"] = randn(_PREFILL_ROWS + batch, dim)
        x = extras["buf"][_PREFILL_ROWS:]
    elif x_form == "token_major":
        x = randn(batch, seqlen, dim).transpose(1, 2)
        assert x.stride() == (seqlen * dim, 1, dim)
    else:
        x = randn(batch, dim, seqlen)
    if spec.get("strided_state", False):
        extras["wide"] = randn(spec["pool_slots"], dim, 2 * state_len)
        conv_state = extras["wide"][:, :, ::2]
    else:
        conv_state = randn(spec["pool_slots"], dim, state_len)
    weight = randn(dim, width)
    bias = randn(dim) if spec.get("bias", False) else None
    slots = spec["slots"]
    if slots == "random":
        indices = torch.randperm(spec["pool_slots"], generator=g, device="cuda")[:batch]
        indices = indices.to(torch.int32)
    elif slots is None:
        indices = None
    else:
        indices = torch.tensor(slots, dtype=torch.int32, device="cuda")
    cache_seqlens = spec.get("cache_seqlens")
    if cache_seqlens is not None:
        cache_seqlens = torch.tensor(cache_seqlens, dtype=torch.int32, device="cuda")
    silu, pad = spec.get("silu", True), spec.get("pad_slot_id", PAD_SLOT_ID)
    return [x, conv_state, weight, bias, silu, cache_seqlens, indices, pad], extras


def _run(args, extras):
    """Drive the op on copies of x and the pool, through the same views."""
    got = list(args)
    if "buf" in extras:
        got_buf = extras["buf"].clone()
        got[0] = got_buf[_PREFILL_ROWS:]
    else:
        got[0] = args[0].clone()
    if "wide" in extras:
        got_wide = extras["wide"].clone()
        got[1] = got_wide[:, :, ::2]
    else:
        got[1] = args[1].clone()
    returned = op(*got)
    assert returned is got[0], "returns the x it was handed"
    if "buf" in extras:
        assert torch.equal(got_buf[:_PREFILL_ROWS], extras["buf"][:_PREFILL_ROWS])
    if "wide" in extras:
        assert torch.equal(got_wide[:, :, 1::2], extras["wide"][:, :, 1::2])
    return returned, got[1]


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — pytest resolves the fixture by this name
    certifying(op)
    args, extras = _build(cell.spec, seed=zlib.crc32(cell.why.encode()))
    weight, bias = args[2].clone(), None if args[3] is None else args[3].clone()
    ref = op.reference(*args)
    returned, state = _run(args, extras)
    assert torch.equal(args[2], weight), "weight is read-only"
    assert bias is None or torch.equal(args[3], bias), "bias is read-only"
    _compare((returned, state), ref)
    if cell.spec["pool_slots"] == ENGINE_SLOTS:
        # Non-vacuity for the large pool: every named slot moved.
        for slot in cell.spec["slots"]:
            assert not torch.equal(state[slot], args[1][slot]), slot


def _small(**overrides):
    spec = dict(dim=512, batch=4, pool_slots=16, slots=(0, 1, 2, 3))
    spec.update(overrides)
    return _build(spec, seed=14)[0]


_REJECTIONS = {
    "width_1": (
        "causal_conv1d only supports width between 2 and 4",
        dict(dim=128, width=1, state_len=1),
    ),
    "width_5": (
        "causal_conv1d only supports width between 2 and 4",
        dict(dim=128, width=5, state_len=4),
    ),
    "state_too_short": ("conv_state_len >= width - 1", dict(state_len=2)),
    "indices_length": ("conv_state_indices must have shape", None),
    "indices_int64": ("conv_state_indices.scalar_type", None),
    "indices_non_contiguous": ("is_contiguous", None),
    "cache_seqlens_int64": ("cache_seqlens.scalar_type", None),
    "no_indices_oversize_pool": ("conv_state must have shape", dict(slots=None)),
    "weight_shape": ("weight must have shape", None),
    "conv_state_dtype": ("conv_state.scalar_type", None),
    "weight_dtype": ("weight type must equal to input type", None),
    "bias_dtype": ("bias.scalar_type", None),
    "float64": ("input_type == at::ScalarType::Float", None),
}


@pytest.mark.parametrize("case", sorted(_REJECTIONS))
def test_op_rejects_every_metadata_violation_loudly(case: str) -> None:
    """Why the entry has no `is_valid`: every metadata violation probed raises."""
    fragment, overrides = _REJECTIONS[case]
    args = _small(**(overrides or {}))
    if case == "indices_length":
        args[6] = args[6][:3]
    elif case == "indices_int64":
        args[6] = args[6].long()
    elif case == "indices_non_contiguous":
        args[6] = torch.arange(8, dtype=torch.int32, device="cuda")[::2]
    elif case == "cache_seqlens_int64":
        args[5] = torch.zeros(4, dtype=torch.int64, device="cuda")
    elif case == "weight_shape":
        args[2] = torch.randn(256, 4, dtype=torch.bfloat16, device="cuda")
    elif case == "conv_state_dtype":
        args[1] = args[1].float()
    elif case == "weight_dtype":
        args[2] = args[2].float()
    elif case == "bias_dtype":
        args[3] = torch.randn(512, dtype=torch.float32, device="cuda")
    elif case == "float64":
        args[0], args[1], args[2] = args[0].double(), args[1].double(), args[2].double()
    with pytest.raises(RuntimeError, match=fragment.replace("(", r"\(").replace(")", r"\)")):
        with validating(op):
            op(*args)
