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
#: Columns past q|k|v in the target's one GDN input-projection output: z | a | b.
_IN_PROJ_TAIL = 6144 + 2 * 48


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
    elif x_form == "2d_in_proj":
        extras["buf"] = randn(_PREFILL_ROWS + batch, dim + _IN_PROJ_TAIL)
        x = extras["buf"][_PREFILL_ROWS:, :dim]
        assert x.stride() == (dim + _IN_PROJ_TAIL, 1)
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
        got[0] = got_buf[_PREFILL_ROWS:, : args[0].shape[1]]
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
        dim = args[0].shape[1]
        assert torch.equal(got_buf[:, dim:], extras["buf"][:, dim:]), "columns past x written"
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


#: One metadata violation per family the op checks: the conv width, the slot
#: indices, the operand dtypes.
_REJECTIONS = {
    "width_5": (
        "causal_conv1d only supports width between 2 and 4",
        dict(dim=128, width=5, state_len=4),
    ),
    "indices_int64": ("conv_state_indices.scalar_type", None),
    "weight_dtype": ("weight type must equal to input type", None),
}


@pytest.mark.parametrize("case", sorted(_REJECTIONS))
def test_op_rejects_metadata_violations_loudly(case: str) -> None:
    """Why the entry has no `is_valid`: the op raises on metadata violations
    itself (one probe per family here)."""
    fragment, overrides = _REJECTIONS[case]
    args = _small(**(overrides or {}))
    if case == "indices_int64":
        args[6] = args[6].long()
    elif case == "weight_dtype":
        args[2] = args[2].float()
    with pytest.raises(RuntimeError, match=fragment.replace("(", r"\(").replace(")", r"\)")):
        with validating(op):
            op(*args)


def test_decode_rows_match_the_trtllm_kernel_bitwise() -> None:
    """From 64 rows the decode shape runs the CuTe kernel: at that boundary x and
    the whole pool are bitwise what `torch.ops.trtllm.causal_conv1d_update` leaves,
    a pad row untouched.
    x is the q|k|v column view of an in_proj output; the pool a per-layer view of a
    two-layer buffer, as the engine hands both over."""
    rows = 64
    g = torch.Generator(device="cuda").manual_seed(rows)
    dim = 10240
    proj = torch.randn(rows, dim + _IN_PROJ_TAIL, generator=g, device="cuda").to(torch.bfloat16)
    pool = torch.randn(ENGINE_SLOTS, 2, dim, 3, generator=g, device="cuda").to(torch.bfloat16)[:, 1]
    weight = (torch.randn(dim, 4, generator=g, device="cuda") * 0.5).to(torch.bfloat16)
    slots = (torch.randperm(ENGINE_SLOTS, generator=g, device="cuda")[:rows]).to(torch.int32)
    slots[rows // 3] = PAD_SLOT_ID
    x_ref, pool_ref = proj.clone(), pool.clone()
    torch.ops.trtllm.causal_conv1d_update(
        x_ref[:, :dim].unsqueeze(-1), pool_ref, weight, None, True, None, slots, PAD_SLOT_ID
    )
    x_got, pool_got = proj.clone(), pool.clone()
    op(x_got[:, :dim], pool_got, weight, None, True, None, slots, PAD_SLOT_ID)
    assert torch.equal(x_got, x_ref), "x differs from the TRT-LLM kernel"
    assert torch.equal(pool_got, pool_ref), "the pool differs from the TRT-LLM kernel"
    assert torch.equal(x_got[rows // 3], proj[rows // 3]), "the pad row was written"
