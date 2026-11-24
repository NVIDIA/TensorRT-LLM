# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the rubin_kda_prefill entry (sm_107).

Each cell runs the op through its guard and the entry's plain-torch reference on identical inputs:
the output, the selected state rows and the snapshot rows within the chunked-prefill bands of an
fp32 recurrence, every other pool row bit for bit.
"""

import zlib

import pytest
import torch
from _rubin_kda_prefill_inputs import (
    HEAD_DIM,
    LOWER_BOUND,
    SCALE,
    assert_scaled_close,
    clone_written,
    cu_seqlens_of,
    prefill_args,
    step_schedule,
)

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.ssm.rubin_kda_prefill import (
    rubin_kda_prefill,
)

__extra_import_path__ = ["../../../.."]
from _validating import certifying, validating  # noqa: E402,F401 — path above; fixture by name

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7),
    reason="Rubin entry: certified on sm_107",
)

# 16-token chunks with bf16 tensor-core products (the entry's note); a bf16 pool adds one rounding
# of the state per chunk.
TOL_OUT = 2e-2
TOL_STATE = {torch.float32: 1e-2, torch.bfloat16: 2e-2}


def _seed(cell) -> int:
    return zlib.crc32(cell.why.encode()) % 2**31


@pytest.mark.parametrize(
    "cell", rubin_kda_prefill.CELLS, ids=[c.why[:60] for c in rubin_kda_prefill.CELLS]
)
@torch.inference_mode()
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — pytest resolves the fixture by this name
    certifying(rubin_kda_prefill)
    args = prefill_args(cell.spec, _seed(cell))
    seq_order, cu_chunks = step_schedule(cell.spec["seq_lens"], "cuda")
    args.update(seq_order=seq_order, cu_chunks=cu_chunks)
    ref_args = clone_written(args, ("state_pool",))
    out = rubin_kda_prefill(**args)
    ref = rubin_kda_prefill.reference(**ref_args)
    torch.cuda.synchronize()

    assert out.dtype == torch.bfloat16 and out.shape == args["q"].shape
    assert_scaled_close("output", out, ref, TOL_OUT)
    pool, ref_pool = args["state_pool"], ref_args["state_pool"]
    written = set(cell.spec["slots"]) | {r for r in cell.spec.get("snapshot_rows", ()) if r >= 0}
    for row in range(pool.shape[0]):
        if row in written:
            assert_scaled_close(f"state row {row}", pool[row], ref_pool[row], TOL_STATE[pool.dtype])
        else:
            assert torch.equal(pool[row], ref_pool[row]), f"unselected row {row} changed"


@torch.inference_mode()
def test_snapshot_equals_the_state_of_a_shorter_prefill() -> None:
    """A snapshot after 64 * c tokens holds what a prefill of only those tokens leaves."""
    spec = dict(
        heads=12,
        seq_lens=(200,),
        slots=(0,),
        fresh=(True,),
        state_dtype="float32",
        snapshot_rows=(3,),
        snapshot_chunks=(2,),
    )
    args = prefill_args(spec, seed=7)
    with validating(rubin_kda_prefill):
        rubin_kda_prefill(**args)
    short = dict(args, snapshot_rows=None, snapshot_chunks=None)
    short.update(
        q=args["q"][:, :128],
        k=args["k"][:, :128],
        v=args["v"][:, :128],
        g=args["g"][:, :128],
        beta=args["beta"][:, :128],
        cu_seqlens=cu_seqlens_of((128,), "cuda"),
        state_indices=torch.tensor([1], dtype=torch.int32, device="cuda"),
    )
    args["state_pool"][1].zero_()
    rubin_kda_prefill(**short)
    torch.cuda.synchronize()
    # The two runs may tile the 128 tokens differently; 72 more tokens move the state far more.
    assert_scaled_close(
        "snapshot", args["state_pool"][3], args["state_pool"][1], TOL_STATE[torch.float32]
    )


@torch.inference_mode()
def test_state_row_beyond_int32_offset() -> None:
    """A state row more than 2**31 elements into the pool updates like the first row (the pool is
    a strided view of the KV-cache memory)."""
    heads, tokens = 8, 256
    row_elems = heads * HEAD_DIM * HEAD_DIM
    far_row = (2**31) // row_elems + 1
    free, _ = torch.cuda.mem_get_info()
    if free < (far_row + 1) * row_elems * 2 + (1 << 30):
        pytest.skip("needs a state pool beyond 4 GiB")
    gen = torch.Generator(device="cuda").manual_seed(0)
    shape = (1, tokens, heads, HEAD_DIM)

    def bf16(scale: float = 1.0) -> torch.Tensor:
        return (torch.randn(*shape, generator=gen, device="cuda") * scale).to(torch.bfloat16)

    inputs = dict(
        q=bf16(),
        k=bf16(),
        v=bf16(0.5),
        g=bf16(),
        beta=torch.randn(1, tokens, heads, generator=gen, device="cuda").to(torch.bfloat16),
        cu_seqlens=cu_seqlens_of((tokens,), "cuda"),
        A_log=torch.zeros(heads, device="cuda"),
        dt_bias=torch.zeros(heads * HEAD_DIM, device="cuda"),
        scale=SCALE,
        lower_bound=LOWER_BOUND,
    )
    seed = (torch.randn(heads, HEAD_DIM, HEAD_DIM, generator=gen, device="cuda") * 0.1).to(
        torch.bfloat16
    )
    pool = torch.zeros(far_row + 1, heads, HEAD_DIM, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    results = []
    for row in (0, far_row):
        pool[row].copy_(seed)
        output = rubin_kda_prefill(
            **inputs,
            state_pool=pool,
            state_indices=torch.tensor([row], dtype=torch.int32, device="cuda"),
        )
        results.append((output, pool[row].clone()))
    for got, want in zip(results[1], results[0], strict=True):
        assert torch.equal(got, want)


@torch.inference_mode()
@pytest.mark.parametrize("seq_lens", [(100, 200), (300,)], ids=["varlen", "single-sequence"])
def test_output_survives_the_next_call(seq_lens) -> None:
    """The output is new: a second call with the same shapes leaves the first call's output."""
    spec = dict(
        heads=2,
        seq_lens=seq_lens,
        slots=tuple(range(len(seq_lens))),
        fresh=(True,) * len(seq_lens),
        state_dtype="float32",
    )
    first = rubin_kda_prefill(**prefill_args(spec, seed=0))
    kept = first.clone()
    second = rubin_kda_prefill(**prefill_args(spec, seed=1))
    torch.cuda.synchronize()
    assert not torch.equal(second, kept), "the two calls must compute different outputs"
    assert torch.equal(first, kept), "a later call wrote an earlier call's output"


def test_guard_refuses_repeated_slots() -> None:
    spec = dict(heads=2, seq_lens=(4, 4), slots=(1, 1), fresh=(True, True), state_dtype="float32")
    with pytest.raises(AssertionError, match="unique slot"):
        with validating(rubin_kda_prefill):
            rubin_kda_prefill(**prefill_args(spec, seed=0))


def test_guard_refuses_a_snapshot_past_the_sequence() -> None:
    """The kernel skips such a snapshot without a word; the guard names it."""
    spec = dict(
        heads=2,
        seq_lens=(70,),
        slots=(0,),
        fresh=(True,),
        state_dtype="float32",
        snapshot_rows=(1,),
        snapshot_chunks=(2,),
    )
    with pytest.raises(AssertionError, match="inside its sequence"):
        with validating(rubin_kda_prefill):
            rubin_kda_prefill(**prefill_args(spec, seed=0))
