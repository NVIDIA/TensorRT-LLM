# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the dilated_short_conv_paged entry."""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_103__tp1.ops.ssm.dilated_short_conv_paged import (  # noqa: E501
    ENGINE_SLOTS,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_103__tp1.ops.ssm.dilated_short_conv_paged import (  # noqa: E501
    dilated_short_conv_paged as op,
)

__extra_import_path__ = ["../../../.."]
from _validating import certifying, validating  # noqa: E402,F401 — path above; fixture by name

assert torch.cuda.is_available(), "dilated_short_conv_paged requires a CUDA device"


def _compare(
    out: tuple[torch.Tensor, torch.Tensor], ref: tuple[torch.Tensor, torch.Tensor]
) -> None:
    """`(out, conv_states)` pairs: the returned activation with the pool as
    the call left it, against `reference`'s pair.

    Activation at torch's default band: bf16 and fp16 were measured bitwise
    equal to the reference (at 10240 channels and 461 slots too); fp32
    differs only by tap order. The pool is a copy, gated bitwise over the
    whole pool -- the lock on every slot the batch did not name.
    """
    (act, pool), (ref_act, ref_pool) = out, ref
    assert act.dtype == ref_act.dtype and act.shape == ref_act.shape
    assert act.is_contiguous()
    torch.testing.assert_close(act, ref_act)
    assert torch.equal(pool, ref_pool), "the published state must be bitwise exact"


KERNEL_SIZE = 4
DILATION = 3
STATE_LEN = (KERNEL_SIZE - 1) * DILATION
PAD_SLOT_ID = -1


def _build(spec, seed):
    """`__call__`'s positional arguments for one spec, freshly allocated, in the
    call site's form: `x` a bare `.t()` of a token-major activation, int64
    `query_start_loc`, int32 slots.

    The checkpoint's taps are small (its four positions average 1e-4 .. 0.0185),
    so the weight is scaled down rather than left at unit variance.
    """
    g = torch.Generator(device="cuda").manual_seed(seed)
    channels, lengths = spec["channels"], spec["lengths"]
    num_seqs = len(lengths)

    def randn(*shape, scale=1.0):
        return (scale * torch.randn(*shape, generator=g, device="cuda")).to(torch.bfloat16)

    starts = [0]
    for length in lengths:
        starts.append(starts[-1] + length)
    qsl = torch.tensor(starts, dtype=torch.int64, device="cuda")
    x = randn(starts[-1], channels).t()
    weight = randn(channels, 1, KERNEL_SIZE, scale=0.3)
    pool = randn(spec.get("pool_slots", 8), channels, STATE_LEN)
    slots = spec.get("slots") or tuple(range(num_seqs))
    cache_indices = torch.tensor(slots, dtype=torch.int32, device="cuda")
    initial = spec.get("initial", (True,) * num_seqs)
    his = torch.tensor(initial, dtype=torch.bool, device="cuda")
    return [x, weight, None, pool, qsl, cache_indices, his, DILATION, True, PAD_SLOT_ID]


def _run(args):
    """Drive the op on a copy of the pool; return (out, pool after)."""
    got = list(args)
    got[3] = args[3].clone()
    return op(*got), got[3]


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — pytest resolves the fixture by this name
    certifying(op)
    args = _build(cell.spec, seed=zlib.crc32(cell.why.encode()))
    x, weight = args[0].clone(), args[1].clone()
    ref = op.reference(*args)
    out, pool = _run(args)
    assert out.data_ptr() != args[0].data_ptr()
    assert torch.equal(args[0], x) and torch.equal(args[1], weight), "x and weight are read-only"
    _compare((out, pool), ref)
    if cell.spec.get("pool_slots") == ENGINE_SLOTS:
        for slot in cell.spec["slots"]:  # non-vacuity of the whole-pool gate
            assert not torch.equal(pool[slot], args[3][slot]), slot


def _small(lengths, slots, seed, **spec):
    return _build(dict(channels=256, lengths=lengths, slots=slots, **spec), seed=seed)


def test_cuda_graph_capture_and_replay() -> None:
    """Captured once at the MTP-7 verify's 32 requests of 8 tokens, replayed after
    the batch tensors' values change in place."""
    batch, p = 32, 8
    args = _small((p,) * batch, tuple(range(batch)), seed=25, pool_slots=batch + 2)
    x, weight, _, pool, qsl, cidx, his = args[:7]
    live = pool.clone()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            op(x, weight, None, live, qsl, cidx, his, DILATION, True, PAD_SLOT_ID)
    torch.cuda.current_stream().wait_stream(stream)

    graph = torch.cuda.CUDAGraph()
    live.copy_(pool)
    with torch.cuda.graph(graph):
        captured = op(x, weight, None, live, qsl, cidx, his, DILATION, True, PAD_SLOT_ID)

    g = torch.Generator(device="cuda").manual_seed(26)
    # Every row but the last shrinks to one token; the slot map reverses and shifts by one.
    qsl.copy_(torch.tensor([*range(batch), batch * p], dtype=torch.int32, device="cuda"))
    cidx.copy_(torch.tensor(list(range(batch, 0, -1)), dtype=torch.int32, device="cuda"))
    his.copy_(torch.tensor([i % 2 == 0 for i in range(batch)], dtype=torch.bool, device="cuda"))
    x.copy_(torch.randn(x.shape, generator=g, device="cuda").to(x.dtype))
    live.copy_(pool)
    graph.replay()
    torch.cuda.synchronize()

    ref_out, ref_pool = op.reference(
        x, weight, None, pool, qsl, cidx, his, DILATION, True, PAD_SLOT_ID
    )
    torch.testing.assert_close(captured, ref_out)
    assert torch.equal(live, ref_pool)


def test_mismatched_operands_raise() -> None:
    """Refused loudly by the torch calls the entry is composed from -- why it
    has no `is_valid`. Probed at a pool of another dtype (the index copy)."""
    args = _small((4, 4), None, seed=27)
    args[3] = args[3].float()
    with pytest.raises(RuntimeError) as excinfo:
        with validating(op):
            op(*args)
    fragment = "index_copy_(): self and source expected to have the same dtype"
    assert fragment in str(excinfo.value), str(excinfo.value)
