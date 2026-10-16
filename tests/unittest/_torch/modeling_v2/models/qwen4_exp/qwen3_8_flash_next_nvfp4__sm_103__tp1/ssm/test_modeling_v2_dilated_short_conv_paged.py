# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the dilated_short_conv_paged entry.

The one engine op used here (`test_dilation_one_agrees_with_the_engine`) is an
independent oracle, not a reference: at dilation 1 this operator is exactly
`torch.ops.trtllm.causal_conv1d_fwd`, written from the other end.
"""

import zlib

import pytest
import torch
import torch.nn.functional as F

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
    """`__call__`'s positional arguments for one spec, freshly allocated.

    The checkpoint's taps are small (its four positions average 1e-4 .. 0.0185),
    so the weight is scaled down rather than left at unit variance.
    """
    g = torch.Generator(device="cuda").manual_seed(seed)
    channels, lengths = spec["channels"], spec["lengths"]
    kernel_size, dilation = spec.get("kernel_size", KERNEL_SIZE), spec.get("dilation", DILATION)
    dtype = spec.get("dtype", torch.bfloat16)
    num_seqs, state_len = len(lengths), (kernel_size - 1) * dilation

    def randn(*shape, scale=1.0):
        return (scale * torch.randn(*shape, generator=g, device="cuda")).to(dtype)

    starts = [0]
    for length in lengths:
        starts.append(starts[-1] + length)
    qsl = torch.tensor(starts, dtype=spec.get("qsl_dtype", torch.int32), device="cuda")
    if spec.get("x_form", "channel_major") == "token_major":
        x = randn(starts[-1], channels).t()
    else:
        x = randn(channels, starts[-1])
    weight = randn(channels, 1, kernel_size, scale=0.3)
    bias = randn(channels, scale=0.1) if spec.get("bias", False) else None
    pool = randn(spec.get("pool_slots", 8), channels, state_len)
    if spec.get("no_indices", False):
        cache_indices = None
    else:
        slots = spec.get("slots") or tuple(range(num_seqs))
        index_dtype = spec.get("index_dtype", torch.int32)
        cache_indices = torch.tensor(slots, dtype=index_dtype, device="cuda")
    initial = spec.get("initial", (True,) * num_seqs)
    his = None if initial == "none" else torch.tensor(initial, dtype=torch.bool, device="cuda")
    silu, pad = spec.get("silu", True), spec.get("pad_slot_id", PAD_SLOT_ID)
    return [x, weight, bias, pool, qsl, cache_indices, his, dilation, silu, pad]


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


def test_published_state_is_the_raw_input_tail() -> None:
    """Independent of the reference, on both sides of the state length."""
    args = _small((STATE_LEN + 5, 2), (3, 1), seed=7)
    _, pool = _run(args)
    x, before = args[0], args[3]
    end0, end1 = STATE_LEN + 5, STATE_LEN + 7
    assert torch.equal(pool[3], x[:, end0 - STATE_LEN : end0])
    assert torch.equal(pool[1], torch.cat([before[1][:, 2:], x[:, end0:end1]], dim=1))


def test_impulse_response_pins_the_tap_map() -> None:
    """Tap `k` lights output `t` exactly when the impulse sits at window column
    `t + k * dilation` -- crossing the pool boundary, where a paged
    implementation goes wrong."""
    channels, tokens = 8, 6
    qsl = torch.tensor([0, tokens], dtype=torch.int32, device="cuda")
    cidx = torch.tensor([1], dtype=torch.int32, device="cuda")
    his = torch.tensor([True], dtype=torch.bool, device="cuda")
    for k in range(KERNEL_SIZE):
        weight = torch.zeros(channels, 1, KERNEL_SIZE, device="cuda")
        weight[:, 0, k] = 1.0
        for column in range(STATE_LEN + tokens):
            pool = torch.zeros(2, channels, STATE_LEN, device="cuda")
            x = torch.zeros(channels, tokens, device="cuda")
            if column < STATE_LEN:
                pool[1, :, column] = 1.0
            else:
                x[:, column - STATE_LEN] = 1.0
            out = op(x, weight, None, pool, qsl, cidx, his, DILATION, False, PAD_SLOT_ID)
            expected = torch.zeros(channels, tokens, device="cuda")
            t = column - k * DILATION
            if 0 <= t < tokens:
                expected[:, t] = 1.0
            assert torch.equal(out, expected), (k, column)


def test_the_last_tap_is_the_current_token() -> None:
    weight = torch.zeros(16, 1, KERNEL_SIZE, device="cuda")
    weight[:, 0, KERNEL_SIZE - 1] = 1.0
    g = torch.Generator(device="cuda").manual_seed(10)
    x = torch.randn(16, 8, generator=g, device="cuda")
    pool = torch.randn(4, 16, STATE_LEN, generator=g, device="cuda")
    qsl = torch.tensor([0, 5, 8], dtype=torch.int32, device="cuda")
    cidx = torch.tensor([3, 0], dtype=torch.int32, device="cuda")
    his = torch.tensor([True, True], dtype=torch.bool, device="cuda")
    out = op(x, weight, None, pool, qsl, cidx, his, DILATION, False, PAD_SLOT_ID)
    torch.testing.assert_close(out, x)


def test_the_carry_is_actually_read() -> None:
    """Dropping the carry moves the first `state_len` outputs of each sequence
    and nothing else -- invisible to any tolerance on a prefill batch."""
    args = _small((40, 40), (1, 2), seed=11)
    with_carry, _ = _run(args)
    without, _ = _run(args[:6] + [torch.zeros_like(args[6])] + args[7:])
    for begin, end in ((0, 40), (40, 80)):
        head = with_carry[:, begin : begin + STATE_LEN] - without[:, begin : begin + STATE_LEN]
        tail = with_carry[:, begin + STATE_LEN : end] - without[:, begin + STATE_LEN : end]
        assert head.abs().max().item() > 0.05
        assert torch.equal(tail, torch.zeros_like(tail))


def test_discrimination_against_the_wrong_variants() -> None:
    """Five plausible mis-implementations, each `reference` on altered inputs.

    Measured max |deviation| from the entry (fp32, activation off, these
    seeds): correct under 1e-5; taps reversed 4.89; dilation 1 over the same
    taps 4.50; carry reversed 4.38; carry dropped 2.80; the batch as one
    sequence 4.36 -- the gate at 1.0 sits five orders above the correct
    result. The last two touch only nine columns per sequence.
    """
    args = _small((11, 13, 7), (2, 0, 1), seed=12, dtype=torch.float32)
    args[8] = False
    x, weight, _, pool, qsl, cidx, his = args[:7]
    out, _ = _run(args)
    correct, _ = op.reference(*args)
    assert (out - correct).abs().max().item() < 1e-5

    def variant(**changes):
        names = (
            "x",
            "weight",
            "bias",
            "conv_states",
            "query_start_loc",
            "cache_indices",
            "has_initial_state",
            "dilation",
            "silu_activation",
            "pad_slot_id",
        )
        kwargs = dict(zip(names, args))
        kwargs.update(changes)
        return op.reference(**kwargs)[0]

    wrong = {
        "flipped_taps": variant(weight=weight.flip(-1)),
        "dilation_1": variant(
            conv_states=pool[:, :, -(KERNEL_SIZE - 1) :].contiguous(), dilation=1
        ),
        "reversed_state": variant(conv_states=pool.flip(-1)),
        "no_carry": variant(has_initial_state=torch.zeros_like(his)),
        "one_sequence": variant(
            query_start_loc=torch.tensor([0, x.shape[1]], dtype=torch.int32, device="cuda"),
            cache_indices=cidx[:1],
            has_initial_state=his[:1],
        ),
    }
    for name, value in wrong.items():
        deviation = (out - value).abs().max().item()
        print(f"discrimination {name}: {deviation:.6g}")
        assert deviation > 1.0, name


def test_chunked_prefill_matches_one_shot() -> None:
    """The pool is the whole contract between steps: a chunk plus a remainder
    is bitwise the one-shot call, including a remainder shorter than the state."""
    tokens = 64
    args = _small((tokens,), (2,), seed=13, bias=True)
    x, weight, bias, pool, _, cidx, his = args[:7]
    whole = pool.clone()
    one_shot = op(x, weight, bias, whole, args[4], cidx, his, DILATION, True, PAD_SLOT_ID)
    for split in (1, 2, STATE_LEN, tokens - 1):
        chunked = pool.clone()
        first = op(
            x[:, :split],
            weight,
            bias,
            chunked,
            torch.tensor([0, split], dtype=torch.int32, device="cuda"),
            cidx,
            his,
            DILATION,
            True,
            PAD_SLOT_ID,
        )
        second = op(
            x[:, split:].contiguous(),
            weight,
            bias,
            chunked,
            torch.tensor([0, tokens - split], dtype=torch.int32, device="cuda"),
            cidx,
            torch.ones_like(his),
            DILATION,
            True,
            PAD_SLOT_ID,
        )
        assert torch.equal(torch.cat([first, second], dim=1), one_shot), split
        assert torch.equal(chunked, whole), split


def test_token_major_view_is_bitwise_the_contiguous_call() -> None:
    spec = dict(
        channels=256, lengths=(9, 1, 20), slots=(1, 3, 0), pool_slots=4, initial=(True, False, True)
    )
    view_out, view_pool = _run(_build(dict(spec, x_form="token_major"), seed=14))
    args = _build(dict(spec, x_form="token_major"), seed=14)
    args[0] = args[0].contiguous()
    packed_out, packed_pool = _run(args)
    assert torch.equal(view_out, packed_out) and torch.equal(view_pool, packed_pool)


def test_dilation_one_agrees_with_the_engine() -> None:
    """An independent oracle on the ragged bookkeeping: fp32 activations within
    the default band (2.4e-7), the published pool bitwise."""
    import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*

    args = _small((23, 1, 64, 1), (3, 1, 0, 2), seed=15, dilation=1, dtype=torch.float32)
    x, weight, _, pool, qsl, cidx, his = args[:7]
    engine_x, engine_pool = x.clone(), pool.clone()
    torch.ops.trtllm.causal_conv1d_fwd(
        engine_x,
        weight.squeeze(1).contiguous(),
        None,
        engine_pool,
        qsl,
        cidx,
        his,
        True,
        PAD_SLOT_ID,
        None,
    )
    mine, mine_pool = _run(args[:7] + [1, True, PAD_SLOT_ID])
    torch.testing.assert_close(mine, engine_x)
    assert torch.equal(mine_pool, engine_pool)


def test_silu_off_is_the_bare_convolution() -> None:
    args = _small((12, 3), (1, 0), seed=16, bias=True)
    raw, _ = _run(args[:8] + [False, PAD_SLOT_ID])
    with_act, _ = _run(args)
    assert torch.equal(with_act, F.silu(raw.float()).to(raw.dtype))


def test_rounding_is_two_roundings() -> None:
    """The conv rounds on its store and the SiLU rounds again; a single fused
    rounding disagrees on a measurable fraction (0.2496 at these values)."""
    args = _small((64, 64), (0, 1), seed=23)
    out, _ = _run(args)
    two, _ = op.reference(*args)
    fp32 = [a.float() if i in (0, 1, 3) else a for i, a in enumerate(args)]
    fp32[8] = False
    acc, _ = op.reference(*fp32)
    one = F.silu(acc).to(out.dtype)
    assert torch.equal(out, two)
    mismatches = (one != two).float().mean().item()
    print(f"single-rounding disagrees on {mismatches:.4f} of the elements")
    assert mismatches > 0.0


def test_repeat_calls_are_bitwise_stable() -> None:
    args = _small((33, 1), (2, 3), seed=24)
    first = _run(args)
    for _ in range(3):
        again = _run(args)
        assert torch.equal(first[0], again[0]) and torch.equal(first[1], again[1])


def test_cuda_graph_capture_and_replay() -> None:
    """Captured once, replayed after the batch tensors' values change in place."""
    args = _small((4, 4, 4, 4), (0, 1, 2, 3), seed=25, pool_slots=6)
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
    qsl.copy_(torch.tensor([0, 1, 2, 3, 16], dtype=torch.int32, device="cuda"))
    cidx.copy_(torch.tensor([4, 3, 2, 1], dtype=torch.int32, device="cuda"))
    his.copy_(torch.tensor([True, False, True, False], dtype=torch.bool, device="cuda"))
    x.copy_(torch.randn(x.shape, generator=g, device="cuda").to(x.dtype))
    live.copy_(pool)
    graph.replay()
    torch.cuda.synchronize()

    ref_out, ref_pool = op.reference(
        x, weight, None, pool, qsl, cidx, his, DILATION, True, PAD_SLOT_ID
    )
    torch.testing.assert_close(captured, ref_out)
    assert torch.equal(live, ref_pool)


@pytest.mark.parametrize(
    "case,fragment",
    [
        ("pool_dtype", "index_copy_(): self and source expected to have the same dtype"),
        ("weight_2d", "weight should have at least three dimensions"),
        ("weight_dtype", "expected scalar type BFloat16 but found Float"),
        ("bias_dtype", "Input type (c10::BFloat16) and bias type (float) should be the same"),
    ],
)
def test_mismatched_operands_raise(case: str, fragment: str) -> None:
    """Refused loudly by the torch calls the entry is composed from -- why it
    has no `is_valid`."""
    args = _small((4, 4), None, seed=27)
    if case == "pool_dtype":
        args[3] = args[3].float()
    elif case == "weight_2d":
        args[1] = args[1].squeeze(1).contiguous()
    elif case == "weight_dtype":
        args[1] = args[1].float()
    elif case == "bias_dtype":
        args[2] = torch.zeros(args[0].shape[0], device="cuda")
    with pytest.raises(RuntimeError) as excinfo:
        with validating(op):
            op(*args)
    assert fragment in str(excinfo.value), str(excinfo.value)
