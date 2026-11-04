# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the cute_dsl_indexer_q_side_fused catalog entry."""

import math
import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog.attention.cute_dsl_indexer_q_side_fused import (  # noqa: E501
    build,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.attention.cute_dsl_indexer_q_side_fused import (  # noqa: E501
    cute_dsl_indexer_q_side_fused as op,
)

__extra_import_path__ = [".."]
from _validating import certifying, validating  # noqa: E402,F401 — needs the path declared above

assert torch.cuda.is_available(), (
    "the CuTe DSL kernel runs on a GPU; this test cannot run without one"
)

#: The indexer geometry and its folded softmax scale (fp32 2**-6 exactly).
HEADS, HEAD_DIM, ROPE_DIM = 32, 128, 64
WEIGHT_SCALE = HEAD_DIM**-0.5 * HEADS**-0.5
#: e4m3 NaN: the poison both code buffers start from.
POISON = 0x7F
#: Mistral Large 4's `rope_parameters` (interleaved cells): YaRN with no magnitude
#: scaling, and LeRoPE's per-frequency exp(alpha); its max_seq_len bounds the table.
#: The FP8 checkpoint's factor is the default; a cell's `factor` key overrides it
#: (the NVFP4 checkpoint's is 2048, its max_seq_len 32768).
ML4_THETA, ML4_FACTOR, ML4_ORIGINAL_MAX, ML4_BETA_FAST, ML4_BETA_SLOW = 1.0e4, 512.0, 512, 32.0, 1.0
ML4_MAX_SEQ_LEN = 9216
NVFP4_FACTOR, NVFP4_MAX_SEQ_LEN = 2048.0, 32768
#: The NVFP4 checkpoint's own context: max_seq_len 1048576 in chunk steps of 32768 rows.
LONG_MAX_SEQ_LEN = 1048576

_COMPILED: dict = {}


def _compiled(is_neox: bool):
    if is_neox not in _COMPILED:
        _COMPILED[is_neox] = build(torch.cuda.current_device(), is_neox=is_neox)
    return _COMPILED[is_neox]


def _as_table(angles: torch.Tensor) -> torch.Tensor:
    """`[positions, 64]` fp32 `[cos | sin]`, built in fp64 and rounded once."""
    return torch.cat([angles.cos(), angles.sin()], dim=1).float().cuda().contiguous()


def _yarn_table(positions: int, gen: torch.Generator, factor: float = ML4_FACTOR) -> torch.Tensor:
    """Mistral Large 4's indexer table: YaRN frequencies, each scaled by a LeRoPE exp(alpha).

    The YaRN ramp runs between the correction dims of beta_fast and beta_slow
    and is taken on the unscaled frequencies (so it does not depend on the
    factor); alpha is a checkpoint tensor, so a seeded draw stands in for it.
    """
    half = ROPE_DIM // 2
    pos_freqs = ML4_THETA ** (torch.arange(0, ROPE_DIM, 2, dtype=torch.float64) / ROPE_DIM)
    lerope = torch.exp(0.1 * torch.randn(half, generator=gen, dtype=torch.float64))
    extrapolation, interpolation = lerope / pos_freqs, lerope / (factor * pos_freqs)

    def correction_dim(rotations: float) -> float:
        return (ROPE_DIM * math.log(ML4_ORIGINAL_MAX / (rotations * 2 * math.pi))) / (
            2 * math.log(ML4_THETA)
        )

    low = max(math.floor(correction_dim(ML4_BETA_FAST)), 0)
    high = min(math.ceil(correction_dim(ML4_BETA_SLOW)), ROPE_DIM - 1)
    ramp = ((torch.arange(half, dtype=torch.float64) - low) / (high - low)).clamp(0, 1)
    inv = interpolation * ramp + extrapolation * (1 - ramp)
    return _as_table(torch.arange(positions, dtype=torch.float64)[:, None] * inv[None, :])


def _identity_table(positions: int) -> torch.Tensor:
    """A NoPE layer's table: cos 1, sin 0 at every position."""
    return _as_table(torch.zeros(positions, ROPE_DIM // 2, dtype=torch.float64))


def _table(spec, seed) -> torch.Tensor:
    """The cell's `[max_pos, 64]` fp32 rope table; a yarn table's LeRoPE alpha is drawn from `seed`.

    `table_dtype="bfloat16"` rounds the table's cos and sin through bf16, as the
    targets hold theirs (in the activation dtype, widened to fp32 for this op).
    """
    max_pos = spec["max_pos"]
    if spec["table"] == "yarn":
        cpu = torch.Generator().manual_seed(seed)
        table = _yarn_table(max_pos, cpu, spec.get("factor", ML4_FACTOR))
    else:
        table = _identity_table(max_pos)
    if spec.get("table_dtype") == "bfloat16":
        table = table.to(torch.bfloat16).float()
    return table


def _build(spec, seed, table=None):
    """`[q, position_ids, cos_sin, wq, weight_scale, q_fp8_out, weights_out]` for one cell.

    Both outputs start poisoned, so a row the kernel skipped cannot pass.
    Mistral Large 4's head weights come out of a bf16 projection and are cast to
    fp32, so its cells draw bf16-representable `wq`. `cached` starts an `arange`
    step at that position, as a chunk after a request's first runs behind its
    cached prefix. `tail` draws each row from the table's last `tail_rows` rows
    (64 unless given). `first` puts row 0 at position `first` and every other
    row at 0: a decode batch whose other rows are one-key dummies (the boot
    warm-up's fillers, attention DP's padding rows). `table` passes a table
    built once for several calls.
    """
    g = torch.Generator(device="cuda").manual_seed(seed)
    tokens, max_pos = spec["tokens"], spec["max_pos"]
    q = (torch.randn(tokens, HEADS, HEAD_DIM, generator=g, device="cuda") * 2.0).to(torch.bfloat16)
    if spec["positions"] == "arange":
        cached = spec.get("cached", 0)
        assert cached == 0 or cached + tokens <= max_pos, (cached, tokens, max_pos)
        pos = (cached + torch.arange(tokens, device="cuda")) % max_pos
    elif spec["positions"] == "tail":
        rows = spec.get("tail_rows", 64)
        pos = max_pos - 1 - torch.randint(0, rows, (tokens,), generator=g, device="cuda")
    elif spec["positions"] == "first":
        pos = torch.zeros(tokens, dtype=torch.int64, device="cuda")
        pos[0] = spec["first"]
    else:
        pos = torch.randint(0, max_pos, (tokens,), generator=g, device="cuda")
    wq = torch.randn(tokens, HEADS, generator=g, device="cuda")
    wq = wq.to(torch.bfloat16).float()
    if table is None:
        table = _table(spec, seed)
    q_fp8_out = torch.full((tokens, HEADS, HEAD_DIM), POISON, dtype=torch.uint8, device="cuda")
    q_fp8_out = q_fp8_out.view(torch.float8_e4m3fn)
    weights_out = torch.full((tokens, HEADS), float("nan"), device="cuda")
    return [q, pos.to(torch.int32), table, wq, WEIGHT_SCALE, q_fp8_out, weights_out]


def _kw(spec) -> dict:
    return dict(is_neox=spec["is_neox"], compiled=_compiled(spec["is_neox"]))


def _compare(out, ref) -> None:
    """Bit-exact codes and weights.

    The reference reproduces every rounding the kernel makes: the rope FMA
    (fp64 rounded once to fp32), the bf16 store of the rotated half, the
    power-of-two scale from exponent bits, an exact multiply by its inverse,
    the e4m3 cast, and two fp32 products for the weights. Nothing is left to
    accumulation order, so the band is zero; a wrong rope mode or table moves
    most codes (the rope-mode guard and the identity-table tests measure it).
    """
    (q_fp8, weights), (ref_fp8, ref_weights) = out, ref
    codes, want = q_fp8.view(torch.uint8), ref_fp8.view(torch.uint8)
    bad = codes != want
    assert not bool(bad.any()), (
        f"{int(bad.sum())} of {bad.numel()} e4m3 codes differ; first at "
        f"{tuple(int(i) for i in bad.nonzero()[0])}"
    )
    wbad = weights.view(torch.int32) != ref_weights.view(torch.int32)
    assert not bool(wbad.any()), f"{int(wbad.sum())} of {wbad.numel()} weights differ"


def _rope_codes_moved(out, ref) -> float:
    """Fraction of the rope half's e4m3 codes that differ."""
    got, want = out[0].view(torch.uint8), ref[0].view(torch.uint8)
    return float((got[..., :ROPE_DIM] != want[..., :ROPE_DIM]).float().mean())


def _assert_replay_matches_eager(spec, seed) -> None:
    """Capture the call at this batch, replay it on new data, and require the eager bytes.

    A capture freezes the launcher, the grid and every pointer; what changes
    between replays is the data behind them. So the static inputs take a second
    set of queries, positions, table and head weights, the outputs go back to
    poison, and the replay must write bitwise what an eager call on that data
    writes into buffers of its own. A boot capture replays once per `_replays`
    entry, each on data of its own.
    """
    static = _build(spec, seed)
    kw = _kw(spec)
    op(*static, **kw)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream), torch.cuda.graph(graph, stream=stream):
        op(*static, **kw)
    torch.cuda.current_stream().wait_stream(stream)
    for k, replay in enumerate(_replays(spec)):
        fresh, eager = _build(replay, seed + 1 + k), _build(replay, seed + 1 + k)
        for i in (0, 1, 2, 3):
            static[i].copy_(fresh[i])
        static[5].view(torch.uint8).fill_(POISON)
        static[6].fill_(float("nan"))
        graph.replay()
        op(*eager, **kw)
        torch.cuda.synchronize()
        assert torch.equal(static[5].view(torch.uint8), eager[5].view(torch.uint8)), (
            f"replay {k}: the replay wrote other codes than the eager call"
        )
        assert torch.equal(static[6].view(torch.int32), eager[6].view(torch.int32)), (
            f"replay {k}: the replay wrote other weights than the eager call"
        )


def _replays(spec) -> list:
    """The data a captured cell replays on, in order.

    A boot capture (`positions="first"`: the warm-up batch, its longest dummy in
    row 0 beside one-key fillers) replays as serving runs the graph: attention
    DP's one live row at the table's last position behind one-key padding rows
    at 0, then a full batch at positions spread over the table. Any other
    captured cell replays once, on data drawn as its own.
    """
    if spec["positions"] == "first":
        return [dict(spec, first=spec["max_pos"] - 1), dict(spec, positions="spread")]
    return [spec]


def _seed(cell) -> int:
    return zlib.crc32(cell.why.encode()) % 2**31


def _position_sweep(spec, seed) -> str:
    """`position_sweep` (lo, hi): every table row lo .. hi - 1, `tokens` rows per call.

    Call k reads positions lo + k * tokens onward, on fresh queries and head
    weights over the one table, and is gated as a cell is: the caller's buffers
    written and returned, `q` untouched, codes and weights bitwise the
    reference. The table must come out of the sweep unwritten too. Returns a
    summary line.
    """
    lo, hi = spec["position_sweep"]
    rows = spec["tokens"]
    assert (hi - lo) % rows == 0 and hi <= spec["max_pos"], (lo, hi, rows)
    table, kw = _table(spec, seed), _kw(spec)
    table_before = table.clone()
    read = torch.zeros(spec["max_pos"], dtype=torch.bool, device="cuda")
    for k in range((hi - lo) // rows):
        args = _build(dict(spec, cached=lo + k * rows), seed + 1 + k, table=table)
        q_before = args[0].clone()
        out = op(*args, **kw)
        assert out[0] is args[5] and out[1] is args[6], "the op writes the caller's buffers"
        assert torch.equal(args[0], q_before), "q must not be written"
        _compare(out, op.reference(*args, **kw))
        read[args[1].long()] = True
    assert torch.equal(table, table_before), "the rope table must not be written"
    assert bool(read[lo:hi].all()), "a table row of the sweep was never read"
    return f"{(hi - lo) // rows} calls, every row {lo}..{hi - 1} read, codes and weights bitwise"


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811
    spec = cell.spec
    if spec.get("graph"):
        # Before arming: the guard reads the positions back to the host, which a
        # capture forbids. `graph` is True (capture at the cell's rows) or the
        # batches to capture, each on draws of its own.
        batches = (spec["tokens"],) if spec["graph"] is True else spec["graph"]
        for i, batch in enumerate(batches):
            _assert_replay_matches_eager(dict(spec, tokens=batch), _seed(cell) + 2 * i)
    certifying(op)
    if "position_sweep" in spec:
        print(f"\n  {cell.why}: {_position_sweep(spec, _seed(cell))}")
        return
    args, kw = _build(spec, seed=_seed(cell)), _kw(spec)
    q_before = args[0].clone()
    out = op(*args, **kw)
    assert out[0] is args[5] and out[1] is args[6], "the op writes the caller's buffers"
    assert torch.equal(args[0], q_before), "q must not be written"
    _compare(out, op.reference(*args, **kw))


def test_a_fresh_launch_overwrites_every_row() -> None:
    """Two calls into the same buffers: the second result is the second input's."""
    spec = dict(
        tokens=5,
        positions="spread",
        max_pos=ML4_MAX_SEQ_LEN,
        is_neox=False,
        table="yarn",
    )
    first, second = _build(spec, seed=31), _build(spec, seed=32)
    kw = _kw(spec)
    out_fp8, out_w = first[5], first[6]
    op(*first, **kw)
    op(*second[:5], out_fp8, out_w, **kw)
    _compare((out_fp8, out_w), op.reference(*second, **kw))


#: Interleaved calls a non-cell test makes: a NoPE layer's, and a small RoPE layer's.
_IDENTITY = dict(is_neox=False, table="identity", max_pos=ML4_MAX_SEQ_LEN, positions="spread")
_SMALL = dict(is_neox=False, table="yarn", max_pos=ML4_MAX_SEQ_LEN, positions="spread", tokens=8)


def test_an_identity_table_is_no_rope() -> None:
    """A NoPE layer's call is a plain per-(token, head) ue8m0 quantization of `q`.

    With bf16 subnormals in both halves and a -0 in the rope half, the call is
    first bit-exact against the entry's reference, then checked against a
    quantization written here from `q` alone: weights bitwise, codes value for
    value. The one bitwise difference is the sign of a zero code in the rope
    half: the kernel flushes a rope-half subnormal to a signed zero and the
    rotation then adds the partner's `x * 0`, so a negative subnormal or -0
    whose partner product is +0 comes out +0 (code 0x00 where plain
    quantization writes 0x80). Both decode to zero. And the same call against
    the roped reference moves most rope codes, so the NoPE cells would see a
    table that rotates.
    """
    spec = dict(_IDENTITY, tokens=256)
    args, kw = _build(spec, seed=35), _kw(spec)
    q = args[0]
    tiny = torch.tensor([2.0**-130, -(2.0**-130), 2.0**-127, -(2.0**-133)], dtype=torch.bfloat16)
    q[3, 5, :4] = tiny.cuda()
    q[3, 5, 70:74] = tiny.cuda()
    q[7, 0, :] = tiny.cuda().repeat(HEAD_DIM // 4)
    q[3, 5, 8:10] = torch.tensor([1.0, -0.0], dtype=torch.bfloat16).cuda()
    out = op(*args, **kw)
    _compare(out, op.reference(*args, **kw))

    row = q.float()
    amax = row.abs().amax(dim=-1)
    raw = torch.maximum(amax, torch.tensor(1.0e-12, device="cuda")) * torch.tensor(
        1.0 / 448.0, dtype=torch.float32, device="cuda"
    )
    # Round up to a power of two through frexp (raw = m * 2**e, 0.5 <= m < 1):
    # exact, where CUDA's fp64 log2 of an exact power of two is not.
    mantissa, exponent = torch.frexp(raw)
    scale = torch.ldexp(torch.ones_like(raw), torch.where(mantissa == 0.5, exponent - 1, exponent))
    codes = (row / scale[..., None]).to(torch.float8_e4m3fn)
    weights = (args[3] * scale) * torch.tensor(WEIGHT_SCALE, dtype=torch.float32, device="cuda")
    assert torch.equal(out[1].view(torch.int32), weights.view(torch.int32)), "weights differ"
    assert torch.equal(out[0].float(), codes.float()), "a code decodes to another value"
    got, want = out[0].view(torch.uint8), codes.view(torch.uint8)
    bits = q.view(torch.int16)
    negative_zero_or_subnormal = ((bits & 0x7F80) == 0) & (bits < 0)
    off = got != want
    assert not bool(off[..., ROPE_DIM:].any()), "a nope code differs"
    assert not bool((off[..., :ROPE_DIM] & ~negative_zero_or_subnormal[..., :ROPE_DIM]).any()), (
        "a rope code differs away from a negative zero or subnormal"
    )
    assert int(got[3, 5, 9]) == 0x00 and int(want[3, 5, 9]) == 0x80, "-0 beside a +1 partner"
    assert bool((got[off] == 0x00).all() and (want[off] == 0x80).all()), (
        "a differing code is not a zero of the other sign"
    )

    roped = list(args)
    roped[2] = _yarn_table(ML4_MAX_SEQ_LEN, torch.Generator().manual_seed(35))
    moved = _rope_codes_moved(out, op.reference(*roped, **kw))
    assert moved > 0.5, f"only {moved:.2f} of the rope codes moved under a rotating table"


def test_a_32768_step_sees_a_wrong_table_row_past_9215() -> None:
    """The NVFP4 32768-row cells pass only if every row reads its own table row.

    A read wrapped at the FP8 target's 9216 rows, or one position early, is the
    slip a kernel or a table sized for the old envelope would make. On the
    factor 2048 table each moves at least one rope code in every row from 9216
    up (about 0.6 and 0.3 of them), so the cells' bitwise gate refuses it.
    """
    spec = dict(
        is_neox=False,
        table="yarn",
        factor=NVFP4_FACTOR,
        max_pos=NVFP4_MAX_SEQ_LEN,
        positions="arange",
        tokens=NVFP4_MAX_SEQ_LEN,
    )
    args, kw = _build(spec, seed=37), _kw(spec)
    out = op(*args, **kw)
    pos = args[1]
    for name, wrong in (
        ("wrapped at 9216", pos % ML4_MAX_SEQ_LEN),
        ("one position early", (pos - 1).clamp(min=0)),
    ):
        wrong_args = list(args)
        wrong_args[1] = wrong
        ref = op.reference(*wrong_args, **kw)
        got, want = out[0].view(torch.uint8), ref[0].view(torch.uint8)
        moved = (got[..., :ROPE_DIM] != want[..., :ROPE_DIM]).flatten(1).any(dim=1)
        unseen = int((~moved[ML4_MAX_SEQ_LEN:]).sum())
        assert unseen == 0, f"{name}: {unseen} rows from 9216 up would pass the bitwise gate"


def test_a_1m_chunk_sees_a_wrong_table_row() -> None:
    """The NVFP4 1M chunk cells pass only if every row reads its own row of the 1048576 rows.

    A read wrapped at the delivered target's 32768 rows, or one position early,
    is the slip a table or a cached-prefix offset sized for the old envelope
    would make. At the last chunk of a 1M prompt (positions 1015808..1048575),
    on the factor 2048 table rounded through bf16 as the target holds it, each
    moves at least one rope code in every row (about 0.9 and 0.3 of them), so
    the cells' bitwise gate refuses it.
    """
    spec = dict(
        is_neox=False,
        table="yarn",
        factor=NVFP4_FACTOR,
        max_pos=LONG_MAX_SEQ_LEN,
        positions="arange",
        tokens=NVFP4_MAX_SEQ_LEN,
        cached=LONG_MAX_SEQ_LEN - NVFP4_MAX_SEQ_LEN,
        table_dtype="bfloat16",
    )
    args, kw = _build(spec, seed=38), _kw(spec)
    out = op(*args, **kw)
    _compare(out, op.reference(*args, **kw))
    pos = args[1]
    for name, wrong in (
        ("wrapped at 32768", pos % NVFP4_MAX_SEQ_LEN),
        ("one position early", pos - 1),
    ):
        wrong_args = list(args)
        wrong_args[1] = wrong
        ref = op.reference(*wrong_args, **kw)
        got, want = out[0].view(torch.uint8), ref[0].view(torch.uint8)
        moved = got[..., :ROPE_DIM] != want[..., :ROPE_DIM]
        unseen = int((~moved.flatten(1).any(dim=1)).sum())
        assert unseen == 0, f"{name}: {unseen} of the chunk's rows would pass the bitwise gate"
        print(f"\n  {name}: {float(moved.float().mean()):.3f} of the rope codes move")


def test_guard_refuses_a_rope_mode_the_launcher_was_not_built_for() -> None:
    args = _build(_SMALL, seed=33)
    neox = _compiled(True)
    # Unguarded, the NeoX launcher rotates interleaved queries the NeoX way:
    # well-formed rows whose rope codes are mostly wrong.
    out = op(*args, is_neox=False, compiled=neox)
    moved = _rope_codes_moved(out, op.reference(*args, is_neox=False, compiled=neox))
    assert moved > 0.5, f"only {moved:.2f} of the rope codes moved"
    with pytest.raises(AssertionError, match="compiled for is_neox"):
        with validating(op):
            op(*_build(_SMALL, seed=33), is_neox=False, compiled=neox)


def test_guard_refuses_a_position_past_the_rope_table() -> None:
    # Unguarded, the kernel reads past the table: not driven here.
    args = _build(_SMALL, seed=34)
    args[1][2] = args[2].shape[0]
    with pytest.raises(AssertionError, match="rope table"):
        with validating(op):
            op(*args, **_kw(_SMALL))


def _strided_head_weights(args):
    """Head weights as a `[T, 160]` wk|weights_proj output leaves them: its last 32 columns."""
    kw = torch.zeros(args[3].shape[0], HEAD_DIM + HEADS, device="cuda")
    kw[:, HEAD_DIM:] = args[3]
    args[3] = kw[:, HEAD_DIM:]


def _cast(index: int, dtype: torch.dtype):
    return lambda a: a.__setitem__(index, a[index].to(dtype))


@pytest.mark.parametrize(
    "mutate, error, message",
    [
        (_cast(3, torch.bfloat16), TypeError, r"wq must be torch\.float32, got torch\.bfloat16"),
        (_strided_head_weights, ValueError, r"wq must be contiguous, got stride \(160, 1\)"),
        (_cast(2, torch.bfloat16), TypeError, r"cos_sin must be torch\.float32"),
        (_cast(1, torch.int64), TypeError, r"position_ids must be torch\.int32"),
    ],
    ids=["bf16 head weights", "strided head weights", "bf16 table", "int64 positions"],
)
def test_the_launcher_refuses_what_it_was_not_compiled_for(mutate, error, message) -> None:
    """The note's refusals, pinned: raised before any launch, so both outputs stay poisoned."""
    args = _build(_SMALL, seed=36)
    mutate(args)
    with pytest.raises(error, match=message):
        op(*args, **_kw(_SMALL))
    torch.cuda.synchronize()
    assert bool((args[5].view(torch.uint8) == POISON).all()), "a refused call wrote codes"
    assert bool(args[6].isnan().all()), "a refused call wrote weights"
