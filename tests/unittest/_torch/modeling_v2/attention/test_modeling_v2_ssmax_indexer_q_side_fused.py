# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the ssmax_indexer_q_side_fused catalog entry.

Every cell is gated bit for bit, codes and weights, against the entry's `reference` (the
composed references of `ssmax_scaling` and `cute_dsl_indexer_q_side_fused`, each bit-exact
for its own op), and against the three calls it replaces run on the same inputs:
`ssmax_scaling`, the head weights' fp32 copy and `cute_dsl_indexer_q_side_fused`. Nothing is
left to accumulation order, so the band is zero.

The old-versus-new tests state that equivalence on its own, with `torch.equal` on the codes' and
weights' bit patterns: over every cell's random inputs, and over tensors captured from a served
decode step when TRTLLM_MODELING_V2_FUSE_CAPTURE names their directory (`ix_*.pt`, one file per
call: `q`, `positions`, `s`, `b`, the head weights `wq`, `weight_scale` and the name of the rope
table, saved beside them as `table_<name>.pt`), and at the 1M config over the NVFP4 checkpoint's
own indexer layers: each one's s and b and its table (the 1048576-row table the targets build from
the checkpoint's rope parameters and `lerope.alpha`, the identity table on a NoPE indexer layer)
at every position 0..1048575 (skipped where the checkpoint is absent from LLM_MODELS_ROOT).
"""

import json
import math
import os
import zlib
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from test_common.llm_data import llm_models_root

from tensorrt_llm._torch._experimental.modeling_v2.catalog.attention.cute_dsl_indexer_q_side_fused import (  # noqa: E501
    build,
    cute_dsl_indexer_q_side_fused,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.attention.ssmax_indexer_q_side_fused import (  # noqa: E501
    ssmax_indexer_q_side_fused as op,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.attention.ssmax_scaling import (
    ssmax_scaling,
)

__extra_import_path__ = [".."]
from _validating import certifying, validating  # noqa: E402,F401

assert torch.cuda.is_available(), "ssmax_indexer_q_side_fused requires a CUDA device"

HEADS, HEAD_DIM, ROPE_DIM = 32, 128, 64
WEIGHT_SCALE = HEAD_DIM**-0.5 * HEADS**-0.5
#: e4m3 NaN: the poison the code buffer starts from.
POISON = 0x7F
#: The NVFP4 checkpoint's `rope_parameters`: YaRN factor 2048 over an original 512 positions,
#: theta 1e4, beta_fast 32, beta_slow 1, and LeRoPE's per-frequency exp(alpha).
THETA, FACTOR, ORIGINAL_MAX, BETA_FAST, BETA_SLOW = 1.0e4, 2048.0, 512, 32.0, 1.0

_COMPILED = {}


def _compiled():
    if "q_side" not in _COMPILED:
        _COMPILED["q_side"] = build(torch.cuda.current_device(), is_neox=False)
    return _COMPILED["q_side"]


def _yarn_table(positions: int, gen: torch.Generator) -> torch.Tensor:
    """The indexer table the targets build, `[positions, 64]` fp32 `[cos | sin]` rounded
    through bf16: YaRN frequencies, each scaled by a seeded LeRoPE exp(alpha)."""
    half = ROPE_DIM // 2
    pos_freqs = THETA ** (torch.arange(0, ROPE_DIM, 2, dtype=torch.float64) / ROPE_DIM)
    lerope = torch.exp(0.3 * torch.randn(half, generator=gen, dtype=torch.float64))
    extrapolation, interpolation = lerope / pos_freqs, lerope / (FACTOR * pos_freqs)

    def correction_dim(rotations: float) -> float:
        return (ROPE_DIM * math.log(ORIGINAL_MAX / (rotations * 2 * math.pi))) / (
            2 * math.log(THETA)
        )

    low = max(math.floor(correction_dim(BETA_FAST)), 0)
    high = min(math.ceil(correction_dim(BETA_SLOW)), ROPE_DIM - 1)
    ramp = ((torch.arange(half, dtype=torch.float64) - low) / (high - low)).clamp(0, 1)
    inv = interpolation * ramp + extrapolation * (1 - ramp)
    angles = torch.arange(positions, dtype=torch.float64)[:, None] * inv[None, :]
    table = torch.cat([angles.cos(), angles.sin()], dim=1).float()
    return table.to(torch.bfloat16).float().cuda().contiguous()


def _identity_table(positions: int) -> torch.Tensor:
    """A NoPE indexer layer's table: cos 1, sin 0 at every position."""
    return torch.cat([torch.ones(positions, 32), torch.zeros(positions, 32)], 1).cuda()


def _queries(shape, kind: str, g: torch.Generator) -> torch.Tensor:
    if kind == "normal":
        return (2 * torch.randn(shape, device="cuda", generator=g)).to(torch.bfloat16)
    exponent = torch.randint(-133, 127, shape, device="cuda", generator=g).float()
    mantissa = 1 + torch.rand(shape, device="cuda", generator=g)
    sign = 2 * torch.randint(0, 2, shape, device="cuda", generator=g).float() - 1
    return (sign * mantissa * torch.exp2(exponent)).to(torch.bfloat16)


def _build(spec, seed, table=None):
    """The call's nine arguments, both outputs poisoned. The head weights are the last 32
    columns of a bf16 `[T, 160]` buffer (`kw_bf16`, the targets' view) or a contiguous fp32
    copy (`fp32`); s and b hold the ranges' four corners on heads 0..3. `table`, when given,
    is the rope table instead of the spec's own."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    cpu = torch.Generator().manual_seed(seed)
    tokens, max_pos = spec["tokens"], spec["max_pos"]
    q = _queries((tokens, HEADS, HEAD_DIM), spec["q"], g)
    if spec["positions"] == "arange":
        cached = spec.get("cached", 0)
        assert cached + tokens <= max_pos or not cached, (cached, tokens, max_pos)
        pos = (cached + torch.arange(tokens, device="cuda")) % max_pos
    elif spec["positions"] == "tail":
        pos = max_pos - 1 - torch.randint(0, 64, (tokens,), generator=g, device="cuda")
    else:
        pos = torch.randint(0, max_pos, (tokens,), generator=g, device="cuda")
    s, b = (
        torch.empty(HEADS, device="cuda").uniform_(*spec[k], generator=g).bfloat16()
        for k in ("s_range", "b_range")
    )
    corners = [(sv, bv) for sv in spec["s_range"] for bv in spec["b_range"]]
    s[:4] = torch.tensor([c[0] for c in corners], device="cuda").bfloat16()
    b[:4] = torch.tensor([c[1] for c in corners], device="cuda").bfloat16()
    if table is None:
        table = _yarn_table(max_pos, cpu) if spec["table"] == "yarn" else _identity_table(max_pos)
    kw = torch.randn(tokens, HEAD_DIM + HEADS, generator=g, device="cuda").bfloat16()
    wq = kw[:, HEAD_DIM:] if spec["wq"] == "kw_bf16" else kw[:, HEAD_DIM:].float().contiguous()
    q_fp8 = torch.full((tokens, HEADS, HEAD_DIM), POISON, dtype=torch.uint8, device="cuda")
    weights = torch.full((tokens, HEADS), float("nan"), device="cuda")
    return [
        q, pos.to(torch.int32), s, b, table, wq, WEIGHT_SCALE,
        q_fp8.view(torch.float8_e4m3fn), weights,
    ]  # fmt: skip


def _calls(spec, seed):
    """The cell's calls' arguments: one call, or for a `position_sweep` (lo, hi) one call of
    `tokens` rows at each of lo, lo + tokens, ..., every call on fresh inputs over the cell's
    one table."""
    if "position_sweep" not in spec:
        yield _build(spec, seed)
        return
    lo, hi = spec["position_sweep"]
    rows = spec["tokens"]
    assert (hi - lo) % rows == 0 and hi <= spec["max_pos"], (lo, hi, rows)
    table = _build(dict(spec, tokens=1), seed)[4]
    for k in range((hi - lo) // rows):
        args = _build(spec, seed + 1 + k, table)
        args[1] = torch.arange(lo + k * rows, lo + (k + 1) * rows, dtype=torch.int32, device="cuda")
        yield args


def _replaced(args):
    """The three calls the entry replaces, into fresh buffers."""
    q, pos, s, b, table, wq, scale, q_fp8, weights = args
    out_fp8, out_w = torch.empty_like(q_fp8), torch.empty_like(weights)
    return cute_dsl_indexer_q_side_fused(
        ssmax_scaling(q, pos, s, b), pos, table, wq.float().contiguous(), scale, out_fp8, out_w,
        is_neox=False, compiled=_compiled(),
    )  # fmt: skip


def _compare(out, want, what: str) -> None:
    (q_fp8, weights), (want_fp8, want_weights) = out, want
    bad = q_fp8.view(torch.uint8) != want_fp8.view(torch.uint8)
    assert not bool(bad.any()), (
        f"{what}: {int(bad.sum())} of {bad.numel()} e4m3 codes differ; first at "
        f"{tuple(int(i) for i in bad.nonzero()[0])}"
    )
    wbad = weights.view(torch.int32) != want_weights.view(torch.int32)
    assert not bool(wbad.any()), f"{what}: {int(wbad.sum())} of {wbad.numel()} weights differ"


def _assert_replay_matches_eager(spec, seed) -> None:
    """Capture at this batch, replay on new inputs, and require an eager call's bytes."""
    static, fresh, eager = _build(spec, seed), _build(spec, seed + 1), _build(spec, seed + 1)
    op(*static)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream), torch.cuda.graph(graph, stream=stream):
        op(*static)
    torch.cuda.current_stream().wait_stream(stream)
    for i in (0, 1, 2, 3, 4, 5):
        static[i].copy_(fresh[i])
    static[7].view(torch.uint8).fill_(POISON)
    static[8].fill_(float("nan"))
    graph.replay()
    op(*eager)
    torch.cuda.synchronize()
    _compare((static[7], static[8]), (eager[7], eager[8]), "the replay against an eager call")


def _seed(cell) -> int:
    return zlib.crc32(cell.why.encode()) % 2**31


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:48] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811
    spec = cell.spec
    if spec.get("graph"):
        # Before arming: the guard reads the positions back to the host, which a capture
        # forbids.
        _assert_replay_matches_eager(spec, _seed(cell))
    certifying(op)
    for args in _calls(spec, _seed(cell)):
        q_before, wq_before = args[0].clone(), args[5].clone()
        out = op(*args)
        assert out[0] is args[7] and out[1] is args[8], "the op writes the caller's buffers"
        assert torch.equal(args[0], q_before) and torch.equal(args[5], wq_before), (
            "q and the head weights must not be written"
        )
        where = f"positions {int(args[1].min())}..{int(args[1].max())}"
        _compare(out, op.reference(*args), f"the reference at {where}")
        _compare(
            out,
            _replaced(args),
            f"ssmax_scaling + fp32 copy + cute_dsl_indexer_q_side_fused at {where}",
        )


def test_ssmax_and_rope_both_move_the_codes() -> None:
    """The gate sees the scaling and the table: without SSMax, or on the identity table, a
    rope layer's call answers differently in most rows."""
    spec = dict(op.CELLS[0].spec, tokens=64, graph=False)
    args = _build(spec, 61)
    out = op(*args)
    unscaled = list(args)
    unscaled[2] = torch.zeros_like(args[2])
    unscaled[3] = torch.ones_like(args[3])
    no_rope = list(args)
    no_rope[4] = _identity_table(spec["max_pos"])
    for name, wrong in (("no SSMax", unscaled), ("no rope", no_rope)):
        ref = op.reference(*wrong)
        moved = (out[0].view(torch.uint8) != ref[0].view(torch.uint8)).any(-1).float().mean()
        assert float(moved) > 0.5, f"{name}: only {float(moved):.2f} of the rows moved"


def test_a_fresh_launch_overwrites_every_row() -> None:
    spec = dict(op.CELLS[0].spec, tokens=5, graph=False)
    first, second = _build(spec, 31), _build(spec, 32)
    op(*first)
    op(*second[:7], first[7], first[8])
    _compare((first[7], first[8]), op.reference(*second), "the second call's inputs")


@pytest.mark.parametrize(
    "case, match",
    [
        ("int64 positions", "positions must be int32"),
        ("head weights with a strided head dim", "wq must be"),
        ("bf16 table", "cos_sin must be fp32"),
        ("s of 16 heads", "s must be"),
        ("a position past the table", "rope table"),
    ],
)
def test_guard_refuses(case, match) -> None:
    spec = dict(op.CELLS[0].spec, tokens=8, graph=False)
    args = _build(spec, 36)
    if case == "int64 positions":
        args[1] = args[1].long()
    elif case == "head weights with a strided head dim":
        args[5] = torch.zeros(8, HEADS, 2, device="cuda")[..., 0]
    elif case == "bf16 table":
        args[4] = args[4].bfloat16()
    elif case == "s of 16 heads":
        args[2] = args[2][:16]
    else:
        args[1][2] = args[4].shape[0]
    with validating(op):
        with pytest.raises(ValueError, match=match):
            op(*args)


#: The directory of tensors captured from a served decode step.
_CAPTURE_ENV = "TRTLLM_MODELING_V2_FUSE_CAPTURE"


def _assert_old_equals_new(args, what: str) -> None:
    """The fused call into poisoned buffers against the three calls it replaces."""
    q, pos, s, b, table, wq, scale, q_fp8, weights = args
    new_fp8 = torch.full_like(q_fp8.view(torch.uint8), POISON).view(torch.float8_e4m3fn)
    new_w = torch.full_like(weights, float("nan"))
    op(q, pos, s, b, table, wq, scale, new_fp8, new_w)
    old_fp8, old_w = _replaced(args)
    assert torch.equal(new_fp8.view(torch.uint8), old_fp8.view(torch.uint8)), f"{what}: codes"
    assert torch.equal(new_w.view(torch.int32), old_w.view(torch.int32)), f"{what}: weights"


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:48] for c in op.CELLS])
def test_bit_equal_to_the_unfused_composition(cell) -> None:
    for args in _calls(cell.spec, _seed(cell) + 7):
        _assert_old_equals_new(args, cell.why)


@pytest.mark.skipif(
    not os.environ.get(_CAPTURE_ENV), reason=f"{_CAPTURE_ENV} names no captured decode step"
)
def test_bit_equal_to_the_unfused_composition_on_captured_decode_tensors() -> None:
    root = Path(os.environ[_CAPTURE_ENV])
    files = sorted(root.glob("ix_*.pt"))
    assert files, f"no ix_*.pt under {root}"
    tables, rows = {}, 0
    for f in files:
        d = torch.load(f, map_location="cuda")
        name = d["table"]
        if name not in tables:
            tables[name] = torch.load(root / f"table_{name}.pt", map_location="cuda")
        tokens = d["q"].shape[0]
        # The head weights as the targets pass them: the last 32 columns of [T, 160].
        kw = torch.zeros(tokens, HEAD_DIM + HEADS, dtype=d["wq"].dtype, device="cuda")
        kw[:, HEAD_DIM:] = d["wq"]
        args = [
            d["q"], d["positions"], d["s"], d["b"], tables[name], kw[:, HEAD_DIM:],
            d["weight_scale"],
            torch.empty(tokens, HEADS, HEAD_DIM, dtype=torch.float8_e4m3fn, device="cuda"),
            torch.empty(tokens, HEADS, device="cuda"),
        ]  # fmt: skip
        _assert_old_equals_new(args, f.name)
        rows += tokens
    print(f"\n  {len(files)} captured indexer calls, {rows} rows: bit-equal")


# ---- every indexer layer of the checkpoint at every position of the 1M config

#: The NVFP4 checkpoint, read under LLM_MODELS_ROOT for its indexer layers' SSMax s and b and
#: the rope table's inputs.
CHECKPOINTS = ("Mistral-Large-4-1T-A52B-NVFP4", "mistral-large-4-nvfp4")
#: Every position the 1M config reaches (max_seq_len 1048576, the table's rows), in calls of
#: one chunked step's 32768 rows.
LONG_MAX_SEQ_LEN, STEP_ROWS = 1048576, 32768


def _checkpoint_dir() -> Path | None:
    root = llm_models_root()
    for name in CHECKPOINTS if root is not None else ():
        path = Path(root) / name
        names = ("params.json", "consolidated.safetensors.index.json")
        if all((path / f).is_file() for f in names):
            return path
    return None


@pytest.fixture(scope="module")
def checkpoint_indexer() -> list:
    """The NVFP4 checkpoint's indexer layers, each (layer, s, b, table): its SSMax s and b
    (`[32]` each, as stored) and the table its call reads -- the 1048576-row `[cos | sin]`
    indexer table the targets build (`_rope_tables` over the rope parameters the engine adapts
    from params.json and `lerope.alpha` widened to fp32, as the weight table loads it), or the
    identity table on a NoPE indexer layer."""
    path = _checkpoint_dir()
    if path is None:
        pytest.skip(f"none of {CHECKPOINTS} is under LLM_MODELS_ROOT ({llm_models_root()})")
    from safetensors import safe_open

    from tensorrt_llm._torch._experimental.modeling_v2.models.mistral_large4.mistral_large_4_nvfp4__sm_103__tep4 import (  # noqa: E501
        modeling as M,
    )
    from tensorrt_llm._torch.models.checkpoints.mistral.config_loader import MistralConfigLoader

    cfg = MistralConfigLoader().load_pretrained_config(str(path))
    index = json.loads((path / "consolidated.safetensors.index.json").read_text())["weight_map"]
    keys = {
        (layer, p): f"layers.{layer}.attention.indexer.ssmax_scaling.{p}"
        for layer in range(cfg.num_hidden_layers)
        for p in ("s", "b")
        if f"layers.{layer}.attention.indexer.ssmax_scaling.{p}" in index
    }
    keys["alpha"] = "lerope.alpha"
    got = {}
    for shard in sorted({index[k] for k in keys.values()}):
        with safe_open(str(path / shard), framework="pt", device="cpu") as handle:
            for at, key in keys.items():
                if index[key] == shard:
                    got[at] = handle.get_tensor(key)
    rope_cfg = SimpleNamespace(
        rope_parameters=cfg.rope_parameters, qk_rope_head_dim=cfg.qk_rope_head_dim
    )
    tables = M._rope_tables(
        rope_cfg, got["alpha"].float(), LONG_MAX_SEQ_LEN, torch.bfloat16, torch.device("cuda")
    )
    layers = sorted({at[0] for at in keys if at != "alpha"})
    return [
        (
            layer,
            got[layer, "s"].cuda(),
            got[layer, "b"].cuda(),
            tables.identity if cfg.indexer_nope_layer_flags[layer] else tables.indexer,
        )
        for layer in layers
    ]


def test_every_layer_and_position_bit_equal_to_the_unfused_composition(checkpoint_indexer):
    """The NVFP4 checkpoint's own indexer layers (their s and b, the 1048576-row table they
    read, the identity table on a NoPE indexer layer) at every position 0..1048575 of the 1M
    config: the fused call's codes and weights are bitwise the composition it replaces,
    `ssmax_scaling`, the head weights' fp32 copy and `cute_dsl_indexer_q_side_fused`
    (`torch.equal` on the bit patterns). Each call is one chunked step's 32768 consecutive
    positions, half the rows normal queries and half drawn over every bf16 binade, the head
    weights the last 32 columns of a bf16 `[T, 160]` buffer, the same inputs at every layer."""
    assert len(checkpoint_indexer) == 44, len(checkpoint_indexer)
    calls = 0
    for k in range(LONG_MAX_SEQ_LEN // STEP_ROWS):
        g = torch.Generator(device="cuda").manual_seed(1 << 20 | k)
        half = STEP_ROWS // 2
        q = torch.cat(
            [
                _queries((half, HEADS, HEAD_DIM), "normal", g),
                _queries((STEP_ROWS - half, HEADS, HEAD_DIM), "wide", g),
            ]
        )
        kw = torch.randn(STEP_ROWS, HEAD_DIM + HEADS, generator=g, device="cuda").bfloat16()
        start = k * STEP_ROWS
        positions = torch.arange(start, start + STEP_ROWS, dtype=torch.int32, device="cuda")
        for layer, s, b, table in checkpoint_indexer:
            args = [
                q, positions, s, b, table, kw[:, HEAD_DIM:], WEIGHT_SCALE,
                torch.empty(STEP_ROWS, HEADS, HEAD_DIM, dtype=torch.float8_e4m3fn, device="cuda"),
                torch.empty(STEP_ROWS, HEADS, device="cuda"),
            ]  # fmt: skip
            _assert_old_equals_new(args, f"layer {layer} at positions {start}..")
            calls += 1
    print(
        f"\n  {len(checkpoint_indexer)} indexer layers x positions 0..{LONG_MAX_SEQ_LEN - 1}, "
        f"{calls} calls: bit-equal"
    )
