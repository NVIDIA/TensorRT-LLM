# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the ssmax_scaling_fused_q catalog entry.

Every cell is gated bit for bit, three ways:

* against the entry's `reference` (`ssmax_scaling`'s reference chain, then the tail). The
  computation is a chain of bf16 roundings with nothing left to accumulation order, so there
  is no band to derive; `ssmax_scaling`'s test derives where the chain sits from the closed
  form, and this op computes the same chain.
* against the two calls it replaces, `ssmax_scaling` and the tail's copy (or zero fill), run
  on the same inputs: the targets' forward is unchanged bit for bit.
* the fused query's bytes outside the tail -- the other heads, the 512 absorbed columns --
  start poisoned and must come back untouched.

A `position_sweep` cell (every position 0..1048575 of the 1M config, 32 calls of 32768 rows) is
gated so at every call. The old-versus-new tests state the equivalence on its own: the
unfused composition the targets ran (`ssmax_scaling`, then the tail's copy or zero fill into a
poisoned fused query) and the fused call, on identical inputs, compared with `torch.equal` on
the bit patterns -- over every cell's random inputs (every call of a sweep), and over tensors
captured from a served decode step when
TRTLLM_MODELING_V2_FUSE_CAPTURE names their directory (`mla_*.pt`, one file per call: `q`,
`positions`, `s`, `b`, `rope`). `test_every_layer_and_position_bit_equal_to_the_unfused_composition`
states it at the 1M config over the NVFP4 checkpoint's own s and b: every layer, every position
0..1048575, at each call site's layout of the fused query (skipped where the checkpoint is absent
from LLM_MODELS_ROOT). `test_device_log_is_torchs_log` pins the one transcendental: the kernel's
libdevice logf returns torch.log's fp32 bits, so even the unrounded log agrees.
"""

import json
import os
import zlib
from pathlib import Path

import pytest
import torch
import triton
import triton.language as tl
from test_common.llm_data import llm_models_root
from triton.language.extra import libdevice

from tensorrt_llm._torch._experimental.modeling_v2.catalog.attention.ssmax_scaling import (
    ssmax_scaling,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.attention.ssmax_scaling_fused_q import (
    ssmax_scaling_fused_q as op,
)

__extra_import_path__ = [".."]
from _validating import certifying, validating  # noqa: E402,F401

assert torch.cuda.is_available(), "ssmax_scaling_fused_q requires a CUDA device"

#: The fused query FlashMLA reads: 64 kernel heads of [absorbed 512 | rope tail 64].
FQ_HEADS, LATENT, TAIL = 64, 512, 64
#: bf16 NaN with a payload no op writes: the poison every output byte starts from.
POISON = 0x7FA5


def _seed(text: str) -> int:
    return zlib.crc32(text.encode())


def _queries(shape, kind: str, g: torch.Generator) -> torch.Tensor:
    if kind == "normal":
        return (4 * torch.randn(shape, device="cuda", generator=g)).to(torch.bfloat16)
    # Every bf16 binade from the smallest subnormal (2^-133) up, both signs.
    exponent = torch.randint(-133, 127, shape, device="cuda", generator=g).float()
    mantissa = 1 + torch.rand(shape, device="cuda", generator=g)
    sign = 2 * torch.randint(0, 2, shape, device="cuda", generator=g).float() - 1
    return (sign * mantissa * torch.exp2(exponent)).to(torch.bfloat16)


def _poisoned_fq(tokens: int, heads: int = FQ_HEADS) -> torch.Tensor:
    fq = torch.full((tokens, heads, LATENT + TAIL), POISON, dtype=torch.int16, device="cuda")
    return fq.view(torch.bfloat16)


def _build(spec, seed: int):
    """`(q, positions, s, b, fq)`: s and b are rank `rank`'s view into a `param_heads` draw,
    four of its heads on the ranges' corners; `fq` is a poisoned [T, fq_heads, 576] fused
    query (64 heads unless the cell names the native decode's scratch)."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    tokens, heads = spec["tokens"], spec["heads"]
    q = _queries((tokens, heads, 128), spec["q"], g)
    cached = spec.get("cached", (0,) * len(spec["contexts"]))
    assert len(cached) == len(spec["contexts"]), (cached, spec["contexts"])
    assert all(c + n <= spec["max_seq_len"] for n, c in zip(spec["contexts"], cached))
    parts = [torch.arange(c, c + n, device="cuda") for n, c in zip(spec["contexts"], cached)]
    if spec["decode_rows"]:
        top = spec["max_seq_len"]
        rows = torch.randint(1, top, (spec["decode_rows"],), device="cuda", generator=g)
        rows[0] = top - 1
        parts.append(rows)
    positions = torch.cat(parts).to(spec["positions_dtype"])
    assert positions.numel() == tokens
    full = [
        torch.empty(spec["param_heads"], device="cuda").uniform_(*spec[k], generator=g).bfloat16()
        for k in ("s_range", "b_range")
    ]
    lo = spec["rank"] * heads
    if spec["corners"]:
        at = lo + (4 * spec["rank"]) % heads
        corners = [(s, b) for s in spec["s_range"] for b in spec["b_range"]]
        for i, values in enumerate(zip(*corners)):
            full[i][at : at + 4] = torch.tensor(values, device="cuda").bfloat16()
    s, b = (t[lo : lo + heads] for t in full)
    fq = _poisoned_fq(tokens, spec.get("fq_heads", FQ_HEADS))
    return q, positions, s, b, fq


def _sweep(spec, seed: int):
    """A `position_sweep` (lo, hi) cell's calls, `(q, positions, s, b, fq)` each: every
    `tokens` rows of positions lo..hi - 1 in order, each call on a fresh q and a fresh poisoned
    fused query, the cell's own s and b."""
    lo, hi = spec["position_sweep"]
    rows = spec["tokens"]
    assert (hi - lo) % rows == 0 and hi <= spec["max_seq_len"], (lo, hi, rows)
    _, _, s, b, _ = _build(spec, seed)
    for k in range((hi - lo) // rows):
        q, _, _, _, fq = _build(spec, seed + 1 + k)
        start = lo + k * rows
        positions = torch.arange(start, start + rows, dtype=spec["positions_dtype"], device="cuda")
        yield q, positions, s, b, fq


def _bits_equal(got: torch.Tensor, want: torch.Tensor, what: str) -> None:
    assert got.shape == want.shape, (what, got.shape, want.shape)
    differ = got.contiguous().view(torch.int16) != want.contiguous().view(torch.int16)
    n = int(differ.sum())
    if n:
        at = tuple(int(i) for i in differ.nonzero()[0])
        raise AssertionError(f"{what}: {n} of {got.numel()} elements differ; first at {at}")


def _call(q, positions, s, b, fq, spec):
    """The targets' call: `q` scaled in place (or into a fresh buffer), the tail into `fq`."""
    out = q if spec["inplace"] else torch.empty_like(q)
    return op(q, positions, s, b, out, fq[:, : spec["heads"], LATENT:], spec["rope"])


def _check(spec, q_in, positions, s, b, fq, out):
    """Gate one call's results against the reference and the replaced pair of calls."""
    heads = spec["heads"]
    ref_out, ref_tail = op.reference(
        q_in, positions, s, b, out, fq[:, :heads, LATENT:], spec["rope"]
    )
    _bits_equal(out, ref_out, "out against the reference")
    _bits_equal(fq[:, :heads, LATENT:], ref_tail, "tail against the reference")
    old = ssmax_scaling(q_in, positions, s, b)
    _bits_equal(out, old, "out against ssmax_scaling")
    want_tail = old[..., TAIL:] if spec["rope"] else torch.zeros_like(old[..., TAIL:])
    _bits_equal(fq[:, :heads, LATENT:], want_tail, "tail against the replaced copy")
    rest = fq.view(torch.int16).clone()
    rest[:, :heads, LATENT:] = POISON
    assert bool((rest == POISON).all()), "a byte of the fused query outside the tail was written"


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:48] for c in op.CELLS])
def test_certified_cells(cell, certifying):  # noqa: F811
    spec, seed = cell.spec, _seed(cell.why)
    if "position_sweep" in spec:
        certifying(op)
        calls = 0
        for q, positions, s, b, fq in _sweep(spec, seed):
            q_in = q.clone()
            out = _call(q, positions, s, b, fq, spec)
            _check(spec, q_in, positions, s, b, fq, out)
            calls += 1
        print(
            f"\n  {cell.why}: {calls} calls, each bitwise the reference and ssmax_scaling + the tail"
        )
        return
    if spec["graph"]:
        # Captured first and replayed on new inputs, then held to an eager call on them.
        q, positions, s, b, fq = _build(spec, seed)
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            _call(q.clone(), positions, s, b, fq.clone(), spec)
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = _call(q, positions, s, b, fq, spec)
        fresh_q, fresh_positions, _, _, _ = _build(spec, seed + 1)
        q.copy_(fresh_q)
        positions.copy_(fresh_positions)
        fq.view(torch.int16).fill_(POISON)
        q_in = q.clone()
        graph.replay()
        torch.cuda.synchronize()
        eager_fq = torch.full_like(fq.view(torch.int16), POISON).view(torch.bfloat16)
        eager = _call(q_in.clone(), positions, s, b, eager_fq, spec)
        _bits_equal(out, eager, "the replay against an eager call")
        _bits_equal(fq, eager_fq, "the replayed fused query against an eager call")
    else:
        q, positions, s, b, fq = _build(spec, seed)
        q_in = q.clone()
    certifying(op)
    s_before, b_before, positions_before = s.clone(), b.clone(), positions.clone()
    fq.view(torch.int16).fill_(POISON)
    if spec["inplace"]:
        q.copy_(q_in)
    out = _call(q, positions, s, b, fq, spec)
    assert out.shape == q_in.shape and out.dtype == torch.bfloat16
    assert (out.data_ptr() == q.data_ptr()) == spec["inplace"]
    if not spec["inplace"]:
        assert torch.equal(q, q_in), "an out-of-place call must not write q"
    for now, then in ((s, s_before), (b, b_before), (positions, positions_before)):
        assert torch.equal(now, then), "the entry must not mutate positions, s or b"
    _check(spec, q_in, positions, s, b, fq, out)
    print(
        f"\n  {cell.why}: bitwise the reference and ssmax_scaling + the tail "
        f"{'copy' if spec['rope'] else 'zero fill'}; {int(torch.isinf(out).sum())} overflowed"
    )


def test_rows_are_independent() -> None:
    """A padded decode capture runs dummy rows beside the live ones."""
    spec = dict(op.CELLS[0].spec, tokens=16, decode_rows=16)
    q, positions, s, b, fq = _build(spec, 51)
    live_fq = fq.clone()
    live = _call(q[:3].clone(), positions[:3].clone(), s, b, live_fq[:3], spec)
    q[3:] = float("nan")
    q[3:, :, ::2] = float("inf")
    positions[3], positions[4], positions[5:] = -1, -2, 2**30
    out = _call(q, positions, s, b, fq, spec)
    _bits_equal(out[:3], live, "the live rows beside poisoned ones")
    _bits_equal(fq[:3], live_fq[:3], "the live rows' fused query beside poisoned ones")


def test_gates_see_a_wrong_tail() -> None:
    """The rope-layer and NoPE-layer tails differ, and the gate tells them apart; so does a
    tail written from the wrong half."""
    spec = dict(op.CELLS[0].spec, tokens=16, decode_rows=16, rope=True)
    q, positions, s, b, fq = _build(spec, 52)
    q_in = q.clone()
    out = _call(q, positions, s, b, fq, spec)
    _check(spec, q_in, positions, s, b, fq, out)
    with pytest.raises(AssertionError, match="tail against the reference"):
        _check(dict(spec, rope=False), q_in, positions, s, b, fq, out)
    fq[:, : spec["heads"], LATENT:] = out[..., :TAIL]
    with pytest.raises(AssertionError, match="tail against the reference"):
        _check(spec, q_in, positions, s, b, fq, out)


def _guard_inputs():
    spec = dict(op.CELLS[0].spec, tokens=16, decode_rows=16)
    q, positions, s, b, fq = _build(spec, 41)
    return spec, q, positions, s, b, fq


@pytest.mark.parametrize(
    "case",
    [
        "s of the unsliced 64 heads",
        "positions of length 1",
        "tail of another width",
        "out partly overlapping q",
        "tail inside q",
        "q with a strided head dim",
    ],
)
def test_guard_refuses(case) -> None:
    spec, q, positions, s, b, fq = _guard_inputs()
    heads = spec["heads"]
    out, tail = q, fq[:, :heads, LATENT:]
    if case == "s of the unsliced 64 heads":
        s = torch.cat([s] * 4)
    elif case == "positions of length 1":
        positions = positions[:1]
    elif case == "tail of another width":
        tail = fq[:, :heads, LATENT - TAIL :]
    elif case == "out partly overlapping q":
        flat = torch.empty(q.numel() + 64, dtype=q.dtype, device="cuda")
        flat[: q.numel()] = q.flatten()
        q = flat[: q.numel()].view_as(q)
        out = flat[64:].view_as(q)
    elif case == "tail inside q":
        tail = q[..., TAIL:]
    else:
        q = torch.empty(*q.shape, 2, dtype=q.dtype, device="cuda")[..., 0]
        out = q
    with validating(op):
        with pytest.raises(ValueError, match="ssmax_scaling_fused_q"):
            op(q, positions, s, b, out, tail, True)


#: The directory of tensors captured from a served decode step.
_CAPTURE_ENV = "TRTLLM_MODELING_V2_FUSE_CAPTURE"


def _unfused(q, positions, s, b, rope: bool, fq_heads: int = FQ_HEADS):
    """The composition the targets ran: ssmax_scaling, then the tail's copy or zero fill."""
    heads = q.shape[1]
    out = ssmax_scaling(q, positions, s, b)
    fq = _poisoned_fq(q.shape[0], fq_heads)
    if rope:
        fq[:, :heads, LATENT:].copy_(out[..., TAIL:])
    else:
        fq[:, :heads, LATENT:].zero_()
    return out, fq


def _fused(q, positions, s, b, rope: bool, fq_heads: int = FQ_HEADS):
    """The targets' fused call: `q` scaled in place, the tail written into the fused query."""
    out = q.clone()
    fq = _poisoned_fq(q.shape[0], fq_heads)
    op(out, positions, s, b, out, fq[:, : q.shape[1], LATENT:], rope)
    return out, fq


def _assert_old_equals_new(
    q, positions, s, b, rope: bool, what: str, fq_heads: int = FQ_HEADS
) -> None:
    old_out, old_fq = _unfused(q, positions, s, b, rope, fq_heads)
    new_out, new_fq = _fused(q, positions, s, b, rope, fq_heads)
    assert torch.equal(new_out.view(torch.int16), old_out.view(torch.int16)), f"{what}: out"
    assert torch.equal(new_fq.view(torch.int16), old_fq.view(torch.int16)), f"{what}: fused q"


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:48] for c in op.CELLS])
def test_bit_equal_to_the_unfused_composition(cell) -> None:
    spec, seed = cell.spec, _seed(cell.why) + 7
    calls = _sweep(spec, seed) if "position_sweep" in spec else [_build(spec, seed)]
    for q, positions, s, b, _ in calls:
        _assert_old_equals_new(
            q, positions, s, b, spec["rope"], cell.why, spec.get("fq_heads", FQ_HEADS)
        )


@pytest.mark.skipif(
    not os.environ.get(_CAPTURE_ENV), reason=f"{_CAPTURE_ENV} names no captured decode step"
)
def test_bit_equal_to_the_unfused_composition_on_captured_decode_tensors() -> None:
    files = sorted(Path(os.environ[_CAPTURE_ENV]).glob("mla_*.pt"))
    assert files, f"no mla_*.pt under {os.environ[_CAPTURE_ENV]}"
    rows = 0
    for f in files:
        d = torch.load(f, map_location="cuda")
        _assert_old_equals_new(d["q"], d["positions"], d["s"], d["b"], d["rope"], f.name)
        rows += d["q"].shape[0]
    print(f"\n  {len(files)} captured MLA calls, {rows} rows: bit-equal")


@triton.jit
def _logf_kernel(x_ptr, y_ptr, n, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(x_ptr + i, mask=i < n)
    tl.store(y_ptr + i, libdevice.log(x), mask=i < n)


def test_device_log_is_torchs_log() -> None:
    """libdevice's logf, compiled as the kernel compiles it (no implicit FMA contraction),
    returns torch.log's fp32 bits: at every `pos + 1` for positions 0..2^24 - 1 (every
    position fp32 holds exactly), and at 2^24 random positive finite fp32 bit patterns."""
    g = torch.Generator(device="cuda").manual_seed(5)
    x = torch.cat(
        [
            torch.arange(1, 2**24 + 1, dtype=torch.float32, device="cuda"),
            torch.randint(
                1, 0x7F800000, (2**24,), generator=g, device="cuda", dtype=torch.int32
            ).view(torch.float32),
        ]
    )
    y = torch.empty_like(x)
    _logf_kernel[(triton.cdiv(x.numel(), 1024),)](
        x, y, x.numel(), BLOCK=1024, enable_fp_fusion=False
    )
    assert torch.equal(y.view(torch.int32), torch.log(x).view(torch.int32))


# ---- every layer of the checkpoint at every position of the 1M config

#: The NVFP4 checkpoint, read under LLM_MODELS_ROOT for its per-layer SSMax s and b.
CHECKPOINTS = ("Mistral-Large-4-1T-A52B-NVFP4", "mistral-large-4-nvfp4")
#: Every position the 1M config reaches (max_seq_len 1048576), in calls of one chunked step's
#: 32768 rows.
LONG_MAX_SEQ_LEN, STEP_ROWS = 1048576, 32768
#: The call sites' layouts, (a rank's heads, the fused query's heads): dep4's 64 heads into
#: FlashMLA's 64-head query; a tp4 / tep4 rank's 16 into the native decode's `[T, 16, 576]`
#: scratch and into FlashMLA's padded 64-head one (the prefill scratch, and the decode scratch
#: under `_GEN_ATTN_FLASH_MLA`); a tp8 / tep8 rank's 8 the same. A layout's ranks together
#: hold all 64 heads.
LAYOUTS = {
    "dep4": (64, 64),
    "tp4_native": (16, 16),
    "tp4_padded": (16, 64),
    "tp8_native": (8, 8),
    "tp8_padded": (8, 64),
}


def _checkpoint_dir() -> Path | None:
    root = llm_models_root()
    for name in CHECKPOINTS if root is not None else ():
        path = Path(root) / name
        names = ("params.json", "consolidated.safetensors.index.json")
        if all((path / f).is_file() for f in names):
            return path
    return None


@pytest.fixture(scope="module")
def checkpoint_mla() -> list:
    """Every layer's MLA SSMax s and b (`[64]` each, as stored) and whether it ropes its query
    (not a NoPE layer), read off the NVFP4 checkpoint."""
    path = _checkpoint_dir()
    if path is None:
        pytest.skip(f"none of {CHECKPOINTS} is under LLM_MODELS_ROOT ({llm_models_root()})")
    from safetensors import safe_open

    from tensorrt_llm._torch.models.checkpoints.mistral.config_loader import MistralConfigLoader

    cfg = MistralConfigLoader().load_pretrained_config(str(path))
    index = json.loads((path / "consolidated.safetensors.index.json").read_text())["weight_map"]
    keys = {
        (layer, p): f"layers.{layer}.attention.ssmax_scaling.{p}"
        for layer in range(cfg.num_hidden_layers)
        for p in ("s", "b")
    }
    got = {}
    for shard in sorted({index[k] for k in keys.values()}):
        with safe_open(str(path / shard), framework="pt", device="cpu") as handle:
            for at, key in keys.items():
                if index[key] == shard:
                    got[at] = handle.get_tensor(key).cuda()
    return [
        (got[layer, "s"], got[layer, "b"], not cfg.nope_layer_flags[layer])
        for layer in range(cfg.num_hidden_layers)
    ]


@pytest.mark.parametrize("layout", list(LAYOUTS))
def test_every_layer_and_position_bit_equal_to_the_unfused_composition(checkpoint_mla, layout):
    """The NVFP4 checkpoint's own s and b of every layer (all 72, RoPE and NoPE), at every
    position 0..1048575 of the 1M config, in each call site's layout (`LAYOUTS`, every rank's
    slice of the 64 heads): the fused call is bitwise the composition it replaces,
    `ssmax_scaling` then the tail's copy or zero fill -- the scaled query and the whole fused
    query, the tail written and every other byte untouched, `torch.equal` on the bit patterns.
    Each call is one chunked step's 32768 consecutive positions, half the rows normal queries
    and half drawn over every bf16 binade, the same queries at every layer."""
    heads, fq_heads = LAYOUTS[layout]
    assert all(s.shape == b.shape == (64,) for s, b, _ in checkpoint_mla)
    calls = 0
    for k in range(LONG_MAX_SEQ_LEN // STEP_ROWS):
        g = torch.Generator(device="cuda").manual_seed(1 << 20 | k)
        half = STEP_ROWS // 2
        q = torch.cat(
            [
                _queries((half, heads, 128), "normal", g),
                _queries((STEP_ROWS - half, heads, 128), "wide", g),
            ]
        )
        start = k * STEP_ROWS
        positions = torch.arange(start, start + STEP_ROWS, dtype=torch.int32, device="cuda")
        for layer, (s_all, b_all, rope) in enumerate(checkpoint_mla):
            for lo in range(0, 64, heads):
                what = f"{layout} layer {layer} heads {lo}..{lo + heads - 1} at {start}.."
                _assert_old_equals_new(
                    q, positions, s_all[lo : lo + heads], b_all[lo : lo + heads], rope, what,
                    fq_heads,
                )  # fmt: skip
                calls += 1
    print(
        f"\n  {layout}: {len(checkpoint_mla)} layers x positions 0..{LONG_MAX_SEQ_LEN - 1}, "
        f"{calls} calls: bit-equal"
    )
