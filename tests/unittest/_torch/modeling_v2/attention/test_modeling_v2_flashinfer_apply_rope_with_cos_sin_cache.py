# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the flashinfer_apply_rope_with_cos_sin_cache entry:
the cells, the out-of-place route at straddling lane counts, the wrong
variants the band rejects, the guards and CUDA-graph replay."""

import math
import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog.attention.flashinfer_apply_rope_with_cos_sin_cache import (  # noqa: E501
    TABLE_ROWS,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.attention.flashinfer_apply_rope_with_cos_sin_cache import (  # noqa: E501
    flashinfer_apply_rope_with_cos_sin_cache as op,
)

__extra_import_path__ = [".."]
from _validating import certifying, validating  # noqa: E402,F401

assert torch.cuda.is_available(), "flashinfer_apply_rope_with_cos_sin_cache requires a CUDA device"

DEV = "cuda"

#: The kernel forms each output as `x1 * c - x2 * s` (or `x2 * c + x1 * s`) in
#: fp32: two products and a sum, each rounded with relative error u = 2^-24,
#: so the fp32 value lies within 3u * (|x1 c| + |x2 s|) of the exact one;
#: 2^-22 = 4u. Rounding that value to bf16 moves it by at most half a bf16
#: spacing, and the reference is the exact value rounded once, another half
#: spacing away. So each element is within one spacing of the reference (the
#: spacing above |ref|, the wider one at a power of two) plus 2^-22 times its
#: own product magnitude, which is what bounds the cancellation elements where
#: the result is tiny next to its terms.
FP32_EVAL = 2.0**-22
#: An element lands off the once-rounded reference only when the exact value
#: lies within the fp32 error of a bf16 rounding midpoint: about
#: 2 * 4u / 2^-8 = 1.2e-4 per element at worst. Measured at most 1.23e-5 over
#: every rotary_dim the per-head-dim kernel serves; the gate allows 1e-3.
MAX_OFF_FRACTION = 1e-3

#: Step-5's `rope_scaling` block (text_config), applied by its reference to the
#: full-attention layers' frequencies (and so to their indexer's).
LLAMA3 = dict(factor=1.0, low_freq_factor=1.0, high_freq_factor=32.0, original_max=1048576.0)

#: The decode positions span the table: Step-5's max_seq_len is 9216.
CHUNK_START = 8704


def _seed(text: str) -> int:
    return zlib.crc32(text.encode()) % 2**31


def _frequencies(rotary_dim: int, theta: float, llama3: bool) -> torch.Tensor:
    """fp32 inverse frequencies `theta^(-2i / rotary_dim)`, and the llama3
    wavelength blend when the layer carries it (at factor 1.0 it changes one
    of r64's 32 frequencies by an fp32 ulp)."""
    inv = 1.0 / (
        theta ** (torch.arange(0, rotary_dim, 2, device=DEV, dtype=torch.float32) / rotary_dim)
    )
    if not llama3:
        return inv
    factor, orig = LLAMA3["factor"], LLAMA3["original_max"]
    lo, hi = LLAMA3["low_freq_factor"], LLAMA3["high_freq_factor"]
    wavelength = 2 * math.pi / inv
    scaled = torch.where(wavelength > orig / lo, inv / factor, inv)
    blend = (orig / wavelength - lo) / (hi - lo)
    smoothed = (1 - blend) * scaled / factor + blend * scaled
    in_band = (wavelength >= orig / hi) & (wavelength <= orig / lo)
    return torch.where(in_band, smoothed, scaled)


def _table(
    rotary_dim: int, theta: float, llama3: bool, valued: str, rows: int = TABLE_ROWS
) -> torch.Tensor:
    """`[rows, rotary_dim]` fp32 `[cos | sin]` of `position * frequency` in
    fp32; `valued="bf16"` rounds both halves to bf16 (Step-5's stored table)
    and widens them back."""
    angles = torch.outer(
        torch.arange(rows, device=DEV, dtype=torch.float32), _frequencies(rotary_dim, theta, llama3)
    )
    cos, sin = angles.cos(), angles.sin()
    if valued == "bf16":
        cos, sin = cos.bfloat16().float(), sin.bfloat16().float()
    else:
        assert valued == "fp32", valued
    return torch.cat([cos, sin], dim=1).contiguous()


def _positions(kind: str, tokens: int, seed: int) -> torch.Tensor:
    if kind == "prefill":
        return torch.arange(tokens, device=DEV, dtype=torch.int32)
    if kind == "chunk":
        return torch.arange(CHUNK_START, CHUNK_START + tokens, device=DEV, dtype=torch.int32)
    assert kind == "decode", kind
    g = torch.Generator(device=DEV).manual_seed(seed)
    pos = torch.randint(0, TABLE_ROWS, (tokens,), generator=g, device=DEV, dtype=torch.int32)
    pos[-1] = TABLE_ROWS - 1
    if tokens > 1:
        pos[0] = 0
    return pos


class _Case:
    """One call's tensors: q and k (views of `buffer` when packed), and what
    the call must leave untouched."""

    def __init__(self, spec: dict, seed: int):
        self.spec = spec
        tokens, hq, hk = spec["tokens"], spec["hq"], spec["hk"]
        width, live = spec["width"], spec.get("live", spec["width"])
        g = torch.Generator(device=DEV).manual_seed(seed)
        heads = hq + 2 * hk if spec["layout"] == "packed" else hq + hk
        buffer = torch.zeros(tokens, heads, width, device=DEV, dtype=torch.bfloat16)
        buffer[..., :live] = torch.randn(tokens, heads, live, generator=g, device=DEV).bfloat16()
        flat = buffer.reshape(tokens, heads * width)
        if spec["layout"] == "packed":
            self.query = flat[:, : hq * width]
            self.key = flat[:, hq * width : (hq + hk) * width]
        else:
            assert spec["layout"] == "split", spec["layout"]
            self.query = flat[:, : hq * width].contiguous()
            self.key = flat[:, hq * width :].contiguous()
        self.buffer = buffer
        self.before = buffer.clone()
        self.query0, self.key0 = self.query.clone(), self.key.clone()
        self.positions = _positions(spec["positions"], tokens, seed)
        self.table = _table(spec["rotary_dim"], spec["theta"], spec["llama3"], spec["table"])

    def args(self) -> tuple:
        return self.positions, self.query, self.key, self.spec["head_size"], self.table, True

    def pristine_args(self) -> tuple:
        return self.positions, self.query0, self.key0, self.spec["head_size"], self.table, True

    def assert_untouched(self) -> None:
        """Every dim at or past rotary_dim, and the v heads of a packed buffer,
        are bit-unchanged (the zero pad of the 256-wide layout included)."""
        r = self.spec["rotary_dim"]
        hs = self.spec["head_size"]
        for now, then in ((self.query, self.query0), (self.key, self.key0)):
            a = now.reshape(now.shape[0], -1, hs)
            b = then.reshape(then.shape[0], -1, hs)
            assert torch.equal(a[..., r:], b[..., r:]), "the call wrote past rotary_dim"
        if self.spec["layout"] == "packed":
            vh = self.spec["hq"] + self.spec["hk"]
            assert torch.equal(self.buffer[:, vh:], self.before[:, vh:]), (
                "the call wrote the v heads"
            )


def _spacing_above(ref: torch.Tensor) -> torch.Tensor:
    """Distance from |ref| to the next larger bf16 value, in fp64."""
    mag = ref.abs().contiguous()
    nxt = (mag.view(torch.int16) + 1).view(torch.bfloat16)
    return nxt.double() - mag.double()


def _product_scale(positions, x, head_size, table) -> torch.Tensor:
    """|x1 c| + |x2 s| (and its pair twin) per rotated element; zero elsewhere."""
    r = table.shape[1]
    h = r // 2
    rows = table.double()[positions[: x.shape[0]].long()]
    c, s = rows[:, None, :h].abs(), rows[:, None, h:r].abs()
    heads = x.reshape(x.shape[0], -1, head_size).double().abs()
    scale = torch.zeros_like(heads)
    scale[..., :h] = heads[..., :h] * c + heads[..., h:r] * s
    scale[..., h:r] = heads[..., h:r] * c + heads[..., :h] * s
    return scale.reshape(x.shape)


def _measure(out, ref, scale) -> tuple[float, float]:
    """(worst error / bound, fraction of elements != reference)."""
    o, r = out.double(), ref.double()
    bound = _spacing_above(ref) + FP32_EVAL * scale
    ratio = ((o - r).abs() / bound).max().item()
    off = (out.reshape(-1) != ref.reshape(-1)).double().mean().item()
    return ratio, off


def _compare(out, ref, scale) -> tuple[float, float]:
    assert out.shape == ref.shape and out.dtype == ref.dtype, (
        out.shape,
        ref.shape,
        out.dtype,
        ref.dtype,
    )
    assert not torch.isnan(out).any()
    ratio, off = _measure(out, ref, scale)
    assert ratio <= 1.0, f"an element is {ratio:.3g}x outside the derived band"
    assert off <= MAX_OFF_FRACTION, f"{off:.3e} of elements differ from the once-rounded reference"
    return ratio, off


def _check(case: _Case, outs, refs) -> list[tuple[float, float]]:
    stats = []
    for out, ref, x0 in zip(outs, refs, (case.query0, case.key0)):
        scale = _product_scale(case.positions, x0, case.spec["head_size"], case.table)
        stats.append(_compare(out.reshape(ref.shape), ref, scale))
    return stats


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying):  # noqa: F811
    certifying(op)
    case = _Case(cell.spec, _seed(cell.why))
    ref_q, ref_k = op.reference(*case.pristine_args())
    out_q, out_k = op(*case.args())
    assert out_q is case.query and out_k is case.key, "the entry returns the tensors it rotated"
    case.assert_untouched()
    (rq, oq), (rk, ok) = _check(case, (out_q, out_k), (ref_q, ref_k))
    print(f"\n  {cell.why}: band ratio q {rq:.5f} k {rk:.5f}; off q {oq:.3e} k {ok:.3e}")


def _sliding_h192(**kw) -> dict:
    return dict(
        head_size=192, width=192, rotary_dim=192, theta=1e4, llama3=False, table="fp32", **kw
    )


def test_in_place_op_equals_out_of_place_flashinfer():
    """At head 192 / rotary 192 the op's result is bit-equal to FlashInfer's
    out-of-place `apply_rope_with_cos_sin_cache` on the same inputs, which is
    inside the band: the op takes the out-of-place route there."""
    from flashinfer.rope import apply_rope_with_cos_sin_cache

    case = _Case(_sliding_h192(hq=64, hk=4, tokens=128, positions="decode", layout="split"), 7)
    ref_q, ref_k = op.reference(*case.pristine_args())
    new_q, new_k = apply_rope_with_cos_sin_cache(
        case.positions, case.query0, case.key0, 192, case.table, True
    )
    stats = _check(case, (new_q, new_k), (ref_q, ref_k))
    op(*case.args())
    print(f"\n  out-of-place {stats}")
    assert torch.equal(case.query, new_q) and torch.equal(case.key, new_k)


@pytest.mark.parametrize("rotary_dim", [192, 160, 96])
def test_straddling_lane_groups_are_right_and_deterministic(rotary_dim):
    """Rotary 192, 160 and 96 on 192-wide heads give a token 12, 10 and 6
    lanes, which straddle warps, so the op rotates them out of place. Each
    repeat of the call on the same inputs is inside the band and bit-equal to
    the first."""
    spec = dict(
        head_size=192,
        width=192,
        rotary_dim=rotary_dim,
        theta=1e4,
        llama3=False,
        hq=64,
        hk=4,
        tokens=128,
        positions="decode",
        layout="packed",
        table="fp32",
    )
    first = None
    for _ in range(20):
        case = _Case(spec, 17)
        out_q, out_k = op(*case.args())
        if first is None:
            refs = op.reference(*case.pristine_args())
            case.assert_untouched()
            _check(case, (out_q, out_k), refs)
            first = case.buffer.clone()
        else:
            assert torch.equal(case.buffer, first), "a repeat differs from the first call"


_GRAPH_SITES = (
    (
        "full dp h192 r64",
        dict(head_size=192, width=192, rotary_dim=64, theta=1e7, llama3=True, hq=64, hk=4),
    ),
    (
        "sliding dp h192 r192",
        dict(head_size=192, width=192, rotary_dim=192, theta=1e4, llama3=False, hq=64, hk=4),
    ),
    (
        "sliding dp pad256 r192",
        dict(
            head_size=256, width=256, live=192, rotary_dim=192, theta=1e4, llama3=False, hq=64, hk=4
        ),
    ),
    (
        "indexer dp h256 r32",
        dict(head_size=256, width=256, rotary_dim=32, theta=1e7, llama3=True, hq=16, hk=1),
    ),
)


@pytest.mark.parametrize("site", [s for _, s in _GRAPH_SITES], ids=[n for n, _ in _GRAPH_SITES])
def test_cuda_graph_capture_and_replay(site, certifying):  # noqa: F811
    """The decode step is captured: warm up on a side stream, capture at 128
    rows, refill positions, q and k in place, replay; the result is the
    reference of the new inputs and bitwise the eager result. At h192 r192
    the out-of-place route allocates its temporaries while the graph is
    captured."""
    certifying(op)
    spec = dict(site, tokens=128, positions="decode", layout="packed", table="fp32")
    case = _Case(spec, 1)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        op(*case.args())
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        op(*case.args())
    fresh = _Case(spec, 2)
    fresh.positions = _positions("decode", 128, 3)
    case.buffer.copy_(fresh.buffer)
    case.positions.copy_(fresh.positions)
    graph.replay()
    torch.cuda.synchronize()
    ref_q, ref_k = op.reference(*fresh.pristine_args())
    _check(fresh, (case.query, case.key), (ref_q, ref_k))
    op(*fresh.args())
    assert torch.equal(case.query, fresh.query) and torch.equal(case.key, fresh.key)


def test_band_refuses_wrong_variants():
    """Each wrong computation lands outside the band, on the full-layer and
    indexer decode cells. The fp32-built table against Step-5's bf16-valued one
    is the table difference a target chooses between: recorded, and the gate
    tells the two apart."""
    sites = {
        "full dp h192 r64": dict(
            head_size=192, width=192, rotary_dim=64, theta=1e7, llama3=True, hq=64, hk=4
        ),
        "indexer dp h256 r32": dict(
            head_size=256, width=256, rotary_dim=32, theta=1e7, llama3=True, hq=16, hk=1
        ),
        "sliding dp pad256 r192": dict(
            head_size=256, width=256, live=192, rotary_dim=192, theta=1e4, llama3=False, hq=64, hk=4
        ),
        "sliding dp h192 r192": dict(
            head_size=192, width=192, rotary_dim=192, theta=1e4, llama3=False, hq=64, hk=4
        ),
    }
    for name, site in sites.items():
        spec = dict(site, tokens=128, positions="decode", layout="split", table="fp32")
        base = _Case(spec, 5)
        ref_q, _ = op.reference(*base.pristine_args())
        scale = _product_scale(base.positions, base.query0, spec["head_size"], base.table)
        r = spec["rotary_dim"]
        variants = {}

        c = _Case(spec, 5)
        op(c.positions, c.query, c.key, spec["head_size"], c.table, False)
        variants["interleaved pairs"] = c.query
        c = _Case(spec, 5)
        op(
            c.positions.clamp(max=TABLE_ROWS - 2) + 1,
            c.query,
            c.key,
            spec["head_size"],
            c.table,
            True,
        )
        variants["position + 1"] = c.query
        c = _Case(spec, 5)
        inverse = torch.cat([c.table[:, : r // 2], -c.table[:, r // 2 :]], dim=1).contiguous()
        op(c.positions, c.query, c.key, spec["head_size"], inverse, True)
        variants["inverse rotation"] = c.query

        for label, out in variants.items():
            ratio, off = _measure(out, ref_q, scale)
            print(f"\n  {name} {label}: band ratio {ratio:.3g}, off {off:.3e}")
            with pytest.raises(AssertionError):
                _compare(out, ref_q, scale)

        # The table choice: fp32-built values through the op, against Step-5's
        # arithmetic (the bf16-valued table, rotated exactly, rounded once).
        bf16_table = _table(r, spec["theta"], spec["llama3"], "bf16")
        step5_q, _ = op.reference(
            base.positions, base.query0, base.key0, spec["head_size"], bf16_table, True
        )
        c = _Case(spec, 5)
        op(*c.args())
        ratio, off = _measure(c.query, step5_q, scale)
        diff = (c.query.double() - step5_q.double()).reshape(128, -1, spec["head_size"])[..., :r]
        row = (
            step5_q.double()
            .reshape(128, -1, spec["head_size"])[..., :r]
            .abs()
            .amax(dim=-1, keepdim=True)
        )
        rel_ulp = (diff.abs() / row).max().item() / 2.0**-8
        print(
            f"  {name} fp32 table vs Step-5 bf16 table: off {off:.3e}, band ratio {ratio:.3g}, "
            f"max |diff| {rel_ulp:.3f} bf16 ulp of the head's max"
        )
        assert off > MAX_OFF_FRACTION


@pytest.mark.parametrize("rotary_dim", [16, 48])
def test_guard_refuses_a_rotary_dim_rope_quantize_rotates_wrongly(rotary_dim):
    """On a 192-wide head: rotary 16 and 48 have halves of 8 and 24, not a
    whole 16-element vector, and the kernel misreads the half a lane's vector
    spans, in place or out of place. Unguarded the op answers outside the band
    with no error."""
    spec = dict(
        head_size=192,
        width=192,
        rotary_dim=rotary_dim,
        theta=1e4,
        llama3=False,
        hq=16,
        hk=1,
        tokens=128,
        positions="decode",
        layout="split",
        table="fp32",
    )
    case = _Case(spec, 11)
    ref_q, _ = op.reference(*case.pristine_args())
    op(*case.args())
    ratio, off = _measure(
        case.query, ref_q, _product_scale(case.positions, case.query0, 192, case.table)
    )
    print(f"\n  r{rotary_dim} unguarded: band ratio {ratio:.3g}, off {off:.3e}")
    assert ratio > 1.0
    case = _Case(spec, 11)
    with pytest.raises(AssertionError, match="RopeQuantizeKernel"):
        with validating(op):
            op(*case.args())


@pytest.mark.parametrize("rotary_dim", [32, 96, 128, 160])
def test_guard_admits_the_rotary_dims_rope_quantize_gets_right(rotary_dim):
    """On a 192-wide head the guard admits every rotary dim whose half is a
    whole 16-element vector: r32 and r128 (in place) and r96 and r160 (out of
    place), all right at 8192 prefill rows."""
    spec = dict(
        head_size=192,
        width=192,
        rotary_dim=rotary_dim,
        theta=1e7,
        llama3=False,
        hq=64,
        hk=4,
        tokens=8192,
        positions="prefill",
        layout="packed",
        table="fp32",
    )
    case = _Case(spec, 12)
    refs = op.reference(*case.pristine_args())
    with validating(op):
        outs = op(*case.args())
    _check(case, outs, refs)


def test_guard_refuses_positions_past_the_table():
    """The table is a view of larger storage the test owns: positions 9216 and
    9220 read that storage's next rows, with no error."""
    owner = _table(64, 1e7, True, "fp32", rows=TABLE_ROWS + 16)
    spec = dict(
        head_size=192,
        width=192,
        rotary_dim=64,
        theta=1e7,
        llama3=True,
        hq=16,
        hk=1,
        tokens=4,
        positions="prefill",
        layout="split",
        table="fp32",
    )
    case = _Case(spec, 13)
    case.table = owner[:TABLE_ROWS]
    case.positions = torch.tensor(
        [0, TABLE_ROWS - 1, TABLE_ROWS, TABLE_ROWS + 4], device=DEV, dtype=torch.int32
    )
    ref_q, ref_k = op.reference(case.positions, case.query0, case.key0, 192, owner, True)
    op(*case.args())
    torch.cuda.synchronize()
    assert (
        _measure(case.query, ref_q, _product_scale(case.positions, case.query0, 192, owner))[0]
        <= 1.0
    )
    with pytest.raises(AssertionError, match="outside the table"):
        with validating(op):
            op(
                case.positions,
                case.query0.clone(),
                case.key0.clone(),
                192,
                owner[:TABLE_ROWS],
                True,
            )


def test_guard_refuses_fewer_positions_than_tokens():
    """`positions` is a 3-element view of a 5-element buffer the test owns: the
    last two tokens are rotated by the buffer's next entries, with no error."""
    buffer = torch.tensor([3, 4, 5, 7000, 9000], device=DEV, dtype=torch.int32)
    spec = dict(
        head_size=192,
        width=192,
        rotary_dim=64,
        theta=1e7,
        llama3=True,
        hq=16,
        hk=1,
        tokens=5,
        positions="prefill",
        layout="split",
        table="fp32",
    )
    case = _Case(spec, 14)
    ref_q, _ = op.reference(buffer, case.query0, case.key0, 192, case.table, True)
    op(buffer[:3], case.query, case.key, 192, case.table, True)
    torch.cuda.synchronize()
    assert (
        _measure(case.query, ref_q, _product_scale(buffer, case.query0, 192, case.table))[0] <= 1.0
    )
    with pytest.raises(AssertionError, match="positions for"):
        with validating(op):
            op(buffer[:3], case.query0.clone(), case.key0.clone(), 192, case.table, True)


def test_guard_refuses_a_key_dtype_other_than_the_query():
    """An fp16 key under a bf16 query is rotated as bf16 bits, with no error."""
    spec = dict(
        head_size=192,
        width=192,
        rotary_dim=64,
        theta=1e7,
        llama3=True,
        hq=16,
        hk=1,
        tokens=8,
        positions="decode",
        layout="split",
        table="fp32",
    )
    case = _Case(spec, 15)
    key16 = case.key0.half()
    _, ref_k = op.reference(case.positions, case.query0, key16, 192, case.table, True)
    out = key16.clone()
    op(case.positions, case.query, out, 192, case.table, True)
    torch.cuda.synchronize()
    err = (out.double() - ref_k.double()).abs().max().item()
    print(f"\n  fp16 key under a bf16 query: max abs error {err:.3g}")
    assert err > 0.5
    with pytest.raises(AssertionError, match="key dtype"):
        with validating(op):
            op(case.positions, case.query0.clone(), key16.clone(), 192, case.table, True)
