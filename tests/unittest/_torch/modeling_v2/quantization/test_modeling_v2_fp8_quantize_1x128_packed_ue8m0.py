# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the fp8_quantize_1x128_packed_ue8m0 catalog entry.

The entry owns the contract. This file turns a cell into an activation,
gates the op against the entry's reference bit for bit -- the codes, and the scale words
read through the scale's own strides -- and holds what a cell cannot carry: the layout the
GEMM reads, row independence up to max_num_tokens (32768), every exponent boundary, the
edge inputs, the loud refusals the entry's `note` records, and K 768's half-empty last
scale word written over a dirty allocation.
"""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog.quantization.fp8_quantize_1x128_packed_ue8m0 import (  # noqa: E501
    E4M3_MAX,
    UE8M0_BIAS,
    block_exponents,
    scale_leading_dim,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.quantization.fp8_quantize_1x128_packed_ue8m0 import (  # noqa: E501
    fp8_quantize_1x128_packed_ue8m0 as op,
)

__extra_import_path__ = [".."]
from _validating import certifying, validating  # noqa: E402,F401 — needs the path declared above

assert torch.cuda.is_available(), "fp8_quantize_1x128_packed_ue8m0 requires a CUDA device"

DEV = "cuda"

#: max_num_tokens: the largest eager step the NVFP4 target takes (chunked prefill off).
MAX_TOKENS = 32768

#: The values planted on a block's exponent boundary: 7 * 2**sh is exactly 448 * 2**(sh - 6),
#: so amax / 448 is a power of two there; the other two are the bf16 neighbours of 7.
_ON_BOUNDARY, _ABOVE, _BELOW = 7.0, 7.03125, 6.96875


def _activation(m: int, k: int, seed: int) -> torch.Tensor:
    """A bf16 `[m, k]` activation with about 60 exponents and planted exponent boundaries.

    Rows of N(0, 1), each 128-block scaled by its own 2**sh, sh in [-30, 30]. One element
    of every block but each fourth is planted on the block's exponent boundary (exactly,
    one bf16 step above, one below): where amax / 448 is a power of two, a scale that
    rounds up from one ulp high takes the next exponent, and these blocks are where it
    would show.
    """
    g = torch.Generator(device=DEV).manual_seed(seed)
    nb = k // 128
    x = torch.randn(m, nb, 128, device=DEV, generator=g).to(torch.bfloat16).float()
    kind = torch.arange(m * nb, device=DEV).reshape(m, nb) % 4
    sign = torch.randint(0, 2, (m, nb), device=DEV, generator=g).float() * 2 - 1
    pos = torch.randint(0, 128, (m, nb, 1), device=DEV, generator=g)
    plant = torch.tensor([0.0, _ON_BOUNDARY, _ABOVE, _BELOW], device=DEV)[kind] * sign
    old = x.gather(2, pos).squeeze(2)
    x.scatter_(2, pos, torch.where(kind > 0, plant, old).unsqueeze(2))
    sh = torch.randint(-30, 31, (m, nb, 1), device=DEV, generator=g).float()
    return (x * torch.exp2(sh)).reshape(m, k).to(torch.bfloat16)


def _bytes(scale: torch.Tensor, nb: int) -> torch.Tensor:
    """The UE8M0 bytes `[M, nb]` of a packed scale, read through its strides."""
    return scale.contiguous().view(torch.uint8).reshape(scale.shape[0], -1)[:, :nb].long()


def _compare(out: tuple[torch.Tensor, torch.Tensor], ref: tuple[torch.Tensor, torch.Tensor]):
    """Bit for bit, with no band to derive.

    The op and the reference divide by the same power of two, an exact operation, and
    round the quotient once to e4m3; the exponent is an integer. A difference is therefore
    a different exponent or a different rounding, never accumulation order. Codes compare
    as bytes (a NaN code included); the scale compares word for word through both
    tensors' strides, which is every exponent byte.
    """
    codes, scale = out
    ref_codes, ref_scale = ref
    assert codes.dtype == ref_codes.dtype == torch.float8_e4m3fn, (codes.dtype, ref_codes.dtype)
    assert codes.shape == ref_codes.shape and codes.is_contiguous(), (codes.shape, codes.stride())
    assert scale.dtype == ref_scale.dtype == torch.int32, (scale.dtype, ref_scale.dtype)
    assert scale.shape == ref_scale.shape, (scale.shape, ref_scale.shape)
    assert scale.stride() == ref_scale.stride(), (scale.stride(), ref_scale.stride())
    bad = codes.view(torch.uint8) != ref_codes.view(torch.uint8)
    assert not bool(bad.any()), (
        f"{int(bad.sum())} of {bad.numel()} codes differ, first at {bad.nonzero()[0].tolist()}"
    )
    nb = (codes.shape[1] + 127) // 128
    got, want = _bytes(scale, nb), _bytes(ref_scale, nb)
    wrong = got != want
    assert not bool(wrong.any()), (
        f"{int(wrong.sum())} exponent bytes differ, first at {wrong.nonzero()[0].tolist()}: "
        f"{got[wrong][0].item()} vs {want[wrong][0].item()}"
    )
    assert torch.equal(scale, ref_scale), "a byte slot past the last block is not zero"


def _boundary_blocks(x: torch.Tensor) -> int:
    """How many blocks have amax / 448 exactly a power of two."""
    m, k = x.shape
    amax = x.float().abs().reshape(m, k // 128, 128).amax(dim=2).double() / E4M3_MAX
    return int((torch.frexp(amax)[0] == 0.5).sum())


# ── the certified cells ───────────────────────────────────────────────────


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying):  # noqa: F811 — pytest resolves the fixture by this name
    certifying(op)
    m, k = cell.spec["m"], cell.spec["k"]
    seed = zlib.crc32(cell.why.encode())
    x = _activation(m, k, seed)
    if not cell.spec["graph"]:
        out = op(x)
        _compare(out, op.reference(x))
    else:
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            op(x)
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = op(x)
        for step in (1, 2):
            x.copy_(_activation(m, k, seed + step))
            graph.replay()
            torch.cuda.synchronize()
            _compare(out, op.reference(x))
            eager = op(x)
            assert torch.equal(out[0].view(torch.uint8), eager[0].view(torch.uint8))
            assert torch.equal(out[1], eager[1]) and out[1].stride() == eager[1].stride()
    exps = block_exponents(x)
    on = _boundary_blocks(x)
    assert on > 0, "the cell's input never reaches an exponent boundary"
    print(f"  {cell.why[:44]}: {exps.unique().numel()} exponents, {on} blocks on a boundary")


# ── the layout the GEMM reads ─────────────────────────────────────────────


@pytest.mark.parametrize("k", [2048, 8192, 24576, 1024, 3072, 768])
@pytest.mark.parametrize("m", [1, 2, 3, 4, 5, 6, 7, 16, 17, 9215, 9216, 16385, 32767, 32768])
def test_scale_layout_strides_storage_and_padding(m, k):
    """The scale is MN-major int32 with column stride align(M, 4), read through its strides.

    Its storage ends at the last real row, so only a strided read is in bounds; the pad
    slots inside it are zero; the codes view a 4-row-padded allocation. K 8192 and 24576
    are the 16- and 48-column scales dep4's wo and dense layer-0 down read; K 1024, 3072
    and 768 the 2-, 6- and 2-column scales of the tep8 and tp8 wo, dense layer-0 down and
    shared-expert down (768's second column holds two blocks).
    """
    codes, scale = op(_activation(m, k, seed=m))
    lead, cols = scale_leading_dim(m), -(-k // 512)
    assert scale.shape == (m, cols) and scale.stride() == (1, lead), (scale.shape, scale.stride())
    words = scale.untyped_storage().nbytes() // 4
    assert words == (cols - 1) * lead + m, words
    flat = torch.as_strided(scale, (words,), (1,), 0)
    pads = [flat[c * lead + r].item() for c in range(cols - 1) for r in range(m, lead)]
    assert all(p == 0 for p in pads), pads
    if m % 4:
        with pytest.raises(RuntimeError, match="out of bounds for storage"):
            torch.as_strided(scale, (cols * lead,), (1,), 0)
    assert codes.stride() == (k, 1) and codes.untyped_storage().nbytes() == lead * k


@pytest.mark.parametrize("k", [6144, 2048, 1536, 8192, 24576, 1024, 3072, 768])
def test_every_eager_row_count_is_a_prefix_of_the_largest(k):
    """Rows are independent, so the certified max_num_tokens covers every eager row count.

    Each M up to max_num_tokens returns, bit for bit, the first M rows of the 32768-row call.
    """
    x = _activation(MAX_TOKENS, k, seed=k)
    full_codes, full_scale = op(x)
    _compare((full_codes, full_scale), op.reference(x))
    rows = (1, 2, 3, 5, 7, 8, 9, 15, 16, 17, 31, 33, 127, 129, 1023, 1025, 4095, 4097, 8191)
    for m in rows + (9215, 9216, 9217, 16383, 16384, 16385, 24577, 32765, 32766, 32767):
        codes, scale = op(x[:m])
        assert torch.equal(codes.view(torch.uint8), full_codes[:m].view(torch.uint8)), m
        assert torch.equal(scale, full_scale[:m]), m


@pytest.mark.parametrize("m", [1, 5, 4097, 32767])
def test_k768_empty_byte_slots_are_written_over_a_dirty_allocation(m):
    """K 768 is six blocks in two scale words: the op writes 0 into bytes 2 and 3 of the second.

    The op's two allocations (the 4-row-padded codes, then the `[2, align(M, 4)]` int32
    scale) are first made, filled with 0xFF and freed in the same sizes and order, so the
    caching allocator hands the op those blocks back. The control: the code rows past M,
    which the kernel never writes, still read 0xFF afterwards. So the empty byte slots and
    the column-0 pad slots read 0 because the kernel wrote them, not because the memory was
    fresh.
    """
    k = 768
    x = _activation(m, k, seed=m)
    lead = scale_leading_dim(m)
    codes_block = torch.full((lead * k,), 0xFF, dtype=torch.uint8, device=DEV)
    scale_block = torch.full((2 * lead * 4,), 0xFF, dtype=torch.uint8, device=DEV)
    where = (codes_block.data_ptr(), scale_block.data_ptr())
    del scale_block, codes_block
    codes, scale = op(x)
    assert (codes.data_ptr(), scale.data_ptr()) == where, "the op's outputs left the 0xFF blocks"
    unwritten = torch.as_strided(codes.view(torch.uint8), (lead * k,), (1,), 0)[m * k :]
    assert unwritten.numel() == (lead - m) * k > 0 and bool((unwritten == 0xFF).all())
    _compare((codes, scale), op.reference(x))
    slots = scale.contiguous().view(torch.uint8).reshape(m, 8)
    assert not bool(slots[:, 6:].any()), "an empty byte slot of the second word is not 0"
    assert bool(slots[:, :6].ne(0).all()), "a live block's byte is 0"
    flat = torch.as_strided(scale, (lead + m,), (1,), 0)
    assert not bool(flat[m:lead].any()), "a pad slot of the first column is not 0"


# ── the exponent ──────────────────────────────────────────────────────────


def test_every_exponent_boundary():
    """Every exponent from the floor's 2**-42 to the largest a bf16 needs, 2**119.

    Each on its boundary amax = 448 * 2**e (byte e + 127, the block max coded 448), one
    bf16 step above (byte e + 128, coded 224) and one below (byte e + 127, coded 448 after
    rounding).
    """
    exps = torch.arange(-42, 120, device=DEV)
    nb = exps.numel()
    on = torch.ldexp(torch.full((nb,), 7.0, device=DEV), exps + 6).to(torch.bfloat16)
    above = (on.view(torch.int16) + 1).view(torch.bfloat16)
    below = (on.view(torch.int16) - 1).view(torch.bfloat16)
    rows = torch.stack([on, above, below])
    x = torch.zeros(3, nb, 128, dtype=torch.bfloat16, device=DEV)
    x[:, :, 7] = rows
    x[:, :, 100] = -rows * 0.5
    x = x.reshape(3, nb * 128)
    codes, scale = op(x)
    _compare((codes, scale), op.reference(x))
    got = _bytes(scale, nb)
    assert torch.equal(got[0], exps + UE8M0_BIAS)
    assert torch.equal(got[1], exps + UE8M0_BIAS + 1)
    assert torch.equal(got[2], exps + UE8M0_BIAS)
    lead = codes.reshape(3, nb, 128)[:, :, 7].float()
    assert bool((lead[0] == 448).all() and (lead[1] == 224).all() and (lead[2] == 448).all())


def test_edge_blocks():
    """One block per edge input, pinned to its measured bytes and codes.

    Each also equals the reference: zero, NaN, an all-NaN block, +-inf, -0.0, the largest
    bf16, and values under the amax floor.
    """
    blocks = torch.zeros(9, 128, dtype=torch.bfloat16, device=DEV)
    blocks[1, 0], blocks[1, 1], blocks[1, 2] = float("nan"), 1.0, 3.0
    blocks[2] = float("nan")
    blocks[3, 0], blocks[3, 1], blocks[3, 2] = float("inf"), 1.0, -2.0
    blocks[4, 0], blocks[4, 1] = float("-inf"), 1.0
    blocks[5, 0], blocks[5, 1] = -0.0, 1.0
    blocks[6, 0], blocks[6, 1] = 3.3895313892515355e38, 1.0
    blocks[7, 0], blocks[7, 1] = 1e-12, 5e-13
    blocks[8, 0], blocks[8, 1] = 9.2e-41, -1e-39
    x = blocks.reshape(1, -1).contiguous()
    codes, scale = op(x)
    _compare((codes, scale), op.reference(x))
    b = _bytes(scale, 9)[0].tolist()
    c = codes.view(torch.uint8).reshape(9, 128)[:, :3].tolist()
    assert b == [85, 120, 85, 254, 254, 119, 247, 85, 85], b
    assert c[0] == [0x00, 0x00, 0x00], c[0]  # zero block, scale 2**-42
    assert c[1] == [0x7F, 0x70, 0x7C], c[1]  # NaN in place; 1.0 and 3.0 at 2**-7
    assert c[2] == [0x7F, 0x7F, 0x7F], c[2]  # all NaN: floor scale, NaN codes
    assert c[3] == [0x7E, 0x00, 0x80], c[3]  # inf -> 448, finite -> +-0
    assert c[4][:2] == [0xFE, 0x00], c[4]  # -inf -> -448
    assert c[5][:2] == [0x80, 0x78], c[5]  # -0.0 keeps its sign
    assert c[6][:2] == [0x78, 0x00], c[6]  # bf16 max / 2**120 = 255 -> 256
    assert c[7][:2] == [0x49, 0x41], c[7]  # under the floor: 1e-12 * 2**42 -> 4.5
    assert c[8][:2] == [0x00, 0x80], c[8]  # bf16 subnormals -> +-0


def test_discrimination_against_wrong_readings():
    """Each plausible wrong reading of the recipe or the layout gets some bytes wrong.

    Scored against one activation as the fraction of codes or exponent bytes it gets
    wrong. The bitwise gate rejects any nonzero fraction; these show the cells' inputs
    make each one nonzero.
    """
    m, k = 64, 6144
    x = _activation(m, k, seed=7)
    codes, scale = op(x)
    nb = k // 128
    got = _bytes(scale, nb)
    amax = x.float().abs().reshape(m, nb, 128).amax(dim=2).double().clamp(min=1e-10) / E4M3_MAX
    exact = block_exponents(x)

    def byte_miss(e):
        return (e + UE8M0_BIAS != got).float().mean().item()

    def code_miss(s):
        q = (x.double() / s.repeat_interleave(128, dim=1)).float().clamp(-E4M3_MAX, E4M3_MAX)
        wrong = q.to(torch.float8_e4m3fn).view(torch.uint8) != codes.view(torch.uint8)
        return wrong.float().mean().item()

    def word_bytes(words):
        return words.contiguous().view(torch.uint8).reshape(m, -1).long()

    big_endian = word_bytes(scale).reshape(m, -1, 4).flip(-1).reshape(m, -1)
    dense_rows = word_bytes(torch.as_strided(scale, (m, k // 512), (k // 512, 1)))
    rows = {
        "exact recipe (codes)": code_miss(torch.ldexp(torch.ones_like(amax), exact)),
        "exact recipe (bytes)": byte_miss(exact),
        "floor(log2) exponent": byte_miss(torch.floor(torch.log2(amax)).long()),
        "nearest exponent": byte_miss(torch.round(torch.log2(amax)).long()),
        "exponent one high on boundaries": byte_miss(
            torch.where(torch.frexp(amax)[0] == 0.5, exact + 1, exact)
        ),
        "plain amax / 448 scale (codes)": code_miss(amax),
        "bytes big-endian in the word": (big_endian != got).float().mean().item(),
        "scale storage read as dense rows": (dense_rows != got).float().mean().item(),
    }
    for name, miss in rows.items():
        print(f"  {name:34s} wrong {miss:.4f}")
    assert rows.pop("exact recipe (codes)") == 0.0 and rows.pop("exact recipe (bytes)") == 0.0
    for name, miss in rows.items():
        assert miss > 0.0, name


# ── behaviour around the call ─────────────────────────────────────────────


def test_registered_fake_matches_eager():
    from torch._subclasses.fake_tensor import FakeTensorMode

    shapes = ((1, 6144), (3, 2048), (17, 1536), (9216, 6144), (32767, 1536), (32768, 6144))
    dep4 = ((5, 8192), (32768, 8192), (3, 24576), (32767, 24576))
    tep8 = ((1, 1024), (32767, 1024), (5, 3072), (32768, 3072), (3, 768), (32768, 768))
    for m, k in shapes + dep4 + tep8:
        e_codes, e_scale = op(torch.zeros(m, k, dtype=torch.bfloat16, device=DEV))
        eager = [(tuple(t.shape), t.stride(), t.dtype) for t in (e_codes, e_scale)]
        with FakeTensorMode():
            f_codes, f_scale = op(torch.empty(m, k, dtype=torch.bfloat16, device=DEV))
            fake = [(tuple(t.shape), t.stride(), t.dtype) for t in (f_codes, f_scale)]
        assert fake == eager, (m, k, fake, eager)


def test_input_untouched_deterministic_and_stream_independent():
    x = _activation(129, 6144, seed=3)
    before = x.clone()
    c1, s1 = op(x)
    c2, s2 = op(x)
    alt = torch.cuda.Stream()
    alt.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(alt):
        c3, s3 = op(x)
    alt.synchronize()
    assert torch.equal(x, before)
    for c, s in ((c2, s2), (c3, s3)):
        assert torch.equal(c.view(torch.uint8), c1.view(torch.uint8)) and torch.equal(s, s1)


def test_refusals_are_the_ops_own():
    """Every precondition the note lists is enforced loudly by the op.

    That is why the entry has no `is_valid`: restating one would only swap its error for a
    worse one.
    """
    bf = dict(dtype=torch.bfloat16, device=DEV)
    cases = [
        ("Input matrix dtype must be BF16", lambda: op(torch.randn(4, 128, device=DEV))),
        (
            "Input matrix dtype must be BF16",
            lambda: op(torch.randn(4, 128, dtype=torch.float16, device=DEV)),
        ),
        ("input must be a matrix", lambda: op(torch.randn(2, 4, 128, **bf))),
        ("must be a multiple of 16, but got 6152", lambda: op(torch.randn(4, 6152, **bf))),
        ("self must be contiguous", lambda: op(torch.randn(8, 12288, **bf)[:, :6144])),
        ("not registered with any CUDA device", lambda: op(torch.empty(0, 6144, **bf))),
    ]
    for message, call in cases:
        with pytest.raises(RuntimeError, match=message):
            call()
    with pytest.raises(NotImplementedError):
        op(torch.randn(4, 128, dtype=torch.bfloat16))
