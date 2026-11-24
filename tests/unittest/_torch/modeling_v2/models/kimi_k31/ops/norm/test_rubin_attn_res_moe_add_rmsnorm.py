# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the K3.1 Rubin entries rubin_attn_res_moe_add_rmsnorm and its ``_mx`` twin.

Every cell: the deferred MoE residual update bit for bit against the reference's two bf16 adds,
the output against the fp32 reference, bit-identical repeats and untouched inputs; the ``_mx``
entry returns the plain entry's bf16 bits plus the MXFP8 form of its output, bit for bit (atom
bytes past the rows are not written, so not compared); with ``mx_exact`` that form is the catalog
``mxfp8_quantize``'s codes and swizzled scales of the output, compared with that quantizer itself,
also on output groups in the subnormal-scale band (``amax * rcp(448)`` in (2^-127, 2^-126)), at
infinity and at NaN (a group of NaN, NaN in half a group, one NaN), and on the quantizer's NaN rule
by parity (every odd value a NaN: a NaN maximum; every even one: the odd maximum; one number among
NaNs at an even and at an odd index). Every ``_mx`` cell's output has an all-zero group and groups
whose amax lies below, across and above where the two rules part (the low-M GEMM floors the amax
at ~1e-4, so below ~5.35e-5 its scale is the larger one).

helix8's KDA hand-off (``attn_res_mx_pdl``) on the device: at the verify rows of b1..b21 (up to one
128-row tile) the one-tile form, in the rule of the GEMM ``rubin_mxfp8_linear`` runs for the fused
KDA projection (``[6288, 7168]``), fed to ``rubin_mxfp8_linear_prequantized`` writes the bits
``rubin_mxfp8_linear`` writes on the form's BF16 rows, eagerly and replayed in a CUDA graph on new
inputs; the other rule's scales differ on the tiny groups.
Needs the kernel at its family path (``kernels/norm/rubin_attn_res.py``).
"""

import zlib

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7),
    reason="sm_107 kernel (VR200)",
)

from tensorrt_llm._torch._experimental.modeling_v2.catalog.quantization.mxfp8_quantize import (  # noqa: E402,E501
    mxfp8_quantize,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.gemm.rubin_mxfp8_gemm import (  # noqa: E402,E501
    mxfp8_scales_to_r128c4,
    quantize_mxfp8_reference,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.gemm.rubin_mxfp8_linear import (  # noqa: E402,E501
    mxfp8_act_layout,
    rubin_mxfp8_linear,
    rubin_mxfp8_linear_prequantized,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.norm._rubin_norm_common import (  # noqa: E402,E501
    mxfp8_exact_reference,
    mxfp8_reference,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.norm.rubin_attn_res_moe_add_rmsnorm import (  # noqa: E402,E501
    rubin_attn_res_moe_add_rmsnorm,
    rubin_attn_res_moe_add_rmsnorm_mx,
)

__extra_import_path__ = ["../../../.."]
from _validating import certifying  # noqa: E402,F401 — path above; fixture by name

H = 7168
EPS = 1e-5
# As test_rubin_attn_res_rmsnorm: the kernel rounds where the reference rounds.
MIN_COSINE, MAX_RELATIVE_L2 = 0.9999, 1e-3


def _inputs(rows: int, candidates: int, seed: int):
    g = torch.Generator(device="cuda").manual_seed(seed)

    def rn(*shape, sc=1.0):
        return (torch.randn(*shape, generator=g, device="cuda") * sc).to(torch.bfloat16)

    residual = rn(rows, H)
    residual[0, :64] = 0  # an all-zero 1x32 group of the update (the amax floor)
    return (
        residual,
        rn(rows, H, sc=0.5),  # routed
        rn(rows, H, sc=0.5),  # shared
        rn(candidates - 1, rows, H),  # block
        rn(H, sc=0.05),  # res_weight
        (1.0 + rn(H, sc=0.1).float()).to(torch.bfloat16),  # rms_weight
        (1.0 + rn(H, sc=0.1).float()).to(torch.bfloat16),  # out_weight
        EPS,
        EPS,
    )


def _tiny_out_weight(w: torch.Tensor) -> torch.Tensor:
    """``out_weight`` with output 1x32 groups near the MXFP8 rules' floor: one all zero, then
    |normed| (~1-3) times 1e-6 .. 5e-5, so group amaxes fall below ~5.35e-5 (where the low-M
    GEMM's floor at ~1e-4 picks a larger scale than the catalog quantizer), across it and across
    the floor itself (where both rules pick the same scale)."""
    w = w.clone()
    w[64:96] = 0
    for i, c in enumerate((1e-6, 1e-5, 2e-5, 2.5e-5, 4e-5, 5e-5)):
        w[96 + 32 * i : 128 + 32 * i] = c
    return w


def _assert_tiny_groups(out: torch.Tensor, near_floor: bool = False) -> None:
    amax = out.float().abs().view(out.shape[0], -1, 32).amax(-1)
    assert bool((amax == 0).any()), "no all-zero group"
    assert bool(((amax > 0) & (amax < 2.0**-15)).any()), "no group where the two rules part"
    if near_floor:
        between = (amax > 5.4e-5) & (amax < 1e-4)
        assert bool(between.any()), "no group between the rules' split and the floor"


def _atom_offsets(rows: int) -> torch.Tensor:
    """Byte offsets of the M rows' scales in one 128-row block of atoms (R128c4 of one tile)."""
    row = torch.arange(rows, device="cuda").view(rows, 1)
    grp = torch.arange(H // 32, device="cuda").view(1, -1)
    off = (grp // 4) * 512 + (row % 32) * 16 + (row % 128 // 32) * 4 + grp % 4
    return off.reshape(-1)


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.view(torch.uint8) if t.element_size() == 1 else t.view(torch.int16)


def _assert_reference_close(out: torch.Tensor, ref: torch.Tensor) -> None:
    a, e = out.float().flatten(), ref.float().flatten()
    cosine = torch.nn.functional.cosine_similarity(a, e, dim=0).item()
    rel = ((a - e).norm() / (e.norm() + 1e-12)).item()
    assert cosine > MIN_COSINE and rel < MAX_RELATIVE_L2, f"cos {cosine:.7f} rel {rel:.2e}"


def _seed(cell) -> int:
    return zlib.crc32(cell.why.encode()) % 2**31


@pytest.mark.parametrize(
    "cell",
    rubin_attn_res_moe_add_rmsnorm.CELLS,
    ids=[c.why[:56] for c in rubin_attn_res_moe_add_rmsnorm.CELLS],
)
def test_moe_add_cells(cell, certifying) -> None:  # noqa: F811 — fixture resolved by name
    certifying(rubin_attn_res_moe_add_rmsnorm)
    spec = cell.spec
    args = _inputs(spec["rows"], spec["candidates"], _seed(cell))
    before = [x.clone() for x in args[:7]]
    prefix, out = rubin_attn_res_moe_add_rmsnorm(*args, early_trigger=spec["early_trigger"])
    prefix2, out2 = rubin_attn_res_moe_add_rmsnorm(*args, early_trigger=spec["early_trigger"])
    ref_prefix, ref_out = rubin_attn_res_moe_add_rmsnorm.reference(
        *args, early_trigger=spec["early_trigger"]
    )
    assert torch.equal(_bits(prefix), _bits(prefix2)) and torch.equal(_bits(out), _bits(out2))
    assert torch.equal(_bits(prefix), _bits(ref_prefix)), "the update is not two bf16 adds"
    residual, routed, shared = args[:3]
    assert torch.equal(_bits(prefix), _bits(residual + (routed + shared)))
    _assert_reference_close(out, ref_out)
    for old, new in zip(before, args[:7]):
        assert torch.equal(_bits(old), _bits(new)), "an input was mutated"


@pytest.mark.parametrize(
    "cell",
    rubin_attn_res_moe_add_rmsnorm_mx.CELLS,
    ids=[c.why[:56] for c in rubin_attn_res_moe_add_rmsnorm_mx.CELLS],
)
def test_moe_add_mx_cells(cell, certifying) -> None:  # noqa: F811 — fixture resolved by name
    certifying(rubin_attn_res_moe_add_rmsnorm_mx)
    spec = cell.spec
    rows = spec["rows"]
    args = _inputs(rows, spec["candidates"], _seed(cell))
    blocks = spec.get("mx_atom_blocks", 0)
    exact = spec.get("mx_exact", False)
    mx_args = (spec.get("entry_trigger", False), blocks > 0, max(blocks, 1))
    kw = dict(early_trigger=spec["early_trigger"], mx_exact=exact)
    # Groups the floor of the low-M quantization changes (or would change, with mx_exact).
    args = (*args[:6], _tiny_out_weight(args[6]), *args[7:])
    got = rubin_attn_res_moe_add_rmsnorm_mx(*args, *mx_args, **kw)
    plain = rubin_attn_res_moe_add_rmsnorm(*args, early_trigger=spec["early_trigger"])
    ref = rubin_attn_res_moe_add_rmsnorm_mx.reference(*args, *mx_args, **kw)
    for i in range(2):
        assert torch.equal(_bits(got[i]), _bits(plain[i])), f"bf16 output {i}"
    _assert_reference_close(got[1], ref[1])
    _assert_tiny_groups(got[1])
    # The codes and scales are a function of the kernel's own output.
    form = mxfp8_exact_reference if exact else mxfp8_reference
    xq, xs = form(got[1], blocks > 0, max(blocks, 1))
    assert torch.equal(_bits(got[2]), _bits(xq)), "codes"
    if exact:
        # The quantizer the dense MXFP8 GEMM's caller runs today: the same codes, and the same
        # swizzled scale bytes for the M rows (it also zeroes the tile's rows past M).
        cq, cs = mxfp8_quantize(got[1], True, 32)
        assert torch.equal(_bits(got[2]), _bits(cq)), "codes against mxfp8_quantize"
        assert blocks == 1 and cs.numel() == got[3].numel()
    if blocks:
        # Only the atom bytes of the M rows are written.
        row = torch.arange(rows, device="cuda").view(rows, 1)
        grp = torch.arange(H // 32, device="cuda").view(1, -1)
        off = ((grp // 4) * blocks + row // 128) * 512 + (row % 32) * 16 + (row % 128 // 32) * 4
        off = (off + grp % 4).reshape(-1)
        assert tuple(got[3].shape) == ((H // 128) * blocks * 512,)
        assert torch.equal(got[3][off], xs[off]), "atom scales"
        if exact:
            assert torch.equal(got[3][off], cs[off]), "scales against mxfp8_quantize"
    else:
        assert tuple(got[3].shape) == (rows, H // 32) and torch.equal(got[3], xs), "scales"


def _edge_out_weight(w: torch.Tensor) -> torch.Tensor:
    """``out_weight`` putting output groups in the subnormal-scale band, at infinity and at NaN."""
    w = w.clone()
    w[128:160] = 1.6e-36  # |normed| ~ 1.5-3: amax * rcp(448) in (2^-127, 2^-126)
    w[160:192] = 1.0e-36
    w[192:224] = 2.2e-36
    w[224:256] = 4.0e-37
    w[256:288] = float("inf")  # a group of infinities
    w[300] = float("inf")  # one infinity in a group
    w[320:352] = float("nan")  # a group of NaN
    w[352:384:2] = float("nan")  # NaN in half a group (the even values): the odd maximum
    w[389] = float("nan")  # one NaN in a group
    w[417:448:2] = float("nan")  # every odd value a NaN: a NaN maximum, byte 255
    w[448:480] = torch.where(torch.arange(32, device=w.device) == 6, w[448:480], float("nan"))
    w[480:512] = torch.where(torch.arange(32, device=w.device) == 5, w[480:512], float("nan"))
    return w  # 448-479: one number at an even index among NaNs; 480-511: at an odd index


_EXACT_CELLS = [c for c in rubin_attn_res_moe_add_rmsnorm_mx.CELLS if c.spec.get("mx_exact", False)]


@pytest.mark.parametrize("cell", _EXACT_CELLS, ids=[c.why[:56] for c in _EXACT_CELLS])
def test_moe_add_mx_exact_edge_groups(cell) -> None:
    spec = cell.spec
    rows = spec["rows"]
    args = _inputs(rows, spec["candidates"], _seed(cell) + 1)
    args = (*args[:6], _edge_out_weight(args[6]), *args[7:])
    blocks = spec.get("mx_atom_blocks", 0)
    assert blocks == 1, "the mx_exact cells write the swizzled scales of one 128-row block"
    mx_args = (spec.get("entry_trigger", False), True, 1)
    kw = dict(early_trigger=spec["early_trigger"], mx_exact=True)
    got = rubin_attn_res_moe_add_rmsnorm_mx(*args, *mx_args, **kw)
    plain = rubin_attn_res_moe_add_rmsnorm(*args, early_trigger=spec["early_trigger"])
    for i in range(2):
        assert torch.equal(_bits(got[i]), _bits(plain[i])), f"bf16 output {i}"
    out = got[1]
    sf = out.float().abs().view(rows, -1, 32).amax(-1) * (1.0 / 448.0)
    assert int(((sf > 2.0**-127) & (sf < 2.0**-126)).sum()) > 0, "no group in the band"
    assert bool(out.isinf().any()) and bool(out.isnan().any())
    g = out.view(rows, -1, 32).isnan()
    odd_nan = g[..., 1::2].all(-1) & ~g[..., 0::2].all(-1)  # vecMax NaN, finite evens
    even_nan = g[..., 0::2].all(-1) & ~g[..., 1::2].all(-1)  # vecMax = the odd maximum
    assert bool(odd_nan.any()) and bool(even_nan.any()), "no group of the NaN rule by parity"
    cq, cs = mxfp8_quantize(out, True, 32)
    xq, xs = mxfp8_exact_reference(out, True, 1)
    assert torch.equal(_bits(got[2]), _bits(cq)), "codes against mxfp8_quantize"
    assert torch.equal(_bits(got[2]), _bits(xq)), "codes against the reference"
    row = torch.arange(rows, device="cuda").view(rows, 1)
    grp = torch.arange(H // 32, device="cuda").view(1, -1)
    off = (grp // 4) * 512 + (row % 32) * 16 + (row % 128 // 32) * 4 + grp % 4
    off = off.reshape(-1)
    assert torch.equal(got[3][off], cs.view(torch.uint8)[off]), "scales against mxfp8_quantize"
    assert torch.equal(got[3][off], xs[off]), "scales against the reference"


# helix8's fused KDA projection at TP8 ([q | k | v | g | f_a | b], 6284 rows padded to 16), and the
# verify rows of its decode steps under attn_res_mx_pdl (b x 6 at k5, up to one 128-row tile) on
# both sides of the shared GEMM's low-M rows (in-kernel quantization) and of its mid-M token tiles.
KDA_N = 6288
KDA_ROWS = (6, 12, 18, 24, 30, 32, 36, 54, 60, 64, 66, 96, 126, 128)


@pytest.fixture(scope="module")
def kda_weight() -> tuple[torch.Tensor, torch.Tensor]:
    """An MXFP8 ``[KDA_N, H]`` weight (E4M3 values, flat R128c4 1x32 scales) of row norm ~1."""
    g = torch.Generator(device="cuda").manual_seed(31)
    w = (torch.randn(KDA_N, H, generator=g, device="cuda") * H**-0.5).to(torch.bfloat16)
    values, scales = quantize_mxfp8_reference(w)
    return values, mxfp8_scales_to_r128c4(scales)


def _kda_inputs(rows: int, seed: int) -> tuple:
    args = _inputs(rows, 5, seed)
    return (*args[:6], _tiny_out_weight(args[6]), *args[7:])


def _kda_mx(args: tuple, exact: bool) -> tuple:
    """helix8's call: one 128-row block of atoms, early trigger, no entry trigger."""
    return rubin_attn_res_moe_add_rmsnorm_mx(
        *args, False, True, 1, early_trigger=True, mx_exact=exact
    )


def _kda_rule(rows: int) -> bool:
    """``mx_exact`` of the GEMM ``rubin_mxfp8_linear`` runs for ``rows`` rows (helix8's
    ``_kda_mx_exact``): off where the shared low-M / mid-M GEMM's layout is named."""
    return mxfp8_act_layout(rows, KDA_N, H)[1] == "r128c4"


@pytest.mark.parametrize("seed", [0, 1])
@pytest.mark.parametrize("rows", KDA_ROWS)
def test_mx_form_is_the_kda_projections_own_quantization(rows, seed, kda_weight) -> None:
    w, ws = kda_weight
    exact = _kda_rule(rows)
    args = _kda_inputs(rows, 1000 * rows + seed)
    _, normed, codes, scales = _kda_mx(args, exact)
    _assert_tiny_groups(normed, near_floor=True)
    got = rubin_mxfp8_linear_prequantized(codes, scales, w, ws)
    want = rubin_mxfp8_linear(normed, w, ws)
    assert torch.equal(_bits(got), _bits(want)), "the hand-off changes the projection's output"
    # The other rule's form differs on the tiny groups: the bits above are the rule's, not luck.
    _, _, _, other = _kda_mx(args, not exact)
    off = _atom_offsets(rows)
    assert not torch.equal(scales[off], other[off]), "no group where the two rules part"


@pytest.mark.parametrize("rows", [24, 54])
def test_mx_form_hand_off_replays_in_a_graph(rows, kda_weight) -> None:
    """The mix and the prequantized projection captured together (PDL-chained, as on a decode
    step), replayed on new inputs."""
    w, ws = kda_weight
    exact = _kda_rule(rows)
    static = _kda_inputs(rows, rows)
    for _ in range(2):  # compile and warm up outside the capture
        _, _, codes, scales = _kda_mx(static, exact)
        rubin_mxfp8_linear_prequantized(codes, scales, w, ws)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _, normed, codes, scales = _kda_mx(static, exact)
        out = rubin_mxfp8_linear_prequantized(codes, scales, w, ws)
    for dst, src in zip(static[:7], _kda_inputs(rows, rows + 7)[:7]):
        dst.copy_(src)
    graph.replay()
    torch.cuda.synchronize()
    _assert_tiny_groups(normed)
    want = rubin_mxfp8_linear(normed, w, ws)
    assert torch.equal(_bits(out), _bits(want)), "the replayed hand-off changes the projection"


@pytest.mark.parametrize("mx", [False, True], ids=["bf16", "mx"])
@pytest.mark.parametrize("rows,candidates", [(12, 2), (24, 3), (54, 5), (66, 8), (126, 9)])
def test_updated_into_a_bank_row(rows, candidates, mx) -> None:
    """``updated``: the kernel stores the updated prefix into the next bank row (helix8
    ``mix_bank_in_mix``: a block boundary's snapshot) and returns that row; the same bits as a
    new tensor, the snapshot rows it reads untouched."""
    args = _inputs(rows, candidates, 1000 * rows + candidates)
    bank = torch.empty(candidates, rows, H, device="cuda", dtype=torch.bfloat16)
    bank[: candidates - 1].copy_(args[3])
    args = (*args[:3], bank[: candidates - 1], *args[4:])
    snapshots = bank[: candidates - 1].clone()
    if mx:
        mx_args = (False, True, 1)
        entry = rubin_attn_res_moe_add_rmsnorm_mx
        kw = dict(early_trigger=True, mx_exact=False)
    else:
        mx_args = ()
        entry = rubin_attn_res_moe_add_rmsnorm
        kw = dict(early_trigger=True)
    entry.is_valid(*args, *mx_args, **kw, updated=bank[candidates - 1])
    got = entry(*args, *mx_args, **kw, updated=bank[candidates - 1])
    plain = entry(*args, *mx_args, **kw)
    assert got[0].data_ptr() == bank[candidates - 1].data_ptr(), "not stored into the bank row"
    for i in range(2 + 2 * mx):
        if i == 3:  # atom bytes past the rows are not written
            continue
        assert torch.equal(_bits(got[i]), _bits(plain[i])), f"output {i}"
    assert torch.equal(_bits(bank[: candidates - 1]), _bits(snapshots)), "a snapshot changed"
    with pytest.raises(AssertionError):
        entry.is_valid(*args, *mx_args, **kw, updated=bank[0])  # a row the launch reads
