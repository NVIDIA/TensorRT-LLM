# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the GLM-5.2 target's cute_dsl_moe_nvfp4_experts entry."""

import zlib

import pytest
import torch

if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 3):
    pytest.skip("the vendored CuTe DSL grouped GEMMs are built for sm_103", allow_module_level=True)

from tensorrt_llm._torch._experimental.modeling_v2.catalog.quantization.fp4_quantize import (  # noqa: E402, E501
    fp4_quantize,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tep4.kernels.moe import (  # noqa: E402, E501
    moe_fc1_kblock,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tep4.ops.moe.cute_dsl_moe_nvfp4_experts import (  # noqa: E402, E501
    PREFILL_MAX_ROWS,
    CompiledDecodeExperts,
    CompiledPairExperts,
    allocate_decode_c,
    allocate_pair_c,
    allocate_sfc,
    build_decode,
    build_prefill_pair,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tep4.ops.moe.cute_dsl_moe_nvfp4_experts import (  # noqa: E402, E501
    cute_dsl_moe_nvfp4_experts as op,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tep4.ops.moe.cute_dsl_moe_routing import (  # noqa: E402, E501
    TOP_K,
    allocate_maps,
    reference_maps,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tep4.ops.moe.cute_dsl_moe_topk_routing_rep import (  # noqa: E402, E501
    reference_slot_maps,
)

__extra_import_path__ = ["../../../.."]
from _validating import assert_within_ulp, validating  # noqa: E402 — needs the path declared above

HIDDEN, INTER, EXPERTS, GLOBAL_EXPERTS = 6144, 2048, 64, 256
#: Context replica slots past the 64 home experts in the replica cells' stacks.
REPLICAS = 8
#: The target's capacity: the decode pair is compiled for the token budget.
MAX_ROWS = PREFILL_MAX_ROWS

_STATE: dict = {}


def _sf_atoms(s: torch.Tensor) -> torch.Tensor:
    """Row-major `[N, K/16]` scale bytes -> `[N/128, K/64, 512]` atoms, the layout the
    target's weight loader writes (restated so the test does not import the loader)."""
    n, sf_k = s.shape
    return (
        s.reshape(n // 128, 4, 32, sf_k // 4, 4)
        .permute(0, 3, 2, 1, 4)
        .contiguous()
        .reshape(n // 128, sf_k // 4, 512)
    )


def _nvfp4(w: torch.Tensor):
    """`(packed codes [R, K/2], scale bytes [R, K/16])` of a float matrix, global scale 1."""
    packed, sf = fp4_quantize.reference(w, torch.ones(1, device=w.device), 16, False, False)
    return packed, sf.view(w.shape[0], -1)


def _weights(depth: int = EXPERTS, fc1_kblk: bool = False, fc2_kblk: bool = False):
    """The first `depth` slots of one expert stack for the session (`EXPERTS + REPLICAS`
    deep), NVFP4 in the kernels' layout: a contiguous prefix, as the target's prefill pair
    reads the home slots of a stack with replica slots. Scaled so each GEMM's sums are
    O(1): the FC1 output's e4m3 scales then sit well inside e4m3's range. `fc1_kblk` /
    `fc2_kblk`: the same FC1 / FC2 weight in the k-block layout, as the loader writes it by
    default."""
    if "weights" not in _STATE:
        g = torch.Generator(device="cuda").manual_seed(20260)
        u8 = {"dtype": torch.uint8, "device": "cuda"}
        slots = EXPERTS + REPLICAS
        fc1_w = torch.empty(slots, 2 * INTER, HIDDEN // 2, **u8)
        fc1_s = torch.empty(slots, 2 * INTER // 128, HIDDEN // 64, 512, **u8)
        fc2_w = torch.empty(slots, HIDDEN, INTER // 2, **u8)
        fc2_s = torch.empty(slots, HIDDEN // 128, INTER // 64, 512, **u8)
        for e in range(slots):
            p, s = _nvfp4(torch.randn(2 * INTER, HIDDEN, generator=g, device="cuda") * 0.02)
            fc1_w[e], fc1_s[e] = p, _sf_atoms(s)
            p, s = _nvfp4(torch.randn(HIDDEN, INTER, generator=g, device="cuda") * 0.03)
            fc2_w[e], fc2_s[e] = p, _sf_atoms(s)
        e4 = torch.float8_e4m3fn
        _STATE["weights"] = (fc1_w, fc1_s.view(e4), fc2_w, fc2_s.view(e4))
    key = "weights" + ("_kblk" if fc1_kblk else "") + ("_kblk2" if fc2_kblk else "")
    if key not in _STATE:
        fc1_w, fc1_s, fc2_w, fc2_s = _STATE["weights"]
        _STATE[key] = (
            moe_fc1_kblock.to_kblock(fc1_w) if fc1_kblk else fc1_w,
            fc1_s,
            moe_fc1_kblock.to_kblock(fc2_w) if fc2_kblk else fc2_w,
            fc2_s,
        )
    return tuple(t[:depth] for t in _STATE[key])


def _buffers():
    """The target's per-slot buffers at its capacity, with the replica slots' tiles, one set
    for the session."""
    if "buffers" not in _STATE:
        u8 = {"dtype": torch.uint8, "device": "cuda"}
        _STATE["buffers"] = dict(
            xq=torch.zeros(MAX_ROWS, HIDDEN // 2, **u8),
            xsf=torch.zeros(MAX_ROWS, HIDDEN // 16, **u8),
            w=torch.zeros(MAX_ROWS, TOP_K, dtype=torch.float32, device="cuda"),
            out=torch.zeros(MAX_ROWS, HIDDEN, dtype=torch.bfloat16, device="cuda"),
            maps=allocate_maps(MAX_ROWS, "cuda", replicas=REPLICAS),
            c=allocate_decode_c(MAX_ROWS, INTER, "cuda"),
            pc=allocate_pair_c(MAX_ROWS, INTER, "cuda", REPLICAS),
            sfc=allocate_sfc(MAX_ROWS, INTER, "cuda", REPLICAS),
        )
    return _STATE["buffers"]


def _compiled(pair: bool, replicas: int = 0, fc1_kblk: bool = False, fc2_kblk: bool = False):
    key = (
        (f"pair{replicas}" if pair else "decode")
        + ("_kblk" if fc1_kblk else "")
        + ("_kblk2" if fc2_kblk else "")
    )
    if key not in _STATE:
        if pair:
            _STATE[key] = build_prefill_pair(
                HIDDEN,
                INTER,
                EXPERTS + replicas,
                torch.cuda.current_device(),
                fc1_kblk=fc1_kblk,
                fc2_kblk=fc2_kblk,
            )
        else:
            b = _buffers()
            _STATE[key] = build_decode(
                c=b["c"],
                sfc=b["sfc"],
                maps=b["maps"],
                xq=b["xq"],
                xsf=b["xsf"],
                out=b["out"],
                hidden=HIDDEN,
                inter=INTER,
                experts=EXPERTS,
                max_rows=MAX_ROWS,
                fc1_kblk=fc1_kblk,
                fc2_kblk=fc2_kblk,
                dyn_slots=4,
            )
    return _STATE[key]


def _write_maps(maps: dict, ref: dict) -> None:
    """Write canonical maps into a capacity-sized set, the rest left as garbage."""
    for k, t in maps.items():
        t.fill_(-7)
        t[: ref[k].numel()].copy_(ref[k])


def _assert_dyn_counters_reset(compiled) -> None:
    """Every dynamic-FC2 slot's counter is back to zero after a launch: its last CTA resets
    both words, so the next launch on the slot (every layer's, every step) starts at tile 0.
    A missed reset would make that launch skip tiles."""
    ctr = getattr(compiled, "dyn_ctr", None)
    if ctr is not None:
        torch.cuda.synchronize()
        assert not bool(ctr.any()), f"dynamic FC2 counters not reset: {ctr[:, :2].tolist()}"


def _compare(out: torch.Tensor, ref: torch.Tensor) -> None:
    """The two sides share every rounding but three, so the band is set by those.

    1. FC2 adds each of a token's (up to eight) contributions into the bf16 output with
       atomics: one bf16 rounding per add, in an order that varies, against one rounding
       of the fp32 sum. Up to ~4.5 ulp of the row scale at the worst element, ~0.8 ulp
       relative RMS.
    2. FC1 requantizes its SwiGLU to NVFP4. Its fp32 sums differ from the reference's in
       order (~5e-6 relative) and its sigmoid and per-block reciprocal are approximate,
       so a value within that distance of an e2m1 midpoint can take the other code: one
       e2m1 step on one intermediate. At 8192 rows about 1e-4 of the intermediates do;
       one such step moves its token's outputs by up to ~8 ulp of the row scale, so a
       token hit twice approaches 16. Relative RMS: ~0.5 ulp.
    3. Its e4m3 block scale is `amax * (1/6) * norm_const` against the reference's
       `norm_const * (amax / 6)`, which can round across an e4m3 step only on an exact
       tie: a few blocks per million, bounded like (2).

    Hence 32 ulp at the worst element and 2 ulp relative RMS (twice the ~1 estimated).
    A missing expert contribution is ~35% relative RMS; a wrong gate/up pairing, scale
    atom or row gather is O(1).
    """
    assert_within_ulp(out, ref, element_ulp=32.0, rms_ulp=2.0)


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell) -> None:
    spec = cell.spec
    rows, offset, pair = spec["rows"], spec["offset"], spec["pair"]
    replicas = spec.get("replicas", 0)
    fc1_kblk = spec.get("fc1_kblk", False)
    fc2_kblk = spec.get("fc2_kblk", False)
    g = torch.Generator(device="cuda").manual_seed(zlib.crc32(cell.why.encode()) % 2**31)
    fc1_w, fc1_s, fc2_w, fc2_s = _weights(EXPERTS + replicas, fc1_kblk, fc2_kblk)
    b = _buffers()
    compiled = _compiled(pair, replicas, fc1_kblk, fc2_kblk)

    # The NVFP4 input as the closing collective or the quantize op writes it.
    xq_full, xsf_full = _nvfp4(
        torch.randn(rows, HIDDEN, generator=g, device="cuda").to(torch.bfloat16).float()
    )
    xq, xsf = b["xq"][:rows], b["xsf"][:rows]
    xq.copy_(xq_full)
    xsf.copy_(xsf_full)
    # Top-8 of 256 with about half the winners in this rank's window, skewed so some
    # experts take several 128-row tiles at the large counts.
    bias = torch.randn(GLOBAL_EXPERTS, generator=g, device="cuda") * 0.5
    bias[offset : offset + EXPERTS] += 1.0
    choice = torch.randn(rows, GLOBAL_EXPERTS, generator=g, device="cuda") * 0.3 + bias
    ids = torch.topk(choice, TOP_K, dim=-1).indices.to(torch.int32)
    raw = torch.rand(rows, TOP_K, generator=g, device="cuda") + 0.1
    topk_w = b["w"][:rows]
    topk_w.copy_(raw / raw.sum(-1, keepdim=True) * 2.5)
    maps = b["maps"]
    if replicas:
        # The context replica top-k's maps: home slots `e - offset`, and replica slot
        # `64 + h` for this rank's share of hot expert `hot[h]`'s rows (a random half here).
        hot = torch.randperm(GLOBAL_EXPERTS, generator=g, device="cuda")[:replicas]
        flat = ids.reshape(-1).long()
        slots = torch.where((flat >= offset) & (flat < offset + EXPERTS), flat - offset, -1)
        is_hot = flat.unsqueeze(1) == hot.unsqueeze(0)
        mine = torch.rand(flat.numel(), generator=g, device="cuda") < 0.5
        slots = torch.where(
            is_hot.any(1), torch.where(mine, EXPERTS + is_hot.int().argmax(1), -1), slots
        )
        _write_maps(maps, reference_slot_maps(slots, EXPERTS + replicas))
    else:
        _write_maps(maps, reference_maps(ids, offset))
    alpha1 = torch.rand(EXPERTS + replicas, generator=g, device="cuda") + 0.5
    alpha2 = torch.rand(EXPERTS + replicas, generator=g, device="cuda") + 0.5
    norm_const = torch.full((1,), spec.get("norm_const", 1.0), device="cuda")
    c = b["pc"] if pair else b["c"]
    sfc = b["sfc"]

    out = b["out"][:rows]
    join = None
    if spec.get("accumulate"):
        # The shared expert's partial lands on another stream well after FC1 starts:
        # an FC2 that did not wait adds onto NaN, then the copy overwrites its work.
        initial = (torch.randn(rows, HIDDEN, generator=g, device="cuda") * 0.5).to(torch.bfloat16)
        out.fill_(float("nan"))
        join = torch.cuda.Stream()
        join.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(join):
            torch.cuda._sleep(20_000_000)
            out.copy_(initial)
    else:
        out.zero_()
        initial = out.clone()
    with validating(op):
        result = op(
            xq, xsf, maps, topk_w, fc1_w, fc1_s, fc2_w, fc2_s,
            norm_const, alpha1, alpha2, out, c, sfc, join, compiled, spec.get("dyn_slot"),
        )  # fmt: skip
    assert result is out, "the op accumulates into `out` and returns it"
    _assert_dyn_counters_reset(compiled)
    # `reference` reads `out` as the value FC2 accumulates onto: the value before the call.
    ref = op.reference(
        xq, xsf, maps, topk_w, fc1_w, fc1_s, fc2_w, fc2_s,
        norm_const, alpha1, alpha2, initial, c, sfc, join, compiled,
    )  # fmt: skip
    _compare(out, ref)


def test_fc1_kblock_layout_round_trips() -> None:
    """The k-block relayout is a byte permutation: each 128-row x 128 B box is contiguous,
    each expert and each 128-row block keep their byte ranges, and `from_kblock` inverts it."""
    w = torch.randint(0, 256, (2, 2 * INTER, HIDDEN // 2), dtype=torch.uint8, device="cuda")
    k = moe_fc1_kblock.to_kblock(w)
    assert k.shape == w.shape and k.dtype == w.dtype
    assert torch.equal(moe_fc1_kblock.from_kblock(k), w)
    box = 128 * 128
    # Expert 1, row block 3, k-block 5: one contiguous 16 KB run.
    flat = k.reshape(-1)
    start = (1 * 2 * INTER * HIDDEN // 2) + 3 * 128 * HIDDEN // 2 + 5 * box
    assert torch.equal(flat[start : start + box].view(128, 128), w[1, 384:512, 640:768])
    # Each 128-row block is its own byte range (so is each 256-row N tile the decode L2
    # prefetch addresses): the block's slice of `k` is the block's own k-block image.
    for rb in range(0, 2 * INTER, 128):
        assert torch.equal(moe_fc1_kblock.from_kblock(k[:, rb : rb + 128]), w[:, rb : rb + 128])


def _fc1_outputs(
    pair: bool, rows: int, fc1_kblk: bool, ids: torch.Tensor, offset: int, g_seed: int
):
    """Run the op once on `rows` rows routed by `ids`, the FC1 output buffers prefilled with
    a sentinel, and return copies of the live tiles' C and SFC: FC1's output, bitwise."""
    g = torch.Generator(device="cuda").manual_seed(g_seed)
    fc1_w, fc1_s, fc2_w, fc2_s = _weights(EXPERTS, fc1_kblk)
    b = _buffers()
    compiled = _compiled(pair, 0, fc1_kblk)
    xq_full, xsf_full = _nvfp4(
        torch.randn(rows, HIDDEN, generator=g, device="cuda").to(torch.bfloat16).float()
    )
    xq, xsf = b["xq"][:rows], b["xsf"][:rows]
    xq.copy_(xq_full)
    xsf.copy_(xsf_full)
    topk_w = b["w"][:rows]
    topk_w.fill_(0.125)
    ref = reference_maps(ids, offset)
    _write_maps(b["maps"], ref)
    tiles = int(ref["num_non_exiting_tiles"].item())
    alpha = torch.rand(EXPERTS, generator=g, device="cuda") + 0.5
    norm_const = torch.ones(1, device="cuda")
    c = b["pc"] if pair else b["c"]
    c.fill_(0xA5)
    b["sfc"].fill_(0x5A)
    out = b["out"][:rows]
    out.zero_()
    op(
        xq, xsf, b["maps"], topk_w, fc1_w, fc1_s, fc2_w, fc2_s,
        norm_const, alpha, alpha, out, c, b["sfc"], None, compiled,
    )  # fmt: skip
    torch.cuda.synchronize()
    return c[:, : tiles * 128].clone(), b["sfc"][:, :tiles].clone(), tiles


@pytest.mark.parametrize(
    "pair,rows",
    [(False, 32), (False, 64), (False, 128), (False, 192), (True, 128), (True, 192), (True, 7513)],
)
def test_fc1_kblock_output_is_bitwise_row_major(pair: bool, rows: int) -> None:
    """The k-block layout changes only where B's bytes live: each TMA box lands in shared
    memory as the same image, so FC1's C and SFC must be bitwise those of the row-major
    weight, on the decode GG1 (32 requests x 1/2/4/6 rows) and on the prefill pair's GG1 at
    the MTP3 / MTP5 decode counts (N128) and at a context step (N256)."""
    g = torch.Generator(device="cuda").manual_seed(rows * 2 + int(pair))
    # Skewed top-8 so several experts take more than one tile at the context count.
    bias = torch.randn(GLOBAL_EXPERTS, generator=g, device="cuda") * 0.5
    bias[:EXPERTS] += 1.0
    choice = torch.randn(rows, GLOBAL_EXPERTS, generator=g, device="cuda") * 0.3 + bias
    ids = torch.topk(choice, TOP_K, dim=-1).indices.to(torch.int32)
    c0, s0, tiles = _fc1_outputs(pair, rows, False, ids, 0, rows)
    c1, s1, _ = _fc1_outputs(pair, rows, True, ids, 0, rows)
    assert tiles > 0
    # The sentinel is gone wherever FC1 writes: the comparison is not of two untouched buffers.
    assert not bool((c0 == 0xA5).all())
    assert torch.equal(c0, c1), f"C differs at {int((c0 != c1).sum())} bytes"
    assert torch.equal(s0, s1), f"SFC differs at {int((s0 != s1).sum())} bytes"


def _fc2_outputs(
    pair: bool,
    rows: int,
    kblk: bool,
    ids: torch.Tensor,
    replicas: int,
    slot: int,
    dyn_slot: int | None,
    g_seed: int,
    fc2_bytes_kblk: bool | None = None,
):
    """Run the op once with FC1 and FC2 both k-block (`kblk`) or both row-major, top-k
    weight only on `slot`, into a zeroed `out`, and return a copy of `out`. With one
    non-zero term per output element, FC2's bf16 atomics leave a deterministic result,
    so two runs compare bitwise. `fc2_bytes_kblk` overrides the FC2 weight's byte order
    alone (the negative control: kernels compiled for one layout handed the other)."""
    g = torch.Generator(device="cuda").manual_seed(g_seed)
    fc1_w, fc1_s, fc2_w, fc2_s = _weights(EXPERTS + replicas, kblk, kblk)
    if fc2_bytes_kblk is not None:
        fc2_w = _weights(EXPERTS + replicas, kblk, fc2_bytes_kblk)[2]
    b = _buffers()
    compiled = _compiled(pair, replicas, kblk, kblk)
    xq_full, xsf_full = _nvfp4(
        torch.randn(rows, HIDDEN, generator=g, device="cuda").to(torch.bfloat16).float()
    )
    xq, xsf = b["xq"][:rows], b["xsf"][:rows]
    xq.copy_(xq_full)
    xsf.copy_(xsf_full)
    if replicas:
        # Replica-slot maps as in `test_certified_cells`: hot experts' rows on slot 64 + h.
        hot = torch.randperm(GLOBAL_EXPERTS, generator=g, device="cuda")[:replicas]
        flat = ids.reshape(-1).long()
        slots = torch.where(flat < EXPERTS, flat, -1)
        is_hot = flat.unsqueeze(1) == hot.unsqueeze(0)
        mine = torch.rand(flat.numel(), generator=g, device="cuda") < 0.5
        slots = torch.where(
            is_hot.any(1), torch.where(mine, EXPERTS + is_hot.int().argmax(1), -1), slots
        )
        _write_maps(b["maps"], reference_slot_maps(slots, EXPERTS + replicas))
    else:
        _write_maps(b["maps"], reference_maps(ids, 0))
    topk_w = b["w"][:rows]
    topk_w.zero_()
    topk_w[:, slot] = torch.rand(rows, generator=g, device="cuda") + 0.25
    alpha = torch.rand(EXPERTS + replicas, generator=g, device="cuda") + 0.5
    norm_const = torch.ones(1, device="cuda")
    out = b["out"][:rows]
    out.zero_()
    op(
        xq, xsf, b["maps"], topk_w, fc1_w, fc1_s, fc2_w, fc2_s, norm_const, alpha, alpha,
        out, b["pc"] if pair else b["c"], b["sfc"], None, compiled, dyn_slot,
    )  # fmt: skip
    torch.cuda.synchronize()
    _assert_dyn_counters_reset(compiled)
    return out.clone()


@pytest.mark.parametrize(
    "pair,rows,dyn_slot,replicas",
    [
        (False, 32, 0, 0),
        (False, 64, 1, 0),
        (False, 32, None, 0),
        (False, 64, None, 0),
        (False, 128, None, 0),
        (False, 192, None, 0),
        (True, 128, None, 0),
        (True, 192, None, 0),
        (True, 7513, None, 0),
        (True, 7513, None, REPLICAS),
    ],
)
def test_fc2_kblock_output_is_bitwise_row_major(
    pair: bool, rows: int, dyn_slot: int | None, replicas: int
) -> None:
    """FC1 and FC2 both k-block against both row-major, on every FC2 kernel: the decode
    dynamic FC2 (<= 64 rows with a `dyn_slot`), the static decode FC2 (its fallback, also
    past 64 rows when the prefill pair did not build) and the prefill pair's GG2 at the
    MTP3 / MTP5 decode counts (N128) and at a context step (N256), on the 64-deep stack and
    the 72-deep replica stack. Each top-k slot in turn carries the only non-zero weight, so
    every expanded row's FC2 product is compared bitwise, not within a band."""
    g = torch.Generator(device="cuda").manual_seed(rows * 4 + int(pair) * 2 + bool(replicas))
    bias = torch.randn(GLOBAL_EXPERTS, generator=g, device="cuda") * 0.5
    bias[:EXPERTS] += 1.0
    choice = torch.randn(rows, GLOBAL_EXPERTS, generator=g, device="cuda") * 0.3 + bias
    ids = torch.topk(choice, TOP_K, dim=-1).indices.to(torch.int32)
    seed = rows + 7 * replicas
    for slot in range(TOP_K):
        o0 = _fc2_outputs(pair, rows, False, ids, replicas, slot, dyn_slot, seed + slot)
        o1 = _fc2_outputs(pair, rows, True, ids, replicas, slot, dyn_slot, seed + slot)
        assert bool(o0.abs().amax() > 0), "FC2 wrote nothing: the comparison would be vacuous"
        assert torch.equal(o0.view(torch.int16), o1.view(torch.int16)), (
            f"slot {slot}: out differs at {int((o0 != o1).sum())} elements"
        )
    # The comparison can fail: k-block kernels on the row-major FC2 bytes give other numbers.
    bad = _fc2_outputs(pair, rows, True, ids, replicas, 0, dyn_slot, seed, fc2_bytes_kblk=False)
    good = _fc2_outputs(pair, rows, True, ids, replicas, 0, dyn_slot, seed)
    assert not torch.equal(bad, good)


def _unreachable(*args) -> None:
    raise AssertionError("the guard should have refused before the kernel ran")


def _shapes():
    """Operands with the right shapes and no content: the guards read only metadata."""
    if "empty" not in _STATE:
        u8 = {"dtype": torch.uint8, "device": "cuda"}
        e4 = {"dtype": torch.float8_e4m3fn, "device": "cuda"}
        _STATE["empty"] = (
            torch.empty(EXPERTS, 2 * INTER, HIDDEN // 2, **u8),
            torch.empty(EXPERTS, 2 * INTER // 128, HIDDEN // 64, 512, **e4),
            torch.empty(EXPERTS, HIDDEN, INTER // 2, **u8),
            torch.empty(EXPERTS, HIDDEN // 128, INTER // 64, 512, **e4),
        )
    return _STATE["empty"]


def _args(rows: int, pair: bool, max_rows: int = MAX_ROWS):
    u8 = {"dtype": torch.uint8, "device": "cuda"}
    f32 = {"dtype": torch.float32, "device": "cuda"}
    bias = torch.zeros(1, **f32)
    compiled = (
        CompiledPairExperts({}, bias, HIDDEN, INTER, EXPERTS)
        if pair
        else CompiledDecodeExperts(
            _unreachable, _unreachable, bias, HIDDEN, INTER, EXPERTS, max_rows
        )
    )
    fc1_w, fc1_s, fc2_w, fc2_s = _shapes()
    cap = max(rows, 128)
    return dict(
        xq=torch.zeros(rows, HIDDEN // 2, **u8),
        xsf=torch.zeros(rows, HIDDEN // 16, **u8),
        maps=allocate_maps(cap, "cuda"),
        topk_w=torch.zeros(rows, TOP_K, **f32),
        fc1_w=fc1_w,
        fc1_s=fc1_s,
        fc2_w=fc2_w,
        fc2_s=fc2_s,
        norm_const=torch.ones(1, **f32),
        alpha1=torch.ones(EXPERTS, **f32),
        alpha2=torch.ones(EXPERTS, **f32),
        out=torch.zeros(rows, HIDDEN, dtype=torch.bfloat16, device="cuda"),
        c=(allocate_pair_c if pair else allocate_decode_c)(cap, INTER, "cuda"),
        sfc=allocate_sfc(cap, INTER, "cuda"),
        join=None,
        compiled=compiled,
    )


def test_guard_refuses_rows_past_the_compiled_capacity() -> None:
    """The decode pair's A layout is compiled for its capacity; past it FC1 reads rows
    of the next allocation."""
    args = _args(65, pair=False, max_rows=64)
    with pytest.raises(AssertionError, match="compiled for up to 64 rows"):
        with validating(op):
            op(**args)


def test_guard_refuses_the_decode_c_on_the_prefill_pair() -> None:
    """The two FC1 outputs have different row pitches; the wrong one is garbage in FC2."""
    args = _args(65, pair=True)
    args["c"] = allocate_decode_c(128, INTER, "cuda")
    with pytest.raises(AssertionError, match="for the prefill pair"):
        with validating(op):
            op(**args)


def test_guard_refuses_bf16_routing_weights() -> None:
    """FC2 reads `token_final_scales` as fp32 whatever it is handed."""
    args = _args(8, pair=False)
    args["topk_w"] = args["topk_w"].to(torch.bfloat16)
    with pytest.raises(AssertionError, match="topk_w must be"):
        with validating(op):
            op(**args)


def test_guard_refuses_maps_too_small_for_the_rows() -> None:
    """The tile maps are read up to the live tile count, past a short buffer's end."""
    args = _args(512, pair=True)
    args["maps"] = allocate_maps(16, "cuda")
    with pytest.raises(AssertionError, match="rows need"):
        with validating(op):
            op(**args)
