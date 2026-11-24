# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests of the rubin_dsa_nvfp4_gather_fp8 entry (NVFP4 latent rows into an FP8 scratch)."""

import zlib

import numpy as np
import pytest
import torch

# l0_gb300 collects this directory: any GPU but sm_107 skips the module before a family import.
if torch.cuda.is_available() and torch.cuda.get_device_capability() != (10, 7):
    pytest.skip("certified on sm_107 only", allow_module_level=True)

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.attention.rubin_dsa_nvfp4_gather_fp8 import (  # noqa: E402,E501
    CODE_BYTES,
    E2M1_MAGNITUDES,
    LATENT_DIM,
    SCALE_BYTES,
    nvfp4_rows_to_e4m3,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.attention.rubin_dsa_nvfp4_gather_fp8 import (  # noqa: E402,E501
    rubin_dsa_nvfp4_gather_fp8 as op,
)

__extra_import_path__ = ["../../../.."]
if torch.cuda.is_available():  # the GPU cells' guard interposer: a CPU run needs neither
    from _validating import certifying  # noqa: E402,F401 -- path above; fixture by name

cpu = pytest.mark.cpu_only
gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="drives the kernel on sm_107")

AMAX100 = 100.0 / (448.0 * 6.0)


# ---- an independent derivation of the rounding ---------------------------------------------------


def _e4m3_table() -> tuple[np.ndarray, np.ndarray]:
    """(bytes, values) of every finite non-negative E4M3 code, decoded from its bit fields."""
    codes = np.arange(0x7F, dtype=np.int64)  # 0x7F is NaN
    exp, man = codes >> 3, codes & 7
    values = np.where(exp == 0, man / 8.0 * 2.0**-6, (1 + man / 8.0) * 2.0 ** (exp - 7))
    return codes, values


def _to_e4m3_bytes(x: np.ndarray) -> np.ndarray:
    """Round float64 ``x`` to E4M3 by search: nearest value, ties to the even code, saturating."""
    codes, values = _e4m3_table()
    mag = np.minimum(np.abs(x), values[-1])
    hi = np.clip(np.searchsorted(values, mag), 1, len(values) - 1)
    lo = hi - 1
    d_lo, d_hi = mag - values[lo], values[hi] - mag
    pick = np.where(
        d_lo < d_hi, lo, np.where(d_hi < d_lo, hi, np.where(codes[lo] % 2 == 0, lo, hi))
    )
    pick = np.where(mag == values[hi], hi, pick)
    return (codes[pick] | np.where(np.signbit(x), 0x80, 0)).astype(np.uint8)


def _independent(codes: torch.Tensor, scales: torch.Tensor, g: float) -> torch.Tensor:
    """The decoder's bytes derived without torch's arithmetic or its E4M3 conversion."""
    c = codes.numpy().astype(np.int64)
    nib = np.stack((c & 15, c >> 4), axis=-1).reshape(c.shape[0], LATENT_DIM)
    val = np.array(E2M1_MAGNITUDES)[nib & 7] * np.where(nib >= 8, -1.0, 1.0)
    s = scales.view(torch.uint8).numpy().astype(np.int64)
    _, table = _e4m3_table()
    sv = table[s & 0x7F] * np.where(s >= 0x80, -1.0, 1.0)
    sv = np.repeat(sv, 16, axis=-1)
    if g == 1.0:
        x = val * sv
    else:
        s16 = (sv.astype(np.float32) * np.float32(g)).astype(np.float16)
        x = (val.astype(np.float16) * s16).astype(np.float64)
    return torch.from_numpy(_to_e4m3_bytes(x))


def _finite_scale_bytes(n: int, gen: torch.Generator) -> torch.Tensor:
    """Random E4M3 bytes over every finite code, both signs (NaN, 0x7F and 0xFF, left out)."""
    b = torch.randint(0, 254, (n,), generator=gen, dtype=torch.int64)
    return torch.where(b >= 0x7F, b + 1, b).to(torch.uint8)


def _normal_scale_bytes(n: int, gen: torch.Generator) -> torch.Tensor:
    """Positive block scales of a write with amax up to a few hundred: 2^-6 .. 2^5."""
    exp = torch.randint(1, 13, (n,), generator=gen)
    man = torch.randint(0, 8, (n,), generator=gen)
    return (exp * 8 + man).to(torch.uint8)


def _every_code_and_scale() -> tuple[torch.Tensor, torch.Tensor]:
    """Rows holding all 16 codes against every finite scale code: one 16-value block per scale."""
    blocks = torch.cat([torch.arange(0x7F), torch.arange(0x80, 0xFF)]).to(torch.uint8)
    per_row = LATENT_DIM // 16
    rows = -(-blocks.numel() // per_row)
    scales = torch.full((rows * per_row,), 0x38, dtype=torch.uint8)  # 1.0 pads the last row
    scales[: blocks.numel()] = blocks
    pairs = torch.arange(8, dtype=torch.uint8)
    block_codes = (2 * pairs) | ((2 * pairs + 1) << 4)  # element 2b low nibble, 2b + 1 high
    codes = block_codes.repeat(rows * per_row).view(rows, CODE_BYTES)
    return codes, scales.view(rows, SCALE_BYTES).view(torch.float8_e4m3fn)


# ---- CPU ---------------------------------------------------------------------------------------


@cpu
def test_decoder_pinned_rows():
    codes = torch.zeros(1, CODE_BYTES, dtype=torch.uint8)
    scales = torch.full((1, SCALE_BYTES), 0x38, dtype=torch.uint8)  # 1.0
    codes[0, 0] = 0x21  # element 0: code 1 (0.5), element 1: code 2 (1.0)
    scales[0, 0] = 0x40  # 2.0 over elements 0-15
    codes[0, 8] = 0xF7  # elements 16, 17: 6.0 and -6.0
    scales[0, 1] = 0x7E  # 448: 2688 saturates to 448
    codes[0, 16] = 0x31  # elements 32, 33: 0.5 and 1.5
    scales[0, 2] = 0x01  # 2^-9: 2^-10 ties to 0, 1.5 x 2^-9 ties up to 2^-8
    codes[0, 24] = 0x98  # elements 48, 49: -0.0 and -0.5
    scales[0, 3] = 0x30  # 0.5
    out = nvfp4_rows_to_e4m3(codes, scales.view(torch.float8_e4m3fn), torch.ones(1))
    got = out.view(torch.uint8)[0]
    assert got[:2].tolist() == [0x38, 0x40]  # 1.0, 2.0
    assert got[16:18].tolist() == [0x7E, 0xFE]  # 448, -448
    assert got[32:34].tolist() == [0x00, 0x02]  # 0, 2^-8
    assert got[48:50].tolist() == [0x80, 0xA8]  # -0.0, -0.25
    assert got[2:16].tolist() == [0] * 14


@cpu
@pytest.mark.parametrize("g", [1.0, AMAX100, 1e-3], ids=lambda g: f"g{g:.4g}")
def test_decoder_matches_an_independent_derivation(g):
    codes, scales = _every_code_and_scale()
    out = nvfp4_rows_to_e4m3(codes, scales, torch.tensor([g]))
    assert torch.equal(out.view(torch.uint8), _independent(codes, scales, g))


@cpu
def test_decoder_matches_on_random_rows():
    gen = torch.Generator().manual_seed(7)
    rows = (1 << 14) + 100  # past one decode chunk
    codes = torch.randint(0, 256, (rows, CODE_BYTES), generator=gen, dtype=torch.uint8)
    scales = _finite_scale_bytes(rows * SCALE_BYTES, gen).view(rows, SCALE_BYTES)
    out = nvfp4_rows_to_e4m3(codes, scales.view(torch.float8_e4m3fn), torch.tensor([AMAX100]))
    # The rows on both sides of the first chunk's end (the derivation is slow on all of them).
    edge = torch.cat([torch.arange(64), torch.arange((1 << 14) - 64, rows)])
    want = _independent(codes[edge], scales[edge], AMAX100)
    assert torch.equal(out.view(torch.uint8)[edge], want)


def _small_pool(tokens: int, seed: int):
    gen = torch.Generator().manual_seed(seed)
    codes = torch.randint(0, 256, (tokens, CODE_BYTES), generator=gen, dtype=torch.uint8)
    scales = _normal_scale_bytes(tokens * SCALE_BYTES, gen).view(tokens, SCALE_BYTES)
    return codes, scales.view(torch.float8_e4m3fn)


@cpu
def test_reference_gathers_valid_pairs_and_leaves_the_rest():
    codes, scales = _small_pool(10, seed=1)
    index = torch.tensor([[2, 0, -1], [3, 1, 9]], dtype=torch.int32)
    out = torch.full((2, 3, LATENT_DIM), 0x5A, dtype=torch.uint8).view(torch.float8_e4m3fn)
    g = torch.tensor([AMAX100])
    got, compact = op.reference(codes, scales, index, out, torch.empty_like(index), g, 0, 0, 9)
    # Pool row 9 is at the bound num_pool_tokens = 9: it reads as -1 like the -1 slot.
    assert compact.tolist() == [[0, 1, -1], [3, 4, -1]]
    rows = nvfp4_rows_to_e4m3(codes, scales, g).view(torch.uint8)
    got = got.view(torch.uint8)
    for (r, c), row in {(0, 0): 2, (0, 1): 0, (1, 0): 3, (1, 1): 1}.items():
        assert torch.equal(got[r, c], rows[row])
    assert torch.all(got[0, 2] == 0x5A) and torch.all(got[1, 2] == 0x5A)
    assert torch.all(out.view(torch.uint8) == 0x5A), "the reference writes a copy"


def _guard_args(**overrides) -> dict:
    index = torch.tensor([[2, 0, -1]], dtype=torch.int32)
    args = dict(
        host_pool_pointers=torch.zeros(1, 2, 2, dtype=torch.int64),
        host_pool_mapping=torch.zeros(1, 2, dtype=torch.int32),
        global_indices=index,
        out=torch.empty((1, 3, LATENT_DIM), dtype=torch.float8_e4m3fn),
        compact_indices=torch.empty_like(index),
        global_dequant_scale=torch.ones(1),
        layer_idx=0,
        residual_dim=0,
        num_pool_tokens=10,
    )
    args.update(overrides)
    return args


@cpu
def test_guard_admits_a_target_call():
    op.is_valid(**_guard_args())


@cpu
@pytest.mark.parametrize(
    "overrides,match",
    [
        (dict(residual_dim=64), "residual_dim must be 0"),
        (dict(out=torch.empty((1, 3, 512), dtype=torch.float8_e4m3fn)), "576-wide"),
        (dict(num_pool_tokens=2), "num_pool_tokens"),
    ],
    ids=["residual", "width", "bound"],
)
def test_guard_refuses(overrides, match):
    with pytest.raises(AssertionError, match=match):
        op.is_valid(**_guard_args(**overrides))


# ---- GPU (sm_107) ----------------------------------------------------------------------------


def _seed(text: str) -> int:
    return zlib.crc32(text.encode()) % 2**31


def _build(spec: dict, seed: int) -> tuple[dict, dict]:
    """Call arguments on the GPU, and the pool the reference takes in place of the host tables."""
    gen = torch.Generator().manual_seed(seed)
    tokens, rows, topk = spec["pool_tokens"], spec["rows"], spec["topk"]
    scale_bytes = _finite_scale_bytes if spec["scales"] == "wide" else _normal_scale_bytes
    pools = []
    for _ in range(spec["pools"]):
        codes = torch.randint(0, 256, (tokens, CODE_BYTES), generator=gen, dtype=torch.uint8)
        scales = scale_bytes(tokens * SCALE_BYTES, gen).view(tokens, SCALE_BYTES)
        pools.append((codes.cuda(), scales.view(torch.float8_e4m3fn).cuda()))
    pointers = torch.zeros((len(pools), 2, 2), dtype=torch.int64)
    for p, (codes, scales) in enumerate(pools):
        pointers[p, 0, 0], pointers[p, 0, 1] = codes.data_ptr(), scales.data_ptr()
    layer = len(pools) - 1  # layer l reads pool l: the last layer the last pool
    mapping = torch.tensor([[p, 0] for p in range(len(pools))], dtype=torch.int32)
    index = torch.randint(0, tokens, (rows, topk), generator=gen, dtype=torch.int32)
    if spec["invalid"]:
        index[:, topk - spec["invalid"] :] = -1
    args = dict(
        host_pool_pointers=pointers,
        host_pool_mapping=mapping,
        global_indices=index.cuda(),
        # A sentinel: the reference copies it, so pairs the op must not write are compared too.
        out=torch.full((rows, topk, LATENT_DIM), 0x5A, dtype=torch.uint8, device="cuda").view(
            torch.float8_e4m3fn
        ),
        compact_indices=torch.full((rows, topk), -7, dtype=torch.int32, device="cuda"),
        global_dequant_scale=torch.tensor([spec["scale"]], dtype=torch.float32, device="cuda"),
        layer_idx=layer,
        residual_dim=0,
        num_pool_tokens=tokens,
    )
    return args, dict(data_pool=pools[layer][0], scale_pool=pools[layer][1])


def _reference(args: dict, pool: dict, out_before: torch.Tensor):
    rest = {k: v for k, v in args.items() if k not in ("host_pool_pointers", "host_pool_mapping")}
    return op.reference(**pool, **dict(rest, out=out_before))


@gpu
@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:48] for c in op.CELLS])
def test_certified_cells(cell, certifying):  # noqa: F811
    certifying(op)
    args, pool = _build(cell.spec, _seed(cell.why))
    before = args["out"].clone()
    out, compact = op(**args)
    ref_out, ref_compact = _reference(args, pool, before)
    torch.cuda.synchronize()
    assert out is args["out"] and compact is args["compact_indices"]
    assert torch.equal(compact, ref_compact)
    assert torch.equal(out.view(torch.uint8), ref_out.view(torch.uint8))


@gpu
def test_graph_replay_gathers_new_indices():
    """Captured on one Top-K, replayed on another written into the same buffer."""
    spec = dict(op.CELLS[1].spec, invalid=300)
    args, pool = _build(spec, seed=11)
    op(**args)  # eager first, as the engine warms up before capture
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        op(**args)
    gen = torch.Generator().manual_seed(12)
    fresh = torch.randint(0, spec["pool_tokens"], args["global_indices"].shape, generator=gen)
    fresh[: spec["rows"] // 2, :700] = -1
    args["global_indices"].copy_(fresh.to(torch.int32))
    before = args["out"].clone()
    graph.replay()
    ref_out, ref_compact = _reference(args, pool, before)
    torch.cuda.synchronize()
    assert torch.equal(args["compact_indices"], ref_compact)
    assert torch.equal(args["out"].view(torch.uint8), ref_out.view(torch.uint8))


@gpu
def test_pool_rows_at_the_bound_read_as_minus_one():
    """Outside the guard (which refuses such a call): rows at or past ``num_pool_tokens`` are
    dropped like -1, and their scratch rows keep their bytes."""
    args, pool = _build(dict(op.CELLS[0].spec, rows=4), seed=13)
    tokens = args["num_pool_tokens"]
    args["global_indices"][:, :3] = torch.tensor([tokens, tokens + 5, 2**31 - 1], device="cuda")
    before = args["out"].clone()
    out, compact = op(**args)
    ref_out, ref_compact = _reference(args, pool, before)
    torch.cuda.synchronize()
    assert torch.all(compact[:, :3] == -1)
    assert torch.equal(compact, ref_compact)
    assert torch.equal(out.view(torch.uint8), ref_out.view(torch.uint8))
