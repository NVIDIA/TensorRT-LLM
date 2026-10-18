# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the musespark_open preattn_prep entry.

The entry's claim is stronger than closeness: it is the built-in
pre-attention prep, bit for bit. So besides the cells (gated against the
entry's float64 reference within a stated band), every configuration is also
run through the built-in ``MuseSparkOpenAttention`` -- ``preprocess_qkv`` then
``convert_qkv``, exactly what ``Attention.forward`` calls between the QKV
projection and the attention op -- and the two results must be equal as bits.
The built-in module is built at this target's per-rank geometry (16 query and
2 key-value heads of 64) on one GPU.

``MODELING_V2_TIMING=1`` also runs the microbenchmark (printed, never gated).
"""

import os
import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep import (  # noqa: E501
    attention as target_attention,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.ops.attention.preattn_prep import (  # noqa: E501
    CKPT_EPS,
    CKPT_KV_HEADS,
    CKPT_MAX_POSITIONS,
    CKPT_Q_HEADS,
    CKPT_Q_SCALE,
    HEAD_DIM,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.ops.attention.preattn_prep import (  # noqa: E501
    preattn_prep as op,
)

__extra_import_path__ = ["../../../../.."]
from _validating import (  # noqa: E402,F401 — path above; fixture by name
    assert_within_ulp,
    certifying,
)

assert torch.cuda.is_available(), "preattn_prep requires a CUDA device"

DEV = "cuda"
Q_WIDTH = CKPT_Q_HEADS * HEAD_DIM
KV_WIDTH = CKPT_KV_HEADS * HEAD_DIM
PROJ_WIDTH = 2 * (Q_WIDTH + KV_WIDTH)
OUT_WIDTH = Q_WIDTH + 2 * KV_WIDTH
#: Layer 0 is a sliding-window RoPE layer, layer 1 a full-attention NoPE layer.
ROPE_LAYER, NOPE_LAYER = 0, 1


def _attention(layer_idx: int):
    """The built-in attention module at one tep8 rank's geometry."""
    from tensorrt_llm._torch.configs.musespark_open import MuseSparkOpenConfig
    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm._torch.models.modeling_musespark_open import MuseSparkOpenAttention
    from tensorrt_llm.mapping import Mapping

    config = MuseSparkOpenConfig(
        num_attention_heads=CKPT_Q_HEADS,
        num_key_value_heads=CKPT_KV_HEADS,
        moe={"topk": 8, "num_experts": 256, "override_routed_hidden_dim": 4096},
    )
    model_config = ModelConfig(
        pretrained_config=config, mapping=Mapping(world_size=1, tp_size=1, rank=0)
    )
    with torch.device(DEV):
        attention = MuseSparkOpenAttention(model_config, layer_idx)
    assert attention.uses_rope == (layer_idx == ROPE_LAYER)
    assert attention.num_heads == CKPT_Q_HEADS and attention.num_key_value_heads == CKPT_KV_HEADS
    assert attention.scale_query_by == CKPT_Q_SCALE
    assert attention.q_norm.variance_epsilon == CKPT_EPS
    return attention


_MODULES = {}


def _module(rope: bool):
    key = ROPE_LAYER if rope else NOPE_LAYER
    if key not in _MODULES:
        _MODULES[key] = _attention(key)
    return _MODULES[key]


def _permutation() -> torch.Tensor:
    """Checkpoint column order -> low-latency layout ``[q | k | v | gate]`` (as
    ``musespark_open_qkv_layout.prepare_weight_layout`` permutes the weight rows)."""
    interleaved = torch.arange(2 * Q_WIDTH, device=DEV).reshape(CKPT_Q_HEADS, 2 * HEAD_DIM)
    return torch.cat(
        (
            interleaved[:, :HEAD_DIM].reshape(-1),
            torch.arange(2 * Q_WIDTH, PROJ_WIDTH, device=DEV),
            interleaved[:, HEAD_DIM:].reshape(-1),
        )
    )


def _projection(rows: int, seed: int, zero_heads: bool = False) -> torch.Tensor:
    """A BF16 projection in checkpoint order, each 64-wide head at its own
    log-normal scale (sigma 2 in ln), so the norm sees six decades of input."""
    g = torch.Generator(device=DEV).manual_seed(seed)
    heads = PROJ_WIDTH // HEAD_DIM
    x = torch.randn(rows, heads, HEAD_DIM, generator=g, device=DEV)
    x = x * torch.exp(2.0 * torch.randn(rows, heads, 1, generator=g, device=DEV))
    if zero_heads:
        x[:, ::5] = 0.0
    return x.reshape(rows, PROJ_WIDTH).to(torch.bfloat16)


def _positions(rows: int, seed: int, kind: str = "uniform") -> torch.Tensor:
    g = torch.Generator(device=DEV).manual_seed(seed + 7)
    low = CKPT_MAX_POSITIONS - 2048 if kind == "tail" else 0
    return torch.randint(
        low, CKPT_MAX_POSITIONS, (1, rows), generator=g, device=DEV, dtype=torch.int32
    )


def _builtin_prep(attention, projection: torch.Tensor, positions: torch.Tensor):
    """What ``Attention.forward`` runs between the projection and the attention op."""
    q, k, v, gate = attention.preprocess_qkv(projection, positions)
    qkv, k, v = attention.convert_qkv(q, k, v)
    assert k is None and v is None
    return qkv, gate


def _call_args(attention, projection, positions, permuted):
    rope = attention.uses_rope
    return (
        projection,
        attention.rotary_emb.rotary_cos_sin if rope else None,
        positions if rope else None,
        CKPT_Q_HEADS,
        CKPT_KV_HEADS,
        attention.q_norm.variance_epsilon,
        attention.scale_query_by,
        permuted,
    )


def _assert_bits_equal(got: torch.Tensor, want: torch.Tensor, what: str) -> None:
    assert got.dtype == want.dtype == torch.bfloat16, what
    assert got.shape == want.shape, (what, got.shape, want.shape)
    diff = got.contiguous().view(torch.int16) != want.contiguous().view(torch.int16)
    if diff.any():
        rows, cols = diff.nonzero(as_tuple=True)
        raise AssertionError(
            f"{what}: {int(diff.sum())} element(s) differ from the built-in prep, first at "
            f"row {int(rows[0])} col {int(cols[0])}: {got[rows[0], cols[0]].item()} vs "
            f"{want[rows[0], cols[0]].item()}"
        )


def _assert_within_reference(qkv_out, gate_out, ref_qkv, ref_gate) -> None:
    """Q and K heads within 4 "scaled ulps" (units of 2^-8 times the head's
    largest magnitude) element by element and 0.05 ulp relative RMS; V and the
    gate exactly (they are copies).

    Derivation: the op and the float64 reference round at the same three
    points (norm, query scale, RoPE). The op's norm sum and ``rsqrt.approx``
    are fp32, so a norm output on a BF16 rounding boundary can land one ulp
    away; the query scale (5.47) multiplies that ulp before the next rounding,
    so a query element can sit two of its own ulps -- up to 4 scaled ulps --
    from the mirror, and RoPE mixes two such elements with |cos| + |sin| <=
    sqrt(2) before rounding again; only a small fraction of the elements differ
    at all (``test_bit_identity_at_scale`` prints the observed worst and RMS
    distance over 32768 x 18 heads). The built-in prep is exactly as far from
    the mirror: it is the same bits.
    """
    rows = qkv_out.shape[0]
    for lo, hi in ((0, Q_WIDTH), (Q_WIDTH, Q_WIDTH + KV_WIDTH)):
        got = qkv_out[:, lo:hi].reshape(-1, HEAD_DIM)
        want = ref_qkv[:, lo:hi].reshape(-1, HEAD_DIM)
        assert_within_ulp(got, want, element_ulp=4.0, rms_ulp=0.05)
    _assert_bits_equal(qkv_out[:, Q_WIDTH + KV_WIDTH :], ref_qkv[:, Q_WIDTH + KV_WIDTH :], "v")
    _assert_bits_equal(gate_out, ref_gate, "gate")
    assert qkv_out.shape == (rows, OUT_WIDTH) and gate_out.shape == (rows, Q_WIDTH)


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:48] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — pytest resolves the fixture by this name
    certifying(op)
    spec = cell.spec
    rows, rope = spec["rows"], spec["rope"]
    permuted = spec.get("permuted", False)
    seed = zlib.crc32(cell.why.encode())
    attention = _module(rope)
    projection = _projection(rows, seed, zero_heads=spec.get("zero_heads", False))
    positions = _positions(rows, seed, spec.get("positions", "uniform"))
    operand = projection[:, _permutation()].contiguous() if permuted else projection
    before = operand.clone()

    args = _call_args(attention, operand, positions, permuted)
    qkv_out, gate_out = op(*args)
    assert torch.equal(operand, before), "the projection is read-only"
    assert qkv_out.is_contiguous() and gate_out.is_contiguous()
    _assert_within_reference(qkv_out, gate_out, *op.reference(*args))

    # The claim: the built-in prep, bit for bit (on the checkpoint-order projection;
    # with the low-latency layout the built-in re-gathers that order first).
    want_qkv, want_gate = _builtin_prep(attention, projection.clone(), positions)
    _assert_bits_equal(qkv_out, want_qkv, "qkv")
    _assert_bits_equal(gate_out, want_gate.reshape(rows, -1), "gate")


@pytest.mark.parametrize("rope", (True, False), ids=("rope", "nope"))
def test_bit_identity_at_scale(rope) -> None:
    """Rows and heads are independent in both implementations, so one wide call
    samples 32768 x 18 heads against the built-in prep."""
    attention = _module(rope)
    rows = 32768
    projection = _projection(rows, seed=4242 + rope)
    positions = _positions(rows, seed=4242 + rope)
    args = _call_args(attention, projection, positions, False)
    qkv_out, gate_out = op(*args)
    want_qkv, want_gate = _builtin_prep(attention, projection.clone(), positions)
    _assert_bits_equal(qkv_out, want_qkv, "qkv")
    _assert_bits_equal(gate_out, want_gate.reshape(rows, -1), "gate")

    # The same sample against the float64 reference: where the band sits.
    ref_qkv, ref_gate = op.reference(*args)
    heads = qkv_out[:, : Q_WIDTH + KV_WIDTH].reshape(-1, HEAD_DIM).float()
    ref_heads = ref_qkv[:, : Q_WIDTH + KV_WIDTH].reshape(-1, HEAD_DIM).float()
    scale = ref_heads.abs().amax(-1, keepdim=True).clamp_min(1e-9) * 2.0**-8
    worst = ((heads - ref_heads).abs() / scale).max().item()
    rms = (
        (heads - ref_heads).pow(2).mean().sqrt() / ref_heads.pow(2).mean().sqrt()
    ).item() / 2.0**-8
    differ = (heads != ref_heads).float().mean().item()
    print(
        f"\n  vs float64 reference: worst element {worst:.3f} ulp of its head's scale, "
        f"relative RMS {rms:.4f} ulp, {differ:.2e} of elements differ"
    )
    _assert_within_reference(qkv_out, gate_out, ref_qkv, ref_gate)


def _special_heads(kind: str, rows: int, g: torch.Generator) -> torch.Tensor:
    """``[rows, 36, 64]`` float32 heads of one special-value class (exact in BF16)."""
    heads = PROJ_WIDTH // HEAD_DIM
    shape = (rows, heads, HEAD_DIM)
    sign = torch.where(torch.rand(shape, generator=g, device=DEV) < 0.5, -1.0, 1.0)
    unit = torch.randn(shape, generator=g, device=DEV)
    # BF16 subnormals: exponent field 0, mantissa 1..127, i.e. k * 2^-133.
    subnormal = sign * torch.randint(1, 128, shape, generator=g, device=DEV) * 2.0**-133
    tiny_normal = sign * torch.randint(128, 512, shape, generator=g, device=DEV) * 2.0**-133
    if kind == "subnormal":
        return subnormal
    if kind == "smallest-normal":
        return sign * 2.0**-126
    if kind == "tiny-normal":
        return tiny_normal
    if kind == "half-subnormal":
        return torch.where(torch.arange(HEAD_DIM, device=DEV) % 2 == 0, subnormal, unit)
    if kind == "subnormal-norm-output":
        # One element at 1e30 makes rsqrt ~ 1e-29, so the 1e-10 elements
        # normalize to ~1e-39: a subnormal norm output, then scaled and rotated.
        x = unit * 1e-10
        x[..., 5] = 1e30
        return x
    if kind == "near-max":
        return sign * torch.empty(shape, device=DEV).uniform_(1e38, 3.3e38, generator=g)
    if kind == "inf":
        x = unit.clone()
        x[..., 7] = float("inf")
        x[..., 9] = float("-inf")
        return x
    if kind == "nan":
        x = unit.clone()
        x[..., 3] = float("nan")
        return x
    if kind == "zero":
        return torch.zeros(shape, device=DEV)
    raise ValueError(kind)


_SPECIAL_KINDS = (
    "zero",
    "subnormal",
    "smallest-normal",
    "tiny-normal",
    "half-subnormal",
    "subnormal-norm-output",
    "near-max",
    "inf",
    "nan",
)


@pytest.mark.parametrize("kind", _SPECIAL_KINDS)
@pytest.mark.parametrize("rope", (True, False), ids=("rope", "nope"))
def test_bit_identity_on_special_values(rope, kind) -> None:
    """Subnormal and smallest-normal heads give RoPE outputs in the subnormal
    range, which the fast-math built-in RoPE flushes to signed zero; the entry
    flushes them too. Heads mixing 1e30 with 1e-10 give subnormal norm
    outputs; infinities and NaN propagate the same way in both."""
    attention = _module(rope)
    rows = 64
    g = torch.Generator(device=DEV).manual_seed(zlib.crc32(kind.encode()) + rope)
    projection = _special_heads(kind, rows, g).reshape(rows, PROJ_WIDTH).to(torch.bfloat16)
    positions = _positions(rows, seed=zlib.crc32(kind.encode()))
    qkv_out, gate_out = op(*_call_args(attention, projection, positions, False))
    want_qkv, want_gate = _builtin_prep(attention, projection.clone(), positions)
    _assert_bits_equal(qkv_out, want_qkv, f"qkv {kind}")
    _assert_bits_equal(gate_out, want_gate.reshape(rows, -1), f"gate {kind}")


def test_the_builtin_prep_here_runs_the_kernels_the_entry_reproduces() -> None:
    """The adoption-time check the target applies before selecting ``preattn_prep``."""
    assert target_attention._builtin_prep_kernel_mismatch() is None


@pytest.mark.parametrize("pdl", (False, True), ids=("plain", "pdl"))
@pytest.mark.parametrize("rope", (True, False), ids=("rope", "nope"))
def test_the_compiled_kernel_flushes_and_waits_before_reading(rope, pdl) -> None:
    """PTX of the launched kernel: the RoPE FMAs and sine products carry
    ``.ftz``; with PDL, ``griddepcontrol.wait`` comes before the first global
    load and no ``launch_dependents`` is emitted."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.ops.attention.preattn_prep import (  # noqa: E501
        _preattn_prep_kernel,
    )

    attention = _module(rope)
    projection = _projection(1, seed=1)
    positions = _positions(1, seed=1)
    qkv_out = torch.empty(1, OUT_WIDTH, dtype=torch.bfloat16, device=DEV)
    gate_out = torch.empty(1, Q_WIDTH, dtype=torch.bfloat16, device=DEV)
    compiled = _preattn_prep_kernel[(1, CKPT_Q_HEADS + CKPT_KV_HEADS)](
        projection,
        qkv_out,
        gate_out,
        attention.rotary_emb.rotary_cos_sin if rope else None,
        positions.reshape(-1) if rope else None,
        projection.stride(0),
        qkv_out.stride(0),
        gate_out.stride(0),
        CKPT_EPS,
        CKPT_Q_SCALE,
        NUM_Q_HEADS=CKPT_Q_HEADS,
        NUM_KV_HEADS=CKPT_KV_HEADS,
        QKV_PERMUTED=False,
        APPLY_ROPE=rope,
        LAUNCH_WITH_PDL=pdl,
        num_warps=1,
        launch_pdl=pdl,
    )
    ptx = compiled.asm["ptx"]
    if rope:
        assert ptx.count("fma.rn.ftz.f32") >= 8 and ptx.count("mul.rn.ftz.f32") >= 8
    else:
        assert "fma.rn.ftz.f32" not in ptx
    assert "launch_dependents" not in ptx
    first_load = ptx.index("ld.global")
    if pdl:
        assert ptx.index("griddepcontrol.wait") < first_load
    else:
        assert "griddepcontrol.wait" not in ptx


def test_bit_identity_with_the_one_row_low_latency_path() -> None:
    """The low-latency weight layout's own one-row prep (views into the
    projection, in-place norms) is the other built-in path at M = 1."""
    from tensorrt_llm._torch.models.musespark_open_qkv_layout import preprocess_one_row

    attention = _module(True)
    for seed in range(64):
        projection = _projection(1, seed=900 + seed)[:, _permutation()].contiguous()
        positions = _positions(1, seed=900 + seed)
        qkv_out, gate_out = op(*_call_args(attention, projection, positions, True))
        want_qkv, _, _, want_gate = preprocess_one_row(attention, projection.clone(), positions)
        _assert_bits_equal(qkv_out, want_qkv, "qkv")
        _assert_bits_equal(gate_out, want_gate, "gate")


@pytest.mark.parametrize("rows", (1, 4, 8, 9))
@pytest.mark.parametrize("permuted", (False, True), ids=("ckpt_layout", "ll_layout"))
@pytest.mark.parametrize("rope", (True, False), ids=("rope", "nope"))
def test_the_adopted_attention_runs_the_entry_on_decode_rows(rope, permuted, rows, monkeypatch):
    """The target's attention: with the entry handed in (decode) and at most 8
    rows it calls the entry, otherwise the built-in prep; the bits are the
    built-in prep's either way."""
    builtin = _attention(ROPE_LAYER if rope else NOPE_LAYER)
    adopted = target_attention.MuseSparkOpenTepAttention.adopt(
        _attention(ROPE_LAYER if rope else NOPE_LAYER)
    )
    projection = _projection(rows, seed=31 * rows + rope)
    positions = _positions(rows, seed=31 * rows + rope)
    want_qkv, want_gate = _builtin_prep(builtin, projection.clone(), positions)

    operand = projection
    if permuted:
        perm = _permutation()
        adopted._qkv_layout_inverse = torch.argsort(perm)
        operand = projection[:, perm].contiguous()
    calls = []
    original = type(op).__call__

    def counted(self, *args, **kwargs):
        # The target always launches with PDL (a precondition of its compositions).
        assert args[8] is True
        calls.append(args[0].shape[0])
        return original(self, *args, **kwargs)

    monkeypatch.setattr(type(op), "__call__", counted)
    adopted._preattn_prep = op
    try:
        got_qkv, got_gate = _builtin_prep(adopted, operand.clone(), positions)
    finally:
        adopted._preattn_prep = None
    assert calls == ([rows] if rows <= 8 else [])
    _assert_bits_equal(got_qkv, want_qkv, "qkv")
    _assert_bits_equal(got_gate.reshape(rows, -1), want_gate.reshape(rows, -1), "gate")

    # Without the entry (a prefill step) the adopted module is the built-in one.
    calls.clear()
    got_qkv, _ = _builtin_prep(adopted, operand.clone(), positions)
    assert calls == []
    _assert_bits_equal(got_qkv, want_qkv, "qkv without the entry")


def test_cuda_graph_capture_and_replay() -> None:
    attention = _module(True)
    static_in = _projection(8, seed=8)
    positions = _positions(8, seed=8)
    args = _call_args(attention, static_in, positions, False)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            op(*args)
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_qkv, graph_gate = op(*args)
    static_in.copy_(_projection(8, seed=18))
    positions.copy_(_positions(8, seed=18))
    graph.replay()
    torch.cuda.synchronize()
    eager_qkv, eager_gate = op(*args)
    _assert_bits_equal(graph_qkv, eager_qkv, "qkv graph vs eager")
    _assert_bits_equal(graph_gate, eager_gate, "gate graph vs eager")


@pytest.mark.parametrize("rope", (True, False), ids=("rope", "nope"))
def test_the_pdl_launch_computes_the_same_bits(rope) -> None:
    """Eagerly and in a CUDA graph right behind the projection GEMM."""
    attention = _module(rope)
    weight = torch.randn(PROJ_WIDTH, 8192, device=DEV, dtype=torch.bfloat16) * 0.02
    x = torch.randn(8, 8192, device=DEV, dtype=torch.bfloat16)
    positions = _positions(8, seed=11)

    def step(enable_pdl):
        projection = torch.nn.functional.linear(x, weight)
        return op(*_call_args(attention, projection, positions, False), enable_pdl)

    want_qkv, want_gate = step(False)
    got_qkv, got_gate = step(True)
    _assert_bits_equal(got_qkv, want_qkv, "qkv pdl eager")
    _assert_bits_equal(got_gate, want_gate, "gate pdl eager")
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        step(True)
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_qkv, graph_gate = step(True)
    x.copy_(torch.randn(8, 8192, device=DEV, dtype=torch.bfloat16))
    graph.replay()
    torch.cuda.synchronize()
    want_qkv, want_gate = step(False)
    _assert_bits_equal(graph_qkv, want_qkv, "qkv pdl graph")
    _assert_bits_equal(graph_gate, want_gate, "gate pdl graph")


def test_wrong_variants_land_outside_the_band() -> None:
    """Mis-readings that keep the shapes -- NeoX pairing for GPT-J, RoPE
    skipped, gate and query halves swapped, no query scale -- each fail the
    band the correct result passes."""
    attention = _module(True)
    projection = _projection(64, seed=77)
    positions = _positions(64, seed=77)
    args = _call_args(attention, projection, positions, False)
    ref_qkv, ref_gate = op.reference(*args)
    qkv_out, gate_out = op(*args)
    _assert_within_reference(qkv_out, gate_out, ref_qkv, ref_gate)

    # NeoX pairing: rotate (x[i], x[i + 32]) instead of (x[2i], x[2i + 1]).
    unrotated, _ = op.reference(projection, None, None, *args[3:])
    table = args[1][positions.reshape(-1).long()].double()
    cos, sin = table[:, None, 0, :], table[:, None, 1, :]
    heads = unrotated[:, : Q_WIDTH + KV_WIDTH].reshape(64, -1, HEAD_DIM).double()
    first, second = heads[..., :32], heads[..., 32:]
    neox = torch.cat((first * cos - second * sin, second * cos + first * sin), -1)
    neox = torch.cat(
        [neox.reshape(64, -1).to(torch.bfloat16), unrotated[:, Q_WIDTH + KV_WIDTH :]], -1
    )

    q_gate = projection[:, : 2 * Q_WIDTH].reshape(64, CKPT_Q_HEADS, 2, HEAD_DIM)
    swapped = torch.cat([q_gate.flip(2).reshape(64, -1), projection[:, 2 * Q_WIDTH :]], -1)
    variants = {
        "gate and query halves swapped": op.reference(swapped, *args[1:])[0],
        "no query scale": op.reference(*args[:6], 1.0, args[7])[0],
        "RoPE skipped on a RoPE layer": unrotated,
        "NeoX pairing": neox,
    }
    for name, wrong in variants.items():
        with pytest.raises(AssertionError):
            _assert_within_reference(wrong, gate_out, ref_qkv, ref_gate)
        print(f"  {name}: rejected")


def test_malformed_operands_raise_rather_than_return() -> None:
    attention = _module(True)
    projection = _projection(4, seed=3)
    positions = _positions(4, seed=3)
    args = list(_call_args(attention, projection, positions, False))
    with pytest.raises(ValueError):
        op(projection.float(), *args[1:])
    with pytest.raises(ValueError):
        op(projection[:, :-64], *args[1:])
    with pytest.raises(ValueError):
        op(projection, args[1], positions[:, :3], *args[3:])
    with pytest.raises(ValueError):
        op(projection, None, positions, *args[3:])
    with pytest.raises(ValueError):
        op(projection, args[1].reshape(args[1].shape[0], -1), positions, *args[3:])


def test_is_valid_refuses_positions_outside_the_table() -> None:
    attention = _module(True)
    projection = _projection(2, seed=5)
    positions = torch.tensor([[0, CKPT_MAX_POSITIONS]], dtype=torch.int32, device=DEV)
    args = _call_args(attention, projection, positions, False)
    with pytest.raises(ValueError, match="outside the RoPE table"):
        op.is_valid(*args)


# --------------------------------------------------------------------------
# Microbenchmark: printed, never gated.
# --------------------------------------------------------------------------
_TIMING = os.environ.get("MODELING_V2_TIMING", "0") == "1"


def _graph_time_us(fn, replays: int = 200) -> float:
    """Median-of-5 replay time of a CUDA graph capturing ``fn`` once, in microseconds."""
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    for _ in range(20):
        graph.replay()
    torch.cuda.synchronize()
    samples = []
    for _ in range(5):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(replays):
            graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000.0 / replays)
    return sorted(samples)[2]


@pytest.mark.skipif(not _TIMING, reason="set MODELING_V2_TIMING=1 to run the microbenchmark")
def test_microbenchmark_against_the_builtin_prep() -> None:
    """Per-layer and per-step (46 RoPE + 16 NoPE layers) prep cost, in CUDA graphs.

    ``prep`` graphs hold only the prep; ``gemm+prep`` graphs put each layer's
    BF16 QKV GEMM (distinct weights per layer, so they stream from HBM) in front
    of it, as in the decoder layer.
    """
    rope_mod, nope_mod = _module(True), _module(False)
    layers = [rope_mod if (62 - i - 1) % 4 else nope_mod for i in range(62)]
    assert sum(m is nope_mod for m in layers) == 16
    hidden = 8192
    weights = [
        torch.randn(PROJ_WIDTH, hidden, device=DEV, dtype=torch.bfloat16) * 0.01 for _ in range(62)
    ]
    print()
    for rows in (1, 4, 6, 8):
        projection = _projection(rows, seed=rows)
        positions = _positions(rows, seed=rows)
        x = torch.randn(rows, hidden, device=DEV, dtype=torch.bfloat16)
        perm = _permutation()
        projection_ll = projection[:, perm].contiguous()

        def builtin_one(mod):
            return _builtin_prep(mod, projection, positions)

        def fused_one(mod):
            return op(*_call_args(mod, projection, positions, False))

        def fused_pdl_one(mod):
            return op(*_call_args(mod, projection, positions, False), True)

        res = {}
        for name, fn in (
            ("builtin", builtin_one),
            ("fused", fused_one),
            ("fused+pdl", fused_pdl_one),
        ):
            res[f"{name} rope layer"] = (
                _graph_time_us(lambda fn=fn: [fn(rope_mod) for _ in range(20)]) / 20
            )
            res[f"{name} nope layer"] = (
                _graph_time_us(lambda fn=fn: [fn(nope_mod) for _ in range(20)]) / 20
            )
            res[f"{name} 62-layer step"] = _graph_time_us(lambda fn=fn: [fn(m) for m in layers])

            def gemm_step(fn=fn):
                for w, m in zip(weights, layers):
                    qkv = torch.nn.functional.linear(x, w)
                    if fn is builtin_one:
                        _builtin_prep(m, qkv, positions)
                    else:
                        op(*_call_args(m, qkv, positions, False), fn is fused_pdl_one)

            res[f"{name} 62-layer gemm+prep"] = _graph_time_us(gemm_step, replays=50)
        res["gemm only 62-layer"] = _graph_time_us(
            lambda: [torch.nn.functional.linear(x, w) for w in weights], replays=50
        )
        if rows == 1:
            from tensorrt_llm._torch.models.musespark_open_qkv_layout import preprocess_one_row

            buf = projection_ll.clone()
            res["ll one-row rope layer"] = (
                _graph_time_us(
                    lambda: [preprocess_one_row(rope_mod, buf, positions) for _ in range(20)]
                )
                / 20
            )
            res["ll one-row 62-layer step"] = _graph_time_us(
                lambda: [preprocess_one_row(m, buf, positions) for m in layers]
            )
        for key, value in res.items():
            print(f"  M={rows} {key:<28} {value:8.2f} us")
        saved = res["builtin 62-layer step"] - res["fused 62-layer step"]
        saved_g = res["builtin 62-layer gemm+prep"] - res["fused 62-layer gemm+prep"]
        print(f"  M={rows} saved per step: prep-only {saved:.1f} us, with GEMM {saved_g:.1f} us")
        saved_p = res["builtin 62-layer gemm+prep"] - res["fused+pdl 62-layer gemm+prep"]
        print(f"  M={rows} saved per step with GEMM, PDL launch: {saved_p:.1f} us")
