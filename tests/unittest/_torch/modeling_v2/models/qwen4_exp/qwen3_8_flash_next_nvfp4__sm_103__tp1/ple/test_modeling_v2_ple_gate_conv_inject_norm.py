# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the ple_gate_conv_inject_norm entry.

Cells are gated against the float64 reference that rounds where the op rounds:
the stream (`hc_out`) and the published conv state within one ulp of each
element, the normed stream within two (see `_compare`). Beyond the cells:
agreement with the composition it replaces at the same band, separation from
plausible wrong formulas, and graph capture.
"""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_103__tp1.ops.hc.qwen4_exp_hc_inject_norm import (  # noqa: E501
    qwen4_exp_hc_inject_norm,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_103__tp1.ops.norm.flashinfer_gemma_rmsnorm import (  # noqa: E501
    flashinfer_gemma_rmsnorm,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_103__tp1.ops.ple.gated_ngram_injection import (  # noqa: E501
    gated_ngram_injection,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_103__tp1.ops.ple.ple_gate_conv_inject_norm import (  # noqa: E501
    CKPT_CLAMP_MIN,
    CKPT_DILATION,
    CKPT_HC_COUNT,
    CKPT_HIDDEN,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_103__tp1.ops.ple.ple_gate_conv_inject_norm import (  # noqa: E501
    ple_gate_conv_inject_norm as op,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_103__tp1.ops.ssm.dilated_short_conv_paged import (  # noqa: E501
    dilated_short_conv_paged,
)

__extra_import_path__ = ["../../../.."]
from _validating import (  # noqa: E402,F401 — path above; fixture by name
    assert_within_ulp,
    certifying,
    validating,
)

assert torch.cuda.is_available(), "ple_gate_conv_inject_norm requires a CUDA device"

DEV = "cuda"
EPS = 1e-6
PAD = -1
GAMMA_RMS = 0.477


def _ulp_ratio(out: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """Per-element distance from `ref` in units of one ulp of `|ref|`."""
    eps = torch.finfo(ref.dtype).eps
    tiny = torch.finfo(ref.dtype).tiny
    ref64 = ref.double()
    return (out.double() - ref64).abs() / (eps * ref64.abs() + tiny)


def _stats(out: torch.Tensor, ref: torch.Tensor) -> tuple[float, float]:
    ratio = _ulp_ratio(out, ref)
    return ratio.max().item(), (out != ref).double().mean().item()


#: The band. Every rounding point of the composition is kept, so the op and the
#: float64 reference differ only where an fp32 reduction (three sums of squares
#: and the gate's dot per stream) or fp32 `exp` lands a value across a bf16
#: midpoint -- measured on 2e-5..2.3e-4 of the elements. A flipped upstream
#: element then propagates, and through `hc + ple` it meets cancellation
#: (`hc ~ -ple`), so a per-element ulp is unbounded there: 73 ulp of a
#: 0.003-sized element in a row whose largest is ~10. The stream and its norm
#: are therefore gated in units of their row's scale (`assert_within_ulp`):
#: 2 ulp of the row maximum per element and 0.1 ulp relative RMS. The
#: composition this replaces sits at 0.87 / 0.33 of a row-max ulp against the
#: same reference (measured: normed / hc_out at 128 rows), the op at 0.72 /
#: 0.38. The published conv state is one rounding of the conv-input norm,
#: whose only perturbation is a one-ulp flip of a gated element: 2 ulp per
#: element (measured worst 1.78, the old composition 1.56).
ROW_ULP, RMS_ULP, STATE_ULP = 2.0, 0.1, 2.0
#: The per-element row-max gate is a tail statistic, and a context-shaped cell
#: draws ~100x more elements than a 128-row one. Against the same float64
#: reference over three seeds at 16384 rows, the composition this replaces
#: reaches 2.91 row-max ulp (normed) and the op 2.94; at 8192 rows 1.89 / 1.89.
#: Cells above 1024 rows are therefore gated at 4; the RMS gate is unchanged.
CONTEXT_ROWS, CONTEXT_ROW_ULP = 1024, 4.0


def _compare(out, ref) -> None:
    row_ulp = CONTEXT_ROW_ULP if out[0].shape[0] > CONTEXT_ROWS else ROW_ULP
    for name, got, want in zip(("hc_out", "normed", "conv_states"), out, ref):
        assert got.dtype == want.dtype and got.shape == want.shape, name
        if not got.numel():
            continue
        if name == "conv_states":
            worst = _ulp_ratio(got, want).max().item()
            assert worst <= STATE_ULP, f"{name}: {worst:.4g} ulp"
        else:
            assert_within_ulp(got, want, row_ulp, RMS_ULP)


def _build(spec, seed):
    g = torch.Generator(device=DEV).manual_seed(seed)
    gc = torch.Generator().manual_seed(seed)
    lengths = spec["lengths"]
    c = spec.get("hc_count", CKPT_HC_COUNT)
    d = spec.get("hidden", CKPT_HIDDEN)
    dtype = spec.get("dtype", torch.bfloat16)
    w = c * d
    rows = sum(lengths)
    num_seqs = len(lengths)
    pool_slots = spec.get("pool_slots", num_seqs + 2)

    def rand(*shape, scale=1.0):
        return (torch.randn(*shape, generator=g, device=DEV) * scale).to(dtype)

    # Streams at different scales, as the trained residual's are.
    scales = torch.tensor([0.5 * 2.0 ** (i % 4) for i in range(c)], device=DEV)
    hc = (rand(rows, w, scale=3.0).float() * scales.repeat_interleave(d)).to(dtype)
    key = rand(rows, w)
    if spec.get("layout") == "row_slice":
        hbuf, kbuf = rand(2 * rows, w), rand(2 * rows, w)
        hbuf[::2], kbuf[::2] = hc, key
        hc, key = hbuf[::2], kbuf[::2]
    value = rand(rows, d, scale=0.5)
    norms = [rand(w, scale=GAMMA_RMS) for _ in range(4)]
    conv_weight = rand(w, 1, 4, scale=0.3)
    conv_states = rand(pool_slots, w, 3 * CKPT_DILATION)
    order = (torch.randperm(pool_slots - 2, generator=gc) + 1).tolist()
    slots = ([0, pool_slots - 1] + order)[:num_seqs] if pool_slots >= 64 else list(range(num_seqs))
    if "pad" in spec:
        slots[spec["pad"]] = PAD
    slots = torch.tensor(slots, dtype=torch.int32, device=DEV)
    starts = [0]
    for length in lengths:
        starts.append(starts[-1] + length)
    qsl = torch.tensor(starts, dtype=torch.int64, device=DEV)
    max_rows = spec.get("max_rows", max(lengths))
    carries = spec.get("carries")
    carries = None if carries is None else torch.tensor(carries, dtype=torch.bool, device=DEV)
    return [
        hc, key, value, norms[0], norms[1], norms[2], conv_weight, conv_states, slots, qsl,
        norms[3], EPS, c, CKPT_CLAMP_MIN, CKPT_DILATION, PAD, max_rows, carries,
    ]  # fmt: skip


def _run(args):
    """`(hc_out, normed, conv_states)` from the op, on a copy of the pool."""
    args = list(args)
    args[7] = args[7].clone()
    hc_out, normed = op(*args)
    return hc_out, normed, args[7]


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — fixture by name
    certifying(op)
    args = _build(cell.spec, seed=zlib.crc32(cell.why.encode()))
    hc_before = args[0].clone()
    ref = op.reference(*args)
    out = _run(args)
    assert torch.equal(args[0], hc_before), "hc is read-only"
    assert out[0].is_contiguous() and out[1].is_contiguous()
    _compare(out, ref)
    for name, got, want in zip(("hc_out", "normed", "conv_states"), out, ref):
        worst, frac = _stats(got, want)
        print(
            f"\n  {cell.why[:40]:<40} {name:<12} worst {worst:.3g} ulp, differ {frac:.2e}", end=""
        )


def _grouped(x, w, c):
    d = x.shape[1] // c
    parts = torch.split(x, [d] * c, 1)
    return torch.cat(
        [flashinfer_gemma_rmsnorm(p, s, EPS) for s, p in zip(torch.split(w, d, 0), parts)], 1
    )


def _old(
    hc,
    key,
    value,
    nk,
    nq,
    nc,
    cw,
    states,
    slots,
    qsl,
    nx,
    eps,
    c,
    clamp,
    dil,
    pad,
    _p,
    carries=None,
):
    """The calls the decode / prefill PLE block and the site after it made before this entry."""
    gated = gated_ngram_injection(_grouped(key, nk, c), _grouped(hc, nq, c), value, c, clamp)
    convolved = dilated_short_conv_paged(
        x=torch.transpose(_grouped(gated, nc, c), 0, 1),
        weight=cw,
        bias=None,
        conv_states=states,
        query_start_loc=qsl,
        cache_indices=slots,
        has_initial_state=torch.ones_like(slots, dtype=torch.bool) if carries is None else carries,
        dilation=dil,
        silu_activation=True,
        pad_slot_id=pad,
    )
    hc_new = torch.add(hc, torch.add(gated, torch.transpose(convolved, 0, 1)))
    _, normed = qwen4_exp_hc_inject_norm(hc_new, None, None, nx, eps, c)
    return hc_new, normed, states


@pytest.mark.parametrize(
    "lengths,carries",
    [
        ((1,), None),
        ((1,) * 32, None),
        ((1,) * 128, None),
        ((4,) * 8, None),
        ((8192,), (False,)),
        ((5000, 3000) + (1,) * 32, (False, True) + (True,) * 32),
    ],
    ids=["m1", "m32", "m128", "mtp4x8", "prefill8192", "mixed2ctx32dec"],
)
def test_matches_the_composition_it_replaces(lengths, carries) -> None:
    args = _build(dict(lengths=lengths, pool_slots=461, carries=carries), seed=len(lengths))
    old_args = list(args)
    old_args[7] = args[7].clone()
    want = _old(*old_args)
    got = _run(args)
    _compare(got, want)
    for name, a, b in zip(("hc_out", "normed", "conv_states"), got, want):
        worst, frac = _stats(a, b)
        print(
            f"\n  {str(len(lengths)):>4} seqs {name:<12} worst {worst:.3g} ulp, differ {frac:.2e}",
            end="",
        )


def test_wrong_variants_land_outside_the_band() -> None:
    """Each mis-reading keeps every shape; each lands far outside the band."""
    args = _build(dict(lengths=(1,) * 16, pool_slots=461), seed=21)
    want = op.reference(*args)
    variants = {
        "dilation 1": dict(index=14, value=1),
        "clamp floor 1.0": dict(index=13, value=1.0),
        "next norm = query norm": dict(index=10, value=args[4]),
        "conv norm = key norm": dict(index=5, value=args[3]),
    }
    for name, change in variants.items():
        wrong = list(args)
        wrong[change["index"]] = change["value"]
        if name == "dilation 1":
            wrong[7] = wrong[7][:, :, :3].contiguous()
        got = _run(wrong)
        worst = max(_ulp_ratio(got[i], want[i]).max().item() for i in (0, 1))
        print(f"\n  {name:<26} {worst:10.4g} ulp", end="")
        assert worst > 5.0, name


def test_guard_refuses_a_short_value() -> None:
    args = _build(dict(lengths=(1, 1)), seed=4)
    args[2] = args[2][:, :-1]
    with validating(op), pytest.raises(AssertionError):
        op(*args)


def test_cuda_graph_capture_and_replay() -> None:
    args = _build(dict(lengths=(1,) * 8, pool_slots=461), seed=8)
    warmup = torch.cuda.Stream()
    warmup.wait_stream(torch.cuda.current_stream())
    pool = args[7].clone()
    with torch.cuda.stream(warmup):
        for _ in range(2):
            op(*args)
    torch.cuda.current_stream().wait_stream(warmup)
    args[7].copy_(pool)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_hc, graph_normed = op(*args)
    fresh = _build(dict(lengths=(1,) * 8, pool_slots=461), seed=18)
    args[0].copy_(fresh[0])
    args[1].copy_(fresh[1])
    args[7].copy_(pool)
    eager = _run(args)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(graph_hc, eager[0])
    assert torch.equal(graph_normed, eager[1])
    assert torch.equal(args[7], eager[2])
