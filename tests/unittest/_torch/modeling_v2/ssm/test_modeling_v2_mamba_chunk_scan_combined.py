# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the mamba_chunk_scan_combined entry (Mamba2 SSD prefill).

Every cell is driven the way the target calls it: the metadata comes from the
engine's own `Mamba2Metadata.prepare`, the initial states are gathered from a
pool laid out like the V2 hybrid manager's (strided slot axis, the layer's
sub-page of each slot), and the final states are scattered back -- the exact
glue the entry's `note` prescribes. The first call per autotune key in a
process benchmarks the kernels, so the cold-call test re-runs this file as a
child: `python <this file> cold-call`.
"""

import os
import subprocess
import sys
import zlib
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from triton.compiler.errors import CompilationError

from tensorrt_llm._torch._experimental.modeling_v2.catalog.ssm.mamba_chunk_scan_combined import (  # noqa: E501
    CHUNK_SIZE,
    DSTATE,
    HEADDIM,
    NGROUPS,
    NHEADS,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.ssm.mamba_chunk_scan_combined import (  # noqa: E501
    mamba_chunk_scan_combined as op,
)
from tensorrt_llm._torch.modules.mamba.mamba2_metadata import Mamba2Metadata

__extra_import_path__ = [".."]
from _validating import certifying, validating  # noqa: E402,F401 — path above; fixture by name

assert torch.cuda.is_available(), "mamba_chunk_scan_combined requires a CUDA device"

DEV = "cuda"
#: One bf16 unit, the `_validating.ULP` convention; every band below is in it.
U = 2.0**-8
#: fp16's subnormal spacing: below it an fp16 result cannot carry a relative band.
FP16_FLOOR = 2.0**-24
DT_LIMIT = (0.0, float("inf"))

#: `_gate` bands (element, relative RMS) in units of U; see `_gate`.
Y_BAND = (4.0, 1.0)
STATE_BAND = (2.0, 0.5)


def _gate(got: torch.Tensor, want: torch.Tensor, mag: torch.Tensor, band: tuple) -> tuple:
    """`|got - want| <= band[0] * U * mag` elementwise, and the RMS of the
    difference <= `band[1] * U` relative to the RMS of `want`.

    `mag` is the reference run on |x|, |B|, |C|, |D| and |initial_states|:
    an elementwise bound on the sum of the magnitudes of every term the output
    is built from. It is the scale a rounding error lives at, where `want`
    itself is not: a row whose D skip cancels its C.S term (D runs negative
    here) has a tiny output and a full-size error, 115 U of its own row
    maximum while 1.9 U of `mag` -- measured.

    Derivation, y: the kernel rounds three operands to bf16 before a tensor
    core dot -- the intra-chunk `C.B * decay * dt` weights, the scaled B of the
    chunk states, the carried state before `C @ S` -- each at most U of the
    terms it scales, and rounds `out` once (U); the reference rounds once more
    (U): 4 at worst on the element, measured at most 1.98 over the cells. The
    RMS is half a unit or so (0.32 to 0.69 measured), gated at 1.

    Derivation, states (fp32 out): one bf16 rounding of the scaled B per term
    (U) plus the fp32 decay from chunk cumsums; measured at most 1.03 on the
    element and 0.24 RMS, gated at 2 and 0.5. An fp16 `state_dtype` adds the
    carried states' fp16 roundings, 2^-11 each (measured 1.29 on the element),
    and fp16's subnormal floor for states below 2^-14.
    """
    floor = FP16_FLOOR if got.dtype == torch.float16 else 0.0
    diff = (got.float() - want.float()).abs()
    allowed = band[0] * U * mag.float() + floor
    worst = (diff / allowed.clamp_min(1e-30)).max().item()
    assert bool((diff <= allowed).all()), f"element error {worst:.3f} of the {band[0]} U band"
    rel_rms = (diff.pow(2).mean().sqrt() / want.float().pow(2).mean().sqrt()).item() / U
    assert rel_rms <= band[1], f"relative RMS {rel_rms:.3f} U exceeds {band[1]} U"
    return worst * band[0], rel_rms


def _seed(text: str) -> int:
    return zlib.crc32(text.encode()) % 2**31


def _metadata(lengths, initial) -> Mamba2Metadata:
    """The engine's own metadata for one all-context step."""
    n = len(lengths)
    mm = Mamba2Metadata(max_batch_size=n, chunk_size=CHUNK_SIZE)
    seq_lens = torch.tensor(lengths, dtype=torch.int32)
    mm.prepare(
        SimpleNamespace(
            seq_lens=seq_lens,
            num_contexts=n,
            seq_lens_cuda=seq_lens.to(DEV),
            num_ctx_tokens=sum(lengths),
            kv_cache_manager=None,
            request_ids=None,
            kv_cache_params=SimpleNamespace(
                num_cached_tokens_per_seq=[17 if carried else 0 for carried in initial]
            ),
        )
    )
    return mm


def _case(spec: dict, seed: int) -> SimpleNamespace:
    """The layer's weights, one step's activations, a pool and the metadata.

    The weights are drawn from the checkpoint's ranges: A_log in [-3.2, 6.1]
    (A down to -457), dt_bias in [-8.1, -1.3], D in [-1.13, 3.94], stored bf16.
    x/B/C are SiLU outputs (the conv's activation), dt the raw in_proj slice.
    """
    gen = torch.Generator(device=DEV).manual_seed(seed)
    lengths, initial = spec["lengths"], spec["initial"]
    n, total = len(lengths), sum(lengths)
    param_dtype = spec.get("param_dtype", torch.bfloat16)
    nheads, ngroups = spec.get("nheads", NHEADS), spec.get("ngroups", NGROUPS)

    def uniform(lo, hi):
        return torch.empty(nheads, device=DEV).uniform_(lo, hi, generator=gen)

    A = -torch.exp(uniform(-3.2, 6.1))
    dt_bias = uniform(-8.1, -1.3).to(torch.bfloat16).to(param_dtype)
    D = uniform(-1.13, 3.94).to(torch.bfloat16).to(param_dtype)

    xw, bw = nheads * HEADDIM, ngroups * DSTATE
    width = xw + 2 * bw  # x | B | C: 4608 at tep4, 18432 at dep4
    conv = F.silu(torch.randn(total, width, device=DEV, generator=gen) * 2).to(torch.bfloat16)
    proj = torch.randn(total, xw + width + nheads, device=DEV, generator=gen)
    dt = proj.to(torch.bfloat16)[:, -nheads:]  # z | xBC | dt: a strided slice
    if spec.get("layout", "token_major") == "channel_major":
        cm = conv.t().contiguous()  # the channel-major conv kernel's [width, T]
        x = cm[:xw].view(nheads, HEADDIM, total).permute(2, 0, 1)
        B = cm[xw : xw + bw].view(ngroups, DSTATE, total).permute(2, 0, 1)
        C = cm[xw + bw :].view(ngroups, DSTATE, total).permute(2, 0, 1)
    else:
        x = conv[:, :xw].view(total, nheads, HEADDIM)
        B = conv[:, xw : xw + bw].view(total, ngroups, DSTATE)
        C = conv[:, xw + bw :].view(total, ngroups, DSTATE)

    slots_total = spec.get("pool_slots", max(8, n + 3))
    layers = spec.get("pool_layers", 2)
    base = torch.empty(
        slots_total,
        layers,
        nheads,
        HEADDIM,
        DSTATE,
        device=DEV,
        dtype=spec.get("pool_dtype", torch.float16),
    )
    base.fill_(1.5)  # the other layers' sub-pages: a value the op must not touch
    pool = base[:, layers // 2]
    pool.copy_(torch.randn(pool.shape, device=DEV, generator=gen) * 3)
    perm = torch.randperm(slots_total, generator=torch.Generator().manual_seed(seed))
    slots = perm[:n].to(torch.int32).to(DEV)
    return SimpleNamespace(
        lengths=lengths,
        A=A,
        dt_bias=dt_bias,
        D=D,
        x=x,
        B=B,
        C=C,
        dt=dt,
        base=base,
        pool=pool,
        slots=slots,
        mm=_metadata(lengths, initial),
    )


def _call_args(case: SimpleNamespace, state_dtype) -> list:
    """`__call__`'s arguments as the target builds them: gather, zero-fill, out."""
    n = len(case.lengths)
    mm = case.mm
    init = None
    if mm.use_initial_states:
        init = case.pool[case.slots].to(torch.float32 if state_dtype is None else state_dtype)
        init.masked_fill_(~mm.has_initial_states[:n].view(n, 1, 1, 1), 0)
    out = torch.empty_like(case.x[None], memory_format=torch.contiguous_format)
    return [
        case.x[None],
        case.dt[None],
        case.A,
        case.B[None],
        case.C[None],
        CHUNK_SIZE,
        case.D,
        None,
        case.dt_bias,
        init,
        mm.seq_idx,
        mm.chunk_indices,
        mm.chunk_offsets,
        mm.cu_seqlens[: n + 1],
        True,
        DT_LIMIT,
        out,
        False,
        True,
        state_dtype,
    ]


def _magnitude(args: list) -> tuple:
    """`reference` on the absolute values: `_gate`'s scale, in fp32."""
    mags = list(args)
    for i in (0, 3, 4, 6, 9):  # x, B, C, D, initial_states
        if mags[i] is not None:
            mags[i] = mags[i].abs()
    mags[16] = torch.empty((), dtype=torch.float32)  # out: only its dtype is read
    mags[19] = torch.float32
    return op.reference(*mags)


def _check(args: list, final: torch.Tensor) -> None:
    ref_y, ref_final = op.reference(*args)
    mag_y, mag_final = _magnitude(args)
    out = args[16]
    assert final.dtype == ref_final.dtype and final.shape == ref_final.shape
    measured = _gate(out, ref_y, mag_y, Y_BAND) + _gate(final, ref_final, mag_final, STATE_BAND)
    print("y element/RMS, state element/RMS (U):", ", ".join(f"{v:.3f}" for v in measured))


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — pytest resolves the fixture by this name
    certifying(op)
    case = _case(cell.spec, _seed(cell.why))
    state_dtype = cell.spec.get("state_dtype", torch.float32)
    args = _call_args(case, state_dtype)
    before = case.base.clone()
    final = op(*args)
    assert torch.equal(case.base, before), (
        "the op reads gathered states; it must not touch the pool"
    )
    _check(args, final)
    # The scatter: the one rounding to the pool's dtype, into the named slots only.
    case.pool[case.slots] = final.to(case.pool.dtype)
    expect = before.clone()
    expect[:, case.base.shape[1] // 2][case.slots.long()] = final.to(case.pool.dtype)
    assert torch.equal(case.base, expect)


def test_the_carried_state_is_read() -> None:
    """Discrimination: zeroing a carried row's initial state moves that row only."""
    spec = dict(lengths=(64, 100), initial=(True, True))
    case = _case(spec, seed=5)
    args = _call_args(case, torch.float32)
    op(*args)
    with_carry = args[16].clone()
    args[9] = args[9].clone()
    args[9][1] = 0
    op(*args)
    without = args[16]
    assert torch.equal(with_carry[:, :64], without[:, :64])
    assert (with_carry[:, 64:].float() - without[:, 64:].float()).abs().max().item() > 0.1


#: Even heads first, then odd: position j holds a head h with h % G == j // (H / G),
#: so the reference's contiguous grouping reads the interleaved one.
_INTERLEAVED = torch.arange(NHEADS).view(NHEADS // NGROUPS, NGROUPS).t().reshape(-1)


@pytest.mark.parametrize(
    "wrong",
    ["no_softplus", "no_dt_bias", "no_d_skip", "groups_interleaved", "carry_dropped"],
)
def test_band_rejects_a_wrong_reference(wrong: str) -> None:
    """The gate is tight enough to see a plausible misreading of the recurrence."""
    spec = dict(lengths=(100, 300, 37), initial=(True, False, True))
    args = _call_args(_case(spec, seed=6), torch.float32)
    final = op(*args)
    bad = list(args)
    if wrong == "no_softplus":
        bad[14] = False
    elif wrong == "no_dt_bias":
        bad[8] = torch.zeros_like(args[8])
    elif wrong == "no_d_skip":
        bad[6] = torch.zeros_like(args[6])
    elif wrong == "groups_interleaved":  # head h on group h % G rather than h // G
        order = _INTERLEAVED
        bad[0] = args[0][:, :, order]
        bad[1] = args[1][:, :, order]
        bad[2] = args[2][order]
        bad[6] = args[6][order]
        bad[8] = args[8][order]
        bad[9] = args[9][:, order]
    elif wrong == "carry_dropped":
        bad[9] = torch.zeros_like(args[9])
    ref_y, ref_final = op.reference(*bad)
    mag_y, mag_final = _magnitude(args)
    if wrong == "groups_interleaved":
        inverse = torch.argsort(_INTERLEAVED)
        ref_y, ref_final = ref_y[:, :, inverse], ref_final[:, inverse]
    with pytest.raises(AssertionError):
        _gate(args[16], ref_y, mag_y, Y_BAND)
        _gate(final, ref_final, mag_final, STATE_BAND)


def test_the_recommended_glue_issues_no_host_sync() -> None:
    """Gather, zero-fill, the op and the scatter, warm: nothing synchronizes.

    The prefill is not captured, but a hidden sync would serialize every
    Mamba layer of every prefill step behind the host.
    """
    spec = dict(lengths=(100, 300, 37), initial=(True, False, True))
    case = _case(spec, seed=7)
    for _ in range(2):  # compile and autotune outside the check
        op(*_call_args(case, torch.float32))
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        final = op(*_call_args(case, torch.float32))
        case.pool[case.slots] = final.to(case.pool.dtype)
    finally:
        torch.cuda.set_sync_debug_mode("default")


_REJECTED = ("initial_states_dtype", "initial_states_without_chunk_metadata", "out_none")


@pytest.mark.parametrize("case_name", _REJECTED)
def test_op_rejects_loudly_what_is_valid_does_not_restate(case_name: str) -> None:
    """What the op refuses itself -- the reason `is_valid` stays narrow."""
    spec = dict(lengths=(100, 300, 37), initial=(True, False, True))
    args = _call_args(_case(spec, seed=8), torch.float32)
    if case_name == "initial_states_dtype":
        args[9] = args[9].to(torch.float16)  # the scan needs it in state_dtype
        expected = CompilationError
    elif case_name == "initial_states_without_chunk_metadata":
        args[11] = args[12] = None
        expected = AssertionError
    elif case_name == "out_none":
        args[16] = None
        expected = AttributeError
    with pytest.raises(expected):
        with validating(op):
            op(*args)


@pytest.mark.skipif(
    torch.cuda.get_device_capability() != (10, 7), reason="the FlashInfer refusal is sm_107's"
)
def test_flashinfer_ssd_refuses_sm107(monkeypatch) -> None:
    """`TRTLLM_USE_MAMBA_FI_SSD=1` reroutes the entry to FlashInfer, which raises here."""
    monkeypatch.setenv("TRTLLM_USE_MAMBA_FI_SSD", "1")
    spec = dict(lengths=(100, 300, 37), initial=(True, False, True))
    with pytest.raises(ValueError, match="SM107"):
        op(*_call_args(_case(spec, seed=9), torch.float32))


def _guard_spec(hazard: str) -> dict:
    # Without seq_idx and with carried states the op asserts by itself; the
    # silent case is the fresh batch.
    initial = (False, False, False) if hazard == "seq_idx_missing" else (True, False, True)
    return dict(lengths=(100, 300, 37), initial=initial)


def _guard_args(hazard: str) -> list:
    """A fresh argument list exhibiting one silent violation."""
    case = _case(_guard_spec(hazard), seed=10)
    args = _call_args(case, torch.float32)
    if hazard == "seq_idx_missing":
        args[10] = None
    elif hazard == "seq_idx_stale":
        args[10] = _metadata((300, 100, 37), (True, False, True)).seq_idx
    elif hazard == "chunk_metadata_stale":
        stale = _metadata((101, 299, 37), (True, False, True))
        args[11], args[12] = stale.chunk_indices, stale.chunk_offsets
        assert args[11].numel() == case.mm.chunk_indices.numel()
    return args


@pytest.mark.parametrize("hazard", ["seq_idx_missing", "seq_idx_stale", "chunk_metadata_stale"])
def test_guard_refuses_what_the_op_answers_silently_wrong(hazard: str) -> None:
    """First the hazard, unguarded: the op returns and misses the reference.
    Then the guard refuses the same call."""
    args = _guard_args(hazard)
    good = _call_args(_case(_guard_spec(hazard), seed=10), torch.float32)
    final = op(*args)
    ref_y, ref_final = op.reference(*good)
    mag_y, mag_final = _magnitude(good)
    with pytest.raises(AssertionError):
        _gate(args[16], ref_y, mag_y, Y_BAND)
        _gate(final, ref_final, mag_final, STATE_BAND)

    with pytest.raises(AssertionError):
        with validating(op):
            op(*_guard_args(hazard))


def test_guard_refuses_cu_seqlens_past_the_tokens() -> None:
    """A last row claiming tokens past `x` reads past the activations; refused
    before the kernel runs (not driven unguarded: the read is out of bounds)."""
    args = _call_args(_case(_guard_spec("cu_seqlens_past_the_tokens"), seed=10), torch.float32)
    cu = args[13].clone()
    cu[-1] += 13
    args[13] = cu
    with pytest.raises(AssertionError, match="span the packed tokens"):
        with validating(op):
            op(*args)


# The cold-call child: the first call per autotune key in a process must be
# as correct as a warm one. Autotuning re-launches every SSD kernel per
# candidate config; all of them write fresh outputs, so it should be.

CHILD_TIMEOUT_S = 600


def _ssd_autotuners() -> list:
    from tensorrt_llm._torch.modules.mamba import (
        ssd_bmm,
        ssd_chunk_scan,
        ssd_chunk_state,
        ssd_state_passing,
    )

    kernels = (
        ssd_chunk_state._chunk_cumsum_fwd_kernel,
        ssd_chunk_state._chunk_state_fwd_kernel,
        ssd_chunk_state._chunk_state_varlen_kernel,
        ssd_state_passing._state_passing_fwd_kernel,
        ssd_bmm._bmm_chunk_fwd_kernel,
        ssd_chunk_scan._chunk_scan_fwd_kernel,
    )
    tuners = []
    for node in kernels:
        while node is not None and not isinstance(getattr(node, "cache", None), dict):
            node = getattr(node, "fn", None)
        assert node is not None, "an SSD kernel lost its autotuner"
        tuners.append(node)
    return tuners


def _child_cold_call() -> None:
    tuners = _ssd_autotuners()
    assert not any(t.cache for t in tuners), "the child must be the first SSD call in its process"
    case = _case(dict(lengths=(100, 300, 37), initial=(True, False, True)), seed=11)
    args = _call_args(case, torch.float32)
    final = op(*args)
    assert all(len(t.cache) == 1 for t in tuners), "each kernel must have tuned exactly one key"
    _check(args, final)


_CHILD_MODES = {"cold-call": _child_cold_call}


def test_cold_call_matches_the_reference() -> None:
    """The first call of a process, carried states included, within the band."""
    root = str(Path(__file__).resolve().parents[1])  # tests/unittest/_torch/modeling_v2
    existing = os.environ.get("PYTHONPATH")
    env = dict(os.environ, PYTHONPATH=root if not existing else os.pathsep.join([root, existing]))
    proc = subprocess.run(
        [sys.executable, __file__, "cold-call"],
        capture_output=True,
        text=True,
        timeout=CHILD_TIMEOUT_S,
        env=env,
    )
    print(proc.stdout)
    assert proc.returncode == 0, (
        f"child exited {proc.returncode}\n--- stdout ---\n{proc.stdout}\n"
        f"--- stderr ---\n{proc.stderr[-4000:]}"
    )


if __name__ == "__main__":
    _CHILD_MODES[sys.argv[1]]()
