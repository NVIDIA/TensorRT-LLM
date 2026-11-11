# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the glm5_next fp8 tp4 mhc_fused_hc entry.

Each cell runs every tactic ``fused_hc_tactic`` can return at its row count on this device -- the
autotuner's candidates, the cache-miss fallback and ``DETERMINISTIC_TACTIC`` -- against one
reference.
"""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.glm5_next.glm_5_3_flash_fp8__sm_10x__tp4.ops.mhc.mhc_fused_hc import (  # noqa: E501
    DETERMINISTIC_TACTIC,
    HIDDEN,
    default_fused_hc_tactic,
    fused_hc_mma_enabled,
    fused_hc_scratch,
    fused_hc_tactic,
    fused_hc_tactics,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.glm5_next.glm_5_3_flash_fp8__sm_10x__tp4.ops.mhc.mhc_fused_hc import (  # noqa: E501
    mhc_fused_hc as op,
)
from tensorrt_llm._torch.autotuner import AutoTuner, autotune
from tensorrt_llm._utils import get_sm_version

__extra_import_path__ = [".", "../../../.."]
from _glm53_mhc_inputs import (  # noqa: E402
    block_output,
    coefficients,
    generator,
    mixer,
    norm_weight,
    scalars,
    streams,
)
from _validating import assert_within_ulp, certifying, validating  # noqa: E402,F401

pytestmark = pytest.mark.skipif(
    get_sm_version() != 100, reason="the glm5_next fp8 tp4 entries are certified on sm_100"
)

#: Absolute band of the FP32 coefficients; see `_compare`.
COEFF_ATOL = 5e-3
_CUDA = torch.device("cuda")


def _compare(out, ref) -> None:
    """The streams within 3 ulp per element of the row scale and 0.5 ulp relative RMS; ``post``
    and ``comb`` within 5e-3 absolute; the layer input within 4 ulp per element and 1 ulp RMS.

    The streams are the same FP32 sums on both sides, rounded to BF16 once, so they differ only
    in rare one-step rounding flips (at most 2 ulp of the row scale). The coefficients differ by
    more: the FMA backends feed the mixer the FP32 streams before their BF16 rounding, while the
    reference (like the tensor-core backends and the unfused path) uses the stored BF16 streams.
    Each stream element then differs by up to half a BF16 step, which moves a mixer output by
    about 0.15% of its spread; a CPU transcription of that difference at these inputs measured
    at most 6e-4 on ``pre``, 8e-4 on ``post`` and 1.5e-3 on ``comb`` over 3,328 rows, against a
    5e-3 band. The layer input is rounded to BF16 at the same places on both sides, but a ``pre``
    off by 6e-4 flips some of its first rounding: the same transcription measured 2.6 ulp per
    element and 0.35 ulp RMS, with the bound at two BF16 steps (4 ulp). A transposed ``comb``, a
    lost hidden split or a wrong mixer moves every output by tens to hundreds of times its band.
    """
    residual_out, post, comb, layer_input = out
    ref_residual, ref_post, ref_comb, ref_layer_input = ref
    assert post.shape == ref_post.shape and comb.shape == ref_comb.shape
    assert_within_ulp(residual_out, ref_residual, element_ulp=3.0, rms_ulp=0.5)
    torch.testing.assert_close(post, ref_post, rtol=0.0, atol=COEFF_ATOL)
    torch.testing.assert_close(comb, ref_comb, rtol=0.0, atol=COEFF_ATOL)
    assert_within_ulp(layer_input, ref_layer_input, element_ulp=4.0, rms_ulp=1.0)


def _matches(out, ref) -> bool:
    try:
        _compare(out, ref)
    except AssertionError:
        return False
    return True


def _inputs(rows: int, norm: bool, seed: int, std: float = 1.0) -> list:
    """``[x, residual, post, comb, fn, scale, base, norm_weight]`` of one boundary."""
    gen = generator(seed)
    post, comb = coefficients(rows, gen)
    fn, scale, base = mixer(gen)
    x, residual = block_output(rows, gen, std), streams(rows, gen, std)
    return [x, residual, post, comb, fn, scale, base, norm_weight(gen) if norm else None]


def _tactics(rows: int) -> list:
    """Every tactic ``fused_hc_tactic`` can return at ``rows`` rows on this device."""
    mma = fused_hc_mma_enabled()
    sm103 = torch.cuda.get_device_capability() == (10, 3)
    tactics = dict.fromkeys(fused_hc_tactics(rows, HIDDEN, mma, sm103))
    tactics[default_fused_hc_tactic(rows, HIDDEN, mma)] = None
    tactics[DETERMINISTIC_TACTIC] = None
    return list(tactics)


def _run(args: list, tactic):
    return op(*args, tactic, *fused_hc_scratch(args[0].shape[0], tactic, _CUDA))


def _reference(args: list, tactic=DETERMINISTIC_TACTIC):
    return op.reference(*args, tactic, *fused_hc_scratch(args[0].shape[0], tactic, _CUDA))


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying):  # noqa: F811
    certifying(op)
    spec = cell.spec
    rows = spec["rows"]
    args = _inputs(rows, spec["norm"], seed=zlib.crc32(cell.why.encode()) % 2**31)
    args += scalars(spec)
    held = [t.clone() for t in args[:4]]
    ref = _reference(args)
    for tactic in _tactics(rows):
        out = _run(args, tactic)
        assert out[0].shape == args[1].shape and out[3].shape == args[0].shape
        try:
            _compare(out, ref)
        except AssertionError as e:
            raise AssertionError(f"tactic {tactic}: {e}") from e
    for name, before, after in zip(("x", "residual", "post", "comb"), held, args[:4]):
        assert torch.equal(before, after), f"the op wrote its input {name}"


def test_scalars_reach_their_kernel_parameters() -> None:
    """Distinct, exaggerated scalars on small streams, where every epsilon moves the outputs.

    GLM-5.3-Flash's two epsilon pairs are equal and too small to move unit-scale streams, so the
    cells cannot tell a swapped pair from a right one. Here the streams' mean square is about
    4e-3, next to ``norm_eps``; the references with a pair swapped must fail the band, and the op
    must pass it, on the fallback tactic of each backend family.
    """
    rows = 64
    args = _inputs(rows, norm=True, seed=11, std=0.05)
    right = [3, 2.5e-3, 0.25, 0.05, 1.5, 1e-6]
    wrong = {
        "norm_eps and layer_input_eps": [3, 1e-6, 0.25, 0.05, 1.5, 2.5e-3],
        "hc_eps and sinkhorn_eps": [3, 2.5e-3, 0.05, 0.25, 1.5, 1e-6],
        "post_mult": [3, 2.5e-3, 0.25, 0.05, 2.0, 1e-6],
        "sinkhorn_iters": [20, 2.5e-3, 0.25, 0.05, 1.5, 1e-6],
    }
    tactics = dict.fromkeys(
        (default_fused_hc_tactic(rows, HIDDEN, fused_hc_mma_enabled()), DETERMINISTIC_TACTIC)
    )
    for tactic in tactics:
        with validating(op):
            out = _run(args + right, tactic)
        _compare(out, _reference(args + right))
        for name, scalars_ in wrong.items():
            assert not _matches(out, _reference(args + scalars_)), (
                f"the band cannot tell {name} apart"
            )


@pytest.mark.parametrize("all_in_one", [False, True], ids=["fallback", "all_in_one_fma"])
def test_cuda_graph_capture_and_replay(all_in_one: bool) -> None:
    """A captured decode boundary replays on new contents of its inputs and workspaces.

    The all-in-one FMA kernel zeroes its accumulators and counters inside the captured launch.
    """
    spec = op.CELLS[1].spec
    rows = spec["rows"]
    if all_in_one:
        tactic = ("fused_all_fma", 2, 1, 0, 1)
    else:
        tactic = default_fused_hc_tactic(rows, HIDDEN, fused_hc_mma_enabled())
    args = _inputs(rows, spec["norm"], seed=31) + scalars(spec)
    scratch = fused_hc_scratch(rows, tactic, _CUDA)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        op(*args, tactic, *scratch)
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = op(*args, tactic, *scratch)
    for seed in (32, 33):
        for captured, fresh in zip(args[:8], _inputs(rows, spec["norm"], seed=seed)):
            captured.copy_(fresh)
        graph.replay()
        torch.cuda.synchronize()
        _compare(out, _reference(args))


def test_tuned_tactic_is_a_candidate_and_holds(monkeypatch) -> None:
    """A cold tuner gives the fallback; a tuning pass picks a candidate that is then looked up."""
    monkeypatch.delenv("FORCE_DETERMINISTIC", raising=False)
    spec = op.CELLS[1].spec
    rows = spec["rows"]
    args = _inputs(rows, spec["norm"], seed=5) + scalars(spec)
    tuner = AutoTuner.get()
    tuner.clear_cache()
    try:
        assert fused_hc_tactic(*args) == default_fused_hc_tactic(
            rows, HIDDEN, fused_hc_mma_enabled()
        )
        with autotune():
            tuned = fused_hc_tactic(*args)
        assert tuned in _tactics(rows), f"the tuner returned {tuned}, not a candidate"
        assert fused_hc_tactic(*args) == tuned, "the tuned tactic was not looked up"
        with validating(op):
            out = _run(args, tuned)
        _compare(out, _reference(args))
    finally:
        tuner.clear_cache()


def test_force_deterministic_pins_the_tactic(monkeypatch) -> None:
    monkeypatch.setenv("FORCE_DETERMINISTIC", "1")
    spec = op.CELLS[3].spec
    args = _inputs(spec["rows"], spec["norm"], seed=3) + scalars(spec)
    assert fused_hc_tactic(*args) == DETERMINISTIC_TACTIC


def test_guard_refuses_what_the_kernel_misreads() -> None:
    """Workspaces the op does not check, and operands it reads as dense rows."""
    spec = op.CELLS[2].spec
    rows = spec["rows"]
    args = _inputs(rows, spec["norm"], seed=7) + scalars(spec)
    all_fma = ("fused_all_fma", 2, 1, 0, 1)
    all_mma = ("fused_all_mma", 0, 16, 0, 1)
    y, r, done = fused_hc_scratch(rows, all_fma, _CUDA)
    _, _, mma_done = fused_hc_scratch(rows, all_mma, _CUDA)
    my, mr, _ = fused_hc_scratch(rows, all_mma, _CUDA)

    def with_(index: int, value) -> list:
        changed = list(args)
        changed[index] = value
        return changed

    cases = {
        "a short counter array, all-in-one FMA": [*args, all_fma, y, r, done[:1]],
        "short accumulators, all-in-one FMA": [*args, all_fma, y[:1], r[:1], done],
        "no counter, all-in-one TF32": [*args, all_mma, my, mr, mma_done[:0]],
        "comb transposed": [*with_(3, args[3].transpose(1, 2)), all_fma, y, r, done],
        "fn of another width": [*with_(4, args[4][:, :HIDDEN].contiguous()), all_fma, y, r, done],
    }
    for name, call in cases.items():
        with pytest.raises(AssertionError):
            with validating(op):
                op(*call)
            pytest.fail(f"is_valid accepted {name}")
