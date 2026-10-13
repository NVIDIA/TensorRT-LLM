# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the chunk_gated_delta_rule entry.

Every indexed in-place cell is preceded by a throwaway warm-up at the same
autotune key: the first such call in a process is a known defect (see the
entry's note), asserted by the cold-call tests at the bottom rather than
tripped over by the cells. Those that need a process of their own re-run this
file as a child: `python <this file> <mode>`.
"""

import os
import subprocess
import sys
import zlib
from pathlib import Path
from typing import Optional

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_103__tp1.ops.ssm.chunk_gated_delta_rule import (  # noqa: E501
    HEAD_K_DIM,
    HEAD_V_DIM,
    NUM_K_HEADS,
    NUM_V_HEADS,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_103__tp1.ops.ssm.chunk_gated_delta_rule import (  # noqa: E501
    chunk_gated_delta_rule as op,
)

__extra_import_path__ = ["../../../.."]
from _validating import certifying, validating  # noqa: E402,F401 — path above; fixture by name

assert torch.cuda.is_available(), "chunk_gated_delta_rule requires a CUDA device"


#: `_compare`'s band; see its docstring.
RTOL, ATOL_OUTPUT, ATOL_STATE = 1.6e-2, 2e-3, 2e-2


def _compare(out: tuple, ref: tuple) -> None:
    """`(o, final_state, pool)`: the call's two returns plus the
    `initial_state` it was handed, against `reference`'s triple; `None`
    components must match.

    Wider than bf16's default because the op is a chunked reformulation
    whose WY operands are rounded to bf16 -- the same recurrence, not the
    same sum. rtol stays at the bf16 default; atol is 2e-3 on `o` and 2e-2
    on states, about 4x and 6x the largest deviations measured warm over
    every cell against this reference (`o` 4.88e-4, one bf16 ulp, on
    magnitudes up to 7.9e-2; states 3.57e-3 on up to 8.3e-1).
    Discriminated: seven wrong references (no gating, beta = 1, q/k swapped,
    initial state dropped, scale = 1, key heads reversed, `[K, V]`
    transposed state) all fail the `o` gate, the closest 11x outside it.
    """
    for got, want, atol in zip(out, ref, (ATOL_OUTPUT, ATOL_STATE, ATOL_STATE)):
        assert (got is None) == (want is None), (got is None, want is None)
        if got is None:
            continue
        assert got.dtype == want.dtype and got.shape == want.shape, (
            (got.dtype, got.shape),
            (want.dtype, want.shape),
        )
        torch.testing.assert_close(got.float(), want.float(), rtol=RTOL, atol=atol)


DEV = "cuda"
SCALE = HEAD_K_DIM**-0.5


def _l2norm(x: torch.Tensor) -> torch.Tensor:
    xf = x.float()
    return xf / torch.sqrt(xf.pow(2).sum(-1, keepdim=True) + 1e-6)


def _operands(
    seqlens: Optional[tuple],
    seed: int,
    batch: int = 1,
    num_tokens: Optional[int] = None,
    num_k_heads: int = NUM_K_HEADS,
    num_v_heads: int = NUM_V_HEADS,
    normalize_qk: bool = True,
):
    """Operands in the domain the GDN prefill path produces: unit-norm q/k,
    `g = -exp(A_log) * softplus(a + dt_bias) < 0`, `beta` a sigmoid. The delta
    rule is only contractive there."""
    gen = torch.Generator(device=DEV).manual_seed(seed)
    if seqlens is None:
        cu_seqlens, total = None, num_tokens
    else:
        total = sum(seqlens)
        cu_seqlens = torch.tensor(
            [0] + torch.tensor(seqlens).cumsum(0).tolist(), dtype=torch.long, device=DEV
        )
    shape_qk = (batch, total, num_k_heads, HEAD_K_DIM)
    q = torch.randn(shape_qk, generator=gen, device=DEV)
    k = torch.randn(shape_qk, generator=gen, device=DEV)
    if normalize_qk:
        q, k = _l2norm(q), _l2norm(k)
    q, k = q.to(torch.bfloat16), k.to(torch.bfloat16)
    v = torch.randn(
        batch, total, num_v_heads, HEAD_V_DIM, generator=gen, device=DEV, dtype=torch.bfloat16
    )
    a_log = torch.log(torch.rand(num_v_heads, generator=gen, device=DEV) * 15 + 1)
    dt_bias = torch.randn(num_v_heads, generator=gen, device=DEV)
    a = torch.randn(batch, total, num_v_heads, generator=gen, device=DEV)
    g = (-torch.exp(a_log) * torch.nn.functional.softplus(a + dt_bias)).float()
    beta = torch.rand(batch, total, num_v_heads, generator=gen, device=DEV).sigmoid().float()
    return q, k, v, g, beta, cu_seqlens


def _pool(num_slots: int, seed: int, num_v_heads: int = NUM_V_HEADS) -> torch.Tensor:
    """An fp32 recurrent-state pool, `[slots, H, V, K]` (K innermost)."""
    gen = torch.Generator(device=DEV).manual_seed(seed)
    shape = (num_slots, num_v_heads, HEAD_V_DIM, HEAD_K_DIM)
    return torch.randn(shape, generator=gen, device=DEV) * 0.1


def _prime_indexed_state_kernel(num_k_heads=NUM_K_HEADS, num_v_heads=NUM_V_HEADS) -> None:
    """The certified warm-up: one throwaway indexed call on a 1-slot scratch
    pool, at the key every indexed cell uses (H, K, V and every dtype)."""
    q, k, v, g, beta, cu_seqlens = _operands(
        (64,), seed=999_983, num_k_heads=num_k_heads, num_v_heads=num_v_heads
    )
    op(
        q,
        k,
        v,
        g,
        beta,
        None,
        _pool(1, seed=999_984, num_v_heads=num_v_heads),
        torch.zeros(1, dtype=torch.int32, device=DEV),
        True,
        False,
        cu_seqlens,
        False,
        None,
    )


def _gate_output(o: torch.Tensor, ref: tuple) -> None:
    """`_compare`, on `o` alone."""
    _compare((o, None, None), (ref[0], None, None))


def _gate_pool(pool: torch.Tensor, ref: tuple) -> None:
    """`_compare`, on the pool alone."""
    _compare((ref[0], None, pool), (ref[0], None, ref[2]))


def _build(spec, seed):
    """`__call__`'s 13 arguments for one spec, plus the wide output buffer."""
    q, k, v, g, beta, cu_seqlens = _operands(
        spec.get("seqlens"),
        seed,
        batch=spec.get("batch", 1),
        num_tokens=spec.get("num_tokens"),
        normalize_qk=spec.get("normalize_qk", True),
    )
    num_seqs = q.shape[0] if cu_seqlens is None else len(cu_seqlens) - 1
    initial, indices, inplace = None, None, False
    if spec["initial"] == "indexed":
        initial = _pool(spec["slots"], seed=seed + 1)
        indices = torch.tensor(
            spec.get("indices", tuple(range(num_seqs))), dtype=torch.int32, device=DEV
        )
        inplace = True
    elif spec["initial"] == "positional":
        initial = _pool(num_seqs, seed=seed + 1)
    wide, output = None, None
    shape_o = (q.shape[0], q.shape[1], NUM_V_HEADS, HEAD_V_DIM)
    if spec.get("output") == "buffer":
        output = torch.empty(shape_o, device=DEV, dtype=torch.bfloat16)
    elif spec.get("output") == "window":
        wide = torch.zeros(1, 200, NUM_V_HEADS, HEAD_V_DIM, device=DEV, dtype=torch.bfloat16)
        output = wide[:, 50 : 50 + q.shape[1]]
    args = [
        q,
        k,
        v,
        g,
        beta,
        spec.get("scale"),
        initial,
        indices,
        inplace,
        spec.get("output_final_state", False),
        cu_seqlens,
        spec.get("l2norm", False),
        output,
    ]
    return args, wide


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — pytest resolves the fixture by this name
    spec = cell.spec
    if spec["initial"] == "indexed":
        _prime_indexed_state_kernel()
    certifying(op)
    args, wide = _build(spec, seed=zlib.crc32(cell.why.encode()) % 2**31)
    pool = args[6]
    pool_before = None if pool is None else pool.clone()
    ref = op.reference(*args)

    o, final_state = op(*args)

    _compare((o, final_state, pool), ref)
    if args[12] is not None:
        assert o.data_ptr() == args[12].data_ptr(), "output= must be written in place"
    if wide is not None:
        assert torch.count_nonzero(wide[:, :50]) == 0 and torch.count_nonzero(wide[:, 146:]) == 0
    if spec["initial"] == "positional":
        assert torch.equal(pool, pool_before), "a positional initial state is read-only"
    if spec["initial"] == "indexed":
        named = args[7].tolist()
        unnamed = [s for s in range(pool.shape[0]) if s not in set(named)]
        assert torch.equal(pool[unnamed], pool_before[unnamed]), "unnamed slots written"
        assert not torch.equal(pool[named], pool_before[named]), "named slots did not move"


def test_op_rejects_these_domains_loudly() -> None:
    """Why `is_valid` guards only `output`: every other refusal is the op's."""
    q, k, v, g, beta, cu = _operands((80,), seed=67)
    idx1 = torch.zeros(1, dtype=torch.int32, device=DEV)
    cases = [
        (AssertionError, "does not support float32", dict(q=q.float(), k=k.float(), v=v.float())),
        (AssertionError, "", dict(k=k.half())),
        (AssertionError, "beta must be of shape", dict(beta=beta[0])),
        (ValueError, "batch size is expected to be 1", dict(q=q.expand(2, -1, -1, -1))),
        (
            ValueError,
            "number of initial-state indices",
            dict(
                initial_state=_pool(4, seed=671),
                initial_state_indices=torch.tensor([0, 1], dtype=torch.int32, device=DEV),
                inplace_indexed_state_update=True,
            ),
        ),
        (ValueError, "number of initial states", dict(initial_state=_pool(3, seed=672))),
        (
            ValueError,
            "require inplace_indexed_state_update=True",
            dict(initial_state=_pool(1, seed=673), initial_state_indices=idx1),
        ),
    ]
    for exc, match, override in cases:
        kwargs = dict(q=q, k=k, v=v, g=g, beta=beta, cu_seqlens=cu)
        kwargs.update(override)
        with pytest.raises(exc, match=match):
            with validating(op):
                op(**kwargs)


def test_guard_refuses_a_noncontiguous_output() -> None:
    """Unguarded, a head-slice view of a wider buffer is written through
    shape-derived strides and the result is wrong."""
    q, k, v, g, beta, cu = _operands((96,), seed=61)
    ref = op.reference(q, k, v, g, beta, cu_seqlens=cu)
    head_slice = torch.zeros(1, 96, 2 * NUM_V_HEADS, HEAD_V_DIM, device=DEV, dtype=torch.bfloat16)[
        :, :, :NUM_V_HEADS
    ]
    assert not head_slice.is_contiguous()
    o, _ = op(q, k, v, g, beta, cu_seqlens=cu, output=head_slice)
    with pytest.raises(AssertionError):
        _gate_output(o, ref)
    with pytest.raises(AssertionError, match="output must be contiguous"):
        with validating(op):
            op(q, k, v, g, beta, cu_seqlens=cu, output=head_slice)


def test_guard_refuses_an_output_of_another_dtype() -> None:
    """Unguarded, an fp32 buffer is filled but the returned bf16 tensor no
    longer aliases it -- the allocation the buffer existed to avoid."""
    q, k, v, g, beta, cu = _operands((96,), seed=62)
    buf = torch.zeros(1, 96, NUM_V_HEADS, HEAD_V_DIM, device=DEV, dtype=torch.float32)
    o, _ = op(q, k, v, g, beta, cu_seqlens=cu, output=buf)
    assert o.dtype == torch.bfloat16 and o.data_ptr() != buf.data_ptr(), "hazard gone"
    with pytest.raises(AssertionError, match="output must have q's dtype"):
        with validating(op):
            op(q, k, v, g, beta, cu_seqlens=cu, output=buf)


def test_gate_discriminates_wrong_variants() -> None:
    """Seven references wrong in one way each must all fail the `o` gate the
    correct one passes. Positional path: no warm-up needed."""
    q, k, v, g, beta, cu = _operands((128, 64), seed=71)
    initial = _pool(2, seed=711)
    base = dict(q=q, k=k, v=v, g=g, beta=beta, scale=SCALE, initial_state=initial, cu_seqlens=cu)
    o, final_state = op(**base, output_final_state=True)
    ref = op.reference(**base, output_final_state=True)
    _compare((o, final_state, initial), ref)

    variants = {
        "no gating": dict(g=torch.zeros_like(g)),
        "beta forced to 1": dict(beta=torch.ones_like(beta)),
        "q and k swapped": dict(q=k, k=q),
        "initial state dropped": dict(initial_state=None),
        "scale left at 1": dict(scale=1.0),
        "key heads reversed": dict(q=q.flip(2).contiguous(), k=k.flip(2).contiguous()),
        "initial state [K, V]": dict(initial_state=initial.transpose(-1, -2).contiguous()),
    }
    for name, override in variants.items():
        wrong = op.reference(**{**base, **override})
        with pytest.raises(AssertionError):
            _gate_output(wrong[0], ref)


def test_first_indexed_call_for_a_key_deviates_from_the_reference() -> None:
    """The cold call, asserted wrong: the defect is the product.

    A head geometry (5 key / 20 value heads) nothing else here uses, so this is
    the first indexed call at its key. Measured: output 1.7e-2 off on a
    reference magnitude of 6.1e-2, against 4.2e-4 warm. The state is asserted
    right -- this gate domain decays the corrupted initial state away by the
    end of the sequence (1.7e-3 against the 2e-2 gate). A `DID NOT RAISE` means
    the autotuner stopped corrupting `h0`: update the entry's note.
    """
    q, k, v, g, beta, cu = _operands((96,), seed=83, num_k_heads=5, num_v_heads=20)
    pool = _pool(4, seed=831, num_v_heads=20)
    indices = torch.tensor([1], dtype=torch.int32, device=DEV)
    args = (q, k, v, g, beta, None, pool, indices, True, False, cu, False, None)
    ref = op.reference(*args)
    o, _ = op(*args)
    with pytest.raises(AssertionError, match="Tensor-likes are not close"):
        _gate_output(o, ref)
    _gate_pool(pool, ref)


# ---------------------------------------------------------------------------
# The cold call at this checkpoint's geometry. "First indexed call in the
# process" is a property of the process, and the cells above have warmed this
# key by the time pytest reaches these, so each runs in a fresh interpreter.

CHILD_TIMEOUT_S = 420
#: The GDN prefill call shape, packed varlen, state by slot index.
CHILD_SEQLENS = (256, 96)


def _state_kernel_autotuner():
    """The `triton.autotune` wrapper around `chunk_gated_delta_rule_fwd_kernel_h_blockdim64`;
    a key in its `cache` is a key whose benchmark launches already happened."""
    from tensorrt_llm._torch.modules.fla.chunk_delta_h import (
        chunk_gated_delta_rule_fwd_kernel_h_blockdim64 as node,
    )

    while node is not None:
        if isinstance(getattr(node, "cache", None), dict):
            return node
        node = getattr(node, "fn", None)
    raise AssertionError("no triton autotuner found around the chunked state kernel")


def _checkpoint_case(seed: int = 101):
    q, k, v, g, beta, cu = _operands(CHILD_SEQLENS, seed=seed)
    pool = _pool(4, seed=seed + 1)
    indices = torch.tensor([2, 0], dtype=torch.int32, device=DEV)
    buf = torch.empty(
        1, sum(CHILD_SEQLENS), NUM_V_HEADS, HEAD_V_DIM, device=DEV, dtype=torch.bfloat16
    )
    return (q, k, v, g, beta, None, pool, indices, True, False, cu, False, buf)


def _report(tag: str, o, pool, ref) -> None:
    print(
        f"{tag}: max|o - ref| = {(o.float() - ref[0].float()).abs().max().item():.3e}; "
        f"max|pool - ref| = {(pool - ref[2]).abs().max().item():.3e}"
    )


def _child_cold_call() -> None:
    autotuner = _state_kernel_autotuner()
    assert not autotuner.cache, "the child must be the first indexed call in its process"
    args = _checkpoint_case()
    ref = op.reference(*args)
    o, _ = op(*args)
    print("autotune keys after the call:", list(autotuner.cache))
    _report("cold", o, args[6], ref)
    with pytest.raises(AssertionError, match="Tensor-likes are not close"):
        _gate_output(o, ref)
    _gate_pool(args[6], ref)


def _child_warmup_then_first_real_call() -> None:
    autotuner = _state_kernel_autotuner()
    _prime_indexed_state_kernel()
    primed = set(autotuner.cache)
    assert len(primed) == 1, f"one warm-up, one autotune key; got {primed}"
    args = _checkpoint_case()
    pool_before = args[6].clone()
    ref = op.reference(*args)
    o, _ = op(*args)
    assert set(autotuner.cache) == primed, "the real call re-benchmarked: warm-up missed its key"
    _report("first real call after a scratch warm-up", o, args[6], ref)
    _compare((o, None, args[6]), (ref[0], None, ref[2]))
    assert torch.equal(args[6][[1, 3]], pool_before[[1, 3]]), "unnamed slots written"


def _child_warmup_key_controls() -> None:
    """Which warm-ups consume the real call's benchmark; asserted on the
    autotuner's cache, so it keeps its meaning once the kernel is fixed."""
    autotuner = _state_kernel_autotuner()
    assert not autotuner.cache
    idx = torch.zeros(1, dtype=torch.int32, device=DEV)
    # (a) a different value-head count.
    q, k, v, g, beta, cu = _operands((64,), seed=201, num_k_heads=8, num_v_heads=32)
    op(q, k, v, g, beta, None, _pool(1, 202, 32), idx, True, False, cu, False, None)
    # (b) int32 cu_seqlens.
    q, k, v, g, beta, cu = _operands((64,), seed=203)
    op(q, k, v, g, beta, None, _pool(1, 204), idx, True, False, cu.int(), False, None)
    # (c) the positional path.
    q, k, v, g, beta, cu = _operands((64,), seed=205)
    op(q, k, v, g, beta, None, _pool(1, 206), None, False, True, cu, False, None)
    missed = set(autotuner.cache)
    print("keys after the three mismatched warm-ups:", list(missed))
    assert len(missed) == 3, f"each mismatched warm-up owns a key of its own; got {missed}"
    # (d) the matching one, differing from the real call in T and Hg only.
    _prime_indexed_state_kernel(num_k_heads=8)
    matched = set(autotuner.cache) - missed
    assert len(matched) == 1, f"the matching warm-up must add exactly one key; got {matched}"
    op(*_checkpoint_case(seed=207))
    assert set(autotuner.cache) == missed | matched, "the real call hit a key no warm-up reached"


_CHILD_MODES = {
    "cold-call": _child_cold_call,
    "warmup-then-real": _child_warmup_then_first_real_call,
    "warmup-key-controls": _child_warmup_key_controls,
}


def _run_child(mode: str) -> None:
    # `_validating` lives four directories up; the child has no pytest to
    # resolve `__extra_import_path__` for it.
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[4]))
    proc = subprocess.run(
        [sys.executable, __file__, mode],
        capture_output=True,
        text=True,
        timeout=CHILD_TIMEOUT_S,
        env=env,
    )
    print(proc.stdout)
    assert proc.returncode == 0, (
        f"child `{mode}` exited {proc.returncode}\n--- stdout ---\n{proc.stdout}\n"
        f"--- stderr ---\n{proc.stderr}"
    )


def test_cold_indexed_call_at_this_checkpoints_geometry() -> None:
    """The first indexed call of a process at 16 / 48 heads, K = V = 128, in
    the mixer's shape (256 + 96 packed, by slot, into `output=`): output
    measured 4.034e-02 off against the 2e-3 gate (reference max 7.5e-02),
    asserted wrong; the state stays inside its gate in this gate domain."""
    _run_child("cold-call")


def test_scratch_warmup_makes_the_first_real_call_correct() -> None:
    """One throwaway call on a 1-slot scratch pool, then the process's first
    real call is correct: 4.883e-04 / 2.595e-03 on the same operands."""
    _run_child("warmup-then-real")


def test_only_a_key_matching_warmup_consumes_the_benchmark() -> None:
    """A different `H`, int32 `cu_seqlens`, or the positional path each
    benchmark a key of their own; `T` and `Hg` are not in the key."""
    _run_child("warmup-key-controls")


if __name__ == "__main__":
    _CHILD_MODES[sys.argv[1]]()
