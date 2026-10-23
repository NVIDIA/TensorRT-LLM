# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the chunk_gated_delta_rule entry.

The first indexed in-place call per autotune key in a process must be as correct as a
warm one: the state kernel's autotuner restores the indexed state its benchmark launches
write, and the pool's dtype is part of the key. "First call in the process" is a property
of the process, so the cold-call test re-runs this file as a child:
`python <this file> <mode>`.
"""

import os
import subprocess
import sys
import zlib
from pathlib import Path

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_103__tp1.ops.ssm.chunk_gated_delta_rule import (  # noqa: E501
    ENGINE_SLOTS,
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
    Discriminated when measured: seven wrong references (no gating, beta = 1, q/k swapped,
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


def _l2norm(x: torch.Tensor) -> torch.Tensor:
    xf = x.float()
    return xf / torch.sqrt(xf.pow(2).sum(-1, keepdim=True) + 1e-6)


def _operands(seqlens: tuple, seed: int):
    """Operands in the domain the GDN prefill path produces: unit-norm q/k,
    `g = -exp(A_log) * softplus(a + dt_bias) < 0`, `beta` a sigmoid. The delta
    rule is only contractive there."""
    gen = torch.Generator(device=DEV).manual_seed(seed)
    total = sum(seqlens)
    cu_seqlens = torch.tensor(
        [0] + torch.tensor(seqlens).cumsum(0).tolist(), dtype=torch.long, device=DEV
    )
    shape_qk = (1, total, NUM_K_HEADS, HEAD_K_DIM)
    q = torch.randn(shape_qk, generator=gen, device=DEV)
    k = torch.randn(shape_qk, generator=gen, device=DEV)
    q, k = _l2norm(q).to(torch.bfloat16), _l2norm(k).to(torch.bfloat16)
    v = torch.randn(
        1, total, NUM_V_HEADS, HEAD_V_DIM, generator=gen, device=DEV, dtype=torch.bfloat16
    )
    a_log = torch.log(torch.rand(NUM_V_HEADS, generator=gen, device=DEV) * 15 + 1)
    dt_bias = torch.randn(NUM_V_HEADS, generator=gen, device=DEV)
    a = torch.randn(1, total, NUM_V_HEADS, generator=gen, device=DEV)
    g = (-torch.exp(a_log) * torch.nn.functional.softplus(a + dt_bias)).float()
    beta = torch.rand(1, total, NUM_V_HEADS, generator=gen, device=DEV).sigmoid().float()
    return q, k, v, g, beta, cu_seqlens


def _pool(num_slots: int, seed: int, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """A recurrent-state pool, `[slots, H, V, K]` (K innermost), at `dtype`."""
    gen = torch.Generator(device=DEV).manual_seed(seed)
    shape = (num_slots, NUM_V_HEADS, HEAD_V_DIM, HEAD_K_DIM)
    return (torch.randn(shape, generator=gen, device=DEV) * 0.1).to(dtype)


def _gate_output(o: torch.Tensor, ref: tuple) -> None:
    """`_compare`, on `o` alone."""
    _compare((o, None, None), (ref[0], None, None))


def _gate_pool(pool: torch.Tensor, ref: tuple) -> None:
    """`_compare`, on the pool alone."""
    _compare((ref[0], None, pool), (ref[0], None, ref[2]))


def _build(spec, seed):
    """`__call__`'s 13 arguments for one spec."""
    q, k, v, g, beta, cu_seqlens = _operands(spec["seqlens"], seed)
    pool = _pool(ENGINE_SLOTS, seed=seed + 1, dtype=spec.get("pool_dtype", torch.float32))
    indices = torch.tensor(spec["indices"], dtype=torch.int32, device=DEV)
    output = None
    if spec.get("output") == "buffer":
        shape_o = (q.shape[0], q.shape[1], NUM_V_HEADS, HEAD_V_DIM)
        output = torch.empty(shape_o, device=DEV, dtype=torch.bfloat16)
    return [q, k, v, g, beta, None, pool, indices, True, False, cu_seqlens, False, output]


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — pytest resolves the fixture by this name
    certifying(op)
    args = _build(cell.spec, seed=zlib.crc32(cell.why.encode()) % 2**31)
    pool = args[6]
    pool_before = pool.clone()
    ref = op.reference(*args)

    o, final_state = op(*args)

    _compare((o, final_state, pool), ref)
    if args[12] is not None:
        assert o.data_ptr() == args[12].data_ptr(), "output= must be written in place"
    named = args[7].tolist()
    unnamed = [s for s in range(pool.shape[0]) if s not in set(named)]
    assert torch.equal(pool[unnamed], pool_before[unnamed]), "unnamed slots written"
    assert not torch.equal(pool[named], pool_before[named]), "named slots did not move"


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


# ---------------------------------------------------------------------------
# The cold call. The cells above have already benchmarked this autotune key by
# the time pytest reaches it, so it runs in a fresh interpreter.

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


def _checkpoint_case(seed: int = 101, pool_dtype: torch.dtype = torch.float32):
    q, k, v, g, beta, cu = _operands(CHILD_SEQLENS, seed=seed)
    pool = _pool(4, seed=seed + 1, dtype=pool_dtype)
    indices = torch.tensor([2, 0], dtype=torch.int32, device=DEV)
    buf = torch.empty(
        1, sum(CHILD_SEQLENS), NUM_V_HEADS, HEAD_V_DIM, device=DEV, dtype=torch.bfloat16
    )
    return (q, k, v, g, beta, None, pool, indices, True, False, cu, False, buf)


def _child_cold_call(pool_dtype: torch.dtype = torch.float32) -> None:
    autotuner = _state_kernel_autotuner()
    assert not autotuner.cache, "the child must be the first indexed call in its process"
    args = _checkpoint_case(pool_dtype=pool_dtype)
    ref = op.reference(*args)
    o, _ = op(*args)
    assert len(autotuner.cache) == 1, "the call must have benchmarked exactly one key"
    _gate_output(o, ref)
    _gate_pool(args[6], ref)


_CHILD_MODES = {
    "cold-call-bf16-pool": lambda: _child_cold_call(torch.bfloat16),
}


def _run_child(mode: str) -> None:
    # `_validating` lives four directories up; the child has no pytest to
    # resolve `__extra_import_path__` for it.
    root = str(Path(__file__).resolve().parents[4])
    existing = os.environ.get("PYTHONPATH")
    env = dict(os.environ, PYTHONPATH=root if not existing else os.pathsep.join([root, existing]))
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


def test_cold_indexed_call_over_a_bf16_pool() -> None:
    """The first indexed call of a process at 16 / 48 heads, K = V = 128, over a bf16
    pool, in the mixer's shape (256 + 96 packed, by slot, into `output=`), matches the
    reference."""
    _run_child("cold-call-bf16-pool")


if __name__ == "__main__":
    _CHILD_MODES[sys.argv[1]]()
