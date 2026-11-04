# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Mid-sequence state snapshots of the indexed KDA prefill kernel.

``trtllm::kda_prefill`` with ``snapshot_pool`` / ``snapshot_rows`` /
``snapshot_chunks`` stores sequence ``b``'s recurrent state after its first
``64 * snapshot_chunks[b]`` tokens into ``snapshot_pool[snapshot_rows[b]]``
while it prefills the whole sequence (the in-forward snapshots of the KDA
layers). The snapshot must be bitwise the final state of a prefill that stops
at that token (same kernels, same fp32 arithmetic up to that point), and the
outputs and end states of the call must be bitwise those of the call without
snapshots.
"""

import pytest
import torch

pytest.importorskip("fla")

from kda_prefill_test_utils import assert_kda_close  # noqa: E402

from tensorrt_llm._torch.modules.kimi_kda._kda_kernels import (  # noqa: E402
    KDAKernelDispatch,
    is_intree_prefill_available,
)

NUM_HEADS = 96
HEAD_DIM = 128
CHUNK = 64


def _has_supported_gpu() -> bool:
    if not torch.cuda.is_available():
        return False
    return torch.cuda.get_device_capability(0) in {(10, 0), (10, 3), (10, 7)}


pytestmark = pytest.mark.skipif(
    not _has_supported_gpu(),
    reason="Kimi K3 is supported only on Blackwell SM100/SM103 or Rubin SM107",
)


@pytest.fixture(scope="module")
def dispatch() -> KDAKernelDispatch:
    if not is_intree_prefill_available():
        pytest.skip("the indexed KDA prefill kernel is unavailable")
    optimized = KDAKernelDispatch(use_optimized_prefill=True, use_optimized_decode=False)
    assert optimized.prefill_kernel_path == "optimized"
    return optimized


@pytest.fixture(scope="module")
def gate_params() -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cuda").manual_seed(0)
    a_log = torch.randn(NUM_HEADS, generator=generator, dtype=torch.float32, device="cuda") * 0.5
    dt_bias = (
        torch.randn(NUM_HEADS * HEAD_DIM, generator=generator, dtype=torch.float32, device="cuda")
        * 0.1
    )
    return a_log, dt_bias


def _make_inputs(total_tokens: int, seed: int) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator(device="cuda").manual_seed(seed)

    def rand(*shape: int, dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
        return torch.randn(*shape, generator=generator, dtype=torch.float32, device="cuda").to(
            dtype
        )

    q = rand(1, total_tokens, NUM_HEADS, HEAD_DIM)
    k = rand(1, total_tokens, NUM_HEADS, HEAD_DIM)
    v = rand(1, total_tokens, NUM_HEADS, HEAD_DIM)
    g = rand(1, total_tokens, NUM_HEADS, HEAD_DIM)
    beta = rand(1, total_tokens, NUM_HEADS, dtype=torch.float32)
    # The kernels expect caller-normalized Q/K (fused_kda_post_conv's output).
    q = torch.nn.functional.normalize(q.float(), dim=-1).to(q.dtype)
    k = torch.nn.functional.normalize(k.float(), dim=-1).to(k.dtype)
    return q, k, v, g, beta


def _make_initial_states(num_sequences: int, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cuda").manual_seed(seed)
    state = torch.randn(
        num_sequences,
        NUM_HEADS,
        HEAD_DIM,
        HEAD_DIM,
        generator=generator,
        dtype=torch.float32,
        device="cuda",
    )
    return state * torch.linspace(0.5, 1.5, HEAD_DIM, device="cuda").view(1, 1, -1, 1)


def _cu_seqlens(lengths: list[int]) -> torch.Tensor:
    return torch.tensor(
        [0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.long, device="cuda"
    )


def _gather_sequences(
    inputs: tuple[torch.Tensor, ...], lengths: list[int], keep: list[int]
) -> tuple[torch.Tensor, ...]:
    """Pack the first ``keep[b]`` tokens of every sequence ``b``."""
    starts = [0, *torch.tensor(lengths).cumsum(0).tolist()]
    index = torch.cat(
        [torch.arange(starts[b], starts[b] + keep[b], device="cuda") for b in range(len(lengths))]
    )
    return tuple(t.index_select(1, index).contiguous() for t in inputs)


def _run(
    dispatch: KDAKernelDispatch,
    gate_params: tuple[torch.Tensor, torch.Tensor],
    inputs: tuple[torch.Tensor, ...],
    lengths: list[int],
    initial_states: torch.Tensor | None,
    *,
    snapshot_offsets: list[int] | None = None,
    single_sequence_length: int | None = None,
    varlen_is_aligned: bool | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Prefill ``lengths`` from slots ``0..B-1``; snapshots go to rows ``0..B-1`` of a separate
    snapshot pool.

    Returns the output, the end states and the snapshot rows (``None`` without
    snapshots). ``varlen_is_aligned`` overrides the multi-sequence K123 variant
    (mask-free for chunk-aligned lengths, masked otherwise).
    """
    if varlen_is_aligned is None:
        varlen_is_aligned = all(length % CHUNK == 0 for length in lengths)
    num_sequences = len(lengths)
    q, k, v, g, beta = (t.clone() for t in inputs)
    a_log, dt_bias = gate_params
    pool = torch.full(
        (num_sequences + 1, NUM_HEADS, HEAD_DIM, HEAD_DIM),
        float("nan"),
        dtype=torch.float32,
        device="cuda",
    )
    snapshot_pool = torch.full_like(pool, float("nan"))
    if initial_states is not None:
        pool[:num_sequences] = initial_states
    else:
        pool[:num_sequences] = 0
    state_indices = torch.arange(num_sequences, dtype=torch.int32, device="cuda")
    snapshot_kwargs = {}
    if snapshot_offsets is not None:
        assert all(offset % CHUNK == 0 for offset in snapshot_offsets if offset >= 0)
        rows = [b if offset >= 0 else -1 for b, offset in enumerate(snapshot_offsets)]
        # As _prepare_inforward_split emits it: no snapshot is row -1, chunk 0.
        chunks = [offset // CHUNK if offset >= 0 else 0 for offset in snapshot_offsets]
        snapshot_kwargs = {
            "snapshot": (
                snapshot_pool,
                torch.tensor(rows, dtype=torch.int32, device="cuda"),
                torch.tensor(chunks, dtype=torch.int32, device="cuda"),
            )
        }
    output, final_state = dispatch.prefill_chunk_kda(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        A_log=a_log,
        dt_bias=dt_bias,
        scale=HEAD_DIM**-0.5,
        initial_state=None,
        safe_gate=True,
        lower_bound=-5.0,
        cu_seqlens=_cu_seqlens(lengths),
        state_pool=pool,
        state_indices=state_indices,
        varlen_is_aligned=varlen_is_aligned,
        single_sequence_length=single_sequence_length,
        **snapshot_kwargs,
    )
    assert final_state is None
    torch.cuda.synchronize()
    snapshots = snapshot_pool[:num_sequences].clone() if snapshot_offsets else None
    # The extra rows are never addressed.
    assert torch.isnan(pool[num_sequences]).all()
    assert torch.isnan(snapshot_pool[num_sequences]).all()
    return output, pool[:num_sequences].clone(), snapshots


def _fla_states(
    gate_params: tuple[torch.Tensor, torch.Tensor],
    inputs: tuple[torch.Tensor, ...],
    lengths: list[int],
    initial_states: torch.Tensor | None,
) -> torch.Tensor:
    reference = KDAKernelDispatch(use_optimized_prefill=False, use_optimized_decode=False)
    q, k, v, g, beta = (t.clone() for t in inputs)
    a_log, dt_bias = gate_params
    _, final_state = reference.prefill_chunk_kda(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        A_log=a_log,
        dt_bias=dt_bias,
        scale=HEAD_DIM**-0.5,
        initial_state=initial_states.clone() if initial_states is not None else None,
        safe_gate=True,
        lower_bound=-5.0,
        cu_seqlens=_cu_seqlens(lengths),
    )
    return final_state


def _assert_bitwise(name: str, actual: torch.Tensor, expected: torch.Tensor) -> None:
    if not torch.equal(actual, expected):
        diff = (actual.float() - expected.float()).abs()
        raise AssertionError(
            f"{name}: not bitwise equal (max abs diff {diff.max().item():.3e}, "
            f"{(diff > 0).sum().item()} of {diff.numel()} elements differ)"
        )


CASES = [
    # (sequence lengths, snapshot offsets (-1: none), with initial states)
    ([65], [64], False),
    ([200], [128], False),
    ([200], [128], True),
    ([256], [128], True),
    ([2049], [2048], False),
    ([2100], [2048], True),
    ([16384 + 77], [16384], True),
    ([200, 130, 300], [128, -1, 256], True),
    ([256, 128, 192], [128, 64, -1], False),
    ([129, 4096 + 5, 64 + 1], [64, 4096, 64], True),
]


@pytest.mark.parametrize("lengths,snapshot_offsets,with_initial_state", CASES)
@torch.no_grad()
def test_snapshot_is_prefix_prefill_state(
    dispatch, gate_params, lengths, snapshot_offsets, with_initial_state
):
    seed = sum(lengths) + 7 * len(lengths) + int(with_initial_state)
    inputs = _make_inputs(sum(lengths), seed)
    initial_states = _make_initial_states(len(lengths), seed + 1) if with_initial_state else None

    plain_out, plain_end, _ = _run(dispatch, gate_params, inputs, lengths, initial_states)
    snap_out, snap_end, snapshots = _run(
        dispatch,
        gate_params,
        inputs,
        lengths,
        initial_states,
        snapshot_offsets=snapshot_offsets,
    )
    # The snapshot store must not perturb the rest of the computation.
    _assert_bitwise("output with snapshots", snap_out, plain_out)
    _assert_bitwise("end state with snapshots", snap_end, plain_end)

    # Prefill that stops at each snapshot point, from the same initial states.
    # Multi-sequence batches keep the K123 variant of the full batch: the
    # mask-free and masked variants agree only up to fp32 rounding.
    prefix_lengths = [
        offset if offset >= 0 else length for length, offset in zip(lengths, snapshot_offsets)
    ]
    prefix_inputs = _gather_sequences(inputs, lengths, prefix_lengths)
    _, prefix_end, _ = _run(
        dispatch,
        gate_params,
        prefix_inputs,
        prefix_lengths,
        initial_states,
        varlen_is_aligned=all(length % CHUNK == 0 for length in lengths),
    )
    for b, offset in enumerate(snapshot_offsets):
        if offset < 0:
            continue
        _assert_bitwise(f"snapshot of sequence {b}", snapshots[b], prefix_end[b])
        # What a bf16 pool stores: the same fp32 state rounded once.
        _assert_bitwise(
            f"bf16 snapshot of sequence {b}",
            snapshots[b].to(torch.bfloat16),
            prefix_end[b].to(torch.bfloat16),
        )
        assert torch.isfinite(snapshots[b]).all()
    for b, offset in enumerate(snapshot_offsets):
        if offset < 0:
            assert torch.isnan(snapshots[b]).all(), f"sequence {b} wrote an unrequested snapshot"


@pytest.mark.parametrize(
    "lengths,snapshot_offsets", [([2100], [2048]), ([200, 4096 + 5], [128, 4096])]
)
@torch.no_grad()
def test_snapshot_and_end_state_match_fla(dispatch, gate_params, lengths, snapshot_offsets):
    inputs = _make_inputs(sum(lengths), 11)
    initial_states = _make_initial_states(len(lengths), 12)
    _, end_states, snapshots = _run(
        dispatch,
        gate_params,
        inputs,
        lengths,
        initial_states,
        snapshot_offsets=snapshot_offsets,
    )
    fla_end = _fla_states(gate_params, inputs, lengths, initial_states)
    assert_kda_close("end state vs FLA", end_states, fla_end)
    prefix_inputs = _gather_sequences(inputs, lengths, snapshot_offsets)
    fla_prefix = _fla_states(gate_params, prefix_inputs, snapshot_offsets, initial_states)
    assert_kda_close("snapshot vs FLA prefix state", snapshots, fla_prefix)


def test_snapshot_argument_validation(dispatch, gate_params):
    inputs = _make_inputs(128, 3)
    pool = torch.zeros(3, NUM_HEADS, HEAD_DIM, HEAD_DIM, dtype=torch.float32, device="cuda")
    rows32 = torch.ones(1, dtype=torch.int32, device="cuda")
    q, k, v, g, beta = inputs
    a_log, dt_bias = gate_params
    common = dict(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        A_log=a_log,
        dt_bias=dt_bias,
        scale=HEAD_DIM**-0.5,
        initial_state=None,
        safe_gate=True,
        lower_bound=-5.0,
        cu_seqlens=_cu_seqlens([128]),
        state_pool=pool,
        state_indices=torch.zeros(1, dtype=torch.int32, device="cuda"),
    )
    with pytest.raises(ValueError, match="int32"):
        dispatch.prefill_chunk_kda(**common, snapshot=(pool, rows32.long(), rows32))
    with pytest.raises(ValueError, match="V-first rows"):
        dispatch.prefill_chunk_kda(**common, snapshot=(pool.bfloat16(), rows32, rows32))
    reference = KDAKernelDispatch(use_optimized_prefill=False, use_optimized_decode=False)
    with pytest.raises(RuntimeError, match="optimized prefill path"):
        reference.prefill_chunk_kda(
            **{**common, "state_pool": None, "state_indices": None},
            snapshot=(pool, rows32, rows32),
        )
