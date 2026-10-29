# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Production wiring of the opt-in FlashInfer KDA prefill (``TLLM_KDA_FLASHINFER_PREFILL=1``).

The schedule, CUDA-graph and soundness tests, under the opt-in switch:

- ``KimiK3MambaMetadata.prepare`` publishes the stable descending length order and the
  16-token chunk prefix sums of the context requests, in storage that survives prepares;
- a prepared schedule is bitwise equal to the one the op generates on the device, through
  the mixer and through the op across layers, modes and CUDA-graph replays with a
  refreshed schedule;
- the op runs on the caller's stream and honours replaced packed sequence metadata.
"""

from itertools import accumulate
from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.modules.kimi_kda._kda_flashinfer_schedule import FLASHINFER_PREFILL_ENV
from tensorrt_llm._torch.modules.kimi_kda.kimi_k3_mamba_metadata import KimiK3MambaMetadata

NUM_HEADS = 96
HEAD_DIM = 128
LOWER_BOUND = -5.0


def _flashinfer_device() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability() in (
        (10, 0),
        (10, 3),
        (10, 7),
    )


needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
needs_flashinfer_kda = pytest.mark.skipif(
    not _flashinfer_device(), reason="FlashInfer KDA requires SM100/SM103/SM107"
)


@pytest.fixture
def switch_on(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(FLASHINFER_PREFILL_ENV, "1")


def _make_attention_metadata(context_lengths: list[int]) -> SimpleNamespace:
    seq_lens = torch.tensor(context_lengths or [1], dtype=torch.int)
    num_contexts = len(context_lengths)
    return SimpleNamespace(
        seq_lens=seq_lens,
        seq_lens_cuda=seq_lens.cuda(),
        num_contexts=num_contexts,
        num_ctx_tokens=sum(context_lengths),
        kv_cache_manager=None,
        request_ids=None,
        kv_cache_params=SimpleNamespace(
            num_cached_tokens_per_seq=torch.zeros(num_contexts, dtype=torch.int),
        ),
    )


def _expected_schedule(lengths: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
    order = sorted(range(len(lengths)), key=lambda i: (-lengths[i], i))
    chunks = list(accumulate(((length + 15) // 16 for length in lengths), initial=0))
    as_cuda = dict(device="cuda", dtype=torch.int32)
    return torch.tensor(order, **as_cuda), torch.tensor(chunks, **as_cuda)


def _flashinfer_prefill_op():
    from tensorrt_llm._torch.custom_ops import cute_dsl_kimi_k3_flashinfer_ops

    return cute_dsl_kimi_k3_flashinfer_ops.kda_prefill_flashinfer


# --------------------------------------------------------------------------- metadata wiring
@needs_cuda
@pytest.mark.parametrize("lengths", [[128, 256], [1, 64, 65, 129], [17, 0, 17, 16], [300], []])
def test_prepare_publishes_schedule(switch_on: None, lengths: list[int]) -> None:
    metadata = KimiK3MambaMetadata(max_batch_size=4, chunk_size=128)
    slots = torch.tensor([3, 1, 2, 0], dtype=torch.int32, device="cuda")
    metadata.state_indices.copy_(slots)
    metadata.prepare(_make_attention_metadata(lengths))
    order, chunks = _expected_schedule(lengths)
    assert metadata.kda_max_seqlen == max(lengths, default=0)
    torch.testing.assert_close(metadata.kda_seq_order, order)
    torch.testing.assert_close(metadata.kda_cu_chunks, chunks)
    torch.testing.assert_close(metadata.state_indices, slots)


@needs_cuda
def test_prepare_refreshes_schedule_without_changing_storage(switch_on: None) -> None:
    metadata = KimiK3MambaMetadata(max_batch_size=4, chunk_size=128)
    metadata.prepare(_make_attention_metadata([17, 65]))
    pointers = (metadata.kda_seq_order.data_ptr(), metadata.kda_cu_chunks.data_ptr())
    metadata.prepare(_make_attention_metadata([64, 18]))
    order, chunks = _expected_schedule([64, 18])
    assert metadata.kda_max_seqlen == 64
    torch.testing.assert_close(metadata.kda_seq_order, order)
    torch.testing.assert_close(metadata.kda_cu_chunks, chunks)
    assert pointers == (metadata.kda_seq_order.data_ptr(), metadata.kda_cu_chunks.data_ptr())
    metadata.prepare(_make_attention_metadata([]))
    assert metadata.kda_max_seqlen == 0


@needs_cuda
def test_prepare_schedule_excludes_decode_requests(switch_on: None) -> None:
    metadata = KimiK3MambaMetadata(max_batch_size=4, chunk_size=128)
    batch = _make_attention_metadata([17, 65])
    batch.seq_lens = torch.tensor([17, 65, 1, 1], dtype=torch.int)
    batch.seq_lens_cuda = batch.seq_lens.cuda()
    metadata.prepare(batch)
    order, chunks = _expected_schedule([17, 65])
    torch.testing.assert_close(metadata.kda_seq_order, order)
    torch.testing.assert_close(metadata.kda_cu_chunks, chunks)


# --------------------------------------------------------------------------- prepared vs device schedule
@needs_flashinfer_kda
@pytest.mark.parametrize("heads", [6, 96])
@torch.inference_mode()
def test_module_with_prepared_schedule_matches_device_generation(
    switch_on: None, heads: int
) -> None:
    pytest.importorskip("fla")
    from tests.unittest._torch.modules.kimi_kda.kda_module_test_utils import (
        clone_pool,
        make_attention,
        make_batch,
    )

    module = make_attention(heads, hidden_size=768, bf16_weights=True, flashinfer=True)
    lengths = [17, 65]
    batch = make_batch(heads, lengths, hidden_size=768, initial="mixed", padded=True)
    prepared = KimiK3MambaMetadata(max_batch_size=4, chunk_size=128)
    prepared.prepare(_make_attention_metadata(lengths))
    prepared.state_indices[: batch.n].copy_(batch.indices)
    prepared.has_initial_states[: batch.n].copy_(batch.metadata.has_initial_states)
    prepared.use_initial_states = batch.metadata.use_initial_states
    states = [clone_pool(batch.state), clone_pool(batch.state)]
    convs = [batch.conv.clone(), batch.conv.clone()]
    outputs = [
        module.forward_prefill(
            batch.hidden, batch.cu, metadata, batch.n, conv, state, batch.indices
        )
        for metadata, conv, state in zip((batch.metadata, prepared), convs, states)
    ]
    torch.testing.assert_close(outputs[0], outputs[1], rtol=0, atol=0)
    torch.testing.assert_close(states[0], states[1], rtol=0, atol=0)
    torch.testing.assert_close(convs[0], convs[1], rtol=0, atol=0)


@needs_flashinfer_kda
@pytest.mark.parametrize("mode", ["engine", "decomp", "bt16", "auto", "small-bh"])
@pytest.mark.parametrize("heads", [6, 96])
@torch.inference_mode()
def test_prepared_schedule_matches_device_generation_across_layers_and_replays(
    switch_on: None, mode: str, heads: int
) -> None:
    kda_prefill_flashinfer = _flashinfer_prefill_op()
    torch.manual_seed(1901)
    metadata = KimiK3MambaMetadata(max_batch_size=4, chunk_size=128)
    metadata.prepare(_make_attention_metadata([0, 16, 47, 65]))
    offsets = metadata.query_start_loc_long
    slots = torch.tensor([4, 1, 3, 2], dtype=torch.int32, device="cuda")
    q, k, v, g = [
        torch.randn(1, 128, heads, 128, dtype=torch.bfloat16, device="cuda") * 0.2 for _ in range(4)
    ]
    beta = torch.randn(1, 128, heads, dtype=torch.bfloat16, device="cuda")
    a_log = torch.linspace(-2, 1, heads, device="cuda")
    bias = torch.linspace(-1, 1, heads * 128, device="cuda")
    row_size = heads * 128 * 128
    initial = torch.randn(6, row_size + 128, dtype=torch.bfloat16, device="cuda") * 0.1
    initial[slots[::2].long(), :row_size] = 0
    # Two independent layer pools consume the same prepared scheduling buffers.
    backing = [initial.clone(), initial.clone()]
    states = [tensor[:, :row_size].view(6, heads, 128, 128) for tensor in backing]
    pointers = (
        offsets.data_ptr(),
        metadata.kda_seq_order.data_ptr(),
        metadata.kda_cu_chunks.data_ptr(),
    )

    def run(prepared: bool) -> list[torch.Tensor]:
        return [
            kda_prefill_flashinfer(
                q,
                k,
                v,
                g,
                beta,
                state,
                slots,
                offsets,
                a_log,
                bias,
                128**-0.5,
                LOWER_BOUND,
                mode,
                128,
                metadata.kda_seq_order if prepared else None,
                metadata.kda_cu_chunks if prepared else None,
            )
            for state in states
        ]

    run(True)
    run(False)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = run(True)
    # Fixed token/sequence capacity, but different ordering and chunk counts.
    for lengths in ([0, 16, 47, 65], [17, 15, 64, 32], [64, 0, 0, 64]):
        metadata.prepare(_make_attention_metadata(lengths))
        assert pointers == (
            metadata.query_start_loc_long.data_ptr(),
            metadata.kda_seq_order.data_ptr(),
            metadata.kda_cu_chunks.data_ptr(),
        )
        immutable = [q, k, v, g, beta, offsets, slots, a_log, bias]
        immutable += [metadata.kda_seq_order, metadata.kda_cu_chunks]
        snapshots = [tensor.clone() for tensor in immutable]
        for tensor in backing:
            tensor.copy_(initial)
        expected = run(False)
        expected_states = [tensor.clone() for tensor in backing]
        for replay in (False, True):
            for tensor in backing:
                tensor.copy_(initial)
            if replay:
                graph.replay()
                actual = captured
            else:
                actual = run(True)
            for output, reference, tensor, reference_state in zip(
                actual, expected, backing, expected_states
            ):
                assert torch.isfinite(output).all()
                torch.testing.assert_close(output, reference, rtol=0, atol=0)
                torch.testing.assert_close(tensor, reference_state, rtol=0, atol=0)
                torch.testing.assert_close(tensor[[0, 5]], initial[[0, 5]], rtol=0, atol=0)
                torch.testing.assert_close(
                    tensor[:, row_size:], initial[:, row_size:], rtol=0, atol=0
                )
            for tensor, snapshot in zip(immutable, snapshots):
                torch.testing.assert_close(tensor, snapshot, rtol=0, atol=0)


# --------------------------------------------------------------------------- soundness
def _make_inputs(total_tokens: int, seed: int, batch_size: int = 1) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator(device="cuda").manual_seed(seed)

    def random_tensor(*shape: int, dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
        return torch.randn(*shape, generator=generator, dtype=torch.float32, device="cuda").to(
            dtype
        )

    q, k, v, g = (random_tensor(batch_size, total_tokens, NUM_HEADS, HEAD_DIM) for _ in range(4))
    beta = random_tensor(batch_size, total_tokens, NUM_HEADS, dtype=torch.float32)
    return q, k, v, g, beta


def _make_cu_seqlens(sequence_lengths: list[int]) -> torch.Tensor:
    offsets = [0, *torch.tensor(sequence_lengths).cumsum(0).tolist()]
    return torch.tensor(offsets, dtype=torch.long, device="cuda")


@pytest.fixture(scope="module")
def gate_params() -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cuda").manual_seed(0)
    a_log = torch.randn(NUM_HEADS, generator=generator, dtype=torch.float32, device="cuda") * 0.5
    dt_bias = (
        torch.randn(NUM_HEADS * HEAD_DIM, generator=generator, dtype=torch.float32, device="cuda")
        * 0.1
    )
    return a_log, dt_bias


def run_flashinfer_prefill(
    gate_params: tuple[torch.Tensor, torch.Tensor],
    inputs: tuple[torch.Tensor, ...],
    cu_seqlens: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The opt-in op on a fresh FP32 pool, glued as the dispatch glues it."""
    from tensorrt_llm._torch.modules.mamba.recurrent_state_cache import reset_recurrent_state_rows

    q, k, v, g, beta = inputs
    a_log, dt_bias = gate_params
    num_sequences = q.shape[0] if cu_seqlens is None else cu_seqlens.shape[0] - 1
    if cu_seqlens is None:
        cu_seqlens = (
            torch.arange(num_sequences + 1, device=q.device, dtype=torch.int64) * q.shape[1]
        )
        q, k, v, g = (t.reshape(1, -1, *t.shape[2:]) for t in (q, k, v, g))
        beta = beta.reshape(1, -1, beta.shape[-1])
    else:
        cu_seqlens = cu_seqlens.to(torch.int64)
    indices = torch.arange(num_sequences, dtype=torch.int32, device=q.device)
    pool = torch.full(
        (num_sequences, q.shape[-2], v.shape[-1], k.shape[-1]),
        float("nan"),
        dtype=torch.float32,
        device=q.device,
    )
    fresh = torch.zeros(num_sequences, dtype=torch.bool, device=q.device)
    reset_recurrent_state_rows(pool, indices, fresh)
    output = _flashinfer_prefill_op()(
        *(t.clone() for t in (q, k, v, g)),
        beta.to(torch.bfloat16),
        pool,
        indices,
        cu_seqlens,
        a_log.float().contiguous(),
        dt_bias.float().contiguous(),
        k.shape[-1] ** -0.5,
        LOWER_BOUND,
    )
    return output.reshape_as(inputs[0]), pool


def _fla_reference(
    gate_params: tuple[torch.Tensor, torch.Tensor],
    inputs: tuple[torch.Tensor, ...],
    cu_seqlens: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """FLA ``chunk_kda`` on raw Q/K (L2 norm in the kernel), as the FlashInfer op takes them.

    The in-tree dispatch's FLA branch expects Q/K already normalized by the caller, so it
    is not the reference for an op that normalizes internally.
    """
    from fla.ops.kda import chunk_kda

    q, k, v, g, beta = (t.clone() for t in inputs)
    a_log, dt_bias = gate_params
    output, final_state = chunk_kda(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        A_log=a_log,
        dt_bias=dt_bias,
        scale=k.shape[-1] ** -0.5,
        initial_state=None,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        use_beta_sigmoid_in_kernel=True,
        safe_gate=True,
        lower_bound=LOWER_BOUND,
        state_v_first=True,
        cu_seqlens=None if cu_seqlens is None else cu_seqlens.clone(),
    )
    return output, final_state


@needs_flashinfer_kda
@pytest.mark.parametrize("regime", ["eqlen", "varlen"])
@torch.no_grad()
def test_flashinfer_prefill_uses_current_stream(
    switch_on: None, gate_params: tuple[torch.Tensor, torch.Tensor], regime: str
) -> None:
    """Inputs, indexed state updates and consumers stay on the current stream."""
    pytest.importorskip("fla")
    from tests.unittest._torch.modules.kimi_kda.kda_prefill_test_utils import assert_kda_close

    lengths = [100, 257, 300]
    if regime == "eqlen":
        inputs = _make_inputs(256, seed=31337, batch_size=2)
        cu_seqlens = None
    else:
        inputs = _make_inputs(sum(lengths), seed=31337)
        cu_seqlens = _make_cu_seqlens(lengths)
    reference_output, reference_state = _fla_reference(gate_params, inputs, cu_seqlens)
    run_flashinfer_prefill(gate_params, inputs, cu_seqlens)
    torch.cuda.synchronize()

    execution_stream = torch.cuda.Stream()
    with torch.cuda.stream(execution_stream):
        torch.cuda._sleep(1 << 28)
        streamed = tuple(tensor * 1.0 for tensor in inputs)
        streamed_cu = _make_cu_seqlens(lengths) if regime == "varlen" else None
        output, pool = run_flashinfer_prefill(gate_params, streamed, streamed_cu)
        output = output * 1.0
        state = pool * 1.0
    torch.cuda.synchronize()
    assert_kda_close(f"stream_{regime}/output", output, reference_output)
    assert_kda_close(f"stream_{regime}/state", state, reference_state)


@needs_flashinfer_kda
@pytest.mark.parametrize(
    "aligned_lengths,unaligned_lengths",
    [([128, 256, 192], [100, 257, 219]), ([256], [300])],
    ids=["varlen", "single_sequence"],
)
@torch.no_grad()
def test_flashinfer_replaced_packed_metadata_matches_fla(
    switch_on: None,
    gate_params: tuple[torch.Tensor, torch.Tensor],
    aligned_lengths: list[int],
    unaligned_lengths: list[int],
) -> None:
    """Each invocation uses its current packed sequence boundaries."""
    pytest.importorskip("fla")
    from tests.unittest._torch.modules.kimi_kda.kda_prefill_test_utils import assert_kda_close

    for iteration, lengths in enumerate((aligned_lengths, unaligned_lengths, aligned_lengths)):
        inputs = _make_inputs(sum(lengths), seed=11 + iteration)
        cu = _make_cu_seqlens(lengths)
        output, pool = run_flashinfer_prefill(gate_params, inputs, cu)
        expected_output, expected_state = _fla_reference(gate_params, inputs, cu)
        assert_kda_close("metadata/output", output, expected_output)
        assert_kda_close("metadata/state", pool, expected_state)
