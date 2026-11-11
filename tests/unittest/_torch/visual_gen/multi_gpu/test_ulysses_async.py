# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Multi-rank tests for the async Ulysses A2A op.

Exercises the production ``ulysses_a2a_async_prepare`` / ``ulysses_a2a_async_push``
/ ``ulysses_a2a_async_barrier`` chain end-to-end across multiple ranks, plus
CPU-only dispatch coverage. Test surfaces:

1. ``test_slot_ring_wraparound`` — eager mode (cudaMemcpyBatchAsync path).
   Loops the pair more than ``kNumSlots`` times so the slot ring wraps multiple
   times, asserts byte-exact match vs ``all_to_all_4d`` on every iteration.
   Catches off-by-one slot-reuse bugs.

2. ``test_capture_smoke`` — under-capture mode (per-peer cudaMemcpyAsync path).
   Warms up the slot out-of-capture, then captures a ``torch.cuda.CUDAGraph``
   containing one prepare+async pair, replays it K times with fresh inputs.
   Smoke-tests the cuda_graph path used by the e2e production benchmark.

3. ``test_multi_pg`` — independent process groups and their slot rings.

4. ``test_forward_async_receive_view_gate`` — seven CPU dispatch cases covering
   opt-in receive views and the default/denied fused-copy path.

5. ``test_batch_one_receive_view_lifetime`` — batch-one NHD receive views,
   repeated slot reuse with a delayed reader, in eager and CUDA graph modes.

Run with:
    pytest tests/unittest/_torch/visual_gen/multi_gpu/test_ulysses_async.py -v
"""

import os

os.environ["TLLM_DISABLE_MPI"] = "1"

from typing import Callable

import pytest
import torch
import torch.distributed as dist

from tensorrt_llm._torch.distributed import all_to_all_4d

from ._worker_pool import run_distributed

# Loop count must comfortably exceed kNumSlots so the ring wraps at least
# twice. kNumSlots is 3 today; 8 iterations = ~2.67 full rotations.
NUM_ITERS = 8
K_NUM_SLOTS = 3
READ_DELAY_GPU_CYCLES = 2_000_000


def _slot_ring_logic(rank: int, world_size: int):
    """Loop the async A2A prepare/async pair NUM_ITERS times; assert byte-exact
    match against all_to_all_4d on every iteration.

    Stream pattern mirrors production (parallel.py:_issue_async): Phase 1 on
    the default (compute) stream, Phase 2 on a dedicated side stream gated by
    an event recorded after Phase 1.
    """
    device = torch.device(f"cuda:{rank}")
    torch.manual_seed(1234 + rank)

    # Production-scale shapes (LTX-2 ws=4: H=32, D=128, S_local=720 region).
    B, S_local, H, D = 2, 128, world_size * 8, 128
    pg = dist.group.WORLD
    pg_boxed = pg.boxed()
    side_stream = torch.cuda.Stream(device=device)

    for it in range(NUM_ITERS):
        # Fresh input each iteration (different bytes) so a stale slot would
        # produce a stale value and the byte-exact compare would catch it.
        torch.manual_seed(1000 * it + rank)
        x = torch.randn(B, S_local, H, D, dtype=torch.bfloat16, device=device)

        # Phase 1 on default stream.
        recv_5d, send_h = torch.ops.trtllm.ulysses_a2a_async_prepare(x, pg_boxed)
        ev = torch.cuda.Event()
        ev.record()
        # Phase 2 on side stream.
        with torch.cuda.stream(side_stream):
            ev.wait()
            torch.ops.trtllm.ulysses_a2a_async_push(send_h, pg_boxed)
            torch.ops.trtllm.ulysses_a2a_async_barrier(pg_boxed)
        ev_done = torch.cuda.Event()
        with torch.cuda.stream(side_stream):
            ev_done.record()
        torch.cuda.current_stream().wait_event(ev_done)
        torch.cuda.synchronize()

        # Async op produces [P, B, S_local, H/P, D]; permute to [B, P*S_local, H/P, D]
        # to match the all_to_all_4d output layout.
        async_out = (
            recv_5d.permute(1, 0, 2, 3, 4)
            .contiguous()
            .view(B, world_size * S_local, H // world_size, D)
        )

        ref = all_to_all_4d(x, scatter_dim=2, gather_dim=1, process_group=pg)

        assert torch.equal(async_out, ref), (
            f"rank {rank} iter {it}: async A2A ≠ all_to_all_4d reference "
            f"(slot ring wrap = {it % 3})"
        )


def _capture_smoke_logic(rank: int, world_size: int):
    """Smoke-test the under-capture branch: warm up slots out-of-capture, then
    capture a torch.cuda.CUDAGraph containing one prepare+async pair and replay
    K times with fresh inputs. Exercises the per-peer cudaMemcpyAsync loop
    (production cuda_graph path) instead of the eager cudaMemcpyBatchAsync.
    """
    device = torch.device(f"cuda:{rank}")
    torch.manual_seed(1234 + rank)

    B, S_local, H, D = 2, 128, world_size * 8, 128
    P = world_size
    pg = dist.group.WORLD
    pg_boxed = pg.boxed()
    side_stream = torch.cuda.Stream(device=device)

    # Static buffers (graph captures pointers, not values).
    x_static = torch.randn(B, S_local, H, D, dtype=torch.bfloat16, device=device)
    out_static = torch.empty(B, P * S_local, H // P, D, dtype=torch.bfloat16, device=device)

    # Warmup: must allocate ALL kNumSlots ring entries out-of-capture (allocation
    # is not capture-safe; the cudaStreamIsCapturing guard in getOrAllocSlot
    # enforces this). One warmup call only allocates 1 slot — the captured
    # _prepare advances mNextIdx and would hit an unallocated slot. So warm up
    # >= kNumSlots times. Mirror production's stream pattern (Phase 2 on side
    # stream) so cudaMemcpyBatchAsync sees the steady-state stream context.
    for _ in range(K_NUM_SLOTS):
        recv_w, sh_w = torch.ops.trtllm.ulysses_a2a_async_prepare(x_static, pg_boxed)
        ev_w = torch.cuda.Event()
        ev_w.record()
        with torch.cuda.stream(side_stream):
            ev_w.wait()
            torch.ops.trtllm.ulysses_a2a_async_push(sh_w, pg_boxed)
            torch.ops.trtllm.ulysses_a2a_async_barrier(pg_boxed)
        ev_w_done = torch.cuda.Event()
        with torch.cuda.stream(side_stream):
            ev_w_done.record()
        torch.cuda.current_stream().wait_event(ev_w_done)
        torch.cuda.synchronize()
        del recv_w, sh_w

    # Capture.
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        recv_5d, send_h = torch.ops.trtllm.ulysses_a2a_async_prepare(x_static, pg_boxed)
        ev = torch.cuda.Event()
        ev.record()
        with torch.cuda.stream(side_stream):
            ev.wait()
            torch.ops.trtllm.ulysses_a2a_async_push(send_h, pg_boxed)
            torch.ops.trtllm.ulysses_a2a_async_barrier(pg_boxed)
        ev_done = torch.cuda.Event()
        with torch.cuda.stream(side_stream):
            ev_done.record()
        torch.cuda.current_stream().wait_event(ev_done)
        captured_out = recv_5d.permute(1, 0, 2, 3, 4).contiguous().view(B, P * S_local, H // P, D)
        out_static.copy_(captured_out)

    # Replay K times. K > kNumSlots*2 to exercise wrap-around under capture.
    K = 8
    for it in range(K):
        torch.manual_seed(2000 * it + rank)
        new_x = torch.randn(B, S_local, H, D, dtype=torch.bfloat16, device=device)
        x_static.copy_(new_x)
        g.replay()
        torch.cuda.synchronize()
        ref = all_to_all_4d(new_x, scatter_dim=2, gather_dim=1, process_group=pg)
        assert torch.equal(out_static, ref), (
            f"rank {rank} capture replay {it}: async A2A ≠ all_to_all_4d reference"
        )


def _multi_pg_logic(rank: int, world_size: int):
    """Two distinct PGs spanning the same ranks but with different group_names;
    alternate calls between them; verify each gets its own slot ring and
    byte-exact output vs all_to_all_4d on its own group.
    """
    device = torch.device(f"cuda:{rank}")
    torch.manual_seed(4242 + rank)

    B, S_local, H, D = 2, 64, world_size * 8, 128
    # Two PGs, same membership (all ranks), but new_group assigns distinct
    # group_names — so getOrCreateOp must cache one op per PG-name and
    # set_group_info must register both.
    pg_a = dist.new_group(ranks=list(range(world_size)))
    pg_b = dist.new_group(ranks=list(range(world_size)))
    assert pg_a.group_name != pg_b.group_name, "new_group should yield distinct names"
    pg_a_boxed = pg_a.boxed()
    pg_b_boxed = pg_b.boxed()
    side_stream = torch.cuda.Stream(device=device)

    def _issue(pg_boxed, pg_obj, x):
        recv_5d, send_h = torch.ops.trtllm.ulysses_a2a_async_prepare(x, pg_boxed)
        ev = torch.cuda.Event()
        ev.record()
        with torch.cuda.stream(side_stream):
            ev.wait()
            torch.ops.trtllm.ulysses_a2a_async_push(send_h, pg_boxed)
            torch.ops.trtllm.ulysses_a2a_async_barrier(pg_boxed)
        ev_done = torch.cuda.Event()
        with torch.cuda.stream(side_stream):
            ev_done.record()
        torch.cuda.current_stream().wait_event(ev_done)
        torch.cuda.synchronize()
        async_out = (
            recv_5d.permute(1, 0, 2, 3, 4)
            .contiguous()
            .view(B, world_size * S_local, H // world_size, D)
        )
        ref = all_to_all_4d(x, scatter_dim=2, gather_dim=1, process_group=pg_obj)
        return async_out, ref

    # Alternate between the two PGs for 2*kNumSlots iterations to exercise
    # both slot rings wrapping.
    for it in range(NUM_ITERS):
        torch.manual_seed(3000 * it + rank)
        x = torch.randn(B, S_local, H, D, dtype=torch.bfloat16, device=device)
        pg_boxed, pg_obj, tag = (pg_a_boxed, pg_a, "A") if it % 2 == 0 else (pg_b_boxed, pg_b, "B")
        async_out, ref = _issue(pg_boxed, pg_obj, x)
        assert torch.equal(async_out, ref), (
            f"rank {rank} iter {it} PG={tag}: async A2A ≠ all_to_all_4d reference"
        )


def _run(world_size: int, test_fn: Callable[..., None], *args: object) -> None:
    if torch.cuda.device_count() < world_size:
        pytest.skip(f"Test requires {world_size} GPUs, only {torch.cuda.device_count()} available")
    run_distributed(world_size, test_fn, *args, preload=("tensorrt_llm",))


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "batch_size,layout_name,expect_fused_copy,local_sequence_length,local_kv_sequence_length,issue_order,accepts_unowned_inputs",
    [
        pytest.param(1, "NHD", False, 3, 3, ("v", "q", "k"), True, id="nhd-b1-optin-view"),
        pytest.param(2, "NHD", True, 3, 3, ("v", "q", "k"), True, id="nhd-b2-optin-copy"),
        pytest.param(1, "HND", True, 3, 3, ("v", "q", "k"), True, id="hnd-b1-optin-copy"),
        pytest.param(1, "NHD", False, 3, 5, ("q", "k", "v"), True, id="nhd-b1-cross-optin-view"),
        pytest.param(1, "NHD", True, 3, 3, ("v", "q", "k"), None, id="nhd-b1-default-copy"),
        pytest.param(1, "NHD", True, 3, 3, ("v", "q", "k"), False, id="nhd-b1-denied-copy"),
        pytest.param(1, "NHD", True, 3, 5, ("q", "k", "v"), False, id="nhd-b1-cross-denied-copy"),
    ],
)
def test_forward_async_receive_view_gate(
    monkeypatch: pytest.MonkeyPatch,
    batch_size: int,
    layout_name: str,
    expect_fused_copy: bool,
    local_sequence_length: int,
    local_kv_sequence_length: int,
    issue_order: tuple[str, str, str],
    accepts_unowned_inputs: bool | None,
) -> None:
    from tensorrt_llm._torch.visual_gen.attention.backends import parallel

    world_size = 2
    local_heads = 2
    head_dim = 8

    def make_receive_buffer(sequence_length: int, offset: int) -> torch.Tensor:
        receive_shape = (world_size, batch_size, sequence_length, local_heads, head_dim)
        element_count = world_size * batch_size * sequence_length * local_heads * head_dim
        base = torch.arange(element_count, dtype=torch.float32).reshape(receive_shape)
        return (base + offset).to(torch.bfloat16)

    receive_buffers = tuple(
        make_receive_buffer(sequence_length, offset)
        for sequence_length, offset in zip(
            (local_sequence_length, local_kv_sequence_length, local_kv_sequence_length), (0, 1, 2)
        )
    )
    local_inputs = tuple(torch.empty(0) for _ in range(3))
    receive_by_input = {
        id(local_input): receive_buffer
        for local_input, receive_buffer in zip(local_inputs, receive_buffers)
    }
    layout = (
        parallel.AttentionTensorLayout.NHD
        if layout_name == "NHD"
        else parallel.AttentionTensorLayout.HND
    )
    fused_calls = []
    recorded = {}
    issued_inputs = []

    def issue_async(local_input: torch.Tensor) -> torch.Tensor:
        issued_inputs.append(local_input)
        return receive_by_input[id(local_input)]

    def reference_layout(receive_buffer: torch.Tensor, is_hnd: bool) -> torch.Tensor:
        output = receive_buffer.permute(1, 0, 2, 3, 4).reshape(
            batch_size, world_size * receive_buffer.shape[2], local_heads, head_dim
        )
        return output.transpose(1, 2).contiguous() if is_hnd else output

    def fused_post_unscatter(
        query_receive: torch.Tensor,
        key_receive: torch.Tensor,
        value_receive: torch.Tensor,
        *,
        is_hnd: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        fused_calls.append(True)
        # The native op allocates owned outputs; clones model that boundary here.
        return (
            reference_layout(query_receive, is_hnd).clone(),
            reference_layout(key_receive, is_hnd).clone(),
            reference_layout(value_receive, is_hnd).clone(),
        )

    def record_backend(**forward_kwargs: object) -> torch.Tensor:
        recorded.update(forward_kwargs)
        query = forward_kwargs["q"]
        key = forward_kwargs["k"]
        value = forward_kwargs["v"]
        assert isinstance(query, torch.Tensor)
        assert isinstance(key, torch.Tensor)
        assert isinstance(value, torch.Tensor)
        if local_sequence_length != local_kv_sequence_length:
            sequence_dim = 2 if layout_name == "HND" else 1
            return (
                query
                + key.mean(dim=sequence_dim, keepdim=True)
                + value.mean(dim=sequence_dim, keepdim=True)
            )
        return query + key + value

    monkeypatch.setattr(parallel, "_ulysses_post_unscatter", fused_post_unscatter)

    class RecordingBackend(parallel.AttentionBackend):
        preferred_layout = layout
        num_heads = local_heads
        head_dim = receive_buffers[0].shape[-1]

        def forward(self, **forward_kwargs: object) -> torch.Tensor:
            return record_backend(**forward_kwargs)

    if accepts_unowned_inputs is not None:
        RecordingBackend.accepts_unowned_inputs = accepts_unowned_inputs
    attention = object.__new__(parallel.UlyssesAttention)
    attention.inner_backend = RecordingBackend()
    attention.process_group = object()
    attention.world_size = world_size
    monkeypatch.setattr(attention, "_issue_async", issue_async)
    monkeypatch.setattr(attention, "_join_async", lambda: None)
    monkeypatch.setattr(
        attention, "_output_a2a", lambda output, batch_argument, sequence_argument: output
    )
    output = attention.forward_async(
        lambda: local_inputs[0],
        lambda: local_inputs[1],
        lambda: local_inputs[2],
        issue_order=issue_order,
    )
    is_hnd = layout == parallel.AttentionTensorLayout.HND
    expected = tuple(reference_layout(receive_buffer, is_hnd) for receive_buffer in receive_buffers)
    expected_output = (
        expected[0] + expected[1] + expected[2]
        if local_sequence_length == local_kv_sequence_length
        else (
            expected[0]
            + expected[1].mean(dim=2 if is_hnd else 1, keepdim=True)
            + expected[2].mean(dim=2 if is_hnd else 1, keepdim=True)
        )
    )
    assert torch.equal(output, expected_output)
    assert issued_inputs[0] is local_inputs[("q", "k", "v").index(issue_order[0])]
    assert [id(local_input) for local_input in issued_inputs] == [
        id(local_inputs[("q", "k", "v").index(name)]) for name in issue_order
    ]
    assert bool(fused_calls) == expect_fused_copy
    assert len(fused_calls) == int(expect_fused_copy)
    assert recorded["batch_size"] == batch_size
    assert recorded["seq_len"] == world_size * local_sequence_length
    assert recorded["seq_len_kv"] == world_size * local_kv_sequence_length
    for argument_name, receive_buffer, expected_tensor in zip(
        ("q", "k", "v"), receive_buffers, expected
    ):
        tensor = recorded[argument_name]
        assert isinstance(tensor, torch.Tensor)
        assert torch.equal(tensor, expected_tensor)
        if not expect_fused_copy:
            assert tensor.data_ptr() == receive_buffer.data_ptr()
        else:
            assert tensor.data_ptr() != receive_buffer.data_ptr()


def test_slot_ring_wraparound():
    """Loop _prepare/_async ≥ 2*kNumSlots iterations on ws=2; assert byte-exact
    match against all_to_all_4d on each iteration (eager mode).
    """
    _run(2, _slot_ring_logic)


def test_capture_smoke():
    """Capture a CUDAGraph containing one _prepare/_async pair on ws=2; replay
    K iterations with fresh inputs; assert byte-exact match against
    all_to_all_4d (per-peer cudaMemcpyAsync path under capture).
    """
    _run(2, _capture_smoke_logic)


def test_multi_pg():
    """Two distinct ProcessGroups on ws=2; alternate _prepare/_async calls
    between them; assert byte-exact match against each group's all_to_all_4d.

    Exercises PG-name caching in ``getOrCreateOp`` and ``set_group_info``
    re-registration across multiple groups — each PG has its own group_name,
    so each must yield its own cached ``AsyncUlyssesOp`` instance + slot ring.
    """
    _run(2, _multi_pg_logic)


def _receive_view_lifetime_logic(
    rank: int, world_size: int, capture: bool, with_gates: bool
) -> None:
    """Read aliased receives after rank skew, before the next call reuses their slots."""
    from tensorrt_llm._torch.visual_gen.attention.backends.interface import AttentionTensorLayout
    from tensorrt_llm._torch.visual_gen.attention.backends.parallel import UlyssesAttention

    device = torch.device(f"cuda:{rank}")
    batch, seq, local_heads, head_dim = 1, 128, 4, 32
    shape = (batch, seq, world_size * local_heads, head_dim)
    pg = dist.group.WORLD
    # Keep enough layers to wrap the receive-slot ring repeatedly.
    layers, repeats = 2 * K_NUM_SLOTS + 2, 4
    received = []

    class DelayedNHDBackend:
        accepts_unowned_inputs = True
        preferred_layout = AttentionTensorLayout.NHD
        num_heads = local_heads
        num_kv_heads = local_heads
        head_dim = shape[-1]

        def forward(
            self,
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            **kwargs: torch.Tensor | int,
        ) -> torch.Tensor:
            # Confirm receive aliases, not the default fused-copy path.
            recv_v, recv_q, recv_k = received[-3:]
            assert q.data_ptr() == recv_q.data_ptr()
            assert k.data_ptr() == recv_k.data_ptr()
            assert v.data_ptr() == recv_v.data_ptr()
            assert kwargs["batch_size"] == batch
            assert kwargs["seq_len"] == world_size * seq
            assert kwargs["seq_len_kv"] == world_size * seq
            # Delay the READ, not just the next issue. Without the reverse collective's
            # ordering, the faster peer can push the next layer into these same slots.
            if rank == 0:
                torch.cuda._sleep(READ_DELAY_GPU_CYCLES)
            out = q + k + v
            if with_gates:
                out = out + kwargs["gate_compress"] + kwargs["gate_fine"]
            return out

    attention = UlyssesAttention(DelayedNHDBackend(), pg, async_ulysses=True)
    issue = attention._issue_async

    def record_issue(x: torch.Tensor) -> torch.Tensor:
        recv = issue(x)
        received.append(recv)
        return recv

    attention._issue_async = record_issue
    inputs = []
    gates = []
    for layer in range(layers):
        # Values stay below bf16's exact-integer range, including all replay updates.
        inputs.append(
            tuple(
                torch.full(shape, 8 * layer + 2 * rank + i, dtype=torch.bfloat16, device=device)
                for i in range(3)
            )
        )
        gates.append(
            tuple(torch.full(shape, i, dtype=torch.bfloat16, device=device) for i in (1, 2))
        )

    def reference() -> list[torch.Tensor]:
        expected = []
        for qkv, layer_gates in zip(inputs, gates):
            q, k, v = (all_to_all_4d(x, 2, 1, pg) for x in qkv)
            out = q + k + v
            if with_gates:
                gc, gf = (all_to_all_4d(x, 2, 1, pg) for x in layer_gates)
                out = out + gc + gf
            expected.append(all_to_all_4d(out, 1, 2, pg))
        return expected

    def run_chain() -> list[torch.Tensor]:
        outputs = []
        for qkv, layer_gates in zip(inputs, gates):
            kwargs = (
                dict(gate_compress=layer_gates[0], gate_fine=layer_gates[1]) if with_gates else {}
            )
            outputs.append(attention.forward_async(*(lambda x=x: x for x in qkv), **kwargs))
        return outputs

    # Warm every IPC slot, the shared side stream, all NCCL collectives, and tensor ops
    # before capture. Synchronization is intentionally OUTSIDE the layer chain.
    for _ in range(2):
        run_chain()
    torch.cuda.synchronize()
    received.clear()
    graph = None
    if capture:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            outputs = run_chain()
    for replay in range(repeats):
        for qkv in inputs:
            for x in qkv:
                x.add_(1)
        expected = reference()
        if graph is None:
            received.clear()
            outputs = run_chain()
        else:
            graph.replay()
        # No intermediate synchronize, clone, reference collective, or host tensor read
        # occurs between the real forward_async calls.
        torch.cuda.synchronize()
        for layer, (got, want) in enumerate(zip(outputs, expected)):
            assert torch.equal(got, want), (
                f"rank={rank},capture={capture},gates={with_gates},replay={replay},layer={layer}"
            )


@pytest.mark.parametrize("capture", [False, True])
@pytest.mark.parametrize("with_gates", [False, True])
def test_batch_one_receive_view_lifetime(capture: bool, with_gates: bool) -> None:
    _run(2, _receive_view_lifetime_logic, capture, with_gates)
