"""Host-side tests of the shared FP4 MLA decode scratch (CPU tensors, fake buffer pool)."""

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.attention.backends.fp4_mla import fp4_mla_cutedsl_4x_plan as plan4x
from tensorrt_llm._torch.attention.backends.fp4_mla import layout, packed_plan

CPU = torch.device("cpu")


class _Block:
    def __init__(self, buffer: torch.Tensor, reserved: bool) -> None:
        self.buffer = buffer
        self.is_reserved = reserved


class _FakePool:
    """CPU stand-in for memory_buffer_utils.Buffers with its best-fit / reserve rules."""

    def __init__(self) -> None:
        self.buffers: dict[str, list[_Block]] = {}
        self.allocations = 0

    def get_buffer(self, shape, dtype, name, reserve):
        nbytes = dtype.itemsize
        for dim in shape:
            nbytes *= int(dim)
        best = None
        for block in self.buffers.get(name, []):
            if block.buffer.numel() >= nbytes and (
                best is None or block.buffer.numel() < best.buffer.numel()
            ):
                best = block
        if best is None:
            self.allocations += 1
            best = _Block(torch.empty((nbytes,), dtype=torch.uint8), reserve)
            self.buffers.setdefault(name, []).append(best)
        elif reserve:
            best.is_reserved = True
        return best.buffer[:nbytes].view(dtype).view(list(shape))


def _metadata(max_sequences: int = 128) -> SimpleNamespace:
    return SimpleNamespace(
        fp4_mla_state=SimpleNamespace(workspaces={}),
        max_num_sequences=max_sequences,
        max_num_requests=max_sequences,
    )


@pytest.fixture
def pool(monkeypatch):
    fake = _FakePool()
    monkeypatch.setattr(layout, "_workspace_pool", lambda: fake)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    return fake


def test_buckets_share_one_block(pool):
    a, b = _metadata(), _metadata()
    ta = layout._ensure_workspace_tensor(
        a, "ws", (4, 8), dtype=torch.float32, device=CPU, capacity_numel=1000
    )
    tb = layout._ensure_workspace_tensor(
        b, "ws", (20, 10), dtype=torch.float32, device=CPU, capacity_numel=1000
    )
    assert ta.shape == (4, 8) and tb.shape == (20, 10)
    assert ta.data_ptr() == tb.data_ptr()
    assert pool.allocations == 1
    (key,) = pool.buffers
    assert key == "fp4_mla_ws:ws:cpu"
    (block,) = pool.buffers[key]
    assert block.buffer.numel() == 1000 * 4 and block.is_reserved
    # the per-state entry serves repeat requests without going back to the pool
    tc = layout._ensure_workspace_tensor(a, "ws", (2, 2), dtype=torch.float32, device=CPU)
    assert tc.data_ptr() == ta.data_ptr()
    assert pool.allocations == 1


def test_growth_past_capacity_keeps_the_old_block(pool):
    a, b = _metadata(), _metadata()
    ta = layout._ensure_workspace_tensor(
        a, "ws", (10,), dtype=torch.int32, device=CPU, capacity_numel=16
    )
    tb = layout._ensure_workspace_tensor(
        b, "ws", (64,), dtype=torch.int32, device=CPU, capacity_numel=16
    )
    assert pool.allocations == 2
    (key,) = pool.buffers
    # the first block is still there, reserved: a captured graph may point at it
    assert sorted(block.buffer.numel() for block in pool.buffers[key]) == [16 * 4, 64 * 4]
    assert all(block.is_reserved for block in pool.buffers[key])
    assert ta.data_ptr() != tb.data_ptr()


def test_unshared_workspace_stays_per_state(pool):
    a, b = _metadata(), _metadata()
    ta = layout._ensure_workspace_tensor(a, "vp", (8,), dtype=torch.uint8, device=CPU, shared=False)
    tb = layout._ensure_workspace_tensor(b, "vp", (8,), dtype=torch.uint8, device=CPU, shared=False)
    assert ta.data_ptr() != tb.data_ptr()
    assert pool.allocations == 0


def test_capture_never_allocates(pool, monkeypatch):
    a = _metadata()
    layout._ensure_workspace_tensor(
        a, "ws", (8,), dtype=torch.float32, device=CPU, capacity_numel=32
    )
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    b = _metadata()
    # a hit inside the existing block is fine, a miss raises
    layout._ensure_workspace_tensor(b, "ws", (32,), dtype=torch.float32, device=CPU)
    with pytest.raises(ValueError, match="capturing"):
        layout._ensure_workspace_tensor(b, "ws", (33,), dtype=torch.float32, device=CPU)
    with pytest.raises(ValueError, match="capturing"):
        layout._ensure_workspace_tensor(b, "other", (1,), dtype=torch.float32, device=CPU)
    with pytest.raises(ValueError, match="capturing"):
        layout._ensure_workspace_tensor(b, "vp", (1,), dtype=torch.uint8, device=CPU, shared=False)
    assert pool.allocations == 1


def test_pool_key_names_device_and_workspace():
    assert layout._workspace_pool_key("x", torch.device("cpu")) == "fp4_mla_ws:x:cpu"
    assert layout._workspace_pool_key("x", torch.device("cuda", 3)) == "fp4_mla_ws:x:cuda3"


def test_max_sequences_prefers_sequences_then_requests():
    assert layout._max_sequences(SimpleNamespace(max_num_sequences=12, max_num_requests=8)) == 12
    assert layout._max_sequences(SimpleNamespace(max_num_sequences=None, max_num_requests=8)) == 8
    assert layout._max_sequences(SimpleNamespace()) == 0


@pytest.mark.parametrize("m_tiles", [1, 2, 4])
@pytest.mark.parametrize("max_requests", [1, 7, 23, 64, 128, 256])
def test_packed_split_task_bound_covers_every_batch(monkeypatch, m_tiles, max_requests):
    monkeypatch.delenv(packed_plan._FP4_MLA_SPLIT_KV_ENV, raising=False)
    monkeypatch.setattr(packed_plan, "_get_sm_count", lambda device: 212)
    bound = packed_plan.packed_split_task_bound(max_requests, m_tiles, CPU)
    worst = max(
        packed_plan._select_packed_split_count(
            requests, m_tiles, CPU, max_kv_len=1 << 20, is_cuda_graph=True
        )
        * requests
        for requests in range(1, max_requests + 1)
    )
    assert bound == max(worst, 32)
    # eager calls never exceed the graph rule: they only cap the splits further by the KV length
    for requests in (1, max_requests):
        for max_kv_len in (256, 4096, 1 << 20):
            splits = packed_plan._select_packed_split_count(
                requests, m_tiles, CPU, max_kv_len=max_kv_len, is_cuda_graph=False
            )
            assert max(splits * requests, 32) <= bound


def test_packed_split_task_bound_honours_the_override(monkeypatch):
    monkeypatch.setattr(packed_plan, "_get_sm_count", lambda device: 212)
    monkeypatch.setenv(packed_plan._FP4_MLA_SPLIT_KV_ENV, "5")
    assert packed_plan.packed_split_task_bound(100, 4, CPU) == 500


@pytest.mark.parametrize("pairs", [1, 2])
@pytest.mark.parametrize("max_requests", [1, 23, 46, 114, 128, 300])
def test_plan_4x_task_capacity_bound_is_the_max(pairs, max_requests):
    bound = plan4x.plan_4x_task_capacity_bound(max_requests, pairs)
    assert bound == max(
        plan4x.plan_4x_task_capacity(requests, pairs) for requests in range(1, max_requests + 1)
    )
    assert plan4x.plan_4x_task_capacity_bound(max_requests + 1, pairs) >= bound


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_real_pool_shares_one_cuda_block():
    """Two metadata objects (two CUDA-graph buckets) get views of one reserved block of the process pool."""
    from tensorrt_llm._torch.memory_buffer_utils import get_memory_buffers

    device = torch.device("cuda", torch.cuda.current_device())
    name = "_test_fp4_mla_workspace_shared"
    key = layout._workspace_pool_key(name, device)
    pool = get_memory_buffers()
    before = len(pool.buffers.get(key, []))
    a, b = _metadata(), _metadata()
    ta = layout._ensure_workspace_tensor(
        a, name, (3, 64), dtype=torch.float32, device=device, capacity_numel=4096
    )
    tb = layout._ensure_workspace_tensor(
        b, name, (16, 256), dtype=torch.float32, device=device, capacity_numel=4096
    )
    assert ta.data_ptr() == tb.data_ptr()
    assert ta.is_contiguous() and tb.is_contiguous()
    blocks = pool.buffers[key]
    assert len(blocks) == before + 1
    assert blocks[-1].is_reserved and blocks[-1].buffer.numel() == 4096 * 4
    tb.fill_(1.5)
    torch.cuda.synchronize()
    assert torch.equal(ta.flatten(), tb.flatten()[: ta.numel()])


def test_bounded_caller_upgrades_a_request_sized_block(pool):
    a, b, c = _metadata(), _metadata(), _metadata()
    ta = layout._ensure_workspace_tensor(a, "ws", (10,), dtype=torch.float32, device=CPU)
    assert pool.allocations == 1
    # a caller that knows the capacity bound replaces the request-sized block once ...
    tb = layout._ensure_workspace_tensor(
        b, "ws", (5,), dtype=torch.float32, device=CPU, capacity_numel=100
    )
    assert pool.allocations == 2 and tb.data_ptr() != ta.data_ptr()
    (key,) = pool.buffers
    assert sorted(block.buffer.numel() for block in pool.buffers[key]) == [10 * 4, 100 * 4]
    # ... and everybody after it fits in the capacity block
    tc = layout._ensure_workspace_tensor(
        c, "ws", (64,), dtype=torch.float32, device=CPU, capacity_numel=100
    )
    assert pool.allocations == 2 and tc.data_ptr() == tb.data_ptr()
    assert layout._workspace_capacity_numel(c, "ws", torch.float32, CPU) == 100
    assert layout._workspace_capacity_numel(c, "missing", torch.float32, CPU) == 0


def test_capture_hit_ignores_a_larger_bound(pool, monkeypatch):
    a, b = _metadata(), _metadata()
    layout._ensure_workspace_tensor(
        a, "ws", (8,), dtype=torch.float32, device=CPU, capacity_numel=8
    )
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    # inside capture a request that fits is served from the existing block even if its bound is larger
    tb = layout._ensure_workspace_tensor(
        b, "ws", (4,), dtype=torch.float32, device=CPU, capacity_numel=4096
    )
    assert tb.shape == (4,) and pool.allocations == 1


def test_latest_request_shape_is_cached_per_state(pool):
    a = _metadata()
    big = layout._ensure_workspace_tensor(
        a, "ws", (4, 8), dtype=torch.float32, device=CPU, capacity_numel=64
    )
    small = layout._ensure_workspace_tensor(a, "ws", (2, 3), dtype=torch.float32, device=CPU)
    assert small.shape == (2, 3) and small.data_ptr() == big.data_ptr()
    assert a.fp4_mla_state.workspaces["ws"].shape == (
        4,
        8,
    )  # the cached view keeps the larger request
    again = layout._ensure_workspace_tensor(a, "ws", (8, 8), dtype=torch.float32, device=CPU)
    assert again.shape == (8, 8) and again.data_ptr() == big.data_ptr() and pool.allocations == 1


def test_unshared_capacity_is_the_per_state_entry(pool):
    a = _metadata()
    layout._ensure_workspace_tensor(a, "vp", (6, 7), dtype=torch.uint8, device=CPU, shared=False)
    assert layout._workspace_capacity_numel(a, "vp", torch.uint8, CPU) == 42
    assert layout._workspace_capacity_numel(a, "vp", torch.float32, CPU) == 0


@pytest.mark.parametrize("max_requests", [64, 128, 256, 576])
def test_packed_split_scratch_bound_covers_every_query_width(monkeypatch, max_requests):
    monkeypatch.setenv("TRTLLM_FP4_MLA_PACKED_Q", "1")
    monkeypatch.delenv(packed_plan._FP4_MLA_SPLIT_KV_ENV, raising=False)
    monkeypatch.setattr(packed_plan, "_get_sm_count", lambda device: 212)
    meta = SimpleNamespace(max_num_sequences=max_requests, max_total_draft_tokens=3)
    slots, slot_rows = packed_plan.packed_split_scratch_bound(
        meta, 128, 4, CPU, max_requests=max_requests
    )
    per_width = []
    for q in (1, 2, 3, 4):
        rows = packed_plan._fp4_mla_packed_rows_per_seq(128, q)
        bound_q = packed_plan.packed_split_task_bound(max_requests, rows // 128, CPU)
        per_width.append((bound_q, bound_q * rows))
    assert slots == max(b for b, _ in per_width)
    assert slot_rows == max(r for _, r in per_width)
    # q = 1 asks for more slots than q = 4 at small engines: the bound must not depend on which call comes first
    if max_requests <= 256:
        assert per_width[0][0] > per_width[3][0]
