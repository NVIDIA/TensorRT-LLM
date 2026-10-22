# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Steady one-model MTP decode prepare (q35_steady_prepare.py): the staging kernel writes bitwise what the
full DecoderInputPreparer pass writes; the skip keys and recorded batches invalidate on any change."""

import types

import pytest
import torch
from _torch.helpers import create_mock_cuda_graph_runner

from tensorrt_llm._torch.pyexecutor import q35_steady_prepare as steady
from tensorrt_llm._torch.pyexecutor.cuda_graph_runner import KeyType
from tensorrt_llm._torch.pyexecutor.q35_steady_prepare import SteadyMtpPrepare, stage_steady_inputs

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

SLOTS, MAX_TOKENS, MAX_DRAFT, BATCH = 8, 64, 8, 4


def _buffers():
    i32 = dict(dtype=torch.int32, device="cuda")
    return dict(
        input_ids=torch.full((MAX_TOKENS,), -1, **i32),
        draft_tokens=torch.full((MAX_TOKENS,), -1, **i32),
        num_accepted=torch.full((BATCH,), -1, **i32),
        prev_batch_idx=torch.full((MAX_TOKENS,), -1, **i32),
        prev_pos_idx=torch.full((MAX_TOKENS,), -1, **i32),
        prev_pos_offsets=torch.zeros((MAX_TOKENS,), **i32),
        prev_kv_offsets=torch.zeros((BATCH,), **i32),
        positions=torch.full((MAX_TOKENS,), -1, **i32),
        mrope_positions=torch.full((3, 1, MAX_TOKENS), -1, **i32),
        gather_ids=torch.full((MAX_TOKENS,), -1, **i32),
        kv_lens_cuda=torch.full((BATCH,), -1, **i32),
    )


def _reference(buf, b, t, slots, accepted, positions, dev, kv_lens=None):
    """The full pass's ops for an all-extend, all-previous-tensor batch.

    DecoderInputPreparer._copy_token_ids_to_device, _copy_position_ids_to_device, the gather-id
    copy of prepare_inputs and TrtllmAttentionMetadata.prepare()'s kv_lens_cuda copy.
    """
    d = t - 1
    n = b * t
    pin = lambda x: torch.tensor(x, dtype=torch.int32).pin_memory()  # noqa: E731
    buf["num_accepted"][:b].copy_(pin(accepted), non_blocking=True)
    buf["prev_pos_offsets"].zero_()
    buf["prev_kv_offsets"].zero_()
    previous_slots = buf["prev_batch_idx"][:b]
    previous_slots.copy_(pin(slots), non_blocking=True)
    new_tokens = dev.new_tokens.transpose(0, 1)[previous_slots, :t].flatten()
    buf["input_ids"][:n].copy_(new_tokens, non_blocking=True)
    buf["draft_tokens"][: b * d].copy_(
        dev.next_draft_tokens[previous_slots, :d].flatten(), non_blocking=True
    )
    kv_len_offsets = dev.new_tokens_lens - t
    buf["prev_pos_idx"][:n].copy_(pin([s for s in slots for _ in range(t)]), non_blocking=True)
    buf["prev_pos_offsets"][:n].copy_(
        dev.new_tokens_lens[buf["prev_pos_idx"][:n]], non_blocking=True
    )
    buf["prev_kv_offsets"][:b].copy_(kv_len_offsets[previous_slots], non_blocking=True)
    buf["positions"][:n].copy_(pin(positions), non_blocking=True)
    buf["mrope_positions"][:, :, :n].copy_(
        buf["positions"][:n].view(1, 1, -1).expand(3, 1, -1), non_blocking=True
    )
    buf["gather_ids"][:n].copy_(pin(list(range(n))), non_blocking=True)
    if kv_lens is not None:
        buf["kv_lens_cuda"][:b].copy_(pin(kv_lens), non_blocking=True)


def _step(gen, b, t):
    slots = torch.randperm(SLOTS, generator=gen)[:b].tolist()
    accepted = torch.randint(0, t, (b,), generator=gen).tolist()
    bases = torch.randint(1000, 200000, (b,), generator=gen).tolist()
    positions = [p for base in bases for p in range(base, base + t)]
    return slots, accepted, positions


def _device_tensors(gen):
    rnd = lambda *shape: torch.randint(0, 1 << 20, shape, generator=gen, dtype=torch.int32)  # noqa: E731
    flat = rnd(MAX_DRAFT + 1 + SLOTS * 1 + SLOTS * MAX_DRAFT).cuda()
    return types.SimpleNamespace(
        new_tokens=rnd(MAX_DRAFT + 1, SLOTS, 1).cuda(),
        new_tokens_lens=torch.randint(
            1, MAX_DRAFT + 2, (SLOTS,), generator=gen, dtype=torch.int32
        ).cuda(),
        # a view into a larger buffer, like the fused sampler store
        next_draft_tokens=flat[MAX_DRAFT + 1 + SLOTS :].view(SLOTS, MAX_DRAFT),
    )


def _engine_kwargs(actual):
    return {k: v for k, v in actual.items() if k not in ("mrope_positions", "kv_lens_cuda")}


@pytest.mark.parametrize("block", [None, 64])
@pytest.mark.parametrize("b,t", [(1, 7), (3, 8), (4, 9)])
@pytest.mark.parametrize("mrope,with_kv", [(True, True), (False, False)])
def test_staging_kernel_matches_full_pass(b, t, mrope, with_kv, block):
    gen = torch.Generator().manual_seed(100 * b + t)
    for _ in range(3):
        dev = _device_tensors(gen)
        slots, accepted, positions = _step(gen, b, t)
        kv_lens = [p + t for p in positions[::t]] if with_kv else None
        expected, actual = _buffers(), _buffers()
        _reference(expected, b, t, slots, accepted, positions, dev, kv_lens)
        slab = torch.tensor(
            slots + accepted + positions + (kv_lens or []), dtype=torch.int32, device="cuda"
        )
        stage_steady_inputs(
            slab,
            b,
            t,
            dev,
            mrope_positions=actual["mrope_positions"] if mrope else None,
            kv_lens_cuda=actual["kv_lens_cuda"] if with_kv else None,
            block=block,
            **_engine_kwargs(actual),
        )
        torch.cuda.synchronize()
        for name in expected:
            if name == "mrope_positions" and not mrope:
                continue
            assert torch.equal(actual[name], expected[name]), (b, t, name)


@pytest.mark.parametrize("mrope", [True, False])
def test_static_graph_inputs_equal_engine_buffers(mrope):
    """The graph's static ids / positions get exactly what CUDAGraphRunner.replay() would copy."""
    gen = torch.Generator().manual_seed(11)
    b, t = 4, 7
    dev = _device_tensors(gen)
    slots, accepted, positions = _step(gen, b, t)
    actual = _buffers()
    static_ids = torch.full((32,), -1, dtype=torch.int32, device="cuda")
    static_pos = torch.full((3, 1, 32) if mrope else (1, 32), -1, dtype=torch.int32, device="cuda")
    slab = torch.tensor(slots + accepted + positions, dtype=torch.int32, device="cuda")
    stage_steady_inputs(
        slab,
        b,
        t,
        dev,
        mrope_positions=actual["mrope_positions"] if mrope else None,
        static_input_ids=static_ids,
        static_position_ids=static_pos,
        **_engine_kwargs(actual),
    )
    torch.cuda.synchronize()
    n = b * t
    assert torch.equal(static_ids[:n], actual["input_ids"][:n])
    if mrope:
        assert torch.equal(static_pos[:, :, :n], actual["mrope_positions"][:, :, :n])
    else:
        assert torch.equal(static_pos[:, :n], actual["positions"][:n].view(1, -1))
    assert int(static_ids[n]) == -1  # nothing past the batch


def _variants():
    caches = getattr(steady._steady_stage_kernel, "device_caches", None)
    if caches is None:
        pytest.skip("Triton without JITFunction.device_caches")
    return sum(len(entry[0]) for entry in caches.values())


def test_one_kernel_variant_for_all_batch_sizes():
    """With a fixed block the staging kernel compiles once for b = 1..4 (ints and alignments
    unspecialized); negative control: the per-batch block compiles a variant per batch size."""
    gen = torch.Generator().manual_seed(3)
    t = 8  # with kv lens and static inputs: a combination no other test launches

    def launch(b, block):
        dev = _device_tensors(gen)
        slots, accepted, positions = _step(gen, b, t)
        kv_lens = [p + t for p in positions[::t]]
        slab = torch.tensor(
            slots + accepted + positions + kv_lens, dtype=torch.int32, device="cuda"
        )
        actual = _buffers()
        static_ids = torch.full((MAX_TOKENS,), -1, dtype=torch.int32, device="cuda")
        static_pos = torch.full((3, 1, MAX_TOKENS), -1, dtype=torch.int32, device="cuda")
        stage_steady_inputs(
            slab,
            b,
            t,
            dev,
            mrope_positions=actual["mrope_positions"],
            kv_lens_cuda=actual["kv_lens_cuda"],
            static_input_ids=static_ids,
            static_position_ids=static_pos,
            block=block,
            **_engine_kwargs(actual),
        )
        torch.cuda.synchronize()
        n = b * t
        assert torch.equal(static_ids[:n], actual["input_ids"][:n])

    before = _variants()
    for b in (1, 2, 3, 4):
        launch(b, 64)
    assert _variants() - before == 1
    before = _variants()
    for b in (1, 2, 4):
        launch(b, None)  # BLOCK 8, 16, 32
    assert _variants() - before == 3


def test_zero_copy_slab_matches_device_slab():
    """A pinned host slab read through its device mapping gives the same writes as a device slab."""
    gen = torch.Generator().manual_seed(25)
    b, t = 4, 7
    dev = _device_tensors(gen)
    slots, accepted, positions = _step(gen, b, t)
    kv_lens = [p + t for p in positions[::t]]
    host = torch.tensor(slots + accepted + positions + kv_lens, dtype=torch.int32).pin_memory()
    results = []
    for slab in (host.cuda(), host):
        actual = _buffers()
        stage_steady_inputs(
            slab,
            b,
            t,
            dev,
            mrope_positions=actual["mrope_positions"],
            kv_lens_cuda=actual["kv_lens_cuda"],
            block=64,
            **_engine_kwargs(actual),
        )
        torch.cuda.synchronize()
        results.append(actual)
    for name in results[0]:
        assert torch.equal(results[0][name], results[1][name]), name
    expected = _buffers()
    _reference(expected, b, t, slots, accepted, positions, dev, kv_lens)
    for name in expected:
        assert torch.equal(results[1][name], expected[name]), name


class _Cache:
    def __init__(self, n):
        self.num_blocks = n


class _Mapper:
    def __init__(self, rows):
        self.rows = rows

    def get_index(self, rid):
        return self.rows[rid]


def _manager(pools=2, rows=4, max_blocks=16):
    m = types.SimpleNamespace()
    m.host_kv_cache_block_offsets = torch.zeros(pools, rows, 2, max_blocks, dtype=torch.int32)
    m.kv_cache_map = {}
    m.index_mapper = _Mapper({})
    m.max_beam_width = 1
    return m


def _add(m, rid, row, pages):
    m.index_mapper.rows[rid] = row
    m.kv_cache_map[rid] = _Cache(len(pages))
    for pool in range(m.host_kv_cache_block_offsets.shape[0]):
        m.host_kv_cache_block_offsets[pool, row, 0, : len(pages)] = (
            torch.tensor(pages) + 1000 * pool
        )


def test_block_key_sees_a_reallocated_last_block():
    """The rewind frees the last block and the next step re-allocates it on another page at the
    same block count: the key changes (negative control: the block counts alone do not)."""
    m = _manager()
    _add(m, 5, 2, [7, 8, 9])
    _add(m, 6, 0, [3, 4])
    key = steady._block_counts(m, [5, 6])
    assert steady._block_counts(m, [5, 6]) == key
    # request 6: last block freed (rewind), then re-allocated on page 11
    m.kv_cache_map[6].num_blocks = 1
    assert steady._block_counts(m, [5, 6]) != key
    _add(m, 6, 0, [3, 11])
    new_key = steady._block_counts(m, [5, 6])
    assert new_key != key
    assert [k[0] for k in new_key] == [k[0] for k in key]  # same counts
    # a block appended
    _add(m, 5, 2, [7, 8, 9, 12])
    assert steady._block_counts(m, [5, 6]) != new_key
    # pages before the tail do not enter the key (they cannot change on this path)
    tail_key = steady._block_counts(m, [5, 6])
    m.host_kv_cache_block_offsets[0, 2, 0, 0] = 99
    assert steady._block_counts(m, [5, 6]) == tail_key


@pytest.mark.parametrize(
    "attr, value",
    [
        ("enable_block_reuse", True),
        ("enable_swa_scratch_reuse", True),
        ("_use_per_layer_page_tables", True),
        ("sparse_metadata_batch", object()),
    ],
)
def test_block_key_unavailable(attr, value):
    """No key (the block-offset copies always run) for an unknown request, a manager without the
    V2 host table, and one whose device table is not a plain copy of it or whose blocks can
    change otherwise."""
    m = _manager()
    _add(m, 5, 1, [1, 2])
    assert steady._block_counts(m, [5]) is not None
    assert steady._block_counts(m, [5, 7]) is None  # unknown request
    assert steady._block_counts(types.SimpleNamespace(kv_cache_map={}), [5]) is None
    setattr(m, attr, value)
    assert steady._block_counts(m, [5]) is None


def _request(rid, slot):
    return types.SimpleNamespace(
        py_request_id=rid,
        py_seq_slot=slot,
        py_batch_idx=slot,
        is_dummy=False,
        py_multimodal_data=None,
        py_needs_onehot_draft_probs=False,
    )


def test_recorded_batch_matches_only_itself():
    """A record matches the same requests on the same slots, metadata and draft length; any
    change sends the step to the full pass (negative controls)."""
    t = 4
    state = SteadyMtpPrepare(
        max_batch_size=BATCH,
        max_num_tokens=MAX_TOKENS,
        max_tokens_per_step=t,
        use_mrope=False,
        has_static_inputs=False,
    )
    # Not TrtllmAttentionMetadata: the step falls back to attn_metadata.prepare().
    attn = types.SimpleNamespace(
        is_cuda_graph=True, seq_lens_kv=torch.full((BATCH,), t, dtype=torch.int32)
    )
    spec = object()
    requests = [_request(5, 2), _request(6, 0)]
    state.record_batch(
        requests=requests,
        prompt_lengths=[100, 200],
        sequence_lengths=[t, t],
        tokens_per_step=t,
        runtime_draft_len=t - 1,
        num_extra_kv_tokens=t - 2,
        use_full_generation_page_table=False,
        kv_cache_manager=None,
        attn_metadata=attn,
        spec_metadata=spec,
        static_tensors=None,
    )
    rec = state.record
    assert rec is not None and not rec.attn_steady and rec.static_ids is None
    assert rec.request_ids == [5, 6] and rec.slots == [2, 0] and rec.seq_lens_kv == [t, t]
    dev = types.SimpleNamespace(next_draft_tokens=object(), new_tokens_lens=object())

    def matches(reqs=requests, attn_metadata=attn, spec_metadata=spec, device=dev, draft=t - 1):
        return state.matches(reqs, attn_metadata, spec_metadata, device, draft)

    assert matches()
    assert not matches(reqs=requests[:1])
    assert not matches(reqs=requests[::-1])
    assert not matches(reqs=[_request(5, 2), _request(6, 1)])  # slot changed
    assert not matches(attn_metadata=types.SimpleNamespace(**vars(attn)))
    assert not matches(spec_metadata=object())
    assert not matches(device=None)
    assert not matches(device=types.SimpleNamespace(next_draft_tokens=None, new_tokens_lens=1))
    assert not matches(draft=t - 2)
    for name, value in (
        ("is_dummy", True),
        ("py_multimodal_data", {}),
        ("py_needs_onehot_draft_probs", True),
        ("py_batch_idx", None),
    ):
        changed = _request(6, 0)
        setattr(changed, name, value)
        assert not matches(reqs=[requests[0], changed]), name
    state.invalidate()
    assert not matches()


@pytest.mark.parametrize("use_mrope", [False, True])
def test_replay_skips_only_staged_static_copies(use_mrope):
    """A replay skips its input-id / position copies only when the static inputs were staged for
    its token count, and clears the mark; any other replay copies them."""
    batch_size = 4
    runner = create_mock_cuda_graph_runner(batch_size, use_mrope=use_mrope, max_num_tokens=128)
    key = KeyType(batch_size=batch_size, draft_len=0)
    num_tokens = runner._get_num_tokens_for_key(key)
    attn_metadata = object()

    def make_inputs(value):
        shape = (3, 1, num_tokens) if use_mrope else (1, num_tokens)
        return {
            "attn_metadata": attn_metadata,
            "input_ids": torch.full((num_tokens,), value, device="cuda", dtype=torch.int32),
            "position_ids": torch.full(shape, value, device="cuda", dtype=torch.int32),
        }

    runner.capture(key, lambda inputs: inputs["input_ids"].clone(), make_inputs(1))
    static_ids = runner.shared_static_tensors["input_ids"][:num_tokens]
    static_pos = runner.shared_static_tensors["position_ids"][..., :num_tokens]

    def replay(value, staged):
        static_ids.fill_(-7)
        static_pos.fill_(-7)
        runner.set_static_inputs_staged(staged)
        runner.replay(key, make_inputs(value))
        torch.cuda.synchronize()
        return int(static_ids[0]), int(static_pos.flatten()[0])

    assert replay(2, num_tokens) == (-7, -7)  # staged: the kernel's values stay
    assert replay(3, None) == (3, 3)  # the mark was cleared by the replay
    assert replay(4, num_tokens + 1) == (4, 4)  # staged for another token count
