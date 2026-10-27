# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Steady one-model MTP decode prepare: the staging kernel writes what the full prepare pass writes, and the
recorded batch / block-table key invalidate on any change."""

import types

import pytest
import torch
from _torch.helpers import create_mock_cuda_graph_runner

from tensorrt_llm._torch.pyexecutor.cuda_graph_runner import KeyType
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2
from tensorrt_llm._torch.pyexecutor.steady_spec_prepare import SteadyMtpPrepare, stage_steady_inputs
from tensorrt_llm._torch.speculative.spec_sampler_base import SampleStateTensorsSpec

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

SLOTS, MAX_TOKENS, MAX_DRAFT, BATCH = 8, 64, 8, 4


def _buffers():
    i32 = dict(dtype=torch.int32, device="cuda")
    return dict(
        input_ids=torch.full((MAX_TOKENS,), -1, **i32),
        draft_tokens=torch.full((MAX_TOKENS,), -1, **i32),
        prev_batch_idx=torch.full((MAX_TOKENS,), -1, **i32),
        prev_pos_idx=torch.full((MAX_TOKENS,), -1, **i32),
        prev_pos_offsets=torch.zeros((MAX_TOKENS,), **i32),
        prev_kv_offsets=torch.zeros((BATCH,), **i32),
        positions=torch.full((MAX_TOKENS,), -1, **i32),
        mrope_positions=torch.full((3, 1, MAX_TOKENS), -1, **i32),
        gather_ids=torch.full((MAX_TOKENS,), -1, **i32),
        kv_lens_cuda=torch.full((BATCH,), -1, **i32),
    )


def _reference(buf, b, t, slots, positions, dev, kv_lens):
    """The full pass: DecoderInputPreparer's token / position / gather-id copies and kv_lens_cuda."""
    n = b * t
    pin = lambda x: torch.tensor(x, dtype=torch.int32).pin_memory()  # noqa: E731
    previous_slots = buf["prev_batch_idx"][:b]
    previous_slots.copy_(pin(slots))
    buf["input_ids"][:n].copy_(dev.new_tokens.transpose(0, 1)[previous_slots, :t].flatten())
    buf["draft_tokens"][: b * (t - 1)].copy_(
        dev.next_draft_tokens[previous_slots, : t - 1].flatten()
    )
    buf["prev_pos_idx"][:n].copy_(pin([s for s in slots for _ in range(t)]))
    buf["prev_pos_offsets"][:n].copy_(dev.new_tokens_lens[buf["prev_pos_idx"][:n]])
    buf["prev_kv_offsets"][:b].copy_((dev.new_tokens_lens - t)[previous_slots])
    buf["positions"][:n].copy_(pin(positions))
    buf["mrope_positions"][:, :, :n].copy_(buf["positions"][:n].view(1, 1, -1).expand(3, 1, -1))
    buf["gather_ids"][:n].copy_(pin(list(range(n))))
    if kv_lens is not None:
        buf["kv_lens_cuda"][:b].copy_(pin(kv_lens))


def _device_tensors(gen):
    rnd = lambda *shape: torch.randint(0, 1 << 20, shape, generator=gen, dtype=torch.int32)  # noqa: E731
    flat = rnd(SLOTS + SLOTS * MAX_DRAFT).cuda()
    return types.SimpleNamespace(
        new_tokens=rnd(MAX_DRAFT + 1, SLOTS, 1).cuda(),
        new_tokens_lens=torch.randint(
            1, MAX_DRAFT + 2, (SLOTS,), generator=gen, dtype=torch.int32
        ).cuda(),
        # a view into a larger buffer, like the fused sampler store
        next_draft_tokens=flat[SLOTS:].view(SLOTS, MAX_DRAFT),
    )


@pytest.mark.parametrize("host_slab", [False, True])
@pytest.mark.parametrize("block", [None, 64])
@pytest.mark.parametrize("b,t", [(1, 7), (3, 8), (4, 9)])
@pytest.mark.parametrize("mrope", [True, False])
def test_staging_kernel_matches_full_pass(mrope, b, t, block, host_slab):
    gen = torch.Generator().manual_seed(100 * b + t)
    n = b * t
    for _ in range(3):
        dev = _device_tensors(gen)
        slots = torch.randperm(SLOTS, generator=gen)[:b].tolist()
        bases = torch.randint(1000, 200000, (b,), generator=gen).tolist()
        positions = [p for base in bases for p in range(base, base + t)]
        kv_lens = [p + t for p in positions[::t]] if mrope else None
        expected, actual = _buffers(), _buffers()
        _reference(expected, b, t, slots, positions, dev, kv_lens)
        slab = torch.tensor(slots + positions + (kv_lens or []), dtype=torch.int32).pin_memory()
        static_ids = torch.full((MAX_TOKENS,), -1, dtype=torch.int32, device="cuda")
        static_pos = torch.full(
            (3, 1, MAX_TOKENS) if mrope else (1, MAX_TOKENS), -1, dtype=torch.int32, device="cuda"
        )
        stage_steady_inputs(
            slab if host_slab else slab.cuda(),
            b,
            t,
            dev,
            static_input_ids=static_ids,
            static_position_ids=static_pos,
            block=block,
            **{k: v for k, v in actual.items() if k not in ("mrope_positions", "kv_lens_cuda")},
            mrope_positions=actual["mrope_positions"] if mrope else None,
            kv_lens_cuda=actual["kv_lens_cuda"] if kv_lens else None,
        )
        torch.cuda.synchronize()
        for name in expected:
            if name != "mrope_positions" or mrope:
                assert torch.equal(actual[name], expected[name]), (b, t, name)
        assert torch.equal(static_ids[:n], expected["input_ids"][:n]) and int(static_ids[n]) == -1
        pos_ref = (
            expected["mrope_positions"][:, :, :n]
            if mrope
            else expected["positions"][:n].view(1, -1)
        )
        assert torch.equal(static_pos[..., :n], pos_ref)


def _manager(pools=2, rows=4, max_blocks=16):
    m = object.__new__(KVCacheManagerV2)
    m.host_kv_cache_block_offsets = torch.zeros(pools, rows, 2, max_blocks, dtype=torch.int32)
    m.kv_cache_map = {}
    rows_of = {}
    m.index_mapper = types.SimpleNamespace(rows=rows_of, get_index=rows_of.__getitem__)
    m.max_beam_width = 1
    m.max_seq_len = 4096
    m.max_attention_window_vec = [None]
    m.enable_swa_scratch_reuse = False
    m.enable_block_reuse = False
    m._use_per_layer_page_tables = False
    m.sparse_metadata_batch = None
    return m


def _add(m, rid, row, pages):
    m.index_mapper.rows[rid] = row
    m.kv_cache_map[rid] = types.SimpleNamespace(num_blocks=len(pages))
    for pool in range(m.host_kv_cache_block_offsets.shape[0]):
        m.host_kv_cache_block_offsets[pool, row, 0, : len(pages)] = (
            torch.tensor(pages) + 1000 * pool
        )


def test_block_key_sees_a_reallocated_last_block():
    m = _manager()
    _add(m, 5, 2, [7, 8, 9])
    _add(m, 6, 0, [3, 4])
    key = m.batch_block_table_key([5, 6])
    assert m.batch_block_table_key([5, 6]) == key
    # The rewind frees the last block and the next step re-allocates it on another page.
    _add(m, 6, 0, [3, 11])
    assert m.batch_block_table_key([5, 6]) != key
    assert m.batch_block_table_key([5, 7]) is None  # unknown request
    m.enable_block_reuse = True
    assert m.batch_block_table_key([5, 6]) is None


@pytest.mark.parametrize("change", ["window", "beams"])
def test_block_key_unavailable_where_earlier_blocks_or_other_beams_change(change):
    m = _manager()
    _add(m, 5, 0, [7, 8, 9])
    assert m.batch_block_table_key([5]) is not None
    if change == "window":
        m.max_attention_window_vec = [None, 1024]
    else:
        m.max_beam_width = 2
    assert m.batch_block_table_key([5]) is None


def _request(rid, slot, **overrides):
    fields = dict(
        py_request_id=rid,
        py_seq_slot=slot,
        py_batch_idx=slot,
        is_dummy=False,
        py_multimodal_data=None,
        py_needs_onehot_draft_probs=False,
    )
    fields.update(overrides)
    return types.SimpleNamespace(**fields)


def test_recorded_batch_matches_only_itself():
    t = 4
    state = SteadyMtpPrepare(
        max_batch_size=BATCH,
        max_num_tokens=MAX_TOKENS,
        max_tokens_per_step=t,
        use_mrope=False,
        has_static_inputs=False,
    )
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
        attn_metadata=attn,
        spec_metadata=spec,
        static_tensors=None,
    )
    dev = SampleStateTensorsSpec(
        new_tokens=None, next_draft_tokens=object(), new_tokens_lens=object()
    )

    def matches(reqs=requests, attn_metadata=attn, draft=t - 1):
        return state.matches(reqs, attn_metadata, spec, dev, draft)

    assert matches()
    assert not matches(reqs=requests[::-1])
    assert not matches(reqs=[requests[0], _request(6, 1)])
    assert not matches(reqs=[requests[0], _request(6, 0, is_dummy=True)])
    assert not matches(attn_metadata=types.SimpleNamespace(**vars(attn)))
    assert not matches(draft=t - 2)
    state.invalidate()
    assert not matches()


@pytest.mark.parametrize("use_mrope", [False, True])
def test_replay_skips_only_staged_static_copies(use_mrope):
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
        runner.replay(key, make_inputs(value), static_inputs_staged=staged)
        torch.cuda.synchronize()
        return int(static_ids[0]), int(static_pos.flatten()[0])

    assert replay(2, num_tokens) == (-7, -7)  # staged: the kernel's values stay
    assert replay(3, 0) == (3, 3)
    assert replay(4, num_tokens + 1) == (4, 4)  # staged for another token count
