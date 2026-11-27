# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Steady one-model MTP decode prepare: the staging kernel writes what the full prepare pass writes, the
recorded batch / block-table key invalidate on any change, and consecutive steady prepare_inputs steps hand
the model what forced full passes hand it."""

import types
from contextlib import nullcontext
from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch
from _torch.helpers import create_mock_cuda_graph_runner
from _torch.pyexecutor.engine.runners.decoder.utils import decoder_config, input_preparer
from _torch.pyexecutor.engine_test_utils import DummyModel

import tensorrt_llm
from tensorrt_llm._torch.attention.backends.interface import AttentionRuntimeFeatures
from tensorrt_llm._torch.attention.backends.trtllm import TrtllmAttentionMetadata
from tensorrt_llm._torch.kernels.spec_step_copies.op import slot_scatter_supported
from tensorrt_llm._torch.pyexecutor import steady_spec_prepare
from tensorrt_llm._torch.pyexecutor.cuda_graph_runner import KeyType
from tensorrt_llm._torch.pyexecutor.engine.runners.decoder.inputs import DecoderInputBuffers
from tensorrt_llm._torch.pyexecutor.engine.runners.decoder.metadata import DecoderMetadata
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2
from tensorrt_llm._torch.pyexecutor.resource_manager import ResourceManager, ResourceManagerType
from tensorrt_llm._torch.pyexecutor.sampler.sampler import TorchSampler
from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests
from tensorrt_llm._torch.pyexecutor.steady_spec_prepare import SteadyMtpPrepare, stage_steady_inputs
from tensorrt_llm._torch.speculative.spec_sampler_base import (
    SampleStateTensorsSpec,
    SpecSampler,
    StepOutputRows,
)
from tensorrt_llm.llmapi import MTPDecodingConfig
from tensorrt_llm.llmapi.llm_args import KvCacheConfig
from tensorrt_llm.mapping import Mapping

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

SLOTS, MAX_TOKENS, MAX_DRAFT, BATCH = 8, 64, 8, 4


def _buffers(batch: int = BATCH) -> dict:
    """The staging kernel's destination buffers, the per-request ones sized for ``batch`` requests."""
    i32 = dict(dtype=torch.int32, device="cuda")
    return dict(
        input_ids=torch.full((MAX_TOKENS,), -1, **i32),
        draft_tokens=torch.full((MAX_TOKENS,), -1, **i32),
        prev_batch_idx=torch.full((MAX_TOKENS,), -1, **i32),
        prev_pos_idx=torch.full((MAX_TOKENS,), -1, **i32),
        prev_pos_offsets=torch.zeros((MAX_TOKENS,), **i32),
        prev_kv_offsets=torch.zeros((batch,), **i32),
        positions=torch.full((MAX_TOKENS,), -1, **i32),
        mrope_positions=torch.full((3, 1, MAX_TOKENS), -1, **i32),
        gather_ids=torch.full((MAX_TOKENS,), -1, **i32),
        kv_lens_cuda=torch.full((batch,), -1, **i32),
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
@pytest.mark.parametrize("loads_first", [False, True])
def test_staging_kernel_matches_full_pass(mrope, b, t, block, host_slab, loads_first):
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
            loads_first=loads_first,
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


@pytest.mark.parametrize("host_slab", [False, True])
@pytest.mark.parametrize("b,t", [(1, 2), (5, 7)])
@pytest.mark.parametrize("mode", ["legacy", "loads_first", "padded", "output_rows", "admit"])
def test_staged_rng_matches_uploaded_rng(b, t, host_slab, mode):
    """With an RNG slab the kernel writes SpecMetadata's Philox buffers exactly as
    _populate_request_rng_state uploads them: request_* per request, and row j of request r as
    (seed_r, offset_r + j * _RNG_SLOT_SPAN)."""
    from tensorrt_llm._torch.speculative.interface import _RNG_SLOT_SPAN

    gen = torch.Generator().manual_seed(7 * b + t)
    dev = _device_tensors(gen)
    slots = torch.randperm(SLOTS, generator=gen)[:b].tolist()
    num_rows = b + 2 if mode == "padded" else b
    positions = [100 * row + column for row in range(num_rows) for column in range(t)]
    seeds = torch.randint(0, 1 << 62, (num_rows,), generator=gen, dtype=torch.int64).tolist()
    offsets = torch.randint(0, 1 << 40, (num_rows,), generator=gen, dtype=torch.int64).tolist()
    expected_seeds = torch.tensor([s for s in seeds for _ in range(t)], dtype=torch.int64)
    expected_offsets = torch.tensor(
        [o + j * _RNG_SLOT_SPAN for o in offsets for j in range(t)], dtype=torch.int64
    )

    rng_buffers = tuple(
        torch.full((MAX_TOKENS,), -1, dtype=torch.int64, device="cuda") for _ in range(4)
    )
    rng_slab = torch.tensor(seeds + offsets, dtype=torch.int64).pin_memory()
    from_host = mode == "admit"
    host_tokens = list(range(b * t)) if from_host else []
    slab = torch.tensor(slots + positions + host_tokens, dtype=torch.int32).pin_memory()
    rows = _with_output_rows(dev, slots).rows if mode == "output_rows" else None
    stage_steady_inputs(
        slab if host_slab else slab.cuda(),
        b,
        t,
        dev,
        **{
            k: v
            for k, v in _buffers(batch=num_rows).items()
            if k not in ("mrope_positions", "kv_lens_cuda")
        },
        mrope_positions=None,
        loads_first=mode != "legacy",
        num_rows=num_rows,
        from_host=from_host,
        rows=rows,
        rng_slab=rng_slab if host_slab else rng_slab.cuda(),
        rng_buffers=rng_buffers,
        rng_span=_RNG_SLOT_SPAN,
    )
    torch.cuda.synchronize()
    request_seeds, request_offsets, row_seeds, row_offsets = (x.cpu() for x in rng_buffers)
    n = num_rows * t
    assert request_seeds[:num_rows].tolist() == seeds
    assert request_offsets[:num_rows].tolist() == offsets
    assert torch.equal(row_seeds[:n], expected_seeds)
    assert torch.equal(row_offsets[:n], expected_offsets)
    assert request_seeds[num_rows] == request_offsets[num_rows] == -1
    assert row_seeds[n] == row_offsets[n] == -1


def _rng_meta():
    """Just enough of SpecMetadata to run _populate_request_rng_state."""
    from tensorrt_llm._torch.speculative.interface import SpecMetadata

    meta = types.SimpleNamespace(
        is_all_greedy_sample=False,
        max_num_requests=BATCH,
        request_seeds=None,
        request_offsets=None,
        seeds=None,
        offsets=None,
        temperatures=None,
        pack_rng_uploads=False,
        _rng_window_counter={},
        _rng_slot_owner={},
        rng_accept_slot=3,
    )
    meta._take_rng_window_offsets = lambda *args: SpecMetadata._take_rng_window_offsets(meta, *args)
    return meta


@pytest.mark.parametrize("uniform", [True, False], ids=["deferred", "uploaded"])
def test_populate_before_stage_matches_the_upload(uniform):
    """The steady MTP step populates the sampling parameters before staging (defer_rows = its rows per request) and
    the staging kernel writes the Philox buffers from the deferred values: over consecutive steps of seeded and
    unseeded requests, the same buffers and window counters as the uploads of the full pass. A batch whose requests'
    rows differ is uploaded."""
    from tensorrt_llm._torch.pyexecutor.sampler.sampler_common import request_random_seed
    from tensorrt_llm._torch.speculative.interface import (
        _RNG_SLOT_SPAN,
        OneModelSamplingParams,
        SpecMetadata,
    )

    t = 4
    seeds = [None, 123, None, (1 << 63) + 5]
    requests = [
        types.SimpleNamespace(
            sampling_config=types.SimpleNamespace(seed=seed),
            py_seq_slot=slot,
            py_request_id=100 + slot,
        )
        for slot, seed in zip((5, 2, 7, 0), seeds)
    ]
    for request in requests:
        # The sampling parameters LlmRequest caches; only the seed is read here.
        request.py_spec_sampling_params = OneModelSamplingParams(
            temperature=1.0,
            top_k=torch.iinfo(torch.int32).max,
            top_p=1.0,
            min_p=0.0,
            is_greedy=False,
            is_temperature_only=True,
            seed=request_random_seed(request),
        )
    b = len(requests)
    rows = [t] * b if uniform else [t, 1, t, t]
    normalized = [(1.0, 0, 1.0, 0.0, n) for n in rows]
    staged, uploaded = _rng_meta(), _rng_meta()
    dev = _device_tensors(torch.Generator().manual_seed(3))
    slab = torch.tensor([0, 1, 2, 3] + list(range(b * t)), dtype=torch.int32, device="cuda")
    for step in range(3):
        deferred = SpecMetadata._populate_request_rng_state(
            staged, requests, normalized, defer_rows=t
        )
        assert SpecMetadata._populate_request_rng_state(uploaded, requests, normalized) is None
        if uniform:
            request_seeds, request_offsets = deferred
            stage_steady_inputs(
                slab,
                b,
                t,
                dev,
                **{
                    k: v
                    for k, v in _buffers().items()
                    if k not in ("mrope_positions", "kv_lens_cuda")
                },
                mrope_positions=None,
                rng_slab=torch.tensor(
                    request_seeds + request_offsets, dtype=torch.int64, device="cuda"
                ),
                rng_buffers=(
                    staged.request_seeds,
                    staged.request_offsets,
                    staged.seeds,
                    staged.offsets,
                ),
                rng_span=_RNG_SLOT_SPAN,
            )
        else:
            assert deferred is None
        torch.cuda.synchronize()
        for name, n in (
            ("request_seeds", b),
            ("request_offsets", b),
            ("seeds", sum(rows)),
            ("offsets", sum(rows)),
        ):
            assert torch.equal(getattr(staged, name)[:n], getattr(uploaded, name)[:n]), (step, name)
        assert staged._rng_window_counter == uploaded._rng_window_counter


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


# Plain values and tensors of a metadata object (other objects are shared or derived).
_PLAIN = (int, float, bool, str, bytes, list, tuple, type(None), torch.Tensor)


def _snapshot(value):
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, (list, tuple)):
        return type(value)(_snapshot(v) for v in value)
    return value


def _same(a, b):
    if isinstance(a, torch.Tensor):
        return (
            isinstance(b, torch.Tensor)
            and (a.shape, a.dtype) == (b.shape, b.dtype)
            and torch.equal(a, b)
        )
    if isinstance(a, (list, tuple)):
        return type(a) is type(b) and len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    return a == b


def _step_state(preparer, inputs, attn_metadata, spec_metadata, b, t):
    """What a step hands the model: its inputs, the step buffers, every tensor and plain field of the
    attention metadata and its KV-cache params, and the speculative metadata fields both passes set
    (host_position_ids is read by DFlash only, which has no steady prepare)."""
    n = b * t
    buffers = preparer._buffers
    state = {
        "input_ids": inputs["input_ids"],
        "position_ids": inputs["position_ids"],
        "draft_tokens": buffers.draft_tokens_cuda[: n - b],
        "previous_batch_indices": buffers.previous_batch_indices_cuda[:b],
        "previous_pos_indices": buffers.previous_pos_indices_cuda[:n],
        "previous_pos_id_offsets": buffers.previous_pos_id_offsets_cuda[:n],
        "previous_kv_lens_offsets": buffers.previous_kv_lens_offsets_cuda[:b],
        "gather_ids": buffers.gather_ids_cuda[:n],
    }
    for prefix, obj in (
        ("attn", attn_metadata),
        ("kv_cache_params", attn_metadata.kv_cache_params),
    ):
        for name, value in vars(obj).items():
            if isinstance(value, _PLAIN):
                state[f"{prefix}.{name}"] = value
    for name in (
        "num_tokens",
        "num_generations",
        "seq_lens",
        "request_ids",
        "draft_tokens",
        "gather_ids",
    ):
        state[f"spec.{name}"] = getattr(spec_metadata, name)
    return {name: _snapshot(value) for name, value in state.items()}


def _repage_last_block(manager, request, other):
    """Bring the last block of ``request`` back on another page at its block count: free it, let
    ``other`` take a page, re-grow."""
    rid = request.py_request_id
    cache = manager.kv_cache_map[rid]
    other_cache = manager.kv_cache_map[other.py_request_id]
    capacity, before = cache.capacity, manager.batch_block_table_key([rid])
    for _ in range(4):
        assert cache.resize((cache.num_blocks - 1) * manager.tokens_per_block)
        assert other_cache.resize(other_cache.capacity + manager.tokens_per_block)
        assert cache.resize(capacity)
        after = manager.batch_block_table_key([rid])
        if after != before:
            break
    assert after[0][0] == before[0][0] and after != before


def _previous_step_tensors(gen, t, num_slots=8):
    """The previous step's device sampler outputs, per sequence slot."""

    def rand(low, high, shape):
        return torch.randint(low, high, shape, generator=gen, dtype=torch.int32, device="cuda")

    return SampleStateTensorsSpec(
        new_tokens=rand(0, 1000, (t, num_slots, 1)),
        new_tokens_lens=rand(1, t + 1, (num_slots,)),
        next_draft_tokens=rand(0, 1000, (num_slots, t - 1)),
    )


def _with_output_rows(stored, slots):
    """``stored`` plus the previous step's output rows of the requests on ``slots``, in order."""
    index = torch.tensor(slots, device="cuda")
    rows = StepOutputRows(
        next_new_tokens=stored.new_tokens[:, index, 0].t().contiguous(),
        new_tokens_lens=stored.new_tokens_lens[index],
        next_draft_tokens=stored.next_draft_tokens[index],
        slots=tuple(slots),
    )
    return SampleStateTensorsSpec(
        new_tokens=stored.new_tokens,
        new_tokens_lens=stored.new_tokens_lens,
        next_draft_tokens=stored.next_draft_tokens,
        rows=rows,
    )


def _sample_stores(sampler, stored, requests):
    """Scatter real sampler outputs in a different order from the next batch."""
    batch = ScheduledRequests()
    batch.generation_requests = list(reversed(requests))
    rows = _with_output_rows(stored, [r.py_seq_slot for r in batch.generation_requests]).rows
    outputs = dict(
        new_tokens=rows.next_new_tokens,
        new_tokens_lens=rows.new_tokens_lens,
        next_new_tokens=rows.next_new_tokens,
        next_draft_tokens=rows.next_draft_tokens,
        native_uniform_verify=True,
    )
    sampler.store.next_new_tokens.fill_(-1)
    sampler.store.next_draft_tokens.fill_(-1)
    sampler.store.new_tokens_lens.zero_()
    state = sampler.sample_async(batch, outputs, [])
    assert state.device.stores_ready is not None
    return state.device


@pytest.mark.parametrize("mode", ["plain", "tp", "attention_dp_padded"])
def test_prepare_inputs_steady_steps_match_forced_full_passes(mode):
    """Steady decode steps of DecoderInputPreparer on real TrtllmAttentionMetadata hand the model
    what forced full passes hand it; with steady_decode_prepare also with attention DP padding and
    at new requests' first steps."""
    draft_len, tokens_per_block = 3, 16
    t = draft_len + 1
    attention_dp = mode == "attention_dp_padded"
    opted_in = mode != "plain"
    config = decoder_config(
        spec_config=MTPDecodingConfig(max_draft_len=draft_len),
        spec_decode_fast_paths=not attention_dp,
        enable_attention_dp=attention_dp,
        steady_decode_prepare=opted_in,
    )
    exchanges = []

    def allgather(values):
        exchanges.append(list(values))
        return np.asarray([values], dtype=np.int64)

    model = DummyModel(config.dtype)
    manager = KVCacheManagerV2(
        KvCacheConfig(max_tokens=2048, enable_block_reuse=False),
        tensorrt_llm.bindings.internal.batch_manager.CacheType.SELF,
        num_layers=1,
        num_kv_heads=model.config.num_key_value_heads,
        head_dim=model.config.head_dim,
        tokens_per_block=tokens_per_block,
        max_seq_len=512,
        max_batch_size=config.max_batch_size,
        mapping=Mapping(),
        dtype=tensorrt_llm.bindings.DataType.HALF,
    )
    try:
        sampler = None
        if mode == "tp" and slot_scatter_supported():
            sampler = SpecSampler(
                TorchSampler.Args(
                    max_seq_len=512,
                    max_draft_len=draft_len,
                    max_num_sequences=SLOTS,
                    max_beam_width=1,
                    max_total_draft_tokens=draft_len,
                )
            )
            sampler.stores_on_side_stream = True
        requests = manager.add_dummy_requests(
            [1, 2, 3], token_nums=[10, 17, 33], is_gen=True, max_num_draft_tokens=draft_len
        )
        (other,) = manager.add_dummy_requests([4], token_nums=[3], is_gen=True)
        for request, slot in zip(requests, [5, 2, 7]):
            request.is_dummy_request = False
            request.py_seq_slot = request.py_batch_idx = slot
            assert manager.kv_cache_map[request.py_request_id].resize(
                request.max_beam_num_tokens + draft_len
            )
        request_ids = [request.py_request_id for request in requests]
        rows = list(requests)
        if attention_dp:
            (padding,) = manager.add_dummy_requests(
                [9], token_nums=[1], is_gen=True, max_num_draft_tokens=draft_len
            )
            padding.is_cuda_graph_dummy = True
            rows += [padding, padding]
        batch = ScheduledRequests()
        batch.generation_requests = rows
        resource_manager = ResourceManager({ResourceManagerType.KV_CACHE_MANAGER: manager})

        arms = {}
        for arm in ("steady", "full"):
            buffers = DecoderInputBuffers.for_config(config, False)
            for buffer in vars(buffers).values():
                if isinstance(buffer, torch.Tensor):
                    buffer.zero_()
            preparer = input_preparer(config, model=model, buffers=buffers)
            preparer.dist = types.SimpleNamespace(tp_cp_allgather_int64=allgather)
            disabled_graph_runner = types.SimpleNamespace(enabled=False)
            with patch.object(SteadyMtpPrepare, "_warm_stage_kernel") as warm_stage_kernel:
                preparer.attach_cuda_graph_runners(disabled_graph_runner, None)
                warm_stage_kernel.assert_not_called()
            assert preparer._steady_mtp is None
            enabled_graph_runner = types.SimpleNamespace(
                enabled=True,
                shared_static_tensors={},
                fusible_engram_hook=None,
                take_agreed_replay_key=lambda _: None,
            )
            preparer.attach_cuda_graph_runners(enabled_graph_runner, None)
            assert preparer._steady_mtp is not None
            metadata = DecoderMetadata(
                config=config,
                mapping=Mapping(),
                model=model,
                buffers=preparer._buffers,
                attention_runtime_features=AttentionRuntimeFeatures(),
            )
            metadata.keep_steady_growth_blocks = preparer.steady_decode_prepare_enabled
            attn_metadata = metadata.setup_attention(manager).create_cuda_graph_metadata(
                len(rows), False, draft_len
            )
            assert type(attn_metadata) is TrtllmAttentionMetadata
            spec_metadata = Mock(
                _force_non_greedy_for_capture=False,
                context_prompt_lookahead_tokens=None,
                supports_steady_prepare=not attention_dp,
                runtime_draft_len=draft_len,
            )
            spec_metadata.dp_num_tokens = lambda spec=spec_metadata: spec.num_tokens
            arms[arm] = (preparer, attn_metadata, spec_metadata)
        # Both arms start from the same metadata buffer contents.
        steady_fields, full_fields = (vars(arms[arm][1]) for arm in ("steady", "full"))
        for name, value in steady_fields.items():
            twin = full_fields.get(name)
            if isinstance(value, torch.Tensor) and isinstance(twin, torch.Tensor):
                if value.data_ptr() != twin.data_ptr():
                    value.zero_()
                    twin.zero_()

        # Accepted tokens per request and step. None: request 1 re-pages its last block (which holds
        # draft slots only) and no request advances, so that is the batch's only block-table change.
        # A tuple: new requests (ids, tokens, slots) replace the batch and take their first step.
        schedule = [[1, 1, 1], [4, 4, 4], None, [2, 3, 1], [4, 4, 4], [3, 1, 2], [1, 2, 4]]
        schedule[4:4] = [([11, 12, 13], [20, 9, 41], [1, 6, 3]), [3, 2, 4]]
        # The second batch takes the first batch's slots, which the steady arm last uploaded.
        schedule += [([21, 22, 23], [5, 30, 12], [5, 2, 7]), [2, 4, 1]]
        forced = {len(schedule)}
        schedule.append([3, 1, 2])
        joins = {step for step, accepted in enumerate(schedule, 1) if isinstance(accepted, tuple)}
        # Without steady_decode_prepare a first step and the step after it take full passes.
        steady_steps = [
            step
            for step in range(1, len(schedule) + 1)
            if step not in forced and (opted_in or not joins & {step, step - 1})
        ]
        gen = torch.Generator(device="cuda").manual_seed(0)
        drafts = np.random.default_rng(0)
        keys = [manager.batch_block_table_key(request_ids)]
        copied, admitted, by_row = [], [], []
        for step, accepted in enumerate([[0, 0, 0]] + schedule):
            joining = isinstance(accepted, tuple)
            use_stores = sampler is not None and step == 2
            previous_slots = [request.py_seq_slot for request in requests]
            if accepted is None:
                _repage_last_block(manager, requests[0], other)
            elif joining:
                for request in requests:
                    manager.free_resources(request)
                ids, token_nums, slots = accepted
                requests = manager.add_dummy_requests(
                    ids, token_nums=token_nums, is_gen=True, max_num_draft_tokens=draft_len
                )
                for request, slot in zip(requests, slots):
                    request.is_dummy_request = False
                    request.py_seq_slot = slot
                    request.py_draft_tokens = drafts.integers(0, 1000, draft_len).tolist()
                    assert manager.kv_cache_map[request.py_request_id].resize(
                        request.max_beam_num_tokens + draft_len
                    )
                request_ids = [request.py_request_id for request in requests]
                batch.generation_requests = requests + rows[len(requests) :]
            else:
                for request, count in zip(requests, accepted):
                    for _ in range(count):
                        request.add_new_token(1, 0)
                    assert manager.kv_cache_map[request.py_request_id].resize(
                        request.max_beam_num_tokens + draft_len
                    )
            if step > 0:
                keys.append(manager.batch_block_table_key(request_ids))
            stored = _previous_step_tensors(gen, t)
            idle = joining and accepted[0][0] == 11  # the device was idle
            states = {}
            for arm, (preparer, attn_metadata, spec_metadata) in arms.items():
                previous = None if idle else _with_output_rows(stored, previous_slots)
                if arm == "full" or step in forced:
                    preparer._steady_mtp.invalidate()
                if joining:
                    for request in requests:
                        request.py_batch_idx = None
                if use_stores:
                    previous = _sample_stores(sampler, stored, requests)
                wait_patch = (
                    patch.object(
                        torch.cuda.Stream,
                        "wait_event",
                        autospec=True,
                        side_effect=torch.cuda.Stream.wait_event,
                    )
                    if use_stores
                    else nullcontext()
                )
                with (
                    patch.object(
                        manager, "copy_batch_block_offsets", wraps=manager.copy_batch_block_offsets
                    ) as copy,
                    patch.object(
                        preparer,
                        "_apply_steady_mtp_fast_prepare",
                        wraps=preparer._apply_steady_mtp_fast_prepare,
                    ) as steady,
                    patch.object(
                        steady_spec_prepare,
                        "stage_steady_inputs",
                        wraps=steady_spec_prepare.stage_steady_inputs,
                    ) as stage,
                    wait_patch as wait_event,
                ):
                    prepared = preparer.prepare_inputs(
                        scheduled_requests=batch,
                        kv_cache_manager=manager,
                        attn_metadata=attn_metadata,
                        spec_metadata=spec_metadata,
                        new_tensors_device=previous,
                        resource_manager=resource_manager,
                        enable_spec_decode=True,
                        runtime_draft_len=draft_len,
                        is_dummy=False,
                    )
                    inputs = preparer.preprocess_inputs(
                        prepared.kwargs, enable_spec_decode=True, runtime_draft_len=draft_len
                    )
                torch.cuda.synchronize()
                assert steady.called == (arm == "steady" and step in steady_steps), (arm, step)
                if steady.called:
                    copied.append(copy.called)
                    admitted.append(steady.call_args.kwargs["admit"])
                    by_row.append(stage.call_args.kwargs["rows"] is not None)
                states[arm] = _step_state(
                    preparer, inputs, attn_metadata, spec_metadata, len(rows), t
                )
                if use_stores:
                    wait_event.assert_any_call(torch.cuda.current_stream(), previous.stores_ready)
                    expected_ids = stored.new_tokens[:t, previous_slots, 0].T.reshape(-1)
                    expected_drafts = stored.next_draft_tokens[previous_slots, :draft_len].flatten()
                    expected_offsets = stored.new_tokens_lens[previous_slots]
                    assert torch.equal(states[arm]["input_ids"], expected_ids), arm
                    assert torch.equal(states[arm]["draft_tokens"], expected_drafts), arm
                    assert torch.equal(
                        states[arm]["previous_pos_id_offsets"],
                        expected_offsets.repeat_interleave(t),
                    ), arm
                    assert torch.equal(
                        states[arm]["previous_kv_lens_offsets"], expected_offsets - t
                    ), arm
            mismatches = sorted(
                name
                for name in states["steady"].keys() | states["full"].keys()
                if not _same(states["steady"].get(name), states["full"].get(name))
            )
            assert not mismatches, (step, mismatches)
        assert arms["steady"][0]._steady_mtp.record.attn_steady
        if attention_dp:
            # Each step of each arm exchanged its counts once, steady or not.
            assert len(exchanges) == 2 * (len(schedule) + 1), exchanges
        # A steady step copies the block table exactly when it changed.
        changed = [keys[step] != keys[step - 1] for step in steady_steps]
        assert copied == changed and any(changed) and not all(changed), (copied, changed)
        assert admitted == [step in joins for step in steady_steps], admitted
        # A step that verifies its previous drafts reads them from the previous step's output rows.
        assert by_row == [
            opted_in and step not in joins and not (sampler is not None and step == 2)
            for step in steady_steps
        ], by_row
    finally:
        manager.shutdown()
