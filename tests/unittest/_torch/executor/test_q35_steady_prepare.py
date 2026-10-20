# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TRTLLM_Q35_STEADY_PREPARE: the one-launch staging kernel writes exactly what the full
_prepare_tp_inputs pass writes with its H2D copies and torch ops (overlap one-model MTP steady
decode), bitwise, for b = 1..4 and k = 6..8, with the per-batch or the engine-wide block; a corrupted
slot is caught (negative control); one kernel variant serves every batch size; the block-offset skip
key (TRTLLM_Q35_STEADY_PREPARE_SKIP_BLOCKS) changes when the last block is re-allocated at the same
block count; a pinned host slab read through its device mapping (TRTLLM_Q35_STEADY_PREPARE_ZEROCOPY) gives the
same writes."""

import types

import pytest
import torch

from tensorrt_llm._torch.pyexecutor import q35_steady_prepare as steady
from tensorrt_llm._torch.pyexecutor.q35_steady_prepare import stage_steady_inputs

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
    """The full pass's ops for an all-extend, all-previous-tensor batch (model_engine.py)."""
    d = t - 1
    n = b * t
    pin = lambda x: torch.tensor(x, dtype=torch.int32).pin_memory()  # noqa: E731
    buf["num_accepted"][:b].copy_(pin(accepted), non_blocking=True)
    buf["prev_pos_offsets"] *= 0
    buf["prev_kv_offsets"] *= 0
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
    if kv_lens is not None:  # TrtllmAttentionMetadata.prepare(): kv_lens_cuda[:n] <- pinned kv lens
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
        new_tokens=rnd(MAX_DRAFT + 1, SLOTS, 1).cuda(),  # store.next_new_tokens
        new_tokens_lens=torch.randint(
            1, MAX_DRAFT + 2, (SLOTS,), generator=gen, dtype=torch.int32
        ).cuda(),
        # a view into a larger buffer, like the fused sampler store
        next_draft_tokens=flat[MAX_DRAFT + 1 + SLOTS :].view(SLOTS, MAX_DRAFT),
    )


@pytest.mark.parametrize("block", [None, 64])
@pytest.mark.parametrize("t", [7, 8, 9])
@pytest.mark.parametrize("b", [1, 2, 3, 4])
@pytest.mark.parametrize("mrope", [True, False])
@pytest.mark.parametrize("with_kv", [True, False])
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
        kwargs = {k: v for k, v in actual.items() if k not in ("mrope_positions", "kv_lens_cuda")}
        stage_steady_inputs(
            slab,
            b,
            t,
            dev,
            mrope_positions=actual["mrope_positions"] if mrope else None,
            kv_lens_cuda=actual["kv_lens_cuda"] if with_kv else None,
            block=block,
            **kwargs,
        )
        torch.cuda.synchronize()
        for name in expected:
            if name == "mrope_positions" and not mrope:
                continue
            assert torch.equal(actual[name], expected[name]), (b, t, name)


@pytest.mark.parametrize("mrope", [True, False])
def test_static_graph_inputs_equal_engine_buffers(mrope):
    """HAS_STATIC: the graph's static ids / positions get exactly what replay() would copy."""
    gen = torch.Generator().manual_seed(11)
    b, t = 4, 7
    dev = _device_tensors(gen)
    slots, accepted, positions = _step(gen, b, t)
    actual = _buffers()
    static_ids = torch.full((32,), -1, dtype=torch.int32, device="cuda")
    static_pos = torch.full((3, 1, 32) if mrope else (1, 32), -1, dtype=torch.int32, device="cuda")
    slab = torch.tensor(slots + accepted + positions, dtype=torch.int32, device="cuda")
    kwargs = {k: v for k, v in actual.items() if k not in ("mrope_positions", "kv_lens_cuda")}
    stage_steady_inputs(
        slab,
        b,
        t,
        dev,
        mrope_positions=actual["mrope_positions"] if mrope else None,
        static_input_ids=static_ids,
        static_position_ids=static_pos,
        **kwargs,
    )
    torch.cuda.synchronize()
    n = b * t
    # what CUDAGraphRunner.replay copies
    assert torch.equal(static_ids[:n], actual["input_ids"][:n])
    if mrope:
        assert torch.equal(static_pos[:, :, :n], actual["mrope_positions"][:, :, :n])
    else:
        assert torch.equal(static_pos[:, :n], actual["positions"][:n].view(1, -1))
    assert int(static_ids[n]) == -1  # nothing past the batch


def test_corrupted_slot_is_caught():
    gen = torch.Generator().manual_seed(7)
    b, t = 2, 7
    dev = _device_tensors(gen)
    slots, accepted, positions = _step(gen, b, t)
    expected, actual = _buffers(), _buffers()
    _reference(expected, b, t, slots, accepted, positions, dev)
    wrong = [(slots[0] + 1) % SLOTS] + slots[1:]
    slab = torch.tensor(wrong + accepted + positions, dtype=torch.int32, device="cuda")
    stage_steady_inputs(
        slab,
        b,
        t,
        dev,
        mrope_positions=actual["mrope_positions"],
        **{k: v for k, v in actual.items() if k not in ("mrope_positions", "kv_lens_cuda")},
    )
    torch.cuda.synchronize()
    assert not torch.equal(actual["input_ids"], expected["input_ids"])


def _variants():
    caches = getattr(steady._steady_stage_kernel, "device_caches", None)
    if caches is None:
        pytest.skip("Triton without JITFunction.device_caches")
    return sum(len(entry[0]) for entry in caches.values())


def test_one_kernel_variant_for_all_batch_sizes():
    """With a fixed block the staging kernel compiles once for b = 1..4 (ints and alignments unspecialized);
    negative control: the per-batch block compiles a variant per batch size."""
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
            **{k: v for k, v in actual.items() if k not in ("mrope_positions", "kv_lens_cuda")},
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
    """The rewind frees the last block and the next step re-allocates it on another page at the same block
    count: the key changes (negative control: the block counts alone do not)."""
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


def test_block_key_unavailable():
    m = _manager()
    _add(m, 5, 1, [1, 2])
    assert steady._block_counts(m, [5, 7]) is None  # unknown request
    m.enable_block_reuse = True
    assert steady._block_counts(m, [5]) is None
    m.enable_block_reuse = False
    m.enable_swa_scratch_reuse = True
    assert steady._block_counts(m, [5]) is None
    assert steady._block_counts(types.SimpleNamespace(kv_cache_map={}), [5]) is None


@pytest.mark.parametrize("b", [1, 4])
def test_zero_copy_slab_matches_device_slab(b):
    """TRTLLM_Q35_STEADY_PREPARE_ZEROCOPY: the kernel reading a pinned host slab through its device mapping writes
    exactly what it writes from a device copy of the slab."""
    gen = torch.Generator().manual_seed(21 + b)
    t = 7
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
            **{k: v for k, v in actual.items() if k not in ("mrope_positions", "kv_lens_cuda")},
        )
        torch.cuda.synchronize()
        results.append(actual)
    for name in results[0]:
        assert torch.equal(results[0][name], results[1][name]), name
    expected = _buffers()
    _reference(expected, b, t, slots, accepted, positions, dev, kv_lens)
    for name in expected:
        assert torch.equal(results[1][name], expected[name]), name


class _SsmCache:
    def __init__(self, base):
        self.base = base

    def get_ssm_block_base_index(self, group):
        assert group == 3
        return self.base


def _mamba_setup(device="cuda"):
    i32 = dict(dtype=torch.int32, device=device)
    manager = types.SimpleNamespace(
        kv_cache_map={5: _SsmCache(11), 6: _SsmCache(4)},
        ssm_layer_group_id=3,
        local_num_mamba_layers=45,
        cuda_state_indices=torch.tensor([11, 4, 0, 0], **i32),
        _dummy_request_mask=torch.zeros(4, dtype=torch.bool, device=device),
        _request_id_to_is_dummy={5: False, 6: False},
    )
    mamba = types.SimpleNamespace(
        state_indices=torch.tensor([11, 4, 0, 0], **i32),
        state_indices_long=torch.tensor([11, 4], dtype=torch.long, device=device),
    )
    return types.SimpleNamespace(kv_cache_manager=manager, mamba_metadata=mamba)


def test_verify_checks_mamba_indices_against_the_manager():
    """VERIFY's Mamba ground truth: the manager's and the metadata's slot indices must be the KV caches' SSM base
    blocks and the dummy mask the batch's flags; a stale index or mask is reported (negative controls)."""
    check = steady.SteadyMtpPrepare._mamba_ground_truth
    attn = _mamba_setup()
    assert check(attn, [5, 6]) == []
    attn.kv_cache_manager.kv_cache_map[
        6
    ].base = 9  # the request moved to another SSM block, uploads skipped
    bad = check(attn, [5, 6])
    assert any("cuda_state_indices" in m for m in bad) and any(
        "mamba.state_indices" in m for m in bad
    )
    attn = _mamba_setup()
    attn.kv_cache_manager._dummy_request_mask[3] = True  # a stale dummy flag past the batch
    assert any("_dummy_request_mask" in m for m in check(attn, [5, 6]))
    attn = _mamba_setup()
    attn.kv_cache_manager.local_num_mamba_layers = 0  # no local Mamba layers: nothing to check
    attn.kv_cache_manager.kv_cache_map[6].base = 9
    assert check(attn, [5, 6]) == []
