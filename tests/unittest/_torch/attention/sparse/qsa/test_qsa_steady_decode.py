# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""QSA metadata of a steady one-model MTP decode step (``RuntimeHooks.qsa``): the one-launch block-offset copy
writes what the native copy and the QSA unscale write, and the steady step unscales only the tables it copied
and leaves the row mapping to the forward only under ``steady_replay_gap`` on a CUDA-graph decode step."""

import sys
from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.attention.backends import trtllm
from tensorrt_llm._torch.attention.backends.sparse.qsa import metadata as qsa_metadata
from tensorrt_llm._torch.attention.backends.sparse.qsa.kernels import (
    triton_qsa_copy_block_offsets,
    triton_qsa_unscale_block_table,
)
from tensorrt_llm._torch.attention.backends.sparse.qsa.metadata import QSAAttentionMetadata
from tensorrt_llm._torch.attention.backends.trtllm import TrtllmAttentionMetadata
from tensorrt_llm._torch.pyexecutor.runtime_hooks import QsaRuntimePolicy

_CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


def _pinned(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.pin_memory()


@_CUDA
@pytest.mark.parametrize(
    "pools,width,num_seqs", [(1, 8, 1), (3, 64, 5), (2, 1028, 7), (4, 2048, 3)]
)
@pytest.mark.parametrize("qsa_pool,role", [(0, 0), (1, 0), (1, 1)])
def test_copy_block_offsets_matches_native_copy_and_unscale(pools, width, num_seqs, qsa_pool, role):
    from tensorrt_llm.bindings.internal.batch_manager.kv_cache_manager_v2_utils import (
        copy_batch_block_offsets_to_device,
    )

    if qsa_pool >= pools:
        pytest.skip("no such pool")
    gen = torch.Generator().manual_seed(pools * 1000 + width + num_seqs)
    src_rows = num_seqs + 9
    host = torch.randint(0, 1 << 16, (pools, src_rows, 2, width), generator=gen, dtype=torch.int32)
    host[torch.rand(host.shape, generator=gen) < 0.2] = -1  # BAD_PAGE_INDEX
    host = _pinned(host)
    copy_index = _pinned(torch.randperm(src_rows, generator=gen)[:num_seqs].to(torch.int32))
    scales = _pinned(torch.randint(1, 9, (pools,), generator=gen, dtype=torch.int32))
    kv_offset = _pinned(torch.randint(-50, 50, (pools,), generator=gen, dtype=torch.int32))
    dst_rows = num_seqs + 2  # two canary rows past the batch
    expected = torch.full((pools, dst_rows, 2, width), -7, dtype=torch.int32, device="cuda")
    actual = expected.clone()
    expected_table = torch.full((dst_rows, width), -9, dtype=torch.int32, device="cuda")
    actual_table = expected_table.clone()
    scale = int(scales[qsa_pool])

    copy_batch_block_offsets_to_device(
        host, expected, copy_index, scales, kv_offset, torch.cuda.current_stream().cuda_stream
    )
    triton_qsa_unscale_block_table(
        scaled_block_table=expected[qsa_pool, :num_seqs, role, :],
        block_table=expected_table[:num_seqs],
        page_index_scale=scale,
    )
    triton_qsa_copy_block_offsets(
        host_block_offsets=host,
        block_offsets=actual,
        copy_index=copy_index,
        index_scales=scales,
        kv_offset=kv_offset,
        num_seqs=num_seqs,
        qsa_table=actual_table,
        qsa_pool=qsa_pool,
        qsa_page_index_scale=scale,
        qsa_role=role,
    )
    torch.cuda.synchronize()
    assert torch.equal(actual, expected)
    assert torch.equal(actual_table, expected_table)

    # Without the slot table it is the native copy alone.
    plain = torch.full_like(expected, -7)
    triton_qsa_copy_block_offsets(
        host_block_offsets=host,
        block_offsets=plain,
        copy_index=copy_index,
        index_scales=scales,
        kv_offset=kv_offset,
        num_seqs=num_seqs,
    )
    torch.cuda.synchronize()
    assert torch.equal(plain, expected)


@_CUDA
def test_copy_block_offsets_rejects_mismatched_tables():
    host = torch.zeros((2, 4, 2, 8), dtype=torch.int32).pin_memory()
    idx = torch.zeros(2, dtype=torch.int32).pin_memory()
    scales = torch.ones(2, dtype=torch.int32).pin_memory()
    with pytest.raises(ValueError):
        triton_qsa_copy_block_offsets(
            host_block_offsets=host,
            block_offsets=torch.zeros((2, 4, 2, 16), dtype=torch.int32, device="cuda"),
            copy_index=idx,
            index_scales=scales,
            kv_offset=scales,
            num_seqs=2,
        )
    with pytest.raises(ValueError):
        triton_qsa_copy_block_offsets(
            host_block_offsets=host,
            block_offsets=torch.zeros((2, 4, 2, 8), dtype=torch.int32, device="cuda"),
            copy_index=idx,
            index_scales=scales,
            kv_offset=scales,
            num_seqs=2,
            qsa_table=torch.zeros((2, 8), dtype=torch.int32, device="cuda"),
            qsa_pool=2,
        )


def _metadata(policy: QsaRuntimePolicy, *, draft: bool, cuda_graph: bool = True, contexts: int = 0):
    md = object.__new__(QSAAttentionMetadata)
    md.qsa_policy = policy
    md.qsa_has_local_layers = True
    md.is_cuda_graph = cuda_graph
    md._num_contexts = contexts
    md.request_ids = [3, 4]
    md._steady_block_table_keys = ("t0", "d0")
    md.kv_cache_manager = SimpleNamespace(name="target")
    md.draft_kv_cache_manager = SimpleNamespace(name="draft") if draft else None
    md.qsa_draft_block_table = object() if draft else None
    return md


def _record_calls(monkeypatch, md, new_keys, fused=(False, False)):
    calls = []

    def parent_step(self, kv_lens, prompt_lens=None):
        calls.append(
            ("parent", kv_lens) if prompt_lens is None else ("parent", kv_lens, prompt_lens)
        )
        self._steady_block_table_keys = new_keys

    monkeypatch.setattr(TrtllmAttentionMetadata, "prepare_steady_decode_step", parent_step)
    for name in (
        "_refresh_qsa_speculative_snapshot_state",
        "_refresh_qsa_block_table",
        "_refresh_qsa_draft_block_table",
        "_refresh_qsa_token_mapping",
    ):
        monkeypatch.setattr(md, name, lambda name=name: calls.append(name), raising=False)
    monkeypatch.setattr(
        md, "_copy_changed_block_offsets", lambda last: calls.append("fused") or fused
    )
    return calls


@pytest.mark.parametrize(
    "new_keys, draft, unscaled",
    [
        (("t0", "d0"), True, []),  # nothing changed: no unscale
        (("t1", "d0"), True, ["_refresh_qsa_block_table"]),
        (("t0", "d1"), True, ["_refresh_qsa_draft_block_table"]),
        (
            (None, None),
            True,
            ["_refresh_qsa_block_table", "_refresh_qsa_draft_block_table"],
        ),  # no key: copied every step
        (("t1", None), False, ["_refresh_qsa_block_table"]),  # no draft manager
    ],
)
def test_steady_step_unscales_only_copied_tables(monkeypatch, new_keys, draft, unscaled):
    md = _metadata(QsaRuntimePolicy(steady_decode=True), draft=draft)
    calls = _record_calls(monkeypatch, md, new_keys)
    md.prepare_steady_decode_step([10, 11])
    assert calls == [
        ("parent", [10, 11]),
        "_refresh_qsa_speculative_snapshot_state",
        *unscaled,
        "_refresh_qsa_token_mapping",
    ]


@pytest.mark.parametrize(
    "cuda_graph, contexts, replay_gap", [(True, 0, True), (False, 0, False), (True, 1, False)]
)
@pytest.mark.parametrize("fused", [(False, False), (True, False), (True, True)])
def test_replay_gap_fuses_the_unscale_and_leaves_the_mapping(
    monkeypatch, cuda_graph, contexts, replay_gap, fused
):
    md = _metadata(
        QsaRuntimePolicy(steady_decode=True, steady_replay_gap=True),
        draft=True,
        cuda_graph=cuda_graph,
        contexts=contexts,
    )
    calls = _record_calls(monkeypatch, md, ("t1", "d1"), fused=fused)
    md.prepare_steady_decode_step([7, 8])
    taken = fused if replay_gap else (False, False)
    assert calls == [
        *(["fused"] if replay_gap else []),
        ("parent", [7, 8]),
        "_refresh_qsa_speculative_snapshot_state",
        *([] if taken[0] else ["_refresh_qsa_block_table"]),
        *([] if taken[1] else ["_refresh_qsa_draft_block_table"]),
        *([] if replay_gap else ["_refresh_qsa_token_mapping"]),
    ]


@pytest.mark.parametrize("policy", [QsaRuntimePolicy(), QsaRuntimePolicy(steady_decode=True)])
def test_the_prompt_lengths_of_joining_requests_reach_the_parent(monkeypatch, policy):
    md = _metadata(policy, draft=False)
    calls = _record_calls(monkeypatch, md, ("t0", None))
    md.prepare_steady_decode_step([5, 6], [3, 4])
    assert calls[0] == ("parent", [5, 6], [3, 4])


def test_without_the_policy_the_step_is_the_parents(monkeypatch):
    md = _metadata(QsaRuntimePolicy(), draft=True)
    calls = _record_calls(monkeypatch, md, ("t1", "d1"))
    md.prepare_steady_decode_step([1, 2])
    assert calls == [("parent", [1, 2])]


def _supports(monkeypatch, policy, fmha_plans=(), **over):
    md = _metadata(policy, draft=False)
    md.enable_flash_mla = False
    md.enable_helix = False
    md.enable_context_mla_with_cached_kv = False
    md.fp4_mla_state = None
    md.mixed_precision_high_block_table = None
    md.kv_cache_params = SimpleNamespace(use_full_generation_page_table=False)
    monkeypatch.setattr(md, "_has_mixed_precision_staging", lambda: False, raising=False)
    # No enabled FMHA library plans steps in prepare() (the parent's condition).
    monkeypatch.setattr(trtllm, "step_planning_fmha_lib_classes", lambda: fmha_plans)
    for name, value in over.items():
        setattr(md, name, value)
    return md.supports_steady_decode_step


def test_supports_steady_decode_step_follows_the_policy(monkeypatch):
    assert not _supports(monkeypatch, QsaRuntimePolicy())
    assert _supports(monkeypatch, QsaRuntimePolicy(steady_decode=True))
    assert not _supports(
        monkeypatch, QsaRuntimePolicy(steady_decode=True), qsa_has_local_layers=False
    )
    assert not _supports(monkeypatch, QsaRuntimePolicy(steady_decode=True), enable_helix=True)
    assert not _supports(monkeypatch, QsaRuntimePolicy(steady_decode=True), fmha_plans=(object,))
    assert not _supports(
        monkeypatch,
        QsaRuntimePolicy(steady_decode=True),
        kv_cache_params=SimpleNamespace(use_full_generation_page_table=True),
    )


@pytest.mark.parametrize(
    "change, plain",
    [
        (None, True),
        ("per_layer", False),
        ("expansion", False),
        ("width", False),
        ("override", False),
    ],
)
def test_plain_copy_detection(monkeypatch, change, plain):
    from tensorrt_llm._torch.pyexecutor.kv_cache import kv_cache_manager_v2 as kvm

    class Manager(kvm.KVCacheManagerV2):
        def __init__(self):  # no pools: only the fields the check reads
            pass

    manager = Manager()
    width = 6 if change == "width" else 8
    manager._use_per_layer_page_tables = change == "per_layer"
    manager._page_table_materializer = SimpleNamespace(uses_device_expansion=change == "expansion")
    manager.kv_cache_type = None
    manager.host_kv_cache_block_offsets = torch.zeros((2, 4, 2, width), dtype=torch.int32)
    manager._stream = SimpleNamespace(cuda_stream=0)
    if change == "override":
        Manager.copy_batch_block_offsets = lambda self, *a, **k: None
    monkeypatch.setattr(
        torch.cuda, "current_stream", lambda *a, **k: SimpleNamespace(cuda_stream=0)
    )
    block_offsets = torch.zeros((2, 4, 2, width), dtype=torch.int32)
    assert sys.modules[QSAAttentionMetadata.__module__] is qsa_metadata
    assert qsa_metadata._plain_block_offsets_copy(manager, block_offsets) is plain
