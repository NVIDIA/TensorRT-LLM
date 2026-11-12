# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``common.kda_metadata``: the KDA step metadata, with the mid-chunk snapshot plan.

The mid-chunk interface with the cache manager (the runtime side is ``KimiK3HybridCacheManager``):

* ``get_kda_midchunk_snapshots(request_ids)`` is called once per step that has context
  requests, after ``Mamba2Metadata.prepare``, with the context requests' ids in batch order, and
  returns ``(context index, pool slot, offset)`` tuples: at most one per context request, the
  offset a multiple of 64 strictly inside that request's chunk;
* the plan becomes ``kda_snapshot`` with ``stage_rows`` None (the snapshot goes to a slot of its
  own); a manager without the method plans none; a step with an in-forward plan too fails.

The SM107 glue (each target's ``modeling.py``: tp1, dep16 and the two helix8 targets) uses these very
classes. On the GPU, ``prepare``'s FlashInfer prefill schedule is compared with main's
``KimiK3MambaMetadata`` and the CuTe DSL prefill's fields with their values.
"""

import importlib
from types import SimpleNamespace

import pytest
import torch

_FAMILY = "tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl"
# The KDA glue of each SM107 target: its modeling.py.
_SM107_KDA = tuple(
    f"{_FAMILY}.kimi_k3_nvfp4__sm_107__{target}.modeling"
    for target in ("tp1", "dep16", "helix8_moeep8", "helix8_moetp2ep4")
)


def _fam():
    return importlib.import_module(f"{_FAMILY}.common.kda_metadata")


class _Manager:
    """A cache manager stub that plans the given snapshots for the given request ids."""

    def __init__(self, plans):
        self.plans = plans
        self.calls = []

    def get_kda_midchunk_snapshots(self, request_ids):
        self.calls.append(list(request_ids))
        return [(i, *self.plans[rid]) for i, rid in enumerate(request_ids) if rid in self.plans]


def _bare_metadata(cls, device="cpu"):
    metadata = cls.__new__(cls)
    metadata._state_indices_long = torch.zeros(8, dtype=torch.long, device=device)
    metadata.kda_snapshot = None
    return metadata


@pytest.mark.cpu_only
def test_constants():
    fam = _fam()
    assert fam.KDA_PREFILL_CHUNK_SIZE == 64 == fam.KDA_MIDCHUNK_SNAPSHOT_ALIGNMENT
    assert fam.K3MambaMetadata.prefill_chunk_alignment == 64


@pytest.mark.cpu_only
@pytest.mark.parametrize("glue", _SM107_KDA)
def test_sm107_glue_uses_the_family_metadata(glue):
    """Each glue's KDA takes the family's plan class, and whatever else of the metadata it names is
    the family's (tp1 imports only the plan class)."""
    fam, sm107 = _fam(), importlib.import_module(glue)
    assert sm107.KDASnapshotPlan is fam.KDASnapshotPlan
    for name in ("K3MambaMetadata", "build_kda_snapshot_plan"):
        if hasattr(sm107, name):
            assert getattr(sm107, name) is getattr(fam, name), name
    if hasattr(sm107, "KDA_PREFILL_CHUNK_SIZE"):
        assert sm107.KDA_PREFILL_CHUNK_SIZE == fam.KDA_PREFILL_CHUNK_SIZE


@pytest.mark.cpu_only
def test_build_plan():
    fam = _fam()
    snapshots = [(2, 9, 64), (0, 5, 128)]
    lengths = [200, 70, 130]
    a = fam.build_kda_snapshot_plan(snapshots, lengths, torch.device("cpu"))
    assert a.num_snapshots == 2 and a.stage_rows is None
    assert a.slots.tolist() == [5, 9]
    assert a.token_indices.tolist() == [128, 270 + 64]
    assert a.chunks.tolist() == [2, -1, 1]
    assert a.pool_rows.tolist() == [5, -1, 9]
    assert a.staged_rows.tolist() == [3, -1, 4]
    assert fam.build_kda_snapshot_plan([], lengths, torch.device("cpu")) is None
    for bad in ([(0, 1, 100)], [(1, 1, 64)], [(0, 1, 64), (0, 2, 128)], [(3, 1, 64)]):
        with pytest.raises(ValueError):
            fam.build_kda_snapshot_plan(bad, [200, 64], torch.device("cpu"))


@pytest.mark.cpu_only
def test_midchunk_fetch():
    fam = _fam()
    metadata = _bare_metadata(fam.K3MambaMetadata)
    manager = _Manager({12: (7, 64), 14: (3, 128)})
    attn = SimpleNamespace(kv_cache_manager=manager, request_ids=[11, 12, 14, 99])
    metadata._fetch_midchunk_snapshots(attn, [100, 130, 200])
    assert manager.calls == [[11, 12, 14]]  # the context requests only
    plan = metadata.kda_snapshot
    assert plan.num_snapshots == 2 and plan.stage_rows is None
    assert plan.slots.tolist() == [7, 3]
    assert plan.token_indices.tolist() == [100 + 64, 230 + 128]
    assert plan.chunks.tolist() == [-1, 1, 2]
    assert plan.pool_rows.tolist() == [-1, 7, 3]

    # No snapshot this step, no method on the manager, or no request ids: no plan.
    for attn in (
        SimpleNamespace(kv_cache_manager=_Manager({}), request_ids=[1]),
        SimpleNamespace(kv_cache_manager=object(), request_ids=[1]),
        SimpleNamespace(kv_cache_manager=manager, request_ids=None),
    ):
        metadata.kda_snapshot = None
        metadata._fetch_midchunk_snapshots(attn, [100])
        assert metadata.kda_snapshot is None

    # An in-forward plan of the same step.
    metadata.kda_snapshot = plan
    with pytest.raises(RuntimeError, match="both mid-chunk and in-forward"):
        metadata._fetch_midchunk_snapshots(
            SimpleNamespace(kv_cache_manager=manager, request_ids=[12]), [130]
        )


@pytest.mark.cpu_only
def test_inforward_split():
    from tensorrt_llm._torch.pyexecutor.kv_cache.mamba_cache_manager import InforwardSnapshotEntry

    fam = _fam()
    manager = SimpleNamespace(conv_state_shape=(12, 3), state_index={11: 7, 13: 2}.__getitem__)
    attn = SimpleNamespace(seq_lens=torch.tensor([200, 130, 96]), request_ids=[11, 12, 13])
    plan = {
        11: InforwardSnapshotEntry(row=1, point=1152, num_before=128, num_after=72),
        13: InforwardSnapshotEntry(row=0, point=64, num_before=64, num_after=32),
    }
    metadata = _bare_metadata(fam.K3MambaMetadata)
    metadata._prepare_inforward_split(attn, manager, plan, 3, 3)
    got = metadata.kda_snapshot
    # (sequence, the request's live slot, the point inside the chunk) and the staging rows.
    want = fam.build_kda_snapshot_plan(
        [(0, 7, 128), (2, 2, 64)], [200, 130, 96], torch.device("cpu")
    )
    for field in ("slots", "token_indices", "chunks", "pool_rows", "staged_rows"):
        assert torch.equal(getattr(got, field), getattr(want, field)), field
    assert got.stage_rows.tolist() == [1, 0]
    with pytest.raises(RuntimeError, match="generation requests"):
        metadata._prepare_inforward_split(attn, manager, plan, 3, 4)


def _fake_mamba_prepare(self, attn_metadata):
    """``Mamba2Metadata.prepare``'s part the KDA metadata reads: slots aliasing a manager buffer."""
    n = attn_metadata.seq_lens.shape[0]
    self.state_indices = attn_metadata.slots[:n]
    self.state_indices_cpu = attn_metadata.slots.cpu()
    self._state_indices_aliased_ptr = self.state_indices.data_ptr()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the metadata buffers live on the GPU")
@pytest.mark.parametrize("plans", [{}, {12: (7, 64)}, {11: (5, 128), 13: (6, 64)}])
def test_prepare(plans, monkeypatch):
    from tensorrt_llm._torch.modules.kimi_kda.kimi_k3_mamba_metadata import KimiK3MambaMetadata
    from tensorrt_llm._torch.modules.mamba.mamba2_metadata import Mamba2Metadata

    fam = _fam()
    monkeypatch.setattr(Mamba2Metadata, "prepare", _fake_mamba_prepare)
    seq_lens = torch.tensor([300, 130, 70, 1, 1], dtype=torch.int32)
    slots = torch.tensor([3, 1, 4, 0, 2], dtype=torch.int32, device="cuda")
    results = []
    for cls in (fam.K3MambaMetadata, KimiK3MambaMetadata):
        manager = _Manager(plans)
        attn = SimpleNamespace(
            seq_lens=seq_lens,
            seq_lens_cuda=seq_lens.cuda(),
            num_contexts=3,
            request_ids=[11, 12, 13, 14, 15],
            kv_cache_manager=manager,
            slots=slots,
        )
        metadata = cls(max_batch_size=8, chunk_size=128, max_num_tokens=1024)
        metadata.prepare(attn)
        torch.cuda.synchronize()
        results.append((metadata, manager))
    (fam_md, fam_manager), (main_md, main_manager) = results
    # main's FlashInfer prefill schedule (16-token chunks, longest first)
    for field in ("kda_seq_order", "kda_cu_chunks", "state_indices"):
        assert torch.equal(getattr(fam_md, field), getattr(main_md, field)), field
    assert fam_md.kda_max_seqlen == main_md.kda_max_seqlen == 300
    assert fam_md.kda_cu_chunks.tolist() == [0, 19, 28, 33]
    # the CuTe DSL prefill's: the int64 slots and the 64-token chunk table
    assert fam_md.state_indices_long.tolist() == [3, 1, 4, 0, 2]
    assert fam_md.state_indices_long.dtype == torch.long
    assert fam_md.kda_chunk_indices.tolist() == [
        [0, 0],
        [0, 1],
        [0, 2],
        [0, 3],
        [0, 4],
        [1, 0],
        [1, 1],
        [1, 2],
        [2, 0],
        [2, 1],
    ]
    assert fam_md.kda_varlen_is_aligned is False
    assert fam_md.kda_single_sequence_length is None
    assert fam_manager.calls == [[11, 12, 13]] and main_manager.calls == []
    if not plans:
        assert fam_md.kda_snapshot is None
        return
    want = fam.build_kda_snapshot_plan(
        [(i, *plans[rid]) for i, rid in enumerate([11, 12, 13]) if rid in plans],
        [300, 130, 70],
        torch.device("cuda"),
    )
    for field in ("slots", "token_indices", "chunks", "pool_rows", "staged_rows"):
        assert torch.equal(getattr(fam_md.kda_snapshot, field), getattr(want, field)), field
