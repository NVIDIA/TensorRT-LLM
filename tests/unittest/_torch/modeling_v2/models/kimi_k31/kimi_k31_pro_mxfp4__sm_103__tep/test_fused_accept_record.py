# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The tep target's DFlash2 accept record runs as one launch with the K3.1 KV cache managers.

``K31DFlashWorker._k31_accept_record`` must take ``k31_dflash2_accept_record`` for a generation-only step on every
manager class the K3.1 target selects (the KDA hybrid manager and its DSA subclass, each holding the family's KDA
replay state), and leave exactly what the torch path leaves: ``update_mamba_states`` (the replay state's
``record_acceptance``), then ``_prepare_attn_metadata_for_dflash`` and ``_prepare_kv_for_draft_forward``. Compared
bit for bit: the replay counts updated in place, the KV lengths updated in place, the rewind and the worker's
pending-rewind state; with dummy requests sharing a slot. A replay state without the family's capability keeps the
torch path.
"""

import types

import pytest
import torch

from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE

if not IS_CUTLASS_DSL_AVAILABLE:
    pytest.skip("the accept record is a CuTe DSL kernel", allow_module_level=True)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 3),
    reason="the K3.1 DFlash2 bookkeeping kernels are certified on sm_103",
)

_PKG = "tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31"
SLOTS = 40
DUMMY_SLOT = SLOTS - 1


def _modules():
    import importlib

    modeling = importlib.import_module(f"{_PKG}.kimi_k31_pro_mxfp4__sm_103__tep.modeling")
    kda = importlib.import_module(f"{_PKG}.ops.ssm.kda_cache_manager")
    kda_dsa = importlib.import_module(f"{_PKG}.ops.ssm.kda_dsa_cache_manager")
    return modeling, kda, kda_dsa


class _AttnMetadata:
    """The attention metadata fields the accept record and the torch path touch."""

    def __init__(self, manager, num_seqs, kv_lens, state_indices):
        self.kv_cache_manager = manager
        self.num_seqs = num_seqs
        self.num_contexts = 0
        self.kv_lens_cuda = kv_lens
        self.mamba_metadata = types.SimpleNamespace(state_indices=state_indices)
        self.saved_fields = []
        self.spec_dec_updates = 0

    def prepare_for_spec_dec(self, *fields):
        self.saved_fields.append(fields)

    def update_for_spec_dec(self, *args, **kwargs):
        self.spec_dec_updates += 1


def _manager(cls, replay_cls, prev, dummy, num_gens):
    """A manager of class ``cls`` with only the state ``update_mamba_states`` reads."""
    manager = cls.__new__(cls)
    replay = replay_cls(4)
    replay.prev_num_accepted_tokens = prev
    manager._speculative_state = replay
    manager._kda_replay = replay
    manager.kda_context_only = False
    manager.local_num_mamba_layers = 3
    manager._dummy_request_mask_buffer = types.SimpleNamespace(tensor=dummy)
    manager._generation_state_indices = torch.arange(num_gens, dtype=torch.int32, device="cuda")
    return manager


def _inputs(num_gens, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    num_accepted = torch.randint(1, 6, (num_gens,), generator=g, device="cuda", dtype=torch.int32)
    slots = torch.randperm(SLOTS - 1, generator=g, device="cuda")[:num_gens].to(torch.int32)
    dummy = torch.zeros(SLOTS, dtype=torch.bool, device="cuda")
    if num_gens > 2:
        # The last two requests are CUDA-graph padding: dummies sharing one slot.
        slots[-2:] = DUMMY_SLOT
        dummy[num_gens - 2 : num_gens] = True
    # The V2 manager keeps the replay counts in the KV cache pool: one int32 per slot at the slot pitch.
    prev = torch.randint(0, 5, (SLOTS, 1040), generator=g, device="cuda", dtype=torch.int32)[:, 0]
    kv_lens = torch.randint(100, 120000, (num_gens,), generator=g, device="cuda", dtype=torch.int32)
    return num_accepted, slots, dummy, prev, kv_lens


def _strided_copy(prev):
    """A copy of the replay counts with the same slot stride."""
    pool = torch.empty(prev.shape[0], prev.stride(0), dtype=prev.dtype, device=prev.device)
    out = pool[:, 0]
    out.copy_(prev)
    return out


def _worker(modeling):
    worker = modeling.K31DFlashWorker.__new__(modeling.K31DFlashWorker)
    worker._draft_reads_no_attn_metadata = True
    return worker


def _manager_classes():
    _, kda, kda_dsa = _modules()
    return [
        kda.K31HybridCacheManager,
        kda_dsa.KimiK3DSAHybridCacheManager,
    ]


@pytest.mark.parametrize("manager_index", range(2), ids=["kda", "kda_dsa"])
@pytest.mark.parametrize("num_gens", [1, 2, 3, 7, 16])
def test_family_managers_take_the_fused_record(manager_index, num_gens, monkeypatch) -> None:
    modeling, kda, _ = _modules()
    cls = _manager_classes()[manager_index]
    spec = types.SimpleNamespace(is_cuda_graph=False)
    num_accepted, slots, dummy, prev, kv_lens = _inputs(
        num_gens, seed=17 * num_gens + manager_index
    )

    # Torch path, as K31DFlashWorker._forward_impl runs it.
    manager_t = _manager(cls, kda.K31KDAReplayState, _strided_copy(prev), dummy, num_gens)
    meta_t = _AttnMetadata(manager_t, num_gens, kv_lens.clone(), slots)
    worker_t = _worker(modeling)
    manager_t.update_mamba_states(
        attn_metadata=meta_t,
        num_accepted_tokens=num_accepted,
        state_indices=meta_t.mamba_metadata.state_indices,
    )
    worker_t._prepare_attn_metadata_for_dflash(meta_t, spec)
    worker_t._prepare_kv_for_draft_forward(meta_t, num_accepted, 0, num_gens)

    # One launch.
    manager_f = _manager(cls, kda.K31KDAReplayState, _strided_copy(prev), dummy, num_gens)
    meta_f = _AttnMetadata(manager_f, num_gens, kv_lens.clone(), slots)
    worker_f = _worker(modeling)
    assert worker_f._k31_accept_record(meta_f, spec, num_accepted, 0, num_gens), (
        "the K3.1 manager must take the one-launch accept record"
    )
    torch.cuda.synchronize()

    replay_t, replay_f = manager_t._speculative_state, manager_f._speculative_state
    assert torch.equal(replay_f.prev_num_accepted_tokens, replay_t.prev_num_accepted_tokens)
    assert torch.equal(meta_f.kv_lens_cuda, meta_t.kv_lens_cuda)
    assert worker_f._kv_rewind_amount.dtype == worker_t._kv_rewind_amount.dtype
    assert torch.equal(worker_f._kv_rewind_amount, worker_t._kv_rewind_amount)
    assert (worker_f._kv_rewind_nc, worker_f._kv_rewind_bs, worker_f._kv_rewind_pending) == (
        worker_t._kv_rewind_nc,
        worker_t._kv_rewind_bs,
        worker_t._kv_rewind_pending,
    )
    assert meta_f.saved_fields == meta_t.saved_fields
    assert meta_f.spec_dec_updates == meta_t.spec_dec_updates


def test_family_replay_state_declares_the_fused_record() -> None:
    _, kda, _ = _modules()
    assert kda.K31KDAReplayState.fused_accept_record is True


def test_replay_state_without_the_capability_keeps_the_torch_path(monkeypatch) -> None:
    modeling, kda, _ = _modules()
    num_accepted, slots, dummy, prev, kv_lens = _inputs(5, seed=3)

    class _Changed(kda.K31KDAReplayState):
        fused_accept_record = False

    manager = _manager(kda.K31HybridCacheManager, _Changed, prev, dummy, 5)
    meta = _AttnMetadata(manager, 5, kv_lens, slots)
    assert not _worker(modeling)._k31_accept_record(
        meta, types.SimpleNamespace(is_cuda_graph=False), num_accepted, 0, 5
    )
