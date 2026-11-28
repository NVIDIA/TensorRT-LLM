# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The tep target's DFlash2 worker (``K31DFlashWorker``) on one GPU, built without its constructor.

Side-stream drafter context projection (``_k31_context_begin`` / ``_k31_context_join``), on any GPU: the joined
latents equal the serial projection bit for bit, replayed from a CUDA graph on new inputs.

Accept record (``_k31_accept_record``), on sm_103 with the CuTe DSL: it must take ``k31_dflash2_accept_record`` for a
generation-only step on the manager the K3.1 checkpoint selects (the KDA hybrid manager's DSA subclass, holding the
family's KDA replay state), and leave exactly what the torch path leaves: ``update_mamba_states`` (the replay state's
``record_acceptance``), then ``_prepare_attn_metadata_for_dflash`` and ``_prepare_kv_for_draft_forward``. Compared
bit for bit: the replay counts updated in place, the KV lengths updated in place, the rewind and the worker's
pending-rewind state; with dummy requests sharing a slot.
"""

import importlib
import types

import pytest
import torch
import torch.nn.functional as F

from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.kimi_k31_pro_mxfp4__sm_103__tep import modeling
from tensorrt_llm._torch.modules.multi_stream_utils import with_multi_stream

__extra_import_path__ = [".."]
from _k31_utils import on_sm  # noqa: E402 — needs the path declared above

#: The context projection tests: any GPU.
_GPU = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
#: The accept record tests: the CuTe DSL kernel, certified on sm_103.
_ACCEPT_RECORD_SM = on_sm(
    103, reason="the K3.1 DFlash2 bookkeeping kernels are certified on sm_103"
)
_ACCEPT_RECORD_DSL = pytest.mark.skipif(
    not IS_CUTLASS_DSL_AVAILABLE, reason="the accept record is a CuTe DSL kernel"
)

_PKG = "tensorrt_llm._torch.modeling_v2.models.kimi_k31"


# Context projection


H, CAPTURES, LATENT, LAYERS = 1024, 3, 576, 5


class _Drafter:
    """The drafter methods the context projection calls, on plain torch ops."""

    context_kv_only = False

    def __init__(self, seed: int):
        g = torch.Generator(device="cuda").manual_seed(seed)
        self.fc_weight = (
            torch.randn(H, CAPTURES * H, generator=g, device="cuda") * 0.02
        ).bfloat16()
        self.latent_weight = (
            torch.randn(LAYERS * LATENT, H, generator=g, device="cuda") * 0.02
        ).bfloat16()
        self._fused_kv_weight = self.latent_weight
        self.model = types.SimpleNamespace(embed_tokens=None)

    def project_target_hidden(self, hidden_states):
        x = F.linear(hidden_states, self.fc_weight).float()
        return (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)).bfloat16()

    def k31_context_latents(self, projected):
        n = projected.shape[0]
        return F.linear(projected, self.latent_weight).view(n, LAYERS, LATENT)


def _step(num_gens: int, width: int, seed: int):
    g = torch.Generator(device="cuda").manual_seed(seed)
    captured = torch.randn(
        num_gens * width + 3, CAPTURES * H, generator=g, device="cuda"
    ).bfloat16()
    spec = types.SimpleNamespace(
        runtime_draft_len=width - 1, get_hidden_states=lambda n: captured[:n] if n else None
    )
    attn = types.SimpleNamespace(num_seqs=num_gens, num_contexts=0, num_ctx_tokens=0)
    return spec, attn, captured


def _context_worker():
    worker = modeling.K31DFlashWorker.__new__(modeling.K31DFlashWorker)
    worker._ctx_kv_buf = torch.empty(1, dtype=torch.bfloat16, device="cuda")
    worker._ctx_page_table = None
    worker._k31_fused_prepare_serves = lambda *args: True
    return worker


def _serial(drafter, captured, rows):
    return drafter.k31_context_latents(drafter.project_target_hidden(captured[:rows]))


@_GPU
def test_cuda_graph_replay(monkeypatch) -> None:
    num_gens, width = 3, 5
    drafter = _Drafter(seed=1)
    spec, attn, captured = _step(num_gens, width, seed=2)
    worker = _context_worker()
    out = {}
    with with_multi_stream(True):
        worker._k31_context_begin(drafter, attn, spec, num_gens * width)
        worker._k31_context_join()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.stream(stream), torch.cuda.graph(graph, stream=stream):
            worker._k31_context_begin(drafter, attn, spec, num_gens * width)
            out["latents"] = worker._k31_context_join() * 2
        torch.cuda.current_stream().wait_stream(stream)
    g = torch.Generator(device="cuda").manual_seed(3)
    for _ in range(3):
        captured.copy_(torch.randn(captured.shape, generator=g, device="cuda").bfloat16())
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(out["latents"], _serial(drafter, captured, num_gens * width) * 2)


# Accept record


SLOTS = 40
DUMMY_SLOT = SLOTS - 1


def _cache_managers():
    kda = importlib.import_module(f"{_PKG}.ops.ssm.kda_cache_manager")
    kda_dsa = importlib.import_module(f"{_PKG}.ops.ssm.kda_dsa_cache_manager")
    return kda, kda_dsa


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


def _record_worker():
    worker = modeling.K31DFlashWorker.__new__(modeling.K31DFlashWorker)
    worker._draft_reads_no_attn_metadata = True
    return worker


@_ACCEPT_RECORD_SM
@_ACCEPT_RECORD_DSL
def test_the_dsa_manager_takes_the_fused_record(monkeypatch) -> None:
    kda, kda_dsa = _cache_managers()
    cls, num_gens = kda_dsa.KimiK3DSAHybridCacheManager, 7
    spec = types.SimpleNamespace(is_cuda_graph=False)
    num_accepted, slots, dummy, prev, kv_lens = _inputs(num_gens, seed=17 * num_gens + 1)

    # Torch path, as K31DFlashWorker._forward_impl runs it.
    manager_t = _manager(cls, kda.K31KDAReplayState, _strided_copy(prev), dummy, num_gens)
    meta_t = _AttnMetadata(manager_t, num_gens, kv_lens.clone(), slots)
    worker_t = _record_worker()
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
    worker_f = _record_worker()
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


@_ACCEPT_RECORD_SM
@_ACCEPT_RECORD_DSL
def test_draft_tail_warmup_for_one_tp_request() -> None:
    worker = modeling.K31DFlashWorker.__new__(modeling.K31DFlashWorker)
    worker.guided_decoder = None
    worker._kv_rewind_amount = torch.tensor([-1], dtype=torch.int32, device="cuda")
    # TP agreement returns rank zero's accepted tokens and count from one packed row.
    packed = torch.tensor([[11, 12, 13, 14, 15, 2]], dtype=torch.int32, device="cuda")
    tokens, counts = packed[:, :-1].contiguous(), packed[:, -1].contiguous()
    assert counts.stride() == (6,)
    drafts = torch.tensor([[21, 22, 23, 24]], dtype=torch.int32, device="cuda")
    kv_lens = torch.tensor([7], dtype=torch.int32, device="cuda")
    metadata = types.SimpleNamespace(
        is_cuda_graph=True, batch_indices_cuda=torch.tensor([0], device="cuda")
    )

    result = worker._k31_draft_tail(
        types.SimpleNamespace(kv_lens_cuda=kv_lens), metadata, tokens, drafts, 1, counts
    )
    assert result.tolist() == [[12, 21, 22, 23, 24]]
    assert kv_lens.tolist() == [7]
