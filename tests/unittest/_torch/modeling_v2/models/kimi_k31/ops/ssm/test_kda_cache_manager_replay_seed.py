# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3.1 KDA cache manager: the replay caches of transferred requests seeded in all KDA layers at once.

When the KV transfer of a disaggregated generation request completes (and when a suspended request resumes), the
manager seeds the request's replay caches in every local KDA layer: the committed conv window copied into the q / k
/ v replay conv caches, their draft columns and the k / g / v / beta caches zeroed. The family's KDAReplayState
stacks each buffer kind over the layers and seeds them with one launch per kind (seed_kda_replay_layers); the
engine's per-layer loop is the fallback. The two must write the same bits.

The manager is the one the K3.1 targets build (KimiK3DSAHybridCacheManager) with K3.1's geometry on real KV cache
manager V2 pools: 69 KDA layers interleaved 3:1 with 23 MLA (DSA) layers, 128-wide heads at a generation layout's
head count, a width-4 short convolution, an fp32 recurrent state and the DFlash2 drafter's replay width. Every
state and replay buffer starts with random contents (pending drafts in every slot), so a missed or stray write
shows.
"""

import importlib
from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.ssm import (
    kda_cache_manager as family_cm,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.ssm.kda_dsa_cache_manager import (
    KimiK3DSAHybridCacheManager,
)
from tensorrt_llm._torch.modules.kimi_kda import cache_manager as engine_cm
from tensorrt_llm._torch.pyexecutor.resource_manager import CacheTypeCpp, DataType
from tensorrt_llm.llmapi.llm_args import (
    DeepSeekSparseAttentionConfig,
    KvCacheConfig,
    MambaStateConfig,
    MTPDecodingConfig,
)
from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

HEAD_DIM = 128
CONV_WIDTH = 4
NUM_LAYERS = 92
# K3.1: three KDA layers, then one MLA layer.
KDA_MASK = [layer % 4 != 3 for layer in range(NUM_LAYERS)]
MLA_MASK = [not is_kda for is_kda in KDA_MASK]
NUM_KDA_LAYERS = sum(KDA_MASK)
MAX_BATCH_SIZE = 8
TOKENS_PER_BLOCK = 32


def _text_config(heads: int) -> SimpleNamespace:
    kda_layers = [layer + 1 for layer, is_kda in enumerate(KDA_MASK) if is_kda]
    mla_layers = [layer + 1 for layer, is_mla in enumerate(MLA_MASK) if is_mla]
    return SimpleNamespace(
        architectures=["KimiLinearForCausalLM"],
        model_type="kimi_linear",
        num_hidden_layers=NUM_LAYERS,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        linear_attn_config={
            "head_dim": HEAD_DIM,
            "num_heads": heads,
            "short_conv_kernel_size": CONV_WIDTH,
            "kda_layers": kda_layers,
            "full_attn_layers": mla_layers,
        },
        dsa_config={"dsa_layers": mla_layers, "index_topk_freq": 1},
        index_topk_freq=1,
        index_skip_topk_offset=1,
        dtype=torch.bfloat16,
    )


def _make_manager(heads: int, num_spec: int) -> KimiK3DSAHybridCacheManager:
    return KimiK3DSAHybridCacheManager(
        mamba_d_state=HEAD_DIM,
        mamba_d_conv=CONV_WIDTH,
        mamba_num_heads=heads,
        mamba_n_groups=heads,
        mamba_head_dim=HEAD_DIM,
        mamba_num_layers=NUM_KDA_LAYERS,
        mamba_layer_mask=KDA_MASK,
        mamba_cache_dtype=torch.bfloat16,
        mamba_ssm_cache_dtype=torch.float32,
        kv_cache_config=KvCacheConfig(
            max_tokens=2048,
            enable_block_reuse=False,
            mamba_state_config=MambaStateConfig(
                periodic_snapshot_interval=0, additional_snapshot_offsets_from_end=[0]
            ),
        ),
        kv_cache_type=CacheTypeCpp.SELFKONLY,
        num_layers=NUM_LAYERS - NUM_KDA_LAYERS,
        num_kv_heads=1,
        head_dim=576,
        tokens_per_block=TOKENS_PER_BLOCK,
        max_seq_len=256,
        max_batch_size=MAX_BATCH_SIZE,
        mapping=Mapping(world_size=1, rank=0, tp_size=1, pp_size=1),
        dtype=DataType.BF16,
        layer_mask=MLA_MASK,
        sparse_attention_config=DeepSeekSparseAttentionConfig(
            index_n_heads=4, index_head_dim=128, index_topk=64, indexer_k_dtype="fp8"
        ),
        pretrained_config=_text_config(heads),
        spec_config=MTPDecodingConfig(max_draft_len=num_spec),
        kda_replay_num_spec=num_spec,
    )


def _state_views(manager) -> dict[str, torch.Tensor]:
    """Every KDA state and replay buffer view of the manager, by name."""
    replay = manager._kda_replay
    views = {"prev_num_accepted_tokens": replay.prev_num_accepted_tokens}
    for layer, (ssm, conv) in enumerate(zip(manager.all_ssm_states, manager.all_conv_states)):
        views[f"{layer}.ssm"] = ssm
        views[f"{layer}.conv"] = conv
        for name, view in replay.pool_layers[layer].items():
            views[f"{layer}.{name}"] = view
    return views


def _randomize(views: dict[str, torch.Tensor], num_spec: int) -> None:
    gen = torch.Generator(device="cuda").manual_seed(7)
    for name, view in views.items():
        if view.dtype == torch.int32:
            view.copy_(
                torch.randint(1, num_spec + 1, view.shape, generator=gen, device="cuda").to(
                    view.dtype
                )
            )
        else:
            view.copy_(torch.randn(view.shape, generator=gen, device="cuda"))


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.view({2: torch.int16, 4: torch.int32}[t.element_size()])


def _snapshot(views: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    torch.cuda.synchronize()
    return {name: _bits(view.clone()) for name, view in views.items()}


@pytest.mark.parametrize(
    ("heads", "num_spec"),
    [(6, 4), (12, 7), (96, 4)],
    ids=["tp16-w5", "tp8-w8", "dp-w5"],
)
def test_transfer_seeding_matches_the_per_layer_loop_bit_for_bit(request, heads, num_spec):
    manager = _make_manager(heads, num_spec)
    request.addfinalizer(manager.shutdown)
    replay = manager._kda_replay
    assert isinstance(replay, family_cm.K31KDAReplayState)
    request_ids = list(range(100, 104))
    assert manager.add_dummy_requests(request_ids, token_nums=[32] * len(request_ids), is_gen=False)
    slots = dict(zip(request_ids, manager.get_state_indices(request_ids)))
    # Two transfers complete in one step, one later; request 102 stays as it was.
    transfers = [[103, 100], [101]]
    seeded = {slots[rid] for batch in transfers for rid in batch}
    views = _state_views(manager)
    _randomize(views, num_spec)
    initial = _snapshot(views)

    for batch in transfers:
        manager.on_state_transfer_complete(batch)
    batched = _snapshot(views)
    plan = replay._batched_seed_plan
    assert plan, "the batched seeding must be planned for K3.1's V2 pools"
    assert plan[1].conv_pool.shape[:2] == (NUM_KDA_LAYERS, replay.prev_num_accepted_tokens.shape[0])

    for name, view in views.items():
        view.copy_(initial[name].view(view.dtype))
    replay._batched_seed_plan = False  # the engine's per-layer loop
    for batch in transfers:
        manager.on_state_transfer_complete(batch)
    looped = _snapshot(views)

    for name in views:
        assert torch.equal(batched[name], looped[name]), f"{name} differs from the per-layer loop"
    # And the seeding did happen, in the seeded slots only.
    committed = CONV_WIDTH - 1
    for name, view in views.items():
        after, before = batched[name], initial[name]
        untouched = [slot for slot in range(view.shape[0]) if slot not in seeded]
        assert torch.equal(after[untouched], before[untouched]), f"{name}: another slot moved"
        if name.endswith((".ssm", ".conv")):
            assert torch.equal(after, before), f"{name}: the live states must be read only"
    for layer in (0, NUM_KDA_LAYERS - 1):
        conv = views[f"{layer}.conv"]
        dim = conv.shape[1] // 3
        for section, cache_name in enumerate(("kda_conv_q", "kda_conv_k", "kda_conv_v")):
            cache = views[f"{layer}.{cache_name}"]
            for slot in seeded:
                assert torch.equal(
                    _bits(cache[slot, :, :committed]),
                    _bits(conv[slot, section * dim : (section + 1) * dim]),
                )
                assert cache[slot, :, committed:].eq(0).all()
        for cache_name in ("kda_k_cache", "kda_g_cache", "kda_v_cache", "kda_beta_cache"):
            assert views[f"{layer}.{cache_name}"][sorted(seeded)].eq(0).all()
    assert views["prev_num_accepted_tokens"][sorted(seeded)].eq(0).all()


def _synthetic_state(num_layers: int = 3, num_spec: int = 4, heads: int = 2):
    """A family replay state over per-layer views, laid out like the engine's standalone allocation."""
    dim = heads * HEAD_DIM
    slots = 6
    fields, _ = engine_cm.allocate_kda_replay_fields(
        num_local_layers=num_layers,
        cache_size=slots,
        num_speculative_tokens=num_spec,
        conv_dim=3 * dim,
        conv_kernel_size=CONV_WIDTH,
        num_heads=heads,
        device=torch.device("cuda"),
    )
    state = family_cm.K31KDAReplayState(num_spec)
    for name, value in fields.items():
        setattr(state, name, value)
    conv = torch.randn(num_layers, slots, 3 * dim, CONV_WIDTH - 1, device="cuda").to(torch.bfloat16)
    state._conv_states = tuple(conv.unbind(0))
    state._conv_section_dims = (dim, dim, dim)
    state._committed_window = CONV_WIDTH - 1
    return state


@pytest.fixture
def emitted(monkeypatch):
    """The messages the TRT-LLM logger emits, with a fresh once-per-key memory."""
    messages = []
    monkeypatch.setattr(logger, "_appeared_keys", set())
    monkeypatch.setattr(logger, "log", lambda severity, *msg: messages.append((severity, *msg)))
    return messages


def test_seeding_plan_is_logged(emitted):
    state = _synthetic_state()
    assert state._get_batched_seed_plan() is not None
    assert [(severity, "seeded in all 3 local KDA layers" in msg) for severity, msg in emitted] == [
        (logger.INFO, True)
    ]


def test_declined_plan_is_logged_once_per_reason(emitted):
    # One layer's live conv state in a tensor of its own: the layers are no longer evenly spaced.
    states = []
    for _ in range(2):
        state = _synthetic_state()
        state._conv_states = (state._conv_states[0].clone(), *state._conv_states[1:])
        states.append(state)
    for state in states:
        assert state._get_batched_seed_plan() is None
        assert state._batched_seed_plan is False
    assert len(emitted) == 1
    severity, message = emitted[0]
    assert severity == logger.WARNING
    assert "seeded layer by layer" in message and "live convolution states" in message


def test_declined_plan_seeds_layer_by_layer_with_the_same_result(emitted):
    slots = torch.tensor([4, 1], dtype=torch.long, device="cuda")
    results = []
    for split_layout in (False, True):
        torch.manual_seed(5)
        state = _synthetic_state()
        state.prev_num_accepted_tokens.fill_(3)
        buffers = (
            state.kda_conv_q,
            state.kda_conv_k,
            state.kda_conv_v,
            state.kda_k_cache,
            state.kda_g_cache,
            state.kda_v_cache,
            state.kda_beta_cache,
        )
        for buffer in buffers:
            buffer.copy_(torch.randn(buffer.shape, device="cuda"))
        if split_layout:
            # The g caches of the layers moved apart by unequal gaps (same values): that kind cannot be stacked.
            per_layer = state.kda_g_cache[0].numel()
            storage = torch.empty(3 * per_layer + 64 + 192, dtype=torch.float32, device="cuda")
            g_layers = [
                storage[offset : offset + per_layer].view(state.kda_g_cache.shape[1:])
                for offset in (0, per_layer + 64, 2 * per_layer + 64 + 192)
            ]
            for layer, view in enumerate(g_layers):
                view.copy_(state.kda_g_cache[layer])
            state.pool_layers = [
                {**state._layer_cache_fields(layer), "kda_g_cache": g_layers[layer]}
                for layer in range(3)
            ]
            for layer in state.pool_layers:
                layer.pop("prev_num_accepted_tokens")
        state.seed_transferred_slots(slots)
        assert bool(state._batched_seed_plan) is not split_layout
        layers = [state._layer_cache_fields(layer) for layer in range(3)]
        results.append(
            [
                _bits(layer[name].clone())
                for layer in layers
                for name in sorted(layer)
                if layer[name] is not None
            ]
        )
    torch.cuda.synchronize()
    assert any("kda_g_cache" in message for _, message in emitted)
    for got, want in zip(*results):
        assert torch.equal(got, want)


# ---- the pending-draft commit's kernel, compiled once the manager's pools are bound


@pytest.fixture
def fresh_compile_caches(monkeypatch):
    """Empty kernel compile caches for the test (the variants other tests compiled are not visible)."""
    from tensorrt_llm._torch.custom_ops import cute_dsl_kimi_k3_kda_mtp_ops as kernel_ops

    commit_op = importlib.import_module(
        "tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.ssm.kda_mtp_commit_pending_drafts"
    )
    compiled = {}
    monkeypatch.setattr(kernel_ops, "_compiled_cache", compiled)
    monkeypatch.setattr(commit_op, "_RUN_VARIANTS", set())
    return compiled


@pytest.mark.parametrize(
    ("heads", "num_spec"),
    [(12, 4), (12, 7), (6, 4), (6, 7), (3, 4), (3, 7), (96, 4)],
    ids=["tp8-w5", "tp8-w8", "tp16-w5", "tp16-w8", "tp32-w5", "tp32-w8", "dp-w5"],
)
def test_pending_draft_commit_is_compiled_with_the_pools(
    request, fresh_compile_caches, heads, num_spec
):
    """The manager compiles the commit's kernel when it binds its pools: a suspended request's commit later (one
    slot, every KDA layer) compiles nothing, and warming again changes no state or replay buffer."""
    manager = _make_manager(heads, num_spec)
    request.addfinalizer(manager.shutdown)
    replay = manager._kda_replay
    assert isinstance(replay, family_cm.K31KDAReplayState)
    torch.cuda.synchronize()
    compiled = len(fresh_compile_caches)
    assert compiled == 1, f"{compiled} commit variants compiled with the pools"
    views = _state_views(manager)
    _randomize(views, num_spec)
    before = _snapshot(views)
    replay.warm_commit_pending_drafts()
    after = _snapshot(views)
    for name in before:
        assert torch.equal(before[name], after[name]), f"warming changed {name}"
    request_ids = [200]
    assert manager.add_dummy_requests(request_ids, token_nums=[32], is_gen=False)
    (slot,) = manager.get_state_indices(request_ids)
    replay.prev_num_accepted_tokens[slot] = num_spec
    replay.commit_pending_drafts(slot)
    torch.cuda.synchronize()
    assert len(fresh_compile_caches) == compiled, (
        "the first suspended-request commit compiled its kernel"
    )


def test_unrun_commit_layout_raises_under_capture(request, fresh_compile_caches):
    """Under CUDA graph capture the commit refuses a layout it has not run (it would compile inside the capture)."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.ssm.kda_mtp_commit_pending_drafts import (
        kda_mtp_commit_pending_drafts,
    )

    manager = _make_manager(6, 4)
    request.addfinalizer(manager.shutdown)
    replay = manager._kda_replay
    fields = replay._layer_cache_fields(0)
    conv = replay._conv_states[0]
    two_slots = torch.tensor([1, 2], dtype=torch.int32, device="cuda")
    none_pending = torch.zeros(2, dtype=torch.int32, device="cuda")
    args = (
        replay._ssm_states[0],
        fields["kda_conv_q"],
        fields["kda_conv_k"],
        fields["kda_conv_v"],
        fields["kda_k_cache"],
        fields["kda_g_cache"],
        fields["kda_v_cache"],
        fields["kda_beta_cache"],
        two_slots,
        none_pending,
        fields["kda_conv_q"].shape[2] - conv.shape[2],
        conv.shape[2] + 1,
    )
    torch.cuda.synchronize()
    compiled = len(fresh_compile_caches)
    graph = torch.cuda.CUDAGraph()
    with pytest.raises(RuntimeError, match="compile its kernel inside it"):
        with torch.cuda.graph(graph):
            kda_mtp_commit_pending_drafts(*args)
    assert len(fresh_compile_caches) == compiled
