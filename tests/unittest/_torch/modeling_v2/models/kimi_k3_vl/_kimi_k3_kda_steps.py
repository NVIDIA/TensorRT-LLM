# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Step sequences of a target's ``kda.K3DecodeKDA`` on one GPU and a real ``KimiK3HybridCacheManager``.

One TP16 rank's shapes (6 heads, hidden 7168, a single-rank Mapping), random checkpoint-like weights finalized the
way the loader finalizes them, and the manager in each of its speculative states (none; the K3 verify kernels'
per-draft records; kda_mtp_decode's replay caches; the sequential verify's buffers). Steps are attention metadata in
the ``KimiK3MambaMetadata`` layout the layer reads. ``run_twice`` runs a step from the same pools twice and requires
the same bits (outputs and every pool of the layer); ``committed_window_follows`` checks the replay caches' committed
conv window against the live pool after a prefill or plain decode.

With kda_mtp_decode's replay caches (no draft records) the manager keeps the replay caches and the accepted-draft
count in a pool of their own, with replay slots of their own (``get_replay_state_indices``). ``request_slots`` pairs
each request with another request's replay slot, so no request's replay slot equals its state slot whatever numbers
the manager assigned; a layer that addresses the replay caches by the state slot then fails the checks.

Not a test file: the targets' ``test_*_kda_steps.py`` import it (``__extra_import_path__``).
"""

from __future__ import annotations

import dataclasses
import importlib
import weakref
from types import SimpleNamespace

import torch

__extra_import_path__ = ["."]  # this file's own import of its neighbour _kimi_k3_parity
from _kimi_k3_parity import assert_bitwise  # noqa: E402

PACKAGE = "tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl"
HEADS = 6
HIDDEN = 7168
HEAD_DIM = 128


def sm100() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability() in ((10, 0), (10, 3))


def target_module(target: str, name: str):
    return importlib.import_module(f"{PACKAGE}.{target}.{name}")


# ---------------------------------------------------------------------------------------------------------------------
# Modules
# ---------------------------------------------------------------------------------------------------------------------
def kda_config(heads: int = HEADS, hidden: int = HIDDEN) -> SimpleNamespace:
    """Kimi K3's KDA config at one rank's head count (a single-rank Mapping keeps every head)."""
    return SimpleNamespace(
        hidden_size=hidden,
        rms_norm_eps=1e-5,
        linear_attn_config=dict(
            num_heads=heads,
            head_dim=HEAD_DIM,
            short_conv_kernel_size=4,
            use_full_rank_gate=True,
            gate_lower_bound=-5.0,
        ),
    )


def build_module(target: str, layer_idx: int = 0, seed: int = 17, aux_stream=None, k3_buffers=None):
    """The target's ``K3DecodeKDA`` with random checkpoint-like weights (the default inits, a non-trivial norm weight
    and gate bias), finalized the way the weight loader finalizes it, on the current CUDA device."""
    from tensorrt_llm._torch.model_config import ModelConfig

    cls = target_module(target, "kda").K3DecodeKDA
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        torch.manual_seed(seed)
        with torch.device("cuda"):
            module = cls(kda_config(), layer_idx, aux_stream=aux_stream, model_config=ModelConfig())
        gen = torch.Generator(device="cuda").manual_seed(seed)
        with torch.no_grad():
            module.o_norm.weight.copy_(
                1 + 0.1 * torch.randn(module.o_norm.weight.shape, generator=gen, device="cuda")
            )
            module.dt_bias.copy_(
                0.5 * torch.randn(module.dt_bias.shape, generator=gen, device="cuda")
            )
            module.finalize_decode_weights()
    module.k3_buffers = k3_buffers
    return module.eval()


# ---------------------------------------------------------------------------------------------------------------------
# The cache manager
# ---------------------------------------------------------------------------------------------------------------------
#: The manager's speculative state: none; the K3 verify kernels' per-draft records (7 drafts); the replay caches of
#: kda_mtp_decode (3 drafts); the sequential verify's buffers (9 drafts, above kda_mtp_decode's 7).
MODES = {"plain": None, "records": 7, "replay": 3, "sequential": 9}


def build_manager(mode: str, max_batch_size: int = 8, num_layers: int = 1):
    from tensorrt_llm._torch.modules.kimi_kda.cache_manager import KimiK3HybridCacheManager
    from tensorrt_llm._torch.pyexecutor.resource_manager import CacheTypeCpp, DataType
    from tensorrt_llm.llmapi.llm_args import KvCacheConfig, MTPDecodingConfig
    from tensorrt_llm.mapping import Mapping

    num_spec = MODES[mode]
    spec = MTPDecodingConfig(max_draft_len=num_spec) if num_spec else None
    mgr = KimiK3HybridCacheManager(
        mamba_d_state=HEAD_DIM,
        mamba_d_conv=4,
        mamba_num_heads=HEADS,
        mamba_n_groups=HEADS,
        mamba_head_dim=HEAD_DIM,
        mamba_num_layers=num_layers,
        mamba_layer_mask=[True] * num_layers + [False],
        mamba_cache_dtype=torch.bfloat16,
        mamba_ssm_cache_dtype=torch.float32,
        kv_cache_config=KvCacheConfig(max_tokens=64 * max_batch_size, enable_block_reuse=False),
        kv_cache_type=CacheTypeCpp.SELF,
        num_layers=1,
        num_kv_heads=4,
        head_dim=64,
        tokens_per_block=32,
        max_seq_len=512,
        max_batch_size=max_batch_size,
        mapping=Mapping(world_size=1, rank=0, tp_size=1, pp_size=1),
        dtype=DataType.HALF,
        spec_config=spec,
        layer_mask=[False] * num_layers + [True],
        vocab_size=1024,
        conv_state_layout="q_k_v",
        kda_replay_num_spec=num_spec if mode in ("records", "replay") else None,
        kda_draft_records=mode == "records",
    )
    cache = mgr.mamba_layer_cache(0)
    has_replay = bool(getattr(cache, "has_kda_replay_caches", False))
    expected = {
        "plain": (False, False),
        "records": (True, True),
        "replay": (True, False),
        "sequential": (False, False),
    }[mode]
    assert (has_replay, has_replay and cache.kda_state_tok is not None) == expected, (
        mode,
        type(cache),
    )
    # Only kda_mtp_decode's replay caches have slots of their own; the draft records share the state slot.
    assert (mgr.replay_layer_group_id is not None) == (mode == "replay"), mode
    if mode == "sequential":
        assert cache.intermediate_conv_window is not None, "no sequential-verify buffers"
    return mgr


#: Per manager, the replay slot ``request_slots`` gave each state slot (managers with a replay pool of their own).
_REPLAY_SLOTS: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()


def request_slots(mgr, count: int, first_id: int) -> list:
    """State slots of ``count`` new requests. With a replay pool of its own the requests' replay slots (the manager's,
    rotated so that none equals its request's state slot) are kept for ``replay_slots_of``."""
    ids = list(range(first_id, first_id + count))
    mgr.add_dummy_requests(ids, token_nums=[8] * count, is_gen=False)
    slots = [int(s) for s in mgr.get_state_indices(ids, [False] * count)]
    assert len(set(slots)) == count, slots
    if mgr.replay_layer_group_id is not None:
        replay = [int(r) for r in mgr.get_replay_state_indices(ids)]
        assert len(set(replay)) == count, replay
        _REPLAY_SLOTS.setdefault(mgr, {}).update(zip(slots, crossed(slots, replay)))
    return slots


def crossed(state: list, replay: list) -> list:
    """``replay`` rotated by the first offset at which no entry equals the ``state`` entry of its row: still one unique
    replay slot per request, and different from the request's state slot in every row."""
    for shift in range(1, len(replay)):
        rotated = replay[shift:] + replay[:shift]
        if all(r != s for r, s in zip(rotated, state)):
            return rotated
    raise AssertionError(f"no rotation of the replay slots {replay} avoids the state slots {state}")


def replay_slots_of(mgr, slots: list) -> list:
    """The replay slots of the requests at state ``slots``: the state slots unless the manager has a replay pool of
    its own."""
    if mgr.replay_layer_group_id is None:
        return list(slots)
    replay = _REPLAY_SLOTS[mgr]
    return [replay[s] for s in slots]


def fill_pools(mgr, seed: int) -> None:
    """Random finite pool contents (the replay records and accepted counts zero), so no step starts from zeros."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    cache = mgr.mamba_layer_cache(0)
    with torch.no_grad():
        cache.conv.copy_(
            (torch.randn(cache.conv.shape, generator=gen, device="cuda") * 0.5).bfloat16()
        )
        cache.temporal.copy_(torch.randn(cache.temporal.shape, generator=gen, device="cuda") * 0.05)
        for name in ("kda_conv_q", "kda_conv_k", "kda_conv_v"):
            t = getattr(cache, name, None)
            if t is not None:
                t.copy_((torch.randn(t.shape, generator=gen, device="cuda") * 0.5).to(t.dtype))
        for name in ("prev_num_accepted_tokens", "kda_state_tok"):
            t = getattr(cache, name, None)
            if t is not None:
                t.zero_()


def pool_tensors(mgr) -> dict:
    """Every tensor of the layer's cache view: the pools the layer reads or writes."""
    cache = mgr.mamba_layer_cache(0)
    return {
        f.name: getattr(cache, f.name)
        for f in dataclasses.fields(cache)
        if isinstance(getattr(cache, f.name), torch.Tensor)
    }


def snapshot(mgr) -> dict:
    return {name: t.clone() for name, t in pool_tensors(mgr).items()}


def restore(mgr, snap: dict) -> None:
    with torch.no_grad():
        for name, t in pool_tensors(mgr).items():
            t.copy_(snap[name])


def accept(mgr, slots: list, accepted: list) -> None:
    """The sampler accepted ``accepted[n]`` drafts of request n: what the engine's ``update_mamba_states`` records
    (replay modes) or promotes (the sequential verify)."""
    from tensorrt_llm._torch.pyexecutor.kv_cache.mamba_cache_manager.common import (
        MambaAcceptanceBatch,
    )

    slot_t = torch.tensor(slots, dtype=torch.int32, device="cuda")
    acc = torch.tensor(accepted, dtype=torch.int32, device="cuda")
    batch = MambaAcceptanceBatch(
        attention_metadata=None,
        num_contexts=0,
        num_generations=len(slots),
        num_accepted_tokens=acc + 1,
        accepted_positions=acc,
        source_state_indices=torch.arange(len(slots), dtype=torch.int32, device="cuda"),
        destination_state_indices=slot_t,
        is_dummy_request=None,
    )
    if mgr.replay_layer_group_id is None:
        mgr._speculative_state.update(batch)
    else:
        # The accepted counts go to the requests' replay slots (KDAReplayRegion.update).
        replay_t = torch.tensor(replay_slots_of(mgr, slots), dtype=torch.int32, device="cuda")
        mgr._speculative_state.update(batch, replay_t)


# ---------------------------------------------------------------------------------------------------------------------
# One step
# ---------------------------------------------------------------------------------------------------------------------
def _aligned_int32(values: list) -> torch.Tensor:
    """An int32 CUDA tensor at a 16-byte-aligned address (as the metadata's generation slots are)."""
    t = torch.tensor(values, dtype=torch.int32, device="cuda")
    assert t.data_ptr() % 16 == 0
    return t


class StepMetadata(SimpleNamespace):
    """The attention metadata stand-in: a namespace the engine's model attributes can hold a weak reference to, as
    they hold the real ``AttentionMetadata`` (a plain ``SimpleNamespace`` cannot be weakly referenced)."""


def make_step(
    mgr,
    ctx_slots: list,
    ctx_lens: list,
    has_init: list,
    gen_slots: list,
    tokens_per_gen: int = 1,
) -> StepMetadata:
    """The attention metadata of one step: context requests (packed first), then generation requests of
    ``tokens_per_gen`` tokens each, with the fields the KDA layers read (the ``KimiK3MambaMetadata`` layout)."""
    n, g = len(ctx_slots), len(gen_slots)
    num_ctx_tokens = sum(ctx_lens)
    starts = [0]
    for length in ctx_lens + [tokens_per_gen] * g:
        starts.append(starts[-1] + length)
    order = sorted(range(n), key=lambda i: (-ctx_lens[i], i))
    chunks = [0]
    for length in ctx_lens:
        chunks.append(chunks[-1] + (length + 15) // 16)
    state_indices = _aligned_int32(ctx_slots + gen_slots)
    generation_state_indices = _aligned_int32(gen_slots) if g else None
    if mgr.replay_layer_group_id is None:
        # The replay caches, if any, share the state slots: the metadata aliases the state vectors.
        replay_state_indices = state_indices
        generation_replay_state_indices = generation_state_indices
    else:
        replay_state_indices = _aligned_int32(replay_slots_of(mgr, ctx_slots + gen_slots))
        generation_replay_state_indices = (
            _aligned_int32(replay_slots_of(mgr, gen_slots)) if g else None
        )
    mamba = SimpleNamespace(
        state_indices=state_indices,
        generation_state_indices=generation_state_indices,
        replay_state_indices=replay_state_indices,
        generation_replay_state_indices=generation_replay_state_indices,
        query_start_loc=torch.tensor(starts, dtype=torch.int32, device="cuda"),
        query_start_loc_long=torch.tensor(starts, dtype=torch.int64, device="cuda"),
        has_initial_states=torch.tensor(has_init + [False] * g, dtype=torch.bool, device="cuda"),
        use_initial_states=any(has_init),
        kda_max_seqlen=max(ctx_lens, default=0),
        kda_seq_order=torch.tensor(order, dtype=torch.int32, device="cuda"),
        kda_cu_chunks=torch.tensor(chunks, dtype=torch.int32, device="cuda"),
    )
    return StepMetadata(
        mamba_metadata=mamba,
        num_contexts=n,
        num_generations=g,
        num_ctx_tokens=num_ctx_tokens,
        num_tokens=num_ctx_tokens + g * tokens_per_gen,
        seq_lens=torch.tensor(ctx_lens + [tokens_per_gen] * g, dtype=torch.int32),
        kv_cache_manager=mgr,
    )


def hidden_for(metadata, seed: int) -> torch.Tensor:
    gen = torch.Generator(device="cuda").manual_seed(seed)
    return (torch.randn(metadata.num_tokens, HIDDEN, generator=gen, device="cuda") * 0.1).bfloat16()


def run_twice(mgr, run, what: str) -> torch.Tensor:
    """Run ``run()`` from the current pools, restore them, run it again: the outputs and every pool must be the same
    bits (no step reads state it does not own, or a buffer a previous call left). Returns the output; the pools stay as
    the second run left them."""
    before = snapshot(mgr)
    with torch.inference_mode():
        first = run().clone()
    torch.cuda.synchronize()
    after_first = snapshot(mgr)
    restore(mgr, before)
    with torch.inference_mode():
        second = run().clone()
    torch.cuda.synchronize()
    assert_bitwise(second, first, f"{what}: output")
    for name, t in pool_tensors(mgr).items():
        assert_bitwise(t, after_first[name], f"{what}: pool {name}")
    return second


def committed_window_follows(mgr, slots: list, what: str) -> None:
    """With replay caches, the committed conv window (columns 0..2 of ``kda_conv_q/k/v``) at each request's replay
    slot equals the live conv pool's q / k / v sections at its state slot, bit for bit: what ``commit_conv_window``
    keeps after a prefill or plain decode. Without the caches there is nothing to check."""
    cache = mgr.mamba_layer_cache(0)
    if not bool(getattr(cache, "has_kda_replay_caches", False)):
        return
    rows = torch.tensor(slots, dtype=torch.long, device="cuda")
    replay_rows = torch.tensor(replay_slots_of(mgr, slots), dtype=torch.long, device="cuda")
    dim = HEADS * HEAD_DIM
    for i, name in enumerate(("kda_conv_q", "kda_conv_k", "kda_conv_v")):
        live = cache.conv[rows, i * dim : (i + 1) * dim]
        window = getattr(cache, name)[replay_rows, :, : live.shape[-1]]
        assert_bitwise(window, live, f"{what}: {name} committed window")


def decode_step_of(target: str, metadata):
    """The target's own ``decode_step`` classification of ``metadata``'s step."""
    return target_module(target, "step").decode_step(metadata, metadata.num_tokens)
