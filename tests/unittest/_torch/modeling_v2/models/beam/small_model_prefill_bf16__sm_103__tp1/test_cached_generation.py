# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU tests for cached greedy generation and prefix reuse on the Beam six-block model.

Configuration under test: the six-block model of ``test_small_model_prefill.py`` (bf16, sm_103,
eager TP1, generated weights) driven by ``generation.BeamSmallModelGenerator`` -- one
``KVCacheManagerV2`` with whole-page block reuse, 32-token pages and a 1024-token capacity,
TRTLLM attention through the catalog ``thop_attention`` op, and the existing ``TorchSampler``.

What is checked, without model files or captured data:

* the request lifecycle -- create, prefill, decode, sampled tokens and finish reason,
  release -- with its counters, and that nothing runs after the finishing token;
* that a decode step appends exactly its own keys and values, one row per layer at the
  token's absolute position, and changes no other written slot;
* prefix reuse through the manager: a matching prefix is claimed page by page and only the
  rest of the prompt is computed, a different first token misses, and two live requests
  sharing a prefix never write each other's slots;
* the sampler's greedy choice on identical logits (lowest index among exact maxima);
* the sliding window at the cache level: at query position 512 the sliding layer excludes
  key 0 exactly and the full layer keeps it;
* rejections before any kernel runs: malformed or foreign request state, capacity overflow,
  positions that do not continue the cache, a step that does not begin where the request's
  cache history ends (rewound into cached tokens -- a prefix reused from another request
  included -- or leaving a gap), a cached prefix without paged-context FMHA and a block-offset
  table that does not address the request's pages, each leaving every written slot unchanged;
  and weights that change -- in place or by replacing a parameter -- while a generator holds a
  cache computed from them, refused before the step looks up a prefix or allocates.

Numerical agreement with the source model is established separately.

Invocation (fp32 steps need the TF32 overrides removed before torch is imported; the
generation-phase attention kernels compile on first use and need the installation's ``pip``
on ``PATH``)::

    env -u TORCH_ALLOW_TF32_CUBLAS_OVERRIDE -u NVIDIA_TF32_OVERRIDE python -m pytest \\
        tests/unittest/_torch/modeling_v2/models/beam/small_model_prefill_bf16__sm_103__tp1

Dependencies: a TensorRT-LLM build for sm_103 with flashinfer importable and about 130 GiB of
free device memory (generated state and loaded model). No model files.
"""

from __future__ import annotations

import contextlib
import gc
import math
import types
import zlib
from collections import OrderedDict

import numpy as np
import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog.attention.thop_attention import (
    thop_attention,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.beam.full_model_nvfp4__sm_103__tp4 import (
    _attention,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.beam.small_model_prefill_bf16__sm_103__tp1 import (  # noqa: E501
    generation,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.beam.small_model_prefill_bf16__sm_103__tp1 import (  # noqa: E501
    model as model_module,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.beam.small_model_prefill_bf16__sm_103__tp1.generation import (  # noqa: E501
    BeamSmallModelGenerator,
    Lifecycle,
)
from tensorrt_llm._torch.attention.backends.interface import AttentionRuntimeFeatures
from tensorrt_llm._torch.attention.backends.trtllm import TrtllmAttentionMetadata
from tensorrt_llm._torch.metadata import KVCacheParams
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest, LlmRequestState
from tensorrt_llm._torch.pyexecutor.sampler import TorchSampler
from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests
from tensorrt_llm._torch.pyexecutor.seq_slot_manager import SeqSlotManager
from tensorrt_llm.bindings.executor import FinishReason

assert torch.cuda.is_available(), "the Beam six-block model requires a CUDA device"

pytestmark = pytest.mark.high_cuda_memory

VOCAB = 151936
TPB = 32

CONFIG = types.MappingProxyType(
    dict(
        num_hidden_layers=6,
        hidden_size=5120,
        intermediate_size=20480,
        moe_intermediate_size=2560,
        num_attention_heads=48,
        num_key_value_heads=8,
        head_dim=128,
        vocab_size=VOCAB,
        rms_norm_eps=1e-6,
        hidden_act="silu",
        attention_bias=False,
        attention_dropout=0.0,
        sliding_window=512,
        layer_types=("sliding_attention",) * 5 + ("full_attention",),
        mlp_layer_types=("dense",) * 3 + ("sparse",) * 3,
        rope_parameters={
            "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
            "full_attention": None,
        },
        num_experts=256,
        num_experts_per_tok=8,
        n_group=8,
        topk_group=4,
        norm_topk_prob=True,
        routed_scaling_factor=2.5,
        tie_word_embeddings=False,
        final_logit_softcapping=30.0,
        max_position_embeddings=1048576,
    )
)


def model_config() -> types.SimpleNamespace:
    fields = dict(CONFIG)
    fields["layer_types"] = list(fields["layer_types"])
    fields["mlp_layer_types"] = list(fields["mlp_layer_types"])
    fields["rope_parameters"] = {
        k: (dict(v) if v else v) for k, v in fields["rope_parameters"].items()
    }
    return types.SimpleNamespace(**fields)


def _half_width(name: str, shape: tuple) -> tuple[float, float]:
    if name == model_module.EMBED:
        return math.sqrt(3.0), 0.0
    if name.endswith("norm.weight") or name.endswith("layernorm.weight"):
        return 0.4, 1.0
    if name.endswith("gate_bias"):
        return 0.75 * math.sqrt(3.0), 0.0
    if name.endswith("e_score_correction_bias"):
        return 0.05 * math.sqrt(3.0), 0.0
    return math.sqrt(3.0 / shape[-1]), 0.0


def generate_state(seed: int) -> "OrderedDict[str, torch.Tensor]":
    """All 2394 tensors, each from its own generator keyed by seed and name, fp32 rounded once."""
    state = OrderedDict()
    for name, (shape, dtype) in model_module.weight_schema(model_config()).items():
        g = torch.Generator(device="cuda").manual_seed(zlib.crc32(f"{seed}|{name}".encode()))
        half, offset = _half_width(name, shape)
        t = torch.rand(shape, generator=g, device="cuda", dtype=torch.float32)
        state[name] = (t.mul_(2.0 * half).sub_(half).add_(offset)).to(dtype)
    return state


def token_stream(name: str, length: int) -> list:
    g = torch.Generator(device="cuda").manual_seed(zlib.crc32(f"17|{name}".encode()))
    return torch.randint(0, VOCAB, (length,), generator=g, device="cuda").tolist()


_CACHE: dict = {}


@pytest.fixture(scope="module", autouse=True)
def device_memory():
    """Release the cached model before the next module runs."""
    yield
    _CACHE.clear()
    gc.collect()
    torch.cuda.empty_cache()


def loaded() -> model_module.BeamSmallModelPrefillBf16Sm103Tp1:
    """The seed-17 model; its generated state is dropped once loaded."""
    if "model" not in _CACHE:
        state = generate_state(17)
        model = model_module.BeamSmallModelPrefillBf16Sm103Tp1(model_config(), device="cuda")
        model.load_weights(state)
        del state
        torch.cuda.empty_cache()
        _CACHE["model"] = model
    return _CACHE["model"]


@contextlib.contextmanager
def counting_attention_calls():
    """Count thop attention calls (and their row counts) while letting them run."""
    cls = type(thop_attention)
    original = cls.__call__
    calls: list = []

    def counted(self, *args, **kwargs):
        calls.append(
            (kwargs["local_layer_idx"], kwargs["q"].shape[0], kwargs["q"].detach().clone())
        )
        return original(self, *args, **kwargs)

    cls.__call__ = counted
    try:
        yield calls
    finally:
        cls.__call__ = original


def read_slot(mgr, request_id: int, layer: int, position: int):
    """(K [8, 128], V [8, 128]) of one token slot, through the manager's own addressing."""
    pool = int(mgr.impl.get_layer_group_id(model_module.LayerId(mgr.layer_offsets[layer])))
    cache = mgr.kv_cache_map[request_id]
    pages = np.frombuffer(
        cache.get_base_page_indices(pool), dtype=np.int32, count=int(cache.num_blocks)
    )
    buf = mgr.get_buffers(layer, kv_layout="HND")
    row = int(pages[position // TPB]) * mgr.get_layer_page_index_scale(layer) // 2
    return buf[row, 0, :, position % TPB, :], buf[row, 1, :, position % TPB, :]


def written_bytes(mgr, request_id: int, length: int) -> torch.Tensor:
    """Every written K/V slot of a request, all six layers, as one int16 tensor."""
    rows = []
    for layer in range(6):
        for p in range(length):
            k, v = read_slot(mgr, request_id, layer, p)
            rows.append(torch.cat((k, v)).view(torch.int16))
    return torch.stack(rows).cpu()


def bitwise_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    if a.dtype != b.dtype or a.shape != b.shape:
        return False
    view = {torch.bfloat16: torch.int16, torch.float32: torch.int32}.get(a.dtype)
    return bool(torch.equal(a.contiguous().view(view), b.contiguous().view(view)))


# ------------------------------------------------------------------ lifecycle and stopping


def test_generation_lifecycle_counters_and_length_stop() -> None:
    """create -> prefill -> decode until LENGTH -> release, with counters and call counts."""
    model = loaded()
    prompt = token_stream("generation_prompt", 17)
    caller = list(prompt)
    with BeamSmallModelGenerator(model) as gen, counting_attention_calls() as calls:
        state = gen.create_request(prompt, max_new_tokens=5)
        assert state.lifecycle is Lifecycle.NEW
        first = gen.prefill(state)
        assert state.lifecycle is Lifecycle.PREFILLED and state.consumed == 17
        assert first.logits.shape == (VOCAB,) and first.logits.dtype == torch.float32
        assert first.token_id == int(torch.argmax(first.logits)) == state.pending_token
        assert [rows for _, rows, _ in calls] == [17] * 6
        steps = [first]
        while state.lifecycle is not Lifecycle.FINISHED:
            steps.append(gen.decode(state))
        assert state.finish_reason == FinishReason.LENGTH
        assert len(state.output_token_ids) == 5 and steps[-1].finish_reason == FinishReason.LENGTH
        # The last sampled token is returned but never consumed: P + G - 1 cached tokens.
        assert state.consumed == 17 + 5 - 1
        assert [s.position for s in steps] == [16, 17, 18, 19, 20]
        assert len(calls) == 6 * 5 and all(rows == 1 for _, rows, _ in calls[6:])
        for step in steps:
            assert (
                bool(torch.isfinite(step.logits).all()) and float(step.logits.abs().max()) <= 30.0
            )
            assert step.token_id == int(torch.argmax(step.logits))
        with pytest.raises(ValueError, match="finished"):
            gen.decode(state)
        assert len(calls) == 30
        gen.release(state)
        assert state.lifecycle is Lifecycle.RELEASED
        for call in (gen.decode, gen.prefill, gen.release):
            with pytest.raises(ValueError, match="released"):
                call(state)
    assert prompt == caller


def test_end_id_stops_before_the_budget() -> None:
    """END_ID at the first sampled token after the first that is new to the run: exactly j
    tokens, END_ID, and no model call after the j-th; END_ID also wins at the token limit."""
    model = loaded()
    prompt = token_stream("generation_prompt", 17)
    with BeamSmallModelGenerator(model) as gen:
        free = gen.generate(gen.create_request(prompt, max_new_tokens=8, reuse=False))
        assert free.finish_reason == FinishReason.LENGTH and len(free.token_ids) == 8
        tokens = free.token_ids
        j = next(j for j in range(2, 9) if tokens[j - 1] not in tokens[: j - 1])
        for state in list(gen.live_requests):
            gen.release(state)
        for budget in (8, j):
            with counting_attention_calls() as calls:
                state = gen.create_request(
                    prompt, max_new_tokens=budget, end_id=tokens[j - 1], reuse=False
                )
                out = gen.generate(state)
            assert out.token_ids == tokens[:j] and out.finish_reason == FinishReason.END_ID
            assert len(calls) == 6 * j and state.consumed == 17 + j - 1
            gen.release(state)


@pytest.mark.parametrize(
    "case",
    [
        "empty",
        "token_out_of_range",
        "zero_budget",
        "over_capacity",
        "bad_end_id",
        "end_id_no_sampling",
    ],
)
def test_create_request_rejects_invalid_requests(case) -> None:
    model = loaded()
    with BeamSmallModelGenerator(model) as gen:
        kwargs = dict(max_new_tokens=2)
        prompt = [1, 2, 3]
        if case == "empty":
            prompt = []
        elif case == "token_out_of_range":
            prompt = [VOCAB]
        elif case == "zero_budget":
            kwargs["max_new_tokens"] = 0
        elif case == "over_capacity":
            prompt, kwargs["max_new_tokens"] = [1] * 1000, 25
        elif case == "bad_end_id":
            kwargs["end_id"] = -1
        else:
            kwargs.update(end_id=5, sample=False)
        with pytest.raises(ValueError):
            gen.create_request(prompt, **kwargs)
        assert gen.live_requests == ()


def test_handles_are_owned_and_bounded() -> None:
    """A third live state, a foreign handle and a decode before prefill are refused."""
    model = loaded()
    with BeamSmallModelGenerator(model) as gen, BeamSmallModelGenerator(model) as other:
        a = gen.create_request([1, 2, 3], max_new_tokens=1)
        b = gen.create_request([4, 5, 6], max_new_tokens=1)
        with pytest.raises(ValueError, match="live"):
            gen.create_request([7], max_new_tokens=1)
        foreign = other.create_request([1, 2, 3], max_new_tokens=1)
        with counting_attention_calls() as calls:
            with pytest.raises(ValueError, match="another generator"):
                gen.prefill(foreign)
            with pytest.raises(ValueError, match="new"):
                gen.decode(a)
        assert calls == []
        for state in (a, b):
            gen.release(state)
        other.release(foreign)


# ------------------------------------------------------------------ cache contents


def test_decode_appends_exactly_its_own_keys_and_values() -> None:
    """Each explicit token lands at its absolute position in all six layers, as the packed
    K/V the step passed to attention, and no earlier slot changes."""
    model = loaded()
    prompt = token_stream("append_prompt", 30)
    tokens = token_stream("append_tokens", 4)
    with BeamSmallModelGenerator(model) as gen:
        state = gen.create_request(prompt, max_new_tokens=4, sample=False)
        gen.prefill(state)
        for token in tokens:
            position = state.consumed
            before = written_bytes(gen.manager, state.request_id, position)
            with counting_attention_calls() as calls:
                gen.decode(state, token)
            assert [layer for layer, _, _ in calls] == list(range(6))
            for layer, rows, qkv in calls:
                assert rows == 1
                k, v = read_slot(gen.manager, state.request_id, layer, position)
                assert bitwise_equal(k, qkv[0, 48 * 128 : 56 * 128].view(8, 128))
                assert bitwise_equal(v, qkv[0, 56 * 128 :].view(8, 128))
            assert torch.equal(written_bytes(gen.manager, state.request_id, position), before)
            assert state.consumed == position + 1
            record = state.ledger[-1]
            assert (
                record["positions"] == [position, position] and record["input"] == "teacher_forced"
            )
            assert record["capacity"] >= position + 1 and record["history"] == position + 1
        gen.release(state)


def test_prefix_reuse_claims_whole_pages_and_computes_the_rest() -> None:
    """B claims A's two common pages and computes 17 rows; the claimed slots are A's bytes and
    equal what a reuse-disabled control computes; a different first token misses."""
    model = loaded()
    common = token_stream("reuse_common", 64)
    a_prompt = common + token_stream("reuse_suffix_a", 17)
    b_prompt = common + token_stream("reuse_suffix_b", 17)
    miss = [(common[0] + VOCAB // 2) % VOCAB] + common[1:] + token_stream("reuse_suffix_b", 17)
    with BeamSmallModelGenerator(model) as gen:
        a = gen.create_request(a_prompt, max_new_tokens=1, sample=False)
        gen.prefill(a)
        b = gen.create_request(b_prompt, max_new_tokens=1, sample=False)
        with counting_attention_calls() as calls:
            gen.prefill(b)
        assert b.reused == 64 and b.computed_prompt_tokens == 17
        assert [rows for _, rows, _ in calls] == [17] * 6
        assert list(gen.manager.kv_cache_map[b.request_id].committed_tokens[:64]) == common
        claimed = written_bytes(gen.manager, b.request_id, 64)
        assert torch.equal(claimed, written_bytes(gen.manager, a.request_id, 64))
        # Layers 0-3 of the common tokens depend on the dense blocks only, whose arithmetic per
        # token does not depend on the rest of the prompt: a recomputation reproduces them bitwise.
        dense = claimed[: 4 * 64]
        gen.release(b)
        gen.release(a)
        m = gen.create_request(miss, max_new_tokens=1, sample=False)
        gen.prefill(m)
        assert m.reused == 0 and m.computed_prompt_tokens == 81
        gen.release(m)
    with BeamSmallModelGenerator(model) as gen:
        control = gen.create_request(b_prompt, max_new_tokens=1, sample=False, reuse=False)
        gen.prefill(control)
        assert control.reused == 0 and control.computed_prompt_tokens == 81
        assert torch.equal(written_bytes(gen.manager, control.request_id, 64)[: 4 * 64], dense)
        gen.release(control)


def test_two_live_requests_stay_isolated() -> None:
    """A and B share a reused prefix and advance alternately; neither step changes any slot the
    other has written, shared or private."""
    model = loaded()
    common = token_stream("reuse_common", 64)
    with BeamSmallModelGenerator(model) as gen:
        a = gen.create_request(common + token_stream("iso_a", 17), max_new_tokens=3, sample=False)
        b = gen.create_request(common + token_stream("iso_b", 17), max_new_tokens=3, sample=False)
        gen.prefill(a)
        gen.prefill(b)
        assert b.reused == 64
        for token_a, token_b in zip(token_stream("iso_ta", 3), token_stream("iso_tb", 3)):
            for state, other, token in ((a, b, token_a), (b, a, token_b)):
                before = written_bytes(gen.manager, other.request_id, other.consumed)
                gen.decode(state, token)
                assert torch.equal(
                    written_bytes(gen.manager, other.request_id, other.consumed), before
                )
        gen.release(a)
        gen.release(b)


def test_weights_cannot_change_under_an_open_cache() -> None:
    """load_weights is refused while a generator is open. A parameter replaced by another tensor
    (``load_state_dict(assign=True)``) or changed in place is detected before the next step: a
    prefill that would reuse a released request's prefix claims nothing and calls no attention,
    and a live request's decode allocates nothing. Restoring the bound parameter makes the cache
    usable again; after close a new generator starts from an empty cache."""
    model = loaded()
    tensors = model.checkpoint_tensors()
    original = model.layers[0].self_attn_k_proj
    assert tensors["model.layers.0.self_attn.k_proj.weight"] is original
    prompt = token_stream("reuse_common", 64) + token_stream("reuse_suffix_a", 17)

    def assign_k_proj(tensor: torch.Tensor) -> None:
        model.load_state_dict({"layers.0.self_attn_k_proj": tensor}, strict=False, assign=True)

    with BeamSmallModelGenerator(model) as gen:
        released = gen.create_request(prompt, max_new_tokens=1, sample=False)
        gen.prefill(released)
        gen.release(released)
        live = gen.create_request(
            token_stream("generation_prompt", 17), max_new_tokens=2, sample=False
        )
        gen.prefill(live)
        versions = {n: t._version for n, t in tensors.items()}
        with pytest.raises(RuntimeError, match="open generator"):
            model.load_weights(tensors)
        assert {n: t._version for n, t in tensors.items()} == versions
        state = gen.create_request(prompt, max_new_tokens=1, sample=False)
        capacity = gen.manager.kv_cache_map[live.request_id].capacity
        try:
            assign_k_proj(torch.zeros_like(original))
            assert model.layers[0].self_attn_k_proj is not original
            assert original._version == versions["model.layers.0.self_attn.k_proj.weight"]
            with counting_attention_calls() as calls:
                with pytest.raises(ValueError, match="changed"):
                    gen.prefill(state)
                with pytest.raises(ValueError, match="changed"):
                    gen.decode(live, 7)
            assert calls == [] and state.request_id not in gen.manager.kv_cache_map
            assert state.lifecycle is Lifecycle.NEW and state.reused == 0
            assert live.lifecycle is Lifecycle.PREFILLED and live.consumed == 17
            assert gen.manager.kv_cache_map[live.request_id].capacity == capacity
        finally:
            assign_k_proj(original)
        assert model.layers[0].self_attn_k_proj is original
        gen.prefill(state)
        assert state.reused == 64
        gen.decode(live, 7)
        with torch.no_grad():
            original.mul_(1.0)
        with counting_attention_calls() as calls, pytest.raises(ValueError, match="changed"):
            gen.decode(live, 7)
        assert calls == []
    model.load_weights(tensors)
    with BeamSmallModelGenerator(model) as gen:
        state = gen.create_request(prompt, max_new_tokens=1, sample=False)
        gen.prefill(state)
        assert state.reused == 0
        gen.release(state)


# ------------------------------------------------------------------ guards before dispatch


@contextlib.contextmanager
def step_metadata(gen, state, num_tokens: int, past: int, context: bool, **overrides):
    md = TrtllmAttentionMetadata(
        max_num_requests=1,
        max_num_tokens=model_module.MAX_TOKENS,
        kv_cache_manager=gen.manager,
        num_contexts=1 if context else 0,
        kv_cache_params=KVCacheParams(use_cache=True, num_cached_tokens_per_seq=[past]),
        seq_lens=torch.tensor([num_tokens], dtype=torch.int),
        request_ids=[state.request_id],
        prompt_lens=[num_tokens if context else state.prompt_len],
        runtime_features=overrides.pop(
            "runtime_features", AttentionRuntimeFeatures(cache_reuse=True)
        ),
    )
    md.prepare()
    yield md


_BAD_STEPS = [
    "positions_restart",
    "two_token_generation",
    "cached_context_without_paged_fmha",
    "unknown_request",
    "beyond_capacity",
    "block_offset_corrupted",
    "rewound_into_history",
    "gap_after_history",
]


@pytest.mark.parametrize("case", _BAD_STEPS)
def test_model_rejects_inconsistent_steps_before_dispatch(case) -> None:
    """A step whose positions, metadata or block table do not describe the request's own cache
    is refused with ValueError before the embedding or any attention call, and no written slot
    changes. That includes a generation step that does not begin where the cache's history ends
    -- one token back, into the cached history, or one token past it -- although the pages and
    capacity it addresses are valid."""
    model = loaded()
    prompt = token_stream("guard_prompt", 40)
    with BeamSmallModelGenerator(model) as gen:
        state = gen.create_request(prompt, max_new_tokens=2, sample=False)
        gen.prefill(state)
        request = state._request
        request.add_new_token(7, 0)
        # Capacity for the next token: none beyond the history for "beyond_capacity", one more
        # token than usual for "gap_after_history", so that only the history check refuses it.
        for _ in range({"beyond_capacity": 0, "gap_after_history": 2}.get(case, 1)):
            assert gen.manager.try_allocate_generation(request)
        past, ids = state.consumed, torch.tensor([[7]], device="cuda")
        assert gen.manager.kv_cache_map[state.request_id].history_length == past == 40
        positions = torch.tensor([[past]], device="cuda")
        md_args = dict(num_tokens=1, past=past, context=False)
        if case == "positions_restart":
            positions = torch.tensor([[0]], device="cuda")
        elif case == "two_token_generation":
            ids, positions = (
                torch.tensor([[7, 8]], device="cuda"),
                torch.tensor([[past, past + 1]], device="cuda"),
            )
            md_args = dict(num_tokens=2, past=past - 1, context=False)
        elif case == "cached_context_without_paged_fmha":
            md_args = dict(
                num_tokens=1, past=past, context=True, runtime_features=AttentionRuntimeFeatures()
            )
        elif case in ("rewound_into_history", "gap_after_history"):
            start = past - 1 if case == "rewound_into_history" else past + 1
            md_args = dict(num_tokens=1, past=start, context=False)
            positions = torch.tensor([[start]], device="cuda")
        written = written_bytes(gen.manager, state.request_id, past)
        with step_metadata(gen, state, **md_args) as md:
            if case == "block_offset_corrupted":
                md.kv_cache_block_offsets[0, 0, 0, 0] += 2
            elif case == "unknown_request":
                md.request_ids = [state.request_id + 1000]
            with counting_attention_calls() as calls, pytest.raises(ValueError):
                model(ids, positions, md)
            assert calls == []
        assert torch.equal(written_bytes(gen.manager, state.request_id, past), written)
        gen.release(state)


def test_model_rejects_a_context_step_rewound_over_a_reused_prefix() -> None:
    """B claims A's two committed pages and holds 64 tokens. A context step for B that starts at
    token 32 -- inside those shared pages, on valid pages with capacity and positions that match
    it -- is refused before any attention call, and the shared pages keep A's bytes."""
    model = loaded()
    common = token_stream("reuse_common", 64)
    a_prompt = common + token_stream("reuse_suffix_a", 17)
    b_prompt = common + token_stream("reuse_suffix_b", 17)
    with BeamSmallModelGenerator(model) as gen:
        a = gen.create_request(a_prompt, max_new_tokens=1, sample=False)
        gen.prefill(a)
        b = gen.create_request(b_prompt, max_new_tokens=1, sample=False)
        # The steps prefill takes before its model call: claim the prefix, size the cache.
        request = b._request
        batch = ScheduledRequests()
        batch.context_requests_last_chunk = [request]
        gen.seq_slots.prepare_resources(batch)
        assert gen.manager.prepare_context(request) and request.context_current_position == 64
        assert gen.manager.resize_context(request, request.context_remaining_length)
        cache = gen.manager.kv_cache_map[b.request_id]
        assert cache.history_length == 64 and cache.capacity == 81
        shared = written_bytes(gen.manager, a.request_id, 81)
        ids = torch.tensor([b_prompt[32:]], device="cuda")
        positions = torch.arange(32, 81, device="cuda").view(1, -1)
        with step_metadata(gen, b, num_tokens=49, past=32, context=True) as md:
            with counting_attention_calls() as calls, pytest.raises(ValueError, match="holds 64"):
                model(ids, positions, md)
        assert calls == []
        assert torch.equal(written_bytes(gen.manager, a.request_id, 81), shared)
        gen.release(b)
        gen.release(a)


# ------------------------------------------------------------------ sampler and window


def test_torch_sampler_greedy_choice_on_identical_logits() -> None:
    """The generator's sampler step picks the lowest index among exact maxima, on a context row
    and on a generation row."""
    sampler = TorchSampler(generation.SAMPLER_ARGS)
    slots = SeqSlotManager(generation.GENERATOR_CACHE["max_batch_size"])
    request = LlmRequest(
        request_id=1,
        max_new_tokens=4,
        input_tokens=[1, 2, 3],
        sampling_config=generation.greedy_sampling_config(),
        is_streaming=False,
    )
    batch = ScheduledRequests()
    batch.context_requests_last_chunk = [request]
    slots.prepare_resources(batch)
    request.context_chunk_size = 3
    rows = []
    tied = torch.zeros(VOCAB, device="cuda")
    tied[[90000, 1234, 151935]] = 2.5
    rows.append((tied, 1234))
    distinct = torch.linspace(-30.0, 30.0, VOCAB, device="cuda").flip(0)
    distinct[77] = 30.0
    rows.append((distinct, 0))
    for step, (logits, expected) in enumerate(rows):
        generation.sample_step(sampler, request, logits, is_context=step == 0)
        assert request.get_last_tokens(0) == expected == int(torch.argmax(logits))
        if step == 0:
            assert request.state == LlmRequestState.GENERATION_IN_PROGRESS
    slots.free_resources(request)


def test_sliding_window_excludes_exactly_the_aged_key() -> None:
    """Controlled keys through the generator's cache manager and the catalog attention op: at
    query position 512 the sliding layer's output ignores key 0 bit for bit and depends on key
    1, while the full layer depends on key 0."""
    mgr = generation.build_cache_manager()
    g = torch.Generator(device="cuda").manual_seed(512)
    q = torch.randn(1, 48 * 128, generator=g, device="cuda").to(torch.bfloat16)
    kv = torch.randn(513, 2 * 8 * 128, generator=g, device="cuda").to(torch.bfloat16)
    request_ids = iter(range(100, 200))

    def attend(layer: int, kv_rows: torch.Tensor) -> torch.Tensor:
        rid = next(request_ids)
        tokens = [(rid * 7919 + j * 104729) % VOCAB for j in range(512)]
        request = LlmRequest(
            request_id=rid,
            max_new_tokens=1,
            input_tokens=tokens,
            sampling_config=generation.greedy_sampling_config(),
            is_streaming=False,
        )
        try:
            assert mgr.prepare_context_cache(request, reuse_limit=0) == 0
            assert mgr.prepare_context(request) and mgr.resize_context(request, 512)
            window = 512 if layer < 5 else mgr.max_seq_len
            outputs = []
            zero_q = torch.zeros(512, 48 * 128, dtype=torch.bfloat16, device="cuda")
            for start, count, context, rows in (
                (0, 512, True, torch.cat((zero_q, kv_rows[:512]), 1)),
                (512, 1, False, torch.cat((q, kv_rows[512:513]), 1)),
            ):
                if not context:
                    request.move_to_next_context_chunk()
                    request.state = LlmRequestState.GENERATION_IN_PROGRESS
                    batch = ScheduledRequests()
                    batch.context_requests_last_chunk = [request]
                    mgr.update_context_resources(batch)
                    request.add_new_token(5, 0)
                    assert mgr.try_allocate_generation(request)
                md = TrtllmAttentionMetadata(
                    max_num_requests=1,
                    max_num_tokens=1024,
                    kv_cache_manager=mgr,
                    num_contexts=1 if context else 0,
                    kv_cache_params=KVCacheParams(
                        use_cache=True, num_cached_tokens_per_seq=[start]
                    ),
                    seq_lens=torch.tensor([count], dtype=torch.int),
                    request_ids=[rid],
                    prompt_lens=[count if context else 512],
                    runtime_features=AttentionRuntimeFeatures(cache_reuse=True),
                )
                md.prepare()
                out = torch.empty(count, 48 * 128, dtype=torch.bfloat16, device="cuda")
                thop_attention(
                    q=rows.contiguous(),
                    output=out,
                    num_heads=48,
                    num_kv_heads=8,
                    head_size=128,
                    attention_window_size=window,
                    **_attention.step_args(md, layer),
                    **_attention.CALL_CONSTANTS,
                )
                outputs.append(out)
            torch.cuda.synchronize()
            return outputs[-1].clone()
        finally:
            mgr.free_resources(request)

    def perturbed(position: int) -> torch.Tensor:
        rows = kv.clone()
        rows[position, : 8 * 128] = -rows[position, : 8 * 128]
        rows[position, 8 * 128 :] = rows[position, 8 * 128 :] * 4.0 + 1.0
        return rows

    try:
        sliding, full = attend(0, kv), attend(5, kv)
        assert bitwise_equal(attend(0, perturbed(0)), sliding)
        assert not bitwise_equal(attend(0, perturbed(1)), sliding)
        assert not bitwise_equal(attend(5, perturbed(0)), full)
    finally:
        mgr.shutdown()
