# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Standalone MLA DSpark drafter context KV in the target's KV cache manager V2.

The context server builds the drafter's context KV from the target hidden
states during prefill and never drafts; the disaggregated transfer carries the
pages and the draft history; the generation server drafts from them.
"""

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.disaggregation.native.auxiliary import (
    _decode_draft_history,
    _encode_draft_history,
)
from tensorrt_llm._torch.models.modeling_dspark import MLADSparkForCausalLM
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest, SamplingConfig
from tensorrt_llm._torch.pyexecutor.resource_manager import (
    CacheTypeCpp,
    DataType,
    ResourceManagerType,
)
from tensorrt_llm._torch.speculative.dspark import DSparkWorker
from tensorrt_llm.llmapi.llm_args import DSparkDecodingConfig, KvCacheConfig
from tensorrt_llm.mapping import Mapping

needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")

_DRAFT_CONTEXT_LIMIT_ENV = "TLLM_DSPARK_DRAFT_CONTEXT_LIMIT"
_DRAFT_CONTEXT_WINDOW_ENV = "TLLM_DSPARK_DRAFT_CONTEXT_WINDOW"
_DRAFT_CONTEXT_SINK_ENV = "TLLM_DSPARK_DRAFT_CONTEXT_SINK"


@pytest.fixture(autouse=True)
def _no_inherited_draft_context_limit(monkeypatch):
    monkeypatch.delenv(_DRAFT_CONTEXT_LIMIT_ENV, raising=False)
    monkeypatch.delenv(_DRAFT_CONTEXT_WINDOW_ENV, raising=False)
    monkeypatch.delenv(_DRAFT_CONTEXT_SINK_ENV, raising=False)


def _sm100f() -> bool:
    if not torch.cuda.is_available():
        return False
    from tensorrt_llm._utils import is_sm_100f

    return is_sm_100f()


needs_sm100f = pytest.mark.skipif(not _sm100f(), reason="CuTe DSL MLA decode needs SM100f")


HIDDEN = 64
INTERMEDIATE = 32
HEADS = 4
Q_LORA = 16
KV_LORA = 512
NOPE = 128
ROPE = 64
V_DIM = 128
LAYERS = 2
BLOCK = 3
LATENT = KV_LORA + ROPE
VOCAB = 512
RANK = 16
NUM_CAPTURE = 2
PAGE = 32
MAX_SEQ_LEN = 256


def _tiny_mla_config():
    from transformers import PretrainedConfig

    cfg = PretrainedConfig(
        architectures=["K3DSparkModel"],
        model_type="k3_dspark",
        hidden_size=HIDDEN,
        intermediate_size=INTERMEDIATE,
        num_hidden_layers=LAYERS,
        num_attention_heads=HEADS,
        num_key_value_heads=HEADS,
        q_lora_rank=Q_LORA,
        kv_lora_rank=KV_LORA,
        qk_nope_head_dim=NOPE,
        qk_rope_head_dim=ROPE,
        v_head_dim=V_DIM,
        vocab_size=VOCAB,
        rms_norm_eps=1e-6,
        max_position_embeddings=MAX_SEQ_LEN,
        rope_theta=50000.0,
        block_size=BLOCK,
        mask_token_id=VOCAB - 3,
        target_layer_ids=[0, 1],
        markov_rank=RANK,
        markov_head_type="vanilla",
        tie_word_embeddings=False,
        rope_parameters={
            "rope_type": "yarn",
            "factor": 4.0,
            "original_max_position_embeddings": 64,
            "rope_theta": 50000.0,
            "beta_fast": 32,
            "beta_slow": 1,
            "mscale": 1.0,
            "mscale_all_dim": 1.0,
        },
    )
    cfg.dflash_config = {"mask_token_id": VOCAB - 3, "target_layer_ids": [0, 1]}
    cfg.torch_dtype = torch.bfloat16
    return cfg


def _tiny_mla_weights(seed=11):
    g = torch.Generator().manual_seed(seed)

    def rnd(*shape):
        return (torch.randn(*shape, generator=g) * 0.05).to(torch.bfloat16)

    w = {
        "context_proj.weight": rnd(HIDDEN, HIDDEN * NUM_CAPTURE),
        "context_norm.weight": rnd(HIDDEN).abs() + 1.0,
        "final_norm.weight": rnd(HIDDEN).abs() + 1.0,
        "embed_tokens.weight": rnd(VOCAB, HIDDEN),
        "markov_head.markov_w1.weight": rnd(VOCAB, RANK),
        "markov_head.markov_w2.weight": rnd(VOCAB, RANK),
    }
    for i in range(LAYERS):
        p = f"layers.{i}."
        w[p + "input_layernorm.weight"] = rnd(HIDDEN).abs() + 1.0
        w[p + "post_attention_layernorm.weight"] = rnd(HIDDEN).abs() + 1.0
        w[p + "self_attn.q_a_proj.weight"] = rnd(Q_LORA, HIDDEN)
        w[p + "self_attn.q_a_layernorm.weight"] = rnd(Q_LORA).abs() + 1.0
        w[p + "self_attn.q_b_proj.weight"] = rnd(HEADS * (NOPE + ROPE), Q_LORA)
        w[p + "self_attn.kv_a_proj_with_mqa.weight"] = rnd(LATENT, HIDDEN)
        w[p + "self_attn.kv_a_layernorm.weight"] = rnd(KV_LORA).abs() + 1.0
        w[p + "self_attn.kv_b_proj.weight"] = rnd(HEADS * (NOPE + V_DIM), KV_LORA)
        w[p + "self_attn.o_proj.weight"] = rnd(HIDDEN, HEADS * V_DIM)
        w[p + "mlp.gate_proj.weight"] = rnd(INTERMEDIATE, HIDDEN)
        w[p + "mlp.up_proj.weight"] = rnd(INTERMEDIATE, HIDDEN)
        w[p + "mlp.down_proj.weight"] = rnd(HIDDEN, INTERMEDIATE)
    return w


def _build_drafter(weights, *, backend="VANILLA", context_kv_only=False):
    from tensorrt_llm._torch.model_config import ModelConfig

    model_config = ModelConfig(pretrained_config=_tiny_mla_config(), attn_backend="TRTLLM")
    drafter = MLADSparkForCausalLM(
        model_config, dflash_attention_backend=backend, context_kv_only=context_kv_only
    ).to("cuda")
    drafter.load_weights(dict(weights))
    return drafter


def _context_latents(drafter, captured, positions):
    """What the context server stores per token: [N, L, 1, latent]."""
    projected = drafter.project_target_hidden(captured)
    latent, v = drafter.precompute_context_kv(projected, positions)
    assert v is None
    return latent


# ---------------------------------------------------------------------------
# Context-KV-only drafter (disaggregated context server).
# ---------------------------------------------------------------------------


@needs_cuda
def test_context_kv_only_drafter_keeps_only_the_context_kv_path():
    weights = _tiny_mla_weights()
    full = _build_drafter(weights)
    ctx = _build_drafter(weights, context_kv_only=True)

    backbone = ctx.draft_model_full.model
    assert backbone.embed_tokens is None and backbone.norm is None
    assert ctx.lm_head is None and ctx.markov_w1 is None
    for layer in backbone.layers:
        attn = layer.self_attn
        assert layer.mlp is None
        assert attn.q_b_proj is None and attn.kv_b_proj is None and attn.o_proj is None
        assert attn.kv_a_proj_with_mqa is not None and attn.kv_a_layernorm is not None

    def param_bytes(module):
        return sum(p.numel() * p.element_size() for p in module.parameters())

    kept = (HIDDEN * HIDDEN * NUM_CAPTURE + HIDDEN) * 2 + LAYERS * (LATENT * HIDDEN + KV_LORA) * 2
    assert param_bytes(ctx) == kept
    assert param_bytes(ctx) < param_bytes(full) / 4

    torch.manual_seed(0)
    captured = torch.randn(37, HIDDEN * NUM_CAPTURE, device="cuda", dtype=torch.bfloat16)
    positions = torch.arange(5, 42, device="cuda")
    assert torch.equal(
        _context_latents(ctx, captured, positions), _context_latents(full, captured, positions)
    )


def test_context_kv_only_is_rejected_for_a_drafter_that_drafts_elsewhere():
    from tensorrt_llm._torch.models.modeling_dspark import _build_dspark_draft

    for embedded, kv_lora_rank in ((False, None), (True, KV_LORA)):
        spec = SimpleNamespace(
            context_kv_only=True,
            attention_backend="AUTO",
            draft_is_embedded_in_target=embedded,
        )
        model_config = SimpleNamespace(spec_config=spec)
        pretrained = SimpleNamespace(kv_lora_rank=kv_lora_rank, qk_rope_head_dim=ROPE)
        draft_config = SimpleNamespace(pretrained_config=pretrained)
        with pytest.raises(ValueError, match="standalone MLA DSpark"):
            _build_dspark_draft(model_config, draft_config, None, None)


# ---------------------------------------------------------------------------
# CTX builds the drafter context and does not draft.
# ---------------------------------------------------------------------------


def _unified_manager(max_tokens=1024, max_seq_len=MAX_SEQ_LEN, enable_block_reuse=False):
    from tensorrt_llm._torch.pyexecutor.kv_cache_manager_v2 import KVCacheManagerV2

    return KVCacheManagerV2(
        KvCacheConfig(max_tokens=max_tokens, enable_block_reuse=enable_block_reuse),
        CacheTypeCpp.SELF,
        num_layers=1,
        num_kv_heads=1,
        head_dim=64,
        tokens_per_block=PAGE,
        max_seq_len=max_seq_len,
        max_batch_size=4,
        mapping=Mapping(),
        dtype=DataType.HALF,
        is_disagg=True,
        draft_num_layers=LAYERS,
        draft_num_kv_heads=1,
        draft_head_dim=LATENT,
        draft_dtype=DataType.BF16,
        draft_kv_factor=1,
        draft_total_num_kv_heads=1,
        draft_scratch_tokens=BLOCK + 1,
        draft_attention_backend="VANILLA",
    )


def _request(request_id, num_tokens):
    return LlmRequest(
        request_id=request_id,
        max_new_tokens=8,
        input_tokens=list(range(10, 10 + num_tokens)),
        sampling_config=SamplingConfig(),
        is_streaming=False,
    )


def _worker(context_kv_only=False):
    config = DSparkDecodingConfig(
        max_draft_len=BLOCK - 1,
        speculative_model="unused",
        attention_backend="VANILLA",
        context_kv_only=context_kv_only,
    )
    return DSparkWorker(config, Mapping())


class _Captured:
    """The slice of DFlashSpecMetadata the drafter context reads."""

    def __init__(self, request_ids, hidden_states, runtime_draft_len=BLOCK - 1):
        self.request_ids = request_ids
        self.batch_indices_cuda = torch.arange(4, dtype=torch.int32, device="cuda")
        self.max_num_requests = 4
        self.hidden_size = HIDDEN
        self.runtime_draft_len = runtime_draft_len
        self._hidden = hidden_states

    def get_hidden_states(self, num_tokens):
        return self._hidden[:num_tokens]


def _resources(manager):
    return SimpleNamespace(
        get_resource_manager=lambda kind: (
            manager if kind == ResourceManagerType.KV_CACHE_MANAGER else None
        )
    )


def _context_step(
    worker, drafter, manager, request, captured, max_seq_len=MAX_SEQ_LEN, before_store=None
):
    """One context forward of the whole prompt as the executor runs it.

    ``max_seq_len`` is the sequence bound the executor reports for this manager;
    ``before_store`` runs after the drafter cache is prepared and before the
    prompt's context is stored.
    """
    prompt_len = captured.shape[0]
    assert manager.prepare_context(request)
    assert manager.resize_context(request, prompt_len)
    spec = _Captured([request.py_request_id], captured)
    attn = SimpleNamespace(
        num_seqs=1,
        num_contexts=1,
        num_ctx_tokens=prompt_len,
        _seq_lens=torch.tensor([prompt_len]),
        max_seq_len=max_seq_len,
        kv_cache_params=SimpleNamespace(num_cached_tokens_per_seq=[0]),
    )
    worker.prepare_managed_draft_cache(drafter, spec, attn, _resources(manager))
    if before_store is not None:
        before_store()
    worker._store_prefill_context(
        drafter, spec, attn, torch.arange(prompt_len, device="cuda"), prompt_len
    )
    update = worker.snapshot_managed_draft_history()
    torch.cuda.synchronize()
    update.publish()


def _batch_step(worker, drafter, manager, entries, max_seq_len=MAX_SEQ_LEN, prepared=False):
    """One context forward over several requests, as the executor runs it.

    ``entries`` are ``(request, hidden states of the tokens it computes, position of the first
    of them)``: the position is nonzero when an earlier chunk ran or the target reused a prefix.
    ``prepared`` says the caller has already run the manager's context preparation and resize.
    """
    request_ids = [request.py_request_id for request, _, _ in entries]
    lengths = [rows.shape[0] for _, rows, _ in entries]
    starts = [first for _, _, first in entries]
    for request, rows, _ in entries:
        if not prepared:
            assert manager.prepare_context(request)
            assert manager.resize_context(request, rows.shape[0])
    spec = _Captured(request_ids, torch.cat([rows for _, rows, _ in entries]))
    attn = SimpleNamespace(
        num_seqs=len(entries),
        num_contexts=len(entries),
        num_ctx_tokens=sum(lengths),
        _seq_lens=torch.tensor(lengths),
        max_seq_len=max_seq_len,
        kv_cache_params=SimpleNamespace(num_cached_tokens_per_seq=starts),
    )
    positions = torch.cat(
        [torch.arange(first, first + n, device="cuda") for first, n in zip(starts, lengths)]
    )
    worker.prepare_managed_draft_cache(drafter, spec, attn, _resources(manager))
    worker._store_prefill_context(drafter, spec, attn, positions, sum(lengths))
    update = worker.snapshot_managed_draft_history()
    torch.cuda.synchronize()
    update.publish()


def _draft_pages(manager, request_id):
    """Per-layer [tokens, latent] contents of a request's draft pages."""
    table = manager.get_draft_block_table([request_id])[0].long()
    num_blocks = manager.get_draft_num_blocks(request_id)
    pages = table[:num_blocks].cuda()
    return [
        manager.get_draft_buffers(layer, kv_layout="HND")[pages, 0, 0].reshape(-1, LATENT)
        for layer in range(LAYERS)
    ]


@needs_cuda
def test_context_server_builds_the_draft_context_and_does_not_draft(monkeypatch):
    weights = _tiny_mla_weights()
    drafter = _build_drafter(weights, context_kv_only=True)
    reference = _build_drafter(weights)
    manager = _unified_manager()
    worker = _worker(context_kv_only=True)
    worker.set_draft_model(drafter)
    try:
        prompt_len = 45
        torch.manual_seed(1)
        captured = torch.randn(prompt_len, HIDDEN * NUM_CAPTURE, device="cuda").bfloat16()
        request = _request(101, prompt_len)
        _context_step(worker, drafter, manager, request, captured)

        history = manager.get_draft_history(101)
        assert (history.valid_length, history.position) == (prompt_len, prompt_len)
        expected = _context_latents(reference, captured, torch.arange(prompt_len, device="cuda"))
        stored = _draft_pages(manager, 101)
        for layer in range(LAYERS):
            assert stored[layer][:prompt_len].abs().sum() > 0
            assert torch.equal(stored[layer][:prompt_len], expected[:, layer, 0])

        # A generation row reaches a context server only as a dummy.
        manager._draft_dummy_request_ids.add(66)
        worker._check_no_real_generation(SimpleNamespace(request_ids=[101, 66]), 1, 2)
        with pytest.raises(RuntimeError, match="context_kv_only"):
            worker._check_no_real_generation(SimpleNamespace(request_ids=[101, 77]), 1, 2)
    finally:
        manager.shutdown()


@needs_cuda
def test_context_server_forward_never_runs_the_drafter(monkeypatch):
    """The worker's forward on a context batch stores the drafter context and
    emits placeholder draft tokens without touching the block decode."""
    drafter = _build_drafter(_tiny_mla_weights(), context_kv_only=True)
    manager = _unified_manager()
    worker = _worker(context_kv_only=True)
    worker.set_draft_model(drafter)

    def never(*args, **kwargs):
        raise AssertionError("a context-KV-only drafter must not draft")

    monkeypatch.setattr(drafter, "dflash_forward", never)
    monkeypatch.setattr(worker, "prepare_1st_drafter_inputs", never)
    for name in (
        "_prepare_attn_metadata_for_dflash",
        "_prepare_kv_for_draft_forward",
        "_restore_attn_metadata_from_spec_dec",
        "_apply_kv_rewind_after_draft",
    ):
        monkeypatch.setattr(worker, name, lambda *args, **kwargs: None)
    width = BLOCK
    accepted = torch.full((1, width), 4, dtype=torch.int32, device="cuda")
    num_accepted = torch.ones(1, dtype=torch.int32, device="cuda")
    monkeypatch.setattr(
        worker, "sample_and_accept_draft_tokens", lambda *args: (accepted, num_accepted)
    )
    monkeypatch.setattr(worker, "_prepare_next_new_tokens", lambda accepted, drafts, *args: drafts)
    try:
        prompt_len = 40
        request = _request(21, prompt_len)
        assert manager.prepare_context(request)
        assert manager.resize_context(request, prompt_len)
        captured = torch.randn(prompt_len, HIDDEN * NUM_CAPTURE, device="cuda").bfloat16()
        spec = _Captured([21], captured)
        spec.is_cuda_graph = False
        attn = SimpleNamespace(
            num_seqs=1,
            num_contexts=1,
            num_ctx_tokens=prompt_len,
            _seq_lens=torch.tensor([prompt_len]),
            max_seq_len=MAX_SEQ_LEN,
            kv_cache_params=SimpleNamespace(num_cached_tokens_per_seq=[0]),
        )
        resources = _resources(manager)
        worker.prepare_managed_draft_cache(drafter, spec, attn, resources)
        out = worker._forward_impl(
            torch.zeros(prompt_len, dtype=torch.long, device="cuda"),
            torch.arange(prompt_len, device="cuda").unsqueeze(0),
            torch.zeros(prompt_len, HIDDEN, device="cuda"),
            torch.zeros(1, VOCAB, device="cuda"),
            attn,
            spec,
            drafter,
            resources,
        )
        assert out["next_draft_tokens"].shape == (1, BLOCK - 1)
        assert not out["next_draft_tokens"].any()
        assert int(worker._ctx_len[worker._req_to_slot[21]]) == prompt_len
    finally:
        manager.shutdown()


# ---------------------------------------------------------------------------
# The position table follows the sequence bound the cache manager is served at.
# ---------------------------------------------------------------------------


@needs_cuda
def test_position_table_follows_a_ceiling_published_after_it_was_built():
    """KV cache estimation builds the table at its probe's bound; serving needs the served one."""
    weights = _tiny_mla_weights()
    drafter = _build_drafter(weights, context_kv_only=True)
    device = torch.device("cuda")

    drafter._runtime_position_ceiling = 70
    probe = drafter._mla_freqs_cis(device)
    drafter._runtime_position_ceiling = 200
    served = drafter._mla_freqs_cis(device)

    assert (probe.shape[0], served.shape[0]) == (70, 200)
    # A position's phase does not depend on the length of the table it is read from.
    assert torch.equal(served[:70], probe)
    assert drafter._mla_freqs_cis(device) is served

    # Positions the probe's table could not encode.
    torch.manual_seed(3)
    captured = torch.randn(40, HIDDEN * NUM_CAPTURE, device="cuda").bfloat16()
    positions = torch.arange(150, 190, device="cuda")
    direct = _build_drafter(weights, context_kv_only=True)
    direct._runtime_position_ceiling = 200
    assert torch.equal(
        _context_latents(drafter, captured, positions),
        _context_latents(direct, captured, positions),
    )


@needs_cuda
def test_context_server_stores_prompts_longer_than_its_estimation_probe():
    """The worker rebinds to the served cache manager after estimation, whose sequence bound
    is larger than the probe's; the prompts it then stores reach positions the probe's table
    does not have."""
    weights = _tiny_mla_weights()
    drafter = _build_drafter(weights, context_kv_only=True)
    reference = _build_drafter(weights)
    worker = _worker(context_kv_only=True)
    worker.set_draft_model(drafter)
    probe_len, probe_bound, prompt_len = 40, 64, 150
    probe = _unified_manager(max_tokens=256, max_seq_len=probe_bound)
    served = _unified_manager()
    try:
        torch.manual_seed(4)
        warmup = torch.randn(probe_len, HIDDEN * NUM_CAPTURE, device="cuda").bfloat16()
        _context_step(
            worker, drafter, probe, _request(1, probe_len), warmup, max_seq_len=probe_bound
        )

        captured = torch.randn(prompt_len, HIDDEN * NUM_CAPTURE, device="cuda").bfloat16()

        def table_covers_the_prompt():
            assert drafter._mla_freqs_cis(torch.device("cuda")).shape[0] > prompt_len

        _context_step(
            worker,
            drafter,
            served,
            _request(2, prompt_len),
            captured,
            before_store=table_covers_the_prompt,
        )
        expected = _context_latents(reference, captured, torch.arange(prompt_len, device="cuda"))
        stored = _draft_pages(served, 2)
        for layer in range(LAYERS):
            assert torch.equal(stored[layer][:prompt_len], expected[:, layer, 0])
    finally:
        probe.shutdown()
        served.shutdown()


# ---------------------------------------------------------------------------
# What the context server stores when requests share a batch, arrive in chunks, or reuse a prefix.
# ---------------------------------------------------------------------------


def _assert_context_stored(manager, request_id, expected, length):
    stored = _draft_pages(manager, request_id)
    for layer in range(LAYERS):
        differs = (stored[layer][:length] != expected[:length, layer, 0]).any(dim=-1).nonzero()
        assert differs.numel() == 0, (
            f"layer {layer}: the drafter context of request {request_id} differs from a single "
            f"pass at tokens {differs.flatten()[:8].tolist()} of {length}"
        )


@needs_cuda
def test_manager_reads_the_draft_context_limit_when_it_is_built(monkeypatch):
    manager = _unified_manager()
    try:
        assert manager.draft_context_limit is None
    finally:
        manager.shutdown()
    monkeypatch.setenv(_DRAFT_CONTEXT_LIMIT_ENV, "64")
    manager = _unified_manager()
    try:
        assert manager.draft_context_limit == 64
    finally:
        manager.shutdown()


@needs_cuda
def test_context_server_stores_requests_that_share_a_batch():
    weights = _tiny_mla_weights()
    drafter = _build_drafter(weights, context_kv_only=True)
    reference = _build_drafter(weights)
    manager = _unified_manager()
    worker = _worker(context_kv_only=True)
    worker.set_draft_model(drafter)
    try:
        torch.manual_seed(5)
        lengths = (45, 70, 33)
        captured = [torch.randn(n, HIDDEN * NUM_CAPTURE, device="cuda").bfloat16() for n in lengths]
        requests = [_request(10 + i, n) for i, n in enumerate(lengths)]
        _batch_step(worker, drafter, manager, [(r, c, 0) for r, c in zip(requests, captured)])
        for request, hidden, n in zip(requests, captured, lengths):
            history = manager.get_draft_history(request.py_request_id)
            assert (history.valid_length, history.position) == (n, n)
            expected = _context_latents(reference, hidden, torch.arange(n, device="cuda"))
            _assert_context_stored(manager, request.py_request_id, expected, n)
    finally:
        manager.shutdown()


@needs_cuda
def test_context_server_stores_a_prompt_that_arrives_in_chunks():
    weights = _tiny_mla_weights()
    drafter = _build_drafter(weights, context_kv_only=True)
    reference = _build_drafter(weights)
    manager = _unified_manager()
    worker = _worker(context_kv_only=True)
    worker.set_draft_model(drafter)
    try:
        torch.manual_seed(6)
        prompt_len, chunks = 100, (37, 33, 30)
        captured = torch.randn(prompt_len, HIDDEN * NUM_CAPTURE, device="cuda").bfloat16()
        request = _request(20, prompt_len)
        start = 0
        for n in chunks:
            _batch_step(worker, drafter, manager, [(request, captured[start : start + n], start)])
            start += n
            request.context_current_position = start
        history = manager.get_draft_history(20)
        assert (history.valid_length, history.position) == (prompt_len, prompt_len)
        expected = _context_latents(reference, captured, torch.arange(prompt_len, device="cuda"))
        _assert_context_stored(manager, 20, expected, prompt_len)
    finally:
        manager.shutdown()


@needs_cuda
@pytest.mark.parametrize(
    "draft_context_limit", [None, "40"], ids=["no_limit", "limit_below_prompt"]
)
def test_context_server_appends_to_the_prefix_a_later_turn_reuses(monkeypatch, draft_context_limit):
    """Turn two matches the blocks turn one committed, so the target computes (and the drafter
    stores) only the tail; the context of the whole prompt must be what one pass would store.

    A draft context limit below both prompts changes nothing here: only a generation server acts
    on it."""
    if draft_context_limit is not None:
        monkeypatch.setenv(_DRAFT_CONTEXT_LIMIT_ENV, draft_context_limit)
    weights = _tiny_mla_weights()
    drafter = _build_drafter(weights, context_kv_only=True)
    reference = _build_drafter(weights)
    manager = _unified_manager(enable_block_reuse=True)
    worker = _worker(context_kv_only=True)
    worker.set_draft_model(drafter)
    first_len, second_len = 100, 170
    try:
        torch.manual_seed(7)
        captured = torch.randn(second_len, HIDDEN * NUM_CAPTURE, device="cuda").bfloat16()
        expected = _context_latents(reference, captured, torch.arange(second_len, device="cuda"))

        first = _request(30, first_len)
        _batch_step(worker, drafter, manager, [(first, captured[:first_len], 0)])
        first.context_current_position = first_len
        manager.update_context_resources(SimpleNamespace(context_requests=[first]))
        manager.free_resources(first)

        second = _request(31, second_len)
        assert manager.prepare_context(second)
        reused = second.context_current_position
        assert 0 < reused <= first_len
        history = manager.get_draft_history(31)
        assert (history.valid_length, history.position) == (reused, reused)
        assert manager.resize_context(second, second_len - reused)
        _batch_step(worker, drafter, manager, [(second, captured[reused:], reused)], prepared=True)
        history = manager.get_draft_history(31)
        assert (history.valid_length, history.position) == (second_len, second_len)
        _assert_context_stored(manager, 31, expected, second_len)
    finally:
        manager.shutdown()


@needs_cuda
def test_context_chunks_after_an_overflowed_one_stay_inside_the_position_table(monkeypatch):
    """A chunk past the sequence bound drops the request's drafter context; the chunks after it
    must be dropped as well, not stored from a length that restarted at zero."""
    drafter = _build_drafter(_tiny_mla_weights(), context_kv_only=True)
    manager = _unified_manager()
    worker = _worker(context_kv_only=True)
    worker.set_draft_model(drafter)
    bound, prompt_len, chunks = 64, 90, (40, 40, 10)
    positions_seen = []
    precompute = drafter.precompute_context_kv

    def recording(projected, positions):
        positions_seen.append(int(positions.max()))
        return precompute(projected, positions)

    monkeypatch.setattr(drafter, "precompute_context_kv", recording)
    try:
        torch.manual_seed(8)
        captured = torch.randn(prompt_len, HIDDEN * NUM_CAPTURE, device="cuda").bfloat16()
        request = _request(40, prompt_len)
        start = 0
        for n in chunks:
            _batch_step(
                worker,
                drafter,
                manager,
                [(request, captured[start : start + n], start)],
                max_seq_len=bound,
            )
            start += n
            request.context_current_position = start
        # Only the first chunk fits the bound, and every position stored is inside the table.
        assert positions_seen == [chunks[0] - 1]
        assert max(positions_seen) < drafter._runtime_position_ceiling
        history = manager.get_draft_history(40)
        assert (history.valid_length, history.position) == (0, prompt_len)
    finally:
        manager.shutdown()


@needs_cuda
@pytest.mark.parametrize("fused", [False, True], ids=["eager", "fused"])
def test_generation_server_drafts_from_the_transferred_context(monkeypatch, fused):
    """CTX builds the history, the transfer carries pages + history, GEN's draft
    equals a drafter that saw the same target hidden states locally."""
    monkeypatch.setenv("TLLM_SPEC_TAIL_FUSION", "1" if fused else "0")
    weights = _tiny_mla_weights()
    ctx_drafter = _build_drafter(weights, context_kv_only=True)
    gen_drafter = _build_drafter(weights)
    ctx_manager, gen_manager = _unified_manager(), _unified_manager()
    ctx_worker, gen_worker = _worker(context_kv_only=True), _worker()
    ctx_worker.set_draft_model(ctx_drafter)
    gen_worker.set_draft_model(gen_drafter)
    width = BLOCK  # K + 1 target tokens per generation step
    try:
        prompt_len = 70
        torch.manual_seed(2)
        captured = torch.randn(prompt_len, HIDDEN * NUM_CAPTURE, device="cuda").bfloat16()
        _context_step(ctx_worker, ctx_drafter, ctx_manager, _request(5, prompt_len), captured)

        # The transfer: prompt pages in page order, the history through the aux record.
        gen_request = _request(9, prompt_len)
        assert gen_manager.prepare_context(gen_request)
        assert gen_manager.resize_context(gen_request, prompt_len + 2 * width)
        meta = _decode_draft_history(_encode_draft_history(ctx_manager.export_draft_history(5)))
        gen_manager.restore_draft_history(9, meta)
        num_pages = -(-prompt_len // PAGE)
        src_table = ctx_manager.get_draft_block_table([5])[0][:num_pages].long().cuda()
        dst_table = gen_manager.get_draft_block_table([9])[0][:num_pages].long().cuda()
        for layer in range(LAYERS):
            gen_pool = gen_manager.get_draft_buffers(layer, kv_layout="HND")
            ctx_pool = ctx_manager.get_draft_buffers(layer, kv_layout="HND")
            gen_pool[dst_table] = ctx_pool[src_table]

        # One generation step on GEN.
        num_accepted = 2
        gen_captured = torch.randn(width, HIDDEN * NUM_CAPTURE, device="cuda").bfloat16()
        spec = _Captured([9], gen_captured)
        attn = SimpleNamespace(
            num_seqs=1,
            num_contexts=0,
            num_ctx_tokens=0,
            max_seq_len=MAX_SEQ_LEN,
            _seq_lens=torch.zeros(1, dtype=torch.int32),
            _seq_lens_cuda=torch.zeros(1, dtype=torch.int32, device="cuda"),
        )
        gen_worker.prepare_managed_draft_cache(gen_drafter, spec, attn, _resources(gen_manager))
        slot = gen_worker._req_to_slot[9]
        assert int(gen_worker._ctx_len[slot]) == prompt_len
        accepted = torch.randint(0, VOCAB - 4, (1, width), device="cuda", dtype=torch.int32)
        inputs = gen_worker.prepare_1st_drafter_inputs(
            input_ids=torch.zeros(width, dtype=torch.long, device="cuda"),
            position_ids=torch.arange(width, device="cuda"),
            hidden_states=torch.zeros(width, HIDDEN, device="cuda"),
            accepted_tokens=accepted,
            num_accepted_tokens=torch.tensor([num_accepted], device="cuda", dtype=torch.int32),
            attn_metadata=attn,
            spec_metadata=spec,
            draft_model=gen_drafter,
            total_target_tokens=width,
        )
        assert ("mla_block_fixup" in inputs) == fused
        context_len = prompt_len + num_accepted
        assert int(gen_worker._ctx_len[slot]) == context_len
        assert inputs["query_positions"][0].tolist() == list(
            range(context_len, context_len + BLOCK)
        )
        drafted = gen_drafter.dflash_forward(
            noise_embedding=inputs["noise_embedding"],
            query_positions=inputs["query_positions"],
            num_ctx_per_req=inputs["num_ctx_per_req"],
            ctx_k_cache=inputs["ctx_k_cache"],
            ctx_v_cache=inputs["ctx_v_cache"],
            ctx_cache_batch_idx=inputs["ctx_cache_batch_idx"],
            ctx_kv_cache=inputs["ctx_kv_cache"],
            ctx_page_table=inputs["ctx_page_table"],
            block_fixup=inputs.get("mla_block_fixup"),
        )
        update = gen_worker.snapshot_managed_draft_history()
        torch.cuda.synchronize()
        update.publish()
        history = gen_manager.get_draft_history(9)
        assert (history.valid_length, history.position) == (context_len, context_len)

        # Reference: a dense context of the same latents, computed locally.
        positions = torch.arange(context_len, device="cuda")
        all_captured = torch.cat([captured, gen_captured[:num_accepted]])
        latents = _context_latents(gen_drafter, all_captured, positions)
        dense = torch.zeros(1, LAYERS, context_len + BLOCK, 1, LATENT, device="cuda")
        dense = dense.to(torch.bfloat16)
        dense[0, :, :context_len] = latents.permute(1, 0, 2, 3)
        expected = gen_drafter.dflash_forward(
            noise_embedding=inputs["noise_embedding"],
            query_positions=inputs["query_positions"],
            num_ctx_per_req=torch.tensor([context_len], device="cuda"),
            ctx_k_cache=dense,
            ctx_v_cache=None,
            ctx_cache_batch_idx=torch.zeros(1, dtype=torch.long, device="cuda"),
        )
        # GEN's pages hold exactly the latents of the same hidden states.
        table = gen_manager.get_draft_block_table([9])[0].long().cuda()
        for layer in range(LAYERS):
            pool = gen_manager.get_draft_buffers(layer, kv_layout="HND")[:, 0, 0]
            assert torch.equal(
                pool[table[positions // PAGE], positions % PAGE], latents[:, layer, 0]
            )
        # The paged and the dense reads differ only in reduction order.
        error = (drafted.float() - expected.float()).abs().max()
        torch.testing.assert_close(drafted, expected, rtol=5e-2, atol=5e-2)

        # Negative control: a zero-initialized prompt context drafts far off.
        dense[0, :, :prompt_len] = 0
        zeroed = gen_drafter.dflash_forward(
            noise_embedding=inputs["noise_embedding"],
            query_positions=inputs["query_positions"],
            num_ctx_per_req=torch.tensor([context_len], device="cuda"),
            ctx_k_cache=dense,
            ctx_v_cache=None,
            ctx_cache_batch_idx=torch.zeros(1, dtype=torch.long, device="cuda"),
        )
        assert (zeroed.float() - expected.float()).abs().max() > 10 * error
    finally:
        ctx_manager.shutdown()
        gen_manager.shutdown()


# ---------------------------------------------------------------------------
# A prompt above TLLM_DSPARK_DRAFT_CONTEXT_LIMIT, and a prompt whose prefix the context server reused.
# ---------------------------------------------------------------------------


def _copy_draft_pages(src_manager, src_pages, dst_manager, dst_pages):
    """The transfer of the drafter's pages, page for page (NIXL moves the same bytes)."""
    src = torch.as_tensor(list(src_pages), dtype=torch.long, device="cuda")
    dst = torch.as_tensor(list(dst_pages), dtype=torch.long, device="cuda")
    for layer in range(LAYERS):
        dst_pool = dst_manager.get_draft_buffers(layer, kv_layout="HND")
        dst_pool[dst] = src_manager.get_draft_buffers(layer, kv_layout="HND")[src]


def _generation_attn():
    return SimpleNamespace(
        num_seqs=1,
        num_contexts=0,
        num_ctx_tokens=0,
        max_seq_len=MAX_SEQ_LEN,
        _seq_lens=torch.zeros(1, dtype=torch.int32),
        _seq_lens_cuda=torch.zeros(1, dtype=torch.int32, device="cuda"),
    )


def _dense_draft(drafter, inputs, latents):
    """What the drafter drafts from a dense context holding exactly ``latents`` ([N, L, 1, latent])."""
    context_len = latents.shape[0]
    dense = torch.zeros(1, LAYERS, context_len + BLOCK, 1, LATENT, device="cuda")
    dense = dense.to(torch.bfloat16)
    dense[0, :, :context_len] = latents.permute(1, 0, 2, 3)
    return drafter.dflash_forward(
        noise_embedding=inputs["noise_embedding"],
        query_positions=inputs["query_positions"],
        num_ctx_per_req=torch.tensor([context_len], device="cuda"),
        ctx_k_cache=dense,
        ctx_v_cache=None,
        ctx_cache_batch_idx=torch.zeros(1, dtype=torch.long, device="cuda"),
    )


@needs_cuda
@pytest.mark.parametrize("fused", [False, True], ids=["eager", "fused"])
def test_generation_server_drafts_a_prompt_above_the_limit_from_an_empty_context(
    monkeypatch, fused
):
    """TLLM_DSPARK_DRAFT_CONTEXT_LIMIT below the prompt: the generation server keeps the pages the
    transfer filled but drafts as a drafter that never saw the prompt, with the eager and the fused
    (TLLM_SPEC_TAIL_FUSION) generation prepare alike. Its context starts empty at the first
    generation step and holds exactly the accepted tokens, from the first page on; the history it
    publishes keeps counting the prompt."""
    monkeypatch.setenv("TLLM_SPEC_TAIL_FUSION", "1" if fused else "0")
    weights = _tiny_mla_weights()
    ctx_drafter = _build_drafter(weights, context_kv_only=True)
    gen_drafter = _build_drafter(weights)
    ctx_manager = _unified_manager()
    monkeypatch.setenv(_DRAFT_CONTEXT_LIMIT_ENV, "64")
    gen_manager = _unified_manager()
    ctx_worker, gen_worker = _worker(context_kv_only=True), _worker()
    ctx_worker.set_draft_model(ctx_drafter)
    gen_worker.set_draft_model(gen_drafter)
    width = BLOCK  # K + 1 target tokens per generation step
    accepted_per_step = (1, 3, 2, 3, 1, 2, 3, 3, 1)
    try:
        prompt_len = 70
        torch.manual_seed(12)
        captured = torch.randn(prompt_len, HIDDEN * NUM_CAPTURE, device="cuda").bfloat16()
        _context_step(ctx_worker, ctx_drafter, ctx_manager, _request(5, prompt_len), captured)

        gen_request = _request(9, prompt_len)
        assert gen_manager.prepare_context(gen_request)
        assert gen_manager.resize_context(gen_request, prompt_len + len(accepted_per_step) * width)
        meta = _decode_draft_history(_encode_draft_history(ctx_manager.export_draft_history(5)))
        gen_manager.restore_draft_history(9, meta)
        num_pages = -(-prompt_len // PAGE)
        _copy_draft_pages(
            ctx_manager,
            ctx_manager.get_draft_block_indices([5])[0][:num_pages],
            gen_manager,
            gen_manager.get_draft_block_indices([9])[0][:num_pages],
        )
        history = gen_manager.get_draft_history(9)
        assert (history.valid_length, history.position) == (0, prompt_len)

        hidden, context_len = [], 0
        for num_accepted in accepted_per_step:
            gen_captured = torch.randn(width, HIDDEN * NUM_CAPTURE, device="cuda").bfloat16()
            spec = _Captured([9], gen_captured)
            attn = _generation_attn()
            gen_worker.prepare_managed_draft_cache(gen_drafter, spec, attn, _resources(gen_manager))
            slot = gen_worker._req_to_slot[9]
            assert int(gen_worker._ctx_len[slot]) == context_len
            assert int(gen_worker._ctx_position_offset[slot]) == prompt_len
            accepted = torch.randint(0, VOCAB - 4, (1, width), device="cuda", dtype=torch.int32)
            inputs = gen_worker.prepare_1st_drafter_inputs(
                input_ids=torch.zeros(width, dtype=torch.long, device="cuda"),
                position_ids=torch.arange(width, device="cuda"),
                hidden_states=torch.zeros(width, HIDDEN, device="cuda"),
                accepted_tokens=accepted,
                num_accepted_tokens=torch.tensor([num_accepted], device="cuda", dtype=torch.int32),
                attn_metadata=attn,
                spec_metadata=spec,
                draft_model=gen_drafter,
                total_target_tokens=width,
            )
            assert ("mla_block_fixup" in inputs) == fused
            hidden.append(gen_captured[:num_accepted])
            context_len += num_accepted
            assert int(gen_worker._ctx_len[slot]) == context_len
            assert inputs["query_positions"][0].tolist() == list(
                range(context_len, context_len + BLOCK)
            )
            drafted = gen_drafter.dflash_forward(
                noise_embedding=inputs["noise_embedding"],
                query_positions=inputs["query_positions"],
                num_ctx_per_req=inputs["num_ctx_per_req"],
                ctx_k_cache=inputs["ctx_k_cache"],
                ctx_v_cache=inputs["ctx_v_cache"],
                ctx_cache_batch_idx=inputs["ctx_cache_batch_idx"],
                ctx_kv_cache=inputs["ctx_kv_cache"],
                ctx_page_table=inputs["ctx_page_table"],
                block_fixup=inputs.get("mla_block_fixup"),
            )
            update = gen_worker.snapshot_managed_draft_history()
            torch.cuda.synchronize()
            update.publish()
            history = gen_manager.get_draft_history(9)
            assert (history.valid_length, history.position) == (
                context_len,
                prompt_len + context_len,
            )

            # Reference: a drafter that saw only the accepted tokens.
            latents = _context_latents(
                gen_drafter, torch.cat(hidden), torch.arange(context_len, device="cuda")
            )
            expected = _dense_draft(gen_drafter, inputs, latents)
            torch.testing.assert_close(drafted, expected, rtol=5e-2, atol=5e-2)

        # The accepted tokens' context sits in the request's pages from the first one on.
        table = gen_manager.get_draft_block_table([9])[0].long().cuda()
        positions = torch.arange(context_len, device="cuda")
        for layer in range(LAYERS):
            pool = gen_manager.get_draft_buffers(layer, kv_layout="HND")[:, 0, 0]
            assert torch.equal(
                pool[table[positions // PAGE], positions % PAGE], latents[:, layer, 0]
            )
        # Negative control: the prompt's context, which the drafter must not read, drafts far off.
        error = (drafted.float() - expected.float()).abs().max()
        prompt_latents = _context_latents(
            gen_drafter, captured, torch.arange(prompt_len, device="cuda")
        )
        with_prompt = _dense_draft(gen_drafter, inputs, torch.cat([prompt_latents, latents]))
        assert (with_prompt.float() - expected.float()).abs().max() > 10 * error
    finally:
        ctx_manager.shutdown()
        gen_manager.shutdown()


@needs_cuda
@pytest.mark.parametrize("sink", [0, 4], ids=["window", "sink_and_window"])
@pytest.mark.parametrize("fused", [False, True], ids=["eager", "fused"])
def test_generation_server_drafts_a_prompt_above_the_limit_from_its_last_tokens(
    monkeypatch, fused, sink
):
    """TLLM_DSPARK_DRAFT_CONTEXT_WINDOW with the limit below the prompt: the generation server
    drafts as a drafter whose context is the prompt's first ``sink`` and last ``window`` tokens at
    the positions 0.. they get in that context, with the eager and the fused generation prepare
    alike. Its accepted tokens follow them in the pages, and the history it publishes keeps
    counting the prompt."""
    monkeypatch.setenv("TLLM_SPEC_TAIL_FUSION", "1" if fused else "0")
    weights = _tiny_mla_weights()
    ctx_drafter = _build_drafter(weights, context_kv_only=True)
    gen_drafter = _build_drafter(weights)
    ctx_manager = _unified_manager()
    window = 16
    monkeypatch.setenv(_DRAFT_CONTEXT_LIMIT_ENV, "64")
    monkeypatch.setenv(_DRAFT_CONTEXT_WINDOW_ENV, str(window))
    monkeypatch.setenv(_DRAFT_CONTEXT_SINK_ENV, str(sink))
    gen_manager = _unified_manager()
    ctx_worker, gen_worker = _worker(context_kv_only=True), _worker()
    ctx_worker.set_draft_model(ctx_drafter)
    gen_worker.set_draft_model(gen_drafter)
    width = BLOCK  # K + 1 target tokens per generation step
    accepted_per_step = (1, 3, 2, 3, 1, 2)
    bound = sink + window
    try:
        prompt_len = 70
        torch.manual_seed(13)
        captured = torch.randn(prompt_len, HIDDEN * NUM_CAPTURE, device="cuda").bfloat16()
        _context_step(ctx_worker, ctx_drafter, ctx_manager, _request(5, prompt_len), captured)

        gen_request = _request(9, prompt_len)
        assert gen_manager.prepare_context(gen_request)
        assert gen_manager.resize_context(gen_request, prompt_len + len(accepted_per_step) * width)
        meta = _decode_draft_history(_encode_draft_history(ctx_manager.export_draft_history(5)))
        gen_manager.restore_draft_history(9, meta)
        num_pages = -(-prompt_len // PAGE)
        _copy_draft_pages(
            ctx_manager,
            ctx_manager.get_draft_block_indices([5])[0][:num_pages],
            gen_manager,
            gen_manager.get_draft_block_indices([9])[0][:num_pages],
        )
        history = gen_manager.get_draft_history(9)
        assert (history.valid_length, history.position) == (bound, prompt_len)
        received = _draft_pages(gen_manager, 9)

        # The context the drafter keeps: the sink, then the last tokens rebased behind it.
        kept = torch.cat([captured[:sink], captured[prompt_len - window :]])
        hidden, context_len = [kept], bound
        for step, num_accepted in enumerate(accepted_per_step):
            gen_captured = torch.randn(width, HIDDEN * NUM_CAPTURE, device="cuda").bfloat16()
            spec = _Captured([9], gen_captured)
            attn = _generation_attn()
            gen_worker.prepare_managed_draft_cache(gen_drafter, spec, attn, _resources(gen_manager))
            slot = gen_worker._req_to_slot[9]
            assert int(gen_worker._ctx_len[slot]) == context_len
            assert int(gen_worker._ctx_position_offset[slot]) == prompt_len - bound
            if step == 0:
                # The move ran before the first draft: the sink pages are as received, the window
                # holds the last latents with their RoPE slice at the new positions.
                moved = _draft_pages(gen_manager, 9)
                window_latents = _context_latents(
                    gen_drafter, kept, torch.arange(bound, device="cuda")
                )
                for layer in range(LAYERS):
                    assert torch.equal(moved[layer][:sink], received[layer][:sink])
                    assert torch.equal(
                        moved[layer][sink:bound, :KV_LORA],
                        received[layer][prompt_len - window : prompt_len, :KV_LORA],
                    )
                    torch.testing.assert_close(
                        moved[layer][:bound].float(),
                        window_latents[:, layer, 0].float(),
                        rtol=2e-2,
                        atol=2e-2,
                    )
            accepted = torch.randint(0, VOCAB - 4, (1, width), device="cuda", dtype=torch.int32)
            inputs = gen_worker.prepare_1st_drafter_inputs(
                input_ids=torch.zeros(width, dtype=torch.long, device="cuda"),
                position_ids=torch.arange(width, device="cuda"),
                hidden_states=torch.zeros(width, HIDDEN, device="cuda"),
                accepted_tokens=accepted,
                num_accepted_tokens=torch.tensor([num_accepted], device="cuda", dtype=torch.int32),
                attn_metadata=attn,
                spec_metadata=spec,
                draft_model=gen_drafter,
                total_target_tokens=width,
            )
            assert ("mla_block_fixup" in inputs) == fused
            hidden.append(gen_captured[:num_accepted])
            context_len += num_accepted
            assert int(gen_worker._ctx_len[slot]) == context_len
            assert inputs["query_positions"][0].tolist() == list(
                range(context_len, context_len + BLOCK)
            )
            drafted = gen_drafter.dflash_forward(
                noise_embedding=inputs["noise_embedding"],
                query_positions=inputs["query_positions"],
                num_ctx_per_req=inputs["num_ctx_per_req"],
                ctx_k_cache=inputs["ctx_k_cache"],
                ctx_v_cache=inputs["ctx_v_cache"],
                ctx_cache_batch_idx=inputs["ctx_cache_batch_idx"],
                ctx_kv_cache=inputs["ctx_kv_cache"],
                ctx_page_table=inputs["ctx_page_table"],
                block_fixup=inputs.get("mla_block_fixup"),
            )
            update = gen_worker.snapshot_managed_draft_history()
            torch.cuda.synchronize()
            update.publish()
            history = gen_manager.get_draft_history(9)
            assert (history.valid_length, history.position) == (
                context_len,
                prompt_len - bound + context_len,
            )

            # Reference: a drafter that saw the kept tokens and then the accepted ones.
            latents = _context_latents(
                gen_drafter, torch.cat(hidden), torch.arange(context_len, device="cuda")
            )
            expected = _dense_draft(gen_drafter, inputs, latents)
            torch.testing.assert_close(drafted, expected, rtol=5e-2, atol=5e-2)

        # Negative control: the whole prompt's context, which the drafter must not read, drafts
        # far off.
        error = (drafted.float() - expected.float()).abs().max()
        prompt_latents = _context_latents(
            gen_drafter, captured, torch.arange(prompt_len, device="cuda")
        )
        whole = _dense_draft(gen_drafter, inputs, torch.cat([prompt_latents, latents[bound:]]))
        assert (whole.float() - expected.float()).abs().max() > 10 * error
    finally:
        ctx_manager.shutdown()
        gen_manager.shutdown()


def _transceiver(manager):
    """The page and history bookkeeping of a KvCacheTransceiverV2 serving ``manager``, without its
    transfer worker."""
    from tensorrt_llm._torch.disaggregation.resource.cache_reuse import create_cache_reuse_adapter
    from tensorrt_llm._torch.disaggregation.resource.kv_extractor import KVRegionExtractorV1
    from tensorrt_llm._torch.disaggregation.transceiver import KvCacheTransceiverV2

    transceiver = object.__new__(KvCacheTransceiverV2)
    transceiver._kv_cache_manager = manager
    transceiver._reuse_adapter = create_cache_reuse_adapter(manager)
    transceiver._page_table = KVRegionExtractorV1(manager).page_table
    return transceiver


@needs_cuda
@pytest.mark.parametrize(
    "draft_context_limit", [None, "120"], ids=["no_limit", "limit_below_prompt"]
)
def test_generation_server_receives_the_whole_context_of_a_prompt_whose_prefix_was_reused(
    monkeypatch, draft_context_limit
):
    """Turn two of a conversation on a context server with block reuse: the target reuses turn
    one's blocks and the drafter stores only the tail. A generation server without block reuse
    must still receive the drafter context of the whole prompt: the draft pages the transceiver
    sends and receives cover every prompt block, the reused ones included, and the generation
    server binds the whole context, or an empty one at the prompt's position when the prompt is
    above TLLM_DSPARK_DRAFT_CONTEXT_LIMIT. The transport itself (NIXL) is a page copy here."""
    from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequestType

    weights = _tiny_mla_weights()
    ctx_drafter = _build_drafter(weights, context_kv_only=True)
    gen_drafter = _build_drafter(weights)
    reference = _build_drafter(weights)
    ctx_manager = _unified_manager(enable_block_reuse=True)
    if draft_context_limit is not None:
        monkeypatch.setenv(_DRAFT_CONTEXT_LIMIT_ENV, draft_context_limit)
    gen_manager = _unified_manager()
    ctx_worker, gen_worker = _worker(context_kv_only=True), _worker()
    ctx_worker.set_draft_model(ctx_drafter)
    gen_worker.set_draft_model(gen_drafter)
    first_len, second_len = 100, 170
    try:
        torch.manual_seed(13)
        captured = torch.randn(second_len, HIDDEN * NUM_CAPTURE, device="cuda").bfloat16()
        expected = _context_latents(reference, captured, torch.arange(second_len, device="cuda"))

        first = _request(30, first_len)
        _batch_step(ctx_worker, ctx_drafter, ctx_manager, [(first, captured[:first_len], 0)])
        first.context_current_position = first_len
        ctx_manager.update_context_resources(SimpleNamespace(context_requests=[first]))
        ctx_manager.free_resources(first)
        second = _request(31, second_len)
        assert ctx_manager.prepare_context(second)
        reused = second.context_current_position
        assert 0 < reused <= first_len
        assert ctx_manager.resize_context(second, second_len - reused)
        _batch_step(
            ctx_worker,
            ctx_drafter,
            ctx_manager,
            [(second, captured[reused:], reused)],
            prepared=True,
        )

        # Send side, as the context server's transceiver prepares it.
        ctx_transceiver = _transceiver(ctx_manager)
        ctx_transceiver._pack_draft_history(second)
        sent = ctx_transceiver._create_kv_slice(second)

        # Receive side: a generation-only request for the same prompt, admitted as a generation
        # server admits it.
        gen_request = LlmRequest(
            request_id=32,
            max_new_tokens=8,
            input_tokens=list(range(10, 10 + second_len)),
            sampling_config=SamplingConfig(),
            is_streaming=False,
            llm_request_type=LlmRequestType.LLMREQUEST_TYPE_GENERATION_ONLY,
        )
        assert gen_manager.prepare_disagg_gen_init(gen_request)
        gen_transceiver = _transceiver(gen_manager)
        received = gen_transceiver._create_kv_slice(gen_request)

        assert (sent.token_range.start, sent.token_range.end) == (0, second_len)
        assert (received.token_range.start, received.token_range.end) == (0, second_len)
        num_pages = -(-second_len // PAGE)
        draft_groups = [
            group_id
            for group_id, group in enumerate(ctx_transceiver._page_table.layer_groups)
            if getattr(group, "cp_as_tp", False)
        ]
        assert draft_groups, "the drafter's pages travel as a layer group of their own"
        for group_id in draft_groups:
            sent_pages = sent.block_ids_per_layer_groups[group_id].tolist()
            received_pages = received.block_ids_per_layer_groups[group_id].tolist()
            # Every prompt page, reused or computed, and exactly the pages the drafters read.
            assert sent_pages == ctx_manager.get_draft_block_indices([31])[0][:num_pages]
            assert received_pages == gen_manager.get_draft_block_indices([32])[0][:num_pages]
        _copy_draft_pages(ctx_manager, sent_pages, gen_manager, received_pages)

        # The history rides the aux record.
        gen_request.py_draft_transfer_history = _decode_draft_history(
            _encode_draft_history(second.py_draft_transfer_history)
        )
        gen_transceiver._restore_draft_history(gen_request)

        history = gen_manager.get_draft_history(32)
        bound = second_len if draft_context_limit is None else 0
        assert (history.valid_length, history.position) == (bound, second_len)
        _assert_context_stored(gen_manager, 32, expected, second_len)
        gen_worker.prepare_managed_draft_cache(
            gen_drafter,
            _Captured([32], torch.zeros(BLOCK, HIDDEN * NUM_CAPTURE, device="cuda").bfloat16()),
            _generation_attn(),
            _resources(gen_manager),
        )
        slot = gen_worker._req_to_slot[32]
        assert int(gen_worker._ctx_len[slot]) == bound
        assert int(gen_worker._ctx_position_offset[slot]) == second_len - bound
    finally:
        ctx_manager.shutdown()
        gen_manager.shutdown()
