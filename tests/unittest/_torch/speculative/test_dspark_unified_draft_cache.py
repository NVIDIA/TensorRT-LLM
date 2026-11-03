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

import contextlib
from functools import partial
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from tensorrt_llm._torch.disaggregation.native.auxiliary import (
    _decode_draft_history,
    _encode_draft_history,
)
from tensorrt_llm._torch.models.modeling_dspark import MLADSparkForCausalLM, _build_mla_block_fixup
from tensorrt_llm._torch.pyexecutor.input_commit import InputCommit, input_commit_scope
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest, SamplingConfig
from tensorrt_llm._torch.pyexecutor.resource_manager import (
    CacheTypeCpp,
    DataType,
    ResourceManagerType,
)
from tensorrt_llm._torch.speculative.dflash import DFlashSpecMetadata
from tensorrt_llm._torch.speculative.dspark import DSparkWorker
from tensorrt_llm._torch.speculative.interface import SpeculativeDecodingMode
from tensorrt_llm._torch.speculative.spec_tail_kernels import dflash_paged_latent_prepare
from tensorrt_llm.llmapi.llm_args import DSparkDecodingConfig, KvCacheConfig
from tensorrt_llm.mapping import Mapping

needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def _sm100f() -> bool:
    if not torch.cuda.is_available():
        return False
    from tensorrt_llm._utils import is_sm_100f

    return is_sm_100f()


needs_sm100f = pytest.mark.skipif(not _sm100f(), reason="CuTe DSL MLA decode needs SM100f")

# A tiny MLA drafter with the real latent geometry (the kernels are specialised
# on kv_lora_rank 512 / rope 64).
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


@needs_cuda
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


def test_unified_draft_cache_rejects_settings_it_does_not_honour(monkeypatch):
    from tensorrt_llm._torch.pyexecutor._util import KvCacheCreator

    creator = object.__new__(KvCacheCreator)
    creator._speculative_config = SimpleNamespace(context_kv_only=False)
    creator._is_disagg = True
    monkeypatch.setattr(KvCacheCreator, "_uses_unified_standalone_draft_cache", lambda self: True)
    monkeypatch.setattr(KvCacheCreator, "_is_standalone_dspark", lambda self: False)
    monkeypatch.setattr(KvCacheCreator, "_is_embedded_dspark", lambda self: False)
    monkeypatch.setenv("TLLM_DRAFT_KV_GPU_BUDGET_GIB", "16")
    with pytest.raises(ValueError, match="TLLM_DRAFT_KV_GPU_BUDGET_GIB"):
        creator._validate_standalone_draft_cache()
    monkeypatch.delenv("TLLM_DRAFT_KV_GPU_BUDGET_GIB")
    creator._validate_standalone_draft_cache()

    creator._speculative_config.context_kv_only = True
    creator._validate_standalone_draft_cache()
    creator._is_disagg = False
    with pytest.raises(ValueError, match="context_kv_only"):
        creator._validate_standalone_draft_cache()


@needs_cuda
def test_context_server_takes_the_drafter_out_of_the_kv_budget_not_the_headroom():
    """The KV budget gives back (1 - fraction) of what the drafter keeps resident,
    so a context-KV-only drafter does not shrink PyTorch's headroom."""
    from tensorrt_llm._torch.pyexecutor._util import KvCacheCreator

    head = torch.nn.Linear(16, 8, bias=False, device="cuda")
    draft = torch.nn.Module()
    draft.fc = torch.nn.Linear(32, 16, bias=False, device="cuda")
    draft.lm_head = head  # shared with the target: not the drafter's
    draft.table = torch.zeros(100, device="cuda")  # derived, not registered
    draft.blocks = draft.fc.weight[:, :16]  # a view: counted once
    model = torch.nn.Module()
    model.model = torch.nn.Linear(16, 16, bias=False, device="cuda")
    model.lm_head = head
    model.draft_model = draft
    metadata = _projecting_metadata(hidden=16, max_num_tokens=8)
    metadata.bind_capture_projection([draft.fc.weight[:, :16], draft.fc.weight[:, 16:]])

    creator = object.__new__(KvCacheCreator)
    creator._model_engine = SimpleNamespace(model=model, spec_metadata=metadata)
    creator._speculative_config = SimpleNamespace(context_kv_only=True)
    resident = (
        draft.fc.weight.nbytes
        + draft.table.nbytes
        + metadata.projected_captures.nbytes
        + metadata.batch_indices_cuda.nbytes
    )
    assert creator._get_context_server_spec_reserve(0.7) == int((1.0 - 0.7) * resident)
    creator._speculative_config.context_kv_only = False
    assert creator._get_context_server_spec_reserve(0.7) == 0


# ---------------------------------------------------------------------------
# The fused generation-step preparation against the eager unified path.
# ---------------------------------------------------------------------------


def _eager_gen_prepare(inputs):
    """The eager DFlashWorker / _build_mla_block_fixup semantics of one step."""
    nc, ng, width = inputs["num_contexts"], inputs["num_gens"], inputs["width"]
    rows = torch.arange(nc, nc + ng, device="cuda")
    slots = inputs["slots"].long()
    nacc = inputs["num_accepted"][rows].long()
    old = inputs["ctx_len"][slots].clone()
    counts = inputs["counts"][rows]
    tables = inputs["tables"][rows]
    j = torch.arange(width, device="cuda")
    pos = old[:, None] + j
    col = torch.minimum(pos, (counts * PAGE - 1)[:, None]).clamp(min=0)
    pages = tables.gather(1, (col // PAGE).clamp(max=tables.shape[1] - 1)).long()
    mask = (j[None, :] < nacc[:, None]).to(torch.int8)
    max_ctx = inputs["max_ctx"]
    now = torch.clamp(old + nacc, max=max_ctx)
    live = slots != inputs["dummy_slot"]
    new_len = torch.where(live, now, 0)
    offset = inputs["position_offset"].clone()
    offset[slots[live]] += (old + nacc - max_ctx).clamp_min(0)[live]
    allocated = (counts * PAGE - BLOCK).clamp(min=0)
    num_ctx = torch.minimum(new_len, allocated)
    fixup = _build_mla_block_fixup(num_ctx, tables, BLOCK, PAGE)
    ctx_len = inputs["ctx_len"].clone()
    ctx_len[slots] = new_len
    ctx_len[inputs["dummy_slot"]] = 0
    return dict(
        store_pages=pages.reshape(-1),
        store_offsets=(col % PAGE).reshape(-1),
        store_mask=mask.reshape(-1),
        ctx_positions=pos.reshape(-1),
        query_positions=now[:, None] + torch.arange(BLOCK, device="cuda"),
        num_ctx=num_ctx,
        blk_pages=fixup.pages,
        blk_offsets=fixup.offsets,
        blk_seq_lens=fixup.seq_lens_i32,
        blk_kv_bounds=fixup.kv_bounds_i32,
        ctx_len=ctx_len,
        position_offset=offset,
    )


@needs_cuda
@pytest.mark.parametrize("unified", [False, True])
def test_fused_gen_prepare_matches_the_eager_unified_path(unified):
    torch.manual_seed(1)
    num_contexts, num_gens, width, max_blocks = 1, 5, 4, 40
    dummy_slot = 8
    max_ctx = 600
    batch = num_contexts + num_gens
    counts = torch.randint(3, max_blocks, (batch,), device="cuda")
    ctx_len = torch.zeros(dummy_slot + 1, dtype=torch.long, device="cuda")
    slots = torch.tensor([0, 3, 5, 2, dummy_slot], device="cuda")
    # Global lengths, one of them at the max_ctx clamp.
    ctx_len[slots[:-1]] = torch.tensor([7, 130, 61, max_ctx - 1], device="cuda")
    inputs = dict(
        num_contexts=num_contexts,
        num_gens=num_gens,
        width=width,
        slots=slots,
        num_accepted=torch.randint(1, width + 1, (batch,), device="cuda", dtype=torch.int32),
        ctx_len=ctx_len,
        counts=counts,
        tables=torch.randint(0, 1000, (batch, max_blocks), device="cuda", dtype=torch.int32),
        max_ctx=max_ctx,
        dummy_slot=dummy_slot,
        position_offset=torch.randint(0, 9, (dummy_slot + 1,), device="cuda"),
    )
    expected = _eager_gen_prepare(inputs)

    ctx_len_k = inputs["ctx_len"].clone()
    offset_k = inputs["position_offset"].clone()
    accepted = torch.randint(0, VOCAB, (batch, width), device="cuda", dtype=torch.int32)
    seq_lens = torch.zeros(batch, dtype=torch.int32, device="cuda")
    prep = dflash_paged_latent_prepare(
        inputs["num_accepted"],
        accepted,
        slots,
        ctx_len_k,
        counts,
        inputs["tables"],
        seq_lens,
        num_contexts,
        dummy_slot,
        max_ctx,
        PAGE,
        BLOCK,
        width,
        position_offset=offset_k if unified else None,
    )._asdict()
    for key in (
        "store_pages",
        "store_offsets",
        "store_mask",
        "ctx_positions",
        "query_positions",
        "num_ctx",
        "blk_pages",
        "blk_offsets",
        "blk_seq_lens",
        "blk_kv_bounds",
    ):
        assert torch.equal(prep[key].to(expected[key].dtype), expected[key]), key
    assert prep["blk_owned"] is None
    assert torch.equal(ctx_len_k, expected["ctx_len"])
    expected_offset = expected["position_offset"] if unified else inputs["position_offset"]
    assert torch.equal(offset_k, expected_offset)
    assert seq_lens[num_contexts:].eq(width).all()


# ---------------------------------------------------------------------------
# CTX builds the drafter context and does not draft; GEN drafts from what the
# transfer carried.
# ---------------------------------------------------------------------------


def _unified_manager(max_tokens=1024):
    from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2

    return KVCacheManagerV2(
        KvCacheConfig(max_tokens=max_tokens, enable_block_reuse=False),
        CacheTypeCpp.SELF,
        num_layers=1,
        num_kv_heads=1,
        head_dim=64,
        tokens_per_block=PAGE,
        max_seq_len=MAX_SEQ_LEN,
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
        # Non-None selects the fused generation-step preparation.
        self.captured_hidden_states = hidden_states
        self.host_position_ids = None
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


def _context_step(worker, drafter, manager, request, captured):
    """One context forward of the whole prompt as the executor runs it."""
    prompt_len = captured.shape[0]
    assert manager.prepare_context(request)
    assert manager.resize_context(request, prompt_len)
    spec = _Captured([request.py_request_id], captured)
    attn = SimpleNamespace(
        num_seqs=1,
        num_contexts=1,
        num_ctx_tokens=prompt_len,
        _seq_lens=torch.tensor([prompt_len]),
        max_seq_len=MAX_SEQ_LEN,
        kv_cache_params=SimpleNamespace(num_cached_tokens_per_seq=[0]),
    )
    worker.prepare_managed_draft_cache(drafter, spec, attn, _resources(manager))
    worker._store_prefill_context(
        drafter, spec, attn, torch.arange(prompt_len, device="cuda"), prompt_len
    )
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


@contextlib.contextmanager
def _no_host_sync():
    """Fail on any host-device synchronization inside the block."""
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        yield
    finally:
        torch.cuda.set_sync_debug_mode("default")


def _context_batch(chunks):
    """Attention metadata of a context-only batch of ``(start, length)`` chunks."""
    lengths = [length for _, length in chunks]
    attn = SimpleNamespace(
        num_seqs=len(chunks),
        num_contexts=len(chunks),
        num_ctx_tokens=sum(lengths),
        _seq_lens=torch.tensor(lengths),
        max_seq_len=MAX_SEQ_LEN,
        kv_cache_params=SimpleNamespace(num_cached_tokens_per_seq=[start for start, _ in chunks]),
    )
    positions = torch.cat(
        [torch.arange(start, start + length, device="cuda") for start, length in chunks]
    )
    return attn, positions


def _projecting_metadata(hidden=HIDDEN, num_capture=NUM_CAPTURE, max_num_tokens=128):
    return DFlashSpecMetadata(
        max_draft_len=BLOCK - 1,
        max_total_draft_tokens=BLOCK - 1,
        spec_dec_mode=SpeculativeDecodingMode.DSPARK,
        max_num_requests=4,
        layers_to_capture=list(range(10, 10 + num_capture)),
        hidden_size=hidden,
        max_num_tokens=max_num_tokens,
        dtype=torch.bfloat16,
        project_captures=True,
    )


def _capture(metadata, captures):
    """The target forward's capture hooks, in layer order."""
    for i, capture in enumerate(captures):
        metadata.maybe_capture_hidden_states(10 + i, capture)


@needs_cuda
@pytest.mark.parametrize(
    "num_tokens,hidden,num_capture", [(37, HIDDEN, NUM_CAPTURE), (1024, 7168, 5)]
)
def test_projected_captures_sum_the_fc_of_the_concatenated_captures(
    num_tokens, hidden, num_capture
):
    """The per-layer FP32 partials sum to fc of the concatenated captures: the
    single fc GEMM's accumulation in another order, rounded once."""
    torch.manual_seed(0)
    weight = (torch.randn(hidden, hidden * num_capture, device="cuda") * hidden**-0.5).bfloat16()
    captures = [
        torch.randn(num_tokens, hidden, device="cuda").bfloat16() for _ in range(num_capture)
    ]
    metadata = _projecting_metadata(hidden, num_capture, max_num_tokens=num_tokens + 5)
    assert metadata.captured_hidden_states is None
    blocks = [weight[:, i * hidden : (i + 1) * hidden] for i in range(num_capture)]
    metadata.bind_capture_projection(blocks)
    assert metadata.projected_captures.shape == (num_tokens + 5, hidden)
    assert metadata.projected_captures.dtype == torch.float32
    # The first captured layer overwrites whatever an earlier step left.
    metadata.projected_captures.fill_(float("nan"))
    with _no_host_sync():
        _capture(metadata, captures)
        metadata.maybe_capture_hidden_states(99, captures[0])  # not a captured layer
    projected = metadata.get_projected_captures(num_tokens)
    assert metadata.get_hidden_states(num_tokens) is None

    concatenated = torch.cat(captures, dim=-1)
    exact = concatenated.double() @ weight.double().t()
    # FP32 accuracy, far below one BF16 unit (2**-8 relative).
    torch.testing.assert_close(projected.double(), exact, rtol=1e-4, atol=1e-4)
    # Against the single BF16-output fc GEMM: at most its own rounding apart.
    torch.testing.assert_close(
        projected.to(torch.bfloat16), F.linear(concatenated, weight), rtol=2**-6, atol=2**-6
    )


@needs_cuda
def test_drafter_finishes_the_projected_captures_like_project_target_hidden():
    drafter = _build_drafter(_tiny_mla_weights(), context_kv_only=True)
    blocks = drafter.capture_projection_blocks(NUM_CAPTURE)
    assert [tuple(block.shape) for block in blocks] == [(HIDDEN, HIDDEN)] * NUM_CAPTURE
    assert all(
        block.untyped_storage().data_ptr() == drafter.fc.weight.data_ptr() for block in blocks
    )
    # 2 * HIDDEN input features do not split into 3 captures.
    assert drafter.capture_projection_blocks(NUM_CAPTURE + 1) is None

    torch.manual_seed(3)
    captures = [torch.randn(41, HIDDEN, device="cuda").bfloat16() for _ in range(NUM_CAPTURE)]
    metadata = _projecting_metadata()
    with pytest.raises(RuntimeError, match="bind_capture_projection"):
        _capture(metadata, captures)
    metadata.bind_capture_projection(blocks)
    _capture(metadata, captures)
    got = drafter.finish_capture_projection(metadata.get_projected_captures(41))
    want = drafter.project_target_hidden(torch.cat(captures, dim=-1))
    assert got.dtype == want.dtype == torch.bfloat16
    torch.testing.assert_close(got, want)

    # A drafter without a plain fc projection keeps the concatenated captures.
    fallback = _projecting_metadata()
    fallback.bind_capture_projection(None)
    _capture(fallback, captures)
    assert fallback.get_projected_captures(41) is None
    assert torch.equal(fallback.get_hidden_states(41), torch.cat(captures, dim=-1))


@needs_cuda
@pytest.mark.parametrize("project", [False, True])
def test_context_store_does_not_wait_for_the_gpu(project):
    """A batch with a continued chunk and a new request stores the drafter
    context without a host sync; with fc folded into the capture the latents
    match up to the projection's accumulation order."""
    drafter = _build_drafter(_tiny_mla_weights(), context_kv_only=True)
    manager = _unified_manager()
    worker = _worker(context_kv_only=True)
    worker.set_draft_model(drafter)
    try:
        torch.manual_seed(4)
        lengths = {301: 48, 302: 30}
        captured = {
            rid: [torch.randn(n, HIDDEN, device="cuda").bfloat16() for _ in range(NUM_CAPTURE)]
            for rid, n in lengths.items()
        }
        for rid, n in lengths.items():
            request = _request(rid, n)
            assert manager.prepare_context(request)
            assert manager.resize_context(request, n)

        # (request id, start, length) per step: 301's prompt arrives in two chunks.
        steps = [[(301, 0, 20)], [(301, 20, 28), (302, 0, 30)]]
        expected = {rid: [] for rid in lengths}
        for step, chunks in enumerate(steps):
            attn, positions = _context_batch([(start, n) for _, start, n in chunks])
            captures = [
                torch.cat([captured[rid][layer][start : start + n] for rid, start, n in chunks])
                for layer in range(NUM_CAPTURE)
            ]
            request_ids = [rid for rid, _, _ in chunks]
            if project:
                spec = _projecting_metadata()
                spec.request_ids = request_ids
            else:
                spec = _Captured(request_ids, torch.cat(captures, dim=-1))
            worker.prepare_managed_draft_cache(drafter, spec, attn, _resources(manager))
            # The first step builds the drafter's lazy tables.
            guard = _no_host_sync() if step > 0 else contextlib.nullcontext()
            with guard:
                if project:
                    _capture(spec, captures)
                worker._store_prefill_context(drafter, spec, attn, positions, attn.num_ctx_tokens)

            projected = drafter.project_target_hidden(torch.cat(captures, dim=-1))
            offset = 0
            for rid, start, n in chunks:
                latent, _ = drafter.precompute_context_kv(
                    projected[offset : offset + n], positions[offset : offset + n]
                )
                expected[rid].append(latent)
                offset += n

        torch.cuda.synchronize()
        for rid, n in lengths.items():
            slot = worker._req_to_slot[rid]
            assert int(worker._ctx_len[slot]) == worker._ctx_len_host[slot] == n
            assert int(worker._ctx_position_offset[slot]) == 0
            want = torch.cat(expected[rid])
            stored = _draft_pages(manager, rid)
            for layer in range(LAYERS):
                if project:
                    torch.testing.assert_close(
                        stored[layer][:n], want[:, layer, 0], rtol=2e-2, atol=2e-2
                    )
                else:
                    assert torch.equal(stored[layer][:n], want[:, layer, 0])
    finally:
        manager.shutdown()


@needs_cuda
@pytest.mark.parametrize("fused", [False, True])
def test_generation_server_drafts_from_the_transferred_context(monkeypatch, fused):
    """CTX builds the history, the transfer carries pages + history, GEN's draft
    equals a drafter that saw the same target hidden states locally."""
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
        if not fused:
            spec.captured_hidden_states = None
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
        assert (inputs["block_placement"] is not None) == fused
        context_len = prompt_len + num_accepted
        assert int(gen_worker._ctx_len[slot]) == context_len
        # DSpark's shift_label layout drafts K tokens from K block slots.
        query_width = gen_worker._compute_block_size
        assert query_width == BLOCK - 1
        assert inputs["query_positions"][0].tolist() == list(
            range(context_len, context_len + query_width)
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
        pos = positions
        for layer in range(LAYERS):
            pool = gen_manager.get_draft_buffers(layer, kv_layout="HND")[:, 0, 0]
            assert torch.equal(pool[table[pos // PAGE], pos % PAGE], latents[:, layer, 0])
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


def _gen_step_drafter_inputs(weights, ctx_manager, prompt_len, gen_ids, staged):
    """One GEN decode step for requests transferred from CTX request 5: the drafter's
    inputs, its draft and the published histories, with the managed-cache uploads of
    the step either copied directly or staged in an input commit as the decoder runner
    does on a CUDA-graph step."""
    gen_drafter = _build_drafter(weights)
    gen_manager = _unified_manager()
    gen_worker = _worker()
    gen_worker.set_draft_model(gen_drafter)
    width = BLOCK
    try:
        meta = _decode_draft_history(_encode_draft_history(ctx_manager.export_draft_history(5)))
        num_pages = -(-prompt_len // PAGE)
        src_table = ctx_manager.get_draft_block_table([5])[0][:num_pages].long().cuda()
        for request_id in gen_ids:
            request = _request(request_id, prompt_len)
            assert gen_manager.prepare_context(request)
            assert gen_manager.resize_context(request, prompt_len + 2 * width)
            gen_manager.restore_draft_history(request_id, meta)
            dst_table = gen_manager.get_draft_block_table([request_id])[0][:num_pages]
            dst_table = dst_table.long().cuda()
            for layer in range(LAYERS):
                gen_pool = gen_manager.get_draft_buffers(layer, kv_layout="HND")
                gen_pool[dst_table] = ctx_manager.get_draft_buffers(layer, kv_layout="HND")[
                    src_table
                ]
        num_seqs = len(gen_ids)
        torch.manual_seed(4)
        gen_captured = torch.randn(num_seqs * width, HIDDEN * NUM_CAPTURE, device="cuda").bfloat16()
        spec = _Captured(list(gen_ids), gen_captured)
        attn = SimpleNamespace(
            num_seqs=num_seqs,
            num_contexts=0,
            num_ctx_tokens=0,
            max_seq_len=MAX_SEQ_LEN,
            _seq_lens=torch.zeros(num_seqs, dtype=torch.int32),
            _seq_lens_cuda=torch.zeros(num_seqs, dtype=torch.int32, device="cuda"),
        )
        commit = InputCommit() if staged else None
        if staged:
            commit.begin()
        with input_commit_scope(commit):
            gen_worker.prepare_managed_draft_cache(gen_drafter, spec, attn, _resources(gen_manager))
        if staged:
            commit.commit()
            # Each managed-cache destination is uploaded once per step.
            assert commit.num_merged_runs == 0
        slots = [gen_worker._req_to_slot[request_id] for request_id in gen_ids]
        state = {
            "ctx_len": gen_worker._ctx_len[slots].clone(),
            "ctx_position_offset": gen_worker._ctx_position_offset[slots].clone(),
            "snapshot_slots": gen_worker._managed_snapshot_slots[:num_seqs].clone(),
        }
        num_accepted = torch.tensor([2, 3, 1, 3][:num_seqs], device="cuda", dtype=torch.int32)
        accepted = torch.randint(0, VOCAB - 4, (num_seqs, width), device="cuda", dtype=torch.int32)
        inputs = gen_worker.prepare_1st_drafter_inputs(
            input_ids=torch.zeros(num_seqs * width, dtype=torch.long, device="cuda"),
            position_ids=torch.arange(num_seqs * width, device="cuda"),
            hidden_states=torch.zeros(num_seqs * width, HIDDEN, device="cuda"),
            accepted_tokens=accepted,
            num_accepted_tokens=num_accepted,
            attn_metadata=attn,
            spec_metadata=spec,
            draft_model=gen_drafter,
            total_target_tokens=num_seqs * width,
        )
        state["query_positions"] = inputs["query_positions"].clone()
        state["num_ctx_per_req"] = inputs["num_ctx_per_req"].clone()
        state["drafted"] = gen_drafter.dflash_forward(
            noise_embedding=inputs["noise_embedding"],
            query_positions=inputs["query_positions"],
            num_ctx_per_req=inputs["num_ctx_per_req"],
            ctx_k_cache=inputs["ctx_k_cache"],
            ctx_v_cache=inputs["ctx_v_cache"],
            ctx_cache_batch_idx=inputs["ctx_cache_batch_idx"],
            ctx_kv_cache=inputs["ctx_kv_cache"],
            ctx_page_table=inputs["ctx_page_table"],
        )
        update = gen_worker.snapshot_managed_draft_history()
        torch.cuda.synchronize()
        update.publish()
        state["histories"] = [
            (h.valid_length, h.position)
            for h in (gen_manager.get_draft_history(request_id) for request_id in gen_ids)
        ]
        return state
    finally:
        gen_manager.shutdown()


@needs_cuda
@pytest.mark.parametrize("gen_ids", [(9,), (9, 11, 13)])
def test_staged_managed_cache_uploads_give_the_drafter_the_same_inputs(gen_ids):
    """The first GEN step of transferred requests assigns their slots (length reset) and
    restores their histories (length = prompt) in the same step. Staged in an input
    commit, the drafter must see what direct copies give it: the restored lengths and
    positions, hence the same query positions, context counts and draft."""
    weights = _tiny_mla_weights()
    ctx_drafter = _build_drafter(weights, context_kv_only=True)
    ctx_manager = _unified_manager()
    ctx_worker = _worker(context_kv_only=True)
    ctx_worker.set_draft_model(ctx_drafter)
    try:
        prompt_len = 70
        torch.manual_seed(2)
        captured = torch.randn(prompt_len, HIDDEN * NUM_CAPTURE, device="cuda").bfloat16()
        _context_step(ctx_worker, ctx_drafter, ctx_manager, _request(5, prompt_len), captured)
        direct = _gen_step_drafter_inputs(weights, ctx_manager, prompt_len, gen_ids, False)
        staged = _gen_step_drafter_inputs(weights, ctx_manager, prompt_len, gen_ids, True)
    finally:
        ctx_manager.shutdown()
    assert direct["ctx_len"].tolist() == [prompt_len] * len(gen_ids)
    for key in ("ctx_len", "ctx_position_offset", "snapshot_slots", "query_positions"):
        assert torch.equal(staged[key], direct[key]), key
    assert torch.equal(staged["num_ctx_per_req"], direct["num_ctx_per_req"])
    assert torch.equal(staged["drafted"], direct["drafted"])
    assert staged["histories"] == direct["histories"]


# ---------------------------------------------------------------------------
# Block reuse on the context server: the reused prefix brings its draft pages.
# ---------------------------------------------------------------------------


def _kda_manager_with_draft_layers(
    prompt_end_offset,
    policy,
    pool_ratio=None,
    *,
    max_tokens=512,
    num_kv_heads=4,
    head_dim=64,
    draft_num_kv_heads=1,
):
    from tensorrt_llm._torch.modules.kimi_kda.cache_manager import KimiK3HybridCacheManager
    from tensorrt_llm.llmapi.llm_args import BlockReuseConfig, MambaStateConfig

    kv_cache_config = KvCacheConfig(
        max_tokens=max_tokens,
        pool_ratio=pool_ratio,
        enable_block_reuse=True,
        enable_partial_reuse=False,
        block_reuse_config=BlockReuseConfig(policy=policy, max_num_turns=1),
        mamba_state_config=MambaStateConfig(
            periodic_snapshot_interval=0,
            additional_snapshot_offsets_from_end=[prompt_end_offset],
        ),
    )
    return KimiK3HybridCacheManager(
        mamba_d_state=8,
        mamba_d_conv=4,
        mamba_num_heads=4,
        mamba_n_groups=1,
        mamba_head_dim=8,
        mamba_num_layers=1,
        mamba_layer_mask=[True, False],
        mamba_cache_dtype=torch.bfloat16,
        mamba_ssm_cache_dtype=torch.bfloat16,
        kv_cache_config=kv_cache_config,
        kv_cache_type=CacheTypeCpp.SELF,
        num_layers=1,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        tokens_per_block=PAGE,
        max_seq_len=MAX_SEQ_LEN,
        max_batch_size=4,
        mapping=Mapping(world_size=1, rank=0, tp_size=1, pp_size=1),
        layer_mask=[False, True],
        vocab_size=1024,
        dtype=DataType.HALF,
        conv_state_layout="q_k_v",
        draft_num_layers=LAYERS,
        draft_num_kv_heads=draft_num_kv_heads,
        draft_head_dim=LATENT,
        draft_dtype=DataType.BF16,
        draft_kv_factor=1,
        draft_total_num_kv_heads=draft_num_kv_heads,
        draft_scratch_tokens=BLOCK + 1,
        draft_attention_backend="CUTEDSL",
    )


def _write_draft_positions(manager, request_id, positions):
    """Mark each draft position with a value naming its position and layer."""
    table = manager.get_draft_block_table([request_id])[0].long().cuda()
    pos = torch.as_tensor(positions, device="cuda")
    for layer in range(LAYERS):
        pool = manager.get_draft_buffers(layer, kv_layout="HND")
        value = (pos * LAYERS + layer).to(torch.bfloat16)[:, None].expand(-1, LATENT)
        pool[table[pos // PAGE], 0, 0, pos % PAGE] = value


@needs_cuda
@pytest.mark.parametrize("policy", ["per_request", "per_conversation"])
def test_reused_prefix_brings_its_draft_pages_and_history(monkeypatch, policy):
    """A KDA mid-chunk snapshot at 64 makes the next request with the same prefix
    reuse 64 tokens; its draft history and draft pages start from the same
    prefix, so the context server stores only the new tail."""
    from tensorrt_llm._torch.modules.kimi_kda import _kda_kernels
    from tensorrt_llm._torch.modules.kimi_kda import cache_manager as kda_cache_manager
    from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests
    from tensorrt_llm._torch.pyexecutor.scheduler.scheduler import _get_forced_context_chunk_size

    monkeypatch.setattr(kda_cache_manager, "_KDA_MIDCHUNK_SNAPSHOT", True)
    monkeypatch.setattr(_kda_kernels, "kda_prefill_supports_state_snapshots", lambda: True)
    prompt = list(range(100, 200))
    point = 64
    mgr = _kda_manager_with_draft_layers(len(prompt) - point, policy)
    try:
        assert mgr.draft_layer_ids
        first = _llm_request(1, prompt)
        mgr.prepare_expect_snapshot_points([first])
        assert mgr.prepare_context(first)
        history = mgr.get_draft_history(1)
        assert (history.valid_length, history.position) == (0, 0)
        chunk = _get_forced_context_chunk_size(first)
        assert chunk == len(prompt)
        assert mgr.resize_context(first, chunk)
        first.context_chunk_size = chunk
        batch = ScheduledRequests()
        batch.append_context_request(first)
        mgr.prepare_resources(batch)
        # What the drafter stores in this context step, and what it publishes.
        _write_draft_positions(mgr, 1, range(len(prompt)))
        torch.cuda.synchronize()
        mgr.set_draft_history(1, len(prompt), len(prompt))
        first.context_current_position = len(prompt)
        mgr.update_context_resources(batch)
        assert mgr.kv_cache_map[1].num_committed_tokens == point
        expected = [
            mgr.get_draft_buffers(layer, kv_layout="HND")[
                mgr.get_draft_block_table([1])[0][: point // PAGE].long().cuda()
            ].clone()
            for layer in range(LAYERS)
        ]
        mgr.free_resources(first)

        second = _llm_request(2, prompt + list(range(500, 510)))
        mgr.prepare_expect_snapshot_points([second])
        assert mgr.prepare_context(second)
        assert second.context_current_position == point
        history = mgr.get_draft_history(2)
        assert (history.valid_length, history.position) == (point, point)
        assert mgr.resize_context(second, second.context_remaining_length)
        reused = mgr.get_draft_block_table([2])[0][: point // PAGE].long().cuda()
        for layer in range(LAYERS):
            pages = mgr.get_draft_buffers(layer, kv_layout="HND")[reused]
            assert torch.equal(pages, expected[layer])
            values = pages[:, 0, 0, :, 0].reshape(-1)
            assert torch.equal(
                values, (torch.arange(point, device="cuda") * LAYERS + layer).bfloat16()
            )
        mgr.free_resources(second)
    finally:
        mgr.shutdown()


def _pool_group_ratio(mgr):
    """Share of each pool group (storage order) in the native ``initial_pool_ratio``,
    which has one entry per lifecycle and sums the lifecycles of a pool group."""
    config = mgr.kv_cache_manager_py_config
    life_cycle_groups, pool_groups = mgr._predict_pool_groups(config)
    assert len(config.initial_pool_ratio) == len(life_cycle_groups)
    index = {key: i for i, (key, _) in enumerate(pool_groups)}
    shares = [0.0] * len(pool_groups)
    for key, share in zip(life_cycle_groups, config.initial_pool_ratio):
        shares[index[key]] += share
    return shares


@needs_cuda
def test_target_pool_ratio_keeps_its_split_next_to_the_draft_pool():
    """kv_cache_config.pool_ratio names the target's layer groups only; the
    standalone draft layers get their own pool with the share their bytes need."""
    mgr = _kda_manager_with_draft_layers(0, "per_request", pool_ratio=[0.75, 0.25])
    try:
        ratio = _pool_group_ratio(mgr)
        assert len(ratio) == 3 and abs(sum(ratio) - 1) < 1e-6
        # Standalone-draft pool groups precede the target groups in storage.
        draft_share = ratio[0]
        assert 0 < draft_share < 1
        assert ratio[1] / ratio[2] == pytest.approx(3.0)
        assert mgr.get_draft_buffers(0, kv_layout="HND").shape[1:] == (1, 1, PAGE, LATENT)
    finally:
        mgr.shutdown()


@needs_cuda
def test_draft_pool_holds_as_many_tokens_as_the_target_attention_pool():
    """The drafter keeps one context entry per target token: with an explicit
    pool_ratio, the draft pool group holds as many blocks as the target's
    full-attention group, and the target groups keep their split of the rest."""
    from tensorrt_llm.runtime.kv_cache_manager_v2 import GPU_LEVEL

    heads, head_dim = 256, 512
    # Slots of several 2 MiB pool-size grains, so the split rounds to whole blocks.
    mgr = _kda_manager_with_draft_layers(
        0,
        "per_request",
        pool_ratio=[0.75, 0.25],
        max_tokens=1024,
        num_kv_heads=heads,
        head_dim=head_dim,
        draft_num_kv_heads=heads,
    )
    try:
        ratio = _pool_group_ratio(mgr)
        stats = mgr._get_storage_statistics(GPU_LEVEL)
        slot_bytes = [sum(stat.slot_sizes) for stat in stats]
        draft_bytes = LAYERS * heads * LATENT * 2 * PAGE
        attention_bytes = 2 * heads * head_dim * 2 * PAGE
        # Standalone-draft pool groups precede the target groups in storage.
        assert slot_bytes[0] == draft_bytes
        attention = slot_bytes.index(attention_bytes)
        state = 3 - attention
        assert mgr._predict_pool_group_block_bytes(mgr.kv_cache_manager_py_config) == (
            [draft_bytes],
            [attention_bytes if group == attention else None for group in (1, 2)],
        )
        assert ratio[state] / ratio[attention] == pytest.approx(3.0)
        assert ratio[0] / draft_bytes == pytest.approx(ratio[attention] / attention_bytes)
        assert abs(stats[0].total - stats[attention].total) <= 1
    finally:
        mgr.shutdown()


def test_k3_ctx_draft_pool_holds_the_mla_token_count():
    """Kimi K3 CTX layout, slot bytes from its startup log, tokens_per_block 128:
    two FP8 MLA layers (full history), 22 FP4 MLA layers plus their HP-tail ring,
    the KDA state, and five DSpark drafter layers. pool_ratio [0.13, 0.02, 0.4,
    0.45] names the four target pool groups in storage order; the draft gets the
    share that holds as many tokens as FP4 MLA, the scarcer full-history group."""
    from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import (
        KVCacheManagerV2,
        Role,
        _draft_ratio_for_token_parity,
    )
    from tensorrt_llm.runtime.kv_cache_manager_v2 import (
        AttentionLayerConfig,
        BufferConfig,
        CacheDomain,
        DataRole,
        LayerId,
        SsmLayerConfig,
    )

    tokens_per_block, max_seq_len = 128, 1 << 20
    layers = []

    def attention(buffers, window=None, cache_domain=CacheDomain.TARGET):
        layers.append(
            AttentionLayerConfig(
                layer_id=LayerId(len(layers)),
                buffers=[BufferConfig(role=role, size=size) for role, size in buffers],
                sliding_window_size=window,
                cache_domain=cache_domain,
            )
        )

    for _ in range(2):
        attention([(Role.KEY, 576 * tokens_per_block)], window=max_seq_len + 7)
    for _ in range(22):
        attention([(Role.KEY, 40_960), (Role.KEY_BLOCK_SCALE, 5_120)])
    layers.append(
        SsmLayerConfig(
            layer_id=LayerId(len(layers)),
            buffers=[
                BufferConfig(role=DataRole("ssm_state"), size=217_055_232),
                BufferConfig(role=DataRole("conv_state"), size=15_261_696),
            ],
        )
    )
    for _ in range(22):
        attention([(Role.MLA_HP_TAIL, 24_192)], window=256)
    for _ in range(5):
        attention([(Role.KEY, 1_152 * tokens_per_block)], cache_domain=CacheDomain.STANDALONE_DRAFT)

    manager = SimpleNamespace(max_seq_len=max_seq_len)
    manager._predict_pool_groups = partial(KVCacheManagerV2._predict_pool_groups, manager)
    draft, target = KVCacheManagerV2._predict_pool_group_block_bytes(
        manager, SimpleNamespace(tokens_per_block=tokens_per_block, layers=layers)
    )
    assert draft == [737_280]
    assert target == [147_456, None, 1_013_760, None]

    quota = 48_191_714_099
    ratios = [0.13, 0.02, 0.4, 0.45]
    share = _draft_ratio_for_token_parity(draft[0], ratios, target)
    draft_blocks = share * quota / draft[0]
    mla_blocks = ratios[2] * (1 - share) * quota / target[2]
    assert draft_blocks == pytest.approx(mla_blocks)
    assert share * quota == pytest.approx(10.86e9, rel=1e-3)
    assert mla_blocks == pytest.approx(14_730, abs=1)


def _llm_request(request_id, tokens):
    return LlmRequest(
        request_id=request_id,
        max_new_tokens=1,
        input_tokens=tokens,
        sampling_config=SamplingConfig(),
        is_streaming=False,
    )
