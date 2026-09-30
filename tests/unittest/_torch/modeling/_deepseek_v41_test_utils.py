# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared fixtures for CSA2 replay model and cache tests."""

import traceback
import weakref
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from test_modeling_deepseekv41 import _tiny_v41_model_config

import tensorrt_llm
from tensorrt_llm._torch.attention.backends.sparse.csa2.cache_manager import (
    CSA2CacheManager,
    CSA2CacheRole,
)
from tensorrt_llm._torch.attention.backends.sparse.csa2.ced import prepare_ced_global_kv
from tensorrt_llm._torch.attention.backends.sparse.csa2.indexer import CSA2Indexer
from tensorrt_llm._torch.attention.backends.sparse.csa2.metadata import CSA2TrtllmMetadata
from tensorrt_llm._torch.attention.backends.sparse.csa2.quantization import read_index_rows
from tensorrt_llm._torch.metadata import KVCacheParams
from tensorrt_llm._torch.models.modeling_deepseekv41 import DeepseekV41ForCausalLM
from tensorrt_llm._torch.modules.mhc.hyper_connection import HCState
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest, SamplingConfig
from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests
from tensorrt_llm._torch.utils import model_extra_attrs
from tensorrt_llm.llmapi.llm_args import KvCacheConfig
from tensorrt_llm.models.modeling_utils import QuantConfig


def _init_random_weights(model, seed: int = 0) -> None:
    """Initialize floating parameters reproducibly with unit norm/quantization scales."""
    generator = torch.Generator(device="cuda").manual_seed(seed)
    for name, param in model.named_parameters():
        if "weight_scale" in name and param.dtype == torch.uint8:
            # MXFP8 scales store the biased UE8M0 exponent; 127 represents one.
            param.data.fill_(127)
            continue
        if not param.dtype.is_floating_point:
            continue
        # Packed MoE matrices also end in `_weight`; filling those with ones
        # collapses hidden channels and hides input differences after BF16 rounding.
        if "norm" in name or "weight_scale" in name:
            param.data.fill_(1.0)
        elif param.dtype == torch.float8_e4m3fn:
            values = torch.empty_like(param, dtype=torch.float32)
            values.normal_(0.0, 0.02, generator=generator)
            param.data.copy_(values)
        else:
            param.data.normal_(0.0, 0.02, generator=generator)
    for layer in model.model.layers:
        attention = layer.self_attn
        if attention.projection_quantization == "mxfp8":
            scales = torch.full(
                (attention.o_lora_rank, attention.o_a_proj.shape[-1] // 32),
                127,
                dtype=torch.uint8,
                device=attention.o_a_proj.device,
            )
            packed = torch.ops.trtllm.block_scale_interleave(scales)
            attention.o_a_proj_scale = torch.stack([packed] * attention.n_local_groups)


def _apply_swa_floor(metadata, floors: torch.Tensor) -> None:
    """Independent reference masking for explicit layer-loop comparisons."""
    # SWA slots are resolved when a forward enters its first layer; resolve
    # them now so the layers below read the masked slots.
    metadata._ensure_swa_slots()
    width = metadata.kv_cache_manager.layout.window_size
    logical = (
        metadata.csa2_positions[:, None] - width + 1 + torch.arange(width, device=floors.device)
    )
    for layer, indices in metadata.csa2_swa_indices.items():
        metadata.csa2_swa_indices[layer] = indices.masked_fill(logical < floors[:, None], -1)


@pytest.fixture(params=["auto", "fp8"])
def cache_config(request, monkeypatch):
    monkeypatch.setenv("TRTLLM_V41_DECODER_BOUNDED_REPLAY", "0")
    model_config, sparse = _tiny_v41_model_config()
    # Keep the released role pattern with a smaller owner index: layer 5 owns
    # global KV/candidates; layer 6 reuses Top-K; layer 7 rescores candidates.
    text = model_config.pretrained_config.text_config
    text.candidate_source_layer_id = 5
    text.candidate_topk_blocks = 72
    text.candidate_block_size = 8
    model_config.extra_attrs["kv_cache_dtype"] = request.param
    model_config.quant_config = QuantConfig(
        kv_cache_quant_algo="FP8" if request.param == "fp8" else None
    )
    return model_config, sparse, request.param


@pytest.fixture
def model_and_config(cache_config):
    model_config, sparse, dtype = cache_config
    model = DeepseekV41ForCausalLM(model_config).cuda()
    _init_random_weights(model)
    assert model.model.layers[5].self_attn.layer.candidate_source == 5
    assert model.model.layers[7].self_attn.layer.candidate_source == 5
    yield model, model_config, sparse, dtype
    del model
    torch.cuda.empty_cache()


@contextmanager
def cache_case(
    config,
    sparse,
    dtype,
    lengths,
    past,
    num_contexts,
    *,
    reuse=False,
    scratch=False,
    spec_config=None,
    tokens_per_block=128,
):
    text = config.pretrained_config
    batch = len(lengths)
    manager = CSA2CacheManager(
        kv_cache_config=KvCacheConfig(
            dtype=dtype,
            enable_block_reuse=reuse,
            enable_swa_scratch_reuse=scratch,
            max_tokens=4096,
            event_buffer_max_size=0,
        ),
        kv_cache_type=tensorrt_llm.bindings.internal.batch_manager.CacheType.SELFKONLY,
        num_layers=text.num_hidden_layers,
        num_kv_heads=1,
        head_dim=text.head_dim,
        tokens_per_block=tokens_per_block,
        max_seq_len=2048,
        max_batch_size=batch,
        mapping=text.mapping,
        dtype=tensorrt_llm.bindings.DataType.FP8
        if dtype == "fp8"
        else tensorrt_llm.bindings.DataType.BF16,
        vocab_size=text.vocab_size,
        max_num_tokens=2048,
        sparse_attention_config=sparse,
        model_config=config,
        spec_config=spec_config,
        pretrained_config=text,
    )
    requests = []
    try:
        for i, (n, p) in enumerate(zip(lengths, past)):
            req = LlmRequest(
                request_id=i,
                max_new_tokens=256,
                input_tokens=list(range(p + n)),
                sampling_config=SamplingConfig(),
                is_streaming=False,
            )
            assert manager.prepare_context(req)
            requests.append(req)
            kv = manager.kv_cache_map[req.py_request_id]
            kv.enable_swa_scratch_reuse = scratch
            # Admission selects OPTIONAL groups and resumes the cache here,
            # just as the scheduler does before preparing attention metadata.
            assert manager.resize_context(req, p + n - req.context_current_position)
            # Prefix warm-up must be able to write the whole prefix: advancing
            # the manager's history to p here would reclaim its early SWA pages.
            assert kv.resize(kv.capacity, 0)
        scheduled = ScheduledRequests()
        scheduled.context_requests_last_chunk = requests[:num_contexts]
        scheduled.generation_requests = requests[num_contexts:]
        manager.prepare_resources(scheduled)
        metadata = CSA2TrtllmMetadata(
            seq_lens=torch.tensor(lengths, dtype=torch.int32),
            num_contexts=num_contexts,
            max_num_requests=batch,
            max_num_tokens=2048,
            kv_cache_params=KVCacheParams(use_cache=True, num_cached_tokens_per_seq=list(past)),
            kv_cache_manager=manager,
            request_ids=list(range(batch)),
            prompt_lens=[
                n if i < num_contexts else p for i, (p, n) in enumerate(zip(past, lengths))
            ],
            mapping=text.mapping,
        )
        # Exercise actual index queries even for short rows and generation.
        buffers = {}
        for layer, role in manager._layer_roles:
            buffer = manager.get_buffers(layer, role)
            buffers.setdefault(buffer.data_ptr(), buffer)
        # Initialize storage to finite bytes before cold writes / prefix warm-up.
        # Both compared paths restore identical history; this is not a quality test.
        for buffer in buffers.values():
            buffer.zero_()
        # Weights already use a local fixed generator. Keep input IDs similarly
        # reproducible so FP8/chunking failures do not depend on test ordering.
        generator = torch.Generator(device="cuda").manual_seed(0)
        ids = torch.randint(
            0,
            text.vocab_size,
            (sum(lengths),),
            device="cuda",
            dtype=torch.int32,
            generator=generator,
        )
        positions = torch.cat(
            [
                torch.arange(p, p + n, device="cuda", dtype=torch.int32)
                for p, n in zip(past, lengths)
            ]
        ).unsqueeze(0)
        attrs = config.extra_attrs.copy()
        attrs["attention_metadata"] = weakref.ref(metadata)
        with model_extra_attrs(attrs):
            metadata.prepare()
            yield metadata, ids, positions, list(buffers.values())
    except RuntimeError:
        # Native cleanup may abort after a poisoned CUDA context. Report the
        # original kernel failure first so teardown does not hide its traceback.
        traceback.print_exc()
        raise
    finally:
        for req in requests:
            manager.free_resources(req)
        manager.shutdown()


@contextmanager
def canonical_routing():
    """Keep Top-K scores exact while fixing ties and attention reduction order."""
    original = CSA2Indexer.forward
    select = CSA2Indexer.select_prepared_scores

    def stable_select(indexer, scores, output, row_starts, row_ends, **kwargs):
        result = select(indexer, scores, output, row_starts, row_ends, **kwargs)
        columns = torch.arange(scores.shape[1], device=scores.device)
        valid = torch.ones_like(scores, dtype=torch.bool)
        if row_starts is not None:
            valid &= columns >= row_starts[:, None]
        if row_ends is not None:
            valid &= columns < row_ends[:, None]
        bounded = scores.masked_fill(~valid, -torch.inf)
        ranked = bounded.argsort(dim=-1, descending=True, stable=True)[:, : output.shape[1]]
        expected_scores = bounded.gather(1, ranked)
        actual_scores = bounded.gather(1, output.long().clamp(0, scores.shape[1] - 1))
        actual_scores.masked_fill_(output < 0, -torch.inf)
        padding = output.shape[1] - ranked.shape[1]
        # The real selector still runs and must select the same score multiset;
        # only equally scored membership is resolved by logical column order.
        torch.testing.assert_close(
            actual_scores.sort(dim=-1, descending=True).values,
            torch.nn.functional.pad(expected_scores, (0, padding), value=-torch.inf),
            rtol=0,
            atol=0,
        )
        ranked.masked_fill_(expected_scores == -torch.inf, -1)
        output.copy_(torch.nn.functional.pad(ranked, (0, padding), value=-1))
        return result

    def forward(indexer, state, query_start, count):
        output = original(indexer, state, query_start, count)
        selected = state.metadata.csa2_indices[indexer.layer_idx]
        state.metadata.csa2_indices[indexer.layer_idx] = selected.sort(
            dim=-1, descending=True
        ).values
        return output

    with (
        patch.object(CSA2Indexer, "forward", forward),
        patch.object(CSA2Indexer, "select_prepared_scores", stable_select),
    ):
        yield


@contextmanager
def capture_layers(model):
    values, hooks = {}, []

    def save(name):
        def hook(module, args, output):
            if isinstance(output, HCState):
                values[name] = (output.residual.clone(), output.pre_mix.clone())
            else:
                values[name] = (output.clone(),)

        return hook

    def save_routing(module, args, kwargs, output):
        metadata = kwargs["attn_metadata"]
        for layer, indices in metadata.csa2_indices.items():
            values[f"topk{layer}"] = (indices.clone(),)
        for layer, blocks in metadata.csa2_candidates.items():
            values[f"candidates{layer}"] = (blocks.clone(),)

    owner = model.model.layers[5]
    modules = {
        "wkv": owner.self_attn.compressor.wkv,
        "index_wk": owner.self_attn.index_wk,
        "q_swa": owner.self_attn.wkv,
        # The CuTe fast path bypasses this module; capture it when dispatched.
        "q_b": owner.self_attn.wq_b,
    }
    modules.update({f"layer{i}": layer for i, layer in enumerate(model.model.layers)})

    def save_attention_input(module, args, kwargs):
        # The fused collapse/norm bypasses input_layernorm.forward. Observe
        # the normalized tensor actually consumed by attention in both paths.
        values["u20"] = (kwargs["hidden_states"].clone(),)

    try:
        hooks.append(model.model.layers[-1].register_forward_hook(save_routing, with_kwargs=True))
        hooks.append(
            owner.self_attn.register_forward_pre_hook(save_attention_input, with_kwargs=True)
        )
        for name, module in modules.items():
            hooks.append(module.register_forward_hook(save(name)))
        with canonical_routing():
            yield values
    finally:
        for hook in hooks:
            hook.remove()


@contextmanager
def global_precompute_scope(metadata):
    """Clean up metadata when exercising individual layers outside model.forward."""
    try:
        yield
    finally:
        metadata.csa2_precomputed_kv_layers.clear()
        metadata.reset_routing()


@contextmanager
def explicit_global_production(owner, metadata):
    """Exercise the producer API without a separate runtime execution mode."""
    frozen = []

    def prepare(module, args, kwargs):
        with (
            patch.object(
                module.hc_attn,
                "hc_coeffs",
                side_effect=AssertionError("producer ran mHC coefficients"),
            ),
            patch.object(
                module.self_attn, "forward", side_effect=AssertionError("producer ran attention")
            ),
        ):
            prepare_ced_global_kv(module, kwargs["hc_state"], metadata)
        for role in (CSA2CacheRole.GLOBAL, CSA2CacheRole.INDEX):
            buffer = metadata.kv_cache_manager.get_buffers(module.layer_idx, role).view(torch.uint8)
            frozen.append((buffer, buffer.clone()))

    def check_unchanged(module, args, output):
        for buffer, before in frozen:
            torch.testing.assert_close(buffer, before, rtol=0, atol=0)

    with global_precompute_scope(metadata):
        hook = owner.register_forward_pre_hook(prepare, with_kwargs=True)
        post_hook = owner.register_forward_hook(check_unchanged)
        try:
            yield
        finally:
            hook.remove()
            post_hook.remove()


def request(i, prompt_len=257):
    return SimpleNamespace(
        py_request_id=i,
        py_seq_slot=i,
        py_ced_replay=None,
        is_dummy=False,
        prompt_len=prompt_len,
    )


def enable_ced(model):
    model.model.decoder_replay_split = 5
    model.model.ced_kv_precompute = True
    model.model.decoder_replay_window = 128
    # Match post_load_weights: CED materializes this boundary for decoder queries,
    # and decode graphs must use the same post-update rounding there.
    model.model.layers[4]._v41_defer_post_mapping = False


def run(
    model,
    metadata,
    reqs,
    ids,
    starts,
    lengths,
    num_contexts,
    *,
    return_context_logits=False,
    pass_requests=True,
):
    metadata.request_ids = [r.py_request_id for r in reqs]
    metadata.num_contexts = num_contexts
    metadata.seq_lens = torch.tensor(lengths, dtype=torch.int32)
    metadata.prompt_lens = lengths[:num_contexts] + [
        p + n for p, n in zip(starts[num_contexts:], lengths[num_contexts:])
    ]
    metadata.kv_cache_params.num_cached_tokens_per_seq = list(starts)
    metadata.prepare_context_replay(reqs[:num_contexts])
    metadata.prepare()
    positions = torch.cat(
        [torch.arange(p, p + n, device="cuda", dtype=torch.int32) for p, n in zip(starts, lengths)]
    ).view(1, -1)
    with model_extra_attrs(
        {**model.model_config.extra_attrs, "attention_metadata": weakref.ref(metadata)}
    ):
        output = model(
            attn_metadata=metadata,
            input_ids=ids,
            position_ids=positions,
            context_requests=reqs[:num_contexts] if num_contexts and pass_requests else None,
            return_context_logits=return_context_logits,
        )
        return output


def global_bytes(metadata, start, end):
    """Read exact completed GLOBAL rows, including rows in a partial tail page."""
    manager = metadata.kv_cache_manager
    result = {}
    for layer in manager.layout.kv_source_layer_ids:
        ratio = manager.layout.compress_ratios[layer]
        rows_per_page = manager.tokens_per_block // ratio
        for role in (CSA2CacheRole.GLOBAL, CSA2CacheRole.INDEX):
            table = manager.get_cache_indices(metadata.request_ids[0], layer, role)
            pages = torch.tensor(table, dtype=torch.long, device="cuda")
            logical = torch.arange(start // ratio, end // ratio, device="cuda")
            slots = pages[logical // rows_per_page] * rows_per_page + logical % rows_per_page
            result[layer, role] = (
                manager.get_main_buffer(layer)[slots].clone()
                if role == CSA2CacheRole.GLOBAL
                else read_index_rows(manager.get_index_pages(layer), slots).clone()
            )
    return result
