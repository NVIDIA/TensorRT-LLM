# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fused CSA2 sparse decode: kernel contract, CUDA graph replay and module generation routing."""

import pytest
import torch

import tensorrt_llm  # noqa: F401  (loads the torch.ops.trtllm library)
from tensorrt_llm._torch.attention.backends.sparse.csa2 import backend as csa2_backend
from tensorrt_llm._torch.attention.backends.sparse.csa2 import gen_fmha
from tensorrt_llm._torch.attention.backends.sparse.csa2.metadata import CSA2TrtllmMetadata
from tensorrt_llm._torch.attention.backends.sparse.csa2.params import CSA2Layout, CSA2Mode
from tensorrt_llm._torch.attention.backends.sparse.csa2.quantization import pack_rows, unpack_rows

HEADS = 64
DIM = 512
WINDOW = 128
TOPK = 512
SCALE = DIM**-0.5
# DSpark verification: every generation request carries one target row and five draft rows.
DSPARK_ROWS = 6
DSPARK_MAX_SEQ = 2048


def _fused_available() -> bool:
    return torch.cuda.is_available() and csa2_backend._fused_decode_supported(
        torch.cuda.current_device()
    )


pytestmark = pytest.mark.skipif(
    not _fused_available(), reason="Fused CSA2 decode requires an SM100-family GPU"
)


def _random_rows(count: int, generator: torch.Generator) -> torch.Tensor:
    """BF16 rows with per-channel magnitudes spanning several binades (exercises the group scales)."""
    rows = torch.randn(count, DIM, device="cuda", generator=generator)
    channel_scale = torch.exp2(torch.randint(-3, 5, (DIM,), device="cuda", generator=generator))
    return (rows * channel_scale).bfloat16()


def _pools(swa_rows: int, main_rows: int, seed: int):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    swa_pool = pack_rows(_random_rows(swa_rows, generator), "swa")
    main_pool = pack_rows(_random_rows(main_rows, generator), "main")
    return swa_pool, main_pool


def _reference(q, swa_pool, swa_slots, main_pool, main_slots, sinks, chunk=32):
    """Exact FP32 attention over the dequantized selected rows; invalid rows are excluded."""
    swa_kv = unpack_rows(swa_pool, DIM, "swa", torch.float32)
    main_kv = None if main_pool is None else unpack_rows(main_pool, DIM, "main", torch.float32)
    out = torch.empty(q.shape, dtype=torch.float32, device=q.device)
    for start in range(0, q.shape[0], chunk):
        end = min(q.shape[0], start + chunk)
        slots = swa_slots[start:end].long()
        valid = (slots >= 0) & (slots < swa_kv.shape[0])
        kv = swa_kv[slots.clamp(0, swa_kv.shape[0] - 1)]
        if main_kv is not None and main_slots is not None:
            main = main_slots[start:end].long()
            main_valid = (main >= 0) & (main < main_kv.shape[0])
            kv = torch.cat((kv, main_kv[main.clamp(0, main_kv.shape[0] - 1)]), 1)
            valid = torch.cat((valid, main_valid), 1)
        logits = torch.einsum("mhd,mkd->mhk", q[start:end].float(), kv) * SCALE
        logits.masked_fill_(~valid[:, None, :], -torch.inf)
        sink = sinks[None, :, None].expand(end - start, -1, 1)
        probs = torch.cat((logits, sink), -1).softmax(-1)[..., :-1]
        out[start:end] = torch.einsum("mhk,mkd->mhd", probs.nan_to_num(0.0), kv)
    return out


def _check_close(actual, expected, dead=None):
    """BF16-class agreement: Frobenius error near the BF16 rounding floor, rows aligned."""
    live = torch.ones(actual.shape[0], dtype=torch.bool, device=actual.device)
    if dead is not None and dead.any():
        assert torch.equal(actual[dead], torch.zeros_like(actual[dead])), "dead rows must be zero"
        live = ~dead
    if not live.any():
        return
    actual, expected = actual[live].float(), expected[live]
    assert torch.isfinite(actual).all()
    floor = (expected.bfloat16().float() - expected).norm() / expected.norm()
    error = (actual - expected).norm() / expected.norm()
    assert error <= max(3e-3, 2.5 * floor), (float(error), float(floor))
    cosine = torch.nn.functional.cosine_similarity(actual.flatten(1), expected.flatten(1), dim=-1)
    assert cosine.min() >= 0.9999, float(cosine.min())


def _decode(q, swa_pool, swa_indices, sinks, main_pool=None, main_indices=None, mapping=None):
    output = torch.empty_like(q, dtype=torch.bfloat16)
    table = requests = visible = None
    page_size = max_positions = 0
    if mapping is not None:
        table, page_size, max_positions, requests, visible = mapping
    torch.ops.trtllm.csa2_sparse_decode(
        q,
        swa_pool,
        swa_indices,
        main_pool,
        main_indices,
        table,
        requests,
        visible,
        page_size,
        max_positions,
        sinks,
        SCALE,
        output,
        csa2_backend._fused_decode_workspace(q.device.index),
    )
    return output


def _selections(tokens, swa_rows, main_rows, generator, holes=0.0, dead=0):
    swa = torch.randint(0, swa_rows, (tokens, WINDOW), device="cuda", generator=generator)
    main = torch.randint(0, main_rows, (tokens, TOPK), device="cuda", generator=generator)
    if holes:
        swa.masked_fill_(torch.rand(swa.shape, device="cuda", generator=generator) < holes / 2, -1)
        main.masked_fill_(torch.rand(main.shape, device="cuda", generator=generator) < holes, -1)
    # Rows past the pool are invalid too.
    swa[:, 0] = torch.where(swa[:, 1] % 7 == 0, swa_rows + 3, swa[:, 0])
    dead_rows = torch.zeros(tokens, dtype=torch.bool, device="cuda")
    if dead:
        dead_rows[torch.randperm(tokens, device="cuda", generator=generator)[:dead]] = True
        swa[dead_rows] = -1
        main[dead_rows] = -1
    return swa, main.int(), dead_rows


@pytest.mark.parametrize("batch", [1, 128])
@torch.inference_mode()
def test_fused_decode_matches_fp32_reference(batch):
    tokens = 6 * batch
    generator = torch.Generator(device="cuda").manual_seed(1000 + batch)
    swa_pool, main_pool = _pools(4096, 16384, seed=batch)
    swa, main, dead = _selections(tokens, 4096, 16384, generator, holes=0.1, dead=min(3, tokens))
    q = (torch.randn(tokens, HEADS, DIM, device="cuda", generator=generator) * 0.5).bfloat16()
    sinks = torch.randn(HEADS, device="cuda", generator=generator)
    output = _decode(q, swa_pool, swa, sinks, main_pool, main)
    _check_close(output, _reference(q, swa_pool, swa, main_pool, main, sinks), dead)


@torch.inference_mode()
def test_fused_decode_swa_only():
    tokens = 6 * 32
    generator = torch.Generator(device="cuda").manual_seed(2032)
    swa_pool, _ = _pools(1024, 64, seed=7)
    swa, _, dead = _selections(tokens, 1024, 64, generator, holes=0.2, dead=2)
    q = torch.randn(tokens, HEADS, DIM, device="cuda", generator=generator).bfloat16()
    sinks = torch.randn(HEADS, device="cuda", generator=generator)
    output = _decode(q, swa_pool, swa, sinks)
    _check_close(output, _reference(q, swa_pool, swa, None, None, sinks), dead)


def _logical_case(batch, page_size, ratio, generator):
    """Requests with 6 rows each, scattered pages (some missing) and DSpark-like positions."""
    tokens_per_request = 6
    tokens = batch * tokens_per_request
    max_positions = 4096 // ratio
    pages_per_request = max_positions // page_size
    # Requests may share physical pages; the pool stays small at any batch.
    physical_pages = 515
    table = torch.randint(
        0, physical_pages, (batch, pages_per_request), device="cuda", generator=generator
    ).int()
    table.masked_fill_(torch.rand(table.shape, device="cuda", generator=generator) < 0.05, -1)
    requests = torch.arange(batch, device="cuda").repeat_interleave(tokens_per_request)
    # One query row addresses a request the table does not have: all its main rows are invalid.
    requests[-1] = batch
    positions = torch.randint(
        700, 4096 - 8, (batch,), device="cuda", generator=generator
    ).repeat_interleave(tokens_per_request) + torch.arange(
        tokens_per_request, device="cuda"
    ).repeat(batch)
    visible = (positions + 1) // ratio
    # Sorted logical selections padded with -1; some beyond the visible length or the table.
    scores = torch.rand(tokens, max_positions + 64, device="cuda", generator=generator)
    logical = scores.topk(TOPK, dim=1).indices.sort(dim=1).values.int()
    logical.masked_fill_(torch.rand(logical.shape, device="cuda", generator=generator) < 0.05, -1)
    mapping = (table, page_size, max_positions, requests.long(), visible.long())
    slots = CSA2TrtllmMetadata._map_global_slots(
        logical, requests.long(), table, visible.long(), page_size, max_positions
    )
    return tokens, physical_pages * page_size, logical, mapping, slots


def _mixed_batch():
    """Requests whose rows give every CTA two whole tokens and leave a few tokens split across CTAs."""
    return torch.cuda.get_device_properties(0).multi_processor_count // 3 + 1


@pytest.mark.parametrize("ratio", [1, 2])
@torch.inference_mode()
def test_fused_decode_maps_logical_main_selections(ratio):
    """Ratio two pages hold 64 compressed rows and ratio one 128, as in the release layout."""
    page_size = 128 // ratio
    generator = torch.Generator(device="cuda").manual_seed(4000 + ratio)
    tokens, main_rows, logical, mapping, slots = _logical_case(32, page_size, ratio, generator)
    swa_pool, main_pool = _pools(2048, main_rows, seed=10 * ratio)
    swa, _, _ = _selections(tokens, 2048, 16, generator, holes=0.05)
    q = torch.randn(tokens, HEADS, DIM, device="cuda", generator=generator).bfloat16()
    sinks = torch.randn(HEADS, device="cuda", generator=generator)
    mapped = _decode(q, swa_pool, swa, sinks, main_pool, logical, mapping=mapping)
    # In-kernel mapping selects exactly the rows the production mapping resolves.
    physical = _decode(q, swa_pool, swa, sinks, main_pool, slots.int())
    assert torch.equal(mapped, physical)


@torch.inference_mode()
def test_fused_decode_fp8_query():
    """E4M3 Q (whole and split tokens) gives the bytes of its exact BF16 conversion."""
    generator = torch.Generator(device="cuda").manual_seed(6000)
    tokens, main_rows, logical, mapping, _ = _logical_case(_mixed_batch(), 64, 2, generator)
    swa_pool, main_pool = _pools(2048, main_rows, seed=6)
    swa, _, _ = _selections(tokens, 2048, 16, generator, holes=0.05)
    sinks = torch.randn(HEADS, device="cuda", generator=generator)
    q = torch.randn(tokens, HEADS, DIM, device="cuda", generator=generator).to(torch.float8_e4m3fn)
    actual = _decode(q, swa_pool, swa, sinks, main_pool, logical, mapping=mapping)
    expected = _decode(q.bfloat16(), swa_pool, swa, sinks, main_pool, logical, mapping=mapping)
    assert torch.equal(actual, expected)


@torch.inference_mode()
def test_fused_decode_mxfp8_output():
    """The MXFP8 output is inverse_rope_mxfp8 of the BF16 output byte for byte (values and every
    scale, padding rows included), for whole and split tokens."""
    from tensorrt_llm._torch.attention.backends.sparse.csa2.projection import inverse_rope_mxfp8

    generator = torch.Generator(device="cuda").manual_seed(7000)
    tokens, main_rows, logical, mapping, _ = _logical_case(_mixed_batch(), 64, 2, generator)
    swa_pool, main_pool = _pools(2048, main_rows, seed=7)
    swa, _, _ = _selections(tokens, 2048, 16, generator, holes=0.05)
    sinks = torch.randn(HEADS, device="cuda", generator=generator)
    q = torch.randn(tokens, HEADS, DIM, device="cuda", generator=generator).to(torch.float8_e4m3fn)
    positions = torch.randint(0, 4096, (tokens,), device="cuda", generator=generator).int()
    angles = torch.rand(4096, 32, device="cuda", generator=generator) * 6.2832
    cos_sin = torch.cat((angles.cos(), angles.sin()), 1)
    table, page_size, max_positions, requests, visible = mapping
    bf16 = _decode(q, swa_pool, swa, sinks, main_pool, logical, mapping=mapping)
    data, scale = inverse_rope_mxfp8(bf16.flatten(1), positions, cos_sin, 8)
    out = torch.empty_like(data)
    out_scale = torch.full_like(scale, 0x55)
    torch.ops.trtllm.csa2_sparse_decode(
        q,
        swa_pool,
        swa,
        main_pool,
        logical,
        table,
        requests,
        visible,
        page_size,
        max_positions,
        sinks,
        SCALE,
        out,
        csa2_backend._fused_decode_workspace(q.device.index),
        out_scale,
        positions,
        cos_sin,
    )
    assert torch.equal(out.view(torch.uint8), data.view(torch.uint8))
    assert torch.equal(out_scale, scale)


@torch.inference_mode()
def test_fused_decode_graph_replay():
    """Graph replays and eager launches, back to back on changing inputs, agree byte for byte."""
    generator = torch.Generator(device="cuda").manual_seed(5000)
    page_size = 64
    tokens, main_rows, logical, mapping, _ = _logical_case(_mixed_batch(), page_size, 2, generator)
    swa_pool, main_pool = _pools(2048, main_rows, seed=11)
    swa, _, _ = _selections(tokens, 2048, 16, generator, holes=0.05)
    q = torch.randn(tokens, HEADS, DIM, device="cuda", generator=generator).bfloat16()
    sinks = torch.randn(HEADS, device="cuda", generator=generator)
    table, _, max_positions, requests, visible = mapping
    output = torch.empty_like(q)
    workspace = csa2_backend._fused_decode_workspace(q.device.index)

    def run():
        torch.ops.trtllm.csa2_sparse_decode(
            q,
            swa_pool,
            swa,
            main_pool,
            logical,
            table,
            requests,
            visible,
            page_size,
            max_positions,
            sinks,
            SCALE,
            output,
            workspace,
        )

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        run()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for step in range(8):
        # New queries, selections and visibility between replays.
        q.normal_(generator=generator)
        swa.copy_(swa.roll(step + 1, dims=1))
        logical.copy_(logical.roll(1, dims=0))
        visible.sub_(step)
        graph.replay()
        replayed = output.clone()
        run()
        assert torch.equal(replayed, output)
    torch.cuda.synchronize()


# --------------------------------------------------------------------------------------------------
# Module generation routing
# --------------------------------------------------------------------------------------------------


def _attention_layers(layout):
    from tensorrt_llm._torch.attention.backends.interface import (
        PositionalEmbeddingParams,
        RopeParams,
    )
    from tensorrt_llm._torch.attention.backends.sparse.csa2.module import DeepseekV41Attention
    from tensorrt_llm._torch.attention.backends.sparse.csa2.params import CSA2Params
    from tensorrt_llm.functional import PositionEmbeddingType

    torch.manual_seed(77)
    layers = []
    with torch.device("cuda"):
        for layer_idx in range(len(layout.compress_ratios)):
            model = DeepseekV41Attention(
                layout,
                layer_idx,
                PositionalEmbeddingParams(
                    type=PositionEmbeddingType.rope_gptj,
                    rope=RopeParams(dim=64, theta=160000, max_positions=DSPARK_MAX_SEQ),
                    is_neox=False,
                ),
                hidden_size=128,
                num_heads=HEADS,
                q_lora_rank=128,
                o_lora_rank=128,
                num_groups=8,
                index_heads=32,
                sparse_params=CSA2Params(layout=layout),
                projection_quantization="bf16",
                kv_cache_dtype="fp8",
            )
            for name, parameter in model.named_parameters():
                if name.endswith("norm.weight"):
                    parameter.data.fill_(1.0)
                elif parameter.dtype.is_floating_point:
                    parameter.data.copy_(torch.randn_like(parameter, dtype=torch.float32) * 0.05)
            model.attn_sink.data.normal_()
            layers.append(model)
    return layers


def _forward(layers, metadata, hidden):
    """One packed model forward through every layer; returns each layer's attention output."""
    attention = []
    metadata.reset_routing()
    for model in layers:
        project = model._project_output

        def capture(output, positions, **kwargs):
            attention.append(output.clone())
            return project(output, positions, **kwargs)

        model._project_output = capture
        try:
            model(hidden, metadata.csa2_positions, metadata)
        finally:
            del model._project_output
    return attention


def _prepare(metadata, requests, contexts, lengths, cached):
    from tensorrt_llm._torch.metadata import KVCacheParams

    metadata.request_ids = [request.py_request_id for request in requests]
    metadata.num_contexts = contexts
    metadata.prompt_lens = [request.prompt_len for request in requests]
    metadata.seq_lens = torch.tensor(lengths, dtype=torch.int32)
    metadata.kv_cache_params = KVCacheParams(use_cache=True, num_cached_tokens_per_seq=cached)
    metadata.prepare()


def _record_fused_calls(monkeypatch):
    """Record the inputs of every fused decode launch (cloned) for an independent reference."""
    calls = []
    op = torch.ops.trtllm.csa2_sparse_decode

    def recorder(q, swa_pool, swa, main_pool, main, table, requests, visible, *args):
        cloned = [
            None if tensor is None else tensor.clone()
            for tensor in (q, swa_pool, swa, main_pool, main, table, requests, visible)
        ]
        calls.append((*cloned, args[0], args[1], args[2]))
        return op(q, swa_pool, swa, main_pool, main, table, requests, visible, *args)

    monkeypatch.setattr(torch.ops.trtllm, "csa2_sparse_decode", recorder)
    return calls


def _call_reference(call):
    q, swa_pool, swa, main_pool, main, table, requests, visible, page_size, limit, sinks = call
    slots = main
    if table is not None:
        slots = CSA2TrtllmMetadata._map_global_slots(
            main, requests, table, visible, page_size, limit
        )
    return _reference(q, swa_pool, swa, main_pool, slots, sinks)


def _rel(a, b):
    return float((a.float() - b.float()).norm() / b.float().norm())


@pytest.fixture
def dspark_cache(request):
    """SWA-only, Full, Reindex and Reuse layers with index_topk 512 and DSpark-style requests.

    Four requests generate DSPARK_ROWS rows each. Their cached prompts reach up
    to three times index_topk visible compressed positions, so the Full and
    Reindex indexers run a real top-k; one short prompt keeps every position.
    A fifth, fresh request supplies the context rows of a mixed iteration.
    """
    from tensorrt_llm._torch.attention.backends.sparse.csa2.cache_manager import CSA2CacheManager
    from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest
    from tensorrt_llm.bindings import DataType, SamplingConfig
    from tensorrt_llm.bindings.internal.batch_manager import CacheType
    from tensorrt_llm.llmapi.llm_args import DraftTargetDecodingConfig, KvCacheConfig
    from tensorrt_llm.mapping import Mapping

    ratio = request.param
    layout = CSA2Layout((0, ratio, ratio, ratio), (1,), (1, 2), index_topk=TOPK, window_size=WINDOW)
    manager = CSA2CacheManager(
        KvCacheConfig(
            enable_block_reuse=False,
            max_tokens=16384,
            enable_swa_scratch_reuse=True,
            dtype="fp8",
        ),
        CacheType.SELFKONLY,
        layout=layout,
        num_layers=4,
        tokens_per_block=128,
        mapping=Mapping(),
        max_seq_len=DSPARK_MAX_SEQ,
        max_batch_size=5,
        max_input_len=1984,
        max_num_tokens=4096,
        dtype=DataType.BF16,
        vocab_size=1024,
        spec_config=DraftTargetDecodingConfig(
            max_draft_len=DSPARK_ROWS - 1, speculative_model="csa2-fused-decode-test-draft"
        ),
    )
    requests = []
    for request_id, length in zip((21, 22, 23, 24, 25), (1543, 1100, 527, 46, 300)):
        llm_request = LlmRequest(
            request_id=request_id,
            max_new_tokens=16,
            input_tokens=list(range(length)),
            sampling_config=SamplingConfig(),
            is_streaming=False,
        )
        assert manager.prepare_context(llm_request)
        assert manager.resize_context(llm_request, llm_request.context_chunk_size)
        requests.append(llm_request)
    manager._stream.synchronize()
    yield manager, requests
    for llm_request in requests:
        manager.free_resources(llm_request)
    manager.shutdown()


def _dspark_prefill(layers, manager, requests):
    """Prefill the generating requests up to their last DSPARK_ROWS positions."""
    generating = requests[:4]
    cached = [request.prompt_len - DSPARK_ROWS for request in generating]
    metadata = CSA2TrtllmMetadata(max_num_requests=5, max_num_tokens=4096, kv_cache_manager=manager)
    _prepare(metadata, generating, 4, cached, [0] * 4)
    _forward(layers, metadata, torch.randn(sum(cached), 128, device="cuda", dtype=torch.bfloat16))
    return cached


def _dspark_step(manager, requests, cached, mixed, cuda_graph=False):
    """Fresh metadata for one verification step (the fresh request prefills first when mixed)."""
    generating, fresh = requests[:4], requests[4]
    metadata = CSA2TrtllmMetadata(max_num_requests=5, max_num_tokens=4096, kv_cache_manager=manager)
    metadata.is_spec_decoding_enabled = True
    metadata.is_cuda_graph = cuda_graph
    batch = ([fresh] if mixed else []) + generating
    lengths = ([fresh.prompt_len] if mixed else []) + [DSPARK_ROWS] * len(generating)
    _prepare(metadata, batch, int(mixed), lengths, ([0] if mixed else []) + cached)
    return metadata, sum(lengths)


def _check_dspark_selections(calls):
    """The recorded launches carry the index top-k of the Full and Reindex layers."""
    swa, full, reindex, reuse = calls
    assert swa[4] is None
    for call in (full, reindex, reuse):
        # Logical selections, mapped by the kernel through the owner's page table.
        assert call[5] is not None and call[4].shape[1] == TOPK
    logical, visible = full[4].long(), full[7]
    assert (visible > TOPK).any() and (visible < TOPK).any(), visible
    for call in (full, reindex):
        valid = (call[4] >= 0) & (call[4].long() < visible[:, None])
        # Every row selects min(visible, index_topk) distinct visible positions.
        torch.testing.assert_close(valid.sum(1), visible.clamp(max=TOPK), atol=0, rtol=0)
        assert all(
            row[mask].unique().numel() == int(mask.sum()) for row, mask in zip(call[4], valid)
        )
    # The Reindex layer runs its own indexer; the Reuse layer consumes its selections.
    long_rows = visible > TOPK
    assert not torch.equal(reindex[4][long_rows], logical[long_rows].int())
    assert torch.equal(reuse[4], reindex[4])


@pytest.mark.parametrize("dspark_cache", [1, 2], indirect=True)
@torch.inference_mode()
def test_module_dspark_generation_with_reindex(monkeypatch, dspark_cache):
    """DSpark verification rows through Full, Reindex and Reuse layers with a real index top-k."""
    manager, requests = dspark_cache
    layers = _attention_layers(manager.layout)
    assert [model.layer.mode for model in layers] == [
        CSA2Mode.SWA,
        CSA2Mode.FULL,
        CSA2Mode.REINDEX,
        CSA2Mode.REUSE,
    ]
    assert all(model.backend.fused_decode for model in layers)
    cached = _dspark_prefill(layers, manager, requests)
    generation_rows = DSPARK_ROWS * len(cached)
    # Pure verification, then a mixed iteration in which a fresh prompt prefills first.
    for mixed in (False, True):
        metadata, tokens = _dspark_step(manager, requests, cached, mixed)
        step = torch.randn(tokens, 128, device="cuda", dtype=torch.bfloat16)
        calls = _record_fused_calls(monkeypatch)
        fused = _forward(layers, metadata, step)
        monkeypatch.undo()
        assert len(calls) == len(layers)
        assert all(call[0].shape[0] == generation_rows for call in calls)
        _check_dspark_selections(calls)
        # The staged baseline runs the same step on its own metadata.
        staged_metadata, _ = _dspark_step(manager, requests, cached, mixed)
        for model in layers:
            model.backend.fused_decode = False
        staged = _forward(layers, staged_metadata, step)
        for model in layers:
            model.backend.fused_decode = True
        assert any(not key[4] for key in staged_metadata._csa2_query_tiles)
        for layer_idx, (new, old, call) in enumerate(zip(fused, staged, calls)):
            rows = slice(new.shape[0] - generation_rows, new.shape[0])
            torch.testing.assert_close(new[: rows.start], old[: rows.start], atol=0, rtol=0)
            reference = _call_reference(call)
            _check_close(new[rows], reference)
            # Staged attention reads the same keys: only its staging precision differs.
            assert _rel(old[rows], reference) < 0.1, layer_idx
            assert _rel(new[rows], reference) < 0.5 * _rel(old[rows], reference), layer_idx


@pytest.mark.parametrize("dspark_cache", [2], indirect=True)
@torch.inference_mode()
def test_module_dspark_generation_graph(monkeypatch, dspark_cache):
    """A captured verification step: every replayed launch matches FP32 on the inputs it received."""
    manager, requests = dspark_cache
    layers = _attention_layers(manager.layout)
    cached = _dspark_prefill(layers, manager, requests)
    metadata, tokens = _dspark_step(manager, requests, cached, mixed=False, cuda_graph=True)
    hidden = torch.randn(tokens, 128, device="cuda", dtype=torch.bfloat16)
    for _ in range(2):
        _forward(layers, metadata, hidden)
    assert all(key[4] for key in getattr(metadata, "_csa2_query_tiles", {}))
    # Cloned inside the capture, the recorded inputs are refreshed by every replay.
    calls = _record_fused_calls(monkeypatch)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = _forward(layers, metadata, hidden)
    monkeypatch.undo()
    assert len(calls) == len(layers)
    for _ in range(2):
        hidden.normal_()
        graph.replay()
        _check_dspark_selections(calls)
        for output, call in zip(captured, calls):
            _check_close(output, _call_reference(call))
    torch.cuda.synchronize()


@pytest.mark.parametrize("dspark_cache", [2], indirect=True)
@torch.inference_mode()
def test_module_generation_mxfp8_output(monkeypatch, dspark_cache):
    """64-head MXFP8 layers on FP8 staging: the projection's E4M3 Q feeds the kernel, whose MXFP8
    o_a input equals inverse_rope_mxfp8 of the same launch's BF16 output and projects bitwise
    like it."""
    from tensorrt_llm._torch.attention.backends.sparse.csa2.projection import inverse_rope_mxfp8
    from tensorrt_llm._torch.modules.linear import (
        flashinfer_mxfp8_autotune,
        flashinfer_mxfp8_decode_graph_capture,
    )

    from .test_module import _native_model

    manager, requests = dspark_cache
    layers = []
    for layer_idx in range(len(manager.layout.compress_ratios)):
        model, _ = _native_model(
            grouped_block=128,
            layout=manager.layout,
            attention_heads=HEADS,
            num_groups=8,
            kv_cache_dtype="fp8",
            layer_idx=layer_idx,
            max_positions=DSPARK_MAX_SEQ,
        )
        assert model.backend.fused_decode and model._o_a_mxfp8_scale is not None
        model._fp8_q_min_tokens = 1
        for linear in (model.wq_a, model.wq_b, model.wkv):
            linear.quant_method.tune_mxfp8_backends = True
        layers.append(model)
    cached = _dspark_prefill(layers, manager, requests)
    metadata, tokens = _dspark_step(manager, requests, cached, mixed=False)
    hidden = torch.randn(tokens, 128, device="cuda", dtype=torch.bfloat16)
    calls = []
    op = torch.ops.trtllm.csa2_sparse_decode

    def recorder(*args):
        calls.append(
            [
                arg.clone() if index < 8 and arg is not None else arg
                for index, arg in enumerate(args)
            ]
        )
        return op(*args)

    def step():
        metadata.reset_routing()
        return [model(hidden, metadata.csa2_positions, metadata) for model in layers]

    # Decode steps project as the executor's decode graphs do, which is when the Q-B GEMM
    # writes the E4M3 Q.
    with flashinfer_mxfp8_decode_graph_capture():
        with flashinfer_mxfp8_autotune():
            step()
        monkeypatch.setattr(torch.ops.trtllm, "csa2_sparse_decode", recorder)
        projected = step()
        monkeypatch.undo()
        assert len(calls) == len(layers)
        for model, actual, call in zip(layers, projected, calls):
            assert len(call) == 17 and call[0].dtype == torch.float8_e4m3fn
            output, workspace, scale, positions, cos_sin = call[12:]
            bf16 = torch.empty(call[0].shape, dtype=torch.bfloat16, device="cuda")
            op(*call[:12], bf16, workspace)
            data, expected_scale = inverse_rope_mxfp8(bf16.flatten(1), positions, cos_sin, 8)
            assert torch.equal(output.view(torch.uint8), data.view(torch.uint8))
            assert torch.equal(scale, expected_scale)
            expected = model._project_output_mxfp8(bf16.flatten(1), metadata.csa2_positions)
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)


# SWA, an index-ratio-2 staging group (Full + two Reuse) and a ratio-1 group (Full + Reuse).
GEN_FMHA_LAYOUT = CSA2Layout(
    (0, 2, 2, 2, 1, 1), (1, 4), (1, 4), index_topk=TOPK, window_size=WINDOW
)
GEN_FMHA_LAYERS = [False, True, True, True, False, False]
# Enough DSpark requests for a decode step of _GEN_FMHA_MIN_ROWS rows.
GEN_FMHA_REQUESTS = -(-csa2_backend._GEN_FMHA_MIN_ROWS // DSPARK_ROWS)
GEN_FMHA_PROMPTS = (1543, 1100, 527, 1300, 300, 900)


@pytest.fixture
def large_decode_cache():
    from tensorrt_llm._torch.attention.backends.sparse.csa2.cache_manager import CSA2CacheManager
    from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest
    from tensorrt_llm.bindings import DataType, SamplingConfig
    from tensorrt_llm.bindings.internal.batch_manager import CacheType
    from tensorrt_llm.llmapi.llm_args import DraftTargetDecodingConfig, KvCacheConfig
    from tensorrt_llm.mapping import Mapping

    manager = CSA2CacheManager(
        # Every request stays in its context while the step runs: a byte budget that holds all of
        # their SWA rows.
        KvCacheConfig(
            enable_block_reuse=False,
            max_gpu_total_bytes=16 << 30,
            enable_swa_scratch_reuse=False,
            dtype="fp8",
        ),
        CacheType.SELFKONLY,
        layout=GEN_FMHA_LAYOUT,
        num_layers=len(GEN_FMHA_LAYOUT.compress_ratios),
        tokens_per_block=128,
        mapping=Mapping(),
        max_seq_len=DSPARK_MAX_SEQ,
        max_batch_size=GEN_FMHA_REQUESTS,
        max_input_len=max(GEN_FMHA_PROMPTS),
        max_num_tokens=8192,
        dtype=DataType.BF16,
        vocab_size=1024,
        spec_config=DraftTargetDecodingConfig(
            max_draft_len=DSPARK_ROWS - 1, speculative_model="csa2-gen-fmha-test-draft"
        ),
    )
    requests = []
    for index in range(GEN_FMHA_REQUESTS):
        request = LlmRequest(
            request_id=100 + index,
            max_new_tokens=16,
            input_tokens=list(range(GEN_FMHA_PROMPTS[index % len(GEN_FMHA_PROMPTS)])),
            sampling_config=SamplingConfig(),
            is_streaming=False,
        )
        assert manager.prepare_context(request)
        assert manager.resize_context(request, request.context_chunk_size)
        requests.append(request)
    manager._stream.synchronize()
    yield manager, requests
    for request in requests:
        manager.free_resources(request)
    manager.shutdown()


def _large_decode_step(manager, requests, cached, cuda_graph=False):
    metadata = CSA2TrtllmMetadata(
        max_num_requests=len(requests), max_num_tokens=8192, kv_cache_manager=manager
    )
    metadata.is_spec_decoding_enabled = True
    metadata.is_cuda_graph = cuda_graph
    _prepare(metadata, requests, 0, [DSPARK_ROWS] * len(requests), cached)
    return metadata


@pytest.mark.skipif(
    not torch.cuda.is_available()
    or not gen_fmha.gen_fmha_device_supported(torch.cuda.current_device()),
    reason="the gen_fmha core needs sm_103",
)
@torch.inference_mode()
def test_ratio2_group_large_decode_on_gen_fmha(monkeypatch, large_decode_cache):
    """A large decode step: the ratio-2 staging group attends with the gen_fmha core within the staged
    cubin's error, the other layers keep the fused decode, and graph replays equal eager steps."""
    manager, requests = large_decode_cache
    layers = _attention_layers(manager.layout)
    assert [model.backend.gen_fmha_layer for model in layers] == GEN_FMHA_LAYERS
    cached = [request.prompt_len - DSPARK_ROWS for request in requests]
    start = 0
    while start < len(requests):
        # Prefill in batches of at most 8192 tokens, up to each request's last DSPARK_ROWS positions.
        end = start
        while end < len(requests) and sum(cached[start : end + 1]) <= 8192:
            end += 1
        metadata = CSA2TrtllmMetadata(
            max_num_requests=end - start, max_num_tokens=8192, kv_cache_manager=manager
        )
        _prepare(metadata, requests[start:end], end - start, cached[start:end], [0] * (end - start))
        tokens = sum(cached[start:end])
        _forward(layers, metadata, torch.randn(tokens, 128, device="cuda", dtype=torch.bfloat16))
        start = end
    rows = DSPARK_ROWS * len(requests)
    assert rows >= csa2_backend._GEN_FMHA_MIN_ROWS
    hidden = torch.randn(rows, 128, device="cuda", dtype=torch.bfloat16)
    cores = []
    core = gen_fmha.gen_sparse_attention

    def record_core(*args, **kwargs):
        cores.append(1)
        return core(*args, **kwargs)

    # The fused decode on every layer gives each layer's exact inputs for the FP32 reference.
    with monkeypatch.context() as patch:
        patch.setattr(csa2_backend, "_GEN_FMHA_MIN_ROWS", 1 << 30)
        calls = _record_fused_calls(patch)
        fused = _forward(layers, _large_decode_step(manager, requests, cached), hidden)
    assert len(calls) == len(layers)
    # The staged native cubin on the same inputs sets the error the core must stay within.
    with monkeypatch.context() as patch:
        patch.setattr(csa2_backend, "_GEN_FMHA_MIN_ROWS", 1 << 30)
        for model in layers:
            patch.setattr(model.backend, "fused_decode", False)
        cubin = _forward(layers, _large_decode_step(manager, requests, cached), hidden)
    monkeypatch.setattr(gen_fmha, "gen_sparse_attention", record_core)
    with monkeypatch.context() as patch:
        kept = _record_fused_calls(patch)
        mixed = _forward(layers, _large_decode_step(manager, requests, cached), hidden)
    # Only the ratio-2 group leaves the fused decode for the core.
    assert len(cores) == sum(GEN_FMHA_LAYERS) and len(kept) == len(layers) - sum(GEN_FMHA_LAYERS)
    for layer_idx, (output, call) in enumerate(zip(mixed, calls)):
        if not GEN_FMHA_LAYERS[layer_idx]:
            assert torch.equal(output, fused[layer_idx]), layer_idx
            continue
        assert torch.isfinite(output).all(), layer_idx
        reference = _call_reference(call)
        error, cubin_error = _rel(output, reference), _rel(cubin[layer_idx], reference)
        assert error <= max(1.25 * cubin_error, 2e-2), (layer_idx, error, cubin_error)
        drift, cubin_drift = (
            _rel(output, fused[layer_idx]),
            _rel(cubin[layer_idx], fused[layer_idx]),
        )
        assert drift <= max(1.25 * cubin_drift, 2e-2), (layer_idx, drift, cubin_drift)

    # A captured step replays what an eager step of the same inputs computes.
    metadata = _large_decode_step(manager, requests, cached, cuda_graph=True)
    for _ in range(2):
        _forward(layers, metadata, hidden)
    cores.clear()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = _forward(layers, metadata, hidden)
    assert len(cores) == sum(GEN_FMHA_LAYERS)
    for _ in range(2):
        hidden.normal_()
        graph.replay()
        replayed = [output.clone() for output in captured]
        eager = _forward(layers, metadata, hidden)
        for layer_idx, (expected, actual) in enumerate(zip(eager, replayed)):
            assert torch.equal(actual, expected), layer_idx
    torch.cuda.synchronize()
