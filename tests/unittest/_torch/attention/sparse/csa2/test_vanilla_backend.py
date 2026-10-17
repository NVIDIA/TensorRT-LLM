# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CSA2 Vanilla golden: analytic checks and a comparison with the TRTLLM backend.

The analytic tests run the golden on CPU tensors over a stub cache whose page
lists are shuffled and partly unmapped, so page translation, group geometry,
selection and attention are each checked against hand-derived values. The GPU
test drives ``CSA2TrtllmAttention`` and the golden through the same multi-layer
forwards on separate cache managers, the way the CSA2 module calls them.
"""

import inspect
from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.attention.backends.interface import (
    AttentionForwardArgs,
    AttentionInputType,
    MLAParams,
)
from tensorrt_llm._torch.attention.backends.sparse.csa2.backend import CSA2TrtllmAttention
from tensorrt_llm._torch.attention.backends.sparse.csa2.cache_manager import CSA2CacheRole
from tensorrt_llm._torch.attention.backends.sparse.csa2.metadata import CSA2TrtllmMetadata
from tensorrt_llm._torch.attention.backends.sparse.csa2.params import (
    CSA2BackendForwardArgs,
    CSA2ForwardState,
    CSA2Layout,
    CSA2Mode,
    CSA2Params,
    RowTransform,
)
from tensorrt_llm._torch.attention.backends.sparse.csa2.quantization import (
    INDEX_PAGE_ROWS,
    pack_rows,
    read_index_rows,
    row_bytes,
    unpack_rows,
    write_packed_index_rows,
)
from tensorrt_llm._torch.attention.backends.sparse.csa2.vanilla_backend import (
    CSA2VanillaAttention,
    CSA2VanillaIndexer,
)
from tensorrt_llm._torch.attention.backends.sparse.registry import (
    get_vanilla_sparse_attn_attention_backend,
)
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization.mode import QuantAlgo

_HEAD_DIM = 512
_INDEX_DIM = 128


def _reference(q, rows, valid, sink):
    """FP32 sink attention: q [n, H, D], rows [n, R, D], valid [n, R], sink [H] -> [n, H, D]."""
    scores = torch.einsum("nhd,nrd->nhr", q.float(), rows.float()) * _HEAD_DIM**-0.5
    scores = scores.masked_fill(~valid[:, None, :], -torch.inf)
    sinks = sink.float()[None, :, None].expand(q.shape[0], -1, 1)
    probabilities = torch.cat((scores, sinks), dim=-1).softmax(-1)[..., :-1]
    return torch.einsum("nhr,nrd->nhd", probabilities, rows.float())


def _decoded(rows, cache_format):
    """Rows [n, dim] as the cache stores and decodes them."""
    return unpack_rows(pack_rows(rows, cache_format), rows.shape[-1], cache_format)


def _halves(pairs):
    """BF16 [n, 128] index rows whose first and second 64 channels hold (a, b).

    Each value is exactly representable in the MXFP4 index format, so the
    scores the golden computes are exact multiples of 64.
    """
    return torch.tensor([[a] * 64 + [b] * 64 for a, b in pairs], dtype=torch.bfloat16)


class _StubCache:
    """Host page lists and byte pools of a CSA2 cache with one KV owner.

    Args:
        layout: Model layer layout.
        block: Tokens per SWA page; a GLOBAL page holds ``block // r`` entries.
        max_seq_len: Positions a request may address.
        swa_pages: ``{request_id: pages}`` shared by every layer's SWA pool.
        global_pages: ``{request_id: pages}`` of the KV owner's GLOBAL pages.
    """

    def __init__(self, layout, block, max_seq_len, swa_pages, global_pages):
        self.layout = layout
        self.tokens_per_block = block
        self.max_seq_len = max_seq_len
        self.context_swa_layer_limit = None
        self._swa_pages = swa_pages
        self._global_pages = global_pages
        self.swa = [
            torch.zeros(16, row_bytes(_HEAD_DIM, "swa"), dtype=torch.uint8)
            for _ in layout.compress_ratios
        ]
        self.main = torch.zeros(
            2 * INDEX_PAGE_ROWS, row_bytes(_HEAD_DIM, "main"), dtype=torch.uint8
        )
        self.index = torch.zeros(
            2, INDEX_PAGE_ROWS, 1, row_bytes(_INDEX_DIM, "index"), dtype=torch.uint8
        )

    def get_cache_indices(self, request_id, layer_idx, role):
        pages = self._swa_pages if role == CSA2CacheRole.SWA else self._global_pages
        return pages[request_id]

    def get_swa_buffer(self, layer_idx):
        return self.swa[layer_idx]

    def get_main_buffer(self, layer_idx):
        return self.main

    def get_index_pages(self, layer_idx):
        return self.index

    def put_swa(self, layer_idx, slot, row):
        self.swa[layer_idx][slot] = pack_rows(row[None], "swa")[0]

    def put_global(self, slot, main_row, index_row):
        self.main[slot] = pack_rows(main_row[None], "main")[0]
        write_packed_index_rows(
            self.index, torch.tensor([slot]), pack_rows(index_row[None], "index")
        )


def _request_metadata(cache, request_ids, past, new, num_contexts):
    """Request metadata fields the golden reads, for a prepared batch."""
    seq_lens = torch.tensor(new, dtype=torch.int32)
    return SimpleNamespace(
        kv_cache_manager=cache,
        request_ids=list(request_ids),
        seq_lens=seq_lens,
        kv_lens_cuda=seq_lens + torch.tensor(past, dtype=torch.int32),
        num_seqs=len(new),
        num_contexts=num_contexts,
        num_tokens=sum(new),
        csa2_indices={},
        csa2_candidates={},
        csa2_precomputed_kv_layers=set(),
        csa2_replay_mode=None,
        csa2_query_layer_start=0,
        csa2_replay_start_positions=torch.zeros(len(new), dtype=torch.int64),
        _csa2_last_layer=-1,
    )


def _call(backend, q, state, start, end, sink, phase):
    """Run one phase tile [start, end) of the model forward through ``backend``."""
    output = torch.empty(end - start, q.shape[1] * _HEAD_DIM, dtype=q.dtype)
    backend.forward(
        q[start:end].flatten(1),
        None,
        None,
        SimpleNamespace(num_tokens=end - start, num_query_heads=q.shape[1]),
        forward_args=AttentionForwardArgs(
            attention_input_type=phase,
            attention_sinks=sink,
            output=output,
            sparse_backend_args=CSA2BackendForwardArgs(state=state, query_start=start),
        ),
    )
    return output.view(end - start, q.shape[1], _HEAD_DIM)


def _ratio_two_scenario():
    """A ratio-2 owner, its REUSE consumer and a SWA layer over two requests.

    Layout (2, 2, 0) with window 4, ``index_topk`` 2, 4-token SWA pages and
    2-entry GLOBAL pages. Request 7 is a context chunk at positions 9..12 after
    9 cached tokens, so its first new group also holds cached position 8;
    request 9 is one generation token at position 3. Page lists are shuffled.
    Request 7's SWA positions 0..7 and GLOBAL entries 0, 1 have no page, and
    request 9's GLOBAL page lies in the second 64-row index page.
    """
    layout = CSA2Layout((2, 2, 0), (0,), (0,), index_topk=2, window_size=4)
    cache = _StubCache(
        layout,
        block=4,
        max_seq_len=16,
        swa_pages={7: [-1, -1, 0, 3], 9: [2, -1, -1, -1]},
        global_pages={7: [-1, 4, 2, -1], 9: [40, -1, -1, -1]},
    )
    generator = torch.Generator().manual_seed(1203)

    def randn(*shape):
        return torch.randn(*shape, generator=generator).bfloat16()

    # Cached SWA history of every layer: position 8 of request 7 (row 0) and
    # positions 0..2 of request 9 (rows 8..10).
    swa_rows_of = {(7, 8): 0, (9, 0): 8, (9, 1): 9, (9, 2): 10}
    cached_swa = {(layer, *token): randn(_HEAD_DIM) for layer in range(3) for token in swa_rows_of}
    for (layer, request, position), row in cached_swa.items():
        cache.put_swa(layer, swa_rows_of[request, position], row)
    # Cached groups: entries 2, 3 of request 7 (rows 8, 9) and 0 of request 9
    # (row 80). With the negative head weight below, an entry with constant
    # index key v scores -128 * v, so request 7's entries 2..5 (keys 3, 0.5,
    # 2, 1) rank 3, 5, 4, 2.
    cached_main = {(7, 2): randn(_HEAD_DIM), (7, 3): randn(_HEAD_DIM), (9, 0): randn(_HEAD_DIM)}
    cache.put_global(8, cached_main[7, 2], _halves([(3.0, 3.0)])[0])
    cache.put_global(9, cached_main[7, 3], _halves([(0.5, 0.5)])[0])
    cache.put_global(80, cached_main[9, 0], _halves([(1.0, 1.0)])[0])
    metadata = _request_metadata(cache, (7, 9), past=(9, 3), new=(4, 1), num_contexts=1)
    inputs = {
        "q": randn(5, 2, _HEAD_DIM),
        "sink": torch.randn(2, generator=generator),
        "swa": [randn(5, _HEAD_DIM) for _ in range(3)],
        # Rows 0..2 are request 7's groups 4, 5 and request 9's group 1;
        # rows 3 and 4 pad the compressor output and must never be written.
        "main": randn(5, _HEAD_DIM),
        "index_k": _halves([(2.0, 2.0), (1.0, 1.0), (4.0, 4.0), (6.0, 6.0), (6.0, 6.0)]),
        "index_q": torch.ones(5, 1, _INDEX_DIM, dtype=torch.bfloat16),
        # Unmapped entries would read as zero keys and score 0, the best
        # score here, if page validity were ignored.
        "weights": -torch.ones(5, 1),
    }
    return layout, cache, metadata, inputs, cached_swa, cached_main


def _scenario_states(layout, metadata, inputs):
    """Per-layer forward states of the ratio-2 scenario."""
    states = []
    for layer_idx in range(3):
        kwargs = {}
        mode = layout.layer(layer_idx).mode
        if mode == CSA2Mode.FULL:
            kwargs.update(main_kv=inputs["main"], index_k=inputs["index_k"])
        if mode in (CSA2Mode.FULL, CSA2Mode.REINDEX):
            kwargs.update(index_q=inputs["index_q"], index_weights=inputs["weights"])
        states.append(
            CSA2ForwardState(metadata=metadata, swa_kv=inputs["swa"][layer_idx], **kwargs)
        )
    return states


def test_vanilla_contract_matches_production_backend():
    assert CSA2VanillaAttention.Metadata is CSA2TrtllmMetadata
    assert CSA2VanillaAttention.support_fused_rope()
    assert CSA2VanillaAttention.compute_backend == "vanilla"
    for method in ("__init__", "forward"):
        golden = inspect.signature(getattr(CSA2VanillaAttention, method)).parameters
        production = inspect.signature(getattr(CSA2TrtllmAttention, method)).parameters
        assert list(golden) == list(production)
    attention = CSA2VanillaAttention(3, 2, _HEAD_DIM, kv_cache_dtype="fp8")
    # FP8 staging is accepted, but the golden always stages exact BF16.
    assert attention.staging_dtype == torch.bfloat16 and not attention.native_fp8
    assert attention.indexer is None
    # The CSA2 module hard-codes TRTLLM, so the registry keeps VANILLA closed.
    with pytest.raises(ValueError, match="csa2"):
        get_vanilla_sparse_attn_attention_backend(CSA2Params())


@pytest.mark.parametrize(
    "kwargs, message",
    [
        pytest.param(dict(head_dim=256), "head_dim=512", id="head_dim"),
        pytest.param(dict(num_kv_heads=2), "one KV head", id="kv_heads"),
        pytest.param(dict(kv_cache_dtype="nvfp4"), "KV cache dtype", id="kv_dtype"),
        pytest.param(
            dict(quant_config=QuantConfig(kv_cache_quant_algo=QuantAlgo.NVFP4)),
            "only FP8",
            id="kv_quant",
        ),
        pytest.param(
            dict(
                mla_params=MLAParams(
                    q_lora_rank=1280,
                    kv_lora_rank=512,
                    qk_nope_head_dim=448,
                    qk_rope_head_dim=64,
                    v_head_dim=512,
                    rope_append=False,
                )
            ),
            "rank/nope 448",
            id="mla",
        ),
        pytest.param(
            dict(sparse_params=CSA2Params(use_packed_sparse_attention=True)),
            "packed sparse attention",
            id="packed",
        ),
        pytest.param(
            dict(sparse_params=CSA2Params(compute_backend="cutlass")),
            "Unknown CSA2 attention backend",
            id="compute_backend",
        ),
    ],
)
def test_vanilla_constructor_rejects_unsupported_configs(kwargs, message):
    arguments = dict(layer_idx=0, num_heads=2, head_dim=_HEAD_DIM) | kwargs
    with pytest.raises(ValueError, match=message):
        CSA2VanillaAttention(**arguments)


def test_vanilla_select_keeps_best_finite_entries_sorted():
    scores = torch.tensor(
        [
            [0.5, -torch.inf, 3.0, 1.0, 2.0],
            [-torch.inf, 1.0, -torch.inf, -torch.inf, -torch.inf],
            [-torch.inf] * 5,
        ]
    )
    expected = torch.tensor([[2, 4], [1, -1], [-1, -1]], dtype=torch.int32)
    torch.testing.assert_close(CSA2VanillaIndexer.select(scores, 2), expected)
    # Fewer columns than K pad the row; no columns select nothing.
    torch.testing.assert_close(
        CSA2VanillaIndexer.select(scores[:, :1], 3),
        torch.tensor([[0, -1, -1], [-1, -1, -1], [-1, -1, -1]], dtype=torch.int32),
    )
    torch.testing.assert_close(
        CSA2VanillaIndexer.select(scores[:, :0], 2), torch.full((3, 2), -1, dtype=torch.int32)
    )


def test_vanilla_candidates_pin_newest_block_and_mask_future_entries():
    inf = torch.inf
    # Blocks of 4 entries; two blocks are published per query.
    scores = torch.tensor(
        [
            # Block maxima 5, 4, 1: the newest block {8, 9} replaces block 1.
            [5, 1, 1, 1, 4, 1, 1, 1, 0, 1],
            # Visible 6: later entries are masked and the newest block is {4..7}.
            [1, 2, 3, 1, 9, 4, -inf, -inf, -inf, -inf],
            # Nothing visible.
            [-inf] * 10,
        ]
    )
    visible = torch.tensor([10, 6, 0])
    candidates = CSA2VanillaIndexer.candidates(scores, visible, 2, 4, 8)
    expected = torch.tensor(
        [
            # Blocks 0 and 2 in ascending order; positions 10, 11 do not exist.
            [0, 1, 2, 3, 8, 9, -1, -1],
            # Entries 6, 7 of the newest block are not visible yet.
            [0, 1, 2, 3, 4, 5, -1, -1],
            [-1] * 8,
        ],
        dtype=torch.int32,
    )
    torch.testing.assert_close(candidates, expected)
    # With a budget covering every block, all reachable blocks are listed.
    torch.testing.assert_close(
        CSA2VanillaIndexer.candidates(scores[:1], visible[:1], 4, 4, 12),
        torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, -1, -1]], dtype=torch.int32),
    )


@torch.inference_mode()
def test_vanilla_forward_writes_selects_and_attends_with_independent_geometry():
    layout, cache, metadata, inputs, cached_swa, cached_main = _ratio_two_scenario()
    states = _scenario_states(layout, metadata, inputs)
    backends = [
        CSA2VanillaAttention(layer, 2, _HEAD_DIM, sparse_params=CSA2Params(layout=layout))
        for layer in range(3)
    ]
    outputs = []
    for layer_idx, backend in enumerate(backends):
        context = _call(
            backend,
            inputs["q"],
            states[layer_idx],
            0,
            4,
            inputs["sink"],
            AttentionInputType.context_only,
        )
        generation = _call(
            backend,
            inputs["q"],
            states[layer_idx],
            4,
            5,
            inputs["sink"],
            AttentionInputType.generation_only,
        )
        outputs.append(torch.cat((context, generation)))

    # Request 7's groups 4, 5 (group 4 completes cached position 8) land on
    # its third GLOBAL page (rows 4, 5); request 9's group 1 on page 40 (row
    # 81, in the second index page). Padding rows 3, 4 are never written.
    written = torch.tensor([4, 5, 81])
    torch.testing.assert_close(cache.main[written], pack_rows(inputs["main"][:3], "main"))
    torch.testing.assert_close(
        read_index_rows(cache.index, written), pack_rows(inputs["index_k"][:3], "index")
    )
    untouched = torch.ones(cache.main.shape[0], dtype=torch.bool)
    untouched[torch.tensor([4, 5, 8, 9, 80, 81])] = False
    assert not cache.main[untouched].any()
    # New SWA rows: request 7's positions 9..12 (rows 1, 2, 3, 12), request 9's 3 (row 11).
    for layer_idx in range(3):
        torch.testing.assert_close(
            cache.swa[layer_idx][torch.tensor([1, 2, 3, 12, 11])],
            pack_rows(inputs["swa"][layer_idx], "swa"),
        )

    # Positions 9, 10 see entries 0..4 and positions 11, 12 entries 0..5, of
    # which 0 and 1 are unmapped; request 9 sees its two entries.
    expected_indices = torch.tensor([[3, 4], [3, 4], [3, 5], [3, 5], [0, 1]], dtype=torch.int32)
    torch.testing.assert_close(metadata.csa2_indices[0], expected_indices)
    assert set(metadata.csa2_indices) == {0}

    new_main = _decoded(inputs["main"][:3], "main")
    main_rows = {key: _decoded(row[None], "main")[0] for key, row in cached_main.items()}
    main_rows.update({(7, 4): new_main[0], (7, 5): new_main[1], (9, 1): new_main[2]})
    tokens = [(7, 9), (7, 10), (7, 11), (7, 12), (9, 3)]
    # Mapped window positions: request 7's positions 0..7 have no SWA page.
    windows = [[8, 9], [8, 9, 10], [8, 9, 10, 11], [9, 10, 11, 12], [0, 1, 2, 3]]
    for layer_idx in range(3):
        new_swa = _decoded(inputs["swa"][layer_idx], "swa")
        swa_rows = {
            key[1:]: _decoded(row[None], "swa")[0]
            for key, row in cached_swa.items()
            if key[0] == layer_idx
        }
        swa_rows.update({token: new_swa[row] for row, token in enumerate(tokens)})
        for row, ((request, _), window) in enumerate(zip(tokens, windows)):
            keys = [swa_rows[request, p] for p in window]
            if layout.layer(layer_idx).mode != CSA2Mode.SWA:
                keys += [main_rows[request, int(e)] for e in expected_indices[row]]
            keys = torch.stack(keys)[None]
            expected = _reference(
                inputs["q"][row : row + 1],
                keys,
                torch.ones(keys.shape[:2], dtype=torch.bool),
                inputs["sink"],
            )
            torch.testing.assert_close(
                outputs[layer_idx][row : row + 1].float(), expected, atol=2e-2, rtol=2e-2
            )

    # A layer cannot run again in the same forward: REUSE could read stale routing.
    with pytest.raises(ValueError, match="reused across forwards"):
        _call(
            backends[0],
            inputs["q"],
            states[0],
            0,
            4,
            inputs["sink"],
            AttentionInputType.context_only,
        )


@torch.inference_mode()
def test_vanilla_reindex_consumer_selects_inside_published_candidates():
    # Layer 0 (ratio 1) owns the cache and publishes candidates; layer 1
    # reindexes the same keys within them. Entry halves (a, b): layer 0 scores
    # a, layer 1 scores b.
    layout = CSA2Layout(
        (1, 1),
        (0,),
        (0, 1),
        candidate_source_layer_id=0,
        candidate_topk_blocks=2,
        candidate_block_size=2,
        index_topk=1,
        window_size=2,
    )
    cache = _StubCache(
        layout, block=4, max_seq_len=8, swa_pages={3: [2, 3]}, global_pages={3: [1, 0]}
    )
    keys = _halves([(0.5, 1.0), (3.0, 0.5), (1.0, 6.0), (0.5, 4.0), (2.0, 2.0), (1.5, 3.0)])
    for entry in range(4):
        # Entries 0..3 sit on GLOBAL page 1 (rows 4..7), entries 4.. on page 0.
        cache.put_global(4 + entry, torch.zeros(_HEAD_DIM, dtype=torch.bfloat16), keys[entry])
    metadata = _request_metadata(cache, (3,), past=(4,), new=(2,), num_contexts=1)
    swa = torch.zeros(2, _HEAD_DIM, dtype=torch.bfloat16)
    first_half = torch.cat((torch.ones(64), torch.zeros(64))).bfloat16().expand(2, 1, -1)
    states = [
        CSA2ForwardState(
            metadata=metadata,
            swa_kv=swa,
            main_kv=torch.zeros(2, _HEAD_DIM, dtype=torch.bfloat16),
            index_k=keys[4:],
            index_q=first_half,
            index_weights=torch.ones(2, 1),
        ),
        CSA2ForwardState(
            metadata=metadata,
            swa_kv=swa,
            index_q=first_half.flip(-1),
            index_weights=torch.ones(2, 1),
        ),
    ]
    q = torch.zeros(2, 2, _HEAD_DIM, dtype=torch.bfloat16)
    for layer_idx in range(2):
        backend = CSA2VanillaAttention(
            layer_idx, 2, _HEAD_DIM, sparse_params=CSA2Params(layout=layout)
        )
        _call(backend, q, states[layer_idx], 0, 2, None, AttentionInputType.context_only)

    # Position 4 sees entries 0..4: block {0, 1} holds the best entry and the
    # newest block {4, 5} is pinned, with entry 5 not yet visible.
    torch.testing.assert_close(
        metadata.csa2_candidates[0],
        torch.tensor([[0, 1, 4, -1], [0, 1, 4, 5]], dtype=torch.int32),
    )
    torch.testing.assert_close(
        metadata.csa2_indices[0], torch.tensor([[1], [1]], dtype=torch.int32)
    )
    # Entry 2 scores highest for layer 1 but is not a candidate.
    torch.testing.assert_close(
        metadata.csa2_indices[1], torch.tensor([[4], [5]], dtype=torch.int32)
    )


@torch.inference_mode()
def test_vanilla_attention_only_contract_decodes_pool_rows():
    generator = torch.Generator().manual_seed(77)
    q = torch.randn(3, 2, _HEAD_DIM, generator=generator).bfloat16()
    swa = torch.randn(3, 4, _HEAD_DIM, generator=generator).bfloat16()
    main = torch.randn(3, 3, _HEAD_DIM, generator=generator).bfloat16()
    sink = torch.randn(2, generator=generator)
    swa_ids = torch.arange(12).reshape(3, 4)
    swa_ids[0, 2:] = -1
    swa_ids[2] = -1
    main_ids = torch.arange(9).reshape(3, 3)
    main_ids[0, 1] = -1
    main_ids[1, 2] = 50  # outside the pool
    main_ids[2] = -1
    args = AttentionForwardArgs(
        attention_input_type=AttentionInputType.generation_only,
        attention_sinks=sink,
        sparse_backend_args=CSA2BackendForwardArgs(
            swa_pool=pack_rows(swa.flatten(0, 1), "swa"),
            swa_indices=swa_ids,
            main_pool=pack_rows(main.flatten(0, 1), "main"),
            topk_indices=main_ids,
        ),
    )
    actual = CSA2VanillaAttention(0, 2, _HEAD_DIM).forward(
        q.flatten(1), None, None, SimpleNamespace(num_tokens=3, num_query_heads=2), args
    )
    rows = torch.cat((_decoded(swa, "swa"), _decoded(main, "main")), dim=1)
    valid = torch.cat((swa_ids >= 0, (main_ids >= 0) & (main_ids < 9)), dim=1)
    expected = _reference(q, rows, valid, sink)
    torch.testing.assert_close(actual.view_as(q).float(), expected, atol=2e-2, rtol=2e-2)
    # A query without rows reduces to the sink alone: zero output.
    assert not actual[2].any()


def _replay(metadata, states):
    metadata.csa2_replay_mode = "encoder"


def _precomputed(metadata, states):
    metadata.csa2_precomputed_kv_layers = {0}


def _source_batch(metadata, states):
    states[0].main_kv = states[0].main_kv[:3]


@pytest.mark.parametrize(
    "modify, start, end, message",
    [
        pytest.param(_replay, 0, 4, "bounded SWA replay", id="replay"),
        pytest.param(_precomputed, 0, 4, "precomputed global KV", id="ced_prepass"),
        pytest.param(_source_batch, 0, 4, "query batch", id="source_batch"),
        pytest.param(None, 0, 5, "context/generation boundary", id="mixed_tile"),
    ],
)
@torch.inference_mode()
def test_vanilla_rejects_unmodeled_geometry_before_writing(modify, start, end, message):
    layout, cache, metadata, inputs, _, _ = _ratio_two_scenario()
    states = _scenario_states(layout, metadata, inputs)
    if modify is not None:
        modify(metadata, states)
    pools = [pool.clone() for pool in (*cache.swa, cache.main, cache.index)]
    backend = CSA2VanillaAttention(0, 2, _HEAD_DIM, sparse_params=CSA2Params(layout=layout))
    with pytest.raises(ValueError, match=message):
        _call(
            backend,
            inputs["q"],
            states[0],
            start,
            end,
            inputs["sink"],
            AttentionInputType.context_only,
        )
    for before, after in zip(pools, (*cache.swa, cache.main, cache.index)):
        torch.testing.assert_close(after, before, atol=0, rtol=0)


# --------------------------------------------------------------------------
# Comparison with CSA2TrtllmAttention on real cache managers (SM100 family).
# --------------------------------------------------------------------------

# Every mode: 0 FULL ratio 2, 1 REUSE of 0, 2 FULL ratio 1 and candidate
# source, 3 REINDEX consuming its candidates, 4 REUSE of 3, 5 SWA only.
_LAYOUT = CSA2Layout(
    (2, 2, 1, 1, 1, 0),
    (0, 2),
    (0, 2, 3),
    candidate_source_layer_id=2,
    candidate_topk_blocks=8,
    candidate_block_size=8,
    index_topk=32,
    window_size=128,
)
_HEADS, _INDEX_HEADS = 8, 32
_PROMPTS = {11: 330, 12: 300}
# Forward 1 prefills both requests; forward 2 continues request 11's context
# from an odd position (a ratio-2 group straddles the forwards) and decodes one
# token of request 12.
_FORWARDS = (
    dict(num_contexts=2, starts=(0, 0), lengths=(301, 290)),
    dict(num_contexts=1, starts=(301, 290), lengths=(29, 1)),
)


def _rope_table(positions: int, dim: int = 64) -> torch.Tensor:
    """FP32 [positions, dim] interleaved RoPE table: cosines, then sines."""
    inverse = 10000.0 ** -(torch.arange(0, dim, 2, dtype=torch.float32) / dim)
    angles = torch.arange(positions, dtype=torch.float32)[:, None] * inverse[None, :]
    return torch.cat((angles.cos(), angles.sin()), dim=-1).cuda()


def _forward_inputs(forward: int, starts, lengths):
    """Seeded module outputs of one forward, shared by every implementation."""
    generator = torch.Generator(device="cuda").manual_seed(4100 + forward)
    total = sum(lengths)

    def randn(*shape, scale=1.0):
        return (torch.randn(*shape, generator=generator, device="cuda") * scale).bfloat16()

    positions = torch.cat(
        [torch.arange(s, s + n, device="cuda") for s, n in zip(starts, lengths)]
    ).int()
    table = _rope_table(512)
    layers = []
    for layer_idx in range(len(_LAYOUT.compress_ratios)):
        layer = _LAYOUT.layer(layer_idx)
        swa_norm = (randn(_HEAD_DIM, scale=0.1).float() + 1.0).bfloat16()
        state = dict(
            swa_kv=randn(total, _HEAD_DIM),
            swa_transform=RowTransform(norm=(swa_norm, 1e-6), rope=(positions, table, 64)),
        )
        if layer.mode == CSA2Mode.FULL:
            ratio = layer.compress_ratio
            groups = torch.cat(
                [
                    torch.arange(s // ratio, (s + n) // ratio, device="cuda")
                    for s, n in zip(starts, lengths)
                ]
            )
            compressed = torch.zeros(total, dtype=torch.int32, device="cuda")
            compressed[: groups.numel()] = (groups * ratio).int()
            index_norm = (randn(_INDEX_DIM, scale=0.1).float() + 1.0).bfloat16()
            state.update(
                main_kv=randn(total, _HEAD_DIM),
                index_k=randn(total, _INDEX_DIM),
                main_transform=RowTransform(rope=(compressed, table, 64)),
                index_transform=RowTransform(norm=(index_norm, 1e-6), rope=(compressed, table, 64)),
            )
        if layer.mode in (CSA2Mode.FULL, CSA2Mode.REINDEX):
            state.update(
                index_q=randn(total, _INDEX_HEADS, _INDEX_DIM),
                index_weights=torch.randn(total, _INDEX_HEADS, generator=generator, device="cuda")
                * (_INDEX_DIM * _INDEX_HEADS) ** -0.5,
            )
        layers.append(
            dict(
                q=randn(total, _HEADS, _HEAD_DIM),
                sink=torch.randn(_HEADS, generator=generator, device="cuda"),
                state=state,
            )
        )
    return layers


class _CacheRun:
    """One implementation's cache manager, requests, request metadata and layers."""

    def __init__(self, implementation: str):
        from tensorrt_llm._torch.attention.backends.sparse.csa2.cache_manager import (
            CSA2CacheManager,
        )
        from tensorrt_llm._torch.attention.backends.utils import create_attention
        from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest
        from tensorrt_llm.bindings import DataType, SamplingConfig
        from tensorrt_llm.bindings.internal.batch_manager import CacheType
        from tensorrt_llm.llmapi.llm_args import KvCacheConfig
        from tensorrt_llm.mapping import Mapping

        self.manager = CSA2CacheManager(
            KvCacheConfig(
                enable_block_reuse=False,
                max_tokens=4096,
                enable_swa_scratch_reuse=False,
                dtype="fp8",
            ),
            CacheType.SELFKONLY,
            layout=_LAYOUT,
            num_layers=len(_LAYOUT.compress_ratios),
            tokens_per_block=128,
            mapping=Mapping(),
            max_seq_len=512,
            max_batch_size=2,
            max_input_len=512,
            max_num_tokens=1024,
            dtype=DataType.BF16,
            vocab_size=1024,
        )
        self.requests = []
        try:
            for request_id, prompt in _PROMPTS.items():
                request = LlmRequest(
                    request_id=request_id,
                    max_new_tokens=8,
                    input_tokens=list(range(prompt)),
                    sampling_config=SamplingConfig(),
                    is_streaming=False,
                )
                assert self.manager.prepare_context(request)
                assert self.manager.resize_context(request, request.context_chunk_size)
                self.requests.append(request)
            self.manager._stream.synchronize()
            self.metadata = CSA2TrtllmMetadata(
                max_num_requests=2, max_num_tokens=1024, kv_cache_manager=self.manager
            )
            params = CSA2Params(layout=_LAYOUT)
            if implementation == "VANILLA":
                self.layers = [
                    CSA2VanillaAttention(i, _HEADS, _HEAD_DIM, sparse_params=params)
                    for i in range(len(_LAYOUT.compress_ratios))
                ]
            else:
                self.layers = [
                    create_attention(
                        "TRTLLM", i, _HEADS, _HEAD_DIM, num_kv_heads=1, sparse_params=params
                    )
                    for i in range(len(_LAYOUT.compress_ratios))
                ]
        except BaseException:
            self.close()
            raise

    def forward(self, num_contexts, starts, lengths, inputs):
        """Prepare one model forward and run every layer phase by phase, as the module does."""
        from tensorrt_llm._torch.metadata import KVCacheParams

        metadata = self.metadata
        metadata.request_ids = [request.py_request_id for request in self.requests]
        metadata.num_contexts = num_contexts
        metadata.prompt_lens = list(_PROMPTS.values())
        metadata.seq_lens = torch.tensor(lengths, dtype=torch.int32)
        metadata.kv_cache_params = KVCacheParams(
            use_cache=True, num_cached_tokens_per_seq=list(starts)
        )
        metadata.prepare()
        metadata.begin_model_forward()
        outputs = []
        for layer_idx, (backend, layer_inputs) in enumerate(zip(self.layers, inputs)):
            state = CSA2ForwardState(metadata=metadata, **layer_inputs["state"])
            q = layer_inputs["q"]
            output = torch.empty_like(q)
            swa_only = _LAYOUT.layer(layer_idx).mode == CSA2Mode.SWA
            for start, end in ((0, metadata.num_ctx_tokens), (metadata.num_ctx_tokens, q.shape[0])):
                if start == end:
                    continue
                tile = q[start:end]
                tile_metadata = metadata.get_query_tile_metadata(
                    tile,
                    0 if swa_only else _LAYOUT.index_topk,
                    query_start=start,
                    staging_dtype=backend.staging_dtype,
                    layer_idx=layer_idx,
                )
                args = AttentionForwardArgs(
                    attention_input_type=(
                        AttentionInputType.context_only
                        if tile_metadata.num_contexts
                        else AttentionInputType.generation_only
                    ),
                    attention_sinks=layer_inputs["sink"],
                    output=output[start:end].flatten(1),
                    sparse_backend_args=CSA2BackendForwardArgs(state=state, query_start=start),
                )
                backend.forward(tile.flatten(1), None, None, tile_metadata, forward_args=args)
            outputs.append(output)
        torch.cuda.synchronize()
        return outputs

    def cache_rows(self, starts, lengths):
        """Packed rows this forward wrote: SWA rows per layer, main and index rows per owner.

        Rows are addressed through this run's own page lists, so the two
        implementations may place them on different pages.
        """

        def slots(layer_idx, role, rows_per_page, ratio):
            result = []
            for request, start, length in zip(self.requests, starts, lengths):
                pages = self.manager.get_cache_indices(request.py_request_id, layer_idx, role)
                result += [
                    pages[i // rows_per_page] * rows_per_page + i % rows_per_page
                    for i in range(start // ratio, (start + length) // ratio)
                ]
            return torch.tensor(result, dtype=torch.int64, device="cuda")

        block = self.manager.tokens_per_block
        rows = {}
        for layer_idx, ratio in enumerate(_LAYOUT.compress_ratios):
            swa = slots(layer_idx, CSA2CacheRole.SWA, block, 1)
            rows["swa", layer_idx] = self.manager.get_swa_buffer(layer_idx)[swa]
            if layer_idx in _LAYOUT.kv_source_layer_ids:
                groups = slots(layer_idx, CSA2CacheRole.GLOBAL, block // ratio, ratio)
                rows["main", layer_idx] = self.manager.get_main_buffer(layer_idx)[groups]
                rows["index", layer_idx] = read_index_rows(
                    self.manager.get_index_pages(layer_idx), groups
                )
        return rows

    def close(self):
        for request in self.requests:
            self.manager.free_resources(request)
        self.manager.shutdown()


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
    reason="CSA2 trtllm-gen requires an SM100-family GPU",
)
@torch.inference_mode()
def test_vanilla_matches_trtllm_across_modes_and_phases(monkeypatch):
    from tensorrt_llm._torch.attention.backends.sparse.csa2.indexer import CSA2Indexer

    runs = {}
    golden_selection = {}
    current = {}

    def inject_golden_selection(indexer, state, query_start, count):
        logical = golden_selection[current["forward"], indexer.layer_idx].clone()
        state.metadata.csa2_indices[indexer.layer_idx] = logical
        return logical

    try:
        for name in ("VANILLA", "TRTLLM", "INJECTED"):
            runs[name] = _CacheRun(name)
        for forward, geometry in enumerate(_FORWARDS):
            inputs = _forward_inputs(forward, geometry["starts"], geometry["lengths"])
            golden = runs["VANILLA"].forward(inputs=inputs, **geometry)
            golden_metadata = runs["VANILLA"].metadata
            for layer_idx, published in golden_metadata.csa2_indices.items():
                golden_selection[forward, layer_idx] = published.clone()
            native = runs["TRTLLM"].forward(inputs=inputs, **geometry)
            current["forward"] = forward
            with monkeypatch.context() as patch:
                patch.setattr(CSA2Indexer, "forward", inject_golden_selection)
                injected = runs["INJECTED"].forward(inputs=inputs, **geometry)

            # Attention with identical selections isolates the kernels; BF16
            # staging leaves a relative error near 3e-3.
            for layer_idx in range(len(_LAYOUT.compress_ratios)):
                reference = golden[layer_idx].float()
                error = (injected[layer_idx].float() - reference).norm() / reference.norm()
                assert error <= 1e-2, (layer_idx, error)
            # Selections: identical domains. Production scores some layers with
            # BF16 index weights, so near-tied entries at the top-k boundary may
            # swap; bound the recall of the golden's entries per row instead.
            native_metadata = runs["TRTLLM"].metadata
            for layer_idx in (0, 2, 3):
                expected = golden_metadata.csa2_indices[layer_idx]
                actual = native_metadata.csa2_indices[layer_idx]
                counts = (expected >= 0).sum(-1)
                torch.testing.assert_close((actual >= 0).sum(-1), counts, atol=0, rtol=0)
                found = (expected[:, :, None] == actual[:, None, :]).any(-1) & (expected >= 0)
                recall = torch.where(counts > 0, found.sum(-1) / counts.clamp(min=1), 1.0)
                assert recall.mean() >= 0.98 and recall.min() >= 0.9, (layer_idx, recall)
            candidates = native_metadata.csa2_candidates[2]
            same = (candidates == golden_metadata.csa2_candidates[2]).all(-1)
            assert same.float().mean() >= 0.95
            # The composed run inherits those swaps, so bound its mean error.
            for layer_idx in range(len(_LAYOUT.compress_ratios)):
                error = (native[layer_idx].float() - golden[layer_idx].float()).abs().mean()
                assert error <= 5e-3, (layer_idx, error)
            # New cache rows: the fused norm/RoPE store may round a few bytes differently.
            golden_rows = runs["VANILLA"].cache_rows(geometry["starts"], geometry["lengths"])
            native_rows = runs["TRTLLM"].cache_rows(geometry["starts"], geometry["lengths"])
            for key, expected in golden_rows.items():
                mismatch = (native_rows[key] != expected).float().mean()
                assert mismatch <= 1e-2, (key, mismatch)
    finally:
        for run in runs.values():
            run.close()
