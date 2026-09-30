# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Attention DP with sharded Engram and bounded encoder/decoder replay.

Run directly on one node, or under ``mpirun -n 8 pytest ... -k world8`` on two
four-GPU nodes. The external-MPI path uses node-local ranks for CUDA devices.
"""

import pickle
import sys
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from types import SimpleNamespace

import cloudpickle
import pytest
import torch
from _torch.moe.test_deepseek_v41_parallel_moe import _local_mpi_topology
from _torch.moe.test_deepseek_v41_parallel_moe import (
    parallel_moe_executor as _parallel_moe_executor,
)
from mpi4py import MPI
from mpi4py.futures import MPIPoolExecutor

import tensorrt_llm
from tensorrt_llm._torch.attention.backends.sparse.csa2.metadata import CSA2TrtllmMetadata
from tensorrt_llm._torch.attention.backends.sparse.csa2.params import CSA2Layout
from tensorrt_llm._torch.distributed import AllReduceStrategy
from tensorrt_llm._torch.models.modeling_deepseekv41 import DeepseekV41Engram
from tensorrt_llm._torch.modules.engram import EngramConfig, EngramHashProvider
from tensorrt_llm._torch.modules.engram.functional import engram_gate
from tensorrt_llm.mapping import Mapping

cloudpickle.register_pickle_by_value(sys.modules[__name__])
MPI.pickle.__init__(cloudpickle.dumps, cloudpickle.loads, pickle.HIGHEST_PROTOCOL)
pytestmark = pytest.mark.threadleak(enabled=False)
parallel_moe_executor = _parallel_moe_executor

_HEAD_DIM = 128
_HIDDEN_DIM = 64
_HC_MULT = 4


def _buckets(num_heads: int) -> list[int]:
    return [32 + (head * 19) % 97 for head in range(num_heads)]


def _checkpoint(num_heads: int) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(31)
    rows = sum(_buckets(num_heads))
    return {
        "weight": (torch.randn(rows, _HEAD_DIM, generator=generator) * 0.1).to(torch.float8_e4m3fn),
        "scale": torch.randint(
            124, 129, (rows, _HEAD_DIM // 32), dtype=torch.uint8, generator=generator
        ).view(torch.float8_e8m0fnu),
    }


def _config(num_heads: int) -> EngramConfig:
    return EngramConfig(
        max_ngram_size=4,
        n_head_per_ngram=num_heads // 3,
        n_embed_per_ngram=num_heads // 3 * _HEAD_DIM,
        layer_ids=[0],
        hidden_size=_HIDDEN_DIM,
        hc_mult=_HC_MULT,
        dtype=torch.bfloat16,
    )


def _hash_provider(num_heads: int) -> EngramHashProvider:
    provider = EngramHashProvider.__new__(EngramHashProvider)
    provider._pending_history_seeds = {}
    provider.config = _config(num_heads)
    provider._lookup_table = torch.arange(1024)
    provider._pad_id = 0
    provider._multipliers = {0: torch.tensor([17, 31, 43, 59])}
    width = num_heads // 3
    buckets = _buckets(num_heads)
    provider._moduli_ints = {
        0: [buckets[start : start + width] for start in range(0, num_heads, width)]
    }
    provider._modules = {0: [torch.tensor(sizes) for sizes in provider._moduli_ints[0]]}
    provider._device = None
    provider._cached_hashes = None
    provider._cached_hashes_store = {}
    provider._history = None
    provider._history_row_of = {}
    provider._history_free_rows = []
    provider._history_host_staging = {}
    provider._decode_row_map = None
    return provider


def _module(world_size: int, rank: int, num_heads: int) -> DeepseekV41Engram:
    mapping = Mapping(
        world_size=world_size,
        rank=rank,
        tp_size=world_size,
        enable_attention_dp=True,
        gpus_per_node=torch.cuda.device_count(),
    )
    with pytest.MonkeyPatch.context() as settings:
        settings.setenv("TRTLLM_MXFP8_GEMM_BACKEND", "trtllm")
        with torch.device("cuda"):
            module = DeepseekV41Engram(
                layer_id=0,
                config=_config(num_heads),
                vocab_sizes_flat=_buckets(num_heads),
                stream=torch.cuda.Stream(),
                mapping=mapping,
                allreduce_strategy=AllReduceStrategy.NCCL,
            )
        module.multi_head_embedding.load_weights(_checkpoint(num_heads))
    generator = torch.Generator().manual_seed(47)
    out_features, in_features = module.kv_proj.weight.shape
    weight = (torch.randn(out_features, in_features, generator=generator) * 0.01).to(
        torch.float8_e4m3fn
    )
    scale = torch.randint(
        124, 129, (out_features // 32, in_features // 32), dtype=torch.uint8, generator=generator
    ).view(torch.float8_e8m0fnu)
    module.kv_proj.load_weights([{"weight": weight, "scale": scale}])
    module.post_load_weights()
    assert module.kv_proj.quant_method.use_cutlass, "dequantized WKV is not native FP8 execution"
    assert module.kv_proj.weight.dtype == torch.float8_e4m3fn
    assert module.kv_proj.weight_scale.dtype == torch.uint8
    assert module.multi_head_embedding.weight.is_pinned()
    assert module.multi_head_embedding.scale.is_pinned()
    assert module.multi_head_embedding.shard_heads == (num_heads % world_size == 0)
    assert module.multi_head_embedding.weight.shape[0] < sum(_buckets(num_heads))
    return module


def _dense_lookup(indices: torch.Tensor, num_heads: int) -> torch.Tensor:
    checkpoint = _checkpoint(num_heads)
    scales = torch.exp2(checkpoint["scale"].view(torch.uint8).float() - 127)
    table = (checkpoint["weight"].float() * scales.repeat_interleave(32, dim=-1)).to(
        device="cuda", dtype=torch.bfloat16
    )
    offsets = torch.tensor([0, *_buckets(num_heads)[:-1]], device="cuda").cumsum(0)
    return table[indices + offsets].flatten(1)


def _indices(rank: int, num_tokens: int, num_heads: int) -> torch.Tensor:
    tokens = torch.arange(num_tokens, device="cuda")[:, None]
    heads = torch.arange(num_heads, device="cuda")[None, :]
    buckets = torch.tensor(_buckets(num_heads), device="cuda")
    return (tokens * 7 + heads * 13 + rank * 17 + 3) % buckets


def _hidden(rank: int, num_tokens: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(53 + rank)
    return torch.randn(num_tokens, _HC_MULT, _HIDDEN_DIM, generator=generator).to(
        device="cuda", dtype=torch.bfloat16
    )


def _expected(
    module: DeepseekV41Engram, hidden: torch.Tensor, embeddings: torch.Tensor
) -> torch.Tensor:
    if hidden.shape[0] == 0:
        return hidden
    # Compare collectives against an independent lookup with the same FP8 arithmetic.
    return engram_gate(
        hidden,
        module.kv_proj(embeddings),
        module.query_norm_weight,
        module.key_norm_weight,
        module.norm_eps,
        add_residual=True,
    )


def _check_forward(module: DeepseekV41Engram, rank: int, counts: list[int], num_heads: int) -> None:
    indices = _indices(rank, counts[rank], num_heads)
    hidden = _hidden(rank, counts[rank])
    prefetched = module.precompute(indices, all_rank_num_tokens=counts)
    torch.cuda.current_stream().wait_event(module.sync_event)
    assert prefetched.embeddings.shape[0] == sum(counts)
    projection_inputs = []
    hook = module.kv_proj.register_forward_pre_hook(
        lambda _, args: projection_inputs.append(args[0])
    )
    try:
        actual = module(hidden, prefetched, add_residual=True)
    finally:
        hook.remove()
    reference = _dense_lookup(indices, num_heads)
    if counts[rank]:
        assert len(projection_inputs) == 1
        torch.testing.assert_close(projection_inputs[0], reference, rtol=0, atol=0)
        assert reference.count_nonzero() > 0
    else:
        assert not projection_inputs
    torch.testing.assert_close(actual, _expected(module, hidden, reference), rtol=0, atol=0)


def _check_graph(module: DeepseekV41Engram, rank: int, world_size: int, num_heads: int) -> None:
    provider = _hash_provider(num_heads)
    capacity = 4
    counts = [capacity] * world_size
    tokens = torch.tensor([11 + rank, 23 + rank], device="cuda")
    full_tokens = tokens.clone()
    hashes = provider.compute_hashes(
        tokens,
        position_ids=torch.arange(2, device="cuda"),
        request_ids=[rank + 1],
        seq_lens_host=torch.tensor([2]),
        max_seq_len=32,
        padded_num_tokens=capacity,
    )[0]
    pointer = hashes.data_ptr()
    hidden = _hidden(rank, capacity)

    def forward() -> torch.Tensor:
        cached = provider.compute_hashes(tokens, padded_num_tokens=capacity)[0]
        prefetched = module.precompute(cached, all_rank_num_tokens=counts)
        torch.cuda.current_stream().wait_event(module.sync_event)
        return module(hidden, prefetched, add_residual=True)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            forward()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    MPI.COMM_WORLD.Barrier()
    graph = torch.cuda.CUDAGraph()
    projection_inputs = []
    hook = module.kv_proj.register_forward_pre_hook(
        lambda _, args: projection_inputs.append(args[0])
    )
    try:
        with torch.cuda.graph(graph):
            captured = forward()
    finally:
        hook.remove()
    assert len(projection_inputs) == 1

    for step in range(3):
        real_count = 1 + (rank + step) % 3
        next_tokens = torch.arange(real_count, device="cuda") + 37 + rank * 7 + step * 13
        positions = torch.arange(real_count, device="cuda") + full_tokens.numel()
        full_tokens = torch.cat((full_tokens, next_tokens))
        refreshed = provider.refresh_captured_hashes(
            next_tokens,
            position_ids=positions,
            request_ids=[rank + 1],
            seq_lens_host=torch.tensor([real_count]),
            max_seq_len=32,
            padded_num_tokens=capacity,
        )[0]
        assert refreshed.data_ptr() == pointer
        whole_prompt = _hash_provider(num_heads).compute_hashes(full_tokens)[0]
        torch.testing.assert_close(
            refreshed[:real_count], whole_prompt[-real_count:], rtol=0, atol=0
        )
        assert not refreshed[real_count:].count_nonzero()
        hidden.add_(0.0625)
        graph.replay()
        torch.cuda.synchronize()
        reference = _dense_lookup(refreshed, num_heads)
        torch.testing.assert_close(projection_inputs[0], reference, rtol=0, atol=0)
        torch.testing.assert_close(captured, _expected(module, hidden, reference), rtol=0, atol=0)


@torch.inference_mode()
def _run_rank(world_size: int, num_heads: int) -> bool:
    rank = tensorrt_llm.mpi_rank()
    local_comm = MPI.COMM_WORLD.Split_type(MPI.COMM_TYPE_SHARED)
    local_rank = local_comm.Get_rank()
    local_comm.Free()
    torch.cuda.set_device(local_rank)
    module = _module(world_size, rank, num_heads)
    for counts in (
        [3] * world_size,
        [2, 5] + [1] * (world_size - 2),
        [0, 5] + [0] * (world_size - 2),
        [0] * world_size,
    ):
        _check_forward(module, rank, counts, num_heads)
    _check_graph(module, rank, world_size, num_heads)
    torch.cuda.synchronize()
    return True


@pytest.mark.parametrize(
    "world_size,num_heads",
    [
        pytest.param(8, 24, id="world8-head-sharded"),
        pytest.param(2, 3, id="world2-row-sharded"),
    ],
)
def test_attention_dp_engram(world_size: int, num_heads: int) -> None:
    external_world = tensorrt_llm.mpi_world_size()
    if external_world > 1:
        if external_world != world_size:
            pytest.skip(f"MPI launch has {external_world} ranks, test requires {world_size}")
        assert _run_rank(world_size, num_heads)
        return
    if torch.cuda.device_count() < world_size:
        pytest.skip(f"need {world_size} local GPUs or an external MPI launch")
    with MPIPoolExecutor(max_workers=world_size) as executor:
        assert all(executor.map(_run_rank, [world_size] * world_size, [num_heads] * world_size))


_REPLAY_HIDDEN_SIZE = 128
_REPLAY_WINDOW = 4
_REPLAY_MAX_TOKENS = 256


def _replay_attention(layout: CSA2Layout, mapping: Mapping, layer_idx: int = 0) -> torch.nn.Module:
    from tensorrt_llm._torch.attention.backends.interface import (
        PositionalEmbeddingParams,
        RopeParams,
    )
    from tensorrt_llm._torch.attention.backends.sparse.csa2.module import DeepseekV41Attention
    from tensorrt_llm.functional import PositionEmbeddingType

    module = DeepseekV41Attention(
        layout,
        layer_idx,
        PositionalEmbeddingParams(
            type=PositionEmbeddingType.rope_gptj,
            rope=RopeParams(dim=64, theta=160000, max_positions=512),
            is_neox=False,
        ),
        hidden_size=_REPLAY_HIDDEN_SIZE,
        num_heads=8,
        q_lora_rank=32,
        o_lora_rank=16,
        num_groups=2,
        index_heads=8,
        mapping=mapping,
        compute_backend="trtllm",
    )
    for name, parameter in module.named_parameters():
        if name.endswith("norm.weight"):
            parameter.fill_(1)
        else:
            parameter.normal_(std=0.03)
    return module


@contextmanager
def _replay_metadata(
    mapping: Mapping, layout: CSA2Layout, length: int, cached: int
) -> Iterator[CSA2TrtllmMetadata]:
    from tensorrt_llm._torch.attention.backends.sparse.csa2.cache_manager import CSA2CacheManager
    from tensorrt_llm._torch.metadata import KVCacheParams
    from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest
    from tensorrt_llm.bindings import DataType, SamplingConfig
    from tensorrt_llm.bindings.internal.batch_manager import CacheType
    from tensorrt_llm.llmapi.llm_args import KvCacheConfig

    manager = CSA2CacheManager(
        KvCacheConfig(
            enable_block_reuse=False,
            max_gpu_total_bytes=128 << 20,
            enable_swa_scratch_reuse=False,
            dtype="fp8",
        ),
        CacheType.SELFKONLY,
        num_layers=1,
        tokens_per_block=128,
        max_seq_len=512,
        max_batch_size=1,
        max_input_len=_REPLAY_MAX_TOKENS,
        max_num_tokens=_REPLAY_MAX_TOKENS,
        mapping=mapping,
        dtype=DataType.BF16,
        vocab_size=128,
        layout=layout,
    )
    request = None
    try:
        if length:
            request = LlmRequest(
                request_id=1703,
                max_new_tokens=4,
                input_tokens=list(range(cached or length)),
                sampling_config=SamplingConfig(),
                is_streaming=False,
            )
            assert manager.prepare_context(request)
            assert manager.resize_context(request, request.context_chunk_size)
            manager._stream.synchronize()
        metadata = CSA2TrtllmMetadata(
            max_num_requests=1,
            max_num_tokens=_REPLAY_MAX_TOKENS,
            kv_cache_manager=manager,
            mapping=mapping,
        )
        metadata.request_ids = [request.py_request_id] if request is not None else []
        metadata.num_contexts = int(bool(length))
        metadata.prompt_lens = [cached or length] if length else []
        metadata.seq_lens = torch.tensor(
            [cached or length] if length else [], dtype=torch.int32, device="cpu"
        )
        metadata.kv_cache_params = KVCacheParams(
            use_cache=True, num_cached_tokens_per_seq=[0] if length else []
        )
        if length:
            metadata.prepare()
        else:
            # A zero-row MoE peer has no attention request to prepare.
            metadata.csa2_positions = torch.empty(0, dtype=torch.int32, device="cuda")
            metadata.reset_routing()
        yield metadata
    finally:
        if request is not None:
            manager.free_resources(request)
        manager.shutdown()


def _replay_boundary(attention: torch.nn.Module) -> torch.nn.Module:
    from tensorrt_llm._torch.models.modeling_deepseekv41 import DeepseekV41Model
    from tensorrt_llm._torch.modules.rms_norm import RMSNorm

    model = DeepseekV41Model.__new__(DeepseekV41Model)
    torch.nn.Module.__init__(model)
    norm = RMSNorm(hidden_size=_REPLAY_HIDDEN_SIZE, eps=1e-6, dtype=torch.bfloat16)
    norm.weight.fill_(1)
    model.layers = [
        SimpleNamespace(
            layer_idx=0,
            engram=None,
            input_layernorm=norm,
            self_attn=attention,
            _decoder_global_input=lambda state: norm(state.residual[:, 0]),
        )
    ]
    model.disagg_remote_tail_replay = False
    model.decoder_replay_split = 0
    model.decoder_replay_window = _REPLAY_WINDOW
    model.ced_kv_precompute = True
    model._decoder_replay_observed = False
    model._decoder_replay_reuse_warning_emitted = False
    return model


def _prepare_replay(model, metadata, all_token_states_required, requests=None):
    from tensorrt_llm._torch.distributed import Distributed
    from tensorrt_llm._torch.pyexecutor.engine.runners.common import get_all_rank_num_tokens

    model.prepare_adp_inputs(
        metadata, all_token_states_required=all_token_states_required, requests=requests
    )
    metadata.all_rank_num_tokens = get_all_rank_num_tokens(
        metadata,
        enable_attention_dp=True,
        mapping=metadata.mapping,
        dist=Distributed.get(metadata.mapping),
    )


def _check_dp_replay_case(
    mapping: Mapping,
    layout: CSA2Layout,
    model: torch.nn.Module,
    reference_attention: torch.nn.Module,
    moe: torch.nn.Module,
    reference_moe: torch.nn.Module,
    counts: list[int],
    cached_counts: list[int],
    required_states: list[bool],
    prompt_ends: list[int] | None = None,
) -> None:
    from tensorrt_llm._torch.metadata import KVCacheParams
    from tensorrt_llm._torch.modules.mhc.hyper_connection import HCState

    rank = mapping.tp_rank
    count, cached = counts[rank], cached_counts[rank]
    generator = torch.Generator(device="cuda").manual_seed(1711 + rank)
    full_hidden = torch.randn(
        cached + count,
        _REPLAY_HIDDEN_SIZE,
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )
    norm = model.layers[0].input_layernorm
    with ExitStack() as stack:
        metadata = stack.enter_context(_replay_metadata(mapping, layout, count, cached))
        reference_metadata = stack.enter_context(_replay_metadata(Mapping(), layout, count, cached))
        metadata.all_rank_num_tokens = counts
        if prompt_ends is not None and metadata.num_contexts:
            metadata.decoder_context_ends = (prompt_ends[rank],)
        if cached:
            for attention, step_metadata in (
                (model.layers[0].self_attn, metadata),
                (reference_attention, reference_metadata),
            ):
                attention(norm(full_hidden[:cached]), step_metadata.csa2_positions, step_metadata)
                step_metadata.num_contexts = 0
                step_metadata.seq_lens = torch.tensor([count], dtype=torch.int32, device="cpu")
                step_metadata.kv_cache_params = KVCacheParams(
                    use_cache=True, num_cached_tokens_per_seq=[cached]
                )
                step_metadata.prepare()
        hidden = full_hidden[cached:]
        positions = metadata.csa2_positions
        input_ids = torch.arange(count, device="cuda") + rank * 17
        state = HCState.resolved(hidden[:, None, :], pre_mix=torch.ones(count, 1, 1, device="cuda"))
        original_seq_lens = metadata.seq_lens.clone()
        original_prompt_lens = list(metadata.prompt_lens)
        _prepare_replay(model, metadata, required_states[rank])
        _, plan = model._plan_bounded_replay(metadata, required_states[rank])
        expected_counts = [
            min(length, _REPLAY_WINDOW) if not cache and not required else length
            for length, cache, required in zip(counts, cached_counts, required_states)
        ]
        if prompt_ends is not None:
            expected_counts = [
                len(set(range(length)) & set(range(max(0, end - _REPLAY_WINDOW), end)))
                if not cache and not required
                else length
                for length, end, cache, required in zip(
                    counts, prompt_ends, cached_counts, required_states
                )
            ]
        assert plan is not None
        kept = expected_counts[rank]
        start = count - kept
        new_positions, new_ids, replay_state = model._enter_bounded_replay(
            plan, metadata, positions, input_ids, state
        )
        assert metadata.all_rank_num_tokens == expected_counts
        torch.testing.assert_close(new_positions, positions[start:], rtol=0, atol=0)
        torch.testing.assert_close(new_ids, input_ids[start:], rtol=0, atol=0)
        if start == 0:
            assert new_positions is positions
            assert replay_state.residual is state.residual
        if not plan.updates_local_metadata:
            assert replay_state is state
            assert metadata.csa2_replay_query_rows is None
        else:
            assert metadata.csa2_replay_query_rows is plan.rows

        # Full-prefill parity is invalid for this approximation. Build the same
        # bounded replay independently of the DP planner, retaining all GLOBAL KV.
        if start and kept:
            reference_attention.prepare_global_cache(norm(hidden), reference_metadata)
            reference_metadata.seq_lens = torch.tensor([kept], dtype=torch.int32, device="cpu")
            reference_metadata.prompt_lens = [kept]
            reference_metadata.kv_cache_params.num_cached_tokens_per_seq = [start]
            reference_metadata.set_swa_bounded_replay([count], decoder=True)
            reference_metadata.prepare()
            reference_metadata.csa2_precomputed_kv_layers = {0}
        if kept:
            actual_attention = model.layers[0].self_attn(
                norm(replay_state.residual[:, 0]), new_positions, metadata
            )
            expected_attention = reference_attention(
                norm(hidden[start:]), reference_metadata.csa2_positions, reference_metadata
            )
            torch.testing.assert_close(actual_attention, expected_attention, rtol=0, atol=0)
        else:
            actual_attention = expected_attention = hidden[:0]
        logits = torch.arange(8, device="cuda", dtype=torch.float32)[None, :].expand(kept, -1)
        logits = torch.sin(logits + new_positions[:, None].float() + rank)
        if any(expected_counts):
            actual = moe(actual_attention, logits, all_rank_num_tokens=metadata.all_rank_num_tokens)
            expected = reference_moe(expected_attention, logits)
        else:
            actual = expected = hidden[:0]
        torch.cuda.synchronize()
        assert MPI.COMM_WORLD.allgather(actual.shape[0]) == expected_counts
        if kept:
            assert torch.isfinite(actual).all()
            reference_moe.check_accuracy(actual, expected)
        restored = model._exit_bounded_replay(plan, metadata, actual)
        assert restored.shape == hidden.shape
        assert not restored[:start].count_nonzero()
        torch.testing.assert_close(restored[start:], actual, rtol=0, atol=0)
        assert metadata.all_rank_num_tokens == counts
        torch.testing.assert_close(metadata.seq_lens, original_seq_lens, rtol=0, atol=0)
        assert metadata.prompt_lens == original_prompt_lens
        assert metadata.kv_cache_params.num_cached_tokens_per_seq == ([cached] if count else [])
        assert metadata.csa2_replay_query_rows is None


def _check_dp_encoder_replay_case(
    mapping: Mapping,
    moe: torch.nn.Module,
    reference_moe: torch.nn.Module,
    reused_peer: bool,
) -> None:
    from tensorrt_llm._torch.attention.backends.sparse.csa2.cache_manager import (
        CSA2CacheManager,
        CSA2CacheRole,
    )
    from tensorrt_llm._torch.attention.backends.sparse.csa2.ced import complete_encoder_replay
    from tensorrt_llm._torch.attention.backends.sparse.csa2.quantization import read_index_rows
    from tensorrt_llm._torch.metadata import KVCacheParams
    from tensorrt_llm._torch.modules.mhc.hyper_connection import HCState
    from tensorrt_llm._torch.pyexecutor.ced_replay import EncoderReplay, encoder_replay_tokens
    from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest
    from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests
    from tensorrt_llm.bindings import DataType, SamplingConfig
    from tensorrt_llm.bindings.internal.batch_manager import CacheType
    from tensorrt_llm.llmapi.llm_args import KvCacheConfig
    from tensorrt_llm.runtime.kv_cache_manager_v2 import _introspection

    layout = CSA2Layout((1, 1), (0, 1), (0, 1), index_topk=256, window_size=_REPLAY_WINDOW)
    rank = mapping.tp_rank
    prefix = 128 if rank == 0 or reused_peer else 0
    suffix = 5 if rank == 0 else 1 if reused_peer else 3
    generator = torch.Generator(device="cuda").manual_seed(1733 + rank)
    inputs = torch.randn(
        133, _REPLAY_HIDDEN_SIZE, device="cuda", dtype=torch.bfloat16, generator=generator
    )

    @contextmanager
    def cache(local_mapping: Mapping) -> Iterator[tuple]:
        manager = CSA2CacheManager(
            KvCacheConfig(
                enable_block_reuse=True,
                enable_swa_scratch_reuse=False,
                max_gpu_total_bytes=128 << 20,
                dtype="fp8",
            ),
            CacheType.SELFKONLY,
            num_layers=2,
            tokens_per_block=128,
            max_seq_len=512,
            max_batch_size=1,
            max_input_len=_REPLAY_MAX_TOKENS,
            max_num_tokens=_REPLAY_MAX_TOKENS,
            mapping=local_mapping,
            dtype=DataType.BF16,
            vocab_size=4096,
            layout=layout,
        )
        requests = []
        try:
            assert manager._decoder_replay_window == _REPLAY_WINDOW
            assert manager._encoder_optional_groups
            metadata = CSA2TrtllmMetadata(
                max_num_requests=1,
                max_num_tokens=_REPLAY_MAX_TOKENS,
                kv_cache_manager=manager,
                mapping=local_mapping,
            )
            yield manager, metadata, requests
        finally:
            for request in requests:
                manager.free_resources(request)
            manager.shutdown()

    def admit(case: tuple, request_id: int, tokens: list[int]) -> LlmRequest:
        manager, _, requests = case
        request = LlmRequest(
            request_id=request_id,
            max_new_tokens=4,
            input_tokens=tokens,
            sampling_config=SamplingConfig(),
            is_streaming=False,
        )
        assert manager.prepare_context(request)
        requests.append(request)
        request.py_seq_slot = 0
        request.context_chunk_size = len(tokens) - request.context_current_position
        assert manager.resize_context(request, request.context_chunk_size)
        scheduled = ScheduledRequests()
        scheduled.context_requests_last_chunk = [request]
        manager.prepare_resources(scheduled)
        return request

    def global_bytes(metadata: CSA2TrtllmMetadata, end: int) -> dict:
        manager = metadata.kv_cache_manager
        result = {}
        logical = torch.arange(end, device="cuda")
        for layer in layout.kv_source_layer_ids:
            for role in (CSA2CacheRole.GLOBAL, CSA2CacheRole.INDEX):
                pages = torch.tensor(
                    manager.get_cache_indices(metadata.request_ids[0], layer, role),
                    device="cuda",
                )
                slots = pages[logical // 128] * 128 + logical % 128
                values = (
                    manager.get_main_buffer(layer)[slots]
                    if role == CSA2CacheRole.GLOBAL
                    else read_index_rows(manager.get_index_pages(layer), slots)
                )
                result[layer, role] = values.view(torch.uint8).clone()
        return result

    def check_floor(metadata: CSA2TrtllmMetadata, layer: int, floor: int) -> None:
        logical = metadata.csa2_positions[:, None] - _REPLAY_WINDOW + 1
        logical = logical + torch.arange(_REPLAY_WINDOW, device="cuda")
        assert (metadata.csa2_swa_indices[layer][logical < floor] == -1).all()

    with pytest.MonkeyPatch.context() as settings, ExitStack() as stack:
        settings.setenv("TRTLLM_V41_DECODER_BOUNDED_REPLAY", "1")
        settings.setenv("TRTLLM_V41_ENCODER_REPLAY", "1")
        actual_case = stack.enter_context(cache(mapping))
        reference_case = stack.enter_context(cache(Mapping()))
        with torch.device("cuda"):
            attention = [_replay_attention(layout, mapping, i) for i in range(2)]
            reference_attention = [_replay_attention(layout, Mapping(), i) for i in range(2)]
            for actual, expected in zip(attention, reference_attention):
                expected.load_state_dict(actual.state_dict())
            model = _replay_boundary(attention[1])
            model.layers[0].layer_idx = 1
            model.layers = [None, model.layers[0]]
            model.decoder_replay_split = 1
        norm = model.layers[1].input_layernorm

        def run(requests: list[LlmRequest], frozen: list[dict] | None = None) -> None:
            request = requests[0]
            start = request.context_current_position - encoder_replay_tokens(request)
            length = request.context_chunk_size + encoder_replay_tokens(request)
            counts = MPI.COMM_WORLD.allgather(length)
            metadata, reference_metadata = actual_case[1], reference_case[1]
            for step_metadata, step_request in zip((metadata, reference_metadata), requests):
                step_metadata.request_ids = [step_request.py_request_id]
                step_metadata.num_contexts = 1
                step_metadata.seq_lens = torch.tensor([length], dtype=torch.int32, device="cpu")
                step_metadata.prompt_lens = [length]
                step_metadata.kv_cache_params = KVCacheParams(
                    use_cache=True, num_cached_tokens_per_seq=[start]
                )
                if step_metadata is metadata:
                    step_metadata.prepare_context_replay([step_request])
                elif encoder_replay_tokens(step_request):
                    step_metadata.set_swa_bounded_replay([step_request.context_current_position])
                step_metadata.prepare()
                step_metadata.begin_model_forward()
                check_floor(step_metadata, 0, start)
            metadata.all_rank_num_tokens = counts
            assert MPI.COMM_WORLD.allgather(metadata.num_tokens) == counts
            hidden = inputs[start : start + length]
            positions = metadata.csa2_positions
            ids = torch.arange(start, start + length, device="cuda")
            _prepare_replay(model, metadata, False, [request])
            _, plan = model._plan_bounded_replay(metadata, False, [request])
            assert plan is not None

            def layer_output(
                layer: int, actual_hidden: torch.Tensor, expected_hidden: torch.Tensor
            ) -> torch.Tensor:
                actual = attention[layer](norm(actual_hidden), metadata.csa2_positions, metadata)
                expected = reference_attention[layer](
                    norm(expected_hidden), reference_metadata.csa2_positions, reference_metadata
                )
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                logits = torch.arange(8, device="cuda", dtype=torch.float32)[None, :]
                logits = torch.sin(logits + metadata.csa2_positions[:, None].float() + rank)
                output = moe(actual, logits, all_rank_num_tokens=metadata.all_rank_num_tokens)
                reference = reference_moe(expected, logits)
                assert torch.isfinite(output).all()
                reference_moe.check_accuracy(output, reference)
                return output

            encoder = layer_output(0, hidden, hidden)
            state = HCState.resolved(
                encoder[:, None, :], pre_mix=torch.ones(length, 1, 1, device="cuda")
            )
            kept = min(length, _REPLAY_WINDOW)
            decoder_start = start + length - kept
            reference_attention[1].prepare_global_cache(norm(encoder), reference_metadata)
            reference_metadata.seq_lens = torch.tensor([kept], dtype=torch.int32, device="cpu")
            reference_metadata.prompt_lens = [kept]
            reference_metadata.kv_cache_params.num_cached_tokens_per_seq = [decoder_start]
            reference_metadata.set_decoder_query_boundary(1)
            reference_metadata.prepare()
            reference_metadata.csa2_precomputed_kv_layers = {1}
            logical = reference_metadata.csa2_positions[:, None] - _REPLAY_WINDOW + 1
            logical = logical + torch.arange(_REPLAY_WINDOW, device="cuda")
            reference_metadata.csa2_swa_indices[1].masked_fill_(logical < decoder_start, -1)
            new_positions, new_ids, replay_state = model._enter_bounded_replay(
                plan, metadata, positions, ids, state
            )
            expected_counts = [min(count, _REPLAY_WINDOW) for count in counts]
            assert metadata.all_rank_num_tokens == expected_counts
            assert MPI.COMM_WORLD.allgather(replay_state.residual.shape[0]) == expected_counts
            torch.testing.assert_close(new_positions, positions[-kept:], rtol=0, atol=0)
            torch.testing.assert_close(new_ids, ids[-kept:], rtol=0, atol=0)
            check_floor(metadata, 1, decoder_start)
            # Match encoder quantization so this comparison isolates the decoder cache and rows.
            output = layer_output(1, replay_state.residual[:, 0], encoder[-kept:])
            restored = model._exit_bounded_replay(plan, metadata, output)
            assert restored.shape == hidden.shape
            assert not restored[:-kept].count_nonzero()
            torch.testing.assert_close(restored[-kept:], output, rtol=0, atol=0)
            assert metadata.all_rank_num_tokens == counts
            assert metadata.seq_lens.tolist() == [length]
            assert metadata.prompt_lens == [length]
            assert metadata.kv_cache_params.num_cached_tokens_per_seq == [start]
            assert metadata.csa2_replay_query_rows is None
            if frozen:
                for step_metadata, snapshot in zip((metadata, reference_metadata), frozen):
                    for key, value in global_bytes(step_metadata, prefix).items():
                        torch.testing.assert_close(value, snapshot[key], rtol=0, atol=0)
                if prefix:
                    assert isinstance(request.py_ced_replay, EncoderReplay)
                    assert not request.py_ced_replay.consumed
                else:
                    assert request.py_ced_replay is None
                complete_encoder_replay([request])
                if prefix:
                    assert request.py_ced_replay.consumed
                    assert encoder_replay_tokens(request) == 0

        warm = [admit(case, 1741, list(range(129))) for case in (actual_case, reference_case)]
        run(warm)
        frozen = []
        for case, request in zip((actual_case, reference_case), warm):
            manager, metadata, active_requests = case
            kv = manager.kv_cache_map[request.py_request_id]
            kv.commit(list(range(128)))
            kv.stop_committing()
            _introspection.drop_optional_pages(kv, 128)
            assert kv.num_committed_tokens == 128
            frozen.append(global_bytes(metadata, prefix))
            manager.free_resources(request)
            active_requests.remove(request)
        tokens = list(range(prefix + suffix)) if prefix else list(range(2000, 2000 + suffix))
        requests = [admit(case, 1742, tokens) for case in (actual_case, reference_case)]
        for request in requests:
            assert request.context_current_position == prefix
            assert encoder_replay_tokens(request) == (_REPLAY_WINDOW if prefix else 0)
        expected_counts = [9, 5 if reused_peer else 3]
        assert (
            MPI.COMM_WORLD.allgather(
                requests[0].context_chunk_size + encoder_replay_tokens(requests[0])
            )
            == expected_counts
        )
        run(requests, frozen)


@torch.inference_mode()
def _run_dp_decoder_replay() -> None:
    from _torch.moe.quantize_utils import MXFP4MXFP8QuantizeUtil
    from transformers import PretrainedConfig

    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm._torch.moe.fused_moe import DefaultMoeRoutingMethod, create_moe
    from tensorrt_llm._torch.moe.fused_moe.communication.allgather_reducescatter import (
        AllGatherReduceScatter,
    )
    from tensorrt_llm._torch.utils import (
        get_per_request_prefill_cuda_graph_flag,
        set_per_request_prefill_cuda_graph_flag,
    )
    from tensorrt_llm.models.modeling_utils import QuantAlgo, QuantConfig

    local_rank, local_size = _local_mpi_topology()
    torch.cuda.set_device(local_rank)
    world_size = tensorrt_llm.mpi_world_size()
    mapping = Mapping(
        world_size=world_size,
        rank=tensorrt_llm.mpi_rank(),
        gpus_per_node=local_size,
        tp_size=world_size,
        moe_ep_size=world_size,
        moe_tp_size=1,
        enable_attention_dp=True,
    )
    quant_config = QuantConfig(quant_algo=QuantAlgo.W4A8_MXFP4_MXFP8)
    config = ModelConfig(
        pretrained_config=PretrainedConfig(
            hidden_size=_REPLAY_HIDDEN_SIZE,
            intermediate_size=256,
            num_experts=8,
            torch_dtype=torch.bfloat16,
        ),
        mapping=mapping,
        quant_config=quant_config,
        moe_backend="CUTLASS",
        max_num_tokens=_REPLAY_MAX_TOKENS,
    )
    previous_graph_flag = get_per_request_prefill_cuda_graph_flag()
    set_per_request_prefill_cuda_graph_flag(False)
    try:
        with torch.device("cuda"):
            torch.manual_seed(1707)
            layout = CSA2Layout((1,), (0,), (0,), index_topk=16, window_size=_REPLAY_WINDOW)
            attention = _replay_attention(layout, mapping)
            reference_attention = _replay_attention(layout, Mapping())
            reference_attention.load_state_dict(attention.state_dict())
            model = _replay_boundary(attention)
            routing = DefaultMoeRoutingMethod(top_k=2)
            quantizer = MXFP4MXFP8QuantizeUtil(
                num_experts=8,
                dtype=torch.bfloat16,
                intermediate_size=256,
                hidden_size=_REPLAY_HIDDEN_SIZE,
                quant_config=quant_config,
            )
            weights = quantizer.create_weights(input_hidden_alignment=128)
            reference = quantizer.create_ref_module(routing)
            reference.load_weights([weights])
            moe = create_moe(routing_method=routing, model_config=config, reduce_results=True)
        with moe:
            moe.load_weights([weights])
            moe.post_load_weights()
            assert isinstance(moe.comm, AllGatherReduceScatter)
            for counts, cached, required in (
                ([9, 7], [0, 0], [False, False]),
                ([9, 3], [0, 0], [False, False]),
                ([9, 1], [0, 3], [False, False]),
                ([9, 0], [0, 0], [False, False]),
                ([9, 7], [0, 0], [False, True]),
            ):
                _check_dp_replay_case(
                    mapping,
                    layout,
                    model,
                    reference_attention,
                    moe,
                    reference,
                    counts,
                    cached,
                    required,
                )
            for reused_peer in (False, True):
                _check_dp_encoder_replay_case(mapping, moe, reference, reused_peer)
            for ends in ([50, 7], [50, 50]):
                _check_dp_replay_case(
                    mapping,
                    layout,
                    model,
                    reference_attention,
                    moe,
                    reference,
                    [9, 7],
                    [0, 0],
                    [False, False],
                    prompt_ends=ends,
                )
    finally:
        set_per_request_prefill_cuda_graph_flag(previous_graph_flag)


@pytest.mark.parametrize("parallel_moe_executor", [pytest.param(2, id="world2")], indirect=True)
def test_attention_dp_decoder_bounded_replay(parallel_moe_executor: MPIPoolExecutor | None) -> None:
    """Encoder recovery and decoder compaction preserve CSA2 and sharded MoE results."""
    if parallel_moe_executor is None:
        _run_dp_decoder_replay()
        return
    results = [parallel_moe_executor.submit(_run_dp_decoder_replay) for _ in range(2)]
    assert all(result.result() is None for result in results)
