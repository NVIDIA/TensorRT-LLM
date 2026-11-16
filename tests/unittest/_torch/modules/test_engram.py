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
"""DeepSeek-V4.1 Engram prefill, disaggregated handoff and graphed decode."""

import contextlib
import multiprocessing
import os
import queue
import time
import uuid
from collections.abc import Sequence
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from cuda.bindings import driver as cuda
from utils.util import skip_pre_blackwell

from tensorrt_llm._torch.models import modeling_deepseekv41 as v41
from tensorrt_llm._torch.models.modeling_utils import MetaInitMode
from tensorrt_llm._torch.modules.engram import EngramConfig, EngramHashProvider
from tensorrt_llm._torch.modules.engram import _host_memory as host_memory
from tensorrt_llm._torch.modules.engram import engram as engram_module
from tensorrt_llm._torch.modules.engram import functional as engram_functional
from tensorrt_llm._torch.modules.engram.projection import EngramFp8Projection
from tensorrt_llm._torch.modules.linear import (
    flashinfer_mxfp8_autotune,
    flashinfer_mxfp8_decode_graph_capture,
)


def _reference_v41_engram_gate(
    hidden: torch.Tensor,
    kv: torch.Tensor,
    query_weight: torch.Tensor,
    key_weight: torch.Tensor,
    eps: float,
    add_residual: bool,
) -> torch.Tensor:
    """Keys-first gate with a single cast after the FP32 residual addition."""
    _, hc, dim = hidden.shape
    keys, value = kv.float().split([hc * dim, dim], dim=-1)
    keys = keys.reshape(-1, hc, dim)
    h = hidden.float()
    weight = query_weight.float() * key_weight.float()
    rstd = (h.square().mean(-1) + eps).rsqrt() * (keys.square().mean(-1) + eps).rsqrt()
    dot = (h * weight * keys).sum(-1) * rstd * dim**-0.5
    gate = torch.sigmoid(torch.copysign(dot.abs().clamp_min(1e-6).sqrt(), dot))
    value = gate.unsqueeze(-1) * value.unsqueeze(-2)
    return (h + value if add_residual else value).to(hidden.dtype)


def _hashes(
    provider: EngramHashProvider, tokens: list[int], positions: Sequence[int]
) -> torch.Tensor:
    return provider.compute_hashes(
        torch.tensor(tokens, dtype=torch.long, device="cuda"),
        position_ids=torch.tensor(positions, dtype=torch.long, device="cuda"),
        request_ids=[41],
        seq_lens_host=torch.tensor([len(tokens)], dtype=torch.int32),
        max_seq_len=64,
    )[1].clone()


@skip_pre_blackwell
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Engram FP8 execution requires CUDA")
@pytest.mark.parametrize("add_residual", [False, True])
@torch.no_grad()
def test_v41_engram_gate_parity(monkeypatch: pytest.MonkeyPatch, add_residual: bool) -> None:
    if torch.cuda.get_device_capability()[0] != 10 or not hasattr(
        torch.ops.trtllm, "mxfp8_mxfp8_gemm"
    ):
        pytest.skip("Native MXFP8 Engram WKV requires a compiled Blackwell backend")
    monkeypatch.delenv("TRTLLM_MXFP8_GEMM_BACKEND", raising=False)
    torch.manual_seed(151)
    buckets = [101, 103, 107, 109]
    config = EngramConfig(
        layer_ids=[1],
        max_ngram_size=3,
        n_head_per_ngram=2,
        n_embed_per_ngram=256,
        hidden_size=256,
        hc_mult=4,
        norm_eps=1e-20,
        dtype=torch.bfloat16,
    )
    # Fix tokenizer metadata, not the hashing or request-history implementation.
    mapping = SimpleNamespace(
        compressed_tokenizer=SimpleNamespace(lookup_table=torch.arange(256) // 2),
        pad_id=0,
        layer_multipliers={1: torch.tensor([17, 31, 47])},
        vocab_size_across_layers={1: [buckets[:2], buckets[2:]]},
    )
    monkeypatch.setattr(engram_module, "NgramHashMapping", lambda **kwargs: mapping)
    aggregate, generation = EngramHashProvider(config), EngramHashProvider(config)
    with torch.device("cuda"):
        module = v41.DeepseekV41Engram(1, config, vocab_sizes_flat=buckets)
    table = module.multi_head_embedding
    table_pointers = (table.weight.data_ptr(), table.scale.data_ptr())
    module.cuda()
    table.to(device="cuda", dtype=torch.bfloat16)
    assert table._requires_standard_hf_loading
    assert table.weight.device.type == table.scale.device.type == "cpu"
    assert table.weight.is_pinned() and table.scale.is_pinned()
    assert table.weight.dtype == torch.float8_e4m3fn
    assert table.scale.dtype == torch.float8_e8m0fnu
    assert table_pointers == (table.weight.data_ptr(), table.scale.data_ptr())
    assert module.short_conv is None
    assert isinstance(module.kv_proj, EngramFp8Projection)
    method = module.kv_proj.quant_method
    assert method.use_cutlass, "dequantized WKV is not native FP8 execution"
    assert method.backend == "trtllm"
    assert module.kv_proj._use_flashinfer_mxfp8_decode_graph_default

    table_weight = torch.randn(sum(buckets), 128).to(torch.float8_e4m3fn)
    table_scale = torch.randint(125, 130, (sum(buckets), 4), dtype=torch.uint8)
    table.load_weights([{"weight": table_weight, "scale": table_scale.view(torch.float8_e8m0fnu)}])
    weight = torch.randn(1280, 512).to(torch.float8_e4m3fn)
    scale = torch.randint(123, 130, (40, 16), dtype=torch.uint8)
    checkpoint = {"layers.1.engram.wkv.weight": weight, "layers.1.engram.wkv.scale": scale}
    forwarded = v41._remap_deepseek_v41_checkpoint_keys(checkpoint, num_hidden_layers=2)
    stem = "model.layers.1.engram.kv_proj"
    assert forwarded[stem + ".weight"] is weight
    assert forwarded[stem + ".scale"] is scale
    assert forwarded.census.folded == 0
    module.kv_proj.load_weights(
        [{"weight": forwarded[stem + ".weight"], "scale": forwarded[stem + ".scale"]}]
    )
    torch.testing.assert_close(
        module.kv_proj.weight.view(torch.uint8).cpu(), weight.view(torch.uint8), rtol=0, atol=0
    )
    module.query_norm_weight.copy_(torch.randn_like(module.query_norm_weight))
    module.key_norm_weight.copy_(torch.randn_like(module.key_norm_weight))
    module.post_load_weights()
    assert module._gate_norm_product.dtype == torch.float32
    pointers = (
        table.weight.data_ptr(),
        table.scale.data_ptr(),
        module._gate_norm_product.data_ptr(),
    )
    module.warmup_kernels()

    if add_residual:
        # Release gate geometry, with a small unused projection and table.
        gate_config = replace(config, hidden_size=5120, n_embed_per_ngram=64)
        with torch.device("cuda"):
            gate_module = v41.DeepseekV41Engram(1, gate_config, vocab_sizes_flat=buckets)
        gate_module.query_norm_weight.copy_(torch.randn_like(gate_module.query_norm_weight))
        gate_module.key_norm_weight.copy_(torch.randn_like(gate_module.key_norm_weight))
        gate_module.post_load_weights()
        launch = Mock(wraps=engram_functional._engram_gate_kernel.run)
        with monkeypatch.context() as gate_patch:
            gate_patch.setattr(engram_functional._engram_gate_kernel, "run", launch)
            gate_module.warmup_kernels()
        assert [call.kwargs["PRECOMPUTED_WEIGHT"] for call in launch.call_args_list] == [
            False,
            torch.cuda.get_device_capability() == (10, 3),
        ]
        gate_hidden = torch.randn(3, 4, 5120, dtype=torch.bfloat16, device="cuda")
        gate_kv = torch.randn(3, 25600, dtype=torch.bfloat16, device="cuda")
        expected_gate = _reference_v41_engram_gate(
            gate_hidden,
            gate_kv,
            gate_module.query_norm_weight,
            gate_module.key_norm_weight,
            gate_config.norm_eps,
            True,
        )
        for product in (None, gate_module._gate_norm_product):
            actual_gate = v41.engram_gate(
                gate_hidden,
                gate_kv,
                gate_module.query_norm_weight,
                gate_module.key_norm_weight,
                gate_config.norm_eps,
                add_residual=True,
                norm_weight_product=product,
            )
            torch.testing.assert_close(actual_gate, expected_gate, rtol=8e-3, atol=2e-6)

    table_reference = (
        table_weight.float() * torch.exp2(table_scale.float() - 127).repeat_interleave(32, 1)
    ).cuda()
    reference_offsets = torch.tensor(
        [sum(buckets[:head]) for head in range(len(buckets))], device="cuda"
    )
    expanded_scale = (
        torch.exp2(scale.float() - 127).repeat_interleave(32, 0).repeat_interleave(32, 1)
    )
    projection_reference = (weight.float() * expanded_scale).cuda().t()

    def check_output(
        hidden: torch.Tensor,
        hashes: torch.Tensor,
        embeddings: torch.Tensor,
        kv: torch.Tensor,
        actual: torch.Tensor,
    ) -> None:
        expected_embeddings = (
            table_reference[hashes + reference_offsets].flatten(1).to(hidden.dtype)
        )
        torch.testing.assert_close(embeddings, expected_embeddings, rtol=0, atol=0)
        reference_kv = expected_embeddings.float() @ projection_reference
        assert torch.isfinite(kv).all() and reference_kv.norm() > 0
        # Preserve the native W8A8 tolerance; checkpoint bytes and lookup are exact.
        assert (kv.float() - reference_kv).norm() / reference_kv.norm() < 0.04
        expected = _reference_v41_engram_gate(
            hidden,
            kv,
            module.query_norm_weight,
            module.key_norm_weight,
            config.norm_eps,
            add_residual,
        )
        assert actual.dtype == torch.bfloat16 and actual.is_contiguous()
        assert torch.isfinite(actual).all()
        torch.testing.assert_close(actual, expected, rtol=8e-3, atol=2e-6)

    prompt = [11, 13, 15]
    hashes = _hashes(aggregate, prompt, range(len(prompt)))
    hidden = torch.randn(3, 4, 256, dtype=torch.bfloat16, device="cuda")
    embeddings = module.precompute(hashes)
    kv = module.precompute_kv(embeddings)
    actual = module(hidden, embeddings, add_residual=add_residual)
    check_output(hidden, hashes, embeddings, kv, actual)

    input_ids = torch.tensor([17], device="cuda")
    position_ids = torch.tensor([3], device="cuda")
    lengths = torch.tensor([1], dtype=torch.int32)
    hidden = torch.randn(1, 4, 256, dtype=torch.bfloat16, device="cuda")
    generation.compute_hashes(input_ids)
    captured_hashes = generation._cached_hashes[1]
    hash_pointer = captured_hashes.data_ptr()

    def forward() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        embeddings = module.precompute(generation.compute_hashes(input_ids)[1])
        kv = module.precompute_kv(embeddings)
        return (
            embeddings,
            kv,
            module(hidden, embeddings, add_residual=add_residual, precomputed_kv=kv),
        )

    assert method.enable_flashinfer_auto(), (
        "the pinned Blackwell runtime must provide MXFP8 FlashInfer"
    )
    with flashinfer_mxfp8_autotune():
        forward()
    method.mark_flashinfer_autotuned()
    flashinfer = Mock(wraps=method._flashinfer_mxfp8)
    monkeypatch.setattr(method, "_flashinfer_mxfp8", flashinfer)
    warm = torch.cuda.Stream()
    warm.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warm):
        for _ in range(3):
            forward()
    torch.cuda.current_stream().wait_stream(warm)
    torch.cuda.synchronize()
    assert flashinfer.call_count == 0, "eager execution must retain its native backend"
    graph = torch.cuda.CUDAGraph()
    with flashinfer_mxfp8_decode_graph_capture(), torch.cuda.graph(graph):
        embeddings, kv, actual = forward()
    capture_calls = flashinfer.call_count
    assert capture_calls > 0, "decode capture must use the tuned FlashInfer kernel"

    reads = []

    def get_tokens_range(beam: int, start: int, end: int) -> list[int]:
        reads.append((beam, start, end))
        return prompt[start:end]

    model = SimpleNamespace(model=SimpleNamespace(use_engram=True, engram_hash_provider=generation))
    request = SimpleNamespace(py_request_id=41, prompt_len=3, get_tokens_range=get_tokens_range)
    v41.DeepseekV41ForCausalLM.prepare_disagg_generation_request(model, request)
    assert reads == [(0, 1, 3)]
    assert 41 in generation._pending_history_seeds
    previous_hashes = None
    for position, token in enumerate([21, 23, 25], start=len(prompt)):
        input_ids.fill_(token)
        position_ids.fill_(position)
        if position < 5:
            hidden.copy_(torch.randn_like(hidden))
        else:
            hidden.zero_()
        refreshed = generation.refresh_captured_hashes(
            input_ids,
            position_ids=position_ids,
            request_ids=[41],
            seq_lens_host=lengths,
            max_seq_len=64,
        )[1]
        assert refreshed.data_ptr() == hash_pointer
        expected_hashes = _hashes(aggregate, [token], [position])
        torch.testing.assert_close(refreshed, expected_hashes, rtol=0, atol=0)
        if previous_hashes is not None:
            assert not torch.equal(refreshed, previous_hashes)
        previous_hashes = refreshed.clone()
        graph.replay()
        torch.cuda.synchronize()
        check_output(hidden, expected_hashes, embeddings, kv, actual)
        native_kv = module.precompute_kv(embeddings)
        assert flashinfer.call_count == capture_calls
        assert (kv.float() - native_kv.float()).norm() / native_kv.float().norm() < 1e-3
        assert not generation._pending_history_seeds
        assert pointers == (
            table.weight.data_ptr(),
            table.scale.data_ptr(),
            module._gate_norm_product.data_ptr(),
        )
    assert reads == [(0, 1, 3)], "later decode steps must not reseed prompt history"
    v41.DeepseekV41ForCausalLM.release_request_state(model, 41)
    assert 41 not in generation._pending_history_seeds
    assert 41 not in generation._history_row_of


def _check_host_numa(tensor: torch.Tensor) -> None:
    """``tensor`` is pinned and, with host-NUMA VMM, mapped on the GPU's host NUMA node."""
    assert tensor.device.type == "cpu" and tensor.is_pinned()
    numa_node = engram_module.host_numa_node(torch.cuda.current_device())
    if numa_node is None:
        return
    _, handle = cuda.cuMemRetainAllocationHandle(tensor.data_ptr())
    _, properties = cuda.cuMemGetAllocationPropertiesFromHandle(handle)
    cuda.cuMemRelease(handle)
    assert properties.location.type == cuda.CUmemLocationType.CU_MEM_LOCATION_TYPE_HOST_NUMA
    assert properties.location.id == numa_node


def _sharded_lookup_reference(
    table: engram_module.ShardedFp8MultiHeadEmbedding, indices: torch.Tensor
) -> torch.Tensor:
    """Dequantize owned rows on the host; row shards return zero for unowned rows."""
    heads = slice(table.head_start, table.head_start + table.num_local_heads)
    index = indices[:, heads].cpu() + table.offsets.cpu()
    if not table.shard_heads:
        owned = (index >= table.vocab_start_idx) & (index < table.vocab_end_idx)
        index = torch.where(owned, index - table.vocab_start_idx, 0)
    scale = torch.exp2(table.scale.view(torch.uint8).float() - 127)
    rows = table.weight.float() * scale.repeat_interleave(table.block_size, dim=1)
    values = rows[index]
    if not table.shard_heads:
        values = torch.where(owned[..., None], values, 0)
    return values.to(table.dtype)


def _host_table(
    buckets: list[int], tp_size: int, tp_rank: int, device: int = 0, *, check_numa: bool = True
) -> engram_module.ShardedFp8MultiHeadEmbedding:
    """Build, move and fill a table whose weight and scale shards stay in host memory."""
    with torch.device(f"cuda:{device}"):
        table = engram_module.ShardedFp8MultiHeadEmbedding(
            buckets, 128, dtype=torch.bfloat16, tp_size=tp_size, tp_rank=tp_rank
        )
    pointers = (table.weight.data_ptr(), table.scale.data_ptr())
    table.cuda(device)
    assert pointers == (table.weight.data_ptr(), table.scale.data_ptr())
    if check_numa:
        _check_host_numa(table.weight)
        _check_host_numa(table.scale)
    rows = sum(buckets)
    table.load_weights(
        [
            {
                "weight": torch.randn(rows, 128).to(torch.float8_e4m3fn),
                "scale": torch.randint(120, 135, (rows, 4), dtype=torch.uint8).view(
                    torch.float8_e8m0fnu
                ),
            }
        ]
    )
    # The device view aliases the host table instead of copying it.
    weight_view, scale_view = table.storage()
    assert weight_view.device == scale_view.device == torch.device("cuda", device)
    assert weight_view.data_ptr() == table.weight.data_ptr()
    assert scale_view.data_ptr() == table.scale.data_ptr()
    return table


@pytest.mark.skipif(not torch.cuda.is_available(), reason="UVA lookup requires CUDA")
@pytest.mark.parametrize(
    "num_heads,tp_size,tp_rank",
    [(4, 2, 1), (3, 2, 1)],
    ids=["head_shard-background", "row_shard-background"],
)
@torch.no_grad()
def test_sharded_fp8_lookup_host_tables(num_heads: int, tp_size: int, tp_rank: int) -> None:
    """Host tables map on the GPU's host NUMA node and look up exactly."""
    torch.manual_seed(7)
    buckets = [16411 + 6 * head for head in range(num_heads)]
    table = _host_table(buckets, tp_size, tp_rank)
    assert table.shard_heads == (tp_size > 1 and num_heads % tp_size == 0)
    indices = torch.stack([torch.randint(0, n, (517,)) for n in buckets], dim=1).cuda()
    actual = table(indices, background=True)
    torch.cuda.synchronize()
    expected = _sharded_lookup_reference(table, indices)
    assert expected.count_nonzero() > 0
    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="UVA lookup requires CUDA")
@pytest.mark.parametrize(
    "num_heads", [4, 3], ids=["head_shard-background-gpu0", "row_shard-background-gpu0"]
)
@torch.no_grad()
def test_sharded_fp8_lookup_after_host_numa_oom(
    monkeypatch: pytest.MonkeyPatch, num_heads: int
) -> None:
    """A local NUMA OOM that outlasts page-cache eviction preserves exact host-table lookup for both shard layouts."""
    if host_memory.host_numa_node(0) is None:
        pytest.skip("host NUMA VMM is unavailable")
    create = Mock(return_value=(cuda.CUresult.CUDA_ERROR_OUT_OF_MEMORY,))
    monkeypatch.setattr(cuda, "cuMemCreate", create)
    # A free eviction lock and an eviction that reports success without running its child: each table
    # retries once, then falls back.
    monkeypatch.setattr(
        host_memory, "_eviction_lock", lambda mode: contextlib.nullcontext((-1, False))
    )
    evict = Mock(return_value=True)
    monkeypatch.setattr(host_memory, "evict_numa_node_page_cache", evict)
    torch.manual_seed(11)
    buckets = [16411 + 6 * head for head in range(num_heads)]
    table = _host_table(buckets, tp_size=2, tp_rank=1, check_numa=False)
    assert create.call_count == 4 and evict.call_count == 2
    assert table.weight.is_pinned() and table.scale.is_pinned()
    indices = torch.stack([torch.randint(0, n, (517,)) for n in buckets], dim=1).cuda()
    actual = table(indices, background=True)
    torch.cuda.synchronize()
    expected = _sharded_lookup_reference(table, indices)
    assert expected.count_nonzero() > 0
    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)


@skip_pre_blackwell
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Engram hashing requires CUDA")
def test_engram_hashes_retain_only_captured_and_latest_shapes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Context passes of many token counts keep only the captured shape's buffers and the latest
    pass's, match the general path exactly, and a graph captured in between keeps replaying right."""
    layers, ngram, heads = (1, 15), 4, 3
    bound = torch.iinfo(torch.int64).max // 256
    mapping = SimpleNamespace(
        compressed_tokenizer=SimpleNamespace(lookup_table=torch.arange(256) // 2),
        pad_id=0,
        layer_multipliers={
            layer: torch.tensor([bound - 2 * (layer + i) for i in range(ngram)]) for layer in layers
        },
        vocab_size_across_layers={
            layer: [[101 + 2 * (o * heads + h) for h in range(heads)] for o in range(ngram - 1)]
            for layer in layers
        },
    )
    monkeypatch.setattr(engram_module, "NgramHashMapping", lambda **kwargs: mapping)
    config = EngramConfig(layer_ids=list(layers), max_ngram_size=ngram, n_head_per_ngram=heads)
    provider, reference = EngramHashProvider(config), EngramHashProvider(config)
    monkeypatch.setattr(reference, "_compute_decode_hashes", lambda *args: None)
    generator = torch.Generator().manual_seed(29)

    def arguments(requests: dict[int, list[int]], padded: int | None = None) -> dict:
        return dict(
            input_ids=torch.tensor(sum(requests.values(), []), device="cuda"),
            position_ids=torch.tensor(
                [p for tokens in requests.values() for p in range(len(tokens))], device="cuda"
            ),
            request_ids=list(requests),
            seq_lens_host=torch.tensor([len(t) for t in requests.values()], dtype=torch.int32),
            max_seq_len=1024,
            padded_num_tokens=padded,
        )

    def release(requests: dict[int, list[int]]) -> None:
        for request_id in requests:
            provider.release_request_state(request_id)
            reference.release_request_state(request_id)

    def decode(step: int) -> dict[int, list[int]]:
        return {9000 + 10 * step + i: [(7 * step + 3 * i) % 256] for i in range(3)}

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    capture = arguments(decode(0), padded=4)
    with torch.cuda.stream(stream):
        warm = provider.compute_hashes(**capture)
    stream.synchronize()
    pointers = {layer: value.data_ptr() for layer, value in warm.items()}
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        captured = {
            layer: value.clone() for layer, value in provider.compute_hashes(**capture).items()
        }
    release(decode(0))
    captured_shape = tuple((4, (ngram - 1) * heads) for _ in layers)

    sizes = [1, 3, 7, 16, 31, 64, 97, 128, 200, 255, 333, 401, 512, 640, 777, 1000, 90, 45, 9, 2]
    for step, size in enumerate(sizes, start=1):
        requests = {step: torch.randint(0, 256, (size,), generator=generator).tolist()}
        kwargs = arguments(requests)
        actual, expected = provider.compute_hashes(**kwargs), reference.compute_hashes(**kwargs)
        for layer in layers:
            torch.testing.assert_close(actual[layer], expected[layer], rtol=0, atol=0)
        release(requests)
        retained = set(provider._cached_hashes_store)
        assert retained == {captured_shape, tuple((size, (ngram - 1) * heads) for _ in layers)}
        if step % 4 == 0:
            kwargs = arguments(decode(step), padded=4)
            with torch.cuda.stream(stream):
                refreshed = provider.refresh_captured_hashes(**kwargs)
                expected = reference.compute_hashes(**kwargs)
                graph.replay()
            stream.synchronize()
            assert {layer: value.data_ptr() for layer, value in refreshed.items()} == pointers
            for layer in layers:
                torch.testing.assert_close(captured[layer], expected[layer], rtol=0, atol=0)
            release(decode(step))


class _OwnerRows:
    """Checkpoint rows whose first read, by the owner, lets the next processes start; with
    ``imported`` it reads its rows only once an importer finished ``load_weights``."""

    def __init__(self, rows: torch.Tensor, loading, imported=None) -> None:
        self.rows, self.loading, self.imported = rows, loading, imported

    def __getitem__(self, index):
        self.loading.set()
        # Meanwhile the next processes join this owner.
        if self.imported is not None:
            if not self.imported.wait(timeout=120):
                raise TimeoutError("an importer waited for the table inside load_weights")
        else:
            time.sleep(3)
        return self.rows[index]


def _node_shared_table_process(
    rank: int, scenario: str, key: str, loading, imported, write, written, results
) -> None:
    """A worker builds a table with the share key as the model loader does and looks it up; then the
    owner overwrites a row and every other process reports whether its table shows it."""
    torch.cuda.set_device(0)
    torch.manual_seed(13)
    buckets = [16411 + 6 * head for head in range(4)]
    rows = sum(buckets)
    checkpoint = {
        "weight": torch.randn(rows, 128).to(torch.float8_e4m3fn),
        "scale": torch.randint(120, 135, (rows, 4), dtype=torch.uint8).view(torch.float8_e8m0fnu),
    }
    indices = torch.stack([torch.randint(0, n, (517,)) for n in buckets], dim=1).cuda()
    with MetaInitMode():
        plain = engram_module.ShardedFp8MultiHeadEmbedding(buckets, 128, dtype=torch.bfloat16)
    plain.cuda()
    plain.load_weights([checkpoint])
    with torch.no_grad():
        expected = plain(indices).cpu()
    del plain

    if rank == 0:
        if scenario == "owner_numa_oom":
            cuda.cuMemCreate = Mock(return_value=(cuda.CUresult.CUDA_ERROR_OUT_OF_MEMORY,))
            # Eviction reports success without running its child, so the owner retries once.
            host_memory.evict_numa_node_page_cache = Mock(return_value=True)
        # A rank of a multi-rank worker must not wait inside load_weights for another worker.
        gate = imported if scenario == "import_before_fill" else None
        checkpoint = {
            **checkpoint,
            "weight": _OwnerRows(checkpoint["weight"], loading, gate),
        }
    if rank > 0 and not loading.wait(timeout=300):
        raise TimeoutError("the owner did not start loading the table")
    with MetaInitMode():
        table = engram_module.ShardedFp8MultiHeadEmbedding(
            buckets, 128, dtype=torch.bfloat16, share_key=key
        )
    table.cuda()
    table.load_weights([checkpoint])
    if rank == 1:
        imported.set()
    table.post_load_weights()
    with torch.no_grad():
        results.put((rank, torch.equal(table(indices).cpu(), expected)))
    row = table.weight.view(torch.uint8)[0]
    if rank == 0:
        if not write.wait(timeout=600):
            raise TimeoutError("no write request")
        row.fill_(0x5A)
        written.set()
    else:
        if not written.wait(timeout=600):
            raise TimeoutError("the owner did not write")
        results.put((rank, bool((row == 0x5A).all())))


def _collect(results, count: int, processes, deadline: float) -> list:
    """``count`` results of the table processes; fails once one of them exits with an error."""
    outcomes = []
    while len(outcomes) < count:
        try:
            outcomes.append(results.get(timeout=5))
        except queue.Empty:
            assert all(process.exitcode in (None, 0) for process in processes)
            assert time.monotonic() < deadline, "the table processes timed out"
    return outcomes


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs two GPUs")
@pytest.mark.parametrize("scenario", ["owner_numa_oom", "import_before_fill"])
def test_sharded_fp8_tables_node_shared(monkeypatch: pytest.MonkeyPatch, scenario: str) -> None:
    """One process per GPU loads a node-shared table and the others map that copy in place, finish
    load_weights before the owner fills it and wait in post_load_weights."""
    if host_memory.host_numa_node(0) is None:
        pytest.skip("host NUMA VMM is unavailable")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    gpus = visible.split(",") if visible else [str(i) for i in range(torch.cuda.device_count())]
    gpus = gpus[:4]
    context = multiprocessing.get_context("spawn")
    loading = context.Event()
    imported, write, written = context.Event(), context.Event(), context.Event()
    results = context.Queue()
    key = uuid.uuid4().hex
    processes = [
        context.Process(
            target=_node_shared_table_process,
            args=(rank, scenario, key, loading, imported, write, written, results),
        )
        for rank in range(len(gpus))
    ]
    for gpu, process in zip(gpus, processes):
        # Launchers start each single-process worker with only its own GPU visible.
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", gpu)
        process.start()
    try:
        deadline = time.monotonic() + 600
        lookups = _collect(results, len(processes), processes, deadline)
        write.set()
        sightings = _collect(results, len(processes) - 1, processes, deadline)
    finally:
        write.set()
        written.set()
        for process in processes:
            process.join(timeout=120)
        for process in processes:
            if process.is_alive():
                process.kill()
                process.join()
    assert all(process.exitcode == 0 for process in processes)
    assert all(equal for _, equal in lookups)
    # Importers read the owner's memory in place.
    assert all(seen for _, seen in sightings)


def _allocation_numa_node(tensor: torch.Tensor) -> int | None:
    """Host NUMA node of the VMM allocation behind ``tensor``; None for other memory."""
    result, handle = cuda.cuMemRetainAllocationHandle(tensor.data_ptr())
    if result != cuda.CUresult.CUDA_SUCCESS:
        return None
    _, properties = cuda.cuMemGetAllocationPropertiesFromHandle(handle)
    cuda.cuMemRelease(handle)
    if properties.location.type != cuda.CUmemLocationType.CU_MEM_LOCATION_TYPE_HOST_NUMA:
        return None
    return properties.location.id


def _numa_placement_process(
    gpu: str, tp_size: int, tp_rank: int, after, joining, done, wait_s: float, key: str, results
) -> None:
    """Join shard ``tp_rank`` of a node-shared table (2 s after ``after`` is set, if given); report
    ownership, the owner's allocation node, and whether rows and lookups match a private copy."""
    torch.cuda.set_device(0)
    engram_module._OFF_HOME_CLAIM_WAIT_S = wait_s
    torch.manual_seed(17)
    buckets = [16411 + 6 * head for head in range(4)]
    rows = sum(buckets)
    checkpoint = {
        "weight": torch.randn(rows, 128).to(torch.float8_e4m3fn),
        "scale": torch.randint(120, 135, (rows, 4), dtype=torch.uint8).view(torch.float8_e8m0fnu),
    }
    indices = torch.stack([torch.randint(0, n, (517,)) for n in buckets], dim=1).cuda()
    shard = {"tp_size": tp_size, "tp_rank": tp_rank}
    with MetaInitMode():
        private = engram_module.ShardedFp8MultiHeadEmbedding(
            buckets, 128, dtype=torch.bfloat16, **shard
        )
    private.cuda()
    private.load_weights([checkpoint])
    with torch.no_grad():
        expected = private(indices).cpu()
    expected_rows = [tensor.view(torch.uint8).clone() for tensor in (private.weight, private.scale)]
    del private

    if after is not None:
        if not after.wait(timeout=300):
            raise TimeoutError("the worker to follow did not start joining")
        time.sleep(2)
    joining.set()
    with MetaInitMode():
        table = engram_module.ShardedFp8MultiHeadEmbedding(
            buckets, 128, dtype=torch.bfloat16, share_key=key, **shard
        )
    table.cuda()
    table.load_weights([checkpoint])
    table.post_load_weights()
    with torch.no_grad():
        actual = table(indices).cpu()
    shared = table._share is not None
    owner = shared and table._share._listener is not None
    rows_equal = all(
        torch.equal(tensor.view(torch.uint8), rows)
        for tensor, rows in zip((table.weight, table.scale), expected_rows)
    )
    lookups_equal = torch.equal(actual.view(torch.int16), expected.view(torch.int16))
    node = _allocation_numa_node(table.weight) if owner else None
    results.put((gpu, owner, shared, node, rows_equal, lookups_equal))
    # Owners serve their tables until every worker has reported, as a running server would.
    if not done.wait(timeout=600):
        raise TimeoutError("the test did not finish")


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs GPUs on two host NUMA nodes")
def test_sharded_fp8_tables_node_shared_numa_placement(monkeypatch: pytest.MonkeyPatch) -> None:
    """Shard r of a 2-way split is owned and allocated on host node r although a worker off that
    node joins first, and rows and lookups match a private copy bit for bit."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    gpus = visible.split(",") if visible else [str(i) for i in range(torch.cuda.device_count())]
    gpu_nodes = [host_memory.host_numa_node(index) for index in range(len(gpus))]
    if None in gpu_nodes or len(set(gpu_nodes)) < 2:
        pytest.skip("needs host NUMA VMM and GPUs on two host NUMA nodes")
    host_nodes = sorted(set(gpu_nodes))
    by_node = [[g for g, n in zip(gpus, gpu_nodes) if n == node] for node in host_nodes[:2]]
    if min(len(on_node) for on_node in by_node) < 2:
        pytest.skip("needs two GPUs on each of two host NUMA nodes")
    (a0, a1), (b0, b1) = by_node[0][:2], by_node[1][:2]
    # Workers as (gpu, tp_size, tp_rank, index of the worker this one joins after): each shard has a
    # worker on both nodes, like DEP2 instances on GPUs (0, 2) and (3, 1), and the one off the
    # shard's home node joins first.
    wait_s = 60.0
    workers = [(a0, 2, 0, 1), (b0, 2, 0, None), (b1, 2, 1, 3), (a1, 2, 1, None)]
    owners = {a0: host_nodes[0], b1: host_nodes[1]}
    context = multiprocessing.get_context("spawn")
    joining = [context.Event() for _ in workers]
    done = context.Event()
    results = context.Queue()
    key = uuid.uuid4().hex
    processes = []
    for index, (gpu, tp_size, tp_rank, after) in enumerate(workers):
        process = context.Process(
            target=_numa_placement_process,
            args=(
                gpu,
                tp_size,
                tp_rank,
                None if after is None else joining[after],
                joining[index],
                done,
                wait_s,
                key,
                results,
            ),
        )
        # Launchers start each single-process worker with only its own GPU visible.
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", gpu)
        process.start()
        processes.append(process)
    try:
        outcomes = _collect(results, len(processes), processes, time.monotonic() + 600)
    finally:
        done.set()
        for process in processes:
            process.join(timeout=120)
        for process in processes:
            if process.is_alive():
                process.kill()
                process.join()
    assert all(process.exitcode == 0 for process in processes)
    assert all(shared and rows and lookups for _, _, shared, _, rows, lookups in outcomes), outcomes
    owned = {gpu: node for gpu, owner, _, node, _, _ in outcomes if owner}
    assert owned == owners, owned


@skip_pre_blackwell
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Engram FP8 execution requires CUDA")
@pytest.mark.parametrize("tp", [4, 8])
@torch.no_grad()
def test_v41_wkv_shards_match_replicated_columns(monkeypatch: pytest.MonkeyPatch, tp: int) -> None:
    """Each rank's column shard of the precomputed Engram WKV equals the replicated
    projection's columns bit for bit, eagerly (native GEMM) and with the tuned decode-graph
    backends, at every decode size the precompute serves."""
    if torch.cuda.get_device_capability()[0] != 10 or not hasattr(
        torch.ops.trtllm, "mxfp8_mxfp8_gemm"
    ):
        pytest.skip("Native MXFP8 Engram WKV requires a compiled Blackwell backend")
    monkeypatch.delenv("TRTLLM_MXFP8_GEMM_BACKEND", raising=False)
    torch.manual_seed(173)
    out_features, in_features = 2048, 512
    with torch.device("cuda"):
        full = EngramFp8Projection(in_features, out_features, torch.bfloat16)
    weight = torch.randn(out_features, in_features).to(torch.float8_e4m3fn)
    scale = torch.randint(118, 126, (out_features // 32, in_features // 32), dtype=torch.uint8)
    full.load_weights([{"weight": weight, "scale": scale}])
    shards = [v41._wkv_shard_projection(full, rank, tp) for rank in range(tp)]
    rows = out_features // tp
    for rank, shard in enumerate(shards):
        assert shard.out_features == rows
        # The E4M3 rows are a view of the replicated weight, not a copy.
        assert shard.weight.data_ptr() == full.weight.data_ptr() + rank * rows * in_features
    inputs = {
        m: torch.randn(m, in_features, device="cuda").bfloat16() * 3 for m in (1, 6, 40, 48, 64)
    }

    def check() -> None:
        for m, x in inputs.items():
            expected = full(x)
            actual = torch.cat([shard(x) for shard in shards], dim=1)
            assert torch.equal(actual, expected), f"M={m}"

    check()
    methods = [full.quant_method] + [shard.quant_method for shard in shards]
    if not all(method.enable_flashinfer_auto() for method in methods):
        pytest.skip("the pinned Blackwell runtime must provide MXFP8 FlashInfer")
    with flashinfer_mxfp8_autotune(), flashinfer_mxfp8_decode_graph_capture():
        check()
    for method in methods:
        method.mark_flashinfer_autotuned()
    with flashinfer_mxfp8_decode_graph_capture():
        check()
