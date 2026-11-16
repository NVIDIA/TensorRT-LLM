# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pinned-host PLE lookup, loading, and CUDA-graph parity."""

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.modules.qwen4_exp.ple import (
    PLEMetadata,
    Qwen4ExpNGramEmbedding,
    Qwen4ExpPinnedHostEmbedding,
    Qwen4ExpPLE,
    _uses_ple_host_offload,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


@pytest.fixture(autouse=True)
def _enable_ple_host_offload(monkeypatch) -> None:
    monkeypatch.setenv("TRTLLM_QWEN4_EXP_PLE_HOST_OFFLOAD", "1")


def _tiny_config(*, fp8: bool = True) -> SimpleNamespace:
    return SimpleNamespace(
        ngram_size=2,
        heads_per_ngram=1,
        vocab_size=16,
        eos_token_id=2,
        seed=1234,
        ngram_vocab_size_base=3,
        make_ngram_vocab_size_divisible_by=4,
        quantization_config={
            "quant_method": "fp8",
            "modules_to_not_convert": ([] if fp8 else ["ple.ple_embedding.ngram_embedding"]),
        },
    )


@pytest.mark.parametrize("table_dtype", [torch.bfloat16, torch.float8_e4m3fn])
def test_pinned_lookup_masks_rows_outside_local_shard(table_dtype) -> None:
    embedding = Qwen4ExpPinnedHostEmbedding(
        11,
        13,
        dtype=table_dtype,
        vocab_start_index=4,
        vocab_end_index=15,
    ).to("cuda")
    weight = embedding.weight
    values = torch.arange(weight.numel(), dtype=torch.float32).reshape_as(weight)
    values = (
        (values.remainder(32) - 16).to(table_dtype)
        if table_dtype == torch.float8_e4m3fn
        else values.to(table_dtype)
    )
    with torch.no_grad():
        weight.copy_(values)

    ids = torch.tensor([[3, 4, 14], [15, -1, 9]], device="cuda", dtype=torch.long)
    output = torch.full((*ids.shape, 13), -7, device="cuda", dtype=torch.bfloat16)
    output_ptr = output.data_ptr()
    result = embedding.gather(ids, out=output)

    expected = torch.zeros_like(result)
    valid = (ids >= 4) & (ids < 15)
    expected[valid] = values[(ids[valid] - 4).cpu()].to(torch.bfloat16).cuda()
    assert weight.device.type == "cpu" and weight.is_pinned()
    assert result.data_ptr() == output_ptr
    assert embedding._mapped_host_ptr == weight.data_ptr()
    torch.testing.assert_close(result, expected, rtol=0, atol=0)


def test_fp8_lookup_fuses_scale_and_preserves_table_address() -> None:
    module = Qwen4ExpNGramEmbedding(_tiny_config(), embedding_dim=2, dtype=torch.bfloat16).to(
        "cuda"
    )
    weight = module.ngram_embedding.weight
    table_ptr = weight.data_ptr()
    table = torch.tensor(
        [[-48.0, 72.0], [-80.0, 64.0], [-36.0, 36.0], [-26.0, 30.0]],
        dtype=torch.float8_e4m3fn,
    )
    with torch.no_grad():
        weight.copy_(table)
    scale = torch.tensor(0.0002, dtype=torch.bfloat16)
    module.configure_fp8_weight_storage(scale, torch.float8_e4m3fn)

    ids = torch.arange(4, device="cuda").reshape(4, 1)
    result = module.embed(ids).squeeze(1)
    expected = (table.float() * scale.item()).to(torch.bfloat16).cuda()
    assert module.ngram_embedding.weight.data_ptr() == table_ptr
    torch.testing.assert_close(result, expected, rtol=0, atol=0)

    module.to("cuda")
    assert module.ngram_embedding.weight.data_ptr() == table_ptr
    assert module.ngram_embedding.weight.device.type == "cpu"


@pytest.mark.parametrize("use_fp8", [False, True])
def test_mapper_loads_pinned_table_in_place(use_fp8, monkeypatch) -> None:
    from tensorrt_llm._torch.models.checkpoints.hf.qwen4_exp_weight_mapper import (
        Qwen4ExpHfWeightMapper,
    )

    module = Qwen4ExpNGramEmbedding(
        _tiny_config(fp8=use_fp8), embedding_dim=2, dtype=torch.bfloat16
    ).to("cuda")
    table = module.ngram_embedding.weight
    table_ptr = table.data_ptr()
    source_dtype = torch.float8_e4m3fn if use_fp8 else torch.bfloat16
    source = torch.tensor([[-4.0, 7.0], [-8.0, 6.0], [-3.0, 3.0], [-2.0, 5.0]], dtype=source_dtype)
    leaves = {
        "ngram_embedding.shard_1.weight": source[2:],
        "ngram_embedding.shard_0.weight": source[:2],
    }
    if use_fp8:
        leaves["ngram_embedding.weight_scale"] = torch.tensor(0.125, dtype=torch.bfloat16)

    mapper = Qwen4ExpHfWeightMapper()
    monkeypatch.setattr(mapper, "_ngram_module_for_prefix", lambda _prefix: module)
    mapper._load_ngram_tables({"model.layers.1.ple": leaves})

    assert table.device.type == "cpu" and table.is_pinned()
    assert table.data_ptr() == table_ptr
    torch.testing.assert_close(table, source, rtol=0, atol=0)


def test_lookup_replays_with_stable_cuda_graph_addresses() -> None:
    embedding = Qwen4ExpPinnedHostEmbedding(
        8,
        7,
        dtype=torch.bfloat16,
        vocab_start_index=0,
        vocab_end_index=8,
    ).to("cuda")
    with torch.no_grad():
        embedding.weight.copy_(torch.arange(56, dtype=torch.bfloat16).reshape(8, 7))
    static_ids = torch.tensor([0, 2, 7], device="cuda", dtype=torch.long)
    static_output = torch.empty((3, 7), device="cuda", dtype=torch.bfloat16)
    embedding.gather(static_ids, out=static_output)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        embedding.gather(static_ids, out=static_output)
    ids_ptr = static_ids.data_ptr()
    output_ptr = static_output.data_ptr()
    table_ptr = embedding.weight.data_ptr()
    static_ids.copy_(torch.tensor([6, 1, 4], device="cuda"))
    graph.replay()
    torch.cuda.synchronize()

    expected = embedding.weight[torch.tensor([6, 1, 4])].cuda()
    assert static_ids.data_ptr() == ids_ptr
    assert static_output.data_ptr() == output_ptr
    assert embedding.weight.data_ptr() == table_ptr
    torch.testing.assert_close(static_output, expected, rtol=0, atol=0)


def test_prefetch_fork_join_replays_in_cuda_graph() -> None:
    config = _tiny_config(fp8=False)
    config.hidden_size = 8
    config.ple_embed_dim = 4
    config.ple_conv_kernel_size = 2
    config.hc_count = 2
    config.rms_norm_eps = 1e-6
    module = Qwen4ExpPLE(config, dtype=torch.bfloat16, ple_layer_index=0, layer_id=1).to("cuda")
    table = module.ple_embedding.ngram_embedding.weight
    with torch.no_grad():
        table.copy_(torch.arange(table.numel(), dtype=torch.bfloat16).reshape_as(table))

    input_ids = torch.tensor([3, 4, 5], device="cuda", dtype=torch.long)
    state_indices = torch.arange(3, device="cuda", dtype=torch.long)
    metadata = PLEMetadata.build(
        input_ids,
        torch.ones(3, device="cuda", dtype=torch.long),
        state_indices,
        is_decode=True,
        eos_token_id=2,
        is_cuda_graph=True,
    )
    ngram_context = torch.full((3, 1), 2, device="cuda", dtype=torch.long)

    module.start_prefetch(metadata, ngram_context)
    module._consume_prefetched_embeddings(metadata)
    torch.cuda.synchronize()
    buffer_key = (torch.cuda.current_device(), 3)
    buffer_ptr = module._graph_prefetch_buffers[buffer_key].data_ptr()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        module.start_prefetch(metadata, ngram_context)
        graph_output, _ = module._consume_prefetched_embeddings(metadata)

    metadata.padded_tokens.copy_(torch.tensor([[6], [1], [7]], device="cuda"))
    graph.replay()
    torch.cuda.synchronize()
    _, ngram_ids = module._prepare_ngram_lookup(metadata, ngram_context)
    expected = table[ngram_ids.cpu()].cuda().flatten(start_dim=-2)
    assert module._graph_prefetch_buffers[buffer_key].data_ptr() == buffer_ptr
    assert module._prefetch_state is None
    torch.testing.assert_close(graph_output, expected, rtol=0, atol=0)


def test_host_offload_setting_is_strict(monkeypatch) -> None:
    monkeypatch.delenv("TRTLLM_QWEN4_EXP_PLE_HOST_OFFLOAD")
    assert not _uses_ple_host_offload()
    monkeypatch.setenv("TRTLLM_QWEN4_EXP_PLE_HOST_OFFLOAD", "yes")
    assert _uses_ple_host_offload()
    monkeypatch.setenv("TRTLLM_QWEN4_EXP_PLE_HOST_OFFLOAD", "sometimes")
    with pytest.raises(ValueError, match="must be a boolean"):
        _uses_ple_host_offload()


def test_aborted_prefetch_can_be_restarted() -> None:
    config = _tiny_config(fp8=False)
    config.hidden_size = 8
    config.ple_embed_dim = 4
    config.ple_conv_kernel_size = 2
    config.hc_count = 2
    config.rms_norm_eps = 1e-6
    module = Qwen4ExpPLE(config, dtype=torch.bfloat16, ple_layer_index=0, layer_id=1).to("cuda")
    metadata = PLEMetadata.build(
        torch.tensor([3], device="cuda"),
        torch.ones(1, device="cuda", dtype=torch.long),
        torch.zeros(1, device="cuda", dtype=torch.long),
        is_decode=True,
        eos_token_id=2,
    )
    ngram_context = torch.full((1, 1), 2, device="cuda", dtype=torch.long)

    module.start_prefetch(metadata, ngram_context)
    module.abort_prefetch()
    module.start_prefetch(metadata, ngram_context)
    module._consume_prefetched_embeddings(metadata)

    assert module._prefetch_state is None
