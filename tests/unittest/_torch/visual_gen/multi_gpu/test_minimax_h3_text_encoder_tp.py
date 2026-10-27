# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Real Qwen3-VL layer-50 parity with full-world and subset text-encoder TP."""

from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextModel

from tensorrt_llm._torch.visual_gen.models.minimax_h3.packing import MINIMAX_H3_TEXT_ENCODER_LAYER
from tensorrt_llm._torch.visual_gen.models.minimax_h3.pipeline_minimax_h3 import (
    _MiniMaxH3TextEncoderMapping,
)
from tensorrt_llm._torch.visual_gen.models.minimax_h3.text_encoder_tp import (
    MiniMaxH3TensorParallelTextEncoder,
)


def _assert_projection_shards(
    candidate: MiniMaxH3TensorParallelTextEncoder,
    reference: Qwen3VLTextModel,
    group: dist.ProcessGroup,
    rank: int,
) -> None:
    embedding = candidate.model.model.embed_tokens
    mapping = embedding.mapping
    assert mapping.tp_size == 2
    assert mapping.tp_rank == rank
    assert mapping.tp_group == [0, 1]
    assert mapping.tp_group_pg is group
    assert mapping.tp_group_name == group.group_name
    assert dist.get_world_size(mapping.tp_group_pg) == 2
    assert dist.get_rank(mapping.tp_group_pg) == rank
    assert mapping._use_torch_distributed_allreduce
    torch.testing.assert_close(
        embedding.weight, reference.embed_tokens.weight.chunk(2, dim=0)[rank], rtol=0, atol=0
    )
    assert len(candidate.model.model.layers) == MINIMAX_H3_TEXT_ENCODER_LAYER == 50
    for layer, hf_layer in zip(candidate.model.model.layers, reference.layers):
        projections = (
            layer.self_attn.qkv_proj,
            layer.self_attn.o_proj,
            layer.mlp.gate_up_proj,
            layer.mlp.down_proj,
        )
        for projection in projections:
            mapping = projection.mapping
            assert mapping.tp_size == 2
            assert mapping.tp_rank == rank
            assert mapping.tp_group == [0, 1]
            assert mapping.tp_group_pg is group
            assert mapping.tp_group_name == group.group_name
            assert dist.get_world_size(mapping.tp_group_pg) == 2
            assert dist.get_rank(mapping.tp_group_pg) == rank
            assert mapping._use_torch_distributed_allreduce

        expected_qkv = torch.cat(
            [
                projection.weight.chunk(2, dim=0)[rank]
                for projection in (
                    hf_layer.self_attn.q_proj,
                    hf_layer.self_attn.k_proj,
                    hf_layer.self_attn.v_proj,
                )
            ],
            dim=0,
        )
        expected_gate_up = torch.cat(
            [
                projection.weight.chunk(2, dim=0)[rank]
                for projection in (hf_layer.mlp.gate_proj, hf_layer.mlp.up_proj)
            ],
            dim=0,
        )
        expected_weights = (
            expected_qkv,
            hf_layer.self_attn.o_proj.weight.chunk(2, dim=1)[rank],
            expected_gate_up,
            hf_layer.mlp.down_proj.weight.chunk(2, dim=1)[rank],
        )
        for projection, expected in zip(projections, expected_weights):
            torch.testing.assert_close(projection.weight, expected, rtol=0, atol=0)


def _worker(rank: int, port: int, world_size: int) -> None:
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=180),
    )
    try:
        # Every rank creates the subgroup; only ranks 0 and 1 encode. Idle ranks
        # never enter TP collectives, exposing accidental world-sized reductions.
        group = (
            dist.group.WORLD
            if world_size == 2
            else dist.new_group(ranks=[0, 1], timeout=timedelta(seconds=180))
        )
        if rank < 2:
            device = torch.device("cuda", rank)
            torch.manual_seed(19807)
            config = Qwen3VLTextConfig(
                vocab_size=128,
                hidden_size=256,
                intermediate_size=512,
                num_hidden_layers=51,
                num_attention_heads=4,
                num_key_value_heads=2,
                head_dim=64,
                max_position_embeddings=128,
                attention_dropout=0.0,
                rope_scaling={
                    "rope_type": "default",
                    "mrope_section": [16, 8, 8],
                    "mrope_interleaved": True,
                },
            )
            config._attn_implementation = "eager"
            reference = Qwen3VLTextModel(config).to(device=device, dtype=torch.bfloat16).eval()
            mapping = _MiniMaxH3TextEncoderMapping(
                world_size=2, rank=rank, tp_size=2, gpus_per_node=2, tp_group_pg=group
            )
            candidate = MiniMaxH3TensorParallelTextEncoder(config, mapping, device)
            # Use the released conditional-generation checkpoint's language-model
            # prefix and the real loader, including the unused final decoder layer.
            candidate.load_weights(
                {
                    f"model.language_model.{name}": value
                    for name, value in reference.state_dict().items()
                }
            )
            _assert_projection_shards(candidate, reference, group, rank)
            with torch.inference_mode():
                for seq_len in (1, 7, 19):
                    input_ids = ((torch.arange(seq_len, device=device) * 17 + 3) % 128)[None]
                    position_ids = torch.arange(seq_len, device=device).view(1, 1, -1)
                    outputs = reference(
                        input_ids=input_ids,
                        position_ids=position_ids.expand(3, 1, -1),
                        use_cache=False,
                        output_hidden_states=True,
                        return_dict=True,
                    )
                    # With 51 HF layers, hidden_states[50] is the unnormalized
                    # output of the first 50, not the final RMSNorm output.
                    expected = outputs.hidden_states[MINIMAX_H3_TEXT_ENCODER_LAYER]
                    first_layer_outputs: list[torch.Tensor] = []
                    handle = candidate.model.model.layers[0].register_forward_hook(
                        lambda module, args, output: first_layer_outputs.append(
                            output[0] + output[1]
                        )
                    )
                    try:
                        actual = candidate.encode(input_ids)
                    finally:
                        handle.remove()
                    torch.testing.assert_close(
                        first_layer_outputs[0].unsqueeze(0),
                        outputs.hidden_states[1],
                        rtol=2e-2,
                        atol=5e-3,
                    )
                    assert actual.shape == (1, seq_len, config.hidden_size)
                    assert actual.dtype == torch.bfloat16
                    assert torch.isfinite(actual).all()
                    # BF16 GEMM and fused residual/RMSNorm rounding accumulates
                    # across 50 layers, even without TP. Keep a tighter first-layer
                    # check above and bound both outliers and aggregate error here.
                    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=8e-2)
                    relative_error = (
                        actual.float() - expected.float()
                    ).norm() / expected.float().norm()
                    assert relative_error < 3e-2, relative_error.item()
        dist.barrier()
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("world_size", [2, 4], ids=["tp2_world2", "tp2_world4"])
def test_minimax_h3_text_encoder_tp(monkeypatch: pytest.MonkeyPatch, world_size: int) -> None:
    from ._visual_gen_dist_utils import spawn_with_retry

    if torch.cuda.device_count() < world_size:
        pytest.skip(f"Requires {world_size} CUDA GPUs")
    monkeypatch.setenv("TLLM_DISABLE_MPI", "1")
    spawn_with_retry(
        lambda port: mp.spawn(_worker, args=(port, world_size), nprocs=world_size, join=True)
    )
