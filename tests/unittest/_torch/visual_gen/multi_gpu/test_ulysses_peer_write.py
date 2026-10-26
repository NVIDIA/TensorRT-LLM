# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Four-rank parity for compiled, reusable Ulysses peer-write buffers."""

from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from tensorrt_llm._torch.distributed import all_to_all_4d
from tensorrt_llm._torch.visual_gen.attention.backends.flash_attn4 import FlashAttn4Attention
from tensorrt_llm._torch.visual_gen.attention.backends.parallel import UlyssesAttention
from tensorrt_llm._torch.visual_gen.attention.backends.ulysses_peer_write import (
    UlyssesPeerWrite,
    supports_peer_write,
)


def _worker(rank: int, port: int) -> None:
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=4,
        timeout=timedelta(seconds=240),
    )
    try:
        # Use a distinct process group for rendezvous and exchange.
        group = dist.new_group([0, 1, 2, 3])
        device = torch.device("cuda", rank)
        assert supports_peer_write(group, device)
        torch.set_grad_enabled(False)
        torch.manual_seed(42 + rank)
        for n in (17, 33):
            with torch.inference_mode():
                state = UlyssesPeerWrite(group, n, 8, 128, device)
            assert not state.qkv.is_inference()
            backend = FlashAttn4Attention(num_heads=2, head_dim=128, dtype=torch.bfloat16)
            attention = UlyssesAttention(backend, group)
            candidate = UlyssesAttention(backend, group)
            candidate.peer_write = state
            mask = torch.ones(1, n * 4, device=device, dtype=torch.bool)
            mask[:, -3:] = False
            # Different strides exercise fused QKV projection views and contiguous RoPE outputs.
            storage = torch.randn(1, n, 3, 8, 128, device=device, dtype=torch.bfloat16)
            q, k, v = storage[:, :, 0].clone(), storage[:, :, 1].clone(), storage[:, :, 2]
            if n == 33:
                q, k = storage[:, :, 0], storage[:, :, 1]
            compiled_exchange = torch.compile(state.exchange_qkv, fullgraph=True)
            compiled_output = torch.compile(state.exchange_output, fullgraph=True)
            compiled_attention = torch.compile(candidate.forward, fullgraph=False)
            for step in range(8):
                q.add_(0.00390625)
                k.sub_(0.001953125)
                v.add_(0.0078125)
                torch.cuda._sleep(10000 * ((step + rank) % 4))
                reference = [all_to_all_4d(t, 2, 1, process_group=group) for t in (q, k, v)]
                actual = compiled_exchange(q, k, v)
                for a, b in zip(actual, reference):
                    torch.testing.assert_close(a, b, rtol=0, atol=0)
                # The reverse exchange must reproduce the original sequence shard exactly.
                result = compiled_output(actual[0])
                torch.testing.assert_close(result, q, rtol=0, atol=0)
                for key_mask in (None, mask):
                    assert attention._supports_packed_self_attention(
                        q, k, v, {"key_padding_mask": key_mask}
                    )
                    expected = attention.forward(q, k, v, key_padding_mask=key_mask)
                    actual_output = compiled_attention(q, k, v, key_padding_mask=key_mask)
                    # Ensure parity cannot pass by silently taking the packed-QKV fallback.
                    assert actual_output.data_ptr() == state.output.data_ptr()
                    torch.testing.assert_close(actual_output, expected, rtol=0, atol=0)
                # Consume the borrowed result before the next exchange overwrites it.
                assert result.data_ptr() == state.output.data_ptr()
            # Mismatched batch/sequence/dtype must select the existing attention path.
            assert not state.matches(q.repeat(2, 1, 1, 1), k, v)
            assert not state.matches(q[:, :-1], k, v)
            assert not state.matches(q.float(), k, v)
            with torch.cuda.stream(torch.cuda.Stream()):
                with pytest.raises(RuntimeError, match="owning CUDA stream"):
                    state.exchange_output(reference[0])
        _check_h3_integration(rank, device)
        dist.barrier()
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count() < 4, reason="Requires four CUDA GPUs")
def test_ulysses_peer_write(monkeypatch: pytest.MonkeyPatch) -> None:
    from ._visual_gen_dist_utils import spawn_with_retry

    if torch.cuda.get_device_capability() != (10, 0):
        pytest.skip("Peer-write H3 path targets SM100")
    monkeypatch.setenv("TLLM_DISABLE_MPI", "1")
    spawn_with_retry(lambda port: mp.spawn(_worker, args=(port,), nprocs=4, join=True))


def _check_h3_integration(rank: int, device: torch.device) -> None:
    from types import SimpleNamespace

    from tensorrt_llm._torch.visual_gen.config import DiffusionModelConfig
    from tensorrt_llm._torch.visual_gen.mapping import VisualGenMapping
    from tensorrt_llm._torch.visual_gen.models.minimax_h3.transformer_minimax_h3 import (
        MiniMaxH3Transformer3DModel,
    )
    from tensorrt_llm.visual_gen.args import AttentionConfig, CudaGraphConfig

    mapping = VisualGenMapping(world_size=4, rank=rank, ulysses_size=4, parallel_vae_size=4)
    config = DiffusionModelConfig(
        pretrained_config=SimpleNamespace(
            num_attention_heads=4,
            attention_head_dim=128,
            hidden_size=128,
            num_layers=2,
            num_refiner_layers=1,
            ffn_dim=32,
            in_channels=2,
            audio_in_channels=3,
            patch_size=(1, 1, 1),
            text_dim=5,
            freq_dim=8,
            time_embed_hidden_dim=12,
            time_embed_dim=6,
            rope_freq_dim=1,
            rope_theta=10000.0,
            norm_eps=1e-5,
            qk_norm_eps=1e-5,
            final_norm_eps=1e-5,
        ),
        visual_gen_mapping=mapping,
        mapping=mapping.to_llm_mapping(),
        attention=AttentionConfig(backend="FA4"),
        cuda_graph=CudaGraphConfig(enable=False),
    )
    torch.manual_seed(19)
    model = MiniMaxH3Transformer3DModel(config).to(device).eval()
    for name, parameter in model.named_parameters():
        if "norm" in name and name.endswith("weight"):
            parameter.fill_(1)
        else:
            parameter.normal_(0, 0.02)
    for length in (5, 9):
        inputs = dict(
            hidden_states=torch.randn(1, length - 2, 2, device=device),
            audio_hidden_states=torch.randn(1, 1, 3, device=device),
            encoder_hidden_states=torch.randn(1, 1, 5, device=device),
            timestep=torch.tensor([1.0], device=device),
            conditioning_timesteps=torch.tensor([0.0, 0.75], device=device),
            token_tags=torch.tensor([1, 2] + [0] * (length - 2), device=device),
            timestep_indices=torch.tensor([0] + [1] * (length - 1), device=device),
            position_ids=torch.arange(length * 3, device=device).reshape(length, 3),
            video_indices=torch.arange(2, length, device=device),
            audio_indices=torch.tensor([1], device=device),
            text_indices=torch.tensor([0], device=device),
        )
        model._peer_write_eligible = False
        for block in model.transformer_blocks:
            block.attn.attn.peer_write = None
        expected = model(**inputs)
        model._peer_write_eligible = True
        actual = model(**inputs)
        state = model._peer_write_workspace
        assert state is not None and state.shape[1] == (length + 3) // 4
        assert all(block.attn.attn.peer_write is state for block in model.transformer_blocks)
        torch.testing.assert_close(actual.sample, expected.sample, rtol=0, atol=0)
        torch.testing.assert_close(actual.audio_sample, expected.audio_sample, rtol=0, atol=0)
