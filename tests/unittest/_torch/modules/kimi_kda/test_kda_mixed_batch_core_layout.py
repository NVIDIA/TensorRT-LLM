# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A mixed prefill+decode batch must concatenate KDA cores of the same rank.

``_forward_impl`` cats the prefill core with the decode core. Prefill returns
``[tokens, heads, head_dim]`` through ``_store_core``. The FLA decode fallback
used to return the gated norm after ``squeeze(1)``, which stays
``[tokens * heads, head_dim]`` and makes that cat raise. Decode-only tests
never cat, so they miss it.

Needs 1 GPU + fla. Random weights, no checkpoint, and no optimized kernels:
both prefill and decode are forced onto FLA, which is the dispatch that hit
the failure.
"""

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("fla")

from tensorrt_llm._torch.modules.kimi_kda import KimiKDALinearAttention  # noqa: E402

HIDDEN_SIZE = 512
NUM_HEADS = 4
HEAD_DIM = 128
CONV_WIDTH = 4


class _Cfg:
    hidden_size = HIDDEN_SIZE
    rms_norm_eps = 1e-6
    linear_attn_config = {
        "num_heads": NUM_HEADS,
        "head_dim": HEAD_DIM,
        "short_conv_kernel_size": CONV_WIDTH,
        "use_full_rank_gate": True,
        "gate_lower_bound": -5.0,
    }


class _LayerCache:
    def __init__(self, conv: torch.Tensor, temporal: torch.Tensor) -> None:
        self.conv = conv
        self.temporal = temporal
        self.kda_qkg_cache = None
        self.has_kda_replay_caches = False


class _KvCacheManager:
    def __init__(self, layer_cache: _LayerCache) -> None:
        self._layer_cache = layer_cache

    def mamba_layer_cache(self, _layer_idx: int) -> _LayerCache:
        return self._layer_cache


def _make_fla_runtime() -> KimiKDALinearAttention:
    if not torch.cuda.is_available():
        pytest.skip("mixed-batch KDA core layout needs a CUDA device")
    runtime = KimiKDALinearAttention(_Cfg(), layer_idx=0).to("cuda")
    torch.nn.init.normal_(runtime.dt_bias, std=0.1)
    # Prefill's packed convolution reads this buffer. finalize_decode_weights
    # only builds it on the optimized decode path, which this test disables.
    runtime._build_mtp_conv_weights()
    runtime._dispatch.prefill_kernel_path = "fla"
    runtime._dispatch.decode_kernel_path = "fla"
    return runtime


def _mixed_metadata(num_ctx_tokens: int, num_decodes: int) -> SimpleNamespace:
    num_prefills = 1
    batch = num_prefills + num_decodes
    device = "cuda"
    state_indices = torch.arange(batch, device=device, dtype=torch.int32)
    cu_seqlens = torch.tensor([0, num_ctx_tokens], device=device, dtype=torch.long)
    mamba = SimpleNamespace(
        state_indices=state_indices,
        query_start_loc_long=cu_seqlens,
        query_start_loc=cu_seqlens.to(torch.int32),
        has_initial_states=torch.zeros(num_prefills, device=device, dtype=torch.bool),
        use_initial_states=False,
        kda_chunk_indices=torch.tensor([[0, 0]], device=device, dtype=torch.long),
        kda_varlen_is_aligned=num_ctx_tokens % 64 == 0,
        kda_single_sequence_length=num_ctx_tokens,
        _arange_buffer=torch.arange(batch + 1, dtype=torch.int32, device=device),
    )
    return SimpleNamespace(
        mamba_metadata=mamba,
        num_contexts=num_prefills,
        num_ctx_tokens=num_ctx_tokens,
        num_tokens=num_ctx_tokens + num_decodes,
        seq_lens=torch.ones(batch, dtype=torch.int32, device=device),
        kv_cache_manager=_KvCacheManager(
            _LayerCache(
                conv=torch.randn(
                    batch,
                    3 * NUM_HEADS * HEAD_DIM,
                    CONV_WIDTH - 1,
                    dtype=torch.bfloat16,
                    device=device,
                )
                * 0.02,
                temporal=torch.randn(
                    batch,
                    NUM_HEADS,
                    HEAD_DIM,
                    HEAD_DIM,
                    dtype=torch.float32,
                    device=device,
                )
                * 0.01,
            )
        ),
    )


@torch.no_grad()
def test_fla_mixed_batch_cores_share_prefill_layout():
    torch.manual_seed(0)
    runtime = _make_fla_runtime()
    num_ctx_tokens, num_decodes = 4, 2
    metadata = _mixed_metadata(num_ctx_tokens, num_decodes)
    hidden = (
        torch.randn(
            num_ctx_tokens + num_decodes,
            HIDDEN_SIZE,
            dtype=torch.bfloat16,
            device="cuda",
        )
        * 0.05
    )

    core = runtime._forward_impl(hidden, metadata)

    assert core.shape == (
        num_ctx_tokens + num_decodes,
        NUM_HEADS,
        HEAD_DIM,
    )
    assert torch.isfinite(core).all()
