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

from types import SimpleNamespace

import pytest
import torch
from fmha_test_utils import FakeAttention

from tensorrt_llm._torch.attention.backends.fmha.flashinfer_trtllm_gen import (
    FlashInferTrtllmGenFmha,
)
from tensorrt_llm._torch.attention.backends.fmha.interface import FmhaPhase
from tensorrt_llm._torch.attention.backends.interface import (
    AttentionForwardArgs,
    AttentionInputType,
)
from tensorrt_llm.bindings import DataType
from tensorrt_llm.quantization.mode import QuantMode


def test_flashinfer_fp8_mode_remains_implementation_local() -> None:
    attn = FakeAttention()
    attn.quant_mode = int(QuantMode.from_description(use_fp8_kv_cache=True))
    fmha = FlashInferTrtllmGenFmha(attn)
    output = torch.empty(1, dtype=torch.bfloat16)

    assert fmha._use_fp8_context_fmha(output, AttentionInputType.context_only)
    assert fmha._use_fp8_context_fmha(output, AttentionInputType.mixed)
    assert not fmha._use_fp8_context_fmha(output, AttentionInputType.generation_only)


@pytest.mark.parametrize(
    "num_contexts,num_generations,dtype,expect_fallback",
    [
        (1, 0, torch.bfloat16, True),
        (4, 0, torch.bfloat16, True),
        (5, 0, torch.bfloat16, False),
        (1, 1, torch.bfloat16, False),
        (2, 9, torch.bfloat16, False),
        (2, 19, torch.bfloat16, False),
        (2, 38, torch.bfloat16, False),
        (4, 1, torch.bfloat16, False),
        (0, 1, torch.bfloat16, False),
        (1, 0, torch.float16, False),
    ],
)
def test_small_context_fallback_preserves_mixed_batch_generation(
    num_contexts: int,
    num_generations: int,
    dtype: torch.dtype,
    expect_fallback: bool,
) -> None:
    attn = FakeAttention()
    attn.head_dim = 256
    attn.sparse_params = None
    attn.position_embedding_type = 0
    fmha = FlashInferTrtllmGenFmha(attn)
    metadata = SimpleNamespace(
        num_contexts=num_contexts,
        num_generations=num_generations,
        helix_position_offsets=None,
        num_sparse_topk=0,
        use_spec_decoding=False,
        kv_cache_block_offsets=object(),
        kv_cache_manager=None,
        tokens_per_block=32,
        is_cross=False,
        beam_width=1,
    )
    if num_contexts == 0:
        input_type = AttentionInputType.generation_only
        phases = (None, FmhaPhase.GENERATION)
    elif num_generations == 0:
        input_type = AttentionInputType.context_only
        phases = (None, FmhaPhase.CONTEXT)
    else:
        input_type = AttentionInputType.mixed
        phases = (None, FmhaPhase.CONTEXT, FmhaPhase.GENERATION)
    num_tokens = num_contexts + num_generations
    q = torch.empty((num_tokens, 3 * attn.head_dim), dtype=dtype)
    forward_args = AttentionForwardArgs(
        output=torch.empty((num_tokens, attn.head_dim), dtype=dtype),
        attention_input_type=input_type,
        is_fused_qkv=True,
    )

    for phase in phases:
        supported, reason = fmha._is_supported_with_reason(
            q, None, None, attn, metadata, forward_args, phase=phase
        )
        assert supported is not expect_fallback, reason
        if expect_fallback:
            assert "small-batch BF16 context attention" in reason


@pytest.mark.parametrize("kv_cache_dtype", [DataType.FP8, DataType.NVFP4])
@pytest.mark.parametrize(
    "dtype,sm_version",
    [(torch.bfloat16, 100), (torch.float16, 103)],
    ids=["small_bf16_batch", "sm103_fp16"],
)
def test_flashinfer_quantized_kv_context_avoids_fp16_bf16_fallback(
    monkeypatch: pytest.MonkeyPatch,
    kv_cache_dtype: DataType,
    dtype: torch.dtype,
    sm_version: int,
) -> None:
    monkeypatch.setattr(
        "tensorrt_llm._torch.attention.backends.fmha.flashinfer_trtllm_gen.get_sm_version",
        lambda: sm_version,
    )
    fmha = object.__new__(FlashInferTrtllmGenFmha)
    fmha.kv_factor = 2
    monkeypatch.setattr(fmha, "_get_total_num_blocks", lambda _: 0)
    attn = SimpleNamespace(
        is_mla_enable=False,
        sparse_params=None,
        position_embedding_type=0,
        head_dim=256,
        num_heads=32,
        num_kv_heads=2,
    )
    q_hidden_size = attn.num_heads * attn.head_dim
    qkv_hidden_size = q_hidden_size + 2 * attn.num_kv_heads * attn.head_dim
    q = torch.empty((1, qkv_hidden_size), dtype=dtype)
    metadata = SimpleNamespace(
        num_contexts=1,
        helix_position_offsets=None,
        num_sparse_topk=0,
        use_spec_decoding=False,
        is_spec_dec_tree=False,
        kv_cache_block_offsets=object(),
        kv_cache_manager=SimpleNamespace(dtype=kv_cache_dtype),
        is_cross=False,
        is_spec_decoding_enabled=False,
        tokens_per_block=64,
        beam_width=1,
    )
    forward_args = AttentionForwardArgs(
        output=torch.empty((1, q_hidden_size), dtype=dtype),
        attention_input_type=AttentionInputType.context_only,
        is_fused_qkv=True,
        update_kv_cache=True,
    )

    for phase in (None, FmhaPhase.CONTEXT):
        supported, reason = fmha._is_supported_with_reason(
            q,
            None,
            None,
            attn,
            metadata,
            forward_args,
            phase=phase,
        )

        assert supported, reason
        assert reason == ""


@pytest.mark.parametrize(
    "env,batch,q_len,window_left,expected",
    [
        (None, 40, 8, -1, None),
        ("0", 40, 8, -1, None),
        ("8", 8, 8, -1, None),
        ("8", 40, 8, 2047, None),
        ("8", 40, None, -1, None),
        ("8", 9, 8, -1, [(0, 4), (4, 9)]),
        ("8", 40, 8, -1, [(0, 8), (8, 16), (16, 24), (24, 32), (32, 40)]),
        ("4", 10, 8, -1, [(0, 3), (3, 6), (6, 10)]),
    ],
)
def test_decode_launch_groups(
    monkeypatch: pytest.MonkeyPatch, env, batch, q_len, window_left, expected
) -> None:
    from tensorrt_llm._torch.attention.backends.fmha import flashinfer_trtllm_gen as fi

    if env is None:
        monkeypatch.delenv("TRTLLM_GEN_DECODE_SEQS_PER_LAUNCH", raising=False)
    else:
        monkeypatch.setenv("TRTLLM_GEN_DECODE_SEQS_PER_LAUNCH", env)
    fi._decode_seqs_per_launch.cache_clear()
    rows = batch * (q_len or 1)
    try:
        groups = fi._decode_launch_groups(batch, q_len, rows, rows, window_left)
    finally:
        fi._decode_seqs_per_launch.cache_clear()
    assert groups == expected
    if groups is not None:
        assert groups[0][0] == 0 and groups[-1][1] == batch
        assert all(a < b for a, b in groups)
        assert all(b == c for (_, b), (c, _) in zip(groups, groups[1:]))


def test_decode_launch_groups_requires_exact_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    from tensorrt_llm._torch.attention.backends.fmha import flashinfer_trtllm_gen as fi

    monkeypatch.setenv("TRTLLM_GEN_DECODE_SEQS_PER_LAUNCH", "4")
    fi._decode_seqs_per_launch.cache_clear()
    try:
        assert fi._decode_launch_groups(16, 8, 16 * 8 - 1, 16 * 8, -1) is None
        assert fi._decode_launch_groups(16, 8, 16 * 8, 16 * 8 + 8, -1) is None
    finally:
        fi._decode_seqs_per_launch.cache_clear()


_FMHA_WORKSPACE_BYTES = 32 << 20


@pytest.mark.parametrize(
    "env,batch,kv_heads,head_dim,window_left,expected",
    [
        # Automatic: ~36 x sqrt(kv_heads) CTAs per sequence in whole waves of 152 SMs.
        (None, 1, 16, 64, -1, 1),
        (None, 6, 16, 64, -1, 6),
        (None, 32, 16, 64, -1, 8),
        (None, 8, 2, 64, -1, 3),
        (None, 10, 2, 64, -1, 4),
        ("auto", 16, 1, 64, -1, 4),
        # A sliding window keeps the kernel's own split.
        (None, 8, 16, 64, 2047, 1),
        ("8", 8, 16, 64, 2047, 1),
        # Forced, and capped by what the workspace holds for the head dim.
        ("1", 32, 16, 64, -1, 1),
        ("4", 1, 16, 64, -1, 4),
        ("16", 32, 16, 64, -1, 8),
        (None, 32, 16, 128, -1, 6),
        (None, 32, 16, 256, -1, 3),
    ],
)
def test_decode_sm_scale(
    monkeypatch: pytest.MonkeyPatch, env, batch, kv_heads, head_dim, window_left, expected
) -> None:
    from tensorrt_llm._torch.attention.backends.fmha import flashinfer_trtllm_gen as fi

    if env is None:
        monkeypatch.delenv("TRTLLM_GEN_DECODE_SM_SCALE", raising=False)
    else:
        monkeypatch.setenv("TRTLLM_GEN_DECODE_SM_SCALE", env)
    fi._decode_sm_scale_override.cache_clear()
    try:
        scale = fi._decode_sm_scale(
            batch, kv_heads, head_dim, 152, _FMHA_WORKSPACE_BYTES, window_left
        )
    finally:
        fi._decode_sm_scale_override.cache_clear()
    assert scale == expected


def test_decode_sm_scale_rejects_bad_env(monkeypatch: pytest.MonkeyPatch) -> None:
    from tensorrt_llm._torch.attention.backends.fmha import flashinfer_trtllm_gen as fi

    monkeypatch.setenv("TRTLLM_GEN_DECODE_SM_SCALE", "eight")
    fi._decode_sm_scale_override.cache_clear()
    try:
        with pytest.raises(ValueError, match="TRTLLM_GEN_DECODE_SM_SCALE"):
            fi._decode_sm_scale_override()
    finally:
        fi._decode_sm_scale_override.cache_clear()


def test_flashinfer_decode_reads_sm_count() -> None:
    """The split works only while flashinfer's decode reads its module's get_device_sm_count."""
    import inspect

    flashinfer_decode = pytest.importorskip("flashinfer.decode")
    fn = inspect.unwrap(flashinfer_decode.trtllm_batch_decode_with_kv_cache)
    assert fn.__globals__ is vars(flashinfer_decode)
    assert "get_device_sm_count" in fn.__code__.co_names


def test_trtllm_gen_decode_reports_scaled_sm_count(monkeypatch: pytest.MonkeyPatch) -> None:
    flashinfer_decode = pytest.importorskip("flashinfer.decode")
    from tensorrt_llm._torch.attention.backends.fmha import flashinfer_trtllm_gen as fi

    def real(device):
        return 152

    seen = []

    def decode(**kwargs):
        seen.append(flashinfer_decode.get_device_sm_count(None))
        if kwargs.get("fail"):
            raise RuntimeError("launch failed")

    monkeypatch.setattr(flashinfer_decode, "get_device_sm_count", real)
    monkeypatch.setattr(flashinfer_decode, "trtllm_batch_decode_with_kv_cache", decode)
    fi._trtllm_gen_decode(6)
    fi._trtllm_gen_decode(1)
    with pytest.raises(RuntimeError):
        fi._trtllm_gen_decode(4, fail=True)
    assert seen == [6 * 152, 152, 4 * 152]
    assert flashinfer_decode.get_device_sm_count is real

    # A caller that replaced flashinfer for the call keeps its own SM count.
    stand_in = SimpleNamespace(trtllm_batch_decode_with_kv_cache=decode)
    monkeypatch.setattr(fi, "flashinfer", SimpleNamespace(decode=stand_in))
    fi._trtllm_gen_decode(6)
    assert seen[-1] == 152
    assert flashinfer_decode.get_device_sm_count is real
