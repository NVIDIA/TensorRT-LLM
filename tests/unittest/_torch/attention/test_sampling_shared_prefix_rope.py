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
"""Rotary embedding of the n-sample shared-prefix library.

The library rotates the Q and K heads of the packed QKV itself when the model
fuses RoPE into the attention op (Qwen2, Llama, ...); these tests check the
rotation against an fp32 reference, at page boundaries, under CUDA graph
replay, the cases the library declines, and end to end on a small Qwen2.
"""

import os
from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.attention.backends.fmha.sampling_shared_prefix import (
    LIB_NAME,
    SamplingSharedPrefixFmha,
    _fused_rope_supported,
)
from tensorrt_llm._torch.attention.backends.interface import RopeParams
from tensorrt_llm.functional import PositionEmbeddingType, RotaryScalingType

needs_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

TPB = 32


def _rope_params(dim, theta=10000.0, scale_type=RotaryScalingType.none, scale=1.0, positions=4096):
    return RopeParams(
        dim=dim, theta=theta, scale_type=scale_type, scale=scale, max_positions=positions
    )


def _reference_rope(x, positions, rotary_dim, theta, scale):
    """HF-style NeoX rotation in fp32. ``x``: [tokens, heads, head_dim]."""
    half = rotary_dim // 2
    inv_freq = 1.0 / (
        theta ** (torch.arange(0, rotary_dim, 2, device=x.device, dtype=torch.float64) / rotary_dim)
    )
    angle = positions.double()[:, None] * inv_freq[None, :] / scale
    cos = torch.cos(angle).float()[:, None, :]
    sin = torch.sin(angle).float()[:, None, :]
    xf = x.float()
    out = xf.clone()
    x1, x2 = xf[..., :half], xf[..., half:rotary_dim]
    out[..., :half] = x1 * cos - x2 * sin
    out[..., half:rotary_dim] = x2 * cos + x1 * sin
    return out


def _make_qkv(tokens, num_heads, num_kv_heads, head_dim, seed, row_pad=0):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    width = (num_heads + 2 * num_kv_heads) * head_dim
    buf = torch.randn(tokens, width + row_pad, generator=gen, device="cuda").bfloat16()
    return buf[:, :width]  # a strided view, like a slice of a larger activation


def _rope_case_kv_lens(q_len):
    # Device KV lengths around page boundaries, including the first token of a
    # page (kv_len - q_len + j on a multiple of the page size) and position 0.
    return [q_len, q_len + 1, TPB, TPB + 1, 2 * TPB - 1, 2 * TPB, 2 * TPB + q_len, 1000, 3999]


@needs_gpu
@pytest.mark.parametrize(
    "num_heads, num_kv_heads, head_dim, rotary_dim, q_len, scale_type, scale",
    [
        (20, 20, 128, 128, 3, RotaryScalingType.none, 1.0),
        (14, 2, 64, 64, 1, RotaryScalingType.none, 1.0),
        (14, 2, 64, 64, 3, RotaryScalingType.linear, 4.0),
    ],
)
def test_rope_qkv_inplace_matches_fp32_reference(
    num_heads, num_kv_heads, head_dim, rotary_dim, q_len, scale_type, scale
):
    from tensorrt_llm._torch.attention.backends.fmha.sampling_shared_prefix_kernels import (
        rope_qkv_inplace,
    )

    theta = 1000000.0
    _, table = _rope_params(rotary_dim, theta, scale_type, scale).create_rope_const_params()
    kv_lens_list = _rope_case_kv_lens(q_len)
    rows = len(kv_lens_list)
    kv_lens = torch.tensor(kv_lens_list, dtype=torch.int32, device="cuda")
    qkv = _make_qkv(rows * q_len, num_heads, num_kv_heads, head_dim, seed=3, row_pad=8)
    original = qkv.clone()
    rope_qkv_inplace(
        qkv,
        table,
        kv_lens,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        rotary_dim=rotary_dim,
        q_len=q_len,
    )
    positions = (kv_lens.long()[:, None] - q_len + torch.arange(q_len, device="cuda")).reshape(-1)
    q_size, kv_size = num_heads * head_dim, num_kv_heads * head_dim
    tokens = rows * q_len
    for start, heads in ((0, num_heads), (q_size, num_kv_heads)):
        before = original[:, start : start + heads * head_dim].view(tokens, heads, head_dim)
        after = qkv[:, start : start + heads * head_dim].view(tokens, heads, head_dim)
        expected = _reference_rope(before, positions, rotary_dim, theta, scale)
        # One bf16 rounding of an fp32 rotation.
        torch.testing.assert_close(after.float(), expected, atol=2e-2, rtol=1e-2)
        # Dimensions past the rotary dim are not touched.
        assert torch.equal(after[..., rotary_dim:], before[..., rotary_dim:])
    # V is not touched.
    assert torch.equal(qkv[:, q_size + kv_size :], original[:, q_size + kv_size :])


@needs_gpu
def test_rope_position_is_not_shifted_at_page_boundaries():
    """The row at kv_len = k * page_size + 1 holds the first token of page k:
    its position is k * page_size, and the previous row's is one less."""
    from tensorrt_llm._torch.attention.backends.fmha.sampling_shared_prefix_kernels import (
        rope_qkv_inplace,
    )

    head_dim, heads, kv_heads = 64, 4, 2
    _, table = _rope_params(head_dim).create_rope_const_params()
    kv_lens = torch.tensor([TPB, TPB + 1, TPB + 2], dtype=torch.int32, device="cuda")
    # Identical inputs: outputs differ only through the position.
    qkv = _make_qkv(1, heads, kv_heads, head_dim, seed=5).repeat(3, 1).contiguous()
    original = qkv.clone()
    rope_qkv_inplace(
        qkv,
        table,
        kv_lens,
        num_heads=heads,
        num_kv_heads=kv_heads,
        head_dim=head_dim,
        rotary_dim=head_dim,
        q_len=1,
    )
    for row, position in enumerate((TPB - 1, TPB, TPB + 1)):
        expected = _reference_rope(
            original[row : row + 1, : heads * head_dim].view(1, heads, head_dim),
            torch.tensor([position], device="cuda"),
            head_dim,
            10000.0,
            1.0,
        )
        torch.testing.assert_close(
            qkv[row, : heads * head_dim].view(1, heads, head_dim).float(),
            expected,
            atol=2e-2,
            rtol=1e-2,
        )


@needs_gpu
@pytest.mark.parametrize("q_len", [3])
def test_rope_cuda_graph_replay_matches_eager(q_len):
    """Positions come from the device KV lengths: one captured launch follows
    lengths that change between replays, bit for bit like an eager call."""
    from tensorrt_llm._torch.attention.backends.fmha.sampling_shared_prefix_kernels import (
        rope_qkv_inplace,
    )

    heads, kv_heads, head_dim = 14, 2, 64
    _, table = _rope_params(head_dim, 1000000.0).create_rope_const_params()
    rows = 6
    kv_lens = torch.full((rows,), 100, dtype=torch.int32, device="cuda")
    static = _make_qkv(rows * q_len, heads, kv_heads, head_dim, seed=7).contiguous()
    kwargs = dict(
        num_heads=heads, num_kv_heads=kv_heads, head_dim=head_dim, rotary_dim=head_dim, q_len=q_len
    )
    graph = torch.cuda.CUDAGraph()
    # Warm up, then capture on a copy so the captured op rotates ``static``.
    scratch = static.clone()
    rope_qkv_inplace(scratch, table, kv_lens, **kwargs)
    with torch.cuda.graph(graph):
        rope_qkv_inplace(static, table, kv_lens, **kwargs)
    for step, lengths in enumerate(([TPB + 1] * rows, [5, 31, 32, 33, 64, 2047], [q_len] * rows)):
        kv_lens.copy_(torch.tensor(lengths, dtype=torch.int32))
        fresh = _make_qkv(rows * q_len, heads, kv_heads, head_dim, seed=20 + step).contiguous()
        eager = fresh.clone()
        rope_qkv_inplace(eager, table, kv_lens, **kwargs)
        static.copy_(fresh)
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(static, eager), f"step {step}: replay differs from eager"


def _fake_attn(
    position_type=PositionEmbeddingType.rope_gpt_neox, rope=None, head_dim=64, table=True, **extra
):
    rope = rope if rope is not None else _rope_params(head_dim)
    attrs = dict(
        is_mla_enable=False,
        sparse_params=None,
        predicted_tokens_per_seq=1,
        attention_chunk_size=None,
        head_dim=head_dim,
        num_heads=14,
        num_kv_heads=2,
        quant_mode=0,
        position_embedding_type=int(position_type),
        rope_params=rope,
        rotary_cos_sin=(
            torch.zeros(rope.max_positions * rope.dim, dtype=torch.float32, device="cuda")
            if table
            else None
        ),
    )
    attrs.update(extra)
    return SimpleNamespace(**attrs)


@needs_gpu
def test_library_accepts_supported_rope_and_declines_the_rest():
    pytest.importorskip("flashinfer")
    ok = RotaryScalingType.none
    assert SamplingSharedPrefixFmha._is_available(_fake_attn(rope=_rope_params(64, scale_type=ok)))
    assert SamplingSharedPrefixFmha._is_available(
        _fake_attn(rope=_rope_params(64, scale_type=RotaryScalingType.linear, scale=2.0))
    )
    # Partial rotary dim.
    assert SamplingSharedPrefixFmha._is_available(_fake_attn(head_dim=128, rope=_rope_params(64)))
    # A module that rotates by itself hands no pos_embd_params.
    assert SamplingSharedPrefixFmha._is_available(
        _fake_attn(position_type=PositionEmbeddingType.learned_absolute)
    )

    for scale_type in (
        RotaryScalingType.yarn,
        RotaryScalingType.llama3,
        RotaryScalingType.dynamic,
        RotaryScalingType.longrope,
        RotaryScalingType.mrope,
    ):
        attn = _fake_attn(rope=_rope_params(64, scale_type=scale_type, scale=2.0))
        assert not _fused_rope_supported(attn), scale_type
        assert not SamplingSharedPrefixFmha._is_available(attn), scale_type
    for position_type in (
        PositionEmbeddingType.rope_gptj,
        PositionEmbeddingType.long_rope,
        PositionEmbeddingType.mrope,
        PositionEmbeddingType.yarn,
    ):
        assert not SamplingSharedPrefixFmha._is_available(_fake_attn(position_type=position_type))
    # No table, rotary dim past the head or not a power of two.
    assert not SamplingSharedPrefixFmha._is_available(_fake_attn(table=False))
    assert not SamplingSharedPrefixFmha._is_available(
        _fake_attn(head_dim=64, rope=_rope_params(128))
    )
    assert not SamplingSharedPrefixFmha._is_available(
        _fake_attn(head_dim=128, rope=_rope_params(96))
    )


# --------------------------------------------------------------------------- #
# End to end on a small Qwen2 (fused RoPE, the default)
# --------------------------------------------------------------------------- #


def _qwen2_small_path():
    candidates = []
    if os.environ.get("TLLM_ROPE_TEST_MODEL"):
        candidates.append(os.environ["TLLM_ROPE_TEST_MODEL"])
    try:
        from utils.llm_data import llm_models_root

        root = llm_models_root(check=False)
        if root is not None:
            candidates += [str(root / "Qwen2.5-0.5B"), str(root / "Qwen" / "Qwen2.5-0.5B")]
    except Exception:
        pass
    return next((c for c in candidates if os.path.isdir(c)), None)


def _run_n_sampling(model_path, prompts, n, max_tokens, env, cuda_graph):
    from tensorrt_llm import LLM, SamplingParams
    from tensorrt_llm.llmapi import CudaGraphConfig, KvCacheConfig

    previous = {key: os.environ.get(key) for key in env}
    os.environ.update(env)
    try:
        llm = LLM(
            model=model_path,
            max_batch_size=len(prompts) * n,
            max_num_tokens=8192,
            max_seq_len=max(len(p) for p in prompts) + max_tokens + 1,
            kv_cache_config=KvCacheConfig(
                enable_block_reuse=True, use_kv_cache_manager_v2=True, free_gpu_memory_fraction=0.3
            ),
            cuda_graph_config=(
                CudaGraphConfig(max_batch_size=len(prompts) * n) if cuda_graph else None
            ),
            skip_tokenizer_init=True,
        )
        with llm:
            sampling = SamplingParams(
                n=n, top_k=1, max_tokens=max_tokens, ignore_eos=True, end_id=-1, logprobs=2
            )
            outputs = llm.generate(prompts, sampling_params=sampling)
        return [
            [
                (
                    tuple(c.token_ids),
                    [sorted((v.logprob for v in d.values()), reverse=True) for d in c.logprobs],
                )
                for c in out.outputs
            ]
            for out in outputs
        ]
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


@needs_gpu
@pytest.mark.skipif(_qwen2_small_path() is None, reason="Qwen2.5-0.5B checkpoint not available")
@pytest.mark.threadleak(enabled=False)
@pytest.mark.parametrize("cuda_graph", [True], ids=["cuda_graph"])
def test_qwen2_fused_rope_n_sampling_diverges_only_at_near_ties_e2e(cuda_graph):
    """Qwen2 fuses RoPE into the attention op by default. With the library on,
    greedy n-sample children may leave the regular kernel's sequence only where
    its own top-2 logprob gap is within rounding (0.125 nats), at and around
    page boundaries."""
    model_path = _qwen2_small_path()
    vocab = 150000
    prompts = [
        [1024 + ((17 * i + 13 * p) % (vocab - 1024)) for p in range(n)]
        for i, n in enumerate((511, 512, 513))
    ]
    n, max_tokens = 6, 40
    base_env = {"TLLM_WORKER_USE_SINGLE_PROCESS": "1", "TLLM_ALLOW_N_GREEDY_DECODING": "1"}
    reference = _run_n_sampling(model_path, prompts, n, max_tokens, base_env, cuda_graph)
    candidate = _run_n_sampling(
        model_path,
        prompts,
        n,
        max_tokens,
        base_env
        | {
            "TLLM_FMHA_LIBS": f"+{LIB_NAME}",
            "TLLM_SAMPLING_SHARED_PREFIX_MIN_SHARED_TOKENS": "0",
            "TLLM_SAMPLING_SHARED_PREFIX_MIN_GROUP": "1",
        },
        cuda_graph,
    )
    for ref_samples, cand_samples in zip(reference, candidate):
        assert len(cand_samples) == n
        for (ref_tokens, ref_lp), (cand_tokens, _) in zip(ref_samples, cand_samples):
            assert len(cand_tokens) == len(ref_tokens)
            if cand_tokens == ref_tokens:
                continue
            pos = next(i for i, (a, b) in enumerate(zip(ref_tokens, cand_tokens)) if a != b)
            gap = ref_lp[pos][0] - ref_lp[pos][1] if len(ref_lp[pos]) > 1 else 0.0
            assert gap <= 0.125, (pos, gap)
