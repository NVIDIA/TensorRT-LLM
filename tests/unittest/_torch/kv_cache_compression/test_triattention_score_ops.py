# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CUDA score ops vs an independent PyTorch oracle.

`_launch_tri_score_perhead` calls the compiled `trtllm` fold + paged-score
ops unconditionally (no Triton fallback; the original Triton score kernel has
been deleted). These tests score the same paged pools through a pure-PyTorch
oracle and compare the two implementations across the geometry matrix the
launcher must cover: both CUDA load paths (vectorized 8-frequency chunks and
strided scalar), both aggregations, every supported pool dtype (bf16/fp16/
fp32, plus functional-only fp8_e4m3fn/int8 with per-layer dequantization
scales), and GQA group sizes with and without a dedicated template
instantiation.
"""

import pytest
import torch
from conftest import encode_block_offsets as _encode_block_offsets
from conftest import torch_tri_score_oracle as _torch_tri_score_oracle

from tensorrt_llm._torch.kv_cache_compression.triattention.triattention_kernels import (
    _FixedScoreGroup,
)


def _require_score_ops() -> None:
    """The compiled score ops are a hard prerequisite for these tests."""
    assert hasattr(torch.ops.trtllm, "tri_attention_fold_score_coefficients"), (
        "TriAttention fold op is not loaded"
    )
    assert hasattr(torch.ops.trtllm, "tri_attention_paged_score"), (
        "TriAttention paged score op is not loaded"
    )


def _oracle_reference(
    group: _FixedScoreGroup,
    pools: list,
    oracle_inputs: dict,
    request_count: int,
    seq_lens: list,
    round_starts: torch.Tensor,
    prompt_len: int,
    aggregation: str,
    sentinel: float,
) -> torch.Tensor:
    """Sentinel-filled oracle scores in the ops' [request, layer, head, token] layout.

    The oracle scores every cached token in [0, seq_len); the score ops write
    only the decode region [prompt_len, seq_len) at column origin 0. Slice the
    prompt columns off each oracle row and leave every column the ops must not
    touch at the sentinel, so the comparison covers the write MASK as well as
    the values.
    """
    num_layers = group.num_layers
    oracle = _torch_tri_score_oracle(
        pools,
        oracle_inputs["page_ids"][:request_count],
        seq_lens,
        round_starts[:request_count].tolist(),
        oracle_inputs["q_real"],
        oracle_inputs["q_imag"],
        oracle_inputs["mlr_coef"],
        oracle_inputs["freq_scale_sq"],
        oracle_inputs["omega"],
        oracle_inputs["offsets"],
        list(range(num_layers)),
        aggregation,
    )
    reference = torch.full_like(group.output, sentinel)
    for request in range(request_count):
        width = seq_lens[request] - prompt_len
        for layer in range(num_layers):
            reference[request, layer, :, :width] = oracle[request * num_layers + layer][
                :, prompt_len:
            ]
    return reference


def _build_case(
    *,
    request_count: int,
    max_requests: int,
    num_layers: int,
    page_count: int,
    tokens_per_block: int,
    head_dim: int,
    num_q_heads: int,
    num_kv_heads: int,
    dtype: torch.dtype,
    offsets: list,
    prompt_len: int,
    seed: int,
    pools: list | None = None,
    kv_scales: torch.Tensor | None = None,
):
    device = torch.device("cuda", torch.cuda.current_device())
    torch.manual_seed(seed)
    num_freqs = head_dim // 2
    # Callers may inject prebuilt pools: the quantized tests build a quantized
    # pool (plus its dequantized fp32 twin for the oracle reference leg) from
    # ONE set of randoms and pass the quantized pool in here.
    if pools is None:
        pools = [
            torch.randn(
                max_requests * page_count,
                2,
                num_kv_heads,
                tokens_per_block,
                head_dim,
                device=device,
            ).to(dtype)
            for _ in range(num_layers)
        ]
    page_ids = torch.randperm(max_requests * page_count).view(max_requests, page_count).to(device)
    q_real = torch.randn(num_layers, num_q_heads, num_freqs, device=device)
    q_imag = torch.randn(num_layers, num_q_heads, num_freqs, device=device)
    mlr_coef = torch.randn(num_layers, num_q_heads, num_freqs, device=device)
    freq_scale_sq = torch.rand(num_freqs, device=device) + 0.5
    omega = torch.rand(num_freqs, device=device) * 0.05
    offsets_t = torch.tensor(offsets, dtype=torch.float32, device=device)
    capacity = page_count * tokens_per_block
    group = _FixedScoreGroup(
        pools,
        list(range(num_layers)),
        max_requests,
        page_count,
        capacity,
        num_q_heads,
        _encode_block_offsets(page_ids),
        [0] * num_layers,
        q_real,
        q_imag,
        mlr_coef,
        freq_scale_sq,
        omega,
        offsets_t,
        output_width=capacity - prompt_len,
        kv_scales=kv_scales,
    )
    # LIFETIME: the group records only raw device ADDRESSES of the scored
    # layer pools (layer_base_addrs) and references just the anchor pool
    # (pools[0], the kernel dtype witness). Production pools are owned by the
    # KV-cache manager, so the group deliberately does not hold them; the
    # test must keep the whole pool list alive itself. Dropping it here frees
    # every non-anchor layer pool, whose blocks the caching allocator then
    # recycles for the fold/output/reference tensors allocated later in the
    # test — a use-after-free that reads back sentinel/coefficient bytes as
    # K data (observed as layer>=1 inf/NaN score garbage).
    group.test_pools_keepalive = pools
    round_starts = (torch.arange(max_requests, dtype=torch.int32, device=device) + 9).contiguous()
    token_starts = torch.full((max_requests,), prompt_len, dtype=torch.int32, device=device)
    seq_lens = [capacity - ((request * 3) % 5) for request in range(request_count)]
    valid_seq_lens = torch.tensor(seq_lens, dtype=torch.int32, device=device)
    phase = (round_starts.float()[:, None, None] + offsets_t[None, :, None]) * omega[None, None, :]
    mean_cos = torch.cos(phase).mean(dim=1)
    mean_sin = torch.sin(phase).mean(dim=1)
    # Everything the PyTorch oracle needs to rebuild the reference leg
    # independently (it recomputes its own mean phases from these).
    oracle_inputs = dict(
        page_ids=page_ids,
        q_real=q_real,
        q_imag=q_imag,
        mlr_coef=mlr_coef,
        freq_scale_sq=freq_scale_sq,
        omega=omega,
        offsets=offsets_t,
    )
    return (
        group,
        round_starts,
        token_starts,
        valid_seq_lens,
        seq_lens,
        mean_cos,
        mean_sin,
        oracle_inputs,
    )


# ---------------------------------------------------------------------------
# Quantized (fp8_e4m3fn / int8) KV pools — FUNCTIONAL kernel-level coverage
# ONLY. End-to-end quantized-KV eviction is NOT validated here or anywhere
# else yet: nothing in the pipeline produces quantized pools, so these tests
# exercise the scalar score path + coefficient scale fold in isolation. The
# reference leg dequantizes the SAME quantized elements to fp32 pools and
# runs the PyTorch oracle on them, so quantization error cancels and only the
# scale-fold/loading math is under test.
# ---------------------------------------------------------------------------

_QUANTIZED_GEOMETRY = dict(
    request_count=3,
    max_requests=4,
    num_layers=2,
    page_count=4,
    tokens_per_block=4,
    head_dim=8,
    num_q_heads=4,
    num_kv_heads=2,
    prompt_len=5,
    offsets=[1.0, 2.0, 4.0],
)


def _quantize_pools(raw_pools: list, dtype: torch.dtype):
    """Per-layer amax quantization of fp32 pools.

    Returns (quantized pools, dequantized fp32 twin pools, per-layer scales).
    The twin holds values dequantized FROM the quantized elements — NOT the
    pre-quantization randoms — so quantization error is present in both legs
    identically and the CUDA-vs-reference tolerance can stay tight.
    """
    quant_max = 448.0 if dtype == torch.float8_e4m3fn else 127.0
    quantized, dequantized, scales = [], [], []
    for raw in raw_pools:
        scale = raw.abs().amax().float() / quant_max
        if dtype == torch.int8:
            quant = torch.round(raw / scale).clamp(-127, 127).to(torch.int8)
        else:
            quant = (raw / scale).to(torch.float8_e4m3fn)
        quantized.append(quant)
        dequantized.append(quant.to(torch.float32) * scale)
        scales.append(scale)
    return quantized, dequantized, torch.stack(scales)


def _build_quantized_raw_pools(seed: int, device: torch.device) -> list:
    """The shared fp32 randoms both quantized-test pools derive from."""
    torch.manual_seed(seed)
    g = _QUANTIZED_GEOMETRY
    return [
        torch.randn(
            g["max_requests"] * g["page_count"],
            2,
            g["num_kv_heads"],
            g["tokens_per_block"],
            g["head_dim"],
            device=device,
        )
        for _ in range(g["num_layers"])
    ]


# One entry per geometry class the launcher must cover. expected_vectorized
# white-boxes the launch-time path selection so the matrix provably exercises
# both CUDA load paths.
_CASES = [
    pytest.param(
        dict(
            head_dim=128,
            tokens_per_block=32,
            num_q_heads=8,
            num_kv_heads=2,
            dtype=torch.bfloat16,
            offsets=[1.0, 2.0, 4.0],
            aggregation="mean",
        ),
        True,
        id="production_bf16_f64_group4_mean",
    ),
    pytest.param(
        dict(
            head_dim=128,
            tokens_per_block=32,
            num_q_heads=8,
            num_kv_heads=2,
            dtype=torch.bfloat16,
            offsets=[1.0, 2.0, 4.0, 8.0],
            aggregation="max",
        ),
        True,
        id="max_aggregation_four_offsets",
    ),
    pytest.param(
        dict(
            head_dim=128,
            tokens_per_block=32,
            num_q_heads=4,
            num_kv_heads=2,
            dtype=torch.float16,
            offsets=[1.0, 2.0, 4.0],
            aggregation="mean",
        ),
        True,
        id="fp16_pool",
    ),
    pytest.param(
        dict(
            head_dim=128,
            tokens_per_block=32,
            num_q_heads=6,
            num_kv_heads=2,
            dtype=torch.bfloat16,
            offsets=[1.0, 2.0, 4.0],
            aggregation="mean",
        ),
        True,
        id="group3_generic_head_mapping",
    ),
    pytest.param(
        dict(
            head_dim=32,
            tokens_per_block=32,
            num_q_heads=4,
            num_kv_heads=2,
            dtype=torch.bfloat16,
            offsets=[1.0, 2.0, 4.0],
            aggregation="mean",
        ),
        True,
        id="f16_runtime_chunk_count",
    ),
    pytest.param(
        dict(
            head_dim=8,
            tokens_per_block=4,
            num_q_heads=4,
            num_kv_heads=2,
            dtype=torch.bfloat16,
            offsets=[1.0, 2.0, 4.0],
            aggregation="mean",
        ),
        False,
        id="tiny_f4_scalar",
    ),
    pytest.param(
        dict(
            head_dim=8,
            tokens_per_block=4,
            num_q_heads=4,
            num_kv_heads=2,
            dtype=torch.bfloat16,
            offsets=[1.0, 2.0, 4.0],
            aggregation="max",
        ),
        False,
        id="tiny_f4_scalar_max",
    ),
    pytest.param(
        dict(
            head_dim=12,
            tokens_per_block=4,
            num_q_heads=4,
            num_kv_heads=2,
            dtype=torch.bfloat16,
            offsets=[1.0, 2.0, 4.0],
            aggregation="mean",
        ),
        False,
        id="f6_nonpow2_masked_tail",
    ),
    # fp32 pools always take the scalar path; group=3 additionally exercises
    # its runtime GQA loop against the reference (the other generic-group
    # case is vectorized, the other scalar cases use a templated group size).
    pytest.param(
        dict(
            head_dim=8,
            tokens_per_block=4,
            num_q_heads=6,
            num_kv_heads=2,
            dtype=torch.float32,
            offsets=[1.0, 2.0, 4.0],
            aggregation="mean",
        ),
        False,
        id="fp32_scalar_generic_group3",
    ),
]


class TestTriAttentionScoreOps:
    @pytest.mark.parametrize("case,expected_vectorized", _CASES)
    def test_cuda_ops_match_torch_oracle(self, case, expected_vectorized):
        _require_score_ops()
        case = dict(case)  # parametrize reuses the dict across reruns
        aggregation = case.pop("aggregation")
        request_count = 3
        prompt_len = 5
        (
            group,
            round_starts,
            token_starts,
            valid_seq_lens,
            seq_lens,
            mean_cos,
            mean_sin,
            oracle_inputs,
        ) = _build_case(
            request_count=request_count,
            max_requests=4,
            num_layers=2,
            page_count=4,
            prompt_len=prompt_len,
            seed=20260719,
            **case,
        )
        assert group._use_vectorized == expected_vectorized
        device = group.output.device
        sentinel = -54321.0

        group.output.fill_(sentinel)
        valid_widths_cuda = torch.empty(request_count, dtype=torch.int32, device=device)
        cuda_scores = group.launch(
            request_count,
            valid_seq_lens,
            valid_widths_cuda,
            round_starts,
            token_starts,
            mean_cos,
            mean_sin,
            aggregation,
        ).clone()

        # The oracle reads the same stored pool elements (up-cast to fp32,
        # like the ops' loads), so the legs differ only by coefficient-fold
        # association and reduction order.
        reference = _oracle_reference(
            group,
            group.test_pools_keepalive,
            oracle_inputs,
            request_count,
            seq_lens,
            round_starts,
            prompt_len,
            aggregation,
            sentinel,
        )

        assert valid_widths_cuda.tolist() == [seq_len - prompt_len for seq_len in seq_lens]
        # Sentinel-filled outputs make the comparison cover the write MASK as
        # well as the values: any stray or missing store breaks equality.
        # The ops' fp32 math tracks this oracle to ~2e-6 on these geometries,
        # so 1e-4 is a tight gate with ample margin.
        torch.testing.assert_close(cuda_scores, reference[:request_count], rtol=1e-4, atol=1e-4)

    @pytest.mark.parametrize("dtype", [torch.float8_e4m3fn, torch.int8], ids=["fp8_e4m3fn", "int8"])
    @pytest.mark.parametrize("aggregation", ["mean", "max"])
    def test_quantized_pool_matches_dequantized_reference(self, dtype, aggregation):
        """Scale-folded scoring of quantized pools == dense scoring of their fp32 twin.

        The max aggregation additionally covers the per-offset coefficient
        planes, which must all carry the folded per-layer scale.
        """
        _require_score_ops()
        device = torch.device("cuda", torch.cuda.current_device())
        seed = 20260721
        raw_pools = _build_quantized_raw_pools(seed, device)
        quant_pools, dequant_pools, kv_scales = _quantize_pools(raw_pools, dtype)
        (
            quant_group,
            round_starts,
            token_starts,
            valid_seq_lens,
            seq_lens,
            mean_cos,
            mean_sin,
            oracle_inputs,
        ) = _build_case(
            dtype=dtype,
            seed=seed,
            pools=quant_pools,
            kv_scales=kv_scales,
            **_QUANTIZED_GEOMETRY,
        )
        # Quantized pools must never select the vectorized load path.
        assert quant_group._use_vectorized is False
        request_count = _QUANTIZED_GEOMETRY["request_count"]
        sentinel = -54321.0

        quant_group.output.fill_(sentinel)
        valid_widths_cuda = torch.empty(request_count, dtype=torch.int32, device=device)
        cuda_scores = quant_group.launch(
            request_count,
            valid_seq_lens,
            valid_widths_cuda,
            round_starts,
            token_starts,
            mean_cos,
            mean_sin,
            aggregation,
        ).clone()

        # Reference leg: the ORACLE over the dequantized-from-quantized fp32
        # twin pools (same page tables and calibration as the quantized group).
        reference = _oracle_reference(
            quant_group,
            dequant_pools,
            oracle_inputs,
            request_count,
            seq_lens,
            round_starts,
            _QUANTIZED_GEOMETRY["prompt_len"],
            aggregation,
            sentinel,
        )

        assert valid_widths_cuda.tolist() == [
            seq_len - _QUANTIZED_GEOMETRY["prompt_len"] for seq_len in seq_lens
        ]
        # Quantization error is identical in both legs (the reference pool is
        # dequantized from the quantized values), so the only differences are
        # scale-fold association (q*(s*c) vs (q*s)*c), the approximate sqrt,
        # and reduction order — hence a tolerance close to the float cases'.
        torch.testing.assert_close(cuda_scores, reference[:request_count], rtol=3e-3, atol=3e-3)

    def test_quantized_pool_missing_scales_raises(self):
        with pytest.raises(ValueError, match="require per-layer kv_scales"):
            _build_case(dtype=torch.int8, seed=20260722, **_QUANTIZED_GEOMETRY)

    def test_scales_with_float_pool_raises(self):
        device = torch.device("cuda", torch.cuda.current_device())
        scales = torch.ones(_QUANTIZED_GEOMETRY["num_layers"], device=device)
        with pytest.raises(ValueError, match="only valid for quantized"):
            _build_case(
                dtype=torch.bfloat16, seed=20260722, kv_scales=scales, **_QUANTIZED_GEOMETRY
            )

    def test_negative_scale_raises(self):
        """Positivity is enforced host-side in the C++ op, at launch time.

        The |K| coefficient fold assumes |scale * K_q| == scale * |K_q|, which
        breaks silently for non-positive scales, so the op must refuse them.
        """
        _require_score_ops()
        device = torch.device("cuda", torch.cuda.current_device())
        seed = 20260722
        raw_pools = _build_quantized_raw_pools(seed, device)
        quant_pools, _, kv_scales = _quantize_pools(raw_pools, torch.int8)
        bad_scales = kv_scales.clone()
        bad_scales[0] = -bad_scales[0]
        group, round_starts, token_starts, valid_seq_lens, _, mean_cos, mean_sin, _ = _build_case(
            dtype=torch.int8,
            seed=seed,
            pools=quant_pools,
            kv_scales=bad_scales,
            **_QUANTIZED_GEOMETRY,
        )
        request_count = _QUANTIZED_GEOMETRY["request_count"]
        valid_widths = torch.empty(request_count, dtype=torch.int32, device=device)
        with pytest.raises(RuntimeError, match="strictly positive"):
            group.launch(
                request_count,
                valid_seq_lens,
                valid_widths,
                round_starts,
                token_starts,
                mean_cos,
                mean_sin,
                "mean",
            )

    def test_short_scales_raise(self):
        """kv_scales must cover every calibrated layer, enforced at launch.

        The Python group only gates presence; the extent contract lives in the
        C++ ops (segments index the fold tables by absolute layer id on
        device, where a short scale tensor could not be range-checked).
        """
        _require_score_ops()
        device = torch.device("cuda", torch.cuda.current_device())
        seed = 20260722
        raw_pools = _build_quantized_raw_pools(seed, device)
        quant_pools, _, kv_scales = _quantize_pools(raw_pools, torch.int8)
        short_scales = kv_scales[:1]  # the geometry calibrates two layers
        group, round_starts, token_starts, valid_seq_lens, _, mean_cos, mean_sin, _ = _build_case(
            dtype=torch.int8,
            seed=seed,
            pools=quant_pools,
            kv_scales=short_scales,
            **_QUANTIZED_GEOMETRY,
        )
        request_count = _QUANTIZED_GEOMETRY["request_count"]
        valid_widths = torch.empty(request_count, dtype=torch.int32, device=device)
        with pytest.raises(RuntimeError, match="one scale per calibrated layer"):
            group.launch(
                request_count,
                valid_seq_lens,
                valid_widths,
                round_starts,
                token_starts,
                mean_cos,
                mean_sin,
                "mean",
            )

    def test_unsupported_pool_dtype_raises(self):
        _require_score_ops()
        # fp32 pools stay supported (the existing tiny-geometry unit suite
        # drives them through this launcher); fp64 is genuinely outside the
        # op's coverage and must fail loudly instead of routing elsewhere.
        request_count = 2
        group, round_starts, token_starts, valid_seq_lens, _, mean_cos, mean_sin, _ = _build_case(
            request_count=request_count,
            max_requests=2,
            num_layers=1,
            page_count=2,
            prompt_len=1,
            seed=20260720,
            head_dim=8,
            tokens_per_block=4,
            num_q_heads=2,
            num_kv_heads=1,
            dtype=torch.float64,
            offsets=[1.0, 2.0],
        )
        valid_widths = torch.empty(request_count, dtype=torch.int32, device=group.output.device)
        with pytest.raises(RuntimeError, match="unsupported KV pool dtype"):
            group.launch(
                request_count,
                valid_seq_lens,
                valid_widths,
                round_starts,
                token_starts,
                mean_cos,
                mean_sin,
                "mean",
            )
