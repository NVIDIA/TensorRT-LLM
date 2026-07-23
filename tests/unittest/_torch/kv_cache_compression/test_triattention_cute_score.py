# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The SM100 TriAttention CuTe scorer (the only score path) vs PyTorch oracles.

Two layers of coverage over the production score leg (workspace metadata
staging, the compiled runner launch, and the decode-window gather -- the same
sequence ``run_eviction_round`` fires). The kernel-numerics matrix drives a
single-layer workspace across the supported page geometries (permuted
physical pages, ragged valid lengths, GQA group 4 riding the padded MMA tile)
against inline oracle math. The launch-path matrix drives multi-layer
workspaces across the named production geometries (Qwen3, GPT-OSS, the
originally validated 128-token-page shape) against the shared pure-PyTorch
oracle, sweeps request counts up to the workspace capacity, and checks the
per-request decode-width metadata the selection reduce kernels consume. The
contract test pins the loud-failure behavior: unsupported geometry raises
from the CuTe runner's own validation at workspace construction -- there is
no fallback score kernel.
"""

import pytest
import torch
from conftest import encode_block_offsets as _encode_block_offsets
from conftest import torch_tri_score_oracle as _torch_tri_score_oracle

requires_sm100 = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 0),
    reason="TriAttention score requires SM100",
)


def _make_score_workspace(
    *,
    layer_pools,
    max_requests,
    seq_len,
    num_q_heads,
    q_real,
    q_imag,
    mlr_coef,
    freq_scale_sq,
    omega,
    offsets,
    decode_width=None,
    eviction_mode="per_head",
):
    """A score-only workspace over one shared page-table slot."""
    from tensorrt_llm._torch.kv_cache_compression.triattention.triattention import (
        prepare_eviction_workspace,
    )

    num_layers = len(layer_pools)
    return prepare_eviction_workspace(
        eviction_mode=eviction_mode,
        layer_pools=layer_pools,
        dense_groups=[list(range(num_layers))],
        dense_layers=list(range(num_layers)),
        page_representatives=[0],
        max_requests=max_requests,
        seq_len=seq_len,
        num_q_heads=num_q_heads,
        num_freqs=int(q_real.shape[-1]),
        keep_count=1,
        q_real=q_real,
        q_imag=q_imag,
        mlr_coef=mlr_coef,
        freq_scale_sq=freq_scale_sq,
        offsets=offsets,
        omega=omega,
        decode_width=decode_width,
        layer_group_representative={layer: 0 for layer in range(num_layers)},
        layer_pool_keys=[("pool", 0)] * num_layers,
    )


def _write_block_offsets(ws, encoded):
    """Load a test page table into the workspace's staged block-offset plane."""
    ws.block_offsets_device.zero_()
    ws.block_offsets_device[:, : encoded.shape[1], :, : encoded.shape[-1]].copy_(encoded)


def _launch_split_scores(
    ws, request_count, valid_seq_lens, valid_widths, token_starts, mean_cos, mean_sin
):
    """The production score-only leg: stage metadata, fire the compiled
    runner, gather each request's decode window (``run_eviction_round``'s
    per-head sequence, parameterized by the request count)."""
    num_segments = request_count * ws.num_layers
    torch.sub(
        valid_seq_lens[:request_count],
        token_starts[:request_count],
        out=valid_widths[:request_count],
    )
    torch.index_select(
        valid_seq_lens, 0, ws.seg_req[:num_segments], out=ws.seg_seq_len[:num_segments]
    )
    ws.cute_token_starts[:request_count].copy_(token_starts[:request_count])
    assert request_count in ws.runner._compiled
    ws.runner.launch(request_count, mean_cos, mean_sin)
    group_size = ws.num_q_heads // ws.num_kv_heads
    source = (
        ws.cute_scratch[: ws.num_kv_heads * 8 * num_segments * ws.bucket_seq_len]
        .view(ws.num_kv_heads, 8, request_count, ws.num_layers, ws.bucket_seq_len)[:, :group_size]
        .permute(2, 3, 0, 1, 4)
    )
    columns = token_starts[:request_count].to(torch.int64).view(-1, 1, 1, 1, 1) + ws.gather_columns
    columns = columns.clamp_(max=ws.bucket_seq_len - 1).expand(
        request_count, ws.num_layers, ws.num_kv_heads, group_size, ws.decode_width
    )
    output = torch.full(
        (request_count, ws.num_layers, ws.num_q_heads, ws.decode_width),
        float("nan"),
        dtype=torch.float32,
        device=ws.device,
    )
    torch.gather(
        source,
        4,
        columns,
        out=output.view(request_count, ws.num_layers, ws.num_kv_heads, group_size, ws.decode_width),
    )
    return output


@requires_sm100
@pytest.mark.parametrize(
    "tokens_per_block,page_permutation,valid_lens,num_freqs,num_q_heads",
    [
        # The originally validated geometry: 128-token pages, identity table.
        (128, [0, 1], None, 32, 8),
        # GPT-OSS geometry: 32-token pages; a 64-token compute tile spans two
        # pages, so a shuffled physical-page table catches fragment mix-ups.
        # Ragged tails land mid-tile: the second page fragment of the last
        # tile is clamped, and scores past the valid length are unspecified.
        (32, [3, 1, 4, 7, 5, 0, 2, 6], [250, 198], 32, 8),
        # Qwen3 geometry: 128-element K rows (64 frequencies) and GQA group
        # 4, which rides the MMA tile N=8 with zeroed padding columns.
        (32, [3, 1, 4, 7, 5, 0, 2, 6], [250, 198], 64, 4),
    ],
)
def test_cute_score_matches_torch_mean_oracle(
    tokens_per_block: int,
    page_permutation: list,
    valid_lens: "list | None",
    num_freqs: int,
    num_q_heads: int,
) -> None:
    pytest.importorskip("cutlass")

    torch.manual_seed(20260720)
    device = torch.device("cuda")
    seq_len = 256
    num_pages = seq_len // tokens_per_block
    assert sorted(page_permutation) == list(range(num_pages))
    pool = (
        0.125 * torch.randn(num_pages, 2, 1, tokens_per_block, 2 * num_freqs, device=device)
    ).to(torch.bfloat16)
    q_real = 0.125 * torch.randn(1, num_q_heads, num_freqs, device=device)
    q_imag = 0.125 * torch.randn_like(q_real)
    mlr_coef = 0.125 * torch.randn_like(q_real)
    freq_scale_sq = torch.linspace(0.5, 1.5, num_freqs, device=device)
    omega = torch.linspace(0.01, 0.03, num_freqs, device=device)
    offsets = torch.tensor([1.0, 2.0, 4.0], device=device)
    round_starts = torch.tensor([float(seq_len), float(seq_len + 1)], device=device)
    phase = (round_starts[:, None, None] + offsets[None, :, None]) * omega[None, None]
    mean_cos = torch.cos(phase).mean(dim=1).contiguous()
    mean_sin = torch.sin(phase).mean(dim=1).contiguous()

    ws = _make_score_workspace(
        layer_pools=[pool],
        max_requests=2,
        seq_len=seq_len,
        num_q_heads=num_q_heads,
        q_real=q_real,
        q_imag=q_imag,
        mlr_coef=mlr_coef,
        freq_scale_sq=freq_scale_sq,
        omega=omega,
        offsets=offsets,
    )
    # Native block-offset staging layout ([pool_slot, request, K/V plane,
    # block] int32): K-plane entries encode physical_page * kv_factor with
    # kv_factor == 2. Both requests read the same (permuted) page sequence.
    k_plane = [2 * page for page in page_permutation]
    v_plane = [2 * page + 1 for page in page_permutation]
    _write_block_offsets(
        ws,
        torch.tensor([[[k_plane, v_plane], [k_plane, v_plane]]], dtype=torch.int32, device=device),
    )
    keys = (
        torch.cat([pool[page, 0, 0] for page in page_permutation], dim=0)
        .reshape(seq_len, 2 * num_freqs)
        .float()
    )
    k_real = keys[:, :num_freqs]
    k_imag = keys[:, num_freqs:]
    magnitude = torch.sqrt(k_real.square() + k_imag.square())
    if valid_lens is None:
        valid_lens = [seq_len, seq_len]
    valid_seq_lens = torch.tensor(valid_lens, dtype=torch.int32, device=device)
    valid_widths = torch.tensor(valid_lens, dtype=torch.int32, device=device)
    token_starts_device = torch.zeros(2, dtype=torch.int32, device=device)
    for request_count in (1, 2):
        actual = _launch_split_scores(
            ws,
            request_count,
            valid_seq_lens,
            valid_widths,
            token_starts_device,
            mean_cos,
            mean_sin,
        )
        assert actual.shape == (request_count, 1, num_q_heads, seq_len)
        for request in range(request_count):
            rotated_real = freq_scale_sq * (k_real * mean_cos[request] + k_imag * mean_sin[request])
            rotated_imag = freq_scale_sq * (k_imag * mean_cos[request] - k_real * mean_sin[request])
            expected = (
                q_real[0, :, None] * rotated_real[None]
                + q_imag[0, :, None] * rotated_imag[None]
                + mlr_coef[0, :, None] * freq_scale_sq[None, None] * magnitude[None]
            ).sum(dim=-1)
            valid = valid_lens[request]
            torch.testing.assert_close(
                actual[request, 0, :, :valid],
                expected[:, :valid],
                rtol=5.0e-3,
                atol=5.0e-3,
            )

    torch.cuda.synchronize()
    # The CuTe runner is the only score path, compiled eagerly at workspace
    # construction; prove setup actually built it.
    assert ws.runner is not None


def _build_case(
    *,
    max_requests: int,
    num_layers: int,
    page_count: int,
    tokens_per_block: int,
    head_dim: int,
    num_q_heads: int,
    num_kv_heads: int,
    prompt_len: int,
    seed: int,
    offsets: tuple = (1.0, 2.0, 4.0),
):
    device = torch.device("cuda", torch.cuda.current_device())
    torch.manual_seed(seed)
    num_freqs = head_dim // 2
    # The 0.125 scaling keeps the BF16 key/coefficient products small so the
    # kernel-vs-oracle tolerance can stay tight across the frequency sum.
    pools = [
        (
            0.125
            * torch.randn(
                max_requests * page_count,
                2,
                num_kv_heads,
                tokens_per_block,
                head_dim,
                device=device,
            )
        ).to(torch.bfloat16)
        for _ in range(num_layers)
    ]
    page_ids = torch.randperm(max_requests * page_count).view(max_requests, page_count).to(device)
    q_real = 0.125 * torch.randn(num_layers, num_q_heads, num_freqs, device=device)
    q_imag = 0.125 * torch.randn(num_layers, num_q_heads, num_freqs, device=device)
    mlr_coef = 0.125 * torch.randn(num_layers, num_q_heads, num_freqs, device=device)
    freq_scale_sq = torch.rand(num_freqs, device=device) + 0.5
    omega = torch.rand(num_freqs, device=device) * 0.05
    offsets_t = torch.tensor(offsets, dtype=torch.float32, device=device)
    capacity = page_count * tokens_per_block
    ws = _make_score_workspace(
        layer_pools=pools,
        max_requests=max_requests,
        seq_len=capacity,
        num_q_heads=num_q_heads,
        q_real=q_real,
        q_imag=q_imag,
        mlr_coef=mlr_coef,
        freq_scale_sq=freq_scale_sq,
        omega=omega,
        offsets=offsets_t,
        decode_width=capacity - prompt_len,
    )
    _write_block_offsets(ws, _encode_block_offsets(page_ids))
    round_starts = (torch.arange(max_requests, dtype=torch.int32, device=device) + 9).contiguous()
    token_starts = torch.full((max_requests,), prompt_len, dtype=torch.int32, device=device)
    # Ragged valid lengths whose tails land mid-page and mid-compute-tile.
    seq_lens = [capacity - ((request * 3) % 5) for request in range(max_requests)]
    valid_seq_lens = torch.tensor(seq_lens, dtype=torch.int32, device=device)
    phase = (round_starts.float()[:, None, None] + offsets_t[None, :, None]) * omega[None, None, :]
    mean_cos = torch.cos(phase).mean(dim=1).contiguous()
    mean_sin = torch.sin(phase).mean(dim=1).contiguous()
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
        ws,
        pools,
        token_starts,
        valid_seq_lens,
        seq_lens,
        mean_cos,
        mean_sin,
        oracle_inputs,
    )


def _geometry(max_requests, num_layers, page_count, tokens_per_block, head_dim, num_q, num_kv):
    return dict(
        max_requests=max_requests,
        num_layers=num_layers,
        page_count=page_count,
        tokens_per_block=tokens_per_block,
        head_dim=head_dim,
        num_q_heads=num_q,
        num_kv_heads=num_kv,
    )


# One entry per supported production geometry: the Qwen3 shape (64
# frequencies, GQA group 4 riding the padded MMA tile), the GPT-OSS shape
# (32 frequencies, group 8, 32-token pages spanning two page fragments per
# compute tile), and the originally validated 128-token-page shape.
_CASES = [
    pytest.param(_geometry(4, 2, 4, 32, 128, 8, 2), id="qwen3_f64_group4_tpb32"),
    pytest.param(_geometry(2, 3, 4, 32, 64, 8, 1), id="gptoss_f32_group8_tpb32"),
    pytest.param(_geometry(2, 2, 2, 128, 64, 8, 1), id="original_f32_group8_tpb128"),
]


@requires_sm100
@pytest.mark.parametrize("case", _CASES)
def test_cute_kernel_matches_torch_oracle(case):
    pytest.importorskip("cutlass")
    case = dict(case)  # parametrize reuses the dict across reruns
    prompt_len = 5
    max_requests = case["max_requests"]
    num_layers = case["num_layers"]
    (
        ws,
        pools,
        token_starts,
        valid_seq_lens,
        seq_lens,
        mean_cos,
        mean_sin,
        oracle_inputs,
    ) = _build_case(prompt_len=prompt_len, seed=20260719, **case)
    device = ws.device

    oracle = _torch_tri_score_oracle(
        pools,
        oracle_inputs["page_ids"],
        seq_lens,
        [int(start) for start in range(9, 9 + max_requests)],
        oracle_inputs["q_real"],
        oracle_inputs["q_imag"],
        oracle_inputs["mlr_coef"],
        oracle_inputs["freq_scale_sq"],
        oracle_inputs["omega"],
        oracle_inputs["offsets"],
        list(range(num_layers)),
    )

    # The compiled runner serves every request count up to the workspace
    # capacity and nothing beyond it; cover one, an intermediate count, and
    # the capacity.
    assert max_requests + 1 not in ws.runner._compiled
    for request_count in dict.fromkeys((1, max_requests - 1, max_requests)):
        valid_widths = torch.full((max_requests,), -1, dtype=torch.int32, device=device)
        scores = _launch_split_scores(
            ws,
            request_count,
            valid_seq_lens,
            valid_widths,
            token_starts,
            mean_cos,
            mean_sin,
        )
        assert scores.shape == (
            request_count,
            num_layers,
            case["num_q_heads"],
            ws.decode_width,
        )
        # The score leg owns the per-request decode widths the selection
        # reduce kernels consume.
        assert valid_widths[:request_count].tolist() == [
            seq_lens[request] - prompt_len for request in range(request_count)
        ]
        for request in range(request_count):
            width = seq_lens[request] - prompt_len
            for layer in range(num_layers):
                torch.testing.assert_close(
                    scores[request, layer, :, :width],
                    oracle[request * num_layers + layer][:, prompt_len : prompt_len + width],
                    rtol=5e-3,
                    atol=5e-3,
                )


# The loud-failure contract: unsupported geometry raises from the CuTe
# runner's own validation during the eager compile at workspace
# construction, surfaced as the no-fallback RuntimeError.
def test_unsupported_geometry_raises_at_workspace_construction():
    pytest.importorskip("cutlass")
    device = torch.device("cuda", torch.cuda.current_device())
    torch.manual_seed(20260722)
    num_layers, max_requests, page_count, tokens_per_block, head_dim = 2, 2, 2, 4, 8
    num_freqs = head_dim // 2
    # fp32 pools with a 4-token page and 4 frequencies sit far outside the
    # CuTe contract on every device.
    pools = [
        torch.randn(max_requests * page_count, 2, 1, tokens_per_block, head_dim, device=device)
        for _ in range(num_layers)
    ]
    calib = torch.randn(num_layers, 2, num_freqs, device=device)
    with pytest.raises(RuntimeError, match="no other score path exists"):
        _make_score_workspace(
            layer_pools=pools,
            max_requests=max_requests,
            seq_len=page_count * tokens_per_block,
            num_q_heads=2,
            q_real=calib,
            q_imag=calib.clone(),
            mlr_coef=calib.clone(),
            freq_scale_sq=torch.rand(num_freqs, device=device) + 0.5,
            omega=torch.rand(num_freqs, device=device) * 0.05,
            offsets=torch.tensor([1.0, 2.0], dtype=torch.float32, device=device),
            decode_width=page_count * tokens_per_block - 1,
        )
