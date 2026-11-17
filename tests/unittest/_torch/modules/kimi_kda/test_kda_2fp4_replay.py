# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native 2FP4 replay, QDQ prefill boundaries, and indexed checkpoint tests."""

from collections.abc import Iterator

import pytest
import torch
import triton
import triton.language as tl

from tensorrt_llm._torch.modules.kimi_kda import kda_2fp4 as ops

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
    reason="Native NVFP4 replay requires Blackwell",
)


@pytest.fixture(autouse=True)
def _ieee_fp32_reference() -> Iterator[None]:
    # The reference dequantizes FP4 operands to FP32; TF32 would round them
    # again before GEMM and would no longer represent the native FP4 MMA.
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous


def _e2m1_rne(normalized: torch.Tensor) -> torch.Tensor:
    levels = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6], device=normalized.device)
    # argmin breaks ties toward the lower index, so explicitly prefer even
    # E2M1 codes on exact midpoints.
    distances = (normalized.abs()[..., None] - levels).abs()
    indices = distances.argmin(-1)
    lower_odd = (indices % 2 == 1) & (indices < 7)
    upper = (indices + 1).clamp(max=7)
    ties = distances.gather(-1, upper[..., None]).squeeze(-1) == distances.amin(-1)
    indices = torch.where(lower_odd & ties, upper, indices)
    return levels[indices] * normalized.sign()


def _quantize(values: torch.Tensor, outer: torch.Tensor) -> torch.Tensor:
    grouped = values.reshape(*values.shape[:-1], 8, 16)
    safe_outer = torch.where(outer == 0, 1.0, outer)
    scales = (grouped.abs().amax(-1) / (6 * safe_outer)).clamp(max=448)
    scales = scales.to(torch.float8_e4m3fn).float()
    expanded = (scales * outer).repeat_interleave(16, -1)
    normalized = torch.where(expanded == 0, 0.0, values / expanded)
    return _e2m1_rne(normalized) * expanded


@triton.jit
def _native_dynamic_operands(
    pair,
    raw_gate,
    a_log,
    bias,
    prefix,
    cumulative_out,
    normalized_out,
    scale_out,
    gate_stride,
    H: tl.constexpr,
    LOWER_BOUND: tl.constexpr,
):
    batch, head = tl.program_id(0), tl.program_id(1)
    col = tl.arange(0, 128)
    shifted = tl.load(raw_gate + batch * gate_stride + head * 128 + col).to(tl.float32)
    shifted += tl.load(bias + head * 128 + col)
    a = tl.exp(tl.load(a_log + head))
    if LOWER_BOUND is None:
        step = -a * tl.where(shifted <= 20.0, tl.log(1.0 + tl.exp(shifted)), shifted)
    else:
        step = LOWER_BOUND * tl.div_rn(1.0, 1.0 + tl.exp(-a * shifted))
    offset = (batch * H + head) * 128 + col
    cumulative = tl.load(prefix + offset) + step
    tl.store(cumulative_out + offset, cumulative)
    rows = tl.arange(0, 2)
    offsets = (batch * H + head) * 256 + rows[:, None] * 128 + col[None, :]
    dynamic = tl.load(pair + offsets) * tl.exp(cumulative)[None, :]
    # Native exp/div and outer-scale multiplication must precede the
    # independent PyTorch RNE quantizer: a one-ULP difference at an FP4
    # midpoint can otherwise select an entirely different code.
    outer = tl.max(tl.max(tl.abs(dynamic), 1), 0) * (1.0 / (6.0 * 448.0))
    safe_outer = tl.where(outer == 0, 1.0, outer)
    amax = tl.max(tl.abs(dynamic.reshape(2, 8, 16)), 2)
    scales = tl.minimum(448.0, tl.div_rn(amax, 6.0 * safe_outer)).to(tl.float8e4nv)
    expanded = tl.broadcast_to(scales.to(tl.float32)[:, :, None] * outer, (2, 8, 16))
    expanded = expanded.reshape(2, 128)
    normalized = tl.where(expanded == 0, 0.0, tl.div_rn(dynamic, expanded))
    tl.store(normalized_out + offsets, normalized)
    tl.store(scale_out + offsets, expanded)


def _checkpoint_operands(
    pair: torch.Tensor,
    raw_gate: torch.Tensor,
    a_log: torch.Tensor,
    bias: torch.Tensor,
    prefix: torch.Tensor,
    lower_bound: float | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    cumulative = torch.empty_like(prefix)
    normalized, scales = torch.empty_like(pair), torch.empty_like(pair)
    _native_dynamic_operands[(pair.shape[0], pair.shape[1])](
        pair,
        raw_gate,
        a_log,
        bias,
        prefix,
        cumulative,
        normalized,
        scales,
        raw_gate.stride(0),
        pair.shape[1],
        lower_bound,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return _e2m1_rne(normalized) * scales, cumulative


def _qdq(state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    outer = state.abs().amax((-1, -2), keepdim=True) / (6 * 448)
    primary = _quantize(state, outer)
    return primary, _quantize(state - primary, outer)


def _read_checkpoint(pool: torch.Tensor, slots: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Independent decoding of the physical bytes, without Triton helpers."""
    rows = pool[slots.long()].squeeze(-2).contiguous()
    outer = rows[..., 18432:18436].contiguous().view(torch.float32)[..., None]
    levels = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6], device=pool.device)
    decoded = []
    for value_offset, scale_offset in [(0, 16384), (8192, 17408)]:
        packed = rows[..., value_offset : value_offset + 8192].reshape(*rows.shape[:2], 128, 64)
        codes = torch.stack((packed & 15, packed >> 4), -1).flatten(-2)
        scales = rows[..., scale_offset : scale_offset + 1024].contiguous()
        scales = scales.view(torch.float8_e4m3fn).float().reshape(*rows.shape[:2], 128, 8)
        values = levels[(codes & 7).long()] * torch.where(codes & 8 != 0, -1.0, 1.0)
        decoded.append(values * (scales * outer).repeat_interleave(16, -1))
    return tuple(decoded)


def _assert_repacked_close(actual: torch.Tensor, expected: torch.Tensor) -> None:
    # A different FP32 MMA reduction order can cross an FP4 or E4M3
    # midpoint. Bound the resulting quantized error without requiring the
    # discontinuous quantizers to produce identical codes at those points.
    error = (actual - expected).norm() / expected.norm()
    assert error < 0.0015
    assert (actual - expected).abs().amax() < expected.abs().amax() * 0.02


_REPLAY_SLOTS = 6


def _replay_indices(batch: int) -> torch.Tensor:
    """Replay slots of the batch rows, unrelated to their state slots."""
    return torch.tensor([4, 0, 5, 2][:batch], dtype=torch.int32, device="cuda")


def _paged_history(
    heads: int,
    capacity: int,
    *,
    batch: int,
    tokens: int = 0,
    rows: int = _REPLAY_SLOTS,
) -> ops.ReplayHistory:
    """K/U/G histories and lengths as views of one page per replay slot.

    Like KV-cache pool pages, every replay slot's K, U, G and length planes
    share one strided page, so each history has a large slot stride and the
    length is strided. ``tokens`` adds the DSpark candidate buffers, otherwise
    the decode arrival counters.
    """
    plane = heads * capacity * 128
    # BF16 elements: K and U planes, G as FP32, a 16-byte length row, padding.
    pages = torch.zeros((rows, 4 * plane + 64), dtype=torch.bfloat16, device="cuda")
    shape = (rows, heads, capacity, 128)
    k = pages[:, :plane].view(shape)
    u = pages[:, plane : 2 * plane].view(shape)
    g = pages[:, 2 * plane : 4 * plane].view(torch.float32).view(shape)
    length = pages[:, 4 * plane : 4 * plane + 8].view(torch.int32)[:, 0]
    if tokens:
        return ops.ReplayHistory(
            k,
            u,
            g,
            length,
            torch.empty((batch, tokens, 3 * heads * 128), dtype=torch.bfloat16, device="cuda"),
            torch.empty(batch, dtype=torch.int32, device="cuda"),
        )
    return ops.ReplayHistory(
        k, u, g, length, arrivals=torch.zeros(batch, dtype=torch.int32, device="cuda")
    )


def _fixture(batch: int, heads: int) -> tuple:
    torch.manual_seed(123)
    # Dense head payloads but noncontiguous V2 slots.
    backing = torch.zeros(
        (22, heads, 1, ops.STATE_BYTES_PER_HEAD), dtype=torch.uint8, device="cuda"
    )
    pool = backing[::2]
    slots = torch.tensor([9, 1, 7, 3][:batch], dtype=torch.int32, device="cuda")
    history = _paged_history(heads, ops.HISTORY_SIZE, batch=batch)
    state = torch.randn((batch, heads, 128, 128), device="cuda") * 0.1
    ops.store_checkpoint(pool, slots, state)
    return pool, slots, history, state


def _fused_inputs(batch: int, heads: int, slots: int = 11) -> dict[str, torch.Tensor]:
    width = heads * 128
    return {
        "conv_state": torch.randn((2 * slots, 3 * width, 3), device="cuda", dtype=torch.bfloat16)[
            ::2
        ],
        "conv_weight": torch.randn((3 * width, 4), device="cuda", dtype=torch.bfloat16) * 0.2,
        "z": torch.randn((batch, heads, 128), device="cuda", dtype=torch.bfloat16),
        "norm_weight": torch.ones(128, device="cuda", dtype=torch.bfloat16),
    }


def _verification_history(batch: int, heads: int, tokens: int) -> ops.ReplayHistory:
    return _paged_history(heads, ops.HISTORY_SIZE + tokens, batch=batch, tokens=tokens)


def _verification_descriptors(
    pool: torch.Tensor, conv: torch.Tensor, history: ops.ReplayHistory
) -> torch.Tensor:
    return torch.tensor(
        [ops.commit_descriptor(pool, conv, history)], dtype=torch.int64, device="cuda"
    )


@pytest.mark.parametrize(
    ("heads", "tokens", "scheduling_sms"),
    [(1, 1, None), (3, 4, None), (3, 8, None), (12, 7, None), (12, 4, 1), (96, 7, 1)],
)
def test_dspark_native_verify_acceptance_and_rollover(
    heads: int,
    tokens: int,
    scheduling_sms: int | None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only accepted candidates advance Conv/state; rejected suffixes are ignored."""
    if scheduling_sms is not None:
        from tensorrt_llm._torch.custom_ops import cute_dsl_kda_2fp4_mtp_replay as replay_ops

        # A small scheduling budget exercises the packed kernel without
        # allocating a production-sized batch in this test.
        monkeypatch.setattr(replay_ops, "_num_sms", lambda: scheduling_sms)
    batch, capacity = 3, 16 + tokens
    pool, slots, _, _ = _fixture(batch, heads)
    replay = _replay_indices(batch)
    live = replay.long()
    lengths = torch.tensor([11, 15, 3], device="cuda", dtype=torch.int32)
    history = _verification_history(batch, heads, tokens)
    shape = (batch, heads, capacity, 128)
    # Batch row b's history: row batch - 1 - b of each random tensor.
    rows = torch.arange(batch - 1, -1, -1, device="cuda")
    history.k[live] = (torch.randn(shape, device="cuda").bfloat16() * 0.03)[rows]
    history.u[live] = (torch.randn(shape, device="cuda").bfloat16() * 0.03)[rows]
    history.g[live] = (-torch.rand(shape, device="cuda").cumsum(2) * 0.02)[rows]
    history.length[live] = lengths
    dummy = torch.tensor([False, False, True], device="cuda")
    fused = _fused_inputs(batch * tokens, heads)
    a_log = torch.full((heads,), -2.0, device="cuda")
    bias = torch.full((heads, 128), -1.0, device="cuda")
    raw = torch.randn(batch * tokens, 4 * heads * 128, device="cuda", dtype=torch.bfloat16)[
        :, : 3 * heads * 128
    ]
    gate = torch.randn(batch * tokens, heads * 128, device="cuda", dtype=torch.bfloat16)
    beta = torch.randn(batch * tokens, heads, device="cuda", dtype=torch.bfloat16)
    before_pool, before_conv, before_lengths = (
        pool.clone(),
        fused["conv_state"].clone(),
        history.length.clone(),
    )
    primary, residual = _read_checkpoint(pool, slots)
    expected_k, expected_u, expected_g = history.k.clone(), history.u.clone(), history.g.clone()
    ref_fused = dict(fused, conv_state=before_conv.clone())
    outputs = []
    for token in range(tokens):
        x = raw.reshape(batch, tokens, -1)[:, token]
        conv, next_conv = _conv_reference(x, ref_fused, slots)
        ref_fused["conv_state"][slots.long()] = next_conv
        q, k, v = (t.reshape(batch, heads, 128) for t in conv.split(heads * 128, -1))
        qn = (
            q.float() * 128**-0.5 / (q.float().square().sum(-1, keepdim=True) + 1e-6).sqrt()
        ).bfloat16()
        kn = (k.float() / (k.float().square().sum(-1, keepdim=True) + 1e-6).sqrt()).bfloat16()
        position = lengths.long() + token
        prefix = expected_g[live, :, position - 1]
        pair = torch.stack((kn.float(), qn.float()), -2)
        dynamic, cumulative = _checkpoint_operands(
            pair, gate.reshape(batch, tokens, -1)[:, token], a_log, bias, prefix, -5.0
        )
        checkpoint = primary @ dynamic.transpose(-1, -2)
        mask = (
            torch.arange(capacity, device="cuda")[None, None, :, None]
            < position[:, None, None, None]
        )
        adjusted = torch.where(
            mask, expected_k[live].float() * (cumulative[:, :, None] - expected_g[live]).exp(), 0
        ).bfloat16()
        coefficients = (adjusted.float() @ pair.transpose(-1, -2)).bfloat16()
        history_pair = expected_u[live].float().transpose(-1, -2) @ coefficients.float()
        update = torch.sigmoid(beta.reshape(batch, tokens, heads)[:, token].float())[..., None] * (
            v.float() - checkpoint[..., 0] - history_pair[..., 0]
        )
        core = (
            checkpoint[..., 1]
            + history_pair[..., 1]
            + update * (kn.float() * qn.float()).sum(-1, keepdim=True)
        ).bfloat16()
        output_fused = dict(fused, z=fused["z"].reshape(batch, tokens, heads, 128)[:, token])
        outputs.append(_norm_reference(core, output_fused))
        expected_k[live, :, position] = kn
        expected_u[live, :, position] = update.bfloat16()
        expected_g[live, :, position] = cumulative
    inactive = (
        torch.arange(capacity, device="cuda")[None, None, :, None]
        >= history.length[:, None, None, None]
    )
    history.u.masked_fill_(inactive, float("nan"))
    actual = ops.verify(
        raw,
        **fused,
        gate=gate,
        beta=beta,
        pool=pool,
        slots=slots,
        replay_indices=replay,
        dummy=dummy,
        history=history,
        A_log=a_log,
        dt_bias=bias,
        lower_bound=-5.0,
        norm_eps=1e-6,
    )
    expected = torch.stack(outputs, 1)
    expected[dummy] = 0
    torch.testing.assert_close(actual.view_as(expected), expected, atol=0.003, rtol=0.03)
    torch.testing.assert_close(pool, before_pool, rtol=0, atol=0)
    torch.testing.assert_close(fused["conv_state"], before_conv, rtol=0, atol=0)
    torch.testing.assert_close(history.length, before_lengths, rtol=0, atol=0)
    torch.testing.assert_close(history.old_lengths[~dummy], lengths[~dummy], rtol=0, atol=0)
    descriptor = _verification_descriptors(pool, fused["conv_state"], history)
    accepted = torch.tensor([min(tokens, 2), tokens, 0], dtype=torch.int32, device="cuda")
    ops.commit_verified(descriptor, slots, replay, accepted, dummy, heads, capacity, tokens)
    for b in range(batch):
        slot, row = slots[b].item(), live[b].item()
        if dummy[b]:
            torch.testing.assert_close(pool[slot], before_pool[slot], rtol=0, atol=0)
            assert history.length[row] == lengths[b]
            continue
        count = accepted[b].item()
        window = torch.cat((before_conv[slot], raw.reshape(batch, tokens, -1)[b, :count].T), -1)
        torch.testing.assert_close(fused["conv_state"][slot], window[:, -3:], rtol=0, atol=0)
        n = lengths[b].item() + count
        assert history.length[row] == (0 if n >= 16 else n)
        if n >= 16:
            last_g = history.g[row, :, n - 1]
            adjusted = (
                history.k[row, :, :n].float() * (last_g[:, None] - history.g[row, :, :n]).exp()
            ).bfloat16()
            dense = (primary[b] + residual[b]) * last_g.exp()[:, None]
            dense += history.u[row, :, :n].float().transpose(-1, -2) @ adjusted.float()
            target_primary, target_residual = _qdq(dense)
            restored = ops.materialize_checkpoint(
                pool, slots[b : b + 1], torch.ones(1, device="cuda", dtype=torch.bool)
            )
            _assert_repacked_close(restored[0], target_primary + target_residual)
        else:
            torch.testing.assert_close(pool[slot], before_pool[slot], rtol=0, atol=0)

    # The next round must ignore the rejected suffix and start from the
    # accepted Conv window/history, including the freshly repacked row.
    scalar_pool, scalar_conv = pool.clone(), fused["conv_state"].clone()
    # The ordinary one-token kernel consumes fixed-capacity, 16-row histories.
    scalar_history = ops.ReplayHistory(
        history.k[:, :, : ops.HISTORY_SIZE].contiguous(),
        history.u[:, :, : ops.HISTORY_SIZE].contiguous(),
        history.g[:, :, : ops.HISTORY_SIZE].contiguous(),
        history.length.clone(),
        arrivals=torch.zeros(batch, dtype=torch.int32, device="cuda"),
    )
    next_raw = raw.reshape(batch, tokens, -1)[:, 0]
    kwargs = dict(
        conv_weight=fused["conv_weight"],
        gate=gate.reshape(batch, tokens, -1)[:, 0],
        beta=beta.reshape(batch, tokens, heads)[:, 0],
        z=fused["z"].reshape(batch, tokens, heads, 128)[:, 0],
        norm_weight=fused["norm_weight"],
        slots=slots,
        replay_indices=replay,
        dummy=dummy,
        A_log=a_log,
        dt_bias=bias,
        lower_bound=-5.0,
        norm_eps=1e-6,
    )
    reference = ops.replay(
        next_raw, conv_state=scalar_conv, pool=scalar_pool, history=scalar_history, **kwargs
    )
    actual = ops.verify(
        next_raw, conv_state=fused["conv_state"], pool=pool, history=history, **kwargs
    )
    torch.testing.assert_close(actual, reference, atol=0.003, rtol=0.03)
    ops.commit_verified(
        descriptor,
        slots,
        replay,
        torch.ones_like(accepted),
        dummy,
        heads,
        capacity,
        tokens,
    )
    torch.testing.assert_close(history.length[live], scalar_history.length[live], rtol=0, atol=0)
    torch.testing.assert_close(fused["conv_state"], scalar_conv, rtol=0, atol=0)


@pytest.mark.parametrize("layout", ["affine", "indirect"])
def test_dspark_commit_layer_descriptors_and_reduced_token_width(layout: str) -> None:
    """Address independent layers and keep the configured stride when T shrinks."""
    batch, heads, layers, max_tokens, tokens = 4, 3, 3, 8, 4
    capacity = ops.HISTORY_SIZE + max_tokens
    width = heads * 128
    if layout == "affine":
        pooled_states = torch.zeros(
            (11, layers, heads, 1, ops.STATE_BYTES_PER_HEAD), dtype=torch.uint8, device="cuda"
        )
        pooled_conv = torch.randn(11, layers, 3 * width, 3, device="cuda").bfloat16()
    dummy = torch.tensor([False, False, False, True], device="cuda")
    accepted = torch.tensor([4, 2, 0, 4], device="cuda", dtype=torch.int32)
    descriptors, records = [], []
    replay = _replay_indices(batch)
    for layer in range(layers):
        pool, slots, _, _ = _fixture(batch, heads)
        torch.manual_seed(150 + layer)
        if layout == "affine":
            pool = pooled_states[:, layer]
            conv = pooled_conv[:, layer]
        else:
            pool = torch.zeros(
                (11 * (layer + 1), heads, 1, ops.STATE_BYTES_PER_HEAD),
                dtype=torch.uint8,
                device="cuda",
            )[:: layer + 1]
            conv = torch.randn(11 * (layer + 1), 3 * width, 3, device="cuda").bfloat16()[
                :: layer + 1
            ]
        ops.store_checkpoint(pool, slots, torch.randn(batch, heads, 128, 128, device="cuda") * 0.1)
        history = _verification_history(batch, heads, max_tokens)
        history.k.normal_(std=0.03)
        history.u.normal_(std=0.03)
        history.g.copy_(-torch.rand_like(history.g).cumsum(2) * 0.02)
        live = replay.long()
        old = torch.tensor([12 + layer, 7, 5, 3], device="cuda", dtype=torch.int32)
        history.length[live] = old
        raw = torch.randn(batch * tokens, 4 * width, device="cuda").bfloat16()[:, : 3 * width]
        fused = _fused_inputs(batch * tokens, heads)
        fused["conv_state"] = conv
        before_pool, before_conv = pool.clone(), conv.clone()
        ops.verify(
            raw,
            **fused,
            gate=torch.randn(batch * tokens, width, device="cuda").bfloat16(),
            beta=torch.randn(batch * tokens, heads, device="cuda").bfloat16(),
            pool=pool,
            slots=slots,
            replay_indices=replay,
            dummy=dummy,
            history=history,
            A_log=torch.full((heads,), -2.0, device="cuda"),
            dt_bias=torch.full((heads, 128), -1.0, device="cuda"),
            lower_bound=-5.0,
            norm_eps=1e-6,
        )
        torch.testing.assert_close(pool, before_pool, atol=0, rtol=0)
        torch.testing.assert_close(conv, before_conv, atol=0, rtol=0)
        descriptors.append(_verification_descriptors(pool, conv, history))
        records.append((pool, conv, history, live, old, raw, before_pool, before_conv))
    ops.commit_verified(
        torch.cat(descriptors),
        slots,
        replay,
        accepted,
        dummy,
        heads,
        capacity,
        max_tokens,
    )
    for pool, conv, history, live, old, raw, before_pool, before_conv in records:
        for row in range(batch):
            slot, active_row = slots[row].item(), live[row].item()
            count = 0 if dummy[row] else accepted[row].item()
            n = old[row].item() + count
            window = torch.cat((before_conv[slot], raw.view(batch, tokens, -1)[row, :count].T), -1)
            torch.testing.assert_close(conv[slot], window[:, -3:], atol=0, rtol=0)
            assert history.length[active_row] == (0 if n >= ops.HISTORY_SIZE else n)
            if count and n >= ops.HISTORY_SIZE:
                primary, residual = _read_checkpoint(before_pool, slots[row : row + 1])
                end = history.g[active_row, :, n - 1]
                adjusted = (
                    history.k[active_row, :, :n].float()
                    * (end[:, None] - history.g[active_row, :, :n]).exp()
                ).bfloat16()
                state = (primary + residual)[0] * end.exp()[:, None]
                state += history.u[active_row, :, :n].float().transpose(-1, -2) @ adjusted.float()
                target_primary, target_residual = _qdq(state)
                actual = ops.materialize_checkpoint(
                    pool, slots[row : row + 1], torch.ones(1, device="cuda", dtype=torch.bool)
                )
                _assert_repacked_close(actual[0], target_primary + target_residual)
            else:
                torch.testing.assert_close(pool[slot], before_pool[slot], atol=0, rtol=0)


def _conv_reference(
    raw: torch.Tensor, fused: dict[str, torch.Tensor], slots: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    window = torch.cat((fused["conv_state"][slots.long()], raw[..., None]), dim=-1)
    acc = torch.zeros_like(raw, dtype=torch.float32)
    for index in range(4):
        acc = torch.addcmul(acc, window[..., index].float(), fused["conv_weight"][:, index].float())
    return torch.nn.functional.silu(acc).bfloat16(), window[..., 1:]


def _norm_reference(core: torch.Tensor, fused: dict[str, torch.Tensor]) -> torch.Tensor:
    values = core.float()
    values *= torch.rsqrt(values.square().mean(-1, keepdim=True) + 1e-6)
    return (values * fused["norm_weight"].float() * torch.sigmoid(fused["z"].float())).bfloat16()


@pytest.mark.parametrize("heads", [1, 3, 12])
def test_checkpoint_qdq_and_fresh_rows(heads: int) -> None:
    pool, slots, _, state = _fixture(3, heads)
    initial = torch.ones(3, dtype=torch.bool, device="cuda")
    restored = ops.materialize_checkpoint(pool, slots, initial)
    primary, residual = _read_checkpoint(pool, slots)
    torch.testing.assert_close(restored, primary + residual, atol=2e-7, rtol=2e-6)
    # FP32 division/FMA ordering can choose different FP4 codes right at a
    # quantizer midpoint. Check packing accuracy separately from decoding.
    reference_primary, reference_residual = _qdq(state)
    error = (restored - state).norm()
    reference_error = (reference_primary + reference_residual - state).norm()
    torch.testing.assert_close(error, reference_error, atol=0, rtol=0.01)
    assert error < (primary - state).norm() / 4
    initial[1] = False
    # A fresh row must not read even an invalid/uninitialized checkpoint.
    pool[slots[1].long()] = 255
    restored = ops.materialize_checkpoint(pool, slots, initial)
    assert torch.count_nonzero(restored[1]) == 0
    torch.testing.assert_close(restored[0], primary[0] + residual[0], atol=2e-6, rtol=2e-5)


@pytest.mark.parametrize("heads", [1, 3, 12])
@pytest.mark.parametrize("lower_bound", [None, -5.0])
def test_native_replay_rollover(heads: int, lower_bound: float | None) -> None:
    pool, slots, history, state = _fixture(3, heads)
    replay = _replay_indices(3)
    live = replay.long()
    primary, residual = _read_checkpoint(pool, slots)
    dummy = torch.zeros(3, dtype=torch.bool, device="cuda")
    a_log = torch.full((heads,), -2.0, device="cuda")
    bias = torch.full((heads, 128), -1.0, device="cuda")
    fused = _fused_inputs(3, heads)
    keys, updates, gates = [], [], []
    for step in range(35):
        # Sections of a packed projection have padded batch strides.
        packed = torch.randn((3, 4 * heads * 128), device="cuda", dtype=torch.bfloat16)
        raw = packed[:, : 3 * heads * 128]
        conv, expected_window = _conv_reference(raw, fused, slots)
        q, k, v = (part.reshape(3, heads, 128) for part in conv.split(heads * 128, dim=-1))
        g = torch.randn((3, heads * 128 + 16), device="cuda", dtype=torch.bfloat16)[
            :, : heads * 128
        ]
        beta = torch.randn((3, heads + 8), device="cuda", dtype=torch.bfloat16)[:, :heads]
        qn = (
            q.float() * 128**-0.5 / (q.float().square().sum(-1, keepdim=True) + 1e-6).sqrt()
        ).bfloat16()
        kn = (k.float() / (k.float().square().sum(-1, keepdim=True) + 1e-6).sqrt()).bfloat16()
        raw_g = g.float().reshape(3, heads, 128) + bias
        a = a_log.exp()[None, :, None]
        decay_g = (
            -a * torch.nn.functional.softplus(raw_g)
            if lower_bound is None
            else lower_bound * torch.sigmoid(a * raw_g)
        )
        cumulative = (gates[-1] if gates else 0.0) + decay_g
        pair = torch.stack((kn.float(), qn.float()), -2)
        dynamic, native_cumulative = _checkpoint_operands(
            pair,
            g,
            a_log,
            bias,
            gates[-1] if gates else torch.zeros_like(cumulative),
            lower_bound,
        )
        torch.testing.assert_close(native_cumulative, cumulative, atol=2e-6, rtol=2e-6)
        cumulative = native_cumulative
        checkpoint = primary @ dynamic.transpose(-1, -2)
        if keys:
            adjusted = (
                torch.stack(keys, -2).float()
                * (cumulative[..., None, :] - torch.stack(gates, -2)).exp()
            ).bfloat16()
            coefficients = (adjusted.float() @ pair.transpose(-1, -2)).bfloat16()
            history_pair = torch.stack(updates, -1).float() @ coefficients.float()
        else:
            history_pair = torch.zeros_like(checkpoint)
        update = torch.sigmoid(beta.float())[..., None] * (
            v.float() - checkpoint[..., 0] - history_pair[..., 0]
        )
        expected = (
            checkpoint[..., 1]
            + history_pair[..., 1]
            + update * (kn.float() * qn.float()).sum(-1, keepdim=True)
        ).bfloat16()
        before = pool.clone()
        before_raw = raw.clone()
        actual = ops.replay(
            raw,
            **fused,
            gate=g,
            beta=beta,
            pool=pool,
            slots=slots,
            replay_indices=replay,
            dummy=dummy,
            history=history,
            A_log=a_log,
            dt_bias=bias,
            lower_bound=lower_bound,
            norm_eps=1e-6,
        )
        torch.testing.assert_close(actual, _norm_reference(expected, fused), atol=0.002, rtol=0.02)
        torch.testing.assert_close(raw, before_raw, atol=0, rtol=0)
        torch.testing.assert_close(
            fused["conv_state"][slots.long()], expected_window, atol=0, rtol=0
        )
        # Follow actual BF16 cached updates to isolate replay/commit math
        # from BF16 rounding at near-exact midpoint updates.
        torch.testing.assert_close(history.g[live, :, step % 16], cumulative, atol=2e-6, rtol=2e-6)
        keys.append(history.k[live, :, step % 16].clone())
        updates.append(history.u[live, :, step % 16].clone())
        gates.append(history.g[live, :, step % 16].clone())
        if (step + 1) % 16 == 0:
            cumulative = gates[-1]
            adjusted = (
                torch.stack(keys, -2).float()
                * (cumulative[..., None, :] - torch.stack(gates, -2)).exp()
            ).bfloat16()
            state = (primary + residual) * cumulative.exp()[..., None, :]
            state += torch.stack(updates, -1).float() @ adjusted.float()
            primary, residual = _qdq(state)
            restored = ops.materialize_checkpoint(pool, slots, ~dummy)
            _assert_repacked_close(restored, primary + residual)
            primary, residual = _read_checkpoint(pool, slots)
            keys, updates, gates = [], [], []
        else:
            assert torch.equal(pool, before)
        assert torch.all(history.length[live] == (step + 1) % 16)
        # Replay slots of other requests are untouched.
        untouched = torch.ones(_REPLAY_SLOTS, dtype=torch.bool, device="cuda")
        untouched[live] = False
        assert torch.count_nonzero(history.length[untouched]) == 0


@pytest.mark.parametrize("dspark", [False, True])
def test_dummy_rows_do_not_mutate_checkpoint_or_history(dspark: bool) -> None:
    pool, slots, history, _ = _fixture(1, 3)
    tokens = 4 if dspark else 1
    if dspark:
        history = _verification_history(1, 3, tokens)
    # Dummy rows must not dereference their state or replay slots.
    replay = torch.full((1,), 1 << 30, dtype=torch.int32, device="cuda")
    slots.fill_(-1)
    before = pool.clone()
    fused = _fused_inputs(tokens, 3)
    conv_before = fused["conv_state"].clone()
    x = torch.randn((tokens, 3 * 3 * 128), dtype=torch.bfloat16, device="cuda")
    result = ops.replay(
        x,
        **fused,
        gate=x[:, : 3 * 128],
        beta=x[:, :3],
        pool=pool,
        slots=slots,
        replay_indices=replay,
        dummy=torch.ones(1, dtype=torch.bool, device="cuda"),
        history=history,
        A_log=torch.zeros(3, device="cuda"),
        dt_bias=torch.zeros(3, 128, device="cuda"),
        lower_bound=-5.0,
        norm_eps=1e-6,
    )
    if dspark:
        ops.commit_verified(
            _verification_descriptors(pool, fused["conv_state"], history),
            slots,
            replay,
            torch.ones_like(slots),
            torch.ones(1, device="cuda", dtype=torch.bool),
            3,
            16 + tokens,
            tokens,
        )
    assert torch.count_nonzero(result) == 0
    assert torch.equal(pool, before)
    assert torch.equal(fused["conv_state"], conv_before)
    assert torch.count_nonzero(history.length) == 0


def test_rne_midpoints_and_e4m3_maximum_scale() -> None:
    pool, slots, _, _ = _fixture(1, 1)
    values = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 6.0], device="cuda")
    dense = values.repeat(128 * 128 // 8).reshape(1, 1, 128, 128)
    ops.store_checkpoint(pool, slots, dense)
    primary, _ = _read_checkpoint(pool, slots)
    expected = torch.tensor([0.0, 1.0, 1.0, 2.0, 2.0, 4.0, 4.0, 6.0], device="cuda")
    torch.testing.assert_close(primary[0, 0, 0, :8], expected, atol=0, rtol=0)
    assert torch.isfinite(ops.materialize_checkpoint(pool, slots, torch.ones_like(slots))).all()


@pytest.mark.parametrize("dspark", [False, True])
def test_replay_never_reads_residual(dspark: bool) -> None:
    pool, slots, history, _ = _fixture(1, 3)
    tokens = 4 if dspark else 1
    if dspark:
        history = _verification_history(1, 3, tokens)
    history.u.fill_(float("nan"))
    fused = _fused_inputs(tokens, 3)
    conv_before = fused["conv_state"].clone()
    x = torch.randn((tokens, 3 * 3 * 128), dtype=torch.bfloat16, device="cuda")
    kwargs = dict(
        **fused,
        gate=x[:, : 3 * 128],
        beta=x[:, :3],
        pool=pool,
        slots=slots,
        replay_indices=_replay_indices(1),
        dummy=torch.zeros(1, dtype=torch.bool, device="cuda"),
        history=history,
        A_log=torch.zeros(3, device="cuda"),
        dt_bias=torch.zeros(3, 128, device="cuda"),
        lower_bound=-5.0,
        norm_eps=1e-6,
    )
    expected = ops.replay(x, **kwargs)
    history.length.zero_()
    fused["conv_state"].copy_(conv_before)
    pool[slots.long(), :, :, 8192:16384] = 255
    pool[slots.long(), :, :, 17408:18432] = 127  # NaN E4M3 residual scales
    actual = ops.replay(x, **kwargs)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.parametrize("dspark", [False, True])
def test_runtime_batch_reuses_compilation_and_supports_cuda_graphs(
    monkeypatch: pytest.MonkeyPatch,
    dspark: bool,
) -> None:
    pool, slots, history, _ = _fixture(4, 3)
    from tensorrt_llm._torch.custom_ops import cute_dsl_kda_2fp4_replay as replay_ops

    replay = _replay_indices(4)
    tokens = 4 if dspark else 1
    if dspark:
        history = _verification_history(4, 3, tokens)
    x = torch.randn((4 * tokens, 3 * 3 * 128), dtype=torch.bfloat16, device="cuda")
    fused = _fused_inputs(4 * tokens, 3)
    descriptors = _verification_descriptors(pool, fused["conv_state"], history) if dspark else None
    accepted = torch.ones_like(slots)
    beta = torch.zeros((4 * tokens, 3), dtype=torch.bfloat16, device="cuda")
    dummy = torch.zeros(4, dtype=torch.bool, device="cuda")
    a_log = torch.zeros(3, device="cuda")
    bias = torch.zeros((3, 128), device="cuda")
    compile_kernel = replay_ops.cute.compile
    compilations = []

    def counted_compile(*args, **kwargs):
        compilations.append(1)
        return compile_kernel(*args, **kwargs)

    monkeypatch.setattr(replay_ops.cute, "compile", counted_compile)

    def run(batch: int) -> torch.Tensor:
        output = ops.replay(
            x[: batch * tokens],
            conv_state=fused["conv_state"],
            conv_weight=fused["conv_weight"],
            gate=x[: batch * tokens, : 3 * 128],
            beta=beta[: batch * tokens],
            z=fused["z"][: batch * tokens],
            norm_weight=fused["norm_weight"],
            pool=pool,
            slots=slots[:batch],
            replay_indices=replay[:batch],
            dummy=dummy[:batch],
            history=history,
            A_log=a_log,
            dt_bias=bias,
            lower_bound=-5.0,
            norm_eps=1e-6,
        )
        if dspark:
            ops.commit_verified(
                descriptors,
                slots[:batch],
                replay[:batch],
                accepted[:batch],
                dummy[:batch],
                3,
                16 + tokens,
                tokens,
            )
        return output

    run(4)
    count = len(compilations)
    for batch in [1, 3, 2, 4]:
        run(batch)
        assert len(compilations) == count, "Batch size caused a new replay compilation"
    padded_slots = torch.empty(8, dtype=torch.int32, device="cuda")
    padded_replay = torch.empty(8, dtype=torch.int32, device="cuda")
    padded_dummy = torch.zeros(8, dtype=torch.bool, device="cuda")
    for context_prefix in [1, 2, 3]:
        selected_slots = padded_slots[context_prefix : context_prefix + 4]
        selected_slots.copy_(slots)
        selected_replay = padded_replay[context_prefix : context_prefix + 4]
        selected_replay.copy_(replay)
        selected_dummy = padded_dummy[context_prefix : context_prefix + 4]
        ops.replay(
            x,
            **fused,
            gate=x[:, : 3 * 128],
            beta=beta,
            pool=pool,
            slots=selected_slots,
            replay_indices=selected_replay,
            dummy=selected_dummy,
            history=history,
            A_log=a_log,
            dt_bias=bias,
            lower_bound=-5.0,
            norm_eps=1e-6,
        )
        assert len(compilations) == count, "Mixed-batch metadata caused replay JIT"
    torch.cuda.synchronize()
    history.length.zero_()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = run(3)
    graph.replay()
    graph.replay()
    torch.cuda.synchronize()
    assert torch.all(history.length[replay[:3].long()] == 2)
    assert torch.isfinite(output).all()


def _dense(history: ops.ReplayHistory) -> ops.ReplayHistory:
    return ops.ReplayHistory(
        history.k.contiguous(),
        history.u.contiguous(),
        history.g.contiguous(),
        history.length.contiguous(),
        history.candidate_x,
        history.old_lengths,
        history.arrivals,
    )


@pytest.mark.parametrize("dspark", [False, True])
def test_paged_and_dense_histories_match(dspark: bool) -> None:
    """Page-strided replay slots and dense histories give bit-identical results,
    including a window commit (decode) and an accepted-prefix rollover (DSpark)."""
    batch, heads = 3, 3
    tokens = 4 if dspark else 1
    capacity = ops.HISTORY_SIZE + (tokens if dspark else 0)
    replay = _replay_indices(batch)
    lengths = torch.tensor([12, 7, 15], dtype=torch.int32, device="cuda")
    dummy = torch.tensor([False, True, False], device="cuda")
    results = []
    for paged in (True, False):
        pool, slots, _, _ = _fixture(batch, heads)
        torch.manual_seed(7)
        history = _paged_history(heads, capacity, batch=batch, tokens=tokens if dspark else 0)
        history.k.copy_(torch.randn(history.k.shape, device="cuda") * 0.03)
        history.u.copy_(torch.randn(history.u.shape, device="cuda") * 0.03)
        history.g.copy_(-torch.rand(history.g.shape, device="cuda").cumsum(2) * 0.02)
        history.length[replay.long()] = lengths
        if not paged:
            history = _dense(history)
        fused = _fused_inputs(batch * tokens, heads)
        x = torch.randn((batch * tokens, 3 * heads * 128), dtype=torch.bfloat16, device="cuda")
        kwargs = dict(
            gate=torch.randn((batch * tokens, heads * 128), dtype=torch.bfloat16, device="cuda"),
            beta=torch.randn((batch * tokens, heads), dtype=torch.bfloat16, device="cuda"),
            pool=pool,
            slots=slots,
            replay_indices=replay,
            dummy=dummy,
            history=history,
            A_log=torch.full((heads,), -2.0, device="cuda"),
            dt_bias=torch.full((heads, 128), -1.0, device="cuda"),
            lower_bound=-5.0,
            norm_eps=1e-6,
        )
        outputs = [ops.replay(x, **fused, **kwargs)]
        if dspark:
            accepted = torch.tensor([2, 4, 4], dtype=torch.int32, device="cuda")
            ops.commit_verified(
                _verification_descriptors(pool, fused["conv_state"], history),
                slots,
                replay,
                accepted,
                dummy,
                heads,
                capacity,
                tokens,
            )
        else:
            outputs.append(ops.replay(x, **fused, **kwargs))
        torch.cuda.synchronize()
        results.append(
            [*outputs, pool, fused["conv_state"]]
            + [t.contiguous() for t in (history.k, history.u, history.g, history.length)]
        )
    for paged_result, dense_result in zip(*results, strict=True):
        assert torch.equal(paged_result, dense_result)
