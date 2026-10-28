# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native 2FP4 replay, QDQ prefill boundaries, and indexed checkpoint tests."""

from collections.abc import Iterator
from types import SimpleNamespace
from unittest.mock import patch

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


def _fixture(batch: int, heads: int) -> tuple:
    torch.manual_seed(123)
    # Dense head payloads but noncontiguous V2 slots, with live-history rows
    # deliberately unrelated to physical slot ids.
    backing = torch.zeros(
        (22, heads, 1, ops.STATE_BYTES_PER_HEAD), dtype=torch.uint8, device="cuda"
    )
    pool = backing[::2]
    slots = torch.tensor([9, 1, 7, 3][:batch], dtype=torch.int32, device="cuda")
    slot_map = torch.full((11,), -1, dtype=torch.int32, device="cuda")
    slot_map[slots.long()] = torch.arange(batch - 1, -1, -1, device="cuda", dtype=torch.int32)
    shape = (batch, heads, 16, 128)
    history = ops.ReplayHistory(
        torch.empty(shape, dtype=torch.bfloat16, device="cuda"),
        torch.empty(shape, dtype=torch.bfloat16, device="cuda"),
        torch.empty(shape, dtype=torch.float32, device="cuda"),
        torch.zeros(batch, dtype=torch.int32, device="cuda"),
        slot_map,
    )
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


def _verification_history(history: ops.ReplayHistory, tokens: int) -> ops.ReplayHistory:
    rows, heads, _, dim = history.k.shape
    shape = (rows, heads, 16 + tokens, dim)
    return ops.ReplayHistory(
        history.k.new_empty(shape),
        history.u.new_empty(shape),
        history.g.new_empty(shape),
        history.length,
        history.slot_map,
        history.k.new_empty((rows, tokens, 3 * heads * dim)),
        history.length.new_empty(rows),
    )


def _verification_descriptors(
    pool: torch.Tensor, conv: torch.Tensor, history: ops.ReplayHistory
) -> torch.Tensor:
    return torch.tensor(
        [
            [
                pool.data_ptr(),
                pool.stride(0),
                conv.data_ptr(),
                conv.stride(0),
                history.k.data_ptr(),
                history.u.data_ptr(),
                history.g.data_ptr(),
                history.length.data_ptr(),
                history.old_lengths.data_ptr(),
                history.candidate_x.data_ptr(),
            ]
        ],
        dtype=torch.int64,
        device="cuda",
    )


@pytest.mark.parametrize(("heads", "tokens"), [(1, 1), (3, 4), (3, 8), (12, 7)])
def test_dspark_native_verify_acceptance_and_rollover(heads: int, tokens: int) -> None:
    """Only accepted candidates advance Conv/state; rejected suffixes are ignored."""
    batch, capacity = 3, 16 + tokens
    pool, slots, original, _ = _fixture(batch, heads)
    shape = (batch, heads, capacity, 128)
    live = original.slot_map[slots.long()].long()
    lengths = torch.tensor([11, 15, 3], device="cuda", dtype=torch.int32)
    history = ops.ReplayHistory(
        torch.randn(shape, device="cuda").bfloat16() * 0.03,
        torch.randn(shape, device="cuda").bfloat16() * 0.03,
        -torch.rand(shape, device="cuda").cumsum(2) * 0.02,
        original.length,
        original.slot_map,
        torch.empty(batch, tokens, 3 * heads * 128, device="cuda", dtype=torch.bfloat16),
        torch.empty(batch, device="cuda", dtype=torch.int32),
    )
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
    actual = ops.verify(
        raw,
        **fused,
        gate=gate,
        beta=beta,
        pool=pool,
        slots=slots,
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
    ops.commit_verified(
        descriptor, slots, history.slot_map, accepted, dummy, heads, capacity, tokens
    )
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
    scalar_history = ops.ReplayHistory(
        history.k.clone(),
        history.u.clone(),
        history.g.clone(),
        history.length.clone(),
        history.slot_map,
    )
    next_raw = raw.reshape(batch, tokens, -1)[:, 0]
    kwargs = dict(
        conv_weight=fused["conv_weight"],
        gate=gate.reshape(batch, tokens, -1)[:, 0],
        beta=beta.reshape(batch, tokens, heads)[:, 0],
        z=fused["z"].reshape(batch, tokens, heads, 128)[:, 0],
        norm_weight=fused["norm_weight"],
        slots=slots,
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
        history.slot_map,
        torch.ones_like(accepted),
        dummy,
        heads,
        capacity,
        tokens,
    )
    torch.testing.assert_close(history.length, scalar_history.length, rtol=0, atol=0)
    torch.testing.assert_close(fused["conv_state"], scalar_conv, rtol=0, atol=0)


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
def test_native_replay_rollover_and_partial_flush(heads: int, lower_bound: float | None) -> None:
    pool, slots, history, state = _fixture(3, heads)
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
        live = history.slot_map[slots.long()].long()
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
        assert torch.all(history.length == (step + 1) % 16)
    ops.commit_history(pool, slots, history)
    assert torch.count_nonzero(history.length) == 0
    restored = ops.materialize_checkpoint(pool, slots, ~dummy)
    cumulative = gates[-1]
    adjusted = (
        torch.stack(keys, -2).float() * (cumulative[..., None, :] - torch.stack(gates, -2)).exp()
    ).bfloat16()
    state = (primary + residual) * cumulative.exp()[..., None, :]
    state += torch.stack(updates, -1).float() @ adjusted.float()
    primary, residual = _qdq(state)
    _assert_repacked_close(restored, primary + residual)


@pytest.mark.parametrize("dspark", [False, True])
def test_dummy_rows_do_not_mutate_checkpoint_or_history(dspark: bool) -> None:
    pool, slots, history, _ = _fixture(1, 3)
    tokens = 4 if dspark else 1
    if dspark:
        history = _verification_history(history, tokens)
    history.slot_map.fill_(-1)
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
            history.slot_map,
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
        history = _verification_history(history, tokens)
    fused = _fused_inputs(tokens, 3)
    conv_before = fused["conv_state"].clone()
    x = torch.randn((tokens, 3 * 3 * 128), dtype=torch.bfloat16, device="cuda")
    kwargs = dict(
        **fused,
        gate=x[:, : 3 * 128],
        beta=x[:, :3],
        pool=pool,
        slots=slots,
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


@pytest.mark.parametrize("prefill_only", [False, True])
@pytest.mark.parametrize("dspark", [False, True])
def test_v2_owns_checkpoint_and_only_live_requests_own_history(
    prefill_only: bool, dspark: bool
) -> None:
    from tensorrt_llm._torch.disaggregation.resource.kv_extractor import (
        build_page_table_from_manager,
    )
    from tensorrt_llm._torch.disaggregation.resource.page import MambaLayerGroup
    from tensorrt_llm._torch.pyexecutor.kv_cache.mamba_cache_manager import (
        MambaHybridCacheManagerV2,
        MambaRole,
    )
    from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest, SamplingConfig
    from tensorrt_llm._torch.pyexecutor.resource_manager import CacheTypeCpp
    from tensorrt_llm.llmapi.llm_args import DSparkDecodingConfig, KvCacheConfig
    from tensorrt_llm.mapping import Mapping

    config = KvCacheConfig(
        max_tokens=512,
        max_gpu_total_bytes=16 << 20,
        host_cache_size=0,
        mamba_state_config={
            "kda_2fp4_replay": True,
            "kda_2fp4_prefill_only": prefill_only,
        },
    )
    manager = MambaHybridCacheManagerV2(
        mamba_d_state=128,
        mamba_d_conv=4,
        mamba_num_heads=3,
        mamba_n_groups=3,
        mamba_head_dim=128,
        mamba_num_layers=2,
        mamba_layer_mask=[True, True, False],
        mamba_cache_dtype=torch.bfloat16,
        mamba_ssm_cache_dtype=torch.float32,
        kv_cache_config=config,
        kv_cache_type=CacheTypeCpp.SELFKONLY,
        num_layers=1,
        layer_mask=[False, False, True],
        num_kv_heads=1,
        head_dim=64,
        tokens_per_block=32,
        max_seq_len=128,
        max_batch_size=4,
        mapping=Mapping(),
        conv_state_layout="q_k_v",
        spec_config=DSparkDecodingConfig(max_draft_len=3) if dspark else None,
    )
    request = LlmRequest(
        request_id=123,
        max_new_tokens=16,
        input_tokens=[1, 2, 3, 4],
        sampling_config=SamplingConfig(),
        is_streaming=False,
    )
    try:
        assert manager.ssm_bytes == 3 * ops.STATE_BYTES_PER_HEAD
        assert all(state.dtype == torch.uint8 for state in manager.all_ssm_states)
        assert bool(manager._kda_2fp4_buffers) is not prefill_only
        page_table = build_page_table_from_manager(manager)
        state_views = [
            view
            for group in page_table.layer_groups
            if isinstance(group, MambaLayerGroup)
            for view in group.pool_views
            if MambaRole.SSM_STATE in view.pool_role
        ]
        assert state_views
        assert all(view.bytes_per_head == ops.STATE_BYTES_PER_HEAD for view in state_views)
        if prefill_only:
            assert manager._kda_2fp4_slot_map is None
            assert manager.mamba_layer_cache(0).history is None
            assert manager.intermediate_conv_states is None
            # A context-only rank can still receive a synthetic DP decode
            # row, without owning any accepted-history/commit scratch.
            manager.update_mamba_states(
                SimpleNamespace(num_seqs=1, num_contexts=0),
                torch.ones(1, device="cuda", dtype=torch.int32),
                state_indices=torch.zeros(1, device="cuda", dtype=torch.int32),
            )
            return
        assert manager._kda_2fp4_buffers[0][0].shape[0] == 5  # B + persistent padding slot
        assert manager._kda_2fp4_buffers[0][0].shape[2] == (20 if dspark else 16)
        assert manager.intermediate_ssm_states is None
        if dspark:
            assert manager.intermediate_conv_states.shape == (2, 4, 4, 3 * 3 * 128)
        assert manager.prepare_context(request)
        assert manager.resize_context(request, num_tokens=4)
        torch.cuda.synchronize()
        previous_mode = torch.cuda.get_sync_debug_mode()
        torch.cuda.set_sync_debug_mode("error")
        try:
            manager._setup_state_indices([request], num_contexts=1)
        finally:
            torch.cuda.set_sync_debug_mode(previous_mode)
        slots = manager.cuda_state_indices[:1].clone()
        x = torch.randn((1, 3 * 3 * 128), dtype=torch.bfloat16, device="cuda")
        dummy = torch.zeros(1, dtype=torch.bool, device="cuda")
        expected_states, expected_conv, layer_inputs = {}, {}, {}
        for layer in manager.mamba_pp_layers:
            cache = manager.mamba_layer_cache(layer)
            ops.store_checkpoint(
                cache.temporal, slots, torch.randn((1, 3, 128, 128), device="cuda") * 0.1
            )
            fused = _fused_inputs(1, 3, cache.conv.shape[0])
            fused["conv_state"] = cache.conv
            layer_inputs[layer] = fused
        for _ in range(3):
            for layer in manager.mamba_pp_layers:
                cache = manager.mamba_layer_cache(layer)
                ops.replay(
                    x,
                    **layer_inputs[layer],
                    gate=x[:, : 3 * 128],
                    beta=torch.zeros((1, 3), dtype=torch.bfloat16, device="cuda"),
                    pool=cache.temporal,
                    slots=slots,
                    dummy=dummy,
                    history=cache.history,
                    A_log=torch.zeros(3, device="cuda"),
                    dt_bias=torch.zeros(3, 128, device="cuda"),
                    lower_bound=-5.0,
                    norm_eps=1e-6,
                )
            if dspark:
                manager.update_mamba_states(
                    SimpleNamespace(num_seqs=1, num_contexts=0),
                    torch.ones(1, device="cuda", dtype=torch.int32),
                    state_indices=slots,
                )
        for layer in manager.mamba_pp_layers:
            cache = manager.mamba_layer_cache(layer)
            assert cache.history.length.sum() == 3
            primary, residual = _read_checkpoint(cache.temporal, slots)
            live = cache.history.slot_map[slots.long()].long()
            gates = cache.history.g[live, :, :3]
            cumulative = gates[:, :, -1]
            adjusted = (
                cache.history.k[live, :, :3].float() * (cumulative[:, :, None, :] - gates).exp()
            ).bfloat16()
            state = (primary + residual) * cumulative.exp()[..., None, :]
            state += cache.history.u[live, :, :3].float().transpose(-1, -2) @ adjusted.float()
            primary, residual = _qdq(state)
            expected_states[layer] = primary + residual
            expected_conv[layer] = cache.conv[slots.long()].clone()
        from tensorrt_llm._torch.custom_ops import cute_dsl_kda_2fp4_commit as commit_ops

        flush_stream = torch.cuda.Stream()
        with patch.object(commit_ops.cute, "compile", side_effect=AssertionError("live flush JIT")):
            with torch.cuda.stream(flush_stream):
                manager.suspend_request(request)
        torch.cuda.current_stream().wait_stream(flush_stream)
        assert not manager._kda_2fp4_requests
        for buffers in manager._kda_2fp4_buffers:
            assert torch.count_nonzero(buffers[3]) == 0
        for layer in manager.mamba_pp_layers:
            cache = manager.mamba_layer_cache(layer)
            restored = ops.materialize_checkpoint(cache.temporal, slots, ~dummy)
            _assert_repacked_close(restored, expected_states[layer])
            torch.testing.assert_close(
                cache.conv[slots.long()], expected_conv[layer], atol=0, rtol=0
            )
        assert manager.resume_request(request)
        manager._setup_state_indices([request])
        assert manager.mamba_layer_cache(0).history.length.sum() == 0
        for layer in manager.mamba_pp_layers:
            cache = manager.mamba_layer_cache(layer)
            resumed_slots = manager.cuda_state_indices[:1]
            restored = ops.materialize_checkpoint(cache.temporal, resumed_slots, ~dummy)
            _assert_repacked_close(restored, expected_states[layer])
            torch.testing.assert_close(
                cache.conv[resumed_slots.long()], expected_conv[layer], atol=0, rtol=0
            )
        manager.free_resources(request)
        assert not manager._kda_2fp4_requests
        assert len(manager._kda_2fp4_free_rows) == 5
    finally:
        manager.shutdown()


@torch.no_grad()
def test_model_prefill_qdq_then_native_decode() -> None:
    from tensorrt_llm._torch.modules.kimi_kda import KimiKDALinearAttention
    from tensorrt_llm._torch.pyexecutor.kv_cache.mamba_cache_manager import Kda2Fp4State

    torch.manual_seed(123)
    heads, batch = 3, 2
    cfg = SimpleNamespace(
        hidden_size=512,
        rms_norm_eps=1e-6,
        linear_attn_config={
            "num_heads": heads,
            "head_dim": 128,
            "short_conv_kernel_size": 4,
            "use_full_rank_gate": True,
            "gate_lower_bound": -5.0,
        },
    )
    layer = KimiKDALinearAttention(cfg, layer_idx=0).cuda()
    torch.nn.init.normal_(layer.dt_bias, std=0.1)
    layer.finalize_decode_weights()
    pool, slots, history, _ = _fixture(batch, heads)
    conv = torch.randn((11, 3 * heads * 128, 3), dtype=torch.bfloat16, device="cuda") * 0.01
    flags = torch.tensor([True, False], dtype=torch.bool, device="cuda")
    metadata = SimpleNamespace(
        use_initial_states=True,
        has_initial_states=flags,
        kda_chunk_indices=torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], device="cuda"),
        kda_varlen_is_aligned=True,
        kda_single_sequence_length=None,
        query_start_loc=torch.tensor([0, 128, 256], dtype=torch.int32, device="cuda"),
        state_indices=slots,
        _arange_buffer=torch.arange(3, dtype=torch.int32, device="cuda"),
    )
    dense = torch.zeros((11, heads, 128, 128), device="cuda")
    dense[slots.long()] = ops.materialize_checkpoint(pool, slots, flags)
    reference_conv = conv.clone()
    cache = Kda2Fp4State(conv=conv, temporal=pool, history=history)
    x = torch.randn((256, 512), dtype=torch.bfloat16, device="cuda") * 0.1
    reference = layer.forward_prefill(
        x, metadata.query_start_loc.long(), metadata, batch, reference_conv, dense, slots
    )
    actual = layer.forward_prefill(
        x, metadata.query_start_loc.long(), metadata, batch, conv, pool, slots, cache
    )
    torch.testing.assert_close(actual, reference, atol=0, rtol=0)
    torch.testing.assert_close(conv, reference_conv, atol=0, rtol=0)
    primary, residual = _qdq(dense[slots.long()])
    restored = ops.materialize_checkpoint(pool, slots, torch.ones_like(flags))
    torch.testing.assert_close(restored, primary + residual, atol=0.002, rtol=0.02)

    # A later scheduler chunk starts from the persistent quantized checkpoint,
    # not the unquantized temporary final state of the previous call.
    dense[slots.long()] = restored
    reference_conv = conv.clone()
    metadata.has_initial_states = torch.ones_like(flags)
    metadata.query_start_loc = torch.tensor([0, 64, 128], dtype=torch.int32, device="cuda")
    metadata.kda_chunk_indices = torch.tensor([[0, 0], [1, 0]], device="cuda")
    x = torch.randn((128, 512), dtype=torch.bfloat16, device="cuda") * 0.1
    reference = layer.forward_prefill(
        x, metadata.query_start_loc.long(), metadata, batch, reference_conv, dense, slots
    )
    actual = layer.forward_prefill(
        x, metadata.query_start_loc.long(), metadata, batch, conv, pool, slots, cache
    )
    torch.testing.assert_close(actual, reference, atol=0, rtol=0)
    torch.testing.assert_close(conv, reference_conv, atol=0, rtol=0)
    primary, residual = _qdq(dense[slots.long()])
    restored = ops.materialize_checkpoint(pool, slots, torch.ones_like(flags))
    torch.testing.assert_close(restored, primary + residual, atol=0.002, rtol=0.02)
    assert torch.count_nonzero(history.length) == 0

    for _ in range(17):
        x = torch.randn((batch, 512), dtype=torch.bfloat16, device="cuda") * 0.1
        output = torch.empty((batch, heads, 128), dtype=torch.bfloat16, device="cuda")
        actual = layer.forward_decode(
            x,
            conv,
            pool,
            slots,
            metadata,
            cache,
            output=output,
            dummy_mask=torch.zeros(batch, dtype=torch.bool, device="cuda"),
        )
        assert actual.data_ptr() == output.data_ptr()
        assert torch.isfinite(actual).all()
    assert torch.all(history.length == 1)

    # Exercise the production controller with a context prefix and an
    # unaligned one-token generation suffix, including its BCG output view.
    history.length[history.slot_map[slots[0].long()].long()] = 0
    metadata.has_initial_states = torch.zeros(1, dtype=torch.bool, device="cuda")
    metadata.use_initial_states = False
    metadata.query_start_loc_long = torch.tensor([0, 256], device="cuda")
    metadata.query_start_loc = metadata.query_start_loc_long.int()
    metadata.kda_chunk_indices = torch.tensor([[0, i] for i in range(4)], device="cuda")
    metadata.kda_single_sequence_length = 256
    manager = SimpleNamespace(
        mamba_layer_cache=lambda layer_idx: cache,
        _dummy_request_mask=torch.zeros(2, dtype=torch.bool, device="cuda"),
    )
    attention = SimpleNamespace(
        mamba_metadata=metadata,
        num_contexts=1,
        num_ctx_tokens=256,
        num_tokens=257,
        seq_lens=torch.tensor([256, 1]),
        kv_cache_manager=manager,
    )
    hidden = torch.randn((257, 512), dtype=torch.bfloat16, device="cuda") * 0.1
    output = torch.empty((257, heads, 128), dtype=torch.bfloat16, device="cuda")
    actual = layer._forward_impl(hidden, attention, output=output)
    assert actual.data_ptr() == output.data_ptr()
    assert torch.isfinite(actual).all()
    assert history.length[history.slot_map[slots[1].long()].long()] == 2

    # DSpark verification (including a zero-draft T=1 round) uses the
    # packed-state controller instead of the dense legacy MTP backend.
    history = _verification_history(history, 8)
    history.length.zero_()
    cache = Kda2Fp4State(conv=conv, temporal=pool, history=history)
    attention.num_contexts = 0
    attention.num_ctx_tokens = 0
    attention.seq_lens = torch.ones(batch, dtype=torch.int32)
    dummy = torch.zeros(batch, device="cuda", dtype=torch.bool)
    descriptors = _verification_descriptors(pool, conv, history)
    for tokens in (4, 1):
        attention.num_tokens = batch * tokens
        hidden = torch.randn((batch * tokens, 512), device="cuda", dtype=torch.bfloat16) * 0.1
        before_pool, before_conv, before_lengths = (
            pool.clone(),
            conv.clone(),
            history.length.clone(),
        )
        actual = layer._forward_impl(hidden, attention)
        assert actual.shape == (batch * tokens, heads, 128)
        assert torch.isfinite(actual).all()
        torch.testing.assert_close(pool, before_pool, rtol=0, atol=0)
        torch.testing.assert_close(conv, before_conv, rtol=0, atol=0)
        torch.testing.assert_close(history.length, before_lengths, rtol=0, atol=0)
        ops.commit_verified(
            descriptors, slots, history.slot_map, torch.ones_like(slots), dummy, heads, 24, 8
        )
    assert torch.all(history.length == 2)


@pytest.mark.parametrize("dspark", [False, True])
def test_runtime_batch_reuses_compilation_and_supports_cuda_graphs(
    monkeypatch: pytest.MonkeyPatch,
    dspark: bool,
) -> None:
    pool, slots, history, _ = _fixture(4, 3)
    from tensorrt_llm._torch.custom_ops import cute_dsl_kda_2fp4_replay as replay_ops

    tokens = 4 if dspark else 1
    if dspark:
        history = _verification_history(history, tokens)
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
                history.slot_map,
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
    padded_dummy = torch.zeros(8, dtype=torch.bool, device="cuda")
    for context_prefix in [1, 2, 3]:
        selected_slots = padded_slots[context_prefix : context_prefix + 4]
        selected_slots.copy_(slots)
        selected_dummy = padded_dummy[context_prefix : context_prefix + 4]
        ops.replay(
            x,
            **fused,
            gate=x[:, : 3 * 128],
            beta=beta,
            pool=pool,
            slots=selected_slots,
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
    live = history.slot_map[slots[:3].long()].long()
    assert torch.all(history.length[live] == 2)
    assert torch.isfinite(output).all()
