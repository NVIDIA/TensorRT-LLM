# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CUDA tests of the DFlash draft context cache in FP8 (E4M3 codes with per-layer scales).

The drafter writes ``e4m3(clamp(x / s, +-448))`` through ``modeling_dflash.dflash_store_fp8_kv``
(the worker's context writes and the block append alike) and its non-causal windowed attention
dequantizes what it gathers once (``code * s``) before the BF16 SDPA. The DFlash 2 Beam drafter
has five such layers, a 4096-token window, eight-row blocks and 10 query / 2 KV heads of 128
per TP4 rank on 32-token pages.

What is checked, against references that never call the code under test:

* stored bytes equal an independent table-based E4M3 encoder (round to nearest, ties to even,
  saturation at +-448) for non-unit scales, signed zeros, subnormals, ties and out-of-range
  values, at the pages and slots the block table names; the worker's context write puts each
  layer's codes in that layer's pool with that layer's scale;
* the windowed read over an FP8 pool against explicit FP32 masked attention over the
  independently decoded values, at the 4095/4096/4097 window boundary, within the approved
  read-path bounds (relative RMS 0.00388, worst row 0.0129);
* fault controls: an inverted, omitted, doubled or other layer's scale on the read exceeds
  those bounds; poisoned stale-tail and foreign pages contribute nothing (bitwise), while a
  corrupted block table that addresses them is detected;
* the drafter's FP8 capability and scale configuration refuse what they cannot serve.

Needs one CUDA device; no model files.
"""

from __future__ import annotations

import types

import pytest
import torch
from torch import nn

from tensorrt_llm._torch.models.modeling_dflash import (
    DFlashForCausalLM,
    dflash_store_fp8_kv,
    dflash_window_gather,
    dflash_windowed_block_attention,
)
from tensorrt_llm._torch.speculative.dflash import DFlashWorker
from tensorrt_llm.quantization.mode import QuantAlgo

WINDOW = 4096
BLOCK = 8
PAGE = 32
HQ, HKV, D = 10, 2, 128
SCALE = D**-0.5
#: Approved read-path bounds of the drafter against its FP32 oracle (F3).
REL_RMS_BOUND = 0.00388
WORST_ROW_BOUND = 0.0129
#: Scales of the five Beam draft layers, and further non-unit scales.
DRAFT_SCALES = (2.0, 2.0, 2.0, 8.0, 4.0)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")


# ---------------------------------------------------------------------------------------------
# Independent E4M3 codec: every finite code's value, nearest with ties to even
# ---------------------------------------------------------------------------------------------


def _e4m3_values() -> torch.Tensor:
    """Value of each code 0..126 (positive, finite), from the bit fields: exponent bias 7."""
    values = []
    for code in range(127):
        exponent, mantissa = code >> 3, code & 7
        if exponent == 0:
            values.append(mantissa / 8.0 * 2.0**-6)
        else:
            values.append((1.0 + mantissa / 8.0) * 2.0 ** (exponent - 7))
    return torch.tensor(values, dtype=torch.float64)


_VALUES = _e4m3_values()


def encode(x: torch.Tensor, scale: float) -> torch.Tensor:
    """uint8 E4M3 codes of ``fp32(x) * fp32(1/scale)``, saturated to +-448, ties to even."""
    y = (x.float() * (1.0 / scale)).double().cpu()
    mag = y.abs().clamp(max=448.0)
    values = _VALUES
    hi = torch.searchsorted(values, mag.flatten()).clamp(max=126)
    lo = (hi - 1).clamp(min=0)
    d_hi = (values[hi] - mag.flatten()).abs()
    d_lo = (mag.flatten() - values[lo]).abs()
    pick_lo = (d_lo < d_hi) | ((d_lo == d_hi) & (lo % 2 == 0))
    code = torch.where(pick_lo, lo, hi).view(mag.shape)
    sign = (y < 0) | ((y == 0) & torch.signbit(y))
    return (code | (sign.long() << 7)).to(torch.uint8).to(x.device)


def decode(codes: torch.Tensor, scale: float) -> torch.Tensor:
    """FP32 values ``value(code) * scale`` of uint8 E4M3 codes."""
    c = codes.long().cpu()
    mag = _VALUES[(c & 0x7F).clamp(max=126)]
    value = torch.where((c & 0x80) != 0, -mag, mag)
    return (value * scale).float().to(codes.device)


def test_independent_codec_matches_the_e4m3_definition():
    """The codec's anchors: 448 max, 2^-9 smallest subnormal, ties to even, saturation."""
    x = torch.tensor([448.0, 464.0, 1000.0, 2.0**-9, 2.0**-10, 1.0625, 1.1875, -0.0, -3.0])
    codes = encode(x, 1.0).tolist()
    assert codes[0] == codes[1] == codes[2] == 0x7E
    assert codes[3] == 1 and codes[4] == 0  # 2^-10 is the tie between 0 and 2^-9: even 0
    assert codes[5] == 0x38 and codes[6] == 0x3A  # 1.0625 -> 1.0, 1.1875 -> 1.25 (even)
    assert codes[7] == 0x80
    assert decode(encode(x, 1.0), 1.0).tolist()[8] == -3.0


# ---------------------------------------------------------------------------------------------
# Pool fixtures
# ---------------------------------------------------------------------------------------------


def _fp8_pool(ctx_lens, poison: int = 0):
    """An FP8 HND pool with interleaved per-request pages (see the BF16 windowed tests)."""
    rows = len(ctx_lens)
    pages_per_row = (max(ctx_lens) + BLOCK + PAGE - 1) // PAGE + 1
    total = rows * pages_per_row
    pool = torch.full((total, 2, HKV, PAGE, D), poison, dtype=torch.uint8, device="cuda")
    tables = torch.arange(total, dtype=torch.int32, device="cuda").view(pages_per_row, rows).t()
    return pool.view(torch.float8_e4m3fn), tables.contiguous()


def _place(pool, tables, row, positions, k_codes, v_codes):
    raw = pool.view(torch.uint8)
    for i, p in enumerate(positions):
        page = int(tables[row, p // PAGE])
        raw[page, 0, :, p % PAGE] = k_codes[i]
        raw[page, 1, :, p % PAGE] = v_codes[i]


def _oracle_keys(ctx: int, j: int) -> list:
    q = ctx + j
    return [k for k in range(0, ctx + BLOCK) if abs(q - k) <= WINDOW - 1]


def _fp32_attention(q, k_vals, v_vals, ctx):
    """[BLOCK, HQ, D] FP32 masked softmax attention over decoded keys at positions 0..ctx+7."""
    out = torch.empty(BLOCK, HQ, D, dtype=torch.float32, device=q.device)
    group = HQ // HKV
    for j in range(BLOCK):
        keys = torch.tensor(_oracle_keys(ctx, j), device=q.device)
        kj, vj = k_vals[keys], v_vals[keys]  # [n, HKV, D]
        for h in range(HQ):
            logits = (kj[:, h // group].double() @ q[j, h].double()) * SCALE
            p = torch.softmax(logits, dim=0)
            out[j, h] = (p[:, None] * vj[:, h // group].double()).sum(0).float()
    return out


def _metrics(out, ref):
    o, r = out.double().reshape(-1, D), ref.double().reshape(-1, D)
    rel_rms = float((o - r).norm() / r.norm())
    worst = float(((o - r).abs().amax(1) / r.abs().amax(1).clamp_min(1e-30)).max())
    return rel_rms, worst


def _case(ctx: int, scale: float, seed: int):
    """One request at context ``ctx``: random K/V (unit-ish magnitudes x scale x 8), encoded
    independently and placed in an FP8 pool; returns the inputs of the read and its oracle."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    n = ctx + BLOCK
    k = (torch.randn(n, HKV, D, generator=g) * 8 * scale).to(torch.bfloat16).cuda()
    v = (torch.randn(n, HKV, D, generator=g) * 8 * scale).to(torch.bfloat16).cuda()
    q = (torch.randn(1, BLOCK, HQ, D, generator=g) * 0.25).to(torch.bfloat16).cuda()
    kc, vc = encode(k, scale), encode(v, scale)
    pool, tables = _fp8_pool([ctx])
    _place(pool, tables, 0, range(n), kc, vc)
    gather = dflash_window_gather(tables, torch.tensor([ctx], device="cuda"), BLOCK, WINDOW, PAGE)
    ref = _fp32_attention(q[0].float(), decode(kc, scale), decode(vc, scale), ctx)
    return q, pool, tables, gather, kc, vc, ref


# ---------------------------------------------------------------------------------------------
# Writes
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("scale", [2.0, 8.0, 4.0, 0.5, 1.5])
def test_store_writes_the_independent_codes_at_the_addressed_slots(scale):
    torch.manual_seed(int(scale * 10))
    n = 70
    x = (torch.randn(n, HKV, D, device="cuda") * 64 * scale).to(torch.bfloat16)
    specials = torch.tensor(
        [448.0, -448.0, 464.0, 500.0, -1000.0, 0.0, -0.0, 2.0**-9, 2.0**-10, 1.0625, 1.1875],
        device="cuda",
    )
    x[: len(specials), 0, 0] = (specials * scale).to(torch.bfloat16)
    y = torch.flip(x, dims=[0]).contiguous()
    pool, tables = _fp8_pool([n])
    positions = torch.randperm(n, device="cuda")
    pages = tables[0].long()[positions // PAGE]
    dflash_store_fp8_kv(pool, pages, positions % PAGE, x, y, scale)
    raw = pool.view(torch.uint8)
    got_k = raw[pages, 0, :, positions % PAGE]
    got_v = raw[pages, 1, :, positions % PAGE]
    assert torch.equal(got_k, encode(x, scale)), "K codes differ from the independent encoder"
    assert torch.equal(got_v, encode(y, scale)), "V codes differ from the independent encoder"
    written = torch.zeros(pool.shape[0], PAGE, dtype=torch.bool, device="cuda")
    written[pages, positions % PAGE] = True
    untouched = raw.permute(0, 3, 1, 2, 4)[~written]
    assert int(untouched.count_nonzero()) == 0, "a slot outside the addressed ones was written"
    assert int(((got_k & 0x7F) == 0x7F).sum() + ((got_v & 0x7F) == 0x7F).sum()) == 0


def _bare_worker(tables, pool_layers, scales):
    """The worker state ``_store_context_kv_paged`` reads, bound to a managed FP8 pool."""
    worker = DFlashWorker.__new__(DFlashWorker)
    nn.Module.__init__(worker)
    worker._ctx_block_tables = tables
    worker._ctx_page_table = None
    worker._ctx_page_size = PAGE
    worker._ctx_kv_buf = pool_layers
    worker._ctx_kv_scales = tuple(scales)
    worker._ctx_paged_append = None
    return worker


def test_worker_context_write_quantizes_each_layer_with_its_scale():
    """``DFlashWorker._store_context_kv_paged`` over an FP8 pool: layer l's codes, scaled by
    layer l's scale, at the page of (block-table row, position) of every row."""
    torch.manual_seed(5)
    layers = len(DRAFT_SCALES)
    n = 40
    _, tables = _fp8_pool([64, 64])
    pool_layers = [_fp8_pool([64, 64])[0] for _ in range(layers)]
    worker = _bare_worker(tables, pool_layers, DRAFT_SCALES)
    k = (torch.randn(n, layers, HKV, D, device="cuda") * 100).to(torch.bfloat16)
    v = (torch.randn(n, layers, HKV, D, device="cuda") * 100).to(torch.bfloat16)
    rows = torch.tensor([0] * 20 + [1] * 20, device="cuda")
    positions = torch.cat([torch.arange(20), torch.arange(30, 50)]).cuda()
    worker._store_context_kv_paged(k, v, rows, positions)
    for layer, scale in enumerate(DRAFT_SCALES):
        raw = pool_layers[layer].view(torch.uint8)
        pages = tables[rows, positions // PAGE].long()
        got_k = raw[pages, 0, :, positions % PAGE]
        got_v = raw[pages, 1, :, positions % PAGE]
        assert torch.equal(got_k, encode(k[:, layer], scale)), f"layer {layer} K"
        assert torch.equal(got_v, encode(v[:, layer], scale)), f"layer {layer} V"
    # Wrong-layer ownership would differ: layers 3 (s=8) and 4 (s=4) codes are not layer 0's.
    raw3 = pool_layers[3].view(torch.uint8)
    pages = tables[rows, positions // PAGE].long()
    assert not torch.equal(raw3[pages, 0, :, positions % PAGE], encode(k[:, 3], DRAFT_SCALES[0]))


# ---------------------------------------------------------------------------------------------
# Reads
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("ctx", [4095, 4096, 4097, 300])
@pytest.mark.parametrize("scale", [2.0, 8.0, 4.0, 0.5, 1.5])
def test_windowed_read_matches_the_fp32_oracle(ctx, scale):
    q, pool, _, gather, _, _, ref = _case(ctx, scale, seed=ctx + int(scale * 10))
    out = dflash_windowed_block_attention(q, pool, gather, SCALE, kv_scale=scale)[0]
    assert out.dtype == torch.bfloat16 and bool(torch.isfinite(out).all())
    rel_rms, worst = _metrics(out, ref)
    assert rel_rms <= REL_RMS_BOUND, f"rel_rms {rel_rms:.3e} > {REL_RMS_BOUND}"
    assert worst <= WORST_ROW_BOUND, f"worst row {worst:.3e} > {WORST_ROW_BOUND}"


@pytest.mark.parametrize(
    "fault,factor",
    [("inverted", 0.5), ("omitted", 1.0), ("doubled", 4.0), ("wrong_layer", 8.0)],
)
def test_wrong_read_scale_exceeds_the_bounds(fault, factor):
    """Fault controls at s = 2: the correct read is inside the bounds, each wrong factor is not."""
    q, pool, _, gather, _, _, ref = _case(4096, 2.0, seed=77)
    good = _metrics(dflash_windowed_block_attention(q, pool, gather, SCALE, kv_scale=2.0)[0], ref)
    assert good[0] <= REL_RMS_BOUND and good[1] <= WORST_ROW_BOUND
    bad = _metrics(dflash_windowed_block_attention(q, pool, gather, SCALE, kv_scale=factor)[0], ref)
    assert bad[0] > REL_RMS_BOUND and bad[1] > WORST_ROW_BOUND, f"{fault}: {bad}"


def test_reading_an_fp8_pool_without_its_scale_is_refused():
    q, pool, _, gather, _, _, _ = _case(64, 2.0, seed=3)
    with pytest.raises(ValueError, match="FP8 pool"):
        bf16_pool = torch.zeros(pool.shape, dtype=torch.bfloat16, device="cuda")
        dflash_windowed_block_attention(q, bf16_pool, gather, SCALE, kv_scale=2.0)
    with pytest.raises((RuntimeError, TypeError, NotImplementedError)):
        # Codes are not BF16 values: the read without a scale cannot run on FP8 storage.
        dflash_windowed_block_attention(q, pool, gather, SCALE)
        torch.cuda.synchronize()


@pytest.mark.parametrize("ctx", [4095, 4096, 4097])
def test_stale_tail_and_foreign_pages_contribute_nothing(ctx):
    """Poisoned codes (448 x s) past the block and on another request's pages leave the output
    bitwise unchanged; a block table row that points a valid page at the foreign page is
    detected by the oracle."""
    scale = 4.0
    q, pool, tables, gather, kc, vc, ref = _case(ctx, scale, seed=ctx)
    clean = dflash_windowed_block_attention(q, pool, gather, SCALE, kv_scale=scale)
    # A second request owning interleaved pages, and stale codes past ctx + BLOCK.
    pool2, tables2 = _fp8_pool([ctx, ctx], poison=0x7E)
    n = ctx + BLOCK
    _place(pool2, tables2, 0, range(n), kc, vc)
    raw = pool2.view(torch.uint8)
    last_page = int(tables2[0, (n - 1) // PAGE])
    raw[last_page, :, :, (n - 1) % PAGE + 1 :] = 0x7E  # stale tail of the row's own page
    gather2 = dflash_window_gather(
        tables2[:1], torch.tensor([ctx], device="cuda"), BLOCK, WINDOW, PAGE
    )
    poisoned = dflash_windowed_block_attention(q, pool2, gather2, SCALE, kv_scale=scale)
    assert torch.equal(clean, poisoned), "a stale or foreign slot reached the read"
    # Fault: the row's first valid page replaced by the other request's (poisoned) page.
    bad_tables = tables2[:1].clone()
    first = (ctx - (WINDOW - 1)) // PAGE if ctx >= WINDOW else 0
    bad_tables[0, first] = tables2[1, first]
    gather3 = dflash_window_gather(
        bad_tables, torch.tensor([ctx], device="cuda"), BLOCK, WINDOW, PAGE
    )
    wrong = dflash_windowed_block_attention(q, pool2, gather3, SCALE, kv_scale=scale)[0]
    rel_rms, worst = _metrics(wrong, ref)
    assert rel_rms > REL_RMS_BOUND or worst > WORST_ROW_BOUND, "foreign page not detected"


# ---------------------------------------------------------------------------------------------
# Capability and configuration
# ---------------------------------------------------------------------------------------------


def _stub_drafter(backend="TRTLLM", causal=False, window=4096, fp8_config=True, layers=5):
    model = DFlashForCausalLM.__new__(DFlashForCausalLM)
    nn.Module.__init__(model)
    model.config = types.SimpleNamespace(num_hidden_layers=layers)
    model.model_config = types.SimpleNamespace(
        quant_config=types.SimpleNamespace(
            kv_cache_quant_algo=QuantAlgo.FP8 if fp8_config else None
        )
    )
    model.dflash_attention_backend = backend
    model._kv_cache_scales = None
    w = window - 1
    model._resolve_block_attention = lambda layer: (causal, (w, 0 if causal else w))
    return model


def test_drafter_fp8_configuration_accepts_only_what_it_reads():
    model = _stub_drafter()
    assert model.fp8_kv_cache_unsupported_reason() is None
    model.set_kv_cache_scales(DRAFT_SCALES)
    assert model.kv_cache_scales == DRAFT_SCALES
    assert model.kv_cache_storage_dtype == torch.float8_e4m3fn
    model.set_kv_cache_scales(list(DRAFT_SCALES))  # identical: accepted
    with pytest.raises(ValueError, match="fixed"):
        model.set_kv_cache_scales((1.0,) * 5)
    for bad in [(2.0,) * 4, (2.0, 2.0, 2.0, 0.0, 4.0), (2.0, 2.0, float("nan"), 8.0, 4.0)]:
        with pytest.raises(ValueError, match="finite positive"):
            _stub_drafter().set_kv_cache_scales(bad)
    # Finite positive, but the FP32 scale or reciprocal the store/read use is zero, subnormal
    # or infinite: 1e-300 -> s = 0; 1e300 -> s = inf; 1e-40 -> subnormal s, 1/s = inf.
    for bad in (1e-300, 1e300, 1e-40):
        stub = _stub_drafter()
        with pytest.raises(ValueError, match="layer 3 scale .* unusable in FP32"):
            stub.set_kv_cache_scales((2.0, 2.0, 2.0, bad, 4.0))
        assert stub.kv_cache_scales is None
    with pytest.raises(ValueError, match="not configured as FP8"):
        _stub_drafter(fp8_config=False).set_kv_cache_scales(DRAFT_SCALES)
    for kwargs, reason in [
        ({"backend": "VANILLA"}, "not TRTLLM"),
        ({"backend": "FA4"}, "not TRTLLM"),
        ({"causal": True}, "non-causal windowed"),
        ({"window": 0}, "non-causal windowed"),
    ]:
        stub = _stub_drafter(**kwargs)
        assert reason in stub.fp8_kv_cache_unsupported_reason()
        with pytest.raises(ValueError, match="cannot store its KV cache in FP8"):
            stub.set_kv_cache_scales(DRAFT_SCALES)
