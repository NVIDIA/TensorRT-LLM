# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the GEMV's producer epilogue (kernels/mx_gemv PR_*, modeling.QKV_PRODFOLD).

The sparse layers' fused QKV + index projection [M, 6144] x [2560, 6144]^T with m3_producer's norm / RoPE / quantize /
cache insert in its split-K epilogue is held, bit for bit, to the plain GEMV followed by the producer
(ops/attention/m3_producer at the TP4 heads, prewait_step_inputs as the model runs it): q, index-q, every byte of the
NVFP4 K / V pool, its scale pool and the index-K cache (unwritten bytes included), and with PR_Y the packed bf16 y.
MXFP8 activations at the model's (split 8, ring 8) and bf16 activations at the shared table's split 4.
"""

import zlib
from types import SimpleNamespace

import pytest
import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_1_flash_nvfp4__sm_107__tp4.kernels.mx_gemv import (  # noqa: E501
    op as gemv,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_1_flash_nvfp4__sm_107__tp4.ops.attention.m3_producer import (  # noqa: E501
    m3_producer,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7),
    reason="the sm_107 TP4 target runs on SM107 only",
)

N, K = 2560, 6144
HEADS_Q = 16
PAGE = 128
PAGES = 512
MAX_POSITIONS = 1 << 18
EPS = 1e-6
KV_QUANT = (1.0, 0.37, 0.61)
MODEL_CFG = (8, 8)  # modeling.MX_GEMV_CFG[QKV_SPARSE]


def _case(rows: int, scale: float, bad_slots: bool, seed: int):
    g = torch.Generator(device="cuda").manual_seed(seed)
    w = (torch.randn(N, K, generator=g, device="cuda") / K**0.5).to(torch.bfloat16)
    x = (torch.randn(rows, K, generator=g, device="cuda") * scale).to(torch.bfloat16)
    wq, wsf = torch.ops.trtllm.mxfp8_quantize(w, True)
    xq, xsf = torch.ops.trtllm.mxfp8_quantize(x, True)
    rotary = torch.randn(MAX_POSITIONS, 2, 32, device="cuda", generator=g)
    weights = [(torch.randn(128, device="cuda", generator=g) * 0.3).bfloat16() for _ in range(4)]
    slots = torch.randperm(PAGES * PAGE, device="cuda", generator=g)[:rows].int()
    if bad_slots and rows >= 3:
        slots[1] = -1
        slots[2] = PAGES * PAGE + 5
    positions = torch.randint(
        0, MAX_POSITIONS, (rows,), device="cuda", dtype=torch.int32, generator=g
    )
    quant = torch.tensor(KV_QUANT, device="cuda")
    return dict(
        x=x,
        xq=xq.view(rows, K),
        xsf=xsf,
        wq=wq.view(N, K),
        wsf=wsf,
        rotary=rotary,
        weights=weights,
        slots=slots,
        positions=positions,
        quant=quant,
    )


def _caches(seed: int):
    # Random prior contents: an epilogue writing a wrong byte or a byte it must not touch shows up.
    g = torch.Generator(device="cuda").manual_seed(seed)
    return (
        torch.randint(
            0, 256, (PAGES, 2, 1, PAGE, 64), dtype=torch.uint8, device="cuda", generator=g
        ),
        torch.randint(
            0, 256, (PAGES, 2, 1, PAGE, 8), dtype=torch.uint8, device="cuda", generator=g
        ),
        torch.randint(
            0, 256, (PAGES, 1, PAGE, 128), dtype=torch.uint8, device="cuda", generator=g
        ).view(torch.float8_e4m3fn),
    )


def _gemv(c, bf16: bool, prod=None, cfg=MODEL_CFG):
    if bf16:
        return gemv.m3_mx_gemv_bf16(c["x"], c["wq"], c["wsf"], True, None, True, None, prod=prod)
    return gemv.m3_mx_gemv(c["xq"], c["xsf"], c["wq"], c["wsf"], True, cfg, True, None, prod=prod)


def _reference(c, bf16: bool, caches):
    """The plain GEMV, then the producer on its packed output (what the model ran before the fold)."""
    y = _gemv(c, bf16)
    assert y is not None and not gemv.last_prod
    q, index_q = m3_producer(
        y, *caches, c["slots"], c["quant"], EPS, *c["weights"], c["rotary"], c["positions"], HEADS_Q,
        True, True, True,
    )  # fmt: skip
    return y, q, index_q


def _fused(c, bf16: bool, caches, bits: int, cfg=MODEL_CFG):
    rows = c["x"].shape[0]
    q = torch.full((rows, HEADS_Q, 128), 0x7F, dtype=torch.uint8, device="cuda").view(
        torch.float8_e4m3fn
    )
    index_q = torch.full((rows, 1, 128), 0x7F, dtype=torch.uint8, device="cuda").view(
        torch.float8_e4m3fn
    )
    args = gemv.producer_args(
        q, index_q, *caches, c["slots"], c["positions"], c["rotary"], c["quant"], *c["weights"], EPS
    )
    assert args is not None
    y = _gemv(c, bf16, (bits, args), cfg)
    assert y is not None and gemv.last_prod, "the GEMV declined the producer epilogue"
    return y, q, index_q


def _check(c, bf16: bool, bits: int, seed: int, cfg=MODEL_CFG) -> None:
    ref_caches, caches = _caches(seed), _caches(seed)
    y_ref, q_ref, iq_ref = _reference(c, bf16, ref_caches)
    y, q, iq = _fused(c, bf16, caches, bits, cfg)
    torch.cuda.synchronize()
    names = ("q", "index_q", "kv_data", "kv_scale", "index_k")
    for name, a, b in zip(names, (q_ref, iq_ref, *ref_caches), (q, iq, *caches)):
        assert torch.equal(a.view(torch.uint8), b.view(torch.uint8)), (
            f"{name} differs from the GEMV + producer: "
            f"{int((a.view(torch.uint8) != b.view(torch.uint8)).sum())} bytes"
        )
    if bits & gemv.PR_Y:
        assert torch.equal(y_ref.view(torch.int16), y.view(torch.int16)), (
            "y differs from the plain GEMV"
        )


BITS = gemv.PR_ON | gemv.PR_PREWAIT


@pytest.mark.parametrize("bf16", [False, True], ids=["mxfp8", "bf16"])
@pytest.mark.parametrize(
    "rows, scale, bad_slots",
    [(8, 1.0, False), (8, 40.0, True), (8, 1e-3, False), (1, 1.0, False), (3, 3.0, True), (4, 1.0, False),
     (7, 0.5, False)],
)  # fmt: skip
def test_prodfold_is_the_gemv_then_the_producer(bf16, rows, scale, bad_slots) -> None:
    seed = zlib.crc32(f"{bf16}{rows}{scale}{bad_slots}".encode())
    _check(_case(rows, scale, bad_slots, seed), bf16, BITS | gemv.PR_Y, seed)


@pytest.mark.parametrize("bf16", [False, True], ids=["mxfp8", "bf16"])
@pytest.mark.parametrize("prewait", [True, False], ids=["prewait", "postwait"])
def test_prodfold_model_bits(bf16, prewait) -> None:
    """The model's bits (no y) and the PRODUCER_PREWAIT_STEP_INPUTS kill value write the same bytes."""
    bits = gemv.PR_ON | (gemv.PR_PREWAIT if prewait else 0)
    for seed in (5, 6, 7):
        _check(_case(8, 1.0, seed == 6, seed), bf16, bits, seed)


@pytest.mark.parametrize("rows, bad_slots", [(8, False), (5, True)])
def test_prodfold_ring2_writes_the_same_bytes(rows, bad_slots) -> None:
    """QKV_PRODFOLD_RING2: the folded GEMV at (split 8, ring 2) (A still staged in TMEM, the same MMAs in the same
    order) writes every output byte, and y, of the (8, 8) GEMV + producer."""
    seed = 50 + rows
    _check(_case(rows, 1.0, bad_slots, seed), False, BITS | gemv.PR_Y, seed, (8, 2))


def test_prodfold_declines_what_it_does_not_build() -> None:
    """Over 8 rows, a push, or another projection the GEMV declines the epilogue (the model then runs the producer)."""
    c = _case(8, 1.0, False, 3)
    caches = _caches(3)
    q = torch.empty(16, HEADS_Q, 128, dtype=torch.float8_e4m3fn, device="cuda")
    iq = torch.empty(16, 1, 128, dtype=torch.float8_e4m3fn, device="cuda")
    args = gemv.producer_args(
        q, iq, *caches, c["slots"], c["positions"], c["rotary"], c["quant"], *c["weights"], EPS
    )
    assert args is None, "16-row q with 8 slots must be refused"
    wide = torch.randn(16, K, device="cuda").to(torch.bfloat16)
    wq16, wsf16 = torch.ops.trtllm.mxfp8_quantize(wide, True)
    q16 = torch.empty(16, HEADS_Q, 128, dtype=torch.float8_e4m3fn, device="cuda")
    iq16 = torch.empty(16, 1, 128, dtype=torch.float8_e4m3fn, device="cuda")
    slots16 = torch.arange(16, dtype=torch.int32, device="cuda")
    pos16 = torch.arange(16, dtype=torch.int32, device="cuda")
    args16 = gemv.producer_args(
        q16, iq16, *caches, slots16, pos16, c["rotary"], c["quant"], *c["weights"], EPS
    )
    assert args16 is not None
    assert (
        gemv.m3_mx_gemv(
            wq16.view(16, K), wsf16, c["wq"], c["wsf"], True, None, True, None, prod=(BITS, args16)
        )
        is None
    )
    # o_proj-shaped weight: not the 20-head tile
    wo = torch.randn(6144, 2048, device="cuda").to(torch.bfloat16)
    woq, wosf = torch.ops.trtllm.mxfp8_quantize(wo, True)
    x8 = torch.randn(8, 2048, device="cuda").to(torch.bfloat16)
    x8q, x8sf = torch.ops.trtllm.mxfp8_quantize(x8, True)
    assert (
        gemv.m3_mx_gemv(
            x8q.view(8, 2048),
            x8sf,
            woq.view(6144, 2048),
            wosf,
            True,
            None,
            True,
            None,
            prod=(BITS, args),
        )
        is None
    )


def test_prebuild_compiles_the_epilogue_the_forward_takes(monkeypatch) -> None:
    """modeling.prebuild_kernels' QKV_PRODFOLD precompile (the QKV projection's GEMV variants, as its loop runs them)
    builds the kernel ``_linear`` then calls at every row count up to 8, for both activations: after the freeze, the
    epilogue GEMV runs without compiling (else serving would silently fall back to the producer)."""
    import cutlass.cute as cute

    from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_1_flash_nvfp4__sm_107__tp4 import (  # noqa: E501
        modeling,
    )

    if not modeling.QKV_PRODFOLD_TP4:
        pytest.skip("TRTLLM_M31_DISABLE_QKV_PRODFOLD is set")
    frozen = gemv.frozen()
    c = _case(8, 1.0, False, 21)
    try:
        gemv.freeze(False)
        for rows, x_bf16, ar in modeling._gemv_variants(N, K, None, 4):
            cfg = None if x_bf16 else modeling.mx_gemv_cfg(N, K, rows)
            if ar is None and rows <= modeling.QKV_PRODFOLD_MAX_ROWS:
                gemv.precompile(
                    c["wq"], c["wsf"], rows, x_bf16, use_pdl=True,
                    cfg=None if cfg is None else modeling.QKV_PRODFOLD_CFG, prod=modeling.QKV_PRODFOLD_BITS,
                )  # fmt: skip
        gemv.freeze(True)

        def boom(*args, **kwargs):
            raise AssertionError("a compile ran after the prebuild")

        monkeypatch.setattr(cute, "compile", boom)
        for rows in range(1, modeling.QKV_PRODFOLD_MAX_ROWS + 1):
            for bf16 in (False, True):
                cr = _case(rows, 1.0, False, 30 + rows)
                caches = _caches(rows)
                q = torch.empty(rows, HEADS_Q, 128, dtype=torch.float8_e4m3fn, device="cuda")
                iq = torch.empty(rows, 1, 128, dtype=torch.float8_e4m3fn, device="cuda")
                args = gemv.producer_args(
                    q,
                    iq,
                    *caches,
                    cr["slots"],
                    cr["positions"],
                    cr["rotary"],
                    cr["quant"],
                    *cr["weights"],
                    EPS,
                )
                prod = (modeling.QKV_PRODFOLD_BITS, args)
                if bf16:
                    out = gemv.m3_mx_gemv_bf16(
                        cr["x"], c["wq"], c["wsf"], True, None, True, None, prod=prod
                    )
                else:
                    out = gemv.m3_mx_gemv(
                        cr["xq"], cr["xsf"], c["wq"], c["wsf"], True, modeling.QKV_PRODFOLD_CFG, True, None,
                        prod=prod,
                    )  # fmt: skip
                assert out is not None and gemv.last_prod, (rows, bf16, gemv.last_uncompiled)
        torch.cuda.synchronize()
    finally:
        gemv.freeze(frozen)


@pytest.mark.parametrize("bf16", [False, True], ids=["mxfp8", "bf16"])
@pytest.mark.parametrize("rows", [8, 3])
def test_the_decode_attention_glue_folds_the_producer(monkeypatch, bf16, rows) -> None:
    """modeling's QKV_PRODFOLD glue as ``DecodeTarget._attention`` runs it inside a decode graph's warmup or capture
    (``_qkv_prodfold`` then ``_linear(prod=...)``) writes the q, index-q and cache bytes ``_linear`` + the producer
    (``_sparse_qkv_producer``) write; with the switch off it does not fold."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_1_flash_nvfp4__sm_107__tp4 import (  # noqa: E501
        modeling,
    )
    from tensorrt_llm._torch.modules.linear import flashinfer_mxfp8_decode_graph_capture

    c = _case(rows, 1.0, False, 40 + rows)
    i = 5
    core = SimpleNamespace(
        w={
            f"l{i}_qkv_w": c["wq"],
            f"l{i}_qkv_s": c["wsf"],
            f"l{i}_q_norm": c["weights"][0],
            f"l{i}_k_norm": c["weights"][1],
            f"l{i}_idx_q_norm": c["weights"][2],
            f"l{i}_idx_k_norm": c["weights"][3],
        },
        mxfp8=SimpleNamespace(
            quant_method=SimpleNamespace(
                tune_decode_graph_backends=False,
                backend="trtllm",
                _flashinfer_autotuned=False,
                needs_native_autotune=False,
            )
        ),
        main_kv_is_nvfp4=True,
        heads=HEADS_Q,
        kv_heads=1,
        index_heads=1,
        head_dim=128,
        inv_kv_scales={i: c["quant"]},
        rotary_cos_sin=c["rotary"],
        rms_eps=EPS,
    )

    def run(caches, fold_on: bool):
        kv, sc, ik = caches
        manager = SimpleNamespace(
            get_buffers=lambda layer, kv_layout: kv,
            get_block_scale_buffers=lambda layer, kv_layout: sc,
            is_nvfp4_layer=lambda layer: True,
        )
        monkeypatch.setattr(modeling, "QKV_PRODFOLD_TP4", fold_on)
        x = c["x"]
        with flashinfer_mxfp8_decode_graph_capture():
            fold = modeling._qkv_prodfold(
                core, i, rows, c["positions"], manager, lambda layer: ik, c["slots"]
            )
            assert (fold is not None) == fold_on
            if not bf16:
                modeling.m3_prequant.SLOT.publish(x, c["xq"], c["xsf"])
            packed, _ = modeling._linear(
                core, x, f"l{i}_qkv_w", f"l{i}_qkv_s", True, prod=None if fold is None else fold[0]
            )
            assert modeling._PROD_TAKEN[0] == fold_on
            if fold_on:
                return fold[1]
            return modeling._sparse_qkv_producer(
                core, i, packed, c["positions"], manager, lambda layer: ik, c["slots"]
            )

    ref_caches, caches = _caches(rows), _caches(rows)
    q_ref, iq_ref = run(ref_caches, False)
    q, iq = run(caches, True)
    torch.cuda.synchronize()
    names = ("q", "index_q", "kv_data", "kv_scale", "index_k")
    for name, a, b in zip(names, (q_ref, iq_ref, *ref_caches), (q, iq, *caches)):
        assert a.shape == b.shape, name
        assert torch.equal(a.view(torch.uint8), b.view(torch.uint8)), f"{name} differs"
