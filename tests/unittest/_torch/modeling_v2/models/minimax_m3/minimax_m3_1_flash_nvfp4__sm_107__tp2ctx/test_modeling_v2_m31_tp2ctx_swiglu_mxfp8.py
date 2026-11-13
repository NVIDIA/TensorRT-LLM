# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TP2CTX_SWIGLU_MXFP8: the swigluoai activation and down_proj's MXFP8 quantize as one C++ kernel.

The family op minimax_m31_swiglu_oai_mxfp8 must be bitwise the stock pair, the Triton silu_and_mul (limit 7, alpha
1.702, beta 1) followed by mxfp8_quantize(h, True): the E4M3 values, the swizzled UE8M0 scales (padding rows zeroed)
and the optional bf16 activation. The tp2ctx target swaps it into the swigluoai MLPs whose down_proj is a
TP2CTX_NORMQ consumer (kill switch TRTLLM_M31_DISABLE_TP2CTX_SWIGLU_MXFP8). The GPU tests run on SM107, where the
family library carries the op.

From 1024 rows the op launches its rows form (4 rows in flight per CTA on an occupancy-true grid), bitwise the stock
pair on stress inputs, also when its first launch is inside a CUDA-graph capture.
"""

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import torch

import tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3 as _family
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.native import family_ops
from tensorrt_llm._utils import get_sm_version

_TARGET = Path(_family.__file__).resolve().parent / "minimax_m3_1_flash_nvfp4__sm_107__tp2ctx"


def test_tp2ctx_swaps_only_swigluoai_normq_mlps_with_a_kill_switch():
    src = (_TARGET / "modeling.py").read_text()
    assert '_SWIGLU_MXFP8 = not m31_disabled("TP2CTX_SWIGLU_MXFP8")' in src
    assert "module.__class__ = _SwigluMxfp8GatedMLP" in src
    flat = "".join(src.split())  # the condition may wrap across lines
    assert "(module.swiglu_alpha,module.swiglu_beta,module.swiglu_limit)==(1.702,1.0,7.0)" in flat
    # The swap runs after down_proj's quant method became the TP2CTX_NORMQ consumer.
    assert src.index("module.quant_method.__class__ = _NormqMXFP8LinearMethod") < src.index(
        "module.__class__ = _SwigluMxfp8GatedMLP"
    )


def _op():
    if not torch.cuda.is_available() or get_sm_version() != 107:
        pytest.skip("SM107 (Rubin) only")
    op = getattr(family_ops(), "minimax_m31_swiglu_oai_mxfp8", None)
    assert op is not None, "the family library lacks minimax_m31_swiglu_oai_mxfp8"
    return op


def _inputs(rows, d, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(rows, 2 * d, generator=g, device="cuda") * 3.0
    if rows >= 4:
        x[1] = 0.0  # an all-zero row: zero scales and zero values
        x[2, :d] = torch.linspace(
            -12.0, 12.0, d, device="cuda"
        )  # gates across both sides of the clamp
        x[2, d:] = torch.linspace(-12.0, 12.0, d, device="cuda")
        x[3] *= 1e-30  # subnormal-range activations
    return x.to(torch.bfloat16).contiguous()


def _reference(x):
    h = torch.ops.trtllm.silu_and_mul(x, swiglu_limit=7.0, swiglu_alpha=1.702, swiglu_beta=1.0)
    q, sf = torch.ops.trtllm.mxfp8_quantize(h, True)
    return h, q, sf.view(torch.uint8).flatten()


@pytest.mark.parametrize("d", [256, 1536, 3072])
@pytest.mark.parametrize("rows", [1, 127, 1000, 1031, 1160, 4096, 8192, 16384])
def test_bitwise_vs_silu_and_mul_then_mxfp8_quantize(rows, d):
    op = _op()
    x = _inputs(rows, d, seed=rows * 7 + d)
    h_ref, q_ref, sf_ref = _reference(x)
    padded = (rows + 127) // 128 * 128
    h = torch.full((rows, d), float("nan"), dtype=torch.bfloat16, device="cuda")
    q = torch.empty((rows, d), dtype=torch.float8_e4m3fn, device="cuda")
    sf = torch.full(
        (padded * d // 32,), 0xAB, dtype=torch.uint8, device="cuda"
    )  # padding must be rewritten to 0
    op(x, q, sf, h)
    torch.cuda.synchronize()
    assert sf_ref.numel() == sf.numel()
    assert torch.equal(h.view(torch.int16), h_ref.view(torch.int16))
    assert torch.equal(q.view(torch.uint8), q_ref.view(torch.uint8))
    assert torch.equal(sf, sf_ref)


def test_without_the_bf16_output_and_under_graph_replay():
    op = _op()
    rows, d = 4096, 1536
    x = _inputs(rows, d, seed=11)
    _, q_ref, sf_ref = _reference(x)
    q = torch.empty((rows, d), dtype=torch.float8_e4m3fn, device="cuda")
    sf = torch.empty((rows * d // 32,), dtype=torch.uint8, device="cuda")
    op(x, q, sf, None)
    torch.cuda.synchronize()
    assert torch.equal(q.view(torch.uint8), q_ref.view(torch.uint8)) and torch.equal(sf, sf_ref)
    q.zero_()
    sf.zero_()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        op(x, q, sf, None)
    torch.cuda.current_stream().wait_stream(s)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        op(x, q, sf, None)
    q.zero_()
    sf.zero_()
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(q.view(torch.uint8), q_ref.view(torch.uint8)) and torch.equal(sf, sf_ref)


def test_rejects_unsupported_widths():
    op = _op()
    x = torch.zeros(8, 2 * 160, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(RuntimeError, match="multiple of 256"):
        op(x, torch.empty(8, 160, dtype=torch.float8_e4m3fn, device="cuda"),
           torch.empty(128 * 5, dtype=torch.uint8, device="cuda"), None)  # fmt: skip


def test_the_mlp_publishes_the_pair_for_its_down_proj():
    """The swapped MLP's down_proj takes the published (q, sf) for its own bf16 input, so its GEMM reads the operands
    the stock quantize would have produced; a different input is not taken."""
    op = _op()
    from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_1_flash_nvfp4__sm_107__tp2ctx import (  # noqa: E501
        modeling,
    )

    rows, d = modeling._NORMQ_MIN_ROWS, 1536
    x = _inputs(rows, d, seed=5)
    h = torch.empty((rows, d), dtype=torch.bfloat16, device="cuda")
    q = torch.empty((rows, d), dtype=torch.float8_e4m3fn, device="cuda")
    sf = torch.empty((rows * d // 32,), dtype=torch.uint8, device="cuda")
    op(x, q, sf, h)
    modeling._NORMQ_SLOT.publish(h, q, sf)
    assert modeling._NORMQ_SLOT.take(h.clone()) is None
    modeling._NORMQ_SLOT.publish(h, q, sf)
    taken = modeling._NORMQ_SLOT.take(h)
    assert taken is not None and taken[0] is q and taken[1] is sf


# --- SWIGLU_MXFP8_ROWS: the rows form against the one-row launch ---------------------------------------------

# Builds the stress inputs and runs the op for every case; one process per switch setting. Kept as source so the
# subprocesses and the in-process test build the same inputs.
_STRESS_SRC = textwrap.dedent(
    """
    import hashlib

    import torch

    def stress_inputs(rows, d, seed):
        g = torch.Generator(device="cuda").manual_seed(seed)
        x = torch.randn(rows, 2 * d, generator=g, device="cuda")
        # per-32-column block magnitudes 2^-12 .. 2^12 (gate and up blocks scaled independently)
        e = torch.randint(-12, 13, (rows, 2 * d // 32), generator=g, device="cuda").float()
        x = x * torch.exp2(e).repeat_interleave(32, dim=1)
        if rows >= 8:
            x[1] = 0.0  # zero row: zero scales and values
            x[2, :d] = torch.linspace(-12.0, 12.0, d, device="cuda")  # gates across the +-7 clamp
            x[2, d:] = torch.linspace(-12.0, 12.0, d, device="cuda")  # ups across the +-7 clamp
            x[3] *= 1e-36  # down into the bf16 / fp32 subnormal range
            x[4, :32] = 0.0  # one zero K32 block in an otherwise live row
            x[5] = 7.0  # exactly at the clamp
            x[6] = -7.0
        return x.to(torch.bfloat16).contiguous()

    def run_case(op, rows, d, write_h, seed):
        x = stress_inputs(rows, d, seed)
        padded = (rows + 127) // 128 * 128
        q = torch.full((rows, d), 0x7F, dtype=torch.uint8, device="cuda").view(torch.float8_e4m3fn)
        sf = torch.full((padded * d // 32,), 0xAB, dtype=torch.uint8, device="cuda")
        h = torch.full((rows, d), float("nan"), dtype=torch.bfloat16, device="cuda") if write_h else None
        op(x, q, sf, h)
        torch.cuda.synchronize()
        out = {"x": x, "q": q, "sf": sf}
        if write_h:
            out["h"] = h
        return out

    def digest(out):
        # sha256 of every tensor's bytes, and whether every scale byte (pad rows included) was written.
        d = {k: hashlib.sha256(v.contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()
             for k, v in out.items()}
        d["sf_written"] = bool((out["sf"] != 0xAB).all())
        return d
    """
)


def test_rows_form_stress_bitwise_vs_stock():
    """The default (rows) form against the stock pair on the stress inputs."""
    op = _op()
    ns = {}
    exec(_STRESS_SRC, ns)
    for rows, d in ((1160, 3072), (8192, 3072), (4096, 2048)):
        x = ns["stress_inputs"](rows, d, seed=rows + d)
        h_ref, q_ref, sf_ref = _reference(x)
        got = ns["run_case"](op, rows, d, True, seed=rows + d)
        assert torch.equal(got["x"], x)
        assert torch.equal(got["h"].view(torch.int16), h_ref.view(torch.int16))
        assert torch.equal(got["q"].view(torch.uint8), q_ref.view(torch.uint8))
        assert torch.equal(got["sf"], sf_ref)


_CAPTURE_FIRST = textwrap.dedent(
    """
    import torch
    import tensorrt_llm  # noqa: F401  (loads the trtllm op library)
    from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.native import family_ops

    op = family_ops().minimax_m31_swiglu_oai_mxfp8
    rows, d = 8192, 3072
    x = stress_inputs(rows, d, seed=3)
    q = torch.zeros((rows, d), dtype=torch.float8_e4m3fn, device="cuda")
    sf = torch.zeros((rows * d // 32,), dtype=torch.uint8, device="cuda")
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):  # the process's first launch (switch read, occupancy query) is under capture
        op(x, q, sf, None)
    graph.replay()
    torch.cuda.synchronize()
    h_ref = torch.ops.trtllm.silu_and_mul(x, swiglu_limit=7.0, swiglu_alpha=1.702, swiglu_beta=1.0)
    q_ref, sf_ref = torch.ops.trtllm.mxfp8_quantize(h_ref, True)
    assert torch.equal(q.view(torch.uint8), q_ref.view(torch.uint8))
    assert torch.equal(sf, sf_ref.view(torch.uint8).flatten())
    print("CAPTURE_FIRST_OK")
    """
)


def test_rows_form_first_launch_under_graph_capture():
    """The rows form's one-time work (the switch read, the occupancy query) may first run inside a graph capture, and
    the captured launch is the rows form (the occupancy query is not a stream operation, so it must not fail there)."""
    _op()
    env = dict(os.environ, TLLM_LOG_LEVEL="INFO")
    out = subprocess.run(
        [sys.executable, "-c", _STRESS_SRC + _CAPTURE_FIRST],
        env=env,
        check=True,
        timeout=900,
        capture_output=True,
        text=True,
    )
    assert "CAPTURE_FIRST_OK" in out.stdout
    log = out.stdout + out.stderr
    assert "rows form kRows=4 at d 3072 (h elided)" in log, log
    assert "NOT ENGAGED" not in log and "occupancy query failed" not in log, log


@pytest.mark.parametrize("k_rows", [2, 4, 8])
def test_the_rows_form_arguments_keep_the_bits(k_rows):
    """rows_form=False (the SWIGLU_MXFP8_ROWS kill switch: the one-row launch) and every k_rows write the default
    launch's bytes on the stress inputs."""
    op = _op()
    ns = {}
    exec(_STRESS_SRC, ns)
    for rows, d in ((1160, 3072), (8192, 3072)):
        base = ns["run_case"](op, rows, d, True, seed=rows + d)
        for kwargs in (dict(rows_form=False), dict(k_rows=k_rows)):
            got = ns["run_case"](
                lambda x, q, sf, h, kw=kwargs: op(x, q, sf, h, **kw), rows, d, True, seed=rows + d
            )
            for key in ("h", "q", "sf"):
                assert torch.equal(got[key].view(torch.uint8), base[key].view(torch.uint8)), (
                    kwargs,
                    key,
                )
