# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The MTP draft layer's hyper-connection sites on the trunk's fused entries, against the
HuggingFace math in float64 and against the unfused composition of the same math.

`_mtp_front`, `_mtp_tail` and `MTPLayer.shared_head` run over a stand-in core that holds the
draft layer's hyper-connection weights (synthetic, at the checkpoint's element statistics): the
attention-side site with the front end's `x_h + x_e` as its injection at unit weight, the
MoE-side site after the output projection, and the head's stack-final site. `_mtp_tail`'s MoE
is the identity on its mixed rows and its TP sum the identity: the sites around them are under
test, at the rows a draft step runs -- 1 (the later draft steps and every shared head), 4 (the
first draft step of an MTP-3 verify of one request), 32 and 1024 (larger batches -- an
attention-DP rank's decode batch -- and a prefill).

The float64 reference rounds to bf16 where the runtime's activations are bf16 between modules
(the norm outputs, the projections, the residual) and once at each mixer output; the mixers of
the fused path keep their logits in fp32 (the trunk's numerics). Each output must be within
the band of that reference that the unfused composition (per-stream `flashinfer_gemma_rmsnorm`
and `qwen4_exp_gated_residual`) reaches, widened by half.
"""

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.catalog.gemm.cublas_mm import cublas_mm as mtp_cublas_mm
from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.ops.hc.qwen4_exp_gated_residual import (
    qwen4_exp_gated_residual,
)
from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.ops.hc.qwen4_exp_hc_inject_norm import (
    qwen4_exp_hc_inject_norm,
)
from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.ops.hc.qwen4_exp_post_inject import (
    qwen4_exp_post_inject,
)
from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.ops.norm.flashinfer_gemma_rmsnorm import (
    flashinfer_gemma_rmsnorm,
)
from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_103__default import (  # noqa: E501
    modeling as MOD,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 3),
    reason="the fused hyper-connection entries are certified on sm_103",
)

DEV = "cuda"
BF16 = torch.bfloat16
HC = 4
HIDDEN = 2560
WIDTH = HC * HIDDEN
RANK = 320
EPS = 1e-6
#: A TP / TEP rank's gated attention output: 6 of the 24 query heads of 256.
Q_WIDTH = 1536
VOCAB = 4096
P = "l0_mtp_"

# Element statistics of the checkpoint's hyper-connection tensors (the gated_residual
# entry's test): the norm weight's tail puts the normed stream at ~+-30.
DOWN_RMS, UP_RMS, INJECT_RMS, GAMMA_RMS, GAMMA_ABSMAX = 0.0296, 0.0681, 0.0107, 0.477, 6.625


class _Weights(dict):
    """The draft layer's weights; the operands of its attention and MoE, which no site here
    reads, are placeholders."""

    def __missing__(self, key):
        return torch.zeros(1, 1, dtype=BF16, device=DEV)


def _raw(seed: int) -> dict:
    g = torch.Generator(device=DEV).manual_seed(seed)

    def normal(*shape, scale):
        return (torch.randn(*shape, generator=g, device=DEV) * scale).to(BF16)

    def gamma(n):
        t = normal(n, scale=GAMMA_RMS)
        t[0], t[n // 2] = GAMMA_ABSMAX, -GAMMA_ABSMAX
        return t

    w = {
        "embed": normal(VOCAB, HIDDEN, scale=1.0),
        "fhc_down": normal(RANK, WIDTH, scale=DOWN_RMS),
        f"{P}pre_e_norm": gamma(HIDDEN),
        f"{P}pre_h_norm": gamma(WIDTH),
        f"{P}fc_e": normal(HIDDEN, HIDDEN, scale=HIDDEN**-0.5),
        f"{P}fc_h": normal(HIDDEN, HIDDEN, scale=HIDDEN**-0.5),
        f"{P}o_w": normal(HIDDEN, Q_WIDTH, scale=Q_WIDTH**-0.5),
    }
    for tag in ("ahc", "mhc", "fhc"):
        w[f"{P}{tag}_norm"] = gamma(WIDTH)
        w[f"{P}{tag}_down"] = normal(RANK, WIDTH, scale=DOWN_RMS)
        w[f"{P}{tag}_up"] = normal(WIDTH, RANK, scale=UP_RMS)
        if tag != "fhc":
            w[f"{P}{tag}_inject"] = normal(HC, WIDTH, scale=INJECT_RMS)
    return w


def _core(raw: dict):
    """A stand-in core over copies of `raw`, loaded as `post_load_weights` loads the draft
    layer: the first projections fused, then the views."""
    w = _Weights({k: v.clone() for k, v in raw.items()})
    core = SimpleNamespace(
        w=w,
        num_layers=0,
        mtp_layer_idx=0,
        mtp_enabled=True,
        hidden=HIDDEN,
        hc_count=HC,
        stream=WIDTH,
        eps=EPS,
        hc_lowrank=RANK,
        hc_first_cols=-(-(RANK + HC) // 32) * 32,
        max_num_tokens=8192,
        # The shared expert's operands are placeholders (`_Weights`).
        shared_inter=1,
        tp_sum=lambda t: t,
        _hc_first=None,
    )
    MOD.Qwen4ExpDefaultModelingV2._fuse_hc_first_projections(core)
    core._mtp = MOD._mtp_views(core)
    return core, core._mtp


@pytest.fixture
def moe_identity(monkeypatch):
    monkeypatch.setattr(MOD, "_mtp_moe_block", lambda core, M, x, pad_rows=0: x)


# ------------------------------------------------------------------- references ---


def _bf(x: torch.Tensor) -> torch.Tensor:
    return x.to(BF16)


def _norm64(x: torch.Tensor, gamma: torch.Tensor, groups: int) -> torch.Tensor:
    """Gemma RMS norm of each of `groups` equal column groups, float64, rounded once."""
    xs = x.double().unflatten(-1, (groups, -1))
    xs = xs * torch.rsqrt(xs.pow(2).mean(-1, keepdim=True) + EPS)
    return _bf(xs.flatten(-2) * (1.0 + gamma.double()))


def _mm64(a: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """`a @ w^T` for an `nn.Linear` weight, float64, rounded once."""
    return _bf(a.double() @ w.double().t())


def _mixer64(raw, tag, normed, inject=True):
    return qwen4_exp_gated_residual.reference(
        normed,
        raw[f"{P}{tag}_down"],
        raw[f"{P}{tag}_up"],
        raw[f"{P}{tag}_inject"] if inject else None,
        HC,
    )


def _front64(raw, ids, hs):
    rows = hs.shape[0]
    x_e = _mm64(_norm64(raw["embed"][ids], raw[f"{P}pre_e_norm"], 1), raw[f"{P}fc_e"])
    h_n = _norm64(hs, raw[f"{P}pre_h_norm"], HC)
    x_h = _mm64(h_n.reshape(rows * HC, HIDDEN), raw[f"{P}fc_h"]).reshape(rows, WIDTH)
    x = _bf(x_h.double() + x_e.double().repeat(1, HC))
    return (x, *_mixer64(raw, "ahc", _norm64(x, raw[f"{P}ahc_norm"], HC)))


def _inject64(x, block, inject):
    gated = block.double().unsqueeze(-2) * inject.double().unsqueeze(-1)
    return _bf((x.double().unflatten(-1, (HC, HIDDEN)) + gated).flatten(-2))


def _tail64(raw, x, attn, inject):
    x = _inject64(x, _mm64(attn, raw[f"{P}o_w"]), inject)
    mixed, inject = _mixer64(raw, "mhc", _norm64(x, raw[f"{P}mhc_norm"], HC))
    return _inject64(x, mixed, inject)


def _head64(raw, hs):
    return _mixer64(raw, "fhc", _norm64(hs, raw[f"{P}fhc_norm"], HC), inject=False)[0]


# The composition the fused sites replace: one `flashinfer_gemma_rmsnorm` a stream, then
# `qwen4_exp_gated_residual` (bf16 GEMM outputs), and the separate accumulate.


def _grouped(x, gamma):
    parts = torch.split(x, [HIDDEN] * HC, 1)
    return torch.cat(
        [flashinfer_gemma_rmsnorm(p, g, EPS) for g, p in zip(torch.split(gamma, HIDDEN), parts)],
        1,
    )


def _mixer_unfused(raw, tag, normed, inject=True):
    return qwen4_exp_gated_residual(
        normed,
        raw[f"{P}{tag}_down"],
        raw[f"{P}{tag}_up"],
        raw[f"{P}{tag}_inject"] if inject else None,
        HC,
    )


def _front_unfused(raw, ids, hs):
    rows = hs.shape[0]
    e = torch.nn.functional.embedding(ids, raw["embed"])
    x_e = mtp_cublas_mm(
        flashinfer_gemma_rmsnorm(e, raw[f"{P}pre_e_norm"], EPS), raw[f"{P}fc_e"].t()
    )
    h_n = _grouped(hs, raw[f"{P}pre_h_norm"])
    x_h = mtp_cublas_mm(h_n.reshape(rows * HC, HIDDEN), raw[f"{P}fc_h"].t()).reshape(rows, WIDTH)
    x = torch.add(x_h, torch.cat([x_e] * HC, -1))
    return (x, *_mixer_unfused(raw, "ahc", _grouped(x, raw[f"{P}ahc_norm"])))


def _tail_unfused(raw, x, attn, inject):
    x = qwen4_exp_post_inject(x, mtp_cublas_mm(attn, raw[f"{P}o_w"].t()), inject)
    mixed, inject = _mixer_unfused(raw, "mhc", _grouped(x, raw[f"{P}mhc_norm"]))
    return qwen4_exp_post_inject(x, mixed, inject)


def _head_unfused(raw, hs):
    return _mixer_unfused(raw, "fhc", _grouped(hs, raw[f"{P}fhc_norm"]), inject=False)[0]


def _err(got: torch.Tensor, ref: torch.Tensor) -> float:
    """Worst element's distance in units of `eps * max|ref|` (bf16 eps)."""
    ref64 = ref.double()
    unit = torch.finfo(BF16).eps * ref64.abs().max().item()
    return (got.double() - ref64).abs().max().item() / unit


def _rel_l2(got: torch.Tensor, ref: torch.Tensor) -> float:
    ref64 = ref.double()
    return ((got.double() - ref64).norm() / ref64.norm()).item()


def _within(name, got, ref, unfused) -> None:
    """`got` as close to `ref` as the unfused composition `unfused` is, widened by half (and
    never tighter than one eps of max|ref| / 1e-3 relative L2, both below the noise of one
    bf16 rounding of a mixer output)."""
    assert got.shape == ref.shape and got.dtype == ref.dtype, (name, got.shape, ref.shape)
    assert bool(torch.isfinite(got).all()), name
    e_new, e_old = _err(got, ref), _err(unfused, ref)
    l_new, l_old = _rel_l2(got, ref), _rel_l2(unfused, ref)
    print(
        f"{name}: max {e_new:.3f} (unfused {e_old:.3f}) eps*max|ref|, relL2 {l_new:.3g} (unfused {l_old:.3g})"
    )
    assert e_new <= max(1.5 * e_old, 1.0), (name, e_new, e_old)
    assert l_new <= max(1.5 * l_old, 1e-3), (name, l_new, l_old)


# ------------------------------------------------------------------------ tests ---


def _inputs(rows: int, seed: int, strided: bool = False):
    """Token ids, the trunk's `[rows, stream]` residual (with `strided`, the column view of
    the trunk's packed `[final | hc]` output the first draft step reads), the attention
    output and an attention-site injection in the checkpoint's range."""
    g = torch.Generator(device=DEV).manual_seed(seed)
    ids = torch.randint(0, VOCAB, (rows,), generator=g, device=DEV)
    if strided:
        packed = (torch.randn(rows, HIDDEN + WIDTH, generator=g, device=DEV) * 3).to(BF16)
        hs = packed[:, HIDDEN:]
    else:
        hs = (torch.randn(rows, WIDTH, generator=g, device=DEV) * 3).to(BF16)
    attn = torch.randn(rows, Q_WIDTH, generator=g, device=DEV).to(BF16)
    inject = (0.43 + 1.15 * torch.rand(rows, HC, generator=g, device=DEV)).to(BF16)
    return ids, hs, attn, inject


ROWS = (1, 4, 32, 1024)


@pytest.mark.parametrize(
    "rows,strided", [(r, False) for r in ROWS] + [(4, True)], ids=lambda v: str(v)
)
def test_the_front_site(rows, strided) -> None:
    raw = _raw(11)
    core, M = _core(raw)
    ids, hs, _, _ = _inputs(rows, 101 + rows, strided)
    x, mixed, inject = MOD._mtp_front(core, M, raw["embed"], ids, hs)
    ref, unfused = _front64(raw, ids, hs), _front_unfused(raw, ids, hs)
    for name, got, r, o in zip(("x", "mixed", "inject"), (x, mixed, inject), ref, unfused):
        _within(f"front[{rows}].{name}", got, r, o)
    # The unit injection is the bf16 add: the residual is bitwise the replaced one's, whose
    # projections are the same calls.
    _, h_n = qwen4_exp_hc_inject_norm(hs, None, None, M.pre_h_norm, EPS, HC)
    x_h = mtp_cublas_mm(h_n.reshape(rows * HC, HIDDEN), M.fc_h).reshape(rows, WIDTH)
    e = torch.nn.functional.embedding(ids, raw["embed"])
    x_e = mtp_cublas_mm(flashinfer_gemma_rmsnorm(e, M.pre_e_norm, EPS), M.fc_e)
    assert torch.equal(x, torch.add(x_h, torch.cat([x_e] * HC, -1)))


@pytest.mark.parametrize("rows", ROWS)
def test_the_tail_sites(rows, moe_identity) -> None:
    raw = _raw(12)
    core, M = _core(raw)
    _, x, attn, inject = _inputs(rows, 201 + rows)
    out = MOD._mtp_tail(core, M, x, attn, inject, 0)
    _within(
        f"tail[{rows}]", out, _tail64(raw, x, attn, inject), _tail_unfused(raw, x, attn, inject)
    )


# 32: a shared head over one attention-DP rank's decode batch (ht_dep4 drafts under MTP).
@pytest.mark.parametrize("rows", (1, 4, 32))
def test_the_head_site(rows) -> None:
    raw = _raw(13)
    core, _ = _core(raw)
    _, hs, _, _ = _inputs(rows, 301 + rows)
    head = SimpleNamespace(
        core=core, logits_processor=SimpleNamespace(forward=lambda final, *args: final)
    )
    final = MOD.MTPLayer.shared_head(head, hs, None, None, False)
    _within(f"head[{rows}]", final, _head64(raw, hs), _head_unfused(raw, hs))


def test_a_captured_draft_step_replays_the_eager_bits(moe_identity) -> None:
    """Front, tail and head of a 4-row draft step captured in one CUDA graph: the replay is
    bitwise the eager result."""
    raw = _raw(14)
    core, M = _core(raw)
    ids, hs, attn, _ = _inputs(4, 401, strided=True)
    head = SimpleNamespace(
        core=core, logits_processor=SimpleNamespace(forward=lambda final, *args: final)
    )

    def step():
        x, _, inject = MOD._mtp_front(core, M, raw["embed"], ids, hs)
        out = MOD._mtp_tail(core, M, x, attn, inject, 0)
        return out, MOD.MTPLayer.shared_head(head, out, None, None, False)

    eager = step()
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        step()
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = step()
    graph.replay()
    torch.cuda.synchronize()
    for got, want in zip(captured, eager):
        assert torch.equal(got, want)
