# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test: the target's whole Gated-DeltaNet layer against an fp32 PyTorch reference.

Drives `DecodeTarget._gdn_block` and `PrefillTarget._gdn_block` at a rank's two geometries --
the whole layer's 16 key / 48 value heads (tp1, dep4) and a tp4 / tep4 rank's 4 / 12 -- over
the recurrent pool dtypes each serves (fp32 or bf16 at 16 / 48, bf16 at 4 / 12), on a decode
batch, a mixed batch (contexts with and without carried state plus one-token generation rows)
and a context-only batch, and compares the layer output and both state pools with a reference
written from the layer's definition in PyTorch: the input projection (the same GEMM, so the
conv pools are bitwise), the causal conv over each sequence's carried window, SiLU, the q / k
L2 norm, the decay and beta gates, the delta-rule recurrence per token, the sigmoid-gated RMS
norm and the output projection. The reference shares no kernel with the target past the two
GEMMs, so a wrong head split, gate or slot at either geometry fails it.

Bands: the conv pools are bitwise; the recurrent pool is held to rtol 1.6e-2 / atol 2e-2 (the
recurrence entries' state bands), the layer output to a relative L2 distance of 1e-2 and a
max-abs distance of 3e-2 of its own peak.
"""

from __future__ import annotations

import types

import pytest
import torch
import torch.nn.functional as F

from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_103__default import (  # noqa: E501
    modeling as M,
)

assert torch.cuda.is_available(), "the Gated-DeltaNet layer requires a CUDA device"

DEV = "cuda"
BF16 = torch.bfloat16
HIDDEN, K, V, WIDTH = 2560, 128, 128, 4
SLOTS = 461
MAX_BATCH = 256
RTOL_STATE, ATOL_STATE = 1.6e-2, 2e-2
OUT_REL_L2, OUT_MAX_FRAC = 1e-2, 3e-2

#: (key heads, value heads) of a rank: the whole layer's, and a tp4 / tep4 rank's.
GEOMETRIES = {"16x48": (16, 48), "4x12": (4, 12)}
#: (geometry, pool dtype): the pool dtypes each layout serves.
CASES = [("16x48", "fp32"), ("16x48", "bf16"), ("4x12", "bf16")]
STATES = {"fp32": torch.float32, "bf16": BF16}


def _dims(h: int, hv: int) -> types.SimpleNamespace:
    kd, vd = h * K, hv * V
    return types.SimpleNamespace(h=h, hv=hv, kd=kd, vd=vd, conv=2 * kd + vd)


def _slots(n: int) -> list[int]:
    """`n` distinct slots of the pool, both ends first."""
    rest = [(i * 37 + 4) % (SLOTS - 2) + 1 for i in range(max(n - 2, 0))]
    return ([SLOTS - 1, 0] + rest)[:n]


def _core(d, seed: int, state: torch.dtype, one_rank: bool) -> types.SimpleNamespace:
    """The attributes `_gdn_block` reads off the core, with layer 0's pools (the recurrent
    one in `state`); the TP sum is the identity (one rank's partial)."""
    g = torch.Generator(device=DEV).manual_seed(seed)
    conv = torch.randn(SLOTS, d.conv, WIDTH - 1, generator=g, device=DEV, dtype=BF16)
    ssm = (torch.randn(SLOTS, d.hv, V, K, generator=g, device=DEV) * 0.05).to(state)
    return types.SimpleNamespace(
        conv_dim=d.conv,
        key_dim=d.kd,
        value_dim=d.vd,
        num_k_heads=d.h,
        head_k_dim=K,
        num_v_heads=d.hv,
        head_v_dim=V,
        eps=1e-6,
        conv_kernel=WIDTH,
        in_proj_split=(d.conv, d.vd, d.hv, d.hv),
        one_rank=one_rank,
        tp_sum=lambda x: x,
        decode_mm=M._one_rank_projection if one_rank else M._decode_projection,
        decode_in_mm=(
            M._one_rank_projection
            if one_rank
            else (lambda mat_a, mat_b: M.low_m_bf16_mm(mat_a, mat_b, M._DECODE_PROJ_PDL))
        ),
        _bound=types.SimpleNamespace(pools=types.SimpleNamespace(conv={0: conv}, ssm={0: ssm})),
    )


def _clone_core(core: types.SimpleNamespace) -> types.SimpleNamespace:
    out = types.SimpleNamespace(**vars(core))
    pools = core._bound.pools
    out._bound = types.SimpleNamespace(
        pools=types.SimpleNamespace(conv={0: pools.conv[0].clone()}, ssm={0: pools.ssm[0].clone()})
    )
    return out


def _layer(d, seed: int) -> types.SimpleNamespace:
    """Layer weights as the target views them (HF `[out, in]` storage read through `.t()`,
    fp32 gating vectors over the checkpoint's ranges), with no FP8 or NVFP4 copies."""
    g = torch.Generator(device=DEV).manual_seed(seed)

    def lin(n_out, n_in):
        return (torch.randn(n_out, n_in, generator=g, device=DEV) * n_in**-0.5).to(BF16).t()

    return types.SimpleNamespace(
        in_proj=lin(d.conv + d.vd + 2 * d.hv, HIDDEN),
        out=lin(HIDDEN, d.vd),
        conv_w=(torch.randn(d.conv, WIDTH, generator=g, device=DEV) * 0.5).to(BF16),
        a_log=torch.empty(d.hv, device=DEV).uniform_(-5.5, 3.5, generator=g),
        dt_bias=torch.empty(d.hv, device=DEV).uniform_(-8.9, 19.2, generator=g),
        gnorm=(1.0 + 0.1 * torch.randn(V, generator=g, device=DEV)).to(BF16),
        in_proj8=None,
        out8=None,
        in_proj_g=None,
        out_g=None,
        in_proj_m=None,
        out_m=None,
        in_proj_nv=None,
    )


def _x(tokens: int, seed: int) -> torch.Tensor:
    g = torch.Generator(device=DEV).manual_seed(seed)
    return torch.randn(tokens, HIDDEN, generator=g, device=DEV, dtype=BF16)


def _target(cls, core, L=None, step=None):
    """A target over `core` alone; a prefill target's `_gdn_core` break reads layer 0's
    weights and the step's bookkeeping off it."""
    target = object.__new__(cls)
    target.core = core
    target._layers = [L]
    target._step = step
    return target


def _decode_slots(m: int) -> torch.Tensor:
    """The decode step's slot vector: the head of an engine-sized buffer."""
    buf = torch.zeros(MAX_BATCH, dtype=torch.int32, device=DEV)
    buf[:m] = torch.tensor(_slots(m), dtype=torch.int32, device=DEV)
    return buf[:m]


def _prefill_step(ctx_lens: tuple, n_dec: int, has_init: tuple) -> tuple:
    """`PrefillTarget`'s `step.gdn` for a context prefix plus `n_dec` one-token generation
    rows, built the way the forward builds it from the engine's `Mamba2Metadata` buffers."""
    n_ctx = len(ctx_lens)
    batch = n_ctx + n_dec
    n_p = sum(ctx_lens)
    si_buf = torch.zeros(MAX_BATCH, dtype=torch.int32, device=DEV)
    si_buf[:batch] = torch.tensor(_slots(batch), dtype=torch.int32, device=DEV)
    hi_buf = torch.zeros(MAX_BATCH, dtype=torch.bool, device=DEV)
    hi_buf[:n_ctx] = torch.tensor(has_init, dtype=torch.bool, device=DEV)
    si_all, hi_all = si_buf[:batch], hi_buf[:batch]
    si_p, si_d = torch.split(si_all, [n_ctx, n_dec], 0)
    bounds = [0]
    for n in ctx_lens:
        bounds.append(bounds[-1] + n)
    bounds += [n_p + j for j in range(1, n_dec + 1)]
    cu_buf = torch.zeros(MAX_BATCH + 1, dtype=torch.int32, device=DEV)
    cu_buf[: batch + 1] = torch.tensor(bounds, dtype=torch.int32, device=DEV)
    qsl_all = cu_buf[: batch + 1]
    qsl_long = qsl_all.to(torch.int64)
    return (
        n_p,  # num_prefill_tokens
        n_dec,  # num_decode_tokens
        si_p,
        si_all,
        torch.clone(si_d),
        hi_all[:n_ctx],  # hi_p
        hi_all,
        qsl_all[: n_ctx + 1],  # qsl_p
        qsl_all,
        qsl_long,
        qsl_long[: n_ctx + 1],  # qsl_long_p
        n_dec,  # num_decodes
        1,  # gen_p
        None,  # verify
    )


# ------------------------------------------------------------------ the reference ---


def _reference(d, core, L, x, seqs) -> torch.Tensor:
    """The layer over `x`, sequence by sequence: `seqs` is `(first row, rows, slot, carries)`
    per sequence (a generation row carries its state). Updates `core`'s pools in place (the
    conv window bitwise, the recurrent state rounded once to the pool's dtype) and returns
    the layer output."""
    conv_pool, ssm_pool = core._bound.pools.conv[0], core._bound.pools.ssm[0]
    proj = M.cublas_mm(x, L.in_proj)
    mixed, z, a, b = torch.split(proj, [d.conv, d.vd, d.hv, d.hv], 1)
    rep = d.hv // d.h
    w = L.conv_w.float()
    neg_a = -torch.exp(L.a_log.float())
    o = torch.empty(x.shape[0], d.hv, V, dtype=torch.float32, device=DEV)
    for first, rows, slot, carries in seqs:
        seq = slice(first, first + rows)
        window = conv_pool[slot] if carries else torch.zeros_like(conv_pool[slot])
        full = torch.cat([window, mixed[seq].t()], 1)  # [C, W - 1 + rows]
        conv_pool[slot] = full[:, -(WIDTH - 1) :]
        conv = sum(w[:, j : j + 1] * full[:, j : j + rows].float() for j in range(WIDTH))
        conv = F.silu(conv).to(BF16).float().t()  # [rows, C], the kernels' bf16 output
        q, k, v = torch.split(conv, [d.kd, d.kd, d.vd], 1)
        q = q.reshape(rows, d.h, K).repeat_interleave(rep, dim=1)
        k = k.reshape(rows, d.h, K).repeat_interleave(rep, dim=1)
        v = v.reshape(rows, d.hv, V)
        q = q * torch.rsqrt(q.pow(2).sum(-1, keepdim=True) + 1e-6) * K**-0.5
        k = k * torch.rsqrt(k.pow(2).sum(-1, keepdim=True) + 1e-6)
        decay = torch.exp(neg_a * F.softplus(a[seq].float() + L.dt_bias.float(), threshold=20.0))
        beta = torch.sigmoid(b[seq].float())
        h = (
            ssm_pool[slot].float()
            if carries
            else torch.zeros_like(ssm_pool[slot], dtype=torch.float32)
        )
        for t in range(rows):
            h = h * decay[t][:, None, None]
            u = (v[t] - torch.einsum("hvk,hk->hv", h, k[t])) * beta[t][:, None]
            h = h + u[:, :, None] * k[t][:, None, :]
            o[first + t] = torch.einsum("hvk,hk->hv", h, q[t])
        ssm_pool[slot] = h.to(ssm_pool.dtype)
    o = o.to(BF16).float()  # the recurrence entries' bf16 output
    y = o * torch.rsqrt(o.pow(2).mean(-1, keepdim=True) + core.eps) * L.gnorm.float()
    y = y * torch.sigmoid(z.reshape(-1, d.hv, V).float())
    return M.cublas_mm(y.reshape(-1, d.vd).to(BF16), L.out)


def _check(y, y_ref, core, core_ref, what) -> None:
    """Output and both pools against the reference's, at the module bands."""
    assert y.shape == y_ref.shape and y.dtype == y_ref.dtype
    assert torch.isfinite(y).all()
    diff = y.float() - y_ref.float()
    rel = (diff.norm() / y_ref.float().norm()).item()
    peak = (diff.abs().max() / y_ref.float().abs().max()).item()
    ssm, ssm_ref = core._bound.pools.ssm[0], core_ref._bound.pools.ssm[0]
    s_rel = ((ssm.float() - ssm_ref.float()).norm() / ssm_ref.float().norm()).item()
    print(f"{what}: out rel-L2 {rel:.3e}, out max-abs/peak {peak:.3e}, state rel-L2 {s_rel:.3e}")
    assert rel <= OUT_REL_L2 and peak <= OUT_MAX_FRAC, (rel, peak)
    assert torch.equal(core._bound.pools.conv[0], core_ref._bound.pools.conv[0]), (
        "conv pools differ"
    )
    torch.testing.assert_close(ssm.float(), ssm_ref.float(), rtol=RTOL_STATE, atol=ATOL_STATE)


def _case(geometry: str, state: str, seed: int):
    d = _dims(*GEOMETRIES[geometry])
    core = _core(d, seed, STATES[state], one_rank=geometry == "16x48")
    return d, core, _layer(d, seed + 1)


# ---------------------------------------------------------------------- the tests ---


@pytest.mark.parametrize("geometry, state", CASES, ids=[f"{g}-{s}" for g, s in CASES])
def test_decode_block(geometry, state) -> None:
    """A 128-row decode step; slots the step does not name are not written."""
    m = 128
    d, core, L = _case(geometry, state, 100)
    x = _x(m, 300)
    core_ref, before = _clone_core(core), _clone_core(core)
    si = _decode_slots(m)
    y = _target(M.DecodeTarget, core)._gdn_block(x, 0, L, si, None, 1)
    y_ref = _reference(d, core_ref, L, x, [(r, 1, s, True) for r, s in enumerate(_slots(m))])
    _check(y, y_ref, core, core_ref, f"decode m={m}, {geometry} {state} pool")
    unnamed = [s for s in range(SLOTS) if s not in set(_slots(m))]
    assert torch.equal(core._bound.pools.ssm[0][unnamed], before._bound.pools.ssm[0][unnamed])


def test_captured_one_rank_fp32_decode_block() -> None:
    """At one rank over an fp32 pool a captured step's recurrence and gated norm are one
    launch (`gdn_decode_gated_norm_pdl`); run eagerly here against the same reference."""
    m = 32
    d, core, L = _case("16x48", "fp32", 110)
    x = _x(m, 310)
    core_ref = _clone_core(core)
    y = _target(M.DecodeTarget, core)._gdn_block(x, 0, L, _decode_slots(m), None, 1, True)
    y_ref = _reference(d, core_ref, L, x, [(r, 1, s, True) for r, s in enumerate(_slots(m))])
    _check(y, y_ref, core, core_ref, f"captured-form decode m={m}, 16x48 fp32 pool")


def _prefill(core, L, x, step):
    tokens = x.shape[0]
    out = torch.empty([1, tokens, core.num_v_heads, V], dtype=BF16, device=DEV)
    target = _target(M.PrefillTarget, core, L, types.SimpleNamespace(gdn=step, num_tokens=tokens))
    return target._gdn_block(x, 0, L, out)


def _prefill_seqs(ctx_lens, n_dec, has_init) -> list:
    slots = _slots(len(ctx_lens) + n_dec)
    seqs, first = [], 0
    for n, (rows, carries) in enumerate(zip(ctx_lens, has_init)):
        seqs.append((first, rows, slots[n], carries))
        first += rows
    for j in range(n_dec):
        seqs.append((first + j, 1, slots[len(ctx_lens) + j], True))
    return seqs


@pytest.mark.parametrize(
    "ctx_lens, n_dec, has_init",
    [((200, 37, 91), 5, (True, False, True)), ((130, 64), 0, (True, False))],
    ids=["mixed", "context_only"],
)
@pytest.mark.parametrize("geometry, state", CASES, ids=[f"{g}-{s}" for g, s in CASES])
def test_prefill_block(geometry, state, ctx_lens, n_dec, has_init) -> None:
    """Variable-length contexts, with and without carried state, and (mixed) one-token
    generation rows in the same step."""
    tokens = sum(ctx_lens) + n_dec
    d, core, L = _case(geometry, state, 400 + tokens)
    x = _x(tokens, 600 + tokens)
    core_ref = _clone_core(core)
    y = _prefill(core, L, x, _prefill_step(ctx_lens, n_dec, has_init))
    y_ref = _reference(d, core_ref, L, x, _prefill_seqs(ctx_lens, n_dec, has_init))
    _check(y, y_ref, core, core_ref, f"prefill {ctx_lens}+{n_dec}x1, {geometry} {state} pool")


@pytest.mark.parametrize("geometry, state", CASES, ids=[f"{g}-{s}" for g, s in CASES])
def test_decode_block_replays_under_cuda_graph(geometry, state) -> None:
    """The decode block captured into a CUDA graph after one eager call (as the engine's
    warm-up runs it) replays to exactly the eager result."""
    m = 32
    _, core0, L = _case(geometry, state, 700)
    x = _x(m, 702)
    si = _decode_slots(m)
    eager_core = _clone_core(core0)
    y_eager = _target(M.DecodeTarget, eager_core)._gdn_block(x, 0, L, si, None, 1).clone()
    graph_core = _clone_core(core0)
    target = _target(M.DecodeTarget, graph_core)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        target._gdn_block(x, 0, L, si, None, 1)
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    pools, pools0 = graph_core._bound.pools, core0._bound.pools
    pools.conv[0].copy_(pools0.conv[0])
    pools.ssm[0].copy_(pools0.ssm[0])
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        y_graph = target._gdn_block(x, 0, L, si, None, 1)
    assert torch.equal(pools.ssm[0], pools0.ssm[0]), "capture must not execute"
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(y_graph, y_eager)
    assert torch.equal(pools.ssm[0], eager_core._bound.pools.ssm[0])
    assert torch.equal(pools.conv[0], eager_core._bound.pools.conv[0])
