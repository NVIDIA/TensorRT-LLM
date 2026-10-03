# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU certification matrix for the ``moe/k3_moe_front`` catalog entry and its ``K3MoeHeadWorkspace``, with the cells
of ``moe/k3_moe`` behind the front: ``trtllm::k3_moe_front``, then the ``k3_moe`` entry on a plain and on a head_flags
``K3MoeState`` (the head_flags pair: the front with ``publish=True`` releasing the workspace's ready words, k3_moe
acquiring them).

The front's correctness depends on state that outlives a call: the head workspace's two alternating Lamport buffers
(flags[0] says which one a call uses; every call flips it), the readers' re-arm of every word they read, and, with a
head_flags build of k3_moe, the ready words and their epoch (flags[2]), which that k3_moe advances. So beyond single
calls this drives call sequences: layers x steps with the token count dipping and growing back and a random rank late
at every call, two workspaces interleaved, CUDA-graph capture and replay mixed with eager calls, the epoch across both
int32 wraps, the capture refusals, and last a negative control in which one rank swaps two calls and every rank gets a
wrong (and exactly predictable) answer without an error.

    CUDA_VISIBLE_DEVICES=0,1,2,3 python _k3_moe_front_op_matrix.py [--world-size 4]
    srun -N 4 --ntasks-per-node 4 --mpi=pmix python _k3_moe_front_op_matrix.py --launcher srun --world-size 16

Not a pytest module: one fixed sequence of checks inside one W-rank job (they share the workspaces, the states and
their counters). The collected entry point is ``moe/test_modeling_v2_k3_moe_front_op_matrix.py``.

Shapes. The head is row-sharded over the run's W ranks: 3584 / W latent rows and 896 / W router rows per rank (1120 at
W 4; 280 at W 16, Kimi K3 TP16's). The shared activation is TP16's per-rank width at every W (384 columns: two shared
experts of 3072 over 16 ranks). The routed experts are one rank of experts TP4 x EP4 (224 local experts, intermediate
768; rank r's are global ids [224 (r % 4), 224 (r % 4) + 224)), random checkpoint-format MXFP4 put through TRT-LLM's
own TRTLLM-Gen loader.

References. Every rank draws every rank's head slice and every call's input from one seed, so each rank holds the
whole reference. The payloads are exact: x is a multiple of 1/8 in [-1/4, 1/4], the latent-down and shared rows
multiples of 1/16 in [-1/8, 1/8], the router rows multiples of 1/8 in [-1/4, 1/4] (_lockstep.exact_bf16), so every
head and shared sum over K = 7168 is a multiple of 1/128 below 2^9 and exact in fp32 in any summation order: the
front's split-K sums equal the reference's. The reference is the head in fp64 (exact, then fp32; the latent columns
rounded to bf16), gathered, and the stock trtllm::kimi_k3_noaux_tc_mxfp8_quant on it: the front's routing ids and
weights, MXFP8 codes and scales must equal it bit for bit (the front routes with k3_route_quant's selection, weights
and quantization: the kernel's statement), and be bit for bit the same on every rank. The shared activation: this
rank's gate_up in fp64 (exact) rounded to bf16, SiTU in fp32, within 2e-2 of its largest magnitude (the kernel's tanh
and sigmoid are fast approximations). The routed partial of the front + k3_moe pair against the stock TRTLLM-Gen
W4A8_MXFP4_MXFP8 runner on the front's own routing and latent (op-catalog gates: 8 bf16 ulp of the row's max per
element, 4 ulp relative RMS); the head_flags pair bit for bit against the plain one. Call sequences compare every
call bit for bit with the same call made alone (itself checked against the references first).
"""

import math
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _lockstep as ls  # noqa: E402

assert torch.cuda.is_available(), "k3_moe_front requires CUDA devices"

DEADLINE_S = 1200
HIDDEN = 7168  # the MoE input's width, the front's K
LATENT = 3584
EXPERTS = 896
TOP_K = 16
SV = 32
# Kimi K3 TP16's per-rank shared activation: two shared experts of 3072 over 16 ranks.
SHARED_COLS = 384
# Kimi K3's SiTU caps: the front's shared-activation arguments, and k3_moe's build constants.
GATE_CAP, LINEAR_CAP = 4.0, 25.0
RSF = 2.827
# One rank of the routed experts' TP4 x EP4.
I_TP, E_LOCAL, MOE_TP = 768, 224, 4
# 0x80000000: an empty head workspace word.
EMPTY = -(2**31)
# The head workspace per rank: 157,696 int32 words at W 4, 8 and 16.
WORKSPACE_WORDS = 2 * 8 * (LATENT // 8 + EXPERTS // 4) * 4 + 2 * 8 * 8 * EXPERTS
ULP = 2.0**-8
SHARED_TOL = 2e-2
TOKENS = (1, 2, 3, 4, 5, 6, 7, 8)
LAYERS = 3
# Layer l of a step: the front then k3_moe on the plain state, the publishing front then k3_moe on the head_flags state,
# and the front alone.
KINDS = ("fused", "flags", "front")
DIP_STEPS = (8, 8, 8, 2, 7, 8, 1, 1, 8, 3, 8)
REPLAYS = 6

R = None
OPS = None
WS_A = None
WS_B = None
PLAIN = None  # the plain K3MoeState and its layers
FLAGS = None  # the head_flags K3MoeState and its layers
LAYER_WEIGHTS = []  # per layer: every rank's head slice, this rank's gate_up and front weight
BIAS = None
EXPERT_BUFFERS = None
OFFSET = 0
WL = WE = 0  # latent columns and experts per rank
STATS = {
    "y_elt_ulp": 0.0,
    "y_rms_ulp": 0.0,
    "shared_err": 0.0,
    "fronts_checked": 0,
    "rows_bits_as_m8": 0,
    "rows_checked": 0,
}


def i32(v: int) -> int:
    return (v + 2**31) % 2**32 - 2**31


def bits(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().view(torch.uint8)


def same(a: torch.Tensor, b: torch.Tensor) -> bool:
    return a.shape == b.shape and a.dtype == b.dtype and torch.equal(bits(a), bits(b))


def same_result(got, want) -> bool:
    return len(got) == len(want) and all(same(a, b) for a, b in zip(got, want))


def same_bits_on_ranks(*tensors: torch.Tensor) -> bool:
    """Every tensor bit for bit the same on every rank (the raw bytes, gathered)."""
    mine = [bits(t).cpu().numpy().tobytes() for t in tensors]
    every = R.comm.allgather(mine)
    return all(other == every[0] for other in every[1:])


def max_ulp(a: torch.Tensor, b: torch.Tensor) -> int:
    """Largest distance in bf16 ulps between two bf16 tensors (bit patterns as ordered integers)."""

    def ordered(x):
        i = x.contiguous().view(torch.int16).int()
        return torch.where(i < 0, -(i & 0x7FFF), i)

    return int((ordered(a) - ordered(b)).abs().max().item()) if a.numel() else 0


class Call:
    """One call: the MoE input ``x`` (the same on every rank), the layer whose weights it uses, and its kind.

    Kinds: "front" (the k3_moe_front entry, five outputs), "fused" (the entry, then the k3_moe entry on the plain
    K3MoeState's layer: ``(y, shared)``), "flags" (the publishing front, then the k3_moe entry on the head_flags
    K3MoeState's layer with ``head=ws``: ``(y, shared)``).
    """

    def __init__(self, seed: int, tokens: int, layer: int = 0, kind: str = "front", x=None):
        if x is None:
            g = torch.Generator(device="cuda").manual_seed(seed)
            x = ls.exact_bf16(g, (tokens, HIDDEN), -2, 3, 1 / 8)
        self.x, self.layer, self.kind = x, layer, kind

    def first(self, m: int, kind=None) -> "Call":
        """The first ``m`` tokens of this call's input, as a call of ``kind`` (default: this call's)."""
        return Call(0, m, self.layer, kind or self.kind, x=self.x[:m].contiguous())

    def as_kind(self, kind: str) -> "Call":
        return Call(0, 0, self.layer, kind, x=self.x)

    def run(self, ws, x=None):
        x = self.x if x is None else x
        lw = LAYER_WEIGHTS[self.layer]
        if self.kind == "flags":
            ids, w, q, s, shared = OPS.front(
                x, lw.front, BIAS, RSF, SHARED_COLS, GATE_CAP, LINEAR_CAP, ws, publish=True
            )
            return OPS.k3_moe(q, s, ids, w, OFFSET, FLAGS.layers[self.layer], head=ws), shared
        out = OPS.front(x, lw.front, BIAS, RSF, SHARED_COLS, GATE_CAP, LINEAR_CAP, ws)
        if self.kind == "front":
            return out
        ids, w, q, s, shared = out
        return OPS.k3_moe(q, s, ids, w, OFFSET, PLAIN.layers[self.layer]), shared


# ── references ────────────────────────────────────────────────────────────


def front_reference(x, layer: int, xs=None):
    """The unfused front for input ``x`` with layer ``layer``'s weights.

    Every rank's head slice of its input (``xs[r]``; default ``x`` on every rank) in fp64 (exact for these payloads),
    then fp32, gathered (latent columns rounded to bf16); the stock routing + MXFP8 quantization of the gathered head;
    this rank's shared activation. Returns (ids, weights, quantized, scales, shared).
    """
    lw = LAYER_WEIGHTS[layer]
    xs = xs or [x] * R.world
    heads = [(xr.double() @ w.double().t()).float() for xr, w in zip(xs, lw.heads)]
    latent = torch.cat([h[:, :WL] for h in heads], dim=1).bfloat16().contiguous()
    logits = torch.cat([h[:, WL:] for h in heads], dim=1).contiguous()
    ids, w, q, s = torch.ops.trtllm.kimi_k3_noaux_tc_mxfp8_quant(logits, BIAS, latent, RSF)
    gate_up = (xs[R.rank].double() @ lw.gate_up.double().t()).float().bfloat16().float()
    gate, up = gate_up[:, :SHARED_COLS], gate_up[:, SHARED_COLS:]
    shared = (
        GATE_CAP
        * torch.tanh(gate / GATE_CAP)
        * torch.sigmoid(gate)
        * (LINEAR_CAP * torch.tanh(up / LINEAR_CAP))
    )
    return ids, w, q, s, shared.bfloat16()


def verify_front(got, ref, where: str) -> None:
    """The front's five outputs against ``ref`` (see the module docstring), and bit for bit across the ranks."""
    ids, w, q, s, shared = got
    m = ids.shape[0]
    assert ids.dtype == torch.int32 and tuple(ids.shape) == (m, TOP_K), where
    assert w.dtype == torch.bfloat16 and tuple(w.shape) == (m, TOP_K), where
    assert q.dtype == torch.float8_e4m3fn and tuple(q.shape) == (m, LATENT), where
    assert s.dtype == torch.uint8 and tuple(s.shape) == (m, LATENT // SV), where
    assert shared.dtype == torch.bfloat16 and tuple(shared.shape) == (m, SHARED_COLS), where
    names = ("topk_ids", "topk_weights", "quantized", "scales")
    for name, a, b in zip(names, got[:4], ref[:4]):
        if not same(a, b):
            unequal = int((bits(a) != bits(b)).sum()) if a.shape == b.shape else -1
            raise AssertionError(f"{where}: {name} differs from the reference in {unequal} bytes")
    err = ls.rel_err(shared, ref[4])
    STATS["shared_err"] = max(STATS["shared_err"], err)
    STATS["fronts_checked"] += 1
    assert err <= SHARED_TOL, f"{where}: shared activation rel err {err:.3e} > {SHARED_TOL}"
    assert same_bits_on_ranks(ids, w, q, s), f"{where}: ranks disagree on the routing or the latent"


def runner(ids, w, q, s):
    """The stock TRTLLM-Gen W4A8_MXFP4_MXFP8 MoE over this rank's experts, pre-routed (SiTU with Kimi K3's caps)."""
    from tensorrt_llm._torch.moe.fused_moe.routing import RoutingMethodType
    from tensorrt_llm._torch.utils import ActType_TrtllmGen

    p = EXPERT_BUFFERS
    alpha = torch.full((E_LOCAL,), GATE_CAP, dtype=torch.float32, device="cuda")
    beta = torch.full((E_LOCAL,), LINEAR_CAP, dtype=torch.float32, device="cuda")
    return torch.ops.trtllm.mxe4m3_mxe2m1_block_scale_moe_runner(
        None, None, q, s.view(-1), p["w31"], p["w31s"], None, alpha, beta, None, p["w2"], p["w2s"], None, EXPERTS,
        TOP_K, 1, 1, I_TP, LATENT, I_TP, OFFSET, E_LOCAL, 1.0, int(RoutingMethodType.DeepSeekV3),
        int(ActType_TrtllmGen.SiTu), topk_weights=w, topk_ids=ids,
    )  # fmt: skip


def compare(y, ref):
    """Op-catalog gates: |d| <= 8 ulp of the row's max |ref| per element, relative RMS <= 4 ulp; finite."""
    o, r = y.float(), ref.float()
    row = r.abs().amax(dim=1, keepdim=True).clamp_min(1e-12)
    elt = ((o - r).abs() / row).max().item() / ULP
    rms = ((o - r).pow(2).mean().sqrt() / r.pow(2).mean().sqrt().clamp_min(1e-12)).item() / ULP
    return elt, rms, bool(torch.isfinite(o).all()) and elt <= 8.0 and rms <= 4.0


def verify_fused(got, call: Call, ws, where: str) -> None:
    """The (y, shared) of a front + k3_moe pair for ``call``.

    The front alone on the same input (made here, on ``ws``, by every rank) gives the routing and MXFP8 latent k3_moe
    consumed: y within the op-catalog gates of the stock runner on them (all zeros on a rank no token routes to), and
    shared the front's bits.
    """
    y, shared = got
    m = call.x.shape[0]
    assert y.dtype == torch.bfloat16 and tuple(y.shape) == (m, LATENT) and y.is_contiguous(), where
    ids, w, q, s, f_shared = call.as_kind("front").run(ws)
    assert same(shared, f_shared), f"{where}: the shared activation differs from the front's"
    if not bool(((ids >= OFFSET) & (ids < OFFSET + E_LOCAL)).any()):
        assert bool((y == 0).all()), f"{where}: no token routes to this rank, y is not zero"
        return
    elt, rms, ok = compare(y, runner(ids, w, q, s))
    STATS["y_elt_ulp"] = max(STATS["y_elt_ulp"], elt)
    STATS["y_rms_ulp"] = max(STATS["y_rms_ulp"], rms)
    assert ok, f"{where}: y against the stock runner {elt:.2f} ulp per element, {rms:.2f} ulp RMS"


# ── state ─────────────────────────────────────────────────────────────────


def between_barriers(fn):
    """fn() with every rank's kernels done before it and no rank's next call started until every rank has run it.

    Peers push into this rank's head buffers in their calls, so its words are read only there.
    """
    R.barrier()
    value = fn()
    R.barrier()
    return value


def workspace_rearmed(ws) -> bool:
    """Every word of this rank's head buffers empty, flags[1] and the sign-in count flags[3] zero."""
    return between_barriers(
        lambda: bool((ws.uc == EMPTY).all()) and int(ws.flags[1]) == 0 and int(ws.flags[3]) == 0
    )


def epoch_and_words(ws):
    """This rank's head epoch (flags[2]) and its 16 ready words."""
    return between_barriers(lambda: (int(ws.flags[2]), ws.ready[:16].tolist()))


def assert_ready_at(ws, want: int, where: str) -> None:
    epoch, words = epoch_and_words(ws)
    assert epoch == want and all(v == want for v in words), (
        f"{where}: epoch {epoch}, ready {words}, want {want}"
    )


def workspace_snapshot(ws):
    return between_barriers(lambda: [ws.uc.clone(), ws.flags.clone(), ws.ready.clone()])


def moe_rearmed() -> bool:
    """Both K3MoeStates' slabs armed and every layer's counters zero (rank-local state)."""
    torch.cuda.synchronize()
    for st in (PLAIN, FLAGS):
        mod = st.state.mod
        cs = st.state.cs.view(mod.G_CAP, 8, mod.K2_TILES, mod.SFB_GROUP_BYTES)
        if not (bool((st.state.c == -128).all()) and bool((cs[..., :4] == -1).all())):
            return False
        if not all(bool((layer.counters == 0).all()) for layer in st.layers):
            return False
    return True


def flags_call_checked(call: Call, ws, where: str):
    """A head_flags call with its handoff checked on every rank, before and after.

    Before: no ready word the call polls (ids [t], MXFP8 row [8 + t], t < M) already holds epoch + 1; else k3_moe could
    read the routing before the front writes it, and the call is not made. After: the epoch is epoch + 1 and every one
    of the 16 ready words holds it.
    """
    epoch, words = epoch_and_words(ws)
    want = i32(epoch + 1)
    m = call.x.shape[0]
    early = [i for i in [*range(m), *range(8, 8 + m)] if words[i] == want]
    assert R.all_true(not early), (
        f"{where}: ready words {early} already hold {want}; the call would race"
    )
    got = call.run(ws)
    torch.cuda.synchronize()
    assert_ready_at(ws, want, where)
    return got


def alone_results(calls, ws, where: str):
    """Each call made alone (after a barrier) and checked; returns their outputs.

    The front against the unfused reference; the fused calls by verify_fused; head_flags calls also by
    flags_call_checked.
    """
    out = []
    for k, call in enumerate(calls):
        tag = f"{where} {k} ({call.kind}, layer {call.layer}, M {call.x.shape[0]}) alone"
        R.barrier()
        if call.kind == "flags":
            got = flags_call_checked(call, ws, tag)
        else:
            got = call.run(ws)
            torch.cuda.synchronize()
        if call.kind == "front":
            verify_front(got, front_reference(call.x, call.layer), tag)
        else:
            verify_fused(got, call, ws, tag)
        out.append(got)
    return out


# ── checks ────────────────────────────────────────────────────────────────


def check_workspace_is_armed_and_sized() -> None:
    """After create(): 157,696 int32 words per rank behind one multicast mapping, every word empty.

    Flags and ready words zero; the front supports this W at the certified widths (checked at setup); the two
    K3MoeStates are the plain and the head_flags build.
    """
    from tensorrt_llm._torch.cute_dsl_kernels.k3_fused_moe import k3_route_quant_ag as layout

    assert layout.workspace_words(R.world) == WORKSPACE_WORDS == 157_696
    for ws in (WS_A, WS_B):
        assert ws.rank == R.rank and ws.world_size == R.world
        assert ws.uc.dtype == ws.mc.dtype == ws.flags.dtype == ws.ready.dtype == torch.int32
        assert ws.uc.numel() == ws.mc.numel() == WORKSPACE_WORDS
        assert tuple(ws.flags.shape) == (4,) and tuple(ws.ready.shape) == (32,)
        assert between_barriers(lambda ws=ws: bool((ws.uc == EMPTY).all())), "every word empty"
        assert ws.flags.tolist() == [0, 0, 0, 0] and not bool(ws.ready.any()), (
            "flags and ready words zero"
        )
    assert not PLAIN.state.head_flags and FLAGS.state.head_flags


def check_front_single_calls() -> None:
    """The front alone at every M 1-8 (the first M tokens of one 8-token batch) against the unfused reference.

    Each call's outputs the bits of the same rows of the 8-token call and run-to-run identical (routing and latent
    also bit for bit on every rank, in verify_front); after each call this rank's head buffers empty, the sign-in
    count zero, flags[0] flipped once, and the epoch and ready words untouched: the standalone front publishes nothing.
    """
    base = Call(100, 8, layer=0, kind="front")
    out8 = base.run(WS_A)
    torch.cuda.synchronize()
    verify_front(out8, front_reference(base.x, 0), "front M 8 (base)")
    for m in TOKENS:
        call = base.first(m)
        epoch_words = epoch_and_words(WS_A)
        flag0 = between_barriers(lambda: int(WS_A.flags[0]))
        got = call.run(WS_A)
        torch.cuda.synchronize()
        flag1 = between_barriers(lambda: int(WS_A.flags[0]))
        assert flag1 == flag0 ^ 1, f"front M {m}: buffer index {flag0} -> {flag1}"
        assert workspace_rearmed(WS_A), f"front M {m}: head buffers not re-armed after the call"
        verify_front(got, front_reference(call.x, 0), f"front M {m}")
        again = [call.run(WS_A) for _ in range(2)]
        torch.cuda.synchronize()
        assert all(same_result(a, got) for a in again), f"front M {m}: not run-to-run identical"
        assert same_result(got, [t[:m] for t in out8]), (
            f"front M {m}: rows differ from the 8-token call's"
        )
        assert workspace_rearmed(WS_A), f"front M {m}: head buffers not re-armed after the re-runs"
        assert epoch_and_words(WS_A) == epoch_words, (
            f"front M {m}: the standalone front moved the ready words"
        )


def check_front_and_k3_moe_single_calls() -> None:
    """The front then the k3_moe entry at every M 1-8, on the plain K3MoeState and on the head_flags one.

    Layer 0's weights, the first M tokens of one batch. The plain y within the op-catalog gates of the stock runner
    on the front's own routing and latent (all zeros on a rank no token routes to), the shared activation the front's
    bits; the head_flags pair's y and shared the plain pair's bits, its handoff checked before and after
    (flags_call_checked); the plain y's rows within one bf16 ulp of the same rows of the 8-token call (k3_moe's slice
    FC2 groups a token's expert terms by the step's group count; bit-identity counted) and the shared rows their bits;
    run-to-run identical bits; afterwards both states armed with every counter zero and the head buffers empty.
    """
    base = Call(200, 8, layer=0, kind="fused")
    y8, sh8 = base.run(WS_A)
    torch.cuda.synchronize()
    verify_fused((y8, sh8), base, WS_A, "fused M 8 (base)")
    for m in TOKENS:
        plain = base.first(m)
        got = plain.run(WS_A)
        torch.cuda.synchronize()
        verify_fused(got, plain, WS_A, f"fused M {m}")
        y, sh = got
        y_flags, sh_flags = flags_call_checked(plain.as_kind("flags"), WS_A, f"head_flags M {m}")
        assert same(y_flags, y) and same(sh_flags, sh), (
            f"M {m}: the head_flags build differs from the plain one"
        )
        ulp = max_ulp(y, y8[:m])
        STATS["rows_checked"] += 1
        STATS["rows_bits_as_m8"] += int(same(y, y8[:m]))
        assert ulp <= 1, f"fused M {m}: rows {ulp} ulp from the 8-token call's"
        assert same(sh, sh8[:m]), f"fused M {m}: shared rows differ from the 8-token call's"
        rerun = plain.run(WS_A)
        torch.cuda.synchronize()
        assert same_result(rerun, got), f"fused M {m}: not run-to-run identical"
        assert moe_rearmed(), f"fused M {m}: a k3_moe slab is not armed or a counter is not zero"
        assert workspace_rearmed(WS_A), f"fused M {m}: head buffers not re-armed"


def check_head_flags_epoch_wraps() -> None:
    """The ready-word handoff across both int32 wraps of the epoch (check_head_flags of the kernel test, extended).

    From a new workspace's state (ready words zeroed, epoch 0): calls at M 1, 1; the epoch preset to -2 (as 2^32 - 2
    calls later): calls at M 1, 8, 3, 8, so that the M 8 call at epoch -1 waits for 0, the value a word past an earlier
    call's tokens would still hold had k3_moe not re-armed it; then the epoch preset to 2^31 - 2: calls at M 8, 1, 8,
    across 2^31 - 1 -> -2^31. Every call checked by flags_call_checked, its y and shared the plain build's bits; the
    head buffers empty afterwards.
    """
    base = Call(300, 8, layer=0, kind="fused")
    plain = {m: base.first(m).run(WS_A) for m in (1, 3, 8)}
    torch.cuda.synchronize()

    def preset(epoch, zero_words=False):
        R.barrier()
        if zero_words:
            WS_A.ready.zero_()
        WS_A.flags[2] = epoch
        R.barrier()

    preset(0, zero_words=True)
    for item in (1, 1, ("epoch", -2), 1, 8, 3, 8, ("epoch", 2**31 - 2), 8, 1, 8):
        if isinstance(item, tuple):
            preset(item[1])
            continue
        epoch = epoch_and_words(WS_A)[0]
        got = flags_call_checked(base.first(item, "flags"), WS_A, f"epoch {epoch} M {item}")
        assert same_result(got, plain[item]), (
            f"epoch {epoch} M {item}: differs from the plain build"
        )
    assert workspace_rearmed(WS_A)


def check_dip_and_regrow_sequence() -> None:
    """Decode steps of three layers on one workspace, the token count dipping and growing back, a random rank late.

    Each step: the front then k3_moe on the plain state (layer 0's weights), the publishing front then k3_moe on the
    head_flags state (layer 1's), and the front alone (layer 2's), at M 8, 8, 8, 2, 7, 8, 1, 1, 8, 3, 8 with new inputs
    every call, the layers back to back
    with a random rank 5 ms late before every call. Every call returns the bits of the same call made alone (each
    checked first). Afterwards the head buffers are empty, the epoch advanced once per head_flags call with every
    ready word at it, and both states armed: a call after a smaller one reads nothing an older, larger call left.
    """
    calls = [
        Call(4000 + 10 * s + layer, m, layer=layer, kind=KINDS[layer])
        for s, m in enumerate(DIP_STEPS)
        for layer in range(LAYERS)
    ]
    alone = alone_results(calls, WS_A, "dip")
    epoch0 = epoch_and_words(WS_A)[0]
    late = random.Random(7)
    got = []
    for s in range(len(DIP_STEPS)):
        R.barrier()
        for layer in range(LAYERS):
            R.late(late.randrange(R.world))
            got.append(calls[s * LAYERS + layer].run(WS_A))
    torch.cuda.synchronize()
    bad = [k for k, (g, a) in enumerate(zip(got, alone)) if not same_result(g, a)]
    assert not bad, f"calls {bad} of the sequence differ from the same calls alone"
    assert workspace_rearmed(WS_A), "head buffers not re-armed after the sequence"
    assert_ready_at(WS_A, i32(epoch0 + len(DIP_STEPS)), "after the sequence")
    assert moe_rearmed(), "a k3_moe slab is not armed or a counter is not zero after the sequence"


def check_two_workspaces_interleaved() -> None:
    """Two workspaces are two rotations and two epochs.

    20 calls alternate between WS_A and WS_B in an irregular pattern (A A B A B B A A A B, twice), kinds and layers
    cycling, M in 3, 8, 1, 8, 5, back to back: every call returns the bits of the same call alone; each workspace's
    epoch advanced by its own head_flags calls only, every ready word at it; both workspaces' buffers empty. The
    pattern is the same on every rank: calls on one stream run one after the other and each waits for its peers, so
    ranks ordering calls on two workspaces differently would deadlock (not exercised).
    """
    pattern = "AABABBAAAB" * 2
    calls = [
        Call(5000 + i, (3, 8, 1, 8, 5)[i % 5], layer=i % LAYERS, kind=KINDS[i % LAYERS])
        for i in range(len(pattern))
    ]
    alone = alone_results(calls, WS_A, "interleaved")
    epoch_a, epoch_b = epoch_and_words(WS_A)[0], epoch_and_words(WS_B)[0]
    R.barrier()
    got = [c.run(WS_A if which == "A" else WS_B) for c, which in zip(calls, pattern)]
    torch.cuda.synchronize()
    bad = [k for k, (g, a) in enumerate(zip(got, alone)) if not same_result(g, a)]
    assert not bad, f"interleaved calls {bad} differ from the same calls alone"
    flag_calls = {
        w: sum(c.kind == "flags" and p == w for c, p in zip(calls, pattern)) for w in "AB"
    }
    assert_ready_at(WS_A, i32(epoch_a + flag_calls["A"]), "WS_A after the interleaving")
    assert_ready_at(WS_B, i32(epoch_b + flag_calls["B"]), "WS_B after the interleaving")
    assert workspace_rearmed(WS_A) and workspace_rearmed(WS_B)
    assert moe_rearmed()


def check_graph_capture_and_replay() -> None:
    """A captured step replayed with rewritten inputs, eager calls of other token counts between replays.

    The step: the three layers at M 8 on WS_B (the front + k3_moe pairs, plain and head_flags, and the front alone),
    captured once and replayed 6 times with new inputs copied into its static buffers; between replays an eager call of
    another M on WS_B. Every replayed and eager call returns the bits of the same call alone; afterwards the head
    buffers are empty, WS_B's epoch advanced once per head_flags call, replayed or eager, with every ready word at it,
    and both states armed.
    """
    static_bufs = [
        torch.zeros(8, HIDDEN, dtype=torch.bfloat16, device="cuda") for _ in range(LAYERS)
    ]
    shells = [
        Call(0, 0, layer=layer, kind=KINDS[layer], x=static_bufs[layer]) for layer in range(LAYERS)
    ]

    def step():
        return [shell.run(WS_B) for shell in shells]

    warm = [Call(6000 + layer, 8, layer, KINDS[layer]) for layer in range(LAYERS)]
    reps = [
        [Call(6100 + 10 * r + layer, 8, layer, KINDS[layer]) for layer in range(LAYERS)]
        for r in range(REPLAYS)
    ]
    eagers = [
        Call(6500 + r, (3, 1, 6, 5, 2, 7)[r], r % LAYERS, KINDS[r % LAYERS]) for r in range(REPLAYS)
    ]
    alone_reps = [alone_results(rep, WS_A, f"replay {r}") for r, rep in enumerate(reps)]
    alone_eager = alone_results(eagers, WS_A, "eager between replays")

    epoch0 = epoch_and_words(WS_B)[0]
    for static, c in zip(static_bufs, warm):
        static.copy_(c.x)
    R.barrier()
    step()  # every first call of this step eager
    R.barrier()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        outs = step()
    R.barrier()
    flag_calls = 1  # the warm-up step's
    for r in range(REPLAYS):
        for static, c in zip(static_bufs, reps[r]):
            static.copy_(c.x)
        R.barrier()
        graph.replay()
        torch.cuda.synchronize()
        bad = [
            layer for layer in range(LAYERS) if not same_result(outs[layer], alone_reps[r][layer])
        ]
        assert not bad, f"replay {r}: layers {bad} differ from the same calls alone"
        got = eagers[r].run(WS_B)
        torch.cuda.synchronize()
        assert same_result(got, alone_eager[r]), (
            f"eager call after replay {r} differs from the same call alone"
        )
        flag_calls += 1 + int(eagers[r].kind == "flags")
    del graph
    assert workspace_rearmed(WS_B), "head buffers not re-armed after the replays"
    assert_ready_at(WS_B, i32(epoch0 + flag_calls), "WS_B after the replays")
    assert moe_rearmed()


def check_unsupported_shape_raises_on_every_rank() -> None:
    """M 0 and 9 raise ValueError on every rank before touching anything, and the next call is correct.

    The front alone and the two front + k3_moe pairs (plain, and head_flags with the publishing front) at M 0 and 9:
    the front refuses them before touching the workspace (its words, flags and ready words keep their bits); the next
    call returns the bits of the same call made alone.
    """
    nxt = Call(7000, 8, layer=0, kind="fused")
    want = alone_results([nxt], WS_A, "before the unsupported calls")[0]
    before = workspace_snapshot(WS_A)
    raised = []
    for m in (0, 9):
        for kind in KINDS:
            try:
                Call(7100 + m, m, layer=0, kind=kind).run(WS_A)
                raised.append(False)
            except ValueError:
                raised.append(True)
    assert R.all_true(all(raised)), (
        f"M 0 / 9 did not raise ValueError on every rank and kind: {raised}"
    )
    after = workspace_snapshot(WS_A)
    assert all(same(a, b) for a, b in zip(before, after)), (
        "a refused call touched the head workspace"
    )
    R.barrier()
    got = nxt.run(WS_A)
    torch.cuda.synchronize()
    assert same_result(got, want), (
        "the call after the refused ones differs from the same call alone"
    )


def raised_under_capture(fn) -> str:
    """Run ``fn`` under CUDA-graph capture; the message of the RuntimeError it raised, or '' if it raised none."""
    graph, stream = torch.cuda.CUDAGraph(), torch.cuda.Stream()
    resting = torch.cuda.current_stream()
    stream.wait_stream(resting)
    message = ""
    try:
        with torch.cuda.graph(graph, stream=stream):
            try:
                fn()
            except RuntimeError as exc:
                message = str(exc)
    finally:
        # A capture that fails when it ends leaves its own stream current; put the resting one back.
        torch.cuda.set_stream(resting)
    del graph
    return message


def raised_eagerly(fn) -> str:
    """Run ``fn``; the message of the RuntimeError it raised, or '' if it raised none."""
    try:
        fn()
    except RuntimeError as exc:
        return str(exc)
    return ""


def check_create_and_first_compile_refuse_capture() -> None:
    """Under CUDA-graph capture, create() is refused on every rank, and a not yet compiled front configuration raises.

    K3MoeHeadWorkspace.create has the ranks agree before allocating that each of them can (not capturing, enough free
    memory): with every rank capturing, and with the last rank capturing while its peers call it eagerly, every rank
    raises RuntimeError at that agreement ("not every rank can allocate", a capturing rank's message naming the
    capture), so none allocates, and each frees the communicator made for the call. Every rank must call it: a
    rank calling it alone would wait for its peers. A front call of a configuration not yet compiled (other SiTU caps)
    raises RuntimeError under capture before any launch. The workspace keeps its bits and the next call returns the
    bits of the same call made alone.
    """
    from tensorrt_llm._torch.distributed import ops

    nxt = Call(7500, 4, layer=1, kind="front")
    want = alone_results([nxt], WS_A, "before the capture refusals")[0]
    before = workspace_snapshot(WS_A)

    def create():
        OPS.workspace.create(R.mapping, fabric_handle=R.fabric)

    make = ops._get_mnnvl_tp_group_comm
    comms = []

    def recording_make(mapping):
        comms.append(make(mapping))
        return comms[-1]

    capturing = R.world - 1
    ops._get_mnnvl_tp_group_comm = recording_make
    try:
        every = raised_under_capture(create)
        R.barrier()
        one = raised_under_capture(create) if R.rank == capturing else raised_eagerly(create)
    finally:
        ops._get_mnnvl_tp_group_comm = make
    R.barrier()
    first = raised_under_capture(
        lambda: OPS.front(
            nxt.x, LAYER_WEIGHTS[1].front, BIAS, RSF, SHARED_COLS, GATE_CAP + 1.0, LINEAR_CAP, WS_A
        )
    )
    refused = (
        "not every rank can allocate" in every
        and "outside CUDA-graph capture" in every
        and "not every rank can allocate" in one
        and (R.rank != capturing or "outside CUDA-graph capture" in one)
        and "outside CUDA-graph capture" in first
    )
    assert R.all_true(refused), (
        f"rank {R.rank}: every rank capturing {every!r}; rank {capturing} capturing {one!r}; "
        f"uncompiled front {first!r}"
    )
    freed = len(comms) == 2 and all(c == R.MPI.COMM_NULL for c in comms)
    assert R.all_true(freed), "a refused create kept the communicator made for it"
    after = workspace_snapshot(WS_A)
    assert all(same(a, b) for a, b in zip(before, after)), (
        "a refused call touched the head workspace"
    )
    R.barrier()
    got = nxt.run(WS_A)
    torch.cuda.synchronize()
    assert same_result(got, want), (
        "the call after the capture refusals differs from the same call alone"
    )


def check_wrong_call_order_is_detected() -> None:
    """Negative control: rank 0 swaps two same-shaped front calls on one workspace.

    Rank 0 runs c2 then c1, its peers c1 then c2. Nothing raises or hangs (the rotation positions still agree), but
    each call pairs with the peers' call at the same position: every rank's gathered head mixes rank 0's slice of the
    other input with the peers' slices of this one. Every rank returns the same wrong routing and latent, exactly the
    unfused reference of those mixed inputs; the MXFP8 latent's columns [0, 3584 / W) and their scales are bit for bit
    those of rank 0's input's call and the rest those of the peers' input's call; each rank's latent differs from that
    of the call it made in more than half of the mixed-in columns. The shared activation (rank-local) is each rank's
    own input's, bit for bit. A plain call right after is correct again.
    """
    c1, c2 = Call(8000, 8, layer=0, kind="front"), Call(8001, 8, layer=0, kind="front")
    o1, o2 = alone_results([c1, c2], WS_A, "the ordered pair")
    R.barrier()
    if R.rank == 0:
        first, second = c2.run(WS_A), c1.run(WS_A)
    else:
        first, second = c1.run(WS_A), c2.run(WS_A)
    torch.cuda.synchronize()
    sc = WL // SV
    for pos, got, (x0, xp), (o0, op) in (
        (1, first, (c2.x, c1.x), (o2, o1)),
        (2, second, (c1.x, c2.x), (o1, o2)),
    ):
        where = f"swapped pair, position {pos}"
        xs = [x0] + [xp] * (R.world - 1)  # each rank's input at this position
        verify_front(got, front_reference(xp, 0, xs=xs), where)
        q, s = got[2], got[3]
        assert same(q[:, :WL], o0[2][:, :WL]) and same(q[:, WL:], op[2][:, WL:]), (
            f"{where}: latent not the mix"
        )
        assert same(s[:, :sc], o0[3][:, :sc]) and same(s[:, sc:], op[3][:, sc:]), (
            f"{where}: scales not the mix"
        )
        # The call this rank made at this position, correctly ordered, and the columns the swap mixed into it.
        made = o0 if R.rank == 0 else op
        mixed_in = slice(WL, LATENT) if R.rank == 0 else slice(0, WL)
        wrong = (
            (q[:, mixed_in].view(torch.uint8) != made[2][:, mixed_in].view(torch.uint8))
            .float()
            .mean()
            .item()
        )
        assert wrong > 0.5, (
            f"{where}: only {wrong:.3f} of the mixed-in latent differs from the call made"
        )
        assert same(got[4], made[4]), (
            f"{where}: the shared activation is not this rank's own input's"
        )
        if R.rank == 0 and pos == 1:
            print(
                f"[rank 0] swapped pair: {wrong:.3f} of the mixed-in latent codes wrong", flush=True
            )
    c3 = Call(8002, 8, layer=0, kind="front")
    R.barrier()
    got = c3.run(WS_A)
    torch.cuda.synchronize()
    verify_front(got, front_reference(c3.x, 0), "after the swapped pair")


CHECKS = [
    check_workspace_is_armed_and_sized,
    check_front_single_calls,
    check_front_and_k3_moe_single_calls,
    check_head_flags_epoch_wraps,
    check_dip_and_regrow_sequence,
    check_two_workspaces_interleaved,
    check_graph_capture_and_replay,
    check_unsupported_shape_raises_on_every_rank,
    check_create_and_first_compile_refuse_capture,
    # Stays last: it deliberately disagrees on call order.
    check_wrong_call_order_is_detected,
]


# ── setup ─────────────────────────────────────────────────────────────────


def make_experts(seed: int):
    """224 random MXFP4 experts through TRT-LLM's W4A8_MXFP4_MXFP8 TRTLLM-Gen loader: this rank's buffers."""
    from tensorrt_llm._torch.moe.fused_moe.quantization import W4A8MXFP4MXFP8TRTLLMGenFusedMoEMethod

    def rand_mxfp4(rows, k, gen):
        codes = torch.randint(0, 16, (rows, k), dtype=torch.uint8, device="cuda", generator=gen)
        base = 127 + round(0.5 * math.log2(0.01057 / k))
        exps = torch.randint(
            base, base + 6, (rows, k // SV), dtype=torch.uint8, device="cuda", generator=gen
        )
        return (codes[:, 0::2] | (codes[:, 1::2] << 4)).contiguous(), exps

    i_full = I_TP * MOE_TP
    method = W4A8MXFP4MXFP8TRTLLMGenFusedMoEMethod()
    module = SimpleNamespace(
        tp_size=MOE_TP,
        tp_rank=1,
        scaling_vector_size=SV,
        intermediate_size=i_full,
        intermediate_size_per_partition=I_TP,
        hidden_size=LATENT,
    )
    kw = dict(dtype=torch.uint8, device="cuda")
    proc = dict(
        w31=torch.empty(E_LOCAL, 2 * I_TP, LATENT // 2, **kw),
        w31s=torch.empty(E_LOCAL, 2 * I_TP, LATENT // SV, **kw),
        w2=torch.empty(E_LOCAL, LATENT, I_TP // 2, **kw),
        w2s=torch.empty(E_LOCAL, LATENT, I_TP // SV, **kw),
    )
    gen = torch.Generator(device="cuda").manual_seed(seed)
    for e in range(E_LOCAL):
        w1, w1s = rand_mxfp4(i_full, LATENT, gen)
        w3, w3s = rand_mxfp4(i_full, LATENT, gen)
        w2, w2s = rand_mxfp4(LATENT, i_full, gen)
        method.load_expert_w3_w1_weight(module, w1, w3, proc["w31"][e])
        method.load_expert_w2_weight(module, w2, proc["w2"][e])
        method.load_expert_w3_w1_weight_scale_mxfp4(module, w1s, w3s, proc["w31s"][e])
        method.load_expert_w2_weight_scale_mxfp4(module, w2s, proc["w2s"][e])
    torch.cuda.synchronize()
    return proc


def make_layer_weights(layer: int, front_weight):
    """Layer ``layer``'s front weights, exact payloads (see the module docstring).

    Every rank's head slice: its latent-down rows (multiples of 1/16), then its router rows (multiples of 1/8, logits
    of a few units); one seed per rank, so each rank holds them all for the reference. This rank's shared gate_up
    (multiples of 1/16; gate rows, then up rows), and this rank's front weight built from them.
    """
    heads = []
    for r in range(R.world):
        g = torch.Generator(device="cuda").manual_seed(1000 * (layer + 1) + r)
        latent_rows = ls.exact_bf16(g, (WL, HIDDEN), -2, 3, 1 / 16)
        router_rows = ls.exact_bf16(g, (WE, HIDDEN), -2, 3, 1 / 8)
        heads.append(torch.cat([latent_rows, router_rows]).contiguous())
    g = torch.Generator(device="cuda").manual_seed(50_000 + 1000 * layer + R.rank)
    gate_up = ls.exact_bf16(g, (2 * SHARED_COLS, HIDDEN), -2, 3, 1 / 16)
    return SimpleNamespace(heads=heads, gate_up=gate_up, front=front_weight(heads[R.rank], gate_up))


def make_state(head_flags: bool):
    """A K3MoeState on this device and one layer per front layer, all over this rank's experts (own counters each)."""
    device = torch.device("cuda", torch.cuda.current_device())
    state = OPS.K3MoeState(device, I_TP, E_LOCAL, head_flags=head_flags)
    p = EXPERT_BUFFERS
    layers = [state.layer(p["w31"], p["w31s"], p["w2"], p["w2s"]) for _ in range(LAYERS)]
    return SimpleNamespace(state=state, layers=layers)


def _run_one_rank(args) -> int:
    global R, OPS, WS_A, WS_B, PLAIN, FLAGS, LAYER_WEIGHTS, BIAS, EXPERT_BUFFERS, OFFSET, WL, WE
    R = ls.Rank(args)
    assert torch.cuda.get_device_capability() == (10, 0), "k3_moe_front and k3_moe need sm_100"
    import tensorrt_llm._torch.custom_ops  # noqa: F401 -- registers the stock path's ops
    from tensorrt_llm._torch._experimental.modeling_v2.catalog.moe import k3_moe as moe
    from tensorrt_llm._torch._experimental.modeling_v2.catalog.moe import k3_moe_front as front

    OPS = SimpleNamespace(
        front=front.k3_moe_front,
        k3_moe=moe.k3_moe,
        workspace=front.K3MoeHeadWorkspace,
        K3MoeState=moe.K3MoeState,
    )
    WL, WE = LATENT // R.world, EXPERTS // R.world
    OFFSET = (R.rank % MOE_TP) * E_LOCAL
    with torch.inference_mode():
        device = torch.device("cuda", torch.cuda.current_device())
        assert front.weight_supported(R.world, SHARED_COLS, HIDDEN, device), (
            f"k3_moe_front does not support W {R.world} with {SHARED_COLS} shared columns on this device"
        )
        g = torch.Generator(device="cuda").manual_seed(11)
        BIAS = (torch.randn(EXPERTS, generator=g, device="cuda") * 0.05).float()
        LAYER_WEIGHTS = [make_layer_weights(layer, front.front_weight) for layer in range(LAYERS)]
        EXPERT_BUFFERS = make_experts(20260928 + R.rank)
        WS_A = OPS.workspace.create(R.mapping, fabric_handle=R.fabric)
        WS_B = OPS.workspace.create(R.mapping, fabric_handle=R.fabric)
        PLAIN = make_state(head_flags=False)
        FLAGS = make_state(head_flags=True)
        code = ls.run_checks(R, CHECKS)
    if R.rank == 0:
        s = STATS
        print(
            f"[rank 0] world {R.world}; {s['fronts_checked']} front calls bit for bit the reference's routing and "
            f"latent; shared max rel err {s['shared_err']:.3e}; y vs stock runner max {s['y_elt_ulp']:.2f} ulp "
            f"per element, {s['y_rms_ulp']:.2f} ulp RMS; fused rows bit-identical to the 8-token call's "
            f"{s['rows_bits_as_m8']}/{s['rows_checked']}",
            flush=True,
        )
    return code


if __name__ == "__main__":
    ARGS = ls.parse_args(sys.argv[1:])
    if ARGS.rank_worker or ARGS.launcher == "srun":
        sys.exit(_run_one_rank(ARGS))
    ls.spawn(__file__, ARGS, DEADLINE_S)
    print("OK")
