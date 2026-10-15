# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the fused_recurrent_gated_delta_rule_cached_replay_update entry.

The op is stateful across verify steps: what one call leaves in the checkpoint
pool and the history double buffer is what the next call reads, after the
engine's own bookkeeping ran in between. So every cell drives several
consecutive verify steps the way the engine does -- work items before the
forward (partitioned batches), the op once per layer, then the all-layer
commit and `_advance_replay_state` with a random partial acceptance -- and
checks two things at every step:

* each call against the entry's `reference` on that call's own inputs (the
  output and all four state buffers), plus bitwise locks on everything the
  call must not touch;
* every output against an *exact twin*: the entry's `reference` driven on the
  same checkpoint and history held in fp32 and deep enough never to wrap,
  advanced by `p += accepted` alone. The twin's state is the plain
  token-by-token recurrence over every accepted token, independent of the
  wrap, the double buffer and the commit, so a wrong commit or wrong
  bookkeeping surfaces as an output that drifts off it --
  `test_the_twin_sees_a_broken_protocol` shows it does.
"""

from __future__ import annotations

import zlib

import pytest
import torch
import torch.nn.functional as F
from triton.compiler.errors import CompilationError

import tensorrt_llm._torch.modules.fla.cached_replay as _impl
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_103__tp1.ops.ssm.fused_recurrent_gated_delta_rule_cached_replay_update import (  # noqa: E501
    ALL_LAYER_COMMIT_MIN_BATCH,
    HEAD_K_DIM,
    HEAD_V_DIM,
    HISTORY_SIZE,
    NUM_K_HEADS,
    NUM_V_HEADS,
    PACKED_QKV_WIDTH,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_103__tp1.ops.ssm.fused_recurrent_gated_delta_rule_cached_replay_update import (  # noqa: E501
    fused_recurrent_gated_delta_rule_cached_replay_update as op,
)
from tensorrt_llm._torch.modules.fla.cached_replay import (
    CACHED_REPLAY_PARTITION_MIN_BATCH_SIZE,
    commit_gdn_cached_replay_history_layers,
)
from tensorrt_llm._torch.modules.mamba.mamba2_metadata import _build_replay_work_items_triton
from tensorrt_llm._torch.pyexecutor.kv_cache.mamba_cache_manager import (
    ReplayStateUpdateMetadata,
    _advance_replay_state,
)

__extra_import_path__ = ["../../../.."]
from _validating import ULP, certifying, validating  # noqa: E402,F401 — path above; fixture by name

assert torch.cuda.is_available(), (
    "fused_recurrent_gated_delta_rule_cached_replay_update requires a CUDA device"
)

DEV = "cuda"
H, HV, K, V = NUM_K_HEADS, NUM_V_HEADS, HEAD_K_DIM, HEAD_V_DIM
W = PACKED_QKV_WIDTH
HIST = HISTORY_SIZE
OUT_SENTINEL = -7.0
BF16_ULP = ULP[torch.bfloat16]

#: This checkpoint's 36 GDN layers of `linear_attn.A_log` / `linear_attn.dt_bias`
#: (bf16, 48 each): mean, standard deviation, range.
A_LOG_STATS = (1.006, 1.487, (-7.125, 5.0625))
DT_BIAS_STATS = (-4.411, 1.732, (-11.4375, 2.53125))

#: Element gate of the bf16 members, in bf16 ulps of the block they belong to.
_ELEMENT_ULPS = 4.0
#: Relative-RMS gate of the output, in bf16 ulps.
_OUTPUT_RMS_ULPS = 1.5
#: Element gate of the checkpoint state, relative to its (slot, head) scale.
_STATE_TOL = 2.0**-13
#: Gate of the stored cumulative log-decays: |dG| <= atol + rtol * |G|.
_G_ATOL, _G_RTOL = 2.0**-11, 2.0**-18


def _gate(name, got, want, element, rms=None):
    """`got` against `want` per block of the last two dims (a token's
    `[heads, dim]`, or one head's `[V, K]` state): every element within
    `element` of its block's absmax, NaN blocks of `want` excluded."""
    block = got.shape[-2] * got.shape[-1]
    g, w = got.float().reshape(-1, block), want.float().reshape(-1, block)
    keep = ~torch.isnan(w).any(1)
    g, w = g[keep], w[keep]
    scale = w.abs().amax(1, keepdim=True).clamp_min(torch.finfo(torch.float32).tiny)
    ratio = ((g - w).abs() / scale).nan_to_num(nan=float("inf"))
    worst = ratio.max().item() if ratio.numel() else 0.0
    assert worst <= element, (
        f"{name}: an element is {worst:.3e} of its block's absmax from the reference "
        f"(gate {element:.3e})"
    )
    if rms is not None and w.numel():
        rel = ((g - w).pow(2).mean().sqrt() / w.pow(2).mean().sqrt().clamp_min(1e-30)).item()
        assert rel <= rms, f"{name}: relative RMS {rel:.3e} exceeds {rms:.3e}"


def _compare(out: tuple, ref: tuple) -> None:
    """`(returned, ssm_states, old_u, old_k, old_G)` as the call left them,
    against `reference`'s five; a member `None` on both sides is skipped (the
    exact twin is compared on the output alone).

    The kernel is not fp32 throughout, and the bands say where. It rounds the
    L2-normalized q and k to bf16 before every product (2^-9 relative each),
    rounds the history-mixing coefficients to bf16 for its tensor-core dots, and
    stores the output and the history's u and k in bf16. So the output, `old_u`
    and `old_k` are gated per block -- one token's `[heads, dim]` -- at 4 bf16
    ulps of the block's absmax, which keeps a near-silent head from turning
    rounding into relative noise, and the output's relative RMS at 1.5 ulps. The
    checkpoint state is fp32 by construction: the commit splits the decayed
    history update into a bf16 hi and lo part, ~16 significant bits per term, and
    up to 16 terms each within 2^-17 give the 2^-13 gate, relative to one (slot,
    head) state's absmax. `G` only ever enters as exp(G_t - G_j), so an absolute
    error is a relative decay error: the kernel's softplus runs on approximate
    transcendentals whose fp32 sum drops exp(x)'s low bits; atol 2^-11 is an
    eighth of a bf16 ulp of the u it scales, and rtol 2^-18 covers the cumulative
    sum's rounding where |G| reaches thousands on fast-forgetting heads.
    """
    names = ("output", "ssm_states", "old_u", "old_k", "old_G")
    assert len(out) == len(ref) == len(names)
    for name, got, want in zip(names, out, ref):
        assert (got is None) == (want is None), f"{name}: present on one side only"
        if got is None:
            continue
        assert got.dtype == want.dtype and got.shape == want.shape, (
            name,
            (got.dtype, tuple(got.shape)),
            (want.dtype, tuple(want.shape)),
        )
        if name == "old_G":
            torch.testing.assert_close(got, want, rtol=_G_RTOL, atol=_G_ATOL, msg=name)
        elif name == "ssm_states":
            _gate(name, got, want, _STATE_TOL)
        else:
            rms = _OUTPUT_RMS_ULPS * BF16_ULP if name == "output" else None
            _gate(name, got, want, _ELEMENT_ULPS * BF16_ULP, rms)


def _seed(why):
    return zlib.crc32(why.encode())


def _gating(gen, layers):
    """Per-layer A_log / dt_bias drawn around the checkpoint's own statistics;
    head 0 pinned to the corner that forgets everything every token
    (g ~ -400), head 1 to the one that hardly decays (g ~ -1e-8)."""
    (a_mean, a_std, a_range), (d_mean, d_std, d_range) = A_LOG_STATS, DT_BIAS_STATS
    a_log = torch.randn(layers, HV, generator=gen, device=DEV) * a_std + a_mean
    dt_bias = torch.randn(layers, HV, generator=gen, device=DEV) * d_std + d_mean
    a_log, dt_bias = a_log.clamp(*a_range), dt_bias.clamp(*d_range)
    a_log[:, 0], dt_bias[:, 0] = a_range[1], d_range[1]
    a_log[:, 1], dt_bias[:, 1] = a_range[0], d_range[0]
    return a_log.contiguous(), dt_bias.contiguous()


def _unit(x):
    return (x / x.norm(dim=-1, keepdim=True)).to(torch.bfloat16)


class _Bench:
    """The state the cache manager keeps for `layers` GDN layers, laid out as
    `MambaHybridCacheManagerV2` lays it out, driven the way the engine drives
    it: work items before the forward (`_build_replay_work_items_triton`, what
    mamba metadata's prepare runs for a partitioned batch), the op per layer,
    then `update_mamba_states`'s order -- the all-layer commit, then
    `_advance_replay_state`.

    With `exact=True` it also carries the exact twin (see the module docstring).
    """

    def __init__(self, spec, seed, exact=True):
        self.spec = spec
        self.path, self.n, self.t = spec["path"], spec["n"], spec["t"]
        self.layers = spec.get("layers", 2)
        self.slots = spec.get("slots", 64)
        self.steps = spec.get("steps", 8)
        self.gen = torch.Generator(device=DEV).manual_seed(seed)
        self.cpu = torch.Generator().manual_seed(seed)
        n, t, L, S = self.n, self.t, self.layers, self.slots

        if spec.get("edge_slots"):
            inner = torch.randperm(S - 2, generator=self.cpu)[: n - 2] + 1
            chosen = torch.cat([torch.tensor([S - 1]), inner, torch.tensor([0])])
        else:
            chosen = torch.randperm(S, generator=self.cpu)[:n]
        chosen = chosen.to(torch.int32)
        for row in spec.get("pad", ()):
            chosen[row] = -1
        live_rows = chosen >= 0
        dup = tuple(spec.get("dup", ()))
        if dup:
            chosen[list(dup)] = int(chosen[dup[0]])
            # The padding rows' shared slot is nobody's state: the bookkeeping never names it.
            live_rows[list(dup)] = False
        self.state_indices = chosen.to(DEV)
        self.live = self.state_indices[live_rows.to(DEV)].clone()

        # [slots, layers, HV, V, K]: each layer's state is a slot-strided view,
        # the affine layout whose strides the all-layer commit is handed.
        self.backing = self._randn(S, L, HV, V, K) * 0.5
        self.pool = [self.backing[:, i] for i in range(L)]
        self.old_x = torch.zeros(L, S, 2, HIST, HV, V, device=DEV, dtype=torch.bfloat16)
        self.old_B = torch.zeros(L, S, 2, HIST, H, K, device=DEV, dtype=torch.bfloat16)
        self.old_dt = torch.zeros(L, S, 2, HV, HIST, device=DEV)
        self.old_dA = self._randn(L, S, 2, HV, HIST)  # handed over as old_beta; never read
        self.pnat = torch.zeros(S, dtype=torch.int32, device=DEV)
        self.buf = torch.zeros(S, dtype=torch.int32, device=DEV)
        self.a_log, self.dt_bias = _gating(self.gen, L)
        if spec.get("start", "warm") == "warm":
            self.old_x.copy_(self._randn(L, S, 2, HIST, HV, V) * 0.5)
            self.old_B.copy_(_unit(self._randn(L, S, 2, HIST, H, K)))
            raw_a = self._randn(L, S, 2, HV, HIST)
            g = -torch.exp(self.a_log)[:, None, None, :, None] * F.softplus(
                raw_a + self.dt_bias[:, None, None, :, None]
            )
            self.old_dt.copy_(g.cumsum(-1))
            pnat = torch.randint(0, HIST + 1, (S,), generator=self.cpu, dtype=torch.int32)
            live = self.live.long().cpu()
            pnat[live[0]] = HIST  # a wrap at step 0
            if live.numel() > 1:
                pnat[live[1]] = 0  # an append at step 0
            self.pnat.copy_(pnat)
            self.buf.copy_(torch.randint(0, 2, (S,), generator=self.cpu, dtype=torch.int32))
        self.meta = ReplayStateUpdateMetadata(self.pnat, self.buf, t, HIST)
        self.work_items = self.n_writes = None
        if self.path == "all_layer":
            self.work_items = torch.zeros(n, 4, dtype=torch.int32, device=DEV)
            self.n_writes = torch.zeros(1, dtype=torch.int32, device=DEV)

        self.exact = exact
        if exact:
            self.depth = HIST + (self.steps + 1) * t
            self.x_pool = [p.clone() for p in self.pool]
            self.x_u = torch.zeros(L, S, 1, self.depth, HV, V, device=DEV)
            self.x_k = torch.zeros(L, S, 1, self.depth, H, K, device=DEV)
            self.x_G = torch.zeros(L, S, 1, HV, self.depth, device=DEV)
            self.x_pnat = self.pnat.clone()
            self.x_buf = torch.zeros_like(self.buf)
            for s in self.live.tolist():
                p, b = int(self.pnat[s]), int(self.buf[s])
                self.x_u[:, s, 0, :p] = self.old_x[:, s, b, :p].float()
                self.x_k[:, s, 0, :p] = self.old_B[:, s, b, :p].float()
                self.x_G[:, s, 0, :, :p] = self.old_dt[:, s, b, :, :p]

    def _randn(self, *shape, dtype=torch.float32):
        return torch.randn(*shape, generator=self.gen, device=DEV, dtype=dtype)

    def inputs(self):
        """One layer's fresh activations: post-conv `[q|k|v]` rows and the raw
        `a`/`b` projections, as the verify step hands them over."""
        n, t = self.n, self.t
        packed = self._randn(n * t, W, dtype=torch.bfloat16)
        rows = packed.view(n, t, W)
        q = rows[..., : H * K].view(n, t, H, K)
        k = rows[..., H * K : 2 * H * K].view(n, t, H, K)
        v = rows[..., 2 * H * K :].view(n, t, HV, V)
        if self.spec.get("packed", True) is False:
            packed = None  # the mixed batch's form: the views alone, copied by the op
        if self.spec.get("ab", "split") == "strided":
            ab = self._randn(n * t, 2 * HV, dtype=torch.bfloat16)
            b, a = ab[:, :HV].view(n, t, HV), ab[:, HV:].view(n, t, HV)
        else:
            a = self._randn(n, t, HV, dtype=torch.bfloat16)
            b = self._randn(n, t, HV, dtype=torch.bfloat16)
        out_buf = output = None
        lo = 0 if self.spec.get("output") == "prefix" else 5
        if self.spec.get("output") in ("prefix", "suffix"):
            # A slice of a longer token-major buffer: its first rows ("prefix"),
            # or the rows after 5 context tokens, as a mixed batch hands it over.
            out_buf = torch.full((n * t + 5, HV, V), OUT_SENTINEL, dtype=torch.bfloat16, device=DEV)
            output = out_buf[lo : lo + n * t].view(n, t, HV, V)
        return dict(q=q, k=k, v=v, a=a, b=b, packed=packed, output=output, out_buf=out_buf, lo=lo)

    def kwargs(self, layer, act):
        return dict(
            q=act["q"],
            k=act["k"],
            v=act["v"],
            g=act["a"],
            beta=act["b"],
            ssm_states=self.pool[layer],
            state_indices=self.state_indices,
            old_u=self.old_x[layer],
            old_k=self.old_B[layer],
            old_G=self.old_dt[layer],
            old_beta=self.old_dA[layer],
            cache_buf_idx=self.buf,
            prev_num_accepted_tokens=self.pnat,
            history_size=HIST,
            scale=None,
            use_qk_l2norm_in_kernel=True,
            A_log=self.a_log[layer],
            dt_bias=self.dt_bias[layer],
            launch_with_pdl=self.spec.get("pdl", True),
            replay_work_items=self.work_items,
            n_writes=self.n_writes,
            packed_qkv=act["packed"],
            use_all_layer_commit=self.path == "all_layer",
            output=act["output"],
        )

    def exact_kwargs(self, layer, kw):
        return dict(
            kw,
            ssm_states=self.x_pool[layer],
            old_u=self.x_u[layer],
            old_k=self.x_k[layer],
            old_G=self.x_G[layer],
            cache_buf_idx=self.x_buf,
            prev_num_accepted_tokens=self.x_pnat,
            history_size=self.depth,
            replay_work_items=None,
            n_writes=None,
            use_all_layer_commit=False,
            output=None,
        )

    def adopt_exact(self, layer, ref):
        assert torch.equal(ref[1], self.x_pool[layer]), "the twin must never wrap"
        self.x_u[layer].copy_(ref[2])
        self.x_k[layer].copy_(ref[3])
        self.x_G[layer].copy_(ref[4])

    def state(self, layer):
        return self.pool[layer], self.old_x[layer], self.old_B[layer], self.old_dt[layer]

    def prepare(self):
        if self.path == "all_layer":
            self.work_items.zero_()
            _build_replay_work_items_triton(
                self.live, self.pnat, self.buf, self.work_items, self.n_writes, self.t, HIST
            )

    def commit(self):
        if self.path == "all_layer":
            commit_gdn_cached_replay_history_layers(
                ssm_states=self.pool[0],
                ssm_state_num_layers=self.layers,
                ssm_state_layer_stride=self.backing.stride(1),
                ssm_state_slot_stride=self.backing.stride(0),
                old_u=self.old_x,
                old_k=self.old_B,
                old_G=self.old_dt,
                replay_work_items=self.work_items,
                n_writes=self.n_writes,
                history_size=HIST,
            )

    def draw_acceptance(self):
        return torch.randint(
            1, self.t + 1, (self.live.numel(),), generator=self.cpu, dtype=torch.int32
        ).to(DEV)

    def accept(self, accepted):
        _advance_replay_state(self.meta, self.live, accepted)
        if self.exact:
            self.x_pnat[self.live.long()] += accepted

    def writes(self):
        """Where this step's calls append, by the documented rule, and which
        slots wrap: `[slots, 2, HIST]` mask, wrapping slot ids."""
        s = self.live.long()
        p, b = self.pnat[s].long(), self.buf[s].long()
        wrap = p + self.t > HIST
        half, start = torch.where(wrap, 1 - b, b), torch.where(wrap, 0, p)
        mask = torch.zeros(self.slots, 2, HIST, dtype=torch.bool, device=DEV)
        for t in range(self.t):
            mask[s, half, start + t] = True
        return mask, s[wrap]


def _deviation(got, want):
    """The largest element deviation relative to its block's absmax (the
    quantity `_compare` gates)."""
    block = got.shape[-2] * got.shape[-1]
    g, w = got.float().reshape(-1, block), want.float().reshape(-1, block)
    keep = ~torch.isnan(w).any(1)
    g, w = g[keep], w[keep]
    if not w.numel():
        return 0.0
    scale = w.abs().amax(1, keepdim=True).clamp_min(1e-30)
    return ((g - w).abs() / scale).max().item()


def _check_duplicate_padding_slot(cell) -> None:
    """The last rows of a batch share one slot, as CUDA-graph padding repeats the manager's
    dummy slot: the rows on unique slots, and every slot but the shared one, are held to
    the reference run on those rows alone -- the padding rows change nothing about them.
    The shared slot's state and the padding rows' outputs are unspecified."""
    bench = _Bench(cell.spec, seed=_seed(cell.why), exact=False)
    dup = cell.spec["dup"]
    unique = bench.n - len(dup)
    assert dup == tuple(range(unique, bench.n)), "the padding rows trail the batch"
    shared = int(bench.state_indices[unique])
    assert (bench.state_indices[unique:] == shared).all()
    assert shared not in bench.live.tolist()
    others = torch.ones(bench.slots, dtype=torch.bool, device=DEV)
    others[shared] = False
    for _ in range(bench.steps):
        for layer in range(bench.layers):
            act = bench.inputs()
            kw = bench.kwargs(layer, act)
            alone = dict(
                kw,
                q=kw["q"][:unique],
                k=kw["k"][:unique],
                v=kw["v"][:unique],
                g=kw["g"][:unique],
                beta=kw["beta"][:unique],
                state_indices=kw["state_indices"][:unique],
                packed_qkv=act["packed"][: unique * bench.t],
            )
            ref = op.reference(**alone)
            got = op(**kw)
            pool, u, k, G = bench.state(layer)
            _compare(
                (got[:unique], pool[others], u[others], k[others], G[others]),
                (ref[0], *(x[others] for x in ref[1:])),
            )
        bench.accept(bench.draw_acceptance())


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — pytest resolves the fixture by this name
    certifying(op)
    if "dup" in cell.spec:
        return _check_duplicate_padding_slot(cell)
    bench = _Bench(cell.spec, seed=_seed(cell.why))
    pad = list(cell.spec.get("pad", ()))
    wraps = appends = 0
    for _ in range(bench.steps):
        mask, wrapped = bench.writes()
        wraps += wrapped.numel()
        appends += bench.live.numel() - wrapped.numel()
        bench.prepare()
        after_commit = []
        for layer in range(bench.layers):
            act = bench.inputs()
            kw = bench.kwargs(layer, act)
            pool, u, k, G = bench.state(layer)
            before = dict(
                pool=pool.clone(),
                u=u.clone(),
                k=k.clone(),
                G=G.clone(),
                old_beta=bench.old_dA[layer].clone(),
                out_buf=None if act["out_buf"] is None else act["out_buf"].clone(),
            )
            ref = op.reference(**kw)
            exact = op.reference(**bench.exact_kwargs(layer, kw))
            committed = None
            if bench.path == "all_layer":
                committed = op.reference(
                    **dict(kw, replay_work_items=None, n_writes=None, use_all_layer_commit=False)
                )
            got = op(**kw)
            _compare((got, *bench.state(layer)), ref)
            # The locks: in place, and nothing outside the documented writes.
            pool, u, k, G = bench.state(layer)
            assert torch.equal(bench.old_dA[layer], before["old_beta"]), "old_beta is never written"
            assert torch.equal(u[~mask], before["u"][~mask]), "old_u written outside the append"
            assert torch.equal(k[~mask], before["k"][~mask]), "old_k written outside the append"
            Gt, Gt0 = G.transpose(-1, -2), before["G"].transpose(-1, -2)
            assert torch.equal(Gt[~mask], Gt0[~mask]), "old_G written outside the append"
            untouched = torch.ones(bench.slots, dtype=torch.bool, device=DEV)
            if bench.path == "small":
                untouched[wrapped] = False
            assert torch.equal(pool[untouched], before["pool"][untouched]), (
                "the pool moved where the call commits nothing (a partitioned call commits none)"
            )
            if kw["output"] is not None:
                assert got.data_ptr() == kw["output"].data_ptr(), "output must be written in place"
                outside = torch.ones(act["out_buf"].shape[0], dtype=torch.bool, device=DEV)
                outside[act["lo"] : act["lo"] + bench.n * bench.t] = False
                assert torch.equal(act["out_buf"][outside], before["out_buf"][outside]), (
                    "rows of the output buffer outside the view were touched"
                )
                for row in pad:
                    assert (got[row] == OUT_SENTINEL).all(), f"padding row {row} written"
            bench.adopt_exact(layer, exact)
            after_commit.append((got, exact, committed))
        bench.commit()
        for layer, (got, exact, committed) in enumerate(after_commit):
            if committed is not None:
                # The partitioned call plus the manager's commit is the call
                # that commits in-kernel.
                _compare((got, *bench.state(layer)), committed)
            _compare((got, None, None, None, None), (exact[0], None, None, None, None))
        bench.accept(bench.draw_acceptance())
    assert wraps > 0 and appends > 0, (
        f"the cell never wrapped or never appended ({wraps}, {appends})"
    )


# ── CUDA graphs ───────────────────────────────────────────────────────────


@pytest.mark.parametrize("n, t", ((1, 2), (8, 4), (15, 3), (16, 4), (32, 2)))
def test_cuda_graph_replay_on_new_data_is_bitwise_eager(n, t) -> None:
    """Two layers captured with, from 16 sequences on, the all-layer commit after
    them; replayed three times on new activations, with acceptance in between,
    against the same steps run eagerly on a twin state."""
    assert ALL_LAYER_COMMIT_MIN_BATCH == CACHED_REPLAY_PARTITION_MIN_BATCH_SIZE
    path = "all_layer" if n >= ALL_LAYER_COMMIT_MIN_BATCH else "small"
    spec = dict(path=path, n=n, t=t, layers=2, slots=40)
    seed = 7000 + 10 * n + t
    eager, graph, warm = (_Bench(spec, seed, exact=False) for _ in range(3))
    acts = [warm.inputs() for _ in range(2)]
    warm.prepare()
    for layer in range(2):
        op(**warm.kwargs(layer, acts[layer]))
    warm.commit()
    torch.cuda.synchronize()

    graph.prepare()
    captured = torch.cuda.CUDAGraph()
    with torch.cuda.graph(captured):
        outs_graph = [op(**graph.kwargs(layer, acts[layer])) for layer in range(2)]
        graph.commit()
    assert torch.equal(graph.backing, eager.backing) and torch.equal(graph.old_x, eager.old_x)

    refill = torch.Generator(device=DEV).manual_seed(seed + 1)
    for _ in range(3):
        for act in acts:
            for key in ("packed", "a", "b"):
                act[key].copy_(
                    torch.randn(act[key].shape, generator=refill, device=DEV, dtype=act[key].dtype)
                )
        eager.prepare()
        outs_eager = [op(**eager.kwargs(layer, acts[layer])) for layer in range(2)]
        eager.commit()
        graph.prepare()
        captured.replay()
        torch.cuda.synchronize()
        for e, g in zip(outs_eager, outs_graph):
            assert torch.equal(e, g), "replayed output differs from eager"
        for name in ("backing", "old_x", "old_B", "old_dt", "pnat", "buf"):
            assert torch.equal(getattr(eager, name), getattr(graph, name)), f"{name} differs"
        accepted = eager.draw_acceptance()
        eager.accept(accepted)
        graph.accept(accepted)


# ── the harness can see what it exists to see ─────────────────────────────


def _drive(bench, fault):
    """Run `bench` with one protocol fault injected; return the first step at
    which the twin comparison rejected an output, or None."""
    for step in range(bench.steps):
        bench.prepare()
        outs = []
        for layer in range(bench.layers):
            kw = bench.kwargs(layer, bench.inputs())
            if fault == "committed twice":
                kw = dict(kw, replay_work_items=None, n_writes=None, use_all_layer_commit=False)
            exact = op.reference(**bench.exact_kwargs(layer, kw))
            bench.adopt_exact(layer, exact)
            outs.append((op(**kw), exact[0]))
        if fault != "commit skipped":
            bench.commit()
        try:
            for got, want in outs:
                _compare((got, None, None, None, None), (want, None, None, None, None))
        except AssertionError:
            return step
        accepted = bench.draw_acceptance()
        if fault == "rejected tokens kept":
            _advance_replay_state(bench.meta, bench.live, torch.full_like(accepted, bench.t))
            bench.x_pnat[bench.live.long()] += accepted
        else:
            bench.accept(accepted)
    return None


@pytest.mark.parametrize(
    "fault", ("none", "commit skipped", "committed twice", "rejected tokens kept")
)
def test_the_twin_sees_a_broken_protocol(fault) -> None:
    """The twin comparison is what certifies the commit and the bookkeeping, so
    it has to fail when either is broken: the partitioned path with the
    manager's commit left out; a 16-row batch called without work items while
    the manager still commits (the checkpoint advanced twice); and the small
    path advanced as if every drafted token had been accepted. Without a fault
    the same run passes."""
    path = "small" if fault == "rejected tokens kept" else "all_layer"
    n = 8 if path == "small" else 16
    bench = _Bench(dict(path=path, n=n, t=4, steps=4), seed=93)
    first = _drive(bench, fault)
    if fault == "none":
        assert first is None
    else:
        assert first is not None, f"{fault} passed the twin comparison"


def test_the_gate_rejects_near_misses() -> None:
    """`_compare` against plausible wrong answers, each the entry's own
    reference driven on transformed inputs, at a call where one row wraps (and
    commits) and the others append."""
    bench = _Bench(dict(path="small", n=8, t=4), seed=91, exact=False)
    kw = bench.kwargs(0, bench.inputs())
    pool, u, k, G = (x.clone() for x in bench.state(0))
    base = dict(kw, ssm_states=pool, old_u=u, old_k=k, old_G=G)
    ref = op.reference(**base)
    got = op(**kw)
    out = (got, *bench.state(0))
    _compare(out, ref)
    p, b = kw["prev_num_accepted_tokens"], kw["cache_buf_idx"]

    not_restarted = ref[4].clone()
    _, wrapped = bench.writes()
    for s in wrapped.tolist():
        g_start = G[s, b[s].long(), :, p[s].long() - 1]
        not_restarted[s, 1 - b[s].long(), :, : bench.t] += g_start[:, None]
    rivals = {
        "history ignored": op.reference(**dict(base, prev_num_accepted_tokens=torch.zeros_like(p))),
        "the other history half": op.reference(**dict(base, cache_buf_idx=1 - b)),
        "one pending token dropped": op.reference(
            **dict(base, prev_num_accepted_tokens=(p - 1).clamp(min=0))
        ),
        "checkpoint zeroed": op.reference(**dict(base, ssm_states=torch.zeros_like(pool))),
        "G not restarted at a wrap": ref[:4] + (not_restarted,),
    }
    for rival in rivals.values():
        with pytest.raises(AssertionError):
            _compare(out, rival)


# ── properties beyond one reference comparison ────────────────────────────


def test_the_wrapped_kernels_carry_no_autotuner() -> None:
    """Plain `triton.jit`: no trial launch can consume the in-place state on a
    process's first call."""
    for kernel in (_impl._cached_replay_kernel, _impl._cached_replay_layered_commit_kernel):
        names = [cls.__name__ for cls in type(kernel).__mro__]
        assert "Autotuner" not in names and "Heuristics" not in names, names


def _clone_state(bench):
    return {name: getattr(bench, name).clone() for name in ("backing", "old_x", "old_B", "old_dt")}


def _restore_state(bench, saved):
    for name, value in saved.items():
        getattr(bench, name).copy_(value)


def test_the_two_commit_paths_are_bitwise_interchangeable() -> None:
    """The same 16 rows through the in-kernel commit, and through the
    partitioned call plus the manager's all-layer commit: bit for bit."""
    bench = _Bench(dict(path="all_layer", n=16, t=4, layers=2), seed=95, exact=False)
    acts = [bench.inputs() for _ in range(2)]
    saved = _clone_state(bench)
    small = [
        op(
            **dict(
                bench.kwargs(i, acts[i]),
                replay_work_items=None,
                n_writes=None,
                use_all_layer_commit=False,
            )
        )
        for i in range(2)
    ]
    after_small = _clone_state(bench)
    _restore_state(bench, saved)
    bench.prepare()
    assert 0 < int(bench.n_writes) < bench.n, "want both wrapping and appending rows"
    partitioned = [op(**bench.kwargs(i, acts[i])) for i in range(2)]
    bench.commit()
    for a, b in zip(small, partitioned):
        assert torch.equal(a, b)
    for name, value in after_small.items():
        assert torch.equal(getattr(bench, name), value), name


def test_packed_qkv_supersedes_q_k_v_and_launch_knobs_are_neutral() -> None:
    """With `packed_qkv` the kernel reads q/k/v from it alone -- their own
    contents are never read; the separate contiguous form, PDL on or off, and a
    second run are all bitwise the same call."""
    bench = _Bench(dict(path="small", n=6, t=3, layers=1), seed=97, exact=False)
    act = bench.inputs()
    saved = _clone_state(bench)

    def run(**override):
        _restore_state(bench, saved)
        out = op(**dict(bench.kwargs(0, act), **override)).clone()
        return out, _clone_state(bench)

    base = run()
    q, k, v = act["q"], act["k"], act["v"]
    variants = {
        "separate q/k/v": run(
            q=q.contiguous(), k=k.contiguous(), v=v.contiguous(), packed_qkv=None
        ),
        "q/k/v contents zeroed": run(
            q=torch.zeros_like(q.contiguous()),
            k=torch.zeros_like(k.contiguous()),
            v=torch.zeros_like(v.contiguous()),
        ),
        "launch_with_pdl=False": run(launch_with_pdl=False),
        "second run": run(),
    }
    for name, (out, state) in variants.items():
        assert torch.equal(out, base[0]), name
        for key, value in state.items():
            assert torch.equal(value, base[1][key]), (name, key)


# ── guards: accepted and silently wrong ──────────────────────────────────


def test_a_short_state_indices_reads_past_its_end_and_is_refused() -> None:
    """Unguarded, three indices for four rows: the fourth row reads whatever
    follows the view -- here a slot the call never named -- and writes its
    history."""
    bench = _Bench(dict(path="small", n=4, t=2, layers=1), seed=101, exact=False)
    stranger = next(s for s in range(bench.slots) if s not in bench.live.tolist())
    parent = torch.cat(
        [bench.state_indices[:3], torch.tensor([stranger], dtype=torch.int32, device=DEV)]
    )
    kw = dict(bench.kwargs(0, bench.inputs()), state_indices=parent[:3])
    before = bench.old_x[0, stranger].clone()
    op(**kw)
    assert not torch.equal(bench.old_x[0, stranger], before), "the hazard this guard covers is gone"
    with pytest.raises(AssertionError, match="state_indices has shape"):
        with validating(op):
            op(**kw)


def test_a_history_off_history_size_is_misread_and_is_refused() -> None:
    """Unguarded, 16-deep buffers called with history_size=12: the kernel strides
    them as 12 deep and reads other slots' entries. The same call on 12-deep
    copies of the same history is the right answer. (12, not 8: below 9 the
    kernel does not compile -- see `test_op_refuses_what_it_refuses`.)"""
    bench = _Bench(dict(path="small", n=4, t=4, layers=1), seed=103, exact=False)
    live = bench.live.long()
    bench.pnat[live] = torch.tensor([1, 2, 3, 4], dtype=torch.int32, device=DEV)
    kw = bench.kwargs(0, bench.inputs())
    shallow = dict(
        kw,
        old_u=kw["old_u"][:, :, :12].clone(),
        old_k=kw["old_k"][:, :, :12].clone(),
        old_G=kw["old_G"][..., :12].clone(),
        ssm_states=kw["ssm_states"].clone(),
        history_size=12,
    )
    ref = op.reference(**shallow)
    right = op(**shallow)
    _compare((right, None, None, None, None), (ref[0], None, None, None, None))
    wrong = op(**dict(kw, history_size=12))
    gap = _deviation(wrong, right) / BF16_ULP
    assert gap > 64, "the hazard this guard covers is gone"
    with pytest.raises(AssertionError, match="old_u has shape"):
        with validating(op):
            op(**dict(kw, history_size=12))


@pytest.mark.parametrize(
    "name, form",
    (
        ("history", "fewer slots"),
        ("prev_num_accepted_tokens", "fewer slots"),
        ("cache_buf_idx", "fewer slots"),
        ("prev_num_accepted_tokens", "strided"),
        ("cache_buf_idx", "strided"),
    ),
)
def test_bookkeeping_that_does_not_cover_the_pool_is_refused(name, form) -> None:
    """The kernel addresses the history buffers, `prev_num_accepted_tokens` and
    `cache_buf_idx` by the pool's slot id: one with fewer slots than `ssm_states` is read and
    written past its end for the slots beyond it, and a strided per-slot vector is read at the
    wrong elements. Only the refusal runs; the unguarded call is out of bounds."""
    bench = _Bench(dict(path="small", n=4, t=2, layers=1), seed=109, exact=False)
    kw = bench.kwargs(0, bench.inputs())
    last = bench.slots - 1
    if name == "history":
        kw = dict(kw, old_u=kw["old_u"][:last], old_k=kw["old_k"][:last], old_G=kw["old_G"][:last])
        match = "history buffers have"
    else:
        vec = kw[name]
        if form == "fewer slots":
            kw[name] = vec[:last]
        else:
            kw[name] = torch.stack([vec, vec], 1)[:, 0]
        match = f"{name} has shape"
    with pytest.raises(AssertionError, match=match):
        with validating(op):
            op(**kw)


def test_op_refuses_what_it_refuses() -> None:
    """Loud refusals; the guard admits each so the op's own error surfaces."""
    bench = _Bench(dict(path="all_layer", n=16, t=4, layers=1), seed=105, exact=False)
    bench.prepare()
    kw = bench.kwargs(0, bench.inputs())
    with pytest.raises(ValueError, match="Partitioned replay requires the all-layer commit"):
        with validating(op):
            op(**dict(kw, use_all_layer_commit=False))
    with pytest.raises(ValueError, match="use_all_layer_commit requires replay work items"):
        with validating(op):
            op(**dict(kw, replay_work_items=None, n_writes=None))
    with pytest.raises(ValueError, match="must either both be set or both be None"):
        with validating(op):
            op(**dict(kw, n_writes=None))
    with pytest.raises(AssertionError, match="output must be contiguous"):
        wide = torch.empty(16, 4, HV, 2 * V, dtype=torch.bfloat16, device=DEV)
        with validating(op):
            op(**dict(kw, output=wide[..., :V]))
    with pytest.raises(AssertionError, match="g head dimension must be contiguous"):
        with validating(op):
            op(**dict(kw, g=kw["g"].transpose(1, 2).contiguous().transpose(1, 2)))

    def shallow(depth):
        return dict(
            kw,
            history_size=depth,
            old_u=kw["old_u"][:, :, :depth].contiguous(),
            old_k=kw["old_k"][:, :, :depth].contiguous(),
            old_G=kw["old_G"][..., :depth].contiguous(),
        )

    with pytest.raises(AssertionError):  # `history_size >= T`, a bare assert
        with validating(op):
            op(**shallow(2))
    # The history tile is next_pow2(history_size) wide and is a `tl.dot`
    # reduction, which needs 16: history_size 8 (>= T = 4) fails to compile.
    with pytest.raises(CompilationError, match="K >= 16"):
        with validating(op):
            op(**shallow(8))
