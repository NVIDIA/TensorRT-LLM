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
  bookkeeping surfaces as an output that drifts off it.
"""

from __future__ import annotations

import zlib

import pytest
import torch
import torch.nn.functional as F

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
from tensorrt_llm._torch.modules.fla.cache_manager import (
    ReplayStateUpdateMetadata,
    _advance_replay_state,
)
from tensorrt_llm._torch.modules.fla.cached_replay import (
    CACHED_REPLAY_PARTITION_MIN_BATCH_SIZE,
    commit_gdn_cached_replay_history_layers,
)
from tensorrt_llm._torch.modules.mamba.mamba2_metadata import _build_replay_work_items_triton

__extra_import_path__ = ["../../../.."]
from _validating import ULP, certifying, validating  # noqa: E402,F401 — path above; fixture by name

assert torch.cuda.is_available(), (
    "fused_recurrent_gated_delta_rule_cached_replay_update requires a CUDA device"
)

DEV = "cuda"
H, HV, K, V = NUM_K_HEADS, NUM_V_HEADS, HEAD_K_DIM, HEAD_V_DIM
W = PACKED_QKV_WIDTH
HIST = HISTORY_SIZE
BF16_ULP = ULP[torch.bfloat16]

#: This checkpoint's 36 GDN layers of `linear_attn.A_log` / `linear_attn.dt_bias`
#: (bf16, 48 each): mean, standard deviation, range.
A_LOG_STATS = (1.006, 1.487, (-7.125, 5.0625))
DT_BIAS_STATS = (-4.411, 1.732, (-11.4375, 2.53125))

#: Element gate of the bf16 members, in bf16 ulps of the block they belong to.
_ELEMENT_ULPS = 4.0
#: Relative-RMS gate of the output, in bf16 ulps.
_OUTPUT_RMS_ULPS = 1.5
#: Element gate of an fp32 checkpoint state, relative to its (slot, head) scale.
_STATE_TOL = 2.0**-13
#: Element gate of a bf16 checkpoint state, in bf16 ulps of its (slot, head) scale.
_BF16_STATE_ULPS = 2.0
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
    rounding into relative noise, and the output's relative RMS at 1.5 ulps. An fp32
    checkpoint state is fp32 by construction: the commit splits the decayed
    history update into a bf16 hi and lo part, ~16 significant bits per term, and
    up to 16 terms each within 2^-17 give the 2^-13 gate, relative to one (slot,
    head) state's absmax. A bf16 checkpoint is that fp32 commit rounded once, as the
    reference's is, so the two stored values are equal or adjacent, and adjacent bf16
    values are at most 2 bf16 ulps of their head's absmax apart: the bf16 gate is one
    representable step. `G` only ever enters as exp(G_t - G_j), so an absolute
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
            bf16 = got.dtype == torch.bfloat16
            _gate(name, got, want, _BF16_STATE_ULPS * BF16_ULP if bf16 else _STATE_TOL)
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
        live_rows = torch.ones(n, dtype=torch.bool)
        dup = tuple(spec.get("dup", ()))
        if dup:
            chosen[list(dup)] = int(chosen[dup[0]])
            # The padding rows' shared slot is nobody's state: the bookkeeping never names it.
            live_rows[list(dup)] = False
        self.state_indices = chosen.to(DEV)
        self.live = self.state_indices[live_rows.to(DEV)].clone()

        # [slots, layers, HV, V, K]: each layer's state is a slot-strided view,
        # the affine layout whose strides the all-layer commit is handed.
        self.backing = (self._randn(S, L, HV, V, K) * 0.5).to(
            spec.get("state_dtype", torch.float32)
        )
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
        a = self._randn(n, t, HV, dtype=torch.bfloat16)
        b = self._randn(n, t, HV, dtype=torch.bfloat16)
        return dict(q=q, k=k, v=v, a=a, b=b, packed=packed)

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
            launch_with_pdl=False,
            replay_work_items=self.work_items,
            n_writes=self.n_writes,
            packed_qkv=act["packed"],
            use_all_layer_commit=self.path == "all_layer",
            output=None,
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
        """Accepted tokens per live row, bonus token included, by `spec["accept"]`."""
        mode = self.spec.get("accept", "random")
        rows = self.live.numel()
        if mode == "full":
            return torch.full((rows,), self.t, dtype=torch.int32, device=DEV)
        assert mode == "random", mode
        return torch.randint(1, self.t + 1, (rows,), generator=self.cpu, dtype=torch.int32).to(DEV)

    def accept(self, accepted):
        _advance_replay_state(self.meta, self.live, accepted)
        if self.exact:
            self.x_pnat[self.live.long()] += accepted

    def writes(self):
        """Where this step's calls append, by the documented rule, which slots wrap, and how
        many append exactly up to the history's end: `[slots, 2, HIST]` mask, wrapping slot
        ids, the `pnat + t == HIST` count."""
        s = self.live.long()
        p, b = self.pnat[s].long(), self.buf[s].long()
        wrap = p + self.t > HIST
        half, start = torch.where(wrap, 1 - b, b), torch.where(wrap, 0, p)
        mask = torch.zeros(self.slots, 2, HIST, dtype=torch.bool, device=DEV)
        for t in range(self.t):
            mask[s, half, start + t] = True
        return mask, s[wrap], int((p + self.t == HIST).sum())


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
    wraps = appends = to_the_end = 0
    for _ in range(bench.steps):
        mask, wrapped, at_end = bench.writes()
        wraps += wrapped.numel()
        to_the_end += at_end
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
    if cell.spec.get("accept") == "full":
        live = bench.live.numel()
        assert wraps >= 3 * live and to_the_end >= 3 * live, (
            f"full acceptance should wrap every row at least 3 times and append up to the "
            f"history's end as often ({wraps}, {to_the_end} over {live} rows)"
        )


# ── CUDA graphs ───────────────────────────────────────────────────────────


def test_cuda_graph_replay_on_new_data_is_bitwise_eager() -> None:
    """Two layers of 32 sequences x 8 tokens over an fp32 checkpoint captured with the
    all-layer commit after them; replayed three times on new activations, with acceptance
    in between, against the same steps run eagerly on a twin state."""
    n, t = 32, 8
    assert ALL_LAYER_COMMIT_MIN_BATCH == CACHED_REPLAY_PARTITION_MIN_BATCH_SIZE
    assert n >= ALL_LAYER_COMMIT_MIN_BATCH
    spec = dict(path="all_layer", n=n, t=t, layers=2, slots=40, state_dtype=torch.float32)
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
