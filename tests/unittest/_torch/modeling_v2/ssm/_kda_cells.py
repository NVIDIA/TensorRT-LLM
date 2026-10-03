# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""What the ssm/ entry tests share: a real ``MambaHybridCacheManagerV2`` at Kimi K3's per-rank KDA shapes, its
per-layer pools, and the references of the ops' own tests (k3_kda_qkvg's rows decoded from its Lamport buffers; a
float64 plain decode; a float64 verify that keeps each request's committed history).

Not a test file (no ``test_`` prefix): the four entry tests import it by name, as ``comm/`` imports ``_rank_job``.

Shapes: 6 heads, K = V = 128, conv width 4 (the TP16 rank slice); conv states bf16 in the ``[q | k | v]`` layout,
SSM states fp32. The manager coalesces each slot's per-layer states, so every per-layer view is a strided slot view,
the layout the model hands these ops.
"""

from __future__ import annotations

import torch

H = 6
K = V = 128
HK = H * K
W = 4
CONV_DIM = 3 * HK
PROJ = 4 * HK + K + H + 2  # the fused [q | k | v | og | f_a | b | pad] rows (3208)
K_IN = 7168
NUM_SPEC = 7
NT = NUM_SPEC + 1
LOWER_BOUND = -5.0
EPS = 1e-5
SCALE = K**-0.5
TOL_OUT = 2e-2  # bf16 outputs
TOL_STATE = 1e-3  # fp32 states: approximate exp / rcp in the kernels and the bf16 f_b gate


def sm100() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 10


def load_ops():
    """Register the C++ ops and the K3 KDA CuTe DSL ops; return the k3_kda_attn op module (its constants)."""
    import tensorrt_llm  # noqa: F401  (registers the trtllm ops)
    from tensorrt_llm._torch.cute_dsl_kernels.k3_kda_attn import op
    from tensorrt_llm._torch.cute_dsl_kernels.k3_kda_verify import op as _verify  # noqa: F401

    return op


# ---------------------------------------------------------------------------------------------------------------
# The cache objects
# ---------------------------------------------------------------------------------------------------------------
def build_manager(num_layers: int, num_spec: int | None = None, max_batch_size: int = 8):
    """A real MambaHybridCacheManagerV2 with ``num_layers`` KDA layers (and one attention layer). With ``num_spec``:
    MTP-style speculation of ``num_spec`` drafts, the KDA replay caches and the per-token states
    (``kda_token_states``) that k3_kda_verify / k3_kda_attn read and write."""
    from tensorrt_llm._torch.pyexecutor.kv_cache.mamba_cache_manager import (
        MambaHybridCacheManagerV2,
    )
    from tensorrt_llm._torch.pyexecutor.resource_manager import CacheTypeCpp, DataType
    from tensorrt_llm.llmapi.llm_args import KvCacheConfig, MTPDecodingConfig
    from tensorrt_llm.mapping import Mapping

    spec = MTPDecodingConfig(max_draft_len=num_spec) if num_spec else None
    return MambaHybridCacheManagerV2(
        mamba_d_state=K,
        mamba_d_conv=W,
        mamba_num_heads=H,
        mamba_n_groups=H,
        mamba_head_dim=V,
        mamba_num_layers=num_layers,
        mamba_layer_mask=[True] * num_layers + [False],
        mamba_cache_dtype=torch.bfloat16,
        mamba_ssm_cache_dtype=torch.float32,
        kv_cache_config=KvCacheConfig(max_tokens=512, enable_block_reuse=False),
        kv_cache_type=CacheTypeCpp.SELF,
        num_layers=1,
        num_kv_heads=4,
        head_dim=64,
        tokens_per_block=32,
        max_seq_len=128,
        max_batch_size=max_batch_size,
        mapping=Mapping(world_size=1, rank=0, tp_size=1, pp_size=1),
        dtype=DataType.HALF,
        spec_config=spec,
        layer_mask=[False] * num_layers + [True],
        vocab_size=1024,
        conv_state_layout="q_k_v",
        kda_replay_num_spec=num_spec,
        kda_token_states=num_spec is not None,
    )


def request_slots(mgr, count: int, first_id: int) -> torch.Tensor:
    """State slots of ``count`` new requests (the manager's own slot assignment), int32 on the GPU."""
    ids = list(range(first_id, first_id + count))
    mgr.add_dummy_requests(ids, token_nums=[8] * count, is_gen=False)
    slots = mgr.get_state_indices(ids, [False] * count)
    assert len(set(slots)) == count, slots
    return torch.tensor(slots, dtype=torch.int32, device="cuda")


def layer_pools(mgr, layer: int) -> dict:
    """The manager's views of one KDA layer's pools: ``conv`` bf16 [slots, 3 HK, W - 1] (q | k | v channels) and
    ``ssm`` fp32 [slots, H, V, K] (plain decode); with the replay caches also ``cs_q`` / ``cs_k`` / ``cs_v`` fp32
    [slots, HK, W - 1 + num_spec] (dim-contiguous), ``state_tok`` fp32 [slots, num_spec, H, V, K] and ``pending``,
    the accepted-draft record every layer shares (``prev_num_accepted_tokens``)."""
    pools = {"conv": mgr.get_conv_states(layer), "ssm": mgr.get_ssm_states(layer)}
    if mgr.use_kda_replay_update:
        cache = mgr.mamba_layer_cache(layer)
        pools.update(
            cs_q=cache.kda_conv_q,
            cs_k=cache.kda_conv_k,
            cs_v=cache.kda_conv_v,
            state_tok=cache.kda_state_tok,
            pending=cache.prev_num_accepted_tokens,
        )
    return pools


def fill_pools(pools: dict, seed: int) -> None:
    """Random finite contents in every slot of every pool (the pending record zeroed)."""
    g = torch.Generator(device="cuda").manual_seed(seed)

    def rnd(t, scale):
        return torch.randn(t.shape, generator=g, device="cuda") * scale

    pools["conv"].copy_(rnd(pools["conv"], 0.5).bfloat16())
    pools["ssm"].copy_(rnd(pools["ssm"], 0.05))
    for name in ("cs_q", "cs_k", "cs_v"):
        if name in pools:
            pools[name].copy_(rnd(pools[name], 0.5))
    if "state_tok" in pools:
        pools["state_tok"].zero_()
    if "pending" in pools:
        pools["pending"].zero_()


def clone_pools(pools: dict) -> dict:
    """Standalone copies with the same strides (so the same layout constraints hold), e.g. for a reference path."""
    out = {}
    for name, t in pools.items():
        c = torch.empty_strided(t.shape, t.stride(), dtype=t.dtype, device=t.device)
        c.copy_(t)
        out[name] = c
    return out


def snapshot(pools: dict) -> dict:
    return {name: t.clone() for name, t in pools.items()}


def same_rows(a: dict, b: dict, rows, names) -> bool:
    """Bit equality of the given slots (``rows``: list of ints) of the given pools."""
    return all(torch.equal(a[n][rows], b[n][rows]) for n in names)


def other_slots(num_slots: int, used) -> list:
    used = set(int(s) for s in used)
    return [s for s in range(num_slots) if s not in used]


def rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return ((a.float() - b.float()).abs().max() / b.float().abs().max().clamp_min(1e-6)).item()


# ---------------------------------------------------------------------------------------------------------------
# Weights and the projection rows
# ---------------------------------------------------------------------------------------------------------------
def make_weights(seed: int) -> dict:
    """One KDA layer's decode weights at the TP16 shapes; the conv taps are bf16 values (kda_decode reads them as
    bf16, the K3 kernels as fp32)."""
    g = torch.Generator(device="cuda").manual_seed(seed)

    def rnd(*s, scale=1.0):
        return torch.randn(*s, generator=g, device="cuda") * scale

    conv = rnd(3, HK, W, scale=0.3).bfloat16().float()
    return {
        "w": rnd(PROJ, K_IN, scale=0.02).bfloat16(),
        "w_fb": rnd(HK, K, scale=0.05).bfloat16(),
        "w_q": conv[0].contiguous(),
        "w_k": conv[1].contiguous(),
        "w_v": conv[2].contiguous(),
        "w_t": [conv[i].t().bfloat16().contiguous() for i in range(3)],
        "a_log": rnd(H, scale=0.5),
        "dt_bias": rnd(HK, scale=0.5),
        "onorm_w": (1 + 0.1 * rnd(V)).float(),
    }


def decode_rows(p1: torch.Tensor, part: torch.Tensor, buf: int, n: int) -> torch.Tensor:
    """The ``[q | k | v | og | f_a | b | pad]`` rows of tokens 0..n-1 from buffer ``buf`` of a k3_kda_qkvg launch:
    q, k and f_a are bf16 bits; v, og and b the bf16 sum of the two fp32 K-half partials."""
    qkfa = p1.view(3, 8, 2 * HK + K)[buf, :n].view(torch.bfloat16)
    parts = part.view(3, 3, 2, 8, HK)[buf].view(torch.float32)
    v, og, b = ((parts[r, 0, :n] + parts[r, 1, :n]).bfloat16() for r in range(3))
    rows = torch.zeros(n, PROJ, dtype=torch.bfloat16, device=p1.device)
    rows[:, : 2 * HK] = qkfa[:, : 2 * HK]
    rows[:, 2 * HK : 3 * HK] = v
    rows[:, 3 * HK : 4 * HK] = og
    rows[:, 4 * HK : 4 * HK + K] = qkfa[:, 2 * HK :]
    rows[:, 4 * HK + K : 4 * HK + K + H] = b[:, :H]
    return rows


# ---------------------------------------------------------------------------------------------------------------
# float64 references
# ---------------------------------------------------------------------------------------------------------------
def _gate_decay(a_log, dt_bias, g):
    """The lower-bound gate: decay = exp(lower_bound * sigmoid(exp(A_log) * (g + dt_bias))), per key channel."""
    xg = torch.exp(a_log.double()).unsqueeze(-1) * (g + dt_bias.double().view(H, K))
    return torch.exp(LOWER_BOUND * torch.sigmoid(xg))


class F64Decode:
    """The plain KDA decode in float64 torch over copies of a layer's ``conv`` / ``ssm`` pools: conv4 + SiLU, q / k
    L2 norm (q scaled), beta sigmoid, the lower-bound gate, S <- S d; S <- S + beta (v - S k) k^T; o = S q, the
    gated RMSNorm."""

    def __init__(self, wt: dict, pools: dict):
        self.wt = wt
        self.conv = pools["conv"].double().clone()
        self.state = pools["ssm"].double().clone()

    def step(self, raw_qkv, g, beta_raw, gate_raw, slots) -> torch.Tensor:
        """``raw_qkv`` [n, 3 HK] (the projection's q | k | v columns), ``g`` [n, HK] (f_b's output, before dt_bias),
        ``beta_raw`` [n, H], ``gate_raw`` [n, HK] (the output gate); returns [n, H, V]."""
        wt = self.wt
        conv_w = torch.stack([wt["w_q"], wt["w_k"], wt["w_v"]]).double()  # [3, HK, W]
        outs = torch.empty(raw_qkv.shape[0], H, V, dtype=torch.float64, device="cuda")
        for i, s in enumerate(slots.tolist()):
            new = raw_qkv[i].double().view(3, HK)
            win = self.conv[s].view(3, HK, W - 1)
            u = torch.cat([win, new.unsqueeze(-1)], dim=-1)  # oldest first
            act = (u * conv_w).sum(-1)
            act = act * torch.sigmoid(act)
            self.conv[s] = u[:, :, 1:].reshape(3 * HK, W - 1)
            q, k, v = (act[j].view(H, K) for j in range(3))
            q = q / torch.sqrt((q * q).sum(-1, keepdim=True) + 1e-6) * SCALE
            k = k / torch.sqrt((k * k).sum(-1, keepdim=True) + 1e-6)
            beta = torch.sigmoid(beta_raw[i].double())
            decay = _gate_decay(wt["a_log"], wt["dt_bias"], g[i].double().view(H, K))
            st = self.state[s] * decay.unsqueeze(1)
            res = (v - (st * k.unsqueeze(1)).sum(-1)) * beta.unsqueeze(-1)
            st = st + res.unsqueeze(-1) * k.unsqueeze(1)
            self.state[s] = st
            o = (st * q.unsqueeze(1)).sum(-1)
            rms = torch.rsqrt((o * o).mean(-1, keepdim=True) + EPS)
            gate = torch.sigmoid(gate_raw[i].double().view(H, V))
            outs[i] = o * rms * wt["onorm_w"].double() * gate
        return outs

    def step_rows(self, rows: torch.Tensor, slots) -> torch.Tensor:
        """As :meth:`step` from fused projection rows (f_b in float64, rounded to bf16 as the GEMM's output is)."""
        r = rows.double()
        g = (r[:, 4 * HK : 4 * HK + K] @ self.wt["w_fb"].double().t()).bfloat16().double()
        return self.step(
            r[:, : 3 * HK], g, r[:, 4 * HK + K : 4 * HK + K + H], r[:, 3 * HK : 4 * HK], slots
        )


def _conv_silu(win, raw, c, w):
    x = win[0][c] * w[:, 0] + win[1][c] * w[:, 1] + win[2][c] * w[:, 2] + raw[c] * w[:, 3]
    return x * torch.sigmoid(x)


class F64Verify:
    """The speculative verify in float64 over each request's committed history (raw conv inputs and the state after
    the last committed token): every request's 1 + num_spec tokens from the state after its accepted drafts. The
    conv caches start with pending 0, so their window columns 0..2 are the committed raw inputs."""

    def __init__(self, wt: dict, pools: dict, slots, num_spec: int):
        self.wt, self.num_spec = wt, num_spec
        self.slots = slots.tolist()
        self.seq = [
            [
                torch.stack([pools[c][s, :, i].double() for c in ("cs_q", "cs_k", "cs_v")])
                for i in range(3)
            ]
            for s in self.slots
        ]
        self.state = [pools["ssm"][s].double().clone() for s in self.slots]
        self.last = None

    def __call__(self, rows: torch.Tensor) -> torch.Tensor:
        """Outputs [N (1 + num_spec), H, V] for the fused projection rows of N requests."""
        wt, steps = self.wt, self.num_spec + 1
        r = rows.double()
        g_all = (r[:, 4 * HK : 4 * HK + K] @ wt["w_fb"].double().t()).bfloat16().double()
        wq, wk, wv = (wt[n].double() for n in ("w_q", "w_k", "w_v"))
        out = torch.empty(rows.shape[0], H, V, dtype=torch.float64, device="cuda")
        self.last = []
        for n in range(len(self.slots)):
            seq, s_cur = list(self.seq[n]), self.state[n]
            states, raws = [], []
            for t in range(steps):
                row = n * steps + t
                raw = r[row, : 3 * HK].view(3, HK)
                win = seq[-3:]
                q = _conv_silu(win, raw, 0, wq).view(H, K)
                k = _conv_silu(win, raw, 1, wk).view(H, K)
                v = _conv_silu(win, raw, 2, wv).view(H, V)
                q = q * torch.rsqrt((q * q).sum(-1, keepdim=True) + 1e-6) * SCALE
                k = k * torch.rsqrt((k * k).sum(-1, keepdim=True) + 1e-6)
                beta = torch.sigmoid(r[row, 4 * HK + K : 4 * HK + K + H])
                decay = _gate_decay(wt["a_log"], wt["dt_bias"], g_all[row].view(H, K))
                sd = s_cur * decay[:, None, :]
                vn = v - torch.einsum("hvk,hk->hv", sd, k)
                s_cur = sd + beta[:, None, None] * vn[:, :, None] * k[:, None, :]
                o = torch.einsum("hvk,hk->hv", s_cur, q)
                rms = torch.rsqrt((o * o).mean(-1, keepdim=True) + EPS)
                gate = torch.sigmoid(r[row, 3 * HK : 4 * HK].view(H, V))
                out[row] = o * rms * wt["onorm_w"].double() * gate
                states.append(s_cur)
                raws.append(raw)
                seq.append(raw)
            self.last.append((states, raws))
        return out

    def commit(self, accepted) -> None:
        """The sampler accepted ``accepted[n]`` drafts: the golden token and those drafts commit."""
        for n, p in enumerate(accepted):
            states, raws = self.last[n]
            self.seq[n] = (self.seq[n] + raws[: p + 1])[-3:]
            self.state[n] = states[p]


def pending_schedule(num_requests: int, num_spec: int, rnd: int) -> list:
    """Accepted drafts per request after round ``rnd``: every count 0..num_spec over the rounds, different per
    request."""
    return [(3 * n + 5 * rnd + 1 + (rnd * n) % 3) % (num_spec + 1) for n in range(num_requests)]
