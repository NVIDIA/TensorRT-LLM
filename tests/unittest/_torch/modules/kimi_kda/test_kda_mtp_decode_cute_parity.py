# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Parity: in-tree ``trtllm::kda_mtp_decode`` CuTe kernel vs two references.

References:

1. A pure-torch fp32 CPU golden (vendored from the kernel drop's
   ``cpu_reference`` self-check).
2. An FLA sequential reference built from the exact op sequence
   ``KimiKDALinearAttention.forward_verify`` uses in-tree: per-step fp32 causal
   conv + SiLU followed by ``fla.ops.kda.fused_recurrent_kda`` with
   ``use_qk_l2norm/use_gate/use_beta_sigmoid`` in kernel, ``lower_bound``,
   ``state_v_first=True``.

Agreement of all three establishes both that the kernel is internally
correct and that it computes the same function the model's sequential
verify path computes — i.e. it is a drop-in replacement.

Round-2 tests exercise the kernel's replay mode (``num_accepted_tokens``
mixed per request), chained from round-1 CPU-golden outputs. H=8 (K3
state-TP4) and H=6 (TP16) are the production per-rank head counts outside
the drop's benchmark-tuned set {2, 12, 32}; the vendored v_row hoist fix
makes them compile, and this test validates them numerically.

Requires: 1 GPU (sm100 for the CuTe kernel), fla-core, nvidia-cutlass-dsl,
cuda-bindings. Skips cleanly otherwise.
"""

import pytest
import torch
import torch.nn.functional as F

_HAVE_DEPS = True
_DEP_ERR = None
try:
    import cuda.bindings.driver  # noqa: F401
    import cutlass  # noqa: F401
    from fla.ops.kda import fused_recurrent_kda  # noqa: F401
except ImportError as e:
    _HAVE_DEPS = False
    _DEP_ERR = str(e)


def _is_blackwell():
    if not torch.cuda.is_available():
        return False
    prop = torch.cuda.get_device_properties(0)
    return prop.major * 10 + prop.minor in (100, 103)


pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU"),
    pytest.mark.skipif(not _is_blackwell(), reason="needs sm100/sm103"),
    pytest.mark.skipif(not _HAVE_DEPS, reason=f"deps: {_DEP_ERR}"),
]

# (N, H): the drop's benchmark-tuned shapes plus K3's production per-rank
# head counts (96 KDA heads: 12 at TP8, 8 at state-TP 12, 6 at TP16).
SHAPES = [
    (128, 2),
    (32, 12),
    (32, 32),
    (32, 8),
    (32, 6),
]
M = 2  # NUM_SPEC — spec-token count per verify round


def _silu(x):
    return x * torch.sigmoid(x)


def make_conv_data(B, H, K=128, V=128, M=2, W=4, lower_bound=-5.0, scale=None, seed=2025):
    """Random inputs + caches in the op's layout contract (from the drop)."""
    torch.manual_seed(seed)
    scale = K**-0.5 if scale is None else scale
    device = "cuda"
    dim = H * K
    T = 2 * M + 1
    T_total = B * T
    S = W - 1 + M

    x_q = torch.randn(1, T_total, H, K, dtype=torch.bfloat16, device=device)
    x_k = torch.randn(1, T_total, H, K, dtype=torch.bfloat16, device=device)
    x_v = torch.randn(1, T_total, H, V, dtype=torch.bfloat16, device=device)
    g = torch.randn(1, T_total, H, K, dtype=torch.bfloat16, device=device)
    beta = torch.randn(1, T_total, H, dtype=torch.bfloat16, device=device)
    w_q = torch.randn(dim, W, dtype=torch.float32, device=device)
    w_k = torch.randn(dim, W, dtype=torch.float32, device=device)
    w_v = torch.randn(dim, W, dtype=torch.float32, device=device)
    A_log = torch.randn(H, dtype=torch.float32, device=device)
    dt_bias = torch.randn(dim, dtype=torch.float32, device=device)
    # dim-contiguous extended conv caches: allocate [B, S, dim], transpose.
    cs_q = torch.randn(B, S, dim, dtype=torch.float32, device=device).transpose(1, 2)
    cs_k = torch.randn(B, S, dim, dtype=torch.float32, device=device).transpose(1, 2)
    cs_v = torch.randn(B, S, dim, dtype=torch.float32, device=device).transpose(1, 2)
    initial_state_kfirst = torch.randn(B, H, K, V, dtype=torch.float32, device=device)
    initial_state_cute = initial_state_kfirst.permute(0, 1, 3, 2).contiguous()
    cu_seqlens = torch.arange(0, B * T + 1, T, dtype=torch.int32, device=device)
    ssm_state_indices = torch.arange(B, dtype=torch.int32, device=device)
    num_accepted_tokens = torch.zeros(B, dtype=torch.int32, device=device)
    qkg_cache = torch.zeros(B, M, 3, dim, dtype=torch.float32, device=device)
    v_cache = torch.zeros(B, M, H * V, dtype=torch.float32, device=device)
    beta_cache = torch.zeros(B, M, H, dtype=torch.float32, device=device)
    return {
        "x_q": x_q,
        "x_k": x_k,
        "x_v": x_v,
        "w_q": w_q,
        "w_k": w_k,
        "w_v": w_v,
        "g": g,
        "beta": beta,
        "A_log": A_log,
        "dt_bias": dt_bias,
        "cs_q": cs_q,
        "cs_k": cs_k,
        "cs_v": cs_v,
        "initial_state_kfirst": initial_state_kfirst,
        "initial_state_cute": initial_state_cute,
        "qkg_cache": qkg_cache,
        "v_cache": v_cache,
        "beta_cache": beta_cache,
        "cu_seqlens": cu_seqlens,
        "ssm_state_indices": ssm_state_indices,
        "num_accepted_tokens": num_accepted_tokens,
        "B": B,
        "H": H,
        "K": K,
        "V": V,
        "M": M,
        "W": W,
        "T": T,
        "T_total": T_total,
        "lower_bound": lower_bound,
        "scale": scale,
    }


def cpu_reference(data):
    """fp32 pure-torch golden with replay semantics (from the drop)."""
    B, H, K, V, M, W = (data["B"], data["H"], data["K"], data["V"], data["M"], data["W"])
    T = data["T"]
    lower_bound = data["lower_bound"]
    scale = data["scale"]
    w_q = data["w_q"].float().cpu()
    w_k = data["w_k"].float().cpu()
    w_v = data["w_v"].float().cpu()
    A_log = data["A_log"].float().cpu()
    dt_bias = data["dt_bias"].float().cpu()
    x_q = data["x_q"].float().cpu()
    x_k = data["x_k"].float().cpu()
    x_v = data["x_v"].float().cpu()
    g = data["g"].float().cpu()
    beta = data["beta"].float().cpu()
    cs_q = data["cs_q"].float().cpu().clone()
    cs_k = data["cs_k"].float().cpu().clone()
    cs_v = data["cs_v"].float().cpu().clone()
    qkg_cache = data["qkg_cache"].float().cpu().clone()
    v_cache = data["v_cache"].float().cpu().clone()
    beta_cache = data["beta_cache"].float().cpu().clone()
    ht = data["initial_state_kfirst"].float().cpu().clone()
    ht_commit = ht.clone()
    out = torch.zeros(1, B * T, H, V, dtype=torch.float32)

    for n in range(B):
        bos = n * T
        slot = n
        commit_len = int(data["num_accepted_tokens"][n].item())
        T_loop = commit_len + 1 + M
        for h in range(H):
            hk = h * K
            hv = h * V
            hist_q = cs_q[slot, hk : hk + K, : W - 1].clone()
            hist_k = cs_k[slot, hk : hk + K, : W - 1].clone()
            hist_v = cs_v[slot, hv : hv + V, : W - 1].clone()
            h_state = ht[slot, h].clone()
            for i_t in range(T_loop):
                if i_t < commit_len:
                    q_t = qkg_cache[slot, i_t, 0, hk : hk + K]
                    k_t = qkg_cache[slot, i_t, 1, hk : hk + K]
                    gk_t = qkg_cache[slot, i_t, 2, hk : hk + K]
                    v_t = v_cache[slot, i_t, hv : hv + V]
                    beta_t = beta_cache[slot, i_t, h]
                    xq_raw = cs_q[slot, hk : hk + K, W - 1 + i_t]
                    xk_raw = cs_k[slot, hk : hk + K, W - 1 + i_t]
                    xv_raw = cs_v[slot, hv : hv + V, W - 1 + i_t]
                    hist_q = torch.cat([hist_q[:, 1:], xq_raw.unsqueeze(-1)], dim=1)
                    hist_k = torch.cat([hist_k[:, 1:], xk_raw.unsqueeze(-1)], dim=1)
                    hist_v = torch.cat([hist_v[:, 1:], xv_raw.unsqueeze(-1)], dim=1)
                else:
                    token = bos + i_t
                    xq_raw = x_q[0, token, h]
                    xk_raw = x_k[0, token, h]
                    xv_raw = x_v[0, token, h]
                    cq = (torch.cat([hist_q, xq_raw.unsqueeze(-1)], dim=-1) * w_q[hk : hk + K]).sum(
                        dim=-1
                    )
                    ck = (torch.cat([hist_k, xk_raw.unsqueeze(-1)], dim=-1) * w_k[hk : hk + K]).sum(
                        dim=-1
                    )
                    cv = (torch.cat([hist_v, xv_raw.unsqueeze(-1)], dim=-1) * w_v[hv : hv + V]).sum(
                        dim=-1
                    )
                    q_t = F.normalize(cq / (1.0 + torch.exp(-cq)), p=2, dim=-1) * scale
                    k_t = F.normalize(ck / (1.0 + torch.exp(-ck)), p=2, dim=-1)
                    v_t = cv / (1.0 + torch.exp(-cv))
                    gr = g[0, token, h] + dt_bias[hk : hk + K]
                    gk_t = lower_bound * torch.sigmoid(gr * torch.exp(A_log[h]))
                    beta_t = torch.sigmoid(beta[0, token, h])
                    hist_q = torch.cat([hist_q[:, 1:], xq_raw.unsqueeze(-1)], dim=1)
                    hist_k = torch.cat([hist_k[:, 1:], xk_raw.unsqueeze(-1)], dim=1)
                    hist_v = torch.cat([hist_v[:, 1:], xv_raw.unsqueeze(-1)], dim=1)

                decay = torch.exp(gk_t)
                h_state = h_state * decay.unsqueeze(1)
                sum_hk = (h_state * k_t.unsqueeze(1)).sum(dim=0)
                v_new = (v_t - sum_hk) * beta_t
                h_state = h_state + k_t.unsqueeze(1) * v_new.unsqueeze(0)
                o_t = (h_state * q_t.unsqueeze(1)).sum(dim=0)
                if i_t >= commit_len:
                    out[0, bos + i_t, h] = o_t
                if i_t == commit_len:
                    ht_commit[slot, h] = h_state
                    cs_q[slot, hk : hk + K, : W - 1] = hist_q
                    cs_k[slot, hk : hk + K, : W - 1] = hist_k
                    cs_v[slot, hv : hv + V, : W - 1] = hist_v
                if i_t > commit_len:
                    cache_pos = i_t - commit_len - 1
                    qkg_cache[slot, cache_pos, 0, hk : hk + K] = q_t
                    qkg_cache[slot, cache_pos, 1, hk : hk + K] = k_t
                    qkg_cache[slot, cache_pos, 2, hk : hk + K] = gk_t
                    v_cache[slot, cache_pos, hv : hv + V] = v_t
                    beta_cache[slot, cache_pos, h] = beta_t
                    cs_q[slot, hk : hk + K, W - 1 + cache_pos] = xq_raw
                    cs_k[slot, hk : hk + K, W - 1 + cache_pos] = xk_raw
                    cs_v[slot, hv : hv + V, W - 1 + cache_pos] = xv_raw

    return {
        "out": out,
        "recurrent_state": ht_commit,
        "qkg_cache": qkg_cache,
        "v_cache": v_cache,
        "beta_cache": beta_cache,
        "cs_q": cs_q,
        "cs_k": cs_k,
        "cs_v": cs_v,
    }


def cute_run(data, zero_accepted_hint=False):
    """Run the in-tree op on cloned caches; return the drop-format dict."""
    import tensorrt_llm._torch.custom_ops.cute_dsl_kimi_k3_kda_mtp_ops  # noqa: F401

    state = data["initial_state_cute"].clone()
    cs = {}
    for name in ("cs_q", "cs_k", "cs_v"):
        src = data[name]
        dst = torch.empty(
            src.shape[0], src.shape[2], src.shape[1], dtype=src.dtype, device=src.device
        ).transpose(1, 2)
        dst.copy_(src)
        cs[name] = dst
    qkg_cache = data["qkg_cache"].clone()
    v_cache = data["v_cache"].clone()
    beta_cache = data["beta_cache"].clone()
    out = torch.ops.trtllm.kda_mtp_decode(
        x_q=data["x_q"],
        x_k=data["x_k"],
        x_v=data["x_v"],
        w_q=data["w_q"],
        w_k=data["w_k"],
        w_v=data["w_v"],
        cs_q=cs["cs_q"],
        cs_k=cs["cs_k"],
        cs_v=cs["cs_v"],
        g=data["g"],
        beta=data["beta"],
        A_log=data["A_log"],
        dt_bias=data["dt_bias"],
        recurrent_state=state,
        qkg_cache=qkg_cache,
        v_cache=v_cache,
        beta_cache=beta_cache,
        ssm_state_indices=data["ssm_state_indices"],
        cu_seqlens=data["cu_seqlens"],
        num_spec=data["M"],
        num_accepted_tokens=data["num_accepted_tokens"],
        lower_bound=data["lower_bound"],
        scale=data["scale"],
        zero_accepted_hint=zero_accepted_hint,
    )
    return {
        "out": out,
        # committed state back in K-first layout for CPU-golden comparison
        "recurrent_state": state.permute(0, 1, 3, 2).contiguous(),
        "state_v_first": state,
        "qkg_cache": qkg_cache,
        "v_cache": v_cache,
        "beta_cache": beta_cache,
        "cs_q": cs["cs_q"],
        "cs_k": cs["cs_k"],
        "cs_v": cs["cs_v"],
    }


def _fla_sequential_reference(data, num_accepted):
    """Per-request sequential conv+SiLU (fp32 torch) + fused_recurrent_kda.

    Mirrors ``KimiKDALinearAttention.forward_verify``'s op sequence, extended with
    the replay prefix: for request ``n`` with ``a = num_accepted[n]``, the
    processed token sequence is ``a`` cached raw tokens (re-convolved from
    the extended conv-cache slots) followed by the ``1 + M`` new tokens.
    Returns out rows (new tokens only) and the committed state (after the
    first new token, FLA/cute ``[B, H, V, K]`` layout).
    """
    from fla.ops.kda import fused_recurrent_kda

    B, H, K, V, W = data["B"], data["H"], data["K"], data["V"], data["W"]
    T = data["T"]
    dim = H * K
    dev = data["x_q"].device
    out = torch.zeros(1, B * T, H, V, dtype=torch.float32, device=dev)
    committed = torch.zeros(B, H, V, K, dtype=torch.float32, device=dev)

    w_q, w_k, w_v = (data[k].float() for k in ("w_q", "w_k", "w_v"))
    x_q, x_k, x_v = (data[k].float() for k in ("x_q", "x_k", "x_v"))
    g_all, beta_all = data["g"], data["beta"]

    for n in range(B):
        a = int(num_accepted[n])
        bos = n * T
        hist = {
            "q": data["cs_q"][n, :, : W - 1].float().clone(),
            "k": data["cs_k"][n, :, : W - 1].float().clone(),
            "v": data["cs_v"][n, :, : W - 1].float().clone(),
        }
        state = data["initial_state_cute"][n : n + 1].float().clone()

        for i_t in range(a + 1 + M):
            if i_t < a:  # replay a cached token (raw x from cache slots)
                xq = data["cs_q"][n, :, W - 1 + i_t].float()
                xk = data["cs_k"][n, :, W - 1 + i_t].float()
                xv = data["cs_v"][n, :, W - 1 + i_t].float()
                tok = None
                g_t = data["qkg_cache"][n, i_t, 2].float()
                beta_t = data["beta_cache"][n, i_t].float()
                replay = True
            else:
                tok = bos + i_t
                xq = x_q[0, tok].reshape(dim)
                xk = x_k[0, tok].reshape(dim)
                xv = x_v[0, tok].reshape(H * V)
                replay = False

            def conv_step(hist_s, x_raw, w):
                window = torch.cat([hist_s, x_raw.unsqueeze(-1)], dim=-1)
                y = (window * w).sum(dim=-1)
                return y, window[:, 1:]

            cq, hist["q"] = conv_step(hist["q"], xq, w_q)
            ck, hist["k"] = conv_step(hist["k"], xk, w_k)
            cv, hist["v"] = conv_step(hist["v"], xv, w_v)

            if replay:
                # Replayed tokens use the cached post-processed k/g/v/beta
                # exactly as the kernel does (delta rule applied directly).
                k_t = data["qkg_cache"][n, i_t, 1].float().view(H, K)
                gk_t = g_t.view(H, K)
                v_t = data["v_cache"][n, i_t].float().view(H, V)
                st = state[0]
                decay = torch.exp(gk_t)
                st = st * decay.unsqueeze(1)
                sum_hk = torch.einsum("hvk,hk->hv", st, k_t)
                v_new = (v_t - sum_hk) * beta_t.unsqueeze(-1)
                st = st + torch.einsum("hk,hv->hvk", k_t, v_new)
                state = st.unsqueeze(0)
            else:
                # fp32 hand-off into FLA: the comparison target is the
                # mathematical function, not the model's bf16 dataflow.
                q_in = _silu(cq).view(1, 1, H, K)
                k_in = _silu(ck).view(1, 1, H, K)
                v_in = _silu(cv).view(1, 1, H, V)
                o_t, state = fused_recurrent_kda(
                    q=q_in,
                    k=k_in,
                    v=v_in,
                    g=g_all[0, tok].view(1, 1, H, K),
                    beta=beta_all[0, tok].view(1, 1, H).float(),
                    A_log=data["A_log"],
                    dt_bias=data["dt_bias"],
                    initial_state=state,
                    output_final_state=True,
                    use_qk_l2norm_in_kernel=True,
                    use_gate_in_kernel=True,
                    use_beta_sigmoid_in_kernel=True,
                    lower_bound=data["lower_bound"],
                    state_v_first=True,
                )
                out[0, tok] = o_t[0, 0].float()
                if i_t == a:  # first new (golden) token -> committed state
                    committed[n] = state[0].float()
    return out, committed


def _cute_layout_state(cpu_out):
    # CPU golden reports K-first [B, H, K, V]; cute/FLA layout is [B,H,V,K]
    return cpu_out["recurrent_state"].permute(0, 1, 3, 2).contiguous()


def _assert_close(name, a, b, atol, rtol=0.0):
    diff = (a.float() - b.float()).abs()
    denom = b.float().abs().clamp_min(1.0)
    ok = (diff <= atol + rtol * denom).all()
    assert ok, (
        f"{name}: max_abs={diff.max().item():.3e} "
        f"(atol={atol}, worst rel={((diff / denom).max()):.3e})"
    )


@pytest.mark.parametrize("B,H", SHAPES, ids=lambda v: str(v))
def test_round1_zero_accepted(B, H):
    """Fresh verify round (no replay): kernel vs CPU golden vs FLA seq."""
    data = make_conv_data(B, H, M=M, seed=2025)
    T = data["T"]

    cpu = {k: v.cuda() for k, v in cpu_reference(data).items()}
    cute_out = cute_run(data)
    fla_out, fla_committed = _fla_sequential_reference(data, data["num_accepted_tokens"].cpu())

    new_rows = torch.cat([torch.arange(n * T, n * T + 1 + M) for n in range(B)]).cuda()

    # Kernel vs the fp32 golden (tight: fp32 accumulation).
    _assert_close(
        "out(cute vs cpu)", cute_out["out"][0, new_rows], cpu["out"][0, new_rows], atol=2e-2
    )
    _assert_close(
        "state(cute vs cpu)", cute_out["recurrent_state"], cpu["recurrent_state"], atol=1e-4
    )
    for name in ("qkg_cache", "v_cache", "beta_cache", "cs_q", "cs_k", "cs_v"):
        _assert_close(f"{name}(cute vs cpu)", cute_out[name], cpu[name], atol=2e-2)

    # Kernel vs the FLA sequential path (the in-tree fused verify math).
    _assert_close(
        "out(cute vs fla)",
        cute_out["out"][0, new_rows].float(),
        fla_out[0, new_rows],
        atol=5e-2,
        rtol=5e-2,
    )
    _assert_close(
        "state(cute vs fla)", cute_out["state_v_first"], fla_committed, atol=5e-3, rtol=5e-3
    )


@pytest.mark.parametrize("B,H", [(128, 2), (32, 12), (32, 6)], ids=lambda v: str(v))
def test_round2_replay(B, H):
    """Replay round: mixed num_accepted per request, chained from a CPU-
    golden round 1. Validates the kernel's cache-replay state math (the
    path the drop's own self-check never exercised)."""
    data = make_conv_data(B, H, M=M, seed=7)
    T = data["T"]

    # Round 1 (all zero accepted) on the CPU golden to produce the caches
    # and committed state that seed round 2.
    r1 = cpu_reference(data)

    # Round 2 inputs: fresh tokens, caches/state/conv from round 1.
    data2 = make_conv_data(B, H, M=M, seed=8)
    for name in ("qkg_cache", "v_cache", "beta_cache"):
        data2[name] = r1[name].cuda().contiguous()
    for name in ("cs_q", "cs_k", "cs_v"):
        # Preserve the contract's dim-contiguous (transposed) layout.
        src = r1[name].cuda()
        dst = torch.empty(
            src.shape[0], src.shape[2], src.shape[1], dtype=src.dtype, device=src.device
        ).transpose(1, 2)
        dst.copy_(src)
        data2[name] = dst
    data2["initial_state_kfirst"] = r1["recurrent_state"].cuda().contiguous()
    data2["initial_state_cute"] = r1["recurrent_state"].cuda().permute(0, 1, 3, 2).contiguous()
    # Mixed acceptance: 0, 1, 2 cycling across requests.
    accept = torch.arange(B, dtype=torch.int32) % (M + 1)
    data2["num_accepted_tokens"] = accept.cuda()

    cpu2 = {k: v.cuda() for k, v in cpu_reference(data2).items()}
    cute2 = cute_run(data2)
    fla_out2, fla_committed2 = _fla_sequential_reference(data2, accept)

    rows = torch.cat(
        [torch.arange(n * T + int(accept[n]), n * T + int(accept[n]) + 1 + M) for n in range(B)]
    ).cuda()

    _assert_close("out2(cute vs cpu)", cute2["out"][0, rows], cpu2["out"][0, rows], atol=2e-2)
    _assert_close(
        "state2(cute vs cpu)", cute2["recurrent_state"], cpu2["recurrent_state"], atol=1e-4
    )
    _assert_close(
        "out2(cute vs fla)", cute2["out"][0, rows].float(), fla_out2[0, rows], atol=5e-2, rtol=5e-2
    )
    _assert_close(
        "state2(cute vs fla)", cute2["state_v_first"], fla_committed2, atol=5e-3, rtol=5e-3
    )


@pytest.mark.parametrize("B,H", [(32, 12)], ids=lambda v: str(v))
def test_zero_accepted_hint_variant(B, H):
    """The zero_accepted_hint fast variant matches the general variant."""
    data = make_conv_data(B, H, M=M, seed=11)
    general = cute_run(data, zero_accepted_hint=False)
    fast = cute_run(data, zero_accepted_hint=True)
    for name in (
        "out",
        "recurrent_state",
        "qkg_cache",
        "v_cache",
        "beta_cache",
        "cs_q",
        "cs_k",
        "cs_v",
    ):
        _assert_close(f"{name}(fast vs general)", fast[name], general[name], atol=1e-5)


def test_misaligned_state_indices_rejected_after_aligned_warmup():
    """The op enforces the alignment contract supplied by metadata prep."""
    data = make_conv_data(B=1, H=6, M=M, seed=13)
    cute_run(data)

    index_storage = torch.empty(2, dtype=torch.int32, device="cuda")
    index_storage[0] = -1
    index_storage[1:].copy_(data["ssm_state_indices"])
    misaligned_indices = index_storage[1:]
    assert misaligned_indices.is_contiguous()
    assert misaligned_indices.data_ptr() % 16 != 0

    misaligned_data = dict(data)
    misaligned_data["ssm_state_indices"] = misaligned_indices
    with pytest.raises(AssertionError, match="16-byte aligned"):
        cute_run(misaligned_data)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v", "-x"]))
