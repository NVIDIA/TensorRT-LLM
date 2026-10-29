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

Requires: 1 GPU (sm100/sm103/sm107), fla-core, nvidia-cutlass-dsl,
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
    return prop.major * 10 + prop.minor in (100, 103, 107)


pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU"),
    pytest.mark.skipif(not _is_blackwell(), reason="needs sm100/sm103/sm107"),
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
    # raw conv inputs are cached in bf16 like the projections that produce them
    cs_q = torch.randn(B, S, dim, dtype=torch.bfloat16, device=device).transpose(1, 2)
    cs_k = torch.randn(B, S, dim, dtype=torch.bfloat16, device=device).transpose(1, 2)
    cs_v = torch.randn(B, S, dim, dtype=torch.bfloat16, device=device).transpose(1, 2)
    initial_state_kfirst = torch.randn(B, H, K, V, dtype=torch.float32, device=device)
    initial_state_cute = initial_state_kfirst.permute(0, 1, 3, 2).contiguous()
    cu_seqlens = torch.arange(0, B * T + 1, T, dtype=torch.int32, device=device)
    ssm_state_indices = torch.arange(B, dtype=torch.int32, device=device)
    num_accepted_tokens = torch.zeros(B, dtype=torch.int32, device=device)
    # replay caches in the runtime's dtypes: k / v / beta bf16, gate fp32
    k_cache = torch.zeros(B, M, dim, dtype=torch.bfloat16, device=device)
    g_cache = torch.zeros(B, M, dim, dtype=torch.float32, device=device)
    v_cache = torch.zeros(B, M, H * V, dtype=torch.bfloat16, device=device)
    beta_cache = torch.zeros(B, M, H, dtype=torch.bfloat16, device=device)
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
        "k_cache": k_cache,
        "g_cache": g_cache,
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
    # fp32 working copies; new entries are rounded to the caches' dtypes like the kernel does
    k_cache = data["k_cache"].float().cpu().clone()
    g_cache = data["g_cache"].float().cpu().clone()
    v_cache = data["v_cache"].float().cpu().clone()
    beta_cache = data["beta_cache"].float().cpu().clone()
    k_dtype, v_dtype, beta_dtype = (
        data["k_cache"].dtype,
        data["v_cache"].dtype,
        data["beta_cache"].dtype,
    )
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
                    # replayed drafts produce no output; their q is not cached
                    q_t = torch.zeros(K, dtype=torch.float32)
                    k_t = k_cache[slot, i_t, hk : hk + K]
                    gk_t = g_cache[slot, i_t, hk : hk + K]
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
                    k_cache[slot, cache_pos, hk : hk + K] = k_t.to(k_dtype).float()
                    g_cache[slot, cache_pos, hk : hk + K] = gk_t
                    v_cache[slot, cache_pos, hv : hv + V] = v_t.to(v_dtype).float()
                    beta_cache[slot, cache_pos, h] = beta_t.to(beta_dtype).float()
                    cs_q[slot, hk : hk + K, W - 1 + cache_pos] = xq_raw
                    cs_k[slot, hk : hk + K, W - 1 + cache_pos] = xk_raw
                    cs_v[slot, hv : hv + V, W - 1 + cache_pos] = xv_raw

    return {
        "out": out,
        "recurrent_state": ht_commit,
        "k_cache": k_cache,
        "g_cache": g_cache,
        "v_cache": v_cache,
        "beta_cache": beta_cache,
        "cs_q": cs_q,
        "cs_k": cs_k,
        "cs_v": cs_v,
    }


def cute_run(
    data,
    zero_accepted_hint=False,
    packed_token_layout=False,
    out=None,
    fuse_output_norm=False,
    quantize_output=False,
    beta_cache_override=None,
    state_override=None,
):
    """Run the op, optionally substituting a pre-populated beta-cache view or a
    caller-allocated recurrent-state tensor (e.g. a strided slice of a larger cache)."""
    import tensorrt_llm._torch.custom_ops.cute_dsl_kimi_k3_kda_mtp_ops  # noqa: F401

    if state_override is None:
        state = data["initial_state_cute"].clone()
    else:
        state = state_override
        state.copy_(data["initial_state_cute"])
    cs = {}
    for name in ("cs_q", "cs_k", "cs_v"):
        src = data[name]
        dst = torch.empty(
            src.shape[0], src.shape[2], src.shape[1], dtype=src.dtype, device=src.device
        ).transpose(1, 2)
        dst.copy_(src)
        cs[name] = dst
    k_cache = data["k_cache"].clone()
    g_cache = data["g_cache"].clone()
    v_cache = data["v_cache"].clone()
    beta_cache = data["beta_cache"].clone() if beta_cache_override is None else beta_cache_override
    op_kwargs = dict(
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
        k_cache=k_cache,
        g_cache=g_cache,
        v_cache=v_cache,
        beta_cache=beta_cache,
        ssm_state_indices=data["ssm_state_indices"],
        cu_seqlens=data["cu_seqlens"],
        num_spec=data["M"],
        num_accepted_tokens=data["num_accepted_tokens"],
        lower_bound=data["lower_bound"],
        scale=data["scale"],
        zero_accepted_hint=zero_accepted_hint,
        packed_token_layout=packed_token_layout,
        onorm_g=data.get("onorm_g"),
        onorm_weight=data.get("onorm_weight"),
        onorm_eps=data.get("onorm_eps", 1e-5),
        fuse_output_norm=fuse_output_norm,
    )
    output_scale = None
    if quantize_output:
        op_kwargs.pop("fuse_output_norm")
        out, output_scale = torch.ops.trtllm.kda_mtp_decode_fp8(**op_kwargs)
    elif out is None:
        out = torch.ops.trtllm.kda_mtp_decode(**op_kwargs)
    else:
        from tensorrt_llm._torch.custom_ops.cute_dsl_kimi_k3_kda_mtp_ops import kda_mtp_decode_impl

        out = kda_mtp_decode_impl(out=out, **op_kwargs)
    return {
        "out": out,
        "output_scale": output_scale,
        # committed state back in K-first layout for CPU-golden comparison
        "recurrent_state": state.permute(0, 1, 3, 2).contiguous(),
        "state_v_first": state,
        "k_cache": k_cache,
        "g_cache": g_cache,
        "v_cache": v_cache,
        "beta_cache": beta_cache,
        "cs_q": cs["cs_q"],
        "cs_k": cs["cs_k"],
        "cs_v": cs["cs_v"],
    }


def _pack_new_token_projection_views(data):
    """Build production-like section views over compact fused GEMM outputs."""
    B, H, K, M, source_T = (
        data["B"],
        data["H"],
        data["K"],
        data["M"],
        data["T"],
    )
    accepted = data["num_accepted_tokens"].cpu()
    rows = torch.cat(
        [
            torch.arange(n * source_T + int(accepted[n]), n * source_T + int(accepted[n]) + M + 1)
            for n in range(B)
        ]
    ).cuda()
    num_rows = B * (M + 1)
    dim = H * K

    # q/k/v are adjacent sections of [q | k | v | onorm_g]. Their token
    # stride is therefore 4 * dim, not dim.
    qkvg = torch.empty(1, num_rows, 4 * dim, dtype=torch.bfloat16, device="cuda")
    qkvg[..., :dim].copy_(data["x_q"][0, rows].reshape(1, num_rows, dim))
    qkvg[..., dim : 2 * dim].copy_(data["x_k"][0, rows].reshape(1, num_rows, dim))
    qkvg[..., 2 * dim : 3 * dim].copy_(data["x_v"][0, rows].reshape(1, num_rows, dim))
    qkvg[..., 3 * dim : 4 * dim].copy_(data["g"][0, rows].reshape(1, num_rows, dim))

    # beta is the H-wide tail of the padded [f_a | beta] projection.
    bfa_width = ((K + H + 7) // 8) * 8
    bfa = torch.empty(1, num_rows, bfa_width, dtype=torch.bfloat16, device="cuda")
    bfa[..., K : K + H].copy_(data["beta"][0, rows].reshape(1, num_rows, H))

    packed = dict(data)
    packed.update(
        x_q=qkvg[..., :dim].view(1, num_rows, H, K),
        x_k=qkvg[..., dim : 2 * dim].view(1, num_rows, H, K),
        x_v=qkvg[..., 2 * dim : 3 * dim].view(1, num_rows, H, K),
        onorm_g=qkvg[..., 3 * dim : 4 * dim].view(1, num_rows, H, K),
        g=data["g"][:, rows].contiguous(),
        beta=bfa[..., K : K + H],
        cu_seqlens=None,
        T=M + 1,
        T_total=num_rows,
    )
    assert packed["x_q"].stride(1) == 4 * dim
    assert packed["x_k"].stride(1) == 4 * dim
    assert packed["x_v"].stride(1) == 4 * dim
    assert packed["onorm_g"].stride(1) == 4 * dim
    assert packed["beta"].stride(1) == bfa_width
    return packed, rows


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
    M, T = data["M"], data["T"]
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
                g_t = data["g_cache"][n, i_t].float()
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
                k_t = data["k_cache"][n, i_t].float().view(H, K)
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
    for name in ("k_cache", "g_cache", "v_cache", "beta_cache", "cs_q", "cs_k", "cs_v"):
        _assert_close(f"{name}(cute vs cpu)", cute_out[name], cpu[name], atol=2e-2, rtol=4e-3)

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


@pytest.mark.parametrize(
    "B,H,M",
    [(128, 2, 2), (32, 12, 2), (32, 6, 2), (2, 12, 7), (16, 12, 7)],
    ids=lambda v: str(v),
)
def test_round2_replay(B, H, M):
    """Replay round: mixed num_accepted per request, chained from a CPU-
    golden round 1. Validates the kernel's cache-replay state math (the
    path the drop's own self-check never exercised). ``M=7`` covers the
    production MTP7 verify (T_loop up to 15) at the TEP8 shape."""
    data = make_conv_data(B, H, M=M, seed=7)
    T = data["T"]

    # Round 1 (all zero accepted) on the CPU golden to produce the caches
    # and committed state that seed round 2.
    r1 = cpu_reference(data)

    # Round 2 inputs: fresh tokens, caches/state/conv from round 1.
    data2 = make_conv_data(B, H, M=M, seed=8)
    for name in ("k_cache", "g_cache", "v_cache", "beta_cache"):
        data2[name] = r1[name].cuda().to(data2[name].dtype).contiguous()
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


def _round2_inputs(B, H, M):
    """Round-2 inputs chained from a CPU-golden round 1 (caches, conv windows, committed state)
    with mixed acceptance 0..M cycling across requests. Returns (data2, accept)."""
    data = make_conv_data(B, H, M=M, seed=7)
    r1 = cpu_reference(data)
    data2 = make_conv_data(B, H, M=M, seed=8)
    for name in ("k_cache", "g_cache", "v_cache", "beta_cache"):
        data2[name] = r1[name].cuda().to(data2[name].dtype).contiguous()
    for name in ("cs_q", "cs_k", "cs_v"):
        src = r1[name].cuda()
        dst = torch.empty(
            src.shape[0], src.shape[2], src.shape[1], dtype=src.dtype, device=src.device
        ).transpose(1, 2)
        dst.copy_(src)
        data2[name] = dst
    data2["initial_state_kfirst"] = r1["recurrent_state"].cuda().contiguous()
    data2["initial_state_cute"] = r1["recurrent_state"].cuda().permute(0, 1, 3, 2).contiguous()
    accept = torch.arange(B, dtype=torch.int32) % (M + 1)
    data2["num_accepted_tokens"] = accept.cuda()
    return data2, accept


@pytest.mark.parametrize("B,H,M", [(2, 12, 7), (16, 12, 7)], ids=lambda v: str(v))
def test_round2_replay_bf16_math(B, H, M, monkeypatch):
    """bf16 tensor-core operand path (``TRTLLM_KDA_MTP_BF16_MATH=1``) on the MTP7 replay round.
    The conv pools and replay caches do not depend on the recurrence precision and must match
    the 3xTF32 kernel bit for bit; the committed state and the outputs are held to the tolerance
    of the bf16 tensor-core references (FLA, prefill)."""
    data2, accept = _round2_inputs(B, H, M)
    T = data2["T"]
    monkeypatch.setenv("TRTLLM_KDA_MTP_BF16_MATH", "0")
    ref = cute_run(data2)
    monkeypatch.setenv("TRTLLM_KDA_MTP_BF16_MATH", "1")
    fast = cute_run(data2)
    cpu2 = {k: v.cuda() for k, v in cpu_reference(data2).items()}
    rows = torch.cat(
        [torch.arange(n * T + int(accept[n]), n * T + int(accept[n]) + 1 + M) for n in range(B)]
    ).cuda()

    for name in ("cs_q", "cs_k", "cs_v"):
        _assert_close(f"{name}(fast vs 3xTF32)", fast[name], ref[name], atol=0.0)
    for name in ("k_cache", "g_cache", "v_cache", "beta_cache"):
        _assert_close(f"{name}(fast vs 3xTF32)", fast[name], ref[name], atol=0.0)
    state_err = (fast["recurrent_state"] - cpu2["recurrent_state"]).abs().max().item()
    ref_err = (ref["recurrent_state"] - cpu2["recurrent_state"]).abs().max().item()
    out_err = (fast["out"][0, rows].float() - cpu2["out"][0, rows].float()).abs().max().item()
    print(
        f"[bf16_math B={B} H={H} M={M}] state max err vs fp32 golden "
        f"{state_err:.3e} (3xTF32 kernel {ref_err:.3e}), out max err {out_err:.3e}"
    )
    # measured max |err| vs the fp32 golden at (16,12,7): bf16 1.6e-2 (3xTF32: 4e-5)
    tol = 3e-2
    _assert_close(
        "state(fast vs cpu)", fast["recurrent_state"], cpu2["recurrent_state"], atol=tol, rtol=tol
    )
    _assert_close(
        "out(fast vs cpu)", fast["out"][0, rows], cpu2["out"][0, rows], atol=5e-2, rtol=5e-2
    )


@pytest.mark.parametrize("B,H,M", [(2, 12, 7), (16, 12, 7)], ids=lambda v: str(v))
def test_round2_replay_strided_state_cache(B, H, M):
    """Replay round with the recurrent state living in a strided slice of a larger buffer
    (slot stride 3 * H * V * K), like the runtime's per-layer view of the SSM cache. Covers the
    kernel's own global-memory views (TMA state load/store), which only a non-contiguous state
    exercises; ``B=2`` runs the cluster-pair form, ``B=16`` the single-CTA form."""
    data = make_conv_data(B, H, M=M, seed=7)
    T = data["T"]
    r1 = cpu_reference(data)
    data2 = make_conv_data(B, H, M=M, seed=8)
    for name in ("k_cache", "g_cache", "v_cache", "beta_cache"):
        data2[name] = r1[name].cuda().to(data2[name].dtype).contiguous()
    for name in ("cs_q", "cs_k", "cs_v"):
        src = r1[name].cuda()
        dst = torch.empty(
            src.shape[0], src.shape[2], src.shape[1], dtype=src.dtype, device=src.device
        ).transpose(1, 2)
        dst.copy_(src)
        data2[name] = dst
    data2["initial_state_kfirst"] = r1["recurrent_state"].cuda().contiguous()
    data2["initial_state_cute"] = r1["recurrent_state"].cuda().permute(0, 1, 3, 2).contiguous()
    accept = torch.arange(B, dtype=torch.int32) % (M + 1)
    data2["num_accepted_tokens"] = accept.cuda()

    state_shape = tuple(data2["initial_state_cute"].shape)  # (B, H, V, K)
    big = torch.zeros((B, 3) + state_shape[1:], dtype=torch.float32, device="cuda")
    strided_state = big[:, 1]
    assert not strided_state.is_contiguous()

    cpu2 = {k: v.cuda() for k, v in cpu_reference(data2).items()}
    cute2 = cute_run(data2, state_override=strided_state)

    rows = torch.cat(
        [torch.arange(n * T + int(accept[n]), n * T + int(accept[n]) + 1 + M) for n in range(B)]
    ).cuda()
    _assert_close("out2(cute vs cpu)", cute2["out"][0, rows], cpu2["out"][0, rows], atol=2e-2)
    _assert_close(
        "state2(cute vs cpu)", cute2["recurrent_state"], cpu2["recurrent_state"], atol=1e-4
    )
    # the neighbouring slices of the buffer must be untouched
    assert torch.count_nonzero(big[:, 0]) == 0 and torch.count_nonzero(big[:, 2]) == 0


@pytest.mark.parametrize("B,H,M", [(2, 12, 7), (16, 12, 7)], ids=lambda v: str(v))
def test_round2_replay_bf16_state(B, H, M):
    """Replay round with a bf16 recurrent-state pool: the kernel reads the bf16 state (exact in
    tf32) and rounds the committed state to nearest bf16. The CPU golden starts from the same
    bf16-rounded state; its fp32 result is compared against the kernel's bf16 output with a
    half-ulp (2^-9) bf16 tolerance. ``B=2`` runs the cluster-pair form, ``B=16`` the single-CTA form."""
    data = make_conv_data(B, H, M=M, seed=7)
    T = data["T"]
    r1 = cpu_reference(data)
    data2 = make_conv_data(B, H, M=M, seed=8)
    for name in ("k_cache", "g_cache", "v_cache", "beta_cache"):
        data2[name] = r1[name].cuda().to(data2[name].dtype).contiguous()
    for name in ("cs_q", "cs_k", "cs_v"):
        src = r1[name].cuda()
        dst = torch.empty(
            src.shape[0], src.shape[2], src.shape[1], dtype=src.dtype, device=src.device
        ).transpose(1, 2)
        dst.copy_(src)
        data2[name] = dst
    state_bf16 = r1["recurrent_state"].cuda().permute(0, 1, 3, 2).to(torch.bfloat16).contiguous()
    data2["initial_state_cute"] = state_bf16
    data2["initial_state_kfirst"] = state_bf16.permute(0, 1, 3, 2).float().contiguous()
    accept = torch.arange(B, dtype=torch.int32) % (M + 1)
    data2["num_accepted_tokens"] = accept.cuda()

    cpu2 = {k: v.cuda() for k, v in cpu_reference(data2).items()}
    cute2 = cute_run(data2)
    assert cute2["state_v_first"].dtype == torch.bfloat16

    rows = torch.cat(
        [torch.arange(n * T + int(accept[n]), n * T + int(accept[n]) + 1 + M) for n in range(B)]
    ).cuda()
    _assert_close("out2(cute vs cpu)", cute2["out"][0, rows], cpu2["out"][0, rows], atol=2e-2)
    _assert_close(
        "state2(cute bf16 vs cpu fp32)",
        cute2["recurrent_state"].float(),
        cpu2["recurrent_state"],
        atol=2e-3,
        rtol=4e-3,
    )


@pytest.mark.parametrize("B,H,num_spec", [(4, 6, 2), (2, 12, 7)])
def test_packed_projection_views_match_shifted_metadata_replay(B, H, num_spec):
    """Compact row-strided projection views match the legacy shifted layout."""
    data = make_conv_data(B=B, H=H, M=num_spec, seed=10)
    request_accepted = torch.arange(B, dtype=torch.int32, device="cuda")
    request_accepted[-1] = num_spec
    data["num_accepted_tokens"] = request_accepted
    packed, legacy_rows = _pack_new_token_projection_views(data)

    # The production count is pool-level, while request rows may map to any
    # state slot. Move every request's initial pools to a non-identity slot
    # and scatter its accepted count to the same slot.
    slots = torch.arange(B - 1, -1, -1, dtype=torch.int32, device="cuda")
    packed["ssm_state_indices"] = slots
    for name in (
        "initial_state_kfirst",
        "initial_state_cute",
        "k_cache",
        "g_cache",
        "v_cache",
        "beta_cache",
        "cs_q",
        "cs_k",
        "cs_v",
    ):
        request_order = packed[name]
        slot_order = torch.empty_like(request_order)
        slot_order[slots.long()] = request_order
        packed[name] = slot_order
    pool_accepted = torch.zeros(B, dtype=torch.int32, device="cuda")
    pool_accepted[slots.long()] = request_accepted
    packed["num_accepted_tokens"] = pool_accepted

    legacy = cute_run(data)
    actual = cute_run(packed, packed_token_layout=True)
    poison = torch.full(
        (1, B * (num_spec + 1), H, data["V"]),
        float("nan"),
        dtype=torch.bfloat16,
        device="cuda",
    )
    poisoned = cute_run(packed, packed_token_layout=True, out=poison)
    assert poisoned["out"].data_ptr() == poison.data_ptr()
    assert torch.isfinite(poisoned["out"]).all()
    _assert_close("poisoned packed out", poisoned["out"], actual["out"], atol=1e-5)
    _assert_close(
        "packed out",
        actual["out"].reshape(-1, data["H"], data["V"]),
        legacy["out"][0, legacy_rows],
        atol=1e-5,
    )

    # Stage3a oracle: fuse only gated RMSNorm into the packed CuTe consumer.
    # The raw recurrence output above is BF16, matching the explicit BF16
    # round-trip boundary retained in the fused epilogue.
    from tensorrt_llm._torch.modules.mamba.layernorm_gated import rms_norm_gated_token_major

    fused_data = dict(packed)
    fused_data["onorm_weight"] = torch.linspace(
        0.75, 1.25, data["V"], dtype=torch.float32, device="cuda"
    )
    fused_data["onorm_eps"] = 1e-5
    expected_norm = rms_norm_gated_token_major(
        actual["out"].reshape(-1, data["V"]),
        packed["onorm_g"].reshape(-1, data["H"], data["V"]),
        fused_data["onorm_weight"],
        fused_data["onorm_eps"],
        gate_activation="sigmoid",
    )
    fused_norm = cute_run(
        fused_data,
        packed_token_layout=True,
        fuse_output_norm=True,
    )
    assert torch.isfinite(fused_norm["out"]).all()
    fused_norm_diff = (
        fused_norm["out"].float().reshape_as(expected_norm) - expected_norm.float()
    ).abs()
    fused_norm_rel_l2 = (
        fused_norm_diff.norm() / expected_norm.float().norm().clamp_min(1e-12)
    ).item()
    print(
        f"  fused_norm B={B} H={H} M={num_spec}: "
        f"max_abs={fused_norm_diff.max().item():.3e} rel_l2={fused_norm_rel_l2:.3e}"
    )
    _assert_close(
        "fused gated RMSNorm",
        fused_norm["out"].reshape(-1, data["V"]),
        expected_norm,
        atol=2e-2,
        rtol=1e-2,
    )
    fused_fp8 = cute_run(
        fused_data,
        packed_token_layout=True,
        quantize_output=True,
    )
    expected_rows = B * (num_spec + 1)
    assert fused_fp8["out"].dtype is torch.float8_e4m3fn
    assert fused_fp8["out"].shape == (expected_rows, H * data["V"])
    assert torch.isfinite(fused_fp8["out"].float()).all()
    assert fused_fp8["output_scale"].dtype is torch.uint8
    assert fused_fp8["output_scale"].shape == ((expected_rows + 127) // 128 * 128 * H * 4,)
    expected_fp8, expected_scale = torch.ops.trtllm.fp8_quantize_1x128_packed_ue8m0(
        expected_norm.reshape(expected_rows, H * data["V"]),
    )
    assert torch.equal(fused_fp8["out"], expected_fp8)
    rows = torch.arange(expected_rows, device="cuda", dtype=torch.int64).view(-1, 1)
    scale_columns = torch.arange(H * 4, device="cuda", dtype=torch.int64).view(1, -1)
    offsets = (
        ((rows // 128 * H + scale_columns // 4) * 32 + rows % 32) * 4 + (rows % 128) // 32
    ) * 4 + scale_columns % 4
    assert torch.equal(
        fused_fp8["output_scale"][offsets.flatten()],
        expected_scale[offsets.flatten()],
    )
    for name in (
        "recurrent_state",
        "k_cache",
        "g_cache",
        "v_cache",
        "beta_cache",
        "cs_q",
        "cs_k",
        "cs_v",
    ):
        _assert_close(
            f"{name}(packed vs shifted)", actual[name][slots.long()], legacy[name], atol=1e-5
        )
        _assert_close(f"{name}(norm vs raw)", fused_norm[name], actual[name], atol=1e-5)
        _assert_close(f"{name}(fp8 vs raw)", fused_fp8[name], actual[name], atol=1e-5)


@pytest.mark.parametrize("B,H", [(32, 12)], ids=lambda v: str(v))
def test_zero_accepted_hint_variant(B, H):
    """The zero_accepted_hint fast variant matches the general variant."""
    data = make_conv_data(B, H, M=M, seed=11)
    general = cute_run(data, zero_accepted_hint=False)
    fast = cute_run(data, zero_accepted_hint=True)
    for name in (
        "out",
        "recurrent_state",
        "k_cache",
        "g_cache",
        "v_cache",
        "beta_cache",
        "cs_q",
        "cs_k",
        "cs_v",
    ):
        _assert_close(f"{name}(fast vs general)", fast[name], general[name], atol=1e-5)


def test_misaligned_state_indices_after_aligned_warmup():
    """Mixed context/generation metadata remains valid after op warmup."""
    data = make_conv_data(B=1, H=6, M=M, seed=13)
    expected = cute_run(data)

    index_storage = torch.empty(2, dtype=torch.int32, device="cuda")
    index_storage[0] = -1
    index_storage[1:].copy_(data["ssm_state_indices"])
    misaligned_indices = index_storage[1:]
    assert misaligned_indices.is_contiguous()
    assert misaligned_indices.data_ptr() % 16 != 0

    misaligned_data = dict(data)
    misaligned_data["ssm_state_indices"] = misaligned_indices
    actual = cute_run(misaligned_data)

    for name in (
        "out",
        "recurrent_state",
        "k_cache",
        "g_cache",
        "v_cache",
        "beta_cache",
        "cs_q",
        "cs_k",
        "cs_v",
    ):
        _assert_close(f"{name}(misaligned vs aligned)", actual[name], expected[name], atol=1e-5)


@pytest.mark.parametrize("field", ("cu_seqlens", "num_accepted_tokens"))
def test_misaligned_scalar_metadata_after_aligned_warmup(field):
    """Every scalar CuTe metadata argument accepts an offset int32 view."""
    data = make_conv_data(B=1, H=6, M=M, seed=17)
    expected = cute_run(data)

    source = data[field]
    storage = torch.empty(source.numel() + 1, dtype=torch.int32, device="cuda")
    storage[0] = -1
    storage[1:].copy_(source)
    misaligned = storage[1:]
    assert misaligned.is_contiguous()
    assert misaligned.data_ptr() % 16 != 0

    misaligned_data = dict(data)
    misaligned_data[field] = misaligned
    actual = cute_run(misaligned_data)

    for name in (
        "out",
        "recurrent_state",
        "k_cache",
        "g_cache",
        "v_cache",
        "beta_cache",
        "cs_q",
        "cs_k",
        "cs_v",
    ):
        _assert_close(
            f"{name}({field} misaligned vs aligned)", actual[name], expected[name], atol=1e-5
        )


@pytest.mark.parametrize(
    "attention_mode,parallel_size,dtype,pool_size,num_spec,expected",
    (
        pytest.param("dep", 8, torch.float32, 1177, 5, 16, id="dep8-mtp5-fp32"),
        pytest.param("dep", 16, torch.float32, 1177, 7, 16, id="dep16-mtp7-fp32"),
        pytest.param("tep", 8, torch.float32, 1177, 7, 16, id="tep8-mtp7-fp32"),
        pytest.param("tep", 16, torch.float32, 1177, 2, 16, id="tep16-mtp2-fp32"),
        pytest.param("tep", 16, torch.float32, 1177, 5, 8, id="tep16-mtp5-fp32"),
        pytest.param("tep", 16, torch.float32, 1177, 7, 8, id="tep16-mtp7-fp32"),
        pytest.param("tep", 16, torch.float32, 248, 7, 16, id="tep16-mtp7-even-pool"),
        pytest.param("tep", 32, torch.float32, 1177, 7, 4, id="tep32-mtp7-fp32"),
        pytest.param("dep", 16, torch.bfloat16, 1177, 7, 16, id="dep16-mtp7-bf16"),
        pytest.param("tep", 8, torch.bfloat16, 1177, 7, 8, id="tep8-mtp7-bf16"),
        pytest.param("tep", 16, torch.bfloat16, 1177, 7, 4, id="tep16-mtp7-bf16"),
        pytest.param("tep", 32, torch.bfloat16, 1177, 7, 2, id="tep32-mtp7-bf16"),
    ),
)
def test_beta_cache_alignment_is_derived_from_layout_and_dtype(
    attention_mode, parallel_size, dtype, pool_size, num_spec, expected
):
    from tensorrt_llm._torch.custom_ops.cute_dsl_kimi_k3_kda_mtp_ops import (
        _beta_cache_assumed_align,
    )

    local_heads = 96 if attention_mode == "dep" else 96 // parallel_size
    parent = torch.empty(2, pool_size, num_spec, local_heads, dtype=dtype, device="cuda")
    beta_cache = parent[0]

    assert _beta_cache_assumed_align(beta_cache) == expected


def test_beta_cache_alignment_uses_padded_physical_stride():
    from tensorrt_llm._torch.custom_ops.cute_dsl_kimi_k3_kda_mtp_ops import (
        _beta_cache_assumed_align,
    )

    parent = torch.empty(2, 1177, 7, 8, dtype=torch.float32, device="cuda")
    beta_cache = parent[0, ..., :6]

    assert beta_cache.shape == (1177, 7, 6)
    assert beta_cache.stride() == (56, 8, 1)
    assert _beta_cache_assumed_align(beta_cache) == 16


@pytest.mark.parametrize(
    "attention_mode,parallel_size,num_spec,expected_alignment",
    (
        pytest.param("dep", 16, 7, 16, id="dep16-mtp7"),
        pytest.param("tep", 8, 7, 16, id="tep8-mtp7"),
        pytest.param("tep", 16, 5, 8, id="tep16-mtp5"),
        pytest.param("tep", 16, 7, 8, id="tep16-mtp7"),
        pytest.param("tep", 32, 7, 4, id="tep32-mtp7"),
    ),
)
def test_beta_cache_sibling_layer_after_aligned_warmup(
    attention_mode, parallel_size, num_spec, expected_alignment
):
    """A cached kernel accepts the next real per-layer beta-cache slice."""
    from tensorrt_llm._torch.custom_ops.cute_dsl_kimi_k3_kda_mtp_ops import (
        _beta_cache_assumed_align,
    )

    local_heads = 96 if attention_mode == "dep" else 96 // parallel_size
    data = make_conv_data(B=1, H=local_heads, M=num_spec, seed=29)

    parent = torch.zeros(2, 1177, num_spec, local_heads, dtype=torch.float32, device="cuda")
    layer0, layer1 = parent.unbind(0)
    layer0[0].copy_(data["beta_cache"][0])
    layer1[0].copy_(data["beta_cache"][0])

    assert _beta_cache_assumed_align(layer0) == expected_alignment
    assert layer0.data_ptr() % 16 == 0
    assert layer1.data_ptr() % expected_alignment == 0
    if expected_alignment < 16:
        assert layer1.data_ptr() % (2 * expected_alignment) != 0

    expected = cute_run(data, beta_cache_override=layer0)
    actual = cute_run(data, beta_cache_override=layer1)

    for name in (
        "out",
        "recurrent_state",
        "k_cache",
        "g_cache",
        "v_cache",
        "beta_cache",
        "cs_q",
        "cs_k",
        "cs_v",
    ):
        _assert_close(
            f"{name}({attention_mode}{parallel_size}-mtp{num_spec})",
            actual[name],
            expected[name],
            atol=1e-5,
        )


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v", "-x"]))
