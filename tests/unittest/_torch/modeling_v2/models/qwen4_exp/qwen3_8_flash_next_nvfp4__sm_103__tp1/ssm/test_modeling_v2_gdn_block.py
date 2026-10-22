# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test: the target's whole Gated-DeltaNet block against the all-Triton path.

Drives `DecodeTarget._gdn_block` and `PrefillTarget._gdn_block` at the
checkpoint's geometry (hidden 2560, 16 k-heads, 48 v-heads, head dim 128,
fp32 or bf16 recurrent pool of 461 slots) on a decode and a mixed (context + one-token
generation rows) batch, and compares the block output
and both state pools with a reference that composes the Triton recurrences
(`fused_sigmoid_gating_delta_rule_update`'s rows form for decode, bitwise its
packed call at either pool dtype; `chunk_gated_delta_rule` over every row of a
prefill step) around the same projections, convolutions and epilogue.

The reference's context rows run the unfused `extract_transpose_prefill_slice`
-> `causal_conv1d_fwd` -> `fused_gdn_post_conv` chain the target's
`gdn_prefill_conv_post` replaces.

Bands: the conv pools are bitwise (both sides publish the raw projection). The
recurrent pool is held to the looser of the two recurrence entries' state
bands (`flashinfer_chunk_gated_delta_rule`: rtol 1.6e-2, atol 2e-2), and the
block output, after the gated norm and the output projection, to a relative
L2 distance of 1e-2 and a max-abs distance of 3e-2 of its own peak.
"""

import types

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_103__tp1 import (  # noqa: E501
    modeling as M,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_103__tp1.ops.ssm.causal_conv1d_fwd import (  # noqa: E501
    causal_conv1d_fwd,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_103__tp1.ops.ssm.chunk_gated_delta_rule import (  # noqa: E501
    chunk_gated_delta_rule as triton_chunk,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_103__tp1.ops.ssm.extract_transpose_prefill_slice import (  # noqa: E501
    extract_transpose_prefill_slice,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_103__tp1.ops.ssm.fused_gdn_post_conv import (  # noqa: E501
    fused_gdn_post_conv,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_103__tp1.ops.ssm.fused_sigmoid_gating_delta_rule_update import (  # noqa: E501
    fused_sigmoid_gating_delta_rule_update as triton_decode,
)

assert torch.cuda.is_available(), "the Gated-DeltaNet block requires a CUDA device"

DEV = "cuda"
BF16 = torch.bfloat16
HIDDEN, H, HV, K, V, WIDTH = 2560, 16, 48, 128, 128, 4
KD, VD = H * K, HV * V
CONV = 2 * KD + VD
SLOTS = 461
MAX_BATCH = 256
RTOL_STATE, ATOL_STATE = 1.6e-2, 2e-2
OUT_REL_L2, OUT_MAX_FRAC = 1e-2, 3e-2


def _slots(n: int) -> list[int]:
    """`n` distinct slots of the pool, both ends first."""
    rest = [(i * 37 + 4) % (SLOTS - 2) + 1 for i in range(max(n - 2, 0))]
    return ([SLOTS - 1, 0] + rest)[:n]


def _core(seed: int, state: torch.dtype = torch.float32) -> types.SimpleNamespace:
    """The attributes `_gdn_block` reads off the core, with layer 0's pools (the recurrent
    one in `state`)."""
    g = torch.Generator(device=DEV).manual_seed(seed)
    conv = torch.randn(SLOTS, CONV, WIDTH - 1, generator=g, device=DEV, dtype=BF16)
    ssm = (torch.randn(SLOTS, HV, V, K, generator=g, device=DEV) * 0.05).to(state)
    return types.SimpleNamespace(
        conv_dim=CONV,
        key_dim=KD,
        value_dim=VD,
        num_k_heads=H,
        head_k_dim=K,
        num_v_heads=HV,
        head_v_dim=V,
        eps=1e-6,
        conv_kernel=WIDTH,
        in_proj_split=(CONV, VD, HV, HV),
        _bound=types.SimpleNamespace(conv={0: conv}, ssm={0: ssm}),
    )


def _clone_core(core: types.SimpleNamespace) -> types.SimpleNamespace:
    out = types.SimpleNamespace(**vars(core))
    out._bound = types.SimpleNamespace(
        conv={0: core._bound.conv[0].clone()}, ssm={0: core._bound.ssm[0].clone()}
    )
    return out


def _layer(seed: int) -> types.SimpleNamespace:
    """Layer weights as the target views them: HF `[out, in]` storage read
    through `.t()`, fp32 gating vectors over the checkpoint's ranges."""
    g = torch.Generator(device=DEV).manual_seed(seed)

    def lin(n_out, n_in):
        return (torch.randn(n_out, n_in, generator=g, device=DEV) * n_in**-0.5).to(BF16).t()

    # The target's one [q|k|v | z | a | b] input projection, read through `.t()`.
    # The reference blocks run the same GEMM, so the conv pools stay bitwise.
    return types.SimpleNamespace(
        in_proj=lin(CONV + VD + 2 * HV, HIDDEN),
        out=lin(HIDDEN, VD),
        conv_w=(torch.randn(CONV, WIDTH, generator=g, device=DEV) * 0.5).to(BF16),
        a_log=torch.empty(HV, device=DEV).uniform_(-5.5, 3.5, generator=g),
        dt_bias=torch.empty(HV, device=DEV).uniform_(-8.9, 19.2, generator=g),
        gnorm=(1.0 + 0.1 * torch.randn(VD // HV, generator=g, device=DEV)).to(BF16),
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


def _prefill_block(core, x, L, st):
    """`PrefillTarget._gdn_block` over the step `st`, the break's output buffer allocated
    as the forward allocates it."""
    tokens = x.shape[0]
    step = types.SimpleNamespace(gdn=st, num_tokens=tokens)
    out = torch.empty([1, tokens, HV, V], dtype=BF16, device=DEV)
    return _target(M.PrefillTarget, core, L, step)._gdn_block(x, 0, L, out)


def _decode_state(m: int):
    """`DecodeTarget.forward`'s `gdn_state` for `m` generation sequences: the
    slot vector is the head of an engine-sized buffer."""
    buf = torch.zeros(MAX_BATCH, dtype=torch.int32, device=DEV)
    buf[:m] = torch.tensor(_slots(m), dtype=torch.int32, device=DEV)
    return (buf[:m], False, m, 1, None, None)


def _prefill_state(ctx_lens: tuple, n_dec: int, has_init: tuple):
    """`PrefillTarget.forward`'s `gdn_state` for a context prefix plus `n_dec`
    one-token generation rows, built the way the forward builds it from the
    engine's `Mamba2Metadata` buffers."""
    n_ctx = len(ctx_lens)
    batch = n_ctx + n_dec
    n_p = sum(ctx_lens)
    si_buf = torch.zeros(MAX_BATCH, dtype=torch.int32, device=DEV)
    si_buf[:batch] = torch.tensor(_slots(batch), dtype=torch.int32, device=DEV)
    hi_buf = torch.zeros(MAX_BATCH, dtype=torch.bool, device=DEV)
    hi_buf[:n_ctx] = torch.tensor(has_init, dtype=torch.bool, device=DEV)
    si_all, hi_all = si_buf[:batch], hi_buf[:batch]
    si_p, si_d = torch.split(si_all, [n_ctx, n_dec], 0)
    si_d = torch.clone(si_d)
    hi_p = hi_all[:n_ctx]
    bounds = [0]
    for n in ctx_lens:
        bounds.append(bounds[-1] + n)
    bounds += [n_p + j for j in range(1, n_dec + 1)]
    cu_buf = torch.zeros(MAX_BATCH + 1, dtype=torch.int32, device=DEV)
    cu_buf[: batch + 1] = torch.tensor(bounds, dtype=torch.int32, device=DEV)
    cu_long_buf = cu_buf.to(torch.int64)
    qsl_all = cu_buf[: batch + 1]
    qsl_p = qsl_all[: n_ctx + 1]
    qsl_long = cu_long_buf[: batch + 1]
    qsl_long_p = qsl_long[: n_ctx + 1]
    return (
        n_p,
        n_dec,
        si_p,
        si_all,
        si_d,
        hi_p,
        hi_all,
        qsl_p,
        qsl_all,
        qsl_long,
        False,
        n_dec,
        1,
        None,
        qsl_long_p,
        None,
    )


def _post_conv(prefill, decode, a, b, L):
    return fused_gdn_post_conv(
        prefill=prefill,
        decode=decode,
        a=a,
        b=b,
        A_log=L.a_log,
        dt_bias=L.dt_bias,
        num_k_heads=H,
        head_k_dim=K,
        num_v_heads=HV,
        head_v_dim=V,
        l2_norm_eps=M._L2_NORM_EPS,
        softplus_threshold=M._SOFTPLUS_THRESHOLD,
    )


def _triton_decode_block(core, x, L, si_all):
    """The decode block with the Triton single-token recurrence in its rows form (q/k/v
    `[m, 1, heads, 128]` from a contiguous copy, `cu_seqlens` None), which stays on the
    Triton kernel at either pool dtype."""
    m = x.shape[0]
    mixed, z, a, b = torch.split(M.cublas_mm(x, L.in_proj), [CONV, VD, HV, HV], 1)
    M.causal_conv1d_update(
        x=mixed,
        conv_state=core._bound.conv[0],
        weight=L.conv_w,
        bias=None,
        silu_activation=True,
        cache_seqlens=None,
        conv_state_indices=si_all,
        pad_slot_id=M._PAD_SLOT_ID,
    )
    qv, kv, vv = torch.split(torch.reshape(mixed.contiguous(), [m, 1, -1, K]), [H, H, HV], 2)
    out = triton_decode(
        A_log=L.a_log,
        a=torch.reshape(a.contiguous(), [m, 1, HV]),
        dt_bias=L.dt_bias,
        softplus_beta=M._SOFTPLUS_BETA,
        softplus_threshold=M._SOFTPLUS_THRESHOLD,
        q=qv,
        k=kv,
        v=vv,
        b=torch.reshape(b.contiguous(), [m, 1, HV]),
        initial_state_source=core._bound.ssm[0],
        initial_state_indices=si_all,
        scale=None,
        use_qk_l2norm_in_kernel=True,
        cu_seqlens=None,
        output=None,
    )
    return M.DecodeTarget._gdn_epilogue(_target(M.DecodeTarget, core), out, z, L, m)


def _triton_prefill_block(core, x, L, st):
    """The prefill block with every row -- context and generation -- in one
    Triton chunk call."""
    n_p, n_d, si_p, si_all, si_d, hi_p, hi_all, qsl_p, qsl_all, qsl_long = st[:10]
    T = x.shape[0]
    conv, ssm = core._bound.conv[0], core._bound.ssm[0]
    mixed, z, a, b = torch.split(M.cublas_mm(x, L.in_proj), [CONV, VD, HV, HV], 1)
    M.reset_recurrent_state_rows(ssm, si_p, hi_p, conv)
    conv_args = dict(
        weight=L.conv_w,
        bias=None,
        conv_states=conv,
        silu_activation=True,
        pad_slot_id=M._PAD_SLOT_ID,
        out=None,
    )
    if n_d > 0:
        mq_p, mq_d = torch.split(mixed, [n_p, n_d], 0)
        mq_p_t = extract_transpose_prefill_slice(mq_p, n_p, 0, CONV)
        causal_conv1d_fwd(
            x=mq_p_t, query_start_loc=qsl_p, cache_indices=si_p, has_initial_state=hi_p, **conv_args
        )
        M.causal_conv1d_update(
            x=mq_d,
            conv_state=conv,
            weight=L.conv_w,
            bias=None,
            silu_activation=True,
            cache_seqlens=None,
            conv_state_indices=si_d,
            pad_slot_id=M._PAD_SLOT_ID,
        )
        q, k, v, g, beta = _post_conv(mq_p_t, mq_d, a, b, L)
    else:
        mq_t = extract_transpose_prefill_slice(mixed, T, 0, CONV)
        causal_conv1d_fwd(
            x=mq_t,
            query_start_loc=qsl_all,
            cache_indices=si_all,
            has_initial_state=hi_all,
            **conv_args,
        )
        q, k, v, g, beta = _post_conv(mq_t, None, a, b, L)
    out, _ = triton_chunk(
        q,
        k,
        v,
        g,
        beta,
        scale=None,
        initial_state=ssm,
        initial_state_indices=si_all,
        inplace_indexed_state_update=True,
        output_final_state=False,
        cu_seqlens=qsl_long,
        use_qk_l2norm_in_kernel=False,
        output=None,
    )
    return M.PrefillTarget._gdn_epilogue(_target(M.PrefillTarget, core), out, z, L, T)


def _check(y, y_ref, core, core_ref, what):
    """Output and both pools against the reference block's, at the module bands."""
    assert y.shape == y_ref.shape and y.dtype == y_ref.dtype
    assert torch.isfinite(y).all()
    d = y.float() - y_ref.float()
    rel = (d.norm() / y_ref.float().norm()).item()
    peak = (d.abs().max() / y_ref.float().abs().max()).item()
    s_d = core._bound.ssm[0] - core_ref._bound.ssm[0]
    s_rel = (s_d.norm() / core_ref._bound.ssm[0].norm()).item()
    print(
        f"{what}: out rel-L2 {rel:.3e}, out max-abs/peak {peak:.3e}, "
        f"state rel-L2 {s_rel:.3e}, state max-abs {s_d.abs().max().item():.3e}"
    )
    assert rel <= OUT_REL_L2 and peak <= OUT_MAX_FRAC, (rel, peak)
    assert torch.equal(core._bound.conv[0], core_ref._bound.conv[0]), "conv pools differ"
    torch.testing.assert_close(
        core._bound.ssm[0], core_ref._bound.ssm[0], rtol=RTOL_STATE, atol=ATOL_STATE
    )


@pytest.fixture(scope="module")
def warm_triton_chunk():
    """The Triton chunk op's first indexed in-place call per autotune key
    returns a wrong answer (its entry's note); spend it on scratch."""
    core = _core(0)
    L = _layer(0)
    st = _prefill_state((64,), 0, (False,))
    _triton_prefill_block(core, _x(64, 0), L, st)
    torch.cuda.synchronize()


#: The recurrent pool dtypes the target serves (`mamba_ssm_cache_dtype`).
STATES = {"fp32": torch.float32, "bf16": torch.bfloat16}


@pytest.mark.parametrize("state", STATES)
def test_decode_block(state) -> None:
    """A con-128 decode step; the graph-replay test below runs 32 rows."""
    m = 128
    core, L, x = _core(100 + m, STATES[state]), _layer(200 + m), _x(m, 300 + m)
    core_ref = _clone_core(core)
    st = _decode_state(m)
    y = _target(M.DecodeTarget, core)._gdn_block(x, 0, L, st)
    y_ref = _triton_decode_block(core_ref, x, L, st[0])
    _check(y, y_ref, core, core_ref, f"decode m={m}, {state} pool")
    named = set(_slots(m))
    unnamed = [s for s in range(SLOTS) if s not in named]
    original = _core(100 + m, STATES[state])
    assert torch.equal(core._bound.ssm[0][unnamed], original._bound.ssm[0][unnamed]), (
        "unnamed slots written"
    )


@pytest.mark.parametrize("state", STATES)
def test_prefill_block(state, warm_triton_chunk) -> None:  # noqa: F811 — fixture by name
    """A mixed step: three variable-length contexts (two carrying state) plus five
    one-token generation rows."""
    ctx_lens, n_dec, has_init = (200, 37, 91), 5, (True, False, True)
    tokens = sum(ctx_lens) + n_dec
    core, L, x = _core(400 + tokens, STATES[state]), _layer(500 + tokens), _x(tokens, 600 + tokens)
    core_ref = _clone_core(core)
    st = _prefill_state(ctx_lens, n_dec, has_init)
    y = _prefill_block(core, x, L, st)
    y_ref = _triton_prefill_block(core_ref, x, L, st)
    _check(y, y_ref, core, core_ref, f"prefill {ctx_lens}+{n_dec}x1, {state} pool")


@pytest.mark.parametrize("state", STATES)
def test_decode_block_replays_under_cuda_graph(state) -> None:
    """The decode block captured into a CUDA graph after one eager call (the
    eager call compiles the decode kernel and sizes its per-batch scratch, as
    the engine's warmup does) replays to exactly the eager result."""
    m = 32
    core0, L, x = _core(700, STATES[state]), _layer(701), _x(m, 702)
    st = _decode_state(m)
    eager_core = _clone_core(core0)
    y_eager = _target(M.DecodeTarget, eager_core)._gdn_block(x, 0, L, st).clone()
    graph_core = _clone_core(core0)
    target = _target(M.DecodeTarget, graph_core)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        target._gdn_block(x, 0, L, st)
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph_core._bound.conv[0].copy_(core0._bound.conv[0])
    graph_core._bound.ssm[0].copy_(core0._bound.ssm[0])
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        y_graph = target._gdn_block(x, 0, L, st)
    assert torch.equal(graph_core._bound.ssm[0], core0._bound.ssm[0]), "capture must not execute"
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(y_graph, y_eager)
    assert torch.equal(graph_core._bound.ssm[0], eager_core._bound.ssm[0])
    assert torch.equal(graph_core._bound.conv[0], eager_core._bound.conv[0])
