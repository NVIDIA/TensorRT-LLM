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

"""CuTe DSL custom op for Kimi K3 KDA prefill.

The implementation source-integrates the production orchestration from the
KDA ``chunk_kda_fwd`` benchmark. The default pipeline launches fused K123,
Akk inverse, and persistent K4 kernels.

Supported:
  - Equal-length sequences (B ≥ 1, with the existing B=1 padding path)
  - Variable-length sequences (B=1 with cu_seqlens)
  - safe_gate mode (sigmoid + lower_bound) and softplus mode
  - dt_bias
  - use_gate_in_kernel=True with A_log
  - chunk_size=64

The ``trtllm::kda_prefill`` operator updates selected recurrent-state pool rows
in place. Intermediate matrices remain private runner workspace.
"""

import weakref
from typing import Optional

import torch

from ..cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE
from ..flashinfer_utils import IS_FLASHINFER_AVAILABLE

if IS_CUTLASS_DSL_AVAILABLE and IS_FLASHINFER_AVAILABLE:
    import cuda.bindings.driver as cuda_driver
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import from_dlpack

    from ..cute_dsl_kernels.blackwell.kimi_k3_kda.akk_inverse import akk_inv_host as _akk_inv_host
    from ..cute_dsl_kernels.blackwell.kimi_k3_kda.fused_k123 import (
        make_host_function as _fused_make_host,
    )
    from ..cute_dsl_kernels.blackwell.kimi_k3_kda.k4_persistent import (
        BYTES_PER_TENSORMAP as _K4P_BTM,
    )
    from ..cute_dsl_kernels.blackwell.kimi_k3_kda.k4_persistent import NUM_TENSORMAPS as _K4P_NTM
    from ..cute_dsl_kernels.blackwell.kimi_k3_kda.k4_persistent import (
        make_host_fn as _k4p_make_host,
    )
    from ..modules.fla.index import prepare_chunk_indices
else:
    raise ImportError("Kimi K3 KDA prefill requires NVIDIA CUTLASS DSL and FlashInfer")


def _ct(t, etype):
    """Create a CuTe tensor from PyTorch tensor."""
    r = from_dlpack(t, assumed_align=16)
    r.element_type = etype
    return r


def _fits_32bit_stride(tensor: torch.Tensor) -> bool:
    int32_max = 2**31 - 1
    max_offset = int(tensor.storage_offset())
    for size, stride in zip(tensor.shape, tensor.stride()):
        stride = abs(int(stride))
        if stride > int32_max:
            return False
        if size:
            max_offset += (int(size) - 1) * stride
            if max_offset > int32_max:
                return False
    return True


def _dynamic_dlpack_arg(tensor: torch.Tensor):
    wrapper = from_dlpack(
        tensor,
        assumed_align=16,
        use_32bit_stride=_fits_32bit_stride(tensor),
    )
    for dim, stride in enumerate(tensor.stride()):
        if stride == 1:
            return wrapper.mark_layout_dynamic(dim)
    return wrapper


def _layout_key(tensor: torch.Tensor):
    return (
        tensor.dtype,
        tuple(tensor.shape),
        tuple(tensor.stride()),
        _fits_32bit_stride(tensor),
    )


def _dynamic_vector_layout_key(tensor: torch.Tensor):
    """Key a 1D CuTe argument whose sole extent is runtime dynamic."""
    assert tensor.ndim == 1
    return (
        tensor.dtype,
        tuple(tensor.stride()),
        _fits_32bit_stride(tensor),
    )


def _current_cu_stream(device):
    """CUstream handle of torch's CURRENT stream, queried per call.

    The executor runs the model on a dedicated non-blocking
    ``torch.cuda.Stream`` (``py_executor.execution_stream``), not the default
    stream. Every kernel launch must go to this stream: launching on the DSL
    default stream (what ``.launch()`` does without a ``stream`` argument)
    races with the projections that produce q/k/v/g/beta and with the
    consumers of O/final_state — silent, intermittent corruption in the
    runtime while single-stream unit tests pass. This is also why the value
    must never be cached alongside the scratch buffers: the buffers may be
    created under a different stream (e.g. warmup) than later forwards.
    """
    return cuda_driver.CUstream(torch.cuda.current_stream(device).cuda_stream)


# Cached eqlen dummy cu/ci cute wrappers — shared by K123 and akk_inv.
# Avoids per-call torch.empty + from_dlpack overhead (~10-12us each).
_eqlen_dummy_cache = {}


def _get_eqlen_dummies(device, idx_dtype=torch.int64):
    """Returns cached (cu_ct, ci_ct) cute wrappers for eqlen (B+1=2, NT+1=2)."""
    key = (device.index if device.index is not None else 0, idx_dtype)
    if key not in _eqlen_dummy_cache:
        cu_t = torch.empty(2, dtype=idx_dtype, device=device)
        ci_t = torch.empty(1, 2, dtype=idx_dtype, device=device)
        cu_etype = cutlass.Int64 if idx_dtype == torch.int64 else cutlass.Int32
        _eqlen_dummy_cache[key] = (_ct(cu_t, cu_etype), _ct(ci_t, cu_etype))
    return _eqlen_dummy_cache[key]


def _cute_int_type(dtype):
    """Map PyTorch integer dtype to CUTLASS element type."""
    if dtype == torch.int32:
        return cutlass.Int32
    elif dtype == torch.int64:
        return cutlass.Int64
    else:
        raise ValueError(f"Unsupported integer dtype: {dtype}")


# ========== Fused K1+K2+K3 compilation cache ==========
_fused_k123_cache = {}
# id(cu_seqlens) -> bool. Skips per-call GPU->CPU sync on subsequent calls
# when the same cu_seqlens tensor is reused (typical training/inference loop).
_varlen_pure_cache = {}
# id(cu_seqlens) -> int seqlen, populated alongside _varlen_pure_cache for
# single-seq cu_seqlens.
_varlen_single_seqlen_cache = {}

# id(tensor) -> cute_wrapper. The wrappers themselves are stateless views
# over the tensor's storage, so they remain valid as long as the tensor's
# data pointer / shape / strides don't change. Callers that reuse the same
# tensor objects across iterations (typical benchmark pattern) hit the
# cache; per-call fresh tensors (the executor runtime pattern) rebuild.
_input_wrap_cache = {}


def _prune_on_gc(cache, key, *keyobjs):
    """Drop ``cache[key]`` when any of ``keyobjs`` is garbage-collected.

    The id()-keyed caches in this module are only sound while the keyed
    Python objects stay alive — CPython reuses an object's address (its id)
    the moment it is freed, so a dead entry could otherwise be served for an
    unrelated new tensor with recycled id but different contents/storage.
    A finalizer runs at dealloc, before the address can be recycled, so a
    pruned entry can never alias a new object. This also bounds cache growth
    when callers pass fresh tensors every call.
    """
    for o in keyobjs:
        weakref.finalize(o, cache.pop, key, None)


def _ct_cached(t, etype):
    """`_ct(t, etype)` with id(t)-based cache. Returns the same cute wrapper
    for repeated calls with the same tensor object, avoiding per-call
    `from_dlpack` overhead (~5-10us each).

    ONLY use for tensors with process-long lifetime (module params, the
    module-level scratch from ``_get_buffers``): the cached wrapper pins the
    tensor's storage, so the weakref pruning never fires for the keyed
    object and a per-call activation would be pinned forever (~100MB/call
    leak in the executor runtime). Per-call tensors must use plain ``_ct``.
    """
    key = (id(t), etype)
    w = _input_wrap_cache.get(key)
    if w is None:
        w = _ct(t, etype)
        _input_wrap_cache[key] = w
        _prune_on_gc(_input_wrap_cache, key, t)
    return w


# Cache for dt_bias `.float().contiguous().view(H, K)` + cute wrapper.
# dt_bias is typically a nn.Parameter — same object across iterations.
_dt_bias_cache = {}


def _get_dt_bias_ct(dt_bias, H, K):
    """Returns cached cute wrapper for dt_bias.float().view(H, K)."""
    key = (id(dt_bias), H, K)
    entry = _dt_bias_cache.get(key)
    if entry is None:
        # Copy (never view) the keyed object: a view would pin it, making
        # the entry immortal (weakref pruning could never fire). Callers
        # pass a fresh .detach() per call, so an aliasing entry would leak.
        bias_t = dt_bias.detach().float().reshape(H, K).clone()
        entry = (bias_t, _ct(bias_t, cutlass.Float32))
        _dt_bias_cache[key] = entry
        _prune_on_gc(_dt_bias_cache, key, dt_bias)
    return entry[1]


# Cache for the empty 1x1 fp32 bias tensor used when dt_bias is None.
_empty_bias_cache = {}


def _get_empty_bias_ct(device):
    idx = device.index if device.index is not None else 0
    if idx not in _empty_bias_cache:
        t = torch.empty(1, 1, dtype=torch.float32, device=device)
        _empty_bias_cache[idx] = _ct(t, cutlass.Float32)
    return _empty_bias_cache[idx]


# K4 varlen cu_seqlens / chunk_offsets cute wrappers. Caches the int32-cast
# tensor and its mark_layout_dynamic wrapper so they survive multiple calls
# with the same input objects.
_k4_varlen_cu_co_cache = {}


def _get_k4_varlen_cu_co(cu_seqlens, chunk_offsets):
    key = (id(cu_seqlens), id(chunk_offsets))
    entry = _k4_varlen_cu_co_cache.get(key)
    if entry is None:
        # Always materialize copies: if the cached value aliased the keyed
        # object, the entry would pin it and the weakref pruning could
        # never fire (immortal entry).
        cu_int32 = cu_seqlens.to(torch.int32).contiguous()
        if cu_int32 is cu_seqlens:
            cu_int32 = cu_seqlens.clone()
        co_int32 = chunk_offsets.to(torch.int32).contiguous()
        if co_int32 is chunk_offsets:
            co_int32 = chunk_offsets.clone()
        cu_ct = from_dlpack(cu_int32, assumed_align=4).mark_layout_dynamic()
        cu_ct.element_type = cutlass.Int32
        co_ct = from_dlpack(co_int32, assumed_align=4).mark_layout_dynamic()
        co_ct.element_type = cutlass.Int32
        # Hold refs to the int32 tensors so they don't get GC'd and the
        # underlying storage remain valid as long as cu_ct / co_ct live.
        entry = (cu_int32, co_int32, cu_ct, co_ct)
        _k4_varlen_cu_co_cache[key] = entry
        _prune_on_gc(_k4_varlen_cu_co_cache, key, cu_seqlens, chunk_offsets)
    return entry[2], entry[3]


# Cache the (cu_for_k4, chunk_offsets_for_k4) pair keyed by id(cu_seqlens).
# Both tensors are computed on-GPU with no host sync — replaces the previous
# `cu_seqlens.cpu().tolist() + Python loop + torch.tensor` chain that forced
# a 50-200us GPU->CPU stall on the K4 prep path every varlen call.
_varlen_k4_input_cache = {}


def _get_varlen_k4_inputs(cu_seqlens, BT):
    key = id(cu_seqlens)
    entry = _varlen_k4_input_cache.get(key)
    if entry is None:
        # Copy (never alias) the keyed object — see _get_k4_varlen_cu_co.
        cu_int32 = cu_seqlens.to(torch.int32).contiguous()
        if cu_int32 is cu_seqlens:
            cu_int32 = cu_seqlens.clone()
        seq_lens = cu_int32[1:] - cu_int32[:-1]
        chunk_counts = (seq_lens + (BT - 1)) // BT
        zero = torch.zeros(1, dtype=torch.int32, device=cu_int32.device)
        co_int32 = torch.cat([zero, torch.cumsum(chunk_counts, dim=0).to(torch.int32)])
        co_int32 = co_int32.contiguous()
        _varlen_k4_input_cache[key] = (cu_int32, co_int32)
        _prune_on_gc(_varlen_k4_input_cache, key, cu_seqlens)
    return _varlen_k4_input_cache[key]


# ========== BF16 akk_inv compilation cache ==========
_akk_inv_cache = {}
# ========== K4 persistent (varlen via cu_seqlens) compilation cache ==========
_k4p_cache = {}
_k4p_tm_ws = {}

# Buffer cache: avoid re-allocating ~67us of intermediate tensors per call.
# Also caches cute.Tensor wrappers (saves ~7us each call from from_dlpack).
# LRU-bounded: entries are keyed by (B, T, ...) shapes, and the runtime
# executor calls with a different token count per prefill batch — an
# unbounded dict would pin ~T*150KB of scratch per distinct shape forever.
_buf_cache = {}
_BUF_CACHE_MAX_ENTRIES = 8

# Padded-input scratch cache for the eqlen partial-chunk path. Keyed by
# (B, T_padded, H, K, dtype_qkv, dtype_g, dtype_beta, device, real_T).
# real_T is part of the key so the g sentinel tail [real_T:T_padded] = -1e3
# is set once and reused across calls with the same shape. LRU-bounded like
# _buf_cache: real_T varies per prefill batch, so an unbounded dict would pin
# scratch for every distinct token count forever.
_padded_input_cache = {}
_PAD_CACHE_MAX_ENTRIES = 8

# Sentinel-padded g scratch for varlen single-seq Phase 2.1 path. Keyed by
# (B, T_padded, H, K, dtype, device, real_T). The tail [real_T:T_padded] is
# pre-set to -1e3 once at cache init; subsequent calls only overwrite the
# valid prefix [0, real_T).
_g_sentinel_cache = {}


def _get_g_sentinel_buffer(B, T_padded, H, K, dtype_g, device, real_T):
    key = (B, T_padded, H, K, dtype_g, device.index if device.index is not None else 0, real_T)
    e = _g_sentinel_cache.get(key)
    if e is None:
        e = torch.zeros(B, T_padded, H, K, dtype=dtype_g, device=device)
        if real_T < T_padded:
            e[:, real_T:] = -1000.0
        while len(_g_sentinel_cache) >= _PAD_CACHE_MAX_ENTRIES:
            _g_sentinel_cache.pop(next(iter(_g_sentinel_cache)))
        _g_sentinel_cache[key] = e
    else:
        # LRU refresh so hot shapes survive eviction.
        _g_sentinel_cache[key] = _g_sentinel_cache.pop(key)
    return e


def _get_padded_input_buffers(B, T_padded, H, K, dtype_qkv, dtype_g, dtype_beta, device, real_T):
    key = (
        B,
        T_padded,
        H,
        K,
        dtype_qkv,
        dtype_g,
        dtype_beta,
        device.index if device.index is not None else 0,
        real_T,
    )
    e = _padded_input_cache.get(key)
    if e is None:
        q_pad = torch.zeros(B, T_padded, H, K, dtype=dtype_qkv, device=device)
        k_pad = torch.zeros(B, T_padded, H, K, dtype=dtype_qkv, device=device)
        v_pad = torch.zeros(B, T_padded, H, K, dtype=dtype_qkv, device=device)
        beta_pad = torch.zeros(B, T_padded, H, dtype=dtype_beta, device=device)
        # g: zero in [0, real_T), sentinel -1e3 in [real_T, T_padded). Caller's
        # data overwrites the prefix each call; the sentinel tail never moves.
        g_pad = torch.zeros(B, T_padded, H, K, dtype=dtype_g, device=device)
        if real_T < T_padded:
            g_pad[:, real_T:] = -1000.0
        e = (q_pad, k_pad, v_pad, g_pad, beta_pad)
        while len(_padded_input_cache) >= _PAD_CACHE_MAX_ENTRIES:
            _padded_input_cache.pop(next(iter(_padded_input_cache)))
        _padded_input_cache[key] = e
    else:
        # LRU refresh so hot shapes survive eviction.
        _padded_input_cache[key] = _padded_input_cache.pop(key)
    return e


def _get_buffers(dev, dtype_k, B, T, H, K_dim, V_dim, NT, N_seqs, BT, varlen=False):
    """All beta fusion lives in akk_inv kernel epilogue (post-inv column-scale)."""
    key = (dev.index or 0, B, T, H, K_dim, V_dim, NT, N_seqs, varlen)
    if key not in _buf_cache:
        bf16 = cutlass.BFloat16
        fp32 = cutlass.Float32
        # Varlen chunk-tile kernels (akk_inv's Stage-0 cp.async A-tile load,
        # K4's A-tile loads) transfer the full BT-row tile of every chunk
        # and neutralize invalid rows only after the access; when the
        # batch's FINAL chunk is partial they read up to BT-1 rows past the
        # logical T. These are driver-owned scratch buffers, so honor the
        # kernel's boundary-at-the-data contract by allocating one chunk of
        # zeroed slack past T (the eqlen path gets the same guarantee from
        # its 256-multiple input padding). The slack rows are never
        # consumed — OOB rows are zeroed in SMEM or masked at stores.
        # Input tensors (beta) get the opposite treatment: the K123 kernel
        # bounds-checks those loads, since the driver doesn't own them.
        if varlen:
            assert B == 1, f"varlen expects packed B=1 input, got B={B}"
        T_alloc = T + BT if varlen else T

        def _with_slack(ctor, *shape, dtype):
            full = ctor(*shape, device=dev, dtype=dtype)
            return full[:, :T]

        k_scaled = _with_slack(torch.empty, B, T_alloc, H, K_dim, dtype=dtype_k)  # raw, no beta
        kg = _with_slack(torch.empty, B, T_alloc, H, K_dim, dtype=dtype_k)
        q_scaled = _with_slack(torch.empty, B, T_alloc, H, K_dim, dtype=dtype_k)
        gk_last_exp = torch.empty(B, NT, H, K_dim, device=dev, dtype=torch.float32)
        A_qk = _with_slack(torch.zeros, B, T_alloc, H, BT, dtype=dtype_k)
        A_kk = _with_slack(torch.zeros, B, T_alloc, H, BT, dtype=dtype_k)
        O_flat = _with_slack(torch.empty, B, T_alloc, H, V_dim, dtype=dtype_k)
        cu_eqlen = torch.arange(0, (B + 1) * T, T, dtype=torch.int32, device=dev)
        # co_eqlen is consumed only on the eqlen path (varlen K4 derives its
        # chunk offsets from cu_seqlens). Eqlen inputs are pre-padded to a
        # 256-multiple upstream, so T // BT >= 4 there; varlen calls can
        # carry T < BT (short-prompt batches), where arange(step=T // BT)
        # would raise "step must be nonzero" building this dead buffer.
        # Clamp the step so the buffer stays constructible.
        nt_eq = max(T // BT, 1)
        co_eqlen = torch.arange(0, (B + 1) * nt_eq, nt_eq, dtype=torch.int32, device=dev)

        T_total = B * T
        A_kk_flat = A_kk.reshape(T_total, H, BT)
        A_qk_flat = A_qk.reshape(T_total, H, BT)
        KS_flat = k_scaled.reshape(T_total, H, K_dim)
        QS_flat = q_scaled.reshape(T_total, H, K_dim)
        KG_flat = kg.reshape(T_total, H, K_dim)
        O_token = O_flat.reshape(T_total, H, V_dim)
        gk_flat = gk_last_exp.reshape(-1, H, K_dim)

        def _wrap(t, etype):
            r = from_dlpack(t, assumed_align=16).mark_layout_dynamic()
            r.element_type = etype
            return r

        a_ct = _wrap(A_kk_flat, bf16)
        aqc_ct = _wrap(A_qk_flat, bf16)
        ks_ct = _wrap(KS_flat, bf16)  # raw k_scaled (beta absorbed in akk_inv)
        qs_ct = _wrap(QS_flat, bf16)
        kg_ct = _wrap(KG_flat, bf16)
        o_ct = _wrap(O_token, bf16)
        gk_ct = _wrap(gk_flat, fp32)
        cu_eqlen_ct = from_dlpack(cu_eqlen, assumed_align=4).mark_layout_dynamic()
        cu_eqlen_ct.element_type = cutlass.Int32
        co_eqlen_ct = from_dlpack(co_eqlen, assumed_align=4).mark_layout_dynamic()
        co_eqlen_ct.element_type = cutlass.Int32

        # akk_inv views: bf16 storage reinterpreted as fp32 (packed 2x bf16 -> 1x fp32).
        # Layout-dynamic so the single compiled akk_inv (shape-independent
        # key) accepts them regardless of T; the host rebuilds its own
        # runtime-shaped views from the iterator anyway.
        akk_in_view = from_dlpack(A_kk, assumed_align=16).mark_layout_dynamic()
        akk_in_view.element_type = fp32
        akk_out_view = from_dlpack(A_kk, assumed_align=16).mark_layout_dynamic()
        akk_out_view.element_type = fp32

        cute_wrappers = dict(
            a_ct=a_ct,
            aqc_ct=aqc_ct,
            ks_ct=ks_ct,
            qs_ct=qs_ct,
            kg_ct=kg_ct,
            o_ct=o_ct,
            gk_ct=gk_ct,
            cu_eqlen_ct=cu_eqlen_ct,
            co_eqlen_ct=co_eqlen_ct,
            akk_in_view=akk_in_view,
            akk_out_view=akk_out_view,
            # Filled lazily on first launch — saves cache_key tuple build +
            # outer dict lookup on subsequent calls.
            _k123_fns={},
            _akk_inv_fn=None,
        )

        while len(_buf_cache) >= _BUF_CACHE_MAX_ENTRIES:
            _buf_cache.pop(next(iter(_buf_cache)))
        _buf_cache[key] = (
            k_scaled,
            kg,
            q_scaled,
            gk_last_exp,
            A_qk,
            A_kk,
            O_flat,
            cu_eqlen,
            co_eqlen,
            cute_wrappers,
        )
    else:
        # LRU refresh so hot shapes survive eviction.
        _buf_cache[key] = _buf_cache.pop(key)
    return _buf_cache[key]


def _launch_k4_persistent(
    cute_wrappers,
    v_beta,
    state_pool,
    state_indices,
    cu_seqlens,
    chunk_offsets,
    cu_eqlen_passed=False,
    num_sm=148,
    H=None,
    V_dim=None,
):
    """Launch persistent K4 with cached CuTe wrappers.

    Precondition: all sequence lengths in cu_seqlens must be > 0 (a
    zero-length sequence deadlocks the kernel's chunk-loop barriers; see the
    k4_persistent module docstring). Not validated here: cu_seqlens is on the
    GPU and a host-side check would sync the hot path, and the prefill
    runtime never emits zero-length sequences.

    No fast-launch (args-tuple) cache here: such a cache pins the per-call
    activation and state-pool tensors via their CuTe wrappers (the wrapper holds the
    storage, so the keyed object never dies and weakref pruning never
    fires) — a per-call GPU memory leak in the executor runtime. The
    per-call cost is re-wrapping a handful of tensors (~10us each).
    """
    bf16 = cutlass.BFloat16
    N_seqs = cu_seqlens.shape[0] - 1
    dev = v_beta.device
    dev_idx = dev.index or 0

    s_ct = _dynamic_dlpack_arg(state_pool)
    a_ct = cute_wrappers["a_ct"]
    b_ct = cute_wrappers["ks_ct"]  # raw k_scaled (beta absorbed in akk_inv)
    q_ct = cute_wrappers["qs_ct"]
    aqc_ct = cute_wrappers["aqc_ct"]
    kg_ct = cute_wrappers["kg_ct"]
    o_ct = cute_wrappers["o_ct"]
    gk_ct = cute_wrappers["gk_ct"]

    # v is a per-call activation — wrap fresh every call (never cache; see
    # _ct_cached docstring).
    v_view = v_beta.reshape(-1, H, V_dim) if v_beta.dim() == 4 else v_beta
    v_ct = from_dlpack(v_view, assumed_align=16).mark_layout_dynamic()
    v_ct.element_type = bf16

    if cu_eqlen_passed:
        cu_ct = cute_wrappers["cu_eqlen_ct"]
        co_ct = cute_wrappers["co_eqlen_ct"]
    else:
        cu_ct, co_ct = _get_k4_varlen_cu_co(cu_seqlens, chunk_offsets)

    state_indices_ct = _dynamic_dlpack_arg(state_indices)

    # Allocate for the maximum persistent grid once. The scheduler applies
    # min(N_seqs * H, num_sm) at runtime, so pre-minimizing here would turn
    # every distinct request count into another compiled host function.
    tm_key = (dev_idx, num_sm)
    if tm_key not in _k4p_tm_ws:
        tm_ws_t = torch.zeros(
            num_sm * _K4P_NTM * _K4P_BTM,
            dtype=torch.uint8,
            device=dev,
        )
        tm_ct = from_dlpack(tm_ws_t, assumed_align=16)
        tm_ct.element_type = cutlass.Uint8
        _k4p_tm_ws[tm_key] = (tm_ws_t, tm_ct)
    else:
        tm_ws_t, tm_ct = _k4p_tm_ws[tm_key]

    # Launch on torch's CURRENT stream — see _current_cu_stream.
    stream = _current_cu_stream(dev)

    # H and the state V/K dims MUST be in the key: s_ct bakes the inner
    # [H, V, K] shape and all strides at compile time, so the compiled
    # function is head-count- and layout-specific. N_seqs and the
    # token/chunk counts remain runtime values.
    cache_key = (
        dev_idx,
        num_sm,
        H,
        state_pool.shape[-1],
        V_dim,
        _layout_key(state_pool),
        # The sole dimension of this vector is runtime dynamic. Excluding
        # its extent lets warmup serve full and underfilled context batches.
        _dynamic_vector_layout_key(state_indices),
    )
    k4_fn = _k4p_cache.get(cache_key)
    if k4_fn is None:
        host_fn = _k4p_make_host(num_sm=num_sm)
        k4_fn = cute.compile(
            host_fn,
            a_ct,
            b_ct,
            v_ct,
            q_ct,
            aqc_ct,
            kg_ct,
            o_ct,
            gk_ct,
            s_ct,
            state_indices_ct,
            cu_ct,
            co_ct,
            tm_ct,
            N_seqs,
            stream,
        )
        _k4p_cache[cache_key] = k4_fn

    args = (
        a_ct,
        b_ct,
        v_ct,
        q_ct,
        aqc_ct,
        kg_ct,
        o_ct,
        gk_ct,
        s_ct,
        state_indices_ct,
        cu_ct,
        co_ct,
        tm_ct,
        N_seqs,
        stream,
    )
    k4_fn(*args)


def _launch_fused_k123_inv(
    q,
    k,
    g,
    A_log,
    beta,
    scale,
    k_scaled,
    kg,
    q_scaled,
    gk_last_exp,
    A_qk,
    A_kk_inv,
    cu_seqlens,
    chunk_indices,
    is_varlen,
    NT,
    dt_bias=None,
    safe_gate=False,
    lower_bound=None,
    akk_in_view=None,
    akk_out_view=None,
    cute_wrappers=None,
    varlen_pure_override=False,
    varlen_is_aligned=None,
):
    """Persistent K1+K2 (writes A_kk in I+L format with diag=1) chained with
    BF16 akk_inv (in-place inversion). Final A_kk_inv = (I+L)^-1."""

    # No fast-launch (args-tuple) cache: it would pin the per-call q/k/g/beta
    # activations via their cute wrappers (the wrapper holds the storage, so
    # the keyed objects never die and weakref pruning never fires) — a
    # per-call GPU memory leak in the executor runtime. Wrapper gathering
    # below costs ~10us per tensor.
    B, T, H, K = q.shape
    BT = 64
    dev = q.device.index or 0
    T_padded = T if is_varlen else None
    has_bias = dt_bias is not None

    # Prefer the host-derived alignment property supplied by the model runtime.
    # Direct low-level callers that omit it retain the cached device inference
    # fallback for backward compatibility.
    varlen_pure = False
    if is_varlen and cu_seqlens is not None:
        if varlen_pure_override:
            # Caller (Phase 2.1 single-seq path) sentinel-padded the data for
            # this call; the cached per-object verdict stays untouched.
            varlen_pure = True
        elif varlen_is_aligned is not None:
            varlen_pure = varlen_is_aligned
        else:
            _vp_key = id(cu_seqlens)
            if _vp_key not in _varlen_pure_cache:
                cu_cpu = cu_seqlens.cpu().tolist()
                seq_lens = [cu_cpu[i + 1] - cu_cpu[i] for i in range(len(cu_cpu) - 1)]
                _varlen_pure_cache[_vp_key] = all((sl % BT) == 0 for sl in seq_lens)
                _prune_on_gc(_varlen_pure_cache, _vp_key, cu_seqlens)
            varlen_pure = _varlen_pure_cache[_vp_key]
    # B/NT/T are runtime args of the compiled host_fn (see make_host_function),
    # avoiding shape-dependent grid and view specialization. Varlen remains
    # fully shape-independent; eqlen retains T only as a cache discriminator
    # for the raw tensor batch strides described below.
    #
    # cu/ci DTYPE must be part of the key: the kernel reads
    # mCuSeqlens/mChunkIndices with the element type baked at compile time
    # (int64 elements are addressed with stride 8, int32 with stride 4).
    # Reusing an int64-compiled kernel on int32 tensors (or vice versa)
    # misaddresses every cu/ci element — garbage seq ids/chunk starts ->
    # OOB reads (cudaErrorIllegalAddress) or silent corruption. Observed as
    # a crash when a dump-replay batch (int64 cu/ci) preceded a synthetic
    # batch (int32 cu/ci) in one process.
    cu_ci_dtypes = (cu_seqlens.dtype, chunk_indices.dtype) if is_varlen else None
    # Eqlen kernels directly index mBeta/mAqk/mAkk using the input wrappers'
    # compile-time batch stride, which depends on T. Varlen fixes B=1, so that
    # batch stride is never used and all T shapes can keep sharing one compile.
    eqlen_t_key = T if not is_varlen else -1
    cache_key = (
        H,
        is_varlen,
        dev,
        has_bias,
        safe_gate,
        varlen_pure,
        cu_ci_dtypes,
        eqlen_t_key,
    )

    # Inputs are guaranteed contiguous by upstream linear projections.
    # A_log is fp32 model param; .float() is no-op when dtype already matches.
    # q/k/g/beta/cu/ci are per-call activations: plain _ct, never cached
    # (see _ct_cached docstring). The _get_buffers scratch below is
    # module-persistent, so caching its wrappers is safe and worthwhile.
    q_ct = _ct(q, cutlass.BFloat16)
    k_ct = _ct(k, cutlass.BFloat16)
    g_ct = _ct(g, cutlass.BFloat16)
    alog_ct = _ct(A_log if A_log.dtype == torch.float32 else A_log.float(), cutlass.Float32)
    beta_ct = _ct(beta, cutlass.BFloat16)

    ks_ct = _ct_cached(k_scaled, cutlass.BFloat16)
    kg_ct = _ct_cached(kg, cutlass.BFloat16)
    qs_ct = _ct_cached(q_scaled, cutlass.BFloat16)
    gk_ct = _ct_cached(gk_last_exp, cutlass.Float32)
    aqk_ct = _ct_cached(A_qk, cutlass.BFloat16)
    akk_ct = _ct_cached(A_kk_inv, cutlass.BFloat16)

    if is_varlen:
        cu_ct = _ct(cu_seqlens, _cute_int_type(cu_seqlens.dtype))
        ci_ct = _ct(chunk_indices, _cute_int_type(chunk_indices.dtype))
    else:
        cu_ct, ci_ct = _get_eqlen_dummies(q.device, torch.int64)

    if dt_bias is not None:
        bias_ct = _get_dt_bias_ct(dt_bias, H, K)
    else:
        bias_ct = _get_empty_bias_ct(q.device)
    lb_val = float(lower_bound) if lower_bound is not None else 0.0

    # Launch on torch's CURRENT stream — see _current_cu_stream.
    stream = _current_cu_stream(q.device)

    ct_args = (
        q_ct,
        k_ct,
        g_ct,
        alog_ct,
        beta_ct,
        scale,
        ks_ct,
        kg_ct,
        qs_ct,
        gk_ct,
        aqk_ct,
        akk_ct,
        cu_ct,
        ci_ct,
        bias_ct,
        lb_val,
        # Runtime shape scalars (rt_nt / rt_b / rt_t_total in host_fn).
        NT,
        B,
        B * (T_padded if is_varlen else NT * BT),
        stream,
    )

    if cache_key not in _fused_k123_cache:
        host_fn = _fused_make_host(
            B,
            NT,
            H,
            is_varlen=is_varlen,
            T_padded=T_padded,
            has_bias=has_bias,
            use_safe_gate=safe_gate,
            varlen_pure=varlen_pure,
        )
        _fused_k123_cache[cache_key] = cute.compile(host_fn, *ct_args)
    k123_fn = _fused_k123_cache[cache_key]
    k123_fn(*ct_args)

    # ===== Chained BF16 akk_inv (in-place: A_kk_inv = (I+L)^-1) =====
    # Views are layout-dynamic so one compiled akk_inv serves every batch
    # shape (the host builds its own runtime-shaped views from the iterator;
    # the wrapper layout only matters for the call signature).
    if akk_in_view is None:
        akk_in_view = from_dlpack(A_kk_inv, assumed_align=16).mark_layout_dynamic()
        akk_in_view.element_type = cutlass.Float32
        akk_out_view = from_dlpack(A_kk_inv, assumed_align=16).mark_layout_dynamic()
        akk_out_view.element_type = cutlass.Float32

    if is_varlen:
        # Reuse the cached cute wrappers from K123 (same tensor objects).
        akk_cu_ct = cu_ct
        akk_ci_ct = ci_ct
        is_varlen_int = 1
        T_val = T
    else:
        akk_cu_ct, akk_ci_ct = cu_ct, ci_ct
        is_varlen_int = 0
        T_val = NT * BT

    # B/NT/T_val remain runtime args of akk_inv_host. Eqlen T is still part of
    # the cache key because the raw mBeta wrapper bakes its batch stride.
    # cu/ci dtype is a specializer for the same reason as in the K123 key above
    # (element type baked into the compiled reader).
    akk_cache_key = (H, is_varlen, dev, cu_ci_dtypes, eqlen_t_key)
    if akk_cache_key not in _akk_inv_cache:
        _akk_inv_cache[akk_cache_key] = cute.compile(
            _akk_inv_host,
            akk_in_view,
            akk_out_view,
            beta_ct,
            B,
            NT,
            H,
            akk_cu_ct,
            akk_ci_ct,
            is_varlen_int,
            T_val,
            stream,
        )
    akk_fn = _akk_inv_cache[akk_cache_key]
    akk_args = (akk_in_view, akk_out_view, beta_ct, B, NT, akk_cu_ct, akk_ci_ct, T_val, stream)
    akk_fn(*akk_args)


def _chunk_kda_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    state_pool: torch.Tensor,
    state_indices: torch.Tensor,
    scale: float,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
    chunk_size: int = 64,
    safe_gate: bool = False,
    lower_bound: float | None = None,
    use_gate_in_kernel: bool = False,
    A_log: torch.Tensor | None = None,
    dt_bias: torch.Tensor | None = None,
    varlen_is_aligned: bool | None = None,
    single_sequence_length: int | None = None,
) -> torch.Tensor:
    """Run optimized KDA prefill against indexed recurrent-state rows."""
    if safe_gate and lower_bound is None:
        lower_bound = -5.0

    is_varlen = cu_seqlens is not None

    if q.shape[1] == 0:
        raise ValueError("Indexed KDA prefill does not support an empty token batch.")

    B, T, H, K = q.shape
    V_dim = v.shape[-1]
    device = q.device
    BT = 64

    # Phase 1: handle eqlen with T % 64 != 0 entirely on the eqlen path —
    # never borrow varlen's mask code. Pad inputs to a CHUNKS_PER_BLOCK*BT
    # = 256 multiple so the persistent scheduler's `cgs_per_head = NT // 4`
    # divides cleanly and every chunk has 64 valid rows of (zero-padded /
    # sentinel-padded) data. K123 eqlen kernel runs unchanged — it doesn't
    # know partial chunks exist, the boundary is handled at the data
    # boundary (K4's bounded-TMA principle, just expressed via host pad
    # since K123 stores via autovec_copy not TMA).
    #
    # Varlen with non-aligned seq lengths is a separate problem (Phase 2):
    # multi-seq varlen can't be host-padded without repacking memory.
    real_T = T
    needs_eqlen_pad = (not is_varlen) and (T % BT != 0)
    if needs_eqlen_pad:
        if B != 1:
            raise NotImplementedError(
                f"eqlen with B>1 and T % {BT} != 0 not supported (got B={B}, T={T})."
            )
        CPB_BT = 4 * BT  # CHUNKS_PER_BLOCK * BT, the cgs_per_head divisibility unit
        T_padded = ((T + CPB_BT - 1) // CPB_BT) * CPB_BT
        # Pre-allocated padded scratch buffers (per (B,T_padded,H,K,dtype) cache
        # key). torch.cat would reallocate + copy the full 200MB q tensor every
        # call — caching the destination buffer drops that to a single slice
        # copy of the valid prefix (caller already lives in our buffer for
        # subsequent calls reusing the same id, but we re-copy unconditionally
        # since the caller may have updated the data in-place).
        q_pad, k_pad, v_pad, g_pad, beta_pad = _get_padded_input_buffers(
            B, T_padded, H, K, q.dtype, g.dtype, beta.dtype, q.device, real_T
        )
        # q/k/v/beta zero-padded → K1/K2 MMAs naturally produce 0 for OOB rows.
        # Tail [real_T:] of q_pad/k_pad/v_pad/beta_pad is pre-zeroed at cache
        # init and never written, so we only copy the valid prefix.
        q_pad[:, :real_T].copy_(q)
        k_pad[:, :real_T].copy_(k)
        v_pad[:, :real_T].copy_(v)
        beta_pad[:, :real_T].copy_(beta)
        # g uses a -1e3 sentinel so the gate activation saturates to 0 for OOB
        # rows (both safe_gate sigmoid and softplus paths). Plain g=0 gives
        # nonzero activation that would corrupt the cumsum past seq end. The
        # tail is set to -1e3 once at cache init; we only copy the valid prefix.
        g_pad[:, :real_T].copy_(g)
        q, k, v, g, beta = q_pad, k_pad, v_pad, g_pad, beta_pad
        T = T_padded  # downstream buffer alloc + kernel layout use T_padded

    # Phase 2.1: varlen with a SINGLE non-aligned sequence — caller already
    # zero-padded q/k/v/beta to a 64-multiple (FLA convention), but g's tail
    # is also zero, which causes the gate activation to be non-zero past seq
    # end and corrupts the cumsum / GkLast. We sentinel-pad g (cheap: ~5MB
    # copy) and force VARLEN_PURE=1 so all 4 mask sites compile-elide. Same
    # K4 "boundary at the data" principle as the eqlen path.
    #
    # Multi-seq varlen is NOT handled here — its OOB regions overlap with
    # adjacent seqs' data, so sentinel-pad on g would corrupt the next seq.
    # Multi-seq optimization needs per-seq dynamic tensormap (Phase 2.2).
    needs_varlen_single_pad = False
    if is_varlen and cu_seqlens is not None and cu_seqlens.shape[0] == 2 and B == 1:
        if single_sequence_length is not None:
            single_sequence_is_aligned = single_sequence_length % BT == 0
            if varlen_is_aligned is not None and varlen_is_aligned != single_sequence_is_aligned:
                raise ValueError("varlen_is_aligned disagrees with single_sequence_length")
            varlen_is_aligned = single_sequence_is_aligned
            if not single_sequence_is_aligned:
                real_T = single_sequence_length
                needs_varlen_single_pad = True
        else:
            _vl_key = id(cu_seqlens)
            if _vl_key not in _varlen_pure_cache:
                cu_cpu = cu_seqlens.cpu().tolist()
                sl = cu_cpu[1] - cu_cpu[0]
                _varlen_pure_cache[_vl_key] = sl % BT == 0
                _varlen_single_seqlen_cache[_vl_key] = sl
                _prune_on_gc(_varlen_pure_cache, _vl_key, cu_seqlens)
                _prune_on_gc(_varlen_single_seqlen_cache, _vl_key, cu_seqlens)
            varlen_is_aligned = _varlen_pure_cache[_vl_key]
            if not varlen_is_aligned:
                real_T = _varlen_single_seqlen_cache[_vl_key]
                needs_varlen_single_pad = True
    varlen_pure_override = False
    if needs_varlen_single_pad:
        # q/k/v/beta already zero-padded by caller (FLA convention). Re-build g
        # with -1000 sentinel in the tail so VARLEN_PURE=1 path is correct.
        # Cache the resulting g buffer so repeated calls with same input ids
        # don't re-allocate.
        cur_T = q.shape[1]  # caller's padded T
        assert cur_T % BT == 0 and cur_T >= real_T, (
            f"varlen single-seq path expects caller-padded input "
            f"(T={cur_T}, seqlen={real_T}); see "
            "KDAKernelDispatch.prefill_chunk_kda"
        )
        g_pad = _get_g_sentinel_buffer(B, cur_T, H, K, g.dtype, g.device, real_T)
        g_pad[:, :real_T].copy_(g[:, :real_T])
        g = g_pad
        # Force VARLEN_PURE=1 for THIS call only (the data is sentinel-padded
        # now). Never write the override into _varlen_pure_cache: the cached
        # verdict must keep saying "not pure" so a later call with the SAME
        # cu_seqlens object re-runs this sentinel pad — poisoning the cache
        # made the second call skip the pad while still compiling the
        # mask-free variant (silent tail corruption).
        varlen_pure_override = True

    # Phase 2.2 (multi-seq via host repack) was attempted but isn't net positive:
    # the scatter/gather memcpy cost (~800us GPU bandwidth per call) exceeds
    # the kernel mask-elision savings (~250us). Keeping multi-seq non-pure on
    # the original masked path. The right fix is kernel-level dynamic
    # tensormap (K4-style per-tile bounded TMA) but that requires a major
    # K123 kernel refactor — left as future work.
    # multiseq_info = None

    if is_varlen:
        if chunk_indices is None:
            chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
        NT = len(chunk_indices)
        if NT < 4:
            # The persistent K123 scheduler launches NT // 4 cooperative
            # groups per head; fewer than 4 total chunks produces a
            # zero-size grid (DSLCudaRuntimeError at launch). Callers must
            # route such batches to the FLA path — see
            # KDAKernelDispatch.prefill_chunk_kda.
            raise ValueError(
                f"kda_prefill requires >= 4 total varlen chunks (got {NT}); "
                "route small varlen batches to the FLA fallback"
            )
        N_seqs = len(cu_seqlens) - 1
    else:
        NT = T // BT
        N_seqs = B

    # ===== Cached buffers + cute wrappers (avoid alloc + from_dlpack overhead per call) =====
    (
        k_scaled,
        kg,
        q_scaled,
        gk_last_exp,
        A_qk,
        A_kk,
        O_flat,
        cu_eqlen,
        co_eqlen,
        cute_wrappers,
    ) = _get_buffers(device, k.dtype, B, T, H, K, V_dim, NT, N_seqs, BT, varlen=is_varlen)

    # Beta is fused entirely in akk_inv kernel epilogue (post-inv column-scale).
    # No host v*beta and no K1 k_scaled*beta any more.
    _launch_fused_k123_inv(
        q,
        k,
        g,
        A_log,
        beta,
        scale,
        k_scaled,
        kg,
        q_scaled,
        gk_last_exp,
        A_qk,
        A_kk,
        cu_seqlens,
        chunk_indices,
        is_varlen,
        NT,
        dt_bias=dt_bias,
        safe_gate=safe_gate,
        lower_bound=lower_bound,
        akk_in_view=cute_wrappers["akk_in_view"],
        akk_out_view=cute_wrappers["akk_out_view"],
        cute_wrappers=cute_wrappers,
        varlen_pure_override=varlen_pure_override,
        varlen_is_aligned=varlen_is_aligned,
    )

    # ===== K4: persistent kernel (eqlen + varlen via cu_seqlens) =====
    if is_varlen:
        # GPU-side cumsum + cache by id(cu_seqlens). No host sync.
        cu_for_k4, chunk_offsets_for_k4 = _get_varlen_k4_inputs(cu_seqlens, BT)
    else:
        cu_for_k4 = cu_eqlen
        chunk_offsets_for_k4 = co_eqlen

    _launch_k4_persistent(
        cute_wrappers,
        v,
        state_pool,
        state_indices,
        cu_for_k4,
        chunk_offsets_for_k4,
        cu_eqlen_passed=(not is_varlen),
        H=H,
        V_dim=V_dim,
    )

    o = O_flat
    if needs_eqlen_pad:
        # Caller called with original T = real_T; their downstream code expects
        # outputs at that shape. Slice the padded scratch tail back off.
        o = o[:, :real_T]
    return o


class KdaPrefillRunner:
    """Runs the source-integrated Kimi K3 KDA prefill pipeline."""

    @staticmethod
    def forward(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        state_pool: torch.Tensor,
        state_indices: torch.Tensor,
        scale: float,
        cu_seqlens: Optional[torch.Tensor] = None,
        chunk_indices: Optional[torch.Tensor] = None,
        chunk_size: int = 64,
        safe_gate: bool = False,
        lower_bound: Optional[float] = None,
        use_gate_in_kernel: bool = False,
        A_log: Optional[torch.Tensor] = None,
        dt_bias: Optional[torch.Tensor] = None,
        varlen_is_aligned: Optional[bool] = None,
        single_sequence_length: Optional[int] = None,
    ) -> torch.Tensor:
        """Run KDA prefill and update selected recurrent-state rows."""
        if not IS_CUTLASS_DSL_AVAILABLE:
            raise RuntimeError("Kimi K3 KDA prefill requires NVIDIA CUTLASS DSL")
        if chunk_size != 64:
            raise ValueError(f"Kimi K3 KDA prefill requires chunk_size=64, got {chunk_size}")
        if A_log is None:
            raise ValueError("Kimi K3 KDA prefill requires A_log")

        return _chunk_kda_fwd(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            state_pool=state_pool,
            state_indices=state_indices,
            scale=scale,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            chunk_size=chunk_size,
            safe_gate=safe_gate,
            lower_bound=lower_bound,
            use_gate_in_kernel=use_gate_in_kernel,
            A_log=A_log,
            dt_bias=dt_bias,
            varlen_is_aligned=varlen_is_aligned,
            single_sequence_length=single_sequence_length,
        )


def _validate_indexed_state_pool(
    q: torch.Tensor,
    v: torch.Tensor,
    state_pool: torch.Tensor,
    state_indices: torch.Tensor,
    cu_seqlens: Optional[torch.Tensor],
) -> None:
    num_sequences = q.shape[0] if cu_seqlens is None else cu_seqlens.shape[0] - 1
    H, K, V = q.shape[2], q.shape[3], v.shape[3]
    if state_pool.dtype != torch.float32 or state_pool.ndim != 4:
        raise ValueError(
            "Expected state_pool to be a rank-4 fp32 tensor with shape [slots, H, V, K]."
        )
    if state_pool.shape[1:] != (H, V, K):
        raise ValueError(
            f"Expected state_pool shape [slots, {H}, {V}, {K}], got {tuple(state_pool.shape)}."
        )
    expected_inner_strides = (V * K, K, 1)
    if state_pool.stride()[1:] != expected_inner_strides:
        raise ValueError(
            "Indexed KDA prefill requires dense H/V/K inner dimensions; "
            f"expected strides {expected_inner_strides}, got "
            f"{state_pool.stride()[1:]}."
        )
    if state_pool.stride(0) < H * V * K or state_pool.stride(0) % 4 != 0:
        raise ValueError(
            "Indexed KDA prefill requires a non-overlapping, 16-byte-aligned slot stride."
        )
    if state_pool.data_ptr() % 16 != 0:
        raise ValueError("Indexed KDA prefill requires a 16-byte-aligned state pool.")
    if state_indices.ndim != 1 or state_indices.shape[0] != num_sequences:
        raise ValueError(
            f"Expected state_indices shape [{num_sequences}], got {tuple(state_indices.shape)}."
        )
    if state_indices.dtype not in (torch.int32, torch.int64):
        raise ValueError("state_indices must have dtype int32 or int64.")
    if not state_indices.is_contiguous():
        raise ValueError("state_indices must be contiguous.")
    for name, tensor in (
        ("q", q),
        ("v", v),
        ("state_indices", state_indices),
    ):
        if tensor.device != state_pool.device:
            raise ValueError(f"{name} and state_pool must be on the same device.")


if IS_CUTLASS_DSL_AVAILABLE:

    @torch.library.custom_op(
        "trtllm::kda_prefill",
        mutates_args=("state_pool",),
        device_types="cuda",
    )
    def kda_prefill(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        state_pool: torch.Tensor,
        state_indices: torch.Tensor,
        scale: float,
        cu_seqlens: Optional[torch.Tensor] = None,
        chunk_indices: Optional[torch.Tensor] = None,
        chunk_size: int = 64,
        safe_gate: bool = False,
        lower_bound: Optional[float] = None,
        use_gate_in_kernel: bool = False,
        A_log: Optional[torch.Tensor] = None,
        dt_bias: Optional[torch.Tensor] = None,
        varlen_is_aligned: Optional[bool] = None,
        single_sequence_length: Optional[int] = None,
    ) -> torch.Tensor:
        """Run KDA prefill directly against a V-first recurrent-state pool."""
        _validate_indexed_state_pool(
            q,
            v,
            state_pool,
            state_indices,
            cu_seqlens,
        )
        return KdaPrefillRunner.forward(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            state_pool=state_pool,
            state_indices=state_indices,
            A_log=A_log,
            scale=scale,
            dt_bias=dt_bias,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            chunk_size=chunk_size,
            safe_gate=safe_gate,
            lower_bound=lower_bound,
            use_gate_in_kernel=use_gate_in_kernel,
            varlen_is_aligned=varlen_is_aligned,
            single_sequence_length=single_sequence_length,
        )

    @kda_prefill.register_fake
    def _(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        state_pool: torch.Tensor,
        state_indices: torch.Tensor,
        scale: float,
        cu_seqlens: Optional[torch.Tensor] = None,
        chunk_indices: Optional[torch.Tensor] = None,
        chunk_size: int = 64,
        safe_gate: bool = False,
        lower_bound: Optional[float] = None,
        use_gate_in_kernel: bool = False,
        A_log: Optional[torch.Tensor] = None,
        dt_bias: Optional[torch.Tensor] = None,
        varlen_is_aligned: Optional[bool] = None,
        single_sequence_length: Optional[int] = None,
    ) -> torch.Tensor:
        del q, k, g, beta, state_pool, state_indices
        del scale, cu_seqlens, chunk_indices, chunk_size, safe_gate
        del lower_bound, use_gate_in_kernel, A_log, dt_bias
        del varlen_is_aligned, single_sequence_length
        return v.new_empty(v.shape)
