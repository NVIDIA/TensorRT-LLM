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

"""CuTe DSL custom op for Kimi K3 KDA multi-token speculative verify.

Wraps the source-integrated ``kda_decode_mtp_kernel`` (see
``cute_dsl_kernels/blackwell/kimi_k3_kda/kda_mtp_decode.py``) as the
``trtllm::kda_mtp_decode`` operator. One launch fuses, per generation
request: replay of previously-accepted draft tokens from the ``qkg/v/beta``
caches, causal conv + SiLU, Q/K L2 norm, beta sigmoid, lower-bound gate, and
the KDA delta-rule recurrence over the ``1 + num_spec`` new tokens. The
recurrent state and base conv windows are committed **in place** after the
first new (golden) token; the spec tokens are cached for the next round.

State-management contract (differs from the legacy intermediate-buffer +
``update_mamba_states`` promotion flow): after this op returns, the pools
hold the state as of the last golden token, with the new spec tokens pending
in the replay caches. The next round passes ``num_accepted_tokens`` (how
many of those pending drafts the sampler accepted) and the kernel replays
them before the new tokens. No host-side promotion of KDA SSM/conv state is
required or allowed.

Productization deltas vs the drop's host wrapper (``reference.py``):

* ``zero_accepted_hint`` / ``regular_metadata_hint`` are explicit caller
  arguments. The drop derived them by *reading the device tensors*
  (``torch.count_nonzero(...).item()`` / ``torch.equal``) behind
  ``id()``-keyed caches — a host-device sync per novel tensor object plus a
  stale-cache hazard on id reuse. Callers that statically know the pattern
  (benchmarks, the first verify round after prefill) may pass the hints;
  the runtime default (``False``/``False``) is always correct and never
  syncs.
* The compile cache is keyed purely by dtype/shape/stride layouts and
  constexpr flags — no ``id()`` or ``data_ptr`` keys.
* Bias / output-norm / ``pad_slot_id`` arguments (unsupported by the
  specialized kernel, previously validated-then-rejected) are dropped from
  the signature.

Kernel shape contract: ``K == V == 128``, conv width ``W == 4``,
``HV == H``, TILE_V=64. ``num_spec`` is a compile-time constant per cache
allocation. Benchmark-tuned fast variants exist for ``N in (32, 128), H in
(2, 12, 32), num_spec == 2``; other shapes compile the general variant.
"""

from typing import Optional, Tuple

import torch

from tensorrt_llm.logger import logger

from ..cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE

if IS_CUTLASS_DSL_AVAILABLE:
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import from_dlpack

    from ..cute_dsl_kernels.blackwell.kimi_k3_kda.kda_mtp_decode import (
        NUM_THREADS,
        TILE_K,
        kda_decode_mtp_kernel,
    )
else:
    raise ImportError("Kimi K3 KDA MTP decode requires NVIDIA CUTLASS DSL")

_TILE_V = 64


if IS_CUTLASS_DSL_AVAILABLE:

    @cute.jit
    def _run_kda_decode_mtp(
        h0: cute.Tensor,
        x_q: cute.Tensor,
        x_k: cute.Tensor,
        x_v: cute.Tensor,
        w_q: cute.Tensor,
        w_k: cute.Tensor,
        w_v: cute.Tensor,
        cs_q: cute.Tensor,
        cs_k: cute.Tensor,
        cs_v: cute.Tensor,
        A_log: cute.Tensor,
        g: cute.Tensor,
        dt_bias: cute.Tensor,
        beta: cute.Tensor,
        o: cute.Tensor,
        ht: cute.Tensor,
        qkg_cache: cute.Tensor,
        v_cache: cute.Tensor,
        beta_cache: cute.Tensor,
        stage_timing: cute.Tensor,
        ssm_state_indices: cute.Tensor,
        cu_seqlens: cute.Tensor,
        num_accepted_tokens: cute.Tensor,
        precompute_control: cute.Tensor,
        scale: cutlass.Constexpr[float],
        HV: cutlass.Constexpr[int],
        K: cutlass.Constexpr[int],
        V: cutlass.Constexpr[int],
        N: cutlass.Int32,
        NUM_SPEC: cutlass.Constexpr[int],
        TILE_V: cutlass.Constexpr[int],
        KERNEL_WIDTH: cutlass.Constexpr[int],
        lower_bound: cutlass.Constexpr[float],
        USE_FLAT_LAYOUT: cutlass.Constexpr[bool],
        USE_SETMAXREG: cutlass.Constexpr[bool],
        USE_REGULAR_METADATA: cutlass.Constexpr[bool],
        USE_REG_Q_WEIGHTS: cutlass.Constexpr[bool],
        USE_ZERO_ACCEPTED: cutlass.Constexpr[bool],
        FUSE_PRECOMPUTE: cutlass.Constexpr[bool],
        RUNTIME_PRECOMPUTE_FLAG: cutlass.Constexpr[bool],
        PROFILE_STAGES: cutlass.Constexpr[bool],
        stream: cuda.CUstream,
    ):
        if cutlass.const_expr(USE_ZERO_ACCEPTED):
            t_max = 1 + NUM_SPEC
        else:
            t_max = 2 * NUM_SPEC + 1
        smem_qk_layout = cute.make_layout((t_max, K), stride=(K, 1))
        kda_decode_mtp_kernel(
            h0,
            x_q,
            x_k,
            x_v,
            w_q,
            w_k,
            w_v,
            cs_q,
            cs_k,
            cs_v,
            A_log,
            g,
            dt_bias,
            beta,
            o,
            ht,
            qkg_cache,
            v_cache,
            beta_cache,
            smem_qk_layout,
            ssm_state_indices,
            cu_seqlens,
            num_accepted_tokens,
            precompute_control,
            TILE_V,
            scale,
            HV,
            K,
            V,
            NUM_SPEC,
            KERNEL_WIDTH,
            lower_bound,
            USE_FLAT_LAYOUT,
            USE_SETMAXREG,
            USE_REGULAR_METADATA,
            USE_REG_Q_WEIGHTS,
            USE_ZERO_ACCEPTED,
            FUSE_PRECOMPUTE,
            RUNTIME_PRECOMPUTE_FLAG,
            stage_timing,
            PROFILE_STAGES,
        ).launch(grid=(HV, N, 1), block=[NUM_THREADS, 1, 1], stream=stream)


def _require_stride_layout(
    *,
    x_q,
    x_k,
    x_v,
    w_q,
    w_k,
    w_v,
    cs_q,
    cs_k,
    cs_v,
    g,
    beta,
    A_log,
    dt_bias,
    recurrent_state,
    qkg_cache,
    v_cache,
    beta_cache,
    ssm_state_indices,
    cu_seqlens,
    num_accepted_tokens,
    out,
    H,
    HV,
    K,
    V,
    W,
    num_spec,
    T_total,
):
    if x_q.ndim != 4 or x_k.ndim != 4 or x_v.ndim != 4:
        raise ValueError("Expected x_q/x_k/x_v to have shape [1, T, H, D].")
    if x_q.shape != (1, T_total, H, K) or x_k.shape != (1, T_total, H, K):
        raise ValueError(f"Expected x_q/x_k shape [1, {T_total}, {H}, {K}].")
    if x_v.shape != (1, T_total, HV, V):
        raise ValueError(f"Expected x_v shape [1, {T_total}, {HV}, {V}].")
    if g.ndim != 4 or g.shape != (1, T_total, HV, K):
        raise ValueError(f"Expected g shape [1, {T_total}, {HV}, {K}].")
    if beta.ndim != 3 or beta.shape != (1, T_total, HV):
        raise ValueError(f"Expected beta shape [1, {T_total}, {HV}].")
    if out.ndim != 4 or out.shape != (1, T_total, HV, V):
        raise ValueError(f"Expected out shape [1, {T_total}, {HV}, {V}].")

    last_dim_tensors = {
        "x_q": x_q,
        "x_k": x_k,
        "x_v": x_v,
        "g": g,
        "beta": beta,
        "out": out,
        "recurrent_state": recurrent_state,
        "qkg_cache": qkg_cache,
        "v_cache": v_cache,
        "beta_cache": beta_cache,
    }
    for name, tensor in last_dim_tensors.items():
        if tensor.stride(-1) != 1:
            raise ValueError(f"Expected {name} to be contiguous in its last dimension.")

    if w_q.shape != (H * K, W) or w_k.shape != (H * K, W) or w_v.shape != (HV * V, W):
        raise ValueError(f"Expected w_q/w_k shape [{H * K}, {W}] and w_v shape [{HV * V}, {W}].")
    if w_q.stride(1) != 1 or w_k.stride(1) != 1 or w_v.stride(1) != 1:
        raise ValueError("Expected w_q/w_k/w_v to be contiguous in the kernel-width axis.")

    if A_log.ndim != 1 or A_log.shape[0] != H:
        raise ValueError(f"Expected A_log shape [{H}].")
    if dt_bias.ndim != 1 or dt_bias.shape[0] != H * K:
        raise ValueError(f"Expected dt_bias shape [{H * K}].")

    state_s = W - 1 + num_spec
    if cs_q.ndim != 3 or cs_k.ndim != 3 or cs_v.ndim != 3:
        raise ValueError("Expected cs_q/cs_k/cs_v to have shape [pool, dim, S].")
    if cs_q.shape[1] != H * K or cs_k.shape[1] != H * K:
        raise ValueError(f"Expected cs_q/cs_k shape [pool, {H * K}, S].")
    if cs_v.shape[1] != HV * V:
        raise ValueError(f"Expected cs_v shape [pool, {HV * V}, S].")
    if cs_q.shape[2] < state_s or cs_k.shape[2] < state_s or cs_v.shape[2] < state_s:
        raise ValueError(f"Expected conv-state S dimension to be at least {state_s}.")
    if cs_q.stride(1) != 1 or cs_k.stride(1) != 1 or cs_v.stride(1) != 1:
        raise ValueError(
            "Expected cs_q/cs_k/cs_v to use dim-contiguous layout "
            "(allocate as [pool, S, dim] and transpose(1, 2))."
        )

    pool_size = recurrent_state.shape[0]
    if recurrent_state.ndim != 4 or recurrent_state.shape[1:] != (HV, V, K):
        raise ValueError(
            f"Expected recurrent_state shape [pool, {HV}, {V}, {K}] (V-first pool layout)."
        )
    if qkg_cache.ndim != 4 or qkg_cache.shape[1:] != (num_spec, 3, H * K):
        raise ValueError(f"Expected qkg_cache shape [pool, {num_spec}, 3, {H * K}].")
    if v_cache.ndim != 3 or v_cache.shape[1:] != (num_spec, HV * V):
        raise ValueError(f"Expected v_cache shape [pool, {num_spec}, {HV * V}].")
    if beta_cache.ndim != 3 or beta_cache.shape[1:] != (num_spec, HV):
        raise ValueError(f"Expected beta_cache shape [pool, {num_spec}, {HV}].")
    if (
        qkg_cache.shape[0] < pool_size
        or v_cache.shape[0] < pool_size
        or beta_cache.shape[0] < pool_size
    ):
        raise ValueError("Expected cache pool dimensions to cover recurrent_state rows.")

    if ssm_state_indices.ndim != 1 or cu_seqlens.ndim != 1 or num_accepted_tokens.ndim != 1:
        raise ValueError(
            "Expected ssm_state_indices, cu_seqlens, and num_accepted_tokens to be 1D."
        )
    if cu_seqlens.shape[0] != ssm_state_indices.shape[0] + 1:
        raise ValueError("Expected cu_seqlens length to be N + 1.")
    if num_accepted_tokens.shape[0] != ssm_state_indices.shape[0]:
        raise ValueError("Expected num_accepted_tokens length to match N.")


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


def _from_dlpack_arg(tensor: torch.Tensor):
    return from_dlpack(
        tensor,
        assumed_align=16,
        use_32bit_stride=_fits_32bit_stride(tensor),
    )


def _dlpack_arg(tensor: torch.Tensor):
    for dim, stride in enumerate(tensor.stride()):
        if stride == 1:
            return _from_dlpack_arg(tensor).mark_layout_dynamic(dim)
    return _from_dlpack_arg(tensor).mark_layout_dynamic()


def _layout_key(tensor: torch.Tensor, dynamic_layout: bool = False):
    arg = _dlpack_arg(tensor) if dynamic_layout else _from_dlpack_arg(tensor)
    shape_mask = arg.dynamic_shapes_mask
    stride_mask = arg.dynamic_strides_mask
    shape = tuple(None if dynamic else size for size, dynamic in zip(tensor.shape, shape_mask))
    stride = tuple(
        None if dynamic else value for value, dynamic in zip(tensor.stride(), stride_mask)
    )
    return (tensor.dtype, shape, stride, _fits_32bit_stride(tensor))


# (device_index, enabled) -> persistent int32 [1] control tensor. Keys are
# plain values (not tensor identities), so entries never go stale.
_precompute_control_cache = {}


def _precompute_control_tensor(device: torch.device, enabled: bool) -> torch.Tensor:
    dev = torch.device(device)
    key = (dev.index if dev.index is not None else torch.cuda.current_device(), bool(enabled))
    if key not in _precompute_control_cache:
        _precompute_control_cache[key] = torch.tensor(
            [1 if enabled else 0], dtype=torch.int32, device=dev
        )
    return _precompute_control_cache[key]


def _try_flatten_args(
    *,
    recurrent_state: torch.Tensor,
    x_q: torch.Tensor,
    x_k: torch.Tensor,
    x_v: torch.Tensor,
    T_total: int,
    H: int,
    HV: int,
    K: int,
    V: int,
) -> Tuple[bool, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    try:
        h0 = recurrent_state.view(-1, V, K)
        x_q_flat = x_q.view(1, T_total, H * K)
        x_k_flat = x_k.view(1, T_total, H * K)
        x_v_flat = x_v.view(1, T_total, HV * V)
    except RuntimeError:
        return False, recurrent_state, x_q, x_k, x_v
    return True, h0, x_q_flat, x_k_flat, x_v_flat


def _is_benchmark_static_shape(
    N: int, H: int, HV: int, K: int, V: int, W: int, num_spec: int
) -> bool:
    return (
        K == 128
        and V == 128
        and W == 4
        and num_spec == 2
        and H == HV
        and N in (32, 128)
        and H in (2, 12, 32)
    )


# Layout-and-constexpr-keyed compile cache. Request count and packed-token
# length are dynamic; batches sharing the same kernel variant reuse one
# artifact even when their launch grid and token-buffer extents differ.
_compiled_cache = {}


def kda_mtp_decode_impl(
    x_q: torch.Tensor,
    x_k: torch.Tensor,
    x_v: torch.Tensor,
    w_q: torch.Tensor,
    w_k: torch.Tensor,
    w_v: torch.Tensor,
    cs_q: torch.Tensor,
    cs_k: torch.Tensor,
    cs_v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    recurrent_state: torch.Tensor,
    qkg_cache: torch.Tensor,
    v_cache: torch.Tensor,
    beta_cache: torch.Tensor,
    ssm_state_indices: torch.Tensor,
    cu_seqlens: torch.Tensor,
    num_spec: int,
    num_accepted_tokens: torch.Tensor,
    lower_bound: float,
    scale: Optional[float] = None,
    out: Optional[torch.Tensor] = None,
    zero_accepted_hint: bool = False,
    regular_metadata_hint: bool = False,
) -> torch.Tensor:
    """Launch the fused KDA MTP verify kernel. See the module docstring.

    Args (device tensors unless noted):
        x_q/x_k/x_v: post-projection, pre-conv token states
            ``[1, T_total, H, 128]`` bf16 — new tokens only, ``1 +
            num_spec`` per request, packed per ``cu_seqlens``.
        w_q/w_k/w_v: conv weights ``[H*128, W]`` fp32, width-contiguous.
        cs_q/cs_k/cs_v: extended conv caches ``[pool, H*128, >= W-1+M]``
            fp32, dim-contiguous. Columns ``[0, W-1)`` are the committed
            window; tail columns hold raw pending-draft inputs. Mutated.
        g, beta: raw gate ``[1, T, H, 128]`` and beta ``[1, T, H]`` bf16.
        A_log, dt_bias: fp32 ``[H]`` / ``[H*128]``.
        recurrent_state: pool ``[pool, H, V, K]`` fp32, **V-first** layout
            (matches the executor ssm pool and the single-token decode
            kernel). Committed in place.
        qkg_cache/v_cache/beta_cache: replay caches ``[pool, M, 3, H*K]`` /
            ``[pool, M, H*V]`` / ``[pool, M, H]`` fp32. Mutated.
        ssm_state_indices / cu_seqlens / num_accepted_tokens: per-request
            slot, token offsets ``[N+1]``, accepted-draft counts ``[N]``.
        zero_accepted_hint: caller asserts every ``num_accepted_tokens`` is
            zero (compiles the smaller-smem no-replay variant). Wrong hints
            produce wrong results — pass True only when statically known.
        regular_metadata_hint: caller asserts ``cu_seqlens`` is the uniform
            ``arange * (2*num_spec+1)`` pattern and ``ssm_state_indices``
            is ``arange(N)`` (benchmark identity layout).

    Returns the output ``[1, T_total, H, V]`` bf16 (rows for replayed
    positions are zero; only new-token rows are written).
    """
    _, T_total, _, D = x_q.shape
    H = A_log.shape[0]
    HV = g.shape[2] if g.ndim == 4 else H
    K = D
    V_dim = x_v.shape[-1]
    W = w_q.shape[1]
    if K != TILE_K or V_dim != 128 or W != 4:
        raise ValueError("specialized kernel expects K=128, V=128, W=4")
    if HV != H:
        raise ValueError("specialized kernel expects HV == H")
    if scale is None:
        scale = K**-0.5

    N = cu_seqlens.shape[0] - 1
    if out is None:
        out = torch.zeros(1, T_total, HV, V_dim, dtype=x_q.dtype, device=x_q.device)
    if num_accepted_tokens.dtype != torch.int32:
        num_accepted_tokens = num_accepted_tokens.to(torch.int32)

    _require_stride_layout(
        x_q=x_q,
        x_k=x_k,
        x_v=x_v,
        w_q=w_q,
        w_k=w_k,
        w_v=w_v,
        cs_q=cs_q,
        cs_k=cs_k,
        cs_v=cs_v,
        g=g,
        beta=beta,
        A_log=A_log,
        dt_bias=dt_bias,
        recurrent_state=recurrent_state,
        qkg_cache=qkg_cache,
        v_cache=v_cache,
        beta_cache=beta_cache,
        ssm_state_indices=ssm_state_indices,
        cu_seqlens=cu_seqlens,
        num_accepted_tokens=num_accepted_tokens,
        out=out,
        H=H,
        HV=HV,
        K=K,
        V=V_dim,
        W=W,
        num_spec=num_spec,
        T_total=T_total,
    )

    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    precompute_control = _precompute_control_tensor(x_q.device, True)

    use_flat_layout, h0_arg, x_q_arg, x_k_arg, x_v_arg = _try_flatten_args(
        recurrent_state=recurrent_state,
        x_q=x_q,
        x_k=x_k,
        x_v=x_v,
        T_total=T_total,
        H=H,
        HV=HV,
        K=K,
        V=V_dim,
    )
    pool_size = h0_arg.shape[0]
    is_benchmark_static_shape = _is_benchmark_static_shape(N, H, HV, K, V_dim, W, num_spec)
    use_setmaxreg = is_benchmark_static_shape
    use_reg_q_weights = is_benchmark_static_shape
    use_regular_metadata = bool(regular_metadata_hint)
    # The kernel's USE_ZERO_ACCEPTED fast path unrolls exactly
    # 1 + NUM_SPEC == 3 new tokens, so it is only valid for num_spec == 2.
    # For other num_spec fall back to the generic loop (the hint implies
    # num_accepted_tokens is all zeros, so the generic path computes the
    # same result).
    use_zero_accepted = bool(zero_accepted_hint) and num_spec == 2
    # stage_timing is unused (PROFILE_STAGES=False); pass `out` as the
    # placeholder tensor argument like the drop's runner does. The alias is
    # only valid while profiling stays off: with PROFILE_STAGES=True the
    # kernel writes int64 stage deltas through this tensor, corrupting the
    # bf16 output. Enabling profiling requires a dedicated int64 buffer of
    # at least HV * N * 4 elements.
    profile_stages = False
    assert not profile_stages, (
        "stage_timing aliases `out`; allocate a dedicated int64 [HV * N * 4] "
        "buffer before enabling PROFILE_STAGES"
    )
    stage_timing_arg = out

    key = (
        x_q.dtype,
        scale,
        HV,
        K,
        V_dim,
        num_spec,
        W,
        pool_size,
        lower_bound,
        use_flat_layout,
        _layout_key(h0_arg),
        _layout_key(x_q_arg, dynamic_layout=True),
        _layout_key(x_k_arg, dynamic_layout=True),
        _layout_key(x_v_arg, dynamic_layout=True),
        _layout_key(w_q),
        _layout_key(w_k),
        _layout_key(w_v),
        _layout_key(cs_q),
        _layout_key(cs_k),
        _layout_key(cs_v),
        _layout_key(A_log),
        _layout_key(g, dynamic_layout=True),
        _layout_key(dt_bias),
        _layout_key(beta, dynamic_layout=True),
        _layout_key(out, dynamic_layout=True),
        _layout_key(qkg_cache),
        _layout_key(v_cache),
        _layout_key(beta_cache),
        _layout_key(ssm_state_indices, dynamic_layout=True),
        _layout_key(cu_seqlens, dynamic_layout=True),
        _layout_key(num_accepted_tokens, dynamic_layout=True),
        use_setmaxreg,
        use_regular_metadata,
        use_reg_q_weights,
        use_zero_accepted,
    )

    if key not in _compiled_cache:
        logger.info(
            f"kda_mtp_decode: compiling variant N={N} H={HV} T={T_total} "
            f"num_spec={num_spec} zero_accepted={use_zero_accepted} "
            f"regular_metadata={use_regular_metadata} "
            f"static_shape={is_benchmark_static_shape}"
        )
        _compiled_cache[key] = cute.compile(
            _run_kda_decode_mtp,
            _from_dlpack_arg(h0_arg),
            _dlpack_arg(x_q_arg),
            _dlpack_arg(x_k_arg),
            _dlpack_arg(x_v_arg),
            _from_dlpack_arg(w_q),
            _from_dlpack_arg(w_k),
            _from_dlpack_arg(w_v),
            _from_dlpack_arg(cs_q),
            _from_dlpack_arg(cs_k),
            _from_dlpack_arg(cs_v),
            _from_dlpack_arg(A_log),
            _dlpack_arg(g),
            _from_dlpack_arg(dt_bias),
            _dlpack_arg(beta),
            _dlpack_arg(out),
            _from_dlpack_arg(h0_arg),
            _from_dlpack_arg(qkg_cache),
            _from_dlpack_arg(v_cache),
            _from_dlpack_arg(beta_cache),
            _dlpack_arg(stage_timing_arg),
            _dlpack_arg(ssm_state_indices),
            _dlpack_arg(cu_seqlens),
            _dlpack_arg(num_accepted_tokens),
            _from_dlpack_arg(precompute_control),
            scale=scale,
            HV=HV,
            K=K,
            V=V_dim,
            N=N,
            NUM_SPEC=num_spec,
            TILE_V=_TILE_V,
            KERNEL_WIDTH=W,
            lower_bound=lower_bound,
            USE_FLAT_LAYOUT=use_flat_layout,
            USE_SETMAXREG=use_setmaxreg,
            USE_REGULAR_METADATA=use_regular_metadata,
            USE_REG_Q_WEIGHTS=use_reg_q_weights,
            USE_ZERO_ACCEPTED=use_zero_accepted,
            FUSE_PRECOMPUTE=True,
            RUNTIME_PRECOMPUTE_FLAG=False,
            PROFILE_STAGES=profile_stages,
            stream=stream,
        )

    _compiled_cache[key](
        _dlpack_arg(h0_arg),
        _dlpack_arg(x_q_arg),
        _dlpack_arg(x_k_arg),
        _dlpack_arg(x_v_arg),
        _dlpack_arg(w_q),
        _dlpack_arg(w_k),
        _dlpack_arg(w_v),
        _dlpack_arg(cs_q),
        _dlpack_arg(cs_k),
        _dlpack_arg(cs_v),
        _dlpack_arg(A_log),
        _dlpack_arg(g),
        _dlpack_arg(dt_bias),
        _dlpack_arg(beta),
        _dlpack_arg(out),
        _dlpack_arg(h0_arg),
        _dlpack_arg(qkg_cache),
        _dlpack_arg(v_cache),
        _dlpack_arg(beta_cache),
        _dlpack_arg(stage_timing_arg),
        _dlpack_arg(ssm_state_indices),
        _dlpack_arg(cu_seqlens),
        _dlpack_arg(num_accepted_tokens),
        _dlpack_arg(precompute_control),
        N,
        stream,
    )

    return out


if IS_CUTLASS_DSL_AVAILABLE:

    @torch.library.custom_op(
        "trtllm::kda_mtp_decode",
        mutates_args=(
            "cs_q",
            "cs_k",
            "cs_v",
            "recurrent_state",
            "qkg_cache",
            "v_cache",
            "beta_cache",
        ),
        device_types="cuda",
    )
    def kda_mtp_decode(
        x_q: torch.Tensor,
        x_k: torch.Tensor,
        x_v: torch.Tensor,
        w_q: torch.Tensor,
        w_k: torch.Tensor,
        w_v: torch.Tensor,
        cs_q: torch.Tensor,
        cs_k: torch.Tensor,
        cs_v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        recurrent_state: torch.Tensor,
        qkg_cache: torch.Tensor,
        v_cache: torch.Tensor,
        beta_cache: torch.Tensor,
        ssm_state_indices: torch.Tensor,
        cu_seqlens: torch.Tensor,
        num_spec: int,
        num_accepted_tokens: torch.Tensor,
        lower_bound: float,
        scale: Optional[float] = None,
        zero_accepted_hint: bool = False,
        regular_metadata_hint: bool = False,
    ) -> torch.Tensor:
        """Fused KDA multi-token verify with in-place state commit."""
        return kda_mtp_decode_impl(
            x_q=x_q,
            x_k=x_k,
            x_v=x_v,
            w_q=w_q,
            w_k=w_k,
            w_v=w_v,
            cs_q=cs_q,
            cs_k=cs_k,
            cs_v=cs_v,
            g=g,
            beta=beta,
            A_log=A_log,
            dt_bias=dt_bias,
            recurrent_state=recurrent_state,
            qkg_cache=qkg_cache,
            v_cache=v_cache,
            beta_cache=beta_cache,
            ssm_state_indices=ssm_state_indices,
            cu_seqlens=cu_seqlens,
            num_spec=num_spec,
            num_accepted_tokens=num_accepted_tokens,
            lower_bound=lower_bound,
            scale=scale,
            zero_accepted_hint=zero_accepted_hint,
            regular_metadata_hint=regular_metadata_hint,
        )

    @kda_mtp_decode.register_fake
    def _(
        x_q: torch.Tensor,
        x_k: torch.Tensor,
        x_v: torch.Tensor,
        w_q: torch.Tensor,
        w_k: torch.Tensor,
        w_v: torch.Tensor,
        cs_q: torch.Tensor,
        cs_k: torch.Tensor,
        cs_v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        recurrent_state: torch.Tensor,
        qkg_cache: torch.Tensor,
        v_cache: torch.Tensor,
        beta_cache: torch.Tensor,
        ssm_state_indices: torch.Tensor,
        cu_seqlens: torch.Tensor,
        num_spec: int,
        num_accepted_tokens: torch.Tensor,
        lower_bound: float,
        scale: Optional[float] = None,
        zero_accepted_hint: bool = False,
        regular_metadata_hint: bool = False,
    ) -> torch.Tensor:
        return x_q.new_empty(x_v.shape)
