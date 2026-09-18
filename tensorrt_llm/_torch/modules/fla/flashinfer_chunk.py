# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""FlashInfer GDN prefill adapter for ``Qwen3NextGatedDeltaNet.forward_extend``.

Exposes ``chunk_gated_delta_rule`` with a signature call-compatible with the
vendored Triton ``tensorrt_llm._torch.modules.fla.chunk.chunk_gated_delta_rule``.

Differences vs the Triton path are absorbed inside this wrapper:
  * Layout: TRT-LLM ``[1, T, H, D]`` -> FlashInfer packed ``[T, H, D]``.
  * Forget-gate space: TRT-LLM's Triton ``chunk_gated_delta_rule`` consumes
    ``g`` in **log space** (``fla/chunk.py`` doc: "(forget) gating tensor (in
    log space!)"); FlashInfer's ``chunk_gated_delta_rule`` consumes the same
    quantity in **linear space** (default 1.0 = no decay; alpha = exp(log_g)).
    The wrapper converts ``g_linear = exp(g_log)`` before calling FlashInfer,
    unless the caller passes ``g_is_linear=True`` (``fused_gdn_post_conv``
    emits alpha directly for this path, saving the torch.exp launch).
  * Pre-L2-normalize Q/K when ``use_qk_l2norm_in_kernel=True``
    (the FlashInfer prefill kernel does NOT apply L2 norm internally; the
    ``use_qk_l2norm_in_kernel`` parameter on ``flashinfer.chunk_gated_delta_rule``
    is currently a dead arg, see ``flashinfer/gdn_prefill.py:317-356``).
  * SSM-state I/O. With ``inplace_indexed_state_update``, the caller pinning
    ``use_cp=False`` on SM100/SM103 and at most a handful of sequences, the
    pool and the slot indices go straight to FlashInfer's plain chunked
    kernel, which reads and writes the indexed rows itself (``state_indices``):
    no gather, no scatter, no packed scratch. Everything else (scans with many
    sequences -- the indexed variant costs ~1 us per sequence and a mixed
    iteration carries every decode request as a one-token sequence -- the
    chunk-parallel kernel, the fold tails that read slot S1 and write slot S2,
    the target-verify call site with a packed state) uses a
    pre-gather and post-scatter of the indexed SSM state into FlashInfer's
    packed ``[num_seqs, H, V, K]`` layout. TRT-LLM's GDN state pool uses the same
    ``[N, H, V, K]`` logical layout, so the adapter gathers/scatters without
    transposing the last two dims. The SM100/SM103 kernel carries the recurrent
    state in fp32 in TMEM regardless of the initial/output-state I/O dtype, so
    the round-trip stays in the native pool dtype (bf16/fp16) there with no
    precision change; only SM90/SM120 need an fp32 up-cast/down-cast.

This module is only imported when ``TLLM_USE_FLASHINFER_GDN_PREFILL=1`` is set
at process start; do not import it lazily inside hot paths.
"""

import functools
import os
from typing import Literal, Optional, Tuple, Union

import torch

from tensorrt_llm._torch.modules.fla.fused_state_io import (
    cast_scatter_fp32_vk_to_vk,
    gather_cast_vk_to_fp32_vk,
)
from tensorrt_llm._torch.modules.fla.l2norm import l2norm_fwd
from tensorrt_llm._utils import is_sm_100f

# Indexed pool I/O on the plain SM100/SM103 kernel (see the module docstring).
# Escape hatch for A/B runs; measured on VR200 / flashinfer 0.6.18 as host
# 20 us vs 66 us per call and GPU time <= the gather/scatter path at every
# length (results/_analysis_scratch/gdn_prefill_cp_vs_plain_bench.py).
_INDEXED_STATE_IO = os.environ.get("TLLM_GDN_PREFILL_INDEXED_STATE_IO", "1") == "1"
# The kernel's indexed state loads/stores cost ~1 us per sequence on top of the
# scan: with the one-token decode sequences that used to ride along in a mixed
# iteration (100-250 of them) the indexed variant was up to 2x slower than the
# wrapper's gather + scatter (192 decode + one 1024-token chunk: 330 vs 160 us;
# results/_analysis_scratch/gdn_prefill_mixedbatch_bench.txt). Since the decode
# rows take the recurrent kernel, the main scan carries only the context
# segments (two per folded request): up to 32 of them the indexed launch is
# cheaper than gather + scan + scatter both on the GPU (35-42 vs 52 us for 1-24
# segments of 32-2048 tokens, fi_indexed_vs_compact_gpu_bench.py) and on the
# host (two launches fewer per layer, the mixed iterations' critical path).
_INDEXED_STATE_IO_MAX_SEQS = int(os.environ.get("TLLM_GDN_PREFILL_INDEXED_STATE_IO_MAX_SEQS", "32"))


# The indexed single-launch path calls FlashInfer's Blackwell launcher directly
# instead of the public ``flashinfer.chunk_gated_delta_rule``: the public wrapper
# re-validates every argument on every call (~3 us) on top of the launcher (~8 us
# of host time), and this adapter had already checked what the kernel needs.
# Measured on the dev node: adapter 26 us -> ~12 us per fold-tail scan. The
# launcher is an internal FlashInfer symbol; when it is missing (other FlashInfer
# versions) or TLLM_GDN_FI_DIRECT_LAUNCH=0 the public wrapper is used.
_FI_DIRECT_LAUNCH = os.environ.get("TLLM_GDN_FI_DIRECT_LAUNCH", "1") == "1"

# FlashInfer's SM100 CuTe DSL kernel takes int32 cu_seqlens while the Mamba
# metadata hands the GDN mixer int64 ("_long") views of per-iteration buffers.
# Casting at every direct launch cost one eager copy kernel per GDN layer per
# prefill iteration (45 launches, ~2 ms of host time on a 397B mixed iteration).
# The metadata rewrites those buffers exactly once per iteration, in
# Mamba2Metadata.prepare, which invalidates this cache first; within the
# iteration a cast keyed by (storage, length, device) is therefore valid and
# shared by every layer. (Inference-mode tensors carry no version counter, so
# the invalidation is explicit rather than version-keyed.)
_INT32_CU_SEQLENS_CACHE_MAX = 8
_int32_cu_seqlens_cache: dict = {}


def invalidate_int32_cu_seqlens_cache() -> None:
    """Drop the per-iteration int32 casts (called when the metadata rewrites
    its cu_seqlens buffers)."""
    _int32_cu_seqlens_cache.clear()


def _int32_cu_seqlens(cu_seqlens: torch.Tensor) -> torch.Tensor:
    """``cu_seqlens`` as int32, cast at most once per iteration per buffer."""
    if cu_seqlens.dtype == torch.int32:
        return cu_seqlens
    key = (cu_seqlens.data_ptr(), cu_seqlens.numel(), cu_seqlens.device.index)
    cached = _int32_cu_seqlens_cache.get(key)
    if cached is None:
        if len(_int32_cu_seqlens_cache) >= _INT32_CU_SEQLENS_CACHE_MAX:
            _int32_cu_seqlens_cache.clear()
        # The entry keeps the source tensor alive: a freed int64 buffer could
        # otherwise be recycled by a new tensor of the same length (same
        # data_ptr, version 0) and the key would hand out the stale cast.
        cached = (cu_seqlens, cu_seqlens.to(torch.int32))
        _int32_cu_seqlens_cache[key] = cached
    return cached[1]


@functools.lru_cache(maxsize=1)
def _sm100_direct_launcher():
    """``chunk_gated_delta_rule_sm100(q, k, v, gate, beta, output, cu_seqlens,
    initial_state, output_state, scale, ..., state_indices=)`` or None."""
    if not _FI_DIRECT_LAUNCH:
        return None
    try:
        from flashinfer.gdn_kernels.blackwell.gdn_prefill import chunk_gated_delta_rule_sm100
    except ImportError:
        return None
    return chunk_gated_delta_rule_sm100


def indexed_state_io_available(num_seqs: int) -> bool:
    """Whether a scan of ``num_seqs`` sequences pinned to the plain kernel
    (``use_cp=False``, no ``output_state_indices``) takes the single-launch
    indexed pool I/O path below instead of gather + kernel + scatter."""
    return _INDEXED_STATE_IO and num_seqs <= _INDEXED_STATE_IO_MAX_SEQS and is_sm_100f()


def chunk_gated_delta_rule_indexed_direct(
    q3: torch.Tensor,
    k3: torch.Tensor,
    v3: torch.Tensor,
    g2: torch.Tensor,
    beta2: torch.Tensor,
    out3: torch.Tensor,
    cu_seqlens: torch.Tensor,
    pool: torch.Tensor,
    state_indices: torch.Tensor,
    scale: Optional[float] = None,
) -> bool:
    """The indexed single-launch path of ``chunk_gated_delta_rule`` for callers
    holding FlashInfer's own layouts: packed contiguous ``[T, H, D]`` q/k/v, the
    ``[T, HV]`` fp32 linear-space gate and fp32 beta, a ``[T, HV, V]`` output view
    and int32/int64 pool slots (read and written in place). No layout, dtype or
    gate-space work and no re-validation, just the launch: ~12 us of host time
    against ~23 for the generic entry on the same call. Returns False without
    launching when the direct launcher is unavailable or the sequence count is
    above the indexed threshold; the caller then uses the generic entry."""
    launcher = _sm100_direct_launcher() if _INDEXED_STATE_IO else None
    if (
        launcher is None
        or cu_seqlens.shape[0] - 1 > _INDEXED_STATE_IO_MAX_SEQS
        or not is_sm_100f()
    ):
        return False
    # FlashInfer's SM100 CuTe DSL kernel takes int32 cu_seqlens; the public
    # wrapper casts, so the direct launch has to as well (once per iteration,
    # see _int32_cu_seqlens).
    launcher(
        q3,
        k3,
        v3,
        g2,
        beta2,
        out3,
        _int32_cu_seqlens(cu_seqlens),
        pool,
        pool,
        scale if scale else q3.shape[2] ** -0.5,
        state_indices=state_indices,
    )
    return True


# Mirror the @torch.compiler.disable on the legacy Triton wrapper
# (chunk.py:119): Dynamo must not trace this wrapper because it imports
# `flashinfer` lazily and calls into FI's CuTe-DSL kernels, neither of
# which compiles cleanly.
@torch.compiler.disable
def chunk_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: Optional[float] = None,
    initial_state: Optional[torch.Tensor] = None,
    initial_state_indices: Optional[torch.Tensor] = None,
    inplace_indexed_state_update: bool = False,
    output_final_state: bool = False,
    cu_seqlens: Optional[torch.Tensor] = None,
    head_first: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    output: Optional[torch.Tensor] = None,
    state_workspace: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    output_state_indices: Optional[torch.Tensor] = None,
    use_cp: Union[Literal["auto"], bool] = "auto",
    g_is_linear: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Adapter for FlashInfer's chunk_gated_delta_rule.

    ``state_workspace`` optionally supplies an (input, output) tensor pair,
    each shaped ``[capacity, H, V, K]`` in the state I/O dtype.
    Returned states never alias this scratch.

    ``output_state_indices`` (only with ``inplace_indexed_state_update``) makes
    the final states land in different pool slots than the ones the initial
    states were gathered from. The folded save-last prefill uses it to run the
    <= tokens_per_block tail of a context chunk from the snapshot slot at the
    reachable point into the terminal slot without a separate iteration.

    ``g_is_linear`` declares that ``g`` already holds ``alpha = exp(g_log)``
    (fp32), as emitted by ``fused_gdn_post_conv(g_linear=True)``; the wrapper
    then skips its own ``torch.exp``.
    """
    # FlashInfer is imported lazily so importing this module on a non-FlashInfer
    # build does not error until the function is actually called.
    import flashinfer

    # --- Step 1: pre-flight asserts --------------------------------------
    assert head_first is False, "head_first=True is not supported by this wrapper"
    assert q.dim() == 4 and q.shape[0] == 1, f"q must be [1, T, H_q, D_k], got {tuple(q.shape)}"
    assert k.shape[2] == q.shape[2], (
        f"num_q_heads ({q.shape[2]}) must equal num_k_heads ({k.shape[2]})"
    )
    assert q.dtype in (torch.bfloat16, torch.float16), f"q dtype must be bf16/fp16, got {q.dtype}"
    assert g.dtype == torch.float32, f"g must be fp32, got {g.dtype}"
    assert cu_seqlens is not None, "cu_seqlens is required (varlen mode)"
    assert initial_state is not None, "initial_state is required"
    if inplace_indexed_state_update:
        assert initial_state_indices is not None, (
            "inplace_indexed_state_update=True requires initial_state_indices"
        )
    else:
        assert output_state_indices is None, (
            "output_state_indices requires inplace_indexed_state_update=True"
        )

    # --- Step 2: layout [1, T, H, D] -> [T, H, D] ------------------------
    # q/k/v are slices of mixed_qkv produced by torch.split on the last dim,
    # so the squeeze(0) view is *non-contiguous*. The .contiguous() here pays
    # a real copy and is required (FlashInfer reads contiguous TMA tiles).
    q3 = q.squeeze(0).contiguous()
    k3 = k.squeeze(0).contiguous()
    v3 = v.squeeze(0).contiguous()
    # Convert g from Triton's log-space convention to FlashInfer's linear-space
    # alpha (default 1.0 = no decay). FlashInfer's prefill kernel multiplies the
    # SSM state by ``g`` directly each chunk; passing log-space values produces
    # NaN. ``torch.exp`` always returns a fresh contiguous tensor, so no
    # explicit .contiguous() is needed.
    if g_is_linear:
        # Already alpha in fp32 from the producer; a [1, T, H] tensor (or a
        # token-range slice of one) squeezes to a contiguous [T, H] view.
        g2 = g.squeeze(0)
        if not g2.is_contiguous():
            g2 = g2.contiguous()
    else:
        g2 = torch.exp(g.squeeze(0))
    # ``.to(torch.float32)`` allocates a fresh contiguous fp32 tensor when the
    # source is bf16 (the common case), or returns the (already contiguous)
    # input view when beta is already fp32 — no .contiguous() needed.
    beta2 = beta.squeeze(0).to(torch.float32)

    # --- Step 3: emulate use_qk_l2norm_in_kernel -------------------------
    # Use the fused Triton l2norm_fwd (eps=1e-6 matches the Triton/FI decode
    # kernels) instead of F.normalize, which incurs extra kernel launches and
    # an intermediate buffer at short ISL.
    if use_qk_l2norm_in_kernel:
        q3 = l2norm_fwd(q3)
        k3 = l2norm_fwd(k3)

    # --- Output buffer (shared by both state-I/O paths) ------------------
    # FI 0.6.10 accepts `output=`; pre-allocating skips its internal
    # `torch.empty` per call. `num_o_heads` per FI docstring is
    # max(num_q_heads, num_v_heads) — equivalently num_v_heads for GVA.
    total_seq_len = q3.shape[0]
    num_o_heads = max(q3.shape[1], v3.shape[1])
    head_size = q3.shape[2]
    output_buf = (
        output.squeeze(0)
        if output is not None
        else q3.new_empty(total_seq_len, num_o_heads, head_size)
    )

    # --- Step 4a: indexed pool I/O on the plain SM100/SM103 kernel --------
    # One launch instead of gather + kernel + scatter; the pool rows never
    # leave the pool. Only with the plain kernel pinned by the caller
    # (``use_cp=False``) and a handful of sequences: the indexed variant pays
    # ~1 us per sequence, which is what made the reverted main pick fbac78bd2a
    # ~2-3x slower per launch in mixed iterations (their scans carry every
    # decode request as a one-token sequence). The fold tails
    # (``output_state_indices``) read slot S1 and write slot S2, which
    # FlashInfer's single index cannot express.
    if (
        _INDEXED_STATE_IO
        and inplace_indexed_state_update
        and output_state_indices is None
        and use_cp is False
        and cu_seqlens.shape[0] - 1 <= _INDEXED_STATE_IO_MAX_SEQS
        and is_sm_100f()
    ):
        assert initial_state_indices.dtype in (torch.int32, torch.int64), (
            f"state_indices must be int32/int64, got {initial_state_indices.dtype}"
        )
        launcher = _sm100_direct_launcher()
        if launcher is not None:
            # Same call the public wrapper makes after its checks (scale
            # defaulting included); the pool is both the initial and the output
            # state, indexed by ``initial_state_indices``.
            launcher(
                q3,
                k3,
                v3,
                g2,
                beta2,
                output_buf,
                _int32_cu_seqlens(cu_seqlens),
                initial_state,
                initial_state,
                scale if scale else head_size**-0.5,
                state_indices=initial_state_indices,
            )
            return output_buf.unsqueeze(0), None
        flashinfer.chunk_gated_delta_rule(
            q=q3,
            k=k3,
            v=v3,
            g=g2,
            beta=beta2,
            scale=scale,
            initial_state=initial_state,  # the pool [N_pool, H, V, K]
            state_indices=initial_state_indices,  # [num_seqs]: read and write these rows
            output_final_state=True,
            output_state=initial_state,  # in-place update of the indexed rows
            cu_seqlens=cu_seqlens,
            use_qk_l2norm_in_kernel=False,
            output=output_buf,
            use_cp=False,
        )
        return output_buf.unsqueeze(0), None

    # --- Step 4: gather initial state (+ cast dtype only when required) ---
    # TRT-LLM's GDN kernels and FlashInfer both use [N, H, V, K] state layout.
    # The SM100/SM103 kernel carries the recurrent state in fp32 in TMEM
    # regardless of the initial/output-state I/O dtype (the state tensors are
    # only the gmem load/store format), so passing bf16/fp16 state there is
    # numerically identical to the fp32 round-trip while moving half the bytes.
    # SM90/SM120 still require fp32 state. Fuse gather (+ optional cast) and
    # contiguous into a single Triton kernel.
    state_dtype = initial_state.dtype if is_sm_100f() else torch.float32
    num_seqs = cu_seqlens.shape[0] - 1
    if state_workspace is not None:
        assert len(state_workspace) == 2
        for buffer in state_workspace:
            assert buffer.shape[0] >= num_seqs
            assert buffer.shape[1:] == initial_state.shape[1:]
            assert buffer.dtype == state_dtype
            assert buffer.device == initial_state.device
    gathered_init = gather_cast_vk_to_fp32_vk(
        initial_state,
        initial_state_indices,
        out_dtype=state_dtype,
        output=state_workspace[0][:num_seqs] if state_workspace is not None else None,
    )

    # --- Step 5+6: call FlashInfer with pre-allocated output/state buffers
    # `output_state=` likewise skips FI's internal allocation. Only allocate /
    # request final state when a caller actually consumes it (either inplace
    # scatter back to the SSM pool, or return to the caller); otherwise FI
    # skips the final-state aggregation entirely.
    need_state = inplace_indexed_state_update or output_final_state
    if need_state:
        num_seqs = cu_seqlens.shape[0] - 1
        # Match the initial-state dtype (native bf16/fp16 on SM100/SM103, else
        # fp32); FlashInfer writes the final state in this dtype and the scatter
        # below adapts to the destination pool dtype without an extra cast.
        state_buf = (
            state_workspace[1][:num_seqs]
            if state_workspace is not None
            else q3.new_empty(num_seqs, num_o_heads, head_size, head_size, dtype=state_dtype)
        )
        out_packed, out_state = flashinfer.chunk_gated_delta_rule(
            q=q3,
            k=k3,
            v=v3,
            g=g2,
            beta=beta2,
            scale=scale,
            initial_state=gathered_init,
            output_final_state=True,
            cu_seqlens=cu_seqlens,
            use_qk_l2norm_in_kernel=False,  # dead param in FlashInfer; we already normalized
            use_cp=use_cp,
            output=output_buf,
            output_state=state_buf,
        )
    else:
        # FI returns a single tensor (not a tuple) when output_final_state=False.
        out_packed = flashinfer.chunk_gated_delta_rule(
            q=q3,
            k=k3,
            v=v3,
            g=g2,
            beta=beta2,
            scale=scale,
            initial_state=gathered_init,
            output_final_state=False,
            cu_seqlens=cu_seqlens,
            use_qk_l2norm_in_kernel=False,
            use_cp=use_cp,
            output=output_buf,
        )
        out_state = None

    # --- Step 7: cast state back (if needed), scatter / return ---------
    # Fuse cast (out_state.dtype -> destination dtype; a no-op on SM100/SM103
    # where both are the native pool dtype) + optional indexed scatter into a
    # single Triton pass, mirroring Step 4. The inplace branch writes only the
    # slots named by ``initial_state_indices`` and leaves the rest untouched.
    if inplace_indexed_state_update:
        scatter_indices = (
            output_state_indices if output_state_indices is not None else initial_state_indices
        )
        cast_scatter_fp32_vk_to_vk(out_state, initial_state, scatter_indices)
        final_to_return: Optional[torch.Tensor] = None
    elif output_final_state:
        num_seqs_out, num_h_out, v_out, k_out = out_state.shape
        final_to_return = torch.empty(
            num_seqs_out,
            num_h_out,
            v_out,
            k_out,
            dtype=initial_state.dtype,
            device=out_state.device,
        )
        cast_scatter_fp32_vk_to_vk(out_state, final_to_return, None)
    else:
        final_to_return = None

    # --- Step 8: restore output layout ----------------------------------
    out = out_packed.unsqueeze(0)

    # --- Step 9: return -------------------------------------------------
    return out, final_to_return
