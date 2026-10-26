# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Decode-sized mHC boundary: the critical half as one CuTe DSL launch.

See ``cute_dsl_kernels/blackwell/mhc/mhc_split_boundary.py``. ``mhc_post_collapse_norm`` computes a
lagged boundary's new residual stream and the normalized input of the next sublayer; the
boundary's coefficients (``pre`` / ``post`` / ``comb`` for the *next* boundary) are computed
separately from the new residual and can run off the critical path.

The split boundary runs for at most 64 tokens. ``mhc_allreduce_post_collapse_norm`` also
all-reduces the previous sublayer's tensor-parallel partial output in the same launch: over the
IPC workspace when the group is inside one tray, or over the MNNVL fabric-memory workspace when it
spans trays.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass

import torch

from tensorrt_llm.math_utils import pad_up

from ...cute_dsl_utils import graph_safe_from_dlpack
from ...flashinfer_utils import get_env_enable_pdl


@functools.lru_cache(maxsize=None)
def _device_supported() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability() in ((10, 0), (10, 3))


def mhc_split_enabled(num_tokens: int, hidden_size: int, mult: int) -> bool:
    """Whether a boundary of ``num_tokens`` rows can take the split (critical + coefficient) path."""
    if not _device_supported():
        return False
    if mult != 4 or hidden_size % 256 or hidden_size // 8 > 1024:
        return False
    return 0 < num_tokens <= 64


@functools.lru_cache(maxsize=None)
def _compiled_post_collapse_norm(
    m: int, hidden: int, norm_eps: float, post_update: bool, use_pdl: bool, quant: str | None = None
):
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute

    from ...cute_dsl_kernels.blackwell.mhc.mhc_split_boundary import MhcPostCollapseNormKernel

    kernel = MhcPostCollapseNormKernel(
        hidden=hidden, norm_eps=norm_eps, use_pdl=use_pdl, post_update=post_update, quant=quant
    )
    dev = torch.device("cuda")
    x = torch.empty((m, hidden), dtype=torch.bfloat16, device=dev)
    res = torch.empty((m, 4 * hidden), dtype=torch.bfloat16, device=dev)
    post = torch.empty((m, 4), dtype=torch.float32, device=dev)
    comb = torch.empty((m, 16), dtype=torch.float32, device=dev)
    pre = torch.empty((m, 4), dtype=torch.float32, device=dev)
    w = torch.empty((1, hidden), dtype=torch.bfloat16, device=dev)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    return cute.compile(
        kernel,
        graph_safe_from_dlpack(x),
        graph_safe_from_dlpack(res),
        graph_safe_from_dlpack(post),
        graph_safe_from_dlpack(comb),
        graph_safe_from_dlpack(pre),
        graph_safe_from_dlpack(w),
        graph_safe_from_dlpack(res),
        graph_safe_from_dlpack(x),
        cutlass.Int64(0),
        cutlass.Int64(0),
        cutlass.Int64(0),
        stream,
    )


def _quantized_outputs(
    m: int, hidden: int, device: torch.device, mxfp8: bool, nvfp4_scale: torch.Tensor | None
) -> tuple[str | None, tuple[torch.Tensor, torch.Tensor] | None, int]:
    """Kernel quantization mode, its outputs shaped as the native quantizers return them
    (``torch.ops.trtllm.mxfp8_quantize(x, True)`` / ``fp4_quantize(x, scale, 16, False, False)``),
    and the global-scale address."""
    if mxfp8:
        return (
            "mxfp8",
            (
                torch.empty((m, hidden), dtype=torch.float8_e4m3fn, device=device),
                torch.empty((pad_up(m, 128) * hidden // 32,), dtype=torch.uint8, device=device),
            ),
            0,
        )
    if nvfp4_scale is not None:
        assert nvfp4_scale.dtype == torch.float32 and nvfp4_scale.numel() == 1
        return (
            "nvfp4",
            (
                torch.empty((m, hidden // 2), dtype=torch.uint8, device=device),
                torch.empty((m * hidden // 16,), dtype=torch.uint8, device=device),
            ),
            nvfp4_scale.data_ptr(),
        )
    return None, None, 0


class LamportAllReduce:
    """The one-shot Lamport all-reduce workspace of a TP group (``get_allreduce_workspace``).

    ``table`` is the slice ``[2N, 3N + 2)`` of the workspace pointer tensor: the Lamport buffer of
    every rank, then the flag and layout words that the library's one-shot kernels advance.
    """

    mnnvl = False

    def __init__(self, workspace: torch.Tensor, rank: int, nranks: int):
        if workspace.dtype != torch.int64 or workspace.numel() != 3 * nranks + 2:
            raise ValueError("unexpected all-reduce workspace layout")
        self.table = workspace[2 * nranks :].contiguous()
        self.rank = rank
        self.nranks = nranks

    def prepare(self, nbytes: int) -> torch.Tensor:
        """The kernel's workspace table for a launch writing ``nbytes`` per rank buffer."""
        return self.table


class MnnvlLamportAllReduce:
    """The MNNVL (fabric memory) Lamport workspace of a TP group spanning trays.

    It is ``MNNVLAllReduce``'s workspace for the group, so this kernel and the library's MNNVL
    all-reduce kernels advance the same triple buffer and flags. ``table`` holds the local
    buffer, the multicast buffer and the flags; the kernel reads it at run time, so it stays
    valid when the workspace is re-created at a larger size (between CUDA graph captures).
    """

    mnnvl = True

    def __init__(self, mapping):
        self.mapping = mapping
        self.rank = mapping.tp_rank
        self.nranks = mapping.tp_size
        self.table = None
        self._workspace = None
        self._views = None
        self.prepare(0)

    def prepare(self, nbytes: int) -> torch.Tensor:
        """The table for a launch writing ``nbytes`` per rank buffer (grows the workspace)."""
        from ...distributed.ops import get_or_scale_allreduce_mnnvl_workspace

        workspace = get_or_scale_allreduce_mnnvl_workspace(
            self.mapping, torch.bfloat16, nbytes or None
        )
        if workspace is not self._workspace:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("the MNNVL workspace changed during CUDA graph capture")
            local = workspace["uc_buffer"]
            mcast = workspace["handle"].get_mc_buffer(tuple(local.shape), local.dtype, 0)
            flags = workspace["buffer_flags"]
            if self.table is None:
                self.table = torch.zeros((3,), dtype=torch.int64, device=flags.device)
            self.table.copy_(
                torch.tensor(
                    [local.data_ptr(), mcast.data_ptr(), flags.data_ptr()], dtype=torch.int64
                )
            )
            self._workspace = workspace
            self._views = (local, mcast, flags)
        return self.table


def mhc_lamport_allreduce(mapping, strategy) -> "LamportAllReduce | MnnvlLamportAllReduce | None":
    """The workspace for split boundaries that all-reduce their sublayer output themselves.

    A tensor-parallel group with peer access gets the IPC Lamport workspace of the one-shot
    all-reduce (AUTO / ONESHOT strategies); a group spanning trays gets the MNNVL workspace
    (AUTO / MNNVL, where MNNVL all-reduce is available). None otherwise.
    """
    from tensorrt_llm._ipc_utils import can_access_peer

    from ...distributed.ops import AllReduceStrategy, MNNVLAllReduce, get_allreduce_workspace

    if not _device_supported():
        return None
    if mapping.enable_attention_dp or mapping.tp_size not in (2, 4, 8):
        return None
    if mapping.is_multi_node():
        mnnvl = strategy in (AllReduceStrategy.AUTO, AllReduceStrategy.MNNVL)
        if mnnvl and MNNVLAllReduce.is_mnnvl(mapping, torch.bfloat16):
            return MnnvlLamportAllReduce(mapping)
        return None
    if strategy not in (AllReduceStrategy.AUTO, AllReduceStrategy.ONESHOT):
        return None
    if not can_access_peer(mapping):
        return None
    return LamportAllReduce(get_allreduce_workspace(mapping), mapping.tp_rank, mapping.tp_size)


@dataclass
class MoeUnfinalized:
    """A TP rank's MoE output before the top-k combine (``do_finalize=False``).

    ``fc2`` [P, H] holds the permuted expert rows, ``perm`` [M, top_k] int32 maps each
    (token, k) to its row (-1: not on this rank), ``scales`` [M, top_k] are the routing weights
    and ``shared`` [M, H] is the shared-expert output.
    """

    shared: torch.Tensor
    fc2: torch.Tensor
    scales: torch.Tensor
    perm: torch.Tensor

    @property
    def shape(self):
        return self.shared.shape

    def finalize(self) -> torch.Tensor:
        """The rank's partial MoE output (FP32 combine, one BF16 rounding)."""
        perm = self.perm.long()
        valid = (perm >= 0).unsqueeze(-1)
        rows = self.fc2[perm.clamp(min=0)].float() * valid
        routed = (rows * self.scales.float().unsqueeze(-1)).sum(1)
        return (routed + self.shared.float()).to(self.shared.dtype)


@functools.lru_cache(maxsize=None)
def _placeholder_moe_operands(hidden: int, device_index: int):
    dev = torch.device("cuda", device_index)
    return (
        torch.zeros((1, hidden), dtype=torch.bfloat16, device=dev),
        torch.zeros((1, 1), dtype=torch.float32, device=dev),
        torch.full((1, 1), -1, dtype=torch.int32, device=dev),
        torch.zeros((1, hidden), dtype=torch.bfloat16, device=dev),
    )


@functools.lru_cache(maxsize=None)
def _compiled_allreduce_boundary(
    m: int,
    hidden: int,
    norm_eps: float,
    nranks: int,
    use_pdl: bool,
    moe: tuple | None = None,
    ctas_per_token: int = 1,
    mnnvl: bool = False,
    quant: str | None = None,
):
    """``moe`` = (fc2 rows, top_k, scale dtype) for the MoE-finalize input, None otherwise."""
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute

    from ...cute_dsl_kernels.blackwell.mhc.mhc_split_boundary import MhcAllReduceBoundaryKernel

    kernel = MhcAllReduceBoundaryKernel(
        hidden=hidden,
        norm_eps=norm_eps,
        nranks=nranks,
        use_pdl=use_pdl,
        moe_topk=moe[1] if moe else 0,
        ctas_per_token=ctas_per_token,
        mnnvl=mnnvl,
        quant=quant,
    )
    dev = torch.device("cuda")
    x = torch.empty((m, hidden), dtype=torch.bfloat16, device=dev)
    res = torch.empty((m, 4 * hidden), dtype=torch.bfloat16, device=dev)
    post = torch.empty((m, 4), dtype=torch.float32, device=dev)
    comb = torch.empty((m, 16), dtype=torch.float32, device=dev)
    pre = torch.empty((m, 4), dtype=torch.float32, device=dev)
    w = torch.empty((1, hidden), dtype=torch.bfloat16, device=dev)
    table = torch.empty((3 if mnnvl else nranks + 2,), dtype=torch.int64, device=dev)
    if moe:
        rows, top_k, scale_dtype = moe
        fc2 = torch.empty((rows, hidden), dtype=torch.bfloat16, device=dev)
        scales = torch.empty((m, top_k), dtype=scale_dtype, device=dev)
        perm = torch.empty((m, top_k), dtype=torch.int32, device=dev)
        shared = x
    else:
        fc2, scales, perm, shared = _placeholder_moe_operands(hidden, torch.cuda.current_device())
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    return cute.compile(
        kernel,
        graph_safe_from_dlpack(x),
        graph_safe_from_dlpack(res),
        graph_safe_from_dlpack(post),
        graph_safe_from_dlpack(comb),
        graph_safe_from_dlpack(pre),
        graph_safe_from_dlpack(w),
        graph_safe_from_dlpack(res),
        graph_safe_from_dlpack(x),
        graph_safe_from_dlpack(table),
        cutlass.Int32(0),
        graph_safe_from_dlpack(fc2),
        graph_safe_from_dlpack(scales),
        graph_safe_from_dlpack(perm),
        graph_safe_from_dlpack(shared),
        cutlass.Int64(0),
        cutlass.Int64(0),
        cutlass.Int64(0),
        stream,
    )


def mhc_allreduce_post_collapse_norm(
    x_partial: "torch.Tensor | MoeUnfinalized",
    residual_prev: torch.Tensor,
    post_mix_prev: torch.Tensor,
    comb_mix_prev: torch.Tensor,
    pre_mix: torch.Tensor,
    norm_weight: torch.Tensor,
    norm_eps: float,
    allreduce: "LamportAllReduce | MnnvlLamportAllReduce",
    mxfp8: bool = False,
    nvfp4_scale: torch.Tensor | None = None,
) -> tuple[torch.Tensor, ...]:
    """:func:`mhc_post_collapse_norm` of the TP sum of ``x_partial`` (one-shot all-reduce in-kernel).

    The sum is the library one-shot all-reduce's (FP32 accumulation in rank order, BF16 result),
    so the outputs equal ``mhc_post_collapse_norm(allreduce(x_partial), ...)`` bit for bit. A
    :class:`MoeUnfinalized` partial is finalized in the same launch (``x_partial.finalize()``,
    up to the FP32 summation order). ``mxfp8`` / ``nvfp4_scale`` add a quantized copy of the
    normalized output as :func:`mhc_post_collapse_norm` does.
    """
    import cuda.bindings.driver as cuda
    import cutlass

    m, mult, hidden = residual_prev.shape
    assert mult == 4 and residual_prev.dtype == torch.bfloat16
    res_flat = residual_prev.reshape(m, mult * hidden).contiguous()
    moe_key = None
    if isinstance(x_partial, MoeUnfinalized):
        fc2 = x_partial.fc2
        scales = x_partial.scales
        perm = x_partial.perm
        shared = x_partial.shared
        assert fc2.dtype == torch.bfloat16 and fc2.shape[-1] == hidden and fc2.is_contiguous()
        assert perm.dtype == torch.int32 and perm.shape[0] == m and perm.is_contiguous()
        assert scales.shape == perm.shape and scales.is_contiguous()
        assert shared.dtype == torch.bfloat16 and shared.is_contiguous()
        moe_key = (fc2.shape[0], perm.shape[1], scales.dtype)
        x = shared.view(m, hidden)
    else:
        x = x_partial.reshape(m, hidden).to(torch.bfloat16).contiguous()
        fc2, scales, perm, shared = _placeholder_moe_operands(hidden, x.device.index)
    pre = pre_mix.reshape(m, mult).to(torch.float32).contiguous()
    post = post_mix_prev.reshape(m, mult).to(torch.float32).contiguous()
    comb = comb_mix_prev.reshape(m, mult * mult).to(torch.float32).contiguous()
    weight = norm_weight.to(torch.bfloat16).contiguous().view(1, hidden)
    out = torch.empty((m, hidden), dtype=torch.bfloat16, device=residual_prev.device)
    res_out = torch.empty_like(res_flat)
    quant, quantized, scale_addr = _quantized_outputs(m, hidden, out.device, mxfp8, nvfp4_scale)
    table = allreduce.prepare(allreduce.nranks * m * hidden * res_flat.element_size())
    fn = _compiled_allreduce_boundary(
        m,
        hidden,
        float(norm_eps),
        allreduce.nranks,
        get_env_enable_pdl(),
        moe_key,
        # A cluster of 4 CTAs per token for the smallest batches.
        4 if m <= 12 else 1,
        mnnvl=allreduce.mnnvl,
        quant=quant,
    )
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    fn(
        graph_safe_from_dlpack(x),
        graph_safe_from_dlpack(res_flat),
        graph_safe_from_dlpack(post),
        graph_safe_from_dlpack(comb),
        graph_safe_from_dlpack(pre),
        graph_safe_from_dlpack(weight),
        graph_safe_from_dlpack(res_out),
        graph_safe_from_dlpack(out),
        graph_safe_from_dlpack(table),
        cutlass.Int32(allreduce.rank),
        graph_safe_from_dlpack(fc2),
        graph_safe_from_dlpack(scales),
        graph_safe_from_dlpack(perm),
        graph_safe_from_dlpack(shared),
        cutlass.Int64(quantized[0].data_ptr() if quant else 0),
        cutlass.Int64(quantized[1].data_ptr() if quant else 0),
        cutlass.Int64(scale_addr),
        stream,
    )
    if quant:
        return res_out.view(m, mult, hidden), out, quantized
    return res_out.view(m, mult, hidden), out


def mhc_post_collapse_norm(
    x_prev: torch.Tensor | None,
    residual_prev: torch.Tensor,
    post_mix_prev: torch.Tensor | None,
    comb_mix_prev: torch.Tensor | None,
    pre_mix: torch.Tensor,
    norm_weight: torch.Tensor,
    norm_eps: float,
    mxfp8: bool = False,
    nvfp4_scale: torch.Tensor | None = None,
) -> tuple[torch.Tensor, ...]:
    """Critical half of a lagged mHC boundary.

    ``residual_prev`` is ``[M, 4, H]`` bf16 and ``pre_mix`` the previous sublayer's ``[M, 4]``
    pre gates. With ``x_prev`` (``[M, H]`` bf16), ``post_mix_prev`` (``[M, 4]``) and
    ``comb_mix_prev`` (``[M, 4, 4]``) the previous sublayer's post update is applied first;
    with ``x_prev=None`` ``residual_prev`` is the boundary's residual as is.

    Returns ``(residual, layer_input)``: the boundary's residual ``[M, 4, H]`` (``residual_prev``
    itself without a post update) and ``rmsnorm(sum_j pre[j] * residual[:, j], norm_weight)``.
    With ``mxfp8`` a third element holds ``torch.ops.trtllm.mxfp8_quantize(layer_input, True)``
    (the scales of the swizzled layout's padding rows, which no product reads, are left unwritten);
    with an FP32 ``nvfp4_scale [1]`` it holds ``fp4_quantize(layer_input, nvfp4_scale, 16, False,
    False)``.
    """
    import cuda.bindings.driver as cuda
    import cutlass

    m, mult, hidden = residual_prev.shape
    assert mult == 4 and residual_prev.dtype == torch.bfloat16
    post_update = x_prev is not None
    res_flat = residual_prev.reshape(m, mult * hidden).contiguous()
    pre = pre_mix.reshape(m, mult).to(torch.float32).contiguous()
    weight = norm_weight.to(torch.bfloat16).contiguous().view(1, hidden)
    out = torch.empty((m, hidden), dtype=torch.bfloat16, device=residual_prev.device)
    if post_update:
        x = x_prev.reshape(m, hidden).contiguous()
        post = post_mix_prev.reshape(m, mult).to(torch.float32).contiguous()
        comb = comb_mix_prev.reshape(m, mult * mult).to(torch.float32).contiguous()
        res_out = torch.empty_like(res_flat)
    else:
        # Unused operands of the collapse-only variant (shapes must match the compiled ones).
        x = out
        post = pre
        comb = pre.new_empty((m, mult * mult))
        res_out = res_flat
    quant, quantized, scale_addr = _quantized_outputs(m, hidden, out.device, mxfp8, nvfp4_scale)
    fn = _compiled_post_collapse_norm(
        m, hidden, float(norm_eps), post_update, get_env_enable_pdl(), quant
    )
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    fn(
        graph_safe_from_dlpack(x),
        graph_safe_from_dlpack(res_flat),
        graph_safe_from_dlpack(post),
        graph_safe_from_dlpack(comb),
        graph_safe_from_dlpack(pre),
        graph_safe_from_dlpack(weight),
        graph_safe_from_dlpack(res_out),
        graph_safe_from_dlpack(out),
        cutlass.Int64(quantized[0].data_ptr() if quant else 0),
        cutlass.Int64(quantized[1].data_ptr() if quant else 0),
        cutlass.Int64(scale_addr),
        stream,
    )
    if quant:
        return res_out.view(m, mult, hidden), out, quantized
    return res_out.view(m, mult, hidden), out
