# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Plan / run host side of the D576 sparse MLA kernels.

Every query row is one token with ``num_qo_heads`` heads of E4M3 ``[latent 512 | RoPE 64]``. Each row
attends to the KV rows its int32 index row selects (``-1`` entries are masked) up to its live length, in
one pool of ``native`` E4M3 ``[N, 576]`` rows. The BF16 output is ``[rows, heads, 512]``; the natural-log
LSE of every row and head is written as well.

The schedule (kernel family, query tile, KV split, V partition, issue warps, retained K) is the
FlashInfer policy's choice for the planned row count and the device's SM count. A plan whose schedule
is not one of ``QUALIFIED_SCHEDULES`` is rejected: this package carries only the code those schedules
run, the 2CTA M128 kernel without a KV split, which the policy picks for power-of-two plans of 32 rows
or more.

The ``mla_decode_*`` modules: D576 sparse MLA attention (latent 512 + RoPE 64) on Blackwell for Kimi K3.1 DSA
layers. Derived from FlashInfer ``flashinfer/attention/prims_ts``, specialized to the plans
the DSA context-prefill path requests: E4M3 queries, one query row per token,
up to 128 query heads, a single KV pool of ``native`` E4M3 rows, BF16 output, and no attention
sinks, output quantization or extra pool. It keeps the FlashInfer schedule policy and the kernel
it selects for power-of-two plans of 32 rows or more (the 2CTA M128 kernel without a KV split), so
the results are bit-identical to the FlashInfer implementation. Compiled kernels are cached under a
key derived from the plan and the ``mla_decode_*`` modules' sources.
"""

import copy
import functools
import math
import struct
from dataclasses import dataclass, field
from typing import Any

import torch

from .mla_decode_compile_cache import build_and_load_attention_kernel
from .mla_decode_policy import select_sparse_mla_profile

_COMPILE_OPTIONS = "--enable-tvm-ffi --opt-level 2"
_LATENT_DIM = 512
_ROPE_DIM = 64
_QUERY_DIM = _LATENT_DIM + _ROPE_DIM
_WORKSPACE_ALIGNMENT = 256

# The plan envelope this package serves: query heads per call, Top-K width and KV formats.
SPARSE_MLA_HEADS = 96
SPARSE_MLA_TOP_K = 2048
KV_CACHE_FORMATS = ("native",)

# Schedules the policy selects for that envelope on SM100 / SM103 (148-160 SMs), as
# (KV format, family, tile_Q, split_kv, V CTAs, scheduler, issue warps, balanced registers,
#  retained K, retained-K stages, uniform offsets): the 2CTA M128 kernel without a KV split, which
# the policy picks for power-of-two plans of 32 rows or more.
QUALIFIED_SCHEDULES = frozenset(
    {
        ("native", "2cta", 128, 1, 1, "nonpersistent", 1, False, False, 0, False),
        ("native", "2cta", 128, 1, 1, "static", 1, False, False, 0, False),
        ("native", "2cta", 128, 1, 1, "static", 1, True, False, 0, False),
    }
)
_SCHEDULE_FIELDS = (
    "family",
    "tile_size_q",
    "split_kv",
    "head_dim_ctas",
    "scheduler",
    "gather_issue_warps",
    "balanced_registers",
    "reuse_kv",
    "reuse_kv_stages",
    "uniform_offset_cache",
)
_FIXED_SCHEDULE_FIELDS = {"kv_pipeline_stages": 0}


def sparse_mla_unsupported_reason(
    num_qo_heads: int, max_topk: int, kv_cache_format: str
) -> str | None:
    """Why a plan with these parameters cannot run here, or None."""
    if num_qo_heads != SPARSE_MLA_HEADS:
        return f"serves {SPARSE_MLA_HEADS} query heads per call, got {num_qo_heads}."
    if max_topk != SPARSE_MLA_TOP_K:
        return f"serves Top-K {SPARSE_MLA_TOP_K}, got {max_topk}."
    if kv_cache_format not in KV_CACHE_FORMATS:
        return f"serves KV formats {KV_CACHE_FORMATS}, got {kv_cache_format!r}."
    return None


def _schedule_key(kv_cache_format: str, profile) -> tuple:
    return (kv_cache_format,) + tuple(getattr(profile, name) for name in _SCHEDULE_FIELDS)


def _check_schedule(kv_cache_format: str, profile, rows: int, sm_count: int) -> None:
    fixed = all(getattr(profile, name) == value for name, value in _FIXED_SCHEDULE_FIELDS.items())
    if not fixed or _schedule_key(kv_cache_format, profile) not in QUALIFIED_SCHEDULES:
        raise NotImplementedError(
            f"sparse MLA schedule {profile} (rows {rows}, {sm_count} SMs, {kv_cache_format}) is not one of "
            "the qualified schedules"
        )


@dataclass(frozen=True)
class _CompileSpec:
    """Batch-independent compile identity of one plan (the kernel object does not compare)."""

    device_index: int
    kernel_signature: tuple[object, ...]
    num_qo_heads: int
    kv_lora_rank: int
    qk_rope_head_dim: int
    page_size: int
    q_dtype_key: str
    out_dtype_key: str
    max_q_len: int
    packed_query: bool
    split_kv: int = field(compare=False, hash=False)
    kernel: Any = field(compare=False, hash=False, repr=False)
    write_final_lse: bool = True
    lse_mode: str = "none"


@dataclass(frozen=True)
class _WorkspaceSection:
    byte_offset: int
    byte_size: int
    shape: tuple[int, ...]
    dtype: torch.dtype


def _append_workspace_section(byte_end: int, shape: tuple[int, ...], dtype: torch.dtype):
    byte_offset = (
        (byte_end + _WORKSPACE_ALIGNMENT - 1) // _WORKSPACE_ALIGNMENT * _WORKSPACE_ALIGNMENT
    )
    byte_size = math.prod(shape) * torch.empty((), dtype=dtype).element_size()
    return _WorkspaceSection(byte_offset, byte_size, shape, dtype), byte_offset + byte_size


def _workspace_section_view(workspace: torch.Tensor, section: _WorkspaceSection) -> torch.Tensor:
    data = workspace.reshape(-1).view(torch.uint8)
    data = data[section.byte_offset : section.byte_offset + section.byte_size]
    return data.view(section.dtype).view(section.shape)


def _kernel_signature(kernel: Any) -> tuple[object, ...]:
    """All static kernel state except the batch extent."""
    signature = (
        type(kernel).__module__,
        type(kernel).__qualname__,
        kernel.compile_signature(),
    )
    hash(signature)
    return signature


def _resolve_device(device) -> torch.device:
    device = torch.device(device)
    if device.type != "cuda":
        raise ValueError(f"sparse MLA runs on a CUDA device, got {device}")
    index = int(device.index) if device.index is not None else int(torch.cuda.current_device())
    return torch.device("cuda", index)


def _resolve_plan(device, batch_size: int, num_qo_heads: int, max_topk: int, kv_cache_format: str):
    """Choose the schedule, build the kernel object and size the workspace (no compile)."""
    from .mla_decode_2cta_kernel import MlaDecodeTs

    device = _resolve_device(device)
    if torch.cuda.get_device_capability(device) not in ((10, 0), (10, 3)):
        raise NotImplementedError("sparse MLA requires SM100 or SM103")
    if not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError(f"batch_size must be a positive integer, got {batch_size!r}")
    reason = sparse_mla_unsupported_reason(num_qo_heads, max_topk, kv_cache_format)
    if reason is not None:
        raise NotImplementedError(f"sparse MLA {reason}")
    max_rows = batch_size
    capacity = max(256, (max_topk + 127) // 128 * 128)
    sm_count = torch.cuda.get_device_properties(device).multi_processor_count
    profile = select_sparse_mla_profile(
        rows=max_rows,
        heads=num_qo_heads,
        query_length=1,
        capacity=capacity,
        dtype="fp8",
        sm_count=sm_count,
        rope_dim=_ROPE_DIM,
    )
    _check_schedule(kv_cache_format, profile, max_rows, sm_count)
    splits = profile.split_kv
    kernel = MlaDecodeTs(
        page_size=1,
        max_active_clusters=sm_count // 2,
        is_persistent=profile.scheduler != "nonpersistent",
        is_var_seq=False,
        is_var_split_kv=False,
        static_split_kv=splits,
        qkv_dtype="e4m3",
        out_dtype="bf16",
        rope_dim=_ROPE_DIM,
        num_heads=num_qo_heads,
        seq_len_q=1,
        mask_type="dense",
        softmax_rescale_threshold_log2=0.0,
        # Query rows are virtual batch items, so the physical Q length stays compile-time.
        fixed_seqlen=True,
    )
    kernel.sparse_gather_warps = profile.gather_issue_warps
    kernel.sparse_kv_stages = profile.kv_pipeline_stages
    kernel.sparse_uniform_pages = profile.uniform_offset_cache
    kernel.sparse_balanced_registers = profile.balanced_registers
    kernel.lse_base_on_e = False
    # Every qualified schedule runs one KV split: the main kernel finalizes the attention.
    kernel.output_quant = None
    kernel.assume_valid_prefix = False
    kernel.fuse_sparse_epilogue = splits == 1
    kernel.direct_sparse = True
    kernel.direct_sparse_capacities = (max_topk, 0)
    spec = _CompileSpec(
        device_index=device.index,
        kernel_signature=_kernel_signature(kernel),
        num_qo_heads=num_qo_heads,
        kv_lora_rank=_LATENT_DIM,
        qk_rope_head_dim=_ROPE_DIM,
        page_size=1,
        q_dtype_key="float8_e4m3fn",
        out_dtype_key="bfloat16",
        max_q_len=1,
        packed_query=False,
        split_kv=splits,
        kernel=kernel,
    )
    lse_section, byte_end = _append_workspace_section(0, (max_rows, num_qo_heads), torch.float32)
    return device, profile, spec, {"lse": lse_section}, byte_end


@functools.cache
def _get_compiled(spec: _CompileSpec):
    """Compile (or load from the persistent cache) one plan's kernel."""
    import cutlass
    from cutlass import cute

    kernel = copy.copy(spec.kernel)
    q_dtype = cutlass.Float8E4M3FN
    num_heads = spec.num_qo_heads
    # Symbolic extents in the original creation order (the third one is unused).
    physical_pages = cute.sym_int()
    batch_size = cute.sym_int()
    cute.sym_int()

    # Fake tensors pin the launch ABI while page counts, index widths and row counts vary.
    q_stride = (_QUERY_DIM, 1, num_heads * _QUERY_DIM, num_heads * _QUERY_DIM)
    q_latent = cute.runtime.make_fake_tensor(
        q_dtype,
        (num_heads, _LATENT_DIM, 1, batch_size),
        stride=q_stride,
        assumed_align=16,
    )
    q_rope = cute.runtime.make_fake_tensor(
        q_dtype,
        (num_heads, _ROPE_DIM, 1, batch_size),
        stride=q_stride,
        assumed_align=16,
    )
    cache_dim = _QUERY_DIM
    cache_dtype = q_dtype
    c_latent = cute.runtime.make_fake_tensor(
        cache_dtype,
        (1, _LATENT_DIM, physical_pages),
        stride=(cache_dim, 1, cache_dim),
        assumed_align=16,
    )
    c_rope = cute.runtime.make_fake_tensor(
        cache_dtype,
        (1, cache_dim, cute.sym_int()),
        stride=(cache_dim, 1, cache_dim),
        assumed_align=16,
    )
    page_offsets = cute.runtime.make_fake_tensor(
        cutlass.Int32,
        (cute.sym_int(), batch_size),
        stride=(1, cute.sym_int64(divisibility=1)),
        assumed_align=4,
    )
    out = cute.runtime.make_fake_tensor(
        cutlass.BFloat16,
        (num_heads, _LATENT_DIM, 1, batch_size),
        stride=(_LATENT_DIM, 1, num_heads * _LATENT_DIM, num_heads * _LATENT_DIM),
        assumed_align=16,
    )
    lse = cute.runtime.make_fake_tensor(
        cutlass.Float32,
        (num_heads, 1, batch_size),
        stride=(1, num_heads, num_heads),
        assumed_align=4,
    )
    cache_seqs = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32, (batch_size,), stride_order=(0,), assumed_align=4
    )

    def indices():
        return cute.runtime.make_fake_tensor(
            cutlass.Int32,
            (batch_size, cute.sym_int()),
            stride=(cute.sym_int64(), 1),
            assumed_align=4,
        )

    sinks = cute.runtime.make_fake_compact_tensor(
        cutlass.Float32, (num_heads,), stride_order=(0,), assumed_align=4
    )
    sparse_inputs = (
        indices(),
        indices(),
        cache_seqs,
        cache_seqs,
        sinks,
        cutlass.Float32(1.0),
    )
    stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)

    def compile_kernel():
        from .mla_decode_helpers_schedule import mla_build_diagnostics

        with mla_build_diagnostics():
            return cute.compile[cute.FrontendNext](
                kernel,
                q_latent,
                q_rope,
                c_latent,
                c_rope,
                page_offsets,
                out,
                lse,
                cutlass.Int32(spec.split_kv),
                cache_seqs,
                None,
                sparse_inputs,
                cutlass.Float32(1.0),
                cutlass.Float32(1.0),
                stream,
                output_quant_args=None,
                max_q_len=cutlass.Int32(0),
                options=_COMPILE_OPTIONS,
            )

    with torch.cuda.device(spec.device_index):
        return build_and_load_attention_kernel(
            "mla_decode", "main", spec, _COMPILE_OPTIONS, compile_kernel, extra=()
        )


def _host_scale(value, name: str) -> float:
    """Round a positive finite host scale to the kernel's FP32 argument."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a Python int or float")
    rounded = struct.unpack("f", struct.pack("f", float(value)))[0]
    if not math.isfinite(rounded) or rounded <= 0:
        raise ValueError(f"{name} must be a positive finite float32, got {value}")
    return rounded


class BatchSparseMLADecodePagedTSWrapper:
    """D576 sparse MLA over caller-selected KV rows: ``plan`` once per row capacity, then ``run``.

    ``plan`` chooses and compiles the schedule for ``batch_size`` query rows; ``run`` serves any row
    count up to it. A wrapper (and its workspace) supports one in-flight run. CUDA graphs need a
    completed eager ``run`` first, stable tensor addresses and preallocated outputs; the host scales
    are captured as launch arguments.
    """

    def __init__(self, workspace_buffer: torch.Tensor | None = None):
        self._workspace_buffer = workspace_buffer
        self._state = None
        self.workspace_size_bytes = 0

    def plan(
        self,
        device,
        batch_size: int,
        num_qo_heads: int,
        *,
        max_topk: int,
        max_q_len: int = 1,
        packed_query: bool = False,
        q_dtype: torch.dtype = torch.float8_e4m3fn,
        qk_rope_head_dim: int = _ROPE_DIM,
        kv_cache_format: str = "native",
        return_lse: bool = False,
    ) -> None:
        """Plan ``batch_size`` rows of ``num_qo_heads`` heads with Top-K width ``max_topk``.

        Rows are tokens with one query position each (``max_q_len=1``); queries are E4M3 with a 64-wide
        RoPE tail. ``return_lse`` makes ``run`` return a fresh LSE tensor along with the output.
        """
        if max_q_len != 1 or q_dtype != torch.float8_e4m3fn or qk_rope_head_dim != _ROPE_DIM:
            raise NotImplementedError(
                "sparse MLA plans one E4M3 query position per row with a 64-wide RoPE tail"
            )
        device, profile, spec, sections, byte_end = _resolve_plan(
            device, batch_size, num_qo_heads, max_topk, kv_cache_format
        )
        workspace = self._workspace_buffer
        if workspace is None:
            workspace = torch.empty(byte_end, device=device, dtype=torch.uint8)
        if workspace.device != device or workspace.numel() * workspace.element_size() < byte_end:
            raise ValueError(f"the workspace must hold {byte_end} bytes on {device}")
        buffers = {name: _workspace_section_view(workspace, s) for name, s in sections.items()}
        compiled = _get_compiled(spec)
        cache_dim = _QUERY_DIM
        cache_dtype = torch.float8_e4m3fn
        self._state = {
            "device": device,
            "max_rows": batch_size,
            "num_qo_heads": num_qo_heads,
            "max_topk": max_topk,
            "return_lse": return_lse,
            "cache_dim": cache_dim,
            "cache_dtype": cache_dtype,
            "buffers": buffers,
            "compiled": compiled,
            "split_kv": profile.split_kv,
            # The kernel takes a second (extra) KV source; this package always leaves it empty.
            "extra_lengths": torch.zeros((batch_size,), device=device, dtype=torch.int32),
            "extra_indices": torch.zeros((batch_size, 1), device=device, dtype=torch.int32),
            "extra_cache": torch.zeros((1, 1, cache_dim), device=device, dtype=cache_dtype),
            "sinks": torch.full((num_qo_heads,), -torch.inf, device=device, dtype=torch.float32),
        }
        self.workspace_size_bytes = byte_end

    def _pool(self, cache: torch.Tensor) -> torch.Tensor:
        """The pool's rows as ``[rows, 1, record]`` (pages are flattened; indices address rows)."""
        dim = self._state["cache_dim"]
        if cache.dtype != self._state["cache_dtype"] and not (
            cache.dtype in (torch.uint8, torch.float8_e4m3fn)
            and self._state["cache_dtype"] in (torch.uint8, torch.float8_e4m3fn)
        ):
            raise ValueError(f"the KV pool must be {self._state['cache_dtype']}, got {cache.dtype}")
        if cache.shape[-1] != dim or cache.stride(-1) != 1:
            raise ValueError(f"KV records must be {dim} wide with a unit last stride")
        if cache.ndim == 2:
            cache = cache.unsqueeze(1)
        row_stride = cache.stride(0) // dim
        rows = (cache.shape[0] - 1) * row_stride + cache.shape[1]
        return cache.as_strided((rows, 1, dim), (dim, dim, 1))

    def run(
        self,
        query: torch.Tensor,
        kv_cache: torch.Tensor,
        indices: torch.Tensor,
        lengths: torch.Tensor,
        *,
        qo_indptr: torch.Tensor | None = None,
        softmax_scale: float,
        output_scale: float = 1.0,
        out: torch.Tensor | None = None,
        lse: torch.Tensor | None = None,
        validate: bool = False,
    ):
        """Attend ``query`` ``[rows, heads, 576]`` to ``kv_cache`` rows ``indices[r, :lengths[r]]``.

        ``indices`` is int32 ``[rows, K]`` (unit last stride; ``-1`` entries are masked), ``lengths``
        int32 ``[rows]``. ``softmax_scale`` multiplies the QK scores and ``output_scale`` the attended
        V values (they carry the Q / KV descales). ``qo_indptr`` is accepted for API compatibility;
        every row is one token. ``validate`` is accepted and ignored. Returns ``out`` (``[rows, heads,
        512]`` BF16), or ``(out, lse)`` when planned with ``return_lse``.
        """
        del qo_indptr, validate
        state = self._state
        if state is None:
            raise RuntimeError("plan() must be called before run()")
        heads = state["num_qo_heads"]
        rows = int(query.shape[0])
        if query.dtype != torch.float8_e4m3fn or tuple(query.shape[1:]) != (
            heads,
            _QUERY_DIM,
        ):
            raise ValueError(f"query must be E4M3 [rows, {heads}, {_QUERY_DIM}]")
        if rows > state["max_rows"]:
            raise ValueError(f"{rows} rows exceed the planned {state['max_rows']}")
        if indices.dtype != torch.int32 or indices.ndim != 2 or indices.shape[0] != rows:
            raise ValueError("indices must be int32 [rows, K]")
        if lengths.dtype != torch.int32 or tuple(lengths.shape) != (rows,):
            raise ValueError("lengths must be int32 [rows]")
        primary = self._pool(kv_cache)
        if out is None:
            out = torch.empty(
                (rows, heads, _LATENT_DIM), device=state["device"], dtype=torch.bfloat16
            )
        elif out.dtype != torch.bfloat16 or out.numel() != rows * heads * _LATENT_DIM:
            raise ValueError(f"out must be BF16 [rows, {heads}, {_LATENT_DIM}]")
        if lse is None:
            lse = (
                torch.empty((rows, heads), device=state["device"], dtype=torch.float32)
                if state["return_lse"]
                else state["buffers"]["lse"][:rows]
            )
        if rows:
            sparse_inputs = (
                indices,
                state["extra_indices"][:rows],
                lengths,
                state["extra_lengths"][:rows],
                state["sinks"],
                1.0,
            )
            state["compiled"](
                query.view(rows, 1, heads, _QUERY_DIM)[..., :_LATENT_DIM].permute(2, 3, 1, 0),
                query.view(rows, 1, heads, _QUERY_DIM)[..., _LATENT_DIM:].permute(2, 3, 1, 0),
                (primary if primary.dtype == torch.uint8 else primary[..., :_LATENT_DIM]).permute(
                    1, 2, 0
                ),
                state["extra_cache"].permute(1, 2, 0),
                indices.transpose(0, 1),
                out.view(rows, 1, heads, _LATENT_DIM).permute(2, 3, 1, 0),
                lse.view(rows, 1, heads).permute(2, 1, 0),
                state["split_kv"],
                lengths,
                None,
                sparse_inputs,
                _host_scale(softmax_scale, "softmax_scale"),
                _host_scale(output_scale, "output_scale"),
                None,
                1,
            )
        return (out, lse) if state["return_lse"] else out


def get_prims_ts_sparse_mla_decode_workspace_size(
    device,
    batch_size: int,
    num_qo_heads: int,
    *,
    max_topk: int,
    kv_cache_format: str = "native",
    **_plan_kwargs,
) -> int:
    """Workspace bytes of ``plan`` with the same arguments (no allocation, no compile)."""
    return _resolve_plan(device, batch_size, num_qo_heads, max_topk, kv_cache_format)[4]


__all__ = [
    "KV_CACHE_FORMATS",
    "QUALIFIED_SCHEDULES",
    "SPARSE_MLA_HEADS",
    "SPARSE_MLA_TOP_K",
    "BatchSparseMLADecodePagedTSWrapper",
    "get_prims_ts_sparse_mla_decode_workspace_size",
    "sparse_mla_unsupported_reason",
]
