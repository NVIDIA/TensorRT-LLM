# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Weight table and loader: Qwen3.8-Flash-Next-NVFP4 / sm_103 / tp1.

One `W` per weight role (shape, checkpoint source, dtype, load-time transform);
`Qwen4ExpTp1Weights` unrolls the table into declaration, manifest and load. tp1
means no per-rank slicing. Storage is HF `[out, in]`; GEMM views, grouped-norm
slices, fp32 gating params and NVFP4 scalars are derived in `modeling.py` after load.

Transforms:

* trunk attention input projection: `q_proj`, `k_proj`, `v_proj` and the indexer's
  `index_qk_proj` read the same activation, so they are one
  `[q | k | v | gate | index | pad, hidden]` operand and one GEMM (`_attn_in_w`). `q_proj`
  interleaves q and the output gate per head; its q halves go first so the GEMM output's
  leading columns are the packed `[q | k | v]` the fused norm+rope rewrites in place, then
  the gate halves, the index projection and `index_head_dim` zero rows (a row is a whole
  number of `head_dim` heads). The draft (MTP) layer keeps its separate projections;
* fused shared-expert gate|up: `[gate ; up]` (the half order `flashinfer_silu_and_mul` reads);
* NVFP4 routed-expert stacks, laid out for the trunk's routed backend
  (`core.trunk_moe_backend`): `[up ; gate]`, then 128x4 scale swizzle. trtllm-gen also
  interleaves FC1 row by row and 32-row block shuffles weight and scale bytes (FC2:
  shuffle only); CuteDSL interleaves FC1 in 64-row up/gate groups and leaves FC2 in
  checkpoint row order. Under trtllm-gen a second copy in the CuteDSL layout (`fc*_cd`)
  serves the prefill-sized launches (`modeling._CUTE_DSL_MOE_MIN_ROWS`) when the core keeps
  it (`core.moe_prefill_copy`);
* n-gram table: 128 row chunks concatenated into one `[rows, 160]` buffer (the hash
  emits global row indices).

The n-gram table is not in the table: its height is a checkpoint fact, so `load`
registers it with `torch.empty` and adds its rows to the manifest;
`post_load_weights` reconciles the height. Residency (device or host) is set by
`TRTLLM_MODELING_V2_PLE_HOST_OFFLOAD` (`_host_resident_empty`); the manifest is the
same either way. Offloading frees ~47.7 GiB, which the KV cache manager turns into
more slots; several entries certify at most 461 slots, so raising
`kv_cache_fraction` with the offload on needs those cells re-checked.

Not loaded: `mtp.*` when there is no `speculative_config`. With one, the whole MTP module
is loaded, including the draft layer's own QSA indexer (`_index_branch`). MTP experts are
plain row-major FP8 128x128: `[up ; gate]` concat, bf16
`weight_scale_inv` cast to fp32. `lm_head` is the shell's parameter.

`load(model, weights)` fills every declared parameter once and asserts coverage
both ways: every key is consumed or in the non-load set.
"""

import os
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Callable

import torch
from cuda.bindings import runtime as cudart
from torch import nn

from tensorrt_llm._utils import CUASSERT

#: `layers=ALL_LAYERS` is the common case and the default. A sentinel rather
#: than a lambda so the table reads as data, and rather than `True` so the
#: field has one type of answer: which layers.
ALL_LAYERS = object()


@dataclass(frozen=True)
class W:
    """One weight role: what it is, where its bytes come from, what happens on the way.

    `shape` takes the dimension bundle and returns the declared shape; every
    shape in these models is derived from configuration rather than constant.

    `src` says where the bytes come from, in one of three forms:

    * a checkpoint key template, with `{p}` standing for the layer prefix;
    * a sequence of `(template, slice_fn)` pairs, when one parameter is
      assembled from several checkpoint tensors side by side -- `slice_fn`
      takes the dimension bundle and returns an index into the destination;
    * a callable `(d, i) -> [(key, index), ...]`, taking the dimension bundle
      and the layer index. The first two forms are special cases of it; this
      one exists for sources that are not a fixed template -- a per-expert
      row, or a prefix that differs between layer families. A `key` may itself
      be a tuple, meaning the transform is handed those tensors together.

    `layers` says which layers carry this weight: `ALL_LAYERS` for every one,
    `None` for a weight that is not per-layer at all, or a callable
    `d -> iterable[int]` when only some layers have it.

    `transform` runs on the source tensor before the copy, on the
    destination's device. `None` means the checkpoint tensor is already the
    declared shape.
    """

    name: str
    shape: Callable[[Any], tuple[int, ...]]
    src: str | tuple[tuple[str, Callable], ...] | Callable
    dtype: torch.dtype | None = None
    transform: Callable | None = None
    layers: object | None = ALL_LAYERS

    def layer_indices(self, d: SimpleNamespace) -> list[int] | None:
        """The layers this weight exists on, or None if it is not per-layer."""
        if self.layers is None:
            return None
        if self.layers is ALL_LAYERS:
            return list(range(d.num_layers))
        return list(self.layers(d))

    def sources(self, d: SimpleNamespace, i: int | None, prefix: str) -> list[tuple[Any, Any]]:
        """`[(checkpoint key or key tuple, index into the param | None), ...]`."""
        if callable(self.src):
            return list(self.src(d, i))
        if isinstance(self.src, str):
            return [(self.src.format(p=prefix), None)]
        return [(template.format(p=prefix), slice_fn(d)) for template, slice_fn in self.src]


# ----------------------------------------------------------- host residency --
#
# Allocates the one PLE n-gram table. The table stays a `cuda` tensor of the same
# shape/dtype. Registered, device-mapped pages (not a host gather) because the gather's
# row indices are produced on device inside the captured decode graph.

#: `cudaMemoryType` values; the core asserts on them after load.
CUDA_MEMORY_TYPE_HOST = 1
CUDA_MEMORY_TYPE_DEVICE = 2


def memory_kind(ptr: int) -> int:
    """The driver's own answer to "what kind of memory is this pointer".

    Returns a `cudaMemoryType`. This is the residency question asked of the
    process rather than of `nvidia-smi`, and it is the only one a
    working-but-resident implementation cannot answer the way an offloaded one
    does.
    """
    (attributes,) = CUASSERT(cudart.cudaPointerGetAttributes(ptr))
    return int(attributes.type)


#: `cudaHostRegister` flags (`driver_types.h`): mapped, so a kernel may dereference the
#: pages, and portable, for every context in the process.
_CUDA_HOST_REGISTER_PORTABLE = 0x01
_CUDA_HOST_REGISTER_MAPPED = 0x02


def _local_cpus(device) -> set | None:
    """The CPUs sysfs lists as local to `device`'s PCI function and this process may run on,
    or None when that is unknown."""
    try:
        props = torch.cuda.get_device_properties(device)
        path = (
            f"/sys/bus/pci/devices/{props.pci_domain_id:04x}:{props.pci_bus_id:02x}:"
            f"{props.pci_device_id:02x}.0/local_cpulist"
        )
        with open(path) as fh:
            text = fh.read().strip()
    except (OSError, AttributeError, RuntimeError):
        return None
    cpus = set()
    for part in filter(None, text.split(",")):
        lo, _, hi = part.partition("-")
        cpus.update(range(int(lo), int(hi or lo) + 1))
    return (cpus & os.sched_getaffinity(0)) or None


class _RegisteredHostBuffer:
    """A pageable CPU allocation registered (pinned and mapped) for the device, and
    unregistered before its storage is released: at this size glibc serves the
    allocation with `mmap` and returns it with `munmap`, so freeing registered pages
    would leave the driver mapping memory the OS may hand out again.

    Not torch's pinned allocator: it rounds the 47.7 GiB table up to 64 GiB, and on
    GB300 the device's random row gathers from its pages run an order of magnitude
    slower than from registered pageable memory."""

    def __init__(self, nbytes: int, device=None) -> None:
        # The registration faults the pages in on this thread, so under the default local
        # allocation policy they land on the NUMA node of the CPU it runs on. Pinned to the
        # CPUs local to `device` meanwhile, the table sits next to the GPU that gathers from
        # it: on GB300, random row gathers from the other socket's memory run ~1.7x slower,
        # and with a process per GPU half of them would otherwise land there.
        local = _local_cpus(device) if device is not None else None
        previous = os.sched_getaffinity(0) if local else None
        if local:
            os.sched_setaffinity(0, local)
        try:
            self._storage = torch.empty(nbytes, dtype=torch.uint8)
            self.pointer = self._storage.data_ptr()
            CUASSERT(
                cudart.cudaHostRegister(
                    self.pointer, nbytes, _CUDA_HOST_REGISTER_PORTABLE | _CUDA_HOST_REGISTER_MAPPED
                )
            )
        finally:
            if previous:
                os.sched_setaffinity(0, previous)
        # Bound now: at interpreter shutdown the module may already be torn down.
        self._unregister = cudart.cudaHostUnregister

    def __del__(self) -> None:
        pointer, self.pointer = self.pointer, 0
        if pointer:
            self._unregister(pointer)


def _host_resident_empty(
    rows: int, cols: int, dtype, device
) -> tuple[torch.Tensor, _RegisteredHostBuffer]:
    """`[rows, cols]` of `dtype` on `device`, whose bytes live in host memory.

    Returns the tensor and the registered CPU allocation that owns the pages. The
    caller **must** keep the second alive for as long as the first: the tensor
    is a view through a device address and owns nothing.
    """
    itemsize = torch.empty(0, dtype=dtype).element_size()
    assert itemsize == 1, (
        f"the host-resident allocation reinterprets a `|u1` buffer, so its "
        f"element must be one byte; {dtype} is {itemsize}"
    )
    assert device.type == "cuda", f"host offload needs a cuda device to map into; got {device}"

    host = _RegisteredHostBuffer(rows * cols, device)
    (device_pointer,) = CUASSERT(cudart.cudaHostGetDevicePointer(host.pointer, 0))

    # No public torch API builds a tensor over a pointer the caching allocator
    # did not hand out; `__cuda_array_interface__` is the numba/cupy path.
    # `|u1` because the protocol has no float8 typestr; reinterpreted below.
    interface = SimpleNamespace(
        __cuda_array_interface__={
            "shape": (rows, cols),
            "typestr": "|u1",
            "data": (int(device_pointer), False),
            "version": 3,
            "strides": None,
        }
    )
    table = torch.as_tensor(interface, device=device).view(dtype)
    # `torch.as_tensor` copies when it decides the buffer is not already on the
    # requested device; a copy here would be a silent 48 GiB device allocation.
    assert table.data_ptr() == int(device_pointer), (
        "torch.as_tensor copied the host-mapped buffer instead of viewing it, "
        "so the table would be device-resident after all"
    )
    assert table.shape == (rows, cols) and table.dtype == dtype, (table.shape, table.dtype)
    return table, host


# ------------------------------------------------------------------ the table --

_LM = "model.language_model"
_VISUAL = "model.visual"
_MTP = "mtp"
_MTP_LAYER = "mtp.layers.0"

# NVFP4 block size: one e4m3 scale per 16 contiguous elements along K.
SF_VEC = 16

# The FP8 block-scale family's block edge, on both axes of the MTP experts'
# weight scales and along the activation row.
FP8_BLOCK = 128

# Parameter-key suffixes after which to release transform scratch. The expert
# relayouts allocate per-expert scratch 512 times per parameter and the table
# fill stages one shard at a time; without this, peak load memory is the whole
# model rather than one parameter.
_RELEASE_AFTER = (
    "_fc2_s",
    "_fc1_s",
    "_fc2_s_cd",
    "_fc1_s_cd",
    "ple_table",
    "mtp_fc1_w",
    "mtp_fc2_w",
)


def _block32_perm(rows: int, device) -> torch.Tensor:
    """Gather index of the 32-row block shuffle: inside each aligned block of
    32 rows, source row `4u + v` (0<=u<8, 0<=v<4) lands at destination row
    `8v + u`."""
    assert rows % 32 == 0, rows
    src = torch.arange(32)
    dst = (src % 4) * 8 + src // 4
    idx = torch.empty(32, dtype=torch.long)
    idx[dst] = src
    blocks = rows // 32
    return (idx.repeat(blocks) + torch.arange(blocks).repeat_interleave(32) * 32).to(device)


def _interleave_perm(rows: int, device) -> torch.Tensor:
    """Gather index that interleaves the `[up | gate]` halves of a `2*I` row
    stack into up0, gate0, up1, gate1, ..."""
    p = torch.empty(rows, dtype=torch.long)
    p[0::2] = torch.arange(0, rows // 2)
    p[1::2] = torch.arange(rows // 2, rows)
    return p.to(device)


#: The group the CuteDSL FC1 epilogue pairs up and gate rows in: rows `[128k, 128k + 64)`
#: of the fused operand are up rows and the next 64 their gate rows.
_CUTEDSL_GATE_UP_GROUP = 64


def _group_interleave_perm(rows: int, group: int, device) -> torch.Tensor:
    """Gather index that interleaves the `[up | gate]` halves of a `2*I` row stack in
    `group`-row blocks: up[0:g], gate[0:g], up[g:2g], gate[g:2g], ..."""
    assert rows % (2 * group) == 0, (rows, group)
    idx = torch.arange(rows, device=device).view(2, rows // (2 * group), group)
    return idx.transpose(0, 1).reshape(-1)


def _fc1_perm(rows: int, device, core) -> torch.Tensor:
    """The FC1 row gather of the trunk's routed backend. trtllm-gen: interleave then
    32-row block shuffle, composed into one gather. CuteDSL: the 64-row group
    interleave alone."""
    if core.trunk_moe_backend == "CUTEDSL":
        return _group_interleave_perm(rows, _CUTEDSL_GATE_UP_GROUP, device)
    return _interleave_perm(rows, device)[_block32_perm(rows, device)]


def _swizzle(x: torch.Tensor) -> torch.Tensor:
    """128x4 block-scale swizzle of an `[M, C]` uint8 scale matrix, returned
    flat (`M % 128 == 0` and `C % 4 == 0` hold at this geometry, so the result
    is exactly `M * C` bytes)."""
    return torch.ops.trtllm.block_scale_interleave(x.contiguous())


#: Rows after the shared expert's gate|up rows in its fused operand: the per-token gate's
#: one row, then zeros, so the GEMM's output row stays a whole number of 16-byte vectors.
SH_GATE_ROWS = 8


def _cat_rows(t: tuple[torch.Tensor, ...], core) -> torch.Tensor:
    """Fused shared-expert weight bytes: **gate** rows first, the half order
    `flashinfer_silu_and_mul` reads, then up, then the per-token gate's row and
    `SH_GATE_ROWS - 1` zero rows -- one GEMM forms gate|up and the gate logit."""
    gate, up, sh_gate = t
    pad = sh_gate.new_zeros(SH_GATE_ROWS - 1, sh_gate.shape[-1])
    return torch.cat((gate, up, sh_gate, pad), dim=0).contiguous()


def _attn_in_w(t: tuple[torch.Tensor, ...], core) -> torch.Tensor:
    """(q_proj, k_proj, v_proj, index_qk_proj) -> the fused `[q | k | v | gate | index |
    pad]` input projection of one trunk attention layer (see the module docstring).

    `q_proj`'s rows are `[heads, 2, head_dim]`: head `h`'s q rows, then its gate rows.
    """
    q, k, v, idx = t
    heads, head_dim, hidden = core.heads_q, core.head_dim, q.shape[-1]
    qg = q.reshape(heads, 2, head_dim, hidden)
    pad = q.new_zeros(core.index_head_dim, hidden)
    rows = (qg[:, 0].reshape(heads * head_dim, hidden), k, v)
    rows += (qg[:, 1].reshape(heads * head_dim, hidden), idx, pad)
    return torch.cat(rows, dim=0).contiguous()


def _expert_fc1_w(t: tuple[torch.Tensor, ...], core) -> torch.Tensor:
    """(up, gate) packed nibbles -> one expert's FC1 operand."""
    up, gate = t
    w = torch.cat([up, gate], dim=0)
    return torch.index_select(w, 0, _fc1_perm(w.shape[0], w.device, core)).contiguous()


def _expert_fc1_s(t: tuple[torch.Tensor, ...], core) -> torch.Tensor:
    """(up, gate) e4m3 scales -> one expert's FC1 scale operand: the weight's own row
    permutation, then the swizzle."""
    up, gate = t
    s = torch.cat([up.view(torch.uint8), gate.view(torch.uint8)], dim=0)
    rows, cols = s.shape
    s = torch.index_select(s, 0, _fc1_perm(rows, s.device, core))
    return _swizzle(s).reshape(rows, cols)


def _expert_fc2_w(t: torch.Tensor, core) -> torch.Tensor:
    """`down_proj` packed nibbles -> one expert's FC2 operand (no interleave): the
    32-row block shuffle on trtllm-gen, checkpoint row order on CuteDSL."""
    if core.trunk_moe_backend == "CUTEDSL":
        return t.contiguous()
    return torch.index_select(t, 0, _block32_perm(t.shape[0], t.device)).contiguous()


def _expert_fc2_s(t: torch.Tensor, core) -> torch.Tensor:
    """`down_proj` e4m3 scales -> one expert's FC2 scale operand: the weight's own row
    order, then the swizzle."""
    s = t.view(torch.uint8)
    rows, cols = s.shape
    if core.trunk_moe_backend != "CUTEDSL":
        s = torch.index_select(s, 0, _block32_perm(rows, s.device))
    return _swizzle(s).reshape(rows, cols)


#: The layout argument of the second, CuteDSL-layout expert copy (`fc*_cd`).
_CUTEDSL_LAYOUT = SimpleNamespace(trunk_moe_backend="CUTEDSL")


def _expert_fc1_w_cd(t: tuple[torch.Tensor, ...], core) -> torch.Tensor:
    """(up, gate) packed nibbles -> one expert's FC1 operand in the CuteDSL layout."""
    return _expert_fc1_w(t, _CUTEDSL_LAYOUT)


def _expert_fc1_s_cd(t: tuple[torch.Tensor, ...], core) -> torch.Tensor:
    """(up, gate) e4m3 scales -> one expert's FC1 scale operand in the CuteDSL layout."""
    return _expert_fc1_s(t, _CUTEDSL_LAYOUT)


def _expert_fc2_s_cd(t: torch.Tensor, core) -> torch.Tensor:
    """`down_proj` e4m3 scales -> one expert's FC2 scale operand in the CuteDSL layout."""
    return _expert_fc2_s(t, _CUTEDSL_LAYOUT)


def _mtp_fc1_w(t: tuple[torch.Tensor, ...], core) -> torch.Tensor:
    """(up, gate) e4m3 codes -> one draft expert's FC1 operand, `[up ; gate]` on
    the row axis and nothing else. The FP8 block-scale runner reads the
    checkpoint's plain row-major layout."""
    up, gate = t
    return torch.cat([up, gate], dim=0).contiguous()


def _mtp_fc1_s(t: tuple[torch.Tensor, ...], core) -> torch.Tensor:
    """(up, gate) 128x128 block scales -> one draft expert's FC1 scale operand:
    the same `[up ; gate]` concatenation, cast bf16 -> fp32."""
    up, gate = t
    return torch.cat([up, gate], dim=0).to(torch.float32).contiguous()


def _mtp_fc2_s(t: torch.Tensor, core) -> torch.Tensor:
    """`down_proj.weight_scale_inv` -> the FC2 scale operand: a bf16 -> fp32
    cast, no relayout."""
    return t.to(torch.float32).contiguous()


#: The layout argument of the draft experts' NVFP4 copy: the draft MoE runs it on trtllm-gen
#: whatever the trunk's routed backend is.
_TRTLLM_GEN_LAYOUT = SimpleNamespace(trunk_moe_backend="TRTLLM")

#: `448 * 6`: an NVFP4 global scale is this over the tensor's amax (modelopt's recipe).
NVFP4_AMAX_TO_GLOBAL = 448.0 * 6.0


def _fp8_block_dequant(codes: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """`[E, N, K]` e4m3 codes times their `[E, N/128, K/128]` fp32 block scales, in fp32."""
    s = scales.repeat_interleave(FP8_BLOCK, 1).repeat_interleave(FP8_BLOCK, 2)
    return codes.float() * s


def mtp_experts_to_nvfp4(fc1_w, fc1_s, fc2_w, fc2_s, chunk: int = 32) -> tuple:
    """Requantize the draft layer's FP8 128x128 block-scaled experts to NVFP4 in the
    trunk's trtllm-gen layout (`_expert_fc1_w/_s`, `_expert_fc2_w/_s`).

    Per expert and projection (FC1's up|gate share one, as the fused FC1's alpha
    requires): dequantize with the block scales, take the global scale
    `448 * 6 / amax`, then `fp4_quantize` (16-element blocks, e4m3 block scales,
    linear order), which then gets the trunk's row permutation and 128x4 swizzle.
    Returns `(fc1_w [E, 2I, H/2], fc1_s [E, 2I, H/16], fc2_w [E, H, I/2],
    fc2_s [E, H, I/16])` uint8 and the fp32 `[E]` weight global scales `(g1, g2)`;
    a weight's real value is its NVFP4 reconstruction over its global scale.
    `fc1_w` / `fc1_s` are the `[up ; gate]` stacks `_mtp_fc1_w` / `_mtp_fc1_s` build."""
    experts, two_i, hidden = fc1_w.shape
    inter = two_i // 2
    dev = fc1_w.device
    u8 = torch.uint8
    out = (
        torch.empty(experts, two_i, hidden // 2, dtype=u8, device=dev),
        torch.empty(experts, two_i, hidden // SF_VEC, dtype=u8, device=dev),
        torch.empty(experts, hidden, inter // 2, dtype=u8, device=dev),
        torch.empty(experts, hidden, inter // SF_VEC, dtype=u8, device=dev),
    )
    g1 = torch.empty(experts, dtype=torch.float32, device=dev)
    g2 = torch.empty(experts, dtype=torch.float32, device=dev)
    layout = _TRTLLM_GEN_LAYOUT
    for e0 in range(0, experts, chunk):
        e1 = min(e0 + chunk, experts)
        w1 = _fp8_block_dequant(fc1_w[e0:e1], fc1_s[e0:e1])
        w2 = _fp8_block_dequant(fc2_w[e0:e1], fc2_s[e0:e1])
        g1[e0:e1] = NVFP4_AMAX_TO_GLOBAL / w1.abs().amax(dim=(1, 2)).clamp_min(1e-30)
        g2[e0:e1] = NVFP4_AMAX_TO_GLOBAL / w2.abs().amax(dim=(1, 2)).clamp_min(1e-30)
        w1, w2 = w1.to(torch.bfloat16), w2.to(torch.bfloat16)
        for j, e in enumerate(range(e0, e1)):
            q, s = torch.ops.trtllm.fp4_quantize(w1[j], g1[e : e + 1], SF_VEC, False, False)
            s = s.view(two_i, hidden // SF_VEC)
            out[0][e] = _expert_fc1_w((q[:inter], q[inter:]), layout)
            out[1][e] = _expert_fc1_s((s[:inter], s[inter:]), layout)
            q, s = torch.ops.trtllm.fp4_quantize(w2[j], g2[e : e + 1], SF_VEC, False, False)
            out[2][e] = _expert_fc2_w(q, layout)
            out[3][e] = _expert_fc2_s(s.view(hidden, inter // SF_VEC), layout)
        del w1, w2
    return (*out, g1, g2)


def _materialize(entry) -> torch.Tensor:
    """Realize one checkpoint entry: `[:]` on a lazy safetensors slice, but a
    0-dim tensor (every NVFP4 per-tensor scalar on this checkpoint) rejects that
    index."""
    return entry if getattr(entry, "ndim", 1) == 0 else entry[:]


# --- which layers carry a weight ---


def _attn_layers(d):
    """The full-attention layers (`i % 4 == 3` on this checkpoint)."""
    return [i for i, a in enumerate(d.is_attn) if a]


def _gdn_layers(d):
    """The Gated-DeltaNet (linear-attention) layers."""
    return [i for i, a in enumerate(d.is_attn) if not a]


#: The Gated-DeltaNet input projections, in the row order of the concatenated
#: `in_proj_w` (and so the column order of its GEMM output).
_IN_PROJ_PARTS = ("in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b")


def _in_proj_rows(d, part: int) -> tuple[slice]:
    """The row range of `in_proj_w` that input projection `part` fills."""
    start = sum(d.in_proj_split[:part])
    return (slice(start, start + d.in_proj_split[part]),)


def _cute_dsl_copy_layers(d):
    """Every layer when the core keeps the second, CuteDSL-layout copy that prefill-sized
    launches read under trtllm-gen (`core.moe_prefill_copy`); none otherwise, CuteDSL among
    them, whose one copy is it."""
    return range(d.num_layers) if d.moe_prefill_copy else ()


def _vision_blocks(d):
    """The vision tower's blocks. Keyed `l{b}_vis_*`: block `b`, not decoder layer `b`."""
    return range(d.v_depth)


def _mtp_only(d):
    """The MTP module, at its own layer index `num_hidden_layers`, only when it is live."""
    return [d.num_layers] if d.mtp_enabled else []


# --- where the bytes come from ---


def _ple(suffix):
    """A PLE weight. The PLE layer's index is config, so the key is computed."""
    return lambda d, i: [(f"{_LM}.layers.{d.ple_layer}.ple.{suffix}", None)]


def _ngram_prefix(ple_layer: int) -> str:
    """The quantized n-gram table submodule: the 128 shards and the one
    per-tensor scale that dequantizes them both hang off this, one level below
    `ple_embedding` itself."""
    return f"{_LM}.layers.{ple_layer}.ple.ple_embedding.ngram_embedding"


def _visual(suffix):
    """A vision-tower weight that is not per-block."""
    return lambda d, i: [(f"{_VISUAL}.{suffix}", None)]


def _vision_block(suffix):
    """One vision block's weight."""
    return lambda d, i: [(f"{_VISUAL}.blocks.{i}.{suffix}", None)]


def _mtp_key(key):
    """One MTP-module tensor, at its own checkpoint key (the module's prefix is not the
    trunk's layer prefix)."""
    return lambda d, i: [(key, None)]


def _experts(prefix, *suffixes):
    """One source per routed expert, written at index `e`; a pair of suffixes hands the
    transform both tensors together."""

    def build(d, i):
        base = prefix(i)
        out = []
        for e in range(d.num_experts):
            keys = tuple(f"{base}.experts.{e}.{s}" for s in suffixes)
            out.append((keys if len(keys) > 1 else keys[0], (e,)))
        return out

    return build


def _trunk_mlp(i):
    return f"{_LM}.layers.{i}.mlp"


def _mtp_mlp(i):
    return f"{_MTP_LAYER}.mlp"


#: The six per-tensor NVFP4 scalars every routed expert carries, as
#: (parameter suffix, checkpoint suffix). The checkpoint stores modelopt
#: reciprocals; the core folds them into the runner's scalars after load.
_EXPERT_SCALARS = (
    ("e_isc1", "gate_proj.input_scale"),
    ("e_isc1_up", "up_proj.input_scale"),
    ("e_ws2_1", "gate_proj.weight_scale_2"),
    ("e_ws2_1_up", "up_proj.weight_scale_2"),
    ("e_isc2", "down_proj.input_scale"),
    ("e_ws2_2", "down_proj.weight_scale_2"),
)


#: (parameter tag, checkpoint module) of the two per-layer hyper-connection gates.
_HC_MODULES = (("ahc", "attn_hyper_connection"), ("mhc", "mlp_hyper_connection"))


def _hyper_connection(tag, src, layers):
    """The four weights of one gated-residual hyper-connection; `src(name)` names the source."""
    return (
        W(f"{tag}_norm", shape=lambda d: (d.stream,), src=src("hc_norm"), layers=layers),
        W(
            f"{tag}_down",
            shape=lambda d: (d.hc_lowrank, d.stream),
            src=src("input_mix_weight_down"),
            layers=layers,
        ),
        W(
            f"{tag}_up",
            shape=lambda d: (d.stream, d.hc_lowrank),
            src=src("input_mix_weight_up"),
            layers=layers,
        ),
        W(
            f"{tag}_inject",
            shape=lambda d: (d.hc_count, d.stream),
            src=src("block_inject_weight"),
            layers=layers,
        ),
    )


def _mixer(tag, prefix, layers):
    """A stack-final hyper-connection mixer: a layer's gate minus the injection weight."""
    src = _mtp_key if layers is not None else (lambda key: key)
    return (
        W(
            f"{tag}_norm",
            shape=lambda d: (d.stream,),
            src=src(f"{prefix}.hc_norm.weight"),
            layers=layers,
        ),
        W(
            f"{tag}_down",
            shape=lambda d: (d.hc_lowrank, d.stream),
            src=src(f"{prefix}.input_mix_weight_down.weight"),
            layers=layers,
        ),
        W(
            f"{tag}_up",
            shape=lambda d: (d.stream, d.hc_lowrank),
            src=src(f"{prefix}.input_mix_weight_up.weight"),
            layers=layers,
        ),
    )


def _attn_in_fused(tag, src, layers):
    """The trunk attention layers' one input projection: q_proj, k_proj, v_proj and the
    QSA indexer's index_qk_proj, re-laid by `_attn_in_w`."""
    keys = ("q_proj.weight", "k_proj.weight", "v_proj.weight", "indexer.index_qk_proj.weight")
    return (
        W(
            f"{tag}attn_in_w",
            shape=lambda d: (d.attn_in_rows, d.hidden),
            src=lambda d, i: [(tuple(src(k).format(p=f"{_LM}.layers.{i}") for k in keys), None)],
            transform=_attn_in_w,
            layers=layers,
        ),
    )


def _gated_attention(tag, src, layers, fused_in=False):
    """The gated GQA block's six weights. `q_proj` is twice as wide as q: q and the output
    gate interleave per head. With `fused_in`, q/k/v are not declared here: they are part
    of `_attn_in_fused`'s one operand."""
    return (
        *_attn_in_separate(tag, src, layers, fused_in),
        *_attn_out_and_norms(tag, src, layers),
    )


def _attn_in_separate(tag, src, layers, fused_in):
    if fused_in:
        return ()
    return (
        W(
            f"{tag}q_w",
            shape=lambda d: (2 * d.q_width, d.hidden),
            src=src("q_proj.weight"),
            layers=layers,
        ),
        W(
            f"{tag}k_w",
            shape=lambda d: (d.kv_width, d.hidden),
            src=src("k_proj.weight"),
            layers=layers,
        ),
        W(
            f"{tag}v_w",
            shape=lambda d: (d.kv_width, d.hidden),
            src=src("v_proj.weight"),
            layers=layers,
        ),
    )


def _attn_out_and_norms(tag, src, layers):
    return (
        W(
            f"{tag}o_w",
            shape=lambda d: (d.hidden, d.q_width),
            src=src("o_proj.weight"),
            layers=layers,
        ),
        W(f"{tag}q_norm", shape=lambda d: (d.head_dim,), src=src("q_norm.weight"), layers=layers),
        W(f"{tag}k_norm", shape=lambda d: (d.head_dim,), src=src("k_norm.weight"), layers=layers),
    )


def _index_branch(tag, src, layers, fused_in=False):
    """The QSA index branch's three weights: one replicated, unquantized projection
    producing `[index_heads q | 1 k]`, and two per-head gemma norms. With `fused_in`, the
    projection is part of `_attn_in_fused`'s one operand and only the norms are declared."""
    projection = W(
        f"{tag}idx_w",
        shape=lambda d: (d.index_width, d.hidden),
        src=src("index_qk_proj.weight"),
        layers=layers,
    )
    return (
        *(() if fused_in else (projection,)),
        W(
            f"{tag}idx_qn",
            shape=lambda d: (d.index_head_dim,),
            src=src("q_layernorm.weight"),
            layers=layers,
        ),
        W(
            f"{tag}idx_kn",
            shape=lambda d: (d.index_head_dim,),
            src=src("k_layernorm.weight"),
            layers=layers,
        ),
    )


def _shared_expert(tag, prefix, layers):
    """The bf16 shared expert beside the routed ones, its per-token gate in the same
    operand as its gate|up (`_cat_rows`)."""
    return (
        W(
            f"{tag}sh_gu",
            shape=lambda d: (2 * d.shared_inter + SH_GATE_ROWS, d.hidden),
            src=lambda d, i: [
                (
                    (
                        f"{prefix(i)}.shared_expert.gate_proj.weight",
                        f"{prefix(i)}.shared_expert.up_proj.weight",
                        f"{prefix(i)}.shared_expert_gate.weight",
                    ),
                    None,
                )
            ],
            transform=_cat_rows,
            layers=layers,
        ),
        W(
            f"{tag}sh_dn",
            shape=lambda d: (d.hidden, d.shared_inter),
            src=lambda d, i: [(f"{prefix(i)}.shared_expert.down_proj.weight", None)],
            layers=layers,
        ),
    )


u8, f8, f32, i64 = torch.uint8, torch.float8_e4m3fn, torch.float32, torch.int64


class Qwen4ExpTp1Weights:
    """Qwen3.8-Flash-Next-NVFP4 / sm_103 / tp1.

    The table, and the passes over it: `declare` allocates every weight it names,
    `manifest` maps checkpoint keys onto them, `load` fills them. `declare` and
    `manifest` share one "which parameters exist" decision (`_targets`), so they
    cannot disagree about a weight.
    """

    WEIGHTS: tuple[W, ...] = (
        # --- the two per-layer hyper-connection gates: every trunk layer ---
        *(
            w
            for tag, module in _HC_MODULES
            for w in _hyper_connection(
                tag, lambda name, m=module: f"{{p}}.{m}.{name}.weight", ALL_LAYERS
            )
        ),
        # --- the gated GQA layers and their QSA index branch ---
        # q/k/v and the indexer's replicated, unquantized [index_heads q | 1 k]
        # projection as one operand, then o_proj, the q/k norms and the indexer's two
        # per-head gemma norms.
        *_attn_in_fused("", lambda s: f"{{p}}.self_attn.{s}", _attn_layers),
        *_gated_attention("", lambda s: f"{{p}}.self_attn.{s}", _attn_layers, fused_in=True),
        *_index_branch("", lambda s: f"{{p}}.self_attn.indexer.{s}", _attn_layers, fused_in=True),
        # --- the Gated-DeltaNet layers ---
        # The four input projections read the same activation, so they are one
        # weight and one GEMM: rows `[q|k|v (conv_dim) | z (value_dim) | a | b]`
        # (`d.in_proj_split`), each checkpoint tensor copied into its own row
        # range. The forward splits the output into column views.
        W(
            "in_proj_w",
            shape=lambda d: (sum(d.in_proj_split), d.hidden),
            src=tuple(
                (f"{{p}}.linear_attn.{name}.weight", lambda d, j=j: _in_proj_rows(d, j))
                for j, name in enumerate(_IN_PROJ_PARTS)
            ),
            layers=_gdn_layers,
        ),
        W(
            "A_log",
            shape=lambda d: (d.num_v_heads,),
            src="{p}.linear_attn.A_log",
            layers=_gdn_layers,
        ),
        W(
            "dt_bias",
            shape=lambda d: (d.num_v_heads,),
            src="{p}.linear_attn.dt_bias",
            layers=_gdn_layers,
        ),
        # Stored [conv_dim, 1, width]; the conv entries want [conv_dim, width], which is
        # a zero-copy view the targets take once, after load.
        W(
            "conv_w",
            shape=lambda d: (d.conv_dim, 1, d.conv_kernel),
            src="{p}.linear_attn.conv1d.weight",
            layers=_gdn_layers,
        ),
        W(
            "gnorm",
            shape=lambda d: (d.head_v_dim,),
            src="{p}.linear_attn.norm.weight",
            layers=_gdn_layers,
        ),
        W(
            "out_w",
            shape=lambda d: (d.hidden, d.value_dim),
            src="{p}.linear_attn.out_proj.weight",
            layers=_gdn_layers,
        ),
        # --- MoE on every layer: NVFP4 routed experts, a bf16 router and shared expert ---
        W("router", shape=lambda d: (d.num_experts, d.hidden), src="{p}.mlp.gate.weight"),
        W(
            "fc1_w",
            shape=lambda d: (d.num_experts, 2 * d.moe_inter, d.hidden // 2),
            dtype=u8,
            src=_experts(_trunk_mlp, "up_proj.weight", "gate_proj.weight"),
            transform=_expert_fc1_w,
        ),
        W(
            "fc1_s",
            shape=lambda d: (d.num_experts, 2 * d.moe_inter, d.hidden // SF_VEC),
            dtype=u8,
            src=_experts(_trunk_mlp, "up_proj.weight_scale", "gate_proj.weight_scale"),
            transform=_expert_fc1_s,
        ),
        W(
            "fc2_w",
            shape=lambda d: (d.num_experts, d.hidden, d.moe_inter // 2),
            dtype=u8,
            src=_experts(_trunk_mlp, "down_proj.weight"),
            transform=_expert_fc2_w,
        ),
        W(
            "fc2_s",
            shape=lambda d: (d.num_experts, d.hidden, d.moe_inter // SF_VEC),
            dtype=u8,
            src=_experts(_trunk_mlp, "down_proj.weight_scale"),
            transform=_expert_fc2_s,
        ),
        # Under trtllm-gen, the same experts again in the CuteDSL layout: prefill-sized
        # launches run on them (`modeling._CUTE_DSL_MOE_MIN_ROWS`).
        W(
            "fc1_w_cd",
            shape=lambda d: (d.num_experts, 2 * d.moe_inter, d.hidden // 2),
            dtype=u8,
            src=_experts(_trunk_mlp, "up_proj.weight", "gate_proj.weight"),
            transform=_expert_fc1_w_cd,
            layers=_cute_dsl_copy_layers,
        ),
        W(
            "fc1_s_cd",
            shape=lambda d: (d.num_experts, 2 * d.moe_inter, d.hidden // SF_VEC),
            dtype=u8,
            src=_experts(_trunk_mlp, "up_proj.weight_scale", "gate_proj.weight_scale"),
            transform=_expert_fc1_s_cd,
            layers=_cute_dsl_copy_layers,
        ),
        W(
            "fc2_w_cd",
            shape=lambda d: (d.num_experts, d.hidden, d.moe_inter // 2),
            dtype=u8,
            src=_experts(_trunk_mlp, "down_proj.weight"),
            layers=_cute_dsl_copy_layers,
        ),
        W(
            "fc2_s_cd",
            shape=lambda d: (d.num_experts, d.hidden, d.moe_inter // SF_VEC),
            dtype=u8,
            src=_experts(_trunk_mlp, "down_proj.weight_scale"),
            transform=_expert_fc2_s_cd,
            layers=_cute_dsl_copy_layers,
        ),
        *(
            W(name, shape=lambda d: (d.num_experts,), dtype=f32, src=_experts(_trunk_mlp, suffix))
            for name, suffix in _EXPERT_SCALARS
        ),
        *_shared_expert("", _trunk_mlp, ALL_LAYERS),
        # --- not per-layer: the stack-final mixer, which is also the final norm ---
        *_mixer("fhc", f"{_LM}.hyper_connection_mixer", None),
        W(
            "embed",
            shape=lambda d: (d.vocab, d.hidden),
            src=f"{_LM}.embed_tokens.weight",
            layers=None,
        ),
        # --- PLE, on its one decoder layer (the n-gram table itself is declared at load) ---
        W(
            "ple_key_w",
            shape=lambda d: (d.stream, d.hidden),
            src=_ple("key_proj.weight"),
            layers=None,
        ),
        W(
            "ple_val_w",
            shape=lambda d: (d.ple_embed_dim, d.hidden),
            src=_ple("value_proj.weight"),
            layers=None,
        ),
        W("ple_nk", shape=lambda d: (d.stream,), src=_ple("norm_key.weight"), layers=None),
        W("ple_nq", shape=lambda d: (d.stream,), src=_ple("norm_query.weight"), layers=None),
        W("ple_nc", shape=lambda d: (d.stream,), src=_ple("norm_conv.weight"), layers=None),
        W(
            "ple_conv_w",
            shape=lambda d: (d.stream, 1, d.ple_conv_kernel),
            src=_ple("conv1d.weight"),
            layers=None,
        ),
        W(
            "ple_mult",
            shape=lambda d: (d.ngram_size,),
            dtype=i64,
            src=_ple("ple_embedding.layer_multipliers"),
            layers=None,
        ),
        W(
            "ple_vocab",
            shape=lambda d: (d.ngram_heads,),
            dtype=i64,
            src=_ple("ple_embedding.ngram_heads_vocab_sizes"),
            layers=None,
        ),
        W(
            "ple_offs",
            shape=lambda d: (d.ngram_heads,),
            dtype=i64,
            src=_ple("ple_embedding.ngram_heads_offsets"),
            layers=None,
        ),
        # The table's per-tensor scale sits with the **table**, one level below the three
        # metadata tensors above: the shards and this scale are the quantized
        # `ngram_embedding` submodule, while the multipliers, offsets and vocabulary sizes
        # describe the id space and stay on `ple_embedding` itself. The two groups also
        # live in different safetensors files, which is the other way to notice the split.
        W(
            "ple_tscale",
            shape=lambda d: (1,),
            src=lambda d, i: [(f"{_ngram_prefix(d.ple_layer)}.weight_scale", None)],
            layers=None,
        ),
        # --- the vision tower: all bf16, one-to-one, no transform ---
        W(
            "vis_patch_w",
            shape=lambda d: (d.v_hidden, d.v_channels, d.v_tpatch, d.v_patch, d.v_patch),
            src=_visual("patch_embed.proj.weight"),
            layers=None,
        ),
        W(
            "vis_patch_b",
            shape=lambda d: (d.v_hidden,),
            src=_visual("patch_embed.proj.bias"),
            layers=None,
        ),
        W(
            "vis_pos",
            shape=lambda d: (d.v_pos_rows, d.v_hidden),
            src=_visual("pos_embed.weight"),
            layers=None,
        ),
        # LayerNorm **with a bias**, unlike every norm in the language tower.
        W(
            "vis_n1_w",
            shape=lambda d: (d.v_hidden,),
            src=_vision_block("norm1.weight"),
            layers=_vision_blocks,
        ),
        W(
            "vis_n1_b",
            shape=lambda d: (d.v_hidden,),
            src=_vision_block("norm1.bias"),
            layers=_vision_blocks,
        ),
        # One fused `[q | k | v]` projection, MHA, with a bias.
        W(
            "vis_qkv_w",
            shape=lambda d: (3 * d.v_hidden, d.v_hidden),
            src=_vision_block("attn.qkv.weight"),
            layers=_vision_blocks,
        ),
        W(
            "vis_qkv_b",
            shape=lambda d: (3 * d.v_hidden,),
            src=_vision_block("attn.qkv.bias"),
            layers=_vision_blocks,
        ),
        W(
            "vis_proj_w",
            shape=lambda d: (d.v_hidden, d.v_hidden),
            src=_vision_block("attn.proj.weight"),
            layers=_vision_blocks,
        ),
        W(
            "vis_proj_b",
            shape=lambda d: (d.v_hidden,),
            src=_vision_block("attn.proj.bias"),
            layers=_vision_blocks,
        ),
        W(
            "vis_n2_w",
            shape=lambda d: (d.v_hidden,),
            src=_vision_block("norm2.weight"),
            layers=_vision_blocks,
        ),
        W(
            "vis_n2_b",
            shape=lambda d: (d.v_hidden,),
            src=_vision_block("norm2.bias"),
            layers=_vision_blocks,
        ),
        # A **non-gated** MLP: nothing here is split in half, unlike every SwiGLU in the
        # language tower.
        W(
            "vis_fc1_w",
            shape=lambda d: (d.v_inter, d.v_hidden),
            src=_vision_block("mlp.linear_fc1.weight"),
            layers=_vision_blocks,
        ),
        W(
            "vis_fc1_b",
            shape=lambda d: (d.v_inter,),
            src=_vision_block("mlp.linear_fc1.bias"),
            layers=_vision_blocks,
        ),
        W(
            "vis_fc2_w",
            shape=lambda d: (d.v_hidden, d.v_inter),
            src=_vision_block("mlp.linear_fc2.weight"),
            layers=_vision_blocks,
        ),
        W(
            "vis_fc2_b",
            shape=lambda d: (d.v_hidden,),
            src=_vision_block("mlp.linear_fc2.bias"),
            layers=_vision_blocks,
        ),
        # The patch merger. Its norm is applied **before** the 4-to-1 shuffle, so it is
        # `hidden` wide and not `merge_dim` wide.
        W(
            "vis_m_n_w",
            shape=lambda d: (d.v_hidden,),
            src=_visual("merger.norm.weight"),
            layers=None,
        ),
        W("vis_m_n_b", shape=lambda d: (d.v_hidden,), src=_visual("merger.norm.bias"), layers=None),
        W(
            "vis_m_fc1_w",
            shape=lambda d: (d.v_merge_dim, d.v_merge_dim),
            src=_visual("merger.linear_fc1.weight"),
            layers=None,
        ),
        W(
            "vis_m_fc1_b",
            shape=lambda d: (d.v_merge_dim,),
            src=_visual("merger.linear_fc1.bias"),
            layers=None,
        ),
        W(
            "vis_m_fc2_w",
            shape=lambda d: (d.hidden, d.v_merge_dim),
            src=_visual("merger.linear_fc2.weight"),
            layers=None,
        ),
        W(
            "vis_m_fc2_b",
            shape=lambda d: (d.hidden,),
            src=_visual("merger.linear_fc2.bias"),
            layers=None,
        ),
        # --- the MTP draft module, at layer index num_hidden_layers, only when live ---
        #
        # Structurally one more decoder layer of this architecture with a two-branch
        # front end; every shape equals a trunk full-attention layer's except the front
        # end and the expert quantization (FP8 128x128 block scales, not NVFP4).
        W(
            "mtp_pre_e_norm",
            shape=lambda d: (d.hidden,),
            src=_mtp_key(f"{_MTP}.pre_fc_norm_embedding.weight"),
            layers=_mtp_only,
        ),
        # `[hc_count * hidden]`: the module's input is the trunk's four-stream residual.
        W(
            "mtp_pre_h_norm",
            shape=lambda d: (d.stream,),
            src=_mtp_key(f"{_MTP}.pre_fc_norm_hidden.weight"),
            layers=_mtp_only,
        ),
        W(
            "mtp_fc_e",
            shape=lambda d: (d.hidden, d.hidden),
            src=_mtp_key(f"{_MTP}.fc_embedding.weight"),
            layers=_mtp_only,
        ),
        W(
            "mtp_fc_h",
            shape=lambda d: (d.hidden, d.hidden),
            src=_mtp_key(f"{_MTP}.fc_hidden.weight"),
            layers=_mtp_only,
        ),
        *(
            w
            for tag, module in _HC_MODULES
            for w in _hyper_connection(
                f"mtp_{tag}",
                lambda name, m=module: _mtp_key(f"{_MTP_LAYER}.{m}.{name}.weight"),
                _mtp_only,
            )
        ),
        *_gated_attention("mtp_", lambda s: _mtp_key(f"{_MTP_LAYER}.self_attn.{s}"), _mtp_only),
        *_index_branch(
            "mtp_", lambda s: _mtp_key(f"{_MTP_LAYER}.self_attn.indexer.{s}"), _mtp_only
        ),
        W(
            "mtp_router",
            shape=lambda d: (d.num_experts, d.hidden),
            src=_mtp_key(f"{_MTP_LAYER}.mlp.gate.weight"),
            layers=_mtp_only,
        ),
        # e4m3 codes stored row-major with one fp32 scale per 128x128 weight block, the
        # FC1 halves concatenated `[up ; gate]` on the row axis.
        W(
            "mtp_fc1_w",
            shape=lambda d: (d.num_experts, 2 * d.moe_inter, d.hidden),
            dtype=f8,
            src=_experts(_mtp_mlp, "up_proj.weight", "gate_proj.weight"),
            transform=_mtp_fc1_w,
            layers=_mtp_only,
        ),
        W(
            "mtp_fc1_s",
            shape=lambda d: (
                d.num_experts,
                (2 * d.moe_inter) // FP8_BLOCK,
                d.hidden // FP8_BLOCK,
            ),
            dtype=f32,
            src=_experts(_mtp_mlp, "up_proj.weight_scale_inv", "gate_proj.weight_scale_inv"),
            transform=_mtp_fc1_s,
            layers=_mtp_only,
        ),
        # FC2 needs no relayout at all: the checkpoint's `[hidden, I]` e4m3 block is
        # exactly the operand.
        W(
            "mtp_fc2_w",
            shape=lambda d: (d.num_experts, d.hidden, d.moe_inter),
            dtype=f8,
            src=_experts(_mtp_mlp, "down_proj.weight"),
            layers=_mtp_only,
        ),
        W(
            "mtp_fc2_s",
            shape=lambda d: (d.num_experts, d.hidden // FP8_BLOCK, d.moe_inter // FP8_BLOCK),
            dtype=f32,
            src=_experts(_mtp_mlp, "down_proj.weight_scale_inv"),
            transform=_mtp_fc2_s,
            layers=_mtp_only,
        ),
        *_shared_expert("mtp_", _mtp_mlp, _mtp_only),
        # The module's own stack-final mixer, which is its `shared_head`.
        *_mixer("mtp_fhc", f"{_MTP}.hyper_connection_mixer", _mtp_only),
    )

    def dims(self, core) -> SimpleNamespace:
        """Shapes are read off the normalized text config and the composite's vision
        config, which the core holds as `vision_config` before it declares anything."""
        cfg = core.model_config.pretrained_config
        vcfg = core.vision_config
        hidden = cfg.hidden_size
        index_heads, index_kv_heads = cfg.indexer_n_heads, cfg.indexer_kv_heads
        num_k_heads, num_v_heads = cfg.linear_num_key_heads, cfg.linear_num_value_heads
        key_dim = num_k_heads * cfg.linear_key_head_dim
        value_dim = num_v_heads * cfg.linear_value_head_dim
        ngram_heads = cfg.heads_per_ngram * (cfg.ngram_size - 1)
        v_hidden = vcfg.hidden_size
        return SimpleNamespace(
            num_layers=cfg.num_hidden_layers,
            is_attn=[t == "full_attention" for t in cfg.layer_types],
            hidden=hidden,
            hc_count=cfg.hc_count,
            hc_lowrank=cfg.hc_lowrank,
            stream=cfg.hc_count * hidden,
            head_dim=cfg.head_dim,
            q_width=cfg.num_attention_heads * cfg.head_dim,
            kv_width=cfg.num_key_value_heads * cfg.head_dim,
            index_head_dim=cfg.indexer_head_dim,
            index_width=(index_heads + index_kv_heads) * cfg.indexer_head_dim,
            # [q | k | v | gate | index q, k | pad of one index head]
            attn_in_rows=2 * cfg.num_attention_heads * cfg.head_dim
            + 2 * cfg.num_key_value_heads * cfg.head_dim
            + (index_heads + index_kv_heads + 1) * cfg.indexer_head_dim,
            num_v_heads=num_v_heads,
            head_v_dim=cfg.linear_value_head_dim,
            conv_dim=2 * key_dim + value_dim,
            value_dim=value_dim,
            in_proj_split=(2 * key_dim + value_dim, value_dim, num_v_heads, num_v_heads),
            conv_kernel=cfg.linear_conv_kernel_dim,
            num_experts=cfg.num_experts,
            moe_inter=cfg.moe_intermediate_size,
            shared_inter=cfg.shared_expert_intermediate_size,
            vocab=cfg.vocab_size,
            ple_layer=list(cfg.ple_layer_ids)[0] - 1,  # one-based in the config
            ple_embed_dim=cfg.ple_embed_dim,
            ple_conv_kernel=cfg.ple_conv_kernel_size,
            ngram_size=cfg.ngram_size,
            ngram_heads=ngram_heads,
            v_depth=vcfg.depth,
            v_hidden=v_hidden,
            v_inter=vcfg.intermediate_size,
            v_channels=vcfg.in_channels,
            v_tpatch=vcfg.temporal_patch_size,
            v_patch=vcfg.patch_size,
            v_pos_rows=vcfg.num_position_embeddings,
            v_merge_dim=v_hidden * vcfg.spatial_merge_size * vcfg.spatial_merge_size,
            mtp_enabled=getattr(core.model_config, "spec_config", None) is not None,
            trunk_moe_backend=core.trunk_moe_backend,
            moe_prefill_copy=core.moe_prefill_copy,
            dtype=core.model_config.torch_dtype,
        )

    def expected_unconsumed(self, core, weights) -> set:
        """With MTP off, the whole `mtp.` subtree; with it on, nothing, so that any
        unloaded `mtp.*` key fails the coverage assert instead of hiding."""
        if core.mtp_enabled:
            return set()
        return {k for k in weights if k.startswith(f"{_MTP}.")}

    def declare(self, d: SimpleNamespace) -> nn.ParameterDict:
        """Allocate every weight the table declares.

        Meta-init intercepts `torch.empty` here -- real CUDA storage arrives
        when the engine materializes the registry.
        """
        w = nn.ParameterDict()
        for entry in self.WEIGHTS:
            dtype = entry.dtype or d.dtype
            for key, _, _ in self._targets(entry, d):
                w[key] = nn.Parameter(
                    torch.empty(*entry.shape(d), dtype=dtype), requires_grad=False
                )
        return w

    def _targets(self, entry: W, d: SimpleNamespace) -> list[tuple[str, int | None, str]]:
        """`[(param key, layer index | None, checkpoint layer prefix), ...]`.

        One place decides which parameters an entry declares, so `declare` and
        `manifest` cannot disagree about whether a weight exists.
        """
        layers = entry.layer_indices(d)
        if layers is None:
            return [(entry.name, None, "")]
        return [(f"l{i}_{entry.name}", i, f"{_LM}.layers.{i}") for i in layers]

    def manifest(self, core) -> dict[str, list[tuple[Any, Any, Callable | None]]]:
        """param key -> [(checkpoint key, index into the param | None, transform | None)]."""
        d = self.dims(core)
        rows: dict[str, list[tuple[Any, Any, Callable | None]]] = {}
        for entry in self.WEIGHTS:
            for key, i, prefix in self._targets(entry, d):
                rows[key] = [
                    (ckpt, index, entry.transform) for ckpt, index in entry.sources(d, i, prefix)
                ]
        return rows

    def load(self, model, weights) -> None:
        """Fill every declared parameter. The n-gram table is declared here first,
        multi-key rows feed their transform several tensors at once, the per-tensor
        scalars are 0-dim and reject `[:]`, and coverage is asserted in both
        directions."""
        core = model.model
        # The one load-time declaration, before the manifest is built: the table's
        # height comes from the shards, and the manifest addresses it by row range.
        shard_keys = _declare_ngram_table(core, weights)
        manifest = self.manifest(core)
        manifest["ple_table"] = [
            (key, (slice(s * core.ple_rows_per_shard, (s + 1) * core.ple_rows_per_shard),), None)
            for s, key in enumerate(shard_keys)
        ]
        consumed: set = set()

        def fill(param: nn.Parameter, ckpt_key, index, transform) -> None:
            keys = ckpt_key if isinstance(ckpt_key, tuple) else (ckpt_key,)
            for key in keys:
                assert key in weights, f"checkpoint key missing: {key}"
            dst = param.data if index is None else param.data[index]
            # Materialize where the destination lives: the expert relayouts are row
            # gathers and the scale swizzle is a kernel, both on the device.
            src_all = tuple(
                _materialize(weights[key]).to(dst.device, non_blocking=True) for key in keys
            )
            if transform is not None:
                src = transform(src_all if isinstance(ckpt_key, tuple) else src_all[0], core)
            else:
                assert len(src_all) == 1, "a multi-key row needs a transform"
                src = src_all[0]
            assert dst.shape == src.shape, (ckpt_key, tuple(dst.shape), tuple(src.shape))
            assert src.dtype == dst.dtype, (ckpt_key, src.dtype, dst.dtype)
            dst.copy_(src, non_blocking=True)
            consumed.update(keys)

        assert set(manifest.keys()) == set(core.w.keys()), (
            "manifest/parameter drift",
            set(manifest.keys()) ^ set(core.w.keys()),
        )
        for param_key, sources in manifest.items():
            for ckpt_key, index, transform in sources:
                fill(core.w[param_key], ckpt_key, index, transform)
            if param_key.endswith(_RELEASE_AFTER):
                torch.cuda.empty_cache()

        # Shell-registered exception: the base class owns lm_head (untied), and this
        # checkpoint leaves it bf16, so it is a single dense weight. Asserted rather than
        # adapted — a shape change is a different logits path.
        assert tuple(model.lm_head.weight.shape) == (core.vocab, core.hidden), (
            f"the shell built lm_head {tuple(model.lm_head.weight.shape)} instead "
            f"of {(core.vocab, core.hidden)}"
        )
        fill(model.lm_head.weight, "lm_head.weight", None, None)
        # Parameter-side coverage, the other direction: name every registered parameter
        # outside the target's own ParameterDict instead of assuming lm_head is the only
        # one.
        shell = {n for n, _ in model.named_parameters() if not n.startswith("model.w.")}
        assert shell == {"lm_head.weight"}, f"unfed shell parameters: {sorted(shell)}"

        torch.cuda.synchronize()
        leftover = set(weights.keys()) - consumed
        expected = self.expected_unconsumed(core, weights)
        what = "empty" if core.mtp_enabled else "the MTP module"
        assert leftover == expected, (
            f"checkpoint coverage: leftover keys are not exactly {what}; "
            f"unexpected {sorted(leftover - expected)[:8]}, "
            f"missed {sorted(expected - leftover)[:8]}"
        )


MODEL_WEIGHTS = Qwen4ExpTp1Weights()


def _declare_ngram_table(core, weights) -> list[str]:
    """Register the n-gram table with the height its shards concatenate to, and return
    the shard keys in concatenation order.

    The shards are **contiguous row chunks**, not an interleaving: global row `r` lives in
    shard `r // rows_per_shard` at local row `r % rows_per_shard`. That is what makes the
    flat buffer indexable by the hash's global rows.

    **This is the whole of PLE host offload.** `core.ple_host_offload`, resolved at
    construction from `TRTLLM_MODELING_V2_PLE_HOST_OFFLOAD`, picks between an ordinary
    device allocation and host memory the device addresses. Nothing else changes: same
    shape, same dtype, same `ParameterDict` key, same rows filling it shard by shard, same
    gather in the forward. An offloaded tensor is **loaded, not skipped** — it stays
    fully covered in both directions. What the manifest cannot record is *where* it
    landed, so the core asserts that separately against the driver.
    """
    prefix = _ngram_prefix(core.ple_layer)
    keys = [f"{prefix}.shard_{s}.weight" for s in range(core.ple_shards)]
    for key in keys:
        assert key in weights, f"checkpoint key missing: {key}"
    shapes = {tuple(weights[key].shape) for key in keys}
    assert len(shapes) == 1, (
        f"the n-gram table's {len(keys)} shards do not share one shape: "
        f"{sorted(shapes)}; they are concatenated as equal contiguous row "
        f"chunks and an uneven one would shift every row above it"
    )
    rows_per_shard, head_dim = shapes.pop()
    assert head_dim == core.ple_head_dim, (
        f"the n-gram table is {head_dim} wide but {core.ngram_heads} heads must "
        f"tile ple_embed_dim {core.ple_embed_dim}, i.e. {core.ple_head_dim}"
    )
    dtype = weights[keys[0]].dtype
    assert dtype == core.ple_table_dtype, (
        f"the n-gram table is stored {dtype}; this target declares "
        f"{core.ple_table_dtype} and does not convert it"
    )
    device = core.w["embed"].device
    rows = core.ple_shards * rows_per_shard
    if core.ple_host_offload:
        table, core._ple_table_host = _host_resident_empty(rows, head_dim, dtype, device)
    else:
        table = torch.empty(rows, head_dim, dtype=dtype, device=device)
        core._ple_table_host = None
    core.w["ple_table"] = nn.Parameter(table, requires_grad=False)
    # `nn.Parameter` is `_make_subclass`, which shares storage rather than copying — but
    # "rather than copying" is the entire feature here, so it is asserted instead of
    # assumed.
    assert core.w["ple_table"].data_ptr() == table.data_ptr(), (
        "wrapping the n-gram table in nn.Parameter moved its storage"
    )
    core.ple_rows_per_shard = rows_per_shard
    return keys
