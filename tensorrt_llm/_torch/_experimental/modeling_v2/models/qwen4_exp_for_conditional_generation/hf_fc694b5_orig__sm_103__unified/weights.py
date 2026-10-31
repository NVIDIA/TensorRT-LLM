# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Weight table and loader: Qwen3.8-Flash-Next-NVFP4 (hf-fc694b5_orig) / sm_103 / unified.

One `W` per weight role (shape, checkpoint source, the rank's cut, dtype, load-time
transform); `Qwen4ExpUnifiedWeights` unrolls the table into declaration, manifest and load.
Storage is HF `[out, in]`; GEMM views, grouped-norm slices, fp32 gating params and NVFP4
scalars are derived in `modeling.py` after load.

**Every weight is declared at this rank's share.** The shares are computed once, in
`rank_dims`, from the engine's `Mapping`:

* **Attention heads** are split over the attention TP group: the TP group, or one rank
  under attention DP. Rank `r` of four holds q heads `[6r, 6r + 6)`: their rows of
  `q_proj` (q and the output gate interleave per head, so a head's `2 * head_dim` rows
  are contiguous) and their input columns of `o_proj`. The two KV heads are repeated on
  neighbouring ranks: rank `r` holds head `r * 2 // 4`, the head `duplicate_kv_weight`
  gives it and the K/V pool it binds holds (`pool.pool_geometry`). The QSA indexer and
  the q/k norms are replicated. The MTP layer's attention is cut the same way.
* **Gated-DeltaNet heads** are split over `mamba_effective_tp_size` (the TP group, or
  one rank under attention DP), as the recurrent pools are: rank `r` holds key heads
  `[4r, 4r + 4)` and value heads `[12r, 12r + 12)`. Of `in_proj_qkv` and the depthwise
  `conv1d` it reads three row blocks, the q and k rows of its key heads and the v rows
  of its value heads, into one `[q | k | v]` operand: the channel order of the conv pool
  (`pool.check_conv_layout`). `in_proj_z`, `in_proj_a`, `in_proj_b`, `A_log` and
  `dt_bias` are cut by value head, `out_proj` by its value-head input columns.
* **Routed experts** are split over the MoE EP group (tep4, dep4): rank `r` holds the
  contiguous window `[128r, 128r + 128)`, NVFP4 codes and block scales. tp4 splits each
  expert's intermediate instead: rank `r` holds rows `[160r, 160r + 160)` of every expert's
  `up_proj` and `gate_proj` and the same columns of its `down_proj`, all 512 experts. 160
  is short of the layout the runner's operands need (the FC1 row shuffle and scale swizzle
  work in 128-row blocks), so each half is zero-extended to `expert_inter` = 256 rows (FC2
  to 256 columns) with zero codes and zero block scales before the relayout: SwiGLU of a
  zero row pair is zero and a zero FC2 column adds nothing, so the padding is exact.
* **The MTP layer's FP8 block-scale experts** are split by expert over the whole TP
  group at every member, `[128r, 128r + 128)`: the FP8 runner takes `H` and `I` in
  multiples of 128, which 640 / 4 is not.

Everything else is replicated: hyper-connections, the PLE block and its n-gram table,
the router, the shared expert, the embedding, the vision tower, the MTP layer's front
end, router, shared expert and mixer, and every routed expert's per-tensor scalars over
the whole routing space (one activation quantization feeds all 512 experts, so the core
checks their FC1 input scale across all of them, not across its window). `lm_head` is
the shell's: `load` reads the rows the shell built, vocab-parallel under attention TP,
whole under attention DP.

**A rank reads only its share.** Each split weight carries a cut, the rows or columns of
the checkpoint tensor this rank holds, applied to the checkpoint entry before it is
materialized: on a lazy safetensors slice that reads those bytes and nothing else, on an
mmapped tensor it is a view, and either way only the share is copied to the device. An
off-window expert is never read. The kernel relayouts then run on the share, never
before the cut: the 32-row block shuffle and the 128x4 scale swizzle permute within
blocks of rows, so cutting after them would cut through blocks.

Transforms:

* trunk attention input projection: `q_proj`, `k_proj`, `v_proj` and the indexer's
  `index_qk_proj` read the same activation, so they are one
  `[q | k | v | gate | index | pad, hidden]` operand and one GEMM (`_attn_in_w`), at this
  rank's heads: 4352 rows at attention TP, 14080 under attention DP. The q halves go
  first so the GEMM output's leading columns are the packed `[q | k | v]` the fused
  norm+rope rewrites in place, then the gate halves, the replicated index projection and
  `index_head_dim` zero rows (a row is a whole number of `head_dim` heads). The draft
  (MTP) layer keeps its separate projections;
* fused shared-expert gate|up: `[gate ; up ; gate logit ; pad]` (`_cat_rows`);
* NVFP4 routed-expert stacks, laid out for the trunk's routed backend
  (`trunk_moe_backend`): `[up ; gate]`, then 128x4 scale swizzle. trtllm-gen also
  interleaves FC1 row by row and 32-row block shuffles weight and scale bytes (FC2:
  shuffle only); CuteDSL interleaves FC1 in 64-row up/gate groups and leaves FC2 in
  checkpoint row order. Under trtllm-gen a second copy of the window in the CuteDSL
  layout (`fc*_cd`) is declared when the core keeps it (`moe_prefill_copy`);
* n-gram table: 128 row chunks concatenated into one `[rows, 160]` buffer (the hash
  emits global row indices). It is replicated, and host-resident by default
  (`_host_resident_empty`): one copy per rank.

The n-gram table is not in the table: its height is a checkpoint fact, so `load`
registers it with `torch.empty` (`_declare_ngram_table`) and adds its rows to the
manifest (`ngram_table_sources`).

Not loaded: `mtp.*` when there is no `speculative_config`. With one, the MTP module is
loaded at this rank's share, including the draft layer's own QSA indexer. MTP experts
are plain row-major FP8 128x128: `[up ; gate]` concat, bf16 `weight_scale_inv` cast to
fp32; `mtp_experts_to_nvfp4` requantizes a rank's window to NVFP4 after load.

`load(model, weights)` fills every declared parameter once and asserts coverage both
ways: every parameter written, and every checkpoint key consumed (a key read through a
cut is consumed) or predicted by `expected_unconsumed` -- the experts outside this
rank's windows, and the MTP module when there is no speculation.
"""

import os
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Callable

import torch
from cuda.bindings import runtime as cudart
from torch import nn

from tensorrt_llm._torch.pyexecutor.config_utils import mamba_effective_tp_size
from tensorrt_llm._utils import CUASSERT

from .pool import attention_tp_size

#: `layers=ALL_LAYERS` is the common case and the default. A sentinel rather
#: than a lambda so the table reads as data, and rather than `True` so the
#: field has one type of answer: which layers.
ALL_LAYERS = object()

#: An index into a checkpoint tensor: a tuple of slices, one per leading axis.
Cut = tuple[slice, ...]


@dataclass(frozen=True)
class W:
    """One weight role: what it is, where its bytes come from, what happens on the way.

    `shape` takes the dimension bundle and returns the declared shape: this rank's
    share, every width derived from the config and the mapping.

    `src` says where the bytes come from, in one of two forms:

    * a checkpoint key template, with `{p}` standing for the layer prefix (a key with no
      `{p}` is itself);
    * a callable `(d, i) -> [(key, index), ...]` or `[(key, index, cut), ...]`, taking
      the dimension bundle and the layer index. `index` addresses the destination
      parameter (None for all of it). A `key` may itself be a tuple, meaning the
      transform is handed those tensors together. A source that names its own `cut`
      reads one block of a tensor several of whose blocks land in the parameter.

    `cut` takes the dimension bundle and returns this rank's rows or columns of the
    checkpoint tensor, read from the entry before it is materialized; for a tuple key
    it returns one cut (or None) per key. `None` reads the tensor whole.

    `layers` says which layers carry this weight: `ALL_LAYERS` for every one,
    `None` for a weight that is not per-layer at all, or a callable
    `d -> iterable[int]` when only some layers have it.

    `transform` runs on the cut tensor(s) before the copy, on the destination's
    device, with the dimension bundle: the kernels' relayouts, with no slicing of
    their own. `None` means the cut is already the declared shape.
    """

    name: str
    shape: Callable[[Any], tuple[int, ...]]
    src: str | Callable
    dtype: torch.dtype | None = None
    cut: Callable[[Any], Any] | None = None
    transform: Callable | None = None
    layers: object | None = ALL_LAYERS

    def layer_indices(self, d: SimpleNamespace) -> list[int] | None:
        """The layers this weight exists on, or None if it is not per-layer."""
        if self.layers is None:
            return None
        if self.layers is ALL_LAYERS:
            return list(range(d.num_layers))
        return list(self.layers(d))

    def sources(self, d: SimpleNamespace, i: int | None, prefix: str) -> list[tuple]:
        """`[(checkpoint key or key tuple, index into the param | None, cut | None), ...]`."""
        rows = self.src(d, i) if callable(self.src) else [(self.src.format(p=prefix), None)]
        cut = self.cut(d) if self.cut is not None else None
        out = []
        for row in rows:
            if len(row) == 3:
                assert self.cut is None, f"{self.name}: a source with its own cut takes no W.cut"
                out.append(tuple(row))
            else:
                out.append((*row, cut))
        return out


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
        # it: with a process per GPU, half the ranks would otherwise gather from the other
        # socket's memory.
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

#: The PLE n-gram table's storage: the checkpoint's FP8 per-tensor quantization, kept as
#: stored (the gather dequantizes the rows it reads).
PLE_TABLE_DTYPE = torch.float8_e4m3fn

#: How the trunk's routed experts are split (`rank_dims`'s `expert_split`) that this table
#: declares and `load` fills: by expert (`ep`, a window of whole experts) or by each
#: expert's intermediate (`tp`, every expert at a zero-padded slice).
LOADED_EXPERT_SPLITS = ("ep", "tp")

#: The multiple a rank's stored routed-expert intermediate is padded up to. It is the
#: engine's own storage rule for an NVFP4 trtllm-gen shard
#: (`NVFP4TRTLLMGenFusedMoEMethod.resolve_alignments`: a padded intermediate that is not a
#: multiple of 128 drops to a 128 weight alignment), which turns tp4's 160 into 256, the
#: width `fp4_block_scale_moe_runner` certifies the padded 512-expert stack at. 640
#: (tep4, dep4) is already a multiple.
EXPERT_INTER_ALIGN = 128

# Parameter-key suffixes after which to release transform scratch. The expert
# relayouts allocate per-expert scratch once per local expert and the table
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


def _read(entry, cut: Cut | None) -> torch.Tensor:
    """Realize one checkpoint entry, reading only `cut` of it.

    `entry[cut]` on a lazy safetensors slice reads those rows or columns and nothing
    else; on a plain tensor it is a view. Uncut, `[:]` reads it whole -- but a 0-dim
    tensor (every NVFP4 per-tensor scalar on this checkpoint) rejects that index, and no
    0-dim entry has a cut."""
    if cut is None:
        return entry if getattr(entry, "ndim", 1) == 0 else entry[:]
    return entry[cut[0]] if len(cut) == 1 else entry[cut]


# --- cuts: this rank's block of rows or columns of a checkpoint tensor ---


def _span(start: int, width: int) -> slice:
    return slice(start, start + width)


def _q_rows(d) -> Cut:
    """`q_proj` rows of this rank's q heads: each head's q rows, then its gate rows."""
    return (_span(d.q_head_start * 2 * d.head_dim, 2 * d.q_width),)


def _kv_rows(d) -> Cut:
    """`k_proj` / `v_proj` rows of this rank's KV heads."""
    return (_span(d.kv_head_start * d.head_dim, d.kv_width),)


def _o_cols(d) -> Cut:
    """`o_proj` input columns of this rank's q heads."""
    return (slice(None), _span(d.q_head_start * d.head_dim, d.q_width))


def _key_rows(d, section: int) -> Cut:
    """Rows of this rank's key heads in section `section` (0 q, 1 k) of a `[q | k | v]`
    Gated-DeltaNet tensor."""
    return (_span(section * d.key_dim_total + d.k_head_start * d.head_k_dim, d.key_dim),)


def _value_rows(d, base: int = 0) -> Cut:
    """Rows of this rank's value heads, `base` rows into the tensor."""
    return (_span(base + d.v_head_start * d.head_v_dim, d.value_dim),)


def _value_head_rows(d) -> Cut:
    """One row per value head (`in_proj_a`, `in_proj_b`, `A_log`, `dt_bias`)."""
    return (_span(d.v_head_start, d.num_v_heads),)


def _value_cols(d) -> Cut:
    """`out_proj` input columns of this rank's value heads."""
    return (slice(None), _span(d.v_head_start * d.head_v_dim, d.value_dim))


def _qkv_blocks(d) -> tuple[tuple[slice, Cut], ...]:
    """(destination rows, checkpoint cut) of the three blocks of a `[q | k | v]`
    Gated-DeltaNet tensor this rank holds, in the `[q | k | v]` order of its conv pool."""
    return (
        (slice(0, d.key_dim), _key_rows(d, 0)),
        (slice(d.key_dim, 2 * d.key_dim), _key_rows(d, 1)),
        (slice(2 * d.key_dim, d.conv_dim), _value_rows(d, 2 * d.key_dim_total)),
    )


# --- relayouts, on the rank's cut ---


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


def _fc1_perm(rows: int, device, d) -> torch.Tensor:
    """The FC1 row gather of the trunk's routed backend. trtllm-gen: interleave then
    32-row block shuffle, composed into one gather. CuteDSL: the 64-row group
    interleave alone."""
    if d.trunk_moe_backend == "CUTEDSL":
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


def _cat_rows(t: tuple[torch.Tensor, ...], d) -> torch.Tensor:
    """Fused shared-expert weight bytes: **gate** rows first, the half order
    `flashinfer_silu_and_mul` reads, then up, then the per-token gate's row and
    `SH_GATE_ROWS - 1` zero rows -- one GEMM forms gate|up and the gate logit."""
    gate, up, sh_gate = t
    pad = sh_gate.new_zeros(SH_GATE_ROWS - 1, sh_gate.shape[-1])
    return torch.cat((gate, up, sh_gate, pad), dim=0).contiguous()


def _attn_in_w(t: tuple[torch.Tensor, ...], d) -> torch.Tensor:
    """(q_proj, k_proj, v_proj, index_qk_proj), each at this rank's cut -> the fused
    `[q | k | v | gate | index | pad]` input projection of one trunk attention layer (see
    the module docstring).

    `q_proj`'s rows are `[heads, 2, head_dim]`: head `h`'s q rows, then its gate rows.
    """
    q, k, v, idx = t
    heads, head_dim, hidden = d.heads_q, d.head_dim, q.shape[-1]
    qg = q.reshape(heads, 2, head_dim, hidden)
    pad = q.new_zeros(d.index_head_dim, hidden)
    rows = (qg[:, 0].reshape(heads * head_dim, hidden), k, v)
    rows += (qg[:, 1].reshape(heads * head_dim, hidden), idx, pad)
    return torch.cat(rows, dim=0).contiguous()


def _expert_fc1_w(t: tuple[torch.Tensor, ...], d) -> torch.Tensor:
    """(up, gate) packed nibbles -> one expert's FC1 operand."""
    up, gate = t
    w = torch.cat([up, gate], dim=0)
    return torch.index_select(w, 0, _fc1_perm(w.shape[0], w.device, d)).contiguous()


def _expert_fc1_s(t: tuple[torch.Tensor, ...], d) -> torch.Tensor:
    """(up, gate) e4m3 scales -> one expert's FC1 scale operand: the weight's own row
    permutation, then the swizzle."""
    up, gate = t
    s = torch.cat([up.view(torch.uint8), gate.view(torch.uint8)], dim=0)
    rows, cols = s.shape
    s = torch.index_select(s, 0, _fc1_perm(rows, s.device, d))
    return _swizzle(s).reshape(rows, cols)


def _expert_fc2_w(t: torch.Tensor, d) -> torch.Tensor:
    """`down_proj` packed nibbles -> one expert's FC2 operand (no interleave): the
    32-row block shuffle on trtllm-gen, checkpoint row order on CuteDSL."""
    if d.trunk_moe_backend == "CUTEDSL":
        return t.contiguous()
    return torch.index_select(t, 0, _block32_perm(t.shape[0], t.device)).contiguous()


def _expert_fc2_s(t: torch.Tensor, d) -> torch.Tensor:
    """`down_proj` e4m3 scales -> one expert's FC2 scale operand: the weight's own row
    order, then the swizzle."""
    s = t.view(torch.uint8)
    rows, cols = s.shape
    if d.trunk_moe_backend != "CUTEDSL":
        s = torch.index_select(s, 0, _block32_perm(rows, s.device))
    return _swizzle(s).reshape(rows, cols)


def _zero_extend(x: torch.Tensor, dim: int, size: int) -> torch.Tensor:
    """`x` extended along `dim` to `size` with zero bytes: a zero packed byte is two +0.0
    e2m1 codes, a zero scale byte is an e4m3 +0.0 block scale."""
    if x.shape[dim] == size:
        return x
    assert x.shape[dim] < size, (tuple(x.shape), dim, size)
    raw = x.view(torch.uint8)
    pad = list(raw.shape)
    pad[dim] = size - raw.shape[dim]
    return torch.cat((raw, raw.new_zeros(pad)), dim).view(x.dtype)


def _zero_padded(transform: Callable | None, dim: int, unit: int) -> Callable:
    """`transform` run on its source(s) zero-extended along `dim` to the rank's stored
    routed-expert intermediate, `d.expert_inter // unit` there: unit 1 for the rows of an
    `up_proj` / `gate_proj` and of their scales, 2 for `down_proj`'s packed code columns,
    `SF_VEC` for its scale columns. Each half of FC1 is extended before the `[up ; gate]`
    concatenation, so the pad rows sit at the end of each half, where the runner's
    `intermediate_size` puts them. A no-op where the rank holds whole experts."""

    def padded(t, d):
        size = d.expert_inter // unit
        if isinstance(t, tuple):
            t = tuple(_zero_extend(x, dim, size) for x in t)
        else:
            t = _zero_extend(t, dim, size)
        return t if transform is None else transform(t, d)

    return padded


def _expert_inter_rows(d) -> Cut | None:
    """This rank's rows of one expert's `up_proj` or `gate_proj` (or their block scales):
    its slice of the intermediate, or None (whole) when the experts are split by expert."""
    if d.expert_inter_share == d.moe_inter:
        return None
    return (_span(d.expert_inter_start, d.expert_inter_share),)


def _fc1_cut(d):
    """The `(up, gate)` pair's cuts: the same intermediate rows of both halves."""
    rows = _expert_inter_rows(d)
    return None if rows is None else (rows, rows)


def _fc2_cut(unit: int) -> Callable:
    """`down_proj`'s input columns of this rank's intermediate slice, `unit` intermediate
    elements to a stored column (2 for the packed codes, `SF_VEC` for the block scales)."""

    def cut(d):
        if _expert_inter_rows(d) is None:
            return None
        return (
            slice(None),
            _span(d.expert_inter_start // unit, d.expert_inter_share // unit),
        )

    return cut


#: The layout argument of the second, CuteDSL-layout expert copy (`fc*_cd`).
_CUTEDSL_LAYOUT = SimpleNamespace(trunk_moe_backend="CUTEDSL")


def _expert_fc1_w_cd(t: tuple[torch.Tensor, ...], d) -> torch.Tensor:
    """(up, gate) packed nibbles -> one expert's FC1 operand in the CuteDSL layout."""
    return _expert_fc1_w(t, _CUTEDSL_LAYOUT)


def _expert_fc1_s_cd(t: tuple[torch.Tensor, ...], d) -> torch.Tensor:
    """(up, gate) e4m3 scales -> one expert's FC1 scale operand in the CuteDSL layout."""
    return _expert_fc1_s(t, _CUTEDSL_LAYOUT)


def _expert_fc2_s_cd(t: torch.Tensor, d) -> torch.Tensor:
    """`down_proj` e4m3 scales -> one expert's FC2 scale operand in the CuteDSL layout."""
    return _expert_fc2_s(t, _CUTEDSL_LAYOUT)


def _mtp_fc1_w(t: tuple[torch.Tensor, ...], d) -> torch.Tensor:
    """(up, gate) e4m3 codes -> one draft expert's FC1 operand, `[up ; gate]` on
    the row axis and nothing else. The FP8 block-scale runner reads the
    checkpoint's plain row-major layout."""
    up, gate = t
    return torch.cat([up, gate], dim=0).contiguous()


def _mtp_fc1_s(t: tuple[torch.Tensor, ...], d) -> torch.Tensor:
    """(up, gate) 128x128 block scales -> one draft expert's FC1 scale operand:
    the same `[up ; gate]` concatenation, cast bf16 -> fp32."""
    up, gate = t
    return torch.cat([up, gate], dim=0).to(torch.float32).contiguous()


def _mtp_fc2_s(t: torch.Tensor, d) -> torch.Tensor:
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
    """Requantize the draft layer's FP8 128x128 block-scaled experts (this rank's window)
    to NVFP4 in the trunk's trtllm-gen layout (`_expert_fc1_w/_s`, `_expert_fc2_w/_s`).

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


# --- which layers carry a weight ---


def _attn_layers(d):
    """The full-attention layers (`i % 4 == 3` on this checkpoint)."""
    return [i for i, a in enumerate(d.is_attn) if a]


def _gdn_layers(d):
    """The Gated-DeltaNet (linear-attention) layers."""
    return [i for i, a in enumerate(d.is_attn) if not a]


def _trunk_expert_layers(d):
    """Every layer when the routed experts are split in a way this table declares
    (`LOADED_EXPERT_SPLITS`); none otherwise."""
    return range(d.num_layers) if d.expert_split in LOADED_EXPERT_SPLITS else ()


def _cute_dsl_copy_layers(d):
    """The trunk expert layers when the core keeps the second, CuteDSL-layout copy that
    prefill-sized launches read under trtllm-gen (`moe_prefill_copy`); none otherwise."""
    return _trunk_expert_layers(d) if d.moe_prefill_copy else ()


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


def _trunk_window(d) -> tuple[int, int]:
    """(first expert, count) of this rank's routed-expert window."""
    return d.expert_offset, d.local_experts


def _mtp_window(d) -> tuple[int, int]:
    """(first expert, count) of this rank's window of the draft layer's experts."""
    return d.mtp_expert_offset, d.mtp_local_experts


def _routing_space(d) -> tuple[int, int]:
    """Every expert: what the replicated per-tensor scalars cover."""
    return 0, d.num_experts


def _experts(prefix, *suffixes, window):
    """One source per expert of `window(d)`, written at its local index; a pair of
    suffixes hands the transform both tensors together. The checkpoint key depends on
    the rank, which is why the source is computed rather than templated."""

    def build(d, i):
        start, count = window(d)
        base = prefix(i)
        out = []
        for local in range(count):
            keys = tuple(f"{base}.experts.{start + local}.{s}" for s in suffixes)
            out.append((keys if len(keys) > 1 else keys[0], (local,)))
        return out

    return build


def _trunk_mlp(i):
    return f"{_LM}.layers.{i}.mlp"


def _mtp_mlp(i):
    return f"{_MTP_LAYER}.mlp"


def _in_proj(d, i):
    """The Gated-DeltaNet input projections at this rank's heads, in the row order of
    the concatenated `in_proj_w` (and so the column order of its GEMM output):
    `[q | k | v (conv_dim) | z (value_dim) | a | b]`, `d.in_proj_split`."""
    p = f"{_LM}.layers.{i}.linear_attn"
    out = [(f"{p}.in_proj_qkv.weight", (rows,), cut) for rows, cut in _qkv_blocks(d)]
    row = d.conv_dim
    out.append((f"{p}.in_proj_z.weight", (_span(row, d.value_dim),), _value_rows(d)))
    row += d.value_dim
    for name in ("in_proj_a", "in_proj_b"):
        out.append((f"{p}.{name}.weight", (_span(row, d.num_v_heads),), _value_head_rows(d)))
        row += d.num_v_heads
    return out


def _conv(d, i):
    """The depthwise conv's `[q | k | v]` channels of this rank's heads."""
    key = f"{_LM}.layers.{i}.linear_attn.conv1d.weight"
    return [(key, (rows,), cut) for rows, cut in _qkv_blocks(d)]


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

#: The checkpoint tensors of one routed expert that are not per-tensor scalars: what a
#: rank outside its window leaves unread. The draft layer's FP8 experts carry
#: `weight_scale_inv` in place of the NVFP4 `weight_scale`.
_PROJECTIONS = ("gate_proj", "up_proj", "down_proj")
_TRUNK_EXPERT_TENSORS = tuple(f"{p}.{s}" for p in _PROJECTIONS for s in ("weight", "weight_scale"))
_MTP_EXPERT_TENSORS = tuple(
    f"{p}.{s}" for p in _PROJECTIONS for s in ("weight", "weight_scale_inv")
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
    return (
        W(
            f"{tag}_norm",
            shape=lambda d: (d.stream,),
            src=f"{prefix}.hc_norm.weight",
            layers=layers,
        ),
        W(
            f"{tag}_down",
            shape=lambda d: (d.hc_lowrank, d.stream),
            src=f"{prefix}.input_mix_weight_down.weight",
            layers=layers,
        ),
        W(
            f"{tag}_up",
            shape=lambda d: (d.stream, d.hc_lowrank),
            src=f"{prefix}.input_mix_weight_up.weight",
            layers=layers,
        ),
    )


def _attn_in_fused(d, i):
    """The trunk attention layers' one input projection: q_proj, k_proj and v_proj at this
    rank's heads and the QSA indexer's replicated index_qk_proj, re-laid by `_attn_in_w`."""
    p = f"{_LM}.layers.{i}.self_attn"
    keys = (
        f"{p}.q_proj.weight",
        f"{p}.k_proj.weight",
        f"{p}.v_proj.weight",
        f"{p}.indexer.index_qk_proj.weight",
    )
    return [(keys, None, (_q_rows(d), _kv_rows(d), _kv_rows(d), None))]


def _attn_out_and_norms(tag, prefix, layers):
    """o_proj at this rank's q heads, and the replicated per-head q/k norms."""
    return (
        W(
            f"{tag}o_w",
            shape=lambda d: (d.hidden, d.q_width),
            src=f"{prefix}.o_proj.weight",
            cut=_o_cols,
            layers=layers,
        ),
        W(
            f"{tag}q_norm",
            shape=lambda d: (d.head_dim,),
            src=f"{prefix}.q_norm.weight",
            layers=layers,
        ),
        W(
            f"{tag}k_norm",
            shape=lambda d: (d.head_dim,),
            src=f"{prefix}.k_norm.weight",
            layers=layers,
        ),
    )


def _attn_in_separate(tag, prefix, layers):
    """The draft layer's separate q (with its interleaved gate), k and v projections, at
    this rank's heads."""
    return (
        W(
            f"{tag}q_w",
            shape=lambda d: (2 * d.q_width, d.hidden),
            src=f"{prefix}.q_proj.weight",
            cut=_q_rows,
            layers=layers,
        ),
        W(
            f"{tag}k_w",
            shape=lambda d: (d.kv_width, d.hidden),
            src=f"{prefix}.k_proj.weight",
            cut=_kv_rows,
            layers=layers,
        ),
        W(
            f"{tag}v_w",
            shape=lambda d: (d.kv_width, d.hidden),
            src=f"{prefix}.v_proj.weight",
            cut=_kv_rows,
            layers=layers,
        ),
    )


def _index_branch(tag, prefix, layers, fused_in=False):
    """The QSA index branch's three weights: one replicated, unquantized projection
    producing `[index_heads q | 1 k]`, and two per-head gemma norms. With `fused_in`, the
    projection is part of `_attn_in_fused`'s one operand and only the norms are declared."""
    projection = W(
        f"{tag}idx_w",
        shape=lambda d: (d.index_width, d.hidden),
        src=f"{prefix}.index_qk_proj.weight",
        layers=layers,
    )
    return (
        *(() if fused_in else (projection,)),
        W(
            f"{tag}idx_qn",
            shape=lambda d: (d.index_head_dim,),
            src=f"{prefix}.q_layernorm.weight",
            layers=layers,
        ),
        W(
            f"{tag}idx_kn",
            shape=lambda d: (d.index_head_dim,),
            src=f"{prefix}.k_layernorm.weight",
            layers=layers,
        ),
    )


def _shared_expert(tag, prefix, layers):
    """The replicated bf16 shared expert beside the routed ones, its per-token gate in the
    same operand as its gate|up (`_cat_rows`)."""
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


class Qwen4ExpUnifiedWeights:
    """Qwen3.8-Flash-Next-NVFP4 / sm_103 / unified, at one rank's share.

    The table, and the passes over it: `declare` allocates every weight it names,
    `manifest` maps checkpoint keys and cuts onto them, `load` fills them. `declare` and
    `manifest` share one "which parameters exist" decision (`_targets`), so they
    cannot disagree about a weight.
    """

    WEIGHTS: tuple[W, ...] = (
        # --- the two per-layer hyper-connection gates: every trunk layer, replicated ---
        *(
            w
            for tag, module in _HC_MODULES
            for w in _hyper_connection(
                tag, lambda name, m=module: f"{{p}}.{m}.{name}.weight", ALL_LAYERS
            )
        ),
        # --- the gated GQA layers and their QSA index branch ---
        # q/k/v at this rank's heads and the indexer's replicated [index_heads q | 1 k]
        # projection as one operand, then o_proj at this rank's heads, the q/k norms and
        # the indexer's two per-head gemma norms.
        W(
            "attn_in_w",
            shape=lambda d: (d.attn_in_rows, d.hidden),
            src=_attn_in_fused,
            transform=_attn_in_w,
            layers=_attn_layers,
        ),
        *_attn_out_and_norms("", "{p}.self_attn", _attn_layers),
        *_index_branch("", "{p}.self_attn.indexer", _attn_layers, fused_in=True),
        # --- the Gated-DeltaNet layers, at this rank's key and value heads ---
        # The four input projections read the same activation, so they are one
        # weight and one GEMM: rows `[q|k|v (conv_dim) | z (value_dim) | a | b]`
        # (`d.in_proj_split`), each checkpoint block copied into its own row range.
        W(
            "in_proj_w",
            shape=lambda d: (sum(d.in_proj_split), d.hidden),
            src=_in_proj,
            layers=_gdn_layers,
        ),
        W(
            "A_log",
            shape=lambda d: (d.num_v_heads,),
            src="{p}.linear_attn.A_log",
            cut=_value_head_rows,
            layers=_gdn_layers,
        ),
        W(
            "dt_bias",
            shape=lambda d: (d.num_v_heads,),
            src="{p}.linear_attn.dt_bias",
            cut=_value_head_rows,
            layers=_gdn_layers,
        ),
        # Stored [conv_dim, 1, width]; the conv entries want [conv_dim, width], which is
        # a zero-copy view the targets take once, after load.
        W(
            "conv_w",
            shape=lambda d: (d.conv_dim, 1, d.conv_kernel),
            src=_conv,
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
            cut=_value_cols,
            layers=_gdn_layers,
        ),
        # --- MoE on every layer: this rank's window of NVFP4 routed experts, a replicated
        # bf16 router and shared expert ---
        W("router", shape=lambda d: (d.num_experts, d.hidden), src="{p}.mlp.gate.weight"),
        # Each expert at this rank's intermediate share (`_fc1_cut`, `_fc2_cut`), stored
        # zero-padded to `expert_inter` (`_zero_padded`); whole experts under an EP window.
        W(
            "fc1_w",
            shape=lambda d: (d.local_experts, 2 * d.expert_inter, d.hidden // 2),
            dtype=u8,
            src=_experts(_trunk_mlp, "up_proj.weight", "gate_proj.weight", window=_trunk_window),
            cut=_fc1_cut,
            transform=_zero_padded(_expert_fc1_w, 0, 1),
            layers=_trunk_expert_layers,
        ),
        W(
            "fc1_s",
            shape=lambda d: (d.local_experts, 2 * d.expert_inter, d.hidden // SF_VEC),
            dtype=u8,
            src=_experts(
                _trunk_mlp, "up_proj.weight_scale", "gate_proj.weight_scale", window=_trunk_window
            ),
            cut=_fc1_cut,
            transform=_zero_padded(_expert_fc1_s, 0, 1),
            layers=_trunk_expert_layers,
        ),
        W(
            "fc2_w",
            shape=lambda d: (d.local_experts, d.hidden, d.expert_inter // 2),
            dtype=u8,
            src=_experts(_trunk_mlp, "down_proj.weight", window=_trunk_window),
            cut=_fc2_cut(2),
            transform=_zero_padded(_expert_fc2_w, 1, 2),
            layers=_trunk_expert_layers,
        ),
        W(
            "fc2_s",
            shape=lambda d: (d.local_experts, d.hidden, d.expert_inter // SF_VEC),
            dtype=u8,
            src=_experts(_trunk_mlp, "down_proj.weight_scale", window=_trunk_window),
            cut=_fc2_cut(SF_VEC),
            transform=_zero_padded(_expert_fc2_s, 1, SF_VEC),
            layers=_trunk_expert_layers,
        ),
        # Under trtllm-gen, the same window again in the CuteDSL layout, when the core keeps
        # it for prefill-sized launches.
        W(
            "fc1_w_cd",
            shape=lambda d: (d.local_experts, 2 * d.expert_inter, d.hidden // 2),
            dtype=u8,
            src=_experts(_trunk_mlp, "up_proj.weight", "gate_proj.weight", window=_trunk_window),
            cut=_fc1_cut,
            transform=_zero_padded(_expert_fc1_w_cd, 0, 1),
            layers=_cute_dsl_copy_layers,
        ),
        W(
            "fc1_s_cd",
            shape=lambda d: (d.local_experts, 2 * d.expert_inter, d.hidden // SF_VEC),
            dtype=u8,
            src=_experts(
                _trunk_mlp, "up_proj.weight_scale", "gate_proj.weight_scale", window=_trunk_window
            ),
            cut=_fc1_cut,
            transform=_zero_padded(_expert_fc1_s_cd, 0, 1),
            layers=_cute_dsl_copy_layers,
        ),
        W(
            "fc2_w_cd",
            shape=lambda d: (d.local_experts, d.hidden, d.expert_inter // 2),
            dtype=u8,
            src=_experts(_trunk_mlp, "down_proj.weight", window=_trunk_window),
            cut=_fc2_cut(2),
            transform=_zero_padded(None, 1, 2),
            layers=_cute_dsl_copy_layers,
        ),
        W(
            "fc2_s_cd",
            shape=lambda d: (d.local_experts, d.hidden, d.expert_inter // SF_VEC),
            dtype=u8,
            src=_experts(_trunk_mlp, "down_proj.weight_scale", window=_trunk_window),
            cut=_fc2_cut(SF_VEC),
            transform=_zero_padded(_expert_fc2_s_cd, 1, SF_VEC),
            layers=_cute_dsl_copy_layers,
        ),
        # Replicated over the whole routing space, at every split.
        *(
            W(
                name,
                shape=lambda d: (d.num_experts,),
                dtype=f32,
                src=_experts(_trunk_mlp, suffix, window=_routing_space),
            )
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
        # --- the vision tower: all bf16, replicated, one-to-one, no transform ---
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
        # front end; every shape equals a trunk full-attention layer's (cut the same way)
        # except the front end and the expert quantization (FP8 128x128 block scales, not
        # NVFP4). Its keys carry no `{p}`: the module's prefix is not the trunk's.
        W(
            "mtp_pre_e_norm",
            shape=lambda d: (d.hidden,),
            src=f"{_MTP}.pre_fc_norm_embedding.weight",
            layers=_mtp_only,
        ),
        # `[hc_count * hidden]`: the module's input is the trunk's four-stream residual.
        W(
            "mtp_pre_h_norm",
            shape=lambda d: (d.stream,),
            src=f"{_MTP}.pre_fc_norm_hidden.weight",
            layers=_mtp_only,
        ),
        W(
            "mtp_fc_e",
            shape=lambda d: (d.hidden, d.hidden),
            src=f"{_MTP}.fc_embedding.weight",
            layers=_mtp_only,
        ),
        W(
            "mtp_fc_h",
            shape=lambda d: (d.hidden, d.hidden),
            src=f"{_MTP}.fc_hidden.weight",
            layers=_mtp_only,
        ),
        *(
            w
            for tag, module in _HC_MODULES
            for w in _hyper_connection(
                f"mtp_{tag}",
                lambda name, m=module: f"{_MTP_LAYER}.{m}.{name}.weight",
                _mtp_only,
            )
        ),
        *_attn_in_separate("mtp_", f"{_MTP_LAYER}.self_attn", _mtp_only),
        *_attn_out_and_norms("mtp_", f"{_MTP_LAYER}.self_attn", _mtp_only),
        *_index_branch("mtp_", f"{_MTP_LAYER}.self_attn.indexer", _mtp_only),
        W(
            "mtp_router",
            shape=lambda d: (d.num_experts, d.hidden),
            src=f"{_MTP_LAYER}.mlp.gate.weight",
            layers=_mtp_only,
        ),
        # e4m3 codes stored row-major with one fp32 scale per 128x128 weight block, the
        # FC1 halves concatenated `[up ; gate]` on the row axis; this rank's window.
        W(
            "mtp_fc1_w",
            shape=lambda d: (d.mtp_local_experts, 2 * d.moe_inter, d.hidden),
            dtype=f8,
            src=_experts(_mtp_mlp, "up_proj.weight", "gate_proj.weight", window=_mtp_window),
            transform=_mtp_fc1_w,
            layers=_mtp_only,
        ),
        W(
            "mtp_fc1_s",
            shape=lambda d: (
                d.mtp_local_experts,
                (2 * d.moe_inter) // FP8_BLOCK,
                d.hidden // FP8_BLOCK,
            ),
            dtype=f32,
            src=_experts(
                _mtp_mlp,
                "up_proj.weight_scale_inv",
                "gate_proj.weight_scale_inv",
                window=_mtp_window,
            ),
            transform=_mtp_fc1_s,
            layers=_mtp_only,
        ),
        # FC2 needs no relayout at all: the checkpoint's `[hidden, I]` e4m3 block is
        # exactly the operand.
        W(
            "mtp_fc2_w",
            shape=lambda d: (d.mtp_local_experts, d.hidden, d.moe_inter),
            dtype=f8,
            src=_experts(_mtp_mlp, "down_proj.weight", window=_mtp_window),
            layers=_mtp_only,
        ),
        W(
            "mtp_fc2_s",
            shape=lambda d: (
                d.mtp_local_experts,
                d.hidden // FP8_BLOCK,
                d.moe_inter // FP8_BLOCK,
            ),
            dtype=f32,
            src=_experts(_mtp_mlp, "down_proj.weight_scale_inv", window=_mtp_window),
            transform=_mtp_fc2_s,
            layers=_mtp_only,
        ),
        *_shared_expert("mtp_", _mtp_mlp, _mtp_only),
        # The module's own stack-final mixer, which is its `shared_head`.
        *_mixer("mtp_fhc", f"{_MTP}.hyper_connection_mixer", _mtp_only),
    )

    def dims(self, core) -> SimpleNamespace:
        """This rank's dimension bundle, from the core's normalized text config, the
        composite's vision config (the core holds it as `vision_config` before it
        declares anything), the engine's `Mapping` and the core's engine choices."""
        return rank_dims(
            core.model_config.pretrained_config,
            core.vision_config,
            core.model_config.mapping,
            mtp_enabled=getattr(core.model_config, "spec_config", None) is not None,
            trunk_moe_backend=core.trunk_moe_backend,
            moe_prefill_copy=core.moe_prefill_copy,
            dtype=core.model_config.torch_dtype,
        )

    def expected_unconsumed(self, d: SimpleNamespace, keys) -> set:
        """The checkpoint keys this rank predicts it leaves unread: the tensors of every
        routed expert outside its window (its per-tensor scalars are read on every rank;
        under the intermediate split the window is every expert, each read at this rank's
        slice), and, with MTP live, the draft experts outside its draft window; with MTP off, the
        whole `mtp.` subtree. The windows are enumerated from `d`, not read off the
        checkpoint, so a key that belongs to a real weight can never land here."""
        out = set()
        if d.expert_split in LOADED_EXPERT_SPLITS:
            start, count = _trunk_window(d)
            for i in range(d.num_layers):
                for e in range(d.num_experts):
                    if not start <= e < start + count:
                        out.update(
                            f"{_trunk_mlp(i)}.experts.{e}.{t}" for t in _TRUNK_EXPERT_TENSORS
                        )
        if not d.mtp_enabled:
            return out | {k for k in keys if k.startswith(f"{_MTP}.")}
        start, count = _mtp_window(d)
        for e in range(d.num_experts):
            if not start <= e < start + count:
                out.update(f"{_mtp_mlp(None)}.experts.{e}.{t}" for t in _MTP_EXPERT_TENSORS)
        return out

    def declare(self, d: SimpleNamespace) -> nn.ParameterDict:
        """Allocate every weight the table declares, at this rank's share.

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

    def manifest(self, d: SimpleNamespace) -> dict[str, list[tuple]]:
        """param key -> [(checkpoint key, index into the param | None, cut | None,
        transform | None)]."""
        rows: dict[str, list[tuple]] = {}
        for entry in self.WEIGHTS:
            for key, i, prefix in self._targets(entry, d):
                rows[key] = [
                    (ckpt, index, cut, entry.transform)
                    for ckpt, index, cut in entry.sources(d, i, prefix)
                ]
        return rows

    def read(self, weights, ckpt_key, cut, transform, d, device=None) -> torch.Tensor:
        """One manifest source -> the tensor its destination receives.

        Each key of the source is read through its cut -- only this rank's rows or
        columns leave the checkpoint -- moved to `device`, and handed to `transform`.
        The one path `load` takes, so the shard test drives it directly.
        """
        multi = isinstance(ckpt_key, tuple)
        keys = ckpt_key if multi else (ckpt_key,)
        cuts = (cut if cut is not None else (None,) * len(keys)) if multi else (cut,)
        assert len(cuts) == len(keys), (ckpt_key, cut)
        for key in keys:
            assert key in weights, f"checkpoint key missing: {key}"
        src_all = tuple(_read(weights[key], c) for key, c in zip(keys, cuts))
        if device is not None:
            src_all = tuple(t.to(device, non_blocking=True) for t in src_all)
        if transform is not None:
            return transform(src_all if multi else src_all[0], d)
        assert len(src_all) == 1, "a multi-key row needs a transform"
        return src_all[0]

    def load(self, model, weights) -> None:
        """Fill every declared parameter. The n-gram table is declared here first, each
        source is read through its cut, multi-key rows feed their transform several
        tensors at once, and coverage is asserted in both directions."""
        core = model.model
        d = self.dims(core)
        assert d.expert_split in LOADED_EXPERT_SPLITS, (
            f"routed experts split by {d.expert_split!r}: this table declares the splits "
            f"{LOADED_EXPERT_SPLITS}"
        )
        # The one load-time declaration, before the manifest is built: the table's
        # height comes from the shards, and the manifest addresses it by row range.
        rows_per_shard = _declare_ngram_table(core, d, weights)
        manifest = self.manifest(d)
        manifest["ple_table"] = ngram_table_sources(d, rows_per_shard)
        consumed: set = set()

        def fill(param: nn.Parameter, ckpt_key, index, cut, transform) -> None:
            dst = param.data if index is None else param.data[index]
            # Materialize where the destination lives: the expert relayouts are row
            # gathers and the scale swizzle is a kernel, both on the device.
            src = self.read(weights, ckpt_key, cut, transform, d, dst.device)
            assert dst.shape == src.shape, (ckpt_key, tuple(dst.shape), tuple(src.shape))
            assert src.dtype == dst.dtype, (ckpt_key, src.dtype, dst.dtype)
            dst.copy_(src, non_blocking=True)
            consumed.update(ckpt_key if isinstance(ckpt_key, tuple) else (ckpt_key,))

        assert set(manifest.keys()) == set(core.w.keys()), (
            "manifest/parameter drift",
            set(manifest.keys()) ^ set(core.w.keys()),
        )
        for param_key, sources in manifest.items():
            for ckpt_key, index, cut, transform in sources:
                fill(core.w[param_key], ckpt_key, index, cut, transform)
            if param_key.endswith(_RELEASE_AFTER):
                torch.cuda.empty_cache()

        # The shell owns lm_head (untied, bf16 on this checkpoint) and builds it
        # vocab-parallel under attention TP and whole under attention DP; read the rows
        # it built and nothing else.
        cut, pad = lm_head_share(model.lm_head.weight.shape[0], d.vocab, d.tp_rank)
        assert tuple(model.lm_head.weight.shape[1:]) == (d.hidden,), (
            f"the shell built lm_head {tuple(model.lm_head.weight.shape)}; this checkpoint's "
            f"is a dense [{d.vocab}, {d.hidden}] weight"
        )
        fill(model.lm_head.weight, "lm_head.weight", None, cut, pad)
        # Parameter-side coverage, the other direction: name every registered parameter
        # outside the target's own ParameterDict instead of assuming lm_head is the only
        # one.
        shell = {n for n, _ in model.named_parameters() if not n.startswith("model.w.")}
        assert shell == {"lm_head.weight"}, f"unfed shell parameters: {sorted(shell)}"

        torch.cuda.synchronize()
        leftover = set(weights.keys()) - consumed
        expected = self.expected_unconsumed(d, weights.keys())
        assert leftover == expected, (
            "checkpoint coverage: leftover keys are not exactly this rank's off-window "
            f"experts{'' if d.mtp_enabled else ' and the MTP module'}; unexpected "
            f"{sorted(leftover - expected)[:8]}, missed {sorted(expected - leftover)[:8]}"
        )


MODEL_WEIGHTS = Qwen4ExpUnifiedWeights()


def _heads(total: int, ranks: int, rank: int) -> tuple[int, int]:
    """(heads on `rank`, its first global head) when `total` heads are split over `ranks`:
    an even split, or each head repeated on `ranks // total` neighbouring ranks when there
    are more ranks than heads -- the rule the K/V and recurrent pools follow
    (`pool.pool_geometry`)."""
    assert total % ranks == 0 or ranks % total == 0, (total, ranks)
    return -(-total // ranks), rank * total // ranks


def rank_dims(
    cfg,
    vcfg,
    mapping,
    *,
    mtp_enabled: bool,
    trunk_moe_backend: str,
    moe_prefill_copy: bool,
    dtype: torch.dtype,
) -> SimpleNamespace:
    """Every width the table declares at rank `mapping.rank`, and the shares behind them.

    Args:
        cfg: the engine-normalized text config (`Qwen4ExpTextConfig`).
        vcfg: the composite's vision config.
        mapping: this rank's `Mapping`.
        mtp_enabled: whether the engine has a `speculative_config` (the MTP module is live).
        trunk_moe_backend: the routed backend the trunk runs, which picks the expert layout.
        moe_prefill_copy: whether the core keeps the CuteDSL-layout copy of its window.
        dtype: the activation dtype, which every unquantized weight is declared in.
    """
    hidden = cfg.hidden_size
    head_dim = cfg.head_dim
    index_heads, index_kv_heads = cfg.indexer_n_heads, cfg.indexer_kv_heads

    attn_tp = attention_tp_size(mapping)
    attn_rank = mapping.tp_rank if attn_tp > 1 else 0
    heads_q, q_head_start = _heads(cfg.num_attention_heads, attn_tp, attn_rank)
    heads_kv, kv_head_start = _heads(cfg.num_key_value_heads, attn_tp, attn_rank)

    state_tp = mamba_effective_tp_size(mapping)
    state_rank = mapping.tp_rank if state_tp > 1 else 0
    num_k_heads, k_head_start = _heads(cfg.linear_num_key_heads, state_tp, state_rank)
    num_v_heads, v_head_start = _heads(cfg.linear_num_value_heads, state_tp, state_rank)
    key_dim = num_k_heads * cfg.linear_key_head_dim
    value_dim = num_v_heads * cfg.linear_value_head_dim

    num_experts = cfg.num_experts
    # The routed experts' MoE group is the TP group: tp4 splits it by intermediate,
    # tep4 and dep4 by expert.
    assert mapping.moe_tp_size * mapping.moe_ep_size == mapping.tp_size, (
        mapping.moe_tp_size,
        mapping.moe_ep_size,
        mapping.tp_size,
    )
    assert num_experts % mapping.moe_ep_size == 0 and num_experts % mapping.tp_size == 0, (
        num_experts,
        mapping.moe_ep_size,
        mapping.tp_size,
    )
    local_experts = num_experts // mapping.moe_ep_size
    mtp_local_experts = num_experts // mapping.tp_size
    # Each expert's intermediate over the MoE TP ranks (tp4; one rank, the whole of it,
    # otherwise). A share holds whole group-16 scale vectors, so `down_proj`'s scale
    # columns cut on a vector boundary; it is stored zero-padded to EXPERT_INTER_ALIGN.
    moe_inter = cfg.moe_intermediate_size
    assert moe_inter % mapping.moe_tp_size == 0, (moe_inter, mapping.moe_tp_size)
    expert_inter_share = moe_inter // mapping.moe_tp_size
    assert expert_inter_share % SF_VEC == 0, (expert_inter_share, SF_VEC)
    expert_inter = -(-expert_inter_share // EXPERT_INTER_ALIGN) * EXPERT_INTER_ALIGN

    ngram_heads = cfg.heads_per_ngram * (cfg.ngram_size - 1)
    v_hidden = vcfg.hidden_size
    return SimpleNamespace(
        num_layers=cfg.num_hidden_layers,
        is_attn=[t == "full_attention" for t in cfg.layer_types],
        hidden=hidden,
        hc_count=cfg.hc_count,
        hc_lowrank=cfg.hc_lowrank,
        stream=cfg.hc_count * hidden,
        tp_rank=mapping.tp_rank,
        # --- attention, at this rank's heads ---
        head_dim=head_dim,
        heads_q=heads_q,
        q_head_start=q_head_start,
        heads_kv=heads_kv,
        kv_head_start=kv_head_start,
        q_width=heads_q * head_dim,
        kv_width=heads_kv * head_dim,
        index_head_dim=cfg.indexer_head_dim,
        index_width=(index_heads + index_kv_heads) * cfg.indexer_head_dim,
        # [q | k | v | gate | index q, k | pad of one index head]
        attn_in_rows=2 * heads_q * head_dim
        + 2 * heads_kv * head_dim
        + (index_heads + index_kv_heads + 1) * cfg.indexer_head_dim,
        # --- Gated-DeltaNet, at this rank's key and value heads ---
        num_k_heads=num_k_heads,
        k_head_start=k_head_start,
        num_v_heads=num_v_heads,
        v_head_start=v_head_start,
        head_k_dim=cfg.linear_key_head_dim,
        head_v_dim=cfg.linear_value_head_dim,
        key_dim=key_dim,
        key_dim_total=cfg.linear_num_key_heads * cfg.linear_key_head_dim,
        value_dim=value_dim,
        conv_dim=2 * key_dim + value_dim,
        in_proj_split=(2 * key_dim + value_dim, value_dim, num_v_heads, num_v_heads),
        conv_kernel=cfg.linear_conv_kernel_dim,
        # --- MoE ---
        num_experts=num_experts,
        expert_split="ep" if mapping.moe_tp_size == 1 else "tp",
        local_experts=local_experts,
        expert_offset=local_experts * mapping.moe_ep_rank,
        # This rank's intermediate rows `[expert_inter_start, + expert_inter_share)` of
        # every expert it holds, stored `expert_inter` wide (the runner's
        # `intermediate_size`).
        expert_inter_share=expert_inter_share,
        expert_inter_start=expert_inter_share * mapping.moe_tp_rank,
        expert_inter=expert_inter,
        mtp_local_experts=mtp_local_experts,
        mtp_expert_offset=mtp_local_experts * mapping.tp_rank,
        moe_inter=moe_inter,
        shared_inter=cfg.shared_expert_intermediate_size,
        vocab=cfg.vocab_size,
        # --- PLE ---
        ple_layer=list(cfg.ple_layer_ids)[0] - 1,  # one-based in the config
        ple_embed_dim=cfg.ple_embed_dim,
        ple_conv_kernel=cfg.ple_conv_kernel_size,
        ngram_size=cfg.ngram_size,
        ngram_heads=ngram_heads,
        ple_head_dim=cfg.ple_embed_dim // ngram_heads,
        ple_shards=cfg.split_ngram_parts,
        ple_table_dtype=PLE_TABLE_DTYPE,
        # --- the vision tower ---
        v_depth=vcfg.depth,
        v_hidden=v_hidden,
        v_inter=vcfg.intermediate_size,
        v_channels=vcfg.in_channels,
        v_tpatch=vcfg.temporal_patch_size,
        v_patch=vcfg.patch_size,
        v_pos_rows=vcfg.num_position_embeddings,
        v_merge_dim=v_hidden * vcfg.spatial_merge_size * vcfg.spatial_merge_size,
        # --- engine choices ---
        mtp_enabled=mtp_enabled,
        trunk_moe_backend=trunk_moe_backend,
        moe_prefill_copy=moe_prefill_copy,
        dtype=dtype,
    )


def lm_head_share(rows: int, vocab: int, tp_rank: int) -> tuple[Cut, Callable]:
    """The cut and the pad that fill an lm_head the shell built with `rows` rows.

    Whole under attention DP (`rows == vocab`); a vocab-parallel shard under attention TP.
    A vocab that does not split evenly leaves the last shard short, and its missing rows
    are zero, as the built-in LMHead pads them -- so the cut stops at `vocab` and the pad
    fills the rest.
    """
    start = 0 if rows == vocab else tp_rank * rows
    stop = min(start + rows, vocab)
    assert start < stop, (rows, vocab, tp_rank)

    def pad(t: torch.Tensor, d) -> torch.Tensor:
        if t.shape[0] == rows:
            return t.contiguous()
        out = torch.zeros(rows, t.shape[1], dtype=t.dtype, device=t.device)
        out[: t.shape[0]] = t
        return out

    return (slice(start, stop),), pad


def ngram_shard_keys(d: SimpleNamespace) -> list[str]:
    """The n-gram table's shard keys, in concatenation order."""
    prefix = _ngram_prefix(d.ple_layer)
    return [f"{prefix}.shard_{s}.weight" for s in range(d.ple_shards)]


def ngram_table_sources(d: SimpleNamespace, rows_per_shard: int) -> list[tuple]:
    """The `ple_table` manifest rows: shard `s` fills rows `[s * rows_per_shard, ...)`,
    read whole on every rank (the table is replicated)."""
    return [
        (key, (_span(s * rows_per_shard, rows_per_shard),), None, None)
        for s, key in enumerate(ngram_shard_keys(d))
    ]


def _declare_ngram_table(core, d: SimpleNamespace, weights) -> int:
    """Register the n-gram table with the height its shards concatenate to, and return
    the rows each shard holds.

    The shards are **contiguous row chunks**, not an interleaving: global row `r` lives in
    shard `r // rows_per_shard` at local row `r % rows_per_shard`. That is what makes the
    flat buffer indexable by the hash's global rows.

    **This is the whole of PLE host offload.** `core.ple_host_offload` picks between an
    ordinary device allocation and host memory the device addresses. Nothing else
    changes: same shape, same dtype, same `ParameterDict` key, same rows filling it shard
    by shard, same gather in the forward. An offloaded tensor is **loaded, not skipped**
    -- it stays fully covered in both directions. What the manifest cannot record is
    *where* it landed, so the core asserts that separately against the driver.
    """
    keys = ngram_shard_keys(d)
    for key in keys:
        assert key in weights, f"checkpoint key missing: {key}"
    shapes = {tuple(weights[key].shape) for key in keys}
    assert len(shapes) == 1, (
        f"the n-gram table's {len(keys)} shards do not share one shape: "
        f"{sorted(shapes)}; they are concatenated as equal contiguous row "
        f"chunks and an uneven one would shift every row above it"
    )
    rows_per_shard, head_dim = shapes.pop()
    assert head_dim == d.ple_head_dim, (
        f"the n-gram table is {head_dim} wide but {d.ngram_heads} heads must "
        f"tile ple_embed_dim {d.ple_embed_dim}, i.e. {d.ple_head_dim}"
    )
    dtype = weights[keys[0]].dtype
    assert dtype == d.ple_table_dtype, (
        f"the n-gram table is stored {dtype}; this target declares "
        f"{d.ple_table_dtype} and does not convert it"
    )
    device = core.w["embed"].device
    rows = d.ple_shards * rows_per_shard
    if core.ple_host_offload:
        table, core._ple_table_host = _host_resident_empty(rows, head_dim, dtype, device)
    else:
        table = torch.empty(rows, head_dim, dtype=dtype, device=device)
        core._ple_table_host = None
    core.w["ple_table"] = nn.Parameter(table, requires_grad=False)
    # `nn.Parameter` is `_make_subclass`, which shares storage rather than copying -- but
    # "rather than copying" is the entire feature here, so it is asserted instead of
    # assumed.
    assert core.w["ple_table"].data_ptr() == table.data_ptr(), (
        "wrapping the n-gram table in nn.Parameter moved its storage"
    )
    core.ple_rows_per_shard = rows_per_shard
    return rows_per_shard
