# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Weight table and loader: Qwen3.8-Flash-Next-FP8 (hf-236dfdf_orig) / sm_103 / unified.

One `W` per weight role (shape, checkpoint source, the rank's cut, dtype, load-time
transform); `Qwen4ExpFp8Weights` unrolls the table into declaration, manifest and load.
Storage is HF `[out, in]`; GEMM views, grouped-norm slices and fp32 gating params are
derived in `modeling.py` after load.

**The checkpoint.** Block FP8 (`quantization_config`: `quant_method: fp8`,
`weight_block_size: [128, 128]`, dynamic activations), which the engine resolves as
`FP8_BLOCK_SCALES`. Only the routed experts are quantized: every `gate_proj`, `up_proj`
and `down_proj` of the trunk's 48 x 512 experts and of the MTP layer's 512 is e4m3 codes
with one bf16 `weight_scale_inv` per 128x128 weight block, per expert and unfused, and
no per-tensor scalars. Everything else is bf16 (attention, the QSA indexer, Gated-DeltaNet,
the router, the shared expert, hyper-connections, PLE projections, the vision tower,
`embed_tokens`, `lm_head`) except the PLE n-gram table, which is e4m3 with one bf16
per-tensor `weight_scale`, and its three int64 id-space tensors. There is **no**
`model.language_model.norm.weight`: the stack-final hyper-connection mixer is the final
norm.

**Every weight is declared at this rank's share.** The shares are computed once, in
`rank_dims`, from the engine's `Mapping`, at each member: tp1, tp2, tep2, dep2, tp4, tep4
and dep4.

* **Attention heads** are split over the attention TP group: the TP group, or one rank
  under attention DP (and at tp1). Rank `r` of `t` holds q heads `[24r / t, 24(r + 1) / t)`:
  their rows of `q_proj` (q and the output gate interleave per head, so a head's
  `2 * head_dim` rows are contiguous) and their input columns of `o_proj`. The two KV heads
  are split at two ranks and repeated on neighbouring ranks at four: rank `r` holds head
  `r * 2 // t`, the head `duplicate_kv_weight` gives it and the K/V pool it binds holds.
  The QSA indexer and the q/k norms are replicated. The MTP layer's attention is cut the
  same way.
* **Gated-DeltaNet heads** are split over `mamba_effective_tp_size` (the TP group, or
  one rank under attention DP), as the recurrent pools are: rank `r` of `t` holds key heads
  `[16r / t, 16(r + 1) / t)` and value heads `[48r / t, 48(r + 1) / t)`. Of `in_proj_qkv`
  and the depthwise `conv1d` it reads three row blocks, the q and k rows of its key heads
  and the v rows of its value heads, into one `[q | k | v]` operand: the channel order of
  the conv pool. `in_proj_z`, `in_proj_a`, `in_proj_b`, `A_log` and `dt_bias` are cut by
  value head, `out_proj` by its value-head input columns.
* **Routed experts**, the trunk's and the draft layer's, are whole-expert windows over the
  whole TP group at every member: rank `r` of `t` holds experts `[512r / t, 512(r + 1) / t)`
  of every MoE layer -- all 512 at tp1, 256 at tp2, tep2 and dep2, 128 at tp4, tep4 and
  dep4 -- each at its whole 640-row intermediate. The engine's `Mapping` puts tp2 and tp4's
  MoE in tensor parallelism, but 640 / 2 and 640 / 4 rows do not take the 128-row FP8 weight
  block, so the table does not read `moe_tp_size` / `moe_ep_size`: it cuts by expert, never
  through the intermediate, and requantizes nothing. A token's slot outside a window adds
  nothing to that rank's output, so the windows' partial outputs sum to the layer's.

Everything else is replicated: hyper-connections, the PLE block and its n-gram table,
the router, the shared expert, the embedding, the vision tower, the MTP layer's front
end, router, shared expert and mixer. `lm_head` is the shell's: `load` reads the rows the
shell built, vocab-parallel under attention TP, whole under attention DP and at tp1.

**A rank reads only its share.** Each split weight carries a cut, the rows or columns of
the checkpoint tensor this rank holds, applied to the checkpoint entry before it is
materialized: on a lazy safetensors slice that reads those bytes and nothing else, on an
mmapped tensor it is a view, and either way only the share is copied to the device. An
off-window expert is never read.

Transforms:

* trunk attention input projection: `q_proj`, `k_proj`, `v_proj` and the indexer's
  `index_qk_proj` read the same activation, so they are one
  `[q | k | v | gate | index | pad, hidden]` operand and one GEMM (`_attn_in_w`), at this
  rank's heads: 7424 rows at attention TP2, 4352 at attention TP4, 14080 under attention DP
  and at tp1. The q halves go first so the GEMM output's leading columns are the packed
  `[q | k | v]` the fused norm+rope rewrites in place, then the gate halves, the replicated
  index projection and `index_head_dim` zero rows (a row is a whole number of `head_dim`
  heads). The draft (MTP) layer keeps its separate projections;
* fused shared-expert gate|up: `[gate ; up ; gate logit ; pad]` (`_cat_rows`);
* routed experts, the operands of the family's `fp8_block_scale_moe_runner` (`ops/moe/`):
  per expert, FC1 is the `[up ; gate]` row concatenation of the e4m3 codes
  (`_expert_fc1_w`) and of their block scales cast bf16 -> fp32 (`_expert_fc1_s`), and FC2
  is `down_proj`'s codes as stored and its scales cast to fp32 (`_expert_fc2_s`). The
  runner reads this plain row-major layout: no shuffle, no swizzle, no padding. The casts
  are exact;
* n-gram table: 128 row chunks concatenated into one `[rows, 160]` buffer (the hash
  emits global row indices). It is replicated, and host-resident by default
  (`_host_resident_empty`, `PLE_HOST_OFFLOAD_ENV`): one copy per rank.

The n-gram table is not in the table: its height is a checkpoint fact, so `load`
registers it with `torch.empty` (`_declare_ngram_table`) and adds its rows to the
manifest (`ngram_table_sources`).

Not loaded: `mtp.*` when there is no `speculative_config`. With one, the MTP module is
loaded at this rank's share, including the draft layer's own QSA indexer.

**The parameters `modeling.py` binds** (`core.w`, an `nn.ParameterDict`), keyed
`l{i}_<name>` for decoder layer `i`, `l{num_hidden_layers}_mtp_<name>` for the MTP module,
`l{b}_vis_<name>` for vision block `b`, and bare names otherwise:

* every layer: `ahc_` / `mhc_` + `norm`, `down`, `up`, `inject` (the two hyper-connection
  gates); `router`; `fc1_w`, `fc1_s`, `fc2_w`, `fc2_s` (the routed expert window,
  `expert_window`); `sh_gu`, `sh_dn` (the shared expert);
* full-attention layers: `attn_in_w`, `o_w`, `q_norm`, `k_norm`, `idx_qn`, `idx_kn`;
* Gated-DeltaNet layers: `in_proj_w` (`d.in_proj_split`), `A_log`, `dt_bias`, `conv_w`,
  `gnorm`, `out_w`;
* not per layer: `fhc_norm`, `fhc_down`, `fhc_up` (the stack-final mixer), `embed`, the
  PLE block (`ple_key_w`, `ple_val_w`, `ple_nk`, `ple_nq`, `ple_nc`, `ple_conv_w`,
  `ple_mult`, `ple_vocab`, `ple_offs`, `ple_tscale`, and `ple_table`, declared at load),
  the vision tower (`vis_*`);
* the MTP module: `pre_e_norm`, `pre_h_norm`, `fc_e`, `fc_h`, the two gates and the final
  mixer (`ahc_*`, `mhc_*`, `fhc_*`), `q_w`, `k_w`, `v_w`, `o_w`, `q_norm`, `k_norm`,
  `idx_w`, `idx_qn`, `idx_kn`, `router`, `fc1_w`, `fc1_s`, `fc2_w`, `fc2_s`, `sh_gu`,
  `sh_dn`.

`load(model, weights)` fills every declared parameter once and asserts coverage both
ways: every parameter written, and every checkpoint key consumed (a key read through a
cut is consumed) or predicted by `expected_unconsumed` -- the experts outside this
rank's window, and the MTP module when there is no speculation.
"""

import os
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Callable, NamedTuple

import torch
from cuda.bindings import runtime as cudart
from torch import nn

from tensorrt_llm._torch.pyexecutor.config_utils import mamba_effective_tp_size
from tensorrt_llm._utils import CUASSERT

#: `layers=ALL_LAYERS`, the default: the weight is on every layer.
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
    """The driver's own answer to "what kind of memory is this pointer": a `cudaMemoryType`."""
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

    Not torch's pinned allocator: it rounds the 47.7 GiB table up to 64 GiB, and the
    device's random row gathers from its pages are slower than from registered pageable
    memory."""

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


#: The switch between the n-gram table in host memory the device addresses (the default) and
#: an ordinary device allocation (`0`, `false`, `no` or `off`). Output is identical.
PLE_HOST_OFFLOAD_ENV = "TRTLLM_MODELING_V2_PLE_HOST_OFFLOAD"
_PLE_HOST_OFFLOAD_OFF = frozenset({"0", "false", "no", "off"})


def ple_host_offload_enabled() -> bool:
    """Whether the n-gram table is declared in host memory (`PLE_HOST_OFFLOAD_ENV`, on unless
    turned off). The core reads it once, at construction, into `core.ple_host_offload`."""
    value = os.environ.get(PLE_HOST_OFFLOAD_ENV, "1").strip().lower()
    return value not in _PLE_HOST_OFFLOAD_OFF


# ------------------------------------------------------------------ the table --

_LM = "model.language_model"
_VISUAL = "model.visual"
_MTP = "mtp"
_MTP_LAYER = "mtp.layers.0"

#: The FP8 block edge, on both axes of a routed expert's weight scales and along the
#: activation row the runner quantizes (1x128).
FP8_BLOCK = 128

#: The PLE n-gram table's storage: the checkpoint's FP8 per-tensor quantization, kept as
#: stored (the gather dequantizes the rows it reads).
PLE_TABLE_DTYPE = torch.float8_e4m3fn

#: A routed expert's block scales as the runner takes them; the checkpoint stores bf16, which
#: fp32 holds exactly.
EXPERT_SCALE_DTYPE = torch.float32

# Parameter-key suffixes after which to release load scratch: each expert's sources are
# staged on the device one expert at a time and the table fill stages one shard at a time;
# releasing after each parameter keeps peak load memory at one parameter's worth.
_RELEASE_AFTER = (
    "_fc1_w",
    "_fc1_s",
    "_fc2_w",
    "_fc2_s",
    "ple_table",
)


def _read(entry, cut: Cut | None) -> torch.Tensor:
    """Realize one checkpoint entry, reading only `cut` of it.

    `entry[cut]` on a lazy safetensors slice reads those rows or columns and nothing
    else; on a plain tensor it is a view. Uncut, `[:]` reads it whole -- but a 0-dim
    tensor rejects that index (this checkpoint has none; the guard keeps the reader
    general), and no 0-dim entry has a cut."""
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
    """(up, gate) e4m3 codes -> one expert's FC1 operand, `[up ; gate]` on the row axis and
    nothing else: the runner reads the checkpoint's plain row-major layout, up first."""
    up, gate = t
    return torch.cat([up, gate], dim=0).contiguous()


def _expert_fc1_s(t: tuple[torch.Tensor, ...], d) -> torch.Tensor:
    """(up, gate) 128x128 block scales -> one expert's FC1 scale operand: the same `[up ;
    gate]` concatenation, cast bf16 -> fp32."""
    up, gate = t
    return torch.cat([up, gate], dim=0).to(EXPERT_SCALE_DTYPE).contiguous()


def _expert_fc2_s(t: torch.Tensor, d) -> torch.Tensor:
    """`down_proj.weight_scale_inv` -> the FC2 scale operand: a bf16 -> fp32 cast, no
    relayout. FC2's codes need no transform at all: the checkpoint's `[hidden, I]` e4m3
    block is exactly the operand."""
    return t.to(EXPERT_SCALE_DTYPE).contiguous()


# --- which layers carry a weight ---


def _attn_layers(d):
    """The full-attention layers (`i % 4 == 3` on this checkpoint)."""
    return [i for i, a in enumerate(d.is_attn) if a]


def _gdn_layers(d):
    """The Gated-DeltaNet (linear-attention) layers."""
    return [i for i, a in enumerate(d.is_attn) if not a]


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


def _window(d) -> tuple[int, int]:
    """(first expert, count) of this rank's routed-expert window, the trunk's and the draft
    layer's alike."""
    return d.expert_offset, d.local_experts


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


#: The checkpoint tensors of one routed expert, the trunk's or the draft layer's: what a rank
#: outside its window leaves unread. There are no per-tensor scalars.
_PROJECTIONS = ("gate_proj", "up_proj", "down_proj")
_EXPERT_TENSORS = tuple(f"{p}.{s}" for p in _PROJECTIONS for s in ("weight", "weight_scale_inv"))


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


def _routed_experts(tag, prefix, layers):
    """One MoE layer's FP8 128x128 routed experts at this rank's window (`_window`), as
    `fp8_block_scale_moe_runner` takes them: e4m3 codes stored row-major with one fp32 scale
    per 128x128 weight block, the FC1 halves concatenated `[up ; gate]` on the row axis
    (`_expert_fc1_w`, `_expert_fc1_s`), FC2 as stored (`_expert_fc2_s`). Whole experts
    only: no cut reaches the intermediate."""
    return (
        W(
            f"{tag}fc1_w",
            shape=lambda d: (d.local_experts, 2 * d.moe_inter, d.hidden),
            dtype=f8,
            src=_experts(prefix, "up_proj.weight", "gate_proj.weight", window=_window),
            transform=_expert_fc1_w,
            layers=layers,
        ),
        W(
            f"{tag}fc1_s",
            shape=lambda d: (
                d.local_experts,
                (2 * d.moe_inter) // FP8_BLOCK,
                d.hidden // FP8_BLOCK,
            ),
            dtype=EXPERT_SCALE_DTYPE,
            src=_experts(
                prefix, "up_proj.weight_scale_inv", "gate_proj.weight_scale_inv", window=_window
            ),
            transform=_expert_fc1_s,
            layers=layers,
        ),
        W(
            f"{tag}fc2_w",
            shape=lambda d: (d.local_experts, d.hidden, d.moe_inter),
            dtype=f8,
            src=_experts(prefix, "down_proj.weight", window=_window),
            layers=layers,
        ),
        W(
            f"{tag}fc2_s",
            shape=lambda d: (d.local_experts, d.hidden // FP8_BLOCK, d.moe_inter // FP8_BLOCK),
            dtype=EXPERT_SCALE_DTYPE,
            src=_experts(prefix, "down_proj.weight_scale_inv", window=_window),
            transform=_expert_fc2_s,
            layers=layers,
        ),
    )


f8, i64 = torch.float8_e4m3fn, torch.int64


class Qwen4ExpFp8Weights:
    """Qwen3.8-Flash-Next-FP8 / sm_103 / unified, at one rank's share.

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
        # --- MoE on every layer: this rank's window of FP8 routed experts, a replicated
        # bf16 router and shared expert ---
        W("router", shape=lambda d: (d.num_experts, d.hidden), src="{p}.mlp.gate.weight"),
        *_routed_experts("", _trunk_mlp, ALL_LAYERS),
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
        # describe the id space and stay on `ple_embedding` itself.
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
        # except the front end and the attention input projections, which stay separate.
        # Its keys carry no `{p}`: the module's prefix is not the trunk's.
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
        # The trunk's expert form and this rank's same window.
        *_routed_experts("mtp_", _mtp_mlp, _mtp_only),
        *_shared_expert("mtp_", _mtp_mlp, _mtp_only),
        # The module's own stack-final mixer, which is its `shared_head`.
        *_mixer("mtp_fhc", f"{_MTP}.hyper_connection_mixer", _mtp_only),
    )

    def dims(self, core) -> SimpleNamespace:
        """This rank's dimension bundle, from the core's normalized text config, the
        composite's vision config (the core holds it as `vision_config` before it
        declares anything) and the engine's `Mapping`."""
        return rank_dims(
            core.model_config.pretrained_config,
            core.vision_config,
            core.model_config.mapping,
            mtp_enabled=getattr(core.model_config, "spec_config", None) is not None,
            dtype=core.model_config.torch_dtype,
        )

    def expected_unconsumed(self, d: SimpleNamespace, keys) -> set:
        """The checkpoint keys this rank predicts it leaves unread: the tensors of every
        routed expert outside its window, the trunk's and, with MTP live, the draft layer's;
        with MTP off, the whole `mtp.` subtree. The window is enumerated from `d`, not read
        off the checkpoint, so a key that belongs to a real weight can never land here."""
        start, count = _window(d)
        off_window = [e for e in range(d.num_experts) if not start <= e < start + count]
        out = {
            f"{_trunk_mlp(i)}.experts.{e}.{t}"
            for i in range(d.num_layers)
            for e in off_window
            for t in _EXPERT_TENSORS
        }
        if not d.mtp_enabled:
            return out | {k for k in keys if k.startswith(f"{_MTP}.")}
        out.update(f"{_mtp_mlp(None)}.experts.{e}.{t}" for e in off_window for t in _EXPERT_TENSORS)
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
        The one path `fill` takes.
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

    def fill(self, param: nn.Parameter, sources, weights, d: SimpleNamespace) -> set:
        """Fill `param` from its manifest rows `sources`, each materialized where the
        parameter lives (`read`), and return the checkpoint keys they read. The one path
        `load` fills every parameter by, so a test that fills a few parameters drives it."""
        consumed = set()
        for ckpt_key, index, cut, transform in sources:
            dst = param.data if index is None else param.data[index]
            src = self.read(weights, ckpt_key, cut, transform, d, dst.device)
            assert dst.shape == src.shape, (ckpt_key, tuple(dst.shape), tuple(src.shape))
            assert src.dtype == dst.dtype, (ckpt_key, src.dtype, dst.dtype)
            dst.copy_(src, non_blocking=True)
            consumed.update(ckpt_key if isinstance(ckpt_key, tuple) else (ckpt_key,))
        return consumed

    def load(self, model, weights) -> None:
        """Fill every declared parameter. The n-gram table is declared here first, each
        source is read through its cut, multi-key rows feed their transform several
        tensors at once, and coverage is asserted in both directions."""
        core = model.model
        d = self.dims(core)
        # The one load-time declaration, before the manifest is built: the table's
        # height comes from the shards, and the manifest addresses it by row range.
        rows_per_shard = _declare_ngram_table(core, d, weights)
        manifest = self.manifest(d)
        manifest["ple_table"] = ngram_table_sources(d, rows_per_shard)
        consumed: set = set()

        assert set(manifest.keys()) == set(core.w.keys()), (
            "manifest/parameter drift",
            set(manifest.keys()) ^ set(core.w.keys()),
        )
        for param_key, sources in manifest.items():
            consumed |= self.fill(core.w[param_key], sources, weights, d)
            if param_key.endswith(_RELEASE_AFTER):
                torch.cuda.empty_cache()

        # The shell owns lm_head (untied, bf16 on this checkpoint); read the rows it built
        # (`lm_head_share`) and nothing else.
        cut, pad = lm_head_share(model.lm_head.weight.shape[0], d.vocab, d.tp_rank)
        assert tuple(model.lm_head.weight.shape[1:]) == (d.hidden,), (
            f"the shell built lm_head {tuple(model.lm_head.weight.shape)}; this checkpoint's "
            f"is a dense [{d.vocab}, {d.hidden}] weight"
        )
        consumed |= self.fill(
            model.lm_head.weight, [("lm_head.weight", None, cut, pad)], weights, d
        )
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


MODEL_WEIGHTS = Qwen4ExpFp8Weights()


class ExpertWindow(NamedTuple):
    """One MoE layer's routed experts at this rank's window, as the family's
    `fp8_block_scale_moe_runner` takes them: `gemm1_weights`, `gemm1_weights_scale`,
    `gemm2_weights`, `gemm2_weights_scale`, and the window's `local_expert_offset` and
    `local_num_experts` (its `num_experts` stays the routing space, `d.num_experts`)."""

    fc1_w: torch.Tensor  # [count, 2 * moe_inter, hidden] e4m3, rows `[up ; gate]`
    fc1_s: torch.Tensor  # [count, 2 * moe_inter / 128, hidden / 128] fp32
    fc2_w: torch.Tensor  # [count, hidden, moe_inter] e4m3
    fc2_s: torch.Tensor  # [count, hidden / 128, moe_inter / 128] fp32
    offset: int
    count: int


def expert_window(w, d: SimpleNamespace, layer: int) -> ExpertWindow:
    """Decoder layer `layer`'s routed-expert operands out of the declared parameters `w`
    (`declare`), or the MTP module's at `layer == d.num_layers`."""
    assert 0 <= layer < d.num_layers or (layer == d.num_layers and d.mtp_enabled), (
        f"layer {layer}: the trunk has {d.num_layers} MoE layers"
        f"{' and the MTP module is not loaded' if layer == d.num_layers else ''}"
    )
    p = f"l{layer}_mtp_" if layer == d.num_layers else f"l{layer}_"
    start, count = _window(d)
    return ExpertWindow(
        w[f"{p}fc1_w"], w[f"{p}fc1_s"], w[f"{p}fc2_w"], w[f"{p}fc2_s"], start, count
    )


def attention_tp_size(mapping) -> int:
    """The ranks attention heads and the K/V pool's heads are split over
    (`KVCacheManagerV2.__init__`): the TP group, or one rank under attention DP."""
    return 1 if mapping.enable_attention_dp else mapping.tp_size


def _heads(total: int, ranks: int, rank: int) -> tuple[int, int]:
    """(heads on `rank`, its first global head) when `total` heads are split over `ranks`:
    an even split, or each head repeated on `ranks // total` neighbouring ranks when there
    are more ranks than heads -- the rule the K/V and recurrent pools follow."""
    assert total % ranks == 0 or ranks % total == 0, (total, ranks)
    return -(-total // ranks), rank * total // ranks


def rank_dims(cfg, vcfg, mapping, *, mtp_enabled: bool, dtype: torch.dtype) -> SimpleNamespace:
    """Every width the table declares at rank `mapping.rank`, and the shares behind them.

    Args:
        cfg: the engine-normalized text config (`Qwen4ExpTextConfig`).
        vcfg: the composite's vision config.
        mapping: this rank's `Mapping`.
        mtp_enabled: whether the engine has a `speculative_config` (the MTP module is live).
        dtype: the activation dtype, which every unquantized weight is declared in.
    """
    hidden = cfg.hidden_size
    head_dim = cfg.head_dim
    index_heads, index_kv_heads = cfg.indexer_n_heads, cfg.indexer_kv_heads

    # Every member spans its whole world in one TP group (routing admits no other).
    assert mapping.tp_size == mapping.world_size, (mapping.tp_size, mapping.world_size)

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

    # The routed experts, trunk and draft: whole-expert windows over the TP group whatever
    # the Mapping's MoE TP / EP split says, each expert at its whole intermediate, which
    # the FP8 weight blocks tile.
    num_experts = cfg.num_experts
    moe_inter = cfg.moe_intermediate_size
    assert mapping.moe_tp_size * mapping.moe_ep_size == mapping.tp_size, (
        mapping.moe_tp_size,
        mapping.moe_ep_size,
        mapping.tp_size,
    )
    assert num_experts % mapping.tp_size == 0, (num_experts, mapping.tp_size)
    assert moe_inter % FP8_BLOCK == 0 and hidden % FP8_BLOCK == 0, (moe_inter, hidden)
    local_experts = num_experts // mapping.tp_size

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
        # --- MoE: the window `[expert_offset, expert_offset + local_experts)` of every MoE
        # layer, the draft layer's included, at the whole `moe_inter` ---
        num_experts=num_experts,
        local_experts=local_experts,
        expert_offset=local_experts * mapping.tp_rank,
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
    ordinary device allocation and host memory the device addresses; shape, dtype,
    `ParameterDict` key, the shard-by-shard fill and the forward's gather are the same either
    way, so an offloaded table is loaded and covered in both directions like any other. The
    core asserts where it landed against the driver.
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
    # `nn.Parameter` (`_make_subclass`) shares storage rather than copying; host offload
    # depends on that, so it is asserted.
    assert core.w["ple_table"].data_ptr() == table.data_ptr(), (
        "wrapping the n-gram table in nn.Parameter moved its storage"
    )
    core.ple_rows_per_shard = rows_per_shard
    return rows_per_shard
