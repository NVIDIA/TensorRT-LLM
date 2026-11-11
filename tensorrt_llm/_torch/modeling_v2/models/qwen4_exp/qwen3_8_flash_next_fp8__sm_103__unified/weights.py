# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Weight table and loader: Qwen3.8-Flash-Next-FP8 (hf-236dfdf_orig) / sm_103 / unified.

One `W` per weight role (shape, checkpoint source, the rank's cut, dtype, load-time
transform); `Qwen4ExpFp8Weights` unrolls the table into declaration, manifest and load.
Storage is HF `[out, in]`; GEMM views, grouped-norm slices and fp32 gating params are
derived in `modeling.py` after load.

**Checkpoint.** Block FP8 (`weight_block_size: [128, 128]`, dynamic activations; the engine's
`FP8_BLOCK_SCALES`). Only the routed experts (trunk 48 x 512, MTP 512) are quantized: per
expert and unfused, e4m3 `gate_proj` / `up_proj` / `down_proj` with one bf16
`weight_scale_inv` per 128x128 block and no per-tensor scalars. Everything else is bf16,
except the PLE n-gram table (e4m3, one bf16 per-tensor `weight_scale`) and its three int64
id-space tensors. There is no `model.language_model.norm.weight`: the stack-final
hyper-connection mixer is the final norm. `mtp.*` is loaded only with a `speculative_config`.

**Shares** (`rank_dims`, from the engine's `Mapping`: tp1, tp2, tep2, dep2, tp4, tep4, dep4).
Rank `r` of `t` holds:

* attention (the attention TP group; one rank under attention DP): q heads
  `[24r / t, 24(r + 1) / t)`, i.e. their `q_proj` rows (q and output gate interleave per
  head) and `o_proj` input columns, and KV head `r * 2 // t` (split at two ranks, repeated at
  four, as `duplicate_kv_weight` and the K/V pool do). The QSA indexer and q/k norms are
  replicated;
* Gated-DeltaNet (`mamba_effective_tp_size`, as the recurrent pools): key heads
  `[16r / t, 16(r + 1) / t)` and value heads `[48r / t, 48(r + 1) / t)`;
* routed experts, trunk and draft: experts `[512r / t, 512(r + 1) / t)` over the whole TP
  group, each at its whole 640-row intermediate. The cut ignores `moe_tp_size` / `moe_ep_size`:
  640 / 2 and 640 / 4 rows do not tile the 128-row FP8 block, so nothing is requantized. An
  off-window slot adds nothing, so the windows' partial outputs sum to the layer's.

The MTP layer is cut like a trunk attention layer. Everything else is replicated; `lm_head`
is the shell's (vocab-parallel under attention TP, whole under attention DP and at tp1).

A cut is applied to the checkpoint entry before it is materialized, so a rank reads only its
share (a lazy safetensors slice reads only those bytes) and never an off-window expert.

Parameters (`core.w`, an `nn.ParameterDict`) are keyed `l{i}_<name>` for decoder layer `i`,
`l{num_hidden_layers}_mtp_<name>` for the MTP module, `l{b}_vis_<name>` for vision block `b`,
and bare names otherwise.

`load` asserts coverage both ways: every parameter written once, and every checkpoint key
consumed or predicted by `expected_unconsumed`.
"""

import mmap
import os
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Callable, NamedTuple

import torch
from cuda.bindings import runtime as cudart
from torch import nn

from tensorrt_llm._torch.pyexecutor.config_utils import mamba_effective_tp_size
from tensorrt_llm._utils import CUASSERT
from tensorrt_llm.logger import logger

ALL_LAYERS = object()

Cut = tuple[slice, ...]

#: The spellings of a `TRTLLM_MODELING_V2_*` switch: these turn it on, these off.
SWITCH_ON = frozenset({"1", "true", "yes", "on"})
SWITCH_OFF = frozenset({"0", "false", "no", "off"})


def switch(env: str, on: frozenset = SWITCH_ON) -> bool:
    """A `TRTLLM_MODELING_V2_*` switch, default off: unset, empty or a `SWITCH_OFF` spelling is
    off, an `on` spelling on; any other value raises."""
    raw = os.environ.get(env, "")
    value = raw.strip().lower()
    if not value or value in SWITCH_OFF:
        return False
    if value in on:
        return True
    raise ValueError(
        f"{env}={raw!r}: expected {' / '.join(sorted(on))} (on) or "
        f"{' / '.join(sorted(SWITCH_OFF))} (off)"
    )


#: Opt-in (`rr`; default off) block-major copy of every MoE layer's routed experts for
#: `ops/moe/moe_tpb_decode` (attention TP4 only): expert e's 128-row intermediate block b on rank
#: (e + b) % 4, plus this rank's slice of the shared expert. The whole-expert window stays for
#: prefill and the runner, so the copy costs one more window per layer (~31 GB / rank at tep4).
MOE_TPB_ENV = "TRTLLM_MODELING_V2_DECODE_MOE_TPB"


def moe_tpb_requested() -> bool:
    """Whether `MOE_TPB_ENV` asks for the block-major copy (`rr`); another value raises."""
    return switch(MOE_TPB_ENV, on=frozenset({"rr"}))


def moe_tpb_enabled(mapping) -> bool:
    return moe_tpb_requested() and mapping.tp_size == 4 and not mapping.enable_attention_dp


@dataclass(frozen=True)
class W:
    """One weight role; `d` below is the `rank_dims` bundle.

    `shape(d)` is the declared shape at this rank's share. `src` is a key template (`{p}` is
    the layer prefix) or `(d, i) -> [(key, index[, cut]), ...]`: `index` addresses the
    parameter (None for all of it), a tuple `key` hands its tensors to `transform` together.
    `cut(d)` is this rank's rows or columns of the checkpoint tensor (one per key for a tuple
    key; None reads it whole). `layers` is `ALL_LAYERS`, None (not per-layer) or
    `d -> iterable[int]`. `transform(t, d)` relayouts the cut tensor(s) on the target device.
    """

    name: str
    shape: Callable[[Any], tuple[int, ...]]
    src: str | Callable
    dtype: torch.dtype | None = None
    cut: Callable[[Any], Any] | None = None
    transform: Callable | None = None
    layers: object | None = ALL_LAYERS

    def layer_indices(self, d: SimpleNamespace) -> list[int] | None:
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
# The PLE n-gram table is a `cuda` tensor over registered, device-mapped host pages (not a
# host-side gather): the gather's row indices are produced on device inside the decode graph.

#: `cudaMemoryType` of host memory; the core asserts on it after load.
CUDA_MEMORY_TYPE_HOST = 1


def memory_kind(ptr: int) -> int:
    (attributes,) = CUASSERT(cudart.cudaPointerGetAttributes(ptr))
    return int(attributes.type)


#: `cudaHostRegister` flags: mapped (kernels may dereference the pages), portable (all contexts).
_CUDA_HOST_REGISTER_PORTABLE = 0x01
_CUDA_HOST_REGISTER_MAPPED = 0x02


def _local_cpus(device) -> set | None:
    """CPUs local to `device`'s PCI function that this process may run on, or None."""
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
    """A pageable CPU mapping of its own, registered (pinned, mapped) for the device.

    Unregistered before its pages are unmapped: freeing registered pages would leave the
    driver mapping memory the OS may hand out again. The mapping is advised `MADV_HUGEPAGE`
    before the registration faults it in, so the table sits on transparent huge pages whatever
    torch's CPU allocator does (TensorRT LLM imports torch with mimalloc's THP advice off): on
    GB300 the device's random row gathers from 64 KiB pages run 2.5x (64 tokens) to 23x
    (1024 tokens) slower. Not torch's pinned allocator, which rounds the 47.7 GiB table up to
    64 GiB.
    """

    def __init__(self, nbytes: int, device=None) -> None:
        # Registration faults the pages in on this thread, so they land on its NUMA node:
        # run on `device`'s local CPUs meanwhile to put the table next to its GPU.
        local = _local_cpus(device) if device is not None else None
        previous = os.sched_getaffinity(0) if local else None
        if local:
            os.sched_setaffinity(0, local)
        try:
            self._mapping = mmap.mmap(-1, nbytes, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS)
            try:
                self._mapping.madvise(mmap.MADV_HUGEPAGE)
            except OSError as error:  # a kernel without transparent huge pages
                logger.warning(
                    f"madvise(MADV_HUGEPAGE) failed for the host-resident PLE table ({error}); "
                    "the device's row gathers from its base pages run several times slower"
                )
            # Shares the mapping's pages, and keeps the mapping alive while it lives.
            self._storage = torch.frombuffer(self._mapping, dtype=torch.uint8)
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
    """`(table, owner)`: `[rows, cols]` on `device` in host memory; keep `owner` alive."""
    itemsize = torch.empty(0, dtype=dtype).element_size()
    assert itemsize == 1, (
        f"the host-resident allocation reinterprets a `|u1` buffer, so its "
        f"element must be one byte; {dtype} is {itemsize}"
    )
    assert device.type == "cuda", f"host offload needs a cuda device to map into; got {device}"

    host = _RegisteredHostBuffer(rows * cols, device)
    return _device_view(host.pointer, rows, cols, dtype, device), host


def _device_view(host_pointer: int, rows: int, cols: int, dtype, device) -> torch.Tensor:
    """`[rows, cols]` one-byte `dtype` view of mapped host pages at `host_pointer`."""
    (device_pointer,) = CUASSERT(cudart.cudaHostGetDevicePointer(host_pointer, 0))

    # No public torch API wraps a pointer the caching allocator did not hand out. `|u1`
    # because `__cuda_array_interface__` has no float8 typestr; reinterpreted below.
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
    # A copy here (`as_tensor` may copy instead of view) would be a silent 48 GiB allocation.
    assert table.data_ptr() == int(device_pointer), (
        "torch.as_tensor copied the host-mapped buffer instead of viewing it, "
        "so the table would be device-resident after all"
    )
    assert table.shape == (rows, cols) and table.dtype == dtype, (table.shape, table.dtype)
    return table


# ------------------------------------------------------------------ the table --

_LM = "model.language_model"
_VISUAL = "model.visual"
_MTP = "mtp"
_MTP_LAYER = "mtp.layers.0"

#: FP8 block edge: 128x128 expert weight-scale blocks, 1x128 activation quantization.
FP8_BLOCK = 128

#: The n-gram table is kept as stored (per-tensor FP8); the gather dequantizes the rows it reads.
PLE_TABLE_DTYPE = torch.float8_e4m3fn

#: Expert block-scale dtype the runner takes; the checkpoint's bf16 converts exactly.
EXPERT_SCALE_DTYPE = torch.float32

# Parameter-key suffixes after which `load` empties the CUDA cache, so peak load memory
# stays near one parameter (sources are staged one expert or shard at a time).
_RELEASE_AFTER = (
    "_fc1_w",
    "_fc1_s",
    "_fc2_w",
    "_fc2_s",
    "ple_table",
)


def _read(entry, cut: Cut | None) -> torch.Tensor:
    """Materialize `entry`, reading only `cut` of it (a 0-dim entry rejects `[:]`)."""
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
    return (_span(d.kv_head_start * d.head_dim, d.kv_width),)


def _o_cols(d) -> Cut:
    return (slice(None), _span(d.q_head_start * d.head_dim, d.q_width))


def _key_rows(d, section: int) -> Cut:
    """Rows of this rank's key heads in section 0 (q) or 1 (k) of a `[q | k | v]` GDN tensor."""
    return (_span(section * d.key_dim_total + d.k_head_start * d.head_k_dim, d.key_dim),)


def _value_rows(d, base: int = 0) -> Cut:
    return (_span(base + d.v_head_start * d.head_v_dim, d.value_dim),)


def _value_head_rows(d) -> Cut:
    """One row per value head (`in_proj_a`, `in_proj_b`, `A_log`, `dt_bias`)."""
    return (_span(d.v_head_start, d.num_v_heads),)


def _value_cols(d) -> Cut:
    return (slice(None), _span(d.v_head_start * d.head_v_dim, d.value_dim))


def _qkv_blocks(d) -> tuple[tuple[slice, Cut], ...]:
    """(destination rows, checkpoint cut) of this rank's q, k and v blocks, in conv-pool order."""
    return (
        (slice(0, d.key_dim), _key_rows(d, 0)),
        (slice(d.key_dim, 2 * d.key_dim), _key_rows(d, 1)),
        (slice(2 * d.key_dim, d.conv_dim), _value_rows(d, 2 * d.key_dim_total)),
    )


# --- relayouts, on the rank's cut ---


#: Rows after the shared expert's gate|up rows in its fused operand: the per-token gate's
#: one row, then zeros, so the GEMM's output row stays a whole number of 16-byte vectors.
SH_GATE_ROWS = 8

#: `in_proj_w` rows are padded to a multiple of this (`d.in_proj_rows`): FlashInfer's low-M warp
#: split-K GEMM needs 16 | N (tp4/tep4: 4120 -> 4128). The checkpoint fills
#: `sum(d.in_proj_split)` rows; the core zeroes the rest after load.
IN_PROJ_ROW_ALIGN = 16


def _cat_rows(t: tuple[torch.Tensor, ...], d) -> torch.Tensor:
    """(gate, up, gate logit) -> `[gate ; up ; logit ; zeros]`, one GEMM for gate|up and the logit.

    Gate first: the half order `flashinfer_silu_and_mul` reads.
    """
    gate, up, sh_gate = t
    pad = sh_gate.new_zeros(SH_GATE_ROWS - 1, sh_gate.shape[-1])
    return torch.cat((gate, up, sh_gate, pad), dim=0).contiguous()


def _attn_in_w(t: tuple[torch.Tensor, ...], d) -> torch.Tensor:
    """(q_proj, k_proj, v_proj, index_qk_proj) at this rank's cut -> `[q|k|v|gate|index|pad]`.

    `q_proj` rows are `[heads, 2, head_dim]` (q, then gate). q goes first so the GEMM output
    leads with the `[q | k | v]` the fused norm+rope rewrites in place.
    """
    q, k, v, idx = t
    heads, head_dim, hidden = d.heads_q, d.head_dim, q.shape[-1]
    qg = q.reshape(heads, 2, head_dim, hidden)
    pad = q.new_zeros(d.index_head_dim, hidden)
    rows = (qg[:, 0].reshape(heads * head_dim, hidden), k, v)
    rows += (qg[:, 1].reshape(heads * head_dim, hidden), idx, pad)
    return torch.cat(rows, dim=0).contiguous()


def _expert_fc1_w(t: tuple[torch.Tensor, ...], d) -> torch.Tensor:
    """(up, gate) e4m3 codes -> one expert's FC1 operand `[up ; gate]`, plain row-major."""
    up, gate = t
    return torch.cat([up, gate], dim=0).contiguous()


def _expert_fc1_s(t: tuple[torch.Tensor, ...], d) -> torch.Tensor:
    """(up, gate) bf16 block scales -> one expert's `[up ; gate]` FC1 scales in fp32."""
    up, gate = t
    return torch.cat([up, gate], dim=0).to(EXPERT_SCALE_DTYPE).contiguous()


def _expert_fc2_s(t: torch.Tensor, d) -> torch.Tensor:
    """`down_proj` bf16 block scales -> fp32; FC2's e4m3 codes are used as stored."""
    return t.to(EXPERT_SCALE_DTYPE).contiguous()


# --- which layers carry a weight ---


def _attn_layers(d):
    """The full-attention layers (`i % 4 == 3` on this checkpoint)."""
    return [i for i, a in enumerate(d.is_attn) if a]


def _gdn_layers(d):
    return [i for i, a in enumerate(d.is_attn) if not a]


def _vision_blocks(d):
    """The vision tower's blocks. Keyed `l{b}_vis_*`: block `b`, not decoder layer `b`."""
    return range(d.v_depth)


def _mtp_only(d):
    """The MTP module, at its own layer index `num_hidden_layers`, only when it is live."""
    return [d.num_layers] if d.mtp_enabled else []


# --- where the bytes come from ---


def _ple(suffix):
    return lambda d, i: [(f"{_LM}.layers.{d.ple_layer}.ple.{suffix}", None)]


def _ngram_prefix(ple_layer: int) -> str:
    """The quantized n-gram submodule, holding the shards and their per-tensor scale."""
    return f"{_LM}.layers.{ple_layer}.ple.ple_embedding.ngram_embedding"


def _visual(suffix):
    return lambda d, i: [(f"{_VISUAL}.{suffix}", None)]


def _vision_block(suffix):
    return lambda d, i: [(f"{_VISUAL}.blocks.{i}.{suffix}", None)]


def _window(d) -> tuple[int, int]:
    """(first expert, count) of this rank's routed-expert window, trunk and draft alike."""
    return d.expert_offset, d.local_experts


def _experts(prefix, *suffixes, window):
    """One source per expert of `window(d)`, at its local index."""

    def build(d, i):
        start, count = window(d)
        base = prefix(i)
        out = []
        for local in range(count):
            keys = tuple(f"{base}.experts.{start + local}.{s}" for s in suffixes)
            out.append((keys if len(keys) > 1 else keys[0], (local,)))
        return out

    return build


def tpb_blocks(d) -> list[tuple[int, int]]:
    """(expert, intermediate block) of the 640 blocks this rank's block-major copy holds
    (`MOE_TPB_ENV`): j < 512 is expert j's block (rank - j) % 4, j >= 512 is expert
    4 (j - 512) + rank's block 4."""
    blocks = d.moe_inter // FP8_BLOCK
    r, n = d.tp_rank, d.num_experts
    return [
        (j, (r - j) % 4) if j < n else (4 * (j - n) + r, blocks - 1) for j in range(n * blocks // 4)
    ]


def _tpb_experts(prefix, suffixes, cut):
    """One source per block of `tpb_blocks(d)`, at its block index, read through `cut(b)`."""

    def build(d, i):
        base = prefix(i)
        out = []
        for j, (e, b) in enumerate(tpb_blocks(d)):
            keys = tuple(f"{base}.experts.{e}.{s}" for s in suffixes)
            c = cut(b)
            out.append((keys, (j,), (c,) * len(keys)) if len(keys) > 1 else (keys[0], (j,), c))
        return out

    return build


def _tpb_fc1_w(t: tuple[torch.Tensor, ...], d) -> torch.Tensor:
    up, gate = t
    return torch.cat([up, gate], dim=0).contiguous()


def _tpb_fc1_s(t: tuple[torch.Tensor, ...], d) -> torch.Tensor:
    """(up, gate) block-scale rows of one block -> `[2, hidden / 128]` fp32."""
    up, gate = t
    return torch.cat([up, gate], dim=0).to(EXPERT_SCALE_DTYPE).contiguous()


def _tpb_fc2_s(t: torch.Tensor, d) -> torch.Tensor:
    """down_proj's block-scale column of one block -> `[hidden / 128]` fp32."""
    return t.reshape(-1).to(EXPERT_SCALE_DTYPE).contiguous()


def _tpb_sh_gu(t: tuple[torch.Tensor, ...], d) -> torch.Tensor:
    """(gate, up) rows of one 32-column shared-expert item -> `[64, hidden]`.

    8 groups of 4 columns, each `[gate 4 ; up 4]` (`moe_tpb_decode.shared_blocks`).
    """
    gate, up = t
    return (
        torch.stack([gate.reshape(8, 4, -1), up.reshape(8, 4, -1)], dim=1)
        .reshape(64, -1)
        .contiguous()
    )


def _block_shared(tag, prefix, layers):
    """This rank's TP slice of a MoE layer's bf16 shared expert (`moe_tpb_decode.SharedBlocks`).

    Columns `[rank * w, (rank + 1) * w)` (w = shared_inter / 4) in 5 items of 32: gate / up
    rows interleaved by 4 (`tpb_sh_gu [5, 64, hidden]`) and down_proj columns
    (`tpb_sh_dn [5, hidden, 32]`).
    """
    items, cols = 5, 32

    def lo(d, sb):
        return d.tp_rank * (d.shared_inter // 4) + cols * sb

    return (
        W(
            f"{tag}tpb_sh_gu",
            shape=lambda d: (items, 64, d.hidden),
            src=lambda d, i: [
                (
                    (
                        f"{prefix(i)}.shared_expert.gate_proj.weight",
                        f"{prefix(i)}.shared_expert.up_proj.weight",
                    ),
                    (sb,),
                    ((slice(lo(d, sb), lo(d, sb) + cols),), (slice(lo(d, sb), lo(d, sb) + cols),)),
                )
                for sb in range(items)
            ],
            transform=_tpb_sh_gu,
            layers=_tpb_layers(layers),
        ),
        W(
            f"{tag}tpb_sh_dn",
            shape=lambda d: (items, d.hidden, cols),
            src=lambda d, i: [
                (
                    f"{prefix(i)}.shared_expert.down_proj.weight",
                    (sb,),
                    (slice(None), slice(lo(d, sb), lo(d, sb) + cols)),
                )
                for sb in range(items)
            ],
            layers=_tpb_layers(layers),
        ),
    )


def _tpb_layers(layers):
    """`layers` when the block-major copy is on (`d.moe_tpb`), else none."""

    def pick(d):
        if not getattr(d, "moe_tpb", False):
            return []
        return range(d.num_layers) if layers is ALL_LAYERS else list(layers(d))

    return pick


def _block_experts(tag, prefix, layers):
    """One MoE layer's routed experts block-major (`MOE_TPB_ENV`), one entry per `tpb_blocks(d)`."""
    rows = lambda b: (slice(FP8_BLOCK * b, FP8_BLOCK * (b + 1)),)  # noqa: E731
    srow = lambda b: (slice(b, b + 1),)  # noqa: E731
    cols = lambda b: (slice(None), slice(FP8_BLOCK * b, FP8_BLOCK * (b + 1)))  # noqa: E731
    scol = lambda b: (slice(None), slice(b, b + 1))  # noqa: E731
    n = lambda d: len(tpb_blocks(d))  # noqa: E731
    return (
        W(
            f"{tag}tpb_w1",
            shape=lambda d: (n(d), 2 * FP8_BLOCK, d.hidden),
            dtype=f8,
            src=_tpb_experts(prefix, ("up_proj.weight", "gate_proj.weight"), rows),
            transform=_tpb_fc1_w,
            layers=_tpb_layers(layers),
        ),
        W(
            f"{tag}tpb_s1",
            shape=lambda d: (n(d), 2, d.hidden // FP8_BLOCK),
            dtype=EXPERT_SCALE_DTYPE,
            src=_tpb_experts(
                prefix, ("up_proj.weight_scale_inv", "gate_proj.weight_scale_inv"), srow
            ),
            transform=_tpb_fc1_s,
            layers=_tpb_layers(layers),
        ),
        W(
            f"{tag}tpb_w2",
            shape=lambda d: (n(d), d.hidden, FP8_BLOCK),
            dtype=f8,
            src=_tpb_experts(prefix, ("down_proj.weight",), cols),
            layers=_tpb_layers(layers),
        ),
        W(
            f"{tag}tpb_s2",
            shape=lambda d: (n(d), d.hidden // FP8_BLOCK),
            dtype=EXPERT_SCALE_DTYPE,
            src=_tpb_experts(prefix, ("down_proj.weight_scale_inv",), scol),
            transform=_tpb_fc2_s,
            layers=_tpb_layers(layers),
        ),
    )


def _trunk_mlp(i):
    return f"{_LM}.layers.{i}.mlp"


def _mtp_mlp(i):
    return f"{_MTP_LAYER}.mlp"


def _in_proj(d, i):
    """GDN input projections at this rank's heads, in `in_proj_w`'s row order
    `[q | k | v (conv_dim) | z (value_dim) | a | b]` (`d.in_proj_split`)."""
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


#: One routed expert's checkpoint tensors, left unread by a rank outside its window.
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
    """Trunk attention input sources: q/k/v_proj at this rank's heads, replicated index_qk_proj."""
    p = f"{_LM}.layers.{i}.self_attn"
    keys = (
        f"{p}.q_proj.weight",
        f"{p}.k_proj.weight",
        f"{p}.v_proj.weight",
        f"{p}.indexer.index_qk_proj.weight",
    )
    return [(keys, None, (_q_rows(d), _kv_rows(d), _kv_rows(d), None))]


def _mtp_attn_in_fused(d, i):
    """`_attn_in_fused` for the draft layer."""
    p = f"{_MTP_LAYER}.self_attn"
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


def _index_branch(tag, prefix, layers):
    """The QSA index branch's two per-head norms; its projection is part of `attn_in_w`."""
    return (
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
    """The replicated bf16 shared expert; its gate logit shares the gate|up operand."""
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
    """One MoE layer's routed experts at this rank's window, as `fp8_block_scale_moe_runner`
    takes them.

    Plain row-major e4m3 (no shuffle, swizzle or padding) with one fp32 scale per 128x128
    block; whole experts only, no cut reaches the intermediate.
    """
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
    """`declare`, `manifest` and `load` over `WEIGHTS`, at one rank's share.

    `declare` and `manifest` share `_targets`, so they agree on which parameters exist.
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
        W(
            "attn_in_w",
            shape=lambda d: (d.attn_in_rows, d.hidden),
            src=_attn_in_fused,
            transform=_attn_in_w,
            layers=_attn_layers,
        ),
        *_attn_out_and_norms("", "{p}.self_attn", _attn_layers),
        *_index_branch("", "{p}.self_attn.indexer", _attn_layers),
        # --- the Gated-DeltaNet layers, at this rank's key and value heads ---
        W(
            "in_proj_w",
            shape=lambda d: (d.in_proj_rows, d.hidden),
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
        # Stored `[conv_dim, 1, width]`; the core takes the `[conv_dim, width]` view after load.
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
        # --- MoE on every layer: routed-expert window, replicated router and shared expert ---
        W("router", shape=lambda d: (d.num_experts, d.hidden), src="{p}.mlp.gate.weight"),
        *_routed_experts("", _trunk_mlp, ALL_LAYERS),
        *_block_experts("", _trunk_mlp, ALL_LAYERS),
        *_block_shared("", _trunk_mlp, ALL_LAYERS),
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
        # The table's scale is on `ngram_embedding` with the shards, not on `ple_embedding`.
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
        # The merger's norm runs before the 4-to-1 shuffle, so it is `hidden` wide.
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
        # A trunk full-attention layer plus a two-branch front end; keys carry no `{p}`.
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
        W(
            "mtp_attn_in_w",
            shape=lambda d: (d.attn_in_rows, d.hidden),
            src=_mtp_attn_in_fused,
            transform=_attn_in_w,
            layers=_mtp_only,
        ),
        *_attn_out_and_norms("mtp_", f"{_MTP_LAYER}.self_attn", _mtp_only),
        *_index_branch("mtp_", f"{_MTP_LAYER}.self_attn.indexer", _mtp_only),
        W(
            "mtp_router",
            shape=lambda d: (d.num_experts, d.hidden),
            src=f"{_MTP_LAYER}.mlp.gate.weight",
            layers=_mtp_only,
        ),
        *_routed_experts("mtp_", _mtp_mlp, _mtp_only),
        *_block_experts("mtp_", _mtp_mlp, _mtp_only),
        *_block_shared("mtp_", _mtp_mlp, _mtp_only),
        *_shared_expert("mtp_", _mtp_mlp, _mtp_only),
        # The module's own stack-final mixer, which is its `shared_head`.
        *_mixer("mtp_fhc", f"{_MTP}.hyper_connection_mixer", _mtp_only),
    )

    def dims(self, core) -> SimpleNamespace:
        """This rank's `rank_dims` bundle; `core.vision_config` must be set before `declare`."""
        return rank_dims(
            core.model_config.pretrained_config,
            core.vision_config,
            core.model_config.mapping,
            mtp_enabled=getattr(core.model_config, "spec_config", None) is not None,
            dtype=core.model_config.torch_dtype,
        )

    def expected_unconsumed(self, d: SimpleNamespace, keys) -> set:
        """Keys this rank leaves unread: off-window experts, and `mtp.*` without MTP.

        Computed from `d`, not the checkpoint, so a real weight's key can never land here.
        """
        start, count = _window(d)

        def off_window(i):
            if getattr(d, "moe_tpb", False):
                return []  # the block-major copy reads a block of every expert
            return [e for e in range(d.num_experts) if not start <= e < start + count]

        out = {
            f"{_trunk_mlp(i)}.experts.{e}.{t}"
            for i in range(d.num_layers)
            for e in off_window(i)
            for t in _EXPERT_TENSORS
        }
        if not d.mtp_enabled:
            return out | {k for k in keys if k.startswith(f"{_MTP}.")}
        out.update(
            f"{_mtp_mlp(None)}.experts.{e}.{t}"
            for e in off_window(d.num_layers)
            for t in _EXPERT_TENSORS
        )
        return out

    def declare(self, d: SimpleNamespace) -> nn.ParameterDict:
        """Allocate every weight at this rank's share (meta-init intercepts `torch.empty`)."""
        w = nn.ParameterDict()
        for entry in self.WEIGHTS:
            dtype = entry.dtype or d.dtype
            for key, _, _ in self._targets(entry, d):
                w[key] = nn.Parameter(
                    torch.empty(*entry.shape(d), dtype=dtype), requires_grad=False
                )
        return w

    def _targets(self, entry: W, d: SimpleNamespace) -> list[tuple[str, int | None, str]]:
        """`[(param key, layer index | None, checkpoint layer prefix), ...]` of one entry."""
        layers = entry.layer_indices(d)
        if layers is None:
            return [(entry.name, None, "")]
        return [(f"l{i}_{entry.name}", i, f"{_LM}.layers.{i}") for i in layers]

    def manifest(self, d: SimpleNamespace) -> dict[str, list[tuple]]:
        """Param key -> `[(checkpoint key, index | None, cut | None, transform | None), ...]`."""
        rows: dict[str, list[tuple]] = {}
        for entry in self.WEIGHTS:
            for key, i, prefix in self._targets(entry, d):
                rows[key] = [
                    (ckpt, index, cut, entry.transform)
                    for ckpt, index, cut in entry.sources(d, i, prefix)
                ]
        return rows

    def read(self, weights, ckpt_key, cut, transform, d, device=None) -> torch.Tensor:
        """One manifest source -> its destination's tensor: keys read through cuts, transformed."""
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
        """Fill `param` from its manifest rows on the parameter's device; return the keys read."""
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
        """Fill every declared parameter and assert checkpoint coverage both ways."""
        core = model.model
        d = self.dims(core)
        # Declared before the manifest: its height comes from the shards.
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

        # The shell owns lm_head (untied, bf16); read only the rows it built.
        cut, pad = lm_head_share(model.lm_head.weight.shape[0], d.vocab, d.tp_rank)
        assert tuple(model.lm_head.weight.shape[1:]) == (d.hidden,), (
            f"the shell built lm_head {tuple(model.lm_head.weight.shape)}; this checkpoint's "
            f"is a dense [{d.vocab}, {d.hidden}] weight"
        )
        consumed |= self.fill(
            model.lm_head.weight, [("lm_head.weight", None, cut, pad)], weights, d
        )
        # Every parameter outside `core.w` must be lm_head.
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
    """`fp8_block_scale_moe_runner` operands of one MoE layer at this rank's window.

    `offset` / `count` are `local_expert_offset` / `local_num_experts`.
    """

    fc1_w: torch.Tensor  # [count, 2 * moe_inter, hidden] e4m3, rows `[up ; gate]`
    fc1_s: torch.Tensor  # [count, 2 * moe_inter / 128, hidden / 128] fp32
    fc2_w: torch.Tensor  # [count, hidden, moe_inter] e4m3
    fc2_s: torch.Tensor  # [count, hidden / 128, moe_inter / 128] fp32
    offset: int
    count: int
    #: the layer's block-major copy (`MOE_TPB_ENV`), or None
    blocks: "BlockWindow | None" = None


class BlockWindow(NamedTuple):
    """One MoE layer's block-major routed experts and shared-expert slice at this rank."""

    w1: torch.Tensor  # [640, 256, hidden] e4m3, a block's rows `[up ; gate]`
    s1: torch.Tensor  # [640, 2, hidden / 128] fp32
    w2: torch.Tensor  # [640, hidden, 128] e4m3
    s2: torch.Tensor  # [640, hidden / 128] fp32
    rank: int
    sh_gu: torch.Tensor  # [5, 64, hidden] bf16: the slice's gate / up items
    sh_gl: torch.Tensor  # [hidden] bf16: the gate-logit row
    sh_dn: torch.Tensor  # [5, hidden, 32] bf16: the slice's down items


def _block_window(w, d: SimpleNamespace, layer: int) -> BlockWindow | None:
    if not getattr(d, "moe_tpb", False):
        return None
    p = f"l{layer}_mtp_" if layer == d.num_layers else f"l{layer}_"
    return BlockWindow(
        w[f"{p}tpb_w1"],
        w[f"{p}tpb_s1"],
        w[f"{p}tpb_w2"],
        w[f"{p}tpb_s2"],
        d.tp_rank,
        w[f"{p}tpb_sh_gu"],
        w[f"{p}sh_gu"][2 * d.shared_inter],
        w[f"{p}tpb_sh_dn"],
    )


def expert_window(w, d: SimpleNamespace, layer: int) -> ExpertWindow:
    """Routed-expert operands of decoder layer `layer`, or of the MTP module at `d.num_layers`."""
    assert 0 <= layer < d.num_layers or (layer == d.num_layers and d.mtp_enabled), (
        f"layer {layer}: the trunk has {d.num_layers} MoE layers"
        f"{' and the MTP module is not loaded' if layer == d.num_layers else ''}"
    )
    p = f"l{layer}_mtp_" if layer == d.num_layers else f"l{layer}_"
    start, count = _window(d)
    return ExpertWindow(
        w[f"{p}fc1_w"],
        w[f"{p}fc1_s"],
        w[f"{p}fc2_w"],
        w[f"{p}fc2_s"],
        start,
        count,
        _block_window(w, d, layer),
    )


def attention_tp_size(mapping) -> int:
    """Ranks attention and K/V pool heads split over: the TP group, or 1 under attention DP."""
    return 1 if mapping.enable_attention_dp else mapping.tp_size


def _heads(total: int, ranks: int, rank: int) -> tuple[int, int]:
    """(heads on `rank`, first head); with more ranks than heads, neighbours repeat a head."""
    assert total % ranks == 0 or ranks % total == 0, (total, ranks)
    return -(-total // ranks), rank * total // ranks


def rank_dims(cfg, vcfg, mapping, *, mtp_enabled: bool, dtype: torch.dtype) -> SimpleNamespace:
    """Every width the table declares at rank `mapping.rank`, and the shares behind them."""
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

    # Routed experts: whole-expert windows over the TP group, whatever the MoE TP / EP split.
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
        in_proj_rows=-(-(2 * key_dim + 2 * value_dim + 2 * num_v_heads) // IN_PROJ_ROW_ALIGN)
        * IN_PROJ_ROW_ALIGN,
        conv_kernel=cfg.linear_conv_kernel_dim,
        # --- MoE: window `[expert_offset, expert_offset + local_experts)`, whole `moe_inter` ---
        num_experts=num_experts,
        local_experts=local_experts,
        expert_offset=local_experts * mapping.tp_rank,
        moe_tpb=moe_tpb_enabled(mapping),
        moe_inter=moe_inter,
        shared_inter=cfg.shared_expert_intermediate_size,
        vocab=cfg.vocab_size,
        ple_layer=list(cfg.ple_layer_ids)[0] - 1,  # one-based in the config
        ple_embed_dim=cfg.ple_embed_dim,
        ple_conv_kernel=cfg.ple_conv_kernel_size,
        ngram_size=cfg.ngram_size,
        ngram_heads=ngram_heads,
        ple_head_dim=cfg.ple_embed_dim // ngram_heads,
        ple_shards=cfg.split_ngram_parts,
        ple_table_dtype=PLE_TABLE_DTYPE,
        v_depth=vcfg.depth,
        v_hidden=v_hidden,
        v_inter=vcfg.intermediate_size,
        v_channels=vcfg.in_channels,
        v_tpatch=vcfg.temporal_patch_size,
        v_patch=vcfg.patch_size,
        v_pos_rows=vcfg.num_position_embeddings,
        v_merge_dim=v_hidden * vcfg.spatial_merge_size * vcfg.spatial_merge_size,
        mtp_enabled=mtp_enabled,
        dtype=dtype,
    )


def lm_head_share(rows: int, vocab: int, tp_rank: int) -> tuple[Cut, Callable]:
    """The cut and the pad that fill an lm_head the shell built with `rows` rows.

    Whole under attention DP (`rows == vocab`), else a vocab-parallel shard; an uneven vocab
    leaves the last shard short, zero-padded as the built-in LMHead does.
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
    """`ple_table` manifest rows: shard `s`, read whole, fills rows from `s * rows_per_shard`."""
    return [
        (key, (_span(s * rows_per_shard, rows_per_shard),), None, None)
        for s, key in enumerate(ngram_shard_keys(d))
    ]


def _declare_ngram_table(core, d: SimpleNamespace, weights) -> int:
    """Register the n-gram table at the height its shards concatenate to; return rows per shard.

    The shards are contiguous row chunks (global row `r` is shard `r // rows_per_shard`), so
    the flat buffer takes the hash's global rows. On CUDA it is host memory the device
    addresses; otherwise an ordinary allocation on the weights' device.
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
    if device.type != "cuda":
        table = torch.empty(rows, head_dim, dtype=dtype, device=device)
        core._ple_table_host = None
    else:
        table, core._ple_table_host = _host_resident_empty(rows, head_dim, dtype, device)
    core.w["ple_table"] = nn.Parameter(table, requires_grad=False)
    # Host offload relies on `nn.Parameter` sharing storage rather than copying.
    assert core.w["ple_table"].data_ptr() == table.data_ptr(), (
        "wrapping the n-gram table in nn.Parameter moved its storage"
    )
    return rows_per_shard
