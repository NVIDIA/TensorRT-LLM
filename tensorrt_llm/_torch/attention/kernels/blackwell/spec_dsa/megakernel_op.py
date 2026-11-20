# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""specDSA megakernel, host side: fused DeepSeek-V4-Flash decode for 1 to 47 rows on B200 (SM100, 148 SMs).

One cooperative launch streams each row's FP8 indexer K cache once, computes the indexer logits, selects the exact
top-512 compressed positions (sampled threshold with a count certificate and exact fallbacks), maps them to
compressed-KV rows and runs the sparse MLA attention over them plus the 128-row sliding window and the attention sink.

``prepare`` builds the per-step state (split descriptors from the indexer block table and lengths, window positions,
workspace); ``run`` launches the kernel on the current stream and writes ``topk`` (int32 compressed positions) and
``out`` (bf16). The kernel is compiled on first use per (device, batch size), outside CUDA-graph capture, or ahead of
it by ``precompile``.

Contract:
- A state belongs to one batch size and one caller. Its workspace carries a generation counter across launches, so
  launches that share a state, and ``prepare`` calls that refresh it, must be stream-ordered.
- ``topk`` and ``out`` must not alias any input.
- The launch is cooperative and needs all 148 SMs; it fails at launch on a partitioned device (MPS limits, green
  contexts).
- Block tables must hold valid page ids for every page a row touches: the indexer table covers whole 128-position
  tiles (``4 * ceil(n / 128)`` entries), the compressed-KV table covers ``ceil(n / 32)`` entries.
- The kernel does not trap. It records per-row errors in its workspace (see ``error_bits``); a row with an error has
  undefined outputs.
- Under ``compute-sanitizer --tool synccheck`` only 1 to 4 rows complete: from 5 rows on, every pass after the first
  re-initializes its mbarriers without invalidating them (the 3 to 4-row teams run a single pass).
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass

import torch

from .config import (
    DS_V4_FLASH,
    DS_V4_FLASH_FP4,
    INDEX_TILE,
    NUM_SMS,
    SpecDsaConfig,
    check_device,
    supported_spec,
)

__all__ = [
    "ERR_ABORTED",
    "ERR_MISMATCH",
    "ERR_NLIMIT",
    "ERR_STALE",
    "ERR_TIMEOUT",
    "SpecDsaMegakernelState",
    "error_bits",
    "is_supported",
    "precompile",
    "prepare",
    "run",
    "supported_batch_sizes",
]

# Per-row error bits recorded by the kernel (``error_bits``).
ERR_STALE = 1  # the state's lengths differ from run()'s ctx_lens: prepare() was not refreshed
ERR_TIMEOUT = 2  # a wait for another CTA timed out (10 s); the launch was aborted
ERR_NLIMIT = 4  # a compressed length outside [0, max_index_len]
ERR_MISMATCH = 8  # the stream and team CTAs disagreed on the selection path
ERR_ABORTED = 16  # this row stopped because another row or CTA aborted

_BASE = dict(
    debug=False,
    stamps=False,
    phys_block_kv=DS_V4_FLASH.index_page_tokens,
    skew=0,
    prewarm=True,
    coop=True,
    qk_ahead=False,
    gjoin=True,
    pref=True,
    spin_ms=10000,
    relst=True,
    acqpoll=True,
    late_done=4,
)
# Each batch size extends the previous set. From 5 rows on, two teams of 8 CTAs serve the rows in turn while 132 CTAs
# stream the indexer K cache; 9 to 47 rows add the ring of logits slots (ring).
_BS1 = dict(swag=True, nohoist=True, rs24=True, rs24_t4split=True, rs24_lsu=12)
_BS2 = dict(_BS1, math_regs=216, kv_stages=3, wdelay=0, early=True, tearly=True)
_BS4 = dict(
    _BS2,
    nteams=2,
    ppref=False,
    presel=True,
    pdec=True,
    warm1=True,
    swpre=False,
    pdirect=True,
    tsrv1=True,
    btcl=True,
    tl1r=True,
    tsz=True,
    tsz2=True,
    tsz01=True,
    bpf=True,
)
# 3 and 4 rows: one team of 4 CTAs per row (two row groups x two V column halves of 256, one KV gather per tile), so
# every row has its own team and the launch is a single pass. The candidate-row L2 prefetch (pref) is off: it measured
# slower.
_BS4T = dict(_BS2, team4=True, nteams=4, rs24_dry=False, pref=False)
_BS8 = dict(_BS4, tszall=True, dpf=True, b1l=True, b1l_pub=True, b1l_upub=2)
_BS16 = dict(_BS8, ring=True, ring_w12=True)
_CONFIGS: dict[int, dict] = {
    1: _BS1,
    2: _BS2,
    3: _BS4T,
    4: _BS4T,
    **{b: _BS8 for b in range(5, 9)},
    **{b: _BS16 for b in range(9, 48)},
}

# FP4 spec, 5 or more rows: the shared teams are four teams of 4 CTAs (team4: more rows in flight than two teams of 8),
# the candidate publisher batches 3 atomics per lane, the row's split ranges rotate over the stream CTAs (drot: the CTAs
# that hold the dense end of the sequence, and with it the heavy candidate publication, differ from row to row) and two
# dedicated CTAs compute the thresholds (tcta: no stream CTA with an empty range blocks in the threshold wait).
_FP4_SHARED = dict(team4=True, nteams=4, rs24_dry=False, b1l_upub=3, drot=37, tcta=2)
# FP4 spec, 1 to 4 rows (the stream is shorter than the FP8 one, the fixed costs of the start and of the row
# boundaries decide): the K tiles' TMA descriptors are prefetched at entry and the row lengths are loaded before the
# barrier setup (early / tearly) at one row; from two rows on the next row's descriptor is loaded one row ahead (dpf)
# and the split ranges rotate over the stream CTAs from row to row (drot), as the shared-team sets from five rows on
# already do.
_FP4_SMALL = {
    1: dict(early=True, tearly=True),
    2: dict(dpf=True, drot=37),
    3: dict(dpf=True, drot=37),
    4: dict(dpf=True, drot=37),
}
_FP4_OVERRIDES = {**_FP4_SMALL, **{b: _FP4_SHARED for b in range(5, 33)}}

# The knob sets of the megakernel per spec (kernel_key) and batch size: the base knobs plus the set of the batch size
# (and, for the FP4 spec, its overrides).
_KNOBS = {
    spec.kernel_key: {
        b: {
            **_BASE,
            **knobs,
            **(_FP4_OVERRIDES.get(b, {}) if spec.index_k_dtype == "fp4" else {}),
        }
        for b, knobs in _CONFIGS.items()
        if b <= spec.megakernel_max_batch
    }
    for spec in (DS_V4_FLASH, DS_V4_FLASH_FP4)
}

_lock = threading.Lock()
_engines: dict[tuple, dict] = {}


def _kernel_modules() -> tuple:
    from . import megakernel, megakernel_lib

    return megakernel_lib, megakernel


def supported_batch_sizes(config: SpecDsaConfig | None = None) -> range:
    """Batch sizes the megakernel serves for ``config`` (default: DS-V4-Flash): 1 up to ``megakernel_max_batch``."""
    return range(1, supported_spec(config).megakernel_max_batch + 1)


def is_supported(
    device: torch.device | None = None, config: SpecDsaConfig | None = None
) -> tuple[bool, str]:
    """Whether the megakernel serves ``config`` (default: DS-V4-Flash) on ``device``.

    That takes a supported spec with a megakernel, CuTe DSL installed and SM100 with all 148 SMs (one CTA per SM).
    """
    try:
        if supported_spec(config).megakernel_max_batch < 1:
            return False, "the spec has no megakernel"
    except NotImplementedError as e:
        return False, str(e)
    return check_device(device)


def _check_batch_size(batch_size: int, config: SpecDsaConfig) -> None:
    sizes = supported_batch_sizes(config)
    if batch_size not in sizes:
        raise ValueError(f"batch size {batch_size} not in [{sizes.start}, {sizes.stop - 1}]")


def _device_index(device: torch.device) -> int:
    device = torch.device(device)
    return device.index if device.index is not None else torch.cuda.current_device()


def _engine(device: torch.device, batch_size: int, config: SpecDsaConfig | None = None) -> dict:
    config = supported_spec(config)
    _check_batch_size(batch_size, config)
    key = (_device_index(device), batch_size, config.kernel_key)
    with _lock:
        if key not in _engines:
            lib, mk = _kernel_modules()
            kernel = mk.SpecDsaMegakernel(
                num_sms=NUM_SMS,
                policy=lib.POL_EVICT_FIRST if batch_size > 1 else lib.POL_NONE,
                full_attention_tiles=True,
                nrows=batch_size,
                config=config,
                **_KNOBS[config.kernel_key][batch_size],
            )
            _engines[key] = {"kernel": kernel, "fn": None}
        return _engines[key]


def _desc_params(kernel, batch_size: int, row: int) -> tuple[int, bool, int]:
    """(rot, zlast, zc) of row ``row``'s split descriptor; they select the descriptor kernel variant."""
    shared = batch_size > int(kernel.nteams) and not kernel.tcta
    # With shared teams one stream CTA gets an empty range in some rows and computes that row's threshold early
    # (not with dedicated T-server CTAs: every stream CTA has a range in every row).
    zlast = shared and (
        (kernel.tsz and row == batch_size - 1) or (kernel.tsz2 and row >= batch_size - 2)
    )
    zc = -1
    if shared and kernel.tsz01 and (row < 2 or (kernel.tszall and row < batch_size - 2)):
        zc = (kernel.ns - 2) if row < 2 else (kernel.ns - 2 - row // 2)
    # drot: the ranges of row r rotate by (r mod 8) * drot CTAs (8 distinct descriptor-kernel variants per layout)
    return ((row % 8) * int(kernel.drot)) % kernel.ns, bool(zlast), int(zc)


def precompile(device: torch.device, batch_size: int, config: SpecDsaConfig | None = None) -> None:
    """Compile the kernels for ``batch_size`` rows on ``device`` (main kernel about 15 s, plus the descriptor kernels).

    Must run outside CUDA-graph capture. ``run`` compiles on first use otherwise.
    """
    e = _engine(device, batch_size, config)
    _, mk = _kernel_modules()
    with _lock:
        if e["fn"] is None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "spec_dsa: compile the kernel (precompile) before CUDA-graph capture"
                )
            e["fn"] = mk.compile_mk(e["kernel"])
        kernel = e["kernel"]
        for r in range(batch_size):
            rot, zlast, zc = _desc_params(kernel, batch_size, r)
            mk.compile_desc(NUM_SMS, kernel.ns, rot, zlast, zc, kernel.fp4)


@dataclass
class SpecDsaMegakernelState:
    """Per-step metadata and workspace of one caller; see ``prepare``.

    ``ws64`` holds ``WS_WORDS`` int64 words per row (generation counter, error words, selection directives), ``hist``,
    ``board`` and ``region`` the per-row selection scratch. ``gidx``, ``dbg``, ``stamps`` and ``ctl`` are kernel
    arguments used only by diagnostic builds. ``config`` is the spec the state was built for.
    """

    batch_size: int
    desc: torch.Tensor
    swa_local: torch.Tensor
    swa_lens: torch.Tensor
    gidx: torch.Tensor
    ws64: torch.Tensor
    hist: torch.Tensor
    board: torch.Tensor
    region: torch.Tensor
    dbg: torch.Tensor
    stamps: torch.Tensor
    ctl: torch.Tensor
    config: SpecDsaConfig


def _host_ints(t: torch.Tensor, count: int, name: str) -> list[int]:
    v = t.reshape(-1)
    if v.numel() < count:
        raise ValueError(f"{name} has {v.numel()} entries, need {count}")
    return [int(x) for x in v[:count].tolist()]


def prepare(
    ctx_lens: torch.Tensor,
    kv_lens: torch.Tensor,
    idx_block_table: torch.Tensor,
    state: SpecDsaMegakernelState | None = None,
    *,
    config: SpecDsaConfig | None = None,
) -> SpecDsaMegakernelState:
    """Build (or refresh in place) the per-step state for ``B = ctx_lens.numel()`` rows.

    Args:
        ctx_lens: int32 [B], compressed (indexer) length of each row, in [512, 2**18] for DS-V4-Flash.
        kv_lens: int32 [B], uncompressed length of each row (>= 128); the window is its last 128 positions.
        idx_block_table: int32 [B, pages] on the target device, indexer K pages of each row (32 compressed positions
            per page; whole 128-position tiles, so at least ``4 * ceil(n / 128)`` valid entries).
        state: a state of the same batch size to refresh; keeps its workspace (and generation counter) and its spec.
        config: the spec (default: DS-V4-Flash, the only one the kernels serve).

    The lengths are checked on the host: pass host (CPU) tensors to avoid a device-to-host copy; device tensors work
    but synchronize.
    """
    if state is not None:
        if config is not None and config.kernel_key != state.config.kernel_key:
            raise ValueError(f"state is for {state.config.name!r}, got {config.name!r}")
        config = state.config
    config = supported_spec(config)
    batch_size = int(ctx_lens.numel())
    _check_batch_size(batch_size, config)
    device = idx_block_table.device
    if device.type != "cuda":
        raise ValueError("idx_block_table must be a CUDA tensor")
    ctx_host = _host_ints(ctx_lens, batch_size, "ctx_lens")
    kv_host = _host_ints(kv_lens, batch_size, "kv_lens")
    if any(not config.min_index_len <= n <= config.max_index_len for n in ctx_host):
        raise ValueError(
            f"compressed lengths must lie in [{config.min_index_len}, {config.max_index_len}], got {ctx_host}"
        )
    if any(n < config.window for n in kv_host):
        raise ValueError(f"uncompressed lengths must be at least {config.window}, got {kv_host}")
    need_pages = config.pages_per_tile * math.ceil(max(ctx_host) / INDEX_TILE)
    if idx_block_table.dim() != 2 or idx_block_table.shape[0] < batch_size:
        raise ValueError(f"idx_block_table must be [B, pages] with B >= {batch_size}")
    if idx_block_table.shape[1] < need_pages:
        raise ValueError(
            f"idx_block_table has {idx_block_table.shape[1]} pages per row, need {need_pages}"
        )
    if state is not None and state.batch_size != batch_size:
        raise ValueError(f"state is for {state.batch_size} rows, got {batch_size}")
    _, mk = _kernel_modules()
    kernel = _engine(device, batch_size, config)["kernel"]
    rsz = kernel.row_sizes()
    if state is None:
        state = SpecDsaMegakernelState(
            batch_size=batch_size,
            desc=torch.zeros(batch_size * rsz["desc"], dtype=torch.int32, device=device),
            swa_local=torch.empty((batch_size, config.window), dtype=torch.int32, device=device),
            swa_lens=torch.full((batch_size,), config.window, dtype=torch.int32, device=device),
            gidx=torch.full(
                (batch_size, config.selected_rows), -7, dtype=torch.int32, device=device
            ),
            ws64=torch.zeros(batch_size * rsz["ws"], dtype=torch.int64, device=device),
            hist=torch.zeros(batch_size * rsz["hist"], dtype=torch.int32, device=device),
            board=torch.zeros(batch_size * rsz["board"], dtype=torch.int64, device=device),
            region=torch.zeros(batch_size * rsz["region"], dtype=torch.int64, device=device),
            dbg=torch.empty(config.max_index_len + 512, dtype=torch.float32, device=device),
            stamps=torch.zeros(
                NUM_SMS
                * mk.NSTAMP
                * (
                    56
                    if batch_size > 16
                    else (40 if batch_size > 8 else (4 if batch_size > 1 else 1))
                ),
                dtype=torch.int64,
                device=device,
            ),
            ctl=torch.zeros(batch_size * rsz["ctl"], dtype=torch.int32, device=device),
            config=config,
        )
    torch.add(
        torch.tensor(kv_host, dtype=torch.int32).to(device, non_blocking=True).unsqueeze(1),
        torch.arange(-config.window, 0, dtype=torch.int32, device=device),
        out=state.swa_local,
    )
    ctx = torch.tensor(ctx_host, dtype=torch.int32).to(device, non_blocking=True)
    bt = idx_block_table[:batch_size].to(dtype=torch.int32).contiguous()
    with _lock:  # the descriptor kernels compile on first use
        for r in range(batch_size):
            rot, zlast, zc = _desc_params(kernel, batch_size, r)
            mk.build_desc_device(
                ctx[r : r + 1],
                bt[r : r + 1],
                state.desc[r * rsz["desc"] : (r + 1) * rsz["desc"]],
                NUM_SMS,
                kernel.ns,
                rot=rot,
                zlast=zlast,
                zc=zc,
                ladder_words=kernel.fp4,
            )
    return state


def error_bits(state: SpecDsaMegakernelState) -> torch.Tensor:
    """Per-row error bits (``ERR_*``) accumulated over the launches of ``state``: an int64 [B] device view, no sync.

    Zero means every launch so far completed normally. The bits are sticky; a fresh state starts at zero.
    """
    _, mk = _kernel_modules()
    return state.ws64.view(state.batch_size, mk.WS_WORDS)[:, mk.W_ERRB]


def _check_rows(name: str, t: torch.Tensor, row_bytes: int) -> None:
    if t.element_size() != 1 or t.dim() < 2 or t.shape[-1] != row_bytes or not t.is_contiguous():
        raise ValueError(
            f"{name} must be a contiguous 1-byte tensor with {row_bytes}-byte rows, "
            f"got {t.dtype} {tuple(t.shape)}"
        )


def _check_numel(name: str, t: torch.Tensor, numel: int, dtype: torch.dtype | None = None) -> None:
    if t.numel() != numel or (dtype is not None and t.dtype != dtype) or not t.is_contiguous():
        raise ValueError(
            f"{name} must be contiguous with {numel} elements"
            + (f" of {dtype}" if dtype is not None else "")
            + f", got {t.dtype} {tuple(t.shape)}"
        )


def run(
    state: SpecDsaMegakernelState,
    q_idx: torch.Tensor,
    w_idx: torch.Tensor,
    k_cache: torch.Tensor,
    idx_block_table: torch.Tensor,
    ctx_lens: torch.Tensor,
    q_att: torch.Tensor,
    swa_pool: torch.Tensor,
    bt_swa: torch.Tensor,
    c_pool: torch.Tensor,
    bt_c: torch.Tensor,
    sinks: torch.Tensor,
    topk: torch.Tensor,
    out: torch.Tensor,
    softmax_scale: float | None = None,
    *,
    q_scale: torch.Tensor | None = None,
) -> None:
    """Launch the fused kernel on the current stream.

    Args:
        state: from ``prepare`` for these lengths and block tables.
        q_idx: float8_e4m3fn [B, 64, 128] (or [B, 1, 64, 128]), indexer queries; for an FP4 spec packed E2M1, 1 byte
            per two values ([B, 64, 64]).
        w_idx: float32 [B, 64], indexer head weights.
        k_cache: uint8 [pages, ...] with ``config.index_page_bytes`` bytes per page: 4224 for FP8 (32 x 128 FP8 values,
            then 32 float32 scales), 2176 for FP4 (32 x 64 bytes of E2M1 pairs, then 32 int32 of four UE8M0 scales, one
            per 32 values).
        idx_block_table: int32 [B, pages], as given to ``prepare``.
        ctx_lens: int32 [B] on the device, the same lengths as given to ``prepare``.
        q_att: float8_e4m3fn [B, 64, 512], attention queries.
        swa_pool: float8_e4m3fn [..., 512], sliding-window KV rows (128 rows per page); bt_swa: int32 [B, pages].
        c_pool: float8_e4m3fn [..., 512], compressed KV rows (32 rows per page); bt_c: int32 [B, pages], each row
            16-byte aligned (row stride a multiple of 4 entries).
        sinks: float32 [64], per-head attention sink.
        topk: int32 [B, 512] output, the selected compressed positions (in no particular order).
        out: bfloat16 [B, 64 * 512] output.
        softmax_scale: attention softmax scale (default: ``1 / sqrt(kv_dim)`` of the spec).
        q_scale: int32 [B, 64] (or [B, 1, 64]), the four UE8M0 query scales of each head in one int32 (FP4 specs only).

    The shapes are those of the DS-V4-Flash spec; ``state.config`` gives the sizes the checks use.
    """
    config = state.config
    b = state.batch_size
    fp4 = config.index_k_dtype == "fp4"
    if softmax_scale is None:
        softmax_scale = config.default_softmax_scale
    q_values = config.index_head_dim // 2 if fp4 else config.index_head_dim
    _check_numel("q_idx", q_idx, b * config.index_heads * q_values)
    if fp4:
        if q_scale is None:
            raise ValueError("an FP4 spec needs q_scale (the UE8M0 scales of the FP4 queries)")
        _check_numel("q_scale", q_scale, b * config.index_heads, torch.int32)
    elif q_scale is not None:
        raise ValueError("q_scale belongs to FP4 specs")
    if q_idx.element_size() != 1:
        raise ValueError(f"q_idx must be FP8 (1 byte per value), got {q_idx.dtype}")
    _check_numel("w_idx", w_idx, b * config.index_heads, torch.float32)
    _check_numel("q_att", q_att, b * config.attn_heads * config.kv_dim)
    if q_att.element_size() != 1:
        raise ValueError(f"q_att must be FP8 (1 byte per value), got {q_att.dtype}")
    _check_numel("sinks", sinks, config.attn_heads)
    _check_numel("topk", topk, b * config.topk, torch.int32)
    _check_numel("out", out, b * config.attn_heads * config.kv_dim, torch.bfloat16)
    _check_rows("swa_pool", swa_pool, config.kv_dim)
    _check_rows("c_pool", c_pool, config.kv_dim)
    pages = int(k_cache.shape[0])
    if k_cache.element_size() != 1 or k_cache.numel() != pages * config.index_page_bytes:
        raise ValueError(
            f"k_cache must hold {config.index_page_bytes} bytes per page, got {tuple(k_cache.shape)}"
        )
    if bt_c.dtype != torch.int32 or bt_c.stride(-1) != 1 or bt_c.data_ptr() % 16 != 0:
        raise ValueError("bt_c must be int32 with unit stride and a 16-byte aligned base")
    if b > 1 and bt_c.stride(0) % 4 != 0:
        raise ValueError(f"bt_c rows must be 16-byte aligned, got a row stride of {bt_c.stride(0)}")
    e = _engine(q_idx.device, b, config)
    if e["fn"] is None:
        precompile(q_idx.device, b, config)
    e["fn"](
        k_cache.view(pages, -1),
        q_idx.reshape(-1)
        .as_strided(
            (config.index_heads, q_values, b),
            (q_values, 1, config.index_heads * q_values),
        )
        .view(torch.uint8),
        w_idx.reshape(-1).as_strided((config.index_heads, b), (1, config.index_heads)),
        idx_block_table[:b],
        ctx_lens.reshape(-1)[:b],
        state.desc,
        q_att.view(torch.uint8).view(b, config.attn_heads, config.kv_dim),
        swa_pool.view(torch.uint8).view(-1, config.kv_dim),
        c_pool.view(torch.uint8).view(-1, config.kv_dim),
        state.swa_local,
        bt_swa[:b],
        bt_c[:b],
        sinks.reshape(-1).float(),
        state.swa_lens,
        topk,
        state.gidx,
        out,
        state.ws64,
        state.hist,
        state.board,
        state.region,
        state.dbg,
        state.stamps,
        state.ctl,
        pages,
        softmax_scale,
        *(
            (q_scale.reshape(-1).as_strided((config.index_heads, b), (1, config.index_heads)),)
            if fp4
            else ()
        ),
    )
