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
"""Fused DeepSeek-V4-Flash decode sparse attention for 1, 2, 4 or 8 rows on B200 (SM100, 148 SMs).

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
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass

import torch

from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE

__all__ = [
    "DEFAULT_SOFTMAX_SCALE",
    "ERR_ABORTED",
    "ERR_MISMATCH",
    "ERR_NLIMIT",
    "ERR_STALE",
    "ERR_TIMEOUT",
    "MAX_CTX_LEN",
    "MIN_CTX_LEN",
    "SUPPORTED_BATCH_SIZES",
    "Dsv4SparseDecodeState",
    "error_bits",
    "is_supported",
    "precompile",
    "prepare",
    "run",
]

SUPPORTED_BATCH_SIZES = (1, 2, 4, 8)
NUM_SMS = 148
NUM_HEADS = 64
INDEX_DIM = 128
LATENT_DIM = 512
TOPK = 512
WINDOW = 128
INDEX_PAGE_BYTES = 32 * (INDEX_DIM + 4)  # 32 positions x (128 FP8 values + one float32 scale)
MIN_CTX_LEN = TOPK
MAX_CTX_LEN = 1 << 18
DEFAULT_SOFTMAX_SCALE = 1.0 / math.sqrt(512.0)

# Per-row error bits recorded by the kernel (``error_bits``).
ERR_STALE = 1  # the state's lengths differ from run()'s ctx_lens: prepare() was not refreshed
ERR_TIMEOUT = 2  # a wait for another CTA timed out (10 s); the launch was aborted
ERR_NLIMIT = 4  # a compressed length outside [0, MAX_CTX_LEN]
ERR_MISMATCH = 8  # the stream and team CTAs disagreed on the selection path
ERR_ABORTED = 16  # this row stopped because another row or CTA aborted

_BASE = dict(
    debug=False,
    stamps=False,
    phys_block_kv=32,
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
# Each batch size extends the previous set. At 4 and 8 rows two teams of 8 CTAs serve the rows in turn while 132 CTAs
# stream the indexer K cache.
_BS1 = dict(swag=True, nohoist=True, rs24=True, rs24_t4split=True, rs24_lsu=12)
_BS2 = dict(_BS1, math_regs=216, kv_stages=3, wdelay=0)
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
_BS8 = dict(_BS4, tszall=True, dpf=True)
_CONFIGS: dict[int, dict] = {1: _BS1, 2: _BS2, 4: _BS4, 8: _BS8}

_lock = threading.Lock()
_engines: dict[tuple[int, int], dict] = {}


def _kernel_modules() -> tuple:
    from . import cache_policy, megakernel, row_desc

    return cache_policy, megakernel, row_desc


def is_supported(device: torch.device | None = None) -> tuple[bool, str]:
    """Whether ``device`` can run the kernel: CuTe DSL installed, SM100 with all 148 SMs (one CTA per SM)."""
    if not IS_CUTLASS_DSL_AVAILABLE:
        return False, "nvidia-cutlass-dsl is not available"
    if not torch.cuda.is_available():
        return False, "CUDA is not available"
    props = torch.cuda.get_device_properties(
        device if device is not None else torch.cuda.current_device()
    )
    if (props.major, props.minor) != (10, 0):
        return False, f"requires SM100, got SM{props.major}{props.minor}"
    if props.multi_processor_count != NUM_SMS:
        return False, f"requires {NUM_SMS} SMs, got {props.multi_processor_count}"
    return True, ""


def _check_batch_size(batch_size: int) -> None:
    if batch_size not in SUPPORTED_BATCH_SIZES:
        raise ValueError(f"batch size {batch_size} not in {SUPPORTED_BATCH_SIZES}")


def _device_index(device: torch.device) -> int:
    device = torch.device(device)
    return device.index if device.index is not None else torch.cuda.current_device()


def _engine(device: torch.device, batch_size: int) -> dict:
    _check_batch_size(batch_size)
    key = (_device_index(device), batch_size)
    with _lock:
        if key not in _engines:
            cp, mk, _ = _kernel_modules()
            kernel = mk.Dsv4SparseDecodeKernel(
                num_sms=NUM_SMS,
                policy=cp.POL_EVICT_FIRST if batch_size > 1 else cp.POL_NONE,
                full_attention_tiles=True,
                nrows=batch_size,
                **_BASE,
                **_CONFIGS[batch_size],
            )
            _engines[key] = {"kernel": kernel, "fn": None}
        return _engines[key]


def _desc_params(kernel, batch_size: int, row: int) -> tuple[int, bool, int]:
    """(rot, zlast, zc) of row ``row``'s split descriptor; they select the descriptor kernel variant."""
    shared = batch_size > int(kernel.nteams)
    # With shared teams one stream CTA gets an empty range in some rows and computes that row's threshold early.
    zlast = shared and (
        (kernel.tsz and row == batch_size - 1) or (kernel.tsz2 and row >= batch_size - 2)
    )
    zc = -1
    if shared and kernel.tsz01 and (row < 2 or (kernel.tszall and row < batch_size - 2)):
        zc = (kernel.ns - 2) if row < 2 else (kernel.ns - 2 - row // 2)
    return (row * int(kernel.drot)) % kernel.ns, bool(zlast), int(zc)


def precompile(device: torch.device, batch_size: int) -> None:
    """Compile the kernels for ``batch_size`` rows on ``device`` (main kernel about 15 s, plus the descriptor kernels).

    Must run outside CUDA-graph capture. ``run`` compiles on first use otherwise.
    """
    e = _engine(device, batch_size)
    _, mk, rd = _kernel_modules()
    with _lock:
        if e["fn"] is None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "dsv4_sparse_decode: compile the kernel (precompile) before CUDA-graph capture"
                )
            e["fn"] = mk.compile_mk(e["kernel"])
        kernel = e["kernel"]
        for r in range(batch_size):
            rot, zlast, zc = _desc_params(kernel, batch_size, r)
            rd.compile_desc(NUM_SMS, kernel.ns, rot, zlast, zc)


@dataclass
class Dsv4SparseDecodeState:
    """Per-step metadata and workspace of one caller; see ``prepare``.

    ``ws64`` holds ``WS_WORDS`` int64 words per row (generation counter, error words, selection directives), ``hist``,
    ``board`` and ``region`` the per-row selection scratch. ``gidx``, ``dbg``, ``stamps`` and ``ctl`` are kernel
    arguments used only by diagnostic builds.
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


def _host_ints(t: torch.Tensor, count: int, name: str) -> list[int]:
    v = t.reshape(-1)
    if v.numel() < count:
        raise ValueError(f"{name} has {v.numel()} entries, need {count}")
    return [int(x) for x in v[:count].tolist()]


def prepare(
    ctx_lens: torch.Tensor,
    kv_lens: torch.Tensor,
    idx_block_table: torch.Tensor,
    state: Dsv4SparseDecodeState | None = None,
) -> Dsv4SparseDecodeState:
    """Build (or refresh in place) the per-step state for ``B = ctx_lens.numel()`` rows.

    Args:
        ctx_lens: int32 [B], compressed (indexer) length of each row, in [512, 2**18].
        kv_lens: int32 [B], uncompressed length of each row (>= 128); the window is its last 128 positions.
        idx_block_table: int32 [B, pages] on the target device, indexer K pages of each row (32 compressed positions
            per page; whole 128-position tiles, so at least ``4 * ceil(n / 128)`` valid entries).
        state: a state of the same batch size to refresh; keeps its workspace (and generation counter).

    The lengths are checked on the host: pass host (CPU) tensors to avoid a device-to-host copy; device tensors work
    but synchronize.
    """
    batch_size = int(ctx_lens.numel())
    _check_batch_size(batch_size)
    device = idx_block_table.device
    if device.type != "cuda":
        raise ValueError("idx_block_table must be a CUDA tensor")
    ctx_host = _host_ints(ctx_lens, batch_size, "ctx_lens")
    kv_host = _host_ints(kv_lens, batch_size, "kv_lens")
    if any(not MIN_CTX_LEN <= n <= MAX_CTX_LEN for n in ctx_host):
        raise ValueError(
            f"compressed lengths must lie in [{MIN_CTX_LEN}, {MAX_CTX_LEN}], got {ctx_host}"
        )
    if any(n < WINDOW for n in kv_host):
        raise ValueError(f"uncompressed lengths must be at least {WINDOW}, got {kv_host}")
    need_pages = 4 * math.ceil(max(ctx_host) / 128)
    if idx_block_table.dim() != 2 or idx_block_table.shape[0] < batch_size:
        raise ValueError(f"idx_block_table must be [B, pages] with B >= {batch_size}")
    if idx_block_table.shape[1] < need_pages:
        raise ValueError(
            f"idx_block_table has {idx_block_table.shape[1]} pages per row, need {need_pages}"
        )
    if state is not None and state.batch_size != batch_size:
        raise ValueError(f"state is for {state.batch_size} rows, got {batch_size}")
    _, mk, rd = _kernel_modules()
    kernel = _engine(device, batch_size)["kernel"]
    rsz = kernel.row_sizes()
    if state is None:
        state = Dsv4SparseDecodeState(
            batch_size=batch_size,
            desc=torch.zeros(batch_size * rsz["desc"], dtype=torch.int32, device=device),
            swa_local=torch.empty((batch_size, WINDOW), dtype=torch.int32, device=device),
            swa_lens=torch.full((batch_size,), WINDOW, dtype=torch.int32, device=device),
            gidx=torch.full((batch_size, TOPK + WINDOW), -7, dtype=torch.int32, device=device),
            ws64=torch.zeros(batch_size * rsz["ws"], dtype=torch.int64, device=device),
            hist=torch.zeros(batch_size * rsz["hist"], dtype=torch.int32, device=device),
            board=torch.zeros(batch_size * rsz["board"], dtype=torch.int64, device=device),
            region=torch.zeros(batch_size * rsz["region"], dtype=torch.int64, device=device),
            dbg=torch.empty(MAX_CTX_LEN + 512, dtype=torch.float32, device=device),
            stamps=torch.zeros(
                NUM_SMS * mk.NSTAMP * (4 if batch_size > 1 else 1), dtype=torch.int64, device=device
            ),
            ctl=torch.zeros(batch_size * rsz["ctl"], dtype=torch.int32, device=device),
        )
    torch.add(
        torch.tensor(kv_host, dtype=torch.int32).to(device, non_blocking=True).unsqueeze(1),
        torch.arange(-WINDOW, 0, dtype=torch.int32, device=device),
        out=state.swa_local,
    )
    ctx = torch.tensor(ctx_host, dtype=torch.int32).to(device, non_blocking=True)
    bt = idx_block_table[:batch_size].to(dtype=torch.int32).contiguous()
    with _lock:  # the descriptor kernels compile on first use
        for r in range(batch_size):
            rot, zlast, zc = _desc_params(kernel, batch_size, r)
            rd.build_desc_device(
                ctx[r : r + 1],
                bt[r : r + 1],
                state.desc[r * rsz["desc"] : (r + 1) * rsz["desc"]],
                NUM_SMS,
                kernel.ns,
                rot=rot,
                zlast=zlast,
                zc=zc,
            )
    return state


def error_bits(state: Dsv4SparseDecodeState) -> torch.Tensor:
    """Per-row error bits (``ERR_*``) accumulated over the launches of ``state``: an int64 [B] device view, no sync.

    Zero means every launch so far completed normally. The bits are sticky; a fresh state starts at zero.
    """
    _, mk, _ = _kernel_modules()
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
    state: Dsv4SparseDecodeState,
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
    softmax_scale: float = DEFAULT_SOFTMAX_SCALE,
) -> None:
    """Launch the fused kernel on the current stream.

    Args:
        state: from ``prepare`` for these lengths and block tables.
        q_idx: float8_e4m3fn [B, 64, 128] (or [B, 1, 64, 128]), indexer queries.
        w_idx: float32 [B, 64], indexer head weights.
        k_cache: uint8 [pages, ...] with 4224 bytes per page, indexer K pages (32 x 128 FP8 values, then 32 float32
            scales).
        idx_block_table: int32 [B, pages], as given to ``prepare``.
        ctx_lens: int32 [B] on the device, the same lengths as given to ``prepare``.
        q_att: float8_e4m3fn [B, 64, 512], attention queries.
        swa_pool: float8_e4m3fn [..., 512], sliding-window KV rows (128 rows per page); bt_swa: int32 [B, pages].
        c_pool: float8_e4m3fn [..., 512], compressed KV rows (32 rows per page); bt_c: int32 [B, pages], each row
            16-byte aligned (row stride a multiple of 4 entries).
        sinks: float32 [64], per-head attention sink.
        topk: int32 [B, 512] output, the selected compressed positions (in no particular order).
        out: bfloat16 [B, 64 * 512] output.
        softmax_scale: attention softmax scale.
    """
    b = state.batch_size
    _check_numel("q_idx", q_idx, b * NUM_HEADS * INDEX_DIM)
    if q_idx.element_size() != 1:
        raise ValueError(f"q_idx must be FP8 (1 byte per value), got {q_idx.dtype}")
    _check_numel("w_idx", w_idx, b * NUM_HEADS, torch.float32)
    _check_numel("q_att", q_att, b * NUM_HEADS * LATENT_DIM)
    if q_att.element_size() != 1:
        raise ValueError(f"q_att must be FP8 (1 byte per value), got {q_att.dtype}")
    _check_numel("sinks", sinks, NUM_HEADS)
    _check_numel("topk", topk, b * TOPK, torch.int32)
    _check_numel("out", out, b * NUM_HEADS * LATENT_DIM, torch.bfloat16)
    _check_rows("swa_pool", swa_pool, LATENT_DIM)
    _check_rows("c_pool", c_pool, LATENT_DIM)
    pages = int(k_cache.shape[0])
    if k_cache.element_size() != 1 or k_cache.numel() != pages * INDEX_PAGE_BYTES:
        raise ValueError(
            f"k_cache must hold {INDEX_PAGE_BYTES} bytes per page, got {tuple(k_cache.shape)}"
        )
    if bt_c.dtype != torch.int32 or bt_c.stride(-1) != 1 or bt_c.data_ptr() % 16 != 0:
        raise ValueError("bt_c must be int32 with unit stride and a 16-byte aligned base")
    if b > 1 and bt_c.stride(0) % 4 != 0:
        raise ValueError(f"bt_c rows must be 16-byte aligned, got a row stride of {bt_c.stride(0)}")
    e = _engine(q_idx.device, b)
    if e["fn"] is None:
        precompile(q_idx.device, b)
    e["fn"](
        k_cache.view(pages, -1),
        q_idx.reshape(-1)
        .as_strided((NUM_HEADS, INDEX_DIM, b), (INDEX_DIM, 1, NUM_HEADS * INDEX_DIM))
        .view(torch.uint8),
        w_idx.reshape(-1).as_strided((NUM_HEADS, b), (1, NUM_HEADS)),
        idx_block_table[:b],
        ctx_lens.reshape(-1)[:b],
        state.desc,
        q_att.view(torch.uint8).view(b, NUM_HEADS, LATENT_DIM),
        swa_pool.view(torch.uint8).view(-1, LATENT_DIM),
        c_pool.view(torch.uint8).view(-1, LATENT_DIM),
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
    )
