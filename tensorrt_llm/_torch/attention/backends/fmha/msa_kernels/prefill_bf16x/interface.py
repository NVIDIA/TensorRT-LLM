# SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: MIT

"""FlashInfer-style wrapper for Q8KV4 paged sparse prefill (E4M3 or BF16 q)."""

from __future__ import annotations

import functools
import inspect

import logging
import math
import os
from dataclasses import dataclass

import torch


_SCALE_GROUPS = 8


def interleave_v_scales(v_scale: torch.Tensor) -> torch.Tensor:
    """Return linear ``[pages, Hkv, page_size, 8]`` V scales in the kernel's token-quad order.

    TensorRT-LLM: decode_q8kv4.interface.interleave_v_scales of the same upstream commit, inlined (decode_q8kv4 is not
    vendored). Same shape and dtype; byte ``token * 8 + group`` moves to ``(token // 4) * 32 + group * 4 + token % 4``.
    """
    if v_scale.ndim != 4 or v_scale.shape[-1] != _SCALE_GROUPS:
        raise ValueError(f"v_scale must have shape [pages, Hkv, page_size, {_SCALE_GROUPS}]")
    pages, heads, page_size, groups = v_scale.shape
    if page_size % 4:
        raise ValueError("page_size must be a multiple of 4")
    quads = v_scale.reshape(pages, heads, page_size // 4, 4, groups)
    return quads.transpose(-1, -2).contiguous().view(v_scale.shape)
from .jit import BF16_DATA_PATH, E4M3_DATA_PATH, MAX_BLOCK_SCALE_SHIFT, load_extension

logger = logging.getLogger(__name__)

# Set to 0 to keep the split workspace in interleaved device memory on a GPU with several memory
# locality domains, the layout single-domain GPUs use (for A/B measurements).
LOCALITY_DOMAINS_ENV = "FMHA_SM100_PREFILL_BF16X_LOCALITY_DOMAINS"

_HEAD_DIM = 128
_Q_HEADS_PER_KV = 16
_PAGE_SIZE = 128
# TopK list widths the k2q CSR builder and the combine accept.
SUPPORTED_TOPK = (4, 8, 16, 32)
_KV_DATA_ROW_BYTES = _HEAD_DIM // 2
_KV_SCALE_ROW_BYTES = _HEAD_DIM // 16


def _sparse_stack():
    """Return fmha_sm100's k2q CSR builder and split combine (the CuTe-DSL sparse stack)."""
    from ..fmha_sm100 import sparse  # noqa: F401  (puts the cute/ modules on sys.path)
    from src.sm100.fwd.combine import combine

    return sparse.build_k2q_csr, combine


@functools.lru_cache(maxsize=None)
def _combine_accepts(fn, name: str) -> bool:
    """Whether the installed combine takes keyword `name` (the shipped and upstream combines differ)."""
    return name in inspect.signature(fn).parameters


def _locality_hints_type():
    from ..fmha_sm100 import sparse  # noqa: F401  (puts the cute/ modules on sys.path)
    from src.sm100.fwd.combine import LocalityHints

    return LocalityHints


def _locality_domains_disabled() -> bool:
    value = os.environ.get(LOCALITY_DOMAINS_ENV, "").strip().lower() or "1"
    if value not in ("0", "1", "false", "true"):
        raise ValueError(f"{LOCALITY_DOMAINS_ENV} must be 0 or 1, got {value!r}")
    return value in ("0", "false")


@dataclass(frozen=True)
class _DeviceLocality:
    """The memory locality domains of one device: immutable host metadata, no device tensors."""

    num_domains: int
    page_bytes: int
    sm_domains: tuple[int, ...]


_device_locality: dict[int, _DeviceLocality | None] = {}


def _device_locality_domains(device: torch.device, block_scale_shift: int) -> _DeviceLocality | None:
    """The locality domains of ``device``, probed once per process; None when it has one."""
    if _locality_domains_disabled():
        return None
    index = device.index if device.index is not None else torch.cuda.current_device()
    if index not in _device_locality:
        extension = load_extension(device, block_scale_shift)
        locality = None
        num_domains = int(extension.locality_domain_count(index))
        if num_domains > 1:
            try:
                locality = _DeviceLocality(
                    num_domains=num_domains,
                    page_bytes=int(extension.locality_page_bytes(index)),
                    sm_domains=tuple(extension.sm_locality_domains(index).tolist()),
                )
            except RuntimeError as error:
                # The probe needs green contexts, which some profilers' driver shims refuse;
                # interleaved memory serves every domain then.
                logger.warning(
                    "BF16X prefill: locality-domain probe failed, the split workspace stays "
                    "interleaved: %s",
                    error,
                )
        if locality is not None:
            logger.info(
                "BF16X prefill: device %d has %d memory locality domains, the split workspace "
                "is striped over them",
                index,
                locality.num_domains,
            )
        _device_locality[index] = locality
    return _device_locality[index]


class PrefillWorkspace:
    """Split workspace of the BF16X prefill: the partial outputs and partial log-sum-exps the
    forward kernel writes and the combine reads.

    One workspace serves any number of layers and batches that run in order on one stream; two
    streams need two workspaces. On a GPU with several memory locality domains (Rubin, Blackwell)
    the partial pages alternate between the domains and the combine reads every tile from the
    domain of its own SM; elsewhere the partials are ordinary device memory. ``partial_bytes``
    reserves the partial buffer up front (:meth:`required_bytes` of the largest batch); a batch
    that needs more grows it, which on the striped path synchronizes the device first.
    """

    def __init__(
        self, device: torch.device | str | int, *, partial_bytes: int = 0, block_scale_shift: int = 0
    ) -> None:
        device = torch.device(device)
        if device.type != "cuda":
            raise ValueError(f"the prefill workspace lives on a CUDA device, got {device}")
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        self.device = device
        self._block_scale_shift = _check_block_scale_shift(block_scale_shift)
        self._locality = _device_locality_domains(device, self._block_scale_shift)
        self._partials: torch.Tensor | None = None
        self._lse: torch.Tensor | None = None
        self._hints = None
        if self._locality is not None:
            # The kernel's copy of the SM table and its claim counters, exclusive to this workspace.
            self._hints = _locality_hints_type()(
                sm_domains=torch.tensor(self._locality.sm_domains, dtype=torch.int32, device=device),
                tile_counters=torch.zeros(self._locality.num_domains, dtype=torch.int32, device=device),
                page_bytes=self._locality.page_bytes,
            )
        if partial_bytes > 0:
            self._reserve_partials(int(partial_bytes))

    @property
    def num_domains(self) -> int:
        """Memory locality domains the partial buffer is striped over (1: interleaved)."""
        return 1 if self._locality is None else self._locality.num_domains

    def required_bytes(self, topk: int, total_q: int, num_q_heads: int) -> int:
        """Partial-buffer bytes of a batch: ``topk`` slabs of ``total_q x num_q_heads`` BF16
        rows, each padded to the domain stripe."""
        return topk * self._slab_bytes(total_q, num_q_heads)

    def _slab_bytes(self, total_q: int, num_q_heads: int) -> int:
        slab = total_q * num_q_heads * _HEAD_DIM * 2
        if self._locality is None:
            return slab
        stripe = self._locality.num_domains * self._locality.page_bytes
        return -(-slab // stripe) * stripe

    def _reserve_partials(self, nbytes: int) -> None:
        if self._locality is None:
            self._partials = torch.empty(nbytes, dtype=torch.uint8, device=self.device)
            return
        stripe = self._locality.num_domains * self._locality.page_bytes
        nbytes = -(-nbytes // stripe) * stripe
        if self._partials is not None:
            # The buffer being replaced may still be read by the previous batch's kernels.
            torch.cuda.synchronize(self.device)
        extension = load_extension(self.device, self._block_scale_shift)
        try:
            self._partials = extension.alloc_locality_striped(nbytes, self.device.index)
        except RuntimeError as error:
            # A driver that counts domains but cannot place memory in them: interleaved memory
            # serves every domain, without the steering.
            logger.warning(
                "BF16X prefill: localized allocation failed, the split workspace stays "
                "interleaved: %s",
                error,
            )
            self._locality = None
            self._hints = None
            self._partials = torch.empty(nbytes, dtype=torch.uint8, device=self.device)

    def split_buffers(self, topk: int, total_q: int, num_q_heads: int):
        """``(o_partial, lse_partial, locality)`` for one batch: ``o_partial`` is
        ``[topk, total_q, num_q_heads, 128]`` BF16 with padded split slabs, ``lse_partial``
        ``[topk, total_q, num_q_heads]`` FP32, ``locality`` the combine's steering hints or
        None."""
        slab_bytes = self._slab_bytes(total_q, num_q_heads)
        partial_bytes = topk * slab_bytes
        if self._partials is None or self._partials.numel() < partial_bytes:
            self._reserve_partials(partial_bytes)
            # The reservation may have fallen back to interleaved memory, whose slabs are not padded.
            slab_bytes = self._slab_bytes(total_q, num_q_heads)
            partial_bytes = topk * slab_bytes
        o_partial = (
            self._partials[:partial_bytes]
            .view(torch.bfloat16)
            .as_strided(
                (topk, total_q, num_q_heads, _HEAD_DIM),
                (slab_bytes // 2, num_q_heads * _HEAD_DIM, _HEAD_DIM, 1),
            )
        )
        lse_elements = topk * total_q * num_q_heads
        if self._lse is None or self._lse.numel() < lse_elements:
            self._lse = torch.empty(lse_elements, dtype=torch.float32, device=self.device)
        lse_partial = self._lse[:lse_elements].view(topk, total_q, num_q_heads)
        return o_partial, lse_partial, self._hints


def _check_cuda_contiguous(tensor: torch.Tensor, *, name: str) -> None:
    if not tensor.is_cuda:
        raise ValueError(f"{name} must be a CUDA tensor")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")


def _check_same_device(
    reference: torch.Tensor,
    tensor: torch.Tensor,
    *,
    name: str,
) -> None:
    if tensor.device != reference.device:
        raise ValueError(f"{name} must be on {reference.device}")


def _check_cache_view(
    tensor: torch.Tensor, *, name: str, pages: int, num_kv_heads: int, row_bytes: int
) -> None:
    """Token rows contiguous inside each head; head and page strides are free (16-byte multiples)."""
    shape = (pages, num_kv_heads, _PAGE_SIZE, row_bytes)
    if tensor.ndim != 4 or tuple(tensor.shape) != shape:
        raise ValueError(f"{name} must have shape {list(shape)}")
    if tensor.stride(3) != 1 or tensor.stride(2) != row_bytes:
        raise ValueError(f"{name} must keep the 128 token rows of each head contiguous")
    if tensor.stride(1) % 16 or tensor.stride(0) % 16 or tensor.data_ptr() % 16:
        raise ValueError(f"{name} head/page strides and base must be 16-byte aligned")


def _check_global_scale(scale, *, name: str, device: torch.device) -> None:
    if scale is None:
        return
    if not isinstance(scale, torch.Tensor) or scale.numel() != 1 or scale.dtype != torch.float32:
        raise TypeError(f"{name} must be a one-element torch.float32 tensor")
    if scale.device != device:
        raise ValueError(f"{name} must be on {device}")


def _q_data_path(q: torch.Tensor) -> str:
    """The kernel data path selected by the dtype of q."""
    if q.dtype == torch.float8_e4m3fn:
        return E4M3_DATA_PATH
    if q.dtype == torch.bfloat16:
        return BF16_DATA_PATH
    raise TypeError("q must be torch.float8_e4m3fn or torch.bfloat16")


def _check_block_scale_shift(block_scale_shift: int) -> int:
    block_scale_shift = int(block_scale_shift)
    if not 0 <= block_scale_shift <= MAX_BLOCK_SCALE_SHIFT:
        raise ValueError(f"block_scale_shift must be in [0, {MAX_BLOCK_SCALE_SHIFT}]")
    return block_scale_shift


def _check_page_table(
    page_table: torch.Tensor, kv_indptr: torch.Tensor | None, *, batch: int
) -> None:
    """A [batch, max_pages] table (rows contiguous, any row stride) or a flat [total_pages] list
    with kv_indptr."""
    if page_table.dtype != torch.int32:
        raise TypeError("page_table must be torch.int32")
    if not page_table.is_cuda:
        raise ValueError("page_table must be a CUDA tensor")
    if page_table.ndim == 2:
        if kv_indptr is not None:
            raise ValueError("kv_indptr applies only to a flat [total_pages] page_table")
        if page_table.shape[0] != batch or page_table.shape[1] <= 0:
            raise ValueError("page_table must have shape [batch, max_pages]")
        if page_table.stride(1) != 1 or (batch > 1 and page_table.stride(0) < page_table.shape[1]):
            raise ValueError("page_table rows must be contiguous")
        return
    _check_cuda_contiguous(page_table, name="page_table")
    if page_table.ndim != 1 or page_table.numel() == 0:
        raise ValueError("page_table must be [batch, max_pages] or a flat [total_pages] list")
    if kv_indptr is None:
        raise ValueError("a flat [total_pages] page_table requires kv_indptr")
    _check_cuda_contiguous(kv_indptr, name="kv_indptr")
    _check_same_device(page_table, kv_indptr, name="kv_indptr")
    if kv_indptr.dtype != torch.int32 or tuple(kv_indptr.shape) != (batch + 1,):
        raise ValueError("kv_indptr must be torch.int32 with shape [batch + 1]")


def run_prefill(
    q: torch.Tensor,
    paged_kv_cache: tuple[torch.Tensor, torch.Tensor],
    kv_cache_sf: tuple[torch.Tensor, torch.Tensor],
    *,
    kv_indices: torch.Tensor,
    kv_indptr: torch.Tensor | None,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    k2q_row_ptr: torch.Tensor,
    schedule,
    sm_scale: float,
    topk: int,
    seqused_k: torch.Tensor | None = None,
    k_global_scale: torch.Tensor | None = None,
    v_global_scale: torch.Tensor | None = None,
    output_scale: float = 1.0,
    block_scale_shift: int = 0,
    out: torch.Tensor | None = None,
    lse: torch.Tensor | None = None,
    o_partial: torch.Tensor | None = None,
    lse_partial: torch.Tensor | None = None,
    workspace: PrefillWorkspace | None = None,
    o_fp8: torch.Tensor | None = None,
    sf_out: torch.Tensor | None = None,
    write_bf16: bool = True,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Run the Q8KV4 forward over a prepared k2q CSR and schedule, then combine the splits.

    ``paged_kv_cache`` holds the packed E2M1 K/V data views ``[pages, Hkv, 128, 64]`` and
    ``kv_cache_sf`` the E4M3 block-scale views ``[pages, Hkv, 128, 8]`` (K scales linear, V
    scales in token-quad order, see :func:`interleave_v_scales`); all four may be strided views
    into vLLM's page layout. Returns ``(out, lse)`` with ``lse`` None unless ``lse`` is given.

    ``q`` is ``torch.float8_e4m3fn`` (E4M3 data path: E4M3 dequantized K/V, E4M3 P) or
    ``torch.bfloat16`` (BF16 data path: BF16 dequantized K/V, BF16 P, a separate extension).

    ``kv_indices`` maps each request's logical pages to physical pages: a ``[batch, max_pages]``
    int32 table (rows contiguous, any row stride, e.g. vLLM's block table) with ``kv_indptr``
    None, or a flat ``[total_pages]`` list with ``kv_indptr`` (``[batch + 1]``) giving each
    request's first entry.

    Values are ``E2M1(code) x E4M3(block_scale) x global_scale``: ``k_global_scale`` and
    ``v_global_scale`` are optional one-element fp32 CUDA tensors read on the device, so a
    captured graph follows their updates. ``block_scale_shift`` stages every block scale by
    ``2**-shift`` before ``code x scale`` is requantized to E4M3 (3 for TransformerEngine-style
    full-range scales, 0 when the products already fit). ``output_scale`` multiplies the output.

    BF16 data path (bfloat16 ``q``): the forward (K1) folds ``k_global_scale`` into the K dequant
    (``bf16(block_scale) x bf16(k_global_scale)``, not the softmax scale) and applies neither
    ``v_global_scale`` nor ``output_scale``; the combine applies them (its ``output_scale`` tensor
    is ``v_global_scale``, times ``output_scale`` when that is not 1). ``block_scale_shift`` must
    be 0 (the BF16 dequant is exact, there is no staging).

    ``topk`` is the width of the TopK lists the schedule was built from (one split per slot).
    ``seqused_k`` (``[batch]`` int32) replaces each request's KV length for masking and for the
    bottom-right causal alignment, as ``fmha_sm100``'s ``qo_offset`` does.

    The split partials go to ``workspace`` (a :class:`PrefillWorkspace`, reused across calls and
    striped over the memory locality domains where the device has several), else to
    ``o_partial``/``lse_partial``, else to fresh allocations.
    """
    if len(paged_kv_cache) != 2 or len(kv_cache_sf) != 2:
        raise ValueError("paged_kv_cache and kv_cache_sf must be (K, V) pairs")
    k_cache, v_cache = paged_kv_cache
    k_scale, v_scale = kv_cache_sf
    if (
        schedule is None
        or schedule.scheduler_metadata is None
        or schedule.work_count is None
        or schedule.qsplit_indices is None
        or schedule.split_counts is None
    ):
        raise ValueError("run_prefill needs a forward schedule with split metadata")
    data_path = _q_data_path(q)
    _check_cuda_contiguous(q, name="q")
    _check_page_table(kv_indices, kv_indptr, batch=cu_seqlens_q.numel() - 1)
    _check_global_scale(k_global_scale, name="k_global_scale", device=q.device)
    _check_global_scale(v_global_scale, name="v_global_scale", device=q.device)
    block_scale_shift = _check_block_scale_shift(block_scale_shift)
    if data_path == BF16_DATA_PATH and block_scale_shift != 0:
        raise ValueError("bfloat16 q (BF16 data path) requires block_scale_shift 0")
    if topk not in SUPPORTED_TOPK:
        raise ValueError(f"topk must be one of {SUPPORTED_TOPK}, got {topk}")
    total_q, num_q_heads, _ = q.shape
    num_kv_heads = int(k_cache.shape[1]) if k_cache.ndim == 4 else -1
    pages = int(k_cache.shape[0])
    for name, tensor, row_bytes in (
        ("k_cache", k_cache, _KV_DATA_ROW_BYTES),
        ("v_cache", v_cache, _KV_DATA_ROW_BYTES),
        ("k_scale", k_scale, _KV_SCALE_ROW_BYTES),
        ("v_scale", v_scale, _KV_SCALE_ROW_BYTES),
    ):
        _check_same_device(q, tensor, name=name)
        _check_cache_view(
            tensor, name=name, pages=pages, num_kv_heads=num_kv_heads, row_bytes=row_bytes
        )
    if k_cache.dtype != torch.uint8 or v_cache.dtype != torch.uint8:
        raise TypeError("k_cache and v_cache must be torch.uint8")
    if num_q_heads != num_kv_heads * _Q_HEADS_PER_KV:
        raise ValueError(f"q must have {num_kv_heads * _Q_HEADS_PER_KV} heads (16 per KV head)")

    options = dict(device=q.device)
    locality = None
    if workspace is not None:
        if o_partial is not None or lse_partial is not None:
            raise ValueError("pass either workspace or o_partial/lse_partial, not both")
        if workspace.device != q.device:
            raise ValueError(f"workspace must be on {q.device}, got {workspace.device}")
        o_partial, lse_partial, locality = workspace.split_buffers(topk, total_q, num_q_heads)
    if o_partial is None:
        o_partial = torch.empty(
            (topk, total_q, num_q_heads, _HEAD_DIM), dtype=torch.bfloat16, **options
        )
    if lse_partial is None:
        lse_partial = torch.empty((topk, total_q, num_q_heads), dtype=torch.float32, **options)
    if out is None:
        out = torch.empty((total_q, num_q_heads, _HEAD_DIM), dtype=torch.bfloat16, **options)

    _, combine = _sparse_stack()
    # BF16 data path: K1 gets k_global_scale (folded into the K dequant) but not v_global_scale or
    # output_scale; the combine below applies those as its output_scale.
    combine_output_scale = None
    if data_path == BF16_DATA_PATH:
        combine_output_scale = v_global_scale
        if float(output_scale) != 1.0:
            combine_output_scale = (
                v_global_scale * float(output_scale)
                if v_global_scale is not None
                else torch.full((1,), float(output_scale), dtype=torch.float32, device=q.device)
            )
        if combine_output_scale is not None and not _combine_accepts(combine, "output_scale"):
            raise RuntimeError(
                "the BF16 data path needs a combine that takes output_scale to apply "
                "v_global_scale / output_scale"
            )
        forward_v_global_scale = None
        forward_output_scale = 1.0
    else:
        forward_v_global_scale = v_global_scale
        forward_output_scale = float(output_scale)
    load_extension(q.device, block_scale_shift, data_path).run(
        q,
        k_cache,
        v_cache,
        k_scale,
        v_scale,
        kv_indices,
        kv_indptr,
        cu_seqlens_q,
        cu_seqlens_k,
        seqused_k,
        k2q_row_ptr,
        schedule.qsplit_indices,
        schedule.scheduler_metadata,
        schedule.work_count,
        o_partial,
        lse_partial,
        k_global_scale,
        forward_v_global_scale,
        float(sm_scale),
        forward_output_scale,
        None if locality is None else locality.tile_counters,
    )
    # M3.1 overlay v2 (TP2CTX_Q8KV4): the shipped fmha_sm100 combine predates the locality hints and needs cu_seqlens for
    # a varlen output with split_counts. E4M3: the V global scale is already applied in this forward, so no
    # output_scale. BF16: the combine applies v_global_scale (x output_scale) as its output_scale.
    combine_kwargs = dict(split_counts=schedule.split_counts, use_pdl=True)
    if _combine_accepts(combine, "cu_seqlens"):
        combine_kwargs["cu_seqlens"] = cu_seqlens_q
    if locality is not None:
        combine_kwargs["locality"] = locality
    if combine_output_scale is not None:
        combine_kwargs["output_scale"] = combine_output_scale
    # o_proj MXFP8 sink (LongAttn, TP2CTX_COMBINE_MXFP8 on the E4M3X fork): the combine also writes the MXFP8 copy of
    # the final O (o_fp8 [total_q, heads x dim] + swizzled UE8M0 sf_out; bitwise mxfp8_quantize(O, True)), and with
    # write_bf16=False skips the bf16 O store. No silent fallback: a combine without these arguments raises.
    if o_fp8 is not None or sf_out is not None or not write_bf16:
        if o_fp8 is None or sf_out is None:
            raise ValueError("run_prefill: o_fp8 and sf_out go together")
        if not _combine_accepts(combine, "o_fp8"):
            raise RuntimeError("run_prefill: the installed fmha combine has no o_fp8 / sf_out (MXFP8 sink) arguments")
        combine_kwargs["o_fp8"] = o_fp8
        combine_kwargs["sf_out"] = sf_out
        if not write_bf16:
            combine_kwargs["write_bf16"] = False
    combine(o_partial, lse_partial, out, lse, **combine_kwargs)
    return out, lse


@dataclass(frozen=True)
class _PlanState:
    cu_seqlens_q: torch.Tensor
    cu_seqlens_k: torch.Tensor
    kv_indices: torch.Tensor
    kv_indptr: torch.Tensor | None
    seqused_k: torch.Tensor | None
    topk: int
    k2q_row_ptr: torch.Tensor
    schedule: object
    sm_scale: float
    block_scale_shift: int
    out: torch.Tensor
    lse: torch.Tensor


class BatchPrefillWithPagedKVCacheWrapper:
    """Stateful Q8KV4 sparse-prefill wrapper with reusable scheduling state.

    ``workspace`` holds the split partials between ``plan`` and ``run``; wrappers that run in
    order on one stream may share one, and a wrapper without one creates its own.
    """

    def __init__(self, workspace: PrefillWorkspace | None = None) -> None:
        self._plan_state: _PlanState | None = None
        self._workspace = workspace

    def plan(
        self,
        topk_indices: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        page_table: torch.Tensor,
        *,
        total_k: int,
        total_rows: int,
        max_seqlen_q: int,
        max_seqlen_k: int,
        kv_indptr: torch.Tensor | None = None,
        causal: bool = True,
        sm_scale: float | None = None,
        block_scale_shift: int = 0,
        seqused_k: torch.Tensor | None = None,
    ) -> None:
        """Prepare reusable q2k-to-k2q metadata and reserve the split workspace.

        ``page_table`` maps each request's logical pages to physical pages, either as a
        ``[batch, max_pages]`` table (rows contiguous, any row stride) or as a flat
        ``[total_pages]`` list with ``kv_indptr`` (``[batch + 1]``) giving each request's first
        entry. ``topk_indices`` is
        ``[num_kv_heads, total_q, topk]`` with ``topk`` in ``SUPPORTED_TOPK``. See
        :func:`run_prefill` for ``block_scale_shift`` and ``seqused_k``.
        """

        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("plan() must be called outside CUDA Graph capture")
        if not causal:
            raise NotImplementedError("BF16X prefill v1 supports only causal attention")
        _check_same_device(topk_indices, page_table, name="page_table")
        batch = cu_seqlens_q.numel() - 1
        _check_page_table(page_table, kv_indptr, batch=batch)
        if topk_indices.dtype != torch.int32:
            raise TypeError("topk_indices must be torch.int32")
        if (
            topk_indices.ndim != 3
            or topk_indices.shape[0] <= 0
            or topk_indices.shape[2] not in SUPPORTED_TOPK
        ):
            raise ValueError(
                f"topk_indices must have shape [num_kv_heads, total_q, topk], topk in {SUPPORTED_TOPK}"
            )
        _check_cuda_contiguous(topk_indices, name="topk_indices")

        build_k2q_csr, _ = _sparse_stack()
        k2q_row_ptr, _, schedule = build_k2q_csr(
            topk_indices,
            cu_seqlens_q,
            cu_seqlens_k,
            _PAGE_SIZE,
            total_k=int(total_k),
            max_seqlen_k=int(max_seqlen_k),
            max_seqlen_q=int(max_seqlen_q),
            total_rows=int(total_rows),
            qhead_per_kv=_Q_HEADS_PER_KV,
            return_schedule=True,
        )
        total_q = int(topk_indices.shape[1])
        topk = int(topk_indices.shape[2])
        num_q_heads = int(topk_indices.shape[0]) * _Q_HEADS_PER_KV
        if seqused_k is not None:
            _check_cuda_contiguous(seqused_k, name="seqused_k")
            if seqused_k.dtype != torch.int32 or tuple(seqused_k.shape) != (batch,):
                raise ValueError("seqused_k must be torch.int32 with shape [batch]")
        options = dict(device=topk_indices.device)
        scale = 1.0 / math.sqrt(_HEAD_DIM) if sm_scale is None else float(sm_scale)
        if not math.isfinite(scale) or scale <= 0.0:
            raise ValueError("sm_scale must be finite and positive")
        block_scale_shift = _check_block_scale_shift(block_scale_shift)
        device = topk_indices.device
        if self._workspace is None or self._workspace.device != device:
            self._workspace = PrefillWorkspace(device, block_scale_shift=block_scale_shift)
        # Reserve now so run() never grows the buffer, not even under graph capture.
        self._workspace.split_buffers(topk, total_q, num_q_heads)
        self._plan_state = _PlanState(
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            kv_indices=page_table,
            kv_indptr=kv_indptr,
            seqused_k=seqused_k,
            topk=topk,
            k2q_row_ptr=k2q_row_ptr,
            schedule=schedule,
            sm_scale=scale,
            block_scale_shift=block_scale_shift,
            out=torch.empty(
                (total_q, num_q_heads, _HEAD_DIM),
                dtype=torch.bfloat16,
                **options,
            ),
            lse=torch.empty(
                (total_q, num_q_heads),
                dtype=torch.float32,
                **options,
            ),
        )

    def run(
        self,
        q: torch.Tensor,
        paged_kv_cache: tuple[torch.Tensor, torch.Tensor],
        *,
        kv_cache_sf: tuple[torch.Tensor, torch.Tensor],
        kv_global_scale: tuple[torch.Tensor | None, torch.Tensor | None] | None = None,
        out: torch.Tensor | None = None,
        lse: torch.Tensor | None = None,
        return_lse: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Run one layer with the metadata and workspace cached by ``plan``.

        K/V data and scales may be strided views (see :func:`run_prefill`); V block scales are
        read in token-quad order. ``kv_global_scale`` is an optional ``(K, V)`` pair of
        one-element fp32 CUDA tensors.
        """

        state = self._plan_state
        if state is None:
            raise RuntimeError("plan() must be called before run()")
        if q.shape != state.out.shape:
            raise ValueError(f"q must have shape {tuple(state.out.shape)}")

        out_tensor = state.out if out is None else out
        lse_tensor = state.lse if lse is None else lse
        for name, tensor, dtype, shape in (
            ("out", out_tensor, torch.bfloat16, state.out.shape),
            ("lse", lse_tensor, torch.float32, state.lse.shape),
        ):
            _check_cuda_contiguous(tensor, name=name)
            _check_same_device(q, tensor, name=name)
            if tensor.dtype != dtype or tensor.shape != shape:
                raise ValueError(
                    f"{name} must have dtype {dtype} and shape {tuple(shape)}"
                )

        run_prefill(
            q,
            paged_kv_cache,
            kv_cache_sf,
            kv_indices=state.kv_indices,
            kv_indptr=state.kv_indptr,
            cu_seqlens_q=state.cu_seqlens_q,
            cu_seqlens_k=state.cu_seqlens_k,
            k2q_row_ptr=state.k2q_row_ptr,
            schedule=state.schedule,
            sm_scale=state.sm_scale,
            topk=state.topk,
            seqused_k=state.seqused_k,
            k_global_scale=None if kv_global_scale is None else kv_global_scale[0],
            v_global_scale=None if kv_global_scale is None else kv_global_scale[1],
            block_scale_shift=state.block_scale_shift,
            out=out_tensor,
            lse=lse_tensor if return_lse or lse is not None else None,
            workspace=self._workspace,
        )
        if return_lse:
            return out_tensor, lse_tensor
        return out_tensor


__all__ = [
    "LOCALITY_DOMAINS_ENV",
    "SUPPORTED_TOPK",
    "BatchPrefillWithPagedKVCacheWrapper",
    "PrefillWorkspace",
    "interleave_v_scales",
    "run_prefill",
]
