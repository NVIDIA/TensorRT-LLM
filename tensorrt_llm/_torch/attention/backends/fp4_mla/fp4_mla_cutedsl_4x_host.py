"""Torch host entry for the four-CTA-cluster (4x) FP4 MLA decode kernel (``fp4_mla_cutedsl_4x.py``).

Scope: validate the envelope, build the 30 wrapper arguments of
``fp4_mla_4x_full_ctm`` from the mufu16 input contract, compile + cache per (query_len_per_seq, output dtype,
use_ksf_gather4), launch MAIN only.  There is no reset kernel, no workspace contract and no generation gate
in the FULL-only variant, so a call is a pure function of its inputs.

Input contract = ``run_trtllm_fp4_mla_decode_page_native_from_raw`` of the two-CTA kernel
(``fp4_mla_cutedsl_mufu16.py``): the same 11 positional tensors in the same order, so a saved decode case replays
as ``run_fp4_mla_decode_4x_full(*fixture["args"], **fixture["kwargs"])``. ``X79`` in the comments is the
internal version tag of the kernel this host was derived from.

Importing this module does NOT import cutlass/torch-cuda code paths: the kernel module is loaded lazily, so
``validate_envelope`` is usable on a CPU-only host (unit tests) given torch.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import math
import os
import sys
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

__all__ = [
    "EnvelopeGeometry",
    "FULL_ONLY_MAX_PAIRS",
    "SUPPORTED_QUERY_LENS",
    "COMPILE_OPTIONS",
    "KERNEL_ARGV_FLAGS",
    "check_kernel_argv",
    "kernel_module",
    "validate_envelope",
    "zero_queries",
    "compile_variant",
    "run_fp4_mla_decode_4x_full",
    "compiled_variants",
    "vpfl_decision",
    "envelope_reason_static",
    "use_split_mode",
    "split_enabled",
    "pair_tasks",
    "MIN_SM_COUNT",
    "SPLIT_MAX_PAIRS",
]

# --- envelope constants; cross-checked against the kernel module when it loads -------------------------------
FULL_ONLY_MAX_PAIRS = (
    46  # one wave: grid z = n_pairs = B*q/2 <= 46 (BRIEF; X79:113 ORDERED_SIX_WORKERS = 46)
)
SUPPORTED_QUERY_LENS = (
    2,
    4,
    6,
    8,
)  # even q only: a pair (2p, 2p+1) never straddles a sequence (X79:8736-8741)
HEADS = 128  # SMEM_P4_CTA_GROUP_M = 64*2 (X79:430; MUFU:8851-8854)
OUT_COLS = 512  # TRTLLM_V_HEAD_DIM (X79:369)
Q_BYTES = 320  # QK_LOGICAL_DIM // 2 (X79:366, 10223)
K_STORAGE_BYTES = 320  # TRTLLM_K_STORAGE_DIM // 2 = (576+64)/2 (X79:368-372, 10289-10291)
Q_SF_GROUPS = 40  # QK_SF_GROUPS = 640 // 16 (X79:393)
K_SF_GROUPS = 40  # TRTLLM_K_SF_GROUPS = 640 // 16 (X79:375)
PAGE_SIZE = (
    128  # TRTLLM_PAGE_SIZE (X79:374); wrapper Constexpr default is KV_TILE=256 (X79:1190, 360)
)
KV_TILE = 256  # X79:360
PHYSICAL_K = (
    160 * 1024
)  # SMEM_P4_RUNTIME_MAX_KV, the fixed runtime-KV profile (X79:400, 10036-10037, 10200)
MAX_PAGES_PER_SEQ = (
    PHYSICAL_K // PAGE_SIZE
)  # SMEM_P4_RUNTIME_MAX_PAGES = 1280 (X79:401, 1223, 10300)
V_PACK_BLOCK = (
    128  # SMEM_P4_V_N_PER_CTA = 256 // 2 (X79:690); the wrapper's V views assume it (X79:1247)
)
KSF_PAYLOAD_BYTES = PAGE_SIZE * K_SF_GROUPS  # 5120 (X79:10345)
VSF_PAYLOAD_BYTES = OUT_COLS * (PAGE_SIZE // 16)  # 4096 (X79:10356)
CUDA_GRID_Z_MAX = 65535  # X79:37
V_PAGE_BYTES = OUT_COLS * (
    PAGE_SIZE // 2
)  # 32768: one physical page of the V pack (512 rows x 64 B)
# vpfl (warp-11 LSU L2 prefetch of V/VSF): fine while the V pool spans <= ~1 GB, stalls the V issuer 70x at a 1.7 GB
# span (REPORT.md section 5; measured fine at 0.8 GB, cliff at >= 1.25 GB).  The host disables it above this span;
# MLA4X_VPFL_MAX_V_SPAN_MB overrides the threshold (0 = always off), ``use_vpfl=`` forces it per call.
VPFL_MAX_V_SPAN_ENV = "TRTLLM_FP4_MLA_4X_VPFL_MAX_V_SPAN_MB"
VPFL_MAX_V_SPAN_BYTES = int(float(os.environ.get(VPFL_MAX_V_SPAN_ENV, "1024")) * 1024 * 1024)
MIN_SM_COUNT = 4 * FULL_ONLY_MAX_PAIRS  # 184: one wave of 46 four-CTA clusters must be co-resident
# --- split mode (Milestone 2): plan-driven KV split tasks + FP32 partials merged by fp4_mla_packed_reduce ----------
SPLIT_ENV = "TRTLLM_FP4_MLA_4X_SPLIT"  # default on; 0 keeps the one-wave FULL-only kernel (and its envelope) only
SPLIT_MAX_PAIRS = 512  # partial workspace bound (<= ~0.5 GB of FP32 partials at 128 heads x q4)
# Below these KV lengths the per-task fixed cost (~9.5 us) beats the fill: 2x stays. Measured (Rubin, real
# fixtures): B32/Q4 breaks even at ~9K tokens (-2.5 % at 8,704); an under-filled single wave (33 pairs) gains from a
# split only at long KV (-22 % at 135K, nothing at 30K).
SPLIT_MIN_KV_MULTIWAVE_ENV = "TRTLLM_FP4_MLA_4X_SPLIT_MIN_KV_MULTIWAVE"
SPLIT_MIN_KV_MULTIWAVE_DEFAULT = 8192
SPLIT_MIN_KV_ONEWAVE_ENV = "TRTLLM_FP4_MLA_4X_SPLIT_MIN_KV_ONEWAVE"
SPLIT_MIN_KV_ONEWAVE_DEFAULT = 65536
# Under-filled single waves: the 2x packed planner cuts every request in two whenever the two-piece plan still fits
# one wave of its 2-CTA clusters (2 x requests x m_tiles <= sm_count // 2, m_tiles = heads x q / 128 packed rows;
# fp4_mla_packed_reduce.packed_plan_kernel), which halves its makespan while a one-wave 4x launch cannot. Measured
# at KV 30K through the backend dispatch (Rubin): B12/Q4 +28 %, B13/Q4 +29 %, B5/Q8 +38 % on 4x; one request
# more and the 2x plan keeps requests whole: B14/Q4 -4 %, B16 -5 %, B20 -8.5 %, B23 -17 %, B7/Q8 -4 %, B9/Q6 -4 %.
# q2 is the exception: there the 2x kernel is DRAM-latency-bound (two tokens of KV reuse per tile) and its
# two-piece plan gains little past ~20 requests (B16 +15 %, B20 -1 %, B24 -16 %, B26 -21 %, B28 -16 % on 4x), so
# the 4x kernel takes over from Q2_MIN_PAIRS pairs (= requests at 128 heads). The env knob replaces the derived
# rule by a flat floor on pair tasks (0 = no floor; the GPU unit tests use it on their small batches).
MIN_PAIRS_ENV = "TRTLLM_FP4_MLA_4X_MIN_PAIRS"
MIN_PAIRS_DERIVED = -1  # default: the 2x two-piece-plan rule above
Q2_MIN_PAIRS = 20


def _env_flag(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return default
    return value.strip().lower() not in ("0", "false", "off", "no")


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return default
    return int(value)


def split_enabled() -> bool:
    """Whether the plan-driven split mode may be used (``TRTLLM_FP4_MLA_4X_SPLIT``, default on)."""
    return _env_flag(SPLIT_ENV, True)


def pair_tasks(num_heads: int, query_len: int, num_requests: int) -> int:
    """256-row pair tasks of one call (rows mode: heads * query_len % 256 == 0)."""
    return num_requests * num_heads * query_len // 256


def use_split_mode(*, num_heads: int, query_len: int, num_requests: int, max_kv_len: int) -> bool:
    """Static choice between the FULL-only launch and the split mode for a call inside the envelope: split when
    the batch exceeds one wave, or when an under-filled wave runs at long KV (the plan may still keep every
    request whole; the device planner decides per step)."""
    if not split_enabled():
        return False
    pairs = pair_tasks(num_heads, query_len, num_requests)
    if pairs > FULL_ONLY_MAX_PAIRS:
        return True
    return pairs < FULL_ONLY_MAX_PAIRS and max_kv_len >= _env_int(
        SPLIT_MIN_KV_ONEWAVE_ENV, SPLIT_MIN_KV_ONEWAVE_DEFAULT
    )


INT32_MAX = 2147483647
LOG2_E = 1.4426950408889634  # X79:395
COMPILE_OPTIONS = "--opt-level 2 --ptxas-options '--uumn'"  # X79:10093
# The kernel module reads sys.argv at import (X79:318-356 via :380 SF_VEC_SIZE and :406 lazy-anchor threshold):
# a driver whose argv happens to carry these flags would silently reconfigure the kernel (hazard H-E2).
KERNEL_ARGV_FLAGS = ("--sf-vec-size", "--lazy-anchor-threshold-log2")

_HERE = Path(__file__).resolve().parent
KERNEL_FILE = _HERE / "fp4_mla_cutedsl_4x.py"
KERNEL_MODULE_NAME = "fp4_mla_cutedsl_4x"
# Names the INTERFACE CONTRACT requires the kernel module to export.
_REQUIRED_KERNEL_NAMES = (
    "fp4_mla_4x_full_ctm",
    "HEADER_SCRATCH_WORDS",
    "FULL_ONLY_MAX_PAIRS",
    "SUPPORTED_QUERY_LENS",
    "_compile_cutedsl",
    "QK_PADDED_DIM",
    "TRTLLM_K_STORAGE_DIM",
    "SMEM_P4_V_N_PER_CTA",
    "OUT_DIM",
    "KV_TILE",
    "QK_SF_GROUPS",
    "TRTLLM_K_SF_GROUPS",
    "SF_VEC_SIZE",
    "TRTLLM_V_HEAD_DIM",
    "SMEM_P4_RUNTIME_MAX_KV",
    "LOG2_E",
)


class EnvelopeError(ValueError):
    """Raised when a call is outside the fair envelope or violates the mufu16 input contract."""


def check_kernel_argv(argv: list[str] | None = None) -> None:
    """Reject argv that the kernel module would consume at import time (X79:318-356, 380, 406)."""
    args = sys.argv[1:] if argv is None else list(argv)
    for arg in args:
        for flag in KERNEL_ARGV_FLAGS:
            if arg == flag or arg.startswith(flag + "="):
                raise EnvelopeError(
                    f"sys.argv carries {flag!r}, which the kernel module reads at import time "
                    f"(X79:318-356); the 4x host refuses to run with kernel-tuning flags in argv (H-E2)"
                )


_KERNEL_LOCK = threading.Lock()
_KERNEL: Any = None


def kernel_module():
    """Load ``fp4_mla_cutedsl_4x.py`` from this directory (once) and verify the INTERFACE CONTRACT exports."""
    global _KERNEL
    with _KERNEL_LOCK:
        if _KERNEL is not None:
            return _KERNEL
        check_kernel_argv()
        module = sys.modules.get(KERNEL_MODULE_NAME)
        if module is None:
            if not KERNEL_FILE.is_file():
                raise ImportError(f"kernel module not found: {KERNEL_FILE}")
            spec = importlib.util.spec_from_file_location(KERNEL_MODULE_NAME, KERNEL_FILE)
            module = importlib.util.module_from_spec(spec)
            sys.modules[KERNEL_MODULE_NAME] = module
            try:
                spec.loader.exec_module(module)
            except BaseException:
                sys.modules.pop(KERNEL_MODULE_NAME, None)
                raise
        missing = [name for name in _REQUIRED_KERNEL_NAMES if not hasattr(module, name)]
        if missing:
            raise ImportError(
                f"{KERNEL_FILE} does not export the INTERFACE CONTRACT names: {missing}"
            )
        if int(module.FULL_ONLY_MAX_PAIRS) != FULL_ONLY_MAX_PAIRS:
            raise ImportError(
                f"kernel FULL_ONLY_MAX_PAIRS={module.FULL_ONLY_MAX_PAIRS} != host {FULL_ONLY_MAX_PAIRS}"
            )
        if tuple(module.SUPPORTED_QUERY_LENS) != SUPPORTED_QUERY_LENS:
            raise ImportError(
                f"kernel SUPPORTED_QUERY_LENS={module.SUPPORTED_QUERY_LENS} != host {SUPPORTED_QUERY_LENS}"
            )
        if int(module.HEADER_SCRATCH_WORDS) < 1:
            raise ImportError("kernel HEADER_SCRATCH_WORDS must be >= 1")
        checks = {
            "QK_PADDED_DIM": 2 * Q_BYTES,
            "TRTLLM_K_STORAGE_DIM": 2 * K_STORAGE_BYTES,
            "SMEM_P4_V_N_PER_CTA": V_PACK_BLOCK,
            "QK_SF_GROUPS": Q_SF_GROUPS,
            "TRTLLM_K_SF_GROUPS": K_SF_GROUPS,
            "SF_VEC_SIZE": 16,
            "TRTLLM_V_HEAD_DIM": OUT_COLS,
            "SMEM_P4_RUNTIME_MAX_KV": PHYSICAL_K,
            "KV_TILE": KV_TILE,
        }
        bad = {
            k: (getattr(module, k), v) for k, v in checks.items() if int(getattr(module, k)) != v
        }
        if bad:
            raise ImportError(f"kernel constants differ from the host's envelope constants: {bad}")
        _KERNEL = module
        return module


# --- geometry -----------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class EnvelopeGeometry:
    """Everything the launch needs that was derived from the tensors (host-side mirror of X79:10200-10422)."""

    num_queries: int  # L = B*q = runtime_l (X79:10206, 10517)
    num_sequences: int  # B = L // q (X79:10222)
    query_len_per_seq: int
    num_pairs: int  # L // 2 = grid z of the FULL-only wrapper (BRIEF)
    physical_m: int  # 128 = runtime_m (X79:10206, 10516)
    physical_k: int  # 163840 (X79:10200)
    q_batch_capacity: int  # q_tma_l (X79:10209-10219, 10518)
    num_cache_pages: int  # kv_cache pages (X79:10299, 10519)
    num_v_cache_pages: int  # v_packed.shape[0] // 512 (X79:10325, 10520)
    v_page_offset: int
    kv_page_stride_bytes: int  # X79:10419
    ksf_page_stride_bytes: int  # X79:10420
    vsf_page_stride_bytes: int  # X79:10421
    softmax_scale_log2: float  # sm_scale * LOG2_E (X79:10422)
    use_ksf_gather4: bool  # X79:10038-10040
    use_consecutive_page_pair: bool
    output_dtype: torch.dtype
    max_kv_len: int
    device: torch.device = field(default_factory=lambda: torch.device("cpu"))
    vpfl_enabled: bool = (
        True  # runtime switch of the V L2 prefetch (vpfl_decision); not part of compile_key
    )
    num_heads: int = HEADS  # rows mode: local heads (heads * q % 256 == 0); Constexpr of the kernel
    num_blocks: int = 0  # 128-row blocks = tokens * heads / 128 = runtime_l (grid z = blocks // 2)
    q_batch_blocks: int = 0  # q_tma_l in blocks
    split_mode: bool = (
        False  # Milestone 2: plan-driven KV split tasks + FP32 partials (Constexpr of the kernel)
    )
    task_capacity: int = 0  # split mode: plan slots (requests * splits) = plan table stride
    used_tasks: int = 0  # split mode: planned tasks launched; grid z = used_tasks * pairs_per_seq

    @property
    def compile_key(self) -> tuple:
        # X79:10041-10049 cache key: (n, page_size, use_mixed_imlp, output_dtype, query_len_per_seq,
        # use_consecutive_page_pair, use_ksf_gather4); use_mixed_imlp is appended by the caller.
        return (
            OUT_COLS,
            PAGE_SIZE,
            self.output_dtype,
            self.query_len_per_seq,
            self.use_consecutive_page_pair,
            self.use_ksf_gather4,
            self.num_heads,
            self.split_mode,
        )


def _storage_bytes_from_offset(t: torch.Tensor) -> int:
    # X79:10237-10240 / 10347-10350
    return t.untyped_storage().nbytes() - t.storage_offset() * t.element_size()


def _check_alignment(name: str, t: torch.Tensor, alignment: int) -> None:
    # X79:1121-1130 _validate_tensor_pointer_alignment; the alignment is what _make_fused_ptrs promises via
    # make_ptr(..., assumed_align=...) at X79:9814-9846 (16 for the FP4 pointers, 32 for scale/output/scratch).
    if t.data_ptr() % alignment != 0:
        raise EnvelopeError(
            f"{name} data pointer must be {alignment}B aligned, got remainder {t.data_ptr() % alignment}"
        )


def _kv_cache_layout(kv_cache: torch.Tensor, page_size: int) -> tuple[int, int, int, int, int]:
    """(num_pages, packed_dim, stride_page, stride_token, stride_packed_dim) -- X79:9849-9878 _kv_cache_3d_layout."""
    if kv_cache.dtype != torch.uint8:
        raise EnvelopeError(f"kv_cache must be torch.uint8, got {kv_cache.dtype}")
    if kv_cache.dim() == 3:
        num_pages, actual_page_size, packed_dim = kv_cache.shape
        stride_page, stride_token, stride_packed = kv_cache.stride()
    elif (
        kv_cache.dim() >= 5
    ):  # TRT-LLM 5D paged layout [P, 1, 128, 1, 320], axes 0/2/4 (X79:9857-9863)
        num_pages, actual_page_size, packed_dim = (
            kv_cache.shape[0],
            kv_cache.shape[2],
            kv_cache.shape[4],
        )
        stride_page, stride_token, stride_packed = (
            kv_cache.stride(0),
            kv_cache.stride(2),
            kv_cache.stride(4),
        )
    else:
        raise EnvelopeError(
            "kv_cache must be [num_pages, page_size, packed_dim] or TRT-LLM's 5D paged layout with page/token/"
            f"packed-dim axes at 0/2/4, got shape={tuple(kv_cache.shape)}"
        )
    if actual_page_size != page_size:
        raise EnvelopeError(f"page_size mismatch: argument={page_size}, cache={actual_page_size}")
    return int(num_pages), int(packed_dim), int(stride_page), int(stride_token), int(stride_packed)


def validate_envelope(
    q_fp4: torch.Tensor,
    q_sf: torch.Tensor,
    kv_cache: torch.Tensor,
    sf_cache: torch.Tensor,
    v_packed: torch.Tensor,
    v_sf: torch.Tensor,
    global_scale: torch.Tensor,
    src_page_ids: torch.Tensor,
    paged_kv_indptr_decode: torch.Tensor,
    kv_lens: torch.Tensor,
    output: torch.Tensor,
    *,
    max_kv_len: int,
    sm_scale: float,
    num_heads: int,
    query_len_per_seq: int,
    q_global_scale: torch.Tensor | None = None,
    page_size: int = PAGE_SIZE,
    v_pack_block: int = V_PACK_BLOCK,
    v_page_offset: int = 0,
    q_batch_capacity: int | None = None,
    assume_consecutive_page_prefix_tiles: int = 0,
    max_pairs: int | None = None,
    require_cuda: bool = True,
    check_device_contents: bool = False,
) -> EnvelopeGeometry:
    """Raise EnvelopeError (a ValueError) / TypeError unless the call is inside the fair envelope and satisfies
    the mufu16 input contract.  Pure metadata unless ``check_device_contents`` (which reads the small int32
    tensors; never do that under CUDA-graph capture).  ``max_pairs`` overrides FULL_ONLY_MAX_PAIRS and exists
    ONLY for the two-wave control experiment (TEST MATRIX "B26/Q4 CONTROL (b)")."""
    check_kernel_argv()
    # --- scalars -------------------------------------------------------------------------------------------------
    if type(query_len_per_seq) is not int or query_len_per_seq not in SUPPORTED_QUERY_LENS:
        raise EnvelopeError(
            f"query_len_per_seq must be one of {SUPPORTED_QUERY_LENS} (even: pairs never straddle a sequence, "
            f"X79:8736-8741), got {query_len_per_seq!r}; odd q / q=1 dispatch to the 2x kernel"
        )
    if (
        type(num_heads) is not int
        or num_heads <= 0
        or num_heads > HEADS
        or (num_heads * query_len_per_seq) % 256 != 0
    ):
        raise EnvelopeError(
            f"rows mode requires 1..{HEADS} local heads with heads * q a multiple of 256 (a 256-row pair never "
            f"straddles a request), got num_heads={num_heads!r} q={query_len_per_seq}; other shapes dispatch 2x"
        )
    if type(page_size) is not int or page_size != PAGE_SIZE:
        raise EnvelopeError(
            f"page-native decode requires page_size={PAGE_SIZE} (X79:10165-10168), got {page_size!r}"
        )
    if type(v_pack_block) is not int or v_pack_block != V_PACK_BLOCK:
        raise EnvelopeError(
            f"the x79 wrapper's V views are fixed to SMEM_P4_V_N_PER_CTA={V_PACK_BLOCK} rows (X79:1242-1272), "
            f"got v_pack_block={v_pack_block!r}"
        )
    if (
        not isinstance(sm_scale, (int, float))
        or isinstance(sm_scale, bool)
        or not math.isfinite(sm_scale)
        or sm_scale <= 0.0
    ):
        raise EnvelopeError(
            f"sm_scale must be finite and positive (X79:10162-10163), got {sm_scale!r}"
        )
    if type(max_kv_len) is not int or max_kv_len <= 0:
        raise EnvelopeError(
            f"max_kv_len must be a positive int (X79:10169-10170), got {max_kv_len!r}"
        )
    if max_kv_len > PHYSICAL_K:
        raise EnvelopeError(
            f"max_kv_len exceeds the fixed runtime-KV profile: {max_kv_len} > {PHYSICAL_K} (X79:10171-10174)"
        )
    if (
        type(assume_consecutive_page_prefix_tiles) is not int
        or assume_consecutive_page_prefix_tiles < 0
    ):
        raise TypeError(
            "assume_consecutive_page_prefix_tiles must be a nonnegative int (X79:10175-10185)"
        )
    if assume_consecutive_page_prefix_tiles != 0:
        # X79:10198-10199 would enable the consecutive-page specialization when the promise covers every tile;
        # the FULL-only host cannot verify page contiguity, so the promise is refused (hazards H-B1/H-E3).
        raise EnvelopeError(
            "consecutive-page promise refused: the 4x host does not assert page contiguity; pass 0 "
            "(use_consecutive_page_pair stays False)"
        )
    if type(v_page_offset) is not int or v_page_offset < 0 or v_page_offset > INT32_MAX:
        raise EnvelopeError(
            f"v_page_offset must be an int in [0, {INT32_MAX}] (X79:10323-10328), got {v_page_offset!r}"
        )
    pairs_bound = FULL_ONLY_MAX_PAIRS if max_pairs is None else int(max_pairs)
    # --- output / L ----------------------------------------------------------------------------------------------
    if output.dtype not in (torch.bfloat16, torch.float16):
        raise TypeError(
            f"output dtype must be torch.float16 or torch.bfloat16 (X79:1116-1118), got {output.dtype}"
        )
    if output.dim() != 3 or output.shape[1] != num_heads or output.shape[2] != OUT_COLS:
        raise EnvelopeError(
            f"output must be [tokens, {num_heads}, {OUT_COLS}] (from_raw permutes it to the kernel's [M,512,L] "
            f"batch-slow view, MUFU:8881 / X79:10246-10249), got {tuple(output.shape)}"
        )
    num_queries = int(output.shape[0])
    if num_queries <= 0 or num_queries > CUDA_GRID_Z_MAX:
        raise EnvelopeError(
            f"queries must be in [1, {CUDA_GRID_Z_MAX}] (X79:10207-10208), got {num_queries}"
        )
    if not output.is_contiguous():
        # kernel-side strides (512, 1, M*512) == a contiguous [L, 128, 512] tensor (X79:10250-10254)
        raise EnvelopeError(
            f"output must be contiguous [tokens,128,512], got strides {output.stride()}"
        )
    if num_queries % query_len_per_seq != 0:
        raise EnvelopeError(
            f"query_len_per_seq={query_len_per_seq} must divide queries={num_queries} (X79:10220-10221)"
        )
    num_sequences = num_queries // query_len_per_seq
    num_blocks = (
        num_queries * num_heads // 128
    )  # 128-row blocks of packed [token, head] rows (exact: heads*q % 256 == 0)
    num_pairs = num_blocks // 2
    if num_pairs > pairs_bound:
        raise EnvelopeError(
            f"outside the one-wave fair envelope: B*q*H/256 = {num_pairs} pair tasks > {pairs_bound} "
            f"(tokens={num_queries}, heads={num_heads}); dispatch the 2x kernel or wait for the multi-wave planner"
        )
    # --- q --------------------------------------------------------------------------------------------------------
    if q_fp4.dtype != torch.uint8 or not q_fp4.is_contiguous():
        raise EnvelopeError("q_fp4 must be a contiguous uint8 tensor (X79:10597-10598)")
    expected_q_bytes = num_queries * num_heads * Q_BYTES
    if q_fp4.numel() != expected_q_bytes:
        raise EnvelopeError(
            f"q_fp4 must contain {expected_q_bytes} bytes (X79:10600-10602), got {q_fp4.numel()}"
        )
    if q_sf.element_size() != 1 or not q_sf.is_contiguous() or q_sf.dim() == 0:
        raise EnvelopeError(
            "q_sf must be a contiguous byte-sized tensor (X79:10603-10604; fp8e4m3 or uint8)"
        )
    expected_q_sf_bytes = num_queries * num_heads * Q_SF_GROUPS
    if q_sf.numel() != expected_q_sf_bytes:
        raise EnvelopeError(
            f"q_sf must contain {expected_q_sf_bytes} bytes (X79:10605-10607), got {q_sf.numel()}"
        )
    _check_alignment("q_fp4", q_fp4, 16)  # X79:10608; make_ptr assumed_align=16 (X79:9815)
    _check_alignment(
        "q_sf", q_sf, 32
    )  # make_ptr assumed_align=32 (X79:9817) is stricter than X79:10609's 16
    if q_batch_capacity is None:
        q_batch_capacity = num_queries
    if type(q_batch_capacity) is not int:
        raise TypeError(
            f"q_batch_capacity must be an int or None (X79:10211-10214), got {type(q_batch_capacity).__name__}"
        )
    if q_batch_capacity < num_queries or q_batch_capacity > CUDA_GRID_Z_MAX:
        raise EnvelopeError(
            f"q_batch_capacity must cover active queries and fit grid.z: queries={num_queries}, "
            f"capacity={q_batch_capacity}, limit={CUDA_GRID_Z_MAX} (X79:10215-10219)"
        )
    if (q_batch_capacity * num_heads) % 128 != 0:
        raise EnvelopeError(
            f"q_batch_capacity * heads must be a multiple of 128 rows, got {q_batch_capacity} x {num_heads}"
        )
    q_batch_blocks = q_batch_capacity * num_heads // 128
    if q_batch_capacity > num_queries:
        # the Q / QSF TMA views extend to q_tma_l = capacity queries (X79:1209, 1321): storage must exist
        need_q = q_batch_capacity * num_heads * Q_BYTES
        have_q = _storage_bytes_from_offset(q_fp4)
        if have_q < need_q:
            raise EnvelopeError(
                f"q_fp4 storage has {have_q}B, requires {need_q}B for capacity {q_batch_capacity} (X79:10236-10245)"
            )
        need_sf = q_batch_capacity * num_heads * Q_SF_GROUPS
        have_sf = _storage_bytes_from_offset(q_sf)
        if have_sf < need_sf:
            raise EnvelopeError(
                f"q_sf storage has {have_sf}B, requires {need_sf}B for capacity {q_batch_capacity} (X79:10346-10355)"
            )
    # --- scales ---------------------------------------------------------------------------------------------------
    if global_scale.dtype != torch.float32 or global_scale.numel() != 1:
        raise TypeError(
            "global_scale (kv_global_scale) must be a scalar FP32 tensor (X79:10610-10611, 10397-10398)"
        )
    if q_global_scale is None:
        q_global_scale = global_scale  # X79:10626
    if q_global_scale.dtype != torch.float32 or q_global_scale.numel() != 1:
        raise TypeError("q_global_scale must be a scalar FP32 tensor (X79:10395-10396)")
    # --- kv_cache -------------------------------------------------------------------------------------------------
    num_cache_pages, packed_dim, stride_page, stride_token, stride_packed = _kv_cache_layout(
        kv_cache, page_size
    )
    if min(stride_page, stride_token, stride_packed) <= 0:
        raise EnvelopeError("kv_cache strides must be positive (X79:9923-9927)")
    if (
        packed_dim < K_STORAGE_BYTES
        or stride_packed != 1
        or stride_token != K_STORAGE_BYTES
        or stride_page < page_size * K_STORAGE_BYTES
        or stride_page % 16 != 0
    ):
        raise EnvelopeError(
            "kv_cache must expose physical [page,128,K640/2] storage with a contiguous token payload and an "
            f"optional 16B-aligned padded page stride (X79:10288-10298); got packed_dim={packed_dim} "
            f"strides=({stride_page},{stride_token},{stride_packed})"
        )
    _check_alignment("kv_cache", kv_cache, 16)  # X79:9929
    if num_cache_pages <= 0:
        raise EnvelopeError("kv_cache must have at least one page")
    # --- v_packed -------------------------------------------------------------------------------------------------
    if v_packed.dtype != torch.uint8:
        raise TypeError(f"v_packed must be torch.uint8 (X79:9889-9890), got {v_packed.dtype}")
    if v_packed.dim() != 2:
        raise EnvelopeError(
            f"v_packed must be 2D (X79:9891-9892), got shape={tuple(v_packed.shape)}"
        )
    expected_v_rows = num_cache_pages * OUT_COLS  # (512 // block) * block rows per page (X79:10305)
    if (
        v_packed.shape[0] < expected_v_rows
        or v_packed.shape[1] < page_size // 2
        or tuple(v_packed.stride()) != (page_size // 2, 1)
    ):
        raise EnvelopeError(
            f"v_packed must expose the persistent physical-page layout [>= {expected_v_rows}, >= {page_size // 2}] "
            f"with strides {(page_size // 2, 1)} (X79:10305-10316), got shape={tuple(v_packed.shape)} "
            f"strides={v_packed.stride()}"
        )
    if v_packed.shape[0] % OUT_COLS != 0:
        raise EnvelopeError(
            f"v_packed row count must be a whole number of physical pages (X79:10317-10322), got {v_packed.shape[0]}"
        )
    _check_alignment("v_packed", v_packed, 16)  # X79:9928
    num_v_cache_pages = int(v_packed.shape[0]) // OUT_COLS
    if num_v_cache_pages > INT32_MAX:
        raise EnvelopeError(
            f"v_packed exceeds the Int32 physical-page limit (X79:10329-10332): {num_v_cache_pages}"
        )
    if v_page_offset + num_cache_pages > num_v_cache_pages:
        # NOTE: num_v_cache_pages == num_cache_pages + 1 in every real fixture (pool layout); never assert equality
        raise EnvelopeError(
            f"v_packed does not cover the requested physical-page range: offset={v_page_offset}, "
            f"pages={num_cache_pages}, available={num_v_cache_pages} (X79:10333-10338)"
        )
    # --- sf_cache / v_sf -------------------------------------------------------------------------------------------
    for name, tensor, payload_bytes, expected_pages in (
        ("sf_cache", sf_cache, KSF_PAYLOAD_BYTES, num_cache_pages),
        ("v_sf", v_sf, VSF_PAYLOAD_BYTES, num_v_cache_pages),
    ):  # X79:10357-10378 (byte-sized: fp8e4m3 real fixtures and uint8 synthetic ones both pass)
        if tensor.element_size() != 1 or tensor.dim() == 0:
            raise EnvelopeError(f"{name} must be a byte-sized paged tensor (X79:10361-10362)")
        if tensor.shape[0] != expected_pages or tensor.stride(-1) != 1:
            raise EnvelopeError(
                f"{name} must expose {expected_pages} physical pages with a contiguous innermost dimension "
                f"(X79:10363-10368), got shape={tuple(tensor.shape)} strides={tensor.stride()}"
            )
        if tensor.numel() < expected_pages * payload_bytes:
            raise EnvelopeError(
                f"{name} has {tensor.numel()} bytes, requires at least {expected_pages * payload_bytes} "
                "(X79:10369-10372)"
            )
        if tensor.stride(0) < payload_bytes:
            raise EnvelopeError(
                f"{name} page stride={tensor.stride(0)}B is smaller than its {payload_bytes}B payload (X79:10373-10376)"
            )
        if tensor.stride(0) % 16 != 0:
            raise EnvelopeError(
                f"{name} page stride={tensor.stride(0)}B must be 16B aligned (X79:10377-10378)"
            )
        _check_alignment(name, tensor, 32)  # make_ptr assumed_align=32 (X79:9818, 9821)
    # --- page table / indptr / kv_lens -------------------------------------------------------------------------------
    if src_page_ids.dtype != torch.int32 or src_page_ids.dim() != 1:
        raise EnvelopeError(
            f"src_page_ids must be a 1D int32 physical-page list (X79:10255-10259), got dtype={src_page_ids.dtype} "
            f"shape={tuple(src_page_ids.shape)}"
        )
    if src_page_ids.numel() == 0 or src_page_ids.stride(0) != 1:
        raise EnvelopeError("src_page_ids must be non-empty and contiguous (X79:10260-10264)")
    page_table_capacity = (
        num_sequences * MAX_PAGES_PER_SEQ
    )  # kernel view (1280*B,) X79:1223, host check X79:10300-10304
    if src_page_ids.numel() > page_table_capacity:
        raise EnvelopeError(
            f"src_page_ids exceeds the bucketed CSR capacity: {src_page_ids.numel()} > {page_table_capacity} "
            "(X79:10300-10304)"
        )
    if (
        paged_kv_indptr_decode.dtype != torch.int32
        or tuple(paged_kv_indptr_decode.shape) != (num_sequences + 1,)
        or paged_kv_indptr_decode.stride(0) != 1
    ):
        raise EnvelopeError(
            f"paged_kv_indptr_decode must be contiguous int32 [{num_sequences + 1}] (X79:10265-10274), got "
            f"dtype={paged_kv_indptr_decode.dtype} shape={tuple(paged_kv_indptr_decode.shape)}"
        )
    if (
        kv_lens.dtype != torch.int32
        or tuple(kv_lens.shape) != (num_sequences,)
        or kv_lens.stride(0) != 1
    ):
        raise EnvelopeError(
            f"kv_lens (valid_k) must be contiguous int32 [{num_sequences}] (X79:10275-10280), "
            f"got dtype={kv_lens.dtype} "
            f"shape={tuple(kv_lens.shape)} stride={kv_lens.stride()}"
        )
    for name, t in (
        ("src_page_ids", src_page_ids),
        ("paged_kv_indptr_decode", paged_kv_indptr_decode),
        ("kv_lens", kv_lens),
        ("global_scale", global_scale),
        ("q_global_scale", q_global_scale),
    ):
        _check_alignment(name, t, 4)  # make_ptr assumed_align=4 (X79:9822-9823, 9833-9845)
    _check_alignment("output", output, 32)  # c_ptr assumed_align=32 (X79:9824-9829)
    # --- device ---------------------------------------------------------------------------------------------------
    device = q_fp4.device
    tensors = (
        q_sf,
        kv_cache,
        sf_cache,
        v_packed,
        v_sf,
        src_page_ids,
        paged_kv_indptr_decode,
        kv_lens,
        output,
        global_scale,
        q_global_scale,
    )
    if any(t.device != device for t in tensors):
        raise EnvelopeError("all decode tensors must share one device (X79:10379-10394)")
    if require_cuda and device.type != "cuda":
        raise EnvelopeError(f"all decode tensors must be CUDA tensors (X79:10393), got {device}")
    # --- device contents (optional; small int32 tensors only) ------------------------------------------------------
    if check_device_contents:
        check_table_contents(
            src_page_ids,
            paged_kv_indptr_decode,
            kv_lens,
            num_cache_pages=num_cache_pages,
            max_kv_len=max_kv_len,
            page_size=page_size,
        )
    use_ksf_gather4 = (
        int(sf_cache.stride(0)) == page_size * K_SF_GROUPS
    )  # X79:10038-10040 with use_consecutive_page_pair False
    return EnvelopeGeometry(
        num_queries=num_queries,
        num_sequences=num_sequences,
        query_len_per_seq=query_len_per_seq,
        num_pairs=num_pairs,
        physical_m=HEADS,
        physical_k=PHYSICAL_K,
        q_batch_capacity=q_batch_capacity,
        num_cache_pages=num_cache_pages,
        num_v_cache_pages=num_v_cache_pages,
        v_page_offset=v_page_offset,
        kv_page_stride_bytes=stride_page,
        ksf_page_stride_bytes=int(sf_cache.stride(0)),
        vsf_page_stride_bytes=int(v_sf.stride(0)),
        softmax_scale_log2=float(sm_scale) * LOG2_E,
        use_ksf_gather4=use_ksf_gather4,
        use_consecutive_page_pair=False,
        output_dtype=output.dtype,
        max_kv_len=max_kv_len,
        device=device,
        num_heads=num_heads,
        num_blocks=num_blocks,
        q_batch_blocks=q_batch_blocks,
    )


def check_table_contents(
    src_page_ids: torch.Tensor,
    paged_kv_indptr_decode: torch.Tensor,
    kv_lens: torch.Tensor,
    *,
    num_cache_pages: int,
    max_kv_len: int,
    page_size: int = PAGE_SIZE,
) -> None:
    """Content checks on the CSR tables (device -> host copy of a few KiB; never under graph capture).
    The kernel gathers pages CSR-style: page_begin = indptr[seq], page_count = indptr[seq+1]-indptr[seq]
    (X79:8762-8763) and reads mPageTable_pl[page_begin + min(logical, page_count-1)] (X79:1629-1640)."""
    indptr = paged_kv_indptr_decode.detach().to("cpu", torch.int64)
    lens = kv_lens.detach().to("cpu", torch.int64)
    if int(indptr[0]) != 0:
        raise EnvelopeError(f"paged_kv_indptr_decode[0] must be 0, got {int(indptr[0])}")
    if bool((indptr[1:] < indptr[:-1]).any()):
        raise EnvelopeError("paged_kv_indptr_decode must be non-decreasing")
    if int(indptr[-1]) > src_page_ids.numel():
        raise EnvelopeError(
            f"paged_kv_indptr_decode[B]={int(indptr[-1])} exceeds src_page_ids.numel()={src_page_ids.numel()} "
            "(CSR gather, X79:1629-1640, 8762-8763)"
        )
    if bool((lens < 0).any()) or int(lens.max()) > max_kv_len:
        raise EnvelopeError(
            f"kv_lens must be in [0, max_kv_len={max_kv_len}], got min={int(lens.min())} max={int(lens.max())}"
        )
    pages_per_seq = indptr[1:] - indptr[:-1]
    need = (lens + page_size - 1) // page_size
    if bool((pages_per_seq < need).any()):
        bad = int((pages_per_seq < need).nonzero()[0])
        raise EnvelopeError(
            f"sequence {bad} has {int(pages_per_seq[bad])} pages for kv_len {int(lens[bad])} (needs {int(need[bad])})"
        )
    if bool((pages_per_seq > MAX_PAGES_PER_SEQ).any()):
        raise EnvelopeError(
            f"a sequence has more than {MAX_PAGES_PER_SEQ} pages (160K-token profile, X79:400-401)"
        )
    used = int(indptr[-1])
    if used > 0:
        ids = src_page_ids.detach()[:used].to("cpu", torch.int64)
        if bool((ids < 0).any()) or int(ids.max()) >= num_cache_pages:
            raise EnvelopeError(
                f"src_page_ids must be in [0, {num_cache_pages}), got min={int(ids.min())} max={int(ids.max())}"
            )


def zero_queries(kv_lens: list[int] | torch.Tensor, query_len_per_seq: int) -> list[int]:
    """Queries whose per-query valid_k is 0 (kernel formula X79:8739: max(valid_k[seq] - (q-1-offset), 0));
    their output rows must be exactly 0 (TEST MATRIX zero_rows_ok)."""
    lens = kv_lens.tolist() if isinstance(kv_lens, torch.Tensor) else list(kv_lens)
    q = int(query_len_per_seq)
    return [
        s * q + o for s, L in enumerate(lens) for o in range(q) if max(int(L) - (q - 1 - o), 0) == 0
    ]


# --- compile / launch ------------------------------------------------------------------------------------------------
_COMPILE_CACHE: dict[tuple, Any] = {}
_EXECUTOR_CACHE: OrderedDict[tuple, Any] = OrderedDict()
_PREPARED_CACHE: OrderedDict[tuple, Any] = OrderedDict()
_CACHE_LOCK = threading.Lock()
_CACHE_CAPACITY = 256  # X79:9934-9935
_VALIDATED_BINDINGS: set[tuple] = set()
_HEADER_SCRATCH: dict[torch.device, torch.Tensor] = {}
_SCRATCH_FILL = 1  # if the kernel implementer kept the header[576] gate (X79:8845-8847), a positive word passes it


@dataclass
class _PreparedCall:  # X79:9938-9946
    executor: Any
    runtime_args: tuple
    execution_args: Any
    adapted_args: Any
    geometry: EnvelopeGeometry

    def run(self):
        return self.executor.run_compiled_program(self.execution_args)


def _header_scratch(device: torch.device) -> torch.Tensor:
    """Int32 scratch behind accum_ptr/row_max_ptr/row_sum_ptr: HEADER_SCRATCH_WORDS words (INTERFACE CONTRACT).
    FULL-only never dereferences the carry/meta views (X79 9420-9423, 9481-9502 are PREFIX/RESUME-only)."""
    t = _HEADER_SCRATCH.get(device)
    if t is None:
        words = max(int(kernel_module().HEADER_SCRATCH_WORDS), 8)
        t = torch.full((words,), _SCRATCH_FILL, dtype=torch.int32, device=device)
        _HEADER_SCRATCH[device] = t
    return t


def _make_wrapper_ptrs(addresses: list[int], output_dtype: torch.dtype) -> tuple:
    """The 16 cute.Pointer args in wrapper order (X79:1160-1175), typed exactly as X79:9795-9846 _make_fused_ptrs."""
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import make_ptr

    if len(addresses) != 20:
        raise ValueError(f"20 pointer addresses expected, got {len(addresses)}")
    g = cute.AddressSpace.gmem
    out_t = cutlass.BFloat16 if output_dtype == torch.bfloat16 else cutlass.Float16  # X79:9788-9792
    spec = (
        (cutlass.Float4E2M1FN, 16),  # q_ptr            X79:9815
        (cutlass.Float4E2M1FN, 16),  # k_ptr            X79:9816
        (cutlass.Uint8, 32),  # q_sf_ptr         X79:9817
        (cutlass.Uint8, 32),  # k_sf_ptr         X79:9818
        (cutlass.Float4E2M1FN, 16),  # b_ptr (v_packed) X79:9819
        (cutlass.Uint8, 32),  # sfa_ptr (unused by the wrapper body; scratch) X79:9820
        (cutlass.Uint8, 32),  # sfb_ptr (v_sf)   X79:9821
        (cutlass.Int32, 4),  # page_table_ptr   X79:9822
        (cutlass.Int32, 4),  # valid_k_ptr      X79:9823
        (out_t, 32),  # c_ptr            X79:9824-9829
        (cutlass.Float32, 32),  # accum_ptr        X79:9830
        (cutlass.Float32, 32),  # row_max_ptr      X79:9831
        (cutlass.Float32, 32),  # row_sum_ptr      X79:9832
        (cutlass.Int32, 4),  # page_indptr_ptr  X79:9833
        (cutlass.Float32, 4),  # q_global_scale   X79:9834-9839
        (cutlass.Float32, 4),  # kv_global_scale  X79:9840-9845
        (cutlass.Int32, 16),  # plan_ptr      (split mode; scratch otherwise)
        (cutlass.Float32, 32),  # partial_ptr   (split mode)
        (cutlass.Float32, 32),  # rowmax_ptr    (split mode)
        (cutlass.Float32, 32),  # rowsum_ptr    (split mode)
    )
    return tuple(
        make_ptr(dtype, int(addr), g, assumed_align=align)
        for (dtype, align), addr in zip(spec, addresses)
    )


def _current_cu_stream():
    """Plain caller-stream handle (rig cute_dsl_utils.py:19-21); deliberately WITHOUT x79's
    DKG_MLA_EXPLICIT_STREAM side effect (X79:637-643 calls torch.cuda.set_stream)."""
    import cuda.bindings.driver as cuda

    return cuda.CUstream(torch.cuda.current_stream().cuda_stream)


def envelope_reason_static(
    *,
    num_heads: int,
    query_len: int,
    num_requests: int,
    max_kv_len: int,
    sm_count: int,
    helix: bool = False,
    softmax_stats: bool = False,
    v_packed: bool = True,
) -> str | None:
    """Dispatch predicate on STATIC shape values only (CUDA-graph buckets replay a fixed path): None when the 4x
    kernel covers the call (one-wave FULL-only launch, or the split mode for larger batches / long KV), else the
    reason it does not (the caller keeps the 2x path). Mirrors validate_envelope's shape rules without touching
    tensors (fp4_mla/__init__.py dispatch)."""
    if query_len not in SUPPORTED_QUERY_LENS:
        return f"query_len {query_len} not in {SUPPORTED_QUERY_LENS}"
    if num_heads <= 0 or num_heads > HEADS or (num_heads * query_len) % 256 != 0:
        return (
            f"num_heads {num_heads} x query_len {query_len} is not a multiple of 256 rows "
            f"(a 256-row pair task must stay inside one request)"
        )
    if num_requests <= 0:
        return "no requests"
    if max_kv_len <= 0 or max_kv_len > PHYSICAL_K:
        return f"max_kv_len {max_kv_len} outside (0, {PHYSICAL_K}]"
    if sm_count < MIN_SM_COUNT:
        return f"sm_count {sm_count} < {MIN_SM_COUNT}"
    if helix:
        return "Helix per-row KV bounds are not supported"
    if softmax_stats:
        return "softmax stats are not emitted by the 4x kernel"
    if not v_packed:
        return "requires the persistent packed-V cache"
    pairs = pair_tasks(num_heads, query_len, num_requests)
    min_pairs = _env_int(MIN_PAIRS_ENV, MIN_PAIRS_DERIVED)
    if min_pairs < 0:
        m_tiles_2x = max(1, num_heads * query_len // 128)
        two_piece_plan_fits = 2 * num_requests * m_tiles_2x <= sm_count // 2
        if two_piece_plan_fits and not (query_len == 2 and pairs >= Q2_MIN_PAIRS):
            return (
                f"{num_requests} requests x {m_tiles_2x} m-tiles: the 2x packed plan cuts every request in two "
                f"inside one wave of {sm_count // 2} clusters and halves its makespan; an under-filled 4x wave "
                f"cannot ({MIN_PAIRS_ENV})"
            )
    elif pairs < min_pairs:
        return f"{pairs} pairs < {min_pairs} ({MIN_PAIRS_ENV})"
    if pairs > FULL_ONLY_MAX_PAIRS:
        if not split_enabled():
            return f"{pairs} pairs > one wave of {FULL_ONLY_MAX_PAIRS} and {SPLIT_ENV}=0"
        if pairs > SPLIT_MAX_PAIRS:
            return f"{pairs} pairs > split-mode bound {SPLIT_MAX_PAIRS}"
        min_kv = _env_int(SPLIT_MIN_KV_MULTIWAVE_ENV, SPLIT_MIN_KV_MULTIWAVE_DEFAULT)
        if max_kv_len < min_kv:
            return (
                f"{pairs} pairs > one wave and max_kv_len {max_kv_len} < {min_kv}: the per-task fixed cost "
                f"of the extra waves outweighs the 4x per-tile edge ({SPLIT_MIN_KV_MULTIWAVE_ENV})"
            )
    return None


def vpfl_decision(geom: EnvelopeGeometry, use_vpfl: bool | None) -> tuple[bool, str]:
    """(enable, reason) for the V L2 prefetch.  None = automatic: on iff the V pool's address span
    (num_v_cache_pages x 32 KiB, an upper bound of the spread of the referenced V pages) is <= VPFL_MAX_V_SPAN_BYTES."""
    if use_vpfl is not None:
        if type(use_vpfl) is not bool:
            raise TypeError("use_vpfl must be a bool or None")
        return use_vpfl, "forced"
    span = geom.num_v_cache_pages * V_PAGE_BYTES
    if span <= VPFL_MAX_V_SPAN_BYTES:
        return (
            True,
            f"auto: V pool span {span / 2**20:.0f} MiB <= {VPFL_MAX_V_SPAN_BYTES / 2**20:.0f} MiB",
        )
    return (
        False,
        f"auto: V pool span {span / 2**20:.0f} MiB > {VPFL_MAX_V_SPAN_BYTES / 2**20:.0f} MiB",
    )


def _scalar_args(geom: EnvelopeGeometry, stream, *, exemplar: bool = False) -> tuple:
    """Wrapper args 17-30 (X79:1176-1189) in order.  With ``exemplar`` the page counts/strides are the type-only
    stand-ins the template uses at compile time (compile_balanced_template.py:136-138); otherwise the real values
    (X79:10513-10528 + requested_cut, dead in FULL-only -> 0)."""
    import cutlass as ctm

    I32, I64, F32 = ctm.Int32, ctm.Int64, ctm.Float32
    if exemplar:
        return (
            (OUT_COLS, PHYSICAL_K),
            I32(geom.physical_m),
            I32(geom.num_blocks),
            I32(geom.q_batch_blocks),
            I32(1),
            I32(1),
            I32(0),
            I64(1),
            I64(1),
            I64(1),
            F32(1.0),
            F32(1.0),
            I32(0),
            I32(1),
            I32(1),
            I32(1),
            stream,
        )
    return (
        (OUT_COLS, PHYSICAL_K),  # problem_size (n, k) X79:10515
        I32(geom.physical_m),  # runtime_m = 128      X79:10516
        I32(geom.num_blocks),  # runtime_l = 128-row blocks (grid z = blocks // 2); == L for H128
        I32(geom.q_batch_blocks),  # q_tma_l in blocks    X79:10518
        I32(geom.num_cache_pages),  # X79:10519
        I32(geom.num_v_cache_pages),  # X79:10520
        I32(geom.v_page_offset),  # X79:10521
        I64(geom.kv_page_stride_bytes),  # X79:10522
        I64(geom.ksf_page_stride_bytes),  # X79:10523
        I64(geom.vsf_page_stride_bytes),  # X79:10524
        F32(geom.softmax_scale_log2),  # X79:10525
        F32(1.0),  # pv_output_scale      X79:10526 / MUFU:8669
        I32(0),  # requested_cut: PREFIX cut, dead in FULL-only (X79:8743-8760)
        I32(1 if geom.vpfl_enabled else 0),  # vpfl_enable (runtime; vpfl_decision)
        I32(
            max(int(geom.task_capacity), 1)
        ),  # task_capacity (split mode plan slots; 1 = unused in FULL mode)
        I32(max(int(geom.used_tasks), 1)),  # used_tasks (split mode grid; 1 = unused in FULL mode)
        stream,  # X79:10527
    )


def _constexpr_kwargs(geom: EnvelopeGeometry, enable_mxi_imlp: bool) -> dict:
    # X79:1190-1194; page_size MUST be passed (the Constexpr default is KV_TILE=256, X79:1190/360); no
    # workspace_contract in the FULL-only wrapper (INTERFACE CONTRACT).
    return dict(
        page_size=PAGE_SIZE,
        use_mixed_imlp=bool(enable_mxi_imlp),
        query_len_per_seq=geom.query_len_per_seq,
        use_consecutive_page_pair=False,
        use_ksf_gather4=bool(geom.use_ksf_gather4),
        heads=int(geom.num_heads),
        split_mode=bool(geom.split_mode),
    )


def _compile(geom: EnvelopeGeometry, ptrs: tuple, stream, enable_mxi_imlp: bool):
    key = (*geom.compile_key, bool(enable_mxi_imlp))
    cached = _COMPILE_CACHE.get(key)
    if cached is not None:
        return cached
    k = kernel_module()
    compiled = k._compile_cutedsl(  # X79:200-210: BaseDSL.enable_pyir() + cute.compile
        k.fp4_mla_4x_full_ctm,
        *ptrs,
        *_scalar_args(geom, stream, exemplar=True),
        **_constexpr_kwargs(geom, enable_mxi_imlp),
        options=COMPILE_OPTIONS,
    )
    _COMPILE_CACHE[key] = compiled
    return compiled


def compiled_variants() -> list[tuple]:
    """Keys of the compiled variants:
    (n, page_size, output_dtype, q, use_consecutive_page_pair, use_ksf_gather4, use_mixed_imlp)."""
    return list(_COMPILE_CACHE)


def compile_variant(
    query_len_per_seq: int,
    output_dtype: torch.dtype = torch.bfloat16,
    *,
    use_ksf_gather4: bool = True,
    enable_mxi_imlp: bool = True,
    device: torch.device | int | None = None,
    num_heads: int = HEADS,
    split_mode: bool = False,
):
    """Compile (or fetch) one variant without real inputs, using type-only exemplar pointers like
    compile_balanced_template.py:131-140.  Returns the compiled callable."""
    if query_len_per_seq not in SUPPORTED_QUERY_LENS:
        raise EnvelopeError(
            f"query_len_per_seq must be one of {SUPPORTED_QUERY_LENS}, got {query_len_per_seq}"
        )
    if output_dtype not in (torch.bfloat16, torch.float16):
        raise TypeError("output dtype must be torch.float16 or torch.bfloat16")
    dev = torch.device(
        f"cuda:{device}" if isinstance(device, int) else (device or torch.cuda.current_device())
    )
    if dev.type != "cuda":
        raise EnvelopeError("compile_variant needs a CUDA device")
    with torch.cuda.device(dev):
        exemplar = torch.zeros(1024, dtype=torch.uint8, device=dev)
        scratch = _header_scratch(dev)
        addresses = [exemplar.data_ptr()] * 20
        addresses[10:13] = [scratch.data_ptr()] * 3
        if (num_heads * query_len_per_seq) % 256 != 0:
            raise EnvelopeError(
                f"num_heads * query_len_per_seq must be a multiple of 256, got {num_heads} x {query_len_per_seq}"
            )
        geom = EnvelopeGeometry(
            num_queries=2,
            num_sequences=1,
            query_len_per_seq=query_len_per_seq,
            num_pairs=1,
            physical_m=HEADS,
            num_heads=num_heads,
            num_blocks=2,
            q_batch_blocks=2,
            split_mode=split_mode,
            task_capacity=1,
            used_tasks=1,
            physical_k=PHYSICAL_K,
            q_batch_capacity=2,
            num_cache_pages=1,
            num_v_cache_pages=1,
            v_page_offset=0,
            kv_page_stride_bytes=PAGE_SIZE * K_STORAGE_BYTES,
            ksf_page_stride_bytes=KSF_PAYLOAD_BYTES,
            vsf_page_stride_bytes=VSF_PAYLOAD_BYTES,
            softmax_scale_log2=1.0,
            use_ksf_gather4=use_ksf_gather4,
            use_consecutive_page_pair=False,
            output_dtype=output_dtype,
            max_kv_len=PHYSICAL_K,
            device=dev,
        )
        ptrs = _make_wrapper_ptrs(addresses, output_dtype)
        return _compile(geom, ptrs, _current_cu_stream(), enable_mxi_imlp)


def _class_defines_callables(value: object, *names: str) -> bool:  # X79:9954-9956
    value_type = type(value)
    return all(callable(getattr(value_type, name, None)) for name in names)


def _cache_get(cache: OrderedDict, key):
    with _CACHE_LOCK:
        v = cache.get(key)
        if v is not None:
            cache.move_to_end(key)
        return v


def _cache_put(cache: OrderedDict, key, value):
    with _CACHE_LOCK:
        existing = cache.get(key)
        if existing is not None:
            cache.move_to_end(key)
            return existing
        cache[key] = value
        if len(cache) > _CACHE_CAPACITY:
            cache.popitem(last=False)
        return value


def run_fp4_mla_decode_4x_full(
    q_fp4: torch.Tensor,
    q_sf: torch.Tensor,
    kv_cache: torch.Tensor,
    sf_cache: torch.Tensor,
    v_packed: torch.Tensor,
    v_sf: torch.Tensor,
    global_scale: torch.Tensor,
    src_page_ids: torch.Tensor,
    paged_kv_indptr_decode: torch.Tensor,
    kv_lens: torch.Tensor,
    output: torch.Tensor,
    *,
    max_kv_len: int,
    sm_scale: float,
    num_heads: int,
    query_len_per_seq: int,
    q_global_scale: torch.Tensor | None = None,
    page_size: int = PAGE_SIZE,
    v_pack_block: int = V_PACK_BLOCK,
    v_page_offset: int = 0,
    q_batch_capacity: int | None = None,
    assume_valid_k_prefix_tiles: int = 0,
    assume_consecutive_page_prefix_tiles: int = 0,
    partition_runtime_valid_k: bool = False,
    enable_mxi_imlp: bool = True,
    validate_contents: bool | None = None,
    use_vpfl: bool | None = None,
    split: dict | None = None,
    _test_only_max_pairs: int | None = None,
) -> EnvelopeGeometry:
    """Launch the 4x FULL-only decode on the caller's current stream.  Positional tensors and keyword names follow
    MUFU:8813-8838 (``run_trtllm_fp4_mla_decode_page_native_from_raw``); ``kv_global_scale = global_scale`` and
    ``q_global_scale`` defaults to it (MUFU:8884-8885 / X79:10626-10627).

    ``validate_contents``: None = read the int32 tables once per new prepared call when not capturing;
    ``use_vpfl``: None = host decision on the V pool span (``vpfl_decision``), True/False = force the V L2 prefetch;
    ``_test_only_max_pairs``: bypass of the 46-pair bound for the two-wave control run ONLY (never in production).
    """
    # interface-compatible no-ops, type-checked like MUFU:8161-8171 / X79:10175-10192 and then ignored
    if type(assume_valid_k_prefix_tiles) is not int or assume_valid_k_prefix_tiles < 0:
        raise TypeError("assume_valid_k_prefix_tiles must be a nonnegative int")
    if type(partition_runtime_valid_k) is not bool:
        raise TypeError("partition_runtime_valid_k must be a bool")
    if type(enable_mxi_imlp) is not bool:
        raise TypeError("enable_mxi_imlp must be a bool")
    if _test_only_max_pairs is not None and (
        type(_test_only_max_pairs) is not int or _test_only_max_pairs < FULL_ONLY_MAX_PAIRS
    ):
        raise EnvelopeError(
            "_test_only_max_pairs must be an int >= FULL_ONLY_MAX_PAIRS (test-only two-wave control)"
        )
    capturing = torch.cuda.is_current_stream_capturing()
    if capturing and validate_contents:
        raise RuntimeError(
            "validate_contents reads device tables and cannot run under CUDA-graph capture"
        )
    geom = validate_envelope(
        q_fp4,
        q_sf,
        kv_cache,
        sf_cache,
        v_packed,
        v_sf,
        global_scale,
        src_page_ids,
        paged_kv_indptr_decode,
        kv_lens,
        output,
        max_kv_len=max_kv_len,
        sm_scale=sm_scale,
        num_heads=num_heads,
        query_len_per_seq=query_len_per_seq,
        q_global_scale=q_global_scale,
        page_size=page_size,
        v_pack_block=v_pack_block,
        v_page_offset=v_page_offset,
        q_batch_capacity=q_batch_capacity,
        assume_consecutive_page_prefix_tiles=assume_consecutive_page_prefix_tiles,
        max_pairs=(CUDA_GRID_Z_MAX if split is not None else _test_only_max_pairs),
        require_cuda=True,
        check_device_contents=False,
    )
    if split is not None:
        # Milestone 2 split mode: plan (fp4_mla_packed_reduce layout), FP32 partial workspace, task_capacity slots.
        # The one-wave bound does not apply (multi-wave by design); grid z = task_capacity * pairs_per_request.
        cap = int(split["task_capacity"])
        used = int(split.get("used_tasks", cap))
        rows = geom.num_queries * geom.num_heads // geom.num_sequences
        plan_t, part_t, max_t, sum_t = (
            split["plan"],
            split["partial"],
            split["maxima"],
            split["sums"],
        )
        if cap <= 0 or used <= 0 or used > cap or used * (rows // 256) > CUDA_GRID_Z_MAX:
            raise EnvelopeError(
                f"used_tasks {used} <= task_capacity {cap} and used x pairs {rows // 256} must fit grid.z"
            )
        if (
            plan_t.dtype != torch.int32
            or not plan_t.is_contiguous()
            or plan_t.numel() < 3 + geom.num_sequences + 3 * cap
        ):
            raise EnvelopeError(
                "split plan must be contiguous int32 with >= 3 + requests + 3 * task_capacity entries"
            )
        for name, t, need in (
            ("partial", part_t, cap * rows * OUT_COLS),
            ("maxima", max_t, cap * rows),
            ("sums", sum_t, cap * rows),
        ):
            if t.dtype != torch.float32 or not t.is_contiguous() or t.numel() < need:
                raise EnvelopeError(
                    f"split {name} must be contiguous float32 with >= {need} elements, got {tuple(t.shape)} {t.dtype}"
                )
            _check_alignment(name, t, 32)
        geom = dataclasses.replace(geom, split_mode=True, task_capacity=cap, used_tasks=used)
    if q_global_scale is None:
        q_global_scale = global_scale
    vpfl_on, _vpfl_reason = vpfl_decision(geom, use_vpfl)
    if vpfl_on != geom.vpfl_enabled:
        geom = dataclasses.replace(geom, vpfl_enabled=vpfl_on)
    import cuda.bindings.driver as cuda

    stream = _current_cu_stream()
    device_index = torch.cuda.current_device()
    context_result, current_context = cuda.cuCtxGetCurrent()  # X79:10400-10405
    if context_result != cuda.CUresult.CUDA_SUCCESS:
        raise RuntimeError(
            f"Failed to query the current CUDA context for FP4 MLA launch: {context_result}."
        )
    scratch = _header_scratch(output.device)
    scratch_ptr = scratch.data_ptr()
    # pointer slots in wrapper order (X79:1160-1175); sfa is unused by the wrapper body and x79/mufu16 pass the
    # output pointer there (X79:10411, 10429); accum/row_max/row_sum get the tiny header scratch (INTERFACE CONTRACT)
    addresses = [
        q_fp4.data_ptr(),
        kv_cache.data_ptr(),
        q_sf.data_ptr(),
        sf_cache.data_ptr(),
        v_packed.data_ptr(),
        output.data_ptr(),
        v_sf.data_ptr(),
        src_page_ids.data_ptr(),
        kv_lens.data_ptr(),
        output.data_ptr(),
        scratch_ptr,
        scratch_ptr,
        scratch_ptr,
        paged_kv_indptr_decode.data_ptr(),
        q_global_scale.data_ptr(),
        global_scale.data_ptr(),
    ]
    if split is not None:
        addresses += [
            split["plan"].data_ptr(),
            split["partial"].data_ptr(),
            split["maxima"].data_ptr(),
            split["sums"].data_ptr(),
        ]
    else:
        addresses += [scratch_ptr] * 4
    compile_key = (*geom.compile_key, bool(enable_mxi_imlp))
    compiled = _COMPILE_CACHE.get(compile_key)
    if compiled is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "FP4 MLA 4x: CUDA-graph capture requires an eager warmup call for this variant first"
            )
        k = kernel_module()
        if _test_only_max_pairs is None and split is None and hasattr(k, "full_validate_envelope"):
            # kernel-side mirror of the gate (fp4_mla_cutedsl_4x.py full_validate_envelope); skipped only for the
            # explicit two-wave control bypass, which it would (correctly) reject
            k.full_validate_envelope(
                queries=geom.num_queries,
                heads=geom.num_heads,
                output_cols=OUT_COLS,
                query_len=geom.query_len_per_seq,
                max_kv_len=max_kv_len,
            )
        compiled = _compile(
            geom, _make_wrapper_ptrs(addresses, output.dtype), stream, enable_mxi_imlp
        )
    supports_prepared = _class_defines_callables(
        compiled, "to", "generate_execution_args", "run_compiled_program"
    )
    prepared_key = (
        id(compiled),
        device_index,
        int(current_context),
        int(stream),
        *addresses,
        geom.physical_m,
        geom.num_queries,
        geom.q_batch_capacity,
        geom.num_heads,
        geom.num_blocks,
        geom.q_batch_blocks,
        geom.num_cache_pages,
        geom.num_v_cache_pages,
        geom.v_page_offset,
        geom.kv_page_stride_bytes,
        geom.ksf_page_stride_bytes,
        geom.vsf_page_stride_bytes,
        geom.softmax_scale_log2,
        output.dtype,
        bool(enable_mxi_imlp),
        geom.vpfl_enabled,
        geom.split_mode,
        geom.task_capacity,
        geom.used_tasks,
    )
    if supports_prepared:
        prepared = _cache_get(_PREPARED_CACHE, prepared_key)
        if prepared is not None:
            prepared.run()
            return prepared.geometry
    binding_key = prepared_key[
        4:
    ]  # pointers + scalars, independent of the compiled object / stream
    if validate_contents or (
        validate_contents is None and not capturing and binding_key not in _VALIDATED_BINDINGS
    ):
        # first sight of this binding: check the CSR tables once (a few KiB of int32, synchronizing copy)
        check_table_contents(
            src_page_ids,
            paged_kv_indptr_decode,
            kv_lens,
            num_cache_pages=geom.num_cache_pages,
            max_kv_len=max_kv_len,
            page_size=page_size,
        )
        _VALIDATED_BINDINGS.add(binding_key)
        if len(_VALIDATED_BINDINGS) > 4 * _CACHE_CAPACITY:
            _VALIDATED_BINDINGS.clear()
    ptrs = _make_wrapper_ptrs(addresses, output.dtype)
    runtime_args = (*ptrs, *_scalar_args(geom, stream))
    if not supports_prepared:  # X79:10529-10531
        compiled(*runtime_args)
        return geom
    executor_key = (id(compiled), device_index, int(current_context))
    executor = _cache_get(_EXECUTOR_CACHE, executor_key)
    if executor is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "FP4 MLA CUDA Graph capture requires an eager warmup for the compiled kernel, device, and CUDA context."
            )
        candidate = compiled.to(device_index)  # X79:10539
        if not _class_defines_callables(
            candidate, "generate_execution_args", "run_compiled_program"
        ):
            compiled(*runtime_args)
            return geom
        executor = _cache_put(_EXECUTOR_CACHE, executor_key, candidate)
    execution_args, adapted_args = executor.generate_execution_args(*runtime_args)  # X79:10546
    prepared = _cache_put(
        _PREPARED_CACHE,
        prepared_key,
        _PreparedCall(executor, runtime_args, execution_args, adapted_args, geom),
    )
    prepared.run()
    return geom


def describe() -> dict:
    """Static description for logs/JSON (no CUDA needed)."""
    import hashlib

    info = {
        "host_file": str(Path(__file__).resolve()),
        "kernel_file": str(KERNEL_FILE),
        "kernel_present": KERNEL_FILE.is_file(),
        "full_only_max_pairs": FULL_ONLY_MAX_PAIRS,
        "supported_query_lens": list(SUPPORTED_QUERY_LENS),
        "compile_options": COMPILE_OPTIONS,
        "physical_k": PHYSICAL_K,
        "page_size": PAGE_SIZE,
    }
    for key, path in (("host_sha256", Path(__file__)), ("kernel_sha256", KERNEL_FILE)):
        try:
            info[key] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        except OSError:
            info[key] = None
    return info
