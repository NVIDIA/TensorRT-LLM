"""MHC CUDA kernels — torch.ops.trtllm interface.

Kernels (cpp/tensorrt_llm/kernels/mhcKernels/):
  - mhc_big_fuse:        fused RMS + sigmoid + Sinkhorn + pre_apply_mix
                         (NUM_SPLITS=1: normal, =16: with split-K reduction)
  - mhc_gemm_sqrsum_fma: FP32 FMA GEMM + sqrsum (fused, inline PTX)
  - mhc_hc_head_apply:   RMS norm + sigmoid + weighted sum
  - mhc_post_mapping:    out = post * x + comb.T @ residual
DeepGEMM wrapper:
  - gemm_rms_dg:         TF32 GEMM + sqrsum via DeepGEMM (optional split-K)

Backend selection for pre_mapping is handled by the autotuner, which profiles
all available backends (FMA, DeepGEMM split-K, DeepGEMM no-split) at warmup.
Falls back to FMA when DeepGEMM is unavailable or autotuner cache misses.
"""

from functools import lru_cache
from typing import Any, List

import torch

from tensorrt_llm._torch.autotuner import (
    AutoTuner,
    ConstraintSpec,
    DynamicTensorSpec,
    OptimizationProfile,
    TunableRunner,
    TuningConfig,
)
from tensorrt_llm._utils import get_sm_version


@lru_cache(maxsize=1)
def _fused_hc_mma_supported() -> bool:
    """tcgen05 TF32 MMA paths (Path B / Path D, "fused_*_mma") require SM100.

    Match the C++ compile guard for the tcgen05 kernels:
    __CUDA_ARCH__ >= 1000 && __CUDA_ARCH__ < 1100. On other GPU generations,
    only the FMA paths (Path E / Path F, "fused_*_fma") are safe to run.
    Called lazily so the module can still be imported on a host without a CUDA
    device (e.g. CPU-only lint / typecheck).
    """
    try:
        sm_version = get_sm_version()
        return 100 <= sm_version < 110
    except Exception:
        return False


_DG_NUM_SPLITS = 16


# ---------------------------------------------------------------------------
# Python API — low-level (kernel-level interfaces)
# ---------------------------------------------------------------------------


def mhc_big_fuse_cuda(
    y_acc: torch.Tensor,
    r_acc: torch.Tensor,
    residual: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    post_mix: torch.Tensor,
    comb_mix: torch.Tensor,
    layer_input: torch.Tensor,
    M: int,
    K: int,
    hidden_size: int,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    num_splits: int = 1,
    block_size: int = 0,
):
    torch.ops.trtllm.mhc_big_fuse(
        y_acc,
        r_acc,
        residual,
        hc_scale,
        hc_base,
        post_mix,
        comb_mix,
        layer_input,
        M,
        K,
        hidden_size,
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_mult_value,
        sinkhorn_repeat,
        num_splits,
        block_size,
    )


_dg_fn_cache = None
_dg_available = None


def _get_dg_fn():
    """Import tf32_hc_prenorm_gemm (standalone deep_gemm or bundled).

    Returns the function, or None if DeepGEMM is unavailable.
    """
    global _dg_fn_cache, _dg_available
    if _dg_available is not None:
        return _dg_fn_cache
    try:
        from deep_gemm import tf32_hc_prenorm_gemm

        _dg_fn_cache = tf32_hc_prenorm_gemm
    except ImportError:
        try:
            from tensorrt_llm.deep_gemm import tf32_hc_prenorm_gemm

            _dg_fn_cache = tf32_hc_prenorm_gemm
        except ImportError:
            _dg_fn_cache = None
    _dg_available = _dg_fn_cache is not None
    return _dg_fn_cache


def mhc_gemm_rms_dg_cuda(
    x: torch.Tensor,
    w_nk: torch.Tensor,
    M: int,
    N: int,
    K: int,
    num_splits: int = 16,
):
    """DeepGEMM TF32 GEMM + sqrsum with optional split-K.

    Args:
        x:     [M, K] bfloat16 input
        w_nk:  [N, K] float32 weight (K-major, as required by DeepGEMM)
        num_splits: 1 for no split-K, >1 for split-K

    Returns (y_acc, r_acc, num_splits):
        num_splits == 1: y_acc [M, N] fp32, r_acc [M] fp32
        num_splits > 1:  y_acc [num_splits, M, N] fp32, r_acc [num_splits, M] fp32
    """
    dg_fn = _get_dg_fn()
    assert dg_fn is not None, "DeepGEMM is not available"
    x = x.contiguous()
    w_nk = w_nk.contiguous()

    if num_splits <= 1:
        y_acc = torch.empty((M, N), dtype=torch.float32, device=x.device)
        r_acc = torch.empty((M,), dtype=torch.float32, device=x.device)
        dg_fn(x, w_nk, y_acc, r_acc)
        return y_acc, r_acc, 1
    else:
        y_acc = torch.empty((num_splits, M, N), dtype=torch.float32, device=x.device)
        r_acc = torch.empty((num_splits, M), dtype=torch.float32, device=x.device)
        dg_fn(x, w_nk, y_acc, r_acc, num_splits=num_splits)
        return y_acc, r_acc, num_splits


def mhc_gemm_rms_fma_cuda(
    x: torch.Tensor,
    w: torch.Tensor | None,
    M: int,
    N: int,
    K: int,
    w_t: torch.Tensor | None = None,
    tile_n: int = 0,
    tile_m: int = 0,
):
    """Split-N FP32 FMA fused GEMM + sqrsum on CUDA cores (no tensor cores).

    Args:
        tile_n: FMA N-tile size. 0 = auto-select.
                Valid: {1,2,3,4,6,8,12,24}.
        tile_m: FMA M-tile size. 0 = auto-select.
                Valid: {1, 2} (2 only with tile_n=8).

    Returns (y_acc [M, N] fp32, r_acc [M] fp32).
    """
    x = x.contiguous()
    if w_t is None:
        w_t = w.t().contiguous()

    y_acc = torch.empty((M, N), dtype=torch.float32, device=x.device)
    r_acc = torch.empty((M,), dtype=torch.float32, device=x.device)

    torch.ops.trtllm.mhc_gemm_sqrsum_fma(
        x,
        w_t,
        y_acc,
        r_acc,
        M,
        N,
        K,
        tile_n,
        tile_m,
    )

    return y_acc, r_acc


# ---------------------------------------------------------------------------
# Autotuner runner for pre_mapping backend selection
# ---------------------------------------------------------------------------


def _mhc_gen_tuning_buckets(x: int):
    """Generate M-dimension tuning buckets for MHC pre_mapping.

    Buckets: 1, 2, 4, 8, 16, 32, 64, 128, 256, 384, 512, 768, 1024, ...
    Small M uses powers-of-2 for fine granularity; large M uses 128 steps.
    """
    buckets = (1, 2, 4, 8, 16, 32, 64, 128)
    if x >= 128:
        x = min(x, 8192)
        x = max(x, 1024)
        buckets += tuple(range(256, x + 1, 128))
    return buckets


def _mhc_map_to_tuning_bucket(x: int) -> int:
    """Map an inference-time M to the nearest tuning bucket (round up)."""
    if x <= 128:
        v = 1
        while v < x:
            v *= 2
        return min(v, 128)
    return ((x + 127) // 128) * 128


_FMA_TILE_N_OPTIONS = (1, 2, 3, 4, 6, 8, 12, 24)
_BIGFUSE_BLOCK_SIZE_OPTIONS = (128, 256, 512)

# Tactic is a tuple: (backend, tile_n, tile_m, bigfuse_bs)
#   backend: "fma", "dg_splitk", "dg_nosplit"
#   tile_n:  FMA N-tile (only used for "fma" backend, 0 otherwise)
#   tile_m:  FMA M-tile (only used for "fma" backend, 0 otherwise)
#   bigfuse_bs: BigFuse block size {128, 256, 512}
_FALLBACK_TACTIC = ("fma", 0, 0, 0)


class MhcPreMappingRunner(TunableRunner):
    """Profiles the full MHC pre_mapping pipeline (GEMM+sqrsum + big_fuse).

    Tactic format: (backend, tile_n, tile_m, bigfuse_bs)
        backend:    "fma" | "dg_splitk" | "dg_nosplit"
        tile_n:     FMA N-tile size (fma backend only)
        tile_m:     FMA M-tile size (fma backend only; 2 requires tile_n=8)
        bigfuse_bs: BigFuse BLOCK_SIZE {128, 256, 512}

    Fallback tactic (-1): uses auto-select for all parameters.
    """

    tuning_config = TuningConfig(
        dynamic_tensor_specs=(
            DynamicTensorSpec(
                input_idx=0,
                dim_idx=0,
                gen_tuning_buckets=_mhc_gen_tuning_buckets,
                map_to_tuning_buckets=_mhc_map_to_tuning_bucket,
            ),
        ),
        # residual (input[2]) dim 0 = M, same as x (input[0]) dim 0
        constraint_specs=(
            ConstraintSpec(
                input_idx=2,
                dim_idx=0,
                infer_shape=lambda shapes: shapes[0][0],
            ),
        ),
    )

    def __init__(
        self,
        n: int,
        hidden_size: int,
        rms_eps: float,
        hc_pre_eps: float,
        hc_sinkhorn_eps: float,
        hc_post_mult_value: float,
        sinkhorn_repeat: int,
    ):
        self.n = n
        self.hidden_size = hidden_size
        self.rms_eps = rms_eps
        self.hc_pre_eps = hc_pre_eps
        self.hc_sinkhorn_eps = hc_sinkhorn_eps
        self.hc_post_mult_value = hc_post_mult_value
        self.sinkhorn_repeat = sinkhorn_repeat

    def unique_id(self):
        return (self.n, self.hidden_size)

    def get_valid_tactics(
        self, inputs: List[torch.Tensor], profile: OptimizationProfile, **kwargs
    ) -> List[Any]:
        N = inputs[1].shape[0]  # w_t is [N, K]
        valid_tile_n = tuple(tn for tn in _FMA_TILE_N_OPTIONS if N % tn == 0)

        tactics = []
        for tn in valid_tile_n:
            for bs in _BIGFUSE_BLOCK_SIZE_OPTIONS:
                tactics.append(("fma", tn, 1, bs))
        if 8 in valid_tile_n:
            for bs in _BIGFUSE_BLOCK_SIZE_OPTIONS:
                tactics.append(("fma", 8, 2, bs))

        if _get_dg_fn() is not None:
            for bs in _BIGFUSE_BLOCK_SIZE_OPTIONS:
                tactics.append(("dg_splitk", 0, 0, bs))
                tactics.append(("dg_nosplit", 0, 0, bs))

        return tactics

    def forward(self, inputs: List[torch.Tensor], *, tactic: Any = -1, **kwargs) -> Any:
        x, w_t, residual, hc_scale, hc_base = inputs

        residual = residual.contiguous()
        hc_scale = hc_scale.to(torch.float32).contiguous()
        hc_base = hc_base.to(torch.float32).contiguous()

        M, K = x.shape
        N = w_t.shape[0]
        n = self.n
        n2 = n * n

        if tactic == -1:
            tactic = _FALLBACK_TACTIC
        backend, tile_n, tile_m, bigfuse_bs = tactic

        num_splits = 1
        if backend == "dg_splitk":
            y_acc, r_acc, num_splits = mhc_gemm_rms_dg_cuda(
                x, w_t, M, N, K, num_splits=_DG_NUM_SPLITS
            )
        elif backend == "dg_nosplit":
            y_acc, r_acc, num_splits = mhc_gemm_rms_dg_cuda(x, w_t, M, N, K, num_splits=1)
        else:
            y_acc, r_acc = mhc_gemm_rms_fma_cuda(
                x, None, M, N, K, w_t=w_t, tile_n=tile_n, tile_m=tile_m
            )

        residual_3d = residual.view(M, n, self.hidden_size)

        post_mix = torch.empty((M, n), dtype=torch.float32, device=x.device)
        comb_mix = torch.empty((M, n2), dtype=torch.float32, device=x.device)
        layer_input = torch.empty((M, self.hidden_size), dtype=torch.bfloat16, device=x.device)

        mhc_big_fuse_cuda(
            y_acc.contiguous(),
            r_acc.contiguous(),
            residual_3d.contiguous(),
            hc_scale,
            hc_base,
            post_mix,
            comb_mix,
            layer_input,
            M,
            K,
            self.hidden_size,
            self.rms_eps,
            self.hc_pre_eps,
            self.hc_sinkhorn_eps,
            self.hc_post_mult_value,
            self.sinkhorn_repeat,
            num_splits=num_splits,
            block_size=bigfuse_bs,
        )

        return post_mix, comb_mix, layer_input


# ---------------------------------------------------------------------------
# Python API — high-level (drop-in for mhc_pre_mapping_fused)
# ---------------------------------------------------------------------------


# Process-wide pre_mapping runner cache keyed on mHC config.
_pre_mapping_runner_cache: dict = {}


def _get_pre_mapping_runner(
    n: int,
    hidden_size: int,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
) -> "MhcPreMappingRunner":
    key = (
        n,
        hidden_size,
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_mult_value,
        sinkhorn_repeat,
    )
    runner = _pre_mapping_runner_cache.get(key)
    if runner is None:
        runner = MhcPreMappingRunner(
            n=n,
            hidden_size=hidden_size,
            rms_eps=rms_eps,
            hc_pre_eps=hc_pre_eps,
            hc_sinkhorn_eps=hc_sinkhorn_eps,
            hc_post_mult_value=hc_post_mult_value,
            sinkhorn_repeat=sinkhorn_repeat,
        )
        _pre_mapping_runner_cache[key] = runner
    return runner


def mhc_pre_mapping_fused(
    x: torch.Tensor,
    w_t: torch.Tensor,
    residual: torch.Tensor,
    n: int,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    hidden_size: int,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
):
    """Full pre-mapping pipeline: GEMM+sqrsum -> big_fuse.

    Backend selection is handled by the autotuner at warmup.
    Falls back to FMA when DeepGEMM is unavailable or cache misses.

    Args:
        w_t: [N, K] float32 weight (row-major, pre-transposed).
    """
    runner = _get_pre_mapping_runner(
        n=n,
        hidden_size=hidden_size,
        rms_eps=rms_eps,
        hc_pre_eps=hc_pre_eps,
        hc_sinkhorn_eps=hc_sinkhorn_eps,
        hc_post_mult_value=hc_post_mult_value,
        sinkhorn_repeat=sinkhorn_repeat,
    )

    tuner = AutoTuner.get()
    _, best_tactic = tuner.choose_one(
        "trtllm::mhc_pre_mapping",
        [runner],
        MhcPreMappingRunner.tuning_config,
        [x, w_t, residual, hc_scale, hc_base],
    )

    return runner(
        inputs=[x, w_t, residual, hc_scale, hc_base],
        tactic=best_tactic,
    )


# ---------------------------------------------------------------------------
# Python API — post_mapping
# ---------------------------------------------------------------------------


def mhc_post_mapping_cuda(
    residual: torch.Tensor,
    x: torch.Tensor,
    post_mix: torch.Tensor,
    comb_mix: torch.Tensor,
    n: int,
) -> torch.Tensor:
    """Post-mapping: out = post * x + comb.T @ residual.

    Args:
        residual: [B, n, hidden_size] bf16
        x:        [B, hidden_size]    bf16
        post_mix: [B, n]             fp32
        comb_mix: [B, n, n]          fp32
        n:        number of hyper-connection heads

    Returns: [B, n, hidden_size] bf16.
    """
    residual = residual.contiguous()
    x = x.contiguous()
    post_mix = post_mix.to(torch.float32).contiguous()
    comb_mix = comb_mix.to(torch.float32).contiguous()

    B = residual.shape[0]
    hidden_size = residual.shape[2]

    out = torch.empty((B, n, hidden_size), dtype=torch.bfloat16, device=x.device)

    torch.ops.trtllm.mhc_post_mapping(
        residual,
        x,
        post_mix,
        comb_mix,
        out,
        B,
        hidden_size,
    )

    return out


# ---------------------------------------------------------------------------
# Python API — fused_hc (post_mapping(prev) + pre_mapping(cur))
# ---------------------------------------------------------------------------


_FUSED_HC_BACKEND_CODE = {
    "fused_half_mma": 0,  # 2-kernel tf32 tcgen05 (fused pmap_gemm + bigfuse)
    "fused_half_fma": 1,  # 2-kernel fp32 FMA ksplit (fused pmap_gemm + bigfuse)
    "fused_all_mma": 2,  # 1-kernel tf32 tcgen05 all-in-one (Path D)
    "fused_all_fma": 3,  # 1-kernel fp32 FMA all-in-one    (Path F)
}

# The SM100/tcgen05 MMA fused-HC C++ kernels are statically instantiated.
# FMA fused-HC paths use runtime hidden_size, but MMA paths must be explicitly
# compiled for each supported hidden size.
_FUSED_HC_MMA_SUPPORTED_HIDDEN_SIZES = {4096, 7168}


def _fused_hc_mma_ks_supported(hidden_size: int, ks: int) -> bool:
    if hidden_size not in _FUSED_HC_MMA_SUPPORTED_HIDDEN_SIZES:
        return False

    block_k = 64
    block_m = 64
    num_warps = 8
    warp_size = 32
    bf16_vec = 8

    if hidden_size % block_k != 0:
        return False
    h_tiles = hidden_size // block_k
    if h_tiles % ks != 0:
        return False

    toks_per_cta = (block_m + ks - 1) // ks
    warps_per_tok = num_warps // toks_per_cta if num_warps > toks_per_cta else 1
    return hidden_size % (warps_per_tok * warp_size * bf16_vec) == 0


# Tactics supported by the half-fused FMA path in `mhcFusedHcFmaLaunch`
# (must stay in sync with the C++ pickFhcFma() table).
_FUSED_HC_HALF_FMA_TN_KS = (
    (1, 1),
    (1, 2),
    (1, 4),
    (1, 8),
    (2, 1),
    (2, 2),
    (2, 4),
    (2, 8),
    (3, 1),
    (3, 2),
    (3, 4),
    (4, 1),
    (4, 2),
    (6, 1),
    (8, 1),
    (12, 1),
    (24, 1),
)
# Tactics for the half-fused MMA path: (num_k_splits,). Matches Path D
# (pickFhcAllInOne) so the autotuner can compare half-fused vs all-in-one at
# the same ks across the full range.
_FUSED_HC_HALF_MMA_KS = (1, 2, 4, 8, 16, 32, 64)
# Tactics for Path D (all-in-one MMA): (num_k_splits,). No bigfuse_bs — the
# bigfuse runs inline inside the single kernel and uses fixed parameters.
_FUSED_HC_ALL_MMA_KS = (1, 2, 4, 8, 16, 32, 64)
# Tactics for Path F (all-in-one FMA): (tile_n, num_k_splits, tile_m).
# Must stay in sync with the C++ pickFhcFmaAllInOne() table.
_FUSED_HC_ALL_FMA_TN_KS_TM = tuple(
    (tn, ks, tm)
    for tm in (1, 2, 4)
    for tn, ks in (
        (1, 1),
        (1, 2),
        (2, 1),
        (2, 2),
        (3, 1),
        (4, 1),
        (6, 1),
        (8, 1),
        (12, 1),
        (24, 1),
    )
)
# Shared BigFuse block-size options (same as pre_mapping autotuner).
_FUSED_HC_BIGFUSE_BS = _BIGFUSE_BLOCK_SIZE_OPTIONS


def _fused_hc_call(
    backend_code: int,
    tile_n: int,
    num_k_splits: int,
    bigfuse_bs: int,
    tile_m: int,
    x_prev,
    residual_prev,
    post_mix_prev,
    comb_mix_prev,
    w_t_cur,
    hc_scale_cur,
    hc_base_cur,
    residual_cur,
    post_mix_cur,
    comb_mix_cur,
    layer_input_cur,
    y_acc_ws,
    r_acc_ws,
    done_counter_ws,
    B: int,
    hidden_size: int,
    n: int,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
):
    torch.ops.trtllm.mhc_fused_hc(
        x_prev,
        residual_prev,
        post_mix_prev,
        comb_mix_prev,
        w_t_cur,
        hc_scale_cur,
        hc_base_cur,
        residual_cur,
        post_mix_cur,
        comb_mix_cur,
        layer_input_cur,
        y_acc_ws,
        r_acc_ws,
        done_counter_ws,
        B,
        hidden_size,
        n,
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_mult_value,
        sinkhorn_repeat,
        backend_code,
        tile_n,
        num_k_splits,
        bigfuse_bs,
        tile_m,
    )


class _FusedHcWorkspaceCache:
    """Size-keyed bounded LRU for the 4 outputs + 3 workspaces of mhc_fused_hc.

    The 4 outputs are consumed by the caller (so they can't alias across
    calls at different B), but repeatedly calling ``torch.empty`` for each
    call inside a CUDA-graph-captured inference loop is wasteful. Keyed on
    ``(B, ws_ks, tile_m, device)``: same-shape calls reuse the previously
    allocated buffers; tensors are stable across graph captures (same ptr
    under the torch allocator's retained block).

    Two bounds keep this from ballooning:

    1. ``_CACHE_MAX_B``: a per-call threshold. Only cache when the request's
       residual_cur footprint is modest (B * n * hidden * 2 bytes; e.g. at
       n=4 hidden=4096 that's 32 KB/token, so a 256-token cap = 8 MB/entry).
       Prefill shapes (B in the tens of thousands) flow straight through to
       ``torch.empty``, which the torch caching allocator already keeps
       cheap on repeated calls.
    2. ``_maxsize``: LRU cap on the number of cached entries. Covers the
       discrete CUDA-graph decode batch sizes plus a few stragglers; decode
       is the only regime that actually benefits from the cache.

    Without these bounds every distinct prefill B leaks ~B * n * hidden * 2
    bytes (≈1.3 GB at B=32768, n=4, hidden=4096). Under a prefill ramp-up
    admitting one new ctx request per iter, the leak reaches tens of GB per
    rank within a dozen iters and blows past HBM.
    """

    __slots__ = ("n", "hidden_size", "_cache", "_maxsize")

    # Up to 48 distinct entries — covers the 35 CUDA-graph decode buckets
    # plus headroom. Each entry at B<=256 is under ~10 MB.
    DEFAULT_MAXSIZE = 48
    # Skip the cache above this B; prefill rides the torch allocator.
    _CACHE_MAX_B = 256

    def __init__(self, n: int, hidden_size: int, maxsize: int = DEFAULT_MAXSIZE):
        self.n = n
        self.hidden_size = hidden_size
        self._maxsize = maxsize
        from collections import OrderedDict

        self._cache: "OrderedDict" = OrderedDict()

    def get(self, B: int, num_k_splits: int, tile_m: int, device):
        n = self.n
        hidden_size = self.hidden_size
        ws_ks = max(1, num_k_splits)
        tm = max(1, tile_m)
        m_batches = (B + tm - 1) // tm
        n2 = n * n
        shape_n = n * (2 + n)

        def _alloc():
            residual_cur = torch.empty((B, n, hidden_size), dtype=torch.bfloat16, device=device)
            post_mix_cur = torch.empty((B, n), dtype=torch.float32, device=device)
            comb_mix_cur = torch.empty((B, n2), dtype=torch.float32, device=device)
            layer_input_cur = torch.empty((B, hidden_size), dtype=torch.bfloat16, device=device)
            if ws_ks == 1:
                y_acc_ws = torch.empty((B, shape_n), dtype=torch.float32, device=device)
                r_acc_ws = torch.empty((B,), dtype=torch.float32, device=device)
            else:
                y_acc_ws = torch.empty((ws_ks, B, shape_n), dtype=torch.float32, device=device)
                r_acc_ws = torch.empty((ws_ks, B), dtype=torch.float32, device=device)
            done_counter_ws = torch.empty((m_batches,), dtype=torch.int32, device=device)
            return (
                residual_cur,
                post_mix_cur,
                comb_mix_cur,
                layer_input_cur,
                y_acc_ws,
                r_acc_ws,
                done_counter_ws,
            )

        if B > self._CACHE_MAX_B:
            return _alloc()

        key = (B, ws_ks, m_batches, device)
        hit = self._cache.get(key)
        if hit is not None:
            self._cache.move_to_end(key)
            return hit
        entry = _alloc()
        self._cache[key] = entry
        if len(self._cache) > self._maxsize:
            self._cache.popitem(last=False)
        return entry


def _alloc_fused_hc_outputs(
    B: int, n: int, hidden_size: int, num_k_splits: int, tile_m: int, device
):
    """Uncached fallback (kept for API compatibility)."""
    return _FusedHcWorkspaceCache(n=n, hidden_size=hidden_size).get(B, num_k_splits, tile_m, device)


# Fallback tactic: backend, tile_n, num_k_splits, bigfuse_bs, tile_m.
#
# The MMA (tcgen05) default delegates ks/bs to the C++ heuristic (pickKSplits
# + selectBigFuseBS). That backend requires SM100+, so on pre-SM100 GPUs we
# fall back to the half-fused FMA path with an explicit ks=1 tactic that is
# valid everywhere.
_FUSED_HC_FALLBACK_TACTIC_MMA = ("fused_half_mma", 0, 0, 0, 1)
_FUSED_HC_FALLBACK_TACTIC_FMA = ("fused_half_fma", 2, 1, 256, 1)


def _get_fused_hc_fallback_tactic(hidden_size: int | None = None):
    mma_ok = _fused_hc_mma_supported()
    if hidden_size is not None:
        mma_ok = mma_ok and hidden_size in _FUSED_HC_MMA_SUPPORTED_HIDDEN_SIZES
    return _FUSED_HC_FALLBACK_TACTIC_MMA if mma_ok else _FUSED_HC_FALLBACK_TACTIC_FMA


class MhcFusedHcRunner(TunableRunner):
    """Profiles the full mhc_fused_hc pipeline (pmap+GEMM+bigfuse + residual out).

    Tactic format: (backend, tile_n, num_k_splits, bigfuse_bs, tile_m)
        backend:       "fused_half_mma" | "fused_half_fma" |
                       "fused_all_mma"  | "fused_all_fma"
        tile_n:        FMA N-tile size (*_fma only; 0 for *_mma)
        num_k_splits:  HIDDEN-axis split (all backends)
        bigfuse_bs:    BigFuse CTA BLOCK_SIZE {128, 256, 512} (fused_half_* only;
                       fused_all_* runs bigfuse inline so this is 0)
        tile_m:        M tokens per CTA (fused_all_fma only; 1 otherwise)

    Backend selection strategy (M-bucketed):
        M <= 32:   prefer "fused_all_fma" (Path F) — single-kernel FMA wins
                   when MMA can't saturate its pipe.
        M >= 64:   prefer "fused_all_mma" (Path D) — single-kernel TF32 MMA
                   wins once M fills at least one BLOCK_M=64 tile.
        In-between: autotuner chooses between all four backends.

    Fallback (-1): delegates to the C++ heuristic (fused_half_mma + auto ks
    + auto bs).
    """

    tuning_config = TuningConfig(
        dynamic_tensor_specs=(
            DynamicTensorSpec(
                input_idx=0,  # x_prev [B, hidden]
                dim_idx=0,
                gen_tuning_buckets=_mhc_gen_tuning_buckets,
                map_to_tuning_buckets=_mhc_map_to_tuning_bucket,
            ),
        ),
        constraint_specs=(
            # residual_prev (input[1]) dim 0 = M
            ConstraintSpec(input_idx=1, dim_idx=0, infer_shape=lambda shapes: shapes[0][0]),
            # post_mix_prev (input[2]) dim 0 = M
            ConstraintSpec(input_idx=2, dim_idx=0, infer_shape=lambda shapes: shapes[0][0]),
            # comb_mix_prev (input[3]) dim 0 = M
            ConstraintSpec(input_idx=3, dim_idx=0, infer_shape=lambda shapes: shapes[0][0]),
        ),
    )

    def __init__(
        self,
        n: int,
        hidden_size: int,
        rms_eps: float,
        hc_pre_eps: float,
        hc_sinkhorn_eps: float,
        hc_post_mult_value: float,
        sinkhorn_repeat: int,
    ):
        self.n = n
        self.hidden_size = hidden_size
        self.rms_eps = rms_eps
        self.hc_pre_eps = hc_pre_eps
        self.hc_sinkhorn_eps = hc_sinkhorn_eps
        self.hc_post_mult_value = hc_post_mult_value
        self.sinkhorn_repeat = sinkhorn_repeat
        self._ws_cache = _FusedHcWorkspaceCache(n=n, hidden_size=hidden_size)

    def unique_id(self):
        return (self.n, self.hidden_size)

    def get_valid_tactics(self, inputs, profile: OptimizationProfile, **kwargs):
        M = inputs[0].shape[0]
        tactics = []
        # The MMA (tcgen05) paths require SM100+. On older archs only the FMA
        # paths are compilable/runnable — we simply never emit MMA tactics.
        mma_ks = tuple(
            ks for ks in _FUSED_HC_HALF_MMA_KS if _fused_hc_mma_ks_supported(self.hidden_size, ks)
        )
        mma_ok = _fused_hc_mma_supported() and bool(mma_ks)
        # Path F (fused_all_fma, 1-kernel FMA) — preferred at small M (<=32)
        # where MMA can't fill BLOCK_M=64. Include for M <= 64 as the
        # crossover is measured per-M.
        if M <= 64:
            for tn, ks, tm in _FUSED_HC_ALL_FMA_TN_KS_TM:
                # Skip grids that wildly oversubscribe SMs.
                m_batches = (M + tm - 1) // tm
                if m_batches * (self.n * (2 + self.n) // tn) * ks > 148 * 4:
                    continue
                # fused_all_fma runs bigfuse inline — no bigfuse_bs tactic axis.
                tactics.append(("fused_all_fma", tn, ks, 0, tm))
        # Path D (fused_all_mma, 1-kernel TF32 MMA) — preferred at mid/large
        # M (>=64). Include when M >= 48 to overlap with Path F at the
        # crossover boundary.
        if mma_ok and M >= 48:
            for ks in mma_ks:
                m_tiles = (M + 63) // 64
                if m_tiles * ks > 148 * 4:
                    continue
                tactics.append(("fused_all_mma", 0, ks, 0, 1))
        # Half-fused FMA path (2-kernel) — useful at smallish M; kept as a
        # fallback option for the autotuner at small M.
        if M <= 512:
            for tn, ks in _FUSED_HC_HALF_FMA_TN_KS:
                if ks > 1 and M * (self.n * (2 + self.n) // tn) >= 148 * 2:
                    continue
                for bs in _FUSED_HC_BIGFUSE_BS:
                    tactics.append(("fused_half_fma", tn, ks, bs, 1))
        # Half-fused MMA path (2-kernel) — always an option when tcgen05 is
        # available.
        if mma_ok:
            for ks in mma_ks:
                m_tiles = (M + 63) // 64
                if m_tiles * ks > 148 * 4:
                    continue
                for bs in _FUSED_HC_BIGFUSE_BS:
                    tactics.append(("fused_half_mma", 0, ks, bs, 1))
        return tactics

    def forward(self, inputs, *, tactic=-1, **kwargs):
        (
            x_prev,
            residual_prev,
            post_mix_prev,
            comb_mix_prev,
            w_t_cur,
            hc_scale_cur,
            hc_base_cur,
        ) = inputs

        x_prev = x_prev.contiguous()
        residual_prev = residual_prev.contiguous()
        post_mix_prev = post_mix_prev.to(torch.float32).contiguous()
        comb_mix_prev = comb_mix_prev.to(torch.float32).contiguous()
        w_t_cur = w_t_cur.to(torch.float32).contiguous()
        hc_scale_cur = hc_scale_cur.to(torch.float32).contiguous()
        hc_base_cur = hc_base_cur.to(torch.float32).contiguous()

        if tactic == -1:
            tactic = _get_fused_hc_fallback_tactic(self.hidden_size)
        backend, tile_n, num_k_splits, bigfuse_bs, tile_m = tactic
        backend_code = _FUSED_HC_BACKEND_CODE[backend]

        B = residual_prev.shape[0]
        (
            residual_cur,
            post_mix_cur,
            comb_mix_cur,
            layer_input_cur,
            y_acc_ws,
            r_acc_ws,
            done_counter_ws,
        ) = self._ws_cache.get(B, num_k_splits, tile_m, x_prev.device)

        _fused_hc_call(
            backend_code,
            tile_n,
            num_k_splits,
            bigfuse_bs,
            tile_m,
            x_prev,
            residual_prev,
            post_mix_prev,
            comb_mix_prev,
            w_t_cur,
            hc_scale_cur,
            hc_base_cur,
            residual_cur,
            post_mix_cur,
            comb_mix_cur,
            layer_input_cur,
            y_acc_ws,
            r_acc_ws,
            done_counter_ws,
            B,
            self.hidden_size,
            self.n,
            self.rms_eps,
            self.hc_pre_eps,
            self.hc_sinkhorn_eps,
            self.hc_post_mult_value,
            self.sinkhorn_repeat,
        )
        return residual_cur, post_mix_cur, comb_mix_cur, layer_input_cur


# Process-wide runner cache keyed on the mHC configuration. Avoids recreating
# a MhcFusedHcRunner (and its workspace cache) on every call, which would also
# defeat the workspace cache inside the runner.
_fused_hc_runner_cache: dict = {}


def _get_fused_hc_runner(
    n: int,
    hidden_size: int,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
) -> "MhcFusedHcRunner":
    key = (
        n,
        hidden_size,
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_mult_value,
        sinkhorn_repeat,
    )
    runner = _fused_hc_runner_cache.get(key)
    if runner is None:
        runner = MhcFusedHcRunner(
            n=n,
            hidden_size=hidden_size,
            rms_eps=rms_eps,
            hc_pre_eps=hc_pre_eps,
            hc_sinkhorn_eps=hc_sinkhorn_eps,
            hc_post_mult_value=hc_post_mult_value,
            sinkhorn_repeat=sinkhorn_repeat,
        )
        _fused_hc_runner_cache[key] = runner
    return runner


def mhc_fused_hc(
    x_prev: torch.Tensor,
    residual_prev: torch.Tensor,
    post_mix_prev: torch.Tensor,
    comb_mix_prev: torch.Tensor,
    w_t_cur: torch.Tensor,
    hc_scale_cur: torch.Tensor,
    hc_base_cur: torch.Tensor,
    n: int,
    hidden_size: int,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
):
    """Fuse the previous block's post_mapping with the current block's pre_mapping.

    The autotuner chooses between four backends:
      * "fused_half_mma" — 2-kernel tcgen05 TF32 pmap+GEMM atomic + bigfuse.
      * "fused_half_fma" — 2-kernel pmap inline + FMA GEMM + sqrsum + bigfuse.
      * "fused_all_mma"  — 1-kernel TF32 tcgen05 all-in-one (Path D).
      * "fused_all_fma"  — 1-kernel FMA all-in-one (Path F).

    Returns:
        residual_cur:      [B, n, hidden] bf16 (new residual, input to the next post_mapping)
        post_mix_cur:      [B, n]         fp32
        comb_mix_cur:      [B, n*n]       fp32
        layer_input_cur:   [B, hidden]    bf16 (input to this block's attn/MoE)
    """
    runner = _get_fused_hc_runner(
        n=n,
        hidden_size=hidden_size,
        rms_eps=rms_eps,
        hc_pre_eps=hc_pre_eps,
        hc_sinkhorn_eps=hc_sinkhorn_eps,
        hc_post_mult_value=hc_post_mult_value,
        sinkhorn_repeat=sinkhorn_repeat,
    )

    tuner = AutoTuner.get()
    _, best_tactic = tuner.choose_one(
        "trtllm::mhc_fused_hc",
        [runner],
        MhcFusedHcRunner.tuning_config,
        [x_prev, residual_prev, post_mix_prev, comb_mix_prev, w_t_cur, hc_scale_cur, hc_base_cur],
    )

    return runner(
        inputs=[
            x_prev,
            residual_prev,
            post_mix_prev,
            comb_mix_prev,
            w_t_cur,
            hc_scale_cur,
            hc_base_cur,
        ],
        tactic=best_tactic,
    )


# ---------------------------------------------------------------------------
# Python API — HCHead
# ---------------------------------------------------------------------------


def mhc_hc_head_cuda(
    x: torch.Tensor,
    fn: torch.Tensor,
    scale: torch.Tensor,
    base: torch.Tensor,
    mult: int,
    hidden_size: int,
    norm_eps: float = 1e-5,
    eps: float = 1e-5,
) -> torch.Tensor:
    """HCHead forward: RMS-normed GEMM -> sigmoid -> weighted sum.

    Args:
        x:           [M, mult, hidden_size] bf16 input
        fn:          [mult, K] fp32 weight (K = mult * hidden_size)
        scale:       [1] fp32
        base:        [mult] fp32
        mult:        number of hyper-connection heads (typically 4)
        hidden_size: per-head dimension
        norm_eps:    RMS norm epsilon
        eps:         sigmoid offset epsilon

    Returns: [M, hidden_size] bf16
    """
    M = x.shape[0]
    K = mult * hidden_size

    x_flat = x.reshape(M, K).contiguous()
    fn_t = fn.to(torch.float32).contiguous()
    scale = scale.to(torch.float32).contiguous()
    base = base.to(torch.float32).contiguous()

    mixes, sqrsum = mhc_gemm_rms_fma_cuda(
        x_flat,
        None,
        M,
        mult,
        K,
        w_t=fn_t,
    )

    out = torch.empty((M, hidden_size), dtype=torch.bfloat16, device=x.device)

    torch.ops.trtllm.mhc_hc_head_apply(
        mixes,
        sqrsum,
        x.reshape(M, mult, hidden_size).contiguous(),
        out,
        scale,
        base,
        M,
        mult,
        hidden_size,
        K,
        float(norm_eps),
        float(eps),
    )

    return out
