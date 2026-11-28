# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""ULL entry points and producer-side quantization helpers for the low-M MXFP8 GEMM."""

from .k3_fp8w_lowm_mxfp8 import MAX_M as MAX_M
from .k3_fp8w_lowm_mxfp8 import MAX_TMEM_TILES as MAX_TMEM_TILES
from .k3_fp8w_lowm_mxfp8 import STAGED_SMEM_BYTES as STAGED_SMEM_BYTES
from .k3_fp8w_lowm_mxfp8 import K3Fp8wLowMMxGemmKernel as K3Fp8wLowMMxGemmKernel
from .k3_fp8w_lowm_mxfp8 import K3LowMMxTactic as K3LowMMxTactic
from .k3_fp8w_lowm_mxfp8 import __all__ as __all__
from .k3_fp8w_lowm_mxfp8 import _amax_bf16x8_words as _amax_bf16x8_words
from .k3_fp8w_lowm_mxfp8 import _amax_pair as _amax_pair
from .k3_fp8w_lowm_mxfp8 import _run_k3_fp8w_lowm_gemm_mxfp8
from .k3_fp8w_lowm_mxfp8 import abs_max_bf16x32 as abs_max_bf16x32
from .k3_fp8w_lowm_mxfp8 import amax_bf16x8_words as amax_bf16x8_words
from .k3_fp8w_lowm_mxfp8 import amax_pair as amax_pair
from .k3_fp8w_lowm_mxfp8 import bf16x2_inv_scale as bf16x2_inv_scale
from .k3_fp8w_lowm_mxfp8 import can_run_mxfp8 as can_run_mxfp8
from .k3_fp8w_lowm_mxfp8 import can_stage_weights as can_stage_weights
from .k3_fp8w_lowm_mxfp8 import default_mx_tactic as default_mx_tactic
from .k3_fp8w_lowm_mxfp8 import mxfp8_quantize_reference as mxfp8_quantize_reference
from .k3_fp8w_lowm_mxfp8 import quant_bf16x8_to_e4m3 as quant_bf16x8_to_e4m3
from .k3_fp8w_lowm_mxfp8 import ring_holds_weights as ring_holds_weights
from .k3_fp8w_lowm_mxfp8 import (
    run_k3_fp8w_lowm_gemm_mxfp8_ag_push as run_k3_fp8w_lowm_gemm_mxfp8_ag_push,
)
from .k3_fp8w_lowm_mxfp8 import (
    run_k3_fp8w_lowm_gemm_mxfp8_ar_push as run_k3_fp8w_lowm_gemm_mxfp8_ar_push,
)
from .k3_fp8w_lowm_mxfp8 import (
    run_k3_fp8w_lowm_gemm_mxfp8_slice_ar_push as run_k3_fp8w_lowm_gemm_mxfp8_slice_ar_push,
)
from .k3_fp8w_lowm_mxfp8 import smem_bytes as smem_bytes
from .k3_fp8w_lowm_mxfp8 import staged_matches_mid_m as staged_matches_mid_m
from .k3_fp8w_lowm_mxfp8 import staged_mx_tactic as staged_mx_tactic
from .k3_fp8w_lowm_mxfp8 import tmem_alloc_cols as tmem_alloc_cols
from .k3_fp8w_lowm_mxfp8 import ue8m0_of_amax as ue8m0_of_amax
from .k3_fp8w_lowm_mxfp8 import validate_tactic as validate_tactic
from .k3_fp8w_lowm_mxfp8 import weights_on_chip as weights_on_chip

run_k3_fp8w_lowm_gemm_mxfp8 = _run_k3_fp8w_lowm_gemm_mxfp8
