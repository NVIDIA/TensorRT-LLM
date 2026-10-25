# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""msa-prewarm: compile every MSA kernel variant serving can select during the decoder warmup.

Modules (none imports torch, triton or the MSA library at import time):
  hook      entry points DecoderRunner calls (around warmup, and at the start of its MSA indexer prewarm)
  core      fmha_sm100 variant enumeration (the library's own dispatch functions), parallel compile, and the
            serve-time observer that logs every inline fmha_sm100 compile after warmup
  sdwarm    sparse-decode prewarm: the Triton sparse decode is warmed for every local batch, decode query length
            and pointer-alignment signature an eager step can reach, at the end of warmup, plus a serve-time
            observer that logs every Triton compile after warmup
  selftest  coverage check on one GPU: real fmha_sm100 proxy plans for every q, variant lookups recorded, nothing
            may be missing

Switches (each defaults to on; 0/false/off/no turns it off): M3AX_STACK_A (master) and M3AX_MSA_PREWARM_VARIANTS
gate everything; M3AX_SPARSE_DECODE_PREWARM gates sdwarm only. Every part engages only on a MiniMax-M3 MSA runner
and never raises: on any failure the stock lazy JIT applies.
"""

VERSION = "msa-prewarm-1+stack-a"
