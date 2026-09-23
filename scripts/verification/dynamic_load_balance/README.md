<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Dynamic load-balance regression checks

Use project Python >=3.10; run from the repository root without `-O`.
The checks use assertions.
The CPU checks execute production methods extracted from this checkout without
importing the TRT-LLM GPU runtime. Install setuptools for the packaging test and
CPU PyTorch for the complete autotune workload check.

```bash
python3 -m unittest discover -s scripts/verification/dynamic_load_balance -p 'test_*.py' -v
python3 scripts/verification/dynamic_load_balance/check_rebalance_owner_handoff.py --output /tmp/rebalance-owner-handoff.json
python3 scripts/verification/dynamic_load_balance/check_on_autotune.py --output /tmp/on-autotune.json
```

The 34 unittest cases cover direct route submission, late route waits, generation
pairing, static/CLC eligibility and cache identity, eight-SM budgets, launch-queue
policy/worker propagation, and native source/header packaging. The handoff check
covers eight ownership contracts; autotune checks 32 gate states plus equal rank
totals, increasing expert counts, unique top-k, deterministic power-law inputs,
helper coverage, and unchanged uniform OFF inputs. `--skip-torch` on the autotune
command reports `partial_no_torch` and omits tensor validation.

GPU checks require Rubin, a compatible TensorRT-LLM build with the new native quantizer
overloads, and the repository's normal test dependencies:

```bash
python3 -m pytest -q tests/unittest/_torch/thop/parallel/test_quantization_sm_budget.py
python3 -m pytest -q tests/unittest/_torch/thop/parallel/test_fp8_block_scale_gemm.py::test_cute_dsl_mxfp8_gemm_rubin_clc_dynamic_prefetch_multi_wave
```

These select 23 quantization cases (byte equality, scale padding, invalid budgets,
FakeTensor/opcheck, CUDA Graph) and two static/CLC numerical cases (`reserved_sms=0/8`).
CPU stand-ins cannot establish GPU completion, kernel accuracy, or performance;
the queue guards detect PyTorch initialization, not every external CUDA context.
These commands do not run full EP8/model accuracy or E2E performance. See the
[integration guide](../../../DYNAMIC_LOAD_BALANCE_SUBMIT_OPT.md) for configuration
and the distinction between current validation requirements and historical tekit results.
