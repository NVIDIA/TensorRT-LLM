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

The unittest suite covers direct route submission, late route waits, generation
pairing, static/CLC eligibility, cache identity, SM budgets, and launch-queue
policy/worker propagation, and native source/header packaging. The handoff and
autotune checks also cover ownership, rank balance,
deterministic helper-bearing inputs, and unchanged uniform OFF inputs.
`--skip-torch` on the autotune
command reports `partial_no_torch` and omits tensor validation.

GPU checks require a supported SM100-family device, a compatible TensorRT-LLM
build with the native quantizer overloads, and the normal test dependencies:

```bash
python3 -m pytest -q tests/unittest/_torch/thop/parallel/test_quantization_sm_budget.py
```

These cover byte equality, scale padding, invalid budgets, FakeTensor/opcheck,
CUDA Graph, and representative reserved-SM settings.
CPU stand-ins cannot establish GPU completion, kernel accuracy, or performance;
the queue guards detect PyTorch initialization, not every external CUDA context.
These commands do not run full EP8/model accuracy or E2E performance. See the
[integration guide](../../../DYNAMIC_LOAD_BALANCE_SUBMIT_OPT.md) for configuration
and the current validation requirements.
