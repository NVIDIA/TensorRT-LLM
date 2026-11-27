<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->
# Manual Beam qualification

These checkout-only integration suites are run explicitly with pytest. They are
not scheduled in L0, nightly or LR CI. Lightweight component/operator tests stay
under `tests/unittest/_torch/modeling_v2/models/beam`; the registered model's
named accuracy gate stays under `tests/integration/defs/accuracy/modeling_v2/beam`.

The TP1 helpers here compose the small components from the unit-test tree. They
are not serving APIs or a bitwise reference for TP4. Use `LLM()` or `trtllm-serve`
for serving. Tests retain their own numerical contracts and optional source-capture
requirements; unavailable optional captures are reported as skips.

Run from the repository root with a matching installed/native TRT-LLM build.
The single-GPU suites require SM103 and generated weights, not a checkpoint:

| Suite | Approximate free device memory | Command |
| --- | --- | --- |
| MoE | 60 GiB | `python -m pytest tests/integration/defs/beam/_components/moe -v` |
| Sparse block | 60 GiB | `python -m pytest tests/integration/defs/beam/_components/sparse_block -v` |
| Six-layer prefill | 200 GiB | `python -m pytest tests/integration/defs/beam/_components/six_layer_model/test_small_model_prefill.py -v` |
| Cached generation | 130 GiB | `python -m pytest tests/integration/defs/beam/_components/six_layer_model/test_cached_generation.py -v` |

Run the distributed suite separately with four ranks and four visible GPUs on
one node. It uses generated tensors rather than a model checkpoint; a two-minute
outer timeout bounds collective failures and process startup:

```bash
env -u TLLM_DISABLE_MPI timeout 120s mpirun -n 4 python -m pytest -q \
    tests/integration/defs/beam/test_distributed_loading.py
```

For optional QA reports, pass an explicit suite to the existing wrapper:

```bash
python tests/scripts/beam_component_qa.py --output /tmp/beam-moe-qa -- \
    tests/integration/defs/beam/_components/moe -v
```

Use a new report directory outside the repository. Reports contain local runtime
and scheduler metadata. Inspect skipped cases and call/teardown results rather
than treating successful collection or setup as qualification.
