<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->
# Local bring-up components

These TP1 components are checkout-only test helpers, not supported serving APIs or a
reference for TP4 bitwise results. Their names describe the isolated component; device and geometry contracts are
documented in each helper. They do not register ModelingV2 targets. Use `LLM()` or `trtllm-serve` for serving.

Run tests through pytest with the repository's test import plugin. Existing L0 selectors
retain the small single-GPU tests. The large MoE, sparse-block, six-layer and
four-rank qualification suites live in
[the manual integration tree](../../../../../../integration/defs/beam/README.md).

The dense and sparse helpers call the serving family's shared attention function,
input normalization (`beam_rmsnorm_fp32`), and activation (`swiglu_fp32`); the other
normalizations use the same FlashInfer entries as serving. Their TP1 geometry and
narrower RoPE schema remain distinct from TP4. Compare each helper against its
declared contract, not against the TP4 model as a bitwise reference.

## Manual QA reports

Some tests expose additional comparison metrics with pytest's `record_property` fixture. Ordinary
pytest runs do not write custom evidence files or collect Git/Slurm metadata.
For a manual report, run the opt-in QA script from the repository root:

```bash
python tests/scripts/beam_component_qa.py --output /tmp/beam-router-qa -- \
    tests/unittest/_torch/modeling_v2/models/beam/_components/moe_router -v
```

Use a new output directory for each run. `report.json` records pytest's exit code,
collection/setup/call/teardown outcomes, comparison properties, and checkout/runtime
identity. A successful setup phase does not imply a successful test; inspect the
call and teardown outcomes too. Missing optional source captures remain explicit
skips; configured invalid captures fail. The script neither stages captures nor
changes numerical checks. Large suites still require their documented resources.

This replaces the `BEAM_*_EVIDENCE` environment variables and per-case JSON files;
metadata is collected once per QA run. Dense weight-accounting checks still compare
all tensors directly, without the optional duplicate hashing pass. Keep reports
outside the public repository because they contain local paths and scheduler IDs.
