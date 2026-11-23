<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Beam components with ModelingV2

This directory contains Beam family operators and a directly constructed 52-layer
NVFP4-expert component. It does not register an engine target: `LLM()` and
`trtllm-serve` cannot select these components through ModelingV2. Component users
construct `BeamFullModelNvfp4Sm103Tp4` from `full_model_nvfp4__sm_103__tp4/model.py`
and provide the checkpoint configuration, rank mapping and device; the component's
`load()` method loads the caller-provided checkpoint before execution. The tests
below exercise the component APIs and their input contracts.

## Configuration and limits

| Setting | Component configuration |
|---|---|
| Hardware and layout | Four GB300 GPUs (SM103), attention TP4, MoE TP4/EP1, PP1 |
| Parameters | Packed NVFP4 experts; BF16 dense tensors; FP32 router projection and selection |
| Attention | Checkpoint-defined sliding-window and full attention, BF16 KV cache |
| Execution | Eager; CUDA graphs and overlap scheduling disabled |

Checkpoint weights are supplied separately and streamed into rank-local storage.
No checkpoint, reference captures or private evaluation artifacts are included.
Other GPU architectures and parallel layouts, speculative decoding, FP8 KV cache,
disaggregated serving, CUDA graphs and overlap scheduling are outside this baseline.

## File organization

- `ops/` contains Beam-only operators. Their tests mirror that directory under
  `tests/unittest/_torch/modeling_v2/models/beam/ops/`.
- `_tp4/` contains shared BF16 layout, loading and component base modules.
- `full_model_nvfp4__sm_103__tp4/` adds packed expert storage and full-depth execution.
- `tests/unittest/_torch/modeling_v2/models/beam/_components/` contains checkout-only
  TP1 bring-up helpers, including the six-layer model and its generation controller.
  They are not installed in the wheel. Intermediate capture instrumentation lives
  in tests; runtime validation and rank-consistency diagnostics remain in the library.

## Run component tests

Run from the repository root with a matching TensorRT-LLM installation and its
native dependencies. `cpu_only` selects tests that do not execute GPU operations;
it does not make the TensorRT-LLM package independent of its native dependencies.

```bash
python -m pytest -m cpu_only \
  tests/unittest/_torch/modeling_v2/models/beam/test_component_contracts.py \
  tests/unittest/_torch/modeling_v2/models/beam/test_diagnostic_mode.py \
  tests/unittest/_torch/modeling_v2/models/beam/full_model_nvfp4__sm_103__tp4/test_checkpoint_layout.py
```

For generated-input operator coverage on one compatible GPU:

```bash
python -m pytest tests/unittest/_torch/modeling_v2/models/beam/ops
```

For the small four-rank load-error and collective checks, with four GPUs visible
and working MPI peer communication:

```bash
env -u TLLM_DISABLE_MPI mpirun -n 4 python -m pytest -q \
  tests/unittest/_torch/modeling_v2/models/beam/test_distributed_loading.py
```

These checks use generated data, not a checkpoint. The collective test includes
capture/replay of the communication operation; this does not qualify whole-model
CUDA graphs. Component checks do not establish whole-model accuracy.

CPU and bounded single-GPU CI coverage is selected explicitly in
the test lists under `tests/integration/test_lists/test-db/`, including
`l0_b300.yml` for GPU operators.
Large component and four-rank checks are manual; do not run the entire Beam test
directory as a lightweight smoke test. Full GSM8K is not part of CI.
