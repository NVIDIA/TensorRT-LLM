<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# VisualGen component tests

Component tests compare a real-weight model component with a reference computed
in the same run, using identical inputs. They complement kernel correctness
checks and end-to-end media quality tests.

## Methodology

- Choose a component that exercises the model code affected by a feature.
  The Wan 2.2 A14B tests load block 0 of the high-noise expert, together with
  its input embedding, conditioning and output head.
- Generate representative inputs at runtime. The Wan tests use the pipeline's
  default first denoising step: seeded Gaussian latents and checkpoint UMT5
  embeddings of existing repository prompts and the default negative prompt.
  No input or output safetensors fixture is required.
- Choose a reference for the behavior being checked. For example, compare the
  BF16 component with HF diffusers in FP32, a runtime feature with eager
  execution, and parallel execution with a single GPU. Quantized variants
  compare with BF16 to measure the feature's loss budget; kernel tests remain
  responsible for arithmetic correctness.
- Verify that the requested feature is active and output shapes match before
  scoring. Compute relative L2 error in FP32 over each complete output tensor:
  `||candidate - reference||_2 / ||reference||_2`.
- Calibrate each feature category's limit against expected numerical variation
  and representative regressions. Keep the limits with the tests. A quantization
  loss budget can hide small defects, so it complements the independent BF16
  reference rather than replacing it.

The Wan tests default to seed 42, following existing Wan regression tests, and
print the seed with each result. Set `TLLM_VISUALGEN_COMPONENT_SEED` to reproduce
another seed or explore additional inputs. References are recomputed each run.

## Running the Wan tests

Provide `Wan2.2-T2V-A14B-Diffusers` under `LLM_MODELS_ROOT` and a built TRT-LLM
environment. The parallel file requires two GPUs.

```bash
cd tests/unittest
LLM_MODELS_ROOT=... pytest -s _torch/visual_gen/test_wan22_t2v_a14b_component.py
LLM_MODELS_ROOT=... pytest -s _torch/visual_gen/multi_gpu/test_wan22_t2v_a14b_component_parallel.py
```
