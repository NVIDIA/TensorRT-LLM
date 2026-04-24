<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# AutoDeploy DeepSeek V4 Ramp-Up

This note is a quick ramp-up for agents working on DeepSeek V4 Flash support in
AutoDeploy. It summarizes the local workflow, what AutoDeploy is doing, what is
reasonable to add or change, and the important DeepSeek V4 model/checkpoint
facts.

For the detailed feature plan, see `deepseek_v4_ad_plan.md`.

For parallel work packets, see `subagents/README.md`.

## Command Rule

Run commands through the project shell setup:

```bash
bash -ic "f9 && <command>"
```

Examples:

```bash
bash -ic "f9 && pytest -q tests/unittest/_torch/auto_deploy/unit"
bash -ic "f9 && python3 -m py_compile path/to/file.py"
bash -ic "f9 && rg -n \"deepseek_v4\" tensorrt_llm/_torch/auto_deploy"
bash -ic "f9 && git status --short"
```

Do not run Python, pytest, or repo commands directly without the `bash -ic
"f9 && ..."` wrapper unless the user explicitly changes this rule.

## Local Checkpoint

The DeepSeek V4 Flash checkpoint is already downloaded here:

```text
/lustre/fs1/portfolios/coreai/projects/coreai_comparch_autodeploy/users/bmarimuthu/dev/hf_home/manual/deepseek-ai__DeepSeek-V4-Flash
```

This directory contains the model config, tokenizer files, safetensors index,
and 46 checkpoint shards. Use this path for local metadata and weight-loading
work instead of assuming the model exists in the default Hugging Face cache.

Useful metadata files:

```text
/lustre/fs1/portfolios/coreai/projects/coreai_comparch_autodeploy/users/bmarimuthu/dev/hf_home/manual/deepseek-ai__DeepSeek-V4-Flash/config.json
/lustre/fs1/portfolios/coreai/projects/coreai_comparch_autodeploy/users/bmarimuthu/dev/hf_home/manual/deepseek-ai__DeepSeek-V4-Flash/model.safetensors.index.json
/lustre/fs1/portfolios/coreai/projects/coreai_comparch_autodeploy/users/bmarimuthu/dev/hf_home/manual/deepseek-ai__DeepSeek-V4-Flash/tokenizer.json
```

Suggested local model path for experiments:

```bash
bash -ic "f9 && python examples/auto_deploy/build_and_run_ad.py --model /lustre/fs1/portfolios/coreai/projects/coreai_comparch_autodeploy/users/bmarimuthu/dev/hf_home/manual/deepseek-ai__DeepSeek-V4-Flash --args.yaml-extra examples/auto_deploy/model_registry/configs/deepseek_v4_flash.yaml"
```

Only use the full run command after the required model/config pieces are ready.
For early work, prefer unit tests with small synthetic tensors or metadata-only
fixtures.

## AutoDeploy Flow: High-Level Mental Model

AutoDeploy takes a Hugging Face-style model and turns it into an optimized
runtime graph for PyTorch backend serving.

The high-level flow is:

1. Load model config and tokenizer metadata.
2. Build a model instance, using an AutoDeploy custom model if registered.
3. Export the model graph.
4. Prefer AutoDeploy canonical ops in the exported graph:
   - `torch_linear_simple`
   - `torch_rmsnorm`
   - `torch_moe`
   - attention/MLA custom ops
   - quantized linear/MoE custom ops
5. Run graph transforms:
   - pattern cleanup
   - quantization transforms
   - MoE and SwiGLU matching
   - sharding and expert parallel transforms
   - cache insertion
   - post-load fusions
6. Load checkpoint weights into the transformed graph.
7. Compile with the selected backend, commonly `torch-cudagraph`.
8. At runtime, `ADExecutor` prepares sequence metadata, cache page tables, and
   padded CUDA graph batch shapes.
9. The runtime executes prefill, mixed prefill/decode, and decode paths using
   AutoDeploy's cache manager and compiled graph segments.

The important idea: model code should be lean and semantic. It should expose
recognizable canonical ops that later transforms can replace with optimized
kernels. Avoid hiding core model behavior inside opaque Python loops if a
custom op or transform will need to recognize it later.

## What Is Allowed To Add Or Change

For DeepSeek V4 support, it is reasonable to add or change:

- AutoDeploy custom model code for `deepseek_v4`.
- Custom AutoDeploy ops:
  - sparse DeepSeek V4 attention source op
  - DeepSeek V4 router op
  - DeepSeek V4 MoE op
  - E8M0/FP8/MXFP4 helper ops if needed
- Quantization readers and transforms.
- Checkpoint load hooks and key remapping.
- FineGrained FP8 linear support for DeepSeek V4 `.scale` tensors.
- Packed MXFP4/FP4 expert loading and layout conversion.
- MoE lowering to Triton, TRT-LLM, FlashInfer, or DeepGEMM-style kernels.
- V4-specific paged cache resource handlers.
- Sequence/cache metadata extensions needed by V4 sparse attention.
- CUDA graph dynamic-op registration and piecewise capture support.
- AutoDeploy model registry YAML and test configs.
- Unit tests, graph tests, and reduced-model integration tests.

Keep changes scoped. Do not refactor unrelated TensorRT-LLM or AutoDeploy code
unless the refactor is required to expose a clean V4 extension point.

## Development Guardrails

- Read `CODING_GUIDELINES.md` before code changes.
- New files need the NVIDIA copyright and Apache-2.0 header.
- Do not guess checkpoint formats. Verify with config, safetensors index, or
  safetensors headers.
- Do not assume generic `fp8` support is enough for DeepSeek V4.
- Do not treat the dense expert fallback as a production path.
- Avoid unpaged long-context V4 caches.
- Keep model code export-friendly.
- Add focused tests with synthetic data before trying full checkpoint runs.
- If committing, use `git commit -s`; do not manually write sign-off lines.

## DeepSeek V4 Model Summary

DeepSeek V4 Flash is a large sparse MoE model with heterogeneous attention.

Key config facts:

| Field | Value |
| - | - |
| Model type | `deepseek_v4` |
| Layers | 43 |
| Hidden size | 4096 |
| Attention heads | 64 |
| KV heads | 1 |
| Head dim | 512 |
| Q LoRA rank | 1024 |
| RoPE head dim | 64 |
| Sliding window | 128 |
| Routed experts | 256 |
| Experts per token | 6 |
| Shared experts | 1 |
| Hash-routed layers | 3 |
| Router scoring | `sqrtsoftplus` |
| Router scale | 1.5 |
| SwiGLU limit | 10.0 |
| Max position embeddings | 1,048,576 |

Attention is not classic DeepSeek V3 MLA. It combines:

- local sliding-window attention
- compressed KV memory
- ratio-4 compressed layers with indexer top-k
- ratio-128 compressed layers
- attention sinks
- inverse RoPE on output
- grouped output projection

The model also uses hyper-connections around attention and MoE blocks.

## Checkpoint Quantization Summary

The checkpoint is mixed precision:

- Attention and output projection linears:
  - weights: `F8_E4M3`
  - scales: `F8_E8M0`
  - runtime target: FineGrained FP8 linear
- Shared expert linears:
  - weights: `F8_E4M3`
  - scales: `F8_E8M0`
  - runtime target: FineGrained FP8 linear or fused shared expert path
- Routed expert linears:
  - weights: packed FP4 stored as `I8`
  - scales: `F8_E8M0`
  - runtime target: MXFP4/FP4 fused MoE
- Router, compressor, embeddings/head, norms, hyper-connections, and attention
  sinks:
  - BF16/F32/I64 as appropriate

Important scale rule:

```text
E8M0 scales are exponent-only.
Use scale.view(torch.uint8) when raw exponent bytes are required.
Do not use scale.to(torch.uint8) for raw-byte preservation.
```

## Recommended Work Order

The current parallel plan is organized into waves:

```text
subagents/wave1
  foundational metadata, E8M0 helpers, router, source attention op, cache
  resources, CUDA graph config

subagents/wave2
  FP8 linears, packed MXFP4 expert loader, attention kernel microfeatures

subagents/wave3
  DeepSeek V4 MoE op

subagents/wave4
  full model integration
```

Files inside a single wave are intended to be worked on in parallel.

## Practical Test Strategy

Prefer this order:

1. CPU metadata tests:
   - checkpoint classifier
   - key naming
   - dtype/shape classification
2. CPU quantization helper tests:
   - E8M0 decode
   - raw-byte preservation
3. Small synthetic graph tests:
   - FineGrained FP8 linears
   - packed MXFP4 expert layout
   - router math
   - sparse attention reference op
4. GPU kernel tests:
   - FP8 linear backend
   - MXFP4 MoE backend
   - attention kernel microfeatures
5. Reduced-layer AutoDeploy run.
6. Full 43-layer load and transform.
7. Serving path with chunked prefill and CUDA graph decode.

Do not start by running the full checkpoint unless the targeted feature already
has small tests passing. Full model runs are expensive and make failures harder
to isolate.
