<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# H3 static FP8 fusion on Rubin

Opt-in single-GPU implementation for the provided uniform ModelOpt FP8 checkpoint.
It uses FP8 QKV/P, trtllm-gen FP16 softmax with `t=0`, BF16 residuals, and FP32
output heads. Video VAE FP8 Linears/component graphs follow Kanghwan's scenario.

The integration base is Kanghwan's `feat/minimax-h3-rubin` at
`0c56fefb9749582e7d532ff26681a10fd5717ea6`. The full fused implementation requires
SM107 with 212 SMs, Ulysses=TP=1, compile on, whole-model CUDA graphs off, and the
fixed 1344×768×124, 24 fps requests. Both fixtures use 50 steps / 49 Transformer calls.
Grace GB300 and four-GPU measurements use the separately documented common recipe.

## Run

Use a TensorRT-LLM runtime whose H3 Python files match the integration base.
The entry checks the forward source before loading weights. A stock wheel with
different H3 Python files is not qualified by this source package.

```bash
python scripts/experiments/minimax_h3_fp8_rubin/run_benchmark.py \
  --model /path/to/MiniMax-H3-FP8 --scene waiter \
  --output /path/to/new-output-directory
```

Use `--scene cycling` for the second fixture. `--validate-only` checks fixture and
source identities on CPU without importing Torch. Output includes one warmup,
three unprofiled timings, actual dispatch counts, output-repeatability state and
the canonical raw RGB/audio archive. Startup compilation/qualification is outside
the timed intervals. Human visual acceptance is a separate result.

The final QK launch layout is selected before model construction. Runtime helpers
preserve the upstream frozen scenario configuration. The output-head clone is
mechanically derived from the current upstream forward, with an inverse AST check;
the two FP32 heads share one exact input conversion.

## Retained kernels

- AdaLN → FP8 producers and selected framework FP8 GEMMs.
- GateUp GEMM + SwiGLU + FP8 quantization.
- Down GEMM + gated residual.
- Attention-output GEMM + gated residual + FFN AdaLN/FP8 production.
- Conditioning cache, reciprocal preparation, QK/RoPE load/launch choices.
- Dedicated scalar FP8 quantization and shared output-head BF16→FP32 conversion.

Kernel mathematical sources are retained from the research branch. The new-base
configuration/forward integration has its own validation status in `provenance.json`;
old GPU measurements do not automatically qualify a changed runtime.

## CPU checks

```bash
python -m unittest discover \
  -s scripts/experiments/minimax_h3_fp8_rubin/tests -p 'test_*_cpu.py'
```

## Sources and attribution

Kanghwan provided the original FP8/scenario foundation, QK/RoPE work, Attention
backend choices, component graphs and preloop scheduling. Pamela's shared thread
and Chenghao's register-epilogue examples informed GEMM fusion; Yiheng's discussion
informed conditioning precomputation. Kernel Factory supplied candidate scheduling
ideas used in the GateUp implementation. The FP8 adaptation, retained fusion
implementation and measured validation are recorded in the complete reports:

- [中文报告](https://sc.talos.nvidia.com/view/home/scratch.taianz_inf/workspace/notes/rubin/runs/20261110-h3-static-fp8/report.html)
- [English report](https://sc.talos.nvidia.com/view/home/scratch.taianz_inf/workspace/notes/rubin/runs/20261110-h3-static-fp8/report-en.html)
- [Original research branch](https://gitlab-master.nvidia.com/ftp/tekit/-/tree/user/taianz/h3-rubin-fp8-functional)

Historical experiments, rejected variants, weights, videos, traces, archives and
compiled artifacts remain outside this clean source package.
