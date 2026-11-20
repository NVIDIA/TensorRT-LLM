<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# MiniMax-H3 static FP8 on Rubin

This scenario implements the uniform static ModelOpt FP8 DiT on one SM107 GPU
with 212 SMs. It keeps BF16 residuals and the original FP32 media heads, uses
trtllm-gen FP16-softmax Attention with `skip_corr_threshold=0`, and reuses the
existing H3 scenario's FP8 Video VAE, component graphs and preloop scheduling.
The Video VAE is prepared from the model's native decoder; it does not need a
separate FP8 Video VAE checkpoint.

Start the Python process with `NVIDIA_TF32_OVERRIDE=0` and select the existing
H3 scenario resolver. Other TF32 settings are outside this scenario's qualification:

```python
from tensorrt_llm import VisualGen, VisualGenArgs

args = VisualGenArgs(
    model="/path/to/MiniMax-H3-FP8",
    pipeline_config={"visual_gen_scenario": "require"},
    attention_config={
        "backend": "CUDNN",
        "quant_attention_config": {"qk_dtype": "fp8", "v_dtype": "fp8"},
    },
    parallel_config={"ulysses_size": 1, "tp_size": 1, "parallel_vae_size": 1},
    torch_compile_config={"enable": True},
    cuda_graph_config={"enable": False},
    compilation_config={"resolutions": [[768, 1344]], "num_frames": [124]},
    cache_config=None,
)
generator = VisualGen(model=args.model, args=args)
```

`visual_gen_scenario="off"` keeps the family implementation. `"auto"` selects
this scenario only when its hardware, checkpoint and execution requirements
match. No new public argument or separate benchmark command is introduced.

## Scope

- All 200 main DiT Linears must be static per-tensor FP8 with the original H3
  shapes. BF16 or mixed checkpoints, including EarlyTop8, are not admitted.
- The fused block serves batch one and packed lengths 37776, 37820 or 38169,
  including the fixed 1344×768×124 Waiter/Cycling workloads. Other packed lengths
  use the existing family AdaLN and FFN math, complete scenario Attention with
  its output projection, and stock FP8QDQ QKV GEMM. They do not enter the
  scenario's dedicated GEMM kernels. Batch, hidden width and dtype requirements
  remain enforced.
- The default loader warmup uses the prompt `"warmup"`, producing packed length
  37712 at 1344×768×124. It takes this fallback; normal requests at the listed
  lengths select the fused kernels independently. Skipping loader warmup is not
  required. The fallback and fused requests require separate GPU qualification.
- Ulysses, TP and parallel VAE are one. GB300 and four-GPU inference keep their
  separately qualified family routes.
- Compile is on; whole-model CUDA Graphs, LoRA, CPU weight offload and diffusion
  caches are off. Weights and calibration scales remain immutable after loading.
- `NVIDIA_TF32_OVERRIDE=0` is required, including during model loading and warmup.
- Component-local text-encoder/decoder graphs are inherited from the existing
  scenario. The conditioning-keyed DiT cache is outside those graph captures.
- CuTe DSL, CUDA 13.4 tooling and the SM107 runtime are required. Exact conversion
  kernels are compiled and qualified before pipeline compilation and warmup.
  Python assertions must be enabled; conversion preparation rejects `python -O`.
- Initialization uses the model/scales CUDA device for allocations and launches.
  Execution on a nonzero CUDA device still needs independent GPU qualification.

## Implementation

`modeling.py` owns the FP8 attention, block and Transformer subclasses;
`conditioning.py` owns each pipeline's exact timestep-keyed AdaLN tables.
`ops.py` defines the compiler boundaries. `kernels/` contains AdaLN→FP8,
GateUp+SwiGLU→FP8, Down+gated residual, attention-output+gated residual,
QK/RoPE load reuse, static quantization and the shared final BF16→FP32 cast.
No shared Linear, GatedMLP or NVFP4 model class is patched globally.

The FP8 attention passes its RoPE cast explicitly to the shared QK orchestration;
the NVFP4 scenario's default cast and behavior are unchanged. Model loading
prepares packed GateUp weights, reciprocals and conversion factors before
`torch.compile` wraps the blocks.

The conditioning cache can retain approximately 1.2 GiB for 64 distinct
two-timestep keys. Reclaiming these allocations across sleep/wake is unverified
and remains follow-up work; this scenario does not clear a cache concurrently
with inference.

## Validation and attribution

This native scenario integration requires its own GPU functional, LPIPS and
performance qualification. Historical measurements of the separately frozen
implementation do not qualify a changed integration. See the linked complete
reports for exact configurations, video comparisons, accepted/rejected changes
and the current validation status:

- [English report](https://sc.talos.nvidia.com/view/home/scratch.taianz_inf/workspace/notes/rubin/runs/20261110-h3-static-fp8/report-en.html)
- [中文报告](https://sc.talos.nvidia.com/view/home/scratch.taianz_inf/workspace/notes/rubin/runs/20261110-h3-static-fp8/report.html)

Kanghwan supplied the H3/scenario foundation, QK/RoPE work, Attention backend
choices and component/preloop work. Pamela's discussion and Chenghao's register
epilogue examples informed GEMM fusion; Yiheng's discussion informed conditioning
precomputation. Kernel Factory supplied scheduling ideas used by the GateUp
kernel. The FP8 adaptations and retained fusion implementation are included here;
search campaigns, rejected variants and experimental measurement harnesses are
maintained with the local report evidence.
