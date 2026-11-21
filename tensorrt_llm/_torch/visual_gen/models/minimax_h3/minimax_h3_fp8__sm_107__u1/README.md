<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# MiniMax-H3 static FP8 on Rubin

This scenario implements the uniform static ModelOpt FP8 DiT on one SM107 GPU
with 212 SMs. It keeps BF16 residuals and the original FP32 media heads, and
runs FP8 Q/K/V through the trtllm-gen attention kernel with FP16 softmax
(`skip_corr_threshold=0`). The `CUDNN` backend with `qk_dtype=v_dtype="fp8"` in
the example below is how the FP8 attention recipe is requested; the scenario
then selects that kernel for it. It reuses the
existing H3 scenario's FP8 Video VAE, component graphs and preloop scheduling.
The Video VAE is prepared from the model's native decoder; it does not need a
separate FP8 Video VAE checkpoint.

Start the Python process with `NVIDIA_TF32_OVERRIDE=0` and select the existing
H3 scenario resolver:

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
this scenario when its hardware, checkpoint and execution requirements match and
otherwise keeps the family implementation; `"require"` raises with the reason.
No new public argument or separate benchmark command is introduced.

## Scope

- All 200 main DiT Linears must be static per-tensor FP8 with the original H3
  shapes. BF16 or mixed checkpoints are not admitted.
- Batch one, any packed sequence length. The fused kernels take the row count at
  run time, so one build of each serves every request shape. The blocks raise on
  another batch size or layout rather than falling back.
- Ulysses, TP and parallel VAE are one. Other GPUs and multi-GPU inference keep
  the family routes.
- Compile is on; whole-model CUDA Graphs, LoRA, CPU weight offload and diffusion
  caches are off. Weights and calibration scales are immutable after loading.
- `NVIDIA_TF32_OVERRIDE=0` is required from process start; the route declines
  otherwise.
- The scenario shares attention and decoder settings with the H3 NVFP4 scenario,
  which are fixed per process. A process that already selected other values for
  them cannot load this scenario; the route declines with that reason.

## Requirements

CuTe DSL and the SM107 runtime are required. The CUDA toolchain is optional:

| Tool | Used for | Without it |
|---|---|---|
| `nvcc`, `ninja` | build the scalar quantize and output-cast kernels at load | the installed operators run |
| `cuobjdump` | check the instructions of the built kernels | the installed operators run |

The built kernels and the installed operators produce the same bytes, so the
toolchain changes speed, not output. `transformer.fp8_kernel_qualification`
reports which one is active and why, and the loader logs a warning when the
built kernels are not used (shown with `TLLM_LOG_LEVEL=warning`). `python -O`
also uses the installed operators, as does a failed build or a failed inspection
of the built binaries. A built kernel whose output differs from the installed
operators fails the load.

Model loading compiles the CuTe DSL kernels (GateUp and the two projections)
and prepares packed GateUp weights and reciprocal scales before `torch.compile`
wraps the blocks; the Triton kernels compile during the loader's warmup request.
The denoise loop compiles no projection kernel: an input outside the compiled
kernel's dtypes, layout or 256-byte alignment takes the unfused cuBLAS path
(whose Triton AdaLN variant compiles on first use; the blocks' own tensors
never take it).
The packed GateUp weights replace the checkpoint layout in place, so the
model's `state_dict()` no longer holds that weight in checkpoint order.

## Implementation

`modeling.py` owns the FP8 attention, block and Transformer subclasses;
`conditioning.py` owns each pipeline's timestep-keyed AdaLN tables. `ops.py`
defines the compiler boundaries. `kernels/` contains:

| File | Kernel |
|---|---|
| `adaln.py` | AdaLN to FP8, and gated residual + AdaLN to FP8 |
| `gate_up_swiglu.py` | GateUp GEMM + SwiGLU to FP8 |
| `projection_residual.py` | FP8 GEMM + gated residual, for the attention output and Down projections |
| `conversion.py`, `pdl.py`, `*.cu`, `*.cpp` | optional scalar FP8 quantize and the shared BF16 to FP32 output cast, with their load-time checks |

Q/K normalization, RoPE and FP8 quantization reuse the H3 NVFP4 scenario's
`fused_fp8_qk` kernel with `rope_halves_equal=True`. No shared Linear, GatedMLP
or NVFP4 model class is patched.

The conditioning cache keeps the AdaLN tables of the 64 most recently used
timestep keys (about 1.2 GiB at two timesteps per key) and evicts the least
recently used key beyond that.

Where the normalized hidden state is NaN (a NaN anywhere in a row, or an Inf
element), the fused AdaLN kernel writes a saturated FP8 code; the unfused
quantize emits the FP8 NaN code. Finite values are unaffected.

## Validation

`tests/unittest/_torch/visual_gen/models/minimax_h3/minimax_h3_fp8__sm_107__u1/`:

| Test | Needs | Checks |
|---|---|---|
| `test_fp8_contracts.py` | CPU | routing, admission, op wiring, load order, cache eviction |
| `test_fp8_kernels.py` | SM107 GPU with 212 SMs | each fused op against the unfused operators at six sequence lengths, the unfused path for inputs outside the kernel contract, non-finite rows |
| `test_fp8_conversions.py` | the same GPU, CUDA toolchain | built conversions byte-identical to the installed operators, directly and through the ops |
| `test_fp8_pipeline.py` | the same GPU, `MiniMax-H3-FP8` checkpoint, `NVIDIA_TF32_OVERRIDE=0` | 8-step end-to-end generation; not in a CI list |

The `rope_halves_equal` option is tested with the NVFP4 scenario's kernel tests
(`test_h3_scenario_fused_fp8_qk.py`).
