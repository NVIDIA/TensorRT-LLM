# Accelerating Video Generation with GEMM Quantization, Attention Quantization, and Attention Sparsification

By NVIDIA TensorRT-LLM Team

## Introduction

Wan 2.2 T2V-A14B refines an 81-frame, 1280×720 video over 50 denoising steps, applying its
high-noise and low-noise transformers to a long spatiotemporal sequence. In a representative
compiled BF16 profile on NVIDIA B200, attention accounts for 71.7% of pipeline-forward time and
linear-layer GEMMs for another 20.8%, making them the natural targets for optimization.

<p align="center">
  <img src="../media/tech_blog27_bf16_time_breakdown.png" alt="Pie chart showing that a compiled dense BF16 path spends 71.7% of pipeline-forward time in attention, 20.8% in GEMMs, and 7.5% in other work" width="1080">
</p>

<p align="center"><sub><em>Figure 1. Representative pipeline-forward breakdown for compiled dense
BF16 on B200.</em></sub></p>

Our earlier post,
[Scaling Video Generation Across NVL72 Rack with TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/blogs/tech_blog/blog25_Scaling_Video_Generation_Across_NVL72_Rack_with_TensorRT-LLM.md),
focused on scale-out. Here we combine three kernel-level optimizations for the transformer
workload: lower-precision linear-layer GEMMs, SAGE's INT8/FP8 attention path, and Skip Softmax's
dynamic rejection of low-contributing attention blocks. Together, NVFP4 GEMMs, SAGE, and a
conservative Skip Softmax operating point reduce pipeline-forward latency from **525.0 to 374.9
seconds**, a **1.40× speedup**; the fastest measured configuration reaches **348.0 seconds, or
1.51×**.

## Table of Contents

- [Three Complementary Optimizations](#three-complementary-optimizations)
- [Workload and Measurements](#workload-and-measurements)
- [Pipeline-Forward Results](#pipeline-forward-results)
- [Reproduction](#reproduction)
- [Conclusion](#conclusion)
- [References](#references)

## Three Complementary Optimizations

Each technique removes a different source of cost:

| Optimization | What it accelerates | Options used here |
| :--- | :--- | :--- |
| GEMM quantization | Eligible linear-layer GEMMs | BF16, `FP8_BLOCK_SCALES`, or `NVFP4` |
| SAGE attention | QK and PV computation inside attention | INT8 Q/K and FP8 V |
| Skip Softmax Attention | Softmax and PV work for dynamically rejected score blocks | Calibrated target sparsity and timestep cutoff |

### FP8 Block-Scaled and NVFP4 GEMMs

With dynamic quantization enabled, VisualGen converts eligible BF16 linear weights while loading
the model, so both `FP8_BLOCK_SCALES` and `NVFP4` run without a separately prequantized checkpoint.
The first uses block-scaled FP8 GEMMs; the second further reduces operand precision with NVFP4.

### SAGE Attention

We use [SageAttention](https://github.com/thu-ml/SageAttention) with BF16 inputs and outputs, INT8
Q/K, FP8 V, and Q/K/V block sizes of 1/16/1. This attention path can be combined with FP8 or NVFP4
projection GEMMs.

### Skip Softmax Attention

[Skip Softmax Attention](blog16_Accelerating_Long_Context_Inference_with_Skip_Softmax_Attention.md),
also called BLASST, keeps the QK calculation but rejects score blocks sufficiently below the
running maximum. Rejected blocks skip exponentiation and the corresponding value accumulation;
the sparsity pattern is determined dynamically rather than stored with the model.

`target_sparsity` requests a calibrated operating point rather than fixing the runtime skip rate.
Each transformer maps it to its own `threshold_scale_factor`, and the kernel uses
`threshold_scale_factor / sequence_length` as the runtime threshold. Because scores vary by layer
and timestep, achieved sparsity varies as well. Wan 2.2's high-noise and low-noise transformers use
separate calibration curves, so the same target sparsity can produce different scale factors.

`disabled_until_timestep` controls when skipping begins. Timesteps descend from near 1 to 0, so a
lower cutoff retains more dense denoising; `0` disables skipping, while `1` enables it from the
first step.

## Workload and Measurements

We keep the generation workload fixed while varying GEMM precision, attention precision, target
sparsity, and the point in denoising at which skipping begins.

| Item | Setting |
| :--- | :--- |
| Checkpoint | [Wan-AI/Wan2.2-T2V-A14B-Diffusers](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B-Diffusers) |
| Accelerator | NVIDIA B200 |
| Generated video | 1280×720, 81 frames at 16 FPS |
| Denoising | 50 steps; guidance 5.0 for the high-noise expert and 4.0 for the low-noise expert; maximum text sequence length 512 |

Classifier-free guidance runs at every step by batching the positive and negative branches
together; CFG parallelism is disabled.

We combine all three GEMM modes with dense or SAGE attention, then sweep the following Skip Softmax
settings:

```text
{BF16, FP8_BLOCK_SCALES, NVFP4}
× {dense attention, SAGE}
× target_sparsity {0.65, 0.70, 0.75}
× disabled_until_timestep {0.86, 0.90, 0.94, 0.97, 1.00}
```

Across the resulting 96 configurations, every setting uses the same seven prompt-and-seed pairs.
Speedup is relative to compiled dense BF16; LPIPS compares each video with an eager BF16 generation
from the same prompt and seed.

## Pipeline-Forward Results

Quantizing the GEMMs alone improves the dense-attention path to 1.07× with FP8 block scaling and
1.11× with NVFP4. SAGE improves each precision family further, with its largest incremental gain
alongside NVFP4. Adding conservative Skip Softmax brings NVFP4 + SAGE to 1.40×.

<p align="center">
  <img src="../media/tech_blog27_latency.png" alt="Grouped bar chart showing pipeline-forward latency for BF16, FP8 block-scaled, and NVFP4 GEMMs with dense attention, SAGE attention, and SAGE plus conservative Skip Softmax" width="1080">
</p>

<p align="center"><sub><em>Figure 2. Pipeline-forward latency as GEMM quantization, SAGE, and
conservative Skip Softmax are combined. Speedups are relative to compiled dense BF16; Skip Softmax
uses `target_sparsity=0.65` and `disabled_until_timestep=0.86`.</em></sub></p>

<!-- TODO(blog27, remove before merge): Add a synchronized comparison of compiled BF16 and
NVFP4 + SAGE + Skip Softmax using the same prompt, seed, resolution, frame count, and denoising
steps. Host the video in an approved NVIDIA media location and keep only its poster image in this
repository. -->

Skip Softmax introduces an explicit quality-speed tradeoff. Figure 3 places every skip-enabled run
next to its no-skip precision-family anchor. LPIPS is measured against eager BF16, so the compiled
dense BF16 anchor starts at 0.118 rather than zero.

<p align="center">
  <img src="../media/tech_blog27_quality_speed_frontier.png" alt="Scatter plot of all 96 configurations showing speedup over compiled dense BF16 against mean LPIPS distance to eager BF16, with dense family anchors marked as stars and three operating points marked by numbered arrows" width="1080">
</p>

<p align="center"><sub><em>Figure 3. Quality-speed frontier across all 96 configurations; ①–③
identify the operating points below.</em></sub></p>

| Point | Goal | Configuration | Latency | Speedup | Mean LPIPS |
| :---: | :--- | :--- | ---: | ---: | ---: |
| ① | Conservative BF16 skipping | BF16 + dense attention + Skip (target 0.65, cutoff 0.86) | 487.6s | 1.08× | 0.141 |
| ② | Balanced operating point | NVFP4 + SAGE + Skip (target 0.65, cutoff 0.86) | 374.9s | 1.40× | 0.504 |
| ③ | Reach the highest measured speed | NVFP4 + SAGE + Skip (target 0.75, cutoff 1.00) | 348.0s | 1.51× | 0.523 |

At point ②, Skip Softmax moves the dense NVFP4 + SAGE path from 1.28× to 1.40× while LPIPS remains
near the matching no-skip anchor (0.504 versus 0.506). Point ③ enables skipping from the beginning
and trades a further increase in LPIPS for the highest measured speed. Across the sweep, the
denoising cutoff changes output similarity more strongly than target sparsity. Start with a
conservative cutoff, then raise target sparsity only when additional speed is needed.

## Reproduction

The [Wan 2.2 Skip Softmax example](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/visual_gen/skip_softmax/wan2.2-t2v-a14b)
packages the component-specific calibration overlays and VisualGen configuration for point ②. The
measurements in this post use the TensorRT-LLM release image
`nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc19`; set `MODEL_DIR` and `TRTLLM_ROOT` to local paths in
that environment.

### Enable the optimized path

Apply the high-noise and low-noise calibration overlays to their matching transformer configs:

```bash
export MODEL_DIR=/path/to/Wan2.2-T2V-A14B-Diffusers
export TRTLLM_ROOT=/path/to/TensorRT-LLM
export EXAMPLE_DIR="$TRTLLM_ROOT/examples/visual_gen/skip_softmax/wan2.2-t2v-a14b"

python "$EXAMPLE_DIR/apply_calibration.py" --model-dir "$MODEL_DIR"
```

The helper merges the matching `sparse_attention_config` overlay into each transformer config and
leaves all other fields unchanged. The packaged
[`visual_gen.yaml`](https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/visual_gen/skip_softmax/wan2.2-t2v-a14b/visual_gen.yaml)
selects the optimization settings for point ②: `target_sparsity=0.65`,
`disabled_until_timestep=0.86`, dynamic NVFP4 GEMM quantization, and the INT8/FP8 SAGE recipe. The
[VisualGen sparse-attention guide](../../visual-gen/features/sparse-attention.md) describes the
checkpoint schema and precedence rules.

Start VisualGen with the packaged configuration:

```bash
trtllm-serve "$MODEL_DIR" --visual_gen_args "$EXAMPLE_DIR/visual_gen.yaml"
```

### Measure latency and quality

The latency and speedup values in Figures 2 and 3 measure the pipeline forward rather than HTTP
handling or video encoding. For each prompt, run one untimed six-step forward at the same
1280×720, 81-frame shape, enough to compile both Wan transformer stages. Synchronize the GPU, time
one complete 50-step forward, and average the resulting times over the seven prompts.

The YAML mirrors the sweep: `torch.compile` is enabled, CUDA graphs are disabled because rc19 did
not support the combination, and `compilation_config.skip_warmup=true` bypasses the generic
multi-shape warmup in favor of the six-step exact-shape forward above. Autotuning is disabled as
well; in this pipeline, it is invoked by the skipped generic warmup, so enabling it would not
affect the sweep.

For the quality axis, generate an eager BF16 reference with the same prompt and seed, compute
AlexNet LPIPS between corresponding frames, average over the 81 frames, and then average over the
seven prompts.

<!-- TODO(blog27, remove before merge): Link the exact seven-prompt manifest, offline timing
harness, LPIPS evaluator environment, result CSV, and figure-generation scripts after they are
published. -->

## Conclusion

For Wan 2.2, lower-precision GEMMs and SAGE shift the dense operating point, while Skip Softmax
provides a final quality-speed control. The balanced NVFP4 + SAGE + Skip configuration reaches
**1.40×**, and the fastest point measured reaches **1.51×**. In practice, choose GEMM and attention
precision first, then tune the denoising cutoff before increasing target sparsity.

These kernel-level gains are orthogonal to the scale-out techniques in
[Scaling Video Generation Across NVL72 Rack with TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/blogs/tech_blog/blog25_Scaling_Video_Generation_Across_NVL72_Rack_with_TensorRT-LLM.md),
so they can be combined with multi-GPU parallelism for higher throughput.

## References

1. [Scaling Video Generation Across NVL72 Rack with TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/blogs/tech_blog/blog25_Scaling_Video_Generation_Across_NVL72_Rack_with_TensorRT-LLM.md)
2. [SageAttention: Accurate 8-Bit Attention for Plug-and-play Inference Acceleration](https://arxiv.org/abs/2410.02367)
3. [BLASST: Dynamic BLocked Attention Sparsity via Softmax Thresholding](https://arxiv.org/abs/2512.12087)
