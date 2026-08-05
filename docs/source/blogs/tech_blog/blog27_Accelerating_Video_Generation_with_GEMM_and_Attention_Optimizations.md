# Accelerating Video Generation with GEMM Quantization, Attention Quantization, and Attention Sparsification

By NVIDIA TensorRT-LLM Team

## Introduction

Generating an 81-frame, 1280×720 video with Wan 2.2 T2V-A14B applies its dual transformers across
50 denoising steps. In a representative compiled BF16 profile on NVIDIA B200, attention accounts
for 71.7% of end-to-end latency and linear-layer GEMMs for another 20.8%. Together, they occupy
more than 90% of runtime, motivating optimizations for both parts of the workload.

<p align="center">
  <img src="../media/tech_blog27_bf16_time_breakdown.png" alt="Pie chart showing that a representative compiled dense BF16 path spends 71.7% of end-to-end latency in attention, 20.8% in GEMMs, and 7.5% in other work" width="1080">
</p>

<p align="center"><sub><em>Figure 1. Pipeline breakdown for the target Wan 2.2 workload with
compiled dense BF16 on B200.</em></sub></p>

In [Scaling Video Generation Across NVL72 Rack with TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/blogs/tech_blog/blog25_Scaling_Video_Generation_Across_NVL72_Rack_with_TensorRT-LLM.md),
we showed how TensorRT-LLM scales video generation across multiple GPUs. This post focuses on
accelerating the work on each GPU with three complementary techniques:

- **FP8 block-scaled or NVFP4 GEMMs** reduce linear-layer cost.
- **SAGE attention** dynamically quantizes Q and K to INT8 and V to FP8 in the attention path.
- **Skip Softmax Attention** skips softmax and value accumulation for attention blocks whose
  scores fall below a dynamic threshold.

Together, NVFP4 GEMMs, SAGE attention, and a conservative Skip Softmax operating point reduce
latency from **525.0 to 374.9 seconds**, a **1.40× end-to-end speedup**. The fastest measured
configuration reaches **348.0 seconds, or 1.51×**.

## Table of Contents

- [Three Complementary Optimizations](#three-complementary-optimizations)
- [Evaluation Setup](#evaluation-setup)
- [End-to-End Results](#end-to-end-results)
- [Reproduction](#reproduction)
- [Conclusion](#conclusion)
- [References](#references)

## Three Complementary Optimizations

Video diffusion repeatedly applies a full transformer to a long, bidirectional spatiotemporal
sequence. The three techniques in this study act on different parts of that work:

| Optimization | What it accelerates | Options used here |
| :--- | :--- | :--- |
| GEMM quantization | Eligible linear-layer GEMMs | BF16, `FP8_BLOCK_SCALES`, or `NVFP4` |
| SAGE attention | QK and PV computation inside attention | INT8 Q/K and FP8 V |
| Skip Softmax Attention | Softmax and PV work for dynamically rejected score blocks | Calibrated target sparsity and timestep cutoff |

Because the techniques target different kernels, they can be enabled independently or composed.

### FP8 Block-Scaled and NVFP4 GEMMs

VisualGen dynamically quantizes eligible BF16 linear weights as the model loads. The FP8 and
NVFP4 paths used here therefore do not require separate prequantized checkpoints.
`FP8_BLOCK_SCALES` uses block-scaled FP8 GEMMs, while `NVFP4` reduces the data width further.
Operations outside the eligible linear layers remain at their configured precision.

### SAGE Attention

[SageAttention](https://github.com/thu-ml/SageAttention) reduces the precision of attention
arithmetic. The recipe used here dynamically quantizes Q and K to INT8 and V to FP8, with block
sizes 1/16/1 and BF16 input and output tensors. This is separate from GEMM quantization: NVFP4 can
accelerate the linear projections while SAGE accelerates the attention operation that consumes
them.

### Skip Softmax Attention

[Skip Softmax Attention](blog16_Accelerating_Long_Context_Inference_with_Skip_Softmax_Attention.md),
also called BLASST, keeps the QK calculation but rejects score blocks sufficiently below the
running maximum. Rejected blocks skip exponentiation and the corresponding value accumulation;
the sparsity pattern is determined dynamically rather than stored with the model.

`target_sparsity` selects an operating point calibrated by ModelOpt. The calibration metadata maps
that request to a `threshold_scale_factor` for each transformer component; the kernel divides this
factor by the sequence length to obtain its runtime threshold. The achieved skip rate can vary by
layer and timestep.

Wan 2.2 has separate high-noise and low-noise transformer components. To use `target_sparsity`
with the public checkpoint, the TensorRT-LLM helper merges a calibrated `sparse_attention_config`
overlay into both `transformer/config.json` and `transformer_2/config.json`. Each component keeps
its own calibration formula, while the rest of both configurations remains unchanged. The
[VisualGen sparse-attention guide](../../visual-gen/features/sparse-attention.md) documents the
checkpoint schema and precedence rules.

<!-- TODO(blog27, remove before merge): Add the approved ModelOpt-produced calibration overlays
and component-aware merge helper to the TensorRT-LLM example directory, then link its GitHub main
URL here. -->

The second control, `disabled_until_timestep`, keeps the beginning of denoising dense. Denoising
moves from a normalized timestep near 1 toward 0, and skipping begins only after the timestep falls
below this cutoff. A lower value leaves a longer dense prefix; `0` disables skipping, while `1`
enables it from the beginning. In this workload, delaying Skip Softmax preserves output similarity
more effectively than applying the same threshold throughout denoising.

## Evaluation Setup

To see how the optimizations compose, we varied GEMM precision, attention precision, target
sparsity, and the point in denoising at which skipping begins.

| Item | Setting |
| :--- | :--- |
| Checkpoint | [Wan-AI/Wan2.2-T2V-A14B-Diffusers](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B-Diffusers) |
| Accelerator | NVIDIA B200 |
| Generated video | 1280×720, 81 frames at 16 FPS |
| Denoising | 50 steps; guidance 5.0 for the high-noise expert and 4.0 for the low-noise expert; maximum text sequence length 512 |

Both guidance values enable classifier-free guidance (CFG) throughout denoising. At each step, the
positive- and negative-prompt branches are processed together as a batch of two; CFG parallelism is
not enabled in this configuration.

The experiment covered the following grid:

```text
{BF16, FP8_BLOCK_SCALES, NVFP4}
× {dense attention, SAGE}
× target_sparsity {0.65, 0.70, 0.75}
× disabled_until_timestep {0.86, 0.90, 0.94, 0.97, 1.00}
```

This produces 90 Skip Softmax variants plus six skip-off anchors. We evaluate each setting with the
same seven prompt-and-seed pairs. Speedup is measured against compiled dense BF16, while LPIPS
compares each generated video with an eager BF16 reference from the same prompt and seed.

## End-to-End Results

The three techniques compound because they reduce different parts of the workload. GEMM
quantization first moves the dense baseline from 1.00× to 1.07× with FP8 block scaling and 1.11×
with NVFP4. SAGE then reduces attention cost, with its largest incremental gain appearing alongside
NVFP4. Adding conservative Skip Softmax brings the combined NVFP4 + SAGE path to 1.40×.

<p align="center">
  <img src="../media/tech_blog27_latency.png" alt="Grouped bar chart showing end-to-end latency for BF16, FP8 block-scaled, and NVFP4 GEMMs with dense attention, SAGE attention, and SAGE plus conservative Skip Softmax" width="1080">
</p>

<p align="center"><sub><em>Figure 2. The three optimizations compose. All speedups use the same
compiled dense BF16 baseline. Skip Softmax uses `target_sparsity=0.65` and
`disabled_until_timestep=0.86`; lower latency is better.</em></sub></p>

<!-- TODO(blog27, remove before merge): Add a synchronized comparison of compiled BF16 and
NVFP4 + SAGE + Skip Softmax using the same prompt, seed, resolution, frame count, and denoising
steps. Host the video in an approved NVIDIA media location and keep only its poster image in this
repository. -->

Figure 3 shows the speed-quality tradeoff across the sweep. Each star is the skip-off anchor for one
GEMM-and-attention family; read nearby Skip Softmax points relative to that star. LPIPS uses eager
BF16 as its image-space reference, so compiled dense BF16 begins at a nonzero distance of 0.118.

<p align="center">
  <img src="../media/tech_blog27_quality_speed_frontier.png" alt="Scatter plot of all 96 configurations showing speedup over compiled dense BF16 against mean LPIPS distance to eager BF16, with dense family anchors marked as stars and three operating points marked by numbered arrows" width="1080">
</p>

<p align="center"><sub><em>Figure 3. The measured quality-speed frontier. Farther right is faster;
lower is closer to eager BF16. Stars are skip-off family anchors, and ①–③ correspond to the
representative operating points below.</em></sub></p>

| Point | Goal | Configuration | Latency | Speedup | Mean LPIPS |
| :---: | :--- | :--- | ---: | ---: | ---: |
| ① | Conservative BF16 skipping | BF16 + dense attention + Skip (target 0.65, cutoff 0.86) | 487.6s | 1.08× | 0.141 |
| ② | Balance the combined optimizations | NVFP4 + SAGE + Skip (target 0.65, cutoff 0.86) | 374.9s | 1.40× | 0.504 |
| ③ | Reach the highest measured speed | NVFP4 + SAGE + Skip (target 0.75, cutoff 1.00) | 348.0s | 1.51× | 0.523 |

At point ②, Skip Softmax moves the dense NVFP4 + SAGE path from 1.28× to 1.40× while mean LPIPS
is 0.504, compared with 0.506 for the matching skip-off anchor. Point ③ starts skipping earlier
and trades a further increase in LPIPS for the highest measured speed. Across the sweep, the
denoising cutoff changes output similarity more strongly than target sparsity. Start with a
conservative cutoff, then raise target sparsity only when additional speed is needed.

## Reproduction

The following flow runs the feature combination used at point ②. Reproducing the complete sweep
also requires the prompt manifest, offline timing harness, and evaluation assets described below.
Use the TensorRT-LLM release image `nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc19` with a
TensorRT-LLM checkout mounted at `$TRTLLM_ROOT` so the calibration package is available inside the
container.

First, materialize the public Wan 2.2 checkpoint so its component configurations can be updated:

```bash
export MODEL_DIR="$PWD/Wan2.2-T2V-A14B-Diffusers"
hf download Wan-AI/Wan2.2-T2V-A14B-Diffusers --local-dir "$MODEL_DIR"
```

Once the approved calibration package is published, apply its high-noise and low-noise overlays to
the corresponding transformer configurations:

```bash
export TRTLLM_ROOT=/workspace/TensorRT-LLM
python "$TRTLLM_ROOT/examples/visual_gen/skip_softmax/wan2.2-t2v-a14b/apply_calibration.py" \
    --model-dir "$MODEL_DIR"
```

<!-- TODO(blog27, remove before merge): Add the calibration overlays and helper at the path used
above, or update this command and link to their approved public location. Verify the command from
a clean download of the Hugging Face checkpoint. -->

Save the following feature configuration as `nvfp4_sage_skip.yaml`:

```yaml
quant_config:
  quant_algo: NVFP4
  dynamic: true

attention_config:
  backend: TRTLLM
  quant_attention_config:
    qk_dtype: int8
    v_dtype: fp8
    q_block_size: 1
    k_block_size: 16
    v_block_size: 1
  sparse_attention_config:
    algorithm: skip_softmax
    target_sparsity: 0.65
    disabled_until_timestep: 0.86

torch_compile_config:
  enable: true
  enable_autotune: false

cuda_graph_config:
  enable: false

parallel_config:
  cfg_size: 1
  ulysses_size: 1
```

Start VisualGen with the local checkpoint and configuration:

```bash
trtllm-serve "$MODEL_DIR" --visual_gen_args nvfp4_sage_skip.yaml
```

From another shell in the same running container, export `MODEL_DIR` again and generate the
representative workload:

```bash
export MODEL_DIR="$PWD/Wan2.2-T2V-A14B-Diffusers"
python -m tensorrt_llm.serve.scripts.benchmark_visual_gen \
    --model "$MODEL_DIR" \
    --backend openai-videos \
    --prompt "A racehorse galloping on a dirt track, kicking up dust, side tracking shot, dramatic lighting" \
    --num-prompts 1 \
    --size 1280x720 \
    --num-frames 81 \
    --fps 16 \
    --num-inference-steps 50 \
    --guidance-scale 5.0 \
    --seed 1007 \
    --extra-body '{"max_sequence_length":512,"extra_params":{"guidance_scale_2":4.0}}' \
    --max-concurrency 1 \
    --no-test-input \
    --save-result
```

This serving flow runs NVFP4 GEMMs, SAGE, and Skip Softmax together. The latency values in Figures
2 and 3 use a narrower pipeline-forward protocol: the sweep bypasses the
pipeline's generic multi-shape warmup, performs one full warmup at the exact 1280×720, 81-frame
shape, synchronizes the GPU, and then times one complete 50-step forward for each prompt. The seven
times are averaged. HTTP handling and video encoding are outside that interval.

The compilation controls mirror the measured sweep rather than prescribe production defaults.
`torch.compile` is enabled, while CUDA graphs are disabled because their combination was not
supported by the release used for this experiment. `enable_autotune` was false; this setting did not
affect the sweep because autotuning was invoked by the built-in warmup, which the sweep bypassed.
For quality evaluation, generate the eager BF16 references with the same prompts and seeds, compute
AlexNet LPIPS for corresponding frames, average over each 81-frame video, and then average over the
prompt set.

<!-- TODO(blog27, remove before merge): Link the exact seven-prompt manifest, offline timing
harness, LPIPS evaluator environment, result CSV, and figure-generation scripts after they are
published. -->

## Conclusion

GEMM quantization, SAGE attention, and Skip Softmax address complementary parts of video
generation: linear layers, attention arithmetic, and dynamically unimportant attention work. They
can be adopted independently or combined. The dense precision choice establishes the main
speed-quality operating region, and the Skip Softmax denoising cutoff provides the final tuning
control within that region.

Improving per-GPU execution and scaling across GPUs are complementary strategies. GEMM
quantization, SAGE, and Skip Softmax compose with the scale-out methods in
[Scaling Video Generation Across NVL72 Rack with TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/blogs/tech_blog/blog25_Scaling_Video_Generation_Across_NVL72_Rack_with_TensorRT-LLM.md),
pairing more efficient work on each GPU with rack-scale throughput.

## References

1. [Scaling Video Generation Across NVL72 Rack with TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/blogs/tech_blog/blog25_Scaling_Video_Generation_Across_NVL72_Rack_with_TensorRT-LLM.md)
2. [SageAttention: Accurate 8-Bit Attention for Plug-and-play Inference Acceleration](https://arxiv.org/abs/2410.02367)
3. [BLASST: Dynamic BLocked Attention Sparsity via Softmax Thresholding](https://arxiv.org/abs/2512.12087)
4. [NVIDIA Model Optimizer](https://github.com/NVIDIA/Model-Optimizer)
5. [Wan-AI/Wan2.2-T2V-A14B-Diffusers](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B-Diffusers)
6. [TensorRT-LLM Visual Generation](../../models/visual-generation.md)
