# Accelerating Video Generation: Quantization and Attention Optimizations in TensorRT-LLM

By NVIDIA TensorRT-LLM Team

## Introduction

Video diffusion transformers repeatedly process long spatiotemporal token sequences across many
denoising steps. Linear layers and attention therefore dominate much of the compute, making them
the two most direct targets for reducing generation latency.

For example, the figure below demonstrates the breakdown of pipeline-forward time for Wan 2.2 T2V-A14B on a single NVIDIA B200. For an 81-frame, 1280×720 video with 50 denoising
steps, attention and linear-layer GEMMs account for 71.7% and 20.8% of compiled BF16
pipeline-forward time, respectively.

<p align="center">
  <img src="../media/tech_blog28_bf16_time_breakdown.png" alt="Pie chart showing that a compiled dense BF16 path spends 71.7% of pipeline-forward time in attention, 20.8% in GEMMs, and 7.5% in other work" width="1080">
</p>

<p align="center"><sub><em>Figure 1. Pipeline-forward breakdown for compiled dense BF16 on
B200.</em></sub></p>

Our earlier post,
[Scaling Video Generation Across NVL72 Rack with TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/blogs/tech_blog/blog25_Scaling_Video_Generation_Across_NVL72_Rack_with_TensorRT-LLM.md),
focused on scale-out. This post covers three complementary acceleration techniques inside one
transformer pipeline: linear-layer quantization, quantized attention, and sparse attention. On
the Wan 2.2 workload above, the combination of NVFP4 linear layers, SAGE quantized attention, and
a conservative Skip Softmax setting reduces pipeline-forward latency from **525.0 to 374.9
seconds**, a **1.40× speedup**. In this blog, we introduce these optimization techniques and discuss the trade off between quality and speed.

## Table of Contents

- [Quantization](#quantization)
- [Attention Optimizations](#attention-optimizations)
- [Results](#results)
- [Agentic Automation and Reproduction](#agentic-automation-and-reproduction)
- [Conclusion](#conclusion)
- [References](#references)

## Quantization

Linear-layer quantization reduces the precision of eligible weights and activations so their GEMMs
can use higher-throughput Tensor Core paths. The available and upcoming VisualGen paths differ in
both numeric precision and scale granularity:

| Linear-layer path | Weight / activation precision | Scale granularity | Dynamic from high-precision weights | Static checkpoint |
| :--- | :--- | :--- | :---: | :---: |
| FP8 per-tensor | FP8 E4M3 / FP8 E4M3 | One scale per tensor | Yes | Yes |
| FP8 blockwise | FP8 E4M3 / FP8 E4M3 | 128×128 weight blocks; 1×128 activation blocks | Yes | Yes |
| FP8 row-wise | FP8 E4M3 / FP8 E4M3 | Per-output-channel weights; per-token activations | Yes ([WIP](https://github.com/NVIDIA/TensorRT-LLM/pull/16847)) | Yes ([WIP](https://github.com/NVIDIA/TensorRT-LLM/pull/16847)) |
| NVFP4 | FP4 E2M1 / FP4 E2M1 | 16-element blocks with FP8 scale factors | Yes | Yes |

Here, *dynamic* means converting high-precision weights while loading the model, while *static*
means loading a prequantized checkpoint with its scales. Static checkpoints can be produced with
[NVIDIA Model Optimizer](https://github.com/NVIDIA/Model-Optimizer), whose offline calibration can
apply model-aware choices such as calibrated scales and keeping sensitive layers at higher
precision. At the same target format, these algorithmic choices generally preserve accuracy
better than quantizing every eligible layer dynamically at load time. This distinction describes
how weights are prepared; activations are quantized as the model runs. All FP8 blockwise and
NVFP4 results below use dynamic quantization from the public BF16 checkpoint. NVFP4 pushes
precision lower for more throughput.

## Attention Optimizations

Attention offers two complementary levers: quantization makes its matrix multiplications cheaper,
while sparsity avoids work that contributes little to the output.

### Quantized Attention

An attention layer performs two matrix multiplications around softmax. QK16PV8 keeps Q and K in
BF16 and quantizes V to FP8, accelerating the second multiplication while retaining full-precision
inputs for the first. SAGE goes further by quantizing Q/K as well as V, allowing both
multiplications to run through an 8-bit path. TensorRT-LLM's SAGE recipe follows the
[SageAttention2](https://arxiv.org/abs/2411.10958) line of work, using INT8 or FP8 Q/K and FP8 V
rather than the paper's per-thread INT4 Q/K recipe.

This post uses SAGE, which can be layered with Skip Softmax to combine quantization and sparsity in
the same attention path.

### Sparse Attention with Skip Softmax

[Skip Softmax Attention](blog16_Accelerating_Long_Context_Inference_with_Skip_Softmax_Attention.md),
also called BLASST, keeps the QK calculation but rejects score blocks sufficiently below the
running maximum. Rejected blocks skip exponentiation and the corresponding value accumulation;
the sparsity pattern is determined dynamically rather than stored with the model.

Two controls shape the tradeoff: how aggressively to skip and how far into denoising to begin.
Calibration also protects sensitive layers by leaving them on dense attention. Together, these
controls turn Skip Softmax into a final tuning knob after linear-layer and attention quantization
have established the main operating point.

## Results

### Experimental setup

We evaluate the three techniques on a fixed Wan 2.2 workload while varying linear-layer precision,
attention precision, and Skip Softmax settings.

| Item | Setting |
| :--- | :--- |
| Checkpoint | [Wan-AI/Wan2.2-T2V-A14B-Diffusers](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B-Diffusers) |
| Accelerator | NVIDIA B200 |
| Generated video | 1280×720, 81 frames at 16 FPS |
| Denoising | 50 steps; guidance 5.0 for the high-noise expert and 4.0 for the low-noise expert; maximum text sequence length 512 |
| Linear-layer paths | BF16 baseline, dynamic `FP8_BLOCK_SCALES`, dynamic `NVFP4` |
| Attention paths | Dense, or SAGE with INT8 Q/K, FP8 V, and Q/K/V block sizes of 1/16/1 |

Classifier-free guidance runs at every step by batching the positive and negative branches
together; CFG parallelism is disabled.

For the FP8 and NVFP4 runs, VisualGen quantizes eligible BF16 weights while loading the model.

The 96 configurations are a characterization sweep, not a recommended per-model tuning cost:

```text
{BF16 reference, FP8_BLOCK_SCALES, NVFP4}
× {dense attention, SAGE}
× {no Skip Softmax, or
   target_sparsity {0.65, 0.70, 0.75}
   × disabled_until_timestep {0.86, 0.90, 0.94, 0.97, 1.00}}
```

Every setting uses the same seven prompt-and-seed pairs. Speedup is relative to compiled dense
BF16, and LPIPS compares each video with an eager BF16 generation from the same prompt and seed.
Latency covers one complete 50-step pipeline forward after compilation warmup; model loading, HTTP
handling, and video encoding are outside the measurement.

### Quality-speed frontier

Skip Softmax introduces an explicit quality-speed tradeoff. Figure 2 places every skip-enabled run
next to its no-skip precision-family anchor. LPIPS is measured against eager BF16, so the compiled
dense BF16 anchor starts at 0.118 rather than zero.

<p align="center">
  <img src="../media/tech_blog28_quality_speed_frontier.png" alt="Scatter plot of all 96 configurations showing speedup over compiled dense BF16 against mean LPIPS distance to eager BF16, with dense family anchors marked as stars and three operating points marked by numbered arrows" width="1080">
</p>

<p align="center"><sub><em>Figure 2. Quality-speed frontier across all 96 configurations; ①–③
identify the operating points below.</em></sub></p>

| Point | Goal | Configuration | Latency | Speedup | Mean LPIPS |
| :---: | :--- | :--- | ---: | ---: | ---: |
| ① | Conservative BF16 skipping | BF16 + dense attention + Skip (target 0.65, cutoff 0.86) | 487.6s | 1.08× | 0.141 |
| ② | Conservative Skip setting within the quantized family | NVFP4 + SAGE + Skip (target 0.65, cutoff 0.86) | 374.9s | 1.40× | 0.504 |
| ③ | Highest measured speed | NVFP4 + SAGE + Skip (target 0.75, cutoff 1.00) | 348.0s | 1.51× | 0.523 |

At point ②, Skip Softmax moves the NVFP4 + SAGE path from 1.28× to 1.40×. Its mean LPIPS is close
to the matching no-skip anchor (0.504 versus 0.506), so the conservative Skip setting adds little
incremental drift within that precision family on these seven prompt-and-seed pairs. The larger
quality shift has already happened between BF16 and the quantized family: at an LPIPS near 0.5,
synchronized video review becomes more useful than the metric alone. Check prompt adherence,
motion, and temporal consistency before choosing between points ② and ③. In this sweep, the
denoising cutoff influences similarity more strongly than target sparsity, making it the better
parameter to tune first.

### Pipeline-forward latency

Quantizing the linear layers alone improves the dense-attention path to 1.07× with FP8 block
scaling and 1.11× with NVFP4. SAGE improves each precision family further, with its largest
incremental gain alongside NVFP4. Adding the conservative Skip Softmax setting brings NVFP4 +
SAGE to 1.40×.

<p align="center">
  <img src="../media/tech_blog28_latency.png" alt="Grouped bar chart showing pipeline-forward latency for BF16, FP8 block-scaled, and NVFP4 linear layers with dense attention, SAGE attention, and SAGE plus conservative Skip Softmax" width="1080">
</p>

<p align="center"><sub><em>Figure 3. Pipeline-forward latency as linear-layer quantization, SAGE,
and conservative Skip Softmax are combined. Speedups are relative to compiled dense BF16; Skip
Softmax uses `target_sparsity=0.65` and `disabled_until_timestep=0.86`.</em></sub></p>

## Agentic Automation and Reproduction

The full 96-run sweep explains the interactions between the three techniques, but adapting the
recipe to another model should use a staged search:

1. Establish a compiled BF16 latency and quality reference.
2. Compare the supported linear-layer quantization paths with dense attention, treating dynamic
   and static weights as separate candidates, then retain only families that pass the quality gate.
3. Compare dense and supported quantized-attention recipes for those families.
4. Starting from `target_sparsity=0.65`, sweep a few conservative denoising cutoffs first.
5. Sweep target sparsity only around the best cutoff, expand locally near the Pareto frontier, and
   validate finalists on a broader prompt set.

An agent can generate the YAML variants, invoke the supplied reproduction command, collect both
pipeline-forward and client-observed latency, run the quality evaluator, and return the Pareto
frontier. The model-specific checkpoint calibration and its `ignore` lists remain authoritative;
the agent should not copy Wan 2.2 thresholds to another model.

<details>
<summary>Starter prompt for a client-side tuning agent</summary>

```text
Tune a TensorRT-LLM VisualGen configuration for a quality-constrained latency target.

Inputs I will provide:
- MODEL_DIR or Hugging Face model ID
- base VisualGen YAML
- one exact generation command
- prompt/seed manifest
- quality-evaluator command and acceptance thresholds
- target GPU and software image

Rules:
1. Do not modify TensorRT-LLM source code or checkpoint weights.
2. Label every FP8/NVFP4 result as dynamic or static. Do not compare or combine the two as if they
   were the same quantization recipe.
3. Keep resolution, frame count, denoising steps, scheduler, guidance, prompts, seeds, warmup,
   and synchronization identical across candidates.
4. Preserve checkpoint sparse-attention calibration and ignore lists. Change only documented
   user controls such as target_sparsity and disabled_until_timestep.
5. Stop evaluating a family when it fails a hard quality threshold. Never describe a recipe as
   quality-preserving from a small LPIPS delta alone.

Procedure:
A. Run compiled BF16 and eager BF16 references.
B. Benchmark the available dense-attention FP8 and NVFP4 anchors; retain passing families.
C. Benchmark each supported quantized-attention recipe for the retained families.
D. At target_sparsity=0.65, sweep a small set of conservative cutoffs. Around the best passing
   cutoff, sweep target sparsity and refine only near the Pareto frontier.
E. Re-run finalists enough times to report variation, then evaluate them on the full manifest.

Return:
- exact checkpoint, YAML, command, environment, and seed manifest for every reported point
- pipeline-forward latency and client-observed end-to-end breakdown
- aggregate and per-prompt quality metrics with pass/fail reasons
- achieved sparsity by layer/timestep when available
- a Pareto table and one recommended configuration, with unresolved risks stated explicitly
```

</details>

The [Wan 2.2 Skip Softmax example](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/visual_gen/skip_softmax/wan2.2-t2v-a14b)
packages the component-specific calibration overlays and VisualGen configuration. The
measurements in this post use the TensorRT-LLM release image
`nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc19`; set `MODEL_DIR` and `TRTLLM_ROOT` to local paths in
that environment.

### Configuration details

VisualGen selects a quantized-attention recipe through
`attention_config.quant_attention_config`. QK16PV8 and SAGE occupy the same field, but use
different backends and scaling layouts:

| Recipe | Backend | Q/K precision | V precision | Q/K/V block sizes | Combines with Skip Softmax |
| :--- | :--- | :--- | :--- | :--- | :--- |
| QK16PV8 | `CUTEDSL` | BF16 | FP8 | 0/0/0 (tensor-wide V scale) | No |
| SAGE used here | `TRTLLM` | INT8 | FP8 | 1/16/1 | Yes |

The current QK16PV8 kernel targets Blackwell `sm_100a` and `sm_103a` with head dimension 128; the
current SAGE path requires Blackwell `sm_100`. Skip Softmax also uses the `TRTLLM` backend, which
is why it composes with SAGE but not the `CUTEDSL` QK16PV8 path.

For Skip Softmax, `target_sparsity` is converted to a threshold through each transformer's
calibration formula; achieved sparsity can therefore vary by layer and timestep.
`disabled_until_timestep` controls when skipping begins as denoising descends from near 1 to 0: a
lower value keeps more early steps dense. Calibration also supplies an `ignore` list for layers
that should stay dense.

### Reproduce point ②

Apply the high-noise and low-noise calibration overlays to their matching transformer configs:

```bash
export MODEL_DIR=/path/to/Wan2.2-T2V-A14B-Diffusers
export TRTLLM_ROOT=/path/to/TensorRT-LLM
export EXAMPLE_DIR="$TRTLLM_ROOT/examples/visual_gen/skip_softmax/wan2.2-t2v-a14b"

python "$EXAMPLE_DIR/apply_calibration.py" --model-dir "$MODEL_DIR"
```

The helper merges the matching `sparse_attention_config` overlay into each transformer config and
leaves all other fields unchanged. Each overlay carries 44 layer names in `ignore`; those layers
remain dense. The packaged
[`visual_gen.yaml`](https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/visual_gen/skip_softmax/wan2.2-t2v-a14b/visual_gen.yaml)
selects point ②: `target_sparsity=0.65`,
`disabled_until_timestep=0.86`, dynamic NVFP4 linear-layer quantization, and the INT8/FP8 SAGE
recipe.

Start VisualGen with the packaged configuration:

```bash
trtllm-serve "$MODEL_DIR" --visual_gen_args "$EXAMPLE_DIR/visual_gen.yaml"
```

### Measure latency and quality

The latency and speedup values in Figures 2 and 3 measure the pipeline forward rather than HTTP
handling or video encoding. For each prompt, run one untimed six-step forward at the same
1280×720, 81-frame shape, enough to compile both Wan transformer stages. Synchronize the GPU, time
one complete 50-step forward, and average the resulting times over the seven prompts.

The YAML mirrors the current sweep: `torch.compile` is enabled, CUDA graphs are disabled because
rc19 did not support this feature combination, and `compilation_config.skip_warmup=true` bypasses
the generic multi-shape warmup in favor of the six-step exact-shape forward above. Autotuning is
disabled as well; in this pipeline, it is invoked by the skipped generic warmup, so enabling it
would not affect the sweep.

For the quality axis, generate an eager BF16 reference with the same prompt and seed, compute
AlexNet LPIPS between corresponding frames, average over the 81 frames, and then average over the
seven prompts. Treat LPIPS as one diagnostic rather than a complete video-quality measure.

## Conclusion

For the measured Wan 2.2 workload, linear-layer quantization first shifts the dense operating
point, SAGE quantized attention reduces attention cost, and Skip Softmax provides a final
quality-speed control. The conservative Skip setting within the NVFP4 + SAGE family reaches
**1.40×**, while the fastest measured point reaches **1.51×**.

The practical sequence is to select a calibrated linear-layer precision, then select an attention
recipe, and finally tune the Skip Softmax denoising cutoff before increasing target sparsity. These
techniques are orthogonal to the scale-out methods in
[Scaling Video Generation Across NVL72 Rack with TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/blogs/tech_blog/blog25_Scaling_Video_Generation_Across_NVL72_Rack_with_TensorRT-LLM.md),
so the same staged reasoning can be combined with multi-GPU parallelism for higher throughput.

## References

1. [Scaling Video Generation Across NVL72 Rack with TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/blogs/tech_blog/blog25_Scaling_Video_Generation_Across_NVL72_Rack_with_TensorRT-LLM.md)
2. [SageAttention: Accurate 8-Bit Attention for Plug-and-play Inference Acceleration](https://arxiv.org/abs/2410.02367)
3. [SageAttention2: Efficient Attention with Thorough Outlier Smoothing and Per-thread INT4 Quantization](https://arxiv.org/abs/2411.10958)
4. [NVIDIA Model Optimizer Diffusers Quantization Example](https://github.com/NVIDIA/Model-Optimizer/tree/main/examples/diffusers)
5. [BLASST: Dynamic BLocked Attention Sparsity via Softmax Thresholding](https://arxiv.org/abs/2512.12087)
