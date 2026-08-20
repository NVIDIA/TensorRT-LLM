# Accelerating Video Generation: Quantization and Attention Optimizations in TensorRT-LLM

By NVIDIA TensorRT-LLM Team

## Introduction

Video diffusion transformers repeatedly process long spatiotemporal token sequences across many
denoising steps. Linear layers and attention therefore dominate much of the compute, making them
the two most direct targets for reducing generation latency.

Wan 2.2 T2V-A14B provides a representative example. It refines an 81-frame, 1280×720 video over
50 denoising steps with separate high-noise and low-noise transformers. In one compiled BF16
profile on NVIDIA B200, attention accounts for 71.7% of pipeline-forward time and linear-layer
GEMMs for another 20.8%. These percentages describe this workload, not video-generation models in
general.

<p align="center">
  <img src="../media/tech_blog27_bf16_time_breakdown.png" alt="Pie chart showing that a compiled dense BF16 path spends 71.7% of pipeline-forward time in attention, 20.8% in GEMMs, and 7.5% in other work" width="1080">
</p>

<p align="center"><sub><em>Figure 1. Representative pipeline-forward breakdown for compiled dense
BF16 on B200.</em></sub></p>

Our earlier post,
[Scaling Video Generation Across NVL72 Rack with TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/blogs/tech_blog/blog25_Scaling_Video_Generation_Across_NVL72_Rack_with_TensorRT-LLM.md),
focused on scale-out. This post covers three complementary acceleration techniques inside one
transformer pipeline: linear-layer quantization, quantized attention, and sparse attention. On
the Wan 2.2 workload above, the combination of NVFP4 linear layers, SAGE quantized attention, and
a conservative Skip Softmax setting reduces pipeline-forward latency from **525.0 to 374.9
seconds**, a **1.40× speedup**. The fastest measured setting reaches **348.0 seconds, or 1.51×**.
We report the associated quality measurements and their limitations instead of treating speedup
alone as evidence that video quality is preserved.

## Table of Contents

- [Quantization](#quantization)
- [Attention Optimizations](#attention-optimizations)
- [Results](#results)
- [Agentic Automation and Reproduction](#agentic-automation-and-reproduction)
- [Conclusion](#conclusion)
- [References](#references)

## Quantization

Linear-layer quantization reduces the precision of eligible weights and activations so their GEMMs
can use higher-throughput Tensor Core paths. BF16 is the unquantized reference in this article; it
is not a quantization option. The measured quantized families are `FP8_BLOCK_SCALES` and `NVFP4`.

The measurements in this article use dynamic quantization: VisualGen converts eligible BF16 linear
weights while loading the model, without a separately prequantized checkpoint. This makes the
workflow reproducible from the public BF16 checkpoint, but the resulting numbers characterize only
this load-time quantization path.

A static checkpoint produced by
[NVIDIA Model Optimizer](https://github.com/NVIDIA/Model-Optimizer) can instead carry calibrated
scales and a model-specific recipe that leaves sensitive layers at a higher precision. This
selective treatment is especially useful when an all-NVFP4 recipe does not meet an application's
quality target. Dynamic and static recipes are not interchangeable; benchmark a static checkpoint
separately when a matching one is available.

## Attention Optimizations

Attention offers two separate controls. Quantized attention reduces the precision of the QK and PV
matrix multiplications. Sparse attention reduces the amount of softmax and PV work performed.

### Quantized Attention

VisualGen exposes two quantized-attention recipe families through
`attention_config.quant_attention_config`:

| Recipe | Backend | QK computation | PV computation | When it is useful |
| :--- | :--- | :--- | :--- | :--- |
| QK16PV8 | `CUTEDSL` | BF16 Q/K | FP8 V with a tensor-wide scale | Reduce the cost of the second attention matrix multiplication while retaining BF16 QK |
| SAGE | `TRTLLM` | INT8 or FP8 Q/K with token-block scales | FP8 V with channel-oriented blocking | Run both attention matrix multiplications through an 8-bit path |

QK16PV8 does not require offline calibration because its V scale is derived at runtime. The SAGE
recipe used in this article takes BF16 inputs, quantizes Q/K to INT8 and V to FP8, and uses Q/K/V
block sizes of 1/16/1. It is part of the same quantized-attention line of work as
[SageAttention2](https://arxiv.org/abs/2411.10958), but it is an 8-bit TensorRT-LLM recipe rather
than the paper's per-thread INT4 Q/K recipe.

The current QK16PV8 kernel targets Blackwell `sm_100a` and `sm_103a` with head dimension 128. The
current TensorRT-LLM SAGE path requires Blackwell `sm_100`.

The two recipes use the same configuration field but different backends. For example:

```yaml
# QK16PV8
attention_config:
  backend: CUTEDSL
  quant_attention_config:
    qk_dtype: bf16
    v_dtype: fp8
    q_block_size: 0
    k_block_size: 0
    v_block_size: 0
```

```yaml
# SAGE recipe used in this article
attention_config:
  backend: TRTLLM
  quant_attention_config:
    qk_dtype: int8
    v_dtype: fp8
    q_block_size: 1
    k_block_size: 16
    v_block_size: 1
```

In the current implementation, SAGE can be stacked with Skip Softmax because both use the
`TRTLLM` backend. QK16PV8 uses `CUTEDSL`, while Skip Softmax requires `TRTLLM`, so that pair cannot
be enabled together.

### Sparse Attention with Skip Softmax

[Skip Softmax Attention](blog16_Accelerating_Long_Context_Inference_with_Skip_Softmax_Attention.md),
also called BLASST, keeps the QK calculation but rejects score blocks sufficiently below the
running maximum. Rejected blocks skip exponentiation and the corresponding value accumulation;
the sparsity pattern is determined dynamically rather than stored with the model.

Users control the operating point with two fields. `target_sparsity` requests a calibrated target
rather than fixing the runtime skip rate. Each transformer maps the target to its own
`threshold_scale_factor`, and the kernel uses the scale factor together with the sequence length
to derive its threshold. Scores vary by layer and timestep, so achieved sparsity can differ from
the target. `disabled_until_timestep` controls when skipping begins: denoising timesteps descend
from near 1 to 0, so a lower cutoff retains more dense denoising; `0` disables skipping and `1`
enables it from the first step.

Calibration also determines where sparse attention is unsafe. Each Wan transformer checkpoint
contains its own `ignore` patterns. Matching attention layers fall back to dense attention, while
the remaining layers use Skip Softmax. This layer-selective fallback is specific to the VisualGen
calibration path; the high-noise and low-noise transformers can have different formulas and ignore
lists. The [VisualGen sparse-attention guide](../../visual-gen/features/sparse-attention.md) and
[Wan 2.2 example](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/visual_gen/skip_softmax/wan2.2-t2v-a14b)
document the checkpoint metadata and user overrides.

## Results

### Workload and measurement boundary

We keep the generation workload fixed while varying linear-layer precision, attention precision,
target sparsity, and the point in denoising at which skipping begins.

| Item | Setting |
| :--- | :--- |
| Checkpoint | [Wan-AI/Wan2.2-T2V-A14B-Diffusers](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B-Diffusers), dynamically quantized for the current draft results |
| Accelerator | NVIDIA B200 |
| Generated video | 1280×720, 81 frames at 16 FPS |
| Denoising | 50 steps; guidance 5.0 for the high-noise expert and 4.0 for the low-noise expert; maximum text sequence length 512 |

Classifier-free guidance runs at every step by batching the positive and negative branches
together; CFG parallelism is disabled.

The 96 configurations are a characterization sweep, not a recommended per-model tuning cost:

```text
{BF16 reference, FP8_BLOCK_SCALES, NVFP4}
× {dense attention, SAGE}
× {no Skip Softmax, or
   target_sparsity {0.65, 0.70, 0.75}
   × disabled_until_timestep {0.86, 0.90, 0.94, 0.97, 1.00}}
```

Every setting uses the same seven prompt-and-seed pairs. Speedup is relative to compiled dense
BF16. LPIPS compares each video with an eager BF16 generation from the same prompt and seed.

The measured latency covers one complete 50-step pipeline forward after compilation warmup. It
does not include model loading, HTTP request handling, or video encoding, so the values are not
client-observed end-to-end latency. Figure 1 breaks down the measured pipeline-forward scope; a
client-side end-to-end breakdown should additionally report request, generation, and encoding
time from the same reproduction command.

### Quality-speed frontier

Skip Softmax introduces an explicit quality-speed tradeoff. Figure 2 places every skip-enabled run
next to its no-skip precision-family anchor. LPIPS is measured against eager BF16, so the compiled
dense BF16 anchor starts at 0.118 rather than zero.

<p align="center">
  <img src="../media/tech_blog27_quality_speed_frontier.png" alt="Scatter plot of all 96 configurations showing speedup over compiled dense BF16 against mean LPIPS distance to eager BF16, with dense family anchors marked as stars and three operating points marked by numbered arrows" width="1080">
</p>

<p align="center"><sub><em>Figure 2. Quality-speed frontier across all 96 configurations; ①–③
identify the operating points below.</em></sub></p>

| Point | Goal | Configuration | Latency | Speedup | Mean LPIPS |
| :---: | :--- | :--- | ---: | ---: | ---: |
| ① | Conservative BF16 skipping | BF16 + dense attention + Skip (target 0.65, cutoff 0.86) | 487.6s | 1.08× | 0.141 |
| ② | Conservative Skip setting within the quantized family | NVFP4 + SAGE + Skip (target 0.65, cutoff 0.86) | 374.9s | 1.40× | 0.504 |
| ③ | Highest measured speed | NVFP4 + SAGE + Skip (target 0.75, cutoff 1.00) | 348.0s | 1.51× | 0.523 |

At point ②, Skip Softmax moves the NVFP4 + SAGE path from 1.28× to 1.40×. Its mean LPIPS is close
to the matching no-skip anchor (0.504 versus 0.506), which suggests that this conservative Skip
setting adds little incremental drift within that precision family on these seven prompt-and-seed
pairs.

That narrow comparison is not a claim of preserved output quality. An absolute LPIPS near 0.5
indicates substantial distance from the eager BF16 reference, and LPIPS becomes less informative
when videos already differ in structure or motion. The 0.002 difference is also too small to
interpret without run-to-run variation. Before selecting either point ② or ③, inspect synchronized
videos and evaluate prompt adherence, motion, temporal consistency, and application-specific human
preference on a broader prompt set. Across this limited sweep, the denoising cutoff changes output
similarity more strongly than target sparsity, so tune the cutoff first.

### Pipeline-forward latency

Quantizing the linear layers alone improves the dense-attention path to 1.07× with FP8 block
scaling and 1.11× with NVFP4. SAGE improves each precision family further, with its largest
incremental gain alongside NVFP4. Adding the conservative Skip Softmax setting brings NVFP4 +
SAGE to 1.40×.

<p align="center">
  <img src="../media/tech_blog27_latency.png" alt="Grouped bar chart showing pipeline-forward latency for BF16, FP8 block-scaled, and NVFP4 linear layers with dense attention, SAGE attention, and SAGE plus conservative Skip Softmax" width="1080">
</p>

<p align="center"><sub><em>Figure 3. Pipeline-forward latency as linear-layer quantization, SAGE,
and conservative Skip Softmax are combined. Speedups are relative to compiled dense BF16; Skip
Softmax uses `target_sparsity=0.65` and `disabled_until_timestep=0.86`.</em></sub></p>

## Agentic Automation and Reproduction

The full 96-run sweep explains the interactions between the three techniques, but adapting the
recipe to another model should use a staged search:

1. Establish a compiled BF16 latency and quality reference.
2. Compare the available prequantized ModelOpt checkpoints with dense attention, then retain only
   precision families that pass the quality gate.
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
packages the component-specific calibration overlays and current VisualGen configuration. The
measurements in this draft use the TensorRT-LLM release image
`nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc19`; set `MODEL_DIR` and `TRTLLM_ROOT` to local paths in
that environment.

### Enable the current measured path

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
selects the currently measured point ②: `target_sparsity=0.65`,
`disabled_until_timestep=0.86`, dynamic NVFP4 linear-layer quantization, and the INT8/FP8 SAGE
recipe. This command reproduces the draft's dynamic path, not the static ModelOpt checkpoint that
would need to be characterized separately.

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
**1.40×**, while the fastest measured point reaches **1.51×**. These results are scoped to the
dynamic-quantization path measured here and should not be generalized to static checkpoints.

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
