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
seconds**, a **1.40× speedup**. In this blog, we introduce these optimization techniques and
discuss the tradeoff between quality and speed.

## Table of Contents

- [Quantization](#quantization)
- [Attention Optimizations](#attention-optimizations)
- [Results](#results)
- [Reproduction](#reproduction)
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
how weights are prepared; activations are quantized as the model runs. The FP8 blockwise and
NVFP4 results below dynamically quantize eligible BF16 weights at load time. The ModelOpt
metadata stored with the checkpoint calibrates Skip Softmax; it does not prequantize those
weights. NVFP4 pushes precision lower for more throughput.

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
| Checkpoint | [Wan-AI/Wan2.2-T2V-A14B-Diffusers](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B-Diffusers) BF16 weights with ModelOpt-generated Skip Softmax metadata for both transformers |
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

Skip Softmax introduces an explicit quality-speed tradeoff. Figure 2 maps every configuration
instead of reducing the sweep to a few hand-picked points. Circles are skip-enabled runs, stars
are the no-skip anchors for each precision family, and the dashed line traces the global Pareto
frontier. Hover over a point for its exact configuration and use the filters to isolate a family,
target sparsity, or denoising cutoff.

<iframe
  src="../../_static/tech_blog28_quality_speed_frontier.html"
  title="Interactive quality-speed frontier for the Wan 2.2 optimization sweep"
  loading="lazy"
  sandbox="allow-scripts"
  style="width: 100%; height: 660px; border: 0;"
></iframe>

<p align="center"><sub><em>Figure 2. Interactive speedup–quality frontier across all 96
configurations. LPIPS is measured against eager BF16; speedup is relative to compiled dense
BF16.</em></sub></p>

<p><a href="../../_static/tech_blog28_quality_speed_frontier.html" target="_blank"
rel="noopener noreferrer">Open the interactive frontier in a separate page</a>.</p>

The plot separates into three broad bands. BF16 stays closest to the eager reference. FP8 block
scaling occupies the middle of the quality-speed range, while NVFP4 reaches the highest speedups
with a larger shift from BF16. SAGE moves every precision family to the right, and Skip Softmax
then fills out the local frontier within that family. The denoising cutoff drives most of the
vertical spread, especially for BF16; target sparsity is the finer adjustment.

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

## Reproduction

The [Wan 2.2 Skip Softmax example](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/visual_gen/skip_softmax/wan2.2-t2v-a14b)
contains the two ModelOpt calibration overlays and the VisualGen configuration for the 1.40×
NVFP4 + SAGE result. Start with a local copy of the public BF16 checkpoint and apply the overlays
to its high-noise and low-noise transformer configs:

```bash
export MODEL_DIR=/path/to/Wan2.2-T2V-A14B-Diffusers
export TRTLLM_ROOT=/path/to/TensorRT-LLM
export EXAMPLE_DIR="$TRTLLM_ROOT/examples/visual_gen/skip_softmax/wan2.2-t2v-a14b"

python "$EXAMPLE_DIR/apply_calibration.py" --model-dir "$MODEL_DIR"
```

The helper adds only the calibrated Skip Softmax metadata. The packaged `visual_gen.yaml` then
selects dynamic NVFP4 linear-layer quantization, SAGE attention, `target_sparsity=0.65`, and
`disabled_until_timestep=0.86`.

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

Generate the gallery prompt through the public VisualGen API:

```bash
python - <<'PY'
import os

from tensorrt_llm import VisualGen, VisualGenArgs

model_dir = os.environ["MODEL_DIR"]
example_dir = os.environ["EXAMPLE_DIR"]
generator = VisualGen(
    model=model_dir,
    args=VisualGenArgs.from_yaml(f"{example_dir}/visual_gen.yaml"),
)
params = generator.default_params
params.height = 720
params.width = 1280
params.num_frames = 81
params.frame_rate = 16
params.num_inference_steps = 50
params.guidance_scale = 5.0
params.max_sequence_length = 512
params.seed = 1004
params.extra_params = {**(params.extra_params or {}), "guidance_scale_2": 4.0}

output = generator.generate(
    inputs=(
        "Drone shot flying over a rugged coastline at sunset, waves crashing on cliffs below, "
        "golden hour lighting"
    ),
    params=params,
)
output.save("wan22_nvfp4_sage_skip.mp4")
PY
```

<details>
<summary>Seven-prompt evaluation manifest</summary>

| ID | Seed | Prompt |
| :--- | ---: | :--- |
| `p01_cat_garden` | 1001 | A cat walking through a sunlit garden, gentle breeze rustling leaves, slow tracking shot |
| `p03_park_kids` | 1003 | Children playing in a busy park, a golden retriever running between them, sunny afternoon, wide shot |
| `p04_drone_coast` | 1004 | Drone shot flying over a rugged coastline at sunset, waves crashing on cliffs below, golden hour lighting |
| `p05_neon_sign` | 1005 | A neon sign reading 'OPEN' flickering in a rainy alley at night, reflections on wet pavement, cinematic |
| `p06_woman_smile` | 1006 | A young woman smiling at the camera, soft studio lighting, slight head tilt, cinematic close-up portrait |
| `p07_horse_gallop` | 1007 | A racehorse galloping on a dirt track, kicking up dust, side tracking shot, dramatic lighting |
| `p10_market` | 1010 | A bustling outdoor street market with people walking and vendors selling fresh fruit, Mediterranean style, midday sun |

</details>

For the reported latency, each prompt first runs one untimed 50-step generation to compile both
Wan transformer stages. After CUDA synchronization, a second 50-step pipeline forward is timed
and synchronized; the seven prompt times are then averaged. Video encoding is outside this timed
region. The eager BF16 quality reference disables compilation, quantization, SAGE, and Skip
Softmax. AlexNet LPIPS is computed between corresponding frames, averaged over all 81 frames and
then over the seven prompts.

The sweep ran on B200 with a TensorRT-LLM rc19-era source build at commit `e1135bbdfa`, including
the Wan precision-cast change from [PR #15318](https://github.com/NVIDIA/TensorRT-LLM/pull/15318).
Use the same build to match the reported numbers; newer releases can produce different compile
paths and timing.

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
