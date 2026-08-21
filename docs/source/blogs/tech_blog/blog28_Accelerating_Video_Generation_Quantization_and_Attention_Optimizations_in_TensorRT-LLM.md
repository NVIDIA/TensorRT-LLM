# Accelerating Video Generation: Quantization and Attention Optimizations in TensorRT-LLM

By NVIDIA TensorRT-LLM Team

## Introduction

Video diffusion transformers repeatedly process long spatiotemporal token sequences across many
denoising steps. Linear layers and attention therefore dominate much of the compute, making them
the two most direct targets for reducing generation latency.

Figure 1 breaks down pipeline-forward time for Wan 2.2 T2V-A14B on a single NVIDIA B200. For an
81-frame, 1280×720 video with 50 denoising steps, attention and linear-layer GEMMs account for
71.7% and 20.8% of compiled BF16 pipeline-forward time, respectively.

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
Skip Softmax reduces pipeline-forward latency from **525.0 to 374.9
seconds**, a **1.40× speedup**. The central question is how to recover that time without giving
up more visual quality than the application can tolerate.

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
instead of reducing the sweep to a few hand-picked points. Squares mark runs without Skip Softmax;
stars mark the conservative configuration (`target_sparsity=0.75`,
`disabled_until_timestep=0.86`); and triangles mark the aggressive end of the sweep
(`target_sparsity=0.75`, `disabled_until_timestep=1.00`). The remaining sweep points are circles,
and the dashed line traces the global Pareto frontier.

The common speedup baseline is compiled BF16 with dense attention and Skip Softmax disabled. Its
mean pipeline-forward latency is **525.0 seconds** across the seven prompts, so a point at
1.40× corresponds to roughly 375 seconds under the same measurement.

<p align="center">
  <img src="../media/tech_blog28_quality_speed_frontier.png" alt="Scatter plot of speedup versus mean LPIPS for the Wan 2.2 optimization sweep, with squares for runs without Skip Softmax, stars for conservative configurations, triangles for aggressive configurations, and a dashed global Pareto frontier" width="1080">
</p>

<p align="center"><sub><em>Figure 2. Speedup–quality frontier across all 96
configurations. LPIPS is measured against eager BF16; speedup is relative to compiled dense
BF16.</em></sub></p>

### Frontier analysis

The table below isolates the points used in the analysis. The six runs without Skip Softmax show
how GEMM and attention quantization move the operating point. The final three runs hold NVFP4 +
SAGE fixed and expose the two Skip Softmax controls. Skip Softmax values are shown as
`target_sparsity / disabled_until_timestep`. Latency is mean pipeline-forward time across the
seven prompts, and LPIPS is measured against the eager BF16 outputs.

| GEMM/attention quantization | Skip Softmax (`target_sparsity / disabled_until_timestep`) | Latency (s) | Speedup | Mean LPIPS |
| :--- | :--- | ---: | ---: | ---: |
| BF16 | Not enabled | 525.0 | 1.000× | 0.1181 |
| BF16 + SAGE | Not enabled | 486.2 | 1.080× | 0.2562 |
| FP8 block | Not enabled | 489.2 | 1.073× | 0.4249 |
| FP8 block + SAGE | Not enabled | 449.3 | 1.169× | 0.3971 |
| NVFP4 | Not enabled | 471.6 | 1.113× | 0.5042 |
| NVFP4 + SAGE | Not enabled | 409.7 | 1.281× | 0.5057 |
| NVFP4 + SAGE | 0.65 / 0.86 | 374.9 | 1.401× | 0.5043 |
| NVFP4 + SAGE | 0.75 / 0.86 | 369.5 | 1.421× | 0.5076 |
| NVFP4 + SAGE | 0.75 / 1.00 | 348.0 | 1.509× | 0.5234 |

The first large quality shift comes from dynamic GEMM quantization. BF16, FP8 block scaling, and
NVFP4 form distinct LPIPS bands, with lower-precision GEMMs moving progressively higher in the
plot. These runs quantize eligible weights while loading the model. Offline static quantization
with ModelOpt can calibrate scales and protect sensitive layers, leaving substantial room to
improve quality at the same numeric format.

SAGE attention has a smaller quality impact than the dynamic GEMM conversion. The separation
between the BF16, FP8, and NVFP4 bands is much larger than the separation between SAGE and
non-SAGE points inside each band, while SAGE consistently moves the operating point toward higher
speedup.

Skip Softmax then fills the space within each family. `target_sparsity` controls how much work can
be rejected, while `disabled_until_timestep` controls how early that rejection begins. Together
they provide a continuum between the conservative stars and aggressive triangles instead of a
single all-or-nothing sparse mode.

This layered structure explains the Pareto frontier. Almost every frontier point enables Skip
Softmax, and every FP8 or NVFP4 point on the frontier combines it with SAGE. Dynamic GEMM
quantization chooses the broad quality band; SAGE and Skip Softmax then recover speed within that
band.

Figure 3 adds a visual check across all seven prompts. For each GEMM/attention family, the top row
uses the conservative Skip Softmax configuration and the bottom row uses the aggressive Skip
Softmax configuration. The eager BF16 output at the left provides a common reference.

<p align="center">
  <img src="../media/tech_blog28_visual_comparison_p01_cat_garden.jpg" alt="First-frame comparison for a cat in a sunlit garden, with an eager BF16 reference and conservative and aggressive Skip Softmax results across six GEMM and attention configurations" width="1080">
</p>

<p align="center">
  <img src="../media/tech_blog28_visual_comparison_p03_park_kids.jpg" alt="First-frame comparison for children in a park, with an eager BF16 reference and conservative and aggressive Skip Softmax results across six GEMM and attention configurations" width="1080">
</p>

<p align="center">
  <img src="../media/tech_blog28_visual_comparison_p04_drone_coast.jpg" alt="First-frame comparison for a coastal drone shot, with an eager BF16 reference and conservative and aggressive Skip Softmax results across six GEMM and attention configurations" width="1080">
</p>

<p align="center">
  <img src="../media/tech_blog28_visual_comparison_p05_neon_sign.jpg" alt="First-frame comparison for a neon OPEN sign, with an eager BF16 reference and conservative and aggressive Skip Softmax results across six GEMM and attention configurations" width="1080">
</p>

<p align="center">
  <img src="../media/tech_blog28_visual_comparison_p06_woman_smile.jpg" alt="First-frame comparison for a studio portrait, with an eager BF16 reference and conservative and aggressive Skip Softmax results across six GEMM and attention configurations" width="1080">
</p>

<p align="center">
  <img src="../media/tech_blog28_visual_comparison_p07_horse_gallop.jpg" alt="First-frame comparison for a galloping racehorse, with an eager BF16 reference and conservative and aggressive Skip Softmax results across six GEMM and attention configurations" width="1080">
</p>

<p align="center">
  <img src="../media/tech_blog28_visual_comparison_p10_market.jpg" alt="First-frame comparison for a street market, with an eager BF16 reference and conservative and aggressive Skip Softmax results across six GEMM and attention configurations" width="1080">
</p>

<p align="center"><sub><em>Figure 3. First-frame comparison across all seven prompts. Conservative
Skip Softmax uses `target_sparsity=0.75` and `disabled_until_timestep=0.86`; aggressive Skip
Softmax uses the same target sparsity and `disabled_until_timestep=1.00`.</em></sub></p>

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

### Agentic optimization (TODO)

Finding a useful speed-quality tradeoff by hand is laborious: every candidate must be generated
across multiple prompts, timed, compared with a quality reference, and folded back into the
frontier. **TODO:** add an agentic workflow that launches these experiments, analyzes the frontier,
and proposes the next configurations to evaluate.

## Conclusion

Accelerating video diffusion is not a single precision switch. It is a process of deciding where
the pipeline can tolerate lower precision and where it can safely avoid work altogether.
TensorRT-LLM exposes linear-layer quantization, quantized attention, and Skip Softmax as
composable controls, so deployments can choose an operating point that matches their own quality
bar instead of inheriting one fixed recipe.

That operating point is model-, prompt-, and hardware-dependent. A useful optimization workflow
therefore combines representative prompts, deployment-relevant latency, aggregate quality
metrics, and direct inspection of generated videos. Offline calibration with ModelOpt and a more
automated frontier search are natural next steps for making this process both more accurate and
less labor-intensive.

These single-GPU techniques are also orthogonal to the scale-out methods in
[Scaling Video Generation Across NVL72 Rack with TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/blogs/tech_blog/blog25_Scaling_Video_Generation_Across_NVL72_Rack_with_TensorRT-LLM.md),
and can be combined with multi-GPU parallelism when the deployment requires higher throughput or
larger workloads.

## References

1. [Scaling Video Generation Across NVL72 Rack with TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/blogs/tech_blog/blog25_Scaling_Video_Generation_Across_NVL72_Rack_with_TensorRT-LLM.md)
2. [SageAttention: Accurate 8-Bit Attention for Plug-and-play Inference Acceleration](https://arxiv.org/abs/2410.02367)
3. [SageAttention2: Efficient Attention with Thorough Outlier Smoothing and Per-thread INT4 Quantization](https://arxiv.org/abs/2411.10958)
4. [NVIDIA Model Optimizer Diffusers Quantization Example](https://github.com/NVIDIA/Model-Optimizer/tree/main/examples/diffusers)
5. [BLASST: Dynamic BLocked Attention Sparsity via Softmax Thresholding](https://arxiv.org/abs/2512.12087)
