*Generated: 2026-11-11*
# MiniMax-H3 NVFP4 scenario on SM107

## TL;DR

Select the optimized MiniMax-H3 pipeline explicitly with `visual_gen_scenario: require`.

- Use a pre-quantized NVFP4 checkpoint, SM107, per-tensor FP8 attention and one or four GPUs.
- Set the 32 recipe fields through `minimax_h3_scenario`; the process keeps one frozen configuration.
- Build with the shipped FMHA cubins and install the recipe's cuDNN environment before launching.

## Contents

1. [Environment](#1-environment)
2. [Run](#2-run)
3. [Settings](#3-settings)
4. [Implementation and tests](#4-implementation-and-tests)

## 1. Environment

Use the repository's supported CUDA build environment and build a wheel from this tree after a full `git lfs pull`.
The scenario requires the SM107a FMHA resources shipped with the source.

The reproduction environment uses cuDNN **9.28.1.4**; this recipe pin is separate from the repository's global requirements.
Install it into the runtime environment and make its libraries visible before starting Python:

```bash
python -m pip install nvidia-cudnn-cu13==9.28.1.4
export LD_LIBRARY_PATH="$(python -c 'import nvidia.cudnn; print(next(iter(nvidia.cudnn.__path__)) + "/lib")'):${LD_LIBRARY_PATH:-}"
python -c 'from importlib.metadata import version; print(version("nvidia-cudnn-cu13"))'
```

For four GPUs on one node, the reproduction environment also sets:

```bash
export NCCL_PROTO=Simple NCCL_MIN_CTAS=64 NCCL_MAX_CTAS=64
```

## 2. Run

Save this as `minimax-h3-sm107.yaml`:

```yaml
attention_config:
  backend: CUDNN
  quant_attention_config: {qk_dtype: fp8, v_dtype: fp8}
torch_compile_config: {enable: true}
cuda_graph_config: {enable: false}
parallel_config: {ulysses_size: 1}
pipeline_config:
  visual_gen_scenario: require
```

Run from the repository root:

```bash
python examples/visual_gen/models/minimax_h3.py \
  --model <approved-prequantized-NVFP4-checkpoint> \
  --visual_gen_args minimax-h3-sm107.yaml --prompt "<prompt>"
```

To serve the scenario, use the configuration files in `examples/visual_gen/configs/` and the request files in `examples/visual_gen/requests/`:

```bash
trtllm-serve <approved-prequantized-NVFP4-checkpoint> \
  --visual_gen_args examples/visual_gen/configs/minimax-h3-nvfp4-sm107-4gpu.yaml --port 8000

curl -s http://localhost:8000/v1/videos/sync -H "Content-Type: application/json" \
  -d @examples/visual_gen/requests/minimax-h3-768x1344x124.json -o out.mp4

python -m tensorrt_llm.serve.scripts.benchmark_visual_gen \
  --workload examples/visual_gen/requests/minimax-h3-768x1344x124-workload.yaml --port 8000
```

`minimax-h3-nvfp4-sm107-1gpu.yaml` is the one-GPU variant; both warm up 768 × 1344 pixels at 124 frames.

The example uses 544 × 960 pixels, 124 frames and seed 42; these are example defaults, not a performance protocol.
For four GPUs, set `parallel_config: {ulysses_size: 4}` and use the same command on one node with four visible GPUs.

Keep the checkpoint's exported quantization configuration: do not add `quant_config: {quant_algo: NVFP4}`.
That override selects dynamic activation quantization and discards the checkpoint's exclusion list.

| Selection | Behavior |
|---|---|
| `off` (default) | Family pipeline; scenario modules remain unloaded. |
| `auto` | Scenario when every routing requirement holds; otherwise family pipeline. |
| `require` | Scenario or an error naming the unmet requirement. |

The route also requires `dynamic_weight_quant=False`, `tp_size=1` and `async_ulysses=False`.
It serves Ulysses 1 and 4; other sizes are rejected under `require`.

## 3. Settings

Overrides belong in the YAML's `pipeline_config` mapping:

```yaml
pipeline_config:
  visual_gen_scenario: require
  minimax_h3_scenario: {skip_corr_threshold: 0.0}
```

The first valid selection freezes the process configuration; an equal later selection returns the same object.
A differing selection raises with the changed field names, so run A/B arms in separate processes.

Importing a consumer before selection binds the defaults; select through the pipeline before importing scenario consumers.
Unknown settings and combinations checked by `ScenarioConfig.validate()` raise before changing the selected state; tuple fields accept YAML lists.
Static scale mode additionally needs `fp8_static_scales`; the kernel checks that path when it consumes the table.

<details>
<summary>All 32 settings and their source defaults</summary>

**One source.** This table is generated from [`ScenarioConfig`](../../tensorrt_llm/_torch/visual_gen/models/minimax_h3/minimax_h3_nvfp4__sm_107__u1_u4/config.py).

| Field | Default |
|---|---|
| `attention_fmha` | `'trtllm-gen-fp16sm'` |
| `skip_corr_threshold` | `1.0` |
| `strided_receive` | `True` |
| `packed_a2a` | `True` |
| `out_a2a_fp4` | `True` |
| `fp8_scale_fusion` | `'qkstatic'` |
| `fp8_static_scales` | `None` |
| `fp8_static_margin` | `1.0` |
| `fp8_fusion_split` | `True` |
| `fp8_fusion_block_h` | `4` |
| `fp8_fusion_warps` | `2` |
| `grouped_cast` | `True` |
| `vamax_fold` | `True` |
| `amax_allreduce` | `'symm'` |
| `fused_quant` | `True` |
| `fused_quant_warps` | `4` |
| `swiglu_tile` | `(8, 512, 4)` |
| `out_quant` | `True` |
| `out_quant_tile` | `(16, 512, 4)` |
| `nvfp4_gemm_backends` | `('cublaslt',)` |
| `vae_fp8` | `True` |
| `vae_fp8_tiled` | `True` |
| `vae_fused_norm` | `True` |
| `vae_fp8_tile` | `(16, 1024, 8)` |
| `vae_fp8_qtile` | `None` |
| `vae_tile` | `(288, 272, 32)` |
| `vae_graph_max_shapes` | `2` |
| `text_encoder_graph` | `True` |
| `text_encoder_graph_max` | `8` |
| `preloop_device_plan` | `True` |
| `preloop_async_copies` | `True` |
| `preloop_noise_early` | `True` |

- **Attention.** `trtllm-gen-fp16sm` requires `fp8_scale_fusion="qkstatic"`; the default threshold is 1.0.
- **Scale exchange.** `amax_allreduce="symm"` uses symmetric memory on the Ulysses group; `nccl` selects the ordinary max reduction.
- **Decoder tiles.** `vae_fp8_qtile=None` resolves to `(32, 8, 8)` at U1 and `(64, 4, 8)` at U4.
- **NVFP4 GEMMs.** `nvfp4_gemm_backends=('cublaslt',)` gives the transformer's NVFP4 Linears one backend, so each GEMM shape has one output; the AutoTuner keys its entries on the tuple. The family's tuple `[cutlass, cublaslt, cutedsl, cuda_core]` is the comparison arm.
- **Validation.** `vae_fused_norm=True` requires `vae_fp8_tiled=True`; `skip_corr_threshold` must lie in `[0, 8]`.

</details>

## 4. Implementation and tests

| Component | Location |
|---|---|
| Routing and requirements | [`scenarios.py`](../../tensorrt_llm/_torch/visual_gen/models/minimax_h3/scenarios.py) |
| Recipe attention, blocks and transformer | [`modeling.py`](../../tensorrt_llm/_torch/visual_gen/models/minimax_h3/minimax_h3_nvfp4__sm_107__u1_u4/modeling.py) |
| Pre-loop, text graph and VAE hooks | [`pipeline.py`](../../tensorrt_llm/_torch/visual_gen/models/minimax_h3/minimax_h3_nvfp4__sm_107__u1_u4/pipeline.py) |
| Recipe Triton kernels | [`kernels/`](../../tensorrt_llm/_torch/visual_gen/models/minimax_h3/minimax_h3_nvfp4__sm_107__u1_u4/kernels/) |
| Shared dense FP8 op | [`denseFp8FmhaOp.cpp`](../../cpp/tensorrt_llm/thop/denseFp8FmhaOp.cpp) |
| Scenario tests | [`tests/`](../../tests/unittest/_torch/visual_gen/models/minimax_h3/minimax_h3_nvfp4__sm_107__u1_u4/) |

From `tests/unittest`, run the CPU resolver contract separately from the hardware tests:

```bash
pytest -m cpu_only _torch/visual_gen/models/minimax_h3/minimax_h3_nvfp4__sm_107__u1_u4/test_minimax_h3_scenario_resolver.py
pytest _torch/visual_gen/models/minimax_h3/minimax_h3_nvfp4__sm_107__u1_u4
pytest _torch/visual_gen/kernels/parallel/test_fmha_dense_fp8.py
```

The latter two commands require a built wheel and the supported GPU; the scenario consumer tests require SM107.
These commands specify checks to run; this guide does not claim they have passed on a particular build.
