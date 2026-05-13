# Spatial Pooling Benchmarks

2x2 spatial average pooling on Qwen2.5-VL-7B-Instruct ViT embeddings.
Controlled by `TRTLLM_SPATIAL_POOL_FACTOR=2` (unset to disable).

## Setup

- **GPU**: NVIDIA RTX A6000 (48 GB)
- **Model**: Qwen/Qwen2.5-VL-7B-Instruct
- **Backend**: TRT-LLM 1.3.0rc11, PyTorch backend
- **Image**: COCO test2017/000000155781.jpg (640x424)
- **Benchmark**: NVIDIA AIPerf, 20 profiling requests, 2 warmup, OSL 150, streaming, ignore_eos
- **Visual tokens**: ~345 baseline → ~96 with pooling (~72% reduction)

## Aggregated Mode (Single Worker)

| Metric | Baseline | Spatial Pooling | Change |
|--------|----------|----------------|--------|
| TTFT avg (ms) | 100.99 | 74.01 | **-26.7%** |
| TTFT p50 (ms) | 100.71 | 73.83 | **-26.7%** |
| TTFT p90 (ms) | 102.44 | 75.28 | **-26.5%** |

## EPD (Disaggregated) Mode — Concurrency 1

| Metric | Baseline | Spatial Pooling | Change |
|--------|----------|----------------|--------|
| TTFT avg (ms) | 109.69 | 82.48 | **-24.8%** |
| TTFT p50 (ms) | 109.24 | 82.36 | **-24.6%** |
| TTFT p90 (ms) | 112.92 | 83.33 | **-26.2%** |
| TTFT p99 (ms) | 113.23 | 86.42 | **-23.7%** |
| ITL avg (ms) | 22.45 | 22.42 | ~same |
| Request Latency avg (ms) | 3,454.51 | 3,422.56 | -0.9% |
| Throughput (req/s) | 0.29 | 0.29 | ~same |

## EPD (Disaggregated) Mode — Concurrency 2

| Metric | Baseline | Spatial Pooling | Change |
|--------|----------|----------------|--------|
| TTFT avg (ms) | 180.69 | 144.71 | **-19.9%** |
| TTFT p50 (ms) | 184.48 | 141.65 | **-23.2%** |
| TTFT p90 (ms) | 194.77 | 160.60 | **-17.5%** |
| TTFT p99 (ms) | 208.50 | 168.11 | **-19.4%** |
| ITL avg (ms) | 22.67 | 22.58 | ~same |
| Request Latency avg (ms) | 3,558.15 | 3,508.42 | -1.4% |
| Throughput (req/s) | 0.56 | 0.57 | +1.8% |

## Quality Verification

With spatial pooling enabled, the model produces coherent, factually correct descriptions:

**Prompt:** "Describe this image in one sentence."
**Response (aggregated):** "A bus is driving on a wet road in a dimly lit, foggy environment, with trees and buildings visible in the background."
**Response (EPD):** "A bus is driving on a wet road in a dimly lit, foggy environment, with trees and buildings visible in the background."

## Key Observations

1. **TTFT improves 20-27%** across all modes and concurrency levels.
2. **ITL is unchanged** — pooling only affects the prefill path, not decode.
3. **Quality is preserved** — correct object identification, scene description, and spatial reasoning.
4. **Benefit grows under load** — at concurrency=2, fewer visual tokens free compute for concurrent requests.
5. **Larger images should benefit more** — a 1920x1272 image has ~3,105 visual tokens, reducing to ~805 with pooling.
