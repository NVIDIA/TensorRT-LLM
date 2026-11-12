<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# TurboQuant Cold-Page KV-Cache Compression (Prototype)

TurboQuant cold-page compression stores supported Attention KV buffers with
[TurboQuant](https://arxiv.org/abs/2504.19874) at 2 to 4 bits per value while
their Pages reside in a Host or Disk cache tier. Like
[NVFP4 cold-page compression](nvfp4_cold_page.md), it is a storage-boundary
optimization: the active GPU cache and Attention kernels continue to use the
model's normal runtime KV type.

For an overview of all available compression methods, see
[KV Cache Compression](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/features/kv-cache-compression.md).

## Motivation

A cold tier that holds more reusable prefixes avoids recomputing them. NVFP4
cold Pages need 4.5 bits per value; TurboQuant reaches 2 to 4 bits per value,
plus a few bytes of scale per vector, so the same Host or Disk quota holds up
to about twice as many Pages and each offload or onboard moves fewer bytes. The
benefit is largest for workloads with substantial prefix reuse and real
cold-tier pressure, and it comes with a larger, setting-dependent quantization
error that must be validated for the workload.

## Feature Behavior

TurboQuant is data-oblivious: it needs no calibration and no checkpoint
metadata. For every quantized K or V vector, one per token and KV head, the
encoder rotates the vector with a random orthogonal matrix, replaces each
rotated coordinate with the index of the nearest value of a Lloyd-Max codebook,
and stores one BF16 scale for the vector. The decoder looks up the codebook
values, applies the scale, and rotates back.

| Setting | Algorithm | Use it for |
| --- | --- | --- |
| `turboquant_mode: mse`, 2, 3, or 4 bits | The paper's Algorithm 1: the smallest reconstruction error for the bit width | The default; K and V reconstruction |
| `turboquant_mode: mse`, 2.5 or 3.5 bits | Two Algorithm 1 instances per vector: in each group of vectors that share a page, layer, buffer, and KV head, half of the channels use one bit more than the other half; channels with more than four times the median channel energy are always in that half | Bit widths between the integer settings, with outlier channels kept more precise |
| `turboquant_mode: prod`, 2, 3, or 4 bits | The paper's Algorithm 2: `mse` with one bit fewer, plus a 1-bit QJL sign of the residual and the residual norm | Unbiased estimates of the inner products between queries and decoded keys, at a larger reconstruction error |

The fractional settings choose the high-bit half from the channel energies in
the Page being encoded: the outlier channels, whose energy exceeds four times
the median channel energy, and then the lowest-numbered other channels until
half of the channels are chosen. The split adapts to every Page without
calibration. The rotation
and QJL matrices are fixed by `turboquant_seed` and generated identically in
every process.

Each cold Page is one compact storage blob. For conventional MHA, MQA, and GQA,
the K and V buffers are encoded; for key-only MLA, the latent Attention key is
encoded, including widths that are not a power of two, such as 576 values.
Values that are not quantized, such as RoPE when `skip_rope_quantization` is
`true`, are copied at the hot-cache precision into the same blob, and auxiliary
roles in the same Attention lifecycle, such as a DSA index key, are appended
losslessly. Non-Attention lifecycles, including GDN, SSM, and Conv state, use
the default lossless cold-page codec.

Disk migration uses pinned Host staging, exactly as for NVFP4: the Host and
Disk tiers share one TurboQuant representation.

| `turboquant_mode` | `turboquant_bits` | Bits per stored value, head dimension 128[^bits] | BF16 Page compression |
| --- | --- | --- | --- |
| `mse` | `2`, `2.5`, `3`, `3.5`, `4` | 2.125, 2.78, 3.125, 3.78, 4.125 | 7.5x, 5.8x, 5.1x, 4.2x, 3.9x |
| `prod` | `2`, `3`, `4` | 2.25, 3.25, 4.25 | 7.1x, 4.9x, 3.8x |
| NVFP4, for comparison | | 4.5 | 3.6x |

[^bits]: Including per-vector scales and residual norms and, for 2.5 and 3.5
    bits, the channel masks at the default 32 tokens per Page. The cold-page
    size formula is in the
    [feature documentation](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/features/kv-cache-compression.md#turboquant-cold-pages).

This is different from an active quantized KV cache. Do not set
`kv_cache_config.dtype: nvfp4` for this feature: TurboQuant decodes into an
FP16, BF16, or FP8 active KV cache, and TensorRT-LLM rejects the configuration
when the active KV cache is NVFP4. Cold-page compression does not change Page
identity, token count, block reuse, scheduling, or the Attention-visible GPU
layout.

## Supported Cache Types

| Cache type | Cold-page behavior |
| --- | --- |
| MHA, MQA, and GQA Attention KV | Supported; K and V are encoded with TurboQuant |
| Key-only MLA Attention KV | Supported; the latent Attention key is encoded with TurboQuant |
| GDN, SSM, and Conv state | Skipped by quantization and preserved losslessly |
| DSA and other auxiliary buffers | Skipped by quantization and preserved losslessly |
| DeepSeek-V4 CSA cache | Supported; the 448 NoPE values of each row are encoded with TurboQuant, and the 64 RoPE values and the indexer cache are preserved losslessly |
| DeepSeek-V4 SWA, HCA, and compressor state | Preserved losslessly |

The current implementation requires the PyTorch backend, the native C++
KVCacheManagerV2, and a GPU with compute capability 8.0 or newer; its kernels
are written in Triton. Hot Attention KV can use FP16, BF16, or FP8. KV-cache
block reuse remains supported because Page and token identity are unchanged.
One-model MTP-EAGLE and EAGLE3 are supported: target and draft KVCMs encode
their cold Pages independently, and draft cold Pages always quantize whole
vectors. HELIX context parallelism is not currently supported.

Set `kv_cache_config.use_kv_cache_manager_v2: true` explicitly. A nonzero Host
or Disk cache is also required for Pages to cross a compression boundary.

### Tested Models

- Qwen3.8-Flash-Next (GQA with partial RoPE, QSA sparse attention, GDN layers)
- GLM-5.3-Flash (kpool sparse MLA latent, KDA layers)

This is a tested-model list, not an exhaustive support list. Other models that
use the supported cache types above are expected to work, subject to their
checkpoint, parallelism, Attention-backend, and hardware requirements.

## Qwen3-8B on One GPU

The following example uses a regular BF16 checkpoint. Attention KV stays BF16
on the GPU and uses 3-bit TurboQuant only while it resides in the 8 GiB Host
tier.

```python
from tensorrt_llm import LLM, SamplingParams
from tensorrt_llm.llmapi import (
    ColdPageQuantizationCompressionConfig,
    KvCacheConfig,
)

with LLM(
    model="Qwen/Qwen3-8B",
    backend="pytorch",
    max_seq_len=4096,
    max_batch_size=4,
    kv_cache_config=KvCacheConfig(
        use_kv_cache_manager_v2=True,
        dtype="auto",
        enable_block_reuse=False,
        host_cache_size=8 << 30,
    ),
    kv_cache_compression_config=ColdPageQuantizationCompressionConfig(
        quant="turboquant",
        turboquant_bits=3,
        turboquant_mode="mse",
    ),
) as llm:
    outputs = llm.generate(
        ["Explain why prefix caching helps agentic workloads."],
        SamplingParams(max_tokens=128, temperature=0.0),
    )
    print(outputs[0].outputs[0].text)
```

The equivalent `trtllm-serve` configuration is:

```yaml
backend: pytorch

kv_cache_config:
  use_kv_cache_manager_v2: true
  dtype: auto
  enable_block_reuse: false
  host_cache_size: 8589934592  # 8 GiB

kv_cache_compression_config:
  algorithm: quantization_for_cold_page
  quant: turboquant
  turboquant_bits: 3     # 2, 2.5, 3, 3.5, or 4
  turboquant_mode: mse   # mse or prod; prod supports 2, 3, and 4 bits
```

```bash
trtllm-serve Qwen/Qwen3-8B --config qwen3-cold-turboquant.yaml
```

The same configuration is available from the LLM API example script:

```bash
python examples/llm-api/llm_kv_cache_compression.py \
  --compression-method turboquant-cold-page \
  --model Qwen/Qwen3-8B \
  --turboquant-bits 3 \
  --turboquant-mode mse
```

This smoke test disables block reuse and covers basic inference and
configuration; a short request can remain entirely on the GPU. To exercise the
cold tier, use the workload in the next section.

## Verify Activation

Configuration alone does not prove that a Page crossed the compression
boundary. The following script keeps the GPU KV cache small, enables block
reuse, pushes a set of long prefixes out to the Host tier, and then asks new
questions about the first prefixes so that their Pages are onboarded and
decoded again. It prints the KV-cache iteration statistics that show the
movement.

```python
from tensorrt_llm import LLM, SamplingParams
from tensorrt_llm.llmapi import (
    ColdPageQuantizationCompressionConfig,
    KvCacheConfig,
)

documents = [
    f"Document {index}: " + " ".join(f"fact-{index}-{word}" for word in range(400))
    for index in range(8)
]
sampling = SamplingParams(max_tokens=16, temperature=0.0)

with LLM(
    model="Qwen/Qwen3-8B",
    backend="pytorch",
    max_seq_len=4096,
    max_batch_size=1,
    enable_iter_perf_stats=True,
    kv_cache_config=KvCacheConfig(
        use_kv_cache_manager_v2=True,
        enable_block_reuse=True,
        max_tokens=8192,  # a GPU tier that the documents overflow
        host_cache_size=8 << 30,
    ),
    kv_cache_compression_config=ColdPageQuantizationCompressionConfig(
        quant="turboquant",
        turboquant_bits=4,
    ),
) as llm:
    for document in documents:  # fill the GPU tier and offload the oldest prefixes
        llm.generate([document + "\nSummarize the document."], sampling)
    for document in documents[:2]:  # reuse prefixes that now live in the Host tier
        llm.generate([document + "\nWhich fact comes first?"], sampling)

    offload_bytes = onboard_bytes = 0
    cached_tokens = {}
    cold_slot_sizes = set()
    for stats in llm.get_stats(timeout=2):
        for pool in stats.get("kvCacheIterationStatsByPoolGroup", {}).values():
            offload_bytes += pool["iterOffloadBytes"]
            onboard_bytes += pool["iterOnboardBytes"]
        tiers = stats.get("kvCacheLevelTiers", [])
        for tier, tokens in zip(tiers, stats.get("iterCachedTokensByLevel", [])):
            cached_tokens[tier] = cached_tokens.get(tier, 0) + tokens
        for pool in stats.get("kvCacheIterationStatsByColdPoolGroup", {}).values():
            cold_slot_sizes.update(pool["slotSize"])
    print("offload bytes:", offload_bytes, "onboard bytes:", onboard_bytes)
    print("reused prompt tokens by tier:", cached_tokens)
    print("cold slot sizes:", sorted(cold_slot_sizes))
```

Look for these signals:

* Offload bytes (`iterOffloadBytes`) become nonzero while the later documents
  push the earlier prefixes out of the GPU tier, and onboard bytes
  (`iterOnboardBytes`) become nonzero when the last two requests reuse them.
* The reused prompt tokens of the last two requests are attributed to the
  `host` tier (`iterCachedTokensByLevel`, labeled by `kvCacheLevelTiers`).
* The cold slot size (`slotSize` of `kvCacheIterationStatsByColdPoolGroup`) is
  the TurboQuant cold-page size rather than the hot Page size: about 1.2 MiB
  instead of 4.5 MiB for Qwen3-8B (36 layers, 8 KV heads, head dimension 128)
  at 4 bits. This shows that the cold tier holds TurboQuant Pages, not lossless
  copies.

For a scripted version of this check on a multi-GPU model, see
`examples/kv_cache_compression/cold_tier_accuracy.py`: it prints the offload and onboard bytes,
the per-tier reuse, and the cold and hot page sizes, and marks a run invalid when no prefix
block was reused from a cold tier.

With `trtllm-serve`, run enough concurrent or reusable long-context requests to
create GPU KV pressure, then inspect the server's `/metrics` endpoint. Both of
the following counters should become nonzero when the workload offloads and
later reuses cold Pages:

* `trtllm_kv_cache_offload_bytes_total`
* `trtllm_kv_cache_onboard_bytes_total`

The counters cover all KVCM migrations, including lifecycles that use the
lossless fallback. In disaggregated serving, each context or generation worker
owns its own local KVCM and cold-tier quota, so put `kv_cache_config` and
`kv_cache_compression_config` on every worker that uses a local Host or Disk
tier, and check each worker.

## Disk Tier

Add a Disk tier behind the Host tier with a positive `disk_cache_size` and an
existing `disk_cache_path` directory. TensorRT-LLM writes the same TurboQuant
cold Pages to Disk through pinned Host staging.

```yaml
kv_cache_config:
  use_kv_cache_manager_v2: true
  host_cache_size: 8589934592    # 8 GiB
  disk_cache_size: 68719476736   # 64 GiB
  disk_cache_path: /path/to/kv-cache-dir

kv_cache_compression_config:
  algorithm: quantization_for_cold_page
  quant: turboquant
  turboquant_bits: 4
```

## Skipping RoPE Quantization

`skip_rope_quantization` has the same meaning and model support as for NVFP4
cold pages: by default all values are quantized, and with
`skip_rope_quantization: true` the position-encoded (RoPE) part of each K vector
is copied at its hot-cache precision while the rest is encoded with TurboQuant.
RoPE preservation is available for DeepSeek-V4, GLM-5 (`glm_moe_dsa`), and the
Qwen3.5 series; other models log a warning and quantize whole vectors, and
draft-model cold Pages always quantize whole vectors.

```yaml
kv_cache_compression_config:
  algorithm: quantization_for_cold_page
  quant: turboquant
  turboquant_bits: 4
  skip_rope_quantization: true
```

DeepSeek-V4 compressed (CSA) rows always keep their 64 RoPE values lossless with
TurboQuant and quantize the 448 NoPE values, whatever `skip_rope_quantization`
says; there is no TurboQuant counterpart of the NVFP4 2FP4 RoPE format, and
`nvfp4_residual_dim` is rejected with `quant: turboquant`. DeepSeek-V4 requires
`tokens_per_block` of 128 or 256, and cold-page compression does not cover the
FP8 `fp8_ds_mla` footer-scale KV layout.

## Accuracy and Performance

TurboQuant is lossy, and its error depends on the bit width and mode. Measure
the accuracy effect on your model and workload before deployment, and compare
it with NVFP4 cold pages at the same cold-tier quota.

### Reconstruction Quality

`tests/microbenchmarks/kv_cache_compression/turboquant_quality.py` measures the normalized MSE
`||x - x_hat||^2 / ||x||^2` of BF16 K/V vectors with head dimension 128 on structured inputs
("outlier channels" has two channels 10-25x larger than the rest, like real K), and the bias and
variance of inner products with unit query vectors:

| Setting | Paper (d = 128) | Gaussian | Outlier channels | Sparse | One-hot | All-equal | Norms 1e-6..1e6 | Inner-product bias / variance x d |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `mse` 2 | 0.117 | 0.115 | 0.110 | 0.116 | 0.115 | 0.127 | 0.115 | |
| `mse` 2.5 | between | 0.073 | 0.045 | 0.041 | 0.032 | 0.055 | 0.073 | |
| `mse` 3 | 0.034 | 0.034 | 0.032 | 0.034 | 0.034 | 0.035 | 0.033 | |
| `mse` 3.5 | between | 0.021 | 0.012 | 0.011 | 0.009 | 0.017 | 0.021 | |
| `mse` 4 | 0.0094 | 0.0092 | 0.0086 | 0.0092 | 0.0092 | 0.0094 | 0.0091 | |
| `prod` 2 | var 0.56 / d | 0.57 | 0.55 | 0.57 | 0.58 | 0.54 | 0.58 | -2e-3 / 0.56 |
| `prod` 3 | var 0.18 / d | 0.18 | 0.18 | 0.18 | 0.18 | 0.21 | 0.18 | -1e-3 / 0.17 |
| `prod` 4 | var 0.047 / d | 0.053 | 0.051 | 0.054 | 0.054 | 0.072 | 0.053 | -5e-4 / 0.051 |
| NVFP4 | | 0.0091 | 0.0084 | 0.0014 | 0 | 0 | 0.51 | |

The per-vector least-squares scale makes `mse` slightly better than the paper's bound. `prod`
trades reconstruction error (about `pi/2` times the residual of the `b - 1`-bit `mse` part) for
unbiased inner products; its inner-product variance matches the paper's Theorem 2.

Repeated offload and onboard of the same Page re-encodes the decoded values every time
(same tool, outlier-channel input, error after one and after four round trips):

| Setting | 1 round trip | 4 round trips |
| --- | --- | --- |
| `mse` 4 / 3 / 2 | 0.0086 / 0.0314 / 0.1092 | 0.0086 / 0.0314 / 0.1092 |
| `mse` 3.5 / 2.5 | 0.0123 / 0.0448 | 0.0137 / 0.0497 |
| `prod` 4 / 3 / 2 | 0.051 / 0.18 / 0.55 | 0.31 / 1.9 / 16.8 |

Integer-bit `mse` reproduces its codes, so the error does not grow. Fractional bits choose the
outlier channels from the decoded data again and mostly keep them. `prod` is unbiased for one
encoding (in expectation over the QJL projection); a re-encoding adds QJL error again, with the
same seeded projection, so its error grows quickly with the number of round trips. Prefer `mse`
when Pages migrate repeatedly.

### Encode and Decode Time

`tests/microbenchmarks/kv_cache_compression/bench_cold_page_codec.py` times one encode and one
decode call (256 Pages per call unless noted, cold Pages in pinned host memory, median of 20)
for TurboQuant and NVFP4 on the same GPU, and the speed of light (SOL) of the same transfer. On
one GB300, with the pinned memory on the GPU's own CPU socket (the GPU reads it at 211 GB/s and
writes it at 193 GB/s) and on the other socket (125 GB/s and 122 GB/s):

| Case | Geometry and TurboQuant setting | Cold Page KiB, TurboQuant / NVFP4 | Own socket: encode / decode us (vs NVFP4) | Other socket: encode / decode us (vs NVFP4) |
| --- | --- | --- | --- | --- |
| `glm5-bf16-mse4` | GLM-5 MLA latent, 78 x {K} x 1 x 32 x 576, BF16, `mse` 4 | 717 / 790 | 1076 / 1106 (0.98x / 1.11x) | 1601 / 1575 (0.92x / 0.94x) |
| `glm5-bf16-mse3` | same, `mse` 3 | 541 / 790 | 848 / 1213 (0.77x / 1.22x) | 1229 / 1522 (0.72x / 0.91x) |
| `glm5-skiprope-bf16-mse4` | same, `mse` 4, `skip_rope_quantization` (512 values quantized) | 941 / 1014 | 1788 / 1908 (1.28x / 1.50x) | 2049 / 3198 (0.93x / 1.50x) |
| `mla-fp8-mse4` | DeepSeek-V3.2 / Kimi-K3 MLA latent, 61 x {K} x 1 x 32 x 576, FP8, `mse` 4 | 560 / 618 | 799 / 857 (0.94x / 1.10x) | 1242 / 1230 (0.93x / 0.94x) |
| `dsv4-csa-bf16-mse4` | DeepSeek-V4 CSA, 18 x {compressed, indexer} x 1 x 32 x {512, 34}, BF16, `mse` 4 | 237 / 220 | 366 / 375 (0.53x / 1.19x) | 547 / 532 (0.53x / 0.98x) |
| `dsv4-csa-skiprope-bf16-mse4` | same, `skip_rope_quantization` (NVFP4 also keeps RoPE) | 237 / 252 | 365 / 383 (1.03x / 1.17x) | 549 / 536 (1.00x / 0.99x) |
| `llama8b-bf16-mse4` | Llama-3.1-8B GQA, 32 x {K, V} x 8 x 32 x 128, BF16, `mse` 4 | 1056 / 1152 | 1455 / 1334 (0.92x / 0.92x) | 2286 / 2244 (0.92x / 0.93x) |
| `llama8b-bf16-prod4` | same, `prod` 4 | 1088 / 1152 | 1854 / 1380 (1.17x / 0.95x) | 2340 / 2310 (0.94x / 0.96x) |
| `gqa-hd64-bf16-mse4` | GQA, 36 x {K, V} x 8 x 32 x 64, BF16, `mse` 4 | 612 / 648 | 851 / 786 (0.94x / 0.96x) | 1329 / 1320 (0.94x / 0.96x) |
| `glm5-bf16-mse4-8pages` | as `glm5-bf16-mse4`, 8 Pages per call | 717 / 790 | 64 / 83 (1.24x / 1.76x) | 73 / 100 (1.03x / 1.47x) |

Conventional K/V Pages are bound by the host link in both formats (96-100% of SOL), so
TurboQuant takes about the NVFP4 time scaled by the cold-page size; only the `prod` encode, which
also projects the residual, is compute bound on the faster link. MLA latents and DeepSeek-V4
CSA rows need more decode work per byte, because every instance (256 or 448 values) is rotated
back on tensor cores: on the faster link their decode takes 1.1-1.2x the NVFP4 time, while
encode stays at or below it; on the slower link both are link bound and TurboQuant is faster.
With `skip_rope_quantization` the GLM-5 latent is a single 512-value instance, whose decode
accumulates its output in two blocks and reads the codes twice; it takes 1.5x the NVFP4 time.
NVFP4's default DeepSeek-V4 format quantizes RoPE as two FP4 components, which makes its encode
slow; `dsv4-csa-skiprope-bf16-mse4` compares like with like. Small calls are bound by fixed
costs, such as the one decode launch per instance of a 576-value latent. The launch shapes of
rows wider than 256 values were measured on GB300 (SM103) and apply to SM100-family GPUs; other
GPUs use generic shapes. Run the benchmark on your system before relying on these numbers.

### End-to-End Accuracy

`examples/kv_cache_compression/cold_tier_accuracy.py` prefills synthetic prefixes of
name-and-code facts, pushes their Pages to the Host tier with unrelated prompts, and then asks
exact-match questions whose prefixes must be onboarded and decoded again. A run counts only if
the iteration statistics of its question phase show Attention Pages reused from the cold tier
(Disk with `--disk`) and bytes onboarded; they do not show which questions those Pages served.
With TP4/EP4 on four
GB300 GPUs, greedy decoding, two 6144-token prefixes, and 400 questions:

| Model | none | NVFP4 | TQ `mse` 4 | TQ `mse` 3.5 | TQ `mse` 3 | TQ `mse` 2.5 | TQ `prod` 4 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen3.8-Flash-Next-FP8 | 400/400 | 400/400 | 400/400 | 400/400 | 400/400 | 395/400 | 398/400 |
| GLM-5.3-Flash | not run | not run | 400/400 | not run | 397/400 | not run | not run |

## Enablement Checklist

1. Run the PyTorch backend on a GPU with compute capability 8.0 or newer.
2. Select the native C++ KVCacheManagerV2 with
   `use_kv_cache_manager_v2: true`.
3. Keep `kv_cache_config.dtype` at the model's intended runtime KV type:
   `auto`, FP16/BF16, or FP8. An active NVFP4 KV cache is rejected.
4. Provision a nonzero Host cache and, optionally, a Disk cache. A positive
   `disk_cache_size` also requires `disk_cache_path` to name an existing
   directory.
5. Set `kv_cache_compression_config.algorithm` to
   `quantization_for_cold_page`, `quant` to `turboquant`, and choose
   `turboquant_bits` and `turboquant_mode`.
6. Apply both cache configurations to every worker that owns a local cold tier.
7. Verify nonzero offload and onboard bytes, and a cold slot size smaller than
   the hot Page, under a workload that creates real KV pressure.

`KvCacheConfig` and KVCacheManagerV2 own capacity, Page/Slot allocation,
cache-level routing, migration batching, completion events, and mapping
publication. `ColdPageQuantizationCompressionConfig` and its codec own only the
cold representation and its encode/decode policy.

For the underlying KVCM lifecycle and codec ABI, see
[KVCacheManagerV2 Cold-Page Codec Design](https://nvidia.github.io/TensorRT-LLM/developer-guide/kv-cache-cold-page-codec.html).
