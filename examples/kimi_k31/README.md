<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Kimi K3.1 NVFP4 serving

The `kimi_k31_pro_nvfp4__sm_10x__default` target serves SM100, SM103, and SM107.
Eligible text-only SM107 layouts with explicit FP8 draft history route to the
separate specialized targets; other supported configurations use this default.

## Aggregated serving

The [aggregated config](k31_nvfp4_config.yaml) uses TP8/EP8, CUDA graphs,
overlap scheduling, NVFP4 main cache, FP4 indexer cache, and BF16 KDA state.
Launch from the repository root:

```bash
trtllm-serve nvidia/kimi_k31_pro_nvfp4 --config examples/kimi_k31/k31_nvfp4_config.yaml
```

With NVFP4 main storage, the final two MLA layers use FP8 by default.
Set `TLLM_KIMI_K3_FP8_KV_LAYERS=""` for all-NVFP4 main storage, or use a
comma-separated layer list for explicit FP8 layers.

## MoE backend

`TRTLLM` is the default. Select `CUTEDSL` for grouped GEMM or
`MEGAMOE_CUTEDSL` for fused expert dispatch, computation, and combine:

```yaml
moe_config:
  backend: CUTEDSL # TRTLLM, CUTEDSL, or MEGAMOE_CUTEDSL
```

All three backends load the native NVFP4 expert weights and use SiTU activation.
The cache precision selectors are independent of the MoE backend. MegaMoE
requires expert-only MoE parallelism (`moe_tensor_parallel_size: 1`), a supported
symmetric-memory runtime, and an expert count divisible by the EP size. Under
attention TP, MegaMoE repeats the replicated tokens' expert computation across
TP ranks; attention DP is preferable for throughput. Kernel availability is
checked by the selected backend at initialization.

## Disaggregated serving with DFlash2 and Helix

For `nvidia/kimi_k31_pro_nvfp4`, use [the CTX config](disagg/k31_nvfp4_ctx_config.yaml) and
[the GEN config](disagg/k31_nvfp4_gen_config.yaml). These use TP8/EP8 context workers and
TP1/CP8/EP8 generation workers. Each worker needs eight GPUs; on GB300
this is two nodes per worker. Use the [worker/proxy launch sequence](../kimi_k3/disagg/README.md#launch-sequence-manual-single-ctx--single-gen)
with these configs and the NVIDIA checkpoint, using eight MPI ranks per worker. Replace `speculative_model`
in both files with the same local RadixArk Kimi-K3.1-Dflash2-Preview checkpoint.

The configs require modeling-v2, KV cache manager V2, and the Python NIXL
transceiver. CTX publishes draft history with its cache pages; GEN binds
that history before drafting. Helix distributes persistent draft history
by assigning global page `g` to rank `g % cp_size`.

Set `speculative_config.context_kv_dtype` to `auto` for BF16 draft history or
`fp8` for FP8 history on both workers. The same choices apply to aggregated
DFlash serving, independently of the main, indexer, and KDA cache formats.

Both workers enable overlap, and GEN enables CUDA graphs. The Helix GEN
config disables rejection sampling, as required by the upstream context-parallel
argument validation. The configs select
all-NVFP4 main storage, FP4 indexer storage, and BF16 KDA recurrent state.
To use the FP8/FP32 recipe, set `kv_cache_config.dtype: fp8`,
`sparse_attention_config.indexer_k_dtype: fp8`, and
`kv_cache_config.mamba_ssm_cache_dtype: float32` on both workers. All three
selectors are independent and resolve to the same NVIDIA checkpoint target;
keep each selector consistent between CTX and GEN for cache transfer.

The context token budget is 8,448, so inputs must fit that budget. The
example verifies five-token blocks (`max_draft_len: 4`); keep the width
identical on both workers. Block reuse is disabled.
