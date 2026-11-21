<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Inkling 2

`Inkling2ForConditionalGeneration` (`thinkingmachines/Inkling-2-Small`): a 48-layer MoE text decoder
(hidden 4608, 384 routed experts, top 8) with a vision tower. The family implements the text model.

## What the architecture needs

* **Attention without rotary embeddings.** Position enters through a relative bias each token mixes
  per head from 16 bias-vs-distance profiles (`rel_logits_proj`), over distances below 1024. 40
  sliding-window layers attend the last 1024 tokens; 8 global layers (5, 11, ..., 47) attend the
  whole prefix and scale q and the relative states by `tau = 1 + 0.1 ln(max((p + 1) / 8192, 1))`.
  Each token and head has a sink logit, and the output is gated by `sigmoid(gate)`. q, k and the
  relative states are L2-clipped; v is RMS-normalized without a weight.
* **Short convolutions.** Width-4 causal convolutions with a residual run on k and v (dynamic
  per-token taps on the global layers) and on each sublayer's output. Their windows are
  per-request state.
* **Block attention residuals.** Every 6 layers the residual stream closes a block; each
  sublayer reads a learned softmax mix of the normalized blocks and the current partial block.
* **Experts.** fp32 sigmoid router with a selection bias; SiTU activation
  (`4 tanh(g / 4) sigmoid(g) * 25 tanh(u / 25)`). The checkpoint stores layers 0-39 in NVFP4 (with
  per-token 4/6 NVFP4 activations) and layers 40-47 in MXFP8.

## Layout

```text
inkling2/
├── config.py               # InklingDims: the decoder's shape and per-layer expert formats
├── kernels/                # Triton: short convolution, paged attention with the relative bias
├── ops/                    # OpWrappers: conv, attention prologue and core, residual mix,
│                           # router, NVFP4 experts (FlashInfer CuTe DSL), MXFP8 experts
└── inkling_2_small_nvfp4__sm_103__tep4/
    ├── weights.py          # weight table: TP4 heads, EP4 experts, kernel layouts
    ├── modeling.py         # the 48-layer forward and the engine shell
    └── cache.py            # KVCacheManagerV2 with the convolution windows
```

## Target `inkling_2_small_nvfp4__sm_103__tep4`

The released checkpoint on one GB300 tray: attention TP4 (12 query and 2 KV heads per rank), MoE
EP4 (96 experts per rank), BF16 KV cache. The baseline runs eagerly, without CUDA graphs, the
overlap scheduler, chunked prefill, KV cache block reuse or speculative decoding.

The family is not routed yet: these are the operators and the target's components; the route that
serves the target through `LLM()` and `trtllm-serve` follows separately.

The NVFP4 experts run FlashInfer's CuTe DSL fused MoE with SiTU and per-token 4/6 activations. The
MXFP8 experts run a grouped Triton MoE that expands the E4M3 weights to bf16 in registers (exactly)
and accumulates in fp32.
