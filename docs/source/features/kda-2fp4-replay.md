<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# KDA 2FP4 checkpoint replay

This opt-in path reduces persistent Kimi KDA recurrent-state storage on
Blackwell SM100/SM103. It supports one-token decode and DSpark verification
with up to seven draft tokens. Activations are BF16 and K=V=128. The
existing recurrent decode remains the default.

Enable it in a serving YAML:

```yaml
kv_cache_config:
  use_kv_cache_manager_v2: true
  mamba_state_config:
    kda_2fp4_replay: true
```

`kda_2fp4_replay` defaults to `false`; omitting it keeps the ordinary
recurrent-state path. Set it explicitly to `true` to enable packed checkpoints.

The checkpoint contains a primary NVFP4 tensor, an NVFP4 residual, their
E4M3 block-16 scales, and a shared FP32 outer scale per head. Rounding is
round-to-nearest-even. The complete payload is owned by V2, including
snapshots, suspension, and head-wise disaggregated transfer. A 128x128
head uses 18,448 bytes, including alignment, versus 32,768 bytes for BF16.
There is no persistent dense state mirror. `mamba_ssm_cache_dtype` selects
the ordinary recurrent path's dtype; it does not change this packed format.

Prefill reconstructs both primary and residual into temporary FP32 rows,
runs the ordinary prefill calculation, and packs the final state. This
also applies to chunked-prefill continuations and prefix-cache hits.
Each scheduled prefill call reads/writes the V2-owned 2FP4 checkpoint;
temporary internal KDA chunk states stay dense and are not separately
quantized.

Decode leaves the checkpoint unchanged between commits. It dynamically
quantizes the decayed K/Q checkpoint operands to NVFP4 and uses native
block-scaled MMA. Only the primary checkpoint participates in replay.
Accepted K/U are cached in BF16 and cumulative per-key G in FP32. The
history window is 16; rollover reconstructs both checkpoint terms and
commits the cached updates. Partial histories are flushed before suspension
or request release. One CuTe DSL launch fuses Conv4, Q/K normalization,
replay, and output RMSNorm/gating without changing their calculations.
A second, guarded CuTe DSL launch commits eligible checkpoints and advances
history lengths. Replay supplies an immutable length snapshot directly to
commit, without a separate metadata launch. Batch size, slot-pool capacity,
and cache/projection row strides are runtime parameters for both kernels.

## DSpark decoding

The same enablement works with DSpark:

```yaml
kv_cache_config:
  use_kv_cache_manager_v2: true
  mamba_state_config:
    kda_2fp4_replay: true
speculative_config:
  decoding_type: DSpark
  max_draft_len: 6
  block_size: 6
  speculative_model: /path/to/dspark-checkpoint
```

Verification uses the primary NVFP4 checkpoint and native NVFP4 projections,
as ordinary decode does. It writes candidate K/U/G and compact raw Conv
inputs without changing either live checkpoint or accepted history length.
One post-sampling Triton launch commits the accepted Conv prefix across all
local KDA layers and advances their history lengths. Rejected candidates
are ignored and overwritten by subsequent rounds.

The single history allocation has capacity `16 + max_draft_len + 1`; there
is no double buffer or per-token dense recurrent-state allocation. When an
accepted prefix reaches or exceeds 16 tokens, the complete accepted prefix
is committed and repacked once, reading both primary and residual. For
example, 15 pending tokens plus eight accepted tokens commit 23 updates.
This commit schedule need not produce bitwise-identical results to ordinary
one-token quantized decoding; validate model accuracy for the intended
DSpark workload. The separate drafter retains its own cache format and does
not inherit the target's packed KDA-state settings.

## Dedicated prefill workers

Use this configuration only on the dedicated context worker:

```yaml
kv_cache_config:
  use_kv_cache_manager_v2: true
  mamba_state_config:
    kda_2fp4_replay: true
    kda_2fp4_prefill_only: true
```

This omits **all K/U/G history allocation**. Synthetic decode warmups become
prefill rows; generation graph capture and decode padding allocation are
skipped. Prefill warmup/autotuning and prefill capture remain enabled.
Actual generation requests are rejected; attention-DP dummy rows are still
allowed. Ordinary Conv4 state remains allocated.

`kda_2fp4_prefill_only` defaults to `false` and requires
`kda_2fp4_replay: true`.

On the generation worker, use `kda_2fp4_replay: true` and leave
`kda_2fp4_prefill_only: false`. Its histories are
allocated once for live-request capacity, independently of the number of
V2 prefix-cache snapshot slots. Request recycling and suspension release
history rows, and new or resumed requests begin with empty histories.
Context and generation workers must both use the packed checkpoint format;
they may use different head-sharded TP sizes as supported by V2 transfer.

Speculative backends other than DSpark, beam search, non-BF16 activations, non-128 head
dimensions, legacy cache managers, and KV-pool rebalancing are not supported
by this initial path. The native checkpoint projection is quantized and is
not numerically equivalent to the default recurrent BF16/FP32 path; validate
model accuracy for the intended workload before deployment.
