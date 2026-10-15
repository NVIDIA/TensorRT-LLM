<!-- Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Single-node core performance model priorities

Snapshot: 2026-10-15, based on main at `c9b18eefa7` and the September
model-priority input. The ordering is provisional, not a permanent support policy.
Scope is only [llm_perf_core.yml](../../test_lists/qa/llm_perf_core.yml), including
multiple GPUs inside one node. Disaggregated, multi-node, accuracy, and L0
perf-sanity lists are outside this change.

The starting inventory contains **172 condition-scoped definitions and 172
distinct test IDs**. Counts below refer to definitions, not runs: a definition
can match multiple systems. The refresh has **175 definitions and 175 distinct
IDs**: seven GLM-5 FP8 definitions are upgraded to GLM-5.3 and three
GLM-5.3-Flash definitions are added.
DeepSeek-V4-Pro-Base FP8 and Nemotron-3-Super retain every original definition.
Definition counts do not establish wall time or passing GPU coverage.

For representative hardware records, condition-matched definition counts are:

| System record | Before | After |
|---|---:|---:|
| 8x B200, SM100, 180000 MiB/GPU | 130 | 133 |
| 8x B300, SM103, 288000 MiB/GPU | 146 | 149 |
| 4x GB300, SM103, 288000 MiB/GPU | 85 | 88 |
| 4x VR200, SM107, 288000 MiB/GPU | 73 | 73 |
| 8x H100, SM90, 81000 MiB/GPU | 49 | 49 |
| 8x RTX-6000 Server, SM120, 97000 MiB/GPU | 46 | 46 |

These are selector counts, not verified executable or passing cases. The three
additional Flash points select SM100/103. Existing Rubin, Hopper and RTX-6000
coverage is preserved. Original Pro-Base FP8 and Super hardware conditions,
parallelism and workloads remain intact.

## GLM upgrade and additions

Before the refresh, core already contained **seven GLM-5 FP8** and **seven
GLM-5.2 NVFP4** definitions. GLM-5.3 replaces the FP8 generation; the NVFP4
cases retain their separate quantization and backend coverage. GLM-5.3-Flash
is an additional model, with no existing model used as a budget donor.

| Existing core slots | Result | Definitions | Deployment and workload | Reason and limits |
|---|---|---:|---|---|
| `glm_5_fp8` in condition 7 | Upgrade to `glm_5.3_fp8` | 7 | TP8/EP8, FP8, five existing ISL/OSL pairs plus low-concurrency and throughput points | Closest same-family replacement. Use the official `GLM-5.3` FP8 directory, explicit DEEPGEMM, attention DP off, FP8 KV, chunking, and batch/graph cap 128. Keep concurrency 512 as queued load for the throughput point. |
| Additional coverage | Add `glm_5.3_flash_fp8` | 3 | TP4/EP4, 1024/1024, concurrency 1/32/128; batch/graph cap 64 and 16384-token budget | Covers the missing KDA + sparse-MLA model using the documented four-GPU recipe under an explicit SM100/103 condition. |
| `glm_5.2_fp4` in condition 7 | Retain unchanged | 7 | Existing NVFP4, TP8/EP8 sweep and latency/throughput points | No supported GLM-5.3 NVFP4 checkpoint/backend has been established by this change. |

The GLM-5.3-Flash points use BF16 KV and no MTP. At concurrency 128, requests
queue behind the batch cap of 64; this is a throughput load point, not a claim
that 128 sequences execute simultaneously. Model-specific startup timeouts
allow cold loading.

Sources for these choices:

- [GLM deployment guide](../../../../docs/source/deployment-guide/deployment-guide-for-glm-5-on-trtllm.md)
  identifies `zai-org/GLM-5.3` as FP8, specifies DEEPGEMM, and gives the TP8/EP8
  recipe. Its GLM-5 NVFP4 checkpoint is not a GLM-5.3 NVFP4 checkpoint.
- [GLM-5.3-Flash deployment guide](../../../../docs/source/deployment-guide/deployment-guide-for-glm-5.3-flash-on-trtllm.md)
  and [accuracy recipes](../accuracy/test_glm53_flash.py) establish FP8, four
  B200 GPUs, TP4/EP4, chunking, and disabled block reuse for the chosen baseline.

## Decisions in priority order

“Present” means listed in core; it does not assert a recent passing GPU run.
Before/after counts are included to distinguish an actual coverage gap from a
model that already has many cases.

| Priority input | Core before → after | Recommendation |
|---|---|---|
| DSv4.1 Flash | 0 → 0 | Deferred in this refresh. Preserve all DeepSeek-V4-Pro-Base FP8 definitions. Any future V4.1 addition needs its own validated one-node recipe and test budget; it does not inherit V4 Pro's MTP or separate-draft DSpark configuration. |
| Kimi K3 | 10 → 10 | Keep existing TP8 B300 and TP4 Rubin latency/throughput and streaming points. It is already well represented. Do not substitute it into four-B200 or eight-Hopper slots based on parameter count alone. |
| GLM 5.3 | 0 → 7 | Replace GLM-5 FP8 first. Keep GLM-5.2 NVFP4 until a supported 5.3 NVFP4 checkpoint and its backend are established. |
| GLM 5.3 Flash | 0 → 3 | Add the four-Blackwell-GPU serving recipe above. Preserve GLM-5.2 NVFP4 and Nemotron-3-Super coverage. Add MTP only as a separately validated recipe with real text and measured acceptance. |
| Qwen 3.8 Flash Next | 6 → 6 | Keep both single-GPU FP8 and NVFP4 MTP3 recipes. The FP8 concurrency-16 point stays behind its existing >260 GiB SM103 gate. Older Qwen3.6-35B points are future budget donors, not safe model-name substitutions: Flash-Next needs PLE host offload, different memory budgets, and different MoE backends. |
| DSv4 Flash | 8 → 8 | Keep four post-trained and four FP8 Base definitions across four/eight GPUs. Preserve all Pro-Base FP8, post-trained Pro and Pro speculative coverage. |
| PineDrift MidnightIsland | 0 → 0 | Defer. The supplied label does not resolve to a model path, architecture, deployment guide, or perf preset in this snapshot. Obtain the exact checkpoint identity, precision, backend, and one-node footprint before allocating slots. Do not infer that the two words denote one or two models. |
| MiniMax M3 | 13 → 13 | Keep the MXFP8 point and NVFP4 bench/serve cases. MSA, FP8 KV, and the bounded capture sequence lengths are distinct coverage. Replacing a long-context case requires validating its own `max_seq_len`; do not copy the short-context cap. |
| Qwen 3.5 (400B) | 21 → 21 | Already covered as `qwen3.5_397b_a17b_{fp8,fp4}` with four/eight-GPU recipes. The four older Qwen3-235B FP8/FP4 definitions are potential budget donors on matching hardware, but retain a GQA MoE sentinel because Qwen3.5 uses hybrid GDN attention. |
| Inkling (withdrawn) | 0 → 0 | Excluded from proposed additions. |
| Nemotron 3 Ultra | 3 → 3 | Keep all three TP4/EP4 serving points. Retain the three Super serving points as well; Ultra does not replace them. Do not replace the small Nano host/latency signals with this much larger checkpoint. |
| Qwen 3.8 (2T) | 3 → 3 | Existing `qwen3.8_max_fp4_mtp` resolves to the 2.4T-A95B NVFP4 checkpoint. Keep TP8 B300 and MTP3; the shorthand “2T” does not justify a different invented path or a four-GPU B200 configuration. |

Supporting deployment references:
[Kimi K3](../../../../docs/source/deployment-guide/deployment-guide-for-kimi-k3-on-trtllm.md),
[Qwen3.8 Flash-Next](../../../../docs/source/deployment-guide/deployment-guide-for-qwen3.8-flash-next-on-trtllm.md),
[Qwen3.8 / Qwen3.5](../../../../docs/source/deployment-guide/deployment-guide-for-qwen3.8-qwen3.5-on-trtllm.md),
[MiniMax M3](../../../../docs/source/deployment-guide/deployment-guide-for-minimax-m3-on-trtllm.md),
[Nemotron 3](../../../../docs/source/deployment-guide/deployment-guide-for-nemotron-3-on-trtllm.md),
and the [model support matrix](../../../../docs/source/models/supported-models.md).
Exact perf checkpoint aliases are in [_model_paths.py](_model_paths.py).

## Coverage to retain

- Keep Qwen3-0.6B and Qwen3-30B PP4/concurrency-1 cases: they intentionally
  amplify host/overlap-scheduler overhead. V4.1 also does not support PP.
- Keep Qwen3-4B EAGLE3: native MTP/DSpark is a different speculative path.
- Keep the GPT-OSS-20B Hopper regression cases and the small dense/BF16 tests.
  A large Blackwell MoE cannot replace their hardware, precision, or latency signal.
- Keep the Nano, Gemma, GLM-5.2 NVFP4 and Qwen3 GQA representatives pending
  evidence that their specific backend/attention coverage is redundant.
- Keep existing L40S, Hopper, RTX-6000 and Rubin coverage for the retained
  models, including every Pro-Base FP8 and Super point. New GLM Flash cases
  select only SM100/103; this condition does not imply support on other tiers.

## Validation and rollout

This is a configuration proposal for GPU validation, not a measured performance
result. Static checks cover list parsing, model resolution, normalization of
labels, generated model/server/client configs, parallelism, memory gates, and
isolation from existing recipes. Local hardware is two RTX PRO 6000 GPUs and
cannot execute the selected four/eight-GPU B200/GB300 deployments.

Before promoting the GLM refresh to the release baseline:

1. Confirm `GLM-5.3` and `GLM-5.3-Flash` are synced
   under `LLM_MODELS_ROOT` on each selected QA pool. Local model-path aliases
   do not provide model synchronization or prove checkpoint availability.
2. Run all seven GLM-5.3 cases on eight B200 GPUs and the three GLM Flash cases
   on four B200 GPUs. Run the B300 variants before treating matching SM103
   conditions as validated there.
   Preserve logs for cold load, CUDA graph capture, memory, TTFT/TPOT and throughput.
3. Confirm checkpoint quantization and GLM accuracy using the existing
   accuracy suite. Synthetic fixed-length perf requests do not establish
   accuracy or speculative acceptance quality.
4. Start new dashboard series for the new model IDs. Do not splice their
   throughput into the old model's baseline or claim a cross-model speedup.
5. Revisit this table when priorities change. Move further slots only after
   checking per-GPU coverage and wall time; the 175-definition count alone
   does not guarantee a stable release-test duration.
