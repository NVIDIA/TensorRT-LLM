<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# FA4 CTA and exp2 autotuning demo

This draft demonstrates selecting FA4's CTA count and exp2 emulation together
using TRT-LLM's existing warmup `AutoTuner`. It is opt-in. TRT-LLM's build now
fetches and patches FA4 b19, then builds a companion FA4 wheel; users do not patch their
installation. This draft still needs GPU and end-to-end performance validation.

## Why tune both?

Disabling 2CTA is a useful fallback for the long-sequence workloads reported in
[TRT-LLM #19292](https://github.com/NVIDIA/TensorRT-LLM/pull/19292).
CTA cooperation and exp2 emulation affect different costs and can interact.
SM100 and SM103 also have different softmax throughput. A single process-wide
choice does not establish the fastest kernel for every GPU and attention shape.

The demo searches dense FP16/BF16 MHA with head dimension 128 on SM100/SM103.
For SM100 it compares 1CTA and eligible 2CTA with the FA4 default exp2 policy and
`ex2_emu_freq` values 0, 4, 8, 16, 32. These values are a bounded exploratory
search, not a calibrated optimal set. SM103 searches the default and hardware
exp2 (`0`). Frequency is a kernel scheduling parameter, not a portable percentage
of operations emulated.

The unchanged FA4 heuristic, including automatic split-KV, participates as a
control and is the fallback for untuned shapes. Explicit candidates use one split
to make CTA comparisons meaningful. `FA_DISABLE_2CTA=1` and FA4's CUDA 12 guard
remove 2CTA candidates. No environment variable or FA4 tuning table is changed
while tuning.

The experiment is restricted to unmasked noncausal attention with equal Q/K/V
head dimensions and counts. Causal attention, padding masks, GQA, other dimensions
and other GPUs continue through the existing backend. Q and KV sequence lengths
may differ. Output and float32 LSE remain available for distributed wrappers;
this does not constitute distributed end-to-end validation.

## Dependency seam

The pinned FA4 private forward API has neither per-call CTA nor exp2 overrides.
Changing its globals would be unsafe across callers, and changing the exp2 table
alone would not distinguish entries in the compiled-kernel cache.

`flash_attn_4_b19.patch` adds `use_2cta` and `ex2_emu_freq` to `_flash_attn_fwd`.
It validates the narrow supported contract, retains FA4's 2CTA guards, adds exp2
frequency to the compilation key, and copies the kernel instance's tuning table
before changing it. Calls omitting both arguments retain their original behavior.
The explicit capability marker prevents the opt-in path from silently pretending
to tune an unpatched dependency.

The patch lives at `3rdparty/patches/flash_attn_4_b19.patch`. CMake applies it
while fetching the pinned source; `scripts/build_wheel.py` builds the validated
result as `flash-attn-4==4.0.0b19+trtllm.1`, retaining `flash_attn.cute`.
The TRT-LLM wheel pins that exact dependency.
Python examples and `trtllm-serve` therefore use the same patched implementation.
Install the generated wheels with `pip install --find-links=build build/tensorrt_llm-*.whl`.
The release container installs the companion FA4 wheel automatically.

See [FA4 build integration](../../../3rdparty/flash-attn-4.md) for source and
editable installs, version-bump checks and dependency provenance. A newer FA4
register-allocation fix may change the winners and needs fresh validation; it
does not replace the per-call tuning API supplied by this patch.

## Run the standalone demo

First run numerical and graph-replay tests in the GPU environment:

```bash
FA_DISABLE_2CTA=0 pytest -q tests/unittest/_torch/visual_gen/test_attention_fa4.py -k autotuned_tactics
pytest -q tests/unittest/_torch/visual_gen/test_fa4_autotuner.py
```

Then run a small smoke case and the three long sequence lengths separately:

```bash
TLLM_VISUAL_GEN_FA4_AUTOTUNE=1 FA_DISABLE_2CTA=0 \
python examples/visual_gen/fa4_autotune/benchmark.py \
  --seq-lens 4096 --output /tmp/fa4-smoke.json

TLLM_VISUAL_GEN_FA4_AUTOTUNE=1 FA_DISABLE_2CTA=0 \
TLLM_AUTOTUNER_LOG_LEVEL_DEBUG_TO_INFO=1 \
python examples/visual_gen/fa4_autotune/benchmark.py \
  --seq-lens 75600 111600 147600 --heads 40 --output /tmp/fa4-long.json
```

Run separately on B200 and B300. The benchmark checks every candidate's output
and LSE against the unchanged FA4 path before timing, excludes first-use JIT from
timings, alternates candidate order over three rounds, records individual samples
and medians, and exercises `FlashAttn4Attention.forward` inside `autotune()`.
It fails on a candidate error rather than reporting an incomplete sweep as success.
These timings describe attention only. They do not measure Wan pipeline latency.

## Use with VisualGen warmup

Set `TLLM_VISUAL_GEN_FA4_AUTOTUNE=1` before constructing the pipeline, select
`attention.backend: FA4`, and keep the existing `torch_compile.enable_autotune`
enabled. Populate warmup with the actual generation shapes. The existing
`TLLM_AUTOTUNER_CACHE_PATH` saves and reloads the selected tactics.

The cache uses exact Q/K/V shapes plus dtype, strides, GPU name/capability,
softmax scale, CUDA/PyTorch/CUTLASS versions and FA4 dispatch flags. Changing
these yields a different cache entry. Profiling happens only in the existing
autotune context; a cache miss outside that context runs the original heuristic.
An outer CUDA graph capture never starts a new search.

With #19292 applied, its default `FA_DISABLE_2CTA=1` is honored, so the demo searches
exp2 policy within 1CTA. Set `FA_DISABLE_2CTA=0` before importing FA4 to evaluate
both CTA choices. A production follow-up should distinguish an explicit user
disable from a conservative untuned default, rather than overriding that policy
implicitly.

## Validation still required before productization

- Run all candidate numerical/LSE and graph tests on B200 and B300 with the exact
  FA4/CUTLASS/CUDA package combination; measure startup/JIT cost too.
- Compare 1CTA/default, 1CTA/tuned, 2CTA/default and 2CTA/tuned under identical
  warmed Wan configurations. Include unchanged FA4 and TRTLLM controls.
- Check generated-media quality and warmed full-pipeline latency; standalone
  attention gains cannot establish an end-to-end improvement.
- Validate cache reload in a new process and distributed execution, including
  Ulysses/Attention2D/Ring callers and their actual local shapes.
- Upstream the per-call dependency API, then reconsider the candidate set and
  default opt-in policy based on measurements.
