<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# examples/visual_gen/configs — Agent Guidance

These YAMLs are a small, curated set of **showcase** `VisualGenArgs` recipes for
TensorRT-LLM VisualGen. They are loaded via `--visual_gen_args` (offline example
scripts) and `trtllm-serve --visual_gen_args`. Keep the set small and high-signal.

## A config earns a slot only if it demonstrates a non-obvious recipe

The bar is whether the config teaches something a user could not trivially guess
— not the raw number of fields it sets. Qualifying examples:

- a multi-GPU parallelism layout (CFG, Ulysses, Attention2D, parallel VAE),
- a full deployment recipe (e.g. the GB200 NVL72 rack configs),
- a non-obvious deployment knob, even a single one — e.g. warmup-shape control
  for a different output modality (text-to-image warmup on a multi-modal
  checkpoint).

## Do not add trivial configs

Do not add a config whose intent a user could reproduce from one line of
documentation. The two common offenders:

- A lone `quant_config: {quant_algo: ...}` per model and precision. Enumerating
  every model × precision would balloon the directory with near-identical files;
  a user who wants FP8/FP4 on one GPU can pass a one-line override or rely on the
  checkpoint's own quantization metadata.
- A single-GPU config whose body is just engine defaults (`VANILLA` /
  `cfg_size: 1` / `ulysses_size: 1` / `cuda_graph off`, no quant) — identical to
  running with no `--visual_gen_args` at all.

When in doubt, ask: does this config encode a non-obvious technique, or just a
value a user would already reach for by hand? If the latter, don't add it.
