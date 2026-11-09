<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->
# Local bring-up components

These TP1 components are checkout-only test helpers, not supported serving APIs or a
reference for TP4 bitwise results. Their names describe the isolated component; device and geometry contracts are
documented in each helper. They do not register ModelingV2 targets. The six-layer prefill helper also
supports the local cached-generation tests. Use `LLM()` or `trtllm-serve` for serving.

Run tests through pytest with the repository's test import plugin. Existing L0 selectors
retain the small single-GPU tests; the large component suites remain manual-only.

TP1 normalization and activation use their FlashInfer contracts; the TP4 input norm
and dense MLP use explicit FP32 Beam operators, so BF16 rounding can differ. TP1 also
accepts a narrower RoPE schema. Compare each helper against its declared contract,
not against the TP4 model as a bitwise reference.
