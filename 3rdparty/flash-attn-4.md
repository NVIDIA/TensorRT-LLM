<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# FlashAttention 4 build integration

FA4 b19 is fetched from the immutable revision in `fetch_content.json`.
`prepare_fa4.py` applies `patches/flash_attn_4_b19.patch` during CMake
FetchContent and validates the package inputs before and after patching.
The source pin, patch digest and expected source digests must agree.

`scripts/build_wheel.py` validates the patched tree again, then stages its Python
sources, LICENSE and AUTHORS under `3rdparty/trtllm_flash_attn/`. FA4 uses CuTe
JIT: its kernels are compiled on the target GPU at runtime. No ahead-of-time
CUDA compilation or separate FA4 distribution is required here.

The build rewrites FA4's internal `flash_attn.cute` imports to
`trtllm_flash_attn`. All TRT-LLM FA4 consumers use this owned package, so a stock
`flash-attn-4` installation cannot overwrite it. `setup.py` includes it in the
TRT-LLM wheel. Release containers install that wheel; both Python examples and
`trtllm-serve` consequently use the same patched sources without modifying
site-packages at startup. The separate pip requirement for FA4 is removed;
its runtime dependencies remain in `requirements.txt`.

For source builds, use the normal `scripts/build_wheel.py` flow before
`pip install -e .`. Out-of-tree builds stage the package with the other wheel
inputs. `TRTLLM_USE_PRECOMPILED` restores the bundled package from a matching
wheel or build directory, including directory link mode. A precompiled artifact
without this package fails with a rebuild instruction.

The bundled `_build_info.py` records the upstream version, revision and patch
digest. The VisualGen autotuner includes this build identity in its persistent
cache key. Tuning remains opt-in with `TLLM_VISUAL_GEN_FA4_AUTOTUNE=1` and does
not modify process-global FA4 policy.

## Updating FA4

1. Select an upstream commit in `fetch_content.json` and review its runtime
   dependencies against TRT-LLM's CUTLASS DSL, QuACK and TVM-FFI pins.
2. Rebase or remove the downstream patch after checking the upstream per-call
   CTA/exp2 API and compilation-cache behavior. A register-allocation fix alone
   does not supply that API.
3. Update the validated revision, version, source/patch digests and preparation
   logic in `prepare_fa4.py` after inspecting the new sources. A changed pin or
   patch fails the build until these checks are deliberately updated. Staging
   rechecks them even when CMake reuses an existing populated source tree.
4. Run the packaging tests and FA4 GPU correctness, graph-replay, distributed
   and performance tests before enabling the new combination.

The b19 revision is `940cd9680f3315f2f06b43ab5bea2c2cf2d96806`, identified from
the [PyPI release provenance](https://pypi.org/integrity/flash-attn-4/4.0.0b19/flash_attn_4-4.0.0b19-py3-none-any.whl/provenance).
Its 50 package Python files match the published b19 wheel byte for byte before
patching. The bundled license remains BSD-3-Clause.
