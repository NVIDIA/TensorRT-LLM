<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# FlashAttention 4 build integration

FA4 b19 is fetched from the immutable revision in `fetch_content.json`.
`prepare_fa4.py` applies `patches/flash_attn_4_b19.patch` during CMake
FetchContent and validates the package inputs before and after patching.
The source pin, patch digest and expected source digests must agree.

`scripts/build_wheel.py` builds a separate
`flash_attn_4-4.0.0b19+trtllm.1-py3-none-any.whl` using FA4's upstream
setuptools backend. It preserves the `flash_attn.cute` import path, includes
FA4's LICENSE and AUTHORS, and adds build provenance in
`flash_attn.cute._trtllm_build_info`. FA4 still JIT-compiles its kernels on
the target GPU. The TRT-LLM wheel contains no FA4 package files.

`requirements.txt` retains stock b19 for bootstrapping a source-build environment.
The build replaces it with the patched wheel before generating Python stubs.
`setup.py` uses the exact patched runtime pin from `requirements-fa4.txt`, so
stock b19 cannot satisfy the TRT-LLM wheel's dependency. Reinstalling the
bootstrap requirements accepts the installed patched local version.

Both wheels are emitted in the selected `--dist_dir` (default `build/`). The
release container resolves dependencies from that directory with `--find-links`.
CI artifact archives and wheel uploads include both wheels; test install paths
that use `--no-deps` install both explicitly. Python examples and `trtllm-serve`
therefore use the same patched FA4 package without any runtime patch step.

## Install the built artifacts

After the normal `scripts/build_wheel.py` source build:

```bash
python -m pip install --find-links=build build/tensorrt_llm-*.whl
```

Use the selected output directory in place of `build` for custom or out-of-tree
builds. `build_wheel.py --install` installs the patched FA4 wheel into its build
interpreter before installing TRT-LLM in editable mode. For a separate Python
environment using a matching precompiled TRT-LLM wheel:

```bash
PIP_FIND_LINKS=/path/to/wheels \
TRTLLM_PRECOMPILED_LOCATION=/path/to/wheels/tensorrt_llm-<version>.whl \
python -m pip install -e .
```

The precompiled mechanism restores TRT-LLM's native artifacts; pip resolves FA4
as a normal dependency from the wheel directory or configured package index.

For index-based releases, publish the patched FA4 wheel to the configured
NVIDIA package index **before** publishing TRT-LLM. A TRT-LLM wheel alone is
insufficient until that exact dependency is available. This draft builds and
archives the companion wheel; it does not publish packages to a public index.
[PyPI does not accept local versions](https://packaging.python.org/en/latest/specifications/version-specifiers/#local-version-identifiers)
such as `+trtllm.1`; the release package channel must support this downstream
version scheme.

## Updating FA4

1. Select an upstream commit in `fetch_content.json` and review its runtime
   dependencies against TRT-LLM's CUTLASS DSL, QuACK and TVM-FFI pins.
2. Rebase or remove the downstream patch after checking the upstream per-call
   CTA/exp2 API and compilation-cache behavior. A register-allocation fix alone
   does not supply that API.
3. Update the validated revision, source/patch digests and upstream/patched
   versions in `prepare_fa4.py`, plus both requirements pins. Any mismatch fails
   the build, including when CMake reuses previously populated sources. Increment
   the downstream version whenever the patch changes; never publish different
   package contents under an existing version. The tuner caches this build identity.
4. Run packaging, GPU correctness, graph-replay, distributed and performance
   tests, then publish the patched dependency before its TRT-LLM consumer.

The b19 revision is `940cd9680f3315f2f06b43ab5bea2c2cf2d96806`, identified from
its [PyPI release provenance](https://pypi.org/integrity/flash-attn-4/4.0.0b19/flash_attn_4-4.0.0b19-py3-none-any.whl/provenance).
Its 50 package Python files match the published b19 wheel byte for byte before
patching. The FA4 wheel retains the BSD-3-Clause license.
