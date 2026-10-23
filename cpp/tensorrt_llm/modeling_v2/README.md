<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# modeling_v2 native code

C++/CUDA used only by modeling_v2 targets
(`tensorrt_llm/_torch/_experimental/modeling_v2/`) lives here, not in
`cpp/tensorrt_llm/kernels/` or `thop/`. Each model family builds into one
library, `libtrtllm_mv2_<family>.so`.

- **Not linked into stock.** Nothing in the family library is linked into
  `libtensorrt_llm.so` or `libth_common.so`.
- **Loaded on demand.** Python loads the family library only when routing
  picks one of the family's targets. With `modeling_v2=off` (the default),
  nothing from here is loaded.

This document is the guide for adding native code to a target. The family
`fixture/` is the smallest complete example: one target with one op. Copy it.

## Layout

```
cpp/tensorrt_llm/modeling_v2/
  CMakeLists.txt              adds every family directory (the only stock hook)
  contractSurface.txt         the stock headers native code may include
  common/                     namespace and torch-registration macros
  <familyCamel>/              one model family -> libtrtllm_mv2_<family>.so
    CMakeLists.txt            trtllm_mv2_family / trtllm_mv2_shared / trtllm_mv2_target
    shared/                   code more than one target of the family calls
    <targetCamel>/            one target
      kernels/                *.cu *.cuh *.h -- compiled for the target's ARCHS, no torch
      ops/                    *.cpp torch bindings
```

### Naming

**Names are the Python package names.** A family is `minimax_m3`; a target is
`minimax_m3_nvfp4__sm_107__tp2`.

**Directories are their camelCase form**, following `CODING_GUIDELINES.md`:
`minimaxM3/minimaxM3Nvfp4Sm107Tp2/`.

The build, the loader and CI compute the same mapping, and
`test_modeling_v2_native_claims.py` checks that it holds in both directions.

## Adding native code to a target

The example below adds an op to target `foo_nvfp4__sm_103__tp4` of family
`foo`.

1. **Declare it.** The declaration goes in `cpp/tensorrt_llm/modeling_v2/foo/CMakeLists.txt`:

   ```cmake
   trtllm_mv2_family(foo)

   trtllm_mv2_target(
     foo foo_nvfp4__sm_103__tp4
     ARCHS 103a
     SOURCES kernels/*.cu ops/*.cpp
     REQUIRES MULTI_DEVICE)
   ```

2. **Write the kernel.** It goes in `fooNvfp4Sm103Tp4/kernels/`, wrapped in the
   target's namespace:

   ```cpp
   #include "tensorrt_llm/modeling_v2/common/namespace.h"

   TRTLLM_MV2_NAMESPACE_BEGIN   // tensorrt_llm::modeling_v2::foo::foo_nvfp4_sm_103_tp4
   namespace kernels
   {
   cudaError_t invokeMyOp(/* ... */, cudaStream_t stream);
   }
   TRTLLM_MV2_NAMESPACE_END
   ```

3. **Bind it to torch** in `fooNvfp4Sm103Tp4/ops/`, through the target's own torch namespace:

   ```cpp
   #include "tensorrt_llm/modeling_v2/common/torchLibrary.h"

   TRTLLM_MV2_LIBRARY_FRAGMENT(m) { m.def("my_op(Tensor x) -> Tensor"); }
   TRTLLM_MV2_LIBRARY_IMPL(CUDA, m) { m.impl("my_op", &TRTLLM_MV2_NS::myOp); }
   ```

4. **Load it** from the Python target package,
   `models/foo/foo_nvfp4__sm_103__tp4/__init__.py`:

   ```python
   from tensorrt_llm._torch._experimental.modeling_v2._native import load_target_ops

   native_ops = load_target_ops(__name__)  # torch.ops.mv2_foo_foo_nvfp4_sm_103_tp4
   ```

   The name is not `ops`: that is the target's `ops/` package, which would
   replace the attribute when it is imported.

5. **Fake every op** in the target's `fake_ops.py`. The loader imports it right
   after opening the library:

   ```python
   from tensorrt_llm._torch._experimental.modeling_v2._native import namespace_of

   _NS = namespace_of(__package__)

   @torch.library.register_fake(f"{_NS}::my_op")
   def _(x):
       return torch.empty_like(x)
   ```

6. **Wrap it in an entry**, `ops/<category>/my_op.py`, which calls
   `native_ops.my_op(...)`. Test the entry like any other catalog entry, under
   `tests/unittest/_torch/modeling_v2/models/foo/<target>/`.

7. **List the tests.**
   - The family's unit tests go in a test-db entry of its own:
     `unittest/_torch/modeling_v2/models/foo`.
   - Its accuracy gates go in `accuracy/modeling_v2/test_foo.py`.

   CBTS selects a family's tests by these two names (see [CI](#ci)).

No stock file changes along the way.

## CMake API

The functions are defined in `cpp/cmake/modules/modelingV2.cmake`.

| Function | What it declares |
|---|---|
| `trtllm_mv2_family(<family>)` | The family; call it first in the family's `CMakeLists.txt`. |
| `trtllm_mv2_target(<family> <target> ARCHS ... SOURCES ... [DEPS ...] [REQUIRES ...])` | One target's code, in `<targetCamel>/`. |
| `trtllm_mv2_shared(<family> ARCHS ... SOURCES ... [DEPS ...] [REQUIRES ...])` | Code several targets of the family call, in `shared/`. Its ops are in `torch.ops.mv2_<family>`. |

| Keyword | Meaning |
|---|---|
| `ARCHS` | `NN`, `NNa` or `NNf`, intersected with the build's requested architectures: `NN`/`NNa` build when `NN` is requested; `NNf` builds when any member of family `NN` is. A target whose intersection is empty is **not built**: no placeholder kernels, and the manifest records why. The declared architectures must cover the SM in the target's name; when moving existing code, keep the architectures it was gated with. |
| `SOURCES` | Paths or globs relative to the unit's directory. `.cu` files are compiled for `ARCHS` without torch; `.cpp` files are the torch bindings. |
| `DEPS` | `cublas`, `cublaslt`, `cuda_driver`, `internal_cutlass`, `nccl`, `trtllm_gen_fmha_interface`. Ask for anything else by extending the module. |
| `REQUIRES` | `MULTI_DEVICE`, `NVSHMEM`. A target whose requirement the build lacks (e.g. a single-device build) is skipped and recorded. |

### Build options

- **`-DBUILD_MODELING_V2=OFF`** builds no family library. The manifest then
  says so, and every target with native code is a `NoTarget`.
- **`-DTRTLLM_MODELING_V2_TARGETS=<family>/<target>;...`** builds only the
  matching targets. Globs are allowed, e.g. `fixture/*`; the default is
  `all`. Pass it through
  `scripts/build_wheel.py --extra-cmake-vars TRTLLM_MODELING_V2_TARGETS=...`.
- **Incremental builds.** A family library is its own build target, so an
  edit relinks only that library:

  ```bash
  cmake --build cpp/build --target trtllm_mv2_<family>
  ```

## Namespaces

| Kind | Name |
|---|---|
| Target torch ops | `torch.ops.mv2_<family>_<target>`, with `__` collapsed to `_` |
| Family shared torch ops | `torch.ops.mv2_<family>`; also holds `contract_fingerprint()` |
| Target C++ | `tensorrt_llm::modeling_v2::<family>::<target>` (`TRTLLM_MV2_NAMESPACE_BEGIN`) |
| Shared C++ | `tensorrt_llm::modeling_v2::<family>::shared` |

**Each torch namespace is claimed exactly once.** The family library's
generated registration unit claims it with `TORCH_LIBRARY`, so a second
library claiming the same namespace fails at load. Op files only add fragments,
through the `TRTLLM_MV2_*` macros; raw `TORCH_LIBRARY*` is rejected by the
claims test.

**Sibling targets can share file and function names.** Every target has its
own C++ namespace, and the family library exports no symbol
(`common/familyLibrary.version`).

## Sharing code

**Sharing moves code up, never sideways:**

1. target
2. family `shared/`
3. stock `cpp/tensorrt_llm/kernels` + `thop`, once a built-in model calls it
   (normal stock review)

A target never includes or calls a sibling target's code. When two targets
need the same kernel, move it to `shared/`. A cross-family `catalog/` is
reserved and not built yet.

## What native code may include

- Its own target's files.
- Its family's `shared/`.
- `common/`.
- The stock headers listed in `contractSurface.txt`, such as
  `tensorrt_llm/common/*` and selected kernel device helpers like
  `kernels/quantization.cuh`.
- System, CUDA, torch and cutlass headers.

It links against `libtensorrt_llm.so` (logger, `TllmException`, mcast
registry and other singletons stay there), `th_utils` and torch.

**The contract fingerprint.** `contractSurface.txt` is also the stock side of
the contract. The build hashes those headers, the global compile definitions,
and the torch and CUDA versions into a *contract fingerprint*:

- `libth_common.so` reports it as `torch.ops.trtllm.mv2_contract_fingerprint()`.
- Each family library reports it as `torch.ops.mv2_<family>.contract_fingerprint()`.
- The loader refuses a family library whose fingerprint differs from the
  loaded `libth_common.so`'s.

Adding a header to the list widens what stock changes can affect targets; the
header's stock owners review it.

**When a target needs stock to change**, for example a new option on a stock
kernel, that change is a separate stock PR. It must leave stock behavior
unchanged by default, and it is reviewed by the stock owners.

## How it loads

`_native.load_target_ops(__name__)` runs these steps:

1. Read `tensorrt_llm/libs/modeling_v2_manifest.json`.
2. Check that the target was built, and built for this GPU.
3. Open the family library.
4. Compare contract fingerprints.
5. Import the target's `fake_ops.py`.

| Outcome | Raised as | Meaning |
|---|---|---|
| The target was skipped or excluded by this build, or does not run on this GPU | `NoTarget` | `auto` keeps the built-in model; `require` reports why |
| The manifest or library is missing, `dlopen` fails, the fingerprint differs, or an op is missing | `NativeLoadError` | A broken installation; never a fallback |

`route` imports a target only on the branch that returns it, so this runs as
part of routing. Every rank then agrees on the result
(`modeling_v2_resolve_on_all_ranks`), or every rank fails.

`_native.available_targets()` lists what the installed wheel supports.

## CI

**CBTS rule.** `jenkins/scripts/cbts/cbts/rules/modeling_v2.py` claims both
trees. What it selects depends on what the PR changes:

| PR changes only… | CI runs… |
|---|---|
| One family (`models/<family>/` and/or `cpp/tensorrt_llm/modeling_v2/<familyCamel>/`) | That family's test-db entries plus the modeling_v2 CPU checks |
| Shared modeling_v2 code (catalog, loader, resolver, `common/`, the CMake module, `contractSurface.txt`) | Every modeling_v2 entry |
| Stock code | The full suite |

**Why the narrowing is safe.** It relies on these CPU checks, which run in
every one of those scopes:

- `test_modeling_v2_native_claims.py`:
  - stock never references a target;
  - targets reach only themselves, their family's `shared/`, `common/` and
    the contract surface;
  - every op has a fake;
  - `modeling_v2` defaults to `off`;
  - families keep their own test-db entries.
- `test_modeling_v2_native_loader.py`: the loader's decisions, and a load
  smoke test of every family library the wheel ships.
- `test_modeling_v2_rank_agreement.py`: all ranks resolve to the same target,
  or all of them fail.

**Pre-merge entries.** Every pre-merge modeling_v2 entry also runs on every
stock PR that falls back to the full suite. Keep a family's pre-merge entry
to one short smoke test on hardware matching its SM, and put the accuracy
matrix post-merge.
