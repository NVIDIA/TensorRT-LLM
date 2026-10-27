<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Native code of modeling_v2 targets

C++/CUDA that only modeling_v2 targets use lives with the Python code of its
model family, in this directory tree, not in `cpp/tensorrt_llm/kernels/` or
`thop/`. Each family builds into one torch-op library,
`libth_modeling_v2_<family>.so`, installed next to `libth_common.so`.

- **Not linked into stock.** Nothing in a family library is linked into
  `libtensorrt_llm.so` or `libth_common.so`.
- **Loaded on demand.** The library opens only when routing picks one of the
  family's targets. With `modeling_v2=off` (the default), nothing loads.
- **No stock file per model.** Adding a model, its kernels included, edits
  nothing under `cpp/`, nor `setup.py` or `scripts/build_wheel.py`: the build
  finds every `models/<family>/csrc/CMakeLists.txt` by itself.

`models/gpt_oss/` is the worked example: its target
`gpt_oss_120b__sm_10x__default` carries one kernel, the MXFP4 block-scale
swizzle its weight loader runs.

## Onboarding a model with native code

The files a new family `foo` with one target adds. Everything except the last
three rows is inside its own directories:

```
tensorrt_llm/_torch/_experimental/modeling_v2/models/foo/
  __init__.py                       docstring only
  routing.py                        @routes("FooForCausalLM") def route(...)
  fake_ops.py                       fakes of the family's shared ops, if it has any
  csrc/
    CMakeLists.txt                  add_modeling_v2_family / _shared / _target
    *.cu *.cuh *.h *.cpp            code several targets of the family share, if any
  foo_nvfp4__sm_103__tp4/           <checkpoint>__<sm>__<parallel>
    __init__.py                     native_ops = load_target_ops(__name__)
    modeling.py  weights.py
    fake_ops.py                     a fake for every op of csrc/
    ops/<category>/<entry>.py       entries only this target calls
    csrc/
      *.cu *.cuh *.h                kernels, compiled for ARCHS without torch
      *Op.cpp                       torch bindings
tests/unittest/_torch/modeling_v2/models/foo/foo_nvfp4__sm_103__tp4/test_modeling_v2_*.py
tests/integration/defs/accuracy/modeling_v2/test_foo.py

models/__init__.py                  + from .foo import routing as foo_routing
tests/integration/test_lists/test-db/l0_<gpu>.yml
                                    + unittest/_torch/modeling_v2/models/foo
.github/CODEOWNERS                  optional: the family's owners
```

Naming is the Python package names throughout: family `foo`, target
`foo_nvfp4__sm_103__tp4`, its torch namespace `mv2_foo_foo_nvfp4_sm_103_tp4`.
`tests/unittest/_torch/modeling_v2/test_modeling_v2_native_claims.py` checks
every rule below on CPU; run it before sending the PR.

## Step by step

The gpt_oss files are the example for each step.

1. **Declare the family and the target** in `models/foo/csrc/CMakeLists.txt`
   (`models/gpt_oss/csrc/CMakeLists.txt`):

   ```cmake
   add_modeling_v2_family(foo)

   add_modeling_v2_target(
     foo foo_nvfp4__sm_103__tp4
     ARCHS 103a
     SOURCES *.cu *.cpp
     REQUIRES MULTI_DEVICE)
   ```

2. **Write the kernel** in `foo_nvfp4__sm_103__tp4/csrc/`, in the target's C++
   namespace (`gpt_oss_120b__sm_10x__default/csrc/swizzleBlockScales.{h,cu}`):

   ```cpp
   #include "tensorrt_llm/modeling_v2/common/namespace.h"

   TRTLLM_MV2_NAMESPACE_BEGIN   // tensorrt_llm::modeling_v2::foo::foo_nvfp4_sm_103_tp4
   namespace kernels
   {
   cudaError_t invokeMyOp(/* ... */, cudaStream_t stream);
   }
   TRTLLM_MV2_NAMESPACE_END
   ```

3. **Bind it to torch** in the same `csrc/`, through the target's torch
   namespace (`swizzleBlockScalesOp.cpp`). Include the family's own headers
   from `models/`:

   ```cpp
   #include "foo/foo_nvfp4__sm_103__tp4/csrc/myOp.h"
   #include "tensorrt_llm/modeling_v2/common/torchLibrary.h"

   TRTLLM_MV2_LIBRARY_FRAGMENT(m) { m.def("my_op(Tensor x) -> Tensor"); }
   TRTLLM_MV2_LIBRARY_IMPL(CUDA, m) { m.impl("my_op", &TRTLLM_MV2_NS::myOp); }
   ```

4. **Load it** from the target package's `__init__.py`:

   ```python
   from tensorrt_llm._torch._experimental.modeling_v2._native import load_target_ops

   native_ops = load_target_ops(__name__)  # torch.ops.mv2_foo_foo_nvfp4_sm_103_tp4
   ```

   The target's modules use it as `from . import native_ops`
   (`gpt_oss_120b__sm_10x__default/weights.py`). The name is not `ops`: that
   is the target's `ops/` package.

5. **Fake every op** in the target's `fake_ops.py`; the loader imports it right
   after it finds the target in the library:

   ```python
   from tensorrt_llm._torch._experimental.modeling_v2._native import namespace_of

   _NS = namespace_of(__package__)

   @torch.library.register_fake(f"{_NS}::my_op")
   def _(x):
       return torch.empty_like(x)
   ```

6. **Test it** under `tests/unittest/_torch/modeling_v2/models/foo/`, and list
   the family's directory in the test list of the GPU its SM names; accuracy
   gates go in `accuracy/modeling_v2/test_foo.py`
   (`test_modeling_v2_gpt_oss_swizzle_block_scales.py`, and
   `unittest/_torch/modeling_v2/models/gpt_oss` in `l0_b300.yml`). CBTS
   selects a family's tests by these two names (see [CI](#ci)).

## Code the family's targets share

When several targets call the same op, its code goes once in the family's
`csrc/`, next to its `CMakeLists.txt`, and its ops are in the family's torch
namespace, `torch.ops.mv2_<family>`:

```cmake
add_modeling_v2_shared(
  foo
  ARCHS 100f
  SOURCES *.cu *.cpp
  REQUIRES MULTI_DEVICE)

# Targets with no code of their own declare no SOURCES and have no csrc/;
# declaring them is what makes the family library carry them.
add_modeling_v2_target(foo foo_nvfp4__sm_100__tp8 ARCHS 100f)
```

- **Fakes** of the shared ops go in the family's `models/foo/fake_ops.py`,
  with `torch_namespace(family_of(__name__))`; the loader imports it when it
  opens the library.
- **Callers** anywhere in the family use
  `load_family_ops(__name__).my_op(...)`.
- **A target without code of its own** is skipped whenever the shared code is,
  so the library never carries a target whose ops it lacks.

Sharing moves code up, never sideways: target `csrc/`, then the family's
`csrc/`, then stock (`cpp/tensorrt_llm/kernels` + `thop`) once a built-in
model calls it, through normal stock review. A target never includes or calls
a sibling target's code.

## CMake API

Defined in `cpp/cmake/modules/modeling_v2.cmake`, in the style of
`add_th_common_family`.

| Function | What it declares |
|---|---|
| `add_modeling_v2_family(<family>)` | The family; first call in `models/<family>/csrc/CMakeLists.txt`. |
| `add_modeling_v2_target(<family> <target> ARCHS ... [SOURCES ...] [DEPS ...] [REQUIRES ...])` | One target's code, in `<target>/csrc/`; without `SOURCES`, a target that runs only its family's shared code. |
| `add_modeling_v2_shared(<family> ARCHS ... SOURCES ... [DEPS ...] [REQUIRES ...])` | Code several targets call, in the family's `csrc/`; its ops are in `torch.ops.mv2_<family>`. |

| Keyword | Meaning |
|---|---|
| `ARCHS` | `NN`, `NNa` or `NNf`, intersected with the build's requested architectures, as flash_mla and deep_ep choose theirs. A target whose intersection is empty is **not built** (the configure step prints why) and is a `NoTarget` at run time. The architectures must cover the SM in the target's name. |
| `SOURCES` | Paths or globs relative to the unit's `csrc/`. `.cu` files compile for `ARCHS` without torch; `.cpp` files are the torch bindings. A per-file option goes on the file in the family's `CMakeLists.txt`: `set_source_files_properties(x.cu PROPERTIES COMPILE_OPTIONS --use_fast_math)`. |
| `DEPS` | `cublas`, `cublaslt`, `cuda_driver`, `internal_cutlass`, `nccl`, `trtllm_gen_fmha_interface`. Anything else is an addition to the module. |
| `REQUIRES` | `MULTI_DEVICE`, `NVSHMEM`. A target the build cannot satisfy is skipped. |

**Build options.** `-DBUILD_MODELING_V2=OFF` builds no family library.
`-DTRTLLM_MODELING_V2_TARGETS=<family>/<target>;...` (globs allowed, e.g.
`gpt_oss/*`) builds only those; pass it with
`scripts/build_wheel.py -D "TRTLLM_MODELING_V2_TARGETS=..."`. A family library
is its own build target, so a kernel edit relinks only it:
`cmake --build cpp/build --target th_modeling_v2_<family>`.

## Namespaces

| Kind | Name |
|---|---|
| Target torch ops | `torch.ops.mv2_<family>_<target>`, with `__` collapsed to `_` |
| Family shared torch ops | `torch.ops.mv2_<family>`; also holds `built_targets()` |
| Target C++ | `tensorrt_llm::modeling_v2::<family>::<target>` (`TRTLLM_MV2_NAMESPACE_BEGIN`) |
| Shared C++ | `tensorrt_llm::modeling_v2::<family>::shared` |

Each torch namespace is claimed exactly once, by the family library's generated
registration, so a second library claiming it fails at load. Op files only add
fragments through the `TRTLLM_MV2_*` macros; raw `TORCH_LIBRARY*` is rejected.
Every target has its own C++ namespace and the library exports no symbol, so
sibling targets can share file and function names.

## What native code may include

- Its own target's `csrc/` and its family's `csrc/`, written from `models/`:
  `"foo/foo_nvfp4__sm_103__tp4/csrc/x.h"`, `"foo/csrc/y.h"`.
- `tensorrt_llm/modeling_v2/common/` (the macros).
- Any stock header (`cpp/include/`, `cpp/`), and system, CUDA, torch and cutlass
  headers.

It may not include another target's or another family's code.

It links against `libtensorrt_llm.so` (logger, `TllmException`, mcast registry
and other singletons stay there), `th_utils` and torch. **When a target needs
stock to change**, e.g. a new option on a stock kernel, that is a separate
stock PR that leaves stock behavior unchanged by default.

## How it loads

`load_target_ops(__name__)` treats a family library the way stock code treats
its optional native modules (`flash_mla`, `deep_ep`):

| Outcome | Raised as | Meaning |
|---|---|---|
| The library is not installed, or does not carry the target | `NoTarget` | This build did not compile it; `auto` keeps the built-in model, `require` reports why |
| The library is installed but fails to open, or an op a `fake_ops.py` declares is missing | `NativeLoadError` | A broken installation; never a fallback |

Packaging follows `flash_mla` and `deep_ep` too: each family writes
`cuda_architectures.txt` into its build directory, and `scripts/build_wheel.py`
installs the library only when that list is not empty.

## CI

| PR changes only… | CI runs… |
|---|---|
| One family (`models/<family>/`, its `csrc/` included) | That family's test-db entries plus the modeling_v2 CPU checks |
| Shared modeling_v2 code (catalog, loader, resolver, `models/__init__.py`, `cpp/tensorrt_llm/modeling_v2/`, the CMake module) | Every modeling_v2 entry |
| Stock code | The full suite |

The narrowing is safe because family libraries are never in the stock
libraries, and the CPU checks that keep it so run in every scope:
`test_modeling_v2_native_claims.py` (native code only in `csrc/`, stock never
names a family, no include reaches another target or family, every op has a
fake, license headers),
`test_modeling_v2_native_loader.py` and `test_modeling_v2_rank_agreement.py`.
Keep a family's pre-merge entry to one short test on hardware matching its SM;
the accuracy matrix goes post-merge.
