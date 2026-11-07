# `3rdparty/`

This directory holds TensorRT-LLM's third-party dependency metadata and
tooling. C++ dependencies are driven by CMake `FetchContent` from
`fetch_content.json`; source vendors use a generated lock and patches. It also
contains tooling that accelerates repeat clones of C++ dependencies.

## Adding new third-party dependencies

The markdown files in this directory contain playbooks for how to add new
third-party dependencies. Please see the document that matches the kind of
dependency you want to add:

* For C++ dependencies compiled into the extension modules via the cmake build
  and re-distributed with the wheel, see [cpp-thirdparty.md](cpp-thirdparty.md)
* For python dependencies declared via wheel metadata and installed in the
  container via pip, see [py-thirdparty.md](py-thirdparty.md)
* For source trees copied into this repository and pinned to an upstream Git
  commit, see [vendor-sources.md](vendor-sources.md)

## FetchContent cache (`--use-3rdparty-cache`)

`scripts/build_wheel.py --use-3rdparty-cache` enables a local bare-repo
cache that accelerates cmake `FetchContent` clones via
`git clone --reference`. It is opt-in; without the flag, no `-D` is
added and the cache code path in `3rdparty/CMakeLists.txt` is bypassed
entirely.

For flow, cache layout, threat model, and design rationale, see
[fetch-cache.md](fetch-cache.md).

## Co-develop third-party dependencies with TensorRT-LLM

The automatic dependency management provided by cmake `FetchContent` is
optimized for normal developers that use the dependencies as-is. If you need to
develop a dependency alongside TensorRT-LLM, point CMake at your own
checkout instead:

```bash
python scripts/build_wheel.py ... \
  --extra-cmake-vars FETCHCONTENT_SOURCE_DIR_DEEPGEMM=/path/to/DeepGEMM
```

The variable name is `FETCHCONTENT_SOURCE_DIR_` followed by the upper-cased
dependency `name`. CMake then skips the download, update and patch steps for that
dependency and never touches your checkout, so you must apply the patches
yourself, if there's one for the dependency you are working on.
