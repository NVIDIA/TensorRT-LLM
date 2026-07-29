# `3rdparty/`

This directory holds TensorRT-LLM's third-party C++ dependencies (driven by
cmake `FetchContent` from `fetch_content.json`) plus tooling that
accelerates repeat clones of those dependencies.

## Adding new third-party dependencies

The markdown files in this directory contain playbooks for how to add new
third-party dependencies. Please see the document that matches the kind of
dependency you want to add:

* For C++ dependencies compiled into the extension modules via the cmake build
  and re-distributed with the wheel, see [cpp-thirdparty.md](cpp-thirdparty.md)
* For python dependencies declared via wheel metadata and installed in the
  container via pip, see [py-thirdparty.md](py-thirdparty.md)

## FetchContent cache (`--use-3rdparty-cache`)

`scripts/build_wheel.py --use-3rdparty-cache` enables a local bare-repo
cache that accelerates cmake `FetchContent` clones via
`git clone --reference`. It is opt-in; without the flag, no `-D` is
added and the cache code path in `3rdparty/CMakeLists.txt` is bypassed
entirely.

For flow, cache layout, threat model, and design rationale, see
[fetch-cache.md](fetch-cache.md).
