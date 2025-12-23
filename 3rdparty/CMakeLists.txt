include(ExternalProject)
include(FetchContent)

if(DEFINED ENV{GITHUB_MIRROR} AND NOT "$ENV{GITHUB_MIRROR}" STREQUAL "")
  set(github_base_url "$ENV{GITHUB_MIRROR}")
else()
  set(github_base_url "https://github.com")
endif()

FetchContent_Declare(
  cppzmq
  GIT_REPOSITORY https://github.com/zeromq/cppzmq
  GIT_TAG v4.10.0 # c94c20743ed7d4aa37835a5c46567ab0790d4acc
  GIT_SHALLOW TRUE
  # NOTE: TensorRT-LLM only uses the headers
  SOURCE_SUBDIR
  dont-add-this-project-with-add-subdirectory)

FetchContent_Declare(
  cutlass
  GIT_REPOSITORY https://github.com/NVIDIA/cutlass
  GIT_TAG v4.3.0 # e67e63c331d6e4b729047c95cf6b92c8454cba89
  GIT_SHALLOW TRUE
  SOURCE_SUBDIR
  dont-add-this-project-with-add-subdirectory)

FetchContent_Declare(
  cxxopts
  GIT_REPOSITORY https://github.com/jarro2783/cxxopts
  GIT_TAG v3.1.1 # eb787304d67ec22f7c3a184ee8b4c481d04357fd
  GIT_SHALLOW TRUE)

set(deep_ep_commit 5be51b228a7c82dbdb213ea58e77bffd12b38af8)
set_property(GLOBAL PROPERTY DEEP_EP_COMMIT "${deep_ep_commit}")
FetchContent_Declare(
  deep_ep_download
  URL ${github_base_url}/deepseek-ai/DeepEP/archive/${deep_ep_commit}.tar.gz)

FetchContent_Declare(
  deepgemm
  GIT_REPOSITORY https://github.com/deepseek-ai/DeepGEMM
  GIT_TAG 4ff3f54d9b7ed3129e4f36f9871232ea7ecab86b # nv_dev branch
  GIT_SUBMODULES_RECURSE
  ON
  SOURCE_SUBDIR
  dont-add-this-project-with-add-subdirectory)

FetchContent_Declare(
  eigen
  GIT_REPOSITORY https://github.com/libeigen/eigen
  GIT_TAG 3.4.0
  GIT_SHALLOW TRUE)

FetchContent_Declare(
  flashmla
  GIT_REPOSITORY https://github.com/deepseek-ai/FlashMLA.git
  GIT_TAG 1408756a88e52a25196b759eaf8db89d2b51b5a1
  GIT_SUBMODULES_RECURSE
  ON
  SOURCE_SUBDIR
  dont-add-this-project-with-add-subdirectory)

FetchContent_Declare(
  googlebenchmark
  GIT_REPOSITORY https://github.com/google/benchmark
  GIT_TAG v1.8.3
  GIT_SHALLOW TRUE)

FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest
  GIT_TAG v1.15.2
  GIT_SHALLOW TRUE)

FetchContent_Declare(
  json
  GIT_REPOSITORY https://github.com/nlohmann/json
  GIT_TAG v3.12.0 # 55f93686c01528224f448c19128836e7df245f72
  GIT_SHALLOW TRUE
  SOURCE_SUBDIR
  dont-add-this-project-with-add-subdirectory)

FetchContent_Declare(
  nanobind
  GIT_REPOSITORY https://github.com/wjakob/nanobind
  GIT_TAG a0ed2587f1089ef7657e2ed49ad6756b01c74e9f)

FetchContent_Declare(
  nvtx
  GIT_REPOSITORY https://github.com/NVIDIA/NVTX
  GIT_TAG v3.1.0-c-cpp # a1ceb0677f67371ed29a2b1c022794f077db5fe7
  GIT_SHALLOW TRUE
  # NOTE: TensorRT-LLM only uses the headers
  SOURCE_SUBDIR
  dont-add-this-project-with-add-subdirectory)

FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11
  GIT_TAG f99ffd7e03001810a3e722bf48ad1a9e08415d7d)

FetchContent_Declare(
  ucxx
  GIT_REPOSITORY https://github.com/rapidsai/ucxx
  GIT_TAG 16eaa57c8d98c8ef54d666a2d2b11e76cfa565f5
  # NOTE: See the notes in cpp/CMakeList.txt where this project is build at
  # configure time and then included via find_package
  SOURCE_SUBDIR
  dont-add-this-project-with-add-subdirectory)

FetchContent_Declare(
  xgrammar
  GIT_REPOSITORY https://github.com/mlc-ai/xgrammar
  GIT_TAG v0.1.25 # e4e816f5f0fe39f5b1601a17a4552307fa3b70ff
  GIT_SHALLOW TRUE
  # NOTE: TensorRT-LLM only uses the headers
  SOURCE_SUBDIR
  dont-add-this-project-with-add-subdirectory)
