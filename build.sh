#!/bin/bash
# Usage:  put this script under TensorRT-LLM/build.sh, then `bash build.sh [COMPONENT] [MODE]`
# Note: Run launch.sh first. Then run this inside the container.
# Example: `bash build.sh python` the most common build, `sh build.sh cpp` cpp-only build. `sh build.sh all debug` to build TRT-LLM debug mode and ModelOpt.
# COMPONENT: ["", "modelopt", "all"]. if empty, build TRT-LLM only
# MODE: ["", "cpp", "debug", "clean", and ANY COMBO like "cpp-debug", "cpp-clean", "debug-clean"]. if empty, default e2e build including pip install. if cpp, only cpp, cpp tests, cpp benchmarks. if clean, clean build.

PATH=$PATH:$HOME/.local/bin

case "$#" in
  0)
    COMPONENT="trtllm"
    MODE=""
    ;;
  1)
    case "$1" in
      "modelopt"|"all")
        COMPONENT=$1
        MODE=""
        ;;
      *)
        COMPONENT="trtllm"
        MODE=$1
        ;;
    esac
    ;;
  2)
    COMPONENT=$1
    MODE=$2
    ;;
esac

if [ "$COMPONENT" = "trtllm" ] || [ "$COMPONENT" = "all" ]; then
  echo "Building TensorRT LLM"
  # cd /code/tensorrt_llm
  git config --global --add safe.directory '*'
  git submodule update --init --recursive
  # git submodule foreach --recursive git reset --hard # if there is dirty changes, try this


  CMD="--build_type="
  if [[ $MODE == *"debug"* ]]; then CMD+="Debug "; else CMD+="Release "; fi # debug or release build
  if [[ $MODE == *"clean"* ]]; then CMD+="--clean "; fi
  if [[ $MODE == *"python"* ]]; then CMD+="--python_bindings --skip_building_wheel "; else CMD+="--install "; fi # python dev use editable pip install
  if [[ $MODE == *"cpp"* ]]; then CMD+="--cpp_only --benchmarks --extra-make-targets google-tests "; fi # cpp build including benchmarks and cpp tests

  echo $CMD
  ./scripts/build_wheel.py --trt_root /usr/local/tensorrt --build_dir $PWD/cpp/build --use_ccache --extra-cmake-vars FAST_BUILD=ON --cuda_architectures native $CMD --no-venv
  if [[ $MODE == *"python"* ]]; then pip install -e .; fi
fi

if [ "$COMPONENT" = "modelopt" ] || [ "$COMPONENT" = "all" ]; then
  # need to mount modelopt at /code/modelopt first
  echo "Pip editable installing ModelOpt"
  cd /code/modelopt
  pip install -e ".[dev]" --extra-index-url https://pypi.nvidia.com --extra-index-url https://urm.nvidia.com/artifactory/api/pypi/nv-shared-pypi/simple
fi
