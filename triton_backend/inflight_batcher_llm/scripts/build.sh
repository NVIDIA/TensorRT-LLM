#!/bin/bash

Help()
{
   # Display Help
   echo "Syntax: build.sh [h|-t <trt_root>|u]"
   echo "options:"
   echo "h     Print this Help."
   echo "t     Location of tensorrt library"
   echo "u     Option to build unit tests"
   echo
}

TRT_ROOT='/usr/local/tensorrt'
BUILD_UNIT_TESTS='false'

# Get the options
while getopts ":ht:u" option; do
   case $option in
      h) # display Help
         Help
         exit;;
      t) # Location of tensorrt
         TRT_ROOT=$OPTARG;;
      u) # Option to build unit tests
         BUILD_UNIT_TESTS='true';;
     \?) # Invalid option
         echo "Error: Invalid option"
         echo ""
         Help
         exit;;
   esac
done

echo "Using TRT_ROOT=${TRT_ROOT}"
echo "Using BUILD_UNIT_TESTS=${BUILD_UNIT_TESTS}"

set -x
apt-get update
apt-get install -y --no-install-recommends rapidjson-dev

BUILD_DIR=$(dirname $0)/../build
mkdir $BUILD_DIR
BUILD_DIR=$(cd -- "$BUILD_DIR" && pwd)
cd $BUILD_DIR

export LD_LIBRARY_PATH="/usr/local/cuda/compat/lib.real:${LD_LIBRARY_PATH}"

BUILD_TESTS_ARG="-DUSE_CXX11_ABI=ON"
if [[ "$BUILD_UNIT_TESTS" == "true" ]]; then
  BUILD_TESTS_ARG="-DBUILD_TESTS=ON -DUSE_CXX11_ABI=ON"
fi

# TODO: Remove specifying Triton version after cmake version is upgraded to 3.31.8
# Get TRITON_SHORT_TAG from docker/Dockerfile.multi
script_dir="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd -P)"
LLM_ROOT="$(cd -- "$script_dir/../../.." >/dev/null 2>&1 && pwd -P)"
TRITON_SHORT_TAG=$("$LLM_ROOT/jenkins/scripts/get_triton_tag.sh" "$LLM_ROOT")
cmake -DCMAKE_INSTALL_PREFIX:PATH=`pwd`/install ${BUILD_TESTS_ARG} -DTRITON_COMMON_REPO_TAG=${TRITON_SHORT_TAG} -DTRITON_CORE_REPO_TAG=${TRITON_SHORT_TAG} -DTRITON_THIRD_PARTY_REPO_TAG=${TRITON_SHORT_TAG} -DTRITON_BACKEND_REPO_TAG=${TRITON_SHORT_TAG} ..
make install

mkdir -p /opt/tritonserver/backends/tensorrtllm
cp libtriton_tensorrtllm.so /opt/tritonserver/backends/tensorrtllm
cp trtllmExecutorWorker /opt/tritonserver/backends/tensorrtllm
