#!/bin/bash

set -ex
TRITON_VER="r25.03"

for i in "$@"; do
    case $i in
        --TRITON_VER=?*) TRITON_VER="${i#*=}";;
        *) ;;
    esac
    shift
done

install_boost() {
	# Install boost version >= 1.78 for boost::span
	# Current libboost-dev apt packages are < 1.78, so install from tar.gz
	wget -O /tmp/boost.tar.gz \
	          https://archives.boost.io/release/1.80.0/source/boost_1_80_0.tar.gz \
	      && (cd /tmp && tar xzf boost.tar.gz) \
	      && mv /tmp/boost_1_80_0/boost /usr/include/boost
}

install_triton() {
    apt-get update && apt-get install -y \
        pigz \
        libxml2-dev \
        libre2-dev \
        libnuma-dev \
        python3-build \
        libb64-dev \
        libarchive-dev

    ARCH=$(uname -m)
    if [ "$ARCH" = "amd64" ];then ARCH="x86_64";fi
    if [ "$ARCH" = "aarch64" ];then ARCH="sbsa";fi

	curl -o /tmp/cuda-keyring.deb \
          https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/${ARCH}/cuda-keyring_1.0-1_all.deb \
      && apt install /tmp/cuda-keyring.deb \
      && rm /tmp/cuda-keyring.deb \
      && apt-get update \
      && apt-get install -y datacenter-gpu-manager=1:3.2.6

    install_boost
    pip3 install distro
    cd /tmp
    git clone https://github.com/triton-inference-server/server -b $TRITON_VER
    cd server
    python3 build.py --repo-tag=common:$TRITON_VER \
      --repo-tag=core:$TRITON_VER --repo-tag=backend:$TRITON_VER \
      --repo-tag=thirdparty:$TRITON_VER --backend=python:$TRITON_VER \
      --repoagent=checksum:$TRITON_VER --enable-gpu --enable-stats --enable-metrics \
      --enable-logging --enable-cpu-metrics \
      --no-container-build --build-dir=`pwd`/build \
      --endpoint grpc --endpoint http --endpoint sagemaker --endpoint vertex-ai \
      --filesystem gcs --filesystem s3 --filesystem azure_storage --backend=ensemble:$TRITON_VER

    mv ./build/opt/tritonserver/ /opt/tritonserver/
    rm -rf /tmp/boost_1_80_0
    rm -rf /tmp/boost.tar.gz
    rm -rf ./build
}

# Install Triton only if base image in Ubuntu
ID=$(grep -oP '(?<=^ID=).+' /etc/os-release | tr -d '"')
if [ "$BUILD_TRITON" == "1" ]; then
    if [ "$ID" == "ubuntu" ]; then
        install_triton
    else
        echo "Skip Triton installation for non-Ubuntu base image"
    fi
fi
