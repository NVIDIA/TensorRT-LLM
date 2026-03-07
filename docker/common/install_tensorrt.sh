#!/bin/bash

set -ex

TRT_VER="10.14.1.48"
# Align with the pre-installed cuDNN / cuBLAS / NCCL versions from
# https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-25-12.html#rel-25-12
CUDA_VER="13.1" # 13.1.0
# Keep the installation for cuDNN if users want to install PyTorch with source codes.
# PyTorch 2.x can compile with cuDNN v9.
CUDNN_VER="9.17.0.29-1"
NCCL_VER="2.28.9-1+cuda13.0"
CUBLAS_VER="13.2.0.9-1"
# Align with the pre-installed CUDA / NVCC / NVRTC versions from
# https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html
NVRTC_VER="13.1.80-1"
CUDA_RUNTIME="13.1.80-1"
CUDA_DRIVER_VERSION="590.44.01-1.el8"

for i in "$@"; do
    case $i in
        --TRT_VER=?*) TRT_VER="${i#*=}";;
        --CUDA_VER=?*) CUDA_VER="${i#*=}";;
        --CUDNN_VER=?*) CUDNN_VER="${i#*=}";;
        --NCCL_VER=?*) NCCL_VER="${i#*=}";;
        --CUBLAS_VER=?*) CUBLAS_VER="${i#*=}";;
        *) ;;
    esac
    shift
done

NVCC_VERSION_OUTPUT=$(nvcc --version)
if [[ $(echo $NVCC_VERSION_OUTPUT | grep -oP "\d+\.\d+" | head -n 1) != ${CUDA_VER} ]]; then
  echo "The version of pre-installed CUDA is not equal to ${CUDA_VER}."
fi

install_ubuntu_requirements() {
    echo "Installing Ubuntu requirements..."
    echo "Updating package lists and installing prerequisites..."
    apt-get update && apt-get install -y --no-install-recommends gnupg2 curl ca-certificates

    ARCH=$(uname -m)
    if [ "$ARCH" = "amd64" ];then ARCH="x86_64";fi
    if [ "$ARCH" = "aarch64" ];then ARCH="sbsa";fi
    echo "Detected architecture: ${ARCH}"

    echo "Downloading and installing CUDA keyring..."
    curl -fsSLO https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/${ARCH}/cuda-keyring_1.1-1_all.deb
    dpkg -i cuda-keyring_1.1-1_all.deb
    rm cuda-keyring_1.1-1_all.deb

    echo "Updating package lists with CUDA repositories..."
    apt-get update

    echo "Checking cuDNN version..."
    if [[ $(apt list --installed | grep libcudnn9) ]] && \
       [[ ! $(apt list --installed | grep "libcudnn9-cuda-13/.* ${CUDNN_VER} ") ]]; then
      echo "Removing incompatible cuDNN version..."
      apt-get remove --purge -y libcudnn9*
    else
      echo "cuDNN version check passed or not installed."
    fi

    echo "Checking NCCL version..."
    if [[ $(apt list --installed | grep libnccl) ]] && \
       [[ ! $(apt list --installed | grep "libnccl2/.* ${NCCL_VER} ") ]]; then
      echo "Removing incompatible NCCL version..."
      apt-get remove --purge -y --allow-change-held-packages libnccl*
    else
      echo "NCCL version check passed or not installed."
    fi

    CUBLAS_CUDA_VERSION=$(echo $CUDA_VER | sed 's/\./-/g')
    echo "Checking cuBLAS version..."
    if [[ $(apt list --installed | grep libcublas) ]] && \
       [[ ! $(apt list --installed | grep "libcublas-${CUBLAS_CUDA_VERSION}/.* ${CUBLAS_VER} ") ]]; then
      echo "Removing incompatible cuBLAS version..."
      apt-get remove --purge -y --allow-change-held-packages libcublas*
    else
      echo "cuBLAS version check passed or not installed."
    fi

    NVRTC_CUDA_VERSION=$(echo $CUDA_VER | sed 's/\./-/g')
    echo "Checking NVRTC installation..."
    if [[ $(apt list --installed | grep cuda-nvrtc-dev) ]]; then
      echo "Removing existing NVRTC version (always reinstall)..."
      apt-get remove --purge -y --allow-change-held-packages cuda-nvrtc-dev*
    else
      echo "No NVRTC installation found."
    fi

    echo "Installing CUDA libraries (cuDNN ${CUDNN_VER}, NCCL ${NCCL_VER}, cuBLAS ${CUBLAS_VER}, NVRTC ${NVRTC_VER})..."
    apt-get install -y --no-install-recommends \
        libcudnn9-cuda-13=${CUDNN_VER} \
        libcudnn9-dev-cuda-13=${CUDNN_VER} \
        libcudnn9-headers-cuda-13=${CUDNN_VER} \
        libnccl2=${NCCL_VER} \
        libnccl-dev=${NCCL_VER} \
        libcublas-${CUBLAS_CUDA_VERSION}=${CUBLAS_VER} \
        libcublas-dev-${CUBLAS_CUDA_VERSION}=${CUBLAS_VER} \
        cuda-nvrtc-dev-${NVRTC_CUDA_VERSION}=${NVRTC_VER}

    echo "Cleaning up apt cache..."
    apt-get clean
    rm -rf /var/lib/apt/lists/*
    echo "Ubuntu requirements installation completed."
}

install_rockylinux_requirements() {
    CUBLAS_CUDA_VERSION=$(echo $CUDA_VER | sed 's/\./-/g')

    ARCH=$(uname -m)
    if [ "$ARCH" = "x86_64" ];then ARCH1="x86_64" && ARCH2="x64" && ARCH3=$ARCH1;fi
    if [ "$ARCH" = "aarch64" ];then ARCH1="aarch64" && ARCH2="aarch64sbsa" && ARCH3="sbsa";fi

    # Download and install packages
    for pkg in \
        "libnccl-${NCCL_VER}.${ARCH1}" \
        "libnccl-devel-${NCCL_VER}.${ARCH1}" \
        "cuda-compat-${CUBLAS_CUDA_VERSION}-${CUDA_DRIVER_VERSION}.${ARCH1}" \
        "cuda-toolkit-${CUBLAS_CUDA_VERSION}-config-common-${CUDA_RUNTIME}.noarch" \
        "cuda-toolkit-13-config-common-${CUDA_RUNTIME}.noarch" \
        "cuda-toolkit-config-common-${CUDA_RUNTIME}.noarch" \
        "libcublas-${CUBLAS_CUDA_VERSION}-${CUBLAS_VER}.${ARCH1}" \
        "libcublas-devel-${CUBLAS_CUDA_VERSION}-${CUBLAS_VER}.${ARCH1}"; do
        wget --retry-connrefused --timeout=180 --tries=10 --continue "https://developer.download.nvidia.com/compute/cuda/repos/rhel8/${ARCH3}/${pkg}.rpm"
    done

    # Remove old packages
    dnf remove -y "libnccl*"

    # Install new packages
    dnf -y install \
        libnccl-${NCCL_VER}.${ARCH1}.rpm \
        libnccl-devel-${NCCL_VER}.${ARCH1}.rpm \
        cuda-compat-${CUBLAS_CUDA_VERSION}-${CUDA_DRIVER_VERSION}.${ARCH1}.rpm \
        cuda-toolkit-${CUBLAS_CUDA_VERSION}-config-common-${CUDA_RUNTIME}.noarch.rpm \
        cuda-toolkit-13-config-common-${CUDA_RUNTIME}.noarch.rpm \
        cuda-toolkit-config-common-${CUDA_RUNTIME}.noarch.rpm \
        libcublas-${CUBLAS_CUDA_VERSION}-${CUBLAS_VER}.${ARCH1}.rpm \
        libcublas-devel-${CUBLAS_CUDA_VERSION}-${CUBLAS_VER}.${ARCH1}.rpm

    # Clean up
    rm -f *.rpm
    dnf clean all
    nvcc --version
}

install_tensorrt() {
    local is_ubuntu=$1
    PY_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[0:2])))')
    PARSED_PY_VERSION=$(echo "${PY_VERSION//./}")

    TRT_CUDA_VERSION=${CUDA_VER}
    # No CUDA 13.1 version for TensorRT yet. Use CUDA 13.0 package instead.
    if [ "$CUDA_VER" = "13.1" ]; then
        TRT_CUDA_VERSION="13.0"
    fi
    TRT_VER_SHORT=$(echo $TRT_VER | cut -d. -f1-3)

    # Check if TensorRT is already installed with the correct version on Ubuntu
    SKIP_INSTALL=false
    if [ "$is_ubuntu" = "true" ]; then
        echo "Checking for existing TensorRT installation..."
        if dpkg -s tensorrt-dev &>/dev/null; then
            INSTALLED_TRT_VER=$(dpkg -s tensorrt-dev | grep '^Version:' | awk '{print $2}' | cut -d'-' -f1)
            echo "Found TensorRT version: ${INSTALLED_TRT_VER}"
            if [ "$INSTALLED_TRT_VER" = "$TRT_VER" ]; then
                echo "TensorRT ${TRT_VER} C++ libraries are installed."
                # Also check if Python package is installed
                echo "Checking for TensorRT Python package..."
                if python3 -c "import tensorrt; print(tensorrt.__version__)" &>/dev/null; then
                    PYTHON_TRT_VER=$(python3 -c "import tensorrt; print(tensorrt.__version__)" 2>/dev/null)
                    echo "Found TensorRT Python package version: ${PYTHON_TRT_VER}"
                    if [ "$PYTHON_TRT_VER" = "$TRT_VER" ]; then
                        echo "TensorRT Python package ${TRT_VER} is already installed. Skipping reinstallation."
                        SKIP_INSTALL=true
                    else
                        echo "Python package version ${PYTHON_TRT_VER} does not match required version ${TRT_VER}. Proceeding with installation."
                    fi
                else
                    echo "TensorRT Python package not found. Proceeding with installation."
                fi
            else
                echo "Installed version ${INSTALLED_TRT_VER} does not match required version ${TRT_VER}. Proceeding with installation."
            fi
        else
            echo "No existing TensorRT installation found. Proceeding with installation."
        fi
    fi

    if [ "$SKIP_INSTALL" = "false" ]; then
        echo "Installing TensorRT ${TRT_VER}..."
        # Remove previous TRT installation
        if [ "$is_ubuntu" = "true" ]; then
            if [[ $(apt list --installed | grep tensorrt) ]]; then
                echo "Removing existing tensorrt packages..."
                apt-get remove --purge -y tensorrt*
            fi
            if [[ $(apt list --installed | grep libnvinfer) ]]; then
                echo "Removing existing libnvinfer packages..."
                apt-get remove --purge -y libnvinfer*
            fi
        fi
        echo "Uninstalling TensorRT Python package..."
        pip3 uninstall -y tensorrt

        if [ -z "$RELEASE_URL_TRT" ];then
            ARCH=${TRT_TARGETARCH}
            if [ -z "$ARCH" ];then ARCH=$(uname -m);fi
            if [ "$ARCH" = "arm64" ];then ARCH="aarch64";fi
            if [ "$ARCH" = "amd64" ];then ARCH="x86_64";fi
            RELEASE_URL_TRT="https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/${TRT_VER_SHORT}/tars/TensorRT-${TRT_VER}.Linux.${ARCH}-gnu.cuda-${TRT_CUDA_VERSION}.tar.gz"
        fi

        echo "Downloading TensorRT from ${RELEASE_URL_TRT}..."
        wget --retry-connrefused --timeout=180 --tries=10 --continue ${RELEASE_URL_TRT} -O /tmp/TensorRT.tar
        echo "Extracting TensorRT to /usr/local/tensorrt..."
        tar -xf /tmp/TensorRT.tar -C /usr/local/
        mv /usr/local/TensorRT-${TRT_VER} /usr/local/tensorrt
        echo "Installing TensorRT Python wheel..."
        pip3 install --no-cache-dir /usr/local/tensorrt/python/tensorrt-*-cp${PARSED_PY_VERSION}-*.whl
        rm -rf /tmp/TensorRT.tar

        echo "Removing static libraries..."
        rm -f /usr/local/tensorrt/lib/libnvinfer_vc_plugin_static.a \
              /usr/local/tensorrt/lib/libnvinfer_plugin_static.a \
              /usr/local/tensorrt/lib/libnvinfer_static.a \
              /usr/local/tensorrt/lib/libnvinfer_dispatch_static.a \
              /usr/local/tensorrt/lib/libnvinfer_lean_static.a \
              /usr/local/tensorrt/lib/libnvonnxparser_static.a \
              /usr/local/tensorrt/lib/libnvinfer_builder_resource_win.so.*
        echo "TensorRT installation completed."

        # Ensure LD_LIBRARY_PATH is set
        if ! grep -q "LD_LIBRARY_PATH=/usr/local/tensorrt/lib" "${ENV}" 2>/dev/null; then
            echo "Setting LD_LIBRARY_PATH in ${ENV}..."
            echo 'export LD_LIBRARY_PATH=/usr/local/tensorrt/lib:$LD_LIBRARY_PATH' >> "${ENV}"
        else
            echo "LD_LIBRARY_PATH already configured."
        fi
    fi
}

# Install base packages depending on the base OS
ID=$(grep -oP '(?<=^ID=).+' /etc/os-release | tr -d '"')
case "$ID" in
  ubuntu)
    install_ubuntu_requirements
    install_tensorrt true
    ;;
  rocky)
    install_rockylinux_requirements
    install_tensorrt false
    ;;
  *)
    echo "Unable to determine OS..."
    exit 1
    ;;
esac
