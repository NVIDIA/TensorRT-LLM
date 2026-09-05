#!/bin/bash

set -ex

# Align with the pre-installed cuDNN / cuBLAS / NCCL versions from
# https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-26-08.html#rel-26-08
CUDA_VER="13.4" # image reports CUDA_VERSION=13.4.1.012
# Keep the installation for cuDNN if users want to install PyTorch with source codes.
# PyTorch 2.x can compile with cuDNN v9.
CUDNN_VER="9.25.0.28-1" # TODO(dlfw-26.08): exact internal build, not yet on the public CUDA apt repo (public max is 9.25.0.15-1)
NCCL_VER="2.30.7-1+cuda13.3"
CUBLAS_VER="13.7.0.27-1"
# Align with the pre-installed CUDA / NVCC / NVRTC versions from
# https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html
NVRTC_VER="13.4.59-1"
CUDA_RUNTIME="13.4.49-1" # TODO(dlfw-26.08): rockylinux only; the public cuda-toolkit-13-4-* rpms are not released yet
CUDA_DRIVER_VERSION="615.65.02-1.el8" # TODO(dlfw-26.08): rockylinux only; taken from the image's CUDA_DRIVER_VERSION, re-check the rpm release suffix once cuda-compat-13-4 is published

for i in "$@"; do
    case $i in
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
    apt-get update && apt-get install -y --no-install-recommends gnupg2 curl ca-certificates
    ARCH=$(uname -m)
    if [ "$ARCH" = "amd64" ];then ARCH="x86_64";fi
    if [ "$ARCH" = "aarch64" ];then ARCH="sbsa";fi

    # this file exists in cuda base image, and has conflicts with cuda-keyring with the following error, so we need to remove it first:
    # E: Conflicting values set for option Signed-By regarding
    # source https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/sbsa/ /: /usr/share/keyrings/cuda-archive-keyring.gpg !=
    rm -f /etc/apt/sources.list.d/cuda.list

    curl -fsSLO https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/${ARCH}/cuda-keyring_1.1-1_all.deb
    dpkg -i cuda-keyring_1.1-1_all.deb
    rm cuda-keyring_1.1-1_all.deb

    apt-get update

    CUDA_MAJOR_VER=$(echo $CUDA_VER | cut -d. -f1)
    CUBLAS_MAJOR_VER=$(echo $CUBLAS_VER | cut -d. -f1)
    NVRTC_CUDA_VERSION=$(echo $CUDA_VER | sed 's/\./-/g')

    # Skip remove+reinstall for any library already at the target version (e.g. pre-installed
    # by the base image at a version not yet published to the public CUDA apt repo).
    installed_pkg_version() {
        dpkg-query -W -f='${Version}' "$1" 2>/dev/null || true
    }

    PKGS_TO_INSTALL=()
    if [[ "$(installed_pkg_version libcudnn9-cuda-13)" != "${CUDNN_VER}" ]]; then
        apt-get remove --purge -y libcudnn9* || true
        PKGS_TO_INSTALL+=(libcudnn9-cuda-13=${CUDNN_VER} libcudnn9-dev-cuda-13=${CUDNN_VER} libcudnn9-headers-cuda-13=${CUDNN_VER})
    fi
    if [[ "$(installed_pkg_version libnccl2)" != "${NCCL_VER}" ]]; then
        apt-get remove --purge -y --allow-change-held-packages libnccl* || true
        PKGS_TO_INSTALL+=(libnccl2=${NCCL_VER} libnccl-dev=${NCCL_VER})
    fi
    # NOTE: package name is libcublas-<CUDA_MAJOR>-<CUDA_MINOR> (matches cuda-nvrtc-dev's
    # naming), not libcublas<CUBLAS_MAJOR>-cuda-<CUDA_MAJOR> (that's a separate, older package).
    if [[ "$(installed_pkg_version libcublas-${NVRTC_CUDA_VERSION})" != "${CUBLAS_VER}" ]]; then
        apt-get remove --purge -y --allow-change-held-packages libcublas* || true
        PKGS_TO_INSTALL+=(libcublas-${NVRTC_CUDA_VERSION}=${CUBLAS_VER} libcublas-dev-${NVRTC_CUDA_VERSION}=${CUBLAS_VER})
    fi
    if [[ "$(installed_pkg_version cuda-nvrtc-dev-${NVRTC_CUDA_VERSION})" != "${NVRTC_VER}" ]]; then
        apt-get remove --purge -y --allow-change-held-packages cuda-nvrtc-dev* || true
        PKGS_TO_INSTALL+=(cuda-nvrtc-dev-${NVRTC_CUDA_VERSION}=${NVRTC_VER})
    fi

    if [ ${#PKGS_TO_INSTALL[@]} -gt 0 ]; then
        apt-get install -y --no-install-recommends "${PKGS_TO_INSTALL[@]}"
    fi

    apt-get clean
    rm -rf /var/lib/apt/lists/*

    # cublas >= 13.4.1.2 installs headers to /usr/include/libcublas/<major>/
    # instead of /usr/local/cuda/include/. Symlink them back for build compatibility.
    CUBLAS_HDR_DIR="/usr/include/libcublas/${CUBLAS_MAJOR_VER}"
    if [ -d "${CUBLAS_HDR_DIR}" ]; then
        for hdr in "${CUBLAS_HDR_DIR}"/*.h; do
            ln -sf "${hdr}" "/usr/local/cuda/include/$(basename ${hdr})"
        done
    fi

    # cublas >= 13.4.1.2 installs .so files to /usr/lib/<arch>-linux-gnu/libcublas/<major>/
    # instead of /usr/local/cuda/lib64/. Symlink them so LD_LIBRARY_PATH=/usr/local/cuda/lib64 finds them.
    ARCH=$(uname -m)
    CUBLAS_LIB_DIR="/usr/lib/${ARCH}-linux-gnu/libcublas/${CUBLAS_MAJOR_VER}"
    if [ -d "${CUBLAS_LIB_DIR}" ]; then
        for lib in "${CUBLAS_LIB_DIR}"/libcublas*.so*; do
            [ -e "${lib}" ] || continue
            target="/usr/local/cuda/lib64/$(basename ${lib})"
            [ -e "${target}" ] || ln -sf "${lib}" "${target}"
        done
    fi
    ldconfig
}

install_rockylinux_requirements() {
    CUBLAS_CUDA_VERSION=$(echo $CUDA_VER | sed 's/\./-/g')
    CUDA_MAJOR_VER=$(echo $CUDA_VER | cut -d. -f1)
    CUBLAS_MAJOR_VER=$(echo $CUBLAS_VER | cut -d. -f1)

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
        "libcublas${CUBLAS_MAJOR_VER}-cuda-${CUDA_MAJOR_VER}-${CUBLAS_VER}.${ARCH1}" \
        "libcublas${CUBLAS_MAJOR_VER}-devel-cuda-${CUDA_MAJOR_VER}-${CUBLAS_VER}.${ARCH1}"; do
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
        libcublas${CUBLAS_MAJOR_VER}-cuda-${CUDA_MAJOR_VER}-${CUBLAS_VER}.${ARCH1}.rpm \
        libcublas${CUBLAS_MAJOR_VER}-devel-cuda-${CUDA_MAJOR_VER}-${CUBLAS_VER}.${ARCH1}.rpm

    # Clean up
    rm -f *.rpm
    dnf clean all
    nvcc --version

    # cublas >= 13.4.1.2 installs headers to /usr/include/libcublas/<major>/
    # instead of /usr/local/cuda/include/. Symlink them back for build compatibility.
    CUBLAS_HDR_DIR="/usr/include/libcublas/${CUBLAS_MAJOR_VER}"
    if [ -d "${CUBLAS_HDR_DIR}" ]; then
        for hdr in "${CUBLAS_HDR_DIR}"/*.h; do
            ln -sf "${hdr}" "/usr/local/cuda/include/$(basename ${hdr})"
        done
    fi

    # cublas >= 13.4.1.2 installs .so files to /usr/lib64/libcublas/<major>/
    # instead of /usr/local/cuda/lib64/. Symlink them so LD_LIBRARY_PATH=/usr/local/cuda/lib64 finds them.
    CUBLAS_LIB_DIR="/usr/lib64/libcublas/${CUBLAS_MAJOR_VER}"
    if [ -d "${CUBLAS_LIB_DIR}" ]; then
        for lib in "${CUBLAS_LIB_DIR}"/libcublas*.so*; do
            [ -e "${lib}" ] || continue
            target="/usr/local/cuda/lib64/$(basename ${lib})"
            [ -e "${target}" ] || ln -sf "${lib}" "${target}"
        done
    fi
    ldconfig
}

# Install base packages depending on the base OS
ID=$(grep -oP '(?<=^ID=).+' /etc/os-release | tr -d '"')
case "$ID" in
  ubuntu)
    install_ubuntu_requirements
    ;;
  rocky)
    install_rockylinux_requirements
    ;;
  *)
    echo "Unable to determine OS..."
    exit 1
    ;;
esac
