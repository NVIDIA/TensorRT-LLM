#!/bin/bash

set -ex

GITHUB_URL="https://github.com"
if [ -n "${GITHUB_MIRROR}" ]; then
    GITHUB_URL=${GITHUB_MIRROR}
    export PIP_INDEX_URL="https://urm.nvidia.com/artifactory/api/pypi/pypi-remote/simple"
fi

set_bash_env() {
  if [ ! -f ${BASH_ENV} ];then
    touch ${BASH_ENV}
  fi
  # In the existing base images, as long as `ENV` is set, it will be enabled by `BASH_ENV`.
  if [ ! -f ${ENV} ];then
    touch ${ENV}
    (echo "test -f ${ENV} && source ${ENV}" && cat ${BASH_ENV}) > /tmp/shinit_f
    mv /tmp/shinit_f ${BASH_ENV}
  fi
}

cleanup() {
  # Clean up apt/dnf cache
  if [ -f /etc/debian_version ]; then
    apt-get clean
    rm -rf /var/lib/apt/lists/*
  elif [ -f /etc/redhat-release ]; then
    dnf clean all
    rm -rf /var/cache/dnf
  fi

  # Clean up temporary files
  rm -rf /tmp/* /var/tmp/*

  # Clean up pip cache
  pip3 cache purge || true

  # Clean up documentation
  rm -rf /usr/share/doc/* /usr/share/man/* /usr/share/info/*

  # Clean up locale files
  find /usr/share/locale -maxdepth 1 -mindepth 1 -type d ! -name 'en*' -exec rm -rf {} +
}

init_ubuntu() {
  apt-get update
  # libibverbs-dev is installed but libmlx5.so is missing, reinstall the package
  apt remove -y ibverbs-providers libibverbs1
  apt-get --reinstall install -y libibverbs-dev
  apt-get install -y --no-install-recommends \
    ccache \
    gdb \
    git-lfs \
    clang \
    lld \
    llvm \
    libclang-rt-dev \
    libffi-dev \
    libstdc++-14-dev \
    libnuma1 \
    libnuma-dev \
    python3-dev \
    python3-pip \
    python-is-python3 \
    wget \
    pigz \
    libzmq3-dev
  if ! command -v mpirun &> /dev/null; then
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends openmpi-bin libopenmpi-dev
  fi
  echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> "${ENV}"
  # Remove previous TRT installation
  if [[ $(apt list --installed | grep libnvinfer) ]]; then
    apt-get remove --purge -y libnvinfer*
  fi
  if [[ $(apt list --installed | grep tensorrt) ]]; then
    apt-get remove --purge -y tensorrt*
  fi
  pip3 uninstall -y tensorrt
}

install_python_rockylinux() {
  PYTHON_VERSION=$1
  PYTHON_MAJOR="3"
  PYTHON_URL="https://www.python.org/ftp/python/${PYTHON_VERSION}/Python-${PYTHON_VERSION}.tgz"
  if [ -n "${GITHUB_MIRROR}" ]; then
    PYTHON_URL="https://urm.nvidia.com/artifactory/api/vcs/downloadTag/vcs-remote/python/cpython/v${PYTHON_VERSION}?ext=tar.gz"
  fi
  dnf makecache --refresh
  dnf install \
    epel-release \
    compiler-rt \
    curl \
    make \
    gcc \
    openssl-devel \
    bzip2-devel \
    llvm-toolset \
    lld \
    libffi-devel \
    numactl \
    numactl-devel \
    zlib-devel \
    xz-devel \
    sqlite-devel \
    -y
  echo "Installing Python ${PYTHON_VERSION}..."
  curl -L ${PYTHON_URL} | tar -zx -C /tmp
  if [ -n "${GITHUB_MIRROR}" ]; then
    mv /tmp/cpython-${PYTHON_VERSION} /tmp/Python-${PYTHON_VERSION}
  fi
  cd /tmp/Python-${PYTHON_VERSION}
  bash -c "./configure --enable-shared --prefix=/opt/python/${PYTHON_VERSION} --enable-ipv6 \
    LDFLAGS=-Wl,-rpath=/opt/python/${PYTHON_VERSION}/lib,--disable-new-dtags && make -j$(nproc) && make install"
  ln -s /opt/python/${PYTHON_VERSION}/bin/python3 /usr/local/bin/python
  echo "export PATH=/opt/python/${PYTHON_VERSION}/bin:\$PATH" >> "${ENV}"
  cd .. && rm -rf /tmp/Python-${PYTHON_VERSION}
}

install_pyp_rockylinux() {
  bash -c "pip3 install 'urllib3<2.0' pytest"
}

install_gcctoolset_rockylinux() {
  dnf install -y gcc gcc-c++ file libtool make wget bzip2 bison flex
  # https://catalog.ngc.nvidia.com/orgs/nvidia/containers/cuda
  echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> "${ENV}"
  dnf install \
    patch \
    vim \
    wget \
    git-lfs \
    gcc-toolset-11 \
    libffi-devel \
    -y
  dnf install \
    openmpi \
    openmpi-devel \
    pigz \
    rdma-core-devel \
    zeromq-devel \
    -y
  echo "source scl_source enable gcc-toolset-11" >> "${ENV}"
  echo 'export PATH=/usr/lib64/openmpi/bin:$PATH' >> "${ENV}"
}

# Install base packages depending on the base OS
ID=$(grep -oP '(?<=^ID=).+' /etc/os-release | tr -d '"')
set_bash_env
case "$ID" in
  ubuntu)
    init_ubuntu
    ;;
  rocky)
    install_python_rockylinux $1
    install_pyp_rockylinux
    install_gcctoolset_rockylinux
    ;;
  *)
    echo "Unable to determine OS..."
    exit 1
    ;;
esac

# Final cleanup
cleanup
