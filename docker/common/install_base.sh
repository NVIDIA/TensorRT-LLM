#!/bin/bash

set -ex

GITHUB_URL="https://github.com"
if [ -n "${GITHUB_MIRROR}" ]; then
    GITHUB_URL=${GITHUB_MIRROR}
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

init_ubuntu() {
  apt-get update
  apt-get install -y --no-install-recommends \
    ccache \
    gdb \
    git-lfs \
    clang \
    lld \
    llvm \
    libclang-rt-dev \
    libffi-dev \
    python3-dev \
    python3-pip \
    python-is-python3 \
    wget \
    pigz
  if ! command -v mpirun &> /dev/null; then
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends openmpi-bin libopenmpi-dev
  fi
  apt-get clean
  rm -rf /var/lib/apt/lists/*
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
  PYTHON_ENV_FILE="/tmp/python${PYTHON_VERSION}_env"
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
    zlib-devel \
    xz-devel \
    -y
  curl -L ${PYTHON_URL} | tar -zx -C /tmp
  cd /tmp/Python-${PYTHON_VERSION}
  bash -c "./configure --enable-shared --prefix=/opt/python/${PYTHON_VERSION} --enable-ipv6 \
    LDFLAGS=-Wl,-rpath=/opt/python/${PYTHON_VERSION}/lib,--disable-new-dtags && make -j$(nproc) && make install"
  ln -s /opt/python/${PYTHON_VERSION}/bin/python3 /usr/local/bin/python
  echo "export PATH=/opt/python/${PYTHON_VERSION}/bin:\$PATH" >> "${PYTHON_ENV_FILE}"
  echo "source ${PYTHON_ENV_FILE}" >> "${ENV}"
  dnf clean all
  cd .. && rm -rf /tmp/Python-${PYTHON_VERSION}
}

install_pyp_rockylinux() {
  bash -c "pip3 install 'urllib3<2.0' pytest"
}

install_gcctoolset_rockylinux() {
  dnf install -y gcc gcc-c++ file libtool make wget bzip2 bison flex
  dnf clean all
  DEVTOOLSET_ENV_FILE="/tmp/gcctoolset_env"
  # https://catalog.ngc.nvidia.com/orgs/nvidia/containers/cuda
  echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> "${ENV}"
  dnf install \
	  vim \
	  wget \
	  git-lfs \
	  gcc-toolset-13 \
	  libffi-devel \
	  -y
  dnf install \
	  openmpi \
	  openmpi-devel \
	  pigz \
	  -y
  echo "source scl_source enable gcc-toolset-13" >> "${DEVTOOLSET_ENV_FILE}"
  echo "source ${DEVTOOLSET_ENV_FILE}" >> "${ENV}"
  echo 'export PATH=/usr/lib64/openmpi/bin:$PATH' >> "${ENV}"
  dnf clean all
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
