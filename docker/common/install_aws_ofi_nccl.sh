#!/bin/bash
set -ex

# Builds and installs the aws-ofi-nccl NCCL plugin so that NCCL collectives go
# over libfabric/EFA on AWS clusters instead of falling back to TCP. The script
# is a no-op on bases that don't ship AWS EFA, so it stays safe to invoke
# unconditionally from Dockerfile.multi.

AWS_OFI_NCCL_VERSION="${AWS_OFI_NCCL_VERSION:-v1.16.2}"
# Upstream tags are prefixed with "v" (e.g. v1.17.0). Accept either form so
# callers passing "1.17.0" still resolve to a valid tag.
[[ "${AWS_OFI_NCCL_VERSION}" =~ ^[0-9] ]] && AWS_OFI_NCCL_VERSION="v${AWS_OFI_NCCL_VERSION}"
AWS_OFI_NCCL_REPO="https://github.com/aws/aws-ofi-nccl.git"
AWS_OFI_NCCL_PREFIX="/opt/aws-ofi-nccl"
EFA_PREFIX="/opt/amazon/efa"
CUDA_PATH="/usr/local/cuda"

# Skip when EFA isn't installed (non-AWS base images).
if [ ! -d "${EFA_PREFIX}" ]; then
  echo "[install_aws_ofi_nccl] ${EFA_PREFIX} not present; skipping (non-AWS base)."
  exit 0
fi
if ! ls "${EFA_PREFIX}"/lib*/libfabric.so* >/dev/null 2>&1; then
  echo "[install_aws_ofi_nccl] libfabric.so not found under ${EFA_PREFIX}; skipping."
  exit 0
fi
if [ ! -f "${EFA_PREFIX}/include/rdma/fabric.h" ]; then
  echo "[install_aws_ofi_nccl] libfabric headers missing at ${EFA_PREFIX}/include;"
  echo "[install_aws_ofi_nccl] cannot build aws-ofi-nccl. Install full EFA SDK to enable."
  exit 0
fi

# hwloc is required by recent aws-ofi-nccl for topology-aware NIC selection;
# configure aborts without it.
if command -v apt-get >/dev/null; then
  apt-get update
  apt-get install -y --no-install-recommends libhwloc-dev
fi

mkdir -p /third-party-source
WORKDIR="$(mktemp -d)"
git clone --depth 1 -b "${AWS_OFI_NCCL_VERSION}" "${AWS_OFI_NCCL_REPO}" "${WORKDIR}/aws-ofi-nccl"
tar -czf "/third-party-source/aws-ofi-nccl-${AWS_OFI_NCCL_VERSION}.tar.gz" \
  -C "${WORKDIR}" aws-ofi-nccl

cd "${WORKDIR}/aws-ofi-nccl"
./autogen.sh

# NCCL ships from the libnccl-dev dpkg with headers in /usr/include and the
# library under the multiarch dir, which configure's --with-nccl prefix layout
# doesn't recognize. Point at both explicitly via CPPFLAGS/LDFLAGS instead.
NCCL_LIBDIR="/usr/lib/$(gcc -print-multiarch)"
CPPFLAGS="-I/usr/include" \
LDFLAGS="-L${NCCL_LIBDIR}" \
  ./configure --prefix="${AWS_OFI_NCCL_PREFIX}" \
              --with-libfabric="${EFA_PREFIX}" \
              --with-cuda="${CUDA_PATH}" \
              --enable-platform-aws \
              --disable-tests
make -j"$(nproc)"
make install

cd /
rm -rf "${WORKDIR}"

# NCCL dlopens libnccl-net.so from LD_LIBRARY_PATH; make the plugin discoverable.
echo "export LD_LIBRARY_PATH=${AWS_OFI_NCCL_PREFIX}/lib:\$LD_LIBRARY_PATH" >> "${ENV}"
