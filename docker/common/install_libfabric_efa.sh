#!/bin/bash
set -ex

# Installs libfabric so NIXL's LIBFABRIC plugin can be compiled. On AWS EFA
# bases the host image usually ships /opt/amazon/efa with an EFA-aware
# libfabric (includes the optimized "efa" provider); prefer that. On non-EFA
# bases fall back to the distro libfabric package, which is enough for the
# plugin to compile and for runtime use over verbs/TCP providers.

EFA_PREFIX="/opt/amazon/efa"

if [ -d "${EFA_PREFIX}" ] && [ -f "${EFA_PREFIX}/include/rdma/fabric.h" ]; then
  echo "[install_libfabric_efa] Reusing AWS EFA libfabric at ${EFA_PREFIX}."
  echo "export LD_LIBRARY_PATH=${EFA_PREFIX}/lib:\$LD_LIBRARY_PATH" >> "${ENV}"
  exit 0
fi

if ! command -v apt-get >/dev/null; then
  echo "[install_libfabric_efa] apt-get unavailable and no EFA libfabric present;" >&2
  echo "[install_libfabric_efa] cannot install libfabric." >&2
  exit 1
fi

apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
  libfabric-dev libfabric1
