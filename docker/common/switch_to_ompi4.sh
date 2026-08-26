#!/bin/bash
#
# The DLFW PyTorch base image (HPC-X 2.50+) defaults /usr/local/mpi and
# /opt/hpcx/ompi to Open MPI 5 / PRTE, which has many breaking changes.
# This script retargets the symlinks to Open MPI 4.
#
# No-op when HPC-X is absent (e.g. Rocky builds that use distro Open MPI).

set -ex

OMPI4_ROOT="/opt/hpcx/ompi4"
OMPI_LINK="/opt/hpcx/ompi"
USR_LOCAL_MPI="/usr/local/mpi"

if [ ! -d /opt/hpcx ]; then
  echo "HPC-X not present; leaving the system MPI stack unchanged"
  exit 0
fi

if [ ! -d "${OMPI4_ROOT}" ]; then
  echo "ERROR: HPC-X is present but ${OMPI4_ROOT} is missing; cannot default Open MPI to 4.x" >&2
  exit 1
fi

if [ ! -x "${OMPI4_ROOT}/bin/orterun" ] && [ ! -x "${OMPI4_ROOT}/bin/mpirun" ]; then
  echo "ERROR: ${OMPI4_ROOT} has no mpirun/orterun" >&2
  exit 1
fi

ln -sfn ompi4 "${OMPI_LINK}"
ln -sfn "${OMPI4_ROOT}" "${USR_LOCAL_MPI}"

# Prefer OMPI 4 for soname lookups; keep OMPI 5 installable but off the
# default search path so a stray libmpi.so.40 cannot pull PRTE back in.
if [ -f /etc/ld.so.conf.d/hpcx.conf ]; then
  sed -i '\|/opt/hpcx/ompi5/lib|d' /etc/ld.so.conf.d/hpcx.conf
fi
ldconfig

echo "Default Open MPI is now:"
mpirun --version | head -1
ompi_info --version | head -1
