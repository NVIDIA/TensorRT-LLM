#!/bin/bash
#
# The DLFW PyTorch base image ships both Open MPI 5 (/opt/hpcx/ompi5) and
# Open MPI 4 (/opt/hpcx/ompi4), and registers both lib dirs in ld.so.conf.
# Both versions expose identical sonames (libmpi.so.40, libpmix.so.2,
# liboshmem.so.40), so the dynamic linker can resolve a load to the wrong
# (ABI-incompatible) version at runtime even when everything was compiled
# against Open MPI 5 — this is what causes mpi4py's
# `MPI_ERR_OTHER: known error not in list` on Comm.Split_type in singleton
# mode. TensorRT-LLM's C++ code and PyTorch in this base image are both
# built against Open MPI 5, so downgrading to Open MPI 4 (as an earlier,
# reverted, revision of this script did) is not an option — instead this
# script removes Open MPI 4 from the linker search path so Open MPI 5 is
# the only candidate.
#
# No-op when HPC-X or the ompi5 install is absent (e.g. Rocky builds that
# use distro Open MPI).

set -ex

OMPI5_ROOT="/opt/hpcx/ompi5"
OMPI_LINK="/opt/hpcx/ompi"
USR_LOCAL_MPI="/usr/local/mpi"

if [ ! -d /opt/hpcx ]; then
  echo "HPC-X not present; leaving the system MPI stack unchanged"
  exit 0
fi

if [ ! -d "${OMPI5_ROOT}" ]; then
  echo "ERROR: HPC-X is present but ${OMPI5_ROOT} is missing; cannot default Open MPI to 5.x" >&2
  exit 1
fi

if [ ! -x "${OMPI5_ROOT}/bin/mpirun" ]; then
  echo "ERROR: ${OMPI5_ROOT} has no mpirun" >&2
  exit 1
fi

ln -sfn ompi5 "${OMPI_LINK}"
ln -sfn "${OMPI5_ROOT}" "${USR_LOCAL_MPI}"

# Drop any ld.so.conf.d entry pointing at ompi4's lib dir so its sonames
# (which collide with ompi5's) are never candidates for resolution.
for conf in /etc/ld.so.conf.d/*.conf; do
  [ -f "${conf}" ] || continue
  sed -i '\|/opt/hpcx/ompi4/lib|d' "${conf}"
done
ldconfig

echo "Default Open MPI is now:"
mpirun --version | head -1
