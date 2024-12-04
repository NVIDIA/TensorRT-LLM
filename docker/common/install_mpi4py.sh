#!/bin/bash

set -ex

GITHUB_URL="https://github.com"
if [ -n "${GITHUB_MIRROR}" ]; then
    GITHUB_URL=${GITHUB_MIRROR}
fi

MPI4PY_VERSION="3.1.5"
RELEASE_URL="${GITHUB_URL}/mpi4py/mpi4py/archive/refs/tags/${MPI4PY_VERSION}.tar.gz"
curl -L ${RELEASE_URL} | tar -zx -C /tmp
# Bypassing compatibility issues with higher versions (>= 69) of setuptools.
sed -i 's/>= 40\.9\.0/>= 40.9.0, < 69/g' /tmp/mpi4py-${MPI4PY_VERSION}/pyproject.toml
pip3 install /tmp/mpi4py-${MPI4PY_VERSION}
rm -rf /tmp/mpi4py*
