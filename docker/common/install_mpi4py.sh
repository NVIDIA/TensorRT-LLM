#!/bin/bash

set -ex

MPI4PY_VERSION="3.1.5"
RELEASE_URL="https://github.com/mpi4py/mpi4py/archive/refs/tags/${MPI4PY_VERSION}.tar.gz"
curl -L ${RELEASE_URL} | tar -zx -C /tmp
# Bypassing compatibility issues with higher versions (>= 69) of setuptools.
sed -i 's/>= 40\.9\.0/>= 40.9.0, < 69/g' /tmp/mpi4py-${MPI4PY_VERSION}/pyproject.toml
pip3 install /tmp/mpi4py-${MPI4PY_VERSION}
rm -rf /tmp/mpi4py*
