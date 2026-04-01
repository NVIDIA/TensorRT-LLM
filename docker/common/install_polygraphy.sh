#!/bin/bash

set -ex

if [ -n "${GITHUB_MIRROR}" ]; then
  export PIP_INDEX_URL="https://urm.nvidia.com/artifactory/api/pypi/pypi-remote/simple"
fi
pip3 install polygraphy==0.49.26
