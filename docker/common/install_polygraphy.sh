#!/bin/bash

set -ex

EXTRA_INDEX_URL=""
if [ -n "${GITHUB_MIRROR}" ]; then
  PIP_INDEX_URL="https://urm.nvidia.com/artifactory/api/pypi/pypi-remote/simple"
fi
pip3 install --no-cache-dir polygraphy==0.49.9
PIP_INDEX_URL=""

# Clean up pip cache and temporary files
pip3 cache purge
rm -rf ~/.cache/pip
rm -rf /tmp/*
