#!/bin/bash

set -ex

EXTRA_INDEX_URL=""
if [ -n "${GITHUB_MIRROR}" ]; then
  EXTRA_INDEX_URL="--extra-index-url https://urm.nvidia.com/artifactory/api/pypi/sw-tensorrt-pypi/simple"
fi

pip3 install ${EXTRA_INDEX_URL} polygraphy==0.49.9

# Clean up pip cache and temporary files
pip3 cache purge
rm -rf ~/.cache/pip
rm -rf /tmp/*
