#!/bin/bash

set -ex

FLASH_ATTN_4_VERSION="4.0.0b11"

if [ -n "${GITHUB_MIRROR}" ]; then
  export PIP_INDEX_URL="https://urm.nvidia.com/artifactory/api/pypi/pypi-remote/simple"
fi
pip3 install "flash-attn-4==${FLASH_ATTN_4_VERSION}"
