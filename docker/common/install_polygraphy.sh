#!/bin/bash

set -ex

RELEASE_URL_PG=https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/secure/9.0.1/tars/polygraphy-0.48.1-py2.py3-none-any.whl
pip3 uninstall -y polygraphy
pip3 install ${RELEASE_URL_PG}
