#!/bin/bash
set -Eeuo pipefail

pip install -e .
pip install -r requirements-dev.txt

hostname
nvidia-smi

echo "Installation completed on $(hostname)"
