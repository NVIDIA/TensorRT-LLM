#!/bin/bash
# Make the Mooncake Python store bindings importable inside a TensorRT-LLM
# container, so the mooncake-store KV connector can start. Images built by
# docker/common/install_mooncake.sh already have this; run it on images that
# predate that, or to change the wheel.
#
# Two things break a plain `pip install mooncake-transfer-engine` in those
# containers, both explained in docker/common/install_mooncake.sh: the CMake
# source build leaves behind an unusable `mooncake` package that shadows or
# collides with the wheel, and the default wheel is linked against
# libcudart.so.12 while these images ship CUDA 13 only. Either way the symptom
# is `ImportError: libmooncake_store.so` after pip reports success.
#
# `mooncake-transfer-engine-cuda13` needs no CUDA 12 shim, so it is the default
# here. Its releases start at 0.3.9 and so cannot match the pin in
# install_mooncake.sh, which is safe: that CMake-built C++ library backs the
# cache transceiver's Mooncake backend, a different feature, while the
# connector only ever talks to the wheel. The wheel also supplies the
# mooncake_master that lands on PATH, so client and master stay matched.
# Revisit only if cache_transceiver_config.backend is set to MOONCAKE.
#
# Set MOONCAKE_WHEEL to override, for example
#   MOONCAKE_WHEEL="mooncake-transfer-engine==0.3.7.post2"
# to match install_mooncake.sh exactly; the libcudart.so.12 shim is then
# applied automatically.
#
# Idempotent and cheap on re-runs: if the install is already correct it exits
# without contacting the network, so it is safe in a SLURM prolog on every node.

set -euo pipefail

MOONCAKE_WHEEL="${MOONCAKE_WHEEL:-mooncake-transfer-engine-cuda13==0.3.13}"
WHEEL_NAME="${MOONCAKE_WHEEL%%[=<>]*}"
SITE_PACKAGES="$(python3 -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"

echo ">> target wheel: ${MOONCAKE_WHEEL}"

# Fast path: already correct, so do not touch the network.
if pip3 show "${WHEEL_NAME}" >/dev/null 2>&1 &&
   python3 -c 'from mooncake.store import MooncakeDistributedStore; MooncakeDistributedStore()' >/dev/null 2>&1; then
    echo ">> already installed and importable; nothing to do"
    python3 -c 'import mooncake.store; print("   resolved extension:", mooncake.store.__file__)'
    exit 0
fi

# Purge every `mooncake` package directory on the search path, whatever wrote
# it, since leftovers cannot be told apart from wheel files reliably.
python3 - <<'PY'
import os
import shutil
import sys
import sysconfig

paths = sysconfig.get_paths()
for entry in list(sys.path) + [paths["purelib"], paths["platlib"]]:
    if not entry:
        continue
    package = os.path.join(entry, "mooncake")
    if os.path.isdir(package):
        print(f">> removing existing mooncake package: {package}")
        shutil.rmtree(package, ignore_errors=True)
PY

# The two distributions install the same `mooncake` package, so leaving both
# registered produces a half-overwritten directory.
for distribution in mooncake-transfer-engine mooncake-transfer-engine-cuda13; do
    if pip3 show "${distribution}" >/dev/null 2>&1; then
        echo ">> unregistering ${distribution}"
        pip3 uninstall -y -q "${distribution}" >/dev/null 2>&1 || true
    fi
done

pip3 install --no-cache-dir "${MOONCAKE_WHEEL}"

# Only the CUDA 12 wheel needs the runtime shim. Drop it into the wheel's own
# RPATH directory so the Python extensions and the mooncake_* binaries all find
# it without LD_LIBRARY_PATH being set in each process.
if ldd "${SITE_PACKAGES}"/mooncake/store*.so 2>/dev/null | grep -q "libcudart.so.12 => not found"; then
    echo ">> wheel needs libcudart.so.12, which this container lacks; installing it"
    pip3 install --no-cache-dir nvidia-cuda-runtime-cu12
    CUDART12="${SITE_PACKAGES}/nvidia/cuda_runtime/lib/libcudart.so.12"
    [[ -f "${CUDART12}" ]] || { echo "libcudart.so.12 not found after install" >&2; exit 1; }
    mkdir -p "${SITE_PACKAGES}/mooncake_transfer_engine.libs"
    ln -sf "${CUDART12}" "${SITE_PACKAGES}/mooncake_transfer_engine.libs/libcudart.so.12"
    echo ">> linked libcudart.so.12 into the wheel's RPATH directory"
fi

# Verify, because every failure above stays silent until a worker starts.
echo ">> verifying"
python3 - <<'PY'
import mooncake.store
from mooncake.store import MooncakeDistributedStore

MooncakeDistributedStore()
print("   mooncake.store imports and instantiates: OK")
print(f"   resolved extension: {mooncake.store.__file__}")
PY

# Every extension module and the master binary must have a resolvable link line.
# ldd the real ELF files, not /usr/local/bin/mooncake_master, which is a Python
# console script.
unresolved=0
for elf in "${SITE_PACKAGES}"/mooncake/*.so "${SITE_PACKAGES}"/mooncake/mooncake_master; do
    [[ -e "${elf}" ]] || continue
    if missing="$(ldd "${elf}" 2>&1 | grep 'not found')"; then
        echo "   $(basename "${elf}"): unresolved -> ${missing}" >&2
        unresolved=1
    fi
done
[[ "${unresolved}" -eq 0 ]] || { echo "unresolved shared libraries; see above" >&2; exit 1; }
echo "   all mooncake ELF link lines resolve: OK"

for entry in mooncake_master mooncake_http_metadata_server; do
    path="$(command -v "${entry}")" || { echo "${entry} is not on PATH" >&2; exit 1; }
    echo "   ${entry}: OK (${path})"
done

echo ">> done"
