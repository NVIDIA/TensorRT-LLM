#!/bin/bash
# Make the Mooncake Python store bindings importable inside a TensorRT-LLM
# container, so the mooncake-store KV connector can start.
#
# Two things break a plain `pip install mooncake-transfer-engine` in the
# containers built by docker/common/install_mooncake.sh:
#
#   1. The CMake source build in that script emits its own, unusable `mooncake`
#      Python package (it omits libmooncake_store.so). mooncake-integration's
#      CMakeLists picks the install directory with
#        python3 -c "import sys; print([s for s in sys.path if 'packages' in s][0])"
#      -- the first sys.path entry whose name merely contains "packages". With
#      nvidia-cutlass-dsl installed that is nvidia_cutlass_dsl/dsl_packages,
#      which nvidia_cutlass_dsl_packages.pth sys.path.insert(0)s, so it shadows
#      anything pip installs. Without it, the package lands in dist-packages and
#      collides with the wheel: CMake writes store.cpython-312-<plat>.so, the
#      wheel writes store.so, and importlib prefers the interpreter-tagged
#      suffix, so the broken extension still wins. Either way the symptom is
#      `ImportError: libmooncake_store.so` *after* pip reports success.
#      Because pip overwrites __init__.py in the collision case, leftovers are
#      not reliably identifiable after the fact -- so this script removes the
#      package directory outright and reinstalls, rather than trying to tell
#      good files from bad.
#
#   2. The `mooncake-transfer-engine` wheel is linked against libcudart.so.12,
#      while containers from pytorch-26.05 on ship CUDA 13 only.
#      `mooncake-transfer-engine-cuda13` is the same project built for CUDA 13
#      and needs no shim, so it is the default here. Its releases start at 0.3.9,
#      so it cannot match the 0.3.7.post2 pin in install_mooncake.sh -- see the
#      note below on why that is safe.
#
# Version drift against /usr/local/Mooncake: that CMake-built C++ library backs
# the *cache transceiver's* Mooncake backend, a different feature. The connector
# only ever talks to the wheel, and the wheel also supplies the mooncake_master
# that lands on PATH, so client and master stay matched. Revisit only if you set
# cache_transceiver_config.backend to MOONCAKE (these configs use NIXL).
#
# Set MOONCAKE_WHEEL to override, e.g.
#   MOONCAKE_WHEEL="mooncake-transfer-engine==0.3.7.post2"
# to match install_mooncake.sh exactly; the libcudart.so.12 shim is then applied
# automatically.
#
# Idempotent, and cheap on re-runs: if the install is already correct it exits
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

# Purge every `mooncake` package directory on the search path, whatever wrote it.
# Distinguishing CMake leftovers from wheel files is unreliable once pip has
# overwritten __init__.py, so remove and reinstall instead.
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
