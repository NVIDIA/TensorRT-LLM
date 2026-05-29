#!/bin/bash
set -ex

# Installs the build-time deps NIXL's LIBFABRIC plugin needs:
#   - libfabric (with headers)
#   - hwloc (with headers)
# These are independent: AWS EFA's /opt/amazon/efa ships libfabric but not
# hwloc, so we resolve each separately.

EFA_PREFIX="/opt/amazon/efa"

have_libfabric_dev() {
  if command -v pkg-config >/dev/null && pkg-config --exists libfabric; then return 0; fi
  if [ -f /usr/include/rdma/fabric.h ] || ls /usr/include/*/rdma/fabric.h >/dev/null 2>&1; then return 0; fi
  return 1
}

have_hwloc_dev() {
  if command -v pkg-config >/dev/null && pkg-config --exists hwloc; then return 0; fi
  if [ -f /usr/include/hwloc.h ] || ls /usr/include/*/hwloc.h >/dev/null 2>&1; then return 0; fi
  return 1
}

detect_pkg_mgr() {
  if command -v apt-get >/dev/null; then echo apt; return; fi
  for m in dnf microdnf yum; do
    if command -v "$m" >/dev/null; then echo "$m"; return; fi
  done
  echo ""
}

PKG_MGR="$(detect_pkg_mgr)"

install_pkgs() {
  # $@ are distro-agnostic logical names; we map to package names per pkg mgr.
  case "$PKG_MGR" in
    apt)
      apt-get update
      DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends "$@"
      ;;
    dnf|microdnf|yum)
      # CRB / PowerTools / RHEL codeready-builder host the *-devel packages
      # on RHEL-family bases; enable best-effort.
      "$PKG_MGR" install -y "${PKG_MGR}-plugins-core" || true
      for repo in crb powertools \
                  "codeready-builder-for-rhel-9-$(uname -m)-rpms" \
                  "codeready-builder-for-rhel-8-$(uname -m)-rpms"; do
        "$PKG_MGR" config-manager --set-enabled "$repo" || true
      done
      "$PKG_MGR" install -y "$@"
      ;;
    *)
      echo "[install_libfabric_efa] No supported package manager; cannot install: $*" >&2
      return 1
      ;;
  esac
}

# --- libfabric --------------------------------------------------------------
if [ -d "${EFA_PREFIX}" ] && [ -f "${EFA_PREFIX}/include/rdma/fabric.h" ]; then
  echo "[install_libfabric_efa] Using AWS EFA libfabric at ${EFA_PREFIX}."
  echo "export LD_LIBRARY_PATH=${EFA_PREFIX}/lib:\$LD_LIBRARY_PATH" >> "${ENV}"
elif have_libfabric_dev; then
  echo "[install_libfabric_efa] libfabric dev already present; skipping libfabric install."
else
  case "$PKG_MGR" in
    apt) install_pkgs libfabric-dev libfabric1 ;;
    dnf|microdnf|yum) install_pkgs libfabric libfabric-devel ;;
    *) echo "[install_libfabric_efa] no pkg mgr and no libfabric source." >&2; exit 1 ;;
  esac
fi

# --- hwloc ------------------------------------------------------------------
if have_hwloc_dev; then
  echo "[install_libfabric_efa] hwloc dev already present; skipping hwloc install."
else
  case "$PKG_MGR" in
    apt) install_pkgs libhwloc-dev ;;
    dnf|microdnf|yum) install_pkgs hwloc-devel ;;
    *) echo "[install_libfabric_efa] no pkg mgr to install hwloc." >&2; exit 1 ;;
  esac
fi
