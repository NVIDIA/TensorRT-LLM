"""UCX env defaults for disagg unit tests (loaded by pytest before test imports)."""

import os

# Allow-list of UCX transports a NIXL agent needs in a single-host unit test.
# Exclusions: ib (UCX setup hang when fabric is down), gdr_copy (UCX rcache
# SIGABRT at process teardown), efa/rocm/ze/gaudi (other-platform transports
# unit tests do not need). Avoid the "cuda" alias -- it pulls in gdr_copy.
os.environ.setdefault("UCX_TLS", "tcp,self,sm,cuda_copy,cuda_ipc,rdmacm,gga")

# Network-interface name prefixes whose ethtool/iface_query hangs UCX worker
# destruction. Skipped via UCX_NET_DEVICES negate-list (^ prefix).
_PROBLEM_NIC_PREFIXES = ("vxlan", "bond", "tailscale", "wg", "tun", "tap")


def _problem_net_devices() -> list[str]:
    sysnet = "/sys/class/net"
    if not os.path.isdir(sysnet):
        return []
    skip = []
    for name in sorted(os.listdir(sysnet)):
        low = name.lower()
        if any(low == p or low.startswith(p) for p in _PROBLEM_NIC_PREFIXES):
            skip.append(name)
    return skip


# Negate-list rather than whitelist: when the host IP sits on a bond, the
# underlying slave NICs have no IPv4 and UCX filters them out -- a whitelist
# that excludes the bond leaves zero TCP devices for NIXL's AM transport.
_skip = _problem_net_devices()
if _skip:
    os.environ.setdefault("UCX_NET_DEVICES", "^" + ",".join(_skip))
