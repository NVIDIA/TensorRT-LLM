#!/usr/bin/env python3
"""Prove that a Mooncake install can actually serve the mooncake-store connector.

Mirrors the ``store.setup(...)`` call in
``tensorrt_llm/_torch/pyexecutor/connectors/mooncake_store/worker.py`` and then
does one round trip, so a pass here means the connector's own startup path will
work. Reads the same ``MOONCAKE_CONFIG_PATH`` file the connector reads.
"""

import json
import os
import socket
import sys

CONFIG_PATH = os.environ.get("MOONCAKE_CONFIG_PATH")
if not CONFIG_PATH:
    sys.exit("Set MOONCAKE_CONFIG_PATH to the Mooncake JSON config first.")

with open(CONFIG_PATH) as handle:
    cfg = json.load(handle)


def parse_size(value):
    if isinstance(value, int):
        return value
    units = {"KiB": 1 << 10, "MiB": 1 << 20, "GiB": 1 << 30, "TiB": 1 << 40}
    for suffix, scale in units.items():
        if value.endswith(suffix):
            return int(float(value[: -len(suffix)]) * scale)
    return int(value)


from mooncake.store import MooncakeDistributedStore  # noqa: E402

print("import mooncake.store: OK")

store = MooncakeDistributedStore()
hostname = cfg.get("local_hostname") or socket.gethostbyname(socket.gethostname())
status = store.setup(
    hostname,
    cfg["metadata_server"],
    parse_size(cfg.get("global_segment_size", "1GiB")),
    parse_size(cfg.get("local_buffer_size", "256MiB")),
    cfg.get("protocol", "tcp"),
    cfg.get("device_name", ""),
    cfg["master_server_address"],
)
if status != 0:
    sys.exit(f"store.setup failed with status {status}")
print(f"store.setup: OK (host={hostname}, master={cfg['master_server_address']})")

key = f"{cfg.get('cache_prefix', 'trtllm')}/smoke-test"
payload = bytes(range(256)) * 4096  # 1 MiB, non-trivial content

assert store.put(key, payload) == 0, "put failed"
print(f"put {len(payload)} bytes: OK")

assert store.is_exist(key) == 1, "is_exist did not report the key"
print("is_exist: OK")

got = store.get(key)
assert got == payload, f"round trip mismatch: got {len(got)} bytes"
print("get + byte-for-byte compare: OK")

store.remove(key)
print("remove: OK")
print("\nPASS: this install can back the mooncake-store connector.")
