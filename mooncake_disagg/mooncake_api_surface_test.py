#!/usr/bin/env python3
"""Exercise every MooncakeDistributedStore method the connector calls.

`mooncake_smoke_test.py` only proves the install loads and can round-trip a byte
string. The connector's hot path never uses `put` or `get`: it registers the KV
pools and then moves pages with the `batch_*_multi_buffers` zero-copy calls.
Those signatures could drift between wheel versions, so this checks them against
real registered GPU memory, in the same order `worker.py` uses them.

Needs a running mooncake_master and MOONCAKE_CONFIG_PATH, same as the connector.
"""

import json
import os
import socket
import sys

import torch

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


import mooncake  # noqa: E402
from mooncake.store import MooncakeDistributedStore  # noqa: E402

print(f"mooncake package: {mooncake.__path__[0]}")

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
assert status == 0, f"setup failed with status {status}"
print("setup: OK")

# Stand in for a KV pool. PageAddressing.page_buffers returns one address per
# layer-group region, so a page is scattered across REGIONS buffers rather than
# contiguous, which is why the batch calls take list[list[int]]. Two strided
# regions here so the scatter-gather path is actually exercised.
PAGES = 8
REGIONS = 2
REGION_BYTES = 128 * 1024
STRIDE = REGION_BYTES  # slots within a region are strided, as in the real layout
pool = torch.empty(REGIONS * PAGES * STRIDE, dtype=torch.uint8, device="cuda")
region_bases = [pool.data_ptr() + r * PAGES * STRIDE for r in range(REGIONS)]

status = store.register_buffer(pool.data_ptr(), pool.numel())
assert status == 0, f"register_buffer failed with status {status}"
print(f"register_buffer: OK ({pool.numel()} bytes of GPU memory at {pool.data_ptr():#x})")

prefix = cfg.get("cache_prefix", "trtllm")
keys = [f"{prefix}/api-surface/page{i}" for i in range(PAGES)]
addresses = [[base + i * STRIDE for base in region_bases] for i in range(PAGES)]
sizes = [[REGION_BYTES] * REGIONS for _ in range(PAGES)]

# Distinct content per (page, region), so a mixed-up address or size cannot pass.
view = pool.view(REGIONS, PAGES, STRIDE)
for r in range(REGIONS):
    for i in range(PAGES):
        view[r, i, :REGION_BYTES] = (i * 31 + r * 97 + 7) % 256
expected = pool.clone()

present = store.batch_is_exist(keys)
assert len(present) == PAGES, f"batch_is_exist returned {len(present)} of {PAGES}"
assert all(status != 1 for status in present), f"keys already present: {present}"
print(f"batch_is_exist (absent): OK {list(present)}")

results = store.batch_put_from_multi_buffers(keys, addresses, sizes)
assert len(results) == PAGES, f"batch_put returned {len(results)} of {PAGES}"
bad = [(k, r) for k, r in zip(keys, results) if not isinstance(r, int) or r < 0]
assert not bad, f"batch_put_from_multi_buffers failed: {bad}"
print(f"batch_put_from_multi_buffers: OK {list(results)}")

present = store.batch_is_exist(keys)
assert all(status == 1 for status in present), f"keys missing after put: {present}"
print("batch_is_exist (present): OK")

pool.zero_()
results = store.batch_get_into_multi_buffers(keys, addresses, sizes)
assert len(results) == PAGES, f"batch_get returned {len(results)} of {PAGES}"
bad = [(k, r) for k, r in zip(keys, results) if not isinstance(r, int) or r < 0]
assert not bad, f"batch_get_into_multi_buffers failed: {bad}"
print(f"batch_get_into_multi_buffers: OK {list(results)}")

torch.cuda.synchronize()
assert torch.equal(pool, expected), "page contents differ after the round trip"
print("GPU page contents byte-for-byte identical: OK")

for key in keys:
    store.remove(key)
store.close()
print("remove + close: OK")

print("\nPASS: the full connector API surface works on this install.")
