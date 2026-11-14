# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""A remote-level donor in its own process, for the remote-level tests (test_remote_level.py).

Usage::

    python remote_donor_cli.py --dir /shared/dir --gib 128

Stop it with SIGTERM: it withdraws its files, deregisters its memory and exits.
"""

import argparse
import os
import signal
import sys
import threading

from tensorrt_llm._torch.pyexecutor.kv_cache.remote_donor import (
    TRANSPORTS,
    make_donor,
    resolve_ucx_devices,
)
from tensorrt_llm.llmapi.llm_args import REMOTE_CACHE_CHUNK_BYTES


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dir", required=True, help="rendezvous directory (shared file system)")
    size = parser.add_mutually_exclusive_group(required=True)
    size.add_argument("--gib", type=float, help="GiB to donate")
    size.add_argument("--bytes", type=int, help="bytes to donate")
    parser.add_argument("--chunk-mib", type=float, default=REMOTE_CACHE_CHUNK_BYTES / (1 << 20))
    parser.add_argument("--numa-node", type=int, default=None)
    parser.add_argument(
        "--device",
        type=int,
        default=None,
        help="CUDA device for a fabric donation's context (default 0)",
    )
    parser.add_argument("--transport", choices=TRANSPORTS, default="nixl")
    parser.add_argument(
        "--ucx-devices", default=None, help="NIXL only; {GPU} is this rank's GPU index"
    )
    parser.add_argument("--name", default=None)
    parser.add_argument(
        "--seconds",
        type=float,
        default=None,
        help="exit after this long (default: until signalled)",
    )
    parser.add_argument(
        "--parent-pid",
        type=int,
        default=None,
        help="exit once this process is gone (a test runner that died without stopping the donor)",
    )
    args = parser.parse_args(argv)
    nbytes = args.bytes if args.bytes is not None else int(args.gib * (1 << 30))
    # The transport's own options: only NIXL takes UCX devices.
    options = {}
    if args.ucx_devices is not None:
        options["ucx_devices"] = resolve_ucx_devices(args.ucx_devices, args.device)
    donor = make_donor(
        args.dir,
        nbytes,
        transport=args.transport,
        chunk_bytes=int(args.chunk_mib * (1 << 20)),
        name=args.name,
        numa_node=args.numa_node,
        device=args.device,
        **options,
    )
    stop = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.set())
    if args.parent_pid is not None:

        def watch_parent() -> None:
            while not stop.wait(1.0):
                if os.getppid() != args.parent_pid:
                    stop.set()

        threading.Thread(target=watch_parent, daemon=True).start()
    try:
        donor.start()
    except Exception as error:
        donor.close()
        donor.publish_failure(error)
        raise
    print(f"READY {donor.name} {donor.transport.kind} {donor.nbytes}", flush=True)
    try:
        stop.wait(args.seconds)
    finally:
        donor.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
