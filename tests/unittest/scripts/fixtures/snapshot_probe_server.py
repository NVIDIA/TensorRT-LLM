# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""HTTP boundary double for the Snapshot probe, not an inference/restore emulator."""

import json
import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path


def main() -> None:
    """Serve controlled HTTP outcomes until the probe terminates the process."""
    address_file, behavior = sys.argv[1:]
    if behavior == "exit":
        return
    if behavior == "no-address":
        time.sleep(60)
        return

    class Handler(BaseHTTPRequestHandler):
        """Respond to the two native HTTP paths used by the probe."""

        def do_GET(self) -> None:
            """Return HTTP health without certifying generation or restore."""
            self.send_response(503 if behavior == "unhealthy" else 200)
            self.end_headers()

        def do_POST(self) -> None:
            """Return a completion, malformed output or a blocked generation."""
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if behavior == "hang":
                time.sleep(60)
            self.send_response(500 if behavior == "http-error" else 200)
            self.end_headers()
            text = "Different" if behavior == "mismatch" else "Berlin"
            if behavior == "empty":
                text = ""
            body = json.dumps(
                {
                    "choices": [{"text": text, "finish_reason": "length"}],
                    "usage": {"completion_tokens": request["max_tokens"]},
                }
            ).encode()
            if behavior == "dribble":
                for byte in body:
                    self.wfile.write(bytes([byte]))
                    self.wfile.flush()
                    time.sleep(0.1)
                return
            self.wfile.write(b"not-json" if behavior == "malformed" else body)

    with HTTPServer(("127.0.0.1", 0), Handler) as server:
        address_file = Path(address_file)
        temporary = address_file.with_suffix(".tmp")
        temporary.write_text(f"127.0.0.1:{server.server_port}\n", encoding="utf-8")
        temporary.replace(address_file)
        server.serve_forever()


if __name__ == "__main__":
    main()
