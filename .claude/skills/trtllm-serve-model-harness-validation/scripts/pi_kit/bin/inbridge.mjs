// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Container side of the pi kit's endpoint bridge. The container has no network (docker
// --network none): pi talks HTTP to 127.0.0.1:<port> inside it, and every TCP connection made
// there is relayed byte for byte to a unix socket the host side listens on, which connects to the
// recorder (or the endpoint). One connection in, one connection out; either side closing or
// failing closes both, so a client cut reaches the recorder as a cut.
//
// usage: node inbridge.mjs <host> <port> <unix socket> <ready file>
import fs from "node:fs";
import net from "node:net";

const [host, port, socketPath, readyFile] = process.argv.slice(2);
if (!host || !port || !socketPath || !readyFile) {
  console.error("usage: inbridge.mjs <host> <port> <unix socket> <ready file>");
  process.exit(2);
}

const server = net.createServer({ allowHalfOpen: true }, (client) => {
  const upstream = net.createConnection({ path: socketPath, allowHalfOpen: true });
  let closed = false;
  const closeBoth = () => {
    if (closed) return;
    closed = true;
    client.destroy();
    upstream.destroy();
  };
  client.on("error", closeBoth);
  upstream.on("error", closeBoth);
  client.on("close", closeBoth);
  upstream.on("close", closeBoth);
  // A half-close is passed on as one; the close of either socket closes the other.
  client.pipe(upstream);
  upstream.pipe(client);
});
server.on("error", (error) => {
  console.error(`inbridge: ${error.message}`);
  process.exit(1);
});
server.listen(Number(port), host, () => {
  fs.writeFileSync(readyFile, `${process.pid}\n`);
});
for (const signal of ["SIGTERM", "SIGINT"]) {
  process.on(signal, () => process.exit(0));
}
