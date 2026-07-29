<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# openengine proto vendoring

The gRPC contract in this package is vendored from **ai-dynamo/openengine**.

- Source: https://github.com/ai-dynamo/openengine
- Pinned commit: `b8d052fec7451bc978fdcee4991ef38c2c586b20`
- Vendored path: `proto/openengine/v1/`
- License: Apache-2.0 (`proto/LICENSE`); the `.proto` files carry NVIDIA copyright headers.

## Contents

- `.proto` sources: `error`, `generation`, `kv`, `lifecycle`, `lora`, `model`, `openengine`, `server`, plus the upstream `README.md`.
- Generated stubs (committed, flat in this package): `*_pb2.py`, `*_pb2_grpc.py`, `*_pb2.pyi`.

## Generation

- Tool: `grpcio-tools==1.64.1`. The gencode targets Protobuf Python 5.26.1.
- Runtime: the stubs work with `protobuf>=5.26`. TensorRT-LLM does not pin `protobuf` directly; it is currently pulled transitively via `smg-grpc-proto` (`protobuf>=5.26.0`). The openengine path should declare `protobuf` explicitly (see RFC https://github.com/NVIDIA/TensorRT-LLM/issues/17016).
- Post-processing: absolute cross-imports `from openengine.v1 import X` are rewritten to package-relative `from . import X`.
- Regenerate with `./generate_stubs.sh`.

## Updating the pin

openengine is experimental. We hold this specific commit until openengine
stabilizes (see RFC https://github.com/NVIDIA/TensorRT-LLM/issues/17016).

To bump:
1. Re-download `proto/openengine/v1/*.proto` (and `proto/LICENSE`) at the new commit.
2. Update the pinned commit above.
3. Run `./generate_stubs.sh`.
4. If protos were added or removed, update the three module lists in `__init__.py` (the `from . import` block, the `None`-fallback block, and `__all__`).
5. Review the stub diff.
