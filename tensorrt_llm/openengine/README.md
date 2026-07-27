<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# TensorRT-LLM OpenEngine sibling server

`trtllm-serve` can expose OpenEngine alongside its normal HTTP server with
`--openengine-port`. The feature remains disabled when the flag is absent and
keeps the OpenEngine bindings out of TensorRT-LLM's required dependencies.

The exact schema revision-1 source revision is recorded in
`OPENENGINE_COMMIT`. Generate Python bindings directly from that schema and
install the OpenEngine runtime dependencies with:

```bash
python scripts/install_openengine.py
python -m pip install -e .
```

The OpenEngine runtime dependencies include gRPC code generation and the
headless OpenCV decoder needed by video-capable input processors. The
`openengine` extra installs those dependencies, while the sibling installer
verifies the pinned source and generates bindings into the local build tree.

The installer rejects a different or dirty sibling proto checkout. Once the
schema is published through Buf, export the immutable module into a checkout
and pass it with `--sibling` and the pinned `--source-identity`. Update
`OPENENGINE_COMMIT` to the immutable BSR module commit at publication. The
installer prints the required `OPENENGINE_SCHEMA_RELEASE` export; sibling
startup fails closed unless that value exactly matches `OPENENGINE_COMMIT`.
