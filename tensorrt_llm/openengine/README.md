<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# TensorRT-LLM OpenEngine sibling server

`trtllm-serve` can expose OpenEngine alongside its normal HTTP server with
`--openengine-port`. The feature remains disabled when the flag is absent and
keeps the OpenEngine bindings out of TensorRT-LLM's required dependencies.

The exact sibling source revision is recorded in `OPENENGINE_COMMIT`. Verify
and install its generated Python package with:

```bash
python scripts/install_openengine.py
python -m pip install -e .
```

The installer rejects a different or dirty sibling package/proto checkout. It
prints the required `OPENENGINE_SCHEMA_RELEASE` export; sibling startup fails
closed unless that value exactly matches `OPENENGINE_COMMIT`. No registry
publication is assumed.
