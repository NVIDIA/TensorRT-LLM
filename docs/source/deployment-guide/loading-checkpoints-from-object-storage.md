<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Loading Checkpoints from Object Storage

TensorRT LLM loads a model from a local Hugging Face checkpoint directory. When
you serve on a fleet of GPU nodes, it is common to keep that directory in an
object store and copy it to each node before starting the server. This guide
shows that workflow for Amazon S3 and for any other object store that exposes
the S3 API.

The workflow has three steps:

1. Upload the Hugging Face checkpoint directory to a bucket once.
2. On each serving node, sync it to a local cache directory.
3. Point `LLM(model=...)` or `trtllm-serve` at the local cache directory.

## Credentials and endpoint

The examples use the standard AWS environment variables, which both the `aws`
CLI and `boto3` read:

```bash
export AWS_ACCESS_KEY_ID="<access-key-id>"
export AWS_SECRET_ACCESS_KEY="<secret-access-key>"
export AWS_DEFAULT_REGION="<region>"
```

For Amazon S3 no endpoint is needed. For another S3-compatible store, also set
its endpoint and pass it to the commands below:

```bash
export S3_ENDPOINT="https://<s3-compatible-endpoint>"
```

Use a key that can only read the bucket (or prefix) that holds the checkpoints
on serving nodes, and keep write access for the upload step.

## Upload a checkpoint

Download the checkpoint from Hugging Face, then copy the directory to the
bucket. Include the model name and revision in the prefix so different versions
do not overwrite each other.

```bash
huggingface-cli download Qwen/Qwen3-8B --revision <revision> --local-dir ./Qwen3-8B-<revision>

# Amazon S3
aws s3 sync ./Qwen3-8B-<revision> s3://<bucket>/checkpoints/Qwen3-8B/<revision>/

# Other S3-compatible stores
aws s3 sync ./Qwen3-8B-<revision> s3://<bucket>/checkpoints/Qwen3-8B/<revision>/ \
  --endpoint-url "$S3_ENDPOINT"
```

## Sync to a local cache on each node

`aws s3 sync` only copies files that are new or changed, so rerunning it on a
node that already has the checkpoint is fast.

```bash
CACHE_DIR=/models/Qwen3-8B/<revision>

# Amazon S3
aws s3 sync s3://<bucket>/checkpoints/Qwen3-8B/<revision>/ "$CACHE_DIR"

# Other S3-compatible stores
aws s3 sync s3://<bucket>/checkpoints/Qwen3-8B/<revision>/ "$CACHE_DIR" \
  --endpoint-url "$S3_ENDPOINT"
```

### Using boto3 instead of the CLI

The same download in Python. It skips files that are already present with the
same size and rejects object keys that would write outside the cache directory.
Because it compares sizes only, it does not notice an object that was replaced
in place with different content of the same size. Publish a changed checkpoint
under a new revision prefix instead of overwriting the objects of an existing one.

```python
import os
from pathlib import Path

import boto3

BUCKET = "<bucket>"
PREFIX = "checkpoints/Qwen3-8B/<revision>/"
CACHE_DIR = Path("/models/Qwen3-8B/<revision>").resolve()

# endpoint_url=None targets Amazon S3; set S3_ENDPOINT for other stores.
s3 = boto3.client("s3", endpoint_url=os.environ.get("S3_ENDPOINT"))

paginator = s3.get_paginator("list_objects_v2")
for page in paginator.paginate(Bucket=BUCKET, Prefix=PREFIX):
    for obj in page.get("Contents", []):
        relative = obj["Key"][len(PREFIX):]
        if not relative or relative.endswith("/"):
            continue
        target = (CACHE_DIR / relative).resolve()
        if not target.is_relative_to(CACHE_DIR):
            raise ValueError(f"Refusing to write outside the cache: {obj['Key']}")
        if target.exists() and target.stat().st_size == obj["Size"]:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        s3.download_file(BUCKET, obj["Key"], str(target))
```

## Serve from the local cache

Point TensorRT LLM at the cache directory as you would at any local Hugging Face
checkpoint:

```python
from tensorrt_llm import LLM

llm = LLM(model="/models/Qwen3-8B/<revision>")
```

or:

```bash
trtllm-serve /models/Qwen3-8B/<revision>
```

## Recommended practices

- Keep the cache on fast local storage, and reuse it across restarts instead of
  downloading the checkpoint again.
- Use one prefix per model revision, so a node never mixes files from two
  versions, and treat a published prefix as read-only: upload a changed
  checkpoint under a new revision instead of overwriting it.
- For large checkpoints, both `aws s3 sync` and `boto3` transfer files in
  parallel multipart requests by default; tune concurrency with the CLI's
  `s3.max_concurrent_requests` setting or a `boto3`
  `TransferConfig` if the download is network bound.
- To read weights directly from object storage without a local copy, see the
  `BaseWeightLoader` interface in [Checkpoint Loading](../features/checkpoint-loading.md).
