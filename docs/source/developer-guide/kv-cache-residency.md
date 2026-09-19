<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Residency groups in KVCacheManagerV2

An active request can keep old KV history without locking all of it to GPU.
These internal C++ controls let a caller keep sinks, recent history, and writable
pages on GPU while older pages remain held. A held page cannot be dropped, but
the existing storage manager may move it to another cache tier.

## Separate data with different GPU needs

`AttentionLayerConfig::residencyGroup` is part of lifecycle identity. Layers with
different values get separate pages even when their attention windows and sinks
match. The default is `0`; values must be non-negative.

For example, use separate layer descriptors for scoring data and sparse KV,
with unique layer IDs and different residency groups. All buffers within one
descriptor still share one page lifecycle. The group value does not select a
storage tier or change the attention window.

The internal Python property is `layer._residency_group`. Set it before creating
the manager. Use `manager.get_layer_group_id(layer_id)` to get the assigned
lifecycle ID for request operations; this ID is not the residency-group value.

## Choose which pages stay locked

Call `KvCache::setResidencyWindow(group, windowSize, numSinkTokens)` for a
full-history attention group. `windowSize` must be positive and includes the
next input token. Sinks are the first `numSinkTokens` tokens of the sequence.
Protection rounds to whole pages and also covers all writable pages.

For example, with four tokens per page, history length 16, capacity 20, a window
of five tokens, and four sink tokens:

| Tokens | Request protection |
| --- | --- |
| 0–3 | Locked: sink page |
| 4–11 | Held: retained history |
| 12–15 | Locked: recent history |
| 16–19 | Locked: writable page |

The Python binding is:

```python
group = manager.get_layer_group_id(sparse_layer_id)
cache._set_residency_window(group, window_size=5, num_sink_tokens=4)
# Restore full GPU residency when required:
cache._set_residency_window(group, window_size=None)
```

This policy can be configured while a request is suspended. Resume locks only
the required pages. Updating an active request acquires newly required pages
before releasing old locks; allocation failure preserves the previous policy.
Existing sliding-window attention and SSM groups keep their own lifecycle rules
and reject this opt-in.

## Preserve history and unfinished work

The residency window changes GPU locks, not which history remains valid. Both
committed and uncommitted history stay held. Suspend/resume and prefix sharing
preserve that history; another request's lock can still keep a shared page on GPU.

Submit writes and prior GPU uses on the request stream before advancing history
or changing residency. Lock release records completion events. Commit also
preserves any unfinished migration event on a held page. Shrinking or closing a
request releases its pages through the existing lifetime rules.

Held pages have invalid ordinary GPU page-table entries, even while their bytes
remain on GPU. Attention must acquire the data it will read before using it.
This API controls page protection; retained host copies, sparse selection, and
selected-entry refetch are separate work. The pure-Python reference backend does
not implement these residency controls.
