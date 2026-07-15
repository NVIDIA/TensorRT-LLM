"""TriAttention KV-cache compression: periodic physical KV eviction driven by
trigonometric importance scoring.

TriAttention is a pure KV-cache compression method. Decode still runs the model's
standard attention over the compacted cache. The manager publishes each request's
cumulative evicted count on ``LlmRequest.py_num_compressed_tokens``; the model
engine subtracts it when building the cached-token metadata. With one-model
speculative decoding, the separate draft KV cache is compacted in the same round
with the target's kept token set (union mode only), so target and draft always
share one physical KV length.

Public surface:
  - ``TriAttention`` -- the ``BaseKVCacheCompressionManager`` (the eviction
    manager; snapshots allocation metadata before forward and compacts the
    finalized prefix in ``on_generation_step_end``). It uses V2 capacity-only
    decode, so there is no KV-cache-manager subclass.
"""

from .triattention import TriAttention

__all__ = ["TriAttention"]
